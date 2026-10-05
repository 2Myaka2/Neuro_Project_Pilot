"""Stage 2 mutation-conditioned heads; no data loading or training pipeline.

Inputs describe one protein and independent mutation queries. Graph endpoints
follow Stage 1: row 0 is sender j, row 1 is receiver i. The upstream
ProteinRecord/MutationQuery validators remain responsible for WT alignment and
the geometric graph contract; this module validates the tensor-level contract.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .protocol import PROTOCOL_VERSION


MUTATION_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
MODEL_SPEC_VERSION = "graph_swarm_v1:model:2"
VARIANTS = ("mutation_self", "graph_static", "graph_swarm", "graph_swarm_t1")
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def architecture_specification():
    """JSON-compatible, fresh copy of the frozen pre-training specification."""
    return {
        "spec_version": MODEL_SPEC_VERSION,
        "variants": list(VARIANTS),
        "mutation_alphabet": MUTATION_ALPHABET,
        "mutation_encoding": "concat(one_hot(WT), one_hot(MUT), MUT-WT)",
        "feature_dim": 384, "hidden_dim": 64, "mutation_dim": 60,
        "main_steps": 4, "secondary_steps": 1,
        "projection": {"dimensions": [384, 64], "activation": "tanh", "bias": True},
        "mutation_mlp": {"dimensions": [60, 64, 64], "activation": "SiLU after first linear only", "bias": [True, True]},
        "message_input_dim": 145,
        "message_input_order": ["receiver", "sender", "edge_attr"],
        "message_mlp": {"dimensions": [145, 64, 64], "activation": "SiLU after first linear only", "bias": [True, True]},
        "attention_mlp": {"dimensions": [145, 32, 1], "activation": "SiLU after first linear only", "bias": [True, False]},
        "attention_normalization": "incoming edges per destination and mutation query; stable softmax",
        "zero_incoming_message": 0,
        "recurrent_cell": {"type": "GRUCell", "input_size": 64, "hidden_size": 64, "bias": True},
        "synchronous_updates": True,
        "shared_across_residues_steps_queries": True,
        "message_refresh": {"mutation_self": "self messages every step; attention inactive",
                            "graph_static": "once from mutation-injected H0; reuse same tensor without detach",
                            "graph_swarm": "every step from current Ht",
                            "graph_swarm_t1": "one step from H0"},
        "readout_dim": 188,
        "readout_input_order": ["mutation_row_hidden", "mean_node_hidden", "q"],
        "readout_mlp": {"dimensions": [188, 64, 1], "activation": "SiLU after first linear only", "bias": [True, True]},
        "output": "one scalar delta-delta-G per query; [B]",
        "dropout": 0, "batch_norm": False, "layer_norm": False,
        "extra_residual": False, "global_vector_inside_recurrence": False,
        "reference_mode": "same q in readout; omit local psi(q) injection only",
    }


@dataclass(frozen=True)
class ModelConfig:
    variant: str = "graph_swarm"
    steps: int | None = None

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown model variant: {self.variant!r}")
        steps = (1 if self.variant == "graph_swarm_t1" else 4) if self.steps is None else self.steps
        allowed = (1,) if self.variant == "graph_swarm_t1" else (4,) if self.variant == "mutation_self" else (1, 4)
        if type(steps) is not int or steps not in allowed:
            raise ValueError("Only frozen steps are allowed: main T=4, secondary/equivalence T=1")
        object.__setattr__(self, "steps", steps)

    def as_dict(self):
        # Keep metadata separate from state_dict so strict shared-weight loading
        # remains possible. Stage 3 must compare this alongside checkpoint tensors.
        return {"graph_protocol_version": PROTOCOL_VERSION,
                "architecture": architecture_specification(),
                "variant": self.variant, "steps": self.steps}


@dataclass(frozen=True)
class ModelDiagnostics:
    """Opt-in live autograd tensors; static messages/attention repeat by identity.

    hidden_states has T+1 entries; messages and attention have T entries.
    Attention entries are None for mutation_self. Values are not detached or
    copied: callers must not mutate them, and should release them after use.
    """
    hidden_states: tuple[torch.Tensor, ...]
    messages: tuple[torch.Tensor, ...]
    attention: tuple[torch.Tensor | None, ...]
    mutation_encoding: torch.Tensor
    mutation_rows: torch.Tensor
    mutation_injected: bool


def _integer_tensor(value, name, ndim=1):
    if not isinstance(value, torch.Tensor) or value.ndim != ndim or value.dtype not in _INTEGER_DTYPES:
        raise ValueError(f"{name} must be an integer tensor with {ndim} dimensions")


def encode_mutations(wt_indices, mut_indices, *, dtype=torch.float32):
    """Exactly 20 standard amino acids; no learned or biochemical descriptors."""
    _integer_tensor(wt_indices, "wt_indices")
    _integer_tensor(mut_indices, "mut_indices")
    if wt_indices.shape != mut_indices.shape or wt_indices.device != mut_indices.device:
        raise ValueError("WT and MUT indices must have matching [B] shapes and devices")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Mutation encoding requires float32 or float64")
    if ((wt_indices < 0) | (wt_indices >= 20) | (mut_indices < 0) | (mut_indices >= 20)).any():
        raise ValueError("Amino-acid indices must be in [0,20)")
    if (wt_indices == mut_indices).any():
        raise ValueError("WT and MUT must differ")
    a = F.one_hot(wt_indices.long(), num_classes=20).to(dtype=dtype)
    b = F.one_hot(mut_indices.long(), num_classes=20).to(dtype=dtype)
    return torch.cat((a, b, b - a), dim=-1)


def incoming_softmax(scores, destination, num_nodes):
    """Stable [B,E] softmax grouped by receiver, never across queries.

    Uses only native PyTorch scatter operations. Empty edge sets remain empty.
    """
    if scores.shape[1] == 0:
        return scores
    indices = destination.unsqueeze(0).expand_as(scores)
    maxima = scores.new_full((scores.shape[0], num_nodes), -torch.inf)
    maxima = maxima.scatter_reduce(1, indices, scores, reduce="amax", include_self=True)
    weights = torch.exp(scores - maxima.gather(1, indices))
    totals = scores.new_zeros((scores.shape[0], num_nodes)).scatter_add(1, indices, weights)
    return weights / totals.gather(1, indices)


class MutationGraphModel(nn.Module):
    """The four specified variants with identical registered module structure.

    mutation_self has no cross-node exchange inside recurrence. Other residues
    still contribute through the final mean readout. Attention is registered
    but inactive in this control. No labels or arbitrary keyword arguments are
    accepted. The T=1 override for primary variants is an equivalence diagnostic.
    """

    def __init__(self, variant="graph_swarm", *, steps=None):
        super().__init__()
        self.config = ModelConfig(variant, steps)
        self.projection = nn.Linear(384, 64, bias=True)
        self.mutation_mlp = nn.Sequential(nn.Linear(60, 64), nn.SiLU(), nn.Linear(64, 64))
        self.message_mlp = nn.Sequential(nn.Linear(145, 64), nn.SiLU(), nn.Linear(64, 64))
        self.attention_mlp = nn.Sequential(nn.Linear(145, 32), nn.SiLU(), nn.Linear(32, 1, bias=False))
        self.gru = nn.GRUCell(input_size=64, hidden_size=64, bias=True)
        self.readout = nn.Sequential(nn.Linear(188, 64), nn.SiLU(), nn.Linear(64, 1))

    def checkpoint_configuration(self):
        return self.config.as_dict()

    def parameter_counts(self):
        """Architecturally active counts for an ordinary injected forward.

        Counts are not a claim that each scalar has nonzero gradient on every
        graph. In particular attention has no effect on singleton neighborhoods.
        """
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        inactive = sum(p.numel() for p in self.attention_mlp.parameters() if p.requires_grad) if self.config.variant == "mutation_self" else 0
        return {"total_registered": total, "effectively_used": total - inactive}

    def _validate_inputs(self, features, seq_pos, edge_index, edge_attr,
                         mutation_positions, wt_indices, mut_indices):
        if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[1] != 384 or features.shape[0] == 0:
            raise ValueError("features must have shape [L,384] with L > 0")
        if features.dtype not in (torch.float32, torch.float64) or not torch.isfinite(features).all():
            raise ValueError("features must be finite float32 or float64")
        if features.dtype != self.projection.weight.dtype or features.device != self.projection.weight.device:
            raise ValueError("features must match model dtype and device; no implicit conversion")
        for name, value in (("seq_pos", seq_pos), ("mutation_positions", mutation_positions),
                            ("wt_indices", wt_indices), ("mut_indices", mut_indices)):
            _integer_tensor(value, name)
        _integer_tensor(edge_index, "edge_index", ndim=2)
        length = features.shape[0]
        if seq_pos.shape != (length,) or not torch.equal(seq_pos.long().sort().values,
                                                       torch.arange(length, device=seq_pos.device)):
            raise ValueError("seq_pos must map each immutable zero-based position exactly once")
        if mutation_positions.numel() == 0 or wt_indices.shape != mutation_positions.shape or mut_indices.shape != mutation_positions.shape:
            raise ValueError("Mutation query arrays must have matching nonempty [B] shapes")
        if ((mutation_positions < 0) | (mutation_positions >= length)).any():
            raise ValueError("mutation_positions must map to seq_pos")
        if edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2,E]")
        edges = edge_index.shape[1]
        if not isinstance(edge_attr, torch.Tensor) or edge_attr.shape != (edges, 17):
            raise ValueError("edge_attr must have shape [E,17]")
        if edge_attr.dtype != features.dtype or not torch.isfinite(edge_attr).all():
            raise ValueError("edge_attr must be finite and match feature dtype")
        values = (seq_pos, edge_index, edge_attr, mutation_positions, wt_indices, mut_indices)
        if any(value.device != features.device for value in values):
            raise ValueError("All inputs must be on the model device")
        if ((edge_index < 0) | (edge_index >= length)).any():
            raise ValueError("Graph endpoints must be current node row indices in [0,L)")
        if (edge_index[0] == edge_index[1]).any() or torch.unique(edge_index, dim=1).shape[1] != edges:
            raise ValueError("Graph edges must exclude self loops and duplicate directed edges")

    def graph_messages(self, hidden, edge_index, edge_attr):
        """All messages come from the same Ht before any GRU update."""
        source, destination = edge_index.long()
        batch, length, _ = hidden.shape
        inputs = torch.cat((hidden[:, destination], hidden[:, source],
                            edge_attr.unsqueeze(0).expand(batch, -1, -1)), dim=-1)
        values = self.message_mlp(inputs)
        scores = self.attention_mlp(inputs).squeeze(-1)
        attention = incoming_softmax(scores, destination, length)
        indices = destination.view(1, -1, 1).expand(batch, -1, 64)
        messages = hidden.new_zeros((batch, length, 64)).scatter_add(
            1, indices, attention.unsqueeze(-1) * values)
        return messages, attention

    def forward(self, features, seq_pos, edge_index, edge_attr,
                mutation_positions, wt_indices, mut_indices, *,
                return_diagnostics=False, inject_mutation=True):
        """Return predictions [B], or (predictions, ModelDiagnostics) if requested.

        inject_mutation=False gives the matched reference trajectory: only the
        local psi(q) addition is omitted; q is still passed to the readout.
        """
        if type(return_diagnostics) is not bool or type(inject_mutation) is not bool:
            raise ValueError("Diagnostic/injection switches must be booleans")
        self._validate_inputs(features, seq_pos, edge_index, edge_attr,
                              mutation_positions, wt_indices, mut_indices)
        q = encode_mutations(wt_indices, mut_indices, dtype=features.dtype)
        mask = seq_pos.unsqueeze(0) == mutation_positions.unsqueeze(1)
        rows = mask.long().argmax(dim=1)
        batch, length = mask.shape
        hidden = torch.tanh(self.projection(features)).unsqueeze(0).expand(batch, -1, -1)
        if inject_mutation:
            hidden = hidden + mask.unsqueeze(-1).to(features.dtype) * self.mutation_mlp(q).unsqueeze(1)
        states, messages_by_step, attention_by_step = [], [], []
        if return_diagnostics:
            states.append(hidden)
        static = self.config.variant == "graph_static"
        self_only = self.config.variant == "mutation_self"
        if static:
            messages, attention = self.graph_messages(hidden, edge_index, edge_attr)
        for _ in range(self.config.steps):
            if self_only:
                inputs = torch.cat((hidden, hidden, hidden.new_zeros((batch, length, 17))), dim=-1)
                messages, attention = self.message_mlp(inputs), None
            elif not static:
                messages, attention = self.graph_messages(hidden, edge_index, edge_attr)
            hidden = self.gru(messages.reshape(-1, 64), hidden.reshape(-1, 64)).reshape(batch, length, 64)
            if return_diagnostics:
                states.append(hidden)
                messages_by_step.append(messages)
                attention_by_step.append(attention)
        mutation_hidden = hidden[torch.arange(batch, device=hidden.device), rows]
        readout_input = torch.cat((mutation_hidden, hidden.mean(dim=1), q), dim=-1)
        predictions = self.readout(readout_input).squeeze(-1)
        if return_diagnostics:
            return predictions, ModelDiagnostics(tuple(states), tuple(messages_by_step),
                tuple(attention_by_step), q, rows, inject_mutation)
        return predictions
