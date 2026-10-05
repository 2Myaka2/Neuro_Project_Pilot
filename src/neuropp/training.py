"""GraphSWARM v1 development engine; importing this module performs no work.

Use DevelopmentDataAccess.load_megascale_csv to prepare separate MutationLabels,
then TrainingProtein(record, labels, access) and DevelopmentDataset. The latter
can be constructed with from_split_file to authenticate the split-file SHA256.
run_experiment owns seed/variant loops, selection and combined output tables.
Full mode has no scientific overrides; smoke defaults to two epochs, seed 42
and the primary pair. Resume=True replays an interrupted epoch from last.pt.
Only completed epochs are checkpointed. CPU determinism is supported; GPU
reduction order and cross-device/version bitwise reproducibility are not promised.
"""

from dataclasses import asdict, dataclass, field
import csv
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Mapping

import numpy as np
import torch

from .data_access import DevelopmentDataAccess, MutationLabels
from .graph import MutationQuery, ProteinRecord
from .models import MutationGraphModel, VARIANTS, architecture_specification
from .splits import file_hash


@dataclass(frozen=True)
class TrainingProtocol:
    optimizer: str = "Adam"
    learning_rate: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    amsgrad: bool = False
    epochs: int = 40
    seeds: tuple[int, ...] = tuple(range(42, 52))
    mutation_microbatch: int = 64
    gradient_clip_global_norm: float = 1.0
    dtype: str = "float32"
    scheduler: str = "none"
    early_stopping: bool = False
    AMP: bool = False
    variants: tuple[str, ...] = VARIANTS

    def as_dict(self):
        result = asdict(self)
        for key in ("betas", "seeds", "variants"):
            result[key] = list(result[key])
        result.update(checkpoint_selection="lowest validation protein macro MAE",
                      exact_validation_tie="keep earlier epoch",
                      primary_comparison=["graph_static", "graph_swarm"],
                      objective="per-protein MSE; one Adam step per protein per epoch",
                      protein_order="default_rng(SeedSequence([seed, epoch])).permutation(sorted IDs)")
        return result

    def validate(self, mode):
        if mode not in ("full", "smoke"):
            raise ValueError("mode must be full or smoke")
        actual, frozen = self.as_dict(), TrainingProtocol().as_dict()
        for key in actual:
            if mode == "smoke" and key in ("epochs", "seeds", "variants"):
                continue
            # JSON equality also distinguishes booleans from numeric overrides.
            if json.dumps(actual[key]) != json.dumps(frozen[key]):
                raise ValueError(f"Frozen training protocol mismatch: {key}")
        if mode == "smoke":
            if type(self.epochs) is not int or self.epochs not in (1, 2):
                raise ValueError("Smoke permits only 1–2 epochs")
            if len(self.seeds) != 1 or type(self.seeds[0]) is not int or self.seeds[0] < 0:
                raise ValueError("Smoke requires one nonnegative integer seed")
            if not self.variants or len(set(self.variants)) != len(self.variants) or not set(self.variants) <= set(VARIANTS):
                raise ValueError("Smoke variants must be distinct frozen architectures")
        return self


@dataclass(frozen=True)
class TrainingProtein:
    record: ProteinRecord
    labels: MutationLabels
    access: DevelopmentDataAccess = field(repr=False, compare=False)

    def __post_init__(self):
        self.validate()

    @property
    def protein_id(self):
        return self.record.protein_id

    def validate(self):
        self.access.require_development_ids((self.record.protein_id, self.labels.protein_id))
        if self.record.protein_id != self.labels.protein_id:
            raise ValueError("ProteinRecord/MutationLabels protein ID mismatch")
        self.record.validate()
        if len(self.labels.queries) == 0 or len(self.labels.queries) != len(self.labels.labels):
            raise ValueError("Expected matching nonempty query and target counts")
        if any(isinstance(y, (bool, np.bool_)) or not isinstance(y, Real)
               or not math.isfinite(y) for y in self.labels.labels):
            raise ValueError("Targets must be finite numbers")
        for query in self.labels.queries:
            if not isinstance(query, MutationQuery):
                raise ValueError("Expected MutationQuery")
            query.validate(self.record)
        return self


@dataclass(frozen=True)
class DevelopmentDataset:
    train: Mapping[str, TrainingProtein]
    validation: Mapping[str, TrainingProtein]
    access: DevelopmentDataAccess
    split_manifest_sha256: str
    feature_signature: str

    def __post_init__(self):
        for name in ("train", "validation"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        self.validate()

    @classmethod
    def from_split_file(cls, train, validation, split_path, feature_signature):
        path = Path(split_path)
        access = DevelopmentDataAccess(json.loads(path.read_text(encoding="utf-8")))
        return cls(train, validation, access, file_hash(path), feature_signature)

    def validate(self, mode=None):
        for value in (self.split_manifest_sha256, self.feature_signature):
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("Split SHA256 and feature signature must be SHA256 hex strings")
        # Preflight all IDs before inspecting any object's labels.
        self.access.require_development_ids((*self.train, *self.validation))
        for split in ("train", "validation"):
            values = getattr(self, split)
            self.access.require_development_ids(values, claimed_split=split)
            if not values:
                raise ValueError(f"Empty {split} dataset")
            if mode == "full":
                expected = {pid for pid, member in self.access.membership.items() if member == split}
                if set(values) != expected:
                    raise ValueError(f"Full mode requires exactly all {len(expected)} {split} proteins")
            for pid, protein in values.items():
                if not isinstance(protein, TrainingProtein) or protein.protein_id != pid:
                    raise ValueError("Dataset mapping requires matching TrainingProtein IDs")
                if dict(protein.access.membership) != dict(self.access.membership):
                    raise ValueError("Protein and dataset split memberships disagree")
                protein.validate()
        return self


@dataclass(frozen=True)
class TensorProtein:
    protein_id: str
    features: torch.Tensor
    seq_pos: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    mutation_position: torch.Tensor
    wt_index: torch.Tensor
    mut_index: torch.Tensor
    targets: torch.Tensor

    def forward_inputs(self, start=0, stop=None):
        """The seven label-free arguments accepted by the frozen Stage 2 head."""
        return (self.features, self.seq_pos, self.edge_index, self.edge_attr,
                self.mutation_position[start:stop], self.wt_index[start:stop], self.mut_index[start:stop])


def prepare_protein(protein, device="cpu"):
    protein.validate()
    record, labels = protein.record, protein.labels
    floating = lambda value: torch.tensor(value, device=device, dtype=torch.float32)
    integer = lambda value: torch.tensor(value, device=device, dtype=torch.int64)
    result = TensorProtein(protein.protein_id, floating(record.features).requires_grad_(False),
        integer(record.seq_pos), integer(record.edge_index), floating(record.edge_attr),
        integer([q.mutation_position for q in labels.queries]), integer([q.wt_index for q in labels.queries]),
        integer([q.mut_index for q in labels.queries]), floating(labels.labels))
    if not all(torch.isfinite(value).all() for value in (result.features, result.edge_attr, result.targets)):
        raise ValueError(f"Float32 conversion overflow: {protein.protein_id}")
    return result


class TrainingFailure(RuntimeError):
    pass


def _context(protein_id, variant, seed, epoch):
    return f"protein={protein_id} variant={variant} seed={seed} epoch={epoch}"


def _finite_gradients(model):
    return all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def accumulate_protein_gradients(model, optimizer, protein, *, variant, seed, epoch, microbatch=64):
    """Accumulate sum(error²)/n_p; no optimizer step or clipping in this helper.

    Accepts float64 TensorProtein fixtures for numerical tests. Production tensor
    preparation and the orchestration API always enforce float32 and batch 64.
    """
    optimizer.zero_grad(set_to_none=True)
    try:
        n = len(protein.targets)
        if n == 0 or type(microbatch) is not int or microbatch <= 0:
            raise ValueError("Nonempty protein and positive microbatch required")
        model.train()
        total = 0.0
        for start in range(0, n, microbatch):
            stop = min(start + microbatch, n)
            with torch.autocast(device_type=protein.features.device.type, enabled=False):
                prediction = model(*protein.forward_inputs(start, stop))
                if prediction.shape != protein.targets[start:stop].shape or not torch.isfinite(prediction).all():
                    raise ValueError("Nonfinite prediction or prediction shape mismatch")
                loss = (prediction - protein.targets[start:stop]).square().sum() / n
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite loss")
                loss.backward()
            total += float(loss.detach())
        if not _finite_gradients(model):
            raise ValueError("Nonfinite gradient")
        return total
    except Exception as exc:
        optimizer.zero_grad(set_to_none=True)
        raise TrainingFailure(f"{_context(protein.protein_id, variant, seed, epoch)}: {exc}") from exc


def train_protein(model, optimizer, protein, *, variant, seed, epoch):
    loss = accumulate_protein_gradients(model, optimizer, protein, variant=variant, seed=seed, epoch=epoch)
    try:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        if not _finite_gradients(model):
            raise ValueError("Nonfinite clipped gradient")
    except Exception as exc:
        optimizer.zero_grad(set_to_none=True)
        raise TrainingFailure(f"{_context(protein.protein_id, variant, seed, epoch)}: {exc}") from exc
    optimizer.step()  # Exactly once, after all microbatches and the finite checks.
    return loss


def protein_order(seed, epoch, protein_ids):
    """Pure order function, independent of model, global RNG and mapping order."""
    ids = sorted(protein_ids)
    if len(set(ids)) != len(ids) or type(seed) is not int or type(epoch) is not int or min(seed, epoch) < 0:
        raise ValueError("Distinct IDs and nonnegative integer seed/epoch required")
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    return tuple(ids[i] for i in rng.permutation(len(ids)))


def _cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def state_hash(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def paired_models(seed, variants=VARIANTS, device="cpu"):
    """One canonical seeded CPU state, strictly loaded into every variant."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        state = _cpu_state(_fresh_cpu_model("graph_swarm"))
        models = {}
        for variant in variants:
            model = _fresh_cpu_model(variant)
            model.load_state_dict(state, strict=True)
            models[variant] = model.to(device)
    return models, state_hash(state)


def _fresh_cpu_model(variant):
    # Owner notebooks may change default dtype/device. Canonical initialization
    # must still be CPU float32 and must restore the caller's defaults.
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device("cpu"):
            return MutationGraphModel(variant)
    finally:
        torch.set_default_dtype(dtype)


def _model_from_state(variant, state, device="cpu"):
    with torch.random.fork_rng(devices=[]):
        model = _fresh_cpu_model(variant)
    model.load_state_dict(state, strict=True)
    return model.to(device)


def make_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=0.0, amsgrad=False)


@dataclass(frozen=True)
class Evaluation:
    metrics: dict
    per_protein: tuple[dict, ...]
    predictions: tuple[dict, ...]


def evaluate_development(model, proteins, *, device="cpu", variant="development", seed=0, epoch=0):
    if not proteins:
        raise ValueError("Development evaluation requires nonempty proteins")
    # Validate every object before making predictions for any protein.
    for pid, protein in proteins.items():
        if pid != protein.protein_id:
            raise ValueError("Evaluation protein ID mismatch")
        protein.validate()
    model.eval()
    predictions, per_protein = [], []
    with torch.inference_mode(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        for pid in sorted(proteins):
            tensor = prepare_protein(proteins[pid], device)
            values = []
            for start in range(0, len(tensor.targets), 64):
                value = model(*tensor.forward_inputs(start, start + 64))
                if value.shape != tensor.targets[start:start + 64].shape or not torch.isfinite(value).all():
                    raise TrainingFailure(f"{_context(pid, variant, seed, epoch)}: nonfinite/invalid evaluation prediction")
                values.extend(value.double().cpu().tolist())
            errors = np.asarray(values) - tensor.targets.double().cpu().numpy()
            if not np.isfinite(errors).all():
                raise TrainingFailure(f"{_context(pid, variant, seed, epoch)}: nonfinite evaluation error")
            per_protein.append({"protein_id": pid, "MAE": float(np.abs(errors).mean()), "mutations": len(values)})
            for i, (prediction, error) in enumerate(zip(values, errors)):
                predictions.append({"protein_id": pid, "mutation_position": int(tensor.mutation_position[i]),
                    "wt_index": int(tensor.wt_index[i]), "mut_index": int(tensor.mut_index[i]),
                    "target": float(tensor.targets[i]), "prediction": prediction,
                    "error": float(error), "abs_error": float(abs(error))})
    errors = np.array([row["error"] for row in predictions], dtype=np.float64)
    metrics = {"protein_macro_MAE": float(np.mean([row["MAE"] for row in per_protein])),
               "pooled_MAE": float(np.abs(errors).mean()), "pooled_RMSE": float(np.sqrt(np.mean(errors ** 2)))}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise TrainingFailure("Nonfinite aggregate development metrics")
    return Evaluation(metrics, tuple(per_protein), tuple(predictions))


def is_better(current_macro_MAE, best_macro_MAE):
    if not math.isfinite(current_macro_MAE):
        raise ValueError("Selection metric must be finite")
    return current_macro_MAE < best_macro_MAE


def seed_effects(static, swarm):
    """Paired macro MAEs; sample SD (ddof=1), zero for a one-seed smoke."""
    left, right = np.asarray(static, dtype=np.float64), np.asarray(swarm, dtype=np.float64)
    if left.ndim != 1 or left.shape != right.shape or not left.size or not np.isfinite([left, right]).all():
        raise ValueError("Expected matching nonempty finite seed vectors")
    differences = left - right
    return {"effects": differences.tolist(), "mean_effect": float(differences.mean()),
            "standard_deviation": float(differences.std(ddof=1)) if len(differences) > 1 else 0.0,
            "positive_seed_count": int((differences > 1e-6).sum()),
            "negative_seed_count": int((differences < -1e-6).sum()),
            "tied_seed_count": int((np.abs(differences) <= 1e-6).sum()), "interpretation": "exploratory validation"}


def crossed_bootstrap(differences):
    """10,000 independent crossed seed × protein draws; fixed RNG seed 2026."""
    panel = np.asarray(differences, dtype=np.float64)
    if panel.ndim != 2 or min(panel.shape) == 0 or not np.isfinite(panel).all():
        raise ValueError("Expected a finite nonempty [seed, protein] effect panel")
    seeds, proteins = panel.shape
    rng = np.random.default_rng(2026)
    samples = np.empty(10_000, dtype=np.float64)
    for i in range(len(samples)):
        seed_indices = rng.integers(seeds, size=seeds)
        protein_indices = rng.integers(proteins, size=proteins)
        samples[i] = panel[np.ix_(seed_indices, protein_indices)].mean()
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"mean_effect": float(panel.mean()), "CI95_low": float(low), "CI95_high": float(high),
            "includes_zero": bool(low <= 0 <= high),
            "positive_seed_count": int((panel.mean(axis=1) > 1e-6).sum()),
            "replicates": 10_000, "rng_seed": 2026, "interpretation": "exploratory validation",
            "eventual_protected_support_rule": "CI95_low > 0 AND at least 7 of 10 seed effects > 1e-6"}


def _atomic_write(path, writer, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w+b" if binary else "w+"
    kwargs = {} if binary else {"encoding": "utf-8", "newline": ""}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode=mode, dir=path.parent, prefix="." + path.name,
                                         suffix=".tmp", delete=False, **kwargs) as stream:
            temporary = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_json(path, value):
    _atomic_write(path, lambda stream: json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False))


def _write_table(path, rows, columns):
    def write(stream):
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    _atomic_write(path, write)


class IncompatibleCheckpoint(ValueError):
    pass


def _check_state(state, model):
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise IncompatibleCheckpoint("Checkpoint state_dict key structure mismatch")
    for key, original in expected.items():
        value = state[key]
        if not isinstance(value, torch.Tensor) or value.shape != original.shape or value.dtype != original.dtype:
            raise IncompatibleCheckpoint(f"Checkpoint tensor shape/dtype mismatch: {key}")
        if not torch.isfinite(value).all():
            raise IncompatibleCheckpoint(f"Nonfinite checkpoint tensor: {key}")


def checkpoint_payload(model, optimizer, metadata, *, epoch, best_epoch, best_validation_macro_MAE,
                       best_model_state, history, kind="last"):
    if kind not in ("best", "last") or not 1 <= best_epoch <= epoch <= metadata["training_protocol"]["epochs"]:
        raise ValueError("Checkpoint requires a completed epoch boundary")
    if not math.isfinite(best_validation_macro_MAE):
        raise ValueError("Checkpoint selection metric must be finite")
    return {**metadata, "checkpoint_version": 1, "kind": kind, "epoch": epoch,
            "best_epoch": best_epoch, "best_validation_macro_MAE": best_validation_macro_MAE,
            "model_configuration": model.checkpoint_configuration(), "model_state_dict": _cpu_state(model),
            "optimizer_state_dict": optimizer.state_dict() if kind == "last" else None,
            # Retain the selected state in last so an interrupted best/last pair
            # can be reconstructed from the single authoritative epoch boundary.
            "best_model_state_dict": best_model_state if kind == "last" else None,
            "history": list(history)}


def save_checkpoint(path, payload):
    _atomic_write(path, lambda stream: torch.save(payload, stream), binary=True)


def load_checkpoint(path, model, optimizer, expected_metadata, *, kind="last"):
    """Validate all compatibility information before changing model/optimizer."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict):
        raise IncompatibleCheckpoint("Invalid checkpoint container")
    for key, expected in expected_metadata.items():
        if saved.get(key) != expected:
            raise IncompatibleCheckpoint(f"Checkpoint metadata mismatch: {key}")
    if saved.get("checkpoint_version") != 1 or saved.get("kind") != kind:
        raise IncompatibleCheckpoint("Checkpoint version/kind mismatch")
    if saved.get("model_configuration") != model.checkpoint_configuration():
        raise IncompatibleCheckpoint("Checkpoint model configuration mismatch")
    if model.checkpoint_configuration()["variant"] != expected_metadata["variant"]:
        raise IncompatibleCheckpoint("Model and expected variant disagree")
    epoch, best_epoch, score = saved.get("epoch"), saved.get("best_epoch"), saved.get("best_validation_macro_MAE")
    if (type(epoch) is not int or type(best_epoch) is not int
            or not 1 <= best_epoch <= epoch <= expected_metadata["training_protocol"]["epochs"]
            or not isinstance(score, (float, int)) or not math.isfinite(score)):
        raise IncompatibleCheckpoint("Invalid completed checkpoint epoch/metric")
    _check_state(saved.get("model_state_dict"), model)
    if kind == "last":
        _check_state(saved.get("best_model_state_dict"), model)
        _check_optimizer(saved.get("optimizer_state_dict"), optimizer, model)
    _check_history(saved, expected_metadata, kind)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    if kind == "last":
        optimizer.load_state_dict(saved["optimizer_state_dict"])
    return saved


def _check_optimizer(state, optimizer, model):
    if optimizer is None or not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
        raise IncompatibleCheckpoint("Resumable checkpoint lacks valid optimizer state")
    expected = optimizer.state_dict()["param_groups"]
    groups = state["param_groups"]
    if len(groups) != len(expected):
        raise IncompatibleCheckpoint("Optimizer parameter groups mismatch")
    parameters = list(model.parameters())
    ids = []
    for group, original in zip(groups, expected):
        if {k: v for k, v in group.items() if k != "params"} != {k: v for k, v in original.items() if k != "params"}:
            raise IncompatibleCheckpoint("Optimizer hyperparameters mismatch")
        if group["params"] != original["params"]:
            raise IncompatibleCheckpoint("Optimizer parameter order/count mismatch")
        ids.extend(group["params"])
    if len(set(ids)) != len(ids) or not set(state["state"]) <= set(ids):
        raise IncompatibleCheckpoint("Optimizer parameter IDs mismatch")
    names = [name for name, _ in model.named_parameters()]
    for identifier, parameter, name in zip(ids, parameters, names):
        entry = state["state"].get(identifier)
        if entry is None:
            if model.config.variant == "mutation_self" and name.startswith("attention_mlp."):
                continue
            raise IncompatibleCheckpoint(f"Missing active Adam parameter state: {name}")
        if set(entry) != {"step", "exp_avg", "exp_avg_sq"}:
            raise IncompatibleCheckpoint("Adam state structure mismatch")
        for name, value in entry.items():
            shape = torch.Size([]) if name == "step" else parameter.shape
            if not isinstance(value, torch.Tensor) or value.shape != shape or not torch.isfinite(value).all():
                raise IncompatibleCheckpoint(f"Invalid Adam state tensor: {name}")
            if name != "step" and value.dtype != parameter.dtype:
                raise IncompatibleCheckpoint("Adam state dtype mismatch")
            if name == "step" and (float(value) < 1 or float(value) != int(value)):
                raise IncompatibleCheckpoint("Invalid Adam completed step count")
            if name == "exp_avg_sq" and (value < 0).any():
                raise IncompatibleCheckpoint("Negative Adam squared moment")


def _check_history(saved, metadata, kind):
    history = saved.get("history")
    if (not isinstance(history, list) or not history or
            len(history) > metadata["training_protocol"]["epochs"] or
            (kind == "last" and len(history) != saved["epoch"])):
        raise IncompatibleCheckpoint("Checkpoint history lacks complete epoch boundaries")
    for epoch, row in enumerate(history, 1):
        if not isinstance(row, dict) or any(row.get(key) != value for key, value in
                {"epoch": epoch, "variant": metadata["variant"], "seed": metadata["seed"]}.items()):
            raise IncompatibleCheckpoint("Checkpoint history epoch/variant/seed mismatch")
        if any(row.get(key) != metadata[key] for key in ("mode", "scientific_status") if key in metadata):
            raise IncompatibleCheckpoint("Checkpoint history mixes full and smoke metadata")
        for key in ("train_protein_MSE", "protein_macro_MAE", "pooled_MAE", "pooled_RMSE"):
            value = row.get(key)
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise IncompatibleCheckpoint("Invalid checkpoint history metric")
    best = min(history, key=lambda row: row["protein_macro_MAE"])
    if best["epoch"] != saved["best_epoch"] or best["protein_macro_MAE"] != saved["best_validation_macro_MAE"]:
        raise IncompatibleCheckpoint("Checkpoint selection/history mismatch")


def _dataset_signature(dataset):
    """Resume additionally pins actual frozen arrays, queries and target order."""
    digest = hashlib.sha256()
    for split in ("train", "validation"):
        for pid, protein in sorted(getattr(dataset, split).items()):
            digest.update(json.dumps([split, pid, protein.record.sequence]).encode())
            for name in ("features", "seq_pos", "edge_index", "edge_attr"):
                value = np.ascontiguousarray(getattr(protein.record, name))
                digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
                digest.update(value.tobytes())
            digest.update(json.dumps([[int(q.mutation_position), int(q.wt_index), int(q.mut_index)]
                                      for q in protein.labels.queries]).encode())
            digest.update(np.asarray(protein.labels.labels, dtype=np.float64).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ExperimentResult:
    manifest: dict
    history: tuple[dict, ...]
    validation_summary: tuple[dict, ...]
    validation_per_protein: tuple[dict, ...]
    validation_predictions: tuple[dict, ...]
    validation_effects: tuple[dict, ...]
    validation_bootstrap: dict
    results_directory: Path
    weights_directory: Path


def _run_metadata(manifest, variant, seed):
    result = {key: manifest[key] for key in ("run_id", "mode", "scientific_status", "training_protocol",
        "split_manifest_sha256", "feature_signature", "dataset_signature", "pytorch_version", "device_description",
        "torch_num_threads")}
    result.update(variant=variant, seed=seed, initialization_hash=manifest["initialization_hashes"][str(seed)])
    return result


def run_experiment(dataset, mode, device, run_id, *, output_root=".", protocol=None, resume=False):
    """Run the frozen development experiment (owner executes real data later).

    A fresh run requires unused results/weights directories. Only explicit
    resume may reuse a matching manifest. Smoke and full metadata are mutually
    incompatible, even if a caller reuses the same run_id. Successful completion
    removes last checkpoints AFTER every combined output is atomically written.
    """
    if not isinstance(run_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id) is None:
        raise ValueError("run_id must be a simple directory name")
    protocol = protocol or (TrainingProtocol() if mode == "full" else
                            TrainingProtocol(epochs=2, seeds=(42,), variants=("graph_static", "graph_swarm")))
    protocol.validate(mode)
    dataset.validate(mode)
    device = torch.device(device)
    device_description = str(device)
    if device.type == "cuda":
        device_description += ":" + torch.cuda.get_device_name(device)
    results = Path(output_root) / "results/graph_swarm_v1" / run_id
    weights = Path(output_root) / "weights/graph_swarm_v1" / run_id
    status = "NON-SCIENTIFIC SMOKE" if mode == "smoke" else "full development; exploratory validation"
    manifest = {"run_id": run_id, "mode": mode, "scientific_status": status,
                "training_protocol": protocol.as_dict(), "architecture": architecture_specification(),
                "split_manifest_sha256": dataset.split_manifest_sha256, "feature_signature": dataset.feature_signature,
                "dataset_signature": _dataset_signature(dataset), "pytorch_version": str(torch.__version__),
                "device_description": device_description, "torch_num_threads": torch.get_num_threads(),
                "train_protein_ids": sorted(dataset.train),
                "validation_protein_ids": sorted(dataset.validation), "initialization_hashes": {}}
    # Compute only the one canonical initialization at a time; no run is trained
    # while checking compatibility with an existing directory.
    initial_states = {}
    for seed in protocol.seeds:
        models, digest = paired_models(seed, variants=("graph_swarm",))
        initial_states[seed] = _cpu_state(models["graph_swarm"])
        manifest["initialization_hashes"][str(seed)] = digest
        del models
    manifest_path = results / "manifest.json"
    completed = False
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        completed = previous.pop("completed", False)
        if type(completed) is not bool or previous != manifest:
            raise IncompatibleCheckpoint("Existing run manifest is incompatible; choose a new run_id")
        if not resume:
            raise FileExistsError("Existing run requires resume=True or a new run_id")
    elif any(folder.exists() and any(folder.iterdir()) for folder in (results, weights)):
        raise IncompatibleCheckpoint("Existing run directory has no compatible manifest")
    else:
        _write_json(manifest_path, {**manifest, "completed": False})
    # Validate ALL existing checkpoints before changing ANY checkpoint in the
    # directory, including a later variant that has not yet been trained.
    for seed in protocol.seeds:
        for variant in protocol.variants:
            model = _model_from_state(variant, initial_states[seed])
            optimizer = make_optimizer(model)
            metadata = _run_metadata(manifest, variant, seed)
            for kind in ("last", "best"):
                path = weights / f"{variant}_seed{seed}_{kind}.pt"
                if path.exists():
                    load_checkpoint(path, model, optimizer if kind == "last" else None, metadata, kind=kind)
                elif completed and kind == "best":
                    raise IncompatibleCheckpoint(f"Completed run lacks selected checkpoint: {path.name}")
            del model, optimizer
    weights.mkdir(parents=True, exist_ok=True)
    history, summary, per_protein, predictions = [], [], [], []
    tags = {"mode": mode, "scientific_status": status}
    for seed in protocol.seeds:
        # Instantiate one device-resident model at a time; paired canonical CPU
        # construction avoids keeping four heads on GPU during a run.
        for variant in protocol.variants:
            model = _model_from_state(variant, initial_states[seed], device)
            optimizer = make_optimizer(model)
            metadata = _run_metadata(manifest, variant, seed)
            best_path = weights / f"{variant}_seed{seed}_best.pt"
            last_path = weights / f"{variant}_seed{seed}_last.pt"
            start, best_epoch, best_score, best_state, local_history = 1, 0, math.inf, None, []
            if completed:
                selected = load_checkpoint(best_path, model, None, metadata, kind="best")
                # Completed histories are stored in the one combined table.
                with (results / "history.csv").open(newline="", encoding="utf-8") as stream:
                    local_history = [_parse_history(row) for row in csv.DictReader(stream)
                                     if row["variant"] == variant and int(row["seed"]) == seed]
                _check_history({**selected, "history": local_history, "epoch": protocol.epochs}, metadata, "last")
                start, best_epoch, best_score = protocol.epochs + 1, selected["best_epoch"], selected["best_validation_macro_MAE"]
            elif last_path.exists():
                selected = load_checkpoint(last_path, model, optimizer, metadata)
                start, best_epoch, best_score = selected["epoch"] + 1, selected["best_epoch"], selected["best_validation_macro_MAE"]
                best_state, local_history = selected["best_model_state_dict"], selected["history"]
            elif best_path.exists():
                # Crash during the first best/last pair: no resumable completed
                # epoch exists. Check compatibility, then restart from canonical.
                load_checkpoint(best_path, model, None, metadata, kind="best")
                model = _model_from_state(variant, initial_states[seed], device)
                optimizer = make_optimizer(model)
            for epoch in range(start, protocol.epochs + 1):
                losses = []
                for pid in protein_order(seed, epoch, dataset.train):
                    tensor = prepare_protein(dataset.train[pid], device)
                    losses.append(train_protein(model, optimizer, tensor, variant=variant, seed=seed, epoch=epoch))
                    del tensor
                validation = evaluate_development(model, dataset.validation, device=device, variant=variant, seed=seed, epoch=epoch)
                score = validation.metrics["protein_macro_MAE"]
                if is_better(score, best_score):
                    best_epoch, best_score, best_state = epoch, score, _cpu_state(model)
                local_history.append({**tags, "variant": variant, "seed": seed, "epoch": epoch,
                                      "train_protein_MSE": float(np.mean(losses)), **validation.metrics})
                # Authoritative last holds selected state AND history; best can
                # always be repaired following a crash between the two writes.
                payload = checkpoint_payload(model, optimizer, metadata, epoch=epoch, best_epoch=best_epoch,
                    best_validation_macro_MAE=best_score, best_model_state=best_state, history=local_history)
                save_checkpoint(last_path, payload)
                best_payload = {**payload, "kind": "best", "epoch": best_epoch,
                                "model_state_dict": best_state, "best_model_state_dict": None,
                                "optimizer_state_dict": None}
                save_checkpoint(best_path, best_payload)
            if not completed:
                model.load_state_dict(best_state, strict=True)
                payload = checkpoint_payload(model, optimizer, metadata, epoch=best_epoch, best_epoch=best_epoch,
                    best_validation_macro_MAE=best_score, best_model_state=best_state, history=local_history, kind="best")
                save_checkpoint(best_path, payload)
            validation = evaluate_development(model, dataset.validation, device=device, variant=variant, seed=seed, epoch=best_epoch)
            run_tags = {**tags, "variant": variant, "seed": seed, "best_epoch": best_epoch}
            history.extend(local_history)
            summary.append({**run_tags, **validation.metrics})
            per_protein.extend({**run_tags, **row} for row in validation.per_protein)
            predictions.extend({**run_tags, **row} for row in validation.predictions)
            del model, optimizer, validation, best_state
    effects, bootstrap = [], {"interpretation": "exploratory validation", **tags, "primary_pair_available": False}
    if {"graph_static", "graph_swarm"} <= set(protocol.variants):
        macro = {(row["variant"], row["seed"]): row["protein_macro_MAE"] for row in summary}
        analysis = seed_effects([macro["graph_static", seed] for seed in protocol.seeds],
                                [macro["graph_swarm", seed] for seed in protocol.seeds])
        effects = [{**tags, "seed": seed, "static_macro_MAE": macro["graph_static", seed],
                    "swarm_macro_MAE": macro["graph_swarm", seed], "static_minus_swarm": effect,
                    "interpretation": "exploratory validation"} for seed, effect in zip(protocol.seeds, analysis["effects"])]
        panel = {(row["variant"], row["seed"], row["protein_id"]): row["MAE"] for row in per_protein}
        differences = np.array([[panel["graph_static", seed, pid] - panel["graph_swarm", seed, pid]
                                 for pid in sorted(dataset.validation)] for seed in protocol.seeds])
        bootstrap = {**crossed_bootstrap(differences), **tags, "primary_pair_available": True, "seed_effect_summary": analysis}
    base = ["mode", "scientific_status", "variant", "seed"]
    tables = {
        "history.csv": (history, base + ["epoch", "train_protein_MSE", "protein_macro_MAE", "pooled_MAE", "pooled_RMSE"]),
        "validation_summary.csv": (summary, base + ["best_epoch", "protein_macro_MAE", "pooled_MAE", "pooled_RMSE"]),
        "validation_per_protein.csv": (per_protein, base + ["best_epoch", "protein_id", "MAE", "mutations"]),
        "validation_predictions.csv": (predictions, base + ["best_epoch", "protein_id", "mutation_position", "wt_index", "mut_index",
                                                       "target", "prediction", "error", "abs_error"]),
        "validation_effects.csv": (effects, ["mode", "scientific_status", "seed", "static_macro_MAE", "swarm_macro_MAE",
                                             "static_minus_swarm", "interpretation"])}
    for filename, (rows, columns) in tables.items():
        _write_table(results / filename, rows, columns)
    _write_json(results / "validation_bootstrap.json", bootstrap)
    manifest["completed"] = True
    _write_json(manifest_path, manifest)
    for seed in protocol.seeds:
        for variant in protocol.variants:
            (weights / f"{variant}_seed{seed}_last.pt").unlink(missing_ok=True)
    return ExperimentResult(manifest, tuple(history), tuple(summary), tuple(per_protein), tuple(predictions),
                            tuple(effects), bootstrap, results, weights)


def _parse_history(row):
    return {key: int(value) if key in ("seed", "epoch") else float(value) if key in
            ("train_protein_MSE", "protein_macro_MAE", "pooled_MAE", "pooled_RMSE") else value
            for key, value in row.items()}
