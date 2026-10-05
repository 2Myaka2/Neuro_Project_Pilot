"""Stage 2 CPU synthetic tests. No dataset, legacy outputs or real labels.

Chain fixtures intentionally exercise the head on a sparse test topology; they
do not replace the fixed k=16 Stage 1 graph builder for real proteins.
"""

import dataclasses
import io
import json
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import torch
from torch.nn import functional as F

from neuropp.graph import make_record
from neuropp.models import (MUTATION_ALPHABET, ModelConfig, MutationGraphModel,
                            VARIANTS, architecture_specification, encode_mutations,
                            incoming_softmax)


def tensor_fixture(length=5, dtype=torch.float64, chain=False):
    generator = torch.Generator().manual_seed(120)
    features = torch.randn(length, 384, generator=generator, dtype=dtype) * 0.1
    pairs = ([(i, i + 1) for i in range(length - 1)] + [(i + 1, i) for i in range(length - 1)]) if chain else [
        (j, i) for i in range(length) for j in range(length) if i != j]
    edges = torch.tensor(pairs, dtype=torch.long).T.contiguous() if pairs else torch.empty(2, 0, dtype=torch.long)
    attrs = torch.randn(len(pairs), 17, generator=generator, dtype=dtype) * 0.1
    return (features, torch.arange(length), edges, attrs,
            torch.tensor([0, length - 1]), torch.tensor([0, 2]), torch.tensor([1, 3]))


def seeded_model(variant="graph_swarm", dtype=torch.float64, steps=None):
    # Preserve the caller's RNG and make every fixture reproducible.
    with torch.random.fork_rng():
        torch.manual_seed(314)
        return MutationGraphModel(variant, steps=steps).to(dtype=dtype).eval()


def propagation_model(variant):
    """One active coordinate, positive injection and sender-dependent messages.

    GRU gates are 1/2, candidate=tanh(message); no hidden candidate term.
    Thus h' = 0.5*h + 0.5*tanh(mean(SiLU(sender_hidden))).
    """
    model = seeded_model(variant)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.mutation_mlp[2].bias[0] = 1
        model.message_mlp[0].weight[0, 64] = 1
        model.message_mlp[2].weight[0, 0] = 1
        model.gru.weight_ih[128, 0] = 1
        model.readout[0].weight[0, 0] = 1
        model.readout[2].weight[0, 0] = 1
    return model


def single_query(inputs, index):
    return (*inputs[:4], *(value[index:index + 1] for value in inputs[4:]))


def synthetic_learning_result():
    """Fixed 60-update trainability check; no search, validation or real labels."""
    model = seeded_model("graph_swarm", torch.float32).train()
    base = tensor_fixture(4, torch.float32)
    inputs = (*base[:4], torch.tensor([0, 1, 2, 3]), torch.tensor([0, 2, 4, 6]),
              torch.tensor([1, 3, 5, 7]))
    # Arbitrary prescribed targets, independent of model outputs and data files.
    target = torch.tensor([-0.75, -0.25, 0.25, 0.75])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial = F.mse_loss(model(*inputs), target).item()
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(*inputs), target)
        loss.backward()
        optimizer.step()
    final = F.mse_loss(model(*inputs), target).item()
    return {"variant": "graph_swarm", "updates": 60, "initial_mse": initial,
            "final_mse": final, "final_initial_ratio": final / initial}


class ModelTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assertClose(self, actual, expected, dtype=torch.float64):
        atol, rtol = (1e-5, 1e-5) if dtype == torch.float32 else (1e-8, 1e-7)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


class MutationEncodingTests(ModelTestCase):
    def test_shape_and_exact_components_for_all_twenty_amino_acids(self):
        self.assertEqual(MUTATION_ALPHABET, "ACDEFGHIKLMNPQRSTVWY")
        wt = torch.arange(20)
        mut = (wt + 1) % 20
        q = encode_mutations(wt, mut, dtype=torch.float64)
        self.assertEqual(q.shape, (20, 60))
        a, b = torch.eye(20, dtype=torch.float64), torch.eye(20, dtype=torch.float64)[mut]
        self.assertTrue(torch.equal(q[:, :20], a))
        self.assertTrue(torch.equal(q[:, 20:40], b))
        self.assertTrue(torch.equal(q[:, 40:], b - a))

    def test_invalid_indices_fail(self):
        for value in (-1, 20, 21):
            for wt, mut in ((value, 1), (0, value)):
                with self.subTest(wt=wt, mut=mut), self.assertRaises(ValueError):
                    encode_mutations(torch.tensor([wt]), torch.tensor([mut]))

    def test_identical_wt_mut_fails(self):
        with self.assertRaisesRegex(ValueError, "must differ"):
            encode_mutations(torch.tensor([2]), torch.tensor([2]))

    def test_noninteger_boolean_and_mismatched_shapes_fail(self):
        for wt, mut in ((torch.tensor([0.0]), torch.tensor([1])),
                        (torch.tensor([False]), torch.tensor([True])),
                        (torch.tensor([[0]]), torch.tensor([1])),
                        (torch.tensor([0, 2]), torch.tensor([1]))):
            with self.subTest(wt=wt), self.assertRaises(ValueError):
                encode_mutations(wt, mut)


class InitializationTests(ModelTestCase):
    def test_injection_only_at_immutable_position(self):
        inputs = list(tensor_fixture())
        permutation = torch.tensor([3, 0, 4, 1, 2])
        inputs[0], inputs[1] = inputs[0][permutation], inputs[1][permutation]
        model = seeded_model()
        _, diagnostics = model(*inputs, return_diagnostics=True)
        z = torch.tanh(model.projection(inputs[0]))
        q = encode_mutations(*inputs[5:], dtype=torch.float64)
        injection = model.mutation_mlp(q)
        expected = z.unsqueeze(0).repeat(2, 1, 1)
        expected[0, 1] += injection[0]
        expected[1, 2] += injection[1]
        self.assertTrue(torch.equal(diagnostics.hidden_states[0], expected))
        self.assertTrue(torch.equal(diagnostics.mutation_rows, torch.tensor([1, 2])))

    def test_joint_row_permutation_preserves_biological_injection(self):
        inputs = tensor_fixture()
        permutation = torch.tensor([3, 0, 4, 1, 2])
        inverse = torch.argsort(permutation)
        permuted = (inputs[0][permutation], inputs[1][permutation], inverse[inputs[2]], *inputs[3:])
        model = seeded_model()
        _, first = model(*inputs, return_diagnostics=True)
        _, second = model(*permuted, return_diagnostics=True)
        self.assertTrue(torch.equal(first.hidden_states[0][:, permutation], second.hidden_states[0]))

    def test_initial_queries_are_independent(self):
        inputs, model = tensor_fixture(), seeded_model()
        _, together = model(*inputs, return_diagnostics=True)
        for query in range(2):
            _, alone = model(*single_query(inputs, query), return_diagnostics=True)
            self.assertClose(together.hidden_states[0][query:query + 1], alone.hidden_states[0])

    def test_reference_omits_only_injection_and_keeps_q_in_readout(self):
        inputs, model = tensor_fixture(), seeded_model()
        captured = []
        hook = model.readout.register_forward_pre_hook(lambda _, args: captured.append(args[0]))
        try:
            _, injected = model(*inputs, return_diagnostics=True)
            _, reference = model(*inputs, return_diagnostics=True, inject_mutation=False)
        finally:
            hook.remove()
        expected = torch.tanh(model.projection(inputs[0])).unsqueeze(0).expand(2, -1, -1)
        self.assertTrue(torch.equal(reference.hidden_states[0], expected))
        self.assertTrue(torch.equal(reference.mutation_encoding, injected.mutation_encoding))
        for readout_input in captured:
            self.assertTrue(torch.equal(readout_input[:, -60:], injected.mutation_encoding))
        self.assertFalse(reference.mutation_injected)


class AggregationTests(ModelTestCase):
    def manual_fixture(self):
        model = seeded_model()
        with torch.no_grad():
            for module in (model.message_mlp, model.attention_mlp):
                for parameter in module.parameters():
                    parameter.zero_()
            model.message_mlp[0].weight[0, [0, 64, 128]] = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
            model.message_mlp[2].weight[0, 0] = 1
            model.attention_mlp[0].weight[0, [0, 64, 128]] = torch.tensor([0.4, 0.7, -0.2], dtype=torch.float64)
            model.attention_mlp[2].weight[0, 0] = 1
        hidden = torch.zeros(2, 4, 64, dtype=torch.float64)
        hidden[:, :, 0] = torch.tensor([[0.2, 0.4, 0.8, 1.6], [-1, 0.3, 2, -0.2]])
        edges = torch.tensor([[0, 2, 1], [1, 1, 2]])
        attrs = torch.zeros(3, 17, dtype=torch.float64)
        attrs[:, 0] = torch.tensor([0.1, 0.5, -0.3])
        return model, hidden, edges, attrs

    def test_hand_calculated_incoming_groups_and_orientation(self):
        model, hidden, edges, attrs = self.manual_fixture()
        messages, attention = model.graph_messages(hidden, edges, attrs)
        expected_attention = torch.empty_like(attention)
        expected_messages = torch.zeros_like(messages)
        for query in range(2):
            # Receiver=1, senders=0,2. Last edge receiver=2, sender=1.
            logits = F.silu(0.4 * hidden[query, edges[1], 0] +
                            0.7 * hidden[query, edges[0], 0] - 0.2 * attrs[:, 0])
            values = F.silu(hidden[query, edges[1], 0] +
                            2 * hidden[query, edges[0], 0] + 3 * attrs[:, 0])
            expected_attention[query, :2] = torch.softmax(logits[:2], dim=0)
            expected_attention[query, 2] = 1
            expected_messages[query, 1, 0] = (expected_attention[query, :2] * values[:2]).sum()
            expected_messages[query, 2, 0] = values[2]
        self.assertClose(attention, expected_attention)
        self.assertClose(messages, expected_messages)

    def test_each_incoming_group_sums_to_one(self):
        model, hidden, edges, attrs = self.manual_fixture()
        _, attention = model.graph_messages(hidden, edges, attrs)
        for destination in (1, 2):
            self.assertClose(attention[:, edges[1] == destination].sum(dim=1), torch.ones(2, dtype=torch.float64))

    def test_zero_incoming_nodes_have_exact_zero_messages(self):
        model, hidden, edges, attrs = self.manual_fixture()
        messages, _ = model.graph_messages(hidden, edges, attrs)
        self.assertEqual(torch.count_nonzero(messages[:, [0, 3]]).item(), 0)

    def test_query_normalizations_and_messages_do_not_mix(self):
        model, hidden, edges, attrs = self.manual_fixture()
        messages, attention = model.graph_messages(hidden, edges, attrs)
        self.assertFalse(torch.equal(attention[0], attention[1]))
        for query in range(2):
            separate_messages, separate_attention = model.graph_messages(hidden[query:query + 1], edges, attrs)
            self.assertClose(messages[query:query + 1], separate_messages)
            self.assertClose(attention[query:query + 1], separate_attention)

    def test_stable_softmax_with_extreme_scores(self):
        scores = torch.tensor([[10000., 9999., -10000., -9999.]], dtype=torch.float64, requires_grad=True)
        attention = incoming_softmax(scores, torch.tensor([1, 1, 2, 2]), 4)
        self.assertTrue(torch.isfinite(attention).all())
        self.assertClose(attention[:, :2], torch.softmax(scores[:, :2], dim=1))
        self.assertClose(attention[:, 2:], torch.softmax(scores[:, 2:], dim=1))
        (attention * torch.arange(4)).sum().backward()
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_singleton_empty_edges_all_variants(self):
        inputs = tensor_fixture(1)
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                prediction, diagnostics = seeded_model(variant)(*inputs, return_diagnostics=True)
                self.assertEqual(prediction.shape, (2,))
                self.assertTrue(torch.isfinite(prediction).all())
                if variant != "mutation_self":
                    for messages, attention in zip(diagnostics.messages, diagnostics.attention):
                        self.assertEqual(torch.count_nonzero(messages).item(), 0)
                        self.assertEqual(attention.shape, (2, 0))


class ArchitectureTests(ModelTestCase):
    def test_exact_modules_dimensions_biases_and_shared_cell(self):
        for variant in VARIANTS:
            model = seeded_model(variant)
            self.assertEqual((model.projection.in_features, model.projection.out_features), (384, 64))
            for module, dimensions, biases in ((model.mutation_mlp, [60, 64, 64], [True, True]),
                    (model.message_mlp, [145, 64, 64], [True, True]),
                    (model.attention_mlp, [145, 32, 1], [True, False]),
                    (model.readout, [188, 64, 1], [True, True])):
                self.assertEqual(len(module), 3)
                self.assertIsInstance(module[1], torch.nn.SiLU)
                for index, linear in enumerate((module[0], module[2])):
                    self.assertEqual((linear.in_features, linear.out_features), tuple(dimensions[index:index + 2]))
                    self.assertEqual(linear.bias is not None, biases[index])
            cells = [module for module in model.modules() if isinstance(module, torch.nn.GRUCell)]
            self.assertEqual(cells, [model.gru])
            self.assertEqual((model.gru.input_size, model.gru.hidden_size, model.gru.bias), (64, 64, True))
            self.assertFalse(any(isinstance(module, (torch.nn.Dropout, torch.nn.LayerNorm, torch.nn.modules.batchnorm._BatchNorm)) for module in model.modules()))
            self.assertEqual(model.config.steps, 1 if variant == "graph_swarm_t1" else 4)

    def test_state_dict_keys_shapes_and_strict_cross_loading(self):
        static, swarm = seeded_model("graph_static"), seeded_model("graph_swarm")
        self.assertEqual({k: tuple(v.shape) for k, v in static.state_dict().items()},
                         {k: tuple(v.shape) for k, v in swarm.state_dict().items()})
        swarm.load_state_dict(static.state_dict(), strict=True)
        static.load_state_dict(swarm.state_dict(), strict=True)

    def test_exact_total_and_effectively_used_parameter_counts(self):
        for variant in VARIANTS:
            model = seeded_model(variant)
            expected = {"total_registered": 88033,
                        "effectively_used": 83329 if variant == "mutation_self" else 88033}
            self.assertEqual(model.parameter_counts(), expected)
            self.assertEqual(sum(p.numel() for p in model.attention_mlp.parameters()), 4704)
        self.assertEqual(seeded_model("graph_static").parameter_counts(), seeded_model().parameter_counts())

    def test_constructor_configuration_is_explicit_and_separate_from_weights(self):
        configs = [seeded_model(variant).checkpoint_configuration() for variant in VARIANTS]
        self.assertEqual(len({json.dumps(config, sort_keys=True) for config in configs}), 4)
        for config in configs:
            self.assertEqual(config["architecture"], architecture_specification())
            self.assertEqual(config["graph_protocol_version"], "graph_swarm_v1:1")
            json.dumps(config, allow_nan=False)
        config = configs[0]
        config["architecture"]["hidden_dim"] = 32
        self.assertEqual(seeded_model("mutation_self").checkpoint_configuration()["architecture"]["hidden_dim"], 64)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            seeded_model().config.steps = 2

    def test_unknown_variants_and_unfrozen_steps_rejected(self):
        for variant, steps in (("new_architecture", None), ("graph_swarm", 2),
                              ("graph_static", True), ("graph_swarm_t1", 4), ("mutation_self", 1)):
            with self.subTest(variant=variant, steps=steps), self.assertRaises(ValueError):
                ModelConfig(variant, steps)

    def test_protocol_model_section_matches_implementation(self):
        root = Path(__file__).resolve().parents[2]
        protocol = json.loads((root / "experiments/graph_swarm_v1/protocol.json").read_text())
        self.assertEqual(protocol["model"], architecture_specification())

    def test_readout_is_mutation_hidden_mean_and_same_q(self):
        inputs, model, captured = tensor_fixture(), seeded_model(), []
        hook = model.readout.register_forward_pre_hook(lambda _, args: captured.append(args[0]))
        try:
            prediction, diagnostics = model(*inputs, return_diagnostics=True)
        finally:
            hook.remove()
        hidden = diagnostics.hidden_states[-1]
        expected = torch.cat((hidden[torch.arange(2), diagnostics.mutation_rows],
                              hidden.mean(dim=1), diagnostics.mutation_encoding), dim=-1)
        self.assertTrue(torch.equal(captured[0], expected))
        self.assertEqual(prediction.shape, (2,))

    def test_mutation_self_message_inputs_and_inactive_attention(self):
        inputs, model, captured = tensor_fixture(), seeded_model("mutation_self"), []
        hook = model.message_mlp.register_forward_pre_hook(lambda _, args: captured.append(args[0]))
        try:
            with mock.patch.object(model.attention_mlp, "forward", side_effect=AssertionError("inactive")):
                _, diagnostics = model(*inputs, return_diagnostics=True)
        finally:
            hook.remove()
        for index, message_input in enumerate(captured):
            self.assertTrue(torch.equal(message_input[:, :, :64], diagnostics.hidden_states[index]))
            self.assertTrue(torch.equal(message_input[:, :, 64:128], diagnostics.hidden_states[index]))
            self.assertEqual(torch.count_nonzero(message_input[:, :, 128:]).item(), 0)

    def test_mutation_self_other_residues_influence_global_readout(self):
        inputs, model = tensor_fixture(), propagation_model("mutation_self")
        with torch.no_grad():
            model.projection.weight[0, 0] = 1
            model.readout[0].weight.zero_()
            model.readout[0].weight[0, 64] = 1  # use only the global mean coordinate
        first = single_query(inputs, 0)
        features = first[0].clone()
        features[3, 0] += 1
        changed = (features, *first[1:])
        self.assertGreater((model(*first) - model(*changed)).abs().item(), 1e-4)


class StaticSwarmTests(ModelTestCase):
    def test_t1_predictions_and_every_active_parameter_gradient_equal(self):
        for dtype in (torch.float32, torch.float64):
            inputs = tensor_fixture(dtype=dtype)
            models = [seeded_model("graph_static", dtype, steps=1),
                      seeded_model("graph_swarm", dtype, steps=1),
                      seeded_model("graph_swarm_t1", dtype)]
            for model in models[1:]:
                model.load_state_dict(models[0].state_dict(), strict=True)
            outputs = []
            for model in models:
                prediction = model(*inputs)
                outputs.append(prediction)
                loss = ((prediction - torch.tensor([0.4, -0.7], dtype=dtype)) ** 2).sum() + 0.2 * prediction.sum()
                loss.backward()
            for other in range(1, len(models)):
                self.assertClose(outputs[0], outputs[other], dtype)
                for (name, first), (other_name, second) in zip(models[0].named_parameters(), models[other].named_parameters()):
                    self.assertEqual(name, other_name)
                    self.assertIsNotNone(first.grad, name)
                    self.assertIsNotNone(second.grad, name)
                    self.assertClose(first.grad, second.grad, dtype)

    def test_t4_constructed_refresh_changes_hidden_states(self):
        inputs = single_query(tensor_fixture(7, chain=True), 0)
        static, swarm = propagation_model("graph_static"), propagation_model("graph_swarm")
        swarm.load_state_dict(static.state_dict(), strict=True)
        _, fixed = static(*inputs, return_diagnostics=True)
        _, refreshed = swarm(*inputs, return_diagnostics=True)
        self.assertTrue(torch.equal(fixed.hidden_states[1], refreshed.hidden_states[1]))
        self.assertEqual(fixed.hidden_states[2][0, 2, 0].item(), 0)
        self.assertGreater(refreshed.hidden_states[2][0, 2, 0].item(), 1e-3)
        self.assertGreater((fixed.hidden_states[4] - refreshed.hidden_states[4]).abs().max().item(), 1e-3)

    def test_refresh_call_counts_static_tensor_identity_and_new_forward(self):
        inputs = tensor_fixture()
        for variant, expected in (("graph_static", 1), ("graph_swarm", 4), ("graph_swarm_t1", 1)):
            model = seeded_model(variant)
            with mock.patch.object(model, "graph_messages", wraps=model.graph_messages) as messages:
                _, diagnostics = model(*inputs, return_diagnostics=True)
                self.assertEqual(messages.call_count, expected)
                _, second = model(*inputs, return_diagnostics=True)
                self.assertEqual(messages.call_count, 2 * expected)
            if variant == "graph_static":
                self.assertTrue(all(value is diagnostics.messages[0] for value in diagnostics.messages))
                self.assertTrue(all(value is diagnostics.attention[0] for value in diagnostics.attention))
                self.assertIsNot(second.messages[0], diagnostics.messages[0])

    def test_static_initial_messages_are_mutation_conditioned(self):
        inputs = single_query(tensor_fixture(7, chain=True), 0)
        model = propagation_model("graph_static")
        _, injected = model(*inputs, return_diagnostics=True)
        _, reference = model(*inputs, inject_mutation=False, return_diagnostics=True)
        self.assertGreater((injected.messages[0] - reference.messages[0]).abs().max().item(), 0.1)


class LocalityTests(ModelTestCase):
    def trajectories(self, variant):
        inputs = single_query(tensor_fixture(7, chain=True), 0)
        model = propagation_model(variant)
        _, injected = model(*inputs, return_diagnostics=True)
        _, reference = model(*inputs, inject_mutation=False, return_diagnostics=True)
        return [torch.linalg.vector_norm(a - b, dim=-1)[0] for a, b in
                zip(injected.hidden_states, reference.hidden_states)]

    def test_swarm_synchronous_distance_bound_at_t0_t1_t2_and_t4(self):
        for step, response in enumerate(self.trajectories("graph_swarm")):
            self.assertLessEqual(response[step + 1:].abs().max().item(), 1e-12)
            self.assertGreater(response[step].item(), 1e-6)

    def test_static_does_not_progress_beyond_first_message_neighborhood(self):
        responses = self.trajectories("graph_static")
        self.assertGreater(responses[0][0].item(), 0)
        self.assertEqual(responses[0][1:].count_nonzero().item(), 0)
        for response in responses[1:]:
            self.assertGreater(response[0].item(), 0)
            self.assertGreater(response[1].item(), 0)
            self.assertLessEqual(response[2:].abs().max().item(), 1e-12)

    def test_self_nonmutated_trajectories_are_identical(self):
        for response in self.trajectories("mutation_self"):
            self.assertGreater(response[0].item(), 0)
            self.assertEqual(response[1:].count_nonzero().item(), 0)


class InvarianceTests(ModelTestCase):
    def test_joint_permutation_on_tie_free_stage1_topology(self):
        rng = np.random.default_rng(88)
        length = 23
        xyz = rng.normal(size=(length, 3))
        for row in range(length):
            distances = np.linalg.norm(xyz - xyz[row], axis=1)
            self.assertEqual(len(np.unique(distances)), length)
        permutation = rng.permutation(length)
        inverse = np.argsort(permutation)
        for dtype, npdtype in ((torch.float64, np.float64), (torch.float32, np.float32)):
            record = make_record("synthetic", "A" * length, rng.normal(size=(length, 384)).astype(npdtype),
                xyz.astype(npdtype), np.arange(length), np.array([f"A:{i + 1}:" for i in range(length)]),
                np.array(["A"] * length), np.array(["A"] * length))
            permuted = make_record(record.protein_id, record.sequence, record.features[permutation],
                record.ca_coordinates[permutation], record.seq_pos[permutation], record.residue_uid[permutation],
                record.residue_amino_acids[permutation], record.feature_amino_acids[permutation])
            mapped = inverse[record.edge_index]
            self.assertEqual(set(map(tuple, mapped.T)), set(map(tuple, permuted.edge_index.T)))
            edge_order = rng.permutation(record.edge_index.shape[1])
            queries = (torch.tensor([2, 19]), torch.tensor([0, 0]), torch.tensor([1, 2]))
            first = (torch.from_numpy(record.features), torch.from_numpy(record.seq_pos),
                     torch.from_numpy(record.edge_index), torch.from_numpy(record.edge_attr), *queries)
            second = (torch.from_numpy(permuted.features), torch.from_numpy(permuted.seq_pos),
                      torch.from_numpy(mapped[:, edge_order]), torch.from_numpy(record.edge_attr[edge_order]), *queries)
            for variant in VARIANTS:
                with self.subTest(dtype=dtype, variant=variant):
                    model = seeded_model(variant, dtype)
                    prediction, diagnostics = model(*first, return_diagnostics=True)
                    permuted_prediction, permuted_diagnostics = model(*second, return_diagnostics=True)
                    self.assertClose(prediction, permuted_prediction, dtype)
                    for before, after in zip(diagnostics.hidden_states, permuted_diagnostics.hidden_states):
                        self.assertClose(before[:, permutation], after, dtype)

    def test_batch_equals_concatenated_independent_predictions_and_states(self):
        for dtype in (torch.float64, torch.float32):
            inputs = tensor_fixture(dtype=dtype)
            for variant in VARIANTS:
                model = seeded_model(variant, dtype)
                prediction, diagnostics = model(*inputs, return_diagnostics=True)
                singles = [model(*single_query(inputs, query), return_diagnostics=True) for query in range(2)]
                self.assertClose(prediction, torch.cat([value[0] for value in singles]), dtype)
                for step, state in enumerate(diagnostics.hidden_states):
                    self.assertClose(state, torch.cat([value[1].hidden_states[step] for value in singles]), dtype)

    def test_changing_query_zero_cannot_change_query_one(self):
        inputs = tensor_fixture()
        changed = (*inputs[:4], torch.tensor([2, 4]), torch.tensor([4, 2]), torch.tensor([5, 3]))
        for variant in VARIANTS:
            model = seeded_model(variant)
            original, first = model(*inputs, return_diagnostics=True)
            updated, second = model(*changed, return_diagnostics=True)
            self.assertClose(original[1], updated[1])
            for before, after in zip(first.hidden_states, second.hidden_states):
                self.assertClose(before[1], after[1])


class AutogradDiagnosticsTests(ModelTestCase):
    def assert_module_gradients(self, model, module_names):
        for name in module_names:
            parameters = list(getattr(model, name).parameters())
            for parameter in parameters:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            # Module-level signal guards against a silently disconnected module;
            # individual scalar gradients may legitimately be zero.
            self.assertGreater(sum(parameter.grad.abs().sum().item() for parameter in parameters), 0, name)

    def test_swarm_finite_gradients_through_all_active_major_modules(self):
        model = seeded_model()
        prediction = model(*tensor_fixture())
        ((prediction - torch.tensor([0.5, -0.5])) ** 2).sum().backward()
        self.assert_module_gradients(model, ["projection", "mutation_mlp", "message_mlp", "attention_mlp", "gru", "readout"])

    def test_static_backpropagates_through_once_computed_messages(self):
        model = seeded_model("graph_static")
        prediction, diagnostics = model(*tensor_fixture(), return_diagnostics=True)
        diagnostics.messages[0].retain_grad()
        diagnostics.attention[0].retain_grad()
        ((prediction - torch.tensor([0.5, -0.5])) ** 2).sum().backward()
        self.assert_module_gradients(model, ["projection", "mutation_mlp", "message_mlp", "attention_mlp", "gru", "readout"])
        self.assertTrue(torch.isfinite(diagnostics.messages[0].grad).all())
        self.assertGreater(diagnostics.messages[0].grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(diagnostics.attention[0].grad).all())

    def test_self_attention_is_registered_but_has_no_gradient(self):
        model = seeded_model("mutation_self")
        model(*tensor_fixture()).sum().backward()
        self.assert_module_gradients(model, ["projection", "mutation_mlp", "message_mlp", "gru", "readout"])
        self.assertTrue(all(parameter.grad is None for parameter in model.attention_mlp.parameters()))

    def test_diagnostics_preserve_prediction_state_and_parameter_gradients(self):
        inputs = tensor_fixture()
        for variant in VARIANTS:
            model = seeded_model(variant)
            before = {key: value.clone() for key, value in model.state_dict().items()}
            normal = model(*inputs)
            normal.sum().backward()
            gradients = {name: None if value.grad is None else value.grad.clone() for name, value in model.named_parameters()}
            model.zero_grad(set_to_none=True)
            prediction, diagnostics = model(*inputs, return_diagnostics=True)
            self.assertTrue(torch.equal(normal, prediction))
            self.assertEqual(len(diagnostics.hidden_states), model.config.steps + 1)
            self.assertEqual(len(diagnostics.messages), model.config.steps)
            self.assertTrue(all(value.requires_grad for value in diagnostics.hidden_states))
            self.assertTrue(all(value.requires_grad for value in diagnostics.messages))
            prediction.sum().backward()
            for name, value in model.named_parameters():
                if gradients[name] is None:
                    self.assertIsNone(value.grad)
                else:
                    self.assertTrue(torch.equal(value.grad, gradients[name]))
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, before[key]))

    def test_labels_and_unexpected_targets_rejected(self):
        for argument in ("labels", "label", "target", "targets", "ddg"):
            with self.subTest(argument=argument), self.assertRaises(TypeError):
                seeded_model()(*tensor_fixture(), **{argument: torch.tensor([0.1, 0.2])})


class InputValidationTests(ModelTestCase):
    def test_malformed_feature_edge_and_query_shapes_fail(self):
        base = tensor_fixture()
        for index, replacement in ((0, torch.zeros(5, 383, dtype=torch.float64)),
                                   (0, torch.zeros(0, 384, dtype=torch.float64)),
                                   (1, torch.arange(4)), (2, base[2].T),
                                   (3, base[3][:, :16]), (4, torch.tensor([0])),
                                   (5, torch.tensor([[0, 2]])), (6, torch.tensor([1]))):
            inputs = list(base)
            inputs[index] = replacement
            with self.subTest(index=index, shape=replacement.shape), self.assertRaises(ValueError):
                seeded_model()(*inputs)
        empty = (*base[:4], *(torch.empty(0, dtype=torch.long) for _ in range(3)))
        with self.assertRaises(ValueError):
            seeded_model()(*empty)

    def test_invalid_immutable_mapping_and_mutation_positions_fail(self):
        base = tensor_fixture()
        for index, replacement in ((1, torch.tensor([0, 1, 1, 3, 4])),
                                   (1, torch.arange(1, 6)), (1, torch.arange(5).float()),
                                   (4, torch.tensor([-1, 4])), (4, torch.tensor([0, 5]))):
            inputs = list(base)
            inputs[index] = replacement
            with self.subTest(index=index), self.assertRaises(ValueError):
                seeded_model()(*inputs)

    def test_nonfinite_and_mismatched_float_dtypes_fail(self):
        base = tensor_fixture()
        for index in (0, 3):
            for bad in (float("nan"), float("inf"), -float("inf")):
                inputs = list(base)
                inputs[index] = inputs[index].clone()
                inputs[index][0, 0] = bad
                with self.subTest(index=index, bad=bad), self.assertRaises(ValueError):
                    seeded_model()(*inputs)
            inputs = list(base)
            inputs[index] = inputs[index].float()
            with self.assertRaises(ValueError):
                seeded_model()(*inputs)

    def test_out_of_range_self_loop_and_duplicate_endpoints_fail(self):
        base = tensor_fixture()
        for pair in ((-1, 1), (5, 1), (1, 1), tuple(base[2][:, 1].tolist())):
            inputs = list(base)
            inputs[2] = inputs[2].clone()
            inputs[2][:, 0] = torch.tensor(pair)
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                seeded_model()(*inputs)


class SerializationLearningTests(ModelTestCase):
    def test_state_dict_serialization_roundtrip_preserves_predictions(self):
        inputs = tensor_fixture()
        for variant in VARIANTS:
            model = seeded_model(variant)
            expected = model(*inputs)
            stream = io.BytesIO()
            torch.save({"configuration": model.checkpoint_configuration(), "state_dict": model.state_dict()}, stream)
            stream.seek(0)
            checkpoint = torch.load(stream, weights_only=True)
            restored = MutationGraphModel(checkpoint["configuration"]["variant"],
                                          steps=checkpoint["configuration"]["steps"]).double().eval()
            self.assertEqual(restored.checkpoint_configuration(), checkpoint["configuration"])
            restored.load_state_dict(checkpoint["state_dict"], strict=True)
            self.assertTrue(torch.equal(expected, restored(*inputs)))

    def test_tiny_synthetic_learning_substantially_reduces_loss(self):
        result = synthetic_learning_result()
        print("synthetic_learning_result=" + json.dumps(result, sort_keys=True))
        self.assertTrue(np.isfinite(result["final_mse"]))
        self.assertLess(result["final_initial_ratio"], 0.05)
        self.assertLess(result["final_mse"], 0.005)


if __name__ == "__main__":
    unittest.main()
