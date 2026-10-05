"""Stage 3: synthetic proteins, temporary CSVs and CPU optimization only."""

import copy
from dataclasses import replace
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from neuropp.data_access import DevelopmentDataAccess, MutationLabels, ProtectedLabelAccess
from neuropp.graph import MutationQuery, make_record
from neuropp.models import MutationGraphModel, VARIANTS
from neuropp.splits import file_hash, generate_split
from neuropp.training import (DevelopmentDataset, IncompatibleCheckpoint, TensorProtein,
    TrainingFailure, TrainingProtein, TrainingProtocol, accumulate_protein_gradients,
    checkpoint_payload, crossed_bootstrap, evaluate_development, is_better, load_checkpoint,
    make_optimizer, paired_models, prepare_protein, protein_order, run_experiment,
    save_checkpoint, seed_effects, state_hash, train_protein)


def split_fixture():
    original = {"train": ["1A32.pdb", *[f"train_{i:03}" for i in range(238)]],
                "val": [f"val_{i:03}" for i in range(31)], "test": [f"test_{i:03}" for i in range(28)]}
    manifest = generate_split(original, {"fixture": "synthetic"})
    return manifest, DevelopmentDataAccess(manifest)


def record_fixture(pid, sequence="ACD"):
    length = len(sequence)
    rng = np.random.default_rng(17)
    return make_record(pid, sequence, rng.normal(0, 0.05, (length, 384)).astype(np.float32),
        rng.normal(size=(length, 3)).astype(np.float64), np.arange(length, dtype=np.int64),
        np.array([f"A:{i + 1}:" for i in range(length)]), np.array(list(sequence)), np.array(list(sequence)))


def protein_fixture(pid, access, count=3, targets=None):
    record = record_fixture(pid)
    queries = tuple(MutationQuery(i % 3, i % 3, (i % 3 + 1) % 20) for i in range(count))
    targets = tuple(np.linspace(-0.4, 0.5, count)) if targets is None else tuple(targets)
    return TrainingProtein(record, MutationLabels(pid, queries, targets), access)


def dataset_fixture(train_count=2, val_count=2):
    manifest, access = split_fixture()
    # Distinct target offsets make protein order affect the optimizer trajectory.
    train = {pid: protein_fixture(pid, access, 3, np.linspace(-0.4, 0.5, 3) + 0.07 * i)
             for i, pid in enumerate(manifest["train"][:train_count])}
    val = {pid: protein_fixture(pid, access, i + 2) for i, pid in enumerate(manifest["validation"][:val_count])}
    return DevelopmentDataset(train, val, access, "a" * 64, "b" * 64)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["WT_name", "mut_type", "aa_seq", "ddG_ML"])
        writer.writeheader()
        writer.writerows(rows)


def csv_row(pid, mutation="A1C", sequence="CCD", ddg="-2.5"):
    return {"WT_name": pid, "mut_type": mutation, "aa_seq": sequence, "ddG_ML": ddg}


class CPUCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)


class DataContractTests(CPUCase):
    def setUp(self):
        self.manifest, self.access = split_fixture()
        self.pid = self.manifest["train"][0]
        self.protein = protein_fixture(self.pid, self.access)

    def test_id_mismatch_rejected(self):
        labels = replace(self.protein.labels, protein_id=self.manifest["train"][1])
        with self.assertRaisesRegex(ValueError, "ID mismatch"):
            TrainingProtein(self.protein.record, labels, self.access)

    def test_empty_queries_and_count_mismatch_rejected(self):
        for queries, values in (((), ()), (self.protein.labels.queries, (1.0,))):
            with self.subTest(count=len(values)), self.assertRaises(ValueError):
                TrainingProtein(self.protein.record, MutationLabels(self.pid, queries, values), self.access)

    def test_nonfinite_and_boolean_targets_rejected(self):
        for target in (float("nan"), float("inf"), -float("inf"), True):
            labels = replace(self.protein.labels, labels=(target, 0.0, 0.0))
            with self.subTest(target=target), self.assertRaises(ValueError):
                TrainingProtein(self.protein.record, labels, self.access)

    def test_invalid_mutation_queries_rejected(self):
        for query in (MutationQuery(-1, 0, 1), MutationQuery(3, 0, 1), MutationQuery(0, 1, 2),
                      MutationQuery(0, 0, 0), MutationQuery(0, 0, 20), MutationQuery(True, 0, 1), {}):
            with self.subTest(query=query), self.assertRaises(ValueError):
                TrainingProtein(self.protein.record, MutationLabels(self.pid, (query,), (1.0,)), self.access)

    def test_protected_training_object_rejected(self):
        for split in ("head_holdout", "legacy_test"):
            with self.subTest(split=split), self.assertRaises(ProtectedLabelAccess):
                protein_fixture(self.manifest[split][0], self.access)

    def test_tensor_preparation_dtypes_and_frozen_feature_copy(self):
        tensor = prepare_protein(self.protein)
        for value in (tensor.features, tensor.edge_attr, tensor.targets):
            self.assertEqual(value.dtype, torch.float32)
            self.assertFalse(value.requires_grad)
        for value in (tensor.seq_pos, tensor.edge_index, tensor.mutation_position, tensor.wt_index, tensor.mut_index):
            self.assertEqual(value.dtype, torch.int64)
        tensor.features[0, 0] += 1
        self.assertNotEqual(float(tensor.features[0, 0]), self.protein.record.features[0, 0])
        self.assertEqual(len(tensor.forward_inputs()), 7)
        self.assertTrue(all(value is not tensor.targets for value in tensor.forward_inputs()))

    def test_float32_target_overflow_rejected(self):
        labels = replace(self.protein.labels, labels=(1e100, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "overflow"):
            prepare_protein(TrainingProtein(self.protein.record, labels, self.access))

    def test_dataset_rejects_wrong_split_and_mapping_identity(self):
        dataset = dataset_fixture()
        pid = next(iter(dataset.train))
        for train in ({"incorrect": dataset.train[pid]}, dataset.validation):
            with self.assertRaises((KeyError, ValueError)):
                replace(dataset, train=train)

    def test_dataset_from_split_file_authenticates_sha256(self):
        dataset = dataset_fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic_split.json"
            path.write_text(json.dumps(self.manifest))
            loaded = DevelopmentDataset.from_split_file(dataset.train, dataset.validation, path, "b" * 64)
            self.assertEqual(loaded.split_manifest_sha256, file_hash(path))


class CSVLoaderTests(CPUCase):
    def setUp(self):
        self.manifest, self.access = split_fixture()
        self.pid = self.manifest["train"][0]
        self.record = record_fixture(self.pid)
        self.records = {self.pid: self.record}

    def test_entire_mixed_request_preflighted_before_source_open(self):
        for split in ("head_holdout", "legacy_test"):
            with self.subTest(split=split), mock.patch.object(Path, "open") as source:
                with self.assertRaises(ProtectedLabelAccess):
                    self.access.load_megascale_csv([self.pid, self.manifest[split][0]], "never.csv", self.records)
                source.assert_not_called()

    def test_unknown_request_preflighted_before_source_open(self):
        with mock.patch.object(Path, "open") as source, self.assertRaises(KeyError):
            self.access.load_megascale_csv([self.pid, "unknown"], "never.csv", self.records)
        source.assert_not_called()

    def test_pilot_filters_sign_duplicates_and_source_order(self):
        rows = [csv_row(self.pid, "wt", "ACD", "not a label"),
                csv_row(self.pid, "A1C", "CCD", "-2.5"), csv_row(self.pid, "C2D", "ADD", "3"),
                csv_row(self.pid, "A1C", "CCD", "-2.5")]
        for mutation, ddg in (("A1A", "1"), ("A0C", "1"), ("A01C", "1"), ("A1X", "1"),
                              ("A1C:D3A", "1"), ("del1", "1"), ("a1C", "1"),
                              ("A1C", "nan"), ("A1C", "inf"), ("A1C", "-inf"), ("A1C", "bad"), ("A1C", ""),
                              ("A1C", "1_0"), ("A1C", "١"), ("A1C", "\u00a01\u00a0")):
            rows.append(csv_row(self.pid, mutation, "ignored sequence", ddg))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.csv"
            write_csv(path, rows)
            loaded = self.access.load_megascale_csv([self.pid], path, self.records)
        labels = loaded[self.pid]
        self.assertEqual(labels.labels, (2.5, -3.0, 2.5))
        self.assertEqual(labels.queries, (MutationQuery(0, 0, 1), MutationQuery(1, 1, 2), MutationQuery(0, 0, 1)))

    def test_unrequested_protected_and_development_label_values_never_converted(self):
        others = [self.manifest["head_holdout"][0], self.manifest["legacy_test"][0], self.manifest["train"][1], "unknown"]
        rows = [csv_row(self.pid, "wt", "ACD"), csv_row(self.pid)]
        rows.extend(csv_row(pid, ddg="DO NOT CONVERT") for pid in others)
        real_float = float
        conversions = []
        def guarded_float(value):
            conversions.append(value)
            if value == "DO NOT CONVERT":
                raise AssertionError("Unrequested label converted")
            return real_float(value)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.csv"
            write_csv(path, rows)
            with mock.patch("neuropp.data_access.float", side_effect=guarded_float, create=True):
                loaded = self.access.load_megascale_csv([self.pid], path, self.records)
        self.assertEqual(set(loaded), {self.pid})
        self.assertEqual(conversions, ["-2.5"])

    def test_unique_wt_sequence_required_and_duplicate_identical_wt_allowed(self):
        for wt_sequences, valid in ((["ACD", "ACD"], True), ([], False), (["ACD", "AAD"], False), (["AAD"], False)):
            rows = [csv_row(self.pid)] + [csv_row(self.pid, "wt", seq) for seq in wt_sequences]
            with self.subTest(wt=wt_sequences), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "synthetic.csv"
                write_csv(path, rows)
                if valid:
                    self.access.load_megascale_csv([self.pid], path, self.records)
                else:
                    with self.assertRaisesRegex(ValueError, "unique matching WT"):
                        self.access.load_megascale_csv([self.pid], path, self.records)

    def test_accepted_row_alignment_errors_raise_instead_of_silent_filter(self):
        for mutation, sequence in (("C1D", "DCD"), ("A4C", "ACDC"), ("A1C", "CCA")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "synthetic.csv"
                write_csv(path, [csv_row(self.pid, "wt", "ACD"), csv_row(self.pid, mutation, sequence)])
                with self.assertRaises(ValueError):
                    self.access.load_megascale_csv([self.pid], path, self.records)

    def test_validation_ids_supported_and_no_requested_mutations_rejected(self):
        pid = self.manifest["validation"][0]
        records = {pid: record_fixture(pid)}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.csv"
            write_csv(path, [csv_row(pid, "wt", "ACD"), csv_row(pid)])
            loaded = self.access.load_megascale_csv([pid], path, records, claimed_split="validation")
            self.assertEqual(set(loaded), {pid})
            write_csv(path, [csv_row(pid, "wt", "ACD")])
            with self.assertRaisesRegex(ValueError, "No finite"):
                self.access.load_megascale_csv([pid], path, records)


class LinearQueryModel(torch.nn.Module):
    """Tiny real-autograd fixture with the exact seven-argument input contract."""
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.03, dtype=dtype))
        self.bias = torch.nn.Parameter(torch.tensor(-0.02, dtype=dtype))
        self.batch_sizes = []
        self.inference_flags = []

    def forward(self, features, seq_pos, edge_index, edge_attr, mutation_position, wt_index, mut_index):
        self.batch_sizes.append(len(mutation_position))
        self.inference_flags.append((not self.training, torch.is_inference_mode_enabled()))
        return self.weight * (mutation_position.to(self.weight.dtype) + 1) + self.bias


def tensor_fixture(count, dtype=torch.float64):
    positions = torch.arange(count, dtype=torch.int64) % 3
    return TensorProtein("synthetic-protein", torch.zeros(3, 384, dtype=dtype), torch.arange(3),
        torch.empty(2, 0, dtype=torch.int64), torch.empty(0, 17, dtype=dtype), positions,
        torch.zeros(count, dtype=torch.int64), torch.ones(count, dtype=torch.int64),
        torch.linspace(-0.1, 0.1, count, dtype=dtype))


class GradientTests(CPUCase):
    def test_microbatch_gradient_equals_full_mse_float64_all_required_counts(self):
        maximum = 0.0
        for count in (1, 63, 64, 65, 129):
            with self.subTest(count=count):
                full, micro = LinearQueryModel(), LinearQueryModel()
                tensor = tensor_fixture(count)
                expected_loss = (full(*tensor.forward_inputs()) - tensor.targets).square().mean()
                expected_loss.backward()
                actual_loss = accumulate_protein_gradients(micro, make_optimizer(micro), tensor,
                    variant="graph_static", seed=42, epoch=1)
                self.assertAlmostEqual(actual_loss, float(expected_loss), places=15)
                for expected, actual in zip(full.parameters(), micro.parameters()):
                    maximum = max(maximum, float((expected.grad - actual.grad).abs().max()))
                    torch.testing.assert_close(actual.grad, expected.grad, atol=1e-14, rtol=1e-12)
        print(f"synthetic_microbatch_full_gradient_max_abs_difference={maximum:.17g}")

    def test_real_frozen_head_microbatch_gradient_matches_full_float64(self):
        dataset = dataset_fixture()
        protein = next(iter(dataset.train.values()))
        protein = protein_fixture(protein.protein_id, dataset.access, 65)
        tensor = prepare_protein(protein)
        tensor = replace(tensor, features=tensor.features.double(), edge_attr=tensor.edge_attr.double(), targets=tensor.targets.double())
        for variant in ("graph_static", "graph_swarm"):
            with self.subTest(variant=variant):
                models, _ = paired_models(42, (variant,))
                full = models[variant].double()
                micro = copy.deepcopy(full)
                (full(*tensor.forward_inputs()) - tensor.targets).square().mean().backward()
                accumulate_protein_gradients(micro, make_optimizer(micro), tensor, variant=variant, seed=42, epoch=1)
                for expected, actual in zip(full.parameters(), micro.parameters()):
                    torch.testing.assert_close(actual.grad, expected.grad, atol=1e-12, rtol=1e-10)

    def test_one_optimizer_step_per_protein_all_required_counts(self):
        for count in (1, 63, 64, 65, 129):
            with self.subTest(count=count):
                model = LinearQueryModel()
                optimizer = make_optimizer(model)
                with mock.patch.object(optimizer, "step", wraps=optimizer.step) as step:
                    train_protein(model, optimizer, tensor_fixture(count), variant="graph_swarm", seed=42, epoch=1)
                self.assertEqual(step.call_count, 1)
                self.assertEqual(float(optimizer.state[model.weight]["step"]), 1)

    def test_final_incomplete_microbatch_is_included(self):
        model = LinearQueryModel()
        train_protein(model, make_optimizer(model), tensor_fixture(129), variant="graph_static", seed=42, epoch=1)
        self.assertEqual(model.batch_sizes, [64, 64, 1])

    def test_denominator_is_total_protein_count(self):
        tensor = replace(tensor_fixture(65), targets=torch.zeros(65, dtype=torch.float64))
        model = LinearQueryModel()
        with torch.no_grad():
            model.weight.zero_()
            model.bias.fill_(0.1)
        loss = accumulate_protein_gradients(model, make_optimizer(model), tensor, variant="graph_static", seed=42, epoch=1)
        self.assertAlmostEqual(loss, 0.01, places=15)
        self.assertAlmostEqual(float(model.bias.grad), 0.2, places=15)

    def test_clip_once_after_all_microbatches_and_finite_gradients(self):
        model = LinearQueryModel()
        tensor = replace(tensor_fixture(129), targets=torch.full((129,), 20.0, dtype=torch.float64))
        optimizer = make_optimizer(model)
        events = []
        full = LinearQueryModel()
        (full(*tensor.forward_inputs()) - tensor.targets).square().mean().backward()
        real_clip = torch.nn.utils.clip_grad_norm_
        def clip(parameters, limit, **kwargs):
            events.append("clip")
            self.assertEqual(model.batch_sizes, [64, 64, 1])
            self.assertEqual(limit, 1.0)
            for parameter, expected in zip(model.parameters(), full.parameters()):
                torch.testing.assert_close(parameter.grad, expected.grad)
            result = real_clip(parameters, limit, **kwargs)
            self.assertLessEqual(float(torch.linalg.vector_norm(torch.stack([p.grad for p in model.parameters()]))), 1.0)
            return result
        real_step = optimizer.step
        def step():
            events.append("step")
            return real_step()
        with mock.patch("neuropp.training.torch.nn.utils.clip_grad_norm_", side_effect=clip) as clipped, mock.patch.object(optimizer, "step", side_effect=step):
            train_protein(model, optimizer, tensor, variant="graph_swarm", seed=42, epoch=1)
        self.assertEqual(clipped.call_count, 1)
        self.assertEqual(events, ["clip", "step"])

    def assert_failure_safe(self, model, tensor):
        optimizer = make_optimizer(model)
        before = copy.deepcopy(model.state_dict())
        with mock.patch.object(optimizer, "step", wraps=optimizer.step) as step:
            with self.assertRaisesRegex(TrainingFailure, "protein=synthetic-protein variant=graph_swarm seed=47 epoch=2"):
                train_protein(model, optimizer, tensor, variant="graph_swarm", seed=47, epoch=2)
            step.assert_not_called()
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertEqual(optimizer.state, {})
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))

    def test_nonfinite_prediction_prevents_step_and_clears_accumulated_gradients(self):
        model = LinearQueryModel()
        real_forward = model.forward
        def forward(*inputs):
            prediction = real_forward(*inputs)
            return prediction if len(model.batch_sizes) == 1 else prediction * float("nan")
        with mock.patch.object(model, "forward", side_effect=forward):
            self.assert_failure_safe(model, tensor_fixture(65))

    def test_nonfinite_loss_prevents_step(self):
        tensor = replace(tensor_fixture(65, torch.float32), targets=torch.full((65,), 1e30, dtype=torch.float32))
        self.assert_failure_safe(LinearQueryModel(torch.float32), tensor)

    def test_nonfinite_gradient_prevents_step(self):
        model = LinearQueryModel()
        handle = model.weight.register_hook(lambda grad: grad * float("inf"))
        try:
            self.assert_failure_safe(model, tensor_fixture(65))
        finally:
            handle.remove()

    def test_finite_gradients_with_overflowing_global_norm_prevent_step(self):
        model = LinearQueryModel(torch.float32)
        for parameter in model.parameters():
            parameter.register_hook(lambda grad: grad * 1e30)
        self.assert_failure_safe(model, tensor_fixture(65, torch.float32))

    def test_engine_forward_never_receives_labels_and_features_remain_frozen(self):
        tensor = tensor_fixture(65)
        model = LinearQueryModel()
        real_forward = model.forward
        def guard(*args, **kwargs):
            self.assertEqual(len(args), 7)
            self.assertEqual(kwargs, {})
            self.assertTrue(all(argument is not tensor.targets for argument in args))
            return real_forward(*args)
        with mock.patch.object(model, "forward", side_effect=guard):
            train_protein(model, make_optimizer(model), tensor, variant="graph_static", seed=42, epoch=1)
        self.assertFalse(tensor.features.requires_grad)
        self.assertIsNone(tensor.features.grad)

    def test_ambient_autocast_cannot_enable_amp_in_engine(self):
        dataset = dataset_fixture(1, 1)
        tensor = prepare_protein(next(iter(dataset.train.values())))
        models, _ = paired_models(42, ("graph_static",))
        expected = models["graph_static"]
        actual = copy.deepcopy(expected)
        train_protein(expected, make_optimizer(expected), tensor, variant="graph_static", seed=42, epoch=1)
        expected_evaluation = evaluate_development(expected, dataset.validation)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            train_protein(actual, make_optimizer(actual), tensor, variant="graph_static", seed=42, epoch=1)
            actual_evaluation = evaluate_development(actual, dataset.validation)
        self.assertEqual(state_hash(actual.state_dict()), state_hash(expected.state_dict()))
        self.assertEqual(actual_evaluation, expected_evaluation)


class DeterminismTests(CPUCase):
    def test_order_independent_of_calls_variants_rng_and_mapping_permutations(self):
        ids = [f"synthetic_{i:03}" for i in range(15)]
        expected = protein_order(42, 3, dict.fromkeys(ids))
        np.random.seed(97)
        np.random.random(1000)
        torch.rand(50)
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                self.assertEqual(expected, protein_order(42, 3, dict.fromkeys(reversed(ids))))
                self.assertEqual(expected, protein_order(42, 3, ids))
        self.assertEqual(set(expected), set(ids))
        self.assertNotEqual(expected, protein_order(42, 4, ids))
        uninterrupted = [protein_order(42, epoch, ids) for epoch in range(1, 41)]
        resumed = [protein_order(42, epoch, reversed(ids)) for epoch in range(17, 41)]
        self.assertEqual(uninterrupted[16:], resumed)

    def test_all_four_paired_initial_states_are_bitwise_identical(self):
        models, digest = paired_models(42)
        reference = models["graph_static"].state_dict()
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                self.assertEqual(digest, state_hash(models[variant].state_dict()))
                for key, value in reference.items():
                    self.assertTrue(torch.equal(value, models[variant].state_dict()[key]))
        repeated, repeated_hash = paired_models(42, reversed(VARIANTS))
        self.assertEqual(repeated_hash, digest)
        _, different_hash = paired_models(43)
        self.assertNotEqual(different_hash, digest)

    def test_initialization_preserves_callers_torch_rng(self):
        state = torch.random.get_rng_state().clone()
        paired_models(42)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_canonical_initialization_is_independent_of_default_dtype(self):
        _, expected_hash = paired_models(42)
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            models, actual_hash = paired_models(42)
            self.assertEqual(torch.get_default_dtype(), torch.float64)
        finally:
            torch.set_default_dtype(previous)
        self.assertEqual(actual_hash, expected_hash)
        self.assertTrue(all(p.dtype == torch.float32 for model in models.values() for p in model.parameters()))

    def test_adam_exact_frozen_settings(self):
        model = LinearQueryModel()
        group = make_optimizer(model).param_groups[0]
        expected = {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0, "amsgrad": False}
        self.assertEqual({key: group[key] for key in expected}, expected)


class MetricTests(CPUCase):
    def test_hand_computed_macro_pooled_and_rmse_inference_microbatching(self):
        _, access = split_fixture()
        train_pid = next(pid for pid, split in access.membership.items() if split == "train")
        val_pid = next(pid for pid, split in access.membership.items() if split == "validation")
        # Constant prediction 0; one protein contributes |2| and another |0|,|4|,|6|.
        proteins = {train_pid: protein_fixture(train_pid, access, 1, (2.0,)),
                    val_pid: protein_fixture(val_pid, access, 3, (0.0, 4.0, 6.0))}
        model = LinearQueryModel(torch.float32)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
        evaluated = evaluate_development(model, proteins)
        by_pid = {row["protein_id"]: row["MAE"] for row in evaluated.per_protein}
        self.assertEqual(by_pid[train_pid], 2.0)
        self.assertEqual(by_pid[val_pid], 10 / 3)
        self.assertEqual(evaluated.metrics["protein_macro_MAE"], (2 + 10 / 3) / 2)
        self.assertEqual(evaluated.metrics["pooled_MAE"], 3.0)
        self.assertEqual(evaluated.metrics["pooled_RMSE"], np.sqrt(14))
        self.assertTrue(all(flags == (True, True) for flags in model.inference_flags))
        self.assertEqual([row["error"] for row in evaluated.predictions if row["protein_id"] == train_pid], [-2.0])
        required = {"protein_id", "mutation_position", "wt_index", "mut_index", "target", "prediction", "error", "abs_error"}
        self.assertEqual(set(evaluated.predictions[0]), required)

    def test_evaluation_uses_final_microbatch_and_never_passes_targets(self):
        dataset = dataset_fixture()
        pid = next(iter(dataset.validation))
        proteins = {pid: protein_fixture(pid, dataset.access, 129)}
        model = LinearQueryModel(torch.float32)
        evaluated = evaluate_development(model, proteins)
        self.assertEqual(model.batch_sizes, [64, 64, 1])
        self.assertEqual(len(evaluated.predictions), 129)

    def test_evaluation_revalidates_protected_objects_before_forward(self):
        dataset = dataset_fixture()
        protected = next(pid for pid, split in dataset.access.membership.items() if split == "head_holdout")
        legitimate = next(iter(dataset.train.values()))
        # Deliberately bypass the frozen constructor to emulate a tampered input.
        forged = copy.copy(legitimate)
        object.__setattr__(forged, "record", record_fixture(protected))
        object.__setattr__(forged, "labels", replace(legitimate.labels, protein_id=protected))
        model = LinearQueryModel(torch.float32)
        with self.assertRaises(ProtectedLabelAccess):
            evaluate_development(model, {protected: forged})
        self.assertEqual(model.batch_sizes, [])

    def test_evaluation_nonfinite_prediction_fails(self):
        dataset = dataset_fixture()
        model = LinearQueryModel(torch.float32)
        with mock.patch.object(model, "forward", return_value=torch.full((2,), float("nan"))):
            with self.assertRaises(TrainingFailure):
                evaluate_development(model, dataset.validation)

    def test_exact_checkpoint_metric_tie_keeps_earlier_epoch(self):
        best, best_epoch = float("inf"), None
        for epoch, value in enumerate((1.0, 0.5, 0.5, 0.7), 1):
            if is_better(value, best):
                best, best_epoch = value, epoch
        self.assertEqual((best, best_epoch), (0.5, 2))
        self.assertFalse(is_better(0.5, 0.5))
        self.assertTrue(is_better(np.nextafter(0.5, 0), 0.5))


def checkpoint_fixture():
    dataset = dataset_fixture()
    models, digest = paired_models(42, ("graph_static",))
    model = models["graph_static"]
    optimizer = make_optimizer(model)
    train_protein(model, optimizer, prepare_protein(next(iter(dataset.train.values()))), variant="graph_static", seed=42, epoch=1)
    metadata = {"variant": "graph_static", "seed": 42, "run_id": "synthetic",
                "training_protocol": TrainingProtocol(epochs=2, seeds=(42,)).as_dict(),
                "split_manifest_sha256": "a" * 64, "feature_signature": "b" * 64,
                "initialization_hash": digest, "pytorch_version": str(torch.__version__), "device_description": "cpu"}
    best = {key: value.clone() for key, value in model.state_dict().items()}
    history = [{"variant": "graph_static", "seed": 42, "epoch": 1,
                "train_protein_MSE": 0.1, "protein_macro_MAE": 0.5, "pooled_MAE": 0.5, "pooled_RMSE": 0.6}]
    payload = checkpoint_payload(model, optimizer, metadata, epoch=1, best_epoch=1,
        best_validation_macro_MAE=0.5, best_model_state=best, history=history)
    return model, optimizer, metadata, payload


class CheckpointTests(CPUCase):
    def test_atomic_roundtrip_model_and_optimizer(self):
        model, optimizer, metadata, payload = checkpoint_fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "last.pt"
            save_checkpoint(path, payload)
            restored = MutationGraphModel("graph_static")
            new_optimizer = make_optimizer(restored)
            saved = load_checkpoint(path, restored, new_optimizer, metadata)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]))
            self.assertEqual(optimizer.state_dict()["param_groups"], new_optimizer.state_dict()["param_groups"])
            for key, entry in optimizer.state_dict()["state"].items():
                for name, value in entry.items():
                    self.assertTrue(torch.equal(value, new_optimizer.state_dict()["state"][key][name]))
            self.assertEqual(saved["best_validation_macro_MAE"], 0.5)
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_atomic_failure_preserves_previous_checkpoint_and_cleans_temporary(self):
        _, _, _, payload = checkpoint_fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "last.pt"
            save_checkpoint(path, payload)
            before = path.read_bytes()
            with mock.patch("neuropp.training.os.replace", side_effect=OSError("synthetic failure")):
                with self.assertRaises(OSError):
                    save_checkpoint(path, {**payload, "epoch": 2})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_incompatible_metadata_rejected_before_state_loading(self):
        model, optimizer, metadata, payload = checkpoint_fixture()
        changes = {"variant": "graph_swarm", "split_manifest_sha256": "c" * 64,
                   "feature_signature": "c" * 64, "initialization_hash": "c" * 64,
                   "training_protocol": {**metadata["training_protocol"], "learning_rate": 0.01}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "last.pt"
            for key, value in changes.items():
                with self.subTest(key=key):
                    save_checkpoint(path, {**payload, key: value})
                    with mock.patch.object(model, "load_state_dict") as load_model, mock.patch.object(optimizer, "load_state_dict") as load_optimizer:
                        with self.assertRaises(IncompatibleCheckpoint):
                            load_checkpoint(path, model, optimizer, metadata)
                        load_model.assert_not_called()
                        load_optimizer.assert_not_called()

    def test_identical_shape_variant_and_step_config_cannot_bypass_configuration(self):
        model, optimizer, metadata, payload = checkpoint_fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "last.pt"
            for variant, steps in (("graph_swarm", 4), ("graph_static", 1), ("graph_swarm_t1", 1)):
                corrupted = copy.deepcopy(payload)
                corrupted["model_configuration"].update(variant=variant, steps=steps)
                save_checkpoint(path, corrupted)
                with self.subTest(variant=variant, steps=steps), self.assertRaisesRegex(IncompatibleCheckpoint, "configuration"):
                    load_checkpoint(path, model, optimizer, metadata)

    def test_structure_shape_dtype_and_nonfinite_state_rejected(self):
        model, optimizer, metadata, payload = checkpoint_fixture()
        key = next(iter(payload["model_state_dict"]))
        for corruption in ("missing", "shape", "dtype", "nan"):
            changed = copy.deepcopy(payload)
            state = changed["model_state_dict"]
            if corruption == "missing":
                state.pop(key)
            elif corruption == "shape":
                state[key] = state[key].flatten()
            elif corruption == "dtype":
                state[key] = state[key].double()
            else:
                state[key].flatten()[0] = float("nan")
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "last.pt"
                save_checkpoint(path, changed)
                with mock.patch.object(model, "load_state_dict") as loading, self.assertRaises(IncompatibleCheckpoint):
                    load_checkpoint(path, model, optimizer, metadata)
                loading.assert_not_called()

    def test_invalid_optimizer_rejected_before_loading_model(self):
        model, optimizer, metadata, payload = checkpoint_fixture()
        for corruption in ("missing", "lr", "shape", "nonfinite", "missing_active", "order", "negative_moment", "step"):
            changed = copy.deepcopy(payload)
            state = changed["optimizer_state_dict"]
            if corruption == "missing":
                changed["optimizer_state_dict"] = None
            elif corruption == "lr":
                state["param_groups"][0]["lr"] = 0.2
            elif corruption == "shape":
                state["state"][0]["exp_avg"] = torch.zeros(1)
            elif corruption == "nonfinite":
                state["state"][0]["exp_avg"].flatten()[0] = float("inf")
            elif corruption == "missing_active":
                state["state"].pop(0)
            elif corruption == "order":
                state["param_groups"][0]["params"].reverse()
            elif corruption == "negative_moment":
                state["state"][0]["exp_avg_sq"].flatten()[0] = -1.0
            else:
                state["state"][0]["step"].fill_(0.5)
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "last.pt"
                save_checkpoint(path, changed)
                with mock.patch.object(model, "load_state_dict") as loading, self.assertRaises(IncompatibleCheckpoint):
                    load_checkpoint(path, model, optimizer, metadata)
                loading.assert_not_called()

    def test_incomplete_and_inconsistent_checkpoint_histories_rejected(self):
        model, optimizer, metadata, payload = checkpoint_fixture()
        histories = [[], [dict(payload["history"][0], epoch=2)],
                     [dict(payload["history"][0], protein_macro_MAE=0.7)],
                     [dict(payload["history"][0], train_protein_MSE=float("nan"))]]
        for history in histories:
            with self.subTest(history=history), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "last.pt"
                save_checkpoint(path, {**payload, "history": history})
                with mock.patch.object(model, "load_state_dict") as loading, self.assertRaises(IncompatibleCheckpoint):
                    load_checkpoint(path, model, optimizer, metadata)
                loading.assert_not_called()

    def test_invalid_checkpoint_epoch_and_kind_rejected(self):
        model, optimizer, metadata, payload = checkpoint_fixture()
        for change in ({"epoch": 0}, {"epoch": 3}, {"best_epoch": 2}, {"best_validation_macro_MAE": float("nan")}, {"kind": "partial"}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "last.pt"
                save_checkpoint(path, {**payload, **change})
                with self.assertRaises(IncompatibleCheckpoint):
                    load_checkpoint(path, model, optimizer, metadata)


class ModeTests(CPUCase):
    def test_protocol_json_matches_frozen_training(self):
        path = Path(__file__).resolve().parents[2] / "experiments/graph_swarm_v1/protocol.json"
        self.assertEqual(json.loads(path.read_text())["training"], TrainingProtocol().as_dict())

    def test_full_rejects_wrong_seeds_epochs_variants_microbatch_and_hyperparameters(self):
        changes = [{"seeds": (42,)}, {"epochs": 2}, {"variants": ("graph_static", "graph_swarm")},
                   {"mutation_microbatch": 32}, {"learning_rate": 0.01}, {"betas": (0.9, 0.99)},
                   {"eps": 1e-7}, {"weight_decay": 0.1}, {"amsgrad": True}, {"optimizer": "SGD"},
                   {"gradient_clip_global_norm": 2.0}, {"dtype": "float64"}, {"scheduler": "cosine"},
                   {"early_stopping": True}, {"AMP": True}, {"epochs": 40.0}, {"amsgrad": 0}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(TrainingProtocol(), **change).validate("full")

    def test_full_rejects_wrong_protein_counts_before_any_output(self):
        dataset = dataset_fixture()
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "209"):
                run_experiment(dataset, "full", "cpu", "blocked-full", output_root=folder)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_full_correct_counts_accepted_without_training(self):
        # All proteins are synthetic; only validate the dataset, never run full.
        manifest, access = split_fixture()
        train = {pid: protein_fixture(pid, access, 1) for pid in manifest["train"]}
        val = {pid: protein_fixture(pid, access, 1) for pid in manifest["validation"]}
        dataset = DevelopmentDataset(train, val, access, "a" * 64, "b" * 64)
        dataset.validate("full")
        with self.assertRaisesRegex(ValueError, "31"):
            replace(dataset, validation={pid: val[pid] for pid in manifest["validation"][:-1]}).validate("full")

    def test_full_protected_id_rejected(self):
        dataset = dataset_fixture()
        protected = next(pid for pid, split in dataset.access.membership.items() if split == "legacy_test")
        with self.assertRaises(ProtectedLabelAccess):
            replace(dataset, train={**dataset.train, protected: next(iter(dataset.train.values()))}).validate("full")

    def test_smoke_preserves_hyperparameters_and_architecture(self):
        protocol = TrainingProtocol(epochs=2, seeds=(42,), variants=("graph_static", "graph_swarm"))
        protocol.validate("smoke")
        for change in ({"learning_rate": 0.01}, {"mutation_microbatch": 1}, {"epochs": 3},
                       {"seeds": (42, 43)}, {"variants": ("invented",)}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(protocol, **change).validate("smoke")


class OrchestrationTests(CPUCase):
    def test_smoke_outputs_combined_tables_tags_and_selected_checkpoints(self):
        dataset = dataset_fixture()
        with tempfile.TemporaryDirectory() as folder:
            result = run_experiment(dataset, "smoke", "cpu", "synthetic-smoke", output_root=folder)
            self.assertEqual(result.manifest["scientific_status"], "NON-SCIENTIFIC SMOKE")
            self.assertTrue(result.manifest["completed"])
            self.assertEqual(len(result.history), 4)
            self.assertEqual(len(result.validation_summary), 2)
            self.assertEqual(len(result.validation_per_protein), 4)
            self.assertEqual(len(result.validation_predictions), 10)
            self.assertEqual({row["scientific_status"] for row in result.history}, {"NON-SCIENTIFIC SMOKE"})
            expected = {"manifest.json", "history.csv", "validation_summary.csv", "validation_per_protein.csv",
                        "validation_predictions.csv", "validation_effects.csv", "validation_bootstrap.json"}
            self.assertEqual({path.name for path in result.results_directory.iterdir()}, expected)
            self.assertEqual({path.name for path in result.weights_directory.iterdir()},
                {"graph_static_seed42_best.pt", "graph_swarm_seed42_best.pt"})
            with (result.results_directory / "history.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 4)
            self.assertEqual(result.validation_bootstrap["interpretation"], "exploratory validation")
            hashes = []
            for path in result.weights_directory.iterdir():
                checkpoint = torch.load(path, weights_only=True)
                hashes.append(checkpoint["initialization_hash"])
                self.assertEqual(checkpoint["model_configuration"]["steps"], 4)
            self.assertEqual(len(set(hashes)), 1)

    def test_smoke_cannot_write_into_full_manifest(self):
        dataset = dataset_fixture()
        with tempfile.TemporaryDirectory() as folder:
            result = run_experiment(dataset, "smoke", "cpu", "same-id", output_root=folder)
            path = result.results_directory / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["mode"] = "full"
            manifest["scientific_status"] = "full development; exploratory validation"
            path.write_text(json.dumps(manifest))
            before = {file.name: file.read_bytes() for file in result.results_directory.iterdir()}
            with self.assertRaises(IncompatibleCheckpoint):
                run_experiment(dataset, "smoke", "cpu", "same-id", output_root=folder, resume=True)
            self.assertEqual(before, {file.name: file.read_bytes() for file in result.results_directory.iterdir()})

    def test_existing_run_requires_explicit_resume_and_pins_data_order(self):
        dataset = dataset_fixture()
        with tempfile.TemporaryDirectory() as folder:
            run_experiment(dataset, "smoke", "cpu", "synthetic", output_root=folder)
            with self.assertRaises(FileExistsError):
                run_experiment(dataset, "smoke", "cpu", "synthetic", output_root=folder)
            pid = next(iter(dataset.train))
            protein = dataset.train[pid]
            labels = replace(protein.labels, labels=tuple(reversed(protein.labels.labels)))
            changed = replace(dataset, train={**dataset.train, pid: TrainingProtein(protein.record, labels, dataset.access)})
            with self.assertRaises(IncompatibleCheckpoint):
                run_experiment(changed, "smoke", "cpu", "synthetic", output_root=folder, resume=True)

    def test_orphan_weights_directory_never_overwritten(self):
        dataset = dataset_fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights/graph_swarm_v1/orphan"
            path.mkdir(parents=True)
            sentinel = path / "owner.pt"
            sentinel.write_bytes(b"preserve")
            with self.assertRaises(IncompatibleCheckpoint):
                run_experiment(dataset, "smoke", "cpu", "orphan", output_root=folder)
            self.assertEqual(sentinel.read_bytes(), b"preserve")
            self.assertFalse((Path(folder) / "results").exists())

    def test_interrupted_mid_epoch_resume_matches_uninterrupted_bitwise_cpu(self):
        dataset = dataset_fixture(train_count=3)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            uninterrupted = run_experiment(dataset, "smoke", "cpu", "continuous", output_root=root)
            real_train = train_protein
            epoch_two_calls = 0
            def crash(model, optimizer, protein, **kwargs):
                nonlocal epoch_two_calls
                value = real_train(model, optimizer, protein, **kwargs)
                if kwargs["variant"] == "graph_static" and kwargs["epoch"] == 2:
                    epoch_two_calls += 1
                    if epoch_two_calls == 2:
                        raise RuntimeError("synthetic crash after partial epoch updates")
                return value
            with mock.patch("neuropp.training.train_protein", side_effect=crash):
                with self.assertRaisesRegex(RuntimeError, "synthetic crash"):
                    run_experiment(dataset, "smoke", "cpu", "resumed", output_root=root)
            last_path = root / "weights/graph_swarm_v1/resumed/graph_static_seed42_last.pt"
            last = torch.load(last_path, weights_only=True)
            self.assertEqual(last["epoch"], 1)
            # Global RNG drift and mapping permutations must not change resume.
            torch.rand(87)
            np.random.random(83)
            reordered = replace(dataset, train=dict(reversed(list(dataset.train.items()))))
            resumed = run_experiment(reordered, "smoke", "cpu", "resumed", output_root=root, resume=True)
            self.assertEqual(uninterrupted.history, resumed.history)
            self.assertEqual(uninterrupted.validation_summary, resumed.validation_summary)
            self.assertEqual(uninterrupted.validation_predictions, resumed.validation_predictions)
            self.assertEqual(uninterrupted.validation_bootstrap, resumed.validation_bootstrap)
            for variant in ("graph_static", "graph_swarm"):
                name = f"{variant}_seed42_best.pt"
                left = torch.load(uninterrupted.weights_directory / name, weights_only=True)["model_state_dict"]
                right = torch.load(resumed.weights_directory / name, weights_only=True)["model_state_dict"]
                self.assertEqual(state_hash(left), state_hash(right))

    def test_resume_completed_run_preserves_outputs(self):
        dataset = dataset_fixture()
        with tempfile.TemporaryDirectory() as folder:
            result = run_experiment(dataset, "smoke", "cpu", "complete", output_root=folder)
            with mock.patch("neuropp.training.train_protein", side_effect=AssertionError("Already trained")):
                resumed = run_experiment(dataset, "smoke", "cpu", "complete", output_root=folder, resume=True)
            self.assertEqual(result.history, resumed.history)
            self.assertEqual(result.validation_summary, resumed.validation_summary)
            self.assertEqual(result.validation_predictions, resumed.validation_predictions)

    def test_resume_repairs_best_after_crash_between_last_and_best_writes(self):
        dataset = dataset_fixture(1, 1)
        with tempfile.TemporaryDirectory() as folder:
            expected = run_experiment(dataset, "smoke", "cpu", "continuous", output_root=folder)
            real_save = save_checkpoint
            def interrupted_save(path, payload):
                if path.name == "graph_static_seed42_best.pt":
                    raise RuntimeError("synthetic best-write crash")
                real_save(path, payload)
            with mock.patch("neuropp.training.save_checkpoint", side_effect=interrupted_save):
                with self.assertRaisesRegex(RuntimeError, "best-write crash"):
                    run_experiment(dataset, "smoke", "cpu", "interrupted", output_root=folder)
            weights = Path(folder) / "weights/graph_swarm_v1/interrupted"
            self.assertEqual(torch.load(weights / "graph_static_seed42_last.pt", weights_only=True)["epoch"], 1)
            self.assertFalse((weights / "graph_static_seed42_best.pt").exists())
            actual = run_experiment(dataset, "smoke", "cpu", "interrupted", output_root=folder, resume=True)
            self.assertEqual(actual.history, expected.history)
            self.assertEqual(actual.validation_predictions, expected.validation_predictions)
            self.assertEqual({p.name for p in weights.iterdir()}, {"graph_static_seed42_best.pt", "graph_swarm_seed42_best.pt"})

    def test_output_write_crash_retains_last_and_resume_aggregates_without_training(self):
        dataset = dataset_fixture(1, 1)
        with tempfile.TemporaryDirectory() as folder:
            expected = run_experiment(dataset, "smoke", "cpu", "continuous", output_root=folder)
            with mock.patch("neuropp.training._write_table", side_effect=OSError("synthetic output crash")):
                with self.assertRaisesRegex(OSError, "output crash"):
                    run_experiment(dataset, "smoke", "cpu", "interrupted", output_root=folder)
            weights = Path(folder) / "weights/graph_swarm_v1/interrupted"
            self.assertEqual(len(list(weights.glob("*_last.pt"))), 2)
            for path in weights.glob("*_last.pt"):
                self.assertEqual(torch.load(path, weights_only=True)["epoch"], 2)
            with mock.patch("neuropp.training.train_protein", side_effect=AssertionError("Completed epochs must not repeat")):
                actual = run_experiment(dataset, "smoke", "cpu", "interrupted", output_root=folder, resume=True)
            self.assertEqual(actual.history, expected.history)
            self.assertEqual(actual.validation_predictions, expected.validation_predictions)
            self.assertEqual(len(actual.history), 4)
            self.assertEqual(list(weights.glob("*_last.pt")), [])

    def test_incompatible_later_variant_rejected_before_any_earlier_update(self):
        dataset = dataset_fixture(1, 1)
        with tempfile.TemporaryDirectory() as folder:
            real_save = save_checkpoint
            def interrupted_save(path, payload):
                if path.name == "graph_static_seed42_best.pt":
                    raise RuntimeError("synthetic checkpoint-pair crash")
                real_save(path, payload)
            with mock.patch("neuropp.training.save_checkpoint", side_effect=interrupted_save):
                with self.assertRaises(RuntimeError):
                    run_experiment(dataset, "smoke", "cpu", "interrupted", output_root=folder)
            weights = Path(folder) / "weights/graph_swarm_v1/interrupted"
            payload = torch.load(weights / "graph_static_seed42_last.pt", weights_only=True)
            payload.update(variant="graph_swarm", feature_signature="c" * 64)
            save_checkpoint(weights / "graph_swarm_seed42_best.pt", payload)
            before = {p.name: p.read_bytes() for p in weights.iterdir()}
            with mock.patch("neuropp.training.train_protein") as training, mock.patch("neuropp.training.save_checkpoint") as saving:
                with self.assertRaises(IncompatibleCheckpoint):
                    run_experiment(dataset, "smoke", "cpu", "interrupted", output_root=folder, resume=True)
                training.assert_not_called()
                saving.assert_not_called()
            self.assertEqual(before, {p.name: p.read_bytes() for p in weights.iterdir()})

    def test_exact_tie_selects_epoch_one_in_orchestration(self):
        dataset = dataset_fixture()
        real_evaluate = evaluate_development
        def tied(*args, **kwargs):
            result = real_evaluate(*args, **kwargs)
            return replace(result, metrics={**result.metrics, "protein_macro_MAE": 0.5})
        with tempfile.TemporaryDirectory() as folder, mock.patch("neuropp.training.evaluate_development", side_effect=tied):
            result = run_experiment(dataset, "smoke", "cpu", "ties", output_root=folder)
            self.assertTrue(all(row["best_epoch"] == 1 for row in result.validation_summary))
            for path in result.weights_directory.iterdir():
                self.assertEqual(torch.load(path, weights_only=True)["best_epoch"], 1)

    def test_all_four_smoke_variants_use_frozen_architecture(self):
        protocol = TrainingProtocol(epochs=1, seeds=(42,))
        with tempfile.TemporaryDirectory() as folder:
            result = run_experiment(dataset_fixture(1, 1), "smoke", "cpu", "four", output_root=folder, protocol=protocol)
            self.assertEqual({row["variant"] for row in result.validation_summary}, set(VARIANTS))
            for path in result.weights_directory.iterdir():
                saved = torch.load(path, weights_only=True)
                expected_steps = 1 if saved["variant"] == "graph_swarm_t1" else 4
                self.assertEqual(saved["model_configuration"]["steps"], expected_steps)


def slow_bootstrap_reference(panel, method="cartesian"):
    rng = np.random.default_rng(2026)
    seeds, proteins = panel.shape
    values = []
    for _ in range(10_000):
        seed_ids = rng.integers(seeds, size=seeds)
        protein_ids = rng.integers(proteins, size=proteins)
        if method == "cartesian":
            value = np.mean([panel[s, p] for s in seed_ids for p in protein_ids])
        elif method == "elementwise":
            value = panel[seed_ids, protein_ids].mean()
        else:
            value = panel.ravel()[rng.integers(panel.size, size=panel.size)].mean()
        values.append(value)
    return np.quantile(values, [0.025, 0.975])


class BootstrapTests(CPUCase):
    def test_crossed_bootstrap_matches_explicit_cartesian_reference(self):
        panel = np.array([[-3.0, 2.0, 7.0], [1.0, 4.0, 9.0]])
        result = crossed_bootstrap(panel)
        low, high = slow_bootstrap_reference(panel)
        self.assertEqual(result["mean_effect"], float(panel.mean()))
        self.assertEqual(result["CI95_low"], low)
        self.assertEqual(result["CI95_high"], high)
        self.assertEqual(result["includes_zero"], bool(low <= 0 <= high))
        self.assertEqual(result["positive_seed_count"], 2)
        self.assertEqual(result["replicates"], 10_000)
        self.assertEqual(result["rng_seed"], 2026)

    def test_crossed_bootstrap_distinguishes_flattened_and_elementwise_errors(self):
        # Interaction terms matter: additive row+column effects give identical
        # elementwise and Cartesian means and cannot detect the indexing bug.
        panel = np.array([[100.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]])
        result = crossed_bootstrap(panel)
        correct = np.array([result["CI95_low"], result["CI95_high"]])
        for method in ("flattened", "elementwise"):
            with self.subTest(method=method):
                incorrect = slow_bootstrap_reference(panel, method)
                self.assertGreater(float(np.max(np.abs(correct - incorrect))), 0.1)

    def test_bootstrap_constant_effect_and_zero_crossing(self):
        positive = crossed_bootstrap(np.ones((10, 4)))
        self.assertEqual((positive["mean_effect"], positive["CI95_low"], positive["CI95_high"]), (1.0, 1.0, 1.0))
        self.assertFalse(positive["includes_zero"])
        self.assertEqual(positive["positive_seed_count"], 10)
        zero = crossed_bootstrap(np.zeros((2, 3)))
        self.assertTrue(zero["includes_zero"])
        self.assertEqual(zero["positive_seed_count"], 0)

    def test_seed_effect_sign_sample_sd_and_tie_threshold(self):
        effects = np.array([2.0, -1.0, 1e-7, -1e-7, 0.0])
        result = seed_effects(effects, np.zeros(5))
        self.assertEqual(result["effects"], effects.tolist())
        self.assertEqual(result["mean_effect"], float(effects.mean()))
        self.assertEqual(result["standard_deviation"], float(effects.std(ddof=1)))
        self.assertEqual((result["positive_seed_count"], result["negative_seed_count"], result["tied_seed_count"]), (1, 1, 3))
        self.assertEqual(seed_effects([2], [1])["standard_deviation"], 0)

    def test_invalid_statistical_arrays_rejected(self):
        for panel in ([], np.zeros((0, 2)), np.zeros((2, 0)), [1, 2], [[float("nan")]], [[float("inf")]]):
            with self.subTest(panel=panel), self.assertRaises(ValueError):
                crossed_bootstrap(panel)
        for left, right in (([], []), ([1], [1, 2]), ([float("nan")], [0])):
            with self.assertRaises(ValueError):
                seed_effects(left, right)


if __name__ == "__main__":
    unittest.main()
