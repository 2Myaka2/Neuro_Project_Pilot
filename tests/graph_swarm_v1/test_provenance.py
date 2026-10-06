"""Synthetic tensor and checkpoint-loading coverage; no real artifacts or labels."""

import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None

from neuropp.local_artifacts import load_existing_record
from neuropp.protocol import ARTIFACT_HASHES, FEATURE_CACHE_SIGNATURE, THERMOMPNN_COMMIT
from neuropp.provenance import (FeatureProvenanceError, artifact_audit,
                               compare_proteinmpnn_state_dict, load_frozen_core)
from neuropp.splits import file_hash, generate_split


@contextmanager
def synthetic_checkpoints(altered=False):
    """The core starts frozen and unchanged; only the ThermoMPNN load can alter it."""
    class Protein(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), requires_grad=False)
            self.register_buffer("counter", torch.tensor(2, dtype=torch.int64))

    class Core(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.prot_mpnn = Protein()

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        (root / "models").mkdir()
        (root / "vanilla_model_weights").mkdir()
        for name in ("config.yaml", "transfer_model.py", "protein_mpnn_utils.py"):
            (root / name).write_text("synthetic fixture only\n")
        original = Protein().state_dict()
        loaded = {"model.prot_mpnn." + key: value.clone() for key, value in original.items()}
        if altered:
            loaded["model.prot_mpnn.weight"][0, 0] += 0.125
        torch.save({"model_state_dict": original}, root / "vanilla_model_weights/v_48_020.pt")
        torch.save({"state_dict": loaded}, root / "models/thermoMPNN_default.pt")
        config = SimpleNamespace(model=SimpleNamespace(freeze_weights=True, load_pretrained=True,
                                                       num_final_layers=2))
        omega = SimpleNamespace(load=lambda _: config, to_container=lambda *args, **kwargs: {})
        modules = {
            "transfer_model": SimpleNamespace(__file__=str(root / "transfer_model.py"), TransferModel=Core),
            "protein_mpnn_utils": SimpleNamespace(__file__=str(root / "protein_mpnn_utils.py")),
        }
        with mock.patch("neuropp.provenance.artifact_audit", side_effect=lambda _: {
            "status": "pending", "checks": {}}), mock.patch(
                "neuropp.provenance.importlib.import_module", side_effect=modules.__getitem__), mock.patch.dict(
                    "sys.modules", {"omegaconf": SimpleNamespace(OmegaConf=omega)}):
            yield root


@unittest.skipIf(torch is None, "PyTorch required for synthetic tensor provenance checks")
class TensorProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.original = {
            "encoder.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            "decoder.bias": torch.tensor([0.25, -0.5], dtype=torch.float64),
            "counter": torch.tensor(2, dtype=torch.int64),
        }

    def test_exact_full_state_match_including_buffers(self):
        report = compare_proteinmpnn_state_dict(copy.deepcopy(self.original), self.original)
        self.assertTrue(report["passed"])
        self.assertTrue(report["exact_match"])
        self.assertEqual(report["compared_tensors"], 3)
        self.assertEqual(report["max_abs_difference"], 0.0)
        self.assertTrue(report["key_set_equal"] and report["shapes_equal"] and report["dtypes_equal"])

    def test_one_float32_ulp_fails_even_when_allclose_would_pass(self):
        changed = copy.deepcopy(self.original)
        before = changed["encoder.weight"][0, 0].clone()
        changed["encoder.weight"][0, 0] = torch.nextafter(before, torch.tensor(float("inf")))
        self.assertTrue(torch.allclose(changed["encoder.weight"], self.original["encoder.weight"]))
        report = compare_proteinmpnn_state_dict(changed, self.original)
        self.assertFalse(report["passed"])
        self.assertFalse(report["exact_match"])
        self.assertEqual(report["max_abs_difference"], float(changed["encoder.weight"][0, 0] - before))
        self.assertEqual(report["mismatched_tensors"][0]["key"], "encoder.weight")

    def test_altered_buffer_fails(self):
        changed = copy.deepcopy(self.original)
        changed["counter"] += 1
        report = compare_proteinmpnn_state_dict(changed, self.original)
        self.assertFalse(report["passed"])
        self.assertEqual(report["max_abs_difference"], 1.0)

    def test_missing_and_unexpected_keys_fail(self):
        changed = copy.deepcopy(self.original)
        changed["unexpected"] = changed.pop("decoder.bias")
        report = compare_proteinmpnn_state_dict(changed, self.original)
        self.assertFalse(report["passed"])
        self.assertFalse(report["key_set_equal"])
        self.assertEqual(report["missing_keys"], ["decoder.bias"])
        self.assertEqual(report["unexpected_keys"], ["unexpected"])

    def test_shape_mismatch_fails(self):
        changed = copy.deepcopy(self.original)
        changed["encoder.weight"] = changed["encoder.weight"].flatten().double()
        report = compare_proteinmpnn_state_dict(changed, self.original)
        self.assertFalse(report["passed"])
        self.assertFalse(report["shapes_equal"])
        self.assertFalse(report["dtypes_equal"])

    def test_dtype_change_fails_without_silent_casting(self):
        changed = copy.deepcopy(self.original)
        changed["encoder.weight"] = changed["encoder.weight"].double()
        report = compare_proteinmpnn_state_dict(changed, self.original)
        self.assertFalse(report["passed"])
        self.assertFalse(report["dtypes_equal"])
        self.assertEqual(report["max_abs_difference"], 0.0)

    def test_nonfinite_tensors_cannot_pass_even_if_equal(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                changed = copy.deepcopy(self.original)
                changed["encoder.weight"][0, 0] = value
                report = compare_proteinmpnn_state_dict(changed, changed)
                self.assertFalse(report["passed"])
                self.assertEqual(report["nonfinite_tensors"], ["encoder.weight"])
                self.assertIsNone(report["max_abs_difference"])
                json.dumps(report, allow_nan=False)

    def test_empty_and_non_tensor_state_cannot_pass(self):
        self.assertFalse(compare_proteinmpnn_state_dict({}, {})["passed"])
        self.assertFalse(compare_proteinmpnn_state_dict({"bad": 1}, {"bad": 1})["passed"])

    def test_runtime_audit_compares_after_thermompnn_checkpoint_loading(self):
        with synthetic_checkpoints() as root:
            core, _, report = load_frozen_core(root)
            self.assertTrue(all(not parameter.requires_grad for parameter in core.prot_mpnn.parameters()))
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["checks"]["proteinmpnn_tensor_match"]["compared_tensors"], 2)
            self.assertEqual(report["feature_signature"], FEATURE_CACHE_SIGNATURE)
            self.assertEqual(FEATURE_CACHE_SIGNATURE,
                             "55c5ee9e7e9c264ce3a110bb22a64ba3872726d976d2905f345635b4a83bb991")
            self.assertEqual(report["feature_spec"]["torch"], str(torch.__version__))

    def test_runtime_versions_preserve_existing_feature_cache_without_rewriting(self):
        with synthetic_checkpoints() as root:
            pdb_path = root / "synthetic.pdb"
            pdb_path.write_text("".join(
                f"ATOM  {i:5d}  {atom:<3} ALA A   1       0.000   0.000   0.000  1.00 90.00           C\n"
                for i, atom in enumerate(("N", "CA", "C", "O"), 1)))
            # The existing cache schema and signature, without any runtime fields.
            metadata = {"feature_signature": "55c5ee9e7e9c264ce3a110bb22a64ba3872726d976d2905f345635b4a83bb991",
                        "pdb_sha256": file_hash(pdb_path), "sequence": "A", "name": "synthetic.pdb"}
            key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
            feature_dir = root / "features"
            feature_dir.mkdir()
            cache_path = feature_dir / (key + ".pt")
            features = torch.arange(384, dtype=torch.float32).reshape(1, 384)
            torch.save({"metadata": metadata, "features": features,
                        "valid": torch.ones(1, dtype=torch.bool),
                        "encoded_sequence": torch.zeros(1, dtype=torch.int64)}, cache_path)
            before = cache_path.read_bytes()
            runtime_signatures = []
            for version in ("2.5.1+cpu", "2.6.0+cu124"):
                with self.subTest(version=version), mock.patch.object(torch, "__version__", version):
                    _, _, report = load_frozen_core(root)
                    self.assertEqual(report["feature_signature"], metadata["feature_signature"])
                    self.assertEqual(report["feature_spec"]["torch"], version)
                    digest = hashlib.sha256(json.dumps(report["feature_spec"], sort_keys=True).encode()).hexdigest()
                    self.assertEqual(report["runtime_provenance_signature"], digest)
                    runtime_signatures.append(digest)
                    record, path = load_existing_record("synthetic.pdb", pdb_path, feature_dir,
                                                        report["feature_signature"])
                    self.assertEqual(path, cache_path)
                    self.assertTrue(torch.equal(torch.from_numpy(record.features), features))
                    self.assertEqual(cache_path.read_bytes(), before)
                    self.assertEqual(list(feature_dir.iterdir()), [cache_path])
            self.assertNotEqual(*runtime_signatures)
        protocol = json.loads((Path(__file__).resolve().parents[2] /
                               "experiments/graph_swarm_v1/protocol.json").read_text())
        self.assertEqual(protocol["provenance"]["feature_cache_signature"], FEATURE_CACHE_SIGNATURE)

    def test_pinned_source_and_checkpoint_mismatches_stop_before_loading(self):
        with synthetic_checkpoints() as root:
            (root / "dataset_splits").mkdir()
            (root / "dataset_splits/mega_splits.pkl").write_bytes(b"synthetic split fixture")
            expected_hashes = {name: file_hash(root / name) for name in ARTIFACT_HASHES}
            with mock.patch("neuropp.provenance.artifact_audit", side_effect=artifact_audit), mock.patch(
                    "neuropp.provenance.ARTIFACT_HASHES", expected_hashes), mock.patch(
                    "neuropp.provenance.subprocess.check_output", side_effect=[THERMOMPNN_COMMIT, ""] * 3):
                self.assertEqual(artifact_audit(root)["status"], "pending")
                for name in ("models/thermoMPNN_default.pt", "vanilla_model_weights/v_48_020.pt"):
                    with self.subTest(checkpoint=name):
                        path = root / name
                        before = path.read_bytes()
                        path.write_bytes(before + b"altered")
                        with mock.patch.object(torch, "load") as loader:
                            with self.assertRaisesRegex(ValueError, "pinned artifact mismatch"):
                                load_frozen_core(root)
                            loader.assert_not_called()
                        path.write_bytes(before)
                for commit, dirty in (("unverified-source", ""), (THERMOMPNN_COMMIT, " M transfer_model.py")):
                    with self.subTest(commit=commit, dirty=dirty), mock.patch(
                            "neuropp.provenance.subprocess.check_output", side_effect=[commit, dirty]):
                        with mock.patch.object(torch, "load") as loader:
                            with self.assertRaisesRegex(ValueError, "pinned artifact mismatch"):
                                load_frozen_core(root)
                            loader.assert_not_called()

    def test_frozen_core_with_altered_loaded_weights_fails_provenance(self):
        with synthetic_checkpoints(altered=True) as root:
            with self.assertRaises(FeatureProvenanceError) as caught:
                load_frozen_core(root)
        report = caught.exception.report
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["checks"]["proteinmpnn_tensor_match"]["passed"])
        self.assertEqual(report["checks"]["proteinmpnn_tensor_match"]["max_abs_difference"], 0.125)

    def test_cli_persists_tensor_failure_and_stops_before_real_features(self):
        from neuropp.cli import main
        original = {"train": ["1A32.pdb", *[f"train_{i:03}" for i in range(238)]],
                    "val": [f"val_{i:03}" for i in range(31)], "test": [f"test_{i:03}" for i in range(28)]}
        for command in ("audit", "check-real"):
            with self.subTest(command=command), synthetic_checkpoints(altered=True) as root:
                output = root / "report.json"
                args = [command, "--thermompnn-dir", str(root), "--output", str(output)]
                if command == "audit":
                    args.append("--runtime")
                else:
                    manifest = generate_split(original, {})
                    split_path = root / "split.json"
                    split_path.write_text(json.dumps(manifest))
                    args += ["--split", str(split_path), "--protein-id", manifest["train"][0]]
                with mock.patch("neuropp.local_artifacts.load_existing_record") as loader:
                    self.assertEqual(main(args), 2)
                    loader.assert_not_called()
                report = json.loads(output.read_text())
                self.assertEqual(report["status"], "blocked")
                self.assertFalse(report["labels_accessed"])
                self.assertFalse(report["checks"]["proteinmpnn_tensor_match"]["exact_match"])
                self.assertEqual(report["checks"]["proteinmpnn_tensor_match"]["max_abs_difference"], 0.125)


if __name__ == "__main__":
    unittest.main()
