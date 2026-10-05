"""Fast synthetic CPU coverage; no real labels, weights or notebooks are read."""

import copy
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from neuropp.cache import CacheError, GraphCache, cache_metadata
from neuropp.data_access import DevelopmentDataAccess, ProtectedLabelAccess, ProteinMetadata, UnknownProteinID
from neuropp.graph import MutationQuery, build_graph, graph_diagnostics, make_record, radial_basis
from neuropp.local_artifacts import read_wt_pdb
from neuropp.protocol import COUNTS, GraphParameters
from neuropp.provenance import artifact_audit
from neuropp.splits import canonical_hash, generate_split, sequence_duplicates, validate_split, write_or_validate_split


def original_ids():
    return {"train": ["1A32.pdb", *[f"train_{i:03}" for i in range(238)]],
            "val": [f"val_{i:03}" for i in range(31)], "test": [f"test_{i:03}" for i in range(28)]}


def split_fixture():
    return generate_split(original_ids(), {"path": "synthetic IDs only", "file_sha256": "synthetic"})


def record_fixture(length=22, dtype=np.float64, coordinates=None):
    xyz = np.random.default_rng(77).normal(size=(length, 3)).astype(dtype) if coordinates is None else coordinates
    return make_record("synthetic", "A" * length, np.zeros((length, 384), dtype=dtype), xyz,
                       np.arange(length), np.array([f"A:{i+1}:" for i in range(length)]),
                       np.array(["A"] * length), np.array(["A"] * length))


def edge_set(edges):
    return set(map(tuple, edges.T.tolist()))


class SplitTests(unittest.TestCase):
    def test_order_invariance(self):
        old = original_ids()
        shuffled = {key: list(reversed(values)) for key, values in old.items()}
        self.assertEqual(generate_split(old, {}), generate_split(shuffled, {}))

    def test_exact_counts_and_sorted_storage(self):
        split = split_fixture()
        self.assertEqual({k: len(split[k]) for k in COUNTS}, COUNTS)
        self.assertTrue(all(split[key] == sorted(split[key]) for key in COUNTS))

    def test_no_overlap_and_original_union(self):
        split = split_fixture()
        self.assertEqual(len(set().union(*(set(split[k]) for k in COUNTS))), 298)
        self.assertEqual(set(split["train"] + split["head_holdout"]), set(original_ids()["train"]))
        self.assertEqual(split["validation"], original_ids()["val"])
        self.assertEqual(split["legacy_test"], original_ids()["test"])

    def test_recipe_independently(self):
        import hashlib
        old = original_ids()
        ordered = sorted((pid for pid in old["train"] if pid != "1A32.pdb"), key=lambda pid: (hashlib.sha256(
            ("graph_swarm_v1:20261004:" + pid).encode("utf-8")).hexdigest(), pid))
        self.assertEqual(split_fixture()["head_holdout"], sorted(ordered[:30]))

    def test_previously_inspected_1a32_always_stays_in_train(self):
        for seed in range(10):
            old = original_ids()
            rng = np.random.default_rng(seed)
            for values in old.values():
                rng.shuffle(values)
            split = generate_split(old, {})
            self.assertIn("1A32.pdb", split["train"])
            self.assertNotIn("1A32.pdb", split["head_holdout"])
            self.assertEqual(split["selection_method"]["ineligible_head_holdout_ids"], ["1A32.pdb"])
            self.assertEqual(split, generate_split(original_ids(), {}))
            self.assertEqual({key: len(split[key]) for key in COUNTS}, COUNTS)
            self.assertEqual(set().union(*(set(split[key]) for key in COUNTS)),
                             set().union(*(set(values) for values in old.values())))

    def test_deterministic_replacement_of_previously_selected_1a32(self):
        import hashlib
        ranked = sorted(original_ids()["train"], key=lambda pid: (
            hashlib.sha256(("graph_swarm_v1:20261004:" + pid).encode("utf-8")).hexdigest(), pid))
        self.assertIn("1A32.pdb", ranked[:30])
        revised = set(split_fixture()["head_holdout"])
        self.assertEqual(set(ranked[:30]) - revised, {"1A32.pdb"})
        self.assertEqual(revised - set(ranked[:30]), {ranked[30]})

    def test_inspected_protein_cannot_be_relocated_outside_original_train(self):
        old = original_ids()
        old["train"][0], old["val"][0] = old["val"][0], old["train"][0]
        with self.assertRaisesRegex(ValueError, "Previously inspected proteins"):
            generate_split(old, {})

    def test_validator_rejects_1a32_holdout_even_with_recomputed_list_hash(self):
        split = split_fixture()
        replacement = split["head_holdout"][0]
        split["train"].remove("1A32.pdb")
        split["train"] = sorted([*split["train"], replacement])
        split["head_holdout"] = sorted(["1A32.pdb", *split["head_holdout"][1:]])
        split["provenance"]["split_lists_sha256"] = canonical_hash({key: split[key] for key in COUNTS})
        with self.assertRaisesRegex(ValueError, "deterministic recipe"):
            validate_split(split)

    def test_existing_different_split_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "split.json"
            generated = write_or_validate_split(path, original_ids(), {})
            self.assertEqual(write_or_validate_split(path, original_ids(), {}), generated)
            generated["train"][0], generated["head_holdout"][0] = generated["head_holdout"][0], generated["train"][0]
            path.write_text(json.dumps(generated))
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                write_or_validate_split(path, original_ids(), {})
            self.assertEqual(path.read_bytes(), before)

    def test_tampered_provenance_rejected(self):
        split = split_fixture()
        split["provenance"]["original_lists_sha256"] = "incorrect"
        with self.assertRaises(ValueError):
            validate_split(split)

    def test_duplicate_sequences_reported_without_reassignment(self):
        split = split_fixture()
        sequences = {pid: pid for key in COUNTS for pid in split[key]}
        sequences[split["train"][0]] = sequences[split["head_holdout"][0]] = "ACDE"
        before = copy.deepcopy(split)
        report = sequence_duplicates(split, sequences)
        self.assertEqual(len(report), 1)
        self.assertEqual(split, before)


class AccessTests(unittest.TestCase):
    def setUp(self):
        self.manifest = split_fixture()
        self.access = DevelopmentDataAccess(self.manifest)
        self.reader = mock.Mock(return_value="synthetic train label")

    def test_allowed_label_access(self):
        for split in ("train", "validation"):
            pid = self.manifest[split][0]
            self.assertEqual(self.access.load_labels([pid], self.reader, split)[pid], "synthetic train label")

    def test_protected_labels_blocked_before_reader(self):
        for split in ("head_holdout", "legacy_test"):
            with self.assertRaises(ProtectedLabelAccess):
                self.access.load_labels([self.manifest[split][0]], self.reader, split)
        self.reader.assert_not_called()

    def test_protected_membership_cannot_be_disguised_as_train(self):
        with self.assertRaises(ProtectedLabelAccess):
            self.access.load_labels([self.manifest["head_holdout"][0]], self.reader, "train")
        self.reader.assert_not_called()

    def test_mixed_batch_is_fully_preflighted(self):
        with self.assertRaises(ProtectedLabelAccess):
            self.access.load_labels([self.manifest["train"][0], self.manifest["legacy_test"][0]], self.reader)
        self.reader.assert_not_called()

    def test_unknown_id_blocked_before_reader(self):
        with self.assertRaises(UnknownProteinID):
            self.access.load_labels(["unknown"], self.reader, "train")
        self.reader.assert_not_called()

    def test_claimed_split_mismatch_and_invalid_split(self):
        with self.assertRaises(ValueError):
            self.access.load_labels([self.manifest["validation"][0]], self.reader, "train")
        with self.assertRaises(ValueError):
            self.access.load_labels([], self.reader, "typo")
        self.reader.assert_not_called()

    def test_manifest_mutation_does_not_change_membership(self):
        pid = self.manifest["legacy_test"][0]
        self.manifest["train"].append(pid)
        with self.assertRaises(ProtectedLabelAccess):
            self.access.load_labels([pid], self.reader)

    def test_metadata_only_protected_access_has_separate_contract(self):
        pid = self.manifest["head_holdout"][0]
        metadata = ProteinMetadata(pid, "AC", ("A:1:", "A:2:"))
        self.assertEqual(self.access.metadata_only([pid], lambda _: metadata)[pid], metadata)
        with self.assertRaises(ValueError):
            self.access.metadata_only([pid], lambda _: {"protein_id": pid, "label": 0.5})

    def test_concrete_label_loader_keeps_labels_separate(self):
        import hashlib
        pid = self.manifest["train"][0]
        record = dataclasses.replace(record_fixture(2), protein_id=pid)
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / hashlib.sha256(pid.encode()).hexdigest()
            folder.mkdir()
            (folder / "metadata.json").write_text(json.dumps({"protein_id": pid}))
            (folder / "labels.json").write_text(json.dumps({"queries": [{"mutation_position": 0, "wt_index": 0, "mut_index": 1}], "labels": [0.5]}))
            loaded = self.access.load_label_directory([pid], root, {pid: record}, "train")[pid]
            self.assertEqual(loaded.labels, (0.5,))
            self.assertIsInstance(loaded.queries[0], MutationQuery)
            self.assertNotIn("label", record.__dict__)

    def test_source_identity_checked_before_concrete_label_file_opens(self):
        import hashlib
        pid = self.manifest["train"][0]
        record = dataclasses.replace(record_fixture(2), protein_id=pid)
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / hashlib.sha256(pid.encode()).hexdigest()
            folder.mkdir()
            (folder / "metadata.json").write_text(json.dumps({"protein_id": self.manifest["head_holdout"][0]}))
            # labels.json intentionally does not exist: a read would fail with FileNotFoundError.
            with self.assertRaises(ProtectedLabelAccess):
                self.access.load_label_directory([pid], root, {pid: record}, "train")


class GraphTests(unittest.TestCase):
    def test_manual_knn_with_frozen_k(self):
        # Each of 18 collinear nodes selects 16 others. Only the two extreme
        # endpoints omit one another in BOTH selections; union includes all else.
        xyz = np.column_stack((np.arange(18, dtype=np.float64), np.zeros((18, 2))))
        edges, _ = build_graph(xyz, np.arange(18))
        expected = {(j, i) for i in range(18) for j in range(18) if i != j and {i, j} != {0, 17}}
        self.assertEqual(edge_set(edges), expected)

    def test_both_directions_no_loops_or_duplicates(self):
        edges, attrs = build_graph(np.random.default_rng(3).normal(size=(30, 3)), np.arange(30))
        pairs = edge_set(edges)
        self.assertEqual(len(pairs), edges.shape[1])
        self.assertTrue(all(i != j and (i, j) in pairs for j, i in pairs))
        self.assertEqual(attrs.shape, (edges.shape[1], 17))

    def test_short_proteins_are_complete_without_loops(self):
        for length in (2, 3, 9, 16, 17):
            edges, attrs = build_graph(np.zeros((length, 3)), np.arange(length))
            self.assertEqual(edge_set(edges), {(j, i) for i in range(length) for j in range(length) if i != j})
            self.assertEqual(attrs.shape, (length * (length - 1), 17))

    def test_singleton_empty_shapes(self):
        for dtype in (np.float32, np.float64):
            edges, attrs = build_graph(np.zeros((1, 3), dtype=dtype), np.array([0]))
            self.assertEqual(edges.shape, (2, 0))
            self.assertEqual(attrs.shape, (0, 17))
            self.assertEqual(attrs.dtype, dtype)

    def test_zero_length_rejected(self):
        with self.assertRaises(ValueError):
            build_graph(np.empty((0, 3)), np.array([], dtype=int))

    def test_nonfinite_coordinates_rejected(self):
        for bad in (np.nan, np.inf, -np.inf):
            xyz = np.zeros((2, 3))
            xyz[0, 0] = bad
            with self.assertRaises(ValueError):
                build_graph(xyz, np.arange(2))

    def test_positions_cannot_be_shifted_or_duplicated(self):
        for positions in ([1, 2, 3], [0, 0, 2], [0.0, 1.0, 2.0]):
            with self.assertRaises(ValueError):
                build_graph(np.zeros((3, 3)), np.asarray(positions))

    def test_rbf_at_each_center_and_no_clipping(self):
        for dtype in (np.float32, np.float64):
            centers = np.linspace(2, 20, 16, dtype=dtype)
            np.testing.assert_array_equal(np.diag(radial_basis(centers)), np.ones(16, dtype=dtype))
            self.assertLess(radial_basis(np.array([25.0], dtype=dtype))[0, -1], radial_basis(np.array([20.0], dtype=dtype))[0, -1])

    def test_sequence_separation_formula(self):
        record = record_fixture(5)
        source, destination = record.edge_index
        expected = np.log1p(abs(record.seq_pos[source] - record.seq_pos[destination])) / np.log1p(5)
        np.testing.assert_allclose(record.edge_attr[:, 16], expected, atol=1e-8, rtol=1e-7)

    def _assert_permutation(self, coordinates, dtype):
        original = record_fixture(len(coordinates), dtype, coordinates.astype(dtype))
        permutation = np.random.default_rng(44).permutation(len(coordinates))
        permuted = make_record(original.protein_id, original.sequence, original.features[permutation],
                              original.ca_coordinates[permutation], original.seq_pos[permutation],
                              original.residue_uid[permutation], original.residue_amino_acids[permutation],
                              original.feature_amino_acids[permutation])
        remapped = permutation[permuted.edge_index]
        self.assertEqual(edge_set(original.edge_index), edge_set(remapped))
        lookup = {tuple(pair): attr for pair, attr in zip(remapped.T, permuted.edge_attr)}
        values = np.array([lookup[tuple(pair)] for pair in original.edge_index.T])
        atol, rtol = (1e-5, 1e-5) if dtype == np.float32 else (1e-8, 1e-7)
        np.testing.assert_allclose(values, original.edge_attr, atol=atol, rtol=rtol)

    def test_joint_permutation_equivariance(self):
        for dtype in (np.float32, np.float64):
            self._assert_permutation(np.random.default_rng(45).normal(size=(25, 3)), dtype)

    def test_equal_distance_ties_use_original_positions(self):
        for dtype in (np.float32, np.float64):
            self._assert_permutation(np.zeros((20, 3)), dtype)
        edges, _ = build_graph(np.zeros((20, 3)), np.arange(20))
        self.assertNotIn((18, 19), edge_set(edges))
        self.assertIn((0, 19), edge_set(edges))

    def test_rigid_translation_rotation_on_tie_free_coordinates(self):
        xyz = np.random.default_rng(99).normal(size=(24, 3))
        rotation, _ = np.linalg.qr(np.random.default_rng(101).normal(size=(3, 3)))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        for dtype in (np.float32, np.float64):
            base = xyz.astype(dtype)
            transformed = (base @ rotation.astype(dtype) + np.array([2, -3, 4], dtype=dtype)).astype(dtype)
            edges, attrs = build_graph(base, np.arange(24))
            other, other_attrs = build_graph(transformed, np.arange(24))
            np.testing.assert_array_equal(edges, other)
            atol, rtol = (1e-5, 1e-5) if dtype == np.float32 else (1e-8, 1e-7)
            np.testing.assert_allclose(attrs, other_attrs, atol=atol, rtol=rtol)

    def test_diagnostics_and_degree_can_exceed_16(self):
        report = graph_diagnostics(record_fixture(18, coordinates=np.column_stack((np.arange(18, dtype=float), np.zeros((18, 2))))))
        self.assertEqual(report["degree"]["max"], 17)
        self.assertEqual(report["connected_components"], 1)
        self.assertEqual(report["reachable_fraction"]["2"], 1.0)

    def test_graph_can_be_disconnected(self):
        cluster = np.column_stack((np.arange(17) * 0.01, np.zeros((17, 2))))
        record = record_fixture(34, coordinates=np.vstack((cluster, cluster + 100)))
        report = graph_diagnostics(record)
        self.assertEqual(report["connected_components"], 2)
        self.assertAlmostEqual(report["reachable_fraction"]["4"], 16 / 33)

    def test_singleton_diagnostics(self):
        report = graph_diagnostics(record_fixture(1))
        self.assertEqual(report["connected_components"], 1)
        self.assertEqual(report["degree"], {"min": 0, "mean": 0.0, "max": 0})
        self.assertEqual(report["edge_distance_angstrom"], {"min": None, "max": None})

    def test_different_scientific_parameters_rejected(self):
        with self.assertRaises(ValueError):
            GraphParameters(k=15)


class ContractTests(unittest.TestCase):
    def test_sequence_features_coordinates_mismatch(self):
        record = record_fixture()
        for updates in ({"sequence": record.sequence[:-1]}, {"features": record.features[:-1]},
                        {"ca_coordinates": record.ca_coordinates[:-1]}, {"residue_uid": record.residue_uid[:-1]}):
            with self.assertRaises(ValueError):
                dataclasses.replace(record, **updates).validate()

    def test_wt_feature_and_pdb_alignment(self):
        record = record_fixture()
        for field in ("feature_amino_acids", "residue_amino_acids"):
            letters = getattr(record, field).copy()
            letters[0] = "C"
            with self.assertRaises(ValueError):
                dataclasses.replace(record, **{field: letters}).validate()

    def test_ambiguous_residue_mapping_fails(self):
        record = record_fixture()
        for uids in (np.array(["A:1:"] * 22), np.array([f"B:{i+1}:" if i == 3 else f"A:{i+1}:" for i in range(22)]),
                     record.residue_uid[::-1]):
            with self.assertRaises(ValueError):
                dataclasses.replace(record, residue_uid=uids).validate()

    def test_nonfinite_features_fail(self):
        record = record_fixture()
        features = record.features.copy()
        features[0, 0] = np.nan
        with self.assertRaises(ValueError):
            dataclasses.replace(record, features=features).validate()

    def test_mutation_query_contains_no_label_and_uses_wt_position(self):
        record = record_fixture()
        self.assertEqual(MutationQuery(3, 0, 1).validate(record).mutation_position, 3)
        self.assertNotIn("label", MutationQuery.__dataclass_fields__)
        for query in (MutationQuery(22, 0, 1), MutationQuery(3, 1, 0), MutationQuery(3, 0, 0), MutationQuery(True, 0, 1)):
            with self.assertRaises(ValueError):
                query.validate(record)

    def test_graph_and_record_do_not_accept_labels(self):
        with self.assertRaises(TypeError):
            build_graph(np.zeros((1, 3)), np.array([0]), label=1)
        self.assertNotIn("label", record_fixture().__dict__)

    def test_pdb_alternate_locations_and_missing_ca_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.pdb"
            path.write_text("ATOM      1  CA AALA A   1       0.000   0.000   0.000  1.00 90.00           C\n")
            with self.assertRaises(ValueError):
                read_wt_pdb(path)
            path.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 90.00           N\n")
            with self.assertRaises(ValueError):
                read_wt_pdb(path)


class CacheTests(unittest.TestCase):
    def test_roundtrip_graph_only_cache(self):
        record = record_fixture()
        with tempfile.TemporaryDirectory() as folder:
            cache = GraphCache(folder)
            self.assertIsNone(cache.read(record))
            path = cache.write(record)
            edges, attrs = cache.read(record)
            np.testing.assert_array_equal(edges, record.edge_index)
            np.testing.assert_array_equal(attrs, record.edge_attr)
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(set(saved.files), {"manifest", "edge_index", "edge_attr"})
                self.assertNotIn("label", str(saved["manifest"].item()))

    def test_coordinates_mapping_features_and_parameters_invalidate_key(self):
        record = record_fixture()
        moved = dataclasses.replace(record, ca_coordinates=record.ca_coordinates + 1)
        self.assertNotEqual(canonical_hash(cache_metadata(record)), canonical_hash(cache_metadata(moved)))
        metadata = cache_metadata(record)
        metadata["graph_parameters"]["k"] = 15
        self.assertNotEqual(canonical_hash(cache_metadata(record)), canonical_hash(metadata))
        metadata = cache_metadata(record)
        metadata["protocol_version"] = "future"
        self.assertNotEqual(canonical_hash(cache_metadata(record)), canonical_hash(metadata))
        for updates in ({"features": record.features + 1}, {"residue_uid": np.array([f"A:{i+101}:" for i in range(22)])}):
            self.assertNotEqual(canonical_hash(cache_metadata(record)), canonical_hash(cache_metadata(dataclasses.replace(record, **updates))))

    def test_incompatible_entry_is_not_reused(self):
        record = record_fixture()
        with tempfile.TemporaryDirectory() as folder:
            cache = GraphCache(folder)
            before = cache.write(record).read_bytes()
            moved = dataclasses.replace(record, ca_coordinates=record.ca_coordinates + 1)
            self.assertIsNone(cache.read(moved))
            cache.path_for(moved).write_bytes(before)
            with self.assertRaises(CacheError):
                cache.read(moved)

    def test_corruption_raises_instead_of_silent_acceptance(self):
        with tempfile.TemporaryDirectory() as folder:
            record, cache = record_fixture(), GraphCache(folder)
            cache.write(record).write_bytes(b"corrupt")
            with self.assertRaises(CacheError):
                cache.get_or_build(record)

    def test_modified_array_checksum_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            record, cache = record_fixture(), GraphCache(folder)
            path = cache.write(record)
            with np.load(path, allow_pickle=False) as saved:
                manifest = saved["manifest"].copy()
            attrs = record.edge_attr.copy()
            attrs[0, 0] += 0.1
            np.savez(path, manifest=manifest, edge_index=record.edge_index, edge_attr=attrs)
            with self.assertRaises(CacheError):
                cache.read(record)

    def test_atomic_failure_preserves_previous_entry(self):
        with tempfile.TemporaryDirectory() as folder:
            record, cache = record_fixture(), GraphCache(folder)
            path = cache.write(record)
            before = path.read_bytes()
            with mock.patch("neuropp.cache.os.replace", side_effect=OSError("synthetic failure")):
                with self.assertRaises(OSError):
                    cache.write(record)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(len(list(Path(folder).iterdir())), 1)


class OperationalTests(unittest.TestCase):
    def test_machine_protocol_matches_scientific_constants(self):
        root = Path(__file__).resolve().parents[2]
        protocol = json.loads((root / "experiments/graph_swarm_v1/protocol.json").read_text())
        self.assertEqual(protocol["graph_parameters"], GraphParameters().as_dict())
        self.assertEqual(protocol["splits"]["counts"], COUNTS)
        self.assertEqual(protocol["splits"]["ineligible_head_holdout_ids"], ["1A32.pdb"])
        self.assertEqual(protocol["splits"]["selection_revision"], "stage1.1")
        self.assertTrue(protocol["provenance"]["proteinmpnn_tensor_comparison"]["required_for_passed_provenance"])
        self.assertEqual(protocol["data"]["feature_dimension"], 384)

    def test_cli_rejects_protected_real_check_before_artifact_loading(self):
        from neuropp.cli import main
        with tempfile.TemporaryDirectory() as folder:
            manifest = split_fixture()
            path = Path(folder) / "split.json"
            output = Path(folder) / "report.json"
            path.write_text(json.dumps(manifest))
            with mock.patch("neuropp.provenance.load_frozen_core") as loader:
                result = main(["check-real", "--split", str(path), "--protein-id", manifest["head_holdout"][0],
                               "--output", str(output)])
                self.assertEqual(result, 2)
                loader.assert_not_called()
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "blocked")
            self.assertFalse(report["labels_accessed"])

    def test_missing_artifacts_cannot_pass_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            audit = artifact_audit(folder)
            self.assertEqual(audit["status"], "blocked")
            self.assertEqual(audit["reason"], "missing artifact")
            self.assertTrue(audit["missing"])

    def test_import_has_no_pipeline_side_effects(self):
        root = Path(__file__).resolve().parents[2]
        script = '''
import sys
from pathlib import Path
from unittest.mock import patch
def forbidden(*args, **kwargs):
    raise AssertionError("pipeline side effect during import")
with patch("subprocess.Popen", forbidden), patch("urllib.request.urlopen", forbidden), patch("socket.socket", forbidden), patch.object(Path, "mkdir", forbidden):
    import neuropp
    import neuropp.protocol, neuropp.splits, neuropp.graph, neuropp.cache
    import neuropp.data_access, neuropp.provenance, neuropp.local_artifacts, neuropp.cli
assert "torch" not in sys.modules
assert "train_thermompnn" not in sys.modules
assert not list(Path.cwd().iterdir())
'''
        with tempfile.TemporaryDirectory() as folder:
            environment = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1")
            subprocess.run([sys.executable, "-c", script], cwd=folder, env=environment, check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


if __name__ == "__main__":
    unittest.main()
