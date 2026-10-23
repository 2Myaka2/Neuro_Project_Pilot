"""CPU-fast synthetic tests only. No repository datasets or old results are read.

Run: python -m unittest discover -s tests -p test_fractal_pilot.py -v
"""

import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import fractal_pilot as fp


def synthetic_split():
    return {"train": list(fp.SMOKE_TRAIN), "validation": list(fp.SMOKE_VALIDATION),
            "head_holdout": ["protected.pdb"], "legacy_test": ["legacy.pdb"]}


def mutation_frame(name="synthetic.pdb"):
    return pd.DataFrame([
        [name, "wt", "ACD", ""],
        [name, "A1C", "CCD", "-0.8"],
        [name, "C2A", "AAD", "0.5"],
        [name, "D3E", "ACE", "-0.3"],
    ], columns=["WT_name", "mut_type", "aa_seq", "ddG_ML"])


def synthetic_protein(name="synthetic.pdb"):
    rows = fp.prepare_mutations(mutation_frame(name), "ACD")
    features = torch.randn(3, 384, generator=torch.Generator().manual_seed(7))
    geometry = fp.packing_descriptors([[0, 0, 0], [5, 0, 0], [14, 0, 0]])
    return fp.Protein(features, geometry, rows,
                      *[torch.tensor(rows[key].to_numpy(), dtype=torch.long)
                        for key in ("index0", "wt_index", "mut_index")],
                      torch.tensor(rows["ddg_exp"].to_numpy(), dtype=torch.float32))


def unit_normalization():
    return {"mean": [1.0, 2.0, 3.0, 4.0], "scale": [2.0, 3.0, 4.0, 5.0]}


def write_pdb(path, entries=None, missing_atom=None):
    entries = entries or [(1, "", "ALA", 0.0), (2, "", "CYS", 5.0), (3, "", "ASP", 14.0)]
    lines = []
    serial = 0
    for number, insertion, aa, x in entries:
        for atom, offset in (("N", -0.5), ("CA", 0), ("C", 0.5), ("O", 1.0)):
            if atom == missing_atom:
                continue
            serial += 1
            lines.append(f"ATOM  {serial:5d} {atom:^4s} {aa:>3s} A{number:4d}{insertion:1s}   "
                         f"{x + offset:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C\n")
    path.write_text("".join(lines), encoding="utf-8")


class GeometryTests(unittest.TestCase):
    def test_manual_counts_and_exponent(self):
        xyz = [[0, 0, 0], [6, 0, 0], [12, 0, 0], [18, 0, 0], [25, 0, 0]]
        expected = np.array([[2, 3, 4], [3, 4, 4], [3, 4, 5], [2, 4, 5], [1, 2, 3]])
        actual = fp.packing_descriptors(xyz)
        np.testing.assert_allclose(actual[:, :3], np.log(expected), atol=1e-14)
        np.testing.assert_allclose(actual[:, 3], np.log(expected[:, 1] / expected[:, 0]) / np.log(2), atol=1e-14)

    def test_translation_rotation_and_permutation(self):
        rng = np.random.default_rng(12)
        xyz = rng.normal(size=(15, 3)) * 7
        rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        before = fp.packing_descriptors(xyz)
        np.testing.assert_array_equal(before, fp.packing_descriptors(xyz @ rotation + [50, -12, 20]))
        permutation = rng.permutation(len(xyz))
        np.testing.assert_array_equal(before[permutation], fp.packing_descriptors(xyz[permutation]))

    def test_short_and_zero_contact(self):
        for xyz in ([[0, 0, 0]], [[0, 0, 0], [50, 0, 0]]):
            np.testing.assert_array_equal(fp.packing_descriptors(xyz), np.zeros((len(xyz), 4)))
        coincident = fp.packing_descriptors([[0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(coincident[:, :3], np.full((2, 3), np.log(2)))
        np.testing.assert_array_equal(coincident[:, 3], 0)

    def test_invalid_coordinates_rejected(self):
        for xyz in ([], [[float("nan"), 0, 0]], [[1, 2]]):
            with self.assertRaises(ValueError):
                fp.packing_descriptors(xyz)

    def test_train_only_residue_weighted_normalization(self):
        split = {"train": ["a", "b"]}
        structures = {"a": {"geometry": np.array([[0, 2, 7, 4]])},
                      "b": {"geometry": np.array([[2, 4, 7, 4], [4, 6, 7, 4]])},
                      "validation": {"geometry": np.full((100, 4), 1e9)}}
        normalization = fp.fit_normalization(structures, split)
        np.testing.assert_allclose(normalization["mean"], [2, 4, 7, 4])
        np.testing.assert_allclose(normalization["std"], [np.sqrt(8 / 3), np.sqrt(8 / 3), 0, 0])
        np.testing.assert_allclose(normalization["scale"][2:], [1, 1])
        structures["validation"]["geometry"] *= -100
        self.assertEqual(normalization, fp.fit_normalization(structures, split))
        self.assertEqual(normalization["residues"], 3)
        with self.assertRaises(KeyError):
            fp.fit_normalization({"a": structures["a"]}, split)


class InputTests(unittest.TestCase):
    def test_pdb_order_insertion_codes_and_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wt.pdb"
            write_pdb(path, [(2, "", "ASP", 14), (1, "A", "CYS", 5), (1, "", "ALA", 0)])
            sequence, xyz = fp.read_wt_pdb(path)
            self.assertEqual(sequence, "ACD")
            np.testing.assert_array_equal(xyz, [[0, 0, 0], [5, 0, 0], [14, 0, 0]])

    def test_missing_backbone_gap_and_extra_chain_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wt.pdb"
            write_pdb(path, missing_atom="O")
            with self.assertRaises(ValueError):
                fp.read_wt_pdb(path)
            write_pdb(path, [(1, "", "ALA", 0), (3, "", "ASP", 14)])
            with self.assertRaises(ValueError):
                fp.read_wt_pdb(path)
            write_pdb(path)
            with path.open("a") as stream:
                stream.write(path.read_text().replace(" A   ", " B   "))
            with self.assertRaises(ValueError):
                fp.read_wt_pdb(path)

    def test_cache_identity_alignment_and_missing_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "synthetic.pdb"
            write_pdb(path)
            split = {"train": [path.name], "validation": [], "head_holdout": [], "legacy_test": []}
            structure = fp.load_geometry([path.name], split, root)[path.name]
            metadata = {"feature_signature": fp.FEATURE_SIGNATURE, "name": path.name,
                        "sequence": "ACD", "pdb_sha256": fp.sha256_file(path)}
            cache_path = root / (fp.canonical_hash(metadata) + ".pt")
            record = synthetic_protein()
            saved = {"metadata": metadata, "features": record.features,
                     "valid": torch.ones(3, dtype=torch.bool), "encoded_sequence": torch.tensor([0, 1, 2]),
                     "baseline_scores": torch.full((3, 21), float("nan"))}
            torch.save(saved, cache_path)
            loaded, key = fp.load_cached_features(path.name, structure, root)
            self.assertTrue(torch.equal(loaded, record.features))
            self.assertEqual(key, fp.canonical_hash(metadata))
            for field, replacement in (("encoded_sequence", torch.tensor([1, 0, 2])),
                                       ("valid", torch.tensor([True, False, True])),
                                       ("features", torch.zeros(2, 384)),
                                       ("metadata", {**metadata, "feature_signature": "wrong"})):
                corrupt = {**saved, field: replacement}
                torch.save(corrupt, cache_path)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    fp.load_cached_features(path.name, structure, root)
            torch.save(saved, cache_path)
            changed = {**structure, "pdb_sha256": "changed"}
            with self.assertRaises(FileNotFoundError):
                fp.load_cached_features(path.name, changed, root)
            cache_path.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "extraction is disabled"):
                fp.load_cached_features(path.name, structure, root)
            self.assertFalse(cache_path.exists())

    def test_mutation_filters_and_negative_ddg_target(self):
        frame = mutation_frame()
        invalid = pd.DataFrame([["synthetic.pdb", name, "ignored", target] for name, target in
                               [("A1A", "1"), ("A0C", "1"), ("A1X", "1"), ("A1C:C2A", "1"),
                                ("A1C", "inf"), ("C2A", "bad")]], columns=frame.columns)
        rows = fp.prepare_mutations(pd.concat([frame, invalid], ignore_index=True), "ACD")
        self.assertEqual(rows["mut_type"].tolist(), ["A1C", "C2A", "D3E"])
        np.testing.assert_allclose(rows["ddg_exp"], [0.8, -0.5, 0.3])
        self.assertEqual(rows["index0"].tolist(), [0, 1, 2])

    def test_sequence_checks(self):
        for row, column, value in ((0, "aa_seq", "CCC"), (1, "mut_type", "C1A"),
                                   (1, "aa_seq", "AAA"), (1, "mut_type", "A99C")):
            frame = mutation_frame()
            frame.loc[row, column] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                fp.prepare_mutations(frame, "ACD")
        frame = pd.concat([mutation_frame(), mutation_frame().iloc[[0]].assign(aa_seq="AAA")])
        with self.assertRaises(ValueError):
            fp.prepare_mutations(frame, "ACD")

    def test_protected_rejection_before_file_access(self):
        split = synthetic_split()
        for names in (["protected.pdb"], ["legacy.pdb"], ["unknown.pdb"], []):
            with mock.patch("fractal_pilot.pd.read_csv") as read:
                with self.assertRaises(ValueError):
                    fp.load_development_labels("does-not-exist.csv", names, split)
                read.assert_not_called()
            with self.assertRaises(ValueError):
                fp.load_geometry(names, split, Path("missing"))
            with self.assertRaises(ValueError):
                fp.load_records(names, split, {}, Path("missing"), "missing.csv")

    def test_label_projection_skips_protected_rows_before_parsing(self):
        # Protected synthetic labels are deliberately invalid; the loader must
        # not even put their fields in a DataFrame used for target conversion.
        split = synthetic_split()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.csv"
            raw = pd.concat([mutation_frame("protected.pdb"), mutation_frame(fp.SMOKE_TRAIN[0]),
                             mutation_frame("legacy.pdb")], ignore_index=True)
            raw.to_csv(path, index=False)
            actual_read = pd.read_csv
            with mock.patch("fractal_pilot.pd.read_csv", wraps=actual_read) as read:
                result = fp.load_development_labels(path, [fp.SMOKE_TRAIN[0]], split)
                self.assertEqual(read.call_args_list[0].kwargs["usecols"], ["WT_name"])
                skip = read.call_args_list[1].kwargs["skiprows"]
                self.assertFalse(skip(0))
                self.assertTrue(all(skip(line) for line in list(range(1, 5)) + list(range(9, 13))))
                self.assertTrue(all(not skip(line) for line in range(5, 9)))
            self.assertEqual(set(result["WT_name"]), {fp.SMOKE_TRAIN[0]})
            self.assertEqual(fp.prepare_mutations(result, "ACD")["ddg_exp"].tolist(), [0.8, -0.5, 0.3])
            raw.loc[0, "aa_seq"] = "unexpected\nmultiline"
            raw.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "physical line"):
                fp.load_development_labels(path, [fp.SMOKE_TRAIN[0]], split)


class ModelTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_masks_and_unchanged_frozen_features(self):
        record = synthetic_protein()
        before = record.features.clone()
        normalized = (record.geometry - [1, 2, 3, 4]) / [2, 3, 4, 5]
        for variant, active in zip(fp.VARIANTS, (0, 3, 4)):
            values = fp.model_inputs(record, variant, unit_normalization())
            self.assertEqual(values.shape, (3, 388))
            self.assertTrue(torch.equal(values[:, :384], before))
            self.assertTrue(torch.equal(values[:, 384 + active:], torch.zeros(3, 4 - active)))
            np.testing.assert_allclose(values[:, 384:384 + active], normalized[:, :active], rtol=1e-6)
        self.assertTrue(torch.equal(before, record.features))

    def test_parameter_counts_architecture_and_paired_initialization(self):
        for seed in (42, 43, 44):
            heads = [fp.build_head(variant, seed) for variant in fp.VARIANTS]
            for head in heads:
                self.assertEqual(sum(p.numel() for p in head.parameters() if p.requires_grad), 26293)
                self.assertEqual([(m.in_features, m.out_features) for m in head.modules()
                                  if isinstance(m, torch.nn.Linear)], [(388, 32), (32, 256), (256, 21)])
                self.assertEqual(sum(isinstance(m, torch.nn.Tanh) for m in head.modules()), 2)
                for key, value in heads[0].state_dict().items():
                    self.assertTrue(torch.equal(value, head.state_dict()[key]))
        self.assertFalse(torch.equal(fp.build_head(fp.VARIANTS[0], 42).project[0].weight,
                                     fp.build_head(fp.VARIANTS[0], 43).project[0].weight))

    def test_mutant_minus_wt(self):
        record = synthetic_protein()
        scores = torch.zeros(3, 21)
        scores[0, 0], scores[0, 1] = 2, 5
        scores[1, 1], scores[1, 0] = -4, -2
        scores[2, 2], scores[2, 3] = 8, 1
        torch.testing.assert_close(fp.mutation_predictions(scores, record), torch.tensor([3.0, 2.0, -7.0]))

    def test_one_step_per_protein_and_mean_mse(self):
        records = {"a": synthetic_protein("a"), "b": synthetic_protein("b")}
        # Distinct mutation counts ensure protein steps are not mutation steps.
        records["b"].pos = records["b"].pos[:1]
        records["b"].wt = records["b"].wt[:1]
        records["b"].mut = records["b"].mut[:1]
        records["b"].target = records["b"].target[:1]
        inputs = {key: fp.model_inputs(record, fp.VARIANTS[0], unit_normalization()) for key, record in records.items()}
        head = fp.build_head(fp.VARIANTS[0], 42)
        reference = copy.deepcopy(head)
        manual_optimizer = torch.optim.Adam(reference.parameters(), lr=0.001)
        manual_losses = []
        for name in ("b", "a"):
            manual_optimizer.zero_grad(set_to_none=True)
            prediction = fp.mutation_predictions(reference(inputs[name]), records[name])
            loss = ((prediction - records[name].target) ** 2).sum() / len(prediction)
            manual_losses.append(loss.item())
            loss.backward()
            manual_optimizer.step()
        optimizer = torch.optim.Adam(head.parameters(), lr=0.001)
        with mock.patch.object(optimizer, "step", wraps=optimizer.step) as step:
            loss = fp.train_epoch(head, optimizer, ["b", "a"], records, inputs)
            self.assertEqual(step.call_count, 2)
        self.assertAlmostEqual(loss, np.mean(manual_losses), places=6)
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(value, head.state_dict()[key])

    def test_macro_and_pooled_metrics(self):
        predictions = pd.DataFrame({"WT_name": ["a", "b", "b", "b"],
                                    "ddg_pred": [4.0, 1, 1, 1], "ddg_exp": [0.0, 0, 0, 0]})
        metrics = fp.prediction_metrics(predictions)
        self.assertEqual(metrics["macro_MAE"], 2.5)
        self.assertEqual(metrics["pooled_MAE"], 1.75)
        self.assertAlmostEqual(metrics["pooled_RMSE"], np.sqrt(19 / 4))

    def test_same_order_and_earlier_epoch_on_tie(self):
        records = {name: synthetic_protein(name) for name in ("a", "b", "c", "validation")}
        orders = []
        for variant in fp.VARIANTS:
            inputs = {name: fp.model_inputs(record, variant, unit_normalization()) for name, record in records.items()}
            with mock.patch("fractal_pilot.train_epoch", return_value=0.0) as epoch, \
                    mock.patch("fractal_pilot.evaluate", return_value=({"macro_MAE": 1.0}, None)):
                _, history, selected = fp.train_head(variant, 42, ["c", "a", "b"], ["validation"],
                                                     records, inputs, 3)
            orders.append([call.args[2].tolist() for call in epoch.call_args_list])
            self.assertEqual(selected, 1)
            self.assertEqual(len(history), 3)
        self.assertEqual(orders[0], orders[1])
        self.assertEqual(orders[0], orders[2])

    def test_synthetic_tiny_data_trainability(self):
        record = synthetic_protein()
        records = {"tiny": record}
        for variant in fp.VARIANTS:
            inputs = {"tiny": fp.model_inputs(record, variant, unit_normalization())}
            head = fp.build_head(variant, 42)
            optimizer = torch.optim.Adam(head.parameters(), lr=0.001)
            with torch.no_grad():
                initial = ((fp.mutation_predictions(head(inputs["tiny"]), record) - record.target) ** 2).mean().item()
            for _ in range(35):
                fp.train_epoch(head, optimizer, ["tiny"], records, inputs)
            with torch.no_grad():
                final = ((fp.mutation_predictions(head(inputs["tiny"]), record) - record.target) ** 2).mean().item()
            self.assertLess(final, initial * 0.05, variant)
            self.assertFalse(record.features.requires_grad)
            self.assertIsNone(record.features.grad)

    def test_paired_effect_direction(self):
        summary = pd.DataFrame([{"model": variant, "seed": seed, "macro_MAE": value}
                                for seed in (42, 43) for variant, value in zip(fp.VARIANTS, [3, 2, 1])])
        effects = fp.paired_effects(summary)
        np.testing.assert_array_equal(effects, [[1, 1, 2], [1, 1, 2]])


class ProtocolAndArtifactTests(unittest.TestCase):
    def test_smoke_full_settings_and_full_confirmation(self):
        split = synthetic_split()
        smoke = fp.experiment_settings()
        self.assertEqual(smoke["seeds"], [42])
        self.assertEqual(smoke["epochs"], 2)
        self.assertFalse(smoke["scientific"])
        self.assertEqual(tuple(map(len, fp.selected_names(split, smoke))), (4, 2))
        with self.assertRaises(ValueError):
            fp.experiment_settings("full")
        full = fp.experiment_settings("full", True)
        self.assertEqual(full["seeds"], [42, 43, 44])
        self.assertEqual(full["epochs"], 20)
        with self.assertRaises(ValueError):
            fp.experiment_settings("invalid")

    def test_locked_split_tampering_rejected(self):
        # Synthetic invalid bytes; no labels are used by split validation.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps(synthetic_split()))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                fp.load_locked_split(path)

    def test_run_protection_and_manifest_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = {"identity": "synthetic"}
            output, weights, saved = fp.open_run(directory, "tiny", manifest)
            with self.assertRaises(ValueError):
                fp.open_run(directory, "tiny", manifest)
            self.assertEqual(fp.open_run(directory, "tiny", manifest, True)[2], saved)
            with self.assertRaises(ValueError):
                fp.open_run(directory, "tiny", {"identity": "changed"}, True)
            for name in ("../outside", "", "a/b"):
                with self.assertRaises(ValueError):
                    fp.open_run(directory, name, manifest)
            self.assertTrue(weights.is_dir())
            self.assertTrue((output / "manifest.json").is_file())

    def test_code_identity_allows_resume_settings_and_notebook_autosave(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "notebooks").mkdir()
            (root / "src/fractal_pilot.py").write_text("# synthetic module\n")
            path = root / "notebooks/03_fractal_pilot.ipynb"
            notebook = {"cells": [
                {"cell_type": "code", "id": "user-settings", "source": ["RESUME = False"]},
                {"cell_type": "code", "id": "work", "source": ["x = 1"], "outputs": []}]}
            path.write_text(json.dumps(notebook))
            with mock.patch("fractal_pilot.subprocess.check_output", return_value="synthetic\n"):
                before = fp.code_identity(root)
                notebook["cells"][0]["source"] = ["RESUME = True"]
                notebook["cells"][1]["outputs"] = [{"text": "autosaved output"}]
                path.write_text(json.dumps(notebook))
                self.assertEqual(before, fp.code_identity(root))
                notebook["cells"][1]["source"] = ["x = 2"]
                path.write_text(json.dumps(notebook))
                self.assertNotEqual(before, fp.code_identity(root))

    def test_synthetic_smoke_outputs_and_completed_resume(self):
        split = synthetic_split()
        settings = fp.experiment_settings()
        train, val = fp.selected_names(split, settings)
        records = {name: synthetic_protein(name) for name in train + val}
        structures = {name: {"geometry": record.geometry, "pdb_sha256": "synthetic"}
                      for name, record in records.items()}
        normalization = fp.fit_normalization(structures, split)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "synthetic.csv"
            labels.write_text("synthetic labels, not real MegaScale\n")
            with mock.patch("fractal_pilot.code_identity", return_value={"revision": "synthetic"}):
                run = fp.run_experiment(root, "tiny", settings, split, train, val,
                                        structures, records, normalization, labels)
                with mock.patch("fractal_pilot.train_head", side_effect=AssertionError("must skip completed")):
                    resumed = fp.run_experiment(root, "tiny", settings, split, train, val,
                                                structures, records, normalization, labels, resume=True)
            pd.testing.assert_frame_equal(run["summary"], resumed["summary"])
            self.assertEqual(len(run["history"]), 6)
            self.assertEqual(len(run["per_protein"]), 6)
            self.assertEqual(len(run["predictions"]), 18)
            output = root / "results/fractal_pilot/tiny"
            self.assertEqual({p.name for p in output.iterdir()}, {"manifest.json", "history.csv",
                              "validation_summary.csv", "validation_per_protein.csv", "validation_predictions.csv"})
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(len(manifest["completed"]), 3)
            self.assertEqual(manifest["protocol_manifest"]["normalization"], normalization)
            self.assertEqual(len(list((root / "weights/fractal_pilot/tiny").glob("*.pt"))), 3)
            checkpoint = root / "weights/fractal_pilot/tiny/baseline_mlp_seed42.pt"
            checkpoint.write_bytes(b"corrupt synthetic checkpoint")
            with mock.patch("fractal_pilot.code_identity", return_value={"revision": "synthetic"}):
                with self.assertRaisesRegex(ValueError, "checksum"):
                    fp.run_experiment(root, "tiny", settings, split, train, val,
                                      structures, records, normalization, labels, resume=True)

    def test_notebook_json_and_syntax_without_execution(self):
        notebook = json.loads((ROOT / "notebooks/03_fractal_pilot.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        ids = [cell["id"] for cell in notebook["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        for index, cell in enumerate(code_cells):
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            ast.parse("".join(cell["source"]), filename=f"notebook-cell-{index}")
        settings = ast.parse("".join(code_cells[0]["source"]))
        self.assertEqual({node.targets[0].id: ast.literal_eval(node.value) for node in settings.body},
                         {"RUN_MODE": "smoke", "RUN_ID": "fractal_pilot_smoke_01", "DEVICE": "cpu",
                          "RESUME": False, "CONFIRM_FULL_RUN": False})


if __name__ == "__main__":
    unittest.main()
