"""Small frozen-feature MLP pilot; imports perform no data or training I/O.

Only train/validation are accessible here. No ThermoMPNN extraction code is used.
The finite-scale packing exponent is derived from c6 and c12, not new information.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd
import torch
from torch import nn


ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
RADII = (6.0, 12.0, 18.0)
CHANNELS = ("c6", "c12", "c18", "d")
VARIANTS = ("baseline_mlp", "density_mlp", "fractal_mlp")
SPLIT_SHA256 = "310839d388e4430a817f849fb9c3f08ca563bc074706bb9b9facffbcb1aeb796"
FEATURE_SIGNATURE = "55c5ee9e7e9c264ce3a110bb22a64ba3872726d976d2905f345635b4a83bb991"
COUNTS = {"train": 209, "validation": 31, "head_holdout": 30, "legacy_test": 28}
SMOKE_TRAIN = ("1A32.pdb", "1AOY.pdb", "1E0L.pdb", "1E6H.pdb")
SMOKE_VALIDATION = ("1I6C.pdb", "1V1C.pdb")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_locked_split(path):
    require(sha256_file(path) == SPLIT_SHA256, "Locked split SHA256 mismatch")
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    seen = set()
    for key, count in COUNTS.items():
        names = split[key]
        require(len(names) == count and len(set(names)) == count, f"Invalid {key} count")
        require(seen.isdisjoint(names), "Overlapping split")
        seen.update(names)
    require("1A32.pdb" in split["train"], "1A32 must remain train")
    require("2MA4.pdb" in split["head_holdout"], "2MA4 must remain protected")
    return split


def require_development(names, split):
    names = list(names)
    allowed = set(split["train"]) | set(split["validation"])
    protected = set(split["head_holdout"]) | set(split["legacy_test"])
    require(bool(names) and len(names) == len(set(names)), "Empty or duplicate proteins")
    require(set(names) <= allowed and protected.isdisjoint(names),
            "Protected or unknown proteins: development access only")
    return names


def experiment_settings(mode="smoke", confirm_full=False):
    require(mode in {"smoke", "full"}, "RUN_MODE must be smoke or full")
    require(mode != "full" or confirm_full is True, "Full mode requires CONFIRM_FULL_RUN = True")
    return {"mode": mode, "seeds": [42] if mode == "smoke" else [42, 43, 44],
            "epochs": 2 if mode == "smoke" else 20, "learning_rate": 0.001,
            "scientific": mode == "full"}


def selected_names(split, settings):
    train = list(SMOKE_TRAIN) if settings["mode"] == "smoke" else list(split["train"])
    val = list(SMOKE_VALIDATION) if settings["mode"] == "smoke" else list(split["validation"])
    require(set(train) <= set(split["train"]) and set(val) <= set(split["validation"]),
            "Incorrect development membership")
    require_development(train + val, split)
    return train, val


def packing_descriptors(coordinates):
    """Natural log counts including self; <= radius, within one WT protein.

    d = (c12-c6)/log(2) is a finite-scale local packing exponent. It is an
    algebraic combination of log counts (linear in c6/c12), an explicit derived
    representation of nonlinear counts, not an established fractal dimension.
    """
    xyz = np.asarray(coordinates, dtype=np.float64)
    require(xyz.ndim == 2 and xyz.shape[1] == 3 and len(xyz) > 0
            and np.isfinite(xyz).all(), "Expected finite nonempty L x 3 coordinates")
    squared = np.sum((xyz[:, None, :] - xyz[None, :, :]) ** 2, axis=-1)
    # Diagonal contributes exactly the central residue, including L=1.
    counts = np.stack([(squared <= r * r).sum(axis=1) for r in RADII], axis=1)
    logs = np.log(counts)
    return np.column_stack([logs, (logs[:, 1] - logs[:, 0]) / np.log(2.0)])


def read_wt_pdb(path):
    """ProteinMPNN ordering: ascending residue number, then insertion code.

    Matches the original parser's first-atom convention and MSE->MET handling.
    Gaps, unknown residues, extra chains and incomplete N/CA/C/O are rejected,
    as required by the original pilot's sequence and validity checks.
    """
    three = "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split()
    amino = dict(zip(three, ALPHABET[:20]))
    residues, chains = {}, set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("HETATM") and line[17:20] == "MSE":
                line = line.replace("HETATM", "ATOM  ").replace("MSE", "MET")
            if not line.startswith("ATOM"):
                continue
            chain = line[21:22]
            chains.add(chain)
            key = (int(line[22:26]), line[26:27].strip())
            aa = amino.get(line[17:20], "-")
            residue = residues.setdefault(key, {"aa": aa, "atoms": {}})
            atom = line[12:16].strip()
            residue["atoms"].setdefault(atom, [float(line[i:i + 8]) for i in (30, 38, 46)])
    require(len(chains) == 1 and next(iter(chains)) in
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", "Expected one recognized chain")
    keys = sorted(residues)
    numbers = sorted({key[0] for key in keys})
    require(numbers == list(range(numbers[0], numbers[-1] + 1)), "PDB residue numbering gap")
    sequence, coordinates = [], []
    for key in keys:
        residue = residues[key]
        atoms = residue["atoms"]
        require(residue["aa"] in ALPHABET[:20] and {"N", "CA", "C", "O"} <= atoms.keys(),
                "Unknown residue or incomplete backbone")
        require(np.isfinite([atoms[a] for a in ("N", "CA", "C", "O")]).all(), "Nonfinite backbone")
        sequence.append(residue["aa"])
        coordinates.append(atoms["CA"])
    return "".join(sequence), np.asarray(coordinates, dtype=np.float64)


def pdb_path_for(directory, name):
    return Path(directory) / (name.split(".pdb")[0].replace("|", ":") + ".pdb")


def load_geometry(names, split, pdb_directory):
    """Calculate once per WT; includes all 209 train structures even in smoke."""
    result = {}
    for name in require_development(names, split):
        path = pdb_path_for(pdb_directory, name)
        if not path.is_file():
            raise FileNotFoundError(f"Missing existing WT PDB: {path}; no automatic extraction/download")
        sequence, coordinates = read_wt_pdb(path)
        result[name] = {"sequence": sequence, "geometry": packing_descriptors(coordinates),
                        "pdb_path": path, "pdb_sha256": sha256_file(path)}
    return result


def load_cached_features(name, structure, cache_directory):
    metadata = {"feature_signature": FEATURE_SIGNATURE, "name": name,
                "sequence": structure["sequence"], "pdb_sha256": structure["pdb_sha256"]}
    key = canonical_hash(metadata)
    path = Path(cache_directory) / f"{key}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing verified frozen features for {name}: {path}; extraction is disabled")
    # mmap leaves unused storages (including old baseline_scores) unread. Never
    # access baseline_scores or predictions; only the four required fields.
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    require(saved["metadata"] == metadata, f"{name}: incompatible feature metadata")
    features, valid, encoded = (saved[k] for k in ("features", "valid", "encoded_sequence"))
    length = len(structure["sequence"])
    require(features.shape == (length, 384) and features.dtype == torch.float32
            and torch.isfinite(features).all() and not features.requires_grad,
            f"{name}: invalid frozen features")
    require(valid.shape == (length,) and valid.dtype == torch.bool and valid.all(),
            f"{name}: incomplete cached coordinates")
    require(encoded.shape == (length,) and encoded.dtype == torch.long
            and ((encoded >= 0) & (encoded < len(ALPHABET))).all(), f"{name}: invalid encoded sequence")
    require("".join(ALPHABET[i] for i in encoded.tolist()) == structure["sequence"],
            f"{name}: WT/coordinate/feature alignment mismatch")
    return features, key


def load_development_labels(path, names, split):
    """Two passes: project IDs first, skip every unrequested row before labels.

    Only selected development rows reach the label DataFrame. MegaScale is a
    one-record-per-line CSV; reject multiline/blank-line variants before the
    second pass so physical skiprows indices cannot drift into protected rows.
    """
    names = require_development(names, split)  # reject before opening any file
    ids = pd.read_csv(path, usecols=["WT_name"], dtype=str, keep_default_na=False,
                      skip_blank_lines=False)["WT_name"]
    with Path(path).open("rb") as stream:
        physical_rows = sum(1 for _ in stream)
    require(physical_rows == len(ids) + 1 and ids.ne("").all(),
            "Expected one nonblank CSV record per physical line")
    keep = ids.isin(names).to_numpy()
    require(set(ids[keep]) == set(names), "Requested proteins absent from label CSV")
    # skiprows runs before label fields are parsed/materialized.
    rows = pd.read_csv(path, usecols=["WT_name", "mut_type", "aa_seq", "ddG_ML"],
                       dtype=str, keep_default_na=False,
                       skiprows=lambda line: line > 0 and not keep[line - 1])
    require(rows["WT_name"].tolist() == ids[keep].tolist(), "CSV row identity mismatch")
    return rows


def prepare_mutations(part, sequence):
    wt = part.loc[part["mut_type"].eq("wt"), "aa_seq"].unique()
    require(len(wt) == 1 and wt[0] == sequence, "Ambiguous or mismatching WT sequence")
    pieces = part["mut_type"].str.extract(r"^([ACDEFGHIKLMNPQRSTVWY])([1-9][0-9]*)([ACDEFGHIKLMNPQRSTVWY])$")
    target = -pd.to_numeric(part["ddG_ML"], errors="coerce")
    keep = pieces[0].notna() & pieces[0].ne(pieces[2]) & np.isfinite(target)
    rows = part.loc[keep, ["WT_name", "mut_type", "aa_seq"]].copy()
    rows["index0"] = pieces.loc[keep, 1].astype(int) - 1
    rows["wt_index"] = pieces.loc[keep, 0].map(ALPHABET.index)
    rows["mut_index"] = pieces.loc[keep, 2].map(ALPHABET.index)
    rows["ddg_exp"] = target.loc[keep].astype(float)
    require(len(rows) > 0, "No valid single substitutions")
    for row in rows.itertuples():
        i = row.index0
        require(0 <= i < len(sequence) and sequence[i] == ALPHABET[row.wt_index], "WT residue mismatch")
        require(row.aa_seq == sequence[:i] + ALPHABET[row.mut_index] + sequence[i + 1:],
                "Mutant sequence mismatch")
    return rows.drop(columns="aa_seq").reset_index(drop=True)


@dataclass
class Protein:
    features: torch.Tensor
    geometry: np.ndarray
    rows: pd.DataFrame
    pos: torch.Tensor
    wt: torch.Tensor
    mut: torch.Tensor
    target: torch.Tensor
    cache_key: str = "synthetic"


def load_records(names, split, structures, cache_directory, label_path):
    names = require_development(names, split)
    # Fail on missing features before reading labels. No cache writes.
    cached = {name: load_cached_features(name, structures[name], cache_directory) for name in names}
    raw = load_development_labels(label_path, names, split)
    records = {}
    for name in names:
        structure = structures[name]
        rows = prepare_mutations(raw.loc[raw["WT_name"].eq(name)].reset_index(drop=True), structure["sequence"])
        records[name] = Protein(cached[name][0], structure["geometry"], rows,
                                *[torch.tensor(rows[col].to_numpy(), dtype=torch.long)
                                  for col in ("index0", "wt_index", "mut_index")],
                                torch.tensor(rows["ddg_exp"].to_numpy(), dtype=torch.float32), cached[name][1])
    return records


def fit_normalization(structures, split):
    # Iterate the locked train list, never other entries present in structures.
    values = np.concatenate([structures[name]["geometry"] for name in split["train"]])
    require(values.ndim == 2 and values.shape[1] == 4 and np.isfinite(values).all(), "Invalid geometry")
    std = values.std(axis=0, ddof=0)
    return {"mean": values.mean(axis=0).tolist(), "std": std.tolist(),
            "scale": np.where(std > 0, std, 1.0).tolist(), "ddof": 0,
            "fitted_split": "train", "proteins": len(split["train"]), "residues": len(values)}


def model_inputs(record, variant, normalization):
    require(variant in VARIANTS, "Unknown model variant")
    geometry = (record.geometry - np.asarray(normalization["mean"])) / np.asarray(normalization["scale"])
    extra = record.features.new_zeros((len(record.features), 4))
    active = {"baseline_mlp": 0, "density_mlp": 3, "fractal_mlp": 4}[variant]
    if active:
        extra[:, :active] = torch.as_tensor(geometry[:, :active], dtype=extra.dtype, device=extra.device)
    # Mask after normalization: inactive channels remain exactly zero.
    return torch.cat([record.features, extra], dim=-1)


class MLPHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(388, 32), nn.Tanh())
        self.readout = nn.Sequential(nn.Linear(32, 256), nn.Tanh(), nn.Linear(256, 21))

    def forward(self, values):
        return self.readout(self.project(values))


def build_head(variant, seed):
    require(variant in VARIANTS, "Unknown model variant")
    # Canonical per-seed initialization independent of preceding runs/RNG use.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return MLPHead()


def mutation_predictions(scores, record):
    pos, wt, mut = [getattr(record, key).to(scores.device) for key in ("pos", "wt", "mut")]
    return scores[pos, mut] - scores[pos, wt]


def prediction_metrics(predictions):
    error = predictions["ddg_pred"] - predictions["ddg_exp"]
    require(len(error) > 0 and np.isfinite(error).all(), "Empty or nonfinite predictions")
    per_protein = error.abs().groupby(predictions["WT_name"]).mean()
    return {"macro_MAE": float(per_protein.mean()), "pooled_MAE": float(error.abs().mean()),
            "pooled_RMSE": float(np.sqrt(np.mean(error ** 2)))}


@torch.no_grad()
def evaluate(head, names, records, inputs):
    head.eval()
    device = next(head.parameters()).device
    parts = []
    for name in names:
        record = records[name]
        result = record.rows[["WT_name", "mut_type", "index0", "ddg_exp"]].copy()
        result["ddg_pred"] = mutation_predictions(head(inputs[name].to(device)), record).cpu().numpy()
        parts.append(result)
    predictions = pd.concat(parts, ignore_index=True)
    return prediction_metrics(predictions), predictions


def train_epoch(head, optimizer, order, records, inputs):
    head.train()
    losses = []
    device = next(head.parameters()).device
    for name in order:
        record = records[name]
        optimizer.zero_grad(set_to_none=True)
        prediction = mutation_predictions(head(inputs[name].to(device)), record)
        loss = nn.functional.mse_loss(prediction, record.target.to(device), reduction="mean")
        require(torch.isfinite(loss), f"Nonfinite loss: {name}")
        loss.backward()
        optimizer.step()  # exactly once per protein, regardless of mutation count
        losses.append(loss.item())
    return float(np.mean(losses))


def train_head(variant, seed, train_names, val_names, records, inputs, epochs, learning_rate=0.001):
    head = build_head(variant, seed)  # CPU only in this lightweight pilot
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    best_mae, best_epoch, best_state = float("inf"), None, None
    history = []
    for epoch in range(1, epochs + 1):
        loss = train_epoch(head, optimizer, rng.permutation(sorted(train_names)), records, inputs)
        metrics, _ = evaluate(head, val_names, records, inputs)
        history.append({"model": variant, "seed": seed, "epoch": epoch, "train_MSE": loss, **metrics})
        if metrics["macro_MAE"] < best_mae:  # exact ties preserve earlier epoch
            best_mae, best_epoch = metrics["macro_MAE"], epoch
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head.eval(), history, best_epoch


def code_identity(root):
    root = Path(root)
    notebook = json.loads((root / "notebooks/03_fractal_pilot.ipynb").read_text(encoding="utf-8"))
    # Outputs/counts and the user settings cell may change on resume. Effective
    # settings are already in the manifest; RESUME itself is an operational flag.
    notebook_source = [{"type": cell["cell_type"], "source": cell["source"]}
                       for cell in notebook["cells"] if cell.get("id") != "user-settings"]
    return {"revision": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
            "module_sha256": sha256_file(root / "src/fractal_pilot.py"),
            "notebook_source_sha256": canonical_hash(notebook_source)}


def make_manifest(root, run_id, settings, split, train_names, val_names, structures,
                  records, normalization, label_path):
    return {"protocol": "fractal_pilot_v1", "run_id": run_id, "settings": settings,
            "split_sha256": SPLIT_SHA256, "feature_signature": FEATURE_SIGNATURE,
            "code": code_identity(root), "geometry": {"radii_angstrom": list(RADII), "channels": list(CHANNELS),
                "mass": "1 + other residues with distance <= r", "exponent": "(c12-c6)/log(2)"},
            "normalization": normalization, "normalization_train_names": list(split["train"]),
            "train_names": train_names, "validation_names": val_names,
            "pdb_sha256": {name: item["pdb_sha256"] for name, item in structures.items()},
            "cache_keys": {name: record.cache_key for name, record in records.items()},
            "label_file_sha256": sha256_file(label_path),
            "models": list(VARIANTS), "parameters_per_model": 26293, "architecture": [388, 32, 256, 21],
            "device": "cpu", "threads": torch.get_num_threads(), "torch": str(torch.__version__),
            "numpy": np.__version__, "pandas": pd.__version__, "target": "-ddG_ML; kcal/mol; positive destabilizing",
            "optimizer": "Adam", "loss": "mean MSE per protein; one optimizer step per protein per epoch",
            "order": "default_rng(seed).permutation(sorted(train_names)) each epoch",
            "selection": "minimum validation protein macro MAE; exact ties keep earlier epoch",
            "primary_effect": "macro_MAE(density_mlp)-macro_MAE(fractal_mlp)",
            "secondary_effects": ["baseline_mlp-density_mlp", "baseline_mlp-fractal_mlp"]}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def open_run(root, run_id, manifest, resume=False):
    require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id)), "Unsafe RUN_ID")
    output = Path(root) / "results" / "fractal_pilot" / run_id
    weights = Path(root) / "weights" / "fractal_pilot" / run_id
    if output.exists() or weights.exists():
        require(resume, "Run directory already exists; use a new RUN_ID or compatible RESUME = True")
        require((output / "manifest.json").is_file() and weights.is_dir(), "Incomplete run directories")
        saved = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        require(saved["protocol_manifest"] == manifest, "Incompatible resume manifest; use a new RUN_ID")
    else:
        output.mkdir(parents=True)
        weights.mkdir(parents=True)
        saved = {"protocol_manifest": manifest, "completed": {}}
        atomic_json(output / "manifest.json", saved)
    return output, weights, saved


def run_experiment(root, run_id, settings, split, train_names, val_names, structures,
                   records, normalization, label_path, resume=False, device="cpu"):
    require(device == "cpu", "This pilot requires DEVICE = 'cpu'")
    expected_train, expected_val = selected_names(split, settings)
    require(train_names == expected_train and val_names == expected_val, "Incorrect run membership")
    require_development(list(records), split)
    require(set(records) == set(train_names + val_names), "Unexpected records")
    manifest = make_manifest(root, run_id, settings, split, train_names, val_names,
                             structures, records, normalization, label_path)
    output, weights, saved = open_run(root, run_id, manifest, resume)
    protocol_hash = canonical_hash(manifest)
    histories, summaries, predictions_all, protein_all = [], [], [], []
    for seed in settings["seeds"]:
        for variant in VARIANTS:
            key = f"{variant}_seed{seed}"
            path = weights / f"{key}.pt"
            inputs = {name: model_inputs(record, variant, normalization) for name, record in records.items()}
            if path.exists():
                if key in saved["completed"]:
                    require(sha256_file(path) == saved["completed"][key]["checkpoint_sha256"],
                            "Completed checkpoint checksum mismatch")
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                require(checkpoint["protocol_hash"] == protocol_hash
                        and checkpoint["model"] == variant and checkpoint["seed"] == seed,
                        "Incompatible completed checkpoint")
                head = build_head(variant, seed)
                head.load_state_dict(checkpoint["state_dict"], strict=True)
            else:
                require(key not in saved["completed"], "Completed checkpoint is missing")
                head, history, epoch = train_head(variant, seed, train_names, val_names, records, inputs,
                                                   settings["epochs"], settings["learning_rate"])
                checkpoint = {"state_dict": head.state_dict(), "model": variant, "seed": seed,
                              "best_epoch": epoch, "history": history, "protocol_hash": protocol_hash}
                temporary = path.with_suffix(".tmp")
                torch.save(checkpoint, temporary)
                temporary.replace(path)
            metrics, predictions = evaluate(head, val_names, records, inputs)
            histories.extend(checkpoint["history"])
            epoch = checkpoint["best_epoch"]
            summaries.append({"model": variant, "seed": seed, "best_epoch": epoch,
                              "trainable_parameters": sum(p.numel() for p in head.parameters()), **metrics})
            predictions = predictions.assign(model=variant, seed=seed, best_epoch=epoch)
            predictions["abs_error"] = (predictions["ddg_pred"] - predictions["ddg_exp"]).abs()
            predictions_all.append(predictions)
            protein_all.append(predictions.groupby("WT_name", as_index=False).agg(
                MAE=("abs_error", "mean"), mutations=("abs_error", "size")).assign(
                    model=variant, seed=seed, best_epoch=epoch))
            for filename, table in (("history.csv", pd.DataFrame(histories)),
                                    ("validation_summary.csv", pd.DataFrame(summaries)),
                                    ("validation_predictions.csv", pd.concat(predictions_all, ignore_index=True)),
                                    ("validation_per_protein.csv", pd.concat(protein_all, ignore_index=True))):
                temporary = output / (filename + ".tmp")
                table.to_csv(temporary, index=False)
                temporary.replace(output / filename)
            saved["completed"][key] = {"model": variant, "seed": seed, "selected_epoch": epoch,
                                       "checkpoint_sha256": sha256_file(path)}
            atomic_json(output / "manifest.json", saved)
            print(f"{key}: selected epoch {epoch}, validation macro MAE {metrics['macro_MAE']:.6f}")
    return {"history": pd.DataFrame(histories), "summary": pd.DataFrame(summaries),
            "per_protein": pd.concat(protein_all, ignore_index=True),
            "predictions": pd.concat(predictions_all, ignore_index=True)}


def paired_effects(summary):
    paired = summary.pivot(index="seed", columns="model", values="macro_MAE")
    require(set(VARIANTS) <= set(paired.columns) and not paired.isna().any().any(), "Incomplete paired runs")
    return pd.DataFrame({"D_fractal": paired["density_mlp"] - paired["fractal_mlp"],
                         "D_density": paired["baseline_mlp"] - paired["density_mlp"],
                         "D_total": paired["baseline_mlp"] - paired["fractal_mlp"]})
