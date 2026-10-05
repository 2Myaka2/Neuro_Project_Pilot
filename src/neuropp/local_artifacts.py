"""Read existing WT structures and selected frozen feature caches, without labels."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .graph import make_record
from .protocol import ALPHABET
from .splits import file_hash

THREE_TO_ONE = dict(zip(
    "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(),
    ALPHABET[:-1]))


def pdb_path_for(pdb_dir, protein_id):
    stem = protein_id.split(".pdb")[0].replace("|", ":")
    if "/" in stem or "\\" in stem or stem in (".", ".."):
        raise ValueError("Invalid protein filename")
    return Path(pdb_dir) / (stem + ".pdb")


def read_wt_pdb(path):
    """Strict single-model, single-chain complete-backbone mapping; no repair."""
    residues = {}
    models = 0
    for line in Path(path).read_text().splitlines():
        if line.startswith("MODEL "):
            models += 1
            if models > 1:
                raise ValueError("Ambiguous multi-model PDB")
        if not line.startswith("ATOM  "):
            continue
        atom = line[12:16].strip()
        if atom not in {"N", "CA", "C", "O"}:
            continue
        if line[16:17] != " ":
            raise ValueError("Ambiguous alternate-location backbone atom")
        chain, number, insertion = line[21:22].strip(), int(line[22:26]), line[26:27].strip()
        if not chain:
            raise ValueError("Explicit PDB chain identifier required")
        name = line[17:20].strip()
        if name not in THREE_TO_ONE:
            raise ValueError(f"Nonstandard residue {name}")
        key = (chain, number, insertion)
        residue = residues.setdefault(key, {"aa": THREE_TO_ONE[name], "atoms": {}})
        if residue["aa"] != THREE_TO_ONE[name] or atom in residue["atoms"]:
            raise ValueError("Ambiguous PDB residue mapping or duplicate atom")
        residue["atoms"][atom] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
    if not residues or len({key[0] for key in residues}) != 1 or list(residues) != sorted(residues):
        raise ValueError("Expected an ordered, nonempty single-chain PDB")
    for residue in residues.values():
        if set(residue["atoms"]) != {"N", "CA", "C", "O"} or not np.isfinite(list(residue["atoms"].values())).all():
            raise ValueError("Missing or nonfinite PDB backbone coordinates")
    return {"sequence": "".join(residue["aa"] for residue in residues.values()),
            "ca_coordinates": np.asarray([residue["atoms"]["CA"] for residue in residues.values()], dtype=np.float64),
            "seq_pos": np.arange(len(residues), dtype=np.int64),
            "residue_uid": np.array([f"{chain}:{number}:{insertion}" for chain, number, insertion in residues]),
            "residue_amino_acids": np.array([residue["aa"] for residue in residues.values()])}


def load_existing_record(protein_id, pdb_path, feature_dir, feature_signature):
    """Whitelist the pre-head features; prior baseline scores never enter the record."""
    import torch

    wt = read_wt_pdb(pdb_path)
    metadata = {"feature_signature": feature_signature, "pdb_sha256": file_hash(pdb_path),
                "sequence": wt["sequence"], "name": protein_id}
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    feature_path = Path(feature_dir) / (key + ".pt")
    if not feature_path.is_file():
        raise FileNotFoundError(f"Missing existing frozen feature cache: {feature_path}")
    saved = torch.load(feature_path, map_location="cpu", weights_only=True)
    if saved["metadata"] != metadata:
        raise ValueError("Frozen-feature provenance metadata mismatch")
    features, valid, encoded = (saved[name] for name in ("features", "valid", "encoded_sequence"))
    length = len(wt["sequence"])
    if features.requires_grad or valid.shape != (length,) or not valid.bool().all():
        raise ValueError("Frozen features have gradients or incomplete residue mask")
    if encoded.shape != (length,) or encoded.dtype not in (torch.int32, torch.int64):
        raise ValueError("Invalid encoded WT sequence mapping")
    if not ((encoded >= 0) & (encoded < 20)).all():
        raise ValueError("Nonstandard encoded WT amino acid")
    letters = np.array([ALPHABET[int(index)] for index in encoded])
    record = make_record(protein_id=protein_id, features=features.numpy(),
                         feature_amino_acids=letters, **wt)
    return record, feature_path
