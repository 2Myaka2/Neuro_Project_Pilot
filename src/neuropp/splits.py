"""Metadata-only deterministic splits; no label or result files are read."""

import hashlib
import io
import json
import pickle
from pathlib import Path

from .protocol import (ARTIFACT_HASHES, COUNTS, HEAD_HOLDOUT_INELIGIBLE_IDS,
                       ORIGINAL_COUNTS, PROTOCOL_VERSION, SELECTION_PREFIX)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _IDsOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("Split pickle must contain plain containers and strings only")


def load_original_ids(path):
    path = Path(path)
    raw = path.read_bytes()
    if path.suffix == ".json":
        original = json.loads(raw)
    elif path.suffix == ".pkl":
        # Upstream includes NumPy objects in unused cross-validation partitions.
        # Only the exact owner-pinned artifact may use the general pickle decoder.
        if hashlib.sha256(raw).hexdigest() == ARTIFACT_HASHES["dataset_splits/mega_splits.pkl"]:
            original = pickle.loads(raw)
            original = {key: list(original[key]) for key in ORIGINAL_COUNTS}
        else:
            original = _IDsOnlyUnpickler(io.BytesIO(raw)).load()
    else:
        raise ValueError("Original IDs must be a metadata-only JSON or restricted pickle")
    return validate_original(original)


def _ids(values):
    if not isinstance(values, (list, tuple)):
        raise ValueError("IDs must be a list of strings")
    if any(not isinstance(x, str) or not x or x != x.strip() for x in values):
        raise ValueError("Protein IDs must be nonempty, unaltered strings")
    if len(set(values)) != len(values):
        raise ValueError("Duplicate protein IDs")
    return sorted(values)


def validate_original(original):
    if not isinstance(original, dict):
        raise ValueError("Original split must be a dictionary")
    result = {key: _ids(original[key]) for key in ORIGINAL_COUNTS}
    if {key: len(ids) for key, ids in result.items()} != ORIGINAL_COUNTS:
        raise ValueError("Expected original counts 239/31/28")
    if len(set().union(*map(set, result.values()))) != sum(ORIGINAL_COUNTS.values()):
        raise ValueError("Original splits overlap")
    return result


def generate_split(original, source):
    original = validate_original(original)
    if not HEAD_HOLDOUT_INELIGIBLE_IDS.issubset(original["train"]):
        raise ValueError("Previously inspected proteins must remain in original train")
    eligible = set(original["train"]) - HEAD_HOLDOUT_INELIGIBLE_IDS
    ranked = sorted(eligible, key=lambda pid: (
        hashlib.sha256((SELECTION_PREFIX + pid).encode("utf-8")).hexdigest(), pid))
    holdout = ranked[:COUNTS["head_holdout"]]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "train": sorted(set(original["train"]) - set(holdout)), "validation": original["val"],
        "head_holdout": sorted(holdout), "legacy_test": original["test"],
        "selection_method": {
            "algorithm": "SHA256", "encoding": "UTF-8", "prefix": SELECTION_PREFIX,
            "sort": ["hash", "protein_id"], "take": 30,
            "eligibility_revision": "stage1.1",
            "ineligible_head_holdout_ids": sorted(HEAD_HOLDOUT_INELIGIBLE_IDS),
            "ineligible_reason": "prior explicit inspection of individual mutation labels/errors",
            "stored_order": "lexicographic", "labels_used": False,
        },
        "counts": dict(COUNTS), "source": dict(source),
        "provenance": {"original_lists_sha256": canonical_hash(original)},
    }
    payload["provenance"]["split_lists_sha256"] = canonical_hash(
        {key: payload[key] for key in COUNTS})
    return payload


def validate_split(payload, original=None):
    if not isinstance(payload, dict):
        raise ValueError("Split manifest must be a dictionary")
    normalized = {key: _ids(payload[key]) for key in COUNTS}
    if any(payload[key] != normalized[key] for key in COUNTS):
        raise ValueError("Split lists must be lexicographically sorted")
    if {key: len(ids) for key, ids in normalized.items()} != COUNTS:
        raise ValueError("Expected new split counts 209/31/30/28")
    if len(set().union(*map(set, normalized.values()))) != sum(COUNTS.values()):
        raise ValueError("New splits overlap")
    recovered = {"train": normalized["train"] + normalized["head_holdout"],
                 "val": normalized["validation"], "test": normalized["legacy_test"]}
    original = validate_original(recovered if original is None else original)
    expected = generate_split(original, payload["source"])
    if payload != expected:
        raise ValueError("Split differs from the deterministic recipe or provenance")
    return payload


def write_or_validate_split(path, original, source):
    path = Path(path)
    expected = generate_split(original, source)
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        validate_split(actual, original)
        if actual != expected:
            raise ValueError("Existing split has different provenance; refusing overwrite")
        return actual
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents a concurrent writer from replacing a different split.
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(expected, indent=2, ensure_ascii=False) + "\n")
    return expected


def sequence_duplicates(payload, sequences):
    """Report exact WT duplicates across splits; never change assignments."""
    validate_split(payload)
    memberships = {pid: split for split in COUNTS for pid in payload[split]}
    if set(sequences) != set(memberships):
        raise ValueError("Sequence metadata must cover exactly the split universe")
    grouped = {}
    for pid, sequence in sequences.items():
        if not isinstance(sequence, str) or not sequence:
            raise ValueError(f"Missing sequence metadata: {pid}")
        grouped.setdefault(sequence, []).append(pid)
    return [{"sequence_sha256": hashlib.sha256(seq.encode()).hexdigest(),
             "proteins": [{"protein_id": pid, "split": memberships[pid]} for pid in sorted(ids)]}
            for seq, ids in sorted(grouped.items())
            if len({memberships[pid] for pid in ids}) > 1]
