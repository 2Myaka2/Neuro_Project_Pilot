"""Development access policy; no protected evaluator or unlock exists."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

from .protocol import COUNTS, PROTECTED_SPLITS
from .splits import validate_split


class ProtectedLabelAccess(PermissionError):
    pass


class UnknownProteinID(KeyError):
    pass


@dataclass(frozen=True)
class ProteinMetadata:
    protein_id: str
    sequence: str
    residue_uid: tuple[str, ...]


@dataclass(frozen=True)
class MutationLabels:
    """Labels are separate from ProteinRecord and MutationQuery model inputs."""
    protein_id: str
    queries: tuple
    labels: tuple[float, ...]


class DevelopmentDataAccess:
    """Preflight the entire batch before invoking ANY per-protein label reader.

    Readers must use isolated, trusted per-protein label sources. A bulk MegaScale
    CSV containing protected rows is not an allowed development label source.
    This is a project API guard, not an operating-system security sandbox.
    """

    def __init__(self, manifest):
        validate_split(manifest)
        self.membership = MappingProxyType({pid: split for split in COUNTS for pid in manifest[split]})

    def split_for(self, protein_id):
        try:
            return self.membership[protein_id]
        except (KeyError, TypeError):
            raise UnknownProteinID(f"Unknown protein_id: {protein_id!r}") from None

    def require_development_ids(self, protein_ids, claimed_split=None):
        ids = tuple(protein_ids)
        actual = [self.split_for(pid) for pid in ids]
        if claimed_split is not None and claimed_split not in COUNTS:
            raise ValueError(f"Unknown split: {claimed_split}")
        protected = [(pid, split) for pid, split in zip(ids, actual) if split in PROTECTED_SPLITS]
        if protected or claimed_split in PROTECTED_SPLITS:
            raise ProtectedLabelAccess(f"Protected labels are unavailable in development: {protected or claimed_split}")
        if claimed_split is not None and any(split != claimed_split for split in actual):
            raise ValueError("Requested split does not match actual protein_id membership")
        return ids

    def load_labels(self, protein_ids, reader, claimed_split=None):
        ids = self.require_development_ids(protein_ids, claimed_split)
        return {pid: reader(pid) for pid in ids}

    def load_label_directory(self, protein_ids, directory, records, claimed_split=None):
        """Concrete loader for separately supplied, per-protein development labels.

        Layout: directory/SHA256(UTF8(protein_id))/metadata.json and labels.json.
        metadata.json contains only {"protein_id": ...}. labels.json contains
        {"queries": [{"mutation_position": ..., "wt_index": ..., "mut_index": ...}],
         "labels": [...]}. All identities are checked before ANY label file opens.
        The owner must supply isolated train/validation files; this does not
        export or parse a bulk CSV containing protected labels.
        """
        from .graph import MutationQuery, ProteinRecord

        ids = self.require_development_ids(protein_ids, claimed_split)
        directories = {}
        for pid in ids:
            folder = Path(directory) / hashlib.sha256(pid.encode("utf-8")).hexdigest()
            identity = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            if set(identity) != {"protein_id"}:
                raise ValueError("Label source metadata must contain only protein_id")
            # Check the SOURCE identity as well as the requested identity.
            self.require_development_ids([identity["protein_id"]], claimed_split)
            if identity["protein_id"] != pid:
                raise ValueError("Label source protein_id does not match the requested protein")
            record = records[pid]
            if not isinstance(record, ProteinRecord) or record.protein_id != pid:
                raise ValueError("Labels require a matching validated label-free protein record")
            record.validate()
            directories[pid] = folder
        result = {}
        for pid in ids:
            raw = json.loads((directories[pid] / "labels.json").read_text(encoding="utf-8"))
            if set(raw) != {"queries", "labels"} or len(raw["queries"]) != len(raw["labels"]):
                raise ValueError("Label source queries/labels contract mismatch")
            queries = tuple(MutationQuery(**query).validate(records[pid]) for query in raw["queries"])
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                   for value in raw["labels"]):
                raise ValueError("Labels must be finite numbers")
            result[pid] = MutationLabels(pid, queries, tuple(float(value) for value in raw["labels"]))
        return result

    def metadata_only(self, protein_ids, reader):
        ids = tuple(protein_ids)
        for pid in ids:
            self.split_for(pid)
        records = {}
        for pid in ids:
            record = reader(pid)
            if not isinstance(record, ProteinMetadata) or record.protein_id != pid:
                raise ValueError("Metadata reader must return matching, label-free ProteinMetadata")
            records[pid] = record
        return records
