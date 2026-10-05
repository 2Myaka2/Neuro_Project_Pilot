"""Atomic, integrity-checked graph-only cache, separate from frozen features."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .graph import ProteinRecord
from .protocol import GraphParameters
from .splits import canonical_hash

CACHE_NAMESPACE = Path("data/graphs/graph_swarm_v1")


class CacheError(ValueError):
    pass


def array_hash(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def cache_metadata(record, parameters=GraphParameters()):
    record.validate()
    return {"format_version": 1, "protein_id": record.protein_id, "sequence": record.sequence,
            "coordinates_sha256": array_hash(record.ca_coordinates),
            "frozen_features_sha256": array_hash(record.features),
            "residue_mapping": {name: array_hash(getattr(record, name)) for name in
                                ("seq_pos", "residue_uid", "residue_amino_acids", "feature_amino_acids")},
            "graph_parameters": parameters.as_dict(), "protocol_version": parameters.protocol_version}


class GraphCache:
    def __init__(self, root=CACHE_NAMESPACE):
        self.root = Path(root)  # construction/import has no I/O

    def path_for(self, record, parameters=GraphParameters()):
        return self.root / (canonical_hash(cache_metadata(record, parameters)) + ".npz")

    def write(self, record, parameters=GraphParameters()):
        metadata = cache_metadata(record, parameters)
        manifest = {"metadata": metadata,
                    "arrays": {name: array_hash(getattr(record, name)) for name in ("edge_index", "edge_attr")}}
        path = self.root / (canonical_hash(metadata) + ".npz")
        self.root.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".graph-", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez(stream, manifest=np.array(json.dumps(manifest, sort_keys=True)),
                         edge_index=record.edge_index, edge_attr=record.edge_attr)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return path

    def read(self, record, parameters=GraphParameters()):
        metadata = cache_metadata(record, parameters)
        path = self.root / (canonical_hash(metadata) + ".npz")
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as saved:
                if set(saved.files) != {"manifest", "edge_index", "edge_attr"}:
                    raise CacheError("Unexpected graph cache fields")
                manifest = json.loads(str(saved["manifest"].item()))
                if manifest["metadata"] != metadata:
                    raise CacheError("Incompatible graph cache metadata")
                arrays = {name: saved[name].copy() for name in ("edge_index", "edge_attr")}
                if manifest["arrays"] != {name: array_hash(value) for name, value in arrays.items()}:
                    raise CacheError("Corrupted graph cache array checksum")
            ProteinRecord(**{**record.__dict__, **arrays}).validate()
            return arrays["edge_index"], arrays["edge_attr"]
        except Exception as exc:
            raise CacheError(f"Invalid graph cache entry {path}: {exc}") from exc

    def get_or_build(self, record, parameters=GraphParameters()):
        found = self.read(record, parameters)
        if found is None:
            self.write(record, parameters)
            return record.edge_index, record.edge_attr
        return found
