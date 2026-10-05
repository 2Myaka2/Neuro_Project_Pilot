"""Label-free C-alpha graphs, immutable residue mappings and diagnostics."""

from dataclasses import dataclass

import numpy as np

from .protocol import ALPHABET, GraphParameters


def _geometry(coordinates, seq_pos):
    xyz = np.asarray(coordinates)
    pos = np.asarray(seq_pos)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError("C-alpha coordinates must have shape [L,3] with L > 0")
    if xyz.dtype not in (np.dtype("float32"), np.dtype("float64")) or not np.isfinite(xyz).all():
        raise ValueError("Coordinates must be finite float32 or float64 values")
    if pos.shape != (len(xyz),) or pos.dtype.kind not in "iu":
        raise ValueError("seq_pos must be an integer [L] array")
    if set(map(int, pos)) != set(range(len(xyz))):
        raise ValueError("seq_pos must map every original zero-based WT position exactly once")
    return xyz, pos


def radial_basis(distances, parameters=GraphParameters()):
    distances = np.asarray(distances)
    if distances.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("Distances must be float32 or float64")
    if not np.isfinite(distances).all() or (distances < 0).any():
        raise ValueError("Distances must be finite and nonnegative")
    centers = np.linspace(parameters.rbf_min, parameters.rbf_max,
                          parameters.rbf_count, dtype=distances.dtype)
    return np.exp(-((distances[..., None] - centers) ** 2) / (2 * parameters.rbf_sigma ** 2))


def build_graph(ca_coordinates, seq_pos, parameters=GraphParameters()):
    xyz, pos = _geometry(ca_coordinates, seq_pos)
    length = len(xyz)
    if length == 1:
        return np.empty((2, 0), dtype=np.int64), np.empty((0, 17), dtype=xyz.dtype)
    # Compute in float64 for stable tie handling; no distance clipping.
    delta = xyz.astype(np.float64)[:, None, :] - xyz.astype(np.float64)[None, :, :]
    distances = np.linalg.norm(delta, axis=-1)
    if not np.isfinite(distances).all():
        raise ValueError("Coordinate distances overflowed")
    relations = set()
    for i in range(length):
        candidates = np.flatnonzero(np.arange(length) != i)
        order = np.lexsort((pos[candidates], distances[i, candidates]))
        for j in candidates[order[:min(parameters.k, length - 1)]]:
            relations.add((int(j), i))
            relations.add((i, int(j)))
    edge_index = np.asarray(sorted(relations), dtype=np.int64).T
    source, destination = edge_index
    edge_distances = distances[source, destination].astype(xyz.dtype)
    separation = (np.log1p(np.abs(pos[source].astype(np.float64) - pos[destination]))
                  / np.log1p(length)).astype(xyz.dtype)
    edge_attr = np.column_stack((radial_basis(edge_distances, parameters), separation))
    return edge_index, edge_attr


@dataclass(frozen=True)
class ProteinRecord:
    protein_id: str
    sequence: str  # canonical WT order, independent of current array row order
    features: np.ndarray
    ca_coordinates: np.ndarray
    seq_pos: np.ndarray
    residue_uid: np.ndarray  # chain:residue_number:insertion_code
    residue_amino_acids: np.ndarray  # PDB residue letters in array row order
    feature_amino_acids: np.ndarray  # decoded frozen-feature WT letters per row
    edge_index: np.ndarray
    edge_attr: np.ndarray

    def validate(self):
        if not isinstance(self.protein_id, str) or not self.protein_id:
            raise ValueError("protein_id is required")
        xyz, pos = _geometry(self.ca_coordinates, self.seq_pos)
        length = len(xyz)
        if not isinstance(self.sequence, str) or len(self.sequence) != length:
            raise ValueError("Sequence/coordinates length mismatch")
        if any(aa not in ALPHABET[:-1] for aa in self.sequence):
            raise ValueError("WT sequence must contain standard amino acids")
        features = np.asarray(self.features)
        if features.shape != (length, 384) or features.dtype.kind != "f" or not np.isfinite(features).all():
            raise ValueError("Frozen features must be finite [L,384] values")
        uid = np.asarray(self.residue_uid)
        if uid.shape != (length,) or uid.dtype.kind not in "US" or len(set(uid.tolist())) != length:
            raise ValueError("Residue UIDs must be unique strings with length L")
        expected = np.array(list(self.sequence))[pos]
        for letters in (self.residue_amino_acids, self.feature_amino_acids):
            if np.asarray(letters).shape != (length,) or not np.array_equal(letters, expected):
                raise ValueError("WT/PDB/frozen-feature sequence alignment mismatch")
        for value in uid.tolist():
            parts = value.split(":")
            if len(parts) != 3 or not parts[0] or not parts[1].lstrip("-").isdigit():
                raise ValueError("Ambiguous residue UID; expected chain:number:insertion_code")
        # Full-length single-chain mapping must be strictly ordered in canonical sequence order.
        canonical_uids = uid[np.argsort(pos)].tolist()
        parsed = [(value.split(":")[0], int(value.split(":")[1]), value.split(":")[2]) for value in canonical_uids]
        if len({item[0] for item in parsed}) != 1 or parsed != sorted(parsed):
            raise ValueError("PDB mapping must describe one chain in original residue order")
        expected_edges, expected_attr = build_graph(xyz, pos)
        edges = np.asarray(self.edge_index)
        attrs = np.asarray(self.edge_attr)
        if edges.dtype.kind not in "iu" or edges.shape != expected_edges.shape or not np.array_equal(edges, expected_edges):
            raise ValueError("Graph topology differs from the frozen protocol")
        atol, rtol = (1e-5, 1e-5) if xyz.dtype == np.float32 else (1e-8, 1e-7)
        if attrs.shape != expected_attr.shape or not np.isfinite(attrs).all() or not np.allclose(attrs, expected_attr, atol=atol, rtol=rtol):
            raise ValueError("Graph edge attributes differ from the frozen protocol")
        return self


def make_record(protein_id, sequence, features, ca_coordinates, seq_pos,
                residue_uid, residue_amino_acids, feature_amino_acids):
    edges, attrs = build_graph(ca_coordinates, seq_pos)
    return ProteinRecord(protein_id, sequence, np.asarray(features), np.asarray(ca_coordinates),
                         np.asarray(seq_pos), np.asarray(residue_uid), np.asarray(residue_amino_acids),
                         np.asarray(feature_amino_acids), edges, attrs).validate()


@dataclass(frozen=True)
class MutationQuery:
    mutation_position: int  # immutable zero-based WT position, NOT a current row index
    wt_index: int
    mut_index: int

    def validate(self, record):
        values = (self.mutation_position, self.wt_index, self.mut_index)
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in values):
            raise ValueError("Mutation indices must be integers")
        if not 0 <= self.mutation_position < len(record.sequence):
            raise ValueError("Mutation position out of range")
        if not (0 <= self.wt_index < 20 and 0 <= self.mut_index < 20) or self.wt_index == self.mut_index:
            raise ValueError("Expected a single substitution between standard amino acids")
        if ALPHABET[self.wt_index] != record.sequence[self.mutation_position]:
            raise ValueError("Mutation WT residue does not match the canonical sequence")
        return self


def graph_diagnostics(record):
    record.validate()
    length = len(record.sequence)
    adjacency = [set() for _ in range(length)]
    for source, destination in record.edge_index.T:
        adjacency[int(source)].add(int(destination))
    degree = np.array([len(neighbors) for neighbors in adjacency])
    unseen, components = set(range(length)), 0
    while unseen:
        components += 1
        frontier = {min(unseen)}
        while frontier:
            unseen -= frontier
            frontier = set().union(*(adjacency[node] for node in frontier)) & unseen
    reachable = {hop: [] for hop in (1, 2, 4)}
    for origin in range(length):
        visited, frontier = {origin}, {origin}
        for hop in range(1, 5):
            frontier = set().union(*(adjacency[node] for node in frontier)) - visited
            visited |= frontier
            if hop in reachable:
                # Mean fraction of OTHER residues reached, not counting the starting node.
                reachable[hop].append((len(visited) - 1) / (length - 1) if length > 1 else 0.0)
    distances = np.linalg.norm(record.ca_coordinates[record.edge_index[0]].astype(np.float64)
                               - record.ca_coordinates[record.edge_index[1]], axis=-1)
    return {"protein_id": record.protein_id, "length": length, "directed_edges": int(len(distances)),
            "degree": {"min": int(degree.min()), "mean": float(degree.mean()), "max": int(degree.max())},
            "connected_components": components,
            "edge_distance_angstrom": {"min": float(distances.min()) if len(distances) else None,
                                        "max": float(distances.max()) if len(distances) else None},
            "reachable_fraction": {str(hop): float(np.mean(values)) for hop, values in reachable.items()},
            "reachability_definition": "mean over start nodes; other residues; singleton=0"}
