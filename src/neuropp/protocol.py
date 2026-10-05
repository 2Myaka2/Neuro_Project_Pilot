"""Frozen scientific constants; split eligibility amendments are recorded separately."""

from dataclasses import asdict, dataclass

PROTOCOL_VERSION = "graph_swarm_v1:1"
SELECTION_PREFIX = "graph_swarm_v1:20261004:"
HEAD_HOLDOUT_INELIGIBLE_IDS = frozenset({"1A32.pdb"})
COUNTS = {"train": 209, "validation": 31, "head_holdout": 30, "legacy_test": 28}
ORIGINAL_COUNTS = {"train": 239, "val": 31, "test": 28}
PROTECTED_SPLITS = frozenset({"head_holdout", "legacy_test"})
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
THERMOMPNN_COMMIT = "370f76ec62bd929f7425e311d8df04a0d094990f"
ARTIFACT_HASHES = {
    "models/thermoMPNN_default.pt": "af449118ccfb4e34971d802321907ed82a8a035ae80f055bf8fc56009a4a838a",
    "vanilla_model_weights/v_48_020.pt": "c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd",
    "dataset_splits/mega_splits.pkl": "9e06230fe8febd8f07f8fab374f153328d4bdde20b78de15ead93b39e61ca1eb",
}


@dataclass(frozen=True)
class GraphParameters:
    k: int = 16
    rbf_count: int = 16
    rbf_min: float = 2.0
    rbf_max: float = 20.0
    rbf_sigma: float = 1.2
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self):
        if asdict(self) != {
            "k": 16, "rbf_count": 16, "rbf_min": 2.0, "rbf_max": 20.0,
            "rbf_sigma": 1.2, "protocol_version": PROTOCOL_VERSION,
        }:
            raise ValueError("Graph parameters differ from the frozen GraphSWARM v1 protocol")

    def as_dict(self):
        return asdict(self)
