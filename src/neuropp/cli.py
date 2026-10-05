"""Explicit stage-1 local commands. No model training or protected evaluation."""

import argparse
import json
import sys
from pathlib import Path

from .protocol import ARTIFACT_HASHES, THERMOMPNN_COMMIT
from .splits import file_hash, load_original_ids, sequence_duplicates, write_or_validate_split


def _output(payload, destination):
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if destination:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("splits", help="Generate or validate the label-free frozen split")
    split.add_argument("--source", default="external/ThermoMPNN/dataset_splits/mega_splits.pkl")
    split.add_argument("--output", default="splits/graph_swarm_v1.json")
    audit = commands.add_parser("audit", help="Check pinned local artifacts and optional runtime freezing")
    audit.add_argument("--thermompnn-dir", default="external/ThermoMPNN")
    audit.add_argument("--runtime", action="store_true")
    audit.add_argument("--output")
    duplicates = commands.add_parser("sequence-audit", help="WT PDB metadata only, including protected split metadata")
    duplicates.add_argument("--split", default="splits/graph_swarm_v1.json")
    duplicates.add_argument("--pdb-dir", default="data/megascale/AlphaFold_model_PDBs")
    duplicates.add_argument("--output")
    real = commands.add_parser("check-real", help="Check a few existing TRAIN/VALIDATION frozen features and graphs")
    real.add_argument("--split", default="splits/graph_swarm_v1.json")
    real.add_argument("--thermompnn-dir", default="external/ThermoMPNN")
    real.add_argument("--pdb-dir", default="data/megascale/AlphaFold_model_PDBs")
    real.add_argument("--feature-dir", default="data/features/thermompnn")
    real.add_argument("--graph-dir", default="data/graphs/graph_swarm_v1")
    real.add_argument("--protein-id", action="append", required=True)
    real.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        if args.command == "splits":
            actual_hash = file_hash(args.source)
            if actual_hash != ARTIFACT_HASHES["dataset_splits/mega_splits.pkl"]:
                raise ValueError("Original split source differs from the pinned source")
            source = {"path": Path(args.source).as_posix(), "file_sha256": actual_hash,
                      "thermompnn_commit": THERMOMPNN_COMMIT,
                      "keys": {"train": "train", "validation": "val", "legacy_test": "test"}}
            payload = write_or_validate_split(args.output, load_original_ids(args.source), source)
            print(json.dumps({"counts": payload["counts"], "file_sha256": file_hash(args.output)}, indent=2))
        elif args.command == "audit":
            from .provenance import artifact_audit, load_frozen_core
            payload = load_frozen_core(args.thermompnn_dir)[2] if args.runtime else artifact_audit(args.thermompnn_dir)
            _output(payload, args.output)
            return 0 if payload["status"] == "passed" else 2
        elif args.command == "sequence-audit":
            from .local_artifacts import pdb_path_for, read_wt_pdb
            from .splits import validate_split
            payload = validate_split(json.loads(Path(args.split).read_text()))
            sequences, unavailable = {}, []
            for split_name in payload["counts"]:
                for pid in payload[split_name]:
                    try:
                        sequences[pid] = read_wt_pdb(pdb_path_for(args.pdb_dir, pid))["sequence"]
                    except (OSError, ValueError) as exc:
                        unavailable.append({"protein_id": pid, "reason": str(exc)})
            report = {"status": "blocked" if unavailable else "passed", "labels_accessed": False,
                      "metadata_source": args.pdb_dir, "sequences_checked": len(sequences), "unavailable": unavailable,
                      "exact_cross_split_duplicates": sequence_duplicates(payload, sequences) if not unavailable else None,
                      "assignments_changed": False}
            _output(report, args.output)
            return 2 if unavailable else 0
        else:
            from .cache import GraphCache
            from .data_access import DevelopmentDataAccess
            from .graph import graph_diagnostics
            from .local_artifacts import load_existing_record, pdb_path_for
            from .provenance import audit_features, load_frozen_core
            import numpy as np
            access = DevelopmentDataAccess(json.loads(Path(args.split).read_text()))
            # Reject protected or unknown IDs BEFORE loading any feature/PDB/model artifacts.
            ids = access.require_development_ids(args.protein_id)
            core, utils, audit = load_frozen_core(args.thermompnn_dir)
            reports = []
            for pid in ids:
                pdb_path = pdb_path_for(args.pdb_dir, pid)
                record, feature_path = load_existing_record(pid, pdb_path, args.feature_dir, audit["feature_signature"])
                fresh, encoded = audit_features(core, utils, utils.parse_PDB(str(pdb_path)))
                if not np.allclose(fresh, record.features, atol=1e-5, rtol=1e-5):
                    raise ValueError("Existing features differ from the pinned pre-head extraction")
                if "".join("ACDEFGHIKLMNPQRSTVWYX"[int(i)] for i in encoded) != record.sequence:
                    raise ValueError("Pinned feature extraction WT sequence differs from the PDB mapping")
                cache = GraphCache(args.graph_dir)
                cache.get_or_build(record)
                reports.append({"split": access.split_for(pid), "feature_cache": str(feature_path),
                                "feature_cache_sha256": file_hash(feature_path), "pre_head_feature_match": True,
                                "stability_head_calls": 0, "graph_cache": str(cache.path_for(record)),
                                "diagnostics": graph_diagnostics(record)})
            _output({"status": "passed", "labels_accessed": False, "provenance": audit, "proteins": reports}, args.output)
        return 0
    except (OSError, ValueError, KeyError, PermissionError, ImportError) as exc:
        _output({"status": "blocked", "reason": str(exc), "labels_accessed": False}, getattr(args, "output", None)
                if args.command != "splits" else None)
        return 2


if __name__ == "__main__":
    sys.exit(main())
