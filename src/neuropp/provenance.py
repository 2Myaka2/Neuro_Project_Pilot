"""Explicit local provenance audit. No downloads, datasets or training."""

import importlib
import subprocess
import sys
from pathlib import Path

from .protocol import ARTIFACT_HASHES, THERMOMPNN_COMMIT
from .splits import file_hash


class FeatureProvenanceError(ValueError):
    """Carry the failed runtime audit so callers can persist its tensor evidence."""

    def __init__(self, report):
        self.report = report
        super().__init__(report["reason"])


def compare_proteinmpnn_state_dict(actual, expected):
    """Compare every stored tensor exactly, including buffers; never relax tolerances."""
    import torch

    actual_keys, expected_keys = set(actual), set(expected)
    report = {
        "passed": False, "compared_tensors": 0, "exact_match": False,
        "max_abs_difference": 0.0,
        "key_set_equal": actual_keys == expected_keys,
        "shapes_equal": True, "dtypes_equal": True,
        "dtype_policy": "identical torch dtype; no casting for equality",
        "comparison": "torch.equal; no tolerance",
        "missing_keys": sorted(expected_keys - actual_keys),
        "unexpected_keys": sorted(actual_keys - expected_keys),
        "mismatched_tensors": [], "nonfinite_tensors": [],
    }
    for key in sorted(actual_keys & expected_keys):
        loaded, original = actual[key], expected[key]
        if not isinstance(loaded, torch.Tensor) or not isinstance(original, torch.Tensor):
            report["mismatched_tensors"].append({"key": key, "reason": "non-tensor state entry"})
            continue
        loaded, original = loaded.detach().cpu(), original.detach().cpu()
        dtype_match = loaded.dtype == original.dtype
        if not dtype_match:
            report["dtypes_equal"] = False
        if loaded.shape != original.shape:
            report["shapes_equal"] = False
            report["mismatched_tensors"].append({
                "key": key, "reason": "shape mismatch",
                "actual_shape": list(loaded.shape), "expected_shape": list(original.shape),
                "actual_dtype": str(loaded.dtype), "expected_dtype": str(original.dtype)})
            continue
        report["compared_tensors"] += 1
        finite = bool(torch.isfinite(loaded).all() and torch.isfinite(original).all())
        if not finite:
            report["nonfinite_tensors"].append(key)
        values_match = torch.equal(loaded, original)
        difference = None
        if finite and loaded.numel():
            # Float64 measures float32 checkpoint discrepancies without float32 subtraction rounding.
            difference = float((loaded.to(torch.float64) - original.to(torch.float64)).abs().max())
            report["max_abs_difference"] = max(report["max_abs_difference"], difference)
        if not dtype_match or not values_match or not finite:
            report["mismatched_tensors"].append({
                "key": key, "reason": "dtype, value or finiteness mismatch",
                "actual_dtype": str(loaded.dtype), "expected_dtype": str(original.dtype),
                "values_equal": values_match, "max_abs_difference": difference})
    if report["nonfinite_tensors"]:
        report["max_abs_difference"] = None
    report["exact_match"] = bool(expected_keys) and report["key_set_equal"] and not report["mismatched_tensors"]
    report["passed"] = report["exact_match"]
    return report


def artifact_audit(thermompnn_dir):
    root = Path(thermompnn_dir).resolve()
    checks = {}
    required = [*ARTIFACT_HASHES, "config.yaml", "transfer_model.py", "protein_mpnn_utils.py"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        return {"status": "blocked", "reason": "missing artifact", "missing": missing, "checks": checks}
    try:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        checks["pinned_version"] = {"passed": commit == THERMOMPNN_COMMIT and not dirty,
                                    "expected": THERMOMPNN_COMMIT, "actual": commit, "tracked_changes": dirty}
        for name, expected in ARTIFACT_HASHES.items():
            actual = file_hash(root / name)
            checks[name] = {"passed": actual == expected, "expected_sha256": expected, "actual_sha256": actual}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"status": "blocked", "reason": str(exc), "checks": checks}
    if not all(check["passed"] for check in checks.values()):
        return {"status": "blocked", "reason": "pinned artifact mismatch", "checks": checks}
    return {"status": "pending", "reason": "runtime tensor identity, freezing and pre-head path audit required", "checks": checks}


def load_frozen_core(thermompnn_dir):
    """Load only hash-verified local artifacts; never import training entry points."""
    root = Path(thermompnn_dir).resolve()
    audit = artifact_audit(root)
    if audit["status"] == "blocked":
        raise ValueError(f"Feature provenance blocked: {audit['reason']}")
    import torch
    from omegaconf import OmegaConf

    sys.path.insert(0, str(root))
    try:
        transfer = importlib.import_module("transfer_model")
        utils = importlib.import_module("protein_mpnn_utils")
        if any(Path(module.__file__).resolve().parent != root for module in (transfer, utils)):
            raise ValueError("ThermoMPNN modules were imported from an unexpected checkout")
        cfg = OmegaConf.load(root / "config.yaml")
        cfg.platform = {"thermompnn_dir": str(root), "accel": "cpu"}
        if not cfg.model.freeze_weights or not cfg.model.load_pretrained or cfg.model.num_final_layers != 2:
            raise ValueError("Expected frozen, pretrained ProteinMPNN with two final layers")
        core = transfer.TransferModel(cfg)
        # This trusted upstream checkpoint is verified by its pinned SHA256 above.
        saved = torch.load(root / "models/thermoMPNN_default.pt", map_location="cpu", weights_only=False)
        state = {key[len("model."):]: value for key, value in saved["state_dict"].items() if key.startswith("model.")}
        core.load_state_dict(state, strict=True)
        original = torch.load(root / "vanilla_model_weights/v_48_020.pt", map_location="cpu", weights_only=True)
        tensor_match = compare_proteinmpnn_state_dict(core.prot_mpnn.state_dict(), original["model_state_dict"])
        tensor_match.update(
            actual_source="core.prot_mpnn.state_dict() after strict ThermoMPNN checkpoint loading",
            expected_source="vanilla_model_weights/v_48_020.pt:model_state_dict",
        )
        audit["checks"]["proteinmpnn_tensor_match"] = tensor_match
        if not tensor_match["passed"]:
            audit.update(status="blocked", labels_accessed=False,
                         reason=f"ProteinMPNN tensor comparison failed; max_abs_difference={tensor_match['max_abs_difference']}")
            raise FeatureProvenanceError(audit)
        initially_frozen = all(not p.requires_grad for p in core.prot_mpnn.parameters())
        if not initially_frozen:
            raise ValueError("ProteinMPNN is not frozen at construction")
        core.eval().requires_grad_(False)
        spec = {"format_version": 1, "commit": THERMOMPNN_COMMIT,
                "checkpoint_sha256": file_hash(root / "models/thermoMPNN_default.pt"),
                "proteinmpnn_sha256": file_hash(root / "vanilla_model_weights/v_48_020.pt"),
                "source_sha256": {name: file_hash(root / name) for name in ["protein_mpnn_utils.py", "transfer_model.py"]},
                "model_config": OmegaConf.to_container(cfg.model, resolve=True),
                "torch": str(torch.__version__), "device": "cpu"}
        # Match the first experiment's JSON serialization exactly.
        import hashlib
        import json
        signature = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        audit["checks"]["proteinmpnn_frozen"] = {"passed": initially_frozen}
        audit["checks"]["feature_path"] = {
            "passed": True, "dimension": 384,
            "path": "prot_mpnn -> two decoder hidden states + WT embedding -> concatenation",
            "excluded": ["light_attention", "both_out", "ddg_out"],
            "evidence": "pinned source; extraction function invokes only prot_mpnn; real-data hooks checked separately"}
        audit.update(status="passed", reason="pinned artifacts, exact ProteinMPNN tensor identity and runtime freezing verified",
                     feature_signature=signature, feature_spec=spec)
        return core, utils, audit
    finally:
        sys.path.pop(0)


def audit_features(core, utils, pdb_batch):
    """Run only the original frozen feature path; forbid stability-head calls."""
    import torch

    def forbidden(*args):
        raise RuntimeError("Stability head must never be invoked by feature extraction")

    modules = [core.both_out, core.ddg_out]
    if core.lightattn:
        modules.append(core.light_attention)
    handles = [module.register_forward_pre_hook(forbidden) for module in modules]
    try:
        with torch.inference_mode():
            packed = utils.tied_featurize([pdb_batch[0]], torch.device("cpu"), None, ca_only=False)
            X, S, mask, _, chain_M, chains = packed[:6]
            hidden, embedding, _ = core.prot_mpnn(X, S, mask, chain_M, packed[12], chains, None)
            features = torch.cat([*hidden[:core.num_final_layers], embedding], dim=-1)[0]
            if features.shape != (len(pdb_batch[0]["seq"]), 384) or not mask[0].bool().all() or not torch.isfinite(features).all():
                raise ValueError("Incomplete, misaligned or nonfinite frozen feature extraction")
            if features.requires_grad or any(p.requires_grad for p in core.prot_mpnn.parameters()):
                raise ValueError("Feature extraction is not frozen")
            return features.cpu().numpy(), S[0].cpu().numpy()
    finally:
        for handle in handles:
            handle.remove()
