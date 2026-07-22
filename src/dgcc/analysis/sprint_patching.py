"""CPU-only primitives for the AMD-2 response-conditioned h_p patching battery.

This module deliberately consumes only content-addressed probe files; it never opens a split.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import torch
from scipy.stats import kendalltau

PAIRING_SEED = 20260723
PROJECTOR_SEED = 20260719
PROJECTOR_SHA256 = "408120a3db83df654ee3d1ead6e54a09b04d0c5d6c7a477dd8da301c822d51ab"
N_PAIRS_PER_RUN = 100
ALPHAS = (0.25, 0.5, 0.75, 1.0)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash exact unnormalised tensor bytes, including dtype and shape."""
    x = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(x.dtype).encode())
    digest.update(np.asarray(x.shape, dtype=np.int64).tobytes())
    digest.update(x.numpy().tobytes())
    return digest.hexdigest()


def a0_interchange(recipient: torch.Tensor, donor: torch.Tensor) -> torch.Tensor:
    if recipient.shape != donor.shape:
        raise ValueError("recipient and donor h_p shapes must match")
    return donor.clone()


def verify_projector_sha256(projector: torch.Tensor, expected_sha256: str = PROJECTOR_SHA256) -> None:
    # The published pin hashes the serialized tensor payload used by the producer.
    # For in-memory callers, accept either raw contiguous bytes or torch-save bytes.
    raw = hashlib.sha256(projector.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    if expected_sha256 not in {raw, tensor_sha256(projector)}:
        raise ValueError("matched-P projector sha256 mismatch")


def a1_projected_splice(recipient: torch.Tensor, donor: torch.Tensor, projector: torch.Tensor, *, expected_sha256: str | None = None) -> torch.Tensor:
    if recipient.shape != donor.shape or recipient.shape[-1] != projector.shape[-1]:
        raise ValueError("projected splice dimensions are invalid")
    if expected_sha256 is not None:
        verify_projector_sha256(projector, expected_sha256)
    p = projector.to(device=recipient.device, dtype=recipient.dtype)
    # P has orthonormal rows, hence ΠP=PᵀP. No normalization is performed.
    projection = lambda h: (h @ p.T) @ p
    return recipient - projection(recipient) + projection(donor)


def mean_ablation(recipient: torch.Tensor, probe_mean: torch.Tensor) -> torch.Tensor:
    if recipient.shape[-1] != probe_mean.shape[-1]:
        raise ValueError("mean-ablation dimensions are invalid")
    return probe_mean.to(device=recipient.device, dtype=recipient.dtype).expand_as(recipient).clone()


def graded_ablation(recipient: torch.Tensor, probe_mean: torch.Tensor, alpha: float) -> torch.Tensor:
    if alpha not in ALPHAS:
        raise ValueError(f"alpha must be one of {ALPHAS}")
    mean = mean_ablation(recipient, probe_mean)
    return (1.0 - alpha) * recipient + alpha * mean


def rescale_mask(ratio: float, *, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """24-channel axis-major DCT scale: x/y modes 1..7 only."""
    if not np.isfinite(ratio):
        raise ValueError("rescale ratio must be finite")
    mask = torch.ones(24, device=device, dtype=dtype)
    mask[1:8] = ratio
    mask[9:16] = ratio
    return mask


def rescale_delta(delta: torch.Tensor, ratio: float) -> torch.Tensor:
    return delta * rescale_mask(ratio, device=delta.device, dtype=delta.dtype)


def load_projector(path: Path | str, *, expected_sha256: str = PROJECTOR_SHA256) -> torch.Tensor:
    path = Path(path)
    if sha256_file(path) != expected_sha256:
        raise ValueError("matched-P projector sha256 mismatch")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or value.shape != (24, 256):
        raise ValueError("matched-P projector must be a 24x256 tensor")
    return value


def load_probe_manifest(path: Path | str, *, required_fields: Sequence[str] = ("h_p",)) -> list[dict[str, np.ndarray]]:
    """Load exactly manifest-listed HDF5 files after digest/size verification."""
    manifest_path = Path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest["files"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError("probe manifest schema is invalid") from exc
    if not isinstance(files, dict) or set(manifest) != {"schema_version", "files"} or not files:
        raise ValueError("probe manifest schema is invalid")
    rows: list[dict[str, np.ndarray]] = []
    seen: set[Path] = set()
    for digest, entry in sorted(files.items()):
        if not isinstance(entry, dict) or set(("path", "sha256", "size")) - set(entry):
            raise ValueError("probe manifest schema is invalid")
        file_path = Path(entry["path"])
        try:
            canonical = file_path.resolve(strict=True)
        except OSError as exc:
            raise ValueError("manifest-listed probe is missing") from exc
        if canonical in seen or canonical.is_symlink() or digest != entry["sha256"] or sha256_file(canonical) != digest or canonical.stat().st_size != entry["size"]:
            raise ValueError("probe manifest digest verification failed")
        seen.add(canonical)
        try:
            with h5py.File(canonical, "r") as handle:
                missing = [field for field in required_fields if field not in handle]
                if missing:
                    raise ValueError(f"probe file missing fields: {missing}")
                rows.append({field: np.asarray(handle[field]) for field in required_fields})
        except OSError as exc:
            raise ValueError("manifest-listed probe is not valid HDF5") from exc
    return rows


def deterministic_pairing(recipients: Sequence[Mapping[str, Any]], donors: Sequence[Mapping[str, Any]], *, ratio: float = 1.0, seed: int = PAIRING_SEED) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Choose goal/p/nuisance-matched donors by rescaled δm distance deterministically."""
    rng = np.random.default_rng(seed)
    pairs = []
    for recipient in sorted(recipients, key=lambda x: str(x["goal_id"])):
        target = np.asarray(recipient["delta_m"], dtype=float) * rescale_mask(ratio, dtype=torch.float64).numpy()
        candidates = sorted(
            (d for d in donors if d.get("goal_id") == recipient.get("goal_id") and d.get("p") == recipient.get("p") and d.get("nuisance") == recipient.get("nuisance")),
            key=lambda d: str(d.get("id", "")),
        )
        if not candidates:
            raise ValueError("no nuisance-matched donor")
        # Stable random tie key fixes ties without depending on input order.
        scored = [(float(np.abs(np.asarray(d["delta_m"], dtype=float) - target).sum()), float(rng.random()), str(d.get("id", "")), d) for d in candidates]
        pairs.append((recipient, min(scored, key=lambda x: x[:3])[3]))
    return pairs


def ranking_change(baseline_q: Sequence[float], patched_q: Sequence[float]) -> dict[str, float]:
    base, patch = np.asarray(baseline_q, dtype=float), np.asarray(patched_q, dtype=float)
    if base.ndim != 1 or patch.shape != base.shape or len(base) < 2:
        raise ValueError("Q rankings require matching vectors with at least two actions")
    tau = kendalltau(base, patch).statistic
    return {"kendall_tau_change": float((0.0 if np.isnan(tau) else tau) - 1.0), "top1_flip": float(np.argmax(base) != np.argmax(patch))}


def estimate_real_minus_null(real: Iterable[Mapping[str, Sequence[float]]], null: Iterable[Mapping[str, Sequence[float]]]) -> dict[str, float]:
    real_rows, null_rows = list(real), list(null)
    if len(real_rows) != len(null_rows) or not real_rows:
        raise ValueError("real and null pair counts must match and be nonempty")
    effects = [{key: ranking_change(r["baseline_q"], r["patched_q"])[key] - ranking_change(n["baseline_q"], n["patched_q"])[key] for key in ("kendall_tau_change", "top1_flip")} for r, n in zip(real_rows, null_rows, strict=True)]
    return {key: float(np.mean([row[key] for row in effects])) for key in effects[0]}
def estimate_by_run(real: Iterable[Mapping[str, Any]], null: Iterable[Mapping[str, Any]], *, bootstrap_draws: int = 10_000) -> dict[str, Any]:
    """AMD-2 estimator: pair effects → run means → paired seed-cluster BCa."""
    real_rows, null_rows = list(real), list(null)
    if len(real_rows) != len(null_rows) or not real_rows:
        raise ValueError("real and null pair counts must match and be nonempty")
    by_run: dict[Any, dict[str, list[float]]] = {}
    for r, n in zip(real_rows, null_rows, strict=True):
        if r.get("run") != n.get("run") or r.get("run") is None:
            raise ValueError("real and null rows require the same run")
        effects = estimate_real_minus_null([r], [n])
        bucket = by_run.setdefault(r["run"], {key: [] for key in effects})
        for key, value in effects.items():
            bucket[key].append(value)
    run_means = {key: [float(np.mean(by_run[run][key])) for run in sorted(by_run)] for key in next(iter(by_run.values()))}
    import sys
    scripts_dir = str(Path(__file__).resolve().parents[3] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from sprint_stats import seed_cluster_bootstrap
    return {"run_means": run_means, "bca": {key: seed_cluster_bootstrap(values, draws=bootstrap_draws) for key, values in run_means.items()}}


def provenance_record(*, arm: str, ckpt_sha256: str, split_sha256: str, claim_sha256: str, operator: str, parameter_pre: torch.Tensor, parameter_post: torch.Tensor) -> dict[str, str]:
    return {"arm": arm, "ckpt_sha256": ckpt_sha256, "split_sha256": split_sha256, "claim_sha256": claim_sha256, "operator": operator, "parameter_pre_sha256": tensor_sha256(parameter_pre), "parameter_post_sha256": tensor_sha256(parameter_post), "interpretation": "response-conditioned h_p mediation"}
