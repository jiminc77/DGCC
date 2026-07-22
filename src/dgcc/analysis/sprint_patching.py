"""CPU-only primitives for the AMD-2 response-conditioned h_p patching battery.

This module deliberately consumes only content-addressed probe files; it never opens a split.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
from scipy.stats import kendalltau

# AMD-2 confirmatory pairing seed, specified in the registered analysis protocol.
PAIRING_SEED = 20260723
PROJECTOR_SEED = 20260719
PROJECTOR_SHA256 = "408120a3db83df654ee3d1ead6e54a09b04d0c5d6c7a477dd8da301c822d51ab"
N_PAIRS_PER_RUN = 100
ALPHAS = (0.25, 0.5, 0.75, 1.0)
_LOADED_PROJECTOR_SEAL = object()
_VERIFIED_BUNDLE_SEAL = object()


@dataclass(frozen=True)
class LoadedProjector:
    """A projector that passed the serialized-file pin and geometry checks."""
    value: torch.Tensor
    sha256: str
    _tensor_sha256: str
    _seal: object


@dataclass(frozen=True)
class VerifiedProbeBundle:
    """Opaque manifest-verified probe inputs for confirmatory estimation."""
    _rows: tuple[Mapping[str, np.ndarray], ...]
    _estimator_records: tuple[Mapping[str, Any], ...]
    manifest_sha256: str
    _seal: object

    @property
    def rows(self) -> tuple[Mapping[str, np.ndarray], ...]:
        return self._rows

    @property
    def estimator_records(self) -> tuple[Mapping[str, Any], ...]:
        return self._estimator_records

    def __len__(self) -> int:
        return len(self._rows)


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


def verify_projector_sha256(projector: torch.Tensor) -> None:
    raw = hashlib.sha256(projector.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    if PROJECTOR_SHA256 not in {raw, tensor_sha256(projector)}:
        raise ValueError("matched-P projector sha256 mismatch")


def _validate_projector_geometry(projector: torch.Tensor) -> None:
    if projector.shape != (24, 256):
        raise ValueError("matched-P projector must be a 24x256 tensor")
    if not torch.is_floating_point(projector) or not torch.isfinite(projector).all():
        raise ValueError("matched-P projector must be finite floating point")
    gram = projector.to(dtype=torch.float64) @ projector.to(dtype=torch.float64).T
    if not torch.allclose(gram, torch.eye(24, dtype=torch.float64), rtol=1e-5, atol=1e-6):
        raise ValueError("matched-P projector rows must be orthonormal")


def load_projector(path: Path | str) -> LoadedProjector:
    path = Path(path)
    if sha256_file(path) != PROJECTOR_SHA256:
        raise ValueError("matched-P projector sha256 mismatch")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise ValueError("matched-P projector must be a 24x256 tensor")
    _validate_projector_geometry(value)
    return LoadedProjector(value=value, sha256=PROJECTOR_SHA256, _tensor_sha256=tensor_sha256(value), _seal=_LOADED_PROJECTOR_SEAL)


def a1_projected_splice(recipient: torch.Tensor, donor: torch.Tensor, projector: LoadedProjector) -> torch.Tensor:
    """Apply A1 only with a pinned, loader-validated matched-P projector."""
    if not isinstance(projector, LoadedProjector) or projector._seal is not _LOADED_PROJECTOR_SEAL:
        raise TypeError("a1_projected_splice requires a LoadedProjector")
    p_source = projector.value
    if tensor_sha256(p_source) != projector._tensor_sha256:
        raise ValueError("matched-P projector was modified after loading")
    _validate_projector_geometry(p_source)
    if recipient.shape != donor.shape or recipient.shape[-1] != p_source.shape[-1]:
        raise ValueError("projected splice dimensions are invalid")
    p = p_source.to(device=recipient.device, dtype=recipient.dtype)
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


def _freeze_record(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_record(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_record(item) for item in value)
    return value


def load_probe_manifest(path: Path | str, *, required_fields: Sequence[str] = ("h_p",)) -> VerifiedProbeBundle:
    """Return an opaque bundle after verifying manifest-listed probes and estimator records."""
    manifest_path = Path(path)
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        files = manifest["files"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError("probe manifest schema is invalid") from exc
    if not isinstance(files, dict) or set(manifest) != {"schema_version", "files"} or not files:
        raise ValueError("probe manifest schema is invalid")
    rows: list[Mapping[str, np.ndarray]] = []
    estimator_records: list[Mapping[str, Any]] = []
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
                missing = [field for field in (*required_fields, "estimator_records") if field not in handle]
                if missing:
                    raise ValueError(f"probe file missing fields: {missing}")
                rows.append({field: np.asarray(handle[field]) for field in required_fields})
                encoded_records = handle["estimator_records"][()]
                if isinstance(encoded_records, bytes):
                    encoded_records = encoded_records.decode()
                parsed_records = json.loads(encoded_records)
                if not isinstance(parsed_records, list) or not all(isinstance(record, dict) for record in parsed_records):
                    raise ValueError("probe estimator records are invalid")
                estimator_records.extend(_freeze_record(record) for record in parsed_records)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("manifest-listed probe is not valid HDF5") from exc
    return VerifiedProbeBundle(tuple(rows), tuple(estimator_records), hashlib.sha256(manifest_bytes).hexdigest(), _VERIFIED_BUNDLE_SEAL)


def exploratory_pairing(recipients: Sequence[Mapping[str, Any]], donors: Sequence[Mapping[str, Any]], *, ratio: float = 1.0, seed: int = PAIRING_SEED) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Unrestricted helper; its output is not a confirmatory pairing artifact."""
    rng = np.random.default_rng(seed)
    pairs = []
    for recipient in sorted(recipients, key=lambda x: str(x["goal_id"])):
        target = np.asarray(recipient["delta_m"], dtype=float) * rescale_mask(ratio, dtype=torch.float64).numpy()
        candidates = sorted((d for d in donors if d.get("goal_id") == recipient.get("goal_id") and d.get("p") == recipient.get("p") and d.get("nuisance") == recipient.get("nuisance")), key=lambda d: str(d.get("id", "")))
        if not candidates:
            raise ValueError("no nuisance-matched donor")
        scored = [(float(np.abs(np.asarray(d["delta_m"], dtype=float) - target).sum()), float(rng.random()), str(d.get("id", "")), d) for d in candidates]
        pairs.append((recipient, min(scored, key=lambda x: x[:3])[3]))
    return pairs


def deterministic_pairing(recipients: Sequence[Mapping[str, Any]], donors: Sequence[Mapping[str, Any]], *, ratio: float = 1.0) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Confirmatory 20260723 pairing: exactly 100 recipients with unique goals."""
    if len(recipients) != N_PAIRS_PER_RUN or len({r.get("goal_id") for r in recipients}) != N_PAIRS_PER_RUN or any(r.get("goal_id") is None for r in recipients):
        raise ValueError("confirmatory pairing requires exactly 100 recipients with unique goals")
    return exploratory_pairing(recipients, donors, ratio=ratio, seed=PAIRING_SEED)


def ranking_change(baseline_q: Sequence[float], patched_q: Sequence[float]) -> dict[str, float]:
    base, patch = np.asarray(baseline_q, dtype=float), np.asarray(patched_q, dtype=float)
    if base.ndim != 1 or patch.shape != base.shape or len(base) < 2:
        raise ValueError("Q rankings require matching vectors with at least two actions")
    tau = kendalltau(base, patch).statistic
    return {"kendall_tau_change": float(1.0 - (0.0 if np.isnan(tau) else tau)), "top1_flip": float(np.argmax(base) != np.argmax(patch))}


def _validate_estimator_inputs(bundle: VerifiedProbeBundle) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    if not isinstance(bundle, VerifiedProbeBundle) or bundle._seal is not _VERIFIED_BUNDLE_SEAL:
        raise TypeError("confirmatory estimator requires a VerifiedProbeBundle")
    real_rows = [row for row in bundle.estimator_records if row.get("arm") == "real"]
    null_rows = [row for row in bundle.estimator_records if row.get("arm") == "null"]
    if len(real_rows) != len(null_rows) or not real_rows or len(real_rows) + len(null_rows) != len(bundle.estimator_records):
        raise ValueError("bundle requires nonempty matched real and null estimator records")
    required = {"pair_id", "goal_id", "run", "operator", "baseline_q", "patched_q"}
    real_keys, null_keys = [], []
    for arm, rows, keys in (("real", real_rows, real_keys), ("null", null_rows, null_keys)):
        for row in rows:
            if not required <= set(row) or any(row[field] is None for field in required):
                raise ValueError(f"{arm} rows require pair_id, goal_id, run, operator, baseline_q, and patched_q")
            keys.append((row["run"], row["pair_id"], row["goal_id"], row["operator"]))
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate {arm} pair identity")
    if set(real_keys) != set(null_keys):
        raise ValueError("real and null rows require one-to-one matching identities")
    for run in {key[0] for key in real_keys}:
        goals = [key[2] for key in real_keys if key[0] == run]
        if len(goals) != N_PAIRS_PER_RUN or len(set(goals)) != N_PAIRS_PER_RUN:
            raise ValueError("each run requires exactly 100 unique goals")
    return real_rows, null_rows


def estimate_from_bundle(bundle: VerifiedProbeBundle, *, bootstrap_draws: int = 10_000) -> dict[str, Any]:
    """AMD-2 estimator consuming only records sealed in a VerifiedProbeBundle."""
    real_rows, null_rows = _validate_estimator_inputs(bundle)
    null_by_key = {(n["run"], n["pair_id"], n["goal_id"], n["operator"]): n for n in null_rows}
    by_run: dict[Any, dict[str, list[float]]] = {}
    for r in real_rows:
        n = null_by_key[(r["run"], r["pair_id"], r["goal_id"], r["operator"])]
        effects = {key: ranking_change(r["baseline_q"], r["patched_q"])[key] - ranking_change(n["baseline_q"], n["patched_q"])[key] for key in ("kendall_tau_change", "top1_flip")}
        bucket = by_run.setdefault(r["run"], {key: [] for key in effects})
        for key, value in effects.items():
            bucket[key].append(value)
    run_means = {key: [float(np.mean(by_run[run][key])) for run in sorted(by_run)] for key in next(iter(by_run.values()))}
    import sys
    scripts_dir = str(Path(__file__).resolve().parents[3] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from sprint_stats import seed_cluster_bootstrap
    return {"run_means": run_means, "bca": {key: seed_cluster_bootstrap(values, draws=bootstrap_draws) for key, values in run_means.items()}, "provenance": {"probe_manifest_sha256": bundle.manifest_sha256, "pairing_seed": PAIRING_SEED}}


def provenance_record(*, arm: str, ckpt_sha256: str, split_sha256: str, claim_sha256: str, operator: str, parameter_pre: torch.Tensor, parameter_post: torch.Tensor, probe_manifest_sha256: str) -> dict[str, str]:
    return {"arm": arm, "ckpt_sha256": ckpt_sha256, "split_sha256": split_sha256, "claim_sha256": claim_sha256, "operator": operator, "parameter_pre_sha256": tensor_sha256(parameter_pre), "parameter_post_sha256": tensor_sha256(parameter_post), "probe_manifest_sha256": probe_manifest_sha256, "pairing_seed": str(PAIRING_SEED), "pairing_cardinality": str(N_PAIRS_PER_RUN), "interpretation": "response-conditioned h_p mediation"}
