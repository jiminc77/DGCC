"""Durable one-shot authorization primitives for the sprint held-out split."""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PRIMITIVE_SCHEMA_VERSION = 1
CANONICAL_SPLIT_SHA256 = "76335ae50efd8164df1f8e241ae69aa30685f201aa6f0554d4a5b077cc1e2754"
PATCH_EVAL_SPLIT_SHA256 = "7afba6d23c0a5d77205549b6f8b0b11a124a7c363dbf4fcd86e4fac7d59e8d72"

# Canonical AMD-3 paired seed set; verdict 5029426419.
AMD3_PAIRED_SEEDS = frozenset({0, 1, 2, 3, 4, 6, 7})
# This is intentionally anchored at the installed source tree, never at CWD.
REPO_ROOT = Path(__file__).resolve().parents[3]
PATCH_EVAL_SPLIT_PATH = REPO_ROOT / "src/dgcc/tasks/splits/t2_patch_eval_v1.json"
# Canonical summaries are derivable from the durable episode rows alone.
CANONICAL_SUMMARY_KEYS = frozenset({
    "n_episodes", "success_rate", "mean_return", "mean_final_d",
    "mean_d_at_done", "mean_min_d", "mean_d_shape_at_done",
    "per_template_success", "per_template_episodes",
    "overestimation_gap_mean", "overestimation_gap_p95",
    "eval_wall_guard_rate",
})
CANONICAL_SPLIT_PATH = REPO_ROOT / "src/dgcc/tasks/splits/t2_sprint_heldout_v1.json"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_ARMS = frozenset({"bb", "v1", "matched", "random"})
_CLAIM_KEYS = frozenset({"schema_version", "claim_before_load", "timestamp", "pid", "run_tag", "arm", "ckpt_sha256", "split_sha256", "seed", "config_sha256", "selection_manifest", "selection_manifest_sha256", "episode_index_start", "episode_namespace", "n_goals", "disposition_receipt_sha256", "legacy_claim_sha256", "generation"})
# This registry is deliberately module-private.  Reflection in the same process is
# out of scope; normal callers can only consume an authority we issued.
_ISSUED_CAPABILITIES: dict[int, tuple["ClaimCapability", str]] = {}
_ISSUED_PATCH_CAPABILITIES: dict[int, tuple["ClaimCapability", str]] = {}

class SprintClaimError(RuntimeError):
    """A held-out evaluation is not authorized."""

class ClaimCapability:
    """Opaque, module-issued authority to consume one durable claim."""
    __slots__ = ("_path",)
    def __init__(self, path: Path) -> None:
        self._path = path

def utc_now() -> str: return datetime.now(timezone.utc).isoformat()

def sha256_file(path: Path) -> str:
    path = Path(path)
    if path.is_symlink():
        raise SprintClaimError(f"symlinked path is not permitted: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)

def _arm(value: str) -> str:
    arm = str(value).lower()
    if arm not in _ALLOWED_ARMS: raise SprintClaimError("arm must be one of bb, v1, matched, random")
    return arm

def validate_checkpoint_arm(path: Path, arm: str) -> None:
    """Bind declared treatment arm to checkpoint metadata before split access."""
    arm = _arm(arm)
    try:
        import torch
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except Exception as exc:
        raise SprintClaimError("checkpoint payload cannot be inspected for sprint_arm") from exc
    declared = payload.get("sprint_arm") if isinstance(payload, dict) else None
    if arm == "bb":
        if declared is not None: raise SprintClaimError("BB declaration requires a baseline checkpoint without sprint_arm")
    elif declared != arm:
        raise SprintClaimError("checkpoint sprint_arm does not match declared arm")

def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result: raise SprintClaimError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

def _json_bytes(raw: bytes, what: str) -> Any:
    try: return json.loads(raw, object_pairs_hook=_reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, SprintClaimError) as exc: raise SprintClaimError(f"{what} is invalid JSON") from exc
def json_file(path: Path, what: str) -> tuple[Any, str]:
    """Read untrusted JSON exactly once, rejecting aliases and duplicate keys."""
    path = Path(path)
    if path.is_symlink():
        raise SprintClaimError(f"{what} must not be a symlink")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SprintClaimError(f"{what} is unreadable") from exc
    return _json_bytes(raw, what), hashlib.sha256(raw).hexdigest()

def canonical_claim_path(run_tag: str, arm: str, generation: str | None = None) -> Path:
    suffix = "" if generation is None else f"_{generation}"
    return REPO_ROOT / "outputs/metrics" / f"p1_{_arm(arm)}_sprint_heldout_{run_tag}{suffix}_claim.json"
def canonical_result_path(run_tag: str, arm: str, generation: str | None = None) -> Path:
    """Canonical published-result path for a claimed held-out evaluation."""
    suffix = "" if generation is None else f"_{generation}"
    return REPO_ROOT / "outputs/metrics" / f"p1_{_arm(arm)}_sprint_heldout_{run_tag}{suffix}.json"

def canonical_raw_path(run_tag: str, arm: str, generation: str | None = None) -> Path:
    suffix = "" if generation is None else f"_{generation}"
    return REPO_ROOT / "outputs/metrics" / f"p1_{_arm(arm)}_sprint_heldout_{run_tag}{suffix}.raw.json.gz"

def canonical_metric_lock_path() -> Path:
    """The sole durable BB metric-lock location."""
    return REPO_ROOT / "outputs/metrics/sprint_metric.lock"

def canonical_patch_claim_path(run_tag: str, arm: str) -> Path:
    return REPO_ROOT / "outputs/metrics" / f"p1_{_arm(arm)}_patch_eval_{run_tag}_claim.json"

def canonical_patch_result_path(run_tag: str, arm: str) -> Path:
    return REPO_ROOT / "outputs/metrics" / f"p1_{_arm(arm)}_patch_eval_{run_tag}.json"

def _validate_patch_claim(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("split_sha256") != PATCH_EVAL_SPLIT_SHA256:
        raise SprintClaimError("patch claim split_sha256 is invalid")
    baseline = dict(value)
    baseline["split_sha256"] = CANONICAL_SPLIT_SHA256
    _validate_claim(baseline)
    return dict(value)

def acquire_patch_claim(claim_path: Path, payload: Mapping[str, Any]) -> ClaimCapability:
    """Create exclusive patch-evaluation authority before opening its split."""
    body = dict(payload)
    body["split_sha256"] = payload.get("split_sha256")
    body.update(schema_version=PRIMITIVE_SCHEMA_VERSION, claim_before_load=True)
    body.setdefault("timestamp", utc_now())
    body.setdefault("pid", os.getpid())
    body["arm"] = _arm(body.get("arm", ""))
    _validate_patch_claim(body)
    expected = canonical_patch_claim_path(body["run_tag"], body["arm"])
    if Path(claim_path).absolute() != expected.absolute() or Path(claim_path).is_symlink():
        raise SprintClaimError("patch claim path must be canonical and not a symlink")
    expected.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(expected, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise SprintClaimError(f"patch-eval claim already exists: {expected}; do NOT delete it to re-run") from exc
    try:
        os.write(fd, (json.dumps(body, indent=1, sort_keys=True) + "\n").encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_dir(expected.parent)
    capability = ClaimCapability(expected)
    _ISSUED_PATCH_CAPABILITIES[id(capability)] = (capability, sha256_file(expected))
    return capability

def consume_patch_claim_and_load_split(capability: ClaimCapability, split_path: Path, *, access_log: Path) -> dict[str, Any]:
    """Consume a patch-only claim and record the sole authorized split access."""
    issued = _ISSUED_PATCH_CAPABILITIES.pop(id(capability), None) if isinstance(capability, ClaimCapability) else None
    if issued is None or issued[0] is not capability:
        raise SprintClaimError("patch split consumption requires an unconsumed acquired claim capability")
    claim_path, issued_digest = capability._path, issued[1]
    try:
        if claim_path.is_symlink():
            raise SprintClaimError("claim must not be a symlink")
        raw_claim = claim_path.read_bytes()
        value = _validate_patch_claim(_json_bytes(raw_claim, "patch claim"))
    except (OSError, SprintClaimError) as exc:
        raise SprintClaimError("patch claim is unreadable or modified") from exc
    if hashlib.sha256(raw_claim).hexdigest() != issued_digest:
        raise SprintClaimError("patch claim was modified after capability issuance")
    requested = Path(split_path)
    if requested.is_symlink() or requested.absolute() != PATCH_EVAL_SPLIT_PATH.absolute() or requested.resolve(strict=True) != PATCH_EVAL_SPLIT_PATH.resolve(strict=True):
        raise SprintClaimError("only the canonical non-symlink patch split is permitted")
    raw = PATCH_EVAL_SPLIT_PATH.read_bytes()
    measured = hashlib.sha256(raw).hexdigest()
    if measured != PATCH_EVAL_SPLIT_SHA256:
        raise SprintClaimError("patch split bytes do not match trusted digest")
    record_access(access_log, "patch_rollout", run_tag=value["run_tag"], arm=value["arm"], split_sha256=measured, claim_sha256=issued_digest)
    payload = _json_bytes(raw, "canonical patch split")
    if payload.get("n_goals") != 100 or len(payload.get("specs", [])) != 100:
        raise SprintClaimError("patch split must contain exactly 100 goals")
    return payload

def validate_claim_payload(value: Any) -> dict[str, Any]:
    """Public trust-boundary wrapper for canonical claim validation."""
    return _validate_claim(value)

def _validate_claim(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) - _CLAIM_KEYS: raise SprintClaimError("claim schema is invalid")
    required = {"schema_version", "claim_before_load", "timestamp", "pid", "run_tag", "arm", "ckpt_sha256", "split_sha256", "seed", "config_sha256", "selection_manifest", "selection_manifest_sha256", "episode_namespace", "n_goals"}
    if not required <= set(value) or value["schema_version"] != PRIMITIVE_SCHEMA_VERSION or value["claim_before_load"] is not True: raise SprintClaimError("claim schema is invalid")
    if not isinstance(value["run_tag"], str) or not value["run_tag"] or not isinstance(value["timestamp"], str) or not isinstance(value["pid"], int) or not isinstance(value["seed"], int) or value["episode_namespace"] != 97_001 or value["n_goals"] != 100: raise SprintClaimError("claim schema is invalid")
    _arm(value["arm"])
    if not _HEX64.fullmatch(str(value["ckpt_sha256"])) or value["split_sha256"] != CANONICAL_SPLIT_SHA256 or not isinstance(value["selection_manifest"], str) or not value["selection_manifest"]: raise SprintClaimError("claim schema is invalid")
    for key in ("config_sha256", "selection_manifest_sha256"):
        if not _HEX64.fullmatch(str(value[key])): raise SprintClaimError("claim schema is invalid")
    reeval = "generation" in value or "legacy_claim_sha256" in value or "disposition_receipt_sha256" in value
    if reeval and (not isinstance(value.get("generation"), str) or not value["generation"] or not _HEX64.fullmatch(str(value.get("legacy_claim_sha256"))) or not _HEX64.fullmatch(str(value.get("disposition_receipt_sha256")))): raise SprintClaimError("claim schema is invalid")
    return dict(value)

def acquire_claim(claim_path: Path, payload: Mapping[str, Any]) -> ClaimCapability:
    """Create exclusive authorization before any split open occurs."""
    body = dict(payload); body.update(schema_version=PRIMITIVE_SCHEMA_VERSION, claim_before_load=True)
    body.setdefault("timestamp", utc_now()); body.setdefault("pid", os.getpid()); body["arm"] = _arm(body.get("arm", ""))
    _validate_claim(body)
    expected = canonical_claim_path(body["run_tag"], body["arm"], body.get("generation"))
    if Path(claim_path).absolute() != expected.absolute() or Path(claim_path).is_symlink(): raise SprintClaimError("claim path must be canonical and not a symlink")
    expected.parent.mkdir(parents=True, exist_ok=True)
    try: fd = os.open(expected, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc: raise SprintClaimError(f"held-out claim already exists: {expected}; do NOT delete it to re-run") from exc
    try: os.write(fd, (json.dumps(body, indent=1, sort_keys=True) + "\n").encode()); os.fsync(fd)
    finally: os.close(fd)
    _fsync_dir(expected.parent)
    capability = ClaimCapability(expected)
    _ISSUED_CAPABILITIES[id(capability)] = (capability, sha256_file(expected))
    return capability

def record_access(log_path: Path, purpose: str, **metadata: Any) -> None:
    if not purpose or not _HEX64.fullmatch(str(metadata.get("claim_sha256", ""))): raise SprintClaimError("access record requires purpose and claim_sha256")
    log_path = Path(log_path); log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": utc_now(), "pid": os.getpid(), "purpose": purpose, **metadata}, sort_keys=True) + "\n"); handle.flush(); os.fsync(handle.fileno())

def consume_claim_and_load_split(capability: ClaimCapability, split_path: Path, *, access_log: Path | None = None, purpose: str = "final_eval") -> dict[str, Any]:
    """The sole production split consumer; arbitrary mappings are not authority."""
    issued = _ISSUED_CAPABILITIES.pop(id(capability), None) if isinstance(capability, ClaimCapability) else None
    if issued is None or issued[0] is not capability: raise SprintClaimError("split consumption requires an unconsumed acquired claim capability")
    claim_path, issued_digest = capability._path, issued[1]
    try:
        if claim_path.is_symlink():
            raise SprintClaimError("claim must not be a symlink")
        raw_claim = claim_path.read_bytes(); value = _validate_claim(_json_bytes(raw_claim, "claim"))
    except (OSError, SprintClaimError) as exc: raise SprintClaimError("claim is unreadable or modified") from exc
    if hashlib.sha256(raw_claim).hexdigest() != issued_digest: raise SprintClaimError("claim was modified after capability issuance")
    requested, canonical = Path(split_path), CANONICAL_SPLIT_PATH
    if requested.is_symlink() or requested.absolute() != canonical.absolute() or requested.resolve(strict=True) != canonical.resolve(strict=True):
        raise SprintClaimError("only the canonical non-symlink sprint split is permitted")
    with canonical.open("rb") as handle: raw = handle.read()
    measured = hashlib.sha256(raw).hexdigest()
    if measured != CANONICAL_SPLIT_SHA256: raise SprintClaimError("canonical split bytes do not match trusted digest")
    record_access(access_log or REPO_ROOT / "outputs/metrics/t2_sprint_heldout_access.log", purpose, run_tag=value["run_tag"], arm=value["arm"], split_sha256=measured, claim_sha256=issued_digest)
    payload = _json_bytes(raw, "canonical split")
    if payload.get("n_goals") != 100 or len(payload.get("specs", [])) != 100: raise SprintClaimError("sprint split must contain exactly 100 goals")
    return payload

def parse_disposition_receipt(path: Path, *, legacy_claim_sha256: str, run_tag: str) -> tuple[dict[str, Any], str]:
    try: receipt, digest = json_file(path, "disposition receipt")
    except SprintClaimError as exc: raise SprintClaimError("disposition receipt is invalid") from exc
    required = {"schema_version", "legacy_claim_sha256", "run_tag", "decision", "decided_by", "decided_at"}
    if not isinstance(receipt, dict) or set(receipt) != required or receipt["schema_version"] != 1 or receipt["legacy_claim_sha256"] != legacy_claim_sha256 or receipt["run_tag"] != run_tag or receipt["decision"] not in {"allow_reevaluation", "keep_legacy"} or not isinstance(receipt["decided_by"], str) or not receipt["decided_by"] or not isinstance(receipt["decided_at"], str) or not receipt["decided_at"]: raise SprintClaimError("disposition receipt schema or identity is invalid")
    return receipt, digest

def atomic_publish(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle: json.dump(payload, handle, indent=1, sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp_name, path); _fsync_dir(path.parent)
    finally:
        if os.path.exists(tmp_name): os.unlink(tmp_name)

def probe_manifest_register(manifest_path: Path, file_path: Path, meta: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path, source = Path(manifest_path), Path(file_path)
    if source.is_symlink() or manifest_path.is_symlink():
        raise SprintClaimError("probe and manifest paths must not be symlinks")
    canonical = source.resolve(strict=True)
    if not canonical.is_file(): raise FileNotFoundError(canonical)
    entry = {"path": str(canonical), "sha256": sha256_file(canonical), "size": canonical.stat().st_size, **dict(meta)}
    if not entry.get("production_goal"): raise ValueError("probe metadata requires production_goal")
    lock_path = manifest_path.with_name(manifest_path.name + ".lock"); lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            manifest = json_file(manifest_path, "probe manifest")[0] if manifest_path.exists() else {"schema_version": PRIMITIVE_SCHEMA_VERSION, "files": {}}
            if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "files"} or not isinstance(manifest["files"], dict): raise SprintClaimError("probe manifest schema is invalid")
            for old_key, old in manifest["files"].items():
                if not isinstance(old, dict) or not isinstance(old.get("path"), str):
                    raise SprintClaimError("probe manifest schema is invalid")
                try:
                    old_canonical = Path(old["path"]).resolve(strict=True)
                except OSError as exc:
                    raise SprintClaimError("probe manifest schema is invalid") from exc
                if old_canonical == canonical:
                    raise SprintClaimError(f"probe manifest entry is immutable: {canonical}")
                if old_key == entry["sha256"] and old_canonical != canonical: raise SprintClaimError("probe aliases are forbidden")
            if entry["sha256"] not in manifest["files"]: manifest["files"][entry["sha256"]] = entry; atomic_publish(manifest_path, manifest)
            return manifest
        finally: fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

# Issued metric lock trust anchor: sha256 of the immutable published lock file
# (commit 44ad038, G9a publication). Consumers refuse any other lock bytes.
PUBLISHED_METRIC_LOCK_SHA256 = "7cb96288c9b27290674488c7ae34c854efe82f0e04628af26b6e93166a562122"


def require_metric_lock(lock_path: Path | None, arm: str) -> None:
    if _arm(arm) == "bb": return
    if lock_path is None: raise SprintClaimError(f"arm {arm!r} requires --lock before split load")
    canonical = canonical_metric_lock_path()
    try:
        resolved = Path(lock_path).resolve(strict=True)
    except OSError as exc:
        raise SprintClaimError(f"metric lock is invalid: {lock_path}") from exc
    if Path(lock_path).is_symlink() or resolved != canonical.resolve():
        raise SprintClaimError("metric lock must be the canonical issued lock path")
    try: value, digest = json_file(resolved, "metric lock")
    except (OSError, SprintClaimError) as exc: raise SprintClaimError(f"metric lock is invalid: {lock_path}") from exc
    if digest != PUBLISHED_METRIC_LOCK_SHA256:
        raise SprintClaimError("metric lock does not match the issued trust anchor")
    required = {"schema_version", "endpoint", "aggregate", "created_at", "bb_claim_sha256", "bb_claim_audit", "primitive_version"}
    if not isinstance(value, dict) or set(value) != required or value["schema_version"] != 1 or value["endpoint"] not in {"success_rate", "return"} or not isinstance(value["aggregate"], (int, float)) or isinstance(value["aggregate"], bool) or not value["created_at"] or not value["primitive_version"]:
        raise SprintClaimError("metric lock schema is invalid")
    hashes = value["bb_claim_sha256"]
    audits = value["bb_claim_audit"]
    if not isinstance(hashes, list) or len(hashes) != len(AMD3_PAIRED_SEEDS) or len(set(hashes)) != len(AMD3_PAIRED_SEEDS) or any(not isinstance(x, str) or not _HEX64.fullmatch(x) for x in hashes):
        raise SprintClaimError("metric lock schema is invalid")
    if not isinstance(audits, list) or len(audits) != len(AMD3_PAIRED_SEEDS):
        raise SprintClaimError("metric lock schema is invalid")
    expected_kinds = {0: "legacy_bundle", 1: "legacy_bundle", 2: "legacy_bundle", 3: "canonical", 4: "canonical", 6: "canonical", 7: "canonical"}
    if not all(
        isinstance(row, dict) and set(row) == {"seed", "kind", "claim_sha256"} and row["seed"] in expected_kinds and row["kind"] == expected_kinds[row["seed"]] and row["claim_sha256"] in hashes
        for row in audits
    ) or {row["seed"] for row in audits} != set(AMD3_PAIRED_SEEDS) or {row["claim_sha256"] for row in audits} != set(hashes):
        raise SprintClaimError("metric lock schema is invalid")

def _is_finite_json(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, int) and not isinstance(value, bool):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_finite_json(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_finite_json(item) for key, item in value.items())
    return False


def canonicalize_episode_ids(
    episodes: list[dict[str, Any]], episode_namespace: int
) -> list[dict[str, Any]]:
    """Assign producer-owned, contiguous episode identities in output order."""
    if isinstance(episode_namespace, bool) or not isinstance(episode_namespace, int):
        raise SprintClaimError("episode namespace must be an integer")
    for ordinal, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise SprintClaimError("evaluation episodes must be objects")
        episode["episode_id"] = episode_namespace + ordinal
    return episodes


def _summary_aggregates(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(episodes)
    mean = lambda key: sum(episode[key] for episode in episodes) / count
    gaps = sorted(
        episode["q_first"] - episode["discounted_return"]
        for episode in episodes if episode["q_first"] is not None
    )
    if gaps:
        position = (len(gaps) - 1) * 0.95
        lower, upper = math.floor(position), math.ceil(position)
        gap_p95 = gaps[lower] + (gaps[upper] - gaps[lower]) * (position - lower)
    else:
        gap_p95 = None
    templates: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        templates.setdefault(episode["init_template"], []).append(episode)
    return {
        "n_episodes": count,
        "success_rate": mean("success"),
        "mean_return": mean("return"),
        "mean_final_d": mean("final_d"),
        "mean_d_at_done": mean("d_at_done"),
        "mean_min_d": mean("min_d"),
        "mean_d_shape_at_done": mean("d_shape_at_done"),
        "per_template_success": {key: sum(row["success"] for row in rows) / len(rows) for key, rows in templates.items()},
        "per_template_episodes": {key: len(rows) for key, rows in templates.items()},
        "overestimation_gap_mean": sum(gaps) / len(gaps) if gaps else None,
        "overestimation_gap_p95": gap_p95,
        "eval_wall_guard_rate": mean("eval_wall_guard"),
    }


def _summary_matches_episodes(summary: Mapping[str, Any], episodes: list[dict[str, Any]]) -> bool:
    for key, expected in _summary_aggregates(episodes).items():
        actual = summary[key]
        if expected is None:
            if actual is not None:
                return False
        elif isinstance(expected, dict):
            if actual != expected:
                return False
        elif not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
            return False
    return True

def _is_canonical_result(body: Any, *, claim: Mapping[str, Any], claim_sha256: str) -> bool:
    required = {
        "generated_at", "run_tag", "arm", "seed", "config_sha256", "ckpt_sha256",
        "split_sha256", "claim_sha256", "selection_manifest",
        "selection_manifest_sha256", "episode_namespace", "summary", "episodes",
    }
    episode_required = {
        "episode_id", "goal_id", "goal_label", "init_template", "success", "steps",
        "return", "discounted_return", "final_d", "d_at_done", "d_at_done_fallback",
        "d_steps", "min_d", "d_initial", "d_shape_initial", "d_shape_at_done",
        "q_first", "eval_wall_guard", "discard_exposure",
    }
    if not isinstance(body, dict) or not required <= set(body):
        return False
    if any(body[key] != claim[key] for key in ("run_tag", "arm", "seed", "config_sha256", "ckpt_sha256", "split_sha256", "selection_manifest", "selection_manifest_sha256", "episode_namespace")) or body["claim_sha256"] != claim_sha256:
        return False
    if not isinstance(body["episodes"], list) or len(body["episodes"]) != 200:
        return False
    if not isinstance(body["summary"], dict) or set(body["summary"]) != CANONICAL_SUMMARY_KEYS or not _is_finite_json(body["summary"]):
        return False
    summary = body["summary"]
    numeric_summary = (
        "success_rate", "mean_return", "mean_final_d", "mean_d_at_done",
        "mean_min_d", "mean_d_shape_at_done", "eval_wall_guard_rate",
    )
    nullable_numeric_summary = ("overestimation_gap_mean", "overestimation_gap_p95")
    if not all(isinstance(summary[key], (int, float)) and not isinstance(summary[key], bool) and math.isfinite(summary[key]) for key in numeric_summary):
        return False
    if not all(summary[key] is None or (isinstance(summary[key], (int, float)) and not isinstance(summary[key], bool) and math.isfinite(summary[key])) for key in nullable_numeric_summary):
        return False
    if not isinstance(summary["n_episodes"], int) or isinstance(summary["n_episodes"], bool) or summary["n_episodes"] != 200:
        return False
    if not isinstance(summary["per_template_success"], dict) or not isinstance(summary["per_template_episodes"], dict):
        return False
    namespace = body["episode_namespace"]
    if not isinstance(namespace, int) or isinstance(namespace, bool):
        return False
    episode_ids: set[int] = set()
    goal_counts: dict[str, int] = {}
    numeric_episode = ("return", "discounted_return", "final_d", "d_at_done", "min_d", "d_initial", "d_shape_initial", "d_shape_at_done")
    for episode in body["episodes"]:
        if not isinstance(episode, dict) or not episode_required <= set(episode) or not _is_finite_json(episode):
            return False
        episode_id = episode["episode_id"]
        if not isinstance(episode_id, int) or isinstance(episode_id, bool) or not namespace <= episode_id < namespace + 200 or episode_id in episode_ids:
            return False
        if not isinstance(episode["goal_id"], str) or not episode["goal_id"] or not isinstance(episode["goal_label"], str) or not isinstance(episode["init_template"], str) or not isinstance(episode["success"], bool) or not isinstance(episode["steps"], int) or isinstance(episode["steps"], bool) or not isinstance(episode["discard_exposure"], int) or isinstance(episode["discard_exposure"], bool) or not isinstance(episode["d_at_done_fallback"], bool) or not isinstance(episode["eval_wall_guard"], bool) or not isinstance(episode["d_steps"], list):
            return False
        if not all(isinstance(episode[key], (int, float)) and not isinstance(episode[key], bool) and math.isfinite(episode[key]) for key in numeric_episode):
            return False
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in episode["d_steps"]):
            return False
        if episode["q_first"] is not None and (not isinstance(episode["q_first"], (int, float)) or isinstance(episode["q_first"], bool) or not math.isfinite(episode["q_first"])):
            return False
        episode_ids.add(episode_id)
        goal_counts[episode["goal_id"]] = goal_counts.get(episode["goal_id"], 0) + 1
    if len(goal_counts) != claim["n_goals"] or any(count != 2 for count in goal_counts.values()):
        return False
    if not _summary_matches_episodes(summary, body["episodes"]):
        return False
    if "generation" in claim and any(body.get(key) != claim[key] for key in ("generation", "disposition_receipt_sha256")):
        return False
    return all(isinstance(body[key], str) and body[key] for key in ("generated_at", "config_sha256", "ckpt_sha256", "selection_manifest", "selection_manifest_sha256"))
def is_canonical_result(body: Any, *, claim: Mapping[str, Any], claim_sha256: str) -> bool:
    """Public trust-boundary wrapper; audits every episode and recomputed summary."""
    return _is_canonical_result(body, claim=claim, claim_sha256=claim_sha256)

def audit_claims(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for claim in Path(directory).glob("*claim*.json"):
        if "_patch_eval_" in claim.name:
            continue  # patch-eval claims live in their own namespace/audit (canonical_patch_claim_path)
        try:
            data, digest = json_file(claim, "claim")
            data = _validate_claim(data)
        except SprintClaimError as exc:
            rows.append({"schema_version": 1, "status": "malformed_claim", "claim": str(claim), "error": str(exc)}); continue
        tag, arm = data["run_tag"], data["arm"]
        candidates = [Path(directory) / f"p1_{arm}_sprint_heldout_{tag}.json", Path(directory) / f"p1_t2_sprint_heldout_{tag}.json"]
        valid = False
        for result in candidates:
            try:
                body, _ = json_file(result, "result")
                valid = _is_canonical_result(body, claim=data, claim_sha256=digest)
            except SprintClaimError: pass
            if valid: break
        if not valid: rows.append({"schema_version": 1, "status": "needs_human_disposition", "claim": str(claim), "claim_sha256": digest, "run_tag": tag, "arm": arm, "re_evaluation_permitted": False})
    return rows
