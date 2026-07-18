"""Durable one-shot authorization primitives for the sprint held-out split."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PRIMITIVE_SCHEMA_VERSION = 1
CANONICAL_SPLIT_SHA256 = "76335ae50efd8164df1f8e241ae69aa30685f201aa6f0554d4a5b077cc1e2754"
# This is intentionally anchored at the installed source tree, never at CWD.
REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_SPLIT_PATH = REPO_ROOT / "src/dgcc/tasks/splits/t2_sprint_heldout_v1.json"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_ARMS = frozenset({"bb", "v1", "matched", "random"})
_CLAIM_KEYS = frozenset({"schema_version", "claim_before_load", "timestamp", "pid", "run_tag", "arm", "ckpt_sha256", "split_sha256", "seed", "config_sha256", "selection_manifest", "selection_manifest_sha256", "episode_index_start", "disposition_receipt_sha256"})
_CAPABILITY_TOKEN = object()

class SprintClaimError(RuntimeError):
    """A held-out evaluation is not authorized."""

class ClaimCapability:
    """Opaque, module-issued authority to consume one durable claim."""
    __slots__ = ("_path", "_token")
    def __init__(self, path: Path, token: object) -> None:
        self._path, self._token = path, token

def utc_now() -> str: return datetime.now(timezone.utc).isoformat()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""): digest.update(chunk)
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

def _validate_claim(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) - _CLAIM_KEYS:
        raise SprintClaimError("claim schema is invalid")
    required = {"schema_version", "claim_before_load", "timestamp", "pid", "run_tag", "arm", "ckpt_sha256", "split_sha256"}
    if set(value) & required != required or value["schema_version"] != PRIMITIVE_SCHEMA_VERSION or value["claim_before_load"] is not True:
        raise SprintClaimError("claim schema is invalid")
    if not isinstance(value["run_tag"], str) or not value["run_tag"] or not isinstance(value["timestamp"], str) or not isinstance(value["pid"], int): raise SprintClaimError("claim schema is invalid")
    _arm(value["arm"])
    if not _HEX64.fullmatch(str(value["ckpt_sha256"])) or value["split_sha256"] != CANONICAL_SPLIT_SHA256: raise SprintClaimError("claim schema is invalid")
    for key in ("config_sha256", "selection_manifest_sha256", "disposition_receipt_sha256"):
        if key in value and not _HEX64.fullmatch(str(value[key])): raise SprintClaimError("claim schema is invalid")
    return dict(value)

def acquire_claim(claim_path: Path, payload: Mapping[str, Any]) -> ClaimCapability:
    """Create exclusive authorization before any split open occurs."""
    claim_path = Path(claim_path).resolve()
    body = dict(payload); body.update(schema_version=PRIMITIVE_SCHEMA_VERSION, claim_before_load=True)
    body.setdefault("timestamp", utc_now()); body.setdefault("pid", os.getpid()); body["arm"] = _arm(body.get("arm", ""))
    _validate_claim(body)
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    try: fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc: raise SprintClaimError(f"held-out claim already exists: {claim_path}; do NOT delete it to re-run") from exc
    try:
        os.write(fd, (json.dumps(body, indent=1, sort_keys=True) + "\n").encode()); os.fsync(fd)
    finally: os.close(fd)
    _fsync_dir(claim_path.parent)
    return ClaimCapability(claim_path, _CAPABILITY_TOKEN)

def record_access(log_path: Path, purpose: str, **metadata: Any) -> None:
    if not purpose or not _HEX64.fullmatch(str(metadata.get("claim_sha256", ""))): raise SprintClaimError("access record requires purpose and claim_sha256")
    log_path = Path(log_path); log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": utc_now(), "pid": os.getpid(), "purpose": purpose, **metadata}, sort_keys=True) + "\n"); handle.flush(); os.fsync(handle.fileno())

def consume_claim_and_load_split(capability: ClaimCapability, split_path: Path, *, access_log: Path | None = None, purpose: str = "final_eval") -> dict[str, Any]:
    """The sole production split consumer; arbitrary mappings are not authority."""
    if not isinstance(capability, ClaimCapability) or capability._token is not _CAPABILITY_TOKEN: raise SprintClaimError("split consumption requires an acquired claim capability")
    claim_path = capability._path
    try: value = _validate_claim(json.loads(claim_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc: raise SprintClaimError("claim is unreadable") from exc
    resolved, canonical = Path(split_path).resolve(strict=True), CANONICAL_SPLIT_PATH.resolve(strict=True)
    if resolved != canonical or "m4" in resolved.name.lower(): raise SprintClaimError("only the canonical non-M4 sprint split is permitted")
    with resolved.open("rb") as handle: raw = handle.read()
    measured = hashlib.sha256(raw).hexdigest()
    if measured != CANONICAL_SPLIT_SHA256: raise SprintClaimError("canonical split bytes do not match trusted digest")
    record_access(access_log or REPO_ROOT / "outputs/metrics/t2_sprint_heldout_access.log", purpose, run_tag=value["run_tag"], arm=value["arm"], split_sha256=measured, claim_sha256=sha256_file(claim_path))
    try: payload = json.loads(raw)
    except json.JSONDecodeError as exc: raise SprintClaimError("canonical split is invalid JSON") from exc
    if payload.get("n_goals") != 100 or len(payload.get("specs", [])) != 100: raise SprintClaimError("sprint split must contain exactly 100 goals")
    return payload

def parse_disposition_receipt(path: Path, *, legacy_claim_sha256: str, run_tag: str) -> tuple[dict[str, Any], str]:
    try: receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise SprintClaimError("disposition receipt is invalid") from exc
    required = {"schema_version", "legacy_claim_sha256", "run_tag", "decision", "decided_by", "decided_at"}
    if not isinstance(receipt, dict) or set(receipt) != required or receipt["schema_version"] != 1 or receipt["legacy_claim_sha256"] != legacy_claim_sha256 or receipt["run_tag"] != run_tag or receipt["decision"] not in {"allow_reevaluation", "keep_legacy"} or not isinstance(receipt["decided_by"], str) or not receipt["decided_by"] or not isinstance(receipt["decided_at"], str) or not receipt["decided_at"]: raise SprintClaimError("disposition receipt schema or identity is invalid")
    return receipt, sha256_file(Path(path))

def atomic_publish(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle: json.dump(payload, handle, indent=1, sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp_name, path); _fsync_dir(path.parent)
    finally:
        if os.path.exists(tmp_name): os.unlink(tmp_name)

def probe_manifest_register(manifest_path: Path, file_path: Path, meta: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path, canonical = Path(manifest_path), Path(file_path).resolve(strict=True)
    if not canonical.is_file(): raise FileNotFoundError(canonical)
    entry = {"path": str(canonical), "sha256": sha256_file(canonical), "size": canonical.stat().st_size, **dict(meta)}
    if not entry.get("production_goal"): raise ValueError("probe metadata requires production_goal")
    lock_path = manifest_path.with_name(manifest_path.name + ".lock"); lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"schema_version": PRIMITIVE_SCHEMA_VERSION, "files": {}}
            if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "files"} or not isinstance(manifest["files"], dict): raise SprintClaimError("probe manifest schema is invalid")
            for old_key, old in manifest["files"].items():
                if old.get("path") == str(canonical) and old_key != entry["sha256"]: raise SprintClaimError(f"probe manifest entry is immutable: {canonical}")
                if old.get("path") == str(canonical) and old_key == entry["sha256"] and old != entry: raise SprintClaimError(f"probe manifest entry is immutable: {canonical}")
                if old_key == entry["sha256"] and old.get("path") != str(canonical): raise SprintClaimError("probe aliases are forbidden")
            if entry["sha256"] not in manifest["files"]: manifest["files"][entry["sha256"]] = entry; atomic_publish(manifest_path, manifest)
            return manifest
        finally: fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

def require_metric_lock(lock_path: Path | None, arm: str) -> None:
    if _arm(arm) == "bb": return
    if lock_path is None: raise SprintClaimError(f"arm {arm!r} requires --lock before split load")
    try: value = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise SprintClaimError(f"metric lock is invalid: {lock_path}") from exc
    required = {"schema_version", "endpoint", "aggregate", "created_at", "bb_claim_sha256", "primitive_version"}
    if not isinstance(value, dict) or set(value) != required or value["schema_version"] != 1 or value["endpoint"] not in {"success_rate", "return"} or not isinstance(value["aggregate"], (int, float)) or isinstance(value["aggregate"], bool) or not value["created_at"] or not value["primitive_version"]: raise SprintClaimError("metric lock schema is invalid")
    hashes = value["bb_claim_sha256"]
    if not isinstance(hashes, list) or len(hashes) != 8 or len(set(hashes)) != 8 or any(not isinstance(x, str) or not _HEX64.fullmatch(x) for x in hashes): raise SprintClaimError("metric lock schema is invalid")

def audit_claims(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for claim in Path(directory).glob("*claim*.json"):
        try: data = _validate_claim(json.loads(claim.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, SprintClaimError) as exc:
            rows.append({"schema_version": 1, "status": "malformed_claim", "claim": str(claim), "error": str(exc)}); continue
        tag, arm = data["run_tag"], data["arm"]
        names = [f"p1_{arm}_sprint_heldout_{tag}.json", f"p1_t2_sprint_heldout_{tag}.json"]
        if not any((Path(directory) / name).exists() for name in names): rows.append({"schema_version": 1, "status": "needs_human_disposition", "claim": str(claim), "claim_sha256": sha256_file(claim), "run_tag": tag, "arm": arm, "re_evaluation_permitted": False})
    return rows
