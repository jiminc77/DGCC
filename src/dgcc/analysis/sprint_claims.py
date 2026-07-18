"""Durable one-shot authorization primitives for the sprint held-out split."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PRIMITIVE_SCHEMA_VERSION = 1
# SHA-256 of the committed canonical t2_sprint_heldout_v1.json bytes.  This is
# deliberately a source constant rather than a digest learned from the file at
# evaluation time: the claim must exist before the split is opened.
CANONICAL_SPLIT_SHA256 = "76335ae50efd8164df1f8e241ae69aa30685f201aa6f0554d4a5b077cc1e2754"
CANONICAL_SPLIT_PATH = Path(__file__).resolve().parents[1] / "tasks/splits/t2_sprint_heldout_v1.json"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class SprintClaimError(RuntimeError):
    """A held-out evaluation is not authorized."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def acquire_claim(claim_path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Create the exclusive authorization before any split open occurs."""
    claim_path = Path(claim_path)
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body.update(schema_version=PRIMITIVE_SCHEMA_VERSION, claim_before_load=True)
    body.setdefault("timestamp", utc_now())
    body.setdefault("pid", os.getpid())
    required = ("run_tag", "arm", "ckpt_sha256", "split_sha256")
    missing = [key for key in required if not body.get(key)]
    if missing: raise SprintClaimError(f"claim payload missing required fields: {', '.join(missing)}")
    if body["split_sha256"] != CANONICAL_SPLIT_SHA256:
        raise SprintClaimError("claim split digest is not the trusted canonical digest")
    try: fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc: raise SprintClaimError(f"held-out claim already exists: {claim_path}; do NOT delete it to re-run") from exc
    try:
        os.write(fd, (json.dumps(body, indent=1, sort_keys=True) + "\n").encode())
        os.fsync(fd)
    finally: os.close(fd)
    _fsync_dir(claim_path.parent)
    return body


def record_access(log_path: Path, purpose: str, **metadata: Any) -> None:
    if not purpose: raise ValueError("access purpose is required")
    if not _HEX64.fullmatch(str(metadata.get("claim_sha256", ""))):
        raise SprintClaimError("access record requires claim_sha256")
    log_path = Path(log_path); log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": utc_now(), "pid": os.getpid(), "purpose": purpose, **metadata}
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n"); handle.flush(); os.fsync(handle.fileno())


def consume_claim_and_load_split(claim: Path | Mapping[str, Any], split_path: Path, *, access_log: Path | None = None, purpose: str = "final_eval") -> dict[str, Any]:
    """The sole production split consumer: open, hash, audit, then parse."""
    claim_path = Path(claim) if not isinstance(claim, Mapping) else None
    value = dict(claim) if isinstance(claim, Mapping) else json.loads(claim_path.read_text(encoding="utf-8"))
    if value.get("split_sha256") != CANONICAL_SPLIT_SHA256:
        raise SprintClaimError("claim does not bind the trusted canonical split digest")
    resolved = Path(split_path).resolve(strict=True)
    canonical = CANONICAL_SPLIT_PATH.resolve(strict=True)
    if resolved != canonical or "m4" in resolved.name.lower():
        raise SprintClaimError("only the canonical non-M4 sprint split is permitted")
    # This is intentionally the first split open.  Do not move it above claim checks.
    with resolved.open("rb") as handle: raw = handle.read()
    measured = hashlib.sha256(raw).hexdigest()
    if measured != CANONICAL_SPLIT_SHA256: raise SprintClaimError("canonical split bytes do not match trusted digest")
    claim_sha = sha256_file(claim_path) if claim_path else hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    record_access(access_log or Path("outputs/metrics/t2_sprint_heldout_access.log"), purpose,
                  run_tag=value.get("run_tag"), arm=value.get("arm"), split_sha256=measured, claim_sha256=claim_sha)
    try: payload = json.loads(raw)
    except json.JSONDecodeError as exc: raise SprintClaimError("canonical split is invalid JSON") from exc
    if payload.get("n_goals") != 100 or len(payload.get("specs", [])) != 100:
        raise SprintClaimError("sprint split must contain exactly 100 goals")
    return payload


def atomic_publish(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp_name, path); _fsync_dir(path.parent)
    finally:
        if os.path.exists(tmp_name): os.unlink(tmp_name)


def probe_manifest_register(manifest_path: Path, file_path: Path, meta: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically content-register a canonical probe path; aliases are refused."""
    manifest_path, file_path = Path(manifest_path), Path(file_path)
    canonical = file_path.resolve(strict=True)
    if not canonical.is_file(): raise FileNotFoundError(canonical)
    entry = {"path": str(canonical), "sha256": sha256_file(canonical), "size": canonical.stat().st_size, **dict(meta)}
    if not entry.get("production_goal"): raise ValueError("probe metadata requires production_goal")
    key = entry["sha256"]
    lock_path = manifest_path.with_name(manifest_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            manifest = {"schema_version": PRIMITIVE_SCHEMA_VERSION, "files": {}}
            if manifest_path.exists(): manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.setdefault("schema_version", PRIMITIVE_SCHEMA_VERSION); manifest.setdefault("files", {})
            for old_key, old in manifest["files"].items():
                if old.get("path") == str(canonical) and old_key != key: raise SprintClaimError(f"probe manifest entry is immutable: {canonical}")
            prior = manifest["files"].get(key)
            if prior is not None and prior != entry: raise SprintClaimError(f"probe manifest entry is immutable: {canonical}")
            if prior is None: manifest["files"][key] = entry; atomic_publish(manifest_path, manifest)
            return manifest
        finally: fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def require_metric_lock(lock_path: Path | None, arm: str) -> None:
    if arm.upper() == "BB": return
    if lock_path is None: raise SprintClaimError(f"arm {arm!r} requires --lock before split load")
    try: value = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise SprintClaimError(f"metric lock is invalid: {lock_path}") from exc
    required = {"schema_version", "endpoint", "aggregate", "created_at", "bb_claim_sha256", "primitive_version"}
    if not isinstance(value, dict) or set(value) != required: raise SprintClaimError("metric lock schema is invalid")
    if value["schema_version"] != 1 or value["endpoint"] not in {"success_rate", "return"} or not isinstance(value["aggregate"], (int, float)) or isinstance(value["aggregate"], bool) or not value["created_at"] or not value["primitive_version"]: raise SprintClaimError("metric lock schema is invalid")
    hashes = value["bb_claim_sha256"]
    if not isinstance(hashes, list) or len(hashes) != 8 or len(set(hashes)) != 8 or any(not isinstance(x, str) or not _HEX64.fullmatch(x) for x in hashes): raise SprintClaimError("metric lock schema is invalid")


def audit_claims(directory: Path) -> list[dict[str, Any]]:
    """Classify durable claims lacking a result; this never re-evaluates or mutates claims."""
    directory = Path(directory); rows = []
    for claim in directory.glob("*claim*.json"):
        try: data = json.loads(claim.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): continue
        tag = data.get("run_tag") or data.get("m4_tag")
        result = directory / f"p1_t2_sprint_heldout_{tag}.json"
        if tag and not result.exists(): rows.append({"schema_version": 1, "status": "needs_human_disposition", "claim": str(claim), "claim_sha256": sha256_file(claim), "re_evaluation_permitted": False})
    return rows
