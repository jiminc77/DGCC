"""Durable one-shot evaluation authorization and artifact publication primitives."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PRIMITIVE_SCHEMA_VERSION = 1


class SprintClaimError(RuntimeError):
    """A held-out evaluation is not authorized."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def acquire_claim(claim_path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Durably create an exclusive pre-load claim or refuse evaluation.

    A claim is intentionally never removed or retried: an interrupted attempt is
    still a held-out access attempt and must be investigated rather than rerun.
    """
    claim_path = Path(claim_path)
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body["schema_version"] = PRIMITIVE_SCHEMA_VERSION
    body["claim_before_load"] = True
    body.setdefault("timestamp", utc_now())
    body.setdefault("pid", os.getpid())
    required = ("run_tag", "arm", "ckpt_sha256", "split_sha256")
    missing = [key for key in required if not body.get(key)]
    if missing:
        raise SprintClaimError(f"claim payload missing required fields: {', '.join(missing)}")
    try:
        fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise SprintClaimError(
            f"held-out claim already exists: {claim_path}; do NOT delete it to re-run"
        ) from exc
    try:
        encoded = (json.dumps(body, indent=1, sort_keys=True) + "\n").encode("utf-8")
        os.write(fd, encoded)
        os.fsync(fd)
    except OSError as exc:
        raise SprintClaimError(f"claim durability failed; evaluation is not authorized: {exc}") from exc
    finally:
        os.close(fd)
    try:
        _fsync_dir(claim_path.parent)
    except OSError as exc:
        raise SprintClaimError(f"claim directory durability failed; evaluation is not authorized: {exc}") from exc
    return body


def record_access(log_path: Path, purpose: str, **metadata: Any) -> None:
    """Append one durable, purpose-coded JSON access record."""
    if not purpose:
        raise ValueError("access purpose is required")
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": utc_now(), "pid": os.getpid(), "purpose": purpose, **metadata}
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_publish(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish JSON and make both file and rename durable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        _fsync_dir(path.parent)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def probe_manifest_register(manifest_path: Path, file_path: Path, meta: Mapping[str, Any]) -> dict[str, Any]:
    """Register a probe by stable path; an existing path may only retain its hash."""
    manifest_path, file_path = Path(manifest_path), Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    key = str(file_path)
    entry = {"sha256": sha256_file(file_path), "size": file_path.stat().st_size, **dict(meta)}
    if not entry.get("production_goal"):
        raise ValueError("probe metadata requires production_goal")
    manifest: dict[str, Any] = {"schema_version": PRIMITIVE_SCHEMA_VERSION, "files": {}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("schema_version", PRIMITIVE_SCHEMA_VERSION)
        manifest.setdefault("files", {})
    prior = manifest["files"].get(key)
    if prior is not None and prior.get("sha256") != entry["sha256"]:
        raise SprintClaimError(f"probe manifest entry is immutable: {key}")
    if prior is None:
        manifest["files"][key] = entry
        atomic_publish(manifest_path, manifest)
    return manifest


def require_metric_lock(lock_path: Path | None, arm: str) -> None:
    """Require a parseable metric lock before any non-BB held-out access."""
    if arm.upper() == "BB":
        return
    if lock_path is None:
        raise SprintClaimError(f"arm {arm!r} requires --lock before split load")
    path = Path(lock_path)
    if not path.is_file():
        raise SprintClaimError(f"metric lock missing for arm {arm!r}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SprintClaimError(f"metric lock is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise SprintClaimError(f"metric lock is invalid: {path}")
