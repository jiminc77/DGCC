"""Crash-safe, append-only registry for training attempts."""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import stat
import threading
import warnings
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TERMINAL = {"SUCCEEDED", "TECHNICAL_FAILURE", "PERFORMANCE_FAILURE", "ALGO_ABORT", "ABORTED", "ORPHANED"}
_SHA256 = set("0123456789abcdef")
_INTERNAL_ARTIFACTS = {"records.jsonl", "owner.lock"}
_GOVERNED_LAUNCH_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "content_address",
        "arm",
        "seed",
        "schedule_arm",
        "mode",
        "policy_delay",
        "config_sha256",
        "code_manifest_sha256",
        "governance_sha256",
        "admitted_manifest_sha256",
        "code_closure_sha256",
        "code_closure_count",
        "neff_guard_sha256",
        "q1_pooled_median",
        "qmin_pooled_median",
        "guard_passed",
        "neff_guard_passed",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256


def _validated_governed_launch_receipt(receipt: Any) -> bytes:
    """Return the canonical, strictly validated governed launch receipt bytes."""
    if not isinstance(receipt, dict) or set(receipt) != _GOVERNED_LAUNCH_RECEIPT_FIELDS:
        raise ValueError("governed_launch_receipt has an invalid schema")
    if (
        receipt["schema_version"] != 2
        or not isinstance(receipt["arm"], str)
        or not receipt["arm"]
        or not isinstance(receipt["seed"], int)
        or isinstance(receipt["seed"], bool)
        or not isinstance(receipt["schedule_arm"], str)
        or not receipt["schedule_arm"]
        or not isinstance(receipt["mode"], str)
        or not receipt["mode"]
        or not isinstance(receipt["policy_delay"], int)
        or isinstance(receipt["policy_delay"], bool)
        or not isinstance(receipt["code_closure_count"], int)
        or isinstance(receipt["code_closure_count"], bool)
        or receipt["code_closure_count"] < 1
        or (
            receipt["admitted_manifest_sha256"] is not None
            and not _valid_sha256(receipt["admitted_manifest_sha256"])
        )
        or not all(
            _valid_sha256(receipt[field])
            for field in (
                "content_address",
                "config_sha256",
                "code_manifest_sha256",
                "governance_sha256",
                "code_closure_sha256",
                "neff_guard_sha256",
            )
        )
        or any(
            type(receipt[field]) not in (int, float)
            or not math.isfinite(receipt[field])
            or not 8.0 <= receipt[field] <= 20.0
            for field in ("q1_pooled_median", "qmin_pooled_median")
        )
        or receipt["guard_passed"] is not True
        or receipt["neff_guard_passed"] is not True
    ):
        raise ValueError("governed_launch_receipt has invalid binding values")
    return json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_durable_file(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        _write_all(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    AttemptRegistry._fsync_dir(path.parent)

def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short write")
        view = view[written:]


def sha256_file(path: Path) -> str:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"not a regular file: {path}")
        digest = hashlib.sha256()
        while block := os.read(fd, 1024 * 1024):
            digest.update(block)
        return digest.hexdigest()
    finally:
        os.close(fd)


class RegistryCorruptionError(RuntimeError):
    """Published evidence is incomplete or fails its append-only integrity checks."""


class AttemptRegistry:
    """Own one attempt lifetime, publishing it only after durable PREPARING evidence."""

    def __init__(
        self,
        root: Path | str,
        *,
        run_tag: str,
        config: dict[str, Any],
        code_sha256: str,
        seed: int,
        retry_parent: str | None = None,
        terminal_anchor_directory: Path | str | None = None,
        governed_launch_receipt: dict[str, Any] | None = None,
    ) -> None:
        if not _valid_sha256(code_sha256):
            raise ValueError("code_sha256 must be lowercase SHA-256 hex")
        receipt_bytes = (
            None
            if governed_launch_receipt is None
            else _validated_governed_launch_receipt(governed_launch_receipt)
        )
        if governed_launch_receipt is not None and governed_launch_receipt["seed"] != seed:
            raise ValueError("governed_launch_receipt seed must match attempt seed")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._fsync_dir(self.root)
        self.terminal_anchor_directory = self._prepare_anchor_directory(
            self.root, terminal_anchor_directory
        )
        self.run_tag, self.attempt_id, self._closed = run_tag, str(uuid.uuid4()), False
        self._previous_hash, self._lock_handle, self._finalize_lock = "0" * 64, None, threading.RLock()
        staging_root = self.root / ".staging"
        staging_root.mkdir(exist_ok=True)
        self._fsync_dir(self.root)
        self.staging_path, self.attempt_path = staging_root / self.attempt_id, self.root / self.attempt_id
        self.staging_path.mkdir(exist_ok=False)
        self._fsync_dir(staging_root)
        self._lock_handle = (self.staging_path / "owner.lock").open("a+b")
        fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX)
        receipt_sha256 = None
        if receipt_bytes is not None:
            reports_path = self.staging_path / "reports"
            reports_path.mkdir(mode=0o700)
            self._fsync_dir(self.staging_path)
            _write_durable_file(reports_path / "governed_launch_receipt.json", receipt_bytes)
            receipt_sha256 = _sha256_bytes(receipt_bytes)
        self._append({"phase": "PREPARING", "attempt_id": self.attempt_id, "run_tag": run_tag,
                      "config_sha256": _sha256_bytes(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()),
                      "code_sha256": code_sha256, "seed": seed, "retry_parent": retry_parent,
                      "governed_launch_receipt_sha256": receipt_sha256, "started_at": _utc_now()})
        self._fsync_dir(self.staging_path)
        os.rename(self.staging_path, self.attempt_path)
        self._fsync_dir(self.root)

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try: os.fsync(fd)
        finally: os.close(fd)

    @classmethod
    def _prepare_anchor_directory(
        cls, root: Path, directory: Path | str | None
    ) -> Path:
        root_resolved = root.resolve(strict=True)
        anchor = (
            root_resolved.parent / "terminal-anchors"
            if directory is None
            else Path(directory).resolve(strict=False)
        )
        try:
            anchor.relative_to(root_resolved)
        except ValueError:
            pass
        else:
            raise ValueError(
                "terminal anchor directory must be outside the registry root"
            )
        anchor.mkdir(parents=True, exist_ok=True)
        cls._fsync_dir(anchor)
        return anchor

    @property
    def record_path(self) -> Path:
        return self.attempt_path / "records.jsonl"

    @staticmethod
    def allowed_artifact_paths(attempt_path: Path) -> list[Path]:
        """Discover only regular, non-internal artifacts for terminal hashing."""
        root = attempt_path.resolve(strict=True)
        try:
            internal_identities = {
                (info.st_dev, info.st_ino)
                for name in _INTERNAL_ARTIFACTS
                for info in [os.stat(attempt_path / name, follow_symlinks=False)]
            }
        except OSError as error:
            raise RegistryCorruptionError(
                f"attempt internal artifact is inaccessible: {attempt_path}"
            ) from error

        allowed: list[Path] = []
        for path in sorted(root.rglob("*")):
            try:
                info = os.lstat(path)
                relative = path.relative_to(root)
            except (OSError, ValueError) as error:
                raise ValueError(f"attempt artifact is inaccessible: {path}") from error
            if (
                not stat.S_ISLNK(info.st_mode)
                and stat.S_ISREG(info.st_mode)
                and relative.name not in _INTERNAL_ARTIFACTS
                and not any(part.startswith(".") for part in relative.parts)
                and (info.st_dev, info.st_ino) not in internal_identities
            ):
                allowed.append(path)
        return allowed
    @staticmethod
    def _artifact_hashes(attempt_path: Path, artifact_paths: list[Path] | None) -> dict[str, str]:
        root = attempt_path.resolve(strict=True)
        try:
            internal_identities = {
                (info.st_dev, info.st_ino)
                for name in _INTERNAL_ARTIFACTS
                for info in [os.stat(attempt_path / name, follow_symlinks=False)]
            }
        except OSError as error:
            raise RegistryCorruptionError(f"attempt internal artifact is inaccessible: {attempt_path}") from error
        hashes: dict[str, str] = {}
        for requested in artifact_paths or []:
            path = Path(requested)
            try:
                before = os.lstat(path)
                resolved = path.resolve(strict=True)
                relative = resolved.relative_to(root)
            except (OSError, ValueError) as error:
                raise ValueError(f"requested artifact is invalid or outside the attempt tree: {path}") from error
            if (stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode)
                    or relative.name in _INTERNAL_ARTIFACTS
                    or any(part.startswith(".") for part in relative.parts)
                    or (before.st_dev, before.st_ino) in internal_identities):
                raise ValueError(f"requested artifact is not an allowed regular artifact: {path}")
            key = str(relative)
            if key in hashes:
                raise ValueError(f"requested artifact is duplicated: {path}")
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                opened = os.fstat(fd)
                if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                    raise ValueError(f"requested artifact identity changed: {path}")
                digest = hashlib.sha256()
                while block := os.read(fd, 1024 * 1024): digest.update(block)
                after = os.stat(path, follow_symlinks=False)
                if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns):
                    raise ValueError(f"requested artifact changed while hashing: {path}")
                hashes[key] = digest.hexdigest()
            finally:
                os.close(fd)
        return hashes

    def _make_record(
        self, payload: dict[str, Any], *, recorded_at: str | None = None
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("attempt is already finalized")
        body = {
            **payload,
            "recorded_at": _utc_now() if recorded_at is None else recorded_at,
            "previous_sha256": self._previous_hash,
        }
        return {
            **body,
            "sha256": _sha256_bytes(
                json.dumps(
                    body, sort_keys=True, separators=(",", ":")
                ).encode()
            ),
        }

    def _append_record(self, record: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("attempt is already finalized")
        payload = {key: value for key, value in record.items() if key != "sha256"}
        if (
            record.get("previous_sha256") != self._previous_hash
            or not _valid_sha256(record.get("sha256"))
            or _sha256_bytes(
                json.dumps(
                    payload, sort_keys=True, separators=(",", ":")
                ).encode()
            )
            != record["sha256"]
        ):
            raise RegistryCorruptionError(
                "prepared record does not continue the attempt chain"
            )
        frame = (
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
            + b"\n"
        )
        path = (
            self.staging_path / "records.jsonl"
            if not self.attempt_path.exists()
            else self.record_path
        )
        fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW,
            0o600,
        )
        try:
            _write_all(fd, frame)
            os.fsync(fd)
        finally:
            os.close(fd)
        self._previous_hash = record["sha256"]
        return record

    def _append(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._append_record(self._make_record(payload))

    def initialized(self, initial_weights_sha256: str) -> None:
        if not _valid_sha256(initial_weights_sha256): raise ValueError("initial_weights_sha256 must be lowercase SHA-256 hex")
        with self._finalize_lock:
            if [record["phase"] for record in self.read_records(self.attempt_path)] != ["PREPARING"]:
                raise RegistryCorruptionError("INITIALIZED is only valid immediately after PREPARING")
            self._append({"phase": "INITIALIZED", "initial_weights_sha256": initial_weights_sha256})

    def record_artifacts(self, artifact_paths: list[Path]) -> dict[str, Any]:
        with self._finalize_lock:
            if self.read_records(self.attempt_path)[-1]["phase"] not in {"PREPARING", "INITIALIZED"}:
                raise RegistryCorruptionError("ARTIFACTS may be recorded only once before terminal")
            return self._append({"phase": "ARTIFACTS", "artifact_sha256": self._artifact_hashes(self.attempt_path, artifact_paths)})

    def _release_owner_lock(self) -> None:
        if self._lock_handle is not None:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN); self._lock_handle.close(); self._lock_handle = None

    def finalize_once(
        self,
        disposition: str,
        *,
        exit_code: int | None = None,
        artifact_paths: list[Path] | None = None,
        detail: str | None = None,
    ) -> bool:
        if disposition not in _TERMINAL:
            raise ValueError(f"invalid terminal disposition: {disposition}")
        with self._finalize_lock:
            if self._closed or self._lock_handle is None:
                return False
            records = self.read_records(self.attempt_path)
            if records[-1]["phase"] == "TERMINAL":
                self._verify_anchor(
                    self.attempt_path, self.terminal_anchor_directory, records
                )
                self._closed = True
                self._previous_hash = records[-1]["sha256"]
                self._release_owner_lock()
                return False
            self._previous_hash = records[-1]["sha256"]
            candidates = (
                self.allowed_artifact_paths(self.attempt_path)
                if artifact_paths is None
                else artifact_paths
            )
            ended_at = _utc_now()
            terminal = self._make_record(
                {
                    "phase": "TERMINAL",
                    "disposition": disposition,
                    "exit_code": exit_code,
                    "ended_at": ended_at,
                    "detail": detail,
                    "artifact_sha256": self._artifact_hashes(
                        self.attempt_path, candidates
                    ),
                },
                recorded_at=ended_at,
            )
            self._publish_terminal_anchor(records[0], terminal)
            self._append_record(terminal)
            self._verify_anchor(
                self.attempt_path,
                self.terminal_anchor_directory,
                [*records, terminal],
            )
            self._closed = True
            self._release_owner_lock()
            if disposition == "SUCCEEDED":
                try:
                    self._reconcile_latest_success(
                        self.root, self.terminal_anchor_directory
                    )
                except Exception as error:
                    warnings.warn(
                        "durable SUCCEEDED terminal committed but "
                        f"latest-success reconciliation failed: {error}",
                        RuntimeWarning,
                    )
            return True

    @classmethod
    def _validate_records(cls, records: list[dict[str, Any]], attempt_path: Path) -> None:
        if not records: raise RegistryCorruptionError(f"empty attempt record: {attempt_path}")
        phases = [r.get("phase") for r in records]
        allowed = ["PREPARING"] + (["INITIALIZED"] if len(phases) > 1 and phases[1] == "INITIALIZED" else [])
        offset = len(allowed)
        if len(phases) > offset and phases[offset] == "ARTIFACTS": allowed.append("ARTIFACTS")
        if len(phases) > len(allowed) and phases[len(allowed)] == "TERMINAL": allowed.append("TERMINAL")
        if phases != allowed: raise RegistryCorruptionError(f"invalid phase sequence: {attempt_path}")
        first = records[0]
        if first.get("attempt_id") != attempt_path.name or not isinstance(first.get("run_tag"), str) or not _valid_sha256(first.get("config_sha256")) or not _valid_sha256(first.get("code_sha256")):
            raise RegistryCorruptionError(f"invalid PREPARING identity: {attempt_path}")
        expected_fields = {
            "PREPARING": {"phase", "attempt_id", "run_tag", "config_sha256", "code_sha256", "seed", "retry_parent", "governed_launch_receipt_sha256", "started_at", "recorded_at", "previous_sha256", "sha256"},
            "INITIALIZED": {"phase", "initial_weights_sha256", "recorded_at", "previous_sha256", "sha256"},
            "ARTIFACTS": {"phase", "artifact_sha256", "recorded_at", "previous_sha256", "sha256"},
            "TERMINAL": {"phase", "disposition", "exit_code", "ended_at", "detail", "artifact_sha256", "recorded_at", "previous_sha256", "sha256"},
        }
        for record in records:
            if set(record) != expected_fields[record["phase"]]:
                raise RegistryCorruptionError(f"invalid record schema: {attempt_path}")
            if record["phase"] == "INITIALIZED" and not _valid_sha256(record.get("initial_weights_sha256")): raise RegistryCorruptionError(f"invalid initialized digest: {attempt_path}")
            if record["phase"] in {"ARTIFACTS", "TERMINAL"} and (not isinstance(record.get("artifact_sha256"), dict) or not all(isinstance(k, str) and _valid_sha256(v) for k, v in record["artifact_sha256"].items())): raise RegistryCorruptionError(f"invalid artifact hashes: {attempt_path}")
            if record["phase"] == "TERMINAL" and record.get("disposition") not in _TERMINAL: raise RegistryCorruptionError(f"invalid terminal disposition: {attempt_path}")
            if record["phase"] == "PREPARING" and record.get("governed_launch_receipt_sha256") is not None:
                receipt_path = attempt_path / "reports" / "governed_launch_receipt.json"
                try:
                    if sha256_file(receipt_path) != record["governed_launch_receipt_sha256"]:
                        raise ValueError("receipt digest mismatch")
                    _validated_governed_launch_receipt(json.loads(receipt_path.read_bytes()))
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                    raise RegistryCorruptionError(
                        f"invalid governed launch receipt: {attempt_path}"
                    ) from error
            if record["phase"] == "PREPARING" and record.get("governed_launch_receipt_sha256") is not None and not _valid_sha256(record["governed_launch_receipt_sha256"]):
                raise RegistryCorruptionError(f"invalid governed launch receipt digest: {attempt_path}")

    @classmethod
    def _read_records_fd(cls, attempt_fd: int, attempt_path: Path) -> list[dict[str, Any]]:
        try:
            fd = os.open("records.jsonl", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=attempt_fd)
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise OSError("records is not regular")
                raw = b"".join(iter(lambda: os.read(fd, 1024 * 1024), b""))
            finally: os.close(fd)
        except OSError as error:
            raise RegistryCorruptionError(f"missing attempt record: {attempt_path}") from error
        if not raw.endswith(b"\n"): raise RegistryCorruptionError(f"torn record tail: {attempt_path}")
        previous, records = "0" * 64, []
        for line in raw.splitlines():
            try:
                record = json.loads(line); supplied = record.pop("sha256")
                if not _valid_sha256(supplied) or record.get("previous_sha256") != previous: raise ValueError("hash chain discontinuity")
                if _sha256_bytes(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()) != supplied: raise ValueError("record digest mismatch")
            except (AttributeError, KeyError, ValueError, TypeError, json.JSONDecodeError) as error: raise RegistryCorruptionError(f"invalid record frame: {attempt_path}") from error
            record["sha256"] = supplied; records.append(record); previous = supplied
        cls._validate_records(records, attempt_path)
        return records

    @classmethod
    def read_records(cls, attempt_path: Path) -> list[dict[str, Any]]:
        attempt_path = Path(attempt_path)
        try:
            if stat.S_ISLNK(os.lstat(attempt_path).st_mode): raise OSError("symlink attempt")
            attempt_fd = os.open(attempt_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                return cls._read_records_fd(attempt_fd, attempt_path)
            finally:
                os.close(attempt_fd)
        except OSError as error:
            raise RegistryCorruptionError(f"missing attempt record: {attempt_path}") from error

    @classmethod
    def _reconcile_latest_success(
        cls, root: Path, anchor_directory: Path
    ) -> None:
        lock_path = root / ".latest-success.lock"
        handle = lock_path.open("a+b")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            cls._reconcile_latest_success_unlocked(
                root, anchor_directory
            )
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    @classmethod
    def _reconcile_latest_success_unlocked(
        cls, root: Path, anchor_directory: Path
    ) -> None:
        candidates: list[tuple[str, str, dict[str, Any]]] = []
        for path in root.iterdir():
            if path.name.startswith(".") or not path.is_dir() or path.is_symlink(): continue
            records = cls.read_records(path)
            if records[-1]["phase"] == "TERMINAL":
                cls._verify_anchor(path, anchor_directory, records)
            if records[-1]["phase"] == "TERMINAL" and records[-1]["disposition"] == "SUCCEEDED":
                candidates.append((str(records[-1]["ended_at"]), str(records[0]["attempt_id"]), records[0]))
        latest = root / "latest-success.json"
        if not candidates:
            if latest.exists(): latest.unlink(); cls._fsync_dir(root)
            return
        _, _, first = max(candidates)
        data = json.dumps({"attempt_id": first["attempt_id"], "run_tag": first["run_tag"]}, sort_keys=True).encode() + b"\n"
        temporary = root / f".latest-{uuid.uuid4().hex}.tmp"
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try: _write_all(fd, data); os.fsync(fd)
        finally: os.close(fd)
        os.replace(temporary, latest); cls._fsync_dir(root)
    @classmethod
    def _quarantine_torn_attempt(cls, root: Path, attempt_path: Path) -> None:
        quarantine_root = root / ".quarantine"
        attempt_quarantine = quarantine_root / attempt_path.name
        quarantine_root.mkdir(exist_ok=True)
        attempt_quarantine.mkdir(exist_ok=True)
        while True:
            destination = attempt_quarantine / uuid.uuid4().hex
            try:
                destination.mkdir()
                break
            except FileExistsError:
                continue
        try:
            os.rename(attempt_path, destination / "attempt")
        except OSError as error:
            raise RegistryCorruptionError(
                f"failed to quarantine torn attempt: {attempt_path}"
            ) from error
        cls._fsync_dir(destination)
        cls._fsync_dir(attempt_quarantine)
        cls._fsync_dir(quarantine_root)
        cls._fsync_dir(root)


    @classmethod
    def recover(
        cls,
        root: Path | str,
        *,
        terminal_anchor_directory: Path | str | None = None,
    ) -> list[str]:
        supplied_root = Path(root)
        try:
            supplied_info = os.lstat(supplied_root)
        except FileNotFoundError:
            return []
        except OSError as error:
            raise RegistryCorruptionError(f"registry root is inaccessible: {supplied_root}") from error
        if stat.S_ISLNK(supplied_info.st_mode) or not stat.S_ISDIR(supplied_info.st_mode):
            raise RegistryCorruptionError(f"registry root is symlink or not a directory: {supplied_root}")
        root = supplied_root.resolve(strict=True)
        anchor_directory = cls._prepare_anchor_directory(
            root, terminal_anchor_directory
        )
        try:
            resolved_info = os.lstat(root)
            if (resolved_info.st_dev, resolved_info.st_ino) != (supplied_info.st_dev, supplied_info.st_ino):
                raise OSError("registry root identity changed")
            root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except OSError as error:
            raise RegistryCorruptionError(f"registry root is inaccessible: {supplied_root}") from error
        recovered: list[str] = []
        try:
            for name in os.listdir(root_fd):
                if name.startswith("."):
                    continue
                attempt_path = root / name
                attempt_fd = -1
                try:
                    child_info = os.lstat(name, dir_fd=root_fd)
                    if stat.S_ISLNK(child_info.st_mode):
                        raise RegistryCorruptionError(f"symlink attempt: {attempt_path}")
                    if not stat.S_ISDIR(child_info.st_mode):
                        continue
                    attempt_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
                    opened_info = os.fstat(attempt_fd)
                    if (opened_info.st_dev, opened_info.st_ino) != (child_info.st_dev, child_info.st_ino):
                        raise OSError("attempt identity changed")
                    lock_info = os.stat("owner.lock", dir_fd=attempt_fd, follow_symlinks=False)
                    if stat.S_ISLNK(lock_info.st_mode) or not stat.S_ISREG(lock_info.st_mode):
                        raise OSError("invalid lock")
                    lock_fd = os.open("owner.lock", os.O_RDWR | os.O_NOFOLLOW, dir_fd=attempt_fd)
                    if (lambda opened: (opened.st_dev, opened.st_ino) == (lock_info.st_dev, lock_info.st_ino) and stat.S_ISREG(opened.st_mode))(os.fstat(lock_fd)) is False:
                        os.close(lock_fd)
                        raise OSError("lock identity changed")
                except RegistryCorruptionError:
                    if attempt_fd >= 0:
                        os.close(attempt_fd)
                    raise
                except OSError as error:
                    if attempt_fd >= 0:
                        os.close(attempt_fd)
                    raise RegistryCorruptionError(f"published attempt missing owner lock: {attempt_path}") from error
                handle = os.fdopen(lock_fd, "a+b", closefd=True)
                try:
                    try: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError: continue
                    try:
                        records = cls._read_records_fd(attempt_fd, attempt_path)
                    except RegistryCorruptionError as error:
                        if str(error).startswith("torn record tail:"):
                            cls._quarantine_torn_attempt(root, attempt_path)
                        raise
                    if records[-1]["phase"] == "TERMINAL":
                        cls._verify_anchor(
                            attempt_path, anchor_directory, records
                        )
                        continue
                    registry = cls.__new__(cls); registry.root, registry.attempt_path, registry.staging_path = root, attempt_path, attempt_path
                    registry.attempt_id, registry.run_tag, registry._closed, registry._previous_hash = name, records[0]["run_tag"], False, records[-1]["sha256"]
                    registry._lock_handle, registry._finalize_lock = handle, threading.RLock()
                    registry.terminal_anchor_directory = anchor_directory
                    if registry.finalize_once("ORPHANED", detail="recovered unlocked nonterminal attempt"): recovered.append(name)
                    handle = None
                finally:
                    if handle is not None: fcntl.flock(handle.fileno(), fcntl.LOCK_UN); handle.close()
                    os.close(attempt_fd)
        finally:
            os.close(root_fd)
        cls._reconcile_latest_success(root, anchor_directory)
        return recovered

    @staticmethod
    def _anchor_payload(
        first: dict[str, Any], terminal: dict[str, Any]
    ) -> dict[str, str]:
        if terminal.get("phase") != "TERMINAL":
            raise RegistryCorruptionError(
                "terminal anchor requires a terminal record"
            )
        return {
            "phase": "PREPARED",
            "attempt_id": str(first["attempt_id"]),
            "prepared_terminal_sha256": str(terminal["sha256"]),
        }

    @classmethod
    def _anchor_target(
        cls,
        directory: Path,
        first: dict[str, Any],
        terminal: dict[str, Any],
    ) -> Path:
        return directory / (
            f"{first['attempt_id']}-{terminal['sha256']}.anchor.json"
        )

    @classmethod
    def _read_anchor(cls, receipt_path: Path) -> dict[str, Any]:
        fd = os.open(receipt_path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("terminal anchor is not a regular file")
            raw = b"".join(iter(lambda: os.read(fd, 65536), b""))
        finally:
            os.close(fd)
        if not raw.endswith(b"\n"):
            raise ValueError("terminal anchor is truncated")
        receipt = json.loads(raw)
        if not isinstance(receipt, dict):
            raise ValueError("terminal anchor is malformed")
        supplied = receipt.pop("sha256")
        if (
            not _valid_sha256(supplied)
            or _sha256_bytes(
                json.dumps(
                    receipt, sort_keys=True, separators=(",", ":")
                ).encode()
            )
            != supplied
        ):
            raise ValueError("terminal anchor digest mismatch")
        return receipt

    def _publish_terminal_anchor(
        self,
        first: dict[str, Any],
        terminal: dict[str, Any],
        *,
        directory: Path | None = None,
    ) -> Path:
        anchor_directory = (
            self.terminal_anchor_directory if directory is None else directory
        )
        payload = self._anchor_payload(first, terminal)
        receipt = {
            **payload,
            "sha256": _sha256_bytes(
                json.dumps(
                    payload, sort_keys=True, separators=(",", ":")
                ).encode()
            ),
        }
        target = self._anchor_target(
            anchor_directory, first, terminal
        )
        encoded = (
            json.dumps(
                receipt, sort_keys=True, separators=(",", ":")
            ).encode()
            + b"\n"
        )
        try:
            fd = os.open(
                target,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW,
                0o600,
            )
        except FileExistsError:
            try:
                if self._read_anchor(target) != payload:
                    raise ValueError("terminal anchor payload mismatch")
            except (
                AttributeError,
                OSError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                raise RegistryCorruptionError(
                    "existing terminal anchor conflicts with prepared terminal"
                ) from error
            return target
        try:
            _write_all(fd, encoded)
            os.fsync(fd)
        finally:
            os.close(fd)
        self._fsync_dir(anchor_directory)
        return target

    @classmethod
    def _verify_anchor(
        cls,
        attempt_path: Path,
        directory: Path,
        records: list[dict[str, Any]] | None = None,
    ) -> Path:
        records = (
            cls.read_records(attempt_path) if records is None else records
        )
        if records[-1].get("phase") != "TERMINAL":
            raise RegistryCorruptionError(
                "terminal anchor verification requires a terminal tail"
            )
        target = cls._anchor_target(
            directory, records[0], records[-1]
        )
        try:
            receipt = cls._read_anchor(target)
            if receipt != cls._anchor_payload(records[0], records[-1]):
                raise ValueError("terminal anchor payload mismatch")
        except (
            AttributeError,
            OSError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise RegistryCorruptionError(
                "terminal anchor verification failed"
            ) from error
        return target


    @classmethod
    def verify_terminal_anchor(
        cls, attempt_path: Path | str, receipt_path: Path | str
    ) -> bool:
        attempt = Path(attempt_path)
        receipt = Path(receipt_path)
        records = cls.read_records(attempt)
        expected = cls._anchor_target(
            receipt.parent, records[0], records[-1]
        )
        if receipt != expected:
            raise RegistryCorruptionError(
                "terminal anchor verification failed"
            )
        cls._verify_anchor(attempt, receipt.parent, records)
        return True
