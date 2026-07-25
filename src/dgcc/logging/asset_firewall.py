"""Fail-closed, auditable access control for V2 launch assets.

This is best-effort application control and not OS isolation. Callers outside
this process can still access the filesystem.
"""
from __future__ import annotations

import errno
import ctypes
import fcntl
import io
import hashlib
import json
import os
import re
import stat
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

PROTECTED_TOKENS = ("heldout", "held-out", "probe")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class AssetAccessError(PermissionError):
    """Raised when an asset is absent from the launch-time allowlist."""


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        count = os.write(fd, view)
        if count <= 0:
            raise OSError("short write")
        view = view[count:]


def _open_descriptor_bound(path: Path) -> int:
    """Open an absolute canonical file through pinned directory descriptors."""
    path = Path(os.path.abspath(path))
    directory_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        parts = path.parts[1:]
        if not parts:
            raise ValueError("path must name a regular file")
        for part in parts[:-1]:
            next_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError(f"not a regular file: {path}")
            return fd
        except Exception:
            os.close(fd)
            raise
    finally:
        os.close(directory_fd)


def _read_descriptor_bound_bytes(path: Path) -> bytes:
    """Read a regular file through descriptor-bound, no-symlink traversal."""
    fd = _open_descriptor_bound(path)
    try:
        return b"".join(iter(lambda: os.read(fd, 1 << 20), b""))
    finally:
        os.close(fd)


def _sha256(path: Path) -> str:
    return hashlib.sha256(_read_descriptor_bound_bytes(path)).hexdigest()


def _rename_noreplace(source: Path, destination: Path) -> None:
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as error:
        raise OSError("atomic no-replace publication requires Linux renameat2") from error
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), destination)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        _write_all(fd, json.dumps(payload, sort_keys=True, indent=1).encode() + b"\n")
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(directory)
    finally: os.close(directory)
    return path


class AssetFirewall:
    """Authorize only explicitly pinned assets and hash-chain protected denials."""

    limitation = "Best-effort application control only; it is not OS-isolated."

    def __init__(self, allowlist: Mapping[str | Path, str], audit_path: str | Path, roles: Mapping[str | Path, str] | None = None):
        if not allowlist:
            raise ValueError("allowlist must contain at least one pinned asset")
        self._allowlist = {Path(path).expanduser().resolve(): digest for path, digest in allowlist.items()}
        if any(not isinstance(digest, str) or not _SHA256.fullmatch(digest) for digest in self._allowlist.values()):
            raise ValueError("allowlist digests must be lowercase SHA-256 hex strings")
        self._roles = {Path(path).expanduser().resolve(): role for path, role in (roles or {}).items()}
        if any(self._protected(path) or self._protected_role(self._roles.get(path, "")) for path in self._allowlist):
            raise ValueError("heldout/probe assets and roles must never appear in a training allowlist")
        self.audit_path = Path(audit_path).expanduser().resolve(strict=False)
        if any(self.audit_path == asset or self.audit_path.is_relative_to(asset) or asset.is_relative_to(self.audit_path) for asset in self._allowlist):
            raise ValueError("audit path must not overlap an allowlisted asset")
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = uuid.uuid4().hex
        self._audit_head = "0" * 64
        # A launch has an auditable genesis even when it records no denials.
        fd = os.open(self.audit_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            self._append_audit_fd(fd, {"type": "GENESIS", "session_id": self.session_id})
            os.fsync(fd)
        except Exception:
            os.close(fd)
            raise
        else:
            os.close(fd)
        self._fsync_audit_parent()

    @staticmethod
    def _protected(path: Path) -> bool:
        return any(token in str(path).lower() for token in PROTECTED_TOKENS)

    @staticmethod
    def _protected_role(role: str) -> bool:
        return any(token in role.lower() for token in PROTECTED_TOKENS)

    def _fsync_audit_parent(self) -> None:
        fd = os.open(self.audit_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)

    def _append_audit_fd(self, fd: int, entry: dict[str, Any]) -> None:
        body = {**entry, "timestamp_utc": datetime.now(UTC).isoformat(), "previous_sha256": self._audit_head}
        entry = {**body, "sha256": hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
        _write_all(fd, json.dumps(entry, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        self._audit_head = entry["sha256"]

    def _read_audit(self) -> list[dict[str, Any]]:
        try:
            fd = os.open(self.audit_path, os.O_RDONLY | os.O_NOFOLLOW)
            try: raw = b"".join(iter(lambda: os.read(fd, 65536), b""))
            finally: os.close(fd)
        except OSError as error:
            raise AssetAccessError("launch audit is missing or inaccessible") from error
        if not raw or not raw.endswith(b"\n"):
            raise AssetAccessError("launch audit is missing or truncated")
        previous, entries = "0" * 64, []
        for line in raw.splitlines():
            try:
                entry = json.loads(line); supplied = entry.pop("sha256")
                if not isinstance(supplied, str) or not _SHA256.fullmatch(supplied) or entry.get("previous_sha256") != previous:
                    raise ValueError("audit chain discontinuity")
                if hashlib.sha256(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != supplied:
                    raise ValueError("audit digest mismatch")
            except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise AssetAccessError("launch audit is tampered") from error
            entry["sha256"] = supplied; entries.append(entry); previous = supplied
        if entries[0].get("type") != "GENESIS" or entries[0].get("session_id") != self.session_id or previous != self._audit_head:
            raise AssetAccessError("launch audit does not belong to this session")
        return entries

    def _deny_protected(self, requested: Path, canonical: Path, operation: str) -> None:
        fd = os.open(self.audit_path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            self._read_audit()
            self._append_audit_fd(fd, {"type": "DENIED", "requested_path": str(requested), "canonical_path": str(canonical), "operation": operation, "allowed": False})
            os.fsync(fd)
        finally:
            os.close(fd)
        raise AssetAccessError(f"heldout/probe assets are forbidden on the training host: {canonical}")

    def _open_authorized(
        self, path: str | Path, operation: str, required_role: str | None = None
    ) -> tuple[int, Path]:
        requested = Path(path).expanduser()
        canonical = requested.resolve(strict=False)
        if self._protected(requested) or self._protected(canonical):
            self._deny_protected(requested, canonical, operation)
        expected = self._allowlist.get(canonical)
        if expected is None or self._protected_role(self._roles.get(canonical, "")):
            raise AssetAccessError(f"asset is not allowlisted: {canonical}")
        if required_role is not None and self._roles.get(canonical) != required_role:
            raise AssetAccessError(
                f"asset must have manifest role {required_role!r}: {canonical}"
            )
        try:
            return _open_descriptor_bound(canonical), canonical
        except (OSError, ValueError) as error:
            raise AssetAccessError("allowlisted asset is missing or unsafe") from error


    def read_bytes(
        self,
        path: str | Path,
        operation: str = "read",
        *,
        required_role: str | None = None,
    ) -> tuple[Path, bytes]:
        fd, canonical = self._open_authorized(path, operation, required_role)
        try:
            payload = b"".join(iter(lambda: os.read(fd, 1 << 20), b""))
        finally:
            os.close(fd)
        if hashlib.sha256(payload).hexdigest() != self._allowlist[canonical]:
            raise AssetAccessError(
                "asset SHA-256 does not match launch allowlist"
            )
        return canonical, payload

    def open(self, path: str | Path, mode: str = "rb"):
        if any(flag in mode for flag in ("w", "a", "+", "x")):
            raise AssetAccessError("firewall permits read-only asset access")
        _, payload = self.read_bytes(path, operation="open")
        if "b" in mode:
            return io.BytesIO(payload)
        return io.StringIO(payload.decode("utf-8"))

    def zero_access_receipt(self) -> dict[str, object]:
        entries = self._read_audit()
        attempts = sum(entry.get("type") == "DENIED" for entry in entries)
        return {"session_id": self.session_id, "audit_head_sha256": self._audit_head, "protected_access_attempts": attempts,
                "protected_access_allowed": 0, "zero_access": attempts == 0, "limitation": self.limitation}


def _validated_launch_manifest(
    manifest_path: str | Path, expected_manifest_sha256: str
) -> tuple[dict[str, Any], Path, dict[Path, str], dict[Path, str]]:
    if (
        not isinstance(expected_manifest_sha256, str)
        or not _SHA256.fullmatch(expected_manifest_sha256)
    ):
        raise ValueError(
            "expected manifest SHA-256 must be lowercase SHA-256 hex"
        )
    requested_manifest = Path(manifest_path).expanduser()
    try:
        raw = _read_descriptor_bound_bytes(requested_manifest)
        if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
            raise AssetAccessError(
                "launch asset manifest SHA-256 does not match independent pin"
            )
        document = json.loads(raw)
        manifest = requested_manifest.resolve(strict=True)
    except AssetAccessError:
        raise
    except (
        AttributeError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        raise ValueError("launch asset manifest is malformed") from error
    if not isinstance(document, dict):
        raise ValueError("launch asset manifest is malformed")
    assets = document.get("assets")
    if (
        document.get("schema_version") != 1
        or not isinstance(assets, list)
        or not assets
    ):
        raise ValueError(
            "launch asset manifest must have schema_version 1 and non-empty assets"
        )
    allowlist: dict[Path, str] = {}
    roles: dict[Path, str] = {}
    for asset in assets:
        if (
            not isinstance(asset, dict)
            or set(asset) != {"path", "sha256", "role"}
        ):
            raise ValueError(
                "each launch asset must contain exactly path, sha256, and role"
            )
        path, digest, role = asset["path"], asset["sha256"], asset["role"]
        if (
            not isinstance(path, str)
            or not isinstance(digest, str)
            or not isinstance(role, str)
            or not role
        ):
            raise ValueError("launch asset fields must be non-empty strings")
        resolved = Path(path).expanduser().resolve()
        if resolved in allowlist or not _SHA256.fullmatch(digest):
            raise ValueError(
                "launch asset paths must be unique with lowercase SHA-256 pins"
            )
        if AssetFirewall._protected(
            resolved
        ) or AssetFirewall._protected_role(role):
            raise ValueError(
                "heldout/probe assets and roles must never appear in a training allowlist"
            )
        allowlist[resolved], roles[resolved] = digest, role
    return document, manifest, allowlist, roles


def read_launch_asset_snapshot(
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    asset_path: str | Path,
    expected_role: str,
) -> tuple[Path, bytes]:
    """Read one bootstrap asset from its independently pinned manifest snapshot."""
    _, _, allowlist, roles = _validated_launch_manifest(
        manifest_path, expected_manifest_sha256
    )
    requested = Path(asset_path).expanduser()
    canonical = requested.resolve(strict=False)
    if roles.get(canonical) != expected_role:
        raise AssetAccessError(
            f"launch manifest must designate asset role {expected_role!r}"
        )
    expected = allowlist.get(canonical)
    if expected is None:
        raise AssetAccessError(f"asset is not allowlisted: {canonical}")
    try:
        payload = _read_descriptor_bound_bytes(canonical)
    except (OSError, ValueError) as error:
        raise AssetAccessError("allowlisted asset is missing or unsafe") from error
    if hashlib.sha256(payload).hexdigest() != expected:
        raise AssetAccessError(
            "asset SHA-256 does not match launch allowlist"
        )
    return canonical, payload


def load_launch_asset_manifest(
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    audit_path: str | Path,
) -> tuple[AssetFirewall, dict[str, Any]]:
    """Validate a pinned V2 launch manifest and construct its firewall."""
    document, manifest, allowlist, roles = _validated_launch_manifest(
        manifest_path, expected_manifest_sha256
    )
    firewall = AssetFirewall(allowlist, audit_path, roles)
    for path, role in roles.items():
        firewall.read_bytes(
            path, operation="launch-validation", required_role=role
        )
    document = dict(document)
    document.update(
        {
            "manifest_path": str(manifest),
            "manifest_sha256": expected_manifest_sha256,
            "asset_roles": {str(key): value for key, value in roles.items()},
        }
    )
    return firewall, document


def persist_launch_receipts(output_dir: str | Path, manifest: Mapping[str, Any], firewall: AssetFirewall) -> dict[str, str]:
    """Publish R1/R2 as one immutable, cross-linked receipt bundle."""
    root = Path(output_dir) / "reports" / "receipt-bundles"; root.mkdir(parents=True, exist_ok=True)
    staging = root / f".{uuid.uuid4().hex}.tmp"; staging.mkdir()
    published = False
    try:
        bundle_id = uuid.uuid4().hex
        r1_name = "r1_launch_allowlist.json"
        r2_name = "r2_zero_protected_access.json"
        r1 = {
            "schema_version": 1,
            "bundle_id": bundle_id,
            "peer_receipt": r2_name,
            "manifest_path": manifest["manifest_path"],
            "manifest_sha256": manifest["manifest_sha256"],
            "assets": manifest["assets"],
            "limitation": AssetFirewall.limitation,
        }
        r2 = {
            "schema_version": 1,
            "bundle_id": bundle_id,
            "peer_receipt": r1_name,
            **firewall.zero_access_receipt(),
        }
        _atomic_json(staging / r1_name, r1)
        _atomic_json(staging / r2_name, r2)
        directory = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        target = root / bundle_id
        _rename_noreplace(staging, target)
        published = True
        parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    first, second = target / r1_name, target / r2_name
    return {"r1": str(first), "r1_sha256": _sha256(first), "r2": str(second), "r2_sha256": _sha256(second)}


def generate_r3_r4_existence_receipt(protected_paths: Iterable[str | Path], fresh_heldout_paths: Iterable[str | Path]) -> dict[str, object]:
    """Record lstat-only R3/R4 evidence; content is never opened."""
    protected, fresh = list(protected_paths), list(fresh_heldout_paths)
    if not protected or not fresh: raise ValueError("R3 and R4 expected path lists must be nonempty")
    def metadata(paths: list[str | Path]) -> list[dict[str, object]]:
        records = []
        for value in paths:
            path = Path(value).expanduser().absolute()
            try:
                info = os.lstat(path)
                exists = True
                record: dict[str, object] = {"path": str(path), "exists": exists, "lstat_mode": info.st_mode, "symlink": stat.S_ISLNK(info.st_mode)}
                if not record["symlink"]: record.update({"size_bytes": info.st_size, "uid": info.st_uid, "gid": info.st_gid})
            except OSError as error:
                if error.errno == errno.ENOENT:
                    record = {"path": str(path), "exists": False}
                else:
                    record = {"path": str(path), "exists": True, "error": type(error).__name__}
            records.append(record)
        return records
    r3, r4 = metadata(protected), metadata(fresh)
    return {"schema_version": 1, "content_opened": False, "identity": {"uid": os.getuid(), "gid": os.getgid(), "cwd": os.getcwd()},
            "R3": {"records": r3, "pass": all(not r["exists"] for r in r3)}, "R4": {"records": r4, "pass": all(not r["exists"] for r in r4)}, "limitation": AssetFirewall.limitation}
