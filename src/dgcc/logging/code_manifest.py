"""Closed-schema validation of executable project code manifests."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

_DIGEST_CHARS = frozenset("0123456789abcdef")
_CODE_ROOTS = ("src/dgcc", "scripts")


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _reject_duplicate_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("code manifest contains duplicate object fields")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - _DIGEST_CHARS:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _read_runtime_file(root: Path, relative_path: str) -> bytes:
    """Read one repo-relative regular file through no-follow descriptors."""
    parts = Path(relative_path).parts
    if not parts or Path(relative_path).is_absolute() or ".." in parts:
        raise ValueError("runtime entry is missing or unsafe")
    try:
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        fd = root_fd
        try:
            for part in parts[:-1]:
                next_fd = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
                )
                os.close(fd)
                fd = next_fd
            file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
            try:
                if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                    raise ValueError("runtime entry is not a regular file")
                chunks: list[bytes] = []
                while chunk := os.read(file_fd, 1 << 20):
                    chunks.append(chunk)
                return b"".join(chunks)
            finally:
                os.close(file_fd)
        finally:
            os.close(fd)
    except OSError as error:
        raise ValueError("runtime entry is missing or unsafe") from error


def required_runtime_files(runtime_root: Path) -> tuple[str, ...]:
    """Return the complete sorted Python closure, rejecting unsafe project entries."""
    root = Path(runtime_root)
    files: list[str] = []
    for relative_root in _CODE_ROOTS:
        directory = root / relative_root
        try:
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError("runtime code root is missing or unsafe")
            for current, directories, names in os.walk(directory, followlinks=False):
                current_path = Path(current)
                for name in directories:
                    if (current_path / name).is_symlink():
                        raise ValueError("runtime code closure contains a symlink")
                for name in names:
                    candidate = current_path / name
                    if name.endswith(".py"):
                        if candidate.is_symlink() or not candidate.is_file():
                            raise ValueError("runtime code closure contains an unsafe Python file")
                        files.append(candidate.relative_to(root).as_posix())
        except OSError as error:
            raise ValueError("runtime code root is missing or unsafe") from error
    return tuple(sorted(files))


def validate_code_manifest_bytes(
    code_manifest_bytes: bytes, *, runtime_root: Path
) -> dict[str, Any]:
    """Validate exact manifest bytes against the executing project's Python closure."""
    try:
        document = json.loads(code_manifest_bytes, object_pairs_hook=_reject_duplicate_json_object)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("code manifest is malformed") from error
    if not isinstance(document, dict) or set(document) != {"schema_version", "files"}:
        raise ValueError("code manifest has unknown or missing fields")
    if document["schema_version"] != 1 or not isinstance(document["files"], list):
        raise ValueError("code manifest has invalid schema")

    entries: dict[str, str] = {}
    for entry in document["files"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ValueError("code manifest entry has unknown or missing fields")
        path, digest = entry["path"], entry["sha256"]
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or Path(path).as_posix() != path
        ):
            raise ValueError("code manifest entry path is unsafe")
        _digest(digest, f"code manifest digest for {path}")
        if path in entries:
            raise ValueError("code manifest contains duplicate entries")
        entries[path] = digest

    expected_paths = required_runtime_files(runtime_root)
    if tuple(sorted(entries)) != expected_paths:
        raise ValueError("code manifest must contain exactly the required runtime closure")
    actual = {path: hashlib.sha256(_read_runtime_file(runtime_root, path)).hexdigest() for path in expected_paths}
    if actual != entries:
        raise ValueError("code manifest runtime closure does not match executable files")
    closure = canonical_json([{"path": path, "sha256": actual[path]} for path in expected_paths])
    return {
        "code_manifest_sha256": hashlib.sha256(code_manifest_bytes).hexdigest(),
        "code_closure_sha256": hashlib.sha256(closure).hexdigest(),
        "code_closure_count": len(expected_paths),
    }
