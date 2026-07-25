"""Durable, fail-closed artifacts for the frozen V2 development panel."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from io import BytesIO
from typing import Any

import numpy as np

PANEL_SCHEMA = 1


@dataclass(frozen=True)
class PanelArtifact:
    path: Path
    canonical_sha256: str
    artifact_sha256: str
    metadata: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(X: np.ndarray, G: np.ndarray, order: np.ndarray, seed: int, transition: int, eval_ordinal: int, schema: int) -> bytes:
    digest = hashlib.sha256()
    for name, value in (("X", X), ("G", G), ("order", order)):
        array = np.ascontiguousarray(value)
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    digest.update(np.asarray([seed, transition, eval_ordinal, schema], dtype="<i8").tobytes())
    return digest.digest()


def canonical_panel_sha256(*, X: np.ndarray, G: np.ndarray, order: np.ndarray, seed: int, transition: int, eval_ordinal: int, schema: int = PANEL_SCHEMA) -> str:
    """Digest canonical content, intentionally independent of NPZ serialization bytes."""
    return _canonical_bytes(X, G, order, seed, transition, eval_ordinal, schema).hex()


def _atomic_replace(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
def _validate_panel(
    X: np.ndarray,
    G: np.ndarray,
    order: np.ndarray,
    *,
    seed: int,
    transition: int,
    eval_ordinal: int,
    schema: int,
) -> None:
    if (
        not isinstance(schema, (int, np.integer))
        or isinstance(schema, (bool, np.bool_))
        or schema != PANEL_SCHEMA
    ):
        raise ValueError(f"unsupported panel schema: {schema}")
    if any(
        not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
        for value in (seed, transition, eval_ordinal)
    ):
        raise ValueError("panel seed, transition, and eval ordinal must be non-boolean integers")
    if transition < 0 or eval_ordinal < 0:
        raise ValueError("panel transition and eval ordinal must be non-negative")
    if X.shape != G.shape or X.ndim != 3 or X.shape[1:] != (32, 3):
        raise ValueError("panel X and G must have matching shape (N, 32, 3)")
    if X.shape[0] == 0:
        raise ValueError("panel must contain at least one row")
    if X.dtype.kind not in "fiu" or G.dtype.kind not in "fiu":
        raise ValueError("panel X and G must be numeric")
    if not np.isfinite(X).all() or not np.isfinite(G).all():
        raise ValueError("panel X and G must be finite")
    if (
        order.shape != (X.shape[0],)
        or order.dtype.kind not in "iu"
        or not np.array_equal(np.sort(order), np.arange(X.shape[0]))
    ):
        raise ValueError("panel order must be an integer permutation of all rows")




def persist_panel(path: Path, *, X: np.ndarray, G: np.ndarray, order: np.ndarray, seed: int, transition: int, eval_ordinal: int, schema: int = PANEL_SCHEMA) -> PanelArtifact:
    X, G, order = np.asarray(X), np.asarray(G), np.asarray(order)
    _validate_panel(
        X,
        G,
        order,
        seed=seed,
        transition=transition,
        eval_ordinal=eval_ordinal,
        schema=schema,
    )
    canonical_sha256 = canonical_panel_sha256(X=X, G=G, order=order, seed=seed, transition=transition, eval_ordinal=eval_ordinal, schema=schema)
    _atomic_replace(path, lambda handle: np.savez(handle, X=X, G=G, order=order, seed=np.int64(seed), transition=np.int64(transition), eval_ordinal=np.int64(eval_ordinal), schema=np.int64(schema), canonical_sha256=np.asarray(canonical_sha256)))
    artifact_sha256 = _sha256_file(path)
    metadata = {"canonical_sha256": canonical_sha256, "artifact_sha256": artifact_sha256, "seed": int(seed), "transition": int(transition), "eval_ordinal": int(eval_ordinal), "schema": int(schema), "states": int(X.shape[0])}
    manifest = path.with_suffix(path.suffix + ".json")
    _atomic_replace(manifest, lambda handle: handle.write((json.dumps(metadata, sort_keys=True) + "\n").encode("utf-8")))
    return PanelArtifact(path, canonical_sha256, artifact_sha256, metadata)


def _load_panel_scalars(data: Any) -> tuple[int, int, int, int, str]:
    required = {
        "X", "G", "order", "seed", "transition", "eval_ordinal", "schema",
        "canonical_sha256",
    }
    if set(data.files) != required:
        raise ValueError("panel NPZ keys do not match schema-1 contract")
    scalars = {name: data[name] for name in ("seed", "transition", "eval_ordinal", "schema")}
    if any(value.shape != () or value.dtype != np.dtype("<i8") for value in scalars.values()):
        raise ValueError("panel integer scalars must be zero-dimensional int64 values")
    canonical = data["canonical_sha256"]
    if canonical.shape != () or canonical.dtype != np.dtype("<U64"):
        raise ValueError("panel canonical_sha256 must be a zero-dimensional U64 value")
    return (
        int(scalars["seed"].item()),
        int(scalars["transition"].item()),
        int(scalars["eval_ordinal"].item()),
        int(scalars["schema"].item()),
        str(canonical.item()),
    )


def _validate_manifest_metadata(metadata: Any, *, canonical: str, artifact: str, seed: int, transition: int, eval_ordinal: int, schema: int, states: int) -> dict[str, Any]:
    expected = {
        "canonical_sha256": canonical, "artifact_sha256": artifact, "seed": seed,
        "transition": transition, "eval_ordinal": eval_ordinal, "schema": schema,
        "states": states,
    }
    if not isinstance(metadata, dict) or set(metadata) != set(expected):
        raise ValueError("panel manifest keys do not match schema-1 contract")
    if (
        not isinstance(metadata["canonical_sha256"], str)
        or not isinstance(metadata["artifact_sha256"], str)
        or any(type(metadata[name]) is not int for name in ("seed", "transition", "eval_ordinal", "schema", "states"))
        or metadata != expected
    ):
        raise ValueError("panel manifest values do not match schema-1 contract")
    return metadata


def load_panel_bytes(
    panel_bytes: bytes,
    metadata_bytes: bytes,
    *,
    path: Path,
    expected_canonical_sha256: str | None = None,
    expected_artifact_sha256: str | None = None,
) -> tuple[PanelArtifact, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Validate a panel and manifest from one authenticated byte snapshot."""
    artifact = hashlib.sha256(panel_bytes).hexdigest()
    try:
        metadata = json.loads(metadata_bytes.decode("utf-8"))
        with np.load(BytesIO(panel_bytes), allow_pickle=False) as data:
            X, G, order = data["X"], data["G"], data["order"]
            seed, transition, eval_ordinal, schema, embedded = _load_panel_scalars(data)
        _validate_panel(
            X, G, order, seed=seed, transition=transition,
            eval_ordinal=eval_ordinal, schema=schema,
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"invalid panel artifact {path}: {exc}") from exc
    canonical = canonical_panel_sha256(
        X=X, G=G, order=order, seed=seed, transition=transition,
        eval_ordinal=eval_ordinal, schema=schema,
    )
    try:
        metadata = _validate_manifest_metadata(
            metadata, canonical=canonical, artifact=artifact, seed=seed,
            transition=transition, eval_ordinal=eval_ordinal, schema=schema,
            states=int(X.shape[0]),
        )
    except ValueError as exc:
        raise RuntimeError(f"invalid panel artifact {path}: {exc}") from exc
    if embedded != canonical:
        raise RuntimeError("panel artifact hash verification failed")
    if expected_canonical_sha256 is not None and canonical != expected_canonical_sha256:
        raise RuntimeError("paired arm panel SHA mismatch")
    if expected_artifact_sha256 is not None and artifact != expected_artifact_sha256:
        raise RuntimeError("panel artifact SHA mismatch")
    return PanelArtifact(path, canonical, artifact, metadata), (X, G, order)


def load_panel(path: Path, *, expected_canonical_sha256: str | None = None) -> PanelArtifact:
    manifest_path = path.with_suffix(path.suffix + ".json")
    try:
        panel_bytes = path.read_bytes()
        metadata_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"invalid panel artifact {path}: {exc}") from exc
    artifact, _arrays = load_panel_bytes(
        panel_bytes, metadata_bytes, path=path,
        expected_canonical_sha256=expected_canonical_sha256,
    )
    return artifact
