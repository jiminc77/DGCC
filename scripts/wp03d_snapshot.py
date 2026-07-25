#!/usr/bin/env python3
"""Create a read-only WP03D snapshot from an explicit provenance manifest only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

MIN_FREE_BYTES = 20 * 1024**3
RETRIES = 2
EXPECTED_COUNTS = {"checkpoint": 10, "raw_gz": 3, "panel": 1}
SNAPSHOT_ROOT = Path("/home/simx2204/v2_research/snapshots")
PRODUCTION_SOURCE_ROOTS = (Path("/home/simx2204/v2_research/outputs"),)
SNAPSHOT_SCOPE_METADATA = {
    "closure_scope": "tested import/update trajectory only",
    "excluded_dormant_resources": ["historical T2 split resource (not copied or claimed closed)"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_free_space(destination_parent: Path) -> None:
    if shutil.disk_usage(destination_parent).free < MIN_FREE_BYTES:
        raise OSError(f"snapshot destination has less than {MIN_FREE_BYTES} free bytes")


def _reject_symlink_components(path: Path) -> None:
    path = path.absolute()
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError(f"symlink components are not permitted: {path}")


def load_provenance(path: Path, *, source_roots: tuple[Path, ...] = PRODUCTION_SOURCE_ROOTS) -> list[dict[str, str]]:
    _reject_symlink_components(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if set(document) != {"assets"} or not isinstance(document["assets"], list):
        raise ValueError("provenance manifest must contain only an assets list")
    approved_roots = tuple(root.resolve() for root in source_roots)
    if not approved_roots:
        raise ValueError("at least one approved source root is required")
    normalized_assets: list[dict[str, str]] = []
    counts = {kind: 0 for kind in EXPECTED_COUNTS}
    destinations: set[str] = set()
    for asset in document["assets"]:
        allowed_fields = {"kind", "source", "destination", "sha256"}
        if not isinstance(asset, dict) or set(asset) != allowed_fields:
            raise ValueError("each asset must provide kind, source, destination, and pinned sha256")
        kind = asset["kind"]
        if kind not in counts:
            raise ValueError(f"unapproved asset kind: {kind}")
        source = Path(asset["source"])
        destination = Path(asset["destination"])
        if (
            not source.is_absolute()
            or ".." in source.parts
            or destination.is_absolute()
            or ".." in destination.parts
        ):
            raise ValueError("sources must be absolute without '..' and destinations safely relative")
        _reject_symlink_components(source)
        source = source.resolve()
        if not any(source.is_relative_to(root) for root in approved_roots):
            raise ValueError(f"source is outside approved roots: {source}")
        if len(asset["sha256"]) != 64 or any(char not in "0123456789abcdef" for char in asset["sha256"]):
            raise ValueError("source sha256 must be a lowercase 64-hex authenticated pin")
        if kind == "raw_gz" and source.suffix != ".gz":
            raise ValueError("raw_gz source must have .gz suffix")
        if str(destination) in destinations:
            raise ValueError("destination paths must be unique")
        destinations.add(str(destination))
        normalized_assets.append({**asset, "source": str(source)})
        counts[kind] += 1
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"snapshot requires exactly {EXPECTED_COUNTS}, got {counts}")
    return normalized_assets
def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _published_file_set(assets: list[dict[str, str]]) -> set[Path]:
    return {Path(asset["destination"]) for asset in assets} | {Path("MANIFEST.json")}


def _verify_staged_snapshot(staging: Path, assets: list[dict[str, str]]) -> None:
    expected = _published_file_set(assets)
    actual = {path.relative_to(staging) for path in staging.rglob("*") if path.is_file()}
    if actual != expected:
        raise RuntimeError(f"staged snapshot file set mismatch: expected {expected}, got {actual}")
    for asset in assets:
        target = staging / asset["destination"]
        if sha256_file(target) != asset["sha256"]:
            raise RuntimeError(f"staged snapshot payload hash mismatch: {target}")


def _fsync_tree(staging: Path) -> None:
    for path in sorted(staging.rglob("*")):
        if path.is_file():
            _fsync_file(path)
    for path in sorted((path for path in staging.rglob("*") if path.is_dir()), reverse=True):
        _fsync_dir(path)
    _fsync_dir(staging)


def verify_snapshot(snapshot_path: Path, *, snapshot_root: Path = SNAPSHOT_ROOT) -> None:
    """Verify published payload digests without trusting the source files."""
    snapshot_path = snapshot_path.absolute()
    snapshot_root = snapshot_root.absolute()
    _reject_symlink_components(snapshot_root)
    if not snapshot_path.is_relative_to(snapshot_root):
        raise ValueError(f"snapshot must be under {snapshot_root}")
    _reject_symlink_components(snapshot_path)
    manifest = json.loads((snapshot_path / "MANIFEST.json").read_text(encoding="utf-8"))
    assets = manifest.get("assets")
    if not isinstance(assets, list):
        raise ValueError("published manifest has no assets list")
    _verify_staged_snapshot(snapshot_path, assets)


def snapshot(
    provenance: Path,
    destination: Path,
    *,
    source_roots: tuple[Path, ...] = PRODUCTION_SOURCE_ROOTS,
    snapshot_root: Path = SNAPSHOT_ROOT,
) -> dict[str, Any]:
    """Copy exactly authenticated assets into a fresh sibling then publish atomically."""
    destination = destination.absolute()
    snapshot_root = snapshot_root.absolute()
    _reject_symlink_components(snapshot_root)
    _reject_symlink_components(destination)
    destination = destination.resolve()
    if not destination.is_relative_to(snapshot_root):
        raise ValueError(f"snapshot destination must be under {snapshot_root}")
    _reject_symlink_components(destination.parent)
    assets = load_provenance(provenance, source_roots=source_roots)
    if destination.exists():
        raise FileExistsError(f"snapshot destination must be absent: {destination}")
    destination_parent = destination.parent
    destination_parent.mkdir(parents=True, exist_ok=True)
    require_free_space(destination_parent)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination_parent))
    try:
        copied: list[dict[str, Any]] = []
        for asset in assets:
            source = Path(asset["source"])
            _reject_symlink_components(source)
            target = staging / asset["destination"]
            if not source.is_file() or source.is_symlink():
                raise FileNotFoundError(f"enumerated source is not a regular file: {source}")
            target.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(1, RETRIES + 1):
                require_free_space(destination_parent)
                before = sha256_file(source)
                expected = asset.get("sha256")
                if expected is not None and before != expected:
                    raise RuntimeError(f"enumerated source does not match pinned SHA-256: {source}")
                shutil.copyfile(source, target)
                copied_hash = sha256_file(target)
                after = sha256_file(source)
                if before == copied_hash == after:
                    os.chmod(target, 0o444)
                    copied.append(
                        {
                            "kind": asset["kind"],
                            "source_path": str(source),
                            "destination": asset["destination"],
                            "sha256": after,
                            "expected_sha256": expected,
                            "size_bytes": source.stat().st_size,
                            "attempt": attempt,
                        }
                    )
                    break
                target.unlink(missing_ok=True)
            else:
                raise RuntimeError(f"source/copy SHA-256 mismatch after {RETRIES} attempts: {source}")
        manifest = {
            "schema_version": 1,
            "assets": copied,
            "source_mutation_attestation": {
                "zero_mutation_observed": True,
                "method": "source SHA-256 immediately before and after each copy matched copied SHA-256",
            },
            "snapshot_scope": SNAPSHOT_SCOPE_METADATA,
        }
        manifest_path = staging / "MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(manifest_path, 0o444)
        _verify_staged_snapshot(staging, copied)
        _fsync_tree(staging)
        if destination.exists():
            raise FileExistsError(f"snapshot destination appeared during staging: {destination}")
        os.rename(staging, destination)
        _fsync_dir(destination_parent)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        verify_snapshot(args.verify)
        return 0
    print(json.dumps(snapshot(args.provenance, args.destination), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
