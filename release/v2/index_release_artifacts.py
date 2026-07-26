#!/usr/bin/env python3
"""Recompute the published preflight file index and the release artifact index.

Both indexes are pure functions of files already on disk. Keeping them in a
script rather than recomputing them by hand is what makes a regeneration
auditable: the pins move together or the script fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8") + b"\n"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> str:
    payload = canonical_json(value)
    if path.exists():
        os.chmod(path, 0o644)
    path.write_bytes(payload)
    os.chmod(path, 0o444)
    return hashlib.sha256(payload).hexdigest()


def build_tree_index(repo: Path, tree: Path) -> dict[str, Any]:
    files = sorted(
        (entry for entry in tree.rglob("*") if entry.is_file()),
        key=lambda entry: entry.relative_to(repo).as_posix(),
    )
    document: dict[str, Any] = {
        "schema_version": 1,
        "root": tree.relative_to(repo).as_posix(),
        "files": [
            {
                "path": entry.relative_to(repo).as_posix(),
                "sha256": sha256_file(entry),
                "size_bytes": entry.stat().st_size,
            }
            for entry in files
        ],
    }
    document["tree_sha256"] = hashlib.sha256(canonical_json(document)).hexdigest()
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args(argv)
    repo = args.repo_root.resolve(strict=True)
    release = repo / "release" / "v2"
    tree = release / "preflight_15_not_admitted"

    index = build_tree_index(repo, tree)
    index_sha = write_json(release / "preflight_15_not_admitted.sha256.json", index)

    artifacts_path = release / "release_artifacts.sha256.json"
    artifacts = json.loads(artifacts_path.read_bytes())
    known = {entry["path"]: entry for entry in artifacts["artifacts"]}
    known.setdefault(
        "release/v2/launch_dryrun_probe.py",
        {
            "path": "release/v2/launch_dryrun_probe.py",
            "role": "real-launcher-gate-dry-run-probe",
        },
    )
    known.setdefault(
        "release/v2/v2_runtime_environment.json",
        {
            "path": "release/v2/v2_runtime_environment.json",
            "role": "prospective-runtime-environment-pin",
        },
    )
    known.setdefault(
        "release/v2/env_digest.py",
        {
            "path": "release/v2/env_digest.py",
            "role": "operational-per-run-environment-digest",
        },
    )
    known.setdefault(
        "release/v2/pin_runtime_environment.py",
        {
            "path": "release/v2/pin_runtime_environment.py",
            "role": "runtime-environment-pin-generator",
        },
    )
    known.setdefault(
        "release/v2/index_release_artifacts.py",
        {
            "path": "release/v2/index_release_artifacts.py",
            "role": "release-index-regenerator",
        },
    )
    known["release/v2/preflight_15_not_admitted.sha256.json"]["role"] = (
        f"authoritative-{len(index['files'])}-file-preflight-index"
    )
    for path, entry in known.items():
        entry["sha256"] = sha256_file(repo / path)
    artifacts["artifacts"] = [known[key] for key in sorted(known)]
    disposition = json.loads((release / "bgt_not_admitted.json").read_bytes())
    artifacts["bgt"]["disposition_sha256"] = disposition["disposition_sha256"]
    matrix = json.loads(
        (release / "preflight_15_not_admitted" / "preflight_matrix.json").read_bytes()
    )
    artifacts["runtime_code_manifest_sha256"] = matrix["code_manifest"][
        "code_manifest_sha256"
    ]
    # source_commit alone would read as "runtime is exactly that commit".
    artifacts["runtime_patches"] = matrix["runtime_patches"]
    artifacts_sha = write_json(artifacts_path, artifacts)

    print(
        canonical_json(
            {
                "preflight_index_sha256": index_sha,
                "preflight_tree_sha256": index["tree_sha256"],
                "indexed_files": len(index["files"]),
                "release_artifacts_sha256": artifacts_sha,
            }
        ).decode(),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
