#!/usr/bin/env python3
"""Prove the historical M4 training-source parity and freeze its source tree."""

from __future__ import annotations

import ast
import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
COMMITS = ("cdd73e2", "b24997f", "786d651")
BUNDLE_SOURCE_COMMIT = "786d651"
CLOSURE_PATHS = (
    "src/dgcc/models",
    "src/dgcc/rl",
    "src/dgcc/envs",
    "src/dgcc/tasks",
    "src/dgcc/goals",
    "src/dgcc/phi",
    "src/dgcc/utils",
    "src/dgcc/logging",
    "src/dgcc/__init__.py",
    "scripts/p1_train.py",
    "configs/p1_t2.yaml",
    "uv.lock",
    "pyproject.toml",
)
EVAL_ONLY_EXCEPTION = "src/dgcc/tasks/splits/t2_sprint_heldout_v1.json"
STATIC_SCAN_PATHS = "all Python blobs in the training closure"
PROOF_PATH = ROOT / "outputs/metrics/sprint_bb_parity_proof.json"
BUNDLE_PATH = ROOT / "outputs/models/frozen_m4_bundle"


def git(*args: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=text)


def blob_map(commit: str, paths: Iterable[str] | None = None) -> dict[str, str]:
    output = git("ls-tree", "-r", commit, "--", *(paths or CLOSURE_PATHS))
    result: dict[str, str] = {}
    for line in output.splitlines():
        metadata, path = line.split("\t", 1)
        _mode, object_type, object_id = metadata.split(" ", 2)
        if object_type != "blob":
            raise RuntimeError(f"non-blob in training closure: {path}")
        if path == EVAL_ONLY_EXCEPTION:
            continue
        result[path] = object_id
    if not result:
        raise RuntimeError(f"empty training closure for {commit}")
    return result


def closure_top_level_packages(blobs: dict[str, str]) -> set[str]:
    packages = set()
    for path in blobs:
        if path.startswith("src/dgcc/"):
            relative = path.removeprefix("src/dgcc/")
            if "/" in relative:
                packages.add(relative.split("/", 1)[0])
    return packages


def imported_top_level_packages(commit: str, blobs: dict[str, str]) -> set[str]:
    required: set[str] = set()
    for path, blob in blobs.items():
        if not path.endswith(".py"):
            continue
        tree = ast.parse(git("cat-file", "blob", blob, text=False), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = (item.name for item in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = (node.module,)
                if node.module == "dgcc":
                    names = (f"dgcc.{item.name}" for item in node.names)
            else:
                continue
            for name in names:
                parts = name.split(".")
                if len(parts) >= 2 and parts[0] == "dgcc":
                    required.add(parts[1])
    return required


def assert_transitive_import_closure(commit: str, blobs: dict[str, str]) -> None:
    missing = imported_top_level_packages(commit, blobs) - closure_top_level_packages(blobs)
    if "src/dgcc/__init__.py" not in blobs:
        missing.add("__init__.py")
    if missing:
        raise RuntimeError(
            f"training closure misses imported dgcc packages for {commit}: {', '.join(sorted(missing))}"
        )


def exception_blobs(commits: Iterable[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for commit in commits:
        output = git("ls-tree", "-r", commit, "--", EVAL_ONLY_EXCEPTION).strip()
        result[commit] = output.split("\t", 1)[0].split(" ", 2)[2] if output else None
    return result


def static_reference_scan(
    commits: Iterable[str], maps: dict[str, dict[str, str]]
) -> dict[str, object]:
    references: dict[str, list[str]] = {}
    for commit in commits:
        hits: list[str] = []
        for path, blob in maps[commit].items():
            if not path.endswith(".py"):
                continue
            source = git("cat-file", "blob", blob, text=False).decode()
            if (
                re.search(r"\bt2_sprint_[A-Za-z0-9_]*", source)
                or 'load_t2_split("heldout")' in source
                or "load_t2_split('heldout')" in source
            ):
                hits.append(path)
        references[commit] = hits
    return {
        "scanned_paths": STATIC_SCAN_PATHS,
        "scanned_python_blobs": {commit: sorted(path for path in maps[commit] if path.endswith(".py")) for commit in commits},
        "needles": [
            "t2_sprint_*",
            'load_t2_split("heldout")',
            "load_t2_split('heldout')",
        ],
        "references": references,
        "all_clear": not any(references.values()),
    }


def compare_blob_maps(maps: dict[str, dict[str, str]], injected_mismatch: bool = False) -> list[dict[str, object]]:
    mutable = {commit: dict(entries) for commit, entries in maps.items()}
    if injected_mismatch:
        first_commit, second_commit = COMMITS[:2]
        path = next(iter(mutable[first_commit]))
        mutable[second_commit][path] = "0" * 40
    paths = sorted(set().union(*[entries.keys() for entries in mutable.values()]))
    return [
        {"path": path, "blobs": {commit: mutable[commit].get(path) for commit in COMMITS}}
        for path in paths
        if len({mutable[commit].get(path) for commit in COMMITS}) != 1
    ]


def build_proof(injected_mismatch: bool = False) -> tuple[dict[str, object], dict[str, dict[str, str]]]:
    maps = {commit: blob_map(commit) for commit in COMMITS}
    for commit, blobs in maps.items():
        assert_transitive_import_closure(commit, blobs)
    scan = static_reference_scan(COMMITS, maps)
    mismatches = compare_blob_maps(maps, injected_mismatch=injected_mismatch)
    raw_exception = exception_blobs(COMMITS)
    verdict = "PASS" if not mismatches and scan["all_clear"] else "FAIL"
    proof: dict[str, object] = {
        "schema_version": 1,
        "commits": list(COMMITS),
        "closure_paths": list(CLOSURE_PATHS),
        "closure_file_count": len(maps[BUNDLE_SOURCE_COMMIT]),
        "closure_blobs": maps,
        "mismatches": mismatches,
        "eval_only_exception": {
            "path": EVAL_ONLY_EXCEPTION,
            "raw_blobs": raw_exception,
            "exclusion_reason": "Evaluation-only sprint held-out split; it is not consumed by the training path.",
            "static_training_path_scan": scan,
            "claim": "All Python blobs in the included training closure were scanned for held-out split references.",
        },
        "verdict": verdict,
    }
    return proof, maps


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_for(bundle: Path, blobs: dict[str, str]) -> dict[str, str]:
    return {path: sha256_bytes((bundle / path).read_bytes()) for path in sorted(blobs)}


def read_manifest(path: Path) -> dict[str, str]:
    try:
        return {
            source_path: digest
            for digest, source_path in (
                line.split("  ", 1) for line in path.read_text().splitlines() if line
            )
        }
    except ValueError as error:
        raise RuntimeError("existing frozen bundle manifest is malformed") from error


def bundle_files(bundle: Path) -> set[str]:
    return {path.relative_to(bundle).as_posix() for path in bundle.rglob("*") if path.is_file()}


def freeze_bundle(blobs: dict[str, str]) -> None:
    manifest_path = BUNDLE_PATH / "MANIFEST.sha256"
    metadata_path = BUNDLE_PATH / "bundle_metadata.json"
    expected = {path: sha256_bytes(git("cat-file", "blob", blob, text=False)) for path, blob in blobs.items()}
    expected_files = set(expected) | {"MANIFEST.sha256", "bundle_metadata.json"}
    if BUNDLE_PATH.exists():
        actual_files = bundle_files(BUNDLE_PATH)
        if actual_files != expected_files:
            raise RuntimeError(
                f"existing frozen bundle file set differs from expected: "
                f"missing={sorted(expected_files - actual_files)}, extra={sorted(actual_files - expected_files)}"
            )
        actual = manifest_for(BUNDLE_PATH, blobs)
        if actual != expected:
            raise RuntimeError("existing frozen bundle differs from verified source")
        if read_manifest(manifest_path) != expected:
            raise RuntimeError("existing frozen bundle manifest differs from verified source")
        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError as error:
            raise RuntimeError("existing frozen bundle metadata is malformed") from error
        if (
            metadata.get("source_commit") != BUNDLE_SOURCE_COMMIT
            or metadata.get("source_blobs") != dict(sorted(blobs.items()))
        ):
            raise RuntimeError("existing frozen bundle metadata differs from verified source")
        return
    for path, blob in blobs.items():
        destination = BUNDLE_PATH / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(git("cat-file", "blob", blob, text=False))
    BUNDLE_PATH.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("".join(f"{digest}  {path}\n" for path, digest in sorted(expected.items())))
    metadata = {
        "schema_version": 1,
        "source_commit": BUNDLE_SOURCE_COMMIT,
        "source_blobs": dict(sorted(blobs.items())),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "script_git_revision": git("rev-parse", "HEAD").strip(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser().parse_args(argv)
    try:
        proof, maps = build_proof()
        PROOF_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROOF_PATH.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
        if proof["verdict"] != "PASS":
            return 1
        freeze_bundle(maps[BUNDLE_SOURCE_COMMIT])
        return 0
    except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
        print(f"sprint parity proof failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
