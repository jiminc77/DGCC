#!/usr/bin/env python3
"""Pin the executing Python environment beside the source code manifest.

The 93-file code manifest pins source bytes and nothing else. Every reviewed V2
artifact was produced under one torch build and one DLO-Lab revision, and the
isolated worktree silently had neither: a CPU-only torch and no simulator at
all. That is a provenance hole the code manifest structurally cannot close,
because identical source under a different accelerator build is a different
experiment.

This writes the environment counterpart: interpreter, accelerator torch build,
simulator revision, and the full resolved package closure with one digest over
it. It also refuses to pin an environment that imports dgcc or genesis from
outside the worktree it is pinning, which is the failure mode that would make
the code manifest a lie.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from importlib.metadata import distributions
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8") + b"\n"


def git_revision(repo: Path) -> dict[str, Any]:
    def run(*command: str) -> str:
        return subprocess.run(
            command, cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()

    return {
        "path": str(repo),
        "commit": run("git", "rev-parse", "HEAD"),
        "remote": run("git", "remote", "get-url", "origin"),
        "clean": run("git", "status", "--porcelain") == "",
    }


def collect(repo: Path) -> dict[str, Any]:
    import torch
    import genesis
    import dgcc

    dgcc_root = Path(dgcc.__file__).resolve().parent.parent.parent
    genesis_root = Path(genesis.__file__).resolve().parent.parent
    if dgcc_root != repo:
        raise ValueError(
            f"dgcc imports from {dgcc_root}, not the worktree being pinned ({repo}); "
            "the code manifest would not describe the code that actually runs"
        )
    if not genesis_root.is_relative_to(repo):
        raise ValueError(
            f"genesis imports from {genesis_root}, outside the worktree being pinned"
        )
    stray = [entry for entry in sys.path if entry and "Workspaces" in entry]
    if stray:
        raise ValueError(f"live-tree paths are importable: {stray}")

    packages = sorted(
        (dist.metadata["Name"].lower(), dist.version)
        for dist in distributions()
        if dist.metadata["Name"]
    )
    closure = [{"name": name, "version": version} for name, version in packages]
    accelerator: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        accelerator["device_name"] = torch.cuda.get_device_name(0)
        accelerator["device_capability"] = list(torch.cuda.get_device_capability(0))
        accelerator["driver_version"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    document: dict[str, Any] = {
        "schema_version": 1,
        "purpose": (
            "Prospective runtime-environment pin for the V2-DEV 15-run tournament. "
            "Binds the interpreter, accelerator torch build, simulator revision, and "
            "resolved package closure that the code manifest alone cannot express."
        ),
        "worktree": str(repo),
        "python": {
            "version": sys.version.split()[0],
            "implementation": sys.implementation.name,
        },
        "accelerator": accelerator,
        "simulator": {
            "distribution": "genesis-world",
            "version": genesis.__version__,
            "install": "editable",
            **git_revision(genesis_root),
        },
        "dgcc_install": {"mode": "editable", "path": str(dgcc_root)},
        "package_closure": closure,
        "package_count": len(closure),
    }
    document["lockfile_digest_sha256"] = hashlib.sha256(
        canonical_json(closure)
    ).hexdigest()
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    repo = args.repo_root.resolve(strict=True)
    document = collect(repo)
    payload = canonical_json(document)
    if args.output.exists():
        os.chmod(args.output, 0o644)
    args.output.write_bytes(payload)
    os.chmod(args.output, 0o444)
    print(
        canonical_json(
            {
                "runtime_environment_sha256": hashlib.sha256(payload).hexdigest(),
                "lockfile_digest_sha256": document["lockfile_digest_sha256"],
                "torch": document["accelerator"]["torch_version"],
                "genesis_commit": document["simulator"]["commit"],
                "packages": document["package_count"],
            }
        ).decode(),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
