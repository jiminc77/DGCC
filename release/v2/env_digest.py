#!/usr/bin/env python3
"""One-line runtime-environment digest for the per-run ledger entry.

The environment is pinned prospectively at preflight in
``v2_runtime_environment.json``, but the launcher does not verify it: adding
that check would change the pinned runtime closure and trigger another manifest
cascade immediately before the tournament. Per-run verification is therefore
operational, and this is the operation.

Run it immediately before each governed launch and paste the single JSON line
into the ledger. It recomputes the package closure digest exactly the way the
pin generator computed it, so ``matches_pin`` is a real comparison and not a
restatement. Exit status is non-zero on drift, so it can gate a launch script.

    uv run python release/v2/env_digest.py --repo-root . \\
        --pin release/v2/v2_runtime_environment.json

This is an operational control, not a launcher-enforced one. It cannot detect
drift introduced after it runs and before the process starts; that window is
small because both happen in the same session on the same host and venv, but it
is not zero. Launcher-level enforcement is deferred to post-tournament.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib.metadata import distributions
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8") + b"\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--pin", type=Path, required=True)
    args = parser.parse_args(argv)
    repo = args.repo_root.resolve(strict=True)
    sys.path.insert(0, str(repo / "src"))

    import torch
    import genesis
    import dgcc

    closure = [
        {"name": dist.metadata["Name"].lower(), "version": dist.version}
        for dist in distributions()
        if dist.metadata["Name"]
    ]
    closure.sort(key=lambda entry: (entry["name"], entry["version"]))
    lockfile_digest = hashlib.sha256(canonical_json(closure)).hexdigest()

    pin_bytes = args.pin.read_bytes()
    pin = json.loads(pin_bytes)
    genesis_root = Path(genesis.__file__).resolve().parent.parent
    import subprocess

    genesis_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=genesis_root,
        check=True, capture_output=True, text=True,
    ).stdout.strip()

    observed = {
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "genesis_world_commit": genesis_commit,
        "lockfile_digest_sha256": lockfile_digest,
        "dgcc_path": str(Path(dgcc.__file__).resolve().parent.parent.parent),
    }
    expected = {
        "torch_version": pin["accelerator"]["torch_version"],
        "torch_cuda_build": pin["accelerator"]["torch_cuda_build"],
        "cuda_available": True,
        "genesis_world_commit": pin["simulator"]["commit"],
        "lockfile_digest_sha256": pin["lockfile_digest_sha256"],
        "dgcc_path": str(repo),
    }
    drift = {
        key: {"expected": expected[key], "observed": observed[key]}
        for key in expected
        if expected[key] != observed[key]
    }
    line = {
        "env_digest_schema": 1,
        "runtime_environment_pin_sha256": hashlib.sha256(pin_bytes).hexdigest(),
        **observed,
        "matches_pin": not drift,
    }
    if drift:
        line["drift"] = drift
    print(canonical_json(line).decode(), end="")
    return 0 if not drift else 1


if __name__ == "__main__":
    raise SystemExit(main())
