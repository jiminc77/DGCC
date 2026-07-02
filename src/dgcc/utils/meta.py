"""Run metadata helpers for generated P0 artifacts."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def get_git_commit_hash(repo_root: str | Path = ".") -> str:
    """Return the current git commit hash, or ``"unknown"`` outside a repo."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repo_root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    commit = result.stdout.strip()
    return commit or "unknown"


def build_run_metadata(*, config: dict[str, Any], seed: int, repo_root: str | Path = ".") -> dict[str, Any]:
    """Build the small reproducibility metadata block required by P0 §2.7."""

    return {
        "seed": int(seed),
        "config": config,
        "commit_hash": get_git_commit_hash(repo_root),
    }
