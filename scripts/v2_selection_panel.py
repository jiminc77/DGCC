#!/usr/bin/env python3
"""CPU-only fixed-development-panel selector diagnostics for every V2 arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

from dgcc.rl.selection import compare_selection_snapshots
from dgcc.rl.sprint_arms import create_sprint_agent
from dgcc.rl.td3 import TD3Config

FORBIDDEN_PATH_TOKENS = ("heldout", "held-out", "patch_eval", "patching_probe")
ARMS = ("bb", "v1", "v2-dmm", "v2-d1m", "v2-d11")


def permitted_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    lowered = str(path).lower()
    if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
        raise ValueError(
            f"selection diagnostics refuse confirmatory/probe path: {path}"
        )
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_panel(path: Path) -> dict[str, np.ndarray]:
    arrays = np.load(path, allow_pickle=False)
    required = {
        "X",
        "G",
        "q1_realized_progress",
        "qmin_realized_progress",
        "checkpoint_sha256",
    }
    missing = required - set(arrays.files)
    if missing:
        raise ValueError(
            f"development panel is missing counterfactual fields: {sorted(missing)}"
        )
    panel = {name: np.asarray(arrays[name]) for name in required}
    if panel["X"].shape != panel["G"].shape or panel["X"].shape[1:] != (32, 3):
        raise ValueError("panel X/G must have matching shape (B, 32, 3)")
    count = panel["X"].shape[0]
    for name in ("q1_realized_progress", "qmin_realized_progress"):
        if panel[name].shape != (count,) or not np.isfinite(panel[name]).all():
            raise ValueError(f"panel {name} must be finite with shape (B,)")
    return panel


def agent_for_arm(arm: str) -> Any:
    return create_sprint_agent(arm, TD3Config(policy_delay=2), device="cpu")


def load_checkpoint(agent: Any, path: Path) -> None:
    agent.load_checkpoint(path, eval_only=True)
    for module_name in ("encoder", "critic", "actor"):
        getattr(agent, module_name).eval()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only diagnostics unexpectedly see a CUDA device")
    panel = load_panel(args.panel)
    expected_checkpoint = str(panel["checkpoint_sha256"].item())
    if expected_checkpoint != sha256_file(args.checkpoint):
        raise ValueError("counterfactual panel was not generated for --checkpoint")
    current = agent_for_arm(args.arm)
    previous = agent_for_arm(args.arm)
    load_checkpoint(current, args.checkpoint)
    load_checkpoint(previous, args.previous_checkpoint)

    state_before = {
        f"{module_name}.{name}": value.detach().clone()
        for module_name in ("encoder", "critic", "actor")
        for name, value in getattr(current, module_name).state_dict().items()
    }
    current_stats, current_snapshot = current.selection_panel(panel["X"], panel["G"])
    _, previous_snapshot = previous.selection_panel(panel["X"], panel["G"])
    temporal = compare_selection_snapshots(current_snapshot, previous_snapshot)
    state_unchanged = all(
        torch.equal(value, state_before[f"{module_name}.{name}"])
        for module_name in ("encoder", "critic", "actor")
        for name, value in getattr(current, module_name).state_dict().items()
    )
    if not state_unchanged:
        raise RuntimeError("selection diagnostics mutated model state")

    q1_progress = panel["q1_realized_progress"].astype(float)
    qmin_progress = panel["qmin_realized_progress"].astype(float)
    histogram = torch.bincount(current_snapshot.q1_selected, minlength=32).tolist()
    result = {
        "schema_version": 1,
        "device": "cpu",
        "data_scope": "development",
        "arm": args.arm,
        "states": len(panel["X"]),
        "panel_sha256": sha256_file(args.panel),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "previous_checkpoint_sha256": sha256_file(args.previous_checkpoint),
        "model_state_unchanged": state_unchanged,
        "selection": current_stats,
        "checkpoint_comparison": temporal,
        "contact_histogram_counts": histogram,
        "counterfactual_selector": {
            "p_q1_p_qmin_agreement": current_stats["q1_qmin_argmax_agreement"],
            "q1_selected_realized_progress_mean": float(q1_progress.mean()),
            "qmin_selected_realized_progress_mean": float(qmin_progress.mean()),
            "q1_minus_qmin_realized_progress_mean": float(
                (q1_progress - qmin_progress).mean()
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--checkpoint", type=permitted_path, required=True)
    parser.add_argument("--previous-checkpoint", type=permitted_path, required=True)
    parser.add_argument("--panel", type=permitted_path, required=True)
    parser.add_argument("--output", type=permitted_path, required=True)
    return parser.parse_args()


def main() -> int:
    result = run(parse_args())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
