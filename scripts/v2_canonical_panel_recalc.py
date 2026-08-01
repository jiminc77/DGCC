#!/usr/bin/env python3
"""Canonical-panel selector diagnostics recalculation over completed cells.

Recomputes the fixed-development-panel diagnostics of every completed
tournament cell on the arm-independent canonical panel registered in
dossier/V2_canonical_panel_definition.md (rd#44 Amendment 4):

  - Charter §9.6 churn family: hard-selector churn, soft-weight JS divergence,
    soft-weight cosine, top-8 overlap, effective-contact count, top weight —
    via agent.selection_panel + compare_selection_snapshots across the cell's
    preserved checkpoints in transition order;
  - Charter §9.4 counterfactual selector, rollout-free component: Q1-vs-Qmin
    argmax agreement per checkpoint. The realized-progress difference needs
    environment rollouts and is intentionally out of scope for this CPU-only
    recalculation (recorded as a limitation in the definition document §4).

CPU-only, checkpoint-read-only; never touches training outputs beyond reading
preserved checkpoint files, never loads heldout data, writes only to the
dossier output path given on the command line.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dgcc.rl.panel_artifacts import load_panel  # noqa: E402
from dgcc.rl.selection import compare_selection_snapshots  # noqa: E402
from dgcc.rl.sprint_arms import create_sprint_agent  # noqa: E402
from dgcc.rl.td3 import TD3Config  # noqa: E402

FORBIDDEN_PATH_TOKENS = ("heldout", "held-out", "patch_eval", "patching_probe")
EXPECTED_CHECKPOINTS = 12
FINAL_TRANSITION = 300_032


def permitted_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    lowered = str(path).lower()
    for token in FORBIDDEN_PATH_TOKENS:
        if token in lowered:
            raise argparse.ArgumentTypeError(f"forbidden path scope: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arm_of_run_tag(run_tag: str) -> str:
    # t2_bb-d2_s0 -> bb ; t2_v1-d2_s1 -> v1 ; t2_v2-dmm_s2 -> v2-dmm
    middle = run_tag.split("_")[1]
    if middle.startswith("v2-"):
        return middle
    return middle.split("-")[0]


def discover_completed_cells(attempts_root: Path) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for attempt_dir in sorted(attempts_root.iterdir()):
        models = attempt_dir / "models"
        records = attempt_dir / "records.jsonl"
        if not models.is_dir() or not records.is_file():
            continue
        checkpoints = sorted(
            models.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1])
        )
        transitions = [int(p.stem.split("_")[1]) for p in checkpoints]
        if len(checkpoints) != EXPECTED_CHECKPOINTS or max(transitions, default=0) != FINAL_TRANSITION:
            continue
        run_tag = None
        with records.open() as handle:
            for line in handle:
                row = json.loads(line)
                if "run_tag" in row:
                    run_tag = str(row["run_tag"])
                    break
        if run_tag is None:
            raise RuntimeError(f"no run_tag in {records}")
        cells.append(
            {
                "attempt_id": attempt_dir.name,
                "run_tag": run_tag,
                "arm": arm_of_run_tag(run_tag),
                "checkpoints": checkpoints,
                "transitions": transitions,
            }
        )
    return cells


def run(args: argparse.Namespace) -> dict[str, Any]:
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only recalculation unexpectedly sees a CUDA device")

    panel_artifact = load_panel(
        args.panel, expected_canonical_sha256=args.expected_panel_sha256
    )
    with np.load(args.panel, allow_pickle=False) as data:
        order = np.asarray(data["order"])
        X = np.asarray(data["X"])[order]
        G = np.asarray(data["G"])[order]

    cells = discover_completed_cells(args.attempts_root)
    if args.expected_cells is not None and len(cells) != args.expected_cells:
        raise RuntimeError(
            f"expected {args.expected_cells} completed cells, found {len(cells)}: "
            f"{[c['run_tag'] for c in cells]}"
        )

    results: list[dict[str, Any]] = []
    for cell in cells:
        agent = create_sprint_agent(cell["arm"], TD3Config(policy_delay=2), device="cpu")
        previous_snapshot = None
        rows: list[dict[str, Any]] = []
        for transition, checkpoint in zip(cell["transitions"], cell["checkpoints"]):
            agent.load_checkpoint(checkpoint, eval_only=True)
            for module_name in ("encoder", "critic", "actor"):
                getattr(agent, module_name).eval()
            stats, snapshot = agent.selection_panel(X, G)
            row: dict[str, Any] = {
                "transition": int(transition),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "selection": {key: float(value) for key, value in stats.items()},
            }
            if previous_snapshot is not None:
                row["churn_vs_previous"] = {
                    key: float(value)
                    for key, value in compare_selection_snapshots(
                        snapshot, previous_snapshot
                    ).items()
                }
            previous_snapshot = snapshot
            rows.append(row)
        results.append(
            {
                "attempt_id": cell["attempt_id"],
                "run_tag": cell["run_tag"],
                "arm": cell["arm"],
                "checkpoints": rows,
            }
        )

    output = {
        "schema_version": 1,
        "device": "cpu",
        "data_scope": "development",
        "panel_path": str(args.panel),
        "panel_canonical_sha256": panel_artifact.canonical_sha256,
        "panel_artifact_sha256": panel_artifact.artifact_sha256,
        "panel_states": int(len(X)),
        "recalc_sha256": sha256_file(Path(__file__).resolve()),
        "scope_note": (
            "9.4 realized-progress difference requires environment rollouts and "
            "is excluded from this CPU-only recalculation by registered design."
        ),
        "cells": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=permitted_path, required=True)
    parser.add_argument("--expected-panel-sha256", required=True)
    parser.add_argument("--attempts-root", type=permitted_path, required=True)
    parser.add_argument("--expected-cells", type=int, default=None)
    parser.add_argument("--output", type=permitted_path, required=True)
    return parser.parse_args()


def main() -> int:
    result = run(parse_args())
    summary = {
        "cells": [
            {
                "run_tag": cell["run_tag"],
                "final_q1_qmin_agreement": cell["checkpoints"][-1]["selection"].get(
                    "q1_qmin_argmax_agreement"
                ),
            }
            for cell in result["cells"]
        ],
        "panel_canonical_sha256": result["panel_canonical_sha256"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
