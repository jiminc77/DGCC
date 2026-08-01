#!/usr/bin/env python3
"""Arm-independent canonical development panel generator (rd#44 Amendment 4).

Generates the frozen V2 canonical selector panel from the committed T2
development split alone: fresh CPU environments, policy-free settled initial
states, and a fixed goal enumeration. No checkpoint, arm, or training history
enters the construction, so the panel is arm-independent by construction.

Definition (registered in dossier/V2_canonical_panel_definition.md):
  - source: committed T2 split payload, split "val" (50 development goals,
    payload order), read through the packaged split file;
  - N = 300 states = 12 batches x 25 envs; batch k (1..12) covers goal slice
    ((k-1) mod 2)*25 .. +25 of the 50 val goals in payload order, round
    r = 1 + (k-1)//2, episode_index = k (distinct per batch so every goal
    draws 6 distinct initial curves across its 6 rounds);
  - batch size 25 stays below the CPU rod solver's segfault threshold
    (SIGSEGV reproduced at n_envs >= 40 on this closure; clean at <= 32);
  - initial curves: build_batch_init_vertices via BatchedEpisodeRunner
    .begin_episodes(seed=500, episode_index=k) on a FRESH env per batch
    (env reset seed = 10_000 + k, init_shape="straight" pre-reset, matching
    the counterfactual worker's selector-independent convention);
  - X row = settled initial centerline (32,3); G row = goal_curve(goal, L);
  - panel order = batch-major env order; canonical SHA-256 via
    dgcc.rl.panel_artifacts.canonical_panel_sha256 (seed=500, transition=0,
    eval_ordinal=0, schema=1).

CPU-only: Genesis is initialized with backend=gs.cpu before any env import
touches it, and the CUDA device mask plus a torch assert keep the whole run
off the GPU (commission COMMISSION_envfix_20260801 boundary §4).
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PANEL_SEED = 500
PANEL_EPISODES_PER_GOAL = 6
PANEL_BATCH_ENVS = 25
PANEL_SPLIT = "val"
PANEL_ENV_RESET_SEED_BASE = 10_000
PANEL_SCHEMA_TRANSITION = 0
PANEL_SCHEMA_EVAL_ORDINAL = 0
FORBIDDEN_PATH_TOKENS = ("heldout", "held-out", "patch_eval", "patching_probe")


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


def init_cpu_genesis() -> None:
    """Initialize Genesis on the CPU backend before dlolab can claim the GPU."""
    import torch

    if torch.cuda.is_available():
        raise RuntimeError("CPU-only panel generation unexpectedly sees a CUDA device")
    import genesis as gs

    if not getattr(gs, "_initialized", False):
        gs.init(seed=0, precision="32", logging_level="warning", backend=gs.cpu)
    if getattr(gs, "backend", None) != gs.cpu:
        raise RuntimeError(f"Genesis backend is not CPU: {getattr(gs, 'backend', 'unknown')!r}")


def generate(args: argparse.Namespace) -> dict[str, Any]:
    init_cpu_genesis()

    from dgcc.envs.dlolab import DLOLabEnv
    from dgcc.goals.dual_goal import goal_curve
    from dgcc.rl.panel_artifacts import persist_panel
    from dgcc.tasks.domain import p1_rope_params
    from dgcc.tasks.episode import BatchedEpisodeRunner, EpisodeConfig
    from dgcc.tasks.t2 import default_split_path, load_t2_payload_bytes, load_t2_split_payload

    split_path = default_split_path().resolve()
    split_bytes = split_path.read_bytes()
    payload = load_t2_payload_bytes(split_bytes)
    val_pairs = load_t2_split_payload(PANEL_SPLIT, payload)
    if args.smoke:
        val_pairs = val_pairs[: args.smoke_goals]
    labels = [str(spec["goal_id"]) for spec, _ in val_pairs]
    goals = [goal for _, goal in val_pairs]

    params = p1_rope_params()
    episodes_per_goal = 1 if args.smoke else PANEL_EPISODES_PER_GOAL
    # The CPU rod solver segfaults at batch sizes >= 40 on this closure
    # (SIGSEGV reproduced at n_envs 40/45/50; clean at <= 32 — same upstream
    # instability class as design doc open item 11). Keep well below it.
    batch_envs = min(PANEL_BATCH_ENVS, len(goals))
    goal_slices = [
        slice(start, min(start + batch_envs, len(goals)))
        for start in range(0, len(goals), batch_envs)
    ]

    X_rows: list[np.ndarray] = []
    G_rows: list[np.ndarray] = []
    batches: list[dict[str, Any]] = []
    batch = 0
    for round_index in range(1, episodes_per_goal + 1):
        for goal_slice in goal_slices:
            batch += 1
            batch_goals = goals[goal_slice]
            env = DLOLabEnv(
                n_envs=len(batch_goals),
                dt=1.0e-3,
                substeps=5,
                rod_damping=10.0,
                rod_angular_damping=5.0,
                initial_settle_steps=0,
                reset_settle_max_steps=10_000,
                move_step_size=0.03,
                move_hold_steps=0,
                grasp_realism=True,
            )
            env.reset(
                params,
                init_shape="straight",
                seed=PANEL_ENV_RESET_SEED_BASE + batch,
            )
            runner = BatchedEpisodeRunner(env, params, EpisodeConfig())
            begin = runner.begin_episodes(
                seed=PANEL_SEED, episode_index=batch, goals=list(batch_goals)
            )
            X = np.asarray(env.get_centerline_batch(), dtype=float)
            if not np.isfinite(X).all():
                raise RuntimeError(f"non-finite settled initial state in batch {batch}")
            converged = np.asarray(begin["reset_settle_converged"], dtype=bool)
            if not converged.all():
                raise RuntimeError(
                    f"reset settle did not converge in batch {batch}: "
                    f"{np.flatnonzero(~converged).tolist()}"
                )
            G = np.asarray(
                [goal_curve(goal, params.length_m) for goal in batch_goals], dtype=float
            )
            X_rows.append(X)
            G_rows.append(G)
            batches.append(
                {
                    "batch": batch,
                    "round": round_index,
                    "episode_index": batch,
                    "goal_ids": labels[goal_slice],
                    "env_reset_seed": PANEL_ENV_RESET_SEED_BASE + batch,
                    "init_shapes": list(begin["init_shapes"]),
                    "curve_seeds": [int(s) for s in np.asarray(begin["curve_seeds"]).ravel()],
                    "reset_settle_steps": [int(s) for s in begin["reset_settle_steps"]],
                    "reset_reseeded_envs": list(begin["reset_reseeded_envs"]),
                    "d_initial_min": float(np.min(begin["d_initial"])),
                    "d_initial_median": float(np.median(begin["d_initial"])),
                    "d_initial_max": float(np.max(begin["d_initial"])),
                }
            )
            del runner
            del env

    X_all = np.concatenate(X_rows, axis=0)
    G_all = np.concatenate(G_rows, axis=0)
    order = np.arange(len(X_all), dtype=np.int64)
    artifact = persist_panel(
        args.output,
        X=X_all,
        G=G_all,
        order=order,
        seed=PANEL_SEED,
        transition=PANEL_SCHEMA_TRANSITION,
        eval_ordinal=PANEL_SCHEMA_EVAL_ORDINAL,
    )

    summary = {
        "schema_version": 1,
        "device": "cpu",
        "data_scope": "development",
        "smoke": bool(args.smoke),
        "panel_states": int(len(X_all)),
        "episodes_per_goal": episodes_per_goal,
        "goals": labels,
        "panel_seed": PANEL_SEED,
        "split": PANEL_SPLIT,
        "split_path": str(split_path),
        "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "panel_path": str(args.output),
        "panel_canonical_sha256": artifact.canonical_sha256,
        "panel_artifact_sha256": artifact.artifact_sha256,
        "panel_manifest_sha256": sha256_file(
            args.output.with_suffix(args.output.suffix + ".json")
        ),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "batches": batches,
    }
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=permitted_path, required=True)
    parser.add_argument("--summary", type=permitted_path, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-goals", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    summary = generate(parse_args())
    print(json.dumps({k: v for k, v in summary.items() if k != "batches"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
