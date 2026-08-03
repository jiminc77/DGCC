#!/usr/bin/env python3
"""Rev 7 A.2/A.3: measure the release-time residual-strain distribution.

The ep44 p0 forensic established that the peak of a primitive is set by how
much elastic strain the rope still carries when the gripper lets go, and that
the shipped hold-until-quiescent exits on VELOCITY alone
(``HOLD_QUIESCENT_VEL``), so a rope that is still-but-loaded is released
immediately.  The proposed repair adds a strain term to the hold exit.

The trial run used threshold 5e-4 with a 4000-step budget and EXHAUSTED the
budget (released at 8.65e-4), so neither the threshold nor the budget is
calibrated.  This script measures the distribution they must be derived from.

For every primitive it records, at the moment the shipped hold exits:
  * the residual max edge strain (the quantity the new term would gate on), and
  * how many additional hold steps are needed to reach each candidate
    threshold, by continuing to hold and watching the strain decay.

The extra hold is APPLIED (not rewound), so the measured distribution is the
distribution of the repaired system rather than of a system that never holds.
Judgement thresholds (AT-1H / AT-4 / ...) are not touched anywhere here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Candidate thresholds, spanning from the resting static strain (~1e-4) up to
# the existing lowering fail-safe scale (LOWER_STRAIN_ABORT = 5e-3).
CANDIDATES = (2.0e-3, 1.5e-3, 1.0e-3, 7.5e-4, 5.0e-4, 2.5e-4, 1.0e-4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-extra-hold", type=int, default=3000)
    parser.add_argument("--backend", default="gpu", choices=["cpu", "gpu"])
    args = parser.parse_args()

    from v2_env_correction_acceptance import (
        battery_episode_plan,
        build_probe_env,
        family_goals,
        init_genesis,
    )
    from v2_stage2_stratified_battery import stratified_actions

    from dgcc.tasks.domain import p1_rope_params
    from dgcc.tasks.episode import BatchedEpisodeRunner, EpisodeConfig

    backend_info = init_genesis(args.backend)
    params = p1_rope_params()
    goals = family_goals()

    rows: list[dict[str, object]] = []
    started = time.time()

    for entry in battery_episode_plan():
        if entry["episode"] > args.episodes:
            continue
        _family, _goal_id, goal = goals[entry["family_index"]]
        env = build_probe_env(1)

        pending: dict[str, object] = {}
        base_execute = env._execute_move

        def execute_with_measurement(lifted, target, vel_threshold):
            final = base_execute(lifted, target, vel_threshold)
            strain0 = float(env._max_edge_strain_batch().max())
            speed0 = float(env.max_node_speed_batch().max())
            reached: dict[str, int | None] = {f"{c:g}": None for c in CANDIDATES}
            for c in CANDIDATES:
                if strain0 < c:
                    reached[f"{c:g}"] = 0
            curve: list[float] = [strain0]
            steps = 0
            while steps < int(args.max_extra_hold):
                if all(v is not None for v in reached.values()):
                    break
                env._set_gripper_positions(final)
                env._step_scene()
                steps += 1
                strain = float(env._max_edge_strain_batch().max())
                if steps % 25 == 0 or steps < 25:
                    curve.append(strain)
                for c in CANDIDATES:
                    if reached[f"{c:g}"] is None and strain < c:
                        reached[f"{c:g}"] = steps
            pending.update(
                strain_at_natural_release=strain0,
                speed_at_natural_release=speed0,
                steps_to_threshold=reached,
                extra_hold_steps_used=steps,
                strain_after_extra_hold=float(env._max_edge_strain_batch().max()),
                strain_curve=curve,
            )
            return final

        env._execute_move = execute_with_measurement
        env.reset(params, init_shape=entry["init_shape"], seed=1_000 + entry["episode"])
        runner = BatchedEpisodeRunner(env, params, EpisodeConfig())
        runner.begin_episodes(
            seed=entry["seed"],
            episode_index=entry["episode"],
            init_shapes=[entry["init_shape"]],
            goals=[goal],
        )
        for k, action in enumerate(stratified_actions(entry["episode"], entry["seed"])):
            pending.clear()
            env.probe_begin_primitive()
            try:
                out = runner.step(
                    np.asarray([action["p"]], dtype=int),
                    np.asarray([action["delta"]], dtype=float),
                    [action["lift"]],
                    rng=np.random.default_rng([88_000, entry["episode"], k]),
                )
            finally:
                probe = env.probe_end_primitive()
            if not pending:
                # grasp-realism failure: no move, hence no release to measure.
                continue
            rows.append(
                {
                    "episode": entry["episode"],
                    "primitive": k,
                    "lift": action["lift"],
                    "family": entry["family"],
                    "init_shape": entry["init_shape"],
                    **pending,
                    "hold_steps_used": int(out["info"]["hold_steps_used"]),
                    "v_peak_total": probe["v_peak_total"],
                    "v_peak_settle": probe["v_peak_settle"],
                    "strain_peak": probe["strain_peak"],
                    "v_at_detach": probe["v_at_detach"],
                    "settle_steps": int(out["settle_steps"][0]),
                    "total_sim_steps": probe["total_sim_steps"],
                    "approach_steps": int(probe["approach_steps"]),
                }
            )
        del runner, env

    strain0 = np.asarray([r["strain_at_natural_release"] for r in rows], dtype=float)
    summary = {
        "n_primitives": len(rows),
        "strain_at_natural_release": {
            "min": float(strain0.min()), "p50": float(np.percentile(strain0, 50)),
            "p90": float(np.percentile(strain0, 90)), "p95": float(np.percentile(strain0, 95)),
            "p99": float(np.percentile(strain0, 99)), "max": float(strain0.max()),
        },
        "per_candidate": {},
    }
    for c in CANDIDATES:
        key = f"{c:g}"
        steps = [r["steps_to_threshold"][key] for r in rows]
        reached = [s for s in steps if s is not None]
        arr = np.asarray(reached, dtype=float) if reached else np.zeros(0)
        summary["per_candidate"][key] = {
            "reached_fraction": len(reached) / len(rows) if rows else 0.0,
            "unreached": len(rows) - len(reached),
            "steps_mean": float(arr.mean()) if arr.size else None,
            "steps_p50": float(np.percentile(arr, 50)) if arr.size else None,
            "steps_p95": float(np.percentile(arr, 95)) if arr.size else None,
            "steps_max": float(arr.max()) if arr.size else None,
        }
    payload = {
        "probe": "rev7-release-strain-distribution",
        "backend": backend_info,
        "n_segments": int(params.n_segments),
        "rope_mass_total_kg": float(params.rope_mass_total_kg),
        "max_extra_hold": int(args.max_extra_hold),
        "candidates": list(CANDIDATES),
        "elapsed_s": round(time.time() - started, 1),
        "summary": summary,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
