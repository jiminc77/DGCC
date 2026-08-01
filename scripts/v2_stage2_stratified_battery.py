#!/usr/bin/env python3
"""Stage 2 stratified remeasurement battery (PREPARED; run after owner pins).

Implements the Stage 2 adjudication's battery specification
(V2_stage2_gate_adjudication.md §5.3, SHA dbbd4145...):

  - 5 T2 goal families x 4 init shapes x 3 seeds x 10 primitives = 600
    primitives, n_envs=1, fresh env per episode (unchanged);
  - action distribution: STRATIFIED random — `lift` is balanced by
    construction (5 low / 5 high per episode in a seeded deterministic
    order, not drawn), `δ_xy` uniform on the ‖δ_xy‖ <= 0.15 disk (z = 0),
    `p` uniform over nodes;
  - fully deterministic under a fixed seed; the complete action sequence's
    SHA-256 is registered in the artifact (Amendment 5 pin target);
  - AT verdicts are reported PER LIFT STRATUM and the gate requires both
    strata to pass;
  - additionally computes the adjudication's criteria, APPROVED by owner
    pins O2/O3 (orchestrator reply 2026-08-02); quality thresholds AT-1/2/4
    remain provisional Rev 2 values reported per stratum as recalibration
    input:
      AT-1H  peak v <= 2.0 m/s AND strain <= 0.02 AND KE/PE <= 1.0, 0 events
      AT-6'  median(settle:move) <= 2.0 and p95 <= 5.0 (distribution basis)
      AT-7a  budget exhaustion with dirty terminal state (arclen_dev > 1e-3):
             0 events;  AT-7b creep (clean terminal): <= 1% reported
  - reuses the probe instrumentation of v2_env_correction_acceptance
    (n_waypoint_steps, per-leg peaks, AT-9b penetration, transient arclen).

Execution guard: this battery is the Stage 2 REMEASUREMENT and must not run
before the owner pins O1-O4 land (P9 repair changes the measured system).
`--authorize <token>` with the adjudication SHA prefix is required to run
the full battery; `--smoke` runs one episode for script validation only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v2_env_correction_acceptance import (  # noqa: E402
    AT_THRESHOLDS,
    BATTERY_PRIMITIVES,
    BATTERY_SLICE_EPISODES,
    battery_episode_plan,
    build_probe_env,
    family_goals,
    init_genesis,
    judge,
    permitted_path,
    sha256_file,
)

ADJUDICATION_SHA_PREFIX = "dbbd4145"
MAX_DELTA_NORM = 0.15
AT1H = {"v": 2.0, "strain": 0.02, "ke_over_pe": 1.0}
AT6_REVISED = {"median": 2.0, "p95": 5.0}
AT7B_RATE = 0.01


def stratified_actions(episode: int, seed: int) -> list[dict[str, Any]]:
    """Deterministic per-episode action schedule: lift balanced 5/5."""
    rng = np.random.default_rng([77_000, episode, seed])
    lifts = np.array(["low"] * 5 + ["high"] * 5, dtype=object)
    rng.shuffle(lifts)
    actions = []
    for k in range(BATTERY_PRIMITIVES):
        radius = MAX_DELTA_NORM * float(np.sqrt(rng.uniform()))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        actions.append(
            {
                "p": int(rng.integers(0, 32)),
                "delta": [radius * np.cos(angle), radius * np.sin(angle), 0.0],
                "lift": str(lifts[k]),
            }
        )
    return actions


def action_sequence_sha256() -> str:
    digest = hashlib.sha256()
    for entry in battery_episode_plan():
        for action in stratified_actions(entry["episode"], entry["seed"]):
            digest.update(
                json.dumps(action, sort_keys=True, separators=(",", ":")).encode()
            )
    return digest.hexdigest()


def run_slice(first: int, last: int, backend_info: dict[str, Any]) -> dict[str, Any]:
    from dgcc.envs.dlolab import LIFT_HEIGHTS
    from dgcc.tasks.domain import p1_rope_params
    from dgcc.tasks.episode import BatchedEpisodeRunner, EpisodeConfig

    params = p1_rope_params()
    goals = family_goals()
    rope_mass = 32 * 1.0e-3
    gravity = 9.81

    primitives: list[dict[str, Any]] = []
    for entry in battery_episode_plan():
        if not (first <= entry["episode"] <= last):
            continue
        _family, _goal_id, goal = goals[entry["family_index"]]
        episode_ordinal = entry["episode"]
        env = build_probe_env(1)
        env.reset(params, init_shape=entry["init_shape"], seed=1_000 + episode_ordinal)
        runner = BatchedEpisodeRunner(env, params, EpisodeConfig())
        runner.begin_episodes(
            seed=entry["seed"],
            episode_index=episode_ordinal,
            init_shapes=[entry["init_shape"]],
            goals=[goal],
        )
        for k, action in enumerate(stratified_actions(episode_ordinal, entry["seed"])):
            env.probe_begin_primitive()
            subfloor = False
            try:
                out = runner.step(
                    np.asarray([action["p"]], dtype=int),
                    np.asarray([action["delta"]], dtype=float),
                    [action["lift"]],
                    rng=np.random.default_rng([88_000, episode_ordinal, k]),
                )
            except RuntimeError as exc:
                if "subfloor" in str(exc):
                    subfloor = True
                raise
            finally:
                probe = env.probe_end_primitive()
            info = out["info"]
            lift_height = float(LIFT_HEIGHTS[action["lift"]])
            grav_pe = rope_mass * gravity * lift_height
            move_steps = probe["move_steps"]
            settle_steps = int(out["settle_steps"][0])
            primitives.append(
                {
                    **{key: entry[key] for key in ("episode", "family", "goal_id", "init_shape", "seed")},
                    "primitive": k,
                    "backend": backend_info["backend"],
                    "lift": action["lift"],
                    "target_z": float(env.last_move_target[0, 2]),
                    "subfloor_target_commanded": subfloor,
                    "n_waypoint_steps": probe["n_waypoint_steps"],
                    "v_peak_premove": probe["v_peak_premove"],
                    "v_peak_lift": probe.get("v_peak_lift", 0.0),
                    "v_peak_translate": probe.get("v_peak_translate", 0.0),
                    "v_peak_lower": probe.get("v_peak_lower", 0.0),
                    "min_node_z": probe["min_node_z"],
                    "ground_penetration_steps": probe["ground_penetration_steps"],
                    "v_peak_move": probe["v_peak_move"],
                    "v_peak_settle": probe["v_peak_settle"],
                    "v_peak_total": probe["v_peak_total"],
                    "v_at_detach": probe["v_at_detach"],
                    "ke_peak": probe["ke_peak"],
                    "grav_pe": grav_pe,
                    "ke_over_pe": probe["ke_peak"] / grav_pe,
                    "strain_peak": probe["strain_peak"],
                    "arclen_peak": probe["arclen_peak"],
                    "arclen_after_settle": probe["arclen_final"],
                    "arclen_dev_after_settle": abs(
                        probe["arclen_final"] / float(params.length_m) - 1.0
                    ),
                    "settle_steps": settle_steps,
                    "settle_converged": bool(info["settle_converged"][0]),
                    "move_steps": move_steps,
                    "hold_steps_used": int(info["hold_steps_used"]),
                    "hold_converged": bool(info["hold_converged"]),
                    "settle_to_move_ratio": settle_steps / move_steps if move_steps else float("inf"),
                    "total_sim_steps": probe["total_sim_steps"],
                }
            )
        primitives[-1]["episode_covenant"] = {
            "nan_incidents": int(runner.nan_incidents),
            "magnitude_incidents": int(runner.magnitude_incidents),
            "arclength_incidents": int(getattr(runner, "arclength_incidents", 0)),
        }
        del runner
        del env
    return {"primitives": primitives, "backend_info": backend_info}


def adjudication_criteria(primitives: list[dict[str, Any]]) -> dict[str, Any]:
    """AT-1H / AT-6' / AT-7a/b verdicts (owner pins O2/O3 approved 2026-08-02)."""
    v = np.asarray([p["v_peak_total"] for p in primitives], dtype=float)
    strain = np.asarray([p["strain_peak"] for p in primitives], dtype=float)
    kepe = np.asarray([p["ke_over_pe"] for p in primitives], dtype=float)
    ratio = np.asarray([p["settle_to_move_ratio"] for p in primitives], dtype=float)
    settle = np.asarray([p["settle_steps"] for p in primitives], dtype=int)
    arclen_dev = np.asarray([p["arclen_dev_after_settle"] for p in primitives], dtype=float)
    at1h_violations = int(
        ((v > AT1H["v"]) | (strain > AT1H["strain"]) | (kepe > AT1H["ke_over_pe"])).sum()
    )
    budget = 10_000
    exhausted = settle >= budget
    at7a = int((exhausted & (arclen_dev > 1.0e-3)).sum())
    at7b = int((exhausted & (arclen_dev <= 1.0e-3)).sum())
    return {
        "AT-1H (O2 approved)": {
            "criterion": "v <= 2.0 AND strain <= 0.02 AND KE/PE <= 1.0; 0 events",
            "violations": at1h_violations,
            "pass": at1h_violations == 0,
        },
        "AT-6-revised (O3 approved)": {
            "criterion": "median(settle:move) <= 2.0 and p95 <= 5.0",
            "median": float(np.median(ratio)),
            "p95": float(np.percentile(ratio, 95)),
            "pass": bool(
                np.median(ratio) <= AT6_REVISED["median"]
                and np.percentile(ratio, 95) <= AT6_REVISED["p95"]
            ),
        },
        "AT-7a-divergence (O3 approved)": {
            "criterion": "budget exhaustion with arclen_dev > 1e-3; 0 events",
            "violations": at7a,
            "pass": at7a == 0,
        },
        "AT-7b-creep (O3 approved)": {
            "criterion": "budget exhaustion with clean terminal; <= 1% reported",
            "events": at7b,
            "rate": round(at7b / len(primitives), 4),
            "pass": at7b / len(primitives) <= AT7B_RATE,
        },
    }


def spawn_slice(out_path: Path, backend: str, first: int, last: int) -> dict[str, Any]:
    env = dict(os.environ)
    if backend == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--section", "slice", "--backend", backend,
        "--slice-first", str(first), "--slice-last", str(last),
        "--out", str(out_path), "--authorize", ADJUDICATION_SHA_PREFIX,
    ]
    completed = subprocess.run(
        command, env=env, timeout=7200,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        tail = completed.stderr.decode(errors="replace")[-2000:]
        raise RuntimeError(f"slice [{first},{last}] exited {completed.returncode}: {tail}")
    return json.loads(out_path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=permitted_path, required=True)
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--section", choices=["driver", "slice"], default="driver")
    parser.add_argument("--slice-first", type=int, default=1)
    parser.add_argument("--slice-last", type=int, default=60)
    parser.add_argument("--smoke", action="store_true", help="one-episode script validation only")
    parser.add_argument(
        "--authorize", default=None,
        help="required for the full battery: adjudication SHA-256 prefix "
             "(the Stage 2 remeasurement must wait for owner pins O1-O4)",
    )
    args = parser.parse_args()

    started = time.time()
    if args.section == "slice":
        if args.authorize != ADJUDICATION_SHA_PREFIX and not args.smoke:
            raise SystemExit("slice execution requires --authorize")
        backend_info = init_genesis(args.backend)
        result = run_slice(args.slice_first, args.slice_last, backend_info)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return 0

    if args.smoke:
        backend_info = init_genesis(args.backend)
        partial = run_slice(1, 1, backend_info)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(partial, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"smoke": True, "primitives": len(partial["primitives"])}))
        return 0

    if args.authorize != ADJUDICATION_SHA_PREFIX:
        raise SystemExit(
            "The stratified Stage 2 remeasurement is gated on the owner pins "
            "(O1-O4). Re-run with --authorize " + ADJUDICATION_SHA_PREFIX +
            " once the orchestrator confirms the pins."
        )

    plan_size = len(battery_episode_plan())
    slices = [
        (first, min(first + BATTERY_SLICE_EPISODES - 1, plan_size))
        for first in range(1, plan_size + 1, BATTERY_SLICE_EPISODES)
    ]
    primitives: list[dict[str, Any]] = []
    backends: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="v2_stratified_") as tmp:
        for first, last in slices:
            partial = spawn_slice(Path(tmp) / f"slice_{first:03d}.json", args.backend, first, last)
            primitives.extend(partial["primitives"])
            backends.add(partial["backend_info"]["backend"])

    expected = plan_size * BATTERY_PRIMITIVES
    if len(primitives) != expected:
        raise RuntimeError(f"merged {len(primitives)} of {expected} primitives")

    strata = {
        stratum: [p for p in primitives if p["lift"] == stratum]
        for stratum in ("low", "high")
    }
    per_stratum = {
        stratum: {
            "rev2": judge(rows),
            "adjudication": adjudication_criteria(rows),
            "n": len(rows),
        }
        for stratum, rows in strata.items()
    }
    overall_adjudication = adjudication_criteria(primitives)

    result = {
        "schema_version": 1,
        "battery": "stage2-stratified",
        "backend": sorted(backends),
        "adjudication_sha_prefix": ADJUDICATION_SHA_PREFIX,
        "action_sequence_sha256": action_sequence_sha256(),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "rev2_thresholds": AT_THRESHOLDS,
        "per_stratum": per_stratum,
        "overall_adjudication": overall_adjudication,
        "primitives": primitives,
        "elapsed_s": round(time.time() - started, 1),
    }
    result["pass_rev2_both_strata"] = all(
        test["pass"]
        for stratum in per_stratum.values()
        for test in stratum["rev2"].values()
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "per_stratum_pass": {
                    stratum: all(test["pass"] for test in data["rev2"].values())
                    for stratum, data in per_stratum.items()
                },
                "overall_adjudication": overall_adjudication,
                "pass_rev2_both_strata": result["pass_rev2_both_strata"],
            },
            indent=2, default=str,
        )
    )
    return 0 if result["pass_rev2_both_strata"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
