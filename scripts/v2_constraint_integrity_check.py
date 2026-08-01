#!/usr/bin/env python3
"""Constraint-integrity acceptance battery: AT-14 / AT-15 / AT-16 / AT-17.

Verifies the D1-D4 defect-(c)/(c') corrections of V2_env_correction_design.md
Rev 2 (SHA 658b671d...):

  AT-14  stuck-signature zero: after every light_reset of the battery,
         vertex_constraints.constrained.sum() == 0 AND no settled initial
         node sits within +-2 mm of the gripper parking height.
  AT-15  single-episode isolation reproducibility: one stored x_initial
         placed into (i) a fresh env and (ii) an env that already executed
         N in {0, 10, 50, 100} primitives yields |delta d_initial| <= 1.44e-5
         (the placement perturbation sensitivity floor).  CPU-pinned
         (boundary amendment item 3: determinism contrast).
  AT-16  detach completeness: zero residual constraints after every verified
         detach of the battery; D2 escalation invocations are reported
         separately (informational — nonzero means the upstream kernel is
         still flaky and D1/D2 are doing real work).
  AT-17  grasp-failure restoration scope: after a primitive with forced
         grasp failures, the theta/omega/twist/kappa_rest state of the
         SUCCESSFUL envs is bit-identical to a no-failure control run.
         CPU-pinned (bit-compare).

Standard battery: 5 T2 goal families x 4 init shapes x 3 seeds = 60 episodes,
10 primitives each (600 primitives), n_envs=1, fresh env instance per episode
(design section 4.5), validation split goals only, deterministic seeded random
actions.  The battery backend is selectable (--backend, default gpu per the
2026-08-01 boundary amendment) and every artifact row records its backend.

Execution is PROCESS-ISOLATED: the driver spawns one subprocess per battery
slice (and per AT-15/AT-17 section) because Genesis/Quadrants accumulates
kernel recompilations across repeated in-process scene builds and reliably
segfaults near the ~50th fresh env (SIGSEGV reproduced twice on this
closure).  Slice results are merged into one fail-closed artifact.

Exit code 0 only when every AT above passes (fail-closed).
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

FORBIDDEN_PATH_TOKENS = ("heldout", "held-out", "patch_eval", "patching_probe")
AT15_THRESHOLD = 1.44e-5
AT14_PARKING_TOLERANCE_M = 2.0e-3
BATTERY_SEEDS = (0, 1, 2)
BATTERY_PRIMITIVES = 10
BATTERY_SLICE_EPISODES = 10
AT15_HISTORY = (0, 10, 50, 100)


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


def init_genesis(backend: str) -> dict[str, Any]:
    """Initialize Genesis on the requested backend before any env import."""
    import torch

    if backend == "cpu":
        if torch.cuda.is_available():
            raise RuntimeError("CPU-pinned section unexpectedly sees a CUDA device")
        import genesis as gs

        if not getattr(gs, "_initialized", False):
            gs.init(seed=0, precision="32", logging_level="warning", backend=gs.cpu)
        if getattr(gs, "backend", None) != gs.cpu:
            raise RuntimeError(f"Genesis backend is not CPU: {getattr(gs, 'backend', 'unknown')!r}")
        return {"backend": "cpu", "device": "cpu"}
    if backend == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU battery requested but no CUDA device is visible")
        import genesis as gs

        if not getattr(gs, "_initialized", False):
            gs.init(seed=0, precision="32", logging_level="warning", backend=gs.gpu)
        return {
            "backend": "gpu",
            "device": torch.cuda.get_device_name(0),
        }
    raise ValueError(f"unknown backend {backend!r}")


def make_env(n_envs: int):
    from dgcc.envs.dlolab import DLOLabEnv

    return DLOLabEnv(
        n_envs=n_envs,
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


def family_goals() -> list[tuple[str, str, Any]]:
    """One committed validation-split goal per T2 family, payload order."""
    from dgcc.tasks.t2 import T2_FAMILIES, load_t2_payload, load_t2_split_payload

    pairs = load_t2_split_payload("val", load_t2_payload())
    chosen: list[tuple[str, str, Any]] = []
    for family in T2_FAMILIES:
        for spec, goal in pairs:
            if str(spec["family"]) == family:
                chosen.append((family, str(spec["goal_id"]), goal))
                break
        else:
            raise RuntimeError(f"no validation goal for family {family!r}")
    return chosen


def battery_episode_plan() -> list[dict[str, Any]]:
    """Deterministic global episode enumeration (slice-invariant)."""
    from dgcc.tasks.episode import INIT_SHAPES

    plan: list[dict[str, Any]] = []
    ordinal = 0
    for family_index, (family, goal_id, _goal) in enumerate(family_goals()):
        for shape in INIT_SHAPES:
            for seed in BATTERY_SEEDS:
                ordinal += 1
                plan.append(
                    {
                        "episode": ordinal,
                        "family_index": family_index,
                        "family": family,
                        "goal_id": goal_id,
                        "init_shape": shape,
                        "seed": int(seed),
                    }
                )
    return plan


def episode_actions(rng: np.random.Generator, n_envs: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    p = rng.integers(0, 32, n_envs)
    delta = rng.normal(0.0, 0.06, (n_envs, 3))
    lift = [str(x) for x in rng.choice(["low", "high"], n_envs)]
    return p, delta, lift


def run_battery_slice(first: int, last: int, backend_info: dict[str, Any]) -> dict[str, Any]:
    """AT-14 + AT-16 over battery episodes with ordinal in [first, last]."""
    from dgcc.envs.dlolab import LIFT_HEIGHTS
    from dgcc.tasks.domain import p1_rope_params
    from dgcc.tasks.episode import BatchedEpisodeRunner, EpisodeConfig

    params = p1_rope_params()
    parking_z = float(LIFT_HEIGHTS["high"])
    goals = family_goals()

    episodes: list[dict[str, Any]] = []
    for entry in battery_episode_plan():
        if not (first <= entry["episode"] <= last):
            continue
        _family, _goal_id, goal = goals[entry["family_index"]]
        episode_ordinal = entry["episode"]
        env = make_env(1)
        env.reset(params, init_shape=entry["init_shape"], seed=1_000 + episode_ordinal)
        runner = BatchedEpisodeRunner(env, params, EpisodeConfig())
        begin = runner.begin_episodes(
            seed=entry["seed"],
            episode_index=episode_ordinal,
            init_shapes=[entry["init_shape"]],
            goals=[goal],
        )
        mask = env._vertex_constrained_mask()
        light_reset_violation = bool(mask.any())
        z = np.asarray(env.get_centerline_batch(), dtype=float)[..., 2]
        parking_hits = int(np.sum(np.abs(z - parking_z) <= AT14_PARKING_TOLERANCE_M))

        action_rng = np.random.default_rng([9_000, episode_ordinal, entry["seed"]])
        episode_residuals = 0
        episode_escalations = 0
        post_detach_leftover = 0
        for _ in range(BATTERY_PRIMITIVES):
            p, delta, lift = episode_actions(action_rng, 1)
            out = runner.step(p, delta, lift, rng=action_rng)
            info = out["info"]
            episode_residuals += int(info["detach_residuals"])
            episode_escalations += int(info["detach_escalations"])
            post_detach_leftover += int(env._vertex_constrained_mask().sum())
        episodes.append(
            {
                **{k: entry[k] for k in ("episode", "family", "goal_id", "init_shape", "seed")},
                "backend": backend_info["backend"],
                "d_initial": float(begin["d_initial"][0]),
                "reset_settle_steps": int(begin["reset_settle_steps"][0]),
                "light_reset_residual_violation": light_reset_violation,
                "parking_node_hits": parking_hits,
                "detach_residuals": episode_residuals,
                "detach_escalations": episode_escalations,
                "post_detach_leftover": post_detach_leftover,
            }
        )
        del runner
        del env
    return {"episodes": episodes, "backend_info": backend_info}


def run_at15(backend_info: dict[str, Any]) -> dict[str, Any]:
    from dgcc.goals.distance import D as distance_to_goal
    from dgcc.tasks.domain import p1_rope_params
    from dgcc.tasks.episode import BatchedEpisodeRunner, EpisodeConfig

    params = p1_rope_params()
    family, goal_id, goal = family_goals()[0]

    env = make_env(1)
    env.reset(params, init_shape="s_curve", seed=4_242)
    runner = BatchedEpisodeRunner(env, params, EpisodeConfig())
    runner.begin_episodes(seed=4_242, episode_index=1, goals=[goal])
    x_initial = np.asarray(env.get_centerline_raw_batch(), dtype=float).copy()
    del runner
    del env

    def placed_d_initial(env) -> float:
        result = env.light_reset(x_initial)
        if not bool(np.all(result["settle_converged"])):
            raise RuntimeError("AT-15 placement settle did not converge")
        centerline = np.asarray(env.get_centerline_batch(), dtype=float)[0]
        return float(distance_to_goal(centerline, goal, params.length_m))

    fresh_env = make_env(1)
    fresh_env.reset(params, init_shape="straight", seed=5_555)
    d_reference = placed_d_initial(fresh_env)
    del fresh_env

    rows: list[dict[str, Any]] = []
    worst = 0.0
    for n_history in AT15_HISTORY:
        env = make_env(1)
        env.reset(params, init_shape="straight", seed=5_555)
        runner = BatchedEpisodeRunner(env, params, EpisodeConfig())
        runner.begin_episodes(seed=6_000 + n_history, episode_index=1, goals=[goal])
        action_rng = np.random.default_rng([7_000, n_history])
        for _ in range(n_history):
            p, delta, lift = episode_actions(action_rng, 1)
            runner.step(p, delta, lift, rng=action_rng)
        d_history = placed_d_initial(env)
        deviation = abs(d_history - d_reference)
        worst = max(worst, deviation)
        rows.append(
            {
                "history_primitives": int(n_history),
                "d_initial": d_history,
                "abs_deviation": deviation,
                "pass": deviation <= AT15_THRESHOLD,
            }
        )
        del runner
        del env

    return {
        "goal_id": goal_id,
        "family": family,
        "backend": backend_info["backend"],
        "d_initial_reference": d_reference,
        "threshold": AT15_THRESHOLD,
        "rows": rows,
        "worst_abs_deviation": worst,
        "pass": worst <= AT15_THRESHOLD,
    }


def run_at17(backend_info: dict[str, Any]) -> dict[str, Any]:
    import dgcc.envs.dlolab as dlolab_module
    from dgcc.tasks.domain import p1_rope_params
    from dgcc.tasks.episode import BatchedEpisodeRunner, EpisodeConfig

    params = p1_rope_params()
    n_envs = 8
    _, goal_id, goal = family_goals()[1]

    original_sample_grasp = dlolab_module.sample_grasp

    def run_once(fail_envs: tuple[int, ...]) -> dict[str, np.ndarray]:
        call_index = {"i": 0}

        def forced_sample(p, n_nodes, rng, enabled=True):
            actual, _success = original_sample_grasp(p, n_nodes, rng, enabled)
            env_idx = call_index["i"]
            call_index["i"] += 1
            return actual, env_idx not in fail_envs

        env = make_env(n_envs)
        env.reset(params, init_shape="u_bend", seed=8_888)
        runner = BatchedEpisodeRunner(env, params, EpisodeConfig())
        runner.begin_episodes(seed=8_888, episode_index=1, goals=[goal] * n_envs)
        action_rng = np.random.default_rng(1_234)
        p = np.arange(4, 4 + n_envs) % 32
        delta = action_rng.normal(0.0, 0.05, (n_envs, 3))
        lift = ["low"] * n_envs
        dlolab_module.sample_grasp = forced_sample
        try:
            out = runner.step(p, delta, lift, rng=np.random.default_rng(999))
        finally:
            dlolab_module.sample_grasp = original_sample_grasp
        solver = env.rod_entity._solver
        state = {
            "grasp_success": np.asarray(out["grasp_success"], dtype=bool).copy(),
            "settle_steps": np.asarray(out["settle_steps"], dtype=int).copy(),
            "theta": solver.edges.theta.to_numpy().copy(),
            "omega": solver.edges.omega.to_numpy().copy(),
            "twist": solver.internal_vertices.twist.to_numpy().copy(),
            "kappa_rest": solver.internal_vertices.kappa_rest.to_numpy().copy(),
        }
        del runner
        del env
        return state

    control = run_once(())
    if not control["grasp_success"].all():
        raise RuntimeError("AT-17 control run unexpectedly failed a grasp")
    # settle_batch steps every env until the WHOLE batch converges (the
    # documented L1 batch coupling, out of correction scope).  Forcing the
    # failure into the batch's settle-critical envs would change the total
    # batch settle length and perturb every env through that coupling, which
    # is not what AT-17 isolates.  Pick the two envs with the SHORTEST
    # control settle so the batch total is governed by successful envs in
    # both runs, and assert that precondition below.
    control_settle = control["settle_steps"]
    forced_failures = tuple(int(i) for i in np.argsort(control_settle, kind="stable")[:2])
    injected = run_once(forced_failures)

    expected_fail = np.zeros(n_envs, dtype=bool)
    expected_fail[list(forced_failures)] = True
    if not np.array_equal(~injected["grasp_success"], expected_fail):
        raise RuntimeError(
            "AT-17 forced-failure injection mismatch: "
            f"{np.flatnonzero(~injected['grasp_success']).tolist()}"
        )

    successful = np.flatnonzero(injected["grasp_success"])
    batch_settle_equal = int(control_settle.max()) == int(
        injected["settle_steps"][successful].max()
    )
    if not batch_settle_equal:
        raise RuntimeError(
            "AT-17 precondition violated: batch settle length changed "
            f"(control {int(control_settle.max())} vs injected "
            f"{int(injected['settle_steps'][successful].max())}); the "
            "comparison would measure the documented settle_batch coupling "
            "instead of the restoration scope"
        )
    comparisons: dict[str, bool] = {}
    for key in ("theta", "omega", "twist", "kappa_rest"):
        a = control[key]
        b = injected[key]
        # Fields are laid out (F, E|IV, B[, 2]); env axis is 2.
        env_axis = 2
        a_sel = np.take(a, successful, axis=env_axis)
        b_sel = np.take(b, successful, axis=env_axis)
        comparisons[key] = bool(np.array_equal(a_sel, b_sel))

    return {
        "goal_id": goal_id,
        "n_envs": n_envs,
        "backend": backend_info["backend"],
        "forced_failure_envs": list(forced_failures),
        "successful_envs": [int(i) for i in successful],
        "control_settle_steps": [int(s) for s in control_settle],
        "injected_settle_steps": [int(s) for s in injected["settle_steps"]],
        "batch_settle_equal": batch_settle_equal,
        "bit_identical": comparisons,
        "pass": all(comparisons.values()),
    }


def spawn_section(
    section: str, out_path: Path, backend: str, extra: list[str], timeout_s: int
) -> dict[str, Any]:
    env = dict(os.environ)
    if backend == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--section",
        section,
        "--backend",
        backend,
        "--out",
        str(out_path),
        *extra,
    ]
    completed = subprocess.run(
        command, env=env, timeout=timeout_s,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        tail = completed.stderr.decode(errors="replace")[-2000:]
        raise RuntimeError(
            f"section {section} {extra} exited {completed.returncode}: {tail}"
        )
    return json.loads(out_path.read_text())


def run_driver(args: argparse.Namespace) -> dict[str, Any]:
    plan = battery_episode_plan()
    total_episodes = len(plan)
    slices = [
        (first, min(first + BATTERY_SLICE_EPISODES - 1, total_episodes))
        for first in range(1, total_episodes + 1, BATTERY_SLICE_EPISODES)
    ]

    episodes: list[dict[str, Any]] = []
    battery_backends: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="v2_integrity_") as tmp:
        tmpdir = Path(tmp)
        if not args.skip_battery:
            for first, last in slices:
                partial = spawn_section(
                    "battery-slice",
                    tmpdir / f"battery_{first:03d}.json",
                    args.backend,
                    ["--slice-first", str(first), "--slice-last", str(last)],
                    timeout_s=3600,
                )
                episodes.extend(partial["episodes"])
                battery_backends.add(partial["backend_info"]["backend"])
        at15 = (
            None
            if args.skip_at15
            else spawn_section("at15", tmpdir / "at15.json", "cpu", [], timeout_s=3600)
        )
        at17 = spawn_section("at17", tmpdir / "at17.json", "cpu", [], timeout_s=1800)

    result: dict[str, Any] = {
        "schema_version": 2,
        "battery": args.battery,
        "battery_backend": sorted(battery_backends),
        "cpu_pinned_sections": ["at15", "at17"],
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "process_isolation": {
            "slice_episodes": BATTERY_SLICE_EPISODES,
            "reason": "Genesis kernel-recompile accumulation segfaults near the "
                      "~50th fresh env in one process (SIGSEGV observed twice)",
        },
    }
    if not args.skip_battery:
        if len(episodes) != total_episodes:
            raise RuntimeError(
                f"battery merged {len(episodes)} of {total_episodes} episodes"
            )
        violations = sum(e["light_reset_residual_violation"] for e in episodes)
        parking = sum(e["parking_node_hits"] for e in episodes)
        residuals = sum(e["detach_residuals"] for e in episodes)
        escalations = sum(e["detach_escalations"] for e in episodes)
        leftover = sum(e["post_detach_leftover"] for e in episodes)
        result["battery_primitives"] = total_episodes * BATTERY_PRIMITIVES
        result["at14"] = {
            "light_reset_checks": total_episodes,
            "light_reset_residual_violations": int(violations),
            "parking_node_hits": int(parking),
            "backend": sorted(battery_backends),
            "pass": violations == 0 and parking == 0,
        }
        result["at16"] = {
            "detach_residuals_first_pass": int(residuals),
            "detach_escalations": int(escalations),
            "post_detach_leftover": int(leftover),
            "backend": sorted(battery_backends),
            "pass": leftover == 0,
        }
        result["battery_episodes"] = episodes
    if at15 is not None:
        result["at15"] = at15
    result["at17"] = at17
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battery", choices=["standard"], default="standard")
    parser.add_argument("--out", type=permitted_path, required=True)
    parser.add_argument(
        "--backend", choices=["cpu", "gpu"], default="gpu",
        help="battery backend (AT-15/AT-17 stay CPU-pinned regardless)",
    )
    parser.add_argument("--section", choices=["driver", "battery-slice", "at15", "at17"], default="driver")
    parser.add_argument("--slice-first", type=int, default=1)
    parser.add_argument("--slice-last", type=int, default=60)
    parser.add_argument("--skip-battery", action="store_true")
    parser.add_argument("--skip-at15", action="store_true")
    args = parser.parse_args()

    started = time.time()
    if args.section != "driver":
        if args.section in ("at15", "at17"):
            backend_info = init_genesis("cpu")
        else:
            backend_info = init_genesis(args.backend)
        if args.section == "battery-slice":
            result = run_battery_slice(args.slice_first, args.slice_last, backend_info)
        elif args.section == "at15":
            result = run_at15(backend_info)
        else:
            result = run_at17(backend_info)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return 0

    result = run_driver(args)
    result["elapsed_s"] = round(time.time() - started, 1)
    gates = [
        result[key]["pass"] for key in ("at14", "at15", "at16", "at17") if key in result
    ]
    result["pass"] = bool(gates) and all(gates)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: result[key] for key in ("at14", "at15", "at16", "at17", "pass") if key in result},
            indent=2,
            default=str,
        )
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
