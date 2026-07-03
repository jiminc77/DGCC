"""P1-M0 throughput probe: transitions/s at n_envs ∈ {64, 128, 256}.

Measures the actual P1 collection path — the batched episode runner over
``DLOLabEnv`` with T2 train goals, random policy actions, and the immutable
settle budget (vel_threshold=1e-3, max_steps=10000) on every settle-bearing
call — then recommends an n_envs and projects T1/T2 run durations under S1
(one run at a time; concurrency was explicitly rejected at plan
reconciliation, so no multi-process measurements are taken here).

Outputs: ``outputs/metrics/p1_throughput.json`` and
``outputs/reports/p1_throughput.md``.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dgcc.envs.dlolab import DLOLabEnv
from dgcc.tasks.domain import (
    P1_N_SEGMENTS,
    SETTLE_MAX_STEPS,
    SETTLE_VEL_THRESHOLD,
    p1_rope_params,
)
from dgcc.tasks.episode import BatchedEpisodeRunner, random_policy_actions
from dgcc.tasks.t2 import load_t2_split
from dgcc.utils.meta import get_git_commit_hash


class Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def env_kwargs(config: dict[str, Any], n_envs: int) -> dict[str, Any]:
    sim = config.get("sim", {})
    return {
        "n_envs": int(n_envs),
        "dt": float(sim.get("dt", 1.0e-3)),
        "substeps": int(sim.get("substeps", 5)),
        "rod_damping": float(sim.get("rod_damping", 10.0)),
        "rod_angular_damping": float(sim.get("rod_angular_damping", 5.0)),
        "initial_settle_steps": int(sim.get("initial_settle_steps", 0)),
        "reset_settle_max_steps": int(sim.get("reset_settle_max_steps", SETTLE_MAX_STEPS)),
        "move_step_size": float(sim.get("move_step_size", 0.03)),
        "move_hold_steps": int(sim.get("move_hold_steps", 0)),
        "grasp_realism": bool(sim.get("grasp_realism", True)),
    }


def cuda_memory_snapshot() -> dict[str, float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        free_b, total_b = torch.cuda.mem_get_info()
        return {
            "max_allocated_gib": float(torch.cuda.max_memory_allocated()) / 2**30,
            "max_reserved_gib": float(torch.cuda.max_memory_reserved()) / 2**30,
            "device_used_gib": float(total_b - free_b) / 2**30,
            "device_total_gib": float(total_b) / 2**30,
        }
    except Exception:
        return {}


def release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def probe_candidate(
    *,
    config: dict[str, Any],
    n_envs: int,
    seed: int,
    goals: list,
) -> dict[str, Any]:
    probe_cfg = config.get("probe", {})
    actions_cfg = config.get("actions", {})
    warmup_rounds = int(probe_cfg.get("warmup_rounds", 1))
    measured_rounds = int(probe_cfg.get("measured_rounds", 6))
    if warmup_rounds + measured_rounds > 10:
        raise ValueError("warmup+measured rounds must fit inside one T=10 episode batch")

    params = p1_rope_params()
    rng = np.random.default_rng(seed + 17 * n_envs)
    episode_goals = [goals[i % len(goals)] for i in range(n_envs)]

    state: dict[str, Any] = {"env": None, "runner": None, "begin_info": None}
    full_rebuilds = 0
    max_full_rebuilds = 3

    def build(build_offset: int) -> tuple[float, float]:
        if state["env"] is not None:
            state["runner"] = None
            state["env"] = None
            release_cuda()
        build_start = time.perf_counter()
        env = DLOLabEnv(**env_kwargs(config, n_envs))
        env.reset(params, init_shape="straight", seed=seed + n_envs + 10_000 * build_offset)
        build_s = time.perf_counter() - build_start
        if not env.supports_per_env_grasp():
            raise RuntimeError("per-env grasp hooks unavailable")
        runner = BatchedEpisodeRunner(env, params)
        reset_start = time.perf_counter()
        begin_info = runner.begin_episodes(
            seed=seed + 31 * n_envs + 10_000 * build_offset, goals=episode_goals
        )
        state.update(env=env, runner=runner, begin_info=begin_info)
        return build_s, time.perf_counter() - reset_start

    build_wall_s, light_reset_wall_s = build(0)

    def one_round() -> dict[str, Any]:
        p, deltas, lifts = random_policy_actions(
            rng,
            n_envs=n_envs,
            n_vertices=P1_N_SEGMENTS,
            delta_min_m=float(actions_cfg.get("delta_min_m", 0.02)),
            delta_max_m=float(actions_cfg.get("delta_max_m", 0.15)),
            lift_choices=tuple(actions_cfg.get("lift_choices", ("low", "high"))),
        )
        return state["runner"].step(p, deltas, lifts, rng=rng)

    def run_round_with_recovery() -> tuple[dict[str, Any] | None, float]:
        """One measured round; on unrecoverable NaN, full scene rebuild (P0 pattern)."""

        nonlocal full_rebuilds
        round_start = time.perf_counter()
        try:
            return one_round(), time.perf_counter() - round_start
        except FloatingPointError as exc:
            full_rebuilds += 1
            print(
                f"round_recovery n_envs={n_envs} rebuild={full_rebuilds} "
                f"error={type(exc).__name__}: {exc} action=full_scene_rebuild"
            )
            if full_rebuilds > max_full_rebuilds:
                raise
            build(full_rebuilds)
            return None, 0.0

    for _ in range(warmup_rounds):
        run_round_with_recovery()

    round_times: list[float] = []
    grasp_successes = 0
    settle_converged = 0
    discarded_rounds = 0
    measured = 0
    while measured < measured_rounds:
        record, round_wall = run_round_with_recovery()
        if record is None:
            continue  # rebuilt mid-measurement; retry the round on the fresh scene
        measured += 1
        round_times.append(round_wall)
        if record.get("discarded"):
            discarded_rounds += 1
            continue
        grasp_successes += int(record["grasp_success"].sum())
        settle_converged += int(record["settle_converged"].sum())

    memory = cuda_memory_snapshot()
    runner_incidents = int(state["runner"].nan_incidents)
    begin_info = state["begin_info"]
    state["runner"] = None
    state["env"] = None
    release_cuda()

    mean_round_s = float(np.mean(round_times))
    transitions_per_s = float(n_envs / mean_round_s) if mean_round_s > 0 else 0.0
    counted = max(1, (measured_rounds - discarded_rounds) * n_envs)
    return {
        "n_envs": int(n_envs),
        "warmup_rounds": warmup_rounds,
        "measured_rounds": measured_rounds,
        "build_wall_s": build_wall_s,
        "light_reset_wall_s": light_reset_wall_s,
        "round_wall_s": round_times,
        "mean_s_per_round": mean_round_s,
        "transitions_per_s": transitions_per_s,
        "grasp_success_rate": grasp_successes / counted,
        "settle_convergence_rate": settle_converged / counted,
        "nan_incidents_runner": runner_incidents,
        "nan_discarded_rounds": discarded_rounds,
        "full_scene_rebuilds": full_rebuilds,
        "reset_settle_converged_rate": float(np.mean(begin_info["reset_settle_converged"])),
        "cuda_memory": memory,
        "settle_budget": {
            "vel_threshold": SETTLE_VEL_THRESHOLD,
            "max_steps": SETTLE_MAX_STEPS,
        },
    }


def project_hours(transitions: float, tps: float) -> float:
    return float(transitions) / tps / 3600.0 if tps > 0 else float("inf")


def build_report(payload: dict[str, Any]) -> str:
    budgets = payload["budgets"]
    lines = [
        "# P1 Throughput Probe (M0)",
        "",
        f"Generated: {payload['generated_at']} · git {payload['git_commit']} · seed {payload['seed']}",
        "",
        "Scope: single-process n_envs scaling exactly as specified (P1.md @M0). "
        "Scheduling is S1 — one training run at a time; concurrent-run probing was "
        "explicitly rejected at plan reconciliation (R2), so no concurrency data exists or is claimed.",
        "",
        f"Settle budget on every settle-bearing call: vel_threshold={SETTLE_VEL_THRESHOLD}, "
        f"max_steps={SETTLE_MAX_STEPS} (global rule 7).",
        "",
        "## Measurements",
        "",
        "| n_envs | transitions/s | s/round | build s | grasp succ | settle conv | VRAM used GiB | NaN inc | rebuilds |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for result in payload["candidates"]:
        if "error" in result:
            lines.append(f"| {result['n_envs']} | FAILED: {result['error']} | | | | | | |")
            continue
        mem = result.get("cuda_memory", {})
        lines.append(
            f"| {result['n_envs']} | {result['transitions_per_s']:.2f} "
            f"| {result['mean_s_per_round']:.1f} | {result['build_wall_s']:.1f} "
            f"| {result['grasp_success_rate']:.3f} | {result['settle_convergence_rate']:.3f} "
            f"| {mem.get('device_used_gib', float('nan')):.1f}/{mem.get('device_total_gib', float('nan')):.0f} "
            f"| {result['nan_incidents_runner']} | {result['full_scene_rebuilds']} |"
        )

    rec = payload["recommendation"]
    tps = rec["transitions_per_s"]
    lines += [
        "",
        f"## Recommendation: n_envs = {rec['n_envs']} (S1)",
        "",
        f"Chosen by maximum measured transitions/s ({tps:.2f} tr/s). "
        "P0 reference: 3.61 tr/s at n_envs=64 with the 5000-step settle budget "
        "(P1 uses the 10000-step budget everywhere, so values are not directly comparable).",
        "",
        "## Projected run durations at the recommended n_envs (S1, serial)",
        "",
        "| Item | Transitions | Hours |",
        "|---|---|---|",
        f"| M2 smoke (1 run) | {budgets['smoke_transitions']:,} | {project_hours(budgets['smoke_transitions'], tps):.1f} |",
        f"| T1 run (each) | {budgets['t1_run_transitions']:,} | {project_hours(budgets['t1_run_transitions'], tps):.1f} |",
        f"| T2 run (each) | {budgets['t2_run_transitions']:,} | {project_hours(budgets['t2_run_transitions'], tps):.1f} |",
        f"| M3 total ({budgets['t1_runs']} runs) | {budgets['t1_runs'] * budgets['t1_run_transitions']:,} | {project_hours(budgets['t1_runs'] * budgets['t1_run_transitions'], tps):.1f} |",
        f"| M4 total ({budgets['t2_runs']} runs) | {budgets['t2_runs'] * budgets['t2_run_transitions']:,} | {project_hours(budgets['t2_runs'] * budgets['t2_run_transitions'], tps):.1f} |",
    ]
    total_tr = (
        budgets["smoke_transitions"]
        + budgets["t1_runs"] * budgets["t1_run_transitions"]
        + budgets["t2_runs"] * budgets["t2_run_transitions"]
    )
    lines += [
        f"| **P1 training total** | {total_tr:,} | {project_hours(total_tr, tps):.1f} |",
        "",
        "Eval episodes (every 25k transitions) and checkpointing are additional "
        "overhead on top of these collection-only projections.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="P1-M0 throughput probe")
    parser.add_argument("--config", type=Path, default=Path("configs/p1_throughput.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    outputs = config.get("outputs", {})
    log_path = Path(outputs.get("stdout_log", "outputs/reports/p1_throughput_stdout.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    original_stdout = sys.stdout
    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)  # type: ignore[assignment]
        try:
            print(f"p1_throughput_probe start {utc_now()} seed={args.seed}")
            goals = [goal for _, goal in load_t2_split("train")]
            candidates_cfg = [int(v) for v in config.get("probe", {}).get("n_env_candidates", [64, 128, 256])]

            candidates: list[dict[str, Any]] = []
            for n_envs in candidates_cfg:
                print(f"probe n_envs={n_envs} start {utc_now()}")
                try:
                    result = probe_candidate(
                        config=config, n_envs=n_envs, seed=int(args.seed), goals=goals
                    )
                    candidates.append(result)
                    print(
                        f"probe n_envs={n_envs} transitions_per_s={result['transitions_per_s']:.3f} "
                        f"mean_s_per_round={result['mean_s_per_round']:.2f} "
                        f"grasp_success_rate={result['grasp_success_rate']:.3f} "
                        f"settle_convergence_rate={result['settle_convergence_rate']:.3f}"
                    )
                except Exception as exc:  # noqa: BLE001 — record and continue probing
                    release_cuda()
                    candidates.append({"n_envs": int(n_envs), "error": f"{type(exc).__name__}: {exc}"})
                    print(f"probe n_envs={n_envs} FAILED {type(exc).__name__}: {exc}")

            valid = [c for c in candidates if "transitions_per_s" in c]
            if not valid:
                raise RuntimeError(f"no probe candidate succeeded: {candidates}")
            best = max(valid, key=lambda c: (c["transitions_per_s"], -c["n_envs"]))

            payload = {
                "generated_at": utc_now(),
                "git_commit": get_git_commit_hash(),
                "seed": int(args.seed),
                "config": config,
                "scheduling": "S1 (one run at a time; R2 concurrency probing rejected at reconciliation)",
                "candidates": candidates,
                "recommendation": {
                    "n_envs": best["n_envs"],
                    "transitions_per_s": best["transitions_per_s"],
                    "rule": "max measured transitions/s, tie-break smaller n_envs",
                },
                "budgets": config.get("budgets", {}),
                "p0_reference": {"n_envs": 64, "transitions_per_s": 3.61, "settle_max_steps": 5000},
            }

            json_path = Path(outputs.get("json", "outputs/metrics/p1_throughput.json"))
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

            report_path = Path(outputs.get("report", "outputs/reports/p1_throughput.md"))
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(build_report(payload), encoding="utf-8")

            print(f"recommended n_envs={best['n_envs']} transitions_per_s={best['transitions_per_s']:.3f}")
            print(f"wrote {json_path} and {report_path}")
            return 0
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    raise SystemExit(main())
