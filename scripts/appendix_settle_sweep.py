"""M7 appendix: settle-budget sweep at immutable 1e-3 threshold.

This report-only measurement is sanctioned by the M6 issue-#7 verdict for M7
numeric-fixing.  It samples 24 deterministic init/action cases and performs one
post-release settle rollout to 20,000 steps, recording the first threshold
crossing and centerline snapshots at 5,000/10,000/20,000.  Rates for lower
budgets are derived from the same rollout; the 1e-3 threshold is not changed.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dgcc.envs.dlolab import DLOLabEnv, MAX_DELTA_NORM, analytic_init_centerline, centerline_arc_length
from dgcc.goals.distance import chamfer_distance
from dgcc.utils.meta import get_git_commit_hash
from gate_g1 import (
    Tee,
    as_jsonable,
    cleanup_env,
    describe_distribution,
    env_kwargs,
    load_config,
    params_from_config,
    write_json,
)

os.environ.pop("DISPLAY", None)

BUDGETS = (5000, 10000, 20000)
INIT_SHAPES = ("straight", "u_bend", "s_curve", "random_smooth")


@dataclass(frozen=True)
class SweepCase:
    case_id: str
    init_shape: str
    init_seed: int
    p: int
    delta: tuple[float, float, float]
    lift: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the M7 appendix settle-budget sweep")
    parser.add_argument("--config", type=Path, default=Path("configs/gate_g1.yaml"))
    parser.add_argument("--seed", type=int, default=8401)
    parser.add_argument("--n-cases", type=int, default=24)
    parser.add_argument("--metrics", type=Path, default=Path("outputs/metrics/settle_budget_sweep.json"))
    parser.add_argument("--plot", type=Path, default=Path("outputs/plots/settle_budget_sweep.png"))
    parser.add_argument("--log", type=Path, default=Path("outputs/reports/appendix_settle_sweep.log"))
    return parser


@contextmanager
def tee_stdout(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original = sys.stdout
    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original, log_file)
        try:
            yield
        finally:
            sys.stdout = original


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_cases(*, seed: int, n_cases: int, n_vertices: int) -> list[SweepCase]:
    if n_cases != 24:
        raise ValueError("assignment requires 24 settle-sweep cases")
    rng = np.random.default_rng(seed)
    cases: list[SweepCase] = []
    for idx in range(n_cases):
        init_shape = INIT_SHAPES[idx % len(INIT_SHAPES)]
        init_seed = int(seed + 500 + idx)
        p = int(rng.integers(4, max(5, n_vertices - 4)))
        xy_norm = float(rng.uniform(0.035, 0.125))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        z = float(rng.uniform(-0.020, 0.035))
        delta = np.asarray([xy_norm * np.cos(theta), xy_norm * np.sin(theta), z], dtype=float)
        norm = float(np.linalg.norm(delta))
        if norm > MAX_DELTA_NORM:
            delta *= MAX_DELTA_NORM / norm
        lift = "high" if idx % 2 == 0 else "low"
        cases.append(
            SweepCase(
                case_id=f"case_{idx:02d}",
                init_shape=init_shape,
                init_seed=init_seed,
                p=p,
                delta=tuple(float(x) for x in delta),
                lift=lift,
            )
        )
    return cases


def initial_vertices(base_params: Any, cases: Sequence[SweepCase]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    vertices: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for case in cases:
        curve = analytic_init_centerline(base_params, case.init_shape, case.init_seed)
        vertices.append(curve)
        metadata.append(
            {
                "case_id": case.case_id,
                "init_shape": case.init_shape,
                "init_seed": case.init_seed,
                "initial_arc_length_m": centerline_arc_length(curve),
                "p": case.p,
                "delta": list(case.delta),
                "delta_norm_m": float(np.linalg.norm(np.asarray(case.delta, dtype=float))),
                "lift": case.lift,
            }
        )
    return np.stack(vertices, axis=0), metadata


def settle_with_budget_snapshots(
    env: DLOLabEnv,
    *,
    budgets: Sequence[int],
    threshold: float,
) -> dict[str, Any]:
    budget_tuple = tuple(int(value) for value in budgets)
    budget_set = set(budget_tuple)
    max_budget = max(budget_tuple)
    first_steps = np.full(env.n_envs, -1, dtype=int)
    snapshots: dict[int, np.ndarray] = {}
    speeds_at_budget: dict[int, np.ndarray] = {}
    speed_summary_at_budget: dict[int, dict[str, Any]] = {}

    for step in range(max_budget + 1):
        speeds = env.max_node_speed_batch()
        newly = (first_steps < 0) & (speeds < float(threshold))
        if np.any(newly):
            first_steps[newly] = step
        if step in budget_set:
            snapshots[step] = env.get_centerline_batch()
            speeds_at_budget[step] = speeds.copy()
            speed_summary_at_budget[step] = describe_distribution(speeds)
            rate = float(np.mean((first_steps >= 0) & (first_steps <= step)))
            print(f"settle_budget_checkpoint step={step} convergence_rate={rate:.3f} max_speed={float(np.max(speeds)):.6g}")
        if step == max_budget:
            break
        env._step_scene()

    return {
        "first_steps": first_steps,
        "snapshots": snapshots,
        "speeds_at_budget": speeds_at_budget,
        "speed_summary_at_budget": speed_summary_at_budget,
    }


def convergence_rate(first_steps: np.ndarray, budget: int) -> float:
    return float(np.mean((first_steps >= 0) & (first_steps <= int(budget))))


def shape_change_stats(
    snapshots: dict[int, np.ndarray],
    first_steps: np.ndarray,
    *,
    length_m: float,
) -> dict[str, Any]:
    nonconverged_5000 = (first_steps < 0) | (first_steps > 5000)
    pairs = ((5000, 10000), (10000, 20000), (5000, 20000))
    results: dict[str, Any] = {}
    for left, right in pairs:
        values = [
            float(chamfer_distance(snapshots[left][idx], snapshots[right][idx], length_m))
            for idx in range(first_steps.shape[0])
            if bool(nonconverged_5000[idx])
        ]
        key = f"{left}_vs_{right}"
        results[key] = {
            "subset": "cases not converged by 5000 steps",
            "n_cases": len(values),
            "values": values,
            "stats": describe_distribution(values),
        }
    return {
        "nonconverged_at_5000_count": int(np.sum(nonconverged_5000)),
        "nonconverged_at_5000_case_ids": None,
        "pairs": results,
    }


def run_sweep(*, config: dict[str, Any], base_params: Any, cases: Sequence[SweepCase], seed: int) -> dict[str, Any]:
    measurement = config.get("measurement", {})
    threshold = float(measurement.get("vel_threshold", 1.0e-3))
    n_envs = len(cases)
    env: DLOLabEnv | None = None
    total_start = time.perf_counter()
    try:
        kwargs = env_kwargs(config, n_envs)
        kwargs["grasp_realism"] = False
        env = DLOLabEnv(**kwargs)
        build_start = time.perf_counter()
        reset_info = env.reset(base_params, init_shape="straight", seed=seed + 10_000)
        build_wall_s = time.perf_counter() - build_start
        if not env.supports_per_env_grasp():
            raise RuntimeError("DLO-Lab per-env grasp hooks are unavailable")

        vertices, case_metadata = initial_vertices(base_params, cases)
        reset_start = time.perf_counter()
        reset_result = env.light_reset(vertices, vel_threshold=threshold, max_steps=5000)
        reset_wall_s = time.perf_counter() - reset_start

        action_start = time.perf_counter()
        action_result = env.step_primitive_batch(
            np.asarray([case.p for case in cases], dtype=int),
            np.asarray([case.delta for case in cases], dtype=float),
            [case.lift for case in cases],
            vel_threshold=threshold,
            max_steps=0,
            rng=np.random.default_rng(seed + 20_000),
        )
        action_wall_s = time.perf_counter() - action_start
        if not bool(np.all(np.asarray(action_result["grasp_success"], dtype=bool))):
            raise RuntimeError("settle sweep runs with grasp_realism off and expects all grasps to succeed")

        settle_start = time.perf_counter()
        settle_result = settle_with_budget_snapshots(env, budgets=BUDGETS, threshold=threshold)
        settle_wall_s = time.perf_counter() - settle_start

        first_steps = np.asarray(settle_result["first_steps"], dtype=int)
        converged_steps = first_steps[first_steps >= 0]
        rates = {str(budget): convergence_rate(first_steps, budget) for budget in BUDGETS}
        shape_changes = shape_change_stats(settle_result["snapshots"], first_steps, length_m=base_params.length_m)
        nonconverged_ids = [case.case_id for case, step in zip(cases, first_steps, strict=True) if step < 0 or step > 5000]
        shape_changes["nonconverged_at_5000_case_ids"] = nonconverged_ids

        per_case: list[dict[str, Any]] = []
        for idx, (case, metadata) in enumerate(zip(cases, case_metadata, strict=True)):
            first = int(first_steps[idx])
            row = {
                **metadata,
                "grasp_success": bool(np.asarray(action_result["grasp_success"], dtype=bool)[idx]),
                "actual_grasp_node": int(np.asarray(action_result["info"]["p_actual"], dtype=int)[idx]),
                "first_convergence_step": None if first < 0 else first,
                "converged_by_budget": {str(budget): bool(first >= 0 and first <= budget) for budget in BUDGETS},
                "max_node_speed_at_budget": {
                    str(budget): float(settle_result["speeds_at_budget"][budget][idx]) for budget in BUDGETS
                },
            }
            if bool((first < 0) or (first > 5000)):
                row["shape_change_length_normalized"] = {
                    f"{left}_vs_{right}": float(
                        chamfer_distance(
                            settle_result["snapshots"][left][idx],
                            settle_result["snapshots"][right][idx],
                            base_params.length_m,
                        )
                    )
                    for left, right in ((5000, 10000), (10000, 20000), (5000, 20000))
                }
            per_case.append(row)

        reset_converged = np.asarray(reset_result["settle_converged"], dtype=bool)
        reset_steps = np.asarray(reset_result["settle_steps"], dtype=int)
        payload = {
            "n_envs": n_envs,
            "budgets": list(BUDGETS),
            "vel_threshold": threshold,
            "method": "single post-release rollout to 20000 steps; first-crossing steps and snapshots at 5000/10000/20000 derive all rates and cutoff shape changes",
            "grasp_realism": False,
            "grasp_realism_reason": "disabled to isolate settle-budget/solver behavior; repeat-variance measurement separately reports realism-on execution noise",
            "reset_info": reset_info,
            "reset_settle": {
                "converged_rate": float(np.mean(reset_converged)),
                "steps": describe_distribution(reset_steps),
            },
            "action": {
                "wall_s": action_wall_s,
                "grasp_success_rate": float(np.mean(np.asarray(action_result["grasp_success"], dtype=bool))),
                "lifts": {"high": sum(case.lift == "high" for case in cases), "low": sum(case.lift == "low" for case in cases)},
                "init_shapes": {shape: sum(case.init_shape == shape for case in cases) for shape in INIT_SHAPES},
            },
            "convergence_rates": rates,
            "first_convergence_steps": {
                "values": [None if step < 0 else int(step) for step in first_steps],
                "converged_count_by_20000": int(converged_steps.size),
                "nonconverged_count_by_20000": int(np.sum(first_steps < 0)),
                "stats_converged_only": describe_distribution(converged_steps),
            },
            "speed_summary_at_budget": {
                str(budget): settle_result["speed_summary_at_budget"][budget] for budget in BUDGETS
            },
            "shape_change_between_cutoffs": shape_changes,
            "cases": per_case,
            "wall_s": time.perf_counter() - total_start,
            "build_wall_s": build_wall_s,
            "initial_reset_wall_s": reset_wall_s,
            "settle_rollout_wall_s": settle_wall_s,
        }
        print(
            "settle_sweep_done "
            f"n_cases={n_envs} wall_s={payload['wall_s']:.3f} "
            f"rates=5000:{rates['5000']:.3f},10000:{rates['10000']:.3f},20000:{rates['20000']:.3f}"
        )
        return payload
    finally:
        cleanup_env(env)


def plot_results(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    result = payload["result"]
    budgets = [str(budget) for budget in result["budgets"]]
    rates = [result["convergence_rates"][budget] for budget in budgets]
    axes[0].bar(budgets, rates, color="#4c78a8", alpha=0.8)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xlabel("max_steps budget")
    axes[0].set_ylabel("Convergence rate")
    axes[0].set_title("Rate at immutable 1e-3 threshold")
    axes[0].grid(axis="y", alpha=0.3)

    steps = [value for value in result["first_convergence_steps"]["values"] if value is not None]
    if steps:
        axes[1].hist(steps, bins=12, color="#59a14f", alpha=0.75)
    axes[1].axvline(5000, color="#e15759", linestyle="--", linewidth=1.2, label="5000")
    axes[1].axvline(10000, color="#f28e2b", linestyle="--", linewidth=1.2, label="10000")
    axes[1].axvline(20000, color="#4e79a7", linestyle="--", linewidth=1.2, label="20000")
    axes[1].set_xlabel("First convergence step")
    axes[1].set_ylabel("Cases")
    axes[1].set_title("First-crossing distribution")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    pairs = result["shape_change_between_cutoffs"]["pairs"]
    labels = list(pairs)
    data = [pairs[label]["values"] for label in labels]
    axes[2].boxplot(data, showfliers=True)
    axes[2].set_xticks(np.arange(1, len(labels) + 1), [label.replace("_vs_", "→") for label in labels])
    axes[2].set_ylabel("Length-normalized Chamfer")
    axes[2].set_title("Cutoff shape changes\n(non-converged by 5000)")
    axes[2].tick_params(axis="x", labelrotation=20)
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("M7 appendix settle-budget sweep (24 cases, one 20k rollout)")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    with tee_stdout(args.log):
        total_start = time.perf_counter()
        config, config_text = load_config(args.config)
        base_params = params_from_config(config)
        cases = make_cases(seed=args.seed, n_cases=args.n_cases, n_vertices=base_params.n_segments)
        print(f"appendix_settle_sweep_start seed={args.seed} config={args.config} n_cases={args.n_cases}")
        print("method=single 20000-step rollout with snapshots at 5000/10000/20000; threshold remains 1e-3")
        result = run_sweep(config=config, base_params=base_params, cases=cases, seed=args.seed)
        payload = {
            "schema_version": 1,
            "measurement": "M7 appendix settle-budget sweep",
            "status": "report_only",
            "sanction": "M6 issue-#7 verdict instruction #3; M7 numeric-fixing appendix",
            "created_utc": utc_now(),
            "commit_hash": get_git_commit_hash(),
            "config_path": str(args.config),
            "config_sha256": sha256_text(config_text),
            "config_yaml": config_text,
            "seed": int(args.seed),
            "rope_params": as_jsonable(base_params.__dict__),
            "design": {
                "n_cases": int(args.n_cases),
                "init_shape_schedule": "balanced cycle over straight/u_bend/s_curve/random_smooth (6 each)",
                "lift_schedule": "alternating high/low (12 high, 12 low)",
                "budgets": list(BUDGETS),
                "threshold_immutable": 1.0e-3,
                "plasticity": "disabled/not used",
            },
            "result": result,
            "convergence_rates": result["convergence_rates"],
            "first_convergence_steps": result["first_convergence_steps"],
            "shape_change_between_cutoffs": result["shape_change_between_cutoffs"],
            "wall_time_s": time.perf_counter() - total_start,
            "outputs": {
                "metrics_json": str(args.metrics),
                "plot_png": str(args.plot),
                "stdout_log": str(args.log),
            },
        }
        write_json(args.metrics, payload)
        plot_results(payload, args.plot)
        print(f"wrote_metrics {args.metrics}")
        print(f"wrote_plot {args.plot}")
        print(f"appendix_settle_sweep_done wall_s={payload['wall_time_s']:.3f}")


if __name__ == "__main__":
    main()
