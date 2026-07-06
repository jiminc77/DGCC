"""P1-O1 greedy oracle feasibility reference for T1 tasks.

Runs the same shared evaluate_episodes path used by the training driver's
deterministic eval, with a hand-coded greedy residual policy. The output is a
feasibility reference, not an attainable-performance upper bound.
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
from dgcc.goals.distance import canonical_shape_flip
from dgcc.models.networks import DELTA_SCALE, goal_residual_flips
from dgcc.rl.evaluation import evaluate_episodes
from dgcc.tasks.domain import RewardConstants, SETTLE_MAX_STEPS, p1_rope_params
from dgcc.tasks.episode import BatchedEpisodeRunner, EpisodeConfig, is_nonfinite_error
from dgcc.tasks.t1 import T1_TASKS, sample_t1_goal
from dgcc.utils.meta import get_git_commit_hash

P_C_INTERPRETATION_RULE = "oracle 성공 → 과제 달성 가능 확정 · oracle ≫ policy → 학습 문제 확정 · oracle ≈ 0 → 판정 불능 (불가능 증명 아님)"
TASK_CHOICES = tuple(T1_TASKS)
DEFAULT_JSON = Path("outputs/metrics/p1_o1_oracle.json")
DEFAULT_REPORT = Path("outputs/reports/p1_o1_oracle.md")
DEFAULT_STDOUT_LOG = Path("outputs/reports/p1_o1_oracle_stdout.log")


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


def env_kwargs(config: dict[str, Any], n_envs: int, *, grasp_realism: bool) -> dict[str, Any]:
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
        "grasp_realism": bool(grasp_realism),
    }


def release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def parse_tasks(raw: str) -> list[str]:
    tasks = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [task for task in tasks if task not in TASK_CHOICES]
    if unknown:
        allowed = ", ".join(TASK_CHOICES)
        raise argparse.ArgumentTypeError(f"unknown task(s) {unknown}; expected comma list from {{{allowed}}}")
    if not tasks:
        raise argparse.ArgumentTypeError("at least one task is required")
    return tasks


def greedy_oracle_actions(
    X: np.ndarray, G_curve: np.ndarray, _rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Residual-greedy policy under the encoder's canonical flip convention."""

    x = np.asarray(X, dtype=float)
    g = np.asarray(G_curve, dtype=float)
    flips = goal_residual_flips(x, g)
    g_aligned = np.where(flips[:, None, None], g[:, ::-1, :], g)
    residual = g_aligned - x
    norms = np.linalg.norm(residual, axis=2)
    p = np.argmax(norms, axis=1).astype(int)
    batch = np.arange(x.shape[0])
    chosen = residual[batch, p, :]
    chosen_norm = np.linalg.norm(chosen, axis=1)
    scale = np.zeros_like(chosen_norm)
    nonzero = chosen_norm > 0.0
    scale[nonzero] = np.minimum(chosen_norm[nonzero], DELTA_SCALE) / chosen_norm[nonzero]
    delta = chosen * scale[:, None]
    if not np.all(np.isfinite(delta)):
        raise FloatingPointError("oracle produced non-finite delta")
    if not np.all(np.linalg.norm(delta, axis=1) <= DELTA_SCALE + 1.0e-12):
        raise AssertionError("oracle delta norm exceeded DELTA_SCALE")
    if not np.all(np.abs(delta) <= DELTA_SCALE + 1.0e-12):
        raise AssertionError("oracle delta per-axis clamp would not be a no-op")
    lift = ["high" if float(vec[2]) > 0.0 else "low" for vec in chosen]
    return p, delta.astype(float, copy=False), lift


def template_stats(episodes: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    for template in sorted({str(ep["init_template"]) for ep in episodes}):
        rows = [ep for ep in episodes if str(ep["init_template"]) == template]
        success = np.asarray([1.0 if row["success"] else 0.0 for row in rows], dtype=float)
        returns = np.asarray([float(row["return"]) for row in rows], dtype=float)
        final_d = np.asarray([float(row["final_d"]) for row in rows], dtype=float)
        d_at_done = np.asarray([float(row.get("d_at_done", row["final_d"])) for row in rows], dtype=float)
        min_d = np.asarray([float(row.get("min_d", row["final_d"])) for row in rows], dtype=float)
        stats[template] = {
            "n": int(len(rows)),
            "success_rate": float(success.mean()) if len(rows) else float("nan"),
            "mean_return": float(returns.mean()) if len(rows) else float("nan"),
            "mean_final_d": float(final_d.mean()) if len(rows) else float("nan"),
            "mean_d_at_done": float(d_at_done.mean()) if len(rows) else float("nan"),
            "mean_min_d": float(min_d.mean()) if len(rows) else float("nan"),
        }
    return stats


def evaluate_task_block(
    *,
    runner: BatchedEpisodeRunner,
    task: str,
    episodes: int,
    seed: int,
    episode_index_start: int,
    action_rng: np.random.Generator,
) -> dict[str, Any]:
    result = evaluate_episodes(
        runner,
        n_episodes=episodes,
        seed=seed + 500,
        episode_index_start=episode_index_start,
        action_fn=greedy_oracle_actions,
        rng=action_rng,
        goal_fn=lambda env_idx, x, goal_rng, _task=task: sample_t1_goal(_task, x, goal_rng),
    )
    result["per_template_stats"] = template_stats(result["episodes"])
    return result


def build_runner(
    *,
    config: dict[str, Any],
    n_envs: int,
    seed: int,
    grasp_realism: bool,
    rebuild_index: int,
    episode_config: EpisodeConfig,
) -> BatchedEpisodeRunner:
    params = p1_rope_params()
    env = DLOLabEnv(**env_kwargs(config, n_envs, grasp_realism=grasp_realism))
    env.reset(params, init_shape="straight", seed=seed + 10_000 * (rebuild_index + 1))
    if not env.supports_per_env_grasp():
        raise RuntimeError("per-env grasp hooks unavailable")
    return BatchedEpisodeRunner(env, params, episode_config)


def write_report(payload: dict[str, Any], report_path: Path) -> None:
    lines: list[str] = []
    lines.append("# P1-O1 Oracle Feasibility Reference")
    lines.append("")
    lines.append("이 리포트는 **feasibility reference**이다. Attainability upper bound가 아니다.")
    lines.append("")
    lines.append("## P-c interpretation rule")
    lines.append("")
    lines.append(f"> {P_C_INTERPRETATION_RULE}")
    lines.append("")
    lines.append("## Oracle policy symbol choices")
    lines.append("")
    lines.append("- Flip convention: `dgcc.models.networks.goal_residual_flips`, which routes through `dgcc.goals.distance.canonical_shape_flip`.")
    lines.append("- Residual: `res = g_aligned - x` using the encoder's index-wise goal correspondence.")
    lines.append("- Action: `p = argmax_i ||res_i||`; `delta = direction(res_p) * min(||res_p||, 0.15)`; `lift = high` iff `res_p[z] > 0`.")
    lines.append("- Delta assertion: norm and per-axis bounds are checked before execution, so the environment clamp should be a no-op for the oracle command.")
    lines.append("")
    lines.append("## evaluate_episodes hook assumption")
    lines.append("")
    lines.append("The oracle uses the same `evaluate_episodes` hook as `p1_train.py::deterministic_eval`: T1 `goal_fn`, `seed + 500`, `rng seed + 501`, and an episode-index base in the 90,000 eval namespace.")
    lines.append("")
    lines.append("## Side-by-side feasibility reference")
    lines.append("")
    lines.append("| task | grasp realism | success | return | final D | d_at_done | min D | NaN incidents |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for pass_name, pass_data in payload["passes"].items():
        realism = "ON" if pass_data["grasp_realism"] else "OFF"
        for task, result in pass_data["blocks"].items():
            lines.append(
                f"| {task} | {realism} | {result['success_rate']:.3f} | {result['mean_return']:.3f} | "
                f"{result['mean_final_d']:.4f} | {result['mean_d_at_done']:.4f} | "
                f"{result['mean_min_d']:.4f} | {result['nan_incidents_during_eval']} |"
            )
    lines.append("")
    lines.append("## Per-template stats")
    lines.append("")
    for task in payload["tasks"]:
        lines.append(f"### {task}")
        lines.append("")
        lines.append("| template | ON success | ON return | ON d_at_done | ON min D | OFF success | OFF return | OFF d_at_done | OFF min D |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        templates = set()
        for pass_data in payload["passes"].values():
            if task in pass_data["blocks"]:
                templates.update(pass_data["blocks"][task]["per_template_stats"].keys())
        for template in sorted(templates):
            cells = []
            for pass_key in ("grasp_realism_on", "grasp_realism_off"):
                block = payload["passes"].get(pass_key, {}).get("blocks", {}).get(task)
                stats = (block or {}).get("per_template_stats", {}).get(template)
                if stats is None:
                    cells.extend(["—", "—", "—", "—"])
                else:
                    cells.extend(
                        [
                            f"{stats['success_rate']:.3f}",
                            f"{stats['mean_return']:.3f}",
                            f"{stats['mean_d_at_done']:.4f}",
                            f"{stats['mean_min_d']:.4f}",
                        ]
                    )
            lines.append(f"| {template} | " + " | ".join(cells) + " |")
        lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="P1-O1 greedy oracle feasibility reference")
    parser.add_argument("--config", type=Path, default=Path("configs/p1_random_reference.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=100, help="episodes per T1 task")
    parser.add_argument("--n-envs", type=int, default=256, help="batched env count, matching driver eval lanes")
    parser.add_argument("--tasks", type=parse_tasks, default=list(TASK_CHOICES), help="comma-separated T1 tasks")
    parser.add_argument("--skip-off-pass", action="store_true", help="run only grasp-realism ON/config-default pass")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--stdout-log", type=Path, default=DEFAULT_STDOUT_LOG)
    args = parser.parse_args()

    if args.episodes < 1:
        raise SystemExit("--episodes must be >= 1")
    if args.n_envs < 1:
        raise SystemExit("--n-envs must be >= 1")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    reward_cfg = config.get("reward", {})
    episode_config = EpisodeConfig(
        reward=RewardConstants(
            alpha=float(reward_cfg.get("alpha", 10.0)),
            c_step=float(reward_cfg.get("c_step", 0.1)),
            r_succ=float(reward_cfg.get("r_succ", 5.0)),
        )
    )
    config_default_grasp = bool(config.get("sim", {}).get("grasp_realism", True))
    passes: list[tuple[str, bool]] = [("grasp_realism_on", config_default_grasp)]
    if not args.skip_off_pass:
        passes.append(("grasp_realism_off", False))

    args.stdout_log.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    with args.stdout_log.open("w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)  # type: ignore[assignment]
        try:
            print(
                f"p1_o1_oracle start {utc_now()} seed={args.seed} n_envs={args.n_envs} "
                f"episodes={args.episodes} tasks={','.join(args.tasks)}"
            )
            payload: dict[str, Any] = {
                "generated_at": utc_now(),
                "git_commit": get_git_commit_hash(),
                "kind": "P1-O1 feasibility reference",
                "label": "feasibility reference",
                "interpretation_rule": P_C_INTERPRETATION_RULE,
                "policy": {
                    "flip_function": "dgcc.models.networks.goal_residual_flips",
                    "canonical_function": f"{canonical_shape_flip.__module__}.{canonical_shape_flip.__name__}",
                    "residual": "g_aligned - x",
                    "p": "argmax_i ||res_i||",
                    "delta": "direction(res_p) * min(||res_p||, 0.15)",
                    "lift": "high iff res_p[z] > 0",
                },
                "evaluate_episodes_assumption": {
                    "path": "dgcc.rl.evaluation.evaluate_episodes",
                    "driver_seed_convention": "seed + 500 for eval reset; seed + 501 for action rng",
                    "episode_index_start": 90_001,
                },
                "seed": int(args.seed),
                "n_envs": int(args.n_envs),
                "episodes_per_task": int(args.episodes),
                "tasks": list(args.tasks),
                "config": config,
                "protocol": {
                    "horizon": episode_config.horizon,
                    "settle_max_steps": episode_config.settle_max_steps,
                    "vel_threshold": episode_config.vel_threshold,
                    "reward": vars(episode_config.reward),
                },
                "passes": {},
            }

            for pass_index, (pass_name, grasp_realism) in enumerate(passes):
                pass_blocks: dict[str, Any] = {}
                rebuilds = 0
                runner = build_runner(
                    config=config,
                    n_envs=int(args.n_envs),
                    seed=int(args.seed) + 100_000 * pass_index,
                    grasp_realism=grasp_realism,
                    rebuild_index=rebuilds,
                    episode_config=episode_config,
                )

                for task_index, task in enumerate(args.tasks):
                    start = time.perf_counter()
                    while True:
                        try:
                            result = evaluate_task_block(
                                runner=runner,
                                task=task,
                                episodes=int(args.episodes),
                                seed=int(args.seed),
                                episode_index_start=90_001 + task_index,
                                action_rng=np.random.default_rng(int(args.seed) + 501),
                            )
                            break
                        except (FloatingPointError, ValueError, RuntimeError) as exc:
                            if not is_nonfinite_error(exc):
                                raise
                            rebuilds += 1
                            print(
                                f"block_recovery pass={pass_name} task={task} rebuild={rebuilds} "
                                f"error={exc} action=full_scene_rebuild (block restarted)"
                            )
                            if rebuilds > 3:
                                raise
                            runner = build_runner(
                                config=config,
                                n_envs=int(args.n_envs),
                                seed=int(args.seed) + 100_000 * pass_index,
                                grasp_realism=grasp_realism,
                                rebuild_index=rebuilds,
                                episode_config=episode_config,
                            )
                    result["wall_s"] = time.perf_counter() - start
                    pass_blocks[task] = result
                    print(
                        f"{pass_name} {task}: episodes={result['n_episodes']} "
                        f"success={result['success_rate']:.3f} return={result['mean_return']:.3f} "
                        f"d_at_done={result['mean_d_at_done']:.4f} min_d={result['mean_min_d']:.4f} "
                        f"wall_s={result['wall_s']:.0f}"
                    )
                payload["passes"][pass_name] = {
                    "label": "feasibility reference",
                    "grasp_realism": bool(grasp_realism),
                    "full_scene_rebuilds": int(rebuilds),
                    "blocks": pass_blocks,
                }
                runner = None  # release before next pass
                release_cuda()

            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
            write_report(payload, args.report)
            print(f"wrote {args.json} and {args.report}")
            return 0
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    raise SystemExit(main())
