"""P1-M2 random-policy reference lines (floor for all later comparisons).

Evaluates the random policy on T1 x 3 tasks (100 episodes each) and the T2
validation split (50 goals x 2 episodes) under the P1 episode protocol, and
writes per-episode arrays (needed by the §3 statistical gate register
bootstraps) to ``outputs/metrics/p1_random_reference.json``.
"""

from __future__ import annotations

import argparse
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
from dgcc.rl.evaluation import evaluate_episodes
from dgcc.tasks.domain import P1_N_SEGMENTS, RewardConstants, SETTLE_MAX_STEPS, p1_rope_params
from dgcc.tasks.episode import BatchedEpisodeRunner, EpisodeConfig, random_policy_actions
from dgcc.tasks.t1 import sample_t1_goal
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


def main() -> int:
    parser = argparse.ArgumentParser(description="P1-M2 random-policy reference")
    parser.add_argument("--config", type=Path, default=Path("configs/p1_random_reference.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    outputs = config.get("outputs", {})
    ref_cfg = config.get("reference", {})
    actions_cfg = config.get("actions", {})
    n_envs = int(ref_cfg.get("n_envs", 100))
    t1_episodes = int(ref_cfg.get("t1_episodes_per_task", 100))

    log_path = Path(outputs.get("stdout_log", "outputs/reports/p1_random_reference.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    original_stdout = sys.stdout
    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)  # type: ignore[assignment]
        try:
            print(f"p1_random_reference start {utc_now()} seed={args.seed} n_envs={n_envs}")
            params = p1_rope_params()
            reward_cfg = config.get("reward", {})
            episode_config = EpisodeConfig(
                reward=RewardConstants(
                    alpha=float(reward_cfg.get("alpha", 10.0)),
                    c_step=float(reward_cfg.get("c_step", 0.1)),
                    r_succ=float(reward_cfg.get("r_succ", 5.0)),
                )
            )

            state: dict[str, Any] = {"env": None, "runner": None, "rebuilds": 0}

            def build_scene() -> None:
                if state["env"] is not None:
                    state["runner"] = None
                    state["env"] = None
                    import gc

                    gc.collect()
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                env = DLOLabEnv(**env_kwargs(config, n_envs))
                env.reset(
                    params,
                    init_shape="straight",
                    seed=int(args.seed) + 10_000 * state["rebuilds"],
                )
                state["env"] = env
                state["runner"] = BatchedEpisodeRunner(env, params, episode_config)

            def run_block(eval_fn, *, max_rebuilds: int = 3):
                """P0-pattern full-scene-rebuild escalation around one block."""

                while True:
                    try:
                        return eval_fn(state["runner"])
                    except FloatingPointError as exc:
                        state["rebuilds"] += 1
                        print(
                            f"block_recovery rebuild={state['rebuilds']} error={exc} "
                            "action=full_scene_rebuild (block restarted)"
                        )
                        if state["rebuilds"] > max_rebuilds:
                            raise
                        build_scene()

            build_scene()
            rng = np.random.default_rng(int(args.seed) + 1)

            def random_action_fn(
                X: np.ndarray, G: np.ndarray, action_rng: np.random.Generator
            ) -> tuple[np.ndarray, np.ndarray, list[str]]:
                del X, G
                return random_policy_actions(
                    action_rng,
                    n_envs=n_envs,
                    n_vertices=P1_N_SEGMENTS,
                    delta_min_m=float(actions_cfg.get("delta_min_m", 0.02)),
                    delta_max_m=float(actions_cfg.get("delta_max_m", 0.15)),
                    lift_choices=tuple(actions_cfg.get("lift_choices", ("low", "high"))),
                )

            blocks: dict[str, Any] = {}
            episode_index = 0
            for task in ("t1a_straighten", "t1b_single_bend", "t1c_endpoint_reposition"):
                start = time.perf_counter()
                result = run_block(
                    lambda active_runner, _task=task: evaluate_episodes(
                        active_runner,
                        n_episodes=t1_episodes,
                        seed=int(args.seed) + 100,
                        episode_index_start=episode_index,
                        action_fn=random_action_fn,
                        rng=rng,
                        goal_fn=lambda i, x, r, __task=_task: sample_t1_goal(__task, x, r),
                    )
                )
                episode_index += 10
                blocks[task] = result
                print(
                    f"{task}: episodes={result['n_episodes']} success={result['success_rate']:.3f} "
                    f"return={result['mean_return']:.3f} final_d={result['mean_final_d']:.4f} "
                    f"wall_s={time.perf_counter() - start:.0f} per_template={result['per_template_success']}"
                )

            val_pairs = load_t2_split("val")
            per_goal = int(ref_cfg.get("t2_val_episodes_per_goal", 2))
            goals = [goal for _, goal in val_pairs for _ in range(per_goal)]
            labels = [spec["goal_id"] for spec, _ in val_pairs for _ in range(per_goal)]
            start = time.perf_counter()
            result = run_block(
                lambda active_runner: evaluate_episodes(
                    active_runner,
                    n_episodes=len(goals),
                    seed=int(args.seed) + 200,
                    episode_index_start=episode_index,
                    action_fn=random_action_fn,
                    rng=rng,
                    goals=goals,
                    goal_labels=labels,
                )
            )
            blocks["t2_val"] = result
            print(
                f"t2_val: episodes={result['n_episodes']} success={result['success_rate']:.3f} "
                f"return={result['mean_return']:.3f} final_d={result['mean_final_d']:.4f} "
                f"wall_s={time.perf_counter() - start:.0f}"
            )

            runner = state["runner"]
            payload = {
                "generated_at": utc_now(),
                "git_commit": get_git_commit_hash(),
                "seed": int(args.seed),
                "config": config,
                "protocol": {
                    "horizon": runner.config.horizon,
                    "settle_max_steps": runner.config.settle_max_steps,
                    "vel_threshold": runner.config.vel_threshold,
                    "reward": vars(runner.config.reward),
                },
                "blocks": blocks,
                "nan_incidents_last_scene": runner.nan_incidents,
                "full_scene_rebuilds": state["rebuilds"],
            }
            json_path = Path(outputs.get("json", "outputs/metrics/p1_random_reference.json"))
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
            print(f"nan_incidents_last_scene={runner.nan_incidents} rebuilds={state['rebuilds']}")
            print(f"wrote {json_path}")
            return 0
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    raise SystemExit(main())
