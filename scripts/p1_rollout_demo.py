"""P1-M0 random-policy episode rollout demo (10 episodes, batched).

Runs one full T=10 episode in each of 10 parallel environments — three T1
episodes (t1a/t1b/t1c goals sampled from the settled init state) and seven T2
train-split goals — under the P1 episode protocol (success early
termination, immutable settle budget, NaN covenant).  Logs per-step D,
reward, and termination to ``outputs/reports/p1_rollout_demo.log`` and writes
``outputs/metrics/p1_rollout_demo.json``.
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
from dgcc.goals.dual_goal import DualGoal
from dgcc.tasks.domain import P1_N_SEGMENTS, SETTLE_MAX_STEPS, p1_rope_params
from dgcc.tasks.episode import BatchedEpisodeRunner, random_policy_actions
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
    parser = argparse.ArgumentParser(description="P1-M0 random-policy rollout demo")
    parser.add_argument("--config", type=Path, default=Path("configs/p1_rollout_demo.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    outputs = config.get("outputs", {})
    n_episodes = int(config.get("demo", {}).get("n_episodes", 10))
    actions_cfg = config.get("actions", {})

    log_path = Path(outputs.get("stdout_log", "outputs/reports/p1_rollout_demo.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    original_stdout = sys.stdout
    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)  # type: ignore[assignment]
        try:
            print(f"p1_rollout_demo start {utc_now()} seed={args.seed} episodes={n_episodes}")
            params = p1_rope_params()

            t2_goals = [goal for _, goal in load_t2_split("train")]
            t1_tasks = ("t1a_straighten", "t1b_single_bend", "t1c_endpoint_reposition")
            assignments = [
                t1_tasks[i] if i < len(t1_tasks) else f"t2_train[{i - len(t1_tasks)}]"
                for i in range(n_episodes)
            ]

            def goal_fn(env_idx: int, x_current: np.ndarray, rng: np.random.Generator) -> DualGoal:
                if env_idx < len(t1_tasks):
                    return sample_t1_goal(t1_tasks[env_idx], x_current, rng)
                return t2_goals[env_idx - len(t1_tasks)]

            build_start = time.perf_counter()
            env = DLOLabEnv(**env_kwargs(config, n_episodes))
            env.reset(params, init_shape="straight", seed=int(args.seed))
            runner = BatchedEpisodeRunner(env, params)
            begin_info = runner.begin_episodes(seed=int(args.seed) + 1, goal_fn=goal_fn)
            print(f"build+reset wall {time.perf_counter() - build_start:.1f}s")
            print(f"assignments: {assignments}")
            print(f"init shapes:  {begin_info['init_shapes']}")
            print("d_initial:    " + " ".join(f"{d:.4f}" for d in begin_info["d_initial"]))

            rng = np.random.default_rng(int(args.seed) + 2)
            episode_returns = np.zeros(n_episodes, dtype=float)
            steps_log: list[dict[str, Any]] = []
            step_idx = 0
            unrecoverable_error: str | None = None
            while not runner.all_done() and step_idx < 2 * runner.config.horizon:
                p, deltas, lifts = random_policy_actions(
                    rng,
                    n_envs=n_episodes,
                    n_vertices=P1_N_SEGMENTS,
                    delta_min_m=float(actions_cfg.get("delta_min_m", 0.02)),
                    delta_max_m=float(actions_cfg.get("delta_max_m", 0.15)),
                    lift_choices=tuple(actions_cfg.get("lift_choices", ("low", "high"))),
                )
                try:
                    record = runner.step(p, deltas, lifts, rng=rng)
                except FloatingPointError as exc:
                    # Unrecoverable non-finite state (runner already retried
                    # reseeding): preserve evidence honestly and stop.
                    step_idx += 1
                    print(f"step {step_idx}: UNRECOVERABLE NaN — {exc}; aborting with partial evidence")
                    steps_log.append({"step": step_idx, "unrecoverable_nan": str(exc)})
                    unrecoverable_error = str(exc)
                    break
                step_idx += 1
                if record.get("discarded"):
                    print(f"step {step_idx}: NaN covenant fired — bad_envs={record['bad_envs'].tolist()}")
                    steps_log.append({"step": step_idx, "discarded": True, "bad_envs": record["bad_envs"].tolist()})
                    continue
                active = record["active"]
                episode_returns += np.where(active, record["reward"], 0.0)
                print(
                    f"step {step_idx}: active={int(active.sum())} "
                    f"d_after=[{' '.join(f'{d:.4f}' for d in record['d_after'])}] "
                    f"reward=[{' '.join(f'{r:+.3f}' for r in record['reward'])}] "
                    f"success={record['success'].astype(int).tolist()} "
                    f"done={record['done'].astype(int).tolist()}"
                )
                steps_log.append(
                    {
                        "step": step_idx,
                        "discarded": False,
                        "active": active.tolist(),
                        "d_before": record["d_before"].tolist(),
                        "d_after": record["d_after"].tolist(),
                        "reward": record["reward"].tolist(),
                        "success": record["success"].tolist(),
                        "done": record["done"].tolist(),
                        "grasp_success": record["grasp_success"].tolist(),
                        "settle_steps": record["settle_steps"].tolist(),
                        "settle_converged": record["settle_converged"].tolist(),
                    }
                )

            print("episodes complete")
            print(f"final t:        {runner.t.tolist()}")
            print(f"succeeded:      {runner.succeeded.astype(int).tolist()}")
            print("final D:        " + " ".join(f"{d:.4f}" for d in runner.d_current))
            print("episode return: " + " ".join(f"{r:+.3f}" for r in episode_returns))
            print(f"nan_incidents:  {runner.nan_incidents}")

            payload = {
                "generated_at": utc_now(),
                "git_commit": get_git_commit_hash(),
                "seed": int(args.seed),
                "config": config,
                "assignments": assignments,
                "init_shapes": begin_info["init_shapes"],
                "d_initial": begin_info["d_initial"].tolist(),
                "steps": steps_log,
                "final": {
                    "t": runner.t.tolist(),
                    "succeeded": runner.succeeded.tolist(),
                    "d_final": runner.d_current.tolist(),
                    "episode_return": episode_returns.tolist(),
                    "nan_incidents": runner.nan_incidents,
                    "unrecoverable_error": unrecoverable_error,
                },
            }
            json_path = Path(outputs.get("json", "outputs/metrics/p1_rollout_demo.json"))
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {json_path}")
            return 0 if unrecoverable_error is None else 1
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    raise SystemExit(main())
