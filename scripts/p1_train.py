"""P1 training driver (M2 smoke, M3 T1 runs, M4 T2 runs).

Implements the §7 protocol on DLOLabEnv: batched collection through the
episode layer (immutable settle budget), UTD=1 updates, deterministic eval
every 25k transitions (T1: 100 episodes; T2: 50 validation goals x 2),
checkpoints every 25k plus best-on-eval, §8 diagnostics with 25k auto-plots,
env-level NaN covenant with P0-pattern full scene rebuild escalation, and the
training-level NaN halt (preserve last checkpoint + factual report).

Usage:
    uv run python scripts/p1_train.py --config configs/p1_t1_a.yaml --seed 0 \
        --run-tag t1a_smoke_s0 --total-override 50000
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
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dgcc.envs.dlolab import DLOLabEnv
from dgcc.goals.dual_goal import goal_curve
from dgcc.rl.diagnostics import DiagnosticsLogger
from dgcc.rl.evaluation import evaluate_episodes
from dgcc.rl.replay import ReplayBuffer
from dgcc.rl.td3 import TD3Agent, TD3Config, TrainingNaNError
from dgcc.tasks.domain import (
    P1_LENGTH_M,
    P1_N_SEGMENTS,
    RewardConstants,
    SETTLE_MAX_STEPS,
    p1_rope_params,
)
from dgcc.tasks.episode import BatchedEpisodeRunner, EpisodeConfig, is_nonfinite_error
from dgcc.tasks.t1 import sample_t1_goal
from dgcc.tasks.t2 import load_t2_split
from dgcc.utils.meta import get_git_commit_hash

T1_TASKS = ("t1a_straighten", "t1b_single_bend", "t1c_endpoint_reposition")


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


class TrainingRun:
    def __init__(self, args: argparse.Namespace) -> None:
        self.config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        self.seed = int(args.seed)
        self.task = str(self.config["task"])
        self.run_tag = args.run_tag or f"{self.task}_s{self.seed}"
        run_cfg = self.config.get("run", {})
        self.total = int(args.total_override or run_cfg.get("total_transitions", 100_000))
        self.n_envs = int(run_cfg.get("n_envs", 256))
        self.eval_every = int(run_cfg.get("eval_every_transitions", 25_000))
        self.device = args.device
        self.params = p1_rope_params()

        reward_cfg = self.config.get("reward", {})
        self.episode_config = EpisodeConfig(
            reward=RewardConstants(
                alpha=float(reward_cfg.get("alpha", 10.0)),
                c_step=float(reward_cfg.get("c_step", 0.1)),
                r_succ=float(reward_cfg.get("r_succ", 5.0)),
            )
        )
        td3_cfg = dict(self.config.get("td3", {}))
        self.agent_config = TD3Config(**{k: v for k, v in td3_cfg.items()})
        self.agent = TD3Agent(self.agent_config, device=self.device)
        self.buffer = ReplayBuffer(self.agent_config.replay_capacity)
        self.diag = DiagnosticsLogger(self.run_tag)
        self.rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)

        self.models_dir = Path("outputs/models") / self.run_tag
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.transitions = 0
        self.episode_index = 0
        self.full_rebuilds = 0
        self.last_checkpoint: Path | None = None
        self.best_success = -1.0
        self.eval_history: list[dict[str, Any]] = []
        self.halt_reason: str | None = None

        if self.task == "t2":
            self.train_goals = [g for _, g in load_t2_split("train")]
            val_pairs = load_t2_split("val")
            per_goal = int(self.config.get("eval", {}).get("t2_episodes_per_goal", 2))
            self.val_goals = [g for _, g in val_pairs for _ in range(per_goal)]
            self.val_labels = [s["goal_id"] for s, _ in val_pairs for _ in range(per_goal)]
        elif self.task in T1_TASKS:
            self.train_goals = None
            self.val_goals = None
        else:
            raise ValueError(f"unknown task {self.task!r}")

        self.env: DLOLabEnv | None = None
        self.runner: BatchedEpisodeRunner | None = None
        self.goal_curves: np.ndarray | None = None

    # ------------------------------------------------------------------

    def build_scene(self) -> None:
        if self.env is not None:
            self.runner = None
            self.env = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.env = DLOLabEnv(**env_kwargs(self.config, self.n_envs))
        self.env.reset(
            self.params,
            init_shape="straight",
            seed=self.seed + 10_000 * (self.full_rebuilds + 1),
        )
        if not self.env.supports_per_env_grasp():
            raise RuntimeError("per-env grasp hooks unavailable")
        self.runner = BatchedEpisodeRunner(self.env, self.params, self.episode_config)
        self.begin_training_episodes()

    def _goal_fn(self, env_idx: int, x: np.ndarray, goal_rng: np.random.Generator):
        return sample_t1_goal(self.task, x, goal_rng)

    def begin_training_episodes(self) -> None:
        assert self.runner is not None
        self.episode_index += 1
        if self.task == "t2":
            self.runner.begin_episodes(
                seed=self.seed,
                episode_index=self.episode_index,
                goal_pool=self.train_goals,
                auto_reset=True,
            )
        else:
            self.runner.begin_episodes(
                seed=self.seed,
                episode_index=self.episode_index,
                goal_fn=self._goal_fn,
                auto_reset=True,
            )
        self.refresh_goal_curves()

    def refresh_goal_curves(self) -> None:
        self.goal_curves = np.stack(
            [goal_curve(g, P1_LENGTH_M) for g in self.runner.goals]
        )

    # ------------------------------------------------------------------

    def collect_round(self) -> int:
        """One batched primitive + buffer insertion. Returns active count."""

        assert self.runner is not None and self.env is not None

        X = self.env.get_centerline_batch()
        p, delta, lift, info = self.agent.select_actions(
            X,
            self.goal_curves,
            step=self.transitions,
            total_budget=self.total,
            rng=self.rng,
            return_info=True,
        )
        self.diag.log_action_info(self.transitions, info["q1_candidates"])

        try:
            record = self.runner.step(p, delta, lift, rng=self.rng)
        except (FloatingPointError, ValueError, RuntimeError) as exc:
            if not is_nonfinite_error(exc):
                raise
            self.full_rebuilds += 1
            print(
                f"round_recovery rebuild={self.full_rebuilds} error={exc} "
                f"action=full_scene_rebuild transitions={self.transitions}"
            )
            if self.full_rebuilds > 5:
                raise
            self.build_scene()
            return 0
        if record.get("discarded"):
            print(f"transition batch discarded (NaN covenant): {record['reason']}")
            return 0

        active = record["active"]
        count = int(active.sum())
        if count:
            self.buffer.add_batch(
                X_before=record["X_before"][active],
                X_after=record["X_after"][active],
                goal_curve=self.goal_curves[active],
                p=np.asarray(p)[active],
                delta=np.asarray(delta)[active],
                lift=np.asarray([1 if v == "high" else 0 for v in lift])[active],
                reward=record["reward"][active],
                done=record["done"][active],
            )
            self.transitions += count
        # Auto-reset may have refreshed goals for finished envs.
        self.refresh_goal_curves()
        return count

    def train_updates(self, n_updates: int) -> None:
        if self.buffer.size < self.agent_config.warmup_transitions:
            return
        for i in range(n_updates * self.agent_config.utd):
            stats = self.agent.update(self.buffer.sample(self.agent_config.batch_size, self.rng))
            if i % 32 == 0:
                self.diag.log_update(self.transitions, stats)
        self.diag.log_replay(
            self.transitions,
            size=self.buffer.size,
            reward_mean=float(self.buffer.reward[: self.buffer.size].mean()),
            done_frac=float(self.buffer.done[: self.buffer.size].mean()),
        )
        self.diag.log_nan_incidents(self.transitions, self.runner.nan_incidents)

    # ------------------------------------------------------------------

    def deterministic_eval(self) -> dict[str, Any]:
        assert self.runner is not None

        def eval_action_fn(X: np.ndarray, G: np.ndarray, _rng: np.random.Generator):
            return self.agent.select_actions(
                X, G, step=self.transitions, total_budget=self.total,
                rng=_rng, deterministic=True,
            )

        eval_cfg = self.config.get("eval", {})
        if self.task == "t2":
            result = evaluate_episodes(
                self.runner,
                n_episodes=len(self.val_goals),
                seed=self.seed + 500,
                episode_index_start=90_000 + self.episode_index,
                action_fn=eval_action_fn,
                rng=np.random.default_rng(self.seed + 501),
                gamma=self.agent_config.gamma,
                goals=self.val_goals,
                goal_labels=self.val_labels,
                q_min_fn=self.agent.q_min_executed,
            )
        else:
            result = evaluate_episodes(
                self.runner,
                n_episodes=int(eval_cfg.get("t1_episodes_per_task", 100)),
                seed=self.seed + 500,
                episode_index_start=90_000 + self.episode_index,
                action_fn=eval_action_fn,
                rng=np.random.default_rng(self.seed + 501),
                gamma=self.agent_config.gamma,
                goal_fn=self._goal_fn,
                q_min_fn=self.agent.q_min_executed,
            )
        return result

    def eval_and_checkpoint(self, *, final: bool = False) -> None:
        start = time.perf_counter()
        while True:
            try:
                result = self.deterministic_eval()
                break
            except (FloatingPointError, ValueError, RuntimeError) as exc:
                if not is_nonfinite_error(exc):
                    raise
                self.full_rebuilds += 1
                print(
                    f"eval_recovery rebuild={self.full_rebuilds} error={exc} "
                    "action=full_scene_rebuild (eval restarted)"
                )
                if self.full_rebuilds > 5:
                    raise
                self.build_scene()
        result["wall_s"] = time.perf_counter() - start
        result["transitions"] = self.transitions
        self.eval_history.append(result)
        summary = {k: v for k, v in result.items() if k != "episodes"}
        self.diag.log_eval(self.transitions, summary)
        print(
            f"eval transitions={self.transitions} success={result['success_rate']:.3f} "
            f"return={result['mean_return']:.3f} final_d={result['mean_final_d']:.4f} "
            f"overest_gap={result['overestimation_gap_mean']} wall_s={result['wall_s']:.0f}"
        )

        ckpt = self.agent.save_checkpoint(self.models_dir / f"ckpt_{self.transitions:07d}.pt")
        self.last_checkpoint = ckpt
        if result["success_rate"] > self.best_success:
            self.best_success = result["success_rate"]
            self.agent.save_checkpoint(self.models_dir / "best.pt")
            print(f"new best checkpoint at {self.transitions} (success={self.best_success:.3f})")
        self.diag.maybe_plot(self.transitions, force=final)
        self.diag.save_history()
        self.save_run_summary()
        # Eval consumed the episode batch; restart training episodes.
        self.begin_training_episodes()

    def save_run_summary(self) -> None:
        payload = {
            "generated_at": utc_now(),
            "git_commit": get_git_commit_hash(),
            "run_tag": self.run_tag,
            "task": self.task,
            "seed": self.seed,
            "config": self.config,
            "td3_config": self.agent_config.to_dict(),
            "reward_constants": vars(self.episode_config.reward),
            "total_budget": self.total,
            "transitions": self.transitions,
            "updates": self.agent.update_count,
            "nan_incidents_env": self.runner.nan_incidents if self.runner else None,
            "full_scene_rebuilds": self.full_rebuilds,
            "halt_reason": self.halt_reason,
            "best_success": self.best_success,
            "last_checkpoint": str(self.last_checkpoint) if self.last_checkpoint else None,
            "evals": [
                {k: v for k, v in ev.items() if k != "episodes"} for ev in self.eval_history
            ],
            "eval_episodes": [
                {"transitions": ev["transitions"], "episodes": ev["episodes"]}
                for ev in self.eval_history
            ],
        }
        path = Path("outputs/metrics") / f"p1_run_{self.run_tag}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------

    def run(self) -> int:
        print(
            f"p1_train start {utc_now()} run_tag={self.run_tag} task={self.task} "
            f"seed={self.seed} total={self.total} n_envs={self.n_envs} device={self.device}"
        )
        print(f"td3={self.agent_config.to_dict()}")
        print(f"reward={vars(self.episode_config.reward)}")
        start_wall = time.perf_counter()
        self.build_scene()
        next_eval = self.eval_every

        try:
            while self.transitions < self.total:
                round_start = time.perf_counter()
                count = self.collect_round()
                collect_s = time.perf_counter() - round_start
                update_start = time.perf_counter()
                self.train_updates(count)
                update_s = time.perf_counter() - update_start
                if count and (self.transitions // count) % 20 == 0:
                    elapsed = time.perf_counter() - start_wall
                    rate = self.transitions / elapsed if elapsed > 0 else 0.0
                    eta_h = (self.total - self.transitions) / rate / 3600 if rate > 0 else 0.0
                    print(
                        f"round transitions={self.transitions}/{self.total} "
                        f"collect_s={collect_s:.1f} update_s={update_s:.1f} "
                        f"rate={rate:.2f}tr/s eta_h={eta_h:.2f} "
                        f"nan_env={self.runner.nan_incidents} rebuilds={self.full_rebuilds}"
                    )
                if self.transitions >= next_eval:
                    self.eval_and_checkpoint()
                    next_eval += self.eval_every
        except TrainingNaNError as exc:
            # Global rule 6, training level: halt + preserve last checkpoint +
            # factual report. No silent continuation.
            self.halt_reason = f"TrainingNaNError: {exc}"
            print(f"TRAINING HALT (rule 6): {self.halt_reason}")
            print(f"last checkpoint preserved: {self.last_checkpoint}")
            self.diag.save_history()
            self.save_run_summary()
            return 2

        self.eval_and_checkpoint(final=True)
        wall_h = (time.perf_counter() - start_wall) / 3600
        print(
            f"run complete transitions={self.transitions} updates={self.agent.update_count} "
            f"wall_h={wall_h:.2f} nan_env={self.runner.nan_incidents} rebuilds={self.full_rebuilds}"
        )
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="P1 training driver")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--total-override", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_tag = args.run_tag or f"{yaml.safe_load(Path(args.config).read_text())['task']}_s{args.seed}"
    log_path = Path("outputs/reports") / f"p1_train_{run_tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)  # type: ignore[assignment]
        try:
            return TrainingRun(args).run()
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    raise SystemExit(main())
