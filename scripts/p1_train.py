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
from dgcc.models.networks import goal_residual_flips
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

# Env-stability operational limits (M3R gate verdict gate-m3r-reconvene-20260710,
# choice D follow-ups — env/driver layer only; training code, hyperparameters,
# reward constants and covenant thresholds unchanged):
#   (a) a discard storm exceeding DISCARD_STORM_REBUILD_AFTER *consecutive*
#       discarded rounds escalates to a forced full scene rebuild (livelock
#       exit — m3r_t1a_s2 stalled 10.5 h at 272 discards with rebuild=0),
#   (b) the full-rebuild limit rises 5 -> 8 and the freshest agent state is
#       checkpointed before a rebuild-limit crash (m3r_t1a_s1 lost ~12k
#       transitions of progress past its last periodic checkpoint).
MAX_FULL_REBUILDS = 8
DISCARD_STORM_REBUILD_AFTER = 10


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
        self.max_full_rebuilds = int(run_cfg.get("max_full_rebuilds", MAX_FULL_REBUILDS))
        self.discard_storm_rebuild_after = int(
            run_cfg.get("discard_storm_rebuild_after", DISCARD_STORM_REBUILD_AFTER)
        )
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
        self.agent = TD3Agent(
            self.agent_config,
            device=self.device,
            reward_constants=self.episode_config.reward,
        )
        self.buffer = ReplayBuffer(self.agent_config.replay_capacity)
        self.diag = DiagnosticsLogger(self.run_tag)
        self.rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)

        self.models_dir = Path("outputs/models") / self.run_tag
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.transitions = 0
        self.episode_index = 0
        self.full_rebuilds = 0
        self._consecutive_discards = 0
        self.last_checkpoint: Path | None = None
        self.best_success = -1.0
        self.eval_history: list[dict[str, Any]] = []
        self.halt_reason: str | None = None
        self._prev_goal_flip = np.full(self.n_envs, -1, dtype=np.int8)
        self._episode_flip_transitions = np.zeros(self.n_envs, dtype=int)
        self._episode_flip_observations = np.zeros(self.n_envs, dtype=int)

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
        self._reset_flip_tracking()

    def refresh_goal_curves(self) -> None:
        self.goal_curves = np.stack(
            [goal_curve(g, P1_LENGTH_M) for g in self.runner.goals]
        )

    def _reset_flip_tracking(
        self,
        env_indices: np.ndarray | list[int] | None = None,
        *,
        prev_flip: np.ndarray | None = None,
        flip_transitions: np.ndarray | None = None,
        flip_observations: np.ndarray | None = None,
    ) -> None:
        prev = self._prev_goal_flip if prev_flip is None else prev_flip
        transitions = (
            self._episode_flip_transitions if flip_transitions is None else flip_transitions
        )
        observations = (
            self._episode_flip_observations if flip_observations is None else flip_observations
        )
        if env_indices is None:
            prev.fill(-1)
            transitions.fill(0)
            observations.fill(0)
            return
        indices = np.asarray(env_indices, dtype=int).reshape(-1)
        if indices.size == 0:
            return
        prev[indices] = -1
        transitions[indices] = 0
        observations[indices] = 0

    def _log_lift_flip_diagnostics(
        self,
        transitions: int,
        *,
        X_before: np.ndarray,
        goal_curves: np.ndarray,
        lift: list[str],
        active: np.ndarray,
        templates: list[str],
        phase: str,
        done: np.ndarray | None = None,
        prev_flip: np.ndarray | None = None,
        flip_transitions: np.ndarray | None = None,
        flip_observations: np.ndarray | None = None,
    ) -> np.ndarray:
        active_arr = np.asarray(active, dtype=bool)
        templates_arr = np.asarray(templates, dtype=object)
        self.diag.log_lift_dist(
            transitions,
            templates=templates_arr,
            lift=lift,
            active=active_arr,
            phase=phase,
        )

        flips = goal_residual_flips(X_before, goal_curves, P1_LENGTH_M).astype(np.int8)
        prev = self._prev_goal_flip if prev_flip is None else prev_flip
        episode_flips = (
            self._episode_flip_transitions if flip_transitions is None else flip_transitions
        )
        observations = (
            self._episode_flip_observations if flip_observations is None else flip_observations
        )

        tracked = active_arr & (prev >= 0)
        changed = tracked & (prev != flips)
        observations[active_arr] += 1
        episode_flips[changed] += 1
        prev[active_arr] = flips[active_arr]

        done_arr = np.zeros_like(active_arr, dtype=bool) if done is None else np.asarray(done, dtype=bool)
        rows: list[dict[str, float | int | str]] = []
        for template in sorted({str(value) for value in templates_arr[active_arr]}):
            template_mask = active_arr & (templates_arr == template)
            n_active = int(template_mask.sum())
            n_tracked = int((tracked & template_mask).sum())
            flip_count = int((changed & template_mask).sum())
            completed = template_mask & done_arr
            rates: list[float] = []
            for env_idx in np.flatnonzero(completed):
                denominator = max(1, int(observations[int(env_idx)]) - 1)
                rates.append(float(episode_flips[int(env_idx)] / denominator))
            rows.append(
                {
                    "template": template,
                    "flip_transitions": flip_count,
                    "n_active": n_active,
                    "n_tracked": n_tracked,
                    "active_transition_rate": float(flip_count / n_tracked)
                    if n_tracked
                    else float("nan"),
                    "completed_episodes": int(completed.sum()),
                    "episode_flicker_rate_mean": float(np.mean(rates))
                    if rates
                    else float("nan"),
                }
            )
        if rows:
            self.diag.log_flip_flicker(transitions, rows, phase=phase)
        if done is not None:
            self._reset_flip_tracking(
                np.flatnonzero(active_arr & done_arr),
                prev_flip=prev,
                flip_transitions=episode_flips,
                flip_observations=observations,
            )
        return flips
    # ------------------------------------------------------------------

    def _register_rebuild(self, *, context: str, error: object) -> bool:
        """Count a full-scene rebuild escalation.

        Returns True when the rebuild limit (verdict (b): 8) is exceeded; in
        that case the freshest agent state has already been checkpointed and
        the caller must raise (lane contract: non-halt crash, exit=1).
        """

        self.full_rebuilds += 1
        print(
            f"{context} rebuild={self.full_rebuilds} error={error} "
            f"action=full_scene_rebuild transitions={self.transitions}"
        )
        if self.full_rebuilds > self.max_full_rebuilds:
            self._preserve_crash_checkpoint()
            return True
        return False

    def _preserve_crash_checkpoint(self) -> None:
        """Preserve the latest agent state before a rebuild-limit crash (verdict (b))."""

        try:
            ckpt = self.agent.save_checkpoint(
                self.models_dir / f"ckpt_crash_{self.transitions:07d}.pt"
            )
            self.last_checkpoint = ckpt
            print(f"crash checkpoint preserved: {ckpt}")
        except Exception as exc:  # keep the original crash path alive
            print(f"crash checkpoint preservation failed: {exc}")
        self.diag.save_history()
        self.save_run_summary()
    # ------------------------------------------------------------------

    def collect_round(self) -> int:
        """One batched primitive + buffer insertion. Returns active count."""

        assert self.runner is not None and self.env is not None and self.goal_curves is not None

        X = self.env.get_centerline_batch()
        goal_curves_before = self.goal_curves.copy()
        templates_before = list(self.runner.init_shapes)
        p, delta, lift, info = self.agent.select_actions(
            X,
            goal_curves_before,
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
            if self._register_rebuild(context="round_recovery", error=exc):
                raise
            self._consecutive_discards = 0
            self._reset_flip_tracking()
            self.build_scene()
            return 0
        if record.get("discarded"):
            bad_envs = record.get("bad_envs", np.flatnonzero(record["active"]))
            self._reset_flip_tracking(np.asarray(bad_envs, dtype=int))
            self.diag.log_nan_incidents(
                self.transitions,
                self.runner.nan_incidents,
                self.runner.magnitude_incidents,
            )
            self._consecutive_discards += 1
            print(
                f"transition batch discarded (NaN covenant): {record['reason']} "
                f"consecutive={self._consecutive_discards}"
            )
            if self._consecutive_discards > self.discard_storm_rebuild_after:
                storm = self._consecutive_discards
                self._consecutive_discards = 0
                if self._register_rebuild(
                    context="discard_storm_escalation",
                    error=f"consecutive_discarded_rounds={storm}",
                ):
                    raise FloatingPointError(
                        f"rebuild limit ({self.max_full_rebuilds}) exceeded during "
                        f"discard-storm escalation (consecutive discarded rounds={storm})"
                    )
                self._reset_flip_tracking()
                self.build_scene()
                return 0
            self.refresh_goal_curves()
            return 0

        active = record["active"]
        self._consecutive_discards = 0
        count = int(active.sum())
        next_transitions = self.transitions + count
        if count:
            self._log_lift_flip_diagnostics(
                next_transitions,
                X_before=X,
                goal_curves=goal_curves_before,
                lift=lift,
                active=active,
                templates=templates_before,
                phase="collect",
                done=record["done"],
            )
            refresh_reset = np.flatnonzero(
                active & ~record["done"] & (self.runner.t < record["t"])
            )
            if refresh_reset.size:
                self._reset_flip_tracking(refresh_reset)
            self.diag.log_step_d(next_transitions, record["d_after"][active], phase="collect")
            self.buffer.add_batch(
                X_before=record["X_before"][active],
                X_after=record["X_after"][active],
                goal_curve=goal_curves_before[active],
                p=np.asarray(p)[active],
                delta=np.asarray(delta)[active],
                lift=np.asarray([1 if v == "high" else 0 for v in lift])[active],
                reward=record["reward"][active],
                done=record["done"][active],
                truncated=record["truncated"][active],
            )
            self.transitions = next_transitions
        self.diag.log_nan_incidents(
            self.transitions,
            self.runner.nan_incidents,
            self.runner.magnitude_incidents,
        )
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
        self.diag.log_nan_incidents(
            self.transitions,
            self.runner.nan_incidents,
            self.runner.magnitude_incidents,
        )

    # ------------------------------------------------------------------

    def deterministic_eval(self) -> dict[str, Any]:
        assert self.runner is not None

        eval_prev_flip = np.full(self.n_envs, -1, dtype=np.int8)
        eval_flip_transitions = np.zeros(self.n_envs, dtype=int)
        eval_flip_observations = np.zeros(self.n_envs, dtype=int)
        magnitude_before = self.runner.magnitude_incidents
        eval_incidents_seen = {
            "nan": self.runner.nan_incidents,
            "magnitude": self.runner.magnitude_incidents,
        }

        def eval_action_fn(X: np.ndarray, G: np.ndarray, _rng: np.random.Generator):
            p, delta, lift = self.agent.select_actions(
                X,
                G,
                step=self.transitions,
                total_budget=self.total,
                rng=_rng,
                deterministic=True,
            )
            assert self.runner is not None
            if (
                self.runner.nan_incidents != eval_incidents_seen["nan"]
                or self.runner.magnitude_incidents != eval_incidents_seen["magnitude"]
            ):
                eval_incidents_seen["nan"] = self.runner.nan_incidents
                eval_incidents_seen["magnitude"] = self.runner.magnitude_incidents
                self._reset_flip_tracking(
                    prev_flip=eval_prev_flip,
                    flip_transitions=eval_flip_transitions,
                    flip_observations=eval_flip_observations,
                )
            if np.all(self.runner.t == 0) and not np.any(self.runner.done):
                self._reset_flip_tracking(
                    prev_flip=eval_prev_flip,
                    flip_transitions=eval_flip_transitions,
                    flip_observations=eval_flip_observations,
                )
            self._log_lift_flip_diagnostics(
                self.transitions,
                X_before=X,
                goal_curves=G,
                lift=lift,
                active=~self.runner.done,
                templates=list(self.runner.init_shapes),
                phase="eval",
                prev_flip=eval_prev_flip,
                flip_transitions=eval_flip_transitions,
                flip_observations=eval_flip_observations,
            )
            return p, delta, lift

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
        result["magnitude_incidents_during_eval"] = (
            self.runner.magnitude_incidents - magnitude_before
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
                if self._register_rebuild(context="eval_recovery", error=exc):
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
            "td_target_bound": dict(self.agent.td_target_bound),
            "total_budget": self.total,
            "transitions": self.transitions,
            "updates": self.agent.update_count,
            "nan_incidents_env": self.runner.nan_incidents if self.runner else None,
            "magnitude_incidents_env": self.runner.magnitude_incidents if self.runner else None,
            "full_scene_rebuilds": self.full_rebuilds,
            "max_full_rebuilds": self.max_full_rebuilds,
            "discard_storm_rebuild_after": self.discard_storm_rebuild_after,
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
        print(
            f"td3={self.agent_config.to_dict()} reward={vars(self.episode_config.reward)} "
            f"td_target_bound={self.agent.td_target_bound}"
        )
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
                        f"nan_env={self.runner.nan_incidents} "
                        f"mag={self.runner.magnitude_incidents} rebuilds={self.full_rebuilds}"
                    )
                if self.transitions >= next_eval:
                    self.eval_and_checkpoint(final=self.transitions >= self.total)
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

        if not self.eval_history or self.eval_history[-1]["transitions"] < self.transitions:
            self.eval_and_checkpoint(final=True)
        else:
            self.diag.maybe_plot(self.transitions, force=True)
            self.diag.save_history()
            self.save_run_summary()
        wall_h = (time.perf_counter() - start_wall) / 3600
        print(
            f"run complete transitions={self.transitions} updates={self.agent.update_count} "
            f"wall_h={wall_h:.2f} nan_env={self.runner.nan_incidents} "
            f"mag={self.runner.magnitude_incidents} rebuilds={self.full_rebuilds}"
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
