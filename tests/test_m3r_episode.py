"""P1-M3R episode/evaluation covenant tests (CPU fake env only)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from dgcc.goals.dual_goal import DualGoal, goal_curve
from dgcc.rl.evaluation import evaluate_episodes
from dgcc.tasks.domain import p1_rope_params
from dgcc.tasks.episode import BatchedEpisodeRunner, EpisodeConfig
from dgcc.tasks.reward import distance_to_goal


LENGTH = 1.0


def straight_line(offset_x: float = 0.0, n: int = 32) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    return np.column_stack((t + offset_x, np.zeros(n), np.zeros(n)))


def constant_curve(x: float, n: int = 32) -> np.ndarray:
    return np.tile(np.array([x, 0.0, 0.0], dtype=float), (n, 1))


def straight_goal(anchor_x: float = 0.5) -> DualGoal:
    return DualGoal(
        shape_template=straight_line(),
        anchor=np.array([anchor_x, 0.0, 0.0], dtype=float),
        anchor_mode="centroid",
        template_name="m3r_test_straight",
    )


def stack_curves(*curves: np.ndarray) -> np.ndarray:
    return np.stack([np.asarray(curve, dtype=float) for curve in curves])


def zero_actions(n_envs: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    return (
        np.zeros(n_envs, dtype=int),
        np.zeros((n_envs, 3), dtype=float),
        ["low"] * n_envs,
    )


class ScriptedBatchEnv:
    """Duck-typed batch env with queued post-step states."""

    def __init__(self, n_envs: int) -> None:
        self.n_envs = int(n_envs)
        self.state: np.ndarray | None = None
        self.step_states: list[np.ndarray] = []
        self.settle_calls: list[dict[str, Any]] = []
        self.magnitude_reset_env_once: int | None = None

    def queue_step(self, *curves: np.ndarray) -> None:
        assert len(curves) == self.n_envs
        self.step_states.append(stack_curves(*curves))

    def light_reset(
        self,
        vertices: np.ndarray,
        *,
        vel_threshold: float,
        max_steps: int,
        reinit_env_indices: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        self.settle_calls.append(
            {
                "method": "light_reset",
                "vel_threshold": vel_threshold,
                "max_steps": max_steps,
                "reinit_env_indices": None
                if reinit_env_indices is None
                else np.asarray(reinit_env_indices, dtype=int).copy(),
            }
        )
        verts = np.asarray(vertices, dtype=float)
        if verts.ndim == 2:
            verts = np.broadcast_to(verts, (self.n_envs, *verts.shape)).copy()
        self.state = verts.copy()
        if self.magnitude_reset_env_once is not None:
            self.state[int(self.magnitude_reset_env_once)] = constant_curve(4.0)
            self.magnitude_reset_env_once = None
        return {
            "settle_converged": np.ones(self.n_envs, dtype=bool),
            "settle_steps": np.zeros(self.n_envs, dtype=int),
        }

    def get_centerline_batch(self) -> np.ndarray:
        assert self.state is not None
        return self.state.copy()

    def get_centerline_raw_batch(self) -> np.ndarray:
        assert self.state is not None
        return self.state.copy()

    def step_primitive_batch(
        self,
        p: np.ndarray,
        delta: np.ndarray,
        lift: Sequence[str],
        *,
        vel_threshold: float,
        max_steps: int,
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        del p, delta, lift, rng
        assert self.state is not None
        self.settle_calls.append(
            {
                "method": "step_primitive_batch",
                "vel_threshold": vel_threshold,
                "max_steps": max_steps,
            }
        )
        x_before = self.state.copy()
        if self.step_states:
            self.state = self.step_states.pop(0).copy()
        return {
            "X_before": x_before,
            "X_after": self.state.copy(),
            "grasp_success": np.ones(self.n_envs, dtype=bool),
            "settle_steps": np.ones(self.n_envs, dtype=int),
            "info": {"settle_converged": np.ones(self.n_envs, dtype=bool)},
        }


def make_runner(n_envs: int, *, horizon: int = 10) -> tuple[BatchedEpisodeRunner, ScriptedBatchEnv]:
    env = ScriptedBatchEnv(n_envs)
    runner = BatchedEpisodeRunner(env, p1_rope_params(), EpisodeConfig(horizon=horizon))
    return runner, env


def test_truncation_semantics_and_record_snapshots() -> None:
    runner, env = make_runner(3, horizon=3)
    goal = straight_goal()
    goal_curve_ = goal_curve(goal, LENGTH)
    far = straight_line(0.2)
    runner.begin_episodes(seed=1, goals=[goal, goal, goal])

    env.queue_step(goal_curve_, far, far)
    first = runner.step(*zero_actions(3), rng=np.random.default_rng(0))
    assert first["done"].tolist() == [True, False, False]
    assert first["truncated"].tolist() == [False, False, False]
    assert first["t"].tolist() == [1, 1, 1]

    env.queue_step(goal_curve_, far, far)
    second = runner.step(*zero_actions(3), rng=np.random.default_rng(1))
    assert second["done"].tolist() == [True, False, False]
    assert second["truncated"].tolist() == [False, False, False]
    assert second["t"].tolist() == [1, 2, 2]

    env.queue_step(goal_curve_, far, goal_curve_)
    third = runner.step(*zero_actions(3), rng=np.random.default_rng(2))
    assert third["done"].tolist() == [True, True, True]
    assert third["success"].tolist() == [True, False, True]
    assert third["truncated"].tolist() == [False, True, False]
    assert third["t"].tolist() == [1, 3, 3]
    assert runner.truncated.tolist() == [False, True, False]


def test_magnitude_covenant_step_and_settle_paths_are_kind_aware() -> None:
    goal = straight_goal()

    step_runner, step_env = make_runner(1)
    step_runner.begin_episodes(seed=2, goals=[goal])
    step_env.queue_step(constant_curve(4.0))
    step_record = step_runner.step(*zero_actions(1), rng=np.random.default_rng(3))

    assert step_record["discarded"] is True
    assert step_record["bad_envs"].tolist() == [0]
    assert step_runner.magnitude_incidents == 1
    assert step_runner.nan_incidents == 0
    assert step_runner.incident_log[-1]["kind"] == "magnitude"
    assert step_runner.incident_log[-1]["max_coord_norm"] == pytest.approx(4.0)
    assert np.linalg.norm(step_env.get_centerline_batch(), axis=-1).max() <= 3.0

    settle_runner, settle_env = make_runner(1)
    settle_env.magnitude_reset_env_once = 0
    begin_info = settle_runner.begin_episodes(seed=4, goals=[goal])

    assert begin_info["reset_reseeded_envs"] == [0]
    assert settle_runner.magnitude_incidents == 1
    assert settle_runner.nan_incidents == 0
    assert settle_runner.incident_log[-1]["kind"] == "magnitude"
    assert settle_runner.incident_log[-1]["max_coord_norm"] == pytest.approx(4.0)
    assert np.linalg.norm(settle_env.get_centerline_batch(), axis=-1).max() <= 3.0


def test_done_env_d_current_and_d_at_done_freeze_across_drift_and_incident() -> None:
    runner, env = make_runner(2, horizon=5)
    goal = straight_goal()
    goal_curve_ = goal_curve(goal, LENGTH)
    far = straight_line(0.2)
    runner.begin_episodes(seed=5, goals=[goal, goal])

    env.queue_step(goal_curve_, far)
    first = runner.step(*zero_actions(2), rng=np.random.default_rng(6))
    done_d = float(first["d_after"][0])
    assert first["done"].tolist() == [True, False]
    assert runner.d_at_done[0] == pytest.approx(done_d)
    assert runner.d_current[0] == pytest.approx(done_d)

    drifted = straight_line(0.7)
    env.queue_step(drifted, far)
    runner.step(*zero_actions(2), rng=np.random.default_rng(7))
    drifted_d = distance_to_goal(drifted, goal, LENGTH)
    assert drifted_d != pytest.approx(done_d)
    assert runner.d_current[0] == pytest.approx(done_d)
    assert runner.d_at_done[0] == pytest.approx(done_d)

    env.queue_step(straight_line(0.9), constant_curve(4.0))
    incident = runner.step(*zero_actions(2), rng=np.random.default_rng(8))
    assert incident["discarded"] is True
    assert runner.d_current[0] == pytest.approx(done_d)
    assert runner.d_at_done[0] == pytest.approx(done_d)
    assert runner.done[0]


def test_goal_node_norm_assertion_passes_normal_goals_and_fires_on_synthetic_violation() -> None:
    ok_runner, _ = make_runner(1)
    ok_runner.begin_episodes(seed=9, goals=[straight_goal()])

    bad_runner, _ = make_runner(1)
    bad_goal = straight_goal(anchor_x=4.5)
    with pytest.raises(ValueError, match="goal-node norm > 4.0 m") as exc_info:
        bad_runner.begin_episodes(seed=10, goals=[bad_goal])
    assert "max=" in str(exc_info.value)


def test_evaluation_rows_include_d_at_done_step_trajectory_and_min_d() -> None:
    runner, env = make_runner(1, horizon=3)
    goal = straight_goal()
    goal_curve_ = goal_curve(goal, LENGTH)
    env.queue_step(straight_line(0.3))
    env.queue_step(straight_line(0.2))
    env.queue_step(goal_curve_)

    def action_fn(
        x: np.ndarray, goal_curves: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        del x, goal_curves, rng
        return zero_actions(1)

    result = evaluate_episodes(
        runner,
        n_episodes=1,
        seed=11,
        episode_index_start=0,
        action_fn=action_fn,
        rng=np.random.default_rng(12),
        goals=[goal],
    )

    row = result["episodes"][0]
    assert row["d_steps"] == pytest.approx([0.3, 0.2, 0.0], abs=1.0e-12)
    assert row["min_d"] == pytest.approx(0.0, abs=1.0e-12)
    assert row["d_at_done"] == pytest.approx(0.0, abs=1.0e-12)
    assert row["d_at_done_fallback"] is False
    assert row["final_d"] == pytest.approx(0.0, abs=1.0e-12)
    assert result["mean_d_at_done"] == pytest.approx(0.0, abs=1.0e-12)
    assert result["mean_min_d"] == pytest.approx(0.0, abs=1.0e-12)
