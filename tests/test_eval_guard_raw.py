"""sprint_spec §5 eval-wall guard + §3 raw-trajectory instrumentation tests.

Both features default OFF; historical behavior must be bit-identical when the
flags are absent.  Guard semantics: an episode reseeded by more than K
discarded rounds is counted as FAILURE and flagged; the batch stops once all
unfinished slots are guarded.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from dgcc.goals.dual_goal import DualGoal
from dgcc.rl.evaluation import evaluate_episodes
from dgcc.tasks.domain import P1_LENGTH_M


def _line(n: int = 32, length: float = P1_LENGTH_M) -> np.ndarray:
    t = np.linspace(0.0, length, n)
    return np.column_stack((t, np.zeros_like(t), np.zeros_like(t)))


def _goal() -> DualGoal:
    return DualGoal(shape_template=_line(), anchor=np.zeros(3))


class FakeEnv:
    def __init__(self, batch: np.ndarray) -> None:
        self._batch = batch

    def get_centerline_batch(self) -> np.ndarray:
        return self._batch


class DiscardStormRunner:
    """Two-env runner: env0 succeeds at step 1; env1 hits discarded rounds
    (bad_envs=[1]) forever.  Without a guard the loop runs to the 2*horizon
    cap; with K it must stop early and fail env1."""

    def __init__(self, discard_forever: bool = True, succeed_env1_after: int | None = None) -> None:
        self.n_envs = 2
        self.config = SimpleNamespace(horizon=10)
        x = _line()
        self.goals = [_goal(), _goal()]
        self.nan_incidents = 0
        self.env = FakeEnv(np.stack([x, x]))
        self.done = np.array([False, False])
        self.succeeded = np.array([False, False])
        self.truncated = np.array([False, False])
        self.t = np.array([0, 0])
        self.d_current = np.array([0.01, 0.5])
        self.d_at_done = np.array([np.nan, np.nan])
        self.init_shapes = ["straight", "straight"]
        self.steps_taken = 0
        self.discards_served = 0
        self._discard_forever = discard_forever
        self._succeed_env1_after = succeed_env1_after

    def begin_episodes(self, **kwargs: Any) -> dict[str, Any]:
        return {"init_shapes": list(self.init_shapes), "d_initial": np.array([0.3, 0.4])}

    def all_done(self) -> bool:
        return bool(np.all(self.done))

    def step(self, p, delta, lift, rng) -> dict[str, Any]:
        x = _line()
        if not self.done[0]:
            # env0 finishes immediately with success.
            self.done[0] = True
            self.succeeded[0] = True
            self.t[0] = 1
            self.d_at_done[0] = 0.01
            active = np.array([True, False])
            return {
                "active": active,
                "d_after": np.array([0.01, 0.5]),
                "X_after": np.stack([x, x]),
                "reward": np.array([5.0, 0.0]),
                "done": self.done.copy(),
                "t": self.t.copy(),
            }
        # env1: discarded rounds (reseed storms)
        self.discards_served += 1
        if self._succeed_env1_after is not None and self.discards_served > self._succeed_env1_after:
            self.done[1] = True
            self.succeeded[1] = True
            self.t[1] = 1
            self.d_at_done[1] = 0.01
            return {
                "active": np.array([False, True]),
                "d_after": np.array([0.01, 0.01]),
                "X_after": np.stack([x, x]),
                "reward": np.array([0.0, 5.0]),
                "done": self.done.copy(),
                "t": self.t.copy(),
            }
        return {"discarded": True, "reason": "storm", "bad_envs": np.array([1])}


def _random_action(X, G, rng):
    B = X.shape[0]
    return (
        np.zeros(B, dtype=int),
        np.zeros((B, 3)),
        ["low"] * B,
    )


def test_guard_off_preserves_unlimited_retries():
    runner = DiscardStormRunner(succeed_env1_after=8)
    rng = np.random.default_rng(0)
    result = evaluate_episodes(
        runner, n_episodes=2, seed=0, episode_index_start=1,
        action_fn=_random_action, rng=rng, goals=runner.goals,
    )
    # env1 eventually succeeded after 8 discarded rounds — no guard, so success.
    rows = {ep["episode_id"]: ep for ep in result["episodes"]}
    assert rows[1]["success"] is True
    assert rows[1]["eval_wall_guard"] is False
    assert result["wall_guard_k"] is None


def test_guard_k_fails_stormy_episode_and_stops():
    runner = DiscardStormRunner(discard_forever=True)
    rng = np.random.default_rng(0)
    result = evaluate_episodes(
        runner, n_episodes=2, seed=0, episode_index_start=1,
        action_fn=_random_action, rng=rng, goals=runner.goals,
        wall_guard_k=5,
    )
    rows = {ep["episode_id"]: ep for ep in result["episodes"]}
    assert rows[0]["success"] is True and rows[0]["eval_wall_guard"] is False
    assert rows[1]["success"] is False and rows[1]["eval_wall_guard"] is True
    assert rows[1]["discard_exposure"] == 6  # K=5 exceeded at the 6th reseed
    # storm stopped shortly after exceeding K, not at the 2*horizon cap
    assert runner.discards_served <= 7
    assert result["eval_wall_guard_rate"] == 0.5
    assert result["wall_guard_k"] == 5


def test_guard_does_not_inflate_success():
    """Even if a guarded episode LATER succeeds, it stays failure (conservative)."""

    runner = DiscardStormRunner(succeed_env1_after=7)
    rng = np.random.default_rng(0)
    result = evaluate_episodes(
        runner, n_episodes=2, seed=0, episode_index_start=1,
        action_fn=_random_action, rng=rng, goals=runner.goals,
        wall_guard_k=5,
    )
    rows = {ep["episode_id"]: ep for ep in result["episodes"]}
    assert rows[1]["eval_wall_guard"] is True
    assert rows[1]["success"] is False


def test_record_raw_fields_present_and_off_by_default():
    runner = DiscardStormRunner(succeed_env1_after=1)
    rng = np.random.default_rng(0)
    result = evaluate_episodes(
        runner, n_episodes=2, seed=0, episode_index_start=1,
        action_fn=_random_action, rng=rng, goals=runner.goals,
        record_raw=True,
    )
    for ep in result["episodes"]:
        assert "x_initial" in ep and "x_steps" in ep and "x_terminal" in ep
        assert "reseed_boundary" in ep and "goal_index" in ep
        assert np.asarray(ep["x_initial"]).shape == (32, 3)
        if ep["x_terminal"] is not None:
            assert np.asarray(ep["x_terminal"]).shape == (32, 3)
    assert result["record_raw"] is True
    # reseed boundary flagged only for the stormy env
    rows = {ep["episode_id"]: ep for ep in result["episodes"]}
    assert rows[0]["reseed_boundary"] is False
    assert rows[1]["reseed_boundary"] is True

    # default OFF: no raw fields
    runner2 = DiscardStormRunner(succeed_env1_after=1)
    result2 = evaluate_episodes(
        runner2, n_episodes=2, seed=0, episode_index_start=1,
        action_fn=_random_action, rng=np.random.default_rng(0), goals=runner2.goals,
    )
    assert all("x_initial" not in ep for ep in result2["episodes"])
    assert result2["record_raw"] is False
