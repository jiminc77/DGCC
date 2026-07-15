"""P2b D_shape eval channel (M4, observation-only — inherited risk #2).

Geometry: flip decided ONCE from the initial state, applied identically to
initial and terminal measurements. Lifecycle: an early-finished episode's
D_shape_at_done comes from ITS terminal X_after (captured while active),
never from post-terminal batch state.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dgcc.goals.distance import flip_consistent_shape_distance  # noqa: E402
from dgcc.goals.dual_goal import DualGoal  # noqa: E402
from dgcc.rl.evaluation import evaluate_episodes, summarize_episodes  # noqa: E402
from dgcc.tasks.domain import P1_LENGTH_M  # noqa: E402


def _line(n: int = 32, length: float = P1_LENGTH_M) -> np.ndarray:
    t = np.linspace(0.0, length, n)
    return np.column_stack((t, np.zeros_like(t), np.zeros_like(t)))


def _arc(n: int = 32, length: float = P1_LENGTH_M, amp: float = 0.2) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    return np.column_stack((t * length, amp * np.sin(np.pi * t), np.zeros_like(t)))


class FakeEnv:
    def __init__(self, batches: list[np.ndarray]) -> None:
        self._batches = batches
        self.calls = 0

    def get_centerline_batch(self) -> np.ndarray:
        batch = self._batches[min(self.calls, len(self._batches) - 1)]
        self.calls += 1
        return batch


class FakeRunner:
    """Two-env scripted runner: env0 finishes at step 1, env1 runs 2 steps.

    After env0 finishes, the batch state for slot 0 keeps MOVING (post-
    terminal drift) — the D_shape channel must ignore that drift.
    """

    def __init__(self, goals: list[DualGoal], x0: np.ndarray, x1_steps: list[np.ndarray],
                 x0_terminal: np.ndarray, x0_drift: np.ndarray) -> None:
        self.n_envs = 2
        self.config = SimpleNamespace(horizon=10)
        self.goals = goals
        self.nan_incidents = 0
        self._step = 0
        self._x0_terminal = x0_terminal
        self._x0_drift = x0_drift
        self._x1_steps = x1_steps
        # env.get_centerline_batch is called once after begin (X_initial) and
        # once per while-loop iteration (2 steps).
        self.env = FakeEnv([
            np.stack([x0, x1_steps[0]]),                      # X_initial capture
            np.stack([x0, x1_steps[0]]),                      # step 0 action input
            np.stack([x0_drift, x1_steps[0]]),                # step 1 action input
        ])
        self.done = np.array([False, False])
        self.succeeded = np.array([True, False])
        self.t = np.array([1, 2])
        self.d_current = np.array([0.01, 0.5])
        self.d_at_done = np.array([0.01, 0.5])
        self.init_shapes = ["straight", "straight"]

    def begin_episodes(self, **kwargs: Any) -> dict[str, Any]:
        return {"init_shapes": list(self.init_shapes), "d_initial": np.array([0.3, 0.4])}

    def all_done(self) -> bool:
        return self._step >= 2

    def step(self, p, delta, lift, *, rng) -> dict[str, Any]:
        self._step += 1
        if self._step == 1:
            # both active; env0 reaches ITS terminal state here (then done)
            return {
                "active": np.array([True, True]),
                "reward": np.array([1.0, 0.1]),
                "d_after": np.array([0.01, 0.45]),
                "X_after": np.stack([self._x0_terminal, self._x1_steps[1]]),
            }
        # env0 now done: inactive, but the batch row DRIFTS post-terminal
        return {
            "active": np.array([False, True]),
            "reward": np.array([0.0, 0.2]),
            "d_after": np.array([0.02, 0.4]),
            "X_after": np.stack([self._x0_drift, self._x1_steps[2]]),
        }


def _action_fn(X, G, rng):
    return np.zeros(len(X), dtype=int), np.zeros((len(X), 3)), ["low"] * len(X)


def test_d_shape_uses_terminal_not_post_terminal_state() -> None:
    goal_line = DualGoal(shape_template=_line(), anchor=np.zeros(3))
    x0_terminal = _line()                       # env0 terminal == goal shape
    x0_drift = _arc(amp=0.3)                    # post-terminal drift, must be ignored
    x1 = [_arc(amp=0.25), _arc(amp=0.2), _arc(amp=0.15)]
    runner = FakeRunner([goal_line, goal_line], _arc(amp=0.1), x1, x0_terminal, x0_drift)

    result = evaluate_episodes(
        runner,
        n_episodes=2,
        seed=0,
        episode_index_start=90_001,
        action_fn=_action_fn,
        rng=np.random.default_rng(0),
        goals=[goal_line, goal_line],
    )
    rows = {ep["episode_id"]: ep for ep in result["episodes"]}
    # env0: terminal centerline == goal shape -> D_shape ~ 0 (drift would not be)
    assert rows[0]["d_shape_at_done"] < 1e-9
    drift_d = flip_consistent_shape_distance(x0_drift, _line(), P1_LENGTH_M, flip=False)
    assert drift_d > 0.01  # the ignored drift state is far from the goal
    # env1: terminal comes from its LAST ACTIVE step (x1[2])
    expected = flip_consistent_shape_distance(x1[2], _line(), P1_LENGTH_M, flip=False)
    assert abs(rows[1]["d_shape_at_done"] - expected) < 1e-12
    # summary channel present
    assert result["mean_d_shape_at_done"] is not None


def test_d_shape_flip_decided_once_from_initial() -> None:
    # Initial state = REVERSED hook -> canonical flip decision is True. The
    # TERMINAL state is the un-reversed hook: re-deciding the flip from the
    # terminal would choose False (d ~ 0); the contract requires the INITIAL
    # decision (True) applied to the terminal, giving the strictly larger
    # flipped distance. This discriminates decide-once from re-decide.
    from dgcc.goals.distance import canonical_shape_flip
    from dgcc.goals.dual_goal import goal_curve

    hook = np.column_stack((
        np.linspace(0.0, P1_LENGTH_M, 32),
        np.concatenate([np.zeros(16), np.linspace(0.0, 0.3, 16)]),
        np.zeros(32),
    ))
    goal_hook = DualGoal(shape_template=hook, anchor=np.zeros(3))
    G = goal_curve(goal_hook, P1_LENGTH_M)
    x_init_reversed = hook[::-1].copy()
    flip = canonical_shape_flip(x_init_reversed, goal_hook, P1_LENGTH_M)
    assert flip is True  # precondition: reversed init selects the flipped orientation

    terminal = hook  # un-reversed: re-decision would flip=False here
    d_decided_once = flip_consistent_shape_distance(terminal, G, P1_LENGTH_M, flip=True)
    d_redecided = flip_consistent_shape_distance(terminal, G, P1_LENGTH_M, flip=False)
    assert d_decided_once > d_redecided + 1e-6  # the two behaviors are distinguishable

    runner = FakeRunner([goal_hook, goal_hook], x_init_reversed,
                        [x_init_reversed, x_init_reversed, x_init_reversed],
                        terminal, terminal)
    result = evaluate_episodes(
        runner, n_episodes=2, seed=0, episode_index_start=90_001,
        action_fn=_action_fn, rng=np.random.default_rng(0),
        goals=[goal_hook, goal_hook],
    )
    row = result["episodes"][0]
    assert abs(row["d_shape_at_done"] - d_decided_once) < 1e-12
    expected_initial = flip_consistent_shape_distance(x_init_reversed, G, P1_LENGTH_M, flip=True)
    assert abs(row["d_shape_initial"] - expected_initial) < 1e-12


def test_summarizer_tolerates_historical_rows_without_d_shape() -> None:
    legacy = [{
        "episode_id": 0, "goal_label": None, "init_template": "straight",
        "success": True, "steps": 1, "return": 1.0, "discounted_return": 1.0,
        "final_d": 0.1, "d_at_done": 0.1, "d_at_done_fallback": False,
        "d_steps": [0.1], "min_d": 0.1, "d_initial": 0.3, "q_first": None,
    }]
    summary = summarize_episodes(legacy)
    assert summary["mean_d_shape_at_done"] is None
