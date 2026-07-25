"""CPU contract for paired one-step Q1/Qmin validation diagnostics."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "p1_train_v2_test", ROOT / "scripts/p1_train.py"
)
assert SPEC is not None and SPEC.loader is not None
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


class FakeEnv:
    def get_centerline_batch(self) -> np.ndarray:
        return np.zeros((2, 32, 3), dtype=float)


class FakeRunner:
    def __init__(self) -> None:
        self.env = FakeEnv()
        self.goals = [object(), object()]

    def begin_episodes(self, *, seed, episode_index, goals):
        del seed, episode_index
        self.goals = goals
        return {"d_initial": np.ones(2, dtype=float)}

    def step(self, p, delta, lift, *, rng):
        del delta, lift, rng
        progress = np.where(np.asarray(p) == 0, 0.1, 0.3)
        return {
            "active": np.ones(2, dtype=bool),
            "d_after": 1.0 - progress,
        }


class FakeAgent:
    def select_actions(self, X, G, *, selector_operator, **kwargs):
        del G, kwargs
        contact = 0 if selector_operator == "q1" else 1
        return (
            np.full(len(X), contact, dtype=int),
            np.zeros((len(X), 3), dtype=float),
            ["low"] * len(X),
        )


def test_counterfactual_selector_uses_identical_starts_and_realized_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(DRIVER, "goal_curve", lambda goal, length: np.zeros((32, 3)))
    run = object.__new__(DRIVER.TrainingRun)
    run.task = "t2"
    run.runner = FakeRunner()
    run.val_goals = [object(), object()]
    run.n_envs = 2
    run.seed = 7
    run.agent = FakeAgent()
    run.transitions = 25_000
    run.total = 300_000

    result = run.counterfactual_selector_metrics(episode_index_start=10_000)
    assert result["states"] == 2.0
    assert result["p_q1_p_qmin_agreement"] == 0.0
    assert result["q1_selected_realized_progress_mean"] == pytest.approx(0.1)
    assert result["qmin_selected_realized_progress_mean"] == pytest.approx(0.3)
    assert result["q1_minus_qmin_realized_progress_mean"] == pytest.approx(-0.2)
