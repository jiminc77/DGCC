"""Env-stability fixes from gate verdict gate-m3r-reconvene-20260710 (choice D).

Covers the driver-level escalation logic only (env layer contract):
  (a) discard-storm livelock exit — > N consecutive discarded rounds force a
      full scene rebuild,
  (b) rebuild limit 5 -> 8 with crash-time preservation of the freshest
      agent checkpoint.

Training code, hyperparameters, reward constants and covenant thresholds are
untouched by the fix; these tests use fakes instead of Genesis/CUDA.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "p1_train", Path(__file__).resolve().parents[1] / "scripts" / "p1_train.py"
)
p1_train = importlib.util.module_from_spec(_SPEC)
sys.modules["p1_train"] = p1_train
_SPEC.loader.exec_module(p1_train)

N_ENVS = 4


class FakeAgent:
    def __init__(self) -> None:
        self.saved: list[Path] = []
        self.fail_save = False

    def save_checkpoint(self, path: Path) -> Path:
        if self.fail_save:
            raise OSError("disk full")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ckpt")
        self.saved.append(path)
        return path

    def select_actions(self, X, G, *, step, total_budget, rng, return_info=False):
        p = np.zeros(N_ENVS, dtype=int)
        delta = np.zeros((N_ENVS, 3), dtype=float)
        lift = ["low"] * N_ENVS
        if return_info:
            return p, delta, lift, {"q1_candidates": np.zeros((N_ENVS, 1))}
        return p, delta, lift


class FakeDiag:
    def __init__(self) -> None:
        self.history_saves = 0

    def log_action_info(self, *args: Any, **kwargs: Any) -> None:
        pass

    def log_nan_incidents(self, *args: Any, **kwargs: Any) -> None:
        pass

    def save_history(self) -> None:
        self.history_saves += 1


class FakeRunner:
    """Scripted runner: pops the next record or raises the queued error."""

    def __init__(self) -> None:
        self.records: list[Any] = []
        self.nan_incidents = 0
        self.magnitude_incidents = 0
        self.init_shapes = ["straight"] * N_ENVS
        self.t = np.zeros(N_ENVS, dtype=int)

    def step(self, p, delta, lift, rng=None):
        item = self.records.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def discarded_record() -> dict[str, Any]:
    return {
        "discarded": True,
        "active": np.ones(N_ENVS, dtype=bool),
        "reason": "magnitude covenant (test)",
        "bad_envs": np.array([0]),
    }


def idle_clean_record() -> dict[str, Any]:
    # Non-discarded round with no active envs: exercises the storm-counter
    # reset without touching the replay/diagnostics pipeline.
    return {"discarded": False, "active": np.zeros(N_ENVS, dtype=bool)}


def make_run(tmp_path: Path, *, storm_after: int = 3, max_rebuilds: int = 8):
    run = object.__new__(p1_train.TrainingRun)
    run.agent = FakeAgent()
    run.diag = FakeDiag()
    run.runner = FakeRunner()
    run.env = object()  # only identity-checked by the collect_round assert
    run.goal_curves = np.zeros((N_ENVS, 32, 3), dtype=float)
    run.rng = np.random.default_rng(0)
    run.transitions = 0
    run.total = 100_000
    run.full_rebuilds = 0
    run._consecutive_discards = 0
    run.max_full_rebuilds = max_rebuilds
    run.discard_storm_rebuild_after = storm_after
    run.models_dir = tmp_path / "models"
    run.models_dir.mkdir(parents=True, exist_ok=True)
    run.last_checkpoint = None
    run._prev_goal_flip = np.full(N_ENVS, -1, dtype=np.int8)
    run._episode_flip_transitions = np.zeros(N_ENVS, dtype=int)
    run._episode_flip_observations = np.zeros(N_ENVS, dtype=int)

    calls = {"build_scene": 0, "refresh": 0, "summary": 0}
    run.build_scene = lambda: calls.__setitem__("build_scene", calls["build_scene"] + 1)
    run.refresh_goal_curves = lambda: calls.__setitem__("refresh", calls["refresh"] + 1)
    run.save_run_summary = lambda: calls.__setitem__("summary", calls["summary"] + 1)
    return run, calls


def env_get_centerline(run) -> None:
    run.env = type(
        "FakeEnv", (), {"get_centerline_batch": lambda self: np.zeros((N_ENVS, 32, 3))}
    )()


# ---------------------------------------------------------------- verdict (b)


def test_default_limits_match_verdict() -> None:
    assert p1_train.MAX_FULL_REBUILDS == 8
    assert p1_train.DISCARD_STORM_REBUILD_AFTER == 10


def test_register_rebuild_below_limit_recovers(tmp_path: Path) -> None:
    run, _ = make_run(tmp_path)
    for expected in range(1, 9):  # rebuilds 1..8 all stay recoverable
        assert run._register_rebuild(context="round_recovery", error="e") is False
        assert run.full_rebuilds == expected
    assert run.agent.saved == []  # no crash checkpoint below the limit


def test_register_rebuild_limit_preserves_crash_checkpoint(tmp_path: Path) -> None:
    run, calls = make_run(tmp_path)
    run.full_rebuilds = 8
    run.transitions = 87_000
    assert run._register_rebuild(context="round_recovery", error="e") is True
    crash = run.models_dir / "ckpt_crash_0087000.pt"
    assert crash.exists()
    assert run.last_checkpoint == crash
    assert run.diag.history_saves == 1
    assert calls["summary"] == 1


def test_crash_checkpoint_failure_keeps_crash_path_alive(tmp_path: Path) -> None:
    run, calls = make_run(tmp_path)
    run.full_rebuilds = 8
    run.agent.fail_save = True
    assert run._register_rebuild(context="round_recovery", error="e") is True
    assert run.last_checkpoint is None
    assert calls["summary"] == 1  # summary still written for the audit trail


def test_collect_round_nonfinite_error_rebuilds_and_resets_storm(tmp_path: Path) -> None:
    run, calls = make_run(tmp_path)
    env_get_centerline(run)
    run._consecutive_discards = 2
    run.runner.records = [FloatingPointError("nan detected in rope state")]
    assert run.collect_round() == 0
    assert run.full_rebuilds == 1
    assert calls["build_scene"] == 1
    assert run._consecutive_discards == 0


def test_collect_round_non_nan_error_propagates(tmp_path: Path) -> None:
    run, _ = make_run(tmp_path)
    env_get_centerline(run)
    run.runner.records = [RuntimeError("unrelated failure")]
    with pytest.raises(RuntimeError, match="unrelated"):
        run.collect_round()
    assert run.full_rebuilds == 0


# ---------------------------------------------------------------- verdict (a)


def test_discard_storm_forces_rebuild_after_threshold(tmp_path: Path) -> None:
    run, calls = make_run(tmp_path, storm_after=3)
    env_get_centerline(run)
    run.runner.records = [discarded_record() for _ in range(4)]
    for _ in range(3):  # at the threshold: no escalation yet
        assert run.collect_round() == 0
    assert calls["build_scene"] == 0
    assert run.full_rebuilds == 0
    assert run._consecutive_discards == 3

    assert run.collect_round() == 0  # 4th consecutive discard exceeds threshold
    assert calls["build_scene"] == 1
    assert run.full_rebuilds == 1
    assert run._consecutive_discards == 0


def test_clean_round_resets_discard_storm_counter(tmp_path: Path) -> None:
    run, calls = make_run(tmp_path, storm_after=3)
    env_get_centerline(run)
    run.runner.records = [
        discarded_record(),
        discarded_record(),
        idle_clean_record(),
        discarded_record(),
        discarded_record(),
        discarded_record(),
    ]
    for _ in range(6):
        run.collect_round()
    # Never more than 3 *consecutive* discards -> no forced rebuild.
    assert calls["build_scene"] == 0
    assert run.full_rebuilds == 0
    assert run._consecutive_discards == 3


def test_discard_storm_at_rebuild_limit_raises_with_crash_checkpoint(
    tmp_path: Path,
) -> None:
    run, _ = make_run(tmp_path, storm_after=1, max_rebuilds=8)
    env_get_centerline(run)
    run.full_rebuilds = 8
    run.transitions = 10_240
    run.runner.records = [discarded_record(), discarded_record()]
    assert run.collect_round() == 0  # first discard: at threshold, no escalation
    with pytest.raises(FloatingPointError, match="discard-storm escalation"):
        run.collect_round()
    assert (run.models_dir / "ckpt_crash_0010240.pt").exists()
    assert run.full_rebuilds == 9
