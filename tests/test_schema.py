from __future__ import annotations

import numpy as np
import pytest

from dgcc.logging.schema import TransitionRecord


def make_record() -> TransitionRecord:
    X_before = np.arange(96, dtype=float).reshape(32, 3) / 100.0
    X_after = X_before + 0.01
    return TransitionRecord(
        X_before=X_before,
        X_after=X_after,
        p=7,
        delta=np.array([0.01, -0.02, 0.03], dtype=float),
        lift="low",
        grasp_success=True,
        settle_steps=123,
        rope_params={
            "length_m": 1.0,
            "n_segments": 50,
            "bend_stiffness": 1.0,
            "twist_stiffness": 1.0,
            "friction": 0.3,
            "radius": 0.005,
        },
        seed=42,
        sim="unit-test",
        timestamp="2026-07-02T00:00:00Z",
        commit_hash="abc1234",
    )


def test_transition_record_round_trip() -> None:
    record = make_record()
    restored = TransitionRecord.from_dict(record.to_dict())

    assert np.allclose(restored.X_before, record.X_before)
    assert np.allclose(restored.X_after, record.X_after)
    assert np.allclose(restored.delta, record.delta)
    assert restored.p == record.p
    assert restored.lift == record.lift
    assert restored.grasp_success is record.grasp_success
    assert restored.settle_steps == record.settle_steps
    assert restored.rope_params == record.rope_params
    assert restored.seed == record.seed
    assert restored.sim == record.sim
    assert restored.timestamp == record.timestamp
    assert restored.commit_hash == record.commit_hash


def test_transition_record_rejects_bad_centerline_shape() -> None:
    data = make_record().to_dict()
    data["X_before"] = np.zeros((31, 3), dtype=float).tolist()

    with pytest.raises(ValueError, match="X_before must have shape"):
        TransitionRecord.from_dict(data)


def test_env_base_imports() -> None:
    from dgcc.envs.base import DLOEnvBase, RopeParams

    assert DLOEnvBase.K == 32
    params = RopeParams(
        length_m=1.0,
        bend_stiffness=1.0,
        twist_stiffness=1.0,
        friction=0.3,
    )
    assert params.n_segments == 50
    assert params.radius == 0.005
