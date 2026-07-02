from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from dgcc.envs.base import RopeParams
from dgcc.envs.dlolab import (
    BEND_BASE,
    MU_K_RATIO,
    MU_S_BASE,
    TWIST_BASE,
    DLOLabEnv,
    analytic_init_centerline,
    centerline_arc_length,
    ensure_genesis_initialized,
    sample_grasp,
)


def _params(*, length_m: float = 0.75, bend: float = 1.0, twist: float = 1.0, friction: float = 1.0) -> RopeParams:
    return RopeParams(
        length_m=length_m,
        n_segments=32,
        bend_stiffness=bend,
        twist_stiffness=twist,
        friction=friction,
        radius=0.005,
    )


@pytest.fixture(scope="module")
def genesis_session():
    return ensure_genesis_initialized(0)


def _fast_env(*, grasp_realism: bool = False, reset_steps: int = 8) -> DLOLabEnv:
    return DLOLabEnv(
        initial_settle_steps=0,
        reset_settle_max_steps=reset_steps,
        move_step_size=0.03,
        move_hold_steps=0,
        grasp_realism=grasp_realism,
    )


def test_sample_grasp_statistics_cpu_only() -> None:
    rng = np.random.default_rng(0)
    draws = [sample_grasp(16, 32, rng, enabled=True) for _ in range(1000)]
    failures = sum(1 for _, success in draws if not success)
    offsets = Counter(p_actual - 16 for p_actual, _ in draws)

    failure_rate = failures / 1000.0
    assert 0.04 <= failure_rate <= 0.06
    assert set(offsets) == {-1, 0, 1}
    # Uniform three-way draw with 1000 samples should keep every bucket well away from zero.
    assert all(count >= 250 for count in offsets.values())

    off_rng = np.random.default_rng(123)
    disabled = [sample_grasp(7, 32, off_rng, enabled=False) for _ in range(100)]
    assert disabled == [(7, True)] * 100


def test_analytic_init_shape_rejects_unknown_without_gpu() -> None:
    with pytest.raises(ValueError, match="init_shape"):
        analytic_init_centerline(_params(), "bent", seed=0)


@pytest.mark.gpu
def test_reset_and_primitive_are_deterministic(genesis_session) -> None:
    params = _params(length_m=0.6)
    env_a = _fast_env(grasp_realism=True, reset_steps=8)
    env_b = _fast_env(grasp_realism=True, reset_steps=8)

    env_a.reset(params, init_shape="random_smooth", seed=17)
    env_b.reset(params, init_shape="random_smooth", seed=17)
    assert np.allclose(env_a.get_centerline(), env_b.get_centerline(), rtol=1e-5, atol=2e-5)

    delta = np.array([0.035, -0.015, 0.0])
    out_a = env_a.step_primitive(16, delta, "low")

    env_c = _fast_env(grasp_realism=True, reset_steps=8)
    env_c.reset(params, init_shape="random_smooth", seed=17)
    out_c = env_c.step_primitive(16, delta, "low")

    # Genesis runs in float32 on the GPU; 2e-5 m is below 0.1% of the primitive scale.
    assert out_a["grasp_success"] == out_c["grasp_success"]
    assert out_a["info"]["p_actual"] == out_c["info"]["p_actual"]
    assert np.allclose(out_a["X_after"], out_c["X_after"], rtol=1e-5, atol=2e-5)

@pytest.mark.gpu
def test_failed_grasp_leaves_rope_unchanged(genesis_session) -> None:
    env = _fast_env(grasp_realism=True, reset_steps=8)
    env.reset(_params(length_m=0.6), init_shape="straight", seed=25)

    result = env.step_primitive(16, np.array([0.04, 0.0, 0.0]), "low")

    assert result["grasp_success"] is False
    assert result["settle_steps"] == 0
    assert np.allclose(result["X_after"], result["X_before"], rtol=0.0, atol=0.0)


@pytest.mark.gpu
def test_delta_norm_is_clamped_to_15cm(genesis_session) -> None:
    env = _fast_env(grasp_realism=False, reset_steps=8)
    env.reset(_params(length_m=0.6), init_shape="straight", seed=3)

    result = env.step_primitive(16, np.array([0.5, 0.0, 0.0]), "low")

    clamped = np.asarray(result["info"]["delta_clamped"], dtype=float)
    assert np.isclose(np.linalg.norm(clamped), 0.15, rtol=0.0, atol=1e-7)
    assert np.linalg.norm(clamped) <= 0.15 + 1e-7


@pytest.mark.gpu
def test_settle_converges_after_low_lift_small_delta(genesis_session) -> None:
    env = _fast_env(grasp_realism=False, reset_steps=8)
    env.reset(_params(length_m=0.6), init_shape="straight", seed=5)

    result = env.step_primitive(16, np.array([0.015, 0.005, 0.0]), "low")

    assert result["grasp_success"] is True
    assert result["info"]["settle_converged"] is True
    assert 0 <= result["settle_steps"] <= 5000
    assert result["info"]["max_node_speed"] < 1e-3


@pytest.mark.gpu
def test_init_shapes_are_finite_distinct_and_qualitative(genesis_session) -> None:
    params = _params(length_m=0.75)
    shapes: dict[str, np.ndarray] = {}
    for init_shape in ("straight", "u_bend", "s_curve", "random_smooth"):
        env = _fast_env(grasp_realism=False, reset_steps=8)
        env.reset(params, init_shape=init_shape, seed=11)
        centerline = env.get_centerline()
        assert centerline.shape == (32, 3)
        assert np.all(np.isfinite(centerline))
        shapes[init_shape] = centerline

    for i, name_a in enumerate(shapes):
        for name_b in list(shapes)[i + 1 :]:
            assert np.linalg.norm(shapes[name_a] - shapes[name_b]) > 1e-2

    end_to_end = {name: float(np.linalg.norm(x[-1] - x[0])) for name, x in shapes.items()}
    assert end_to_end["straight"] > end_to_end["s_curve"] > end_to_end["u_bend"]


@pytest.mark.gpu
def test_param_sweep_length_setters_and_bend_response(genesis_session) -> None:
    for length in (0.5, 1.0, 1.6):
        env = _fast_env(grasp_realism=False, reset_steps=8)
        env.reset(_params(length_m=length), init_shape="straight", seed=21)
        arc_length = centerline_arc_length(env.get_centerline_raw())
        # Float32 solver/state-setting and a short reset settle keep length within 3%.
        assert abs(arc_length - length) <= max(0.015, 0.03 * length)

    sweep_params = _params(length_m=0.75, bend=0.5, twist=2.0, friction=2.0)
    env = _fast_env(grasp_realism=False, reset_steps=8)
    info = env.reset(sweep_params, init_shape="straight", seed=22)
    mapped = info["mapped_parameters"]
    assert mapped["bending_stiffness_E"] == pytest.approx(BEND_BASE * 0.5)
    assert mapped["twisting_stiffness_G"] == pytest.approx(TWIST_BASE * 2.0)
    assert mapped["mu_s"] == pytest.approx(MU_S_BASE * 2.0)
    assert mapped["mu_k"] == pytest.approx(MU_K_RATIO * MU_S_BASE * 2.0)

    primitive_delta = np.array([0.04, 0.035, 0.0])
    soft = _fast_env(grasp_realism=False, reset_steps=8)
    stiff = _fast_env(grasp_realism=False, reset_steps=8)
    soft.reset(_params(length_m=0.75, bend=0.5), init_shape="u_bend", seed=30)
    stiff.reset(_params(length_m=0.75, bend=2.0), init_shape="u_bend", seed=30)
    soft_after = soft.step_primitive(16, primitive_delta, "low")["X_after"]
    stiff_after = stiff.step_primitive(16, primitive_delta, "low")["X_after"]

    assert np.linalg.norm(soft_after - stiff_after) > 1e-5
