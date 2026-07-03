from __future__ import annotations

import numpy as np
import pytest

from dgcc.goals.distance import D, chamfer, correspondence_l2
from dgcc.goals.dual_goal import (
    ANCHOR_CHANNEL_COUNT,
    CG_DIM,
    SHAPE_CHANNEL_COUNT,
    DualGoal,
    c_g,
    goal_curve,
    make_goal,
    make_shape_template,
)


def test_chamfer_is_symmetric_for_different_point_counts() -> None:
    t_a = np.linspace(0.0, 1.0, 17)
    t_b = np.linspace(0.0, 1.0, 43)
    a = np.column_stack((t_a, 0.1 * np.sin(np.pi * t_a), np.zeros_like(t_a)))
    b = np.column_stack((t_b, 0.1 * np.sin(np.pi * t_b) + 0.02, np.zeros_like(t_b)))

    assert chamfer(a, b) == pytest.approx(chamfer(b, a))


def test_length_normalized_distance_is_scale_invariant() -> None:
    base_goal = make_goal("straight", np.zeros(3))
    x = goal_curve(base_goal, 1.0) + np.array([0.0, 0.08, 0.0])

    scaled_goal = make_goal("straight", np.zeros(3))
    x_scaled = goal_curve(scaled_goal, 2.5) + np.array([0.0, 0.08 * 2.5, 0.0])

    assert D(x, base_goal, 1.0) == pytest.approx(D(x_scaled, scaled_goal, 2.5))

def test_correspondence_l2_is_symmetric_including_flip_case() -> None:
    length = 1.4
    x = goal_curve(make_goal("s_curve", np.array([0.25, -0.2, 0.04])), length)
    y = goal_curve(make_goal("u_bend", np.array([-0.15, 0.35, 0.07])), length)[::-1]

    assert correspondence_l2(x, y, length) == pytest.approx(correspondence_l2(y, x, length))
    assert correspondence_l2(x, y, length, shape_only=True) == pytest.approx(
        correspondence_l2(y, x, length, shape_only=True)
    )


def test_correspondence_l2_is_flip_invariant() -> None:
    length = 1.1
    x = goal_curve(make_goal("s_curve", np.array([0.1, -0.1, 0.03])), length)
    y = goal_curve(make_goal("straight", np.array([-0.2, 0.3, 0.06])), length)

    distance = correspondence_l2(x, y, length)
    assert correspondence_l2(x, y[::-1], length) == pytest.approx(distance)
    assert correspondence_l2(x[::-1], y, length) == pytest.approx(distance)


def test_correspondence_l2_shape_only_translation_invariant_absolute_sensitive() -> None:
    length = 1.3
    curve = goal_curve(make_goal("s_curve", np.array([0.2, -0.3, 0.05])), length)
    translation = np.array([0.31, -0.17, 0.09])

    assert correspondence_l2(curve + translation, curve, length, shape_only=True) == pytest.approx(
        0.0,
        abs=1.0e-12,
    )
    assert correspondence_l2(curve + translation, curve, length) == pytest.approx(
        np.linalg.norm(translation) / length
    )


def test_correspondence_l2_scale_normalization() -> None:
    length = 0.9
    scale = 2.75
    x = goal_curve(make_goal("s_curve", np.array([0.12, -0.07, 0.03])), length)
    y = goal_curve(make_goal("u_bend", np.array([-0.18, 0.22, 0.08])), length)

    assert correspondence_l2(x, y, length) == pytest.approx(
        correspondence_l2(x * scale, y * scale, length * scale)
    )
    assert correspondence_l2(x, y, length, shape_only=True) == pytest.approx(
        correspondence_l2(x * scale, y * scale, length * scale, shape_only=True)
    )


def test_correspondence_l2_exact_goal_zero() -> None:
    length = 1.6
    curve = goal_curve(make_goal("u_bend", np.array([0.2, 0.1, 0.04])), length)

    assert correspondence_l2(curve, curve, length) == pytest.approx(0.0, abs=1.0e-12)
    assert correspondence_l2(curve, curve[::-1], length) == pytest.approx(0.0, abs=1.0e-12)
    assert correspondence_l2(curve, curve, length, shape_only=True) == pytest.approx(0.0, abs=1.0e-12)



def test_c_g_dimension_and_documented_channel_split() -> None:
    goal = make_goal("u_bend", np.array([0.1, -0.2, 0.03]))
    x = goal_curve(goal, 1.0) + np.array([0.01, 0.02, 0.0])

    vector = c_g(x, goal, 1.0)

    assert SHAPE_CHANNEL_COUNT == 21
    assert ANCHOR_CHANNEL_COUNT == 3
    assert CG_DIM == 24
    assert vector.shape == (24,)
    assert vector[:SHAPE_CHANNEL_COUNT].shape == (21,)
    assert vector[SHAPE_CHANNEL_COUNT:].shape == (3,)


def test_identical_shape_and_anchor_have_zero_features_and_distance() -> None:
    goal = make_goal("straight", np.array([0.2, -0.1, 0.04]))
    x = goal_curve(goal, 1.3)

    assert np.allclose(c_g(x, goal, 1.3), 0.0, atol=1.0e-12)
    assert D(x, goal, 1.3) == pytest.approx(0.0, abs=1.0e-12)


def test_centroid_and_endpoint_anchor_modes_differ() -> None:
    centroid_goal = make_goal("straight", np.zeros(3), anchor_mode="centroid")
    endpoint_goal = make_goal("straight", np.zeros(3), anchor_mode="endpoint")
    x = goal_curve(centroid_goal, 1.0)

    centroid_vector = c_g(x, centroid_goal, 1.0)
    endpoint_vector = c_g(x, endpoint_goal, 1.0)

    assert not np.allclose(goal_curve(centroid_goal, 1.0), goal_curve(endpoint_goal, 1.0))
    assert not np.allclose(centroid_vector[-3:], endpoint_vector[-3:])


def test_degenerate_inputs_raise() -> None:
    with pytest.raises(ValueError, match="non-zero arc length"):
        DualGoal(shape_template=np.zeros((32, 3)), anchor=np.zeros(3))

    goal = make_goal("straight", np.zeros(3))
    with pytest.raises(ValueError, match="non-zero arc length"):
        D(np.zeros((32, 3)), goal, 1.0)

    with pytest.raises(ValueError, match="positive finite"):
        D(goal_curve(goal, 1.0), goal, 0.0)
    with pytest.raises(ValueError, match="non-zero arc length"):
        correspondence_l2(np.zeros((32, 3)), goal_curve(goal, 1.0), 1.0)

    with pytest.raises(ValueError, match="non-zero arc length"):
        correspondence_l2(goal_curve(goal, 1.0), np.zeros((32, 3)), 1.0)

    with pytest.raises(ValueError, match="positive finite"):
        correspondence_l2(goal_curve(goal, 1.0), goal_curve(goal, 1.0), 0.0)

    with pytest.raises(ValueError, match="at least one point"):
        chamfer(np.empty((0, 3)), goal_curve(goal, 1.0))

    with pytest.raises(ValueError, match="shape"):
        DualGoal(shape_template=make_shape_template("straight")[:, :2], anchor=np.zeros(3))


def test_c_g_exactly_zero_for_exact_goal_curve_both_anchor_modes() -> None:
    # Locks the QA C2 invariant: both curves flow through the same canonical
    # resample path, so an exact goal-curve input yields exactly-zero c_g.
    for anchor_mode in ("centroid", "endpoint"):
        goal = make_goal("s_curve", np.array([0.25, -0.4, 0.05]), anchor_mode=anchor_mode)
        for length in (0.5, 1.0, 1.6):
            x = goal_curve(goal, length)
            vector = c_g(x, goal, length)
            # Shape channels flow through the identical canonical path -> exactly zero.
            assert np.allclose(vector[:-3], 0.0, atol=1.0e-15), (anchor_mode, length)
            # Anchor channels carry only centroid-of-resample interpolation noise
            # (~1e-6 * L floor), not a path asymmetry.
            assert np.abs(vector[-3:]).max() < 1.0e-5 * length, (anchor_mode, length, np.abs(vector[-3:]).max())
