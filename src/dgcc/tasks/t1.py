"""T1 local-shaping goal samplers (P1 §5).

T1-a straighten:          goal = straight template + anchor(current centroid)
T1-b single_bend:         goal = U-family single-curvature template
                          (one u_bend parameter varied: the arc angle)
T1-c endpoint_reposition: goal = keep current shape + move the anchor by a
                          displacement with norm in [0.2, 0.4] m

All samplers are deterministic given the caller-provided
``numpy.random.Generator`` and the current centerline.
"""

from __future__ import annotations

import numpy as np

from dgcc.goals.dual_goal import (
    DualGoal,
    anchor_of,
    canonical_centerline,
    make_shape_template,
)
from dgcc.phi.resample import resample

T1_TASKS: tuple[str, ...] = ("t1a_straighten", "t1b_single_bend", "t1c_endpoint_reposition")

#: T1-b single varied parameter: circular-arc angle (u_bend template is the
#: semicircle, i.e. angle == pi).  One-parameter U-family variation.
T1B_ARC_ANGLE_RANGE: tuple[float, float] = (0.5 * np.pi, 1.5 * np.pi)

#: T1-c anchor displacement magnitude range in meters (spec: [0.2, 0.4] m).
T1C_DISPLACEMENT_RANGE: tuple[float, float] = (0.2, 0.4)

_DENSE_SAMPLES = 257


def arc_template(arc_angle: float) -> np.ndarray:
    """Return a unit-arc-length planar circular-arc template of given angle.

    ``arc_angle == pi`` reproduces the u_bend semicircle family; other values
    are the single-parameter variation required by T1-b.
    """

    angle = float(arc_angle)
    if not np.isfinite(angle) or angle <= 0.0:
        raise ValueError("arc_angle must be a positive finite float")
    radius = 1.0 / angle  # unit arc length: r * angle == 1
    theta = np.linspace(0.0, angle, _DENSE_SAMPLES)
    dense = np.column_stack(
        (
            radius * np.cos(theta),
            radius * np.sin(theta),
            np.zeros_like(theta),
        )
    )
    return resample(dense)


def sample_t1a_goal(X_current: np.ndarray, rng: np.random.Generator) -> DualGoal:
    """Straighten: straight template anchored at the current centroid."""

    del rng  # deterministic given the state; signature kept uniform
    return DualGoal(
        shape_template=make_shape_template("straight"),
        anchor=anchor_of(X_current, "centroid"),
        anchor_mode="centroid",
        template_name="t1a_straighten",
    )


def sample_t1b_goal(X_current: np.ndarray, rng: np.random.Generator) -> DualGoal:
    """Single bend: U-family arc with one varied curvature parameter."""

    low, high = T1B_ARC_ANGLE_RANGE
    angle = float(rng.uniform(low, high))
    return DualGoal(
        shape_template=arc_template(angle),
        anchor=anchor_of(X_current, "centroid"),
        anchor_mode="centroid",
        template_name=f"t1b_single_bend(arc_angle={angle:.6f})",
    )


def sample_t1c_goal(X_current: np.ndarray, rng: np.random.Generator) -> DualGoal:
    """Endpoint reposition: keep the current shape, move the anchor 0.2-0.4 m.

    The displacement is horizontal (xy plane) so the translated goal stays on
    the support plane; its magnitude is uniform in ``[0.2, 0.4]`` m.
    """

    current = canonical_centerline(X_current)
    magnitude = float(rng.uniform(*T1C_DISPLACEMENT_RANGE))
    direction_angle = float(rng.uniform(0.0, 2.0 * np.pi))
    displacement = np.array(
        [magnitude * np.cos(direction_angle), magnitude * np.sin(direction_angle), 0.0]
    )
    return DualGoal(
        shape_template=current,
        anchor=anchor_of(current, "endpoint") + displacement,
        anchor_mode="endpoint",
        template_name=f"t1c_endpoint_reposition(d={magnitude:.6f})",
    )


_SAMPLERS = {
    "t1a_straighten": sample_t1a_goal,
    "t1b_single_bend": sample_t1b_goal,
    "t1c_endpoint_reposition": sample_t1c_goal,
}


def sample_t1_goal(task: str, X_current: np.ndarray, rng: np.random.Generator) -> DualGoal:
    """Sample the T1 goal for ``task`` given the current centerline."""

    key = str(task)
    if key not in _SAMPLERS:
        raise ValueError(f"unknown T1 task {task!r}; expected one of {T1_TASKS}")
    return _SAMPLERS[key](X_current, rng)


__all__ = [
    "T1B_ARC_ANGLE_RANGE",
    "T1C_DISPLACEMENT_RANGE",
    "T1_TASKS",
    "arc_template",
    "sample_t1_goal",
    "sample_t1a_goal",
    "sample_t1b_goal",
    "sample_t1c_goal",
]
