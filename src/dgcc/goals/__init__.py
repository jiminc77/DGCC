"""Dual-goal representation and distance helpers for DGCC."""

from dgcc.goals.distance import D, chamfer, chamfer_distance
from dgcc.goals.dual_goal import (
    ANCHOR_CHANNEL_COUNT,
    ANCHOR_MODES,
    CG_DIM,
    SHAPE_CHANNEL_COUNT,
    SHAPE_CHANNEL_INDICES,
    TEMPLATE_NAMES,
    DualGoal,
    anchor_of,
    c_g,
    compute_c_g,
    goal_curve,
    make_goal,
    make_shape_template,
    template_library,
)

__all__ = [
    "ANCHOR_CHANNEL_COUNT",
    "ANCHOR_MODES",
    "CG_DIM",
    "D",
    "DualGoal",
    "SHAPE_CHANNEL_COUNT",
    "SHAPE_CHANNEL_INDICES",
    "TEMPLATE_NAMES",
    "anchor_of",
    "c_g",
    "chamfer",
    "chamfer_distance",
    "compute_c_g",
    "goal_curve",
    "make_goal",
    "make_shape_template",
    "template_library",
]
