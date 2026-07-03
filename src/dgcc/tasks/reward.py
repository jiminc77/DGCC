"""P1 reward and success judgment on the immutable §8 metric.

Metric routing covenant (inherited risk #4, P1.md §2 rule 4):
reward AND success judgment route exclusively through
:func:`dgcc.goals.distance.correspondence_l2` (length-normalized
correspondence L2 with orientation canonicalization).  The legacy Chamfer
family (``distance.D``, ``distance.chamfer``, ``distance.chamfer_distance``)
is REPORT-ONLY and must never appear on this code path.  Reimplementation of
the metric is forbidden — this module only composes the P0 function.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

# The ONLY metric import allowed on the reward/success path.
from dgcc.goals.distance import correspondence_l2
from dgcc.goals.dual_goal import DualGoal, goal_curve

from dgcc.tasks.domain import EPS_SUCC_COEFF, RewardConstants


def distance_to_goal(
    X: np.ndarray,
    goal: DualGoal | Mapping[str, Any],
    length_m: float,
) -> float:
    """Return the immutable length-normalized D(X, G).

    Both success judgment and reward call this single function so the two can
    never diverge (same D, same code path).
    """

    return correspondence_l2(X, goal_curve(goal, length_m), length_m)


def is_success(d_normalized: float, length_m: float) -> bool:
    """Return the ε_succ = 0.05·L success judgment.

    ``d_normalized`` is the length-normalized D from
    :func:`distance_to_goal`; the raw-distance test
    ``D_raw < 0.05·L`` is algebraically identical to
    ``d_normalized < 0.05`` because ``D_raw = d_normalized · L``.
    """

    return float(d_normalized) * float(length_m) < EPS_SUCC_COEFF * float(length_m)


def step_reward(
    d_before: float,
    d_after: float,
    length_m: float,
    constants: RewardConstants,
) -> tuple[float, bool]:
    """Return ``(r_t, success)`` for one primitive transition.

    ``r_t = α[D(X_t,G) − D(X_{t+1},G)] − c_step + 1{D(X_{t+1},G) < ε_succ}·R_succ``
    with D values already length-normalized by :func:`distance_to_goal`.
    """

    success = is_success(d_after, length_m)
    reward = (
        constants.alpha * (float(d_before) - float(d_after))
        - constants.c_step
        + (constants.r_succ if success else 0.0)
    )
    return float(reward), bool(success)


__all__ = ["distance_to_goal", "is_success", "step_reward"]
