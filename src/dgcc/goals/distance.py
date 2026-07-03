"""Length-normalized bidirectional Chamfer distance for dual goals."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from dgcc.goals.dual_goal import DualGoal, arc_length, canonical_centerline, goal_curve


def chamfer(X: np.ndarray, Y: np.ndarray) -> float:
    """Return symmetric bidirectional mean-of-min point distance.

    The result is ``0.5 * (mean_x min_y ||x-y|| + mean_y min_x ||y-x||)`` and
    is therefore independent of which point cloud is passed first.  Inputs may
    have different point counts, but each must be a finite non-empty ``(N, 3)``
    array.
    """

    x = _validate_point_cloud("X", X)
    y = _validate_point_cloud("Y", Y)
    distances = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2)
    x_to_y = distances.min(axis=1).mean()
    y_to_x = distances.min(axis=0).mean()
    return float(0.5 * (x_to_y + y_to_x))


def D(X: np.ndarray, goal: DualGoal | Mapping[str, Any], length_m: float) -> float:
    """Return §8 length-normalized distance from ``X`` to ``goal``.

    ``X`` is validated as a finite non-degenerate centerline.  The goal is
    reconstructed with :func:`dgcc.goals.dual_goal.goal_curve`, and the raw
    Chamfer distance is divided by ``length_m``.
    """

    x = _validate_centerline("X", X)
    length = _validate_length(length_m)
    return chamfer(x, goal_curve(goal, length)) / length


def chamfer_distance(X: np.ndarray, G: np.ndarray, length_m: float) -> float:
    """Backward-compatible length-normalized Chamfer helper."""

    length = _validate_length(length_m)
    return chamfer(X, G) / length

def correspondence_l2(
    X: np.ndarray,
    G_curve: np.ndarray,
    length_m: float,
    *,
    shape_only: bool = False,
) -> float:
    """Return amended §8 length-normalized correspondence L2 distance.

    Both inputs are canonicalized through the existing 32-node arc-length
    resampling path.  The pointwise RMS L2 correspondence is evaluated for the
    identity orientation and the reversed goal orientation; the smaller value is
    divided by ``length_m``.  When ``shape_only`` is true, each canonical curve's
    centroid is removed before the orientation comparison.
    """

    length = _validate_length(length_m)
    x = canonical_centerline(_validate_centerline("X", X))
    g = canonical_centerline(_validate_centerline("G_curve", G_curve))
    if shape_only:
        x = x - x.mean(axis=0, keepdims=True)
        g = g - g.mean(axis=0, keepdims=True)

    identity = _rms_pointwise_l2(x, g)
    flipped = _rms_pointwise_l2(x, g[::-1])
    return min(identity, flipped) / length


def _rms_pointwise_l2(X: np.ndarray, Y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((X - Y) ** 2, axis=1))))



def _validate_point_cloud(name: str, value: np.ndarray) -> np.ndarray:
    try:
        points = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite float array with shape (N, 3)") from exc
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {points.shape}")
    if points.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one point")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain only finite values")
    return points


def _validate_centerline(name: str, value: np.ndarray) -> np.ndarray:
    points = _validate_point_cloud(name, value)
    if points.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two points")
    if arc_length(points) <= 0.0:
        raise ValueError(f"{name} must have non-zero arc length")
    return points


def _validate_length(length_m: float) -> float:
    length = float(length_m)
    if length <= 0.0 or not np.isfinite(length):
        raise ValueError("length_m must be a positive finite float")
    return length


__all__ = ["D", "chamfer", "chamfer_distance", "correspondence_l2"]
