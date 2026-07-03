"""Length-normalized bidirectional Chamfer distance for dual goals."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.fft import dct

from dgcc.goals.dual_goal import DualGoal, arc_length, canonical_centerline, goal_curve
from dgcc.phi.dct import DCT_NORM, DCT_TYPE, M as DCT_MODE_COUNT
from dgcc.phi.resample import K as RESAMPLED_POINTS


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


def canonical_shape_flip(
    X_before: np.ndarray,
    goal: DualGoal | Mapping[str, Any],
    length_m: float,
    *,
    shape_modes: int = DCT_MODE_COUNT,
) -> bool:
    """Return the M5R2 fixed orientation decision for shape quantities.

    The decision is made once per transition from ``X_before`` only: canonicalize
    both orientations of ``X_before``, compare their mode-1..``shape_modes-1``
    DCT residuals against the goal curve's shape coefficients, and choose the
    flipped orientation only when its residual is strictly smaller.  The returned
    decision is then applied identically to before/after curves for both
    ``D_shape`` and ``c_g_shape`` measurements.
    """

    length = _validate_length(length_m)
    shape_modes = _validate_shape_modes(shape_modes)
    goal_shape = _canonical_shape(goal_curve(goal, length))
    before_identity = _canonical_shape(X_before, flip=False)
    before_flipped = _canonical_shape(X_before, flip=True)

    goal_coeffs = _shape_coeff_vector(goal_shape, shape_modes)
    identity_norm = float(np.linalg.norm(goal_coeffs - _shape_coeff_vector(before_identity, shape_modes)))
    flipped_norm = float(np.linalg.norm(goal_coeffs - _shape_coeff_vector(before_flipped, shape_modes)))
    return bool(flipped_norm < identity_norm)


def flip_consistent_shape_distance(
    X: np.ndarray,
    G_curve: np.ndarray,
    length_m: float,
    *,
    flip: bool,
) -> float:
    """Return centroid-removed correspondence L2 using a caller-fixed flip."""

    length = _validate_length(length_m)
    x = _canonical_shape(X, flip=flip)
    g = _canonical_shape(G_curve, flip=False)
    return _rms_pointwise_l2(x, g) / length


def flip_consistent_shape_cg_norm(
    X: np.ndarray,
    goal: DualGoal | Mapping[str, Any],
    length_m: float,
    *,
    flip: bool,
    shape_modes: int = DCT_MODE_COUNT,
) -> float:
    """Return ``||c_g_shape||`` under the fixed M5R2 orientation convention."""

    length = _validate_length(length_m)
    shape_modes = _validate_shape_modes(shape_modes)
    goal_shape = _canonical_shape(goal_curve(goal, length))
    x_shape = _canonical_shape(X, flip=flip)
    residual = _shape_coeff_vector(goal_shape, shape_modes) - _shape_coeff_vector(x_shape, shape_modes)
    return float(np.linalg.norm(residual))


def flip_consistent_shape_measurements(
    X_before: np.ndarray,
    X_after: np.ndarray,
    goal: DualGoal | Mapping[str, Any],
    length_m: float,
    *,
    shape_modes: int = DCT_MODE_COUNT,
) -> dict[str, float | bool]:
    """Return before/after fixed-orientation shape distances and norms."""

    length = _validate_length(length_m)
    flip = canonical_shape_flip(X_before, goal, length, shape_modes=shape_modes)
    g_curve = goal_curve(goal, length)

    d_before = flip_consistent_shape_distance(X_before, g_curve, length, flip=flip)
    d_after = flip_consistent_shape_distance(X_after, g_curve, length, flip=flip)
    cg_before = flip_consistent_shape_cg_norm(X_before, goal, length, flip=flip, shape_modes=shape_modes)
    cg_after = flip_consistent_shape_cg_norm(X_after, goal, length, flip=flip, shape_modes=shape_modes)

    return {
        "flip": flip,
        "D_shape_before": d_before,
        "D_shape_after": d_after,
        "c_g_shape_norm_before": cg_before,
        "c_g_shape_norm_after": cg_after,
        "delta_D_shape": d_after - d_before,
        "delta_c_g_shape_norm": cg_after - cg_before,
    }


def _rms_pointwise_l2(X: np.ndarray, Y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((X - Y) ** 2, axis=1))))


def _canonical_shape(X: np.ndarray, *, flip: bool = False) -> np.ndarray:
    curve = canonical_centerline(_validate_centerline("X", X))
    if flip:
        curve = curve[::-1].copy()
    return curve - curve.mean(axis=0, keepdims=True)


def _shape_coeff_vector(shape_curve: np.ndarray, shape_modes: int) -> np.ndarray:
    coeffs = dct(shape_curve, type=DCT_TYPE, norm=DCT_NORM, axis=0)
    return coeffs[1:shape_modes, :].T.reshape(3 * (shape_modes - 1))


def _validate_shape_modes(shape_modes: int) -> int:
    modes = int(shape_modes)
    if modes < 2:
        raise ValueError("shape_modes must include at least mode 1")
    if modes > RESAMPLED_POINTS:
        raise ValueError("shape_modes cannot exceed the canonical point count")
    return modes



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


__all__ = [
    "D",
    "canonical_shape_flip",
    "chamfer",
    "chamfer_distance",
    "correspondence_l2",
    "flip_consistent_shape_cg_norm",
    "flip_consistent_shape_distance",
    "flip_consistent_shape_measurements",
]
