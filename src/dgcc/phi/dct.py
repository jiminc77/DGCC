"""DCT-II rope deformation feature map for P0-M4.

``Phi_DCT`` consumes the canonical 32-node centerline produced by
``dgcc.phi.resample.resample`` and returns a stable 24-channel feature vector
with axis-major layout ``[x0..x7, y0..y7, z0..z7]``.  Mode 0 for each axis is
stored as the coordinate mean (centroid component); modes 1..7 are the
orthonormal DCT-II low-frequency shape coefficients.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct

from dgcc.phi.resample import K as RESAMPLED_POINTS


M = 8
"""Number of low-order DCT modes retained per axis."""

PHI_DIM = 3 * M
"""Length of the flattened ``Phi_DCT`` vector."""

DCT_TYPE = 2
DCT_NORM = "ortho"
CHANNEL_LAYOUT_ID = "axis-major-xyz-modes-0-7-v1"
CHANNEL_LAYOUT = tuple(
    f"{axis}{mode}" for axis in ("x", "y", "z") for mode in range(M)
)
"""Stable channel order: ``[x0..x7, y0..y7, z0..z7]``."""


def phi_mode0_indices() -> np.ndarray:
    """Return indices of the three centroid channels ``[x0, y0, z0]``."""

    return np.array([0, M, 2 * M], dtype=int)


def phi_shape_indices() -> np.ndarray:
    """Return indices of the 21 mode>=1 channels used by downstream shape code."""

    mask = np.ones(PHI_DIM, dtype=bool)
    mask[phi_mode0_indices()] = False
    return np.flatnonzero(mask)


def Phi_DCT(X: np.ndarray) -> np.ndarray:
    """Return the 24-channel §7 DCT feature vector for a ``(32, 3)`` centerline.

    The input must already be resampled to 32 points.  DCT-II with
    ``norm="ortho"`` is applied along the rope arc-length axis so fixed-size
    resampling removes dependence on the original discretization density.  The
    DCT mode-0 coefficient is replaced by the per-axis mean to make the three
    centroid channels explicit and to keep translations isolated to mode 0.
    """

    points = _validate_centerline(X)
    coeffs = dct(points, type=DCT_TYPE, norm=DCT_NORM, axis=0)[:M, :]
    coeffs[0, :] = points.mean(axis=0)
    return coeffs.T.reshape(PHI_DIM)


def _validate_centerline(X: np.ndarray) -> np.ndarray:
    try:
        points = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("X must be a finite float array with shape (32, 3)") from exc
    expected = (RESAMPLED_POINTS, 3)
    if points.shape != expected:
        raise ValueError(f"X must have shape {expected}, got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("X must contain only finite values")
    return points
