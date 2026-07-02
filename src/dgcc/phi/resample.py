"""Arc-length centerline resampling.

P0-M1 provides the minimal real ``resample(X_raw) -> (32, 3)`` implementation
needed by the MuJoCo smoke test: normalized arc-length uniform sampling with
linear interpolation. P0-M4 finalizes this module with the full invariance test
suite from the §7 deformation-feature specification.
"""

from __future__ import annotations

import numpy as np


K = 32


def resample(X_raw: np.ndarray) -> np.ndarray:
    """Return a 32-node normalized arc-length resampling of ``X_raw``.

    Args:
        X_raw: Rope centerline samples with shape ``(N, 3)``. ``N`` must be at
            least 2.

    Returns:
        A ``(32, 3)`` float array sampled uniformly in normalized arc length.
    """

    points = np.asarray(X_raw, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"X_raw must have shape (N, 3), got {points.shape}")
    if points.shape[0] < 2:
        raise ValueError("X_raw must contain at least 2 points")
    if not np.all(np.isfinite(points)):
        raise ValueError("X_raw contains non-finite values")

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = cumulative[-1]

    if total <= 0.0:
        raise ValueError(
            "X_raw has zero total arc length (all points coincide); "
            "a physical rope centerline cannot be degenerate"
        )

    normalized = cumulative / total
    targets = np.linspace(0.0, 1.0, K)
    out = np.empty((K, 3), dtype=float)
    for axis in range(3):
        out[:, axis] = np.interp(targets, normalized, points[:, axis])
    return out
