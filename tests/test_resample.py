from __future__ import annotations

import numpy as np

from dgcc.phi.resample import resample


def test_resample_straight_line_preserves_endpoints_and_uniform_spacing() -> None:
    x = np.linspace(0.0, 1.0, 50)
    X_raw = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])

    X = resample(X_raw)

    assert X.shape == (32, 3)
    assert np.allclose(X[0], X_raw[0])
    assert np.allclose(X[-1], X_raw[-1])
    spacing = np.linalg.norm(np.diff(X, axis=0), axis=1)
    assert np.allclose(spacing, spacing[0])


def test_resample_25_points_returns_32_points() -> None:
    t = np.linspace(0.0, 1.0, 25)
    X_raw = np.column_stack([t, t**2, np.zeros_like(t)])

    X = resample(X_raw)

    assert X.shape == (32, 3)
