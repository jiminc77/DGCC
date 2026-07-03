from __future__ import annotations

import numpy as np
import pytest

from dgcc.phi.dct import M, Phi_DCT, phi_mode0_indices, phi_shape_indices
from dgcc.phi.resample import resample


NS = (25, 50, 100)
REL_TOL = 0.02
# DCT units are meters with norm='ortho'.  A 2 cm absolute denominator floor
# keeps numerically tiny high modes from dominating the relative-error check.
ABS_DENOM_FLOOR = 2.0e-2


def _semicircle(n: int) -> np.ndarray:
    theta = np.linspace(0.0, np.pi, n)
    radius = 0.5
    return np.column_stack(
        [
            radius * np.cos(theta),
            radius * np.sin(theta),
            np.zeros_like(theta),
        ]
    )


def _s_curve(n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    return np.column_stack(
        [
            t - 0.5,
            0.15 * np.sin(2.0 * np.pi * (t - 0.5)),
            0.08 * np.sin(np.pi * t),
        ]
    )


def _phis(shape_fn) -> dict[int, np.ndarray]:
    return {n: Phi_DCT(resample(shape_fn(n))) for n in NS}


@pytest.mark.parametrize("shape_fn", [_semicircle, _s_curve])
def test_phi_shape_modes_are_invariant_to_input_discretization(shape_fn) -> None:
    phis = _phis(shape_fn)
    ref = phis[100]
    shape_idx = phi_shape_indices()

    for n in (25, 50):
        denom = np.maximum(np.abs(ref[shape_idx]), ABS_DENOM_FLOOR)
        rel_error = np.abs(phis[n][shape_idx] - ref[shape_idx]) / denom
        assert float(rel_error.max()) < REL_TOL


def test_mode0_channels_are_per_axis_means() -> None:
    X = resample(_s_curve(50))
    phi = Phi_DCT(X)

    assert np.allclose(phi[phi_mode0_indices()], X.mean(axis=0))
    assert phi.shape == (3 * M,)


def test_translation_moves_only_mode0_channels() -> None:
    X = resample(_semicircle(50))
    translation = np.array([0.25, -0.1, 0.075])

    delta_phi = Phi_DCT(X + translation) - Phi_DCT(X)

    assert np.allclose(delta_phi[phi_mode0_indices()], translation)
    assert np.allclose(delta_phi[phi_shape_indices()], 0.0, atol=1.0e-12)


def test_phi_dct_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="shape"):
        Phi_DCT(np.zeros((31, 3)))

    bad = np.zeros((32, 3))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        Phi_DCT(bad)
