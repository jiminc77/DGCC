from __future__ import annotations

import numpy as np
import pytest

from dgcc.phi.dct import M, Phi_DCT, phi_mode0_indices, phi_shape_indices
from dgcc.phi.normalize import DmNormalizer, normalize
from dgcc.phi.resample import resample


def _s_curve(n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    return np.column_stack(
        [
            t - 0.5,
            0.12 * np.sin(2.0 * np.pi * (t - 0.5)),
            0.04 * np.sin(np.pi * t),
        ]
    )


def _straight(length: float = 1.0, n: int = 200) -> np.ndarray:
    s = np.linspace(-0.5 * length, 0.5 * length, n)
    return np.column_stack([s, np.zeros_like(s), np.zeros_like(s)])


def _arc_same_length(length: float = 1.0, angle: float = np.pi / 2.0, n: int = 200) -> np.ndarray:
    s = np.linspace(-0.5 * length, 0.5 * length, n)
    radius = length / angle
    theta = s / radius
    return np.column_stack(
        [
            radius * np.sin(theta),
            radius * (1.0 - np.cos(theta)),
            np.zeros_like(s),
        ]
    )


def test_translation_delta_m_changes_only_mode0_channels() -> None:
    X_before = resample(_s_curve(75))
    translation = np.array([0.1, -0.2, 0.05])
    X_after = X_before + translation

    delta_m = Phi_DCT(X_after) - Phi_DCT(X_before)

    assert np.allclose(delta_m[phi_mode0_indices()], translation)
    assert np.allclose(delta_m[phi_shape_indices()], 0.0, atol=1.0e-12)


def test_uniform_bend_delta_m_is_low_mode_dominated() -> None:
    X_before = resample(_straight())
    X_after = resample(_arc_same_length())

    delta_m = Phi_DCT(X_after) - Phi_DCT(X_before)
    x_modes = delta_m[:M]
    y_modes = delta_m[M : 2 * M]
    z_modes = delta_m[2 * M :]

    low_bend_energy = np.linalg.norm(np.concatenate([x_modes[1:3], y_modes[1:3]]))
    high_bend_energy = np.linalg.norm(np.concatenate([x_modes[5:8], y_modes[5:8]]))

    assert low_bend_energy > 5.0 * high_bend_energy
    assert np.allclose(z_modes, 0.0, atol=1.0e-12)


def test_normalizer_round_trip_and_mode0_passthrough(tmp_path) -> None:
    rng = np.random.default_rng(1234)
    delta_ms = rng.normal(size=(16, 3 * M))
    delta_ms[:, phi_mode0_indices()] = 0.0
    normalizer = DmNormalizer.fit(delta_ms)

    delta_m = rng.normal(size=3 * M)
    normalized = normalizer.normalize(delta_m)
    via_function = normalize(delta_m, normalizer.stats.to_dict())

    assert np.allclose(normalized, via_function)
    assert np.allclose(normalized[phi_mode0_indices()], delta_m[phi_mode0_indices()])
    assert np.allclose(
        normalized[phi_shape_indices()],
        delta_m[phi_shape_indices()] / normalizer.stats.std[phi_shape_indices()],
    )

    path = tmp_path / "dm_stats.json"
    normalizer.save(path)
    loaded = DmNormalizer.load(path)

    assert loaded.stats.channel_layout_id == normalizer.stats.channel_layout_id
    assert loaded.stats.M == normalizer.stats.M
    assert loaded.stats.fit_count == normalizer.stats.fit_count
    assert np.allclose(loaded.stats.std, normalizer.stats.std)
    assert np.allclose(loaded.normalize(delta_m), normalized)


def test_normalizer_rejects_tiny_shape_std() -> None:
    rng = np.random.default_rng(5678)
    delta_ms = rng.normal(size=(8, 3 * M))
    delta_ms[:, phi_shape_indices()[4]] = 1.0

    with pytest.raises(ValueError, match="std for scaled mode>=1"):
        DmNormalizer.fit(delta_ms)


def test_delta_m_of_identical_shapes_is_zero() -> None:
    X = resample(_s_curve(80))

    delta_m = Phi_DCT(X) - Phi_DCT(X)

    assert np.allclose(delta_m, 0.0)
