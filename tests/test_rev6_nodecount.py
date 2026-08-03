"""Rev 6 node-count transition: C1 action mapping, C2 mass, C3 grasp arc, C4.

Pilot: `dossier/pilot_report_nodecount.md` §4.  The governing design principle
is D2 — **every change must be byte-identical at n_segments == 32 with the
0.032 kg rope** — so the n=32 assertions here are equality, not tolerance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dgcc.envs.base import ROPE_MASS_TOTAL_KG_BASE, RopeParams  # noqa: E402
from dgcc.envs.dlolab import (  # noqa: E402
    GRASP_NOISE_ARC_FRACTION,
    PILE_NEIGHBOR_RADIUS_M,
    arc_length_vertex_index,
    grasp_noise_choices,
    grasp_noise_offset_nodes,
    mapped_parameters,
    sample_grasp,
    segment_mass_kg,
    stiffness_bases,
)

K = 32
LEGACY_SEGMENT_MASS = 1.0e-3


def rope(n_segments: int, mass_kg: float = ROPE_MASS_TOTAL_KG_BASE) -> RopeParams:
    return RopeParams(
        length_m=1.0,
        n_segments=n_segments,
        bend_stiffness=1.0,
        twist_stiffness=1.0,
        friction=1.0,
        radius=0.005,
        rope_mass_total_kg=mass_kg,
    )


def straight(n_envs: int, n: int) -> np.ndarray:
    """Rest-state centerline: uniform spacing over x in [-0.5, 0.5]."""

    line = np.zeros((n, 3))
    line[:, 0] = np.linspace(-0.5, 0.5, n)
    return np.broadcast_to(line, (n_envs, n, 3)).copy()


# ------------------------------------------------------- C1 action mapping


def test_c1_identity_for_every_policy_index_at_n32() -> None:
    """Pilot C1 test (i): n == K must be the exact identity, for all p."""

    raw = straight(1, K)
    for p in range(K):
        mapped = arc_length_vertex_index(raw, np.array([p]), K)
        assert mapped.tolist() == [p]


def test_c1_identity_branch_ignores_deformation_at_n32() -> None:
    """The identity is a BRANCH, not a numerical coincidence: a badly
    non-uniform n=32 shape must still map p -> p exactly."""

    raw = straight(1, K)
    raw[0, :, 0] = np.linspace(-0.5, 0.5, K) ** 3  # heavily non-uniform spacing
    mapped = arc_length_vertex_index(raw, np.arange(K), K)
    assert mapped.tolist() == list(range(K))


@pytest.mark.parametrize("n", [64, 100])
def test_c1_endpoints_and_full_reach_at_refined_meshes(n: int) -> None:
    """Pilot C1 test (ii): p=K-1 reaches the LAST vertex and no part of the
    rope is unreachable.  This is the defect the pilot measured — at n=64 the
    old code reached only 49.2% of the rope, at n=100 only 31.3%."""

    raw = straight(1, n)
    mapped = arc_length_vertex_index(raw, np.arange(K), K)
    assert mapped[0] == 0
    assert mapped[-1] == n - 1
    # Reachable span covers the whole rope, and the mapping is a bijection onto
    # an arc-length-uniform subset (strictly increasing).
    assert np.all(np.diff(mapped) > 0)
    reach = mapped[-1] / (n - 1)
    assert reach == 1.0


@pytest.mark.parametrize("n", [64, 100])
def test_c1_is_arc_length_monotone_on_a_deformed_shape(n: int) -> None:
    """Pilot C1 test (iii): on a deformed shape the mapping stays monotone in
    arc length and lands near the requested normalized arc position."""

    t = np.linspace(0.0, 1.0, n)
    raw = np.zeros((1, n, 3))
    raw[0, :, 0] = 0.4 * np.sin(2.0 * np.pi * t)
    raw[0, :, 1] = 0.3 * np.cos(3.0 * np.pi * t)
    raw[0, :, 2] = 0.05 * t

    mapped = arc_length_vertex_index(raw, np.arange(K), K)
    assert np.all(np.diff(mapped) >= 0)

    edges = np.linalg.norm(np.diff(raw[0], axis=0), axis=-1)
    s = np.concatenate([[0.0], np.cumsum(edges)])
    s /= s[-1]
    requested = np.arange(K) / (K - 1)
    # Nearest-vertex quantization error is bounded by half the LARGEST local
    # arc interval — on a deformed shape the vertex spacing is not uniform, so
    # half the AVERAGE interval is not a valid bound.
    assert np.max(np.abs(s[mapped] - requested)) <= 0.5 * np.max(np.diff(s)) + 1e-12


def test_c1_maps_each_environment_independently() -> None:
    """Batched envs hold different deformed shapes, so the mapping must be
    computed per env — a shared env-0 mapping would grasp the wrong material
    point in every other env."""

    n = 64
    raw = np.zeros((2, n, 3))
    raw[0, :, 0] = np.linspace(-0.5, 0.5, n)
    # env 1: all its arc length is packed into the first half of the vertices.
    raw[1, : n // 2, 0] = np.linspace(-0.5, 0.5, n // 2)
    raw[1, n // 2 :, 0] = 0.5

    mapped = arc_length_vertex_index(raw, np.full(2, K // 2), K)
    assert mapped[0] != mapped[1]
    assert mapped[1] < n // 2


def test_c1_rejects_out_of_range_policy_indices() -> None:
    raw = straight(1, 64)
    with pytest.raises(IndexError):
        arc_length_vertex_index(raw, np.array([K]), K)
    with pytest.raises(IndexError):
        arc_length_vertex_index(raw, np.array([-1]), K)


# ------------------------------------------------------------- C2 mass model


def test_c2_segment_mass_is_byte_identical_at_the_historical_domain() -> None:
    params = rope(32, 0.032)
    assert segment_mass_kg(params) == LEGACY_SEGMENT_MASS
    assert mapped_parameters(params)["segment_mass"] == LEGACY_SEGMENT_MASS
    assert stiffness_bases(params)["segment_mass_base"] == LEGACY_SEGMENT_MASS
    assert stiffness_bases()["segment_mass_base"] == LEGACY_SEGMENT_MASS


@pytest.mark.parametrize("n", [32, 64, 100])
def test_c2_total_mass_is_invariant_under_discretization(n: int) -> None:
    """The whole point of C2: refining the mesh must not change the rope."""

    params = rope(n, 0.032)
    assert segment_mass_kg(params) * n == pytest.approx(0.032, rel=0, abs=1e-15)


def test_c2_total_mass_is_carried_through_the_mapped_parameters() -> None:
    params = rope(64, 0.040)
    mapped = mapped_parameters(params)
    assert mapped["rope_mass_total_kg"] == 0.040
    assert mapped["segment_mass"] == pytest.approx(0.040 / 64)


def test_c2_rejects_a_nonphysical_mass() -> None:
    with pytest.raises(ValueError):
        segment_mass_kg(rope(32, 0.0))


# ------------------------------------------------------- C3 grasp noise arc


def test_c3_offset_is_one_vertex_at_n32() -> None:
    assert grasp_noise_offset_nodes(32) == 1
    assert grasp_noise_choices(32) == (-1, 0, 1)


@pytest.mark.parametrize("n,expected", [(32, 1), (64, 2), (100, 3)])
def test_c3_offset_tracks_the_fixed_arc_length(n: int, expected: int) -> None:
    assert grasp_noise_offset_nodes(n) == expected
    # The realized error stays within 10% of the pinned L/31 arc length.
    realized = expected / (n - 1)
    assert abs(realized - GRASP_NOISE_ARC_FRACTION) / GRASP_NOISE_ARC_FRACTION < 0.10


def test_c3_sample_grasp_draw_sequence_is_byte_identical_at_n32() -> None:
    """The RNG stream and every drawn value must match the pre-Rev-6 model."""

    legacy_choices = (-1, 0, 1)

    def legacy(p: int, n: int, rng: np.random.Generator) -> tuple[int, bool]:
        offset = int(rng.choice(legacy_choices))
        return int(np.clip(p + offset, 0, n - 1)), bool(rng.random() >= 0.05)

    a = np.random.default_rng(20260803)
    b = np.random.default_rng(20260803)
    for p in [0, 1, 15, 16, 30, 31]:
        for _ in range(200):
            assert sample_grasp(p, 32, a, True) == legacy(p, 32, b)


def test_c3_noise_never_degenerates_to_zero() -> None:
    for n in (2, 3, 8, 32, 64, 100, 512):
        assert grasp_noise_offset_nodes(n) >= 1


def test_c3_boundary_clamping_is_preserved() -> None:
    rng = np.random.default_rng(7)
    for _ in range(300):
        node, _ = sample_grasp(0, 64, rng, True)
        assert 0 <= node < 64
    rng = np.random.default_rng(7)
    for _ in range(300):
        node, _ = sample_grasp(63, 64, rng, True)
        assert 0 <= node < 64


# --------------------------------------------------------------- C4 constants


def test_c4_pile_radius_is_an_absolute_length_and_unchanged() -> None:
    assert PILE_NEIGHBOR_RADIUS_M == 0.065
