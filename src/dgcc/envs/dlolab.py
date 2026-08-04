"""DLO-Lab/Genesis rod adapter for the P0-M3 primary primitive milestone."""

from __future__ import annotations

import os
from dataclasses import asdict
from math import ceil
from typing import Any, Sequence

import numpy as np

from dgcc.envs.base import ROPE_MASS_TOTAL_KG_BASE, DLOEnvBase, RopeParams
from dgcc.utils.seeding import seed_everything

# --- Rev 10 real-anchor calibration (owner-run harness, adopted 2026-08-04) ---
# These three bases WERE the P0 placeholder values 8.0e5 / 1.0e5 / 1.0e4.  They
# are now the values fitted against the owner's benchtop rope
# (dossier/V2_rope_calibration/fit_real.json, harness
# `scripts/v2_rope_calibration_harness.py`), at the domain this repository
# actually launches: n=64 nodes, 0.040 kg, dt 1e-3, substeps 5.
#
# BEND_BASE 421696.50 -- bisection on the 137 mm droop test (chord angle
#   41.5 deg).  Four-point check against the measured overhang sweep, sim vs
#   real chord angle: 5 cm 3.9/2.1, 10 cm 18.5/26, 15 cm 40.9/47, 20 cm
#   63.1/60 deg (RMS 5.2 deg against a +-2..4 deg measurement uncertainty).
#   The single-point fit is accepted and the remaining shape error is
#   PUBLISHED as a residual instead of being tuned away.
# STRETCH_BASE 2.4e6 -- owner decision (1): 80% of the MEASURED integrator
#   stability ceiling (3.0e6 at substeps=5, dossier/V2_rope_calibration/
#   stability_n64*.json).  The real rope's EA is 1962 N, which would need
#   K ~ 2.2e7 -- unreachable at substeps=5.  The sim rope is therefore ~11.6x
#   more compliant in axial stretch than the real one; that gap is QUANTIFIED
#   in the Rev 10 residual-gap table rather than hidden behind a fitted number.
# TWIST_BASE 42169.65 -- no torsion benchtop test exists, so G is carried at
#   the SAME multiplier as E (4.216965x its P0 base): an isotropy ASSUMPTION,
#   recorded as one.  The order's "42,170" is this value rounded.
#
# Damping (`rod_damping` / `rod_angular_damping`, configs/v2_t2.yaml) is NOT
# recalibrated and stays 10 / 5.  The drop-arrest test's arrest time is
# non-monotone in gamma (10/30/60/120 -> 1.369/2.204/1.828/1.337 s at the
# 0.02 m/s observation threshold), i.e. that test is dominated by fall-phase
# delay plus post-touchdown contact, not by linear damping, and does not
# constrain gamma at all.  The 0.48 s measured arrest is an OPEN residual;
# moving gamma without a constraining test is forbidden (Rev 10 limitation).
STRETCH_BASE = 2.4e6
BEND_BASE = 421696.5034285822
TWIST_BASE = 42169.65034285822
MU_S_BASE = 0.30
MU_K_RATIO = 0.80
# Rev 6 C2 (mass generalization, pilot §4 C2).  Segment mass used to be the
# fixed constant 1.0e-3 kg, which made the TOTAL rope mass a function of the
# discretization: 32 nodes = 32 g, 64 nodes = 64 g, 100 nodes = 100 g.  The
# pilot (§3.2 table C) showed that every "node count" physics comparison run
# against that model is really a mass comparison.  Mass is now a rope property
# (`RopeParams.rope_mass_total_kg`) and the segment mass is DERIVED, so
# changing the discretization is a pure refinement.  The base value 0.032
# reproduces the historical model exactly at n=32 (0.032/32 = 1.0e-3) —
# byte-identical by construction.
MAX_DELTA_NORM = 0.15
# R6: with horizontal δ (R4) the `lift` height alone decides the transit
# height — δz no longer dominates it (design §5.1 R6; historically a
# lift="low" grasp could reach z≈0.14 because δz was added on top).
LIFT_HEIGHTS = {"low": 0.02, "high": 0.15}
# R10: gripper parking height used by reset placement.  Low enough that a
# regressed residual attachment drags its node near the ground instead of
# 0.15 m mid-air (defense in depth for D1), but still 6x the rope resting
# height so the AT-14 parking-z signature (+-2 mm) stays discriminative.
# Historical artifact scans must keep using the OLD constant 0.15
# (= LIFT_HEIGHTS["high"]) — design §5.6 note 9.
GRIPPER_PARK_Z = 0.03
# P9 Rev 3 pile-aware lowering constants (adjudication §3.3, owner pin O1):
# clearance ε above the pile and the strain fail-safe trigger at half the AT-4
# threshold.
# Rev 6 C4 (pilot §4 C4): the neighbourhood radius is an ABSOLUTE geometric
# length — the guard's job is "do not drive the gripper down onto an existing
# pile", which is a distance in metres, not a count of segments.  The former
# comment derived it as "≈ 2 segment intervals"; that derivation is removed so
# a discretization change cannot be read as a reason to move the value.  Value
# unchanged (n=32 behaviour identical).
PILE_NEIGHBOR_RADIUS_M = 0.065
PILE_CLEARANCE_M = 0.010
LOWER_STRAIN_ABORT = 0.005
# Tension-guard slack-resolution budget (Rev 3, owner-approved 2026-08-02):
# a paused lift/translate env waits at most this many steps for its strain
# to relax before it is frozen for the rest of the walk.
TENSION_PAUSE_MAX_STEPS = 500
# Realized hold-quiescence threshold (orchestrator directive 2026-08-02 item
# 4): "quiescent" for hold-before-release means the release-speed acceptance
# criterion (AT-3, 0.05 m/s), not the 1e-3 settle threshold.
HOLD_QUIESCENT_VEL = 0.05
# --- Rev 5 two-stage compliant approach (owner-approved 2026-08-03) -------
# Rev 4 and earlier TELEPORTED the gripper onto the target node in a single
# scene step and attached on the next one, so the 0.15 m/s cap only ever
# applied AFTER the rope was already held.  The contact transient this
# produced is visible in rollout video and is not reproducible on a real
# arm.  Rev 5 replaces the teleport with a speed-limited two-stage approach
# plus a relative-velocity attach gate.  Parameter basis:
#
# APPROACH_V_FAST -- free-space transit speed.  0.75 m/s is 5x the
#   quasi-static manipulation cap and well inside the Cartesian envelope of
#   the 7-DOF arms this policy targets (Franka Panda TCP limit 1.7 m/s), so
#   the transit leg is transferable instead of instantaneous.
# APPROACH_R_SLOW -- radius of the final-approach zone, inside which the
#   commanded speed never exceeds the manipulation speed.  0.04 m is ~1.24
#   rope segment intervals (L/(N-1) = 32.3 mm at the P1 domain), i.e. the
#   gripper is already at manipulation speed a full segment before contact,
#   and it sits below the PILE_NEIGHBOR_RADIUS_M = 0.065 m neighbourhood the
#   lowering guard reasons over.
# APPROACH_A_DEC -- Cartesian deceleration limit used to blend the two
#   stages.  7 m/s^2 is about half the Panda's translational acceleration
#   limit (13 m/s^2), so the profile
#     v(d) = clip(sqrt(v_slow^2 + 2*a*(d - r_slow)), v_slow, v_fast)
#   is realizable by the real arm rather than a step discontinuity.  The
#   braking band is (v_fast^2 - v_slow^2)/(2a) = 38.6 mm wide.
# ATTACH_REL_VEL -- attach gate.  A rigid attach forces the node's velocity
#   to the gripper's, so it injects a velocity discontinuity equal to the
#   gripper-node RELATIVE speed.  The gate is one decade above the settle
#   threshold (SETTLE_VEL_THRESHOLD = 1e-3 m/s, the existing quiescence
#   scale) and 5x tighter than the realized release criterion
#   (HOLD_QUIESCENT_VEL = 0.05 m/s, AT-3): attach IMPOSES a discontinuity
#   where release only removes a constraint.  Energy bound: the injected
#   kinetic energy is <= 0.5*m_seg*v_rel^2 = 5e-8 J, i.e. <= 1e-5 of the
#   low-lift gravitational PE (6.3e-3 J) and five orders of magnitude under
#   the AT-1H KE/PE ceiling of 1.0.
# APPROACH_MAX_STEPS -- fail-closed budget.  Worst-case transit (gripper at
#   one rope end, target node at the other, ~1.1 m) costs ~1.7e3 steps, so
#   3000 leaves headroom while bounding a pathological dwell.  An env that
#   does not clear the gate inside the budget is reported as a GRASP FAILURE
#   and rewound through the Rev 4 snapshot path (no energy injection).
APPROACH_V_FAST = 0.75
APPROACH_V_SLOW = 0.15
APPROACH_R_SLOW = 0.04
APPROACH_A_DEC = 7.0
ATTACH_REL_VEL = 1.0e-2
APPROACH_MAX_STEPS = 3000
GRASP_FAILURE_PROB = 0.05
# Rev 6 C3 (pilot §4 C3, orchestrator technical judgement (b), owner veto
# reserved).  The M3 grasp-realism model is pinned as "±1 node" in
# `domain.py`'s immutable tier, but "1 node" is a DISCRETIZATION unit, not a
# physical one: at n=32 it is ±32.3 mm, at n=64 it would silently become
# ±15.9 mm and at n=100 ±10.1 mm.  A real gripper's placement error does not
# shrink because the simulator refines its mesh, so the invariant is restated
# as a fixed ARC LENGTH — the value it already had at n=32, L/31 — and the
# vertex offset that realizes it is derived per discretization.  At n=32 the
# derived offset is exactly 1, so the model is byte-identical there.
GRASP_NOISE_ARC_FRACTION = 1.0 / 31.0
VALID_INIT_SHAPES = frozenset({"straight", "u_bend", "s_curve", "random_smooth"})
# AT-1H final redefinition (adjudicator O2 + orchestrator, 2026-08-02,
# commit 8d17786): a primitive is a CEILING violation when any of
# v/strain/KE-PE exceeds AT1H_CEILING, and an ABSOLUTE-CAP violation when it
# exceeds AT1H_ABSOLUTE.  The gate is: ceiling rate <= AT1H_CEILING_RATE AND
# zero absolute-cap violations AND every ceiling violator ends clean.  These
# constants are duplicated from `scripts/v2_at1h_confirmatory_precheck.py`
# on purpose -- the training-time counters must be judged by the SAME
# numbers as the battery, so they live next to the adapter that produces
# them and any divergence is a one-line diff instead of a silent drift.
AT1H_GRAVITY = 9.81
AT1H_CEILING = {"v": 2.0, "strain": 0.02, "ke_over_pe": 1.0}
AT1H_ABSOLUTE = {"v": 10.0, "strain": 0.06, "ke_over_pe": 3.0}
AT1H_CEILING_RATE = 0.005
AT1H_CLEAN_TERMINAL_ARCLEN_DEV = 1.0e-3


def grasp_noise_offset_nodes(n_nodes: int) -> int:
    """Vertex offset that realizes ``GRASP_NOISE_ARC_FRACTION`` at ``n_nodes``.

    The rest-state vertex interval is ``L/(n-1)`` of the rope, so in
    length-normalized units one interval is ``1/(n-1)``.  The offset is that
    fixed arc fraction expressed in intervals and rounded to the nearest whole
    vertex (at least 1, so the noise model never silently degenerates to "no
    noise" on a coarse mesh).

    n=32 -> 1 (exactly the historical ±1 vertex, byte-identical)
    n=64 -> 2 (2/63 L = 31.7 mm vs the pinned 32.3 mm)
    n=100 -> 3 (3/99 L = 30.3 mm)
    """

    n = int(n_nodes)
    if n < 2:
        return 0
    interval = 1.0 / (n - 1)
    return max(1, int(round(GRASP_NOISE_ARC_FRACTION / interval)))


def grasp_noise_choices(n_nodes: int) -> tuple[int, int, int]:
    """The three-way ± offset draw, in vertices, for this discretization."""

    offset = grasp_noise_offset_nodes(n_nodes)
    return (-offset, 0, offset)


def arc_length_vertex_index(
    raw: np.ndarray, p: np.ndarray, k_nodes: int
) -> np.ndarray:
    """Rev 6 C1: map policy node indices to raw vertex indices by arc length.

    The policy acts on a ``k_nodes``-point arc-length-uniform resampling of the
    centerline (``DLOEnvBase.K``), while the solver stores ``n`` raw vertices
    whose spacing follows the deformed shape.  Before Rev 6 the adapter used
    the policy index AS a raw vertex index.  That is correct only when the two
    counts coincide; at n=64 the pilot measured (§3.3) that "grasp node 20"
    silently grasped 31.7% along the rope instead of 64.5%, and the rear half
    of the rope left the action space with no exception, warning or log line.

    Mapping: target arc length ``s* = p/(K-1) * S_total`` per environment (each
    env has its own deformed shape, so the cumulative arc length is computed
    per env), then the nearest vertex by ``|s_v - s*|``.  Boundaries are exact:
    ``p=0 -> vertex 0`` and ``p=K-1 -> vertex n-1``.

    ``n == k_nodes`` takes an EXPLICIT identity branch.  Arc-length
    correspondence is identity there to within float noise at the measured
    strain (<1e-3), but "to within float noise" does not prove byte-identity;
    the branch does (pilot design principle D2).

    Parameters
    ----------
    raw: ``(n_envs, n, 3)`` raw vertex positions.
    p: ``(n_envs,)`` policy node indices in ``[0, k_nodes)``.
    """

    policy = np.asarray(p, dtype=int)
    k = int(k_nodes)
    if k < 2:
        raise ValueError("k_nodes must be at least 2")
    if np.any((policy < 0) | (policy >= k)):
        raise IndexError(f"policy node indices must lie inside [0, {k})")
    vertices = np.asarray(raw, dtype=float)
    if vertices.ndim != 3 or vertices.shape[2] != 3:
        raise ValueError(f"raw must have shape (n_envs, n, 3), got {vertices.shape}")
    n = vertices.shape[1]
    if n == k:
        return policy.copy()

    edges = np.linalg.norm(np.diff(vertices, axis=1), axis=-1)  # (n_envs, n-1)
    cumulative = np.concatenate(
        [np.zeros((edges.shape[0], 1)), np.cumsum(edges, axis=1)], axis=1
    )  # (n_envs, n)
    total = cumulative[:, -1:]
    target = (policy / (k - 1.0))[:, None] * total
    return np.abs(cumulative - target).argmin(axis=1).astype(int)


def sample_grasp(
    p: int,
    n_nodes: int,
    rng: np.random.Generator,
    enabled: bool = True,
) -> tuple[int, bool]:
    """Sample the M3 grasp-realism noise/failure model without touching Genesis.

    ``p`` is a RAW VERTEX index here (Rev 6: the caller applies the C1 policy
    mapping first), because the placement error is a physical length applied at
    the actual grasp point.

    Boundary semantics: the ± offset is drawn uniformly and then clamped to the
    valid node range, so the two end nodes self-select with probability 2/3
    (an outward miss re-grasps the end node); interior nodes stay uniform ±.
    Rev 6: the offset magnitude is the fixed arc length of
    :func:`grasp_noise_offset_nodes` (1 vertex at n=32 — byte-identical).
    """

    node = int(p)
    n = int(n_nodes)
    if n < 1:
        raise ValueError("n_nodes must be at least 1")
    if node < 0 or node >= n:
        raise IndexError(f"grasp node {node} outside [0, {n})")
    if not enabled:
        return node, True

    offset = int(rng.choice(grasp_noise_choices(n)))
    actual = int(np.clip(node + offset, 0, n - 1))
    success = bool(rng.random() >= GRASP_FAILURE_PROB)
    return actual, success


def approach_speed(dist: np.ndarray, v_slow: float = APPROACH_V_SLOW) -> np.ndarray:
    """Rev 5 two-stage approach speed profile as a function of remaining distance.

    ``v(d) = clip(sqrt(v_slow^2 + 2*APPROACH_A_DEC*(d - APPROACH_R_SLOW)),
    v_slow, APPROACH_V_FAST)``: free-space transit at ``APPROACH_V_FAST``, a
    constant-deceleration braking band of width
    ``(v_fast^2 - v_slow^2)/(2*a)``, and a hard cap at ``v_slow`` inside
    ``APPROACH_R_SLOW``.  Pure function of the geometry so it is unit-testable
    without a solver.
    """

    d = np.asarray(dist, dtype=float)
    slow = float(v_slow)
    if slow <= 0.0:
        raise ValueError("v_slow must be positive")
    ramp = np.sqrt(
        slow * slow + 2.0 * APPROACH_A_DEC * np.maximum(d - APPROACH_R_SLOW, 0.0)
    )
    return np.clip(ramp, slow, APPROACH_V_FAST)


def approach_brake_distance(v_slow: float = APPROACH_V_SLOW) -> float:
    """Width of the constant-deceleration braking band, in metres."""

    slow = float(v_slow)
    return max(0.0, (APPROACH_V_FAST**2 - slow * slow) / (2.0 * APPROACH_A_DEC))


def centerline_arc_length(points: np.ndarray) -> float:
    """Return the polyline arc length of a ``(N, 3)`` centerline."""

    centerline = np.asarray(points, dtype=float)
    if centerline.ndim != 2 or centerline.shape[1] != 3:
        raise ValueError(f"centerline must have shape (N, 3), got {centerline.shape}")
    if centerline.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum())


def _normalize_init_shape(init_shape: str) -> str:
    shape = str(init_shape).lower()
    if shape not in VALID_INIT_SHAPES:
        allowed = ", ".join(sorted(VALID_INIT_SHAPES))
        raise ValueError(f"init_shape must be one of {{{allowed}}}, got {init_shape!r}")
    return shape


def _scale_curve_to_length(points: np.ndarray, length_m: float) -> np.ndarray:
    curve = np.asarray(points, dtype=float).copy()
    current = centerline_arc_length(curve)
    if current <= 0.0:
        raise ValueError("analytic init curve has zero arc length")
    centroid = curve.mean(axis=0, keepdims=True)
    return centroid + (curve - centroid) * (float(length_m) / current)


def analytic_init_centerline(params: RopeParams, init_shape: str, seed: int) -> np.ndarray:
    """Build a seeded analytic reset centerline with arc length ``params.length_m``."""

    shape = _normalize_init_shape(init_shape)
    n_vertices = int(params.n_segments)
    if n_vertices < 2:
        raise ValueError("params.n_segments must be at least 2")
    length = float(params.length_m)
    radius = float(params.radius)
    t = np.linspace(0.0, 1.0, n_vertices)
    rng = np.random.default_rng(seed)

    if shape == "straight":
        xy = np.column_stack((t - 0.5, np.zeros_like(t)))
    elif shape == "u_bend":
        theta = np.linspace(np.pi, 0.0, n_vertices)
        xy = np.column_stack((np.cos(theta), np.sin(theta)))
    elif shape == "s_curve":
        xy = np.column_stack((t - 0.5, 0.18 * np.sin(2.0 * np.pi * t)))
    else:
        coeffs = rng.normal(0.0, [0.10, 0.055, 0.030, 0.018])
        y = sum(coeffs[k - 1] * np.sin(k * np.pi * t) for k in range(1, 5))
        xy = np.column_stack((t - 0.5, y))

    curve = np.column_stack((xy[:, 0], xy[:, 1], np.zeros_like(t)))
    curve[:, 0] -= float(curve[:, 0].mean())
    curve[:, 1] -= float(curve[:, 1].mean())

    noise_scale = min(0.0015 * length, 0.20 * radius)
    if noise_scale > 0.0:
        noise = rng.normal(0.0, noise_scale, size=curve.shape)
        noise[:, 0] *= 0.25
        noise[:, 2] *= 0.20
        noise[0] *= 0.25
        noise[-1] *= 0.25
        curve += noise

    curve = _scale_curve_to_length(curve, length)
    curve[:, 0] -= float(curve[:, 0].mean())
    curve[:, 1] -= float(curve[:, 1].mean())
    curve[:, 2] -= float(curve[:, 2].min())
    curve[:, 2] += max(radius * 1.25, 0.008)
    return curve.astype(float, copy=False)


class DLOLabUnavailableError(RuntimeError):
    """Raised when the DLO-Lab Genesis package is unavailable."""


class ConstraintCovenantError(RuntimeError):
    """D3 constraint covenant: residual vertex constraints after a reset.

    Raised fail-closed when any ``vertex_constraints.constrained`` slot
    survives a ``light_reset``.  The message starts with
    ``"constraint covenant"`` so log scanners can census it separately from
    the nonfinite/magnitude covenant kinds.
    """

def ensure_genesis_initialized(seed: int | None = None):
    """Import and initialize Genesis for headless GPU DLO-Lab runs."""

    os.environ.pop("DISPLAY", None)
    try:
        import genesis as gs
    except ImportError as exc:  # pragma: no cover - exercised only when install is absent.
        raise DLOLabUnavailableError("DLO-Lab/Genesis is not installed in this environment") from exc

    if not getattr(gs, "_initialized", False):
        # Genesis can only be initialized once per process, so the seed passed to
        # the FIRST caller wins for gs-internal RNG. Per-seed reproducibility in
        # this adapter therefore comes from numpy RNG in reset()/sample_grasp(),
        # not from re-seeding Genesis. Documented for the M3 determinism test.
        gs.init(seed=seed, precision="32", logging_level="warning", backend=gs.gpu)

    # DLO-Lab 1.0.0's sample_centerline kernel references the legacy alias
    # gs.ti_float while the package now exposes gs.qd_float. Provide the alias
    # at runtime instead of patching the gitignored external checkout.
    if not hasattr(gs, "ti_float") and hasattr(gs, "qd_float"):
        gs.ti_float = gs.qd_float
    return gs


def segment_mass_kg(params: RopeParams) -> float:
    """Rev 6 C2: per-segment mass DERIVED from the rope's total mass.

    ``rope_mass_total_kg / n_segments``.  At the historical P1 domain
    (0.032 kg, 32 segments) this is exactly 1.0e-3 kg — the constant the
    adapter used to hard-code.
    """

    n = int(params.n_segments)
    if n < 1:
        raise ValueError("n_segments must be at least 1")
    total = float(params.rope_mass_total_kg)
    if not (total > 0.0):
        raise ValueError("rope_mass_total_kg must be positive")
    return total / n


def stiffness_bases(params: RopeParams | None = None) -> dict[str, float]:
    """Return the simulator-unit bases used for RopeParams multipliers.

    Rev 6 C2: ``segment_mass_base`` is no longer a constant — it depends on the
    discretization.  When ``params`` is supplied the reported value is the one
    actually in force; without it the historical P1 base (0.032 kg / 32) is
    reported so existing diagnostic callers keep their meaning.
    """

    rope_mass = (
        ROPE_MASS_TOTAL_KG_BASE if params is None else float(params.rope_mass_total_kg)
    )
    n_segments = 32 if params is None else int(params.n_segments)
    return {
        "stretch_base_K": STRETCH_BASE,
        "bend_base_E": BEND_BASE,
        "twist_base_G": TWIST_BASE,
        "mu_s_base": MU_S_BASE,
        "mu_k_ratio": MU_K_RATIO,
        "rope_mass_total_kg": rope_mass,
        "segment_mass_base": rope_mass / n_segments,
    }


def mapped_parameters(params: RopeParams) -> dict[str, float]:
    """Map DGCC RopeParams multipliers to DLO-Lab simulator values."""

    mu_s = MU_S_BASE * float(params.friction)
    return {
        "stretching_stiffness_K": STRETCH_BASE,
        "bending_stiffness_E": BEND_BASE * float(params.bend_stiffness),
        "twisting_stiffness_G": TWIST_BASE * float(params.twist_stiffness),
        "mu_s": mu_s,
        "mu_k": MU_K_RATIO * mu_s,
        "segment_mass": segment_mass_kg(params),
        "segment_radius": float(params.radius),
        "rope_mass_total_kg": float(params.rope_mass_total_kg),
    }


class DLOLabEnv(DLOEnvBase):
    """Headless GPU DLO-Lab adapter using the low-level rod_entity API."""

    def __init__(
        self,
        *,
        n_envs: int = 1,
        dt: float = 1.0e-3,
        substeps: int = 5,
        rod_damping: float = 10.0,
        rod_angular_damping: float = 5.0,
        initial_settle_steps: int = 20,
        reset_settle_max_steps: int = 1000,
        move_v_max: float | None = None,
        move_hold_max_steps: int | None = None,
        move_step_size: float = 0.002,
        move_hold_steps: int = 20,
        n_segments: int | None = None,
        rope_mass_total: float | None = None,
        grasp_realism: bool = True,
        at1h_counters: bool = False,
    ) -> None:
        """R1/R2 (env-correction Rev 2 §5.1) parameter semantics.

        ``move_v_max`` [m/s] is the quasi-static gripper velocity cap; the
        per-step displacement is derived as ``move_v_max * dt`` (the legacy
        ``n_steps = max(20, ceil(d / step))`` formula is already an exact
        velocity cap — §0.2).  Supplying ``move_v_max`` activates the
        quasi-static primitive as ONE unit: velocity cap (R1),
        hold-until-quiescent bounded by ``move_hold_max_steps`` (R2, default
        2000), lowering waypoint (R3), horizontal-δ semantics (R4) and the
        subfloor-target assert (R5).  The bundle is indivisible by
        construction so the C3 divergence precondition (velocity cap with
        ``hold=0``) cannot be configured (§1.2 / merge constraint iii).

        ``move_step_size``/``move_hold_steps`` are the DEPRECATED legacy
        displacement-per-step and fixed-hold-count parameters, retained for
        one release; when ``move_v_max`` is absent the adapter behaves
        exactly as before (fixed 20-step floor, ground clip, δ ∈ R³).

        ``at1h_counters`` enables the Rev 3 always-on AT-1H instrumentation
        (design §"상시 로깅 요건"): per-primitive running maxima of node
        speed, edge strain and kinetic energy, sampled after EVERY scene
        step of the primitive so the peaks are identical in definition to
        the ones the acceptance battery measures with its `Probe` wrapper.
        Off by default because it adds one velocity read and one vertex
        read per scene step; the confirmatory training runs turn it on.
        """
        if n_envs < 1:
            raise ValueError("n_envs must be at least 1")
        self.n_envs = int(n_envs)
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.rod_damping = float(rod_damping)
        self.rod_angular_damping = float(rod_angular_damping)
        self.initial_settle_steps = int(initial_settle_steps)
        self.reset_settle_max_steps = int(reset_settle_max_steps)
        self.quasi_static = move_v_max is not None
        if self.quasi_static:
            if float(move_v_max) <= 0.0:
                raise ValueError("move_v_max must be positive")
            self.move_v_max = float(move_v_max)
            self._move_step = self.move_v_max * self.dt
            hold_cap = 2000 if move_hold_max_steps is None else int(move_hold_max_steps)
            if hold_cap < 1:
                raise ValueError(
                    "move_hold_max_steps must be >= 1: the quasi-static bundle "
                    "forbids the hold=0 divergence precondition (design §1.2)"
                )
            self.move_hold_max_steps = hold_cap
            self.move_hold_vel_threshold = HOLD_QUIESCENT_VEL
        else:
            if move_hold_max_steps is not None:
                raise ValueError("move_hold_max_steps requires move_v_max")
            self.move_v_max = None
            self._move_step = float(move_step_size)
            self.move_hold_max_steps = None
            self.move_hold_vel_threshold = HOLD_QUIESCENT_VEL
        self.move_step_size = float(move_step_size)
        self.move_hold_steps = int(move_hold_steps)
        self.grasp_realism = bool(grasp_realism)
        # Rev 6 B: rope discretization and total mass are declared in the
        # `sim` config block and cross-checked against the `RopeParams` handed
        # to `reset()`.  Two independent declarations that must agree is the
        # point: a config that says 64 nodes / 40 g can no longer be paired
        # with a 32-node domain object without the launch failing closed.
        self.cfg_n_segments = None if n_segments is None else int(n_segments)
        self.cfg_rope_mass_total = (
            None if rope_mass_total is None else float(rope_mass_total)
        )
        if self.cfg_n_segments is not None and self.cfg_n_segments < 2:
            raise ValueError("n_segments must be at least 2")
        if self.cfg_rope_mass_total is not None and not (self.cfg_rope_mass_total > 0.0):
            raise ValueError("rope_mass_total must be positive")
        self.last_hold_steps_used = 0
        self.last_hold_converged: bool | None = None
        self.last_waypoint_steps: list[int] = []
        self.last_lower_strain_aborts = 0
        self.last_lower_z: np.ndarray | None = None
        self.last_tension_pause_steps = 0
        self.last_tension_freezes = 0
        # Rev 5 approach/attach telemetry (one primitive's worth).
        # `last_approach_steps` counts COMMANDED-MOTION scene steps and
        # `last_approach_dwell_steps` the frozen-command gate steps, using the
        # same "did the commanded array change" rule the acceptance probe uses
        # to classify move vs hold, so the battery can subtract the approach
        # out of the AT-6 denominator exactly.
        self.last_approach_steps = 0
        self.last_approach_dwell_steps = 0
        self.last_approach_gate_failures = 0
        self.last_attach_rel_vel: np.ndarray = np.full(self.n_envs, np.nan)
        self.last_attach_offset: np.ndarray = np.full(self.n_envs, np.nan)
        # AT-1H always-on counters (Rev 3 상시 로깅 요건).  `_at1h_active`
        # is only true between `_at1h_begin()` and `_at1h_end()`, i.e. for
        # the duration of one primitive (move legs + hold + post-release
        # settle), matching the acceptance battery's per-primitive window.
        self.at1h_counters = bool(at1h_counters)
        self._at1h_active = False
        self._at1h_v_peak = np.zeros(self.n_envs, dtype=float)
        self._at1h_strain_peak = np.zeros(self.n_envs, dtype=float)
        self._at1h_ke_peak = np.zeros(self.n_envs, dtype=float)
        self._at1h_samples = 0

        self.gs: Any | None = None
        self.scene: Any | None = None
        self.rod_entity: Any | None = None
        self.gripper_entity: Any | None = None
        self.gripper_link: Any | None = None
        self.params: RopeParams | None = None
        self.active_node: int | None = None
        self.last_settle_steps = 0
        self.last_settle_converged = False
        self.last_delta_clamped = np.zeros(3, dtype=float)
        self.last_move_target = np.zeros((self.n_envs, 3), dtype=float)
        self.last_grasp_actual_node: int | None = None
        self.last_grasp_policy_node: int | None = None
        self.last_grasp_target_vertex: int | None = None
        self.last_grasp_target_vertices: np.ndarray | None = None
        self.last_grasp_success = False
        self._rng = np.random.default_rng(0)
        self.last_reset_settle_converged: bool | None = None
        self.last_grasp_actual_nodes: np.ndarray | None = None
        self.last_grasp_successes: np.ndarray | None = None
        self.last_settle_steps_batch: np.ndarray | None = None
        self.last_settle_converged_batch: np.ndarray | None = None
        self._batched_active_nodes: np.ndarray | None = None
        self.detach_escalation_total = 0
        self.last_detach_residuals = 0

    def reset(self, params: RopeParams, init_shape: str, seed: int) -> dict[str, Any]:
        self._validate_params(params)
        self._assert_domain_matches_config(params)
        normalized_shape = _normalize_init_shape(init_shape)
        init_vertices = analytic_init_centerline(params, normalized_shape, seed)

        seed_everything(seed)
        self._rng = np.random.default_rng(seed)
        self.gs = ensure_genesis_initialized(seed)
        gs = self.gs

        self.params = params
        self.active_node = None
        self.last_grasp_actual_node = None
        self.last_grasp_success = False
        self.last_settle_steps = 0
        self.last_settle_converged = False
        self.last_reset_settle_converged = None
        self.last_grasp_actual_nodes = None
        self.last_grasp_successes = None
        self.last_settle_steps_batch = None
        self.last_settle_converged_batch = None
        self._batched_active_nodes = None
        self.detach_escalation_total = 0
        self.last_detach_residuals = 0

        mapped = mapped_parameters(params)
        length = float(params.length_m)
        interval = length / float(params.n_segments - 1)
        start_pos = (-0.5 * length, 0.0, max(float(params.radius) * 1.25, 0.008))

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=self.substeps),
            rod_options=gs.options.RODOptions(damping=self.rod_damping, angular_damping=self.rod_angular_damping),
            show_viewer=False,
        )
        self.scene.add_entity(
            material=gs.materials.Rigid(needs_coup=True, coup_friction=mapped["mu_s"]),
            morph=gs.morphs.Plane(fixed=True),
        )
        self.rod_entity = self.scene.add_entity(
            material=gs.materials.ROD.Base(
                segment_radius=float(params.radius),
                segment_mass=mapped["segment_mass"],
                K=mapped["stretching_stiffness_K"],
                E=mapped["bending_stiffness_E"],
                G=mapped["twisting_stiffness_G"],
                static_friction=mapped["mu_s"],
                kinetic_friction=mapped["mu_k"],
                # R9 (env-correction Rev 2 P12): DO NOT CHANGE. Switching the
                # constitutive model is a separate major change; with the
                # quasi-static bundle the measured strain stays <= 1%, so
                # inextensibility is unnecessary. Revisit only if AT-4 fails
                # repeatedly after the P9 pile-aware repair.
                use_inextensible=False,
            ),
            morph=gs.morphs.ParameterizedRod(
                type="rod",
                n_vertices=int(params.n_segments),
                interval=interval,
                radius=float(params.radius),
                rest_state="straight",
                axis="x",
                pos=start_pos,
            ),
        )
        self.gripper_entity = self.scene.add_entity(
            material=gs.materials.Rigid(needs_coup=False),
            morph=gs.morphs.Sphere(
                pos=(0.0, 0.0, LIFT_HEIGHTS["high"]),
                radius=max(float(params.radius) * 1.5, 0.0075),
                fixed=True,
                collision=False,
                visualization=True,
            ),
        )
        self.scene.build(n_envs=self.n_envs)
        self.gripper_link = self.gripper_entity.links[0]
        self.apply_params(params)

        self._rollout(self.initial_settle_steps)
        self._place_rod_vertices(init_vertices)
        self.last_reset_settle_converged = self.settle(max_steps=self.reset_settle_max_steps)

        self._assert_finite()
        return {
            "sim": "dlolab",
            "seed": int(seed),
            "init_shape": normalized_shape,
            "rope_params": asdict(params),
            "n_envs": self.n_envs,
            "n_vertices": int(params.n_segments),
            "length_m": length,
            "interval_m": interval,
            "initial_arc_length_m": centerline_arc_length(init_vertices),
            "mapped_parameters": mapped,
            "stiffness_bases": stiffness_bases(params),
            "reset_settle_converged": self.last_reset_settle_converged,
            "init_vertex_setter": "rod_entity.set_position((n_envs, n_vertices, 3)); rod_entity.set_velocity(zeros)",
            "show_viewer": False,
            "backend": str(getattr(gs, "backend", "unknown")),
        }

    def apply_params(self, params: RopeParams) -> dict[str, float]:
        self._require_reset()
        assert self.gs is not None and self.rod_entity is not None
        import torch

        mapped = mapped_parameters(params)
        dtype = self.gs.tc_float
        device = self.gs.device
        n_envs = self.n_envs
        n_vertices = int(params.n_segments)

        self.rod_entity.set_bending_stiffness(
            torch.full((n_envs,), mapped["bending_stiffness_E"], dtype=dtype, device=device)
        )
        self.rod_entity.set_twisting_stiffness(
            torch.full((n_envs,), mapped["twisting_stiffness_G"], dtype=dtype, device=device)
        )
        self.rod_entity.set_stretching_stiffness(
            torch.full((n_envs,), mapped["stretching_stiffness_K"], dtype=dtype, device=device)
        )
        self.rod_entity.set_mu_s(torch.full((n_envs, n_vertices), mapped["mu_s"], dtype=dtype, device=device))
        self.rod_entity.set_mu_k(torch.full((n_envs, n_vertices), mapped["mu_k"], dtype=dtype, device=device))
        self.rod_entity.set_segment_radius(
            torch.full((n_envs, n_vertices), mapped["segment_radius"], dtype=dtype, device=device)
        )
        self.rod_entity.set_segment_mass(
            torch.full((n_envs, n_vertices), mapped["segment_mass"], dtype=dtype, device=device)
        )
        return mapped

    def get_centerline_raw(self) -> np.ndarray:
        raw = self._raw_batch()
        return raw[0].copy() if self.n_envs == 1 else raw.copy()

    def get_centerline(self) -> np.ndarray:
        self._require_reset()
        assert self.rod_entity is not None
        sampled = np.asarray(self.rod_entity.sample_centerline(self.K), dtype=float)
        return sampled[0].copy() if self.n_envs == 1 else sampled.copy()

    def get_centerline_raw_batch(self) -> np.ndarray:
        """Return native rope vertices with explicit ``(n_envs, N, 3)`` batch axis."""

        return self._raw_batch().copy()

    def get_centerline_batch(self) -> np.ndarray:
        """Return resampled centerlines with explicit ``(n_envs, 32, 3)`` batch axis."""

        self._require_reset()
        assert self.rod_entity is not None
        sampled = np.asarray(self.rod_entity.sample_centerline(self.K), dtype=float)
        return sampled.reshape(self.n_envs, self.K, 3).copy()

    def supports_per_env_grasp(self) -> bool:
        """Return whether DLO-Lab exposes per-environment attach/detach hooks."""

        self._require_reset()
        assert self.rod_entity is not None
        return all(
            hasattr(self.rod_entity, name)
            for name in (
                "attach_to_rigid_link_with_envs_idx",
                "detach_from_rigid_link_with_envs_idx",
            )
        )

    def place_rod_vertices_batch(
        self, vertices: np.ndarray, *, reinit_env_indices: np.ndarray | None = None
    ) -> None:
        """Light-reset all environments from explicit per-env native vertices.

        ``vertices`` may be either ``(N, 3)`` (broadcast to every environment) or
        ``(n_envs, N, 3)`` (distinct curve per environment).  Existing batched
        attachments are cleared, positions are written through
        ``rod_entity.set_position``, and velocities are zeroed.

        ``reinit_env_indices`` scopes the full edge/frame state
        re-initialization (M2 gate F2): only the listed environments get their
        twist/frame/internal state rebuilt from the placed positions; other
        environments keep their evolved twist state (their positions are
        rewritten with identical values).  ``None`` re-initializes every
        environment (fresh placement contract).
        """

        self._place_rod_vertices_batched(vertices, reinit_env_indices=reinit_env_indices)

    def light_reset(
        self,
        vertices: np.ndarray,
        *,
        vel_threshold: float = 1e-3,
        max_steps: int = 5000,
        reinit_env_indices: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Re-place batched native vertices and settle without rebuilding the scene."""

        self.place_rod_vertices_batch(vertices, reinit_env_indices=reinit_env_indices)
        converged, settle_steps = self.settle_batch(
            vel_threshold=vel_threshold,
            max_steps=max_steps,
        )
        # D3 (env-correction Rev 2 §1.4): constraint covenant.  After every
        # light reset the solver-side attachment table must be empty for ALL
        # envs; a survivor means D1/D2 regressed or the upstream kernel
        # changed behavior — fail closed instead of absorbing the
        # contamination into training/eval state.
        constrained_mask = self._vertex_constrained_mask()
        if constrained_mask.any():
            leftover = [(int(e), int(v)) for e, v in zip(*np.nonzero(constrained_mask))]
            raise ConstraintCovenantError(
                f"constraint covenant: residual vertex constraints after light_reset: {leftover}"
            )
        return {
            "settle_converged": converged,
            "settle_steps": settle_steps,
        }

    def _node_velocities_batch(self) -> np.ndarray:
        """Per-env rod vertex velocities with an explicit ``(n_envs, N, 3)`` axis."""

        assert self.rod_entity is not None
        vels = np.asarray(self.rod_entity.get_all_vels(), dtype=float)
        if vels.ndim == 2:
            vels = vels.reshape(1, *vels.shape)
        return vels.reshape(self.n_envs, -1, 3)

    def _approach_v_slow(self) -> float:
        """Final-approach speed cap: never above the manipulation cap in force."""

        if self.quasi_static and self.move_v_max is not None:
            return min(APPROACH_V_SLOW, float(self.move_v_max))
        return APPROACH_V_SLOW

    def _approach_and_attach(
        self, nodes: np.ndarray, eligible: np.ndarray, *, per_env: bool
    ) -> dict[str, Any]:
        """Rev 5: speed-limited two-stage approach followed by an attach gate.

        The gripper tracks the LIVE position of its target node under the
        `approach_speed` profile (fast in free space, braking band, capped at
        the manipulation speed inside ``APPROACH_R_SLOW``).  Because the
        commanded displacement collapses to zero as the gripper converges on
        the node, the commanded gripper velocity decays to the node's own
        velocity, and the gate

            ‖v_gripper_cmd − v_node‖ < ATTACH_REL_VEL  and  ‖p_node − p_grip‖ ≤ v_slow·dt

        is the physical statement "only close the gripper once it is moving
        WITH the material point it is about to constrain".  The gripper sphere
        is non-colliding and uncoupled, so the approach itself applies no force
        to the rope; what the gate removes is the velocity discontinuity the
        rigid attach would otherwise impose.

        ``per_env=True`` attaches each environment the instant it clears the
        gate (batch path, per-env attach hooks).  ``per_env=False`` is the
        single-node ``grasp()`` path, whose upstream API attaches all envs at
        once, so it waits until every eligible env clears the gate together.

        Envs still pending when ``APPROACH_MAX_STEPS`` is exhausted are
        returned unattached; the caller converts them into grasp failures and
        rewinds them through the Rev 4 snapshot path.
        """

        assert self.rod_entity is not None and self.gripper_link is not None
        n_envs = self.n_envs
        env_index = np.arange(n_envs)
        targets = np.asarray(nodes, dtype=int).reshape(n_envs)
        eligible_mask = np.asarray(eligible, dtype=bool).reshape(n_envs)
        if not per_env and eligible_mask.any():
            unique = np.unique(targets[eligible_mask])
            if unique.size != 1:
                raise ValueError(
                    "all-env attach requires one shared target node, got "
                    f"{unique.tolist()}"
                )

        pending = eligible_mask.copy()
        attached = np.zeros(n_envs, dtype=bool)
        rel_at_attach = np.full(n_envs, np.nan)
        offset_at_attach = np.full(n_envs, np.nan)
        v_slow = self._approach_v_slow()
        arrive_tol = v_slow * self.dt
        commanded = self._gripper_positions()
        walk_steps = 0
        dwell_steps = 0
        steps = 0

        while pending.any() and steps < APPROACH_MAX_STEPS:
            previous = commanded
            node_pos = self._raw_batch()[env_index, targets, :]
            offset_vec = node_pos - previous
            dist = np.linalg.norm(offset_vec, axis=1)
            step_len = approach_speed(dist, v_slow) * self.dt
            frac = np.minimum(1.0, step_len / np.maximum(dist, 1.0e-12))
            frac = np.where(dist > 0.0, frac, 0.0)
            commanded = previous + offset_vec * frac[:, None]
            # Done / ineligible envs freeze their commanded pose so their
            # gripper velocity is exactly zero and their attachment (if any)
            # is not dragged while the remaining envs finish approaching.
            commanded[~pending] = previous[~pending]
            v_grip = (commanded - previous) / self.dt
            # Mirrors the acceptance probe's move/hold classification exactly
            # (bitwise equality of the commanded array).
            if np.array_equal(commanded, previous):
                dwell_steps += 1
            else:
                walk_steps += 1
            self._set_gripper_positions(commanded)
            self._step_scene()
            steps += 1

            node_pos = self._raw_batch()[env_index, targets, :]
            node_vel = self._node_velocities_batch()[env_index, targets, :]
            offset = np.linalg.norm(node_pos - commanded, axis=1)
            rel = np.linalg.norm(v_grip - node_vel, axis=1)
            ready = pending & (offset <= arrive_tol) & (rel < ATTACH_REL_VEL)
            if not ready.any():
                continue

            if per_env:
                for env_idx in np.flatnonzero(ready):
                    self.rod_entity.attach_to_rigid_link_with_envs_idx(
                        self.gripper_link, [int(targets[env_idx])], int(env_idx)
                    )
                    if self._batched_active_nodes is not None:
                        self._batched_active_nodes[env_idx] = int(targets[env_idx])
                selected = ready
            else:
                if not bool(np.all(ready[pending])):
                    continue
                node = int(targets[np.flatnonzero(pending)[0]])
                self.rod_entity.attach_to_rigid_link(self.gripper_link, [node])
                self.active_node = node
                selected = pending.copy()
            rel_at_attach[selected] = rel[selected]
            offset_at_attach[selected] = offset[selected]
            attached |= selected
            pending &= ~selected

        gate_failures = int((eligible_mask & ~attached).sum())
        self.last_approach_steps = int(walk_steps)
        self.last_approach_dwell_steps = int(dwell_steps)
        self.last_approach_gate_failures = gate_failures
        self.last_attach_rel_vel = rel_at_attach
        self.last_attach_offset = offset_at_attach
        return {
            "attached": attached,
            "walk_steps": int(walk_steps),
            "dwell_steps": int(dwell_steps),
            "gate_failures": gate_failures,
            "rel_vel_at_attach": rel_at_attach,
            "offset_at_attach": offset_at_attach,
        }

    def policy_nodes_to_vertices(self, p: np.ndarray) -> np.ndarray:
        """Rev 6 C1: per-env policy index -> raw vertex index, by arc length.

        Thin adapter-side binding of :func:`arc_length_vertex_index` to the
        live rope state and this adapter's policy node count ``self.K``.
        """

        self._require_reset()
        return arc_length_vertex_index(self._raw_batch(), p, self.K)

    def _policy_node_to_shared_vertex(self, p: int) -> int:
        """Map ONE policy index to ONE raw vertex shared by every env.

        The single-node ``grasp()`` path attaches through the all-env upstream
        API, which cannot express a different vertex per env.  With the C1
        mapping the per-env deformed shapes could in principle disagree, so the
        disagreement is a fail-closed error rather than a silent env-0 pick.
        """

        mapped = self.policy_nodes_to_vertices(np.full(self.n_envs, int(p), dtype=int))
        unique = np.unique(mapped)
        if unique.size != 1:
            raise RuntimeError(
                f"policy node {int(p)} maps to different raw vertices across envs "
                f"({unique.tolist()}); the all-env grasp path cannot express that. "
                "Use step_primitive_batch(), which attaches per env."
            )
        return int(unique[0])

    def grasp_policy_node(self, p: int) -> bool:
        """Grasp the vertex that policy node ``p`` denotes (Rev 6 C1 entry).

        ``grasp()`` itself stays RAW-vertex by contract — ``active_node``,
        ``_batched_active_nodes`` and the upstream attach/detach kernels all
        speak raw vertex indices, so re-interpreting its argument would make
        the mechanism-level call ambiguous.  Every path where a POLICY index
        actually enters the adapter (this method, ``step_primitive`` and
        ``step_primitive_batch``) applies the mapping first.
        """

        return self.grasp(self._policy_node_to_shared_vertex(p))

    def grasp(self, p: int) -> bool:
        self._require_reset()
        assert self.rod_entity is not None and self.gripper_link is not None
        # Rev 6 C1: ``p`` is a RAW VERTEX index here by contract.  Callers that
        # hold a POLICY index must map it first (``grasp_policy_node``,
        # ``step_primitive``, ``step_primitive_batch``).
        node = int(p)
        n_vertices = self._n_vertices()
        if node < 0 or node >= n_vertices:
            raise IndexError(f"grasp node {node} outside [0, {n_vertices})")

        # Rev 5: no teleport.  The gripper walks in under the two-stage speed
        # profile and only attaches once every env clears the relative-velocity
        # gate; a budget exhaustion is a grasp FAILURE, not a forced attach.
        approach = self._approach_and_attach(
            np.full(self.n_envs, node, dtype=int),
            np.ones(self.n_envs, dtype=bool),
            per_env=False,
        )
        if not bool(approach["attached"].all()):
            self._assert_finite()
            return False
        self._step_scene()
        self._assert_finite()
        return True

    def _prepare_primitive_inputs(self, delta: np.ndarray, lift: str) -> np.ndarray:
        if lift not in LIFT_HEIGHTS:
            raise ValueError(f"lift must be one of {sorted(LIFT_HEIGHTS)}, got {lift!r}")

        delta_vec = np.asarray(delta, dtype=float)
        if delta_vec.shape != (3,):
            raise ValueError(f"delta must have shape (3,), got {delta_vec.shape}")
        if not np.all(np.isfinite(delta_vec)):
            raise ValueError("delta contains non-finite values")
        if self.quasi_static:
            # R4: δ is a HORIZONTAL displacement — the z component is ignored
            # (dimension kept at R³ for checkpoint/action-space compatibility,
            # design §5.6 note 1) and the clamp applies to ‖δ_xy‖.
            delta_vec = delta_vec.copy()
            delta_vec[2] = 0.0
        norm = float(np.linalg.norm(delta_vec))
        if norm > MAX_DELTA_NORM:
            delta_vec = delta_vec * (MAX_DELTA_NORM / norm)
        self.last_delta_clamped = delta_vec.copy()
        return delta_vec

    def move(self, delta: np.ndarray, lift: str) -> np.ndarray:
        delta_vec = self._prepare_primitive_inputs(delta, lift)
        return self._move_prepared(delta_vec, lift)

    def _gripper_z_floor(self) -> float:
        """Lowest allowed gripper target height (support-plane safety).

        Adapter-internal P1 addition (interface unchanged): §6 actions carry
        Δ ∈ R³ including negative z, so a low-lift grasp plus Δz ≈ −0.15 can
        command the gripper (and the attached node) BELOW the ground plane,
        driving rope-plane penetration and solver blow-ups (observed in the
        P1-M2 smoke: ~45% of batches lost to the NaN covenant).  The plane is
        rigid — a physical gripper cannot push through it — so move targets
        are clamped to the rope's resting height above the plane.  The action
        space itself is unchanged; only physically impossible targets are
        clipped, mirroring the existing ‖δ‖ ≤ 0.15 norm clamp.
        """

        radius = float(self.params.radius) if self.params is not None else 0.005
        return max(radius, 0.005)

    def _finalize_move_targets(self, lifted: np.ndarray, deltas: np.ndarray) -> np.ndarray:
        """Shared single/batch target construction with R5 floor handling."""
        target = lifted + deltas
        floor = self._gripper_z_floor()
        if self.quasi_static:
            # R5: with horizontal δ (R4) a subfloor target is unreachable by
            # construction, so the legacy clip is promoted to a fail-closed
            # assert (AT-9): its firing means R4 regressed upstream.
            if bool(np.any(target[:, 2] < floor)):
                raise RuntimeError(
                    "subfloor gripper target commanded despite horizontal-δ "
                    f"semantics (min z {float(target[:, 2].min()):.6f} < floor "
                    f"{floor:.6f}); R4 regressed"
                )
        else:
            target[:, 2] = np.maximum(target[:, 2], floor)
        return target

    def _max_edge_strain_batch(self) -> np.ndarray:
        """Per-env maximum |edge length / rest length − 1| from raw vertices."""
        assert self.params is not None
        raw = np.asarray(self._raw_batch(), dtype=float)
        edges = np.linalg.norm(raw[:, 1:, :] - raw[:, :-1, :], axis=-1)
        rest = float(self.params.length_m) / (raw.shape[1] - 1)
        return np.abs(edges / rest - 1.0).max(axis=1)

    def _pile_aware_lower_z(self, target: np.ndarray) -> np.ndarray:
        """P9 Rev 3 (adjudication §3.3, owner pin O1): pile-aware lower height.

        z_target_lower = max(z_floor, z_pile + ε), where z_pile is the highest
        non-grasped node inside the r_nb neighborhood of the xy target
        (z_floor when the neighborhood is empty).  Pure deterministic function
        of the current state via the public raw-vertex getter; no solver or
        upstream modification.
        """
        floor = self._gripper_z_floor()
        raw = np.asarray(self._raw_batch(), dtype=float)
        dist_xy = np.linalg.norm(raw[:, :, :2] - target[:, None, :2], axis=-1)
        neighborhood = dist_xy < PILE_NEIGHBOR_RADIUS_M
        if self._batched_active_nodes is not None:
            for env_idx, node in enumerate(self._batched_active_nodes):
                if int(node) >= 0:
                    neighborhood[env_idx, int(node)] = False
        elif self.active_node is not None:
            neighborhood[:, int(self.active_node)] = False
        z = raw[:, :, 2]
        z_masked = np.where(neighborhood, z, -np.inf)
        z_pile = np.where(
            neighborhood.any(axis=1), z_masked.max(axis=1), floor
        )
        return np.maximum(floor, z_pile + PILE_CLEARANCE_M)

    def _execute_move(
        self, lifted: np.ndarray, target: np.ndarray, vel_threshold: float
    ) -> np.ndarray:
        """Shared single/batch waypoint walk + hold (design §5.6 note 2).

        Legacy mode: (lift, translate) waypoints and a fixed hold count —
        byte-compatible with the historical behavior.  Quasi-static mode adds
        the R3 lowering waypoint with the P9 Rev 3 pile-aware SET (owner pin
        O1; the two guards are indivisible like D1-D3):
          (c) geometric guard — the lower target is z_pile + ε, never a
              commanded descent through an existing pile;
          (d) strain fail-safe — during the lower leg any env whose max edge
              strain exceeds LOWER_STRAIN_ABORT freezes its descent at the
              current height and transitions to hold there (covers grasped-
              node-in-pile geometries the neighborhood test cannot see).
        The hold is R2 hold-until-quiescent with the REALIZED quiescence
        threshold (orchestrator directive item 4): quiescent means the
        release-speed acceptance criterion (0.05 m/s, AT-3), not the settle
        threshold — this makes AT-3 structural in fact and removes the
        hold-cap exhaustion mode (14% at the 1e-3 threshold).
        """
        current = self._gripper_positions()
        self.last_waypoint_steps = []
        self.last_lower_strain_aborts = 0
        self.last_lower_z = None
        self.last_tension_pause_steps = 0
        self.last_tension_freezes = 0

        walk_waypoints = [lifted, target]
        walk_frozen = np.zeros(self.n_envs, dtype=bool)
        for waypoint in walk_waypoints:
            max_distance = float(np.max(np.linalg.norm(waypoint - current, axis=1)))
            n_steps = max(20, int(ceil(max_distance / self._move_step)))
            self.last_waypoint_steps.append(int(n_steps))
            if not self.quasi_static:
                for alpha in np.linspace(1.0 / n_steps, 1.0, n_steps):
                    pos = (1.0 - alpha) * current + alpha * waypoint
                    self._set_gripper_positions(pos)
                    self._step_scene()
                current = waypoint.copy()
                continue
            # Tension guard (Rev 3, owner-approved 2026-08-02): during the
            # lift/translate legs, an env whose max edge strain exceeds the
            # P9 strain threshold PAUSES its own progress (gripper holds its
            # current commanded pose) until the slack resolves, then resumes
            # — the analogue of a real robot's force-limited pull.  An env
            # whose tension never resolves within TENSION_PAUSE_MAX_STEPS is
            # frozen for the rest of the walk (same fail-safe semantics as
            # the approved lowering guard) and released from wherever it
            # stopped.  This closes the residual lift-leg tension-snap mode
            # (4/600 AT-1H violations in the stratified remeasurement).
            progress = np.zeros(self.n_envs, dtype=int)
            paused = np.zeros(self.n_envs, dtype=int)
            pos = current.copy()
            start = current.copy()
            while True:
                pending = (~walk_frozen) & (progress < n_steps)
                if not pending.any():
                    break
                strain = self._max_edge_strain_batch()
                strained = strain > LOWER_STRAIN_ABORT
                advance = pending & ~strained
                pause = pending & strained
                if pause.any():
                    paused[pause] += 1
                    self.last_tension_pause_steps += int(pause.sum())
                    overdue = pause & (paused > TENSION_PAUSE_MAX_STEPS)
                    if overdue.any():
                        walk_frozen |= overdue
                        self.last_tension_freezes += int(overdue.sum())
                progress[advance] += 1
                alpha = (progress / n_steps)[:, None]
                step_pos = (1.0 - alpha) * start + alpha * waypoint
                keep = walk_frozen | ~ (advance | pause)
                step_pos[keep] = pos[keep]
                step_pos[pause & ~walk_frozen] = pos[pause & ~walk_frozen]
                pos = step_pos
                self._set_gripper_positions(pos)
                self._step_scene()
            current = pos.copy()

        final = target
        if self.quasi_static:
            lowered = target.copy()
            lowered[:, 2] = self._pile_aware_lower_z(target)
            self.last_lower_z = lowered[:, 2].copy()
            max_distance = float(np.max(np.linalg.norm(lowered - current, axis=1)))
            n_steps = max(20, int(ceil(max_distance / self._move_step)))
            self.last_waypoint_steps.append(int(n_steps))
            # Walk-frozen envs (unresolved tension) stay frozen through the
            # lowering leg as well.
            frozen = walk_frozen.copy()
            pos = current.copy()
            for alpha in np.linspace(1.0 / n_steps, 1.0, n_steps):
                step_pos = (1.0 - alpha) * current + alpha * lowered
                step_pos[frozen] = pos[frozen]
                pos = step_pos
                self._set_gripper_positions(pos)
                self._step_scene()
                if not frozen.all():
                    strained = self._max_edge_strain_batch() > LOWER_STRAIN_ABORT
                    newly = strained & ~frozen
                    if newly.any():
                        frozen |= newly
            self.last_lower_strain_aborts = int(frozen.sum())
            final = pos

        if self.quasi_static:
            hold_steps = 0
            threshold = float(self.move_hold_vel_threshold)
            while (
                hold_steps < self.move_hold_max_steps
                and float(np.max(self.max_node_speed_batch())) >= threshold
            ):
                self._set_gripper_positions(final)
                self._step_scene()
                hold_steps += 1
            self.last_hold_steps_used = hold_steps
            self.last_hold_converged = bool(
                float(np.max(self.max_node_speed_batch())) < threshold
            )
        else:
            for _ in range(max(0, self.move_hold_steps)):
                self._set_gripper_positions(final)
                self._step_scene()
            self.last_hold_steps_used = max(0, self.move_hold_steps)
            self.last_hold_converged = None
        self._assert_finite()
        return final

    def _move_prepared(
        self, delta_vec: np.ndarray, lift: str, vel_threshold: float = 1e-3
    ) -> np.ndarray:
        """Run the waypoint move for an already-validated/clamped delta (A7: clamp once)."""
        self._require_reset()
        if self.active_node is None:
            raise RuntimeError("move called before grasp")

        start = self._gripper_positions()
        lifted = start.copy()
        lifted[:, 2] = LIFT_HEIGHTS[lift]
        target = self._finalize_move_targets(lifted, delta_vec.reshape(1, 3))
        self.last_move_target = target.copy()

        final = self._execute_move(lifted, target, vel_threshold)
        return final[0].copy() if self.n_envs == 1 else final.copy()

    def _move_prepared_batch(
        self,
        delta_vecs: np.ndarray,
        lift_values: Sequence[str],
        vel_threshold: float = 1e-3,
    ) -> np.ndarray:
        """Run batched waypoint moves with one target per environment."""

        self._require_reset()
        if self._batched_active_nodes is None:
            raise RuntimeError("batched move called before batched grasp")

        deltas = np.asarray(delta_vecs, dtype=float)
        if deltas.shape != (self.n_envs, 3):
            raise ValueError(f"delta_vecs must have shape ({self.n_envs}, 3), got {deltas.shape}")
        lifts = [str(value) for value in lift_values]
        if len(lifts) != self.n_envs:
            raise ValueError(f"lift_values must contain {self.n_envs} entries")
        for lift in lifts:
            if lift not in LIFT_HEIGHTS:
                raise ValueError(f"lift must be one of {sorted(LIFT_HEIGHTS)}, got {lift!r}")

        start = self._gripper_positions()
        lifted = start.copy()
        lifted[:, 2] = np.asarray([LIFT_HEIGHTS[lift] for lift in lifts], dtype=float)
        target = self._finalize_move_targets(lifted, deltas)
        self.last_move_target = target.copy()

        return self._execute_move(lifted, target, vel_threshold).copy()

    def release(self, vel_threshold: float = 1e-3, max_steps: int = 5000) -> bool:
        self._require_reset()
        assert self.rod_entity is not None
        if self.active_node is not None:
            self.rod_entity.detach_from_rigid_link([self.active_node])
        self.active_node = None
        self._step_scene()
        return self.settle(vel_threshold=vel_threshold, max_steps=max_steps)

    def step_primitive(self, p: int, delta: np.ndarray, lift: str) -> dict[str, Any]:
        delta_vec = self._prepare_primitive_inputs(delta, lift)
        X_before = self.get_centerline()
        # Rev 6 C1: `p` is a POLICY node index; map it to a raw vertex by arc
        # length BEFORE the grasp-realism draw, because the C3 placement error
        # is a physical length applied at the actual grasp point.
        p_vertex = self._policy_node_to_shared_vertex(p)
        p_actual, sampled_success = sample_grasp(
            p_vertex, self._n_vertices(), self._rng, self.grasp_realism
        )
        self.last_grasp_actual_node = p_actual
        self.last_grasp_policy_node = int(p)
        self.last_grasp_target_vertex = int(p_vertex)
        self.last_grasp_success = bool(sampled_success)

        if not sampled_success:
            # A5: measure rather than assert quasi-staticity of the untouched rope.
            measured_speed = self.max_node_speed()
            measured_converged = bool(measured_speed <= 1e-3)
            self.last_settle_steps = 0
            self.last_settle_converged = measured_converged
            X_after = X_before.copy()
            return {
                "X_before": X_before,
                "X_after": X_after,
                "grasp_success": False,
                "settle_steps": 0,
                "info": {
                    "p": int(p),
                    "p_actual": int(p_actual),
                    "grasp_realism": bool(self.grasp_realism),
                    "grasp_failure_prob": GRASP_FAILURE_PROB,
                    "grasp_noise": int(p_actual - p_vertex),
                    "p_vertex": int(p_vertex),
                    "delta_clamped": delta_vec.copy(),
                    "lift": lift,
                    "gripper_target": None,
                    "settle_converged": measured_converged,
                    "max_node_speed": measured_speed,
                    "mapped_parameters": mapped_parameters(self.params) if self.params is not None else None,
                },
            }

        # Rev 5: the attach gate can fail a grasp the sampler accepted, so the
        # rewind target is captured BEFORE the approach starts.  Read-only, and
        # only on the path that actually approaches (the sampled-failure branch
        # above returned already).
        pre_primitive_state = self._snapshot_rod_state()
        grasp_success = self.grasp(p_actual)
        if not grasp_success:
            # Attach gate not cleared inside the budget.  Nothing was attached,
            # so the rope only free-evolved during the approach; rewind it to
            # the pre-primitive solver state (energy-neutral, Rev 4 path) and
            # report the standard grasp-failure contract.
            self._restore_failed_grasp_envs(
                pre_primitive_state, np.arange(self.n_envs)
            )
            self.last_grasp_success = False
            measured_speed = self.max_node_speed()
            measured_converged = bool(measured_speed <= 1e-3)
            self.last_settle_steps = 0
            self.last_settle_converged = measured_converged
            return {
                "X_before": X_before,
                "X_after": X_before.copy(),
                "grasp_success": False,
                "settle_steps": 0,
                "info": {
                    "p": int(p),
                    "p_actual": int(p_actual),
                    "grasp_realism": bool(self.grasp_realism),
                    "grasp_failure_prob": GRASP_FAILURE_PROB,
                    "grasp_noise": int(p_actual - p_vertex),
                    "delta_clamped": delta_vec.copy(),
                    "lift": lift,
                    "gripper_target": None,
                    "settle_converged": measured_converged,
                    "max_node_speed": measured_speed,
                    "attach_gate_failed": True,
                    "approach_steps": int(self.last_approach_steps),
                    "approach_dwell_steps": int(self.last_approach_dwell_steps),
                    "mapped_parameters": mapped_parameters(self.params) if self.params is not None else None,
                },
            }
        target = self._move_prepared(delta_vec, lift)
        settle_converged = self.release()
        X_after = self.get_centerline()
        return {
            "X_before": X_before,
            "X_after": X_after,
            "grasp_success": bool(grasp_success),
            "settle_steps": int(self.last_settle_steps),
            "info": {
                "p": int(p),
                "p_actual": int(p_actual),
                "grasp_realism": bool(self.grasp_realism),
                "grasp_failure_prob": GRASP_FAILURE_PROB,
                "grasp_noise": int(p_actual - p_vertex),
                "delta_clamped": self.last_delta_clamped.copy(),
                "lift": lift,
                "gripper_target": target,
                "settle_converged": bool(settle_converged),
                "max_node_speed": self.max_node_speed(),
                "attach_gate_failed": False,
                "approach_steps": int(self.last_approach_steps),
                "approach_dwell_steps": int(self.last_approach_dwell_steps),
                "attach_rel_vel_max": float(np.nanmax(self.last_attach_rel_vel)),
                "attach_offset_max": float(np.nanmax(self.last_attach_offset)),
                "mapped_parameters": mapped_parameters(self.params) if self.params is not None else None,
            },
        }

    def step_primitive_batch(
        self,
        p: np.ndarray,
        delta: np.ndarray,
        lift: Sequence[str],
        *,
        vel_threshold: float = 1e-3,
        max_steps: int = 5000,
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        """Execute one batched primitive with per-env p/delta/lift/grasp outcomes."""

        self._require_reset()
        assert self.rod_entity is not None and self.gripper_link is not None
        if not self.supports_per_env_grasp():
            raise RuntimeError("DLO-Lab rod_entity lacks per-env attach/detach hooks")

        n_vertices = self._n_vertices()
        p_array = np.asarray(p, dtype=int)
        if p_array.shape != (self.n_envs,):
            raise ValueError(f"p must have shape ({self.n_envs},), got {p_array.shape}")
        # Rev 6 C1: `p` is the POLICY action space, bounded by K (= 32), not by
        # the solver's vertex count.  Validating against `n_vertices` was the
        # exact defect the pilot measured: at n=64 every policy index passed
        # this check and then addressed the wrong material point, and at n=100
        # the rear 68.7% of the rope silently left the action space.
        if np.any((p_array < 0) | (p_array >= self.K)):
            raise IndexError(f"policy grasp nodes must be inside [0, {self.K})")

        delta_array = np.asarray(delta, dtype=float)
        if delta_array.shape != (self.n_envs, 3):
            raise ValueError(f"delta must have shape ({self.n_envs}, 3), got {delta_array.shape}")
        if not np.all(np.isfinite(delta_array)):
            raise ValueError("delta contains non-finite values")
        if self.quasi_static:
            # R4: horizontal-δ semantics — z is ignored (dimension kept at R³)
            # and the norm clamp applies to ‖δ_xy‖.
            delta_array = delta_array.copy()
            delta_array[:, 2] = 0.0
        norms = np.linalg.norm(delta_array, axis=1)
        scale = np.ones_like(norms)
        over = norms > MAX_DELTA_NORM
        scale[over] = MAX_DELTA_NORM / norms[over]
        delta_clamped = delta_array * scale[:, None]

        lift_values = [str(value) for value in lift]
        if len(lift_values) != self.n_envs:
            raise ValueError(f"lift must contain {self.n_envs} entries")
        for lift_value in lift_values:
            if lift_value not in LIFT_HEIGHTS:
                raise ValueError(f"lift must be one of {sorted(LIFT_HEIGHTS)}, got {lift_value!r}")

        # AT-1H window opens BEFORE the grasp so the pre-move constraint
        # impulse (the physical cause of every logged Stage 2 exception) is
        # inside the measured span, matching `probe_begin_primitive()`.
        self._at1h_begin()
        X_before = self.get_centerline_batch()
        grasp_rng = self._rng if rng is None else rng
        raw_before = self.get_centerline_raw_batch()

        # Rev 6 C1/C3: policy index -> raw vertex by per-env arc length, THEN
        # the ± arc-length placement error at the actual grasp point.
        p_vertex = arc_length_vertex_index(raw_before, p_array, self.K)
        sampled = [
            sample_grasp(int(node), n_vertices, grasp_rng, self.grasp_realism)
            for node in p_vertex
        ]
        p_actual = np.asarray([item[0] for item in sampled], dtype=int)
        grasp_success = np.asarray([item[1] for item in sampled], dtype=bool)
        self.last_grasp_actual_nodes = p_actual.copy()
        self.last_grasp_actual_node = int(p_actual[0]) if self.n_envs == 1 else None
        self.last_grasp_target_vertices = p_vertex.copy()
        sampled_success = grasp_success.copy()

        # P9 Rev 4 (residual-energy repair, 2026-08-02): capture the FULL
        # solver-truth rod state BEFORE the primitive perturbs anything, so a
        # failed grasp can be rewound to a state the solver itself produced
        # rather than to a synthesized one.  Snapshotting a state that never
        # existed (positions restored but velocity/theta/omega/twist zeroed and
        # the material frames re-seeded from scratch) breaks the elastic
        # equilibrium the rope was holding and injects energy that is charged to
        # the NEXT primitive -- the measured cause of the AT-1H absolute-cap
        # exceptions.  Rev 5: the snapshot is now UNCONDITIONAL, because the
        # attach gate can fail a grasp the sampler accepted and the rewind
        # target must already exist when the approach starts.  It is a
        # read-only state dump (no scene step, no write), so envs that never
        # fail stay bit-identical to a no-failure run (AT-17).
        pre_primitive_state = self._snapshot_rod_state()

        # Rev 5 two-stage approach + attach gate replaces the one-step
        # teleport-and-attach.  Envs that do not clear the relative-velocity
        # gate inside APPROACH_MAX_STEPS join the sampled failures under the
        # single grasp-failure contract below (snapshot rewind, no energy
        # injection).
        self._batched_active_nodes = np.full(self.n_envs, -1, dtype=int)
        approach = self._approach_and_attach(p_actual, grasp_success, per_env=True)
        grasp_success = grasp_success & approach["attached"]
        self.last_grasp_successes = grasp_success.copy()
        self.last_grasp_success = bool(np.all(grasp_success))
        self._step_scene()

        target = self._move_prepared_batch(
            delta_clamped, lift_values, vel_threshold=vel_threshold
        )

        # D2 (env-correction Rev 2 §1.4): verified detach against solver truth.
        # `_batched_active_nodes` is only released after verification succeeds —
        # discarding it first (the previous behavior) threw away the sole
        # record of which slot leaked.
        detach_residuals, detach_escalations = self._verified_detach_batch()
        self._batched_active_nodes = None
        self._step_scene()

        settle_converged, settle_steps = self.settle_batch(
            vel_threshold=vel_threshold,
            max_steps=max_steps,
        )
        X_after = self.get_centerline_batch()

        restoration_drift_max = 0.0
        restoration_drift_mean = 0.0
        if not np.all(grasp_success):
            raw_after = self.get_centerline_raw_batch()
            # Integrity instrumentation (M4 gate advisory): measure the free-evolution
            # drift being erased by the failure-contract restoration BEFORE overwriting.
            failed_drift = np.linalg.norm(
                raw_after[~grasp_success] - raw_before[~grasp_success], axis=-1
            )
            restoration_drift_max = float(failed_drift.max()) if failed_drift.size else 0.0
            restoration_drift_mean = float(failed_drift.mean()) if failed_drift.size else 0.0
            # D4 (env-correction Rev 2 §1.5, strengthened to satisfy AT-17;
            # repaired in Rev 4): restore ONLY the failed envs, and restore the
            # FULL pre-primitive solver state verbatim through the scoped state
            # kernel.  The design's literal one-liner
            # (`place_rod_vertices_batch(raw_after, reinit_env_indices=failed)`)
            # still zeroes every env's velocities, re-parks every gripper and
            # inserts an extra scene step, which measurably perturbs the
            # successful envs' theta/omega/twist and breaks AT-17's
            # bit-identity requirement.  The scoped restore below writes
            # nothing outside the failed envs and steps nothing, so
            # successful envs stay bit-identical to a no-failure run (AT-17)
            # while failed envs are returned EXACTLY to the state they were in
            # when the primitive began -- a real solver equilibrium, so the
            # rewind is energy-neutral by construction.
            self._restore_failed_grasp_envs(
                pre_primitive_state, np.flatnonzero(~grasp_success)
            )
            X_after = self.get_centerline_batch()
            X_after[~grasp_success] = X_before[~grasp_success]
            settle_steps = settle_steps.copy()
            settle_converged = settle_converged.copy()
            settle_steps[~grasp_success] = 0
            settle_converged[~grasp_success] = self.max_node_speed_batch()[~grasp_success] <= float(
                vel_threshold
            )

        self.last_delta_clamped = delta_clamped[0].copy() if self.n_envs == 1 else delta_clamped.copy()
        self.last_settle_steps_batch = settle_steps.copy()
        self.last_settle_converged_batch = settle_converged.copy()
        self.last_settle_steps = int(np.max(settle_steps)) if settle_steps.size else 0
        self.last_settle_converged = bool(np.all(settle_converged))
        at1h = self._at1h_end(lift_values)
        self._assert_finite()

        return {
            "X_before": X_before,
            "X_after": X_after,
            "grasp_success": grasp_success,
            "settle_steps": settle_steps,
            "info": {
                "p": p_array.copy(),
                "p_actual": p_actual,
                "grasp_realism": bool(self.grasp_realism),
                "grasp_failure_prob": GRASP_FAILURE_PROB,
                "grasp_noise": p_actual - p_vertex,
                "p_vertex": p_vertex,
                "delta_clamped": delta_clamped,
                "lift": np.asarray(lift_values, dtype=object),
                "gripper_target": target,
                "settle_converged": settle_converged,
                "max_node_speed": self.max_node_speed_batch(),
                "mapped_parameters": mapped_parameters(self.params) if self.params is not None else None,
                "grasp_mode": "per-env",
                "restoration_drift_max_m": restoration_drift_max,
                "restoration_drift_mean_m": restoration_drift_mean,
                "detach_residuals": int(detach_residuals),
                "detach_escalations": int(detach_escalations),
                "hold_steps_used": int(self.last_hold_steps_used),
                "hold_converged": self.last_hold_converged,
                "quasi_static": bool(self.quasi_static),
                "n_waypoint_steps": list(self.last_waypoint_steps),
                "lower_strain_aborts": int(self.last_lower_strain_aborts),
                "tension_pause_steps": int(self.last_tension_pause_steps),
                "tension_freezes": int(self.last_tension_freezes),
                "lower_z": None if self.last_lower_z is None else [float(v) for v in self.last_lower_z],
                "at1h": at1h,
                "sampled_grasp_success": sampled_success,
                "approach_steps": int(self.last_approach_steps),
                "approach_dwell_steps": int(self.last_approach_dwell_steps),
                "approach_gate_failures": int(approach["gate_failures"]),
                "attach_rel_vel_max": (
                    float(np.nanmax(self.last_attach_rel_vel))
                    if bool(np.isfinite(self.last_attach_rel_vel).any())
                    else None
                ),
                "attach_offset_max": (
                    float(np.nanmax(self.last_attach_offset))
                    if bool(np.isfinite(self.last_attach_offset).any())
                    else None
                ),
                "approach_v_fast": APPROACH_V_FAST,
                "approach_v_slow": self._approach_v_slow(),
                "approach_r_slow": APPROACH_R_SLOW,
                "attach_rel_vel_threshold": ATTACH_REL_VEL,
            },
        }

    def settle(self, vel_threshold: float = 1e-3, max_steps: int = 5000) -> bool:
        self._require_reset()
        threshold = float(vel_threshold)
        if threshold < 0:
            raise ValueError("vel_threshold must be non-negative")
        self.last_settle_steps = 0
        for step in range(int(max_steps) + 1):
            max_speed = self.max_node_speed()
            if max_speed < threshold:
                self.last_settle_steps = step
                self.last_settle_converged = True
                return True
            if step == max_steps:
                break
            self._step_scene()
        self.last_settle_steps = int(max_steps)
        self.last_settle_converged = False
        return False

    def settle_batch(
        self,
        vel_threshold: float = 1e-3,
        max_steps: int = 5000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Settle all envs together while recording per-env first-converged steps."""

        self._require_reset()
        threshold = float(vel_threshold)
        if threshold < 0:
            raise ValueError("vel_threshold must be non-negative")
        budget = int(max_steps)
        if budget < 0:
            raise ValueError("max_steps must be non-negative")

        steps = np.full(self.n_envs, budget, dtype=int)
        converged = np.zeros(self.n_envs, dtype=bool)
        for step in range(budget + 1):
            speeds = self.max_node_speed_batch()
            newly = (~converged) & (speeds < threshold)
            if np.any(newly):
                steps[newly] = step
                converged[newly] = True
            if bool(np.all(converged)):
                break
            if step == budget:
                break
            self._step_scene()

        self.last_settle_steps_batch = steps.copy()
        self.last_settle_converged_batch = converged.copy()
        self.last_settle_steps = int(np.max(steps)) if steps.size else 0
        self.last_settle_converged = bool(np.all(converged))
        return converged, steps

    def max_node_speed_batch(self) -> np.ndarray:
        """Return per-environment maximum rod vertex speed."""

        self._require_reset()
        assert self.rod_entity is not None
        vels = np.asarray(self.rod_entity.get_all_vels(), dtype=float)
        if vels.size == 0:
            return np.zeros(self.n_envs, dtype=float)
        if vels.ndim == 2:
            vels = vels.reshape(1, *vels.shape)
        speeds = np.linalg.norm(vels, axis=-1)
        return np.max(speeds, axis=1).reshape(self.n_envs)

    def max_node_speed(self) -> float:
        speeds = self.max_node_speed_batch()
        return float(np.max(speeds)) if speeds.size else 0.0

    # -- AT-1H always-on counters (env-correction Rev 3) -----------------

    def _at1h_begin(self) -> None:
        """Open a per-primitive AT-1H measurement window."""
        if not self.at1h_counters:
            return
        self._at1h_v_peak = np.zeros(self.n_envs, dtype=float)
        self._at1h_strain_peak = np.zeros(self.n_envs, dtype=float)
        self._at1h_ke_peak = np.zeros(self.n_envs, dtype=float)
        self._at1h_samples = 0
        self._at1h_active = True

    def _at1h_observe(self) -> None:
        """Fold one post-step sample into the running per-env maxima.

        Definitions are taken verbatim from the acceptance battery probe so
        the training-time counters are directly comparable to the gate
        evidence: speed is the max per-node velocity norm, KE is
        ``0.5 * segment_mass * Σ‖v‖²`` (Rev 6 C2: segment mass is derived from
        the rope's total mass, so KE is discretization-invariant) and strain is
        the max
        ``|edge/rest − 1|``.
        """
        assert self.rod_entity is not None
        vels = np.asarray(self.rod_entity.get_all_vels(), dtype=float)
        if vels.size:
            if vels.ndim == 2:
                vels = vels.reshape(1, *vels.shape)
            speeds = np.linalg.norm(vels, axis=-1)
            np.maximum(
                self._at1h_v_peak, np.max(speeds, axis=1).reshape(self.n_envs),
                out=self._at1h_v_peak,
            )
            ke = 0.5 * segment_mass_kg(self.params) * np.sum(vels ** 2, axis=(1, 2))
            np.maximum(self._at1h_ke_peak, ke.reshape(self.n_envs), out=self._at1h_ke_peak)
        if self.params is not None:
            np.maximum(
                self._at1h_strain_peak, self._max_edge_strain_batch(),
                out=self._at1h_strain_peak,
            )
        self._at1h_samples += 1

    def _at1h_end(self, lift_values: Sequence[str] | None) -> dict[str, Any] | None:
        """Close the window and return the per-env AT-1H counter payload.

        ``grav_pe`` is per-env because ``lift`` is a per-env action; the
        battery's scalar ``rope_mass * g * lift_height`` becomes a vector
        with the same definition.  Rev 6 C2: the rope mass is
        ``params.rope_mass_total_kg`` directly rather than
        ``n_segments * SEGMENT_MASS_BASE``, so the KE/PE denominator no longer
        moves when the discretization changes.
        """
        if not self.at1h_counters:
            return None
        self._at1h_active = False
        if self.params is None:
            return None
        rope_mass = float(self.params.rope_mass_total_kg)
        if lift_values is None:
            heights = np.full(self.n_envs, LIFT_HEIGHTS["high"], dtype=float)
        else:
            heights = np.asarray(
                [LIFT_HEIGHTS[str(value)] for value in lift_values], dtype=float
            )
        grav_pe = rope_mass * AT1H_GRAVITY * heights
        # Terminal-cleanliness input.  The battery reads `arclen_final` from
        # the probe's last post-settle sample; the equivalent here is one
        # vertex read at window close (once per primitive, not per step).
        raw = np.asarray(self._raw_batch(), dtype=float)
        arclen = np.linalg.norm(raw[:, 1:, :] - raw[:, :-1, :], axis=-1).sum(axis=1)
        arclen_dev = np.abs(arclen / float(self.params.length_m) - 1.0)
        return {
            "v_peak": self._at1h_v_peak.copy(),
            "strain_peak": self._at1h_strain_peak.copy(),
            "ke_peak": self._at1h_ke_peak.copy(),
            "grav_pe": grav_pe,
            "ke_over_pe": self._at1h_ke_peak / grav_pe,
            "arclen_dev": arclen_dev,
            "samples": int(self._at1h_samples),
        }


    def _raw_batch(self) -> np.ndarray:
        self._require_reset()
        assert self.rod_entity is not None
        return np.asarray(self.rod_entity.get_all_verts(), dtype=float)

    def _place_rod_vertices_batched(
        self, vertices: np.ndarray, *, reinit_env_indices: np.ndarray | None = None
    ) -> None:
        self._require_reset()
        assert self.rod_entity is not None
        n_vertices = self._n_vertices()
        verts = np.asarray(vertices, dtype=float)
        if verts.shape == (n_vertices, 3):
            batched = np.broadcast_to(verts, (self.n_envs, n_vertices, 3)).copy()
        elif verts.shape == (self.n_envs, n_vertices, 3):
            batched = verts.copy()
        else:
            raise ValueError(
                "vertices must have shape "
                f"({n_vertices}, 3) or ({self.n_envs}, {n_vertices}, 3), got {verts.shape}"
            )
        if not np.all(np.isfinite(batched)):
            raise ValueError("vertices contain non-finite values")

        self._detach_existing_attachments()
        zeros = np.zeros_like(batched)
        self.rod_entity.set_position(batched)
        self.rod_entity.set_velocity(zeros)
        # Reset the gripper to a safe finite pose as well: a primitive that
        # failed with non-finite rope state can leave the gripper at NaN
        # coordinates (it is positioned from raw vertices at grasp time), and
        # a NaN gripper re-poisons every subsequent attach/move — the
        # persistence mechanism behind consecutive covenant discards observed
        # in the P1-M2 smoke.
        safe_gripper = np.zeros((self.n_envs, 3), dtype=float)
        safe_gripper[:, 2] = GRIPPER_PARK_Z
        self._set_gripper_positions(safe_gripper)
        # D1 (env-correction Rev 2 §1.4): unconditionally clear every vertex
        # constraint of the scoped envs at reset placement, BEFORE the state
        # rewrite.  Python-side attachment records are deliberately not
        # consulted — trusting them is the original defect (the upstream
        # detach kernel leaks and `_detach_existing_attachments` only clears
        # what Python remembers).
        self._clear_vertex_constraints(env_indices=reinit_env_indices)
        self._reinitialize_edge_state(batched, env_indices=reinit_env_indices)
        self._step_scene()

    def _reinitialize_edge_state(
        self, batched_positions: np.ndarray, *, env_indices: np.ndarray | None = None
    ) -> None:
        """Recompute the rod's full edge/frame state from placed positions.

        Adapter-internal P1 addition (interface unchanged): the public
        ``set_position``/``set_velocity`` targets only write vertex pos/vel.
        Edge twist state (theta/omega), material frames (d1/d2/d3, refs), and
        internal-vertex state (kb/twist) are incrementally updated by the
        solver, so non-finite values there survive a light reset and
        re-pollute the next step (observed in P1-M2: envs stayed non-finite
        through 3 reseed retries).  This mirrors the build-time
        ``_kernel_finalize_states`` math per environment: frames rebuilt from
        the placed centerline, theta/omega/twist zeroed ("at rest,
        untwisted" — the placement contract the P0 analytic init curves
        assume).  Upstream (frozen c5026a9) is untouched; ``set_theta`` is
        bypassed via the solver kernel because its wrapper passes the wrong
        kwarg name (omega=theta) — documented upstream bug.
        """

        assert self.rod_entity is not None and self.gs is not None
        import torch

        pos = np.asarray(batched_positions, dtype=float)  # (B, V, 3)
        edge = pos[:, 1:, :] - pos[:, :-1, :]  # (B, E, 3)
        length = np.linalg.norm(edge, axis=-1)  # (B, E)
        safe_length = np.where(length > 0.0, length, 1.0)
        d3 = edge / safe_length[..., None]

        n_envs, n_edges = length.shape
        d1 = np.zeros_like(d3)
        # First edge: any unit vector perpendicular to d3[0].
        ref = np.zeros((n_envs, 3))
        smallest = np.argmin(np.abs(d3[:, 0, :]), axis=-1)
        ref[np.arange(n_envs), smallest] = 1.0
        first = np.cross(d3[:, 0, :], ref)
        first /= np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
        d1[:, 0, :] = first
        # Parallel transport along the rod (Rodrigues rotation t_{e-1} -> t_e).
        for e in range(1, n_edges):
            t1 = d3[:, e - 1, :]
            t2 = d3[:, e, :]
            axis = np.cross(t1, t2)
            sin_a = np.linalg.norm(axis, axis=-1)
            cos_a = np.clip(np.sum(t1 * t2, axis=-1), -1.0, 1.0)
            prev = d1[:, e - 1, :]
            rotated = prev.copy()
            mask = sin_a > 1e-12
            if np.any(mask):
                k = axis[mask] / sin_a[mask][..., None]
                v = prev[mask]
                c = cos_a[mask][..., None]
                s = sin_a[mask][..., None]
                rotated[mask] = (
                    v * c
                    + np.cross(k, v) * s
                    + k * np.sum(k * v, axis=-1, keepdims=True) * (1.0 - c)
                )
            # Re-orthonormalize against t2 to keep the frame exact.
            rotated -= np.sum(rotated * t2, axis=-1, keepdims=True) * t2
            rotated /= np.maximum(np.linalg.norm(rotated, axis=-1, keepdims=True), 1e-12)
            d1[:, e, :] = rotated
        d2 = np.cross(d3, d1)

        # Internal vertices: curvature binormal kb, zero twist.
        t_a = d3[:, :-1, :]
        t_b = d3[:, 1:, :]
        denom = 1.0 + np.sum(t_a * t_b, axis=-1, keepdims=True)
        kb = 2.0 * np.cross(t_a, t_b) / np.maximum(denom, 1e-12)

        device = self.gs.device
        tc = self.gs.tc_float

        def t(arr: np.ndarray) -> "torch.Tensor":
            return torch.as_tensor(np.ascontiguousarray(arr), dtype=tc, device=device)

        n_internal = kb.shape[1]
        zeros_e = np.zeros_like(length)
        substep = self.rod_entity._sim.cur_substep_local
        if env_indices is None:
            envs_idx = torch.arange(n_envs, dtype=torch.int32, device=device)
        else:
            scoped = np.asarray(env_indices, dtype=np.int32).reshape(-1)
            if scoped.size == 0:
                return
            envs_idx = torch.as_tensor(scoped, dtype=torch.int32, device=device)
        # One atomic per-env full-state write (pos, vel, fixed, theta, omega,
        # edge, length, frames, kb, twist, kappa_rest).  kappa_rest is zero
        # because the rod is built with rest_state="straight" (zero rest
        # curvature); fixed flags are zero (free rope); twist/theta/omega are
        # zero per the "at rest, untwisted" placement contract.
        self.rod_entity._solver._kernel_set_state(
            substep,
            envs_idx,
            t(pos),
            t(np.zeros_like(pos)),
            torch.zeros((n_envs, pos.shape[1]), dtype=torch.bool, device=device),
            t(zeros_e),
            t(zeros_e),
            t(edge),
            t(length),
            t(d1),
            t(d2),
            t(d3),
            t(d1),
            t(d2),
            t(kb),
            t(np.zeros((n_envs, n_internal))),
            t(np.zeros((n_envs, n_internal, 2))),
        )

    def _detach_existing_attachments(self) -> None:
        if self.rod_entity is None:
            return
        if self.active_node is not None:
            self.rod_entity.detach_from_rigid_link([self.active_node])
            self.active_node = None
        if self._batched_active_nodes is not None:
            for env_idx, node in enumerate(self._batched_active_nodes):
                if int(node) < 0:
                    continue
                if self.supports_per_env_grasp():
                    self.rod_entity.detach_from_rigid_link_with_envs_idx([int(node)], int(env_idx))
                else:
                    self.rod_entity.detach_from_rigid_link([int(node)])
            self._batched_active_nodes = None

    def _vertex_constrained_mask(self) -> np.ndarray:
        """Solver-truth attachment table snapshot, shape ``(n_envs, V)`` bool.

        Reads ``rod_solver.vertex_constraints.constrained`` directly; never
        derives from Python-side records (D1/D2 rationale).
        """
        assert self.rod_entity is not None
        solver = self.rod_entity._solver
        return np.asarray(
            solver.vertex_constraints.constrained.to_numpy(), dtype=bool
        ).T.copy()

    def _clear_vertex_constraints(self, env_indices: np.ndarray | None = None) -> None:
        """D1: force ``constrained=False`` on all vertices of the scoped envs.

        Uses the frozen upstream detach kernels only (no upstream edits):
        the all-envs kernel when unscoped, the per-env kernel otherwise.
        """
        assert self.rod_entity is not None
        solver = self.rod_entity._solver
        n_vertices = self._n_vertices()
        if env_indices is None:
            for i_v in range(n_vertices):
                solver._kernel_detach_vertex(int(i_v))
            return
        scoped = np.asarray(env_indices, dtype=int).reshape(-1)
        for env_idx in scoped:
            for i_v in range(n_vertices):
                solver._kernel_detach_vertex_with_envs_idx(int(i_v), int(env_idx))

    def _verified_detach_batch(self) -> tuple[int, int]:
        """D2: detach every batched attachment, then verify against solver truth.

        Returns ``(residual_after_first_pass, escalations)``.  The retry pass
        re-issues the entity-level detach for residual slots that match the
        recorded node; the escalation pass force-clears any survivor with the
        solver kernel and counts it (AT-16 exposes the counter — a nonzero
        count means the upstream kernel is still flaky and D2 is doing real
        work).  A slot that survives escalation raises fail-closed.
        """
        assert self.rod_entity is not None
        solver = self.rod_entity._solver
        recorded = self._batched_active_nodes
        if recorded is not None:
            for env_idx, node in enumerate(recorded):
                if int(node) < 0:
                    continue
                self.rod_entity.detach_from_rigid_link_with_envs_idx(
                    [int(node)], int(env_idx)
                )
        mask = self._vertex_constrained_mask()
        residual_first = int(mask.sum())
        if residual_first:
            # Retry once through the entity API for recorded slots; residual
            # vertices that Python never recorded go straight to escalation.
            for env_idx, node_idx in zip(*np.nonzero(mask)):
                if recorded is not None and int(recorded[int(env_idx)]) == int(node_idx):
                    self.rod_entity.detach_from_rigid_link_with_envs_idx(
                        [int(node_idx)], int(env_idx)
                    )
            mask = self._vertex_constrained_mask()
        escalations = 0
        if mask.any():
            for env_idx, node_idx in zip(*np.nonzero(mask)):
                solver._kernel_detach_vertex_with_envs_idx(int(node_idx), int(env_idx))
                escalations += 1
            mask = self._vertex_constrained_mask()
            if mask.any():
                leftover = [
                    (int(e), int(v)) for e, v in zip(*np.nonzero(mask))
                ]
                raise ConstraintCovenantError(
                    "constraint covenant: vertex constraints survived forced "
                    f"clear after detach: {leftover}"
                )
        self.last_detach_residuals = residual_first
        self.detach_escalation_total += escalations
        return residual_first, escalations

    # -- P9 Rev 4 faithful failed-grasp rewind ---------------------------
    # `_ROD_STATE_FIELDS` is the exact argument order shared by the frozen
    # upstream `_kernel_get_state` / `_kernel_set_state` pair (rod_solver.py
    # 2207-2306).  Keeping one tuple means the reader and the writer cannot
    # drift apart silently.
    _ROD_STATE_FIELDS = (
        "pos", "vel", "fixed", "theta", "omega", "edge", "length",
        "d1", "d2", "d3", "d1_ref", "d2_ref", "kb", "twist", "kappa_rest",
    )

    def _snapshot_rod_state(self) -> dict[str, "torch.Tensor"]:
        """Read the complete solver-truth rod state for every environment.

        Dual of `_kernel_set_state`: every field the setter writes is read
        back by the frozen upstream `_kernel_get_state`, so a snapshot/restore
        round trip is the identity.  Nothing is synthesized and nothing is
        zeroed, which is the whole point -- the rope's elastic equilibrium
        lives in `theta`/`omega`/`twist` and the material frames, not in the
        vertex positions alone.
        """

        assert self.rod_entity is not None and self.gs is not None
        import torch

        solver = self.rod_entity._solver
        device = self.gs.device
        tc = self.gs.tc_float
        n_envs = self.n_envs
        n_v = int(solver._n_vertices)
        n_e = int(solver._n_edges)
        n_iv = int(solver.n_internal_vertices)

        def real(*shape: int) -> "torch.Tensor":
            return torch.zeros(shape, dtype=tc, device=device)

        snapshot = {
            "pos": real(n_envs, n_v, 3),
            "vel": real(n_envs, n_v, 3),
            "fixed": torch.zeros((n_envs, n_v), dtype=torch.bool, device=device),
            "theta": real(n_envs, n_e),
            "omega": real(n_envs, n_e),
            "edge": real(n_envs, n_e, 3),
            "length": real(n_envs, n_e),
            "d1": real(n_envs, n_e, 3),
            "d2": real(n_envs, n_e, 3),
            "d3": real(n_envs, n_e, 3),
            "d1_ref": real(n_envs, n_e, 3),
            "d2_ref": real(n_envs, n_e, 3),
            "kb": real(n_envs, n_iv, 3),
            "twist": real(n_envs, n_iv),
            "kappa_rest": real(n_envs, n_iv, 2),
        }
        solver._kernel_get_state(
            self.rod_entity._sim.cur_substep_local,
            *(snapshot[name] for name in self._ROD_STATE_FIELDS),
        )
        return snapshot

    def _restore_failed_grasp_envs(
        self, snapshot: dict[str, "torch.Tensor"] | None, failed_env_indices: np.ndarray
    ) -> None:
        """D4 (Rev 4): scoped failed-grasp rewind with zero successful-env impact.

        Clears the failed envs' constraint slots (D1 defense-in-depth) and
        rewrites their full solver state from `snapshot` -- the state captured
        at the top of this primitive, before anything touched the rope --
        through the scoped `_kernel_set_state` path.  No entity-level
        full-batch write, no gripper re-park and no extra scene step happen,
        so envs outside the scope remain bit-identical to an execution without
        any grasp failure (AT-17).

        Rev 3 restored positions only and zeroed velocity/theta/omega/twist
        while re-seeding the material frames by fresh parallel transport.  On a
        rope carrying a tight hinge that is not a rewind: it is a synthesized
        state that was never in equilibrium, and the solver converts the
        resulting unbalanced bending moment into a metre-per-second ejection on
        the next scene step -- attributed to the FOLLOWING primitive, because
        this primitive's AT-1H window has already closed.  Restoring the real
        prior state removes the injection at its source instead of masking it.
        """
        scoped = np.asarray(failed_env_indices, dtype=int).reshape(-1)
        if scoped.size == 0:
            return
        if snapshot is None:
            raise RuntimeError(
                "failed-grasp rewind requested without a pre-primitive snapshot"
            )
        assert self.rod_entity is not None and self.gs is not None
        import torch

        if not bool(torch.isfinite(snapshot["pos"][scoped]).all()):
            raise ValueError("snapshot positions contain non-finite values")

        self._clear_vertex_constraints(env_indices=scoped)
        envs_idx = torch.as_tensor(
            scoped.astype(np.int32), dtype=torch.int32, device=self.gs.device
        )
        self.rod_entity._solver._kernel_set_state(
            self.rod_entity._sim.cur_substep_local,
            envs_idx,
            *(snapshot[name] for name in self._ROD_STATE_FIELDS),
        )

    def _place_rod_vertices(self, vertices: np.ndarray) -> None:
        self._require_reset()
        assert self.rod_entity is not None
        n_vertices = self._n_vertices()
        verts = np.asarray(vertices, dtype=float)
        if verts.shape != (n_vertices, 3):
            raise ValueError(f"vertices must have shape ({n_vertices}, 3), got {verts.shape}")
        if not np.all(np.isfinite(verts)):
            raise ValueError("vertices contain non-finite values")

        batched = np.broadcast_to(verts, (self.n_envs, n_vertices, 3)).copy()
        zeros = np.zeros_like(batched)
        self.rod_entity.set_position(batched)
        self.rod_entity.set_velocity(zeros)
        self._reinitialize_edge_state(batched)
        self._step_scene()

    def _gripper_positions(self) -> np.ndarray:
        self._require_reset()
        assert self.gripper_link is not None
        pos = self.gripper_link.get_pos()
        if hasattr(pos, "detach"):
            pos = pos.detach().cpu().numpy()
        return np.asarray(pos, dtype=float).reshape(self.n_envs, 3).copy()

    def _set_gripper_positions(self, positions: np.ndarray) -> None:
        self._require_reset()
        assert self.gripper_entity is not None and self.gs is not None
        pos = np.asarray(positions, dtype=self.gs.np_float).reshape(self.n_envs, 3)
        self.gripper_entity.set_pos(pos, zero_velocity=True)

    def _step_scene(self) -> None:
        self._require_reset()
        assert self.scene is not None
        self.scene.step(update_visualizer=False, refresh_visualizer=False)
        # AT-1H instrumentation is hooked HERE rather than at each leg's
        # call site so no future leg can be added without being measured,
        # and so the sample lands after the step exactly like the
        # acceptance battery's `Probe._step_scene` override does.
        if self._at1h_active:
            self._at1h_observe()

    def _rollout(self, steps: int) -> None:
        for _ in range(max(0, int(steps))):
            self._step_scene()

    def _assert_finite(self) -> None:
        raw = self._raw_batch()
        if not np.all(np.isfinite(raw)):
            raise FloatingPointError("DLO-Lab rod vertices contain non-finite values")
        assert self.rod_entity is not None
        vels = np.asarray(self.rod_entity.get_all_vels(), dtype=float)
        if not np.all(np.isfinite(vels)):
            raise FloatingPointError("DLO-Lab rod velocities contain non-finite values")

    def _n_vertices(self) -> int:
        if self.params is None:
            raise RuntimeError("DLOLabEnv.reset must be called first")
        return int(self.params.n_segments)

    def _require_reset(self) -> None:
        if self.scene is None or self.rod_entity is None:
            raise RuntimeError("DLOLabEnv.reset must be called first")

    @staticmethod
    def _validate_params(params: RopeParams) -> None:
        if params.length_m <= 0:
            raise ValueError("length_m must be positive")
        if params.n_segments < 2:
            raise ValueError("n_segments must be at least 2")
        if params.radius <= 0:
            raise ValueError("radius must be positive")
        if params.bend_stiffness <= 0:
            raise ValueError("bend_stiffness multiplier must be positive")
        if params.twist_stiffness <= 0:
            raise ValueError("twist_stiffness multiplier must be positive")
        if params.friction < 0:
            raise ValueError("friction multiplier must be non-negative")
        if params.rope_mass_total_kg <= 0:
            raise ValueError("rope_mass_total_kg must be positive")

    def _assert_domain_matches_config(self, params: RopeParams) -> None:
        """Rev 6 B: fail closed when `sim` and the rope domain disagree.

        `sim.n_segments` / `sim.rope_mass_total` are the launch-time
        declaration; `RopeParams` is what the caller actually built.  A silent
        mismatch would train on a different rope than the SHA-pinned config
        advertises, which is the same class of defect the corrected-env
        resolver was written to stop.
        """

        if (
            self.cfg_n_segments is not None
            and int(params.n_segments) != self.cfg_n_segments
        ):
            raise ValueError(
                f"env config: `sim.n_segments` = {self.cfg_n_segments} but the rope "
                f"domain supplies n_segments = {int(params.n_segments)}"
            )
        if self.cfg_rope_mass_total is not None and not np.isclose(
            float(params.rope_mass_total_kg), self.cfg_rope_mass_total, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                f"env config: `sim.rope_mass_total` = {self.cfg_rope_mass_total} but the "
                f"rope domain supplies rope_mass_total_kg = {float(params.rope_mass_total_kg)}"
            )
