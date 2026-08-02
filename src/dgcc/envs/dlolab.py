"""DLO-Lab/Genesis rod adapter for the P0-M3 primary primitive milestone."""

from __future__ import annotations

import os
from dataclasses import asdict
from math import ceil
from typing import Any, Sequence

import numpy as np

from dgcc.envs.base import DLOEnvBase, RopeParams
from dgcc.utils.seeding import seed_everything

STRETCH_BASE = 8.0e5
BEND_BASE = 1.0e5
TWIST_BASE = 1.0e4
MU_S_BASE = 0.30
MU_K_RATIO = 0.80
SEGMENT_MASS_BASE = 1.0e-3
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
# neighborhood radius ≈ 2 segment intervals, clearance ε above the pile, and
# the strain fail-safe trigger at half the AT-4 threshold.
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
GRASP_FAILURE_PROB = 0.05
GRASP_NOISE_CHOICES = (-1, 0, 1)
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


def sample_grasp(
    p: int,
    n_nodes: int,
    rng: np.random.Generator,
    enabled: bool = True,
) -> tuple[int, bool]:
    """Sample the M3 grasp-realism noise/failure model without touching Genesis.

    Boundary semantics: the ±1 offset is drawn uniformly and then clamped to the
    valid node range, so the two end nodes self-select with probability 2/3
    (an outward miss re-grasps the end node); interior nodes stay uniform ±1.
    """

    node = int(p)
    n = int(n_nodes)
    if n < 1:
        raise ValueError("n_nodes must be at least 1")
    if node < 0 or node >= n:
        raise IndexError(f"grasp node {node} outside [0, {n})")
    if not enabled:
        return node, True

    offset = int(rng.choice(GRASP_NOISE_CHOICES))
    actual = int(np.clip(node + offset, 0, n - 1))
    success = bool(rng.random() >= GRASP_FAILURE_PROB)
    return actual, success


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


def stiffness_bases() -> dict[str, float]:
    """Return the simulator-unit bases used for RopeParams multipliers."""

    return {
        "stretch_base_K": STRETCH_BASE,
        "bend_base_E": BEND_BASE,
        "twist_base_G": TWIST_BASE,
        "mu_s_base": MU_S_BASE,
        "mu_k_ratio": MU_K_RATIO,
        "segment_mass_base": SEGMENT_MASS_BASE,
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
        "segment_mass": SEGMENT_MASS_BASE,
        "segment_radius": float(params.radius),
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
        self.last_hold_steps_used = 0
        self.last_hold_converged: bool | None = None
        self.last_waypoint_steps: list[int] = []
        self.last_lower_strain_aborts = 0
        self.last_lower_z: np.ndarray | None = None
        self.last_tension_pause_steps = 0
        self.last_tension_freezes = 0
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
            "stiffness_bases": stiffness_bases(),
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

    def grasp(self, p: int) -> bool:
        self._require_reset()
        assert self.rod_entity is not None and self.gripper_link is not None
        node = int(p)
        n_vertices = self._n_vertices()
        if node < 0 or node >= n_vertices:
            raise IndexError(f"grasp node {node} outside [0, {n_vertices})")

        verts = self._raw_batch()
        self._set_gripper_positions(verts[:, node, :])
        self._step_scene()
        self.rod_entity.attach_to_rigid_link(self.gripper_link, [node])
        self.active_node = node
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
        p_actual, sampled_success = sample_grasp(p, self._n_vertices(), self._rng, self.grasp_realism)
        self.last_grasp_actual_node = p_actual
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
                    "grasp_noise": int(p_actual - int(p)),
                    "delta_clamped": delta_vec.copy(),
                    "lift": lift,
                    "gripper_target": None,
                    "settle_converged": measured_converged,
                    "max_node_speed": measured_speed,
                    "mapped_parameters": mapped_parameters(self.params) if self.params is not None else None,
                },
            }

        grasp_success = self.grasp(p_actual)
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
                "grasp_noise": int(p_actual - int(p)),
                "delta_clamped": self.last_delta_clamped.copy(),
                "lift": lift,
                "gripper_target": target,
                "settle_converged": bool(settle_converged),
                "max_node_speed": self.max_node_speed(),
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
        if np.any((p_array < 0) | (p_array >= n_vertices)):
            raise IndexError(f"grasp nodes must be inside [0, {n_vertices})")

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

        sampled = [
            sample_grasp(int(node), n_vertices, grasp_rng, self.grasp_realism)
            for node in p_array
        ]
        p_actual = np.asarray([item[0] for item in sampled], dtype=int)
        grasp_success = np.asarray([item[1] for item in sampled], dtype=bool)
        self.last_grasp_actual_nodes = p_actual.copy()
        self.last_grasp_successes = grasp_success.copy()
        self.last_grasp_actual_node = int(p_actual[0]) if self.n_envs == 1 else None
        self.last_grasp_success = bool(np.all(grasp_success))

        # P9 Rev 4 (residual-energy repair, 2026-08-02): capture the FULL
        # solver-truth rod state BEFORE the primitive perturbs anything, so a
        # failed grasp can be rewound to a state the solver itself produced
        # rather than to a synthesized one.  Snapshotting a state that never
        # existed (positions restored but velocity/theta/omega/twist zeroed and
        # the material frames re-seeded from scratch) breaks the elastic
        # equilibrium the rope was holding and injects energy that is charged to
        # the NEXT primitive -- the measured cause of the AT-1H absolute-cap
        # exceptions.  Only taken when the sample actually contains a failure
        # (~GRASP_FAILURE_PROB of primitives), so the steady-state cost is nil.
        pre_primitive_state = (
            None if bool(np.all(grasp_success)) else self._snapshot_rod_state()
        )

        env_indices = np.arange(self.n_envs)
        self._set_gripper_positions(raw_before[env_indices, p_actual, :])
        self._step_scene()

        self._batched_active_nodes = np.full(self.n_envs, -1, dtype=int)
        for env_idx, (node, success) in enumerate(zip(p_actual, grasp_success, strict=True)):
            if not success:
                continue
            self.rod_entity.attach_to_rigid_link_with_envs_idx(
                self.gripper_link,
                [int(node)],
                int(env_idx),
            )
            self._batched_active_nodes[env_idx] = int(node)
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
                "grasp_noise": p_actual - p_array,
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
        ``0.5 * SEGMENT_MASS_BASE * Σ‖v‖²`` and strain is the max
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
            ke = 0.5 * SEGMENT_MASS_BASE * np.sum(vels ** 2, axis=(1, 2))
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
        with the same definition (rope mass = n_segments * SEGMENT_MASS_BASE).
        """
        if not self.at1h_counters:
            return None
        self._at1h_active = False
        if self.params is None:
            return None
        rope_mass = float(self.params.n_segments) * SEGMENT_MASS_BASE
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
