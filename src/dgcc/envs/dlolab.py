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
LIFT_HEIGHTS = {"low": 0.02, "high": 0.15}
GRASP_FAILURE_PROB = 0.05
GRASP_NOISE_CHOICES = (-1, 0, 1)
VALID_INIT_SHAPES = frozenset({"straight", "u_bend", "s_curve", "random_smooth"})


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
        move_step_size: float = 0.002,
        move_hold_steps: int = 20,
        grasp_realism: bool = True,
    ) -> None:
        if n_envs < 1:
            raise ValueError("n_envs must be at least 1")
        self.n_envs = int(n_envs)
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.rod_damping = float(rod_damping)
        self.rod_angular_damping = float(rod_angular_damping)
        self.initial_settle_steps = int(initial_settle_steps)
        self.reset_settle_max_steps = int(reset_settle_max_steps)
        self.move_step_size = float(move_step_size)
        self.move_hold_steps = int(move_hold_steps)
        self.grasp_realism = bool(grasp_realism)

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

    def place_rod_vertices_batch(self, vertices: np.ndarray) -> None:
        """Light-reset all environments from explicit per-env native vertices.

        ``vertices`` may be either ``(N, 3)`` (broadcast to every environment) or
        ``(n_envs, N, 3)`` (distinct curve per environment).  Existing batched
        attachments are cleared, positions are written through
        ``rod_entity.set_position``, and velocities are zeroed.
        """

        self._place_rod_vertices_batched(vertices)

    def light_reset(
        self,
        vertices: np.ndarray,
        *,
        vel_threshold: float = 1e-3,
        max_steps: int = 5000,
    ) -> dict[str, np.ndarray]:
        """Re-place batched native vertices and settle without rebuilding the scene."""

        self.place_rod_vertices_batch(vertices)
        converged, settle_steps = self.settle_batch(
            vel_threshold=vel_threshold,
            max_steps=max_steps,
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

    def _move_prepared(self, delta_vec: np.ndarray, lift: str) -> np.ndarray:
        """Run the waypoint move for an already-validated/clamped delta (A7: clamp once)."""
        self._require_reset()
        if self.active_node is None:
            raise RuntimeError("move called before grasp")

        start = self._gripper_positions()
        lifted = start.copy()
        lifted[:, 2] = LIFT_HEIGHTS[lift]
        target = lifted + delta_vec.reshape(1, 3)
        target[:, 2] = np.maximum(target[:, 2], self._gripper_z_floor())
        self.last_move_target = target.copy()

        current = start
        for waypoint in (lifted, target):
            max_distance = float(np.max(np.linalg.norm(waypoint - current, axis=1)))
            n_steps = max(20, int(ceil(max_distance / self.move_step_size)))
            for alpha in np.linspace(1.0 / n_steps, 1.0, n_steps):
                pos = (1.0 - alpha) * current + alpha * waypoint
                self._set_gripper_positions(pos)
                self._step_scene()
            current = waypoint.copy()

        for _ in range(max(0, self.move_hold_steps)):
            self._set_gripper_positions(target)
            self._step_scene()
        self._assert_finite()
        return target[0].copy() if self.n_envs == 1 else target.copy()

    def _move_prepared_batch(self, delta_vecs: np.ndarray, lift_values: Sequence[str]) -> np.ndarray:
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
        target = lifted + deltas
        target[:, 2] = np.maximum(target[:, 2], self._gripper_z_floor())
        self.last_move_target = target.copy()

        current = start
        for waypoint in (lifted, target):
            max_distance = float(np.max(np.linalg.norm(waypoint - current, axis=1)))
            n_steps = max(20, int(ceil(max_distance / self.move_step_size)))
            for alpha in np.linspace(1.0 / n_steps, 1.0, n_steps):
                pos = (1.0 - alpha) * current + alpha * waypoint
                self._set_gripper_positions(pos)
                self._step_scene()
            current = waypoint.copy()

        for _ in range(max(0, self.move_hold_steps)):
            self._set_gripper_positions(target)
            self._step_scene()
        self._assert_finite()
        return target.copy()

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

        target = self._move_prepared_batch(delta_clamped, lift_values)

        for env_idx, node in enumerate(self._batched_active_nodes):
            if int(node) < 0:
                continue
            self.rod_entity.detach_from_rigid_link_with_envs_idx([int(node)], int(env_idx))
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
            raw_after[~grasp_success] = raw_before[~grasp_success]
            self.place_rod_vertices_batch(raw_after)
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


    def _raw_batch(self) -> np.ndarray:
        self._require_reset()
        assert self.rod_entity is not None
        return np.asarray(self.rod_entity.get_all_verts(), dtype=float)

    def _place_rod_vertices_batched(self, vertices: np.ndarray) -> None:
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
        safe_gripper[:, 2] = LIFT_HEIGHTS["high"]
        self._set_gripper_positions(safe_gripper)
        self._reinitialize_edge_state(batched)
        self._step_scene()

    def _reinitialize_edge_state(self, batched_positions: np.ndarray) -> None:
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
        envs_idx = torch.arange(n_envs, dtype=torch.int32, device=device)
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
