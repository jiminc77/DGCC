"""DLO-Lab/Genesis rod adapter for the P0-M1 smoke milestone."""

from __future__ import annotations

import os
from dataclasses import asdict
from math import ceil
from typing import Any

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
        self.last_bent_reset_converged: bool | None = None

    def reset(self, params: RopeParams, init_shape: str, seed: int) -> dict[str, Any]:
        self._validate_params(params)
        seed_everything(seed)
        self.gs = ensure_genesis_initialized(seed)
        gs = self.gs

        self.params = params
        self.active_node = None
        self.last_settle_steps = 0
        self.last_settle_converged = False
        self.last_bent_reset_converged = None

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
        self.settle(max_steps=self.reset_settle_max_steps)

        normalized_shape = init_shape.lower()
        if normalized_shape == "straight":
            pass
        elif normalized_shape in {"bent", "u_bend"}:
            self.last_bent_reset_converged = self._script_bent_reset(seed)
        else:
            raise ValueError("init_shape must be 'straight', 'bent', or 'u_bend'")

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
            "mapped_parameters": mapped,
            "stiffness_bases": stiffness_bases(),
            "bent_reset_converged": self.last_bent_reset_converged,
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

    def move(self, delta: np.ndarray, lift: str) -> np.ndarray:
        self._require_reset()
        if self.active_node is None:
            raise RuntimeError("move called before grasp")
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

        start = self._gripper_positions()
        lifted = start.copy()
        lifted[:, 2] = LIFT_HEIGHTS[lift]
        target = lifted + delta_vec.reshape(1, 3)
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

    def release(self, vel_threshold: float = 1e-3, max_steps: int = 5000) -> bool:
        self._require_reset()
        assert self.rod_entity is not None
        if self.active_node is not None:
            self.rod_entity.detach_from_rigid_link([self.active_node])
        self.active_node = None
        self._step_scene()
        return self.settle(vel_threshold=vel_threshold, max_steps=max_steps)

    def step_primitive(self, p: int, delta: np.ndarray, lift: str) -> dict[str, Any]:
        X_before = self.get_centerline()
        grasp_success = self.grasp(p)
        target = self.move(delta, lift)
        settle_converged = self.release()
        X_after = self.get_centerline()
        return {
            "X_before": X_before,
            "X_after": X_after,
            "grasp_success": bool(grasp_success),
            "settle_steps": int(self.last_settle_steps),
            "info": {
                "p": int(p),
                "delta_clamped": self.last_delta_clamped.copy(),
                "lift": lift,
                "gripper_target": target,
                "settle_converged": bool(settle_converged),
                "max_node_speed": self.max_node_speed(),
                "mapped_parameters": mapped_parameters(self.params) if self.params is not None else None,
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

    def max_node_speed(self) -> float:
        self._require_reset()
        assert self.rod_entity is not None
        vels = np.asarray(self.rod_entity.get_all_vels(), dtype=float)
        if vels.size == 0:
            return 0.0
        return float(np.max(np.linalg.norm(vels, axis=-1)))

    def _script_bent_reset(self, seed: int) -> bool:
        assert self.params is not None
        rng = np.random.default_rng(seed)
        direction = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
        bend_delta = np.array(
            [0.0, direction * min(MAX_DELTA_NORM, 0.20 * float(self.params.length_m)), 0.0],
            dtype=float,
        )
        node = self._n_vertices() // 2
        self.grasp(node)
        self.move(bend_delta, "high")
        return self.release(max_steps=self.reset_settle_max_steps)

    def _raw_batch(self) -> np.ndarray:
        self._require_reset()
        assert self.rod_entity is not None
        return np.asarray(self.rod_entity.get_all_verts(), dtype=float)

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
