"""MuJoCo cable adapter for the P0-M1 smoke milestone."""

from __future__ import annotations

from dataclasses import asdict
from html import escape
from math import ceil
from typing import Any

import mujoco
import numpy as np

from dgcc.envs.base import DLOEnvBase, RopeParams
from dgcc.phi.resample import resample
from dgcc.utils.seeding import seed_everything


BEND_BASE = 4e6
TWIST_BASE = 1e7
MAX_DELTA_NORM = 0.15
LIFT_HEIGHTS = {"low": 0.02, "high": 0.15}


def _require_mujoco_3() -> None:
    major = int(mujoco.__version__.split(".", maxsplit=1)[0])
    if major < 3:
        raise RuntimeError(f"MuJoCo >= 3.x is required, found {mujoco.__version__}")


def _fmt(value: float) -> str:
    return f"{float(value):.12g}"


def _compose_mjcf(
    params: RopeParams,
    *,
    cable_count: int,
    equality_body_names: list[str] | None,
) -> str:
    bend = BEND_BASE * float(params.bend_stiffness)
    twist = TWIST_BASE * float(params.twist_stiffness)
    length = float(params.length_m)
    radius = float(params.radius)
    friction = float(params.friction)
    floor_z = 0.0

    contact_xml = ""
    equality_xml = ""
    if equality_body_names:
        excludes = [
            f'    <exclude name="exclude_adj_{i}" body1="{escape(body1)}" body2="{escape(body2)}"/>'
            for i, (body1, body2) in enumerate(zip(equality_body_names[:-1], equality_body_names[1:]))
        ]
        welds = [
            (
                f'    <weld name="grasp_{i}" body1="{escape(body_name)}" '
                f'body2="grip_mocap" active="false" torquescale="0" relpose="0 0 0 1 0 0 0"/>'
            )
            for i, body_name in enumerate(equality_body_names)
        ]
        if excludes:
            contact_xml = "  <contact>\n" + "\n".join(excludes) + "\n  </contact>\n"
        equality_xml = "  <equality>\n" + "\n".join(welds) + "\n  </equality>\n"

    return f"""<mujoco model="dgcc_mujoco_cable">
  <compiler angle="radian"/>
  <extension>
    <plugin plugin="mujoco.elasticity.cable"/>
  </extension>
  <size memory="4M"/>
  <option integrator="implicitfast" timestep="0.001" iterations="100" ls_iterations="50" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="640" offheight="480"/>
    <quality shadowsize="1024"/>
  </visual>
  <worldbody>
    <light name="key" pos="0 -1 1.5" dir="0 1 -1" diffuse="0.8 0.8 0.8"/>
    <camera name="smoke_cam" pos="{_fmt(length * 0.5)} -1.2 0.55" xyaxes="1 0 0 0 0.42 0.91"/>
    <geom name="floor" type="plane" pos="0 0 {_fmt(floor_z)}" size="2 2 0.01" condim="3" friction="{_fmt(friction)}" rgba="0.85 0.85 0.85 1"/>
    <composite type="cable" curve="s" count="{cable_count} 1 1" size="{_fmt(length)}" initial="free" prefix="rope" offset="0 0 {_fmt(radius)}">
      <plugin plugin="mujoco.elasticity.cable">
        <config key="twist" value="{_fmt(twist)}"/>
        <config key="bend" value="{_fmt(bend)}"/>
      </plugin>
      <joint kind="main" damping="0.015" armature="0.01"/>
      <geom type="capsule" size="{_fmt(radius)}" condim="3" friction="{_fmt(friction)}" rgba="0.1 0.35 0.9 1"/>
    </composite>
    <body name="grip_mocap" mocap="true">
      <geom type="sphere" size="{_fmt(radius * 1.5)}" contype="0" conaffinity="0" rgba="1 0.15 0.1 1"/>
    </body>
  </worldbody>
{contact_xml}{equality_xml}</mujoco>
"""


def _enumerate_cable_body_names(model: mujoco.MjModel) -> list[str]:
    """Enumerate generated cable bodies through MuJoCo's name APIs."""

    names: list[str] = []
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not name or not name.startswith("ropeB"):
            continue
        resolved_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if resolved_id == body_id:
            names.append(name)
    return names


def build_mjcf(params: RopeParams) -> str:
    """Build a MuJoCo MJCF string for a cable rope with per-node welds.

    ``RopeParams.bend_stiffness`` and ``twist_stiffness`` are relative
    multipliers on the MuJoCo cable plugin base values from P0 §6.2.
    """

    _require_mujoco_3()
    if params.length_m <= 0:
        raise ValueError("length_m must be positive")
    if params.n_segments < 2:
        raise ValueError("n_segments must be at least 2")
    if params.radius <= 0:
        raise ValueError("radius must be positive")
    if params.friction < 0:
        raise ValueError("friction must be non-negative")

    # MuJoCo 3.10's cable composite creates count-1 generated rope bodies. The
    # adapter contract exposes n_segments body samples, so the XML uses one
    # extra composite point and still enumerates the generated bodies by API.
    cable_count = int(params.n_segments) + 1
    base_xml = _compose_mjcf(params, cable_count=cable_count, equality_body_names=None)
    base_model = mujoco.MjModel.from_xml_string(base_xml)
    body_names = _enumerate_cable_body_names(base_model)
    if len(body_names) != params.n_segments:
        raise RuntimeError(
            "MuJoCo cable body enumeration mismatch: "
            f"expected {params.n_segments}, got {len(body_names)}"
        )
    return _compose_mjcf(params, cable_count=cable_count, equality_body_names=body_names)


class MuJoCoCableEnv(DLOEnvBase):
    """Single-process CPU MuJoCo cable adapter for P0-M1."""

    def __init__(self) -> None:
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.params: RopeParams | None = None
        self.body_ids: list[int] = []
        self.body_names: list[str] = []
        self.weld_eq_ids: list[int] = []
        self.grip_body_id: int | None = None
        self.mocap_id: int | None = None
        self.active_node: int | None = None
        self.active_weld_id: int | None = None
        self.last_settle_steps = 0
        self.last_settle_converged = False
        self.last_delta_clamped = np.zeros(3, dtype=float)
        self.last_move_target = np.zeros(3, dtype=float)

    def reset(self, params: RopeParams, init_shape: str, seed: int) -> dict[str, Any]:
        seed_everything(seed)
        xml = build_mjcf(params)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.params = params

        mujoco.mj_resetData(self.model, self.data)
        self.body_names = _enumerate_cable_body_names(self.model)
        self.body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self.body_names
        ]
        self.weld_eq_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_{i}")
            for i in range(len(self.body_ids))
        ]
        self.grip_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "grip_mocap")
        self.mocap_id = int(self.model.body_mocapid[self.grip_body_id])
        if self.mocap_id < 0:
            raise RuntimeError("grip_mocap body is not a MuJoCo mocap body")

        if len(self.body_ids) != params.n_segments:
            raise RuntimeError(f"expected {params.n_segments} rope bodies, got {len(self.body_ids)}")
        if any(eq_id < 0 for eq_id in self.weld_eq_ids):
            raise RuntimeError("failed to enumerate per-node grasp weld equalities")

        self.data.eq_active[:] = 0
        if init_shape == "straight":
            pass
        elif init_shape == "u_bend":
            self._apply_u_bend()
        else:
            raise ValueError(f"unsupported init_shape {init_shape!r}; expected 'straight' or 'u_bend'")

        self.active_node = None
        self.active_weld_id = None
        self.last_settle_steps = 0
        self.last_settle_converged = False
        mujoco.mj_forward(self.model, self.data)
        self._assert_finite()
        return {
            "sim": "mujoco",
            "mujoco_version": mujoco.__version__,
            "seed": int(seed),
            "init_shape": init_shape,
            "rope_params": asdict(params),
            "n_rope_bodies": len(self.body_ids),
            "body_names": list(self.body_names),
            "mjcf_note": "MuJoCo 3.10 cable composite uses count=n_segments+1 to expose n_segments body samples.",
        }

    def _apply_u_bend(self) -> None:
        assert self.model is not None and self.data is not None
        ball_joint_ids = [
            joint_id
            for joint_id in range(self.model.njnt)
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_BALL
        ]
        if not ball_joint_ids:
            return
        theta = np.pi / len(ball_joint_ids)
        quat = np.array([np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)], dtype=float)
        for joint_id in ball_joint_ids:
            qadr = int(self.model.jnt_qposadr[joint_id])
            self.data.qpos[qadr : qadr + 4] = quat
        self.data.qvel[:] = 0.0
        mujoco.mj_normalizeQuat(self.model, self.data.qpos)
        mujoco.mj_forward(self.model, self.data)

    def get_centerline_raw(self) -> np.ndarray:
        self._require_reset()
        assert self.data is not None
        return np.asarray(self.data.xpos[self.body_ids], dtype=float).copy()

    def get_centerline(self) -> np.ndarray:
        return resample(self.get_centerline_raw())

    def grasp(self, p: int) -> bool:
        self._require_reset()
        assert self.model is not None and self.data is not None and self.mocap_id is not None
        node = int(p)
        if node < 0 or node >= len(self.body_ids):
            raise IndexError(f"grasp node {node} outside [0, {len(self.body_ids)})")

        body_id = self.body_ids[node]
        weld_id = self.weld_eq_ids[node]
        self.data.eq_active[:] = 0
        self.data.mocap_pos[self.mocap_id] = self.data.xpos[body_id]
        self.data.mocap_quat[self.mocap_id] = self.data.xquat[body_id]
        mujoco.mj_forward(self.model, self.data)
        self.data.eq_active[weld_id] = 1
        mujoco.mj_forward(self.model, self.data)
        self.active_node = node
        self.active_weld_id = weld_id
        self._assert_finite()
        return bool(self.data.eq_active[weld_id])

    def move(self, delta: np.ndarray, lift: str) -> np.ndarray:
        self._require_reset()
        assert self.model is not None and self.data is not None and self.mocap_id is not None
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

        start = self.data.mocap_pos[self.mocap_id].copy()
        lifted = start.copy()
        lifted[2] = LIFT_HEIGHTS[lift]
        target = lifted + delta_vec
        self.last_move_target = target.copy()

        waypoints = (lifted, target)
        current = start
        for waypoint in waypoints:
            distance = float(np.linalg.norm(waypoint - current))
            n_steps = max(20, int(ceil(distance / 0.002)))
            for alpha in np.linspace(1.0 / n_steps, 1.0, n_steps):
                self.data.mocap_pos[self.mocap_id] = (1.0 - alpha) * current + alpha * waypoint
                mujoco.mj_step(self.model, self.data)
                self._assert_finite()
            current = waypoint.copy()
        for _ in range(50):
            self.data.mocap_pos[self.mocap_id] = target
            mujoco.mj_step(self.model, self.data)
            self._assert_finite()
        return target.copy()

    def release(self, vel_threshold: float = 1e-3, max_steps: int = 5000) -> bool:
        self._require_reset()
        assert self.model is not None and self.data is not None
        self.data.eq_active[:] = 0
        self.active_node = None
        self.active_weld_id = None
        mujoco.mj_forward(self.model, self.data)
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
                "mocap_target": target,
                "settle_converged": bool(settle_converged),
                "max_abs_qvel": self.max_abs_qvel(),
            },
        }

    def settle(self, vel_threshold: float = 1e-3, max_steps: int = 5000) -> bool:
        self._require_reset()
        assert self.model is not None and self.data is not None
        threshold = float(vel_threshold)
        if threshold < 0:
            raise ValueError("vel_threshold must be non-negative")
        self.last_settle_steps = 0
        for step in range(int(max_steps) + 1):
            qvel_max = self.max_abs_qvel()
            if qvel_max < threshold:
                self.last_settle_steps = step
                self.last_settle_converged = True
                return True
            if step == max_steps:
                break
            mujoco.mj_step(self.model, self.data)
            self._assert_finite()
        self.last_settle_steps = int(max_steps)
        self.last_settle_converged = False
        return False

    def max_abs_qvel(self) -> float:
        self._require_reset()
        assert self.data is not None
        if self.data.qvel.size == 0:
            return 0.0
        return float(np.max(np.abs(self.data.qvel)))

    def _assert_finite(self) -> None:
        assert self.data is not None
        if not np.all(np.isfinite(self.data.qpos)):
            raise FloatingPointError("MuJoCo qpos contains non-finite values")
        if not np.all(np.isfinite(self.data.qvel)):
            raise FloatingPointError("MuJoCo qvel contains non-finite values")

    def _require_reset(self) -> None:
        if self.model is None or self.data is None:
            raise RuntimeError("MuJoCoCableEnv.reset must be called first")
