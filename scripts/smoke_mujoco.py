"""MuJoCo cable smoke-test CLI for P0-M1."""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import yaml

from dgcc.envs.base import RopeParams
from dgcc.envs.mujoco_cable import MuJoCoCableEnv, build_mjcf
from dgcc.utils.meta import build_run_metadata


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


class SmokeFailure(AssertionError):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MuJoCo cable smoke test")
    parser.add_argument("--seed", type=int, default=0, help="deterministic random seed")
    parser.add_argument("--config", default="configs/smoke_mujoco.yaml", help="YAML config path")
    return parser


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping, got {type(config).__name__}")
    return config


def params_from_config(config: dict[str, Any]) -> RopeParams:
    rope = config.get("rope", {})
    return RopeParams(
        length_m=float(rope.get("length_m", 1.0)),
        n_segments=int(rope.get("n_segments", 50)),
        bend_stiffness=float(rope.get("bend_stiffness", 1.0)),
        twist_stiffness=float(rope.get("twist_stiffness", 1.0)),
        friction=float(rope.get("friction", 0.3)),
        radius=float(rope.get("radius", 0.005)),
    )


def png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    )


def write_png(path: Path, rgb: np.ndarray) -> None:
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected RGB image with shape (H, W, 3), got {image.shape}")
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    data = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(raw, level=6))
        + png_chunk(b"IEND", b"")
    )
    path.write_bytes(data)


def check(condition: bool, label: str, detail: str = "") -> None:
    if condition:
        print(f"PASS {label}" + (f" — {detail}" if detail else ""))
        return
    print(f"FAIL {label}" + (f" — {detail}" if detail else ""))
    raise SmokeFailure(label)


def render_frame(env: MuJoCoCableEnv, path: Path, width: int, height: int) -> np.ndarray:
    assert env.model is not None and env.data is not None
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    try:
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        params = env.params
        length = params.length_m if params is not None else 1.0
        camera.lookat[:] = np.array([length * 0.5, 0.0, 0.02], dtype=float)
        camera.distance = max(0.8, length * 1.4)
        camera.azimuth = 90.0
        camera.elevation = -35.0
        renderer.update_scene(env.data, camera=camera)
        frame = renderer.render()
    finally:
        renderer.close()
    write_png(path, frame)
    return frame


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    seed = int(args.seed)

    output_cfg = config.get("outputs", {})
    log_path = Path(output_cfg.get("stdout_log", "outputs/reports/smoke_mujoco_stdout.log"))
    frame_path = Path(output_cfg.get("frame_png", "outputs/plots/smoke_mujoco_frame.png"))
    meta_path = Path(output_cfg.get("meta_json", "outputs/metrics/smoke_mujoco_meta.json"))
    for path in (log_path, frame_path, meta_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    primitive_wall_time_s: float | None = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            print(f"smoke_mujoco seed={seed} config={config_path}")
            print(f"mujoco_version={mujoco.__version__} MUJOCO_GL={os.environ.get('MUJOCO_GL')}")
            params = params_from_config(config)
            init_shape = str(config.get("init_shape", "straight"))
            render_cfg = config.get("render", {})
            width = int(render_cfg.get("width", 640))
            height = int(render_cfg.get("height", 480))
            thresholds = config.get("thresholds", {})
            track_tolerance = float(thresholds.get("track_distance_m", 0.03))
            settle_threshold = float(thresholds.get("settle_qvel", 1e-3))
            settle_max_steps = int(thresholds.get("settle_max_steps", 5000))
            nan_steps = int(thresholds.get("nan_steps", 2000))
            primitive_cfg = config.get("primitive", {})
            p = int(primitive_cfg.get("p", params.n_segments // 2))
            delta = np.array(primitive_cfg.get("delta", [0.05, 0.0, 0.0]), dtype=float)
            lift = str(primitive_cfg.get("lift", "low"))

            xml = build_mjcf(params)
            compiled = mujoco.MjModel.from_xml_string(xml)
            check(
                compiled.nbody > params.n_segments and compiled.neq >= params.n_segments,
                "1 model compiles from generated MJCF",
                f"nbody={compiled.nbody} neq={compiled.neq} nv={compiled.nv}",
            )

            env = MuJoCoCableEnv()
            reset_info = env.reset(params, init_shape=init_shape, seed=seed)
            frame = render_frame(env, frame_path, width=width, height=height)
            frame_size = frame_path.stat().st_size
            nonuniform = bool(np.std(frame.astype(float)) > 0.0)
            check(
                frame_path.exists() and frame_size > 1024 and nonuniform,
                "2 MUJOCO_GL=egl headless render saved",
                f"path={frame_path} bytes={frame_size}",
            )

            raw = env.get_centerline_raw()
            centerline = env.get_centerline()
            check(
                raw.shape == (params.n_segments, 3) and centerline.shape == (32, 3),
                "3 centerline raw/resample shapes",
                f"raw={raw.shape} resampled={centerline.shape}",
            )

            grasp_success = env.grasp(p)
            target = env.move(delta, lift)
            assert env.data is not None and env.mocap_id is not None
            body_id = env.body_ids[p]
            track_distance = float(np.linalg.norm(env.data.xpos[body_id] - env.data.mocap_pos[env.mocap_id]))
            check(
                grasp_success and track_distance < track_tolerance,
                "4 grasped node tracks mocap after move",
                f"distance={track_distance:.6g} tolerance={track_tolerance:.6g} target={target.tolist()}",
            )

            converged = env.release(vel_threshold=settle_threshold, max_steps=settle_max_steps)
            qvel_max = env.max_abs_qvel()
            check(
                converged and qvel_max < settle_threshold,
                "5 release settle qvel decays below threshold",
                f"max_abs_qvel={qvel_max:.6g} threshold={settle_threshold:.6g} steps={env.last_settle_steps}",
            )

            assert env.model is not None and env.data is not None
            no_nan = True
            for _ in range(nan_steps):
                mujoco.mj_step(env.model, env.data)
                if not (np.all(np.isfinite(env.data.qpos)) and np.all(np.isfinite(env.data.qvel))):
                    no_nan = False
                    break
            check(no_nan, "6 2000 steps produce no NaN in qpos/qvel", f"steps={nan_steps}")

            primitive_env = MuJoCoCableEnv()
            primitive_env.reset(params, init_shape=init_shape, seed=seed)
            start = time.perf_counter()
            result = primitive_env.step_primitive(p, delta, lift)
            primitive_wall_time_s = time.perf_counter() - start
            result_shapes_ok = result["X_before"].shape == (32, 3) and result["X_after"].shape == (32, 3)
            check(
                result_shapes_ok and bool(result["grasp_success"]),
                "7 full step_primitive wall-time measured",
                f"primitive_wall_time_s={primitive_wall_time_s:.6f} settle_steps={result['settle_steps']}",
            )

            metadata = build_run_metadata(config=config, seed=seed)
            metadata.update(
                {
                    "mujoco_version": mujoco.__version__,
                    "reset_info": reset_info,
                    "primitive_wall_time_s": primitive_wall_time_s,
                    "smoke_passed": True,
                }
            )
            meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
            print(f"meta_json={meta_path}")
            print("SMOKE_MUJOCO PASS")
            return 0
        except Exception as exc:
            metadata = build_run_metadata(config=config, seed=seed)
            metadata.update(
                {
                    "mujoco_version": mujoco.__version__,
                    "primitive_wall_time_s": primitive_wall_time_s,
                    "smoke_passed": False,
                    "error": repr(exc),
                }
            )
            meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
            print(f"SMOKE_MUJOCO FAIL: {exc!r}")
            return 1
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    raise SystemExit(main())
