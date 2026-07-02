"""Sanctioned by @M3 Exit (demo script); lives inside the §4 scripts/ directory."""

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

import numpy as np
import yaml

from dgcc.envs.base import RopeParams
from dgcc.envs.dlolab import DLOLabEnv
from dgcc.utils.meta import build_run_metadata

os.environ.pop("DISPLAY", None)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the P0-M3 DLO-Lab primitive demo")
    parser.add_argument("--seed", type=int, default=0, help="deterministic demo seed")
    parser.add_argument("--config", default="configs/demo_primitive.yaml", help="YAML config path")
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
        n_segments=int(rope.get("n_segments", 32)),
        bend_stiffness=float(rope.get("bend_stiffness", 1.0)),
        twist_stiffness=float(rope.get("twist_stiffness", 1.0)),
        friction=float(rope.get("friction", 1.0)),
        radius=float(rope.get("radius", 0.005)),
    )


def env_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    sim = config.get("sim", {})
    return {
        "n_envs": 1,
        "dt": float(sim.get("dt", 1.0e-3)),
        "substeps": int(sim.get("substeps", 5)),
        "rod_damping": float(sim.get("rod_damping", 10.0)),
        "rod_angular_damping": float(sim.get("rod_angular_damping", 5.0)),
        "initial_settle_steps": int(sim.get("initial_settle_steps", 0)),
        "reset_settle_max_steps": int(sim.get("reset_settle_max_steps", 25)),
        "move_step_size": float(sim.get("move_step_size", 0.03)),
        "move_hold_steps": int(sim.get("move_hold_steps", 0)),
        "grasp_realism": bool(sim.get("grasp_realism", True)),
    }


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_rgb_png(path: Path, image: np.ndarray) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be uint8 RGB")
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += _png_chunk(b"IDAT", zlib.compress(raw, level=9))
    payload += _png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _draw_disk(image: np.ndarray, x: int, y: int, color: tuple[int, int, int], radius: int = 2) -> None:
    height, width, _ = image.shape
    for yy in range(max(0, y - radius), min(height, y + radius + 1)):
        for xx in range(max(0, x - radius), min(width, x + radius + 1)):
            if (xx - x) ** 2 + (yy - y) ** 2 <= radius**2:
                image[yy, xx] = color


def _draw_line(image: np.ndarray, a: np.ndarray, b: np.ndarray, color: tuple[int, int, int]) -> None:
    x0, y0 = a.astype(int)
    x1, y1 = b.astype(int)
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for alpha in np.linspace(0.0, 1.0, steps + 1):
        x = int(round((1.0 - alpha) * x0 + alpha * x1))
        y = int(round((1.0 - alpha) * y0 + alpha * y1))
        _draw_disk(image, x, y, color, radius=1)


def save_topdown_trajectory(trajectories: list[np.ndarray], path: Path) -> None:
    width, height, margin = 1000, 800, 50
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    all_xy = np.concatenate([np.asarray(x, dtype=float)[:, :2] for x in trajectories], axis=0)
    xy_min = all_xy.min(axis=0)
    xy_max = all_xy.max(axis=0)
    span = np.maximum(xy_max - xy_min, 1e-6)
    scale = min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])
    center = 0.5 * (xy_min + xy_max)

    def to_pixels(xy: np.ndarray) -> np.ndarray:
        shifted = (np.asarray(xy, dtype=float) - center) * scale
        px = shifted[:, 0] + width / 2.0
        py = height / 2.0 - shifted[:, 1]
        return np.column_stack((px, py))

    for idx, trajectory in enumerate(trajectories):
        alpha = idx / max(1, len(trajectories) - 1)
        color = (int(30 + 190 * alpha), int(80 + 80 * (1.0 - alpha)), int(220 - 150 * alpha))
        pixels = to_pixels(np.asarray(trajectory)[:, :2])
        for a, b in zip(pixels[:-1], pixels[1:]):
            _draw_line(image, a, b, color)
        _draw_disk(image, int(round(pixels[0, 0])), int(round(pixels[0, 1])), (0, 160, 0), radius=3)
        _draw_disk(image, int(round(pixels[-1, 0])), int(round(pixels[-1, 1])), (180, 0, 0), radius=3)

    write_rgb_png(path, image)


def random_delta(rng: np.random.Generator, max_delta_m: float) -> np.ndarray:
    direction = rng.normal(0.0, 1.0, size=2)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        direction = np.array([1.0, 0.0])
    else:
        direction = direction / norm
    magnitude = float(rng.uniform(0.02, max_delta_m))
    return np.array([direction[0] * magnitude, direction[1] * magnitude, 0.0], dtype=float)


def run_demo(config: dict[str, Any], seed: int, config_path: Path) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    params = params_from_config(config)
    primitive_cfg = config.get("primitive", {})
    count = int(primitive_cfg.get("count", 10))
    p_low = int(primitive_cfg.get("p_low", 2))
    p_high_margin = int(primitive_cfg.get("p_high_margin", 2))
    max_delta_m = float(primitive_cfg.get("max_delta_m", 0.08))
    lift_choices = list(primitive_cfg.get("lift_choices", ["low", "high"]))

    env = DLOLabEnv(**env_kwargs(config))
    reset_info = env.reset(params, init_shape="random_smooth", seed=seed)
    trajectories = [env.get_centerline()]
    records: list[dict[str, Any]] = []

    print(f"demo_primitive seed={seed} config={config_path}")
    print(f"reset init_shape=random_smooth n_vertices={reset_info['n_vertices']} length_m={reset_info['length_m']}")

    for idx in range(count):
        p_high = max(p_low + 1, int(params.n_segments) - p_high_margin)
        p = int(rng.integers(p_low, p_high))
        delta = random_delta(rng, max_delta_m)
        lift = str(rng.choice(lift_choices))
        start = time.perf_counter()
        result = env.step_primitive(p, delta, lift)
        wall_time_s = time.perf_counter() - start
        trajectories.append(np.asarray(result["X_after"], dtype=float))

        info = result["info"]
        record = {
            "index": idx,
            "p": p,
            "p_actual": int(info["p_actual"]),
            "grasp_success": bool(result["grasp_success"]),
            "settle_steps": int(result["settle_steps"]),
            "wall_time_s": wall_time_s,
            "lift": lift,
            "delta": delta.tolist(),
        }
        records.append(record)
        print(
            "primitive "
            f"{idx:02d} p={record['p']} p_actual={record['p_actual']} "
            f"grasp_success={record['grasp_success']} settle_steps={record['settle_steps']} "
            f"wall_time_s={record['wall_time_s']:.6f} lift={lift} delta={delta.tolist()}"
        )

    output_cfg = config.get("outputs", {})
    plot_path = Path(output_cfg.get("trajectory_png", "outputs/plots/demo_primitive_trajectory.png"))
    meta_path = Path(output_cfg.get("meta_json", "outputs/metrics/demo_primitive_meta.json"))
    save_topdown_trajectory(trajectories, plot_path)

    metadata = build_run_metadata(config=config, seed=seed)
    metadata.update(
        {
            "config_path": str(config_path),
            "reset_info": reset_info,
            "primitive_records": records,
            "trajectory_plot": str(plot_path),
            "stdout_log": str(output_cfg.get("stdout_log", "outputs/reports/demo_primitive_stdout.log")),
        }
    )
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8")

    print(f"trajectory_png={plot_path}")
    print(f"meta_json={meta_path}")
    print("DEMO_PRIMITIVE PASS")
    return metadata


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    output_cfg = config.get("outputs", {})
    log_path = Path(output_cfg.get("stdout_log", "outputs/reports/demo_primitive_stdout.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            run_demo(config, int(args.seed), config_path)
            return 0
        except Exception as exc:
            print(f"DEMO_PRIMITIVE FAIL {type(exc).__name__}: {exc}")
            return 1
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    raise SystemExit(main())
