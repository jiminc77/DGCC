"""DLO-Lab smoke-test CLI for P0-M1."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dgcc.envs.base import RopeParams
from dgcc.envs.dlolab import DLOLabEnv, ensure_genesis_initialized, stiffness_bases
from dgcc.utils.meta import build_run_metadata

os.environ.pop("DISPLAY", None)
warnings.filterwarnings("ignore", message="cannot create weak reference.*")


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
    parser = argparse.ArgumentParser(description="DLO-Lab smoke test")
    parser.add_argument("--seed", type=int, default=0, help="deterministic random seed")
    parser.add_argument("--config", default="configs/smoke_dlolab.yaml", help="YAML config path")
    return parser


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping, got {type(config).__name__}")
    return config


def params_from_config(config: dict[str, Any], *, bend: float | None = None, friction: float | None = None) -> RopeParams:
    rope = config.get("rope", {})
    return RopeParams(
        length_m=float(rope.get("length_m", 1.0)),
        n_segments=int(rope.get("n_segments", 50)),
        bend_stiffness=float(rope.get("bend_stiffness", 1.0) if bend is None else bend),
        twist_stiffness=float(rope.get("twist_stiffness", 1.0)),
        friction=float(rope.get("friction", 1.0) if friction is None else friction),
        radius=float(rope.get("radius", 0.005)),
    )


def env_kwargs(config: dict[str, Any], *, n_envs: int) -> dict[str, Any]:
    sim = config.get("sim", {})
    return {
        "n_envs": n_envs,
        "dt": float(sim.get("dt", 1.0e-3)),
        "substeps": int(sim.get("substeps", 5)),
        "rod_damping": float(sim.get("rod_damping", 10.0)),
        "rod_angular_damping": float(sim.get("rod_angular_damping", 5.0)),
        "initial_settle_steps": int(sim.get("initial_settle_steps", 20)),
        "reset_settle_max_steps": int(sim.get("reset_settle_max_steps", 1000)),
        "move_step_size": float(sim.get("move_step_size", 0.002)),
        "move_hold_steps": int(sim.get("move_hold_steps", 20)),
    }


def check(condition: bool, label: str, detail: str = "") -> None:
    if condition:
        print(f"PASS {label}" + (f" — {detail}" if detail else ""))
        return
    print(f"FAIL {label}" + (f" — {detail}" if detail else ""))
    raise SmokeFailure(label)


def metric_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def run_primitive(config: dict[str, Any], seed: int, *, bend: float, friction: float) -> tuple[np.ndarray, dict[str, Any]]:
    params = params_from_config(config, bend=bend, friction=friction)
    primitive_cfg = config.get("primitive", {})
    p = int(primitive_cfg.get("p", params.n_segments // 2))
    delta = np.array(primitive_cfg.get("delta", [0.08, -0.04, 0.0]), dtype=float)
    lift = str(primitive_cfg.get("lift", "high"))
    env = DLOLabEnv(**env_kwargs(config, n_envs=1))
    env.reset(params, init_shape=str(config.get("init_shape", "bent")), seed=seed)
    result = env.step_primitive(p, delta, lift)
    return np.asarray(result["X_after"], dtype=float), result


def dependency_versions() -> dict[str, Any]:
    import genesis as gs
    import quadrants
    import torch
    import numpy
    import scipy

    return {
        "genesis": getattr(gs, "__version__", "unknown"),
        "quadrants": getattr(quadrants, "__version__", "unknown"),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "backend": str(getattr(gs, "backend", "unknown")),
        "display": os.environ.get("DISPLAY"),
    }


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    seed = int(args.seed)

    output_cfg = config.get("outputs", {})
    log_path = Path(output_cfg.get("stdout_log", "outputs/reports/smoke_dlolab_stdout.log"))
    meta_path = Path(output_cfg.get("meta_json", "outputs/metrics/smoke_dlolab_meta.json"))
    for path in (log_path, meta_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    primitive_wall_time_s: float | None = None
    stiffness_shape_diff_l2: float | None = None
    track_distance: float | None = None
    settle_converged: bool | None = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            print(f"smoke_dlolab seed={seed} config={config_path}")
            gs = ensure_genesis_initialized(seed)
            check(
                getattr(gs, "_initialized", False) and str(getattr(gs, "backend", "")).lower() != "cpu",
                "1 gs.init(backend=gs.gpu) headless succeeds",
                f"backend={getattr(gs, 'backend', 'unknown')} DISPLAY={os.environ.get('DISPLAY')}",
            )

            params = params_from_config(config)
            batch_env = DLOLabEnv(**env_kwargs(config, n_envs=4))
            batch_info = batch_env.reset(params, init_shape="straight", seed=seed)
            check(
                batch_info["n_envs"] == 4 and batch_env.scene is not None and batch_env.rod_entity is not None,
                "2 rope scene builds with n_envs=4",
                f"n_vertices={batch_info['n_vertices']} bases={stiffness_bases()}",
            )
            sampled = np.asarray(batch_env.rod_entity.sample_centerline(32), dtype=float)
            check(
                sampled.shape == (4, 32, 3),
                "3 sample_centerline(32) shape == (4,32,3)",
                f"shape={sampled.shape}",
            )

            X1, result1 = run_primitive(config, seed, bend=1.0, friction=1.0)
            X2, _ = run_primitive(config, seed, bend=2.0, friction=2.0)
            stiffness_shape_diff_l2 = metric_l2(X1, X2)
            check(
                np.isfinite(stiffness_shape_diff_l2) and stiffness_shape_diff_l2 > 0.0,
                "4 set_bending_stiffness/set_mu_s change dynamics",
                f"final_shape_l2_diff={stiffness_shape_diff_l2:.9g}",
            )

            thresholds = config.get("thresholds", {})
            track_tolerance = float(thresholds.get("track_distance_m", 0.01))
            settle_threshold = float(thresholds.get("settle_vel", 1e-3))
            settle_max_steps = int(thresholds.get("settle_max_steps", 5000))
            nan_steps = int(thresholds.get("nan_steps", 2000))
            primitive_cfg = config.get("primitive", {})
            p = int(primitive_cfg.get("p", params.n_segments // 2))
            delta = np.array(primitive_cfg.get("delta", [0.08, -0.04, 0.0]), dtype=float)
            lift = str(primitive_cfg.get("lift", "high"))

            track_env = DLOLabEnv(**env_kwargs(config, n_envs=1))
            track_env.reset(params, init_shape="straight", seed=seed)
            track_env.grasp(p)
            target = track_env.move(delta, lift)
            vertex = np.asarray(track_env.get_centerline_raw(), dtype=float)[p]
            link_pos = track_env._gripper_positions()[0]
            track_distance = float(np.linalg.norm(vertex - link_pos))
            check(
                track_distance < track_tolerance,
                "5 attach→move→grasped vertex tracks the link",
                f"distance={track_distance:.9g} tolerance={track_tolerance:.9g} target={np.asarray(target).tolist()}",
            )

            settle_converged = track_env.release(vel_threshold=settle_threshold, max_steps=settle_max_steps)
            max_speed = track_env.max_node_speed()
            check(
                settle_converged and max_speed < settle_threshold,
                "6 detach→settle converges",
                f"max_node_speed={max_speed:.9g} threshold={settle_threshold:.9g} steps={track_env.last_settle_steps}",
            )

            no_nan = True
            for _ in range(nan_steps):
                track_env._step_scene()
                raw = np.asarray(track_env.get_centerline_raw(), dtype=float)
                max_speed = track_env.max_node_speed()
                if not (np.all(np.isfinite(raw)) and np.isfinite(max_speed)):
                    no_nan = False
                    break
            check(no_nan, "7 2000 steps no NaN", f"steps={nan_steps}")

            primitive_env = DLOLabEnv(**env_kwargs(config, n_envs=1))
            primitive_env.reset(params, init_shape=str(config.get("init_shape", "bent")), seed=seed)
            start = time.perf_counter()
            primitive_result = primitive_env.step_primitive(p, delta, lift)
            primitive_wall_time_s = time.perf_counter() - start
            shapes_ok = primitive_result["X_before"].shape == (32, 3) and primitive_result["X_after"].shape == (32, 3)
            check(
                shapes_ok and bool(primitive_result["grasp_success"]) and primitive_wall_time_s >= 0.0,
                "8 wall-time for one primitive",
                f"primitive_wall_time_s={primitive_wall_time_s:.6f} settle_steps={primitive_result['settle_steps']}",
            )

            metadata = build_run_metadata(config=config, seed=seed)
            metadata.update(
                {
                    "dependency_versions": dependency_versions(),
                    "stiffness_bases": stiffness_bases(),
                    "batch_reset_info": batch_info,
                    "primitive_wall_time_s": primitive_wall_time_s,
                    "stiffness_shape_diff_l2": stiffness_shape_diff_l2,
                    "track_distance_m": track_distance,
                    "settle_converged": settle_converged,
                    "reference_primitive_settle_steps": int(result1["settle_steps"]),
                    "smoke_passed": True,
                }
            )
            meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8")
            print(f"meta_json={meta_path}")
            print("SMOKE_DLOLAB PASS")
            return 0
        except Exception as exc:
            metadata = build_run_metadata(config=config, seed=seed)
            try:
                versions = dependency_versions()
            except Exception as version_exc:  # pragma: no cover - defensive failure metadata.
                versions = {"error": repr(version_exc)}
            metadata.update(
                {
                    "dependency_versions": versions,
                    "stiffness_bases": stiffness_bases(),
                    "primitive_wall_time_s": primitive_wall_time_s,
                    "stiffness_shape_diff_l2": stiffness_shape_diff_l2,
                    "track_distance_m": track_distance,
                    "settle_converged": settle_converged,
                    "smoke_passed": False,
                    "error": repr(exc),
                }
            )
            meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8")
            print(f"SMOKE_DLOLAB FAIL: {exc!r}")
            return 1
        finally:
            try:
                import genesis as gs

                if getattr(gs, "_initialized", False):
                    gs.destroy()
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr


if __name__ == "__main__":
    raise SystemExit(main())
