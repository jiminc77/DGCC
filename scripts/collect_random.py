"""P0-M4 batched random-policy transition collector.

DLO-Lab's rod_entity source exposes per-environment grasp hooks named
``attach_to_rigid_link_with_envs_idx`` and ``detach_from_rigid_link_with_envs_idx``
(external/DLO-Lab/genesis/engine/entities/rod_entity.py).  This collector uses
those hooks through :meth:`DLOLabEnv.step_primitive_batch`, so grasp failures are
sampled independently per environment and failed envs are restored to the exact
pre-primitive rope state instead of fabricating copied transitions.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from dgcc.envs.base import RopeParams
from dgcc.envs.dlolab import DLOLabEnv, analytic_init_centerline
from dgcc.logging.writer import TransitionWriter, read_transitions
from dgcc.phi.dct import CHANNEL_LAYOUT, CHANNEL_LAYOUT_ID, M, Phi_DCT
from dgcc.phi.normalize import DmNormalizer
from dgcc.phi.resample import resample
from dgcc.utils.meta import get_git_commit_hash

os.environ.pop("DISPLAY", None)

INIT_SHAPES = ("straight", "u_bend", "s_curve", "random_smooth")


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
    parser = argparse.ArgumentParser(description="Collect P0-M4 random DLO-Lab transitions")
    parser.add_argument("--seed", type=int, default=0, help="deterministic collection seed")
    parser.add_argument("--config", default="configs/collect_random.yaml", help="YAML config path")
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="skip collection; recompute dm stats/normalizer/plots from the existing dataset (CPU only)",
    )
    parser.add_argument(
        "--drift-probe-json",
        default=None,
        help="optional path to a restoration-drift probe JSON to embed in dm_stats (M4 gate evidence)",
    )
    return parser


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    config = yaml.safe_load(text) or {}
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping, got {type(config).__name__}")
    return config, text


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


def env_kwargs(config: dict[str, Any], n_envs: int) -> dict[str, Any]:
    sim = config.get("sim", {})
    return {
        "n_envs": int(n_envs),
        "dt": float(sim.get("dt", 1.0e-3)),
        "substeps": int(sim.get("substeps", 5)),
        "rod_damping": float(sim.get("rod_damping", 10.0)),
        "rod_angular_damping": float(sim.get("rod_angular_damping", 5.0)),
        "initial_settle_steps": int(sim.get("initial_settle_steps", 0)),
        "reset_settle_max_steps": int(sim.get("reset_settle_max_steps", 5000)),
        "move_step_size": float(sim.get("move_step_size", 0.03)),
        "move_hold_steps": int(sim.get("move_hold_steps", 0)),
        "grasp_realism": bool(sim.get("grasp_realism", True)),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def as_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(v) for v in value]
    return value


def random_deltas(rng: np.random.Generator, count: int, r_min: float, r_max: float) -> np.ndarray:
    if r_min < 0.0 or r_max <= 0.0 or r_min > r_max:
        raise ValueError("delta range must satisfy 0 <= min <= max")
    if r_max > 0.15 + 1e-12:
        raise ValueError("delta_max_m must not exceed the M3 primitive 0.15 m clamp")
    angles = rng.uniform(0.0, 2.0 * math.pi, size=count)
    radii = np.sqrt(rng.uniform(r_min * r_min, r_max * r_max, size=count))
    return np.column_stack((radii * np.cos(angles), radii * np.sin(angles), np.zeros(count)))


def sample_actions(
    rng: np.random.Generator,
    *,
    n_envs: int,
    n_vertices: int,
    collection_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    p_low = int(collection_cfg.get("p_low", 1))
    p_high_margin = int(collection_cfg.get("p_high_margin", 1))
    p_high = int(n_vertices) - p_high_margin
    if p_low >= p_high:
        raise ValueError(f"invalid p range [{p_low}, {p_high}) for {n_vertices} vertices")
    p = rng.integers(p_low, p_high, size=n_envs, endpoint=False)
    deltas = random_deltas(
        rng,
        n_envs,
        float(collection_cfg.get("delta_min_m", 0.02)),
        float(collection_cfg.get("delta_max_m", 0.15)),
    )
    lift_choices = [str(value) for value in collection_cfg.get("lift_choices", ["low", "high"])]
    lifts = [str(value) for value in rng.choice(lift_choices, size=n_envs)]
    return p.astype(int), deltas.astype(float), lifts


def build_initial_vertices(
    params: RopeParams,
    *,
    n_envs: int,
    episode_index: int,
    seed: int,
    init_shapes: Sequence[str],
) -> tuple[np.ndarray, list[str], list[int]]:
    vertices = []
    shapes = []
    seeds = []
    for env_idx in range(n_envs):
        shape = str(init_shapes[(episode_index * n_envs + env_idx) % len(init_shapes)])
        curve_seed = int(seed + 100_000 * (episode_index + 1) + env_idx)
        vertices.append(analytic_init_centerline(params, shape, curve_seed))
        shapes.append(shape)
        seeds.append(curve_seed)
    return np.stack(vertices), shapes, seeds


def run_probe_candidate(
    *,
    config: dict[str, Any],
    params: RopeParams,
    seed: int,
    n_envs: int,
    rounds: int,
) -> dict[str, Any]:
    collection_cfg = config.get("collection", {})
    init_shapes = [str(value) for value in collection_cfg.get("init_shapes", INIT_SHAPES)]
    vel_threshold = float(collection_cfg.get("vel_threshold", 1.0e-3))
    settle_max_steps = int(collection_cfg.get("settle_max_steps", 5000))
    rng = np.random.default_rng(seed + 17 * n_envs)

    start_build = time.perf_counter()
    env = DLOLabEnv(**env_kwargs(config, n_envs))
    reset_info = env.reset(params, init_shape="straight", seed=seed + n_envs)
    build_wall_s = time.perf_counter() - start_build
    if not env.supports_per_env_grasp():
        raise RuntimeError("per-env grasp hooks unavailable")

    reset_vertices, reset_shapes, _ = build_initial_vertices(
        params,
        n_envs=n_envs,
        episode_index=0,
        seed=seed + 900_000,
        init_shapes=init_shapes,
    )
    reset_start = time.perf_counter()
    reset_result = env.light_reset(
        reset_vertices,
        vel_threshold=vel_threshold,
        max_steps=settle_max_steps,
    )
    light_reset_wall_s = time.perf_counter() - reset_start

    round_times = []
    success_count = 0
    converged_count = 0
    for round_idx in range(int(rounds)):
        p, deltas, lifts = sample_actions(
            rng,
            n_envs=n_envs,
            n_vertices=int(params.n_segments),
            collection_cfg=collection_cfg,
        )
        round_start = time.perf_counter()
        result = env.step_primitive_batch(
            p,
            deltas,
            lifts,
            vel_threshold=vel_threshold,
            max_steps=settle_max_steps,
            rng=rng,
        )
        round_times.append(time.perf_counter() - round_start)
        success = np.asarray(result["grasp_success"], dtype=bool)
        settle_steps = np.asarray(result["settle_steps"], dtype=int)
        success_count += int(success.sum())
        converged_count += int(np.sum(settle_steps != settle_max_steps))

    del env
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    mean_round_s = float(np.mean(round_times)) if round_times else 0.0
    transitions_per_s = float(n_envs / mean_round_s) if mean_round_s > 0.0 else 0.0
    return {
        "n_envs": int(n_envs),
        "rounds": int(rounds),
        "build_wall_s": build_wall_s,
        "light_reset_wall_s": light_reset_wall_s,
        "round_wall_s": round_times,
        "mean_s_per_round": mean_round_s,
        "transitions_per_s": transitions_per_s,
        "reset_settle_steps_max": int(np.max(reset_result["settle_steps"])),
        "reset_converged_rate": float(np.mean(reset_result["settle_converged"])),
        "reset_shapes": reset_shapes,
        "success_rate": success_count / max(1, int(rounds) * n_envs),
        "convergence_rate": converged_count / max(1, int(rounds) * n_envs),
        "reset_info_n_vertices": int(reset_info["n_vertices"]),
    }


def choose_n_envs(config: dict[str, Any], params: RopeParams, seed: int) -> tuple[int, list[dict[str, Any]]]:
    collection_cfg = config.get("collection", {})
    probe_cfg = config.get("probe", {})
    configured = collection_cfg.get("n_envs", "auto")
    if str(configured).lower() != "auto" and configured is not None:
        return int(configured), []

    candidates = [int(value) for value in collection_cfg.get("n_env_candidates", [32])]
    rounds = int(probe_cfg.get("rounds", 2))
    if not bool(probe_cfg.get("enabled", True)):
        return candidates[-1], []

    probe_results: list[dict[str, Any]] = []
    print(f"timing_probe start candidates={candidates} rounds={rounds}")
    for candidate in candidates:
        try:
            result = run_probe_candidate(
                config=config,
                params=params,
                seed=seed,
                n_envs=candidate,
                rounds=rounds,
            )
            probe_results.append(result)
            print(
                "timing_probe "
                f"n_envs={candidate} mean_s_per_round={result['mean_s_per_round']:.6f} "
                f"transitions_per_s={result['transitions_per_s']:.3f} "
                f"success_rate={result['success_rate']:.3f} "
                f"convergence_rate={result['convergence_rate']:.3f}"
            )
        except Exception as exc:
            failure = {"n_envs": int(candidate), "error": f"{type(exc).__name__}: {exc}"}
            probe_results.append(failure)
            print(f"timing_probe n_envs={candidate} failed {failure['error']}")

    valid = [item for item in probe_results if "transitions_per_s" in item and item["transitions_per_s"] > 0.0]
    if not valid:
        raise RuntimeError(f"no timing probe candidate succeeded: {probe_results}")
    chosen = max(valid, key=lambda item: (float(item["transitions_per_s"]), int(item["n_envs"])))
    print(f"timing_probe chosen_n_envs={chosen['n_envs']}")
    return int(chosen["n_envs"]), probe_results


def format_eta(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, sec = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def append_batch_records(
    writer: TransitionWriter,
    *,
    result: dict[str, Any],
    p: np.ndarray,
    deltas: np.ndarray,
    lifts: Sequence[str],
    params: RopeParams,
    round_seed: int,
    commit_hash: str,
) -> int:
    timestamp = utc_now()
    rope_params = asdict(params)
    x_before = np.asarray(result["X_before"], dtype=float)
    x_after = np.asarray(result["X_after"], dtype=float)
    success = np.asarray(result["grasp_success"], dtype=bool)
    settle_steps = np.asarray(result["settle_steps"], dtype=int)
    records = []
    for env_idx in range(x_before.shape[0]):
        records.append(
            {
                "X_before": x_before[env_idx],
                "X_after": x_after[env_idx],
                "p": int(p[env_idx]),
                "delta": np.asarray(deltas[env_idx], dtype=float),
                "lift": str(lifts[env_idx]),
                "grasp_success": bool(success[env_idx]),
                "settle_steps": int(settle_steps[env_idx]),
                "rope_params": rope_params,
                "seed": int(round_seed),
                "sim": "dlolab",
                "timestamp": timestamp,
                "commit_hash": commit_hash,
            }
        )
    writer.append(records)
    return len(records)


def update_h5_meta(path: Path, updates: dict[str, Any]) -> None:
    with h5py.File(path, "r+") as h5:
        meta = json.loads(str(h5.attrs["meta_json"]))
        meta.update(as_jsonable(updates))
        h5.attrs["meta_json"] = json.dumps(meta, ensure_ascii=False, sort_keys=True)
        for key, value in as_jsonable(updates).items():
            if isinstance(value, (str, int, float, bool)):
                h5.attrs[key] = value
        h5.flush()


def collect(config: dict[str, Any], config_text: str, seed: int, config_path: Path) -> dict[str, Any]:
    params = params_from_config(config)
    collection_cfg = config.get("collection", {})
    output_cfg = config.get("outputs", {})
    target_count = int(collection_cfg.get("target_count", 5000))
    max_records = int(collection_cfg.get("max_records", 5300))
    K = int(collection_cfg.get("K", 4))
    if K <= 0:
        raise ValueError("collection.K must be positive")
    vel_threshold = float(collection_cfg.get("vel_threshold", 1.0e-3))
    settle_max_steps = int(collection_cfg.get("settle_max_steps", 5000))
    init_shapes = [str(value) for value in collection_cfg.get("init_shapes", INIT_SHAPES)]
    if not init_shapes:
        raise ValueError("collection.init_shapes must not be empty")

    commit_hash = get_git_commit_hash(".")
    n_envs, probe_results = choose_n_envs(config, params, seed)
    if target_count + n_envs > max_records + n_envs:
        raise ValueError("max_records is inconsistent with target_count")

    dataset_path = Path(output_cfg.get("h5", "outputs/data/p0_random_transitions.h5"))
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if dataset_path.exists():
        dataset_path.unlink()

    meta = {
        "config": config_text,
        "commit_hash": commit_hash,
        "creation_time": utc_now(),
        "collector": "scripts/collect_random.py",
        "config_path": str(config_path),
        "seed": int(seed),
        "target_count": target_count,
        "n_envs": int(n_envs),
        "K": int(K),
        "probe": as_jsonable(probe_results),
        "grasp_mode": "per-env",
        "grasp_mode_evidence": "external/DLO-Lab/genesis/engine/entities/rod_entity.py defines attach_to_rigid_link_with_envs_idx and detach_from_rigid_link_with_envs_idx",
    }

    rng = np.random.default_rng(seed)
    env = DLOLabEnv(**env_kwargs(config, n_envs))
    reset_info = env.reset(params, init_shape="straight", seed=seed)
    if not env.supports_per_env_grasp():
        raise RuntimeError("per-env rod attach/detach hooks unavailable; aborting rather than fabricating failures")

    start_wall = time.perf_counter()
    total_success = 0
    total_converged = 0
    round_times: list[float] = []
    reset_records: list[dict[str, Any]] = []
    round_index = 0
    episode_index = 0
    primitives_in_episode = 0
    recovery_records: list[dict[str, Any]] = []

    print(
        "collect_random start "
        f"seed={seed} config={config_path} n_envs={n_envs} K={K} target_count={target_count} "
        f"commit_hash={commit_hash}"
    )
    print(
        "grasp_mode=per-env evidence="
        "external/DLO-Lab/genesis/engine/entities/rod_entity.py:attach_to_rigid_link_with_envs_idx"
    )
    print(f"reset_info n_vertices={reset_info['n_vertices']} length_m={reset_info['length_m']}")

    with TransitionWriter(dataset_path, meta=meta, mode="w") as writer:
        while writer.record_count < target_count:
            if primitives_in_episode == 0:
                vertices, shapes, curve_seeds = build_initial_vertices(
                    params,
                    n_envs=n_envs,
                    episode_index=episode_index,
                    seed=seed,
                    init_shapes=init_shapes,
                )
                reset_start = time.perf_counter()
                reset_result = env.light_reset(
                    vertices,
                    vel_threshold=vel_threshold,
                    max_steps=settle_max_steps,
                )
                reset_wall_s = time.perf_counter() - reset_start
                reset_record = {
                    "episode_index": int(episode_index),
                    "wall_s": reset_wall_s,
                    "settle_steps_max": int(np.max(reset_result["settle_steps"])),
                    "settle_converged_rate": float(np.mean(reset_result["settle_converged"])),
                    "init_shapes": shapes,
                    "curve_seeds_first_last": [int(curve_seeds[0]), int(curve_seeds[-1])],
                }
                reset_records.append(reset_record)
                print(
                    "light_reset "
                    f"episode={episode_index} wall_s={reset_wall_s:.3f} "
                    f"settle_steps_max={reset_record['settle_steps_max']} "
                    f"converged_rate={reset_record['settle_converged_rate']:.3f}"
                )

            round_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            round_rng = np.random.default_rng(round_seed)
            p, deltas, lifts = sample_actions(
                round_rng,
                n_envs=n_envs,
                n_vertices=int(params.n_segments),
                collection_cfg=collection_cfg,
            )
            round_start = time.perf_counter()
            try:
                result = env.step_primitive_batch(
                    p,
                    deltas,
                    lifts,
                    vel_threshold=vel_threshold,
                    max_steps=settle_max_steps,
                    rng=round_rng,
                )
            except (FloatingPointError, ValueError, RuntimeError) as exc:
                recovery_record = {
                    "attempted_round": int(round_index + 1),
                    "episode_index": int(episode_index),
                    "primitives_in_episode": int(primitives_in_episode),
                    "round_seed": int(round_seed),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                recovery_records.append(recovery_record)
                print(
                    "round_recovery "
                    f"attempted_round={round_index + 1} episode={episode_index} "
                    f"error={recovery_record['error']} action=full_scene_rebuild"
                )
                del env
                gc.collect()
                env = DLOLabEnv(**env_kwargs(config, n_envs))
                env.reset(params, init_shape="straight", seed=seed + 10_000 + len(recovery_records))
                if not env.supports_per_env_grasp():
                    raise RuntimeError("per-env rod attach/detach hooks unavailable after recovery")
                primitives_in_episode = 0
                episode_index += 1
                continue
            round_wall_s = time.perf_counter() - round_start
            appended = append_batch_records(
                writer,
                result=result,
                p=p,
                deltas=deltas,
                lifts=lifts,
                params=params,
                round_seed=round_seed,
                commit_hash=commit_hash,
            )
            round_times.append(round_wall_s)
            success = np.asarray(result["grasp_success"], dtype=bool)
            settle_steps = np.asarray(result["settle_steps"], dtype=int)
            converged = settle_steps != settle_max_steps
            total_success += int(success.sum())
            total_converged += int(converged.sum())
            round_index += 1
            primitives_in_episode += 1
            if primitives_in_episode >= K:
                primitives_in_episode = 0
                episode_index += 1

            elapsed = time.perf_counter() - start_wall
            rate = writer.record_count / elapsed if elapsed > 0 else 0.0
            remaining = max(0, target_count - writer.record_count)
            eta = remaining / rate if rate > 0 else 0.0
            print(
                "round "
                f"{round_index:04d} appended={appended} count={writer.record_count} "
                f"wall_s={round_wall_s:.3f} success_rate_batch={np.mean(success):.3f} "
                f"converged_rate_batch={np.mean(converged):.3f} "
                f"settle_steps_max={int(np.max(settle_steps))} eta={format_eta(eta)}"
            )
            if writer.record_count > max_records:
                raise RuntimeError(
                    f"collection exceeded max_records={max_records}; count={writer.record_count}"
                )

    total_wall_s = time.perf_counter() - start_wall
    with h5py.File(dataset_path, "r") as h5:
        actual_record_count = int(h5.attrs["record_count"])
    update_h5_meta(
        dataset_path,
        {
            "total_wall_time_s": total_wall_s,
            "record_count": actual_record_count,
            "round_count": int(round_index),
            "round_wall_s_mean": float(np.mean(round_times)) if round_times else 0.0,
            "round_wall_s_median": float(np.median(round_times)) if round_times else 0.0,
            "success_rate_running": total_success / max(1, round_index * n_envs),
            "convergence_rate_running": total_converged / max(1, round_index * n_envs),
            "light_reset_records": reset_records,
            "recovery_records": recovery_records,
        },
    )
    print(
        "collection complete "
        f"records>={target_count} dataset={dataset_path} total_wall_s={total_wall_s:.3f} "
        f"rounds={round_index} mean_s_per_round={np.mean(round_times):.3f}"
    )
    return {
        "dataset_path": str(dataset_path),
        "commit_hash": commit_hash,
        "n_envs": int(n_envs),
        "K": int(K),
        "probe": probe_results,
        "total_wall_time_s": total_wall_s,
        "round_count": int(round_index),
        "round_wall_s": round_times,
        "reset_records": reset_records,
        "recovery_records": recovery_records,
    }


def summarize_vector(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    quantile_points = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "quantiles": {f"q{int(q * 100):02d}": float(np.quantile(arr, q)) for q in quantile_points},
    }


def write_histograms(delta_ms: np.ndarray, stats_path: Path, norm_path: Path) -> None:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, M, figsize=(18.0, 7.0), constrained_layout=True)
    for channel, ax in enumerate(axes.reshape(-1)):
        ax.hist(delta_ms[:, channel], bins=50, color="#3366aa", alpha=0.85)
        ax.set_title(CHANNEL_LAYOUT[channel], fontsize=8)
        ax.tick_params(labelsize=6)
    fig.suptitle("δm per-channel histograms")
    fig.savefig(stats_path, dpi=150)
    plt.close(fig)

    mode_norm = np.linalg.norm(delta_ms, axis=1)
    fig, ax = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    ax.hist(mode_norm, bins=80, color="#aa6633", alpha=0.85)
    ax.set_title("δm vector norm histogram")
    ax.set_xlabel("||δm||₂")
    ax.set_ylabel("count")
    fig.savefig(norm_path, dpi=150)
    plt.close(fig)


def compute_dm_outputs(
    dataset_path: Path,
    config: dict[str, Any],
    run_info: dict[str, Any],
    drift_probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_cfg = config.get("outputs", {})
    collection_cfg = config.get("collection", {})
    stats_path = Path(output_cfg.get("dm_stats_json", "outputs/metrics/dm_stats.json"))
    normalizer_path = Path(output_cfg.get("dm_normalizer_json", "outputs/metrics/dm_normalizer.json"))
    hist_path = Path(output_cfg.get("dm_hist_png", "outputs/plots/dm_hist_all_channels.png"))
    norm_hist_path = Path(output_cfg.get("dm_norm_hist_png", "outputs/plots/dm_hist_norms.png"))
    settle_max_steps = int(collection_cfg.get("settle_max_steps", 5000))

    records, meta = read_transitions(dataset_path)
    if not records:
        raise RuntimeError("cannot compute δm stats for an empty transition dataset")
    delta_ms = np.stack([
        Phi_DCT(resample(record.X_after)) - Phi_DCT(resample(record.X_before))
        for record in records
    ])
    success = np.asarray([record.grasp_success for record in records], dtype=bool)
    settle_steps = np.asarray([record.settle_steps for record in records], dtype=int)
    converged = settle_steps != settle_max_steps
    fit_mask = success & converged
    if int(fit_mask.sum()) <= 0:
        raise RuntimeError("normalizer fit set is empty (requires converged successful transitions)")

    normalizer = DmNormalizer.fit(delta_ms[fit_mask])
    normalizer_payload = normalizer.stats.to_dict()
    normalizer_payload.update(
        {
            "fit_filter": "grasp_success == true and settle_steps != settle_max_steps",
            "fit_count_success_converged": int(fit_mask.sum()),
            "source_dataset": str(dataset_path),
            "commit_hash": run_info["commit_hash"],
            "created_at": utc_now(),
        }
    )
    normalizer_path.parent.mkdir(parents=True, exist_ok=True)
    normalizer_path.write_text(json.dumps(as_jsonable(normalizer_payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    per_channel = []
    for channel, name in enumerate(CHANNEL_LAYOUT):
        entry = {"index": channel, "name": name}
        entry.update(summarize_vector(delta_ms[:, channel]))
        per_channel.append(entry)

    cross = {}
    for success_value in (False, True):
        for converged_value in (False, True):
            mask = (success == success_value) & (converged == converged_value)
            cross[f"success_{str(success_value).lower()}__converged_{str(converged_value).lower()}"] = int(mask.sum())

    dm_norm = np.linalg.norm(delta_ms, axis=1)
    stats_payload = {
        "schema_version": 1,
        "source_dataset": str(dataset_path),
        "record_count": len(records),
        "commit_hash": run_info["commit_hash"],
        "created_at": utc_now(),
        "channel_layout_id": CHANNEL_LAYOUT_ID,
        "channel_layout": list(CHANNEL_LAYOUT),
        "M": M,
        "convergence_rule": {
            "converged": "settle_steps != max_steps",
            "max_steps": settle_max_steps,
            "plan": "A1",
        },
        "grasp_mode": "per-env",
        "grasp_mode_evidence": "external/DLO-Lab/genesis/engine/entities/rod_entity.py defines attach_to_rigid_link_with_envs_idx and detach_from_rigid_link_with_envs_idx",
        "totals": {
            "grasp_success": {"true": int(success.sum()), "false": int((~success).sum())},
            "settle_converged": {"true": int(converged.sum()), "false": int((~converged).sum())},
            "cross": cross,
        },
        "rates": {
            "grasp_success": float(np.mean(success)),
            "settle_converged": float(np.mean(converged)),
            "success_and_converged": float(np.mean(fit_mask)),
        },
        "delta_m_norm": summarize_vector(dm_norm),
        "per_channel": per_channel,
        "normalizer": {
            "path": str(normalizer_path),
            "fit_count": int(fit_mask.sum()),
            "std_mean_shape_channels": float(np.mean(normalizer.stats.std[[i for i in range(24) if i not in (0, 8, 16)]])),
        },
        "probe": as_jsonable(run_info.get("probe", [])),
        "n_envs": int(run_info["n_envs"]),
        "K": int(run_info["K"]),
        "total_wall_time_s": float(run_info["total_wall_time_s"]),
        "round_count": int(run_info["round_count"]),
        "round_wall_s_mean": float(np.mean(run_info["round_wall_s"])) if run_info["round_wall_s"] else 0.0,
        "physics_quality_note": (
            "For gate/human review, prefer rates.success_and_converged "
            f"({float(np.mean(fit_mask)):.3f}) over the headline settle_converged rate: failed grasps "
            "are no-op transitions whose converged flag reflects the untouched rope, so the aggregate "
            "conflates populations. Settle non-convergence at the immutable 1e-3/5000 budget affects "
            f"{int((success & ~converged).sum())} successful transitions; they are recorded honestly "
            "(settle_steps == max_steps) and excluded from the normalizer fit. Flagged for the M5/M6 "
            "human gates (plan A1)."
        ),
        "restoration_drift_probe": as_jsonable(drift_probe) if drift_probe else (
            "not measured for this run (instrumentation added post-collection at the M4 gate; "
            "future runs log restoration_drift_max_m/mean_m per round)"
        ),
        "metadata": meta,
        "plots": [str(hist_path), str(norm_hist_path)],
    }

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(as_jsonable(stats_payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_histograms(delta_ms, hist_path, norm_hist_path)
    print(
        "dm_stats complete "
        f"records={len(records)} success_rate={stats_payload['rates']['grasp_success']:.3f} "
        f"converged_rate={stats_payload['rates']['settle_converged']:.3f} "
        f"normalizer_fit_count={fit_mask.sum()} stats={stats_path} normalizer={normalizer_path}"
    )
    print(f"dm_histograms {hist_path} {norm_hist_path}")
    return stats_payload


def run(
    config: dict[str, Any],
    config_text: str,
    seed: int,
    config_path: Path,
    *,
    stats_only: bool = False,
    drift_probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if stats_only:
        output_cfg = config.get("outputs", {})
        stats_path = Path(output_cfg.get("dm_stats_json", "outputs/metrics/dm_stats.json"))
        dataset_path = Path(output_cfg.get("dataset", "outputs/data/p0_random_transitions.h5"))
        prior = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
        run_info = {
            "dataset_path": str(dataset_path),
            "commit_hash": get_git_commit_hash(),
            "probe": prior.get("probe", []),
            "n_envs": int(prior.get("n_envs", 0)),
            "K": int(prior.get("K", 0)),
            "total_wall_time_s": float(prior.get("total_wall_time_s", 0.0)),
            "round_count": int(prior.get("round_count", 0)),
            "round_wall_s": [],
        }
        stats = compute_dm_outputs(dataset_path, config, run_info, drift_probe=drift_probe)
        print("COLLECT_RANDOM STATS-ONLY PASS")
        return {"run_info": run_info, "dm_stats": stats}

    run_info = collect(config, config_text, seed, config_path)
    stats = compute_dm_outputs(Path(run_info["dataset_path"]), config, run_info, drift_probe=drift_probe)
    print("COLLECT_RANDOM PASS")
    return {"run_info": run_info, "dm_stats": stats}


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    config, config_text = load_config(config_path)
    output_cfg = config.get("outputs", {})
    log_path = Path(output_cfg.get("stdout_log", "outputs/reports/collect_random_stdout.log"))
    if args.stats_only:
        # Never clobber the committed collection log with a stats-only recompute.
        log_path = log_path.with_name(log_path.stem + "_stats_only.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            drift_probe = None
            if args.drift_probe_json:
                drift_probe = json.loads(Path(args.drift_probe_json).read_text(encoding="utf-8"))
            run(
                config,
                config_text,
                int(args.seed),
                config_path,
                stats_only=bool(args.stats_only),
                drift_probe=drift_probe,
            )
            return 0
        except Exception as exc:
            print(f"COLLECT_RANDOM FAIL {type(exc).__name__}: {exc}")
            return 1
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    raise SystemExit(main())
