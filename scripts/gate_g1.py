"""Run the P0-M6 G1 stiffness/friction effect-size pilot.

The script uses the DLO-Lab adapter batch APIs added in M4.  The fixture is the
literal sequence list in ``configs/gate_g1.yaml``; this file measures and reports
facts only for the human G1 gate.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from dgcc.envs.base import RopeParams
from dgcc.envs.dlolab import DLOLabEnv, analytic_init_centerline, mapped_parameters, stiffness_bases
from dgcc.goals.distance import chamfer_distance
from dgcc.utils.meta import get_git_commit_hash

os.environ.pop("DISPLAY", None)

VALID_INIT_SHAPES = ("straight", "u_bend", "s_curve", "random_smooth")
VALID_LIFTS = ("low", "high")
AXES = ("stiffness", "friction")
REPORT_BANNED_TERMS = (
    "유의미",
    "공허",
    "springback",
    "recommend",
    "recommendation",
    "권고",
    "추천",
)


@dataclass(frozen=True)
class Primitive:
    p: int
    delta: tuple[float, float, float]
    lift: str


@dataclass(frozen=True)
class SequenceFixture:
    id: str
    init_shape: str
    primitives: tuple[Primitive, ...]


@dataclass(frozen=True)
class RunSpec:
    axis: str
    sequence_index: int
    sequence_id: str
    init_shape: str
    seed_index: int
    init_seed: int
    condition: float
    primitive_count: int


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
    parser = argparse.ArgumentParser(description="Measure the P0-M6 G1 stiffness/friction pilot")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="deterministic bootstrap/probe seed; --stats-only reuses stored cli_seed when omitted",
    )
    parser.add_argument("--config", default="configs/gate_g1.yaml", help="YAML config path")
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="regenerate metrics, report, and plots from stored raw distance lists without running simulations",
    )
    return parser


@contextmanager
def tee_stdout(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original = sys.stdout
    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original, log_file)
        try:
            yield
        finally:
            sys.stdout = original


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    config = yaml.safe_load(text) or {}
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping, got {type(config).__name__}")
    return config, text


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def as_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): as_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def condition_label(value: float) -> str:
    value = float(value)
    return f"{value:.1f}" if value.is_integer() else f"{value:g}"


def pair_key(pair: Sequence[float]) -> str:
    return f"{condition_label(float(pair[0]))}_vs_{condition_label(float(pair[1]))}"


def params_from_config(config: dict[str, Any]) -> RopeParams:
    rope = config.get("rope", {})
    return RopeParams(
        length_m=float(rope.get("length_m", 1.0)),
        n_segments=int(rope.get("n_segments", 50)),
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
        "grasp_realism": bool(sim.get("grasp_realism", False)),
    }


def output_paths(config: dict[str, Any]) -> dict[str, Path]:
    outputs = config.get("outputs", {})
    return {
        "metrics_json": Path(outputs.get("metrics_json", "outputs/metrics/g1_effect_size.json")),
        "stiffness_plot_png": Path(outputs.get("stiffness_plot_png", "outputs/plots/g1_stiffness_distributions.png")),
        "friction_plot_png": Path(outputs.get("friction_plot_png", "outputs/plots/g1_friction_distributions.png")),
        "report_md": Path(outputs.get("report_md", "outputs/reports/g1_report.md")),
        "stdout_log": Path(outputs.get("stdout_log", "outputs/reports/gate_g1_stdout.log")),
        "dm_stats_json": Path(outputs.get("dm_stats_json", "outputs/metrics/dm_stats.json")),
    }


def parse_sequences(config: dict[str, Any], *, n_vertices: int) -> tuple[SequenceFixture, ...]:
    raw_sequences = config.get("sequences")
    if not isinstance(raw_sequences, list):
        raise ValueError("config.sequences must be a list")

    sequences: list[SequenceFixture] = []
    ids: set[str] = set()
    shape_counts: dict[str, int] = defaultdict(int)
    for raw in raw_sequences:
        if not isinstance(raw, dict):
            raise ValueError("each sequence must be a mapping")
        seq_id = str(raw.get("id", ""))
        if not seq_id:
            raise ValueError("each sequence requires id")
        if seq_id in ids:
            raise ValueError(f"duplicate sequence id {seq_id!r}")
        ids.add(seq_id)
        init_shape = str(raw.get("init_shape", ""))
        if init_shape not in VALID_INIT_SHAPES:
            raise ValueError(f"sequence {seq_id} has invalid init_shape {init_shape!r}")
        raw_primitives = raw.get("primitives")
        if not isinstance(raw_primitives, list) or not 2 <= len(raw_primitives) <= 4:
            raise ValueError(f"sequence {seq_id} must contain 2-4 primitives")
        primitives: list[Primitive] = []
        for idx, primitive in enumerate(raw_primitives):
            if not isinstance(primitive, dict):
                raise ValueError(f"sequence {seq_id} primitive {idx} must be a mapping")
            p = int(primitive["p"])
            if p < 0 or p >= n_vertices:
                raise ValueError(f"sequence {seq_id} primitive {idx} p={p} outside [0, {n_vertices})")
            delta_array = np.asarray(primitive["delta"], dtype=float)
            if delta_array.shape != (3,) or not np.all(np.isfinite(delta_array)):
                raise ValueError(f"sequence {seq_id} primitive {idx} delta must be finite shape (3,)")
            if float(np.linalg.norm(delta_array)) > 0.15 + 1e-12:
                raise ValueError(f"sequence {seq_id} primitive {idx} delta exceeds 0.15 m")
            lift = str(primitive["lift"])
            if lift not in VALID_LIFTS:
                raise ValueError(f"sequence {seq_id} primitive {idx} lift={lift!r} invalid")
            primitives.append(Primitive(p=p, delta=tuple(float(x) for x in delta_array), lift=lift))
        shape_counts[init_shape] += 1
        sequences.append(SequenceFixture(id=seq_id, init_shape=init_shape, primitives=tuple(primitives)))

    if len(sequences) != 20:
        raise ValueError(f"G1 fixture must contain 20 sequences, got {len(sequences)}")
    expected_counts = {shape: 5 for shape in VALID_INIT_SHAPES}
    if dict(shape_counts) != expected_counts:
        raise ValueError(f"G1 fixture must contain 5 sequences per init shape, got {dict(shape_counts)}")
    return tuple(sequences)


def measurement_config(config: dict[str, Any]) -> dict[str, Any]:
    measurement = config.get("measurement", {})
    if not isinstance(measurement, dict):
        raise ValueError("config.measurement must be a mapping")
    return measurement


def measurement_lists(config: dict[str, Any]) -> tuple[list[int], list[float], list[float], list[tuple[float, float]]]:
    measurement = measurement_config(config)
    init_seeds = [int(value) for value in measurement.get("init_seeds", [0, 1, 2])]
    stiffness_conditions = [float(value) for value in measurement.get("stiffness_multipliers", [0.5, 1.0, 2.0])]
    friction_conditions = [float(value) for value in measurement.get("friction_multipliers", [0.5, 1.0, 2.0])]
    pairs = [tuple(float(x) for x in pair) for pair in measurement.get("condition_pairs", [(0.5, 1.0), (1.0, 2.0), (0.5, 2.0)])]
    if len(init_seeds) != 3:
        raise ValueError(f"G1 requires exactly 3 init seeds, got {init_seeds}")
    for name, conditions in (("stiffness", stiffness_conditions), ("friction", friction_conditions)):
        if conditions != [0.5, 1.0, 2.0]:
            raise ValueError(f"{name} conditions must be [0.5, 1.0, 2.0], got {conditions}")
    if pairs != [(0.5, 1.0), (1.0, 2.0), (0.5, 2.0)]:
        raise ValueError(f"condition pairs must be [(0.5,1.0),(1.0,2.0),(0.5,2.0)], got {pairs}")
    return init_seeds, stiffness_conditions, friction_conditions, pairs


def make_params(base: RopeParams, axis: str, condition: float) -> RopeParams:
    if axis == "stiffness":
        return RopeParams(
            length_m=base.length_m,
            n_segments=base.n_segments,
            bend_stiffness=base.bend_stiffness * float(condition),
            twist_stiffness=base.twist_stiffness * float(condition),
            friction=base.friction,
            radius=base.radius,
        )
    if axis == "friction":
        return RopeParams(
            length_m=base.length_m,
            n_segments=base.n_segments,
            bend_stiffness=base.bend_stiffness,
            twist_stiffness=base.twist_stiffness,
            friction=base.friction * float(condition),
            radius=base.radius,
        )
    raise ValueError(f"unknown axis {axis!r}")


def build_run_specs(
    sequences: Sequence[SequenceFixture],
    *,
    axis: str,
    init_seeds: Sequence[int],
    conditions: Sequence[float],
) -> list[RunSpec]:
    records: list[RunSpec] = []
    for sequence_index, sequence in enumerate(sequences):
        for seed_index, init_seed in enumerate(init_seeds):
            for condition in conditions:
                records.append(
                    RunSpec(
                        axis=axis,
                        sequence_index=sequence_index,
                        sequence_id=sequence.id,
                        init_shape=sequence.init_shape,
                        seed_index=seed_index,
                        init_seed=int(init_seed),
                        condition=float(condition),
                        primitive_count=len(sequence.primitives),
                    )
                )
    return records


def group_run_specs(records: Sequence[RunSpec], *, mixed_conditions: bool) -> list[list[RunSpec]]:
    grouped: dict[tuple[Any, ...], list[RunSpec]] = defaultdict(list)
    for record in records:
        key = (record.axis, record.primitive_count) if mixed_conditions else (record.axis, record.primitive_count, record.condition)
        grouped[key].append(record)
    return [grouped[key] for key in sorted(grouped)]


def cleanup_env(env: DLOLabEnv | None) -> None:
    if env is not None:
        del env
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def discover_batch_support(env: DLOLabEnv) -> dict[str, Any]:
    env._require_reset()  # existing adapter invariant; discovery is internal to this script.
    assert env.rod_entity is not None
    setter_names = ("set_bending_stiffness", "set_twisting_stiffness", "set_mu_s")
    setter_signatures: dict[str, str] = {}
    setter_envs_idx: dict[str, bool] = {}
    for name in setter_names:
        method = getattr(env.rod_entity, name, None)
        if method is None:
            setter_signatures[name] = "missing"
            setter_envs_idx[name] = False
            continue
        signature = inspect.signature(method)
        setter_signatures[name] = f"{name}{signature}"
        setter_envs_idx[name] = "envs_idx" in signature.parameters
    support = {
        "per_env_grasp": bool(env.supports_per_env_grasp()),
        "per_env_param_setters": bool(all(setter_envs_idx.values())),
        "setter_envs_idx": setter_envs_idx,
        "setter_signatures": setter_signatures,
        "source": "external/DLO-Lab/genesis/engine/entities/rod_entity.py:1288-1423",
    }
    return support


def apply_params_batch(env: DLOLabEnv, params_by_env: Sequence[RopeParams]) -> list[dict[str, float]]:
    env._require_reset()
    assert env.gs is not None and env.rod_entity is not None
    import torch

    if len(params_by_env) != env.n_envs:
        raise ValueError(f"params_by_env length {len(params_by_env)} does not match n_envs={env.n_envs}")
    mapped = [mapped_parameters(params) for params in params_by_env]
    dtype = env.gs.tc_float
    device = env.gs.device
    n_vertices = int(params_by_env[0].n_segments)
    env.rod_entity.set_bending_stiffness(
        torch.tensor([item["bending_stiffness_E"] for item in mapped], dtype=dtype, device=device)
    )
    env.rod_entity.set_twisting_stiffness(
        torch.tensor([item["twisting_stiffness_G"] for item in mapped], dtype=dtype, device=device)
    )
    env.rod_entity.set_stretching_stiffness(
        torch.tensor([item["stretching_stiffness_K"] for item in mapped], dtype=dtype, device=device)
    )
    env.rod_entity.set_mu_s(
        torch.tensor([[item["mu_s"]] * n_vertices for item in mapped], dtype=dtype, device=device)
    )
    env.rod_entity.set_mu_k(
        torch.tensor([[item["mu_k"]] * n_vertices for item in mapped], dtype=dtype, device=device)
    )
    env.rod_entity.set_segment_radius(
        torch.tensor([[item["segment_radius"]] * n_vertices for item in mapped], dtype=dtype, device=device)
    )
    env.rod_entity.set_segment_mass(
        torch.tensor([[item["segment_mass"]] * n_vertices for item in mapped], dtype=dtype, device=device)
    )
    return mapped


def initial_vertices_for_records(base: RopeParams, records: Sequence[RunSpec]) -> np.ndarray:
    vertices = [analytic_init_centerline(base, record.init_shape, record.init_seed) for record in records]
    return np.stack(vertices, axis=0)


def run_timing_probe(
    *,
    config: dict[str, Any],
    base_params: RopeParams,
    sequences_by_id: dict[str, SequenceFixture],
    init_seeds: Sequence[int],
    cli_seed: int,
) -> dict[str, Any]:
    measurement = measurement_config(config)
    probe_cfg = measurement.get("timing_probe", {})
    if not bool(probe_cfg.get("enabled", True)):
        return {"enabled": False}

    axis = str(probe_cfg.get("axis", "stiffness"))
    sequence_id = str(probe_cfg.get("sequence_id", "straight_01"))
    condition = float(probe_cfg.get("condition", 1.0))
    primitive_index = int(probe_cfg.get("primitive_index", 0))
    if axis not in AXES:
        raise ValueError(f"timing probe axis {axis!r} invalid")
    sequence = sequences_by_id[sequence_id]
    if primitive_index < 0 or primitive_index >= len(sequence.primitives):
        raise ValueError(f"timing probe primitive_index {primitive_index} invalid for {sequence_id}")
    records = [
        RunSpec(
            axis=axis,
            sequence_index=0,
            sequence_id=sequence_id,
            init_shape=sequence.init_shape,
            seed_index=seed_index,
            init_seed=int(init_seed),
            condition=condition,
            primitive_count=len(sequence.primitives),
        )
        for seed_index, init_seed in enumerate(init_seeds)
    ]

    env: DLOLabEnv | None = None
    try:
        start_build = time.perf_counter()
        env = DLOLabEnv(**env_kwargs(config, len(records)))
        reset_info = env.reset(base_params, init_shape="straight", seed=cli_seed + 101)
        build_wall_s = time.perf_counter() - start_build
        support = discover_batch_support(env)
        if not support["per_env_grasp"]:
            raise RuntimeError("DLO-Lab per-env grasp hooks are unavailable")

        measurement_cfg = measurement_config(config)
        vertices = initial_vertices_for_records(base_params, records)
        reset_start = time.perf_counter()
        reset_result = env.light_reset(
            vertices,
            vel_threshold=float(measurement_cfg.get("vel_threshold", 1.0e-3)),
            max_steps=int(measurement_cfg.get("settle_max_steps", 5000)),
        )
        light_reset_wall_s = time.perf_counter() - reset_start
        params_by_env = [make_params(base_params, axis, condition) for _ in records]
        mapped = apply_params_batch(env, params_by_env)

        primitive = sequence.primitives[primitive_index]
        p = np.full(len(records), primitive.p, dtype=int)
        delta = np.tile(np.asarray(primitive.delta, dtype=float), (len(records), 1))
        lifts = [primitive.lift] * len(records)
        round_start = time.perf_counter()
        result = env.step_primitive_batch(
            p,
            delta,
            lifts,
            vel_threshold=float(measurement_cfg.get("vel_threshold", 1.0e-3)),
            max_steps=int(measurement_cfg.get("settle_max_steps", 5000)),
            rng=np.random.default_rng(cli_seed + 202),
        )
        round_wall_s = time.perf_counter() - round_start
        settle_steps = np.asarray(result["settle_steps"], dtype=int)
        probe = {
            "enabled": True,
            "axis": axis,
            "condition": condition,
            "sequence_id": sequence_id,
            "primitive_index": primitive_index,
            "n_envs": len(records),
            "build_wall_s": build_wall_s,
            "light_reset_wall_s": light_reset_wall_s,
            "round_wall_s": round_wall_s,
            "per_env_round_s": round_wall_s / max(1, len(records)),
            "reset_converged_rate": float(np.mean(np.asarray(reset_result["settle_converged"], dtype=bool))),
            "round_converged_rate": float(np.mean(settle_steps != int(measurement_cfg.get("settle_max_steps", 5000)))),
            "round_settle_steps_max": int(np.max(settle_steps)) if settle_steps.size else 0,
            "reset_info": reset_info,
            "batch_support": support,
            "mapped_parameters_first_env": mapped[0] if mapped else {},
        }
        print(
            "timing_probe "
            f"axis={axis} condition={condition_label(condition)} n_envs={len(records)} "
            f"round_wall_s={round_wall_s:.3f} per_env_round_s={probe['per_env_round_s']:.3f} "
            f"param_envs_idx={support['per_env_param_setters']}"
        )
        return probe
    finally:
        cleanup_env(env)


def execute_group(
    *,
    config: dict[str, Any],
    base_params: RopeParams,
    sequences_by_id: dict[str, SequenceFixture],
    records: Sequence[RunSpec],
    cli_seed: int,
    group_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not records:
        return [], {}
    primitive_count = records[0].primitive_count
    if any(record.primitive_count != primitive_count for record in records):
        raise ValueError("execute_group requires records with equal primitive_count")

    measurement = measurement_config(config)
    vel_threshold = float(measurement.get("vel_threshold", 1.0e-3))
    settle_max_steps = int(measurement.get("settle_max_steps", 5000))
    axis = records[0].axis
    env: DLOLabEnv | None = None
    try:
        start_build = time.perf_counter()
        env = DLOLabEnv(**env_kwargs(config, len(records)))
        reset_info = env.reset(base_params, init_shape="straight", seed=cli_seed + 10_000 + group_index)
        build_wall_s = time.perf_counter() - start_build
        support = discover_batch_support(env)
        if not support["per_env_grasp"]:
            raise RuntimeError("DLO-Lab per-env grasp hooks are unavailable")

        vertices = initial_vertices_for_records(base_params, records)
        reset_start = time.perf_counter()
        reset_result = env.light_reset(vertices, vel_threshold=vel_threshold, max_steps=settle_max_steps)
        light_reset_wall_s = time.perf_counter() - reset_start
        params_by_env = [make_params(base_params, record.axis, record.condition) for record in records]
        mapped = apply_params_batch(env, params_by_env)

        step_summaries: list[dict[str, Any]] = []
        per_env_steps = np.zeros((len(records), primitive_count), dtype=int)
        per_env_converged = np.zeros((len(records), primitive_count), dtype=bool)
        per_env_success = np.zeros((len(records), primitive_count), dtype=bool)
        for primitive_index in range(primitive_count):
            primitives = [sequences_by_id[record.sequence_id].primitives[primitive_index] for record in records]
            p = np.asarray([primitive.p for primitive in primitives], dtype=int)
            delta = np.asarray([primitive.delta for primitive in primitives], dtype=float)
            lifts = [primitive.lift for primitive in primitives]
            round_start = time.perf_counter()
            result = env.step_primitive_batch(
                p,
                delta,
                lifts,
                vel_threshold=vel_threshold,
                max_steps=settle_max_steps,
                rng=np.random.default_rng(cli_seed + 20_000 + group_index * 17 + primitive_index),
            )
            round_wall_s = time.perf_counter() - round_start
            settle_steps = np.asarray(result["settle_steps"], dtype=int)
            success = np.asarray(result["grasp_success"], dtype=bool)
            converged = settle_steps != settle_max_steps
            per_env_steps[:, primitive_index] = settle_steps
            per_env_converged[:, primitive_index] = converged
            per_env_success[:, primitive_index] = success
            step_summary = {
                "primitive_index": primitive_index,
                "wall_s": round_wall_s,
                "settle_steps_max": int(np.max(settle_steps)) if settle_steps.size else 0,
                "settle_steps_mean": float(np.mean(settle_steps)) if settle_steps.size else 0.0,
                "converged_rate": float(np.mean(converged)) if converged.size else 0.0,
                "grasp_success_rate": float(np.mean(success)) if success.size else 0.0,
            }
            step_summaries.append(step_summary)
            print(
                "batch_step "
                f"axis={axis} group={group_index} primitive={primitive_index + 1}/{primitive_count} "
                f"n_envs={len(records)} wall_s={round_wall_s:.3f} "
                f"converged_rate={step_summary['converged_rate']:.3f}"
            )

        final_centerlines = env.get_centerline_batch()
        final_speeds = env.max_node_speed_batch()
        reset_converged = np.asarray(reset_result["settle_converged"], dtype=bool)
        reset_steps = np.asarray(reset_result["settle_steps"], dtype=int)
        outputs: list[dict[str, Any]] = []
        for env_idx, record in enumerate(records):
            outputs.append(
                {
                    "axis": record.axis,
                    "sequence_index": record.sequence_index,
                    "sequence_id": record.sequence_id,
                    "init_shape": record.init_shape,
                    "seed_index": record.seed_index,
                    "init_seed": record.init_seed,
                    "condition": record.condition,
                    "condition_label": condition_label(record.condition),
                    "rope_params": asdict(params_by_env[env_idx]),
                    "mapped_parameters": mapped[env_idx],
                    "final_centerline": final_centerlines[env_idx].copy(),
                    "final_max_node_speed": float(final_speeds[env_idx]),
                    "reset_settle_converged": bool(reset_converged[env_idx]),
                    "reset_settle_steps": int(reset_steps[env_idx]),
                    "primitive_settle_steps": per_env_steps[env_idx].copy(),
                    "primitive_settle_converged": per_env_converged[env_idx].copy(),
                    "primitive_grasp_success": per_env_success[env_idx].copy(),
                }
            )

        group_summary = {
            "group_index": group_index,
            "axis": axis,
            "primitive_count": primitive_count,
            "n_envs": len(records),
            "conditions": sorted({record.condition for record in records}),
            "sequence_count": len({record.sequence_id for record in records}),
            "build_wall_s": build_wall_s,
            "light_reset_wall_s": light_reset_wall_s,
            "reset_converged_rate": float(np.mean(reset_converged)) if reset_converged.size else 0.0,
            "reset_settle_steps_max": int(np.max(reset_steps)) if reset_steps.size else 0,
            "step_summaries": step_summaries,
            "batch_support": support,
            "reset_info_n_vertices": int(reset_info["n_vertices"]),
        }
        print(
            "batch_done "
            f"axis={axis} group={group_index} primitive_count={primitive_count} n_envs={len(records)} "
            f"reset_converged_rate={group_summary['reset_converged_rate']:.3f}"
        )
        return outputs, group_summary
    finally:
        cleanup_env(env)


def run_full_measurement(
    *,
    config: dict[str, Any],
    base_params: RopeParams,
    sequences: Sequence[SequenceFixture],
    init_seeds: Sequence[int],
    stiffness_conditions: Sequence[float],
    friction_conditions: Sequence[float],
    cli_seed: int,
    mixed_conditions: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_records = build_run_specs(sequences, axis="stiffness", init_seeds=init_seeds, conditions=stiffness_conditions)
    all_records.extend(build_run_specs(sequences, axis="friction", init_seeds=init_seeds, conditions=friction_conditions))
    groups = group_run_specs(all_records, mixed_conditions=mixed_conditions)
    sequences_by_id = {sequence.id: sequence for sequence in sequences}
    outputs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    total_envs = sum(len(group) for group in groups)
    print(
        "full_run_start "
        f"groups={len(groups)} total_env_runs={total_envs} mixed_conditions={mixed_conditions}"
    )
    for group_index, group in enumerate(groups, start=1):
        print(
            "batch_start "
            f"group={group_index}/{len(groups)} axis={group[0].axis} "
            f"primitive_count={group[0].primitive_count} n_envs={len(group)} "
            f"conditions={sorted({record.condition for record in group})}"
        )
        group_outputs, group_summary = execute_group(
            config=config,
            base_params=base_params,
            sequences_by_id=sequences_by_id,
            records=group,
            cli_seed=cli_seed,
            group_index=group_index,
        )
        outputs.extend(group_outputs)
        summaries.append(group_summary)
    return outputs, summaries


def describe_distribution(values: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"n": 0}
    quantiles = np.quantile(arr, [0.0, 0.025, 0.25, 0.5, 0.75, 0.975, 1.0])
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(quantiles[0]),
        "q025": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q975": float(quantiles[5]),
        "max": float(quantiles[6]),
    }


def cohens_d(sample_a: Sequence[float], sample_b: Sequence[float]) -> float:
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    denom_df = a.size + b.size - 2
    if denom_df <= 0:
        return float("nan")
    pooled = math.sqrt(((a.size - 1) * var_a + (b.size - 1) * var_b) / denom_df)
    if pooled == 0.0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


def bootstrap_ci_key(level: float, method: str) -> str:
    percent = int(round(float(level) * 100.0))
    if math.isclose(float(level) * 100.0, float(percent), rel_tol=0.0, abs_tol=1.0e-9):
        level_label = str(percent)
    else:
        level_label = str(float(level)).replace(".", "_")
    return f"bootstrap_ci_{level_label}_{method}"


def bootstrap_d_ci(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    *,
    replicates: int,
    level: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    if a.size < 2 or b.size < 2 or replicates <= 0:
        return {"level": level, "replicates": int(replicates), "low": None, "high": None}
    draws = np.empty(int(replicates), dtype=float)
    for idx in range(int(replicates)):
        a_sample = a[rng.integers(0, a.size, size=a.size)]
        b_sample = b[rng.integers(0, b.size, size=b.size)]
        draws[idx] = cohens_d(a_sample, b_sample)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return {"level": level, "replicates": int(replicates), "low": None, "high": None}
    alpha = (1.0 - float(level)) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    return {
        "level": float(level),
        "replicates": int(replicates),
        "low": float(low),
        "high": float(high),
    }


def bootstrap_d_ci_cluster(
    sample_a_records: Sequence[dict[str, Any]],
    sample_b_records: Sequence[dict[str, Any]],
    *,
    sequence_ids: Sequence[str],
    replicates: int,
    level: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    sequence_ids_tuple = tuple(str(sequence_id) for sequence_id in sequence_ids)
    if len(set(sequence_ids_tuple)) != len(sequence_ids_tuple):
        raise ValueError("sequence_ids must be unique for cluster bootstrap")
    if len(sequence_ids_tuple) < 2 or replicates <= 0:
        return {"level": level, "replicates": int(replicates), "low": None, "high": None}

    a_by_sequence: dict[str, list[float]] = {sequence_id: [] for sequence_id in sequence_ids_tuple}
    b_by_sequence: dict[str, list[float]] = {sequence_id: [] for sequence_id in sequence_ids_tuple}
    for label, records, grouped in (
        ("sample_a", sample_a_records, a_by_sequence),
        ("sample_b", sample_b_records, b_by_sequence),
    ):
        for record in records:
            sequence_id = str(record["sequence_id"])
            if sequence_id not in grouped:
                raise ValueError(f"{label} has sequence_id {sequence_id!r} outside the cluster fixture")
            grouped[sequence_id].append(float(record["distance"]))

    missing = [
        sequence_id
        for sequence_id in sequence_ids_tuple
        if len(a_by_sequence[sequence_id]) == 0 or len(b_by_sequence[sequence_id]) == 0
    ]
    if missing:
        raise ValueError(f"cluster bootstrap missing distances for sequence ids: {missing}")

    draws = np.empty(int(replicates), dtype=float)
    for idx in range(int(replicates)):
        draw_indices = rng.integers(0, len(sequence_ids_tuple), size=len(sequence_ids_tuple))
        a_sample = [
            value
            for draw_index in draw_indices
            for value in a_by_sequence[sequence_ids_tuple[int(draw_index)]]
        ]
        b_sample = [
            value
            for draw_index in draw_indices
            for value in b_by_sequence[sequence_ids_tuple[int(draw_index)]]
        ]
        draws[idx] = cohens_d(a_sample, b_sample)

    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return {"level": level, "replicates": int(replicates), "low": None, "high": None}
    alpha = (1.0 - float(level)) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    return {
        "level": float(level),
        "replicates": int(replicates),
        "low": float(low),
        "high": float(high),
    }


def compute_axis_metrics(
    *,
    axis: str,
    results: Sequence[dict[str, Any]],
    sequences: Sequence[SequenceFixture],
    init_seeds: Sequence[int],
    conditions: Sequence[float],
    pairs: Sequence[tuple[float, float]],
    length_m: float,
    bootstrap_replicates: int,
    bootstrap_level: float,
    rng_iid: np.random.Generator,
    rng_cluster: np.random.Generator,
) -> dict[str, Any]:
    axis_results = [item for item in results if item["axis"] == axis]
    by_key = {
        (item["sequence_id"], int(item["init_seed"]), condition_label(float(item["condition"]))): item
        for item in axis_results
    }

    within_by_condition: dict[str, list[dict[str, Any]]] = {condition_label(condition): [] for condition in conditions}
    for condition in conditions:
        cond_label = condition_label(condition)
        for sequence in sequences:
            for seed_a, seed_b in combinations(init_seeds, 2):
                a = by_key[(sequence.id, int(seed_a), cond_label)]
                b = by_key[(sequence.id, int(seed_b), cond_label)]
                distance = chamfer_distance(a["final_centerline"], b["final_centerline"], length_m)
                within_by_condition[cond_label].append(
                    {
                        "sequence_id": sequence.id,
                        "init_shape": sequence.init_shape,
                        "condition": condition,
                        "seed_pair": [int(seed_a), int(seed_b)],
                        "distance": float(distance),
                    }
                )

    pairwise: dict[str, Any] = {}
    sequence_ids = [sequence.id for sequence in sequences]
    iid_ci_key = bootstrap_ci_key(bootstrap_level, "iid")
    cluster_ci_key = bootstrap_ci_key(bootstrap_level, "cluster")
    for pair in pairs:
        a_condition, b_condition = pair
        a_label = condition_label(a_condition)
        b_label = condition_label(b_condition)
        between_records: list[dict[str, Any]] = []
        for sequence in sequences:
            for seed in init_seeds:
                a = by_key[(sequence.id, int(seed), a_label)]
                b = by_key[(sequence.id, int(seed), b_label)]
                distance = chamfer_distance(a["final_centerline"], b["final_centerline"], length_m)
                between_records.append(
                    {
                        "sequence_id": sequence.id,
                        "init_shape": sequence.init_shape,
                        "seed": int(seed),
                        "condition_pair": [float(a_condition), float(b_condition)],
                        "distance": float(distance),
                    }
                )
        within_records = within_by_condition[a_label] + within_by_condition[b_label]
        between_values = [record["distance"] for record in between_records]
        within_values = [record["distance"] for record in within_records]
        d_value = cohens_d(between_values, within_values)
        pairwise[pair_key(pair)] = {
            "condition_pair": [float(a_condition), float(b_condition)],
            "between_condition_distances": {
                "summary": describe_distribution(between_values),
                "values": between_records,
            },
            "within_condition_noise_floor": {
                "pooled_conditions": [float(a_condition), float(b_condition)],
                "summary": describe_distribution(within_values),
                "values": within_records,
            },
            "cohens_d": d_value,
            iid_ci_key: bootstrap_d_ci(
                between_values,
                within_values,
                replicates=bootstrap_replicates,
                level=bootstrap_level,
                rng=rng_iid,
            ),
            cluster_ci_key: bootstrap_d_ci_cluster(
                between_records,
                within_records,
                sequence_ids=sequence_ids,
                replicates=bootstrap_replicates,
                level=bootstrap_level,
                rng=rng_cluster,
            ),
        }

    return {
        "conditions": [float(condition) for condition in conditions],
        "within_condition_noise_floors": {
            label: {
                "summary": describe_distribution([record["distance"] for record in records]),
                "values": records,
            }
            for label, records in within_by_condition.items()
        },
        "pairwise": pairwise,
    }


def combinations(values: Sequence[int], n: int) -> Iterable[tuple[int, int]]:
    if n != 2:
        raise ValueError("only pair combinations are used")
    for i, left in enumerate(values):
        for right in values[i + 1 :]:
            yield int(left), int(right)


def compute_metrics(
    *,
    config: dict[str, Any],
    results: Sequence[dict[str, Any]],
    sequences: Sequence[SequenceFixture],
    init_seeds: Sequence[int],
    stiffness_conditions: Sequence[float],
    friction_conditions: Sequence[float],
    pairs: Sequence[tuple[float, float]],
    base_params: RopeParams,
    cli_seed: int,
) -> dict[str, Any]:
    measurement = measurement_config(config)
    bootstrap_replicates = int(measurement.get("bootstrap_replicates", 5000))
    bootstrap_level = float(measurement.get("bootstrap_ci", 0.95))
    rng_iid = np.random.default_rng(cli_seed + 60_000)
    rng_cluster = np.random.default_rng(cli_seed + 70_000)
    return {
        "stiffness": compute_axis_metrics(
            axis="stiffness",
            results=results,
            sequences=sequences,
            init_seeds=init_seeds,
            conditions=stiffness_conditions,
            pairs=pairs,
            length_m=float(base_params.length_m),
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_level=bootstrap_level,
            rng_iid=rng_iid,
            rng_cluster=rng_cluster,
        ),
        "friction": compute_axis_metrics(
            axis="friction",
            results=results,
            sequences=sequences,
            init_seeds=init_seeds,
            conditions=friction_conditions,
            pairs=pairs,
            length_m=float(base_params.length_m),
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_level=bootstrap_level,
            rng_iid=rng_iid,
            rng_cluster=rng_cluster,
        ),
    }


def format_ci_short(ci: dict[str, Any] | None) -> str:
    if not ci or ci.get("low") is None:
        return "n/a"
    return f"[{float(ci['low']):.3g}, {float(ci['high']):.3g}]"



def plot_axis_distributions(axis: str, axis_metrics: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pairs = list(axis_metrics["pairwise"].items())
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.2 * len(pairs), 4.2), constrained_layout=True)
    if len(pairs) == 1:
        axes = [axes]
    for ax, (key, metrics) in zip(axes, pairs, strict=True):
        between = np.asarray(
            [record["distance"] for record in metrics["between_condition_distances"]["values"]],
            dtype=float,
        )
        within = np.asarray(
            [record["distance"] for record in metrics["within_condition_noise_floor"]["values"]],
            dtype=float,
        )
        high = max(float(np.max(between)), float(np.max(within)), 1.0e-12)
        bins = np.linspace(0.0, high * 1.05, 22)
        ax.hist(within, bins=bins, alpha=0.62, label="within-seed floor", color="#4c78a8")
        ax.hist(between, bins=bins, alpha=0.62, label="between conditions", color="#f58518")
        iid_ci = metrics.get("bootstrap_ci_95_iid")
        cluster_ci = metrics.get("bootstrap_ci_95_cluster")
        d_value = metrics["cohens_d"]
        ax.set_title(
            f"{axis} {key}\nd={d_value:.3g}, iid {format_ci_short(iid_ci)}\n"
            f"cluster {format_ci_short(cluster_ci)}"
        )
        ax.set_xlabel("length-normalized Chamfer")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
    fig.suptitle(f"G1 {axis} distributions: between-condition distances vs within-condition floors")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def read_physics_quality_context(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"dm_stats_json": str(path), "available": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "dm_stats_json": str(path),
        "available": True,
        "physics_quality_note": data.get("physics_quality_note"),
        "rates": data.get("rates"),
        "convergence_rule": data.get("convergence_rule"),
        "record_count": data.get("record_count"),
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(as_jsonable(data), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def format_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.{digits}g}"

def format_ci(ci: dict[str, Any] | None) -> str:
    if not ci or ci.get("low") is None:
        return "n/a"
    return f"[{format_float(ci['low'])}, {format_float(ci['high'])}]"




def build_report(metrics_payload: dict[str, Any], plot_paths: dict[str, Path]) -> str:
    lines: list[str] = []
    lines.append("# G1 stiffness-validity pilot report")
    lines.append("")
    lines.append(f"- created_at: {metrics_payload['created_at']}")
    lines.append(f"- config: `{metrics_payload['config_path']}`")
    lines.append(f"- stdout log: `{metrics_payload['outputs']['stdout_log']}`")
    lines.append(f"- wall_time_s: {format_float(metrics_payload['wall_time_s'], 4)}")
    if metrics_payload.get("stats_recomputed_at"):
        lines.append(f"- stats_recomputed_at: {metrics_payload['stats_recomputed_at']}")
    if metrics_payload.get("stats_recomputed_at_commit"):
        lines.append(f"- stats_recomputed_at_commit: {metrics_payload['stats_recomputed_at_commit']}")
    lines.append(f"- batching: {metrics_payload['batching']['design']}")
    lines.append("- grasp realism: off for this controlled measurement; p/delta/lift fixture is fixed.")
    lines.append("")
    lines.append("## Fixture")
    design = metrics_payload["measurement_design"]
    lines.append(
        f"- sequences: {design['sequence_count']} fixed sequences "
        f"({', '.join(f'{shape}={count}' for shape, count in design['sequence_counts_by_shape'].items())})"
    )
    lines.append(f"- init seeds per sequence: {design['init_seeds']}")
    lines.append(f"- stiffness multipliers: {design['stiffness_multipliers']}")
    lines.append(f"- friction multipliers: {design['friction_multipliers']} (G1-subordinate reference)")
    first_pair = next(iter(metrics_payload["axes"]["stiffness"]["pairwise"].values()))
    bootstrap_reps = first_pair["bootstrap_ci_95_iid"]["replicates"]
    lines.append(
        "- Bootstrap CIs: i.i.d. resamples distance records; sequence-cluster resamples the "
        f"20 sequence ids with replacement and includes all distances for drawn sequences "
        f"({bootstrap_reps} reps each; rng seeds cli_seed+60000 and cli_seed+70000)."
    )
    lines.append(
        "- Negative d encodes between-condition distance below the within-condition noise floor "
        "in pooled-standard-deviation units."
    )
    lines.append("")

    for axis, title in (("stiffness", "Stiffness block"), ("friction", "Friction reference block")):
        axis_metrics = metrics_payload["axes"][axis]
        lines.append(f"## {title}")
        lines.append("")
        lines.append(
            "| pair | between mean | within-floor mean | d | i.i.d. bootstrap CI | "
            "sequence-cluster bootstrap CI | note |"
        )
        lines.append("| --- | ---: | ---: | ---: | --- | --- | --- |")
        for key, pair_metrics in axis_metrics["pairwise"].items():
            between = pair_metrics["between_condition_distances"]
            within = pair_metrics["within_condition_noise_floor"]
            between_summary = between["summary"]
            within_summary = within["summary"]
            iid_ci = pair_metrics["bootstrap_ci_95_iid"]
            cluster_ci = pair_metrics["bootstrap_ci_95_cluster"]
            lines.append(
                f"| {key} | {format_float(between_summary['mean'])} | "
                f"{format_float(within_summary['mean'])} | {format_float(pair_metrics['cohens_d'])} | "
                f"{format_ci(iid_ci)} | {format_ci(cluster_ci)} | "
                f"n_between={between_summary['n']}; n_within={within_summary['n']} |"
            )
        lines.append("")
        lines.append("Within-condition floors:")
        lines.append("")
        lines.append("| condition | n | mean | std | median |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for condition, floor in axis_metrics["within_condition_noise_floors"].items():
            summary = floor["summary"]
            lines.append(
                f"| {condition} | {summary['n']} | {format_float(summary['mean'])} | "
                f"{format_float(summary['std'])} | {format_float(summary['median'])} |"
            )
        lines.append("")

    lines.append("## Plots")
    lines.append("")
    lines.append(f"- stiffness distributions: `{plot_paths['stiffness']}`")
    lines.append(f"- friction distributions: `{plot_paths['friction']}`")
    lines.append("")
    lines.append("## This-run convergence")
    lines.append("")
    batch_summaries = metrics_payload.get("batching", {}).get("batch_summaries", [])
    reset_rates = [
        float(summary["reset_converged_rate"])
        for summary in batch_summaries
        if summary.get("reset_converged_rate") is not None
    ]
    primitive_rates = [
        float(step["converged_rate"])
        for summary in batch_summaries
        for step in summary.get("step_summaries", [])
        if step.get("converged_rate") is not None
    ]
    if reset_rates:
        if min(reset_rates) == max(reset_rates):
            lines.append(f"- reset_converged_rate: {reset_rates[0]:.1f} across {len(reset_rates)} batches.")
        else:
            lines.append(
                f"- reset_converged_rate: range=[{format_float(min(reset_rates))}, "
                f"{format_float(max(reset_rates))}], mean={format_float(np.mean(reset_rates))} "
                f"across {len(reset_rates)} batches."
            )
    else:
        lines.append("- reset_converged_rate: n/a.")
    if primitive_rates:
        lines.append(
            f"- per-primitive converged_rate: range=[{format_float(min(primitive_rates))}, "
            f"{format_float(max(primitive_rates))}], mean={format_float(np.mean(primitive_rates))} "
            f"across {len(primitive_rates)} primitive summaries."
        )
    else:
        lines.append("- per-primitive converged_rate: n/a.")
    lines.append(
        "- Unsettled final shapes add settle-noise to both between-condition and "
        "within-condition distance distributions."
    )
    lines.append("")
    lines.append("## Physics-quality context")
    lines.append("")
    context = metrics_payload.get("physics_quality_context", {})
    if context.get("available"):
        lines.append(f"- dm_stats: `{context['dm_stats_json']}`")
        rates = context.get("rates") or {}
        if rates:
            lines.append(
                "- dm_stats rates: "
                + ", ".join(f"{key}={format_float(value)}" for key, value in sorted(rates.items()))
            )
        if context.get("physics_quality_note"):
            lines.append(f"- dm_stats note: {context['physics_quality_note']}")
    else:
        lines.append(f"- dm_stats unavailable at `{context.get('dm_stats_json')}`")
    lines.append("")
    report = "\n".join(lines) + "\n"
    lowered = report.lower()
    banned = [term for term in REPORT_BANNED_TERMS if term.lower() in lowered]
    if banned:
        raise ValueError(f"report contains banned G1 verdict/recommendation terms: {banned}")
    return report


def summarize_sequence_counts(sequences: Sequence[SequenceFixture]) -> dict[str, int]:
    counts = {shape: 0 for shape in VALID_INIT_SHAPES}
    for sequence in sequences:
        counts[sequence.init_shape] += 1
    return counts

def recompute_axis_metrics_from_stored_distances(
    *,
    axis: str,
    stored_axis_metrics: dict[str, Any],
    sequences: Sequence[SequenceFixture],
    conditions: Sequence[float],
    pairs: Sequence[tuple[float, float]],
    bootstrap_replicates: int,
    bootstrap_level: float,
    rng_iid: np.random.Generator,
    rng_cluster: np.random.Generator,
) -> dict[str, Any]:
    pairwise: dict[str, Any] = {}
    sequence_ids = [sequence.id for sequence in sequences]
    iid_ci_key = bootstrap_ci_key(bootstrap_level, "iid")
    cluster_ci_key = bootstrap_ci_key(bootstrap_level, "cluster")

    stored_pairwise = stored_axis_metrics.get("pairwise", {})
    for pair in pairs:
        key = pair_key(pair)
        if key not in stored_pairwise:
            raise ValueError(f"stored {axis} metrics missing pair {key}")
        stored_pair = stored_pairwise[key]
        between_records = stored_pair["between_condition_distances"]["values"]
        within_records = stored_pair["within_condition_noise_floor"]["values"]
        between_values = [float(record["distance"]) for record in between_records]
        within_values = [float(record["distance"]) for record in within_records]
        a_condition, b_condition = pair
        pairwise[key] = {
            "condition_pair": [float(a_condition), float(b_condition)],
            "between_condition_distances": {
                "summary": describe_distribution(between_values),
                "values": between_records,
            },
            "within_condition_noise_floor": {
                "pooled_conditions": [float(a_condition), float(b_condition)],
                "summary": describe_distribution(within_values),
                "values": within_records,
            },
            "cohens_d": cohens_d(between_values, within_values),
            iid_ci_key: bootstrap_d_ci(
                between_values,
                within_values,
                replicates=bootstrap_replicates,
                level=bootstrap_level,
                rng=rng_iid,
            ),
            cluster_ci_key: bootstrap_d_ci_cluster(
                between_records,
                within_records,
                sequence_ids=sequence_ids,
                replicates=bootstrap_replicates,
                level=bootstrap_level,
                rng=rng_cluster,
            ),
        }

    stored_floors = stored_axis_metrics.get("within_condition_noise_floors", {})
    within_condition_noise_floors: dict[str, Any] = {}
    for condition in conditions:
        label = condition_label(condition)
        if label not in stored_floors:
            raise ValueError(f"stored {axis} metrics missing within-condition floor {label}")
        records = stored_floors[label]["values"]
        values = [float(record["distance"]) for record in records]
        within_condition_noise_floors[label] = {
            "summary": describe_distribution(values),
            "values": records,
        }

    return {
        "conditions": [float(condition) for condition in conditions],
        "within_condition_noise_floors": within_condition_noise_floors,
        "pairwise": pairwise,
    }


def recompute_metrics_from_stored_distances(
    *,
    config: dict[str, Any],
    stored_payload: dict[str, Any],
    sequences: Sequence[SequenceFixture],
    stiffness_conditions: Sequence[float],
    friction_conditions: Sequence[float],
    pairs: Sequence[tuple[float, float]],
    cli_seed: int,
) -> dict[str, Any]:
    measurement = measurement_config(config)
    bootstrap_replicates = int(measurement.get("bootstrap_replicates", 5000))
    bootstrap_level = float(measurement.get("bootstrap_ci", 0.95))
    rng_iid = np.random.default_rng(cli_seed + 60_000)
    rng_cluster = np.random.default_rng(cli_seed + 70_000)
    stored_axes = stored_payload["axes"]
    return {
        "stiffness": recompute_axis_metrics_from_stored_distances(
            axis="stiffness",
            stored_axis_metrics=stored_axes["stiffness"],
            sequences=sequences,
            conditions=stiffness_conditions,
            pairs=pairs,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_level=bootstrap_level,
            rng_iid=rng_iid,
            rng_cluster=rng_cluster,
        ),
        "friction": recompute_axis_metrics_from_stored_distances(
            axis="friction",
            stored_axis_metrics=stored_axes["friction"],
            sequences=sequences,
            conditions=friction_conditions,
            pairs=pairs,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_level=bootstrap_level,
            rng_iid=rng_iid,
            rng_cluster=rng_cluster,
        ),
    }


def stats_only_result_payload(
    *,
    config: dict[str, Any],
    config_text: str,
    config_path: Path,
    cli_seed: int | None,
) -> dict[str, Any]:
    paths = output_paths(config)
    if not paths["metrics_json"].exists():
        raise FileNotFoundError(f"stats-only requires existing metrics JSON at {paths['metrics_json']}")
    stored_payload = json.loads(paths["metrics_json"].read_text(encoding="utf-8"))
    stats_seed = int(stored_payload.get("cli_seed", 0) if cli_seed is None else cli_seed)

    base_params = params_from_config(config)
    sequences = parse_sequences(config, n_vertices=base_params.n_segments)
    init_seeds, stiffness_conditions, friction_conditions, pairs = measurement_lists(config)
    axes_metrics = recompute_metrics_from_stored_distances(
        config=config,
        stored_payload=stored_payload,
        sequences=sequences,
        stiffness_conditions=stiffness_conditions,
        friction_conditions=friction_conditions,
        pairs=pairs,
        cli_seed=stats_seed,
    )
    physics_context = read_physics_quality_context(paths["dm_stats_json"])

    plot_axis_distributions("stiffness", axes_metrics["stiffness"], paths["stiffness_plot_png"])
    plot_axis_distributions("friction", axes_metrics["friction"], paths["friction_plot_png"])

    metrics_payload = dict(stored_payload)
    metrics_payload.update(
        {
            "config_path": str(config_path),
            "config_sha256": __import__("hashlib").sha256(config_text.encode("utf-8")).hexdigest(),
            "cli_seed": int(stats_seed),
            "base_rope_params": asdict(base_params),
            "stiffness_bases": stiffness_bases(),
            "axes": axes_metrics,
            "outputs": {
                "metrics_json": str(paths["metrics_json"]),
                "stiffness_plot_png": str(paths["stiffness_plot_png"]),
                "friction_plot_png": str(paths["friction_plot_png"]),
                "report_md": str(paths["report_md"]),
                "stdout_log": str(paths["stdout_log"]),
            },
            "physics_quality_context": physics_context,
            "stats_recomputed_at": utc_now(),
            "stats_recomputed_at_commit": get_git_commit_hash(Path.cwd()),
        }
    )
    metrics_payload["measurement_design"] = {
        **metrics_payload.get("measurement_design", {}),
        "sequence_count": len(sequences),
        "sequence_counts_by_shape": summarize_sequence_counts(sequences),
        "fixture_generation_seed": measurement_config(config).get("fixture_generation_seed"),
        "init_seeds": [int(seed) for seed in init_seeds],
        "stiffness_multipliers": [float(value) for value in stiffness_conditions],
        "friction_multipliers": [float(value) for value in friction_conditions],
        "condition_pairs": [[float(a), float(b)] for a, b in pairs],
        "grasp_realism": False,
        "grasp_realism_choice": "off: controlled stiffness/friction effect-size measurement avoids grasp noise confounding",
        "runs_per_axis": len(sequences) * len(init_seeds) * len(stiffness_conditions),
    }

    write_json(paths["metrics_json"], metrics_payload)
    report = build_report(
        metrics_payload,
        {"stiffness": paths["stiffness_plot_png"], "friction": paths["friction_plot_png"]},
    )
    paths["report_md"].parent.mkdir(parents=True, exist_ok=True)
    paths["report_md"].write_text(report, encoding="utf-8")
    print(f"stats_only_source {paths['metrics_json']}")
    print(f"wrote metrics {paths['metrics_json']}")
    print(f"wrote report {paths['report_md']}")
    print(f"wrote plots {paths['stiffness_plot_png']} {paths['friction_plot_png']}")
    return metrics_payload




def final_result_payload(
    *,
    config: dict[str, Any],
    config_text: str,
    config_path: Path,
    cli_seed: int,
) -> dict[str, Any]:
    paths = output_paths(config)
    base_params = params_from_config(config)
    sequences = parse_sequences(config, n_vertices=base_params.n_segments)
    sequences_by_id = {sequence.id: sequence for sequence in sequences}
    init_seeds, stiffness_conditions, friction_conditions, pairs = measurement_lists(config)
    sim_cfg = config.get("sim", {})
    if bool(sim_cfg.get("grasp_realism", False)):
        raise ValueError("G1 controlled measurement requires sim.grasp_realism: false")

    start = time.perf_counter()
    print(f"g1_start created_at={utc_now()} seed={cli_seed} config={config_path}")
    print(f"fixture sequences={len(sequences)} init_seeds={init_seeds} generation_seed={measurement_config(config).get('fixture_generation_seed')}")
    print(f"setter_signature_source external/DLO-Lab/genesis/engine/entities/rod_entity.py:1288-1423")
    probe = run_timing_probe(
        config=config,
        base_params=base_params,
        sequences_by_id=sequences_by_id,
        init_seeds=init_seeds,
        cli_seed=cli_seed,
    )
    mixed_conditions = bool(probe.get("batch_support", {}).get("per_env_param_setters", True))
    if not mixed_conditions:
        print("batching fallback: per-env param setter envs_idx support not found; batching by homogeneous condition")
    results, batch_summaries = run_full_measurement(
        config=config,
        base_params=base_params,
        sequences=sequences,
        init_seeds=init_seeds,
        stiffness_conditions=stiffness_conditions,
        friction_conditions=friction_conditions,
        cli_seed=cli_seed,
        mixed_conditions=mixed_conditions,
    )
    wall_time_s = time.perf_counter() - start

    axes_metrics = compute_metrics(
        config=config,
        results=results,
        sequences=sequences,
        init_seeds=init_seeds,
        stiffness_conditions=stiffness_conditions,
        friction_conditions=friction_conditions,
        pairs=pairs,
        base_params=base_params,
        cli_seed=cli_seed,
    )
    physics_context = read_physics_quality_context(paths["dm_stats_json"])

    plot_axis_distributions("stiffness", axes_metrics["stiffness"], paths["stiffness_plot_png"])
    plot_axis_distributions("friction", axes_metrics["friction"], paths["friction_plot_png"])

    support = probe.get("batch_support", {})
    max_batch_envs = max((summary.get("n_envs", 0) for summary in batch_summaries), default=0)
    metrics_payload = {
        "schema_version": 1,
        "gate": "G1",
        "created_at": utc_now(),
        "config_path": str(config_path),
        "config_sha256": __import__("hashlib").sha256(config_text.encode("utf-8")).hexdigest(),
        "cli_seed": int(cli_seed),
        "commit_hash": get_git_commit_hash(Path.cwd()),
        "base_rope_params": asdict(base_params),
        "stiffness_bases": stiffness_bases(),
        "measurement_design": {
            "sequence_count": len(sequences),
            "sequence_counts_by_shape": summarize_sequence_counts(sequences),
            "fixture_generation_seed": measurement_config(config).get("fixture_generation_seed"),
            "init_seeds": [int(seed) for seed in init_seeds],
            "stiffness_multipliers": [float(value) for value in stiffness_conditions],
            "friction_multipliers": [float(value) for value in friction_conditions],
            "condition_pairs": [[float(a), float(b)] for a, b in pairs],
            "grasp_realism": False,
            "grasp_realism_choice": "off: controlled stiffness/friction effect-size measurement avoids grasp noise confounding",
            "runs_per_axis": len(sequences) * len(init_seeds) * len(stiffness_conditions),
        },
        "batching": {
            "design": (
                "per-env DLO-Lab parameter setters with envs_idx support; mixed 3-condition batches grouped by sequence length"
                if mixed_conditions
                else "homogeneous-condition batches grouped by sequence length"
            ),
            "per_env_stiffness_support_found": bool(support.get("per_env_param_setters", mixed_conditions)),
            "per_env_grasp_support_found": bool(support.get("per_env_grasp", True)),
            "setter_signatures": support.get("setter_signatures", {}),
            "setter_source": support.get("source"),
            "max_batch_envs": int(max_batch_envs),
            "batch_count": len(batch_summaries),
            "batch_summaries": batch_summaries,
        },
        "timing_probe": probe,
        "wall_time_s": wall_time_s,
        "axes": axes_metrics,
        "outputs": {
            "metrics_json": str(paths["metrics_json"]),
            "stiffness_plot_png": str(paths["stiffness_plot_png"]),
            "friction_plot_png": str(paths["friction_plot_png"]),
            "report_md": str(paths["report_md"]),
            "stdout_log": str(paths["stdout_log"]),
        },
        "physics_quality_context": physics_context,
    }

    write_json(paths["metrics_json"], metrics_payload)
    report = build_report(
        metrics_payload,
        {"stiffness": paths["stiffness_plot_png"], "friction": paths["friction_plot_png"]},
    )
    paths["report_md"].parent.mkdir(parents=True, exist_ok=True)
    paths["report_md"].write_text(report, encoding="utf-8")
    print(f"wrote metrics {paths['metrics_json']}")
    print(f"wrote report {paths['report_md']}")
    print(f"wrote plots {paths['stiffness_plot_png']} {paths['friction_plot_png']}")
    print(f"g1_done wall_time_s={wall_time_s:.3f}")
    return metrics_payload


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    config, config_text = load_config(config_path)
    paths = output_paths(config)
    simulation_seed = 0 if args.seed is None else int(args.seed)
    with tee_stdout(paths["stdout_log"]):
        if args.stats_only:
            stats_only_result_payload(
                config=config,
                config_text=config_text,
                config_path=config_path,
                cli_seed=None if args.seed is None else int(args.seed),
            )
        else:
            final_result_payload(
                config=config,
                config_text=config_text,
                config_path=config_path,
                cli_seed=simulation_seed,
            )


if __name__ == "__main__":
    main()
