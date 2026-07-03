"""M7 appendix: same-initial-state repeat-execution variance.

This report-only measurement is sanctioned by the M6 issue-#7 verdict for M7
numeric-fixing.  It isolates repeat execution noise by running four fixed
(init_state, action-sequence) cells, 16 repeats per cell, in one 64-env GPU
batch.  Each cell starts from a shared settled native-vertex state; grasp
realism is measured both ON (execution noise included) and OFF (solver-only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dgcc.envs.dlolab import DLOLabEnv, analytic_init_centerline, centerline_arc_length
from dgcc.goals.distance import chamfer_distance
from dgcc.utils.meta import get_git_commit_hash
from gate_g1 import (
    Tee,
    as_jsonable,
    cleanup_env,
    describe_distribution,
    env_kwargs,
    load_config,
    params_from_config,
    parse_sequences,
    write_json,
)

os.environ.pop("DISPLAY", None)

CELL_SEQUENCE_IDS = {
    "straight": "straight_01",
    "u_bend": "u_bend_01",
    "s_curve": "s_curve_01",
    "random_smooth": "random_smooth_01",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the M7 appendix repeat-variance measurement")
    parser.add_argument("--config", type=Path, default=Path("configs/gate_g1.yaml"))
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--repeats-per-cell", type=int, default=16)
    parser.add_argument("--metrics", type=Path, default=Path("outputs/metrics/repeat_variance.json"))
    parser.add_argument("--plot", type=Path, default=Path("outputs/plots/repeat_variance.png"))
    parser.add_argument("--log", type=Path, default=Path("outputs/reports/appendix_repeat_variance.log"))
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


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_cells(sequences: Sequence[Any]) -> list[Any]:
    by_id = {sequence.id: sequence for sequence in sequences}
    cells = []
    for init_shape, sequence_id in CELL_SEQUENCE_IDS.items():
        if sequence_id not in by_id:
            raise ValueError(f"missing required sequence {sequence_id!r} for {init_shape}")
        sequence = by_id[sequence_id]
        if sequence.init_shape != init_shape:
            raise ValueError(f"sequence {sequence_id!r} init_shape={sequence.init_shape!r}, expected {init_shape!r}")
        if len(sequence.primitives) != 2:
            raise ValueError(f"repeat-variance cells require two-primitive sequences, got {sequence_id}")
        cells.append(sequence)
    return cells


def pairwise_chamfer_values(centerlines: np.ndarray, *, length_m: float) -> list[float]:
    values: list[float] = []
    for left in range(centerlines.shape[0]):
        for right in range(left + 1, centerlines.shape[0]):
            values.append(float(chamfer_distance(centerlines[left], centerlines[right], length_m)))
    return values


def make_initial_vertices(base_params: Any, cells: Sequence[Any], repeats_per_cell: int, seed: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    vertices: list[np.ndarray] = []
    cell_metadata: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(cells):
        init_seed = int(seed + 100 + cell_index)
        curve = analytic_init_centerline(base_params, cell.init_shape, init_seed)
        cell_metadata.append(
            {
                "cell_index": cell_index,
                "cell_id": cell.id,
                "init_shape": cell.init_shape,
                "init_seed": init_seed,
                "initial_arc_length_m": centerline_arc_length(curve),
                "primitive_count": len(cell.primitives),
                "primitives": [
                    {"p": primitive.p, "delta": list(primitive.delta), "lift": primitive.lift}
                    for primitive in cell.primitives
                ],
                "repeat_env_indices": list(range(cell_index * repeats_per_cell, (cell_index + 1) * repeats_per_cell)),
            }
        )
        vertices.extend([curve.copy() for _ in range(repeats_per_cell)])
    return np.stack(vertices, axis=0), cell_metadata


def canonicalize_cell_start_states(env: DLOLabEnv, *, repeats_per_cell: int, n_cells: int) -> dict[str, Any]:
    raw = env.get_centerline_raw_batch()
    canonical = raw.copy()
    pre_pairwise_max: list[float] = []
    for cell_index in range(n_cells):
        start = cell_index * repeats_per_cell
        stop = start + repeats_per_cell
        representative = raw[start].copy()
        pre_pairwise = []
        for env_idx in range(start + 1, stop):
            pre_pairwise.append(float(np.max(np.linalg.norm(raw[env_idx] - representative, axis=1))))
        pre_pairwise_max.append(float(max(pre_pairwise)) if pre_pairwise else 0.0)
        canonical[start:stop] = representative
    env.place_rod_vertices_batch(canonical)
    return {
        "method": "after light_reset, the first settled raw-vertex state in each cell is broadcast to all 16 repeats before the action sequence",
        "pre_broadcast_max_vertex_delta_m_by_cell": pre_pairwise_max,
    }


def run_condition(
    *,
    config: dict[str, Any],
    base_params: Any,
    cells: Sequence[Any],
    repeats_per_cell: int,
    seed: int,
    grasp_realism: bool,
) -> dict[str, Any]:
    n_cells = len(cells)
    n_envs = n_cells * repeats_per_cell
    measurement = config.get("measurement", {})
    vel_threshold = float(measurement.get("vel_threshold", 1.0e-3))
    settle_max_steps = int(measurement.get("settle_max_steps", 5000))
    env: DLOLabEnv | None = None
    wall_start = time.perf_counter()
    try:
        kwargs = env_kwargs(config, n_envs)
        kwargs["grasp_realism"] = bool(grasp_realism)
        env = DLOLabEnv(**kwargs)
        build_start = time.perf_counter()
        reset_info = env.reset(base_params, init_shape="straight", seed=seed + (10_000 if grasp_realism else 20_000))
        build_wall_s = time.perf_counter() - build_start
        if not env.supports_per_env_grasp():
            raise RuntimeError("DLO-Lab per-env grasp hooks are unavailable")

        vertices, cell_metadata = make_initial_vertices(base_params, cells, repeats_per_cell, seed)
        reset_start = time.perf_counter()
        reset_result = env.light_reset(vertices, vel_threshold=vel_threshold, max_steps=settle_max_steps)
        canonicalization = canonicalize_cell_start_states(env, repeats_per_cell=repeats_per_cell, n_cells=n_cells)
        reset_wall_s = time.perf_counter() - reset_start

        primitive_summaries: list[dict[str, Any]] = []
        for primitive_index in range(2):
            p: list[int] = []
            delta: list[tuple[float, float, float]] = []
            lifts: list[str] = []
            for cell in cells:
                primitive = cell.primitives[primitive_index]
                p.extend([primitive.p] * repeats_per_cell)
                delta.extend([primitive.delta] * repeats_per_cell)
                lifts.extend([primitive.lift] * repeats_per_cell)
            step_start = time.perf_counter()
            result = env.step_primitive_batch(
                np.asarray(p, dtype=int),
                np.asarray(delta, dtype=float),
                lifts,
                vel_threshold=vel_threshold,
                max_steps=settle_max_steps,
                rng=np.random.default_rng(seed + (30_000 if grasp_realism else 40_000) + primitive_index),
            )
            step_wall_s = time.perf_counter() - step_start
            settle_steps = np.asarray(result["settle_steps"], dtype=int)
            settle_converged = np.asarray(result["info"]["settle_converged"], dtype=bool)
            grasp_success = np.asarray(result["grasp_success"], dtype=bool)
            p_actual = np.asarray(result["info"]["p_actual"], dtype=int)
            per_cell = []
            for cell_index, cell in enumerate(cells):
                start = cell_index * repeats_per_cell
                stop = start + repeats_per_cell
                per_cell.append(
                    {
                        "cell_id": cell.id,
                        "grasp_success_rate": float(np.mean(grasp_success[start:stop])),
                        "settle_converged_rate": float(np.mean(settle_converged[start:stop])),
                        "settle_steps": describe_distribution(settle_steps[start:stop]),
                        "actual_grasp_nodes": p_actual[start:stop].tolist(),
                    }
                )
            summary = {
                "primitive_index": primitive_index,
                "wall_s": step_wall_s,
                "grasp_success_rate": float(np.mean(grasp_success)),
                "settle_converged_rate": float(np.mean(settle_converged)),
                "settle_steps": describe_distribution(settle_steps),
                "per_cell": per_cell,
            }
            primitive_summaries.append(summary)
            print(
                "repeat_step "
                f"realism={'on' if grasp_realism else 'off'} primitive={primitive_index + 1}/2 "
                f"wall_s={step_wall_s:.3f} grasp_success_rate={summary['grasp_success_rate']:.3f} "
                f"settle_converged_rate={summary['settle_converged_rate']:.3f}"
            )

        final_centerlines = env.get_centerline_batch()
        final_speeds = env.max_node_speed_batch()
        cell_results: list[dict[str, Any]] = []
        all_pairwise: list[float] = []
        for cell_index, cell in enumerate(cells):
            start = cell_index * repeats_per_cell
            stop = start + repeats_per_cell
            values = pairwise_chamfer_values(final_centerlines[start:stop], length_m=base_params.length_m)
            all_pairwise.extend(values)
            cell_results.append(
                {
                    **cell_metadata[cell_index],
                    "pairwise_chamfer_length_normalized": {
                        "values": values,
                        "stats": describe_distribution(values),
                    },
                    "final_max_node_speed": describe_distribution(final_speeds[start:stop]),
                    "reset_settle_converged_rate": float(np.mean(np.asarray(reset_result["settle_converged"], dtype=bool)[start:stop])),
                    "reset_settle_steps": describe_distribution(np.asarray(reset_result["settle_steps"], dtype=int)[start:stop]),
                }
            )

        wall_s = time.perf_counter() - wall_start
        print(
            "repeat_condition_done "
            f"realism={'on' if grasp_realism else 'off'} n_envs={n_envs} wall_s={wall_s:.3f} "
            f"overall_pairwise_mean={np.mean(all_pairwise) if all_pairwise else 0.0:.6g}"
        )
        return {
            "grasp_realism": bool(grasp_realism),
            "n_envs": n_envs,
            "repeats_per_cell": repeats_per_cell,
            "vel_threshold": vel_threshold,
            "settle_max_steps": settle_max_steps,
            "wall_s": wall_s,
            "build_wall_s": build_wall_s,
            "initial_reset_wall_s": reset_wall_s,
            "reset_info": reset_info,
            "canonicalization": canonicalization,
            "primitive_summaries": primitive_summaries,
            "overall_pairwise_chamfer_length_normalized": {
                "values": all_pairwise,
                "stats": describe_distribution(all_pairwise),
            },
            "cells": cell_results,
        }
    finally:
        cleanup_env(env)


def plot_results(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blocks = [("realism ON", payload["realism_on"]), ("realism OFF", payload["realism_off"])]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    labels: list[str] = []
    data: list[list[float]] = []
    colors: list[str] = []
    for block_label, block in blocks:
        for cell in block["cells"]:
            labels.append(f"{cell['init_shape']}\n{block_label.split()[-1]}")
            data.append(cell["pairwise_chamfer_length_normalized"]["values"])
            colors.append("#d95f02" if block_label.endswith("ON") else "#1b9e77")

    box = axes[0].boxplot(data, patch_artist=True, showfliers=True)
    axes[0].set_xticks(np.arange(1, len(labels) + 1), labels)
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    axes[0].set_ylabel("Pairwise length-normalized Chamfer")
    axes[0].set_title("Same-initial-state final-shape spread")
    axes[0].tick_params(axis="x", labelrotation=35)
    axes[0].grid(axis="y", alpha=0.3)

    means_on = [cell["pairwise_chamfer_length_normalized"]["stats"].get("mean", 0.0) for cell in payload["realism_on"]["cells"]]
    means_off = [cell["pairwise_chamfer_length_normalized"]["stats"].get("mean", 0.0) for cell in payload["realism_off"]["cells"]]
    x = np.arange(len(means_on))
    width = 0.36
    axes[1].bar(x - width / 2, means_on, width, label="realism ON", color="#d95f02", alpha=0.75)
    axes[1].bar(x + width / 2, means_off, width, label="realism OFF", color="#1b9e77", alpha=0.75)
    axes[1].set_xticks(x, [cell["init_shape"] for cell in payload["realism_on"]["cells"]], rotation=25, ha="right")
    axes[1].set_ylabel("Mean pairwise length-normalized Chamfer")
    axes[1].set_title("Per-cell means")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()
    fig.suptitle("M7 appendix repeat-execution variance (4 cells × 16 repeats)")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    with tee_stdout(args.log):
        total_start = time.perf_counter()
        config, config_text = load_config(args.config)
        base_params = params_from_config(config)
        sequences = parse_sequences(config, n_vertices=base_params.n_segments)
        cells = select_cells(sequences)
        if args.repeats_per_cell != 16:
            raise ValueError("assignment requires 16 repeats per cell")
        print(f"appendix_repeat_variance_start seed={args.seed} config={args.config}")
        print("method=four fixed G1 two-primitive cells, 16 same-start repeats per cell, realism on/off")
        realism_on = run_condition(
            config=config,
            base_params=base_params,
            cells=cells,
            repeats_per_cell=args.repeats_per_cell,
            seed=args.seed,
            grasp_realism=True,
        )
        realism_off = run_condition(
            config=config,
            base_params=base_params,
            cells=cells,
            repeats_per_cell=args.repeats_per_cell,
            seed=args.seed + 1,
            grasp_realism=False,
        )
        payload = {
            "schema_version": 1,
            "measurement": "M7 appendix repeat-execution variance",
            "status": "report_only",
            "sanction": "M6 issue-#7 verdict instruction #2; M7 numeric-fixing appendix",
            "created_utc": utc_now(),
            "commit_hash": get_git_commit_hash(),
            "config_path": str(args.config),
            "config_sha256": sha256_text(config_text),
            "config_yaml": config_text,
            "seed": int(args.seed),
            "rope_params": as_jsonable(base_params.__dict__),
            "design": {
                "cells": CELL_SEQUENCE_IDS,
                "repeats_per_cell": int(args.repeats_per_cell),
                "n_envs_per_condition": len(cells) * args.repeats_per_cell,
                "distance_metric": "length-normalized bidirectional Chamfer among final 32-point centerlines",
                "threshold_unchanged": 1.0e-3,
                "grasp_realism_on_block": "execution noise object of measurement: ±1 node jitter plus 5% failure model",
                "grasp_realism_off_block": "solver/settle-only repeat spread; deterministic simulation may report zero",
            },
            "realism_on": realism_on,
            "realism_off": realism_off,
            "wall_time_s": time.perf_counter() - total_start,
            "outputs": {
                "metrics_json": str(args.metrics),
                "plot_png": str(args.plot),
                "stdout_log": str(args.log),
            },
        }
        write_json(args.metrics, payload)
        plot_results(payload, args.plot)
        print(f"wrote_metrics {args.metrics}")
        print(f"wrote_plot {args.plot}")
        print(f"appendix_repeat_variance_done wall_s={payload['wall_time_s']:.3f}")


if __name__ == "__main__":
    main()
