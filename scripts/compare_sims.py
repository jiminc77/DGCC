"""Run the P0-M1 simulator comparison scenario on MuJoCo and DLO-Lab."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.pop("DISPLAY", None)
warnings.filterwarnings("ignore", message="cannot create weak reference.*")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from dgcc.envs.base import RopeParams
from dgcc.envs.dlolab import DLOLabEnv
from dgcc.envs.mujoco_cable import MuJoCoCableEnv
from dgcc.utils.meta import get_git_commit_hash


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
    parser = argparse.ArgumentParser(description="Run the P0-M1 two-simulator comparison")
    parser.add_argument("--seed", type=int, default=0, help="base deterministic seed; offsets come from config")
    parser.add_argument("--config", default="configs/compare_sims.yaml", help="YAML comparison config path")
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
        friction=float(rope.get("friction", 1.0)),
        radius=float(rope.get("radius", 0.005)),
    )


def dlolab_env_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    sim_cfg = config.get("simulators", {}).get("dlolab", {}).get("env", {})
    return {
        "n_envs": 1,
        "dt": float(sim_cfg.get("dt", 1.0e-3)),
        "substeps": int(sim_cfg.get("substeps", 5)),
        "rod_damping": float(sim_cfg.get("rod_damping", 10.0)),
        "rod_angular_damping": float(sim_cfg.get("rod_angular_damping", 5.0)),
        "initial_settle_steps": int(sim_cfg.get("initial_settle_steps", 20)),
        "reset_settle_max_steps": int(sim_cfg.get("reset_settle_max_steps", 1000)),
        "move_step_size": float(sim_cfg.get("move_step_size", 0.002)),
        "move_hold_steps": int(sim_cfg.get("move_hold_steps", 20)),
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def straightness(centerline: np.ndarray, length_m: float) -> float:
    X = np.asarray(centerline, dtype=float)
    return float(np.linalg.norm(X[-1] - X[0]) / float(length_m))


def no_nan(centerline: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(np.asarray(centerline, dtype=float))))


def max_velocity_metric(env: Any) -> tuple[str, float]:
    if hasattr(env, "max_abs_qvel"):
        return "max_abs_qvel", float(env.max_abs_qvel())
    if hasattr(env, "max_node_speed"):
        return "max_node_speed", float(env.max_node_speed())
    raise TypeError(
        f"{type(env).__name__} exposes neither max_abs_qvel nor max_node_speed; "
        "adapters must provide a settle velocity metric"
    )


def make_env(sim_name: str, config: dict[str, Any]) -> Any:
    if sim_name == "mujoco":
        return MuJoCoCableEnv()
    if sim_name == "dlolab":
        return DLOLabEnv(**dlolab_env_kwargs(config))
    raise ValueError(f"unknown sim {sim_name!r}")


def execute_primitive(
    env: Any,
    primitive: dict[str, Any],
    *,
    length_m: float,
    settle_threshold: float,
    settle_max_steps: int,
) -> dict[str, Any]:
    p = int(primitive["p"])
    delta = np.array(primitive["delta"], dtype=float)
    lift = str(primitive["lift"])

    X_before = np.asarray(env.get_centerline(), dtype=float)
    before_straightness = straightness(X_before, length_m)
    before_no_nan = no_nan(X_before)

    start = time.perf_counter()
    grasp_success = bool(env.grasp(p))
    target = env.move(delta, lift)
    settle_converged = bool(env.release(vel_threshold=settle_threshold, max_steps=settle_max_steps))
    wall_time_s = time.perf_counter() - start

    X_after = np.asarray(env.get_centerline(), dtype=float)
    after_straightness = straightness(X_after, length_m)
    after_no_nan = no_nan(X_after)
    velocity_name, velocity_value = max_velocity_metric(env)

    return {
        "p": p,
        "delta_requested": delta.tolist(),
        "delta_clamped": np.asarray(getattr(env, "last_delta_clamped", delta), dtype=float).tolist(),
        "lift": lift,
        "wall_time_s": float(wall_time_s),
        "settle_steps": int(getattr(env, "last_settle_steps", 0)),
        "settle_converged": settle_converged,
        "grasp_success": grasp_success,
        "straightness_before": before_straightness,
        "straightness_after": after_straightness,
        "straightness_improvement": float(after_straightness - before_straightness),
        "no_nan_before": before_no_nan,
        "no_nan_after": after_no_nan,
        "no_nan": bool(before_no_nan and after_no_nan),
        "target": jsonable(target),
        velocity_name: velocity_value,
    }


def run_sequence(
    *,
    sim_name: str,
    sequence: dict[str, Any],
    seed: int,
    params: RopeParams,
    config: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    thresholds = config.get("thresholds", {})
    settle_threshold = float(thresholds.get("settle_vel", 1.0e-3))
    settle_max_steps = int(thresholds.get("settle_max_steps", 5000))
    sim_cfg = config.get("simulators", {}).get(sim_name, {})
    init_shape = str(sim_cfg.get("init_shape", config.get("init_shape", "u_bend")))

    env = make_env(sim_name, config)
    reset_start = time.perf_counter()
    reset_info = env.reset(params, init_shape=init_shape, seed=seed)
    reset_wall_time_s = time.perf_counter() - reset_start
    initial_centerline = np.asarray(env.get_centerline(), dtype=float)

    primitive_records: list[dict[str, Any]] = []
    for primitive in sequence.get("primitives", []):
        primitive_records.append(
            execute_primitive(
                env,
                primitive,
                length_m=params.length_m,
                settle_threshold=settle_threshold,
                settle_max_steps=settle_max_steps,
            )
        )
    final_centerline = np.asarray(env.get_centerline(), dtype=float)

    record = {
        "sim": sim_name,
        "sequence_id": str(sequence["id"]),
        "sequence_description": str(sequence.get("description", "")),
        "seed": int(seed),
        "init_shape": init_shape,
        "reset_wall_time_s": float(reset_wall_time_s),
        "reset_info": jsonable(reset_info),
        "straightness_initial": straightness(initial_centerline, params.length_m),
        "straightness_final": straightness(final_centerline, params.length_m),
        "straightness_sequence_improvement": float(
            straightness(final_centerline, params.length_m) - straightness(initial_centerline, params.length_m)
        ),
        "no_nan_initial": no_nan(initial_centerline),
        "no_nan_final": no_nan(final_centerline),
        "primitive_records": primitive_records,
    }
    return record, initial_centerline, final_centerline


def stat_block(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(arr)), "min": float(np.min(arr)), "max": float(np.max(arr))}


def aggregate_records(run_records: list[dict[str, Any]]) -> dict[str, Any]:
    primitives = [primitive for run in run_records for primitive in run["primitive_records"]]
    wall_times = [float(p["wall_time_s"]) for p in primitives]
    settle_steps = [float(p["settle_steps"]) for p in primitives]
    primitive_improvements = [float(p["straightness_improvement"]) for p in primitives]
    sequence_improvements = [float(run["straightness_sequence_improvement"]) for run in run_records]
    converged = [bool(p["settle_converged"]) for p in primitives]
    no_nan_values = [bool(p["no_nan"]) for p in primitives]
    primitive_count = len(primitives)
    run_count = len(run_records)
    converged_count = int(sum(converged))
    return {
        "run_count": run_count,
        "primitive_count": primitive_count,
        "wall_time_s": stat_block(wall_times),
        "settle_steps": stat_block(settle_steps),
        "settle_converged_count": converged_count,
        "settle_nonconverged_count": int(primitive_count - converged_count),
        "settle_convergence_rate": float(converged_count / primitive_count) if primitive_count else None,
        "nan_check_pass_count": int(sum(no_nan_values)),
        "nan_check_fail_count": int(primitive_count - sum(no_nan_values)),
        "straightness_improvement_per_primitive": stat_block(primitive_improvements),
        "straightness_improvement_per_sequence": stat_block(sequence_improvements),
    }


def write_centerline_plot(
    *,
    path: Path,
    sim_name: str,
    sequence_id: str,
    seed: int,
    before: np.ndarray,
    after: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), constrained_layout=True)
    views = ((0, 1, "top-down x/y", "x [m]", "y [m]"), (0, 2, "side x/z", "x [m]", "z [m]"))
    for ax, (i, j, title, xlabel, ylabel) in zip(axes, views, strict=True):
        ax.plot(before[:, i], before[:, j], "o--", markersize=2.5, linewidth=1.2, label="before")
        ax.plot(after[:, i], after[:, j], "o-", markersize=2.5, linewidth=1.2, label="after")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.axis("equal")
    axes[0].legend(loc="best")
    fig.suptitle(f"{sim_name} {sequence_id} seed={seed}")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def rel_link(path: Path, *, start: Path) -> str:
    return os.path.relpath(path, start=start).replace(os.sep, "/")


def generate_report(
    *,
    metrics: dict[str, Any],
    report_path: Path,
    plot_records: list[dict[str, Any]],
) -> str:
    aggregates = metrics["aggregates"]
    mujoco = aggregates["mujoco"]
    dlolab = aggregates["dlolab"]
    metrics_link = rel_link(Path(metrics["outputs"]["metrics_json"]), start=report_path.parent)

    lines: list[str] = []
    lines.append("# M1 Simulator Comparison")
    lines.append("")
    lines.append(f"- commit: `{metrics['commit_hash']}`")
    lines.append(f"- seeds: `{metrics['seeds']}`")
    lines.append(f"- metrics: [`{metrics['outputs']['metrics_json']}`]({metrics_link})")
    lines.append(f"- scripted workload: {len(metrics['config']['sequences'])} sequences × {len(metrics['seeds'])} seeds × 2 sims")
    lines.append("")

    lines.append("## 설치 난이도/소요")
    lines.append("")
    lines.append("| sim | 설치/구동 메모 |")
    lines.append("| --- | --- |")
    lines.append(
        "| MuJoCo cable | `mujoco 3.10.0`; CPU 실행; gravity+ground-plane scene; MJX는 cable plugin 미지원. "
        "MuJoCo 3.10 cable body names are `ropeB_first`/`ropeB_*`/`ropeB_last`, so the adapter enumerates generated names dynamically. "
        "Weld grasp uses identity relpose plus mocap pre-positioning to avoid snap. |"
    )
    lines.append(
        "| DLO-Lab | install SUCCESS under the 2 h timebox: `external/DLO-Lab` clone `c5026a9`, `torch 2.10.0+cu128` ~95 s, "
        "editable `genesis-world 1.0.0` extras ~87 s. Resolver pins: `numpy<2.5`, `fsspec<=2026.2.0`, `packaging<26.0`; `uv pip check` clean. "
        "SharePoint assets returned HTTP 401 but were not needed for `ParameterizedRod`. Runtime alias `gs.ti_float=gs.qd_float` is required for the DLO-Lab 1.0.0 bug; external source remains unpatched. |"
    )
    lines.append("")

    lines.append("## smoke 통과 여부")
    lines.append("")
    lines.append("두 sim 모두 smoke PASS이며 MuJoCo 단독 통과 상태가 아니다.")
    lines.append("")
    lines.append("| sim | result | log | key facts |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(
        "| MuJoCo cable | PASS 7/7 | `outputs/reports/smoke_mujoco_stdout.log` | primitive wall-time 2.62 s; settle 2653 steps; low lift; small delta; no NaN. |"
    )
    lines.append(
        "| DLO-Lab | PASS 8/8 | `outputs/reports/smoke_dlolab_stdout.log` | primitive wall-time 4.34 s; stiffness×2 vs ×1 final-shape L2 diff 0.196; GPU batch `n_envs=4`; no NaN. |"
    )
    lines.append("")

    lines.append("## primitive wall-time")
    lines.append("")
    lines.append("Measured by this comparison run around each grasp→move→release→settle primitive.")
    lines.append("")
    lines.append("| sim | primitives | mean wall-time [s] | max wall-time [s] |")
    lines.append("| --- | ---: | ---: | ---: |")
    for sim_name, agg in (("MuJoCo cable", mujoco), ("DLO-Lab", dlolab)):
        lines.append(
            f"| {sim_name} | {agg['primitive_count']} | "
            f"{fmt_float(agg['wall_time_s']['mean'])} | {fmt_float(agg['wall_time_s']['max'])} |"
        )
    lines.append("")

    lines.append("## settle 안정성")
    lines.append("")
    lines.append(
        f"Comparison threshold: velocity `< {metrics['config']['thresholds']['settle_vel']}` with max "
        f"{metrics['config']['thresholds']['settle_max_steps']} steps per primitive. "
        "DLO-Lab smoke high-lift detach needed 7771 steps at the same 1e-3 threshold, hence smoke used max_steps 12000."
    )
    lines.append("")
    lines.append("| sim | convergence rate | non-converged | settle steps mean | settle steps max | NaN failures | straightness Δ/sequence mean |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for sim_name, agg in (("MuJoCo cable", mujoco), ("DLO-Lab", dlolab)):
        lines.append(
            f"| {sim_name} | {fmt_pct(agg['settle_convergence_rate'])} | {agg['settle_nonconverged_count']} | "
            f"{fmt_float(agg['settle_steps']['mean'], 1)} | {fmt_float(agg['settle_steps']['max'], 0)} | "
            f"{agg['nan_check_fail_count']} | {fmt_float(agg['straightness_improvement_per_sequence']['mean'])} |"
        )
    lines.append("")

    lines.append("## 파라미터화 커버리지")
    lines.append("")
    lines.append("| axis | MuJoCo cable | DLO-Lab |")
    lines.append("| --- | --- | --- |")
    lines.append("| length / segments | Regen-MJCF path covers `length_m` and `n_segments`. | `ParameterizedRod` rebuild covers length and vertices. |")
    lines.append("| bend | Cable plugin bend multiplier. | Runtime `set_bending_stiffness` setter. |")
    lines.append("| twist | Cable plugin twist multiplier. | Runtime `set_twisting_stiffness` setter. |")
    lines.append("| friction | Geom/plane friction from `RopeParams.friction`. | Runtime `set_mu_s`/`set_mu_k` setters. |")
    lines.append("| plasticity | Not supported by the MuJoCo cable adapter. | DLO-Lab runtime setters include `plastic_yield`/`creep`; DGCC P0 keeps them inactive until @M6. |")
    lines.append("")

    lines.append("## 병렬화")
    lines.append("")
    lines.append("| sim | status |")
    lines.append("| --- | --- |")
    lines.append("| MuJoCo cable | CPU single-process adapter. MJX cable path unsupported because the cable plugin is not available in MJX. |")
    lines.append("| DLO-Lab | GPU headless path works; smoke verified batched `n_envs=4` with `sample_centerline(32).shape == (4, 32, 3)`. |")
    lines.append("")

    lines.append("## 육안 궤적 플롯")
    lines.append("")
    for plot in plot_records:
        link = rel_link(Path(plot["path"]), start=report_path.parent)
        alt = f"{plot['sim']} {plot['sequence_id']} seed {plot['seed']}"
        lines.append(f"- {plot['sim']} `{plot['sequence_id']}` seed={plot['seed']}: ![{alt}]({link})")
    lines.append("")

    lines.append("## 리스크")
    lines.append("")
    lines.append("| sim | risks |")
    lines.append("| --- | --- |")
    lines.append(
        "| DLO-Lab | 5-week-old external code with no CI signal in this workspace; `ti_float` alias bug; SharePoint asset HTTP 401; dependency pin fragility around torch/genesis/numpy/fsspec/packaging. |"
    )
    lines.append(
        "| MuJoCo cable | MuJoCo 3.10 generated-name scheme drift; settle sensitivity to delta size; CPU wall-time scales without GPU batching. |"
    )
    lines.append("")

    lines.append("## 요약 표")
    lines.append("")
    lines.append("| 항목 | MuJoCo cable | DLO-Lab |")
    lines.append("| --- | --- | --- |")
    lines.append("| smoke | PASS 7/7 | PASS 8/8 |")
    lines.append(
        f"| compare primitive wall-time mean/max | {fmt_float(mujoco['wall_time_s']['mean'])} / {fmt_float(mujoco['wall_time_s']['max'])} s | "
        f"{fmt_float(dlolab['wall_time_s']['mean'])} / {fmt_float(dlolab['wall_time_s']['max'])} s |"
    )
    lines.append(
        f"| compare settle convergence | {fmt_pct(mujoco['settle_convergence_rate'])}; max steps {fmt_float(mujoco['settle_steps']['max'], 0)} | "
        f"{fmt_pct(dlolab['settle_convergence_rate'])}; max steps {fmt_float(dlolab['settle_steps']['max'], 0)} |"
    )
    lines.append(
        f"| straightness Δ/sequence mean | {fmt_float(mujoco['straightness_improvement_per_sequence']['mean'])} | "
        f"{fmt_float(dlolab['straightness_improvement_per_sequence']['mean'])} |"
    )
    lines.append("| parameter axes | length/segments, bend, twist, friction; no plasticity | length/vertices, bend, twist, friction; plasticity setters present but inactive in P0 |")
    lines.append("| parallelism | CPU single-process | GPU batch verified with `n_envs=4` |")
    lines.append("| notable risks | name drift, settle sensitivity, CPU wall-time | external-code maturity, alias bug, asset 401, pin fragility |")
    lines.append("")
    return "\n".join(lines)


def destroy_genesis_if_needed() -> None:
    try:
        import genesis as gs

        if getattr(gs, "_initialized", False):
            gs.destroy()
    except Exception:
        pass
    gc.collect()


def run_comparison(args: argparse.Namespace, config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    seed_offsets = [int(offset) for offset in config.get("seed_offsets", [0, 1, 2])]
    seeds = [int(args.seed) + offset for offset in seed_offsets]
    sequences = config.get("sequences", [])
    if len(sequences) != 5:
        raise ValueError(f"expected exactly 5 scripted sequences, got {len(sequences)}")
    for sequence in sequences:
        n_primitives = len(sequence.get("primitives", []))
        if n_primitives < 2 or n_primitives > 4:
            raise ValueError(f"sequence {sequence.get('id')} must contain 2-4 primitives, got {n_primitives}")

    params = params_from_config(config)
    outputs_cfg = config.get("outputs", {})
    metrics_path = Path(outputs_cfg.get("metrics_json", "outputs/metrics/sim_comparison_metrics.json"))
    report_path = Path(outputs_cfg.get("report_md", "outputs/reports/sim_comparison.md"))
    plots_dir = Path(outputs_cfg.get("plots_dir", "outputs/plots"))
    for path in (metrics_path, report_path, plots_dir):
        path.mkdir(parents=True, exist_ok=True) if path.suffix == "" else path.parent.mkdir(parents=True, exist_ok=True)

    plot_cfg = config.get("plots", {})
    plot_sequence_ids = set(str(seq_id) for seq_id in plot_cfg.get("sequence_ids", []))
    plot_seed = int(args.seed) + int(plot_cfg.get("seed_offset", 0))
    plot_records: list[dict[str, Any]] = []

    run_records_by_sim: dict[str, list[dict[str, Any]]] = {"mujoco": [], "dlolab": []}
    print(f"compare_sims config={config_path} seed_base={args.seed} seeds={seeds}")
    print(f"rope_params={asdict(params)}")

    for sim_name in ("mujoco", "dlolab"):
        for seed in seeds:
            for sequence in sequences:
                record, initial_centerline, final_centerline = run_sequence(
                    sim_name=sim_name,
                    sequence=sequence,
                    seed=seed,
                    params=params,
                    config=config,
                )
                run_records_by_sim[sim_name].append(record)
                converged = sum(1 for primitive in record["primitive_records"] if primitive["settle_converged"])
                print(
                    "RUN "
                    f"sim={sim_name} seed={seed} seq={record['sequence_id']} "
                    f"primitives={len(record['primitive_records'])} converged={converged}/{len(record['primitive_records'])} "
                    f"straightness={record['straightness_initial']:.4f}->{record['straightness_final']:.4f}"
                )

                if seed == plot_seed and record["sequence_id"] in plot_sequence_ids:
                    plot_path = plots_dir / f"compare_{sim_name}_{record['sequence_id']}.png"
                    write_centerline_plot(
                        path=plot_path,
                        sim_name=sim_name,
                        sequence_id=record["sequence_id"],
                        seed=seed,
                        before=initial_centerline,
                        after=final_centerline,
                    )
                    plot_records.append(
                        {
                            "sim": sim_name,
                            "sequence_id": record["sequence_id"],
                            "seed": int(seed),
                            "path": plot_path.as_posix(),
                        }
                    )

        if sim_name == "dlolab":
            destroy_genesis_if_needed()

    aggregates = {sim_name: aggregate_records(records) for sim_name, records in run_records_by_sim.items()}
    metrics = {
        "schema_version": 1,
        "config_path": config_path.as_posix(),
        "config": config,
        "commit_hash": get_git_commit_hash(),
        "seed_base": int(args.seed),
        "seed_offsets": seed_offsets,
        "seeds": seeds,
        "rope_params": asdict(params),
        "run_records": run_records_by_sim,
        "aggregates": aggregates,
        "plots": plot_records,
        "outputs": {
            "metrics_json": metrics_path.as_posix(),
            "report_md": report_path.as_posix(),
            "stdout_log": Path(config.get("outputs", {}).get("stdout_log", "outputs/reports/compare_sims_stdout.log")).as_posix(),
        },
    }

    metrics_path.write_text(json.dumps(jsonable(metrics), indent=2, sort_keys=True), encoding="utf-8")
    report = generate_report(metrics=metrics, report_path=report_path, plot_records=plot_records)
    report_path.write_text(report, encoding="utf-8")

    for sim_name, agg in aggregates.items():
        print(
            "AGG "
            f"sim={sim_name} primitives={agg['primitive_count']} "
            f"wall_mean={fmt_float(agg['wall_time_s']['mean'])} wall_max={fmt_float(agg['wall_time_s']['max'])} "
            f"settle_rate={fmt_pct(agg['settle_convergence_rate'])} "
            f"straightness_seq_mean={fmt_float(agg['straightness_improvement_per_sequence']['mean'])} "
            f"nan_failures={agg['nan_check_fail_count']}"
        )
    print(f"metrics_json={metrics_path}")
    print(f"report_md={report_path}")
    print("plots=" + ", ".join(plot["path"] for plot in plot_records))
    print("COMPARE_SIMS PASS")
    return metrics


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    log_path = Path(config.get("outputs", {}).get("stdout_log", "outputs/reports/compare_sims_stdout.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            run_comparison(args, config, config_path)
            return 0
        except Exception as exc:
            print(f"COMPARE_SIMS FAIL: {exc!r}")
            return 1
        finally:
            destroy_genesis_if_needed()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    raise SystemExit(main())
