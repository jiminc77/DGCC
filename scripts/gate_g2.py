"""Measure the P0-M5 G2 dual-goal correlation gate.

CPU-only: this script reads the M4 HDF5 transition dataset directly and does not
import simulator/Genesis modules.  The measurement records the Spearman result
as-is against the immutable 0.9 threshold; threshold failures produce human
review proposals in JSON but do not alter Phi, D, populations, or thresholds.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.patches import Circle
import numpy as np
from scipy.stats import spearmanr
import yaml
_PREFERRED_PLOT_FONTS = ("Noto Sans CJK KR", "Noto Sans CJK JP", "Droid Sans Fallback")
_AVAILABLE_PLOT_FONTS = {font.name for font in font_manager.fontManager.ttflist}
_PLOT_FONT_FAMILY = next(
    (name for name in _PREFERRED_PLOT_FONTS if name in _AVAILABLE_PLOT_FONTS),
    "DejaVu Sans",
)
plt.rcParams["font.family"] = [_PLOT_FONT_FAMILY, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from dgcc.goals.distance import D
from dgcc.goals.dual_goal import (
    CG_DIM,
    SHAPE_CHANNEL_COUNT,
    TEMPLATE_NAMES,
    DualGoal,
    c_g,
    goal_curve,
    make_goal,
    make_shape_template,
)
from dgcc.phi.dct import CHANNEL_LAYOUT_ID
from dgcc.utils.meta import get_git_commit_hash


@dataclass(frozen=True)
class TransitionArrays:
    X_before: np.ndarray
    X_after: np.ndarray
    grasp_success: np.ndarray
    settle_steps: np.ndarray
    length_m: np.ndarray
    meta: dict[str, Any]


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
    parser = argparse.ArgumentParser(description="Measure the P0-M5 G2 dual-goal gate")
    parser.add_argument("--seed", type=int, default=0, help="deterministic goal-sampling seed")
    parser.add_argument(
        "--config",
        default="configs/gate_g2.yaml",
        help="YAML config path for dataset, filters, sampling, and outputs",
    )
    return parser


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    config = yaml.safe_load(text) or {}
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping, got {type(config).__name__}")
    return config, text


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


def load_transitions(path: Path) -> TransitionArrays:
    with h5py.File(path, "r") as h5:
        X_before = np.asarray(h5["X_before"][:], dtype=float)
        X_after = np.asarray(h5["X_after"][:], dtype=float)
        grasp_success = np.asarray(h5["grasp_success"][:], dtype=bool)
        settle_steps = np.asarray(h5["settle_steps"][:], dtype=int)
        rope_params_text = [str(value) for value in h5["rope_params"].asstr()[:]]
        meta = json.loads(_attr_to_str(h5.attrs["meta_json"]))

    length_m = np.asarray([_length_from_rope_params(text) for text in rope_params_text], dtype=float)
    if X_before.shape != X_after.shape or X_before.ndim != 3 or X_before.shape[2] != 3:
        raise ValueError("X_before/X_after must both have shape (N, K, 3)")
    if not (len(grasp_success) == len(settle_steps) == len(length_m) == X_before.shape[0]):
        raise ValueError("transition dataset columns have inconsistent lengths")
    return TransitionArrays(
        X_before=X_before,
        X_after=X_after,
        grasp_success=grasp_success,
        settle_steps=settle_steps,
        length_m=length_m,
        meta=meta,
    )


def _attr_to_str(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    return str(value)


def _length_from_rope_params(text: str) -> float:
    data = json.loads(text)
    length = float(data["length_m"])
    if length <= 0.0 or not np.isfinite(length):
        raise ValueError(f"invalid rope length_m in dataset: {length!r}")
    return length


def read_physics_quality_note(path: Path) -> str | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    note = data.get("physics_quality_note")
    return str(note) if note is not None else None


def sample_goals(
    *,
    seed: int,
    lengths_m: np.ndarray,
    sampling_cfg: dict[str, Any],
) -> tuple[list[DualGoal], dict[str, Any]]:
    templates = [str(name) for name in sampling_cfg.get("templates", TEMPLATE_NAMES)]
    if not templates:
        raise ValueError("goal_sampling.templates must contain at least one template")
    template_cache = {name: make_shape_template(name) for name in templates}
    anchor_mode = str(sampling_cfg.get("anchor_mode", "centroid"))
    box = sampling_cfg.get("anchor_box_unit_length", {})
    low = np.asarray(box.get("low", [-1.25, -1.25, 0.02]), dtype=float)
    high = np.asarray(box.get("high", [1.25, 1.25, 0.18]), dtype=float)
    if low.shape != (3,) or high.shape != (3,) or not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise ValueError("goal_sampling.anchor_box_unit_length low/high must be finite shape-(3,) arrays")
    if not np.all(low < high):
        raise ValueError("goal_sampling.anchor_box_unit_length must satisfy low < high per axis")

    rng = np.random.default_rng(seed)
    template_indices = rng.integers(0, len(templates), size=len(lengths_m), endpoint=False)
    anchors_unit = rng.uniform(low=low, high=high, size=(len(lengths_m), 3))
    anchors = anchors_unit * lengths_m[:, None]
    goals = [
        DualGoal(
            shape_template=template_cache[templates[int(template_indices[idx])]],
            anchor=anchors[idx],
            anchor_mode=anchor_mode,
            template_name=templates[int(template_indices[idx])],
        )
        for idx in range(len(lengths_m))
    ]
    counts = {name: int(np.sum(template_indices == idx)) for idx, name in enumerate(templates)}
    spec = {
        "seed": int(seed),
        "rng": "numpy.default_rng(seed)",
        "templates": templates,
        "template_counts_all_transitions": counts,
        "anchor_mode": anchor_mode,
        "anchor_box_unit_length": {"low": low.tolist(), "high": high.tolist()},
        "anchor_sampling": "uniform per axis; sampled_anchor = uniform(low, high) * transition.length_m",
        "goal_per_transition": "one independently sampled template index and anchor per transition",
    }
    return goals, spec


def compute_measurements(
    data: TransitionArrays, goals: list[DualGoal]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(goals) != data.X_before.shape[0]:
        raise ValueError("number of sampled goals must equal transition count")
    delta_d = np.empty(len(goals), dtype=float)
    delta_cg_norm = np.empty(len(goals), dtype=float)
    delta_anchor_norm = np.empty(len(goals), dtype=float)
    delta_shape_norm = np.empty(len(goals), dtype=float)
    for idx, goal in enumerate(goals):
        length = float(data.length_m[idx])
        d_before = D(data.X_before[idx], goal, length)
        d_after = D(data.X_after[idx], goal, length)
        cg_before = c_g(data.X_before[idx], goal, length)
        cg_after = c_g(data.X_after[idx], goal, length)
        delta_d[idx] = d_after - d_before
        delta_cg_norm[idx] = float(np.linalg.norm(cg_after) - np.linalg.norm(cg_before))
        # Report-only component split (architect advisory for the human packet):
        # anchor channels are the last 3, shape channels the first 21.
        delta_anchor_norm[idx] = float(
            np.linalg.norm(cg_after[-3:]) - np.linalg.norm(cg_before[-3:])
        )
        delta_shape_norm[idx] = float(
            np.linalg.norm(cg_after[:-3]) - np.linalg.norm(cg_before[:-3])
        )
    return delta_d, delta_cg_norm, delta_anchor_norm, delta_shape_norm


def summarize_population(
    name: str,
    mask: np.ndarray,
    delta_d: np.ndarray,
    delta_cg_norm: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    values_d = delta_d[mask]
    values_cg = delta_cg_norm[mask]
    if values_d.size < 2:
        rho = np.nan
        pvalue = np.nan
    else:
        result = spearmanr(values_d, values_cg)
        rho = float(result.statistic)
        pvalue = float(result.pvalue)
    passes = bool(np.isfinite(rho) and rho >= threshold)
    return {
        "name": name,
        "rho": rho if np.isfinite(rho) else None,
        "pvalue": pvalue if np.isfinite(pvalue) else None,
        "n": int(values_d.size),
        "threshold": float(threshold),
        "passes": passes,
        "delta_D": _series_summary(values_d),
        "delta_c_g_norm": _series_summary(values_cg),
    }


def _series_summary(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def make_scatter(path: Path, delta_d: np.ndarray, delta_cg_norm: np.ndarray, primary_mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 5.5), constrained_layout=True)
    ax.scatter(delta_d[primary_mask], delta_cg_norm[primary_mask], s=9, alpha=0.45, linewidths=0)
    ax.axhline(0.0, color="0.7", linewidth=0.8)
    ax.axvline(0.0, color="0.7", linewidth=0.8)
    ax.set_xlabel("ΔD = D(after, G) - D(before, G)")
    ax.set_ylabel("Δ||c_g|| = ||c_g(after)|| - ||c_g(before)||")
    ax.set_title("G2 primary population: converged successful transitions")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_goal_consistency_plots(config: dict[str, Any]) -> list[str]:
    qualitative = config.get("qualitative", {})
    sampling = config.get("goal_sampling", {})
    outputs = config.get("outputs", {})
    plots_dir = Path(outputs.get("plots_dir", "outputs/plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)

    lengths = [float(value) for value in qualitative.get("lengths_m", [0.5, 1.0, 1.6])]
    templates = [str(name) for name in sampling.get("templates", TEMPLATE_NAMES)]
    anchor_mode = str(qualitative.get("anchor_mode", sampling.get("anchor_mode", "centroid")))
    anchor = np.asarray(qualitative.get("anchor", [0.0, 0.0, 0.04]), dtype=float)
    radius_fraction = float(qualitative.get("boundary_radius_fraction_of_length", 0.05))
    seed = int(qualitative.get("initial_seed", 20260703))

    paths: list[str] = []
    for length in lengths:
        for template in templates:
            goal = make_goal(template, anchor, anchor_mode=anchor_mode)
            g_curve = goal_curve(goal, length)
            initial = analytic_initial_curve(length, seed + _stable_name_offset(template) + int(round(1000 * length)))
            boundary_radius = radius_fraction * length
            distance_initial = D(initial, goal, length)
            out = plots_dir / f"g2_goal_consistency_{_format_length(length)}_{template}.png"
            plot_goal_consistency(
                out,
                initial=initial,
                goal=g_curve,
                length_m=length,
                template=template,
                boundary_radius=boundary_radius,
                distance_initial=distance_initial,
            )
            paths.append(str(out))
    return paths


def analytic_initial_curve(length_m: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, 32)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    x = length_m * (t - 0.5) - 0.18 * length_m
    y = length_m * (0.11 * np.sin(2.0 * np.pi * t + phase) + 0.04 * np.sin(4.0 * np.pi * t - 0.5 * phase))
    z = 0.04 + length_m * (0.025 * np.sin(np.pi * t + 0.25 * phase))
    return np.column_stack((x, y, z))


def plot_goal_consistency(
    path: Path,
    *,
    initial: np.ndarray,
    goal: np.ndarray,
    length_m: float,
    template: str,
    boundary_radius: float,
    distance_initial: float,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
    for point in goal[::2]:
        ax.add_patch(
            Circle(
                (float(point[0]), float(point[1])),
                boundary_radius,
                facecolor="tab:green",
                edgecolor="none",
                alpha=0.055,
            )
        )
    ax.plot(goal[:, 0], goal[:, 1], color="tab:green", linewidth=2.2, label="goal curve")
    ax.scatter(goal[[0, -1], 0], goal[[0, -1], 1], color="tab:green", s=24)
    ax.plot(initial[:, 0], initial[:, 1], color="tab:blue", linewidth=1.8, label="initial rope curve")
    ax.scatter(initial[[0, -1], 0], initial[[0, -1], 1], color="tab:blue", s=20)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"G2 goal consistency — L={length_m:g} m, template={template}")
    ax.text(
        0.02,
        0.02,
        "잠정 경계 — M7 확정\n"
        f"visual radius ≈ 0.05·L = {boundary_radius:.3f} m\n"
        f"initial D = {distance_initial:.3f}; boundary display only",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.78, "edgecolor": "0.8"},
    )
    ax.legend(loc="upper right")
    pad = max(0.08, boundary_radius * 1.5)
    points = np.vstack((initial[:, :2], goal[:, :2]))
    ax.set_xlim(float(points[:, 0].min() - pad), float(points[:, 0].max() + pad))
    ax.set_ylim(float(points[:, 1].min() - pad), float(points[:, 1].max() + pad))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _stable_name_offset(name: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(name))


def _format_length(length: float) -> str:
    return f"{length:.1f}".replace(".", "p")


def build_proposals(primary: dict[str, Any]) -> list[dict[str, str]]:
    rho = primary.get("rho")
    rho_text = "nan" if rho is None else f"{float(rho):.6g}"
    return [
        {
            "status": "PROPOSAL_REQUIRES_HUMAN_DECISION",
            "topic": "anchor-channel scaling analysis",
            "rationale": f"Primary rho {rho_text} is below the immutable 0.9 threshold; analyze whether anchor channels dominate ||c_g|| before changing any definition.",
        },
        {
            "status": "PROPOSAL_REQUIRES_HUMAN_DECISION",
            "topic": "population effects",
            "rationale": "Compare primary, all-success, and all-transitions sensitivity rows to decide whether non-converged/outlier physics populations should remain separate in later gates.",
        },
    ]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(as_jsonable(payload), indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    config, config_text = load_config(config_path)
    outputs = config.get("outputs", {})
    log_path = Path(outputs.get("stdout_log", "outputs/reports/gate_g2_stdout.log"))

    with tee_stdout(log_path):
        run(args=args, config=config, config_text=config_text, config_path=config_path, log_path=log_path)


def run(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    config_text: str,
    config_path: Path,
    log_path: Path,
) -> None:
    dataset_cfg = config.get("dataset", {})
    outputs = config.get("outputs", {})
    quantitative = config.get("quantitative", {})
    filters = config.get("filters", {})

    dataset_path = Path(dataset_cfg.get("h5", "outputs/data/p0_random_transitions.h5"))
    dm_stats_path = Path(dataset_cfg.get("dm_stats_json", "outputs/metrics/dm_stats.json"))
    metrics_path = Path(outputs.get("metrics_json", "outputs/metrics/g2_correlation.json"))
    scatter_path = Path(outputs.get("scatter_png", "outputs/plots/g2_scatter.png"))
    threshold = float(quantitative.get("threshold_rho", 0.9))
    min_primary_n = int(quantitative.get("minimum_primary_n", 1000))
    settle_max_steps = int(filters.get("settle_max_steps", 5000))

    print(f"G2 measurement start seed={int(args.seed)} config={config_path}")
    data = load_transitions(dataset_path)
    goals, sampling_spec = sample_goals(
        seed=int(args.seed),
        lengths_m=data.length_m,
        sampling_cfg=config.get("goal_sampling", {}),
    )
    delta_d, delta_cg_norm, delta_anchor_norm, delta_shape_norm = compute_measurements(data, goals)

    primary_mask = data.grasp_success & (data.settle_steps != settle_max_steps)
    all_success_mask = data.grasp_success.copy()
    all_transitions_mask = np.ones_like(data.grasp_success, dtype=bool)
    if int(primary_mask.sum()) < min_primary_n:
        raise RuntimeError(
            f"primary population has n={int(primary_mask.sum())}, below required {min_primary_n}"
        )

    primary = summarize_population("primary", primary_mask, delta_d, delta_cg_norm, threshold)
    variants = {
        "all_success": summarize_population(
            "all_success",
            all_success_mask,
            delta_d,
            delta_cg_norm,
            threshold,
        ),
        "all_transitions": summarize_population(
            "all_transitions",
            all_transitions_mask,
            delta_d,
            delta_cg_norm,
            threshold,
        ),
    }

    def _component_rho(values: np.ndarray) -> float | None:
        result = spearmanr(delta_d[primary_mask], values[primary_mask])
        rho_value = float(result.statistic)
        return rho_value if np.isfinite(rho_value) else None

    diagnostics = {
        "note": (
            "REPORT-ONLY diagnostics for the M5 human gate (architect advisories): the spec-defined "
            "quantity is primary.rho on the full ||c_g||; these decompositions quantify WHERE the "
            "signal lives and change no definition or threshold."
        ),
        "component_split_primary": {
            "anchor_channels_only_rho": _component_rho(delta_anchor_norm),
            "shape_channels_only_rho": _component_rho(delta_shape_norm),
            "full_c_g_norm_rho": primary["rho"],
            "definition": "Spearman(delta_D, delta ||c_g subset||) on the primary population",
        },
    }

    make_scatter(scatter_path, delta_d, delta_cg_norm, primary_mask)
    qualitative_paths = make_goal_consistency_plots(config)
    physics_note = read_physics_quality_note(dm_stats_path)

    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "seed": int(args.seed),
        "config_path": str(config_path),
        "config_copy": config,
        "config_text": config_text,
        "commit_hash": get_git_commit_hash(),
        "channel_layout_id": CHANNEL_LAYOUT_ID,
        "c_g_layout": {
            "total_channels": CG_DIM,
            "shape_channels": SHAPE_CHANNEL_COUNT,
            "anchor_channels": 3,
            "order": "[Phi_shape(G)-Phi_shape(X) for mode>=1 channels, anchor(G)-anchor(X) xyz]",
        },
        "source_dataset": str(dataset_path),
        "dataset_record_count": int(data.X_before.shape[0]),
        "primary": primary,
        "variants": variants,
        "filter_definitions": {
            "primary": {
                "definition": "grasp_success AND settle_steps != settle_max_steps",
                "settle_max_steps": settle_max_steps,
                "rationale": "Primary G2 population excludes exact failed-grasp no-ops and non-converged successful transitions per M4 handoff.",
                "n": int(primary_mask.sum()),
            },
            "all_success": {
                "definition": "grasp_success",
                "n": int(all_success_mask.sum()),
            },
            "all_transitions": {
                "definition": "all records, including failed grasp no-ops",
                "n": int(all_transitions_mask.sum()),
            },
        },
        "goal_sampling_spec": sampling_spec,
        "diagnostics": diagnostics,
        "methodology": {
            "delta_D": "D(X_after,G)-D(X_before,G), D is bidirectional Chamfer divided by length_m",
            "delta_c_g_norm": "||c_g(X_after,G)|| - ||c_g(X_before,G)||",
            "spearman": "scipy.stats.spearmanr on fixed sampled goals; threshold comparison recorded as-is",
            "physics_quality_note": physics_note,
        },
        "outputs": {
            "metrics_json": str(metrics_path),
            "scatter_png": str(scatter_path),
            "goal_consistency_plots": qualitative_paths,
            "stdout_log": str(log_path),
        },
        "proposals": [] if primary["passes"] else build_proposals(primary),
    }
    write_json(metrics_path, payload)

    status = "PASS" if primary["passes"] else "SHORTFALL"
    rho = primary["rho"]
    rho_text = "nan" if rho is None else f"{float(rho):.6f}"
    print(f"{status}: primary Spearman rho={rho_text} n={primary['n']} threshold={threshold:.3f}")
    for name, row in variants.items():
        variant_rho = row["rho"]
        variant_rho_text = "nan" if variant_rho is None else f"{float(variant_rho):.6f}"
        print(f"variant {name}: rho={variant_rho_text} n={row['n']}")
    print(f"wrote metrics: {metrics_path}")
    print(f"wrote scatter: {scatter_path}")
    print(f"wrote goal consistency plots: {len(qualitative_paths)}")
    print(f"stdout log: {log_path}")


if __name__ == "__main__":
    main()
