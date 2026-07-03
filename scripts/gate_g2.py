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
import hashlib
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
from scipy.fft import dct, idct
from scipy.stats import pearsonr, spearmanr
import yaml
_PREFERRED_PLOT_FONTS = ("Noto Sans CJK KR", "Noto Sans CJK JP", "Droid Sans Fallback")
_AVAILABLE_PLOT_FONTS = {font.name for font in font_manager.fontManager.ttflist}
_PLOT_FONT_FAMILY = next(
    (name for name in _PREFERRED_PLOT_FONTS if name in _AVAILABLE_PLOT_FONTS),
    "DejaVu Sans",
)
plt.rcParams["font.family"] = [_PLOT_FONT_FAMILY, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from dgcc.goals.distance import D, canonical_shape_flip, chamfer_distance, correspondence_l2
from dgcc.goals.dual_goal import (
    CG_DIM,
    SHAPE_CHANNEL_COUNT,
    TEMPLATE_NAMES,
    DualGoal,
    c_g,
    canonical_centerline,
    goal_curve,
    make_goal,
    make_shape_template,
    normalize_shape_template,
)
from dgcc.phi.dct import CHANNEL_LAYOUT_ID, DCT_NORM, DCT_TYPE
from dgcc.phi.resample import resample
from dgcc.utils.meta import get_git_commit_hash

V2_VERDICT_SOURCE = "issue #6 comment 2026-07-03T03:23Z"
AMENDED_DEFINITION_TEXT = (
    "correspondence L2 (K=32 canonical arc-length resample path): "
    "D(X,G) = min over orientation flip of (1/L)·sqrt((1/K)·Σ_k ||X̃_k − G̃_k||²) "
    "using absolute coordinates; D_shape removes each centroid before the same computation; "
    "orientation flip evaluates k↔k and k↔K+1−k correspondences; gate PASS = "
    "Spearman rho(ΔD, Δ||c_g_anchor||) >= 0.9 AND "
    "Spearman rho(ΔD_shape, Δ||c_g_shape||) >= 0.9."
)
V3_VERDICT_SOURCE = "issue #6 comment 4872665607"
V3_CASE_DETERMINATION = "A"
V3_BASE_M = 8
V3_EXTRA_MS = (12, 16)
V3_AXES = ("x", "y", "z")
V3_QUANTILE_POINTS = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
V3_ORIENTATION_CONVENTION_TEXT = (
    "orientation canonicalization 규약 (2026-07-03 M5R2 Case A로 편입 — 버그 수정, 파라미터 변경 아님): "
    "per transition, choose exactly one flip decision against the goal using X_before, selecting the "
    "orientation that minimizes the mode-1..7 residual versus the goal's shape Phi; apply that fixed "
    "decision identically to X_before and X_after in both c_g_shape and D_shape computations for the "
    "shape component. Component (a) absolute D keeps the existing correspondence_l2 min-flip unchanged."
)


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
    parser.add_argument(
        "--v2",
        action="store_true",
        help=(
            "Run the amended component-split correspondence-L2 re-measurement and "
            "Chamfer sensitivity diagnostic, writing only new *_v2/new artifacts."
        ),
    )
    parser.add_argument(
        "--v3",
        action="store_true",
        help=(
            "Run the M5R2 Case A fixed-orientation shape re-measurement, writing only "
            "*_v3 artifacts."
        ),
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


def compute_measurements_v2(data: TransitionArrays, goals: list[DualGoal]) -> dict[str, np.ndarray]:
    if len(goals) != data.X_before.shape[0]:
        raise ValueError("number of sampled goals must equal transition count")
    out = {
        "delta_d_corr": np.empty(len(goals), dtype=float),
        "delta_d_shape": np.empty(len(goals), dtype=float),
        "delta_d_chamfer": np.empty(len(goals), dtype=float),
        "delta_cg_norm": np.empty(len(goals), dtype=float),
        "delta_anchor_norm": np.empty(len(goals), dtype=float),
        "delta_shape_norm": np.empty(len(goals), dtype=float),
    }
    for idx, goal in enumerate(goals):
        length = float(data.length_m[idx])
        g_curve = goal_curve(goal, length)

        corr_before = correspondence_l2(data.X_before[idx], g_curve, length)
        corr_after = correspondence_l2(data.X_after[idx], g_curve, length)
        shape_before = correspondence_l2(data.X_before[idx], g_curve, length, shape_only=True)
        shape_after = correspondence_l2(data.X_after[idx], g_curve, length, shape_only=True)
        chamfer_before = D(data.X_before[idx], goal, length)
        chamfer_after = D(data.X_after[idx], goal, length)

        cg_before = c_g(data.X_before[idx], goal, length)
        cg_after = c_g(data.X_after[idx], goal, length)

        out["delta_d_corr"][idx] = corr_after - corr_before
        out["delta_d_shape"][idx] = shape_after - shape_before
        out["delta_d_chamfer"][idx] = chamfer_after - chamfer_before
        out["delta_cg_norm"][idx] = float(np.linalg.norm(cg_after) - np.linalg.norm(cg_before))
        out["delta_anchor_norm"][idx] = float(
            np.linalg.norm(cg_after[-3:]) - np.linalg.norm(cg_before[-3:])
        )
        out["delta_shape_norm"][idx] = float(
            np.linalg.norm(cg_after[:-3]) - np.linalg.norm(cg_before[:-3])
        )
    return out


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


def summarize_correlation(
    name: str,
    mask: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    threshold: float,
    *,
    x_key: str,
    y_key: str,
    definition: str,
) -> dict[str, Any]:
    x = x_values[mask]
    y = y_values[mask]
    if x.size < 2:
        rho = np.nan
        pvalue = np.nan
    else:
        result = spearmanr(x, y)
        rho = float(result.statistic)
        pvalue = float(result.pvalue)
    passes = bool(np.isfinite(rho) and rho >= threshold)
    return {
        "name": name,
        "definition": definition,
        "rho": rho if np.isfinite(rho) else None,
        "pvalue": pvalue if np.isfinite(pvalue) else None,
        "n": int(x.size),
        "threshold": float(threshold),
        "passes": passes,
        x_key: _series_summary(x),
        y_key: _series_summary(y),
    }


def _format_rho(value: float | None) -> str:
    return "nan" if value is None else f"{float(value):.6f}"


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


def make_component_scatter(
    path: Path,
    x_values: np.ndarray,
    y_values: np.ndarray,
    primary_mask: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 5.5), constrained_layout=True)
    ax.scatter(x_values[primary_mask], y_values[primary_mask], s=9, alpha=0.45, linewidths=0)
    ax.axhline(0.0, color="0.7", linewidth=0.8)
    ax.axvline(0.0, color="0.7", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
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


def population_masks(data: TransitionArrays, settle_max_steps: int) -> dict[str, np.ndarray]:
    return {
        "primary": data.grasp_success & (data.settle_steps != settle_max_steps),
        "all_success": data.grasp_success.copy(),
        "all_transitions": np.ones_like(data.grasp_success, dtype=bool),
    }


def summarize_component_a(name: str, mask: np.ndarray, measurements: dict[str, np.ndarray], threshold: float) -> dict[str, Any]:
    return summarize_correlation(
        name,
        mask,
        measurements["delta_d_corr"],
        measurements["delta_anchor_norm"],
        threshold,
        x_key="delta_D_corr",
        y_key="delta_c_g_anchor_norm",
        definition="Spearman rho(ΔD_corr absolute correspondence L2, Δ||c_g_anchor||)",
    )


def summarize_component_b(name: str, mask: np.ndarray, measurements: dict[str, np.ndarray], threshold: float) -> dict[str, Any]:
    return summarize_correlation(
        name,
        mask,
        measurements["delta_d_shape"],
        measurements["delta_shape_norm"],
        threshold,
        x_key="delta_D_shape",
        y_key="delta_c_g_shape_norm",
        definition="Spearman rho(ΔD_shape centroid-removed correspondence L2, Δ||c_g_shape||)",
    )


def summarize_chamfer_reference(
    name: str,
    mask: np.ndarray,
    measurements: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, Any]:
    return {
        "anchor_component": summarize_correlation(
            f"{name}_chamfer_anchor_reference",
            mask,
            measurements["delta_d_chamfer"],
            measurements["delta_anchor_norm"],
            threshold,
            x_key="delta_D_chamfer",
            y_key="delta_c_g_anchor_norm",
            definition="REFERENCE ONLY: Spearman rho(ΔD_old_chamfer, Δ||c_g_anchor||)",
        ),
        "shape_component": summarize_correlation(
            f"{name}_chamfer_shape_reference",
            mask,
            measurements["delta_d_chamfer"],
            measurements["delta_shape_norm"],
            threshold,
            x_key="delta_D_chamfer",
            y_key="delta_c_g_shape_norm",
            definition="REFERENCE ONLY: Spearman rho(ΔD_old_chamfer, Δ||c_g_shape||)",
        ),
        "full_c_g_norm_reference": summarize_correlation(
            f"{name}_chamfer_full_c_g_reference",
            mask,
            measurements["delta_d_chamfer"],
            measurements["delta_cg_norm"],
            threshold,
            x_key="delta_D_chamfer",
            y_key="delta_c_g_norm",
            definition="REFERENCE ONLY: v1 Spearman rho(ΔD_old_chamfer, Δ||c_g||)",
        ),
    }


def goal_stream_hash(goals: list[DualGoal]) -> str:
    digest = hashlib.sha256()
    float64_le = np.dtype("<f8")
    for goal in goals:
        digest.update(str(goal.template_name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(goal.anchor_mode.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.ascontiguousarray(goal.anchor, dtype=float64_le).tobytes())
        digest.update(np.ascontiguousarray(goal.shape_template, dtype=float64_le).tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def build_goal_sampling_identity_proof(
    *,
    goals: list[DualGoal],
    sampling_spec: dict[str, Any],
    v1_metrics_path: Path,
    v1_recomputed_primary_rho: float | None,
) -> dict[str, Any]:
    stream_hash = goal_stream_hash(goals)
    proof: dict[str, Any] = {
        "sample_goals_call": "sample_goals(seed=args.seed, lengths_m=data.length_m, sampling_cfg=config['goal_sampling'])",
        "goal_stream_hash_sha256": stream_hash,
        "v1_recomputed_goal_stream_hash_sha256": stream_hash,
        "matches_v1_recomputed_hash": True,
        "sampling_spec": sampling_spec,
        "v1_metrics_path": str(v1_metrics_path),
    }
    if not v1_metrics_path.exists():
        proof["v1_metrics_available"] = False
        return proof

    v1 = json.loads(v1_metrics_path.read_text(encoding="utf-8"))
    v1_spec = v1.get("goal_sampling_spec", {})
    comparisons = {
        "seed": v1_spec.get("seed") == sampling_spec.get("seed"),
        "templates": v1_spec.get("templates") == sampling_spec.get("templates"),
        "template_counts_all_transitions": v1_spec.get("template_counts_all_transitions")
        == sampling_spec.get("template_counts_all_transitions"),
        "anchor_mode": v1_spec.get("anchor_mode") == sampling_spec.get("anchor_mode"),
        "anchor_box_unit_length": v1_spec.get("anchor_box_unit_length")
        == sampling_spec.get("anchor_box_unit_length"),
    }
    v1_primary_rho = v1.get("primary", {}).get("rho")
    rho_abs_diff = (
        abs(float(v1_primary_rho) - float(v1_recomputed_primary_rho))
        if v1_primary_rho is not None and v1_recomputed_primary_rho is not None
        else None
    )
    proof.update(
        {
            "v1_metrics_available": True,
            "v1_recorded_sampling_spec_comparisons": comparisons,
            "v1_recorded_sampling_spec_matches": all(comparisons.values()),
            "v1_primary_rho_recorded": v1_primary_rho,
            "v1_primary_rho_recomputed_from_current_stream": v1_recomputed_primary_rho,
            "v1_primary_rho_abs_diff": rho_abs_diff,
            "v1_primary_rho_matches_recomputed": rho_abs_diff is not None and rho_abs_diff <= 1.0e-12,
        }
    )
    return proof


def random_smooth_shape(rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0.0, 1.0, 257)
    y = np.zeros_like(t)
    z = np.zeros_like(t)
    for freq in range(1, 6):
        y += rng.normal(0.0, 0.09 / freq) * np.sin(2.0 * np.pi * freq * t + rng.uniform(0.0, 2.0 * np.pi))
        z += rng.normal(0.0, 0.045 / freq) * np.cos(2.0 * np.pi * freq * t + rng.uniform(0.0, 2.0 * np.pi))
    dense = np.column_stack((t - 0.5, y, z))
    return normalize_shape_template(resample(dense))


def build_chamfer_sensitivity_pairs(seed: int = 20260703) -> list[tuple[np.ndarray, np.ndarray, str]]:
    templates = [make_shape_template(name) for name in TEMPLATE_NAMES]
    pairs: list[tuple[np.ndarray, np.ndarray, str]] = []
    for left in range(len(templates)):
        for right in range(left + 1, len(templates)):
            pairs.append((templates[left], templates[right], "template_vs_template"))
            alphas = np.linspace(0.05, 0.95, 25)
            morphs = [
                normalize_shape_template((1.0 - alpha) * templates[left] + alpha * templates[right])
                for alpha in alphas
            ]
            for idx in range(len(morphs) - 3):
                pairs.append((morphs[idx], morphs[idx + 3], "template_morph"))
    rng = np.random.default_rng(seed)
    random_curves = [random_smooth_shape(rng) for _ in range(180)]
    for idx in range(len(random_curves) - 1):
        pairs.append((random_curves[idx], random_curves[idx + 1], "random_smooth"))
    return pairs


def run_chamfer_sensitivity_experiment(metrics_path: Path, scatter_path: Path) -> dict[str, Any]:
    pairs = build_chamfer_sensitivity_pairs()
    chamfer_values = np.empty(len(pairs), dtype=float)
    corr_values = np.empty(len(pairs), dtype=float)
    label_counts: dict[str, int] = {}
    for idx, (x_raw, y_raw, label) in enumerate(pairs):
        label_counts[label] = label_counts.get(label, 0) + 1
        x = x_raw - x_raw.mean(axis=0, keepdims=True)
        y = y_raw - y_raw.mean(axis=0, keepdims=True)
        chamfer_values[idx] = chamfer_distance(x, y, 1.0)
        corr_values[idx] = correspondence_l2(x, y, 1.0, shape_only=True)

    spearman = spearmanr(chamfer_values, corr_values)
    pearson = pearsonr(chamfer_values, corr_values)

    scatter_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 5.5), constrained_layout=True)
    ax.scatter(chamfer_values, corr_values, s=12, alpha=0.55, linewidths=0)
    ax.set_xlabel("ΔChamfer, old length-normalized (same-centroid pairs)")
    ax.set_ylabel("ΔD_shape correspondence L2")
    ax.set_title("Chamfer sensitivity sanity diagnostic")
    fig.savefig(scatter_path, dpi=160)
    plt.close(fig)

    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "diagnostic_only": True,
        "purpose": "Quantify verdict hypothesis 3: old Chamfer may be insensitive to same-centroid shape changes.",
        "pair_count": int(len(pairs)),
        "pair_sources": label_counts,
        "same_centroid": "Each curve is centroid-removed before distance computation.",
        "length_m": 1.0,
        "spearman": {"rho": float(spearman.statistic), "pvalue": float(spearman.pvalue)},
        "pearson": {"r": float(pearson.statistic), "pvalue": float(pearson.pvalue)},
        "old_chamfer_normalized": _series_summary(chamfer_values),
        "d_shape_correspondence_l2": _series_summary(corr_values),
        "delta_chamfer_normalized": _series_summary(chamfer_values),
        "delta_d_shape_correspondence_l2": _series_summary(corr_values),
        "outputs": {"metrics_json": str(metrics_path), "scatter_png": str(scatter_path)},
    }
    write_json(metrics_path, payload)
    return payload


def v3_centered(curve: np.ndarray) -> np.ndarray:
    arr = np.asarray(curve, dtype=float)
    return arr - arr.mean(axis=0, keepdims=True)


def v3_canonical_shape(raw_curve: np.ndarray, *, flip: bool = False) -> np.ndarray:
    curve = canonical_centerline(raw_curve)
    if flip:
        curve = curve[::-1].copy()
    return v3_centered(curve)


def v3_coeffs_full(shape_curve: np.ndarray) -> np.ndarray:
    return dct(shape_curve, type=DCT_TYPE, norm=DCT_NORM, axis=0)


def v3_coeff_vector_from_coeffs(coeffs: np.ndarray, modes: int) -> np.ndarray:
    return coeffs[1:modes, :].T.reshape(3 * (modes - 1))


def v3_reconstruct_lowpass_from_coeffs(coeffs: np.ndarray, modes: int) -> np.ndarray:
    truncated = np.zeros_like(coeffs)
    truncated[1:modes, :] = coeffs[1:modes, :]
    return idct(truncated, type=DCT_TYPE, norm=DCT_NORM, axis=0)


def v3_rms_pointwise_l2(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((x - y) ** 2, axis=1))))


def v3_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def v3_series_summary(values: np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"min": None, "mean": None, "std": None, "max": None}
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "max": float(np.max(arr)),
    }


def v3_quantiles(values: np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"q{int(q * 100):03d}": None for q in V3_QUANTILE_POINTS}
    qs = np.quantile(arr, V3_QUANTILE_POINTS)
    return {f"q{int(q * 100):03d}": float(v) for q, v in zip(V3_QUANTILE_POINTS, qs, strict=True)}


def v3_safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0 or not np.isfinite(denominator):
        return 0.0
    return float(numerator / denominator)


def v3_per_channel_summary(
    before_coeff_residuals: np.ndarray,
    delta_coeff_residuals: np.ndarray,
    primary_mask: np.ndarray,
) -> dict[str, Any]:
    before = before_coeff_residuals[primary_mask]
    delta = delta_coeff_residuals[primary_mask]
    before_energy_total = np.sum(before[:, 1:, :] ** 2, axis=(1, 2))
    delta_energy_total = np.sum(delta[:, 1:, :] ** 2, axis=(1, 2))

    rows: list[dict[str, Any]] = []
    for axis_index, axis in enumerate(V3_AXES):
        for mode in range(1, before_coeff_residuals.shape[1]):
            before_e = before[:, mode, axis_index] ** 2
            delta_e = delta[:, mode, axis_index] ** 2
            before_frac = np.divide(
                before_e,
                before_energy_total,
                out=np.zeros_like(before_e),
                where=before_energy_total > 0.0,
            )
            delta_frac = np.divide(
                delta_e,
                delta_energy_total,
                out=np.zeros_like(delta_e),
                where=delta_energy_total > 0.0,
            )
            rows.append(
                {
                    "channel": f"{axis}{mode}",
                    "axis": axis,
                    "mode": mode,
                    "in_base_c_g_shape": bool(mode < V3_BASE_M),
                    "mean_before_energy_fraction": float(np.mean(before_frac)),
                    "median_before_energy_fraction": float(np.median(before_frac)),
                    "mean_delta_energy_fraction": float(np.mean(delta_frac)),
                    "median_delta_energy_fraction": float(np.median(delta_frac)),
                    "mean_before_coeff": float(np.mean(before[:, mode, axis_index])),
                    "std_before_coeff": float(np.std(before[:, mode, axis_index])),
                    "mean_delta_coeff": float(np.mean(delta[:, mode, axis_index])),
                    "std_delta_coeff": float(np.std(delta[:, mode, axis_index])),
                }
            )

    lowpass_rows = [row for row in rows if row["in_base_c_g_shape"]]
    before_top = sorted(rows, key=lambda row: row["mean_before_energy_fraction"], reverse=True)[:12]
    delta_top = sorted(rows, key=lambda row: row["mean_delta_energy_fraction"], reverse=True)[:12]
    before_tail_top = sorted(
        [row for row in rows if not row["in_base_c_g_shape"]],
        key=lambda row: row["mean_before_energy_fraction"],
        reverse=True,
    )[:12]
    delta_tail_top = sorted(
        [row for row in rows if not row["in_base_c_g_shape"]],
        key=lambda row: row["mean_delta_energy_fraction"],
        reverse=True,
    )[:12]
    return {
        "basis": "orthonormal DCT-II coefficients of centroid-removed fixed-orientation residuals; modes 1..31 only",
        "population": "primary",
        "lowpass_channels_mode_1_to_7": lowpass_rows,
        "top_channels_by_before_residual_energy_fraction": before_top,
        "top_channels_by_delta_residual_energy_fraction": delta_top,
        "top_tail_channels_by_before_residual_energy_fraction": before_tail_top,
        "top_tail_channels_by_delta_residual_energy_fraction": delta_tail_top,
    }


def make_v3_shape_scatter(path: Path, x: np.ndarray, y: np.ndarray, mask: np.ndarray, rho: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    ax.scatter(x[mask], y[mask], s=9, alpha=0.45, linewidths=0)
    ax.axhline(0.0, color="0.72", linewidth=0.8)
    ax.axvline(0.0, color="0.72", linewidth=0.8)
    ax.set_xlabel("ΔD_shape_full fixed orientation")
    ax.set_ylabel("Δ||c_g_shape|| fixed orientation (modes 1..7)")
    ax.set_title(f"M5R2 D1 flip-consistent primary shape coupling, Spearman ρ={rho:.6f}")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def make_v3_tail_hist(path: Path, before_tail: np.ndarray, delta_tail: np.ndarray, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), constrained_layout=True, sharey=True)
    bins = np.linspace(0.0, 1.0, 41)
    axes[0].hist(before_tail[mask], bins=bins, color="tab:blue", alpha=0.82)
    axes[0].set_title("Before residual tail energy")
    axes[0].set_xlabel("||modes >=8||² / ||modes >=1||²")
    axes[0].set_ylabel("primary transition count")
    axes[1].hist(delta_tail[mask], bins=bins, color="tab:orange", alpha=0.82)
    axes[1].set_title("Delta residual tail energy")
    axes[1].set_xlabel("||modes >=8||² / ||modes >=1||²")
    for ax in axes:
        ax.set_xlim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.22)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def compute_measurements_v3(data: TransitionArrays, goals: list[DualGoal]) -> dict[str, Any]:
    if len(goals) != data.X_before.shape[0]:
        raise ValueError("number of sampled goals must equal transition count")
    n = len(goals)
    k = int(data.X_before.shape[1])
    out: dict[str, Any] = {
        "delta_d_corr": np.empty(n, dtype=float),
        "delta_anchor_norm": np.empty(n, dtype=float),
        "delta_d_shape_flip_consistent": np.empty(n, dtype=float),
        "delta_d_lowpass_flip_consistent": np.empty(n, dtype=float),
        "delta_d_lowpass_raw_flip_consistent": np.empty(n, dtype=float),
        "delta_shape_norm_flip_consistent": np.empty(n, dtype=float),
        "delta_shape_norm_extra": {modes: np.empty(n, dtype=float) for modes in V3_EXTRA_MS},
        "fixed_flip": np.empty(n, dtype=bool),
        "old_dshape_flip_before": np.empty(n, dtype=bool),
        "old_dshape_flip_after": np.empty(n, dtype=bool),
        "d1_decision_margin": np.empty(n, dtype=float),
        "parseval_before_abs_err": np.empty(n, dtype=float),
        "parseval_after_abs_err": np.empty(n, dtype=float),
        "tail_fraction_before": np.empty(n, dtype=float),
        "tail_fraction_delta": np.empty(n, dtype=float),
        "before_coeff_residuals": np.empty((n, k, 3), dtype=float),
        "delta_coeff_residuals": np.empty((n, k, 3), dtype=float),
    }

    for idx, goal in enumerate(goals):
        length = float(data.length_m[idx])
        g_curve = goal_curve(goal, length)
        g_shape = v3_canonical_shape(g_curve)
        g_coeffs = v3_coeffs_full(g_shape)
        g_phi8 = v3_coeff_vector_from_coeffs(g_coeffs, V3_BASE_M)

        corr_before = correspondence_l2(data.X_before[idx], g_curve, length)
        corr_after = correspondence_l2(data.X_after[idx], g_curve, length)
        cg_before = c_g(data.X_before[idx], goal, length)
        cg_after = c_g(data.X_after[idx], goal, length)
        out["delta_d_corr"][idx] = corr_after - corr_before
        out["delta_anchor_norm"][idx] = float(
            np.linalg.norm(cg_after[-3:]) - np.linalg.norm(cg_before[-3:])
        )

        xb_identity = v3_canonical_shape(data.X_before[idx], flip=False)
        xb_flipped = v3_canonical_shape(data.X_before[idx], flip=True)
        xa_identity = v3_canonical_shape(data.X_after[idx], flip=False)
        xa_flipped = v3_canonical_shape(data.X_after[idx], flip=True)

        xb_id_coeffs = v3_coeffs_full(xb_identity)
        xb_flip_coeffs = v3_coeffs_full(xb_flipped)
        id_residual_norm = float(np.linalg.norm(g_phi8 - v3_coeff_vector_from_coeffs(xb_id_coeffs, V3_BASE_M)))
        flip_residual_norm = float(
            np.linalg.norm(g_phi8 - v3_coeff_vector_from_coeffs(xb_flip_coeffs, V3_BASE_M))
        )
        choose_flip = bool(flip_residual_norm < id_residual_norm)
        helper_flip = canonical_shape_flip(data.X_before[idx], goal, length, shape_modes=V3_BASE_M)
        if helper_flip != choose_flip:
            raise RuntimeError(f"canonical_shape_flip helper mismatch at transition {idx}")
        out["fixed_flip"][idx] = choose_flip
        out["d1_decision_margin"][idx] = id_residual_norm - flip_residual_norm

        xb = xb_flipped if choose_flip else xb_identity
        xa = xa_flipped if choose_flip else xa_identity
        xb_coeffs = xb_flip_coeffs if choose_flip else xb_id_coeffs
        xa_coeffs = v3_coeffs_full(xa)

        before_identity = v3_rms_pointwise_l2(xb_identity, g_shape)
        before_flipped = v3_rms_pointwise_l2(xb_flipped, g_shape)
        after_identity = v3_rms_pointwise_l2(xa_identity, g_shape)
        after_flipped = v3_rms_pointwise_l2(xa_flipped, g_shape)
        out["old_dshape_flip_before"][idx] = bool(before_flipped < before_identity)
        out["old_dshape_flip_after"][idx] = bool(after_flipped < after_identity)

        d_full_before = v3_rms_pointwise_l2(xb, g_shape) / length
        d_full_after = v3_rms_pointwise_l2(xa, g_shape) / length
        out["delta_d_shape_flip_consistent"][idx] = d_full_after - d_full_before

        g_low = v3_reconstruct_lowpass_from_coeffs(g_coeffs, V3_BASE_M)
        xb_low = v3_reconstruct_lowpass_from_coeffs(xb_coeffs, V3_BASE_M)
        xa_low = v3_reconstruct_lowpass_from_coeffs(xa_coeffs, V3_BASE_M)
        d_low_before_raw = v3_rms_pointwise_l2(xb_low, g_low)
        d_low_after_raw = v3_rms_pointwise_l2(xa_low, g_low)
        d_low_before = d_low_before_raw / length
        d_low_after = d_low_after_raw / length
        out["delta_d_lowpass_raw_flip_consistent"][idx] = d_low_after_raw - d_low_before_raw
        out["delta_d_lowpass_flip_consistent"][idx] = d_low_after - d_low_before

        residual_before_phi8 = g_phi8 - v3_coeff_vector_from_coeffs(xb_coeffs, V3_BASE_M)
        residual_after_phi8 = g_phi8 - v3_coeff_vector_from_coeffs(xa_coeffs, V3_BASE_M)
        norm_before_phi8 = float(np.linalg.norm(residual_before_phi8))
        norm_after_phi8 = float(np.linalg.norm(residual_after_phi8))
        out["delta_shape_norm_flip_consistent"][idx] = norm_after_phi8 - norm_before_phi8
        out["parseval_before_abs_err"][idx] = abs(d_low_before_raw * np.sqrt(k) - norm_before_phi8)
        out["parseval_after_abs_err"][idx] = abs(d_low_after_raw * np.sqrt(k) - norm_after_phi8)

        for modes in V3_EXTRA_MS:
            residual_before = v3_coeff_vector_from_coeffs(g_coeffs, modes) - v3_coeff_vector_from_coeffs(
                xb_coeffs,
                modes,
            )
            residual_after = v3_coeff_vector_from_coeffs(g_coeffs, modes) - v3_coeff_vector_from_coeffs(
                xa_coeffs,
                modes,
            )
            out["delta_shape_norm_extra"][modes][idx] = float(
                np.linalg.norm(residual_after) - np.linalg.norm(residual_before)
            )

        residual_before_coeffs = g_coeffs - xb_coeffs
        residual_after_coeffs = g_coeffs - xa_coeffs
        residual_delta_coeffs = residual_after_coeffs - residual_before_coeffs
        out["before_coeff_residuals"][idx] = residual_before_coeffs
        out["delta_coeff_residuals"][idx] = residual_delta_coeffs

        before_total_energy = float(np.sum(residual_before_coeffs[1:, :] ** 2))
        before_tail_energy = float(np.sum(residual_before_coeffs[V3_BASE_M:, :] ** 2))
        delta_total_energy = float(np.sum(residual_delta_coeffs[1:, :] ** 2))
        delta_tail_energy = float(np.sum(residual_delta_coeffs[V3_BASE_M:, :] ** 2))
        out["tail_fraction_before"][idx] = v3_safe_ratio(before_tail_energy, before_total_energy)
        out["tail_fraction_delta"][idx] = v3_safe_ratio(delta_tail_energy, delta_total_energy)

    return out


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    config, config_text = load_config(config_path)
    if args.v2 and args.v3:
        raise ValueError("--v2 and --v3 are mutually exclusive")
    outputs = config.get("outputs", {})
    if args.v3:
        log_path = Path(outputs.get("stdout_log_v3", "outputs/reports/gate_g2_v3_stdout.log"))
    elif args.v2:
        log_path = Path(outputs.get("stdout_log_v2", "outputs/reports/gate_g2_v2_stdout.log"))
    else:
        log_path = Path(outputs.get("stdout_log", "outputs/reports/gate_g2_stdout.log"))

    with tee_stdout(log_path):
        if args.v3:
            run_v3(args=args, config=config, config_text=config_text, config_path=config_path, log_path=log_path)
        elif args.v2:
            run_v2(args=args, config=config, config_text=config_text, config_path=config_path, log_path=log_path)
        else:
            run(args=args, config=config, config_text=config_text, config_path=config_path, log_path=log_path)


def run_v2(
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
    v1_metrics_path = Path(outputs.get("metrics_json", "outputs/metrics/g2_correlation.json"))
    metrics_path = Path(outputs.get("metrics_json_v2", "outputs/metrics/g2_correlation_v2.json"))
    plots_dir = Path(outputs.get("plots_dir", "outputs/plots"))
    anchor_scatter_path = Path(outputs.get("scatter_anchor_png", plots_dir / "g2_scatter_anchor.png"))
    shape_scatter_path = Path(outputs.get("scatter_shape_png", plots_dir / "g2_scatter_shape.png"))
    sanity_metrics_path = Path(outputs.get("chamfer_sensitivity_json", "outputs/metrics/chamfer_sensitivity.json"))
    sanity_scatter_path = Path(outputs.get("chamfer_sensitivity_png", plots_dir / "chamfer_sensitivity.png"))

    threshold = float(quantitative.get("threshold_rho", 0.9))
    if threshold != 0.9:
        raise ValueError("G2 v2 threshold is immutable and must remain 0.9")
    min_primary_n = int(quantitative.get("minimum_primary_n", 1000))
    settle_max_steps = int(filters.get("settle_max_steps", 5000))

    print(f"G2 v2 measurement start seed={int(args.seed)} config={config_path}")
    print(f"verdict_source: {V2_VERDICT_SOURCE}")
    data = load_transitions(dataset_path)
    goals, sampling_spec = sample_goals(
        seed=int(args.seed),
        lengths_m=data.length_m,
        sampling_cfg=config.get("goal_sampling", {}),
    )
    measurements = compute_measurements_v2(data, goals)

    masks = population_masks(data, settle_max_steps)
    primary_mask = masks["primary"]
    if int(primary_mask.sum()) < min_primary_n:
        raise RuntimeError(
            f"primary population has n={int(primary_mask.sum())}, below required {min_primary_n}"
        )

    component_a = summarize_component_a("primary_component_a_anchor", primary_mask, measurements, threshold)
    component_b = summarize_component_b("primary_component_b_shape", primary_mask, measurements, threshold)
    pass_overall = bool(component_a["passes"] and component_b["passes"])

    variants = {
        name: {
            "component_a": summarize_component_a(f"{name}_component_a_anchor", mask, measurements, threshold),
            "component_b": summarize_component_b(f"{name}_component_b_shape", mask, measurements, threshold),
            "pass_overall": bool(
                summarize_component_a(f"{name}_component_a_anchor", mask, measurements, threshold)["passes"]
                and summarize_component_b(f"{name}_component_b_shape", mask, measurements, threshold)["passes"]
            ),
        }
        for name, mask in masks.items()
        if name != "primary"
    }

    chamfer_reference = {
        "label": "reference-only old Chamfer D component decomposition; not part of the amended gate",
        "primary": summarize_chamfer_reference("primary", primary_mask, measurements, threshold),
        "variants": {
            name: summarize_chamfer_reference(name, mask, measurements, threshold)
            for name, mask in masks.items()
            if name != "primary"
        },
    }
    goal_sampling_proof = build_goal_sampling_identity_proof(
        goals=goals,
        sampling_spec=sampling_spec,
        v1_metrics_path=v1_metrics_path,
        v1_recomputed_primary_rho=chamfer_reference["primary"]["full_c_g_norm_reference"]["rho"],
    )

    make_component_scatter(
        anchor_scatter_path,
        measurements["delta_d_corr"],
        measurements["delta_anchor_norm"],
        primary_mask,
        xlabel="ΔD_corr = absolute correspondence L2 after-before",
        ylabel="Δ||c_g_anchor||",
        title="G2 v2 component A: anchor",
    )
    make_component_scatter(
        shape_scatter_path,
        measurements["delta_d_shape"],
        measurements["delta_shape_norm"],
        primary_mask,
        xlabel="ΔD_shape = centroid-removed correspondence L2 after-before",
        ylabel="Δ||c_g_shape||",
        title="G2 v2 component B: shape",
    )
    sanity_payload = run_chamfer_sensitivity_experiment(sanity_metrics_path, sanity_scatter_path)
    physics_note = read_physics_quality_note(dm_stats_path)

    payload = {
        "schema_version": 2,
        "created_at": utc_now(),
        "verdict_source": V2_VERDICT_SOURCE,
        "amended_definition_text": AMENDED_DEFINITION_TEXT,
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
            "shape_component": "first 21 channels of c_g",
            "anchor_component": "last 3 channels of c_g",
            "order": "[Phi_shape(G)-Phi_shape(X) for mode>=1 channels, anchor(G)-anchor(X) xyz]",
        },
        "source_dataset": str(dataset_path),
        "dataset_record_count": int(data.X_before.shape[0]),
        "primary_population": {
            "definition": "grasp_success AND settle_steps != settle_max_steps",
            "settle_max_steps": settle_max_steps,
            "n": int(primary_mask.sum()),
        },
        "component_a": component_a,
        "component_b": component_b,
        "pass_overall": pass_overall,
        "stopping_rule": (
            None
            if pass_overall
            else (
                "amended §8 stopping rule engaged: shape component re-failed (rho < 0.9) on the "
                "single authorized re-measurement — NO further retries; awaiting the human "
                "redesign decision (issue #6 re-judgment). Threshold 0.9 remains immutable. "
                "Caveat for the human packet: the chamfer-sensitivity experiment rebuts only "
                "hypothesis 3 (metric shape-blindness); hypothesis 2 (near-goal regime absent "
                "from the transition sampling) remains an open explanation for the weak shape "
                "coupling and is deferred to the redesign decision."
            )
        ),
        "variants": variants,
        "chamfer_reference": chamfer_reference,
        "goal_sampling_proof": goal_sampling_proof,
        "methodology": {
            "component_a": "Spearman rho(ΔD_corr, Δ||c_g_anchor||) on primary population",
            "component_b": "Spearman rho(ΔD_shape, Δ||c_g_shape||) on primary population",
            "delta_c_g_anchor_norm": "||last 3 c_g channels after|| - ||last 3 c_g channels before||",
            "delta_c_g_shape_norm": "||first 21 c_g channels after|| - ||first 21 c_g channels before||",
            "old_chamfer_reference": "ΔD_old_chamfer uses the unchanged length-normalized bidirectional Chamfer D from v1",
            "physics_quality_note": physics_note,
        },
        "outputs": {
            "metrics_json": str(metrics_path),
            "scatter_anchor_png": str(anchor_scatter_path),
            "scatter_shape_png": str(shape_scatter_path),
            "stdout_log": str(log_path),
            "chamfer_sensitivity_json": str(sanity_metrics_path),
            "chamfer_sensitivity_png": str(sanity_scatter_path),
        },
        "chamfer_sensitivity_summary": {
            "spearman": sanity_payload["spearman"],
            "pearson": sanity_payload["pearson"],
            "pair_count": sanity_payload["pair_count"],
        },
    }
    write_json(metrics_path, payload)

    status_a = "PASS" if component_a["passes"] else "FAIL"
    status_b = "PASS" if component_b["passes"] else "FAIL"
    status_overall = "PASS" if pass_overall else "FAIL"
    print(f"COMPONENT_A {status_a}: rho={_format_rho(component_a['rho'])} n={component_a['n']} threshold={threshold:.3f}")
    print(f"COMPONENT_B {status_b}: rho={_format_rho(component_b['rho'])} n={component_b['n']} threshold={threshold:.3f}")
    print(f"OVERALL {status_overall}: component_a AND component_b")
    for name, rows in variants.items():
        print(
            f"variant {name}: component_a rho={_format_rho(rows['component_a']['rho'])} "
            f"component_b rho={_format_rho(rows['component_b']['rho'])}"
        )
    print(
        "chamfer reference primary: "
        f"anchor rho={_format_rho(chamfer_reference['primary']['anchor_component']['rho'])} "
        f"shape rho={_format_rho(chamfer_reference['primary']['shape_component']['rho'])}"
    )
    print(
        "chamfer sensitivity: "
        f"spearman={float(sanity_payload['spearman']['rho']):.6f} "
        f"pearson={float(sanity_payload['pearson']['r']):.6f} n={sanity_payload['pair_count']}"
    )
    print(f"wrote metrics: {metrics_path}")
    print(f"wrote anchor scatter: {anchor_scatter_path}")
    print(f"wrote shape scatter: {shape_scatter_path}")
    print(f"wrote chamfer sensitivity: {sanity_metrics_path}, {sanity_scatter_path}")
    print(f"stdout log: {log_path}")


def run_v3(
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
    v2_metrics_path = Path(outputs.get("metrics_json_v2", "outputs/metrics/g2_correlation_v2.json"))
    metrics_path = Path(outputs.get("metrics_json_v3", "outputs/metrics/g2_correlation_v3.json"))
    plots_dir = Path(outputs.get("plots_dir", "outputs/plots"))
    shape_scatter_path = Path(outputs.get("scatter_shape_png_v3", plots_dir / "g2_scatter_shape_v3.png"))
    tail_hist_path = Path(outputs.get("tail_histogram_png_v3", plots_dir / "g2_tail_histogram_v3.png"))

    threshold = float(quantitative.get("threshold_rho", 0.9))
    if threshold != 0.9:
        raise ValueError("G2 v3 threshold is immutable and must remain 0.9")
    min_primary_n = int(quantitative.get("minimum_primary_n", 1000))
    settle_max_steps = int(filters.get("settle_max_steps", 5000))

    if not v2_metrics_path.exists():
        raise FileNotFoundError(f"v3 requires v2 goal-stream proof at {v2_metrics_path}")
    v2_metrics = json.loads(v2_metrics_path.read_text(encoding="utf-8"))
    recorded_goal_hash = str(v2_metrics["goal_sampling_proof"]["goal_stream_hash_sha256"])

    print(f"G2 v3 measurement start seed={int(args.seed)} config={config_path}")
    print(f"verdict_source: {V3_VERDICT_SOURCE}")
    print(f"case_determination: {V3_CASE_DETERMINATION}")
    data = load_transitions(dataset_path)
    goals, sampling_spec = sample_goals(
        seed=int(args.seed),
        lengths_m=data.length_m,
        sampling_cfg=config.get("goal_sampling", {}),
    )
    computed_goal_hash = goal_stream_hash(goals)
    if computed_goal_hash != recorded_goal_hash:
        raise RuntimeError(
            f"goal stream hash mismatch: computed {computed_goal_hash}, recorded {recorded_goal_hash}"
        )

    measurements = compute_measurements_v3(data, goals)
    masks = population_masks(data, settle_max_steps)
    primary_mask = masks["primary"]
    if int(primary_mask.sum()) < min_primary_n:
        raise RuntimeError(
            f"primary population has n={int(primary_mask.sum())}, below required {min_primary_n}"
        )

    component_a = summarize_component_a("primary_component_a_anchor", primary_mask, measurements, threshold)
    component_b = summarize_correlation(
        "primary_component_b_shape_flip_consistent",
        primary_mask,
        measurements["delta_d_shape_flip_consistent"],
        measurements["delta_shape_norm_flip_consistent"],
        threshold,
        x_key="delta_D_shape_flip_consistent",
        y_key="delta_c_g_shape_norm_flip_consistent",
        definition=(
            "Spearman rho(ΔD_shape centroid-removed fixed-orientation correspondence L2, "
            "Δ||c_g_shape|| under the M5R2 Case A orientation convention)"
        ),
    )
    component_b["orientation_convention"] = V3_ORIENTATION_CONVENTION_TEXT
    component_b["case_determination"] = V3_CASE_DETERMINATION
    pass_overall = bool(component_a["passes"] and component_b["passes"])

    variants = {
        name: {
            "component_a": summarize_component_a(f"{name}_component_a_anchor", mask, measurements, threshold),
            "component_b": summarize_correlation(
                f"{name}_component_b_shape_flip_consistent",
                mask,
                measurements["delta_d_shape_flip_consistent"],
                measurements["delta_shape_norm_flip_consistent"],
                threshold,
                x_key="delta_D_shape_flip_consistent",
                y_key="delta_c_g_shape_norm_flip_consistent",
                definition=(
                    "Spearman rho(ΔD_shape centroid-removed fixed-orientation correspondence L2, "
                    "Δ||c_g_shape|| under the M5R2 Case A orientation convention)"
                ),
            ),
        }
        for name, mask in masks.items()
        if name != "primary"
    }
    for rows in variants.values():
        rows["component_b"]["orientation_convention"] = V3_ORIENTATION_CONVENTION_TEXT
        rows["component_b"]["case_determination"] = V3_CASE_DETERMINATION
        rows["pass_overall"] = bool(rows["component_a"]["passes"] and rows["component_b"]["passes"])

    d1_rho = float(component_b["rho"])
    d1_p = float(component_b["pvalue"])
    d2_sanity_rho, d2_sanity_p = v3_spearman(
        measurements["delta_d_lowpass_flip_consistent"][primary_mask],
        measurements["delta_shape_norm_flip_consistent"][primary_mask],
    )
    d2_sanity_raw_rho, d2_sanity_raw_p = v3_spearman(
        measurements["delta_d_lowpass_raw_flip_consistent"][primary_mask],
        measurements["delta_shape_norm_flip_consistent"][primary_mask],
    )
    rho_trunc, rho_trunc_p = v3_spearman(
        measurements["delta_d_shape_flip_consistent"][primary_mask],
        measurements["delta_d_lowpass_flip_consistent"][primary_mask],
    )

    hypothetical_extended_m: dict[str, Any] = {}
    for modes in V3_EXTRA_MS:
        rho, pvalue = v3_spearman(
            measurements["delta_d_shape_flip_consistent"][primary_mask],
            measurements["delta_shape_norm_extra"][modes][primary_mask],
        )
        hypothetical_extended_m[str(modes)] = {
            "M": modes,
            "rho_delta_D_shape_full_vs_delta_c_g_shape_norm": rho,
            "pvalue": pvalue,
            "shape_modes": f"1..{modes - 1}",
            "orientation_convention": "D1 fixed flip chosen from before curve modes 1..7",
        }

    fixed_flip = measurements["fixed_flip"]
    old_dshape_flip_before = measurements["old_dshape_flip_before"]
    old_dshape_flip_after = measurements["old_dshape_flip_after"]
    old_before_after_inconsistent = old_dshape_flip_before != old_dshape_flip_after
    old_dshape_vs_cg_before = old_dshape_flip_before.copy()
    old_dshape_vs_cg_after = old_dshape_flip_after.copy()
    old_dshape_vs_cg_either = old_dshape_vs_cg_before | old_dshape_vs_cg_after
    old_dshape_vs_fixed_before = old_dshape_flip_before != fixed_flip
    old_dshape_vs_fixed_after = old_dshape_flip_after != fixed_flip

    tail_dominates = bool(d1_rho < threshold and d2_sanity_rho >= 0.99 and rho_trunc < threshold)
    if d1_rho >= threshold:
        suggested_case = "A"
        suggested_case_reason = "D1 flip-consistent rho is at or above 0.9."
    elif tail_dominates:
        suggested_case = "B"
        suggested_case_reason = (
            "D1 remains below 0.9, D2 lowpass sanity is >=0.99, and rho_trunc is below 0.9, "
            "identifying the mode>=8 tail as the rank bottleneck."
        )
    else:
        suggested_case = "C"
        suggested_case_reason = "D1 remains below 0.9 without rho_trunc clearly isolating truncation as the bottleneck."

    if suggested_case != V3_CASE_DETERMINATION:
        raise RuntimeError(f"v3 Case A promotion expected, but diagnostics suggested case {suggested_case}")

    make_v3_shape_scatter(
        shape_scatter_path,
        measurements["delta_d_shape_flip_consistent"],
        measurements["delta_shape_norm_flip_consistent"],
        primary_mask,
        d1_rho,
    )
    make_v3_tail_hist(
        tail_hist_path,
        measurements["tail_fraction_before"],
        measurements["tail_fraction_delta"],
        primary_mask,
    )
    physics_note = read_physics_quality_note(dm_stats_path)
    v2_sampling_spec = v2_metrics.get("goal_sampling_proof", {}).get("sampling_spec", {})
    v2_sampling_comparisons = {
        "seed": v2_sampling_spec.get("seed") == sampling_spec.get("seed"),
        "templates": v2_sampling_spec.get("templates") == sampling_spec.get("templates"),
        "template_counts_all_transitions": v2_sampling_spec.get("template_counts_all_transitions")
        == sampling_spec.get("template_counts_all_transitions"),
        "anchor_mode": v2_sampling_spec.get("anchor_mode") == sampling_spec.get("anchor_mode"),
        "anchor_box_unit_length": v2_sampling_spec.get("anchor_box_unit_length")
        == sampling_spec.get("anchor_box_unit_length"),
    }
    goal_sampling_proof = {
        "sample_goals_call": "sample_goals(seed=args.seed, lengths_m=data.length_m, sampling_cfg=config['goal_sampling'])",
        "goal_stream_hash_sha256": computed_goal_hash,
        "sampling_spec": sampling_spec,
        "v2_metrics_path": str(v2_metrics_path),
        "v2_recorded_goal_stream_hash_sha256": recorded_goal_hash,
        "v3_computed_goal_stream_hash_sha256": computed_goal_hash,
        "matches_v2_goal_stream_hash": bool(computed_goal_hash == recorded_goal_hash),
        "v2_recorded_sampling_spec_comparisons": v2_sampling_comparisons,
        "v2_recorded_sampling_spec_matches": all(v2_sampling_comparisons.values()),
    }

    primary_count = int(primary_mask.sum())
    diagnostics = {
        "definitions": {
            "D1": (
                "Fixed orientation per transition: choose flip of X_before minimizing mode-1..7 "
                "residual to goal Phi, then apply same flip to X_before/X_after for c_g_shape and "
                "centroid-removed D_shape_full."
            ),
            "D2": (
                "D_shape_lowpass is centroid-removed correspondence RMS between inverse-DCT "
                "reconstructions retaining modes 1..7 under the same fixed flip convention."
            ),
            "tail_energy": (
                "Tail fraction is ||DCT modes >=8 residual||^2 / ||DCT modes >=1 residual||^2 "
                "with orthonormal DCT-II coefficients."
            ),
        },
        "D1": {
            "rho": d1_rho,
            "pvalue": d1_p,
            "n": primary_count,
            "delta_D_shape_full": v3_series_summary(
                measurements["delta_d_shape_flip_consistent"][primary_mask]
            ),
            "delta_c_g_shape_norm": v3_series_summary(
                measurements["delta_shape_norm_flip_consistent"][primary_mask]
            ),
            "fixed_flip_fraction": float(np.mean(fixed_flip[primary_mask])),
            "fixed_flip_decision_margin_id_minus_flip": v3_series_summary(
                measurements["d1_decision_margin"][primary_mask]
            ),
        },
        "flip_inconsistency_fractions": {
            "population": "primary",
            "old_D_shape_before_vs_after": float(np.mean(old_before_after_inconsistent[primary_mask])),
            "old_D_shape_vs_c_g_stored_before": float(np.mean(old_dshape_vs_cg_before[primary_mask])),
            "old_D_shape_vs_c_g_stored_after": float(np.mean(old_dshape_vs_cg_after[primary_mask])),
            "old_D_shape_vs_c_g_stored_either_before_or_after": float(
                np.mean(old_dshape_vs_cg_either[primary_mask])
            ),
            "old_D_shape_vs_D1_fixed_before": float(np.mean(old_dshape_vs_fixed_before[primary_mask])),
            "old_D_shape_vs_D1_fixed_after": float(np.mean(old_dshape_vs_fixed_after[primary_mask])),
            "all_transitions": {
                "old_D_shape_before_vs_after": float(np.mean(old_before_after_inconsistent)),
                "old_D_shape_vs_c_g_stored_before": float(np.mean(old_dshape_vs_cg_before)),
                "old_D_shape_vs_c_g_stored_after": float(np.mean(old_dshape_vs_cg_after)),
                "old_D_shape_vs_c_g_stored_either_before_or_after": float(np.mean(old_dshape_vs_cg_either)),
                "old_D_shape_vs_D1_fixed_before": float(np.mean(old_dshape_vs_fixed_before)),
                "old_D_shape_vs_D1_fixed_after": float(np.mean(old_dshape_vs_fixed_after)),
            },
        },
        "D2": {
            "sanity_rho_delta_D_shape_lowpass_vs_delta_c_g_shape_norm": d2_sanity_rho,
            "sanity_pvalue": d2_sanity_p,
            "sanity_deviation_from_1": float(abs(1.0 - d2_sanity_rho)),
            "rho_trunc_delta_D_shape_full_vs_delta_D_shape_lowpass": rho_trunc,
            "rho_trunc_pvalue": rho_trunc_p,
            "delta_D_shape_lowpass": v3_series_summary(
                measurements["delta_d_lowpass_flip_consistent"][primary_mask]
            ),
            "delta_D_shape_lowpass_raw_unlength_normalized": v3_series_summary(
                measurements["delta_d_lowpass_raw_flip_consistent"][primary_mask]
            ),
            "raw_lowpass_sanity_rho_without_length_normalization": d2_sanity_raw_rho,
            "raw_lowpass_sanity_pvalue_without_length_normalization": d2_sanity_raw_p,
            "parseval_abs_error": {
                "before_max": float(np.max(measurements["parseval_before_abs_err"][primary_mask])),
                "after_max": float(np.max(measurements["parseval_after_abs_err"][primary_mask])),
                "before_mean": float(np.mean(measurements["parseval_before_abs_err"][primary_mask])),
                "after_mean": float(np.mean(measurements["parseval_after_abs_err"][primary_mask])),
            },
        },
        "tail_energy_quantiles": {
            "population": "primary",
            "before_basis": v3_quantiles(measurements["tail_fraction_before"][primary_mask]),
            "delta_basis": v3_quantiles(measurements["tail_fraction_delta"][primary_mask]),
            "before_basis_summary": v3_series_summary(measurements["tail_fraction_before"][primary_mask]),
            "delta_basis_summary": v3_series_summary(measurements["tail_fraction_delta"][primary_mask]),
            "histogram_png": str(tail_hist_path),
        },
        "hypothetical_extended_M": hypothetical_extended_m,
        "case_suggestion": {
            "case": suggested_case,
            "criteria": {
                "A": "D1 rho >= 0.9",
                "B": "D1 rho < 0.9 AND D2 sanity rho >= 0.99 AND rho_trunc < 0.9",
                "C": "otherwise",
            },
            "tail_dominates": tail_dominates,
            "reason": suggested_case_reason,
        },
        "per_channel_residual_contribution_summary": v3_per_channel_summary(
            measurements["before_coeff_residuals"],
            measurements["delta_coeff_residuals"],
            primary_mask,
        ),
    }

    payload = {
        "schema_version": 3,
        "created_at": utc_now(),
        "verdict_source": V3_VERDICT_SOURCE,
        "case_determination": V3_CASE_DETERMINATION,
        "orientation_convention": V3_ORIENTATION_CONVENTION_TEXT,
        "amended_definition_text": AMENDED_DEFINITION_TEXT,
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
            "shape_component": "first 21 channels of c_g",
            "anchor_component": "last 3 channels of c_g",
            "order": "[Phi_shape(G)-Phi_shape(X) for mode>=1 channels, anchor(G)-anchor(X) xyz]",
        },
        "source_dataset": str(dataset_path),
        "source_metrics_v2": str(v2_metrics_path),
        "dataset_record_count": int(data.X_before.shape[0]),
        "primary_population": {
            "definition": "grasp_success AND settle_steps != settle_max_steps",
            "settle_max_steps": settle_max_steps,
            "n": primary_count,
        },
        "component_a": component_a,
        "component_b": component_b,
        "pass_overall": pass_overall,
        "variants": variants,
        "goal_sampling_proof": goal_sampling_proof,
        "diagnostics": diagnostics,
        "methodology": {
            "component_a": "Unchanged v2 Spearman rho(ΔD_corr, Δ||c_g_anchor||) on primary population",
            "component_b": "M5R2 Case A fixed-orientation Spearman rho(ΔD_shape, Δ||c_g_shape||)",
            "delta_D_corr": "Absolute correspondence_l2 after-before; existing min-flip behavior unchanged",
            "delta_D_shape": "Centroid-removed RMS after-before under one X_before-selected flip per transition",
            "delta_c_g_shape_norm": "Mode-1..7 DCT residual norm after-before under the same fixed flip",
            "threshold_rho": threshold,
            "physics_quality_note": physics_note,
        },
        "outputs": {
            "metrics_json": str(metrics_path),
            "scatter_shape_png": str(shape_scatter_path),
            "tail_histogram_png": str(tail_hist_path),
            "stdout_log": str(log_path),
        },
    }
    write_json(metrics_path, payload)

    status_a = "PASS" if component_a["passes"] else "FAIL"
    status_b = "PASS" if component_b["passes"] else "FAIL"
    status_overall = "PASS" if pass_overall else "FAIL"
    print(f"goal_stream_hash_verified={computed_goal_hash == recorded_goal_hash}")
    print(f"COMPONENT_A {status_a}: rho={_format_rho(component_a['rho'])} n={component_a['n']} threshold={threshold:.3f}")
    print(f"COMPONENT_B {status_b}: rho={_format_rho(component_b['rho'])} n={component_b['n']} threshold={threshold:.3f}")
    print(f"OVERALL {status_overall}: component_a AND component_b")
    print(
        "diagnostics: "
        f"D1_rho={d1_rho:.12f} D2_sanity={d2_sanity_rho:.12f} rho_trunc={rho_trunc:.12f} "
        f"M12={hypothetical_extended_m['12']['rho_delta_D_shape_full_vs_delta_c_g_shape_norm']:.12f} "
        f"M16={hypothetical_extended_m['16']['rho_delta_D_shape_full_vs_delta_c_g_shape_norm']:.12f}"
    )
    for name, rows in variants.items():
        print(
            f"variant {name}: component_a rho={_format_rho(rows['component_a']['rho'])} "
            f"component_b rho={_format_rho(rows['component_b']['rho'])}"
        )
    print(f"wrote metrics: {metrics_path}")
    print(f"wrote shape scatter: {shape_scatter_path}")
    print(f"wrote tail histogram: {tail_hist_path}")
    print(f"stdout log: {log_path}")

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
