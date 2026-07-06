"""P1-M3R T1 results report.

Aggregates M3R per-run artifacts only (no re-simulation) and applies the
pre-registered M3R criteria and P-a/P-b/P-c readings.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BOOTSTRAP_SEED = 20260703
BOOTSTRAP_B = 10_000
P_C_INTERPRETATION_RULE = "oracle 성공 → 과제 달성 가능 확정 · oracle ≫ policy → 학습 문제 확정 · oracle ≈ 0 → 판정 불능 (불가능 증명 아님)"
TASKS = {
    "t1a": "t1a_straighten",
    "t1b": "t1b_single_bend",
    "t1c": "t1c_endpoint_reposition",
}
FULL_TO_SHORT = {v: k for k, v in TASKS.items()}
SEEDS = (0, 1, 2)
TEMPLATES = ("straight", "u_bend", "s_curve", "random_smooth")
EXPECTED_RUNS = len(TASKS) * len(SEEDS)
REPORT_PATH = Path("outputs/reports/p1_m3r_results.md")
JSON_PATH = Path("outputs/metrics/p1_m3r_results.json")
PLOT_DIR = Path("outputs/plots")


def run_tag(task: str, seed: int) -> str:
    return f"m3r_{task}_s{seed}"


def run_path(task: str, seed: int) -> Path:
    return Path("outputs/metrics") / f"p1_run_{run_tag(task, seed)}.json"


def diag_path(task: str, seed: int) -> Path:
    return Path("outputs/metrics") / f"p1_diag_{run_tag(task, seed)}.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_complete(run: dict[str, Any]) -> bool:
    return (
        int(run.get("transitions", 0)) >= int(run.get("total_budget", 1 << 60))
        and run.get("halt_reason") is None
    )


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def bootstrap_diff_lb(treatment: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """(bootstrap mean diff, empirical 5th percentile) — registered method."""

    if len(treatment) == 0 or len(reference) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    diffs = np.empty(BOOTSTRAP_B)
    for i in range(BOOTSTRAP_B):
        t = rng.choice(treatment, size=len(treatment), replace=True)
        r = rng.choice(reference, size=len(reference), replace=True)
        diffs[i] = t.mean() - r.mean()
    return float(diffs.mean()), float(np.percentile(diffs, 5))


def corrected_counter_total(diag_file: Path, series_key: str, value_key: str) -> int | None:
    """Sum monotone counter segments across full-scene rebuild resets."""

    if not diag_file.exists():
        return None
    data = load_json(diag_file)
    rows = data.get(series_key)
    if rows is None and value_key in data:
        rows = data.get(value_key)
    if rows is None and value_key != series_key:
        for candidate in ("nan_incidents", "incident_counters"):
            candidate_rows = data.get(candidate)
            if isinstance(candidate_rows, list) and any(
                isinstance(row, dict) and value_key in row for row in candidate_rows
            ):
                rows = candidate_rows
                break
    if rows is None:
        return None
    values: list[int] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = row.get(value_key, row.get(series_key))
        parsed = finite_float(value)
        if parsed is not None:
            values.append(int(parsed))
    if not values:
        return 0
    total, prev = 0, 0
    for value in values:
        if value < prev:
            total += prev
        prev = value
    return total + prev


def final_eval(run: dict[str, Any]) -> dict[str, Any]:
    evals = run.get("evals") or []
    return dict(evals[-1]) if evals else {}


def final_episodes(run: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = run.get("eval_episodes") or []
    if not blocks:
        return []
    return list(blocks[-1].get("episodes") or [])


def episode_d_at_done(ep: dict[str, Any]) -> float:
    value = finite_float(ep.get("d_at_done"))
    if value is not None:
        return value
    return float(ep.get("final_d", float("nan")))


def episode_min_d(ep: dict[str, Any]) -> float:
    value = finite_float(ep.get("min_d"))
    if value is not None:
        return value
    steps = [finite_float(v) for v in ep.get("d_steps", [])]
    finite_steps = [v for v in steps if v is not None]
    if finite_steps:
        return float(min(finite_steps))
    return float(ep.get("final_d", float("nan")))


def mean_or_nan(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def pct_or_nan(values: list[float] | np.ndarray, pct: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, pct)) if arr.size else float("nan")


def episode_stats(episodes: list[dict[str, Any]]) -> dict[str, float | int]:
    success = [1.0 if ep.get("success") else 0.0 for ep in episodes]
    returns = [float(ep.get("return", float("nan"))) for ep in episodes]
    final_d = [float(ep.get("final_d", float("nan"))) for ep in episodes]
    d_at_done = [episode_d_at_done(ep) for ep in episodes]
    min_d = [episode_min_d(ep) for ep in episodes]
    return {
        "n": int(len(episodes)),
        "success_rate": mean_or_nan(success),
        "mean_return": mean_or_nan(returns),
        "mean_final_d": mean_or_nan(final_d),
        "mean_d_at_done": mean_or_nan(d_at_done),
        "mean_min_d": mean_or_nan(min_d),
        "min_d_p10": pct_or_nan(min_d, 10),
        "min_d_p50": pct_or_nan(min_d, 50),
        "min_d_p90": pct_or_nan(min_d, 90),
    }


def per_template_stats(episodes: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for template in TEMPLATES:
        rows = [ep for ep in episodes if ep.get("init_template") == template]
        if rows:
            out[template] = episode_stats(rows)
    return out


def load_runs() -> tuple[dict[str, dict[int, dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]], int]:
    runs: dict[str, dict[int, dict[str, Any]]] = {task: {} for task in TASKS}
    incomplete: list[dict[str, Any]] = []
    present = 0
    for task in TASKS:
        for seed in SEEDS:
            path = run_path(task, seed)
            if not path.exists():
                continue
            present += 1
            data = load_json(path)
            if is_complete(data):
                runs[task][seed] = data
            else:
                incomplete.append(
                    {
                        "run_tag": run_tag(task, seed),
                        "path": str(path),
                        "transitions": int(data.get("transitions", 0)),
                        "total_budget": int(data.get("total_budget", 0)),
                        "halt_reason": data.get("halt_reason"),
                    }
                )

    halted: list[dict[str, Any]] = []
    seen_halted_paths: set[Path] = set()
    for path in sorted(Path("outputs/metrics").glob("p1_run_m3r_t1*_s*.halted*.json")):
        seen_halted_paths.add(path)
        halted.append(halted_record(path))
    for item in incomplete:
        path = Path(item["path"])
        if item.get("halt_reason") and path not in seen_halted_paths:
            halted.append(halted_record(path))
    return runs, incomplete, halted, present


def halted_record(path: Path) -> dict[str, Any]:
    data = load_json(path)
    match = re.search(r"p1_run_(m3r_t1[abc]_s[012])(?:\.(.+))?\.json$", path.name)
    return {
        "run_tag": match.group(1) if match else path.stem.replace("p1_run_", ""),
        "archive": match.group(2) if match and match.group(2) else "",
        "path": str(path),
        "transitions_at_halt": int(data.get("transitions", 0)),
        "updates_at_halt": int(data.get("updates", 0)),
        "halt_reason": data.get("halt_reason"),
        "last_checkpoint": data.get("last_checkpoint"),
        "gap_p95_series": [row.get("overestimation_gap_p95") for row in data.get("evals", [])],
    }


def load_oracle(path: Path = Path("outputs/metrics/p1_o1_oracle.json")) -> dict[str, Any]:
    if not path.exists():
        return {"loaded": False, "path": str(path), "by_task": {}}
    data = load_json(path)
    by_task: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {task: {} for task in TASKS}
    for pass_name, pass_data in (data.get("passes") or {}).items():
        blocks = pass_data.get("blocks") or {}
        for full_name, block in blocks.items():
            task = FULL_TO_SHORT.get(full_name, full_name)
            if task not in by_task:
                continue
            stats = block.get("per_template_stats")
            if stats is None:
                stats = per_template_stats(block.get("episodes") or [])
            for template, template_stats in stats.items():
                by_task[task].setdefault(template, {})[pass_name] = template_stats
    return {"loaded": True, "path": str(path), "label": data.get("label"), "by_task": by_task}


def oracle_cell(oracle: dict[str, Any], task: str, template: str, pass_name: str, key: str) -> str:
    stats = (oracle.get("by_task") or {}).get(task, {}).get(template, {}).get(pass_name)
    if not stats:
        return "—"
    value = finite_float(stats.get(key))
    return "—" if value is None else f"{value:.4f}"


def gap_p95_summary(run: dict[str, Any]) -> dict[str, Any]:
    vals = [finite_float(ev.get("overestimation_gap_p95")) for ev in run.get("evals", [])]
    raw_count = len(run.get("evals", []))
    finite_vals = [v for v in vals if v is not None]
    bounded = raw_count > 0 and len(finite_vals) == raw_count
    return {
        "series": finite_vals,
        "bounded": bool(bounded),
        "max": float(max(finite_vals)) if finite_vals else None,
    }


def clamp_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = load_json(path)
    vals = [finite_float(row.get("td_target_clamp_hit_frac")) for row in data.get("updates", [])]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    tail = vals[-min(32, len(vals)) :]
    return {
        "n": len(vals),
        "max": float(max(vals)),
        "final": float(vals[-1]),
        "tail_mean": float(np.mean(tail)),
        "nonzero_frac": float(np.mean(np.asarray(vals) > 0.0)),
        "series_tail": [float(v) for v in vals[-min(16, len(vals)) :]],
        "steady_nonzero": bool(np.mean(tail) > 0.0),
    }


def eval_metric(eval_row: dict[str, Any], run: dict[str, Any], metric: str) -> float:
    if metric in eval_row:
        value = finite_float(eval_row.get(metric))
        if value is not None:
            return value
    if metric == "mean_d_at_done":
        return float(eval_row.get("mean_final_d", float("nan")))
    if metric == "mean_min_d":
        episodes = final_episodes(run)
        return float(episode_stats(episodes)["mean_min_d"]) if episodes else float("nan")
    return float("nan")


def plot_task(task: str, runs: dict[int, dict[str, Any]], random_success: float, random_return: float) -> str:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / f"p1_m3r_curves_{task}.png"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = (
        ("mean_return", "eval mean return", random_return),
        ("success_rate", "eval success rate", random_success),
        ("mean_d_at_done", "eval d_at_done", None),
    )
    if not runs:
        for ax, (_, label, ref_value) in zip(axes, metrics, strict=True):
            ax.text(0.5, 0.5, "No completed M3R runs", ha="center", va="center", transform=ax.transAxes)
            if ref_value is not None:
                ax.axhline(ref_value, color="gray", linestyle="--", linewidth=1, label="random ref")
                ax.legend(fontsize=7)
            ax.set_title(f"{TASKS[task]} — {label}")
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return str(out)

    xs = sorted({int(ev.get("transitions", 0)) for run in runs.values() for ev in run.get("evals", [])})
    for metric, label, ref_value in metrics:
        ax = axes[list(m[0] for m in metrics).index(metric)]
        mat = np.full((len(runs), len(xs)), np.nan)
        for row_idx, run in enumerate(runs.values()):
            for ev in run.get("evals", []):
                transitions = int(ev.get("transitions", 0))
                if transitions in xs:
                    mat[row_idx, xs.index(transitions)] = eval_metric(ev, run, metric)
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        ax.plot(xs, mean, marker="o")
        ax.fill_between(xs, mean - std, mean + std, alpha=0.25)
        if ref_value is not None:
            ax.axhline(ref_value, color="gray", linestyle="--", linewidth=1, label="random ref")
            ax.legend(fontsize=7)
        ax.set_xlabel("transitions")
        ax.set_title(f"{TASKS[task]} — {label} (mean±std, {len(runs)} seeds)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return str(out)


def run_status(runs: dict[str, dict[int, dict[str, Any]]], incomplete: list[dict[str, Any]], task: str, seed: int) -> str:
    tag = run_tag(task, seed)
    if seed in runs[task]:
        return "complete"
    for row in incomplete:
        if row["run_tag"] == tag:
            return "halted" if row.get("halt_reason") else "incomplete"
    return "missing"


def main() -> int:
    parser = argparse.ArgumentParser(description="P1-M3R T1 results report")
    parser.add_argument("--preliminary", action="store_true", help="allow missing/incomplete M3R runs")
    args = parser.parse_args()

    ref = load_json(Path("outputs/metrics/p1_random_reference.json"))
    oracle = load_oracle()
    runs, incomplete, halted, present = load_runs()
    completed = sum(len(seed_map) for seed_map in runs.values())
    missing = [run_tag(task, seed) for task in TASKS for seed in SEEDS if seed not in runs[task]]
    if missing and not args.preliminary:
        raise SystemExit(f"missing/incomplete M3R runs {missing}; use --preliminary to draft")

    payload: dict[str, Any] = {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "B": BOOTSTRAP_B,
        "preliminary": bool(missing),
        "expected_runs": EXPECTED_RUNS,
        "run_files_present": int(present),
        "completed_runs": int(completed),
        "missing_runs": missing,
        "incomplete_runs": incomplete,
        "halted_runs": halted,
        "tasks": {},
        "p_a": {
            "halt_count": len(halted),
            "target_halt_count": 0,
            "m3r_t1a_s1_status": run_status(runs, incomplete, "t1a", 1),
        },
        "p_b": {},
        "p_c": {
            "interpretation_rule": P_C_INTERPRETATION_RULE,
            "oracle_loaded": bool(oracle.get("loaded")),
            "oracle_path": oracle.get("path"),
        },
        "reward_constants_adjusted": False,
        "schema_debt_note": "HDF5 v3: termination-cause 필드 — M6 P2 승계 추적 항목",
        "hygiene_note": "old p1_t1_report halted-glob would match m3r halted files if re-run",
    }

    lines: list[str] = []
    lines.append("# P1-M3R — T1 결과 리포트")
    lines.append("")
    lines.append(f"> run files present: {present}/{EXPECTED_RUNS}; completed: {completed}/{EXPECTED_RUNS}; preliminary={bool(missing)}")
    if missing:
        lines.append(f"> 미완료/누락 M3R runs: {', '.join(missing)}")
    lines.append("")
    lines.append(
        "사전 등록 기준 M3R (i′): task random success > 1%이면 success-diff bootstrap, "
        "그 외에는 return-diff bootstrap (B=10,000, seed 20260703, CI95 LB=경험적 5퍼센타일 > 0). "
        "(ii) seed 간 최종 성공률 std < 15%p. (iii) t1a ≥ 70%는 factual record. "
        "final_d와 d_at_done은 함께 보고한다."
    )
    lines.append("")

    for task, full_name in TASKS.items():
        block = ref["blocks"][full_name]
        rand_succ = np.asarray([1.0 if ep.get("success") else 0.0 for ep in block.get("episodes", [])], dtype=float)
        rand_ret = np.asarray([float(ep.get("return", float("nan"))) for ep in block.get("episodes", [])], dtype=float)
        random_success = float(rand_succ.mean()) if rand_succ.size else float("nan")
        random_return = float(np.nanmean(rand_ret)) if rand_ret.size else float("nan")
        criterion_metric = "success" if random_success > 0.01 else "return"
        tinfo: dict[str, Any] = {
            "random_success": random_success,
            "random_return": random_return,
            "criterion_i_prime_metric": criterion_metric,
            "seeds": {},
            "template_min_d": {},
        }
        payload["tasks"][task] = tinfo

        lines.append(f"## {full_name}")
        lines.append("")
        lines.append(
            f"random 참조선: success {random_success:.3f}, return {random_return:.3f}; "
            f"criterion (i′) metric = {criterion_metric}"
        )
        lines.append("")
        lines.append("| seed | status | final success | final return | final_d | d_at_done | min_d | i′ LB(5%) | i′ pass | gap p95 bounded/max | clamp hit tail |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|---|---|")

        finals: list[float] = []
        task_gap_bounded: list[bool] = []
        task_gap_maxes: list[float] = []
        combined_episodes: list[dict[str, Any]] = []
        for seed in SEEDS:
            status = run_status(runs, incomplete, task, seed)
            if seed not in runs[task]:
                lines.append(f"| {seed} | {status} | — | — | — | — | — | — | — | — | — |")
                continue
            run = runs[task][seed]
            ev = final_eval(run)
            eps = final_episodes(run)
            combined_episodes.extend(eps)
            eps_stats = episode_stats(eps)
            succ = np.asarray([1.0 if ep.get("success") else 0.0 for ep in eps], dtype=float)
            rets = np.asarray([float(ep.get("return", float("nan"))) for ep in eps], dtype=float)
            treatment = succ if criterion_metric == "success" else rets
            reference = rand_succ if criterion_metric == "success" else rand_ret
            _, metric_lb = bootstrap_diff_lb(treatment, reference)
            significant = bool(metric_lb > 0.0)
            final_success = float(ev.get("success_rate", eps_stats["success_rate"]))
            final_return = float(ev.get("mean_return", eps_stats["mean_return"]))
            final_d = float(ev.get("mean_final_d", eps_stats["mean_final_d"]))
            d_at_done = float(ev.get("mean_d_at_done", eps_stats["mean_d_at_done"]))
            min_d = float(ev.get("mean_min_d", eps_stats["mean_min_d"]))
            finals.append(final_success)

            gap = gap_p95_summary(run)
            task_gap_bounded.append(bool(gap["bounded"]))
            if gap["max"] is not None:
                task_gap_maxes.append(float(gap["max"]))
            clamp = clamp_summary(diag_path(task, seed))
            nan_total = corrected_counter_total(diag_path(task, seed), "nan_incidents", "nan_incidents")
            magnitude_total = corrected_counter_total(
                diag_path(task, seed), "magnitude_incidents", "magnitude_incidents"
            )
            seed_info = {
                "final_success": final_success,
                "final_return": final_return,
                "final_d": final_d,
                "d_at_done": d_at_done,
                "min_d": min_d,
                "criterion_i_prime_lb": metric_lb,
                "criterion_i_prime_pass": significant,
                "gap_p95": gap,
                "td_target_clamp_hit_frac": clamp,
                "nan_incidents_corrected": nan_total,
                "magnitude_incidents_corrected": magnitude_total,
                "full_scene_rebuilds": run.get("full_scene_rebuilds"),
                "per_template_success": ev.get("per_template_success", {}),
            }
            tinfo["seeds"][seed] = seed_info
            clamp_cell = "—" if clamp is None else f"{clamp['tail_mean']:.4f} ({'nonzero' if clamp['steady_nonzero'] else 'zero'})"
            gap_cell = "—" if gap["max"] is None else f"{gap['bounded']}/{gap['max']:.4f}"
            lines.append(
                f"| {seed} | complete | {final_success:.3f} | {final_return:.3f} | {final_d:.4f} | "
                f"{d_at_done:.4f} | {min_d:.4f} | {metric_lb:+.4f} | "
                f"{'pass' if significant else 'fail'} | {gap_cell} | {clamp_cell} |"
            )
        lines.append("")

        if len(finals) >= 2:
            std_pp = float(np.std(finals) * 100)
            tinfo["final_success_std_pp"] = std_pp
            tinfo["criterion_ii_std_lt_15pp"] = bool(std_pp < 15.0)
            lines.append(
                f"기준 (ii): 최종 성공률 std = {std_pp:.2f}%p (완료 {len(finals)} seeds) — "
                f"{'충족' if std_pp < 15.0 else '미달'} (< 15%p)"
            )
        else:
            tinfo["criterion_ii_std_lt_15pp"] = None
            lines.append(f"기준 (ii): 완료 seed {len(finals)}개 — preliminary 산출 불가")
        if task == "t1a":
            if finals:
                tinfo["criterion_iii_t1a_ge_70pct"] = bool(max(finals) >= 0.70)
                lines.append(
                    f"기준 (iii): t1a 최종 성공률 max = {max(finals)*100:.0f}% — 70% 기대 "
                    f"{'충족' if max(finals) >= 0.70 else '미달 (사실 기록)'}"
                )
            else:
                tinfo["criterion_iii_t1a_ge_70pct"] = None
                lines.append("기준 (iii): t1a 완료 seed 없음 — factual record 보류")
        lines.append("")

        task_gap_bounded_value = bool(task_gap_bounded) and all(task_gap_bounded)
        payload["p_b"][task] = {
            "overestimation_gap_p95_bounded": task_gap_bounded_value,
            "overestimation_gap_p95_max": float(max(task_gap_maxes)) if task_gap_maxes else None,
        }

        lines.append("### within-episode min-D / d_at_done distribution by template")
        lines.append("")
        lines.append("| template | n | policy final D mean | policy d_at_done mean | policy min-D mean | min-D p10/p50/p90 | O1 ON d_at_done/min-D | O1 OFF d_at_done/min-D |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        template_stats = per_template_stats(combined_episodes)
        for template in TEMPLATES:
            stats = template_stats.get(template)
            if stats is None:
                lines.append(
                    f"| {template} | 0 | — | — | — | — | "
                    f"{oracle_cell(oracle, task, template, 'grasp_realism_on', 'mean_d_at_done')}/{oracle_cell(oracle, task, template, 'grasp_realism_on', 'mean_min_d')} | "
                    f"{oracle_cell(oracle, task, template, 'grasp_realism_off', 'mean_d_at_done')}/{oracle_cell(oracle, task, template, 'grasp_realism_off', 'mean_min_d')} |"
                )
                continue
            tinfo["template_min_d"][template] = stats
            lines.append(
                f"| {template} | {stats['n']} | {stats['mean_final_d']:.4f} | {stats['mean_d_at_done']:.4f} | {stats['mean_min_d']:.4f} | "
                f"{stats['min_d_p10']:.4f}/{stats['min_d_p50']:.4f}/{stats['min_d_p90']:.4f} | "
                f"{oracle_cell(oracle, task, template, 'grasp_realism_on', 'mean_d_at_done')}/{oracle_cell(oracle, task, template, 'grasp_realism_on', 'mean_min_d')} | "
                f"{oracle_cell(oracle, task, template, 'grasp_realism_off', 'mean_d_at_done')}/{oracle_cell(oracle, task, template, 'grasp_realism_off', 'mean_min_d')} |"
            )
        lines.append("")
        plot_path = plot_task(task, runs[task], random_success, random_return)
        tinfo["curve_path"] = plot_path
        lines.append(f"학습 곡선: `{plot_path}`")
        lines.append("")

    lines.append("## P-a — training-level halt count")
    lines.append("")
    lines.append(
        f"target 0; observed halt records = {len(halted)}. "
        f"m3r_t1a_s1 status = {payload['p_a']['m3r_t1a_s1_status']} (completion highlighted)."
    )
    lines.append("")
    lines.append("| run (archive) | halt 시점 tr | updates | halt_reason | gap p95 series | 보존 ckpt |")
    lines.append("|---|---:|---:|---|---|---|")
    if halted:
        for row in halted:
            gaps = " → ".join("—" if finite_float(v) is None else f"{float(v):.3g}" for v in row["gap_p95_series"]) or "eval 미도달"
            lines.append(
                f"| {row['run_tag']} ({row['archive']}) | {row['transitions_at_halt']:,} | "
                f"{row['updates_at_halt']:,} | {row['halt_reason']} | {gaps} | {row['last_checkpoint'] or '없음'} |"
            )
    else:
        lines.append("| — | 0 | 0 | none | — | — |")
    lines.append("")

    lines.append("## P-b — overestimation gap p95 boundedness")
    lines.append("")
    lines.append("bounded := all `np.isfinite(overestimation_gap_p95)` across evals; max reported per task.")
    lines.append("")
    lines.append("| task | bounded | max gap p95 |")
    lines.append("|---|---|---:|")
    for task in TASKS:
        row = payload["p_b"][task]
        max_cell = "—" if row["overestimation_gap_p95_max"] is None else f"{row['overestimation_gap_p95_max']:.4f}"
        lines.append(f"| {task} | {row['overestimation_gap_p95_bounded']} | {max_cell} |")
    lines.append("")
    lines.append("| run | p95 series | bounded | max gap p95 |")
    lines.append("|---|---|---|---:|")
    any_gap_series = False
    for task in TASKS:
        for seed, run in runs[task].items():
            any_gap_series = True
            tag = run_tag(task, seed)
            gap = gap_p95_summary(run)
            series = " → ".join(f"{value:.4f}" for value in gap["series"]) or "—"
            max_cell = "—" if gap["max"] is None else f"{gap['max']:.4f}"
            lines.append(f"| {tag} | {series} | {gap['bounded']} | {max_cell} |")
    if not any_gap_series:
        lines.append("| — | — | — | — |")
    lines.append("")

    lines.append("## P-c — oracle feasibility reference interpretation")
    lines.append("")
    lines.append(f"> {P_C_INTERPRETATION_RULE}")
    lines.append("")
    lines.append(
        "O1 oracle reference loaded: "
        f"{oracle.get('loaded')} (`{oracle.get('path')}`). Oracle 성공은 feasibility reference이며 upper bound가 아니다."
    )
    lines.append("")

    lines.append("## TD-target clamp hit-rate reading")
    lines.append("")
    lines.append("사전 등록 해석: nonzero steady rate = evidence FOR intrinsic-explosion antithesis.")
    lines.append("")
    lines.append("| run | n | max | final | tail mean | tail series | reading |")
    lines.append("|---|---:|---:|---:|---:|---|---|")
    any_clamp = False
    for task in TASKS:
        for seed in SEEDS:
            tag = run_tag(task, seed)
            summary = clamp_summary(diag_path(task, seed))
            if summary is None:
                continue
            any_clamp = True
            tail_series = " → ".join(f"{value:.4f}" for value in summary["series_tail"])
            lines.append(
                f"| {tag} | {summary['n']} | {summary['max']:.4f} | {summary['final']:.4f} | "
                f"{summary['tail_mean']:.4f} | {tail_series} | "
                f"{'FOR antithesis' if summary['steady_nonzero'] else 'zero steady'} |"
            )
    if not any_clamp:
        lines.append("| — | 0 | — | — | — | — | no completed diag clamp series |")
    lines.append("")

    lines.append("## Stability — NaN vs magnitude incidents")
    lines.append("")
    lines.append("rebuild-reset corrected from diag counter series; `—` means the diag series/file is absent.")
    lines.append("")
    lines.append("| run | nan incidents | magnitude incidents | full rebuilds |")
    lines.append("|---|---:|---:|---:|")
    any_stability = False
    for task in TASKS:
        for seed, run in runs[task].items():
            any_stability = True
            nan_total = corrected_counter_total(diag_path(task, seed), "nan_incidents", "nan_incidents")
            magnitude_total = corrected_counter_total(diag_path(task, seed), "magnitude_incidents", "magnitude_incidents")
            nan_cell = "—" if nan_total is None else str(nan_total)
            mag_cell = "—" if magnitude_total is None else str(magnitude_total)
            lines.append(f"| {run_tag(task, seed)} | {nan_cell} | {mag_cell} | {run.get('full_scene_rebuilds', '—')} |")
    if not any_stability:
        lines.append("| — | — | — | — |")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- reward constants unadjusted — α=10, c_step=0.1, R_succ=5; no reward/capacity/HER change is introduced by this report.")
    lines.append("- schema-debt: HDF5 v3: termination-cause 필드 — M6 P2 승계 추적 항목")
    lines.append("- hygiene: old p1_t1_report halted-glob would match m3r halted files if re-run.")
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    JSON_PATH.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(
        f"wrote {REPORT_PATH} (+json, plots); preliminary={bool(missing)}; "
        f"run files present={present}/{EXPECTED_RUNS}; completed={completed}/{EXPECTED_RUNS}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
