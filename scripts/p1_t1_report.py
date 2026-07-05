"""P1-M3 T1 results report per the pre-registered criteria (issue #12).

Aggregates committed per-run artifacts (no re-simulation):
    (i)   per task, per seed: episode-level bootstrap (B=10,000, percentile,
          CI95 lower bound = empirical 5th percentile, fixed seed 20260703 —
          same register as M2) on the SUCCESS-RATE difference and the
          MEAN-RETURN difference between the run's FINAL eval episodes and
          the task's random-reference episodes;
    (ii)  per task: std of final success rate across completed seeds < 15%p;
    (iii) t1a success >= 70% expectation — factual record only;
    plus: per-template decomposition (risk #5), learning curves
    (mean +/- std over seeds), stability summary (env NaN incidents with
    rebuild-reset correction from diag histories, full rebuilds,
    training-level halts from archived artifacts), reward-constant status.

Outputs:
    outputs/reports/p1_t1_results.md
    outputs/metrics/p1_t1_results.json
    outputs/plots/p1_t1_curves_<task>.png
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BOOTSTRAP_SEED = 20260703
BOOTSTRAP_B = 10_000
TASKS = {
    "t1a": "t1a_straighten",
    "t1b": "t1b_single_bend",
    "t1c": "t1c_endpoint_reposition",
}
SEEDS = (0, 1, 2)
TEMPLATES = ("straight", "u_bend", "s_curve", "random_smooth")


def bootstrap_diff_lb(treatment: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """(bootstrap mean diff, empirical 5th percentile) — register method."""

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    diffs = np.empty(BOOTSTRAP_B)
    for i in range(BOOTSTRAP_B):
        t = rng.choice(treatment, size=len(treatment), replace=True)
        r = rng.choice(reference, size=len(reference), replace=True)
        diffs[i] = t.mean() - r.mean()
    return float(diffs.mean()), float(np.percentile(diffs, 5))


def corrected_nan_total(diag_path: Path) -> int | None:
    """Total env NaN incidents across rebuild-reset segments (STEP_LOG caveat).

    The runner counter resets to 0 on full scene rebuild; the diag series
    preserves the pre-reset maxima.  Total = sum of segment maxima.
    """

    if not diag_path.exists():
        return None
    series = json.loads(diag_path.read_text(encoding="utf-8")).get("nan_incidents", [])
    values = [int(row["nan_incidents"]) for row in series]
    if not values:
        return 0
    total, prev = 0, 0
    for v in values:
        if v < prev:  # reset boundary — bank the finished segment
            total += prev
        prev = v
    return total + prev


def load_runs() -> tuple[dict, list[dict]]:
    """Return ({task: {seed: run_json}}, halted_records)."""

    runs: dict[str, dict[int, dict]] = {k: {} for k in TASKS}
    for task in TASKS:
        for seed in SEEDS:
            p = Path(f"outputs/metrics/p1_run_{task}_s{seed}.json")
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8"))
                if d["transitions"] >= d["total_budget"] and d["halt_reason"] is None:
                    runs[task][seed] = d
    halted = []
    for p in sorted(glob.glob("outputs/metrics/p1_run_*halted*.json")):
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        m = re.search(r"p1_run_(t1[abc]_s\d)\.(.+)\.json", Path(p).name)
        halted.append(
            {
                "run_tag": m.group(1) if m else Path(p).name,
                "archive": m.group(2) if m else "",
                "transitions_at_halt": d["transitions"],
                "updates_at_halt": d["updates"],
                "halt_reason": d["halt_reason"],
                "last_checkpoint": d["last_checkpoint"],
                "gap_trajectory": [float(ev["overestimation_gap_mean"]) for ev in d["evals"]],
            }
        )
    return runs, halted


def main() -> int:
    parser = argparse.ArgumentParser(description="P1-M3 T1 results report")
    parser.add_argument("--preliminary", action="store_true", help="allow missing runs")
    args = parser.parse_args()

    ref = json.loads(Path("outputs/metrics/p1_random_reference.json").read_text(encoding="utf-8"))
    runs, halted = load_runs()

    missing = [f"{t}_s{s}" for t in TASKS for s in SEEDS if s not in runs[t]]
    if missing and not args.preliminary:
        raise SystemExit(f"missing/incomplete runs {missing}; use --preliminary to draft")

    payload: dict = {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "B": BOOTSTRAP_B,
        "preliminary": bool(missing),
        "missing_runs": missing,
        "tasks": {},
        "halted_runs": halted,
        "reward_constants_adjusted": False,
    }
    lines: list[str] = []
    lines.append("# P1-M3 — T1 기준 성능 리포트 (3 tasks × 3 seeds × 1e5 transitions)")
    lines.append("")
    if missing:
        lines.append(f"> **PRELIMINARY DRAFT** — 미완료 run: {', '.join(missing)}")
        lines.append("")
    lines.append(
        "사전 등록 기준 (issue #12): (i) 전 seed random 대비 개선 유의 (episode-level "
        f"bootstrap B={BOOTSTRAP_B}, seed {BOOTSTRAP_SEED}, CI95 하한=경험적 5퍼센타일), "
        "(ii) seed 간 최종 성공률 std < 15%p, (iii) t1a ≥ 70% 기대 (미달 시 사실 기록). "
        "성공률 diff와 return diff를 모두 보고한다 — random 성공률이 0%인 task에서 성공률 "
        "bootstrap은 정보량이 없으므로 (기준 문구의 한계, 게이트 해석 필요) return diff를 병기."
    )
    lines.append("")

    for task, block_name in TASKS.items():
        block = ref["blocks"][block_name]
        rand_succ = np.asarray([1.0 if ep["success"] else 0.0 for ep in block["episodes"]])
        rand_ret = np.asarray([ep["return"] for ep in block["episodes"]], dtype=float)
        tinfo: dict = {"random_success": float(rand_succ.mean()), "random_return": float(rand_ret.mean()), "seeds": {}}
        lines.append(f"## {block_name}")
        lines.append("")
        lines.append(
            f"random 참조선: success {rand_succ.mean():.3f}, return {rand_ret.mean():.3f} "
            f"(n={len(rand_ret)})"
        )
        lines.append("")
        lines.append(
            "| seed | final success | final return | final D | succ diff LB(5%) | ret diff LB(5%) | gap 궤적 |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        finals = []
        for seed in SEEDS:
            if seed not in runs[task]:
                lines.append(f"| {seed} | — 미완료 ({task}_s{seed}) | | | | | |")
                continue
            d = runs[task][seed]
            eps = d["eval_episodes"][-1]["episodes"]
            succ = np.asarray([1.0 if ep["success"] else 0.0 for ep in eps])
            rets = np.asarray([ep["return"] for ep in eps], dtype=float)
            _, s_lb = bootstrap_diff_lb(succ, rand_succ)
            _, r_lb = bootstrap_diff_lb(rets, rand_ret)
            ev = d["evals"][-1]
            gaps = " → ".join(f"{e['overestimation_gap_mean']:.2f}" for e in d["evals"])
            finals.append(float(ev["success_rate"]))
            tinfo["seeds"][seed] = {
                "final_success": ev["success_rate"],
                "final_return": ev["mean_return"],
                "final_d": ev["mean_final_d"],
                "success_diff_lb": s_lb,
                "return_diff_lb": r_lb,
                "success_diff_significant": bool(s_lb > 0),
                "return_diff_significant": bool(r_lb > 0),
                "gap_trajectory": [float(e["overestimation_gap_mean"]) for e in d["evals"]],
                "per_template_success": ev["per_template_success"],
                "wall_h_source_log": None,
                "full_scene_rebuilds": d["full_scene_rebuilds"],
                "nan_incidents_corrected": corrected_nan_total(
                    Path(f"outputs/metrics/p1_diag_{task}_s{seed}.json")
                ),
            }
            lines.append(
                f"| {seed} | {ev['success_rate']:.3f} | {ev['mean_return']:.3f} | "
                f"{ev['mean_final_d']:.4f} | {s_lb:+.4f} | {r_lb:+.3f} | {gaps} |"
            )
        lines.append("")
        if len(finals) >= 2:
            std_pp = float(np.std(finals) * 100)
            tinfo["final_success_std_pp"] = std_pp
            tinfo["criterion_ii_std_lt_15pp"] = bool(std_pp < 15.0)
            lines.append(
                f"기준 (ii): 최종 성공률 std = {std_pp:.2f}%p (완료 {len(finals)} seeds) — "
                f"{'충족' if std_pp < 15 else '미달'} (< 15%p)"
            )
        if task == "t1a" and finals:
            tinfo["criterion_iii_t1a_ge_70pct"] = bool(max(finals) >= 0.70)
            lines.append(
                f"기준 (iii): t1a 최종 성공률 max = {max(finals)*100:.0f}% — 70% 기대 "
                f"{'충족' if max(finals) >= 0.7 else '**미달 (사실 기록)**'}"
            )
        lines.append("")
        # Per-template decomposition (risk #5).
        lines.append("### per-template 최종 성공률 분해 (리스크 #5)")
        lines.append("")
        lines.append("| seed | " + " | ".join(TEMPLATES) + " |")
        lines.append("|---|" + "---|" * len(TEMPLATES))
        for seed in SEEDS:
            if seed not in runs[task]:
                continue
            pt = runs[task][seed]["evals"][-1]["per_template_success"]
            lines.append(
                f"| {seed} | " + " | ".join(f"{pt.get(t, 0.0):.2f}" for t in TEMPLATES) + " |"
            )
        lines.append("")
        payload["tasks"][task] = tinfo

        # Learning-curve plot: mean +/- std over completed seeds.
        if runs[task]:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            xs = sorted({e["transitions"] for d in runs[task].values() for e in d["evals"]})
            for metric, ax, label in (
                ("mean_return", axes[0], "eval mean return"),
                ("success_rate", axes[1], "eval success rate"),
            ):
                mat = np.full((len(runs[task]), len(xs)), np.nan)
                for i, d in enumerate(runs[task].values()):
                    for e in d["evals"]:
                        mat[i, xs.index(e["transitions"])] = e[metric]
                mean = np.nanmean(mat, axis=0)
                std = np.nanstd(mat, axis=0)
                ax.plot(xs, mean, marker="o")
                ax.fill_between(xs, mean - std, mean + std, alpha=0.25)
                ref_v = float(rand_ret.mean()) if metric == "mean_return" else float(rand_succ.mean())
                ax.axhline(ref_v, color="gray", linestyle="--", linewidth=1, label="random ref")
                ax.set_xlabel("transitions")
                ax.set_title(f"{block_name} — {label} (mean±std, {len(runs[task])} seeds)")
                ax.legend()
            fig.tight_layout()
            out_png = Path(f"outputs/plots/p1_t1_curves_{task}.png")
            out_png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_png, dpi=120)
            plt.close(fig)
            lines.append(f"학습 곡선: `outputs/plots/p1_t1_curves_{task}.png`")
            lines.append("")

    # Stability summary.
    lines.append("## 안정성 요약")
    lines.append("")
    lines.append("### 학습 레벨 halt (rule 6) — 전건 non-finite critic gradient norm, 갭 폭주 선행")
    lines.append("")
    lines.append("| run (archive) | halt 시점 tr | updates | gap 궤적 (eval별) | 보존 ckpt |")
    lines.append("|---|---|---|---|---|")
    for h in halted:
        gaps = " → ".join(f"{g:.3g}" for g in h["gap_trajectory"]) or "eval 미도달"
        lines.append(
            f"| {h['run_tag']} ({h['archive']}) | {h['transitions_at_halt']:,} | "
            f"{h['updates_at_halt']:,} | {gaps} | {h['last_checkpoint'] or '없음'} |"
        )
    lines.append("")
    lines.append(
        "완주 run의 env-레벨 NaN incident (rebuild 리셋 보정 합산)과 full rebuild는 각 task "
        "표의 per-seed 필드 및 `p1_t1_results.json` 참조. env 레벨 incident는 전건 NaN "
        "covenant (폐기+재시드)로 회복되었고, 데이터 오염 없음 (replay 유입은 isfinite 게이트 통과분만)."
    )
    lines.append("")
    lines.append("## reward 상수")
    lines.append("")
    lines.append(
        "조정 없음 — α=10, c_step=0.1, R_succ=5 (P0 issue #8 시작값 그대로; 전역 규칙 4 미발동)."
    )
    lines.append("")

    Path("outputs/reports/p1_t1_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path("outputs/metrics/p1_t1_results.json").write_text(
        json.dumps(payload, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote outputs/reports/p1_t1_results.md (+json, plots); preliminary={bool(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
