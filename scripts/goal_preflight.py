"""Goal stability preflight (report-only) — sprint directive 2.

Authority: DGCC#13 comment 4985559491 directive 2. Scope-order rule (leakage
guard): T1b/T1c samples + T2 VAL 50 (+ optionally the sprint held-out split,
which is NOT the M4 held-out). The M4 held-out 100 must not be loaded here;
its preflight completes only after the M4 final held-out evaluation.

Per goal: goal_curve -> MANDATORY z-alignment (analytic_init_centerline floor
idiom: remove z-min, raise to rest height) -> place -> settle(max 10000) ->
drift = correspondence_l2(goal, settled, shape_only=True) + anchor delta
separately. Chamfer/distance.D is covenant-forbidden and not used. Gated by
the settle-converged mask (non-converged goals tallied separately).

Fixed interpretation framing (instruction verbatim): drift is the elastic
relaxation distance toward the straight-rest (kappa_rest=0) equilibrium — it
does NOT mean "the goal is bad". Measurement only; goal definitions unchanged.

GPU required — run ONLY in a seed-boundary window.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from dgcc.envs.dlolab import DLOLabEnv
from dgcc.goals.distance import correspondence_l2
from dgcc.goals.dual_goal import goal_curve
from dgcc.phi.resample import resample
from dgcc.tasks.domain import P1_LENGTH_M, P1_RADIUS, SETTLE_MAX_STEPS, p1_rope_params
from dgcc.tasks.t1 import sample_t1_goal

EPS_SUCC = 0.05  # * L (prereg immutable)


def z_align(curve: np.ndarray) -> np.ndarray:
    """analytic_init_centerline floor idiom: z-min removed, raised to rest height."""

    out = curve.copy()
    out[:, 2] -= out[:, 2].min()
    out[:, 2] += P1_RADIUS
    return out


def _to_native(curve32: np.ndarray, n_native: int) -> np.ndarray:
    """Arc-length linear interpolation of the 32-node goal curve to the
    env's native vertex count (dgcc.phi.resample is 32-node-only)."""

    seg = np.linalg.norm(np.diff(curve32, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s /= s[-1]
    t = np.linspace(0.0, 1.0, n_native)
    return np.column_stack([np.interp(t, s, curve32[:, k]) for k in range(3)])


def measure(env, params, curve32: np.ndarray, length_m: float = P1_LENGTH_M) -> dict:
    n_native = env.get_centerline_raw_batch().shape[1]
    placed = z_align(_to_native(curve32, n_native))
    env.place_rod_vertices_batch(placed[None])
    converged = env.settle(max_steps=SETTLE_MAX_STEPS)
    settled = env.get_centerline_batch()[0]
    drift_shape = correspondence_l2(settled, curve32, length_m, shape_only=True)
    anchor_delta = float(np.linalg.norm(settled.mean(axis=0) - curve32.mean(axis=0)))
    return {"converged": bool(converged), "drift_shape": float(drift_shape), "anchor_delta": anchor_delta}

def patch_eval_lengths(payload: dict) -> tuple[float, ...]:
    """Return the train reference and both committed OOD rope lengths."""
    transform = payload["ood_length_transform"]
    train_length = float(transform["l_train_m"])
    ood_lengths = tuple(float(length) for length in transform["l_ood_m"])
    if len(ood_lengths) != 2 or set(ood_lengths) & {train_length}:
        raise ValueError("patch split must define two OOD lengths distinct from L_train")
    return tuple(sorted((train_length, *ood_lengths)))


def load_patch_eval_split(path: Path | None = None) -> tuple[dict, list[dict], bool]:
    """Load the committed patch split without touching protected T2 splits."""
    split_path = path or REPO / "src/dgcc/tasks/splits/t2_patch_eval_v1.json"
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    specs = payload["specs"]
    if payload.get("n_goals") != 100 or len(specs) != 100:
        raise ValueError("t2_patch_eval_v1 must contain exactly 100 goals")
    patch_eval_lengths(payload)
    return payload, specs, False


def append_patch_breakdown(lines: list[str], rows: list[dict]) -> None:
    """Append converged-only length × family drift results for the patch split."""
    lines.append("| rope length (m) | family | n | drift median | drift max | >eps | non-converged |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    lengths = sorted({float(row["rope_length_m"]) for row in rows})
    families = sorted({str(row["template"]) for row in rows})
    for length in lengths:
        for family in families:
            group = [row for row in rows if row["rope_length_m"] == length and row["template"] == family]
            conv = [row for row in group if row["converged"]]
            drifts = np.array([row["drift_shape"] for row in conv])
            over = int((drifts > EPS_SUCC).sum()) if len(drifts) else 0
            median = f"{np.median(drifts):.4f}" if len(drifts) else "—"
            maximum = f"{drifts.max():.4f}" if len(drifts) else "—"
            lines.append(
                f"| {length:.2f} | {family} | {len(conv)} | {median} | {maximum} | {over} | {len(group) - len(conv)} |"
            )
    lines.append("")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1-samples", type=int, default=25, help="sampled goals per T1 task (t1b/t1c)")
    parser.add_argument("--include-sprint-split", action="store_true",
                        help="also preflight t2_sprint_heldout_v1 (NOT the M4 held-out); requires --lock")
    parser.add_argument("--lock", type=Path, default=None,
                        help="canonical issued metric lock; mandatory gate for --include-sprint-split")
    parser.add_argument("--include-patch-split", action="store_true",
                        help="also preflight t2_patch_eval_v1 at its train/OOD rope lengths")
    parser.add_argument("--report", type=Path, default=Path("outputs/reports/goal_preflight.md"))
    parser.add_argument("--json", type=Path, default=Path("outputs/metrics/goal_preflight.json"))
    args = parser.parse_args()
    if args.include_sprint_split:
        # Fail-closed before any environment work: the sprint split is unreachable
        # without the canonical issued metric lock (non-BB caller boundary).
        from dgcc.analysis.sprint_claims import require_metric_lock
        require_metric_lock(args.lock, "v1")  # strictest non-BB gate; preflight is arm-agnostic

    from dgcc.tasks.t2 import load_t2_split

    t0 = time.time()
    params = p1_rope_params()
    env = DLOLabEnv(n_envs=1, dt=0.001, substeps=5, rod_damping=10.0, rod_angular_damping=5.0)
    env.reset(params, init_shape="straight", seed=910_000)

    blocks: dict[str, list[dict]] = {}
    rng = np.random.default_rng(20260716)

    # T1b / T1c sampled goals (goal_fn needs a current state: use settled straight)
    base_state = env.get_centerline_batch()[0]
    for task in ("t1b_single_bend", "t1c_endpoint_reposition"):
        rows = []
        for i in range(args.t1_samples):
            goal = sample_t1_goal(task, base_state, np.random.default_rng(920_000 + i))
            curve = goal_curve(goal, P1_LENGTH_M)
            r = measure(env, params, curve)
            r["label"] = f"{task}#{i}"
            r["template"] = goal.template_name or task
            rows.append(r)
        blocks[task] = rows

    # T2 val 50 (leakage guard: val ONLY here)
    rows = []
    for spec, goal in load_t2_split("val"):
        curve = goal_curve(goal, P1_LENGTH_M)
        r = measure(env, params, curve)
        r["label"] = spec["goal_id"]
        r["template"] = str(spec["family"])
        rows.append(r)
    blocks["t2_val"] = rows

    if args.include_sprint_split:
        # Non-BB caller boundary: the sprint split may not be read without the issued metric lock.
        from dgcc.analysis.sprint_claims import require_metric_lock
        require_metric_lock(args.lock, "v1")  # strictest non-BB gate; preflight is arm-agnostic
        sprint = json.loads((REPO / "src/dgcc/tasks/splits/t2_sprint_heldout_v1.json").read_text())
        from dgcc.tasks.t2 import build_t2_goal
        rows = []
        for spec in sprint["specs"]:
            goal = build_t2_goal(spec)
            curve = goal_curve(goal, P1_LENGTH_M)
            r = measure(env, params, curve)
            r["label"] = spec["goal_id"]
            r["template"] = str(spec["family"])
            rows.append(r)
        blocks["t2_sprint_heldout"] = rows
    if args.include_patch_split:
        from dgcc.tasks.t2 import build_t2_goal

        patch_payload, patch_specs, access_recorded = load_patch_eval_split()
        rows = []
        for length_m in patch_eval_lengths(patch_payload):
            patch_params = replace(params, length_m=length_m)
            env.reset(patch_params, init_shape="straight", seed=910_000)
            for spec in patch_specs:
                goal = build_t2_goal(spec)
                curve = goal_curve(goal, length_m)
                r = measure(env, patch_params, curve, length_m)
                r["label"] = spec["goal_id"]
                r["template"] = str(spec["family"])
                r["rope_length_m"] = length_m
                rows.append(r)
        blocks["t2_patch_eval_v1"] = rows

    payload = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "framing": "drift = elastic relaxation distance to straight-rest equilibrium (NOT goal quality)",
               "eps_succ": EPS_SUCC, "wall_s": time.time() - t0, "blocks": blocks}
    if args.include_patch_split:
        payload["patch_split_access"] = {
            "loaded": True,
            "recorded": access_recorded,
            "purpose": "preflight",
        }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=1) + "\n")

    lines = ["# Goal stability preflight (report-only)", "",
             f"> Authority: DGCC#13 4985559491 directive 2 · generated {payload['generated_at']} · wall {payload['wall_s']:.0f}s",
             "> Framing (fixed): drift = elastic relaxation toward straight-rest (kappa_rest=0) equilibrium — measurement only, goal definitions unchanged.",
             "> Metric: correspondence_l2(goal, settled, shape_only=True) + anchor delta; Chamfer forbidden. Converged-mask gated.",
             "> Leakage guard: T2 val only; M4 held-out preflight completes after the M4 final held-out evaluation.", ""]
    if args.include_patch_split:
        lines.append(
            "> Patch split loaded for purpose=preflight; no dedicated record_access path is available."
        )
        lines.append("")
    for name, rows in blocks.items():
        conv = [r for r in rows if r["converged"]]
        nonconv = len(rows) - len(conv)
        drifts = np.array([r["drift_shape"] for r in conv]) if conv else np.array([])
        over = int((drifts > EPS_SUCC).sum()) if len(drifts) else 0
        lines.append(f"## {name} — n={len(rows)} (converged {len(conv)}, non-converged {nonconv})")
        if len(drifts):
            lines.append(f"- drift_shape: median {np.median(drifts):.4f} · p90 {np.quantile(drifts, 0.9):.4f} · max {drifts.max():.4f}")
            lines.append(f"- drift > eps(0.05): **{over}/{len(conv)}** ({over/len(conv):.0%})")
            per_t: dict[str, list[float]] = {}
            for r in conv:
                per_t.setdefault(r["template"], []).append(r["drift_shape"])
            lines.append("| template | n | drift median | drift max | >eps |")
            lines.append("|---|---:|---:|---:|---:|")
            for t in sorted(per_t):
                arr = np.array(per_t[t])
                lines.append(f"| {t} | {len(arr)} | {np.median(arr):.4f} | {arr.max():.4f} | {(arr > EPS_SUCC).sum()} |")
        lines.append("")
        if name == "t2_patch_eval_v1":
            append_patch_breakdown(lines, rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")
    print(f"preflight report: {args.report} (wall {payload['wall_s']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
