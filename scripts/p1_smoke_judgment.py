"""P1-M2 smoke judgment per the frozen §3 statistical gate register.

Computes the pre-registered criteria from committed per-episode artifacts
(no re-simulation):
    (i)  one-sided episode-level bootstrap (B=10,000, percentile method,
         CI95 lower bound = empirical 5th percentile) on the mean-return
         difference between the smoke run's FINAL eval episodes and the
         random-reference T1-a episodes, AND non-negative least-squares slope
         of mean eval return vs transitions over all eval points;
    (ii) no training-level NaN/divergence halt;
    (iii) overestimation-gap trend (record only, no threshold).

Writes ``outputs/metrics/p1_smoke_judgment.json``.  Bootstrap seed is fixed
and logged (register: "all bootstrap seeds fixed and logged").
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BOOTSTRAP_SEED = 20260703
BOOTSTRAP_B = 10_000


def bootstrap_diff_lb(
    treatment: np.ndarray,
    reference: np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    b: int = BOOTSTRAP_B,
) -> tuple[float, float]:
    """Return (mean diff of bootstrap distribution, empirical 5th percentile)."""

    rng = np.random.default_rng(seed)
    diffs = np.empty(b)
    for i in range(b):
        t = rng.choice(treatment, size=len(treatment), replace=True)
        r = rng.choice(reference, size=len(reference), replace=True)
        diffs[i] = t.mean() - r.mean()
    return float(diffs.mean()), float(np.percentile(diffs, 5))


def main() -> int:
    parser = argparse.ArgumentParser(description="P1-M2 registered smoke judgment")
    parser.add_argument("--run-json", type=Path, default=Path("outputs/metrics/p1_run_t1a_smoke_s0.json"))
    parser.add_argument("--reference-json", type=Path, default=Path("outputs/metrics/p1_random_reference.json"))
    parser.add_argument("--reference-block", type=str, default="t1a_straighten")
    parser.add_argument("--out", type=Path, default=Path("outputs/metrics/p1_smoke_judgment.json"))
    args = parser.parse_args()

    run = json.loads(args.run_json.read_text(encoding="utf-8"))
    ref = json.loads(args.reference_json.read_text(encoding="utf-8"))

    final_block = run["eval_episodes"][-1]
    smoke_returns = np.asarray([ep["return"] for ep in final_block["episodes"]], dtype=float)
    rand_returns = np.asarray(
        [ep["return"] for ep in ref["blocks"][args.reference_block]["episodes"]], dtype=float
    )

    diff_mean, lb = bootstrap_diff_lb(smoke_returns, rand_returns)

    evals = [(ev["transitions"], ev["mean_return"]) for ev in run["evals"]]
    x = np.asarray([e[0] for e in evals], dtype=float)
    y = np.asarray([e[1] for e in evals], dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])

    criterion_i = bool(lb > 0 and slope >= 0)
    payload = {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "B": BOOTSTRAP_B,
        "smoke_mean_return": float(smoke_returns.mean()),
        "random_mean_return": float(rand_returns.mean()),
        "diff_mean": diff_mean,
        "ci95_lower_bound_5th_pct": lb,
        "slope_per_transition": slope,
        "criterion_i": criterion_i,
        "criterion_ii_no_divergence": run["halt_reason"] is None,
        "criterion_iii_overest_gap_trend_record_only": [
            float(ev["overestimation_gap_mean"]) for ev in run["evals"]
        ],
        "eval_points": evals,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(json.dumps(payload, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
