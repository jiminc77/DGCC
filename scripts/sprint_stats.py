#!/usr/bin/env python3
"""Preregistered sprint statistics.

This deliberately uses local NumPy/SciPy rather than rliable: rliable adds five
packages and its resampling unit does not match this experiment's paired seed
clusters.  The primary interval is BCa (never percentile bootstrap).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import norm, t

from dgcc.analysis.sprint_claims import PRIMITIVE_SCHEMA_VERSION, SprintClaimError, require_metric_lock

B = 10_000
RNG_SEED = 20260703
ALPHA = 0.05
SIGMA_GOAL = 1.8605  # eb439489: M4 held-out pooled per-goal return, ddof=1.
RETURN_EQUIV_MARGIN = 0.465  # eb439489: 0.25 * sigma_goal.
SUCCESS_EQUIV_MARGIN = 0.05


def _finite(values: Sequence[float]) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError("statistic requires a nonempty finite one-dimensional sample")
    return a


def seed_differences(v1: Mapping[int, Sequence[float]], bb: Mapping[int, Sequence[float]]) -> np.ndarray:
    """Paired seed effects; each sequence is its seed's goal×episode block."""
    if set(v1) != set(bb):
        raise ValueError("V1 and BB must have identical paired seed sets")
    return np.asarray([_finite(v1[s]).mean() - _finite(bb[s]).mean() for s in sorted(v1)], dtype=float)


def _bca_quantiles(observed: float, bootstrap: np.ndarray, jackknife: np.ndarray, alpha: float) -> tuple[float, float, float]:
    # half-count avoids +/- infinity when all bootstrap values fall on one side.
    z0 = norm.ppf((np.count_nonzero(bootstrap < observed) + 0.5) / (len(bootstrap) + 1.0))
    center = jackknife.mean()
    delta = center - jackknife
    denominator = 6.0 * np.sum(delta ** 2) ** 1.5
    acceleration = 0.0 if denominator == 0 else float(np.sum(delta ** 3) / denominator)
    def adjusted(p: float) -> float:
        z = norm.ppf(p)
        return float(norm.cdf(z0 + (z0 + z) / (1.0 - acceleration * (z0 + z))))
    return z0, acceleration, adjusted(alpha)


def bca_interval(observed: float, bootstrap: Sequence[float], jackknife: Sequence[float], *, alpha: float = ALPHA, two_sided: bool = False) -> dict[str, float]:
    """BCa interval. One-sided output is the preregistered 95% lower bound."""
    boot, jack = _finite(bootstrap), _finite(jackknife)
    z0, acceleration, lower_q = _bca_quantiles(float(observed), boot, jack, alpha / 2 if two_sided else alpha)
    lower = float(np.quantile(boot, lower_q))
    output = {"lower": lower, "z0": z0, "acceleration": acceleration, "lower_quantile": lower_q}
    if two_sided:
        _, _, upper_q = _bca_quantiles(float(observed), boot, jack, 1 - alpha / 2)
        output.update(upper=float(np.quantile(boot, upper_q)), upper_quantile=upper_q)
    return output


def seed_cluster_bootstrap(seed_effects: Sequence[float], *, draws: int = B, rng_seed: int = RNG_SEED, alpha: float = ALPHA) -> dict[str, Any]:
    """Paired seed-cluster BCa: resample entire paired seed blocks together."""
    effects = _finite(seed_effects)
    if np.all(effects == 0):
        return {"estimate": 0.0, "ci": [0.0, 0.0], "lower": 0.0, "degenerate": True, "trigger_return_endpoint": True, "method": "BCa seed-cluster"}
    rng = np.random.default_rng(rng_seed)
    samples = effects[rng.integers(0, len(effects), size=(draws, len(effects)))].mean(axis=1)
    jackknife = np.asarray([np.delete(effects, i).mean() for i in range(len(effects))])
    result = bca_interval(float(effects.mean()), samples, jackknife, alpha=alpha, two_sided=True)
    result.update(estimate=float(effects.mean()), ci=[result["lower"], result["upper"]], degenerate=False, trigger_return_endpoint=False, method="BCa seed-cluster")
    return result


def hierarchical_seed_cluster_bootstrap(v1: Mapping[int, Sequence[float]], bb: Mapping[int, Sequence[float]], *, draws: int = B, rng_seed: int = RNG_SEED, alpha: float = ALPHA) -> dict[str, Any]:
    """Sensitivity path: resample seed blocks, then within-seed goal×episode blocks."""
    if set(v1) != set(bb): raise ValueError("V1 and BB must have identical paired seed sets")
    seeds = sorted(v1)
    pairs = [( _finite(v1[s]), _finite(bb[s])) for s in seeds]
    if any(len(x) != len(y) for x, y in pairs): raise ValueError("paired within-seed blocks must have equal length")
    effects = np.array([x.mean() - y.mean() for x, y in pairs])
    rng = np.random.default_rng(rng_seed)
    values = np.empty(draws)
    for draw in range(draws):
        per_seed = []
        for index in rng.integers(0, len(seeds), size=len(seeds)):
            x, y = pairs[index]; chosen = rng.integers(0, len(x), size=len(x))
            per_seed.append((x[chosen] - y[chosen]).mean())
        values[draw] = np.mean(per_seed)
    jack = np.array([np.delete(effects, i).mean() for i in range(len(effects))])
    out = bca_interval(float(effects.mean()), values, jack, alpha=alpha, two_sided=True)
    out.update(estimate=float(effects.mean()), ci=[out["lower"], out["upper"]], method="BCa hierarchical seed then within-seed goal×episode")
    return out


def welch_seed_interval(seed_effects: Sequence[float], *, alpha: float = ALPHA) -> list[float]:
    values = _finite(seed_effects); n = len(values)
    if n < 2: raise ValueError("Welch interval requires at least two seed effects")
    se = values.std(ddof=1) / math.sqrt(n)
    q = t.ppf(1 - alpha / 2, n - 1)
    return [float(values.mean() - q * se), float(values.mean() + q * se)]


def iqm(values: Sequence[float]) -> float:
    a = np.sort(_finite(values)); return float(np.mean(a[math.floor(.25 * len(a)):math.ceil(.75 * len(a))]))


def holm_bonferroni(one_sided_p: Mapping[str, float], *, primary_passed: bool, alpha: float = ALPHA) -> dict[str, Any]:
    if set(one_sided_p) != {"2", "3"}: raise ValueError("Holm family must be exactly {'2', '3'}")
    if not primary_passed:
        return {key: {"status": "untested_primary_failed"} for key in one_sided_p}
    ordered = sorted(one_sided_p.items(), key=lambda row: row[1]); rejected = True; out = {}
    m = len(ordered)
    for rank, (name, p) in enumerate(ordered):
        if not 0 <= p <= 1: raise ValueError("p-values must be in [0, 1]")
        threshold = alpha / (m - rank); rejected = rejected and p <= threshold
        out[name] = {"p": p, "threshold": threshold, "reject": rejected}
    return out


def tost_paired(seed_effects: Sequence[float], margin: float, *, alpha: float = ALPHA) -> dict[str, Any]:
    values = _finite(seed_effects); n = len(values)
    if n < 2 or margin <= 0: raise ValueError("TOST requires n >= 2 and a positive margin")
    mean = float(values.mean()); se = values.std(ddof=1) / math.sqrt(n); q = t.ppf(1 - alpha, n - 1)
    ci = [mean - q * se, mean + q * se]
    equivalent = ci[0] > -margin and ci[1] < margin
    return {"n": n, "estimate": mean, "margin": margin, "ci90": ci, "equivalent": equivalent, "status": "provisional_directional", "limitation": "n=5 limitation: recovery at or below the equivalence margin cannot be excluded"}


def guard_sensitivity(guarded_failure: Sequence[float], excluded: Sequence[float], common_support: Sequence[float], v1_guard_rate: float, bb_guard_rate: float) -> dict[str, Any]:
    a, b, c = _finite(guarded_failure), _finite(excluded), _finite(common_support)
    return {"guarded_failure": seed_cluster_bootstrap(a), "excluded_nonrandom_dropout": seed_cluster_bootstrap(b), "common_support_auxiliary": seed_cluster_bootstrap(c), "guard_confounding": abs(v1_guard_rate - bb_guard_rate) > .02, "guard_rate_difference_percentage_points": 100 * (v1_guard_rate - bb_guard_rate)}


def primary_decision(seed_effects: Sequence[float], *, endpoint: str) -> dict[str, Any]:
    stat = seed_cluster_bootstrap(seed_effects)
    passed = not stat["degenerate"] and stat["lower"] > 0
    return {"endpoint": endpoint, "primary": stat, "welch_seed_interval": welch_seed_interval(seed_effects), "iqm_seed_effect": iqm(seed_effects), "iqm_not_standalone_decision": True, "state": "1_pass" if passed else "1_fail", "primary_passed": passed}


def bb_three_way_sensitivity(reuse3: Sequence[float], new5: Sequence[float], *, draws: int = B) -> dict[str, Any]:
    """Report BB reuse/new/pooled blocks; non-overlap marks a batch effect."""
    reuse, new = _finite(reuse3), _finite(new5)
    if len(reuse) != 3 or len(new) != 5:
        raise ValueError("BB sensitivity requires reuse3 and new5 seed blocks")
    old = seed_cluster_bootstrap(reuse, draws=draws)
    fresh = seed_cluster_bootstrap(new, draws=draws)
    pooled = seed_cluster_bootstrap(np.concatenate([reuse, new]), draws=draws)
    nonoverlap = old["ci"][1] < fresh["ci"][0] or fresh["ci"][1] < old["ci"][0]
    return {"reuse3": old, "new5": fresh, "pooled8": pooled, "batch_effect": nonoverlap}


def judgment_tree(primary: Mapping[str, Any], p_2: float, p_3: float) -> dict[str, Any]:
    """Gatekeeper: ②/③ remain untested after a ① failure."""
    passed = bool(primary.get("primary_passed"))
    return {
        "primary_state": "1_pass" if passed else "1_fail",
        "secondary": holm_bonferroni({"2": p_2, "3": p_3}, primary_passed=passed),
    }

def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f: return json.load(f)

def _claim_summary(path: Path) -> tuple[dict[str, Any], str]:
    payload = _load_json(path)
    if not isinstance(payload, dict): raise SprintClaimError("claim/result must be an object")
    return payload, hashlib.sha256(path.read_bytes()).hexdigest()

def _zero_assert(claims: list[Path]) -> None:
    root = claims[0].parent
    forbidden = []
    for pattern in ("*v1*result*", "*v1*claim*", "*v1*raw*", "*v1*purpose*", "*matched*", "*random*"):
        forbidden.extend(root.glob(pattern))
    if forbidden: raise SprintClaimError("metric lock refused: non-BB held-out artifact exists")

def publish_metric_lock(bb_claims: Sequence[Path], lock: Path) -> dict[str, Any]:
    """Fail closed and publish the exact G9a metric-lock schema using O_EXCL+fsync."""
    paths = [Path(p) for p in bb_claims]
    if len(paths) != 8: raise SprintClaimError("exactly eight BB claims are required")
    _zero_assert(paths)
    rows = [_claim_summary(p) for p in paths]
    payloads = [x[0] for x in rows]
    if any(str(x.get("arm", "")).lower() != "bb" for x in payloads): raise SprintClaimError("metric lock requires BB results/claims only")
    seeds = [x.get("seed") for x in payloads]
    if len(set(seeds)) != 8: raise SprintClaimError("BB claims must have eight unique seeds")
    summaries = [x.get("summary", x) for x in payloads]
    try:
        success = sum(float(x["success_rate"]) * float(x["n_episodes"]) for x in summaries) / sum(float(x["n_episodes"]) for x in summaries)
    except (KeyError, TypeError, ZeroDivisionError) as exc: raise SprintClaimError("BB results lack success-rate episode summaries") from exc
    endpoint = "return" if success <= .01 else "success_rate"
    body = {"schema_version": 1, "endpoint": endpoint, "aggregate": success, "created_at": datetime.now(timezone.utc).isoformat(), "bb_claim_sha256": [x[1] for x in rows], "primitive_version": str(PRIMITIVE_SCHEMA_VERSION)}
    target = Path(lock); target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(fd, (json.dumps(body, indent=1, sort_keys=True) + "\n").encode()); os.fsync(fd)
    finally: os.close(fd)
    directory = os.open(target.parent, os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)
    return body


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    lock = sub.add_parser("lock"); lock.add_argument("--bb-claims", nargs=8, required=True); lock.add_argument("--lock", required=True)
    judge = sub.add_parser("judge"); judge.add_argument("--lock", required=True); judge.add_argument("--seed-effects", required=True); judge.add_argument("--json", required=True); judge.add_argument("--md", required=True)
    args = parser.parse_args(argv)
    if args.command == "lock": print(json.dumps(publish_metric_lock(args.bb_claims, Path(args.lock)), indent=1)); return 0
    metric = _load_json(Path(args.lock)); require_metric_lock(Path(args.lock), "v1")
    effects = _load_json(Path(args.seed_effects)); decision = primary_decision(effects, endpoint=metric["endpoint"])
    practical = 10.0; ci = decision["primary"]["ci"]
    text = f"# Sprint primary judgment\n\nEndpoint: {metric['endpoint']}\n\nPreregistered practical threshold +10%p 대비 {decision['primary']['estimate'] * 100 - practical:.3f}%p [{ci[0] * 100:.3f}, {ci[1] * 100:.3f}].\n\n+10%p is a benchmark, not an AND gate. Return IQM benchmark is 0.5σ={SIGMA_GOAL / 2:.3f}.\n"
    Path(args.json).write_text(json.dumps(decision, indent=1) + "\n"); Path(args.md).write_text(text); return 0

if __name__ == "__main__": raise SystemExit(main())
