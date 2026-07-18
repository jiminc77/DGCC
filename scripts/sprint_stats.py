#!/usr/bin/env python3
"""Preregistered sprint statistics (paired seed clusters)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import norm, t

from dgcc.analysis import sprint_claims
from dgcc.analysis.sprint_claims import PRIMITIVE_SCHEMA_VERSION, SprintClaimError, require_metric_lock

B = 10_000
RNG_SEED = 20260703
ALPHA = 0.05
SIGMA_GOAL = 1.8605
RETURN_EQUIV_MARGIN = 0.465
SUCCESS_EQUIV_MARGIN = 0.05


def _finite(values: Sequence[float]) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError("statistic requires a nonempty finite one-dimensional sample")
    return a


def seed_differences(v1: Mapping[int, Sequence[float]], bb: Mapping[int, Sequence[float]]) -> np.ndarray:
    if set(v1) != set(bb):
        raise ValueError("V1 and BB must have identical paired seed sets")
    return np.asarray([_finite(v1[s]).mean() - _finite(bb[s]).mean() for s in sorted(v1)], dtype=float)


def _bca_adjusted_quantile(observed: float, bootstrap: np.ndarray, jackknife: np.ndarray, p: float) -> tuple[float, float, float]:
    if len(jackknife) < 2:
        raise ValueError("BCa requires at least two seed effects for jackknife acceleration")
    # Preregistered definition: z0 = Phi^-1(#(theta* < theta_hat) / B).
    # At k=0 or B this is infinite; clip only to the adjacent representable
    # probability, not a half-count correction, so the formula remains exact off
    # the unavoidable numerical boundary.
    proportion = np.count_nonzero(bootstrap < observed) / len(bootstrap)
    bounded = float(np.clip(proportion, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)))
    z0 = float(norm.ppf(bounded))
    delta = jackknife.mean() - jackknife
    denominator = 6.0 * np.sum(delta ** 2) ** 1.5
    acceleration = 0.0 if denominator == 0 else float(np.sum(delta ** 3) / denominator)
    z = norm.ppf(p)
    adjusted = float(norm.cdf(z0 + (z0 + z) / (1.0 - acceleration * (z0 + z))))
    return z0, acceleration, adjusted


def bca_lower_bound(observed: float, bootstrap: Sequence[float], jackknife: Sequence[float], *, alpha: float = ALPHA) -> dict[str, float]:
    """The sole primary-decision API: one-sided 95% BCa lower bound."""
    boot, jack = _finite(bootstrap), _finite(jackknife)
    z0, acceleration, quantile = _bca_adjusted_quantile(float(observed), boot, jack, alpha)
    return {"lower": float(np.quantile(boot, quantile)), "z0": z0, "acceleration": acceleration, "lower_quantile": quantile}


def bca_two_sided_interval(observed: float, bootstrap: Sequence[float], jackknife: Sequence[float], *, alpha: float = ALPHA) -> dict[str, float]:
    """Reporting-only two-sided BCa interval; never use this for primary gating."""
    boot, jack = _finite(bootstrap), _finite(jackknife)
    lower = bca_lower_bound(observed, boot, jack, alpha=alpha / 2)
    _, _, upper_q = _bca_adjusted_quantile(float(observed), boot, jack, 1 - alpha / 2)
    return {**lower, "upper": float(np.quantile(boot, upper_q)), "upper_quantile": upper_q}


def bca_interval(observed: float, bootstrap: Sequence[float], jackknife: Sequence[float], *, alpha: float = ALPHA, two_sided: bool = False) -> dict[str, float]:
    """Compatibility wrapper; new decisions must call the explicit APIs."""
    return bca_two_sided_interval(observed, bootstrap, jackknife, alpha=alpha) if two_sided else bca_lower_bound(observed, bootstrap, jackknife, alpha=alpha)


def _seed_bootstrap(effects: np.ndarray, draws: int, rng_seed: int) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    return effects[rng.integers(0, len(effects), size=(draws, len(effects)))].mean(axis=1)


def seed_cluster_bootstrap(seed_effects: Sequence[float], *, draws: int = B, rng_seed: int = RNG_SEED, alpha: float = ALPHA) -> dict[str, Any]:
    effects = _finite(seed_effects)
    if len(effects) < 2:
        raise ValueError("seed-cluster BCa requires at least two seed effects; n=1 has no jackknife acceleration")
    samples = _seed_bootstrap(effects, draws, rng_seed)
    jackknife = np.asarray([np.delete(effects, i).mean() for i in range(len(effects))])
    primary = bca_lower_bound(float(effects.mean()), samples, jackknife, alpha=alpha)
    reporting = bca_two_sided_interval(float(effects.mean()), samples, jackknife, alpha=alpha)
    return {"estimate": float(effects.mean()), "primary_lower": primary["lower"], "primary_bca": primary, "reporting_two_sided_bca": reporting, "ci": [reporting["lower"], reporting["upper"]], "degenerate": bool(np.all(effects == effects[0])), "method": "BCa seed-cluster"}


def iqm(values: Sequence[float]) -> float:
    a = np.sort(_finite(values))
    return float(np.mean(a[math.floor(.25 * len(a)):math.ceil(.75 * len(a))]))


def hierarchical_seed_cluster_bootstrap(v1: Mapping[int, Sequence[float]], bb: Mapping[int, Sequence[float]], *, draws: int = B, rng_seed: int = RNG_SEED, alpha: float = ALPHA) -> dict[str, Any]:
    """Sensitivity path: resample paired seeds then their within-seed blocks."""
    if set(v1) != set(bb): raise ValueError("V1 and BB must have identical paired seed sets")
    seeds = sorted(v1)
    pairs = [(_finite(v1[seed]), _finite(bb[seed])) for seed in seeds]
    if len(seeds) < 2 or any(len(left) != len(right) for left, right in pairs):
        raise ValueError("hierarchical BCa requires two paired seeds with equal within-seed blocks")
    effects = np.asarray([left.mean() - right.mean() for left, right in pairs])
    rng = np.random.default_rng(rng_seed)
    samples = np.empty(draws)
    for draw, selected_seeds in enumerate(rng.integers(0, len(seeds), size=(draws, len(seeds)))):
        samples[draw] = np.mean([
            (pairs[index][0][chosen] - pairs[index][1][chosen]).mean()
            for index in selected_seeds
            for chosen in [rng.integers(0, len(pairs[index][0]), size=len(pairs[index][0]))]
        ])
    jack = np.asarray([np.delete(effects, index).mean() for index in range(len(effects))])
    report = bca_two_sided_interval(float(effects.mean()), samples, jack, alpha=alpha)
    return {"estimate": float(effects.mean()), "ci": [report["lower"], report["upper"]], "reporting_two_sided_bca": report, "method": "BCa hierarchical seed then within-seed goal×episode"}
def iqm_seed_cluster_bootstrap(seed_effects: Sequence[float], *, draws: int = B, rng_seed: int = RNG_SEED, alpha: float = ALPHA) -> dict[str, Any]:
    effects = _finite(seed_effects)
    if len(effects) < 2:
        raise ValueError("IQM seed-cluster BCa requires at least two seed effects")
    rng = np.random.default_rng(rng_seed)
    samples = np.asarray([iqm(effects[row]) for row in rng.integers(0, len(effects), size=(draws, len(effects)))])
    observed = iqm(effects)
    jack = np.asarray([iqm(np.delete(effects, i)) for i in range(len(effects))])
    report = bca_two_sided_interval(observed, samples, jack, alpha=alpha)
    return {"estimate": observed, "ci": [report["lower"], report["upper"]], "reporting_two_sided_bca": report, "method": "BCa seed-cluster IQM"}


def welch_seed_interval(seed_effects: Sequence[float], *, alpha: float = ALPHA) -> list[float]:
    values = _finite(seed_effects)
    if len(values) < 2: raise ValueError("Welch interval requires at least two seed effects")
    se = values.std(ddof=1) / math.sqrt(len(values)); q = t.ppf(1 - alpha / 2, len(values) - 1)
    return [float(values.mean() - q * se), float(values.mean() + q * se)]


def holm_secondary_decisions(seed_effects: Mapping[str, Sequence[float]], *, primary_passed: bool, draws: int = B, rng_seed: int = RNG_SEED, alpha: float = ALPHA) -> dict[str, Any]:
    """Direct bootstrap one-sided p-values and Holm-adjusted one-sided BCa bounds."""
    if set(seed_effects) != {"2", "3"}: raise ValueError("Holm family must be exactly {'2', '3'}")
    if not primary_passed: return {name: {"status": "untested_primary_failed"} for name in seed_effects}
    effects = {name: _finite(value) for name, value in seed_effects.items()}
    if any(len(value) < 2 for value in effects.values()): raise ValueError("Holm BCa requires at least two seed effects per endpoint")
    boots = {name: _seed_bootstrap(value, draws, rng_seed + int(name)) for name, value in effects.items()}
    # H1: effect > 0; include equality to make the discrete bootstrap p conservative.
    pvalues = {name: float(np.count_nonzero(value <= 0) / len(value)) for name, value in boots.items()}
    ordered = sorted(pvalues, key=pvalues.get); rejected = True; out: dict[str, Any] = {}
    for rank, name in enumerate(ordered):
        threshold = alpha / (len(ordered) - rank)
        jack = np.asarray([np.delete(effects[name], i).mean() for i in range(len(effects[name]))])
        bound = bca_lower_bound(float(effects[name].mean()), boots[name], jack, alpha=threshold)
        rejected = rejected and pvalues[name] <= threshold
        out[name] = {"p_one_sided": pvalues[name], "holm_rank": rank + 1, "holm_alpha": threshold, "holm_lower": bound["lower"], "holm_bca": bound, "reject": rejected}
    return out


def holm_bonferroni(one_sided_p: Mapping[str, float], *, primary_passed: bool, alpha: float = ALPHA) -> dict[str, Any]:
    # Legacy p-value-only reporting helper; decisions use holm_secondary_decisions.
    if set(one_sided_p) != {"2", "3"}: raise ValueError("Holm family must be exactly {'2', '3'}")
    if not primary_passed: return {key: {"status": "untested_primary_failed"} for key in one_sided_p}
    ordered = sorted(one_sided_p.items(), key=lambda row: row[1]); rejected = True; out = {}
    for rank, (name, p) in enumerate(ordered):
        if not 0 <= p <= 1: raise ValueError("p-values must be in [0, 1]")
        threshold = alpha / (len(ordered) - rank); rejected = rejected and p <= threshold
        out[name] = {"p": p, "threshold": threshold, "reject": rejected}
    return out


def tost_paired(seed_effects: Sequence[float], margin: float, *, holm_2_completed: bool = False, alpha: float = ALPHA) -> dict[str, Any]:
    values = _finite(seed_effects); n = len(values)
    if n < 2 or margin <= 0: raise ValueError("TOST requires n >= 2 and a positive margin")
    mean = float(values.mean()); se = values.std(ddof=1) / math.sqrt(n); q = t.ppf(1 - alpha, n - 1)
    ci = [mean - q * se, mean + q * se]; equivalent = ci[0] > -margin and ci[1] < margin
    if n == 5:
        status, limitation = "provisional_unadjusted", "n=5 limitation: unadjusted TOST is provisional and cannot be confirmatory"
    elif n == 8 and not holm_2_completed:
        status, limitation = "confirmatory_pending_holm_2", "n=8: confirmatory TOST is pending the two-control-family Holm adjustment"
    elif n == 8:
        status, limitation = "confirmatory_after_holm_2", "n=8: two-control-family Holm adjustment completed"
    else:
        status, limitation = "unsupported_sample_size", "TOST status is defined only for n=5 provisional or n=8 confirmatory"
    return {"n": n, "estimate": mean, "margin": margin, "ci90": ci, "equivalent": equivalent, "status": status, "limitation": limitation}


def primary_decision(seed_effects: Sequence[float], *, endpoint: str) -> dict[str, Any]:
    stat = seed_cluster_bootstrap(seed_effects)
    passed = stat["primary_lower"] > 0
    return {"endpoint": endpoint, "primary": stat, "welch_seed_interval": welch_seed_interval(seed_effects), "iqm_seed_effect": iqm_seed_cluster_bootstrap(seed_effects), "iqm_not_standalone_decision": True, "state": "1_pass" if passed else "1_fail", "primary_passed": passed}


def _canonical_paths(claim: Path, claim_body: Mapping[str, Any]) -> tuple[Path, Path]:
    expected_claim = sprint_claims.canonical_claim_path(
        claim_body["run_tag"], claim_body["arm"], claim_body.get("generation")
    )
    if claim.absolute() != expected_claim.absolute():
        raise SprintClaimError("BB claim must be at its canonical metrics path")
    return (
        sprint_claims.canonical_result_path(claim_body["run_tag"], claim_body["arm"], claim_body.get("generation")),
        sprint_claims.canonical_raw_path(claim_body["run_tag"], claim_body["arm"], claim_body.get("generation")),
    )

def _zero_assert() -> None:
    root = sprint_claims.canonical_metric_lock_path().parent
    for arm in ("v1", "matched", "random"):
        if any(root.glob(f"p1_{arm}_sprint_heldout_*.json")) or any(root.glob(f"p1_{arm}_sprint_heldout_*.raw.json.gz")):
            raise SprintClaimError("metric lock refused: non-BB canonical held-out artifact exists")
    log = root / "t2_sprint_heldout_access.log"
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            try: row = json.loads(line)
            except json.JSONDecodeError as exc: raise SprintClaimError("access log is not valid JSONL") from exc
            if not isinstance(row, dict): raise SprintClaimError("access log is not valid JSONL")
            if str(row.get("arm", "")).lower() != "bb" or not row.get("purpose"):
                raise SprintClaimError("metric lock refused: non-BB or purpose-less held-out access record exists")


def _validated_bb_pair(claim_path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    claim, digest = sprint_claims.json_file(claim_path, "BB claim")
    claim = sprint_claims.validate_claim_payload(claim)
    result_path, raw_path = _canonical_paths(claim_path, claim)
    if not result_path.is_file() or not raw_path.is_file():
        raise SprintClaimError("canonical BB result and raw artifact are required")
    result, _ = sprint_claims.json_file(result_path, "BB result")
    if not sprint_claims.is_canonical_result(result, claim=claim, claim_sha256=digest):
        raise SprintClaimError("BB result failed canonical episode and summary audit")
    return claim, digest, result["summary"]


def publish_metric_lock(bb_claims: Sequence[Path], lock: Path) -> dict[str, Any]:
    canonical_lock = sprint_claims.canonical_metric_lock_path()
    if Path(lock).absolute() != canonical_lock.absolute():
        raise SprintClaimError("metric lock path must be the canonical metrics lock path")
    paths = [Path(p) for p in bb_claims]
    if len(paths) != 8: raise SprintClaimError("exactly eight BB claim/result pairs are required")
    _zero_assert()
    rows = [_validated_bb_pair(path) for path in paths]
    if any(row[0]["arm"] != "bb" for row in rows): raise SprintClaimError("metric lock requires BB claim/result pairs only")
    seeds = {row[0]["seed"] for row in rows}
    if seeds != set(range(8)): raise SprintClaimError("BB claims must have seed set exactly {0,...,7}")
    total = sum(row[2]["n_episodes"] for row in rows)
    success = sum(row[2]["success_rate"] * row[2]["n_episodes"] for row in rows) / total
    endpoint = "return" if success <= .01 else "success_rate"
    body = {"schema_version": 1, "endpoint": endpoint, "aggregate": success, "created_at": datetime.now(timezone.utc).isoformat(), "bb_claim_sha256": [row[1] for row in rows], "primitive_version": str(PRIMITIVE_SCHEMA_VERSION)}
    target = canonical_lock; target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try: os.write(fd, (json.dumps(body, indent=1, sort_keys=True) + "\n").encode()); os.fsync(fd)
    finally: os.close(fd)
    directory = os.open(target.parent, os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)
    return body
def _seed_mapping(values: Mapping[int | str, Sequence[float]]) -> dict[int, Sequence[float]]:
    normalized = {int(seed): rows for seed, rows in values.items()}
    if set(normalized) != set(range(8)):
        raise ValueError("sensitivity analysis requires seed set exactly {0,...,7}")
    return normalized

def bb_three_way_sensitivity(v1: Mapping[int | str, Sequence[float]], bb: Mapping[int | str, Sequence[float]], *, draws: int = B) -> dict[str, Any]:
    """Report preregistered reuse/new/pooled seed-cluster BCa sensitivity."""
    left, right = _seed_mapping(v1), _seed_mapping(bb)
    effects = {seed: float(_finite(left[seed]).mean() - _finite(right[seed]).mean()) for seed in range(8)}
    groups = {"reuse": (0, 1, 2), "new": (3, 4, 5, 6, 7), "pooled": tuple(range(8))}
    report = {name: seed_cluster_bootstrap([effects[seed] for seed in seeds], draws=draws, rng_seed=RNG_SEED + min(seeds)) for name, seeds in groups.items()}
    reuse_ci, new_ci = report["reuse"]["ci"], report["new"]["ci"]
    report["batch_effect_flag"] = reuse_ci[1] < new_ci[0] or new_ci[1] < reuse_ci[0]
    report["batch_effect_rule"] = "reuse/new two-sided BCa intervals do not overlap"
    return report

def guard_sensitivity(arms: Mapping[str, Mapping[int | str, Sequence[Mapping[str, Any]]]], *, draws: int = B) -> dict[str, Any]:
    """Dual guard analyses, a common-support comparison, and activation-rate CIs."""
    if set(arms) != {"v1", "bb"}: raise ValueError("guard sensitivity requires exactly v1 and bb arms")
    normalized = {arm: _seed_mapping(rows) for arm, rows in arms.items()}

    def rate(rows: Sequence[Mapping[str, Any]], exclude: bool) -> float:
        usable = [row for row in rows if not (exclude and row["eval_wall_guard"])]
        if not usable: raise ValueError("guard exclusion left a seed with no episodes")
        return float(np.mean([
            bool(row["success"]) and not bool(row["eval_wall_guard"])
            for row in usable
        ]))

    def common_support_rate(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        by_goal: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            by_goal.setdefault(str(row["goal_id"]), []).append(row)
        return {
            goal_id: rate(goal_rows, False)
            for goal_id, goal_rows in by_goal.items()
            if not any(bool(row["eval_wall_guard"]) for row in goal_rows)
        }

    policies = {}
    for name, exclude in (("guarded_as_failure", False), ("guarded_excluded_nonrandom_dropout", True)):
        policies[name] = {
            "v1_minus_bb": seed_cluster_bootstrap([
                rate(normalized["v1"][seed], exclude) - rate(normalized["bb"][seed], exclude)
                for seed in range(8)
            ], draws=draws),
            "dropout_limitation": "guarded episodes excluded; dropout is non-random" if exclude else None,
        }
    activation = {
        arm: seed_cluster_bootstrap([
            float(np.mean([bool(row["eval_wall_guard"]) for row in rows]))
            for rows in values.values()
        ], draws=draws)
        for arm, values in normalized.items()
    }
    difference = activation["v1"]["estimate"] - activation["bb"]["estimate"]
    common_effects = []
    for seed in range(8):
        v1_common = common_support_rate(normalized["v1"][seed])
        bb_common = common_support_rate(normalized["bb"][seed])
        common_goals = sorted(set(v1_common) & set(bb_common))
        if not common_goals:
            raise ValueError("common-support restriction left a seed with no paired goals")
        common_effects.append(float(np.mean([
            v1_common[goal_id] - bb_common[goal_id] for goal_id in common_goals
        ])))
    common = seed_cluster_bootstrap(common_effects, draws=draws)
    guarded_as_failure = policies["guarded_as_failure"]["v1_minus_bb"]
    guarded_excluded = policies["guarded_excluded_nonrandom_dropout"]["v1_minus_bb"]
    sign_flip = guarded_as_failure["estimate"] * guarded_excluded["estimate"] < 0
    significance_flip = (
        (guarded_as_failure["primary_lower"] > 0)
        != (guarded_excluded["primary_lower"] > 0)
    )
    guard_confounded = sign_flip or significance_flip
    return {
        "policies": policies,
        "common_support": {
            "v1_minus_bb": common,
            "definition": "paired seed effect over the intersection of goal_id values unguarded in both arms",
        },
        "activation_rates": activation,
        "activation_rate_difference": difference,
        "activation_rate_confounding_flag": abs(difference) > .02,
        "activation_rate_confounding_rule": "absolute arm difference exceeds 2 percentage points",
        "guard_confounded": guard_confounded,
        "guard_confounding_rule": "guarded-as-failure and guarded-excluded analyses reverse sign or primary significance",
        "unconditional_claim_prohibited": guard_confounded,
    }


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle: return json.load(handle)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    lock = sub.add_parser("lock"); lock.add_argument("--bb-claims", nargs=8, required=True); lock.add_argument("--lock", required=True)
    judge = sub.add_parser("judge"); judge.add_argument("--lock", required=True); judge.add_argument("--seed-effects", required=True); judge.add_argument("--json", required=True); judge.add_argument("--md", required=True)
    args = parser.parse_args(argv)
    if args.command == "lock": print(json.dumps(publish_metric_lock(args.bb_claims, Path(args.lock)), indent=1)); return 0
    metric = _load_json(Path(args.lock)); require_metric_lock(Path(args.lock), "v1")
    seed_input = _load_json(Path(args.seed_effects))
    if isinstance(seed_input, dict) and {"effects", "v1", "bb"} <= set(seed_input):
        decision = primary_decision(seed_input["effects"], endpoint=metric["endpoint"])
        decision["bb_three_way_sensitivity"] = bb_three_way_sensitivity(seed_input["v1"], seed_input["bb"])
        if "guard_episodes" in seed_input:
            decision["guard_sensitivity"] = guard_sensitivity(seed_input["guard_episodes"])
            if decision["guard_sensitivity"]["guard_confounded"]:
                decision["unconditional_claim_prohibited"] = True
                decision["claim_limitation"] = "Guard-confounded sensitivity: do not make an unconditional claim from this result."
    else: decision = primary_decision(seed_input, endpoint=metric["endpoint"])
    if metric["endpoint"] == "success_rate": benchmark = "+10%p benchmark applies to success_rate only"
    else: benchmark = f"Return benchmark: 0.5σ={SIGMA_GOAL / 2:.3f} (return units)"
    Path(args.json).write_text(json.dumps(decision, indent=1) + "\n")
    sensitivity = "\n\n## Sensitivity analysis\n\nIncluded in JSON judgment." if "bb_three_way_sensitivity" in decision else ""
    if decision.get("unconditional_claim_prohibited"):
        sensitivity += "\n\n**Guard-confounded sensitivity: do not make an unconditional claim from this result.**"
    Path(args.md).write_text(f"# Sprint primary judgment\n\nEndpoint: {metric['endpoint']}\n\n{benchmark}{sensitivity}\n")
    return 0

if __name__ == "__main__": raise SystemExit(main())
