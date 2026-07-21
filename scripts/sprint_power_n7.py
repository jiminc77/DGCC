#!/usr/bin/env python3
"""Deterministic synthetic power/MDE report for the paired n=7 sprint design.

This is reporting-only: it reads only already-published summary fields and uses
Monte Carlo synthetic paired seed effects.  It never opens held-out raw episodes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sprint_stats import seed_cluster_bootstrap

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs/reports/sprint_power_n7.md"
RNG_SEED = 20260721
REPLICATIONS = 1_000
BOOTSTRAP_DRAWS = 2_000  # Reduced from the registered 10,000 for simulation only.
EFFECT_GRID = np.arange(0.0, 0.2001, 0.01)
SUCCESS_REGISTERED_MDE = 0.10
RETURN_REGISTERED_MDE = 0.930


def summary(path: str) -> dict[str, Any]:
    """Read only a published aggregate summary, not episode-level observations."""
    with (ROOT / path).open(encoding="utf-8") as handle:
        return json.load(handle)["summary"]


def source_values(metric: str) -> dict[str, np.ndarray]:
    paths = {
        "m4_3_seed_heldout": [
            "outputs/metrics/p1_t2_heldout_m4_t2_s0.json",
            "outputs/metrics/p1_t2_heldout_m4_t2_s1.json",
            "outputs/metrics/p1_t2_heldout_m4_t2_s2.json",
        ],
        "retro_bb_3_seed": [
            "outputs/metrics/p1_t2_sprint_heldout_m4_t2_s0.json",
            "outputs/metrics/p1_t2_sprint_heldout_m4_t2_s1.json",
            "outputs/metrics/p1_t2_sprint_heldout_m4_t2_s2.json",
        ],
        "new_bb_2_seed": [
            "outputs/metrics/p1_bb_sprint_heldout_sprint_t2_bb_s3.json",
            "outputs/metrics/p1_bb_sprint_heldout_sprint_t2_bb_s4.json",
        ],
    }
    return {name: np.asarray([summary(path)[metric] for path in members], dtype=float) for name, members in paths.items()}


def variance_proxy(values: dict[str, np.ndarray]) -> tuple[float, list[tuple[str, int, float, float]]]:
    """Use the largest observed arm-level seed SD; paired delta SD is sqrt(2) times it."""
    rows = []
    for name, sample in values.items():
        sd = float(np.std(sample, ddof=1))
        rows.append((name, len(sample), float(np.var(sample, ddof=1)), sd))
    largest_sd = max(row[3] for row in rows)
    return float(np.sqrt(2.0) * largest_sd), rows


def power(effect: float, n: int, effect_sd: float, *, stream: int) -> float:
    rng = np.random.default_rng(RNG_SEED + stream + n)
    effects = rng.normal(loc=effect, scale=effect_sd, size=(REPLICATIONS, n))
    passed = 0
    for replicate in effects:
        result = seed_cluster_bootstrap(
            replicate, draws=BOOTSTRAP_DRAWS, rng_seed=int(rng.integers(0, 2**32 - 1))
        )
        passed += result["primary_lower"] > 0.0
    return passed / REPLICATIONS


def endpoint(name: str, metric: str, grid: np.ndarray, registered_mde: float, stream: int) -> dict[str, Any]:
    effect_sd, sources = variance_proxy(source_values(metric))
    powers = {
        n: np.asarray([power(float(effect), n, effect_sd, stream=stream + index * 10_000) for index, effect in enumerate(grid)])
        for n in (8, 7)
    }
    def mde(n: int) -> float | None:
        qualifying = grid[powers[n] >= 0.80]
        return None if not len(qualifying) else float(qualifying[0])
    registered_index = int(np.where(np.isclose(grid, registered_mde))[0][0])
    return {
        "name": name,
        "unit": "%p" if metric == "success_rate" else "return",
        "effect_sd": effect_sd,
        "sources": sources,
        "powers": powers,
        "mde8": mde(8),
        "mde7": mde(7),
        "registered_power8": float(powers[8][registered_index]),
        "registered_power7": float(powers[7][registered_index]),
        "registered_mde": registered_mde,
    }


def effect_text(value: float, unit: str) -> str:
    return f"{value * 100:.0f}%p" if unit == "%p" else f"{value:.3f}"


def main() -> None:
    success = endpoint("성공률", "success_rate", EFFECT_GRID, SUCCESS_REGISTERED_MDE, 10)
    # Return grid is expressed in return units while retaining the requested 0--+20%p success grid separately.
    return_grid = np.unique(np.concatenate((np.arange(0.0, 0.6001, 0.05), np.arange(0.60, 1.2001, 0.01), np.arange(1.25, 2.0001, 0.05), [RETURN_REGISTERED_MDE])))
    returned = endpoint("return", "mean_return", return_grid, RETURN_REGISTERED_MDE, 20_000_000)
    rows = []
    source_sections = []
    for result in (success, returned):
        mde_increase = result["mde7"] - result["mde8"]
        power_loss = result["registered_power8"] - result["registered_power7"]
        rows.append(
            f"| {result['name']} | {effect_text(result['mde8'], result['unit'])} | {effect_text(result['mde7'], result['unit'])} | {effect_text(mde_increase, result['unit'])} | {result['registered_power8']:.1%} → {result['registered_power7']:.1%} | {power_loss:.1%}p |"
        )
        source_rows = "\n".join(
            f"| {name} | {n} | {variance:.6f} | {sd:.6f} |" for name, n, variance, sd in result["sources"]
        )
        source_sections.append(
            f"### {result['name']}\n\n| 관측 대용치(공개 summary) | seed 수 | 표본분산(ddof=1) | seed SD |\n|---|---:|---:|---:|\n{source_rows}\n\n"
            f"시뮬레이션 paired-effect SD = `sqrt(2) × max(seed SD)` = **{result['effect_sd']:.6f}**. 이는 V1과 BB의 seed 변동이 독립이라고 둔 보수적 대용치이며, 실제 V1−BB 상관은 아직 미관측이다."
        )
    report = f"""# Paired n=7 power and MDE reassessment

## Purpose and fixed method

AMD-3의 seed 5 전체 제외 후 confirmatory lock 전 첨부하는 **보고 전용** 재산정이다. 사전등록 문턱과 판단 규칙은 변경하지 않는다. 효과는 synthetic paired seed effects `Normal(δ, SD_proxy)`로만 생성했으며, held-out raw/실데이터를 재평가하거나 접촉하지 않았다.

- Monte Carlo: **{REPLICATIONS:,} replications**/grid point, fixed RNG seed `{RNG_SEED}`.
- Decision in every replication: registered `scripts/sprint_stats.py::seed_cluster_bootstrap`, paired seed-cluster BCa, delete-one-seed jackknife, **one-sided 95% lower bound > 0**.
- Simulation bootstrap draws: **{BOOTSTRAP_DRAWS:,}** (registered decision engine default `B=10,000`; reduced only for this Monte Carlo report and declared here).
- Success grid: 0 to +20%p in +1%p steps; registered practical benchmark: +10%p.
- Return grid: 0 to 2.000 (0.010 increments around 0.600--1.200, otherwise 0.050); registered practical benchmark: `0.5 σ_goal = 0.930`, with `σ_goal=1.8605`.

## n=8 → n=7 result

| endpoint | n=8 80% MDE | n=7 80% MDE | MDE increase | registered MDE-point power (n=8 → n=7) | power loss |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

`n=7` therefore has lower power at each registered MDE point and requires a larger grid-resolved effect to reach 80% simulated power. These quantities characterize precision only; they do not add an effect-size gate.

## Variance evidence and proxy

M4 3-seed held-out summaries and the available retro/new BB summaries provide the requested between-seed range. `m4_3_seed_heldout` is the M4 standard held-out series; `retro_bb_3_seed` is its sprint-heldout reuse series; `new_bb_2_seed` is the completed new-BB s3/s4 series. Values are summary fields only.

{chr(10).join(source_sections)}

## Interpretation limits

1. This is a deterministic planning simulation, not a result from V1−BB effects and not evidence that either endpoint will pass.
2. The n=7 design is fixed by AMD-3; no replacement seed is modeled.
3. The confirmatory criterion remains solely the preregistered one-sided 95% BCa lower bound greater than zero. The +10%p and 0.930 values remain reporting benchmarks, not AND gates.
4. Sparse source seed counts (3/3/2) and unknown within-pair V1−BB correlation make the variance proxy uncertain; the conservative `sqrt(2)` construction is explicitly not a claimed empirical paired-difference variance.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")
    print(f"wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
