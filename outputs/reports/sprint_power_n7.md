# Paired n=7 power and MDE reassessment

## Purpose and fixed method

AMD-3의 seed 5 전체 제외 후 confirmatory lock 전 첨부하는 **보고 전용** 재산정이다. 사전등록 문턱과 판단 규칙은 변경하지 않는다. 효과는 synthetic paired seed effects `Normal(δ, SD_proxy)`로만 생성했으며, held-out raw/실데이터를 재평가하거나 접촉하지 않았다.

- Monte Carlo: **1,000 replications**/grid point, fixed RNG seed `20260721`.
- Decision in every replication: registered `scripts/sprint_stats.py::seed_cluster_bootstrap`, paired seed-cluster BCa, delete-one-seed jackknife, **one-sided 95% lower bound > 0**.
- Simulation bootstrap draws: **2,000** (registered decision engine default `B=10,000`; reduced only for this Monte Carlo report and declared here).
- Success grid: 0 to +20%p in +1%p steps; registered practical benchmark: +10%p.
- Return grid: 0 to 2.000 (0.010 increments around 0.600--1.200, otherwise 0.050); registered practical benchmark: `0.5 σ_goal = 0.930`, with `σ_goal=1.8605`.

## n=8 → n=7 result

| endpoint | n=8 80% MDE | n=7 80% MDE | MDE increase | registered MDE-point power (n=8 → n=7) | power loss |
|---|---:|---:|---:|---:|---:|
| 성공률 | 14%p | 15%p | 1%p | 61.8% → 57.5% | 4.3%p |
| return | 0.830 | 0.900 | 0.070 | 85.1% → 81.1% | 4.0%p |

`n=7` therefore has lower power at each registered MDE point and requires a larger grid-resolved effect to reach 80% simulated power. These quantities characterize precision only; they do not add an effect-size gate.

## Variance evidence and proxy

M4 3-seed held-out summaries and the available retro/new BB summaries provide the requested between-seed range. `m4_3_seed_heldout` is the M4 standard held-out series; `retro_bb_3_seed` is its sprint-heldout reuse series; `new_bb_2_seed` is the completed new-BB s3/s4 series. Values are summary fields only.

### 성공률

| 관측 대용치(공개 summary) | seed 수 | 표본분산(ddof=1) | seed SD |
|---|---:|---:|---:|
| m4_3_seed_heldout | 3 | 0.008108 | 0.090046 |
| retro_bb_3_seed | 3 | 0.000758 | 0.027538 |
| new_bb_2_seed | 2 | 0.012012 | 0.109602 |

시뮬레이션 paired-effect SD = `sqrt(2) × max(seed SD)` = **0.155000**. 이는 V1과 BB의 seed 변동이 독립이라고 둔 보수적 대용치이며, 실제 V1−BB 상관은 아직 미관측이다.
### return

| 관측 대용치(공개 summary) | seed 수 | 표본분산(ddof=1) | seed SD |
|---|---:|---:|---:|
| m4_3_seed_heldout | 3 | 0.274694 | 0.524113 |
| retro_bb_3_seed | 3 | 0.030266 | 0.173971 |
| new_bb_2_seed | 2 | 0.483181 | 0.695112 |

시뮬레이션 paired-effect SD = `sqrt(2) × max(seed SD)` = **0.983037**. 이는 V1과 BB의 seed 변동이 독립이라고 둔 보수적 대용치이며, 실제 V1−BB 상관은 아직 미관측이다.

## Interpretation limits

1. This is a deterministic planning simulation, not a result from V1−BB effects and not evidence that either endpoint will pass.
2. The n=7 design is fixed by AMD-3; no replacement seed is modeled.
3. The confirmatory criterion remains solely the preregistered one-sided 95% BCa lower bound greater than zero. The +10%p and 0.930 values remain reporting benchmarks, not AND gates.
4. Sparse source seed counts (3/3/2) and unknown within-pair V1−BB correlation make the variance proxy uncertain; the conservative `sqrt(2)` construction is explicitly not a claimed empirical paired-difference variance.
