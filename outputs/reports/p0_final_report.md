# DGCC P0 final report — HUMAN sign-off

이 문서는 P0 종료 승인과 2026-07-03 issue #8 HUMAN SIGN-OFF를 반영한 최종 사실 요약이다. M7 sign-off는 완료되었고, §5의 수치는 사람 판정으로 확정되었다. 주요 출처: `P0.md`, `STEP_LOG.md`, `outputs/reports/sim_comparison.md`, `outputs/reports/g1_report.md`, `outputs/metrics/g2_correlation.json`, `outputs/metrics/g2_correlation_v2.json`, `outputs/metrics/g2_correlation_v3.json`, `outputs/metrics/g1_effect_size.json`, `outputs/metrics/g1_template_decomposition.json`, `outputs/metrics/repeat_variance.json`, `outputs/metrics/settle_budget_sweep.json`, `outputs/metrics/dm_stats.json`.

## 1. Primary sim 결정과 근거 (M2)

**사람 결정:** M2 HUMAN DECISION은 `(A) DLO-Lab primary`이다. 이후 M3+ 파이프라인은 `DLOLabEnv`를 primary adapter로 사용하고, MuJoCo cable adapter는 M1 상태의 frozen fallback으로 남긴다. [source: `STEP_LOG.md`]

| 항목 | MuJoCo cable | DLO-Lab | 출처 |
|---|---:|---:|---|
| smoke | PASS 7/7 | PASS 8/8 | `outputs/reports/sim_comparison.md` |
| compare workload | 5 sequences × 3 seeds × 2 sims | same | `outputs/reports/sim_comparison.md` |
| primitive wall-time mean / max | 5.8256 s / 6.6293 s | 4.7724 s / 11.9963 s | `outputs/reports/sim_comparison.md` |
| settle convergence @ velocity `<0.001`, max 5000 | 0.0% (30 non-converged) | 100.0% (0 non-converged) | `outputs/reports/sim_comparison.md` |
| settle steps mean / max | 5000.0 / 5000 | 1471.7 / 3935 | `outputs/reports/sim_comparison.md` |
| parameter axes | length/segments, bend, twist, friction; no plasticity | length/vertices, bend, twist, friction; plasticity setters present but inactive in P0 | `outputs/reports/sim_comparison.md` |
| parallelism | CPU single-process; MJX cable unsupported | GPU headless batch verified with `n_envs=4` | `outputs/reports/sim_comparison.md` |

Comparison caveat: the settle convergence metric is not directly identical across sims (`max_abs_qvel` for MuJoCo vs `max_node_speed` for DLO-Lab), but DLO-Lab was selected by the human gate after seeing the full comparison. [source: `outputs/reports/sim_comparison.md`, `STEP_LOG.md`]

## 2. 인터페이스·로깅·δm 파이프라인 요약 + 테스트 현황

- **Interface:** P0 §5 defines `DLOEnvBase` with `K=32`, `reset(params, init_shape, seed)`, raw/arc-length centerline accessors, `step_primitive(p, delta, lift)`, and `settle(vel_threshold=1e-3, max_steps=5000)`. `step_primitive` returns `X_before`, `X_after`, `grasp_success`, `settle_steps`, and `info`. [source: `P0.md`]
- **M3 primary adapter:** `DLOLabEnv` implements grasp → move → release → settle, grasp realism with ±1 node execution noise plus 5% failure probability, 4 init shapes (`straight`, `u_bend`, `s_curve`, `random_smooth`), and parameter sweeps over length, bend/twist stiffness, and friction. The M3 log records 1000-draw failure rate 4.9% inside the 5%±1%p acceptance band. [source: `STEP_LOG.md`]
- **Logging:** transition records are stored in an h5py column layout with config and commit metadata; `outputs/data/p0_random_transitions.h5` is data output and remains gitignored. [source: `P0.md`, `STEP_LOG.md`, `outputs/metrics/dm_stats.json`]
- **Φ_DCT / δm layout:** `Phi_DCT(X)` uses DCT-II with `M=8`, axis-major `axis-major-xyz-modes-0-7-v1` layout, 24 total channels. Mode 0 is the 3-channel centroid block; mode ≥1 gives 21 shape channels for normalization/shape deltas. [source: `P0.md`, `outputs/metrics/dm_stats.json`]
- **Invariance:** the Φ invariance test measured max relative error 1.73%, below the 2% requirement. [source: `STEP_LOG.md`]
- **Dataset:** M4 collected 5,056 transitions with `n_envs=64`; grasp success is 0.946598 (94.7%), aggregate settle convergence is 0.536986 (53.7%), and success∧converged is 0.483584 (48.4%). The normalizer fit uses 2,445 success∧converged records. [source: `outputs/metrics/dm_stats.json`]
- **Current test suite:** `uv run pytest tests/ -q` produced `50 passed, 14 warnings in 79.59s` in this M7 run.

## 3. G2 결과와 판정 — full saga

| 단계 | 정의 / 사건 | 결과 | 판정·의미 | 출처 |
|---|---|---:|---|---|
| v1 | Original mixed-norm G2: Spearman `ρ(ΔD, Δ‖c_g‖)` on primary success∧converged population (`n=2445`) | ρ=0.1260506804, threshold 0.9 | FAIL. Diagnostics showed anchor-only ρ=0.9287929157 but shape-only ρ=0.0233946865, so mixed norm collapsed the signal. | `outputs/metrics/g2_correlation.json`, `STEP_LOG.md` |
| Human diagnosis + §8 amendment | Human verdict accepted v1 miss as a measurement-construct problem, not an immediate dual-goal design failure; §8 was amended to component-split G2: anchor AND shape, each threshold ρ≥0.9, correspondence L2 with orientation flip. | 1 authorized re-measurement | Gate threshold unchanged at 0.9; v1 artifacts preserved. | `STEP_LOG.md`, `P0.md` |
| v2 | Component split under amended §8 | anchor ρ=0.9846875812 PASS; shape ρ=0.2570711208 FAIL | OVERALL FAIL under amended rule. Chamfer sensitivity sanity was ρ=0.9165367228 over 248 pairs, so the failure was not explained by Chamfer insensitivity alone. | `outputs/metrics/g2_correlation_v2.json` |
| M5R2 D1/D2 diagnosis | D1 found flip-decision inconsistency; old D_shape vs c_g orientation disagreed in 0.6756646217 (67.6%) of primary cases. D2 Parseval/lowpass sanity found `rho_trunc=0.9999941495`, so DCT truncation/tail was not the cause. | Case A | Human-approved Case A treated orientation canonicalization as a bug fix, not a parameter change. | `outputs/metrics/g2_correlation_v3.json`, `STEP_LOG.md`, `P0.md` |
| v3 | Case A: choose one flip against the goal using `X_before` and apply the same decision to `X_before` and `X_after` for shape `c_g` and `D_shape`; anchor component keeps min-flip correspondence L2. | component (a) anchor ρ=0.9846875812 PASS; component (b) shape ρ=0.9999941495 PASS; OVERALL PASS | G2 closes as PASS. Caveat: component (b) after Case A is largely an orientation-consistency/bug-fix validation under the fixed convention; the non-tautological empirical signal is component (a). | `outputs/metrics/g2_correlation_v3.json`, `P0.md` |

Final G2 status: **OVERALL PASS** with source metrics in `outputs/metrics/g2_correlation_v3.json`. The v1 finding still matters for P1 risk: random far-goal shape coupling was weak before the orientation bug fix, and near-goal behavior remains unprobed. [source: `outputs/metrics/g2_correlation.json`, `outputs/metrics/g2_correlation_v3.json`, `STEP_LOG.md`]

## 4. G1 결과와 판정

**Fixture:** 20 fixed sequences (`straight=5`, `u_bend=5`, `s_curve=5`, `random_smooth=5`) × seeds `[0,1,2]` × stiffness multipliers `[0.5,1.0,2.0]`, with friction `[0.5,1.0,2.0]` measured as a subordinate reference block. Grasp realism was off for the controlled measurement. [source: `outputs/reports/g1_report.md`, `outputs/metrics/g1_effect_size.json`]

### Pooled stiffness effect-size matrix

| pair | between mean | within-floor mean | Cohen's d | i.i.d. 95% CI | sequence-cluster 95% CI | source |
|---|---:|---:|---:|---:|---:|---|
| 0.5_vs_1.0 | 0.0465329 | 0.0443103 | 0.0611288 | [-0.264927, 0.369707] | [-0.292368, 0.373292] | `outputs/metrics/g1_effect_size.json` |
| 1.0_vs_2.0 | 0.0522068 | 0.0540651 | -0.0335125 | [-0.317310, 0.270086] | [-0.273123, 0.168205] | `outputs/metrics/g1_effect_size.json` |
| 0.5_vs_2.0 | 0.0666730 | 0.0529987 | 0.235856 | [-0.0568308, 0.557556] | [0.0409446, 0.456666] | `outputs/metrics/g1_effect_size.json` |

### Pooled friction reference matrix

| pair | between mean | within-floor mean | Cohen's d | i.i.d. 95% CI | sequence-cluster 95% CI | source |
|---|---:|---:|---:|---:|---:|---|
| 0.5_vs_1.0 | 0.0262704 | 0.0409265 | -0.460699 | [-0.742106, -0.180638] | [-0.844434, -0.196798] | `outputs/metrics/g1_effect_size.json` |
| 1.0_vs_2.0 | 0.0257735 | 0.0417548 | -0.515630 | [-0.757625, -0.267773] | [-0.847487, -0.255221] | `outputs/metrics/g1_effect_size.json` |
| 0.5_vs_2.0 | 0.0300093 | 0.0448200 | -0.412076 | [-0.647576, -0.161474] | [-0.701019, -0.202290] | `outputs/metrics/g1_effect_size.json` |

**Human verdict:** option **(b)** was adopted: stiffness is demoted as a primary OOD axis and P1 OOD should be reorganized around length (+ discretization). Friction is **not** promoted to primary; option **(c)** springback task is **not** adopted; plasticity activation is rejected for P0/P1 continuity. [source: `STEP_LOG.md`]

> “quasi-static pick-and-place regime에서 탄성/마찰 파라미터 OOD 전이 주장은 그 자체로 검증력 없음.” — M6/G1 human verdict methodological observation. [source: `STEP_LOG.md`]

Template decomposition is appendix material, not a new G1 gate decision. It shows pooling heterogeneity: `u_bend` stiffness d is positive (`0.801315` for 0.5_vs_1.0, `0.739159` for 0.5_vs_2.0), while `random_smooth` stiffness d is negative (`-0.891970`, `-0.609613`, `-0.334725` across the three stiffness pairs). Each template-specific cluster bootstrap has only 5 sequence clusters, so this is a small-n caution. [source: `outputs/metrics/g1_template_decomposition.json`]

## 5. 확정 수치 고정표 — issue #8 HUMAN SIGN-OFF 반영

| 항목 | 확정값 | 근거·조건 |
|---|---|---|
| reward 상수 | `α=10`, `c_step=0.1`, `R_succ=5` — P1 시작값으로 채택 | P0 무증거 항목(RL 미수행). P1 baseline 안정화 중 조정 허용하되 변경 시 `STEP_LOG.md` 기록 필수, P1 종료 시 최종 잠금. [source: issue #8 HUMAN SIGN-OFF, `P0.md`] |
| 성공 임계 `ε_succ` | `ε_succ=0.05·L` 확정 | 순수 실행 노이즈 median `0.031512·L` 대비 약 1.6x 여유. mean `0.046626·L`는 5% grasp-failure 쌍이 부풀린 값이므로 판단 기준으로 부적합. 에피소드 내 재파지 가능. P1 파일럿에서 성공률 천장 관찰 시 재상정. [source: issue #8 HUMAN SIGN-OFF, `outputs/metrics/repeat_variance.json`] |
| settle 기준·예산 | velocity threshold `1e-3` 불변, `max_steps` `5000 → 10000` 증액 | Sweep: 10000에서 100% 수렴, first-crossing max `7608 < 10000`. A2(quasi-static) 보증과 usable 데이터 `48.4% → ~95%` 이득이 비용을 상회. 기존 M4 데이터셋은 재수집하지 않고 플래그 유지; P1 수집부터 적용. [source: issue #8 HUMAN SIGN-OFF, `outputs/metrics/settle_budget_sweep.json`, `outputs/metrics/dm_stats.json`] |
| grasp realism | ±1 node execution noise + 5% failure probability 유지 확정 | 1000-draw probe 4.9% 검증 및 Appendix B 정량화. [source: issue #8 HUMAN SIGN-OFF, `STEP_LOG.md`, `outputs/metrics/repeat_variance.json`] |
| OOD primary | length train `[0.8,1.2]` → OOD `{0.5,0.6,1.4,1.6}` m 확정; density-preserving `n_segments` `{25,30,70,80}` 채택 | G1 verdict (b). Discretization 축은 별도로 고정 `L`에서 `N∈{25,100}`; Φ 불변성 실측 `1.73% < 2%`. [source: issue #8 HUMAN SIGN-OFF, `STEP_LOG.md`] |
| OOD 보조/참고 | 보조 = initial/goal shape 분포; stiffness/friction = reference-only 확정 | G1 pooled stiffness `d≤0.24`, friction negative `d`; 본문 claim 제외, appendix 보고. 방법론적 관찰 문구 유지. [source: issue #8 HUMAN SIGN-OFF, `outputs/metrics/g1_effect_size.json`, `outputs/metrics/g1_template_decomposition.json`] |
| `D` (reward·성공 판정) | 길이 정규화 correspondence L2로 통일 확정; orientation canonicalization 규약 포함 | G2 v3가 검증한 metric↔`c_g` 정합 조합. Chamfer는 보고·참고 지표로만 병기. [source: issue #8 HUMAN SIGN-OFF, `outputs/metrics/g2_correlation_v3.json`, `P0.md`] |

## 6. P1에 넘길 리스크·미해결

1. **Settle budget implementation boundary:** HUMAN SIGN-OFF fixes `vel_threshold=1e-3` and raises future collection `max_steps` to 10000, but the existing M4 dataset is not recollected; its non-convergence flags remain part of the audit trail. [source: issue #8 HUMAN SIGN-OFF, `outputs/metrics/dm_stats.json`, `outputs/metrics/settle_budget_sweep.json`]
2. **Shape-channel coupling:** v1 mixed-norm diagnostics found shape-only ρ=0.0233946865 under random far goals; v2 shape component before Case A was ρ=0.2570711208. Case A fixes orientation consistency, but near-goal shape behavior remains an open hypothesis rather than P0 evidence. [source: `outputs/metrics/g2_correlation.json`, `outputs/metrics/g2_correlation_v2.json`, `outputs/metrics/g2_correlation_v3.json`, `STEP_LOG.md`]
3. **DLO-Lab external-code risk:** DLO-Lab remains young external code; runtime `ti_float` aliasing is required, SharePoint assets originally returned HTTP 401, and dependency pins around torch/genesis/numpy/fsspec/packaging are fragile. [source: `outputs/reports/sim_comparison.md`, `STEP_LOG.md`]
4. **Metric boundary:** reward and success judgment use length-normalized correspondence L2 with orientation canonicalization; Chamfer remains a report/reference metric only. P1 specs should preserve this boundary explicitly. [source: issue #8 HUMAN SIGN-OFF, `P0.md`, `outputs/metrics/g2_correlation_v3.json`]
5. **Template heterogeneity:** pooled G1 numbers hide template-specific sign changes; `u_bend` is positive while `random_smooth` is negative for stiffness, with only 5 sequence clusters per template. [source: `outputs/metrics/g1_template_decomposition.json`]
6. **Rendering/datagen operational note:** `assets/dlo-lab.zip` exists on disk and is gitignored. Future DLO-Lab rendering/datagen should wire LuisaRender correctly and use the official datagen `--raytracer` path. [source: `STEP_LOG.md`]

## APPENDICES — artifact pointers and key tables

### Appendix A — G1 template decomposition

Artifacts: `outputs/metrics/g1_template_decomposition.json`, `outputs/plots/g1_template_decomposition.png`. Recompute-only from stored G1 raw distance lists; no new simulation. Small-n caveat: each template has 5 sequence clusters, 15 between-condition distances, and 30 pooled within-floor distances. [source: `outputs/metrics/g1_template_decomposition.json`]

| template | stiffness d 0.5_vs_1.0 | stiffness d 1.0_vs_2.0 | stiffness d 0.5_vs_2.0 | friction d 0.5_vs_1.0 | friction d 1.0_vs_2.0 | friction d 0.5_vs_2.0 |
|---|---:|---:|---:|---:|---:|---:|
| straight | 0.348485 | 0.206073 | 0.365720 | 0.128318 | 0.194466 | 0.225581 |
| u_bend | 0.801315 | 0.462745 | 0.739159 | -0.132553 | -0.190021 | 0.077062 |
| s_curve | -0.232485 | -0.317238 | 0.047356 | -0.998389 | -0.630994 | -0.528301 |
| random_smooth | -0.891970 | -0.609613 | -0.334725 | -0.846355 | -1.487515 | -1.160505 |

### Appendix B — Repeat execution variance

Artifacts: `outputs/metrics/repeat_variance.json`, `outputs/plots/repeat_variance.png`, `outputs/reports/appendix_repeat_variance.log`. Design: 4 cells, 16 repeats per cell, 64 envs per condition, length-normalized bidirectional Chamfer among final 32-point centerlines, immutable threshold 0.001. [source: `outputs/metrics/repeat_variance.json`]

| block | grasp realism | n pairwise distances | mean | median | q97.5 | max |
|---|---|---:|---:|---:|---:|---:|
| realism ON | ±1 node + 5% failure | 480 | 0.0466261 | 0.0315116 | 0.238442 | 0.319246 |
| realism OFF | disabled | 480 | 0.00000281661 | 0.0 | 0.0000245813 | 0.0000245813 |

### Appendix C — Settle budget sweep

Artifacts: `outputs/metrics/settle_budget_sweep.json`, `outputs/plots/settle_budget_sweep.png`, `outputs/reports/appendix_settle_sweep.log`. Design: 24 cases, budgets `[5000,10000,20000]`, balanced init-shape cycle, threshold 0.001 unchanged, plasticity disabled. [source: `outputs/metrics/settle_budget_sweep.json`]

| budget max_steps | convergence rate | speed mean at budget | speed max at budget |
|---:|---:|---:|---:|
| 5000 | 0.833333 | 0.00130576 | 0.00821782 |
| 10000 | 1.0 | 0.00138805 | 0.00675888 |
| 20000 | 1.0 | 0.00130990 | 0.00608431 |

First-convergence stats over 24 converged cases: min 1110, median 2434, mean 2931.875, max 7608. Four cases were not converged by 5000 (`case_06`, `case_07`, `case_09`, `case_16`). For that subset, shape change 5000_vs_20000 has mean 0.00221368 and max 0.00490197. [source: `outputs/metrics/settle_budget_sweep.json`]
