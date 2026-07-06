# P1-M3R — T1 결과 리포트

> run files present: 0/9; completed: 0/9; preliminary=True
> 미완료/누락 M3R runs: m3r_t1a_s0, m3r_t1a_s1, m3r_t1a_s2, m3r_t1b_s0, m3r_t1b_s1, m3r_t1b_s2, m3r_t1c_s0, m3r_t1c_s1, m3r_t1c_s2

사전 등록 기준 M3R (i′): task random success > 1%이면 success-diff bootstrap, 그 외에는 return-diff bootstrap (B=10,000, seed 20260703, CI95 LB=경험적 5퍼센타일 > 0). (ii) seed 간 최종 성공률 std < 15%p. (iii) t1a ≥ 70%는 factual record. final_d와 d_at_done은 함께 보고한다.

## t1a_straighten

random 참조선: success 0.040, return -5.357; criterion (i′) metric = success

| seed | status | final success | final return | final_d | d_at_done | min_d | i′ LB(5%) | i′ pass | gap p95 bounded/max | clamp hit tail |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| 0 | missing | — | — | — | — | — | — | — | — | — |
| 1 | missing | — | — | — | — | — | — | — | — | — |
| 2 | missing | — | — | — | — | — | — | — | — | — |

기준 (ii): 완료 seed 0개 — preliminary 산출 불가
기준 (iii): t1a 완료 seed 없음 — factual record 보류

### within-episode min-D / d_at_done distribution by template

| template | n | policy final D mean | policy d_at_done mean | policy min-D mean | min-D p10/p50/p90 | O1 ON d_at_done/min-D | O1 OFF d_at_done/min-D |
|---|---:|---:|---:|---:|---:|---:|---:|
| straight | 0 | — | — | — | — | —/— | —/— |
| u_bend | 0 | — | — | — | — | —/— | —/— |
| s_curve | 0 | — | — | — | — | —/— | —/— |
| random_smooth | 0 | — | — | — | — | —/— | —/— |

학습 곡선: `outputs/plots/p1_m3r_curves_t1a.png`

## t1b_single_bend

random 참조선: success 0.000, return -4.057; criterion (i′) metric = return

| seed | status | final success | final return | final_d | d_at_done | min_d | i′ LB(5%) | i′ pass | gap p95 bounded/max | clamp hit tail |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| 0 | missing | — | — | — | — | — | — | — | — | — |
| 1 | missing | — | — | — | — | — | — | — | — | — |
| 2 | missing | — | — | — | — | — | — | — | — | — |

기준 (ii): 완료 seed 0개 — preliminary 산출 불가

### within-episode min-D / d_at_done distribution by template

| template | n | policy final D mean | policy d_at_done mean | policy min-D mean | min-D p10/p50/p90 | O1 ON d_at_done/min-D | O1 OFF d_at_done/min-D |
|---|---:|---:|---:|---:|---:|---:|---:|
| straight | 0 | — | — | — | — | —/— | —/— |
| u_bend | 0 | — | — | — | — | —/— | —/— |
| s_curve | 0 | — | — | — | — | —/— | —/— |
| random_smooth | 0 | — | — | — | — | —/— | —/— |

학습 곡선: `outputs/plots/p1_m3r_curves_t1b.png`

## t1c_endpoint_reposition

random 참조선: success 0.000, return -3.169; criterion (i′) metric = return

| seed | status | final success | final return | final_d | d_at_done | min_d | i′ LB(5%) | i′ pass | gap p95 bounded/max | clamp hit tail |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| 0 | missing | — | — | — | — | — | — | — | — | — |
| 1 | missing | — | — | — | — | — | — | — | — | — |
| 2 | missing | — | — | — | — | — | — | — | — | — |

기준 (ii): 완료 seed 0개 — preliminary 산출 불가

### within-episode min-D / d_at_done distribution by template

| template | n | policy final D mean | policy d_at_done mean | policy min-D mean | min-D p10/p50/p90 | O1 ON d_at_done/min-D | O1 OFF d_at_done/min-D |
|---|---:|---:|---:|---:|---:|---:|---:|
| straight | 0 | — | — | — | — | —/— | —/— |
| u_bend | 0 | — | — | — | — | —/— | —/— |
| s_curve | 0 | — | — | — | — | —/— | —/— |
| random_smooth | 0 | — | — | — | — | —/— | —/— |

학습 곡선: `outputs/plots/p1_m3r_curves_t1c.png`

## P-a — training-level halt count

target 0; observed halt records = 0. m3r_t1a_s1 status = missing (completion highlighted).

| run (archive) | halt 시점 tr | updates | halt_reason | gap p95 series | 보존 ckpt |
|---|---:|---:|---|---|---|
| — | 0 | 0 | none | — | — |

## P-b — overestimation gap p95 boundedness

bounded := all `np.isfinite(overestimation_gap_p95)` across evals; max reported per task.

| task | bounded | max gap p95 |
|---|---|---:|
| t1a | False | — |
| t1b | False | — |
| t1c | False | — |

| run | p95 series | bounded | max gap p95 |
|---|---|---|---:|
| — | — | — | — |

## P-c — oracle feasibility reference interpretation

> oracle 성공 → 과제 달성 가능 확정 · oracle ≫ policy → 학습 문제 확정 · oracle ≈ 0 → 판정 불능 (불가능 증명 아님)

O1 oracle reference loaded: False (`outputs/metrics/p1_o1_oracle.json`). Oracle 성공은 feasibility reference이며 upper bound가 아니다.

## TD-target clamp hit-rate reading

사전 등록 해석: nonzero steady rate = evidence FOR intrinsic-explosion antithesis.

| run | n | max | final | tail mean | tail series | reading |
|---|---:|---:|---:|---:|---|---|
| — | 0 | — | — | — | — | no completed diag clamp series |

## Stability — NaN vs magnitude incidents

rebuild-reset corrected from diag counter series; `—` means the diag series/file is absent.

| run | nan incidents | magnitude incidents | full rebuilds |
|---|---:|---:|---:|
| — | — | — | — |

## Notes

- reward constants unadjusted — α=10, c_step=0.1, R_succ=5; no reward/capacity/HER change is introduced by this report.
- schema-debt: HDF5 v3: termination-cause 필드 — M6 P2 승계 추적 항목
- hygiene: old p1_t1_report halted-glob would match m3r halted files if re-run.

