# P1-M3R — T1 결과 리포트

> run files present: 9/9; completed: 9/9; preliminary=False

사전 등록 기준 M3R (i′): task random success > 1%이면 success-diff bootstrap, 그 외에는 return-diff bootstrap (B=10,000, seed 20260703, CI95 LB=경험적 5퍼센타일 > 0). (ii) seed 간 최종 성공률 std < 15%p. (iii) t1a ≥ 70%는 factual record. final_d와 d_at_done은 함께 보고한다.

## t1a_straighten

random 참조선: success 0.040, return -5.357; criterion (i′) metric = success

| seed | status | final success | final return | final_d | d_at_done | min_d | i′ LB(5%) | i′ pass | gap p95 bounded/max | clamp hit tail |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| 0 | complete | 0.470 | 1.381 | 0.1706 | 0.1706 | 0.0858 | +0.3400 | pass | True/8.9930 | 0.0000 (zero) |
| 1 | complete | 0.300 | -0.200 | 0.2357 | 0.2357 | 0.1636 | +0.1800 | pass | True/9.1950 | 0.0000 (zero) |
| 2 | complete | 0.330 | 0.981 | 0.1036 | 0.1036 | 0.0771 | +0.2100 | pass | True/8.6551 | 0.0000 (zero) |

기준 (ii): 최종 성공률 std = 7.41%p (완료 3 seeds) — 충족 (< 15%p)
기준 (iii): t1a 최종 성공률 max = 47% — 70% 기대 미달 (사실 기록)

### within-episode min-D / d_at_done distribution by template

| template | n | policy final D mean | policy d_at_done mean | policy min-D mean | min-D p10/p50/p90 | O1 ON d_at_done/min-D | O1 OFF d_at_done/min-D |
|---|---:|---:|---:|---:|---:|---:|---:|
| straight | 75 | 0.0906 | 0.0906 | 0.0608 | 0.0107/0.0290/0.2350 | 0.0939/0.0582 | 0.1536/0.0851 |
| u_bend | 75 | 0.2161 | 0.2161 | 0.1446 | 0.0792/0.1016/0.2495 | 0.2695/0.1613 | 0.2773/0.1812 |
| s_curve | 75 | 0.2376 | 0.2376 | 0.1386 | 0.0493/0.0979/0.2750 | 0.2869/0.1668 | 0.2790/0.1733 |
| random_smooth | 75 | 0.1356 | 0.1356 | 0.0913 | 0.0259/0.0545/0.2612 | 0.2621/0.1414 | 0.2552/0.1521 |

학습 곡선: `outputs/plots/p1_m3r_curves_t1a.png`

## t1b_single_bend

random 참조선: success 0.000, return -4.057; criterion (i′) metric = return

| seed | status | final success | final return | final_d | d_at_done | min_d | i′ LB(5%) | i′ pass | gap p95 bounded/max | clamp hit tail |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| 0 | complete | 0.000 | -4.309 | 0.3669 | 0.3669 | 0.2434 | -0.8038 | fail | True/13.8657 | 0.0000 (zero) |
| 1 | complete | 0.010 | -0.942 | 0.2127 | 0.2127 | 0.1758 | +2.7184 | pass | True/4.2837 | 0.0000 (zero) |
| 2 | complete | 0.060 | -1.809 | 0.2290 | 0.2290 | 0.1686 | +1.0680 | pass | True/10.0443 | 0.0000 (zero) |

기준 (ii): 최종 성공률 std = 2.62%p (완료 3 seeds) — 충족 (< 15%p)

### within-episode min-D / d_at_done distribution by template

| template | n | policy final D mean | policy d_at_done mean | policy min-D mean | min-D p10/p50/p90 | O1 ON d_at_done/min-D | O1 OFF d_at_done/min-D |
|---|---:|---:|---:|---:|---:|---:|---:|
| straight | 75 | 0.2667 | 0.2667 | 0.1912 | 0.0812/0.2050/0.2789 | 0.2304/0.1543 | 0.2201/0.1808 |
| u_bend | 75 | 0.2709 | 0.2709 | 0.1981 | 0.0510/0.2139/0.2752 | 0.2519/0.1533 | 0.2242/0.1473 |
| s_curve | 75 | 0.2822 | 0.2822 | 0.1973 | 0.1232/0.2064/0.2609 | 0.2804/0.1522 | 0.2262/0.1703 |
| random_smooth | 75 | 0.2583 | 0.2583 | 0.1972 | 0.0864/0.2172/0.2678 | 0.2392/0.1384 | 0.2333/0.1655 |

학습 곡선: `outputs/plots/p1_m3r_curves_t1b.png`

## t1c_endpoint_reposition

random 참조선: success 0.000, return -3.169; criterion (i′) metric = return

| seed | status | final success | final return | final_d | d_at_done | min_d | i′ LB(5%) | i′ pass | gap p95 bounded/max | clamp hit tail |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| 0 | complete | 0.000 | -0.730 | 0.2866 | 0.2866 | 0.1811 | +1.8306 | pass | True/13.6250 | 0.0000 (zero) |
| 1 | complete | 0.000 | -0.819 | 0.2800 | 0.2800 | 0.1838 | +1.8085 | pass | True/8.4033 | 0.0000 (zero) |
| 2 | complete | 0.000 | 0.104 | 0.2379 | 0.2379 | 0.1624 | +2.7284 | pass | True/3.8494 | 0.0000 (zero) |

기준 (ii): 최종 성공률 std = 0.00%p (완료 3 seeds) — 충족 (< 15%p)

### within-episode min-D / d_at_done distribution by template

| template | n | policy final D mean | policy d_at_done mean | policy min-D mean | min-D p10/p50/p90 | O1 ON d_at_done/min-D | O1 OFF d_at_done/min-D |
|---|---:|---:|---:|---:|---:|---:|---:|
| straight | 75 | 0.2913 | 0.2913 | 0.1742 | 0.1106/0.1673/0.2423 | 0.2812/0.1869 | 0.2354/0.1499 |
| u_bend | 75 | 0.2730 | 0.2730 | 0.1893 | 0.1247/0.1814/0.2636 | 0.3230/0.1721 | 0.2270/0.1630 |
| s_curve | 75 | 0.2448 | 0.2448 | 0.1633 | 0.1201/0.1563/0.2039 | 0.2283/0.1511 | 0.2001/0.1278 |
| random_smooth | 75 | 0.2635 | 0.2635 | 0.1764 | 0.1252/0.1747/0.2336 | 0.2537/0.1682 | 0.2281/0.1427 |

학습 곡선: `outputs/plots/p1_m3r_curves_t1c.png`

## P-a — training-level halt count

target 0; observed halt records = 0. m3r_t1a_s1 status = complete (completion highlighted).

| run (archive) | halt 시점 tr | updates | halt_reason | gap p95 series | 보존 ckpt |
|---|---:|---:|---|---|---|
| — | 0 | 0 | none | — | — |

## P-b — overestimation gap p95 boundedness

bounded := all `np.isfinite(overestimation_gap_p95)` across evals; max reported per task.

| task | bounded | max gap p95 |
|---|---|---:|
| t1a | True | 9.1950 |
| t1b | True | 13.8657 |
| t1c | True | 13.6250 |

| run | p95 series | bounded | max gap p95 |
|---|---|---|---:|
| m3r_t1a_s0 | 4.0495 → 8.9930 → 0.1647 → 1.7773 | True | 8.9930 |
| m3r_t1a_s1 | 8.6822 → 9.1950 → 5.1707 → 3.4738 | True | 9.1950 |
| m3r_t1a_s2 | 8.6551 → 7.8440 → 1.1077 → 0.0246 | True | 8.6551 |
| m3r_t1b_s0 | 13.8657 → 13.6553 → 7.8622 → 6.6395 | True | 13.8657 |
| m3r_t1b_s1 | 3.1761 → 2.4624 → 4.2837 → 0.0989 | True | 4.2837 |
| m3r_t1b_s2 | 10.0443 → 4.6259 → 3.1508 → 3.4830 | True | 10.0443 |
| m3r_t1c_s0 | 13.6250 → 9.9116 → 1.8449 → 2.8553 | True | 13.6250 |
| m3r_t1c_s1 | 8.4033 → 3.7823 → 3.5205 → 1.3468 | True | 8.4033 |
| m3r_t1c_s2 | 3.8494 → 2.7068 → 0.0032 → 1.5276 | True | 3.8494 |

## P-c — oracle feasibility reference interpretation

> oracle 성공 → 과제 달성 가능 확정 · oracle ≫ policy → 학습 문제 확정 · oracle ≈ 0 → 판정 불능 (불가능 증명 아님)

O1 oracle reference loaded: True (`outputs/metrics/p1_o1_oracle.json`). Oracle 성공은 feasibility reference이며 upper bound가 아니다.

## TD-target clamp hit-rate reading

사전 등록 해석: nonzero steady rate = evidence FOR intrinsic-explosion antithesis.

| run | n | max | final | tail mean | tail series | reading |
|---|---:|---:|---:|---:|---|---|
| m3r_t1a_s0 | 2976 | 0.0000 | 0.0000 | 0.0000 | 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 | zero steady |
| m3r_t1a_s1 | 2976 | 0.0000 | 0.0000 | 0.0000 | 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 | zero steady |
| m3r_t1a_s2 | 2976 | 0.0000 | 0.0000 | 0.0000 | 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 | zero steady |
| m3r_t1b_s0 | 2976 | 0.0000 | 0.0000 | 0.0000 | 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 | zero steady |
| m3r_t1b_s1 | 2976 | 0.0000 | 0.0000 | 0.0000 | 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 | zero steady |
| m3r_t1b_s2 | 2976 | 0.0000 | 0.0000 | 0.0000 | 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 | zero steady |
| m3r_t1c_s0 | 2976 | 0.0000 | 0.0000 | 0.0000 | 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 | zero steady |
| m3r_t1c_s1 | 2976 | 0.0000 | 0.0000 | 0.0000 | 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 | zero steady |
| m3r_t1c_s2 | 2976 | 0.0000 | 0.0000 | 0.0000 | 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 → 0.0000 | zero steady |

## Stability — NaN vs magnitude incidents

rebuild-reset corrected from diag counter series; `—` means the diag series/file is absent.

| run | nan incidents | magnitude incidents | full rebuilds |
|---|---:|---:|---:|
| m3r_t1a_s0 | 150 | — | 4 |
| m3r_t1a_s1 | 125 | — | 6 |
| m3r_t1a_s2 | 94 | — | 4 |
| m3r_t1b_s0 | 108 | — | 2 |
| m3r_t1b_s1 | 83 | — | 4 |
| m3r_t1b_s2 | 107 | — | 3 |
| m3r_t1c_s0 | 73 | — | 0 |
| m3r_t1c_s1 | 111 | — | 4 |
| m3r_t1c_s2 | 187 | — | 5 |

## Notes

- reward constants unadjusted — α=10, c_step=0.1, R_succ=5; no reward/capacity/HER change is introduced by this report.
- schema-debt: HDF5 v3: termination-cause 필드 — M6 P2 승계 추적 항목
- hygiene: old p1_t1_report halted-glob would match m3r halted files if re-run.

