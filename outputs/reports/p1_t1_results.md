# P1-M3 — T1 기준 성능 리포트 (3 tasks × 3 seeds × 1e5 transitions)

> **비고 — 미완주 seed:** t1a_s1, t1b_s2, t1c_s1 (rule 6 halt 종결/처분 대기 — 하단 안정성 요약의 halt 표 참조)

사전 등록 기준 (issue #12): (i) 전 seed random 대비 개선 유의 (episode-level bootstrap B=10000, seed 20260703, CI95 하한=경험적 5퍼센타일), (ii) seed 간 최종 성공률 std < 15%p, (iii) t1a ≥ 70% 기대 (미달 시 사실 기록). 성공률 diff와 return diff를 모두 보고한다 — random 성공률이 0%인 task에서 성공률 bootstrap은 정보량이 없으므로 (기준 문구의 한계, 게이트 해석 필요) return diff를 병기.

## t1a_straighten

random 참조선: success 0.040, return -5.357 (n=100)

| seed | final success | final return | final D | succ diff LB(5%) | ret diff LB(5%) | gap 궤적 |
|---|---|---|---|---|---|---|
| 0 | 0.090 | -1.090 | 0.3346 | -0.0100 | +3.589 | 0.58 → 35.79 → 0.40 → 0.53 |
| 1 | — 미완료 (t1a_s1) | | | | | |
| 2 | 0.090 | -5.881 | 0.3645 | -0.0100 | -10.276 | 0.65 → 0.51 → 0.51 → 4.13 |

기준 (ii): 최종 성공률 std = 0.00%p (완료 2 seeds) — 충족 (< 15%p)
기준 (iii): t1a 최종 성공률 max = 9% — 70% 기대 **미달 (사실 기록)**

### per-template 최종 성공률 분해 (리스크 #5)

| seed | straight | u_bend | s_curve | random_smooth |
|---|---|---|---|---|
| 0 | 0.24 | 0.00 | 0.00 | 0.12 |
| 2 | 0.16 | 0.04 | 0.00 | 0.16 |

학습 곡선: `outputs/plots/p1_t1_curves_t1a.png`

## t1b_single_bend

random 참조선: success 0.000, return -4.057 (n=100)

| seed | final success | final return | final D | succ diff LB(5%) | ret diff LB(5%) | gap 궤적 |
|---|---|---|---|---|---|---|
| 0 | 0.000 | -1.732 | 0.2978 | +0.0000 | +1.919 | 35.06 → 13.29 → 4.24 → 0.57 |
| 1 | 0.010 | -1.741 | 0.2738 | +0.0000 | +1.907 | 1.15 → 0.51 → 0.46 → 0.59 |
| 2 | — 미완료 (t1b_s2) | | | | | |

기준 (ii): 최종 성공률 std = 0.50%p (완료 2 seeds) — 충족 (< 15%p)

### per-template 최종 성공률 분해 (리스크 #5)

| seed | straight | u_bend | s_curve | random_smooth |
|---|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | 0.00 | 0.04 | 0.00 | 0.00 |

학습 곡선: `outputs/plots/p1_t1_curves_t1b.png`

## t1c_endpoint_reposition

random 참조선: success 0.000, return -3.169 (n=100)

| seed | final success | final return | final D | succ diff LB(5%) | ret diff LB(5%) | gap 궤적 |
|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.473 | 0.2527 | +0.0000 | +3.142 | -0.87 → -0.37 → -0.71 → 0.62 |
| 1 | — 미완료 (t1c_s1) | | | | | |
| 2 | 0.000 | 1.089 | 1.2042 | +0.0000 | +3.531 | -0.32 → -1.39 → -1.02 → -0.50 |

기준 (ii): 최종 성공률 std = 0.00%p (완료 2 seeds) — 충족 (< 15%p)

### per-template 최종 성공률 분해 (리스크 #5)

| seed | straight | u_bend | s_curve | random_smooth |
|---|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | 0.00 | 0.00 | 0.00 | 0.00 |

학습 곡선: `outputs/plots/p1_t1_curves_t1c.png`

## 안정성 요약

### 학습 레벨 halt (rule 6) — 전건 non-finite critic gradient norm, 갭 폭주 선행

| run (archive) | halt 시점 tr | updates | gap 궤적 (eval별) | 보존 ckpt |
|---|---|---|---|---|
| t1a_s1 (halted-20260704T1120Z) | 13,056 | 7,985 | eval 미도달 | 없음 |
| t1a_s1 (halted-retry-20260705T2111Z) | 53,504 | 48,470 | 2.26e+03 → 1.5e+08 | outputs/models/t1a_s1/ckpt_0050176.pt |
| t1b_s2 (halted-20260706T0500Z) | 65,024 | 60,090 | 191 → 1.09e+06 | outputs/models/t1b_s2/ckpt_0050176.pt |
| t1c_s1 (halted-20260705T2200Z) | 99,328 | 94,461 | 87.9 → 705 → 1.19e+07 | outputs/models/t1c_s1/ckpt_0075008.pt |

완주 run의 env-레벨 NaN incident (rebuild 리셋 보정 합산)과 full rebuild는 각 task 표의 per-seed 필드 및 `p1_t1_results.json` 참조. env 레벨 incident는 전건 NaN covenant (폐기+재시드)로 회복되었고, 데이터 오염 없음 (replay 유입은 isfinite 게이트 통과분만).

## reward 상수

조정 없음 — α=10, c_step=0.1, R_succ=5 (P0 issue #8 시작값 그대로; 전역 규칙 4 미발동).

