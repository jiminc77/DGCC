# P1 최종 보고서 — HUMAN sign-off + reward 상수 잠금 (M6)

> 원천: `p1_m3r_results.md` (M3, 9/9) · `p1_t2_results.md` (M4, 판정 (i)(ii) PASS, verdict gate-m4-final-20260717 A) ·
> `p1_random_reference.json` · run/diag JSON 전건. 본 보고서는 종합이며 원천 수치를 재계산하지 않는다.
> 주장 금지 준수: OOD 전이 주장 없음 · near-goal shape coupling 검증 주장 없음 · "response 인코딩 여부" 주장 없음.

## 1. 기준 성능표 (random 참조선 대비)

### T1 (M3R — 통제 재수행, 1e5 tr × 3 seeds, n_envs=256)

random 참조선: t1a 4.0% / return −5.357 · t1b 0.0% / −4.057 · t1c 0.0% / −3.169

| task | s0 | s1 | s2 | (i′) 판정 | (ii) std | 비고 |
|---|---|---|---|---|---|---|
| t1a_straighten (success-diff) | **47%** | 30% | 33% | 3/3 PASS (LB +0.34/+0.18/+0.21) | 7.41%p ✓ | (iii) 47% < 70% 기대 미달 — 사실 기록 |
| t1b_single_bend (return-diff) | 0% (LB −0.80 FAIL) | 1% (+2.72) | 6% (+1.07) | 2/3 | 2.62%p ✓ | 게이트 verdict choice B로 수납 |
| t1c_endpoint_reposition (return-diff) | 0% (+1.83) | 0% (+1.81) | 0% (+2.73) | 3/3 PASS | 0.00%p ✓ | 성공 0이나 return 전건 유의 상회 |

per-template 분해 (M3R 필수 항목): t1a — straight min-D 0.061 최량, u_bend/s_curve 0.14 열위; t1b/t1c — 전 템플릿 0.16–0.20 (원표: p1_m3r_results.md).

### T2 (M4 — goal-conditioned, 3e5 tr × 3 seeds, n_envs=1024)

random 참조선: t2_val 0.0% / return −3.388. 판정 (i)(ii) 전건 PASS (verdict A).

| seed | val (선택 ckpt) | held-out (200 rows, 1회) | (i) LB | val→HO |
|---|---|---|---|---|
| s0 | 24% | **14.5%** | +0.105 | −9.5%p |
| s1 (rerun) | 19% | **24.0%** | +0.190 | +5.0%p |
| s2 | 33% | **32.5%** | +0.270 | −0.5%p |
| mean±std | 25.3±5.8% | **23.7±7.4%** | | |

per-family (held-out): u 최약 (0/6.7/23.3%), s·smooth_random 강세 (~0.24–0.53). per-template: s2 straight 44% 최고, s0 u_bend 10% 최저 (원표: p1_t2_results.md §4).

## 2. 안정성 요약

- **발산(halt) 0건** — M3R 9 + M4 3 (+크래시 s1 시도) 전 run에서 training-level halt 없음 (M3 1차의 halt 4건은 M3R F1/F2/F3+S1/S2/S3로 해소 — gap p95 전건 유계, clamp-hit tail 전건 0).
- **과대추정 갭**: M4 전 seed 음수 안착 (말기 −1.4~−2.9); 유일한 양수 연쇄는 s1r 초기 (+12.5→+4.2→+3.2 → eval4 음수 전환, 개입 없음).
- **argmax entropy**: M4 3 seed 모두 시작~종료 3.44–3.47 ≈ ln(32)=3.466 — eval-state 집합에서 argmax p 분포가 거의 균등 유지 (수치 기록; §6-2 P2 인계 항목).
- **NaN/magnitude incidents** (env-level, 전건 covenant 회복): M3R nan 0–73 / mag 0–72 per run; M4 s0 28/11 · s1r 25/17 · s2 ≥16/≥10 (rebuild @292,864 카운터 리셋 아티팩트 각주 — p1_t2_results.md §6). in-eval incident 0 (held-out 포함).
- **크래시 1건**: m4_t2_s1 1차 — rebuild 한도(8) 초과 fail-closed (@225,280), 아카이브 후 동일-seed 재실행 (verdict A). eval-wall 장벽(최대 14,486s)은 episodic settle-pocket으로 귀속.

## 3. Reward 상수 잠금표 (issue #8 sign-off 조건 이행)

| 상수 | P1 시작값 (P0 §5 확정) | 조정 이력 (STEP_LOG 대조) | 최종값 |
|---|---|---|---|
| α | 10.0 | **없음** | **10.0** |
| c_step | 0.1 | **없음** | **0.1** |
| R_succ | 5.0 | **없음** | **5.0** |

STEP_LOG 전수 대조 결과 조정 기록 0건 ("reward 상수 조정 없음" — M3 report 시점 명기, 이후 배치 불변 규약 유지).
**잠금 확정 — verdict gate-m6-signoff-20260717 choice A (comment 5008514879, 2026-07-17T23:32:44Z, 검증 5/5): α=10.0, c_step=0.1, R_succ=5.0 최종 잠금. 이후 변경 불가 — sprint 전 arm (sprint_spec §0 승계) 및 P2+ 동일 적용.**
불변 등급 재확인: ε_succ=0.05·L · settle 1e-3/10000 · grasp realism ±1node/5% · D=길이 정규화 correspondence L2+canonicalization (Chamfer 보고용만) · K=32 · M=8.

## 4. 하이퍼파라미터 최종표 + 예산 실사용량

TD3 (§7 시작값 그대로 — 조정 0건): γ=0.95 · τ=0.005 · lr=3e-4 · batch 256 · replay 5e5 · UTD 1 · warmup 5k (M4: 10,240 — verdict choice B 승인 항목) · grad-clip 10 · Huber δ=1.0 (S1) · policy_noise 0.05/clip 0.1 · ε_p 1.0→0.1 (30%) · σ_u=0.03 · critic LN (S2) · TD clamp ±1498.0 (S3 유도).

| 단계 | runs | transitions | wall (실측) |
|---|---|---|---|
| M3R | 9 × 100,096 | 900,864 | 137.2 h (8.3–26.4 h/run; 직렬 2-lane) |
| M4 스모크 | 2048 FAIL + 1024 PASS | ~40k | ~4 h |
| M4 본 | 3 × 300,032 (+크래시 225,280) | 1,125,376 | 45.8 h (12.93+10.00+8.88+~14 크래시) |
| 합계 | | ~2.07M | ~187 h |

## 5. 승계 리스크 6건 처리 결과 (P1.md §4 대조)

| # | 리스크 | 처리 |
|---|---|---|
| 1 | settle 예산 경계 | **이행** — P1 전 수집 10000; P0 데이터셋 재수집·재사용 없음 (필터 경로는 미사용으로 종결) |
| 2 | shape coupling 열린 가설 | **관찰 기록만** — D_shape 추이 (M4 §5: 학습과 함께 하강, s0 0.307→0.077); 검증 주장 없음 |
| 3 | DLO-Lab 외부 코드 | **이행** — c5026a9 고정, ti_float alias 유지, NaN covenant 전 구간 작동 (halt 0, 전건 회복); F-a/F-b 재현성 수정은 wrapper/driver 층 |
| 4 | metric 경계 | **이행** — reward·판정 L2 단일 경로 (AST import 검사 테스트); Chamfer 보고용만; 노이즈 바닥도 CL2 재실측 (0.0315 Chamfer-유래 폐기, 실측 ~0) |
| 5 | template 이질성 | **이행** — per-template 분해 M3R/M4 보고 전건 포함 (§1) |
| 6 | 렌더링/datagen | **해당없음** — state 기반, 렌더링 작업 0건 |

## 6. P2 인계 리스크·미해결 항목 (실측 기반)

1. **argmax 균등성**: eval-state에서 argmax p 엔트로피가 ln(32) 근방 유지 (§2) — per-state Q 지형이 노드 간 근소 차이라는 계측 사실. probing 시 p-선택 latent 신호가 약할 수 있음 (해석은 P2).
2. **체크포인트 선택 민감도**: s1r에서 val-최대(19% @275k)와 최종(13% @300k)이 다른 ckpt; 선택 ckpt의 held-out은 24%로 val을 상회 — val 50 goals 표본 잡음이 선택을 좌우할 수 있는 규모 (±5%p 스윙 실측).
3. **seed 간 분산**: T2 held-out 14.5–32.5% (std 7.4%p) — 3 seeds는 분포 특성화에 부족; P2 probe 결론은 seed 조건부로 다뤄야 함.
4. **u-family 약점**: 전 seed held-out u-family ≤23.3% (s1r 0%) — 비대칭 goal 설계 의도와의 관계는 미해석.
5. **D_shape 관찰**: 하강 추이 + s1r 말기 반등 (0.098→0.159) — coupling 가설 검증은 P2 §5 범위.
6. **eval-wall 꼬리**: episodic settle-pocket (최대 14,486s) 원인 미규명 (K=5 가드로 완주에는 영향 없음) — P2 대량 평가 설계 시 churn 상한 필요.
7. **M3R↔M4 seed-비교성 단절 (F-a)** — 교차-마일스톤 seed 비교는 init_hash 확인 필수.
8. **s2 인시던트 카운터 리셋 아티팩트** — rebuild 시 카운터 보존은 다음 배치 계측 수정 항목.

## 7. P2 인수물 목록

| 항목 | 경로 |
|---|---|
| 체크포인트 MANIFEST (M3R 9 + M4 3) | `outputs/models/MANIFEST.md` (커밋) + 선택 manifest JSON ×3 |
| Latent API + 문서 | `src/dgcc/analysis/latent_api.py` · `docs/latent_api.md` · `scripts/extract_latents.py` |
| random 참조선 | `outputs/metrics/p1_random_reference.json` |
| T2 분할 (train/val/held-out) | `src/dgcc/tasks/splits/t2_v1.json` (+ sprint 전용 `t2_sprint_heldout_v1.json`) |
| 데이터셋 | `outputs/data/p1_t2_val_sample.h5` (250, v2) · latent h5 ×12 (`outputs/data/latents/`) · P0 `p0_random_transitions.h5` (v1, 재사용 필터 규약) |
| held-out 결과 (1회 규약 증빙 포함) | `outputs/metrics/p1_t2_heldout_*.json` · claim ×3 · `t2_heldout_access.log` |
| 보고서 사슬 | `p1_m3r_results.md` · `p1_t2_results.md` · `p1_m4_smoke_report.md` · forensics · `noise_floor_cl2.md` · `a2_spike2_report.md` |

---
**상태: 승인 완료 — verdict gate-m6-signoff-20260717 choice A (comment 5008514879). P1 종료, reward 상수 최종 잠금 발효.** #15는 GNG-2 전 close 금지 규칙에 따라 open 유지 (sign-off는 수령·집행됨). P2는 별도 명세로 진행한다.
