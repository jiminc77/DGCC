# P1-M4 — T2 본 학습 결과 (goal-conditioned, 3 seeds × 3×10⁵ transitions)

> 사전등록 재확인 (held-out 평가 직전 수행): Decision research-dashboard#35·#36 CLOSED-final ·
> sprint_spec@`82230d8` · claim 파일 부재 · `t2_heldout_access.log` 공백 확인 → held-out 1회 평가 개시.
> 접근 로그: 3건 (s0/s1/s2 각 1회, 2026-07-17T13:11–13:26Z) — **held-out 총 접근 3회 = 규약대로 seed당 정확히 1회.**

## 1. 판정 (사전 등록 기준 — P1.md M4)

| 기준 | 내용 | 결과 |
|---|---|---|
| (i) | 3 seed 모두 held-out 성공률이 random 참조선 대비 유의 개선 (bootstrap 95% CI) | **PASS ×3** — 5th-pct LB: s0 +0.105 · s1 +0.190 · s2 +0.270 (모두 >0) |
| (ii) | val→held-out 성공률 하락 < 15%p | **PASS ×3** — s0 +9.5%p · s1 −5.0%p(상승) · s2 +0.5%p |

- Bootstrap 규약: episode-level success-diff vs `p1_random_reference.json` `blocks.t2_val` (100 rows, 성공률 0.0), B=10,000, seed 20260703 — HER halfway와 동일 규약. 산출: `outputs/metrics/p1_m4_judgment.json`.
- (i′) 비고: "성공률 ≤1% 시 return-기반 대체 판정" 경로는 **미발동** (전 seed held-out ≥14.5%).
- random 참조선은 T2 validation에서 측정된 값(성공 0/100)이다. held-out 전용 random 측정은 수행하지 않았다
  (held-out 접근 최소화 규약 우선; 참조선 분포 근거는 §8 한계 참조).

## 2. 최종 성능표

| seed | init_hash | 최종 val | 선택 ckpt (val 최대) | held-out (200 rows) | held-out return | held-out final_d | val→HO |
|---|---|---|---|---|---|---|---|
| s0 | a2ff322e | 24% | ckpt_0300032 (24%) | **14.5%** | +1.185 | 0.077 | −9.5%p |
| s1 (rerun) | 06912e7d | 13% | ckpt_0275456 (19%) | **24.0%** | +1.624 | 0.086 | +5.0%p |
| s2 | 1b619736 | 33% | ckpt_0300032 (33%) | **32.5%** | +2.229 | 0.070 | −0.5%p |
| **mean±std** | | 23.3±8.2% | | **23.7±7.4%** | +1.68 | 0.078 | |

체크포인트 선택: val 50 goals × 2 episodes 성공률 최대 (동률 시 return, 그다음 최소 transitions) — held-out은 선택에 비접근. Claim 파일 3건 (`p1_heldout_claim_m4_t2_s{0,1,2}.json`, O_CREAT|O_EXCL) 영구 보존.

## 3. 학습 곡선 (val success %, 25k 간격 12 evals)

| @tr | 25.6k | 50.2k | 75.8k | 100.4k | 126.0k | 150.5k | 175.1k | 200.7k | 225.3k | 250.9k | 275.5k | 300.0k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0 | 0 | 2 | 0 | 0 | 4 | 4 | 6 | 12 | 12 | 10 | 22 | **24** |
| s1r | 0 | 0 | 0 | 0 | 1 | 2 | 5 | 10 | 9 | 9 | 19 | **13** |
| s2 | 2 | 4 | 10 | 15 | 30 | 27 | 26 | 24 | 31 | 30 | 28 | **33** |

- 공통 패턴: 초반 저성과 → 중후반 가속 (s2는 조기 가속, 126k에 30% 도달). return은 s1r 초기 음수(−4.74)에서 전 seed 양수로 회복.
- HER 조건부 절차: halfway(150,528)에서 s0 val 4/100 vs random 0/100 — bootstrap 5th-pct LB +0.0100 > 0 (ANY-significant) → **CONT, HER 미도입 확정** (s1r/s2는 규약상 halfway 정지 불요).

## 4. 분해 (리스크 #5 필수 항목)

### Per-goal-family (held-out 성공률)

| seed | l | s | smooth_random | u | zigzag |
|---|---|---|---|---|---|
| s0 | 0.053 | 0.238 | 0.184 | 0.067 | 0.154 |
| s1r | 0.184 | 0.310 | 0.526 | 0.000 | 0.154 |
| s2 | 0.158 | 0.476 | 0.474 | 0.233 | 0.269 |

### Per-init-template (held-out 성공률)

| seed | straight | u_bend | s_curve | random_smooth |
|---|---|---|---|---|
| s0 | 0.12 | 0.10 | 0.14 | 0.22 |
| s1r | 0.20 | 0.28 | 0.28 | 0.20 |
| s2 | 0.44 | 0.24 | 0.32 | 0.30 |

- u-family goal이 전 seed 최약 (s1r 0%) — 사실 기록. s/smooth_random family가 상대적 강세.

## 5. D_shape 추이 — 관찰 기록만 (리스크 #2; 해석·주장 없음)

val eval의 `mean_d_shape_at_done` (12 evals): s0 0.307→0.077 (단조 하강 경향, 최저 0.077) · s1r 0.283→0.159 (최저 0.098 @200.7k, 말기 반등) · s2 0.168→0.112 (최저 0.062 @250.9k). held-out d_shape_at_done: s0 0.073 · s1r 0.091 · s2 0.060. 학습 진행과 함께 하강하는 추이가 관찰되었다 — near-goal shape coupling에 대한 어떤 검증 주장도 하지 않는다 (P0 열린 가설 유지).

## 6. 안정성 요약

| run | wall_h | nan_env | magnitude | rebuild (한도 8) | halt | in-eval nan/mag |
|---|---|---|---|---|---|---|
| s0 | 12.93 | 28 | 11 | 3 | 0 | 0/0 |
| s1 (크래시, 참고) | ~14 (@225,280 중단) | — | — | **9 → 한도 초과 크래시** | fail-closed | — |
| s1 rerun | 10.00 | 25 | 17 | 1 | 0 | 0/0 |
| s2 | 8.88 | ≥16* | ≥10* | 1 | 0 | 0/0 |

\* s2의 run-complete 카운터는 최종 rebuild(@292,864)가 scene을 재생성하며 리셋되어 0/0으로 인쇄됨 — round 로그 누적 최대치(16/10)를 하한으로 보고. 계측 아티팩트로 기록 (코드 수정은 배치 경계 규약상 다음 배치 설계 항목).

- 과대추정 갭: 전 seed 초기 이후 음수 안착 (s1r만 초기 +12.5→+4.2→+3.2 양수 연쇄 후 eval4에서 음수 전환 — gap-감시 규약상 관찰 기록, 개입 없음). 말기 폭주 없음.
- **크래시 인시던트 (s1 1차):** eval9 도중 rope-state 회복 실패 연쇄로 rebuild 9회째 요청 → 한도 8 초과 fail-closed 크래시 (@225,280). 아카이브 `.crashed-20260716T1609Z` (13-file manifest), 동일 seed 재실행 verdict (gate-m4-incident-s1-20260716 choice A). 성능 기반 seed 선별·재실행 금지 규약 준수 — 재실행은 기술 결함 사유만.

## 7. Eval-wall 표 (원인 귀속 포함)

| run | 12 evals wall_s | 비고 |
|---|---|---|
| s0 | 473 / **13,191** / 473 / 434 / 431 / 431 / 433 / 432 / 432 / 432 / 473 / 424 | eval2 13,191s = episodic settle-pocket (K=5 retry guard 경유; soft gate choice A로 계속) |
| s1 크래시 (참고) | 1,691 / 981 / 433 / **14,486** / 435 / **6,349** / 436 / 2,401 (8 evals) | 장벽 2회 → trigger ① 발동, gate choice A; eval9 중 크래시 |
| s1 rerun | 946 / 1,343 / 909 / 989 / 436 / 950 / 875 / 1,109 / 910 / 438 / 437 / 1,146 | 전건 <3,600s — eval-포켓 부재, 처리량 크래시 시도 대비 2배 |
| s2 | 434 / 435 / 435 / 436 / 834 / 593 / 870 / 435 / 435 / 436 / 435 / 435 | 전건 정상 |

귀속: 장벽 evals은 특정 episode의 settle 미수렴 반복(재시도 가드 K=5, `evaluation.py`)에 기인하는 episodic 현상으로, seed·시점 재현성 없음 (s1 크래시 시도 vs 재실행 대비가 근거). eval-churn 상한은 다음 배치 설계 항목 (본 배치 중 변경 금지 준수).

## 8. 한계 (Limitations)

1. **M3R↔M4 seed-비교성 단절 (F-a):** M4는 F-a 수정(TD3Agent 생성 후 torch re-seed) 적용 — 동일 seed번호여도 M3R 초기 가중치와 상이. init_hash로 판별 가능 (재실행 s1 init 06912e7d = 크래시 s1과 byte-identical — F-a 수정의 실증).
2. **스모크 게이트 설계 한계:** n_envs 사다리 판정(2048 FAIL→1024 PASS)의 처리량 임계는 저비용-eval 조건부 투영 — 장벽 eval 발생 시 실제 wall은 투영 초과 (s0 12.93h vs 투영 ~9h). 다음 배치에서 eval-wall 분포 반영 필요.
3. random 참조선은 t2_val 측정값(0/100)으로, held-out 분포에서의 직접 측정이 아니다 (held-out 접근 최소화 우선). LB가 +0.105 이상으로 큰 격차여서 판정에 영향 없다고 판단하나 한계로 기록.
4. s2 인시던트 카운터 리셋 아티팩트 (§6 각주).
5. 곡선 관찰(§5)은 관찰 기록이며 어떤 coupling 주장도 아니다. stiffness/friction OOD 전이 주장 없음 (G1 verdict (b) reference-only 준수).

## 9. M3R 부록 정정

`p1_m3r_results.md`의 Stability 표는 magnitude-incident 열이 없었다 — M4부터 magnitude 열을 표준 포함 (§6). M3R 수치 자체는 불변 (아카이브 immutable).

## 10. 산출물 목록

- run JSON: `outputs/metrics/p1_run_m4_t2_s{0,1,2}.json` (+ 크래시 s1 아카이브 `.crashed-20260716T1609Z`)
- 선택 manifest: `outputs/metrics/p1_m4_ckpt_selection_m4_t2_s{0,1,2}.json`
- held-out 결과: `outputs/metrics/p1_t2_heldout_m4_t2_s{0,1,2}.json` · claim ×3 · access log
- 판정: `outputs/metrics/p1_m4_judgment.json` · 스모크: `outputs/reports/p1_m4_smoke_report.md`
- 체크포인트: `outputs/models/m4_t2_s{0,1,2}/` (git 제외; MANIFEST는 M5 범위)
