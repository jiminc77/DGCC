# sprint_spec.md — paper-sprint 델타 명세 (v1, 사전등록)

> 이 문서는 P1.md의 델타다 — 여기 없는 것은 전부 P1.md를 따른다. 사전등록 핀: 본 파일의 커밋 SHA를 research-dashboard Decision #36에 인용한다. σ_goal 수치(§2)는 M4 완주 직후 2차 커밋으로 고정하며 그 diff는 해당 1줄로 한정한다(2차 SHA도 #36에 재핀).
> 근거: Decision #35(투고 전략)·#36(설계 개정 12건) · 4-관점 적대 리뷰(통계·거버넌스·산술·레드팀) 반영 · 2026-07-16.

## 0. 승계 (P1.md 전 조항 유효)

환경·백본 v2(F1–F3+S1–S3)·covenant·reward(α=10, c_step=0.1, R_succ=5)·ε_succ=0.05·L·D=correspondence_l2·settle 10000·로깅 스키마. **금지 유지:** reward/ε/D 변경, P1.md 수정, M4 held-out 사전 접촉. 이슈는 `type:paper-sprint` 라벨(phase:* 금지), "paper-sprint" 마일스톤. 스프린트 이슈·에픽 본문에 `human_blocked`·`### GATE REQUEST` 리터럴 사용 금지(gate-watcher legacy 텍스트 매칭 오발동 — 인용은 백틱 안에서만).

## 1. 변경 ① — 그리드 (본 비교 실험)

| arm | 정의 | seed |
|---|---|---|
| BB | black-box baseline — M4 3 seed 재사용 + 5 seed 보강 | **8** (3 재사용 + 5 신규) |
| V1 | BB + δm 예측 보조 head: encoder 출력에 f_resp MLP, L = L_TD3 + λ·MSE(δm̂, δm), λ=1.0. δm = Φ_DCT(X_after)−Φ_DCT(X_before) (M=8, 기존 파이프라인) | **8** |
| matched-dim | V1과 동형·동차원, target만 next-latent consistency(EMA) — **동일 안정화(S1–S3) 적용** | **5** |
| random-target | V1과 동형, target = 고정 무작위 24차원 신호(seed 고정) — supervised-aux 일반효과 통제 | **5** |

× {t1a(4 arm × seed 5), T2(상단 표)} × **n_envs 1024** (M4 본 실행 레짐 승계 — warmup 10,240 등 configs/p1_t2.yaml 동일).

- **1024 근거 (2048 폐기):** (a) 2048은 M4 스모크에서 수집 23.1 tr/s < floor 31.04로 **FAIL 강등된 레짐** (b) BB-T2가 M4 3 seed(@1024) 재사용이므로 비교가능성이 1024를 강제. **256↔2048 동등성 검정은 삭제** — M4 3-seed 완주가 T2@1024 레짐 검증을 겸한다. t1a@1024는 그리드 앞 스모크 1 run(1×10⁴ tr)으로 확인.
- **seed 정렬(paired):** 전 arm은 동일 seed 집합을 사용한다(대조군 5는 그 부분집합) — seed 단위 paired/blocked 분석 가능하게.
- **BB parity:** 재사용 3 seed와 신규 5 seed는 **동일 training config/commit**(F1–F3+S1–S3 포함, SHA 핀). 재사용 seed의 checkpoint는 sprint 프로토콜(동일 val-50·cadence·selection metric, eval-wall guard·raw 로깅 동일 적용)의 **eval-only 재실행으로 재선택**해 선택 절차를 통일한다. 감도분석(사전등록): BB를 재사용-only(n=3)·신규-only(n=5)·pooled(n=8) 세 방식으로 병행 보고하고, 두 subsample의 CI가 사전 문턱 이상 이탈하면 batch effect로 표시한다.
- **예산 (실측 기준):** T2 신규 23 runs × 3×10⁵ + t1a 20 runs × 1×10⁵ = 8.9×10⁶ tr ÷ **실효 9.4 tr/s**(1024 스모크 실측, collect+update+eval 포함) ≈ **11.0 GPU일** + eval-guard 마진 ~1일. 절삭 압박 시 t1a seed 5→3 우선(MASTER_PLAN §2 절삭 순서).

## 2. 변경 ② — 통계 사전등록 (unblinding 전 확정)

**판정 순서(사전 지정):** 그리드 unblinding 전에 **BB의 sprint-heldout 성공률(재사용 3+신규 5)을 V1 열람 전에 먼저 확정**하고, ≤1%(무정보)이면 return 기준(i′)을, >1%이면 성공률 기준을 **confirmatory로 정확히 하나 지정**한다(둘 다 계산해 유리한 쪽 선택 금지 — 나머지는 exploratory 병기).

- **성공률 기준 (confirmatory 게이트):** V1−BB의 T2 sprint-heldout 성공률 차이에 대해 **seed 단위 stratified bootstrap(rliable 구현, BCa, B=10,000, RNG seed 고정) 95% CI 하한 > 0**. **+10%p는 판정 게이트가 아니라 실용적 유의성 벤치마크** — 관측 효과크기와 CI를 "사전등록 실용 문턱 +10%p 대비 X%p [L, U]"로 보고한다(AND 결합 금지 — 결합 시 등록 MDE 지점에서 검정력이 ~50%로 절단됨).
- **return 기준 (i′) (confirmatory 게이트):** return-diff의 seed 단위 stratified bootstrap 95% CI 하한 > 0. 효과크기 벤치마크: **IQM return-diff ≥ 0.5·σ_goal**. **σ_goal = M4 3-seed held-out per-goal return(100 goals × 3 seed = 300 관측)의 pooled 표본표준편차(ddof=1)** **= 1.8605 (확정 2026-07-17 — 산출: p1_t2_heldout_m4_t2_s{0,1,2}.json 300관측; 벤치마크 0.5·σ_goal=0.930, Δ_equiv 0.25·σ_goal=0.465)**. n=3 seed-level std는 사용하지 않는다(95% CI가 [0.52s, 6.29s] — 12배 변동).
- **부트스트랩 세부:** percentile 금지·BCa, degenerate 케이스(전 seed 성공 0)는 CI=[0,0] 처리 후 (i′) 트리거. seed 수준 Welch t-구간 병기(일치 확인). IQM은 8 seed 상의 IQM — 단독 판정 금지, CI와 병기.
- **t1a: secondary** — 동일 분석 적용·보고만, GNG-2 분기 비관여. 보조 보고: BB가 M4 판정 (i)(random 참조선 대비)를 통과했는지 병기(V1−BB 차의 해석 기반).
- **Confirmatory family·게이트키핑:** 확정적 검정 = {① 위의 primary 성능, ② 주 task sufficiency-under-shift에서 oracle δm 주입 시 per-point Q-ranking 변화(V1, CI 하한>0), ③ 주 task necessity(마스킹) 성능 하락(V1, CI 하한>0), ④ ②③의 cross-arm 통제}. ①을 먼저 검정하고 **통과 시에만 ②③을 Holm–Bonferroni로** 검정한다. patching 배터리의 나머지(graded corruption·2차 task·대조 patching)는 exploratory — 추론적 주장 없이 점추정+CI만.
- **Cross-arm 통제(④)의 판정:** 대조군 n=5에서는 **provisional-directional** — 회복 효과의 CI를 보고하고 부재 판정은 **TOST 동등성 검정(사전등록 마진 Δ_equiv = 성공률 +5%p / return 0.25·σ_goal)**으로 하며 "n=5 한계(Δ_equiv 이하의 회복은 배제 불가)"를 명기한다. **분기 A 진입 시 대조군 seed를 8로 보강한 후에만 confirmatory 확정** — 논문 판정 수치는 8/8/8/8 결과로 한다.

## 3. 변경 ③ — 계측 선행물 (그리드 시작 조건)

final/held-out eval에 raw 궤적(x_steps·x_initial·x_terminal·goal index) 저장(주기 eval 제외) · truncated/min_d-fallback/reseed 경계 플래그 · probe h5 보존. sprint-heldout(§6)에도 동일 적용. M4 3 seed는 구현 후 checkpoint eval-only 재실행으로 소급 — **소급 재실행의 평가 대상은 val 50과 t2_sprint_heldout_v1에 한정한다. M4 held-out 100은 재실행하지 않는다(P1.md "최종 1회" 규약 유지).**

## 4. 변경 ④ — mechanism (M5 선행 + patching)

- M5 latent 추출 API를 그리드와 병행 구현(z_resp/h_i/critic trunk 노출 — P1.md M5 사양 준용, probe 학습은 여전히 금지).
- patching 4종: 대조(전 arm 동일 패치) · sufficiency-under-shift(OOD 길이, oracle δm 주입 — **성분별 재스케일**: 면내 shape ∝ 길이, z·anchor 절대) · graded corruption · necessity(마스킹). full = 주 task만, 2차 task는 necessity+sufficiency. 통계 취급은 §2 게이트키핑을 따른다. 분기 A 확정 시 **비-DCT 기저 ablation(raw 변위 or PCA-8) = T2 1 arm × seed 8** 추가.

## 5. 변경 ⑤ — eval-wall guard (sprint 한정, 신설)

그리드의 모든 학습 중/최종 eval에서 **episode당 discarded-재시도 상한 K=5** — 초과한 episode는 **failure로 계상**(보수적 방향 — arm 내부 성공률 과대추정 불가)하고 `eval_wall_guard=true` 플래그를 raw 기록에 남긴다. 근거: m4_t2_s0 eval2 wall 13,191s(eval 중 incident 200회 — evaluation.py의 discarded→continue 무제한 재시도) 실측.

- **구현:** evaluation.py의 **config flag(기본값 off = 현행 무제한 재시도)**로 하며 sprint run config에서만 on으로 한다. **M4 잔여 seed의 주기 eval과 M4 held-out 최종 평가는 flag off로 실행**되며, 모든 eval 산출물 메타데이터에 flag 값을 기록한다.
- **감도분석(사전등록 — guard는 arm 간 차이에 대해선 편향 방향 미지):** arm별 guard 발동률(guarded episode 비율)을 CI와 함께 보고한다. Primary V1−BB 차를 (a) guarded=failure(주 규칙) (b) guarded episode 제외(완주분만 — 비무작위 탈락임을 명기) 두 규칙으로 병행 산출해, 부호/유의가 뒤집히면 guard-교란으로 표시하고 해당 결과를 무조건 주장하지 않는다. 양 arm 모두 비-guarded인 goal에 한정한 common-support 대응비교를 보조 보고한다. **arm 간 발동률 차 > 2%p면 교란으로 간주해 원인 조사를 보고에 포함한다.**

## 6. 변경 ⑥ — sprint 전용 held-out (신설)

`src/dgcc/tasks/splits/t2_sprint_heldout_v1.json` — T2 goal 생성기 동일·**신규 seed**로 100 goals 생성·커밋(그리드 시작 전). **grid primary 지표와 GNG-2 판정은 이 split만 사용**한다. **M4 held-out 100은 P1 판정 전용으로 보존 — sprint의 어떤 실험도 접촉 금지**(P1.md L197·L204와의 충돌 원천 차단). val 50은 checkpoint 선택용으로 공용(leakage 아님 — 학습 신호 아님·기존 규약 동일).

- **허용 접촉(사전등록):** (1) 생성·중복 검사 (2) goal 안정성 preflight(정책 무관 측정) (3) run당 최종 1회 정책 평가 — **그 외 로드는 감사 로그상 위반으로 취급한다.**
- **사용 규칙:** run마다 정확히 1회, **val-50으로만 선택된 사전확정 최종 checkpoint**에서 평가한다(기존 클레임 메커니즘 준용 + 접근 감사 로그). 어떤 run의 held-out 수치도 재계산하지 않는다(재-checkpoint 선택·재평가 금지). GNG-2 이후 추가되는 모든 arm/seed도 동일 규칙을 따르고 기존 run은 재평가하지 않는다. **Primary confirmatory 추정치 = GNG-2 시점의 8/8(V1/BB) 결과로 사전 지정** — 보강 후 갱신값은 대조군 confirmatory 확정(§2 ④)과 exploratory 보고에만 사용한다.

## 7. 게이트

GNG-1(**7/19~21**): T2 baseline 3-seed + O1-T2 + preflight → ICLR 창 유지/폐쇄, T1 처분. GNG-2(**8/20 최종 기한, 조기 충족 시 8/16 허용**): patching 1차 결과로 분기 A/B/C (기준 verbatim = Decision A). 게이트 요청은 class:hard만.

**사전등록 하드 데드라인(사건 기준):** Decision A·B 게시와 본 spec의 커밋·SHA 핀은 날짜가 아니라 **M4 held-out 평가 실행 전**이 데드라인이다 — 그 전에 미완이면 gjc에게 held-out 평가 보류를 먼저 지시한다.

## 부록 A — 사전등록 수정 기록 (AMD)

- **AMD-1 (2026-07-18, gate-sprint-gpu-start-20260718 verdict A — comment 5010694414)**: §4의 sufficiency-under-shift "OOD 길이"가 수치 미정의였음을 인정하고, `t2_patch_eval_v1`의 OOD 길이를 **0.75 m / 1.25 m (L_train=1.0 m 대비 ±25% 양방향 대칭)**으로 unblinding 전 human 게이트에서 확정 — confirmatory 지위 유지. 본 공백은 구현 시점에 발견되어 게이트 안건 ③-b로 상정됐으며(논문에 정직 보고 대상), rd#36 재핀(커밋 SHA+json sha256+수치 verbatim)은 판정자 측이 집행하며 재핀 완료 전 patch-eval 관련 unblinding 금지.

- **AMD-2 (2026-07-19, gate-sprint-patching-intervention-20260718 verdict (b) — comment 5013441718)**: GNG-2 조건③ mechanism을 "oracle δm 주입" → **"δm-matched donor h_p interchange의 real−null 대비"**로 재표현(실행체 = 정본 §⑤ `Q(s_r,g,p,u; h_p ← h_p(s_d))`; h_p는 goal-conditioned latent라 δm 외 nuisance 동반 — δm 귀속은 paired real−null 대비로만 성립, 해석은 "response-conditioned h_p mediation"으로 제한). 채택: **A0 confirmatory + A1 exploratory**(matched-P projected splice — 학습 P 재사용, seed 20260719, P 텐서 sha256 408120a3db83df654ee3d1ead6e54a09b04d0c5d6c7a477dd8da301c822d51ab), A2 미채택(향후 별도 게이트), B 기각. **잔여 자유도 핀**: ① 페어 수 = run당 100 (recipient = 각 heldout episode의 초기 상태·goal, goal당 1페어) ② 페어링 = goal별 결정론적: donor는 동일 goal·동일 p·nuisance 매칭 조건에서 |δm(s_d)−rescale(δm_target)| 최소(생성 seed 20260723 고정) ③ null draw = real 페어당 1 (동일 생성 프로토콜에서 δm(s_d′)=δm(s_r)·micro-state만 상이, paired) ④ 조건③ 집계 = per-pair Δ(Kendall τ 변화·top-1 flip)의 real−null 차 → run 평균 → §2 동일 paired seed-cluster BCa one-sided 95% 하한>0 (V1 n=8) ⑤ ckpt 규칙 = 각 run의 val-50 selection manifest ckpt(=heldout claim ckpt_sha256, "lock ckpt"). **eval 규약 상속**: Q-ranking 계열 = content-addressed probe manifest만 소비(재로드 금지, manifest sha256 d04ce48a69748a701c6ec90addacbc0da44c0a1ab9f3f757893cd8c4cf086564 시점 기준·이후 등재분 포함 갱신), 성능 계열 = t2_patch_eval_v1 one-shot. rd#36 재핀은 판정자 측 집행 — 재핀 완료 전 patching 실행·관련 unblinding 금지.
- **AMD-3 (2026-07-21, gate-sprint-incident-s5-reconvene-20260720 verdict (C) — comment 5029426419)**: BB/V1 seed 정렬 0–7에서 **seed 5 페어 전체 제외(paired n=7, 대체 seed 도입 금지)**. 사유 = 재현된 rebuild-한도 크래시(기술 결함 클래스 — F-a byte-일치 재실행에서 재현·조기화: eval2@50,176→eval1@25,600, 양건 rebuild 9·mag covenant 지배·매번 상이 env; 아카이브 s5-20260719T1835Z·s5r-20260721T0107Z·claim 미사용). 성능 사유 seed 선별 금지 규칙 비저촉(성능 무관 기술 결함·증거 아카이브). 편향 방향: BB 평균 상향 가능 → V1−BB 델타 주장에 보수적(민감도 노트 G6a 리포트 명기 예정). seed 5 부분 굤적(50,176·25,600tr)은 어떤 통계에도 미산입. 동일 클래스가 다른 seed에서 발생 시 soft 재소집. G9 통계는 paired n=7 — 검정력·최소 검출 효과크기 재산정을 confirmatory lock 전 첨부. rd#36 재핀은 판정자 측 집행 — V1(G6b) unblinding 전 등록 필수.
