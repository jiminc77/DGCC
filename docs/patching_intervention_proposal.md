# G10 patching 개입 지점 사전등록 제안서 (게이트 안건 초안)

> 작성: 2026-07-18 (orchestrator steer ② 이행 — architect 설계 분석 72-InterventionProposal 기반, 코드 라인 전수 대조). 지위: **human 핀 전 G10 patch-eval 산출물 생성·열람·unblinding 금지.** 승인 시 AMD-2로 sprint_spec 부록 기록.

## 1. 실행 불가 확정 사실 (코드 인용)

"z_resp에 oracle δm 주입"은 현재 아키텍처에서 Q에 도달하는 인과 경로가 없다:

- Q의 런타임 입력은 `(h_p, u)`뿐 — `src/dgcc/rl/sprint_arms.py:174`의 `self.critic(h_p, u)`, `src/dgcc/models/networks.py:164-167`의 `_QHead`는 `[h_p,u]` concat.
- `f_resp`(z_resp)는 aux-loss 전용 — `sprint_arms.py:191-192`에서만 MSE에 소비, Q forward 미연결.
- 차원 불일치: `z_resp`는 (B,256), oracle δm는 (B,24) (`sprint_arms.py:55-67`, `latent_api.py:38`).
- latent API는 읽기 전용 — `extract()`는 override/write 인자 없음 (`latent_api.py:152-207`).
- 정본 §⑤(docs/sprint_contracts.md:66)는 애초에 **h_p 이식** `Q(s_r,g,p,u; h_p ← h_p(s_d))`로 등록했고 :72에서 aux 출력 patch를 기각.

## 2. 후보와 권고

### A0 — 전체 donor h_p 이식 (**confirmatory 권고**)

정본 수식 verbatim: `Q(s_r,g,p,u; h_p ← h_p(s_d))`

- `h_r = Enc(s_r,g)[p]`, `h_d = Enc(s_d,g)[p]`, `h_p' = h_d`, `Q_patch = Q(h_p', u_r)`.
- Donor는 동일 `p`, goal·anchor·호길이 파라미터화 등 비-δm nuisance는 recipient와 매칭.
- Donor simulator 조건·재스케일 정본 그대로: `oracle δm(s_d) = rescale(δm_target)`, axis-major에서 r=L_ood/L_train일 때 `m(r) = (1,r×7 | 1,r×7 | 1×8)` — x/y modes 1–7만 r 곱, x0/y0/z0·전 z modes 절대 유지.
- 무정규화.

### A1 — matched-P projected donor splice (**exploratory 보강안**)

`P∈R^{24×256}`(고정 Gaussian-QR, seed 20260719), 행 직교이므로 `P⁺=Pᵀ`, `Π_P=PᵀP`:

`h_p' = h_r + P⁺(P h_d − P h_r) = (I−Π_P) h_r + Π_P h_d`

- recipient의 ker(P) 성분 보존, donor의 row-space 성분만 이식. 전 arm에 동일 고정 P·동일 연산.
- **명칭 규율**: "δm-정렬 부분공간" 호칭 금지 — P는 δm 사상으로 학습된 것이 아님. 정확한 명칭 = **matched-P projected latent subspace**.

### A2 — direct δm target steering (**비권고, 유지 시 calibration human-핀 조건부**)

`h_p' = h_r + P⁺(N(rescale(δm_target)) − P h_r)` — 사상 `N: R²⁴_DCT→R²⁴_proj`가 필수. identity/z-score는 축 의미 대응 근거 없음 — 유지하려면 training-only 데이터로 고정한 선형 calibration `N(δ)=C·diag(σ_δ)⁻¹(δ−μ_δ)` (`C=argmin Σ‖P h_i − C z(δm_i)‖²+λ‖C‖²`)과 통계량·SHA를 결과 관측 전 핀. confirmatory 대체 불가 — 별도 exploratory estimand.

### B — critic 입력 확장 (**기각**)

`Q([h_p,u,δm])` 류는 frozen `_QHead` 입력 계약·checkpoint shape 변경, BB parity 파괴(BB는 plain TD3Agent — `sprint_arms.py:319-322`), estimand 자체가 상이("h_p 매개" → "δm 입력 재학습 critic 활용"), 전 arm 재학습 필요. 현 sprint 제외 — 후속 architecture arm의 별도 사전등록 안건.

## 3. Cross-arm 동형성·null control

- 전 arm의 critic 공통 경계가 `(h_p,u)` — A0는 전 arm 동일 적용(BB의 f_resp 부재는 무관). A1도 연산상 동형이나 P의 학습 의미는 matched arm 한정 — arm 간 동일 δm semantics 주장 금지.
- Null donor: 정본대로 `δm(s_d′)=δm(s_r)`·micro-state만 상이·goal/anchor/호길이 동일 — 각 실개입(A0/A1)과 동일 operator·동일 정규화의 paired null.
- 해석 한정: full donor swap은 비-δm latent nuisance 동반 가능 — 결론은 "response-conditioned h_p mediation"으로 제한 서술.

## 4. 구현 경계

전용 frozen intervention evaluator 신설: `features→encoder→same-p h_p→critic` 학습 경로 재사용, donor/recipient 텐서·operator·parameter pre/post hash·arm/ckpt/split SHA 기록. latent extraction API의 read-only/frozen 계약 불변.

## 5. 게이트 판정 요청 항목 (human 핀 — 결과 관측 전)

1. **선택지**: (a) A0 confirmatory 단독 / (b) A0 confirmatory + A1 exploratory / (c) +A2(calibration 핀 포함) / — B는 기각 확인.
2. **사상·정규화**: A0/A1 무정규화 확인, A2 유지 시 N의 fit dataset·정칙화·SHA.
3. **null control 구성** 승인 (§3 그대로).
4. **lock**: 본 결정문·수식·코드/spec SHA를 핀하기 전 patch-eval 산출물 생성·열람·unblinding 금지 확약.
