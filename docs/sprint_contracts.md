# Sprint contracts — G2a §②–⑧

정본: `.gjc/_session-019f729f-defd-7000-96d5-1c65711d5ffb/plans/ralplan/019f729f-defd-7000-96d5-1c65711d5ffb/pending-approval.md` §G2a 항목 ②–⑧. 이 문서는 그 항목의 수치·명칭·절차를 동결한다. 구현 전 계약이며, 아래의 **소스 대조**는 현 baseline의 사실을 기록한다.

## ② V1 `f_resp` 텐서·학습 계약

- 입력: `concat[h_p (B,256), u (B,4)] = (B,260)`.
- 구조: `Linear(260,256) → ReLU → Linear(256,256) → ReLU (=z_resp, B,256) → Linear(256,24) = δm̂`.
- 감독값: `δm = Φ_DCT(X_after) − Φ_DCT(X_before)`.
- `Φ_DCT`는 `src/dgcc/phi/dct.py`의 `M=8`, axis-major 24ch를 사용한다. 재구현 금지.

### Adapter joint step

baseline `critic_update()`는 loss/backward/clip/step을 내부 소유하므로 adapter가 다음을 명시 override한다.

```text
q_loss  = critic_loss(batch)                # baseline 수식 그대로 (target 계산 포함)
aux     = f_resp(h_p, u); aux_loss = mse(aux, delta_m)
loss    = q_loss + 1.0 * aux_loss
opt.zero_grad(); loss.backward()
clip_grad_norm_(encoder+critic+f_resp 파라미터 합집합, 10.0)
opt.step(); soft_update(baseline 모듈만)     # f_resp target망 없음
```

- `λ=1.0`.
- optimizer는 기존 encoder+critic optimizer에 `f_resp` param group을 추가하며 lr은 동일하다.
- clip param set은 encoder+critic+`f_resp` 파라미터 합집합이며 10.0이다.
- soft update는 baseline 모듈만 대상으로 한다. actor/target 경로는 무오염이다.

### 소스 대조

- `src/dgcc/models/networks.py:40-46`은 `U_DIM=4`, `EMBED_DIM=256`을, `:137-148`은 encoder 출력 `(B,32,256)`을 명시한다. 선택된 `h_p` 폭 256은 이 출력에서 따른다.
- `src/dgcc/models/networks.py:151-167`은 critic 입력 `torch.cat([h_p,u])` 및 `Linear(EMBED_DIM + U_DIM,256)`을 사용한다.
- `src/dgcc/rl/td3.py:300-345`에서 baseline `critic_update()`가 target 계산, loss, `zero_grad`, `backward`, gradient clip, `step`을 내부 수행한다. `:376-402`는 target soft update가 `update()`에서 baseline target modules에 수행됨을 보인다.
- `src/dgcc/phi/dct.py:18-30,47-60`은 `M=8`, `PHI_DIM=24`, axis-major `[x0..x7,y0..y7,z0..z7]`, `Phi_DCT`를 확인한다.
- **불일치:** 현재 baseline에는 `f_resp`/`z_resp` 및 adapter override가 없다. 이는 본 신규 sprint 계약의 구현 대상이며 baseline 변경 근거가 아니다.

## ③ matched-dim 계약

- target은 EMA target encoder를 next state `s′`에 적용한 `h_p′`이다.
- `p′*`는 기존 double-Q decoupling 규칙, 즉 `Q_target_1` argmax 규칙을 그대로 사용한다.
- target `h_p′`에 고정 무작위 직교 projection `P∈R^{24×256}`을 적용하여 24ch target을 만든다.
- `P`: seed **20260719**, Gaussian→QR, config에 명기한다.
- stop-grad, EMA `τ=0.005`, S1–S3 동일 적용.
- head는 V1과 동형이며 파라미터 수가 일치한다.

### 소스 대조

- `src/dgcc/rl/td3.py:62-70`은 `tau=0.005`와 gradient clip 10.0을, `:250-277`은 target encoder의 next-state `h_next` 및 `Q_target_1` argmax (`select_p_star`)를 확인한다.
- `src/dgcc/rl/td3.py:87-92`은 `select_p_star`가 Q1 target candidates만으로 argmax함을 명시한다. `:376-384`은 EMA 대상이 encoder/critic/actor baseline 모듈임을 확인한다.
- **불일치:** projection `P`, seed 20260719, stop-grad matched-dim target, 동형 head는 현재 baseline에 없다. 신규 arm 구현 대상이다.

## ④ random-target 계약

- 사전등록 seed **20260718**의 `N(0,1)` 24ch 고정 벡터를 사용한다.
- 이 벡터는 전 구간·전 run에서 불변이다.
- config에 seed·분포·스케일을 명기한다.

### 소스 대조

- `src/dgcc/phi/dct.py:18-22`는 24ch 차원(`M=8`, `PHI_DIM=3*M`)을 확인한다.
- **불일치:** seed 20260718 random target과 그 config는 현재 baseline에 없다. 신규 arm 구현 대상이다.

## ⑤ patching 단일 개입 계약

- 개입 수식: **`Q(s_r,g,p,u; h_p ← h_p(s_d))`**. donor 상태 `s_d`의 `h_p`를 recipient forward의 동일 `p` 위치에 이식한다.
- donor는 simulator에서 다음을 만족하도록 구성한다: `oracle δm(s_d) = rescale(δm_target)`.
  - 재스케일 마스크: x/y modes 1–7 × `L_ood/L_train` 스케일.
  - `x0/y0/z0` 및 z modes는 절대 유지.
- 비-δm nuisance (goal, anchor 위치, 호길이 파라미터화)는 recipient와 매칭한다.
- null control은 `δm(s_d′)=δm(s_r)`이고 micro-state만 다른 donor다. Q-ranking 변화 ≈0 기대를 통해 δm-특이 효과를 격리한다.
- 순수 aux 출력(`δm̂`) patch는 기각 유지한다. Q 경로 밖이다.
- necessity는 동일 `h_p`의 mean-ablation 마스킹이다.
- estimator: per-point Q-ranking 변화는 Kendall `τ` + top-1 flip율, 성능은 patch-only split one-shot rollout의 success/return이다.

### 소스 대조

- `src/dgcc/analysis/latent_api.py:152-207`은 실제 training feature path, encoder, selected `h_p` 및 Q1/Q2 trunk/출력 추출 경로를 제공한다.
- `src/dgcc/models/networks.py:161-171,174-183`은 critic이 `h_p,u`에서 Q를 계산함을 확인한다.
- `src/dgcc/phi/dct.py:5-7,27-30`은 mode 0 centroid와 axis-major channel order를 확인한다.
- **불일치:** donor simulator 구성, rescale mask, `h_p` interchange/mean-ablation, null control 및 patch-only rollout estimator는 현재 source에 없다. 신규 patching 구현 대상이다.

## ⑥ `t2_patch_eval_v1` 계약

A8 승인 조건부 patch-only 평가 split은 `src/dgcc/tasks/splits/t2_patch_eval_v1.json`이다.

- T2 생성기 신규 seed + OOD 길이 변형(§4 sufficiency-under-shift 범위)을 사용한다.
- sprint-heldout·M4-heldout과 파라미터-수준 불교차를 검증한다.
- grid unblinding(G9a lock) 전에 커밋한다.
- one-shot rollout 규약을 사용하며 G-EV claim primitive를 재사용하고 전용 접근 로그를 남긴다.

### 소스 대조

- `src/dgcc/tasks/splits/t2_patch_eval_v1.json`은 현재 존재하지 않는다.
- **불일치:** 신규 split, generator seed/OOD 길이, 불교차 검증, one-shot rollout 및 G-EV claim/access log는 아직 source에 없다. 이 계약은 생성·검증·G9a 이전 커밋의 요구사항이다.

## ⑦ checkpoint schema v2 + latent API v2 계약

- checkpoint schema v2는 baseline 6-module payload를 보존하고 `sprint_arm` namespace를 추가한다.
- legacy BB checkpoint load regression을 보장한다.
- `LATENT_SPEC` 기존 키는 불변이며 `z_resp`를 versioned 확장한다.
- round-trip/frozen/injection 테스트를 계약한다.

### 소스 대조

- `src/dgcc/rl/td3.py:492-535`은 baseline checkpoint의 6 modules (`encoder`, `critic`, `actor`, `encoder_target`, `critic_target`, `actor_target`)와 optimizer payload 저장·로드를 확인한다.
- `src/dgcc/analysis/latent_api.py:34-47`은 현재 `LATENT_SPEC` 키를, `:67-114`는 frozen checkpoint load를, `:118-147`은 6-module parameter digest와 metadata를, `:190-207`은 키별 extraction/shape assertion을 확인한다.
- **불일치:** schema v2, `sprint_arm` namespace, versioned `z_resp`, legacy regression 및 injection 지원은 현재 source에 없다. 기존 `LATENT_SPEC` 키는 변경하지 않고 확장해야 한다.

## ⑧ probe H5 versioned schema + content-addressed manifest 계약

probe H5는 versioned schema로 다음 필드를 모두 생산한다.

```text
{x_before, x_after, goal, goal_id, p, u, episode_id, step_index,
 truncated/reseed/guard flags, ckpt sha256, split sha256, claim sha256}
```

- content-addressed manifest: `outputs/metrics/sprint_probe_manifest.json`.
- manifest에는 파일별 sha256·size·생산 goal을 기록하고 원자 게시한다.
- G5/G6a/G6b가 생산하며 G10은 manifest만 소비한다.

### 소스 대조

- `src/dgcc/analysis/latent_api.py:138-147`은 현재 extraction metadata에 `ckpt_sha256` 및 `latent_spec`을 포함하고, `:59-64`는 파일 SHA-256 helper를 제공한다.
- `src/dgcc/analysis/latent_api.py:152-207`은 현재 추출 API가 `X`, goal curve, `p`, delta, lift 및 latent tensors를 다룸을 확인한다.
- **불일치:** versioned probe H5의 전 필드, `split sha256`/`claim sha256`, content-addressed manifest, 파일 size/생산 goal, 원자 게시 및 G10 manifest-only 소비는 현재 source에 없다. 신규 evaluator/patching 구현 대상이다.
