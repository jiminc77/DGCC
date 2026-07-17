# P1-M5 Latent API — P2 사용 설명서

P2 probing이 소비하는 frozen-critic latent 추출 인터페이스. 코드 원본:
`src/dgcc/analysis/latent_api.py` (`LATENT_SPEC`이 이름·shape 계약의 단일 원천 —
본 문서와 불일치 시 코드가 우선하며 같은 커밋에서 동기화할 것).

## 계약 요약

- 체크포인트는 **frozen**으로 로드된다: 전 모듈 eval mode + `requires_grad_(False)`.
  `parameter_sha256()`으로 전후 불변성을 검사할 수 있다 (`extract_latents.py`는 매 실행 자동 검사).
- 추출은 학습 코드 경로를 그대로 통과한다: §6 입력 계약 (`build_node_features` — canonical flip 포함),
  encoder forward, `_QHead.forward(return_hidden=True)`. 수학 재구현 없음.
- goal 조건화는 오직 encoder 입력 residual 채널로만 들어간다 (§6) — latent에 별도 goal 채널은 없다.

## 층 이름·shape (LATENT_SPEC)

| 이름 | shape | 의미 |
|---|---|---|
| `encoder_h` | (B, 32, 256) | per-node h_i (local 128 ⊕ global 128) — P2 "encoder per-node h_i" |
| `h_p` | (B, 256) | 선택 노드 p의 embedding |
| `q1_trunk_hidden1/2` | (B, 256) | critic Q1 trunk post-LN post-ReLU 층 1/2 — P2 "critic trunk 중간층" |
| `q2_trunk_hidden1/2` | (B, 256) | critic Q2 동일 |
| `q1`, `q2`, `q_min` | (B,) | 최종 Q heads on [h_p, u]; q_min = min(q1, q2) |
| `flip_before` | (B,) bool | feature 구축에 사용된 canonical flip 결정 |

## 사용법

```python
from dgcc.analysis.latent_api import FrozenLatentExtractor

ex = FrozenLatentExtractor.from_checkpoint("outputs/models/m4_t2_s2/ckpt_0300032.pt")
lat = ex.extract(X, G_curve, p, delta, lift)   # lift: "high"/"low" 또는 0/1
```

배치 추출 (v2 transition h5 → latent h5):

```bash
uv run python scripts/extract_latents.py \
    --checkpoint outputs/models/m4_t2_s2/ckpt_0300032.pt \
    --transitions outputs/data/p1_t2_val_sample.h5 \
    --out outputs/data/latents/m4_t2_s2.h5
```

출력 h5: LATENT_SPEC 이름별 dataset + join-back 열 (`p`, `delta`, `lift`,
`episode_id`, `step_index`, `goal_id`) + `meta_json` attrs (ckpt sha256 · agent
config · git hash · 입력 표본 sha256/meta · generated_at).

## P2 Controls A–F가 요구하는 연결

- **latent**: 위 출력 h5.
- **대응 transition**: 입력 v2 h5 (`dgcc.rl.replay.read_v2_transitions`) — 출력의
  join-back 열과 행 순서가 1:1 동일 (같은 파일의 같은 인덱스).
- **δm 계산 경로**: P0 §8 metric 경로 재사용 — `dgcc.goals.distance` (correspondence
  L2 + canonicalization). δm 계산·예측 실험 자체는 P2 범위이며 P1 코드에는 없다.

## 체크포인트 색인

`outputs/models/MANIFEST.md` — M3(M3R) 9 + M4 3 best 체크포인트의 sha256/성능/config/git hash.

## 금지 (P1 범위)

probe 학습 (ridge/MLP 등), δm 예측 실험, latent 해석 주장 — 전부 P2에서.
