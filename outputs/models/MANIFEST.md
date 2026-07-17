# P1 체크포인트 매니페스트 (M5 — P2 인수 인터페이스)

> 파일들은 git 제외 (`outputs/models/` gitignored). 본 매니페스트가 커밋되는 유일한 색인이다.
> sha16 = 체크포인트 파일 sha256 앞 16자. init_hash = F-a `initial_weights_sha256` 앞 8자 (M3R 시대 run은 미계측 — F-a는 M4-prep 커밋 1db7913부터).
> 체크포인트 내부 포맷: `TD3Agent.save_checkpoint` payload (config/reward_constants/td_target_bound/metadata/update_count + encoder/critic/actor + targets + optimizers). 로드는 `FrozenLatentExtractor.from_checkpoint` (frozen) 또는 `TD3Agent.load_checkpoint` (학습 재개 불가 — resume 코드 경로 없음, 관례 유지).

## M3 (M3R 통제 재수행 — 최종 M3 결과, issue #12)

config: `configs/p1_t1_{a,b,c}.yaml` · budget 1e5 tr · n_envs 256 · 학습 데이터: 해당 run의 on-policy 수집 (fresh, settle 10000) · best.pt = best-on-val(태스크별 100 episodes) 유지 규약

| run | best.pt sha16 | best val succ @tr | final succ | git (run 시점) |
|---|---|---|---|---|
| m3r_t1a_s0 | 4d6e5d0591dfa76f | 0.47 @100,096 | 0.47 | 5d4822c |
| m3r_t1a_s1 | abbe1e38fe3b8676 | 0.30 @100,096 | 0.30 | 3ac95fe |
| m3r_t1a_s2 | 9169ac9342916176 | 0.33 @100,096 | 0.33 | c0fb3a4 |
| m3r_t1b_s0 | b756a87ca4f27cf3 | 0.00 @25,088 | 0.00 | 848346c |
| m3r_t1b_s1 | 41088309c0d91566 | 0.08 @25,088 | 0.01 | 606544f |
| m3r_t1b_s2 | 21b02eac55369403 | 0.06 @25,088 | 0.06 | b8153a7 |
| m3r_t1c_s0 | 45e62d3279e22794 | 0.00 @25,088 | 0.00 | b36f865 |
| m3r_t1c_s1 | 590ee9d6c8a06e95 | 0.00 @25,088 | 0.00 | 975010b |
| m3r_t1c_s2 | 479f1b6a73384591 | 0.01 @75,008 | 0.00 | b2119a6 |

주: t1b/t1c의 0-성공 run에서 best.pt는 동률 규약(최초 eval)에 따른 것 — M3R 보고서(`p1_m3r_results.md`)의 판정 문맥과 함께 읽을 것. M3R↔M4 same-seed 비교성은 F-a로 공식 단절 (M4 보고서 §8).

## M4 (T2 goal-conditioned, issue #13)

config: `configs/p1_t2.yaml` · budget 3e5 tr · n_envs 1024 (스모크 사다리 판정) · 학습 데이터: on-policy 수집 (T2 train 500 goals) · 선택 = val-최대 규칙 (`p1_m4_ckpt_selection_*.json`, held-out 비접근)

| run | 선택 ckpt | sha256 (full, selection manifest 수록) | val succ | held-out succ | init_hash |
|---|---|---|---|---|---|
| m4_t2_s0 | ckpt_0300032.pt | 6512f5e7… | 0.24 | 0.145 | a2ff322e |
| m4_t2_s1 (rerun) | ckpt_0275456.pt | (manifest 참조) | 0.19 | 0.240 | 06912e7d |
| m4_t2_s2 | ckpt_0300032.pt | (manifest 참조) | 0.33 | 0.325 | 1b619736 |

전체 sha256·선택 규칙·근거 eval row는 `outputs/metrics/p1_m4_ckpt_selection_m4_t2_s{0,1,2}.json` (커밋됨)이 원본.

## 아카이브 (참고 — 사용 금지)

`*.crashed-*`, `*.interrupted-*`, `*.stuck-*`, `*.halted-*` 접미 디렉터리는 인시던트 보존본 (manifest sha256 별도), pre-M3R `t1*` 디렉터리는 M3 1차 시도 흔적 — 어느 것도 P2 인수 대상이 아니다.

## M5 latent 추출 실행 기록 (exit 증빙)

`scripts/extract_latents.py`를 위 12개 best 체크포인트 전부에 대해 T2 val 표본
(`outputs/data/p1_t2_val_sample.h5`, 250 records, random policy, seed 0)으로 실행 — 12/12 성공,
산출 `outputs/data/latents/<run>.h5` (gitignored; meta_json에 ckpt sha256/config/git hash/표본 sha256 수록).
