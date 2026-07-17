# P1 체크포인트 매니페스트 (M5 — P2 인수 인터페이스)

> 파일들은 git 제외 (`outputs/models/` gitignored). 본 매니페스트가 커밋되는 유일한 색인이다.
> sha256은 전체 다이제스트를 수록한다. init_hash = F-a `initial_weights_sha256` 앞 8자 (M3R 시대 run은 미계측 — F-a는 M4-prep 커밋 1db7913부터).
> 체크포인트 내부 포맷: `TD3Agent.save_checkpoint` payload (config/reward_constants/td_target_bound/metadata/update_count + encoder/critic/actor + targets + optimizers). 로드는 `FrozenLatentExtractor.from_checkpoint` (frozen) 또는 `TD3Agent.load_checkpoint` (학습 재개 불가 — resume 코드 경로 없음, 관례 유지).

## M3 (M3R 통제 재수행 — 최종 M3 결과, issue #12)

config: `configs/p1_t1_{a,b,c}.yaml` · budget 1e5 tr · n_envs 256 · 학습 데이터: 해당 run의 on-policy 수집 (fresh, settle 10000) · best.pt = best-on-val(태스크별 100 episodes) 유지 규약 · run git = 학습 시점 커밋 (`p1_run_*.json` `git_commit`)

| run | best ckpt (경로) | best sha256 | best val succ @tr | final ckpt (경로) | final sha256 | run git |
|---|---|---|---|---|---|---|
| m3r_t1a_s0 | `outputs/models/m3r_t1a_s0/best.pt` | 4d6e5d0591dfa76f9f86f47e9a4aae75a3d2c8581d33521fed614dd51c7d7f00 | 0.47 @100,096 | `outputs/models/m3r_t1a_s0/ckpt_0100096.pt` | a7e68be3c805a48e364997e66acc77d1b9b588bb84b4e5d3175f8be40e0f07c7 | 5d4822c77dfd5558cdadba8164c0cb9b166ee8a4 |
| m3r_t1a_s1 | `outputs/models/m3r_t1a_s1/best.pt` | abbe1e38fe3b86764dab70307e43fdff068260df890e6bd448680abde054c260 | 0.30 @100,096 | `outputs/models/m3r_t1a_s1/ckpt_0100096.pt` | be3de843e352c0387b4a6112c2e0c5d17d070e91c34b00609666e18f83da58f6 | 3ac95feed7c4d42901e59d363d62ec643418572e |
| m3r_t1a_s2 | `outputs/models/m3r_t1a_s2/best.pt` | 9169ac934291617663b0699bee5b7407ee8a974f24411f08103422cfc1e79634 | 0.33 @100,096 | `outputs/models/m3r_t1a_s2/ckpt_0100096.pt` | aa9f5e076b99b72a3b98fa7c92d2773b9c9e10c3c46835b9932077ba11028de6 | c0fb3a4a0b52ee61674f52fefc0affb109eb1970 |
| m3r_t1b_s0 | `outputs/models/m3r_t1b_s0/best.pt` | b756a87ca4f27cf32661ae085d708d5a2c52b2fa11f0adcc434b3024d7ae51d8 | 0.00 @25,088 | `outputs/models/m3r_t1b_s0/ckpt_0100096.pt` | 58dc51a4f4bf17b3369a6ff176d898c1994cb2a2399accee92a4e56514c421fc | 848346c9e7b68f8c2c598db8553c7af85620eeb0 |
| m3r_t1b_s1 | `outputs/models/m3r_t1b_s1/best.pt` | 41088309c0d9156601ef9b13b2c72588515f1159c066927c5773e0d99eb6741c | 0.08 @25,088 | `outputs/models/m3r_t1b_s1/ckpt_0100096.pt` | 129d230910ea3e135bd62fd94e6899288f5d948350de51b8bde4438b8c8c898e | 606544f7ee73616164c4f7cd22d453aaa84e3460 |
| m3r_t1b_s2 | `outputs/models/m3r_t1b_s2/best.pt` | 21b02eac553694039011f5d6d80456dd3e808ca9a7b5271303c3d8b547efbd28 | 0.06 @25,088 | `outputs/models/m3r_t1b_s2/ckpt_0100096.pt` | edc5640a7ca70674fb7c5ad94b19903bf9868f5fa82e78691d9d3678452735c6 | b8153a7485eed18b551db2dd2dff008065c59360 |
| m3r_t1c_s0 | `outputs/models/m3r_t1c_s0/best.pt` | 45e62d3279e2279410a4984f585fcce9d96c86970a2b0bfad6a306f1f9ac6841 | 0.00 @25,088 | `outputs/models/m3r_t1c_s0/ckpt_0100096.pt` | 7faf65194ee279f5c59f0554eb9d8e18f6e8d816dd4a769e21f6718c466b1d9d | b36f8654ab4fe742681c80167f969a1478067ab1 |
| m3r_t1c_s1 | `outputs/models/m3r_t1c_s1/best.pt` | 590ee9d6c8a06e95fde97acff266e8a0ce9806e24d4bc0e28e6c485f5acdc5b9 | 0.00 @25,088 | `outputs/models/m3r_t1c_s1/ckpt_0100096.pt` | ebd4da20976ea54ea484cc79ec3c6d35b3d8a8226663150979fc492a6e2fae4c | 975010bcae617a61de071a997673fc3e967e995b |
| m3r_t1c_s2 | `outputs/models/m3r_t1c_s2/best.pt` | 479f1b6a73384591400e20882bc723fbb4d7fada90c3dfd1ab9ec3e4ef6ba582 | 0.01 @75,008 | `outputs/models/m3r_t1c_s2/ckpt_0100096.pt` | f4bfb268dccefc9c95c101b37a1d04c63d0af2982446f3c7dd948d74b9fddb00 | b2119a6f50ca728d36ef1b3924b5a6e5d9adaed0 |

주: t1b/t1c의 0-성공 run에서 best.pt는 동률 규약(최초 eval)에 따른 것 — M3R 보고서(`p1_m3r_results.md`)의 판정 문맥과 함께 읽을 것. M3R↔M4 same-seed 비교성은 F-a로 공식 단절 (M4 보고서 §8). GNG-1 처분: t1b/t1c "미학습·판정 불능" (그리드 제외).

## M4 (T2 goal-conditioned, issue #13)

config: `configs/p1_t2.yaml` · budget 3e5 tr · n_envs 1024 (스모크 사다리 판정) · 학습 데이터: on-policy 수집 (T2 train 500 goals) · 선택 = val-최대 규칙 (`p1_m4_ckpt_selection_*.json`, held-out 비접근)

| run | 선택 ckpt (경로) | 선택 sha256 | val | HO | final ckpt | final sha256 | init_hash | run git |
|---|---|---|---|---|---|---|---|---|
| m4_t2_s0 | `outputs/models/m4_t2_s0/ckpt_0300032.pt` | 6512f5e7e5e956efdf2f6e511002bafc3395a374f4862c17c89c757c1928f913 | 0.24 | 0.145 | `outputs/models/m4_t2_s0/ckpt_0300032.pt` | 6512f5e7e5e956efdf2f6e511002bafc3395a374f4862c17c89c757c1928f913 | a2ff322e | cdd73e2480d082052c70dce323c5b3e80a68d1fe |
| m4_t2_s1 (rerun) | `outputs/models/m4_t2_s1/ckpt_0275456.pt` | 1e2f665621147de8a472f7efcad3eac91dc7753ae54cb8c586f24217bbd873a1 | 0.19 | 0.240 | `outputs/models/m4_t2_s1/ckpt_0300032.pt` | 636214c1c462cbd2669b881436617ecce90d3bc94df1952ca1927c7e8a1ff02d | 06912e7d | b24997fed396e8b294b0526d351a43dddcb5cb4c |
| m4_t2_s2 | `outputs/models/m4_t2_s2/ckpt_0300032.pt` | 96d07ac4396416865c32d416e3054522ffd232c2ce5ecd5e0ace4a781784a86d | 0.33 | 0.325 | `outputs/models/m4_t2_s2/ckpt_0300032.pt` | 96d07ac4396416865c32d416e3054522ffd232c2ce5ecd5e0ace4a781784a86d | 1b619736 | 786d651a4b0f6013971bf1d8f23b125062223679 |

선택 규칙·근거 eval row는 `outputs/metrics/p1_m4_ckpt_selection_m4_t2_s{0,1,2}.json` (커밋됨) 참조.

## 아카이브 (참고 — 사용 금지)

`*.crashed-*`, `*.interrupted-*`, `*.stuck-*`, `*.halted-*` 접미 디렉터리는 인시던트 보존본 (manifest sha256 별도), pre-M3R `t1*` 디렉터리는 M3 1차 시도 흔적 — 어느 것도 P2 인수 대상이 아니다.

## M5 latent 추출 실행 기록 (exit 증빙)

`scripts/extract_latents.py`를 위 12개 best 체크포인트 전부에 대해 T2 val 표본
(`outputs/data/p1_t2_val_sample.h5`, 250 records, random policy, seed 0)으로 실행 — 12/12 성공,
산출 `outputs/data/latents/<run>.h5` (gitignored; meta_json에 ckpt sha256/config/git hash/표본 sha256 수록).
