# DGCC — Implementation

**Deformation-Grounded Contact Critic** 구현 코드 레포. 연구 관리·문서는 [research-dashboard](https://github.com/jiminc77/research-dashboard) 참조.

## 현재 단계: P1 진행 중 (M0–M3 완료, M4 진행)

- 실행 명세: [`P1.md`](P1.md) — P0 위에 HACMan-style black-box contact critic baseline을 구축하는 P1 brief. Milestone = `@goal` 블록 (M0–M6), GitHub issue #9–#15 대응. 커밋 규약 `P1-M<k>: <요약>`.

| P1 milestone | issue | 상태 |
|---|---|---|
| M0 task layer / M1 networks·TD3 / M2 smoke·계측 | #9 · #10 · #11 | done |
| M3 T1 본학습 (M3R 통제 재수행 경유 — verdict choice B) | #12 (closed, gate archive) | done |
| M4 T2 본학습 (n_envs=1024, 스모크 관문 통과; s0 완주 24%, s1 재실행 진행, s2 대기) | #13 | running |
| M5 latent API / M6 sign-off·reward 잠금 | #14 · #15 | ready (M6는 GNG-2 전 close 금지) |

- paper-sprint 사전등록: Decision rd#35/#36 · sprint_spec@`82230d8` · 에픽 #17 — sprint 전용 held-out `t2_sprint_heldout_v1.json` (M4 held-out은 P1 판정 전용).
- P1 산출물 (M0–M3): `src/dgcc/tasks/` (T1/T2 task·episode·reward), `src/dgcc/models/`+`src/dgcc/rl/` (§6 네트워크, §7 TD3 decoupled double-Q, replay v2, §8 계측), 커밋된 T2 분할 `src/dgcc/tasks/splits/t2_v1.json`, `outputs/reports/p1_m3r_results.md` (M3R 9/9), `outputs/metrics/p1_random_reference.json`, F-a/F-b 재현성 수정 (`initial_weights_sha256`, rebuild-독립 eval ordinal).
- 수치 정책 (issue #8 sign-off 승계): 불변 — ε_succ=0.05·L, settle 1e-3/10000, grasp realism ±1node/5%, D = 길이 정규화 correspondence L2 + orientation canonicalization (Chamfer는 보고용), K=32, M=8. 조정 허용 (STEP_LOG 기록, M6 잠금) — α=10, c_step=0.1, R_succ=5, RL 하이퍼파라미터.
- HUMAN GATE: M2 (스모크 2회 실패 시), M3/M4 (판정 미달 분기 + M4 HER 중간 체크), M6 (sign-off + reward 상수 잠금).

## P0 (완료)

- 실행 명세: [`P0.md`](P0.md) — gajae-code(gjc)가 `ralplan → ultragoal`로 실행한 P0 brief. Milestone = `@goal` 블록 (M0–M7), GitHub issue #1–#8 대응.
- 최종 보고서: [`outputs/reports/p0_final_report.md`](outputs/reports/p0_final_report.md)
- 실행 환경: `ssh AILAB-simx-remote` → `/home/simx2204/Workspaces/DGCC` (RTX 6000, Ubuntu 22.04, headless)
- HUMAN GATE: M2 (primary sim 결정), M5 (G2), M6 (G1), M7 (sign-off). P0는 issue #8 HUMAN SIGN-OFF로 종료 승인되었다.

## P0 Milestone 상태

| Milestone | Issue | 대표 커밋 / 산출물 | 상태 |
|---|---:|---|---|
| M0 bootstrap | #1 | `a35a9cc`, `3182bc3`, `4390ec7`; package skeleton, base interfaces, schema tests | closed |
| M1 two-sim bring-up | #2 | `e2bda5b`, `2c29bb1`, `88e6572`; `outputs/reports/sim_comparison.md` | closed |
| M2 primary sim gate | #3 | `0fa0a67`, `201a846`; human decision: DLO-Lab primary | closed |
| M3 primitive/API | #4 | `12c67e4`, `d2f148d`; `DLOLabEnv`, grasp realism, init shapes, parameter sweeps | closed |
| M4 logging/δm | #5 | `53ffca7`, `af6773f`; `outputs/metrics/dm_stats.json`, gitignored h5 dataset | closed |
| M5 G2 gate | #6 | `f4cab54` plus M5R/M5R2 commits through `d595fa8`; `outputs/metrics/g2_correlation_v3.json` | closed after M5R2 PASS |
| M6 G1 gate | #7 | `6da316a`, `dc715b4`, `e146c71`, `b1652c5`; `outputs/reports/g1_report.md` | closed with verdict (b) |
| M7 final sign-off | #8 | `1f2a518`, follow-up sign-off decision commit; `outputs/reports/p0_final_report.md` | closed with HUMAN SIGN-OFF |

## 실행 (참고)

```bash
cd /home/simx2204/Workspaces/DGCC
# P1 (현재)
gjc ralplan --interactive "P1.md 명세를 읽고 실행 계획 수립"
gjc ultragoal create-goals --brief-file P1.md
# P0 (완료)
gjc ralplan --interactive "P0.md 명세를 읽고 실행 계획 수립"
gjc ultragoal create-goals --brief-file P0.md
```

## 재현 (P1 산출물)

```bash
# P1-M0 task layer + throughput probe + rollout demo
uv run pytest tests/test_tasks.py -q
uv run python scripts/throughput_probe.py --seed 0 --config configs/p1_throughput.yaml
uv run python scripts/p1_rollout_demo.py --seed 0 --config configs/p1_rollout_demo.yaml
```

## 재현 (P0 산출물)

```bash
# environment
uv venv --python 3.12 .venv
uv pip install -e .

# DLO-Lab lane (external clone is gitignored; assets are not committed)
git clone https://github.com/UMass-Embodied-AGI/DLO-Lab external/DLO-Lab
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -e "external/DLO-Lab[dlo-lab]"

# suite
uv run pytest tests/ -q

# M1 smokes/comparison
MUJOCO_GL=egl uv run python scripts/smoke_mujoco.py --seed 0
uv run python scripts/smoke_dlolab.py --seed 0
uv run python scripts/compare_sims.py --seed 0

# M3 primitive demo
uv run python scripts/demo_primitive.py --seed 0 --config configs/demo_primitive.yaml

# M4 transition/δm pipeline
uv run python scripts/collect_random.py --seed 0 --config configs/collect_random.yaml
uv run python scripts/collect_random.py --stats-only --config configs/collect_random.yaml

# M5 G2 gate history
uv run python scripts/gate_g2.py --seed 0 --config configs/gate_g2.yaml
uv run python scripts/gate_g2.py --seed 0 --config configs/gate_g2.yaml --v2
uv run python scripts/gate_g2.py --seed 0 --config configs/gate_g2.yaml --v3

# M6 G1 gate
uv run python scripts/gate_g1.py --seed 42 --config configs/gate_g1.yaml
uv run python scripts/gate_g1.py --stats-only --config configs/gate_g1.yaml

# M7 appendix artifacts
uv run python scripts/appendix_g1_decomposition.py --config configs/gate_g1.yaml --stats-only-style
uv run python scripts/appendix_repeat_variance.py --config configs/gate_g1.yaml --seed 7301
uv run python scripts/appendix_settle_sweep.py --config configs/gate_g1.yaml --seed 8401 --n-cases 24
```

주의: `outputs/` 산출물의 `commit_hash` 메타데이터는 실행 시점 HEAD를 기록하므로, 해당 코드가 포함된 커밋의 부모 해시일 수 있다 (run-then-commit). `outputs/data/`, DLO-Lab external clone, assets, 대용량 h5는 커밋하지 않는다.

## 구조

```
src/dgcc/{envs,phi,goals,logging,utils}/   # P0-M0에서 생성
src/dgcc/tasks/                            # P1-M0: T1/T2 task·episode·reward 레이어 (+ splits/t2_v1.json)
scripts/  configs/  tests/
outputs/{data,metrics,plots,reports}/   # data는 git 제외
STEP_LOG.md                             # 모든 milestone 기록
P0.md  P1.md                            # 단계별 명세
```

## P0 종료와 P1 경계

P0는 simulator selection, common interface, logging/δm, G2/G1 pilot measurement, final sign-off report까지 완료했다. issue #8 HUMAN SIGN-OFF로 `outputs/reports/p0_final_report.md`의 §5 수치표가 확정되었다. P1은 [`P1.md`](P1.md) 명세로 진행 중이며, P1 명세 밖 항목(P2+ probe 실험, DGCC variants, f_resp/response head, GreedyResp, matched-dim latent, OOD 평가 sweep, 실로봇)은 선행하지 않는다.

## 규칙 (P0/P1 전역 규칙 발췌)

명세 밖 구현 금지 · 모호성은 human_blocked로 에스컬레이션 · 게이트 임계/불변 수치 변경 금지 · milestone 단위 커밋 `P<phase>-M<k>: <요약>` · 대용량 데이터/asset 커밋 금지 · reward/성공 판정에 Chamfer 사용 금지 (correspondence L2만).