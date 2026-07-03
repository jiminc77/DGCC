# DGCC — Implementation

**Deformation-Grounded Contact Critic** 구현 코드 레포. 연구 관리·문서는 [research-dashboard](https://github.com/jiminc77/research-dashboard) 참조.

## 현재 단계: P0 완료

- 실행 명세: [`P0.md`](P0.md) — gajae-code(gjc)가 `ralplan → ultragoal`로 실행한 P0 brief. Milestone = `@goal` 블록 (M0–M7), GitHub issue #1–#8 대응.
- 최종 보고서: [`outputs/reports/p0_final_report.md`](outputs/reports/p0_final_report.md)
- 실행 환경: `ssh AILAB-simx-remote` → `/home/simx2204/Workspaces/DGCC` (RTX 6000, Ubuntu 22.04, headless)
- HUMAN GATE: M2 (primary sim 결정), M5 (G2), M6 (G1), M7 (sign-off). P0는 issue #8 HUMAN SIGN-OFF로 종료 승인되었다. P1은 별도 명세로 시작한다.

## Milestone 상태

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
gjc ralplan --interactive "P0.md 명세를 읽고 실행 계획 수립"
gjc ultragoal create-goals --brief-file P0.md
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

## 구조 (P0-M0에서 생성됨)

```
src/dgcc/{envs,phi,goals,logging,utils}/
scripts/  configs/  tests/
outputs/{data,metrics,plots,reports}/   # data는 git 제외
STEP_LOG.md                             # 모든 milestone 기록
P0.md                                   # P0 명세
```

## P0 종료와 P1 경계

P0는 simulator selection, common interface, logging/δm, G2/G1 pilot measurement, final sign-off report까지 완료했다. issue #8 HUMAN SIGN-OFF로 `outputs/reports/p0_final_report.md`의 §5 수치표가 확정되었다. P1은 별도 명세로 시작하며, 이 레포에서 P1 명세 없이 RL 학습 루프, actor/critic, replay buffer, baseline port, probe experiment를 선행하지 않는다.

## 규칙 (P0.md 전역 규칙 발췌)

명세 밖 구현 금지 · 모호성은 human_blocked로 에스컬레이션 · 게이트 임계 변경 금지 · milestone 단위 커밋 `P0-M<k>: <요약>` · 대용량 데이터/asset 커밋 금지.