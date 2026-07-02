# DGCC — Implementation

**Deformation-Grounded Contact Critic** 구현 코드 레포. 연구 관리·문서는 [research-dashboard](https://github.com/jiminc77/research-dashboard) 참조.

## 현재 단계: P0 (환경·파일럿)

- 실행 명세: [`P0.md`](P0.md) — gajae-code(gjc)가 `ralplan → ultragoal`로 실행하는 brief. Milestone = `@goal` 블록 (M0–M7), 각각 GitHub issue #1–#8에 대응.
- 실행 환경: `ssh AILAB-simx-remote` → `/home/simx2204/Workspaces/DGCC` (RTX 6000, Ubuntu 22.04, headless)
- HUMAN GATE: M2 (primary sim 결정), M5 (G2), M6 (G1), M7 (sign-off) — gjc는 측정·보고까지만 수행하고 `human_blocked`로 정지한다.

## 실행 (참고)

```bash
cd /home/simx2204/Workspaces/DGCC
gjc ralplan --interactive "P0.md 명세를 읽고 실행 계획 수립"
gjc ultragoal create-goals --brief-file P0.md
```

## 재현 (P0-M1 기준)

```bash
uv venv --python 3.12 .venv && uv pip install -e .
# DLO-Lab lane (M1): genesis-world 1.0.0은 로컬 clone에서 editable 설치
git clone https://github.com/UMass-Embodied-AGI/DLO-Lab external/DLO-Lab
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -e "external/DLO-Lab[dlo-lab]"
# smokes (headless)
MUJOCO_GL=egl uv run python scripts/smoke_mujoco.py --seed 0
uv run python scripts/smoke_dlolab.py --seed 0
uv run python scripts/compare_sims.py --seed 0
```

주의: `outputs/` 산출물의 `commit_hash` 메타데이터는 실행 시점 HEAD를 기록하므로, 해당 코드가 포함된 커밋의 부모 해시일 수 있다 (run-then-commit).

## 구조 (P0-M0에서 생성됨)

```
src/dgcc/{envs,phi,goals,logging,utils}/
scripts/  configs/  tests/
outputs/{data,metrics,plots,reports}/   # data는 git 제외
STEP_LOG.md                             # 모든 milestone 기록
```

## 규칙 (P0.md 전역 규칙 발췌)

명세 밖 구현 금지 · 모호성은 human_blocked로 에스컬레이션 · 게이트 임계 변경 금지 · milestone 단위 커밋 `P0-M<k>: <요약>` · 대용량 데이터/asset 커밋 금지.
