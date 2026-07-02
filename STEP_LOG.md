# STEP_LOG

- 2026-07-02T17:00:31+00:00 — M0 start.
- 2026-07-02T17:02:00+00:00 — Created P0 §4 workspace skeleton, pyproject, stubs, base interface, transition schema, and tests.
- 2026-07-02T17:02:00+00:00 — Created uv Python 3.12 virtualenv and installed editable package with M0 dependencies.
- 2026-07-02T17:02:00+00:00 — Verified `uv run pytest tests/` and base import one-liner.
- 2026-07-02T17:02:00+00:00 — gh CLI 없음 — issue #1 수동 처리 필요.
- 2026-07-02T17:02:00+00:00 — M0 complete.
- 2026-07-02T17:15:12+00:00 — M0 review-gate fixes: writer.py stub callables (QA C5), resample.py M1-minimal/M4-finalize docstring note; tests 3 passed + red-team 34 passed.
- 2026-07-02T18:13:46+00:00 — M1 start: two-sim bring-up (MuJoCo-first order per approved plan).
- 2026-07-02T18:13:46+00:00 — M1 MuJoCo lane: adapter+smoke 7/7 PASS (gravity+ground-plane scene; MuJoCo 3.10 name-scheme handled via enumeration; viscosity hack removed after review).
- 2026-07-02T18:13:46+00:00 — M1 DLO-Lab lane: install SUCCESS under 2h timebox (torch 2.10.0+cu128, genesis-world 1.0.0, pins numpy<2.5/fsspec<=2026.2.0/packaging<26.0; assets HTTP 401 — not needed); smoke 8/8 PASS. M1 failure-halt rule NOT triggered.
- 2026-07-02T18:13:46+00:00 — M1 comparison: 5 seq x 3 seeds x 2 sims; MuJoCo settle 0/30 converged @5000 steps vs DLO-Lab 30/30; report outputs/reports/sim_comparison.md.
- 2026-07-02T18:13:47+00:00 — gh CLI 없음 — issue #2 수동 처리 필요.
- 2026-07-02T18:13:47+00:00 — M1 complete.
- 2026-07-02T18:27:43+00:00 — M1 gate fixes: [dlo-lab] extra + commented compat pins in pyproject, README 재현 section, settle metric-definition caveat in sim_comparison.md (architect MEDIUM advisories).
- 2026-07-02T18:31:35+00:00 — M1 code-lane LOW fixes: resample rejects degenerate zero-length centerline (+test), compare_sims raises on missing velocity metric, genesis one-shot seed semantics documented.

## M2 HUMAN GATE — primary sim 결정 요청 (2026-07-02T18:45Z)

M1 비교 결과 요약 (outputs/reports/sim_comparison.md, 결정 문구 없음 — 판단은 사람 몫):

- **smoke:** 두 sim 모두 통과 (MuJoCo 7/7, DLO-Lab 8/8). MuJoCo 단독 통과 상황 아님 — 비대칭 없음.
- **MuJoCo cable 장점:** 설치 단순(pip 1개), 성숙한 코드베이스, 결정성 검증됨(동일 seed 2회 bit-identical), 의존성 리스크 낮음. **단점:** compare 시나리오에서 settle 수렴 0/30 (5000 step, max_abs_qvel<1e-3 기준 — 관절공간 메트릭이라 DLO-Lab 수치와 정의가 다름, 보고서 캐비앳 참조), CPU 단일 프로세스(병렬화 없음, MJX cable 미지원), primitive 평균 5.8 s.
- **DLO-Lab 장점:** settle 수렴 30/30 (평균 1472 step, max_node_speed 기준), GPU 배치(n_envs=4 검증) — 향후 RL 데이터 수집에 유리, 파라미터 런타임 setter 풍부(소성 포함, P0에선 비활성), primitive 평균 4.8 s. **단점:** 공개 5주차 외부 코드(ti_float 버그 런타임 alias 필요), asset SharePoint 401, 의존성 pin 취약성(torch/genesis/numpy/fsspec/packaging), wall-time 분산 큼(max 12 s).
- **파라미터화:** 양쪽 모두 length/bend/twist/friction 커버. 소성은 DLO-Lab만 (미활성).
- 상세 수치·플롯: outputs/reports/sim_comparison.md, outputs/metrics/sim_comparison_metrics.json, outputs/plots/compare_*.png

gh CLI 없음 — issue #3 수동 처리 필요 (결정 후 코멘트+close).

- 2026-07-02T22:48:52+00:00 — **M2 HUMAN DECISION: (A) DLO-Lab primary.** 사람이 재개 지시로 명시. 이후 모든 파이프라인(M3~)은 DLOLabEnv adapter만 사용; MuJoCo adapter는 M1 상태로 동결(삭제 금지). issue #3 수동 처리 필요: 결정 코멘트 + close (gh CLI 없음).

- 2026-07-02T22:51:11+00:00 — **북키핑 정정:** gh CLI 사용 가능해짐(/usr/bin/gh, jiminc77 인증). 이전 "gh CLI 없음 — issue #1/#2/#3 수동 처리 필요" 노트는 폐기. issue #1, #2는 evidence와 함께 close 완료 확인; issue #3은 인간 결정 코멘트(4871069692) 위에 evidence 코멘트(4871088553) 추가 후 close 완료. 이후 milestone부터 전역 규칙 6의 gh 경로 정상 사용.
- 2026-07-02T22:51:11+00:00 — **운영 노트 (DLO-Lab asset):** assets/dlo-lab.zip 확보됨 (149MB, gitignore 대상 — 커밋 금지). 향후 DLO-Lab 렌더링/datagen 시 LuisaRender 경로를 올바르게 연결하고 공식 datagen을 --raytracer 플래그로 실행할 것.
