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
- 2026-07-02T23:16:58+00:00 — M3 start: primary(DLO-Lab) adapter를 §5 인터페이스로 완성.
- 2026-07-02T23:16:58+00:00 — M3 impl: grasp realism(±1 node, 5% fail, config off), 4 init shapes(해석 곡선+seeded noise, rod_entity.set_position 직접 배치), param sweep 반영 확인, tests/test_primitive.py 14 passed(68s), demo 10 primitives(log+plot+meta). 1000-draw failure stat 4.9% ∈ [4,6]%.
- 2026-07-02T23:16:58+00:00 — M3 note: MuJoCo adapter 동결 유지(변경 없음); legacy init 'bent' 제거 → smoke_dlolab은 u_bend + realism off 사용.
- 2026-07-02T23:16:58+00:00 — M3 complete (게이트 검증 후 issue #4 close).
- 2026-07-02T23:34:01+00:00 — M3 정정: 'test_primitive.py 14 passed' 표기는 전체 tests/ 스위트 기준(14) — test_primitive.py 자체는 8개(현재 9개, friction 응답 테스트 A4 추가)였음 (architect A8).
- 2026-07-02T23:34:01+00:00 — M3 gate 후속 fix: sample_grasp 경계 클램프 semantics docstring(A3), grasp 실패 분기 settle_converged 실측정(A5), move 이중 클램프 제거 _move_prepared(A7), friction 동역학 응답 테스트(A4). M4 설계 입력으로 기록: 비수렴 transition은 settle_steps==max_steps로 판별해 필터/플래그(A1, 스키마 변경 없이), 대량 수집 전 경량 reset 경로/scene teardown 필요(A2).
- 2026-07-03T01:50:07+00:00 — M4 start: §7 Φ/δm 파이프라인 + transition 수집.
- 2026-07-03T01:50:07+00:00 — M4 phi: DCT-II ortho, layout axis-major-xyz-modes-0-7-v1 (mode0=centroid 분리, mode≥1 21ch), 불변성 실측 max rel err 1.73% < 2% (N=25 vs 100). normalize: mode≥1만 std 스케일, tiny-std는 raise.
- 2026-07-03T01:50:07+00:00 — M4 writer: h5py 컬럼형 레이아웃, TransitionWriter 증분 append, config+commit meta, round-trip/slice-read 테스트 8개.
- 2026-07-03T01:50:07+00:00 — M4 수집: 5,056 transitions (n_envs=64 배치, per-env grasp via attach_to_rigid_link_with_envs_idx, 31분). success 94.7%, settle 수렴 53.7% (A1: settle_steps==max로 판별, 통계 분리; normalizer는 converged-success 2,445건 fit). outputs/data/p0_random_transitions.h5 (gitignore). 실패 env는 계약대로 X_after==X_before 정확 복원 (문서화).
- 2026-07-03T02:10:16+00:00 — M4 gate fixes: step_primitive_batch가 복원 전 free-drift를 실측/보고(restoration_drift_max/mean_m); 실측 프로브(32env, 6500 step round-equivalent) — 삭제되는 drift 상한 max 3.36mm(비수렴 t0)/0.77mm(수렴 t0), rope 길이의 ≤0.34%로 primitive 변형(20~150mm) 대비 무시 가능 → 복원은 정직한 계약 집행으로 확정. dm_stats에 physics_quality_note(+success_and_converged 48.4% headline)와 drift probe 증거 내장(--stats-only 재계산 경로 추가, 수집 로그 비파괴).
- 2026-07-03T02:33:59+00:00 — M5 start: §8 이원 goal·거리 구현과 G2 측정.
- 2026-07-03T02:33:59+00:00 — M5 impl: DualGoal(template 정규화: centroid 제거+단위 호길이; anchor 기본 centroid[O5], endpoint 선택), c_g 24ch(21 shape mode≥1 + 3 anchor), 길이 정규화 bidirectional Chamfer. tests/test_goal_distance.py 6 passed, 전체 39 passed.
- 2026-07-03T02:33:59+00:00 — **M5/G2 측정 결과 (있는 그대로): primary Spearman ρ=0.126 (n=2445 converged-success) — 임계 0.9 대폭 미달.** variants: all-success ρ=0.161(n=4786), all-transitions ρ=0.165(n=5056). 임계/정의 무변경; g2_correlation.json에 PROPOSAL(사람 결정 필요)만 기재. 정성 자료 9장 + scatter 생성.
