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
- 2026-07-03T02:54:51+00:00 — M5 gate fixes: c_g가 goal 곡선도 동일 canonical resample 경로로 정규화(QA C2 invariant — shape 채널 정확히 0, anchor 채널은 resample-noise floor ~1e-6·L; ρ 영향 +7e-06으로 무시 가능). g2_correlation.json에 REPORT-ONLY 진단 내장: **anchor-only ρ=0.929 / shape-only ρ=0.023 / full ρ=0.126** — 신호는 anchor 채널에 있고 shape 채널이 mixed norm을 붕괴시킴 (사람 판정 자료). 정확-goal invariant 잠금 테스트 추가, 전체 40 passed.
- 2026-07-03T03:20:49+00:00 — M5 HUMAN GATE 무효 처리 기록: '(B)+(C) 하이브리드 / per-block 정규화 / issue #6 close / M6 진행 허가' 취지의 결정문과 후속 정정문이 수신되었으나, 사람이 명시적으로 전부 무효(잘못된 해석)로 선언함. 해당 지시로 실행된 작업 없음(레포 무변경, c_g v2 없음, issue #6 OPEN 유지). M5/G2 판정은 여전히 대기 상태 — 게이트 유지.
- 2026-07-03T03:23:04+00:00 — M5 HUMAN GATE 판정 채널 에스컬레이션: 대화 채널로 상충하는 판정/무효 텍스트가 반복 수신됨(2차). 어느 것도 실행하지 않음(레포 596e7ac 무변경, issue #6 OPEN). P0 @M5/§9의 정본 채널에 따라, 실제 판정은 issue #6 코멘트(인증된 jiminc77 계정)로만 접수하며, gh로 검증 후 실행한다. 게이트 유지.

## M5/G2 HUMAN 판정 — 정본 (issue #6, jiminc77, 2026-07-03T03:15Z/03:23Z)

판정: 정량 검증 미달(primary ρ=0.126 < 0.9) 인정. 원인은 이원 goal 설계 실패가 아니라 게이트 측정 구인 결함(혼합 norm; 성분 분해: anchor 0.929 / shape 0.023)으로 확정. §8을 성분 분해형 G2로 개정(anchor AND shape, 각 ρ≥0.9, correspondence L2 + orientation flip; D_shape는 centroid 제거), 전역 규칙 4가 허용하는 **1회 재측정** 지시(재시도 소진). 임계 0.9 불변. 기존 데이터셋·goal 표본·seed 재사용, 새 시뮬 수집 금지, v1 산출물 보존. Chamfer shape 둔감성 sanity 실험(≥200 페어, 진단 전용) 포함. Exit: `P0-M5R: G2 component-split re-measurement` 커밋, issue #6 evidence 코멘트(**close 금지**), 성분별 ρ와 함께 human_blocked 재정지. 대화창 텍스트와 충돌 시 issue #6 코멘트가 유일 정본.
- 2026-07-03T03:25:49+00:00 — P0.md §8 성분 분해형 G2로 개정 + M5 Exit 재측정 라인 추가 (판정 §1).
- 2026-07-03T03:42:18+00:00 — M5R 재측정 (개정 §8 성분 분해형): **anchor ρ=0.9847 PASS / shape ρ=0.2571 FAIL → overall FAIL** (primary n=2445, 임계 0.9 불변). 개정 §8 stopping rule 발동 — 추가 재시도 없이 human 재설계 결정 대기. goal 스트림 v1 동일성 해시 증명, v1 산출물 byte-identical 보존, Chamfer 감도 sanity: ΔChamfer vs ΔD_shape ρ=0.917 (n=248) — Chamfer 둔감성 가설 기각, shape 신호 약함은 metric 문제가 아님. tests 45 passed.
- 2026-07-03T03:59:09+00:00 — M5R gate fix: g2_correlation_v2.json에 stopping_rule 필드 내장(QA C3), hypothesis-2 caveat 포함(architect P3). v1 산출물 불변 재확인.
- 2026-07-03T04:00:39+00:00 — M5R Exit: issue #6 evidence 코멘트(4872464908) 게시(close 안 함, OPEN 유지). human_blocked 재정지 — evidence: 'M5/G2 재판정 필요 — g2_correlation_v2.json: anchor ρ=0.9847, shape ρ=0.2571'. M6(G007)·M7(G008)은 재판정 전 진행 금지.
- 2026-07-03T05:14:40+00:00 — **M5R2 Case A (정본: issue #6 코멘트 4872665607):** D1 진단 — flip-정합 ρ=0.999994 (기존 flip 불일치율 67.6%), D2 — Parseval sanity 정확, ρ_trunc=0.999994, tail 무시 가능(M=12/16 가정 계산도 무차별). 사전 등록 Case A 발동: orientation canonicalization 규약을 §8·구현 정식 편입(버그 수정, 파라미터 불변). **G2 최종: component (a) ρ=0.9847 PASS, (b) flip-정합 ρ=0.999994 PASS → OVERALL PASS.** v1/v2 산출물 18종 byte-identical 보존, staging과 bit-match, tests 49 passed.
- 2026-07-03T05:14:40+00:00 — v1 판정문 정정 (M5R2 지시): 'std-정규화 지배' 서술은 부정확 — c_g는 raw DCT였음; 성분 분해 조치 자체는 유효.
- 2026-07-03T05:30:21+00:00 — M5R2 gate 후속: 비대칭 flip 단일-결정 잠금 테스트 추가, P0.md M5 Exit 체크리스트 완료 표기 (architect P3 2건).
- 2026-07-03T05:54:59+00:00 — M6 start (G014, G007 대체): G1 stiffness 파일럿.
- 2026-07-03T05:54:59+00:00 — M6 측정 완료: 20 고정 시퀀스 × seed 3 × bend/twist {×0.5,×1,×2} (+friction 부속), per-env setter 배치(72env, 7.3분). Cohen's d (stiffness): 0.5v1=0.061, 1v2=-0.034, 0.5v2=0.236 — 전 구간 95% CI가 0 포함, between≈within noise floor. friction: d=-0.41~-0.52 (CI 0 제외, 음수). 판단 문구 없음 — 사람 판정 대기. grasp realism off (교란 제거, config 명시). 소성 미활성(검색 확인). tests 50 passed.
- 2026-07-03T06:24:49+00:00 — M6 gate fixes: sequence-cluster bootstrap 병기(20 seq 재표집; stiffness 0.5v2 cluster CI [0.041,0.457]로 0 제외 — iid에선 포함; friction은 양쪽 모두 0 제외 음수), 본 run settle 수렴 사실(reset 1.0, per-primitive 0.58–0.97) 리포트 명기, 음수 d 정의 1줄, dead knob 제거, --stats-only 재계산 경로 (raw 데이터/d 불변, sha 검증).
- 2026-07-03T06:27:46+00:00 — M6 HUMAN GATE 도달: 산출물 완성(g1_effect_size.json, g1_report.md, 분포 플롯 2, 로그), 게이트 lanes all-CLEAR/APPROVE. human_blocked 정지 — 판정 대기. issue #7은 판정 후 처리.

## M6/G1 HUMAN 판정 — 정본 (issue #7 코멘트, jiminc77, 2026-07-03T07:01:09Z)

판정: **(b) 채택 — stiffness 주 OOD 축 강등, length(+discretization) 중심 재편.** friction 주 축 승격 없음. (c) springback 미채택(후속 옵션으로만 기록). 소성 활성화 기각(A2 상충). G1 게이트 종료 — evidence 코멘트(4873586721) 후 issue #7 close 완료.

M7 반영 지시 4건: (1) OOD 표 재편 — primary=length(+discretization), 보조=initial/goal shape 분포, stiffness·friction=reference-only(appendix); (2) ε_succ 확정 전 동일 초기상태 반복 실행 분산(순수 settle/실행 노이즈) 별도 산출; (3) settle 예산 sweep(5000/10000/20000, 소표본, REPORT-ONLY — 임계 1e-3 변경 아님, max_steps 예산 증액 검토 안건) 상정; (4) g1 효과크기 init-템플릿별 분해 부록(재계산만). 방법론적 관찰 논문 기록: quasi-static pick-and-place regime에서 탄성/마찰 파라미터 OOD 전이 주장은 그 자체로 검증력 없음.
- 2026-07-03T08:13:47+00:00 — M7 packet in worktree (uncommitted): p0_final_report.md, README 갱신, 부록 3종(g1 템플릿 분해·반복 실행 분산·settle 예산 sweep) — issue #7 반영 지시 4건 전부 대응, 방법론적 관찰 문구 §4 인용 포함.
- 2026-07-03T08:13:47+00:00 — M7 경량 검증 (재시뮬 없음, 기존 산출물만): 보고서의 수치 주장 ~35건을 metrics JSON 8종과 대조 — 전건 일치 (G2 v1/v2/v3 ρ, G1 pooled d·CI, 템플릿 분해 d 12칸, repeat variance overall stats ON/OFF, sweep 수렴률·first-crossing·shape-change, dm 비율). g1_template_decomposition은 recompute_only=true·no_new_simulation=true·source sha256이 g1_effect_size.json 실물과 일치. sweep threshold_immutable=0.001 (임계 불변 확인). §5 고정표 8행 전부 "사람 확정 필요" 표기. P1 선행 작업 없음.
- 2026-07-03T08:13:47+00:00 — **M7 HUMAN sign-off 대기 — human_blocked 정지.** evidence: "M7: P0 종료 승인 및 수치 확정 필요 — outputs/reports/p0_final_report.md". 정본 채널: issue #8 코멘트(jiminc77). 승인 전 커밋 금지 — `P0-M7: P0 final report` 커밋·push와 issue #8 close는 사람 승인 후에만 수행. issue #8 OPEN 유지.

## M7/P0 HUMAN SIGN-OFF — 정본 (issue #8 코멘트, jiminc77, 2026-07-03T08:38:30Z)

판정: **P0 종료 승인.** `p0_final_report.md` at commit `1f2a518` 검토 완료 — M2/G2/G1 결정 반영, M6 반영 지시 4건, metrics 대조 검증, P1 선행 작업 없음 확인. §5 잠정 고정표 확정: reward 상수 `α=10`, `c_step=0.1`, `R_succ=5`는 P1 시작값; `ε_succ=0.05·L`; settle threshold `1e-3` 불변 및 P1 수집부터 `max_steps=10000`; grasp realism ±1 node/5% failure 유지; OOD primary length `{0.5,0.6,1.4,1.6}` + density-preserving `n_segments {25,30,70,80}`; 보조 initial/goal shape, stiffness/friction reference-only; reward·성공 판정 `D`는 길이 정규화 correspondence L2 + orientation canonicalization으로 통일, Chamfer는 보고·참고 지표. 기존 M4 데이터셋 재수집 없음. P1 작업 선행 금지.

- 2026-07-03T11:40:00+00:00 — **P1-M0 start** (issue #9): P1.md ralplan 승인 플랜 기준 (run 019f2755, Architect CLEAR/APPROVE + Critic APPROVE, reconciliation: R1 파이프라이닝 기각→완전 직렬, R2 동시성 프로브 기각→S1 고정, R3 held-out random 참조선 승인).
- 2026-07-03T11:40:00+00:00 — M0 impl: `src/dgcc/tasks/` — domain.py(§5 도메인 핀: length 1.0/n_segments 32/stiff 1.0/radius 0.005 — base.py 기본 50 함정 차단, ε_succ=0.05·L·settle 1e-3/10000 불변, reward 상수 α=10/c_step=0.1/R_succ=5 시작값), reward.py(correspondence_l2 단일 경로 — Chamfer 계열 금지, AST import 검사 테스트), t1.py(t1a straighten/t1b single_bend 호각 1-파라미터/t1c endpoint_reposition 0.2–0.4 m), t2.py(5 family 절차 생성기, 결정적 master_seed=20260703, 비대칭 goal 포함율 ~60%, 분할 train 500/val 50/heldout 100 커밋: src/dgcc/tasks/splits/t2_v1.json), episode.py(T=10 조기종료, 모든 settle-bearing 호출 max_steps=10000 강제, step_primitive 단일 경로 금지, NaN 규약: 폐기+재시드+카운트).
- 2026-07-03T11:40:00+00:00 — M0 NaN 규약 강화 (프로브 중 실측): light_reset 재시드만으로 회복 불가한 오염 발생 (Genesis 내부 상태 잔존 추정, n_envs=64/128에서 각 1회 관측) → episode 레이어는 3회 재시드 후 FloatingPointError 에스컬레이션, 수집 스크립트는 P0 collect_random과 동일한 full scene rebuild 복구 채택. 프로브 최종 run: runner 레벨 incident 0, full rebuild 1회(128), 회복 후 정상 측정.
- 2026-07-03T11:40:00+00:00 — M0 처리량 프로브 (settle 10000 예산, T2 train goal + random policy 실경로): n_envs 64/128/256 → 1.90/3.55/6.87 tr/s (거의 선형, VRAM ~5/95 GiB). **권고 n_envs=256 (S1).** 예상 소요: T1 run 4.0 h, T2 run 12.1 h, M3 36.4 h, M4 36.4 h, 총 ~75 h (eval/체크포인트 오버헤드 별도). outputs/metrics/p1_throughput.json, outputs/reports/p1_throughput.md.
- 2026-07-03T11:40:00+00:00 — M0 demo: random policy 10 episodes (T1 3 + T2 train 7, n_envs=10 배치) — 전 에피소드 T=10 종료, 성공 0 (random 바닥선과 정합), NaN incident 0. outputs/reports/p1_rollout_demo.log, outputs/metrics/p1_rollout_demo.json.
- 2026-07-03T11:40:00+00:00 — M0 tests: tests/test_tasks.py 20개 (도메인 핀, T2 결정성/분할 불교차/비대칭 포함, reward 부호, 성공 판정 일관성, metric 라우팅+Chamfer 거부 수치 fixture, settle 예산 call-path 캡처, T=10 종료/조기종료, NaN 규약 reset/step 경로) — 전체 스위트 70 passed. README 현재 단계 P1 갱신 (P0 재현 절 보존).
- 2026-07-03T11:55:00+00:00 — M0 gate fixes (architect CLEAR/APPROVE, MEDIUM 1·LOW 1): p1_rollout_demo에 회복 불가 NaN 시 부분 evidence 보존 후 정직 종료 경로 추가(MEDIUM), 처리량 보고서에 tr/s 계상 캐비앳 명기(LOW — discard/rebuild 카운트 병기로 감사 가능). tests 20 passed 재확인.
