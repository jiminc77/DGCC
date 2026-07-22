# G6a BB 종결 리포트

생성: `uv run python scripts/generate_sprint_g6a_report.py`. 수치는 생성 시 원천 JSON/로그에서 추출했다. BB 사실만 기록하며 unblinding 전 성능 비교·V1 추론은 하지 않는다.

## 1. Held-out (7 seed)
|seed|구분|선택 ckpt/sha256|claim sha256|success|mean return|원천|
|-:|---|---|---|--:|--:|---|
|0|reuse|`outputs/models/m4_t2_s0/ckpt_0275456.pt` / `ffe4fa1fb2b3da7b25cc2ed2d7f23eeaeb64742453e5b4abf02a3ffe0793af8e`|`80736ee0db31abdfe83f5dca7d6fc7e2edbadf65c7d4031f7374796a34990430`|0.275|1.962038|`outputs/metrics/p1_t2_sprint_heldout_m4_t2_s0.json`|
|1|reuse|`outputs/models/m4_t2_s1/ckpt_0300032.pt` / `636214c1c462cbd2669b881436617ecce90d3bc94df1952ca1927c7e8a1ff02d`|`4d046cabdbdeb1d2ff2b60c98b1859e905d8d4bcbc1c346eef29f17f206ff85f`|0.245|1.698587|`outputs/metrics/p1_t2_sprint_heldout_m4_t2_s1.json`|
|2|reuse|`outputs/models/m4_t2_s2/ckpt_0125952.pt` / `63db0e6331298f960ec278466bab735710d1740558e79128bdf656021e8c3f8e`|`cc37b6e641d7b63a4d838d6f6be16e12595d3edc8ddf0fcdc1901fc24450291d`|0.300|2.027144|`outputs/metrics/p1_t2_sprint_heldout_m4_t2_s2.json`|
|3|new|`/home/simx2204/Workspaces/DGCC/outputs/models/sprint_t2_bb_s3/ckpt_0275456.pt` / `84d16c24d2323f7281a8b3e5e21199a07d33e02e530d4fd5b6672ec5d81dc7ab`|`b0e1f3fac7e16a99c48bddeb59884cb65d3b758877ff0a1fabe0c40e5dd11de3`|0.070|0.611288|`outputs/metrics/p1_bb_sprint_heldout_sprint_t2_bb_s3.json`|
|4|new|`/home/simx2204/Workspaces/DGCC/outputs/models/sprint_t2_bb_s4/ckpt_0300032.pt` / `a445b04d07a2c1cb2494b7f0a1c538d0872982d1e9661231d00619a57d717203`|`92b952b7d27b932fae181241eb437fa609e8c14fbad98f6a13a2dda49bee1e84`|0.225|1.594325|`outputs/metrics/p1_bb_sprint_heldout_sprint_t2_bb_s4.json`|
|6|new|`/home/simx2204/Workspaces/DGCC/outputs/models/sprint_t2_bb_s6/ckpt_0300032.pt` / `cfb61c5cabf3533595ead71451d5647e6de06bb4aaa8ffd1567c98681151f071`|`01ac48fa9752e384c3b1f064e86be2a87d2d349e418eb02a91d05b01ea299b6d`|0.025|-0.647203|`outputs/metrics/p1_bb_sprint_heldout_sprint_t2_bb_s6.json`|
|7|new|`/home/simx2204/Workspaces/DGCC/outputs/models/sprint_t2_bb_s7/ckpt_0200704.pt` / `7e7c025265e8773a202610134487c9996979169bbb56008b5e602651acf38f7b`|`b4a2a8bb13c3e10c8b365df5e00917f3e8c5fa421cb9de061cdd464e38211356`|0.100|0.645181|`outputs/metrics/p1_bb_sprint_heldout_sprint_t2_bb_s7.json`|

## 2. 학습 궤적 및 eval-wall 감시
|seed|transitions|final val success|final val return|wall h|eval-wall max s|<3600s 전건|
|-:|--:|--:|--:|--:|--:|---|
|3|300032|0.150|0.467498|9.65|1094.435|yes|
|4|300032|0.200|1.443344|9.35|1174.542|yes|
|6|300032|0.040|-0.648595|9.46|1374.935|yes|
|7|300032|0.060|-0.096360|10.98|1235.547|yes|

## 3. Stability (A-4 재집계)
`outputs/metrics/sprint_g6a_stability.json`은 현행 4건과 아카이브 crash/kill 전부를 `--log`와 동등한 개별 `recount_log` 호출로 재집계한다. reported는 로그 종결 행, recounted는 rebuild/reset 경계 합산 하한이다.
|log|상태|reported nan/mag|recount 하한 nan/mag|rebuilds|경계|
|---|---|---|---|--:|---|
|`outputs/archive/sprint_crash/s5-20260719T1835Z/p1_sprint_train_sprint_t2_bb_s5.log.crashed-20260719T1835Z`|archived|—/—|3/8|9|9|
|`outputs/archive/sprint_crash/s5r-20260721T0107Z/p1_sprint_train_sprint_t2_bb_s5.log.crashed-20260721T0107Z`|archived|—/—|0/1|9|9|
|`outputs/archive/sprint_crash/s6-killed-20260720T0312Z/p1_sprint_train_sprint_t2_bb_s6.log.killed-20260720T0312Z`|archived|—/—|12/13|3|3|
|`outputs/archive/sprint_crash/s7-killed-20260721T0559Z/p1_sprint_train_sprint_t2_bb_s7.log.killed-20260721T0559Z`|archived|—/—|6/10|1|1|
|`outputs/archive/sprint_crash/s7r-20260722T0117Z/p1_sprint_train_sprint_t2_bb_s7.log.crashed-20260722T0117Z`|archived|—/—|0/1|9|9|
|`outputs/reports/p1_sprint_train_sprint_t2_bb_s3.log`|complete|20/7|82/69|1|1|
|`outputs/reports/p1_sprint_train_sprint_t2_bb_s4.log`|complete|24/20|45/38|1|1|
|`outputs/reports/p1_sprint_train_sprint_t2_bb_s6.log`|complete|0/0|37/31|3|3|
|`outputs/reports/p1_sprint_train_sprint_t2_bb_s7.log`|complete|29/14|48/28|2|2|

인프라/기술 사건 분류: ENOSPC=s7 attempt 1, reaper/job-cancel=s3 launch 및 s6 kill, rebuild-limit=s5×2 및 s7r. 이 분류는 STEP_LOG 사건 기록이며 학습 halt와 구분한다.

## 4. Settle-pocket 및 경합-상관 대조
|계열|포켓 지속|경합 기록|결과|
|---|---:|---|---|
|s0|3.66h|기준 전례|완주|
|s5|5.2h|무경합|rebuild-limit crash|
|s5r|12h|경합 추정|rebuild-limit crash|
|s7r|12.5h|경합 집중 창|rebuild-limit crash|
|s6/s6r|3-chain / 없음|경합 중 완주 사례|s6 kill 후 s6r 완주|
STEP_LOG 시각 대조에서는 포켓 지속과 경합 기록을 병기하며 인과 귀속은 하지 않는다 — 미통제 관찰이다. batch-effect는 사전등록 3-way 감도분석 대상이다.

## 5. AMD-3 및 F-a
AMD-3 (verdict comment 5029426419): seed 5 페어(BB+V1)를 기술 결함으로 제외하고 대체하지 않는다; BB 평균이 상향되는 방향이므로 V1−BB 델타에는 보수적 방향의 민감도 노트다.
|seed/attempt|initial_weights_sha256|byte-일치 증거|
|---|---|---|
|sprint_t2_bb_s3|`b1ee2f73778061694a5bb5977d53c144bd0f3a67eb556083cf9d0a51eb76a910`|run JSON/아카이브 run JSON|
|sprint_t2_bb_s4|`b31ccdf019d52483020210da9f6e647b945d0b25e3552a7eb86c23f21203311e`|run JSON/아카이브 run JSON|
|sprint_t2_bb_s5|`d96ec5cba24c38016b5e0db31a9d979a4984c2eba43a563f596d12fb89e6e710`|run JSON/아카이브 run JSON|
|sprint_t2_bb_s6|`00d53f65687288d083f03c80c2bb1a10d967e9e5fa9157ba1814c6d9940ead67`|run JSON/아카이브 run JSON|
|sprint_t2_bb_s7|`5ce71f924ad3f185e95ea6e58c00d7205d2c1812c6d24ef4f80a33ab254380c9`|run JSON/아카이브 run JSON|
s5 original↔retry, s6 kill↔retry, s7 3 attempts의 F-a byte-일치는 STEP_LOG 사건 기록으로 교차 확인한다.

## 6. 재시도 회계
|seed|attempt 이력|종결|
|---|---|---|
|5|crash → retry crash|AMD-3 제외|
|6|job-cancel kill → fresh retry|완주|
|7|ENOSPC kill → rebuild-limit crash → fresh retry|완주|

## Limitations
포켓 진입은 확률적이며, 경합 환경은 통제 실험이 아니다. 재사용 3 seed에는 retro probe가 없어 mechanism 분석 표본에 포함하지 않는다. held-out는 위 one-shot 결과만 사용한다.
