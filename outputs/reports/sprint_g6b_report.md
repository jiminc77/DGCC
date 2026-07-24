# G6b 캠페인 감시 리포트

생성: `uv run python scripts/generate_sprint_g6b_report.py`. 원천 아티팩트만 스캔한 증분 감시표이며, unblinding 전 arm 간 비교·해석은 포함하지 않는다.

## Per-run 감시

|arm|seed|상태|transitions|halt|nan/mag/rebuild|wall h|init hash|eval-wall max s|
|-|-:|---|--:|---|--:|---|--:|
|v1|0|complete|300032|—|23/23/4|14.72|`a2ff322eb9cfd650d78a976422d372d331562b0d893a2c87914faa692b72b10a`|11138.240|
|v1|1|complete|300032|—|86/59/3|18.20|`06912e7d30d471d57834a8504f10648011f58911f04edefc8078e893a9ec36e8`|6277.605|
|v1|2|complete|300032|—|107/100/2|10.24|`1b619736a4679b318c77533fd78168b218c841ce76328377e8f0fac404fb353f`|529.697|
|v1|3|observed|—|—|0/0/0|—|`—`|—|
|v1|4|pending|—|—|—/—/—|—|`—`|—|
|v1|6|pending|—|—|—/—/—|—|`—`|—|
|v1|7|pending|—|—|—/—/—|—|`—`|—|
|matched|0|pending|—|—|—/—/—|—|`—`|—|
|matched|1|pending|—|—|—/—/—|—|`—`|—|
|matched|2|pending|—|—|—/—/—|—|`—`|—|
|matched|3|pending|—|—|—/—/—|—|`—`|—|
|matched|4|pending|—|—|—/—/—|—|`—`|—|
|random|0|pending|—|—|—/—/—|—|`—`|—|
|random|1|pending|—|—|—/—/—|—|`—`|—|
|random|2|pending|—|—|—/—/—|—|`—`|—|
|random|3|pending|—|—|—/—/—|—|`—`|—|
|random|4|pending|—|—|—/—/—|—|`—`|—|

## Held-out

|arm|seed|상태|success|mean return|ckpt / sha256|claim sha256|원천|
|---|-:|---|--:|--:|---|---|---|
|v1|0|complete|0.215|1.532166|`/home/simx2204/Workspaces/DGCC/outputs/models/sprint_t2_v1_s0/ckpt_0250880.pt` / `6b2b138606b9cf2bff03af286728c0ae93ee544f6d79a2f595f9b0635c53f305`|`52878be75013006e489fbeaebc98db8be2753eb9183d98461925648e9a56d630`|`outputs/metrics/p1_v1_sprint_heldout_sprint_t2_v1_s0.json`|
|v1|1|complete|0.120|0.962135|`/home/simx2204/Workspaces/DGCC/outputs/models/sprint_t2_v1_s1/ckpt_0225280.pt` / `cb4a8f514a968c12cc983980db783ce62e5bbda829eec65f4577776ce2eb4805`|`6eea2b4833653a47b3ac07f91b488087a129eeabab3112bb9ed33af8e7063511`|`outputs/metrics/p1_v1_sprint_heldout_sprint_t2_v1_s1.json`|
|v1|2|complete|0.330|2.243659|`/home/simx2204/Workspaces/DGCC/outputs/models/sprint_t2_v1_s2/ckpt_0300032.pt` / `f78fe7bddb0702d86ee1509340629de1c24ca2657f1d44984932b2f2cc937c69`|`40c497f8ebbab68e067993ea2b3ff24e53836ff714b1d5c8bd13d5e87778eaf8`|`outputs/metrics/p1_v1_sprint_heldout_sprint_t2_v1_s2.json`|

## Arm별 사실 집계

|arm|grid runs|complete|observed|pending|heldout results|
|---|--:|--:|--:|--:|--:|
|v1|7|3|1|3|3|
|matched|5|0|0|5|0|
|random|5|0|0|5|0|

`pending`은 해당 그리드 태그의 run/selection/heldout/claim/log 아티팩트가 아직 없는 행이다. `observed`는 일부 아티팩트만 있어 완료 판정을 하지 않은 행이다.
