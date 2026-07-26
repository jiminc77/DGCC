# G6b V1-phase stability recount (interim)

## V1-phase A-4 stability recount (interim, campaign held)

학습 로그 원본 재집계(scripts/sprint_stability_recount.py, 규약: run JSON reported는 최종 rebuild-reset 이후 창만 반영 — 재집계는 전 구간 하한). settle-pocket 관측: v1_s0 2.75h·v1_s1 2.3h(양건 자가 회복·guard 미발동), v1_s2–s4 pocket-free.

| run | reported nan | reported mag | recounted nan (A-4 하한) | recounted mag | rebuilds |
|---|---:|---:|---:|---:|---:|
| v1_s0 | 0 | 0 | 23 | 23 | 4 |
| v1_s1 | 0 | 0 | 86 | 59 | 3 |
| v1_s2 | 57 | 70 | 107 | 100 | 2 |
| v1_s3 | 0 | 0 | 83 | 94 | 1 |
| v1_s4 | 18 | 16 | 59 | 55 | 1 |

운영 통계이며 성능 endpoint가 아니다. 잔여 10런(matched/random)은 V2 토너먼트 후 재개 시 동일 표에 병합한다.
