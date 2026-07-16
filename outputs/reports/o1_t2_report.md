# P1-O1 Oracle Feasibility Reference

이 리포트는 **feasibility reference**이다. Attainability upper bound가 아니다.

## P-c interpretation rule

> oracle 성공 → 과제 달성 가능 확정 · oracle ≫ policy → 학습 문제 확정 · oracle ≈ 0 → 판정 불능 (불가능 증명 아님)

## Oracle policy symbol choices

- Flip convention: `dgcc.models.networks.goal_residual_flips`, which routes through `dgcc.goals.distance.canonical_shape_flip`.
- Residual: `res = g_aligned - x` using the encoder's index-wise goal correspondence.
- Action: `p = argmax_i ||res_i||`; `delta = direction(res_p) * min(||res_p||, 0.15)`; `lift = high` iff `res_p[z] > 0`.
- Delta assertion: norm and per-axis bounds are checked before execution, so the environment clamp should be a no-op for the oracle command.

## evaluate_episodes hook assumption

The oracle uses the same `evaluate_episodes` hook as `p1_train.py::deterministic_eval`: T1 `goal_fn`, `seed + 500`, `rng seed + 501`, and an episode-index base in the 90,000 eval namespace.

## Side-by-side feasibility reference

| task | grasp realism | success | return | final D | d_at_done | min D | NaN incidents |
|---|---|---:|---:|---:|---:|---:|---:|
| t2 | ON | 0.010 | -0.725 | 0.2324 | 0.2324 | 0.1293 | 1 |
| t2 | OFF | 0.000 | -0.854 | 0.2046 | 0.2046 | 0.1234 | 0 |

## Per-template stats

### t2

| template | ON success | ON return | ON d_at_done | ON min D | OFF success | OFF return | OFF d_at_done | OFF min D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random_smooth | 0.040 | -0.946 | 0.2258 | 0.1249 | 0.000 | -1.056 | 0.2160 | 0.1244 |
| s_curve | 0.000 | -0.679 | 0.1973 | 0.1220 | 0.000 | -0.543 | 0.1982 | 0.1220 |
| straight | 0.000 | -0.315 | 0.2333 | 0.1341 | 0.000 | -1.072 | 0.2104 | 0.1347 |
| u_bend | 0.000 | -0.960 | 0.2732 | 0.1362 | 0.000 | -0.745 | 0.1938 | 0.1125 |

| family | ON success | ON d_at_done | ON min D | OFF success | OFF d_at_done | OFF min D |
|---|---:|---:|---:|---:|---:|---:|
| l | 0.000 | 0.2824 | 0.1267 | 0.000 | 0.1927 | 0.1211 |
| s | 0.000 | 0.2323 | 0.1328 | 0.000 | 0.1949 | 0.1295 |
| smooth_random | 0.000 | 0.2598 | 0.1368 | 0.000 | 0.2388 | 0.1298 |
| u | 0.062 | 0.2010 | 0.1289 | 0.000 | 0.2097 | 0.1208 |
| zigzag | 0.000 | 0.1924 | 0.1182 | 0.000 | 0.1847 | 0.1116 |

