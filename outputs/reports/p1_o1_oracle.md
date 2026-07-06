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
| t1a_straighten | ON | 0.190 | -0.994 | 0.2281 | 0.2281 | 0.1319 | 0 |
| t1b_single_bend | ON | 0.000 | -1.888 | 0.2505 | 0.2505 | 0.1496 | 0 |
| t1c_endpoint_reposition | ON | 0.000 | -0.748 | 0.2716 | 0.2716 | 0.1696 | 0 |
| t1a_straighten | OFF | 0.170 | -1.740 | 0.2413 | 0.2413 | 0.1479 | 60 |
| t1b_single_bend | OFF | 0.010 | -1.638 | 0.2259 | 0.2259 | 0.1660 | 6 |
| t1c_endpoint_reposition | OFF | 0.000 | -0.259 | 0.2226 | 0.2226 | 0.1458 | 0 |

## Per-template stats

### t1a_straighten

| template | ON success | ON return | ON d_at_done | ON min D | OFF success | OFF return | OFF d_at_done | OFF min D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random_smooth | 0.120 | -1.793 | 0.2621 | 0.1414 | 0.160 | -2.088 | 0.2552 | 0.1521 |
| s_curve | 0.000 | -1.616 | 0.2869 | 0.1668 | 0.000 | -2.031 | 0.2790 | 0.1733 |
| straight | 0.640 | 1.840 | 0.0939 | 0.0582 | 0.520 | 0.244 | 0.1536 | 0.0851 |
| u_bend | 0.000 | -2.410 | 0.2695 | 0.1613 | 0.000 | -3.084 | 0.2773 | 0.1812 |

### t1b_single_bend

| template | ON success | ON return | ON d_at_done | ON min D | OFF success | OFF return | OFF d_at_done | OFF min D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random_smooth | 0.000 | -1.644 | 0.2392 | 0.1384 | 0.000 | -1.986 | 0.2333 | 0.1655 |
| s_curve | 0.000 | -1.823 | 0.2804 | 0.1522 | 0.000 | -1.419 | 0.2262 | 0.1703 |
| straight | 0.000 | -1.611 | 0.2304 | 0.1543 | 0.000 | -1.806 | 0.2201 | 0.1808 |
| u_bend | 0.000 | -2.476 | 0.2519 | 0.1533 | 0.040 | -1.340 | 0.2242 | 0.1473 |

### t1c_endpoint_reposition

| template | ON success | ON return | ON d_at_done | ON min D | OFF success | OFF return | OFF d_at_done | OFF min D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random_smooth | 0.000 | -0.649 | 0.2537 | 0.1682 | 0.000 | -0.393 | 0.2281 | 0.1427 |
| s_curve | 0.000 | -0.334 | 0.2283 | 0.1511 | 0.000 | -0.051 | 0.2001 | 0.1278 |
| straight | 0.000 | -0.786 | 0.2812 | 0.1869 | 0.000 | -0.327 | 0.2354 | 0.1499 |
| u_bend | 0.000 | -1.223 | 0.3230 | 0.1721 | 0.000 | -0.263 | 0.2270 | 0.1630 |

