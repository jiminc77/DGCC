# Goal stability preflight (report-only)

> Authority: DGCC#13 4985559491 directive 2 · generated 2026-07-19T11:08:00.286700+00:00 · wall 183s
> Framing (fixed): drift = elastic relaxation toward straight-rest (kappa_rest=0) equilibrium — measurement only, goal definitions unchanged.
> Metric: correspondence_l2(goal, settled, shape_only=True) + anchor delta; Chamfer forbidden. Converged-mask gated.
> Leakage guard: T2 val only; M4 held-out preflight completes after the M4 final held-out evaluation.

> Patch split loaded for purpose=preflight; no dedicated record_access path is available.

## t1b_single_bend — n=25 (converged 25, non-converged 0)
- drift_shape: median 0.0003 · p90 0.0007 · max 0.0009
- drift > eps(0.05): **0/25** (0%)
| template | n | drift median | drift max | >eps |
|---|---:|---:|---:|---:|
| t1b_single_bend(arc_angle=1.588411) | 1 | 0.0000 | 0.0000 | 0 |
| t1b_single_bend(arc_angle=1.589637) | 1 | 0.0000 | 0.0000 | 0 |
| t1b_single_bend(arc_angle=1.615034) | 1 | 0.0000 | 0.0000 | 0 |
| t1b_single_bend(arc_angle=1.985915) | 1 | 0.0000 | 0.0000 | 0 |
| t1b_single_bend(arc_angle=2.095629) | 1 | 0.0000 | 0.0000 | 0 |
| t1b_single_bend(arc_angle=2.121605) | 1 | 0.0000 | 0.0000 | 0 |
| t1b_single_bend(arc_angle=2.129524) | 1 | 0.0000 | 0.0000 | 0 |
| t1b_single_bend(arc_angle=2.773205) | 1 | 0.0002 | 0.0002 | 0 |
| t1b_single_bend(arc_angle=2.885834) | 1 | 0.0002 | 0.0002 | 0 |
| t1b_single_bend(arc_angle=2.990990) | 1 | 0.0003 | 0.0003 | 0 |
| t1b_single_bend(arc_angle=3.100971) | 1 | 0.0003 | 0.0003 | 0 |
| t1b_single_bend(arc_angle=3.119899) | 1 | 0.0003 | 0.0003 | 0 |
| t1b_single_bend(arc_angle=3.241187) | 1 | 0.0003 | 0.0003 | 0 |
| t1b_single_bend(arc_angle=3.277540) | 1 | 0.0003 | 0.0003 | 0 |
| t1b_single_bend(arc_angle=3.323900) | 1 | 0.0004 | 0.0004 | 0 |
| t1b_single_bend(arc_angle=3.429274) | 1 | 0.0004 | 0.0004 | 0 |
| t1b_single_bend(arc_angle=3.661787) | 1 | 0.0005 | 0.0005 | 0 |
| t1b_single_bend(arc_angle=4.042971) | 1 | 0.0006 | 0.0006 | 0 |
| t1b_single_bend(arc_angle=4.068994) | 1 | 0.0006 | 0.0006 | 0 |
| t1b_single_bend(arc_angle=4.083375) | 1 | 0.0006 | 0.0006 | 0 |
| t1b_single_bend(arc_angle=4.306803) | 1 | 0.0007 | 0.0007 | 0 |
| t1b_single_bend(arc_angle=4.326843) | 1 | 0.0007 | 0.0007 | 0 |
| t1b_single_bend(arc_angle=4.387440) | 1 | 0.0008 | 0.0008 | 0 |
| t1b_single_bend(arc_angle=4.600744) | 1 | 0.0008 | 0.0008 | 0 |
| t1b_single_bend(arc_angle=4.662722) | 1 | 0.0009 | 0.0009 | 0 |

## t1c_endpoint_reposition — n=25 (converged 25, non-converged 0)
- drift_shape: median 0.0000 · p90 0.0000 · max 0.0000
- drift > eps(0.05): **0/25** (0%)
| template | n | drift median | drift max | >eps |
|---|---:|---:|---:|---:|
| t1c_endpoint_reposition(d=0.201121) | 1 | 0.0000 | 0.0000 | 0 |
| t1c_endpoint_reposition(d=0.201199) | 1 | 0.0000 | 0.0000 | 0 |
| t1c_endpoint_reposition(d=0.202816) | 1 | 0.0000 | 0.0000 | 0 |
| t1c_endpoint_reposition(d=0.226427) | 1 | 0.0000 | 0.0000 | 0 |
| t1c_endpoint_reposition(d=0.233412) | 1 | 0.0000 | 0.0000 | 0 |
| t1c_endpoint_reposition(d=0.235066) | 1 | 0.0000 | 0.0000 | 0 |
| t1c_endpoint_reposition(d=0.235570) | 1 | 0.0000 | 0.0000 | 0 |
| t1c_endpoint_reposition(d=0.276548) | 1 | 0.0000 | 0.0000 | 0 |
| t1c_endpoint_reposition(d=0.283718) | 1 | 0.0000 | 0.0000 | 0 |
| t1c_endpoint_reposition(d=0.290412) | 1 | 0.0000 | 0.0000 | 0 |
| t1c_endpoint_reposition(d=0.297414) | 1 | 0.0000 | 0.0000 | 0 |
| t1c_endpoint_reposition(d=0.298619) | 1 | 0.0000 | 0.0000 | 0 |
| t1c_endpoint_reposition(d=0.306340) | 1 | 0.0000 | 0.0000 | 0 |
| t1c_endpoint_reposition(d=0.308655) | 1 | 0.0000 | 0.0000 | 0 |
| t1c_endpoint_reposition(d=0.311606) | 1 | 0.0000 | 0.0000 | 0 |
| t1c_endpoint_reposition(d=0.318314) | 1 | 0.0000 | 0.0000 | 0 |
| t1c_endpoint_reposition(d=0.333117) | 1 | 0.0000 | 0.0000 | 0 |
| t1c_endpoint_reposition(d=0.357384) | 1 | 0.0000 | 0.0000 | 0 |
| t1c_endpoint_reposition(d=0.359040) | 1 | 0.0000 | 0.0000 | 0 |
| t1c_endpoint_reposition(d=0.359956) | 1 | 0.0000 | 0.0000 | 0 |
| t1c_endpoint_reposition(d=0.374180) | 1 | 0.0000 | 0.0000 | 0 |
| t1c_endpoint_reposition(d=0.375455) | 1 | 0.0000 | 0.0000 | 0 |
| t1c_endpoint_reposition(d=0.379313) | 1 | 0.0000 | 0.0000 | 0 |
| t1c_endpoint_reposition(d=0.392892) | 1 | 0.0000 | 0.0000 | 0 |
| t1c_endpoint_reposition(d=0.396838) | 1 | 0.0000 | 0.0000 | 0 |

## t2_val — n=50 (converged 50, non-converged 0)
- drift_shape: median 0.0003 · p90 0.0186 · max 0.0732
- drift > eps(0.05): **4/50** (8%)
| template | n | drift median | drift max | >eps |
|---|---:|---:|---:|---:|
| l | 7 | 0.0086 | 0.0155 | 0 |
| s | 14 | 0.0000 | 0.0003 | 0 |
| smooth_random | 11 | 0.0000 | 0.0003 | 0 |
| u | 8 | 0.0006 | 0.0009 | 0 |
| zigzag | 10 | 0.0278 | 0.0732 | 4 |

## t2_patch_eval_v1 — n=300 (converged 300, non-converged 0)
- drift_shape: median 0.0009 · p90 0.0294 · max 0.1051
- drift > eps(0.05): **17/300** (6%)
| template | n | drift median | drift max | >eps |
|---|---:|---:|---:|---:|
| l | 66 | 0.0101 | 0.0231 | 0 |
| s | 69 | 0.0000 | 0.0016 | 0 |
| smooth_random | 54 | 0.0000 | 0.0013 | 0 |
| u | 51 | 0.0006 | 0.0019 | 0 |
| zigzag | 60 | 0.0295 | 0.1051 | 17 |

| rope length (m) | family | n | drift median | drift max | >eps | non-converged |
|---:|---|---:|---:|---:|---:|---:|
| 0.75 | l | 22 | 0.0133 | 0.0231 | 0 | 0 |
| 0.75 | s | 23 | 0.0002 | 0.0016 | 0 | 0 |
| 0.75 | smooth_random | 18 | 0.0002 | 0.0013 | 0 | 0 |
| 0.75 | u | 17 | 0.0013 | 0.0019 | 0 | 0 |
| 0.75 | zigzag | 20 | 0.0397 | 0.1051 | 9 | 0 |
| 1.00 | l | 22 | 0.0100 | 0.0174 | 0 | 0 |
| 1.00 | s | 23 | 0.0000 | 0.0004 | 0 | 0 |
| 1.00 | smooth_random | 18 | 0.0000 | 0.0005 | 0 | 0 |
| 1.00 | u | 17 | 0.0006 | 0.0009 | 0 | 0 |
| 1.00 | zigzag | 20 | 0.0224 | 0.0734 | 6 | 0 |
| 1.25 | l | 22 | 0.0080 | 0.0142 | 0 | 0 |
| 1.25 | s | 23 | 0.0000 | 0.0001 | 0 | 0 |
| 1.25 | smooth_random | 18 | 0.0000 | 0.0003 | 0 | 0 |
| 1.25 | u | 17 | 0.0002 | 0.0004 | 0 | 0 |
| 1.25 | zigzag | 20 | 0.0146 | 0.0554 | 2 | 0 |

