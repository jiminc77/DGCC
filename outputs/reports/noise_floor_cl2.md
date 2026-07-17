# Execution-noise floor — correspondence_l2 (empirical)

> HUMAN directive (s1r->s2 window): repeated same-(p,u) executions without restore, per template; the prior 0.0315 floor was Chamfer-derived and is DISCARDED. This measurement is the official spike-2 judgment floor.

| template | n pairs | median | p90 | max | margin vs eps=0.05 (eps - p90) |
|---|---:|---:|---:|---:|---:|
| straight | 18 | 0.0000 | 0.0000 | 0.0000 | +0.0500 |
| u_bend | 18 | 0.0000 | 0.0000 | 0.0000 | +0.0500 |
| s_curve | 18 | 0.0000 | 0.0000 | 0.0000 | +0.0500 |
| random_smooth | 18 | 0.0000 | 0.0000 | 0.0000 | +0.0500 |

- Wall: 650s · REPS=4 · PU_SETS=3 · settle max 10000
- Interpretation: negative margin means execution noise alone can exceed the success threshold for that template.
