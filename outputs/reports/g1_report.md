# G1 stiffness-validity pilot report

- created_at: 2026-07-03T05:50:47Z
- config: `configs/gate_g1.yaml`
- stdout log: `outputs/reports/gate_g1_stdout.log`
- wall_time_s: 438.6
- stats_recomputed_at: 2026-07-03T06:20:29Z
- stats_recomputed_at_commit: 6da316a232c2f602c82ddeba630d5c535b9795b1
- batching: per-env DLO-Lab parameter setters with envs_idx support; mixed 3-condition batches grouped by sequence length
- grasp realism: off for this controlled measurement; p/delta/lift fixture is fixed.

## Fixture
- sequences: 20 fixed sequences (straight=5, u_bend=5, s_curve=5, random_smooth=5)
- init seeds per sequence: [0, 1, 2]
- stiffness multipliers: [0.5, 1.0, 2.0]
- friction multipliers: [0.5, 1.0, 2.0] (G1-subordinate reference)
- Bootstrap CIs: i.i.d. resamples distance records; sequence-cluster resamples the 20 sequence ids with replacement and includes all distances for drawn sequences (5000 reps each; rng seeds cli_seed+60000 and cli_seed+70000).
- Negative d encodes between-condition distance below the within-condition noise floor in pooled-standard-deviation units.

## Stiffness block

| pair | between mean | within-floor mean | d | i.i.d. bootstrap CI | sequence-cluster bootstrap CI | note |
| --- | ---: | ---: | ---: | --- | --- | --- |
| 0.5_vs_1.0 | 0.0465329 | 0.0443103 | 0.0611288 | [-0.264927, 0.369707] | [-0.292368, 0.373292] | n_between=60; n_within=120 |
| 1.0_vs_2.0 | 0.0522068 | 0.0540651 | -0.0335125 | [-0.31731, 0.270086] | [-0.273123, 0.168205] | n_between=60; n_within=120 |
| 0.5_vs_2.0 | 0.066673 | 0.0529987 | 0.235856 | [-0.0568308, 0.557556] | [0.0409446, 0.456666] | n_between=60; n_within=120 |

Within-condition floors:

| condition | n | mean | std | median |
| --- | ---: | ---: | ---: | ---: |
| 0.5 | 60 | 0.0432438 | 0.0338795 | 0.0347589 |
| 1.0 | 60 | 0.0453767 | 0.0348623 | 0.0345731 |
| 2.0 | 60 | 0.0627535 | 0.0744187 | 0.0286874 |

## Friction reference block

| pair | between mean | within-floor mean | d | i.i.d. bootstrap CI | sequence-cluster bootstrap CI | note |
| --- | ---: | ---: | ---: | --- | --- | --- |
| 0.5_vs_1.0 | 0.0262704 | 0.0409265 | -0.460699 | [-0.742106, -0.180638] | [-0.844434, -0.196798] | n_between=60; n_within=120 |
| 1.0_vs_2.0 | 0.0257735 | 0.0417548 | -0.51563 | [-0.757625, -0.267773] | [-0.847487, -0.255221] | n_between=60; n_within=120 |
| 0.5_vs_2.0 | 0.0300093 | 0.04482 | -0.412076 | [-0.647576, -0.161474] | [-0.701019, -0.20229] | n_between=60; n_within=120 |

Within-condition floors:

| condition | n | mean | std | median |
| --- | ---: | ---: | ---: | ---: |
| 0.5 | 60 | 0.0439918 | 0.0404981 | 0.030552 |
| 1.0 | 60 | 0.0378613 | 0.0272256 | 0.0290654 |
| 2.0 | 60 | 0.0456482 | 0.0405404 | 0.0256227 |

## Plots

- stiffness distributions: `outputs/plots/g1_stiffness_distributions.png`
- friction distributions: `outputs/plots/g1_friction_distributions.png`

## This-run convergence

- reset_converged_rate: 1.0 across 6 batches.
- per-primitive converged_rate: range=[0.583333, 0.972222], mean=0.791667 across 18 primitive summaries.
- Unsettled final shapes add settle-noise to both between-condition and within-condition distance distributions.

## Physics-quality context

- dm_stats: `outputs/metrics/dm_stats.json`
- dm_stats rates: grasp_success=0.946598, settle_converged=0.536986, success_and_converged=0.483584
- dm_stats note: For gate/human review, prefer rates.success_and_converged (0.484) over the headline settle_converged rate: failed grasps are no-op transitions whose converged flag reflects the untouched rope, so the aggregate conflates populations. Settle non-convergence at the immutable 1e-3/5000 budget affects 2341 successful transitions; they are recorded honestly (settle_steps == max_steps) and excluded from the normalizer fit. Flagged for the M5/M6 human gates (plan A1).

