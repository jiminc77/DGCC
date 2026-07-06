# P1 Throughput Probe (M0)

Generated: 2026-07-06T18:07:38Z · git ecfa1feea37485ac63d66f6953d4c98ca14f951e · seed 0

Scope: single-process n_envs scaling exactly as specified (P1.md @M0). Scheduling is S1 — one training run at a time; concurrent-run probing was explicitly rejected at plan reconciliation (R2), so no concurrency data exists or is claimed.

Settle budget on every settle-bearing call: vel_threshold=0.001, max_steps=10000 (global rule 7).

## Measurements

| n_envs | transitions/s | s/round | build s | grasp succ | settle conv | VRAM used GiB | NaN inc | rebuilds |
|---|---|---|---|---|---|---|---|---|
| 512 | 13.61 | 37.6 | 15.2 | 0.951 | 0.926 | 4.3/95 | 0 | 0 |
| 1024 | 21.87 | 46.8 | 7.4 | 0.950 | 0.926 | 4.3/95 | 0 | 0 |

## Recommendation: n_envs = 1024 (S1)

Chosen by maximum measured transitions/s (21.87 tr/s). P0 reference: 3.61 tr/s at n_envs=64 with the 5000-step settle budget (P1 uses the 10000-step budget everywhere, so values are not directly comparable).

## Projected run durations at the recommended n_envs (S1, serial)

| Item | Transitions | Hours |
|---|---|---|
| M2 smoke (1 run) | 50,000 | 0.6 |
| T1 run (each) | 100,000 | 1.3 |
| T2 run (each) | 300,000 | 3.8 |
| M3 total (9 runs) | 900,000 | 11.4 |
| M4 total (3 runs) | 900,000 | 11.4 |
| **P1 training total** | 1,850,000 | 23.5 |

Eval episodes (every 25k transitions) and checkpointing are additional overhead on top of these collection-only projections. Accounting caveat: transitions/s counts every executed primitive (n_envs per round) including post-done-env and NaN-discarded rounds; discard/rebuild counts are reported in the measurement table so usable-data throughput can be derived.
