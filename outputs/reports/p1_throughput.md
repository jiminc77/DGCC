# P1 Throughput Probe (M0)

Generated: 2026-07-03T11:38:28Z · git 236baae33d5213ae3991d83ff4fa5620d293a1c6 · seed 0

Scope: single-process n_envs scaling exactly as specified (P1.md @M0). Scheduling is S1 — one training run at a time; concurrent-run probing was explicitly rejected at plan reconciliation (R2), so no concurrency data exists or is claimed.

Settle budget on every settle-bearing call: vel_threshold=0.001, max_steps=10000 (global rule 7).

## Measurements

| n_envs | transitions/s | s/round | build s | grasp succ | settle conv | VRAM used GiB | NaN inc | rebuilds |
|---|---|---|---|---|---|---|---|---|
| 64 | 1.90 | 33.7 | 7.8 | 0.964 | 0.906 | 4.9/95 | 0 | 0 |
| 128 | 3.55 | 36.1 | 4.9 | 0.954 | 0.958 | 5.0/95 | 0 | 1 |
| 256 | 6.87 | 37.3 | 7.5 | 0.936 | 0.940 | 5.0/95 | 0 | 0 |

## Recommendation: n_envs = 256 (S1)

Chosen by maximum measured transitions/s (6.87 tr/s). P0 reference: 3.61 tr/s at n_envs=64 with the 5000-step settle budget (P1 uses the 10000-step budget everywhere, so values are not directly comparable).

## Projected run durations at the recommended n_envs (S1, serial)

| Item | Transitions | Hours |
|---|---|---|
| M2 smoke (1 run) | 50,000 | 2.0 |
| T1 run (each) | 100,000 | 4.0 |
| T2 run (each) | 300,000 | 12.1 |
| M3 total (9 runs) | 900,000 | 36.4 |
| M4 total (3 runs) | 900,000 | 36.4 |
| **P1 training total** | 1,850,000 | 74.8 |

Eval episodes (every 25k transitions) and checkpointing are additional overhead on top of these collection-only projections.
