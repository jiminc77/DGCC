# P1 Throughput Probe (M0)

Generated: 2026-07-09T22:59:15Z · git 0556273c4a428edb9925ba2f9859ed95e052bd4c · seed 0

Scope: single-process n_envs scaling exactly as specified (P1.md @M0). Scheduling is S1 — one training run at a time; concurrent-run probing was explicitly rejected at plan reconciliation (R2), so no concurrency data exists or is claimed.

Settle budget on every settle-bearing call: vel_threshold=0.001, max_steps=10000 (global rule 7).

## Measurements

| n_envs | transitions/s | s/round | build s | grasp succ | settle conv | VRAM used GiB | NaN inc | rebuilds |
|---|---|---|---|---|---|---|---|---|
| 4096 | 63.57 | 64.4 | 17.0 | 0.949 | 0.931 | 4.5/95 | 1 | 0 |
| 8192 | 83.48 | 98.1 | 11.1 | 0.951 | 0.942 | 5.0/95 | 0 | 0 |
| 16384 | FAILED: RuntimeError: [cuda_driver.h:operator()@94] CUDA Error CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered while calling module_load_data_ex (cuModuleLoadDataEx) | | | | | | |

## Recommendation: n_envs = 8192 (S1)

Chosen by maximum measured transitions/s (83.48 tr/s). P0 reference: 3.61 tr/s at n_envs=64 with the 5000-step settle budget (P1 uses the 10000-step budget everywhere, so values are not directly comparable).

## Projected run durations at the recommended n_envs (S1, serial)

| Item | Transitions | Hours |
|---|---|---|
| M2 smoke (1 run) | 50,000 | 0.2 |
| T1 run (each) | 100,000 | 0.3 |
| T2 run (each) | 300,000 | 1.0 |
| M3 total (9 runs) | 900,000 | 3.0 |
| M4 total (3 runs) | 900,000 | 3.0 |
| **P1 training total** | 1,850,000 | 6.2 |

Eval episodes (every 25k transitions) and checkpointing are additional overhead on top of these collection-only projections. Accounting caveat: transitions/s counts every executed primitive (n_envs per round) including post-done-env and NaN-discarded rounds; discard/rebuild counts are reported in the measurement table so usable-data throughput can be derived.
