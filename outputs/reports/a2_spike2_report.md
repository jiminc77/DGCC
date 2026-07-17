# A-2 Spike-2 Report — state save/restore feasibility (CORRECTED: PASS)

> Authority: DGCC#13 comment 4994491311 (HUMAN, 2026-07-16T16:56:28Z) + in-chat floor directive.
> Executed in the s1r→s2 seed-boundary window (2026-07-17, no training interference).
> external/DLO-Lab untouched (c5026a9); all code wrapper-level, scratch-only (/tmp/dgcc_ops/a2_spike2.py, noise_floor_cl2.py).

## Verdict

**A-2 state save→restore is FEASIBLE — corrected to PASS.** Path B (`scene.save_checkpoint()` / `load_checkpoint()`) restores the full physics state to within execution noise; re-executing the same (p,u) reproduces the outcome at the noise floor.

The spike-1 FAIL (EVIDENCE 4994965388) is superseded: its premise ("twist getters absent") was factually wrong (`rod_entity.get_state()` returns theta/omega/twist/frames/kappa_rest — rod_entity.py:175), its restore path was a zero-twist REPLACEMENT (`place_rod_vertices_batch` → `_reinitialize_edge_state`, dlolab.py:882), and its 0.0315 floor was Chamfer-derived (discarded).

## Empirical noise floor (correspondence_l2) — `outputs/reports/noise_floor_cl2.md`

Same-seed pipeline (reset→settle→same (p,u)) repeated 4× per (template × 3 (p,u) sets), pairwise deviations (n=18/template):

| template | median | p90 | max | margin vs ε=0.05 |
|---|---:|---:|---:|---:|
| straight | 0.0 | 0.0 | 0.0 | +0.0500 |
| u_bend | 0.0 | 0.0 | 0.0 | +0.0500 |
| s_curve | 0.0 | 0.0 | 0.0 | +0.0500 |
| random_smooth | 0.0 | 4.1e-06 | 4.1e-06 | +0.0500 |

**The stack is effectively deterministic under identical reset/action sequences.** All templates hold the full +0.05 margin (random_smooth included — the "negative margin" question is answered: NO).

## Spike-2 results (4 shapes × 2 trials)

| metric | Path A (field-wise `_kernel_set_state`) | Path B (`scene.save/load_checkpoint`) | floor |
|---|---:|---:|---:|
| restore_dev median | 0.1065 | **9.6e-07** | — |
| outcome_dev median | 0.3240 | **6.4e-05** | ~0–4.1e-06 |

- **Path B: restore exact to ~1e-6; same-(p,u) outcome reproduced at ~6e-5 — at/near floor across all shapes. PASS.** (Checkpoint pickles ALL active solver arrays incl. the rigid gripper — scene.py:1536/1553.)
- Path A: wrapper-level field marshaling via `get_state` → `_kernel_set_state` leaves ~0.1 restore residual (candidate unrestored state: solver-internal contact/lambda arrays, field layout/transpose in scratch marshaling — Path B's full-array dump sidesteps all of it). Path A is NOT required given Path B works; noted for completeness.

## Comparison vs spike-1 (superseded)

| | spike-1 (2026-07-16) | spike-2 (2026-07-17) |
|---|---|---|
| restore path | positions + zero velocities + zero-twist edge re-init | full scene checkpoint (all solver arrays) |
| floor | 0.0315 hardcoded (Chamfer-derived) | measured CL2: ≤4.1e-06 |
| restore_dev | 0.15–0.29 | ~1e-06 |
| outcome_dev | 0.12–0.31 (0/12 pass) | ~6e-05 (at floor) |
| verdict | FAIL (premise wrong) | **PASS** |

## Implications (GNG-1 input; no action taken — decisions are human's)

- A-2 lookahead oracle is implementable with `scene.save_checkpoint`/`load_checkpoint` as the snapshot/restore mechanism (checkpoint I/O cost: pickle to disk per candidate — throughput profiling needed at design time).
- Plan B1 (finite-difference δm from logged transitions) remains prepared in parallel per the directive, independent of this PASS.
