# P1-M4 Pre-Start Smoke Gate Report

> Verdict basis: gate-m3r-reconvene-2-20260713 choice B — smoke required before ANY main lane; per-rung thresholds pre-registered (2048 ≥ 31.04 tr/s · 1024 ≥ 17.52 · 256 stability-only + gate re-report). Data discarded (archived out of live namespace with sha256 manifests). Consensus plan pending-approval sha256 78292468….

## Rung 1 — n_envs=2048 (smoke_m4_t2_s100, seed 100, 5e4 discard)

- Drill: **full supervisor path exercised and verified** — launch → readiness-verified PGID handoff (atomic dual-file publication) → flush-confirmed STOP @26,624 (fired only after the eval-1 run-JSON flush) → all-member-T verify → durable quiescent marker (+archive mirror) → 60 s no-advance hold (log size + run-JSON sha unchanged) → CONT → completion 51,200, exit 0, wall 1.34 h.
- Stability: TrainingNaNError 0 · rebuilds 0 · storm deadlock 0 · evals 2 (success 1%→0%, return −0.60/−0.86 vs t2_val random −3.39; gaps bounded; mean_d_shape_at_done 0.218 — D_shape channel live).
- **Throughput: FAIL.** Per-round collection rates (P1_LOG_EVERY_ROUND dense roundlog, n=25 valid rounds): median **23.1 tr/s < floor 31.04**; warm-up rounds 45.2 tr/s degrade to ~23.1 post-warmup. The 38.8 tr/s probe was pure collection (no episode protocol / auto-reset settle costs). Additional finding: **13/38 rounds covenant-discarded (34%)** — batch-discard probability scales with n_envs (vs ~10% at 256); designated gate evidence.
- Effective throughput (collect+update+eval): 10.6 tr/s → projected 7.9 h per 3e5 run (reference only; the gate metric is collection).
- Disposition: pre-registered demotion → 1024 re-smoke. Archive: `outputs/archive/m4smoke/20260715T0957Z/` (11 files, MANIFEST.sha256).

## Rung 2 — n_envs=1024 (smoke1024_m4_t2_s101, seed 101, 5e4 discard)

- Drill: same supervisor path, STOP/hold/CONT verified again; completion 50,176, exit 0, wall 1.48 h.
- Stability: **PASS** — TrainingNaNError 0 · rebuilds 1/8 (ordinary recovery) · storm deadlock 0 · evals 2 (terminal: success 0%, return −2.18 vs random −3.39; mean_d_shape_at_done 0.269).
- **Throughput: PASS** — per-round collection median **25.7 tr/s ≥ floor 17.52** (n=49 valid rounds; min 13.0, max 25.9); discard 20/69 rounds (29%).
- **Effective throughput 9.4 tr/s → projected wall ≈ 8.8 h per 3e5 run; 3 seeds serial ≈ 27 h.**
- Archive: `outputs/archive/m4smoke/20260715T1127Z/` (11 files, MANIFEST.sha256).

## Decisions in force for P5

1. **Main regime: n_envs=1024, warmup_transitions=10,240** (ladder result; `configs/p1_t2.yaml` updated with judgment-basis comment).
2. **Lane composition: SERIAL** (pre-registered O1 rule — projected 3-run wall 27 h < 60 h threshold; no parallel spot-check needed).
3. HUMAN checklist gate (#13 comment 4978169975 directive 2) satisfied pre-P5: held-out evaluator + selection tool exist; `tests/test_t2_heldout_eval.py` 12 passed at gate time.
4. Main lanes launch with `env -u P1_LOG_EVERY_ROUND` (zero-roundlog post-run assertion applies); HER halfway STOP at 1.5e5 per seed via the drill-validated supervisor.

## F-a/F-b production evidence (first-ever)

- `initial_weights_sha256` banner + run-JSON persistence live (rung 1: 74a469c8…).
- `eval_episode_index_start` recorded per eval row (90_001/90_002) — rebuild-independent indexing live.
