# DGCC P1 M3R m3r_t1a_s2 Reproducibility Forensics (read-only, 2026-07-14)

Full report: agent://7-T1aS2Forensics (architect lane, openai-codex/gpt-5.6-sol). Key content preserved here for reboot-durability of the session artifact. NO mid-run intervention performed; live lane untouched. Recommendation withheld until active run completes (user constraint).

## PROVEN
1. Launch commits differ (interrupted: 3ac95fe · current: 6a997f6) but tracked executable content is IDENTICAL — compare 3ac95fe..HEAD touches only STEP_LOG.md + archive manifest. run-JSON git_commit fields are flush-time, not launch provenance.
2. **Reproducibility contract broken before training starts: `scripts/p1_train.py:124-134` constructs TD3Agent (random init of Encoder/TwinCritic/Actor + targets) BEFORE `torch.manual_seed(self.seed)`** → fresh processes start from different weights for the same CLI seed. (priority-1 finding, confidence 0.99)
3. No deterministic CUDA policy anywhere (no use_deterministic_algorithms / cudnn.deterministic / CUBLAS_WORKSPACE_CONFIG).
4. First logged update already diverges @5,120 tr (critic_loss 1.662 vs 0.593; q1_mean −0.379 vs +0.040) — before any rebuild, incident counters still equal.
5. Incident/rebuild histories differ: interrupted rebuilds @20,992/@47,360; current first rebuild @51,712 (after the compared 50k eval).
6. **Evaluator episode seeds depend on rebuild history** (`scripts/p1_train.py:525-554` episode_index_start ← self.episode_index) → eval indices 90,002/90,004 (interrupted) vs 90,001/90,002 (current): 41/43% vs 0/0% is partly evaluation-set-confounded, not purely policy. (priority-2 finding, confidence 0.99)
7. Contention differs: interrupted early ~1.25-1.9 tr/s contended (collect 199s, update 15s); current 2.7-5.6 tr/s solo.
8. Eval incident/wall differences: interrupted 25k/50k nan 0/0 mag 0/0 wall 387/686s; current nan 1/9 mag 21/19 wall 2115/2188s.

## RULED OUT
- Tracked scripts/src/configs/uv.lock changes between launches (API compare: STEP_LOG + manifest only).
- Config/hyper/CLI drift: driver start lines byte-identical except timestamp; run-JSON config echoes identical (seed 2, n_envs 256, budget 1e5, TD3 dict, reward, v_max 1498.0, rebuild limits 8/10).
- Kernel change (6.8.0-124-generic both boots).
- Direct wall-clock input to learning logic (timers report-only).
- Eval template-count imbalance (25×4 both attempts; curve seeds differ though).

## HYPOTHESES (ranked)
1. Different initial network weights (late torch seed) — highest; unprovable byte-level post hoc (initial weights not archived) but corroborated by first-update stats.
2. GPU/Genesis numerical nondeterminism amplified by outcome-driven auto-reset/reseed/rebuild counters — high.
3. Contention-sensitive kernel scheduling as simulator-divergence trigger — medium (no kernel traces survive).
4. Untracked external/DLO-Lab (.gitignore'd, commit-pinned c5026a9 by convention but unverifiable post hoc) or .venv drift — low-medium; note uv.lock resolves torch 2.12.1 while env has manually-installed 2.10.0+cu128 → lockfile is not a complete runtime fingerprint.
5. Inherited OMP/MKL values / driver-stack change across boots — low/unprovable (current: NVRM 590.48.01, CUDA 12.8, torch 2.10.0+cu128, genesis-world 1.0.0, quadrants 0.8.0; no historical banners logged).

## Verdict
Divergence is NOT from tracked code/config/hyper/seed/budget/kernel change. The driver's reproducibility design does not seed model initialization and does not enforce deterministic CUDA; simulator incidents, auto-resets, discards, and rebuilds amplify the initial difference, and the evaluator's rebuild-dependent indexing additionally confounds eval-to-eval comparison. Neither attempt is invalidated as an individual stochastic run; treating them as a controlled exact rerun is invalid. Prereg never claimed bitwise reproducibility ("env 비결정성" divergence is documented precedent: t1a_s1 rerun 2026-07-11), so the CURRENT run remains contract-valid; the two PROVEN code findings (seed ordering, eval indexing) are candidate gate-agenda items — any fix is post-gate, never mid-batch.

## Recommendation
WITHHELD until the active m3r_t1a_s2 run completes (user constraint). Draft direction to finalize post-run: report both findings at the S6 reconvene gate as evidence items (affects interpretation of "same-seed rerun" language and cross-attempt eval comparisons); propose fix approval for P2-prep or M4-prep window (seed-before-construct + eval-index decoupling + optional initial-weights hash logging), never mid-M3R.
