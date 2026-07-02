# STEP_LOG

- 2026-07-02T17:00:31+00:00 — M0 start.
- 2026-07-02T17:02:00+00:00 — Created P0 §4 workspace skeleton, pyproject, stubs, base interface, transition schema, and tests.
- 2026-07-02T17:02:00+00:00 — Created uv Python 3.12 virtualenv and installed editable package with M0 dependencies.
- 2026-07-02T17:02:00+00:00 — Verified `uv run pytest tests/` and base import one-liner.
- 2026-07-02T17:02:00+00:00 — gh CLI 없음 — issue #1 수동 처리 필요.
- 2026-07-02T17:02:00+00:00 — M0 complete.
- 2026-07-02T17:15:12+00:00 — M0 review-gate fixes: writer.py stub callables (QA C5), resample.py M1-minimal/M4-finalize docstring note; tests 3 passed + red-team 34 passed.
- 2026-07-02T18:13:46+00:00 — M1 start: two-sim bring-up (MuJoCo-first order per approved plan).
- 2026-07-02T18:13:46+00:00 — M1 MuJoCo lane: adapter+smoke 7/7 PASS (gravity+ground-plane scene; MuJoCo 3.10 name-scheme handled via enumeration; viscosity hack removed after review).
- 2026-07-02T18:13:46+00:00 — M1 DLO-Lab lane: install SUCCESS under 2h timebox (torch 2.10.0+cu128, genesis-world 1.0.0, pins numpy<2.5/fsspec<=2026.2.0/packaging<26.0; assets HTTP 401 — not needed); smoke 8/8 PASS. M1 failure-halt rule NOT triggered.
- 2026-07-02T18:13:46+00:00 — M1 comparison: 5 seq x 3 seeds x 2 sims; MuJoCo settle 0/30 converged @5000 steps vs DLO-Lab 30/30; report outputs/reports/sim_comparison.md.
- 2026-07-02T18:13:47+00:00 — gh CLI 없음 — issue #2 수동 처리 필요.
- 2026-07-02T18:13:47+00:00 — M1 complete.
- 2026-07-02T18:27:43+00:00 — M1 gate fixes: [dlo-lab] extra + commented compat pins in pyproject, README 재현 section, settle metric-definition caveat in sim_comparison.md (architect MEDIUM advisories).
