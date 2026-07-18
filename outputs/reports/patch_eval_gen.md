# Patch-only evaluation split generation report

- Generator: `dgcc.tasks.t2` unmodified; seed swap only.
- Seed: **20260722**. It differs from T2 master 20260703, sprint held-out 20260716, random-target 20260718, and matched-dimension projection 20260719.
- Goals: 100 · family distribution: `{"l": 22, "s": 23, "smooth_random": 18, "u": 17, "zigzag": 20}`.
- OOD length metadata: training config `P1_LENGTH_M=1.00 m` is a fixed singleton range `[1.00, 1.00]`; patch evaluation uses `[0.75, 1.25] m`, ratios `L_ood/L_train=[0.75, 1.25]`. Both values are outside the generator/domain configuration's training boundary, in shorter and longer directions. The donor mask scales x/y DCT modes 1--7 only; x0/y0/z0 and all z modes remain absolute.
- Parameter-level deduplication (purpose: `dedup_check`, key: family+params+anchor rounded to 12 decimals): t2_v1 all 650 = **0**, t2_sprint_heldout_v1 100 = **0**, M4 heldout 100 = **0**.
- File: `src/dgcc/tasks/splits/t2_patch_eval_v1.json`.
- Preflight stability measurement is deferred: after GPU approval, G10 takes precedence.
