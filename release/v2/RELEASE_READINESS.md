# V2 Round-8 Release Readiness

## Status

**READY FOR ORCHESTRATOR LAUNCH AUTHORIZATION ON THE LOCKED 15-RUN SCHEDULE.** Runtime algorithms at source commit `12befdac5cc9d2af448373de81fcce9d86768701` are externally approved with operational fixes. BGT is formally not admitted to this tournament, and all 15 R1/R2/no-training preflight cells pass.

No GPU, training, evaluation, heldout/probe content read, or live-tree mutation was performed while preparing this package. Production training has not started.

## Round-8 blocking procedures

| # | Procedure | Status | Evidence |
|---|---|---|---|
| 1 | Tournament run count | **PASS — 15 LOCKED** | `bgt_not_admitted.json` locks exactly five arms × seeds `{0,1,2}`, no redistribution, seed-block interleaving, and no 14-run path. Artifact SHA-256: `7f59cdffdccd5c06f26e8358045bd4ae9a218422d623b93535e37ebb58131679`; embedded disposition SHA-256: `32ddb8a711ec07ff6550288faf2c7a2468a9c8c31dee7607d75d8583667f17a5`. |
| 2 | BGT disposition | **PASS — NOT ADMITTED** | The R5 input provenance was never preregistered, so retroactive asset selection was rejected. No rank calibration or GPU latency benchmark was run. BGT code remains inactive and retained for a separately preregistered post-winner extension. |
| 3 | Final code-manifest pin | **PASS** | `final_code_manifest.json`: SHA-256 `5bba47a6602717f9ed52daaa051da448d54543b42578b75db63b39574df814d9`; 93 files; closure SHA-256 `1823c83978469f249fa28f369cdec474212641d083bea737fe3cb0c7a3bdf635`. Final 15-cell governance SHA-256: `87053627c0a01f07b158f481b8308e19320b04d95ac9dbbf5cbb3aa2da817408`; protocol-validation base governance SHA-256: `932ab4c2c86b4ad8bf702b9ca58ae9e3f8950697fdea6456051c653d79c2e53b`. |
| 4 | G3 R1–R4 | **PASS WITH DECLARED APPLICATION-LEVEL LIMITATION** | Every final cell has an immutable R1 allowlist and zero-access R2 audit receipt. The live-tree existence receipt (`c51ad72…`) honestly records two legacy manifests and remains preserved out of scope. The authoritative V2 firewall receipt is sparse-sandbox R3/R4 SHA-256 `58f365b45d9f0d62d2e165f8bfd3e34414b9f16142a5faa2f9809e7713648c71`; the final matrix binds it and a matching fresh existence receipt SHA-256 `253d620fd50ed2c7568ef02f89d5a003b4923c76d9809175ba690a0da560a787` under distinct roles. Both record R3/R4 PASS, no protected content opened, same UID, and no claim of OS isolation. |
| 5 | Full CPU report bytes | **PASS** | `V2DEV_CPU_TEST_REPORT.json` is published byte-for-byte at SHA-256 `b04829a42de204359918cf10a36b4034c46db0f48182259b35e3ed9ff7babc07`; it records `442 passed, 7 deselected`. |
| 6 | All-cell no-training preflight | **PASS — 15/15** | `preflight_15_not_admitted/preflight_matrix.json` SHA-256 `b21a7526c28710dbd9105edff886485da4ffd962b5a1aeefe79eb3f3341dd6de`; all 15 cells pass protocol validation and contain nonempty R1/R2 pins. Full 98-file index SHA-256 `1f34a274e47548bc626bbd696a71c3670311f5d0384d0d6333f85d0716342930`, tree SHA-256 `9c3f06d04a9a7d469dde2a1f367adb1efe327dd9c0269bf041d5e97ec4edfeef`. The registry smoke records PREPARING → INITIALIZED → TERMINAL with verified terminal-anchor SHA-256 `a723af5a26596ced9ae48813785b76e927d6d2f78cee94762475a7ff26de39b7`. |

## Authoritative execution location

V2 production runs execute from the isolated v2-dev worktree `/home/simx2204/v2_research/impl/DGCC`; the live tree `/home/simx2204/Workspaces/DGCC` remains exclusively assigned to the G6b campaign and delayed AMD-5 original V1 s6/s7 runs. V2 arms therefore share one final runtime closure without contaminating the original-protocol closure.

**V2 production runs execute from the isolated v2-dev worktree; the live-tree R3 failure is retained as an honest record of the campaign tree and is out of scope for the V2 launch firewall, whose authoritative receipts are the sparse-sandbox R3/R4 (`58f365b4…`).**

The sparse sandbox is receipt evidence for the same pinned V2 closure and same actual UID. It is not a second code lineage and does not imply OS, mount-namespace, ACL, or kernel isolation.

## Locked schedule

BGT is not admitted to this tournament. The only authorized schedule is:

- `BB-D2` × seeds `{0,1,2}`
- `V1-D2` × seeds `{0,1,2}`
- `DMM` × seeds `{0,1,2}`
- `D1M` × seeds `{0,1,2}`
- `D11` × seeds `{0,1,2}`

This is exactly **15 runs**, seed-block interleaved. Amendment 2 prohibits reallocating BGT's three runs; no remaining candidate receives another seed. There is no 14-run path.

BGT remains a post-winner exploratory extension candidate. Its code is retained in an inactive state. Future admission requires a separately preregistered R5 protocol that pins assets, transitions, state selection, cardinality, and the rank-input producer before execution, followed by the synchronized GPU latency gate.

## Governance boundaries

- `v2_launch_code_manifest_sha256` is non-null in the final external governance artifact.
- `bgt_admitted_manifest_sha256` is permanently null for this tournament.
- `original_worktree_head_sha256` and `original_config_sha256` remain null by design. Original V1 s6/s7 launches therefore fail closed until the owner supplies independent pins.
- The attempt registry is **application-enforced append-only, non-reusing, hash-chained and externally anchored; it is not OS-level WORM**.
- R1/R2 are best-effort application controls, not OS isolation. R3/R4 are separate host evidence and cannot be omitted.

## Canonical guard and provenance copies

No guard artifact was deleted. Only the verified pair is authoritative for launch; all other copies remain outside the allowlist as provenance.

| Artifact | SHA-256 | Role |
|---|---|---|
| `V2_neff_guard_beta015363_verified.json` | `7a5b517ab108c8b0afa79d7e544e3f3d1eee40d5e56df6371d94a97bbcaedda5` | Canonical launch JSON |
| `V2_neff_guard_beta015363_verified.npz` | `72bc4218e829399d58e88b1604846eb51d9a1d7db42dfdab75b796acb09f7879` | Canonical arrays |
| `V2_neff_guard_beta015363.json` | `53d6dffd7f0b3333995d78bccf7104fc1bff75f5f038bf7dde4f0e9df2c135a8` | Intermediate |
| `V2_neff_guard_beta015363.npz` | `72bc4218e829399d58e88b1604846eb51d9a1d7db42dfdab75b796acb09f7879` | Intermediate duplicate arrays |
| `V2_neff_guard_beta015363_intermediate_keyfix.json` | `84138777cfaf238b87ef4deb8216280300f4ac8454ecceb9868be21f1ec7039a` | Intermediate key correction |
| `V2_neff_guard_beta015363_intermediate_duplicate.npz` | `72bc4218e829399d58e88b1604846eb51d9a1d7db42dfdab75b796acb09f7879` | Intermediate duplicate arrays |
| `V2_neff_guard.json` | `4ed37ac058475f2b0af46b27e7677e151d5cf332b93ce876660a5ff3b119155d` | Superseded beta `0.010` JSON |
| `V2_neff_guard.npz` | `8169166ce04e39dfc04da273e0e2977a6b8cca1f96887850c02f2bc090168137` | Superseded beta `0.010` arrays |

## Remaining launch authority

Only the orchestrator's explicit tournament-start authorization remains. The committed package does not start training, evaluation, or a GPU process.
