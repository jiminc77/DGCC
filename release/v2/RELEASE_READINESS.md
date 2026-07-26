# V2 Round-8 Release Readiness

## Status

**READY FOR ORCHESTRATOR LAUNCH AUTHORIZATION ON THE LOCKED 15-RUN SCHEDULE.** Runtime algorithms at source commit `12befdac5cc9d2af448373de81fcce9d86768701` are externally approved with operational fixes. BGT is formally not admitted to this tournament, and all 15 R1/R2/no-training preflight cells pass.

No GPU, training, evaluation, heldout/probe content read, or live-tree mutation was performed while preparing this package. Production training has not started.

## Round-8 blocking procedures

| # | Procedure | Status | Evidence |
|---|---|---|---|
| 1 | Tournament run count | **PASS — 15 LOCKED** | `bgt_not_admitted.json` locks exactly five arms × seeds `{0,1,2}`, no redistribution, seed-block interleaving, and no 14-run path. Artifact SHA-256: `9d78cb14ca900fb27b6996b21f3f1927d3dadf80e168886738bf79dea06d2701`; embedded disposition SHA-256: `2f7000456907f7b5bfb498a7cf91be6f4591303dc9c43e2cf87fec582392687e`. |
| 2 | BGT disposition | **PASS — NOT ADMITTED** | The R5 input provenance was never preregistered, so retroactive asset selection was rejected. No rank calibration or GPU latency benchmark was run. BGT code remains inactive and retained for a separately preregistered post-winner extension. |
| 3 | Final code-manifest pin | **PASS** | `final_code_manifest.json`: SHA-256 `68f17972930569c8a8bb87efae67230df531c4c0982e877a11a44acdbba2b440`; 93 files; closure SHA-256 `72148182eaf96c95c952a173810e2e72e6b1f699a94ca3c72eec087eaadcc51c`. Final 15-cell governance SHA-256: `827bfa490c87bae1cd97f282461911e1fcc27bcb8b79e59b948cc71037e6dc06`; protocol-validation base governance SHA-256: `72a1a904e0502d63af6fc22071d9063ae595572ee1376b4bfc641b1ac02f922d`. |
| 4 | G3 R1–R4 | **PASS WITH DECLARED APPLICATION-LEVEL LIMITATION** | Every final cell has an immutable R1 allowlist and zero-access R2 audit receipt. The live-tree existence receipt (`c51ad72…`) honestly records two legacy manifests and remains preserved out of scope. The authoritative V2 firewall receipt is sparse-sandbox R3/R4 SHA-256 `58f365b45d9f0d62d2e165f8bfd3e34414b9f16142a5faa2f9809e7713648c71`; the final matrix binds it and a matching fresh existence receipt SHA-256 `253d620fd50ed2c7568ef02f89d5a003b4923c76d9809175ba690a0da560a787` under distinct roles. Both record R3/R4 PASS, no protected content opened, same UID, and no claim of OS isolation. |
| 5 | Full CPU report bytes | **PASS** | `V2DEV_CPU_TEST_REPORT.json` is published byte-for-byte at SHA-256 `b04829a42de204359918cf10a36b4034c46db0f48182259b35e3ed9ff7babc07`; it records `442 passed, 7 deselected`. |
| 6 | All-cell no-training preflight | **PASS — 15/15** | `preflight_15_not_admitted/preflight_matrix.json` SHA-256 `2c7188294452e6bc298f5ec230cff56c2819fced016d44f81767fc36aa2ddac0`; all 15 cells pass protocol validation, contain nonempty R1/R2 pins, and carry a real-launcher dry-run receipt that runs the launcher on production argv to a constructed agent. 15/15 resolve all eleven allowlisted roles at published paths -- including `t2_split` and `runtime_environment` -- construct the correct agent class (`TD3Agent` for BB-D2, `SprintTD3Agent` for V1-D2, `SelectionWeightedTD3Agent` for DMM/D1M/D11), and complete `build_scene` (Genesis init, `DLOLabEnv` at 1024 envs, first reset, grasp hooks, batched runner, first episodes) on the GPU with zero transitions. The environment is pinned prospectively in `v2_runtime_environment.json` (torch `2.10.0+cu128`, genesis-world `c5026a94`, lockfile digest `a45115e7…`, 215 packages) and each cell allowlists it under role `runtime_environment`. The pinned runtime is `12befdac` plus the single B4 seam repair recorded in `preflight_matrix.runtime_patches`. Full 128-file index SHA-256 `fd46fb79a53c09a0be750efb814164419af4d67e6cea52107c6e576736fd05d3`, tree SHA-256 `e910e87966e530d9fad28bfb025c14cd9e9476524b456819a114460bd5716a1c`. The registry smoke records PREPARING → INITIALIZED → TERMINAL with verified terminal-anchor SHA-256 `584fa619072b26fb8e9efd281e840a450ec508669094872826c30ad91e723abe`. |

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
