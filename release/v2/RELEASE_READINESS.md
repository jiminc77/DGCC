# V2 Round-8 Release Readiness

## Status

**NOT READY FOR PRODUCTION LAUNCH.** Runtime algorithms at source commit `12befdac5cc9d2af448373de81fcce9d86768701` are externally approved with operational fixes. Final BGT disposition and final per-cell R1/R2/preflight evidence remain outstanding.

No GPU, training, evaluation, heldout/probe content read, or live-tree mutation was performed while preparing this package.

## Round-8 blocking procedures

| # | Procedure | Status | Evidence |
|---|---|---|---|
| 1 | Tournament run count | **PASS** | Governance and `EXEC_PLAN.md` permit only 18 runs when BGT is admitted or 15 when formally not-admitted, with no redistribution. Repository/dossier audit found no 14-run plan. |
| 2 | BGT disposition | **PENDING OWNER GPU WINDOW** | Governance admission pin remains null. No GPU measurement or not-admitted lock was fabricated before the promised explicit approval. |
| 3 | Final code-manifest pin | **PASS** | `final_code_manifest.json`: SHA-256 `5bba47a6602717f9ed52daaa051da448d54543b42578b75db63b39574df814d9`; 93 files; closure SHA-256 `1823c83978469f249fa28f369cdec474212641d083bea737fe3cb0c7a3bdf635`. Independent pending-BGT governance SHA-256: `932ab4c2c86b4ad8bf702b9ca58ae9e3f8950697fdea6456051c653d79c2e53b`. |
| 4 | G3 R1–R4 | **PARTIAL — R3/R4 PASS** | The live-tree existence receipt (`c51ad72…`) honestly records two legacy manifests present and remains preserved. It is out of scope for the V2 launch firewall because no V2 run executes from the campaign tree. The authoritative V2 firewall receipt is the sparse-sandbox `os.lstat`-only R3/R4 receipt SHA-256 `58f365b45d9f0d62d2e165f8bfd3e34414b9f16142a5faa2f9809e7713648c71`; it passes with no protected content opened and explicitly does not claim OS isolation. Final R1/R2 bundles wait for the final 15/18-cell schedule. |
| 5 | Full CPU report bytes | **PASS** | `V2DEV_CPU_TEST_REPORT.json` is published byte-for-byte at SHA-256 `b04829a42de204359918cf10a36b4034c46db0f48182259b35e3ed9ff7babc07`; it records `442 passed, 7 deselected`. |
| 6 | All-cell no-training preflight | **PENDING FINAL DISPOSITION** | `generate_no_training_preflight.py` compiled and completed a synthetic 15-cell smoke, including one independently anchored PREPARING → INITIALIZED → TERMINAL registry attempt. Authoritative execution is forbidden until BGT is admitted or formally not-admitted. |

## Authoritative execution location

V2 production runs execute from the isolated v2-dev worktree `/home/simx2204/v2_research/impl/DGCC`; the live tree `/home/simx2204/Workspaces/DGCC` remains exclusively assigned to the G6b campaign and delayed AMD-5 original V1 s6/s7 runs. V2 arms therefore share one final runtime closure without contaminating the original-protocol closure.

**V2 production runs execute from the isolated v2-dev worktree; the live-tree R3 failure is retained as an honest record of the campaign tree and is out of scope for the V2 launch firewall, whose authoritative receipts are the sparse-sandbox R3/R4 (`58f365b4…`).**

The sparse sandbox is receipt evidence for the same pinned V2 closure and same actual UID. It is not a second code lineage and does not imply OS, mount-namespace, ACL, or kernel isolation.

## Schedule contract

Only these schedules are valid:

- BGT admitted: `BB-D2`, `V1-D2`, `DMM`, `D1M`, `D11`, and `BGT`, each at seeds `{0,1,2}` — **18 runs**.
- BGT formally not-admitted: the same five non-BGT arms, each at seeds `{0,1,2}` — **15 runs**, no redistribution.

There is no 14-run schedule.

## Governance boundaries

- `v2_launch_code_manifest_sha256` is non-null in the external release governance artifact.
- `bgt_admitted_manifest_sha256` stays null until the separately approved synchronized-GPU gate finishes.
- `original_worktree_head_sha256` and `original_config_sha256` remain null by design. Original V1 s6/s7 launches therefore fail closed until the owner supplies independent pins.
- The attempt registry is **application-enforced append-only, non-reusing, hash-chained and externally anchored; it is not OS-level WORM**.
- R1/R2 are best-effort application controls, not OS isolation. R3/R4 are separate real-host evidence and cannot be omitted.

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

## Human-only release gates

1. Orchestrator sends explicit GPU-window approval; only then run the registered R5 latency protocol once.
2. Pin an admitted manifest and select 18 runs, or publish a formal not-admitted cutoff and select 15 runs.
3. Run the final no-training matrix generator on the resolved schedule and pin its R1/R2 and per-cell outputs.
