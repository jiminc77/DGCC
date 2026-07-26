# V2 no-training preflight evidence

`generate_no_training_preflight.py` creates the final arm/seed launch-evidence matrix without starting an agent, using CUDA, training, or evaluation.

The authoritative production `--repo-root` is `/home/simx2204/v2_research/impl/DGCC`. `/home/simx2204/Workspaces/DGCC` is the campaign/original-protocol tree and must never be used or modified by this generator.

## Required inputs

- Owner-pinned isolated repo root `/home/simx2204/v2_research/impl/DGCC`, sparse training sandbox root `/home/simx2204/v2_research/runtime/DGCC-v2-12befdac`, runtime source commit `12befdac…`, and evidence-base commit `cfb5d823…`. These identities are constants, not caller-selected expectations.
- Final schema-v3 not-admitted governance plus the schema-v2 protocol-validation base governance. The former authorizes only 15 cells; the latter is retained solely so existing protocol validators can authenticate non-BGT manifests.
- Immutable `bgt_not_admitted.json`, exact runtime code manifest, canonical N_eff guard JSON, V2 config bytes, and the owner-accepted sparse-sandbox R3/R4 receipt.
- Nonempty protected-path and fresh V2 heldout-path identifier lists rooted under the pinned sparse sandbox.

The final schedule is fixed at 15 cells: five non-BGT arms at seeds `{0,1,2}`, seed-block interleaved, with no redistribution. The generator has no admitted/18-cell branch.

## Sensitive-path boundary

Protected and fresh heldout/probe paths are canonicalized under the pinned sparse sandbox and passed only to `generate_r3_r4_existence_receipt`, which uses `os.lstat`. Their contents are never opened or read. The generated footprint must exactly match the separately pinned authoritative R3/R4 receipt. R1 assets must not contain protected path tokens or protected roles.

The allowlist must cover every role the driver demands, not just the ones the launcher resolves before it starts: `p1_train.TrainingRun.__init__` reads `config` and then `t2_split`, and a missing `t2_split` fails the run after the agent exists but before its first transition.

R1 allowlist roles are a launcher contract, not free-form labels: `read_launch_asset_snapshot` matches the role string exactly, so the governance asset is published as `execution_governance` because that is the role `p1_sprint_train.py` requests. The canonical `configs/v2_t2.yaml` is additionally pinned as `canonical_config_source`, which proves the per-cell copy is byte-identical to it.

R1/R2 remain application-level controls. They are not OS-level WORM or OS isolation. R3/R4 provide the separate real-host existence/absence evidence required by the launch contract.

## Outputs

The output directory is staged and renamed atomically; an existing authoritative output is never replaced. Cell assets are written under the staging directory but pinned at their published paths, and the R1/R2 receipt bundles plus the launcher dry-run run after the rename so they observe the allowlist a launcher will actually resolve. A failure during that seal deletes the published tree, so partial evidence never survives.

- `preflight_matrix.json`: separately labeled runtime/evidence commits, formal BGT disposition, exactly 15 cells, final and protocol governance hashes, authoritative and fresh R3/R4 hashes, and no-GPU/no-training/no-eval/live-tree attestations derived from validated roots.
- `schedule_disposition` embeds the formal not-admitted artifact SHA, self-digest, exact rationale, and future-extension clause.
- `r3_r4_existence_receipt.json`: fresh `lstat`-only footprint whose records match the authoritative sparse-sandbox receipt.
- `cells/<ordinal-arm-seed>/launch_manifest.json`: exact protocol cell.
- `cells/<...>/asset_manifest.json`: R1 launch allowlist.
- `cells/<...>/protected_access_audit.jsonl`: hash-chained R2 audit with zero protected accesses.
- `cells/<...>/protocol_preflight_receipt.json`: authenticated protocol receipt.
- `cells/<...>/v2_t2.yaml`: byte-identical copy of the pinned config. `p1_sprint_train.py` resolves `sprint.amd5_preflight` relative paths against the directory of its `--config`, so a governed launch must pass this copy; one config SHA still covers all 15 cells while `manifest_path` selects the cell's own manifest.
- `cells/<...>/launch_dryrun_receipt.json`: the real launcher driven on production argv until `TrainingRun.run` is entered, one statement before `build_scene`. Everything before it is real -- asset gate, receipt binding, firewall config load, factory call, agent construction, initial-weights digest, and the `t2_split` read the driver performs right after the agent exists. Nothing after it runs: no scene, environment, transition, update, or evaluation. `CUDA_VISIBLE_DEVICES=""` forces cpu without altering the production argv, the attempt registry is re-rooted to scratch so the production registry is untouched, and the agent is released at the boundary.
- `cells/<...>/reports/receipt-bundles/<id>/`: immutable cross-linked R1/R2 receipts.
- `registry-smoke/`: one PREPARING → INITIALIZED → TERMINAL no-training attempt and independently verified terminal anchor.

The authoritative invocation requires `bgt_not_admitted.json`, `execution_governance.not-admitted.json`, `execution_governance.pending-bgt.json` as the protocol-validation base, and `r3_r4_training_sandbox_receipt.json`. The output name and root are fixed, exclusive-create, and forbidden from the live tree.
