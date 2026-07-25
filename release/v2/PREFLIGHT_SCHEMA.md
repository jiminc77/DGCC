# V2 no-training preflight evidence

`generate_no_training_preflight.py` creates the final arm/seed launch-evidence matrix without starting an agent, using CUDA, training, or evaluation.

The authoritative production `--repo-root` is `/home/simx2204/v2_research/impl/DGCC`. `/home/simx2204/Workspaces/DGCC` is the campaign/original-protocol tree and must never be used or modified by this generator.

## Required inputs

- Exact isolated release root and approved source commit.
- Independently pinned execution-governance JSON.
- Exact runtime code manifest, canonical N_eff guard JSON, and V2 config bytes.
- One final BGT disposition:
  - `admitted`: governance contains the non-null admission-manifest SHA and `--bgt-admission` supplies those exact bytes.
  - `not-admitted`: governance keeps the admission pin null and `--bgt-cutoff-state` supplies the formal hash-chained cutoff state.
- Real training-host root plus nonempty old protected-path and fresh V2 heldout-path identifier lists.

The final schedule is derived rather than supplied: 18 cells for admitted BGT or 15 cells for not-admitted BGT, always seeds `{0,1,2}` and never redistribution. Any 14-run or duplicate-cell shape is rejected.

## Sensitive-path boundary

Protected and fresh heldout/probe paths are passed only to `generate_r3_r4_existence_receipt`, which uses `os.lstat`. Their contents are never opened or read. R1 assets must not contain protected path tokens or protected roles. The receipt records the actual UID/GID, host, working directory, path existence metadata, and `content_opened=false`.

R1/R2 remain application-level controls. They are not OS-level WORM or OS isolation. R3/R4 provide the separate real-host existence/absence evidence required by the launch contract.

## Outputs

The output directory is staged and renamed atomically; an existing authoritative output is never replaced.

- `preflight_matrix.json`: source commit, final BGT disposition, 15/18 count, all cells, artifact hashes, G3 links, and no-GPU/no-training/no-eval attestations.
- `r3_r4_existence_receipt.json`: shared real-host `lstat`-only receipt.
- `cells/<ordinal-arm-seed>/launch_manifest.json`: exact protocol cell.
- `cells/<...>/asset_manifest.json`: R1 launch allowlist.
- `cells/<...>/protected_access_audit.jsonl`: hash-chained R2 audit with zero protected accesses.
- `cells/<...>/protocol_preflight_receipt.json`: authenticated protocol receipt.
- `cells/<...>/reports/receipt-bundles/<id>/`: immutable cross-linked R1/R2 receipts.
- `registry-smoke/`: one PREPARING → INITIALIZED → TERMINAL no-training attempt and independently verified terminal anchor.

Example invocation is intentionally deferred until the orchestrator supplies the final BGT disposition and, for admission, the separately approved GPU-latency artifact. Running the tool while BGT is merely pending is forbidden.
