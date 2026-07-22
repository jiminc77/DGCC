#!/usr/bin/env python3
"""One-shot patch-only rollout evaluator; GPU rollout execution is injectable."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from dgcc.analysis.sprint_claims import (
    PATCH_EVAL_SPLIT_PATH,
    PATCH_EVAL_SPLIT_SHA256,
    REPO_ROOT,
    SprintClaimError,
    acquire_patch_claim,
    atomic_publish,
    canonical_patch_claim_path,
    canonical_patch_result_path,
    consume_patch_claim_and_load_split,
    sha256_file,
    validate_checkpoint_arm,
)

PATCH_ACCESS_LOG = REPO / "outputs/metrics/t2_patch_eval_access.log"
CONDITIONS = ("unpatched", "a0_real", "a0_null", "a1_real", "a1_null")
LENGTH_RATIOS = (0.75, 1.0, 1.25)


def canonical_paths(run_tag: str, arm: str) -> tuple[Path, Path]:
    return canonical_patch_claim_path(run_tag, arm), canonical_patch_result_path(run_tag, arm)


def _config_sha(config: str) -> str:
    return sha256_file(REPO / config)


def load_selection_manifest(path: Path, run_tag: str, arm: str, seed: int, config: str) -> tuple[dict[str, Any], str]:
    """Use the same checkpoint-selection identity contract as held-out eval."""
    from sprint_heldout_eval import load_selection_manifest as load
    return load(path, run_tag, arm, seed, config)


def aggregate_rollouts(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, dict[str, float | int]]]:
    buckets: dict[str, dict[str, list[dict[str, Any]]]] = {condition: {str(ratio): [] for ratio in LENGTH_RATIOS} for condition in CONDITIONS}
    for row in rows:
        condition, ratio = row.get("condition"), row.get("length_ratio")
        if condition not in CONDITIONS or ratio not in LENGTH_RATIOS:
            raise SprintClaimError("rollout row has an invalid condition or OOD length ratio")
        if not isinstance(row.get("success"), bool) or not isinstance(row.get("return"), (int, float)):
            raise SprintClaimError("rollout row requires boolean success and numeric return")
        buckets[condition][str(ratio)].append(row)
    summary: dict[str, dict[str, dict[str, float | int]]] = {}
    for condition, ratios in buckets.items():
        summary[condition] = {}
        for ratio, bucket in ratios.items():
            if not bucket:
                raise SprintClaimError(f"rollout result is missing {condition} at length {ratio}")
            summary[condition][ratio] = {
                "n_episodes": len(bucket),
                "success_rate": sum(row["success"] for row in bucket) / len(bucket),
                "mean_return": sum(float(row["return"]) for row in bucket) / len(bucket),
            }
    return summary


def run_patch_rollouts(*, payload: dict[str, Any], manifest: dict[str, Any], config: str, seed: int, device: str) -> list[dict[str, Any]]:
    """GPU-only seam: apply sealed A0/A1 Q-path patches and return episode rows.

    Policy action selection remains standard; only critic-Q measurements receive h_p
    substitution.  This CPU entry point deliberately has no implicit simulator path.
    """
    raise RuntimeError("patch rollout execution requires the GPU patch-forward backend; mock run_patch_rollouts for CPU contract tests")


def canonical_result_payload(*, run_tag: str, arm: str, seed: int, manifest: dict[str, Any], selection_manifest: str, selection_sha: str, claim_sha: str, config_sha: str, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    episodes = list(rows)
    return {
        "schema_version": 1,
        "run_tag": run_tag,
        "arm": arm,
        "seed": seed,
        "config_sha256": config_sha,
        "ckpt_sha256": manifest["ckpt_sha256"],
        "split_sha256": PATCH_EVAL_SPLIT_SHA256,
        "claim_sha256": claim_sha,
        "selection_manifest": selection_manifest,
        "selection_manifest_sha256": selection_sha,
        "operator": "sealed_sprint_patching",
        "operator_sha256": sha256_file(REPO / "src/dgcc/analysis/sprint_patching.py"),
        "pre_post_hashes": sorted({
            (str(row.get("pre_hash", "")), str(row.get("post_hash", "")))
            for row in episodes
        }),
        "patch_q_path_only": True,
        "conditions": aggregate_rollouts(episodes),
        "episodes": episodes,
    }


def main(rollout_runner: Callable[..., list[dict[str, Any]]] = run_patch_rollouts) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--config", default="configs/p1_t2.yaml")
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--claim", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    expected_claim, expected_out = canonical_paths(args.run_tag, args.arm)
    claim_path, out_path = Path(args.claim), Path(args.out)
    if claim_path.is_symlink() or out_path.is_symlink() or claim_path.absolute() != expected_claim.absolute() or out_path.absolute() != expected_out.absolute():
        raise SprintClaimError("claim and out paths must be canonical non-symlink patch run-identity paths")
    selection_path = Path(args.selection_manifest)
    if selection_path.is_symlink() or not selection_path.is_absolute():
        raise SprintClaimError("selection manifest path must be absolute and not a symlink")
    manifest, selection_sha = load_selection_manifest(selection_path, args.run_tag, args.arm, args.seed, args.config)
    validate_checkpoint_arm(Path(manifest["selected_ckpt"]), args.arm)
    config_sha = _config_sha(args.config)
    capability = acquire_patch_claim(expected_claim, {
        "run_tag": args.run_tag, "arm": args.arm, "ckpt_sha256": manifest["ckpt_sha256"],
        "split_sha256": PATCH_EVAL_SPLIT_SHA256, "seed": args.seed, "config_sha256": config_sha,
        "selection_manifest": str(selection_path), "selection_manifest_sha256": selection_sha,
        "episode_namespace": 97_001, "n_goals": 100,
    })
    payload = consume_patch_claim_and_load_split(capability, PATCH_EVAL_SPLIT_PATH, access_log=PATCH_ACCESS_LOG)
    rows = rollout_runner(payload=payload, manifest=manifest, config=args.config, seed=args.seed, device=args.device)
    canonical = canonical_result_payload(run_tag=args.run_tag, arm=args.arm, seed=args.seed, manifest=manifest, selection_manifest=str(selection_path.resolve()), selection_sha=selection_sha, claim_sha=sha256_file(expected_claim), config_sha=config_sha, rows=rows)
    atomic_publish(expected_out, canonical)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
