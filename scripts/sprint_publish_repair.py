"""Recover a failed sprint publish from durable raw episodes without split access."""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dgcc.analysis.sprint_claims import (
    CANONICAL_SPLIT_SHA256,
    _summary_aggregates,
    atomic_publish,
    canonical_raw_path,
    canonical_result_path,
    canonicalize_episode_ids,
    json_file,
    utc_now,
    validate_claim_payload,
)

EPISODE_INDEX_START = 97_001
_RAW_ONLY_EPISODE_KEYS = frozenset({
    "x_initial", "x_steps", "x_terminal", "probe_p", "probe_u",
})


def repair(run_tag: str, arm: str) -> Path:
    """Publish a canonical result from an already-consumed claim and raw rows."""
    result_path = canonical_result_path(run_tag, arm)
    claim_path = result_path.with_name(f"{result_path.stem}_claim.json")
    raw_path = canonical_raw_path(run_tag, arm)
    claim_value, claim_sha = json_file(claim_path, "claim")
    claim = validate_claim_payload(claim_value)
    if claim["run_tag"] != run_tag or claim["arm"] != arm.lower():
        raise RuntimeError("claim does not match repair identity")
    if claim["split_sha256"] != CANONICAL_SPLIT_SHA256:
        raise RuntimeError("claim is not for the canonical split")
    manifest_path = Path(claim["selection_manifest"])
    if manifest_path.is_symlink():
        raise RuntimeError("selection manifest must not be a symlink")
    manifest, manifest_sha = json_file(manifest_path, "selection manifest")
    if manifest_sha != claim["selection_manifest_sha256"]:
        raise RuntimeError("selection manifest digest does not match claim")
    with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
        raw: Any = json.load(handle)
    if not isinstance(raw, dict) or raw.get("run_tag") != run_tag or not isinstance(raw.get("episodes"), list):
        raise RuntimeError("raw artifact does not match repair identity")
    episodes = raw["episodes"]
    if len(episodes) != 200 or any(not isinstance(episode, dict) for episode in episodes):
        raise RuntimeError("raw artifact must contain 200 episode objects")
    canonicalize_episode_ids(episodes, int(claim["episode_namespace"]))
    for episode in episodes:
        for key in _RAW_ONLY_EPISODE_KEYS:
            episode.pop(key, None)
    summary = _summary_aggregates(episodes)
    payload = {
        "generated_at": utc_now(), "run_tag": run_tag, "arm": arm.lower(),
        "seed": claim["seed"], "config_sha256": claim["config_sha256"],
        "ckpt_sha256": claim["ckpt_sha256"], "split_sha256": claim["split_sha256"],
        "claim_sha256": claim_sha, "selection_manifest": claim["selection_manifest"],
        "selection_manifest_sha256": manifest_sha,
        "episode_namespace": claim["episode_namespace"],
        "selector_version": manifest["selector_version"], "val_rows": manifest["val_rows"],
        "summary": summary, "episodes": episodes,
        "repair_source": "raw_gz", "repaired_at": utc_now(),
    }
    atomic_publish(result_path, payload)
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    repair(args.run_tag, args.arm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
