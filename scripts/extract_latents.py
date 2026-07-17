"""P1-M5 latent extraction driver: v2 transition h5 in → latent h5 out.

Usage:
    uv run python scripts/extract_latents.py \
        --checkpoint outputs/models/m4_t2_s0/ckpt_0300032.pt \
        --transitions outputs/data/p1_t2_val_sample.h5 \
        --out outputs/data/latents/m4_t2_s0.h5

Output layout: one dataset per LATENT_SPEC name plus pass-through
identification columns (p, delta, lift, episode_id, step_index, goal_id),
attrs ``meta_json`` = {checkpoint hash/config, transitions provenance, git
hash, generated_at, record_count}.  docs/latent_api.md documents the layout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dgcc.analysis.latent_api import LATENT_SPEC, FrozenLatentExtractor, sha256_file
from dgcc.rl.replay import read_v2_transitions
from dgcc.utils.meta import get_git_commit_hash

BATCH = 256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--transitions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    columns, transitions_meta = read_v2_transitions(args.transitions)
    count = len(columns["p"])
    extractor = FrozenLatentExtractor.from_checkpoint(args.checkpoint, device=args.device)
    params_before = extractor.parameter_sha256()

    chunks: dict[str, list[np.ndarray]] = {name: [] for name in LATENT_SPEC}
    for start in range(0, count, BATCH):
        sel = slice(start, min(start + BATCH, count))
        latents = extractor.extract(
            columns["X_before"][sel],
            columns["goal_curve"][sel],
            columns["p"][sel],
            columns["delta"][sel],
            columns["lift"][sel],
        )
        for name in LATENT_SPEC:
            chunks[name].append(latents[name])

    if extractor.parameter_sha256() != params_before:
        raise AssertionError("frozen guarantee violated: parameters changed during extraction")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit_hash(),
        "record_count": count,
        "extractor": extractor.metadata(),
        "transitions_file": str(args.transitions),
        "transitions_sha256": sha256_file(Path(args.transitions)),
        "transitions_meta": transitions_meta,
    }
    with h5py.File(out_path, "w") as h5:
        h5.attrs["meta_json"] = json.dumps(meta, ensure_ascii=False, sort_keys=True)
        h5.attrs["record_count"] = count
        for name in LATENT_SPEC:
            data = np.concatenate(chunks[name], axis=0)
            assert data.shape[0] == count, (name, data.shape)
            h5.create_dataset(name, data=data)
        # Pass-through identification columns for P2 join-back.
        str_dtype = h5py.string_dtype(encoding="utf-8")
        h5.create_dataset("p", data=np.asarray(columns["p"], dtype=np.int64))
        h5.create_dataset("delta", data=np.asarray(columns["delta"], dtype=np.float64))
        h5.create_dataset(
            "lift", data=np.asarray([str(v) for v in columns["lift"]], dtype=object), dtype=str_dtype
        )
        h5.create_dataset("episode_id", data=np.asarray(columns["episode_id"], dtype=np.int64))
        h5.create_dataset("step_index", data=np.asarray(columns["step_index"], dtype=np.int64))
        h5.create_dataset(
            "goal_id",
            data=np.asarray([str(v) for v in columns["goal_id"]], dtype=object),
            dtype=str_dtype,
        )

    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()[:16]
    print(f"latents written: {out_path} records={count} sha16={digest} ckpt={extractor.ckpt_sha256[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
