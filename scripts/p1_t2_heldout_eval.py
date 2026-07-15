"""P1-M4 held-out one-shot evaluator (T2, 100 goals x 2 episodes = 200 rows).

Leakage contract (P1.md M4): the held-out split is evaluated EXACTLY ONCE per
run tag, at the end of training, against the val-selected checkpoint. The
once-guard is a durable exclusive claim file created BEFORE episode 1
(O_CREAT|O_EXCL, fsync(file) + fsync(parent dir)); a pre-existing claim hard-
refuses evaluation — investigate, never silently re-run.

Approval basis: gate-m3r-reconvene-2-20260713 choice B (M4 go); consensus
plan pending-approval sha256 78292468... (P2).

Usage:
    uv run python scripts/p1_t2_heldout_eval.py \
        --run-tag m4_t2_s0 --config configs/p1_t2.yaml --seed 0 \
        --selection-manifest outputs/metrics/p1_m4_ckpt_selection_m4_t2_s0.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: Held-out eval episode-index namespace — disjoint from the val namespace
#: (val uses 90_001+ via eval_episode_index_start) so curve seeds never
#: collide between val and held-out evaluations.
HELDOUT_EPISODE_INDEX_START = 95_001
HELDOUT_EPISODES_PER_GOAL = 2


class HeldoutClaimError(RuntimeError):
    """Raised when the one-shot claim cannot be acquired or persisted."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def claim_path_for(run_tag: str, claim_dir: Path = Path("outputs/metrics")) -> Path:
    return claim_dir / f"p1_heldout_claim_{run_tag}.json"


def acquire_heldout_claim(path: Path, payload: dict[str, Any]) -> None:
    """Create the exclusive claim durably, or raise. NEVER overwrites.

    Durability failure after creation also raises (no permission to
    evaluate): a claim that could vanish on power loss does not protect the
    leakage contract.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise HeldoutClaimError(
            f"held-out claim already exists: {path} — this run tag has already "
            "claimed (or attempted) its single held-out evaluation. Investigate; "
            "do NOT delete the claim to re-run."
        ) from exc
    try:
        os.write(fd, (json.dumps(payload, indent=1) + "\n").encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def atomic_publish_json(path: Path, payload: dict[str, Any]) -> None:
    """tmp + fsync + rename + parent-dir fsync (same pattern as the archive manifest)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def load_selection_manifest(path: Path) -> dict[str, Any]:
    """Load and validate checkpoint provenance; reject inconsistent manifests."""

    manifest = json.loads(path.read_text(encoding="utf-8"))
    for key in ("run_tag", "selected_ckpt", "ckpt_sha256", "selection_rule"):
        if not manifest.get(key):
            raise HeldoutClaimError(f"selection manifest missing required field {key!r}: {path}")
    ckpt = Path(manifest["selected_ckpt"])
    if not ckpt.exists():
        raise HeldoutClaimError(f"selected checkpoint does not exist: {ckpt}")
    actual = sha256_file(ckpt)
    if actual != manifest["ckpt_sha256"]:
        raise HeldoutClaimError(
            f"checkpoint sha256 mismatch: manifest {manifest['ckpt_sha256']} != on-disk {actual}"
        )
    return manifest


def expand_heldout_goals(pairs: list[tuple[dict[str, Any], Any]], per_goal: int = HELDOUT_EPISODES_PER_GOAL):
    """Expand (spec, goal) pairs to per-episode goal/label/family rows."""

    goals = [g for _, g in pairs for _ in range(per_goal)]
    labels = [s["goal_id"] for s, _ in pairs for _ in range(per_goal)]
    families = [str(s["family"]) for s, _ in pairs for _ in range(per_goal)]
    return goals, labels, families


def per_family_success(episodes: list[dict[str, Any]], families: list[str]) -> dict[str, float]:
    by_family: dict[str, list[bool]] = {}
    for row in episodes:
        fam = families[row["episode_id"] % len(families)] if len(families) else "?"
        by_family.setdefault(fam, []).append(bool(row["success"]))
    return {fam: float(sum(v)) / len(v) for fam, v in sorted(by_family.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--config", default="configs/p1_t2.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from dgcc.tasks.t2 import load_t2_split

    manifest = load_selection_manifest(Path(args.selection_manifest))
    if manifest["run_tag"] != args.run_tag:
        raise HeldoutClaimError(
            f"selection manifest run_tag {manifest['run_tag']!r} != --run-tag {args.run_tag!r}"
        )
    ckpt = Path(manifest["selected_ckpt"])

    pairs = load_t2_split("heldout")
    goals, labels, families = expand_heldout_goals(pairs)
    expected_rows = len(pairs) * HELDOUT_EPISODES_PER_GOAL
    assert len(pairs) == 100 and expected_rows == 200, (len(pairs), expected_rows)

    # ---- exclusive durable claim BEFORE any episode -----------------------
    claim = claim_path_for(args.run_tag)
    acquire_heldout_claim(
        claim,
        {
            "run_tag": args.run_tag,
            "ckpt_path": str(ckpt),
            "ckpt_sha256": manifest["ckpt_sha256"],
            "selection_manifest": str(args.selection_manifest),
            "episode_index_start": HELDOUT_EPISODE_INDEX_START,
            "n_goals": len(pairs),
            "episodes_per_goal": HELDOUT_EPISODES_PER_GOAL,
            "created_at": utc_now(),
            "pid": os.getpid(),
        },
    )
    print(f"held-out claim acquired: {claim}")

    # ---- evaluation via the driver's exact eval code path -----------------
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "p1_train", Path(__file__).resolve().parent / "p1_train.py"
    )
    p1_train = importlib.util.module_from_spec(spec)
    sys.modules["p1_train"] = p1_train
    spec.loader.exec_module(p1_train)

    run_args = argparse.Namespace(
        config=args.config,
        seed=args.seed,
        run_tag=f"{args.run_tag}_heldout",
        total_override=None,
        device=args.device,
    )
    run = p1_train.TrainingRun(run_args)
    run.agent.load_checkpoint(ckpt)
    # Swap the val episode set for the held-out set so deterministic_eval —
    # the SAME code path used for every val eval — scores held-out goals.
    run.val_goals = goals
    run.val_labels = labels
    run.build_scene()

    start = time.perf_counter()
    result = run.deterministic_eval(episode_index_start=HELDOUT_EPISODE_INDEX_START)
    wall_s = time.perf_counter() - start
    episodes = result["episodes"]
    assert len(episodes) == expected_rows, (
        f"held-out evaluated {len(episodes)} rows, expected exactly {expected_rows}"
    )

    payload = {
        "generated_at": utc_now(),
        "run_tag": args.run_tag,
        "seed": args.seed,
        "config": args.config,
        "ckpt_path": str(ckpt),
        "ckpt_sha256": manifest["ckpt_sha256"],
        "selection_manifest": str(args.selection_manifest),
        "selection_rule": manifest["selection_rule"],
        "episode_index_start": HELDOUT_EPISODE_INDEX_START,
        "n_goals": len(pairs),
        "episodes_per_goal": HELDOUT_EPISODES_PER_GOAL,
        "wall_s": wall_s,
        "per_family_success": per_family_success(episodes, families),
        "summary": {k: v for k, v in result.items() if k != "episodes"},
        "episodes": episodes,
    }
    out = Path("outputs/metrics") / f"p1_t2_heldout_{args.run_tag}.json"
    atomic_publish_json(out, payload)
    print(
        f"held-out result published: {out} success={result['success_rate']:.3f} "
        f"return={result['mean_return']:.3f} n={len(episodes)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
