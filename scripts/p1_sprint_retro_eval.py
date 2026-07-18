"""Sprint retroactive eval-only rerun for the M4 BB-reuse seeds (sprint_spec §1/§3).

Per M4 seed:
  1. Re-evaluate EVERY saved training checkpoint on val-50 under the sprint
     protocol (eval-wall guard K=5; same cadence/metric) — selection procedure
     unification for BB parity ("checkpoint eval-only 재실행으로 재선택").
  2. Re-select by the M4 rule (max val success; tie -> max return -> min
     transitions).
  3. One-shot sprint-heldout (t2_sprint_heldout_v1, 100 goals x 2 episodes)
     on the re-selected checkpoint with raw trajectories (§3), an exclusive
     claim, and an access audit log.

Covenants: M4 held-out 100 NEVER touched (retro scope = val 50 + sprint
heldout only — sprint_spec §3); claim files are O_CREAT|O_EXCL and never
deleted; sprint heldout episode namespace = 97_001 (disjoint from val 90k+
and M4 heldout 95k+).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from dgcc.analysis.sprint_claims import (
    CANONICAL_SPLIT_PATH,
    CANONICAL_SPLIT_SHA256,
    SprintClaimError,
    acquire_claim,
    atomic_publish,
    audit_claims,
    consume_claim_and_load_split,
    probe_manifest_register,
    sha256_file,
    parse_disposition_receipt,
    validate_checkpoint_arm,
)

SPRINT_HELDOUT_EPISODE_INDEX_START = 97_001
SPRINT_ACCESS_LOG = Path("outputs/metrics/t2_sprint_heldout_access.log")
WALL_GUARD_K = 5


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()




def build_run(config_path: str, seed: int, run_tag: str, device: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "p1_train", Path(__file__).resolve().parent / "p1_train.py"
    )
    p1_train = importlib.util.module_from_spec(spec)
    sys.modules["p1_train"] = p1_train
    spec.loader.exec_module(p1_train)
    run_args = argparse.Namespace(
        config=config_path, seed=seed, run_tag=run_tag, total_override=None, device=device
    )
    run = p1_train.TrainingRun(run_args)
    # Sprint protocol: guard ON for every eval in this script.
    run.config.setdefault("eval", {})["wall_guard_k"] = WALL_GUARD_K
    return run


def eval_with_recovery(run, *, episode_index_start: int, record_raw: bool = False, record_probe: bool = False, max_rebuilds: int = 8):
    """Mirror p1_train.eval_and_checkpoint's NaN-recovery loop (scene rebuild + retry)."""

    from dgcc.tasks.episode import is_nonfinite_error

    rebuilds = 0
    while True:
        try:
            return run.deterministic_eval(
                episode_index_start=episode_index_start, record_raw=record_raw, record_probe=record_probe
            )
        except (FloatingPointError, ValueError, RuntimeError) as exc:
            if not is_nonfinite_error(exc):
                raise
            rebuilds += 1
            if rebuilds > max_rebuilds:
                raise
            print(f"eval_recovery rebuild={rebuilds} error={exc}")
            run.build_scene()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", help="e.g. m4_t2_s0")
    parser.add_argument("--selection-out")
    parser.add_argument("--claim")
    parser.add_argument("--out")
    parser.add_argument("--human-disposition-receipt")
    parser.add_argument("--audit-legacy", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--config", default="configs/p1_t2.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.m4_tag = args.run_tag
    if args.audit_legacy:
        print(json.dumps({"schema_version": 1, "legacy": audit_claims(REPO / "outputs/metrics")}, indent=1))
        return 0
    if not args.run_tag or not args.selection_out or not args.claim or not args.out or args.seed is None:
        raise SprintClaimError("--run-tag, --selection-out, --claim, and --out are required for evaluation")
    if not args.human_disposition_receipt or not Path(args.human_disposition_receipt).is_file():
        raise SprintClaimError("new canonical claim requires a human disposition receipt")
    expected_selection = REPO / "outputs/metrics" / f"p1_sprint_retro_val_{args.run_tag}.json"
    legacy_claim = REPO / "outputs/metrics" / f"p1_bb_sprint_heldout_{args.run_tag}_claim.json"
    expected_claim = REPO / "outputs/metrics" / f"p1_bb_sprint_heldout_{args.run_tag}_reeval_claim.json"
    expected_out = REPO / "outputs/metrics" / f"p1_t2_sprint_heldout_{args.run_tag}.json"
    if any(Path(v).is_symlink() for v in (args.selection_out, args.claim, args.out)) or tuple(Path(v).absolute() for v in (args.selection_out, args.claim, args.out)) != (expected_selection.absolute(), expected_claim.absolute(), expected_out.absolute()):
        raise SprintClaimError("selection, claim, and out paths must be canonical non-symlink run-identity paths")
    if not legacy_claim.is_file(): raise SprintClaimError("disposition receipt must bind an existing legacy claim")
    _, receipt_sha = parse_disposition_receipt(args.human_disposition_receipt, legacy_claim_sha256=sha256_file(legacy_claim), run_tag=args.run_tag)
    receipt = json.loads(Path(args.human_disposition_receipt).read_text(encoding="utf-8"))
    if receipt["decision"] != "allow_reevaluation": raise SprintClaimError("disposition receipt does not allow re-evaluation")

    models_dir = REPO / "outputs/models" / args.m4_tag
    ckpts = sorted(models_dir.glob("ckpt_*.pt"))
    ckpts = [c for c in ckpts if "crash" not in c.name]
    assert ckpts, f"no checkpoints under {models_dir}"


    run = build_run(args.config, args.seed, f"{args.m4_tag}_sprint_retro", args.device)
    run.build_scene()

    # ---- 1. val-50 sprint-protocol re-evaluation of every checkpoint -------
    rows = []
    for ckpt in ckpts:
        transitions = int(ckpt.stem.split("_")[1])
        run.agent.load_checkpoint(ckpt)
        start = time.perf_counter()
        result = eval_with_recovery(run, episode_index_start=90_001 + transitions // 25_000)
        wall = time.perf_counter() - start
        rows.append(
            {
                "ckpt": str(ckpt),
                "ckpt_sha256": sha256_file(ckpt),
                "transitions": transitions,
                "success_rate": result["success_rate"],
                "mean_return": result["mean_return"],
                "eval_wall_guard_rate": result.get("eval_wall_guard_rate"),
                "wall_guard_k": result.get("wall_guard_k"),
                "wall_s": wall,
                "val_rows": [
                    [result["episodes"][i]["success"], result["episodes"][i + 1]["success"]]
                    for i in range(0, 100, 2)
                ],
            }
        )
        print(
            f"retro-val {args.m4_tag} @{transitions}: succ={result['success_rate']:.3f} "
            f"ret={result['mean_return']:.3f} guard_rate={result.get('eval_wall_guard_rate')} wall={wall:.0f}s"
        )
        run.begin_training_episodes()

    # ---- 2. re-selection (M4 rule) ----------------------------------------
    selected = max(rows, key=lambda r: (r["success_rate"], r["mean_return"], -r["transitions"]))
    retro_val = {
        "run_tag": args.m4_tag,
        "arm": "BB",
        "seed": args.seed,
        "task": "t2",
        "config_sha256": sha256_file(Path(args.config)),
        "ckpt_sha256": selected["ckpt_sha256"],
        "val_rows": selected["val_rows"],
        "selector_version": "sprint-retro-v1",
        "selected_ckpt": selected["ckpt"],
    }
    out_val = Path(args.selection_out)
    atomic_publish(out_val, retro_val)
    selection_sha = sha256_file(out_val)
    print(f"retro selection: {selected['ckpt']} (val {selected['success_rate']:.3f})")

    # Claim creation uses a source-anchored digest before the sole split consumer.
    claim = Path(args.claim)
    validate_checkpoint_arm(Path(selected["ckpt"]), "bb")
    capability = acquire_claim(
        claim,
        {
            "run_tag": args.m4_tag, "arm": "bb", "ckpt_sha256": selected["ckpt_sha256"],
            "split_sha256": CANONICAL_SPLIT_SHA256, "episode_index_start": SPRINT_HELDOUT_EPISODE_INDEX_START,
            "episode_namespace": SPRINT_HELDOUT_EPISODE_INDEX_START, "seed": args.seed,
            "config_sha256": retro_val["config_sha256"], "generation": "reeval",
            "legacy_claim_sha256": sha256_file(legacy_claim),
            "selection_manifest": str(out_val.resolve()), "selection_manifest_sha256": selection_sha,
            "disposition_receipt_sha256": receipt_sha,
        },
    )
    payload = consume_claim_and_load_split(capability, CANONICAL_SPLIT_PATH, access_log=SPRINT_ACCESS_LOG)
    from dgcc.tasks.t2 import build_t2_goal
    pairs = [(spec, build_t2_goal(spec)) for spec in payload["specs"]]
    split_sha = CANONICAL_SPLIT_SHA256
    goals = [g for _, g in pairs for _ in range(2)]
    labels = [s["goal_id"] for s, _ in pairs for _ in range(2)]
    print(f"sprint heldout claim acquired: {claim}")

    run.agent.load_checkpoint(selected["ckpt"])
    run.val_goals = goals
    run.val_labels = labels
    run.build_scene()
    start = time.perf_counter()
    result = eval_with_recovery(
        run, episode_index_start=SPRINT_HELDOUT_EPISODE_INDEX_START, record_raw=True, record_probe=True
    )
    wall = time.perf_counter() - start
    episodes = result["episodes"]
    assert len(episodes) == 200, len(episodes)

    raw_path = Path("outputs/metrics") / f"p1_raw_sprint_heldout_{args.m4_tag}.json.gz"
    with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
        json.dump({"m4_tag": args.m4_tag, "episodes": episodes}, handle)
    probe_path = Path("outputs/metrics") / f"p1_probe_sprint_heldout_{args.m4_tag}.h5"
    import importlib.util
    probe_spec = importlib.util.spec_from_file_location(
        "sprint_heldout_eval", REPO / "scripts" / "sprint_heldout_eval.py"
    )
    probe_module = importlib.util.module_from_spec(probe_spec)
    assert probe_spec.loader is not None
    probe_spec.loader.exec_module(probe_module)
    probe_module.write_probe_h5(
        probe_path,
        episodes,
        ckpt_sha=selected["ckpt_sha256"],
        split_sha=split_sha,
        claim_sha=sha256_file(claim),
    )
    probe_manifest_register(
        Path("outputs/metrics/sprint_probe_manifest.json"),
        probe_path,
        {"production_goal": "G-EV", "run_tag": args.m4_tag},
    )
    for ep in episodes:
        for key in ("x_initial", "x_steps", "x_terminal"):
            ep.pop(key, None)

    payload = {
        "generated_at": utc_now(),
        "m4_tag": args.m4_tag,
        "seed": args.seed,
        "ckpt": selected["ckpt"],
        "ckpt_sha256": selected["ckpt_sha256"],
        "split": "t2_sprint_heldout_v1",
        "split_sha256": split_sha,
        "claim_sha256": sha256_file(claim),
        "selection_manifest_sha256": selection_sha,
        "disposition_receipt_sha256": receipt_sha,
        "episode_index_start": SPRINT_HELDOUT_EPISODE_INDEX_START,
        "protocol": {"wall_guard_k": WALL_GUARD_K, "record_raw": True},
        "wall_s": wall,
        "raw_artifact": str(raw_path),
        "probe_artifact": str(probe_path),
        "summary": {k: v for k, v in result.items() if k != "episodes"},
        "episodes": episodes,
    }
    out = Path(args.out)
    atomic_publish(out, payload)
    print(
        f"sprint heldout published: {out} success={result['success_rate']:.3f} "
        f"return={result['mean_return']:.3f} guard_rate={result.get('eval_wall_guard_rate')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
