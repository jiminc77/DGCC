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


def load_sprint_heldout():
    from dgcc.tasks.t2 import build_t2_goal

    split_path = REPO / "src" / "dgcc" / "tasks" / "splits" / "t2_sprint_heldout_v1.json"
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    assert payload["n_goals"] == 100 and payload["overlap_with_t2_v1"] == 0
    # Access audit (sprint_spec §6: 허용 접촉 외 로드는 위반)
    SPRINT_ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(SPRINT_ACCESS_LOG, "a", encoding="utf-8") as log:
        log.write(
            f"{utc_now()} pid={os.getpid()} n_goals={payload['n_goals']} "
            f"argv={' '.join(sys.argv[:4])}\n"
        )
    pairs = [(spec, build_t2_goal(spec)) for spec in payload["specs"]]
    return pairs, sha256_file(split_path)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m4-tag", required=True, help="e.g. m4_t2_s0")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", default="configs/p1_t2.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    models_dir = Path("outputs/models") / args.m4_tag
    ckpts = sorted(models_dir.glob("ckpt_*.pt"))
    ckpts = [c for c in ckpts if "crash" not in c.name]
    assert ckpts, f"no checkpoints under {models_dir}"

    claim = Path("outputs/metrics") / f"p1_sprint_heldout_claim_{args.m4_tag}.json"
    if claim.exists():
        raise SystemExit(f"sprint heldout already claimed for {args.m4_tag}: {claim}")

    run = build_run(args.config, args.seed, f"{args.m4_tag}_sprint_retro", args.device)
    run.build_scene()

    # ---- 1. val-50 sprint-protocol re-evaluation of every checkpoint -------
    rows = []
    for ckpt in ckpts:
        transitions = int(ckpt.stem.split("_")[1])
        run.agent.load_checkpoint(ckpt)
        start = time.perf_counter()
        result = run.deterministic_eval(episode_index_start=90_001 + transitions // 25_000)
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
        "generated_at": utc_now(),
        "m4_tag": args.m4_tag,
        "seed": args.seed,
        "protocol": {"wall_guard_k": WALL_GUARD_K, "val_goals": 50, "episodes_per_goal": 2},
        "selection_rule": "max val success_rate; tie -> max mean_return; tie -> min transitions",
        "selected_ckpt": selected["ckpt"],
        "selected_ckpt_sha256": selected["ckpt_sha256"],
        "selected_transitions": selected["transitions"],
        "rows": rows,
    }
    out_val = Path("outputs/metrics") / f"p1_sprint_retro_val_{args.m4_tag}.json"
    out_val.write_text(json.dumps(retro_val, indent=1) + "\n", encoding="utf-8")
    print(f"retro selection: {selected['ckpt']} (val {selected['success_rate']:.3f})")

    # ---- 3. one-shot sprint-heldout on the re-selected checkpoint ----------
    pairs, split_sha = load_sprint_heldout()
    goals = [g for _, g in pairs for _ in range(2)]
    labels = [s["goal_id"] for s, _ in pairs for _ in range(2)]

    fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "m4_tag": args.m4_tag,
                "ckpt": selected["ckpt"],
                "ckpt_sha256": selected["ckpt_sha256"],
                "split_sha256": split_sha,
                "episode_index_start": SPRINT_HELDOUT_EPISODE_INDEX_START,
                "created_at": utc_now(),
                "pid": os.getpid(),
            },
            handle,
            indent=1,
        )
    print(f"sprint heldout claim acquired: {claim}")

    run.agent.load_checkpoint(selected["ckpt"])
    run.val_goals = goals
    run.val_labels = labels
    run.build_scene()
    start = time.perf_counter()
    result = run.deterministic_eval(
        episode_index_start=SPRINT_HELDOUT_EPISODE_INDEX_START, record_raw=True
    )
    wall = time.perf_counter() - start
    episodes = result["episodes"]
    assert len(episodes) == 200, len(episodes)

    raw_path = Path("outputs/metrics") / f"p1_raw_sprint_heldout_{args.m4_tag}.json.gz"
    with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
        json.dump({"m4_tag": args.m4_tag, "episodes": episodes}, handle)
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
        "episode_index_start": SPRINT_HELDOUT_EPISODE_INDEX_START,
        "protocol": {"wall_guard_k": WALL_GUARD_K, "record_raw": True},
        "wall_s": wall,
        "raw_artifact": str(raw_path),
        "summary": {k: v for k, v in result.items() if k != "episodes"},
        "episodes": episodes,
    }
    out = Path("outputs/metrics") / f"p1_t2_sprint_heldout_{args.m4_tag}.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    print(
        f"sprint heldout published: {out} success={result['success_rate']:.3f} "
        f"return={result['mean_return']:.3f} guard_rate={result.get('eval_wall_guard_rate')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
