"""P1-M5 helper: collect a small T2-validation transition sample (v2 h5).

Random policy over the 50 committed T2 val goals — this is a CODE-PATH sample
for `scripts/extract_latents.py` (M5 exit criterion), not a training or
evaluation artifact.  Held-out goals are never touched here.

Usage:
    uv run python scripts/p1_t2_val_sample.py --seed 0 \
        --out outputs/data/p1_t2_val_sample.h5 --rounds 5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dgcc.envs.dlolab import DLOLabEnv
from dgcc.goals.dual_goal import goal_curve
from dgcc.models.networks import DELTA_SCALE, K_NODES
from dgcc.rl.replay import goal_spec_hash, write_v2_transitions
from dgcc.tasks.domain import P1_LENGTH_M, p1_rope_params
from dgcc.tasks.episode import BatchedEpisodeRunner, EpisodeConfig
from dgcc.tasks.t2 import load_t2_split
from dgcc.utils.meta import get_git_commit_hash


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--out", default="outputs/data/p1_t2_val_sample.h5")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    params = p1_rope_params()
    pairs = load_t2_split("val")
    goal_pool = [g for _, g in pairs]
    goal_ids = {id(g): s["goal_id"] for s, g in pairs}

    env = DLOLabEnv(n_envs=args.n_envs, dt=0.001, substeps=5, rod_damping=10.0, rod_angular_damping=5.0)
    env.reset(params, init_shape="straight", seed=args.seed)
    runner = BatchedEpisodeRunner(env, params, EpisodeConfig())
    runner.begin_episodes(seed=args.seed, episode_index=1, goal_pool=goal_pool, auto_reset=True)

    rows: dict[str, list] = {name: [] for name in (
        "X_before", "X_after", "p", "delta", "lift", "grasp_success", "settle_steps",
        "rope_params", "seed", "sim", "timestamp", "commit_hash", "task_id", "goal_id",
        "goal_spec_hash", "goal_curve", "episode_id", "step_index", "reward", "done",
        "provenance",
    )}
    commit = get_git_commit_hash()
    now = datetime.now(timezone.utc).isoformat()
    params_json = json.dumps(params.__dict__, sort_keys=True, default=float)

    for round_idx in range(args.rounds):
        curves = np.stack([goal_curve(g, P1_LENGTH_M) for g in runner.goals])
        X = env.get_centerline_batch()
        p = rng.integers(0, K_NODES, size=args.n_envs)
        delta = rng.uniform(-DELTA_SCALE, DELTA_SCALE, size=(args.n_envs, 3))
        lift = ["high" if v > 0.5 else "low" for v in rng.random(args.n_envs)]
        record = runner.step(p, delta, lift, rng=rng)
        if record.get("discarded"):
            print(f"round {round_idx}: discarded ({record['reason']}) — skipped")
            continue
        active = np.asarray(record["active"], dtype=bool)
        for i in np.flatnonzero(active):
            g = runner.goals[i]
            rows["X_before"].append(record["X_before"][i])
            rows["X_after"].append(record["X_after"][i])
            rows["p"].append(int(p[i]))
            rows["delta"].append(delta[i])
            rows["lift"].append(lift[i])
            rows["grasp_success"].append(bool(record["grasp_success"][i]))
            rows["settle_steps"].append(int(record["settle_steps"][i]))
            rows["rope_params"].append(params_json)
            rows["seed"].append(args.seed)
            rows["sim"].append("dlolab")
            rows["timestamp"].append(now)
            rows["commit_hash"].append(commit)
            rows["task_id"].append("t2")
            rows["goal_id"].append(goal_ids.get(id(g), "t2_val_unknown"))
            rows["goal_spec_hash"].append(goal_spec_hash(curves[i]))
            rows["goal_curve"].append(curves[i])
            rows["episode_id"].append(round_idx)
            rows["step_index"].append(int(runner.t[i]) if hasattr(runner, "t") else 0)
            rows["reward"].append(float(record["reward"][i]))
            rows["done"].append(bool(record["done"][i]))
            rows["provenance"].append("m5_t2_val_sample")

    count = len(rows["p"])
    meta = {
        "purpose": "M5 latent-extraction code-path sample (T2 val, random policy)",
        "generated_at": now,
        "git_commit": commit,
        "seed": args.seed,
        "n_envs": args.n_envs,
        "rounds": args.rounds,
        "record_count": count,
    }
    write_v2_transitions(args.out, rows, meta)
    print(f"sample written: {args.out} records={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
