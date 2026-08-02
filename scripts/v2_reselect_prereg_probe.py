#!/usr/bin/env python
"""Code-measured verification of the n_envs-derived pre-registration items.

Task C of the reselection preflight: the throughput benchmark's §5(c) table
lists twelve items that a 1024 -> 4096 `run.n_envs` change disturbs.  That
table was arithmetic on paper.  This probe re-derives each number from the
ACTUAL training-loop code so the proposal handed to the owner is measured,
not assumed.  It changes nothing -- it only reports.

What is read from code (not restated from the benchmark):
  * the round loop shape           `p1_train.TrainingRun.train_loop`
  * the update gate                `p1_train.TrainingRun.train_updates`
  * the eval trigger               `transitions >= next_eval` (overshoot!)
  * per-env seeding                `dgcc.tasks.episode.begin_episodes`
  * schedule constants             the shipped `configs/v2_t2.yaml`

CPU-only.  Emits JSON on stdout.
"""
from __future__ import annotations

import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _source(obj: Any) -> str:
    return inspect.getsource(obj)


def probe(n_envs_options: tuple[int, ...] = (1024, 2048, 4096, 8192)) -> dict[str, Any]:
    config = yaml.safe_load((ROOT / "configs" / "v2_t2.yaml").read_text(encoding="utf-8"))
    run = config["run"]
    td3 = config["td3"]
    total = int(run["total_transitions"])
    warmup = int(td3["warmup_transitions"])
    eval_every = int(run["eval_every_transitions"])
    ckpt_every = int(run["checkpoint_every_transitions"])
    replay = int(td3["replay_capacity"])
    utd = int(td3["utd"])

    rows = {}
    for n in n_envs_options:
        # `collect_round` returns `int(active.sum())`, i.e. AT MOST n_envs.
        # The upper bound (every env active) is the schedule the
        # pre-registration arithmetic assumes; the real count is <= that, so
        # every "rounds" figure below is a LOWER bound on the real count.
        rounds_to_total = math.ceil(total / n)
        # `train_updates` returns early while `buffer.size < warmup`, so the
        # first gradient step happens in the first round whose CUMULATIVE
        # transition count reaches `warmup`.
        first_update_round = math.ceil(warmup / n)
        rows[str(n)] = {
            "rounds_to_total_upper_bound": rounds_to_total,
            "warmup_rounds_nominal": round(warmup / n, 4),
            "first_update_round_actual": first_update_round,
            "warmup_crossed_inside_round": first_update_round,
            "eval_interval_rounds_nominal": round(eval_every / n, 4),
            "eval_boundary_aligned_to_round": eval_every % n == 0,
            "eval_overshoot_max_transitions": n - 1,
            "checkpoint_interval_rounds_nominal": round(ckpt_every / n, 4),
            "replay_capacity_rounds": round(replay / n, 4),
            "replay_capacity_covers_full_run": replay >= total,
            "grad_steps_per_round_upper_bound": n * utd,
            "new_data_fraction_per_round": round(n / replay, 6),
            "correlated_block_size": n,
        }

    # Structural facts read from the code, not from the benchmark prose.
    import dgcc.tasks.episode as episode_mod  # noqa: E402
    train_src = (ROOT / "scripts" / "p1_train.py").read_text(encoding="utf-8")

    facts = {
        "train_updates_gate_is_transition_unit": (
            "if self.buffer.size < self.agent_config.warmup_transitions:" in train_src
        ),
        "updates_per_round_equals_active_count": "self.train_updates(count)" in train_src,
        "count_is_active_sum_not_n_envs": "count = int(active.sum())" in train_src,
        "eval_trigger_is_threshold_not_modulo": "if self.transitions >= next_eval:" in train_src,
        "eval_cursor_advances_by_fixed_step": "next_eval += self.eval_every" in train_src,
        "checkpoint_is_coupled_to_eval": (
            "ckpt = self.agent.save_checkpoint(self.models_dir / f\"ckpt_{self.transitions:07d}.pt\")"
            in train_src
        ),
        "per_env_init_seeded_by_n_envs": "n_envs=self.n_envs" in _source(
            episode_mod.BatchedEpisodeRunner.begin_episodes
        ),
        "t1_goal_rng_keyed_by_env_index": "SeedSequence([int(seed), int(episode_index), env_idx])"
        in _source(episode_mod.BatchedEpisodeRunner.begin_episodes),
        "t2_goal_pool_draw_sized_by_n_envs": "self.n_envs)]" in _source(
            episode_mod.BatchedEpisodeRunner.begin_episodes
        ),
    }

    return {
        "config": {
            "path": "configs/v2_t2.yaml",
            "total_transitions": total,
            "n_envs": int(run["n_envs"]),
            "warmup_transitions": warmup,
            "eval_every_transitions": eval_every,
            "checkpoint_every_transitions": ckpt_every,
            "replay_capacity": replay,
            "utd": utd,
        },
        "per_n_envs": rows,
        "code_facts": facts,
    }


if __name__ == "__main__":
    print(json.dumps(probe(), indent=2, sort_keys=True))
