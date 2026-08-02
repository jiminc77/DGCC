#!/usr/bin/env python3
"""Code-measured eval/checkpoint schedule and run-termination audit.

Amendment 6 appendix item: `eval_every_transitions` 25,000 -> 24,576
(= 6 x n_envs 4,096).  This script does not guess the schedule; it replays the
exact arithmetic of `P1Trainer.run()` in `scripts/p1_train.py`::

    next_eval = self.eval_every                      # :1403
    while self.transitions < self.total:             # :1406
        count = self.collect_round()                 # :1408  -> int(active.sum())
        self.transitions += count                    # collect_round :908-938
        if self.transitions >= next_eval:            # :1427
            self.eval_and_checkpoint(final=self.transitions >= self.total)
            next_eval += self.eval_every             # :1429
    ...
    if not self.eval_history or self.eval_history[-1]["transitions"] < self.transitions:
        self.eval_and_checkpoint(final=True)         # :1444-1445

`count` is `int(active.sum())` over the batch, i.e. `n_envs` in the nominal
(no-discard) round; discarded rounds only ever lower it, which delays but never
advances the termination round.  The nominal schedule below is therefore the
earliest-termination bound.

Note: `checkpoint_every_transitions` is not read by `p1_train.py` at all --
`eval_and_checkpoint()` writes the checkpoint on the eval boundary -- so the
two keys must be kept equal or the config claims a schedule the code does not
implement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def simulate(total: int, n_envs: int, eval_every: int) -> dict[str, Any]:
    transitions = 0
    rounds = 0
    next_eval = eval_every
    evals: list[dict[str, Any]] = []
    while transitions < total:
        rounds += 1
        transitions += n_envs
        if transitions >= next_eval:
            evals.append(
                {
                    "round": rounds,
                    "transitions": transitions,
                    "boundary": next_eval,
                    "exact_multiple": transitions == next_eval,
                    "final_flag": transitions >= total,
                }
            )
            next_eval += eval_every
    terminal = None
    if not evals or evals[-1]["transitions"] < transitions:
        terminal = {"round": rounds, "transitions": transitions, "final_flag": True}
    return {
        "total_transitions": total,
        "n_envs": n_envs,
        "eval_every_transitions": eval_every,
        "eval_every_is_multiple_of_n_envs": eval_every % n_envs == 0,
        "rounds_per_eval": (eval_every / n_envs),
        "termination_round": rounds,
        "termination_transitions": transitions,
        "overshoot_transitions": transitions - total,
        "interval_evals": evals,
        "n_interval_evals": len(evals),
        "terminal_eval": terminal,
        "n_evals_total": len(evals) + (1 if terminal else 0),
        "all_interval_boundaries_exact": all(e["exact_multiple"] for e in evals),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "v2_t2.yaml",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    run_cfg = cfg.get("run", {})
    total = int(run_cfg["total_transitions"])
    n_envs = int(run_cfg["n_envs"])
    eval_every = int(run_cfg.get("eval_every_transitions", 25_000))
    ckpt_every = run_cfg.get("checkpoint_every_transitions")

    trainer_src = (Path(__file__).resolve().parents[1] / "scripts" / "p1_train.py").read_text()
    report = {
        "config_path": str(args.config),
        "config_sha256": sha256_file(args.config),
        "p1_train_sha256": hashlib.sha256(trainer_src.encode()).hexdigest(),
        "checkpoint_every_transitions": ckpt_every,
        "checkpoint_key_read_by_trainer": "checkpoint_every_transitions" in trainer_src,
        "checkpoint_written_by": "eval_and_checkpoint() on the eval boundary",
        "configured": simulate(total, n_envs, eval_every),
        "legacy_25000_for_contrast": simulate(total, n_envs, 25_000),
    }
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
