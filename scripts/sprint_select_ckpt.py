"""Select a new sprint run checkpoint using guard-on val-50 evaluation only."""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dgcc.analysis.sprint_claims import SprintClaimError, atomic_publish, sha256_file, validate_checkpoint_arm

WALL_GUARD_K = 5
VAL_EPISODE_INDEX_START = 90_001
SELECTOR_VERSION = "sprint-select-v1"


def _retro_module():
    """Load the established sprint evaluation path without changing its behavior."""
    spec = importlib.util.spec_from_file_location(
        "p1_sprint_retro_eval_shared", REPO / "scripts" / "p1_sprint_retro_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def select_checkpoint(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the preregistered success, return, then earlier-checkpoint rule."""
    if not rows:
        raise SprintClaimError("no checkpoint evaluation rows")
    return max(rows, key=lambda row: (row["success_rate"], row["mean_return"], -row["transitions"]))


def selection_manifest(*, run_tag: str, arm: str, seed: int, config_path: Path, selected: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_tag": run_tag,
        "arm": arm,
        "seed": seed,
        "task": "t2",
        "config_sha256": sha256_file(config_path),
        "ckpt_sha256": selected["ckpt_sha256"],
        "val_rows": selected["val_rows"],
        "selector_version": SELECTOR_VERSION,
        "selected_ckpt": selected["ckpt"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--config", required=True)
    parser.add_argument("--selection-out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config_path = Path(args.config)
    config_path = config_path if config_path.is_absolute() else REPO / config_path
    if config_path.is_symlink() or not config_path.is_file():
        raise SprintClaimError("config must be a non-symlink file")
    output = Path(args.selection_out)
    if output.is_symlink():
        raise SprintClaimError("selection output must not be a symlink")

    models_dir = REPO / "outputs/models" / args.run_tag
    checkpoints = sorted(path for path in models_dir.glob("ckpt_*.pt") if "crash" not in path.name)
    if not checkpoints:
        raise SprintClaimError(f"no checkpoints under {models_dir}")

    retro = _retro_module()
    run = retro.build_run(str(config_path), args.seed, f"{args.run_tag}_sprint_select", args.device)
    run.build_scene()
    rows = []
    for checkpoint in checkpoints:
        validate_checkpoint_arm(checkpoint, args.arm)
        try:
            transitions = int(checkpoint.stem.split("_")[1])
        except (IndexError, ValueError) as exc:
            raise SprintClaimError(f"checkpoint name lacks transition count: {checkpoint.name}") from exc
        run.agent.load_checkpoint(checkpoint)
        started = time.perf_counter()
        result = retro.eval_with_recovery(
            run, episode_index_start=VAL_EPISODE_INDEX_START + transitions // 25_000
        )
        wall_s = time.perf_counter() - started
        episodes = result.get("episodes", [])
        if len(episodes) != 100:
            raise SprintClaimError("val selection evaluation must produce exactly 100 episodes")
        rows.append({
            "ckpt": str(checkpoint), "ckpt_sha256": sha256_file(checkpoint),
            "transitions": transitions, "success_rate": result["success_rate"],
            "mean_return": result["mean_return"], "wall_s": wall_s,
            "val_rows": [[episodes[index]["success"], episodes[index + 1]["success"]] for index in range(0, 100, 2)],
        })
        run.begin_training_episodes()

    selected = select_checkpoint(rows)
    atomic_publish(output, selection_manifest(
        run_tag=args.run_tag, arm=args.arm, seed=args.seed,
        config_path=config_path, selected=selected,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
