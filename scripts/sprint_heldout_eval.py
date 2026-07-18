"""One-shot sprint held-out evaluator (100 goals × 2 episodes)."""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from dgcc.analysis.sprint_claims import (  # noqa: E402
    SprintClaimError,
    acquire_claim,
    atomic_publish,
    probe_manifest_register,
    record_access,
    require_metric_lock,
    sha256_file,
    utc_now,
)

EPISODE_INDEX_START = 97_001
WALL_GUARD_K = 5
SPRINT_ACCESS_LOG = Path("outputs/metrics/t2_sprint_heldout_access.log")
PROBE_MANIFEST = Path("outputs/metrics/sprint_probe_manifest.json")


def load_selection_manifest(path: Path, run_tag: str) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for key in ("selected_ckpt", "ckpt_sha256", "selection_rule"):
        if not manifest.get(key):
            raise SprintClaimError(f"selection manifest missing {key}: {path}")
    if manifest.get("run_tag", manifest.get("m4_tag")) != run_tag:
        raise SprintClaimError("selection manifest run tag does not match --run-tag")
    ckpt = Path(manifest["selected_ckpt"])
    if not ckpt.is_file() or sha256_file(ckpt) != manifest["ckpt_sha256"]:
        raise SprintClaimError("selection checkpoint sha256 does not match disk")
    # Sprint selection is explicitly the val-50 rule, never held-out selection.
    if "val" not in str(manifest["selection_rule"]).lower():
        raise SprintClaimError("selection manifest must attest a val-50 selection rule")
    return manifest


def validate_sprint_split_path(split_path: Path) -> None:
    if "m4" in split_path.name.lower():
        raise SprintClaimError("M4 held-out split is forbidden for sprint evaluation")
    expected = REPO / "src/dgcc/tasks/splits/t2_sprint_heldout_v1.json"
    if split_path.resolve() != expected.resolve() and split_path.name != "synthetic_sprint_split.json":
        raise SprintClaimError("only t2_sprint_heldout_v1 split is permitted")


def load_sprint_split(split_path: Path):
    validate_sprint_split_path(split_path)
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    if payload.get("n_goals") != 100 or len(payload.get("specs", [])) != 100:
        raise SprintClaimError("sprint split must contain exactly 100 goals")
    from dgcc.tasks.t2 import build_t2_goal
    return [(spec, build_t2_goal(spec)) for spec in payload["specs"]]


def write_probe_h5(path: Path, episodes: list[dict[str, Any]], *, ckpt_sha: str, split_sha: str, claim_sha: str) -> None:
    import h5py
    path.parent.mkdir(parents=True, exist_ok=True)
    # One row per episode is retained even when a backend does not expose steps.
    def values(key: str, default: Any) -> np.ndarray:
        return np.asarray([episode.get(key, default) for episode in episodes])
    with h5py.File(path, "w") as h5:
        h5.attrs["schema_version"] = 1
        for key, default in (("x_before", []), ("x_after", []), ("goal", []), ("p", -1), ("u", []),
                             ("episode_id", -1), ("step_index", 0)):
            h5.create_dataset(key, data=values({"x_before": "x_initial", "x_after": "x_terminal"}.get(key, key), default))
        h5.create_dataset("goal_id", data=np.asarray([str(e.get("goal_id", "")) for e in episodes], dtype=h5py.string_dtype()))
        flags = h5.create_group("flags")
        for key in ("truncated", "reseed", "guard"):
            flags.create_dataset(key, data=values(key, False).astype(bool))
        for key, value in (("ckpt_sha256", ckpt_sha), ("split_sha256", split_sha), ("claim_sha256", claim_sha)):
            h5.attrs[key] = value


def build_run(config: str, seed: int, run_tag: str, device: str):
    spec = importlib.util.spec_from_file_location("p1_train", REPO / "scripts/p1_train.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["p1_train"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    run = module.TrainingRun(argparse.Namespace(config=config, seed=seed, run_tag=run_tag, total_override=None, device=device))
    run.config.setdefault("eval", {})["wall_guard_k"] = WALL_GUARD_K
    return run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--claim", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lock")
    parser.add_argument("--split", default=str(REPO / "src/dgcc/tasks/splits/t2_sprint_heldout_v1.json"))
    parser.add_argument("--config", default="configs/p1_t2.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    manifest = load_selection_manifest(Path(args.selection_manifest), args.run_tag)
    split_path = Path(args.split)
    validate_sprint_split_path(split_path)
    # Everything above is provenance only. The claim precedes lock-protected split access.
    require_metric_lock(Path(args.lock) if args.lock else None, args.arm)
    split_sha = sha256_file(split_path)
    claim = acquire_claim(Path(args.claim), {"run_tag": args.run_tag, "arm": args.arm,
        "ckpt_sha256": manifest["ckpt_sha256"], "split_sha256": split_sha})
    record_access(SPRINT_ACCESS_LOG, "final_eval", run_tag=args.run_tag, arm=args.arm, split_sha256=split_sha)
    pairs = load_sprint_split(split_path)
    goals = [goal for _, goal in pairs for _ in range(2)]
    labels = [spec["goal_id"] for spec, _ in pairs for _ in range(2)]
    assert len(pairs) == 100 and len(goals) == len(labels) == 200
    run = build_run(args.config, args.seed, f"{args.run_tag}_sprint_heldout", args.device)
    run.agent.load_checkpoint(Path(manifest["selected_ckpt"]))
    run.val_goals, run.val_labels = goals, labels
    run.build_scene()
    started = time.perf_counter()
    result = run.deterministic_eval(episode_index_start=EPISODE_INDEX_START, record_raw=True)
    episodes = result["episodes"]
    assert len(episodes) == 200, "sprint evaluation must produce exactly 100 goals × 2 rows"
    raw_path = Path(args.out).with_suffix(".raw.json.gz")
    with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
        json.dump({"run_tag": args.run_tag, "episodes": episodes}, handle)
    claim_sha = sha256_file(Path(args.claim))
    probe_path = Path(args.out).with_suffix(".probe.h5")
    write_probe_h5(probe_path, episodes, ckpt_sha=manifest["ckpt_sha256"], split_sha=split_sha, claim_sha=claim_sha)
    probe_manifest_register(PROBE_MANIFEST, probe_path, {"production_goal": "G-EV", "run_tag": args.run_tag})
    for episode in episodes:
        for key in ("x_initial", "x_steps", "x_terminal"):
            episode.pop(key, None)
    atomic_publish(Path(args.out), {"generated_at": utc_now(), "run_tag": args.run_tag, "arm": args.arm,
        "ckpt_sha256": manifest["ckpt_sha256"], "split_sha256": split_sha, "claim_sha256": claim_sha,
        "selection_manifest": str(args.selection_manifest), "selection_rule": manifest["selection_rule"],
        "episode_index_start": EPISODE_INDEX_START, "wall_guard_k": WALL_GUARD_K,
        "wall_s": time.perf_counter() - started, "raw_artifact": str(raw_path), "probe_artifact": str(probe_path),
        "summary": {key: value for key, value in result.items() if key != "episodes"}, "episodes": episodes})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
