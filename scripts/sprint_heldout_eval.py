"""One-shot canonical sprint held-out evaluator (100 goals × 2 episodes)."""
from __future__ import annotations
import argparse, gzip, hashlib, importlib.util, json, sys, time
from pathlib import Path
from typing import Any
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO / "src"))
from dgcc.analysis.sprint_claims import (CANONICAL_SPLIT_PATH, CANONICAL_SPLIT_SHA256, SprintClaimError, acquire_claim, atomic_publish, consume_claim_and_load_split, probe_manifest_register, require_metric_lock, sha256_file, utc_now)
EPISODE_INDEX_START = 97_001; WALL_GUARD_K = 5
SPRINT_ACCESS_LOG = Path("outputs/metrics/t2_sprint_heldout_access.log"); PROBE_MANIFEST = Path("outputs/metrics/sprint_probe_manifest.json")

def _config_sha(path: str) -> str: return sha256_file(REPO / path)
def canonical_paths(run_tag: str, arm: str) -> tuple[Path, Path]:
    stem = f"p1_{arm.lower()}_sprint_heldout_{run_tag}"
    return Path("outputs/metrics") / f"{stem}_claim.json", Path("outputs/metrics") / f"{stem}.json"

def load_selection_manifest(path: Path, run_tag: str, arm: str, seed: int, config: str) -> dict[str, Any]:
    try: value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc: raise SprintClaimError("selection manifest is invalid") from exc
    required = {"run_tag", "arm", "seed", "task", "config_sha256", "ckpt_sha256", "val_rows", "selector_version", "selected_ckpt"}
    if not isinstance(value, dict) or set(value) != required: raise SprintClaimError("selection manifest schema is invalid")
    if value["run_tag"] != run_tag or value["arm"] != arm or value["seed"] != seed or value["task"] != "t2" or value["config_sha256"] != _config_sha(config): raise SprintClaimError("selection manifest identity does not match invocation")
    rows = value["val_rows"]
    if not isinstance(rows, list) or len(rows) != 50 or any(not isinstance(row, list) or len(row) != 2 for row in rows): raise SprintClaimError("selection manifest val_rows must be exactly 50×2")
    ckpt = Path(value["selected_ckpt"])
    if not ckpt.is_file() or not isinstance(value["ckpt_sha256"], str) or sha256_file(ckpt) != value["ckpt_sha256"]: raise SprintClaimError("selection checkpoint sha256 does not match disk")
    return value

def _pairs(payload: dict[str, Any]):
    from dgcc.tasks.t2 import build_t2_goal
    return [(spec, build_t2_goal(spec)) for spec in payload["specs"]]

def write_probe_h5(path: Path, episodes: list[dict[str, Any]], *, ckpt_sha: str, split_sha: str, claim_sha: str) -> None:
    import h5py
    required = ("x_initial", "x_terminal", "probe_p", "probe_u", "goal_label", "reseed_boundary", "eval_wall_guard")
    for ep in episodes:
        missing = [key for key in required if key not in ep or ep[key] is None]
        if missing: raise SprintClaimError(f"probe episode missing raw fields: {', '.join(missing)}")
        if not ep["probe_p"] or not ep["probe_u"] or len(ep["probe_p"]) != len(ep["probe_u"]): raise SprintClaimError("probe requires non-empty aligned per-step p/u")
        if np.asarray(ep["x_initial"]).shape != np.asarray(ep["x_terminal"]).shape or np.asarray(ep["x_initial"]).ndim != 2: raise SprintClaimError("probe centerline shape is invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs.update(schema_version=1, ckpt_sha256=ckpt_sha, split_sha256=split_sha, claim_sha256=claim_sha)
        text = h5py.string_dtype()
        h5.create_dataset("x_before", data=np.asarray([e["x_initial"] for e in episodes]))
        h5.create_dataset("x_after", data=np.asarray([e["x_terminal"] for e in episodes]))
        p_group = h5.create_group("p")
        u_group = h5.create_group("u")
        for index, episode in enumerate(episodes):
            p_group.create_dataset(str(index), data=np.asarray(episode["probe_p"]))
            u_group.create_dataset(str(index), data=np.asarray(episode["probe_u"]))
        h5.create_dataset("goal_id", data=np.asarray([str(e["goal_label"]) for e in episodes], dtype=text))
        flags = h5.create_group("flags")
        flags.create_dataset("reseed_boundary", data=np.asarray([e["reseed_boundary"] for e in episodes], dtype=bool))
        flags.create_dataset("eval_wall_guard", data=np.asarray([e["eval_wall_guard"] for e in episodes], dtype=bool))

def build_run(config: str, seed: int, run_tag: str, device: str):
    spec = importlib.util.spec_from_file_location("p1_train", REPO / "scripts/p1_train.py"); module = importlib.util.module_from_spec(spec); sys.modules["p1_train"] = module
    assert spec.loader is not None; spec.loader.exec_module(module)
    run = module.TrainingRun(argparse.Namespace(config=config, seed=seed, run_tag=run_tag, total_override=None, device=device)); run.config.setdefault("eval", {})["wall_guard_k"] = WALL_GUARD_K
    return run

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--run-tag", required=True); p.add_argument("--arm", required=True); p.add_argument("--selection-manifest", required=True); p.add_argument("--claim", required=True); p.add_argument("--out", required=True); p.add_argument("--lock"); p.add_argument("--config", default="configs/p1_t2.yaml"); p.add_argument("--seed", type=int, required=True); p.add_argument("--device", default="cuda"); args = p.parse_args()
    expected_claim, expected_out = canonical_paths(args.run_tag, args.arm)
    if Path(args.claim) != expected_claim or Path(args.out) != expected_out: raise SprintClaimError("claim and out paths must be canonical run-identity paths")
    manifest = load_selection_manifest(Path(args.selection_manifest), args.run_tag, args.arm, args.seed, args.config); require_metric_lock(Path(args.lock) if args.lock else None, args.arm)
    acquire_claim(expected_claim, {"run_tag":args.run_tag,"arm":args.arm,"ckpt_sha256":manifest["ckpt_sha256"],"split_sha256":CANONICAL_SPLIT_SHA256,"seed":args.seed,"config_sha256":manifest["config_sha256"],"selection_manifest":str(args.selection_manifest)})
    payload = consume_claim_and_load_split(expected_claim, CANONICAL_SPLIT_PATH, access_log=SPRINT_ACCESS_LOG); pairs = _pairs(payload); goals=[g for _,g in pairs for _ in range(2)]; labels=[s["goal_id"] for s,_ in pairs for _ in range(2)]
    run=build_run(args.config,args.seed,f"{args.run_tag}_sprint_heldout",args.device); run.agent.load_checkpoint(Path(manifest["selected_ckpt"])); run.val_goals,run.val_labels=goals,labels; run.build_scene(); started=time.perf_counter(); result=run.deterministic_eval(episode_index_start=EPISODE_INDEX_START,record_raw=True,record_probe=True); episodes=result["episodes"]
    if len(episodes)!=200: raise SprintClaimError("sprint evaluation must produce 200 episodes")
    raw_path=expected_out.with_suffix(".raw.json.gz"); gzip.open(raw_path,"wt",encoding="utf-8").write(json.dumps({"run_tag":args.run_tag,"episodes":episodes}))
    claim_sha=sha256_file(expected_claim); probe_path=expected_out.with_suffix(".probe.h5"); write_probe_h5(probe_path,episodes,ckpt_sha=manifest["ckpt_sha256"],split_sha=CANONICAL_SPLIT_SHA256,claim_sha=claim_sha); probe_manifest_register(PROBE_MANIFEST,probe_path,{"production_goal":"G-EV","run_tag":args.run_tag})
    for ep in episodes:
        for key in ("x_initial","x_steps","x_terminal","probe_p","probe_u"): ep.pop(key,None)
    atomic_publish(expected_out,{"generated_at":utc_now(),"run_tag":args.run_tag,"arm":args.arm,"seed":args.seed,"config_sha256":manifest["config_sha256"],"ckpt_sha256":manifest["ckpt_sha256"],"split_sha256":CANONICAL_SPLIT_SHA256,"claim_sha256":claim_sha,"selection_manifest":str(args.selection_manifest),"selector_version":manifest["selector_version"],"val_rows":manifest["val_rows"],"summary":{k:v for k,v in result.items() if k!="episodes"},"episodes":episodes})
    return 0
if __name__ == "__main__": raise SystemExit(main())
