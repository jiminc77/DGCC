"""One-shot canonical sprint held-out evaluator (100 goals × 2 episodes)."""
from __future__ import annotations
import argparse, gzip, hashlib, importlib.util, json, sys, time
from pathlib import Path
from typing import Any
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO / "src"))
from dgcc.analysis.sprint_claims import (CANONICAL_SPLIT_PATH, CANONICAL_SPLIT_SHA256, CANONICAL_SUMMARY_KEYS, REPO_ROOT, SprintClaimError, acquire_claim, atomic_publish, canonicalize_episode_ids, consume_claim_and_load_split, json_file, probe_manifest_register, require_metric_lock, sha256_file, utc_now, validate_checkpoint_arm)
EPISODE_INDEX_START = 97_001; WALL_GUARD_K = 5
SPRINT_ACCESS_LOG = REPO / "outputs/metrics/t2_sprint_heldout_access.log"; PROBE_MANIFEST = REPO / "outputs/metrics/sprint_probe_manifest.json"
DRIVER_ONLY_RESULT_KEYS = frozenset({
    "nan_incidents_during_eval", "magnitude_incidents_during_eval",
    "wall_guard_k", "record_raw", "record_probe",
})

def _config_sha(path: str) -> str: return sha256_file(REPO / path)
def canonical_paths(run_tag: str, arm: str) -> tuple[Path, Path]:
    stem = f"p1_{arm.lower()}_sprint_heldout_{run_tag}"
    return REPO_ROOT / "outputs/metrics" / f"{stem}_claim.json", REPO_ROOT / "outputs/metrics" / f"{stem}.json"

def load_selection_manifest(path: Path, run_tag: str, arm: str, seed: int, config: str) -> tuple[dict[str, Any], str]:
    try: value, digest = json_file(path, "selection manifest")
    except SprintClaimError as exc: raise SprintClaimError("selection manifest is invalid") from exc
    required = {"run_tag", "arm", "seed", "task", "config_sha256", "ckpt_sha256", "val_rows", "selector_version", "selected_ckpt"}
    if not isinstance(value, dict) or set(value) != required: raise SprintClaimError("selection manifest schema is invalid")
    if value["run_tag"] != run_tag or value["arm"] != arm or value["seed"] != seed or value["task"] != "t2" or value["config_sha256"] != _config_sha(config): raise SprintClaimError("selection manifest identity does not match invocation")
    rows = value["val_rows"]
    if not isinstance(rows, list) or len(rows) != 50 or any(not isinstance(row, list) or len(row) != 2 for row in rows): raise SprintClaimError("selection manifest val_rows must be exactly 50×2")
    ckpt = Path(value["selected_ckpt"])
    if ckpt.is_symlink() or not ckpt.is_file() or not isinstance(value["ckpt_sha256"], str) or sha256_file(ckpt) != value["ckpt_sha256"]: raise SprintClaimError("selection checkpoint sha256 does not match disk")
    return value, digest

def _pairs(payload: dict[str, Any]):
    from dgcc.tasks.t2 import build_t2_goal
    return [(spec, build_t2_goal(spec)) for spec in payload["specs"]]

def write_probe_h5(path: Path, episodes: list[dict[str, Any]], *, ckpt_sha: str, split_sha: str, claim_sha: str) -> None:
    import h5py
    required = ("probe_p", "probe_u", "probe_x_before", "probe_x_after", "probe_goal_curve", "goal_id", "episode_id", "step_index", "truncated", "reseed_boundary", "eval_wall_guard")
    if len(episodes) != 200: raise SprintClaimError("probe requires exactly 200 episodes")
    validated = []
    for ep in episodes:
        missing = [key for key in required if key not in ep or ep[key] is None]
        if missing: raise SprintClaimError(f"probe episode missing frozen-schema fields: {', '.join(missing)}")
        p_raw, u_raw = np.asarray(ep["probe_p"]), np.asarray(ep["probe_u"])
        p, u = p_raw, np.asarray(ep["probe_u"], dtype=np.float64)
        before, after, goal = (np.asarray(ep[key], dtype=np.float64) for key in ("probe_x_before", "probe_x_after", "probe_goal_curve"))
        steps = np.asarray(ep["step_index"])
        scalar_flags = ("truncated", "reseed_boundary", "eval_wall_guard")
        if p_raw.dtype.kind not in "iu" or u_raw.dtype.kind != "f" or not isinstance(ep["episode_id"], (int, np.integer)) or not isinstance(ep["goal_id"], str) or not ep["goal_id"] or any(not isinstance(ep[key], (bool, np.bool_)) for key in scalar_flags):
            raise SprintClaimError("probe frozen schema dtypes are invalid")
        if p.ndim != 1 or not len(p) or u.shape != (len(p), 3) or before.shape != (len(p), 32, 3) or after.shape != (len(p), 32, 3) or goal.shape != (32, 3) or steps.shape != (len(p),):
            raise SprintClaimError("probe frozen schema shapes are invalid")
        if steps.dtype.kind not in "iu" or not all(np.all(np.isfinite(x)) for x in (p, u, before, after, goal)):
            raise SprintClaimError("probe values must be finite")
        validated.append((p, u, before, after, goal, steps))
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs.update(schema_version=3, ckpt_sha256=ckpt_sha, split_sha256=split_sha, claim_sha256=claim_sha)
        text = h5py.string_dtype()
        for index, (p, u, before, after, goal, steps) in enumerate(validated):
            group = h5.create_group(str(index))
            group.create_dataset("x_before", data=before)
            group.create_dataset("x_after", data=after)
            group.create_dataset("goal", data=goal)
            group.create_dataset("goal_id", data=str(episodes[index]["goal_id"]), dtype=text)
            group.create_dataset("p", data=p)
            group.create_dataset("u", data=u)
            group.create_dataset("episode_id", data=np.asarray(episodes[index]["episode_id"], dtype=np.int64))
            group.create_dataset("step_index", data=steps)
            for key in ("truncated", "reseed_boundary", "eval_wall_guard"):
                group.create_dataset(key, data=np.asarray(episodes[index][key], dtype=np.bool_))

def build_run(config: str, seed: int, run_tag: str, device: str):
    spec = importlib.util.spec_from_file_location("p1_train", REPO / "scripts/p1_train.py"); module = importlib.util.module_from_spec(spec); sys.modules["p1_train"] = module
    assert spec.loader is not None; spec.loader.exec_module(module)
    run = module.TrainingRun(argparse.Namespace(config=config, seed=seed, run_tag=run_tag, total_override=None, device=device)); run.config.setdefault("eval", {})["wall_guard_k"] = WALL_GUARD_K
    return run

def canonical_result_payload(*, run_tag: str, arm: str, seed: int, manifest: dict[str, Any], selection_manifest: str, selection_sha: str, claim_sha: str, result: dict[str, Any]) -> dict[str, Any]:
    """Build the audit-facing canonical result after frozen driver fields are removed."""
    summary = {key: value for key, value in result.items() if key not in DRIVER_ONLY_RESULT_KEYS | {"episodes"}}
    if set(summary) != CANONICAL_SUMMARY_KEYS:
        raise SprintClaimError("evaluation result does not match canonical summary schema")
    canonicalize_episode_ids(result["episodes"], EPISODE_INDEX_START)
    return {"generated_at": utc_now(), "run_tag": run_tag, "arm": arm, "seed": seed, "config_sha256": manifest["config_sha256"], "ckpt_sha256": manifest["ckpt_sha256"], "split_sha256": CANONICAL_SPLIT_SHA256, "claim_sha256": claim_sha, "selection_manifest": selection_manifest, "selection_manifest_sha256": selection_sha, "episode_namespace": EPISODE_INDEX_START, "selector_version": manifest["selector_version"], "val_rows": manifest["val_rows"], "summary": summary, "episodes": result["episodes"]}


def preclaim_payload_self_check() -> None:
    """Fail schema drift before acquiring the irreversible split authority."""
    synthetic = {key: 0.0 for key in CANONICAL_SUMMARY_KEYS}
    synthetic.update({"n_episodes": 0, "per_template_success": {}, "per_template_episodes": {}, "episodes": []})
    synthetic.update({key: 0 for key in DRIVER_ONLY_RESULT_KEYS})
    canonical_result_payload(
        run_tag="schema-check", arm="bb", seed=0,
        manifest={"config_sha256": "0", "ckpt_sha256": "0", "selector_version": "schema-check", "val_rows": []},
        selection_manifest="schema-check", selection_sha="0", claim_sha="0", result=synthetic,
    )
def publish_or_quarantine(path: Path, *, result: dict[str, Any], payload: dict[str, Any]) -> None:
    """Atomically publish or durably retain the complete rejected driver result."""
    try:
        atomic_publish(path, payload)
    except Exception:
        atomic_publish(path.with_name(f"{path.stem}_quarantine.json"), result)
        raise


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--run-tag", required=True); p.add_argument("--arm", required=True); p.add_argument("--selection-manifest", required=True); p.add_argument("--claim", required=True); p.add_argument("--out", required=True); p.add_argument("--lock"); p.add_argument("--config", default="configs/p1_t2.yaml"); p.add_argument("--seed", type=int, required=True); p.add_argument("--device", default="cuda"); args = p.parse_args()
    expected_claim, expected_out = canonical_paths(args.run_tag, args.arm)
    if Path(args.claim).is_symlink() or Path(args.out).is_symlink() or Path(args.claim).absolute() != expected_claim.absolute() or Path(args.out).absolute() != expected_out.absolute(): raise SprintClaimError("claim and out paths must be canonical non-symlink run-identity paths")
    manifest, selection_sha = load_selection_manifest(Path(args.selection_manifest), args.run_tag, args.arm, args.seed, args.config); require_metric_lock(Path(args.lock) if args.lock else None, args.arm)
    validate_checkpoint_arm(Path(manifest["selected_ckpt"]), args.arm)
    selection_path = Path(args.selection_manifest)
    if selection_path.is_symlink():
        raise SprintClaimError("selection manifest must not be a symlink")
    if not selection_path.is_absolute():
        raise SprintClaimError("selection manifest path must be absolute")
    preclaim_payload_self_check()
    capability = acquire_claim(expected_claim, {"run_tag":args.run_tag,"arm":args.arm,"ckpt_sha256":manifest["ckpt_sha256"],"split_sha256":CANONICAL_SPLIT_SHA256,"seed":args.seed,"config_sha256":manifest["config_sha256"],"selection_manifest":str(selection_path),"selection_manifest_sha256":selection_sha,"episode_namespace":EPISODE_INDEX_START,"n_goals":100})
    payload = consume_claim_and_load_split(capability, CANONICAL_SPLIT_PATH, access_log=SPRINT_ACCESS_LOG); pairs = _pairs(payload); goals=[g for _,g in pairs for _ in range(2)]; labels=[s["goal_id"] for s,_ in pairs for _ in range(2)]
    run=build_run(args.config,args.seed,f"{args.run_tag}_sprint_heldout",args.device); run.agent.load_checkpoint(Path(manifest["selected_ckpt"])); run.val_goals,run.val_labels=goals,labels; run.build_scene(); started=time.perf_counter(); result=run.deterministic_eval(episode_index_start=EPISODE_INDEX_START,record_raw=True,record_probe=True); episodes=result["episodes"]
    if len(episodes)!=200: raise SprintClaimError("sprint evaluation must produce 200 episodes")
    canonicalize_episode_ids(episodes, EPISODE_INDEX_START)
    raw_path=expected_out.with_suffix(".raw.json.gz"); gzip.open(raw_path,"wt",encoding="utf-8").write(json.dumps({"run_tag":args.run_tag,"episodes":episodes}))
    claim_sha=sha256_file(expected_claim); probe_path=expected_out.with_suffix(".probe.h5"); write_probe_h5(probe_path,episodes,ckpt_sha=manifest["ckpt_sha256"],split_sha=CANONICAL_SPLIT_SHA256,claim_sha=claim_sha); probe_manifest_register(PROBE_MANIFEST,probe_path,{"production_goal":"G-EV","run_tag":args.run_tag})
    for ep in episodes:
        for key in ("x_initial","x_steps","x_terminal","probe_p","probe_u"): ep.pop(key,None)
    try:
        canonical = canonical_result_payload(run_tag=args.run_tag, arm=args.arm, seed=args.seed, manifest=manifest, selection_manifest=str(Path(args.selection_manifest).resolve()), selection_sha=selection_sha, claim_sha=claim_sha, result=result)
    except Exception:
        atomic_publish(expected_out.with_name(f"{expected_out.stem}_quarantine.json"), result)
        raise
    publish_or_quarantine(expected_out, result=result, payload=canonical)
    return 0
if __name__ == "__main__": raise SystemExit(main())
