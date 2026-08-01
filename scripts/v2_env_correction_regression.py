#!/usr/bin/env python3
"""Divergence-mode regression: does the correction remove the C3 blow-up?

Directly verifies design §1.2/§5.5: the impulsive-primitive divergence mode
(velocity cap WITHOUT hold — condition C3) and its removal by the recommended
quasi-static bundle.  Four conditions replay the SAME stored initial state
and the SAME reproduced policy actions:

  current      legacy move_step_size=0.03,  hold=0   (production behavior)
  slow_only    legacy move_step_size=1.5e-4, hold=0  (the forbidden middle
               state: velocity cap alone — C3's precondition)
  slow_hold    legacy move_step_size=1.5e-4, hold=300 (C6-style)
  recommended  quasi-static bundle (R1-R5: v_max=0.15, hold-until-quiescent,
               lowering waypoint, horizontal δ)

Replay material: V2-D1M seed-0 final recorded eval (record_raw) episodes
54 (s family, goal t2-0260 — the divergence case the design mandates) and
80 (u family, goal t2-0441).  The stored x_initial/x_steps rows are read
from the preserved tournament artifact (read-only) and the four per-episode
policy actions are reproduced by running the preserved ckpt_0300032
deterministically on the recorded pre-step states — the same construction
the design probe used.  The episode-54 signature (δz ≈ −0.15 in 3 of 4
actions) is asserted so the replayed action set provably matches §5.5.

Pass: the recommended condition converges every settle and produces zero
arc-length covenant events (|L/L0−1| > 0.05).  Other conditions are
reported for contrast (their failures are expected, not gating).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FORBIDDEN_PATH_TOKENS = ("heldout", "held-out", "patch_eval", "patching_probe")
PRIMITIVES = 4
ARCLEN_COVENANT = 0.05
EPISODES = (54, 80)
DELTA_Z_SIGNATURE_EPISODE = 54
CONDITIONS: dict[str, dict[str, Any]] = {
    "current": {"move_step_size": 0.03, "move_hold_steps": 0},
    "slow_only": {"move_step_size": 1.5e-4, "move_hold_steps": 0},
    "slow_hold": {"move_step_size": 1.5e-4, "move_hold_steps": 300},
    "recommended": {"move_v_max": 0.15, "move_hold_max_steps": 2000},
}


def permitted_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    lowered = str(path).lower()
    for token in FORBIDDEN_PATH_TOKENS:
        if token in lowered:
            raise argparse.ArgumentTypeError(f"forbidden path scope: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def init_genesis(backend: str) -> dict[str, Any]:
    import torch

    import genesis as gs

    if backend == "cpu":
        if torch.cuda.is_available():
            raise RuntimeError("CPU run unexpectedly sees a CUDA device")
        if not getattr(gs, "_initialized", False):
            gs.init(seed=0, precision="32", logging_level="warning", backend=gs.cpu)
        return {"backend": "cpu", "device": "cpu"}
    if not torch.cuda.is_available():
        raise RuntimeError("GPU requested but no CUDA device is visible")
    if not getattr(gs, "_initialized", False):
        gs.init(seed=0, precision="32", logging_level="warning", backend=gs.gpu)
    return {"backend": "gpu", "device": torch.cuda.get_device_name(0)}


def load_replay_material(eval_path: Path, checkpoint: Path) -> dict[int, dict[str, Any]]:
    """Recorded states + reproduced deterministic policy actions per episode."""
    from dgcc.goals.dual_goal import goal_curve
    from dgcc.rl.sprint_arms import create_sprint_agent
    from dgcc.rl.td3 import TD3Config
    from dgcc.tasks.domain import p1_rope_params
    from dgcc.tasks.t2 import build_t2_goal

    with gzip.open(eval_path) as handle:
        record = json.load(handle)
    agent = create_sprint_agent("v2-d1m", TD3Config(policy_delay=2), device="cpu")
    agent.load_checkpoint(checkpoint, eval_only=True)
    for module_name in ("encoder", "critic", "actor"):
        getattr(agent, module_name).eval()

    length = p1_rope_params().length_m
    material: dict[int, dict[str, Any]] = {}
    for index in EPISODES:
        episode = record["episodes"][index]
        goal = build_t2_goal(episode["goal_label"])
        curve = goal_curve(goal, length)
        x_initial = np.asarray(episode["x_initial"], dtype=float)
        x_steps = np.asarray(episode["x_steps"], dtype=float)
        states = [x_initial] + [x_steps[k] for k in range(PRIMITIVES - 1)]
        actions = []
        for state in states:
            p, delta, lift = agent.select_actions(
                state[None, ...],
                curve[None, ...],
                step=int(record["transitions"]),
                total_budget=int(record["transitions"]),
                rng=np.random.default_rng(0),
                deterministic=True,
            )
            actions.append(
                {
                    "p": int(np.asarray(p).ravel()[0]),
                    "delta": [float(v) for v in np.asarray(delta).reshape(3)],
                    "lift": str(np.asarray(lift).ravel()[0]),
                }
            )
        material[index] = {
            "goal_id": str(episode["goal_label"]["goal_id"]),
            "family": str(episode["goal_label"]["family"]),
            "x_initial": x_initial,
            "actions": actions,
        }
    signature = sum(
        1
        for action in material[DELTA_Z_SIGNATURE_EPISODE]["actions"]
        if action["delta"][2] <= -0.14
    )
    material["episode54_delta_z_signature"] = signature
    if signature < 3:
        raise RuntimeError(
            "episode-54 replay signature mismatch: expected δz ≈ −0.15 in >=3 "
            f"of {PRIMITIVES} actions, found {signature} — the reproduced "
            "action set does not match design §5.5"
        )
    return material


def build_env(condition: dict[str, Any], backend_kwargs: dict[str, Any]):
    from dgcc.envs.dlolab import DLOLabEnv

    return DLOLabEnv(
        n_envs=1,
        dt=1.0e-3,
        substeps=5,
        rod_damping=10.0,
        rod_angular_damping=5.0,
        initial_settle_steps=0,
        reset_settle_max_steps=10_000,
        grasp_realism=False,  # deterministic replay: no grasp noise/failure
        **condition,
    )


def run_condition(name: str, condition: dict[str, Any], material: dict[int, dict[str, Any]]) -> dict[str, Any]:
    from dgcc.tasks.domain import p1_rope_params

    params = p1_rope_params()
    episodes_out: list[dict[str, Any]] = []
    for index in EPISODES:
        item = material[index]
        env = build_env(condition, {})
        env.reset(params, init_shape="straight", seed=31_337)
        env.light_reset(item["x_initial"][None, ...])
        rows: list[dict[str, Any]] = []
        failure: str | None = None
        for k, action in enumerate(item["actions"]):
            try:
                out = env.step_primitive_batch(
                    np.asarray([action["p"]], dtype=int),
                    np.asarray([action["delta"]], dtype=float)[None, ...].reshape(1, 3),
                    [action["lift"]],
                    vel_threshold=1.0e-3,
                    max_steps=10_000,
                    rng=np.random.default_rng(777),
                )
            except (FloatingPointError, ValueError, RuntimeError) as exc:
                failure = f"{type(exc).__name__}: {exc}"
                break
            raw = np.asarray(env.get_centerline_raw_batch(), dtype=float)[0]
            edges = np.linalg.norm(raw[1:] - raw[:-1], axis=-1)
            arclen = float(edges.sum())
            rows.append(
                {
                    "primitive": k,
                    "settle_converged": bool(out["info"]["settle_converged"][0]),
                    "settle_steps": int(out["settle_steps"][0]),
                    "arclen_after_settle": arclen,
                    "arclen_rel_dev": abs(arclen / float(params.length_m) - 1.0),
                    "max_node_speed": float(np.max(env.max_node_speed_batch())),
                    "hold_steps_used": int(out["info"].get("hold_steps_used", 0)),
                    "hold_converged": out["info"].get("hold_converged"),
                }
            )
        del env
        settles_converged = all(r["settle_converged"] for r in rows) and failure is None
        covenant_events = sum(r["arclen_rel_dev"] > ARCLEN_COVENANT for r in rows)
        episodes_out.append(
            {
                "episode": index,
                "goal_id": item["goal_id"],
                "family": item["family"],
                "primitives": rows,
                "failure": failure,
                "all_settles_converged": settles_converged,
                "arclen_covenant_events": int(covenant_events),
            }
        )
    return {
        "condition": name,
        "parameters": condition,
        "episodes": episodes_out,
        "all_settles_converged": all(e["all_settles_converged"] for e in episodes_out),
        "arclen_covenant_events": int(sum(e["arclen_covenant_events"] for e in episodes_out)),
        "failures": [e["failure"] for e in episodes_out if e["failure"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-record", type=permitted_path, required=True)
    parser.add_argument("--checkpoint", type=permitted_path, required=True)
    parser.add_argument("--out", type=permitted_path, required=True)
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    args = parser.parse_args()

    backend_info = init_genesis(args.backend)
    material = load_replay_material(args.eval_record, args.checkpoint)

    conditions = [
        run_condition(name, params, material) for name, params in CONDITIONS.items()
    ]
    recommended = next(c for c in conditions if c["condition"] == "recommended")
    result = {
        "schema_version": 1,
        "backend": backend_info,
        "eval_record": str(args.eval_record),
        "eval_record_sha256": sha256_file(args.eval_record),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "episode54_delta_z_signature": material["episode54_delta_z_signature"],
        "replayed_actions": {
            str(index): material[index]["actions"] for index in EPISODES
        },
        "conditions": conditions,
        "pass": recommended["all_settles_converged"]
        and recommended["arclen_covenant_events"] == 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "episode54_delta_z_signature": result["episode54_delta_z_signature"],
                "conditions": [
                    {
                        "condition": c["condition"],
                        "all_settles_converged": c["all_settles_converged"],
                        "arclen_covenant_events": c["arclen_covenant_events"],
                        "failures": c["failures"],
                    }
                    for c in conditions
                ],
                "pass": result["pass"],
            },
            indent=2,
        )
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
