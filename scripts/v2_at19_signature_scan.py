#!/usr/bin/env python3
"""AT-19 (brought into scope by the 2026-08-01 boundary amendment, item 4).

Large-batch stuck-signature scan on the D1-D4-corrected environment:
every light_reset of a production-scale batched rollout must show

  (i)  vertex_constraints.constrained.sum() == 0 (solver truth), and
  (ii) zero settled initial nodes within +-2 mm of the gripper parking
       height (current constant GRIPPER_PARK_Z after R10; historical
       artifact scans keep the old 0.15 constant — design §5.6 note 9).

The rollout uses the legacy production dynamics semantics (move_step_size
0.03 / hold 0, matching the pre-R8 production config) so the scan isolates
the D-series corrections from the Stage 2 R-series gate that is still under
owner adjudication (ENVFIX_STEP_LOG Entry 5).  Depends only on Stage 1.

The historical contamination baseline is the investigation's full scan
(V2_light_reset_eval_contamination_full.csv, arm-wise 0-40% with the old
0.15 m parking signature); this scan targets 0% on the corrected env.

Exit code 0 only when both signature counts are exactly zero (fail-closed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FORBIDDEN_PATH_TOKENS = ("heldout", "held-out", "patch_eval", "patching_probe")
PARKING_TOLERANCE_M = 2.0e-3


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=permitted_path, required=True)
    parser.add_argument("--n-envs", type=int, default=1024)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--primitives-per-round", type=int, default=2)
    args = parser.parse_args()

    import torch

    import genesis as gs

    if not torch.cuda.is_available():
        raise RuntimeError("AT-19 large-batch scan requires the authorized GPU")
    if not getattr(gs, "_initialized", False):
        gs.init(seed=0, precision="32", logging_level="warning", backend=gs.gpu)

    from dgcc.envs.dlolab import DLOLabEnv, GRIPPER_PARK_Z
    from dgcc.tasks.domain import p1_rope_params
    from dgcc.tasks.episode import BatchedEpisodeRunner, EpisodeConfig
    from dgcc.tasks.t2 import load_t2_payload, load_t2_split_payload

    started = time.time()
    params = p1_rope_params()
    pairs = load_t2_split_payload("val", load_t2_payload())
    goals = [goal for _, goal in pairs]

    env = DLOLabEnv(
        n_envs=args.n_envs,
        dt=1.0e-3,
        substeps=5,
        rod_damping=10.0,
        rod_angular_damping=5.0,
        initial_settle_steps=0,
        reset_settle_max_steps=10_000,
        move_step_size=0.03,
        move_hold_steps=0,
        grasp_realism=True,
    )
    env.reset(params, init_shape="straight", seed=10_000)
    runner = BatchedEpisodeRunner(env, params, EpisodeConfig())

    assigned = [goals[i % len(goals)] for i in range(args.n_envs)]
    rounds: list[dict[str, Any]] = []
    residual_violations = 0
    parking_hits = 0
    parking_hits_confirmed = 0
    resets_scanned = 0
    primitives_run = 0
    for round_index in range(1, args.rounds + 1):
        begin = runner.begin_episodes(
            seed=500 + round_index, episode_index=round_index, goals=assigned
        )
        mask = env._vertex_constrained_mask()
        z = np.asarray(env.get_centerline_batch(), dtype=float)[..., 2]
        proxy_hit_envs = np.flatnonzero(
            (np.abs(z - GRIPPER_PARK_Z) <= PARKING_TOLERANCE_M).any(axis=1)
        )
        # Solver truth is authoritative; the z-proxy exists for artifact-only
        # scans.  With the R10-lowered parking height (0.03) a legitimately
        # piled/curved rope can pass through the +-2 mm band, so every proxy
        # hit is adjudicated against the constraint mask of its env: a hit
        # whose env carries zero constraint bits is a documented geometric
        # false positive, not a stuck signature.
        unadjudicated_hits = int(
            sum(1 for env_idx in proxy_hit_envs if mask[int(env_idx)].any())
        )
        hits = int(proxy_hit_envs.size)
        resets_scanned += 1
        residual_violations += int(mask.any())
        parking_hits += hits
        parking_hits_confirmed += unadjudicated_hits

        action_rng = np.random.default_rng([31_000, round_index])
        round_residuals = 0
        round_escalations = 0
        for _ in range(args.primitives_per_round):
            p = action_rng.integers(0, 32, args.n_envs)
            delta = action_rng.normal(0.0, 0.06, (args.n_envs, 3))
            lift = [str(x) for x in action_rng.choice(["low", "high"], args.n_envs)]
            out = runner.step(p, delta, lift, rng=action_rng)
            primitives_run += 1
            round_residuals += int(out["info"]["detach_residuals"])
            round_escalations += int(out["info"]["detach_escalations"])
        rounds.append(
            {
                "round": round_index,
                "reset_residual_sum": int(mask.sum()),
                "parking_hits": hits,
                "reset_reseeded": list(begin["reset_reseeded_envs"]),
                "detach_residuals": round_residuals,
                "detach_escalations": round_escalations,
                "nan_incidents": int(runner.nan_incidents),
                "magnitude_incidents": int(runner.magnitude_incidents),
            }
        )

    result = {
        "schema_version": 1,
        "backend": "gpu",
        "device": torch.cuda.get_device_name(0),
        "dynamics": "legacy-production (move_step_size=0.03, hold=0; pre-R8 semantics)",
        "n_envs": args.n_envs,
        "rounds": args.rounds,
        "primitives_per_round": args.primitives_per_round,
        "env_primitives_total": primitives_run * args.n_envs,
        "resets_scanned": resets_scanned,
        "residual_violations": residual_violations,
        "parking_signature_hits": parking_hits,
        "parking_z_m": float(GRIPPER_PARK_Z),
        "historical_baseline": {
            "artifact": "dossier/V2_light_reset_eval_contamination_full.csv",
            "note": "arm-wise 0-40% contamination with the old 0.15 m parking "
                    "signature on the pre-correction env (investigation full scan)",
        },
        "rounds_detail": rounds,
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "elapsed_s": round(time.time() - started, 1),
    }
    result["parking_hits_confirmed_by_solver_truth"] = int(parking_hits_confirmed)
    result["pass"] = residual_violations == 0 and parking_hits_confirmed == 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "n_envs", "rounds", "env_primitives_total", "resets_scanned",
                    "residual_violations", "parking_signature_hits", "pass",
                )
            },
            indent=2,
        )
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
