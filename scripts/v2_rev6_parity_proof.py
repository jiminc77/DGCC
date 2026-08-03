#!/usr/bin/env python3
"""Rev 6 byte-identity proof (pilot design principle D2).

Runs a short, fully deterministic primitive sequence through the production
batched path and dumps every observable at full float precision.  The SAME file
is run against the pre-Rev-6 tree and the Rev-6 tree at ``n_segments = 32`` /
``rope_mass_total = 0.032 kg``; the two JSON artifacts must be byte-identical.

CPU backend only.  GPU tail events are explicitly not reproducible in this
codebase (the AT-1H verdict is rate-based for that reason), so a byte
comparison is only meaningful on the determinism-pinned backend.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--primitives", type=int, default=4)
    parser.add_argument("--n-envs", type=int, default=2)
    args = parser.parse_args()

    import genesis as gs  # noqa: F401  (import side effects match the battery)

    from dgcc.envs.dlolab import DLOLabEnv, ensure_genesis_initialized, mapped_parameters
    from dgcc.tasks.domain import p1_rope_params

    ensure_genesis_initialized(0)
    params = p1_rope_params()

    records: list[dict[str, object]] = []
    for episode in range(args.episodes):
        env = DLOLabEnv(
            n_envs=args.n_envs,
            dt=1.0e-3,
            substeps=5,
            rod_damping=10.0,
            rod_angular_damping=5.0,
            initial_settle_steps=0,
            reset_settle_max_steps=10_000,
            move_v_max=0.15,
            move_hold_max_steps=2000,
            grasp_realism=True,
            at1h_counters=True,
        )
        info = env.reset(params, init_shape="s_curve", seed=1_000 + episode)
        rng = np.random.default_rng([4242, episode])
        for k in range(args.primitives):
            radius = 0.15 * float(np.sqrt(rng.uniform()))
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            delta = np.tile(
                np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0]),
                (args.n_envs, 1),
            )
            p = rng.integers(0, 32, size=args.n_envs)
            lift = ["low" if (k + i) % 2 == 0 else "high" for i in range(args.n_envs)]
            out = env.step_primitive_batch(
                p,
                delta,
                lift,
                vel_threshold=1.0e-3,
                max_steps=10_000,
                rng=np.random.default_rng([99_000, episode, k]),
            )
            at1h = out["info"]["at1h"]
            records.append(
                {
                    "episode": episode,
                    "primitive": k,
                    "p": p.tolist(),
                    "p_actual": np.asarray(out["info"]["p_actual"]).tolist(),
                    "grasp_success": np.asarray(out["grasp_success"]).tolist(),
                    "settle_steps": np.asarray(out["settle_steps"]).tolist(),
                    "X_after": np.asarray(out["X_after"], dtype=float).tolist(),
                    "gripper_target": np.asarray(
                        out["info"]["gripper_target"], dtype=float
                    ).tolist(),
                    "n_waypoint_steps": list(out["info"]["n_waypoint_steps"]),
                    "hold_steps_used": int(out["info"]["hold_steps_used"]),
                    "approach_steps": int(out["info"]["approach_steps"]),
                    "approach_dwell_steps": int(out["info"]["approach_dwell_steps"]),
                    "at1h_v_peak": np.asarray(at1h["v_peak"], dtype=float).tolist(),
                    "at1h_strain_peak": np.asarray(at1h["strain_peak"], dtype=float).tolist(),
                    "at1h_ke_peak": np.asarray(at1h["ke_peak"], dtype=float).tolist(),
                    "at1h_grav_pe": np.asarray(at1h["grav_pe"], dtype=float).tolist(),
                    "at1h_ke_over_pe": np.asarray(at1h["ke_over_pe"], dtype=float).tolist(),
                    "at1h_samples": int(at1h["samples"]),
                }
            )
        del env

    payload = {
        "proof": "rev6-byte-identity",
        "n_segments": int(params.n_segments),
        "mapped_parameters": mapped_parameters(params),
        "rope_length_m": float(params.length_m),
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
