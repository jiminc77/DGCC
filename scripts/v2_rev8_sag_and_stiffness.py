#!/usr/bin/env python3
"""Rev 8 measurement (NO repair, NO threshold change, NO adoption proposal).

Two questions from the orchestrator, both measurement-only:

  (1) MECHANISM.  Why does the low-lift settle tail grow at 40 g?
      Hypothesis under test: same stiffness + 25% more mass -> more static
      sag -> at lift=low (2 cm) the rope drags on the plane.
      Measured as: the resting configuration on the plane, and the HELD
      configuration when a mid node is raised to the low (0.02 m) and high
      (0.15 m) lift heights -- lowest node, how many nodes are within contact
      distance of the plane, and the hang depth below the gripper.

  (2) STIFFNESS-MATCHED PREDICTION.  If EI/EA/GJ are raised by the same 1.25
      the mass was, the dimensionless sag ratio is preserved.  Does the
      low-lift settle tail come back to the 32 g level?  Small sample only.

Both arms override the module constants AT RUNTIME; repository source is not
modified and nothing here is adopted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def build_params(mass_kg: float):
    from dgcc.envs.base import RopeParams

    return RopeParams(
        length_m=1.0,
        n_segments=32,
        bend_stiffness=1.0,
        twist_stiffness=1.0,
        friction=1.0,
        radius=0.005,
        rope_mass_total_kg=mass_kg,
    )


def apply_stiffness_scale(scale: float) -> dict[str, float]:
    import dgcc.envs.dlolab as dlolab

    if not hasattr(dlolab, "_REV8_BASE"):
        dlolab._REV8_BASE = (dlolab.STRETCH_BASE, dlolab.BEND_BASE, dlolab.TWIST_BASE)
    k0, e0, g0 = dlolab._REV8_BASE
    dlolab.STRETCH_BASE = k0 * scale
    dlolab.BEND_BASE = e0 * scale
    dlolab.TWIST_BASE = g0 * scale
    return {
        "stretch": dlolab.STRETCH_BASE,
        "bend": dlolab.BEND_BASE,
        "twist": dlolab.TWIST_BASE,
    }


def sag_cell(mass_kg: float, scale: float, shapes: list[str]) -> list[dict]:
    """Resting sag and held sag at both lift heights."""

    from v2_env_correction_acceptance import build_probe_env

    from dgcc.envs.dlolab import LIFT_HEIGHTS

    stiff = apply_stiffness_scale(scale)
    params = build_params(mass_kg)
    radius = float(params.radius)
    out: list[dict] = []

    for shape in shapes:
        env = build_probe_env(1)
        env.reset(params, init_shape=shape, seed=1234)
        rest = np.asarray(env._raw_batch(), dtype=float)[0]
        rec = {
            "mass_kg": mass_kg,
            "stiffness_scale": scale,
            "stiffness": stiff,
            "init_shape": shape,
            "rest_min_z": float(rest[:, 2].min()),
            "rest_mean_z": float(rest[:, 2].mean()),
            "rest_contact_nodes": int((rest[:, 2] <= radius).sum()),
            "rest_below_plane_nodes": int((rest[:, 2] < 0.0).sum()),
        }
        # Held configuration: grasp the middle node, raise to each lift height,
        # let it settle, and look at what hangs below.
        for lift, height in LIFT_HEIGHTS.items():
            env.reset(params, init_shape=shape, seed=1234)
            node = env._n_vertices() // 2
            if not env.grasp(node):
                rec[f"held_{lift}"] = None
                continue
            target = env._gripper_positions().copy()
            target[:, 2] = height
            # walk up under the same speed cap the primitive uses
            start = env._gripper_positions().copy()
            steps = max(20, int(np.ceil(abs(height - start[0, 2]) / env._move_step)))
            for a in np.linspace(1.0 / steps, 1.0, steps):
                env._set_gripper_positions((1.0 - a) * start + a * target)
                env._step_scene()
            for _ in range(2000):
                env._set_gripper_positions(target)
                env._step_scene()
                if float(env.max_node_speed_batch().max()) < 0.05:
                    break
            held = np.asarray(env._raw_batch(), dtype=float)[0]
            rec[f"held_{lift}"] = {
                "gripper_z": float(target[0, 2]),
                "min_z": float(held[:, 2].min()),
                "contact_nodes": int((held[:, 2] <= radius).sum()),
                "below_plane_nodes": int((held[:, 2] < 0.0).sum()),
                "hang_depth": float(target[0, 2] - held[:, 2].min()),
                "max_edge_strain": float(env._max_edge_strain_batch().max()),
            }
        out.append(rec)
        del env
    return out


def sample_cell(mass_kg: float, scale: float, episodes: int) -> list[dict]:
    """Battery-identical action distribution, small sample, AT-6' inputs."""

    from v2_env_correction_acceptance import battery_episode_plan, build_probe_env, family_goals
    from v2_stage2_stratified_battery import stratified_actions

    from dgcc.tasks.episode import BatchedEpisodeRunner, EpisodeConfig

    apply_stiffness_scale(scale)
    params = build_params(mass_kg)
    goals = family_goals()
    rows: list[dict] = []

    for entry in battery_episode_plan():
        if entry["episode"] > episodes:
            continue
        _f, _g, goal = goals[entry["family_index"]]
        env = build_probe_env(1)
        env.reset(params, init_shape=entry["init_shape"], seed=1_000 + entry["episode"])
        runner = BatchedEpisodeRunner(env, params, EpisodeConfig())
        runner.begin_episodes(
            seed=entry["seed"], episode_index=entry["episode"],
            init_shapes=[entry["init_shape"]], goals=[goal],
        )
        for k, action in enumerate(stratified_actions(entry["episode"], entry["seed"])):
            env.probe_begin_primitive()
            try:
                out = runner.step(
                    np.asarray([action["p"]], dtype=int),
                    np.asarray([action["delta"]], dtype=float),
                    [action["lift"]],
                    rng=np.random.default_rng([88_000, entry["episode"], k]),
                )
            finally:
                probe = env.probe_end_primitive()
            settle = int(out["settle_steps"][0])
            denom = probe["move_steps_excl_approach"]
            rows.append({
                "mass_kg": mass_kg, "stiffness_scale": scale,
                "episode": entry["episode"], "primitive": k, "lift": action["lift"],
                "settle_steps": settle,
                "move_steps_excl_approach": denom,
                "settle_to_move_ratio": settle / denom if denom else float("inf"),
                "ground_penetration_steps": probe["ground_penetration_steps"],
                "min_node_z": probe["min_node_z"],
                "v_peak_total": probe["v_peak_total"],
                "strain_peak": probe["strain_peak"],
                "settle_converged": bool(out["info"]["settle_converged"][0]),
                "arclen_dev_after_settle": abs(probe["arclen_final"] / 1.0 - 1.0),
            })
        del runner, env
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backend", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--section", choices=["sag", "sample", "both"], default="both")
    args = ap.parse_args()

    from v2_env_correction_acceptance import init_genesis

    backend = init_genesis(args.backend)
    started = time.time()
    shapes = ["straight", "u_bend", "s_curve", "random_smooth"]
    payload: dict[str, object] = {"probe": "rev8-sag-and-stiffness", "backend": backend}

    if args.section in ("sag", "both"):
        sag: list[dict] = []
        for mass, scale in ((0.032, 1.0), (0.040, 1.0), (0.040, 1.25)):
            sag.extend(sag_cell(mass, scale, shapes))
        payload["sag"] = sag

    if args.section in ("sample", "both"):
        samples: list[dict] = []
        for mass, scale in ((0.040, 1.25),):
            samples.extend(sample_cell(mass, scale, args.episodes))
        payload["samples"] = samples

    payload["elapsed_s"] = round(time.time() - started, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sections": args.section, "elapsed_s": payload["elapsed_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
