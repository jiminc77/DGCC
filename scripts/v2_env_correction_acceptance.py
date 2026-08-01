#!/usr/bin/env python3
"""Quasi-static primitive acceptance battery: AT-1 .. AT-9 remeasurement.

Runs the standard battery (5 T2 goal families x 4 init shapes x 3 seeds = 60
episodes, 10 primitives each = 600 primitives, n_envs=1, fresh env per
episode, validation split only) on the QUASI-STATIC primitive (R1-R6 + R10,
with D1-D4 already merged so the AT-14/AT-16 instrumentation is live —
design §2.4 remeasurement requirement) and judges:

  AT-1  peak node velocity            <= 4 x v_max (0.60 m/s)
  AT-2  peak KE / gravitational PE    <= 0.25
  AT-3  node speed at detach          <= 0.05 m/s
  AT-4  peak edge strain              <= 0.01
  AT-5  post-settle |L/L0 - 1|        <= 1e-3
  AT-6  settle:move time ratio        <= 2
  AT-7  every settle converges        100%
  AT-8  covenant events               0 (nonfinite + magnitude census plus
        directly measured |L/L0-1| > 0.05 arc-length events; the R7
        production covenant re-verifies this at the Stage 3 gate)
  AT-9  subfloor gripper targets      0 (the R5 assert never fires)

Instrumentation is a `Probe` subclass wrapping production methods via
``super()`` (design §5.5) — production logic is not modified.  Per-step
classification: a step preceded by a gripper command with a changed target is
a MOVE step, with an identical target a HOLD step, otherwise a SETTLE/other
step.  Backend is selectable and recorded (2026-08-01 boundary amendment:
backend annotation mandatory).  Process-isolated slices as in the constraint
integrity battery (Genesis recompile-accumulation segfault).

Exit code 0 only when AT-1..AT-9 all pass (fail-closed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FORBIDDEN_PATH_TOKENS = ("heldout", "held-out", "patch_eval", "patching_probe")
BATTERY_SEEDS = (0, 1, 2)
BATTERY_PRIMITIVES = 10
BATTERY_SLICE_EPISODES = 10
V_MAX = 0.15
HOLD_MAX_STEPS = 2000
GRAVITY = 9.81
AT_THRESHOLDS = {
    "at1_peak_v": 4.0 * V_MAX,
    "at2_ke_over_pe": 0.25,
    "at3_v_at_detach": 0.05,
    "at4_strain": 0.01,
    "at5_arclen_dev": 1.0e-3,
    "at6_settle_move_ratio": 2.0,
    "at8_arclen_covenant": 0.05,
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
            raise RuntimeError("CPU section unexpectedly sees a CUDA device")
        if not getattr(gs, "_initialized", False):
            gs.init(seed=0, precision="32", logging_level="warning", backend=gs.cpu)
        return {"backend": "cpu", "device": "cpu"}
    if not torch.cuda.is_available():
        raise RuntimeError("GPU battery requested but no CUDA device is visible")
    if not getattr(gs, "_initialized", False):
        gs.init(seed=0, precision="32", logging_level="warning", backend=gs.gpu)
    return {"backend": "gpu", "device": torch.cuda.get_device_name(0)}


def build_probe_env(n_envs: int):
    """Probe subclass: per-step kinematic instrumentation via super() wrapping."""
    from dgcc.envs.dlolab import DLOLabEnv, SEGMENT_MASS_BASE

    class ProbeEnv(DLOLabEnv):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.probe_active = False
            self.probe_rows: list[tuple[str, float, float, float, float]] = []
            self.probe_v_at_detach: float = float("nan")
            self._probe_last_cmd: np.ndarray | None = None
            self._probe_pending_cmd: str | None = None

        # -- instrumentation taps ---------------------------------------
        def _set_gripper_positions(self, positions: np.ndarray) -> None:
            if self.probe_active:
                pos = np.asarray(positions, dtype=float)
                if self._probe_last_cmd is not None and np.array_equal(
                    pos, self._probe_last_cmd
                ):
                    self._probe_pending_cmd = "hold"
                else:
                    self._probe_pending_cmd = "move"
                self._probe_last_cmd = pos.copy()
            super()._set_gripper_positions(positions)

        def _step_scene(self) -> None:
            super()._step_scene()
            if not self.probe_active:
                return
            phase = self._probe_pending_cmd or "settle"
            self._probe_pending_cmd = None
            vels = np.asarray(self.rod_entity.get_all_vels(), dtype=float)
            if vels.ndim == 2:
                vels = vels[None, ...]
            speed = float(np.linalg.norm(vels[0], axis=-1).max())
            ke = float(0.5 * SEGMENT_MASS_BASE * np.sum(vels[0] ** 2))
            raw = np.asarray(self._raw_batch(), dtype=float)[0]
            edges = np.linalg.norm(raw[1:] - raw[:-1], axis=-1)
            arclen = float(edges.sum())
            rest = float(self.params.length_m) / (raw.shape[0] - 1)
            strain = float(np.abs(edges / rest - 1.0).max())
            min_z = float(raw[:, 2].min())
            self.probe_rows.append((phase, speed, ke, arclen, strain, min_z))

        def _verified_detach_batch(self):
            if self.probe_active:
                self.probe_v_at_detach = float(np.max(self.max_node_speed_batch()))
            return super()._verified_detach_batch()

        # -- per-primitive bookkeeping ----------------------------------
        def probe_begin_primitive(self) -> None:
            self.probe_rows = []
            self.probe_v_at_detach = float("nan")
            self._probe_last_cmd = None
            self._probe_pending_cmd = None
            self.probe_active = True

        def probe_end_primitive(self) -> dict[str, Any]:
            self.probe_active = False
            rows = self.probe_rows
            phases = np.asarray([r[0] for r in rows])
            speed = np.asarray([r[1] for r in rows], dtype=float)
            ke = np.asarray([r[2] for r in rows], dtype=float)
            arclen = np.asarray([r[3] for r in rows], dtype=float)
            strain = np.asarray([r[4] for r in rows], dtype=float)
            min_z = np.asarray([r[5] for r in rows], dtype=float)
            move_mask = phases == "move"
            hold_mask = phases == "hold"
            settle_mask = phases == "settle"
            # T2 (adjudication §1.8-1): attribute the move-phase peak to the
            # lift/translate/lower leg.  Move-classified steps are ordered:
            # a leading remainder (teleport/attach gripper commands) followed
            # by exactly the per-waypoint walks recorded by the adapter.
            move_speed = speed[move_mask]
            waypoint_steps = [int(n) for n in self.last_waypoint_steps]
            leg_names = ["lift", "translate", "lower"][: len(waypoint_steps)]
            walk_total = int(sum(waypoint_steps))
            leg_peaks: dict[str, float] = {}
            premove_peak = 0.0
            if len(move_speed) >= walk_total > 0:
                lead = len(move_speed) - walk_total
                premove_peak = float(move_speed[:lead].max()) if lead else 0.0
                cursor = lead
                for name, count in zip(leg_names, waypoint_steps):
                    segment = move_speed[cursor:cursor + count]
                    leg_peaks[f"v_peak_{name}"] = (
                        float(segment.max()) if segment.size else 0.0
                    )
                    cursor += count
            return {
                "total_sim_steps": int(len(rows)),
                "move_steps": int(move_mask.sum()),
                "hold_steps": int(hold_mask.sum()),
                "settle_steps_observed": int(settle_mask.sum()),
                "n_waypoint_steps": waypoint_steps,
                "v_peak_premove": premove_peak,
                **leg_peaks,
                "v_peak_move": float(speed[move_mask | hold_mask].max()) if (move_mask | hold_mask).any() else 0.0,
                "v_peak_settle": float(speed[settle_mask].max()) if settle_mask.any() else 0.0,
                "v_peak_total": float(speed.max()) if len(rows) else 0.0,
                "v_at_detach": self.probe_v_at_detach,
                "ke_peak": float(ke.max()) if len(rows) else 0.0,
                "strain_peak": float(strain.max()) if len(rows) else 0.0,
                "arclen_peak": float(arclen.max()) if len(rows) else 0.0,
                "arclen_final": float(arclen[-1]) if len(rows) else float("nan"),
                # T2 (adjudication §1.8-2 / AT-9b): actual ground penetration.
                # z is the node CENTER; center below the plane (z < 0) is an
                # unambiguous penetration regardless of the 5 mm rope radius.
                "min_node_z": float(min_z.min()) if len(rows) else float("nan"),
                "ground_penetration_steps": int((min_z < 0.0).sum()),
            }

    return ProbeEnv(
        n_envs=n_envs,
        dt=1.0e-3,
        substeps=5,
        rod_damping=10.0,
        rod_angular_damping=5.0,
        initial_settle_steps=0,
        reset_settle_max_steps=10_000,
        move_v_max=V_MAX,
        move_hold_max_steps=HOLD_MAX_STEPS,
        grasp_realism=True,
    )


def family_goals() -> list[tuple[str, str, Any]]:
    from dgcc.tasks.t2 import T2_FAMILIES, load_t2_payload, load_t2_split_payload

    pairs = load_t2_split_payload("val", load_t2_payload())
    chosen: list[tuple[str, str, Any]] = []
    for family in T2_FAMILIES:
        for spec, goal in pairs:
            if str(spec["family"]) == family:
                chosen.append((family, str(spec["goal_id"]), goal))
                break
        else:
            raise RuntimeError(f"no validation goal for family {family!r}")
    return chosen


def battery_episode_plan() -> list[dict[str, Any]]:
    from dgcc.tasks.episode import INIT_SHAPES

    plan: list[dict[str, Any]] = []
    ordinal = 0
    for family_index, (family, goal_id, _goal) in enumerate(family_goals()):
        for shape in INIT_SHAPES:
            for seed in BATTERY_SEEDS:
                ordinal += 1
                plan.append(
                    {
                        "episode": ordinal,
                        "family_index": family_index,
                        "family": family,
                        "goal_id": goal_id,
                        "init_shape": shape,
                        "seed": int(seed),
                    }
                )
    return plan


def episode_actions(rng: np.random.Generator, n_envs: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    p = rng.integers(0, 32, n_envs)
    delta = rng.normal(0.0, 0.06, (n_envs, 3))
    lift = [str(x) for x in rng.choice(["low", "high"], n_envs)]
    return p, delta, lift


def run_slice(first: int, last: int, backend_info: dict[str, Any]) -> dict[str, Any]:
    from dgcc.envs.dlolab import LIFT_HEIGHTS
    from dgcc.tasks.domain import p1_rope_params
    from dgcc.tasks.episode import BatchedEpisodeRunner, EpisodeConfig

    params = p1_rope_params()
    goals = family_goals()
    rope_mass = 32 * 1.0e-3

    primitives: list[dict[str, Any]] = []
    for entry in battery_episode_plan():
        if not (first <= entry["episode"] <= last):
            continue
        _family, _goal_id, goal = goals[entry["family_index"]]
        episode_ordinal = entry["episode"]
        env = build_probe_env(1)
        env.reset(params, init_shape=entry["init_shape"], seed=1_000 + episode_ordinal)
        runner = BatchedEpisodeRunner(env, params, EpisodeConfig())
        runner.begin_episodes(
            seed=entry["seed"],
            episode_index=episode_ordinal,
            init_shapes=[entry["init_shape"]],
            goals=[goal],
        )
        action_rng = np.random.default_rng([9_000, episode_ordinal, entry["seed"]])
        for k in range(BATTERY_PRIMITIVES):
            p, delta, lift = episode_actions(action_rng, 1)
            env.probe_begin_primitive()
            subfloor = False
            try:
                out = runner.step(p, delta, lift, rng=action_rng)
            except RuntimeError as exc:
                if "subfloor" in str(exc):
                    subfloor = True
                    raise
                raise
            finally:
                probe = env.probe_end_primitive()
            info = out["info"]
            lift_height = float(LIFT_HEIGHTS[str(lift[0])])
            grav_pe = rope_mass * GRAVITY * lift_height
            move_steps = probe["move_steps"]
            settle_steps = int(out["settle_steps"][0])
            primitives.append(
                {
                    "episode": episode_ordinal,
                    "primitive": k,
                    "family": entry["family"],
                    "goal_id": entry["goal_id"],
                    "init_shape": entry["init_shape"],
                    "seed": entry["seed"],
                    "backend": backend_info["backend"],
                    "lift": str(lift[0]),
                    "target_z": float(env.last_move_target[0, 2]),
                    "subfloor_target_commanded": subfloor,
                    "n_waypoint_steps": probe["n_waypoint_steps"],
                    "v_peak_premove": probe["v_peak_premove"],
                    "v_peak_lift": probe.get("v_peak_lift", 0.0),
                    "v_peak_translate": probe.get("v_peak_translate", 0.0),
                    "v_peak_lower": probe.get("v_peak_lower", 0.0),
                    "min_node_z": probe["min_node_z"],
                    "ground_penetration_steps": probe["ground_penetration_steps"],
                    "v_peak_move": probe["v_peak_move"],
                    "v_peak_settle": probe["v_peak_settle"],
                    "v_peak_total": probe["v_peak_total"],
                    "v_at_detach": probe["v_at_detach"],
                    "ke_peak": probe["ke_peak"],
                    "grav_pe": grav_pe,
                    "ke_over_pe": probe["ke_peak"] / grav_pe,
                    "strain_peak": probe["strain_peak"],
                    "arclen_peak": probe["arclen_peak"],
                    "arclen_after_settle": probe["arclen_final"],
                    "arclen_dev_after_settle": abs(
                        probe["arclen_final"] / float(params.length_m) - 1.0
                    ),
                    "settle_steps": settle_steps,
                    "settle_converged": bool(out["info"]["settle_converged"][0]),
                    "move_steps": move_steps,
                    "hold_steps_used": int(info["hold_steps_used"]),
                    "hold_converged": bool(info["hold_converged"]),
                    "settle_to_move_ratio": settle_steps / move_steps if move_steps else float("inf"),
                    "total_sim_steps": probe["total_sim_steps"],
                    "detach_residuals": int(info["detach_residuals"]),
                    "detach_escalations": int(info["detach_escalations"]),
                    "post_detach_leftover": int(env._vertex_constrained_mask().sum()),
                }
            )
        covenant = {
            "nan_incidents": int(runner.nan_incidents),
            "magnitude_incidents": int(runner.magnitude_incidents),
        }
        primitives[-1]["episode_covenant"] = covenant
        del runner
        del env
    return {"primitives": primitives, "backend_info": backend_info}


def judge(primitives: list[dict[str, Any]]) -> dict[str, Any]:
    def collect(key: str) -> np.ndarray:
        return np.asarray([p[key] for p in primitives], dtype=float)

    v_peak = collect("v_peak_total")
    ke_over_pe = collect("ke_over_pe")
    v_detach = collect("v_at_detach")
    strain = collect("strain_peak")
    arclen_dev = collect("arclen_dev_after_settle")
    ratio = collect("settle_to_move_ratio")
    converged = np.asarray([p["settle_converged"] for p in primitives], dtype=bool)
    subfloor = np.asarray([p["subfloor_target_commanded"] for p in primitives], dtype=bool)
    arclen_peak_rel = np.abs(collect("arclen_peak") - 1.0)
    nan_events = sum(
        p.get("episode_covenant", {}).get("nan_incidents", 0) for p in primitives
    )
    magnitude_events = sum(
        p.get("episode_covenant", {}).get("magnitude_incidents", 0) for p in primitives
    )
    arclen_covenant_events = int(
        (arclen_peak_rel > AT_THRESHOLDS["at8_arclen_covenant"]).sum()
    )

    tests = {
        "AT-1": {
            "criterion": f"peak node velocity <= {AT_THRESHOLDS['at1_peak_v']} m/s",
            "min": float(v_peak.min()), "max": float(v_peak.max()),
            "violations": int((v_peak > AT_THRESHOLDS["at1_peak_v"]).sum()),
        },
        "AT-2": {
            "criterion": f"peak KE / grav PE <= {AT_THRESHOLDS['at2_ke_over_pe']}",
            "min": float(ke_over_pe.min()), "max": float(ke_over_pe.max()),
            "violations": int((ke_over_pe > AT_THRESHOLDS["at2_ke_over_pe"]).sum()),
        },
        "AT-3": {
            "criterion": f"node speed at detach <= {AT_THRESHOLDS['at3_v_at_detach']} m/s",
            "min": float(v_detach.min()), "max": float(v_detach.max()),
            "violations": int((v_detach > AT_THRESHOLDS["at3_v_at_detach"]).sum()),
        },
        "AT-4": {
            "criterion": f"peak edge strain <= {AT_THRESHOLDS['at4_strain']}",
            "min": float(strain.min()), "max": float(strain.max()),
            "violations": int((strain > AT_THRESHOLDS["at4_strain"]).sum()),
        },
        "AT-5": {
            "criterion": f"post-settle |L/L0-1| <= {AT_THRESHOLDS['at5_arclen_dev']}",
            "min": float(arclen_dev.min()), "max": float(arclen_dev.max()),
            "violations": int((arclen_dev > AT_THRESHOLDS["at5_arclen_dev"]).sum()),
        },
        "AT-6": {
            "criterion": f"settle:move ratio <= {AT_THRESHOLDS['at6_settle_move_ratio']}",
            "min": float(ratio.min()), "max": float(ratio.max()),
            "violations": int((ratio > AT_THRESHOLDS["at6_settle_move_ratio"]).sum()),
        },
        "AT-7": {
            "criterion": "every settle converges",
            "violations": int((~converged).sum()),
        },
        "AT-8": {
            "criterion": "covenant events == 0 (nonfinite + magnitude + arc-length)",
            "nan_incidents": int(nan_events),
            "magnitude_incidents": int(magnitude_events),
            "arclen_covenant_events": arclen_covenant_events,
            "violations": int(nan_events + magnitude_events + arclen_covenant_events),
        },
        "AT-9": {
            "criterion": "subfloor gripper targets == 0",
            "violations": int(subfloor.sum()),
        },
    }
    for test in tests.values():
        test["pass"] = test["violations"] == 0

    # T2 repairs (Stage 2 adjudication §1.8; informational until the Rev 3
    # criteria are pinned by the owner — reported, not gated):
    hold_cap_exhausted = sum(1 for p in primitives if not p["hold_converged"])
    tests["AT-3"]["hold_cap_exhausted"] = int(hold_cap_exhausted)
    tests["AT-3"]["hold_cap_exhausted_rate"] = round(
        hold_cap_exhausted / len(primitives), 4
    )
    penetration = np.asarray(
        [p.get("ground_penetration_steps", 0) for p in primitives], dtype=int
    )
    min_node_z = np.asarray(
        [p.get("min_node_z", np.nan) for p in primitives], dtype=float
    )
    tests["AT-9b (reported)"] = {
        "criterion": "actual node ground penetration (center z < 0) == 0",
        "primitives_with_penetration": int((penetration > 0).sum()),
        "penetration_steps_total": int(penetration.sum()),
        "min_node_z_overall": float(np.nanmin(min_node_z)),
        "informational_pending_rev3": True,
        "pass": True,
    }
    quantiles = [50, 90, 95, 99]
    tests["arclen_transient_stats (reported)"] = {
        "criterion": "transient |L/L0-1| evidence for the R7 covenant "
                     "threshold review (0.05 -> 0.02 candidate)",
        "max": float(arclen_peak_rel.max()),
        **{f"p{q}": float(np.percentile(arclen_peak_rel, q)) for q in quantiles},
        "count_above_0.02": int((arclen_peak_rel > 0.02).sum()),
        "count_above_0.05": int((arclen_peak_rel > 0.05).sum()),
        "informational_pending_rev3": True,
        "pass": True,
    }
    return tests


def spawn_section(out_path: Path, backend: str, first: int, last: int, timeout_s: int) -> dict[str, Any]:
    env = dict(os.environ)
    if backend == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--section", "slice", "--backend", backend,
        "--slice-first", str(first), "--slice-last", str(last),
        "--out", str(out_path),
    ]
    completed = subprocess.run(
        command, env=env, timeout=timeout_s,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        tail = completed.stderr.decode(errors="replace")[-2000:]
        raise RuntimeError(f"slice [{first},{last}] exited {completed.returncode}: {tail}")
    return json.loads(out_path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battery", choices=["standard"], default="standard")
    parser.add_argument("--out", type=permitted_path, required=True)
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--section", choices=["driver", "slice"], default="driver")
    parser.add_argument("--slice-first", type=int, default=1)
    parser.add_argument("--slice-last", type=int, default=60)
    args = parser.parse_args()

    started = time.time()
    if args.section == "slice":
        backend_info = init_genesis(args.backend)
        result = run_slice(args.slice_first, args.slice_last, backend_info)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return 0

    plan_size = len(battery_episode_plan())
    slices = [
        (first, min(first + BATTERY_SLICE_EPISODES - 1, plan_size))
        for first in range(1, plan_size + 1, BATTERY_SLICE_EPISODES)
    ]
    primitives: list[dict[str, Any]] = []
    backends: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="v2_acceptance_") as tmp:
        for first, last in slices:
            partial = spawn_section(
                Path(tmp) / f"slice_{first:03d}.json", args.backend, first, last, 5400
            )
            primitives.extend(partial["primitives"])
            backends.add(partial["backend_info"]["backend"])

    expected = plan_size * BATTERY_PRIMITIVES
    if len(primitives) != expected:
        raise RuntimeError(f"merged {len(primitives)} of {expected} primitives")

    tests = judge(primitives)
    result = {
        "schema_version": 1,
        "battery": args.battery,
        "backend": sorted(backends),
        "v_max": V_MAX,
        "hold_max_steps": HOLD_MAX_STEPS,
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "primitives_measured": len(primitives),
        "acceptance": tests,
        "step_statistics": {
            "total_sim_steps_mean": float(np.mean([p["total_sim_steps"] for p in primitives])),
            "total_sim_steps_max": int(np.max([p["total_sim_steps"] for p in primitives])),
            "settle_steps_mean": float(np.mean([p["settle_steps"] for p in primitives])),
            "hold_steps_mean": float(np.mean([p["hold_steps_used"] for p in primitives])),
            "move_steps_mean": float(np.mean([p["move_steps"] for p in primitives])),
        },
        "at16_companion": {
            "detach_residuals": int(sum(p["detach_residuals"] for p in primitives)),
            "detach_escalations": int(sum(p["detach_escalations"] for p in primitives)),
            "post_detach_leftover": int(sum(p["post_detach_leftover"] for p in primitives)),
        },
        "primitives": primitives,
        "elapsed_s": round(time.time() - started, 1),
    }
    result["pass"] = all(test["pass"] for test in tests.values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"acceptance": tests, "pass": result["pass"], "backend": result["backend"]}, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
