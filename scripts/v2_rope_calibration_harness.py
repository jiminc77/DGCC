#!/usr/bin/env python3
"""Real-rope anchoring harness: three benchtop tests + a sequential fit.

MEASUREMENT ONLY.  Nothing here changes repository source, acceptance
thresholds, or the shipped domain constants: rope discretization, mass,
stiffness (E/K/G) and damping (gamma_lin/gamma_ang) are all supplied per run
and applied as runtime overrides, so a calibration sweep cannot leak into the
production environment.

Tests
-----
A  edge droop      Rope lies on a virtual desk with an overhang hanging past
                   the edge; measure the droop of the free end.  Swept over
                   overhang length, then the overhang `c` that produces a
                   given droop angle is interpolated.
B  hanging weight  Top node held, a mass hung on the bottom node; measure the
                   elongation in mm.
C  drop settle     One end held up, released; measure the time to quiescence.

Support / load model (recorded because it is a modelling decision, not a
detail):

  * The desk in test A and the clamp in test B are realized with the rod
    solver's own per-vertex ``fixed`` flags rather than a collision body.  A
    rope resting on a desk is held by normal force plus friction; over the
    supported span that is kinematically indistinguishable from "these
    vertices do not move", and it removes desk friction/penetration as a
    confound in a test whose whole purpose is to isolate BENDING.  It is the
    standard cantilever idealization.  Consequence: the desk height itself is
    physically irrelevant (gravity is uniform); it only has to be high enough
    that the drooping end never reaches the ground plane.
  * The hanging weight in test B is applied by raising the END VERTEX's mass
    by the load, through the public ``set_segment_mass`` setter, so the load
    enters as m_load * g exactly like a real weight.  There is no external
    force API on the rod entity, and pulling with the gripper would be
    position- not force-controlled, which is the wrong boundary condition.
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

GRAVITY = 9.81


# --------------------------------------------------------------- environment


def apply_stiffness(bend: float | None, stretch: float | None, twist: float | None) -> dict:
    """Runtime override of the simulator-unit bases (repo source untouched)."""

    import dgcc.envs.dlolab as dl

    if not hasattr(dl, "_CAL_BASE"):
        dl._CAL_BASE = (dl.BEND_BASE, dl.STRETCH_BASE, dl.TWIST_BASE)
    b0, s0, t0 = dl._CAL_BASE
    dl.BEND_BASE = b0 if bend is None else float(bend)
    dl.STRETCH_BASE = s0 if stretch is None else float(stretch)
    dl.TWIST_BASE = t0 if twist is None else float(twist)
    return {"bend_E": dl.BEND_BASE, "stretch_K": dl.STRETCH_BASE, "twist_G": dl.TWIST_BASE}


def rope_params(n_segments: int, mass_kg: float, length_m: float = 1.0, radius: float = 0.005):
    from dgcc.envs.base import RopeParams

    return RopeParams(
        length_m=length_m,
        n_segments=n_segments,
        bend_stiffness=1.0,
        twist_stiffness=1.0,
        friction=1.0,
        radius=radius,
        rope_mass_total_kg=mass_kg,
    )


def make_env(gamma_lin: float, gamma_ang: float, dt: float = 1.0e-3, substeps: int = 5):
    """dt/substeps are the integrator settings the stability ceiling depends on."""
    from dgcc.envs.dlolab import DLOLabEnv

    return DLOLabEnv(
        n_envs=1,
        dt=dt,
        substeps=substeps,
        rod_damping=float(gamma_lin),
        rod_angular_damping=float(gamma_ang),
        initial_settle_steps=0,
        reset_settle_max_steps=1,
        move_v_max=0.15,
        move_hold_max_steps=2000,
        grasp_realism=False,
    )


# ------------------------------------------------------------------ kinematics


def place(env, vertices: np.ndarray) -> None:
    """Write a centerline and rebuild the edge/frame state (no fixed flags)."""

    env._place_rod_vertices(np.asarray(vertices, dtype=float))


def set_fixed(env, fixed_idx: np.ndarray) -> None:
    """Pin the given vertices in place via the solver's own `fixed` flags.

    Read-modify-write of the full solver state through the frozen
    get/set kernel pair, i.e. the same path the failed-grasp rewind uses.
    """

    import torch

    snap = env._snapshot_rod_state()
    snap["fixed"][:] = False
    if len(fixed_idx):
        snap["fixed"][:, np.asarray(fixed_idx, dtype=int)] = True
    envs_idx = torch.zeros(1, dtype=torch.int32, device=env.gs.device)
    env.rod_entity._solver._kernel_set_state(
        env.rod_entity._sim.cur_substep_local,
        envs_idx,
        *(snap[name] for name in env._ROD_STATE_FIELDS),
    )


def set_vertex_masses(env, masses: np.ndarray) -> None:
    import torch

    env.rod_entity.set_segment_mass(
        torch.as_tensor(
            np.asarray(masses, dtype=float).reshape(1, -1),
            dtype=env.gs.tc_float, device=env.gs.device,
        )
    )


def free_speed(env, fixed_idx: np.ndarray | None) -> float:
    """Max speed over the vertices that are actually free to move.

    A pinned vertex keeps accumulating a velocity in the solver state even
    though `fixed` holds its position, so `max_node_speed_batch()` -- a max
    over ALL vertices -- never falls below any threshold once anything is
    clamped.  Quiescence has to be judged on the free set.
    """

    vel = env._node_velocities_batch()[0]
    if fixed_idx is not None and len(fixed_idx):
        mask = np.ones(vel.shape[0], dtype=bool)
        mask[np.asarray(fixed_idx, dtype=int)] = False
        vel = vel[mask]
    if vel.size == 0:
        return 0.0
    return float(np.linalg.norm(vel, axis=-1).max())


def relax(env, threshold: float = 1.0e-3, max_steps: int = 40_000,
          fixed_idx: np.ndarray | None = None) -> tuple[int, bool]:
    for step in range(max_steps):
        if free_speed(env, fixed_idx) < threshold:
            return step, True
        env._step_scene()
    return max_steps, False


def verts(env) -> np.ndarray:
    return np.asarray(env._raw_batch(), dtype=float)[0]


# ---------------------------------------------------------------------- tests


def test_a_droop(env, params, overhangs, desk_z: float = 0.75, threshold: float = 1.0e-3):
    """Rope on a desk with `overhang` past the edge; measure the free-end droop."""

    n = int(params.n_segments)
    L = float(params.length_m)
    interval = L / (n - 1)
    rows = []
    for overhang in overhangs:
        supported = L - float(overhang)
        if supported <= 2 * interval:
            continue
        line = np.zeros((n, 3))
        line[:, 0] = np.linspace(-supported, float(overhang), n)
        line[:, 2] = desk_z
        place(env, line)
        edge_i = int(np.searchsorted(line[:, 0], 0.0, side="right") - 1)
        pinned = np.arange(0, edge_i + 1)
        set_fixed(env, pinned)
        steps, converged = relax(env, threshold, fixed_idx=pinned)
        v = verts(env)
        tip_edge = v[-1] - v[-2]
        chord = v[-1] - v[edge_i]
        rows.append({
            "overhang_m": float(overhang),
            "supported_m": supported,
            "edge_vertex": edge_i,
            "free_vertices": n - 1 - edge_i,
            "tip_tangent_angle_deg": float(
                np.degrees(np.arctan2(-tip_edge[2], abs(tip_edge[0]) + 1e-12))
            ),
            "chord_angle_deg": float(
                np.degrees(np.arctan2(-chord[2], abs(chord[0]) + 1e-12))
            ),
            "tip_drop_m": float(desk_z - v[-1, 2]),
            "profile_xz": [[float(a), float(b)] for a, b in zip(v[edge_i:, 0], v[edge_i:, 2])],
            "relax_steps": steps,
            "relax_converged": converged,
            "angle_resolved": bool(n - 1 - edge_i >= 4),
        })
    return rows


def droop_angle_at(env, params, overhang: float, desk_z: float = 0.75,
                   metric: str = "tip_tangent_angle_deg") -> float:
    row = test_a_droop(env, params, [overhang], desk_z)
    return row[0][metric] if row else float("nan")


def interpolate_overhang_for_angle(rows, target_deg: float,
                                   metric: str = "tip_tangent_angle_deg") -> float | None:
    """Linear interpolation of the overhang that yields `target_deg`."""

    pts = sorted(((r["overhang_m"], r[metric]) for r in rows), key=lambda t: t[0])
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if (y0 - target_deg) * (y1 - target_deg) <= 0 and y1 != y0:
            return float(x0 + (target_deg - y0) * (x1 - x0) / (y1 - y0))
    return None


def relax_span(env, fixed_idx, threshold: float, max_steps: int, span_tol: float,
               window: int = 500) -> tuple[int, bool, bool]:
    """Relax until the free vertices are quiescent OR the span stops changing."""

    def span_now() -> float:
        v = verts(env)
        return float(np.linalg.norm(v[-1] - v[0]))

    last = span_now()
    for step in range(max_steps):
        if free_speed(env, fixed_idx) < threshold:
            return step, True, True
        env._step_scene()
        if (step + 1) % window == 0:
            cur = span_now()
            if abs(cur - last) < span_tol:
                return step + 1, False, True
            last = cur
    return max_steps, False, False


def test_b_hang(env, params, loads_kg, top_z: float = 1.60, threshold: float = 1.0e-3,
                hang_max_steps: int = 120_000, span_tol: float = 1.0e-6):
    """Top vertex clamped, `load` hung on the bottom vertex; elongation in mm."""

    n = int(params.n_segments)
    L = float(params.length_m)
    seg_mass = float(params.rope_mass_total_kg) / n
    rows = []
    baseline_span = None
    for load in [0.0, *loads_kg]:
        column = np.zeros((n, 3))
        column[:, 2] = np.linspace(top_z, top_z - L, n)
        place(env, column)
        masses = np.full(n, seg_mass)
        masses[-1] = seg_mass + float(load)
        set_vertex_masses(env, masses)
        set_fixed(env, np.array([0]))
        # Converge on the MEASURAND.  A heavily loaded rope can creep for a
        # long time at a speed just above the velocity threshold while the
        # span -- the thing test B reports -- has already stopped moving, so
        # the span's own stability is the correct stopping rule here; the
        # velocity rule is kept as the primary (tighter) criterion.
        steps, converged, span_stable = relax_span(
            env, np.array([0]), threshold, hang_max_steps, span_tol
        )
        v = verts(env)
        span = float(np.linalg.norm(v[-1] - v[0]))
        arc = float(np.linalg.norm(np.diff(v, axis=0), axis=1).sum())
        if load == 0.0:
            baseline_span = span
        rows.append({
            "load_kg": float(load),
            "load_N": float(load) * GRAVITY,
            "span_m": span,
            "arc_len_m": arc,
            "elongation_mm": None if baseline_span is None else (span - baseline_span) * 1000.0,
            "arc_strain": arc / L - 1.0,
            "relax_steps": steps,
            "relax_converged": converged,
            "span_stable": span_stable,
        })
    set_vertex_masses(env, np.full(n, seg_mass))
    return rows


def test_c_drop(env, params, lift_m: float = 0.20, threshold: float = 1.0e-3,
                max_steps: int = 60_000):
    """One end held at `lift_m`, released; time to quiescence."""

    n = int(params.n_segments)
    L = float(params.length_m)
    rest_z = max(float(params.radius), 0.005)
    line = np.zeros((n, 3))
    line[:, 0] = np.linspace(-L / 2.0, L / 2.0, n)
    line[:, 2] = rest_z
    place(env, line)
    relax(env, threshold, 20_000)

    held = verts(env)
    held[-1, 2] = rest_z + float(lift_m)
    # ease the last few vertices up so the initial state is a plausible lift
    ramp = min(8, n - 2)
    for j in range(1, ramp + 1):
        held[-1 - j, 2] = rest_z + float(lift_m) * (1.0 - j / (ramp + 1.0))
    place(env, held)
    set_fixed(env, np.array([n - 1]))
    pre_steps, pre_conv = relax(env, threshold, 20_000, fixed_idx=np.array([n - 1]))

    set_fixed(env, np.array([], dtype=int))  # release
    steps, converged = relax(env, threshold, max_steps)
    return {
        "lift_m": float(lift_m),
        "pre_release_relax_steps": pre_steps,
        "pre_release_converged": pre_conv,
        "settle_steps": steps,
        "settle_time_s": steps * float(env.dt),
        "settle_converged": converged,
        "threshold_m_per_s": threshold,
    }


# ------------------------------------------------------------------- fitting


def safe_eval(evaluate, x: float):
    """Evaluate a trial parameter, returning None if the solver goes unstable.

    High stiffness at a fixed dt/substeps is a CFL-type stability limit, not a
    modelling error: the trial simply cannot be represented by this
    integrator.  Treating it as a bracket constraint keeps the fit alive and
    makes the limit visible instead of crashing the run.
    """

    try:
        value = evaluate(x)
    except (FloatingPointError, RuntimeError, ValueError) as error:
        return None, f"{type(error).__name__}: {error}"[:160]
    if value is None or not np.isfinite(value):
        return None, "non-finite result"
    return float(value), None


def bisect(evaluate, lo: float, hi: float, target: float, *, increasing: bool,
           rel_tol: float = 0.05, max_iter: int = 12):
    """Bracketed bisection on a monotone response.  Returns (x, value, trace)."""

    trace = []
    # Pull the unstable end in until both bracket ends evaluate.
    for _ in range(8):
        v, err = safe_eval(evaluate, hi)
        trace.append({"x": float(hi), "value": v, "error": err, "phase": "bracket-hi"})
        if v is not None:
            break
        hi = float(np.sqrt(lo * hi)) if lo > 0 else 0.5 * (lo + hi)
    v_lo, err_lo = safe_eval(evaluate, lo)
    trace.append({"x": float(lo), "value": v_lo, "error": err_lo, "phase": "bracket-lo"})
    v_hi = v
    if v_lo is None or v_hi is None:
        usable = [t for t in trace if t["value"] is not None]
        if not usable:
            return lo, None, {"bracketed": False, "reason": "no stable trial", "trace": trace}
        best = min(usable, key=lambda t: abs(t["value"] - target))
        return best["x"], best["value"], {"bracketed": False, "trace": trace}
    ok = (v_lo <= target <= v_hi) if increasing else (v_hi <= target <= v_lo)
    if not ok:
        best = min(trace, key=lambda t: abs(t["value"] - target))
        return best["x"], best["value"], {"bracketed": False, "trace": trace}
    for _ in range(max_iter):
        mid = np.sqrt(lo * hi) if lo > 0 and hi > 0 else 0.5 * (lo + hi)
        value, err = safe_eval(evaluate, mid)
        trace.append({"x": float(mid), "value": value, "error": err, "phase": "bisect"})
        if value is None:
            hi = mid          # unstable side; pull the bracket in
            continue
        if target != 0 and abs(value - target) / abs(target) <= rel_tol:
            return float(mid), float(value), {"bracketed": True, "trace": trace}
        if (value < target) == increasing:
            lo = mid
        else:
            hi = mid
    usable = [t for t in trace if t["value"] is not None]
    best = min(usable, key=lambda t: abs(t["value"] - target))
    return best["x"], best["value"], {"bracketed": True, "trace": trace}


def run_fit(args, params, targets: dict) -> dict:
    """Sequential fit: A -> E, B -> K, C -> gamma, with one optional re-pass."""

    result = {"targets": targets, "passes": []}
    bend = args.bend_e
    stretch = args.stretch_k
    gamma = args.gamma_lin

    for sweep in range(1, int(args.fit_passes) + 1):
        rec: dict = {"pass": sweep}

        if targets.get("droop_overhang_mm") is not None:
            c_target = float(targets["droop_overhang_mm"]) / 1000.0
            angle_target = float(targets.get("droop_angle_deg", 41.5))

            def eval_bend(e):
                apply_stiffness(e, stretch, args.twist_g)
                env = make_env(gamma, args.gamma_ang, args.dt, args.substeps)
                env.reset(params, init_shape="straight", seed=0)
                a = droop_angle_at(env, params, c_target, args.desk_z, args.droop_metric)
                del env
                return a

            # stiffer rope -> smaller droop angle at a fixed overhang
            bend, value, info = bisect(
                eval_bend, args.bend_lo, args.bend_hi, angle_target,
                increasing=False, rel_tol=args.rel_tol,
            )
            rec["bend_E"] = {"value": bend, "sim_angle_deg": value,
                             "target_angle_deg": angle_target, **info}

        if targets.get("elongation_mm"):
            loads = list(targets.get("loads_kg", [0.5, 1.0, 2.0]))
            elong = list(targets["elongation_mm"])
            ref_i = len(loads) // 2

            def eval_stretch(k):
                apply_stiffness(bend, k, args.twist_g)
                env = make_env(max(gamma, args.hang_relax_damping), args.gamma_ang, args.dt, args.substeps)
                env.reset(params, init_shape="straight", seed=0)
                rows = test_b_hang(env, params, [loads[ref_i]], args.hang_top_z)
                del env
                return rows[-1]["elongation_mm"]

            stretch, value, info = bisect(
                eval_stretch, args.stretch_lo, args.stretch_hi, elong[ref_i],
                increasing=False, rel_tol=args.rel_tol,
            )
            rec["stretch_K"] = {"value": stretch, "sim_elongation_mm": value,
                                "target_elongation_mm": elong[ref_i],
                                "reference_load_kg": loads[ref_i], **info}

        if targets.get("settle_time_s") is not None:
            def eval_gamma(g):
                apply_stiffness(bend, stretch, args.twist_g)
                env = make_env(g, args.gamma_ang, args.dt, args.substeps)
                env.reset(params, init_shape="straight", seed=0)
                out = test_c_drop(env, params, args.drop_lift)
                del env
                return out["settle_time_s"]

            gamma, value, info = bisect(
                eval_gamma, args.gamma_lo, args.gamma_hi, float(targets["settle_time_s"]),
                increasing=False, rel_tol=args.rel_tol,
            )
            rec["gamma_lin"] = {"value": gamma, "sim_settle_time_s": value,
                                "target_settle_time_s": targets["settle_time_s"], **info}

        result["passes"].append(rec)

    result["fitted"] = {"bend_E": bend, "stretch_K": stretch,
                        "twist_G": args.twist_g, "gamma_lin": gamma,
                        "gamma_ang": args.gamma_ang}
    return result


# ---------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backend", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--section", default="all",
                    choices=["all", "A", "B", "C", "fit", "stability"])
    ap.add_argument("--n-segments", type=int, default=64)
    ap.add_argument("--mass-kg", type=float, default=0.040)
    ap.add_argument("--length-m", type=float, default=1.0)
    ap.add_argument("--bend-e", type=float, default=None)
    ap.add_argument("--stretch-k", type=float, default=None)
    ap.add_argument("--twist-g", type=float, default=None)
    ap.add_argument("--dt", type=float, default=1.0e-3)
    ap.add_argument("--substeps", type=int, default=5)
    ap.add_argument("--gamma-lin", type=float, default=10.0)
    ap.add_argument("--gamma-ang", type=float, default=5.0)
    ap.add_argument("--desk-z", type=float, default=0.75)
    ap.add_argument("--hang-top-z", type=float, default=1.60)
    ap.add_argument("--drop-lift", type=float, default=0.20)
    ap.add_argument("--overhangs", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40")
    ap.add_argument("--loads-kg", default="0.5,1.0,2.0")
    ap.add_argument("--droop-angle-deg", type=float, default=41.5)
    ap.add_argument("--droop-metric", default="tip_tangent_angle_deg",
                    choices=["tip_tangent_angle_deg", "chord_angle_deg"])
    ap.add_argument("--threshold", type=float, default=1.0e-3)
    ap.add_argument(
        "--hang-relax-damping", type=float, default=60.0,
        help="linear damping used ONLY for test B (dynamic relaxation).  Test B "
             "reports a STATIC equilibrium; with `vel *= exp(-gamma*dt)` damping "
             "the equilibrium is where forces balance at v=0, which is "
             "independent of gamma, so raising it shortens the path without "
             "moving the answer.  Test C measures a DYNAMIC quantity and never "
             "uses this.",
    )
    # fit targets / brackets
    ap.add_argument("--target-droop-overhang-mm", type=float, default=None)
    ap.add_argument("--target-elongation-mm", default=None,
                    help="comma separated, matching --loads-kg")
    ap.add_argument("--target-settle-time-s", type=float, default=None)
    ap.add_argument("--bend-lo", type=float, default=1.0e3)
    ap.add_argument("--bend-hi", type=float, default=1.0e7)
    ap.add_argument("--stretch-lo", type=float, default=1.0e4)
    ap.add_argument("--stretch-hi", type=float, default=1.0e8)
    ap.add_argument("--gamma-lo", type=float, default=0.5)
    ap.add_argument("--gamma-hi", type=float, default=200.0)
    ap.add_argument("--rel-tol", type=float, default=0.05)
    ap.add_argument("--fit-passes", type=int, default=2)
    args = ap.parse_args()

    from dgcc.envs.dlolab import ensure_genesis_initialized

    if args.backend == "cpu":
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    ensure_genesis_initialized(0)

    params = rope_params(args.n_segments, args.mass_kg, args.length_m)
    overhangs = [float(v) for v in args.overhangs.split(",") if v.strip()]
    loads = [float(v) for v in args.loads_kg.split(",") if v.strip()]
    started = time.time()

    payload: dict = {
        "harness": "rope-calibration",
        "n_segments": int(params.n_segments),
        "rope_mass_total_kg": float(params.rope_mass_total_kg),
        "length_m": float(params.length_m),
        "radius_m": float(params.radius),
        "gamma_lin": args.gamma_lin,
        "gamma_ang": args.gamma_ang,
        "dt": args.dt,
        "substeps": args.substeps,
        "support_model": "solver per-vertex `fixed` flags (cantilever idealization)",
        "load_model": "end-vertex mass increased by the load (m_load * g)",
    }

    if args.section == "stability":
        # Largest E / K this integrator can carry at the configured dt and
        # substeps.  A calibration target above this ceiling cannot be
        # represented without changing dt/substeps -- that is a modelling
        # limit worth knowing before fitting, not a fit failure.
        ladder = [1.0e4, 3.0e4, 1.0e5, 3.0e5, 1.0e6, 3.0e6, 1.0e7, 3.0e7, 1.0e8]
        report = {}
        for name in ("stretch_K", "bend_E"):
            rows = []
            for value in ladder:
                kw = {"bend": None, "stretch": None, "twist": None}
                kw["stretch" if name == "stretch_K" else "bend"] = value
                apply_stiffness(kw["bend"], kw["stretch"], kw["twist"])
                ok, note = True, None
                try:
                    e = make_env(args.gamma_lin, args.gamma_ang, args.dt, args.substeps)
                    e.reset(params, init_shape="straight", seed=0)
                    line = np.zeros((int(params.n_segments), 3))
                    line[:, 0] = np.linspace(-0.5, 0.5, int(params.n_segments))
                    line[:, 2] = 0.4
                    place(e, line)
                    set_fixed(e, np.array([0]))
                    relax(e, args.threshold, 800, fixed_idx=np.array([0]))
                    e._assert_finite()
                    del e
                except Exception as error:  # noqa: BLE001 - probe records any failure
                    ok, note = False, f"{type(error).__name__}: {error}"[:120]
                rows.append({"value": value, "stable": ok, "note": note})
            stable = [r["value"] for r in rows if r["stable"]]
            report[name] = {"ladder": rows,
                            "max_stable": max(stable) if stable else None}
        payload["stability"] = report
        apply_stiffness(None, None, None)
    elif args.section == "fit":
        targets = {
            "droop_overhang_mm": args.target_droop_overhang_mm,
            "droop_angle_deg": args.droop_angle_deg,
            "elongation_mm": ([float(v) for v in args.target_elongation_mm.split(",")]
                              if args.target_elongation_mm else None),
            "loads_kg": loads,
            "settle_time_s": args.target_settle_time_s,
        }
        payload["fit"] = run_fit(args, params, targets)
        fitted = payload["fit"]["fitted"]
        apply_stiffness(fitted["bend_E"], fitted["stretch_K"], fitted["twist_G"])
        env = make_env(fitted["gamma_lin"], fitted["gamma_ang"], args.dt, args.substeps)
        env.reset(params, init_shape="straight", seed=0)
        payload["verification"] = {
            "A": test_a_droop(env, params, overhangs, args.desk_z, args.threshold),
            "B": None,  # filled below with the relaxation env
            "C": test_c_drop(env, params, args.drop_lift, args.threshold),
        }
        benv = make_env(max(fitted["gamma_lin"], args.hang_relax_damping), args.gamma_ang, args.dt, args.substeps)
        benv.reset(params, init_shape="straight", seed=0)
        payload["verification"]["B"] = test_b_hang(
            benv, params, loads, args.hang_top_z, args.threshold
        )
        del benv
        payload["verification"]["A_overhang_for_target_angle_m"] = (
            interpolate_overhang_for_angle(payload["verification"]["A"],
                                           args.droop_angle_deg, args.droop_metric)
        )
        del env
    else:
        payload["stiffness"] = apply_stiffness(args.bend_e, args.stretch_k, args.twist_g)
        env = make_env(args.gamma_lin, args.gamma_ang, args.dt, args.substeps)
        env.reset(params, init_shape="straight", seed=0)
        if args.section in ("all", "A"):
            payload["A"] = test_a_droop(env, params, overhangs, args.desk_z, args.threshold)
            payload["A_overhang_for_target_angle_m"] = interpolate_overhang_for_angle(
                payload["A"], args.droop_angle_deg, args.droop_metric
            )
        if args.section in ("all", "B"):
            benv = make_env(max(args.gamma_lin, args.hang_relax_damping), args.gamma_ang, args.dt, args.substeps)
            benv.reset(params, init_shape="straight", seed=0)
            payload["B"] = test_b_hang(benv, params, loads, args.hang_top_z, args.threshold)
            payload["B_relax_damping"] = max(args.gamma_lin, args.hang_relax_damping)
            del benv
        if args.section in ("all", "C"):
            payload["C"] = test_c_drop(env, params, args.drop_lift, args.threshold)
        del env

    payload["elapsed_s"] = round(time.time() - started, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items()
                      if k not in ("A", "B", "verification", "fit")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
