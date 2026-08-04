#!/usr/bin/env python3
"""Rev 10 calibration driver: anchor the sim rope to the owner's bench data.

Owner decision (option 1): anchor BENDING fully, take axial stiffness up to the
integrator's stability limit, and DISCLOSE the residual gap.  substeps stays 5
and the discretization goes to 64.

Stages
  1  bending E   weighted least squares over the four drape angles
  2  axial K     set below the measured stability ceiling, verified by sweep,
                 then the realized EA and the residual gap are measured
  3  damping     bisection on the 0.20 m drop-to-rest time
  4  re-pass     stage 1 repeated at the fitted K/damping (interaction check)
  5  disclosure  residual-gap table, including strain at the task load

Nothing here writes repository constants; it reports the values to adopt.
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

from v2_rope_calibration_harness import (  # noqa: E402
    apply_stiffness, make_env, relax, rope_params, set_fixed, set_vertex_masses,
    test_a_droop, test_b_hang, test_c_drop, verts,
)

GRAVITY = 9.81
METRIC = "chord_angle_exit_deg"


def drape_sse(env, params, overhangs, targets, sigmas, desk_z, threshold):
    rows = test_a_droop(env, params, overhangs, desk_z, threshold)
    sse = 0.0
    detail = []
    for r, t, sg in zip(rows, targets, sigmas):
        resid = r[METRIC] - t
        sse += (resid / sg) ** 2
        detail.append({
            "overhang_m": r["overhang_m"],
            "realized_overhang_m": r["realized_overhang_m"],
            "free_vertices": r["free_vertices"],
            "sim_deg": r[METRIC], "target_deg": t, "sigma_deg": sg,
            "residual_deg": resid, "within_uncertainty": bool(abs(resid) <= sg),
            "relax_converged": r["relax_converged"],
        })
    return sse, detail, rows


def fit_bending(args, params, bend_grid, stretch, gamma_lin, gamma_ang):
    overhangs = args.overhangs
    results = []
    for e in bend_grid:
        apply_stiffness(e, stretch, e * args.twist_ratio)
        env = make_env(gamma_lin, gamma_ang, args.dt, args.substeps)
        env.reset(params, init_shape="straight", seed=0)
        try:
            sse, detail, _ = drape_sse(env, params, overhangs, args.targets,
                                       args.sigmas, args.desk_z, args.threshold)
        except (FloatingPointError, RuntimeError, ValueError) as err:
            sse, detail = float("inf"), [{"error": f"{type(err).__name__}: {err}"[:140]}]
        del env
        results.append({"bend_E": float(e), "weighted_sse": float(sse), "points": detail})
        print(f"  E={e:.4g}  SSE={sse:.4f}", flush=True)
    best = min(results, key=lambda r: r["weighted_sse"])
    return best, results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets-json", type=Path,
                    default=Path("/home/simx2204/v2_research/dossier/real_rope_targets.json"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backend", default="gpu")
    ap.add_argument("--n-segments", type=int, default=64)
    ap.add_argument("--mass-kg", type=float, default=0.040)
    ap.add_argument("--dt", type=float, default=1.0e-3)
    ap.add_argument("--substeps", type=int, default=5)
    ap.add_argument("--desk-z", type=float, default=0.75)
    ap.add_argument("--threshold", type=float, default=1.0e-3)
    ap.add_argument("--twist-ratio", type=float, default=0.1,
                    help="G/E ratio; default keeps the shipped 1e4/1e5 = 0.1")
    ap.add_argument("--stability-ceiling-K", type=float, default=3.0e6)
    ap.add_argument("--k-safety", type=float, default=0.80)
    ap.add_argument("--gamma-ratio", type=float, default=0.5,
                    help="gamma_ang / gamma_lin; default keeps the shipped 5/10")
    ap.add_argument("--task-load-N", type=float, default=0.4)
    ap.add_argument("--stage", default="all")
    args = ap.parse_args()

    tj = json.loads(args.targets_json.read_text())
    keys = sorted(tj["drape_angles_deg"], key=lambda k: int(k))
    args.overhangs = [int(k) / 100.0 for k in keys]
    args.targets = [float(tj["drape_angles_deg"][k]) for k in keys]
    args.sigmas = [float(tj["drape_uncertainty_deg"][k]) for k in keys]
    ea_target = float(tj["elongation"]["EA_N"])
    loads = [float(v) for v in tj["elongation"]["loads_kg"]]
    settle_target = float(tj["drop_settle_s"])

    from dgcc.envs.dlolab import ensure_genesis_initialized
    ensure_genesis_initialized(0)
    params = rope_params(args.n_segments, args.mass_kg)
    started = time.time()
    out: dict = {
        "driver": "rev10-calibration", "targets": tj,
        "n_segments": args.n_segments, "mass_kg": args.mass_kg,
        "dt": args.dt, "substeps": args.substeps,
        "shipped": {"bend_E": 1.0e5, "stretch_K": 8.0e5, "twist_G": 1.0e4,
                    "gamma_lin": 10.0, "gamma_ang": 5.0},
    }

    # ---- stage 2 first: K is fixed by the stability ceiling, not by fitting.
    k_target = args.stability_ceiling_K * args.k_safety
    print(f"[stage 2] K candidate {k_target:.4g} (= {args.k_safety:.0%} of ceiling)", flush=True)
    k_rows = []
    k_final = None
    for factor in (1.0, 0.75, 0.5, 0.25):
        k = k_target * factor
        apply_stiffness(1.0e5, k, 1.0e4)
        ok, note, elong = True, None, None
        try:
            env = make_env(60.0, 30.0, args.dt, args.substeps)
            env.reset(params, init_shape="straight", seed=0)
            rows = test_b_hang(env, params, loads, 1.60, args.threshold)
            env._assert_finite()
            elong = rows
            del env
        except Exception as err:  # noqa: BLE001
            ok, note = False, f"{type(err).__name__}: {err}"[:140]
        k_rows.append({"K": float(k), "stable": ok, "note": note,
                       "hang": elong})
        print(f"  K={k:.4g} stable={ok}", flush=True)
        if ok and k_final is None:
            k_final = k
            break
    out["stage2_axial"] = {"candidate": k_target, "sweep": k_rows, "selected_K": k_final}
    if k_final is None:
        out["error"] = "no stable K found"
        args.out.write_text(json.dumps(out, indent=2) + "\n")
        return 1

    # ---- stage 1: bending
    print("[stage 1] bending grid", flush=True)
    grid = [1.0e4, 3.0e4, 1.0e5, 3.0e5, 1.0e6, 3.0e6, 1.0e7]
    best, all_rows = fit_bending(args, params, grid, k_final, 10.0, 5.0)
    lo_i = max(0, grid.index(best["bend_E"]) - 1)
    hi_i = min(len(grid) - 1, grid.index(best["bend_E"]) + 1)
    refine = list(np.geomspace(grid[lo_i], grid[hi_i], 7))[1:-1]
    print("[stage 1] refine", flush=True)
    best_r, rows_r = fit_bending(args, params, refine, k_final, 10.0, 5.0)
    if best_r["weighted_sse"] < best["weighted_sse"]:
        best = best_r
    out["stage1_bending"] = {"grid": all_rows, "refine": rows_r, "best": best}
    e_final = best["bend_E"]

    # ---- stage 3: damping
    print("[stage 3] damping bisection", flush=True)
    lo, hi = 0.2, 40.0
    trace = []
    g_final = None
    for _ in range(11):
        mid = float(np.sqrt(lo * hi))
        apply_stiffness(e_final, k_final, e_final * args.twist_ratio)
        env = make_env(mid, mid * args.gamma_ratio, args.dt, args.substeps)
        env.reset(params, init_shape="straight", seed=0)
        try:
            c = test_c_drop(env, params, 0.20, args.threshold)
            t = c["settle_time_s"]
        except Exception as err:  # noqa: BLE001
            t = None
            c = {"error": f"{type(err).__name__}: {err}"[:140]}
        del env
        trace.append({"gamma_lin": mid, "settle_time_s": t, "detail": c})
        print(f"  gamma={mid:.3f} settle={t}", flush=True)
        if t is None:
            hi = mid
            continue
        if abs(t - settle_target) / settle_target <= 0.10:
            g_final = mid
            break
        if t > settle_target:
            lo = mid       # too slow -> more damping
        else:
            hi = mid
    if g_final is None:
        usable = [r for r in trace if r["settle_time_s"] is not None]
        g_final = min(usable, key=lambda r: abs(r["settle_time_s"] - settle_target))["gamma_lin"]
    out["stage3_damping"] = {"target_s": settle_target, "trace": trace,
                             "selected_gamma_lin": g_final,
                             "selected_gamma_ang": g_final * args.gamma_ratio}

    # ---- stage 4: bending re-pass at fitted K/gamma
    print("[stage 4] bending re-pass", flush=True)
    around = list(np.geomspace(e_final / 3.0, e_final * 3.0, 5))
    best2, rows2 = fit_bending(args, params, around, k_final, g_final,
                               g_final * args.gamma_ratio)
    out["stage4_repass"] = {"rows": rows2, "best": best2}
    if best2["weighted_sse"] < best["weighted_sse"]:
        e_final = best2["bend_E"]
        best = best2

    fitted = {"bend_E": e_final, "stretch_K": k_final,
              "twist_G": e_final * args.twist_ratio,
              "gamma_lin": g_final, "gamma_ang": g_final * args.gamma_ratio}
    out["fitted"] = fitted

    # ---- stage 5: verification + residual-gap disclosure
    print("[stage 5] verification", flush=True)
    apply_stiffness(fitted["bend_E"], fitted["stretch_K"], fitted["twist_G"])
    env = make_env(fitted["gamma_lin"], fitted["gamma_ang"], args.dt, args.substeps)
    env.reset(params, init_shape="straight", seed=0)
    sse, detail, a_rows = drape_sse(env, params, args.overhangs, args.targets,
                                    args.sigmas, args.desk_z, args.threshold)
    c_ver = test_c_drop(env, params, 0.20, args.threshold)
    del env
    benv = make_env(60.0, 30.0, args.dt, args.substeps)
    benv.reset(params, init_shape="straight", seed=0)
    b_rows = test_b_hang(benv, params, loads, 1.60, args.threshold)
    del benv

    one_kg = next((r for r in b_rows if abs(r["load_kg"] - 1.0) < 1e-9), None)
    sim_ea = None
    if one_kg and one_kg["elongation_mm"]:
        sim_ea = 1.0 * GRAVITY * float(params.length_m) / (one_kg["elongation_mm"] / 1000.0)
    task_strain_sim = args.task_load_N / sim_ea if sim_ea else None
    task_strain_real = args.task_load_N / ea_target
    out["verification"] = {
        "A": {"weighted_sse": sse, "points": detail, "rows": a_rows},
        "B": b_rows, "C": c_ver,
    }
    out["residual_gap"] = {
        "EA_real_N": ea_target,
        "EA_sim_N": sim_ea,
        "EA_ratio_sim_over_real": (sim_ea / ea_target) if sim_ea else None,
        "elongation_1kg_sim_mm": one_kg["elongation_mm"] if one_kg else None,
        "elongation_1kg_real_mm": 1.0 * GRAVITY * 1.0 / ea_target * 1000.0,
        "task_load_N": args.task_load_N,
        "task_strain_sim": task_strain_sim,
        "task_strain_real": task_strain_real,
        "task_strain_ratio": (task_strain_sim / task_strain_real) if task_strain_sim else None,
        "stability_ceiling_K": args.stability_ceiling_K,
        "K_required_for_real_EA": None,
    }
    if sim_ea:
        out["residual_gap"]["K_required_for_real_EA"] = k_final * ea_target / sim_ea

    out["elapsed_s"] = round(time.time() - started, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"fitted": fitted, "residual_gap": out["residual_gap"]},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
