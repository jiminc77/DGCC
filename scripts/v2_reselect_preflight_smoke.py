#!/usr/bin/env python
"""Bounded GPU smoke for the reselection preflight (task D-8).

Proves four things and nothing else:

  (i)   the EFFECTIVE environment is the corrected one (logged, then
        re-asserted on the constructed adapter);
  (ii)  the Rev 3 tension guard is on the executed code path;
  (iii) the AT-1H per-episode counter file is produced;
  (iv)  the n_envs=4096 training loop runs: finite losses, zero NaN /
        magnitude covenant incidents, gradient updates actually fire.

EXPLICITLY NOT claimed and NOT produced:
  * no performance / learning claim of any kind,
  * no checkpoint retained (`--rounds` stops before eval, and the receipt
    asserts the models dir stayed empty),
  * no budget consumption -- this run is scratch and its transitions are
    not part of any pre-registered run,
  * heldout is never touched: T2 training draws from the `train` split and
    this smoke performs no evaluation at all.

Usage:
    python scripts/v2_reselect_preflight_smoke.py --rounds 4 --out receipt.json
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "v2_t2.yaml")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--n-envs-override",
        type=int,
        default=None,
        help="VALIDATION ONLY: run at a different n_envs (e.g. on the CPU "
             "backend while the GPU is occupied). Setting this stamps the "
             "receipt dry_run=true so it can never be mistaken for the "
             "canonical n_envs=4096 evidence.",
    )
    args = parser.parse_args()

    import torch

    import p1_train as trainer  # noqa: E402  (path is set above)

    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes.decode("utf-8"))
    n_envs = int(config["run"]["n_envs"])
    dry_run = args.n_envs_override is not None
    if dry_run:
        n_envs = int(args.n_envs_override)
        config = {**config, "run": {**config["run"], "n_envs": n_envs}}
    warmup = int(config["td3"]["warmup_transitions"])

    receipt: dict[str, Any] = {
        "generated_at": utc_now(),
        "kind": "reselection-preflight-smoke",
        "dry_run": dry_run,
        "not_admissible_as": [
            "performance evidence",
            "learning-dynamics evidence",
            "budget consumption",
            "checkpointed run",
        ],
        "config_path": str(args.config),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "n_envs": n_envs,
        "warmup_transitions": warmup,
        "rounds_requested": int(args.rounds),
        "seed": int(args.seed),
    }

    run_args = argparse.Namespace(
        config=args.config,
        seed=int(args.seed),
        run_tag=f"reselect_smoke_s{args.seed}",
        total_override=n_envs * int(args.rounds),
        device=args.device,
        asset_manifest=None,
        expected_asset_manifest_sha256=None,
        allow_legacy_env=False,
    )

    build_log = io.StringIO()
    t0 = time.perf_counter()
    run = trainer.TrainingRun(run_args, None, config=config)
    with contextlib.redirect_stdout(build_log):
        run.build_scene()
    receipt["build_s"] = round(time.perf_counter() - t0, 2)
    env_param_log = [
        line for line in build_log.getvalue().splitlines() if line.startswith("env-params:")
    ]
    print("\n".join(env_param_log), flush=True)

    env = run.env
    # (i) effective environment
    receipt["effective_env"] = {
        "logged_lines": env_param_log,
        "quasi_static": bool(env.quasi_static),
        "move_v_max": float(env.move_v_max),
        "move_hold_max_steps": int(env.move_hold_max_steps),
        "derived_step_m_per_sim_step": float(env.move_v_max) * float(env.dt),
        "reset_settle_max_steps": int(env.reset_settle_max_steps),
        "grasp_realism": bool(env.grasp_realism),
        "at1h_counters": bool(env.at1h_counters),
        "legacy_move_step_size_inactive": env.move_v_max is not None,
        "resolved_kwargs": run.effective_env_kwargs,
    }
    # (ii) tension guard: the Rev 3 guard block is inside `if self.quasi_static`
    # in `_execute_move`, so quasi_static == True is the structural proof that
    # the guarded walk (not the legacy `np.linspace` walk) is the executed
    # path.  Observed counters are reported factually below.
    from dgcc.envs.dlolab import (
        HOLD_QUIESCENT_VEL,
        LOWER_STRAIN_ABORT,
        TENSION_PAUSE_MAX_STEPS,
    )

    receipt["tension_guard"] = {
        "reachable_iff_quasi_static": True,
        "quasi_static": bool(env.quasi_static),
        "strain_threshold": LOWER_STRAIN_ABORT,
        "pause_budget_steps": TENSION_PAUSE_MAX_STEPS,
        "hold_quiescent_vel": HOLD_QUIESCENT_VEL,
    }

    rounds: list[dict[str, Any]] = []
    pause_steps_total = 0
    freezes_total = 0
    lower_aborts_total = 0
    update_count_before = run.agent.update_count
    for i in range(int(args.rounds)):
        r0 = time.perf_counter()
        count = run.collect_round()
        collect_s = time.perf_counter() - r0
        u0 = time.perf_counter()
        run.train_updates(count)
        update_s = time.perf_counter() - u0
        info = getattr(run.runner.env, "last_tension_pause_steps", 0)
        pause_steps_total += int(info)
        freezes_total += int(getattr(run.runner.env, "last_tension_freezes", 0))
        lower_aborts_total += int(getattr(run.runner.env, "last_lower_strain_aborts", 0))
        rounds.append({
            "round": i + 1,
            "transitions_added": int(count),
            "transitions_total": int(run.transitions),
            "collect_s": round(collect_s, 2),
            "update_s": round(update_s, 2),
            "updates_so_far": int(run.agent.update_count),
            "nan_incidents": int(run.runner.nan_incidents),
            "magnitude_incidents": int(run.runner.magnitude_incidents),
            "tension_pause_steps": int(getattr(run.runner.env, "last_tension_pause_steps", 0)),
            "tension_freezes": int(getattr(run.runner.env, "last_tension_freezes", 0)),
            "lower_strain_aborts": int(getattr(run.runner.env, "last_lower_strain_aborts", 0)),
            "hold_steps_used": int(getattr(run.runner.env, "last_hold_steps_used", 0)),
            "hold_converged": getattr(run.runner.env, "last_hold_converged", None),
        })
        print(json.dumps(rounds[-1]), flush=True)

    receipt["rounds"] = rounds
    receipt["tension_guard"]["observed_pause_steps"] = pause_steps_total
    receipt["tension_guard"]["observed_freezes"] = freezes_total
    receipt["tension_guard"]["observed_lower_strain_aborts"] = lower_aborts_total

    # (iv) loop health
    # `DiagnosticsLogger.history()` returns lists of flat float dicts; every
    # scalar the TD3 update reported must be finite (this is the NaN=0 /
    # finite-loss claim, taken from what the run actually logged rather than
    # from a re-computation).
    history = run.diag.history()
    losses: list[float] = []
    scalar_names: set[str] = set()
    for row in history.get("updates", []):
        for name, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                losses.append(float(value))
                scalar_names.add(name)
    for row in history.get("replay", []):
        for name, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                losses.append(float(value))
                scalar_names.add(name)
    receipt["loop_health"] = {
        "gradient_updates": int(run.agent.update_count - update_count_before),
        "gradient_updates_expected_nonzero": int(run.transitions) >= warmup,
        "nan_incidents": int(run.runner.nan_incidents),
        "magnitude_incidents": int(run.runner.magnitude_incidents),
        "full_scene_rebuilds": int(run.full_rebuilds),
        "logged_scalars_checked": len(losses),
        "logged_scalar_names": sorted(scalar_names),
        "all_logged_scalars_finite": all(math.isfinite(v) for v in losses),
        "replay_size": int(run.buffer.size),
        "replay_reward_finite": bool(
            np.all(np.isfinite(run.buffer.reward[: run.buffer.size]))
        ),
    }

    # (iii) AT-1H counter file
    # A short smoke almost always ends mid-episode (horizon is 10 primitives),
    # so the in-flight accumulators are flushed exactly as a real halt/eval
    # boundary would flush them -- `episode_complete=false` marks them.
    open_rows = run.flush_at1h_open_episodes()
    at1h_path = run._at1h_path
    at1h_rows: list[dict[str, Any]] = []
    if at1h_path.is_file():
        for line in at1h_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                at1h_rows.append(json.loads(line))
    receipt["at1h"] = {
        "counters_file": str(at1h_path),
        "file_exists": at1h_path.is_file(),
        "episode_rows": len(at1h_rows),
        "rows_from_completed_episodes": sum(
            1 for r in at1h_rows if r.get("episode_complete")
        ),
        "rows_flushed_in_flight": open_rows,
        "sample_rows": at1h_rows[:3],
        "run_summary": run.at1h_run_summary(),
        "note": (
            "complete rows are emitted when an episode terminates (horizon 10 "
            "primitives); in-flight accumulators are flushed with "
            "episode_complete=false so no measured primitive is discarded"
        ),
    }

    models = sorted(p.name for p in run.models_dir.glob("*"))
    receipt["no_checkpoint_retained"] = {"models_dir": str(run.models_dir), "entries": models}

    checks = {
        "i_corrected_env": bool(env.quasi_static) and float(env.move_v_max) == 0.15,
        "ii_tension_guard_on_path": bool(env.quasi_static),
        "iii_at1h_counters_live": (
            receipt["at1h"]["run_summary"]["primitives"] > 0
            and receipt["at1h"]["file_exists"]
            and receipt["at1h"]["episode_rows"] > 0
        ),
        "iv_loop_healthy": (
            receipt["loop_health"]["nan_incidents"] == 0
            and receipt["loop_health"]["magnitude_incidents"] == 0
            and receipt["loop_health"]["all_logged_scalars_finite"]
            and receipt["loop_health"]["replay_reward_finite"]
            and (
                receipt["loop_health"]["gradient_updates"] > 0
                or not receipt["loop_health"]["gradient_updates_expected_nonzero"]
            )
        ),
        "no_checkpoint": not models,
    }
    receipt["checks"] = checks
    verdict = "PASS" if all(checks.values()) else "FAIL"
    receipt["verdict"] = f"{verdict} (DRY-RUN, NOT canonical evidence)" if dry_run else verdict
    receipt["wall_s"] = round(time.perf_counter() - t0, 2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"verdict": receipt["verdict"], "checks": checks}, indent=2), flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
