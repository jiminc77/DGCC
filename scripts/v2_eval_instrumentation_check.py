#!/usr/bin/env python3
"""AT-18: eval-instrumentation integrity under forced covenant injection.

Design §5.5 (V2_env_correction_design.md Rev 2): with a covenant incident
FORCE-INJECTED into an eval (artificial non-finite X_after rows for chosen
envs at a chosen primitive ordinal), verify:

  (i)   B1 — the restarted slot's `return`/`discounted_return` reflect ONLY
        the post-restart attempt (accumulators reset, per-slot discount
        exponent restarts at the slot's own t=0);
  (ii)  B2 — `restart_steps` matches the actually injected restart step
        indices exactly, and `restart_count` agrees;
  (iii) B3 — the d_current of envs NOT reseeded by the recovery is
        bit-identical across the discarded round (direct regression test of
        the scoped rewrite).

Instrumentation is a Probe env subclass (poisons X_after rows on the chosen
call ordinal; production logic untouched) plus an audit runner subclass that
mirrors every step's rewards/active mask/restart set into a ledger the
verdicts are computed against.  CPU or GPU (backend recorded).

Exit code 0 only when all three items pass (fail-closed).
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
N_ENVS = 4
POISON_ENV = 2
POISON_CALL_ORDINAL = 3  # 1-based step_primitive_batch call to poison
GAMMA = 0.95


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
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    args = parser.parse_args()

    import torch

    import genesis as gs

    if args.backend == "cpu":
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if not getattr(gs, "_initialized", False):
            gs.init(seed=0, precision="32", logging_level="warning", backend=gs.cpu)
        device = "cpu"
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested but no CUDA device is visible")
        if not getattr(gs, "_initialized", False):
            gs.init(seed=0, precision="32", logging_level="warning", backend=gs.gpu)
        device = torch.cuda.get_device_name(0)

    from dgcc.envs.dlolab import DLOLabEnv
    from dgcc.rl.evaluation import evaluate_episodes
    from dgcc.tasks.domain import p1_rope_params
    from dgcc.tasks.episode import BatchedEpisodeRunner, EpisodeConfig
    from dgcc.tasks.t2 import load_t2_payload, load_t2_split_payload

    started = time.time()
    params = p1_rope_params()

    class PoisonEnv(DLOLabEnv):
        """Force one covenant incident: NaN X_after rows for POISON_ENV."""

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.calls = 0
            self.poison_armed = True

        def step_primitive_batch(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            result = super().step_primitive_batch(*args, **kwargs)
            self.calls += 1
            if self.poison_armed and self.calls == POISON_CALL_ORDINAL:
                self.poison_armed = False
                poisoned = np.asarray(result["X_after"], dtype=float).copy()
                poisoned[POISON_ENV] = np.nan
                result = dict(result)
                result["X_after"] = poisoned
            return result

    audit: dict[str, Any] = {
        "steps": [],  # per non-discarded step: rewards, active
        "restarts": [],  # (step_index, restart_envs)
        "b3": None,
    }

    class AuditRunner(BatchedEpisodeRunner):
        def step(self, p, delta, lift, *, rng=None):
            d_before_snapshot = self.d_current.copy()
            done_before = self.done.copy()
            record = super().step(p, delta, lift, rng=rng)
            if record.get("discarded"):
                step_ordinal = len(audit["steps"])
                restarted = np.asarray(
                    record.get("restart_envs", []), dtype=int
                )
                audit["restarts"].append(
                    {"step_index": step_ordinal, "restart_envs": restarted.tolist()}
                )
                reseeded = np.asarray(record.get("bad_envs", []), dtype=int)
                innocent = np.setdiff1d(np.arange(self.n_envs), reseeded)
                audit["b3"] = {
                    "reseeded_envs": reseeded.tolist(),
                    "innocent_envs": innocent.tolist(),
                    "d_current_unchanged": bool(
                        np.array_equal(
                            d_before_snapshot[innocent], self.d_current[innocent]
                        )
                    ),
                    "restarted_d_changed": bool(
                        not np.array_equal(
                            d_before_snapshot[restarted], self.d_current[restarted]
                        )
                    )
                    if restarted.size
                    else None,
                    "done_before": done_before.tolist(),
                }
            else:
                audit["steps"].append(
                    {
                        "rewards": np.asarray(record["reward"], dtype=float).tolist(),
                        "active": np.asarray(record["active"], dtype=bool).tolist(),
                    }
                )
            return record

    pairs = load_t2_split_payload("val", load_t2_payload())
    goals = [goal for _, goal in pairs[:N_ENVS]]

    env = PoisonEnv(
        n_envs=N_ENVS,
        dt=1.0e-3,
        substeps=5,
        rod_damping=10.0,
        rod_angular_damping=5.0,
        initial_settle_steps=0,
        reset_settle_max_steps=10_000,
        move_v_max=0.15,
        move_hold_max_steps=2000,
        grasp_realism=False,
    )
    env.reset(params, init_shape="s_curve", seed=2_024)
    runner = AuditRunner(env, params, EpisodeConfig())

    def action_fn(X, G, rng):
        p = rng.integers(0, 32, N_ENVS)
        delta = rng.normal(0.0, 0.05, (N_ENVS, 3))
        lift = [str(v) for v in rng.choice(["low", "high"], N_ENVS)]
        return p, delta, lift

    result = evaluate_episodes(
        runner,
        n_episodes=N_ENVS,
        seed=2_024,
        episode_index_start=1,
        action_fn=action_fn,
        rng=np.random.default_rng(11),
        gamma=GAMMA,
        goals=goals,
        record_raw=True,
    )
    episodes = result["episodes"]

    if audit["b3"] is None:
        raise RuntimeError("the forced covenant incident never fired")
    if not any(POISON_ENV in r["restart_envs"] for r in audit["restarts"]):
        raise RuntimeError(
            f"poisoned env {POISON_ENV} was never restarted: {audit['restarts']}"
        )

    # Expected per-slot return/discounted from the audit mirror: only steps
    # after the slot's LAST restart count, discounted from the slot's own 0.
    expected_return = np.zeros(N_ENVS)
    expected_discounted = np.zeros(N_ENVS)
    slot_counter = np.zeros(N_ENVS, dtype=int)
    restarts_by_step: dict[int, list[int]] = {}
    for entry in audit["restarts"]:
        restarts_by_step.setdefault(entry["step_index"], []).extend(
            entry["restart_envs"]
        )
    for index, step in enumerate(audit["steps"]):
        for slot in restarts_by_step.get(index, []):
            expected_return[slot] = 0.0
            expected_discounted[slot] = 0.0
            slot_counter[slot] = 0
        rewards = np.asarray(step["rewards"])
        active = np.asarray(step["active"], dtype=bool)
        expected_return += np.where(active, rewards, 0.0)
        expected_discounted += np.where(
            active, (GAMMA ** slot_counter) * rewards, 0.0
        )
        slot_counter += active.astype(int)
    # Note: restart resets attach to the FOLLOWING non-discarded step index,
    # matching evaluate_episodes' step_index semantics (discarded rounds do
    # not advance step_index).

    checks: dict[str, Any] = {}
    poisoned_row = next(ep for ep in episodes if ep["episode_id"] == POISON_ENV)
    expected_restarts = [
        entry["step_index"]
        for entry in audit["restarts"]
        if POISON_ENV in entry["restart_envs"]
    ]
    checks["ii_restart_steps"] = {
        "recorded": poisoned_row["restart_steps"],
        "expected": expected_restarts,
        "restart_count": poisoned_row["restart_count"],
        "pass": poisoned_row["restart_steps"] == expected_restarts
        and poisoned_row["restart_count"] == len(expected_restarts),
    }
    return_deltas = {}
    ok_returns = True
    for ep in episodes:
        slot = int(ep["episode_id"])
        delta_r = abs(float(ep["return"]) - float(expected_return[slot]))
        delta_d = abs(
            float(ep["discounted_return"]) - float(expected_discounted[slot])
        )
        return_deltas[slot] = {"return": delta_r, "discounted": delta_d}
        ok_returns &= delta_r < 1e-9 and delta_d < 1e-9
    checks["i_post_restart_only_returns"] = {
        "deltas": return_deltas,
        "pass": bool(ok_returns),
    }
    checks["iii_innocent_d_current_invariant"] = {
        **audit["b3"],
        "pass": bool(audit["b3"]["d_current_unchanged"]),
    }

    output = {
        "schema_version": 1,
        "backend": args.backend,
        "device": device,
        "n_envs": N_ENVS,
        "poison_env": POISON_ENV,
        "poison_call_ordinal": POISON_CALL_ORDINAL,
        "restarts_observed": audit["restarts"],
        "checks": checks,
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "elapsed_s": round(time.time() - started, 1),
    }
    output["pass"] = all(check["pass"] for check in checks.values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checks": checks, "pass": output["pass"]}, indent=2, default=str))
    return 0 if output["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
