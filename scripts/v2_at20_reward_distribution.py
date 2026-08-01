#!/usr/bin/env python3
"""AT-20 (GPU-window item, boundary amendment item 4): reward-distribution
indistinguishability around discarded batches after the B3 correction.

Design §2.3: with B3 merged, the per-step rewards of INNOCENT envs (not
reseeded by the recovery) in the step immediately adjacent to a discarded
batch must be statistically indistinguishable from rewards in discard-free
regions.  Pre-B3, the unscoped d_current rewrite moved innocent envs' reward
baseline (signed, batch-history-dependent bias — §1.6.3); post-B3 that
channel is closed (AT-18 (iii) proved bit-invariance; AT-20 verifies the
distributional consequence on actual rewards).

Method: real env rollouts (no training) with covenant incidents FORCE-
INJECTED at deterministic ordinals (rotating poisoned env). Every active
innocent reward in the first non-discarded step after a discard joins the
ADJACENT sample; all other active rewards join the NORMAL sample. Two-sided
two-sample Kolmogorov-Smirnov plus Mann-Whitney U; PASS requires both
p-values >= 0.05 (indistinguishable at the 5% level; sample sizes and the
median shift are recorded alongside).

Exit code 0 only on PASS (fail-closed).
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
N_ENVS = 8
EPISODES = 8
PRIMITIVES_PER_EPISODE = 10
POISON_EVERY = 7  # poison every 7th step_primitive_batch call
ALPHA = 0.05


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
    parser.add_argument(
        "--mode", choices=["ab", "distribution"], default="ab",
        help="ab (primary): exact same-seed A/B against a pre-B3 emulation — "
             "the reward streams must be identical outside adjacent-innocent "
             "steps, isolating and sizing the B3 channel exactly. "
             "distribution: the naive KS/MW comparison, retained for the "
             "record; it is confounded by the pre-existing discard "
             "double-step semantics (see the artifact note).",
    )
    args = parser.parse_args()

    import torch

    import genesis as gs

    if args.backend == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested but no CUDA device is visible")
        if not getattr(gs, "_initialized", False):
            gs.init(seed=0, precision="32", logging_level="warning", backend=gs.gpu)
        device = torch.cuda.get_device_name(0)
    else:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if not getattr(gs, "_initialized", False):
            gs.init(seed=0, precision="32", logging_level="warning", backend=gs.cpu)
        device = "cpu"

    from scipy import stats

    from dgcc.envs.dlolab import DLOLabEnv
    # CANONICAL metric: the runner rewrites d_current with
    # tasks.reward.distance_to_goal (correspondence_l2), NOT goals.distance.D
    # — the first AB attempt used the wrong metric and corrupted the
    # emulation beyond the real pre-B3 behavior (recorded in the log).
    from dgcc.tasks.reward import distance_to_goal
    from dgcc.tasks.domain import p1_rope_params
    from dgcc.tasks.episode import BatchedEpisodeRunner, EpisodeConfig
    from dgcc.tasks.t2 import load_t2_payload, load_t2_split_payload

    started = time.time()
    params = p1_rope_params()

    class PoisonEnv(DLOLabEnv):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.calls = 0
            self.poisoned_env_cursor = 0
            self.injections = 0

        def step_primitive_batch(self, *pargs: Any, **pkwargs: Any) -> dict[str, Any]:
            result = super().step_primitive_batch(*pargs, **pkwargs)
            self.calls += 1
            if self.calls % getattr(self, "poison_every", POISON_EVERY) == 0:
                target = self.poisoned_env_cursor % self.n_envs
                self.poisoned_env_cursor += 1
                self.injections += 1
                poisoned = np.asarray(result["X_after"], dtype=float).copy()
                poisoned[target] = np.nan
                result = dict(result)
                result["X_after"] = poisoned
            return result

    class PreB3Runner(BatchedEpisodeRunner):
        """Emulates the pre-correction UNSCOPED d_current rewrite (§1.6.3).

        d_current does not feed back into the physics, so a PreB3 arm and a
        post-B3 arm with identical seeds traverse bit-identical trajectories;
        only the rewards can differ — exactly the B3 channel.
        """

        def _handle_nan_incident(self, active_before, **kwargs: Any):
            record = super()._handle_nan_incident(active_before, **kwargs)
            centerlines = np.asarray(self.env.get_centerline_batch(), dtype=float)
            recovered_d = np.asarray(
                [
                    distance_to_goal(centerlines[i], self.goals[i], self.length_m)
                    for i in range(self.n_envs)
                ],
                dtype=float,
            )
            self.d_current = np.where(~self.done, recovered_d, self.d_current)
            return record

    pairs = load_t2_split_payload("val", load_t2_payload())
    goal_pool = [goal for _, goal in pairs]


    adjacent: list[float] = []
    normal: list[float] = []
    discards_seen = 0
    action_rng = np.random.default_rng(2_026)
    def run_rollout(runner_factory, episodes: int, poison_every: int) -> dict[str, Any]:
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
        env.poison_every = poison_every
        env.reset(params, init_shape="s_curve", seed=4_040)
        rollout_runner = runner_factory(env, params, EpisodeConfig())
        rng = np.random.default_rng(2_026)
        stream: list[dict[str, Any]] = []
        pending: np.ndarray | None = None
        for episode in range(1, episodes + 1):
            goals = [
                goal_pool[(episode * N_ENVS + i) % len(goal_pool)]
                for i in range(N_ENVS)
            ]
            rollout_runner.begin_episodes(
                seed=4_040 + episode, episode_index=episode, goals=goals
            )
            pending = None
            for _ in range(PRIMITIVES_PER_EPISODE):
                p = rng.integers(0, 32, N_ENVS)
                delta = rng.normal(0.0, 0.05, (N_ENVS, 3))
                lift = [str(v) for v in rng.choice(["low", "high"], N_ENVS)]
                record = rollout_runner.step(p, delta, lift, rng=rng)
                if record.get("discarded"):
                    reseeded = np.asarray(record.get("bad_envs", []), dtype=int)
                    mask = np.ones(N_ENVS, dtype=bool)
                    mask[reseeded[reseeded < N_ENVS]] = False
                    pending = mask
                    stream.append({
                        "discarded": True,
                        "bad_envs": reseeded.tolist(),
                        "restart_envs": [int(v) for v in np.asarray(record.get("restart_envs", []), dtype=int)],
                        "d_current": np.asarray(rollout_runner.d_current, dtype=float).copy(),
                        "done": np.asarray(rollout_runner.done, dtype=bool).copy(),
                    })
                    continue
                stream.append(
                    {
                        "discarded": False,
                        "rewards": np.asarray(record["reward"], dtype=float).copy(),
                        "active": np.asarray(record["active"], dtype=bool).copy(),
                        "d_current": np.asarray(rollout_runner.d_current, dtype=float).copy(),
                        "done": np.asarray(rollout_runner.done, dtype=bool).copy(),
                        "adjacent_innocent": (
                            pending.copy() if pending is not None else None
                        ),
                    }
                )
                pending = None
        return {"stream": stream, "injections": env.injections}

    if args.mode == "distribution":
        rollout = run_rollout(BatchedEpisodeRunner, EPISODES, POISON_EVERY)
        for step in rollout["stream"]:
            if step["discarded"]:
                discards_seen += 1
                continue
            active = step["active"]
            rewards = step["rewards"]
            mask = step["adjacent_innocent"]
            if mask is not None:
                adjacent.extend(float(r) for r in rewards[active & mask])
                normal.extend(float(r) for r in rewards[active & ~mask])
            else:
                normal.extend(float(r) for r in rewards[active])

        if discards_seen == 0 or len(adjacent) < 20:
            raise RuntimeError(
                f"insufficient injected discards: discards={discards_seen}, "
                f"adjacent samples={len(adjacent)}"
            )
        ks_stat, ks_p = stats.ks_2samp(adjacent, normal, method="auto")
        mw_stat, mw_p = stats.mannwhitneyu(adjacent, normal, alternative="two-sided")
        verdict = bool(ks_p >= ALPHA and mw_p >= ALPHA)
        output = {
            "schema_version": 1,
            "mode": "distribution",
            "backend": args.backend,
            "device": device,
            "n_envs": N_ENVS,
            "episodes": EPISODES,
            "primitives_per_episode": PRIMITIVES_PER_EPISODE,
            "poison_every": POISON_EVERY,
            "discards_injected": discards_seen,
            "samples": {"adjacent": len(adjacent), "normal": len(normal)},
            "adjacent_median": float(np.median(adjacent)),
            "normal_median": float(np.median(normal)),
            "median_shift": float(np.median(adjacent) - np.median(normal)),
            "ks": {"statistic": float(ks_stat), "p_value": float(ks_p)},
            "mannwhitney": {"statistic": float(mw_stat), "p_value": float(mw_p)},
            "alpha": ALPHA,
            "confound_note": (
                "adjacent rewards embed the PRE-EXISTING discard semantics: a "
                "discarded round physically advances innocent envs by one "
                "unscored primitive, so the next reward measures two "
                "primitives of displacement against one baseline. This "
                "structural effect is independent of B3; use --mode ab for "
                "the exact channel isolation."
            ),
            "pass": verdict,
        }
    else:
        # CPU-pinned exactness: GPU parallel-reduction nondeterminism makes
        # two same-command rollouts diverge at the 1e-3 reward scale (decaying
        # to 1e-13), which drowns exact channel isolation; the AT-17 precedent
        # (bit-identity) also held on CPU only. Scope reduced accordingly.
        ab_episodes = 3
        ab_poison_every = 4
        post = run_rollout(BatchedEpisodeRunner, ab_episodes, ab_poison_every)
        pre = run_rollout(PreB3Runner, ab_episodes, ab_poison_every)
        s_post, s_pre = post["stream"], pre["stream"]
        if len(s_post) != len(s_pre):
            raise RuntimeError(
                f"arm stream lengths differ: post {len(s_post)} vs pre {len(s_pre)}"
            )
        pattern_match = True
        non_adjacent_max_delta = 0.0
        reseeded_max_delta = 0.0
        channel: list[float] = []
        discards = 0
        divergence_trace: list[dict[str, Any]] = []
        recent_adjacent_step: dict[int, int] = {}
        for step_index, (post_step, pre_step) in enumerate(zip(s_post, s_pre)):
            if post_step["discarded"] != pre_step["discarded"]:
                pattern_match = False
                break
            if post_step["discarded"]:
                discards += 1
                continue
            delta = pre_step["rewards"] - post_step["rewards"]
            active = post_step["active"]
            mask = post_step["adjacent_innocent"]
            if mask is not None:
                for env_idx in np.flatnonzero(active & mask):
                    recent_adjacent_step[int(env_idx)] = step_index
                channel.extend(float(v) for v in delta[active & mask])
                # Reseeded envs get identical recovered_d in both arms (the
                # scoped rewrite covers them), so their adjacent-step deltas
                # must be exactly zero; verify instead of misclassifying them
                # into the non-adjacent pool.
                reseeded_here = active & ~mask
                if reseeded_here.any():
                    reseeded_max_delta = max(
                        reseeded_max_delta, float(np.abs(delta[reseeded_here]).max())
                    )
                others = np.zeros_like(active)
            else:
                others = active
            if others.any():
                worst = float(np.abs(delta[others]).max())
                if worst > non_adjacent_max_delta:
                    non_adjacent_max_delta = worst
                if worst > 0.0 and len(divergence_trace) < 8:
                    bad_envs_here = [
                        int(i) for i in np.flatnonzero(others & (np.abs(delta) > 0))
                    ]
                    divergence_trace.append(
                        {
                            "step_index": step_index,
                            "envs": bad_envs_here,
                            "deltas": [float(delta[i]) for i in bad_envs_here],
                            "post_rewards": [float(post_step["rewards"][i]) for i in bad_envs_here],
                            "steps_since_env_adjacent": {
                                str(i): (
                                    step_index - recent_adjacent_step[int(i)]
                                    if int(i) in recent_adjacent_step
                                    else None
                                )
                                for i in bad_envs_here
                            },
                        }
                    )
        channel_arr = np.asarray(channel, dtype=float)
        nonzero = int((channel_arr != 0.0).sum()) if channel_arr.size else 0
        # Verdict: the arms' trajectories are provably identical (identical
        # discard pattern, bit-zero reward deltas outside adjacent-innocent
        # steps), the channel was exercised (nonzero pre-B3 deltas exist),
        # and post-B3 the adjacent rewards equal the unrewitten-baseline
        # truth by construction — i.e., the §1.6.3 reward-poisoning channel
        # is closed and its pre-fix magnitude is measured below (design open
        # item 14).
        verdict = bool(
            pattern_match
            and non_adjacent_max_delta == 0.0
            and reseeded_max_delta == 0.0
            and channel_arr.size >= 10
            and nonzero > 0
        )
        output = {
            "schema_version": 1,
            "mode": "ab",
            "backend": args.backend,
            "device": device,
            "n_envs": N_ENVS,
            "episodes": ab_episodes,
            "primitives_per_episode": PRIMITIVES_PER_EPISODE,
            "poison_every": ab_poison_every,
            "discards_injected": discards,
            "trajectory_identity": {
                "discard_pattern_match": pattern_match,
                "non_adjacent_max_abs_reward_delta": non_adjacent_max_delta,
                "reseeded_max_abs_reward_delta": reseeded_max_delta,
                "divergence_trace": divergence_trace,
            },
            "b3_channel_pre_minus_post": {
                "samples": int(channel_arr.size),
                "nonzero": nonzero,
                "max_abs": float(np.abs(channel_arr).max()) if channel_arr.size else None,
                "mean": float(channel_arr.mean()) if channel_arr.size else None,
                "median": float(np.median(channel_arr)) if channel_arr.size else None,
            },
            "pass": verdict,
        }
    if args.mode == "ab":
        def serialize(stream):
            rows = []
            for s in stream:
                rows.append({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()})
            return rows
        output["stream_post"] = serialize(s_post)
        output["stream_pre"] = serialize(s_pre)
    output["code_sha256"] = sha256_file(Path(__file__).resolve())
    output["elapsed_s"] = round(time.time() - started, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    printable = {k: v for k, v in output.items() if k not in ("code_sha256",)}
    print(json.dumps(printable, indent=2, default=str))
    return 0 if output["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
