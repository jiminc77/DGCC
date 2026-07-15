"""Shared episode-batch evaluation for P1 (§7 eval protocol + §8 gap data).

Used by the M2 random-policy reference, the training driver's deterministic
evals (every 25k transitions), and later M3/M4 reports.  Returns per-episode
arrays so the §3 statistical gate register (episode-level / goal-level
bootstraps) can be computed from logged data without re-simulation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from dgcc.goals.distance import canonical_shape_flip, flip_consistent_shape_distance
from dgcc.goals.dual_goal import DualGoal, goal_curve
from dgcc.tasks.domain import P1_LENGTH_M
from dgcc.tasks.episode import BatchedEpisodeRunner

ActionFn = Callable[[np.ndarray, np.ndarray, np.random.Generator], tuple[np.ndarray, np.ndarray, list[str]]]
QMinFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def evaluate_episodes(
    runner: BatchedEpisodeRunner,
    *,
    n_episodes: int,
    seed: int,
    episode_index_start: int,
    action_fn: ActionFn,
    rng: np.random.Generator,
    gamma: float = 0.95,
    goals: Sequence[DualGoal] | None = None,
    goal_fn: Any | None = None,
    goal_labels: Sequence[str] | None = None,
    q_min_fn: QMinFn | None = None,
) -> dict[str, Any]:
    """Run ``n_episodes`` evaluation episodes in batches over the runner's env.

    ``goals``: fixed per-episode goal list of length >= n_episodes (T2 style);
    ``goal_fn``: state-dependent sampler (T1 style).  Exactly one required.
    ``q_min_fn``: optional Q(s, a) evaluator for the §8 overestimation gap
    (recorded at each episode's FIRST executed action vs the realized
    discounted return).
    """

    if (goals is None) == (goal_fn is None):
        raise ValueError("provide exactly one of goals or goal_fn")

    n_envs = runner.n_envs
    batches = math.ceil(n_episodes / n_envs)
    episodes: list[dict[str, Any]] = []
    incidents_before = runner.nan_incidents

    for batch_index in range(batches):
        episode_base = batch_index * n_envs
        if goals is not None:
            batch_goals = [
                goals[(episode_base + slot) % len(goals)] for slot in range(n_envs)
            ]
            begin_info = runner.begin_episodes(
                seed=seed,
                episode_index=episode_index_start + batch_index,
                goals=batch_goals,
            )
        else:
            begin_info = runner.begin_episodes(
                seed=seed,
                episode_index=episode_index_start + batch_index,
                goal_fn=goal_fn,
            )
        goal_curves = np.stack([goal_curve(g, P1_LENGTH_M) for g in runner.goals])

        # P2b D_shape (M4, observation-only — risk #2): initial centerlines
        # captured immediately after begin_episodes; the orientation flip is
        # decided ONCE per episode from the initial state and applied
        # identically to the initial and terminal measurements.
        X_initial = np.asarray(runner.env.get_centerline_batch(), dtype=float)
        shape_flips = [
            canonical_shape_flip(X_initial[slot], runner.goals[slot], P1_LENGTH_M)
            for slot in range(n_envs)
        ]
        # Terminal centerline per episode: last X_after row observed while the
        # slot was still ACTIVE (same lifecycle hook as d_at_done) — never the
        # post-terminal batch state of early-finished environments.
        x_terminal: list[np.ndarray | None] = [None] * n_envs

        returns = np.zeros(n_envs, dtype=float)
        discounted = np.zeros(n_envs, dtype=float)
        q_first = np.full(n_envs, np.nan, dtype=float)
        d_step_traces: list[list[float]] = [[] for _ in range(n_envs)]
        step_index = 0
        while not runner.all_done() and step_index < 2 * runner.config.horizon:
            X = runner.env.get_centerline_batch()
            p, delta, lift = action_fn(X, goal_curves, rng)
            if q_min_fn is not None and step_index == 0:
                lift_num = np.asarray([1 if v == "high" else 0 for v in lift])
                q_first = np.asarray(q_min_fn(X, goal_curves, p, delta, lift_num), dtype=float)
            record = runner.step(p, delta, lift, rng=rng)
            if record.get("discarded"):
                continue
            active = record["active"]
            for slot in np.flatnonzero(active):
                d_step_traces[int(slot)].append(float(record["d_after"][int(slot)]))
                x_terminal[int(slot)] = np.asarray(record["X_after"][int(slot)], dtype=float)
            returns += np.where(active, record["reward"], 0.0)
            discounted += np.where(active, (gamma**step_index) * record["reward"], 0.0)
            step_index += 1

        for slot in range(n_envs):
            episode_id = episode_base + slot
            if episode_id >= n_episodes:
                continue
            final_d = float(runner.d_current[slot])
            d_at_done = float(runner.d_at_done[slot])
            d_at_done_fallback = not np.isfinite(d_at_done)
            if d_at_done_fallback:
                d_at_done = final_d
            d_steps = [float(v) for v in d_step_traces[slot][: runner.config.horizon]]
            min_d = float(np.min(d_steps)) if d_steps else final_d
            flip = shape_flips[slot]
            d_shape_initial = flip_consistent_shape_distance(
                X_initial[slot], goal_curves[slot], P1_LENGTH_M, flip=flip
            )
            terminal = x_terminal[slot]
            d_shape_at_done = (
                d_shape_initial
                if terminal is None
                else flip_consistent_shape_distance(
                    terminal, goal_curves[slot], P1_LENGTH_M, flip=flip
                )
            )
            episodes.append(
                {
                    "episode_id": episode_id,
                    "goal_label": (
                        goal_labels[episode_id % len(goal_labels)]
                        if goal_labels is not None
                        else None
                    ),
                    "init_template": begin_info["init_shapes"][slot],
                    "success": bool(runner.succeeded[slot]),
                    "steps": int(runner.t[slot]),
                    "return": float(returns[slot]),
                    "discounted_return": float(discounted[slot]),
                    "final_d": float(runner.d_current[slot]),
                    "d_at_done": d_at_done,
                    "d_at_done_fallback": bool(d_at_done_fallback),
                    "d_steps": d_steps,
                    "min_d": min_d,
                    "d_initial": float(begin_info["d_initial"][slot]),
                    "d_shape_initial": float(d_shape_initial),
                    "d_shape_at_done": float(d_shape_at_done),
                    "q_first": None if np.isnan(q_first[slot]) else float(q_first[slot]),
                }
            )

    return summarize_episodes(episodes) | {
        "episodes": episodes,
        "nan_incidents_during_eval": runner.nan_incidents - incidents_before,
    }


def summarize_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-episode rows into §8 report scalars."""

    success = np.asarray([ep["success"] for ep in episodes], dtype=bool)
    returns = np.asarray([ep["return"] for ep in episodes], dtype=float)
    final_d = np.asarray([ep["final_d"] for ep in episodes], dtype=float)
    d_at_done = np.asarray([ep["d_at_done"] for ep in episodes], dtype=float)
    min_d = np.asarray([ep["min_d"] for ep in episodes], dtype=float)
    # P2b D_shape rows are absent in pre-M4 logged episodes; guard for reuse
    # of this summarizer over historical data.
    d_shape_rows = [ep["d_shape_at_done"] for ep in episodes if "d_shape_at_done" in ep]

    per_template: dict[str, float] = {}
    per_template_n: dict[str, int] = {}
    for template in sorted({ep["init_template"] for ep in episodes}):
        mask = np.asarray([ep["init_template"] == template for ep in episodes], dtype=bool)
        per_template[template] = float(success[mask].mean()) if mask.any() else float("nan")
        per_template_n[template] = int(mask.sum())

    gaps = [
        ep["q_first"] - ep["discounted_return"]
        for ep in episodes
        if ep["q_first"] is not None
    ]
    return {
        "n_episodes": len(episodes),
        "success_rate": float(success.mean()) if episodes else float("nan"),
        "mean_return": float(returns.mean()) if episodes else float("nan"),
        "mean_final_d": float(final_d.mean()) if episodes else float("nan"),
        "mean_d_at_done": float(d_at_done.mean()) if episodes else float("nan"),
        "mean_min_d": float(min_d.mean()) if episodes else float("nan"),
        "mean_d_shape_at_done": float(np.mean(d_shape_rows)) if d_shape_rows else None,
        "per_template_success": per_template,
        "per_template_episodes": per_template_n,
        "overestimation_gap_mean": float(np.mean(gaps)) if gaps else None,
        "overestimation_gap_p95": float(np.quantile(gaps, 0.95)) if gaps else None,
    }


__all__ = ["evaluate_episodes", "summarize_episodes"]
