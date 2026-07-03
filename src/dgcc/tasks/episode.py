"""Batched P1 episode wrapper over the DLO-Lab batch primitive API.

Contract (P1 §5, global rules 6-7):
    * Episodes run at most ``T = EPISODE_HORIZON = 10`` primitives and end
      early on success (``D < ε_succ``).  Re-grasping inside an episode is
      automatic (each primitive grasps anew).
    * Every settle-bearing simulator call made by this wrapper —
      ``light_reset`` and ``step_primitive_batch`` — passes
      ``vel_threshold=SETTLE_VEL_THRESHOLD`` and
      ``max_steps=SETTLE_MAX_STEPS`` (10000).  The DLOLabEnv per-call defaults
      (5000) are never relied upon.
    * ``DLOLabEnv.step_primitive`` (the single-env path) has NO ``max_steps``
      parameter and settles at 5000 — it is FORBIDDEN on all P1 collection
      paths and is never called here.
    * NaN covenant (env level): when the rope state goes non-finite, the
      affected transition batch is discarded, the offending environments are
      re-seeded with fresh init curves, and the incident counter increases
      (aggregated in the M6 report).

The wrapper is duck-typed over the environment: it requires ``n_envs``,
``get_centerline_batch()``, ``get_centerline_raw_batch()``,
``light_reset(vertices, *, vel_threshold, max_steps)`` and
``step_primitive_batch(p, delta, lift, *, vel_threshold, max_steps, rng)``,
which allows CPU-only fake environments in tests.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dgcc.envs.base import RopeParams
from dgcc.envs.dlolab import analytic_init_centerline
from dgcc.goals.dual_goal import DualGoal
from dgcc.tasks.domain import (
    EPISODE_HORIZON,
    INIT_SHAPES,
    RewardConstants,
    SETTLE_MAX_STEPS,
    SETTLE_VEL_THRESHOLD,
)
from dgcc.tasks.reward import distance_to_goal, step_reward

GoalFn = Callable[[int, np.ndarray, np.random.Generator], DualGoal]

#: Max reseed retries before a non-finite state becomes a hard failure.
MAX_RESEED_ATTEMPTS = 3


def is_nonfinite_error(exc: BaseException) -> bool:
    """Classify NaN-covenant exceptions from the simulator layer.

    The adapter surfaces non-finite rope state as ``FloatingPointError``
    (finiteness asserts) or ``ValueError`` whose message mentions
    ``non-finite`` (placement input validation on the failed-grasp
    restoration path).  P0's collector caught the same family
    (FloatingPointError/ValueError/RuntimeError) before full rebuilds.
    """

    if isinstance(exc, FloatingPointError):
        return True
    return isinstance(exc, (ValueError, RuntimeError)) and "non-finite" in str(exc)


def build_batch_init_vertices(
    params: RopeParams,
    *,
    n_envs: int,
    episode_index: int,
    seed: int,
    init_shapes: Sequence[str] = INIT_SHAPES,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Build per-env seeded init curves cycling the four templates uniformly."""

    vertices = []
    shapes = []
    seeds = []
    for env_idx in range(int(n_envs)):
        shape = str(init_shapes[(episode_index * n_envs + env_idx) % len(init_shapes)])
        curve_seed = int(seed + 100_000 * (episode_index + 1) + env_idx)
        vertices.append(analytic_init_centerline(params, shape, curve_seed))
        shapes.append(shape)
        seeds.append(curve_seed)
    return np.stack(vertices), shapes, seeds


def random_policy_actions(
    rng: np.random.Generator,
    *,
    n_envs: int,
    n_vertices: int,
    delta_min_m: float = 0.02,
    delta_max_m: float = 0.15,
    lift_choices: Sequence[str] = ("low", "high"),
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Sample the shared M0/M2 random policy (uniform p, planar delta, lift)."""

    p = rng.integers(0, int(n_vertices), size=int(n_envs))
    radii = rng.uniform(float(delta_min_m), float(delta_max_m), size=int(n_envs))
    angles = rng.uniform(0.0, 2.0 * np.pi, size=int(n_envs))
    deltas = np.column_stack((radii * np.cos(angles), radii * np.sin(angles), np.zeros(int(n_envs))))
    lifts = [str(lift_choices[int(idx)]) for idx in rng.integers(0, len(lift_choices), size=int(n_envs))]
    return p.astype(int), deltas.astype(float), lifts


@dataclass
class EpisodeConfig:
    """P1 episode protocol constants (immutable tier unless noted)."""

    horizon: int = EPISODE_HORIZON
    vel_threshold: float = SETTLE_VEL_THRESHOLD
    settle_max_steps: int = SETTLE_MAX_STEPS
    reward: RewardConstants = field(default_factory=RewardConstants)


class BatchedEpisodeRunner:
    """Run batched task episodes with reward, termination, and NaN covenant."""

    def __init__(self, env: Any, params: RopeParams, config: EpisodeConfig | None = None) -> None:
        self.env = env
        self.params = params
        self.config = config or EpisodeConfig()
        self.n_envs = int(env.n_envs)
        self.length_m = float(params.length_m)

        self.goals: list[DualGoal] = []
        self.init_shapes: list[str] = []
        self.t = np.zeros(self.n_envs, dtype=int)
        self.done = np.ones(self.n_envs, dtype=bool)
        self.succeeded = np.zeros(self.n_envs, dtype=bool)
        self.d_current = np.full(self.n_envs, np.nan, dtype=float)
        self.nan_incidents = 0
        self.incident_log: list[dict[str, Any]] = []
        self._base_seed = 0
        self._reseed_counter = 0

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def begin_episodes(
        self,
        *,
        seed: int,
        episode_index: int = 0,
        init_shapes: Sequence[str] = INIT_SHAPES,
        goals: Sequence[DualGoal] | None = None,
        goal_fn: GoalFn | None = None,
    ) -> dict[str, Any]:
        """Reset all environments to seeded init states and assign goals.

        Exactly one of ``goals`` (fixed per-env goals, e.g. T2) or ``goal_fn``
        (state-dependent sampler, e.g. T1) must be provided.  ``goal_fn`` is
        called with the settled post-reset centerline of each environment.
        """

        if (goals is None) == (goal_fn is None):
            raise ValueError("provide exactly one of goals or goal_fn")

        self._base_seed = int(seed)
        vertices, shapes, curve_seeds = build_batch_init_vertices(
            self.params,
            n_envs=self.n_envs,
            episode_index=int(episode_index),
            seed=int(seed),
            init_shapes=init_shapes,
        )
        centerlines, reset_result, reset_reseeded = self._settle_finite(
            vertices, reason="non-finite state during episode reset settle"
        )

        if goals is not None:
            if len(goals) != self.n_envs:
                raise ValueError(f"goals must contain {self.n_envs} entries, got {len(goals)}")
            assigned = list(goals)
        else:
            assert goal_fn is not None
            assigned = []
            for env_idx in range(self.n_envs):
                goal_rng = np.random.default_rng(
                    np.random.SeedSequence([int(seed), int(episode_index), env_idx])
                )
                assigned.append(goal_fn(env_idx, centerlines[env_idx], goal_rng))

        self.goals = assigned
        self.init_shapes = shapes
        self.t = np.zeros(self.n_envs, dtype=int)
        self.done = np.zeros(self.n_envs, dtype=bool)
        self.succeeded = np.zeros(self.n_envs, dtype=bool)
        self.d_current = np.asarray(
            [
                distance_to_goal(centerlines[i], self.goals[i], self.length_m)
                for i in range(self.n_envs)
            ],
            dtype=float,
        )
        return {
            "init_shapes": shapes,
            "curve_seeds": curve_seeds,
            "reset_settle_converged": np.asarray(reset_result["settle_converged"], dtype=bool),
            "reset_settle_steps": np.asarray(reset_result["settle_steps"], dtype=int),
            "reset_reseeded_envs": reset_reseeded.tolist(),
            "d_initial": self.d_current.copy(),
        }

    def step(
        self,
        p: np.ndarray,
        delta: np.ndarray,
        lift: Sequence[str],
        *,
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        """Execute one batched primitive and return per-env transition data.

        Environments whose episode already ended are still stepped physically
        (the batch API moves every environment) but their transitions are
        reported with ``active=False`` and must not be consumed as data.
        """

        if not self.goals:
            raise RuntimeError("begin_episodes must be called first")

        active_before = ~self.done
        d_before = self.d_current.copy()

        try:
            result = self.env.step_primitive_batch(
                np.asarray(p, dtype=int),
                np.asarray(delta, dtype=float),
                list(lift),
                vel_threshold=self.config.vel_threshold,
                max_steps=self.config.settle_max_steps,
                rng=rng,
            )
        except (FloatingPointError, ValueError, RuntimeError) as exc:
            if not is_nonfinite_error(exc):
                raise
            return self._handle_nan_incident(active_before, reason=f"env raise: {exc}")

        x_after = np.asarray(result["X_after"], dtype=float)
        finite_mask = np.isfinite(x_after).all(axis=(1, 2))
        if not finite_mask.all():
            return self._handle_nan_incident(
                active_before,
                reason="non-finite X_after rows",
                bad_envs=np.flatnonzero(~finite_mask),
            )

        d_after = np.empty(self.n_envs, dtype=float)
        rewards = np.empty(self.n_envs, dtype=float)
        successes = np.zeros(self.n_envs, dtype=bool)
        for i in range(self.n_envs):
            d_after[i] = distance_to_goal(x_after[i], self.goals[i], self.length_m)
            rewards[i], successes[i] = step_reward(
                d_before[i], d_after[i], self.length_m, self.config.reward
            )

        self.t = self.t + active_before.astype(int)
        newly_done = active_before & (successes | (self.t >= self.config.horizon))
        self.succeeded = self.succeeded | (active_before & successes)
        self.done = self.done | newly_done
        self.d_current = d_after

        return {
            "discarded": False,
            "active": active_before,
            "reward": rewards,
            "d_before": d_before,
            "d_after": d_after,
            "success": successes,
            "done": self.done.copy(),
            "t": self.t.copy(),
            "X_before": np.asarray(result["X_before"], dtype=float),
            "X_after": x_after,
            "grasp_success": np.asarray(result["grasp_success"], dtype=bool),
            "settle_steps": np.asarray(result["settle_steps"], dtype=int),
            "settle_converged": np.asarray(result["info"]["settle_converged"], dtype=bool),
            "info": result["info"],
        }

    def all_done(self) -> bool:
        """Return whether every environment's episode has terminated."""

        return bool(np.all(self.done))

    # ------------------------------------------------------------------
    # NaN covenant (global rule 6, env level)
    # ------------------------------------------------------------------

    def _handle_nan_incident(
        self,
        active_before: np.ndarray,
        *,
        reason: str,
        bad_envs: np.ndarray | None = None,
    ) -> dict[str, Any]:
        raw = np.asarray(self.env.get_centerline_raw_batch(), dtype=float)
        if bad_envs is None:
            finite_rows = np.isfinite(raw).all(axis=(1, 2))
            bad_envs = np.flatnonzero(~finite_rows)
            if bad_envs.size == 0:
                # The env raised but every row reads finite now; treat every
                # environment as suspect rather than silently continuing.
                bad_envs = np.arange(self.n_envs)
        bad_envs = np.asarray(bad_envs, dtype=int)

        replacement = raw.copy()
        for env_idx in bad_envs:
            replacement[env_idx] = self._fresh_init_curve(int(env_idx))
        if not np.isfinite(replacement).all():
            # A non-reseeded row is itself non-finite (e.g. it degraded during
            # the failed primitive); treat those rows as bad too.
            extra = np.flatnonzero(~np.isfinite(replacement).all(axis=(1, 2)))
            for env_idx in extra:
                replacement[env_idx] = self._fresh_init_curve(int(env_idx))
            bad_envs = np.union1d(bad_envs, extra)

        centerlines, _, recovery_reseeded = self._settle_finite(
            replacement, reason=f"recovery settle after: {reason}"
        )
        bad_envs = np.union1d(bad_envs, recovery_reseeded)

        # Reseeded environments restart their episode (same goal, t=0);
        # untouched environments continue with a re-measured D.
        for env_idx in bad_envs:
            self.t[env_idx] = 0
            self.done[env_idx] = False
            self.succeeded[env_idx] = False
        self.d_current = np.asarray(
            [
                distance_to_goal(centerlines[i], self.goals[i], self.length_m)
                for i in range(self.n_envs)
            ],
            dtype=float,
        )

        self.nan_incidents += int(bad_envs.size)
        self.incident_log.append(
            {
                "reason": reason,
                "bad_envs": bad_envs.tolist(),
                "reseed_counter": self._reseed_counter,
            }
        )
        return {
            "discarded": True,
            "active": active_before,
            "reason": reason,
            "bad_envs": bad_envs.copy(),
            "nan_incidents": self.nan_incidents,
        }


    # ------------------------------------------------------------------
    # Finite-settle helper shared by reset and incident recovery
    # ------------------------------------------------------------------

    def _fresh_init_curve(self, env_idx: int) -> np.ndarray:
        self._reseed_counter += 1
        new_seed = self._base_seed + 1_000_000 + self._reseed_counter
        shape = self.init_shapes[env_idx] if self.init_shapes else INIT_SHAPES[env_idx % len(INIT_SHAPES)]
        return analytic_init_centerline(self.params, shape, new_seed)

    def _settle_finite(
        self,
        vertices: np.ndarray,
        *,
        reason: str,
    ) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
        """``light_reset`` with the immutable budget plus finiteness enforcement.

        Any environment whose raw vertices or resampled centerline come back
        non-finite is reseeded with a fresh init curve and the settle is
        retried (NaN covenant, counted as incidents).  Raises
        ``FloatingPointError`` when states remain non-finite after
        ``MAX_RESEED_ATTEMPTS`` retries — silent continuation is forbidden.
        """

        current = np.asarray(vertices, dtype=float).copy()
        reseeded: set[int] = set()
        for attempt in range(MAX_RESEED_ATTEMPTS + 1):
            reset_result = self.env.light_reset(
                current,
                vel_threshold=self.config.vel_threshold,
                max_steps=self.config.settle_max_steps,
            )
            raw = np.asarray(self.env.get_centerline_raw_batch(), dtype=float)
            centerlines = np.asarray(self.env.get_centerline_batch(), dtype=float)
            bad = ~(
                np.isfinite(raw).all(axis=(1, 2)) & np.isfinite(centerlines).all(axis=(1, 2))
            )
            bad_indices = np.flatnonzero(bad)
            if bad_indices.size == 0:
                if reseeded:
                    self.nan_incidents += len(reseeded)
                    self.incident_log.append(
                        {
                            "reason": reason,
                            "bad_envs": sorted(reseeded),
                            "reseed_counter": self._reseed_counter,
                        }
                    )
                return centerlines, reset_result, np.asarray(sorted(reseeded), dtype=int)
            if attempt == MAX_RESEED_ATTEMPTS:
                raise FloatingPointError(
                    f"rope state still non-finite after {MAX_RESEED_ATTEMPTS} reseed retries "
                    f"({reason}; envs {bad_indices.tolist()})"
                )
            current = raw.copy()
            for env_idx in bad_indices:
                current[env_idx] = self._fresh_init_curve(int(env_idx))
                reseeded.add(int(env_idx))
        raise AssertionError("unreachable")

__all__ = [
    "BatchedEpisodeRunner",
    "EpisodeConfig",
    "build_batch_init_vertices",
    "random_policy_actions",
]
