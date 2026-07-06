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
from dgcc.goals.dual_goal import DualGoal, goal_curve
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
MAX_COORD_NORM_M = 3.0
GOAL_NODE_NORM_BOUND_M = 4.0


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
    """Run batched task episodes with F1/F2/F3 rewards, termination, and covenants."""

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
        self.truncated = np.zeros(self.n_envs, dtype=bool)
        self.d_current = np.full(self.n_envs, np.nan, dtype=float)
        self.d_at_done = np.full(self.n_envs, np.nan, dtype=float)
        self.nan_incidents = 0
        self.magnitude_incidents = 0
        self.incident_log: list[dict[str, Any]] = []
        self._base_seed = 0
        self._reseed_counter = 0
        self._auto_reset = False
        self._goal_fn: GoalFn | None = None
        self._goal_pool: list[DualGoal] | None = None
        self._episode_counter = 0
        self.episodes_completed = 0
        self.episodes_succeeded = 0

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
        auto_reset: bool = False,
        goal_pool: Sequence[DualGoal] | None = None,
    ) -> dict[str, Any]:
        """Reset all environments to seeded init states and assign goals.

        Exactly one of ``goals`` (fixed per-env goals, e.g. T2) or ``goal_fn``
        (state-dependent sampler, e.g. T1) must be provided.  ``goal_fn`` is
        called with the settled post-reset centerline of each environment.
        """

        if goal_pool is not None and goals is None and goal_fn is None:
            # Auto-reset pool mode (T2 training): initial goals drawn from the pool.
            pool_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 777]))
            goals = [goal_pool[int(i)] for i in pool_rng.integers(0, len(goal_pool), self.n_envs)]
        if (goals is None) == (goal_fn is None):
            raise ValueError("provide exactly one of goals or goal_fn (or goal_pool)")
        self._auto_reset = bool(auto_reset)
        self._goal_fn = goal_fn
        self._goal_pool = list(goal_pool) if goal_pool is not None else None
        self._episode_counter = int(episode_index)
        if self._auto_reset and self._goal_fn is None and self._goal_pool is None:
            raise ValueError("auto_reset requires goal_fn or goal_pool for episode refresh")

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
        self._assert_goal_node_norms(assigned)

        self.goals = assigned
        self.init_shapes = shapes
        self.t = np.zeros(self.n_envs, dtype=int)
        self.done = np.zeros(self.n_envs, dtype=bool)
        self.succeeded = np.zeros(self.n_envs, dtype=bool)
        self.truncated = np.zeros(self.n_envs, dtype=bool)
        self.d_current = np.asarray(
            [
                distance_to_goal(centerlines[i], self.goals[i], self.length_m)
                for i in range(self.n_envs)
            ],
            dtype=float,
        )
        self.d_at_done = np.full(self.n_envs, np.nan, dtype=float)
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
        """Execute one F1/F3 batched primitive and return per-env transition data.

        Environments whose episode already ended are still stepped physically
        (the batch API moves every environment) but their transitions are
        reported with ``active=False`` and must not be consumed as data.
        In auto-reset training, ``d_at_done`` is stale/NaN until the next
        terminal step for the refreshed env because reset starts a new episode
        immediately after the terminal record is snapshotted.
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
                bad_kind="nonfinite",
            )
        mag_bad_rows = (np.linalg.norm(x_after, axis=-1) > MAX_COORD_NORM_M).any(axis=-1)
        if mag_bad_rows.any():
            return self._handle_nan_incident(
                active_before,
                reason="magnitude covenant: coord norm > 3 m",
                bad_envs=np.flatnonzero(mag_bad_rows),
                bad_kind="magnitude",
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
        terminal_success = active_before & successes
        truncated_now = active_before & (self.t >= self.config.horizon) & ~successes
        newly_done = terminal_success | truncated_now
        self.succeeded = self.succeeded | terminal_success
        self.truncated = self.truncated | truncated_now
        self.done = self.done | newly_done
        self.d_current = np.where(active_before, d_after, self.d_current)
        self.d_at_done[newly_done] = d_after[newly_done]
        self.episodes_completed += int(newly_done.sum())
        self.episodes_succeeded += int(terminal_success.sum())

        record_done = self.done.copy()
        record_t = self.t.copy()
        record_truncated = self.truncated.copy()
        record_d_at_done = self.d_at_done.copy()
        if self._auto_reset and self.done.any():
            self._refresh_done_episodes()

        return {
            "discarded": False,
            "active": active_before,
            "reward": rewards,
            "d_before": d_before,
            "d_after": d_after,
            "success": successes,
            "done": record_done,
            "t": record_t,
            "truncated": record_truncated,
            "d_at_done": record_d_at_done,
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
    # Auto-reset (vectorized training collection; evals keep batch semantics)
    # ------------------------------------------------------------------

    def _refresh_done_episodes(self) -> None:
        """Replace finished environments with fresh episodes in place.

        Keeps every environment active on every round (a batch that waits for
        its slowest episode idles early-succeeding envs — pathologically, a
        better policy would collect data SLOWER).  Episode protocol per env is
        unchanged: T=10 horizon, early success stop, fresh seeded init curve
        and fresh goal on reset.  Active environments keep their positions;
        the shared settle after placement is a no-op for already-settled
        ropes (quasi-static protocol).
        """

        done_envs = np.flatnonzero(self.done)
        raw = np.asarray(self.env.get_centerline_raw_batch(), dtype=float)
        replacement = raw.copy()
        for env_idx in done_envs:
            self._episode_counter += 1
            shape = INIT_SHAPES[self._episode_counter % len(INIT_SHAPES)]
            curve_seed = self._base_seed + 500_000 + self._episode_counter
            replacement[env_idx] = analytic_init_centerline(self.params, shape, curve_seed)
            self.init_shapes[env_idx] = shape

        centerlines, _, reseeded = self._settle_finite(
            replacement,
            reason="non-finite state during auto-reset settle",
            reinit_env_indices=done_envs,
        )
        refreshed_envs = np.union1d(done_envs, reseeded)
        for env_idx in refreshed_envs:
            i = int(env_idx)
            goal_rng = np.random.default_rng(
                np.random.SeedSequence([self._base_seed, 31_337, self._episode_counter, i])
            )
            if self._goal_fn is not None:
                self.goals[i] = self._goal_fn(i, centerlines[i], goal_rng)
            else:
                assert self._goal_pool is not None
                self.goals[i] = self._goal_pool[int(goal_rng.integers(0, len(self._goal_pool)))]
            self.t[i] = 0
            self.done[i] = False
            self.succeeded[i] = False
            self.truncated[i] = False
            self.d_at_done[i] = np.nan
        self._assert_goal_node_norms([self.goals[int(env_idx)] for env_idx in refreshed_envs])
        self.d_current = np.asarray(
            [
                distance_to_goal(centerlines[i], self.goals[i], self.length_m)
                for i in range(self.n_envs)
            ],
            dtype=float,
        )

    # ------------------------------------------------------------------
    # NaN/magnitude covenant (global rule 6 + F2 magnitude bound)
    # ------------------------------------------------------------------

    def _assert_goal_node_norms(self, goals: Sequence[DualGoal]) -> None:
        """Assert S3's goal-node norm premise at each goal assignment."""

        if len(goals) == 0:
            return
        curves = np.stack([goal_curve(goal, self.length_m) for goal in goals])
        norms = np.linalg.norm(curves, axis=-1)
        max_norm = float(norms.max())
        if bool((norms > GOAL_NODE_NORM_BOUND_M).any()):
            raise ValueError(
                "goal-node norm > 4.0 m — S3 bound premise violated "
                f"(max={max_norm:.6g} m)"
            )

    @staticmethod
    def _nonfinite_rows(*arrays: np.ndarray) -> np.ndarray:
        rows = np.zeros(np.asarray(arrays[0], dtype=float).shape[0], dtype=bool)
        for array in arrays:
            rows |= ~np.isfinite(np.asarray(array, dtype=float)).all(axis=(1, 2))
        return rows

    @staticmethod
    def _magnitude_bad_rows(*arrays: np.ndarray) -> np.ndarray:
        rows = np.zeros(np.asarray(arrays[0], dtype=float).shape[0], dtype=bool)
        for array in arrays:
            norms = np.linalg.norm(np.asarray(array, dtype=float), axis=-1)
            rows |= (norms > MAX_COORD_NORM_M).any(axis=-1)
        return rows

    @staticmethod
    def _max_coord_norm_by_row(*arrays: np.ndarray) -> np.ndarray:
        out = np.full(np.asarray(arrays[0], dtype=float).shape[0], np.nan, dtype=float)
        for array in arrays:
            norms = np.linalg.norm(np.asarray(array, dtype=float), axis=-1)
            row_max = np.where(np.isnan(norms), -np.inf, norms).max(axis=-1)
            row_max = np.where(row_max == -np.inf, np.nan, row_max)
            replace = np.isnan(out) | (~np.isnan(row_max) & (row_max > out))
            out = np.where(replace, row_max, out)
        return out

    def _classify_invalid_rows(
        self, *arrays: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nonfinite = self._nonfinite_rows(*arrays)
        magnitude = self._magnitude_bad_rows(*arrays) & ~nonfinite
        return nonfinite, magnitude, self._max_coord_norm_by_row(*arrays)

    def _record_incidents(
        self,
        *,
        reason: str,
        bad_envs: np.ndarray,
        kinds: Sequence[str],
        max_coord_norms: Sequence[float],
    ) -> None:
        envs = np.asarray(bad_envs, dtype=int)
        row_kinds = np.asarray(kinds, dtype=object)
        row_max_norms = np.asarray(max_coord_norms, dtype=float)
        for kind in ("nonfinite", "magnitude"):
            mask = row_kinds == kind
            if not bool(mask.any()):
                continue
            env_list = [int(i) for i in envs[mask]]
            if kind == "nonfinite":
                self.nan_incidents += len(env_list)
            else:
                self.magnitude_incidents += len(env_list)
            finite_maxes = row_max_norms[mask]
            finite_maxes = finite_maxes[~np.isnan(finite_maxes)]
            max_coord_norm = float(finite_maxes.max()) if finite_maxes.size else float("nan")
            self.incident_log.append(
                {
                    "reason": reason,
                    "bad_envs": env_list,
                    "kind": kind,
                    "max_coord_norm": max_coord_norm,
                    "reseed_counter": self._reseed_counter,
                }
            )

    @staticmethod
    def _kind_from_masks(
        env_idx: int,
        nonfinite: np.ndarray,
        magnitude: np.ndarray,
        fallback_kind: str,
    ) -> str:
        if bool(nonfinite[env_idx]):
            return "nonfinite"
        if bool(magnitude[env_idx]):
            return "magnitude"
        return fallback_kind

    def _handle_nan_incident(
        self,
        active_before: np.ndarray,
        *,
        reason: str,
        bad_envs: np.ndarray | None = None,
        bad_kind: str | None = None,
    ) -> dict[str, Any]:
        raw = np.asarray(self.env.get_centerline_raw_batch(), dtype=float)
        nonfinite_rows, magnitude_rows, max_norm_by_row = self._classify_invalid_rows(raw)
        fallback_kind = bad_kind or ("magnitude" if "magnitude covenant" in reason else "nonfinite")
        if bad_envs is None:
            bad_mask = nonfinite_rows | magnitude_rows
            bad_envs = np.flatnonzero(bad_mask)
            if bad_envs.size == 0:
                # The env raised but every row reads finite now; treat every
                # environment as suspect rather than silently continuing.
                bad_envs = np.arange(self.n_envs)
        bad_envs = np.unique(np.asarray(bad_envs, dtype=int))

        kind_by_env: dict[int, str] = {}
        max_norm_by_env: dict[int, float] = {}
        for env_idx in bad_envs:
            i = int(env_idx)
            kind_by_env[i] = bad_kind or self._kind_from_masks(
                i, nonfinite_rows, magnitude_rows, fallback_kind
            )
            max_norm_by_env[i] = float(max_norm_by_row[i])

        replacement = raw.copy()
        for env_idx in bad_envs:
            replacement[int(env_idx)] = self._fresh_init_curve(int(env_idx))

        extra_nonfinite, extra_magnitude, extra_max_norm = self._classify_invalid_rows(replacement)
        extra = np.flatnonzero(extra_nonfinite | extra_magnitude)
        for env_idx in extra:
            i = int(env_idx)
            kind_by_env.setdefault(
                i, self._kind_from_masks(i, extra_nonfinite, extra_magnitude, fallback_kind)
            )
            max_norm_by_env.setdefault(i, float(extra_max_norm[i]))
            replacement[i] = self._fresh_init_curve(i)

        bad_envs = np.asarray(sorted(kind_by_env), dtype=int)
        centerlines, _, recovery_reseeded = self._settle_finite(
            replacement,
            reason=f"recovery settle after: {reason}",
            reinit_env_indices=bad_envs,
        )
        all_reseeded = np.union1d(bad_envs, recovery_reseeded)

        # Reseeded active environments restart their episode (same goal, t=0).
        # Done envs may be physically re-placed, but their completed metrics
        # stay frozen for eval batches without auto-reset.
        active_mask = np.asarray(active_before, dtype=bool)
        restart_envs = all_reseeded[active_mask[all_reseeded]]
        for env_idx in restart_envs:
            i = int(env_idx)
            self.t[i] = 0
            self.done[i] = False
            self.succeeded[i] = False
            self.truncated[i] = False
            self.d_at_done[i] = np.nan

        recovered_d = np.asarray(
            [
                distance_to_goal(centerlines[i], self.goals[i], self.length_m)
                for i in range(self.n_envs)
            ],
            dtype=float,
        )
        self.d_current = np.where(~self.done, recovered_d, self.d_current)

        if bad_envs.size:
            self._record_incidents(
                reason=reason,
                bad_envs=bad_envs,
                kinds=[kind_by_env[int(i)] for i in bad_envs],
                max_coord_norms=[max_norm_by_env[int(i)] for i in bad_envs],
            )
        return {
            "discarded": True,
            "active": active_before,
            "reason": reason,
            "bad_envs": all_reseeded.copy(),
            "nan_incidents": self.nan_incidents,
            "magnitude_incidents": self.magnitude_incidents,
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
        reinit_env_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
        """``light_reset`` with the immutable budget plus F2 state-covenant enforcement.

        Any environment whose raw vertices or resampled centerline come back
        non-finite or outside the 3 m coordinate-norm bound is reseeded with a
        fresh init curve and the settle is retried.  Raises
        ``FloatingPointError`` when states remain invalid after
        ``MAX_RESEED_ATTEMPTS`` retries — silent continuation is forbidden.

        ``reinit_env_indices`` scopes the adapter's full state
        re-initialization to the re-placed environments (M2 gate F2);
        ``None`` re-initializes every environment.  Environments reseeded by
        the retry loop are added to the scope automatically.
        """

        current = np.asarray(vertices, dtype=float).copy()
        reseeded: set[int] = set()
        kind_by_env: dict[int, str] = {}
        max_norm_by_env: dict[int, float] = {}
        scope = (
            None
            if reinit_env_indices is None
            else {int(i) for i in np.asarray(reinit_env_indices).reshape(-1)}
        )
        for attempt in range(MAX_RESEED_ATTEMPTS + 1):
            reinit = None if scope is None else np.asarray(sorted(scope | reseeded), dtype=int)
            try:
                reset_result = self.env.light_reset(
                    current,
                    vel_threshold=self.config.vel_threshold,
                    max_steps=self.config.settle_max_steps,
                    reinit_env_indices=reinit,
                )
            except TypeError:
                # Older duck-typed envs without the scoping kwarg.
                reset_result = self.env.light_reset(
                    current,
                    vel_threshold=self.config.vel_threshold,
                    max_steps=self.config.settle_max_steps,
                )
            raw = np.asarray(self.env.get_centerline_raw_batch(), dtype=float)
            centerlines = np.asarray(self.env.get_centerline_batch(), dtype=float)
            nonfinite_rows, magnitude_rows, max_norm_by_row = self._classify_invalid_rows(
                raw, centerlines
            )
            bad_indices = np.flatnonzero(nonfinite_rows | magnitude_rows)
            if bad_indices.size == 0:
                if reseeded:
                    reseeded_envs = np.asarray(sorted(reseeded), dtype=int)
                    self._record_incidents(
                        reason=reason,
                        bad_envs=reseeded_envs,
                        kinds=[kind_by_env[int(i)] for i in reseeded_envs],
                        max_coord_norms=[max_norm_by_env[int(i)] for i in reseeded_envs],
                    )
                return centerlines, reset_result, np.asarray(sorted(reseeded), dtype=int)
            if attempt == MAX_RESEED_ATTEMPTS:
                raise FloatingPointError(
                    f"rope state still invalid after {MAX_RESEED_ATTEMPTS} reseed retries "
                    f"({reason}; envs {bad_indices.tolist()})"
                )
            current = raw.copy()
            for env_idx in bad_indices:
                i = int(env_idx)
                kind = self._kind_from_masks(i, nonfinite_rows, magnitude_rows, "nonfinite")
                if i in kind_by_env and kind_by_env[i] == "magnitude" and kind == "nonfinite":
                    kind_by_env[i] = "nonfinite"
                else:
                    kind_by_env.setdefault(i, kind)
                previous = max_norm_by_env.get(i, float("nan"))
                observed = float(max_norm_by_row[i])
                if np.isnan(previous) or (not np.isnan(observed) and observed > previous):
                    max_norm_by_env[i] = observed
                current[i] = self._fresh_init_curve(i)
                reseeded.add(i)
        raise AssertionError("unreachable")

__all__ = [
    "BatchedEpisodeRunner",
    "EpisodeConfig",
    "GOAL_NODE_NORM_BOUND_M",
    "MAX_COORD_NORM_M",
    "build_batch_init_vertices",
    "random_policy_actions",
]
