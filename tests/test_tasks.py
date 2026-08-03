"""P1-M0 task-layer tests (CPU only — no Genesis/GPU required).

Covers the M0 exit contract: goal-generation determinism, split
disjointness, reward sign sanity, success-judgment consistency (same D code
path), T=10 termination — plus the plan-mandated guards: pinned §5 rope
domain, metric routing (correspondence_l2 only, Chamfer family rejected),
settle-budget call-path capture (max_steps=10000 on every settle-bearing
call), and the env-level NaN covenant.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any

import numpy as np
import pytest

import dgcc.tasks.reward as reward_module
from dgcc.goals import distance
from dgcc.goals.dual_goal import anchor_of, goal_curve
from dgcc.tasks.domain import (
    EPISODE_HORIZON,
    EPS_SUCC_COEFF,
    RewardConstants,
    SETTLE_MAX_STEPS,
    SETTLE_VEL_THRESHOLD,
    p1_rope_params,
)
from dgcc.tasks.episode import (
    BatchedEpisodeRunner,
    build_batch_init_vertices,
    random_policy_actions,
)
from dgcc.tasks.reward import distance_to_goal, is_success, step_reward
from dgcc.tasks.t1 import (
    T1C_DISPLACEMENT_RANGE,
    sample_t1a_goal,
    sample_t1b_goal,
    sample_t1c_goal,
)
from dgcc.tasks.t2 import (
    T2_FAMILIES,
    T2_SPLIT_SIZES,
    build_t2_goal,
    default_split_path,
    generate_t2_payload,
    load_t2_payload,
    load_t2_split,
    payload_json,
)

LENGTH = 1.0


def straight_line(offset_x: float = 0.0, n: int = 32) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    return np.column_stack((t + offset_x, np.zeros(n), np.zeros(n)))


def straight_goal() -> Any:
    from dgcc.goals.dual_goal import DualGoal

    return DualGoal(
        shape_template=straight_line(),
        anchor=np.array([0.5, 0.0, 0.0]),
        anchor_mode="centroid",
        template_name="test_straight",
    )


# ---------------------------------------------------------------------------
# Fake batch env (duck-typed DLOLabEnv surface used by the episode runner)
# ---------------------------------------------------------------------------


class FakeBatchEnv:
    """Records settle-bearing call kwargs and applies scripted dynamics."""

    def __init__(self, n_envs: int, *, approach_fraction: float = 0.0) -> None:
        self.n_envs = int(n_envs)
        self.state: np.ndarray | None = None
        self.targets: np.ndarray | None = None
        self.approach_fraction = float(approach_fraction)
        self.settle_calls: list[dict[str, Any]] = []
        self.fail_next_step_with_nan_env: int | None = None
        self.corrupt_reset_env_once: int | None = None

    def light_reset(self, vertices: np.ndarray, *, vel_threshold: float, max_steps: int) -> dict:
        self.settle_calls.append(
            {"method": "light_reset", "vel_threshold": vel_threshold, "max_steps": max_steps}
        )
        verts = np.asarray(vertices, dtype=float)
        if verts.ndim == 2:
            verts = np.broadcast_to(verts, (self.n_envs, *verts.shape)).copy()
        self.state = verts.copy()
        if self.corrupt_reset_env_once is not None:
            self.state[self.corrupt_reset_env_once] = np.nan
            self.corrupt_reset_env_once = None
        return {
            "settle_converged": np.ones(self.n_envs, dtype=bool),
            "settle_steps": np.zeros(self.n_envs, dtype=int),
        }

    def get_centerline_batch(self) -> np.ndarray:
        assert self.state is not None
        return self.state.copy()

    def get_centerline_raw_batch(self) -> np.ndarray:
        assert self.state is not None
        return self.state.copy()

    def step_primitive_batch(
        self,
        p: np.ndarray,
        delta: np.ndarray,
        lift: list[str],
        *,
        vel_threshold: float,
        max_steps: int,
        rng: np.random.Generator | None = None,
    ) -> dict:
        del p, lift, rng
        assert self.state is not None
        self.settle_calls.append(
            {
                "method": "step_primitive_batch",
                "vel_threshold": vel_threshold,
                "max_steps": max_steps,
            }
        )
        if self.fail_next_step_with_nan_env is not None:
            bad = self.fail_next_step_with_nan_env
            self.fail_next_step_with_nan_env = None
            self.state[bad] = np.nan
            raise FloatingPointError("fake non-finite rope state")

        x_before = self.state.copy()
        if self.targets is not None and self.approach_fraction > 0.0:
            self.state = self.state + self.approach_fraction * (self.targets - self.state)
        else:
            self.state = self.state + np.asarray(delta, dtype=float)[:, None, :] * 0.0
        return {
            "X_before": x_before,
            "X_after": self.state.copy(),
            "grasp_success": np.ones(self.n_envs, dtype=bool),
            "settle_steps": np.full(self.n_envs, 7, dtype=int),
            "info": {"settle_converged": np.ones(self.n_envs, dtype=bool)},
        }


def make_runner(n_envs: int = 4, **fake_kwargs: Any) -> tuple[BatchedEpisodeRunner, FakeBatchEnv]:
    env = FakeBatchEnv(n_envs, **fake_kwargs)
    runner = BatchedEpisodeRunner(env, p1_rope_params())
    return runner, env


# ---------------------------------------------------------------------------
# §5 pinned rope domain
# ---------------------------------------------------------------------------


def test_p1_rope_domain_pinned_field_by_field() -> None:
    params = p1_rope_params()
    assert params.length_m == 1.0
    assert params.n_segments == 32  # base.py default is 50 — must be overridden
    assert params.bend_stiffness == 1.0
    assert params.twist_stiffness == 1.0
    assert params.friction == 1.0
    assert params.radius == 0.005


# ---------------------------------------------------------------------------
# Goal-generation determinism and splits
# ---------------------------------------------------------------------------


def test_t2_generation_deterministic() -> None:
    assert payload_json(generate_t2_payload()) == payload_json(generate_t2_payload())


def test_t2_committed_split_matches_regeneration() -> None:
    committed = default_split_path().read_text(encoding="utf-8")
    assert committed == payload_json(generate_t2_payload())


def test_t2_split_sizes_and_disjointness() -> None:
    payload = load_t2_payload()
    splits = payload["splits"]
    for name, size in T2_SPLIT_SIZES.items():
        assert len(splits[name]) == size
        assert len(set(splits[name])) == size
    train, val, heldout = (set(splits[k]) for k in ("train", "val", "heldout"))
    assert not train & val
    assert not train & heldout
    assert not val & heldout
    all_ids = {spec["goal_id"] for spec in payload["specs"]}
    assert train | val | heldout == all_ids


def test_t2_splits_cover_families_and_include_asymmetric_goals() -> None:
    payload = load_t2_payload()
    by_id = {spec["goal_id"]: spec for spec in payload["specs"]}
    for name in T2_SPLIT_SIZES:
        specs = [by_id[goal_id] for goal_id in payload["splits"][name]]
        assert {spec["family"] for spec in specs} == set(T2_FAMILIES)
        assert any(spec["asymmetric"] for spec in specs), f"no asymmetric goals in {name}"


def test_t2_goal_reconstruction_finite() -> None:
    for _, goal in load_t2_split("val"):
        curve = goal_curve(goal, LENGTH)
        assert curve.shape == (32, 3)
        assert np.isfinite(curve).all()
        assert distance_to_goal(curve, goal, LENGTH) < 1.0e-9


def test_t1_samplers_deterministic_and_in_spec() -> None:
    x = straight_line()
    goal_a = sample_t1a_goal(x, np.random.default_rng(0))
    np.testing.assert_allclose(goal_a.anchor, anchor_of(x, "centroid"))
    # straight template: colinear after normalization
    template = goal_a.shape_template
    direction = template[-1] - template[0]
    residual = template - template[0] - np.outer(
        (template - template[0]) @ direction / (direction @ direction), direction
    )
    assert float(np.abs(residual).max()) < 1.0e-9

    b1 = sample_t1b_goal(x, np.random.default_rng(7))
    b2 = sample_t1b_goal(x, np.random.default_rng(7))
    np.testing.assert_array_equal(b1.shape_template, b2.shape_template)

    c1 = sample_t1c_goal(x, np.random.default_rng(11))
    c2 = sample_t1c_goal(x, np.random.default_rng(11))
    np.testing.assert_array_equal(c1.anchor, c2.anchor)
    displacement = np.linalg.norm(c1.anchor - anchor_of(x, "endpoint"))
    lo, hi = T1C_DISPLACEMENT_RANGE
    assert lo <= displacement <= hi
    # shape preserved: shape-only distance between goal curve and current shape
    assert distance.correspondence_l2(
        goal_curve(c1, LENGTH), x, LENGTH, shape_only=True
    ) < 1.0e-6


# ---------------------------------------------------------------------------
# Reward sign sanity and success consistency (same D code path)
# ---------------------------------------------------------------------------


def test_reward_sign_sanity() -> None:
    goal = straight_goal()
    d_far = distance_to_goal(straight_line(offset_x=0.30), goal, LENGTH)
    d_near = distance_to_goal(straight_line(offset_x=0.10), goal, LENGTH)
    assert d_near < d_far

    constants = RewardConstants()
    reward_closer, _ = step_reward(d_far, d_near, LENGTH, constants)
    reward_away, _ = step_reward(d_near, d_far, LENGTH, constants)
    assert reward_closer > 0.0
    assert reward_away < 0.0


def test_success_judgment_consistent_with_reward_bonus() -> None:


    goal = straight_goal()
    constants = RewardConstants()
    d_success = distance_to_goal(straight_line(offset_x=0.03), goal, LENGTH)
    d_fail = distance_to_goal(straight_line(offset_x=0.30), goal, LENGTH)
    assert d_success < EPS_SUCC_COEFF < d_fail

    assert is_success(d_success, LENGTH)
    assert not is_success(d_fail, LENGTH)

    reward_with, success_with = step_reward(d_fail, d_success, LENGTH, constants)
    reward_without, success_without = step_reward(d_fail, d_fail, LENGTH, constants)
    assert success_with and not success_without
    # the success bonus is exactly R_succ and matches is_success
    base = constants.alpha * (d_fail - d_success) - constants.c_step
    assert reward_with == pytest.approx(base + constants.r_succ)


# ---------------------------------------------------------------------------
# Metric routing: correspondence_l2 only (inherited risk #4)
# ---------------------------------------------------------------------------


def test_reward_module_imports_only_correspondence_l2() -> None:
    tree = ast.parse(inspect.getsource(reward_module))
    distance_imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "dgcc.goals.distance":
            distance_imports |= {alias.name for alias in node.names}
    assert distance_imports == {"correspondence_l2"}


def test_reward_path_never_calls_chamfer_family(monkeypatch: pytest.MonkeyPatch) -> None:
    def _forbidden(*_args: Any, **_kwargs: Any) -> float:
        raise AssertionError("Chamfer family called on the reward/success path")

    monkeypatch.setattr(distance, "D", _forbidden)
    monkeypatch.setattr(distance, "chamfer", _forbidden)
    monkeypatch.setattr(distance, "chamfer_distance", _forbidden)

    goal = straight_goal()
    d = distance_to_goal(straight_line(offset_x=0.2), goal, LENGTH)
    reward, success = step_reward(d + 0.1, d, LENGTH, RewardConstants())
    assert np.isfinite(reward) and isinstance(success, bool)


def test_metric_is_correspondence_not_chamfer() -> None:
    # Numeric fixture where the two metrics disagree: identical straight
    # segments shifted along their own axis overlap for Chamfer but carry a
    # uniform per-index offset for correspondence L2.
    goal = straight_goal()
    x = straight_line(offset_x=0.30)
    g = goal_curve(goal, LENGTH)

    correspondence = distance.correspondence_l2(x, g, LENGTH)
    chamfer_value = distance.chamfer_distance(x, g, LENGTH)
    assert abs(correspondence - chamfer_value) > 0.1  # genuinely different
    assert distance_to_goal(x, goal, LENGTH) == pytest.approx(correspondence)
    assert distance_to_goal(x, goal, LENGTH) != pytest.approx(chamfer_value)


# ---------------------------------------------------------------------------
# Settle budget bound to the actual rollout call path (global rule 7)
# ---------------------------------------------------------------------------


def test_settle_budget_10000_on_every_settle_bearing_call() -> None:
    runner, env = make_runner(n_envs=4)
    goal = straight_goal()
    runner.begin_episodes(seed=3, goals=[goal] * 4)
    rng = np.random.default_rng(0)
    for _ in range(3):
        p, deltas, lifts = random_policy_actions(rng, n_envs=4, n_vertices=32)
        runner.step(p, deltas, lifts, rng=rng)

    budget_calls = [call for call in env.settle_calls if call["max_steps"] == SETTLE_MAX_STEPS]
    assert len(env.settle_calls) >= 4  # 1 light_reset + 3 primitives
    assert len(budget_calls) == len(env.settle_calls), env.settle_calls
    assert all(call["vel_threshold"] == SETTLE_VEL_THRESHOLD for call in env.settle_calls)
    assert {call["method"] for call in env.settle_calls} == {
        "light_reset",
        "step_primitive_batch",
    }


def test_runner_never_uses_step_primitive_single_path() -> None:
    import dgcc.tasks.episode as episode_module

    source = inspect.getsource(episode_module)
    assert "step_primitive_batch" in source
    assert ".step_primitive(" not in source


# ---------------------------------------------------------------------------
# Episode protocol: T=10 termination and early success
# ---------------------------------------------------------------------------


def test_episode_terminates_at_horizon_without_success() -> None:
    runner, env = make_runner(n_envs=2)  # static dynamics: never reaches goal
    far_goal = straight_goal()
    runner.begin_episodes(seed=5, goals=[far_goal] * 2)
    rng = np.random.default_rng(1)

    steps = 0
    while not runner.all_done():
        p, deltas, lifts = random_policy_actions(rng, n_envs=2, n_vertices=32)
        record = runner.step(p, deltas, lifts, rng=rng)
        steps += 1
        assert steps <= EPISODE_HORIZON, "episode failed to terminate at T=10"
    assert steps == EPISODE_HORIZON
    assert not record["success"].any()
    assert (record["t"] == EPISODE_HORIZON).all()


def test_episode_early_termination_on_success() -> None:
    runner, env = make_runner(n_envs=2, approach_fraction=1.0)
    goal = straight_goal()
    env_targets = np.stack([goal_curve(goal, LENGTH)] * 2)
    runner.begin_episodes(seed=9, goals=[goal] * 2)
    env.targets = env_targets

    rng = np.random.default_rng(2)
    p, deltas, lifts = random_policy_actions(rng, n_envs=2, n_vertices=32)
    record = runner.step(p, deltas, lifts, rng=rng)
    assert record["success"].all()
    assert record["done"].all()
    assert runner.all_done()
    assert (record["t"] == 1).all()
    # positive reward: large D improvement plus success bonus
    assert (record["reward"] > 0.0).all()


def test_active_mask_freezes_finished_episodes() -> None:
    runner, env = make_runner(n_envs=2, approach_fraction=1.0)
    goal = straight_goal()
    env.targets = np.stack([goal_curve(goal, LENGTH)] * 2)
    runner.begin_episodes(seed=13, goals=[goal] * 2)
    rng = np.random.default_rng(3)

    p, deltas, lifts = random_policy_actions(rng, n_envs=2, n_vertices=32)
    first = runner.step(p, deltas, lifts, rng=rng)
    assert first["active"].all() and first["done"].all()

    second = runner.step(p, deltas, lifts, rng=rng)
    assert not second["active"].any()
    assert (second["t"] == 1).all()  # t is not advanced for finished episodes


# ---------------------------------------------------------------------------
# NaN covenant (global rule 6, env level)
# ---------------------------------------------------------------------------


def test_nan_covenant_discards_reseeds_and_counts() -> None:
    runner, env = make_runner(n_envs=3)
    goal = straight_goal()
    runner.begin_episodes(seed=21, goals=[goal] * 3)
    rng = np.random.default_rng(4)

    env.fail_next_step_with_nan_env = 1
    p, deltas, lifts = random_policy_actions(rng, n_envs=3, n_vertices=32)
    record = runner.step(p, deltas, lifts, rng=rng)

    assert record["discarded"] is True
    assert record["bad_envs"].tolist() == [1]
    assert runner.nan_incidents == 1
    assert runner.incident_log and runner.incident_log[0]["bad_envs"] == [1]
    # reseeded env restarts its episode with a finite state
    assert np.isfinite(env.get_centerline_batch()).all()
    assert runner.t[1] == 0 and not runner.done[1]
    # the runner re-settled with the immutable budget during recovery
    assert env.settle_calls[-1]["method"] == "light_reset"
    assert env.settle_calls[-1]["max_steps"] == SETTLE_MAX_STEPS

    # the discarded step produced no transition; a normal step still works
    follow_up = runner.step(p, deltas, lifts, rng=rng)
    assert follow_up["discarded"] is False
    assert follow_up["active"].all()


# ---------------------------------------------------------------------------
# Init-vertex batching helper
# ---------------------------------------------------------------------------


def test_build_batch_init_vertices_shapes_and_determinism() -> None:
    params = p1_rope_params()
    verts1, shapes1, seeds1 = build_batch_init_vertices(params, n_envs=8, episode_index=0, seed=42)
    verts2, shapes2, seeds2 = build_batch_init_vertices(params, n_envs=8, episode_index=0, seed=42)
    np.testing.assert_array_equal(verts1, verts2)
    assert shapes1 == shapes2 and seeds1 == seeds2
    assert verts1.shape == (8, 32, 3)
    assert set(shapes1) == {"straight", "u_bend", "s_curve", "random_smooth"}


def test_nan_covenant_covers_reset_settle_path() -> None:
    runner, env = make_runner(n_envs=3)
    goal = straight_goal()
    env.corrupt_reset_env_once = 2
    info = runner.begin_episodes(seed=33, goals=[goal] * 3)

    assert info["reset_reseeded_envs"] == [2]
    assert runner.nan_incidents == 1
    assert np.isfinite(info["d_initial"]).all()
    assert np.isfinite(env.get_centerline_batch()).all()
    # both settle attempts used the immutable budget
    assert all(call["max_steps"] == SETTLE_MAX_STEPS for call in env.settle_calls)
    assert len([c for c in env.settle_calls if c["method"] == "light_reset"]) == 2


def test_nan_covenant_catches_nonfinite_valueerror() -> None:
    # The failed-grasp restoration path surfaces contamination as
    # ValueError("vertices contain non-finite values") — the covenant must
    # treat it identically to FloatingPointError (P0 collector precedent).
    runner, env = make_runner(n_envs=2)
    goal = straight_goal()
    runner.begin_episodes(seed=44, goals=[goal] * 2)

    original = env.step_primitive_batch

    def raising_step(*args, **kwargs):
        env.settle_calls.append({"method": "step_primitive_batch", "vel_threshold": kwargs["vel_threshold"], "max_steps": kwargs["max_steps"]})
        env.state[1] = np.nan
        raise ValueError("vertices contain non-finite values")

    env.step_primitive_batch = raising_step  # type: ignore[method-assign]
    rng = np.random.default_rng(5)
    p, deltas, lifts = random_policy_actions(rng, n_envs=2, n_vertices=32)
    record = runner.step(p, deltas, lifts, rng=rng)
    assert record["discarded"] is True
    assert runner.nan_incidents >= 1
    env.step_primitive_batch = original  # type: ignore[method-assign]
    follow_up = runner.step(p, deltas, lifts, rng=rng)
    assert follow_up["discarded"] is False


def test_auto_reset_keeps_all_envs_active() -> None:
    runner, env = make_runner(n_envs=3, approach_fraction=1.0)
    goal = straight_goal()
    env.targets = np.stack([goal_curve(goal, LENGTH)] * 3)
    runner.begin_episodes(seed=71, goal_fn=lambda i, x, r: goal, auto_reset=True)
    rng = np.random.default_rng(6)

    p, deltas, lifts = random_policy_actions(rng, n_envs=3, n_vertices=32)
    first = runner.step(p, deltas, lifts, rng=rng)
    # transition record reports the terminal step faithfully...
    assert first["success"].all() and first["done"].all()
    assert (first["t"] == 1).all()
    assert runner.episodes_completed == 3 and runner.episodes_succeeded == 3
    # ...but the runner immediately refreshed the finished episodes.
    assert not runner.done.any()
    assert (runner.t == 0).all()
    # env state was re-placed with fresh init curves (targets moved them away
    # from the goal, so D is large again) and the next step counts fully.
    env.targets = None
    second = runner.step(p, deltas, lifts, rng=rng)
    assert second["active"].all()
