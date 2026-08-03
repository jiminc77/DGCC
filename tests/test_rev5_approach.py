"""Rev 5 two-stage approach + attach gate (env-correction, owner-approved 2026-08-03).

The physical claim under test is that the gripper no longer teleports onto the
target node: it walks in under a bounded-deceleration speed profile and only
closes once its velocity RELATIVE to the node is below `ATTACH_REL_VEL`.  Both
halves are exercised without Genesis — the speed profile is a pure function and
the walk loop is driven against a scripted stub solver — so a regression is
caught by `pytest tests/` instead of only by the 2-hour GPU battery.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dgcc.envs.dlolab import (  # noqa: E402
    APPROACH_A_DEC,
    APPROACH_MAX_STEPS,
    APPROACH_R_SLOW,
    APPROACH_V_FAST,
    APPROACH_V_SLOW,
    ATTACH_REL_VEL,
    DLOLabEnv,
    HOLD_QUIESCENT_VEL,
    approach_brake_distance,
    approach_speed,
)

SETTLE_VEL_THRESHOLD = 1.0e-3


# --------------------------------------------------------------- speed profile


def test_speed_profile_is_capped_inside_the_slow_radius() -> None:
    inside = np.linspace(0.0, APPROACH_R_SLOW, 25)
    assert np.all(approach_speed(inside) <= APPROACH_V_SLOW + 1e-12)
    assert approach_speed(np.array([APPROACH_R_SLOW]))[0] == pytest.approx(APPROACH_V_SLOW)


def test_speed_profile_reaches_free_space_speed_beyond_the_braking_band() -> None:
    brake = approach_brake_distance()
    # Analytic band width: (v_fast^2 - v_slow^2) / (2a).
    assert brake == pytest.approx(
        (APPROACH_V_FAST**2 - APPROACH_V_SLOW**2) / (2.0 * APPROACH_A_DEC)
    )
    assert approach_speed(np.array([APPROACH_R_SLOW + brake]))[0] == pytest.approx(
        APPROACH_V_FAST
    )
    far = np.array([APPROACH_R_SLOW + brake + 0.5, 2.0])
    assert np.all(approach_speed(far) == APPROACH_V_FAST)


def test_speed_profile_respects_the_declared_deceleration_limit() -> None:
    """v(d)^2 - v(d-h)^2 <= 2*a*h everywhere: the ramp is realizable by the arm."""

    d = np.linspace(0.0, 0.4, 4001)
    v = approach_speed(d)
    h = float(d[1] - d[0])
    implied = np.diff(v**2) / (2.0 * h)
    assert implied.max() <= APPROACH_A_DEC + 1e-6


def test_speed_profile_follows_the_slow_cap_supplied_by_the_caller() -> None:
    slow = 0.05
    assert approach_speed(np.array([0.0]), slow)[0] == pytest.approx(slow)
    with pytest.raises(ValueError):
        approach_speed(np.array([0.1]), 0.0)


def test_attach_gate_sits_between_the_two_existing_quiescence_scales() -> None:
    """Derived threshold, not a free parameter: one decade above settle,
    5x tighter than the realized release criterion (AT-3)."""

    assert ATTACH_REL_VEL == pytest.approx(10.0 * SETTLE_VEL_THRESHOLD)
    assert ATTACH_REL_VEL == pytest.approx(HOLD_QUIESCENT_VEL / 5.0)
    assert SETTLE_VEL_THRESHOLD < ATTACH_REL_VEL < HOLD_QUIESCENT_VEL


# ------------------------------------------------------------------ walk loop


class _StubRod:
    """Records attach calls; the walk loop needs nothing else from the entity."""

    def __init__(self) -> None:
        self.per_env_attachments: list[tuple[int, int]] = []
        self.all_env_attachments: list[int] = []

    def attach_to_rigid_link_with_envs_idx(self, _link, nodes, env_idx) -> None:
        self.per_env_attachments.append((int(env_idx), int(nodes[0])))

    def attach_to_rigid_link(self, _link, nodes) -> None:
        self.all_env_attachments.append(int(nodes[0]))


def make_env(
    n_envs: int,
    node_positions: np.ndarray,
    node_velocities: np.ndarray,
    gripper_start: np.ndarray,
) -> DLOLabEnv:
    """Adapter instance wired to a scripted kinematic stub (no Genesis)."""

    env = object.__new__(DLOLabEnv)
    env.n_envs = n_envs
    env.dt = 1.0e-3
    env.quasi_static = True
    env.move_v_max = APPROACH_V_SLOW
    env.rod_entity = _StubRod()
    env.gripper_link = object()
    env.active_node = None
    env._batched_active_nodes = np.full(n_envs, -1, dtype=int)
    env.last_approach_steps = 0
    env.last_approach_dwell_steps = 0
    env.last_approach_gate_failures = 0
    env.last_attach_rel_vel = np.full(n_envs, np.nan)
    env.last_attach_offset = np.full(n_envs, np.nan)

    state = {"pos": np.asarray(node_positions, dtype=float).copy()}
    vel = np.asarray(node_velocities, dtype=float)
    commanded = {"pos": np.asarray(gripper_start, dtype=float).copy()}
    env.scene_steps = 0

    env._raw_batch = lambda: state["pos"].copy()
    env._node_velocities_batch = lambda: vel.copy()
    env._gripper_positions = lambda: commanded["pos"].copy()

    def set_gripper(positions: np.ndarray) -> None:
        commanded["pos"] = np.asarray(positions, dtype=float).copy()

    def step_scene() -> None:
        env.scene_steps += 1
        state["pos"] = state["pos"] + vel * env.dt

    env._set_gripper_positions = set_gripper
    env._step_scene = step_scene
    return env


def test_quiescent_rope_attaches_and_never_exceeds_the_slow_cap_near_the_node() -> None:
    node = np.array([[[0.30, 0.0, 0.005]]])  # (1 env, 1 node, 3)
    env = make_env(
        1,
        node,
        np.zeros_like(node),
        np.array([[0.0, 0.0, 0.15]]),
    )
    speeds = []
    original = env._set_gripper_positions
    previous = {"pos": env._gripper_positions()}

    def recording(positions: np.ndarray) -> None:
        pos = np.asarray(positions, dtype=float)
        speeds.append(np.linalg.norm(pos - previous["pos"], axis=1) / env.dt)
        previous["pos"] = pos.copy()
        original(positions)

    env._set_gripper_positions = recording
    result = env._approach_and_attach(
        np.zeros(1, dtype=int), np.ones(1, dtype=bool), per_env=True
    )

    assert bool(result["attached"].all())
    assert env.rod_entity.per_env_attachments == [(0, 0)]
    assert env._batched_active_nodes.tolist() == [0]
    # No teleport: the whole transit is resolved in many bounded steps.
    assert result["walk_steps"] > 300
    assert env.scene_steps == result["walk_steps"] + result["dwell_steps"]
    commanded_speed = np.concatenate(speeds)
    assert commanded_speed.max() <= APPROACH_V_FAST + 1e-9
    # The final approach — the last r_slow of travel — is at manipulation speed.
    slow_leg = commanded_speed[-int(APPROACH_R_SLOW / (APPROACH_V_SLOW * env.dt)) :]
    assert slow_leg.max() <= APPROACH_V_SLOW + 1e-9
    # Attach happened essentially on top of the node with ~zero relative speed.
    assert result["rel_vel_at_attach"][0] < ATTACH_REL_VEL
    assert result["offset_at_attach"][0] <= APPROACH_V_SLOW * env.dt


def test_moving_node_that_never_quiesces_is_a_grasp_failure_not_a_forced_attach() -> None:
    node = np.array([[[0.02, 0.0, 0.005]]])
    # 0.5 m/s node: the gripper can track its position but never gets the
    # relative speed under the gate while its own command lags by one step.
    vel = np.array([[[0.0, 0.0, 0.5]]])
    env = make_env(1, node, vel, np.array([[0.0, 0.0, 0.005]]))
    result = env._approach_and_attach(
        np.zeros(1, dtype=int), np.ones(1, dtype=bool), per_env=True
    )
    assert not bool(result["attached"].any())
    assert result["gate_failures"] == 1
    assert env.rod_entity.per_env_attachments == []
    assert env._batched_active_nodes.tolist() == [-1]
    assert env.scene_steps == APPROACH_MAX_STEPS


def test_ineligible_envs_are_never_approached_or_attached() -> None:
    nodes = np.array(
        [
            [[0.10, 0.0, 0.005], [0.20, 0.0, 0.005]],
            [[0.10, 0.0, 0.005], [0.20, 0.0, 0.005]],
        ]
    )
    env = make_env(
        2,
        nodes,
        np.zeros_like(nodes),
        np.array([[0.0, 0.0, 0.03], [0.0, 0.0, 0.03]]),
    )
    eligible = np.array([True, False])
    result = env._approach_and_attach(
        np.array([0, 1], dtype=int), eligible, per_env=True
    )
    assert result["attached"].tolist() == [True, False]
    assert env.rod_entity.per_env_attachments == [(0, 0)]
    assert env._batched_active_nodes.tolist() == [0, -1]
    assert result["gate_failures"] == 0
    # The ineligible env's gripper never moved off its parking pose.
    assert env._gripper_positions()[1].tolist() == [0.0, 0.0, 0.03]


def test_all_env_attach_path_waits_for_every_environment() -> None:
    # Two envs, same node index, different distances: the shared-attach path
    # must not fire until BOTH have cleared the gate.
    nodes = np.array([[[0.05, 0.0, 0.005]], [[0.35, 0.0, 0.005]]])
    env = make_env(
        2,
        nodes,
        np.zeros_like(nodes),
        np.array([[0.0, 0.0, 0.03], [0.0, 0.0, 0.03]]),
    )
    result = env._approach_and_attach(
        np.zeros(2, dtype=int), np.ones(2, dtype=bool), per_env=False
    )
    assert bool(result["attached"].all())
    assert env.rod_entity.all_env_attachments == [0]
    assert env.rod_entity.per_env_attachments == []
    assert env.active_node == 0
    assert np.all(result["rel_vel_at_attach"] < ATTACH_REL_VEL)


def test_all_env_attach_path_rejects_mixed_target_nodes() -> None:
    nodes = np.array([[[0.05, 0.0, 0.005], [0.1, 0.0, 0.005]]] * 2)
    env = make_env(
        2, nodes, np.zeros_like(nodes), np.zeros((2, 3))
    )
    with pytest.raises(ValueError, match="one shared target node"):
        env._approach_and_attach(
            np.array([0, 1], dtype=int), np.ones(2, dtype=bool), per_env=False
        )


def test_approach_step_classification_matches_the_probe_rule() -> None:
    """walk vs dwell must be the acceptance probe's move/hold rule verbatim,
    otherwise the AT-6 denominator correction is wrong."""

    node = np.array([[[0.001, 0.0, 0.0]]])
    env = make_env(1, node, np.zeros_like(node), np.zeros((1, 3)))
    commands: list[np.ndarray] = []
    original = env._set_gripper_positions

    def recording(positions: np.ndarray) -> None:
        commands.append(np.asarray(positions, dtype=float).copy())
        original(positions)

    env._set_gripper_positions = recording
    result = env._approach_and_attach(
        np.zeros(1, dtype=int), np.ones(1, dtype=bool), per_env=True
    )
    start = np.zeros((1, 3))
    expected_moves = sum(
        0 if np.array_equal(cmd, start if i == 0 else commands[i - 1]) else 1
        for i, cmd in enumerate(commands)
    )
    expected_holds = len(commands) - expected_moves
    assert result["walk_steps"] == expected_moves
    assert result["dwell_steps"] == expected_holds


# ------------------------------------------------- Rev 7 release gate


def test_release_gate_threshold_and_budget_match_the_measured_distribution() -> None:
    """Owner-adjudicated values, derived from the 80-primitive distribution
    (residual strain at natural release p50 6.90e-4 / p95 2.26e-3 / max 4.04e-3).
    Pinned here so a silent edit cannot drift them."""

    from dgcc.envs.dlolab import RELEASE_STRAIN_MAX_STEPS, RELEASE_STRAIN_THRESHOLD

    assert RELEASE_STRAIN_THRESHOLD == 7.5e-4
    assert RELEASE_STRAIN_MAX_STEPS == 600
    # Must sit below the Rev 4 ep44 p0 release value (1.21e-3), which still
    # whipped, and above the observed floor (1.96e-5).
    assert 1.96e-5 < RELEASE_STRAIN_THRESHOLD < 1.21e-3


def test_release_gate_is_best_effort_not_a_grasp_failure() -> None:
    """Budget exhaustion must release anyway and be COUNTED, never converted
    into a grasp failure -- otherwise grasp-success statistics and the
    failure-restoration path move together and nothing is attributable."""

    import inspect

    from dgcc.envs.dlolab import DLOLabEnv

    body = inspect.getsource(DLOLabEnv._execute_move)
    gate = body[body.index("RELEASE_STRAIN_MAX_STEPS"):]
    assert "last_release_strain_relaxed" in gate
    # the gate loop must not raise or flag failure
    assert "raise" not in gate
    assert "grasp_success" not in gate
