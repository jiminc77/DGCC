"""V2-DEV cadence, selection-objective, diagnostics, and BGT contracts (CPU)."""

from __future__ import annotations

import copy
import json
import types

import numpy as np
import pytest
import torch

from dgcc.rl.selection import (
    CONTACT_WEIGHT_BETA,
    compare_selection_snapshots,
    contact_softmax_weights,
    selection_snapshot,
)
from dgcc.rl.sprint_arms import SprintTD3Agent, create_sprint_agent
from dgcc.rl.td3 import TD3Agent, TD3Config
from dgcc.rl.v2_arms import (
    BGTAgent,
    SelectionWeightedTD3Agent,
    selection_weighted_actor_loss,
)

K = 32


def curve(offset: float = 0.0) -> np.ndarray:
    t = np.linspace(-0.5, 0.5, K)
    return np.column_stack((t + offset, 0.1 * np.sin(np.pi * t), np.zeros(K)))


def batch(size: int = 3, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "X_before": np.stack([curve(0.001 * i) for i in range(size)]),
        "X_after": np.stack([curve(0.001 * i + 0.01) for i in range(size)]),
        "goal_curve": np.stack([curve(0.03 + 0.001 * i) for i in range(size)]),
        "p": rng.integers(0, K, size=size),
        "delta": rng.uniform(-0.03, 0.03, size=(size, 3)),
        "lift": rng.integers(0, 2, size=size),
        "reward": rng.normal(size=size),
        "done": np.zeros(size, dtype=bool),
    }


def module_state(agent: TD3Agent) -> list[torch.Tensor]:
    modules = (
        agent.encoder,
        agent.critic,
        agent.actor,
        agent.encoder_target,
        agent.critic_target,
        agent.actor_target,
    )
    return [
        value.detach().clone()
        for module in modules
        for value in module.state_dict().values()
    ]


def test_policy_delay_two_updates_actor_and_targets_every_second_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = TD3Agent(TD3Config(policy_delay=2))
    calls = {"critic": 0, "actor": 0, "target": 0}
    monkeypatch.setattr(agent, "features", lambda *args, **kwargs: torch.empty(0))

    def critic_update(self, _batch, **_kwargs):
        calls["critic"] += 1
        return {"critic_loss": 1.0}

    def actor_update(self, _batch, **_kwargs):
        calls["actor"] += 1
        return {"actor_loss": 2.0}

    def target_update(self):
        calls["target"] += 1

    monkeypatch.setattr(agent, "critic_update", types.MethodType(critic_update, agent))
    monkeypatch.setattr(agent, "actor_update", types.MethodType(actor_update, agent))
    monkeypatch.setattr(
        agent, "soft_update_targets", types.MethodType(target_update, agent)
    )

    empty_batch = {"X_before": None, "goal_curve": None}
    flags = [agent.update(empty_batch)["actor_updated"] for _ in range(5)]
    assert flags == [0.0, 1.0, 0.0, 1.0, 0.0]
    assert calls == {"critic": 5, "actor": 2, "target": 2}
    assert agent.update_count == 5
    assert agent.actor_update_count == agent.target_update_count == 2


def test_policy_delay_one_matches_native_update_sequence() -> None:
    torch.manual_seed(12)
    agent = TD3Agent(TD3Config(policy_delay=1, policy_noise=0.0), device="cpu")
    native = copy.deepcopy(agent)
    sample = batch(size=2, seed=4)

    agent.update(sample)
    feats = native.features(sample["X_before"], sample["goal_curve"])
    native.critic_update(sample, feats_before=feats)
    native.actor_update(sample, feats_before=feats)
    native.soft_update_targets()

    for actual, expected in zip(module_state(agent), module_state(native), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert agent.actor_update_count == agent.target_update_count == 1


def test_cadence_counts_round_trip_in_checkpoint(tmp_path) -> None:
    agent = TD3Agent(TD3Config(policy_delay=2))
    agent.update_count = 7
    agent.actor_update_count = 3
    agent.target_update_count = 3
    path = agent.save_checkpoint(tmp_path / "cadence.pt")
    restored = TD3Agent(TD3Config(policy_delay=2))
    restored.load_checkpoint(path)
    assert (
        restored.update_count,
        restored.actor_update_count,
        restored.target_update_count,
    ) == (
        7,
        3,
        3,
    )


def test_v2_checkpoint_round_trip_requires_matching_candidate_metadata(
    tmp_path,
) -> None:
    agent = SelectionWeightedTD3Agent(
        TD3Config(), weight_operator="qmin", eval_operator="qmin"
    )
    path = agent.save_checkpoint(tmp_path / "v2.pt")
    restored = SelectionWeightedTD3Agent(
        TD3Config(), weight_operator="qmin", eval_operator="qmin"
    )
    restored.load_checkpoint(path)
    assert restored.to_dict()["v2_arm"] == agent.to_dict()["v2_arm"]
    incompatible = SelectionWeightedTD3Agent(
        TD3Config(), weight_operator="q1", eval_operator="qmin"
    )
    with pytest.raises(ValueError, match="incompatible"):
        incompatible.load_checkpoint(path)


def test_constant_scores_use_uniform_detached_weights() -> None:
    scores = torch.full((2, K), 12345.0, requires_grad=True)
    weights = contact_softmax_weights(scores)
    torch.testing.assert_close(weights, torch.full_like(weights, 1.0 / K))
    assert not weights.requires_grad


def test_weight_path_has_no_critic_gradient() -> None:
    q1 = torch.tensor([[3.0, 2.0, 1.0]], requires_grad=True)
    q2 = torch.tensor([[0.3, 0.2, 0.1]], requires_grad=True)
    loss, weights = selection_weighted_actor_loss(
        q1, q2, weight_operator="q1", eval_operator="qmin"
    )
    loss.backward()
    torch.testing.assert_close(q1.grad, torch.zeros_like(q1))
    torch.testing.assert_close(q2.grad, -weights)
    assert not weights.requires_grad


@pytest.mark.parametrize(
    ("weight_operator", "eval_operator"),
    (("qmin", "qmin"), ("q1", "qmin"), ("q1", "q1")),
)
def test_operator_combinations_match_manual_formula(
    weight_operator, eval_operator
) -> None:
    q1 = torch.tensor([[0.3, 0.1, -0.2], [0.4, 0.0, 0.2]])
    q2 = torch.tensor([[0.2, 0.0, -0.1], [0.1, 0.3, 0.2]])
    loss, weights = selection_weighted_actor_loss(
        q1,
        q2,
        weight_operator=weight_operator,
        eval_operator=eval_operator,
    )
    qmin = torch.minimum(q1, q2)
    scores = q1 if weight_operator == "q1" else qmin
    values = q1 if eval_operator == "q1" else qmin
    expected_weights = torch.softmax(scores / CONTACT_WEIGHT_BETA, dim=1)
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(loss, -(expected_weights * values).sum(dim=1).mean())


def test_uniform_fallback_matches_native_all_contact_objective() -> None:
    q1 = torch.full((2, K), 2.0)
    q2 = torch.arange(2 * K, dtype=torch.float32).reshape(2, K) / 100.0
    loss, weights = selection_weighted_actor_loss(
        q1, q2, weight_operator="q1", eval_operator="qmin"
    )
    qmin = torch.minimum(q1, q2)
    torch.testing.assert_close(weights, torch.full_like(weights, 1.0 / K))
    torch.testing.assert_close(loss, -qmin.mean())


def test_policy_delay_one_plus_uniform_weights_matches_native_update() -> None:
    config = TD3Config(policy_delay=1, policy_noise=0.0)
    torch.manual_seed(44)
    native = SprintTD3Agent(config, arm="v1")
    torch.manual_seed(44)
    candidate = SelectionWeightedTD3Agent(
        config, weight_operator="q1", eval_operator="qmin"
    )

    def make_constant_q1(agent):
        original = agent._q_all_candidates

        def q_candidates(self, head, h, u):
            if head is self.critic.q1:
                return torch.full(h.shape[:2], 100.0, dtype=h.dtype, device=h.device)
            return original(head, h, u)

        agent._q_all_candidates = types.MethodType(q_candidates, agent)

    make_constant_q1(native)
    make_constant_q1(candidate)
    sample = batch(size=2, seed=9)
    native.update(sample)
    candidate.update(sample)
    for actual, expected in zip(
        module_state(candidate), module_state(native), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_beta_contact_is_locked_and_metadata_never_uses_contact_tau() -> None:
    agent = SelectionWeightedTD3Agent(
        TD3Config(), weight_operator="qmin", eval_operator="qmin"
    )
    assert agent.to_dict()["v2_arm"]["beta_contact"] == 0.010
    assert "tau" not in agent.to_dict()["v2_arm"]
    with pytest.raises(ValueError, match="charter-locked"):
        SelectionWeightedTD3Agent(
            TD3Config(),
            weight_operator="qmin",
            eval_operator="qmin",
            beta_contact=0.025,
        )


def test_selection_actor_update_reuses_one_actor_and_twin_critic_forward() -> None:
    agent = SelectionWeightedTD3Agent(
        TD3Config(), weight_operator="qmin", eval_operator="qmin"
    )
    calls = {"actor": 0, "q1": 0, "q2": 0}
    hooks = [
        agent.actor.register_forward_hook(
            lambda *_: calls.__setitem__("actor", calls["actor"] + 1)
        ),
        agent.critic.q1.register_forward_hook(
            lambda *_: calls.__setitem__("q1", calls["q1"] + 1)
        ),
        agent.critic.q2.register_forward_hook(
            lambda *_: calls.__setitem__("q2", calls["q2"] + 1)
        ),
    ]
    try:
        stats = agent.actor_update(batch(size=2))
    finally:
        for hook in hooks:
            hook.remove()
    assert calls == {"actor": 1, "q1": 1, "q2": 1}
    assert stats["contact_weight_neff"] > 1.0


def test_factory_maps_all_three_operator_candidates() -> None:
    expected = {
        "v2-dmm": ("qmin", "qmin"),
        "v2-d1m": ("q1", "qmin"),
        "v2-d11": ("q1", "q1"),
    }
    for arm, operators in expected.items():
        agent = create_sprint_agent(arm, TD3Config())
        assert isinstance(agent, SelectionWeightedTD3Agent)
        assert (agent.weight_operator, agent.eval_operator) == operators
        assert agent.candidate_id == arm


def test_snapshot_comparison_is_exact_on_identical_panel() -> None:
    q1 = torch.arange(2 * K, dtype=torch.float32).reshape(2, K)
    q2 = q1.flip(1)
    weights = contact_softmax_weights(q1)
    snapshot = selection_snapshot(q1, q2, weights).cpu()
    stats = compare_selection_snapshots(snapshot, snapshot)
    assert stats == pytest.approx(
        {
            "soft_weight_js_to_previous_checkpoint": 0.0,
            "soft_weight_cosine_to_previous_checkpoint": 1.0,
            "top8_contact_overlap": 1.0,
            "hard_q1_churn": 0.0,
            "hard_qmin_churn": 0.0,
        }
    )


def test_selection_panel_exposes_all_lift_diagnostics() -> None:
    agent = SelectionWeightedTD3Agent(
        TD3Config(), weight_operator="qmin", eval_operator="qmin"
    )
    X = np.stack([curve(), curve(0.01)])
    G = np.stack([curve(0.03), curve(0.04)])
    stats, snapshot = agent.selection_panel(X, G)
    required = {
        "lift_near_threshold_all_045_055",
        "lift_near_threshold_selected_045_055",
        "lift_flip_rate_under_plusminus_002",
        "q_continuous_lift_minus_q_hard_lift",
        "q_at_lift_1_minus_q_at_lift_0",
        "selected_lift_entropy",
    }
    assert required <= stats.keys()
    assert snapshot.weights.shape == (2, K)


def test_bgt_is_fail_closed_without_r5_fields() -> None:
    with pytest.raises(ValueError, match="fail-closed"):
        create_sprint_agent("v2-bgt", TD3Config())


def test_counterfactual_qmin_selector_uses_clipped_twin_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = TD3Agent(TD3Config())
    q1 = torch.zeros((2, K))
    q2 = torch.zeros((2, K))
    q1[:, 0] = 10.0
    q1[:, 1] = 5.0
    q2[:, 0] = -10.0
    q2[:, 1] = 4.0
    monkeypatch.setattr(agent, "features", lambda X, G: torch.zeros((len(X), K, 256)))
    agent.encoder = torch.nn.Identity()
    monkeypatch.setattr(
        agent.actor, "forward", lambda h: torch.zeros((*h.shape[:2], 4))
    )

    def q_candidates(self, head, _h, _u):
        return q1 if head is self.critic.q1 else q2

    monkeypatch.setattr(
        agent, "_q_all_candidates", types.MethodType(q_candidates, agent)
    )
    X = np.stack([curve(), curve(0.01)])
    G = np.stack([curve(0.03), curve(0.04)])
    p_q1, *_ = agent.select_actions(
        X,
        G,
        step=0,
        total_budget=1,
        rng=np.random.default_rng(0),
        deterministic=True,
        selector_operator="q1",
    )
    p_qmin, *_ = agent.select_actions(
        X,
        G,
        step=0,
        total_budget=1,
        rng=np.random.default_rng(0),
        deterministic=True,
        selector_operator="qmin",
    )
    np.testing.assert_array_equal(p_q1, [0, 0])
    np.testing.assert_array_equal(p_qmin, [1, 1])


def test_bgt_gate_is_deterministic_and_scores_only_actor_top_two() -> None:
    agent = BGTAgent(
        TD3Config(),
        margin=1.0e9,
        onset_transition=100,
        calibration_sha256="a" * 64,
    )
    X = np.stack([curve(), curve(0.01), curve(0.02)])
    G = np.stack([curve(0.03), curve(0.04), curve(0.05)])
    calls: list[int] = []
    hook = agent.f_resp.register_forward_hook(
        lambda _m, inputs, _out: calls.append(inputs[0].shape[0])
    )
    try:
        first = agent.select_actions(
            X,
            G,
            step=100,
            total_budget=300_000,
            rng=np.random.default_rng(9),
            deterministic=True,
            return_info=True,
        )
        second = agent.select_actions(
            X,
            G,
            step=100,
            total_budget=300_000,
            rng=np.random.default_rng(123),
            deterministic=True,
            return_info=True,
        )
    finally:
        hook.remove()
    for left, right in zip(first[:3], second[:3], strict=True):
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right
    assert calls == [2 * len(X), 2 * len(X)]
    p, _, _, info = first
    assert np.all(info["bgt_eligible"])
    assert all(
        contact in pair for contact, pair in zip(p, info["bgt_top2"], strict=True)
    )


def test_bgt_before_onset_is_native_q1_and_draws_no_deterministic_rng() -> None:
    agent = BGTAgent(
        TD3Config(),
        margin=1.0e9,
        onset_transition=100,
        calibration_sha256="b" * 64,
    )
    X = np.stack([curve(), curve(0.01)])
    G = np.stack([curve(0.03), curve(0.04)])
    rng = np.random.default_rng(5)
    before = json.dumps(rng.bit_generator.state, sort_keys=True)
    p, _, _, info = agent.select_actions(
        X,
        G,
        step=99,
        total_budget=300_000,
        rng=rng,
        deterministic=True,
        return_info=True,
    )
    after = json.dumps(rng.bit_generator.state, sort_keys=True)
    np.testing.assert_array_equal(p, info["bgt_top2"][:, 0])
    assert not np.any(info["bgt_eligible"])
    assert before == after
