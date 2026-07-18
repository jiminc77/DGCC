from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from dgcc.phi.dct import Phi_DCT
from dgcc.rl.sprint_arms import (
    MATCHED_PROJECTION_SEED,
    RANDOM_TARGET_SEED,
    ResponseHead,
    SprintTD3Agent,
    create_sprint_agent,
    delta_m_from_batch,
    matched_projection,
    random_target,
)
from dgcc.rl.td3 import TD3Agent, TD3Config, select_p_star


def curve(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 32)
    return np.column_stack((t, .1 * np.sin(2 * np.pi * t + rng.random()), .02 * t))


def batch(n: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(21)
    return {"X_before": np.stack([curve(i) for i in range(n)]), "X_after": np.stack([curve(30 + i) for i in range(n)]), "goal_curve": np.stack([curve(60 + i) for i in range(n)]), "p": rng.integers(0, 32, n), "delta": rng.uniform(-.1, .1, (n, 3)), "lift": rng.integers(0, 2, n), "reward": rng.normal(size=n), "done": np.zeros(n, dtype=bool)}


def digest(agent: TD3Agent) -> str:
    h = hashlib.sha256()
    for module in (agent.encoder, agent.critic, agent.actor, agent.encoder_target, agent.critic_target, agent.actor_target):
        for value in module.state_dict().values(): h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def assert_nested_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right, strict=True):
            assert_nested_equal(left_value, right_value)
    else:
        assert left == right




def test_response_contract_and_dct_target() -> None:
    head = ResponseHead()
    h, u = torch.randn(3, 256), torch.randn(3, 4)
    assert head.z_resp(h, u).shape == (3, 256)
    assert head(h, u).shape == (3, 24)
    assert head.input.in_features == 260
    b = batch()
    expected = np.stack([Phi_DCT(a) - Phi_DCT(b) for b, a in zip(b["X_before"], b["X_after"], strict=True)])
    np.testing.assert_allclose(delta_m_from_batch(b), expected)
    assert not torch.allclose(head(h, u), head(h, u + 1))


def test_aux_isolates_actor_and_targets() -> None:
    agent = SprintTD3Agent(TD3Config(policy_noise=0.0))
    b = batch()
    actor = {k: v.clone() for k, v in agent.actor.state_dict().items()}
    targets = {name: {k: v.clone() for k, v in getattr(agent, name).state_dict().items()} for name in ("encoder_target", "critic_target", "actor_target")}
    stats = agent.critic_update(b)
    assert stats["aux_loss"] > 0
    assert any(p.grad is not None for p in agent.encoder.parameters())
    assert any(p.grad is not None for p in agent.f_resp.parameters())
    assert all(p.grad is None for p in agent.actor.parameters())
    assert all(torch.equal(v, agent.actor.state_dict()[k]) for k, v in actor.items())
    for name, saved in targets.items(): assert all(torch.equal(v, getattr(agent, name).state_dict()[k]) for k, v in saved.items())


def test_lambda_zero_matches_baseline_and_rng_init_is_preserved() -> None:
    config = TD3Config(policy_noise=0.0)
    torch.manual_seed(7); base = TD3Agent(config)
    torch.manual_seed(7); sprint = SprintTD3Agent(config, aux_weight=0.0)
    assert digest(base) == digest(sprint)
    b = batch()
    g1, g2 = torch.Generator().manual_seed(4), torch.Generator().manual_seed(4)
    baseline = base.critic_update(b, generator=g1)
    adapted = sprint.critic_update(b, generator=g2)
    assert adapted.keys() == baseline.keys()
    for key in baseline:
        assert adapted[key] == pytest.approx(baseline[key], abs=1e-7)
    assert torch.equal(g1.get_state(), g2.get_state())
    for left, right in zip(base.encoder.parameters(), sprint.encoder.parameters(), strict=True): assert torch.equal(left, right)
    for left, right in zip(base.critic.parameters(), sprint.critic.parameters(), strict=True): assert torch.equal(left, right)
    base_optimizer = base.critic_optimizer.state_dict()
    sprint_optimizer = sprint.critic_optimizer.state_dict()
    assert_nested_equal(base_optimizer, {
        "state": {key: sprint_optimizer["state"][key] for key in base_optimizer["state"]},
        "param_groups": sprint_optimizer["param_groups"][:-1],
    })
    torch.manual_seed(7); base = TD3Agent(config); base.critic_update(b, generator=torch.Generator().manual_seed(4)); base_next = torch.rand(1)
    torch.manual_seed(7); sprint = SprintTD3Agent(config, aux_weight=0.0); sprint.critic_update(b, generator=torch.Generator().manual_seed(4)); sprint_next = torch.rand(1)
    assert torch.equal(base_next, sprint_next)

def test_bb_factory_is_baseline_and_sprint_arms_are_available() -> None:
    assert type(create_sprint_agent("bb")) is TD3Agent
    assert isinstance(create_sprint_agent("matched"), SprintTD3Agent)
    assert isinstance(create_sprint_agent("random"), SprintTD3Agent)


class TestMatched:
    def test_projection_is_reproducible_and_orthonormal(self) -> None:
        first = matched_projection(MATCHED_PROJECTION_SEED)
        second = matched_projection(MATCHED_PROJECTION_SEED)
        assert torch.equal(first, second)
        torch.testing.assert_close(first @ first.T, torch.eye(24), atol=1e-6, rtol=1e-6)

    def test_target_is_stop_grad_and_uses_baseline_p_star(self) -> None:
        agent = SprintTD3Agent(TD3Config(policy_noise=0.0), arm="matched")
        b = batch()
        target = agent.matched_target(b)
        assert target.requires_grad is False
        assert all(parameter.grad is None for parameter in agent.encoder_target.parameters())
        assert agent.projection.requires_grad is False
        with torch.no_grad():
            h_next = agent.encoder_target(agent.features(b["X_after"], b["goal_curve"]))
            u_all = agent.actor_target(h_next)
            candidates = agent._q_all_candidates(agent.critic_target.q1, h_next, u_all)
            p_star = select_p_star(candidates)
            expected = agent.projection @ h_next[
                torch.arange(h_next.shape[0]), p_star
            ].unsqueeze(-1)
        torch.testing.assert_close(target, expected.squeeze(-1))

    def test_head_is_v1_isomorphic_and_aux_leaves_target_grads_empty(self) -> None:
        v1 = SprintTD3Agent(TD3Config(policy_noise=0.0), arm="v1")
        matched = SprintTD3Agent(TD3Config(policy_noise=0.0), arm="matched")
        v1_shapes = [parameter.shape for parameter in v1.f_resp.parameters()]
        matched_shapes = [parameter.shape for parameter in matched.f_resp.parameters()]
        assert matched_shapes == v1_shapes
        assert sum(parameter.numel() for parameter in matched.f_resp.parameters()) == sum(
            parameter.numel() for parameter in v1.f_resp.parameters()
        )
        matched.critic_update(batch())
        assert all(parameter.grad is None for parameter in matched.encoder_target.parameters())
        assert matched.projection.grad is None

    def test_uses_baseline_ema_tau(self) -> None:
        agent = SprintTD3Agent(TD3Config(tau=0.005), arm="matched")
        online = next(agent.encoder.parameters())
        target = next(agent.encoder_target.parameters())
        with torch.no_grad():
            online.fill_(1.0)
            target.zero_()
        agent.soft_update_targets()
        torch.testing.assert_close(target, torch.full_like(target, 0.005))
        assert agent.config.tau == pytest.approx(0.005)

    def test_critic_optimizer_excludes_projection_buffer(self) -> None:
        agent = SprintTD3Agent(arm="matched")
        optimizer_param_ids = {
            id(parameter)
            for group in agent.critic_optimizer.param_groups
            for parameter in group["params"]
        }
        assert id(agent.projection) not in optimizer_param_ids

    def test_checkpoint_records_seed_and_regenerates_projection(self, tmp_path) -> None:
        source = SprintTD3Agent(arm="matched", projection_seed=MATCHED_PROJECTION_SEED)
        path = source.save_checkpoint(tmp_path / "matched.pt")
        payload = torch.load(path, weights_only=False)
        assert payload["sprint_arm"]["projection_seed"] == MATCHED_PROJECTION_SEED
        assert "P" not in payload["sprint_arm"]
        restored = SprintTD3Agent(arm="matched", projection_seed=1)
        restored.load_checkpoint(path)
        assert restored.projection_seed == MATCHED_PROJECTION_SEED
        assert torch.equal(restored.projection, source.projection)

    @pytest.mark.parametrize(("source_arm", "destination_arm"), [("matched", "random"), ("random", "matched")])
    def test_cross_arm_v2_checkpoint_is_rejected(
        self, source_arm: str, destination_arm: str, tmp_path
    ) -> None:
        path = SprintTD3Agent(arm=source_arm).save_checkpoint(tmp_path / f"{source_arm}.pt")
        assert torch.load(path, weights_only=False)["sprint_arm"]["schema_version"] == 2
        with pytest.raises(ValueError, match="incompatible sprint checkpoint"):
            SprintTD3Agent(arm=destination_arm).load_checkpoint(path)


class TestRandom:
    def test_fixed_across_updates_and_regeneration(self) -> None:
        agent = SprintTD3Agent(TD3Config(policy_noise=0.0), arm="random")
        expected = agent.random_target.clone()
        agent.critic_update(batch())
        assert torch.equal(agent.random_target, expected)
        assert torch.equal(SprintTD3Agent(arm="random").random_target, expected)

    def test_target_is_bitwise_stable_across_full_update_and_soft_update(self) -> None:
        agent = SprintTD3Agent(TD3Config(policy_noise=0.0), arm="random")
        expected = agent.random_target.detach().cpu().numpy().tobytes()
        agent.update(batch())
        assert agent.random_target.detach().cpu().numpy().tobytes() == expected
        agent.soft_update_targets()
        assert agent.random_target.detach().cpu().numpy().tobytes() == expected

    def test_critic_optimizer_excludes_random_target_buffer(self) -> None:
        agent = SprintTD3Agent(arm="random")
        optimizer_param_ids = {
            id(parameter)
            for group in agent.critic_optimizer.param_groups
            for parameter in group["params"]
        }
        assert id(agent.random_target) not in optimizer_param_ids

    def test_registered_seed_is_reproducible(self) -> None:
        assert torch.equal(random_target(RANDOM_TARGET_SEED), random_target(20260718))

    def test_is_independent_of_run_seed(self) -> None:
        torch.manual_seed(1)
        first = SprintTD3Agent(arm="random").random_target.clone()
        torch.manual_seed(999)
        second = SprintTD3Agent(arm="random").random_target
        assert torch.equal(first, second)

    def test_head_is_v1_isomorphic(self) -> None:
        v1 = SprintTD3Agent(arm="v1")
        random = SprintTD3Agent(arm="random")
        assert [parameter.shape for parameter in random.f_resp.parameters()] == [
            parameter.shape for parameter in v1.f_resp.parameters()
        ]
        assert sum(parameter.numel() for parameter in random.f_resp.parameters()) == sum(
            parameter.numel() for parameter in v1.f_resp.parameters()
        )

    def test_checkpoint_records_seed_and_regenerates_target(self, tmp_path) -> None:
        source = SprintTD3Agent(arm="random", target_seed=RANDOM_TARGET_SEED)
        path = source.save_checkpoint(tmp_path / "random.pt")
        payload = torch.load(path, weights_only=False)
        assert payload["sprint_arm"]["arm"] == "random"
        assert payload["sprint_arm"]["target_seed"] == RANDOM_TARGET_SEED
        assert "target" not in payload["sprint_arm"]
        restored = SprintTD3Agent(arm="random", target_seed=1)
        restored.load_checkpoint(path)
        assert restored.target_seed == RANDOM_TARGET_SEED
        assert torch.equal(restored.random_target, source.random_target)