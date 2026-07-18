from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from dgcc.phi.dct import Phi_DCT
from dgcc.rl.sprint_arms import ResponseHead, SprintTD3Agent, create_sprint_agent, delta_m_from_batch
from dgcc.rl.td3 import TD3Agent, TD3Config


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
    torch.manual_seed(7); base = TD3Agent(TD3Config(policy_noise=0.0))
    torch.manual_seed(7); sprint = SprintTD3Agent(TD3Config(policy_noise=0.0), aux_weight=0.0)
    assert digest(base) == digest(sprint)
    b = batch()
    g1, g2 = torch.Generator().manual_seed(4), torch.Generator().manual_seed(4)
    baseline = base.critic_update(b, generator=g1)
    adapted = sprint.critic_update(b, generator=g2)
    assert adapted["critic_loss"] == pytest.approx(baseline["critic_loss"], abs=1e-7)
    for left, right in zip(base.encoder.parameters(), sprint.encoder.parameters(), strict=True): assert torch.allclose(left, right)
    for left, right in zip(base.critic.parameters(), sprint.critic.parameters(), strict=True): assert torch.allclose(left, right)


def test_bb_factory_is_baseline_and_future_arms_are_explicit() -> None:
    assert type(create_sprint_agent("bb")) is TD3Agent
    with pytest.raises(NotImplementedError): create_sprint_agent("matched")
