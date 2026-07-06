"""P1-M3R backbone tests for F1/S1/S2/S3 changes."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from dgcc.models.networks import EMBED_DIM, U_DIM, TwinCritic, parameter_count
from dgcc.rl.replay import ReplayBuffer
from dgcc.rl.td3 import TD3Agent, TD3Config, derive_td_target_bound, td_target
from dgcc.tasks.domain import RewardConstants

K = 32


def _agent_batch(size: int) -> dict[str, np.ndarray]:
    return {
        "X_before": np.zeros((size, K, 3), dtype=float),
        "X_after": np.zeros((size, K, 3), dtype=float),
        "goal_curve": np.zeros((size, K, 3), dtype=float),
        "p": np.zeros(size, dtype=np.int64),
        "delta": np.zeros((size, 3), dtype=float),
        "lift": np.zeros(size, dtype=np.int64),
        "reward": np.zeros(size, dtype=float),
        "done": np.zeros(size, dtype=bool),
        "truncated": np.zeros(size, dtype=bool),
        "flip_before": np.zeros(size, dtype=bool),
        "flip_after": np.zeros(size, dtype=bool),
    }


def _add_replay_batch(
    buffer: ReplayBuffer,
    *,
    start: int,
    count: int,
    truncated: np.ndarray,
) -> None:
    buffer.add_batch(
        X_before=np.full((count, K, 3), start, dtype=float),
        X_after=np.zeros((count, K, 3), dtype=float),
        goal_curve=np.zeros((count, K, 3), dtype=float),
        p=np.arange(start, start + count),
        delta=np.zeros((count, 3), dtype=float),
        lift=np.zeros(count, dtype=np.int64),
        reward=np.arange(count, dtype=float),
        done=np.zeros(count, dtype=bool),
        truncated=truncated,
        flip_before=np.zeros(count, dtype=bool),
        flip_after=np.zeros(count, dtype=bool),
    )


def test_td_target_f1_bootstraps_only_truncation() -> None:
    y = td_target(
        reward=torch.tensor([1.0, 1.0, 1.0]),
        done=torch.tensor([True, True, False]),
        gamma=0.95,
        q_min=torch.tensor([2.0, 2.0, 2.0]),
        truncated=torch.tensor([False, True, False]),
    )
    assert y.tolist() == pytest.approx([1.0, 2.9, 2.9])


def test_replay_buffer_truncated_required_and_wrap_preserved() -> None:
    buffer = ReplayBuffer(capacity=5)
    _add_replay_batch(
        buffer,
        start=0,
        count=3,
        truncated=np.array([False, True, False]),
    )
    _add_replay_batch(
        buffer,
        start=10,
        count=4,
        truncated=np.array([True, False, True, False]),
    )

    expected = {12: True, 13: False, 2: False, 10: True, 11: False}
    stored = {
        int(p): bool(t)
        for p, t in zip(buffer.p[: buffer.size], buffer.truncated[: buffer.size])
    }
    assert stored == expected

    sample = buffer.sample(64, np.random.default_rng(0))
    assert "truncated" in sample
    for p, truncated in zip(sample["p"], sample["truncated"], strict=True):
        assert bool(truncated) is expected[int(p)]

    with pytest.raises(TypeError):
        buffer.add_batch(
            X_before=np.zeros((1, K, 3)),
            X_after=np.zeros((1, K, 3)),
            goal_curve=np.zeros((1, K, 3)),
            p=np.zeros(1, dtype=np.int64),
            delta=np.zeros((1, 3)),
            lift=np.zeros(1, dtype=np.int64),
            reward=np.zeros(1),
            done=np.zeros(1, dtype=bool),
            flip_before=np.zeros(1, dtype=bool),
            flip_after=np.zeros(1, dtype=bool),
        )


def test_huber_fixtures_and_critic_loss_match() -> None:
    assert F.huber_loss(
        torch.tensor([0.0]), torch.tensor([0.5]), delta=1.0
    ).item() == pytest.approx(0.125)
    assert F.huber_loss(
        torch.tensor([0.0]), torch.tensor([3.0]), delta=1.0
    ).item() == pytest.approx(2.5)
    assert TD3Config().to_dict()["huber_delta"] == 1.0

    agent = TD3Agent(TD3Config(policy_noise=0.0, huber_delta=1.0))
    with torch.no_grad():
        for module in (agent.encoder, agent.critic):
            for param in module.parameters():
                param.zero_()

    def fixed_target(
        batch: dict[str, np.ndarray], *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        del batch, generator
        return torch.tensor([3.0], dtype=torch.float32, device=agent.device)

    agent.compute_target = fixed_target  # type: ignore[method-assign]
    stats = agent.critic_update(_agent_batch(1))
    assert stats["critic_loss"] == pytest.approx(5.0)
    assert "td_target_clamp_hit_frac" in stats


def test_derive_bound_clamp_metadata_and_hit_fraction(tmp_path) -> None:
    bound = derive_td_target_bound(RewardConstants(), gamma=0.95)
    assert bound == {
        "d_max": 7.0,
        "r_min": -70.1,
        "r_max": 74.9,
        "v_max": 1498.0,
    }

    y = td_target(
        reward=torch.tensor([0.0, 1.0]),
        done=torch.tensor([False, False]),
        gamma=0.95,
        q_min=torch.tensor([1.0e9, 2.0]),
        truncated=torch.tensor([False, False]),
        v_max=bound["v_max"],
    )
    assert y[0].item() == 1498.0
    assert y[1].item() == pytest.approx(2.9)

    agent = TD3Agent(TD3Config(policy_noise=0.0))
    assert agent.to_dict()["td_target_bound"] == bound
    payload = torch.load(agent.save_checkpoint(tmp_path / "agent.pt"), weights_only=False)
    assert payload["td_target_bound"] == bound
    assert payload["metadata"]["td_target_bound"] == bound

    def fixed_candidates(
        critic_head: nn.Module, h: torch.Tensor, u_all: torch.Tensor
    ) -> torch.Tensor:
        del critic_head, u_all
        return torch.zeros((h.shape[0], h.shape[1]), dtype=torch.float32, device=h.device)

    def fixed_critic_forward(
        h: torch.Tensor, u: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del u
        q = torch.tensor([1.0e9, 1.0], dtype=torch.float32, device=h.device)
        return q, q.clone()

    agent._q_all_candidates = fixed_candidates  # type: ignore[method-assign]
    agent.critic_target.forward = fixed_critic_forward  # type: ignore[method-assign]
    target = agent.compute_target(_agent_batch(2), generator=torch.Generator().manual_seed(0))
    assert target.tolist() == pytest.approx([1498.0, 0.95])
    assert agent.last_clamp_hit_frac == pytest.approx(0.5)


def test_s2_layernorm_param_delta_and_target_key_parity() -> None:
    critic = TwinCritic()
    for head in (critic.q1, critic.q2):
        assert isinstance(head.ln1, nn.LayerNorm)
        assert isinstance(head.ln2, nn.LayerNorm)
        assert head.ln1.normalized_shape == (256,)
        assert head.ln2.normalized_shape == (256,)

    # Pre-S2 critic covenant: two heads with Linear(260,256), Linear(256,256),
    # Linear(256,1).  Each new LayerNorm(256) contributes weight+bias = 512.
    pre_s2_head = (EMBED_DIM + U_DIM) * 256 + 256 + 256 * 256 + 256 + 256 + 1
    assert parameter_count(critic) - 2 * pre_s2_head == 2_048

    agent = TD3Agent()
    for online, target in (
        (agent.encoder, agent.encoder_target),
        (agent.critic, agent.critic_target),
        (agent.actor, agent.actor_target),
    ):
        assert set(online.state_dict()) == set(target.state_dict())


def test_v1_shaped_critic_state_dict_is_not_strictly_loadable() -> None:
    critic = TwinCritic()
    v1_shaped = {
        key: value for key, value in critic.state_dict().items() if ".ln" not in key
    }
    with pytest.raises(RuntimeError, match="Missing key"):
        critic.load_state_dict(v1_shaped, strict=True)
