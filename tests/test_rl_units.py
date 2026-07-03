"""P1-M1 RL unit tests (CPU only).

Covers the M1 exit contract: hand-computed target values (target vs online
critics distinguished), double-Q decoupling (selection critic ≠ evaluation
pair), actor gradient isolation (u only, никогда via p or trunk), replay v2
round-trip + v1 migration, checkpoint save/load equality — plus the §6
encoder input contract (goal sensitivity + canonical_shape_flip consistency)
and the training-level NaN halt.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dgcc.goals.distance import canonical_shape_flip
from dgcc.logging.writer import TransitionWriter
from dgcc.models.networks import (
    Actor,
    Encoder,
    TwinCritic,
    build_node_features,
    goal_residual_flips,
    parameter_count,
)
from dgcc.rl.replay import (
    PROVENANCE_P0_REUSE,
    ReplayBuffer,
    ReplaySchemaError,
    ingest_v1_transitions,
    read_v2_transitions,
    write_v2_transitions,
)
from dgcc.rl.td3 import (
    TD3Agent,
    TD3Config,
    TrainingNaNError,
    epsilon_schedule,
    select_p_star,
    smooth_target_u,
    td_target,
)

K = 32


def smooth_curve(seed: int, arc: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, K)
    y = 0.1 * rng.standard_normal() * np.sin(np.pi * t + rng.uniform(0, 2 * np.pi))
    curve = np.column_stack((arc * (t - 0.5), y, 0.02 * np.sin(2 * np.pi * t)))
    return curve


def make_batch(size: int = 6, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "X_before": np.stack([smooth_curve(seed + i) for i in range(size)]),
        "X_after": np.stack([smooth_curve(seed + 100 + i) for i in range(size)]),
        "goal_curve": np.stack([smooth_curve(seed + 200 + i) for i in range(size)]),
        "p": rng.integers(0, K, size=size),
        "delta": rng.uniform(-0.1, 0.1, size=(size, 3)),
        "lift": rng.integers(0, 2, size=size),
        "reward": rng.normal(0.0, 1.0, size=size),
        "done": rng.random(size) < 0.2,
    }


# ---------------------------------------------------------------------------
# Hand-computed target pieces
# ---------------------------------------------------------------------------


def test_select_p_star_hand_table() -> None:
    q1 = torch.tensor([[1.0, 3.0, 2.0], [5.0, 4.0, 0.0]])
    assert select_p_star(q1).tolist() == [1, 0]


def test_td_target_hand_computed() -> None:
    y = td_target(
        reward=torch.tensor([1.0, 2.0]),
        done=torch.tensor([False, True]),
        gamma=0.95,
        q_min=torch.tensor([10.0, 10.0]),
    )
    assert y.tolist() == pytest.approx([1.0 + 0.95 * 10.0, 2.0])


def test_smooth_target_u_clips_noise_and_box() -> None:
    u = torch.tensor([[0.14, -0.14, 0.0, 0.95]])
    noise = torch.tensor([[0.5, -0.5, 0.05, 0.5]])  # clipped to ±0.1
    smoothed = smooth_target_u(u, noise, noise_clip=0.1)
    assert smoothed[0, 0].item() == pytest.approx(0.15)  # 0.14+0.1 → clamp 0.15
    assert smoothed[0, 1].item() == pytest.approx(-0.15)
    assert smoothed[0, 2].item() == pytest.approx(0.05)
    assert smoothed[0, 3].item() == pytest.approx(1.0)  # lift clamped to [0,1]


def test_decoupling_selection_critic_differs_from_evaluation() -> None:
    # Selection uses Q_target_1 ONLY: p* = argmax q1 even though q2's argmax
    # differs; evaluation is min(q1, q2) AT p*, not at q2's favourite.
    q1 = torch.tensor([[0.0, 10.0]])
    q2 = torch.tensor([[100.0, -5.0]])
    p_star = select_p_star(q1)
    assert p_star.tolist() == [1]
    q_min_at_star = torch.minimum(q1[0, p_star], q2[0, p_star])
    y = td_target(torch.tensor([1.0]), torch.tensor([False]), 0.95, q_min_at_star)
    assert y.item() == pytest.approx(1.0 + 0.95 * -5.0)


def test_epsilon_schedule_linear_first_30_percent() -> None:
    config = TD3Config()
    assert epsilon_schedule(0, 100_000, config) == pytest.approx(1.0)
    assert epsilon_schedule(15_000, 100_000, config) == pytest.approx(0.55)
    assert epsilon_schedule(30_000, 100_000, config) == pytest.approx(0.1)
    assert epsilon_schedule(90_000, 100_000, config) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Target computation uses TARGET networks only (C4/F4)
# ---------------------------------------------------------------------------


def _perturb(module: torch.nn.Module, scale: float = 0.5, seed: int = 0) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in module.parameters():
            param.add_(torch.randn(param.shape, generator=gen, device=param.device) * scale)


def test_compute_target_ignores_online_and_uses_target_nets() -> None:
    agent = TD3Agent(TD3Config(policy_noise=0.0))  # noise-free for exactness
    batch = make_batch()
    y0 = agent.compute_target(batch)

    # Perturbing ONLINE nets must not change the target.
    _perturb(agent.encoder, seed=1)
    _perturb(agent.critic, seed=2)
    _perturb(agent.actor, seed=3)
    y1 = agent.compute_target(batch)
    assert torch.allclose(y0, y1), "target computation leaked online networks"

    # Perturbing TARGET critic 1 must change the target (selection + eval).
    _perturb(agent.critic_target.q1, seed=4)
    y2 = agent.compute_target(batch)
    assert not torch.allclose(y0, y2), "target computation ignored Q_target_1"


def test_compute_target_matches_manual_formula() -> None:
    agent = TD3Agent(TD3Config(policy_noise=0.0))
    batch = make_batch(seed=7)
    y = agent.compute_target(batch)

    with torch.no_grad():
        feats = agent.features(batch["X_after"], batch["goal_curve"])
        h = agent.encoder_target(feats)
        u_all = agent.actor_target(h)
        b, k = h.shape[0], h.shape[1]
        q1 = agent.critic_target.q1(h.reshape(b * k, -1), u_all.reshape(b * k, -1)).reshape(b, k)
        p_star = q1.argmax(dim=1)
        idx = torch.arange(b, device=h.device)
        q1_s, q2_s = agent.critic_target(h[idx, p_star], u_all[idx, p_star])
        manual = torch.as_tensor(batch["reward"], dtype=torch.float32, device=h.device) + 0.95 * (
            1.0 - torch.as_tensor(batch["done"], dtype=torch.float32, device=h.device)
        ) * torch.minimum(q1_s, q2_s)
    assert torch.allclose(y, manual, atol=1e-6)


# ---------------------------------------------------------------------------
# Actor gradient isolation
# ---------------------------------------------------------------------------


def test_actor_gradient_flows_only_through_u() -> None:
    agent = TD3Agent()
    batch = make_batch(seed=11)

    encoder_before = [p.detach().clone() for p in agent.encoder.parameters()]
    critic_before = [p.detach().clone() for p in agent.critic.parameters()]
    stats = agent.actor_update(batch)
    assert np.isfinite(stats["actor_loss"])

    # Encoder and critic parameters are untouched by the actor update.
    for before, after in zip(encoder_before, agent.encoder.parameters(), strict=True):
        assert torch.equal(before, after)
    for before, after in zip(critic_before, agent.critic.parameters(), strict=True):
        assert torch.equal(before, after)
    # Actor received gradients (u path) — its params moved.
    grads = [p.grad for p in agent.actor.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_critic_update_trains_encoder_and_critic() -> None:
    agent = TD3Agent()
    batch = make_batch(seed=13)
    encoder_before = [p.detach().clone() for p in agent.encoder.parameters()]
    critic_before = [p.detach().clone() for p in agent.critic.parameters()]
    stats = agent.critic_update(batch)
    assert np.isfinite(stats["critic_loss"])
    moved_encoder = any(
        not torch.equal(b, a) for b, a in zip(encoder_before, agent.encoder.parameters(), strict=True)
    )
    moved_critic = any(
        not torch.equal(b, a) for b, a in zip(critic_before, agent.critic.parameters(), strict=True)
    )
    assert moved_encoder and moved_critic


# ---------------------------------------------------------------------------
# §6 encoder input contract
# ---------------------------------------------------------------------------


def test_encoder_features_goal_sensitivity() -> None:
    x = smooth_curve(3)
    g1 = smooth_curve(4)
    g2 = smooth_curve(5)
    f1, _ = build_node_features(x[None], g1[None])
    f2, _ = build_node_features(x[None], g2[None])
    assert f1.shape == (1, 32, 7)
    np.testing.assert_array_equal(f1[..., :4], f2[..., :4])  # x, sigma unchanged
    assert np.abs(f1[..., 4:] - f2[..., 4:]).max() > 1e-6  # residual responds to goal


def test_encoder_flip_consistency_matches_canonical_shape_flip() -> None:
    # Asymmetric curve; goal is its reversal — the canonical decision must
    # come from goals.distance.canonical_shape_flip and the residual channel
    # must use the flip-aligned goal ordering.
    t = np.linspace(0.0, 1.0, K)
    x = np.column_stack((t - 0.5, 0.25 * t**2, np.zeros(K)))  # J-like, asymmetric
    g = x[::-1].copy()

    expected_flip = canonical_shape_flip(x, {"shape_template": g, "anchor": g.mean(axis=0)}, 1.0)
    flips = goal_residual_flips(x[None], g[None])
    assert bool(flips[0]) == bool(expected_flip)

    feats, flips2 = build_node_features(x[None], g[None])
    g_aligned = g[::-1] if flips2[0] else g
    np.testing.assert_allclose(feats[0, :, 4:], g_aligned - x)


def test_parameter_count_recorded_range() -> None:
    total = parameter_count(Encoder(), TwinCritic(), Actor())
    assert 300_000 < total < 3_000_000  # §6 guideline ~1-2M; actual recorded in log


# ---------------------------------------------------------------------------
# Replay v2 round-trip + v1 migration
# ---------------------------------------------------------------------------


def _v2_columns(count: int = 4) -> dict[str, list | np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "X_before": rng.normal(size=(count, 32, 3)),
        "X_after": rng.normal(size=(count, 32, 3)),
        "p": rng.integers(0, 32, size=count),
        "delta": rng.normal(size=(count, 3)),
        "lift": ["low", "high", "low", "high"][:count],
        "grasp_success": rng.random(count) < 0.9,
        "settle_steps": rng.integers(0, 10000, size=count),
        "rope_params": ["{}"] * count,
        "seed": np.arange(count),
        "sim": ["dlolab"] * count,
        "timestamp": ["2026-07-03T00:00:00Z"] * count,
        "commit_hash": ["deadbeef"] * count,
        "task_id": ["t2"] * count,
        "goal_id": [f"t2-{i:04d}" for i in range(count)],
        "goal_spec_hash": ["0" * 16] * count,
        "goal_curve": rng.normal(size=(count, 32, 3)),
        "episode_id": np.arange(count),
        "step_index": np.arange(count),
        "reward": rng.normal(size=count),
        "done": rng.random(count) < 0.5,
        "provenance": ["p1_fresh"] * count,
    }


def test_replay_v2_roundtrip(tmp_path: Path) -> None:
    columns = _v2_columns()
    path = tmp_path / "v2.h5"
    write_v2_transitions(path, columns, {"config": {}, "commit_hash": "deadbeef"})
    loaded, meta = read_v2_transitions(path)
    assert meta["commit_hash"] == "deadbeef"
    np.testing.assert_allclose(loaded["X_before"], columns["X_before"])
    np.testing.assert_allclose(loaded["goal_curve"], columns["goal_curve"])
    np.testing.assert_array_equal(loaded["done"], columns["done"])
    assert loaded["lift"] == columns["lift"]
    assert loaded["goal_id"] == columns["goal_id"]


def test_replay_v2_rejects_missing_field(tmp_path: Path) -> None:
    columns = _v2_columns()
    columns.pop("goal_curve")
    with pytest.raises(ReplaySchemaError):
        write_v2_transitions(tmp_path / "bad.h5", columns, {})


def test_replay_v1_migration_filters_and_flags(tmp_path: Path) -> None:
    path = tmp_path / "v1.h5"
    base = {
        "X_before": np.zeros((32, 3)),
        "X_after": np.ones((32, 3)),
        "delta": np.zeros(3),
        "lift": "low",
        "rope_params": {"length_m": 1.0},
        "seed": 0,
        "sim": "dlolab",
        "timestamp": "2026-07-03T00:00:00Z",
        "commit_hash": "cafe",
    }
    records = [
        dict(base, p=1, grasp_success=True, settle_steps=1200),  # kept
        dict(base, p=2, grasp_success=True, settle_steps=5000),  # non-converged
        dict(base, p=3, grasp_success=False, settle_steps=0),  # failed grasp
    ]
    meta = {"config": "collection:\n  settle_max_steps: 5000\n", "commit_hash": "cafe"}
    with TransitionWriter(path, meta=meta) as writer:
        writer.append(records)

    result = ingest_v1_transitions(path)
    assert result["provenance"] == PROVENANCE_P0_REUSE
    assert result["total_records"] == 3
    assert result["kept_records"] == 1
    assert result["p"].tolist() == [1]
    assert result["settle_max_steps"] == 5000


def test_replay_buffer_wraps_and_samples() -> None:
    buffer = ReplayBuffer(capacity=5)
    rng = np.random.default_rng(0)
    for start in (0, 3):
        count = 3
        buffer.add_batch(
            X_before=np.full((count, 32, 3), start, dtype=float),
            X_after=np.zeros((count, 32, 3)),
            goal_curve=np.zeros((count, 32, 3)),
            p=np.arange(start, start + count),
            delta=np.zeros((count, 3)),
            lift=np.zeros(count, dtype=int),
            reward=np.arange(count, dtype=float),
            done=np.zeros(count, dtype=bool),
        )
    assert buffer.size == 5
    batch = buffer.sample(8, rng)
    assert batch["X_before"].shape == (8, 32, 3)
    assert batch["p"].shape == (8,)


# ---------------------------------------------------------------------------
# Checkpoint save/load equality
# ---------------------------------------------------------------------------


def test_checkpoint_save_load_equality(tmp_path: Path) -> None:
    agent1 = TD3Agent()
    agent1.update(make_batch(seed=17))  # move away from init
    path = agent1.save_checkpoint(tmp_path / "ckpt.pt")

    agent2 = TD3Agent()
    agent2.load_checkpoint(path)
    assert agent2.update_count == agent1.update_count
    for module1, module2 in (
        (agent1.encoder, agent2.encoder),
        (agent1.critic, agent2.critic),
        (agent1.actor, agent2.actor),
        (agent1.encoder_target, agent2.encoder_target),
        (agent1.critic_target, agent2.critic_target),
        (agent1.actor_target, agent2.actor_target),
    ):
        for key, value in module1.state_dict().items():
            assert torch.equal(value, module2.state_dict()[key]), key
    # identical behaviour on a fresh batch
    batch = make_batch(seed=19)
    assert torch.allclose(
        agent1.compute_target(batch, generator=torch.Generator().manual_seed(0)),
        agent2.compute_target(batch, generator=torch.Generator().manual_seed(0)),
    )


# ---------------------------------------------------------------------------
# Training-level NaN halt (global rule 6)
# ---------------------------------------------------------------------------


def test_nan_halt_raises_before_optimizer_step() -> None:
    agent = TD3Agent()
    batch = make_batch(seed=23)
    batch["reward"] = np.full_like(batch["reward"], np.nan)

    params_before = [p.detach().clone() for p in agent.critic.parameters()]
    with pytest.raises(TrainingNaNError):
        agent.critic_update(batch)
    for before, after in zip(params_before, agent.critic.parameters(), strict=True):
        assert torch.equal(before, after), "NaN halt must not apply a parameter step"
