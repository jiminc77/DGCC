"""CPU-only equivalence contracts against the authenticated 289c543 source tree."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CLOSURE_SCRIPT = ROOT / "scripts/restore_289c543_closure.py"

RUNNER = r'''
import numpy as np
import sys
import torch
from dgcc.rl.td3 import TD3Agent, TD3Config

out = sys.argv[1]
base_commit = sys.argv[2]
rng = np.random.default_rng(941)
torch.manual_seed(7331)
config = TD3Config(batch_size=4, replay_capacity=16, policy_noise=0.037)
config.policy_delay = 1
agent = TD3Agent(config, device="cpu")
batches = []
for _ in range(9):
    batches.append({
        "X_before": rng.normal(size=(4, 32, 3)).astype(np.float32),
        "X_after": rng.normal(size=(4, 32, 3)).astype(np.float32),
        "goal_curve": rng.normal(size=(4, 32, 3)).astype(np.float32),
        "flip_before": rng.integers(0, 2, size=4, dtype=np.int8).astype(bool),
        "flip_after": rng.integers(0, 2, size=4, dtype=np.int8).astype(bool),
        "p": rng.integers(0, 32, size=4, dtype=np.int64),
        "delta": rng.uniform(-0.1, 0.1, size=(4, 3)).astype(np.float32),
        "lift": rng.integers(0, 2, size=4).astype(np.float32),
        "reward": rng.normal(size=4).astype(np.float32),
        "done": rng.integers(0, 2, size=4).astype(bool),
        "truncated": rng.integers(0, 2, size=4).astype(bool),
    })
trace = []
for batch in batches:
    target = agent.compute_target(batch)
    stats = agent.update(batch)
    actor_updated = bool(stats.pop("actor_updated", True))
    trace.append({
        "target": target,
        "stats": stats,
        "actor_updated": actor_updated,
        "rng": torch.get_rng_state().clone(),
        "counters": {
            "update": agent.update_count,
            "actor": getattr(agent, "actor_update_count", agent.update_count),
            "target": getattr(agent, "target_update_count", agent.update_count),
        },
    })
torch.save({
    "base_commit": base_commit,
    "trace": trace,
    "modules": {name: module.state_dict() for name, module in {
        "encoder": agent.encoder, "critic": agent.critic, "actor": agent.actor,
        "encoder_target": agent.encoder_target, "critic_target": agent.critic_target,
        "actor_target": agent.actor_target,
    }.items()},
    "optimizers": {
        "critic": agent.critic_optimizer.state_dict(),
        "actor": agent.actor_optimizer.state_dict(),
    },
    "counts": {
        "update": agent.update_count,
        "actor": getattr(agent, "actor_update_count", agent.update_count),
        "target": getattr(agent, "target_update_count", agent.update_count),
    },
}, out)
'''


def _run_snapshot(pythonpath: Path, output: Path, cwd: Path, base_commit: str) -> dict:
    env = {"PYTHONPATH": str(pythonpath), "PYTHONDONTWRITEBYTECODE": "1", "PATH": os.environ["PATH"]}
    result = subprocess.run(
        [sys.executable, "-c", RUNNER, str(output), base_commit],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return torch.load(output, map_location="cpu", weights_only=False)


def _assert_exact(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            _assert_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left)) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_exact(a, b)
    elif isinstance(left, float):
        assert left == right
    else:
        assert left == right


def test_289c543_cpu_trajectory_matches_current_d1_in_isolated_namespaces(tmp_path: Path) -> None:
    closure = tmp_path / "closure"
    restore = subprocess.run([sys.executable, str(CLOSURE_SCRIPT), "--destination", str(closure)], cwd=ROOT, capture_output=True, text=True)
    assert restore.returncode == 0, restore.stderr
    metadata = json.loads((closure / "closure_metadata.json").read_text(encoding="utf-8"))
    base_commit = metadata["source_commit"]
    assert isinstance(base_commit, str) and len(base_commit) == 40
    assert set(metadata) == {"schema_version", "source_commit", "files"}
    historical = _run_snapshot(closure / "src", tmp_path / "historical.pt", tmp_path, base_commit)
    current = _run_snapshot(ROOT / "src", tmp_path / "current.pt", tmp_path, base_commit)
    assert historical["base_commit"] == base_commit
    _assert_exact(historical, current)
    actor_updates = [item["actor_updated"] for item in historical["trace"]]
    assert actor_updates == [True] * 9
    assert historical["counts"] == {"update": 9, "actor": 9, "target": 9}


def _batch() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(12)
    return {
        "X_before": rng.normal(size=(2, 32, 3)).astype(np.float32), "X_after": rng.normal(size=(2, 32, 3)).astype(np.float32),
        "goal_curve": rng.normal(size=(2, 32, 3)).astype(np.float32), "p": np.array([1, 7]),
        "delta": rng.uniform(-0.1, 0.1, (2, 3)).astype(np.float32), "lift": np.array([0, 1], dtype=np.float32),
        "reward": np.array([0.1, -0.2], dtype=np.float32), "done": np.array([False, True]), "truncated": np.array([False, False]),
    }


def test_uniform_qmin_actor_gradient_and_gradient_isolation() -> None:
    from dgcc.rl.td3 import TD3Agent, TD3Config

    torch.manual_seed(9)
    agent = TD3Agent(TD3Config(policy_delay=1), device="cpu")
    batch = _batch()
    feats = agent.features(batch["X_before"], batch["goal_curve"])
    with torch.no_grad():
        h = agent.encoder(feats)
        u = agent.actor(h)
        q1 = agent._q_all_candidates(agent.critic.q1, h, u)
        q2 = agent._q_all_candidates(agent.critic.q2, h, u)
    expected = -torch.minimum(q1, q2).mean().item()
    before = copy.deepcopy(agent.actor.state_dict())
    stats = agent.actor_update(batch, feats_before=feats)
    assert stats["actor_loss"] == pytest.approx(expected, rel=0, abs=1e-7)
    assert any(not torch.equal(before[key], value) for key, value in agent.actor.state_dict().items())
    assert all(parameter.grad is None for parameter in agent.encoder.parameters())
    assert all(parameter.grad is None for parameter in agent.critic.parameters())


def test_policy_delay_protocol_isolates_actor_and_targets() -> None:
    from dgcc.rl.td3 import TD3Agent, TD3Config

    torch.manual_seed(19)
    agent = TD3Agent(TD3Config(policy_delay=2, policy_noise=0.031), device="cpu")
    batch = _batch()
    actor_before = copy.deepcopy(agent.actor.state_dict())
    targets_before = {
        name: copy.deepcopy(module.state_dict())
        for name, module in {
            "encoder": agent.encoder_target,
            "critic": agent.critic_target,
            "actor": agent.actor_target,
        }.items()
    }
    agent.update(batch, generator=torch.Generator().manual_seed(2))
    assert agent.actor_update_count == agent.target_update_count == 0
    _assert_exact(actor_before, agent.actor.state_dict())
    for name, state in targets_before.items():
        _assert_exact(state, getattr(agent, f"{name}_target").state_dict())
    agent.update(batch, generator=torch.Generator().manual_seed(2))
    assert agent.actor_update_count == agent.target_update_count == 1
    assert all(
        any(not torch.equal(before[key], value) for key, value in getattr(agent, f"{name}_target").state_dict().items())
        for name, before in targets_before.items()
    )
