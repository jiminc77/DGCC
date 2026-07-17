"""P1-M5 latent API contract tests (P1.md M5 검증 항목).

Covers: same-input reproducibility, frozen guarantee (parameter immutability),
shape contract, and Q recomputation agreement with the training-side
``q_min_executed`` code path.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dgcc.analysis.latent_api import LATENT_SPEC, FrozenLatentExtractor, lift_to_float
from dgcc.rl.td3 import TD3Agent, TD3Config


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> str:
    torch.manual_seed(1234)
    agent = TD3Agent(TD3Config(), device="cpu")
    path = tmp_path_factory.mktemp("m5") / "ckpt_test.pt"
    agent.save_checkpoint(path)
    return str(path)


@pytest.fixture(scope="module")
def batch() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    B = 6
    X = rng.normal(0.0, 0.1, size=(B, 32, 3))
    G = rng.normal(0.0, 0.1, size=(B, 32, 3))
    return {
        "X": X,
        "G": G,
        "p": rng.integers(0, 32, size=B),
        "delta": rng.uniform(-0.15, 0.15, size=(B, 3)),
        "lift": np.asarray(["high", "low", "high", "low", "low", "high"]),
    }


def test_shape_contract(checkpoint, batch):
    ex = FrozenLatentExtractor.from_checkpoint(checkpoint)
    out = ex.extract(batch["X"], batch["G"], batch["p"], batch["delta"], batch["lift"])
    assert set(out) == set(LATENT_SPEC)
    B = batch["X"].shape[0]
    for name, shape in LATENT_SPEC.items():
        assert out[name].shape == (B, *shape[1:]), name


def test_same_input_reproducibility(checkpoint, batch):
    ex = FrozenLatentExtractor.from_checkpoint(checkpoint)
    a = ex.extract(batch["X"], batch["G"], batch["p"], batch["delta"], batch["lift"])
    b = ex.extract(batch["X"], batch["G"], batch["p"], batch["delta"], batch["lift"])
    for name in LATENT_SPEC:
        np.testing.assert_array_equal(a[name], b[name])


def test_frozen_guarantee(checkpoint, batch):
    ex = FrozenLatentExtractor.from_checkpoint(checkpoint)
    before = ex.parameter_sha256()
    for module in (ex.agent.encoder, ex.agent.critic, ex.agent.actor):
        assert not module.training
        assert all(not p.requires_grad for p in module.parameters())
    ex.extract(batch["X"], batch["G"], batch["p"], batch["delta"], batch["lift"])
    assert ex.parameter_sha256() == before


def test_q_recomputation_matches_training_path(checkpoint, batch):
    ex = FrozenLatentExtractor.from_checkpoint(checkpoint)
    out = ex.extract(batch["X"], batch["G"], batch["p"], batch["delta"], batch["lift"])
    lift_num = lift_to_float(batch["lift"])
    q_min_train = ex.agent.q_min_executed(
        batch["X"], batch["G"], batch["p"], batch["delta"], lift_num
    )
    np.testing.assert_allclose(out["q_min"], q_min_train, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(out["q_min"], np.minimum(out["q1"], out["q2"]))


def test_q_recomputable_from_extracted_latents(checkpoint, batch):
    """Feeding the extracted h_p back through the critic head reproduces Q."""

    ex = FrozenLatentExtractor.from_checkpoint(checkpoint)
    out = ex.extract(batch["X"], batch["G"], batch["p"], batch["delta"], batch["lift"])
    h_p = torch.as_tensor(out["h_p"], dtype=torch.float32)
    u = torch.cat(
        [
            torch.as_tensor(batch["delta"], dtype=torch.float32),
            torch.as_tensor(lift_to_float(batch["lift"]), dtype=torch.float32).reshape(-1, 1),
        ],
        dim=-1,
    )
    with torch.no_grad():
        q1 = ex.agent.critic.q1(h_p, u)
    np.testing.assert_allclose(q1.numpy(), out["q1"], rtol=1e-5, atol=1e-5)


def test_checkpoint_hash_recorded(checkpoint):
    ex = FrozenLatentExtractor.from_checkpoint(checkpoint)
    meta = ex.metadata()
    assert meta["ckpt_sha256"] == ex.ckpt_sha256 and len(ex.ckpt_sha256) == 64
    assert meta["latent_spec"] == {k: list(v) for k, v in LATENT_SPEC.items()}
