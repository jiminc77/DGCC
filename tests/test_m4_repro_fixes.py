"""M4-prep reproducibility fixes F-a / F-b (gate-m3r-reconvene-2-20260713, choice B).

F-a: all RNGs seeded before TD3Agent construction; same seed -> identical
     initial-weights hash across fresh constructions.
F-b: deterministic-eval episode indexing is a pure function of the one-based
     successful-eval ordinal — independent of rebuild history.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

_SPEC = importlib.util.spec_from_file_location("p1_train", _REPO / "scripts" / "p1_train.py")
p1_train = importlib.util.module_from_spec(_SPEC)
sys.modules["p1_train"] = p1_train
_SPEC.loader.exec_module(p1_train)

import torch  # noqa: E402

from dgcc.rl.td3 import TD3Agent, TD3Config  # noqa: E402
from dgcc.tasks.domain import RewardConstants  # noqa: E402


def _fresh_agent(seed: int) -> TD3Agent:
    # Mirrors the driver's F-a ordering: seed strictly before construction.
    torch.manual_seed(seed)
    return TD3Agent(
        TD3Config(),
        device="cpu",
        reward_constants=RewardConstants(alpha=10.0, c_step=0.1, r_succ=5.0),
    )


def test_same_seed_identical_initial_hash() -> None:
    h1 = p1_train.initial_weights_sha256(_fresh_agent(7))
    h2 = p1_train.initial_weights_sha256(_fresh_agent(7))
    assert h1 == h2, "F-a broken: same seed must yield identical initial weights"


def test_different_seed_different_initial_hash() -> None:
    h1 = p1_train.initial_weights_sha256(_fresh_agent(7))
    h2 = p1_train.initial_weights_sha256(_fresh_agent(8))
    assert h1 != h2, "hash must be sensitive to the seed"


def test_hash_covers_target_networks() -> None:
    agent = _fresh_agent(7)
    before = p1_train.initial_weights_sha256(agent)
    with torch.no_grad():
        first_param = next(agent.encoder_target.parameters())
        first_param.add_(1.0)
    after = p1_train.initial_weights_sha256(agent)
    assert before != after, "hash must cover target networks too"


def test_driver_seeds_before_agent_construction() -> None:
    src = inspect.getsource(p1_train.TrainingRun.__init__)
    assert src.index("torch.manual_seed") < src.index("TD3Agent("), (
        "F-a regression: torch.manual_seed must precede TD3Agent construction"
    )
    assert "initial_weights_sha256(self.agent)" in src


def test_eval_index_first_is_90001() -> None:
    assert p1_train.eval_episode_index_start(0) == 90_001


def test_eval_index_is_ordinal_only() -> None:
    # Two attempts with wildly different rebuild histories but the same number
    # of completed evals must evaluate the identical episode set.
    attempt_a_rebuilds = 0  # noqa: F841 — rebuild history is deliberately unused
    attempt_b_rebuilds = 6  # noqa: F841
    for completed in range(4):
        assert p1_train.eval_episode_index_start(completed) == 90_001 + completed


def test_driver_eval_path_is_rebuild_decoupled() -> None:
    src = inspect.getsource(p1_train.TrainingRun.deterministic_eval)
    assert "90_000 + self.episode_index" not in src, (
        "F-b regression: eval indexing must not depend on rebuild-coupled episode_index"
    )
    gate_src = inspect.getsource(p1_train.TrainingRun.eval_and_checkpoint)
    assert "eval_episode_index_start(self._eval_ordinal)" in gate_src
    # Ordinal captured before the retry loop, incremented after success only.
    assert gate_src.index("eval_episode_index_start(self._eval_ordinal)") < gate_src.index(
        "while True"
    )
    assert gate_src.index("self._eval_ordinal += 1") > gate_src.index("while True")
