from __future__ import annotations

import torch
import pytest

from dgcc.rl.sprint_arms import SprintTD3Agent
from dgcc.rl.td3 import TD3Agent, TD3Config


def test_v2_checkpoint_eval_round_trip_rejects_inexact_resume(tmp_path) -> None:
    source = SprintTD3Agent(TD3Config(), aux_weight=0.25)
    source.update_count = 9
    path = source.save_checkpoint(tmp_path / "v2.pt")
    payload = torch.load(path, weights_only=False)
    assert payload["sprint_arm"]["schema_version"] == 2
    assert payload["sprint_arm"]["arm"] == "v1"
    restored = SprintTD3Agent(TD3Config())
    with pytest.raises(ValueError, match="all checkpoints are evaluation-only"):
        restored.load_checkpoint(path)
    restored.load_checkpoint(path, eval_only=True)
    assert restored.fresh_restart_only
    with pytest.raises(RuntimeError, match="fresh_restart_only"):
        restored.update({})
    assert restored.update_count == 0
    assert restored.aux_weight == .25
    for a, b in zip(source.f_resp.parameters(), restored.f_resp.parameters(), strict=True):
        assert torch.equal(a, b)


def test_legacy_baseline_checkpoint_is_rejected_for_sprint_evaluation(tmp_path) -> None:
    legacy = TD3Agent(TD3Config())
    legacy.update_count = 3
    path = legacy.save_checkpoint(tmp_path / "legacy.pt")
    adapter = SprintTD3Agent(TD3Config())
    before = [p.clone() for p in adapter.f_resp.parameters()]
    with pytest.raises(ValueError, match="all checkpoints are evaluation-only"):
        adapter.load_checkpoint(path)
    with pytest.raises(ValueError, match="incompatible sprint evaluation checkpoint"):
        adapter.load_checkpoint(path, eval_only=True)
    assert adapter.fresh_restart_only is False
    for left, right in zip(before, adapter.f_resp.parameters(), strict=True):
        assert torch.equal(left, right)


def test_v2_checkpoint_config_drift_is_resume_error_but_eval_allowed(tmp_path) -> None:
    path = SprintTD3Agent(TD3Config()).save_checkpoint(tmp_path / "v2.pt")
    restored = SprintTD3Agent(TD3Config(lr=1e-3))
    with pytest.raises(ValueError, match="all checkpoints are evaluation-only"):
        restored.load_checkpoint(path)
    restored.load_checkpoint(path, eval_only=True)
    assert restored.fresh_restart_only
    with pytest.raises(RuntimeError, match="fresh_restart_only"):
        restored.update({})
