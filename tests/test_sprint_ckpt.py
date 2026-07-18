from __future__ import annotations

import torch

from dgcc.rl.sprint_arms import SprintTD3Agent
from dgcc.rl.td3 import TD3Agent, TD3Config


def test_v2_checkpoint_round_trip(tmp_path) -> None:
    source = SprintTD3Agent(TD3Config(), aux_weight=0.25)
    source.update_count = 9
    path = source.save_checkpoint(tmp_path / "v2.pt")
    payload = torch.load(path, weights_only=False)
    assert payload["sprint_arm"]["schema_version"] == 2
    assert payload["sprint_arm"]["arm"] == "v1"
    restored = SprintTD3Agent(TD3Config())
    restored.load_checkpoint(path)
    assert restored.update_count == 9
    assert restored.aux_weight == .25
    for a, b in zip(source.f_resp.parameters(), restored.f_resp.parameters(), strict=True): assert torch.equal(a, b)


def test_legacy_baseline_checkpoint_loads(tmp_path) -> None:
    legacy = TD3Agent(TD3Config())
    legacy.update_count = 3
    path = legacy.save_checkpoint(tmp_path / "legacy.pt")
    adapter = SprintTD3Agent(TD3Config())
    before = [p.clone() for p in adapter.f_resp.parameters()]
    adapter.load_checkpoint(path)
    assert adapter.update_count == 3
    for left, right in zip(legacy.encoder.parameters(), adapter.encoder.parameters(), strict=True): assert torch.equal(left, right)
    for left, right in zip(before, adapter.f_resp.parameters(), strict=True): assert torch.equal(left, right)
