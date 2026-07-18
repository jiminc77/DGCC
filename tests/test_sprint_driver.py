"""CPU-only contracts for the sprint training entry point."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_driver():
    spec = importlib.util.spec_from_file_location("p1_sprint_train_test", ROOT / "scripts/p1_sprint_train.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_t2_config_diff_is_only_sprint_block():
    baseline = yaml.safe_load((ROOT / "configs/p1_t2.yaml").read_text())
    sprint = yaml.safe_load((ROOT / "configs/sprint_t2_v1.yaml").read_text())
    assert {key: value for key, value in sprint.items() if key != "sprint"} == baseline
    assert sprint["sprint"] == {
        "arm": "v1", "aux_weight": 1.0,
        "eval": {"wall_guard_k": 5, "record_raw_final": True},
    }


def test_t1a_config_preserves_smoke_regime_except_budget_and_sprint_block():
    baseline = yaml.safe_load((ROOT / "configs/p1_t1a_sprint_smoke.yaml").read_text())
    sprint = yaml.safe_load((ROOT / "configs/sprint_t1a_v1.yaml").read_text())
    assert sprint["run"] == {**baseline["run"], "total_transitions": 100000}
    for key in ("task", "eval", "td3", "reward", "her", "sim"):
        assert sprint[key] == baseline[key]
    assert sprint["sprint"]["arm"] == "v1"


def test_arm_routing_and_fa_initial_hash_match_baseline():
    driver = load_driver()
    base = driver.load_base_driver(None)
    factory = driver.load_factory()
    from dgcc.rl.sprint_arms import SprintTD3Agent
    from dgcc.rl.td3 import TD3Agent, TD3Config
    from dgcc.tasks.domain import RewardConstants

    config = TD3Config(replay_capacity=32)
    reward = RewardConstants()
    bb = driver.create_seeded_agent(factory, "bb", config, reward, 17, "cpu", 1.0)
    v1 = driver.create_seeded_agent(factory, "v1", config, reward, 17, "cpu", 1.0)
    assert type(bb) is TD3Agent
    assert isinstance(v1, SprintTD3Agent)
    torch.manual_seed(17)
    baseline = TD3Agent(config, device="cpu", reward_constants=reward)
    assert base.initial_weights_sha256(bb) == base.initial_weights_sha256(baseline)


def test_source_bundle_rejected_for_v1_before_startup(tmp_path: Path):
    driver = load_driver()
    with pytest.raises(SystemExit) as error:
        driver.main(["--config", "configs/sprint_t2_v1.yaml", "--arm", "v1", "--source-bundle", str(tmp_path)])
    assert error.value.code == 2


def test_tampered_bundle_refuses_validation(tmp_path: Path):
    driver = load_driver()
    source = tmp_path / "src/dgcc/__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("x = 1\n")
    digest = driver.sha256_file(source)
    (tmp_path / "MANIFEST.sha256").write_text(f"{digest}  src/dgcc/__init__.py\n")
    (tmp_path / "bundle_metadata.json").write_text(json.dumps({
        "source_commit": "786d651", "source_blobs": {"src/dgcc/__init__.py": "gitblob"},
    }))
    source.write_text("x = 2\n")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        driver.validate_source_bundle(tmp_path)
