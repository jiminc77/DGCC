"""CPU-only contracts for the sprint training entry point."""
from __future__ import annotations

import importlib.util
import json
import hashlib
import subprocess
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


@pytest.mark.parametrize(
    ("arm", "sprint_block"),
    [
        ("v1", {"arm": "v1", "aux_weight": 1.0, "eval": {"wall_guard_k": 5, "record_raw_final": True}}),
        ("matched", {"arm": "matched", "aux_weight": 1.0, "projection_seed": 20260719, "eval": {"wall_guard_k": 5, "record_raw_final": True}}),
        ("random", {"arm": "random", "aux_weight": 1.0, "target_seed": 20260718, "eval": {"wall_guard_k": 5, "record_raw_final": True}}),
    ],
)
def test_t2_config_diff_is_only_sprint_block(arm: str, sprint_block: dict) -> None:
    baseline = yaml.safe_load((ROOT / "configs/p1_t2.yaml").read_text())
    sprint = yaml.safe_load((ROOT / f"configs/sprint_t2_{arm}.yaml").read_text())
    assert {key: value for key, value in sprint.items() if key != "sprint"} == baseline
    assert sprint["sprint"] == sprint_block


@pytest.mark.parametrize(
    ("arm", "sprint_block"),
    [
        ("v1", {"arm": "v1", "aux_weight": 1.0, "eval": {"wall_guard_k": 5, "record_raw_final": True}}),
        ("matched", {"arm": "matched", "aux_weight": 1.0, "projection_seed": 20260719, "eval": {"wall_guard_k": 5, "record_raw_final": True}}),
        ("random", {"arm": "random", "aux_weight": 1.0, "target_seed": 20260718, "eval": {"wall_guard_k": 5, "record_raw_final": True}}),
    ],
)
def test_t1a_config_preserves_smoke_regime_except_budget_and_sprint_block(
    arm: str, sprint_block: dict
) -> None:
    baseline = yaml.safe_load((ROOT / "configs/p1_t1a_sprint_smoke.yaml").read_text())
    sprint = yaml.safe_load((ROOT / f"configs/sprint_t1a_{arm}.yaml").read_text())
    assert set(sprint) == set(baseline) | {"sprint"}
    normalized = {**sprint, "run": {**sprint["run"], "total_transitions": baseline["run"]["total_transitions"]}}
    assert {key: value for key, value in normalized.items() if key != "sprint"} == baseline
    assert sprint["sprint"] == sprint_block

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


def write_authenticated_bundle(bundle: Path) -> tuple[dict, str]:
    proof = json.loads((ROOT / "outputs/metrics/sprint_bb_parity_proof.json").read_text())
    source_commit = proof["commits"][0]
    source_blobs = proof["closure_blobs"][source_commit]
    manifest = []
    for relative in source_blobs:
        source = bundle / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(subprocess.run(
            ["git", "show", f"{source_commit}:{relative}"],
            cwd=ROOT, check=True, capture_output=True,
        ).stdout)
        manifest.append(f"{hashlib.sha256(source.read_bytes()).hexdigest()}  {relative}\n")
    (bundle / "MANIFEST.sha256").write_text("".join(manifest))
    (bundle / "bundle_metadata.json").write_text(json.dumps({
        "source_commit": source_commit, "source_blobs": source_blobs,
    }))
    return proof, source_commit


def test_coherent_tampered_bundle_refuses_proof_validation(tmp_path: Path):
    driver = load_driver()
    proof, source_commit = write_authenticated_bundle(tmp_path)
    driver.validate_source_bundle(tmp_path)
    relative = "src/dgcc/__init__.py"
    source = tmp_path / relative
    source.write_text("tampered = True\n")
    metadata_path = tmp_path / "bundle_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["source_blobs"][relative] = subprocess.run(
        ["git", "hash-object", str(source)], check=True, capture_output=True, text=True,
    ).stdout.strip()
    metadata_path.write_text(json.dumps(metadata))
    manifest_path = tmp_path / "MANIFEST.sha256"
    manifest_path.write_text("".join(
        f"{hashlib.sha256((tmp_path / path).read_bytes()).hexdigest()}  {path}\n"
        for path in proof["closure_blobs"][source_commit]
    ))
    with pytest.raises(RuntimeError, match="parity proof"):
        driver.main([
            "--config", "configs/sprint_t2_v1.yaml",
            "--arm", "bb",
            "--source-bundle", str(tmp_path),
        ])
