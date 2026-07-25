"""CPU-only contracts for the sprint training entry point."""
from __future__ import annotations

import importlib.util
import json
import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

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
    d2_config = TD3Config(replay_capacity=32, policy_delay=2)
    reward = RewardConstants()
    bb = driver.create_seeded_agent(factory, "bb", config, reward, 17, "cpu", 1.0)
    bb_d2 = driver.create_seeded_agent(factory, "bb-d2", d2_config, reward, 17, "cpu", 1.0)
    v1 = driver.create_seeded_agent(factory, "v1", config, reward, 17, "cpu", 1.0)
    v1_d2 = driver.create_seeded_agent(factory, "v1-d2", d2_config, reward, 17, "cpu", 1.0)
    with pytest.raises(ValueError, match="policy_delay"):
        driver.create_seeded_agent(factory, "bb-d2", config, reward, 17, "cpu", 1.0)
    assert type(bb) is TD3Agent
    assert type(bb_d2) is TD3Agent
    assert isinstance(v1, SprintTD3Agent)
    assert isinstance(v1_d2, SprintTD3Agent)
    torch.manual_seed(17)
    baseline = TD3Agent(config, device="cpu", reward_constants=reward)
    assert base.initial_weights_sha256(bb) == base.initial_weights_sha256(baseline)


def test_t2_validation_pairs_expand_in_order_and_reject_invalid_counts() -> None:
    base = load_driver().load_base_driver(None)
    first, second = object(), object()
    labels, goals = base.expand_t2_validation_pairs(
        [("first", first), ("second", second)], 3
    )
    assert labels == ["first", "first", "first", "second", "second", "second"]
    assert goals == [first, first, first, second, second, second]
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            base.expand_t2_validation_pairs([("first", first)], invalid)


@pytest.mark.parametrize("guard_role", ("neff_guard", "wrong_role", None))
@pytest.mark.parametrize("guard_passed", (True, False))
def test_bgt_driver_rejects_manifest_pin_not_bound_by_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, guard_role: str | None, guard_passed: bool
) -> None:
    driver = load_driver()
    governed_pin = "ab" * 32
    configured_pin = "cd" * 32
    real_load_module = driver.load_module

    def load_module(path, name):
        if Path(path).name == "v2_protocol_preflight.py":
            return SimpleNamespace(
                DEFAULT_GOVERNANCE=tmp_path / "governance.json",
                REQUIRED_RUNTIME_FILES=frozenset({"runtime.py"}),
                validate_manifest_bytes=lambda _manifest_bytes, _governance_bytes, **kwargs: {
                    "admitted_manifest_sha256": governed_pin,
                    "arm": kwargs["expected_arm"],
                    "seed": kwargs["expected_seed"],
                    "schedule_arm": driver.GOVERNED_SCHEDULE_ARMS[kwargs["expected_arm"]],
                    "config_sha256": kwargs["expected_config_sha256"],
                    "code_manifest_sha256": kwargs["expected_code_manifest_sha256"],
                    "code_closure_count": 1,
                    "code_closure_sha256": "ef" * 32,
                    "neff_guard_sha256": hashlib.sha256(kwargs["neff_guard_bytes"]).hexdigest(),
                    "q1_pooled_median": 12.0,
                    "qmin_pooled_median": 12.0,
                    "guard_passed": guard_passed,
                    "neff_guard_passed": True,
                },
            )
        return real_load_module(path, name)

    monkeypatch.setattr(driver, "load_module", load_module)
    launch = tmp_path / "launch.json"
    governance = tmp_path / "governance.json"
    code_manifest = tmp_path / "code-manifest.json"
    neff_guard = tmp_path / "neff-guard.json"
    protected_bgt_manifest = tmp_path / "protected-bgt-manifest.json"
    for path in (launch, governance, code_manifest, neff_guard, protected_bgt_manifest):
        path.write_bytes(
            b'{"guard_passed":true,"q1_pooled_median":12.0,"qmin_pooled_median":12.0}'
            if path == neff_guard
            else b"{}"
        )
    config = tmp_path / "v2-bgt.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "task": "t2",
                "td3": {"policy_delay": 2},
                "sprint": {
                    "arm": "v2-bgt",
                    "beta_contact": 0.015363,
                    "amd5_preflight": {
                        "manifest_path": "launch.json",
                        "governance_path": "governance.json",
                        "code_manifest_path": "code-manifest.json",
                        "neff_guard_path": "neff-guard.json",
                    },
                    "bgt": {
                        "manifest_path": "protected-bgt-manifest.json",
                        "expected_manifest_sha256": configured_pin,
                    },
                },
            }
        )
    )
    asset_manifest = tmp_path / "assets.json"
    asset_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "assets": [
                    {
                        "path": str(config),
                        "sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                        "role": "config",
                    },
                    {
                        "path": str(launch),
                        "sha256": hashlib.sha256(launch.read_bytes()).hexdigest(),
                        "role": "preflight_manifest",
                    },
                    {
                        "path": str(governance),
                        "sha256": hashlib.sha256(governance.read_bytes()).hexdigest(),
                        "role": "execution_governance",
                    },
                    {
                        "path": str(code_manifest),
                        "sha256": hashlib.sha256(code_manifest.read_bytes()).hexdigest(),
                        "role": "code_manifest",
                    },
                    *(
                        [
                            {
                                "path": str(neff_guard),
                                "sha256": hashlib.sha256(neff_guard.read_bytes()).hexdigest(),
                                "role": guard_role,
                            }
                        ]
                        if guard_role is not None
                        else []
                    ),
                ],
            }
        )
    )
    asset_manifest_sha256 = hashlib.sha256(
        asset_manifest.read_bytes()
    ).hexdigest()
    with pytest.raises(SystemExit) as error:
        driver.main(
            [
                "--config",
                str(config),
                "--arm",
                "v2-bgt",
                "--asset-manifest",
                str(asset_manifest),
                "--expected-asset-manifest-sha256",
                asset_manifest_sha256,
                "--device",
                "cpu",
            ]
        )
    assert error.value.code == 2


def test_governed_constructor_failure_retains_preparing_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    driver = load_driver()
    base = driver.load_base_driver(None)
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    monkeypatch.setattr(driver, "load_base_driver", lambda _bundle: base)
    monkeypatch.setattr(driver, "load_factory", lambda _bundle: object())
    launch, governance, code_manifest, neff_guard = (
        tmp_path / "launch.json",
        tmp_path / "governance.json",
        tmp_path / "code-manifest.json",
        tmp_path / "neff-guard.json",
    )
    for path in (launch, governance, code_manifest, neff_guard):
        path.write_bytes(
            b'{"guard_passed":true,"q1_pooled_median":12.0,"qmin_pooled_median":12.0}'
            if path == neff_guard
            else b"{}"
        )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "task": "t1a_straighten",
                "td3": {"policy_delay": 2},
                "sprint": {
                    "arm": "v2-d1m",
                    "beta_contact": 0.015363,
                    "amd5_preflight": {
                        "manifest_path": "launch.json",
                        "governance_path": "governance.json",
                        "code_manifest_path": "code-manifest.json",
                        "neff_guard_path": "neff-guard.json",
                    },
                },
            }
        )
    )
    asset_manifest = tmp_path / "assets.json"
    asset_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "assets": [
                    {
                        "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "role": role,
                    }
                    for path, role in (
                        (config, "config"),
                        (launch, "preflight_manifest"),
                        (governance, "execution_governance"),
                        (code_manifest, "code_manifest"),
                        (neff_guard, "neff_guard"),
                    )
                ],
            }
        )
    )
    expected_assets = hashlib.sha256(asset_manifest.read_bytes()).hexdigest()

    def load_module(path, name):
        if Path(path).name != "v2_protocol_preflight.py":
            raise AssertionError(f"unexpected module: {path}")
        return SimpleNamespace(
            DEFAULT_GOVERNANCE=governance,
            REQUIRED_RUNTIME_FILES=frozenset({"runtime.py"}),
            validate_manifest_bytes=lambda _manifest, _governance, **kwargs: {
                "schema_version": 2,
                "content_address": "a" * 64,
                "arm": kwargs["expected_arm"],
                "seed": kwargs["expected_seed"],
                "schedule_arm": driver.GOVERNED_SCHEDULE_ARMS[kwargs["expected_arm"]],
                "mode": "confirmatory",
                "policy_delay": 2,
                "config_sha256": kwargs["expected_config_sha256"],
                "code_manifest_sha256": kwargs["expected_code_manifest_sha256"],
                "governance_sha256": "b" * 64,
                "admitted_manifest_sha256": None,
                "code_closure_count": 1,
                "code_closure_sha256": "c" * 64,
                "neff_guard_sha256": hashlib.sha256(kwargs["neff_guard_bytes"]).hexdigest(),
                "q1_pooled_median": 12.0,
                "qmin_pooled_median": 12.0,
                "guard_passed": True,
                "neff_guard_passed": True,
            },
        )

    class FailingRun:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("constructor failed")

    monkeypatch.setattr(driver, "load_module", load_module)
    monkeypatch.setattr(base, "TrainingRun", FailingRun)
    with pytest.raises(RuntimeError, match="constructor failed"):
        driver.main(
            [
                "--config",
                str(config),
                "--arm",
                "v2-d1m",
                "--asset-manifest",
                str(asset_manifest),
                "--expected-asset-manifest-sha256",
                expected_assets,
                "--device",
                "cpu",
            ]
        )
    attempt = next(
        path
        for path in (tmp_path / "outputs" / "attempts").iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    preparing = base.AttemptRegistry.read_records(attempt)[0]
    receipt = attempt / "reports" / "governed_launch_receipt.json"
    assert preparing["governed_launch_receipt_sha256"] == hashlib.sha256(
        receipt.read_bytes()
    ).hexdigest()
    assert base.AttemptRegistry.read_records(attempt)[-1]["artifact_sha256"][
        "reports/governed_launch_receipt.json"
    ] == hashlib.sha256(receipt.read_bytes()).hexdigest()

def test_base_entrypoint_hashes_log_only_after_durable_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = load_driver().load_base_driver(None)

    received_configs: list[dict] = []

    class FakeRun:
        def __init__(self, _args, _registry, config):
            received_configs.append(config)

        def run(self) -> int:
            print("final buffered line")
            return 0

    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"task": "t1a_straighten"}))
    monkeypatch.setattr(base, "TrainingRun", FakeRun)
    monkeypatch.setattr(
        "sys.argv", ["p1_train.py", "--config", str(config), "--device", "cpu"]
    )
    monkeypatch.chdir(tmp_path)

    assert base.main() == 0
    attempt = next(
        path
        for path in (tmp_path / "outputs" / "attempts").iterdir()
        if not path.name.startswith(".") and path.is_dir()
    )
    terminal = base.AttemptRegistry.read_records(attempt)[-1]
    preparing = base.AttemptRegistry.read_records(attempt)[0]
    log_path = attempt / "reports" / "p1_train.log"
    assert received_configs == [{"task": "t1a_straighten"}]
    assert preparing["config_sha256"] == hashlib.sha256(
        json.dumps(received_configs[0], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert terminal["disposition"] == "SUCCEEDED"
    assert terminal["artifact_sha256"]["reports/p1_train.log"] == hashlib.sha256(
        log_path.read_bytes()
    ).hexdigest()


@pytest.mark.parametrize(
    ("error_type", "disposition"),
    (
        (RuntimeError, "TECHNICAL_FAILURE"),
        (KeyboardInterrupt, "ABORTED"),
    ),
)
def test_base_entrypoint_hashes_exception_artifacts_and_anchor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error_type,
    disposition: str,
) -> None:
    base = load_driver().load_base_driver(None)

    class FailingRun:
        def __init__(self, _args, registry, config):
            self.registry = registry

        def run(self) -> int:
            artifact = self.registry.attempt_path / "metrics" / "failure.json"
            artifact.parent.mkdir()
            artifact.write_text('{"before":"failure"}\n')
            print("failure log line")
            raise error_type("stop")

    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"task": "t1a_straighten"}))
    monkeypatch.setattr(base, "TrainingRun", FailingRun)
    monkeypatch.setattr(
        "sys.argv", ["p1_train.py", "--config", str(config), "--device", "cpu"]
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(error_type):
        base.main()
    attempt = next(
        path
        for path in (tmp_path / "outputs" / "attempts").iterdir()
        if not path.name.startswith(".") and path.is_dir()
    )
    terminal = base.AttemptRegistry.read_records(attempt)[-1]
    assert terminal["disposition"] == disposition
    assert set(terminal["artifact_sha256"]) == {
        "metrics/failure.json",
        "reports/p1_train.log",
    }
    receipt = next((tmp_path / "outputs" / "terminal-anchors").iterdir())
    assert base.AttemptRegistry.verify_terminal_anchor(attempt, receipt)

def test_base_entrypoint_recovers_before_constructing_new_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = load_driver().load_base_driver(None)
    prior = base.AttemptRegistry(
        tmp_path / "outputs" / "attempts",
        run_tag="interrupted",
        config={"task": "t1a_straighten"},
        code_sha256="a" * 64,
        seed=0,
    )
    prior._lock_handle.close()

    class FakeRun:
        def __init__(self, _args, _registry, config):
            pass

        def run(self) -> int:
            return 0

    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"task": "t1a_straighten"}))
    monkeypatch.setattr(base, "TrainingRun", FakeRun)
    monkeypatch.setattr(
        "sys.argv", ["p1_train.py", "--config", str(config), "--device", "cpu"]
    )
    monkeypatch.chdir(tmp_path)

    assert base.main() == 0
    assert base.AttemptRegistry.read_records(prior.attempt_path)[-1]["disposition"] == "ORPHANED"


def test_base_entrypoint_preserves_primary_exception_when_finalization_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = load_driver().load_base_driver(None)

    class FailingRun:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self) -> int:
            raise RuntimeError("primary failure")

    def fail_finalization(*_args, **_kwargs):
        raise OSError("terminal append unavailable")

    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"task": "t1a_straighten"}))
    monkeypatch.setattr(base, "TrainingRun", FailingRun)
    monkeypatch.setattr(base.AttemptRegistry, "finalize_once", fail_finalization)
    monkeypatch.setattr(
        "sys.argv", ["p1_train.py", "--config", str(config), "--device", "cpu"]
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="primary failure"):
        base.main()
    assert "attempt finalization failed after RuntimeError" in capsys.readouterr().err

@pytest.mark.parametrize("arm", ("v1", "bb-d2", "v1-d2"))
def test_source_bundle_rejected_for_nonlegacy_arms_before_startup(tmp_path: Path, arm: str):
    driver = load_driver()
    with pytest.raises(SystemExit) as error:
        driver.main(["--config", "configs/sprint_t2_v1.yaml", "--arm", arm, "--source-bundle", str(tmp_path)])
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
def test_release_artifact_inventory_is_reference_only_with_sensitive_content_denied() -> None:
    inventory = json.loads((ROOT / "configs/release_artifact_paths.json").read_text())
    assert inventory["bgt"] == {
        "status": "conditional",
        "exclusion_artifact_required": False,
    }
    assert set(inventory) == {
        "schema_version",
        "purpose",
        "reference_artifacts",
        "sensitive_references_denied_content_not_opened",
        "explicitly_excluded_nonexistent_artifacts",
        "bgt",
    }
    assert inventory["sensitive_references_denied_content_not_opened"]
    assert not any(
        "bgt_exclusion" in path
        for path in inventory["reference_artifacts"]
    )


def test_frozen_restore_uses_full_786d651_commit_not_289c543() -> None:
    spec = importlib.util.spec_from_file_location(
        "restore_frozen_bundle_test", ROOT / "scripts/restore_frozen_bundle.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    blobs = module.source_blobs()
    assert module.SOURCE_COMMIT == "786d651a4b0f6013971bf1d8f23b125062223679"
    assert blobs
    assert "src/dgcc/rl/td3.py" in blobs
