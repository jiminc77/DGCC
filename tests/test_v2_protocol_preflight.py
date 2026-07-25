from __future__ import annotations

import importlib.util
import json
import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
PIN = "0123456789abcdef" * 4
BGT_PIN = "fedcba9876543210" * 4
NEFF_PIN = "7a5b517ab108c8b0afa79d7e544e3f3d1eee40d5e56df6371d94a97bbcaedda5"
NEFF_GUARD = Path("/home/simx2204/v2_research/dossier/V2_neff_guard_beta015363_verified.json")
NEFF_GUARD_BYTES = NEFF_GUARD.read_bytes()


def load_tool():
    spec = importlib.util.spec_from_file_location("v2_protocol_preflight", ROOT / "scripts" / "v2_protocol_preflight.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def governance() -> dict:
    document = json.loads((ROOT / "configs" / "v2_execution_governance.json").read_text())
    document["v2_launch_code_manifest_sha256"] = PIN
    return document


def manifest(**overrides: object) -> dict:
    value = {"arm": "v2-dmm", "mode": "v2_live", "policy_delay": 2, "binding_beta": 0.015363, "code_path": "v2", "config_lineage": "v2", "merged_final_code": True, "code_manifest_sha256": PIN, "neff_guard_sha256": NEFF_PIN, "config_sha256": BGT_PIN, "schedule_arm": "DMM", "seed": 0}
    value.update(overrides)
    return value


def test_clean_v2_launch_requires_independent_code_pin() -> None:
    receipt = load_tool().validate_manifest(manifest(), governance())
    assert receipt["policy_delay"] == 2
    bad_governance = governance()
    bad_governance["v2_launch_code_manifest_sha256"] = BGT_PIN
    with pytest.raises(ValueError, match="independent governance pin"):
        load_tool().validate_manifest(manifest(), bad_governance)
def test_governed_receipt_binds_authenticated_runtime_identities() -> None:
    tool = load_tool()
    receipt = tool.validate_manifest_bytes(
        json.dumps(manifest()).encode(),
        json.dumps(governance()).encode(),
        expected_arm="v2-dmm",
        expected_seed=0,
        expected_config_sha256=BGT_PIN,
        expected_code_manifest_sha256=PIN,
        neff_guard_bytes=NEFF_GUARD_BYTES,
    )
    assert receipt["arm"] == "v2-dmm"
    assert receipt["seed"] == 0
    assert receipt["schedule_arm"] == "DMM"
    assert receipt["config_sha256"] == BGT_PIN
    assert receipt["code_manifest_sha256"] == PIN
    assert receipt["neff_guard_sha256"] == NEFF_PIN
    assert receipt["neff_guard_passed"] is True
    for kwargs in (
        {"expected_arm": "v2-d1m"},
        {"expected_seed": 1},
        {"expected_config_sha256": PIN},
        {"expected_code_manifest_sha256": BGT_PIN},
    ):
        with pytest.raises(ValueError):
            tool.validate_manifest_bytes(
                json.dumps(manifest()).encode(), json.dumps(governance()).encode(),
                neff_guard_bytes=NEFF_GUARD_BYTES, **kwargs
            )
def test_code_manifest_binds_exact_runtime_closure(tmp_path: Path) -> None:
    tool = load_tool()
    files = {}
    for relative_path in tool.required_runtime_files(ROOT):
        source = ROOT / relative_path
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = source.read_bytes()
        destination.write_bytes(payload)
        files[relative_path] = hashlib.sha256(payload).hexdigest()
    document = {
        "schema_version": 1,
        "files": [{"path": path, "sha256": files[path]} for path in sorted(files)],
    }
    manifest_bytes = json.dumps(document).encode()
    receipt = tool.validate_code_manifest_bytes(manifest_bytes, runtime_root=tmp_path)
    assert receipt["code_closure_count"] == len(files)
    assert receipt["code_manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()

    for relative_path in (
        "scripts/p1_train.py",
        "scripts/v2_r5_calibrate.py",
        "src/dgcc/rl/td3.py",
    ):
        target = tmp_path / relative_path
        original = target.read_bytes()
        target.write_bytes(original + b"\n# manifest mutation\n")
        with pytest.raises(ValueError, match="runtime closure"):
            tool.validate_code_manifest_bytes(manifest_bytes, runtime_root=tmp_path)
        target.write_bytes(original)



@pytest.mark.parametrize(
    ("arm", "mode", "schedule_arm"),
    (("bb-d2", "bb_d2", "BB-D2"), ("v1-d2", "v1_d2", "V1-D2")),
)
def test_d2_launches_require_live_code_and_policy_delay_two(
    arm: str, mode: str, schedule_arm: str
) -> None:
    tool = load_tool()
    launch = manifest(
        arm=arm,
        mode=mode,
        schedule_arm=schedule_arm,
        d2_lineage=arm,
    )
    assert tool.validate_manifest(launch, governance())["policy_delay"] == 2
    with pytest.raises(ValueError):
        tool.validate_manifest(launch | {"policy_delay": 1}, governance())
    with pytest.raises(ValueError):
        tool.validate_manifest(launch | {"code_path": "original"}, governance())



@pytest.mark.parametrize("overrides", [{"policy_delay": 1}, {"policy_delay": True}, {"binding_beta": .015364}, {"seed": 3}, {"seed": True}, {"schedule_arm": "D1M"}, {"code_manifest_sha256": "0" * 64}, {"config_sha256": "0" * 64}, {"extra": 1}, {"nested": {"metrics": 1}}, {"nested": {"ranking": 1}}])
def test_v2_launch_rejects_drift_unknown_and_outcome_fields(overrides: dict) -> None:
    with pytest.raises(ValueError):
        load_tool().validate_manifest(manifest(**overrides), governance())


def test_governance_requires_exact_beta_schedule_and_schema() -> None:
    for field, value in (("binding_beta", .015364), ("extra", True)):
        document = governance()
        document[field] = value
        with pytest.raises(ValueError):
            load_tool().validate_manifest(manifest(), document)
    document = governance()
    document["tournament_schedule"]["arms"]["DMM"] = [1, 2, 3]
    with pytest.raises(ValueError):
        load_tool().validate_manifest(manifest(), document)
def test_arm_seed_pin_is_independent_of_schedule() -> None:
    document = governance()
    document["arm_seeds"]["DMM"] = [1, 2, 3]
    with pytest.raises(ValueError, match="seed ordering"):
        load_tool().validate_manifest(manifest(), document)


def test_original_launch_requires_non_null_governance_pins() -> None:
    original = manifest(
        arm="bb",
        mode="original_amd5_v1",
        policy_delay=1,
        code_path="original",
        config_lineage="original",
        schedule_arm=None,
        scenario="s6",
        worktree_head=PIN,
        pinned_worktree_head=PIN,
        config_sha256=BGT_PIN,
        pinned_config_sha256=BGT_PIN,
    )
    with pytest.raises(ValueError, match="independent worktree/config pins"):
        load_tool().validate_manifest(original, governance())
    document = governance()
    document["original_worktree_head_sha256"] = PIN
    document["original_config_sha256"] = BGT_PIN
    assert load_tool().validate_manifest(original, document)
    with pytest.raises(ValueError, match="independent governance pins"):
        load_tool().validate_manifest(original | {"config_sha256": PIN}, document)



def test_bgt_identity_is_conditional_and_pinned() -> None:
    document = governance()
    document["bgt_admitted_manifest_sha256"] = BGT_PIN
    with pytest.raises(ValueError):
        load_tool().validate_manifest(manifest(admitted_manifest_sha256=BGT_PIN), document)
    receipt = load_tool().validate_manifest(
        manifest(
            arm="v2-bgt",
            schedule_arm="BGT",
            admitted_manifest_sha256=BGT_PIN,
        ),
        document,
    )
    assert receipt["admitted_manifest_sha256"] == BGT_PIN
    with pytest.raises(ValueError):
        load_tool().validate_manifest(manifest(arm="v2-bgt", schedule_arm="BGT"), document)


def test_governance_rejects_boolean_seed_aliases() -> None:
    document = governance()
    document["seeds"]["discovery"][0] = False
    document["arm_seeds"]["DMM"][0] = False
    document["tournament_schedule"]["arms"]["DMM"][0] = False
    with pytest.raises(ValueError, match="integers, not booleans"):
        load_tool().validate_manifest(manifest(), document)

def test_neff_guard_requires_authenticated_canonical_evidence() -> None:
    tool = load_tool()
    with pytest.raises(ValueError, match="required"):
        tool.validate_manifest_bytes(json.dumps(manifest()).encode(), json.dumps(governance()).encode())
    with pytest.raises(ValueError, match="authenticated bytes"):
        tool.validate_manifest_bytes(
            json.dumps(manifest()).encode(), json.dumps(governance()).encode(),
            neff_guard_bytes=b"{}",
        )
    with pytest.raises(ValueError, match="independent governance pin"):
        tool.validate_manifest(
            manifest(neff_guard_sha256=PIN), governance()
        )
    missing_pin = governance()
    del missing_pin["neff_guard_sha256"]
    with pytest.raises(ValueError):
        tool.validate_manifest(manifest(), missing_pin)
    wrong_pin = governance()
    wrong_pin["neff_guard_sha256"] = PIN
    with pytest.raises(ValueError, match="canonical N_eff guard"):
        tool.validate_manifest(manifest(), wrong_pin)


def test_neff_guard_rejects_false_out_of_range_and_malformed_provenance() -> None:
    tool = load_tool()

    def rejects(mutator: object) -> None:
        document = json.loads(NEFF_GUARD_BYTES)
        mutator(document)
        payload = json.dumps(document).encode()
        pin = hashlib.sha256(payload).hexdigest()
        tool.NEFF_GUARD_SHA256 = pin
        launch, rules = manifest(neff_guard_sha256=pin), governance()
        rules["neff_guard_sha256"] = pin
        with pytest.raises(ValueError):
            tool.validate_manifest_bytes(
                json.dumps(launch).encode(), json.dumps(rules).encode(),
                neff_guard_bytes=payload,
            )

    rejects(lambda document: document.update(guard_passed=False))
    rejects(lambda document: document.update(q1_pooled_median=20.1))
    rejects(lambda document: document.update(panel_sha256="0" * 64))


def test_neff_guard_valid_receipt_contains_verdict() -> None:
    receipt = load_tool().validate_manifest_bytes(
        json.dumps(manifest()).encode(), json.dumps(governance()).encode(),
        neff_guard_bytes=NEFF_GUARD_BYTES,
    )
    assert receipt["q1_pooled_median"] == 12.000547577506525
    assert receipt["qmin_pooled_median"] == 13.038685532339265
    assert receipt["neff_guard_passed"] is True



def test_receipts_are_exclusive(tmp_path: Path) -> None:
    tool = load_tool()
    launch, receipt = tmp_path / "launch.json", tmp_path / "receipt.json"
    launch.write_text(json.dumps(manifest()))
    (tmp_path / "governance.json").write_text(json.dumps(governance()))
    guard = tmp_path / "guard.json"
    guard.write_bytes(NEFF_GUARD_BYTES)
    assert tool.main(["--manifest", str(launch), "--receipt", str(receipt), "--governance", str(tmp_path / "governance.json"), "--neff-guard", str(guard)]) == 0
    with pytest.raises(FileExistsError):
        tool.main(["--manifest", str(launch), "--receipt", str(receipt), "--governance", str(tmp_path / "governance.json"), "--neff-guard", str(guard)])
