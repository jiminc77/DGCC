"""CPU-only contracts for the V2-BGT R5 calibration tool."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from dgcc.rl.td3 import TD3Config
from dgcc.rl.v2_arms import validate_bgt_manifest
from dgcc.logging.code_manifest import required_runtime_files

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "v2_r5_calibrate", ROOT / "scripts/v2_r5_calibrate.py"
)
assert SPEC is not None and SPEC.loader is not None
R5 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R5)
LINEAGE_PINS = {
    "development_split_sha256": "0123456789abcdef" * 4,
    "checkpoint_sha256": "fedcba9876543210" * 4,
    "panel_sha256": "0011223344556677" * 4,
    "config_sha256": "8899aabbccddeeff" * 4,
    "code_sha256": "13579bdf2468ace0" * 4,
}
def code_manifest(tmp_path: Path) -> tuple[Path, str]:
    files = [
        {"path": relative_path, "sha256": hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()}
        for relative_path in required_runtime_files(ROOT)
    ]
    path = tmp_path / "code-manifest.json"
    path.write_bytes(json.dumps({"schema_version": 1, "files": files}).encode())
    return path, R5.sha256_file(path)





def rows(transition: int, *, correct: bool) -> list[dict]:
    result = []
    for seed in (0, 1, 2):
        for index in range(10):
            predicted = [1.0, 0.0]
            measured = [1.0, 0.0] if correct or index < 4 else [0.0, 1.0]
            result.append({"seed": seed, "transition": transition,
                           "normalized_q1_margin": 0.1 + index / 100,
                           "predicted_progress_top2": predicted,
                           "measured_progress_top2": measured})
    return result
def development_payload(
    *, passing_from: int | None, code_sha256: str = LINEAGE_PINS["code_sha256"]
) -> dict:
    return {
        "schema_version": 1, "data_scope": "development",
        "development_lineage": {
            "development_split_path": "/development/t2.json",
            "development_split_sha256": LINEAGE_PINS["development_split_sha256"],
            "development_split_role": "development_t2_split",
            "checkpoint_sha256": LINEAGE_PINS["checkpoint_sha256"],
            "panel_sha256": LINEAGE_PINS["panel_sha256"],
            "config_sha256": LINEAGE_PINS["config_sha256"],
            "code_sha256": code_sha256,
        },
        "registered_seeds": [0, 1, 2], "rows_per_seed_transition": 10,
        "rows": sum((rows(t, correct=passing_from is not None and t >= passing_from)
                     for t in R5.R5_TRANSITIONS), []),
    }


def test_r5_calibration_selects_first_passing_checkpoint(tmp_path: Path) -> None:
    input_path, output_path = tmp_path / "development_branches.json", tmp_path / "r5_result.json"
    input_path.write_text(json.dumps(development_payload(passing_from=75_000)))
    result = R5.calibrate(input_path, output_path)
    assert result["rank_calibration_passed"] is True
    assert result["onset_transition"] == 75_000
    assert result["gpu_latency_gate"] == "pending"
    assert result["bgt_admitted"] is False
    assert len(result["output_sha256"]) == 64


def test_r5_calibration_fails_closed_without_passing_checkpoint(tmp_path: Path) -> None:
    input_path, output_path = tmp_path / "development_branches.json", tmp_path / "r5_result.json"
    input_path.write_text(json.dumps(development_payload(passing_from=None)))
    result = R5.calibrate(input_path, output_path)
    assert result["rank_calibration_passed"] is False
    assert result["onset_transition"] is None
    assert result["margin"] is None
    assert result["bgt_admitted"] is False


def protocol_record(code_sha256: str, overhead: float = 0.1) -> dict:
    sample_count = R5.BENCHMARK_BLOCKS * R5.BENCHMARK_REPEATS_PER_ARM
    measured_overhead = float((1.0 + overhead) / 1.0 - 1.0)
    scenarios = {}
    for scenario in ("calibrated", "all_eligible_worst_case"):
        scenarios[scenario] = {}
        for batch_size in R5.BENCHMARK_BATCH_SIZES:
            base = [1.0] * sample_count
            bgt = [1.0 + overhead] * sample_count
            scenarios[scenario][str(batch_size)] = {
                "base_ms": {"samples": base, "p50": 1.0, "p95": 1.0},
                "bgt_ms": {"samples": bgt, "p50": 1.0 + overhead, "p95": 1.0 + overhead},
                "blocks": [
                    {
                        "block": block,
                        "order": "AB" if block % 2 == 0 else "BA",
                        "base_ms": [1.0] * R5.BENCHMARK_REPEATS_PER_ARM,
                        "bgt_ms": [1.0 + overhead] * R5.BENCHMARK_REPEATS_PER_ARM,
                        "paired_overhead_fraction": measured_overhead,
                    }
                    for block in range(R5.BENCHMARK_BLOCKS)
                ],
                "eligibility_fraction": 1.0 if scenario == "all_eligible_worst_case" else 0.5,
                "p95_overhead_fraction": measured_overhead,
            }
    return {"schema_version": 2, "device": "cuda", "warmup": R5.BENCHMARK_WARMUP,
            "blocks": R5.BENCHMARK_BLOCKS, "repeats_per_arm": R5.BENCHMARK_REPEATS_PER_ARM,
            "batch_sizes": list(R5.BENCHMARK_BATCH_SIZES), "scenarios": scenarios,
            "schedule": [list(entry) for entry in R5.benchmark_schedule()],
            "max_complete_selector_p95_overhead_fraction": measured_overhead,
            "latency_gate_passed": measured_overhead <= R5.MAX_P95_OVERHEAD_FRACTION,
            "gpu": {
                "torch_version": "test",
                "torch_cuda_version": "test",
                "device_name": "test",
                "driver_version": "test",
                "power_state": "P0",
                "clocks_mhz": {"graphics": "1", "memory": "2"},
                "power_watts": {"draw": "3", "limit": "4"},
            },
            "peak_memory_bytes": 1,
            "checkpoint_sha256": LINEAGE_PINS["checkpoint_sha256"],
            "panel_sha256": LINEAGE_PINS["panel_sha256"],
            "config_sha256": LINEAGE_PINS["config_sha256"],
            "code_sha256": code_sha256}


def test_r5_admission_recomputes_full_protocol_gate(tmp_path: Path) -> None:
    code_manifest_path, code_manifest_sha256 = code_manifest(tmp_path)
    source = tmp_path / "development_branches.json"
    source.write_text(json.dumps(development_payload(
        passing_from=75_000, code_sha256=code_manifest_sha256
    )))
    calibration = tmp_path / "calibration.json"
    calibration_result = R5.calibrate(source, calibration)
    latency = tmp_path / "latency.json"
    record = protocol_record(code_manifest_sha256)
    record.update(
        {
            "margin": calibration_result["margin"],
            "onset_transition": calibration_result["onset_transition"],
        }
    )
    latency.write_text(json.dumps(record))
    admitted = R5.admit(
        calibration, latency, tmp_path / "admitted.json",
        code_manifest_path=code_manifest_path,
        expected_code_manifest_sha256=code_manifest_sha256,
    )
    assert admitted["bgt_admitted"] is True
    record["max_complete_selector_p95_overhead_fraction"] = 0.0
    latency.write_text(json.dumps(record))
    assert R5.admit(
        calibration, latency, tmp_path / "rejected.json",
        code_manifest_path=code_manifest_path,
        expected_code_manifest_sha256=code_manifest_sha256,
    )["bgt_admitted"] is False
def test_r5_admission_rejects_wrong_code_manifest_identity(tmp_path: Path) -> None:
    code_manifest_path, code_manifest_sha256 = code_manifest(tmp_path)
    source = tmp_path / "development_branches.json"
    source.write_text(json.dumps(development_payload(
        passing_from=75_000, code_sha256=code_manifest_sha256
    )))
    calibration = tmp_path / "calibration.json"
    calibration_result = R5.calibrate(source, calibration)
    latency = tmp_path / "latency.json"
    record = protocol_record(code_manifest_sha256)
    record.update({
        "margin": calibration_result["margin"],
        "onset_transition": calibration_result["onset_transition"],
    })
    latency.write_text(json.dumps(record))
    wrong_manifest = tmp_path / "wrong-code-manifest.json"
    wrong_manifest.write_text('{"closure":"wrong"}\n', encoding="utf-8")
    wrong_sha256 = R5.sha256_file(wrong_manifest)
    with pytest.raises(ValueError, match="code manifest"):
        R5.admit(
            calibration, latency, tmp_path / "wrong-admitted.json",
            code_manifest_path=wrong_manifest,
            expected_code_manifest_sha256=wrong_sha256,
        )


def test_r5_admission_manifest_validates_in_runtime(tmp_path: Path) -> None:
    config = TD3Config()
    code_manifest_path, code_manifest_sha256 = code_manifest(tmp_path)
    payload = development_payload(
        passing_from=75_000, code_sha256=code_manifest_sha256
    )
    lineage = payload["development_lineage"]
    lineage["config_sha256"] = R5._sha256_json(config.to_dict())
    source = tmp_path / "development_branches.json"
    source.write_text(json.dumps(payload))
    calibration = tmp_path / "calibration.json"
    calibration_result = R5.calibrate(source, calibration)

    latency_record = protocol_record(code_manifest_sha256)
    for field in (
        "checkpoint_sha256",
        "panel_sha256",
        "config_sha256",
        "code_sha256",
    ):
        latency_record[field] = lineage[field]
    latency_record.update(
        {
            "margin": calibration_result["margin"],
            "onset_transition": calibration_result["onset_transition"],
        }
    )
    latency = tmp_path / "latency.json"
    latency.write_text(json.dumps(latency_record))
    manifest_path = tmp_path / "admitted.json"
    manifest = R5.admit(
        calibration, latency, manifest_path,
        code_manifest_path=code_manifest_path,
        expected_code_manifest_sha256=code_manifest_sha256,
    )
    assert manifest["bgt_admitted"] is True
    validate_bgt_manifest(
        manifest_path,
        R5.sha256_file(manifest_path),
        manifest["checkpoint_sha256"],
        manifest["panel_sha256"],
        manifest["config_sha256"],
        code_manifest_sha256,
    )
def test_r5_rejects_rank_payload_extra_key_and_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "development_branches.json"
    payload = development_payload(passing_from=75_000)
    payload["unregistered"] = True
    source.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="closed"):
        R5.calibrate(source, tmp_path / "result.json")
    source.write_text(json.dumps(development_payload(passing_from=75_000)))
    output = tmp_path / "result.json"
    R5.calibrate(source, output)
    with pytest.raises(FileExistsError):
        R5.calibrate(source, output)


def test_latency_rejects_aggregate_not_reconstructed_from_blocks() -> None:
    record = protocol_record(LINEAGE_PINS["code_sha256"])
    record["scenarios"]["calibrated"]["1024"]["base_ms"]["samples"][0] = 2.0
    assert R5.valid_latency_protocol(record) is False


def test_latency_requires_exact_schedule() -> None:
    record = protocol_record(LINEAGE_PINS["code_sha256"])
    record["schedule"] = record["schedule"][1:]
    assert R5.valid_latency_protocol(record) is False

def test_latency_protocol_requires_complete_gpu_operating_state() -> None:
    for field in (
        "torch_version",
        "torch_cuda_version",
        "device_name",
        "driver_version",
        "power_state",
        "clocks_mhz",
        "power_watts",
    ):
        record = protocol_record(LINEAGE_PINS["code_sha256"])
        del record["gpu"][field]
        assert R5.valid_latency_protocol(record) is False
    for group, field in (("clocks_mhz", "memory"), ("power_watts", "limit")):
        record = protocol_record(LINEAGE_PINS["code_sha256"])
        del record["gpu"][group][field]
        assert R5.valid_latency_protocol(record) is False


def test_r5_admission_rejects_hand_written_calibration_boolean(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "schema_version": 1,
        "data_scope": "development",
        "input_sha256": LINEAGE_PINS["checkpoint_sha256"],
        "rank_calibration_passed": True,
        "margin": 0.2,
        "onset_transition": 75_000,
    }))
    latency = tmp_path / "latency.json"
    code_manifest_path, code_manifest_sha256 = code_manifest(tmp_path)
    record = protocol_record(code_manifest_sha256)
    record.update({"margin": 0.2, "onset_transition": 75_000})
    latency.write_text(json.dumps(record))
    assert R5.admit(
        calibration, latency, tmp_path / "manifest.json",
        code_manifest_path=code_manifest_path,
        expected_code_manifest_sha256=code_manifest_sha256,
    )["bgt_admitted"] is False


def test_tournament_cutoff_records_final_status_and_rejects_identity_drift(tmp_path: Path) -> None:
    code_manifest_path, code_manifest_sha256 = code_manifest(tmp_path)
    identities = (
        LINEAGE_PINS["checkpoint_sha256"],
        LINEAGE_PINS["panel_sha256"],
        LINEAGE_PINS["config_sha256"],
    )
    state = R5.tournament_cutoff(
        tmp_path / "missing-manifest.json", identities[0], tmp_path / "cutoff.json",
        code_manifest_path, code_manifest_sha256, identities[1], identities[2],
        LINEAGE_PINS["development_split_sha256"],
    )
    assert state["status"] == "not-admitted"
    assert "scenario_b_lock" not in state
    different_manifest = tmp_path / "different-code-manifest.json"
    different_manifest.write_text('{"closure":"other"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="code manifest"):
        R5.tournament_cutoff(
            tmp_path / "missing-manifest.json", identities[0], tmp_path / "cutoff.json",
            different_manifest, R5.sha256_file(different_manifest), identities[1], identities[2],
            LINEAGE_PINS["development_split_sha256"],
        )


def test_tournament_cutoff_rejects_present_wrong_manifest_pin_without_state(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    state = tmp_path / "cutoff.json"
    with pytest.raises(ValueError, match="authenticated pin"):
        code_manifest_path, code_manifest_sha256 = code_manifest(tmp_path)
        R5.tournament_cutoff(
            manifest,
            LINEAGE_PINS["development_split_sha256"],
            state,
            code_manifest_path,
            code_manifest_sha256,
            LINEAGE_PINS["checkpoint_sha256"],
            LINEAGE_PINS["panel_sha256"],
            LINEAGE_PINS["config_sha256"],
        )
    assert not state.exists()


def test_tournament_cutoff_is_explicit_cli_subcommand(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    code_manifest_path, code_manifest_sha256 = code_manifest(tmp_path)
    monkeypatch.setattr("sys.argv", [
        "v2_r5_calibrate.py", "tournament_cutoff", "--manifest", str(tmp_path / "manifest.json"),
        "--expected-manifest-sha256", LINEAGE_PINS["development_split_sha256"],
        "--state", str(tmp_path / "state.json"),
        "--code-manifest", str(code_manifest_path),
        "--expected-code-manifest-sha256", code_manifest_sha256,
        "--checkpoint-sha256", LINEAGE_PINS["checkpoint_sha256"],
        "--panel-sha256", LINEAGE_PINS["panel_sha256"],
        "--config-sha256", LINEAGE_PINS["config_sha256"],
    ])
    assert R5.parse_args().command == "tournament_cutoff"
def test_r5_latency_never_writes_a_benchmark_admission_manifest() -> None:
    source = (ROOT / "scripts/v2_r5_calibrate.py").read_text()
    assert "benchmark-manifest" not in source
    assert "BenchmarkBGTAgent" in source



def test_complete_selector_protocol_has_exact_warmup_measurement_order_and_counts() -> None:
    calls, syncs = [], []
    batches = {scenario: {size: (np.full((size, 1), size), np.zeros((size, 1)))
                          for size in R5.BENCHMARK_BATCH_SIZES}
               for scenario in ("calibrated", "all_eligible_worst_case")}
    def selector(arm: str):
        def call(X, _G, force):
            calls.append((arm, int(X[0, 0]), force))
        return call
    tick = iter(range(0, 20_000_000, 1_000))
    result = R5.complete_selector_protocol({"base": selector("base"), "bgt": selector("bgt")}, batches,
                                           synchronize_call=lambda: syncs.append(None), timer_ns=lambda: next(tick))
    assert len(calls) == 2 * 2 * (2 * R5.BENCHMARK_WARMUP + 2 * R5.BENCHMARK_BLOCKS * R5.BENCHMARK_REPEATS_PER_ARM)
    assert len(syncs) == 2 * 2 * 2 * R5.BENCHMARK_BLOCKS * R5.BENCHMARK_REPEATS_PER_ARM * 2
    assert calls[:100] == [("base", 1024, False)] * 50 + [("bgt", 1024, False)] * 50
    assert calls[100:500] == [("base", 1024, False)] * 200 + [("bgt", 1024, False)] * 200
    worst = result["all_eligible_worst_case"]["1024"]
    assert len(worst["base_ms"]["samples"]) == 1000
    assert [block["order"] for block in worst["blocks"]] == ["AB", "BA", "AB", "BA", "AB"]


def test_benchmark_batches_are_deterministic_and_scenarios_share_panel_rows() -> None:
    X = np.arange(1200 * 32 * 3).reshape(1200, 32, 3)
    G = X + 1
    first, second = R5.benchmark_batches(X, G), R5.benchmark_batches(X, G)
    for size in R5.BENCHMARK_BATCH_SIZES:
        assert np.array_equal(first["calibrated"][size][0], second["calibrated"][size][0])
        assert np.array_equal(first["calibrated"][size][0], first["all_eligible_worst_case"][size][0])


def test_r5_refuses_forbidden_asset_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="refuses"):
        R5.permitted_path(tmp_path / "t2_sprint_heldout_v1.json")


def test_r5_gpu_latency_code_exists_but_is_not_executed() -> None:
    source = (ROOT / "scripts/v2_r5_calibrate.py").read_text()
    assert 'device == "cuda"' in source
    assert "torch.cuda.synchronize()" in source


def test_r5_benchmark_schedule_is_fixed_and_cpu_inspectable() -> None:
    schedule = R5.benchmark_schedule()
    assert len(schedule) == 40
    assert schedule[:2] == (("calibrated", 1024, 0, "base", 200), ("calibrated", 1024, 0, "bgt", 200))
    assert schedule[2:4] == (("calibrated", 1024, 1, "bgt", 200), ("calibrated", 1024, 1, "base", 200))
