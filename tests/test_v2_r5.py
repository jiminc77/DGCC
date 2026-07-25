"""CPU-only contracts for the V2-BGT R5 calibration tool."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "v2_r5_calibrate", ROOT / "scripts/v2_r5_calibrate.py"
)
assert SPEC is not None and SPEC.loader is not None
R5 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R5)


def rows(transition: int, *, correct: bool) -> list[dict]:
    result = []
    for seed in (0, 1, 2):
        for index in range(10):
            predicted = [1.0, 0.0]
            measured = [1.0, 0.0] if correct or index < 4 else [0.0, 1.0]
            result.append(
                {
                    "seed": seed,
                    "transition": transition,
                    "normalized_q1_margin": 0.1 + index / 100,
                    "predicted_progress_top2": predicted,
                    "measured_progress_top2": measured,
                }
            )
    return result


def test_r5_calibration_selects_first_passing_checkpoint(tmp_path: Path) -> None:
    input_path = tmp_path / "development_branches.json"
    output_path = tmp_path / "r5_result.json"
    input_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "data_scope": "development",
                "rows": rows(50_000, correct=False) + rows(75_000, correct=True),
            }
        )
    )
    result = R5.calibrate(input_path, output_path)
    assert result["rank_calibration_passed"] is True
    assert result["onset_transition"] == 75_000
    assert result["gpu_latency_gate"] == "pending"
    assert result["bgt_admitted"] is False
    assert len(result["output_sha256"]) == 64


def test_r5_calibration_fails_closed_without_passing_checkpoint(tmp_path: Path) -> None:
    input_path = tmp_path / "development_branches.json"
    output_path = tmp_path / "r5_result.json"
    input_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "data_scope": "development",
                "rows": rows(150_000, correct=False),
            }
        )
    )
    result = R5.calibrate(input_path, output_path)
    assert result["rank_calibration_passed"] is False
    assert result["onset_transition"] is None
    assert result["margin"] is None
    assert result["bgt_admitted"] is False


def test_r5_admission_requires_synchronized_gpu_latency(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "rank_calibration_passed": True,
                "margin": 0.2,
                "onset_transition": 75_000,
            }
        )
    )
    cpu_latency = tmp_path / "cpu_latency.json"
    cpu_latency.write_text(json.dumps({"device": "cpu", "latency_gate_passed": True}))
    rejected = R5.admit(calibration, cpu_latency, tmp_path / "rejected.json")
    assert rejected["bgt_admitted"] is False
    assert "sprint_bgt_config" not in rejected

    gpu_latency = tmp_path / "gpu_latency.json"
    gpu_latency.write_text(json.dumps({"device": "cuda", "latency_gate_passed": True}))
    admitted = R5.admit(calibration, gpu_latency, tmp_path / "admitted.json")
    assert admitted["bgt_admitted"] is True
    assert admitted["sprint_bgt_config"]["margin"] == 0.2
    assert len(admitted["sprint_bgt_config"]["calibration_sha256"]) == 64


def test_r5_refuses_forbidden_asset_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="refuses"):
        R5.permitted_path(tmp_path / "t2_sprint_heldout_v1.json")


def test_r5_gpu_latency_code_exists_but_is_not_executed() -> None:
    source = (ROOT / "scripts/v2_r5_calibrate.py").read_text()
    assert 'device == "cuda"' in source
    assert "torch.cuda.synchronize()" in source
