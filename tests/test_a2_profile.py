from __future__ import annotations

import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from sprint_a2_profile import FALLBACK_DECISION, PASS_DECISION, profile, write_profile


def _clock(values: list[float]):
    iterator = iter(values)
    return lambda: next(iterator)


def test_profile_pass_writes_metrics_json_and_calculates_medians(tmp_path) -> None:
    unloaded = []
    result = profile(
        lambda: "checkpoint",
        unloaded.append,
        lambda: None,
        repeats=3,
        patch_count=100,
        # cycle samples: 2, 4, 6; patch samples: 0.1, 0.2, 0.3
        clock=_clock([0, 2, 2, 2.1, 10, 14, 14, 14.2, 20, 26, 26, 26.3]),
    )

    assert unloaded == ["checkpoint", "checkpoint", "checkpoint"]
    assert result["checkpoint_load_unload_seconds"]["median_seconds"] == 4
    assert result["patch_overhead_seconds"]["median_seconds"] == pytest.approx(0.2)
    assert result["patch_overhead_seconds"]["p95_seconds"] == pytest.approx(0.29)
    assert result["projected_total_seconds"] == pytest.approx(20)
    assert result["status"] == "PASS"
    assert result["decision"] == PASS_DECISION

    output = tmp_path / "outputs/metrics/sprint_a2_profile.json"
    write_profile(result, output)
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_profile_recommends_plan_b1_when_projection_exceeds_gpu_day() -> None:
    result = profile(
        lambda: object(),
        lambda _: None,
        lambda: None,
        repeats=1,
        patch_count=86_401,
        clock=_clock([0, 1, 1, 2]),
    )

    assert result["status"] == "FALLBACK"
    assert result["projected_total_seconds"] == 86_401
    assert result["decision"] == FALLBACK_DECISION
