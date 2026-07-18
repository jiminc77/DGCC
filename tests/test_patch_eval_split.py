"""CPU-only contract tests for the patch-only T2 evaluation split."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "gen_patch_eval.py"
SPLIT = REPO / "src" / "dgcc" / "tasks" / "splits" / "t2_patch_eval_v1.json"


def _generator_module():
    spec = importlib.util.spec_from_file_location("gen_patch_eval", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_patch_eval_regeneration_is_deterministic_and_matches_committed_split():
    generator = _generator_module()
    first = generator.build_patch_eval_payload()
    second = generator.build_patch_eval_payload()
    committed = json.loads(SPLIT.read_text(encoding="utf-8"))
    assert first == second
    assert committed == first


def test_patch_eval_has_100_goals_and_reproducible_zero_parameter_overlap():
    generator = _generator_module()
    payload = generator.build_patch_eval_payload()
    dedup = payload["dedup_check_access"]
    assert payload["n_goals"] == 100
    assert len(payload["specs"]) == 100
    assert len(payload["goal_ids"]) == 100
    assert dedup["purpose"] == "dedup_check"
    assert dedup["overlap_counts"] == {
        "t2_v1_all_650": 0,
        "t2_sprint_heldout_v1": 0,
        "m4_heldout_100": 0,
    }


def test_patch_eval_schema_records_ood_length_mask_and_seed_contract():
    generator = _generator_module()
    payload = generator.build_patch_eval_payload()
    assert payload["version"] == "t2-patch-eval-v1"
    assert payload["master_seed"] == 20260722
    assert payload["master_seed"] not in {20260703, 20260716, 20260718, 20260719}
    assert set(payload["families"]) == {"s", "u", "l", "zigzag", "smooth_random"}
    assert set(payload["family_distribution"]) == set(payload["families"])
    transform = payload["ood_length_transform"]
    assert transform["l_train_m"] == pytest.approx(1.0)
    assert transform["l_ood_m"] == [0.75, 1.25]
    assert transform["l_ood_over_l_train"] == [0.75, 1.25]
    assert transform["rescale_mask"] == {
        "scaled": "x/y DCT modes 1-7",
        "absolute": "x0, y0, z0, and z DCT modes 1-7",
    }
