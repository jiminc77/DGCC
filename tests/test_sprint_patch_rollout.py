from __future__ import annotations

import importlib.util
from pathlib import Path
import dgcc.analysis.sprint_claims as claims

import pytest


def _module():
    path = Path(__file__).parents[1] / "scripts/sprint_patch_rollout.py"
    spec = importlib.util.spec_from_file_location("sprint_patch_rollout", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _rows(module):
    return [
        {"condition": condition, "length_ratio": ratio, "success": True, "return": ratio}
        for condition in module.CONDITIONS for ratio in module.LENGTH_RATIOS
    ]


def test_aggregate_has_every_condition_and_ood_length() -> None:
    module = _module()
    result = module.aggregate_rollouts(_rows(module))
    assert set(result) == set(module.CONDITIONS)
    assert result["a0_real"]["0.75"] == {"n_episodes": 1, "success_rate": 1.0, "mean_return": 0.75}


def test_aggregate_rejects_missing_rollout_cell() -> None:
    module = _module()
    with pytest.raises(module.SprintClaimError, match="missing"):
        module.aggregate_rollouts(_rows(module)[:-1])


def test_canonical_paths_use_patch_eval_identity() -> None:
    module = _module()
    claim, result = module.canonical_paths("trial", "bb")
    assert claim.name == "p1_bb_patch_eval_trial_claim.json"
    assert result.name == "p1_bb_patch_eval_trial.json"
def test_patch_claim_is_one_shot_and_records_dedicated_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    split = tmp_path / "t2_patch_eval_v1.json"
    split.write_text('{"n_goals":100,"specs":[' + ",".join("{}" for _ in range(100)) + "]}", encoding="utf-8")
    monkeypatch.setattr(claims, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(claims, "PATCH_EVAL_SPLIT_PATH", split)
    monkeypatch.setattr(claims, "PATCH_EVAL_SPLIT_SHA256", claims.sha256_file(split))
    payload = {
        "run_tag": "synthetic", "arm": "bb", "ckpt_sha256": "c" * 64,
        "split_sha256": claims.PATCH_EVAL_SPLIT_SHA256, "seed": 7,
        "config_sha256": "d" * 64, "selection_manifest": "/selection.json",
        "selection_manifest_sha256": "e" * 64, "episode_namespace": 97_001, "n_goals": 100,
    }
    claim = claims.canonical_patch_claim_path("synthetic", "bb")
    capability = claims.acquire_patch_claim(claim, payload)
    access_log = tmp_path / "t2_patch_eval_access.log"
    assert claims.consume_patch_claim_and_load_split(capability, split, access_log=access_log)["n_goals"] == 100
    assert '"purpose": "patch_rollout"' in access_log.read_text(encoding="utf-8")
    with pytest.raises(claims.SprintClaimError, match="unconsumed"):
        claims.consume_patch_claim_and_load_split(capability, split, access_log=access_log)
    with pytest.raises(claims.SprintClaimError, match="already exists"):
        claims.acquire_patch_claim(claim, payload)
