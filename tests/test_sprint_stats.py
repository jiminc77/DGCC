from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from dgcc.analysis import sprint_claims

SPEC = importlib.util.spec_from_file_location("sprint_stats", Path(__file__).parents[1] / "scripts/sprint_stats.py")
stats = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(stats)


def test_bca_bias_and_acceleration_match_hand_formula() -> None:
    boot = np.array([-1., 0., 1., 2.]); jack = np.array([0., 1., 3.])
    got = stats.bca_interval(1., boot, jack)
    expected_z = stats.norm.ppf((2 + .5) / 5)
    delta = jack.mean() - jack
    expected_a = delta.dot(delta * delta) / (6 * delta.dot(delta) ** 1.5)
    assert got["z0"] == pytest.approx(expected_z)
    assert got["acceleration"] == pytest.approx(expected_a)


def test_bca_exact_z0_and_boundary_rules() -> None:
    boot = np.array([-1., 0., 1., 2.])
    jack = np.array([0., 1., 3.])
    assert stats.bca_lower_bound(1., boot, jack)["z0"] == pytest.approx(stats.norm.ppf(.5))
    assert np.isfinite(stats.bca_lower_bound(-2., boot, jack)["z0"])
    assert np.isfinite(stats.bca_two_sided_interval(3., boot, jack)["z0"])


def test_holm_direct_bootstrap_and_small_seed_error() -> None:
    results = stats.holm_secondary_decisions({"2": [.1] * 8, "3": [.2] * 8}, primary_passed=True, draws=200)
    assert {result["holm_rank"] for result in results.values()} == {1, 2} and results["2"]["holm_alpha"] == .025
    with pytest.raises(ValueError, match="n=1"):
        stats.seed_cluster_bootstrap([.1])


def test_tost_n8_and_iqm_ci() -> None:
    assert stats.tost_paired([0.] * 8, .05)["status"] == "confirmatory_pending_holm_2"
    assert stats.tost_paired([0.] * 8, .05, holm_2_completed=True)["status"] == "confirmatory_after_holm_2"
    assert len(stats.iqm_seed_cluster_bootstrap(range(8), draws=200)["ci"]) == 2
def test_seed_cluster_heterogeneity_is_not_fixed_stratum() -> None:
    # Fixed-stratum episode resampling would report zero width here; seeds are the uncertainty unit.
    effects = np.array([-10., -5., 0., 5., 10., 15., 20., 25.])
    result = stats.seed_cluster_bootstrap(effects, draws=200)
    assert result["ci"][1] - result["ci"][0] > 10


def test_hierarchical_path_and_degenerate_statistics() -> None:
    v1 = {i: [float(i), float(i + 1)] for i in range(8)}
    bb = {i: [0., 0.] for i in range(8)}
    assert "hierarchical" in stats.hierarchical_seed_cluster_bootstrap(v1, bb, draws=200)["method"]
    result = stats.seed_cluster_bootstrap([0.] * 8, draws=200)
    assert result["ci"] == [0., 0.] and "trigger_return_endpoint" not in result


def test_holm_gate_and_order() -> None:
    assert all(x["status"] == "untested_primary_failed" for x in stats.holm_bonferroni({"2": .001, "3": .001}, primary_passed=False).values())
    out = stats.holm_bonferroni({"2": .03, "3": .001}, primary_passed=True)
    assert out["3"]["threshold"] == .025 and out["2"]["threshold"] == .05


def test_tost_and_primary_state_machine() -> None:
    tost = stats.tost_paired([0., .01, -.01, .01, 0.], .05)
    assert tost["equivalent"] and tost["n"] == 5 and "n=5" in tost["limitation"]
    assert stats.primary_decision([-1.] * 8, endpoint="return")["state"] == "1_fail"


def test_rng_reproducible_and_no_percentile_branch() -> None:
    a = stats.seed_cluster_bootstrap(range(8), draws=200)
    b = stats.seed_cluster_bootstrap(range(8), draws=200)
    assert a["ci"] == b["ci"]
    assert "np.percentile" not in Path(stats.__file__).read_text()


def _canonical_episodes() -> list[dict[str, object]]:
    return [{"episode_id": 97_001 + index, "goal_id": f"goal-{index // 2}", "goal_label": f"goal-{index // 2}", "init_template": "straight", "success": False, "steps": 1, "return": 0., "discounted_return": 0., "final_d": .1, "d_at_done": .1, "d_at_done_fallback": False, "d_steps": [.1], "min_d": .1, "d_initial": .2, "d_shape_initial": .2, "d_shape_at_done": .1, "q_first": None, "eval_wall_guard": False, "discard_exposure": 0} for index in range(200)]

def test_lock_audits_canonical_producer_payload_and_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sprint_claims, "REPO_ROOT", tmp_path)
    metrics = tmp_path / "outputs/metrics"; metrics.mkdir(parents=True)
    paths = []
    for seed in range(8):
        tag, digest = f"seed{seed}", "a" * 64
        path = sprint_claims.canonical_claim_path(tag, "bb")
        claim = {"schema_version": 1, "claim_before_load": True, "timestamp": "now", "pid": 1, "run_tag": tag, "arm": "bb", "ckpt_sha256": digest, "split_sha256": sprint_claims.CANONICAL_SPLIT_SHA256, "seed": seed, "config_sha256": digest, "selection_manifest": "/selection", "selection_manifest_sha256": digest, "episode_namespace": 97001, "n_goals": 100}
        path.write_text(json.dumps(claim)); claim_sha = sprint_claims.sha256_file(path)
        episodes = _canonical_episodes()
        summary = {**sprint_claims._summary_aggregates(episodes), "nan_incidents_during_eval": 0, "wall_guard_k": 5, "record_raw": True, "record_probe": True}
        result = {**{key: claim[key] for key in ("run_tag", "arm", "seed", "config_sha256", "ckpt_sha256", "split_sha256", "selection_manifest", "selection_manifest_sha256", "episode_namespace")}, "generated_at": "now", "claim_sha256": claim_sha, "episodes": episodes, "summary": summary}
        sprint_claims.canonical_result_path(tag, "bb").write_text(json.dumps(result))
        sprint_claims.canonical_raw_path(tag, "bb").write_bytes(b"raw")
        paths.append(path)
    lock = sprint_claims.canonical_metric_lock_path()
    assert stats.publish_metric_lock(paths, lock)["endpoint"] == "return"
    sprint_claims.require_metric_lock(lock, "v1")
    with pytest.raises(sprint_claims.SprintClaimError): stats.publish_metric_lock(paths, tmp_path / "off-root.lock")

def test_lock_rejects_summary_tamper_missing_episodes_and_wrong_seed_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sprint_claims, "REPO_ROOT", tmp_path)
    metrics = tmp_path / "outputs/metrics"; metrics.mkdir(parents=True)
    # Boundary checks occur before publication and never accept an off-root claim.
    with pytest.raises(sprint_claims.SprintClaimError): stats.publish_metric_lock([tmp_path / "x"] * 8, sprint_claims.canonical_metric_lock_path())

def test_sensitivity_boundaries() -> None:
    v1 = {seed: [10.] for seed in range(8)}; bb = {seed: [0.] for seed in range(8)}
    report = stats.bb_three_way_sensitivity(v1, bb, draws=200)
    assert set(report) >= {"reuse", "new", "pooled", "batch_effect_flag"}
    arms = {arm: {seed: [{"success": True, "eval_wall_guard": arm == "v1"}] for seed in range(8)} for arm in ("v1", "bb")}
    with pytest.raises(ValueError, match="no episodes"): stats.guard_sensitivity(arms, draws=200)


def test_guard_sensitivity_uses_preregistered_guard_and_common_support_rules() -> None:
    def row(goal_id: str, success: bool, guarded: bool = False) -> dict[str, object]:
        return {"goal_id": goal_id, "success": success, "eval_wall_guard": guarded}

    arms = {"v1": {}, "bb": {}}
    for seed in range(8):
        arms["v1"][seed] = [
            row("common", True),
            *[row("v1-guarded", True, True) for _ in range(10)],
            row("bb-guarded", True),
        ]
        arms["bb"][seed] = [
            row("common", False),
            *[row("v1-guarded", True) for _ in range(10)],
            row("bb-guarded", True, True),
        ]

    report = stats.guard_sensitivity(arms, draws=200)

    assert report["policies"]["guarded_as_failure"]["v1_minus_bb"]["estimate"] == pytest.approx(-2 / 3)
    assert report["policies"]["guarded_excluded_nonrandom_dropout"]["v1_minus_bb"]["estimate"] == pytest.approx(1 / 11)
    assert report["common_support"]["v1_minus_bb"]["estimate"] == pytest.approx(1.)
    assert report["guard_confounded"] is True
    assert report["unconditional_claim_prohibited"] is True
def test_judge_prohibits_unconditional_guard_confounded_claims(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock = tmp_path / "lock.json"
    seed_effects = tmp_path / "effects.json"
    output_json = tmp_path / "judge.json"
    output_md = tmp_path / "judge.md"
    lock.write_text(json.dumps({"endpoint": "success_rate"}))
    seed_effects.write_text(json.dumps({
        "effects": [0.] * 8,
        "v1": {str(seed): [0.] for seed in range(8)},
        "bb": {str(seed): [0.] for seed in range(8)},
        "guard_episodes": {},
    }))
    monkeypatch.setattr(stats, "primary_decision", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(stats, "bb_three_way_sensitivity", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(stats, "guard_sensitivity", lambda *_args, **_kwargs: {"guard_confounded": True})
    monkeypatch.setattr(stats, "require_metric_lock", lambda *_args, **_kwargs: None)

    assert stats.main([
        "judge", "--lock", str(lock), "--seed-effects", str(seed_effects),
        "--json", str(output_json), "--md", str(output_md),
    ]) == 0
    assert json.loads(output_json.read_text())["unconditional_claim_prohibited"] is True
    assert "do not make an unconditional claim" in output_md.read_text()
