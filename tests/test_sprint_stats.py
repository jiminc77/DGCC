from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from dgcc.analysis import sprint_claims

SPEC = importlib.util.spec_from_file_location("sprint_stats", Path(__file__).parents[1] / "scripts/sprint_stats.py")
stats = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(stats)
EVAL_SPEC = importlib.util.spec_from_file_location(
    "sprint_heldout_eval", Path(__file__).parents[1] / "scripts/sprint_heldout_eval.py"
)
sprint_heldout_eval = importlib.util.module_from_spec(EVAL_SPEC)
assert EVAL_SPEC.loader is not None
EVAL_SPEC.loader.exec_module(sprint_heldout_eval)


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
    legacy_paths = []
    paths = []
    for seed in sprint_claims.AMD3_PAIRED_SEEDS:
        tag, digest = f"seed{seed}", "a" * 64
        if seed < 3:
            tag = f"m4_t2_s{seed}"
            path = metrics / f"p1_sprint_heldout_claim_{tag}.json"
            claim = {"m4_tag": tag, "split_sha256": sprint_claims.CANONICAL_SPLIT_SHA256}
            path.write_text(json.dumps(claim))
            result_path = metrics / f"p1_t2_sprint_heldout_{tag}.json"
            result_path.write_text(json.dumps({"seed": seed, "split_sha256": sprint_claims.CANONICAL_SPLIT_SHA256, "summary": {"n_episodes": 200, "success_rate": 0.0}}))
            selection_path = metrics / f"p1_sprint_retro_val_{tag}.json"
            selection_path.write_text("{}")
            legacy_paths.extend((path, result_path, selection_path))
        else:
            path = sprint_claims.canonical_claim_path(tag, "bb")
            claim = {"schema_version": 1, "claim_before_load": True, "timestamp": "now", "pid": 1, "run_tag": tag, "arm": "bb", "ckpt_sha256": digest, "split_sha256": sprint_claims.CANONICAL_SPLIT_SHA256, "seed": seed, "config_sha256": digest, "selection_manifest": "/selection", "selection_manifest_sha256": digest, "episode_namespace": 97001, "n_goals": 100}
            path.write_text(json.dumps(claim)); claim_sha = sprint_claims.sha256_file(path)
            episodes = _canonical_episodes()
            result = sprint_heldout_eval.canonical_result_payload(
                run_tag=tag, arm="bb", seed=seed,
                manifest={"config_sha256": digest, "ckpt_sha256": digest, "selector_version": "test", "val_rows": []},
                selection_manifest=claim["selection_manifest"], selection_sha=claim["selection_manifest_sha256"],
                claim_sha=claim_sha, result={"episodes": episodes, **sprint_claims._summary_aggregates(episodes)},
            )
            sprint_claims.canonical_result_path(tag, "bb").write_text(json.dumps(result))
            sprint_claims.canonical_raw_path(tag, "bb").write_bytes(b"raw")
        paths.append(path)
    access_path = metrics / "t2_sprint_heldout_access.log"
    access_path.write_text(json.dumps({"arm": "bb", "purpose": "retro audit"}) + "\n")
    legacy_paths.append(access_path)
    files = {
        str(path.relative_to(tmp_path)): {"sha256": sprint_claims.sha256_file(path), "size": path.stat().st_size}
        for path in legacy_paths
    }
    bundle_path = metrics / "sprint_retro_audit_bundle.json"
    bundle_path.write_text(json.dumps({"schema_version": 1, "split_sha256": sprint_claims.CANONICAL_SPLIT_SHA256, "files": files}))
    # Fixture bundles are synthetic; replace the production immutable anchor for this happy path.
    monkeypatch.setattr(stats, "LEGACY_RETRO_AUDIT_BUNDLE_SHA256", sprint_claims.sha256_file(bundle_path))
    lock = sprint_claims.canonical_metric_lock_path()
    assert stats.publish_metric_lock(paths, lock)["endpoint"] == "return"
    # The freshly issued fixture lock becomes the consumer trust anchor for this repo root.
    monkeypatch.setattr(sprint_claims, "PUBLISHED_METRIC_LOCK_SHA256", sprint_claims.sha256_file(lock))
    sprint_claims.require_metric_lock(lock, "v1")
    original_bundle = bundle_path.read_bytes()
    bundle = json.loads(original_bundle)
    del bundle["files"][str(legacy_paths[0].relative_to(tmp_path))]
    bundle_path.write_text(json.dumps(bundle))
    monkeypatch.setattr(stats, "LEGACY_RETRO_AUDIT_BUNDLE_SHA256", sprint_claims.sha256_file(bundle_path))
    with pytest.raises(sprint_claims.SprintClaimError, match="exact legacy file set"):
        stats._validated_legacy_bb_pair(paths[0])

    bundle_path.write_bytes(original_bundle)
    rewritten_result = json.loads((metrics / "p1_t2_sprint_heldout_m4_t2_s0.json").read_text())
    rewritten_result["summary"]["success_rate"] = 0.5
    rewritten_path = metrics / "p1_t2_sprint_heldout_m4_t2_s0.json"
    rewritten_path.write_text(json.dumps(rewritten_result))
    coherent_bundle = json.loads(original_bundle)
    coherent_bundle["files"][str(rewritten_path.relative_to(tmp_path))] = {
        "sha256": sprint_claims.sha256_file(rewritten_path),
        "size": rewritten_path.stat().st_size,
    }
    bundle_path.write_text(json.dumps(coherent_bundle))
    # The result and bundle agree, but the immutable production-style anchor rejects both rewrites.
    monkeypatch.setattr(stats, "LEGACY_RETRO_AUDIT_BUNDLE_SHA256", hashlib.sha256(original_bundle).hexdigest())
    with pytest.raises(sprint_claims.SprintClaimError, match="immutable SHA-256 anchor"):
        stats._validated_legacy_bb_pair(paths[0])
    with pytest.raises(sprint_claims.SprintClaimError): stats.publish_metric_lock(paths, tmp_path / "off-root.lock")

def test_lock_rejects_summary_tamper_missing_episodes_and_wrong_seed_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sprint_claims, "REPO_ROOT", tmp_path)
    metrics = tmp_path / "outputs/metrics"; metrics.mkdir(parents=True)
    # Boundary checks occur before publication and never accept an off-root claim.
    with pytest.raises(sprint_claims.SprintClaimError): stats.publish_metric_lock([tmp_path / "x"] * 8, sprint_claims.canonical_metric_lock_path())

def test_sensitivity_boundaries() -> None:
    v1 = {seed: [10.] for seed in sprint_claims.AMD3_PAIRED_SEEDS}; bb = {seed: [0.] for seed in sprint_claims.AMD3_PAIRED_SEEDS}
    report = stats.bb_three_way_sensitivity(v1, bb, draws=200)
    assert set(report) >= {"reuse", "new", "pooled", "batch_effect_flag"}
    arms = {arm: {seed: [{"success": True, "eval_wall_guard": arm == "v1"}] for seed in sprint_claims.AMD3_PAIRED_SEEDS} for arm in ("v1", "bb")}
    with pytest.raises(ValueError, match="no episodes"): stats.guard_sensitivity(arms, draws=200)


def test_guard_sensitivity_uses_preregistered_guard_and_common_support_rules() -> None:
    def row(goal_id: str, success: bool, guarded: bool = False) -> dict[str, object]:
        return {"goal_id": goal_id, "success": success, "eval_wall_guard": guarded}

    arms = {"v1": {}, "bb": {}}
    for seed in sprint_claims.AMD3_PAIRED_SEEDS:
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
    lock.write_text(json.dumps({"endpoint": "success_rate", "created_at": "2026-07-22T20:00:00+00:00"}))
    # V1 claims must postdate the lock; the temporal gate is exercised both ways.
    v1_claim = tmp_path / "p1_v1_sprint_heldout_sprint_t2_v1_s0_claim.json"
    v1_claim.write_text(json.dumps({"timestamp": "2026-07-23T00:00:00+00:00"}))
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
    judged = json.loads(output_json.read_text())
    assert judged["lock_predates_all_v1_claims"] is True and judged["v1_claim_files"] == [v1_claim.name]
    assert "do not make an unconditional claim" in output_md.read_text()
    assert "machine-verified to predate" in output_md.read_text()
    # A V1 claim at or before the lock timestamp refuses judgment (fail-closed).
    v1_claim.write_text(json.dumps({"timestamp": "2026-07-22T19:59:59+00:00"}))
    with pytest.raises(sprint_claims.SprintClaimError, match="does not predate"):
        stats.main([
            "judge", "--lock", str(lock), "--seed-effects", str(seed_effects),
            "--json", str(output_json), "--md", str(output_md),
        ])


def test_zero_assert_accepts_anchored_legacy_lines_and_refuses_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sprint_claims, "REPO_ROOT", tmp_path)
    metrics = tmp_path / "outputs/metrics"; metrics.mkdir(parents=True)
    log = metrics / "t2_sprint_heldout_access.log"
    legacy_line = "2026-07-18T01:03:50+00:00 pid=1 argv=scripts/p1_sprint_retro_eval.py --m4-tag m4_t2_s0"
    log.write_text(legacy_line + "\n" + json.dumps({"arm": "bb", "purpose": "final_eval"}) + "\n")
    # Unanchored non-JSONL line refuses the lock.
    with pytest.raises(sprint_claims.SprintClaimError, match="unrecognized non-JSONL"):
        stats._zero_assert()
    # The exact-content anchor admits only the accepted legacy lines.
    monkeypatch.setattr(stats, "LEGACY_ACCESS_LINE_SHA256", frozenset({hashlib.sha256(legacy_line.encode()).hexdigest()}))
    stats._zero_assert()
    # Duplicating an anchored line beyond the frozen count refuses.
    log.write_text((legacy_line + "\n") * 2)
    with pytest.raises(sprint_claims.SprintClaimError, match="duplicated legacy access records"):
        stats._zero_assert()


def test_bundle_access_log_prefix_integrity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sprint_claims, "REPO_ROOT", tmp_path)
    metrics = tmp_path / "outputs/metrics"; metrics.mkdir(parents=True)
    legacy_paths = []
    for seed in range(3):
        tag = f"m4_t2_s{seed}"
        claim = metrics / f"p1_sprint_heldout_claim_{tag}.json"
        claim.write_text(json.dumps({"m4_tag": tag, "split_sha256": sprint_claims.CANONICAL_SPLIT_SHA256}))
        result = metrics / f"p1_t2_sprint_heldout_{tag}.json"
        result.write_text(json.dumps({"seed": seed, "split_sha256": sprint_claims.CANONICAL_SPLIT_SHA256, "summary": {"n_episodes": 200, "success_rate": 0.0}}))
        selection = metrics / f"p1_sprint_retro_val_{tag}.json"; selection.write_text("{}")
        legacy_paths.extend((claim, result, selection))
    access = metrics / "t2_sprint_heldout_access.log"
    prefix = json.dumps({"arm": "bb", "purpose": "retro audit"}) + "\n"
    access.write_text(prefix)
    legacy_paths.append(access)
    files = {str(p.relative_to(tmp_path)): {"sha256": sprint_claims.sha256_file(p), "size": p.stat().st_size} for p in legacy_paths}
    bundle = metrics / "sprint_retro_audit_bundle.json"
    bundle.write_text(json.dumps({"schema_version": 1, "split_sha256": sprint_claims.CANONICAL_SPLIT_SHA256, "files": files}))
    monkeypatch.setattr(stats, "LEGACY_RETRO_AUDIT_BUNDLE_SHA256", sprint_claims.sha256_file(bundle))
    claim0 = metrics / "p1_sprint_heldout_claim_m4_t2_s0.json"
    # Post-epoch append-only growth of the access log is accepted (prefix hash unchanged).
    access.write_text(prefix + json.dumps({"arm": "bb", "purpose": "final_eval"}) + "\n")
    assert stats._validated_legacy_bb_pair(claim0)[0] == {"arm": "bb", "seed": 0}
    # Any mutation of the anchored prefix bytes refuses.
    access.write_text(json.dumps({"arm": "v1", "purpose": "retro audit"}) + "\n" + prefix)
    with pytest.raises(sprint_claims.SprintClaimError, match="access-log prefix"):
        stats._validated_legacy_bb_pair(claim0)


def test_lock_cli_rejects_eight_claims() -> None:
    with pytest.raises(SystemExit) as excinfo:
        stats.main(["lock", "--bb-claims", *(["x.json"] * 8), "--lock", "y.lock"])
    assert excinfo.value.code == 2
