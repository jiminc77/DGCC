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
    assert stats.tost_paired([0.] * 8, .05)["status"] == "confirmatory_holm_2"
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


def test_lock_schema_and_zero_assertion(tmp_path: Path) -> None:
    paths = []
    digest = "a" * 64
    for seed in range(8):
        tag = f"seed{seed}"
        path = tmp_path / f"p1_bb_sprint_heldout_{tag}_claim.json"
        claim = {
            "schema_version": 1, "claim_before_load": True, "timestamp": "now", "pid": 1,
            "run_tag": tag, "arm": "bb", "ckpt_sha256": digest,
            "split_sha256": sprint_claims.CANONICAL_SPLIT_SHA256, "seed": seed,
            "config_sha256": digest, "selection_manifest": "/selection",
            "selection_manifest_sha256": digest, "episode_namespace": 97001, "n_goals": 100,
        }
        path.write_text(json.dumps(claim))
        claim_sha = sprint_claims.sha256_file(path)
        (tmp_path / f"p1_bb_sprint_heldout_{tag}.json").write_text(json.dumps({
            **{key: claim[key] for key in ("run_tag", "arm", "seed", "config_sha256", "ckpt_sha256", "split_sha256", "selection_manifest", "selection_manifest_sha256", "episode_namespace")},
            "claim_sha256": claim_sha, "summary": {"success_rate": 0., "n_episodes": 200},
        }))
        (tmp_path / f"p1_bb_sprint_heldout_{tag}.raw.json.gz").write_bytes(b"raw")
        paths.append(path)
    lock = tmp_path / "metric.lock"
    body = stats.publish_metric_lock(paths, lock)
    assert body["endpoint"] == "return"
    sprint_claims.require_metric_lock(lock, "v1")
    (tmp_path / "p1_v1_sprint_heldout_x.json").write_text("{}")
    with pytest.raises(sprint_claims.SprintClaimError): stats.publish_metric_lock(paths, tmp_path / "another.lock")
