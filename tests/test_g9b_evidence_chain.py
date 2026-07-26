from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "g9b_chain", Path(__file__).parents[1] / "scripts" / "sprint_g9b_evidence_chain.py"
)
assert SPEC and SPEC.loader
chain = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chain)


def _write(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def evidence_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    metrics = tmp_path / "outputs" / "metrics"
    created = "2026-07-22T20:06:20+00:00"
    audit, digests, bundle_files = [], [], {}
    for seed in (0, 1, 2, 3, 4, 6, 7):
        legacy = seed < 3
        path = metrics / (
            f"p1_sprint_heldout_claim_m4_t2_s{seed}.json"
            if legacy else f"p1_bb_sprint_heldout_sprint_t2_bb_s{seed}_claim.json"
        )
        digest = _write(path, {"seed": seed, "arm": "bb"})
        digests.append(digest)
        audit.append({"seed": seed, "kind": "legacy_bundle" if legacy else "canonical", "claim_sha256": digest})
        if legacy:
            bundle_files[str(path.relative_to(tmp_path))] = {"sha256": digest}
    _write(metrics / "sprint_retro_audit_bundle.json", {"files": bundle_files})
    lock = {"created_at": created, "bb_claim_sha256": digests, "bb_claim_audit": audit}
    lock_digest = _write(metrics / "sprint_metric.lock", lock)
    monkeypatch.setattr(chain, "ISSUED_LOCK_SHA256", lock_digest)

    files = {}
    for seed in range(5):
        tag = f"sprint_t2_v1_s{seed}"
        claim = metrics / f"p1_v1_sprint_heldout_{tag}_claim.json"
        claim_digest = _write(claim, {"run_tag": tag, "timestamp": "2026-07-23T00:00:00+00:00"})
        _write(metrics / f"p1_v1_sprint_heldout_{tag}.json", {"claim_sha256": claim_digest})
        files[str(seed)] = {"run_tag": tag}
    _write(metrics / "sprint_probe_manifest.json", {"files": files})
    return tmp_path


def test_synthetic_incomplete_v1_chain_passes(evidence_repo: Path) -> None:
    report = chain.verify(evidence_repo)
    assert report["verdict"] == "incomplete_v1_set"
    assert {row["status"] for row in report["checks"].values()} == {"PASS"}
    assert "aggregate" not in json.dumps(report)


def test_synthetic_chain_fails_closed_on_result_tamper(evidence_repo: Path) -> None:
    result = evidence_repo / "outputs/metrics/p1_v1_sprint_heldout_sprint_t2_v1_s0.json"
    _write(result, {"claim_sha256": "0" * 64})
    report = chain.verify(evidence_repo)
    assert report["verdict"] == "failed"
    assert report["checks"]["v1_result_claim_links"]["status"] == "FAIL"


def test_synthetic_chain_rejects_fewer_than_five_v1_claims(evidence_repo: Path) -> None:
    for seed in (3, 4):
        (evidence_repo / f"outputs/metrics/p1_v1_sprint_heldout_sprint_t2_v1_s{seed}_claim.json").unlink()
    report = chain.verify(evidence_repo)
    assert report["verdict"] == "failed"
    assert report["checks"]["amd5_v1_set"]["status"] == "FAIL"
