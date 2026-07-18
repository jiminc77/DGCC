"""Synthetic contracts for sprint one-shot authorization primitives."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dgcc.analysis import sprint_claims as claims


def payload() -> dict[str, str]:
    return {"run_tag": "synthetic", "arm": "BB", "ckpt_sha256": "c" * 64, "split_sha256": "s" * 64}


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_claim_contains_primitive_preload_attestation(tmp_path: Path) -> None:
    claim = tmp_path / "claim.json"
    claims.acquire_claim(claim, payload())
    stored = json.loads(claim.read_text())
    assert stored["schema_version"] == claims.PRIMITIVE_SCHEMA_VERSION
    assert stored["claim_before_load"] is True
    assert stored["timestamp"] and stored["pid"] > 0


def test_existing_and_crashed_claim_hard_refuse(tmp_path: Path) -> None:
    claim = tmp_path / "claim.json"
    claims.acquire_claim(claim, payload())  # represents crash after claim
    with pytest.raises(claims.SprintClaimError, match="already exists"):
        claims.acquire_claim(claim, payload())


def test_non_bb_requires_parseable_lock(tmp_path: Path) -> None:
    with pytest.raises(claims.SprintClaimError, match="requires --lock"):
        claims.require_metric_lock(None, "V1")
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"metric": "locked"}))
    claims.require_metric_lock(lock, "V1")
    claims.require_metric_lock(None, "BB")


def test_access_append_is_purpose_coded_and_durable_surface(tmp_path: Path) -> None:
    log = tmp_path / "access.log"
    claims.record_access(log, "final_eval", run_tag="synthetic")
    assert json.loads(log.read_text())["purpose"] == "final_eval"


def test_probe_manifest_is_content_addressed_and_immutable(tmp_path: Path) -> None:
    probe = tmp_path / "probe.h5"
    probe.write_bytes(b"synthetic probe")
    manifest = tmp_path / "manifest.json"
    claims.probe_manifest_register(manifest, probe, {"production_goal": "G-EV"})
    claims.probe_manifest_register(manifest, probe, {"production_goal": "G-EV"})
    assert json.loads(manifest.read_text())["files"][str(probe)]["size"] == probe.stat().st_size
    probe.write_bytes(b"modified")
    with pytest.raises(claims.SprintClaimError, match="immutable"):
        claims.probe_manifest_register(manifest, probe, {"production_goal": "G-EV"})


def test_sprint_rejects_m4_split_before_load(tmp_path: Path) -> None:
    evaluator = load_script("sprint_heldout_eval")
    forbidden = tmp_path / "m4_heldout.json"
    forbidden.write_text("not json")
    with pytest.raises(claims.SprintClaimError, match="M4"):
        evaluator.load_sprint_split(forbidden)


def test_retro_claim_and_audit_precede_split_load() -> None:
    source = (ROOT / "scripts/p1_sprint_retro_eval.py").read_text()
    claim_at = source.index("acquire_claim(")
    audit_at = source.index("record_access(")
    load_at = source.index("load_sprint_heldout()", source.index("def main"))
    assert claim_at < audit_at < load_at
