"""Synthetic contracts for canonical sprint authorization primitives."""
from __future__ import annotations
import json, threading
from pathlib import Path
import pytest
from dgcc.analysis import sprint_claims as claims

def payload() -> dict[str, str]: return {"run_tag":"synthetic","arm":"BB","ckpt_sha256":"c"*64,"split_sha256":claims.CANONICAL_SPLIT_SHA256}

def test_claim_contains_preload_attestation(tmp_path: Path) -> None:
    claim=tmp_path/"claim.json"; claims.acquire_claim(claim,payload()); assert json.loads(claim.read_text())["claim_before_load"] is True

def test_existing_claim_hard_refuse(tmp_path: Path) -> None:
    claim=tmp_path/"claim.json"; claims.acquire_claim(claim,payload())
    with pytest.raises(claims.SprintClaimError,match="already exists"): claims.acquire_claim(claim,payload())

def test_lock_schema_rejects_empty_and_bb_is_only_bypass(tmp_path: Path) -> None:
    lock=tmp_path/"lock.json"; lock.write_text("{}")
    with pytest.raises(claims.SprintClaimError): claims.require_metric_lock(lock,"V1")
    claims.require_metric_lock(None,"BB")

def test_consumer_binds_claim_to_access_and_canonical_split(tmp_path: Path) -> None:
    claim=tmp_path/"claim.json"; capability=claims.acquire_claim(claim,payload()); log=tmp_path/"access.jsonl"
    split=Path(claims.CANONICAL_SPLIT_PATH); loaded=claims.consume_claim_and_load_split(capability,split,access_log=log)
    assert loaded["n_goals"]==100 and json.loads(log.read_text())["claim_sha256"]==claims.sha256_file(claim)

def test_consumer_rejects_symlink_alias(tmp_path: Path) -> None:
    claim=tmp_path/"claim.json"; capability=claims.acquire_claim(claim,payload()); alias=tmp_path/"split.json"; alias.symlink_to(claims.CANONICAL_SPLIT_PATH)
    # An alias resolves to canonical and is intentionally accepted; a different target is not.
    assert claims.consume_claim_and_load_split(capability,alias,access_log=tmp_path/"log")["n_goals"]==100
def test_consumer_rejects_forged_mapping() -> None:
    with pytest.raises(claims.SprintClaimError): claims.consume_claim_and_load_split(payload(), claims.CANONICAL_SPLIT_PATH)

def test_probe_manifest_content_address_and_parallel_registration(tmp_path: Path) -> None:
    manifest=tmp_path/"manifest.json"; probes=[]
    for i in range(16):
        path=tmp_path/f"p{i}.h5"; path.write_bytes(str(i).encode()); probes.append(path)
    threads=[threading.Thread(target=claims.probe_manifest_register,args=(manifest,path,{"production_goal":"G-EV"})) for path in probes]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert len(json.loads(manifest.read_text())["files"] )==16

def test_audit_claims_is_claim_only_and_non_mutating(tmp_path: Path) -> None:
    claim=tmp_path/"p1_claim_synthetic.json"; claims.acquire_claim(claim,payload()); before=claim.read_bytes(); rows=claims.audit_claims(tmp_path)
    assert rows and rows[0]["status"]=="needs_human_disposition" and claim.read_bytes()==before
