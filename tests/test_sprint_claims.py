"""Synthetic contracts for canonical sprint authorization primitives."""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import pytest

from dgcc.analysis import sprint_claims as claims


@pytest.fixture
def canonical_sprint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the source-anchored repository only for isolated synthetic tests."""
    root = tmp_path / "repo"
    split = root / "src/dgcc/tasks/splits/t2_sprint_heldout_v1.json"
    split.parent.mkdir(parents=True)
    split.write_text(json.dumps({"n_goals": 100, "specs": [{} for _ in range(100)]}))
    monkeypatch.setattr(claims, "REPO_ROOT", root)
    monkeypatch.setattr(claims, "CANONICAL_SPLIT_PATH", split)
    monkeypatch.setattr(claims, "CANONICAL_SPLIT_SHA256", claims.sha256_file(split))
    claims._ISSUED_CAPABILITIES.clear()
    yield root
    claims._ISSUED_CAPABILITIES.clear()


def payload(*, generation: str | None = None) -> dict[str, object]:
    body: dict[str, object] = {
        "run_tag": "synthetic",
        "arm": "BB",
        "ckpt_sha256": "c" * 64,
        "split_sha256": claims.CANONICAL_SPLIT_SHA256,
        "seed": 7,
        "config_sha256": "d" * 64,
        "selection_manifest": "/synthetic/selection.json",
        "selection_manifest_sha256": "e" * 64,
        "episode_namespace": 97_001,
    }
    if generation is not None:
        body.update(generation=generation, legacy_claim_sha256="a" * 64, disposition_receipt_sha256="b" * 64)
    return body


def claim_path(*, generation: str | None = None) -> Path:
    return claims.canonical_claim_path("synthetic", "bb", generation)


def test_happy_path_contains_preload_attestation_and_access_audit(canonical_sprint: Path) -> None:
    claim = claim_path()
    capability = claims.acquire_claim(claim, payload())
    access_log = canonical_sprint / "access.jsonl"
    loaded = claims.consume_claim_and_load_split(capability, claims.CANONICAL_SPLIT_PATH, access_log=access_log)
    record = json.loads(access_log.read_text())
    assert loaded["n_goals"] == len(loaded["specs"]) == 100
    assert json.loads(claim.read_text())["claim_before_load"] is True
    assert record["claim_sha256"] == claims.sha256_file(claim)
    assert record["split_sha256"] == claims.CANONICAL_SPLIT_SHA256


def test_claim_path_is_canonical_and_independent_of_cwd(canonical_sprint: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = claim_path()
    monkeypatch.chdir(canonical_sprint.parent)
    with pytest.raises(claims.SprintClaimError, match="canonical"):
        claims.acquire_claim(canonical_sprint / "claim.json", payload())
    capability = claims.acquire_claim(expected, payload())
    assert capability._path == expected


def test_existing_canonical_claim_hard_refuses(canonical_sprint: Path) -> None:
    claims.acquire_claim(claim_path(), payload())
    with pytest.raises(claims.SprintClaimError, match="already exists"):
        claims.acquire_claim(claim_path(), payload())


@pytest.mark.parametrize("field", ["seed", "config_sha256", "selection_manifest", "selection_manifest_sha256", "episode_namespace"])
def test_claim_requires_each_provenance_field(canonical_sprint: Path, field: str) -> None:
    body = payload()
    body.pop(field)
    with pytest.raises(claims.SprintClaimError, match="schema"):
        claims.acquire_claim(claim_path(), body)


def test_capability_must_be_registry_issued_and_claim_cannot_be_tampered(canonical_sprint: Path) -> None:
    with pytest.raises(claims.SprintClaimError, match="unconsumed"):
        claims.consume_claim_and_load_split(claims.ClaimCapability(claim_path()), claims.CANONICAL_SPLIT_PATH)
    claim = claim_path()
    capability = claims.acquire_claim(claim, payload())
    body = json.loads(claim.read_text())
    body["seed"] = 8
    claim.write_text(json.dumps(body))
    with pytest.raises(claims.SprintClaimError, match="modified"):
        claims.consume_claim_and_load_split(capability, claims.CANONICAL_SPLIT_PATH)


def test_duplicate_claim_json_keys_are_rejected(canonical_sprint: Path) -> None:
    claim = claim_path()
    capability = claims.acquire_claim(claim, payload())
    claim.write_text('{"seed":7,"seed":8}')
    with pytest.raises(claims.SprintClaimError, match="unreadable or modified"):
        claims.consume_claim_and_load_split(capability, claims.CANONICAL_SPLIT_PATH)


def test_consumer_rejects_split_symlink_alias_and_second_consumption(canonical_sprint: Path) -> None:
    claim = claim_path()
    capability = claims.acquire_claim(claim, payload())
    alias = canonical_sprint / "split-alias.json"
    alias.symlink_to(claims.CANONICAL_SPLIT_PATH)
    with pytest.raises(claims.SprintClaimError, match="canonical non-symlink"):
        claims.consume_claim_and_load_split(capability, alias)
    with pytest.raises(claims.SprintClaimError, match="unconsumed"):
        claims.consume_claim_and_load_split(capability, claims.CANONICAL_SPLIT_PATH)


def test_reevaluation_claim_binds_receipt_and_is_single_use(canonical_sprint: Path) -> None:
    receipt = canonical_sprint / "receipt.json"
    legacy_digest = "a" * 64
    receipt.write_text(json.dumps({"schema_version": 1, "legacy_claim_sha256": legacy_digest, "run_tag": "synthetic", "decision": "allow_reevaluation", "decided_by": "reviewer", "decided_at": "2026-01-01T00:00:00Z"}))
    _, receipt_digest = claims.parse_disposition_receipt(receipt, legacy_claim_sha256=legacy_digest, run_tag="synthetic")
    body = payload(generation="reeval")
    body["disposition_receipt_sha256"] = receipt_digest
    claim = claim_path(generation="reeval")
    capability = claims.acquire_claim(claim, body)
    assert json.loads(claim.read_text())["disposition_receipt_sha256"] == receipt_digest
    claims.consume_claim_and_load_split(capability, claims.CANONICAL_SPLIT_PATH)
    with pytest.raises(claims.SprintClaimError, match="unconsumed"):
        claims.consume_claim_and_load_split(capability, claims.CANONICAL_SPLIT_PATH)
    with pytest.raises(claims.SprintClaimError, match="identity"):
        claims.parse_disposition_receipt(receipt, legacy_claim_sha256="f" * 64, run_tag="synthetic")


def test_probe_manifest_rejects_symlink_alias(canonical_sprint: Path) -> None:
    manifest = canonical_sprint / "manifest.json"
    probe = canonical_sprint / "probe.h5"
    probe.write_bytes(b"first")
    alias = canonical_sprint / "probe-alias.h5"
    alias.symlink_to(probe)
    first = claims.probe_manifest_register(manifest, probe, {"production_goal": "G-EV"})
    with pytest.raises(claims.SprintClaimError, match="symlink"):
        claims.probe_manifest_register(manifest, alias, {"production_goal": "G-EV"})
    assert len(first["files"]) == 1
    probe.write_bytes(b"reopened")
    with pytest.raises(claims.SprintClaimError, match="immutable"):
        claims.probe_manifest_register(manifest, probe, {"production_goal": "G-EV"})
    # A legacy relative spelling must not create a second registration for the
    # same canonical file.
    legacy_manifest = canonical_sprint / "legacy-manifest.json"
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(canonical_sprint)
    try:
        legacy_manifest.write_text(json.dumps({
            "schema_version": 1,
            "files": {claims.sha256_file(probe): {"path": "probe.h5", "sha256": claims.sha256_file(probe), "size": probe.stat().st_size, "production_goal": "G-EV"}},
        }))
        with pytest.raises(claims.SprintClaimError, match="immutable"):
            claims.probe_manifest_register(legacy_manifest, probe.resolve(), {"production_goal": "G-EV"})
    finally:
        monkeypatch.undo()


def test_probe_manifest_content_address_and_parallel_registration(canonical_sprint: Path) -> None:
    manifest = canonical_sprint / "manifest.json"
    probes = []
    for index in range(16):
        probe = canonical_sprint / f"p{index}.h5"
        probe.write_bytes(str(index).encode())
        probes.append(probe)
    threads = [threading.Thread(target=claims.probe_manifest_register, args=(manifest, probe, {"production_goal": "G-EV"})) for probe in probes]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(json.loads(manifest.read_text())["files"]) == 16


def test_metric_lock_has_strict_schema_and_bb_only_bypasses(canonical_sprint: Path) -> None:
    lock = canonical_sprint / "lock.json"
    valid = {"schema_version": 1, "endpoint": "success_rate", "aggregate": 0.5, "created_at": "now", "bb_claim_sha256": [f"{index:064x}" for index in range(8)], "primitive_version": "v1"}
    lock.write_text(json.dumps(valid))
    claims.require_metric_lock(lock, "v1")
    valid["extra"] = True
    lock.write_text(json.dumps(valid))
    with pytest.raises(claims.SprintClaimError, match="schema"):
        claims.require_metric_lock(lock, "v1")
    claims.require_metric_lock(None, "bb")
    with pytest.raises(claims.SprintClaimError, match="requires"):
        claims.require_metric_lock(None, "matched")


def test_audit_requires_complete_canonical_result_schema(canonical_sprint: Path) -> None:
    claim = claim_path()
    claims.acquire_claim(claim, payload())
    before = claim.read_bytes()
    rows = claims.audit_claims(claim.parent)
    assert rows == [{"schema_version": 1, "status": "needs_human_disposition", "claim": str(claim), "claim_sha256": hashlib.sha256(before).hexdigest(), "run_tag": "synthetic", "arm": "bb", "re_evaluation_permitted": False}]
    result = claim.parent / "p1_bb_sprint_heldout_synthetic.json"
    result.write_text(json.dumps({"run_tag": "synthetic", "arm": "bb", "claim_sha256": hashlib.sha256(before).hexdigest()}))
    assert len(claims.audit_claims(claim.parent)) == 1
    def episode() -> dict[str, object]:
        return {
            "episode_id": 1, "goal_id": "goal-1", "goal_label": "goal-1",
            "init_template": "straight", "success": True, "steps": 1,
            "return": 1.0, "discounted_return": 1.0, "final_d": 0.1,
            "d_at_done": 0.1, "d_at_done_fallback": False, "d_steps": [0.1],
            "min_d": 0.1, "d_initial": 0.2, "d_shape_initial": 0.2,
            "d_shape_at_done": 0.1, "q_first": None, "eval_wall_guard": False,
            "discard_exposure": 0,
        }
    result.write_text(json.dumps({
        "generated_at": "2026-01-01T00:00:00Z", "run_tag": "synthetic", "arm": "bb",
        "seed": 7, "config_sha256": "d" * 64, "ckpt_sha256": "c" * 64,
        "split_sha256": claims.CANONICAL_SPLIT_SHA256,
        "claim_sha256": hashlib.sha256(before).hexdigest(),
        "selection_manifest": "/synthetic/selection.json",
        "selection_manifest_sha256": "e" * 64,
        "summary": {
            "n_episodes": 200, "success_rate": 1.0, "mean_return": 1.0,
            "mean_final_d": 0.1, "mean_d_at_done": 0.1, "mean_min_d": 0.1,
            "per_template_success": {}, "per_template_episodes": {},
        },
        "episodes": [episode() for _ in range(200)],
    }))
    assert claims.audit_claims(claim.parent) == []
    assert claim.read_bytes() == before
    body = json.loads(result.read_text())
    body["episodes"] = [{}] * 200
    result.write_text(json.dumps(body))
    assert len(claims.audit_claims(claim.parent)) == 1
    body = json.loads(result.read_text())
    body["episodes"] = [episode() for _ in range(200)]
    body["seed"] = 8
    result.write_text(json.dumps(body))
    assert len(claims.audit_claims(claim.parent)) == 1
    assert claim.read_bytes() == before
