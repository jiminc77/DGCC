#!/usr/bin/env python3
"""Verify the G9b held-out evidence chain without exposing endpoint values."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
ISSUED_LOCK_SHA256 = "7cb96288c9b27290674488c7ae34c854efe82f0e04628af26b6e93166a562122"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("JSON root is not an object")
    return value


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp is not a string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp is timezone-naive")
    return parsed


def _check(name: str, action: Any) -> tuple[dict[str, str], Any | None]:
    try:
        return {"status": "PASS"}, action()
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return {"status": "FAIL", "reason": str(exc)}, None


def verify(repo_root: Path = REPO) -> dict[str, Any]:
    metrics = repo_root / "outputs" / "metrics"
    lock_path = metrics / "sprint_metric.lock"
    context: dict[str, Any] = {}

    def verify_lock() -> None:
        if not lock_path.is_file():
            raise ValueError("canonical lock is missing")
        if _sha256(lock_path) != ISSUED_LOCK_SHA256:
            raise ValueError("lock does not match issued trust anchor")
        lock = _load(lock_path)
        context["lock"] = lock
        context["created_at"] = _timestamp(lock["created_at"])

    lock_status, _ = _check("lock", verify_lock)

    def verify_bb_claims() -> None:
        lock = context["lock"]
        expected = lock["bb_claim_sha256"]
        audit = lock["bb_claim_audit"]
        if not isinstance(expected, list) or len(expected) != 7:
            raise ValueError("lock must contain exactly seven BB claim hashes")
        if not isinstance(audit, list) or len(audit) != 7:
            raise ValueError("lock must contain exactly seven BB claim audit rows")
        actual: list[str] = []
        bundle: dict[str, Any] | None = None
        for row in audit:
            seed = row["seed"]
            if row["kind"] == "legacy_bundle":
                if bundle is None:
                    bundle = _load(metrics / "sprint_retro_audit_bundle.json")
                claim_path = metrics / f"p1_sprint_heldout_claim_m4_t2_s{seed}.json"
            elif row["kind"] == "canonical":
                claim_path = metrics / f"p1_bb_sprint_heldout_sprint_t2_bb_s{seed}_claim.json"
            else:
                raise ValueError("unknown BB claim audit kind")
            if not claim_path.is_file():
                raise ValueError("audited BB claim is missing")
            digest = _sha256(claim_path)
            if row["claim_sha256"] != digest:
                raise ValueError("BB audit claim hash mismatch")
            if bundle is not None and row["kind"] == "legacy_bundle":
                entry = bundle["files"][str(claim_path.relative_to(repo_root))]
                if entry["sha256"] != digest:
                    raise ValueError("legacy bundle claim hash mismatch")
            actual.append(digest)
        if actual != expected:
            raise ValueError("BB claim hashes do not match lock")

    bb_status, _ = _check("bb_claim_hashes", verify_bb_claims)
    v1_claims = sorted(metrics.glob("p1_v1_sprint_heldout_*_claim.json"))

    def verify_times() -> None:
        created = context["created_at"]
        for path in v1_claims:
            if _timestamp(_load(path)["timestamp"]) <= created:
                raise ValueError("V1 claim does not postdate lock")

    time_status, _ = _check("v1_claim_timestamps", verify_times)

    def verify_results() -> None:
        for claim_path in v1_claims:
            result_path = claim_path.with_name(claim_path.name.replace("_claim.json", ".json"))
            if not result_path.is_file():
                raise ValueError("V1 result is missing")
            if _load(result_path)["claim_sha256"] != _sha256(claim_path):
                raise ValueError("V1 result claim digest mismatch")

    result_status, _ = _check("v1_result_claim_links", verify_results)

    def verify_probes() -> None:
        entries = _load(metrics / "sprint_probe_manifest.json")["files"]
        if not isinstance(entries, dict):
            raise ValueError("probe manifest files is not an object")
        registered = {entry.get("run_tag") for entry in entries.values() if isinstance(entry, dict)}
        for claim_path in v1_claims:
            run_tag = _load(claim_path)["run_tag"]
            if run_tag not in registered:
                raise ValueError("V1 run is absent from probe manifest")

    probe_status, _ = _check("v1_probe_manifest", verify_probes)

    def verify_v1_set() -> str:
        count = len(v1_claims)
        if count < 5:
            raise ValueError("fewer than five V1 claims")
        return "incomplete_v1_set" if count < 7 else "complete_v1_set"

    set_status, set_state = _check("amd5_v1_set", verify_v1_set)
    checks = {
        "lock": lock_status,
        "bb_claim_hashes": bb_status,
        "v1_claim_timestamps": time_status,
        "v1_result_claim_links": result_status,
        "v1_probe_manifest": probe_status,
        "amd5_v1_set": set_status,
    }
    valid = all(value["status"] == "PASS" for value in checks.values())
    verdict = set_state if valid else "failed"
    return {"verdict": verdict, "checks": checks}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit machine-readable report")
    args = parser.parse_args(argv)
    report = verify()
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(report["verdict"])
    return 0 if report["verdict"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
