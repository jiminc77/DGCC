#!/usr/bin/env python3
"""Generate final V2 launch preflight evidence without GPU, training, or evaluation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

ARM_SPECS = (
    ("BB-D2", "bb-d2", "bb_d2"),
    ("V1-D2", "v1-d2", "v1_d2"),
    ("DMM", "v2-dmm", "v2_live"),
    ("D1M", "v2-d1m", "v2_live"),
    ("D11", "v2-d11", "v2_live"),
)
APPROVED_RUNTIME_COMMIT = "12befdac5cc9d2af448373de81fcce9d86768701"
EVIDENCE_BASE_COMMIT = "228d0f77bc8decbddb6bfe7c86924100fbf8b031"
FINAL_GOVERNANCE_SHA256 = "87053627c0a01f07b158f481b8308e19320b04d95ac9dbbf5cbb3aa2da817408"
ISOLATED_REPO_ROOT = Path("/home/simx2204/v2_research/impl/DGCC")
TRAINING_SANDBOX_ROOT = Path(
    "/home/simx2204/v2_research/runtime/DGCC-v2-12befdac"
)
FORBIDDEN_LIVE_ROOT = Path("/home/simx2204/Workspaces/DGCC")
SHA256 = frozenset("0123456789abcdef")


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8") + b"\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - SHA256:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def exclusive_bytes(path: Path, payload: bytes, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def exclusive_json(path: Path, value: Any) -> None:
    exclusive_bytes(path, canonical_json(value))


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_host_paths(root: Path, values: list[str], label: str) -> list[Path]:
    if not values:
        raise ValueError(f"at least one {label} path is required")
    resolved: list[Path] = []
    host_root = root.resolve(strict=True)
    for value in values:
        path = Path(value).expanduser()
        candidate = path if path.is_absolute() else host_root / path
        candidate = candidate.resolve(strict=False)
        try:
            candidate.relative_to(host_root)
        except ValueError as error:
            raise ValueError(f"{label} path escapes the training host root") from error
        resolved.append(candidate)
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"duplicate {label} path")
    return resolved


def validate_not_admitted_disposition(
    path: Path,
    *,
    code_manifest_sha256: str,
    config_sha256: str,
    guard_sha256: str,
) -> dict[str, Any]:
    disposition = json.loads(path.read_bytes())
    required_keys = {
        "schema_version",
        "status",
        "tournament_scope",
        "decision_authority",
        "decision_at_utc",
        "reason",
        "future_status",
        "schedule",
        "r5_rank_calibration_executed",
        "gpu_latency_gate_executed",
        "charter_amendment_2_url",
        "code_manifest_sha256",
        "config_sha256",
        "neff_guard_sha256",
        "irreversible_for_tournament",
        "disposition_sha256",
    }
    expected_reason = (
        "BGT admission requires an R5 rank calibration whose input provenance "
        "(checkpoint and panel paths with SHA-256, transitions, state-row "
        "selection and cardinality, rank-input producer) was never preregistered "
        "in the V2-DEV charter. Retroactively selecting those assets immediately "
        "before the first production run would constitute arbitrary evidence "
        "selection. BGT is therefore not admitted to this tournament."
    )
    expected_future_status = (
        "BGT remains a post-winner exploratory extension candidate. Its code is "
        "retained in an inactive state. Any future admission requires a separately "
        "preregistered R5 protocol (assets, transitions, state selection, and "
        "producer pinned before execution) and the synchronized GPU latency gate."
    )
    expected_schedule = {
        "planned_runs": 15,
        "redistribute_bgt_runs": False,
        "seed_block_interleaving": True,
        "arms": {
            "BB-D2": [0, 1, 2],
            "V1-D2": [0, 1, 2],
            "DMM": [0, 1, 2],
            "D1M": [0, 1, 2],
            "D11": [0, 1, 2],
        },
    }
    if not isinstance(disposition, dict) or set(disposition) != required_keys:
        raise ValueError("BGT not-admitted disposition has invalid schema")
    body = {
        key: value
        for key, value in disposition.items()
        if key != "disposition_sha256"
    }
    if (
        disposition["schema_version"] != 1
        or disposition["status"] != "not-admitted"
        or disposition["tournament_scope"] != "V2-DEV discovery tournament"
        or disposition["decision_authority"] != "owner via orchestrator"
        or not isinstance(disposition["decision_at_utc"], str)
        or not disposition["decision_at_utc"].endswith("Z")
        or disposition["reason"] != expected_reason
        or disposition["future_status"] != expected_future_status
        or disposition["schedule"] != expected_schedule
        or disposition["r5_rank_calibration_executed"] is not False
        or disposition["gpu_latency_gate_executed"] is not False
        or disposition["charter_amendment_2_url"]
        != "https://github.com/jiminc77/research-dashboard/issues/44#issuecomment-5079492158"
        or disposition["code_manifest_sha256"] != code_manifest_sha256
        or disposition["config_sha256"] != config_sha256
        or disposition["neff_guard_sha256"] != guard_sha256
        or disposition["irreversible_for_tournament"] is not True
        or disposition["disposition_sha256"] != sha256_bytes(canonical_json(body))
    ):
        raise ValueError(
            "BGT not-admitted disposition does not bind the final 15-run release"
        )
    return disposition


def validate_final_governance(
    governance: dict[str, Any],
    *,
    code_manifest_sha256: str,
    guard_sha256: str,
    disposition_artifact_sha256: str,
    protocol_governance_sha256: str,
    authoritative_r3_r4_sha256: str,
) -> None:
    required_keys = {
        "schema_version",
        "purpose",
        "runtime_source_commit",
        "evidence_base_commit",
        "binding_beta",
        "v2_launch_code_manifest_sha256",
        "neff_guard_sha256",
        "bgt_admitted_manifest_sha256",
        "bgt_not_admitted_artifact_sha256",
        "protocol_governance_sha256",
        "authoritative_r3_r4_receipt_sha256",
        "original_worktree_head_sha256",
        "original_config_sha256",
        "seeds",
        "arm_seeds",
        "algo_abort",
        "pocket_kill",
        "tournament_schedule",
    }
    expected_arms = {name: [0, 1, 2] for name, _, _ in ARM_SPECS}
    expected_schedule = {
        "status": "final_not_admitted",
        "amendment_url": (
            "https://github.com/jiminc77/research-dashboard/issues/44"
            "#issuecomment-5079492158"
        ),
        "planned_runs": 15,
        "arms": expected_arms,
        "redistribute_runs": False,
        "seed_block_interleaving": True,
    }
    if not isinstance(governance, dict) or set(governance) != required_keys:
        raise ValueError("final execution governance has invalid schema")
    if (
        governance["schema_version"] != 3
        or governance["purpose"]
        != "Final owner-pinned V2 launch governance. BGT is irreversibly not admitted to this tournament; only the 15 non-BGT cells are authorized and no removed run is redistributed."
        or governance["runtime_source_commit"] != APPROVED_RUNTIME_COMMIT
        or governance["evidence_base_commit"] != EVIDENCE_BASE_COMMIT
        or governance["binding_beta"] != 0.015363
        or governance["v2_launch_code_manifest_sha256"] != code_manifest_sha256
        or governance["neff_guard_sha256"] != guard_sha256
        or governance["bgt_admitted_manifest_sha256"] is not None
        or governance["bgt_not_admitted_artifact_sha256"]
        != disposition_artifact_sha256
        or governance["protocol_governance_sha256"] != protocol_governance_sha256
        or governance["authoritative_r3_r4_receipt_sha256"]
        != authoritative_r3_r4_sha256
        or governance["original_worktree_head_sha256"] is not None
        or governance["original_config_sha256"] is not None
        or governance["seeds"]
        != {
            "discovery": [0, 1, 2],
            "backup": [3, 4],
            "confirmatory": [8, 9, 10],
            "confirmatory_backup": [11, 12, 13],
        }
        or governance["arm_seeds"] != expected_arms
        or governance["algo_abort"]
        != {
            "primary_paired_metric": "excluded",
            "retry": False,
            "worst_seed_guard": "worst_equivalent",
        }
        or governance["pocket_kill"]
        != {
            "after_hours": 3,
            "disposition": "algo_abort",
            "progress": 0,
            "restart": False,
        }
        or governance["tournament_schedule"] != expected_schedule
    ):
        raise ValueError("final governance does not lock the exact 15-run schedule")


def validate_authoritative_r3_r4(
    path: Path, *, expected_sha256: str, training_host: Path
) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("authoritative R3/R4 receipt does not match governance pin")
    receipt = json.loads(path.read_bytes())
    if (
        not isinstance(receipt, dict)
        or receipt.get("content_opened") is not False
        or receipt.get("training_sandbox_root") != str(training_host)
        or receipt.get("R3", {}).get("pass") is not True
        or receipt.get("R4", {}).get("pass") is not True
        or receipt.get("same_uid_limitation")
        != "Sparse runtime sandbox and application controls; not OS, mount-namespace, ACL, or kernel isolation."
    ):
        raise ValueError("authoritative R3/R4 receipt is not the accepted sparse-sandbox evidence")
    return receipt


def derive_cells(
    governance: dict[str, Any],
    not_admitted_disposition: Path,
    *,
    code_manifest_sha256: str,
    config_sha256: str,
) -> tuple[list[tuple[str, str, str, int]], dict[str, Any]]:
    final_disposition = validate_not_admitted_disposition(
        not_admitted_disposition,
        code_manifest_sha256=code_manifest_sha256,
        config_sha256=config_sha256,
        guard_sha256=governance["neff_guard_sha256"],
    )
    cells = [
        (schedule_arm, arm, mode, seed)
        for seed in (0, 1, 2)
        for schedule_arm, arm, mode in ARM_SPECS
    ]
    if len(cells) != 15 or len(set(cells)) != 15:
        raise ValueError("final schedule is not exactly 15 unique non-BGT cells")
    return cells, {
        "status": "not-admitted",
        "artifact_path": str(not_admitted_disposition.resolve(strict=True)),
        "artifact_sha256": sha256_file(not_admitted_disposition),
        "disposition_sha256": final_disposition["disposition_sha256"],
        "reason": final_disposition["reason"],
        "future_status": final_disposition["future_status"],
        "redistributed": False,
    }


def launch_manifest(
    schedule_arm: str,
    arm: str,
    mode: str,
    seed: int,
    *,
    config_sha256: str,
    code_manifest_sha256: str,
    guard_sha256: str,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "arm": arm,
        "mode": mode,
        "policy_delay": 2,
        "binding_beta": 0.015363,
        "code_path": "v2",
        "config_lineage": "v2",
        "merged_final_code": True,
        "code_manifest_sha256": code_manifest_sha256,
        "neff_guard_sha256": guard_sha256,
        "seed": seed,
        "schedule_arm": schedule_arm,
        "config_sha256": config_sha256,
    }
    if arm in {"bb-d2", "v1-d2"}:
        value["d2_lineage"] = arm
    return value


def build_asset_manifest(assets: list[tuple[Path, str]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "assets": [
            {
                "path": str(path.resolve(strict=True)),
                "sha256": sha256_file(path),
                "role": role,
            }
            for path, role in assets
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--training-host-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--governance", type=Path, required=True)
    parser.add_argument("--protocol-governance", type=Path, required=True)
    parser.add_argument("--authoritative-r3-r4-receipt", type=Path, required=True)
    parser.add_argument("--code-manifest", type=Path, required=True)
    parser.add_argument("--neff-guard", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bgt-not-admitted", type=Path, required=True)
    parser.add_argument("--protected-path", action="append", default=[])
    parser.add_argument("--fresh-heldout-path", action="append", default=[])
    args = parser.parse_args(argv)

    repo = args.repo_root.resolve(strict=True)
    expected_repo = ISOLATED_REPO_ROOT.resolve(strict=True)
    live_root = FORBIDDEN_LIVE_ROOT.resolve(strict=True)
    training_host = args.training_host_root.resolve(strict=True)
    expected_training_host = TRAINING_SANDBOX_ROOT.resolve(strict=True)
    target = args.output.absolute()
    release_root = (repo / "release" / "v2").resolve(strict=True)
    if repo != expected_repo or repo == live_root:
        raise ValueError("repo root is not the owner-pinned isolated V2 worktree")
    if training_host != expected_training_host:
        raise ValueError("training host root is not the accepted sparse V2 sandbox")
    if target.parent.resolve(strict=True) != release_root:
        raise ValueError("output must be created directly under the isolated release root")
    if target.name != "preflight_15_not_admitted":
        raise ValueError("authoritative output name is fixed by the final disposition")
    try:
        target.relative_to(live_root)
    except ValueError:
        isolated_root_validated = True
    else:
        raise ValueError("output path enters the forbidden live tree")
    if target.exists():
        raise FileExistsError(f"authoritative output already exists: {target}")
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_commit != EVIDENCE_BASE_COMMIT:
        raise ValueError("isolated release tree is not the pinned evidence-base commit")

    sys.path.insert(0, str(repo / "src"))
    preflight = load_module(
        repo / "scripts/v2_protocol_preflight.py", "v2_release_preflight"
    )
    from dgcc.logging.asset_firewall import (
        generate_r3_r4_existence_receipt,
        load_launch_asset_manifest,
        persist_launch_receipts,
    )
    from dgcc.logging.attempt_registry import AttemptRegistry

    governance_bytes = args.governance.read_bytes()
    if sha256_bytes(governance_bytes) != FINAL_GOVERNANCE_SHA256:
        raise ValueError("final governance bytes do not match the release root-of-trust pin")
    governance = json.loads(governance_bytes)
    protocol_governance_bytes = args.protocol_governance.read_bytes()
    protocol_governance = json.loads(protocol_governance_bytes)
    code_manifest_bytes = args.code_manifest.read_bytes()
    guard_bytes = args.neff_guard.read_bytes()
    config_bytes = args.config.read_bytes()
    code_manifest_sha256 = sha256_bytes(code_manifest_bytes)
    config_sha256 = sha256_bytes(config_bytes)
    guard_sha256 = sha256_bytes(guard_bytes)
    disposition_artifact_sha256 = sha256_file(args.bgt_not_admitted)
    protocol_governance_sha256 = sha256_bytes(protocol_governance_bytes)
    authoritative_r3_r4_sha256 = sha256_file(args.authoritative_r3_r4_receipt)
    validate_final_governance(
        governance,
        code_manifest_sha256=code_manifest_sha256,
        guard_sha256=guard_sha256,
        disposition_artifact_sha256=disposition_artifact_sha256,
        protocol_governance_sha256=protocol_governance_sha256,
        authoritative_r3_r4_sha256=authoritative_r3_r4_sha256,
    )
    preflight._validate_governance(protocol_governance)
    authoritative_r3_r4 = validate_authoritative_r3_r4(
        args.authoritative_r3_r4_receipt,
        expected_sha256=authoritative_r3_r4_sha256,
        training_host=training_host,
    )
    code_receipt = preflight.validate_code_manifest_bytes(
        code_manifest_bytes, runtime_root=repo
    )
    if (
        code_receipt["code_manifest_sha256"]
        != governance["v2_launch_code_manifest_sha256"]
    ):
        raise ValueError("governance does not pin current code-manifest bytes")
    if guard_sha256 != governance["neff_guard_sha256"]:
        raise ValueError("governance does not pin the supplied canonical N_eff guard")

    cells, disposition_evidence = derive_cells(
        governance,
        args.bgt_not_admitted,
        code_manifest_sha256=code_manifest_sha256,
        config_sha256=config_sha256,
    )
    protected_paths = resolve_host_paths(
        training_host, args.protected_path, "protected"
    )
    fresh_paths = resolve_host_paths(
        training_host, args.fresh_heldout_path, "fresh-heldout"
    )

    stage = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    stage.mkdir(mode=0o700)
    published = False
    try:
        r3_r4 = generate_r3_r4_existence_receipt(protected_paths, fresh_paths)
        r3_r4.update(
            {
                "hostname": socket.gethostname(),
                "argv_sha256": sha256_bytes(canonical_json(sys.argv)),
                "training_host_root": str(training_host),
                "content_opened": False,
            }
        )
        r3_r4_path = stage / "r3_r4_existence_receipt.json"
        exclusive_json(r3_r4_path, r3_r4)
        cell_receipts: list[dict[str, Any]] = []
        first_preflight: dict[str, Any] | None = None
        for ordinal, (schedule_arm, arm, mode, seed) in enumerate(cells, start=1):
            cell_id = f"{ordinal:02d}-{schedule_arm.lower()}-s{seed}"
            cell_dir = stage / "cells" / cell_id
            cell_dir.mkdir(parents=True)
            manifest = launch_manifest(
                schedule_arm,
                arm,
                mode,
                seed,
                config_sha256=config_sha256,
                code_manifest_sha256=code_manifest_sha256,
                guard_sha256=guard_sha256,
            )
            manifest_path = cell_dir / "launch_manifest.json"
            exclusive_json(manifest_path, manifest)
            assets: list[tuple[Path, str]] = [
                (manifest_path, "preflight_manifest"),
                (args.governance, "final_execution_governance"),
                (args.protocol_governance, "protocol_governance"),
                (args.bgt_not_admitted, "bgt_not_admitted_disposition"),
                (
                    args.authoritative_r3_r4_receipt,
                    "authoritative_r3_r4_receipt",
                ),
                (args.code_manifest, "code_manifest"),
                (args.neff_guard, "neff_guard"),
                (args.config, "config"),
            ]
            asset_manifest_path = cell_dir / "asset_manifest.json"
            exclusive_json(asset_manifest_path, build_asset_manifest(assets))
            asset_manifest_sha = sha256_file(asset_manifest_path)
            audit_path = cell_dir / "protected_access_audit.jsonl"
            firewall, asset_document = load_launch_asset_manifest(
                asset_manifest_path, asset_manifest_sha, audit_path
            )
            bundle = persist_launch_receipts(cell_dir, asset_document, firewall)
            manifest_bytes = manifest_path.read_bytes()
            receipt = preflight.validate_manifest_bytes(
                manifest_bytes,
                protocol_governance_bytes,
                expected_arm=arm,
                expected_seed=seed,
                expected_config_sha256=config_sha256,
                expected_code_manifest_sha256=code_manifest_sha256,
                code_manifest_bytes=code_manifest_bytes,
                neff_guard_bytes=guard_bytes,
                runtime_root=repo,
            )
            receipt_path = cell_dir / "protocol_preflight_receipt.json"
            exclusive_json(receipt_path, receipt)
            if first_preflight is None:
                first_preflight = receipt
            cell_receipts.append(
                {
                    "ordinal": ordinal,
                    "cell_id": cell_id,
                    "schedule_arm": schedule_arm,
                    "arm": arm,
                    "seed": seed,
                    "policy_delay": receipt["policy_delay"],
                    "beta": 0.015363,
                    "config_sha256": receipt["config_sha256"],
                    "code_manifest_sha256": receipt["code_manifest_sha256"],
                    "neff_guard_sha256": receipt["neff_guard_sha256"],
                    "admitted_manifest_sha256": receipt["admitted_manifest_sha256"],
                    "launch_manifest_sha256": sha256_file(manifest_path),
                    "asset_manifest_sha256": asset_manifest_sha,
                    "preflight_receipt_sha256": sha256_file(receipt_path),
                    "r1_sha256": bundle["r1_sha256"],
                    "r2_sha256": bundle["r2_sha256"],
                    "pass": True,
                }
            )

        if first_preflight is None:
            raise RuntimeError("final schedule contains no cells")
        smoke_root = stage / "registry-smoke"
        registry = AttemptRegistry(
            smoke_root / "attempts",
            run_tag="v2-round8-no-training-preflight-smoke",
            config={"purpose": "no-training-preflight-smoke"},
            code_sha256=code_manifest_sha256,
            seed=first_preflight["seed"],
            terminal_anchor_directory=smoke_root / "terminal-anchors",
            governed_launch_receipt=first_preflight,
        )
        registry.initialized(sha256_bytes(b"no-training-registry-smoke"))
        if not registry.finalize_once(
            "SUCCEEDED",
            exit_code=0,
            detail="PREPARING -> INITIALIZED -> TERMINAL smoke only; no agent, GPU, training, or eval",
        ):
            raise RuntimeError("registry smoke did not elect a terminal record")
        anchors = sorted((smoke_root / "terminal-anchors").glob("*.json"))
        if len(anchors) != 1 or not AttemptRegistry.verify_terminal_anchor(
            registry.attempt_path, anchors[0]
        ):
            raise RuntimeError("registry smoke anchor verification failed")
        phases = [
            record["phase"]
            for record in AttemptRegistry.read_records(registry.attempt_path)
        ]
        if phases != ["PREPARING", "INITIALIZED", "TERMINAL"]:
            raise RuntimeError("registry smoke lifecycle is incomplete")

        generated_records = {
            section: [
                (record["path"], record["exists"])
                for record in r3_r4[section]["records"]
            ]
            for section in ("R3", "R4")
        }
        authoritative_records = {
            section: [
                (record["path"], record["exists"])
                for record in authoritative_r3_r4[section]["records"]
            ]
            for section in ("R3", "R4")
        }
        if generated_records != authoritative_records:
            raise ValueError(
                "generated R3/R4 footprint differs from the authoritative receipt"
            )
        matrix = {
            "schema_version": 1,
            "runtime_source_commit": APPROVED_RUNTIME_COMMIT,
            "evidence_base_commit": current_commit,
            "schedule_disposition": disposition_evidence,
            "planned_runs": len(cells),
            "allowed_run_counts": [15],
            "fourteen_run_plan_present": False,
            "redistributed": False,
            "code_manifest": code_receipt,
            "governance_sha256": sha256_bytes(governance_bytes),
            "protocol_governance_sha256": protocol_governance_sha256,
            "config_sha256": config_sha256,
            "neff_guard_sha256": guard_sha256,
            "authoritative_r3_r4_receipt_sha256": authoritative_r3_r4_sha256,
            "generated_r3_r4_receipt_sha256": sha256_file(r3_r4_path),
            "r3_pass": (
                authoritative_r3_r4["R3"]["pass"] and r3_r4["R3"]["pass"]
            ),
            "r4_pass": (
                authoritative_r3_r4["R4"]["pass"] and r3_r4["R4"]["pass"]
            ),
            "registry_smoke": {
                "attempt_id": registry.attempt_id,
                "phases": phases,
                "records_sha256": sha256_file(
                    registry.attempt_path / "records.jsonl"
                ),
                "terminal_anchor_sha256": sha256_file(anchors[0]),
                "verified": True,
            },
            "cells": cell_receipts,
            "constraints": {
                "gpu_used": False,
                "training_run": False,
                "eval_run": False,
                "protected_content_opened": False,
                "isolated_release_root_validated": isolated_root_validated,
                "live_tree_mutated": not isolated_root_validated,
            },
        }
        exclusive_json(stage / "preflight_matrix.json", matrix)
        fsync_dir(stage)
        os.rename(stage, target)
        fsync_dir(target.parent)
        published = True
    finally:
        if not published and stage.exists():
            shutil.rmtree(stage)
    print(canonical_json({"output": str(target), "planned_runs": len(cells)}).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
