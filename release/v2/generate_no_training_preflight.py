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
    ("BGT", "v2-bgt", "v2_live"),
)
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
        candidate = candidate.absolute()
        try:
            candidate.relative_to(host_root)
        except ValueError as error:
            raise ValueError(f"{label} path escapes the training host root") from error
        resolved.append(candidate)
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"duplicate {label} path")
    return resolved


def validate_not_admitted_state(
    path: Path,
    *,
    code_manifest_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    state = json.loads(path.read_bytes())
    if not isinstance(state, dict) or set(state) != {
        "schema_version",
        "status",
        "identities",
        "state_sha256",
    }:
        raise ValueError("BGT cutoff state has invalid schema")
    body = {key: value for key, value in state.items() if key != "state_sha256"}
    expected_state_sha = sha256_bytes(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    identities = state.get("identities")
    if (
        state.get("schema_version") != 1
        or state.get("status") != "not-admitted"
        or state.get("state_sha256") != expected_state_sha
        or not isinstance(identities, dict)
        or set(identities)
        != {
            "manifest_sha256",
            "code_final_sha256",
            "checkpoint_sha256",
            "panel_sha256",
            "config_sha256",
        }
        or identities.get("code_final_sha256") != code_manifest_sha256
        or identities.get("config_sha256") != config_sha256
        or any(not require_digest(value, key) for key, value in identities.items())
    ):
        raise ValueError("BGT cutoff state does not bind the final release identities")
    return state


def derive_cells(
    governance: dict[str, Any],
    disposition: str,
    *,
    admitted_manifest: Path | None,
    cutoff_state: Path | None,
    code_manifest_sha256: str,
    config_sha256: str,
) -> tuple[list[tuple[str, str, str, int]], dict[str, Any]]:
    schedule = governance.get("tournament_schedule")
    if not isinstance(schedule, dict):
        raise ValueError("governance lacks tournament schedule")
    if (
        schedule.get("planned_runs") != 18
        or schedule.get("bgt_not_admitted")
        != {"runs": 15, "redistribute_runs": False}
    ):
        raise ValueError("governance must permit exactly 18 or 15 runs; never 14")
    arm_map = schedule.get("arms")
    expected = {name: [0, 1, 2] for name, _, _ in ARM_SPECS}
    if arm_map != expected:
        raise ValueError("governance schedule cells do not match the balanced seed lock")

    bgt_pin = governance.get("bgt_admitted_manifest_sha256")
    if disposition == "admitted":
        pin = require_digest(bgt_pin, "BGT admission governance pin")
        if admitted_manifest is None or cutoff_state is not None:
            raise ValueError("admitted disposition requires exactly one BGT manifest")
        if sha256_file(admitted_manifest) != pin:
            raise ValueError("BGT admission manifest does not match governance pin")
        included = ARM_SPECS
        disposition_evidence = {
            "status": "admitted",
            "manifest_path": str(admitted_manifest.resolve(strict=True)),
            "manifest_sha256": pin,
        }
    else:
        if bgt_pin is not None:
            raise ValueError("not-admitted disposition requires a null admission pin")
        if cutoff_state is None or admitted_manifest is not None:
            raise ValueError("not-admitted disposition requires exactly one cutoff state")
        state = validate_not_admitted_state(
            cutoff_state,
            code_manifest_sha256=code_manifest_sha256,
            config_sha256=config_sha256,
        )
        included = ARM_SPECS[:-1]
        disposition_evidence = {
            "status": "not-admitted",
            "cutoff_state_path": str(cutoff_state.resolve(strict=True)),
            "cutoff_state_sha256": sha256_file(cutoff_state),
            "cutoff_state_digest": state["state_sha256"],
            "redistributed": False,
        }

    cells = [
        (schedule_arm, arm, mode, seed)
        for seed in (0, 1, 2)
        for schedule_arm, arm, mode in included
    ]
    expected_count = 18 if disposition == "admitted" else 15
    if len(cells) != expected_count or len(set(cells)) != expected_count:
        raise ValueError("final schedule is not exactly 18 admitted or 15 not-admitted")
    return cells, disposition_evidence


def launch_manifest(
    schedule_arm: str,
    arm: str,
    mode: str,
    seed: int,
    *,
    config_sha256: str,
    code_manifest_sha256: str,
    guard_sha256: str,
    admitted_manifest_sha256: str | None,
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
    if arm == "v2-bgt":
        value["admitted_manifest_sha256"] = admitted_manifest_sha256
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
    parser.add_argument("--code-manifest", type=Path, required=True)
    parser.add_argument("--neff-guard", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bgt-disposition", choices=("admitted", "not-admitted"), required=True)
    parser.add_argument("--bgt-admission", type=Path)
    parser.add_argument("--bgt-cutoff-state", type=Path)
    parser.add_argument("--protected-path", action="append", default=[])
    parser.add_argument("--fresh-heldout-path", action="append", default=[])
    parser.add_argument(
        "--expected-source-commit",
        default="12befdac5cc9d2af448373de81fcce9d86768701",
    )
    args = parser.parse_args(argv)

    repo = args.repo_root.resolve(strict=True)
    training_host = args.training_host_root.resolve(strict=True)
    target = args.output.absolute()
    if target.exists():
        raise FileExistsError(f"authoritative output already exists: {target}")
    if target.parent.resolve(strict=True) != target.parent:
        raise ValueError("output parent must be a canonical existing directory")
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_commit != args.expected_source_commit:
        raise ValueError("isolated release tree is not the approved source commit")

    sys.path.insert(0, str(repo / "src"))
    preflight = load_module(repo / "scripts/v2_protocol_preflight.py", "v2_release_preflight")
    from dgcc.logging.asset_firewall import (
        generate_r3_r4_existence_receipt,
        load_launch_asset_manifest,
        persist_launch_receipts,
    )
    from dgcc.logging.attempt_registry import AttemptRegistry

    governance_bytes = args.governance.read_bytes()
    governance = json.loads(governance_bytes)
    code_manifest_bytes = args.code_manifest.read_bytes()
    guard_bytes = args.neff_guard.read_bytes()
    config_bytes = args.config.read_bytes()
    code_manifest_sha256 = sha256_bytes(code_manifest_bytes)
    config_sha256 = sha256_bytes(config_bytes)
    guard_sha256 = sha256_bytes(guard_bytes)
    preflight._validate_governance(governance)
    code_receipt = preflight.validate_code_manifest_bytes(
        code_manifest_bytes, runtime_root=repo
    )
    if code_receipt["code_manifest_sha256"] != governance["v2_launch_code_manifest_sha256"]:
        raise ValueError("governance does not independently pin current code-manifest bytes")
    if guard_sha256 != governance["neff_guard_sha256"]:
        raise ValueError("governance does not pin the supplied canonical N_eff guard")

    cells, disposition_evidence = derive_cells(
        governance,
        args.bgt_disposition,
        admitted_manifest=args.bgt_admission,
        cutoff_state=args.bgt_cutoff_state,
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
        admitted_sha = governance.get("bgt_admitted_manifest_sha256")
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
                admitted_manifest_sha256=admitted_sha,
            )
            manifest_path = cell_dir / "launch_manifest.json"
            exclusive_json(manifest_path, manifest)
            assets: list[tuple[Path, str]] = [
                (manifest_path, "preflight_manifest"),
                (args.governance, "execution_governance"),
                (args.code_manifest, "code_manifest"),
                (args.neff_guard, "neff_guard"),
                (args.config, "config"),
            ]
            if arm == "v2-bgt":
                if args.bgt_admission is None:
                    raise ValueError("BGT cell lacks admitted manifest")
                assets.append((args.bgt_admission, "bgt_admission_manifest"))
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
                governance_bytes,
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

        matrix = {
            "schema_version": 1,
            "source_commit": current_commit,
            "schedule_disposition": disposition_evidence,
            "planned_runs": len(cells),
            "allowed_run_counts": [15, 18],
            "fourteen_run_plan_present": False,
            "redistributed": False,
            "code_manifest": code_receipt,
            "governance_sha256": sha256_bytes(governance_bytes),
            "config_sha256": config_sha256,
            "neff_guard_sha256": guard_sha256,
            "r3_r4_receipt_sha256": sha256_file(r3_r4_path),
            "r3_pass": r3_r4["R3"]["pass"],
            "r4_pass": r3_r4["R4"]["pass"],
            "registry_smoke": {
                "attempt_id": registry.attempt_id,
                "phases": phases,
                "records_sha256": sha256_file(registry.attempt_path / "records.jsonl"),
                "terminal_anchor_sha256": sha256_file(anchors[0]),
                "verified": True,
            },
            "cells": cell_receipts,
            "constraints": {
                "gpu_used": False,
                "training_run": False,
                "eval_run": False,
                "protected_content_opened": False,
                "live_tree_mutated": False,
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
