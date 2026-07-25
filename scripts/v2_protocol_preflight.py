#!/usr/bin/env python3
"""Fail-closed launch governance for synthetic AMD-5/V2 protocol manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import math

from dgcc.logging.code_manifest import (
    _read_runtime_file,
    canonical_json,
    required_runtime_files,
    validate_code_manifest_bytes as _validate_code_manifest_bytes,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOVERNANCE = ROOT / "configs" / "v2_execution_governance.json"
DIGEST = set("0123456789abcdef")
OUTCOME_FIELDS = {"outcome", "outcomes", "result", "results", "metric", "metrics", "reward", "rewards", "evaluation", "eval", "performance", "performances", "rank", "ranking", "rankings"}
REQUIRED_RUNTIME_FILES = frozenset(required_runtime_files(ROOT))
NEFF_GUARD_SHA256 = "7a5b517ab108c8b0afa79d7e544e3f3d1eee40d5e56df6371d94a97bbcaedda5"
NEFF_GUARD_SCHEMA = {
    "schema_version": True, "artifact": True, "device": True, "shape": True,
    "beta_contact": True, "checkpoint_sha256": True, "panel_sha256": True,
    "snapshot_manifest_sha256": True, "q_score_inputs_sha256": True,
    "q1_pooled_median": True, "qmin_pooled_median": True,
    "accepted_range": True, "guard_passed": True,
}


def validate_code_manifest_bytes(
    code_manifest_bytes: bytes, *, runtime_root: Path = ROOT
) -> dict[str, Any]:
    """Compatibility wrapper around the shared runtime-closure validator."""
    return _validate_code_manifest_bytes(code_manifest_bytes, runtime_root=runtime_root)




def content_address(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(manifest)).hexdigest()


def _digest(value: Any, label: str, *, allow_none: bool = False) -> str | None:
    if allow_none and value is None:
        return None
    if not isinstance(value, str) or len(value) != 64 or set(value) - DIGEST or len(set(value)) == 1:
        raise ValueError(f"{label} must be a non-degenerate lowercase SHA-256 digest")
    return value


def _reject_unknown(value: Any, schema: dict[str, Any], label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    unknown = set(value) - set(schema)
    missing = {key for key, required in schema.items() if required} - set(value)
    if unknown or missing:
        raise ValueError(f"{label} has unknown or missing fields")


def _reject_outcomes(value: Any) -> None:
    if isinstance(value, dict):
        if set(value) & OUTCOME_FIELDS:
            raise ValueError("launch manifests must not contain outcome fields")
        for child in value.values():
            _reject_outcomes(child)
    elif isinstance(value, list):
        for child in value:
            _reject_outcomes(child)
def _load_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        document: dict[str, Any] = {}
        for key, value in pairs:
            if key in document:
                raise ValueError(f"{label} has duplicate fields")
            document[key] = value
        return document

    try:
        document = json.loads(payload, object_pairs_hook=reject_duplicates)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} JSON is malformed") from error
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be an object")
    return document


def _validate_neff_guard_bytes(
    neff_guard_bytes: bytes, manifest: dict[str, Any], governance: dict[str, Any]
) -> dict[str, Any]:
    guard_pin = _digest(governance["neff_guard_sha256"], "N_eff guard governance pin")
    manifest_pin = _digest(manifest["neff_guard_sha256"], "launch N_eff guard pin")
    if guard_pin != NEFF_GUARD_SHA256 or manifest_pin != guard_pin:
        raise ValueError("launch N_eff guard identity must match the canonical independent governance pin")
    if hashlib.sha256(neff_guard_bytes).hexdigest() != guard_pin:
        raise ValueError("N_eff guard identity does not match authenticated bytes")
    guard = _load_json_bytes(neff_guard_bytes, "N_eff guard")
    _reject_unknown(guard, NEFF_GUARD_SCHEMA, "N_eff guard")
    _reject_outcomes(guard)
    artifact = guard["artifact"]
    arrays = artifact.get("arrays") if isinstance(artifact, dict) else None
    if (
        guard["schema_version"] != 1 or guard["beta_contact"] != 0.015363
        or guard["device"] != "cpu" or guard["shape"] != [10, 300]
        or guard["accepted_range"] != [8.0, 20.0] or guard["guard_passed"] is not True
        or not isinstance(arrays, dict) or set(artifact) != {"path", "sha256", "arrays"}
        or not isinstance(artifact["path"], str) or not artifact["path"]
        or _digest(artifact["sha256"], "N_eff artifact SHA-256") is None
        or set(arrays) != {"q1_neff", "qmin_neff"}
        or any(not isinstance(array, dict) or array != {"dtype": "float64", "shape": [10, 300]} for array in arrays.values())
    ):
        raise ValueError("N_eff guard has invalid canonical artifact metadata")
    checkpoints = guard["checkpoint_sha256"]
    if not isinstance(checkpoints, list) or len(checkpoints) != 10 or len(set(checkpoints)) != 10:
        raise ValueError("N_eff guard must bind ten nondegenerate checkpoint SHAs")
    for index, checkpoint in enumerate(checkpoints):
        _digest(checkpoint, f"N_eff checkpoint SHA-256 {index}")
    for key in ("panel_sha256", "snapshot_manifest_sha256", "q_score_inputs_sha256"):
        _digest(guard[key], f"N_eff {key}")
    medians = (guard["q1_pooled_median"], guard["qmin_pooled_median"])
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 8.0 <= value <= 20.0 for value in medians):
        raise ValueError("N_eff guard pooled medians must be finite and within [8, 20]")
    return {
        "neff_guard_sha256": guard_pin, "q1_pooled_median": medians[0],
        "qmin_pooled_median": medians[1], "guard_passed": True, "neff_guard_passed": True,
    }




def _exclusive_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _validate_governance(governance: dict[str, Any]) -> None:
    _reject_unknown(governance, {
        "schema_version": True, "purpose": True, "binding_beta": True, "v2_launch_code_manifest_sha256": True,
        "neff_guard_sha256": True, "bgt_admitted_manifest_sha256": True, "original_worktree_head_sha256": True,
        "original_config_sha256": True, "seeds": True, "arm_seeds": True, "algo_abort": True,
        "pocket_kill": True, "tournament_schedule": True,
    }, "execution governance")
    if governance["schema_version"] != 2 or governance["binding_beta"] != 0.015363:
        raise ValueError("governance must pin V2 beta 0.015363")
    _digest(governance["v2_launch_code_manifest_sha256"], "V2 launch code-manifest pin", allow_none=True)
    _digest(governance["bgt_admitted_manifest_sha256"], "BGT admitted-manifest pin", allow_none=True)
    _digest(governance["original_worktree_head_sha256"], "original worktree-head pin", allow_none=True)
    _digest(governance["original_config_sha256"], "original config pin", allow_none=True)
    if _digest(governance["neff_guard_sha256"], "N_eff guard governance pin") != NEFF_GUARD_SHA256:
        raise ValueError("governance must pin the canonical N_eff guard")
    required_seeds = {"discovery": [0, 1, 2], "backup": [3, 4], "confirmatory": [8, 9, 10], "confirmatory_backup": [11, 12, 13]}
    required_arm_seeds = {arm: [0, 1, 2] for arm in ("BB-D2", "V1-D2", "DMM", "D1M", "D11", "BGT")}
    seed_maps = (
        governance["seeds"],
        governance["arm_seeds"],
        governance.get("tournament_schedule", {}).get("arms"),
    )
    if any(
        not isinstance(seed_map, dict)
        or any(
            not isinstance(values, list)
            or any(type(value) is not int for value in values)
            for values in seed_map.values()
        )
        for seed_map in seed_maps
    ):
        raise ValueError("governance seed pins must contain integers, not booleans")
    if governance["seeds"] != required_seeds or governance["arm_seeds"] != required_arm_seeds:
        raise ValueError("governance seed ordering does not match the owner pins")
    if governance["algo_abort"] != {"retry": False, "primary_paired_metric": "excluded", "worst_seed_guard": "worst_equivalent"}:
        raise ValueError("governance must encode algo-abort handling")
    if governance["pocket_kill"] != {"after_hours": 3, "progress": 0, "disposition": "algo_abort", "restart": False}:
        raise ValueError("governance must encode the 3h/progress-zero pocket kill")
    expected_schedule = {"status": "locked_by_charter_amendment_2", "amendment_url": "https://github.com/jiminc77/research-dashboard/issues/44#issuecomment-5079492158", "planned_runs": 18, "arms": required_arm_seeds, "bgt_not_admitted": {"runs": 15, "redistribute_runs": False}}
    if governance["tournament_schedule"] != expected_schedule:
        raise ValueError("governance must lock the Amendment 2 balanced schedule")


def validate_manifest(
    manifest: dict[str, Any],
    governance: dict[str, Any],
    *,
    expected_arm: str | None = None,
    expected_seed: int | None = None,
    expected_config_sha256: str | None = None,
    expected_code_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a declarative launch manifest without reading product sources."""
    _validate_governance(governance)
    _reject_outcomes(manifest)
    _reject_unknown(manifest, {
        "arm": True, "mode": True, "policy_delay": True, "binding_beta": True, "code_path": True, "config_lineage": True,
        "merged_final_code": True, "code_manifest_sha256": True, "neff_guard_sha256": True, "seed": True, "schedule_arm": True,
        "d2_lineage": False, "scenario": False, "worktree_head": False, "pinned_worktree_head": False,
        "config_sha256": False, "pinned_config_sha256": False, "admitted_manifest_sha256": False,
    }, "launch manifest")
    arm, mode = manifest["arm"], manifest["mode"]
    if not isinstance(arm, str) or not isinstance(mode, str):
        raise ValueError("launch arm and mode must be strings")
    combinations = {"bb": "original_amd5_v1", "v1": "original_amd5_v1", "bb-d2": "bb_d2", "v1-d2": "v1_d2", "v2-dmm": "v2_live", "v2-d1m": "v2_live", "v2-d11": "v2_live", "v2-bgt": "v2_live"}
    if combinations.get(arm) != mode:
        raise ValueError(f"unknown or forbidden arm/mode combination: {arm!r}/{mode!r}")
    delay, seed = manifest["policy_delay"], manifest["seed"]
    if not isinstance(delay, int) or isinstance(delay, bool):
        raise ValueError("policy_delay must be an integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if manifest["merged_final_code"] is not True:
        raise ValueError("launch requires merged final code")
    if manifest["binding_beta"] != 0.015363:
        raise ValueError("launch must pin V2 beta 0.015363")
    code_pin = _digest(governance["v2_launch_code_manifest_sha256"], "V2 launch code-manifest pin", allow_none=True)
    code_identity = _digest(manifest["code_manifest_sha256"], "launch code-manifest identity")
    if code_pin is None or code_identity != code_pin:
        raise ValueError("launch code-manifest identity must match the independent governance pin")
    if _digest(manifest["neff_guard_sha256"], "launch N_eff guard pin") != governance["neff_guard_sha256"]:
        raise ValueError("launch N_eff guard identity must match the independent governance pin")
    schedule_arm = {"bb-d2": "BB-D2", "v1-d2": "V1-D2", "v2-dmm": "DMM", "v2-d1m": "D1M", "v2-d11": "D11", "v2-bgt": "BGT"}.get(arm)
    if schedule_arm is not None:
        if (
            manifest["schedule_arm"] != schedule_arm
            or seed not in governance["arm_seeds"][schedule_arm]
            or seed not in governance["tournament_schedule"]["arms"][schedule_arm]
        ):
            raise ValueError("launch arm/seed does not occupy an independently pinned Amendment-2 schedule slot")
        if delay != 2 or manifest["code_path"] != "v2" or manifest["config_lineage"] != "v2":
            raise ValueError("V2/D2 launches require V2 lineage and policy_delay == 2")
        config_identity = _digest(manifest.get("config_sha256"), "launch config identity")
        if expected_arm is not None and arm != expected_arm:
            raise ValueError("launch arm does not match the authenticated runtime arm")
        if expected_seed is not None and seed != expected_seed:
            raise ValueError("launch seed does not match the authenticated runtime seed")
        if (
            expected_config_sha256 is not None
            and config_identity != _digest(expected_config_sha256, "authenticated config identity")
        ):
            raise ValueError("launch config identity does not match the authenticated config bytes")
        if (
            expected_code_manifest_sha256 is not None
            and code_identity
            != _digest(expected_code_manifest_sha256, "authenticated code-manifest identity")
        ):
            raise ValueError("launch code-manifest identity does not match the authenticated code-manifest bytes")
        if arm in {"bb-d2", "v1-d2"} and manifest["d2_lineage"] != arm:
            raise ValueError("D2 launches require matching D2 lineage")
        admitted = manifest.get("admitted_manifest_sha256")
        if arm == "v2-bgt":
            if _digest(admitted, "BGT admitted-manifest identity", allow_none=True) != _digest(governance["bgt_admitted_manifest_sha256"], "BGT admitted-manifest pin", allow_none=True) or admitted is None:
                raise ValueError("BGT launches require the independently admitted manifest identity")
        elif admitted is not None:
            raise ValueError("non-BGT launches must not carry an admitted-manifest identity")
    else:
        if delay != 1 or manifest["scenario"] not in {"s6", "s7"} or manifest["code_path"] != "original" or manifest["config_lineage"] != "original":
            raise ValueError("original AMD-5 V1 launch contract violated")
        worktree_pin = _digest(
            governance["original_worktree_head_sha256"],
            "original worktree-head pin",
            allow_none=True,
        )
        config_pin = _digest(
            governance["original_config_sha256"],
            "original config pin",
            allow_none=True,
        )
        if worktree_pin is None or config_pin is None:
            raise ValueError("original launches remain blocked until independent worktree/config pins are supplied")
        if (
            _digest(manifest["worktree_head"], "original worktree-head identity") != worktree_pin
            or _digest(manifest["pinned_worktree_head"], "manifest worktree-head pin") != worktree_pin
            or _digest(manifest["config_sha256"], "original config identity") != config_pin
            or _digest(manifest["pinned_config_sha256"], "manifest config pin") != config_pin
        ):
            raise ValueError("original launch identities must match independent governance pins")
    return {
        "schema_version": 2,
        "content_address": content_address(manifest),
        "arm": arm,
        "seed": seed,
        "schedule_arm": schedule_arm,
        "mode": mode,
        "policy_delay": delay,
        "config_sha256": manifest.get("config_sha256") if schedule_arm is not None else None,
        "code_manifest_sha256": code_identity,
        "neff_guard_sha256": manifest["neff_guard_sha256"],
        "governance_sha256": hashlib.sha256(canonical_json(governance)).hexdigest(),
        "admitted_manifest_sha256": admitted if arm == "v2-bgt" else None,
    }


def validate_manifest_bytes(
    manifest_bytes: bytes,
    governance_bytes: bytes,
    *,
    expected_arm: str | None = None,
    expected_seed: int | None = None,
    expected_config_sha256: str | None = None,
    expected_code_manifest_sha256: str | None = None,
    code_manifest_bytes: bytes | None = None,
    neff_guard_bytes: bytes | None = None,
    runtime_root: Path = ROOT,
) -> dict[str, Any]:
    """Validate exact authenticated JSON bytes and independently computed runtime identities."""
    manifest = _load_json_bytes(manifest_bytes, "launch manifest")
    governance = _load_json_bytes(governance_bytes, "execution governance")
    receipt = validate_manifest(
        manifest,
        governance,
        expected_arm=expected_arm,
        expected_seed=expected_seed,
        expected_config_sha256=expected_config_sha256,
        expected_code_manifest_sha256=expected_code_manifest_sha256,
    )
    if neff_guard_bytes is None:
        raise ValueError("authenticated N_eff guard bytes are required")
    receipt.update(_validate_neff_guard_bytes(neff_guard_bytes, manifest, governance))
    if code_manifest_bytes is not None:
        code_receipt = validate_code_manifest_bytes(
            code_manifest_bytes, runtime_root=runtime_root
        )
        if code_receipt["code_manifest_sha256"] != receipt["code_manifest_sha256"]:
            raise ValueError("launch code-manifest identity does not match authenticated bytes")
        receipt.update(code_receipt)
    return receipt


def validate_manifest_path(
    manifest_path: Path, governance_path: Path = DEFAULT_GOVERNANCE,
    neff_guard_path: Path | None = None,
) -> dict[str, Any]:
    if neff_guard_path is None:
        raise ValueError("N_eff guard path is required")
    return validate_manifest_bytes(
        manifest_path.read_bytes(), governance_path.read_bytes(),
        neff_guard_bytes=neff_guard_path.read_bytes(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--governance", type=Path, default=DEFAULT_GOVERNANCE)
    parser.add_argument("--neff-guard", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    _exclusive_json(args.receipt, validate_manifest_path(args.manifest, args.governance, args.neff_guard))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
