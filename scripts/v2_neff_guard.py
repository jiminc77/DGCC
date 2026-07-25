#!/usr/bin/env python3
"""CPU-only N_eff guard over exactly ten 300-state forward-output panels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

CONTACT_WEIGHT_BETA = 0.015363
EXPECTED_SHAPE = (10, 300)
Q_SCORE_SHAPE = (*EXPECTED_SHAPE, 32)
NEFF_LIMITS = (8.0, 20.0)
EXTERNAL_Q_SCORE_PRODUCER_SHA256 = (
    "d6919541bb867bc646d4ae39c2021d5031fadb72a2a756218cb255646fcca191"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
DIGEST = set("0123456789abcdef")


def _valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and not (set(value) - DIGEST)
        and len(set(value)) > 1
    )


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
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


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(".json")
def _ensure_pair_absent(path: Path) -> None:
    sidecar = _sidecar_path(path)
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"N_eff NPZ and sidecar pair must both be absent: {path}, {sidecar}")
def _exclusive_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
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


def _validate_state_input_manifest(state_inputs: Path, snapshot: Path, manifest: dict[str, Any]) -> None:
    sidecar = _sidecar_path(state_inputs)
    if state_inputs.resolve() == sidecar.resolve() or not sidecar.is_file():
        raise ValueError("external Q-score inputs require a sidecar manifest")
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "artifact",
        "protocol",
        "code",
        "inputs",
    }
    if not isinstance(document, dict) or set(document) != expected or document["schema_version"] != 1:
        raise ValueError("Q-score sidecar has an invalid schema")
    artifact, protocol, code, inputs = (
        document["artifact"],
        document["protocol"],
        document["code"],
        document["inputs"],
    )
    if (
        not isinstance(artifact, dict)
        or set(artifact) != {"path", "sha256", "arrays"}
        or artifact["path"] != state_inputs.name
        or artifact["sha256"] != sha256_file(state_inputs)
        or not _valid_digest(artifact["sha256"])
        or not isinstance(protocol, dict)
        or protocol != {"device": "cpu", "contact_weight_beta": CONTACT_WEIGHT_BETA, "forward_only": True}
        or not isinstance(code, dict)
        or set(code) != {"score_producer_sha256"}
        or code["score_producer_sha256"] != EXTERNAL_Q_SCORE_PRODUCER_SHA256
        or not isinstance(inputs, dict)
        or set(inputs) != {"snapshot_manifest_sha256", "checkpoint_sha256", "panel_sha256"}
    ):
        raise ValueError("Q-score sidecar does not bind approved provenance")
    checkpoints = [item["sha256"] for item in manifest["assets"] if item["kind"] == "checkpoint"]
    panel = next(item["sha256"] for item in manifest["assets"] if item["kind"] == "panel")
    if (
        inputs["snapshot_manifest_sha256"] != sha256_file(snapshot / "MANIFEST.json")
        or inputs["checkpoint_sha256"] != checkpoints
        or inputs["panel_sha256"] != panel
        or not _valid_digest(inputs["snapshot_manifest_sha256"])
        or not all(_valid_digest(item) for item in inputs["checkpoint_sha256"])
        or not _valid_digest(inputs["panel_sha256"])
    ):
        raise ValueError("Q-score sidecar provenance does not match snapshot")


def verify_snapshot(snapshot: Path, expected_manifest_sha256: str) -> dict[str, Any]:
    if not _valid_digest(expected_manifest_sha256):
        raise ValueError("snapshot manifest SHA-256 pin must be lowercase hexadecimal")
    manifest_path = snapshot / "MANIFEST.json"
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("snapshot manifest SHA-256 does not match the independently supplied pin")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = document.get("assets")
    attestation = document.get("source_mutation_attestation")
    if not isinstance(attestation, dict) or attestation.get("zero_mutation_observed") is not True:
        raise ValueError("snapshot manifest lacks a zero-mutation attestation")
    if not isinstance(assets, list) or len(assets) != 14:
        raise ValueError("snapshot manifest must enumerate the approved 14 payloads")
    counts = {"checkpoint": 0, "raw_gz": 0, "panel": 0}
    expected_files = {Path("MANIFEST.json")}
    for asset in assets:
        if not isinstance(asset, dict):
            raise ValueError("snapshot manifest asset must be an object")
        kind = asset.get("kind")
        if kind not in counts:
            raise ValueError("snapshot manifest contains an unapproved asset kind")
        destination_text, payload_sha256 = asset.get("destination"), asset.get("sha256")
        if not isinstance(destination_text, str) or not _valid_digest(payload_sha256):
            raise ValueError("snapshot manifest asset lacks a destination or SHA-256")
        destination = Path(destination_text)
        if destination.is_absolute() or ".." in destination.parts or destination in expected_files:
            raise ValueError("snapshot manifest contains an unsafe or duplicate payload destination")
        expected_files.add(destination)
        counts[kind] += 1
        payload = snapshot / destination
        if not payload.is_file() or sha256_file(payload) != payload_sha256:
            raise ValueError(f"snapshot payload hash mismatch: {payload}")
    actual_files = {path.relative_to(snapshot) for path in snapshot.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise ValueError("snapshot contains files outside its approved manifest set")
    if counts != {"checkpoint": 10, "raw_gz": 3, "panel": 1}:
        raise ValueError("snapshot manifest has an invalid approved payload set")
    return document




def effective_sample_size(scores: np.ndarray) -> np.ndarray:
    if scores.shape != Q_SCORE_SHAPE or scores.dtype != np.float64:
        raise ValueError("Q-scores must be float64 with shape (10, 300, 32)")
    if not np.isfinite(scores).all():
        raise ValueError("Q-scores must be finite")
    logits = scores / CONTACT_WEIGHT_BETA
    logits -= logits.max(axis=-1, keepdims=True)
    weights = np.exp(logits)
    normalized = weights / weights.sum(axis=-1, keepdims=True)
    return 1.0 / np.square(normalized).sum(axis=-1)


def run(
    snapshot: Path, state_inputs: Path, output: Path, expected_manifest_sha256: str
) -> dict[str, Any]:
    """Compute only deterministic array diagnostics; no model, environment, or optimizer is invoked."""
    manifest = verify_snapshot(snapshot, expected_manifest_sha256)
    _validate_state_input_manifest(state_inputs, snapshot, manifest)
    if state_inputs.resolve() == output.resolve() or _sidecar_path(state_inputs).resolve() == _sidecar_path(output).resolve():
        raise ValueError("Q-score input paths must not alias N_eff outputs")
    with np.load(state_inputs, allow_pickle=False) as inputs:
        if set(inputs.files) != {"q1_scores", "qmin_scores"}:
            raise ValueError("Q-score inputs must contain exactly q1_scores and qmin_scores")
        q1_neff = effective_sample_size(inputs["q1_scores"])
        qmin_neff = effective_sample_size(inputs["qmin_scores"])
    q1_median = float(np.median(q1_neff))
    qmin_median = float(np.median(qmin_neff))
    guard_passed = all(
        NEFF_LIMITS[0] <= value <= NEFF_LIMITS[1]
        for value in (q1_median, qmin_median)
    )
    _ensure_pair_absent(output)
    _exclusive_npz(output, q1_neff=q1_neff, qmin_neff=qmin_neff)
    pins = {
        "schema_version": 1,
        "artifact": {
            "path": output.name,
            "sha256": sha256_file(output),
            "arrays": {
                "q1_neff": {"dtype": "float64", "shape": list(EXPECTED_SHAPE)},
                "qmin_neff": {"dtype": "float64", "shape": list(EXPECTED_SHAPE)},
            },
        },
        "device": "cpu",
        "shape": list(EXPECTED_SHAPE),
        "beta_contact": CONTACT_WEIGHT_BETA,
        "checkpoint_sha256": [
            item["sha256"] for item in manifest["assets"] if item["kind"] == "checkpoint"
        ],
        "panel_sha256": next(
            item["sha256"] for item in manifest["assets"] if item["kind"] == "panel"
        ),
        "snapshot_manifest_sha256": sha256_file(snapshot / "MANIFEST.json"),
        "q_score_inputs_sha256": sha256_file(state_inputs),
        "q1_pooled_median": q1_median,
        "qmin_pooled_median": qmin_median,
        "accepted_range": list(NEFF_LIMITS),
        "guard_passed": guard_passed,
    }
    _exclusive_write(
        _sidecar_path(output),
        (json.dumps(pins, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    if not guard_passed:
        raise RuntimeError(
            f"pooled N_eff medians must lie in {NEFF_LIMITS}; "
            f"q1={q1_median}, qmin={qmin_median}"
        )
    return pins


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", "--staged", dest="snapshot", type=Path, required=True)
    parser.add_argument("--snapshot-manifest-sha256", required=True)
    parser.add_argument("--state-inputs", type=Path, required=True)
    parser.add_argument("--output", "--out", dest="output", type=Path, required=True)
    args = parser.parse_args()
    verify_snapshot(args.snapshot, args.snapshot_manifest_sha256)
    state_inputs = args.state_inputs
    print(
        json.dumps(
            run(args.snapshot, state_inputs, args.output, args.snapshot_manifest_sha256),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
