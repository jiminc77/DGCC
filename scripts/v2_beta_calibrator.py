#!/usr/bin/env python3
"""Registered, deterministic beta calibrator for the 07g fixed Q-score panel."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dgcc.rl.selection import contact_softmax_weights as production_contact_softmax_weights

FORBIDDEN_FIELDS = frozenset({
    "return", "reward", "success", "final_distance", "heldout",
    "held_out", "checkpoint_selection_score", "candidate_outcome", "arm_ranking",
})
SHAPE = (10, 300, 32)
ITERATIONS = 64
LOWER = 0.010000
UPPER = 0.025000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    reject_forbidden_fields(value)
    return value


def _normal_field(name: str) -> str:
    return name.casefold().replace("-", "_").replace(" ", "_")


def reject_forbidden_fields(value: Any, location: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{location}: non-string JSON field")
            if _normal_field(key) in FORBIDDEN_FIELDS:
                raise ValueError(f"{location}.{key}: forbidden field")
            reject_forbidden_fields(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_forbidden_fields(child, f"{location}[{index}]")


def require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label}: expected SHA-256 hex string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label}: expected SHA-256 hex string") from exc
    return value.lower()


def _checkpoint_hashes(manifest: dict[str, Any]) -> list[str]:
    values = manifest.get("checkpoint_hashes")
    if not isinstance(values, list) or len(values) != 10:
        raise ValueError("input manifest: exactly 10 checkpoint_hashes required")
    return [require_hash(item, "checkpoint hash") for item in values]


def validate_pins(input_manifest: dict[str, Any], protocol: dict[str, Any], hashes: dict[str, str]) -> None:
    checkpoints = _checkpoint_hashes(input_manifest)
    if checkpoints != _checkpoint_hashes(protocol):
        raise ValueError("checkpoint hashes do not match protocol")
    panel = require_hash(input_manifest.get("fixed_panel_hash"), "input fixed_panel_hash")
    if panel != require_hash(protocol.get("fixed_panel_hash"), "protocol fixed_panel_hash"):
        raise ValueError("fixed panel hash does not match protocol")
    for field, observed in hashes.items():
        expected = require_hash(protocol.get(f"{field}_sha256"), f"protocol {field}_sha256")
        if observed != expected:
            raise ValueError(f"{field} hash does not match protocol")
    if protocol.get("status") != "REGISTERED_BEFORE_EXECUTION":
        raise ValueError("protocol is not registered before execution")
    if protocol.get("rule") != "G=min(M_Q1,M_Qmin); require G(0.025)>=12; 64 bisections on [0.010000,0.025000]; ceil upper to 6 decimals":
        raise ValueError("protocol rule is not the registered 07g rule")


def load_scores(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"q1_scores", "qmin_scores"}:
            raise ValueError("score NPZ must contain only q1_scores and qmin_scores")
        q1, qmin = archive["q1_scores"], archive["qmin_scores"]
    for name, value in (("q1_scores", q1), ("qmin_scores", qmin)):
        if value.dtype != np.dtype("float64") or value.shape != SHAPE:
            raise ValueError(f"{name} must be float64 with shape {SHAPE}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    return q1, qmin


def calibration_softmax_weights(scores: np.ndarray, beta: float) -> np.ndarray:
    """Production stable float64 state-wise softmax over the contact dimension."""
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    scaled = scores / np.float64(beta)
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def median_n_eff(scores: np.ndarray, beta: float) -> float:
    weights = calibration_softmax_weights(scores, beta)
    n_eff = 1.0 / np.sum(weights * weights, axis=-1)
    return float(np.median(n_eff))


def measures(q1: np.ndarray, qmin: np.ndarray, beta: float) -> tuple[float, float, float]:
    m_q1, m_qmin = median_n_eff(q1, beta), median_n_eff(qmin, beta)
    return m_q1, m_qmin, min(m_q1, m_qmin)


def production_measures(
    q1: np.ndarray, qmin: np.ndarray, beta: float
) -> tuple[float, float]:
    medians: list[float] = []
    for scores in (q1, qmin):
        tensor = torch.from_numpy(scores.reshape(-1, SHAPE[-1]))
        weights = production_contact_softmax_weights(tensor, beta_contact=beta)
        n_eff = 1.0 / weights.square().sum(dim=1)
        medians.append(float(torch.median(n_eff)))
    return medians[0], medians[1]


def reserve(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)


def utc_and_kst() -> tuple[str, str]:
    now = dt.datetime.now(dt.timezone.utc)
    return now.isoformat().replace("+00:00", "Z"), now.astimezone(dt.timezone(dt.timedelta(hours=9))).isoformat()


def calibrate(q1: np.ndarray, qmin: np.ndarray) -> tuple[float, list[tuple[float, float, float, float, float, float]]]:
    _, _, g_upper = measures(q1, qmin, UPPER)
    if g_upper < 12.0:
        raise ValueError("G(0.025) < 12; expansion is prohibited")
    lower, upper = LOWER, UPPER
    trace: list[tuple[float, float, float, float, float, float]] = []
    for _ in range(ITERATIONS):
        middle = (lower + upper) / 2.0
        m_q1, m_qmin, g = measures(q1, qmin, middle)
        trace.append((lower, upper, middle, m_q1, m_qmin, g))
        if g >= 12.0:
            upper = middle
        else:
            lower = middle
    beta = math.ceil(upper * 1_000_000.0) / 1_000_000.0
    return beta, trace


def verify_result(
    result_path: Path, trace_path: Path, score_path: Path
) -> dict[str, Any]:
    result = load_json(result_path)
    if result.get("trace_sha256") != sha256_file(trace_path):
        raise ValueError("trace hash does not match result")
    if result.get("score_sha256") != sha256_file(score_path):
        raise ValueError("score hash does not match result")
    q1, qmin = load_scores(score_path)
    expected_beta, expected_rows = calibrate(q1, qmin)
    names = ("lower", "upper", "mid", "M_Q1", "M_Qmin", "G")
    with np.load(trace_path, allow_pickle=False) as trace:
        if set(trace.files) != set(names) or any(
            trace[name].dtype != np.dtype("float64") or trace[name].shape != (64,)
            for name in names
        ):
            raise ValueError("invalid trace schema")
        for index, name in enumerate(names):
            expected = np.asarray(
                [row[index] for row in expected_rows], dtype=np.float64
            )
            if not np.array_equal(trace[name], expected):
                raise ValueError(
                    f"trace {name} does not match deterministic recomputation"
                )
    production_q1, production_qmin = production_measures(
        q1, qmin, expected_beta
    )
    if result.get("beta_contact") != expected_beta:
        raise ValueError("selected beta does not match deterministic recomputation")
    if (
        result.get("M_Q1") != production_q1
        or result.get("M_Qmin") != production_qmin
    ):
        raise ValueError("production medians do not match deterministic recomputation")
    if (
        result.get("outer_guard_pass") is not True
        or result.get("calibration_target_pass") is not True
    ):
        raise ValueError("calibration guard results are not passing")
    if (
        result.get("rng_not_instantiated") is not True
        or result.get("production_run_count_at_registration") != 0
    ):
        raise ValueError("registration invariants missing")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--score-npz", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--code-manifest", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.result == args.trace:
        raise ValueError("result and trace paths must differ")
    result_fd = reserve(args.result)
    try:
        trace_fd = reserve(args.trace)
    except Exception:
        os.close(result_fd)
        raise
    try:
        input_manifest, protocol, code_manifest = (load_json(args.input_manifest), load_json(args.protocol), load_json(args.code_manifest))
        calibrator_path = Path(__file__).resolve()
        hashes = {
            "input_manifest": sha256_file(args.input_manifest), "score": sha256_file(args.score_npz),
            "code_manifest": sha256_file(args.code_manifest), "calibrator": sha256_file(calibrator_path),
        }
        if require_hash(code_manifest.get("calibrator_sha256"), "code manifest calibrator_sha256") != hashes["calibrator"]:
            raise ValueError("calibrator hash does not match code manifest")
        validate_pins(input_manifest, protocol, hashes)
        q1, qmin = load_scores(args.score_npz)
        beta, trace = calibrate(q1, qmin)
        production_q1, production_qmin = production_measures(q1, qmin, beta)
        if not (12.0 <= production_q1 <= 20.0 and 12.0 <= production_qmin <= 20.0):
            raise ValueError("production contact_softmax_weights medians are outside [12,20]")
        names = ("lower", "upper", "mid", "M_Q1", "M_Qmin", "G")
        with os.fdopen(trace_fd, "wb") as handle:
            np.savez(handle, **{name: np.asarray([row[i] for row in trace], dtype=np.float64) for i, name in enumerate(names)})
        trace_fd = -1
        utc, kst = utc_and_kst()
        result = {
            "argv": [str(item) for item in (argv if argv is not None else os.sys.argv[1:])],
            "calibrator_sha256": hashes["calibrator"], "code_manifest_sha256": hashes["code_manifest"],
            "input_manifest_sha256": hashes["input_manifest"], "protocol_sha256": sha256_file(args.protocol),
            "score_sha256": hashes["score"], "trace_sha256": sha256_file(args.trace),
            "beta_contact": beta,
            "M_Q1": production_q1,
            "M_Qmin": production_qmin,
            "outer_guard_pass": 8.0 <= production_q1 <= 20.0
            and 8.0 <= production_qmin <= 20.0,
            "calibration_target_pass": min(production_q1, production_qmin) >= 12.0,
            "iterations": 64,
            "rng_not_instantiated": True,
            "production_run_count_at_registration": 0,
            "stdout_sha256": hashlib.sha256(b"").hexdigest(),
            "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            "utc_timestamp": utc,
            "kst_timestamp": kst,
        }
        with os.fdopen(result_fd, "w", encoding="utf-8") as handle:
            json.dump(result, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        result_fd = -1
        verify_result(args.result, args.trace, args.score_npz)
        return 0
    finally:
        if result_fd != -1:
            os.close(result_fd)
        if trace_fd != -1:
            os.close(trace_fd)


if __name__ == "__main__":
    raise SystemExit(main())
