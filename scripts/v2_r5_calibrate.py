#!/usr/bin/env python3
"""R5 development-only BGT rank calibration and complete-selector latency tool.

The calibration input contains state-cloned top-two branch measurements produced
on development states. This script never opens task split files. GPU latency code
is available through ``latency --device cuda`` but must be run only after the
separate GPU gate is authorized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dgcc.rl.sprint_arms import SprintTD3Agent
from dgcc.rl.td3 import TD3Config
from dgcc.rl.v2_arms import BGTAgent

FORBIDDEN_PATH_TOKENS = ("heldout", "held-out", "patch_eval", "patching_probe")
R5_TRANSITIONS = (50_000, 75_000, 100_000, 125_000, 150_000)


def permitted_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    lowered = str(path).lower()
    if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
        raise ValueError(f"R5 refuses confirmatory/probe path: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cluster_bootstrap_lower(seed_accuracies: list[float]) -> float:
    """Exact 5th percentile over all ordered seed-cluster bootstrap samples."""

    values = np.asarray(seed_accuracies, dtype=float)
    if values.size < 2:
        raise ValueError("R5 requires at least two seed clusters")
    grids = np.indices((values.size,) * values.size).reshape(values.size, -1).T
    means = values[grids].mean(axis=1)
    return float(np.quantile(means, 0.05, method="lower"))


def summarize_transition(rows: list[dict[str, Any]], transition: int) -> dict[str, Any]:
    selected = [row for row in rows if int(row["transition"]) == transition]
    if not selected:
        return {"transition": transition, "available": False, "passed": False}
    seed_rows: dict[int, list[bool]] = {}
    margins: list[float] = []
    for row in selected:
        predicted = np.asarray(row["predicted_progress_top2"], dtype=float)
        measured = np.asarray(row["measured_progress_top2"], dtype=float)
        if predicted.shape != (2,) or measured.shape != (2,):
            raise ValueError("progress pairs must each have shape (2,)")
        if not np.isfinite(predicted).all() or not np.isfinite(measured).all():
            raise ValueError("R5 progress values must be finite")
        if predicted[0] == predicted[1] or measured[0] == measured[1]:
            continue
        seed = int(row["seed"])
        seed_rows.setdefault(seed, []).append(
            bool(predicted.argmax() == measured.argmax())
        )
        margins.append(float(row["normalized_q1_margin"]))
    seed_accuracy = {
        str(seed): float(np.mean(outcomes))
        for seed, outcomes in sorted(seed_rows.items())
    }
    accuracies = list(seed_accuracy.values())
    pooled = float(
        np.mean([outcome for outcomes in seed_rows.values() for outcome in outcomes])
    )
    lower = cluster_bootstrap_lower(accuracies)
    passed = (
        pooled >= 0.60 and sum(value > 0.5 for value in accuracies) >= 2 and lower > 0.5
    )
    return {
        "transition": transition,
        "available": True,
        "pairs": sum(len(outcomes) for outcomes in seed_rows.values()),
        "seed_accuracy": seed_accuracy,
        "pooled_accuracy": pooled,
        "seed_cluster_bootstrap_lower95": lower,
        "normalized_margin_p25": float(np.quantile(margins, 0.25)),
        "passed": passed,
    }


def calibrate(input_path: Path, output_path: Path) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("R5 branch input schema_version must be 1")
    if payload.get("data_scope") != "development":
        raise ValueError("R5 input must declare data_scope='development'")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("R5 input requires non-empty rows")
    transitions = [
        summarize_transition(rows, transition) for transition in R5_TRANSITIONS
    ]
    first_pass = next((row for row in transitions if row["passed"]), None)
    result = {
        "schema_version": 1,
        "input_sha256": sha256_file(input_path),
        "data_scope": "development",
        "transitions": transitions,
        "rank_calibration_passed": first_pass is not None,
        "onset_transition": first_pass["transition"] if first_pass else None,
        "margin": first_pass["normalized_margin_p25"] if first_pass else None,
        "gpu_latency_gate": "pending",
        "bgt_admitted": False,
        "reason": (
            "rank calibration passed; synchronized GPU complete-selector latency is still required"
            if first_pass
            else "no checkpoint passed the preregistered rank-calibration gate by 150k"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["output_sha256"] = sha256_file(output_path)
    return result


def load_sprint_checkpoint(agent: SprintTD3Agent, checkpoint: Path) -> None:
    """Load a V1 checkpoint into an R5-only BGT instance before V2 metadata exists."""

    SprintTD3Agent.load_checkpoint(agent, checkpoint)


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def latency(
    checkpoint: Path,
    panel: Path,
    output_path: Path,
    *,
    device: str,
    margin: float,
    onset_transition: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA latency requested but CUDA is unavailable")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    arrays = np.load(panel, allow_pickle=False)
    X = np.asarray(arrays["X"], dtype=float)
    G = np.asarray(arrays["G"], dtype=float)
    if X.shape != G.shape or X.ndim != 3 or X.shape[1:] != (32, 3):
        raise ValueError("latency panel X/G must have matching shape (B, 32, 3)")

    config = TD3Config()
    base = SprintTD3Agent(config, arm="v1", device=device)
    load_sprint_checkpoint(base, checkpoint)
    bgt = BGTAgent(
        config,
        margin=margin,
        onset_transition=onset_transition,
        calibration_sha256="0" * 64,
        device=device,
    )
    load_sprint_checkpoint(bgt, checkpoint)
    rng_base = np.random.default_rng(20260725)
    rng_bgt = np.random.default_rng(20260725)

    def measure(agent: SprintTD3Agent, rng: np.random.Generator) -> np.ndarray:
        samples = []
        for iteration in range(warmup + repeats):
            synchronize(device)
            started = time.perf_counter_ns()
            agent.select_actions(
                X,
                G,
                step=onset_transition,
                total_budget=300_000,
                rng=rng,
                deterministic=True,
            )
            synchronize(device)
            if iteration >= warmup:
                samples.append((time.perf_counter_ns() - started) / 1e6)
        return np.asarray(samples)

    base_ms = measure(base, rng_base)
    bgt_ms = measure(bgt, rng_bgt)
    result = {
        "schema_version": 1,
        "device": device,
        "checkpoint_sha256": sha256_file(checkpoint),
        "panel_sha256": sha256_file(panel),
        "batch_size": len(X),
        "warmup": warmup,
        "repeats": repeats,
        "base_ms": {
            "p50": float(np.median(base_ms)),
            "p95": float(np.quantile(base_ms, 0.95)),
        },
        "bgt_ms": {
            "p50": float(np.median(bgt_ms)),
            "p95": float(np.quantile(bgt_ms, 0.95)),
        },
        "p95_overhead_fraction": float(
            np.quantile(bgt_ms, 0.95) / np.quantile(base_ms, 0.95) - 1.0
        ),
    }
    result["latency_gate_passed"] = result["p95_overhead_fraction"] <= 0.25
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def admit(
    calibration_path: Path, latency_path: Path, output_path: Path
) -> dict[str, Any]:
    """Combine rank and synchronized-GPU gates into a pinned BGT manifest."""

    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    latency_result = json.loads(latency_path.read_text(encoding="utf-8"))
    passed = (
        calibration.get("rank_calibration_passed") is True
        and latency_result.get("device") == "cuda"
        and latency_result.get("latency_gate_passed") is True
    )
    material = {
        "rank_calibration_sha256": sha256_file(calibration_path),
        "gpu_latency_sha256": sha256_file(latency_path),
        "margin": calibration.get("margin"),
        "onset_transition": calibration.get("onset_transition"),
    }
    admission_sha256 = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    result = {
        "schema_version": 1,
        **material,
        "admission_sha256": admission_sha256,
        "bgt_admitted": passed,
        "reason": (
            "rank and synchronized GPU latency gates passed"
            if passed
            else "BGT remains excluded: rank gate or synchronized GPU latency gate failed"
        ),
    }
    if passed:
        result["sprint_bgt_config"] = {
            "margin": calibration["margin"],
            "onset_transition": calibration["onset_transition"],
            "calibration_sha256": admission_sha256,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    calibration = subparsers.add_parser("calibrate")
    calibration.add_argument("--input", type=permitted_path, required=True)
    calibration.add_argument("--output", type=permitted_path, required=True)
    timing = subparsers.add_parser("latency")
    timing.add_argument("--checkpoint", type=permitted_path, required=True)
    timing.add_argument("--panel", type=permitted_path, required=True)
    timing.add_argument("--output", type=permitted_path, required=True)
    timing.add_argument("--device", choices=("cpu", "cuda"), required=True)
    timing.add_argument("--margin", type=float, required=True)
    timing.add_argument("--onset-transition", type=int, required=True)
    timing.add_argument("--warmup", type=int, default=20)
    timing.add_argument("--repeats", type=int, default=100)
    admission = subparsers.add_parser("admit")
    admission.add_argument("--calibration", type=permitted_path, required=True)
    admission.add_argument("--latency", type=permitted_path, required=True)
    admission.add_argument("--output", type=permitted_path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "calibrate":
        result = calibrate(args.input, args.output)
    elif args.command == "latency":
        result = latency(
            args.checkpoint,
            args.panel,
            args.output,
            device=args.device,
            margin=args.margin,
            onset_transition=args.onset_transition,
            warmup=args.warmup,
            repeats=args.repeats,
        )
    else:
        result = admit(args.calibration, args.latency, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "" and "--device" in os.sys.argv:
        requested = os.sys.argv[os.sys.argv.index("--device") + 1]
        if requested == "cuda":
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES is empty; GPU latency execution is disabled"
            )
    raise SystemExit(main())
