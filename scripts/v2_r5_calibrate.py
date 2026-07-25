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
import stat
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dgcc.logging.code_manifest import validate_code_manifest_bytes
from dgcc.rl.sprint_arms import SprintTD3Agent
from dgcc.rl.td3 import TD3Config
from dgcc.rl.v2_arms import (
    BGT_ADMISSION_MANIFEST_KEYS,
    BGTAdmissionRequiredError,
    BenchmarkBGTAgent,
    bgt_admission_material,
    validate_bgt_development_lineage,
    validate_bgt_manifest,
)

FORBIDDEN_PATH_TOKENS = ("heldout", "held-out", "patch_eval", "patching_probe")
R5_TRANSITIONS = (50_000, 75_000, 100_000, 125_000, 150_000)
BENCHMARK_WARMUP = 50
BENCHMARK_BLOCKS = 5
BENCHMARK_REPEATS_PER_ARM = 200
BENCHMARK_BATCH_SIZES = (1024, 300)
MAX_P95_OVERHEAD_FRACTION = 0.25


def _sha256_json(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_pinned_code_manifest(
    path: Path, expected_sha256: str, *, runtime_root: Path | None = None
) -> tuple[bytes, str]:
    """Read, pin, and validate an executable code-manifest document."""
    if not _pinned_sha256(expected_sha256):
        raise ValueError("code-manifest SHA-256 pin is invalid")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise OSError("not a regular file")
            chunks: list[bytes] = []
            digest = hashlib.sha256()
            while chunk := os.read(fd, 1 << 20):
                digest.update(chunk)
                chunks.append(chunk)
        finally:
            os.close(fd)
    except OSError as error:
        raise ValueError("code-manifest is missing or unsafe") from error
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("code-manifest does not match independent SHA-256 pin")
    code_manifest_bytes = b"".join(chunks)
    receipt = validate_code_manifest_bytes(
        code_manifest_bytes,
        runtime_root=runtime_root or Path(__file__).resolve().parents[1],
    )
    if receipt["code_manifest_sha256"] != actual_sha256:
        raise ValueError("code-manifest document identity changed during validation")
    return code_manifest_bytes, actual_sha256


def benchmark_schedule() -> tuple[tuple[str, int, int, str, int], ...]:
    """Fixed complete-selector AB/BA schedule, inspectable without CUDA."""
    return tuple(
        (scenario, batch_size, block, arm, BENCHMARK_REPEATS_PER_ARM)
        for scenario in ("calibrated", "all_eligible_worst_case")
        for batch_size in BENCHMARK_BATCH_SIZES
        for block in range(BENCHMARK_BLOCKS)
        for arm in (("base", "bgt") if block % 2 == 0 else ("bgt", "base"))
    )


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
def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_json_bytes(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return raw, payload


RANK_PAYLOAD_KEYS = frozenset({
    "schema_version", "data_scope", "development_lineage", "registered_seeds",
    "rows_per_seed_transition", "rows",
})
RANK_ROW_KEYS = frozenset({
    "seed", "transition", "normalized_q1_margin", "predicted_progress_top2",
    "measured_progress_top2",
})


def validate_rank_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if (set(payload) != RANK_PAYLOAD_KEYS or payload.get("schema_version") != 1
            or payload.get("data_scope") != "development"):
        raise ValueError("R5 branch input must be closed schema-versioned development data")
    lineage = development_lineage(payload)
    rows, seeds, row_cardinality = (payload["rows"], payload["registered_seeds"],
                                    payload["rows_per_seed_transition"])
    if (not isinstance(rows, list) or not rows or not isinstance(seeds, list)
            or len(seeds) < 2 or len(set(seeds)) != len(seeds)
            or not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds)
            or not isinstance(row_cardinality, int) or isinstance(row_cardinality, bool)
            or row_cardinality < 1
            or len(rows) != len(seeds) * len(R5_TRANSITIONS) * row_cardinality):
        raise ValueError("R5 input has unregistered seed/transition row cardinality")
    expected = {(seed, transition): 0 for seed in seeds for transition in R5_TRANSITIONS}
    for row in rows:
        if (not isinstance(row, dict) or set(row) != RANK_ROW_KEYS
                or not isinstance(row.get("seed"), int) or isinstance(row["seed"], bool)
                or not isinstance(row.get("transition"), int) or isinstance(row["transition"], bool)
                or (row["seed"], row["transition"]) not in expected
                or not isinstance(row["normalized_q1_margin"], (int, float))
                or isinstance(row["normalized_q1_margin"], bool)
                or not np.isfinite(row["normalized_q1_margin"])):
            raise ValueError("R5 row is outside registered development lineage")
        for key in ("predicted_progress_top2", "measured_progress_top2"):
            values = row[key]
            if (not isinstance(values, list) or len(values) != 2
                    or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                           or not np.isfinite(value) for value in values)):
                raise ValueError("R5 progress pairs must be finite numeric pairs")
        expected[(row["seed"], row["transition"])] += 1
    if any(count != row_cardinality for count in expected.values()):
        raise ValueError("R5 row cardinality does not match registered protocol")
    return lineage, rows

def development_lineage(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return validate_bgt_development_lineage(
            payload.get("development_lineage")
        )
    except BGTAdmissionRequiredError as error:
        raise ValueError(
            "R5 input lacks closed authenticated development lineage"
        ) from error


def cluster_bootstrap_lower(seed_accuracies: list[float]) -> float:
    """Bounded deterministic seed-cluster bootstrap lower fifth percentile."""
    values = np.asarray(seed_accuracies, dtype=float)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("R5 requires at least two finite seed clusters")
    draws = np.random.default_rng(20260725).integers(
        0, values.size, size=(10_000, values.size), endpoint=False
    )
    return float(np.quantile(values[draws].mean(axis=1), 0.05, method="lower"))


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
    input_bytes, payload = read_json_bytes(input_path)
    lineage, rows = validate_rank_payload(payload)
    transitions = [summarize_transition(rows, transition) for transition in R5_TRANSITIONS]
    first_pass = next((row for row in transitions if row["passed"]), None)
    result = {
        "schema_version": 1, "input_sha256": _sha256_json(payload),
        "input_file_sha256": hashlib.sha256(input_bytes).hexdigest(), "data_scope": "development",
        "input_payload_sha256": _sha256_json(payload), "input_payload": payload,
        "development_lineage": lineage, "transitions": transitions,
        "rank_calibration_passed": first_pass is not None,
        "onset_transition": first_pass["transition"] if first_pass else None,
        "margin": first_pass["normalized_margin_p25"] if first_pass else None,
        "gpu_latency_gate": "pending", "bgt_admitted": False,
        "reason": ("rank calibration passed; synchronized GPU complete-selector latency is still required"
                   if first_pass else "no checkpoint passed the preregistered rank-calibration gate by 150k"),
    }
    atomic_json(output_path, result)
    result["output_sha256"] = sha256_file(output_path)
    return result


def load_sprint_checkpoint(agent: SprintTD3Agent, checkpoint: Path) -> None:
    """Load a V1 checkpoint into an R5-only BGT instance before V2 metadata exists."""

    SprintTD3Agent.load_checkpoint(agent, checkpoint, eval_only=True)


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def benchmark_batches(
    X: np.ndarray, G: np.ndarray
) -> dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Choose deterministic panel batches once; scenarios share identical rows."""
    if len(X) < max(BENCHMARK_BATCH_SIZES):
        raise ValueError("latency panel must contain at least 1024 rows")
    permutation = np.random.default_rng(20260725).permutation(len(X))
    batches = {
        batch_size: (X[permutation[:batch_size]], G[permutation[:batch_size]])
        for batch_size in BENCHMARK_BATCH_SIZES
    }
    return {
        "calibrated": batches,
        "all_eligible_worst_case": batches,
    }


def complete_selector_protocol(
    selectors: dict[str, Any],
    batches: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
    *,
    synchronize_call: Any,
    timer_ns: Any,
) -> dict[str, Any]:
    """Run the reviewed warmup and alternating measured-call protocol."""
    results: dict[str, Any] = {}
    for scenario, scenario_batches in batches.items():
        force_all_eligible = scenario == "all_eligible_worst_case"
        scenario_result: dict[str, Any] = {}
        for batch_size, (X_batch, G_batch) in scenario_batches.items():
            for arm in ("base", "bgt"):
                for _ in range(BENCHMARK_WARMUP):
                    selectors[arm](X_batch, G_batch, force_all_eligible)
            blocks: list[dict[str, Any]] = []
            all_samples = {"base": [], "bgt": []}
            for block in range(BENCHMARK_BLOCKS):
                order = ("base", "bgt") if block % 2 == 0 else ("bgt", "base")
                block_samples = {"base": [], "bgt": []}
                for arm in order:
                    for _ in range(BENCHMARK_REPEATS_PER_ARM):
                        synchronize_call()
                        started = timer_ns()
                        selectors[arm](X_batch, G_batch, force_all_eligible)
                        synchronize_call()
                        sample = (timer_ns() - started) / 1e6
                        block_samples[arm].append(sample)
                        all_samples[arm].append(sample)
                paired = (
                    np.asarray(block_samples["bgt"]) / np.asarray(block_samples["base"])
                    - 1.0
                )
                blocks.append(
                    {
                        "block": block,
                        "order": "".join("A" if arm == "base" else "B" for arm in order),
                        "base_ms": block_samples["base"],
                        "bgt_ms": block_samples["bgt"],
                        "paired_overhead_fraction": float(np.median(paired)),
                    }
                )
            base = np.asarray(all_samples["base"])
            bgt = np.asarray(all_samples["bgt"])
            scenario_result[str(batch_size)] = {
                "base_ms": {
                    "samples": all_samples["base"],
                    "p50": float(np.median(base)),
                    "p95": float(np.quantile(base, 0.95)),
                },
                "bgt_ms": {
                    "samples": all_samples["bgt"],
                    "p50": float(np.median(bgt)),
                    "p95": float(np.quantile(bgt, 0.95)),
                },
                "blocks": blocks,
                "p95_overhead_fraction": float(
                    np.quantile(bgt, 0.95) / np.quantile(base, 0.95) - 1.0
                ),
            }
        results[scenario] = scenario_result
    return results


def gpu_details() -> dict[str, Any]:
    """Return queried hardware state, explicitly recording unavailable queries."""
    details: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }
    try:
        properties = torch.cuda.get_device_properties(0)
        details.update(
            {
                "device_name": properties.name,
                "device_capability": list(torch.cuda.get_device_capability(0)),
            }
        )
    except Exception as error:
        details["device_query_unavailable_reason"] = str(error)
    query = (
        "name,driver_version,pstate,clocks.current.graphics,"
        "clocks.current.memory,power.draw,power.limit"
    )
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        values = [part.strip() for part in completed.stdout.strip().split(",")]
        if len(values) != 7 or not all(values):
            raise RuntimeError(f"unexpected nvidia-smi response: {completed.stdout!r}")
        details.update(
            {
                "driver_version": values[1],
                "power_state": values[2],
                "clocks_mhz": {"graphics": values[3], "memory": values[4]},
                "power_watts": {"draw": values[5], "limit": values[6]},
            }
        )
    except Exception as error:
        details["clock_power_query_unavailable_reason"] = str(error)
    return details


def latency(
    checkpoint: Path,
    panel: Path,
    output_path: Path,
    *,
    code_manifest_path: Path,
    expected_code_manifest_sha256: str,
    device: str,
    margin: float,
    onset_transition: int,
    gpu_window_approved: bool,
) -> dict[str, Any]:
    _code_manifest_bytes, code_manifest_sha256 = read_pinned_code_manifest(
        code_manifest_path, expected_code_manifest_sha256
    )
    if device != "cuda":
        raise ValueError("complete-selector benchmark requires --device cuda")
    if not gpu_window_approved:
        raise ValueError("complete-selector benchmark requires --gpu-window-approved")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA latency requested but CUDA is unavailable")
    checkpoint_sha256_before = sha256_file(checkpoint)
    panel_sha256_before = sha256_file(panel)
    arrays = np.load(panel, allow_pickle=False)
    X = np.asarray(arrays["X"], dtype=float)
    G = np.asarray(arrays["G"], dtype=float)
    if X.shape != G.shape or X.ndim != 3 or X.shape[1:] != (32, 3):
        raise ValueError("latency panel X/G must have matching shape (B, 32, 3)")
    if sha256_file(panel) != panel_sha256_before:
        raise RuntimeError("latency panel changed while loading")
    config = TD3Config()
    base = SprintTD3Agent(config, arm="v1", device=device)
    load_sprint_checkpoint(base, checkpoint)
    if sha256_file(checkpoint) != checkpoint_sha256_before:
        raise RuntimeError("latency checkpoint changed while loading")
    identities = {
        "rank_calibration_sha256": checkpoint_sha256_before,
        "gpu_latency_sha256": panel_sha256_before,
        "margin": margin,
        "onset_transition": onset_transition,
        "checkpoint_sha256": checkpoint_sha256_before,
        "panel_sha256": panel_sha256_before,
        "config_sha256": _sha256_json(config.to_dict()),
        "code_sha256": code_manifest_sha256,
    }
    bgt = BenchmarkBGTAgent(
        config,
        margin=margin,
        onset_transition=onset_transition,
        device=device,
    )
    bgt.load_benchmark_checkpoint(checkpoint)

    def select_base(X_batch: np.ndarray, G_batch: np.ndarray, _: bool) -> None:
        base.select_actions(
            X_batch, G_batch, step=onset_transition, total_budget=300_000,
            rng=np.random.default_rng(20260725), deterministic=True,
        )

    def select_bgt(X_batch: np.ndarray, G_batch: np.ndarray, force: bool) -> None:
        bgt.select_actions(
            X_batch, G_batch, step=onset_transition, total_budget=300_000,
            rng=np.random.default_rng(20260725), deterministic=True,
            benchmark_force_all_eligible=force,
        )

    batches = benchmark_batches(X, G)
    torch.cuda.reset_peak_memory_stats()
    results = complete_selector_protocol(
        {"base": select_base, "bgt": select_bgt},
        batches,
        synchronize_call=lambda: synchronize(device),
        timer_ns=time.perf_counter_ns,
    )
    for scenario, scenario_batches in batches.items():
        force = scenario == "all_eligible_worst_case"
        for batch_size, (X_batch, G_batch) in scenario_batches.items():
            _, _, _, info = bgt.select_actions(
                X_batch, G_batch, step=onset_transition, total_budget=300_000,
                rng=np.random.default_rng(20260725), deterministic=True,
                return_info=True, benchmark_force_all_eligible=force,
            )
            results[scenario][str(batch_size)]["eligibility_fraction"] = float(
                np.mean(info["bgt_eligible"])
            )
    overheads = [
        row["p95_overhead_fraction"]
        for scenario in results.values()
        for row in scenario.values()
    ]
    if (sha256_file(checkpoint) != checkpoint_sha256_before
            or sha256_file(panel) != panel_sha256_before):
        raise RuntimeError("latency inputs changed during benchmark")
    result = {
        "schema_version": 2,
        "device": device,
        **{key: identities[key] for key in ("checkpoint_sha256", "panel_sha256", "config_sha256", "code_sha256")},
        "margin": margin,
        "onset_transition": onset_transition,
        "warmup": BENCHMARK_WARMUP,
        "blocks": BENCHMARK_BLOCKS,
        "repeats_per_arm": BENCHMARK_REPEATS_PER_ARM,
        "batch_sizes": list(BENCHMARK_BATCH_SIZES),
        "scenarios": results,
        "schedule": [list(entry) for entry in benchmark_schedule()],
        "gpu": gpu_details(),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "max_complete_selector_p95_overhead_fraction": float(max(overheads)),
    }
    result["latency_gate_passed"] = (
        result["max_complete_selector_p95_overhead_fraction"]
        <= MAX_P95_OVERHEAD_FRACTION
    )
    atomic_json(output_path, result)
    return result


def valid_latency_protocol(latency_result: dict[str, Any]) -> bool:
    """Recompute the latency admission condition from the recorded protocol."""
    if (
        latency_result.get("schema_version") != 2
        or latency_result.get("device") != "cuda"
        or latency_result.get("warmup") != BENCHMARK_WARMUP
        or latency_result.get("blocks") != BENCHMARK_BLOCKS
        or latency_result.get("repeats_per_arm") != BENCHMARK_REPEATS_PER_ARM
        or tuple(latency_result.get("batch_sizes", ())) != BENCHMARK_BATCH_SIZES
        or latency_result.get("schedule") != [list(entry) for entry in benchmark_schedule()]
    ):
        return False
    scenarios = latency_result.get("scenarios")
    if not isinstance(scenarios, dict) or set(scenarios) != {
        "calibrated", "all_eligible_worst_case"
    }:
        return False
    gpu = latency_result.get("gpu")
    if not isinstance(gpu, dict):
        return False
    required_gpu_strings = (
        "torch_version",
        "torch_cuda_version",
        "device_name",
        "driver_version",
        "power_state",
    )
    if any(
        not isinstance(gpu.get(field), str) or not gpu[field]
        for field in required_gpu_strings
    ):
        return False
    clocks = gpu.get("clocks_mhz")
    power = gpu.get("power_watts")
    if (
        not isinstance(clocks, dict)
        or any(
            not isinstance(clocks.get(field), str) or not clocks[field]
            for field in ("graphics", "memory")
        )
        or not isinstance(power, dict)
        or any(
            not isinstance(power.get(field), str) or not power[field]
            for field in ("draw", "limit")
        )
        or not isinstance(latency_result.get("peak_memory_bytes"), int)
        or latency_result["peak_memory_bytes"] < 0
    ):
        return False
    overheads = []
    expected_batches = {str(batch_size) for batch_size in BENCHMARK_BATCH_SIZES}
    for scenario_name, scenario in scenarios.items():
        if not isinstance(scenario, dict) or set(scenario) != expected_batches:
            return False
        for batch_size, batch in scenario.items():
            if not isinstance(batch, dict) or len(batch.get("blocks", ())) != BENCHMARK_BLOCKS:
                return False
            eligibility = batch.get("eligibility_fraction")
            if (
                not isinstance(eligibility, (int, float))
                or not 0.0 <= eligibility <= 1.0
                or (scenario_name == "all_eligible_worst_case" and eligibility != 1.0)
            ):
                return False
            samples_by_arm: dict[str, np.ndarray] = {}
            for arm in ("base", "bgt"):
                measurements = batch.get(f"{arm}_ms")
                samples = measurements.get("samples") if isinstance(measurements, dict) else None
                if not isinstance(samples, list) or len(samples) != (
                    BENCHMARK_BLOCKS * BENCHMARK_REPEATS_PER_ARM
                ):
                    return False
                values = np.asarray(samples, dtype=float)
                if not np.isfinite(values).all() or (values <= 0).any():
                    return False
                if (
                    measurements.get("p50") != float(np.median(values))
                    or measurements.get("p95") != float(np.quantile(values, 0.95))
                ):
                    return False
                samples_by_arm[arm] = values
            reconstructed = {"base": [], "bgt": []}
            for block_number, block in enumerate(batch["blocks"]):
                expected_order = "AB" if block_number % 2 == 0 else "BA"
                if (
                    not isinstance(block, dict)
                    or block.get("block") != block_number
                    or block.get("order") != expected_order
                    or len(block.get("base_ms", ())) != BENCHMARK_REPEATS_PER_ARM
                    or len(block.get("bgt_ms", ())) != BENCHMARK_REPEATS_PER_ARM
                ):
                    return False
                for arm in ("base", "bgt"):
                    values = block[arm + "_ms"]
                    if (not isinstance(values, list) or len(values) != BENCHMARK_REPEATS_PER_ARM
                            or not np.isfinite(np.asarray(values, dtype=float)).all()
                            or (np.asarray(values, dtype=float) <= 0).any()):
                        return False
                    reconstructed[arm].extend(values)
                paired = np.asarray(block["bgt_ms"], dtype=float) / np.asarray(
                    block["base_ms"], dtype=float
                ) - 1.0
                if (
                    not np.isfinite(paired).all()
                    or block.get("paired_overhead_fraction") != float(np.median(paired))
                ):
                    return False
            if any(not np.array_equal(samples_by_arm[arm], np.asarray(reconstructed[arm], dtype=float))
                   for arm in ("base", "bgt")):
                return False
            overhead = batch.get("p95_overhead_fraction")
            reconstructed_base = np.asarray(reconstructed["base"], dtype=float)
            reconstructed_bgt = np.asarray(reconstructed["bgt"], dtype=float)
            computed = float(
                np.quantile(reconstructed_bgt, 0.95)
                / np.quantile(reconstructed_base, 0.95) - 1.0
            )
            if (
                not isinstance(overhead, (int, float))
                or not np.isfinite(overhead)
                or overhead != computed
            ):
                return False
            overheads.append(float(overhead))
    maximum = max(overheads, default=float("inf"))
    return (
        latency_result.get("max_complete_selector_p95_overhead_fraction") == maximum
        and latency_result.get("latency_gate_passed") is (maximum <= MAX_P95_OVERHEAD_FRACTION)
    )


def _pinned_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
        and len(set(value)) > 1
    )


def _validated_rank_calibration(calibration: dict[str, Any]) -> tuple[bool, float | None, int | None]:
    """Recompute rank-gate evidence from embedded development-only source rows."""
    payload = calibration.get("input_payload")
    if (
        calibration.get("schema_version") != 1
        or calibration.get("data_scope") != "development"
        or calibration.get("input_sha256") != _sha256_json(payload)
        or not isinstance(payload, dict)
        or calibration.get("input_payload_sha256") != _sha256_json(payload)
        or payload.get("schema_version") != 1
        or payload.get("data_scope") != "development"
        or not isinstance(payload.get("rows"), list)
        or not payload["rows"]
    ):
        return False, None, None
    try:
        lineage, rows = validate_rank_payload(payload)
        if calibration.get("development_lineage") != lineage:
            return False, None, None
        transitions = [summarize_transition(rows, transition) for transition in R5_TRANSITIONS]
    except (KeyError, TypeError, ValueError):
        return False, None, None
    if any(not transition["available"] for transition in transitions):
        return False, None, None
    first_pass = next((transition for transition in transitions if transition["passed"]), None)
    margin = first_pass["normalized_margin_p25"] if first_pass else None
    onset = first_pass["transition"] if first_pass else None
    if (
        calibration.get("transitions") != transitions
        or calibration.get("rank_calibration_passed") is not (first_pass is not None)
        or calibration.get("margin") != margin
        or calibration.get("onset_transition") != onset
        or first_pass is None
        or not np.isfinite(margin)
        or margin < 0
    ):
        return False, None, None
    return True, float(margin), int(onset)


def admit(
    calibration_path: Path,
    latency_path: Path,
    output_path: Path,
    *,
    code_manifest_path: Path,
    expected_code_manifest_sha256: str,
) -> dict[str, Any]:
    _code_manifest_bytes, code_manifest_sha256 = read_pinned_code_manifest(
        code_manifest_path, expected_code_manifest_sha256
    )
    calibration_bytes, calibration = read_json_bytes(calibration_path)
    latency_bytes, latency_result = read_json_bytes(latency_path)
    rank_passed, margin, onset_transition = _validated_rank_calibration(calibration)
    latency_passed = valid_latency_protocol(latency_result)
    lineage = calibration.get("development_lineage")
    identities_valid = (
        isinstance(lineage, dict)
        and all(_pinned_sha256(latency_result.get(field)) for field in (
            "checkpoint_sha256", "panel_sha256", "config_sha256", "code_sha256"
        ))
        and all(latency_result.get(field) == lineage.get(field) for field in (
            "checkpoint_sha256", "panel_sha256", "config_sha256", "code_sha256"
        ))
        and latency_result.get("code_sha256") == code_manifest_sha256
        and lineage.get("code_sha256") == code_manifest_sha256
    )
    latency_matches_rank = (
        latency_result.get("margin") == margin
        and latency_result.get("onset_transition") == onset_transition
    )
    material = bgt_admission_material(
        {
            "rank_calibration_sha256": hashlib.sha256(calibration_bytes).hexdigest(),
            "gpu_latency_sha256": hashlib.sha256(latency_bytes).hexdigest(),
            "margin": margin,
            "onset_transition": onset_transition,
            "development_lineage": lineage,
            "checkpoint_sha256": latency_result.get("checkpoint_sha256"),
            "panel_sha256": latency_result.get("panel_sha256"),
            "config_sha256": latency_result.get("config_sha256"),
            "code_sha256": code_manifest_sha256,
        }
    )
    passed = rank_passed and latency_passed and identities_valid and latency_matches_rank
    result = {
        "schema_version": 1,
        **material,
        "admission_sha256": _sha256_json(material),
        "bgt_admitted": passed,
        "reason": (
            "rank and approved synchronized-GPU latency gates passed"
            if passed
            else "BGT remains a conditional candidate pending valid rank and GPU admission"
        ),
    }
    if set(result) != BGT_ADMISSION_MANIFEST_KEYS:
        raise RuntimeError("BGT admission producer schema drifted")
    atomic_json(output_path, result)
    return result


def tournament_cutoff(
    manifest_path: Path,
    expected_manifest_sha256: str,
    state_path: Path,
    code_manifest_path: Path,
    expected_code_manifest_sha256: str,
    checkpoint_sha256: str,
    panel_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    _code_manifest_bytes, code_manifest_sha256 = read_pinned_code_manifest(
        code_manifest_path, expected_code_manifest_sha256
    )
    identities = {
        "manifest_sha256": expected_manifest_sha256,
        "code_final_sha256": code_manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "panel_sha256": panel_sha256,
        "config_sha256": config_sha256,
    }
    if not all(_pinned_sha256(value) for value in identities.values()):
        raise ValueError("ambiguous tournament cutoff identities")
    if state_path.exists():
        _state_bytes, state = read_json_bytes(state_path)
        body = {key: value for key, value in state.items() if key != "state_sha256"}
        if (state.get("state_sha256") != _sha256_json(body)
                or state.get("identities") != identities
                or state.get("status") not in {"candidate", "not-admitted"}):
            raise ValueError("ambiguous tournament cutoff state: identity changed")
        if state["status"] == "candidate":
            if (not manifest_path.is_file()
                    or hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected_manifest_sha256):
                raise ValueError("ambiguous tournament cutoff state: admitted manifest changed")
            manifest = validate_bgt_manifest(
                manifest_path,
                expected_manifest_sha256,
                checkpoint_sha256,
                panel_sha256,
                config_sha256,
                code_manifest_sha256,
            )
            if manifest["code_sha256"] != code_manifest_sha256:
                raise ValueError("ambiguous tournament cutoff state: admitted code changed")
        return state
    admitted = False
    try:
        manifest_info = os.lstat(manifest_path)
    except FileNotFoundError:
        manifest_info = None
    if manifest_info is not None:
        if stat.S_ISLNK(manifest_info.st_mode) or not stat.S_ISREG(
            manifest_info.st_mode
        ):
            raise ValueError(
                "tournament cutoff manifest is not a safe regular file"
            )
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected_manifest_sha256:
            raise ValueError(
                "tournament cutoff manifest does not match authenticated pin"
            )
        manifest = validate_bgt_manifest(
            manifest_path,
            expected_manifest_sha256,
            checkpoint_sha256,
            panel_sha256,
            config_sha256,
            code_manifest_sha256,
        )
        if manifest["code_sha256"] != code_manifest_sha256:
            raise ValueError("tournament cutoff code does not match authenticated admitted manifest")
        admitted = True
    state = {
        "schema_version": 1,
        "status": "candidate" if admitted else "not-admitted",
        "identities": identities,
    }
    state["state_sha256"] = _sha256_json(state)
    atomic_json(state_path, state)
    return state


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
    timing.add_argument("--code-manifest", type=permitted_path, required=True)
    timing.add_argument("--expected-code-manifest-sha256", required=True)
    timing.add_argument("--device", choices=("cuda",), required=True)
    timing.add_argument("--margin", type=float, required=True)
    timing.add_argument("--onset-transition", type=int, required=True)
    timing.add_argument("--gpu-window-approved", action="store_true")
    admission = subparsers.add_parser("admit")
    admission.add_argument("--calibration", type=permitted_path, required=True)
    admission.add_argument("--latency", type=permitted_path, required=True)
    admission.add_argument("--output", type=permitted_path, required=True)
    admission.add_argument("--code-manifest", type=permitted_path, required=True)
    admission.add_argument("--expected-code-manifest-sha256", required=True)
    cutoff = subparsers.add_parser("tournament_cutoff")
    cutoff.add_argument("--manifest", type=permitted_path, required=True)
    cutoff.add_argument("--expected-manifest-sha256", required=True)
    cutoff.add_argument("--state", type=permitted_path, required=True)
    cutoff.add_argument("--code-manifest", type=permitted_path, required=True)
    cutoff.add_argument("--expected-code-manifest-sha256", required=True)
    cutoff.add_argument("--checkpoint-sha256", required=True)
    cutoff.add_argument("--panel-sha256", required=True)
    cutoff.add_argument("--config-sha256", required=True)
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
            code_manifest_path=args.code_manifest,
            expected_code_manifest_sha256=args.expected_code_manifest_sha256,
            device=args.device,
            margin=args.margin,
            onset_transition=args.onset_transition,
            gpu_window_approved=args.gpu_window_approved,
        )
    elif args.command == "admit":
        result = admit(
            args.calibration,
            args.latency,
            args.output,
            code_manifest_path=args.code_manifest,
            expected_code_manifest_sha256=args.expected_code_manifest_sha256,
        )
    else:
        result = tournament_cutoff(
            args.manifest,
            args.expected_manifest_sha256,
            args.state,
            args.code_manifest,
            args.expected_code_manifest_sha256,
            args.checkpoint_sha256,
            args.panel_sha256,
            args.config_sha256,
        )
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
