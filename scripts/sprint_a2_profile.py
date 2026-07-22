#!/usr/bin/env python3
"""CPU-safe G10 Path B checkpoint-cycle and patch-overhead profiler.

The operations are injected so this harness never selects a device or touches
split/probe data itself.  The command-line entry point accepts import paths for
real operations; tests use ordinary CPU mocks.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/metrics/sprint_a2_profile.json"
GPU_DAY_SECONDS = 86_400.0
PASS_DECISION = "PASS: projected patch overhead is within one GPU-day."
FALLBACK_DECISION = "FALLBACK: projected patch overhead exceeds one GPU-day; recommend Plan B1."


def _summary(samples: list[float]) -> dict[str, float]:
    """Return deterministic linear-interpolated median and p95 for samples."""
    if not samples:
        raise ValueError("at least one measurement is required")
    ordered = sorted(samples)

    def percentile(percent: float) -> float:
        position = (len(ordered) - 1) * percent
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {"median_seconds": percentile(0.5), "p95_seconds": percentile(0.95)}


def profile(
    load_checkpoint: Callable[[], Any],
    unload_checkpoint: Callable[[Any], None],
    apply_patch: Callable[[], Any],
    *,
    repeats: int,
    patch_count: int,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Measure injected operations and decide whether Path B fits one GPU-day.

    A checkpoint cycle is one load followed by unloading the object returned by
    that load.  Patch overhead is measured independently once per repetition.
    """
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if patch_count < 0:
        raise ValueError("patch_count must be non-negative")

    checkpoint_cycles: list[float] = []
    patch_samples: list[float] = []
    for _ in range(repeats):
        started = clock()
        checkpoint = load_checkpoint()
        unload_checkpoint(checkpoint)
        checkpoint_cycles.append(clock() - started)

        started = clock()
        apply_patch()
        patch_samples.append(clock() - started)

    cycle_summary = _summary(checkpoint_cycles)
    patch_summary = _summary(patch_samples)
    projected_seconds = patch_count * patch_summary["median_seconds"]
    status = "PASS" if projected_seconds <= GPU_DAY_SECONDS else "FALLBACK"
    return {
        "schema_version": 1,
        "repeats": repeats,
        "patch_count": patch_count,
        "checkpoint_load_unload_seconds": {"samples": checkpoint_cycles, **cycle_summary},
        "patch_overhead_seconds": {"samples": patch_samples, **patch_summary},
        "projected_total_seconds": projected_seconds,
        "gpu_day_seconds": GPU_DAY_SECONDS,
        "status": status,
        "decision": PASS_DECISION if status == "PASS" else FALLBACK_DECISION,
    }


def write_profile(result: dict[str, Any], output: Path = DEFAULT_OUTPUT) -> None:
    """Persist the profile under the registered metrics-path convention."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _operation(import_path: str) -> Callable[..., Any]:
    module_name, separator, attribute = import_path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("operation must be MODULE:CALLABLE")
    operation = getattr(importlib.import_module(module_name), attribute)
    if not callable(operation):
        raise TypeError(f"operation is not callable: {import_path}")
    return operation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", required=True, help="MODULE:CALLABLE with no arguments")
    parser.add_argument("--unload", required=True, help="MODULE:CALLABLE accepting the loaded checkpoint")
    parser.add_argument("--patch", required=True, help="MODULE:CALLABLE with no arguments")
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--patch-count", type=int, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = profile(
        _operation(args.load), _operation(args.unload), _operation(args.patch),
        repeats=args.repeats, patch_count=args.patch_count,
    )
    write_profile(result, args.output)
    print(result["decision"])
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
