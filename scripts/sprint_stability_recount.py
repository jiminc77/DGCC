#!/usr/bin/env python3
"""Recount stability counters across full-scene rebuild resets without modifying logs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "outputs" / "reports"
COUNTERS_RE = re.compile(r"\bnan_env=(\d+)\s+mag=(\d+)\s+rebuilds=(\d+)\b")
REBUILD_RE = re.compile(r"\brebuild=(\d+)\b.*\baction=full_scene_rebuild\b")
RUN_TAG_RE = re.compile(r"\brun_tag=([^\s]+)")


def _run_tag(path: Path, text: str) -> str:
    match = RUN_TAG_RE.search(text)
    if match:
        return match.group(1)
    prefix = "p1_sprint_train_"
    stem = path.stem
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def recount_log(path: Path) -> dict[str, Any]:
    """Return conservative per-reset counter maxima from one immutable training log."""
    text = path.read_text(encoding="utf-8", errors="replace")
    reported: dict[str, int] | None = None
    samples: list[tuple[int, int, int, int]] = []
    boundaries: list[dict[str, Any]] = []
    rebuild_lines: list[int] = []
    rebuild_total = 0

    for line_number, line in enumerate(text.splitlines(), start=1):
        rebuild_match = REBUILD_RE.search(line)
        if rebuild_match:
            rebuild_total = max(rebuild_total, int(rebuild_match.group(1)))
            boundaries.append({"line": line_number, "reason": "rebuild"})
            rebuild_lines.append(line_number)

        counter_match = COUNTERS_RE.search(line)
        if not counter_match:
            continue
        nan, mag, rebuilds = map(int, counter_match.groups())
        rebuild_total = max(rebuild_total, rebuilds)
        if line.startswith("run complete "):
            reported = {"nan": nan, "mag": mag}
        samples.append((line_number, nan, mag, rebuilds))

    segments: list[dict[str, int]] = []
    current_max = {"nan": 0, "mag": 0}
    previous: tuple[int, int, int, int] | None = None
    for line_number, nan, mag, rebuilds in samples:
        decreases = previous is not None and (nan < previous[1] or mag < previous[2])
        rebuild_before_sample = previous is not None and any(
            previous[0] < rebuild_line <= line_number for rebuild_line in rebuild_lines
        )
        if decreases or rebuild_before_sample:
            if decreases and not rebuild_before_sample:
                boundaries.append({
                    "line": line_number,
                    "reason": "counter_decrease",
                    "previous": {"nan": previous[1], "mag": previous[2]},
                    "current": {"nan": nan, "mag": mag},
                })
            segments.append(current_max)
            current_max = {"nan": 0, "mag": 0}
        current_max["nan"] = max(current_max["nan"], nan)
        current_max["mag"] = max(current_max["mag"], mag)
        previous = (line_number, nan, mag, rebuilds)
    if samples:
        segments.append(current_max)

    return {
        "run_tag": _run_tag(path, text),
        "reported": reported,
        "recounted_lower_bound": {
            "nan": sum(segment["nan"] for segment in segments),
            "mag": sum(segment["mag"] for segment in segments),
        },
        "rebuilds": rebuild_total,
        "reset_boundaries": sorted(boundaries, key=lambda boundary: boundary["line"]),
    }


def _resolve_log(run_tag: str) -> Path:
    path = REPORTS_DIR / f"p1_sprint_train_{run_tag}.log"
    if not path.is_file():
        raise FileNotFoundError(f"No sprint training log for run tag {run_tag!r}: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run-tag")
    selection.add_argument("--log", type=Path)
    selection.add_argument("--all-sprint", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit machine-readable recount data")
    args = parser.parse_args(argv)

    if args.all_sprint:
        results = [recount_log(path) for path in sorted(REPORTS_DIR.glob("p1_*train*.log"))]
        output: Any = results
    else:
        output = recount_log(_resolve_log(args.run_tag) if args.run_tag else args.log)
    if args.json:
        print(json.dumps(output, sort_keys=True))
    elif isinstance(output, list):
        for result in output:
            print(f"{result['run_tag']}: {result['recounted_lower_bound']}")
    else:
        print(f"{output['run_tag']}: reported={output['reported']} recounted_lower_bound={output['recounted_lower_bound']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
