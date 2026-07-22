#!/usr/bin/env python3
"""Manifest-only CPU entry point for the AMD-2 patching battery.

Rollout execution is intentionally not implemented here; it requires a separately
reviewed patch-forward path and is excluded from this CPU implementation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from dgcc.analysis.sprint_patching import load_probe_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("qrank", "necessity", "graded", "rollout"), required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "rollout":
        parser.error("rollout mode is not implemented in this CPU patching release; separate implementation is planned")
    probes = load_probe_manifest(args.probe_manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"mode": args.mode, "arm": args.arm, "run_tag": args.run_tag, "probe_files": len(probes), "status": "manifest_verified"}, sort_keys=True) + "\n", encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
