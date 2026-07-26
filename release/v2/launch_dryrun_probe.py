#!/usr/bin/env python3
"""Drive the real governed launcher through its asset gate without starting a run.

The 15/15 manifest preflight authenticated the evidence layer only; it never called
``scripts/p1_sprint_train.py``, so two allowlist-resolution defects survived it. This
probe closes that hole by executing the launcher's own ``main()`` on the exact
production argv and stopping at the first side-effecting statement past the gate.

Boundary: ``AttemptRegistry.recover`` is the first call after every asset resolution
and receipt binding check (p1_sprint_train.py:194-354) and the first statement that
touches ``outputs/attempts``. Raising there proves the gate passed while guaranteeing
no attempt record, no agent, no environment, no GPU context, and no training step.

The launcher's bytes are never modified: the probe patches only the shared
``AttemptRegistry.recover`` seam and wraps ``read_launch_asset_snapshot`` with a
recorder that delegates to the real implementation and returns its exact result.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


class GateReached(BaseException):
    """Raised at the first post-gate side effect; not an error."""


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--expected-asset-manifest-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)

    # Keep the pinned runtime tree byte-identical to the code manifest.
    sys.dont_write_bytecode = True
    repo = args.repo_root.resolve(strict=True)
    sys.path.insert(0, str(repo / "src"))

    firewall_module = importlib.import_module("dgcc.logging.asset_firewall")
    registry_module = importlib.import_module("dgcc.logging.attempt_registry")

    resolved: list[dict[str, Any]] = []
    real_snapshot = firewall_module.read_launch_asset_snapshot

    def recording_snapshot(manifest_path, expected_manifest_sha256, asset_path, expected_role):
        path, payload = real_snapshot(
            manifest_path, expected_manifest_sha256, asset_path, expected_role
        )
        resolved.append(
            {
                "role": expected_role,
                "requested_path": str(asset_path),
                "resolved_path": str(path),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        return path, payload

    firewall_module.read_launch_asset_snapshot = recording_snapshot

    def blocked_recover(cls, *unused_args, **unused_kwargs):
        raise GateReached("governed launch gate cleared; halting before attempt registry")

    registry_module.AttemptRegistry.recover = classmethod(blocked_recover)

    launch_argv = [
        "--config", str(args.config),
        "--arm", args.arm,
        "--seed", str(args.seed),
        "--asset-manifest", str(args.asset_manifest),
        "--expected-asset-manifest-sha256", args.expected_asset_manifest_sha256,
    ]
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "probe": "real-launcher-gate-dry-run",
        "launcher_path": str(args.launcher.resolve(strict=True)),
        "launcher_sha256": hashlib.sha256(args.launcher.read_bytes()).hexdigest(),
        "argv": launch_argv,
        "arm": args.arm,
        "seed": args.seed,
        "boundary": "dgcc.logging.attempt_registry.AttemptRegistry.recover",
        "gpu_used": False,
        "training_started": False,
        "attempt_registry_written": False,
    }
    launcher = load_module(args.launcher, "_v2_launch_dryrun_launcher")
    try:
        launcher.main(launch_argv)
    except GateReached:
        receipt["gate_passed"] = True
        receipt["failure"] = None
    except SystemExit as error:
        receipt["gate_passed"] = False
        receipt["failure"] = f"SystemExit({error.code})"
    except BaseException as error:  # noqa: BLE001 - the probe reports, never masks
        receipt["gate_passed"] = False
        receipt["failure"] = f"{type(error).__name__}: {error}"
        receipt["traceback"] = traceback.format_exc()
    else:
        receipt["gate_passed"] = False
        receipt["failure"] = "launcher returned without reaching the gate boundary"
    receipt["resolved_assets"] = resolved
    receipt["resolved_roles"] = sorted(entry["role"] for entry in resolved)

    payload = json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8") + b"\n"
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        args.receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444
    )
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return 0 if receipt["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
