#!/usr/bin/env python3
"""Drive the real governed launcher to a constructed agent without training.

The 15/15 manifest preflight authenticated the evidence layer only; it never
called ``scripts/p1_sprint_train.py``, so two allowlist-resolution defects
survived it. An earlier version of this probe stopped at
``AttemptRegistry.recover``, which is still before agent construction, so it
could not observe the launcher/factory keyword mismatch that killed run 1 or
the missing ``t2_split`` allowlist role behind it. Each boundary certified only
what preceded it.

Boundary now: ``TrainingRun.collect_round``, the first statement of the
training loop. Everything before it is real -- asset-gate resolution, protocol
receipt binding, config parsing through the firewall, the factory call, agent
construction, the initial-weights digest, every allowlist role the driver needs
including ``t2_split``, and the entire simulator bring-up in ``build_scene``:
Genesis init, ``DLOLabEnv`` construction, the first ``env.reset``, the per-env
grasp-hook check, the batched runner, and the first training episodes. Nothing
after it runs: no transition, no update, no evaluation.

``--device-mode cpu`` keeps the older pre-simulator boundary for hosts with no
accelerator and says so in the receipt; it cannot certify the simulator,
because Genesis initializes with ``backend=gs.gpu`` unconditionally.

Three seams are patched, none of them in the code under test:

* ``AttemptRegistry`` is re-rooted at a scratch directory so probing never
  writes to the production attempt registry. The registry implementation
  itself is the real one; only its root moves. The evidence generator's own
  registry smoke uses the same redirection.
* ``read_launch_asset_snapshot`` and ``AssetFirewall.read_bytes`` are wrapped
  with recorders that delegate to the real implementations and return their
  exact results, so the receipt can name every role the launcher resolved.
* ``TrainingRun.run`` raises a sentinel on entry. The agent is released
  immediately afterwards.

GPU: ``--device-mode cpu`` requires ``CUDA_VISIBLE_DEVICES=""`` so the
production argv resolves to cpu untouched. ``--device-mode gpu`` requires a
visible, usable accelerator and is the only mode that reaches the simulator.
Either way the argv handed to the launcher is the production argv.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any


class AgentConstructed(BaseException):
    """Raised on entry to TrainingRun.run; not an error."""


# Charter section 9 mandatory per-validation diagnostics, grouped as the charter
# enumerates them. Grounded in what a completed 300k run actually emitted, not
# invented from the prose.
MANDATORY_SELECTION_DIAGNOSTICS = {
    "weight_entropy": ("contact_weight_entropy",),
    "n_eff": ("contact_weight_neff",),
    "top1_mass": ("contact_weight_top1_mass",),
    "argmax_agreement": ("q1_qmin_argmax_agreement",),
    "rank_tau": ("q1_qmin_rank_tau",),
    "margin": ("q1_top1_top2_margin", "qmin_top1_top2_margin"),
    "churn": (
        "hard_q1_churn",
        "hard_qmin_churn",
        "soft_weight_cosine_to_previous_checkpoint",
        "soft_weight_js_to_previous_checkpoint",
        "top8_contact_overlap",
    ),
    "lift": (
        "lift_flip_rate_under_plusminus_002",
        "lift_near_threshold_all_045_055",
        "lift_near_threshold_selected_045_055",
        "q_at_lift_1_minus_q_at_lift_0",
        "q_continuous_lift_minus_q_hard_lift",
        "selected_lift_entropy",
    ),
}
# Churn is defined against the previous checkpoint, so it is legitimately
# undefined at the first validation of a run. Every other group must be present
# and non-null at every validation.
FIRST_VALIDATION_EXEMPT = ("churn",)


def audit_section_9(summary: dict[str, Any]) -> dict[str, Any]:
    """Check that a completed run actually emitted every mandated diagnostic.

    B6 was invisible to every earlier boundary because the probe stopped at
    "training runs" and never asked "did the mandated diagnostics get written".
    A run can be numerically perfect and still be charter-noncompliant.
    """
    findings: list[dict[str, Any]] = []
    evals = summary.get("evals") or []
    for index, record in enumerate(evals):
        diagnostics = record.get("selection_diagnostics") or {}
        histogram_bins = sum(1 for key in diagnostics if "histogram" in key)
        if histogram_bins == 0:
            findings.append(
                {"validation": record.get("transitions"), "group": "histogram",
                 "missing": ["contact_histogram_count_*"]}
            )
        for group, keys in MANDATORY_SELECTION_DIAGNOSTICS.items():
            if index == 0 and group in FIRST_VALIDATION_EXEMPT:
                continue
            missing = [
                key for key in keys
                if key not in diagnostics or diagnostics[key] is None
            ]
            if missing:
                findings.append(
                    {"validation": record.get("transitions"), "group": group,
                     "missing": missing}
                )
    counterfactual = summary.get("counterfactual_diagnostic") or {}
    return {
        "validations_observed": len(evals),
        "selection_diagnostics_complete": not findings,
        "selection_diagnostic_gaps": findings,
        "counterfactual_status": counterfactual.get("status"),
        "counterfactual_error": counterfactual.get("error"),
        "v2_downstream_ready": summary.get("v2_downstream_ready"),
        "section_9_4_passed": (
            counterfactual.get("status") == "ok"
            and summary.get("v2_downstream_ready") is True
        ),
    }


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
    parser.add_argument(
        "--device-mode", choices=("cpu", "gpu"), default="cpu",
        help="cpu stops before the simulator; gpu runs through build_scene",
    )
    parser.add_argument("--runtime-environment", type=Path, default=None)
    parser.add_argument(
        "--verify-diagnostics", type=int, default=None, metavar="TRANSITIONS",
        help=(
            "run the governed launch to completion at this transition budget in "
            "a scratch registry and assert every charter section 9 diagnostic "
            "was actually written; the post-loop final eval guarantees exactly "
            "one validation at any budget"
        ),
    )
    args = parser.parse_args(argv)

    if args.device_mode == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise RuntimeError(
            'cpu-mode dry-run requires CUDA_VISIBLE_DEVICES="" so the production '
            "argv resolves to cpu without being modified"
        )
    if args.device_mode == "gpu" and os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        raise RuntimeError(
            "gpu-mode dry-run requires a visible accelerator; the simulator "
            "initializes with backend=gs.gpu and cannot run on cpu"
        )
    # Keep the pinned runtime tree byte-identical to the code manifest.
    sys.dont_write_bytecode = True
    repo = args.repo_root.resolve(strict=True)
    sys.path.insert(0, str(repo / "src"))

    firewall_module = importlib.import_module("dgcc.logging.asset_firewall")
    registry_module = importlib.import_module("dgcc.logging.attempt_registry")
    import torch

    if args.device_mode == "gpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "gpu-mode dry-run requires torch.cuda.is_available(); this "
            "environment has no usable accelerator"
        )

    resolved: list[dict[str, Any]] = []

    def record(role: str, requested: Any, path: Any, payload: bytes) -> None:
        entry = {
            "role": role,
            "requested_path": str(requested),
            "resolved_path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        if entry not in resolved:
            resolved.append(entry)

    real_snapshot = firewall_module.read_launch_asset_snapshot

    def recording_snapshot(manifest_path, expected_manifest_sha256, asset_path, expected_role):
        path, payload = real_snapshot(
            manifest_path, expected_manifest_sha256, asset_path, expected_role
        )
        record(expected_role, asset_path, path, payload)
        return path, payload

    firewall_module.read_launch_asset_snapshot = recording_snapshot

    real_read_bytes = firewall_module.AssetFirewall.read_bytes

    def recording_read_bytes(self, path, operation="read", *, required_role=None):
        canonical, payload = real_read_bytes(
            self, path, operation=operation, required_role=required_role
        )
        if required_role is not None:
            record(required_role, path, canonical, payload)
        return canonical, payload

    firewall_module.AssetFirewall.read_bytes = recording_read_bytes

    scratch = Path(tempfile.mkdtemp(prefix="v2-launch-dryrun-"))
    registry_class = registry_module.AttemptRegistry
    real_registry_init = registry_class.__init__
    real_recover = registry_class.recover

    def scratch_init(self, root, **kwargs):
        real_registry_init(self, scratch / "attempts", **kwargs)

    agent_evidence: dict[str, Any] = {}
    scene_evidence: dict[str, Any] = {}

    def scratch_recover(cls, root, **kwargs):
        # Called once, after the driver module exists and after
        # SprintTrainingRun is defined, but before any registry or agent
        # exists. That makes it the only place the remaining seams can be
        # patched without touching the code under test.
        driver = sys.modules["_sprint_p1_train"]
        real_digest = driver.initial_weights_sha256

        def capture_digest(agent):
            value = real_digest(agent)
            agent_evidence.update(
                {
                    "agent_class": type(agent).__name__,
                    "agent_module": type(agent).__module__,
                    "device": str(agent.device),
                    "policy_delay": int(agent.config.policy_delay),
                    "parameter_tensors": sum(
                        1
                        for module in (
                            agent.encoder,
                            agent.critic,
                            agent.actor,
                        )
                        for _ in module.state_dict()
                    ),
                    "probe_initial_weights_sha256": value,
                }
            )
            return value

        driver.initial_weights_sha256 = capture_digest

        if args.device_mode == "gpu" and args.verify_diagnostics is None:
            # First statement of the training loop: build_scene has completed,
            # no transition has been collected.
            def blocked_collect(self):
                scene_evidence.update(
                    {
                        "env_class": type(self.env).__name__,
                        "n_envs": int(self.n_envs),
                        "runner_class": type(self.runner).__name__,
                        "device": str(self.device),
                        "episode_index": int(self.episode_index),
                        "transitions": int(self.transitions),
                    }
                )
                raise AgentConstructed(
                    "scene built and first training episodes begun; "
                    "halting before the first collect_round"
                )

            driver.TrainingRun.collect_round = blocked_collect
        else:
            def blocked_run(self):
                raise AgentConstructed(
                    "agent constructed and driver fully initialized; "
                    "halting before build_scene"
                )

            driver.TrainingRun.run = blocked_run
        return real_recover(scratch / "attempts", **kwargs)

    registry_class.__init__ = scratch_init
    registry_class.recover = classmethod(scratch_recover)

    launch_argv = [
        "--config", str(args.config),
        "--arm", args.arm,
        "--seed", str(args.seed),
        "--asset-manifest", str(args.asset_manifest),
        "--expected-asset-manifest-sha256", args.expected_asset_manifest_sha256,
    ]
    if args.verify_diagnostics is not None:
        launch_argv += ["--total-override", str(args.verify_diagnostics)]
    receipt: dict[str, Any] = {
        "schema_version": 2,
        "probe": "real-launcher-agent-construction-dry-run",
        "launcher_path": str(args.launcher.resolve(strict=True)),
        "launcher_sha256": hashlib.sha256(args.launcher.read_bytes()).hexdigest(),
        "argv": launch_argv,
        "arm": args.arm,
        "seed": args.seed,
        "boundary": (
            "p1_train.TrainingRun.collect_round entry, after build_scene"
            if args.device_mode == "gpu"
            else "p1_train.TrainingRun.run entry, before build_scene"
        ),
        "device_mode": args.device_mode,
        "simulator_verified": args.device_mode == "gpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
        "attempt_registry_root": "scratch (production registry untouched)",
        "gpu_used": args.device_mode == "gpu",
        "training_started": False,
        "transitions_executed": 0,
        "attempt_registry_written": False,
    }
    if args.runtime_environment is not None:
        pinned = json.loads(args.runtime_environment.read_bytes())
        receipt["runtime_environment"] = {
            "sha256": hashlib.sha256(
                args.runtime_environment.read_bytes()
            ).hexdigest(),
            "torch_matches_pin": (
                torch.__version__ == pinned["accelerator"]["torch_version"]
            ),
            "lockfile_digest_sha256": pinned["lockfile_digest_sha256"],
            "genesis_commit": pinned["simulator"]["commit"],
        }
    try:
        launcher = load_module(args.launcher, "_v2_launch_dryrun_launcher")
        try:
            launcher.main(launch_argv)
        except AgentConstructed:
            receipt["agent_constructed"] = True
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
            if args.verify_diagnostics is None:
                receipt["gate_passed"] = False
                receipt["failure"] = (
                    "launcher returned without reaching the boundary"
                )
            else:
                # Completion is the expected outcome in diagnostics mode. The
                # run must be read out of the scratch registry before cleanup.
                summaries = sorted(
                    (scratch / "attempts").glob("*/metrics/run_summary.json")
                )
                if len(summaries) != 1:
                    receipt["gate_passed"] = False
                    receipt["failure"] = (
                        f"expected exactly one scratch run summary, found "
                        f"{len(summaries)}"
                    )
                else:
                    summary = json.loads(summaries[0].read_bytes())
                    audit = audit_section_9(summary)
                    attempt_dir = summaries[0].parent.parent
                    audit["counterfactual_artifacts"] = len(
                        list((attempt_dir / "diagnostics").glob("*.npz"))
                    )
                    audit["transitions"] = summary.get("transitions")
                    receipt["section_9_audit"] = audit
                    receipt["training_started"] = True
                    receipt["transitions_executed"] = summary.get("transitions")
                    passed = (
                        audit["section_9_4_passed"]
                        and audit["selection_diagnostics_complete"]
                        and audit["counterfactual_artifacts"] >= 1
                        and audit["validations_observed"] >= 1
                    )
                    receipt["gate_passed"] = passed
                    receipt["failure"] = None if passed else (
                        "charter section 9 diagnostics incomplete: "
                        f"9.4={audit['counterfactual_status']}, "
                        f"gaps={audit['selection_diagnostic_gaps']}"
                    )
    finally:
        # Release the agent and every driver object as soon as the boundary is
        # reached; nothing constructed here outlives the probe.
        gc.collect()
        shutil.rmtree(scratch, ignore_errors=True)

    receipt.setdefault("agent_constructed", bool(agent_evidence))
    receipt["agent"] = agent_evidence or None
    receipt["scene"] = scene_evidence or None
    receipt["resolved_assets"] = resolved
    receipt["resolved_roles"] = sorted({entry["role"] for entry in resolved})

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
