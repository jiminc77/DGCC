#!/usr/bin/env python3
"""Sprint training entry point: reuse the P1 driver with an explicit agent factory."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
GOVERNED_ARMS = frozenset({"bb-d2", "v1-d2", "v2-dmm", "v2-d1m", "v2-d11", "v2-bgt"})
GOVERNED_SCHEDULE_ARMS = {
    "bb-d2": "BB-D2",
    "v1-d2": "V1-D2",
    "v2-dmm": "DMM",
    "v2-d1m": "D1M",
    "v2-d11": "D11",
    "v2-bgt": "BGT",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest(path: Path) -> dict[str, str]:
    try:
        return {
            source: digest
            for digest, source in (line.split("  ", 1) for line in path.read_text().splitlines() if line)
        }
    except ValueError as error:
        raise RuntimeError("frozen bundle manifest is malformed") from error


def validate_source_bundle(bundle: Path) -> dict[str, Any]:
    """Authenticate a frozen BB bundle against the committed parity proof."""
    manifest_path = bundle / "MANIFEST.sha256"
    metadata_path = bundle / "bundle_metadata.json"
    proof_path = ROOT / "outputs/metrics/sprint_bb_parity_proof.json"
    if not manifest_path.is_file() or not metadata_path.is_file():
        raise RuntimeError("source bundle requires MANIFEST.sha256 and bundle_metadata.json")
    try:
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("source bundle proof or metadata is malformed") from error
    if proof.get("verdict") != "PASS":
        raise RuntimeError("source bundle parity proof verdict is not PASS")
    source_commit = metadata.get("source_commit")
    closure_blobs = proof.get("closure_blobs")
    if not isinstance(source_commit, str) or not isinstance(closure_blobs, dict):
        raise RuntimeError("source bundle proof or metadata lacks source commit")
    expected_blobs = closure_blobs.get(source_commit)
    if not isinstance(expected_blobs, dict):
        raise RuntimeError("source bundle source_commit is not authenticated by parity proof")
    source_blobs = metadata.get("source_blobs")
    if source_blobs != expected_blobs:
        raise RuntimeError("source bundle metadata source_blobs disagrees with parity proof")
    manifest = read_manifest(manifest_path)
    if set(manifest) != set(expected_blobs):
        raise RuntimeError("frozen bundle manifest disagrees with parity proof")
    expected_files = set(expected_blobs) | {"MANIFEST.sha256", "bundle_metadata.json"}
    actual_files = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()}
    if actual_files != expected_files:
        raise RuntimeError("frozen bundle file set disagrees with parity proof")
    for relative, expected_blob in expected_blobs.items():
        source = bundle / relative
        if not source.is_file():
            raise RuntimeError(f"frozen bundle source is missing: {relative}")
        blob = subprocess.run(
            ["git", "hash-object", str(source)], check=True, capture_output=True, text=True
        ).stdout.strip()
        if blob != expected_blob:
            raise RuntimeError(f"frozen bundle proof blob mismatch: {relative}")
        if sha256_file(source) != manifest[relative]:
            raise RuntimeError(f"frozen bundle digest mismatch: {relative}")
    return {
        "sha256": sha256_file(manifest_path),
        "source_commit": source_commit,
        "proof_sha256": sha256_file(proof_path),
    }


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_base_driver(bundle: Path | None) -> ModuleType:
    path = (bundle / "scripts/p1_train.py") if bundle else ROOT / "scripts/p1_train.py"
    return load_module(path, "_sprint_p1_train")


def load_factory(bundle: Path | None = None) -> Any:
    if bundle is None:
        from dgcc.rl.sprint_arms import create_sprint_agent

        return create_sprint_agent
    # BB bundles predate the adapter; load the current adapter while its
    # absolute dgcc imports resolve to the already-loaded frozen package.
    return load_module(ROOT / "src/dgcc/rl/sprint_arms.py", "_sprint_arms").create_sprint_agent


def create_seeded_agent(
    factory: Any,
    arm: str,
    config: Any,
    reward_constants: Any,
    seed: int,
    device: str,
    aux_weight: float,
    projection_seed: int = 20260719,
    target_seed: int = 20260718,
    candidate_kwargs: dict[str, Any] | None = None,
) -> Any:
    """F-a construction seam: seed precedes the sole retained agent creation."""
    torch.manual_seed(seed)
    return factory(
        arm,
        config,
        device=device,
        reward_constants=reward_constants,
        aux_weight=aux_weight,
        projection_seed=projection_seed,
        target_seed=target_seed,
        **(candidate_kwargs or {}),
    )


def assert_bundle_modules(bundle: Path) -> dict[str, str]:
    prefix = (bundle / "src").resolve()
    origins = {
        name: str(Path(module.__file__).resolve())
        for name, module in sys.modules.items()
        if name == "dgcc" or name.startswith("dgcc.")
        if getattr(module, "__file__", None)
    }
    if not origins or any(not Path(origin).is_relative_to(prefix) for origin in origins.values()):
        raise AssertionError("BB source-bundle mode imported dgcc outside the frozen bundle")
    return origins


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P1 sprint training driver")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--arm",
        choices=("bb", "bb-d2", "v1", "v1-d2", "matched", "random", "v2-dmm", "v2-d1m", "v2-d11", "v2-bgt"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--total-override", type=int, default=None)
    parser.add_argument("--asset-manifest", type=Path)
    parser.add_argument("--expected-asset-manifest-sha256", type=str)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if args.source_bundle and args.arm != "bb":
        parser.error("--source-bundle is permitted only for the legacy bb arm")
    if args.arm in GOVERNED_ARMS and (
        args.asset_manifest is None or args.expected_asset_manifest_sha256 is None
    ):
        parser.error("governed launches require --asset-manifest and --expected-asset-manifest-sha256")
    if args.source_bundle:
        # Keep the frozen bundle byte-pristine: without this, importing bundle
        # modules writes __pycache__ into the bundle tree and the exact-tree
        # fail-closed validation refuses every subsequent launch.
        sys.dont_write_bytecode = True
    bundle_info = validate_source_bundle(args.source_bundle) if args.source_bundle else None
    base = load_base_driver(args.source_bundle)
    factory = load_factory(args.source_bundle)
    if args.source_bundle:
        bundle_origins = assert_bundle_modules(args.source_bundle)
    else:
        bundle_origins = None
    if args.arm in GOVERNED_ARMS:
        _, config_bytes = base.read_launch_asset_snapshot(
            args.asset_manifest,
            args.expected_asset_manifest_sha256,
            args.config,
            "config",
        )
        config = yaml.safe_load(config_bytes)
    else:
        config_bytes = None
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    sprint_cfg = config.get("sprint", {})
    if sprint_cfg.get("arm") and sprint_cfg["arm"] != args.arm:
        parser.error("config sprint.arm must match --arm")
    aux_weight = float(sprint_cfg.get("aux_weight", 1.0))
    projection_seed = int(sprint_cfg.get("projection_seed", 20260719))
    target_seed = int(sprint_cfg.get("target_seed", 20260718))
    beta_contact = float(sprint_cfg.get("beta_contact", 0.015363))
    if args.arm in GOVERNED_ARMS and beta_contact != 0.015363:
        parser.error("governed launches require sprint.beta_contact == 0.015363")
    amd5_preflight = sprint_cfg.get("amd5_preflight")
    preflight_receipt = None
    bgt_manifest_bytes = None
    if args.arm in GOVERNED_ARMS:
        manifest_value = (
            amd5_preflight.get("manifest_path") if isinstance(amd5_preflight, dict) else amd5_preflight
        )
        code_manifest_value = (
            amd5_preflight.get("code_manifest_path") if isinstance(amd5_preflight, dict) else None
        )
        neff_guard_value = (
            amd5_preflight.get("neff_guard_path") if isinstance(amd5_preflight, dict) else None
        )
        if not manifest_value or not code_manifest_value or not neff_guard_value:
            parser.error(
                "governed launches require sprint.amd5_preflight manifest_path, "
                "code_manifest_path, and neff_guard_path"
            )
        manifest_path = Path(manifest_value)
        code_manifest_path = Path(code_manifest_value)
        neff_guard_path = Path(neff_guard_value)
        if not manifest_path.is_absolute():
            manifest_path = args.config.parent / manifest_path
        if not code_manifest_path.is_absolute():
            code_manifest_path = args.config.parent / code_manifest_path
        if not neff_guard_path.is_absolute():
            neff_guard_path = args.config.parent / neff_guard_path
        preflight = load_module(
            ROOT / "scripts/v2_protocol_preflight.py", "_v2_protocol_preflight"
        )
        governance_value = (
            amd5_preflight.get("governance_path", preflight.DEFAULT_GOVERNANCE)
            if isinstance(amd5_preflight, dict)
            else preflight.DEFAULT_GOVERNANCE
        )
        governance_path = Path(governance_value)
        if not governance_path.is_absolute():
            governance_path = args.config.parent / governance_path
        try:
            _, manifest_bytes = base.read_launch_asset_snapshot(
                args.asset_manifest,
                args.expected_asset_manifest_sha256,
                manifest_path,
                "preflight_manifest",
            )
            _, governance_bytes = base.read_launch_asset_snapshot(
                args.asset_manifest,
                args.expected_asset_manifest_sha256,
                governance_path,
                "execution_governance",
            )
            _, code_manifest_bytes = base.read_launch_asset_snapshot(
                args.asset_manifest,
                args.expected_asset_manifest_sha256,
                code_manifest_path,
                "code_manifest",
            )
            _, neff_guard_bytes = base.read_launch_asset_snapshot(
                args.asset_manifest,
                args.expected_asset_manifest_sha256,
                neff_guard_path,
                "neff_guard",
            )
            config_sha256 = hashlib.sha256(config_bytes).hexdigest()
            code_manifest_sha256 = hashlib.sha256(code_manifest_bytes).hexdigest()
            neff_guard_sha256 = hashlib.sha256(neff_guard_bytes).hexdigest()
            preflight_receipt = preflight.validate_manifest_bytes(
                manifest_bytes,
                governance_bytes,
                expected_arm=args.arm,
                expected_seed=args.seed,
                expected_config_sha256=config_sha256,
                expected_code_manifest_sha256=code_manifest_sha256,
                code_manifest_bytes=code_manifest_bytes,
                neff_guard_bytes=neff_guard_bytes,
                runtime_root=ROOT,
            )
        except (base.AssetAccessError, OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(f"AMD-5/V2 preflight failed: {error}")
        medians = (
            preflight_receipt.get("q1_pooled_median"),
            preflight_receipt.get("qmin_pooled_median"),
        )
        if (
            preflight_receipt.get("arm") != args.arm
            or preflight_receipt.get("seed") != args.seed
            or preflight_receipt.get("schedule_arm") != GOVERNED_SCHEDULE_ARMS[args.arm]
            or preflight_receipt.get("config_sha256") != config_sha256
            or preflight_receipt.get("code_manifest_sha256") != code_manifest_sha256
            or preflight_receipt.get("neff_guard_sha256") != neff_guard_sha256
            or preflight_receipt.get("guard_passed") is not True
            or preflight_receipt.get("neff_guard_passed") is not True
            or any(
                type(median) not in (int, float)
                or not math.isfinite(median)
                or not 8.0 <= median <= 20.0
                for median in medians
            )
            or preflight_receipt.get("code_closure_count")
            != len(preflight.REQUIRED_RUNTIME_FILES)
            or not isinstance(preflight_receipt.get("code_closure_sha256"), str)
        ):
            parser.error("AMD-5/V2 preflight receipt does not bind this governed launch")
    if args.arm in GOVERNED_ARMS and config.get("td3", {}).get("policy_delay") != 2:
        parser.error("governed arms require explicit td3.policy_delay == 2")
    bgt_cfg = sprint_cfg.get("bgt", {})
    if args.arm == "v2-bgt":
        bgt_manifest_value = bgt_cfg.get("manifest_path")
        if not bgt_manifest_value or not bgt_cfg.get("expected_manifest_sha256"):
            parser.error(
                "BGT launches require bgt.manifest_path and expected_manifest_sha256"
            )
        bgt_manifest_path = Path(bgt_manifest_value)
        if not bgt_manifest_path.is_absolute():
            bgt_manifest_path = args.config.parent / bgt_manifest_path
        try:
            _, bgt_manifest_bytes = base.read_launch_asset_snapshot(
                args.asset_manifest,
                args.expected_asset_manifest_sha256,
                bgt_manifest_path,
                "bgt_admission_manifest",
            )
        except (base.AssetAccessError, OSError, ValueError) as error:
            parser.error(f"BGT admission asset failed: {error}")
    candidate_kwargs = {
        "beta_contact": beta_contact,
        "bgt_manifest_bytes": bgt_manifest_bytes,
        "bgt_expected_manifest_sha256": bgt_cfg.get("expected_manifest_sha256"),
        "bgt_checkpoint_sha256": bgt_cfg.get("checkpoint_sha256"),
        "bgt_panel_sha256": bgt_cfg.get("panel_sha256"),
        "bgt_code_manifest_sha256": code_manifest_sha256 if args.arm == "v2-bgt" else None,
    }
    if args.arm == "v2-bgt" and (
        preflight_receipt is None
        or bgt_cfg.get("expected_manifest_sha256")
        != preflight_receipt.get("admitted_manifest_sha256")
        or bgt_cfg.get("code_manifest_sha256") != code_manifest_sha256
    ):
        parser.error(
            "BGT manifest and code-manifest pins must match independently governed preflight pins"
        )

    effective_config = copy.deepcopy(config)
    effective_config.setdefault("eval", {}).update(sprint_cfg.get("eval", {}))

    class SprintTrainingRun(base.TrainingRun):
        def __init__(
            self,
            run_args: argparse.Namespace,
            registry: Any = None,
            run_config: dict[str, Any] | None = None,
        ) -> None:
            super().__init__(run_args, registry, config=run_config)
            if self.task == "t2":
                val_pairs = base.load_t2_split_payload(
                    "val", self._development_split_payload
                )
                episodes_per_goal = self.config.get("eval", {}).get(
                    "t2_episodes_per_goal", 2
                )
                self.val_labels, self.val_goals = base.expand_t2_validation_pairs(
                    val_pairs, episodes_per_goal
                )

        def create_agent(self) -> Any:
            return create_seeded_agent(
                factory,
                args.arm,
                self.agent_config,
                self.episode_config.reward,
                self.seed,
                self.device,
                aux_weight,
                projection_seed,
                target_seed,
                candidate_kwargs,
            )

        def save_run_summary(self) -> None:
            super().save_run_summary()
            path = self.output_dir / "metrics" / "run_summary.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["sprint"] = {
                "arm": args.arm,
                "aux_weight": aux_weight,
                "amd5_preflight": amd5_preflight,
                "preflight_receipt": preflight_receipt,
                "agent": self.agent.to_dict().get("v2_arm"),
            }
            if bundle_info:
                payload["source_bundle"] = {**bundle_info, "module_origins": bundle_origins}
            path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    run_tag = args.run_tag or f"{effective_config['task']}_{args.arm}_s{args.seed}"
    registry_root = ROOT / "outputs/attempts"
    base.AttemptRegistry.recover(registry_root)
    registry = base.AttemptRegistry(
        registry_root,
        run_tag=run_tag,
        config=effective_config,
        code_sha256=base.sha256_file(Path(__file__).resolve()),
        seed=args.seed,
        governed_launch_receipt=preflight_receipt,
    )
    log_path = registry.attempt_path / "reports" / "p1_sprint_train.log"
    original_stdout = sys.stdout
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            sys.stdout = base.Tee(original_stdout, log_file)
            try:
                run = SprintTrainingRun(args, registry, effective_config)
                exit_code = run.run()
            finally:
                sys.stdout = original_stdout
                log_file.flush()
                os.fsync(log_file.fileno())
    except KeyboardInterrupt as error:
        try:
            registry.finalize_once("ABORTED", detail=str(error))
        except BaseException as finalization_error:
            print(
                "attempt finalization failed after KeyboardInterrupt: "
                f"{finalization_error}",
                file=sys.stderr,
            )
        raise
    except BaseException as error:
        try:
            registry.finalize_once(
                "TECHNICAL_FAILURE", detail=f"{type(error).__name__}: {error}"
            )
        except BaseException as finalization_error:
            print(
                f"attempt finalization failed after {type(error).__name__}: "
                f"{finalization_error}",
                file=sys.stderr,
            )
        raise
    registry.finalize_once(
        "SUCCEEDED" if exit_code == 0 else "TECHNICAL_FAILURE",
        exit_code=exit_code,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
