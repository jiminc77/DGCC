#!/usr/bin/env python3
"""Sprint training entry point: reuse the P1 driver with an explicit agent factory."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument("--arm", choices=("bb", "v1", "matched", "random"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--total-override", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if args.source_bundle and args.arm != "bb":
        parser.error("--source-bundle is only valid for arm=bb")
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
    sprint_cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")).get("sprint", {})
    if sprint_cfg.get("arm") and sprint_cfg["arm"] != args.arm:
        parser.error("config sprint.arm must match --arm")
    aux_weight = float(sprint_cfg.get("aux_weight", 1.0))
    projection_seed = int(sprint_cfg.get("projection_seed", 20260719))
    target_seed = int(sprint_cfg.get("target_seed", 20260718))

    class SprintTrainingRun(base.TrainingRun):
        def __init__(self, run_args: argparse.Namespace) -> None:
            # p1_train has no construction hook.  Retain all of its initialization,
            # then replace only its internally-created agent through the public factory.
            super().__init__(run_args)
            # Keep the baseline's eval parser while putting sprint-only flags
            # where its inherited evaluation path consumes them.
            self.config.setdefault("eval", {}).update(sprint_cfg.get("eval", {}))
            self.agent = create_seeded_agent(
                factory,
                args.arm,
                self.agent_config,
                self.episode_config.reward,
                self.seed,
                self.device,
                aux_weight,
                projection_seed,
                target_seed,
            )
            self.initial_weights_sha256 = base.initial_weights_sha256(self.agent)

        def save_run_summary(self) -> None:
            super().save_run_summary()
            path = Path("outputs/metrics") / f"p1_run_{self.run_tag}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["sprint"] = {"arm": args.arm, "aux_weight": aux_weight}
            if bundle_info:
                payload["source_bundle"] = {**bundle_info, "module_origins": bundle_origins}
            path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    run_tag = args.run_tag or f"{yaml.safe_load(args.config.read_text())['task']}_{args.arm}_s{args.seed}"
    log_path = Path("outputs/reports") / f"p1_sprint_train_{run_tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = base.Tee(original_stdout, log_file)
        try:
            return SprintTrainingRun(args).run()
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    raise SystemExit(main())
