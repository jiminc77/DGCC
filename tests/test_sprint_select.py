"""Synthetic contracts for new-run sprint checkpoint selection."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


def _module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(transitions: int, success: float, mean_return: float) -> dict[str, object]:
    return {
        "ckpt": f"/checkpoints/ckpt_{transitions}.pt",
        "ckpt_sha256": "a" * 64,
        "transitions": transitions,
        "success_rate": success,
        "mean_return": mean_return,
        "val_rows": [[True, False]] * 50,
    }


def test_preregistered_selection_tiebreaks_all_three_levels() -> None:
    select = _module("sprint_select_ckpt", "sprint_select_ckpt.py").select_checkpoint
    assert select([_row(10, 0.6, 99.0), _row(20, 0.7, -99.0)])["transitions"] == 20
    assert select([_row(10, 0.7, 2.0), _row(20, 0.7, 3.0)])["transitions"] == 20
    assert select([_row(20, 0.7, 3.0), _row(10, 0.7, 3.0)])["transitions"] == 10


def test_selection_manifest_passes_heldout_consumer_contract(tmp_path: Path, monkeypatch) -> None:
    selector = _module("sprint_select_ckpt_manifest", "sprint_select_ckpt.py")
    heldout = _module("sprint_heldout_eval_manifest", "sprint_heldout_eval.py")
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "ckpt_100.pt"
    config.write_text("task: t2\n")
    checkpoint.write_bytes(b"synthetic checkpoint")
    selected = _row(100, 0.8, 1.0)
    selected["ckpt"] = str(checkpoint)
    selected["ckpt_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = selector.selection_manifest(
        run_tag="synthetic", arm="bb", seed=7, config_path=config, selected=selected,
    )
    output = tmp_path / "selection.json"
    selector.atomic_publish(output, manifest)
    monkeypatch.setattr(heldout, "REPO", tmp_path)
    loaded, digest = heldout.load_selection_manifest(output, "synthetic", "bb", 7, "config.yaml")
    assert loaded == manifest
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
