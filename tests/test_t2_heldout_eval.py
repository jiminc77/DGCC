"""P2 held-out one-shot evaluator contracts (leakage guard, M4).

Covers: exclusive claim; durability-failure => no permission to evaluate;
exact 200-row expansion; provenance rejection; atomic publication; selection
rule. GPU evaluation itself is exercised in the M4 pipeline, not here.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


heldout = _load("p1_t2_heldout_eval")
select = _load("p1_m4_select_ckpt")


# ---- exclusive claim -------------------------------------------------------

def test_claim_is_exclusive(tmp_path: Path) -> None:
    claim = tmp_path / "claim.json"
    heldout.acquire_heldout_claim(claim, {"run_tag": "x"})
    assert json.loads(claim.read_text())["run_tag"] == "x"
    with pytest.raises(heldout.HeldoutClaimError, match="already exists"):
        heldout.acquire_heldout_claim(claim, {"run_tag": "x"})


def test_claim_durability_failure_denies_permission(tmp_path: Path, monkeypatch) -> None:
    import os as real_os

    def broken_fsync(fd):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(heldout.os, "fsync", broken_fsync)
    with pytest.raises(OSError, match="simulated fsync failure"):
        heldout.acquire_heldout_claim(tmp_path / "claim.json", {"run_tag": "x"})


# ---- exact 200-row expansion (leakage guard: NO heldout load in tests — ----
# ---- HUMAN instruction issue #13 comment 4978169975: code-path validation ---
# ---- via the val split only; the heldout split is loaded exactly once, ----
# ---- at final evaluation, with access logging) ------------------------------

def test_expansion_code_path_via_val_split_only() -> None:
    from dgcc.tasks.t2 import load_t2_split

    pairs = load_t2_split("val")  # 50 goals — validates the code path
    assert len(pairs) == 50
    goals, labels, families = heldout.expand_heldout_goals(pairs)
    assert len(goals) == len(labels) == len(families) == 100
    assert labels[0] == labels[1] and labels[0] != labels[2]  # per-goal pairing


def test_heldout_contract_is_exactly_200_rows_synthetic() -> None:
    # The 100-goal x2 = 200-row contract, proven WITHOUT touching the
    # held-out split: synthetic (spec, goal) pairs of the heldout cardinality.
    synthetic = [({"goal_id": f"g{i:03d}", "family": "s"}, object()) for i in range(100)]
    goals, labels, families = heldout.expand_heldout_goals(synthetic)
    assert len(goals) == 200 and len(labels) == 200 and len(families) == 200


def test_heldout_access_logging_mechanism(tmp_path: Path, monkeypatch) -> None:
    # The audit mechanism itself (no heldout materialization here).
    import dgcc.tasks.t2 as t2

    monkeypatch.chdir(tmp_path)
    log = t2._log_heldout_access(100)
    assert log.exists()
    line = log.read_text().strip()
    assert "n_pairs=100" in line and "pid=" in line


def test_val_split_load_does_not_log(tmp_path: Path, monkeypatch) -> None:
    import dgcc.tasks.t2 as t2

    monkeypatch.chdir(tmp_path)
    t2.load_t2_split("val")
    assert not (tmp_path / "outputs" / "metrics" / "t2_heldout_access.log").exists()


# ---- provenance rejection --------------------------------------------------

def _manifest(tmp_path: Path, ckpt_bytes: bytes = b"weights") -> Path:
    ckpt = tmp_path / "ckpt_0300032.pt"
    ckpt.write_bytes(ckpt_bytes)
    manifest = {
        "run_tag": "m4_t2_s0",
        "selected_ckpt": str(ckpt),
        "ckpt_sha256": heldout.sha256_file(ckpt),
        "selection_rule": select.SELECTION_RULE,
    }
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(manifest))
    return path


def test_manifest_accepts_consistent_provenance(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    manifest = heldout.load_selection_manifest(path)
    assert manifest["run_tag"] == "m4_t2_s0"


def test_manifest_rejects_substituted_checkpoint(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    ckpt = Path(json.loads(path.read_text())["selected_ckpt"])
    ckpt.write_bytes(b"SUBSTITUTED")  # post-selection tamper
    with pytest.raises(heldout.HeldoutClaimError, match="sha256 mismatch"):
        heldout.load_selection_manifest(path)


def test_manifest_rejects_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"run_tag": "x"}))
    with pytest.raises(heldout.HeldoutClaimError, match="missing required field"):
        heldout.load_selection_manifest(path)


# ---- atomic publication -----------------------------------------------------

def test_atomic_publish_creates_final_only(tmp_path: Path) -> None:
    out = tmp_path / "result.json"
    heldout.atomic_publish_json(out, {"ok": True})
    assert json.loads(out.read_text()) == {"ok": True}
    assert not out.with_suffix(".json.tmp").exists()


# ---- selection rule ---------------------------------------------------------

def test_selection_rule_prefers_success_then_return_then_earliest() -> None:
    evals = [
        {"transitions": 100, "success_rate": 0.1, "mean_return": 0.0},
        {"transitions": 200, "success_rate": 0.3, "mean_return": -1.0},
        {"transitions": 300, "success_rate": 0.3, "mean_return": 0.5},
        {"transitions": 400, "success_rate": 0.3, "mean_return": 0.5},
    ]
    best = select.select_best_eval(evals)
    assert best["transitions"] == 300  # max success, max return, earliest


def test_selection_refuses_halted_runs(tmp_path: Path, monkeypatch) -> None:
    run_json = tmp_path / "p1_run_m4_t2_s9.json"
    run_json.write_text(json.dumps({"halt_reason": "TrainingNaNError: x", "evals": []}))
    with pytest.raises(ValueError, match="halted run"):
        select.build_manifest("m4_t2_s9", metrics_dir=tmp_path, models_dir=tmp_path)
