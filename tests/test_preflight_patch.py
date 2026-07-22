"""CPU-only checks for patch preflight parsing and report decomposition."""
from __future__ import annotations

import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "goal_preflight", REPO / "scripts" / "goal_preflight.py"
)
assert SPEC and SPEC.loader
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def test_patch_split_has_committed_train_and_ood_lengths() -> None:
    payload, specs, access_recorded = preflight.load_patch_eval_split()

    assert payload["n_goals"] == len(specs) == 100
    assert preflight.patch_eval_lengths(payload) == (0.75, 1.0, 1.25)
    assert access_recorded is False


def test_patch_report_breakdown_is_length_and_family_with_nonconverged() -> None:
    rows = [
        {"rope_length_m": 0.75, "template": "s", "converged": True, "drift_shape": 0.04},
        {"rope_length_m": 0.75, "template": "s", "converged": False, "drift_shape": 0.99},
        {"rope_length_m": 1.25, "template": "u", "converged": True, "drift_shape": 0.06},
    ]
    lines: list[str] = []

    preflight.append_patch_breakdown(lines, rows)

    assert "| 0.75 | s | 1 | 0.0400 | 0.0400 | 0 | 1 |" in lines
    assert "| 1.25 | u | 1 | 0.0600 | 0.0600 | 1 | 0 |" in lines


def test_sprint_split_mode_fail_closed_without_issued_lock(monkeypatch, tmp_path) -> None:
    import json

    import pytest

    from dgcc.analysis import sprint_claims

    # Missing lock: refused at argument boundary, before any environment work.
    monkeypatch.setattr("sys.argv", ["goal_preflight.py", "--include-sprint-split"])
    with pytest.raises(sprint_claims.SprintClaimError, match="requires --lock"):
        preflight.main()
    # Off-path schema-valid lock: refused by the canonical issued-path gate.
    fake = tmp_path / "lock.json"
    fake.write_text(json.dumps({"schema_version": 1}))
    monkeypatch.setattr("sys.argv", ["goal_preflight.py", "--include-sprint-split", "--lock", str(fake)])
    with pytest.raises(sprint_claims.SprintClaimError, match="canonical issued lock path"):
        preflight.main()
