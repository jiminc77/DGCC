"""CPU fixed-panel diagnostics and counterfactual logging contracts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from dgcc.rl.td3 import TD3Agent, TD3Config

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "v2_selection_panel", ROOT / "scripts/v2_selection_panel.py"
)
assert SPEC is not None and SPEC.loader is not None
PANEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PANEL)


def curve(offset: float) -> np.ndarray:
    t = np.linspace(-0.5, 0.5, 32)
    return np.column_stack((t + offset, 0.1 * np.sin(np.pi * t), np.zeros(32)))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixed_panel_logs_temporal_lift_and_counterfactual_metrics(
    tmp_path: Path,
) -> None:
    agent = TD3Agent(TD3Config(policy_delay=2))
    current = agent.save_checkpoint(tmp_path / "current.pt")
    previous = agent.save_checkpoint(tmp_path / "previous.pt")
    panel = tmp_path / "development_panel.npz"
    np.savez(
        panel,
        X=np.stack([curve(0.0), curve(0.01)]),
        G=np.stack([curve(0.03), curve(0.04)]),
        q1_realized_progress=np.array([0.2, 0.4]),
        qmin_realized_progress=np.array([0.1, 0.3]),
        checkpoint_sha256=np.array(digest(current)),
    )
    output = tmp_path / "diagnostics.json"
    result = PANEL.run(
        argparse.Namespace(
            arm="bb",
            checkpoint=current,
            previous_checkpoint=previous,
            panel=panel,
            output=output,
        )
    )
    assert result["device"] == "cpu"
    assert result["model_state_unchanged"] is True
    assert result["checkpoint_comparison"]["hard_q1_churn"] == 0.0
    assert result["checkpoint_comparison"]["top8_contact_overlap"] == 1.0
    assert result["counterfactual_selector"][
        "q1_minus_qmin_realized_progress_mean"
    ] == pytest.approx(0.1)
    assert len(result["contact_histogram_counts"]) == 32
    assert "lift_near_threshold_all_045_055" in result["selection"]
    assert output.is_file()


def test_fixed_panel_rejects_checkpoint_mismatch(tmp_path: Path) -> None:
    agent = TD3Agent(TD3Config())
    current = agent.save_checkpoint(tmp_path / "current.pt")
    previous = agent.save_checkpoint(tmp_path / "previous.pt")
    panel = tmp_path / "development_panel.npz"
    np.savez(
        panel,
        X=np.stack([curve(0.0)]),
        G=np.stack([curve(0.03)]),
        q1_realized_progress=np.array([0.2]),
        qmin_realized_progress=np.array([0.1]),
        checkpoint_sha256=np.array("0" * 64),
    )
    with pytest.raises(ValueError, match="not generated"):
        PANEL.run(
            argparse.Namespace(
                arm="bb",
                checkpoint=current,
                previous_checkpoint=previous,
                panel=panel,
                output=tmp_path / "out.json",
            )
        )
