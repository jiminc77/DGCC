"""P1b smoke-only dense round logging (P1_LOG_EVERY_ROUND), M4-prep.

Default off; "1" enables; any other value stays off. Main lanes launch with
`env -u P1_LOG_EVERY_ROUND` and assert zero roundlog lines post-run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

_SPEC = importlib.util.spec_from_file_location("p1_train", _REPO / "scripts" / "p1_train.py")
p1_train = importlib.util.module_from_spec(_SPEC)
sys.modules["p1_train"] = p1_train
_SPEC.loader.exec_module(p1_train)


def test_unset_emits_nothing(monkeypatch) -> None:
    monkeypatch.delenv("P1_LOG_EVERY_ROUND", raising=False)
    assert p1_train.roundlog_line(5120, 2048, 12.3, 1.8) is None


def test_zero_emits_nothing(monkeypatch) -> None:
    monkeypatch.setenv("P1_LOG_EVERY_ROUND", "0")
    assert p1_train.roundlog_line(5120, 2048, 12.3, 1.8) is None


def test_one_emits_parseable_line(monkeypatch) -> None:
    monkeypatch.setenv("P1_LOG_EVERY_ROUND", "1")
    line = p1_train.roundlog_line(10240, 2048, 52.7, 2.0)
    assert line == "roundlog transitions=10240 count=2048 collect_s=52.7 update_s=2.0"
    assert line.startswith("roundlog ")  # grep anchor used by the smoke gate
