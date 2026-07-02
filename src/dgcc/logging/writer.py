"""Transition dataset writer stub.

Purpose: write and read transition records with config and commit metadata
(h5py per the approved plan, O4). Implemented in M4.
"""

from __future__ import annotations

from typing import Any

from dgcc.logging.schema import TransitionRecord


def write_transitions(path: str, records: list[TransitionRecord], meta: dict[str, Any]) -> None:
    """Write transition records plus config/commit metadata. Implemented in M4."""
    raise NotImplementedError("write_transitions is implemented in P0-M4")


def read_transitions(path: str) -> tuple[list[TransitionRecord], dict[str, Any]]:
    """Read transition records plus metadata written by ``write_transitions``. Implemented in M4."""
    raise NotImplementedError("read_transitions is implemented in P0-M4")
