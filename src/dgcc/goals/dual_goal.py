"""Dual goal representation stub.

Purpose: hold the shape-template plus anchor goal and ``c_g`` computation.
Implemented in M5.
"""

from __future__ import annotations

import numpy as np


def compute_c_g(X: np.ndarray, goal: dict) -> np.ndarray:
    """Compute the 24-channel dual-goal vector in M5."""
    raise NotImplementedError("dual goal features are implemented in P0-M5")
