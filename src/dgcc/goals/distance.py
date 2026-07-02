"""Goal-distance metric stub.

Purpose: implement length-normalized bidirectional Chamfer distance for DGCC
goals. Implemented in M5.
"""

from __future__ import annotations

import numpy as np


def chamfer_distance(X: np.ndarray, G: np.ndarray, length_m: float) -> float:
    """Compute length-normalized bidirectional Chamfer distance in M5."""
    raise NotImplementedError("goal distance is implemented in P0-M5")
