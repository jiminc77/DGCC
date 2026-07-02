"""Arc-length centerline resampling stub.

Purpose: implement ``resample(X_raw) -> (32, 3)`` for the deformation feature
pipeline. Implemented in M4.
"""

from __future__ import annotations

import numpy as np


def resample(X_raw: np.ndarray) -> np.ndarray:
    """Return a 32-node arc-length resampling of ``X_raw`` in M4."""
    raise NotImplementedError("resample is implemented in P0-M4")
