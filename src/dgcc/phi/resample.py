"""Arc-length centerline resampling stub.

Purpose: implement ``resample(X_raw) -> (32, 3)`` for the deformation feature
pipeline. Minimal version lands in M1 (needed by the smoke (50,3)->(32,3)
assert, per the approved plan F4); finalized in M4 with the §7 invariance test.
"""

from __future__ import annotations

import numpy as np


def resample(X_raw: np.ndarray) -> np.ndarray:
    """Return a 32-node arc-length resampling of ``X_raw`` (minimal in M1, final in M4)."""
    raise NotImplementedError("resample: minimal version lands in P0-M1, finalized in P0-M4")
