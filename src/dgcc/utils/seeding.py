"""Seed-management utilities for reproducible P0 scripts."""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python and NumPy RNGs, returning a NumPy generator."""

    normalized = int(seed)
    random.seed(normalized)
    np.random.seed(normalized)
    return np.random.default_rng(normalized)
