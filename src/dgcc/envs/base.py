"""Shared deformable linear object environment interface.

This module defines only the P0 common contract. Concrete simulator adapters
are implemented in later milestones.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(kw_only=True)
class RopeParams:
    """Rope physical parameters supplied to simulator adapters.

    Attributes:
        length_m: Rope physical length in meters.
        n_segments: Simulator-internal segment count. The P0 interface default
            is 50 segments.
        bend_stiffness: Bend stiffness in simulator-specific units; adapters map
            this value to the underlying simulator representation.
        twist_stiffness: Twist stiffness in simulator-specific units.
        friction: Static-friction-style coefficient; adapters map this value to
            simulator-specific friction parameters.
        radius: Rope radius in meters. The P0 default is 0.005 m.
    """

    length_m: float
    n_segments: int = 50
    bend_stiffness: float
    twist_stiffness: float
    friction: float
    radius: float = 0.005


class DLOEnvBase(ABC):
    """Abstract world-frame rope manipulation environment contract.

    Units are meters and all coordinates are expressed in the world/task frame.
    The external centerline interface always exposes ``K == 32`` arc-length
    resampled nodes, regardless of the simulator-internal segment count.
    """

    K: int = 32

    @abstractmethod
    def reset(self, params: RopeParams, init_shape: str, seed: int) -> dict:
        """Reset the simulator to a seeded initial rope state.

        Args:
            params: Rope physical parameters to apply.
            init_shape: Named initial shape requested by the caller.
            seed: Deterministic seed for reset-time randomness.

        Returns:
            Simulator-specific metadata for the reset. Coordinates remain in
            meters in the world/task frame.
        """
        ...

    @abstractmethod
    def get_centerline_raw(self) -> np.ndarray:
        """Return the simulator-native rope centerline as an ``(N_sim, 3)`` array.

        Coordinates are in meters in the world/task frame. ``N_sim`` is the
        simulator-internal node or segment count and may differ across adapters.
        """
        ...

    @abstractmethod
    def get_centerline(self) -> np.ndarray:
        """Return the external rope centerline as a ``(32, 3)`` float array.

        Coordinates are in meters in the world/task frame. The returned nodes are
        arc-length resampled to ``K == 32`` for all adapters.
        """
        ...

    @abstractmethod
    def step_primitive(self, p: int, delta: np.ndarray, lift: str) -> dict:
        """Execute one grasp-move-release-settle primitive.

        Contract:
            ``grasp(node p) -> move(delta) -> release -> settle`` where
            ``delta`` is a 3-vector in meters satisfying ``||delta|| <= 0.15``.
            ``lift`` is either ``"low"`` or ``"high"`` with heights 0.02 m and
            0.15 m, respectively.

        Returns:
            A dictionary with exactly these contract keys:
            ``X_before`` (``(32, 3)``), ``X_after`` (``(32, 3)``),
            ``grasp_success`` (``bool``), ``settle_steps`` (``int``), and
            ``info`` (adapter-specific metadata dictionary).
        """
        ...

    @abstractmethod
    def settle(self, vel_threshold: float = 1e-3, max_steps: int = 5000) -> bool:
        """Step until the rope is quasi-static or the step budget is exhausted.

        Settling succeeds when the maximum node velocity, or the simulator's
        maximum absolute generalized velocity where applicable, is below
        ``vel_threshold``. ``vel_threshold`` is in meters per second for node
        velocities. ``max_steps`` caps simulator steps.

        Returns:
            ``True`` when the velocity threshold is reached before the step
            budget is exhausted; otherwise ``False``.
        """
        ...
