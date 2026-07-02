"""Transition record schema for P0 logging tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np


@dataclass
class TransitionRecord:
    """Serializable rope transition record.

    Fields:
        X_before: Float centerline array with shape ``(32, 3)``.
        X_after: Float centerline array with shape ``(32, 3)``.
        p: Grasp node index.
        delta: Float primitive displacement array with shape ``(3,)``.
        lift: Lift mode string.
        grasp_success: Whether the grasp succeeded.
        settle_steps: Number of simulator steps used for settling.
        rope_params: Rope parameter dictionary.
        seed: Deterministic seed.
        sim: Simulator identifier.
        timestamp: Record timestamp string.
        commit_hash: Source commit hash string.
    """

    X_before: np.ndarray
    X_after: np.ndarray
    p: int
    delta: np.ndarray
    lift: str
    grasp_success: bool
    settle_steps: int
    rope_params: dict
    seed: int
    sim: str
    timestamp: str
    commit_hash: str

    FIELD_NAMES: ClassVar[tuple[str, ...]] = (
        "X_before",
        "X_after",
        "p",
        "delta",
        "lift",
        "grasp_success",
        "settle_steps",
        "rope_params",
        "seed",
        "sim",
        "timestamp",
        "commit_hash",
    )

    def __post_init__(self) -> None:
        self.X_before = self._validate_array("X_before", self.X_before, (32, 3))
        self.X_after = self._validate_array("X_after", self.X_after, (32, 3))
        self.delta = self._validate_array("delta", self.delta, (3,))
        self._validate_int("p", self.p)
        self._validate_int("settle_steps", self.settle_steps)
        self._validate_int("seed", self.seed)
        self._validate_bool("grasp_success", self.grasp_success)
        self._validate_str("lift", self.lift)
        self._validate_str("sim", self.sim)
        self._validate_str("timestamp", self.timestamp)
        self._validate_str("commit_hash", self.commit_hash)
        if not isinstance(self.rope_params, dict):
            raise TypeError("rope_params must be a dict")

    def to_dict(self) -> dict[str, Any]:
        """Serialize this record to a plain Python dictionary.

        Numpy arrays are converted to nested lists so the result can be written
        to JSON/YAML/HDF5 metadata layers by later milestones.
        """
        return {
            "X_before": self.X_before.tolist(),
            "X_after": self.X_after.tolist(),
            "p": int(self.p),
            "delta": self.delta.tolist(),
            "lift": self.lift,
            "grasp_success": bool(self.grasp_success),
            "settle_steps": int(self.settle_steps),
            "rope_params": dict(self.rope_params),
            "seed": int(self.seed),
            "sim": self.sim,
            "timestamp": self.timestamp,
            "commit_hash": self.commit_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransitionRecord":
        """Deserialize and validate a transition record dictionary."""
        if not isinstance(data, dict):
            raise TypeError("TransitionRecord.from_dict expects a dict")
        expected = set(cls.FIELD_NAMES)
        actual = set(data)
        missing = expected - actual
        extra = actual - expected
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing fields: {sorted(missing)}")
            if extra:
                details.append(f"extra fields: {sorted(extra)}")
            raise ValueError("invalid TransitionRecord fields: " + "; ".join(details))
        return cls(
            X_before=np.asarray(data["X_before"], dtype=float),
            X_after=np.asarray(data["X_after"], dtype=float),
            p=data["p"],
            delta=np.asarray(data["delta"], dtype=float),
            lift=data["lift"],
            grasp_success=data["grasp_success"],
            settle_steps=data["settle_steps"],
            rope_params=data["rope_params"],
            seed=data["seed"],
            sim=data["sim"],
            timestamp=data["timestamp"],
            commit_hash=data["commit_hash"],
        )

    @staticmethod
    def _validate_array(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a float array") from exc
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
        return array

    @staticmethod
    def _validate_int(name: str, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int")

    @staticmethod
    def _validate_bool(name: str, value: Any) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a bool")

    @staticmethod
    def _validate_str(name: str, value: Any) -> None:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a str")
