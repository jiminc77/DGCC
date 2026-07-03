"""Delta-m feature normalization for the §7 Phi/δm pipeline.

Normalization is pure per-channel standard-deviation scaling.  The three
mode-0 centroid delta channels pass through unchanged; only the 21 mode>=1
shape channels are divided by fitted standard deviations.  Tiny standard
deviations on scaled channels raise instead of silently producing infinities or
NaNs.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from dgcc.phi.dct import (
    CHANNEL_LAYOUT_ID,
    M,
    PHI_DIM,
    phi_mode0_indices,
    phi_shape_indices,
)


DEFAULT_STD_EPSILON = 1.0e-12


@dataclass(frozen=True)
class DmStats:
    """Fitted delta-m standard deviations for the stable Phi channel layout."""

    std: np.ndarray
    channel_layout_id: str
    M: int
    fit_count: int
    std_epsilon: float = DEFAULT_STD_EPSILON

    def __post_init__(self) -> None:
        std = _validate_vector("std", self.std)
        object.__setattr__(self, "std", std)
        if self.channel_layout_id != CHANNEL_LAYOUT_ID:
            raise ValueError(
                f"channel_layout_id must be {CHANNEL_LAYOUT_ID!r}, got {self.channel_layout_id!r}"
            )
        if self.M != M:
            raise ValueError(f"M must be {M}, got {self.M}")
        if isinstance(self.fit_count, bool) or not isinstance(self.fit_count, int):
            raise TypeError("fit_count must be an int")
        if self.fit_count <= 0:
            raise ValueError("fit_count must be positive")
        if self.std_epsilon <= 0.0 or not np.isfinite(self.std_epsilon):
            raise ValueError("std_epsilon must be a positive finite float")
        _validate_scaled_std(std, self.std_epsilon)

    def to_dict(self) -> dict[str, Any]:
        """Serialize stats to a JSON-compatible dictionary."""

        return {
            "std": self.std.tolist(),
            "channel_layout_id": self.channel_layout_id,
            "M": self.M,
            "fit_count": self.fit_count,
            "std_epsilon": float(self.std_epsilon),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DmStats":
        """Deserialize stats produced by :meth:`to_dict`."""

        if not isinstance(data, dict):
            raise TypeError("DmStats.from_dict expects a dict")
        required = {"std", "channel_layout_id", "M", "fit_count"}
        missing = required - set(data)
        if missing:
            raise ValueError(f"missing DmStats fields: {sorted(missing)}")
        return cls(
            std=np.asarray(data["std"], dtype=float),
            channel_layout_id=data["channel_layout_id"],
            M=int(data["M"]),
            fit_count=int(data["fit_count"]),
            std_epsilon=float(data.get("std_epsilon", DEFAULT_STD_EPSILON)),
        )


class DmNormalizer:
    """Fit, save/load, and apply delta-m standard-deviation normalization."""

    def __init__(self, stats: DmStats | dict[str, Any]):
        self.stats = _coerce_stats(stats)

    @classmethod
    def fit(
        cls,
        delta_ms: np.ndarray,
        *,
        std_epsilon: float = DEFAULT_STD_EPSILON,
    ) -> "DmNormalizer":
        """Return a normalizer fitted from a ``(K, 24)`` delta-m matrix."""

        return cls(fit(delta_ms, std_epsilon=std_epsilon))

    def normalize(self, delta_m: np.ndarray) -> np.ndarray:
        """Normalize one ``(24,)`` delta-m vector using this normalizer's stats."""

        return normalize(delta_m, self.stats)

    def save(self, path: str | Path) -> None:
        """Write fitted stats as JSON."""

        payload = json.dumps(self.stats.to_dict(), indent=2, sort_keys=True) + "\n"
        Path(path).write_text(payload)

    @classmethod
    def load(cls, path: str | Path) -> "DmNormalizer":
        """Load stats JSON written by :meth:`save`."""

        data = json.loads(Path(path).read_text())
        return cls(DmStats.from_dict(data))


def fit(delta_ms: np.ndarray, *, std_epsilon: float = DEFAULT_STD_EPSILON) -> DmStats:
    """Fit per-channel delta-m standard deviations from a ``(K, 24)`` matrix."""

    matrix = _validate_matrix(delta_ms)
    if std_epsilon <= 0.0 or not np.isfinite(std_epsilon):
        raise ValueError("std_epsilon must be a positive finite float")
    std = matrix.std(axis=0)
    return DmStats(
        std=std,
        channel_layout_id=CHANNEL_LAYOUT_ID,
        M=M,
        fit_count=matrix.shape[0],
        std_epsilon=std_epsilon,
    )


def normalize(delta_m: np.ndarray, stats: DmStats | DmNormalizer | dict[str, Any]) -> np.ndarray:
    """Scale a ``(24,)`` delta-m vector, leaving mode-0 channels unchanged."""

    vector = _validate_vector("delta_m", delta_m)
    fitted = _coerce_stats(stats)
    _validate_scaled_std(fitted.std, fitted.std_epsilon)

    out = vector.copy()
    shape_indices = phi_shape_indices()
    out[shape_indices] = out[shape_indices] / fitted.std[shape_indices]
    out[phi_mode0_indices()] = vector[phi_mode0_indices()]
    return out


def load(path: str | Path) -> DmNormalizer:
    """Load a :class:`DmNormalizer` from JSON."""

    return DmNormalizer.load(path)


def _coerce_stats(stats: DmStats | DmNormalizer | dict[str, Any]) -> DmStats:
    if isinstance(stats, DmNormalizer):
        return stats.stats
    if isinstance(stats, DmStats):
        return stats
    if isinstance(stats, dict):
        return DmStats.from_dict(stats)
    raise TypeError("stats must be DmStats, DmNormalizer, or dict")


def _validate_matrix(name_or_value: Any, value: Any | None = None) -> np.ndarray:
    if value is None:
        name = "delta_ms"
        raw = name_or_value
    else:
        name = str(name_or_value)
        raw = value
    try:
        matrix = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite float array with shape (K, 24)") from exc
    if matrix.ndim != 2 or matrix.shape[1] != PHI_DIM:
        raise ValueError(f"{name} must have shape (K, {PHI_DIM}), got {matrix.shape}")
    if matrix.shape[0] <= 0:
        raise ValueError(f"{name} must contain at least one row")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _validate_vector(name: str, value: Any) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite float array with shape ({PHI_DIM},)") from exc
    if vector.shape != (PHI_DIM,):
        raise ValueError(f"{name} must have shape ({PHI_DIM},), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _validate_scaled_std(std: np.ndarray, std_epsilon: float) -> None:
    shape_indices = phi_shape_indices()
    tiny = shape_indices[std[shape_indices] <= std_epsilon]
    if tiny.size:
        raise ValueError(
            "std for scaled mode>=1 channels is <= std_epsilon "
            f"({std_epsilon:g}) at indices {tiny.tolist()}"
        )
