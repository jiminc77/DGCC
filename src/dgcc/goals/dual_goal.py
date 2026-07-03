"""Dual-goal representation and 24-channel ``c_g`` feature construction.

A :class:`DualGoal` stores a scale-free rope shape template and an absolute
world-frame anchor.  Templates are normalized on construction: the centroid is
subtracted and the remaining curve is divided by its discrete arc length, so
``shape_template`` has unit arc length and zero centroid.  ``goal_curve`` then
reconstructs a world-frame curve as ``shape_template * length_m`` placed so the
configured anchor denotes either the curve centroid (default) or the first
endpoint.

The ``c_g`` layout is stable and explicit:
``[21 Phi_DCT mode>=1 shape channels in axis-major-xyz-modes-0-7-v1 order,
3 anchor delta channels (x, y, z)]``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from dgcc.phi.dct import CHANNEL_LAYOUT_ID, Phi_DCT, phi_shape_indices
from dgcc.phi.resample import K as RESAMPLED_POINTS
from dgcc.phi.resample import resample

AnchorMode = Literal["centroid", "endpoint"]

ANCHOR_MODES: tuple[str, ...] = ("centroid", "endpoint")
TEMPLATE_NAMES: tuple[str, ...] = ("straight", "u_bend", "s_curve")
SHAPE_CHANNEL_INDICES: np.ndarray = phi_shape_indices()
SHAPE_CHANNEL_COUNT = int(SHAPE_CHANNEL_INDICES.size)
ANCHOR_CHANNEL_COUNT = 3
CG_DIM = SHAPE_CHANNEL_COUNT + ANCHOR_CHANNEL_COUNT


@dataclass(frozen=True)
class DualGoal:
    """Scale-free shape template plus absolute anchor target.

    Args:
        shape_template: ``(32, 3)`` curve samples on normalized arc length.  The
            constructor always subtracts the centroid and divides by discrete
            arc length, making the stored template centroid-removed and
            scale-normalized to unit arc length.
        anchor: Absolute world-frame anchor target with shape ``(3,)``.
        anchor_mode: ``"centroid"`` (default) means ``anchor`` is the desired
            curve centroid.  ``"endpoint"`` means ``anchor`` is the desired
            first endpoint.
        template_name: Optional provenance label for analytic template goals.
    """

    shape_template: np.ndarray
    anchor: np.ndarray
    anchor_mode: AnchorMode = "centroid"
    template_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "shape_template",
            normalize_shape_template(self.shape_template),
        )
        object.__setattr__(self, "anchor", _validate_anchor(self.anchor))
        object.__setattr__(self, "anchor_mode", validate_anchor_mode(self.anchor_mode))
        if self.template_name is not None:
            object.__setattr__(self, "template_name", str(self.template_name))


def validate_anchor_mode(anchor_mode: str) -> AnchorMode:
    """Return a supported anchor mode or raise ``ValueError``."""

    mode = str(anchor_mode)
    if mode not in ANCHOR_MODES:
        raise ValueError(f"anchor_mode must be one of {ANCHOR_MODES}, got {anchor_mode!r}")
    return mode  # type: ignore[return-value]


def arc_length(points: np.ndarray) -> float:
    """Return discrete polyline arc length for a finite ``(N, 3)`` curve."""

    curve = _validate_centerline("points", points, min_points=2)
    return float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum())


def normalize_shape_template(shape_template: np.ndarray) -> np.ndarray:
    """Return a centroid-removed, unit-arc-length ``(32, 3)`` template."""

    template = _validate_template(shape_template)
    centered = template - template.mean(axis=0)
    length = arc_length(centered)
    if length <= 0.0:
        raise ValueError("shape_template must have non-zero arc length")
    normalized = centered / length
    # Remove the tiny floating residual left by division so equality tests and
    # anchor placement are stable across platforms.
    normalized = normalized - normalized.mean(axis=0)
    return normalized


def make_shape_template(name: str) -> np.ndarray:
    """Build one of the local analytic unit templates.

    Supported names are ``"straight"``, ``"u_bend"``/``"semicircle"``, and
    ``"s_curve"``.  The analytic curves are sampled densely, resampled by the
    M4 arc-length helper to 32 points, then normalized by
    :func:`normalize_shape_template`.
    """

    key = str(name)
    if key == "semicircle":
        key = "u_bend"
    if key not in TEMPLATE_NAMES:
        raise ValueError(f"unknown goal template {name!r}; expected one of {TEMPLATE_NAMES}")

    dense_n = 257
    if key == "straight":
        t = np.linspace(-0.5, 0.5, dense_n)
        dense = np.column_stack((t, np.zeros_like(t), np.zeros_like(t)))
    elif key == "u_bend":
        theta = np.linspace(0.0, np.pi, dense_n)
        radius = 1.0 / np.pi
        dense = np.column_stack(
            (
                radius * np.cos(theta),
                radius * np.sin(theta),
                np.zeros_like(theta),
            )
        )
    else:  # s_curve
        t = np.linspace(0.0, 1.0, dense_n)
        dense = np.column_stack(
            (
                t - 0.5,
                0.18 * np.sin(2.0 * np.pi * (t - 0.5)),
                0.06 * np.sin(np.pi * t),
            )
        )

    return normalize_shape_template(resample(dense))


def template_library() -> dict[str, np.ndarray]:
    """Return the local analytic goal-template library."""

    return {name: make_shape_template(name) for name in TEMPLATE_NAMES}


def make_goal(
    template: str,
    anchor: np.ndarray,
    *,
    anchor_mode: AnchorMode = "centroid",
) -> DualGoal:
    """Construct a :class:`DualGoal` from a named analytic template."""

    return DualGoal(
        shape_template=make_shape_template(template),
        anchor=anchor,
        anchor_mode=anchor_mode,
        template_name=template,
    )


def coerce_goal(goal: DualGoal | Mapping[str, Any]) -> DualGoal:
    """Accept a ``DualGoal`` or mapping with ``shape_template``/``anchor`` keys."""

    if isinstance(goal, DualGoal):
        return goal
    if isinstance(goal, Mapping):
        if "template" in goal and "shape_template" not in goal:
            shape_template = make_shape_template(str(goal["template"]))
            template_name = str(goal["template"])
        else:
            shape_template = goal["shape_template"]
            template_name = goal.get("template_name")
        return DualGoal(
            shape_template=shape_template,
            anchor=goal["anchor"],
            anchor_mode=goal.get("anchor_mode", "centroid"),
            template_name=template_name,
        )
    raise TypeError("goal must be a DualGoal or mapping")


def goal_curve(goal: DualGoal | Mapping[str, Any], length_m: float) -> np.ndarray:
    """Reconstruct the goal's world-frame ``(32, 3)`` centerline.

    ``length_m`` scales the unit template.  For ``anchor_mode='centroid'``, the
    scaled template's centroid is placed at ``goal.anchor``.  For
    ``anchor_mode='endpoint'``, the first endpoint is placed at ``goal.anchor``.
    """

    typed_goal = coerce_goal(goal)
    length = _validate_length(length_m)
    curve = typed_goal.shape_template * length
    if typed_goal.anchor_mode == "centroid":
        return curve + typed_goal.anchor
    return curve - curve[0] + typed_goal.anchor


def anchor_of(X: np.ndarray, anchor_mode: AnchorMode = "centroid") -> np.ndarray:
    """Return the centroid or first-endpoint anchor of a rope centerline.

    The centerline is first arc-length-resampled to the canonical 32 nodes, so
    centroid anchors are not biased by the input point count.
    """

    mode = validate_anchor_mode(anchor_mode)
    canonical = canonical_centerline(X)
    if mode == "centroid":
        return canonical.mean(axis=0)
    return canonical[0].copy()


def canonical_centerline(X: np.ndarray) -> np.ndarray:
    """Validate and arc-length-resample any finite non-degenerate centerline."""

    points = _validate_centerline("X", X, min_points=2)
    return resample(points)


def phi_shape(X: np.ndarray) -> np.ndarray:
    """Return the 21 mode>=1 Phi channels used by the dual-goal vector."""

    return Phi_DCT(canonical_centerline(X))[SHAPE_CHANNEL_INDICES]


def c_g(X: np.ndarray, goal: DualGoal | Mapping[str, Any], length_m: float) -> np.ndarray:
    """Return ``[Phi_shape(G)-Phi_shape(X), anchor(G)-anchor(X)]`` as ``(24,)``."""

    typed_goal = coerce_goal(goal)
    goal_world = goal_curve(typed_goal, length_m)
    # Canonicalize BOTH curves through the same resample path so that
    # c_g(goal_curve(g), g) == 0 exactly (QA C2 invariant; rho impact ~6.6e-06).
    shape_delta = phi_shape(goal_world) - phi_shape(X)
    anchor_delta = typed_goal.anchor - anchor_of(X, typed_goal.anchor_mode)
    out = np.concatenate((shape_delta, anchor_delta))
    if out.shape != (CG_DIM,):
        raise RuntimeError(f"internal c_g shape error: expected {(CG_DIM,)}, got {out.shape}")
    return out


def compute_c_g(
    X: np.ndarray,
    goal: DualGoal | Mapping[str, Any],
    length_m: float = 1.0,
) -> np.ndarray:
    """Backward-compatible alias for :func:`c_g`."""

    return c_g(X, goal, length_m)


def _validate_anchor(anchor: np.ndarray) -> np.ndarray:
    try:
        arr = np.asarray(anchor, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("anchor must be a finite float array with shape (3,)") from exc
    if arr.shape != (3,):
        raise ValueError(f"anchor must have shape (3,), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("anchor must contain only finite values")
    return arr.copy()


def _validate_template(shape_template: np.ndarray) -> np.ndarray:
    arr = _validate_centerline("shape_template", shape_template, min_points=RESAMPLED_POINTS)
    expected = (RESAMPLED_POINTS, 3)
    if arr.shape != expected:
        raise ValueError(f"shape_template must have shape {expected}, got {arr.shape}")
    return arr


def _validate_centerline(name: str, value: np.ndarray, *, min_points: int) -> np.ndarray:
    try:
        points = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite float array with shape (N, 3)") from exc
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {points.shape}")
    if points.shape[0] < min_points:
        raise ValueError(f"{name} must contain at least {min_points} points")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain only finite values")
    if float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) <= 0.0:
        raise ValueError(f"{name} must have non-zero arc length")
    return points


def _validate_length(length_m: float) -> float:
    length = float(length_m)
    if length <= 0.0 or not np.isfinite(length):
        raise ValueError("length_m must be a positive finite float")
    return length


__all__ = [
    "ANCHOR_CHANNEL_COUNT",
    "ANCHOR_MODES",
    "AnchorMode",
    "CG_DIM",
    "CHANNEL_LAYOUT_ID",
    "DualGoal",
    "SHAPE_CHANNEL_COUNT",
    "SHAPE_CHANNEL_INDICES",
    "TEMPLATE_NAMES",
    "anchor_of",
    "arc_length",
    "c_g",
    "canonical_centerline",
    "coerce_goal",
    "compute_c_g",
    "goal_curve",
    "make_goal",
    "make_shape_template",
    "normalize_shape_template",
    "phi_shape",
    "template_library",
    "validate_anchor_mode",
]
