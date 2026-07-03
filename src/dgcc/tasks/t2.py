"""T2 procedural goal generator (P1 §5).

Families: ``{S, U, L, zigzag, smooth-random}`` with randomized parameters and
explicit asymmetric goals (left/right-asymmetric curvature).  The generator is
fully deterministic from ``T2_MASTER_SEED``; the committed split file
(``src/dgcc/tasks/splits/t2_v1.json``) is the source of truth for
train (500) / validation (50) / held-out eval (100) goals and must match a
fresh regeneration bit-for-bit (tested).

Scope guardrail: exactly the five spec families, no over-generalization —
parameters are bounded, planar curves only, anchor near the workspace origin.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from dgcc.goals.dual_goal import DualGoal
from dgcc.phi.resample import resample
from dgcc.tasks.domain import P1_RADIUS

T2_GENERATOR_VERSION = "t2-v1"
T2_MASTER_SEED = 20260703
T2_FAMILIES: tuple[str, ...] = ("s", "u", "l", "zigzag", "smooth_random")
T2_SPECS_PER_FAMILY = 130  # 5 * 130 = 650 = 500 train + 50 val + 100 held-out
T2_SPLIT_SIZES: dict[str, int] = {"train": 500, "val": 50, "heldout": 100}
T2_SPLIT_FILENAME = "t2_v1.json"

#: Anchor xy sampling half-range in meters (goal centroid near the origin).
ANCHOR_XY_HALF_RANGE = 0.15
#: Goal curves rest on the support plane at the rope radius height.
ANCHOR_Z = P1_RADIUS

_DENSE_SAMPLES = 513


# --------------------------------------------------------------------------
# Parameter sampling per family
# --------------------------------------------------------------------------

def _sample_params(family: str, rng: np.random.Generator) -> tuple[dict[str, float], bool]:
    if family == "s":
        amp1 = float(rng.uniform(0.10, 0.28))
        amp2 = float(rng.uniform(0.10, 0.28))
        return {"amp1": amp1, "amp2": amp2}, abs(amp1 - amp2) > 0.04
    if family == "u":
        theta1 = float(rng.uniform(0.35 * np.pi, 0.85 * np.pi))
        theta2 = float(rng.uniform(0.35 * np.pi, 0.85 * np.pi))
        return {"theta1": theta1, "theta2": theta2}, abs(theta1 - theta2) > 0.15 * np.pi
    if family == "l":
        corner_angle = float(rng.uniform(np.deg2rad(55.0), np.deg2rad(125.0)))
        leg_frac = float(rng.uniform(0.30, 0.70))
        return {"corner_angle": corner_angle, "leg_frac": leg_frac}, abs(leg_frac - 0.5) > 0.08
    if family == "zigzag":
        n_periods = int(rng.integers(2, 4))
        amp1 = float(rng.uniform(0.05, 0.14))
        amp2 = float(rng.uniform(0.05, 0.14))
        return {"n_periods": float(n_periods), "amp1": amp1, "amp2": amp2}, abs(amp1 - amp2) > 0.03
    if family == "smooth_random":
        params: dict[str, float] = {}
        for k in (1, 2, 3):
            params[f"a{k}"] = float(rng.uniform(0.03, 0.12) / k)
            params[f"phi{k}"] = float(rng.uniform(0.0, 2.0 * np.pi))
        # Random low-frequency harmonics with random phases are generically
        # reversal-asymmetric.
        return params, True
    raise ValueError(f"unknown T2 family {family!r}; expected one of {T2_FAMILIES}")


# --------------------------------------------------------------------------
# Dense curve builders (planar; DualGoal normalizes centroid/arc length)
# --------------------------------------------------------------------------

def _curve_s(params: dict[str, float]) -> np.ndarray:
    t = np.linspace(0.0, 1.0, _DENSE_SAMPLES)
    amp = params["amp1"] + (params["amp2"] - params["amp1"]) * t
    return np.column_stack((t - 0.5, amp * np.sin(2.0 * np.pi * t), np.zeros_like(t)))


def _curve_u(params: dict[str, float]) -> np.ndarray:
    # Piecewise-constant curvature arc: heading turns theta1 over the first
    # half of arc length and theta2 over the second half.
    s = np.linspace(0.0, 1.0, _DENSE_SAMPLES)
    heading = np.where(
        s <= 0.5,
        2.0 * params["theta1"] * s,
        params["theta1"] + 2.0 * params["theta2"] * (s - 0.5),
    )
    ds = np.diff(s)
    mid_heading = 0.5 * (heading[:-1] + heading[1:])
    x = np.concatenate(([0.0], np.cumsum(np.cos(mid_heading) * ds)))
    y = np.concatenate(([0.0], np.cumsum(np.sin(mid_heading) * ds)))
    return np.column_stack((x, y, np.zeros_like(x)))


def _curve_l(params: dict[str, float]) -> np.ndarray:
    s = np.linspace(0.0, 1.0, _DENSE_SAMPLES)
    frac = params["leg_frac"]
    turn = np.pi - params["corner_angle"]  # exterior turn at the corner
    heading = np.where(s <= frac, 0.0, turn)
    ds = np.diff(s)
    mid_heading = 0.5 * (heading[:-1] + heading[1:])
    x = np.concatenate(([0.0], np.cumsum(np.cos(mid_heading) * ds)))
    y = np.concatenate(([0.0], np.cumsum(np.sin(mid_heading) * ds)))
    return np.column_stack((x, y, np.zeros_like(x)))


def _curve_zigzag(params: dict[str, float]) -> np.ndarray:
    t = np.linspace(0.0, 1.0, _DENSE_SAMPLES)
    n = float(params["n_periods"])
    amp = params["amp1"] + (params["amp2"] - params["amp1"]) * t
    # Triangle wave for sharper zigzag corners.
    tri = (2.0 / np.pi) * np.arcsin(np.sin(2.0 * np.pi * n * t))
    return np.column_stack((t - 0.5, amp * tri, np.zeros_like(t)))


def _curve_smooth_random(params: dict[str, float]) -> np.ndarray:
    t = np.linspace(0.0, 1.0, _DENSE_SAMPLES)
    y = np.zeros_like(t)
    for k in (1, 2, 3):
        y = y + params[f"a{k}"] * np.sin(np.pi * k * t + params[f"phi{k}"])
    return np.column_stack((t - 0.5, y, np.zeros_like(t)))


_CURVE_BUILDERS = {
    "s": _curve_s,
    "u": _curve_u,
    "l": _curve_l,
    "zigzag": _curve_zigzag,
    "smooth_random": _curve_smooth_random,
}


def t2_unit_template(family: str, params: dict[str, float]) -> np.ndarray:
    """Return the canonical 32-node template curve for a family/params pair."""

    if family not in _CURVE_BUILDERS:
        raise ValueError(f"unknown T2 family {family!r}; expected one of {T2_FAMILIES}")
    return resample(_CURVE_BUILDERS[family](params))


# --------------------------------------------------------------------------
# Deterministic spec generation and splits
# --------------------------------------------------------------------------

def generate_t2_payload() -> dict[str, Any]:
    """Generate the full deterministic T2 goal payload (specs + splits)."""

    specs: list[dict[str, Any]] = []
    index = 0
    for slot in range(T2_SPECS_PER_FAMILY):
        for family in T2_FAMILIES:
            rng = np.random.default_rng(np.random.SeedSequence([T2_MASTER_SEED, index]))
            params, asymmetric = _sample_params(family, rng)
            anchor = [
                float(rng.uniform(-ANCHOR_XY_HALF_RANGE, ANCHOR_XY_HALF_RANGE)),
                float(rng.uniform(-ANCHOR_XY_HALF_RANGE, ANCHOR_XY_HALF_RANGE)),
                float(ANCHOR_Z),
            ]
            specs.append(
                {
                    "goal_id": f"t2-{index:04d}",
                    "family": family,
                    "slot": slot,
                    "params": params,
                    "anchor": anchor,
                    "anchor_mode": "centroid",
                    "asymmetric": bool(asymmetric),
                }
            )
            index += 1

    total = len(specs)
    expected_total = sum(T2_SPLIT_SIZES.values())
    if total != expected_total:
        raise RuntimeError(f"spec count {total} != expected {expected_total}")

    order = np.random.default_rng(np.random.SeedSequence([T2_MASTER_SEED, 999_983])).permutation(total)
    train_end = T2_SPLIT_SIZES["train"]
    val_end = train_end + T2_SPLIT_SIZES["val"]
    splits = {
        "train": sorted(specs[i]["goal_id"] for i in order[:train_end]),
        "val": sorted(specs[i]["goal_id"] for i in order[train_end:val_end]),
        "heldout": sorted(specs[i]["goal_id"] for i in order[val_end:]),
    }
    return {
        "version": T2_GENERATOR_VERSION,
        "master_seed": T2_MASTER_SEED,
        "families": list(T2_FAMILIES),
        "split_sizes": dict(T2_SPLIT_SIZES),
        "specs": specs,
        "splits": splits,
    }


def payload_json(payload: dict[str, Any] | None = None) -> str:
    """Serialize the payload with a stable format for committed storage."""

    data = generate_t2_payload() if payload is None else payload
    return json.dumps(data, indent=1, sort_keys=True) + "\n"


def default_split_path() -> Path:
    """Return the committed split-file path inside the package."""

    return Path(str(resources.files("dgcc.tasks"))) / "splits" / T2_SPLIT_FILENAME


def write_split_file(path: Path | None = None) -> Path:
    """Write the deterministic payload to the committed split file."""

    target = default_split_path() if path is None else Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload_json(), encoding="utf-8")
    return target


def load_t2_payload(path: Path | None = None) -> dict[str, Any]:
    """Load the committed split payload (source of truth for T2 goals)."""

    source = default_split_path() if path is None else Path(path)
    return json.loads(source.read_text(encoding="utf-8"))


def build_t2_goal(spec: dict[str, Any]) -> DualGoal:
    """Reconstruct the :class:`DualGoal` for one committed spec."""

    template = t2_unit_template(str(spec["family"]), dict(spec["params"]))
    return DualGoal(
        shape_template=template,
        anchor=np.asarray(spec["anchor"], dtype=float),
        anchor_mode=str(spec.get("anchor_mode", "centroid")),  # type: ignore[arg-type]
        template_name=f"t2_{spec['family']}:{spec['goal_id']}",
    )


def load_t2_split(split: str, path: Path | None = None) -> list[tuple[dict[str, Any], DualGoal]]:
    """Return ``(spec, goal)`` pairs for one split from the committed file."""

    payload = load_t2_payload(path)
    if split not in payload["splits"]:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(payload['splits'])}")
    wanted = set(payload["splits"][split])
    pairs = [
        (spec, build_t2_goal(spec))
        for spec in payload["specs"]
        if spec["goal_id"] in wanted
    ]
    if len(pairs) != len(wanted):
        raise RuntimeError(f"split {split!r} resolved {len(pairs)} of {len(wanted)} goals")
    return pairs


__all__ = [
    "ANCHOR_XY_HALF_RANGE",
    "ANCHOR_Z",
    "T2_FAMILIES",
    "T2_GENERATOR_VERSION",
    "T2_MASTER_SEED",
    "T2_SPLIT_SIZES",
    "build_t2_goal",
    "default_split_path",
    "generate_t2_payload",
    "load_t2_payload",
    "load_t2_split",
    "payload_json",
    "t2_unit_template",
    "write_split_file",
]
