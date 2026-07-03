"""P1 replay buffer and transition schema v2.

Schema note (consensus plan C3/F2): the P0 v1 transition schema is CLOSED —
``TransitionRecord.FIELD_NAMES`` is a fixed 12-field tuple, ``from_dict``
rejects extra fields, and the v1 writer enforces ``SCHEMA_VERSION = 1``.  P1
therefore defines **schema v2** (v1's 12 fields + task/goal/provenance
fields) with ``SCHEMA_VERSION = 2`` and its own layout validation.  A v1
READ path exists ONLY for the P0-M4 reuse contingency (global rule 7:
``grasp_success ∧ settle_converged`` filter with a provenance flag).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from dgcc.logging.writer import SCHEMA_VERSION as V1_SCHEMA_VERSION

SCHEMA_VERSION = 2
PROVENANCE_FRESH = "p1_fresh"
PROVENANCE_P0_REUSE = "p0_m4_reuse"
LIFT_NAMES = ("low", "high")

_STR = h5py.string_dtype(encoding="utf-8")
#: v2 columnar layout: the 12 v1 fields + P1 task/goal/provenance fields.
V2_DATASET_LAYOUT: dict[str, tuple[tuple[int, ...], Any]] = {
    # --- v1 fields ---
    "X_before": ((32, 3), np.float64),
    "X_after": ((32, 3), np.float64),
    "p": ((), np.int64),
    "delta": ((3,), np.float64),
    "lift": ((), _STR),
    "grasp_success": ((), np.bool_),
    "settle_steps": ((), np.int64),
    "rope_params": ((), _STR),
    "seed": ((), np.int64),
    "sim": ((), _STR),
    "timestamp": ((), _STR),
    "commit_hash": ((), _STR),
    # --- v2 additions ---
    "task_id": ((), _STR),
    "goal_id": ((), _STR),
    "goal_spec_hash": ((), _STR),
    "goal_curve": ((32, 3), np.float64),
    "episode_id": ((), np.int64),
    "step_index": ((), np.int64),
    "reward": ((), np.float64),
    "done": ((), np.bool_),
    "provenance": ((), _STR),
}


class ReplaySchemaError(RuntimeError):
    """Raised when a transition file does not match the expected schema."""


def goal_spec_hash(goal_curve: np.ndarray) -> str:
    """Stable identity hash of a world-frame goal curve."""

    curve = np.ascontiguousarray(np.asarray(goal_curve, dtype=np.float64))
    return hashlib.sha256(curve.tobytes()).hexdigest()[:16]


class ReplayBuffer:
    """Fixed-capacity ring buffer over the v2 training fields.

    Training fields kept in memory: ``X_before``, ``X_after``, ``goal_curve``,
    ``p``, ``delta``, ``lift`` (0=low, 1=high), ``reward``, ``done``.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.size = 0
        self._next = 0
        self.X_before = np.zeros((capacity, 32, 3), dtype=np.float32)
        self.X_after = np.zeros((capacity, 32, 3), dtype=np.float32)
        self.goal_curve = np.zeros((capacity, 32, 3), dtype=np.float32)
        self.p = np.zeros(capacity, dtype=np.int64)
        self.delta = np.zeros((capacity, 3), dtype=np.float32)
        self.lift = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=bool)

    def add_batch(
        self,
        *,
        X_before: np.ndarray,
        X_after: np.ndarray,
        goal_curve: np.ndarray,
        p: np.ndarray,
        delta: np.ndarray,
        lift: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
    ) -> None:
        """Append a batch of transitions (oldest entries overwritten)."""

        count = int(np.asarray(p).shape[0])
        for name, value, tail in (
            ("X_before", X_before, (32, 3)),
            ("X_after", X_after, (32, 3)),
            ("goal_curve", goal_curve, (32, 3)),
            ("delta", delta, (3,)),
        ):
            arr = np.asarray(value)
            if arr.shape != (count, *tail):
                raise ValueError(f"{name} must have shape {(count, *tail)}, got {arr.shape}")
        indices = (self._next + np.arange(count)) % self.capacity
        self.X_before[indices] = np.asarray(X_before, dtype=np.float32)
        self.X_after[indices] = np.asarray(X_after, dtype=np.float32)
        self.goal_curve[indices] = np.asarray(goal_curve, dtype=np.float32)
        self.p[indices] = np.asarray(p, dtype=np.int64)
        self.delta[indices] = np.asarray(delta, dtype=np.float32)
        self.lift[indices] = np.asarray(lift, dtype=np.int64)
        self.reward[indices] = np.asarray(reward, dtype=np.float32)
        self.done[indices] = np.asarray(done, dtype=bool)
        self._next = int((self._next + count) % self.capacity)
        self.size = int(min(self.size + count, self.capacity))

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Uniformly sample a training batch."""

        if self.size == 0:
            raise RuntimeError("cannot sample from an empty replay buffer")
        indices = rng.integers(0, self.size, size=int(batch_size))
        return {
            "X_before": self.X_before[indices].astype(np.float64),
            "X_after": self.X_after[indices].astype(np.float64),
            "goal_curve": self.goal_curve[indices].astype(np.float64),
            "p": self.p[indices].copy(),
            "delta": self.delta[indices].astype(np.float64),
            "lift": self.lift[indices].copy(),
            "reward": self.reward[indices].astype(np.float64),
            "done": self.done[indices].copy(),
        }


# ---------------------------------------------------------------------------
# v2 columnar HDF5 I/O
# ---------------------------------------------------------------------------


def write_v2_transitions(path: Path | str, columns: dict[str, Any], meta: dict[str, Any]) -> None:
    """Write a complete v2 columnar transition file."""

    missing = set(V2_DATASET_LAYOUT) - set(columns)
    extra = set(columns) - set(V2_DATASET_LAYOUT)
    if missing or extra:
        raise ReplaySchemaError(f"v2 columns mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    counts = {name: len(columns[name]) for name in V2_DATASET_LAYOUT}
    if len(set(counts.values())) != 1:
        raise ReplaySchemaError(f"v2 column lengths differ: {counts}")
    count = next(iter(counts.values()))

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(target, "w") as h5:
        h5.attrs["schema_version"] = SCHEMA_VERSION
        h5.attrs["record_count"] = count
        h5.attrs["meta_json"] = json.dumps(meta, ensure_ascii=False, sort_keys=True)
        for name, (tail, dtype) in V2_DATASET_LAYOUT.items():
            values = columns[name]
            if dtype is _STR or dtype == _STR:
                data = np.asarray([str(v) for v in values], dtype=object)
                h5.create_dataset(name, data=data, dtype=_STR)
            else:
                h5.create_dataset(name, data=np.asarray(values, dtype=dtype))


def validate_v2_layout(h5: h5py.File) -> int:
    """Validate the closed v2 layout; return the record count."""

    version = int(h5.attrs.get("schema_version", -1))
    if version != SCHEMA_VERSION:
        raise ReplaySchemaError(f"expected schema_version {SCHEMA_VERSION}, got {version}")
    expected = set(V2_DATASET_LAYOUT)
    actual = {name for name in h5.keys()}
    if expected != actual:
        raise ReplaySchemaError(
            f"v2 datasets mismatch: missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )
    counts = {name: int(h5[name].shape[0]) for name in expected}
    if len(set(counts.values())) != 1:
        raise ReplaySchemaError(f"v2 dataset lengths differ: {counts}")
    count = next(iter(counts.values()))
    if int(h5.attrs.get("record_count", -1)) != count:
        raise ReplaySchemaError("record_count attribute does not match dataset length")
    for name, (tail, _) in V2_DATASET_LAYOUT.items():
        if tuple(h5[name].shape[1:]) != tail:
            raise ReplaySchemaError(f"{name} tail shape {h5[name].shape[1:]} != {tail}")
    return count


def read_v2_transitions(path: Path | str) -> tuple[dict[str, np.ndarray | list[str]], dict[str, Any]]:
    """Read a v2 columnar file → (columns, meta)."""

    with h5py.File(Path(path), "r") as h5:
        validate_v2_layout(h5)
        columns: dict[str, Any] = {}
        for name, (_, dtype) in V2_DATASET_LAYOUT.items():
            if dtype is _STR or dtype == _STR:
                columns[name] = [str(v) for v in h5[name].asstr()[:]]
            else:
                columns[name] = np.asarray(h5[name][:])
        meta = json.loads(h5.attrs["meta_json"])
    return columns, meta


# ---------------------------------------------------------------------------
# v1 read path — P0-M4 reuse contingency ONLY (global rule 7)
# ---------------------------------------------------------------------------


def ingest_v1_transitions(path: Path | str) -> dict[str, Any]:
    """Read a P0 v1 file, apply the reuse filter, and flag provenance.

    Filter: ``grasp_success ∧ settle_converged`` where convergence follows the
    P0 A1 rule ``settle_steps != settle_max_steps`` (budget read from the v1
    file's config meta, default 5000).  Goals/rewards do not exist in v1 —
    assigning them is the caller's responsibility and must be recorded.
    """

    with h5py.File(Path(path), "r") as h5:
        version = int(h5.attrs.get("schema_version", -1))
        if version != V1_SCHEMA_VERSION:
            raise ReplaySchemaError(f"expected v1 schema_version {V1_SCHEMA_VERSION}, got {version}")
        meta = json.loads(h5.attrs["meta_json"]) if "meta_json" in h5.attrs else {}
        config = meta.get("config", {})
        if isinstance(config, str):
            import yaml

            config = yaml.safe_load(config) or {}
        settle_max = int(config.get("collection", {}).get("settle_max_steps", 5000))
        grasp_success = np.asarray(h5["grasp_success"][:], dtype=bool)
        settle_steps = np.asarray(h5["settle_steps"][:], dtype=int)
        converged = settle_steps != settle_max
        keep = grasp_success & converged
        result = {
            "X_before": np.asarray(h5["X_before"][:], dtype=float)[keep],
            "X_after": np.asarray(h5["X_after"][:], dtype=float)[keep],
            "p": np.asarray(h5["p"][:], dtype=int)[keep],
            "delta": np.asarray(h5["delta"][:], dtype=float)[keep],
            "lift": [str(v) for v, k in zip(h5["lift"].asstr()[:], keep) if k],
            "settle_steps": settle_steps[keep],
            "provenance": PROVENANCE_P0_REUSE,
            "settle_max_steps": settle_max,
            "total_records": int(grasp_success.shape[0]),
            "kept_records": int(keep.sum()),
        }
    return result


__all__ = [
    "LIFT_NAMES",
    "PROVENANCE_FRESH",
    "PROVENANCE_P0_REUSE",
    "ReplayBuffer",
    "ReplaySchemaError",
    "SCHEMA_VERSION",
    "V2_DATASET_LAYOUT",
    "goal_spec_hash",
    "ingest_v1_transitions",
    "read_v2_transitions",
    "validate_v2_layout",
    "write_v2_transitions",
]
