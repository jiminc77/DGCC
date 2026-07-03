"""HDF5 transition dataset writer and reader."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, TypeAlias

import h5py
import numpy as np

from dgcc.logging.schema import TransitionRecord


SCHEMA_VERSION = 1
_REQUIRED_META_KEYS = ("config", "commit_hash")
_STRING_FIELDS = {"lift", "rope_params", "sim", "timestamp", "commit_hash"}
_DATASET_LAYOUT: dict[str, tuple[tuple[int, ...], Any]] = {
    "X_before": ((32, 3), np.float64),
    "X_after": ((32, 3), np.float64),
    "p": ((), np.int64),
    "delta": ((3,), np.float64),
    "lift": ((), h5py.string_dtype(encoding="utf-8")),
    "grasp_success": ((), np.bool_),
    "settle_steps": ((), np.int64),
    "rope_params": ((), h5py.string_dtype(encoding="utf-8")),
    "seed": ((), np.int64),
    "sim": ((), h5py.string_dtype(encoding="utf-8")),
    "timestamp": ((), h5py.string_dtype(encoding="utf-8")),
    "commit_hash": ((), h5py.string_dtype(encoding="utf-8")),
}

RecordInput: TypeAlias = TransitionRecord | Mapping[str, Any]
Selection: TypeAlias = slice | Sequence[int] | np.ndarray | int | None


class TransitionDatasetError(RuntimeError):
    """Raised when a transition HDF5 file is missing required schema content."""


class TransitionWriter:
    """Appendable HDF5 writer for columnar ``TransitionRecord`` datasets.

    Empty writes are valid and create a header-only file with zero-length
    datasets. Each ``append`` call flushes the file so collectors can persist
    completed batches incrementally.
    """

    def __init__(
        self,
        path: str | Path,
        meta: Mapping[str, Any] | None = None,
        mode: str = "w",
    ) -> None:
        if mode not in {"w", "a"}:
            raise ValueError("TransitionWriter mode must be 'w' or 'a'")
        self.path = Path(path)
        self._file: h5py.File | None = None
        self.meta: dict[str, Any]

        if mode == "a" and self.path.exists():
            self._open_existing()
            if meta is not None:
                _validate_meta(meta)
            return

        if meta is None:
            raise ValueError("metadata is required when creating a transition file")
        self._open_new(_validate_meta(meta))

    def __enter__(self) -> "TransitionWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def __len__(self) -> int:
        return self.record_count

    @property
    def record_count(self) -> int:
        h5 = self._require_open()
        return _record_count(h5)

    def append(self, records: RecordInput | Iterable[RecordInput]) -> None:
        """Append one record or a batch of records, then flush the file."""
        batch = _coerce_batch(records)
        if not batch:
            self.flush()
            return

        h5 = self._require_open()
        start = _record_count(h5)
        end = start + len(batch)
        for name in TransitionRecord.FIELD_NAMES:
            h5[name].resize((end, *_DATASET_LAYOUT[name][0]))

        h5["X_before"][start:end] = np.stack([record.X_before for record in batch])
        h5["X_after"][start:end] = np.stack([record.X_after for record in batch])
        h5["p"][start:end] = np.asarray([record.p for record in batch], dtype=np.int64)
        h5["delta"][start:end] = np.stack([record.delta for record in batch])
        h5["lift"][start:end] = [record.lift for record in batch]
        h5["grasp_success"][start:end] = np.asarray(
            [record.grasp_success for record in batch],
            dtype=np.bool_,
        )
        h5["settle_steps"][start:end] = np.asarray(
            [record.settle_steps for record in batch],
            dtype=np.int64,
        )
        h5["rope_params"][start:end] = [
            _encode_rope_params(record.rope_params) for record in batch
        ]
        h5["seed"][start:end] = np.asarray([record.seed for record in batch], dtype=np.int64)
        h5["sim"][start:end] = [record.sim for record in batch]
        h5["timestamp"][start:end] = [record.timestamp for record in batch]
        h5["commit_hash"][start:end] = [record.commit_hash for record in batch]
        h5.attrs["record_count"] = end
        self.flush()

    def flush(self) -> None:
        """Flush pending HDF5 writes to disk."""
        h5 = self._require_open()
        h5.flush()

    def close(self) -> None:
        """Flush and close the underlying HDF5 file."""
        if self._file is None:
            return
        self._file.flush()
        self._file.close()
        self._file = None

    def _open_new(self, meta: dict[str, Any]) -> None:
        try:
            self._file = h5py.File(self.path, "w")
        except OSError as exc:
            raise TransitionDatasetError(f"failed to create transition file {self.path}: {exc}") from exc
        self.meta = meta
        _write_metadata(self._file, meta)
        _create_empty_datasets(self._file)
        self._file.flush()

    def _open_existing(self) -> None:
        try:
            self._file = h5py.File(self.path, "r+")
        except OSError as exc:
            raise TransitionDatasetError(f"failed to open transition file {self.path}: {exc}") from exc
        try:
            _validate_layout(self._file, require_appendable=True)
            self.meta = _read_metadata(self._file)
        except Exception:
            self._file.close()
            self._file = None
            raise

    def _require_open(self) -> h5py.File:
        if self._file is None:
            raise ValueError("TransitionWriter is closed")
        return self._file


def write_transitions(
    path: str | Path,
    records: Iterable[RecordInput],
    meta: Mapping[str, Any],
) -> None:
    """Write a complete transition dataset.

    ``records`` may be empty; in that case a header-only HDF5 file is created.
    Each record is validated through ``TransitionRecord`` before writing.
    """
    with TransitionWriter(path, meta=meta, mode="w") as writer:
        writer.append(records)


def read_transitions(
    path: str | Path,
    selection: Selection = None,
) -> tuple[list[TransitionRecord], dict[str, Any]]:
    """Read transition records plus file metadata.

    ``selection`` may be ``None`` (all records), a Python ``slice``, an integer,
    or a sequence of integer indices. Slice and index selections are applied to
    each columnar HDF5 dataset before records are reconstructed.
    """
    h5_path = Path(path)
    if not h5_path.exists():
        raise FileNotFoundError(f"transition file not found: {h5_path}")

    try:
        with h5py.File(h5_path, "r") as h5:
            _validate_layout(h5, require_appendable=False)
            meta = _read_metadata(h5)
            selector = _normalize_selection(selection, _record_count(h5))
            records = _read_selected_records(h5, selector)
    except OSError as exc:
        raise TransitionDatasetError(f"failed to open transition file {h5_path}: {exc}") from exc
    return records, meta


def _validate_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(meta, Mapping):
        raise TypeError("metadata must be a mapping")
    missing = [key for key in _REQUIRED_META_KEYS if key not in meta]
    if missing:
        raise ValueError(f"missing required metadata keys: {missing}")

    validated = dict(meta)
    if not isinstance(validated["config"], str):
        raise TypeError("metadata config must be a str containing a YAML/JSON config copy")
    if not isinstance(validated["commit_hash"], str):
        raise TypeError("metadata commit_hash must be a str")
    if "creation_time" not in validated:
        validated["creation_time"] = _utc_now()
    if not isinstance(validated["creation_time"], str):
        raise TypeError("metadata creation_time must be a str")

    try:
        json.dumps(validated, ensure_ascii=False, sort_keys=True)
    except TypeError as exc:
        raise TypeError("metadata must be JSON serializable") from exc
    return validated


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_metadata(h5: h5py.File, meta: Mapping[str, Any]) -> None:
    meta_json = json.dumps(meta, ensure_ascii=False, sort_keys=True)
    h5.attrs["schema_version"] = SCHEMA_VERSION
    h5.attrs["meta_json"] = meta_json
    h5.attrs["config"] = meta["config"]
    h5.attrs["commit_hash"] = meta["commit_hash"]
    h5.attrs["creation_time"] = meta["creation_time"]
    h5.attrs["record_count"] = 0


def _read_metadata(h5: h5py.File) -> dict[str, Any]:
    if "meta_json" not in h5.attrs:
        raise TransitionDatasetError("transition file is missing meta_json attribute")
    try:
        meta = json.loads(_attr_to_str(h5.attrs["meta_json"]))
    except (TypeError, json.JSONDecodeError) as exc:
        raise TransitionDatasetError("transition file has invalid meta_json attribute") from exc
    if not isinstance(meta, dict):
        raise TransitionDatasetError("transition file meta_json must decode to a dict")
    return meta


def _create_empty_datasets(h5: h5py.File) -> None:
    for name in TransitionRecord.FIELD_NAMES:
        tail_shape, dtype = _DATASET_LAYOUT[name]
        h5.create_dataset(
            name,
            shape=(0, *tail_shape),
            maxshape=(None, *tail_shape),
            chunks=_chunk_shape(tail_shape),
            dtype=dtype,
        )


def _chunk_shape(tail_shape: tuple[int, ...]) -> tuple[int, ...]:
    first_axis = 128 if tail_shape else 1024
    return (first_axis, *tail_shape)


def _validate_layout(h5: h5py.File, *, require_appendable: bool) -> None:
    version = h5.attrs.get("schema_version")
    if version is None:
        raise TransitionDatasetError("transition file is missing schema_version attribute")
    if int(version) != SCHEMA_VERSION:
        raise TransitionDatasetError(
            f"unsupported transition schema_version {version}; expected {SCHEMA_VERSION}"
        )

    missing = [name for name in TransitionRecord.FIELD_NAMES if name not in h5]
    if missing:
        raise TransitionDatasetError(f"transition file is missing datasets: {missing}")

    lengths = set()
    for name in TransitionRecord.FIELD_NAMES:
        dataset = h5[name]
        tail_shape, expected_dtype = _DATASET_LAYOUT[name]
        if dataset.shape[1:] != tail_shape:
            raise TransitionDatasetError(
                f"dataset {name} has shape tail {dataset.shape[1:]}; expected {tail_shape}"
            )
        if require_appendable and dataset.maxshape[0] is not None:
            raise TransitionDatasetError(f"dataset {name} is not appendable")
        if name in _STRING_FIELDS:
            string_info = h5py.check_string_dtype(dataset.dtype)
            if string_info is None or string_info.encoding != "utf-8":
                raise TransitionDatasetError(f"dataset {name} must be a UTF-8 string dataset")
        elif np.dtype(dataset.dtype) != np.dtype(expected_dtype):
            raise TransitionDatasetError(
                f"dataset {name} has dtype {dataset.dtype}; expected {np.dtype(expected_dtype)}"
            )
        lengths.add(dataset.shape[0])

    if len(lengths) != 1:
        raise TransitionDatasetError("transition datasets have inconsistent record counts")
    record_count_attr = h5.attrs.get("record_count")
    if record_count_attr is not None and int(record_count_attr) != next(iter(lengths)):
        raise TransitionDatasetError("record_count attribute does not match dataset length")


def _record_count(h5: h5py.File) -> int:
    if "X_before" not in h5:
        return 0
    return int(h5["X_before"].shape[0])


def _coerce_batch(records: RecordInput | Iterable[RecordInput]) -> list[TransitionRecord]:
    if isinstance(records, TransitionRecord) or isinstance(records, Mapping):
        raw_batch = [records]
    else:
        raw_batch = list(records)
    return [_coerce_record(record) for record in raw_batch]


def _coerce_record(record: RecordInput) -> TransitionRecord:
    if isinstance(record, TransitionRecord):
        return TransitionRecord.from_dict(record.to_dict())
    if isinstance(record, Mapping):
        return TransitionRecord.from_dict(dict(record))
    raise TypeError("records must be TransitionRecord instances or dictionaries")


def _encode_rope_params(rope_params: dict[str, Any]) -> str:
    try:
        return json.dumps(rope_params, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except TypeError as exc:
        raise TypeError("rope_params must be JSON serializable") from exc


def _normalize_selection(selection: Selection, record_count: int) -> slice | list[int]:
    if selection is None:
        return slice(None)
    if isinstance(selection, int):
        index = _normalize_index(selection, record_count)
        return slice(index, index + 1)
    if isinstance(selection, slice):
        return selection
    if isinstance(selection, np.ndarray):
        if selection.ndim != 1:
            raise IndexError("selection ndarray must be one-dimensional")
        return [_normalize_index(int(index), record_count) for index in selection.tolist()]
    if isinstance(selection, Sequence) and not isinstance(selection, (str, bytes, bytearray)):
        return [_normalize_index(int(index), record_count) for index in selection]
    raise TypeError("selection must be None, a slice, an int, or a sequence of ints")


def _normalize_index(index: int, record_count: int) -> int:
    normalized = index + record_count if index < 0 else index
    if normalized < 0 or normalized >= record_count:
        raise IndexError(f"record index {index} out of range for {record_count} records")
    return normalized


def _read_selected_records(h5: h5py.File, selector: slice | list[int]) -> list[TransitionRecord]:
    count = _selection_count(selector, _record_count(h5))
    if count == 0:
        return []

    x_before = _read_array(h5["X_before"], selector)
    x_after = _read_array(h5["X_after"], selector)
    p = _read_array(h5["p"], selector)
    delta = _read_array(h5["delta"], selector)
    lift = _read_strings(h5["lift"], selector)
    grasp_success = _read_array(h5["grasp_success"], selector)
    settle_steps = _read_array(h5["settle_steps"], selector)
    rope_params = _read_strings(h5["rope_params"], selector)
    seed = _read_array(h5["seed"], selector)
    sim = _read_strings(h5["sim"], selector)
    timestamp = _read_strings(h5["timestamp"], selector)
    commit_hash = _read_strings(h5["commit_hash"], selector)

    records: list[TransitionRecord] = []
    for offset in range(count):
        try:
            rope_params_dict = json.loads(rope_params[offset])
        except json.JSONDecodeError as exc:
            raise TransitionDatasetError(f"record {offset} has invalid rope_params JSON") from exc
        record_data = {
            "X_before": x_before[offset],
            "X_after": x_after[offset],
            "p": int(p[offset]),
            "delta": delta[offset],
            "lift": str(lift[offset]),
            "grasp_success": bool(grasp_success[offset]),
            "settle_steps": int(settle_steps[offset]),
            "rope_params": rope_params_dict,
            "seed": int(seed[offset]),
            "sim": str(sim[offset]),
            "timestamp": str(timestamp[offset]),
            "commit_hash": str(commit_hash[offset]),
        }
        try:
            records.append(TransitionRecord.from_dict(record_data))
        except (TypeError, ValueError) as exc:
            raise TransitionDatasetError(f"record {offset} failed schema validation: {exc}") from exc
    return records


def _selection_count(selector: slice | list[int], record_count: int) -> int:
    if isinstance(selector, slice):
        start, stop, step = selector.indices(record_count)
        return len(range(start, stop, step))
    return len(selector)


def _read_array(dataset: h5py.Dataset, selector: slice | list[int]) -> np.ndarray:
    if isinstance(selector, list):
        return np.asarray([dataset[index] for index in selector])
    return np.asarray(dataset[selector])


def _read_strings(dataset: h5py.Dataset, selector: slice | list[int]) -> list[str]:
    view = dataset.asstr()
    if isinstance(selector, list):
        return [str(view[index]) for index in selector]
    return [str(value) for value in view[selector]]


def _attr_to_str(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    return str(value)
