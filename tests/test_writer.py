from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from dgcc.logging.schema import TransitionRecord
from dgcc.logging.writer import (
    SCHEMA_VERSION,
    TransitionDatasetError,
    TransitionWriter,
    read_transitions,
    write_transitions,
)


def make_meta() -> dict[str, object]:
    return {
        "config": "rope:\n  length_m: 1.0\n  lift: 유니코드\ncollector: writer-test\n",
        "commit_hash": "test-commit-ø-123",
        "creation_time": "2026-07-02T00:00:00Z",
        "collector": {"name": "writer-test", "batch_size": 17},
    }


def make_record(index: int) -> TransitionRecord:
    base = np.arange(96, dtype=float).reshape(32, 3) / 100.0
    x_before = base + index * 0.01
    x_after = x_before + np.array(
        [0.001 * (index + 1), -0.0005 * index, 0.00025 * (index % 7)],
        dtype=float,
    )
    lifts = ["low", "high", "弧線", "grasp-🪢", "mañana"]
    lift = f"{lifts[index % len(lifts)]}-{index}"
    return TransitionRecord(
        X_before=x_before,
        X_after=x_after,
        p=index - 25,
        delta=np.array(
            [(-1.0) ** index * (index + 1) * 0.001, index * -0.002, 0.03 + index * 0.0001],
            dtype=float,
        ),
        lift=lift,
        grasp_success=index % 3 == 0,
        settle_steps=index - 5,
        rope_params={
            "length_m": 1.0 + index * 0.01,
            "n_segments": 32 + (index % 5),
            "bend_stiffness": 0.1 + index * 0.001,
            "twist_stiffness": 0.2 + index * 0.002,
            "friction": 0.3 + (index % 4) * 0.01,
            "radius": 0.005 + index * 0.00001,
            "label": lift,
            "flags": {"odd": bool(index % 2), "group": index % 7},
            "schedule": [index, index - 1, f"점-{index}"],
        },
        seed=-1000 + index,
        sim=f"unit-sim-{index % 4}-Δ",
        timestamp=f"2026-07-02T00:{index:02d}:00Z",
        commit_hash=f"record-commit-{index:02x}",
    )


def make_records(count: int = 50) -> list[TransitionRecord]:
    return [make_record(index) for index in range(count)]


def assert_record_equal(actual: TransitionRecord, expected: TransitionRecord) -> None:
    assert np.allclose(actual.X_before, expected.X_before)
    assert np.allclose(actual.X_after, expected.X_after)
    assert np.allclose(actual.delta, expected.delta)
    assert actual.p == expected.p
    assert actual.lift == expected.lift
    assert actual.grasp_success is expected.grasp_success
    assert actual.settle_steps == expected.settle_steps
    assert actual.rope_params == expected.rope_params
    assert actual.seed == expected.seed
    assert actual.sim == expected.sim
    assert actual.timestamp == expected.timestamp
    assert actual.commit_hash == expected.commit_hash


def assert_records_equal(actual: list[TransitionRecord], expected: list[TransitionRecord]) -> None:
    assert len(actual) == len(expected)
    for actual_record, expected_record in zip(actual, expected, strict=True):
        assert_record_equal(actual_record, expected_record)


def assert_writer_layout(path: Path, expected_count: int, expected_meta: dict[str, object]) -> None:
    with h5py.File(path, "r") as h5:
        assert h5.attrs["schema_version"] == SCHEMA_VERSION
        assert int(h5.attrs["record_count"]) == expected_count
        assert json.loads(h5.attrs["meta_json"]) == expected_meta
        assert h5.attrs["config"] == expected_meta["config"]
        assert h5.attrs["commit_hash"] == expected_meta["commit_hash"]
        assert h5.attrs["creation_time"] == expected_meta["creation_time"]
        assert set(h5.keys()) == set(TransitionRecord.FIELD_NAMES)

        assert h5["X_before"].shape == (expected_count, 32, 3)
        assert h5["X_after"].shape == (expected_count, 32, 3)
        assert h5["delta"].shape == (expected_count, 3)
        assert h5["p"].shape == (expected_count,)
        assert h5["grasp_success"].shape == (expected_count,)
        assert h5["settle_steps"].shape == (expected_count,)
        assert h5["seed"].shape == (expected_count,)
        assert h5["X_before"].dtype == np.dtype("float64")
        assert h5["p"].dtype == np.dtype("int64")
        assert h5["grasp_success"].dtype == np.dtype("bool")
        for field in ["lift", "rope_params", "sim", "timestamp", "commit_hash"]:
            string_info = h5py.check_string_dtype(h5[field].dtype)
            assert string_info is not None
            assert string_info.encoding == "utf-8"


def test_write_read_round_trip_and_slice(tmp_path: Path) -> None:
    records = make_records(50)
    meta = make_meta()
    path = tmp_path / "transitions.h5"
    payload = [record if index % 2 else record.to_dict() for index, record in enumerate(records)]

    write_transitions(path, payload, meta)

    restored, restored_meta = read_transitions(path)
    assert restored_meta == meta
    assert_records_equal(restored, records)

    sliced, sliced_meta = read_transitions(path, slice(10, 20))
    assert sliced_meta == meta
    assert_records_equal(sliced, records[10:20])
    assert_writer_layout(path, expected_count=50, expected_meta=meta)


def test_transition_writer_appends_in_batches_like_one_shot(tmp_path: Path) -> None:
    records = make_records(50)
    meta = make_meta()
    one_shot_path = tmp_path / "one-shot.h5"
    batch_path = tmp_path / "batched.h5"

    write_transitions(one_shot_path, records, meta)
    with TransitionWriter(batch_path, meta=meta, mode="w") as writer:
        writer.append(records[:11])
        assert len(writer) == 11
    with TransitionWriter(batch_path, mode="a") as writer:
        writer.append([record.to_dict() for record in records[11:37]])
        writer.flush()
        assert len(writer) == 37
    with TransitionWriter(batch_path, mode="a") as writer:
        writer.append(records[37:])

    one_shot_records, one_shot_meta = read_transitions(one_shot_path)
    batch_records, batch_meta = read_transitions(batch_path)
    assert one_shot_meta == batch_meta == meta
    assert_records_equal(one_shot_records, records)
    assert_records_equal(batch_records, records)
    assert_writer_layout(batch_path, expected_count=50, expected_meta=meta)


@pytest.mark.parametrize(
    ("meta", "match"),
    [
        ({"commit_hash": "abc"}, "config"),
        ({"config": "x"}, "commit_hash"),
        ({"config": {"not": "a string"}, "commit_hash": "abc"}, "config must be a str"),
        ({"config": "x", "commit_hash": 123}, "commit_hash must be a str"),
    ],
)
def test_missing_or_invalid_metadata_raises(tmp_path: Path, meta: dict[str, object], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        write_transitions(tmp_path / "invalid-meta.h5", [], meta)


def test_read_transitions_raises_cleanly_for_missing_and_corrupt_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="transition file not found"):
        read_transitions(tmp_path / "missing.h5")

    corrupt_path = tmp_path / "corrupt.h5"
    corrupt_path.write_bytes(b"not an hdf5 transition dataset")
    with pytest.raises(TransitionDatasetError, match="failed to open transition file"):
        read_transitions(corrupt_path)


def test_empty_record_list_creates_header_only_file(tmp_path: Path) -> None:
    meta = make_meta()
    path = tmp_path / "empty.h5"

    write_transitions(path, [], meta)

    records, restored_meta = read_transitions(path)
    assert records == []
    assert restored_meta == meta
    assert_writer_layout(path, expected_count=0, expected_meta=meta)
