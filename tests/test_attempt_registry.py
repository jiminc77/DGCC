"""Focused filesystem-contract tests for the append-only attempt registry."""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import pytest

import dgcc.logging.attempt_registry as attempt_registry_module
from dgcc.logging.attempt_registry import AttemptRegistry, RegistryCorruptionError


def make_registry(tmp_path: Path, tag: str = "same-tag") -> AttemptRegistry:
    return AttemptRegistry(tmp_path / "attempts", run_tag=tag, config={"a": 1}, code_sha256="c" * 64, seed=7)
def governed_receipt() -> dict[str, object]:
    return {
        "schema_version": 2,
        "content_address": "a" * 64,
        "arm": "v2-d1m",
        "seed": 7,
        "schedule_arm": "D1M",
        "mode": "confirmatory",
        "policy_delay": 2,
        "config_sha256": "b" * 64,
        "code_manifest_sha256": "c" * 64,
        "governance_sha256": "d" * 64,
        "admitted_manifest_sha256": None,
        "code_closure_sha256": "e" * 64,
        "code_closure_count": 4,
        "neff_guard_sha256": "f" * 64,
        "q1_pooled_median": 12.0,
        "qmin_pooled_median": 12.0,
        "guard_passed": True,
        "neff_guard_passed": True,
    }


def make_governed_registry(tmp_path: Path) -> AttemptRegistry:
    return AttemptRegistry(
        tmp_path / "attempts",
        run_tag="governed",
        config={"a": 1},
        code_sha256="c" * 64,
        seed=7,
        governed_launch_receipt=governed_receipt(),
    )



def test_same_tag_attempts_are_distinct_and_append_only(tmp_path: Path) -> None:
    first, second = make_registry(tmp_path), make_registry(tmp_path)
    assert first.attempt_id != second.attempt_id
    first.initialized("a" * 64)
    assert first.finalize_once("SUCCEEDED", exit_code=0)
    assert not first.finalize_once("SUCCEEDED", exit_code=0)
    assert AttemptRegistry.read_records(first.attempt_path)[-1]["disposition"] == "SUCCEEDED"


@pytest.mark.parametrize(
    "disposition",
    (
        "SUCCEEDED",
        "TECHNICAL_FAILURE",
        "PERFORMANCE_FAILURE",
        "ALGO_ABORT",
        "ABORTED",
    ),
)
def test_every_terminal_is_externally_anchored(
    tmp_path: Path, disposition: str
) -> None:
    registry = make_registry(tmp_path)
    assert registry.finalize_once(
        disposition, exit_code=0 if disposition == "SUCCEEDED" else None
    )
    terminal = AttemptRegistry.read_records(registry.attempt_path)[-1]
    receipt = registry.terminal_anchor_directory / (
        f"{registry.attempt_id}-{terminal['sha256']}.anchor.json"
    )
    assert AttemptRegistry.verify_terminal_anchor(
        registry.attempt_path, receipt
    )


def test_anchor_failure_cannot_publish_succeeded_terminal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = make_registry(tmp_path)
    def fail_anchor(*_args, **_kwargs):
        raise OSError("independent publication unavailable")

    with monkeypatch.context() as scoped:
        scoped.setattr(
            registry, "_publish_terminal_anchor", fail_anchor
        )
        with pytest.raises(OSError, match="independent publication"):
            registry.finalize_once("SUCCEEDED", exit_code=0)
    records = AttemptRegistry.read_records(registry.attempt_path)
    assert [record["phase"] for record in records] == ["PREPARING"]
    assert not (tmp_path / "attempts" / "latest-success.json").exists()
    registry.finalize_once("TECHNICAL_FAILURE")
def test_publication_pauses_only_after_durable_preparing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observed: list[bool] = []
    real_rename = attempt_registry_module.os.rename

    def paused_publish(source: Path | str, destination: Path | str) -> None:
        source_path = Path(source)
        observed.append(
            (source_path / "records.jsonl").exists()
            and AttemptRegistry.read_records(source_path)[0]["phase"] == "PREPARING"
            and not Path(destination).exists()
        )
        real_rename(source, destination)

    monkeypatch.setattr(attempt_registry_module.os, "rename", paused_publish)
    registry = make_registry(tmp_path)
    assert observed == [True]
    registry.finalize_once("ABORTED")

def test_governed_receipt_is_durable_before_publication_and_terminal_capture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: list[tuple[str, str, bytes]] = []
    real_rename = attempt_registry_module.os.rename

    def paused_publish(source: Path | str, destination: Path | str) -> None:
        source_path = Path(source)
        receipt = source_path / "reports" / "governed_launch_receipt.json"
        preparing = AttemptRegistry.read_records(source_path)[0]
        observed.append(
            (
                preparing["governed_launch_receipt_sha256"],
                hashlib.sha256(receipt.read_bytes()).hexdigest(),
                receipt.read_bytes(),
            )
        )
        real_rename(source, destination)

    monkeypatch.setattr(attempt_registry_module.os, "rename", paused_publish)
    registry = make_governed_registry(tmp_path)
    assert len(observed) == 1
    recorded_digest, file_digest, receipt_bytes = observed[0]
    assert recorded_digest == file_digest == hashlib.sha256(receipt_bytes).hexdigest()
    assert registry.finalize_once("TECHNICAL_FAILURE")
    terminal = AttemptRegistry.read_records(registry.attempt_path)[-1]
    assert terminal["artifact_sha256"]["reports/governed_launch_receipt.json"] == observed[0][1]


@pytest.mark.parametrize(
    "mutate",
    (
        lambda receipt: receipt.update({"unexpected": "value"}),
        lambda receipt: receipt.__setitem__("code_closure_sha256", "E" * 64),
        lambda receipt: receipt.__setitem__("schedule_arm", {"unsafe": "shape"}),
        lambda receipt: receipt.__setitem__("seed", 8),
        lambda receipt: receipt.__setitem__("guard_passed", False),
        lambda receipt: receipt.__setitem__("neff_guard_passed", False),
        lambda receipt: receipt.__setitem__("q1_pooled_median", float("nan")),
        lambda receipt: receipt.__setitem__("qmin_pooled_median", 20.1),
    ),
)
def test_governed_receipt_rejects_tampered_or_unsafe_schema(
    tmp_path: Path, mutate
) -> None:
    receipt = governed_receipt()
    mutate(receipt)
    with pytest.raises(ValueError, match="governed_launch_receipt"):
        AttemptRegistry(
            tmp_path / "attempts",
            run_tag="governed",
            config={"a": 1},
            code_sha256="c" * 64,
            seed=7,
            governed_launch_receipt=receipt,
        )


def test_governed_preparing_receipt_survives_later_setup_failure(tmp_path: Path) -> None:
    registry = make_governed_registry(tmp_path)
    with pytest.raises(RuntimeError, match="setup"):
        raise RuntimeError("setup failed after registry construction")
    preparing = AttemptRegistry.read_records(registry.attempt_path)[0]
    receipt = registry.attempt_path / "reports" / "governed_launch_receipt.json"
    assert preparing["governed_launch_receipt_sha256"] == hashlib.sha256(
        receipt.read_bytes()
    ).hexdigest()
    registry.finalize_once("TECHNICAL_FAILURE")
def test_governed_receipt_tampering_invalidates_published_attempt(
    tmp_path: Path,
) -> None:
    registry = make_governed_registry(tmp_path)
    (registry.attempt_path / "reports" / "governed_launch_receipt.json").write_text("{}")
    with pytest.raises(RegistryCorruptionError, match="governed launch receipt"):
        AttemptRegistry.read_records(registry.attempt_path)


def test_concurrent_finalizers_append_exactly_one_terminal(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)
    results: list[bool] = []
    barrier = threading.Barrier(3)

    def finalize() -> None:
        barrier.wait()
        results.append(registry.finalize_once("TECHNICAL_FAILURE"))

    workers = [threading.Thread(target=finalize), threading.Thread(target=finalize)]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join()
    records = AttemptRegistry.read_records(registry.attempt_path)
    assert results.count(True) == 1
    assert sum(record.get("phase") == "TERMINAL" for record in records) == 1


def test_finalization_reentry_does_not_duplicate_terminal(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)
    assert registry.finalize_once("ABORTED", detail="signal")
    assert not registry.finalize_once("ABORTED", detail="exit")
    assert sum(record.get("phase") == "TERMINAL" for record in AttemptRegistry.read_records(registry.attempt_path)) == 1


def test_anchor_fsync_before_terminal_append_leaves_prepared_evidence_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = make_registry(tmp_path)
    real_append = registry._append_record

    def fail_terminal_append(record: dict[str, object]) -> dict[str, object]:
        if record["phase"] == "TERMINAL":
            raise OSError("injected terminal append failure")
        return real_append(record)

    monkeypatch.setattr(registry, "_append_record", fail_terminal_append)
    with pytest.raises(OSError, match="terminal append"):
        registry.finalize_once("SUCCEEDED", exit_code=0)
    records = AttemptRegistry.read_records(registry.attempt_path)
    assert records[-1]["phase"] == "PREPARING"
    receipt = next(registry.terminal_anchor_directory.iterdir())
    prepared = json.loads(receipt.read_text())
    assert prepared["phase"] == "PREPARED"
    assert "terminal_disposition" not in prepared
    with pytest.raises(RegistryCorruptionError, match="anchor verification"):
        AttemptRegistry.verify_terminal_anchor(registry.attempt_path, receipt)
    registry._lock_handle.close()  # type: ignore[union-attr]
    assert AttemptRegistry.recover(tmp_path / "attempts") == [registry.attempt_id]
    assert AttemptRegistry.read_records(registry.attempt_path)[-1]["disposition"] == "ORPHANED"
def test_recovery_ignores_staging_and_orphans_unlocked_published(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)
    registry.initialized("a" * 64)
    registry._lock_handle.close()  # type: ignore[union-attr]
    assert AttemptRegistry.recover(tmp_path / "attempts") == [registry.attempt_id]
    assert AttemptRegistry.read_records(registry.attempt_path)[-1]["disposition"] == "ORPHANED"
    (tmp_path / "attempts" / ".staging" / "unpublished").mkdir(parents=True)
    assert AttemptRegistry.recover(tmp_path / "attempts") == []


def test_recovery_and_live_finalizer_do_not_race_to_two_terminals(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)
    recovered: list[str] = []
    worker = threading.Thread(target=lambda: recovered.extend(AttemptRegistry.recover(tmp_path / "attempts")))
    worker.start()
    assert registry.finalize_once("ABORTED")
    worker.join()
    assert recovered == []
    assert sum(record.get("phase") == "TERMINAL" for record in AttemptRegistry.read_records(registry.attempt_path)) == 1


def test_torn_tail_is_quarantined_without_replacing_existing_evidence(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)
    registry._lock_handle.close()  # type: ignore[union-attr]
    with registry.record_path.open("ab") as handle:
        handle.write(b'{"partial"')
    existing = tmp_path / "attempts" / ".quarantine" / registry.attempt_id
    existing.mkdir(parents=True)
    (existing / "keep").write_text("evidence")
    with pytest.raises(RegistryCorruptionError, match="torn"):
        AttemptRegistry.recover(tmp_path / "attempts")
    assert (existing / "keep").read_text() == "evidence"
    quarantined = list(existing.glob("*/attempt"))
    assert len(quarantined) == 1
    assert (quarantined[0] / "records.jsonl").read_bytes().endswith(b'{"partial"')


def test_missing_lock_and_invalid_preparing_hard_stop(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)
    registry.finalize_once("ABORTED")
    (registry.attempt_path / "owner.lock").unlink()
    with pytest.raises(RegistryCorruptionError, match="missing owner lock"):
        AttemptRegistry.recover(tmp_path / "attempts")


def test_terminal_anchor_detects_whole_file_rewrite_and_rechain(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)
    assert registry.finalize_once("PERFORMANCE_FAILURE")
    terminal = AttemptRegistry.read_records(registry.attempt_path)[-1]
    receipt = registry.terminal_anchor_directory / (
        f"{registry.attempt_id}-{terminal['sha256']}.anchor.json"
    )
    assert AttemptRegistry.verify_terminal_anchor(registry.attempt_path, receipt)
    rewritten = AttemptRegistry.read_records(registry.attempt_path)
    previous = "0" * 64
    for record in rewritten:
        record.pop("sha256")
        record["previous_sha256"] = previous
        if record.get("phase") == "TERMINAL":
            record["detail"] = "rewritten"
        canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        record["sha256"] = hashlib.sha256(canonical).hexdigest()
        previous = record["sha256"]
    registry.record_path.write_text("".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in rewritten))
    with pytest.raises(RegistryCorruptionError, match="anchor"):
        AttemptRegistry.verify_terminal_anchor(registry.attempt_path, receipt)


def test_artifacts_may_be_recorded_before_algorithm_abort(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)
    artifact = registry.attempt_path / "checkpoint.bin"
    artifact.write_bytes(b"checkpoint")
    registry.record_artifacts([artifact])
    assert registry.finalize_once("ALGO_ABORT", artifact_paths=[artifact])
    records = AttemptRegistry.read_records(registry.attempt_path)
    assert records[-2]["phase"] == "ARTIFACTS"
    assert records[-1]["disposition"] == "ALGO_ABORT"
def test_requested_invalid_artifacts_fail_closed(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)
    missing = registry.attempt_path / "missing.bin"
    directory = registry.attempt_path / "directory"
    directory.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    for artifact in (missing, directory, outside):
        with pytest.raises((FileNotFoundError, ValueError)):
            registry.record_artifacts([artifact])
        with pytest.raises((FileNotFoundError, ValueError)):
            registry.finalize_once("ALGO_ABORT", artifact_paths=[artifact])
    assert registry.finalize_once("ALGO_ABORT")
    assert AttemptRegistry.read_records(registry.attempt_path)[-1]["phase"] == "TERMINAL"


def test_latest_success_reconciliation_failure_does_not_invert_terminal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = make_registry(tmp_path)

    def fail_reconcile(*_args: object) -> None:
        raise OSError("derived index unavailable")

    monkeypatch.setattr(registry, "_reconcile_latest_success", fail_reconcile)
    with pytest.warns(RuntimeWarning, match="latest-success reconciliation failed"):
        assert registry.finalize_once("SUCCEEDED", exit_code=0)
    assert AttemptRegistry.read_records(registry.attempt_path)[-1]["disposition"] == "SUCCEEDED"
def test_latest_success_is_not_changed_by_newer_failure(tmp_path: Path) -> None:
    success = make_registry(tmp_path)
    success.finalize_once("SUCCEEDED", exit_code=0)
    failure = make_registry(tmp_path)
    failure.finalize_once("TECHNICAL_FAILURE", exit_code=2)
    latest = json.loads((tmp_path / "attempts" / "latest-success.json").read_text())
    assert latest["attempt_id"] == success.attempt_id
def test_latest_success_uses_attempt_id_as_same_second_tiebreaker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(attempt_registry_module, "_utc_now", lambda: "2026-01-01T00:00:00Z")
    first, second = make_registry(tmp_path), make_registry(tmp_path)
    first.finalize_once("SUCCEEDED")
    second.finalize_once("SUCCEEDED")
    latest = json.loads((tmp_path / "attempts" / "latest-success.json").read_text())
    assert latest["attempt_id"] == max(first.attempt_id, second.attempt_id)


def test_recovery_rejects_symlink_root_child_and_lock(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)
    root_link = tmp_path / "root-link"
    root_link.symlink_to(tmp_path / "attempts", target_is_directory=True)
    with pytest.raises(RegistryCorruptionError, match="root is symlink"):
        AttemptRegistry.recover(root_link)
    registry._lock_handle.close()  # type: ignore[union-attr]
    (tmp_path / "attempts" / "linked").symlink_to(registry.attempt_path, target_is_directory=True)
    with pytest.raises(RegistryCorruptionError, match="symlink attempt"):
        AttemptRegistry.recover(tmp_path / "attempts")
    (tmp_path / "attempts" / "linked").unlink()
    lock = registry.attempt_path / "owner.lock"
    lock.unlink()
    lock.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(RegistryCorruptionError, match="missing owner lock"):
        AttemptRegistry.recover(tmp_path / "attempts")


def test_artifact_hardlink_to_registry_internals_is_rejected(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)
    for name in ("records.jsonl", "owner.lock"):
        alias = registry.attempt_path / f"{name}.alias"
        alias.hardlink_to(registry.attempt_path / name)
        with pytest.raises(ValueError, match="not an allowed"):
            registry.record_artifacts([alias])


def test_allowed_artifact_paths_filter_entrypoint_candidates(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)
    artifact = registry.attempt_path / "reports" / "result.txt"
    artifact.parent.mkdir()
    artifact.write_text("ok", encoding="utf-8")
    (registry.attempt_path / ".hidden").write_text("private", encoding="utf-8")
    nested_hidden = registry.attempt_path / "reports" / ".scratch"
    nested_hidden.write_text("private", encoding="utf-8")
    (registry.attempt_path / "directory").mkdir()
    (registry.attempt_path / "symlink").symlink_to(artifact)
    (registry.attempt_path / "records.alias").hardlink_to(registry.record_path)

    candidates = AttemptRegistry.allowed_artifact_paths(registry.attempt_path)
    assert candidates == [artifact]
    assert registry.finalize_once("SUCCEEDED", artifact_paths=candidates)
    terminal = AttemptRegistry.read_records(registry.attempt_path)[-1]
    assert terminal["artifact_sha256"] == {
        "reports/result.txt": hashlib.sha256(b"ok").hexdigest()
    }


def test_artifact_hashing_revalidates_discovered_path_before_terminal(
    tmp_path: Path,
) -> None:
    registry = make_registry(tmp_path)
    artifact = registry.attempt_path / "result.txt"
    artifact.write_text("original", encoding="utf-8")
    candidates = AttemptRegistry.allowed_artifact_paths(registry.attempt_path)
    artifact.unlink()
    artifact.symlink_to(tmp_path / "outside.txt")
    with pytest.raises(ValueError, match="invalid|allowed"):
        registry.finalize_once("SUCCEEDED", artifact_paths=candidates)
    assert AttemptRegistry.read_records(registry.attempt_path)[-1]["phase"] == "PREPARING"
    artifact.unlink()
    assert registry.finalize_once("TECHNICAL_FAILURE")


def test_scalar_record_json_is_registry_corruption(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)
    registry.record_path.write_text("[]\n")
    with pytest.raises(RegistryCorruptionError, match="invalid record frame"):
        AttemptRegistry.read_records(registry.attempt_path)
def test_registry_rejects_duplicate_phase_invalid_digest_and_symlink_recovery(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)
    with pytest.raises(ValueError):
        registry.initialized("not-a-digest")
    registry.initialized("a" * 64)
    with pytest.raises(RegistryCorruptionError, match="INITIALIZED"):
        registry.initialized("b" * 64)
    link = tmp_path / "attempts" / "linked-attempt"
    link.symlink_to(registry.attempt_path, target_is_directory=True)
    with pytest.raises(RegistryCorruptionError, match="symlink attempt"):
        AttemptRegistry.recover(tmp_path / "attempts")
