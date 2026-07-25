from __future__ import annotations

import errno
import hashlib
import importlib.util
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import pytest

from dgcc.logging.asset_firewall import (
    AssetAccessError,
    AssetFirewall,
    generate_r3_r4_existence_receipt,
    load_launch_asset_manifest,
    persist_launch_receipts,
    read_launch_asset_snapshot,
)
import dgcc.logging.asset_firewall as asset_firewall_module
from dgcc.rl.sprint_arms import SprintTD3Agent
from dgcc.rl.td3 import TD3Agent

ROOT = Path(__file__).parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(tmp_path: Path) -> Path:
    assets = []
    for kind, count, suffix in (("checkpoint", 10, ".pt"), ("raw_gz", 3, ".gz"), ("panel", 1, ".npz")):
        for number in range(count):
            source = tmp_path / "source" / f"{kind}-{number}{suffix}"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(f"{kind}:{number}".encode())
            assets.append(
                {
                    "kind": kind,
                    "source": str(source),
                    "destination": f"payload/{source.name}",
                    "sha256": digest(source),
                }
            )
    manifest = tmp_path / "provenance.json"
    manifest.write_text(json.dumps({"assets": assets}))
    return manifest


def test_snapshot_enumerates_only_approved_payloads_and_makes_them_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tool = load_script("wp03d_snapshot")
    monkeypatch.setattr(tool, "MIN_FREE_BYTES", 0)
    destination = tmp_path / "snapshot"
    with pytest.raises(ValueError, match="snapshot destination"):
        tool.snapshot(provenance(tmp_path), destination, source_roots=(tmp_path,))
    result = tool.snapshot(
        provenance(tmp_path),
        destination,
        source_roots=(tmp_path,),
        snapshot_root=tmp_path,
    )
    assert len(result["assets"]) == 14
    assert result["source_mutation_attestation"]["zero_mutation_observed"]
    assert {path.relative_to(destination) for path in destination.rglob("*") if path.is_file()} == {
        *(Path(asset["destination"]) for asset in result["assets"]),
        Path("MANIFEST.json"),
    }
    for asset in result["assets"]:
        assert stat.S_IMODE((destination / asset["destination"]).stat().st_mode) == 0o444

    tool.verify_snapshot(destination, snapshot_root=tmp_path)


def test_snapshot_rejects_bad_enumeration_disk_and_hash_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tool = load_script("wp03d_snapshot")
    document = json.loads(provenance(tmp_path).read_text())
    document["assets"].pop()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        tool.load_provenance(bad)
    monkeypatch.setattr(tool.shutil, "disk_usage", lambda _: type("D", (), {"free": 0})())
    with pytest.raises(OSError):
        tool.snapshot(provenance(tmp_path), tmp_path / "disk", source_roots=(tmp_path,), snapshot_root=tmp_path)
    monkeypatch.setattr(tool, "MIN_FREE_BYTES", 0)
    hashes = iter(["a", "b", "c", "a", "b", "c"])
    monkeypatch.setattr(tool, "sha256_file", lambda _: next(hashes))
    with pytest.raises(RuntimeError):
        tool.snapshot(provenance(tmp_path), tmp_path / "mismatch", source_roots=(tmp_path,), snapshot_root=tmp_path)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "stale").write_text("stale")
    with pytest.raises(FileExistsError):
        tool.snapshot(provenance(tmp_path), occupied, source_roots=(tmp_path,), snapshot_root=tmp_path)


def test_firewall_audits_forbidden_attempt_and_reports_zero_access(tmp_path: Path) -> None:
    protected = tmp_path / "heldout.bin"
    protected.write_bytes(b"synthetic")
    audit = tmp_path / "audit.jsonl"
    with pytest.raises(ValueError):
        AssetFirewall({}, audit)
    with pytest.raises(ValueError, match="must never appear"):
        AssetFirewall({protected: digest(protected)}, audit)
    firewall = AssetFirewall({tmp_path / "ordinary.bin": "0" * 64}, audit)
    with pytest.raises(AssetAccessError):
        firewall.read_bytes(protected)
    assert firewall.zero_access_receipt()["zero_access"] is False
    fresh = AssetFirewall({tmp_path / "ordinary.bin": "0" * 64}, tmp_path / "fresh.jsonl")
    assert fresh.zero_access_receipt()["zero_access"] is True


def test_launch_manifest_rejects_protected_asset_even_when_hash_pinned(
    tmp_path: Path,
) -> None:
    protected = tmp_path / "probe.bin"
    protected.write_bytes(b"synthetic")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "assets": [
                    {
                        "path": str(protected),
                        "sha256": digest(protected),
                        "role": "forbidden",
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="must never appear"):
        load_launch_asset_manifest(manifest, digest(manifest), tmp_path / "audit.jsonl")


def test_firewall_consumers_use_verified_bytes_not_reopened_paths(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "config.yaml"
    asset.write_bytes(b"task: t2\n")
    firewall = AssetFirewall(
        {asset: digest(asset)},
        tmp_path / "audit-bytes.jsonl",
        {asset: "config"},
    )
    canonical, payload = firewall.read_bytes(asset, operation="config-load")
    assert canonical == asset.resolve()
    assert payload == b"task: t2\n"

    asset.unlink()
    asset.symlink_to(tmp_path / "heldout-target.yaml")
    with pytest.raises(AssetAccessError):
        firewall.read_bytes(asset, operation="config-load")
def test_descriptor_bound_open_survives_component_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    component = tmp_path / "authorized"
    component.mkdir()
    asset = component / "config.yaml"
    asset.write_bytes(b"approved")
    malicious = tmp_path / "replacement"
    malicious.mkdir()
    (malicious / "config.yaml").write_bytes(b"unapproved")
    firewall = AssetFirewall(
        {asset: digest(asset)}, tmp_path / "audit-race.jsonl", {asset: "config"}
    )
    original_open = asset_firewall_module.os.open
    swapped = False

    def swap_before_leaf_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == "config.yaml" and not swapped:
            swapped = True
            component.rename(tmp_path / "parked")
            component.symlink_to(malicious, target_is_directory=True)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(asset_firewall_module.os, "open", swap_before_leaf_open)
    assert firewall.read_bytes(asset, required_role="config")[1] == b"approved"


def test_required_role_is_checked_before_any_asset_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = tmp_path / "config.yaml"
    asset.write_bytes(b"approved")
    firewall = AssetFirewall(
        {asset: digest(asset)}, tmp_path / "audit-role.jsonl", {asset: "config"}
    )
    monkeypatch.setattr(
        asset_firewall_module,
        "_open_descriptor_bound",
        lambda _: pytest.fail("asset bytes were opened before role validation"),
    )
    with pytest.raises(AssetAccessError, match="manifest role"):
        firewall.read_bytes(asset, required_role="execution_governance")


def test_preflight_validates_authenticated_byte_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = load_script("v2_protocol_preflight")
    observed = {}
    monkeypatch.setattr(
        preflight,
        "validate_manifest",
        lambda manifest, governance, **_expected: observed.update(
            manifest=manifest, governance=governance
        )
        or {"ok": True},
    )
    monkeypatch.setattr(
        preflight,
        "_validate_neff_guard_bytes",
        lambda guard, _manifest, _governance: observed.update(guard=guard) or {},
    )
    assert preflight.validate_manifest_bytes(
        b'{"source":"authenticated"}',
        b'{"governance":"authenticated"}',
        neff_guard_bytes=b'{"guard":"authenticated"}',
    ) == {"ok": True}
    assert observed == {
        "manifest": {"source": "authenticated"},
        "governance": {"governance": "authenticated"},
        "guard": b'{"guard":"authenticated"}',
    }


def test_bootstrap_snapshot_rejects_config_replacement(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_bytes(b"task: t2\n")
    manifest = tmp_path / "assets.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "assets": [
                    {
                        "path": str(config),
                        "sha256": digest(config),
                        "role": "config",
                    }
                ],
            }
        )
    )
    _, payload = read_launch_asset_snapshot(
        manifest, digest(manifest), config, "config"
    )
    assert payload == b"task: t2\n"
    config.write_bytes(b"task: changed\n")
    with pytest.raises(AssetAccessError, match="SHA-256"):
        read_launch_asset_snapshot(
            manifest, digest(manifest), config, "config"
        )


def test_r3_r4_generator_uses_stat_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protected, fresh = tmp_path / "probe", tmp_path / "fresh-heldout"
    protected.write_bytes(b"x")
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: pytest.fail("content was opened"))
    receipt = generate_r3_r4_existence_receipt([protected], [fresh])
    assert receipt["content_opened"] is False
    assert receipt["R3"]["records"][0]["uid"] == __import__("os").getuid()
    assert receipt["R4"]["pass"] is True
    assert receipt["R4"]["records"][0]["exists"] is False


def test_r3_r4_rejects_empty_symlink_and_lstat_error_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "fresh-heldout"
    with pytest.raises(ValueError):
        generate_r3_r4_existence_receipt([], [missing])
    with pytest.raises(ValueError):
        generate_r3_r4_existence_receipt([missing], [])

    symlink = tmp_path / "fresh-heldout-link"
    symlink.symlink_to(missing)
    symlink_receipt = generate_r3_r4_existence_receipt([missing], [symlink])
    assert symlink_receipt["R4"]["records"][0]["symlink"] is True
    assert symlink_receipt["R4"]["pass"] is False

    def fail_lstat(_: str) -> None:
        raise OSError(errno.EIO, "I/O error")

    monkeypatch.setattr("dgcc.logging.asset_firewall.os.lstat", fail_lstat)
    error_receipt = generate_r3_r4_existence_receipt([missing], [missing])
    assert error_receipt["R4"]["records"][0]["error"] == "OSError"
    assert error_receipt["R4"]["pass"] is False


def snapshot_for_neff(tmp_path: Path) -> Path:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    assets = []
    for kind, count in (("checkpoint", 10), ("raw_gz", 3), ("panel", 1)):
        for index in range(count):
            payload = snapshot / f"{kind}-{index}"
            payload.write_bytes(f"{kind}{index}".encode())
            assets.append({"kind": kind, "destination": payload.name, "sha256": digest(payload)})
    (snapshot / "MANIFEST.json").write_text(
        json.dumps({"assets": assets, "source_mutation_attestation": {"zero_mutation_observed": True}})
    )
    return snapshot


def write_score_sidecar(tool: object, snapshot: Path, scores: Path) -> None:
    manifest = json.loads((snapshot / "MANIFEST.json").read_text())
    document = {
        "schema_version": 1,
        "artifact": {
            "path": scores.name,
            "sha256": digest(scores),
            "arrays": {
                "q1_scores": {"dtype": "float64", "shape": [10, 300, 32]},
                "qmin_scores": {"dtype": "float64", "shape": [10, 300, 32]},
            },
        },
        "protocol": {"device": "cpu", "contact_weight_beta": tool.CONTACT_WEIGHT_BETA, "forward_only": True},
        "code": {
            "score_producer_sha256": tool.EXTERNAL_Q_SCORE_PRODUCER_SHA256
        },
        "inputs": {
            "snapshot_manifest_sha256": digest(snapshot / "MANIFEST.json"),
            "checkpoint_sha256": [item["sha256"] for item in manifest["assets"] if item["kind"] == "checkpoint"],
            "panel_sha256": next(item["sha256"] for item in manifest["assets"] if item["kind"] == "panel"),
        },
    }
    scores.with_suffix(".json").write_text(json.dumps(document))
def test_neff_shape_median_range_and_manifest_pin(tmp_path: Path) -> None:
    tool = load_script("v2_neff_guard")
    snapshot = snapshot_for_neff(tmp_path)
    manifest_sha256 = digest(snapshot / "MANIFEST.json")
    scores = np.full((10, 300, 32), -100.0, dtype=np.float64)
    scores[..., :10] = 0.0
    inputs = tmp_path / "scores.npz"
    np.savez(inputs, q1_scores=scores, qmin_scores=scores)
    write_score_sidecar(tool, snapshot, inputs)
    pins = tool.run(snapshot, inputs, tmp_path / "neff.npz", manifest_sha256)
    values = np.load(tmp_path / "neff.npz")
    assert values["q1_neff"].shape == (10, 300)
    assert pins["q1_pooled_median"] == pytest.approx(10.0)
    with pytest.raises(ValueError):
        tool.run(snapshot, inputs, tmp_path / "bad-pin.npz", "0" * 64)
    np.savez(inputs, q1_scores=np.ones((9, 300, 32), dtype=np.float64), qmin_scores=scores)
    write_score_sidecar(tool, snapshot, inputs)
    with pytest.raises(ValueError):
        tool.run(snapshot, inputs, tmp_path / "bad.npz", manifest_sha256)
    np.savez(inputs, q1_scores=np.ones((10, 300, 2), dtype=np.float64), qmin_scores=np.ones((10, 300, 2), dtype=np.float64))
    write_score_sidecar(tool, snapshot, inputs)
    with pytest.raises(ValueError):
        tool.run(snapshot, inputs, tmp_path / "bad-shape.npz", manifest_sha256)
    concentrated = np.full((10, 300, 32), -100.0, dtype=np.float64)
    concentrated[..., :2] = 0.0
    np.savez(inputs, q1_scores=concentrated, qmin_scores=concentrated)
    write_score_sidecar(tool, snapshot, inputs)
    with pytest.raises(RuntimeError):
        tool.run(snapshot, inputs, tmp_path / "range.npz", manifest_sha256)
def test_neff_rejects_missing_or_mismatched_score_provenance_and_output_replacement(tmp_path: Path) -> None:
    tool = load_script("v2_neff_guard")
    snapshot = snapshot_for_neff(tmp_path)
    manifest_sha256 = digest(snapshot / "MANIFEST.json")
    scores = np.full((10, 300, 32), -100.0, dtype=np.float64)
    scores[..., :10] = 0.0
    inputs, output = tmp_path / "scores.npz", tmp_path / "neff.npz"
    np.savez(inputs, q1_scores=scores, qmin_scores=scores)
    with pytest.raises(ValueError, match="sidecar"):
        tool.run(snapshot, inputs, output, manifest_sha256)
    write_score_sidecar(tool, snapshot, inputs)
    sidecar = json.loads(inputs.with_suffix(".json").read_text())
    sidecar["inputs"]["panel_sha256"] = "0123456789abcdef" * 4
    inputs.with_suffix(".json").write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="provenance"):
        tool.run(snapshot, inputs, output, manifest_sha256)
    write_score_sidecar(tool, snapshot, inputs)
    with pytest.raises(ValueError, match="alias"):
        tool.run(snapshot, inputs, inputs, manifest_sha256)
    assert tool.run(snapshot, inputs, output, manifest_sha256)["guard_passed"]
    with pytest.raises(FileExistsError):
        tool.run(snapshot, inputs, output, manifest_sha256)


class _MaliciousPayload:
    def __reduce__(self):
        return (exec, ("raise RuntimeError('unsafe pickle executed')",))


@pytest.mark.parametrize("agent_type", (TD3Agent, SprintTD3Agent))
def test_checkpoint_loading_rejects_untrusted_pickle_before_execution(
    agent_type, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "malicious.pt"
    torch.save(_MaliciousPayload(), checkpoint)
    with pytest.raises(Exception, match="Weights only load failed|Unsupported global"):
        agent_type().load_checkpoint(checkpoint, eval_only=True)


def test_sprint_checkpoint_loading_rejects_non_mapping_payload(tmp_path: Path) -> None:
    checkpoint = tmp_path / "scalar.pt"
    torch.save(17, checkpoint)
    with pytest.raises(ValueError, match="mapping"):
        SprintTD3Agent().load_checkpoint(checkpoint, eval_only=True)
def test_firewall_fails_closed_for_tampered_audit_and_symlink_receipt(tmp_path: Path) -> None:
    asset = tmp_path / "ordinary.bin"
    asset.write_bytes(b"allowed")
    firewall = AssetFirewall({asset: digest(asset)}, tmp_path / "audit.jsonl")
    firewall.audit_path.write_bytes(b'{"truncated"')
    with pytest.raises(AssetAccessError, match="audit"):
        firewall.zero_access_receipt()
    target = tmp_path / "fresh-heldout"
    target.symlink_to(asset)
    receipt = generate_r3_r4_existence_receipt([tmp_path / "probe"], [target])
    assert receipt["R4"]["records"][0]["symlink"] is True
    assert receipt["R4"]["pass"] is False


def test_receipt_bundle_is_cross_linked_and_unique(tmp_path: Path) -> None:
    asset = tmp_path / "ordinary.bin"
    asset.write_bytes(b"allowed")
    firewall = AssetFirewall({asset: digest(asset)}, tmp_path / "audit.jsonl")
    manifest = {"manifest_path": "manifest.json", "manifest_sha256": "a" * 64, "assets": []}
    first = persist_launch_receipts(tmp_path, manifest, firewall)
    second = persist_launch_receipts(tmp_path, manifest, firewall)
    r1 = json.loads(Path(first["r1"]).read_text())
    r2 = json.loads(Path(first["r2"]).read_text())
    assert Path(first["r1"]).parent != Path(second["r1"]).parent
    assert r1["bundle_id"] == r2["bundle_id"]
    assert r1["peer_receipt"] == Path(first["r2"]).name
def test_manifest_scalar_json_is_a_contract_error(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]")
    with pytest.raises(ValueError, match="malformed"):
        load_launch_asset_manifest(manifest, digest(manifest), tmp_path / "audit.jsonl")


def test_receipt_bundle_publication_refuses_existing_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    asset = tmp_path / "ordinary.bin"
    asset.write_bytes(b"allowed")
    firewall = AssetFirewall({asset: digest(asset)}, tmp_path / "audit.jsonl")
    root = tmp_path / "reports" / "receipt-bundles"
    root.mkdir(parents=True)
    (root / "collision").mkdir()
    monkeypatch.setattr(
        asset_firewall_module.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="collision"),
    )
    with pytest.raises(FileExistsError):
        persist_launch_receipts(
            tmp_path,
            {"manifest_path": "manifest.json", "manifest_sha256": "a" * 64, "assets": []},
            firewall,
        )
    assert not (root / ".collision.tmp").exists()
