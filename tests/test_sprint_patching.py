import hashlib
import json

import h5py
import numpy as np
import pytest
import torch

import dgcc.analysis.sprint_patching as sprint_patching
from dgcc.analysis.sprint_patching import (
    N_PAIRS_PER_RUN,
    PAIRING_SEED,
    a0_interchange,
    a1_projected_splice,
    deterministic_pairing,
    estimate_from_bundle,
    graded_ablation,
    load_probe_manifest,
    load_projector,
    mean_ablation,
    ranking_change,
    rescale_mask,
)


def _bundle(tmp_path, records=()):
    probe = tmp_path / "probe.h5"
    with h5py.File(probe, "w") as handle:
        handle["h_p"] = np.zeros((2, 256))
        handle["estimator_records"] = json.dumps(list(records))
    digest = hashlib.sha256(probe.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "files": {digest: {"path": str(probe), "sha256": digest, "size": probe.stat().st_size}}}))
    return load_probe_manifest(manifest), manifest


def _projector(tmp_path, monkeypatch):
    torch.manual_seed(1)
    value = torch.linalg.qr(torch.randn(256, 24)).Q.T
    path = tmp_path / "projector.pt"
    torch.save(value, path)
    # Test fixture P replaces the production pin; load_projector exposes no override.
    monkeypatch.setattr(sprint_patching, "PROJECTOR_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    return load_projector(path)


def _rows(run=0):
    real, null = [], []
    for index in range(N_PAIRS_PER_RUN):
        identity = {"pair_id": f"p-{index}", "goal_id": f"g-{index}", "run": run, "operator": "a0"}
        real.append({"arm": "real", **identity, "baseline_q": [3, 2, 1], "patched_q": [1, 2, 3]})
        null.append({"arm": "null", **identity, "baseline_q": [3, 2, 1], "patched_q": [3, 2, 1]})
    return real, null


def test_operators_and_graded(tmp_path, monkeypatch):
    projector = _projector(tmp_path, monkeypatch)
    r, d = torch.randn(3, 256), torch.randn(3, 256)
    out = a1_projected_splice(r, d, projector)
    assert torch.allclose(out @ projector.value.T, d @ projector.value.T, atol=1e-5)
    assert torch.equal(a0_interchange(r, d), d)
    assert torch.equal(graded_ablation(r, torch.zeros(256), 1.0), mean_ablation(r, torch.zeros(256)))
    with pytest.raises(TypeError):
        a1_projected_splice(r, d, projector.value)


def test_projector_rejects_unpinned_or_nonorthonormal(tmp_path, monkeypatch):
    path = tmp_path / "bad.pt"
    torch.save(torch.zeros(24, 256), path)
    monkeypatch.setattr(sprint_patching, "PROJECTOR_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError):
        load_projector(path)


def test_projector_rejects_valid_self_digested_replacement(tmp_path):
    path = tmp_path / "replacement.pt"
    torch.save(torch.linalg.qr(torch.randn(256, 24)).Q.T, path)
    with pytest.raises(ValueError):
        load_projector(path)


def test_rescale_mask():
    assert torch.equal(rescale_mask(2.5), torch.tensor([1, *([2.5] * 7), 1, *([2.5] * 7), *([1] * 8)]))


FROZEN_TIE_DONOR_ORDER = (
    "b", "b", "b", "a", "a", "b", "a", "a", "a", "a", "a", "a", "b", "b", "a", "a", "a", "b", "a", "a",
    "a", "b", "b", "a", "a", "a", "a", "a", "a", "b", "b", "a", "b", "a", "b", "b", "a", "b", "b", "b",
    "a", "b", "b", "b", "a", "b", "a", "a", "b", "a", "a", "a", "b", "b", "a", "a", "a", "a", "a", "b",
    "a", "a", "b", "a", "a", "b", "a", "a", "b", "a", "b", "b", "b", "a", "a", "a", "b", "b", "a", "a",
    "b", "a", "b", "b", "b", "b", "a", "a", "b", "a", "a", "a", "a", "a", "a", "b", "a", "b", "a", "b",
)


def test_confirmatory_pairing_requires_100_unique_goals_and_pins_seed():
    recipients = [{"goal_id": f"g-{i}", "p": 2, "nuisance": "n", "delta_m": np.zeros(24)} for i in range(100)]
    donors = [
        {"id": f"{label}-{i}", "goal_id": f"g-{i}", "p": 2, "nuisance": "n", "delta_m": np.zeros(24)}
        for i in range(100) for label in ("a", "b")
    ]
    pairs = deterministic_pairing(recipients, donors)
    assert tuple(donor["id"].split("-")[0] for _, donor in pairs) == FROZEN_TIE_DONOR_ORDER
    assert PAIRING_SEED == 20260723
    with pytest.raises(ValueError): deterministic_pairing(recipients[:-1], donors)
    recipients[-1]["goal_id"] = recipients[0]["goal_id"]
    with pytest.raises(ValueError): deterministic_pairing(recipients, donors)


def test_manifest_bundle_is_verified_and_digest_bound(tmp_path):
    bundle, manifest = _bundle(tmp_path)
    assert len(bundle) == 1
    assert bundle.manifest_sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()
    manifest.write_text(json.dumps({"schema_version": 1, "files": {"0" * 64: {"path": "missing", "sha256": "0" * 64, "size": 0}}}))
    with pytest.raises(ValueError): load_probe_manifest(manifest)


def test_kendall_disruption_is_positive_and_real_minus_null_flows_to_bca(tmp_path):
    assert ranking_change([3, 2, 1], [3, 2, 1])["kendall_tau_change"] == 0.0
    assert ranking_change([3, 2, 1], [3, 1, 2])["kendall_tau_change"] > 0.0
    assert ranking_change([3, 2, 1], [1, 2, 3])["kendall_tau_change"] == 2.0
    real, null = _rows()
    real_two, null_two = _rows(run=1)
    records = [*real, *null, *real_two, *null_two]
    bundle, _ = _bundle(tmp_path, records)
    result = estimate_from_bundle(bundle, bootstrap_draws=10)
    assert result["run_means"]["kendall_tau_change"] == [2.0, 2.0]
    assert result["provenance"]["probe_manifest_sha256"] == bundle.manifest_sha256
    assert result["provenance"]["pairing_seed"] == PAIRING_SEED


def test_estimator_consumes_only_sealed_bundle_records(tmp_path):
    real, null = _rows()
    real_two, null_two = _rows(run=1)
    bundle, _ = _bundle(tmp_path, [*real, *null, *real_two, *null_two])
    assert estimate_from_bundle(bundle, bootstrap_draws=10)["run_means"]["kendall_tau_change"] == [2.0, 2.0]
    forged_real, forged_null = _rows()
    forged_real[0]["patched_q"] = [3, 2, 1]
    with pytest.raises(TypeError):
        estimate_from_bundle(forged_real, forged_null, bundle)
    with pytest.raises(TypeError):
        estimate_from_bundle(object())
