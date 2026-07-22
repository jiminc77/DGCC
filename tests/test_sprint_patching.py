import hashlib
import json

import h5py
import numpy as np
import pytest
import torch

from dgcc.analysis.sprint_patching import (
    N_PAIRS_PER_RUN,
    a0_interchange,
    a1_projected_splice,
    deterministic_pairing,
    estimate_by_run,
    estimate_real_minus_null,
    graded_ablation,
    load_probe_manifest,
    load_projector,
    mean_ablation,
    ranking_change,
    rescale_mask,
)


def _bundle(tmp_path):
    probe = tmp_path / "probe.h5"
    with h5py.File(probe, "w") as handle:
        handle["h_p"] = np.zeros((2, 256))
    digest = hashlib.sha256(probe.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "files": {digest: {"path": str(probe), "sha256": digest, "size": probe.stat().st_size}}}))
    return load_probe_manifest(manifest), manifest


def _projector(tmp_path):
    torch.manual_seed(1)
    value = torch.linalg.qr(torch.randn(256, 24)).Q.T
    path = tmp_path / "projector.pt"
    torch.save(value, path)
    return load_projector(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _rows(run=0):
    real, null = [], []
    for index in range(N_PAIRS_PER_RUN):
        identity = {"pair_id": f"p-{index}", "goal_id": f"g-{index}", "run": run, "operator": "a0"}
        real.append({**identity, "baseline_q": [3, 2, 1], "patched_q": [1, 2, 3]})
        null.append({**identity, "baseline_q": [3, 2, 1], "patched_q": [3, 2, 1]})
    return real, null


def test_operators_and_graded(tmp_path):
    projector = _projector(tmp_path)
    r, d = torch.randn(3, 256), torch.randn(3, 256)
    out = a1_projected_splice(r, d, projector)
    assert torch.allclose(out @ projector.value.T, d @ projector.value.T, atol=1e-5)
    assert torch.equal(a0_interchange(r, d), d)
    assert torch.equal(graded_ablation(r, torch.zeros(256), 1.0), mean_ablation(r, torch.zeros(256)))
    with pytest.raises(TypeError):
        a1_projected_splice(r, d, projector.value)


def test_projector_rejects_unpinned_or_nonorthonormal(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save(torch.zeros(24, 256), path)
    with pytest.raises(ValueError):
        load_projector(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError):
        load_projector(path, expected_sha256="0" * 64)


def test_rescale_mask():
    assert torch.equal(rescale_mask(2.5), torch.tensor([1, *([2.5] * 7), 1, *([2.5] * 7), *([1] * 8)]))


def test_confirmatory_pairing_requires_100_unique_goals():
    recipients = [{"goal_id": f"g-{i}", "p": 2, "nuisance": "n", "delta_m": np.zeros(24)} for i in range(100)]
    donors = [{"id": f"d-{i}", "goal_id": f"g-{i}", "p": 2, "nuisance": "n", "delta_m": np.zeros(24)} for i in range(100)]
    assert len(deterministic_pairing(recipients, donors)) == 100
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
    bundle, _ = _bundle(tmp_path)
    real, null = _rows()
    real_two, null_two = _rows(run=1)
    real.extend(real_two)
    null.extend(null_two)
    assert estimate_real_minus_null(real, null, bundle)["kendall_tau_change"] == 2.0
    result = estimate_by_run(real, null, bundle, bootstrap_draws=10)
    assert result["run_means"]["kendall_tau_change"] == [2.0, 2.0]
    assert result["provenance"]["probe_manifest_sha256"] == bundle.manifest_sha256


def test_estimator_rejects_unverified_or_mismatched_pairing(tmp_path):
    bundle, _ = _bundle(tmp_path)
    real, null = _rows()
    with pytest.raises(TypeError): estimate_by_run(real, null, object())
    null[0]["goal_id"] = "wrong"
    with pytest.raises(ValueError): estimate_by_run(real, null, bundle)
    real, null = _rows()
    real[1]["pair_id"] = real[0]["pair_id"]
    with pytest.raises(ValueError): estimate_by_run(real, null, bundle)
