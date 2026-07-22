import hashlib
import json
import numpy as np
import pytest
import torch
import h5py
from dgcc.analysis.sprint_patching import (a0_interchange, a1_projected_splice, graded_ablation, mean_ablation, rescale_mask, deterministic_pairing, estimate_real_minus_null, estimate_by_run, load_probe_manifest, verify_projector_sha256)


def test_operators_and_graded():
    torch.manual_seed(1); p = torch.linalg.qr(torch.randn(256, 24)).Q.T
    r, d = torch.randn(3, 256), torch.randn(3, 256)
    out = a1_projected_splice(r, d, p)
    assert torch.allclose(out @ p.T, d @ p.T, atol=1e-5)
    assert torch.allclose((out - r) @ (torch.eye(256) - p.T @ p), torch.zeros_like(r), atol=1e-5)
    assert torch.equal(a0_interchange(r, d), d)
    mean = torch.zeros(256)
    assert torch.equal(graded_ablation(r, mean, 1.0), mean_ablation(r, mean))


def test_rescale_mask():
    mask = rescale_mask(2.5)
    assert torch.equal(mask, torch.tensor([1, *([2.5] * 7), 1, *([2.5] * 7), *([1] * 8)]))


def test_pairing_deterministic():
    rec = [{"goal_id": "g", "p": 2, "nuisance": "n", "delta_m": np.zeros(24)}]
    donors = [{"id": "far", "goal_id": "g", "p": 2, "nuisance": "n", "delta_m": np.ones(24)}, {"id": "near", "goal_id": "g", "p": 2, "nuisance": "n", "delta_m": np.zeros(24)}]
    assert deterministic_pairing(rec, donors) == deterministic_pairing(rec, donors)
    assert deterministic_pairing(rec, donors)[0][1]["id"] == "near"


def test_manifest_fail_closed(tmp_path):
    probe = tmp_path / "probe.h5"
    with h5py.File(probe, "w") as f: f["h_p"] = np.zeros((2, 256))
    digest = hashlib.sha256(probe.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "files": {digest: {"path": str(probe), "sha256": digest, "size": probe.stat().st_size}}}))
    assert len(load_probe_manifest(manifest)) == 1
    manifest.write_text(json.dumps({"schema_version": 1, "files": {"0" * 64: {"path": str(probe), "sha256": "0" * 64, "size": probe.stat().st_size}}}))
    with pytest.raises(ValueError): load_probe_manifest(manifest)


def test_estimator_and_projector_rejection():
    real = [{"baseline_q": [3, 2, 1], "patched_q": [1, 2, 3]}]
    null = [{"baseline_q": [3, 2, 1], "patched_q": [3, 2, 1]}]
    result = estimate_real_minus_null(real, null)
    assert result["kendall_tau_change"] == -2.0 and result["top1_flip"] == 1.0
    with pytest.raises(ValueError): verify_projector_sha256(torch.zeros(24, 256))
    rows = [{"run": i, **real[0]} for i in range(2)]
    null_rows = [{"run": i, **null[0]} for i in range(2)]
    assert set(estimate_by_run(rows, null_rows, bootstrap_draws=10)["bca"]) == {"kendall_tau_change", "top1_flip"}
