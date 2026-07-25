import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "v2_beta_calibrator.py"
spec = importlib.util.spec_from_file_location("v2_beta_calibrator", MODULE_PATH)
calibrator = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(calibrator)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_registered_inputs(tmp_path, *, extra_npz=False, bad_dtype=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    # This small fixed synthetic panel has a root strictly inside the registered bracket.
    values = np.array([0.0] + [-0.020] * 31, dtype=np.float64)
    q1 = np.broadcast_to(values, (10, 300, 32)).copy()
    qmin = np.broadcast_to(values * 1.1, (10, 300, 32)).copy()
    if bad_dtype:
        q1 = q1.astype(np.float32)
    score = tmp_path / "scores.npz"
    payload = {"q1_scores": q1, "qmin_scores": qmin}
    if extra_npz:
        payload["extra"] = qmin
    np.savez(score, **payload)
    checkpoints = [hashlib.sha256(f"checkpoint-{i}".encode()).hexdigest() for i in range(10)]
    input_manifest = tmp_path / "input.json"
    input_manifest.write_text(json.dumps({"checkpoint_hashes": checkpoints, "fixed_panel_hash": "a" * 64}))
    code_manifest = tmp_path / "code.json"
    code_manifest.write_text(json.dumps({"calibrator_sha256": digest(MODULE_PATH)}))
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "status": "REGISTERED_BEFORE_EXECUTION",
        "rule": "G=min(M_Q1,M_Qmin); require G(0.025)>=12; 64 bisections on [0.010000,0.025000]; ceil upper to 6 decimals",
        "checkpoint_hashes": checkpoints, "fixed_panel_hash": "a" * 64,
        "input_manifest_sha256": digest(input_manifest), "score_sha256": digest(score),
        "code_manifest_sha256": digest(code_manifest), "calibrator_sha256": digest(MODULE_PATH),
    }))
    return input_manifest, score, protocol, code_manifest


def run(tmp_path):
    inputs = make_registered_inputs(tmp_path)
    result, trace = tmp_path / "result.json", tmp_path / "trace.npz"
    assert calibrator.main(["--input-manifest", str(inputs[0]), "--score-npz", str(inputs[1]), "--protocol", str(inputs[2]), "--code-manifest", str(inputs[3]), "--result", str(result), "--trace", str(trace)]) == 0
    return result, trace, inputs[1]


def test_known_deterministic_root_and_full_trace(tmp_path):
    result, trace, score = run(tmp_path)
    payload = json.loads(result.read_text())
    with np.load(trace, allow_pickle=False) as saved:
        assert set(saved.files) == {"lower", "upper", "mid", "M_Q1", "M_Qmin", "G"}
        assert all(saved[key].shape == (64,) and saved[key].dtype == np.float64 for key in saved.files)
        # The result is the upward six-decimal representation of the final feasible endpoint.
        final_upper = saved["mid"][-1] if saved["G"][-1] >= 12 else saved["upper"][-1]
        expected = np.ceil(final_upper * 1_000_000) / 1_000_000
        assert payload["beta_contact"] == expected
    assert calibrator.verify_result(result, trace, score) == payload


def test_upper_bracket_failure_prohibits_expansion(tmp_path):
    inputs = make_registered_inputs(tmp_path)
    scores = np.zeros((10, 300, 32), dtype=np.float64)
    scores[..., 0] = 1.0
    np.savez(inputs[1], q1_scores=scores, qmin_scores=scores)
    # Refresh the only deliberately altered pin.
    protocol = json.loads(inputs[2].read_text()); protocol["score_sha256"] = digest(inputs[1]); inputs[2].write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="expansion is prohibited"):
        calibrator.main(["--input-manifest", str(inputs[0]), "--score-npz", str(inputs[1]), "--protocol", str(inputs[2]), "--code-manifest", str(inputs[3]), "--result", str(tmp_path / "r"), "--trace", str(tmp_path / "t")])


@pytest.mark.parametrize("field", ["RETURN", "Reward", "SUCCESS", "final_distance", "heldout", "held-out", "checkpoint_selection_score", "candidate_outcome", "arm_ranking"])
def test_forbidden_fields_rejected_at_any_nesting(field):
    with pytest.raises(ValueError, match="forbidden field"):
        calibrator.reject_forbidden_fields({"outer": [{field: 1}]})


def test_hash_tensor_and_schema_mismatches(tmp_path):
    inputs = make_registered_inputs(tmp_path / "hash")
    protocol = json.loads(inputs[2].read_text())
    protocol["score_sha256"] = "0" * 64
    inputs[2].write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="score hash"):
        calibrator.main(["--input-manifest", str(inputs[0]), "--score-npz", str(inputs[1]), "--protocol", str(inputs[2]), "--code-manifest", str(inputs[3]), "--result", str(tmp_path / "hash-r"), "--trace", str(tmp_path / "hash-t")])
    inputs = make_registered_inputs(tmp_path / "extra", extra_npz=True)
    with pytest.raises(ValueError, match="only q1_scores"):
        calibrator.main(["--input-manifest", str(inputs[0]), "--score-npz", str(inputs[1]), "--protocol", str(inputs[2]), "--code-manifest", str(inputs[3]), "--result", str(tmp_path / "extra-r"), "--trace", str(tmp_path / "extra-t")])
    inputs = make_registered_inputs(tmp_path / "schema", bad_dtype=True)
    with pytest.raises(ValueError, match="q1_scores"):
        calibrator.main(["--input-manifest", str(inputs[0]), "--score-npz", str(inputs[1]), "--protocol", str(inputs[2]), "--code-manifest", str(inputs[3]), "--result", str(tmp_path / "schema-r"), "--trace", str(tmp_path / "schema-t")])


def test_exclusive_outputs_rng_free_production_recheck_and_verifier(tmp_path, monkeypatch):
    result, trace, score = run(tmp_path)
    with pytest.raises(FileExistsError):
        run(tmp_path)
    # The module never imports or constructs an RNG; guard the common NumPy entry point too.
    monkeypatch.setattr(np.random, "default_rng", lambda *a, **k: (_ for _ in ()).throw(AssertionError("RNG used")))
    inputs = make_registered_inputs(tmp_path / "rng")
    calibrator.main(["--input-manifest", str(inputs[0]), "--score-npz", str(inputs[1]), "--protocol", str(inputs[2]), "--code-manifest", str(inputs[3]), "--result", str(tmp_path / "rng-r"), "--trace", str(tmp_path / "rng-t")])
    tampered = json.loads(result.read_text()); tampered["trace_sha256"] = "0" * 64; result.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="trace hash"):
        calibrator.verify_result(result, trace, score)

    def production_only_uniform(scores, *, beta_contact):
        return torch.full_like(scores, 1.0 / scores.shape[-1])

    monkeypatch.setattr(
        calibrator, "production_contact_softmax_weights", production_only_uniform
    )
    inputs = make_registered_inputs(tmp_path / "production")
    with pytest.raises(ValueError, match="outside \\[12,20\\]"):
        calibrator.main(["--input-manifest", str(inputs[0]), "--score-npz", str(inputs[1]), "--protocol", str(inputs[2]), "--code-manifest", str(inputs[3]), "--result", str(tmp_path / "production-r"), "--trace", str(tmp_path / "production-t")])
