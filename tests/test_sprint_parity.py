import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/sprint_parity_proof.py"
spec = importlib.util.spec_from_file_location("sprint_parity_proof", SCRIPT)
assert spec and spec.loader
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)


def test_closure_list_is_frozen_regression_guard():
    assert parity.CLOSURE_PATHS == (
        "src/dgcc/models",
        "src/dgcc/rl",
        "src/dgcc/envs",
        "src/dgcc/tasks",
        "src/dgcc/goals",
        "src/dgcc/phi",
        "src/dgcc/utils",
        "scripts/p1_train.py",
        "configs/p1_t2.yaml",
        "uv.lock",
        "pyproject.toml",
    )


def test_only_heldout_split_is_an_eval_only_exception():
    assert parity.EVAL_ONLY_EXCEPTION == "src/dgcc/tasks/splits/t2_sprint_heldout_v1.json"
    proof, maps = parity.build_proof()
    assert proof["eval_only_exception"]["static_training_path_scan"]["all_clear"]
    assert all(parity.EVAL_ONLY_EXCEPTION not in entries for entries in maps.values())


def test_injected_mismatch_fails_with_nonzero_exit_status(monkeypatch, tmp_path):
    proof, _ = parity.build_proof(injected_mismatch=True)
    assert proof["verdict"] == "FAIL"
    assert proof["mismatches"]
    monkeypatch.setattr(parity, "PROOF_PATH", tmp_path / "proof.json")
    assert parity.main(["--inject-mismatch"]) != 0


def test_bundle_manifest_sha256_matches_all_frozen_files():
    proof_path = ROOT / "outputs/metrics/sprint_bb_parity_proof.json"
    bundle = ROOT / "outputs/models/frozen_m4_bundle"
    proof = json.loads(proof_path.read_text())
    assert proof["verdict"] == "PASS"
    manifest = {}
    for line in (bundle / "MANIFEST.sha256").read_text().splitlines():
        digest, path = line.split("  ", 1)
        manifest[path] = digest
    assert manifest == {
        path: hashlib.sha256((bundle / path).read_bytes()).hexdigest()
        for path in manifest
    }
    assert set(manifest) == set(proof["closure_blobs"][parity.BUNDLE_SOURCE_COMMIT])
