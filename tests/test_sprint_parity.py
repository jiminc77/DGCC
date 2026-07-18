import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

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
        "src/dgcc/logging",
        "src/dgcc/__init__.py",
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


def test_injected_mismatch_fails_without_writing_proof(tmp_path):
    proof, _ = parity.build_proof(injected_mismatch=True)
    assert proof["verdict"] == "FAIL"
    assert proof["mismatches"]
    assert not (tmp_path / "proof.json").exists()


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


def frozen_bundle_copy(monkeypatch, tmp_path):
    bundle = tmp_path / "bundle"
    shutil.copytree(ROOT / "outputs/models/frozen_m4_bundle", bundle)
    monkeypatch.setattr(parity, "BUNDLE_PATH", bundle)
    return bundle


def test_bundle_imports_work_with_only_frozen_src(tmp_path):
    bundle_src = ROOT / "outputs/models/frozen_m4_bundle/src"
    env = {
        "PYTHONPATH": str(bundle_src),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": os.environ["PATH"],
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import dgcc.rl.td3, dgcc.rl.replay, dgcc.models.networks, dgcc.logging.writer",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("path", "content"),
    [
        ("unexpected.txt", "extra"),
        ("bundle_metadata.json", '{"source_commit": "tampered"}\n'),
        ("MANIFEST.sha256", "tampered\n"),
    ],
)
def test_bundle_tampering_fails_closed(monkeypatch, tmp_path, path, content):
    bundle = frozen_bundle_copy(monkeypatch, tmp_path)
    target = bundle / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    _, maps = parity.build_proof()
    with pytest.raises(RuntimeError):
        parity.freeze_bundle(maps[parity.BUNDLE_SOURCE_COMMIT])


def test_missing_transitive_closure_fails_nonzero(monkeypatch, tmp_path):
    monkeypatch.setattr(
        parity,
        "CLOSURE_PATHS",
        tuple(path for path in parity.CLOSURE_PATHS if path != "src/dgcc/logging"),
    )
    monkeypatch.setattr(parity, "PROOF_PATH", tmp_path / "proof.json")
    assert parity.main([]) != 0


def test_cli_has_no_injection_flag():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True
    )
    assert "--inject-mismatch" not in result.stdout
