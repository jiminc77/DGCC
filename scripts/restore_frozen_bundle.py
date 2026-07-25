#!/usr/bin/env python3
"""Restore and authenticate the historical 786d651 frozen training bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "786d651a4b0f6013971bf1d8f23b125062223679"
BUNDLE_PATH = ROOT / "outputs/models/frozen_m4_bundle"
SAFE_SOURCE_BLOBS = {
    "configs/p1_t2.yaml": "46f03acb480d64230dc1115d921bc4533a921562",
    "pyproject.toml": "1e847758d51ae3e228ad4add7ec1485db49f5638",
    "scripts/p1_train.py": "c325743c14c6b3cf7f5fff4b9f1341ec459bb9e3",
    "src/dgcc/__init__.py": "e83f02a4e454c7538ae14e79621fc536fcc9126b",
    "src/dgcc/envs/__init__.py": "a6456caebe40c70fb68594db13cec1c20fbb548c",
    "src/dgcc/envs/base.py": "ada640afa5a1ce5f1cb86df8406d195153b4e92f",
    "src/dgcc/envs/dlolab.py": "c21f63843b69bad2441c367b1e75c4b8dd889f1b",
    "src/dgcc/envs/mujoco_cable.py": "5e94fb05aea736a52261236c7acb815f89e78bf8",
    "src/dgcc/envs/params.py": "02aff8a015bc1c9990eebce76380269ebed1bfab",
    "src/dgcc/goals/__init__.py": "81c151aa4716e7f6936fe9601cf978757376cc74",
    "src/dgcc/goals/distance.py": "fa2c8226d4512c6f7f8e31164d50550b48f70fa1",
    "src/dgcc/goals/dual_goal.py": "80beb076e5b731c7ad86f6723a9a4936274e9953",
    "src/dgcc/logging/__init__.py": "c4251d32b8a2949a9f252baeafc07cf61ec38b82",
    "src/dgcc/logging/schema.py": "0ae1b0daa1213b7188bab661779a701c0d1e1f05",
    "src/dgcc/logging/writer.py": "cc9d5d6e188960ab78cf195115e6984b106f8022",
    "src/dgcc/models/__init__.py": "af2ec037273391eadd349292080f46104ac78994",
    "src/dgcc/models/networks.py": "22373e67fbdd29b571e1db0e370fdbcbd8d3c25b",
    "src/dgcc/phi/__init__.py": "dc845c54a116696882fc7c97a1ccc8b77daf33a1",
    "src/dgcc/phi/dct.py": "f05bb0be898bed368b7e1dbff673af4cf8b21b99",
    "src/dgcc/phi/normalize.py": "886a104f4d2f218c5d478112f160be822ce6f947",
    "src/dgcc/phi/resample.py": "afaa8f0c8ab9a45ba2dec417320652ff81539a5e",
    "src/dgcc/rl/__init__.py": "48b1849b535f315f67110a962056663534f09486",
    "src/dgcc/rl/diagnostics.py": "71b1a4ffb81f2bf1dba7366e6b6eda0773aa9b09",
    "src/dgcc/rl/evaluation.py": "4f6da7f79dcf3f9578fc6ac5d4f2b151a6683116",
    "src/dgcc/rl/replay.py": "8af5d6acc6a34fafb29a690a7f79b8aa2fa191b0",
    "src/dgcc/rl/td3.py": "b3303e0ad325030c931faec565d309a7361c4678",
    "src/dgcc/tasks/__init__.py": "127c7c462277fa4bf19dff66d05d63d379503f25",
    "src/dgcc/tasks/domain.py": "1812f4192d283dd73d1367cf431437954d1e992c",
    "src/dgcc/tasks/episode.py": "4037accc97d3397b9e0c6d567789c5a48f4e2637",
    "src/dgcc/tasks/reward.py": "9580c579d67310ba069388fb09aaff38e3006871",
    "src/dgcc/tasks/splits/t2_v1.json": "5081e3343276cbab9a89c3cbeab96762ca795f25",
    "src/dgcc/tasks/t1.py": "422317c7dd1d47d2061a5eeab33b8cafe78f1cf8",
    "src/dgcc/tasks/t2.py": "ed365971b1451f2b78f7c2e367ee82c6e8968b2e",
    "src/dgcc/utils/__init__.py": "36fb924a2358e6dca03e307b24ab333150f97db6",
    "src/dgcc/utils/meta.py": "1f0e32c8ea7dc7b659a6d180898ab67dd6b49030",
    "src/dgcc/utils/seeding.py": "f9e4c454e60596ed2f07c7924c846f88e784f732",
    "uv.lock": "fc90ffafd74c18faf444b02ce46c83b0b3d3388d",
}


def git_bytes(*args: str) -> bytes:
    return subprocess.check_output(("git", *args), cwd=ROOT)


def source_blobs() -> dict[str, str]:
    """Authenticate the compiled-in, exact release closure."""
    authenticated: dict[str, str] = {}
    for path, expected_blob in SAFE_SOURCE_BLOBS.items():
        entry = git_bytes("ls-tree", SOURCE_COMMIT, "--", path).decode().strip()
        if not entry:
            raise RuntimeError(f"safe source path is absent from source commit: {path}")
        metadata_entry, actual_path = entry.split("\t", 1)
        _mode, kind, actual_blob = metadata_entry.split(" ", 2)
        if kind != "blob" or actual_path != path or actual_blob != expected_blob:
            raise RuntimeError(f"safe source blob mismatch: {path}")
        authenticated[path] = expected_blob
    return authenticated


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def expected_metadata(blobs: dict[str, str]) -> dict[str, object]:
    return {"schema_version": 1, "source_commit": SOURCE_COMMIT, "source_blobs": blobs}


def verify_bundle(bundle: Path, blobs: dict[str, str]) -> None:
    expected = {path: digest(git_bytes("show", f"{SOURCE_COMMIT}:{path}")) for path in blobs}
    actual_files = {path.relative_to(bundle).as_posix() for path in bundle.rglob("*") if path.is_file()}
    required = set(expected) | {"MANIFEST.sha256", "bundle_metadata.json"}
    if actual_files != required:
        raise RuntimeError(f"bundle file set mismatch: missing={sorted(required - actual_files)}, extra={sorted(actual_files - required)}")
    manifest: dict[str, str] = {}
    for line in (bundle / "MANIFEST.sha256").read_text().splitlines():
        try:
            file_digest, path = line.split("  ", 1)
        except ValueError as error:
            raise RuntimeError("malformed bundle manifest") from error
        if path in manifest or len(file_digest) != 64:
            raise RuntimeError("malformed bundle manifest")
        manifest[path] = file_digest
    if manifest != expected:
        raise RuntimeError("bundle manifest does not match committed source")
    for path, file_digest in expected.items():
        if digest((bundle / path).read_bytes()) != file_digest:
            raise RuntimeError(f"bundle content digest mismatch: {path}")
    try:
        metadata = json.loads((bundle / "bundle_metadata.json").read_text())
    except json.JSONDecodeError as error:
        raise RuntimeError("malformed bundle metadata") from error
    if (
        metadata.get("schema_version") != 1
        or metadata.get("source_commit") != SOURCE_COMMIT
        or metadata.get("source_blobs") != blobs
    ):
        raise RuntimeError("bundle metadata does not match committed source")


def _safe_destination(destination: Path) -> Path:
    destination = destination.absolute()
    _reject_symlinks(destination)
    destination = destination.resolve()
    artifact_root = (ROOT / "outputs").resolve()
    temp_root = Path(tempfile.gettempdir()).resolve()
    if destination == BUNDLE_PATH.resolve() or artifact_root in destination.parents or temp_root in destination.parents:
        return destination
    raise RuntimeError("destination must be under outputs or the system temporary root")


def _reject_symlinks(path: Path) -> None:
    if path.is_symlink() or any(parent.is_symlink() for parent in (path, *path.parents)):
        raise RuntimeError(f"symlink path is not permitted: {path}")


def restore(destination: Path = BUNDLE_PATH) -> Path:
    destination = _safe_destination(destination)
    _reject_symlinks(destination.parent)
    blobs = source_blobs()
    if destination.exists():
        _reject_symlinks(destination)
        verify_bundle(destination, blobs)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlinks(destination.parent)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for path in blobs:
            target = staging / path
            target.parent.mkdir(parents=True, exist_ok=True)
            _reject_symlinks(target.parent)
            target.write_bytes(git_bytes("show", f"{SOURCE_COMMIT}:{path}"))
        manifest = {path: digest((staging / path).read_bytes()) for path in blobs}
        (staging / "MANIFEST.sha256").write_text("".join(f"{value}  {path}\n" for path, value in manifest.items()))
        (staging / "bundle_metadata.json").write_text(json.dumps(expected_metadata(blobs), indent=2, sort_keys=True) + "\n")
        verify_bundle(staging, blobs)
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=BUNDLE_PATH)
    args = parser.parse_args(argv)
    try:
        restore(args.destination.resolve())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
