#!/usr/bin/env python3
"""Build an authenticated import closure from commit 289c543, not from HEAD."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "289c5434905257ddbdca8542a4ed41c9858e4403"
DEFAULT_DESTINATION = ROOT / "outputs/models/source_289c543_closure"
ROOT_FILES = ("scripts/p1_train.py", "configs/p1_t2.yaml", "pyproject.toml", "uv.lock")
ARTIFACT_ROOT = (ROOT / "outputs/models").resolve()
RESOURCE_CALLS = ("files", "open_text", "open_binary", "read_text", "read_binary")



def git_bytes(*args: str) -> bytes:
    return subprocess.check_output(("git", *args), cwd=ROOT)


def tree_blobs() -> dict[str, str]:
    result: dict[str, str] = {}
    for line in git_bytes("ls-tree", "-r", SOURCE_COMMIT, "--", "src/dgcc", *ROOT_FILES).decode().splitlines():
        metadata, path = line.split("\t", 1)
        _mode, kind, blob = metadata.split(" ", 2)
        if kind != "blob" or path in result:
            raise RuntimeError(f"invalid source-tree entry: {line}")
        result[path] = blob
    if "scripts/p1_train.py" not in result or "src/dgcc/__init__.py" not in result:
        raise RuntimeError("base commit does not contain the required training roots")
    return result


def module_index(blobs: dict[str, str]) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for path in blobs:
        if not path.startswith("src/") or not path.endswith(".py"):
            continue
        relative = path.removeprefix("src/").removesuffix(".py")
        module = relative.replace("/", ".")
        if module.endswith(".__init__"):
            module = module.removesuffix(".__init__")
        index.setdefault(module, set()).add(path)
    return index


def importing_module(path: str) -> str:
    if path.startswith("src/"):
        module = path.removeprefix("src/").removesuffix(".py").replace("/", ".")
        return module.removesuffix(".__init__")
    return ""


def _dotted_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else ""
    return ""


def imports_for(path: str, source: bytes) -> set[str]:
    tree = ast.parse(source, filename=path)
    current = importing_module(path)
    package = current if path.endswith("/__init__.py") else current.rpartition(".")[0]
    imports: set[str] = set()
    importlib_names: set[str] = set()
    resource_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"importlib.resources", "pkg_resources"}:
                    raise RuntimeError(f"unsupported resource import: {path}")
                if alias.name == "importlib":
                    importlib_names.add(alias.asname or alias.name)
                if alias.name == "dgcc" or alias.name.startswith("dgcc."):
                    imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module in {"importlib.resources", "pkg_resources"}:
                raise RuntimeError(f"unsupported resource import: {path}")
            if node.module == "importlib":
                for alias in node.names:
                    if alias.name == "import_module":
                        importlib_names.add(alias.asname or alias.name)
                    elif alias.name == "resources":
                        resource_names.add(alias.asname or alias.name)
            if node.level:
                base_parts = package.split(".") if package else []
                if node.level > len(base_parts) + 1:
                    raise RuntimeError(f"relative import escapes package: {path}")
                prefix = ".".join(base_parts[: len(base_parts) - node.level + 1])
                module = ".".join(part for part in (prefix, node.module or "") if part)
            else:
                module = node.module or ""
            if module == "dgcc" or module.startswith("dgcc."):
                imports.add(module)
                imports.update(f"{module}.{alias.name}" for alias in node.names if alias.name != "*")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted_name(node.func)
        if name == "__import__" or name in importlib_names or any(
            name == f"{importlib_name}.import_module" for importlib_name in importlib_names
        ):
            raise RuntimeError(f"unsupported dynamic import: {path}")
        resource_access = any(
            name == f"{resource_name}.{resource_call}"
            for resource_name in resource_names
            for resource_call in RESOURCE_CALLS
        ) or any(
            name == f"{importlib_name}.resources.{resource_call}"
            for importlib_name in importlib_names
            for resource_call in RESOURCE_CALLS
        )
        dormant_t2_split_lookup = (
            path == "src/dgcc/tasks/t2.py" and name == "resources.files"
        )
        if resource_access and not dormant_t2_split_lookup:
            raise RuntimeError(f"unsupported resource access: {path}")
    return imports


def closure_blobs() -> dict[str, str]:
    blobs = tree_blobs()
    index = module_index(blobs)
    closure = set(ROOT_FILES) | {"src/dgcc/__init__.py"}
    pending = ["scripts/p1_train.py", "src/dgcc/__init__.py"]
    while pending:
        path = pending.pop()
        if not path.endswith(".py"):
            continue
        source = git_bytes("show", f"{SOURCE_COMMIT}:{path}")
        if git_blob_id(source) != blobs[path]:
            raise RuntimeError(f"git show blob mismatch: {path}")
        for module in imports_for(path, source):
            candidates = index.get(module, set())
            # A package import must include its initializer; a named child can be absent
            # only when it is an external attribute rather than a Python module.
            if module == "dgcc" or module in index:
                if not candidates:
                    raise RuntimeError(f"imported module missing from git tree: {module}")
                for candidate in candidates:
                    if candidate not in closure:
                        closure.add(candidate)
                        pending.append(candidate)
        for parent in Path(path).parents:
            initializer = (parent / "__init__.py").as_posix()
            if initializer in blobs and initializer not in closure:
                closure.add(initializer)
                pending.append(initializer)
    result = {path: blobs[path] for path in sorted(closure)}
    if not result:
        raise RuntimeError("authenticated closure is empty")
    return result


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
def git_blob_id(data: bytes) -> str:
    return subprocess.check_output(
        ("git", "hash-object", "--stdin"), cwd=ROOT, input=data, text=False
    ).decode().strip()



def metadata(blobs: dict[str, str]) -> dict[str, object]:
    files = {
        path: {"git_blob": blob, "sha256": sha256(git_bytes("show", f"{SOURCE_COMMIT}:{path}"))}
        for path, blob in blobs.items()
    }
    return {"schema_version": 1, "source_commit": SOURCE_COMMIT, "files": files}


def verify(destination: Path, blobs: dict[str, str]) -> None:
    expected_metadata = metadata(blobs)
    expected_files = set(blobs) | {"MANIFEST.sha256", "closure_metadata.json"}
    actual_files = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise RuntimeError("closure file set mismatch")
    manifest: dict[str, str] = {}
    for line in (destination / "MANIFEST.sha256").read_text().splitlines():
        try:
            digest, path = line.split("  ", 1)
        except ValueError as error:
            raise RuntimeError("malformed closure manifest") from error
        if path in manifest:
            raise RuntimeError("malformed closure manifest")
        manifest[path] = digest
    expected_manifest = {path: item["sha256"] for path, item in expected_metadata["files"].items()}
    if manifest != expected_manifest:
        raise RuntimeError("closure manifest mismatch")
    for path, digest in expected_manifest.items():
        data = (destination / path).read_bytes()
        if sha256(data) != digest or git_blob_id(data) != blobs[path]:
            raise RuntimeError(f"closure content mismatch: {path}")
    if json.loads((destination / "closure_metadata.json").read_text()) != expected_metadata:
        raise RuntimeError("closure metadata mismatch")


def _safe_destination(destination: Path) -> Path:
    destination = destination.absolute()
    _reject_symlinks(destination)
    destination = destination.resolve()
    temp_root = Path(tempfile.gettempdir()).resolve()
    if destination == DEFAULT_DESTINATION.resolve() or ARTIFACT_ROOT in destination.parents or temp_root in destination.parents:
        return destination
    raise RuntimeError("destination must be under the artifact root or system temporary root")


def _reject_symlinks(path: Path) -> None:
    if path.is_symlink() or any(parent.is_symlink() for parent in (path, *path.parents)):
        raise RuntimeError(f"symlink path is not permitted: {path}")


def build(destination: Path = DEFAULT_DESTINATION) -> Path:
    destination = _safe_destination(destination)
    _reject_symlinks(destination.parent)
    blobs = closure_blobs()
    if destination.exists():
        _reject_symlinks(destination)
        verify(destination, blobs)
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
        proof = metadata(blobs)
        manifest = {path: item["sha256"] for path, item in proof["files"].items()}
        (staging / "MANIFEST.sha256").write_text("".join(f"{digest}  {path}\n" for path, digest in manifest.items()))
        (staging / "closure_metadata.json").write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
        verify(staging, blobs)
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args(argv)
    try:
        build(args.destination.resolve())
    except (OSError, RuntimeError, subprocess.CalledProcessError, SyntaxError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
