"""Canonical serialization, content hashes, and atomic run records."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_json(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_data(data: Any) -> str:
    return hashlib.sha256(canonical_json(data)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(root: str | Path) -> dict[str, Any]:
    """Hash the human and executable scientific source of a repository."""
    base = Path(root).resolve()
    selected: list[Path] = []
    for pattern in ("src/**/*.py", "tests/**/*.py", "docs/**/*.md", "benchmarks/**/*.json", "README.md", "pyproject.toml"):
        selected.extend(path for path in base.glob(pattern) if path.is_file())
    files = {str(path.relative_to(base)): sha256_file(path) for path in sorted(set(selected))}
    if not files:
        raise ValueError(f"no scientific source files found under {base}")
    return {
        "schema": "jx-source-manifest/v2",
        "scope": "repository",
        "files": files,
        "tree_sha256": sha256_data(files),
    }


def package_source_manifest(package_dir: str | Path) -> dict[str, Any]:
    """Hash executable Python source from an installed package tree."""
    base = Path(package_dir).resolve()
    selected = sorted(path for path in base.rglob("*.py") if path.is_file())
    files = {
        f"src/jxplanetx/{path.relative_to(base).as_posix()}": sha256_file(path)
        for path in selected
    }
    if not files:
        raise ValueError(f"no executable Python source found under {base}")
    return {
        "schema": "jx-source-manifest/v2",
        "scope": "installed_package",
        "files": files,
        "tree_sha256": sha256_data(files),
    }


def runtime_source_manifest() -> dict[str, Any]:
    """Hash the active checkout, or the installed executable package.

    Editable installs resolve back to the repository and include its scientific
    source and metadata. Normal installs include the Python files that are
    actually executing. Both modes fail closed instead of emitting an empty
    provenance manifest.
    """
    package_dir = Path(__file__).resolve().parent
    repository_root = package_dir.parent.parent
    repository_package = repository_root / "src" / package_dir.name
    if (
        (repository_root / "pyproject.toml").is_file()
        and repository_package.is_dir()
        and repository_package.resolve() == package_dir
    ):
        return source_manifest(repository_root)
    return package_source_manifest(package_dir)


def environment_record() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }


def write_run_record(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "jx-planet-x-run/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment_record(),
        "payload": payload,
    }
    record["payload_sha256"] = sha256_data(payload)
    encoded = json.dumps(record, sort_keys=True, indent=2) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return record
