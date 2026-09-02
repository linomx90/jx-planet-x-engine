from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .errors import ValidationError


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_entry(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    label = str(resolved.relative_to(relative_to.resolve())) if relative_to else path.name
    return {
        "path": label,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def refuse_symlink(path: Path, *, allow_missing: bool = False) -> None:
    # Inspect every existing component without resolving through symlinks.
    if not path.exists() and not allow_missing:
        raise ValidationError(f"required path does not exist: {path}")
    current = path if path.exists() else path.parent
    while True:
        if current.is_symlink():
            raise ValidationError(f"symlink path component is forbidden: {current}")
        if current.parent == current:
            break
        current = current.parent


def ensure_new_directory(path: Path) -> None:
    refuse_symlink(path, allow_missing=True)
    if path.exists():
        if not path.is_dir():
            raise ValidationError(f"output path is not a directory: {path}")
        if any(path.iterdir()):
            raise ValidationError(f"output directory must be absent or empty: {path}")
    else:
        path.mkdir(parents=True, mode=0o700)
    os.chmod(path, 0o700)


def atomic_write_bytes(path: Path, data: bytes, mode: int = 0o600) -> None:
    refuse_symlink(path.parent, allow_missing=True)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    refuse_symlink(path, allow_missing=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(path: Path, text: str, mode: int = 0o600) -> None:
    atomic_write_bytes(path, text.encode("utf-8"), mode)


def load_json(path: Path) -> Any:
    refuse_symlink(path)
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any, mode: int = 0o600) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", mode)


def iter_regular_files(root: Path) -> Iterable[Path]:
    refuse_symlink(root)
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValidationError(f"symlink inside protected tree: {path}")
        if path.is_file():
            file_mode = path.stat().st_mode
            if not stat.S_ISREG(file_mode):
                raise ValidationError(f"non-regular file inside protected tree: {path}")
            yield path


def chmod_tree_read_only(root: Path) -> None:
    for path in iter_regular_files(root):
        os.chmod(path, 0o400)
    for directory in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        os.chmod(directory, 0o500)
    os.chmod(root, 0o500)
