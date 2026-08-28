#!/usr/bin/env python3
"""One-shot registered XP2-v4 save/decode boundary exercise.

This is engineering evidence only.  It emits no tracer metrics, gates, labels,
or scientific result and cannot authorize official A, B, or DOP853 execution.
"""

from __future__ import annotations

import argparse
import decimal
import fcntl
import hashlib
import json
import math
import os
import resource
import signal
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any, Sequence


EXPERIMENT_ID = "jx-xp2-public-synthetic-robustness-v4"
CONTRACT_SCHEMA = "jx-xp2-robustness-contract/v3"
REGISTRATION_SCHEMA = "jx-xp2-v4-engineering-registration/v1"
ATTEMPT_SCHEMA = "jx-xp2-v4-engineering-attempt/v1"
RESULT_SCHEMA = "jx-xp2-v4-engineering-boundary-result/v1"
ARM_IDS = ("M0", "CI01-P0", "AUDIT-CI01-P0")
NANOSECONDS_PER_SECOND = 1_000_000_000


class EngineeringError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def serialized(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_object(payload: bytes) -> dict[str, Any]:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise EngineeringError("duplicate JSON key")
            value[key] = item
        return value

    def finite_float(text: str) -> float:
        value = float(text)
        exact = decimal.Decimal(text)
        if (not math.isfinite(value) or not exact.is_finite()
                or (value == 0.0 and exact != 0)):
            raise EngineeringError("invalid JSON number")
        return value

    def reject_constant(text: str) -> None:
        raise EngineeringError(f"invalid JSON constant: {text}")

    value = json.loads(
        payload.decode("utf-8"), object_pairs_hook=unique_pairs,
        parse_float=finite_float, parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise EngineeringError("JSON input is not an object")
    return value


def read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise EngineeringError("unsafe JSON input")
    return parse_object(path.read_bytes())


def lexical_path(path: Path, label: str, *, leaf_kind: str) -> Path:
    """Reject symlinks in every lexical component before canonical use."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for index, component in enumerate(absolute.parts[1:], start=1):
        current /= component
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise EngineeringError(f"{label} component is missing") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise EngineeringError(f"{label} contains a symlink component")
        final = index == len(absolute.parts) - 1
        if not final and not stat.S_ISDIR(metadata.st_mode):
            raise EngineeringError(f"{label} ancestor is not a directory")
        if final:
            if leaf_kind == "file" and (
                not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
            ):
                raise EngineeringError(f"{label} is not a safe regular file")
            if leaf_kind == "dir" and not stat.S_ISDIR(metadata.st_mode):
                raise EngineeringError(f"{label} is not a safe directory")
    return absolute


class HeldPackageSnapshot:
    """Retain every prefinal package inode and rehash it before authority."""

    def __init__(self, package_root: Path) -> None:
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        absolute = Path(os.path.abspath(os.fspath(package_root)))
        self.directory_fds = [os.open(absolute.anchor, directory_flags)]
        self.directory_bindings: list[tuple[int, str, int, os.stat_result]] = []
        self.files: dict[str, tuple[int, os.stat_result, str, bytes]] = {}
        try:
            for component in absolute.parts[1:]:
                parent = self.directory_fds[-1]
                before = os.stat(component, dir_fd=parent, follow_symlinks=False)
                child = os.open(component, directory_flags, dir_fd=parent)
                opened = os.fstat(child)
                if (not stat.S_ISDIR(before.st_mode) or not stat.S_ISDIR(opened.st_mode)
                        or before.st_dev != opened.st_dev
                        or before.st_ino != opened.st_ino):
                    os.close(child)
                    raise EngineeringError("engineering package component changed")
                self.directory_fds.append(child)
                self.directory_bindings.append((parent, component, child, opened))
            self.root_fd = self.directory_fds[-1]
            self.names = sorted(os.listdir(self.root_fd))
            for name in self.names:
                before = os.stat(name, dir_fd=self.root_fd, follow_symlinks=False)
                flags = os.O_RDONLY | (
                    os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0
                )
                descriptor = os.open(name, flags, dir_fd=self.root_fd)
                opened = os.fstat(descriptor)
                if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1
                        or opened.st_dev != before.st_dev or opened.st_ino != before.st_ino):
                    os.close(descriptor)
                    raise EngineeringError("prefinal package contains unsafe entry")
                chunks: list[bytes] = []
                while True:
                    block = os.read(descriptor, 1024 * 1024)
                    if not block:
                        break
                    chunks.append(block)
                payload = b"".join(chunks)
                self.files[name] = (
                    descriptor, opened, hashlib.sha256(payload).hexdigest(), payload,
                )
        except BaseException:
            self.close()
            raise

    def digest(self, name: str) -> str:
        return self.files[name][2]

    def payload(self, name: str) -> bytes:
        return self.files[name][3]

    def revalidate(self, expected_names: set[str]) -> None:
        if set(self.names) != expected_names \
                or sorted(os.listdir(self.root_fd)) != self.names:
            raise EngineeringError("prefinal package inventory changed")
        stable = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
                  "st_mtime_ns", "st_ctime_ns")
        for index, (parent, name, descriptor, before) in enumerate(self.directory_bindings):
            after = os.fstat(descriptor)
            on_disk = os.stat(name, dir_fd=parent, follow_symlinks=False)
            keys = stable if index == len(self.directory_bindings) - 1 \
                else ("st_dev", "st_ino", "st_mode")
            if any(getattr(before, key) != getattr(after, key) for key in keys) \
                    or any(getattr(after, key) != getattr(on_disk, key) for key in keys):
                raise EngineeringError("prefinal package directory binding changed")
        for name, (descriptor, before, expected_digest, _payload) in self.files.items():
            after = os.fstat(descriptor)
            on_disk = os.stat(name, dir_fd=self.root_fd, follow_symlinks=False)
            if any(getattr(before, key) != getattr(after, key) for key in stable) \
                    or any(getattr(after, key) != getattr(on_disk, key) for key in stable):
                raise EngineeringError("prefinal package file binding changed")
            os.lseek(descriptor, 0, os.SEEK_SET)
            digest = hashlib.sha256()
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
            if digest.hexdigest() != expected_digest:
                raise EngineeringError("prefinal package content changed")

    def close(self) -> None:
        for descriptor, _before, _digest, _payload in reversed(
            list(getattr(self, "files", {}).values())
        ):
            try:
                os.close(descriptor)
            except OSError:
                pass
        self.files = {}
        for descriptor in reversed(getattr(self, "directory_fds", [])):
            try:
                os.close(descriptor)
            except OSError:
                pass
        self.directory_fds = []


def atomic_object(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise EngineeringError("one-shot artifact already exists")
    pending = path.with_name(f".{path.name}.pending")
    if pending.exists() or pending.is_symlink():
        raise EngineeringError("one-shot pending artifact already exists")
    descriptor = os.open(
        pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0), 0o600,
    )
    try:
        payload = serialized(value)
        if os.write(descriptor, payload) != len(payload):
            raise EngineeringError("short engineering artifact write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(pending, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def atomic_object_at(directory_fd: int, name: str, value: dict[str, Any]) -> None:
    if "/" in name or name in {"", ".", ".."}:
        raise EngineeringError("unsafe dirfd-relative artifact name")
    pending = f".{name}.pending"
    for candidate in (name, pending):
        try:
            os.stat(candidate, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        raise EngineeringError("one-shot dirfd-relative artifact already exists")
    descriptor = os.open(
        pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0), 0o600,
        dir_fd=directory_fd,
    )
    payload = serialized(value)
    try:
        if os.write(descriptor, payload) != len(payload):
            raise EngineeringError("short dirfd-relative artifact write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(pending, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
    os.fsync(directory_fd)


def bound_regular_at(directory_fd: int, name: str, payload: bytes) -> None:
    descriptor = os.open(
        name, os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0),
        dir_fd=directory_fd,
    )
    try:
        metadata = os.fstat(descriptor)
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
                or metadata.st_size != len(payload)):
            raise EngineeringError("dirfd-relative artifact binding changed")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        if digest.digest() != hashlib.sha256(payload).digest():
            raise EngineeringError("dirfd-relative artifact content changed")
    finally:
        os.close(descriptor)


def exact_root_inventory_at(
    directory_fd: int, expected: dict[str, str], label: str,
) -> None:
    if set(os.listdir(directory_fd)) != set(expected):
        raise EngineeringError(f"{label} has extras or omissions")
    for name, kind in expected.items():
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if kind == "file":
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise EngineeringError(f"{label} has an unsafe file")
        elif kind == "directory":
            if not stat.S_ISDIR(metadata.st_mode):
                raise EngineeringError(f"{label} has an unsafe directory")
        else:
            raise EngineeringError("unknown exact root inventory kind")


class HeldEvidenceTree:
    """Retain every tree entry and its trusted component chain through PASS."""
    def __init__(self, root: Path, lock_fd: int, label: str) -> None:
        self.label = label
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        absolute = Path(os.path.abspath(os.fspath(root)))
        self.dirs = [os.open(absolute.anchor, flags)]
        self.bindings: list[tuple[int, str, int, os.stat_result, bool]] = []
        self.listings: list[tuple[int, list[str]]] = []
        self.files: list[tuple[int, str, int, os.stat_result, str]] = []
        for component in absolute.parts[1:]:
            parent = self.dirs[-1]
            before = os.stat(component, dir_fd=parent, follow_symlinks=False)
            child = os.open(component, flags, dir_fd=parent)
            opened = os.fstat(child)
            if (not stat.S_ISDIR(before.st_mode) or not stat.S_ISDIR(opened.st_mode)
                    or before.st_dev != opened.st_dev or before.st_ino != opened.st_ino):
                raise EngineeringError(f"{label} component binding changed")
            self.dirs.append(child)
            self.bindings.append((parent, component, child, opened, False))
        self.root_fd = self.dirs[-1]
        lock = os.fstat(lock_fd)
        lock_path = os.stat("execution.lock", dir_fd=self.root_fd, follow_symlinks=False)
        if (not stat.S_ISREG(lock.st_mode) or lock.st_nlink != 1 or lock.st_size != 0
                or lock.st_dev != lock_path.st_dev or lock.st_ino != lock_path.st_ino):
            raise EngineeringError(f"{label} held lock binding changed")
        rows: list[list[Any]] = []

        def scan(directory_fd: int, prefix: str) -> None:
            names = sorted(os.listdir(directory_fd)); self.listings.append((directory_fd, names))
            for name in names:
                before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                relative = f"{prefix}/{name}" if prefix else name
                if stat.S_ISDIR(before.st_mode):
                    child = os.open(name, flags, dir_fd=directory_fd)
                    opened = os.fstat(child)
                    if (not stat.S_ISDIR(opened.st_mode) or opened.st_dev != before.st_dev
                            or opened.st_ino != before.st_ino):
                        raise EngineeringError(f"{label} directory binding changed")
                    self.dirs.append(child)
                    self.bindings.append((directory_fd, name, child, opened, True))
                    rows.append([relative, "D"]); scan(child, relative)
                elif stat.S_ISREG(before.st_mode) and before.st_nlink == 1:
                    descriptor = os.open(
                        name, os.O_RDONLY | (
                            os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0
                        ), dir_fd=directory_fd,
                    )
                    opened = os.fstat(descriptor)
                    if (opened.st_dev != before.st_dev or opened.st_ino != before.st_ino
                            or opened.st_size != before.st_size):
                        raise EngineeringError(f"{label} file binding changed")
                    digest = hashlib.sha256()
                    while True:
                        block = os.read(descriptor, 1024 * 1024)
                        if not block:
                            break
                        digest.update(block)
                    value = digest.hexdigest()
                    self.files.append((directory_fd, name, descriptor, opened, value))
                    rows.append([relative, "F", opened.st_size, value])
                else:
                    raise EngineeringError(f"{label} contains a symlink/hardlink/special file")
        scan(self.root_fd, "")
        self.entry_count = len(rows)
        self.sha256 = hashlib.sha256(canonical(sorted(rows, key=lambda row: row[0]))).hexdigest()

    def revalidate(self) -> None:
        stable = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
                  "st_mtime_ns", "st_ctime_ns")
        for directory_fd, names in self.listings:
            if sorted(os.listdir(directory_fd)) != names:
                raise EngineeringError(f"{self.label} inventory changed")
        for parent, name, descriptor, before, strict in self.bindings:
            after = os.fstat(descriptor)
            on_disk = os.stat(name, dir_fd=parent, follow_symlinks=False)
            keys = stable if strict else ("st_dev", "st_ino", "st_mode")
            if any(getattr(before, key) != getattr(after, key) for key in keys) \
                    or any(getattr(after, key) != getattr(on_disk, key) for key in keys):
                raise EngineeringError(f"{self.label} directory changed")
        for parent, name, descriptor, before, expected in self.files:
            after = os.fstat(descriptor)
            on_disk = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if any(getattr(before, key) != getattr(after, key) for key in stable) \
                    or any(getattr(after, key) != getattr(on_disk, key) for key in stable):
                raise EngineeringError(f"{self.label} file changed")
            os.lseek(descriptor, 0, os.SEEK_SET); digest = hashlib.sha256()
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
            if digest.hexdigest() != expected:
                raise EngineeringError(f"{self.label} file content changed")

    def close(self) -> None:
        for _parent, _name, descriptor, _before, _digest in reversed(self.files):
            os.close(descriptor)
        for descriptor in reversed(self.dirs):
            os.close(descriptor)


class HeldRootFile:
    def __init__(self, directory_fd: int, name: str, payload: bytes) -> None:
        self.directory_fd = directory_fd; self.name = name
        self.descriptor = os.open(
            name, os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0),
            dir_fd=directory_fd,
        )
        self.before = os.fstat(self.descriptor)
        self.digest = hashlib.sha256(payload).hexdigest()
        if (not stat.S_ISREG(self.before.st_mode) or self.before.st_nlink != 1
                or self.before.st_size != len(payload)):
            raise EngineeringError("held runner root file is unsafe")
        self.revalidate()

    def revalidate(self) -> None:
        stable = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
                  "st_mtime_ns", "st_ctime_ns")
        after = os.fstat(self.descriptor)
        on_disk = os.stat(self.name, dir_fd=self.directory_fd, follow_symlinks=False)
        if any(getattr(self.before, key) != getattr(after, key) for key in stable) \
                or any(getattr(after, key) != getattr(on_disk, key) for key in stable):
            raise EngineeringError("held runner root file binding changed")
        os.lseek(self.descriptor, 0, os.SEEK_SET); digest = hashlib.sha256()
        while True:
            block = os.read(self.descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        if digest.hexdigest() != self.digest:
            raise EngineeringError("held runner root file content changed")

    def close(self) -> None:
        os.close(self.descriptor)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def acquire_lock(path: Path, *, create: bool) -> int:
    flags = os.O_RDWR | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
    if create:
        flags |= os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    metadata = os.fstat(descriptor)
    on_disk = os.stat(path, follow_symlinks=False)
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
            or metadata.st_size != 0 or metadata.st_dev != on_disk.st_dev
            or metadata.st_ino != on_disk.st_ino):
        os.close(descriptor)
        raise EngineeringError("unsafe engineering execution lock")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise EngineeringError("engineering output is already owned") from None
    return descriptor


def load_primary(package_root: Path, snapshot: HeldPackageSnapshot) -> Any:
    path = package_root / "run_primary.py"
    module = types.ModuleType("jx_xp2_v4_primary_core")
    module.__file__ = os.fspath(path)
    module.__package__ = None
    code = compile(snapshot.payload("run_primary.py"), os.fspath(path), "exec")
    exec(code, module.__dict__)
    return module


def materialize_held(
    snapshot: HeldPackageSnapshot, name: str, directory: Path,
) -> Path:
    path = directory / name
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | (
        os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0
    )
    descriptor = os.open(path, flags, 0o600)
    payload = snapshot.payload(name)
    try:
        if os.write(descriptor, payload) != len(payload):
            raise EngineeringError("short held-input materialization")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if digest_file(path) != snapshot.digest(name):
        raise EngineeringError("held-input materialization changed")
    return path


def reject_existing_symlink_ancestors(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise EngineeringError(f"{label} contains a symlink component")
        if current != absolute and not stat.S_ISDIR(metadata.st_mode):
            raise EngineeringError(f"{label} ancestor is not a directory")
    return absolute


def assert_self_authority(
    registration: dict[str, Any], package_root: Path, snapshot: HeldPackageSnapshot,
    *, canonical_required: bool,
) -> None:
    self_path = lexical_path(Path(__file__), "engineering runner", leaf_kind="file")
    expected = package_root / "run_engineering_boundary.py"
    locked = registration.get("locked_core_files", {})
    if ((canonical_required and self_path != expected)
            or digest_file(self_path) != snapshot.digest(expected.name)
            or locked.get(expected.name) != snapshot.digest(expected.name)):
        raise EngineeringError("executing engineering runner is not registered")


def assert_gate_authority_absent(
    contract: dict[str, Any], package_root: Path,
) -> None:
    gate = contract["engineering_boundary_gate_v1"]
    paths = [package_root / gate["final_registration_path"]] + [
        package_root / gate[key] for key in (
            "engineering_verifier_scratch_root",
            "engineering_verifier_start_path",
            "engineering_verifier_terminal_path",
            "engineering_verification_receipt_path",
        )
    ]
    for path in paths:
        reject_existing_symlink_ancestors(path.parent, "engineering authority parent")
        if path.exists() or path.is_symlink():
            raise EngineeringError("final or verifier authority exists during runner attempt")


def validate_inherited_lock(path: Path, descriptor: int, label: str) -> None:
    try:
        inherited = os.fstat(descriptor)
        on_disk = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise EngineeringError(f"{label} was not inherited") from exc
    if (path.is_symlink() or not stat.S_ISREG(inherited.st_mode)
            or inherited.st_nlink != 1 or inherited.st_size != 0
            or inherited.st_dev != on_disk.st_dev or inherited.st_ino != on_disk.st_ino):
        raise EngineeringError(f"{label} binding changed")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise EngineeringError(f"{label} is not held") from exc


def proc_rss_bytes(pid: int) -> int:
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
    except (FileNotFoundError, ProcessLookupError):
        return 0
    values: dict[str, int] = {}
    for line in lines:
        if line.startswith(("VmRSS:", "VmHWM:")):
            fields = line.split()
            if len(fields) != 3 or fields[2] != "kB":
                raise EngineeringError("child RSS status is malformed")
            values[fields[0][:-1]] = int(fields[1]) * 1024
    if set(values) != {"VmRSS", "VmHWM"}:
        raise EngineeringError("child RSS status is incomplete")
    return max(values.values())


def validate_engineering_registration(
    contract_path: Path, registration_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], str, HeldPackageSnapshot]:
    safe_registration = lexical_path(
        registration_path, "engineering registration", leaf_kind="file"
    )
    safe_contract = lexical_path(contract_path, "contract", leaf_kind="file")
    package_root = safe_registration.parent
    if safe_contract != package_root / "contract_v1.json" \
            or safe_registration != package_root / "engineering_registration_v1.json":
        raise EngineeringError("engineering inputs are not canonical package paths")
    snapshot = HeldPackageSnapshot(package_root)
    contract = parse_object(snapshot.payload("contract_v1.json"))
    registration = parse_object(snapshot.payload("engineering_registration_v1.json"))
    if set(registration) != {
        "schema", "experiment_id", "artifact_class", "outcomes_generated",
        "scientific_evidence_artifact", "created_before_engineering_output",
        "final_registration_absent", "authorization", "locked_core_files",
        "core_inventory_sha256",
    }:
        raise EngineeringError("engineering registration fields changed")
    if contract.get("schema") != CONTRACT_SCHEMA or contract.get("experiment_id") != EXPERIMENT_ID:
        raise EngineeringError("contract identity changed")
    gate = contract.get("engineering_boundary_gate_v1", {})
    if (registration.get("schema") != REGISTRATION_SCHEMA
            or registration.get("experiment_id") != EXPERIMENT_ID
            or registration.get("artifact_class")
            != "LOCAL_PREOUTPUT_ENGINEERING_BOUNDARY_REGISTRATION"
            or registration.get("outcomes_generated") is not False
            or registration.get("scientific_evidence_artifact") is not False
            or registration.get("created_before_engineering_output") is not True
            or registration.get("final_registration_absent") is not True
            or registration.get("authorization")
            != gate.get("engineering_registration_authorizes_only")):
        raise EngineeringError("engineering registration identity/scope changed")
    locked = registration.get("locked_core_files")
    inventory = set(contract.get("result_policy", {}).get("registered_package_inventory", []))
    core = inventory - {"engineering_registration_v1.json", "registration_v1.json"}
    if (not isinstance(locked, dict) or set(locked) != core
            or any(not isinstance(value, str) or len(value) != 64 for value in locked.values())):
        raise EngineeringError("engineering core inventory changed")
    actual = set(snapshot.names)
    if actual != core | {"engineering_registration_v1.json"}:
        raise EngineeringError("prefinal engineering package has extras or omissions")
    rows = []
    for name in sorted(core):
        actual_digest = snapshot.digest(name)
        if actual_digest != locked[name]:
            raise EngineeringError("engineering core file changed")
        rows.append([name, actual_digest])
    core_digest = hashlib.sha256(canonical(rows)).hexdigest()
    if registration.get("core_inventory_sha256") != core_digest:
        raise EngineeringError("engineering core inventory digest changed")
    snapshot.revalidate(core | {"engineering_registration_v1.json"})
    return contract, registration, core_digest, snapshot


def topology(primary: Any, simulation: Any, *, source_mode: str) -> dict[str, Any]:
    projection = primary.decoded_continuation_projection(
        simulation, source_mode=source_mode,
    )
    mercurius = simulation.ri_mercurius
    whfast = simulation.ri_whfast
    ias15 = simulation.ri_ias15
    ias_count = int(ias15._N_allocated)
    map_count = int(ias15._map_allocated_n)

    def double_array_hash(pointer: Any, count: int) -> str | None:
        if count == 0:
            return None
        return hashlib.sha256(b"".join(
            struct.pack(">d", float(pointer[index])) for index in range(count)
        )).hexdigest()

    def coefficient_hashes(group: Any, count: int) -> list[str | None]:
        return [double_array_hash(getattr(group, f"p{index}"), count) for index in range(7)]
    endpoint = primary.live_archive_endpoint_projection(simulation)
    return {
        "source_mode": source_mode,
        "structural_projection_validation_passed": True,
        "simulation": {
            "N": int(simulation.N), "N_allocated": int(simulation.N_allocated),
            "particles_present": bool(simulation._particles),
        },
        "mercurius": {
            "dcrit_count": int(mercurius._N_allocated_dcrit),
            "dcrit_present": bool(mercurius._dcrit),
            "allocated_particle_backup_count": int(mercurius._N_allocated),
            "particles_backup_present": bool(mercurius._particles_backup),
            "encounter_map_present": bool(mercurius._encounter_map),
            "encounter_N": int(mercurius._encounter_N),
            "encounter_N_active": int(mercurius._encounter_N_active),
            "tponly_encounter": int(mercurius._tponly_encounter),
            "allocated_additional_forces_backup_count": int(
                mercurius._N_allocated_additional_forces
            ),
            "additional_forces_backup_present": bool(
                mercurius._particles_backup_additional_forces
            ),
        },
        "whfast": {
            "particle_count": int(whfast._N_allocated),
            "particle_present": bool(whfast._p_jh),
            "temporary_count": int(whfast._N_allocated_tmp),
            "temporary_present": bool(whfast._p_temp),
            "internal_particle_arrays_present": projection["whfast"][
                "internal_particle_arrays_present"
            ],
        },
        "ias15": {
            "stored_coordinate_count": ias_count,
            "map_count": map_count,
            "map_present": bool(ias15._map),
            "direct_pointer_presence": {
                name.removeprefix("_"): bool(getattr(ias15, name))
                for name in ("_at", "_x0", "_v0", "_a0", "_csx", "_csv", "_csa0")
            },
            "coefficient_pointer_presence": {
                group.removeprefix("_"): [
                    bool(getattr(getattr(ias15, group), f"p{index}"))
                    for index in range(7)
                ] for group in ("_g", "_b", "_csb", "_e", "_br", "_er")
            },
            "direct_array_sha256": {
                name.removeprefix("_"): double_array_hash(
                    getattr(ias15, name), ias_count
                ) for name in ("_at", "_x0", "_v0", "_a0", "_csx", "_csv", "_csa0")
            },
            "coefficient_array_sha256": {
                group.removeprefix("_"): coefficient_hashes(
                    getattr(ias15, group), ias_count
                ) for group in ("_g", "_b", "_csb", "_e", "_br", "_er")
            },
            "map_sha256": (
                None if map_count == 0 else hashlib.sha256(b"".join(
                    struct.pack(">i", int(ias15._map[index]))
                    for index in range(map_count)
                )).hexdigest()
            ),
        },
        "normalized_endpoint_projection": endpoint,
        "normalized_endpoint_sha256": primary.live_archive_endpoint_sha256(simulation),
        "strict_projection_sha256": (
            primary.decoded_state_sha256(simulation) if source_mode == "ARCHIVE" else None
        ),
    }


def enforce_resources(
    primary: Any, contract: dict[str, Any], output_root: Path, started_ns: int,
    *, projected_bytes: int = 0,
) -> None:
    caps = contract["resource_caps_per_execution"]
    elapsed = (time.monotonic_ns() - started_ns) / NANOSECONDS_PER_SECOND
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    output_bytes = primary.directory_bytes(output_root)
    if (elapsed >= float(caps["max_wall_seconds_total"])
            or rss > int(caps["max_peak_rss_bytes_per_process"])
            or output_bytes + projected_bytes > int(caps["max_output_bytes"])
            or shutil.disk_usage(output_root).free - projected_bytes
            < int(caps["minimum_free_disk_bytes"])):
        raise EngineeringError("registered engineering resource cap reached")


def arm_tree_fingerprint(primary: Any, arm_dir: Path, lock_fd: int) -> dict[str, Any]:
    rows = primary.held_tree_inventory(arm_dir, lock_fd, "engineering arm tree")
    return {
        "schema": "jx-xp2-v4-engineering-arm-tree-fingerprint/v1",
        "entry_count": len(rows),
        "sha256": hashlib.sha256(canonical(rows)).hexdigest(),
    }


def exercise_arm(
    primary: Any, contract: dict[str, Any], expanded: dict[str, list[list[Any]]],
    arm_id: str, arm_dir: Path, arm_lock_fd: int, output_root: Path, started_ns: int,
    package_root: Path,
) -> dict[str, Any]:
    spec = primary.arm_specification(contract, arm_id)
    saved_candidate = primary.build_simulation(contract, expanded, spec)
    unsaved_control = primary.build_simulation(contract, expanded, spec)
    for sample_index in range(1, 1001):
        assert_gate_authority_absent(contract, package_root)
        target = sample_index * 50.0
        saved_candidate.integrate(target, exact_finish_time=1)
        unsaved_control.integrate(target, exact_finish_time=1)
        if (float(saved_candidate.t) != target or float(unsaved_control.t) != target):
            raise EngineeringError("engineering integration target mismatch")
        enforce_resources(primary, contract, output_root, started_ns)
    pre_save_live = topology(primary, saved_candidate, source_mode="LIVE_BOUNDARY")
    pre_save_control = topology(primary, unsaved_control, source_mode="LIVE_BOUNDARY")
    if (pre_save_live["normalized_endpoint_projection"]
            != pre_save_control["normalized_endpoint_projection"]
            or pre_save_live["normalized_endpoint_sha256"]
            != pre_save_control["normalized_endpoint_sha256"]):
        raise EngineeringError("saved candidate and unsaved control diverged before save")
    boundary_path = arm_dir / "boundary_50000.bin"
    boundary_hash, boundary_size, boundary_decoded = primary.save_simulation_checkpoint(
        saved_candidate, contract, boundary_path, boundary_mode="ENGINEERING_FULL_SEGMENT",
        end_years=50_000.0, dt_years=float(spec["dt_years"]),
        particle_count=int(saved_candidate.N),
    )
    post_save_candidate = topology(
        primary, saved_candidate, source_mode="LIVE_BOUNDARY"
    )
    if (pre_save_live["normalized_endpoint_projection"]
            != post_save_candidate["normalized_endpoint_projection"]):
        raise EngineeringError("save mutated the normalized candidate endpoint")
    decoded = primary.get_rebound(contract).Simulation(str(boundary_path))
    primary.validate_decoded_continuation_settings(
        primary.decoded_continuation_projection(decoded), end_years=50_000.0,
        dt_years=float(spec["dt_years"]), particle_count=int(decoded.N),
    )
    decoded_boundary = topology(primary, decoded, source_mode="ARCHIVE")
    if (pre_save_live["normalized_endpoint_projection"]
            != decoded_boundary["normalized_endpoint_projection"]
            or pre_save_live["normalized_endpoint_sha256"]
            != decoded_boundary["normalized_endpoint_sha256"]):
        raise EngineeringError("engineering boundary endpoint parity failed")
    unsaved_control.integrate(50_050.0, exact_finish_time=1)
    decoded.integrate(50_050.0, exact_finish_time=1)
    live_continued = topology(primary, unsaved_control, source_mode="LIVE_BOUNDARY")
    decoded_continued = topology(primary, decoded, source_mode="LIVE_BOUNDARY")
    if (live_continued["normalized_endpoint_projection"]
            != decoded_continued["normalized_endpoint_projection"]
            or live_continued["normalized_endpoint_sha256"]
            != decoded_continued["normalized_endpoint_sha256"]):
        raise EngineeringError("engineering 50050 restart parity failed")
    live_path = arm_dir / "continued_live_50050.bin"
    decoded_path = arm_dir / "continued_decoded_50050.bin"
    live_hash, live_size, live_decoded = primary.save_simulation_checkpoint(
        unsaved_control, contract, live_path, boundary_mode="ENGINEERING_CONTINUATION_PROBE",
        end_years=50_050.0, dt_years=float(spec["dt_years"]),
        particle_count=int(unsaved_control.N),
    )
    restart_hash, restart_size, restart_decoded = primary.save_simulation_checkpoint(
        decoded, contract, decoded_path, boundary_mode="ENGINEERING_CONTINUATION_PROBE",
        end_years=50_050.0, dt_years=float(spec["dt_years"]),
        particle_count=int(decoded.N),
    )
    if live_decoded != restart_decoded:
        raise EngineeringError("engineering persisted restart projections differ")
    record = {
        "schema": "jx-xp2-v4-engineering-arm-result/v1",
        "experiment_id": EXPERIMENT_ID, "arm_id": arm_id,
        "configuration_id": spec["configuration_id"], "dt_years": spec["dt_years"],
        "pre_save_live_topology": pre_save_live,
        "pre_save_unsaved_control_topology": pre_save_control,
        "post_save_candidate_topology": post_save_candidate,
        "decoded_boundary_topology": decoded_boundary,
        "continued_live_topology": live_continued,
        "continued_decoded_topology": decoded_continued,
        "boundary_normalized_endpoint_equal": True,
        "saved_candidate_unsaved_control_pre_save_endpoint_equal": True,
        "pre_save_post_save_candidate_normalized_endpoint_equal": True,
        "restart_50050_normalized_endpoint_equal": True,
        "archives": {
            "boundary_50000.bin": [boundary_size, boundary_hash, boundary_decoded],
            "continued_live_50050.bin": [live_size, live_hash, live_decoded],
            "continued_decoded_50050.bin": [restart_size, restart_hash, restart_decoded],
        },
        "tracer_metrics_or_classification_emitted": False,
        "nonpromotable": True,
    }
    atomic_object(arm_dir / "arm_result_v1.json", record)
    enforce_resources(primary, contract, output_root, started_ns)
    return record


def validate_required_coverage(
    contract: dict[str, Any], records: dict[str, Any],
) -> dict[str, Any]:
    gate = contract["engineering_boundary_gate_v1"]
    eligible = [records[arm] for arm in ("CI01-P0", "AUDIT-CI01-P0")]
    accessors = {
        "simulation.N_allocated": lambda row, side: row[side]["simulation"]["N_allocated"],
        "mercurius.allocated_particle_backup_count": lambda row, side: row[side]["mercurius"]["allocated_particle_backup_count"],
        "mercurius.particles_backup_present": lambda row, side: row[side]["mercurius"]["particles_backup_present"],
        "mercurius.encounter_map_present": lambda row, side: row[side]["mercurius"]["encounter_map_present"],
    }
    required = gate["required_live_vs_decoded_must_differ_fields_in_CI01_or_AUDIT"]
    if set(required) != set(accessors):
        raise EngineeringError("engineering must-differ declaration changed")
    differences: dict[str, list[str]] = {}
    for field in required:
        matching = [
            row["arm_id"] for row in eligible
            if accessors[field](row, "pre_save_live_topology")
            != accessors[field](row, "decoded_boundary_topology")
        ]
        if not matching:
            raise EngineeringError(f"required allocator normalization was not exercised: {field}")
        differences[field] = matching
    whfast = [
        row["arm_id"] for row in eligible
        if row["pre_save_live_topology"]["whfast"]["internal_particle_arrays_present"]
        and not row["decoded_boundary_topology"]["whfast"]["internal_particle_arrays_present"]
    ]
    ias15 = [
        row["arm_id"] for row in eligible
        if row["pre_save_live_topology"]["ias15"]["stored_coordinate_count"] > 0
        and row["pre_save_live_topology"]["ias15"]["map_count"] == 0
        and row["decoded_boundary_topology"]["ias15"]["map_count"] == 0
        and row["decoded_boundary_topology"]["structural_projection_validation_passed"] is True
    ]
    if not whfast or not ias15:
        raise EngineeringError("required WHFast/IAS15 topology coverage was not exercised")
    return {
        "required_live_decoded_differences": differences,
        "whfast_live_present_decoded_absent_arms": whfast,
        "ias15_live_stored_positive_map_zero_decoded_strict_arms": ias15,
        "status": "PASS",
    }


def internal_arm(args: argparse.Namespace) -> int:
    contract, registration, core_digest, snapshot = validate_engineering_registration(
        args.contract, args.engineering_registration,
    )
    package_root = Path(os.path.abspath(os.fspath(args.engineering_registration))).parent
    assert_self_authority(
        registration, package_root, snapshot, canonical_required=False,
    )
    assert_gate_authority_absent(contract, package_root)
    for supplied, expected_name, label in (
        (args.seed_manifest, "seed_manifest_v1.json", "seed manifest"),
        (args.initial_states, "initial_states_v1.json", "initial states"),
    ):
        if lexical_path(supplied, label, leaf_kind="file") \
                != package_root / expected_name:
            raise EngineeringError(f"internal {label} path is not canonical")
    primary = load_primary(package_root, snapshot)
    primary.validate_inherited_v2_b_guard(args.v2_lock_fd, contract, package_root)
    primary.validate_inherited_v3_a_guard(args.v3_lock_fd, contract, package_root)
    primary._V2_B_GUARD_FD = args.v2_lock_fd
    primary._V3_A_GUARD_FD = args.v3_lock_fd
    held_inputs = tempfile.TemporaryDirectory(prefix="jx-xp2-v4-held-inputs-")
    try:
        held_root = Path(held_inputs.name)
        held_seed = materialize_held(snapshot, "seed_manifest_v1.json", held_root)
        held_initial = materialize_held(snapshot, "initial_states_v1.json", held_root)
        primary.validate_contract(contract, args.contract)
        _manifest, seeds = primary.validate_seed_manifest(contract, held_seed)
        tracers, _digest = primary.make_tracers(contract, seeds)
        artifact, expanded = primary.expand_initial_states(contract, held_initial)
        primary.validate_initial_pairing(tracers, artifact, expanded)
        primary.validate_runtime(contract)
    finally:
        held_inputs.cleanup()
    gate = contract["engineering_boundary_gate_v1"]
    expected_root = Path(os.path.abspath(os.fspath(
        package_root / gate["engineering_output_root"]
    )))
    output_root = lexical_path(args.output_root, "engineering output", leaf_kind="dir")
    if output_root != expected_root or args.arm_id not in ARM_IDS:
        raise EngineeringError("internal engineering target changed")
    arm_dir = lexical_path(
        output_root / args.arm_id, "engineering arm", leaf_kind="dir"
    )
    validate_inherited_lock(
        output_root / "execution.lock", args.root_lock_fd,
        "engineering root execution lock",
    )
    start_path = output_root / "runner_attempt_v1.json"
    start = read_object(start_path)
    if (set(start) != {
            "schema", "experiment_id", "event", "attempt_index",
            "engineering_registration_sha256", "core_inventory_sha256",
            "arm_ids", "resume_allowed", "scientific_output_authorized",
        } or start.get("schema") != ATTEMPT_SCHEMA
            or start.get("experiment_id") != EXPERIMENT_ID
            or start.get("event") != "START" or start.get("attempt_index") != 1
            or start.get("engineering_registration_sha256")
            != snapshot.digest("engineering_registration_v1.json")
            or start.get("core_inventory_sha256") != core_digest
            or start.get("arm_ids") != list(ARM_IDS)
            or start.get("resume_allowed") is not False
            or start.get("scientific_output_authorized") is not False
            or (output_root / "runner_terminal_v1.json").exists()
            or (output_root / gate["engineering_result_filename"]).exists()):
        raise EngineeringError("internal engineering arm lacks one exact durable START")
    validate_inherited_lock(
        arm_dir / "execution.lock", args.arm_lock_fd,
        "engineering arm execution lock",
    )
    allowed = {"execution.lock"}
    if {path.name for path in arm_dir.iterdir()} != allowed:
        raise EngineeringError("internal engineering arm is not fresh")
    exercise_arm(
        primary, contract, expanded, args.arm_id, arm_dir, args.arm_lock_fd,
        output_root, time.monotonic_ns(), package_root,
    )
    assert_gate_authority_absent(contract, package_root)
    expected_names = set(contract["result_policy"]["registered_package_inventory"]) \
        - {"registration_v1.json"}
    snapshot.revalidate(expected_names)
    snapshot.close()
    return 0


def supervise_arm(
    primary: Any, contract: dict[str, Any], args: argparse.Namespace,
    package_root: Path, arm_id: str, arm_lock: int, root_lock: int,
    started_ns: int,
) -> dict[str, Any]:
    if not isinstance(primary._V2_B_GUARD_FD, int) \
            or not isinstance(primary._V3_A_GUARD_FD, int):
        raise EngineeringError("engineering lineage guards are not held")
    command = [
        sys.executable, os.fspath(args.held_runner_path),
        "--contract", os.fspath(args.contract),
        "--seed-manifest", os.fspath(args.seed_manifest),
        "--initial-states", os.fspath(args.initial_states),
        "--engineering-registration", os.fspath(args.engineering_registration),
        "--output-root", os.fspath(args.output_root),
        "--internal-arm", "--arm-id", arm_id,
        "--root-lock-fd", str(root_lock), "--arm-lock-fd", str(arm_lock),
        "--v2-lock-fd", str(primary._V2_B_GUARD_FD),
        "--v3-lock-fd", str(primary._V3_A_GUARD_FD),
    ]
    environment = primary.child_environment(contract)
    assert_gate_authority_absent(contract, package_root)
    enforce_resources(primary, contract, args.output_root, started_ns)
    process = subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, env=environment, start_new_session=True,
        close_fds=True, pass_fds=(
            root_lock, arm_lock, primary._V2_B_GUARD_FD, primary._V3_A_GUARD_FD,
        ),
    )
    caps = contract["resource_caps_per_execution"]
    arm_started_ns = time.monotonic_ns()
    try:
        while True:
            return_code = process.poll()
            if return_code is not None:
                break
            assert_gate_authority_absent(contract, package_root)
            elapsed = (time.monotonic_ns() - arm_started_ns) / NANOSECONDS_PER_SECOND
            total_elapsed = (time.monotonic_ns() - started_ns) / NANOSECONDS_PER_SECOND
            rss = proc_rss_bytes(process.pid)
            try:
                enforce_resources(primary, contract, args.output_root, started_ns)
            except BaseException:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
            if (elapsed >= float(caps["max_wall_seconds_per_segment_attempt"])
                    or total_elapsed >= float(caps["max_wall_seconds_total"])
                    or rss > int(caps["max_peak_rss_bytes_per_process"])
                    or rss > int(caps["max_aggregate_child_rss_bytes"])):
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise EngineeringError("engineering arm process exceeded a registered cap")
            time.sleep(float(caps["watchdog_poll_seconds"]))
        if return_code != 0:
            raise EngineeringError("engineering arm process failed closed")
    except BaseException:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise
    record = read_object(args.output_root / arm_id / "arm_result_v1.json")
    if (record.get("schema") != "jx-xp2-v4-engineering-arm-result/v1"
            or record.get("experiment_id") != EXPERIMENT_ID
            or record.get("arm_id") != arm_id
            or record.get("nonpromotable") is not True
            or record.get("tracer_metrics_or_classification_emitted") is not False):
        raise EngineeringError("engineering child result identity changed")
    return record


def execute(args: argparse.Namespace) -> int:
    contract, registration, core_digest, snapshot = validate_engineering_registration(
        args.contract, args.engineering_registration,
    )
    package_root = Path(os.path.abspath(os.fspath(args.engineering_registration))).parent
    assert_self_authority(
        registration, package_root, snapshot, canonical_required=True,
    )
    assert_gate_authority_absent(contract, package_root)
    temporary_cache = tempfile.TemporaryDirectory(prefix="jx-xp2-v4-engineering-")
    sys.pycache_prefix = temporary_cache.name
    args.held_runner_path = materialize_held(
        snapshot, "run_engineering_boundary.py", Path(temporary_cache.name)
    )
    primary = load_primary(package_root, snapshot)
    v2_guard = primary.acquire_v2_b_guard(contract, package_root)
    try:
        v3_guard = primary.acquire_v3_a_guard(contract, package_root)
        primary._V2_B_GUARD_FD = v2_guard
        primary._V3_A_GUARD_FD = v3_guard
        try:
            held_root = Path(temporary_cache.name)
            held_seed = materialize_held(snapshot, "seed_manifest_v1.json", held_root)
            held_initial = materialize_held(snapshot, "initial_states_v1.json", held_root)
            primary.validate_contract(contract, args.contract)
            _manifest, seeds = primary.validate_seed_manifest(contract, held_seed)
            tracers, _digest = primary.make_tracers(contract, seeds)
            artifact, expanded = primary.expand_initial_states(contract, held_initial)
            primary.validate_initial_pairing(tracers, artifact, expanded)
            primary.validate_runtime(contract)
            gate = contract["engineering_boundary_gate_v1"]
            for supplied, name in (
                (args.seed_manifest, "seed manifest"),
                (args.initial_states, "initial states"),
            ):
                safe = lexical_path(supplied, name, leaf_kind="file")
                expected = package_root / supplied.name
                if safe != expected:
                    raise EngineeringError(f"{name} is not the canonical package input")
            expected_output = Path(os.path.abspath(os.fspath(
                package_root / gate["engineering_output_root"]
            )))
            supplied_output = Path(os.path.abspath(os.fspath(args.output_root)))
            if supplied_output != expected_output:
                raise EngineeringError("engineering output path changed")
            reject_existing_symlink_ancestors(
                expected_output.parent, "engineering output parent"
            )
            if not expected_output.parent.exists():
                if expected_output.parent.parent != package_root.parent:
                    raise EngineeringError("engineering output parent location changed")
                expected_output.parent.mkdir()
                fsync_directory(expected_output.parent.parent)
            lexical_path(
                expected_output.parent, "engineering output parent", leaf_kind="dir"
            )
            if args.output_root.exists() or args.output_root.is_symlink():
                raise EngineeringError("engineering attempt is one-shot and output must be absent")
            args.output_root = supplied_output
            assert_gate_authority_absent(contract, package_root)
            args.output_root.mkdir()
            fsync_directory(args.output_root.parent)
            root_dir_fd = os.open(
                args.output_root,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0),
            )
            root_lock = acquire_lock(args.output_root / "execution.lock", create=True)
            started_ns = time.monotonic_ns()
            held_root_files: list[HeldRootFile] = [
                HeldRootFile(root_dir_fd, "execution.lock", b"")
            ]
            try:
                registration_sha = snapshot.digest("engineering_registration_v1.json")
                start = {
                    "schema": ATTEMPT_SCHEMA, "experiment_id": EXPERIMENT_ID,
                    "event": "START", "attempt_index": 1,
                    "engineering_registration_sha256": registration_sha,
                    "core_inventory_sha256": core_digest,
                    "arm_ids": list(ARM_IDS), "resume_allowed": False,
                    "scientific_output_authorized": False,
                }
                assert_gate_authority_absent(contract, package_root)
                atomic_object_at(root_dir_fd, "runner_attempt_v1.json", start)
                held_root_files.append(
                    HeldRootFile(root_dir_fd, "runner_attempt_v1.json", serialized(start))
                )
                arm_records: dict[str, Any] = {}
                fingerprints: dict[str, Any] = {}
                arm_locks: list[int] = []
                held_arm_trees: list[HeldEvidenceTree] = []
                try:
                    for arm_id in ARM_IDS:
                        assert_gate_authority_absent(contract, package_root)
                        arm_dir = args.output_root / arm_id
                        arm_dir.mkdir()
                        fsync_directory(args.output_root)
                        arm_lock = acquire_lock(arm_dir / "execution.lock", create=True)
                        arm_locks.append(arm_lock)
                        arm_records[arm_id] = supervise_arm(
                            primary, contract, args, package_root, arm_id, arm_lock,
                            root_lock, started_ns,
                        )
                        fingerprints[arm_id] = arm_tree_fingerprint(
                            primary, arm_dir, arm_lock,
                        )
                    coverage = validate_required_coverage(contract, arm_records)
                    for index, arm_id in enumerate(ARM_IDS):
                        tree = HeldEvidenceTree(
                            args.output_root / arm_id, arm_locks[index],
                            f"engineering runner {arm_id} retained tree",
                        )
                        if (tree.entry_count != fingerprints[arm_id]["entry_count"]
                                or tree.sha256 != fingerprints[arm_id]["sha256"]):
                            raise EngineeringError(
                                "retained engineering arm fingerprint changed"
                            )
                        held_arm_trees.append(tree)
                    for arm_id in ARM_IDS:
                        if arm_tree_fingerprint(
                                primary, args.output_root / arm_id,
                                arm_locks[list(ARM_IDS).index(arm_id)],
                        ) != fingerprints[arm_id]:
                            raise EngineeringError(
                                "engineering arm changed before result publication"
                            )
                    exact_root_inventory_at(
                        root_dir_fd,
                        {
                            "execution.lock": "file", "runner_attempt_v1.json": "file",
                            **{arm_id: "directory" for arm_id in ARM_IDS},
                        },
                        "engineering runner root before result",
                    )
                    for tree in held_arm_trees:
                        tree.revalidate()
                    # Revalidate the immutable prefinal package, the absence of
                    # final authority, and resource capacity immediately before
                    # publishing PASS.  A PASS without its exact result remains
                    # a permanently consumed, invalid v4 gate attempt.
                    expected_package_names = set(
                        contract["result_policy"]["registered_package_inventory"]
                    ) - {"registration_v1.json"}
                    snapshot.revalidate(expected_package_names)
                    result = {
                        "schema": RESULT_SCHEMA, "experiment_id": EXPERIMENT_ID,
                        "status": "PASS", "artifact_class": "NONSCIENTIFIC_ENGINEERING_DIAGNOSTIC",
                        "engineering_registration_sha256": registration_sha,
                        "core_inventory_sha256": core_digest,
                        "runner_start_sha256": hashlib.sha256(serialized(start)).hexdigest(),
                        "arm_tree_fingerprints": fingerprints,
                        "arms": arm_records,
                        "required_topology_coverage": coverage,
                        "scientific_outcomes_gates_labels_or_classification": None,
                        "nonpromotable": True, "authorizes_official_execution": False,
                    }
                    result_payload = serialized(result)
                    result_sha256 = hashlib.sha256(result_payload).hexdigest()
                    terminal = {
                        "schema": ATTEMPT_SCHEMA, "experiment_id": EXPERIMENT_ID,
                        "event": "PASS", "attempt_index": 1,
                        "start_sha256": result["runner_start_sha256"],
                        "engineering_registration_sha256": registration_sha,
                        "arm_tree_fingerprints": fingerprints,
                        "result_filename": gate["engineering_result_filename"],
                        "result_size_bytes": len(result_payload),
                        "result_sha256": result_sha256,
                        "scientific_output_emitted": False,
                    }
                    projected = len(serialized(terminal)) + len(serialized(result))
                    enforce_resources(
                        primary, contract, args.output_root, started_ns,
                        projected_bytes=projected,
                    )
                    assert_gate_authority_absent(contract, package_root)
                    atomic_object_at(
                        root_dir_fd, gate["engineering_result_filename"], result,
                    )
                    bound_regular_at(
                        root_dir_fd, gate["engineering_result_filename"], result_payload,
                    )
                    held_root_files.append(HeldRootFile(
                        root_dir_fd, gate["engineering_result_filename"], result_payload,
                    ))
                    for tree in held_arm_trees:
                        tree.revalidate()
                    for root_file in held_root_files:
                        root_file.revalidate()
                    # PASS is the last runner artifact and only binds a complete,
                    # already-fsynced nonauthorizing result.
                    assert_gate_authority_absent(contract, package_root)
                    snapshot.revalidate(expected_package_names)
                    for arm_id in ARM_IDS:
                        if arm_tree_fingerprint(
                                primary, args.output_root / arm_id,
                                arm_locks[list(ARM_IDS).index(arm_id)],
                        ) != fingerprints[arm_id]:
                            raise EngineeringError(
                                "engineering arm changed before PASS publication"
                            )
                    exact_root_inventory_at(
                        root_dir_fd,
                        {
                            "execution.lock": "file", "runner_attempt_v1.json": "file",
                            gate["engineering_result_filename"]: "file",
                            **{arm_id: "directory" for arm_id in ARM_IDS},
                        },
                        "engineering runner root before PASS",
                    )
                    bound_regular_at(
                        root_dir_fd, "runner_attempt_v1.json", serialized(start),
                    )
                    bound_regular_at(
                        root_dir_fd, gate["engineering_result_filename"], result_payload,
                    )
                    enforce_resources(
                        primary, contract, args.output_root, started_ns,
                        projected_bytes=len(serialized(terminal)),
                    )
                    assert_gate_authority_absent(contract, package_root)
                    atomic_object_at(root_dir_fd, "runner_terminal_v1.json", terminal)
                    held_root_files.append(HeldRootFile(
                        root_dir_fd, "runner_terminal_v1.json", serialized(terminal),
                    ))
                    exact_root_inventory_at(
                        root_dir_fd,
                        {
                            "execution.lock": "file", "runner_attempt_v1.json": "file",
                            gate["engineering_result_filename"]: "file",
                            "runner_terminal_v1.json": "file",
                            **{arm_id: "directory" for arm_id in ARM_IDS},
                        },
                        "engineering runner completed root",
                    )
                    for tree in held_arm_trees:
                        tree.revalidate()
                    for root_file in held_root_files:
                        root_file.revalidate()
                finally:
                    for root_file in reversed(held_root_files):
                        root_file.close()
                    for tree in reversed(held_arm_trees):
                        tree.close()
                    for descriptor in reversed(arm_locks):
                        os.close(descriptor)
            finally:
                os.close(root_lock)
                os.close(root_dir_fd)
        finally:
            os.close(v3_guard)
    finally:
        os.close(v2_guard)
        temporary_cache.cleanup()
        snapshot.close()
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--contract", type=Path, required=True)
    result.add_argument("--seed-manifest", type=Path, required=True)
    result.add_argument("--initial-states", type=Path, required=True)
    result.add_argument("--engineering-registration", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--internal-arm", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--arm-id", choices=ARM_IDS, help=argparse.SUPPRESS)
    result.add_argument("--root-lock-fd", type=int, help=argparse.SUPPRESS)
    result.add_argument("--arm-lock-fd", type=int, help=argparse.SUPPRESS)
    result.add_argument("--v2-lock-fd", type=int, help=argparse.SUPPRESS)
    result.add_argument("--v3-lock-fd", type=int, help=argparse.SUPPRESS)
    return result


def main() -> int:
    args = parser().parse_args()
    hidden = (
        args.arm_id, args.root_lock_fd, args.arm_lock_fd,
        args.v2_lock_fd, args.v3_lock_fd,
    )
    if args.internal_arm:
        if any(value is None for value in hidden):
            raise EngineeringError("internal engineering arm arguments incomplete")
        return internal_arm(args)
    if any(value is not None for value in hidden):
        raise EngineeringError("external engineering runner received internal arguments")
    return execute(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"engineering boundary failed closed: {type(error).__name__}", file=sys.stderr)
        raise SystemExit(2)
