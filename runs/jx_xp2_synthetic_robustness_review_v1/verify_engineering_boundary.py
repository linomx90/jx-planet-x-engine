#!/usr/bin/env python3
"""Independent one-shot verifier for the registered XP2-v4 boundary exercise.

The verifier imports neither numerical runner.  It redecodes every registered
archive through the independent stored-artifact verifier and performs only the
three explicitly registered 50,000 -> 50,050 restart probes.  Its diagnostics
are non-scientific, nonpromotable, and cannot authorize A/B/DOP by themselves.
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
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "jx-xp2-public-synthetic-robustness-v4"
CONTRACT_SCHEMA = "jx-xp2-robustness-contract/v3"
REGISTRATION_SCHEMA = "jx-xp2-v4-engineering-registration/v1"
RUNNER_ATTEMPT_SCHEMA = "jx-xp2-v4-engineering-attempt/v1"
RUNNER_RESULT_SCHEMA = "jx-xp2-v4-engineering-boundary-result/v1"
VERIFIER_ATTEMPT_SCHEMA = "jx-xp2-v4-engineering-verifier-attempt/v1"
VERIFIER_ARM_SCHEMA = "jx-xp2-v4-engineering-verifier-arm/v1"
RECEIPT_SCHEMA = "jx-xp2-v4-engineering-boundary-verification/v1"
ARM_IDS = ("M0", "CI01-P0", "AUDIT-CI01-P0")
ENDPOINT_DOMAIN = b"jx-xp2-mercurius-live-archive-endpoint/v1\0"
NANOSECONDS_PER_SECOND = 1_000_000_000
MAX_ARCHIVE_BYTES = 1_048_576
_HELD_ARCHIVES: list[tuple[Path, int, os.stat_result, str]] = []
_HELD_JSON: list[tuple[Path, int, os.stat_result, str]] = []
_HELD_JSON_AT: list[tuple[int, str, int, os.stat_result, str]] = []

sys.dont_write_bytecode = True


class BoundaryVerificationError(RuntimeError):
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
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise BoundaryVerificationError("duplicate JSON key")
            result[key] = value
        return result

    def finite(text: str) -> float:
        value = float(text); exact = decimal.Decimal(text)
        if (not math.isfinite(value) or not exact.is_finite()
                or (value == 0.0 and exact != 0)):
            raise BoundaryVerificationError("invalid JSON number")
        return value

    def reject(text: str) -> None:
        raise BoundaryVerificationError(f"invalid JSON constant: {text}")

    result = json.loads(
        payload.decode("utf-8"), object_pairs_hook=unique,
        parse_float=finite, parse_constant=reject,
    )
    if not isinstance(result, dict):
        raise BoundaryVerificationError("JSON root is not an object")
    return result


def read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise BoundaryVerificationError("unsafe JSON artifact")
    return parse_object(path.read_bytes())


def read_held_object(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
    descriptor = os.open(path, flags)
    before = os.fstat(descriptor); on_disk = os.stat(path, follow_symlinks=False)
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or before.st_dev != on_disk.st_dev or before.st_ino != on_disk.st_ino):
        os.close(descriptor)
        raise BoundaryVerificationError("held JSON binding is unsafe")
    chunks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        chunks.append(block)
    payload = b"".join(chunks); digest = hashlib.sha256(payload).hexdigest()
    _HELD_JSON.append((path, descriptor, before, digest))
    return parse_object(payload)


def revalidate_held_json() -> None:
    stable = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
              "st_mtime_ns", "st_ctime_ns")
    for path, descriptor, before, expected_digest in _HELD_JSON:
        after = os.fstat(descriptor); on_disk = os.stat(path, follow_symlinks=False)
        if any(getattr(before, key) != getattr(after, key) for key in stable) \
                or any(getattr(after, key) != getattr(on_disk, key) for key in stable):
            raise BoundaryVerificationError("held JSON artifact changed")
        os.lseek(descriptor, 0, os.SEEK_SET); digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        if digest.hexdigest() != expected_digest:
            raise BoundaryVerificationError("held JSON content changed")
    for directory_fd, name, descriptor, before, expected_digest in _HELD_JSON_AT:
        after = os.fstat(descriptor)
        on_disk = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if any(getattr(before, key) != getattr(after, key) for key in stable) \
                or any(getattr(after, key) != getattr(on_disk, key) for key in stable):
            raise BoundaryVerificationError("held dirfd JSON artifact changed")
        os.lseek(descriptor, 0, os.SEEK_SET); digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        if digest.hexdigest() != expected_digest:
            raise BoundaryVerificationError("held dirfd JSON content changed")


def held_json_digest(path: Path) -> str:
    absolute = Path(os.path.abspath(os.fspath(path)))
    for held_path, _descriptor, _before, digest in reversed(_HELD_JSON):
        if Path(os.path.abspath(os.fspath(held_path))) == absolute:
            return digest
    raise BoundaryVerificationError("JSON artifact is not retained")


def close_held_json() -> None:
    for _path, descriptor, _before, _digest in reversed(_HELD_JSON):
        try:
            os.close(descriptor)
        except OSError:
            pass
    _HELD_JSON.clear()
    for _directory_fd, _name, descriptor, _before, _digest in reversed(_HELD_JSON_AT):
        try:
            os.close(descriptor)
        except OSError:
            pass
    _HELD_JSON_AT.clear()


def lexical(path: Path, label: str, *, kind: str, allow_missing: bool = False) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for index, component in enumerate(absolute.parts[1:], start=1):
        current /= component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            if allow_missing:
                break
            raise BoundaryVerificationError(f"{label} component is missing") from None
        if stat.S_ISLNK(metadata.st_mode):
            raise BoundaryVerificationError(f"{label} contains a symlink")
        final = index == len(absolute.parts) - 1
        if not final and not stat.S_ISDIR(metadata.st_mode):
            raise BoundaryVerificationError(f"{label} ancestor is not a directory")
        if final and kind == "file" and (
            not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
        ):
            raise BoundaryVerificationError(f"{label} is not a safe file")
        if final and kind == "dir" and not stat.S_ISDIR(metadata.st_mode):
            raise BoundaryVerificationError(f"{label} is not a safe directory")
    return absolute


class HeldPackage:
    def __init__(self, root: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        absolute = Path(os.path.abspath(os.fspath(root)))
        self.dirs = [os.open(absolute.anchor, flags)]
        self.dir_bindings: list[tuple[int, str, int, os.stat_result]] = []
        self.files: dict[str, tuple[int, os.stat_result, str, bytes]] = {}
        for component in absolute.parts[1:]:
            parent = self.dirs[-1]
            before = os.stat(component, dir_fd=parent, follow_symlinks=False)
            child = os.open(component, flags, dir_fd=parent)
            opened = os.fstat(child)
            if (not stat.S_ISDIR(before.st_mode) or not stat.S_ISDIR(opened.st_mode)
                    or before.st_dev != opened.st_dev or before.st_ino != opened.st_ino):
                raise BoundaryVerificationError("package directory binding changed")
            self.dirs.append(child)
            self.dir_bindings.append((parent, component, child, opened))
        self.root_fd = self.dirs[-1]
        self.names = sorted(os.listdir(self.root_fd))
        for name in self.names:
            before = os.stat(name, dir_fd=self.root_fd, follow_symlinks=False)
            file_flags = os.O_RDONLY | (
                os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0
            )
            descriptor = os.open(name, file_flags, dir_fd=self.root_fd)
            opened = os.fstat(descriptor)
            if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1
                    or opened.st_dev != before.st_dev or opened.st_ino != before.st_ino):
                raise BoundaryVerificationError("package contains an unsafe entry")
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

    def payload(self, name: str) -> bytes:
        return self.files[name][3]

    def digest(self, name: str) -> str:
        return self.files[name][2]

    def revalidate(self, expected: set[str]) -> None:
        if set(self.names) != expected or sorted(os.listdir(self.root_fd)) != self.names:
            raise BoundaryVerificationError("package inventory changed")
        stable = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
                  "st_mtime_ns", "st_ctime_ns")
        for index, (parent, name, descriptor, before) in enumerate(self.dir_bindings):
            after = os.fstat(descriptor)
            on_disk = os.stat(name, dir_fd=parent, follow_symlinks=False)
            keys = stable if index == len(self.dir_bindings) - 1 \
                else ("st_dev", "st_ino", "st_mode")
            if any(getattr(before, key) != getattr(after, key) for key in keys) \
                    or any(getattr(after, key) != getattr(on_disk, key) for key in keys):
                raise BoundaryVerificationError("package directory changed")
        for name, (descriptor, before, expected_digest, _payload) in self.files.items():
            after = os.fstat(descriptor)
            on_disk = os.stat(name, dir_fd=self.root_fd, follow_symlinks=False)
            if any(getattr(before, key) != getattr(after, key) for key in stable) \
                    or any(getattr(after, key) != getattr(on_disk, key) for key in stable):
                raise BoundaryVerificationError("package file binding changed")
            os.lseek(descriptor, 0, os.SEEK_SET)
            digest = hashlib.sha256()
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
            if digest.hexdigest() != expected_digest:
                raise BoundaryVerificationError("package file content changed")

    def close(self) -> None:
        for descriptor, _before, _digest, _payload in reversed(list(self.files.values())):
            os.close(descriptor)
        self.files = {}
        for descriptor in reversed(self.dirs):
            os.close(descriptor)
        self.dirs = []


def load_independent(package_root: Path, held: HeldPackage) -> Any:
    path = package_root / "verify_replay.py"
    module = types.ModuleType("jx_xp2_v4_independent_boundary_core")
    module.__file__ = os.fspath(path)
    module.__package__ = None
    code = compile(held.payload("verify_replay.py"), os.fspath(path), "exec")
    exec(code, module.__dict__)
    return module


def assert_self_authority(
    registration: dict[str, Any], package_root: Path, held: HeldPackage,
    *, canonical_required: bool,
) -> None:
    self_path = lexical(Path(__file__), "engineering verifier", kind="file")
    expected = package_root / "verify_engineering_boundary.py"
    locked = registration["locked_core_files"]
    if ((canonical_required and self_path != expected)
            or digest_file(self_path) != held.digest(expected.name)
            or locked.get(expected.name) != held.digest(expected.name)):
        raise BoundaryVerificationError("executing engineering verifier is not registered")


def materialize_held(held: HeldPackage, name: str, directory: Path) -> Path:
    path = directory / name
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | (
        os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0
    )
    descriptor = os.open(path, flags, 0o600); payload = held.payload(name)
    try:
        if os.write(descriptor, payload) != len(payload):
            raise BoundaryVerificationError("short held verifier materialization")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if digest_file(path) != held.digest(name):
        raise BoundaryVerificationError("held verifier materialization changed")
    return path


def validate_registration(
    contract_path: Path, registration_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], str, HeldPackage]:
    safe_contract = lexical(contract_path, "contract", kind="file")
    safe_registration = lexical(
        registration_path, "engineering registration", kind="file"
    )
    package_root = safe_registration.parent
    if (safe_contract != package_root / "contract_v1.json"
            or safe_registration != package_root / "engineering_registration_v1.json"):
        raise BoundaryVerificationError("engineering package paths are not canonical")
    held = HeldPackage(package_root)
    contract = parse_object(held.payload("contract_v1.json"))
    registration = parse_object(held.payload("engineering_registration_v1.json"))
    if (contract.get("schema") != CONTRACT_SCHEMA
            or contract.get("experiment_id") != EXPERIMENT_ID):
        raise BoundaryVerificationError("contract identity changed")
    initial = parse_object(held.payload("initial_states_v1.json"))
    if (initial.get("schema") != "jx-xp2-barycentric-initial-states/v1"
            or initial.get("experiment_id") != "jx-xp2-public-synthetic-robustness-v1"
            or held.digest("initial_states_v1.json")
            != contract.get("initial_state_policy", {}).get("artifact_sha256")):
        raise BoundaryVerificationError("held initial-state artifact changed")
    if set(registration) != {
        "schema", "experiment_id", "artifact_class", "outcomes_generated",
        "scientific_evidence_artifact", "created_before_engineering_output",
        "final_registration_absent", "authorization", "locked_core_files",
        "core_inventory_sha256",
    }:
        raise BoundaryVerificationError("engineering registration fields changed")
    gate = contract["engineering_boundary_gate_v1"]
    if (registration.get("schema") != REGISTRATION_SCHEMA
            or registration.get("experiment_id") != EXPERIMENT_ID
            or registration.get("artifact_class")
            != "LOCAL_PREOUTPUT_ENGINEERING_BOUNDARY_REGISTRATION"
            or registration.get("outcomes_generated") is not False
            or registration.get("scientific_evidence_artifact") is not False
            or registration.get("created_before_engineering_output") is not True
            or registration.get("final_registration_absent") is not True
            or registration.get("authorization")
            != gate["engineering_registration_authorizes_only"]):
        raise BoundaryVerificationError("engineering registration scope changed")
    inventory = set(contract["result_policy"]["registered_package_inventory"])
    core = inventory - {"engineering_registration_v1.json", "registration_v1.json"}
    locked = registration.get("locked_core_files")
    if not isinstance(locked, dict) or set(locked) != core:
        raise BoundaryVerificationError("engineering registration inventory changed")
    expected_names = core | {"engineering_registration_v1.json"}
    held.revalidate(expected_names)
    rows = []
    for name in sorted(core):
        if locked[name] != held.digest(name):
            raise BoundaryVerificationError("engineering registered core changed")
        rows.append([name, held.digest(name)])
    core_digest = hashlib.sha256(canonical(rows)).hexdigest()
    if registration.get("core_inventory_sha256") != core_digest:
        raise BoundaryVerificationError("engineering core digest changed")
    return contract, registration, core_digest, held


def atomic_object(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise BoundaryVerificationError("one-shot artifact already exists")
    pending = path.with_name(f".{path.name}.pending")
    if pending.exists() or pending.is_symlink():
        raise BoundaryVerificationError("stale pending engineering artifact")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | (
        os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0
    )
    descriptor = os.open(pending, flags, 0o600)
    try:
        payload = serialized(value)
        if os.write(descriptor, payload) != len(payload):
            raise BoundaryVerificationError("short artifact write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(pending, path)
    fsync_dir(path.parent)


def atomic_object_at(directory_fd: int, name: str, value: dict[str, Any]) -> None:
    if "/" in name or name in {"", ".", ".."}:
        raise BoundaryVerificationError("unsafe dirfd-relative artifact name")
    pending = f".{name}.pending"
    for candidate in (name, pending):
        try:
            os.stat(candidate, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        raise BoundaryVerificationError("one-shot dirfd-relative artifact already exists")
    payload = serialized(value)
    descriptor = os.open(
        pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0), 0o600,
        dir_fd=directory_fd,
    )
    try:
        if os.write(descriptor, payload) != len(payload):
            raise BoundaryVerificationError("short dirfd-relative artifact write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(pending, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
    os.fsync(directory_fd)


def read_held_object_at(directory_fd: int, name: str) -> dict[str, Any]:
    descriptor = os.open(
        name, os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0),
        dir_fd=directory_fd,
    )
    before = os.fstat(descriptor)
    on_disk = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or before.st_dev != on_disk.st_dev or before.st_ino != on_disk.st_ino):
        os.close(descriptor)
        raise BoundaryVerificationError("held dirfd JSON binding is unsafe")
    chunks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        chunks.append(block)
    payload = b"".join(chunks)
    digest = hashlib.sha256(payload).hexdigest()
    _HELD_JSON_AT.append((directory_fd, name, descriptor, before, digest))
    return parse_object(payload)


def held_json_digest_at(directory_fd: int, name: str) -> str:
    for held_directory, held_name, _descriptor, _before, digest in reversed(_HELD_JSON_AT):
        if held_directory == directory_fd and held_name == name:
            return digest
    raise BoundaryVerificationError("dirfd JSON artifact is not retained")


def exact_directory_inventory_at(
    directory_fd: int, expected: dict[str, str], label: str,
) -> None:
    if set(os.listdir(directory_fd)) != set(expected):
        raise BoundaryVerificationError(f"{label} has extras or omissions")
    for name, kind in expected.items():
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if kind == "file":
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise BoundaryVerificationError(f"{label} has an unsafe file")
        elif kind == "directory":
            if not stat.S_ISDIR(metadata.st_mode):
                raise BoundaryVerificationError(f"{label} has an unsafe directory")
        else:
            raise BoundaryVerificationError("unknown exact inventory kind")


class HeldEvidenceTree:
    """Retain a complete evidence tree and trusted path chain through PASS."""
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
                raise BoundaryVerificationError(f"{label} component binding changed")
            self.dirs.append(child)
            self.bindings.append((parent, component, child, opened, False))
        self.root_fd = self.dirs[-1]
        lock = os.fstat(lock_fd)
        lock_path = os.stat("execution.lock", dir_fd=self.root_fd, follow_symlinks=False)
        if (not stat.S_ISREG(lock.st_mode) or lock.st_nlink != 1 or lock.st_size != 0
                or lock.st_dev != lock_path.st_dev or lock.st_ino != lock_path.st_ino):
            raise BoundaryVerificationError(f"{label} held lock binding changed")
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
                        raise BoundaryVerificationError(f"{label} directory binding changed")
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
                        raise BoundaryVerificationError(f"{label} file binding changed")
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
                    raise BoundaryVerificationError(
                        f"{label} contains a symlink/hardlink/special file"
                    )
        scan(self.root_fd, "")
        self.entry_count = len(rows)
        self.sha256 = hashlib.sha256(canonical(sorted(rows, key=lambda row: row[0]))).hexdigest()

    def revalidate(self) -> None:
        stable = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
                  "st_mtime_ns", "st_ctime_ns")
        for directory_fd, names in self.listings:
            if sorted(os.listdir(directory_fd)) != names:
                raise BoundaryVerificationError(f"{self.label} inventory changed")
        for parent, name, descriptor, before, strict in self.bindings:
            after = os.fstat(descriptor)
            on_disk = os.stat(name, dir_fd=parent, follow_symlinks=False)
            keys = stable if strict else ("st_dev", "st_ino", "st_mode")
            if any(getattr(before, key) != getattr(after, key) for key in keys) \
                    or any(getattr(after, key) != getattr(on_disk, key) for key in keys):
                raise BoundaryVerificationError(f"{self.label} directory changed")
        for parent, name, descriptor, before, expected in self.files:
            after = os.fstat(descriptor)
            on_disk = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if any(getattr(before, key) != getattr(after, key) for key in stable) \
                    or any(getattr(after, key) != getattr(on_disk, key) for key in stable):
                raise BoundaryVerificationError(f"{self.label} file changed")
            os.lseek(descriptor, 0, os.SEEK_SET); digest = hashlib.sha256()
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
            if digest.hexdigest() != expected:
                raise BoundaryVerificationError(f"{self.label} file content changed")

    def close(self) -> None:
        for _parent, _name, descriptor, _before, _digest in reversed(self.files):
            os.close(descriptor)
        for descriptor in reversed(self.dirs):
            os.close(descriptor)


def fsync_dir(path: Path) -> None:
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
    metadata = os.fstat(descriptor); on_disk = os.stat(path, follow_symlinks=False)
    if (path.is_symlink() or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1 or metadata.st_size != 0
            or metadata.st_dev != on_disk.st_dev or metadata.st_ino != on_disk.st_ino):
        os.close(descriptor)
        raise BoundaryVerificationError("unsafe engineering execution lock")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise BoundaryVerificationError("engineering tree is still owned") from None
    return descriptor


def register_lock(module: Any, path: Path, descriptor: int, label: str) -> None:
    metadata = os.fstat(descriptor); on_disk = os.stat(path, follow_symlinks=False)
    if (path.is_symlink() or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1 or metadata.st_size != 0
            or metadata.st_dev != on_disk.st_dev or metadata.st_ino != on_disk.st_ino):
        raise BoundaryVerificationError(f"{label} binding changed")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise BoundaryVerificationError(f"{label} is not held") from exc
    module._VERIFICATION_LOCK_FDS[(metadata.st_dev, metadata.st_ino)] = descriptor


def lock_fd(module: Any, path: Path) -> int:
    metadata = os.stat(path, follow_symlinks=False)
    try:
        return module._VERIFICATION_LOCK_FDS[(metadata.st_dev, metadata.st_ino)]
    except KeyError as exc:
        raise BoundaryVerificationError("required retained lock is absent") from exc


def final_registration_absent(contract: dict[str, Any], package_root: Path) -> None:
    path = package_root / contract["engineering_boundary_gate_v1"]["final_registration_path"]
    lexical(path.parent, "final registration parent", kind="dir")
    if path.exists() or path.is_symlink():
        raise BoundaryVerificationError("final registration exists during engineering verification")


def verifier_completion_absent(contract: dict[str, Any], package_root: Path) -> None:
    final_registration_absent(contract, package_root)
    gate = contract["engineering_boundary_gate_v1"]
    for key in ("engineering_verifier_terminal_path", "engineering_verification_receipt_path"):
        path = package_root / gate[key]
        if path.exists() or path.is_symlink():
            raise BoundaryVerificationError("verifier completion artifact appeared early")


def normalized_projection(projection: dict[str, Any]) -> dict[str, Any]:
    mercurius = dict(projection["mercurius"])
    normalized_mercurius = (
        "encounter_N", "encounter_N_active", "tponly_encounter",
        "allocated_particle_backup_count",
        "allocated_additional_forces_backup_count", "particles_backup_present",
        "additional_forces_backup_present", "encounter_map_present",
    )
    for field in normalized_mercurius:
        mercurius.pop(field)
    whfast = dict(projection["whfast"]); whfast.pop("internal_particle_arrays_present")
    ias15 = {
        field: projection["ias15"][field]
        for field in ("epsilon_hex", "min_dt_hex", "adaptive_mode", "iterations_max_exceeded")
    }
    return {
        "schema": "jx-xp2-mercurius-live-archive-endpoint/v1",
        "simulation": projection["simulation"], "mercurius": mercurius,
        "whfast": whfast, "ias15": ias15, "particles": projection["particles"],
        "save_load_normalized_mercurius_fields": list(normalized_mercurius),
        "save_load_normalized_whfast_fields": ["internal_particle_arrays_present"],
        "save_load_normalized_ias15_fields": [
            "stored_coordinate_count", "direct_array_sha256",
            "coefficient_array_sha256", "map_count", "map_sha256",
        ],
        "excluded_noncontinuation_fields": projection["excluded_noncontinuation_fields"],
    }


def endpoint_sha256(projection: dict[str, Any]) -> str:
    return hashlib.sha256(ENDPOINT_DOMAIN + canonical(normalized_projection(projection))).hexdigest()


def validate_strict_settings(
    independent: Any, simulation: Any, projection: dict[str, Any],
    *, end_years: float, dt_years: float, particle_count: int,
) -> None:
    if (projection.get("schema")
            != "jx-xp2-mercurius-decoded-continuation-state/v3"
            or list(projection) != [
                "schema", "simulation", "mercurius", "whfast", "ias15",
                "particles", "excluded_noncontinuation_fields",
            ] or len(projection.get("particles", [])) != particle_count):
        raise BoundaryVerificationError("strict archive projection schema changed")
    h = lambda value: struct.pack(">d", float(value)).hex()
    settings = projection["simulation"]; mercurius = projection["mercurius"]
    whfast = projection["whfast"]; ias15 = projection["ias15"]
    exact_settings = {
        "softening_hex": h(0), "dt_last_done_hex": h(dt_years),
        "steps_done": int(round(end_years / dt_years)), "usleep_hex": h(0),
        "save_messages": 1, "status": 0,
        "particle_capacity_covers_logical_count": True,
        "particle_storage_present": True, "active_memory_ranges_pairwise_disjoint": True,
        "N_var": 0, "N_var_config": 0, "variation_config_present": False,
        "var_rescale_warning": 0, "testparticle_hidewarnings": 0, "hash_ctr": 0,
        "particle_lookup_count": 0, "particle_lookup_allocation_count": 0,
        "particle_lookup_present": False, "gravity": "mercurius", "boundary": "none",
        "collision": "none", "exact_finish_time": 1,
        "force_is_velocity_dependent": 0, "gravity_ignore": 0,
        "exit_max_distance_hex": h(0), "exit_min_distance_hex": h(0),
        "track_energy_offset": 0, "energy_offset_hex": h(0),
        "opening_angle2_hex": h(0.25), "boxsize_hex": [h(0)] * 3,
        "boxsize_max_hex": h(0), "root_size_hex": h(-1), "N_root": 1,
        "N_root_xyz": [1, 1, 1], "N_ghost_xyz": [0, 0, 0],
        "collision_resolve_keep_sorted": 0, "collisions_N": 0,
        "minimum_collision_velocity_hex": h(0),
        "gravity_compensated_sums_present": False,
        "gravity_compensated_sums_allocation_count": 0,
        "tree_root_present": False, "tree_needs_update": 0, "messages_present": False,
        "display_view_present": False, "display_data_present": False,
        "server_data_present": False, "collision_storage_present": False,
        "collision_allocation_count": 0, "collisions_plog_hex": h(0),
        "collisions_log_n": 0, "calculate_megno": 0, "megno_n": 0,
        "N_odes": 0, "odes_allocation_count": 0, "odes_warnings": 0,
        "odes_present": False, "extras_present": False,
        "simulationarchive_auto_interval_hex": h(0),
        "simulationarchive_auto_walltime_hex": h(0),
        "simulationarchive_auto_step": 0, "simulationarchive_next_hex": h(0),
        "simulationarchive_next_step": 0, "simulationarchive_filename_present": False,
    }
    if (float(simulation.t) != end_years or float(simulation.dt) != dt_years
            or int(simulation.N) != particle_count or str(simulation.integrator) != "mercurius"
            or settings.get("t_hex") != h(end_years)
            or settings.get("dt_hex") != h(dt_years)
            or settings.get("N") != particle_count
            or any(settings.get(key) != value for key, value in exact_settings.items())
            or any(settings.get(field) != h(0) for field in (
                "megno_Ys_hex", "megno_Yss_hex", "megno_cov_Yt_hex",
                "megno_var_t_hex", "megno_mean_t_hex", "megno_mean_Y_hex",
                "megno_initial_t_hex",
            )) or any(settings.get("callbacks_present", {}).values())):
        raise BoundaryVerificationError("strict archive simulation settings changed")
    if (mercurius.get("r_crit_hill_hex") != h(3)
            or mercurius.get("safe_mode") != 1 or mercurius.get("mode") != 0
            or mercurius.get("is_synchronized") != 1
            or mercurius.get("recalculate_coordinates_this_timestep") != 1
            or mercurius.get("recalculate_r_crit_this_timestep") != 0
            or mercurius.get("encounter_N") != 0
            or mercurius.get("encounter_N_active") != 0
            or mercurius.get("tponly_encounter") != 0
            or mercurius.get("dcrit_storage_present") is not True
            or mercurius.get("dcrit_capacity_covers_logical_count") is not True
            or len(mercurius.get("dcrit_hex", [])) != particle_count
            or mercurius.get("L_callback_present") is not False
            or mercurius.get("allocated_particle_backup_count") != 0
            or mercurius.get("allocated_additional_forces_backup_count") != 0
            or mercurius.get("particles_backup_present") is not False
            or mercurius.get("additional_forces_backup_present") is not False
            or mercurius.get("encounter_map_present") is not False):
        raise BoundaryVerificationError("strict archive MERCURIUS settings changed")
    expected_whfast = {
        "coordinates": "jacobi", "kernel": "default", "corrector": 0,
        "corrector2": 0, "recalculate_coordinates_this_timestep": 0,
        "safe_mode": 1, "keep_unsynchronized": 0, "is_synchronized": 1,
        "timestep_warning": 0, "unsynchronized_recalculation_warning": 0,
        "internal_particle_arrays_present": False,
    }
    if (whfast != expected_whfast or ias15.get("epsilon_hex") != h(1e-9)
            or ias15.get("min_dt_hex") != h(0)
            or ias15.get("adaptive_mode") != "prs23"
            or ias15.get("iterations_max_exceeded") != 0
            or not independent.valid_primary_ias15_continuation(ias15)
            or any(row.get("simulation_reference_bound_to_parent") is not True
                   or row.get("last_collision_hex") != h(0)
                   or row.get("collision_cell_present")
                   or row.get("additional_properties_present")
                   for row in projection["particles"])):
        raise BoundaryVerificationError("strict archive cache/particle settings changed")


def safe_archive(
    independent: Any, contract: dict[str, Any], path: Path,
    binding: list[Any], *, end_years: float, dt_years: float, particle_count: int,
) -> tuple[Any, dict[str, Any], str]:
    if (not isinstance(binding, list) or len(binding) != 3
            or type(binding[0]) is not int or not isinstance(binding[1], str)
            or not isinstance(binding[2], str) or path.is_symlink()):
        raise BoundaryVerificationError("registered engineering archive binding changed")
    flags = os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
    descriptor = os.open(path, flags)
    metadata = os.fstat(descriptor); on_disk = os.stat(path, follow_symlinks=False)
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
            or metadata.st_dev != on_disk.st_dev or metadata.st_ino != on_disk.st_ino
            or not 0 < metadata.st_size <= MAX_ARCHIVE_BYTES
            or metadata.st_size != binding[0]):
        os.close(descriptor)
        raise BoundaryVerificationError("registered engineering archive inode changed")
    digest = hashlib.sha256()
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        digest.update(block)
    if digest.hexdigest() != binding[1]:
        os.close(descriptor)
        raise BoundaryVerificationError("registered engineering archive bytes changed")
    os.lseek(descriptor, 0, os.SEEK_SET)
    _HELD_ARCHIVES.append((path, descriptor, metadata, binding[1]))
    simulation = independent.verifier_rebound(contract).Simulation(
        f"/proc/self/fd/{descriptor}"
    )
    projection = independent.decoded_primary_continuation_projection(simulation)
    validate_strict_settings(
        independent, simulation, projection, end_years=end_years,
        dt_years=dt_years, particle_count=particle_count,
    )
    digest = independent.decoded_primary_state_sha256(simulation)
    if digest != binding[2]:
        raise BoundaryVerificationError("registered engineering decoded digest changed")
    return simulation, projection, digest


def revalidate_held_archives() -> None:
    stable = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
              "st_mtime_ns", "st_ctime_ns")
    for path, descriptor, before, expected_digest in _HELD_ARCHIVES:
        after = os.fstat(descriptor); on_disk = os.stat(path, follow_symlinks=False)
        if any(getattr(before, key) != getattr(after, key) for key in stable) \
                or any(getattr(after, key) != getattr(on_disk, key) for key in stable):
            raise BoundaryVerificationError("held engineering archive changed")
        os.lseek(descriptor, 0, os.SEEK_SET); digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        if digest.hexdigest() != expected_digest:
            raise BoundaryVerificationError("held engineering archive content changed")


def close_held_archives() -> None:
    for _path, descriptor, _before, _digest in reversed(_HELD_ARCHIVES):
        try:
            os.close(descriptor)
        except OSError:
            pass
    _HELD_ARCHIVES.clear()


def validate_coverage(contract: dict[str, Any], records: dict[str, Any]) -> dict[str, Any]:
    gate = contract["engineering_boundary_gate_v1"]
    eligible = [records[arm] for arm in ("CI01-P0", "AUDIT-CI01-P0")]
    accessors = {
        "simulation.N_allocated": lambda row, side: row[side]["simulation"]["N_allocated"],
        "mercurius.allocated_particle_backup_count": lambda row, side: row[side]["mercurius"]["allocated_particle_backup_count"],
        "mercurius.particles_backup_present": lambda row, side: row[side]["mercurius"]["particles_backup_present"],
        "mercurius.encounter_map_present": lambda row, side: row[side]["mercurius"]["encounter_map_present"],
    }
    required = gate["required_live_vs_decoded_must_differ_fields_in_CI01_or_AUDIT"]
    differences = {
        field: [row["arm_id"] for row in eligible if
                accessors[field](row, "pre_save_live_topology")
                != accessors[field](row, "decoded_boundary_topology")]
        for field in required
    }
    whfast = [row["arm_id"] for row in eligible if
              row["pre_save_live_topology"]["whfast"]["internal_particle_arrays_present"]
              and not row["decoded_boundary_topology"]["whfast"]["internal_particle_arrays_present"]]
    ias15 = [row["arm_id"] for row in eligible if
             row["pre_save_live_topology"]["ias15"]["stored_coordinate_count"] > 0
             and row["pre_save_live_topology"]["ias15"]["map_count"] == 0
             and row["decoded_boundary_topology"]["ias15"]["map_count"] == 0]
    if set(required) != set(accessors) or any(not value for value in differences.values()) \
            or not whfast or not ias15:
        raise BoundaryVerificationError("required engineering topology coverage failed")
    return {
        "required_live_decoded_differences": differences,
        "whfast_live_present_decoded_absent_arms": whfast,
        "ias15_live_stored_positive_map_zero_decoded_strict_arms": ias15,
        "status": "PASS",
    }


def validate_runner_topology(value: Any, *, source_mode: str) -> None:
    if not isinstance(value, dict) or set(value) != {
        "source_mode", "structural_projection_validation_passed", "simulation",
        "mercurius", "whfast", "ias15", "normalized_endpoint_projection",
        "normalized_endpoint_sha256", "strict_projection_sha256",
    }:
        raise BoundaryVerificationError("runner topology fields changed")
    ias = value.get("ias15", {})
    direct = {"at", "x0", "v0", "a0", "csx", "csv", "csa0"}
    coefficients = {"g", "b", "csb", "e", "br", "er"}
    if (value["source_mode"] != source_mode
            or value["structural_projection_validation_passed"] is not True
            or set(value["simulation"]) != {"N", "N_allocated", "particles_present"}
            or set(value["mercurius"]) != {
                "dcrit_count", "dcrit_present", "allocated_particle_backup_count",
                "particles_backup_present", "encounter_map_present", "encounter_N",
                "encounter_N_active", "tponly_encounter",
                "allocated_additional_forces_backup_count",
                "additional_forces_backup_present",
            } or set(value["whfast"]) != {
                "particle_count", "particle_present", "temporary_count",
                "temporary_present", "internal_particle_arrays_present",
            } or set(value["ias15"]) != {
                "stored_coordinate_count", "map_count", "map_present",
                "direct_pointer_presence", "coefficient_pointer_presence",
                "direct_array_sha256", "coefficient_array_sha256", "map_sha256",
            } or set(ias.get("direct_pointer_presence", {})) != direct
            or set(ias.get("direct_array_sha256", {})) != direct
            or set(ias.get("coefficient_pointer_presence", {})) != coefficients
            or set(ias.get("coefficient_array_sha256", {})) != coefficients
            or any(not isinstance(rows, list) or len(rows) != 7
                   or any(type(item) is not bool for item in rows)
                   for rows in ias.get("coefficient_pointer_presence", {}).values())
            or any(not isinstance(rows, list) or len(rows) != 7
                   or any(item is not None and not isinstance(item, str) for item in rows)
                   for rows in ias.get("coefficient_array_sha256", {}).values())
            or any(type(item) is not bool
                   for item in ias.get("direct_pointer_presence", {}).values())
            or any(item is not None and not isinstance(item, str)
                   for item in ias.get("direct_array_sha256", {}).values())
            or value["normalized_endpoint_sha256"]
            != hashlib.sha256(
                ENDPOINT_DOMAIN + canonical(value["normalized_endpoint_projection"])
            ).hexdigest()
            or (source_mode == "ARCHIVE" and not isinstance(
                value["strict_projection_sha256"], str
            )) or (source_mode == "LIVE_BOUNDARY"
                   and value["strict_projection_sha256"] is not None)):
        raise BoundaryVerificationError("runner topology content changed")


def validate_runner(
    independent: Any, contract: dict[str, Any], registration_sha: str,
    core_digest: str, root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    expected_root_names = {
        "execution.lock", "runner_attempt_v1.json", "runner_terminal_v1.json",
        "result_v1.json", *ARM_IDS,
    }
    if {path.name for path in root.iterdir()} != expected_root_names:
        raise BoundaryVerificationError("engineering runner tree has extras or omissions")
    start = read_held_object(root / "runner_attempt_v1.json")
    terminal = read_held_object(root / "runner_terminal_v1.json")
    result_path = root / "result_v1.json"; result = read_held_object(result_path)
    start_sha = held_json_digest(root / "runner_attempt_v1.json")
    result_sha = held_json_digest(result_path)
    if (set(start) != {
            "schema", "experiment_id", "event", "attempt_index",
            "engineering_registration_sha256", "core_inventory_sha256",
            "arm_ids", "resume_allowed", "scientific_output_authorized",
        } or start.get("schema") != RUNNER_ATTEMPT_SCHEMA
            or start.get("experiment_id") != EXPERIMENT_ID or start.get("event") != "START"
            or start.get("attempt_index") != 1
            or start.get("engineering_registration_sha256") != registration_sha
            or start.get("core_inventory_sha256") != core_digest
            or start.get("arm_ids") != list(ARM_IDS)
            or start.get("resume_allowed") is not False
            or start.get("scientific_output_authorized") is not False):
        raise BoundaryVerificationError("runner START changed")
    if (set(terminal) != {
            "schema", "experiment_id", "event", "attempt_index", "start_sha256",
            "engineering_registration_sha256", "arm_tree_fingerprints",
            "result_filename", "result_size_bytes", "result_sha256",
            "scientific_output_emitted",
        } or terminal.get("schema") != RUNNER_ATTEMPT_SCHEMA
            or terminal.get("experiment_id") != EXPERIMENT_ID
            or terminal.get("event") != "PASS" or terminal.get("attempt_index") != 1
            or terminal.get("start_sha256") != start_sha
            or terminal.get("engineering_registration_sha256") != registration_sha
            or terminal.get("result_filename") != "result_v1.json"
            or terminal.get("result_size_bytes") != result_path.stat().st_size
            or terminal.get("result_sha256") != result_sha
            or terminal.get("scientific_output_emitted") is not False):
        raise BoundaryVerificationError("runner PASS changed")
    if (set(result) != {
            "schema", "experiment_id", "status", "artifact_class",
            "engineering_registration_sha256", "core_inventory_sha256",
            "runner_start_sha256", "arm_tree_fingerprints", "arms",
            "required_topology_coverage",
            "scientific_outcomes_gates_labels_or_classification",
            "nonpromotable", "authorizes_official_execution",
        } or result.get("schema") != RUNNER_RESULT_SCHEMA
            or result.get("experiment_id") != EXPERIMENT_ID
            or result.get("status") != "PASS"
            or result.get("artifact_class") != "NONSCIENTIFIC_ENGINEERING_DIAGNOSTIC"
            or result.get("engineering_registration_sha256") != registration_sha
            or result.get("core_inventory_sha256") != core_digest
            or result.get("runner_start_sha256") != start_sha
            or result.get("scientific_outcomes_gates_labels_or_classification") is not None
            or result.get("nonpromotable") is not True
            or result.get("authorizes_official_execution") is not False
            or set(result.get("arms", {})) != set(ARM_IDS)
            or result.get("arm_tree_fingerprints")
            != terminal.get("arm_tree_fingerprints")):
        raise BoundaryVerificationError("runner result changed")
    records: dict[str, dict[str, Any]] = {}
    fingerprints: dict[str, Any] = {}
    expected_n = {"M0": 133, "CI01-P0": 134, "AUDIT-CI01-P0": 134}
    expected_dt = {"M0": 0.125, "CI01-P0": 0.125, "AUDIT-CI01-P0": 0.0625}
    expected_configuration = {"M0": "M0", "CI01-P0": "CI01-P0", "AUDIT-CI01-P0": "CI01-P0"}
    fingerprints_value = result.get("arm_tree_fingerprints", {})
    if (set(fingerprints_value) != set(ARM_IDS)
            or any(set(value) != {"schema", "entry_count", "sha256"}
                   or value.get("schema")
                   != "jx-xp2-v4-engineering-arm-tree-fingerprint/v1"
                   or value.get("entry_count") != 5
                   or not isinstance(value.get("sha256"), str)
                   for value in fingerprints_value.values())):
        raise BoundaryVerificationError("runner fingerprint structure changed")
    for arm in ARM_IDS:
        arm_dir = root / arm
        if {path.name for path in arm_dir.iterdir()} != {
            "execution.lock", "arm_result_v1.json", "boundary_50000.bin",
            "continued_live_50050.bin", "continued_decoded_50050.bin",
        }:
            raise BoundaryVerificationError("runner arm tree has extras or omissions")
        independent.verify_unlocked_execution_lock(
            arm_dir / "execution.lock", f"engineering runner {arm} lock"
        )
        record = read_held_object(arm_dir / "arm_result_v1.json")
        if (set(record) != {
                "schema", "experiment_id", "arm_id", "configuration_id", "dt_years",
                "pre_save_live_topology", "pre_save_unsaved_control_topology",
                "post_save_candidate_topology", "decoded_boundary_topology",
                "continued_live_topology", "continued_decoded_topology",
                "boundary_normalized_endpoint_equal",
                "saved_candidate_unsaved_control_pre_save_endpoint_equal",
                "pre_save_post_save_candidate_normalized_endpoint_equal",
                "restart_50050_normalized_endpoint_equal", "archives",
                "tracer_metrics_or_classification_emitted", "nonpromotable",
            } or record.get("schema") != "jx-xp2-v4-engineering-arm-result/v1"
                or record.get("experiment_id") != EXPERIMENT_ID
                or record.get("arm_id") != arm
                or record.get("configuration_id") != expected_configuration[arm]
                or record.get("dt_years") != expected_dt[arm]
                or record.get("nonpromotable") is not True
                or record.get("tracer_metrics_or_classification_emitted") is not False
                or record.get("boundary_normalized_endpoint_equal") is not True
                or record.get("saved_candidate_unsaved_control_pre_save_endpoint_equal") is not True
                or record.get("pre_save_post_save_candidate_normalized_endpoint_equal") is not True
                or record.get("restart_50050_normalized_endpoint_equal") is not True
                or set(record.get("archives", {})) != {
                    "boundary_50000.bin", "continued_live_50050.bin",
                    "continued_decoded_50050.bin",
                }
                or result["arms"].get(arm) != record):
            raise BoundaryVerificationError("runner arm result changed")
        for key in (
            "pre_save_live_topology", "pre_save_unsaved_control_topology",
            "post_save_candidate_topology", "continued_live_topology",
            "continued_decoded_topology",
        ):
            validate_runner_topology(record[key], source_mode="LIVE_BOUNDARY")
        validate_runner_topology(record["decoded_boundary_topology"], source_mode="ARCHIVE")
        boundary_projection = record["decoded_boundary_topology"]["normalized_endpoint_projection"]
        if (record["pre_save_live_topology"]["normalized_endpoint_projection"]
                != boundary_projection
                or record["pre_save_unsaved_control_topology"]["normalized_endpoint_projection"]
                != boundary_projection
                or record["post_save_candidate_topology"]["normalized_endpoint_projection"]
                != boundary_projection
                or record["continued_live_topology"]["normalized_endpoint_projection"]
                != record["continued_decoded_topology"]["normalized_endpoint_projection"]):
            raise BoundaryVerificationError("runner normalized endpoint parity claims changed")
        decoded_digests: dict[str, str] = {}
        for filename, end in (
            ("boundary_50000.bin", 50_000.0),
            ("continued_live_50050.bin", 50_050.0),
            ("continued_decoded_50050.bin", 50_050.0),
        ):
            _simulation, projection, digest = safe_archive(
                independent, contract, arm_dir / filename,
                record["archives"][filename], end_years=end,
                dt_years=expected_dt[arm], particle_count=expected_n[arm],
            )
            topology_key = (
                "decoded_boundary_topology" if end == 50_000.0
                else ("continued_live_topology" if "live" in filename
                      else "continued_decoded_topology")
            )
            stored_topology = record[topology_key]
            if ((end == 50_000.0 and stored_topology.get("strict_projection_sha256") != digest)
                    or (end == 50_050.0
                        and stored_topology.get("strict_projection_sha256") is not None)
                    or stored_topology.get("normalized_endpoint_projection")
                    != normalized_projection(projection)
                    or stored_topology.get("normalized_endpoint_sha256")
                    != endpoint_sha256(projection)):
                raise BoundaryVerificationError("runner archive topology binding changed")
            decoded_digests[filename] = digest
        if (decoded_digests["continued_live_50050.bin"]
                != decoded_digests["continued_decoded_50050.bin"]):
            raise BoundaryVerificationError("runner persisted restart states differ")
        fingerprint_sha = independent.held_verification_tree_fingerprint(
            arm_dir, arm_dir / "execution.lock", f"engineering runner {arm} tree"
        )
        fingerprint = {
            "schema": "jx-xp2-v4-engineering-arm-tree-fingerprint/v1",
            "entry_count": 5, "sha256": fingerprint_sha,
        }
        if result["arm_tree_fingerprints"].get(arm) != fingerprint:
            raise BoundaryVerificationError("runner arm fingerprint changed")
        records[arm] = record; fingerprints[arm] = fingerprint
    coverage = validate_coverage(contract, records)
    if result.get("required_topology_coverage") != coverage:
        raise BoundaryVerificationError("runner topology coverage binding changed")
    return start, terminal, records, fingerprints


def exact_directory_inventory(
    root: Path, expected: dict[str, str], label: str,
    bound_root: os.stat_result | None = None,
) -> os.stat_result:
    metadata = os.stat(root, follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode):
        raise BoundaryVerificationError(f"{label} root is not a directory")
    if bound_root is not None and (
            metadata.st_dev != bound_root.st_dev or metadata.st_ino != bound_root.st_ino):
        raise BoundaryVerificationError(f"{label} root binding changed")
    names = {entry.name: entry for entry in os.scandir(root)}
    if set(names) != set(expected):
        raise BoundaryVerificationError(f"{label} has extras or omissions")
    for name, kind in expected.items():
        entry = names[name]
        if entry.is_symlink():
            raise BoundaryVerificationError(f"{label} contains a symlink")
        child = os.stat(root / name, follow_symlinks=False)
        if kind == "file":
            if not stat.S_ISREG(child.st_mode) or child.st_nlink != 1:
                raise BoundaryVerificationError(f"{label} file binding is unsafe")
        elif kind == "directory":
            if not stat.S_ISDIR(child.st_mode):
                raise BoundaryVerificationError(f"{label} directory binding is unsafe")
        else:
            raise BoundaryVerificationError("unknown exact inventory entry kind")
    return metadata


def save_and_decode(
    independent: Any, contract: dict[str, Any], simulation: Any, path: Path,
    *, dt_years: float, particle_count: int,
) -> tuple[int, str, str, dict[str, Any], str, dict[str, Any], str]:
    pending = path.with_name(f".{path.name}.pending")
    if path.exists() or path.is_symlink() or pending.exists() or pending.is_symlink():
        raise BoundaryVerificationError("independent continuation output already exists")
    live_endpoint = independent.primary_live_archive_endpoint_projection(simulation)
    live_endpoint_sha = independent.primary_live_archive_endpoint_sha256(simulation)
    if hashlib.sha256(ENDPOINT_DOMAIN + canonical(live_endpoint)).hexdigest() \
            != live_endpoint_sha:
        raise BoundaryVerificationError("independent live endpoint digest domains diverged")
    simulation.save_to_file(str(pending))
    descriptor = os.open(
        pending, os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
    )
    try:
        os.fsync(descriptor); metadata = os.fstat(descriptor)
        on_disk = os.stat(pending, follow_symlinks=False)
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
                or metadata.st_dev != on_disk.st_dev or metadata.st_ino != on_disk.st_ino
                or not 0 < metadata.st_size <= MAX_ARCHIVE_BYTES):
            raise BoundaryVerificationError("independent continuation archive binding changed")
        archive_digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            archive_digest.update(block)
        archive_sha = archive_digest.hexdigest(); os.lseek(descriptor, 0, os.SEEK_SET)
        decoded = independent.verifier_rebound(contract).Simulation(
            f"/proc/self/fd/{descriptor}"
        )
        projection = independent.decoded_primary_continuation_projection(decoded)
        validate_strict_settings(
            independent, decoded, projection, end_years=50_050.0,
            dt_years=dt_years, particle_count=particle_count,
        )
        state_sha = independent.decoded_primary_state_sha256(decoded)
        decoded_endpoint = independent.primary_live_archive_endpoint_projection(decoded)
        decoded_endpoint_sha = independent.primary_live_archive_endpoint_sha256(decoded)
        if (hashlib.sha256(ENDPOINT_DOMAIN + canonical(decoded_endpoint)).hexdigest()
                != decoded_endpoint_sha
                or live_endpoint != decoded_endpoint
                or live_endpoint_sha != decoded_endpoint_sha):
            raise BoundaryVerificationError(
                "independent live pre-save and decoded archive endpoints differ"
            )
        before_rename = os.stat(pending, follow_symlinks=False)
        if (before_rename.st_dev != metadata.st_dev
                or before_rename.st_ino != metadata.st_ino):
            raise BoundaryVerificationError("pending continuation archive was replaced")
        os.replace(pending, path); fsync_dir(path.parent)
        published = os.stat(path, follow_symlinks=False)
        if (published.st_dev != metadata.st_dev or published.st_ino != metadata.st_ino):
            raise BoundaryVerificationError("published continuation archive binding changed")
        return (
            metadata.st_size, archive_sha, state_sha,
            live_endpoint, live_endpoint_sha,
            decoded_endpoint, decoded_endpoint_sha,
        )
    finally:
        os.close(descriptor)


def internal_probe(args: argparse.Namespace) -> int:
    contract, registration, core_digest, held = validate_registration(
        args.contract, args.engineering_registration,
    )
    package_root = Path(os.path.abspath(os.fspath(args.engineering_registration))).parent
    assert_self_authority(
        registration, package_root, held, canonical_required=False,
    )
    final_registration_absent(contract, package_root)
    if lexical(args.initial_states, "initial states", kind="file") \
            != package_root / "initial_states_v1.json":
        raise BoundaryVerificationError("internal initial-state path is not canonical")
    independent = load_independent(package_root, held)
    if contract["engineering_boundary_gate_v1"] \
            != independent.expected_engineering_boundary_gate_v1():
        raise BoundaryVerificationError("internal engineering gate declaration changed")
    gate = contract["engineering_boundary_gate_v1"]
    expected_output = Path(os.path.abspath(os.fspath(package_root / gate["engineering_output_root"])))
    output_root = lexical(args.engineering_output_root, "runner output", kind="dir")
    runner_root_metadata = exact_directory_inventory(
        output_root,
        {
            "execution.lock": "file", "runner_attempt_v1.json": "file",
            "result_v1.json": "file", "runner_terminal_v1.json": "file",
            **{arm: "directory" for arm in ARM_IDS},
        },
        "engineering runner root",
    )
    scratch_root = lexical(
        package_root / gate["engineering_verifier_scratch_root"],
        "verifier scratch", kind="dir",
    )
    arm_dir = output_root / args.arm_id; scratch_arm = scratch_root / args.arm_id
    if output_root != expected_output or args.arm_id not in ARM_IDS:
        raise BoundaryVerificationError("internal verifier target changed")
    expected_receipt = Path(os.path.abspath(os.fspath(
        package_root / gate["engineering_verification_receipt_path"]
    )))
    if Path(os.path.abspath(os.fspath(args.receipt))) != expected_receipt:
        raise BoundaryVerificationError("internal verifier receipt path changed")
    if {path.name for path in scratch_arm.iterdir()} != {"execution.lock"}:
        raise BoundaryVerificationError("internal verifier scratch arm is not fresh")
    paths_and_fds = (
        (package_root / contract["xp2_v2_invalid_replay_lineage"]["v2_b_execution_lock_path"], args.v2_lock_fd, "v2 lock"),
        (package_root / contract["xp2_v3_failed_startup_lineage"]["v3_a_execution_lock_path"], args.v3_lock_fd, "v3 lock"),
        (output_root / "execution.lock", args.runner_root_lock_fd, "runner root lock"),
        (arm_dir / "execution.lock", args.runner_arm_lock_fd, "runner arm lock"),
        (scratch_root / "execution.lock", args.scratch_root_lock_fd, "scratch root lock"),
        (scratch_arm / "execution.lock", args.scratch_arm_lock_fd, "scratch arm lock"),
    )
    for path, descriptor, label in paths_and_fds:
        register_lock(independent, path, descriptor, label)
    independent.validate_v2_replay_lineage(contract, package_root)
    independent.validate_v3_failed_startup_lineage(contract, package_root)
    start_path = package_root / gate["engineering_verifier_start_path"]
    start = read_held_object(start_path)
    for name in ("runner_attempt_v1.json", "result_v1.json", "runner_terminal_v1.json"):
        read_held_object(output_root / name)
    if (set(start) != {
            "schema", "experiment_id", "event", "attempt_index",
            "engineering_registration_sha256", "core_inventory_sha256",
            "runner_start_sha256", "runner_result_sha256", "runner_terminal_sha256",
            "arm_ids", "resume_allowed", "scientific_output_authorized",
        } or start.get("schema") != VERIFIER_ATTEMPT_SCHEMA
            or start.get("experiment_id") != EXPERIMENT_ID
            or start.get("event") != "START" or start.get("attempt_index") != 1
            or start.get("engineering_registration_sha256")
            != held.digest("engineering_registration_v1.json")
            or start.get("core_inventory_sha256") != core_digest
            or start.get("arm_ids") != list(ARM_IDS)
            or start.get("resume_allowed") is not False
            or start.get("scientific_output_authorized") is not False
            or start.get("runner_start_sha256")
            != held_json_digest(output_root / "runner_attempt_v1.json")
            or start.get("runner_result_sha256")
            != held_json_digest(output_root / "result_v1.json")
            or start.get("runner_terminal_sha256")
            != held_json_digest(output_root / "runner_terminal_v1.json")):
        raise BoundaryVerificationError("internal verifier lacks exact durable START")
    record = read_held_object(arm_dir / "arm_result_v1.json")
    expected_n = 133 if args.arm_id == "M0" else 134
    expected_dt = 0.0625 if args.arm_id.startswith("AUDIT-") else 0.125
    simulation, _projection, _digest = safe_archive(
        independent, contract, arm_dir / "boundary_50000.bin",
        record["archives"]["boundary_50000.bin"], end_years=50_000.0,
        dt_years=expected_dt, particle_count=expected_n,
    )
    verifier_completion_absent(contract, package_root)
    simulation.integrate(50_050.0, exact_finish_time=1)
    archive_path = scratch_arm / "independent_50050.bin"
    (size, archive_sha, state_sha, live_endpoint, live_endpoint_sha,
     decoded_endpoint, decoded_endpoint_sha) = save_and_decode(
        independent, contract, simulation, archive_path,
        dt_years=expected_dt, particle_count=expected_n,
    )
    expected_state = record["archives"]["continued_decoded_50050.bin"][2]
    expected_endpoint = record["continued_decoded_topology"]["normalized_endpoint_sha256"]
    if state_sha != expected_state or decoded_endpoint_sha != expected_endpoint:
        raise BoundaryVerificationError("independent 50050 restart parity failed")
    result = {
        "schema": VERIFIER_ARM_SCHEMA, "experiment_id": EXPERIMENT_ID,
        "arm_id": args.arm_id, "dt_years": expected_dt,
        "boundary_years": 50_000.0, "continuation_years": 50_050.0,
        "archive_filename": archive_path.name, "archive_size_bytes": size,
        "archive_sha256": archive_sha, "decoded_state_sha256": state_sha,
        "live_pre_save_normalized_endpoint": live_endpoint,
        "live_pre_save_normalized_endpoint_sha256": live_endpoint_sha,
        "decoded_archive_normalized_endpoint": decoded_endpoint,
        "decoded_archive_normalized_endpoint_sha256": decoded_endpoint_sha,
        "live_pre_save_matches_decoded_archive": True,
        "normalized_endpoint_sha256": decoded_endpoint_sha,
        "matches_runner_restarted_archive": True,
        "scientific_metrics_or_classification_emitted": False,
        "nonpromotable": True,
    }
    atomic_object(scratch_arm / "verification_arm_v1.json", result)
    verifier_completion_absent(contract, package_root)
    held.revalidate(set(contract["result_policy"]["registered_package_inventory"])
                    - {"registration_v1.json"})
    revalidate_held_archives()
    revalidate_held_json()
    close_held_archives()
    close_held_json()
    held.close()
    return 0


def proc_rss(pid: int) -> int:
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
    except (FileNotFoundError, ProcessLookupError):
        return 0
    values: dict[str, int] = {}
    for line in lines:
        if line.startswith(("VmRSS:", "VmHWM:")):
            fields = line.split()
            if len(fields) != 3 or fields[2] != "kB":
                raise BoundaryVerificationError("probe RSS status is malformed")
            values[fields[0][:-1]] = int(fields[1]) * 1024
    if set(values) != {"VmRSS", "VmHWM"}:
        raise BoundaryVerificationError("probe RSS status is incomplete")
    return max(values.values())


def directory_bytes(root: Path) -> int:
    """Account a live tree via retained directory descriptors, never symlinks."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        root_fd = os.open(root, flags)
    except OSError as exc:
        raise BoundaryVerificationError("output byte root is missing or unsafe") from exc
    root_metadata = os.fstat(root_fd)
    if not stat.S_ISDIR(root_metadata.st_mode):
        os.close(root_fd)
        raise BoundaryVerificationError("output byte root is not a directory")

    def scan(directory_fd: int) -> int:
        subtotal = 0
        entries = {entry.name: entry for entry in os.scandir(directory_fd)}
        original: dict[str, tuple[str, int | None, int | None]] = {}
        for name in sorted(entries):
            entry = entries[name]
            hinted_directory = entry.is_dir(follow_symlinks=False)
            hinted_file = entry.is_file(follow_symlinks=False)
            if entry.is_symlink():
                raise BoundaryVerificationError("output byte scan encountered symlink")
            try:
                metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError as exc:
                if hinted_file:
                    original[name] = ("file", None, None)
                    continue
                raise BoundaryVerificationError("output directory disappeared") from exc
            if stat.S_ISREG(metadata.st_mode):
                if not hinted_file or metadata.st_nlink != 1:
                    raise BoundaryVerificationError("output file binding is unsafe")
                original[name] = ("file", metadata.st_dev, metadata.st_ino)
            elif stat.S_ISDIR(metadata.st_mode):
                try:
                    child_fd = os.open(name, flags, dir_fd=directory_fd)
                except OSError as exc:
                    raise BoundaryVerificationError("output directory changed type") from exc
                try:
                    opened = os.fstat(child_fd)
                    if (not hinted_directory or not stat.S_ISDIR(opened.st_mode)
                            or opened.st_dev != metadata.st_dev
                            or opened.st_ino != metadata.st_ino):
                        raise BoundaryVerificationError("output child binding changed")
                    child_total = scan(child_fd)
                    after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    if (not stat.S_ISDIR(after.st_mode) or after.st_dev != opened.st_dev
                            or after.st_ino != opened.st_ino):
                        raise BoundaryVerificationError("output directory was replaced")
                    original[name] = ("directory", opened.st_dev, opened.st_ino)
                    subtotal += child_total
                finally:
                    os.close(child_fd)
            else:
                raise BoundaryVerificationError("output byte scan found special file")
        final_names = set(os.listdir(directory_fd))
        for name, (kind, device, inode) in original.items():
            if name not in final_names:
                if kind == "directory":
                    raise BoundaryVerificationError("output directory disappeared")
                continue
            try:
                final = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                if kind == "file":
                    continue
                raise
            if kind == "directory":
                if (not stat.S_ISDIR(final.st_mode) or final.st_dev != device
                        or final.st_ino != inode):
                    raise BoundaryVerificationError("output directory changed")
            else:
                if not stat.S_ISREG(final.st_mode) or final.st_nlink != 1:
                    raise BoundaryVerificationError("output file final type is unsafe")
                subtotal += final.st_size
        for name in sorted(final_names - set(entries)):
            try:
                added = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if stat.S_ISDIR(added.st_mode):
                raise BoundaryVerificationError("output directory was added during scan")
            if not stat.S_ISREG(added.st_mode) or added.st_nlink != 1:
                raise BoundaryVerificationError("unsafe output entry was added")
            subtotal += added.st_size
        return subtotal

    try:
        total = scan(root_fd)
        final_root = os.stat(root, follow_symlinks=False)
        if (not stat.S_ISDIR(final_root.st_mode)
                or final_root.st_dev != root_metadata.st_dev
                or final_root.st_ino != root_metadata.st_ino):
            raise BoundaryVerificationError("output byte root changed")
        return total
    finally:
        os.close(root_fd)


def supervise_probe(
    independent: Any, contract: dict[str, Any], package_root: Path,
    args: argparse.Namespace, arm: str, fds: dict[str, int], started_ns: int,
) -> dict[str, Any]:
    command = [
        sys.executable, os.fspath(args.held_verifier_path),
        "--contract", os.fspath(args.contract),
        "--initial-states", os.fspath(args.initial_states),
        "--engineering-registration", os.fspath(args.engineering_registration),
        "--engineering-output-root", os.fspath(args.engineering_output_root),
        "--receipt", os.fspath(args.receipt), "--internal-probe", "--arm-id", arm,
        "--v2-lock-fd", str(fds["v2"]), "--v3-lock-fd", str(fds["v3"]),
        "--runner-root-lock-fd", str(fds["runner_root"]),
        "--runner-arm-lock-fd", str(fds[f"runner_{arm}"]),
        "--scratch-root-lock-fd", str(fds["scratch_root"]),
        "--scratch-arm-lock-fd", str(fds[f"scratch_{arm}"]),
    ]
    environment = dict(os.environ)
    environment.update(contract["runtime_lock"]["native_thread_environment"])
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    pass_fds = tuple(dict.fromkeys(fds.values()))
    caps = contract["resource_caps_per_execution"]
    final_registration_absent(contract, package_root)
    parent_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    if ((time.monotonic_ns() - started_ns) / NANOSECONDS_PER_SECOND
            >= float(caps["max_wall_seconds_total"])
            or parent_peak > int(caps["max_peak_rss_bytes_per_process"])
            or directory_bytes(args.receipt.parent) > int(caps["max_output_bytes"])
            or shutil.disk_usage(args.receipt.parent).free
            < int(caps["minimum_free_disk_bytes"])):
        raise BoundaryVerificationError("probe cannot start beyond a resource cap")
    process = subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, env=environment, start_new_session=True,
        close_fds=True, pass_fds=pass_fds,
    )
    arm_started = time.monotonic_ns()
    try:
        while process.poll() is None:
            verifier_completion_absent(contract, package_root)
            elapsed = (time.monotonic_ns() - arm_started) / NANOSECONDS_PER_SECOND
            total = (time.monotonic_ns() - started_ns) / NANOSECONDS_PER_SECOND
            rss = proc_rss(process.pid)
            parent_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
            output_bytes = directory_bytes(args.receipt.parent)
            if (elapsed >= float(caps["max_wall_seconds_per_segment_attempt"])
                    or total >= float(caps["max_wall_seconds_total"])
                    or parent_peak > int(caps["max_peak_rss_bytes_per_process"])
                    or rss > int(caps["max_peak_rss_bytes_per_process"])
                    or output_bytes > int(caps["max_output_bytes"])
                    or shutil.disk_usage(args.receipt.parent).free
                    < int(caps["minimum_free_disk_bytes"])):
                os.killpg(process.pid, signal.SIGKILL); process.wait()
                raise BoundaryVerificationError("independent probe exceeded a registered cap")
            time.sleep(float(caps["watchdog_poll_seconds"]))
        if process.returncode != 0:
            raise BoundaryVerificationError("independent probe failed closed")
    except BaseException:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise
    result = read_held_object(
        args.receipt.parent / "engineering_boundary_verifier_scratch"
        / arm / "verification_arm_v1.json"
    )
    scratch_arm = args.receipt.parent / "engineering_boundary_verifier_scratch" / arm
    expected_n = 133 if arm == "M0" else 134
    expected_dt = 0.0625 if arm.startswith("AUDIT-") else 0.125
    if ({path.name for path in scratch_arm.iterdir()} != {
            "execution.lock", "independent_50050.bin", "verification_arm_v1.json",
        } or set(result) != {
            "schema", "experiment_id", "arm_id", "dt_years", "boundary_years",
            "continuation_years", "archive_filename", "archive_size_bytes",
            "archive_sha256", "decoded_state_sha256",
            "live_pre_save_normalized_endpoint",
            "live_pre_save_normalized_endpoint_sha256",
            "decoded_archive_normalized_endpoint",
            "decoded_archive_normalized_endpoint_sha256",
            "live_pre_save_matches_decoded_archive", "normalized_endpoint_sha256",
            "matches_runner_restarted_archive",
            "scientific_metrics_or_classification_emitted", "nonpromotable",
        } or result.get("schema") != VERIFIER_ARM_SCHEMA
            or result.get("experiment_id") != EXPERIMENT_ID or result.get("arm_id") != arm
            or result.get("dt_years") != expected_dt
            or result.get("boundary_years") != 50_000.0
            or result.get("continuation_years") != 50_050.0
            or result.get("live_pre_save_matches_decoded_archive") is not True
            or result.get("matches_runner_restarted_archive") is not True
            or result.get("scientific_metrics_or_classification_emitted") is not False
            or result.get("nonpromotable") is not True
            or result.get("archive_filename") != "independent_50050.bin"):
        raise BoundaryVerificationError("independent probe result changed")
    _simulation, projection, state_sha = safe_archive(
        independent, contract, scratch_arm / "independent_50050.bin",
        [result["archive_size_bytes"], result["archive_sha256"],
         result["decoded_state_sha256"]], end_years=50_050.0,
        dt_years=expected_dt, particle_count=expected_n,
    )
    decoded_endpoint = normalized_projection(projection)
    decoded_endpoint_sha = endpoint_sha256(projection)
    runner_record = read_held_object(
        args.engineering_output_root / arm / "arm_result_v1.json"
    )
    expected_runner_endpoint = runner_record["continued_decoded_topology"]
    if (state_sha != result["decoded_state_sha256"]
            or decoded_endpoint != result["decoded_archive_normalized_endpoint"]
            or result["live_pre_save_normalized_endpoint"] != decoded_endpoint
            or decoded_endpoint_sha != result["normalized_endpoint_sha256"]
            or decoded_endpoint_sha
            != result["decoded_archive_normalized_endpoint_sha256"]
            or decoded_endpoint_sha
            != result["live_pre_save_normalized_endpoint_sha256"]
            or decoded_endpoint
            != expected_runner_endpoint["normalized_endpoint_projection"]
            or decoded_endpoint_sha
            != expected_runner_endpoint["normalized_endpoint_sha256"]):
        raise BoundaryVerificationError("independent probe archive binding changed")
    return result


def execute(args: argparse.Namespace) -> int:
    contract, registration, core_digest, held = validate_registration(
        args.contract, args.engineering_registration,
    )
    package_root = Path(os.path.abspath(os.fspath(args.engineering_registration))).parent
    assert_self_authority(
        registration, package_root, held, canonical_required=True,
    )
    held_verifier_dir = tempfile.TemporaryDirectory(prefix="jx-xp2-v4-held-verifier-")
    args.held_verifier_path = materialize_held(
        held, "verify_engineering_boundary.py", Path(held_verifier_dir.name)
    )
    gate = contract["engineering_boundary_gate_v1"]
    final_registration_absent(contract, package_root)
    if lexical(args.initial_states, "initial states", kind="file") \
            != package_root / "initial_states_v1.json":
        raise BoundaryVerificationError("initial-state path is not canonical")
    independent = load_independent(package_root, held)
    if contract.get("engineering_boundary_gate_v1") \
            != independent.expected_engineering_boundary_gate_v1():
        raise BoundaryVerificationError("engineering gate declaration changed")
    if args.engineering_output_root.resolve() != (
        package_root / gate["engineering_output_root"]
    ).resolve():
        raise BoundaryVerificationError("engineering runner output path changed")
    output_root = lexical(args.engineering_output_root, "runner output", kind="dir")
    expected_receipt = Path(os.path.abspath(os.fspath(
        package_root / gate["engineering_verification_receipt_path"]
    )))
    if Path(os.path.abspath(os.fspath(args.receipt))) != expected_receipt:
        raise BoundaryVerificationError("engineering receipt path changed")
    verification_root = expected_receipt.parent
    lexical(verification_root.parent, "verification parent", kind="dir")
    if not verification_root.exists():
        verification_root.mkdir(); fsync_dir(verification_root.parent)
    lexical(verification_root, "verification root", kind="dir")
    if any(verification_root.iterdir()):
        raise BoundaryVerificationError(
            "engineering verification root is not fresh and empty"
        )
    verification_root_metadata = os.stat(verification_root, follow_symlinks=False)
    verification_root_fd = os.open(
        verification_root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0),
    )
    start_path = package_root / gate["engineering_verifier_start_path"]
    terminal_path = package_root / gate["engineering_verifier_terminal_path"]
    scratch_root = package_root / gate["engineering_verifier_scratch_root"]
    for path in (start_path, terminal_path, expected_receipt, scratch_root):
        if path.exists() or path.is_symlink():
            raise BoundaryVerificationError("engineering verifier attempt is not fresh")
    independent.verify_unlocked_execution_lock(
        package_root / contract["xp2_v2_invalid_replay_lineage"]["v2_b_execution_lock_path"],
        "XP2-v2 B lineage lock",
    )
    independent.verify_unlocked_execution_lock(
        package_root / contract["xp2_v3_failed_startup_lineage"]["v3_a_execution_lock_path"],
        "XP2-v3 A lineage lock",
    )
    independent.validate_v2_replay_lineage(contract, package_root)
    independent.validate_v3_failed_startup_lineage(contract, package_root)
    independent.verify_unlocked_execution_lock(
        output_root / "execution.lock", "engineering runner root lock"
    )
    registration_sha = held.digest("engineering_registration_v1.json")
    runner_start, runner_terminal, records, runner_fingerprints = validate_runner(
        independent, contract, registration_sha, core_digest, output_root,
    )
    runner_root_fingerprint = {
        "entry_count": 22,
        "sha256": independent.held_verification_tree_fingerprint(
            output_root, output_root / "execution.lock", "engineering runner root tree",
        ),
    }
    start = {
        "schema": VERIFIER_ATTEMPT_SCHEMA, "experiment_id": EXPERIMENT_ID,
        "event": "START", "attempt_index": 1,
        "engineering_registration_sha256": registration_sha,
        "core_inventory_sha256": core_digest,
        "runner_start_sha256": held_json_digest(output_root / "runner_attempt_v1.json"),
        "runner_result_sha256": held_json_digest(output_root / "result_v1.json"),
        "runner_terminal_sha256": held_json_digest(output_root / "runner_terminal_v1.json"),
        "arm_ids": list(ARM_IDS), "resume_allowed": False,
        "scientific_output_authorized": False,
    }
    atomic_object_at(verification_root_fd, start_path.name, start)
    if read_held_object_at(verification_root_fd, start_path.name) != start:
        raise BoundaryVerificationError("verifier START publication changed")
    scratch_root.mkdir(); fsync_dir(scratch_root.parent)
    scratch_root_fd = acquire_lock(scratch_root / "execution.lock", create=True)
    register_lock(independent, scratch_root / "execution.lock", scratch_root_fd, "scratch root")
    fds = {
        "v2": lock_fd(independent, package_root / contract["xp2_v2_invalid_replay_lineage"]["v2_b_execution_lock_path"]),
        "v3": lock_fd(independent, package_root / contract["xp2_v3_failed_startup_lineage"]["v3_a_execution_lock_path"]),
        "runner_root": lock_fd(independent, output_root / "execution.lock"),
        "scratch_root": scratch_root_fd,
    }
    for arm in ARM_IDS:
        fds[f"runner_{arm}"] = lock_fd(independent, output_root / arm / "execution.lock")
    scratch_arm_fds: list[int] = []
    scratch_fingerprints: dict[str, Any] = {}
    probe_results: dict[str, Any] = {}
    held_evidence_trees: list[HeldEvidenceTree] = []
    started_ns = time.monotonic_ns()
    try:
        for arm in ARM_IDS:
            final_registration_absent(contract, package_root)
            arm_dir = scratch_root / arm; arm_dir.mkdir(); fsync_dir(scratch_root)
            descriptor = acquire_lock(arm_dir / "execution.lock", create=True)
            scratch_arm_fds.append(descriptor); fds[f"scratch_{arm}"] = descriptor
            register_lock(independent, arm_dir / "execution.lock", descriptor, f"scratch {arm}")
            probe_results[arm] = supervise_probe(
                independent, contract, package_root, args, arm, fds, started_ns,
            )
            sha = independent.held_verification_tree_fingerprint(
                arm_dir, arm_dir / "execution.lock", f"verifier scratch {arm} tree"
            )
            scratch_fingerprints[arm] = {"entry_count": 3, "sha256": sha}
        scratch_sha = independent.held_verification_tree_fingerprint(
            scratch_root, scratch_root / "execution.lock", "verifier scratch tree"
        )
        scratch_fingerprint = {"entry_count": 13, "sha256": scratch_sha}
        # Revalidate every runner arm after the registered continuation probes.
        for arm in ARM_IDS:
            sha = independent.held_verification_tree_fingerprint(
                output_root / arm, output_root / arm / "execution.lock",
                f"engineering runner {arm} final tree",
            )
            if sha != runner_fingerprints[arm]["sha256"]:
                raise BoundaryVerificationError("runner evidence changed during verification")
        exact_directory_inventory(
            output_root,
            {
                "execution.lock": "file", "runner_attempt_v1.json": "file",
                "result_v1.json": "file", "runner_terminal_v1.json": "file",
                **{arm: "directory" for arm in ARM_IDS},
            },
            "engineering runner root", runner_root_metadata,
        )
        if independent.held_verification_tree_fingerprint(
                output_root, output_root / "execution.lock",
                "engineering runner root final tree",
        ) != runner_root_fingerprint["sha256"]:
            raise BoundaryVerificationError("engineering runner root tree changed")
        exact_directory_inventory(
            verification_root,
            {start_path.name: "file", scratch_root.name: "directory"},
            "engineering verification root", verification_root_metadata,
        )
        expected_names = set(contract["result_policy"]["registered_package_inventory"]) \
            - {"registration_v1.json"}
        held.revalidate(expected_names); final_registration_absent(contract, package_root)
        revalidate_held_archives()
        revalidate_held_json()
        for arm in ARM_IDS:
            final_arm_sha = independent.held_verification_tree_fingerprint(
                scratch_root / arm, scratch_root / arm / "execution.lock",
                f"verifier scratch {arm} final tree",
            )
            if final_arm_sha != scratch_fingerprints[arm]["sha256"]:
                raise BoundaryVerificationError("verifier scratch arm changed")
        final_scratch_sha = independent.held_verification_tree_fingerprint(
            scratch_root, scratch_root / "execution.lock", "verifier scratch final tree"
        )
        if final_scratch_sha != scratch_fingerprint["sha256"]:
            raise BoundaryVerificationError("verifier scratch tree changed")
        runner_tree = HeldEvidenceTree(
            output_root, fds["runner_root"], "retained engineering runner root",
        )
        scratch_tree = HeldEvidenceTree(
            scratch_root, fds["scratch_root"], "retained verifier scratch root",
        )
        held_evidence_trees.extend((runner_tree, scratch_tree))
        if (runner_tree.entry_count != runner_root_fingerprint["entry_count"]
                or runner_tree.sha256 != runner_root_fingerprint["sha256"]
                or scratch_tree.entry_count != scratch_fingerprint["entry_count"]
                or scratch_tree.sha256 != scratch_fingerprint["sha256"]):
            raise BoundaryVerificationError("retained engineering evidence fingerprint changed")
        runner_tree.revalidate(); scratch_tree.revalidate()
        receipt = {
            "schema": RECEIPT_SCHEMA, "experiment_id": EXPERIMENT_ID,
            "status": "PASS", "artifact_class": "INDEPENDENT_NONSCIENTIFIC_ENGINEERING_BOUNDARY_VERIFICATION",
            "engineering_registration_sha256": registration_sha,
            "core_inventory_sha256": core_digest,
            "runner_start_sha256": start["runner_start_sha256"],
            "runner_result_sha256": start["runner_result_sha256"],
            "runner_terminal_sha256": start["runner_terminal_sha256"],
            "runner_root_tree_fingerprint": runner_root_fingerprint,
            "runner_arm_tree_fingerprints": runner_fingerprints,
            "verifier_start_sha256": held_json_digest_at(
                verification_root_fd, start_path.name,
            ),
            "verifier_scratch_tree_fingerprint": scratch_fingerprint,
            "verifier_arm_tree_fingerprints": scratch_fingerprints,
            "verifier_arm_results": probe_results,
            "checks": {
                "registered_core_and_engineering_scope": True,
                "runner_START_result_PASS_bijection": True,
                "all_raw_archives_redecoded_and_strict": True,
                "required_allocator_cache_topology_coverage": True,
                "unsaved_control_vs_restarted_endpoint_parity": True,
                "three_independent_50050_restart_probes": True,
                "no_scientific_metrics_gates_labels_or_classification": True,
            },
            "scientific_outcomes_gates_labels_or_classification": None,
            "nonpromotable": True,
            "final_registration_reference": None,
            "authorizes_final_registration_only_with_exact_verifier_PASS_terminal": True,
        }
        receipt_payload = serialized(receipt)
        caps = contract["resource_caps_per_execution"]
        terminal = {
            "schema": VERIFIER_ATTEMPT_SCHEMA, "experiment_id": EXPERIMENT_ID,
            "event": "PASS", "attempt_index": 1,
            "start_sha256": held_json_digest_at(verification_root_fd, start_path.name),
            "engineering_registration_sha256": registration_sha,
            "receipt_filename": expected_receipt.name,
            "receipt_size_bytes": len(receipt_payload),
            "receipt_sha256": hashlib.sha256(receipt_payload).hexdigest(),
            "verifier_scratch_tree_fingerprint": scratch_fingerprint,
            "scientific_output_emitted": False,
        }
        projected = len(receipt_payload) + len(serialized(terminal))
        elapsed = (time.monotonic_ns() - started_ns) / NANOSECONDS_PER_SECOND
        parent_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        if (elapsed >= float(caps["max_wall_seconds_total"])
                or parent_peak > int(caps["max_peak_rss_bytes_per_process"])
                or directory_bytes(verification_root) + projected > int(caps["max_output_bytes"])
                or shutil.disk_usage(verification_root).free
                < int(caps["minimum_free_disk_bytes"]) + projected):
            raise BoundaryVerificationError("verification publication exceeds resource cap")
        atomic_object_at(verification_root_fd, expected_receipt.name, receipt)
        if (read_held_object_at(verification_root_fd, expected_receipt.name) != receipt
                or held_json_digest_at(verification_root_fd, expected_receipt.name)
                != terminal["receipt_sha256"]
                or os.stat(
                    expected_receipt.name, dir_fd=verification_root_fd,
                    follow_symlinks=False,
                ).st_size != terminal["receipt_size_bytes"]):
            raise BoundaryVerificationError("verification receipt publication changed")
        runner_tree.revalidate(); scratch_tree.revalidate()
        held.revalidate(expected_names); final_registration_absent(contract, package_root)
        revalidate_held_archives(); revalidate_held_json()
        exact_directory_inventory(
            output_root,
            {
                "execution.lock": "file", "runner_attempt_v1.json": "file",
                "result_v1.json": "file", "runner_terminal_v1.json": "file",
                **{arm: "directory" for arm in ARM_IDS},
            },
            "engineering runner root before verifier PASS", runner_root_metadata,
        )
        if independent.held_verification_tree_fingerprint(
                output_root, output_root / "execution.lock",
                "engineering runner root before verifier PASS tree",
        ) != runner_root_fingerprint["sha256"]:
            raise BoundaryVerificationError("runner root changed before verifier PASS")
        for arm in ARM_IDS:
            if independent.held_verification_tree_fingerprint(
                    scratch_root / arm, scratch_root / arm / "execution.lock",
                    f"verifier scratch {arm} before PASS tree",
            ) != scratch_fingerprints[arm]["sha256"]:
                raise BoundaryVerificationError("scratch arm changed before verifier PASS")
        if independent.held_verification_tree_fingerprint(
                scratch_root, scratch_root / "execution.lock",
                "verifier scratch before PASS tree",
        ) != scratch_fingerprint["sha256"]:
            raise BoundaryVerificationError("scratch root changed before verifier PASS")
        exact_directory_inventory(
            verification_root,
            {
                start_path.name: "file", scratch_root.name: "directory",
                expected_receipt.name: "file",
            },
            "engineering verification root before PASS", verification_root_metadata,
        )
        elapsed = (time.monotonic_ns() - started_ns) / NANOSECONDS_PER_SECOND
        parent_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        if (elapsed >= float(caps["max_wall_seconds_total"])
                or parent_peak > int(caps["max_peak_rss_bytes_per_process"])
                or directory_bytes(verification_root) + len(serialized(terminal))
                > int(caps["max_output_bytes"])
                or shutil.disk_usage(verification_root).free
                < int(caps["minimum_free_disk_bytes"]) + len(serialized(terminal))):
            raise BoundaryVerificationError("verifier PASS exceeds a registered cap")
        final_registration_absent(contract, package_root)
        revalidate_held_archives(); revalidate_held_json()
        exact_directory_inventory_at(
            verification_root_fd,
            {
                start_path.name: "file", scratch_root.name: "directory",
                expected_receipt.name: "file",
            },
            "retained engineering verification root before PASS",
        )
        atomic_object_at(verification_root_fd, terminal_path.name, terminal)
        exact_directory_inventory_at(
            verification_root_fd,
            {
                start_path.name: "file", scratch_root.name: "directory",
                expected_receipt.name: "file", terminal_path.name: "file",
            },
            "retained completed engineering verification root",
        )
        if read_held_object_at(verification_root_fd, terminal_path.name) != terminal:
            raise BoundaryVerificationError("verifier PASS publication changed")
        revalidate_held_archives(); revalidate_held_json()
        runner_tree.revalidate(); scratch_tree.revalidate()
    finally:
        for descriptor in set(independent._VERIFICATION_LOCK_FDS.values()):
            try:
                os.close(descriptor)
            except OSError:
                pass
        independent._VERIFICATION_LOCK_FDS.clear()
        for tree in reversed(held_evidence_trees):
            try:
                tree.close()
            except OSError:
                pass
        close_held_archives()
        close_held_json()
        try:
            os.close(verification_root_fd)
        except OSError:
            pass
        held.close()
        held_verifier_dir.cleanup()
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--contract", type=Path, required=True)
    result.add_argument("--initial-states", type=Path, required=True)
    result.add_argument("--engineering-registration", type=Path, required=True)
    result.add_argument("--engineering-output-root", type=Path, required=True)
    result.add_argument("--receipt", type=Path, required=True)
    result.add_argument("--internal-probe", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--arm-id", choices=ARM_IDS, help=argparse.SUPPRESS)
    for name in (
        "v2-lock-fd", "v3-lock-fd", "runner-root-lock-fd", "runner-arm-lock-fd",
        "scratch-root-lock-fd", "scratch-arm-lock-fd",
    ):
        result.add_argument(f"--{name}", type=int, help=argparse.SUPPRESS)
    return result


def main() -> int:
    args = parser().parse_args()
    hidden = (
        args.arm_id, args.v2_lock_fd, args.v3_lock_fd,
        args.runner_root_lock_fd, args.runner_arm_lock_fd,
        args.scratch_root_lock_fd, args.scratch_arm_lock_fd,
    )
    if args.internal_probe:
        if any(value is None for value in hidden):
            raise BoundaryVerificationError("internal verifier arguments incomplete")
        return internal_probe(args)
    if any(value is not None for value in hidden):
        raise BoundaryVerificationError("external verifier received internal arguments")
    return execute(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"engineering verification failed closed: {type(error).__name__}", file=sys.stderr)
        raise SystemExit(2)
