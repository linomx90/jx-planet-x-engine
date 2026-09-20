"""Opt-in execution of one closed, replayed CSPICE SPKGEO profile.

The public module imports no astronomy provider.  Each call validates the
already-sealed M1--M4C2A contracts, observes the exact seven local artifacts,
creates two separate private wheel extractions, and runs the private worker
twice under the current supported CPython ABI.  The worker receives a held
SPK file descriptor through Linux ``/proc/self/fd`` and returns only bounded
primitive measurements.  This parent reconstructs and validates every frozen
evidence record.

The held interpreter descriptor supplies the executed bytes.  Its real
``sys.executable`` spelling remains ``argv[0]`` solely for CPython's
relocatable standard-library prefix discovery; this milestone does not retain
or claim standard-library path custody against same-UID substitution.

The resulting receipt is unauthenticated empirical corroboration.  It proves
neither process identity nor process independence and is not a network
sandbox, mapped-page or continuous byte
custody, an instrumented SPKGEO segment trace, exhaustive silent-I/O-fault
exclusion, physical-accuracy qualification, or legal/redistribution authority.
Point observations and bounded byte/call counts do not bound latency of
hostile, remote, FUSE, or otherwise blocking filesystems.  Private 0700 stages
do not defend against the same UID, root, or hostile procfs.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import os
import selectors
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time as _monotonic_time
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import TracebackType
from typing import BinaryIO

from . import cspice_execution_contracts as _evidence
from .artifacts import (
    LocalArtifactVerificationReceipt,
    verify_local_artifact_bytes,
)
from .contracts import (
    ArtifactBinding,
    SolarSystemContractError,
    SolarSystemCoverageError,
    SolarSystemDataError,
    SolarSystemDependencyUnavailableError,
    validate_integrity,
)
from .cspice_execution_contracts import (
    CspiceRuntimeObservation,
    CspiceSelectedSegmentRecord,
    CspiceSpkgeoExecutionLane,
    CspiceSpkgeoExecutionReceipt,
    CspiceTargetChainEvidence,
)
from .cspice_time import (
    CspiceBinary64QueryProjectionReceipt,
    validate_cspice_binary64_query_projection_receipt,
)
from .states import ProviderNativeStateBatch
from .units import KILOMETRE, SECOND


_CONTENT_INTEGRITY_CLASS = "UNAUTHENTICATED_CONTENT_INTEGRITY_ONLY"
_REQUEST_SCHEMA = "jxplanetx.cspice-worker.request.v1"
_RESULT_SCHEMA = "jxplanetx.cspice-worker.result.v1"
_REQUEST_TOKEN = "ONE_HELD_STAGE_ONE_HELD_SPK_DIRECT_SPKGEO_J2000_V1"
_MAXIMUM_REQUEST_BYTES = 65_536
_MAXIMUM_RESULT_BYTES = 2_097_152
_MAXIMUM_STDERR_BYTES = 65_536
_MAXIMUM_PROTOCOL_DEPTH = 16
_MAXIMUM_PROTOCOL_NODES = 32_768
_MAXIMUM_TOTAL_LEGS = 256
_WORKER_TIMEOUT_SECONDS = 45.0
_WORKER_TERM_GRACE_SECONDS = 1.0
_READ_CHUNK = 1 << 20
_MAXIMUM_ZIP_ENTRIES = 2_048
_MAXIMUM_ZIP_MEMBER_BYTES = 1 << 27
_MAXIMUM_ZIP_EXPANDED_BYTES = 1 << 27
_MAXIMUM_ZIP_COMPRESSION_RATIO = 1_000
_MAXIMUM_ZIP_PATH_COMPONENTS = 64
_MAXIMUM_ZIP_PATH_CODEPOINTS = 1_024
_MAXIMUM_ZIP_COMPONENT_CODEPOINTS = 256
_MAXIMUM_ROOT_FD = (1 << 31) - 1
_MAXIMUM_HELD_FILE_BYTES = 1 << 30
_EXPECTED_WORKER_BYTE_LENGTH = 40_504
_EXPECTED_WORKER_SHA256 = "09824658b6acb407ea683ff6d9c24df07bff3684549f5b478844f24c7d6e0951"
_SPK_ID = "artifact.cspice.de440s"
_COMPOSITE_ID = "artifact.cspice.license"
_RULES_ID = "artifact.cspice.naif-rules-license"
_NATIVE_ID = "artifact.cspice.native"
_NUMPY_LICENSE_ID = "artifact.cspice.numpy-license"
_NUMPY_ID = "artifact.cspice.numpy-wheel"
_SPICEYPY_ID = "artifact.cspice.spiceypy-wheel"
_EXPECTED_ARTIFACT_IDS = (
    _SPK_ID,
    _COMPOSITE_ID,
    _RULES_ID,
    _NATIVE_ID,
    _NUMPY_LICENSE_ID,
    _NUMPY_ID,
    _SPICEYPY_ID,
)
_SPK_LENGTH = 32_726_016
_SPK_SHA256 = "c1c7feeab882263fc493a9d5a5b2ddd71b54826cdf65d8d17a76126b260a49f2"
_NATIVE_LENGTH = 3_561_056
_NATIVE_SHA256 = "1d9273fc9afce5201e904569a9439a0b010d2461dfc88c0de549d2a1b5ecdf84"
_COMPOSITE_LENGTH = 42_166
_COMPOSITE_SHA256 = "285cdb7a837de375dbaecf7578949449262bc5ecb1625e8f4972e5dd82233a9b"
_RULES_LENGTH = 23_487
_RULES_SHA256 = "ae85f851646e7c4f0a762db852907bc090a2ab50c815eb2a6cd8639e96b7e047"
_NUMPY_LICENSE_LENGTH = 47_768
_NUMPY_LICENSE_SHA256 = "2046a3130e50b11c01659b3a0d963e6ae0b7436ff8e89cbcfd9e87bc6112d595"
_MIT_LENGTH = 1_088
_MIT_SHA256 = "77f7749eb7c6acfbaefed8c60c7a84820f905328d83ec4c5454c593571b9fc1f"
_NOTICE_LENGTH = 4_230
_NOTICE_SHA256 = "0e22deb48f71267138cb2a965a7eb613381ad6954f6948ca8e58cd4659e48b27"
_PROVIDER_SEALS = {
    "cp312-cp312": "0c7feb4f60a9bef084242a937bfdd64e6837e9e79182fa6e3a661c043d2a7d38",
    "cp314-cp314": "2a560649817b498d98172a3e6da9a63be0363a773ee6dae77f98b3d4d590d09f",
}
_ARTIFACT_SEALS = {
    "cp312-cp312": {
        _SPK_ID: "277963896fdb435de8271940b264b8fdc238e3e6aaeb213fc53e019bc34e6b35",
        _COMPOSITE_ID: "0f9a751a968cac5f01c4bc7c1db5f58fa9da76a77e359a8cfe4415534a646716",
        _RULES_ID: "1a78506ed326c16c71ee5c9fd9196422fe008dcb7c4e709589f865a82da253ff",
        _NATIVE_ID: "4e582c17e6004ced2a79ff5b32f36118f98b3e66eb36f69b9746b321ab1ac240",
        _NUMPY_LICENSE_ID: "4ca69e5b37603b5af6096d5e8905f5f60670515702886a61aadceb8f041646a0",
        _NUMPY_ID: "ed09186d187737bb3a2a61a3eba1953a77b4ced4962548b69f98c55dc6130685",
        _SPICEYPY_ID: "2f16cf35c0faefbcbfae88f9fa0b3242a56c5344049821c4d5787aab9a84e135",
    },
    "cp314-cp314": {
        _SPK_ID: "277963896fdb435de8271940b264b8fdc238e3e6aaeb213fc53e019bc34e6b35",
        _COMPOSITE_ID: "0f9a751a968cac5f01c4bc7c1db5f58fa9da76a77e359a8cfe4415534a646716",
        _RULES_ID: "1a78506ed326c16c71ee5c9fd9196422fe008dcb7c4e709589f865a82da253ff",
        _NATIVE_ID: "4e582c17e6004ced2a79ff5b32f36118f98b3e66eb36f69b9746b321ab1ac240",
        _NUMPY_LICENSE_ID: "4ca69e5b37603b5af6096d5e8905f5f60670515702886a61aadceb8f041646a0",
        _NUMPY_ID: "a6d4130cdaf1e137ff551e57526faf62c33dbdc5cef588ab4c1f54e9506befab",
        _SPICEYPY_ID: "abacd5cacfff5e9c65779bd72a0b4681fbf6427b33e0059e5d39fa4898855978",
    },
}


def _identifier(value: object, label: str) -> str:
    if type(value) is not str:
        raise SolarSystemContractError(f"{label} must be an exact string")
    if len(value) > 256:
        raise SolarSystemContractError(f"{label} exceeds its code-point cap")
    if not value or value.strip() != value or any(ord(character) < 0x20 or ord(character) > 0x7E for character in value):
        raise SolarSystemContractError(f"{label} must be nonempty trimmed printable ASCII")
    return value


def _sha256_text(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SolarSystemContractError(f"{label} must be a lowercase SHA-256 digest")
    if value == "0" * 64:
        raise SolarSystemContractError(f"{label} cannot be all zero")
    return value


def _exact_worker_sha256(value: object, label: str) -> str:
    """Validate an untrusted worker digest without leaking ContractError."""

    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or value == "0" * 64
    ):
        raise SolarSystemDataError(f"worker {label} is not a nonzero lowercase SHA-256 digest")
    return value


def _exact_fd(value: object, label: str) -> int:
    if type(value) is not int or value < 0 or value > _MAXIMUM_ROOT_FD:
        raise SolarSystemContractError(f"{label} must be an exact in-range descriptor integer")
    return value


def _current_abi() -> str:
    if sys.implementation.name != "cpython":
        raise SolarSystemDependencyUnavailableError("M4C2B requires exact supported CPython")
    pair = sys.version_info[:3]
    if pair == (3, 12, 13):
        return "cp312-cp312"
    if pair == (3, 14, 4):
        return "cp314-cp314"
    raise SolarSystemDependencyUnavailableError("current CPython version is outside the calibrated ABI roster")


def _require_platform() -> None:
    flags = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
    if sys.platform != "linux" or any(not hasattr(os, name) for name in flags):
        raise SolarSystemDependencyUnavailableError("M4C2B requires the exact Linux openat/procfd profile")
    if os.open not in os.supports_dir_fd or os.stat not in os.supports_dir_fd:
        raise SolarSystemDependencyUnavailableError("required dir-fd operations are unavailable")
    if not os.path.isdir("/proc/self/fd"):
        raise SolarSystemDependencyUnavailableError("Linux procfd is unavailable")


def _require_exact_provider_roster(
    projection: CspiceBinary64QueryProjectionReceipt,
    abi: str,
) -> dict[str, ArtifactBinding]:
    _evidence._require_cspice_provider_profile(projection)
    provider = projection.effective_query.provider
    if provider.content_sha256 != _PROVIDER_SEALS[abi]:
        raise SolarSystemContractError("provider seal is outside the exact production seven-artifact profile")
    if type(provider.artifacts) is not tuple or tuple(item.artifact_id for item in provider.artifacts) != _EXPECTED_ARTIFACT_IDS:
        raise SolarSystemContractError("provider artifact roster must be the exact sorted seven-member profile")
    by_id = {item.artifact_id: item for item in provider.artifacts}
    for artifact_id, expected_seal in _ARTIFACT_SEALS[abi].items():
        artifact = by_id[artifact_id]
        validate_integrity(artifact)
        if artifact.content_sha256 != expected_seal:
            raise SolarSystemContractError(f"{artifact_id} differs from its exact production binding")
    if provider.identity.ordered_implementation_artifact_ids != (_SPICEYPY_ID, _NUMPY_ID, _NATIVE_ID):
        raise SolarSystemContractError("implementation artifact order differs from the closed profile")
    if provider.ordered_load_artifact_ids != (_SPK_ID,):
        raise SolarSystemContractError("load roster differs from the one-direct-SPK profile")
    if (
        by_id[_SPICEYPY_ID].license_artifact_id != _COMPOSITE_ID
        or by_id[_NATIVE_ID].license_artifact_id != _RULES_ID
        or by_id[_SPK_ID].license_artifact_id != _RULES_ID
        or by_id[_NUMPY_ID].license_artifact_id != _NUMPY_LICENSE_ID
    ):
        raise SolarSystemContractError("provider license/provenance backedges differ from the closed profile")
    return by_id


def _components(relative: str) -> tuple[str, ...]:
    path = PurePosixPath(relative)
    parts = path.parts
    if not parts or path.is_absolute() or path.as_posix() != relative or len(parts) > 64:
        raise SolarSystemContractError("artifact locator is outside the normalized relative profile")
    if any(part in ("", ".", "..") or len(part) > 256 for part in parts):
        raise SolarSystemContractError("artifact locator component is outside its cap")
    return parts


def _walk_directory(root_fd: int, parts: tuple[str, ...]) -> int:
    current = os.dup(root_fd)
    try:
        for part in parts:
            following = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=current)
            os.close(current)
            current = following
        return current
    except Exception:
        os.close(current)
        raise


def _open_relative(root_fd: int, relative: str) -> int:
    parts = _components(relative)
    parent = _walk_directory(root_fd, parts[:-1])
    try:
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=parent)
    finally:
        os.close(parent)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise SolarSystemDataError("secure artifact open did not yield a regular file")
    except Exception:
        os.close(fd)
        raise
    return fd


def _stat_key(value: os.stat_result) -> tuple[int, ...]:
    names = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_uid", "st_gid", "st_size", "st_mtime_ns", "st_ctime_ns")
    result = tuple(getattr(value, name) for name in names)
    if any(type(item) is not int for item in result):
        raise SolarSystemDataError("filesystem stat observation is not exact-integer data")
    return result


def _stable_directory_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    result = (value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid)
    if any(type(item) is not int for item in result) or not stat.S_ISDIR(value.st_mode):
        raise SolarSystemDataError("stage directory identity is invalid")
    return result


def _hash_fd(fd: int, expected_length: int | None = None) -> tuple[int, str, tuple[int, ...]]:
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode) or before.st_size < 0 or before.st_size > _MAXIMUM_HELD_FILE_BYTES:
        raise SolarSystemDataError("held file is not a bounded regular file")
    if expected_length is not None and before.st_size != expected_length:
        raise SolarSystemDataError("held file length differs from its exact declaration")
    digest = hashlib.sha256()
    offset = 0
    while offset < before.st_size:
        block = os.pread(fd, min(_READ_CHUNK, before.st_size - offset), offset)
        if not block:
            raise SolarSystemDataError("held file returned an early EOF")
        digest.update(block)
        offset += len(block)
    if os.pread(fd, 1, before.st_size) != b"":
        raise SolarSystemDataError("held file grew during bounded observation")
    after = os.fstat(fd)
    if _stat_key(before) != _stat_key(after):
        raise SolarSystemDataError("held file metadata changed during observation")
    return before.st_size, digest.hexdigest(), _stat_key(before)


def _read_fd(fd: int, expected_length: int) -> bytes:
    if expected_length > _MAXIMUM_ZIP_EXPANDED_BYTES:
        raise SolarSystemDataError("requested held-file read exceeds its cap")
    result = bytearray()
    while len(result) < expected_length:
        block = os.pread(fd, min(_READ_CHUNK, expected_length - len(result)), len(result))
        if not block:
            raise SolarSystemDataError("held file returned an early EOF")
        result.extend(block)
    if os.pread(fd, 1, expected_length) != b"":
        raise SolarSystemDataError("held file has trailing bytes")
    return bytes(result)


class _Stage:
    """A private tracked stage removed without recursive pathname deletion."""

    def __init__(self) -> None:
        self.tmp_fd = -1
        self.root_fd = -1
        self.name = ""
        self.files: set[str] = set()
        self.directories: set[str] = set()
        try:
            self.tmp_fd = os.open("/tmp", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
            path = tempfile.mkdtemp(prefix="jx-cspice-m4c2b-", dir="/tmp")
            self.name = os.path.basename(path)
            os.chmod(path, 0o700)
            self.root_fd = os.open(self.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=self.tmp_fd)
            self.root_identity = _stable_directory_identity(os.fstat(self.root_fd))
        except Exception:
            if self.root_fd >= 0:
                os.close(self.root_fd)
            if self.tmp_fd >= 0 and self.name:
                try:
                    os.rmdir(self.name, dir_fd=self.tmp_fd)
                except OSError:
                    pass
            if self.tmp_fd >= 0:
                os.close(self.tmp_fd)
            raise

    def __enter__(self) -> _Stage:
        return self

    def _ensure_parent(self, relative: str) -> tuple[int, str]:
        parts = _components(relative)
        current = os.dup(self.root_fd)
        accumulated: list[str] = []
        try:
            for part in parts[:-1]:
                accumulated.append(part)
                joined = "/".join(accumulated)
                try:
                    os.mkdir(part, 0o700, dir_fd=current)
                    self.directories.add(joined)
                except FileExistsError:
                    pass
                following = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=current)
                info = os.fstat(following)
                if stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != os.geteuid():
                    os.close(following)
                    raise SolarSystemDataError("stage directory is not private and euid-owned")
                os.close(current)
                current = following
            return current, parts[-1]
        except Exception:
            os.close(current)
            raise

    def create_file(self, relative: str) -> int:
        if relative in self.files:
            raise SolarSystemDataError("stage file would be overwritten")
        parent, leaf = self._ensure_parent(relative)
        try:
            fd = os.open(leaf, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=parent)
        finally:
            os.close(parent)
        self.files.add(relative)
        return fd

    def open_file(self, relative: str) -> int:
        return _open_relative(self.root_fd, relative)

    def create_runtime_directory(self) -> int:
        try:
            os.mkdir("runtime", 0o700, dir_fd=self.root_fd)
            self.directories.add("runtime")
        except FileExistsError as exc:
            raise SolarSystemDataError("runtime stage directory already exists") from exc
        return os.open("runtime", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=self.root_fd)

    def replace_with_hardlink(self, source: str, target: str) -> None:
        source_fd = self.open_file(source)
        try:
            target_fd = self.open_file(target)
            try:
                source_info = os.fstat(source_fd)
                target_info = os.fstat(target_fd)
                if source_info.st_size != target_info.st_size:
                    raise SolarSystemDataError("native source and wheel member lengths differ")
                if _hash_fd(source_fd)[1] != _hash_fd(target_fd)[1]:
                    raise SolarSystemDataError("native source and wheel member bytes differ")
            finally:
                os.close(target_fd)
        finally:
            os.close(source_fd)
        target_parts = _components(target)
        parent = _walk_directory(self.root_fd, target_parts[:-1])
        leaf = target_parts[-1]
        try:
            os.unlink(leaf, dir_fd=parent)
        finally:
            os.close(parent)
        os.link(source, target, src_dir_fd=self.root_fd, dst_dir_fd=self.root_fd, follow_symlinks=False)
        left = self.open_file(source)
        try:
            right = self.open_file(target)
            try:
                if _stat_key(os.fstat(left))[:2] != _stat_key(os.fstat(right))[:2]:
                    raise SolarSystemDataError("native hardlink identity does not match")
            finally:
                os.close(right)
        finally:
            os.close(left)

    def _unlink_file(self, relative: str) -> None:
        parts = _components(relative)
        parent = _walk_directory(self.root_fd, parts[:-1])
        leaf = parts[-1]
        try:
            info = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                raise SolarSystemDataError("tracked stage file was replaced by a special node")
            os.unlink(leaf, dir_fd=parent)
        finally:
            os.close(parent)

    def _remove_directory(self, relative: str) -> None:
        parts = _components(relative)
        parent = _walk_directory(self.root_fd, parts[:-1])
        try:
            info = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode):
                raise SolarSystemDataError("tracked stage directory identity changed")
            os.rmdir(parts[-1], dir_fd=parent)
        finally:
            os.close(parent)

    def cleanup(self) -> None:
        failures: list[Exception] = []
        for relative in sorted(self.files, key=lambda item: (item.count("/"), item), reverse=True):
            try:
                self._unlink_file(relative)
            except Exception as exc:
                failures.append(exc)
        for relative in sorted(self.directories, key=lambda item: (item.count("/"), item), reverse=True):
            try:
                self._remove_directory(relative)
            except Exception as exc:
                failures.append(exc)
        try:
            observed = os.stat(self.name, dir_fd=self.tmp_fd, follow_symlinks=False)
            if _stable_directory_identity(observed) != self.root_identity:
                raise SolarSystemDataError("private stage pathname identity changed")
            os.close(self.root_fd)
            self.root_fd = -1
            os.rmdir(self.name, dir_fd=self.tmp_fd)
        except Exception as exc:
            failures.append(exc)
        finally:
            if self.root_fd >= 0:
                os.close(self.root_fd)
                self.root_fd = -1
            os.close(self.tmp_fd)
            self.tmp_fd = -1
        if failures:
            raise SolarSystemDataError("private stage cleanup failed closed") from failures[0]

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None) -> bool:
        self.cleanup()
        return False


def _write_all(fd: int, value: bytes) -> None:
    offset = 0
    while offset < len(value):
        written = os.write(fd, value[offset:])
        if written <= 0:
            raise SolarSystemDataError("stage write made no progress")
        offset += written


def _copy_artifact(source_root_fd: int, artifact: ArtifactBinding, stage: _Stage) -> None:
    source = _open_relative(source_root_fd, artifact.logical_locator)
    try:
        destination = stage.create_file(artifact.logical_locator)
        try:
            before = _stat_key(os.fstat(source))
            if before[6] != artifact.byte_length:
                raise SolarSystemDataError("source artifact length changed after verification")
            digest = hashlib.sha256()
            offset = 0
            while offset < artifact.byte_length:
                block = os.pread(source, min(_READ_CHUNK, artifact.byte_length - offset), offset)
                if not block:
                    raise SolarSystemDataError("source artifact returned early EOF during copy")
                _write_all(destination, block)
                digest.update(block)
                offset += len(block)
            if os.pread(source, 1, artifact.byte_length) != b"":
                raise SolarSystemDataError("source artifact grew during copy")
            os.fsync(destination)
            after = _stat_key(os.fstat(source))
            if before != after or digest.hexdigest() != artifact.artifact_sha256:
                raise SolarSystemDataError("source artifact changed during independent stage copy")
        finally:
            os.close(destination)
    finally:
        os.close(source)


@dataclass(frozen=True, slots=True)
class _WheelCatalog:
    artifact_id: str
    data: bytes
    files: tuple[str, ...]
    directories: tuple[str, ...]
    record_path: str
    record_rows: dict[str, tuple[str, int] | None]


def _zip_name(value: str, is_directory: bool) -> str:
    if type(value) is not str or "\x00" in value or "\\" in value:
        raise SolarSystemDataError("wheel member name is unsafe")
    normalized = value[:-1] if is_directory and value.endswith("/") else value
    if not normalized or len(normalized) > _MAXIMUM_ZIP_PATH_CODEPOINTS:
        raise SolarSystemDataError("wheel member path is outside its cap")
    path = PurePosixPath(normalized)
    parts = path.parts
    if path.is_absolute() or path.as_posix() != normalized or len(parts) > _MAXIMUM_ZIP_PATH_COMPONENTS:
        raise SolarSystemDataError("wheel member path is not normalized relative POSIX")
    if any(part in ("", ".", "..") or len(part) > _MAXIMUM_ZIP_COMPONENT_CODEPOINTS for part in parts):
        raise SolarSystemDataError("wheel member path component is unsafe")
    return normalized


def _raw_central_names(data: bytes) -> tuple[bytes, ...]:
    position = data.rfind(b"PK\x05\x06", max(0, len(data) - 65_557))
    if position < 0 or position + 22 > len(data):
        raise SolarSystemDataError("wheel lacks a bounded EOCD")
    signature, disk, central_disk, disk_entries, total_entries, central_size, central_offset, comment_length = struct.unpack_from("<4s4H2LH", data, position)
    if signature != b"PK\x05\x06" or disk != 0 or central_disk != 0 or disk_entries != total_entries or position + 22 + comment_length != len(data):
        raise SolarSystemDataError("wheel uses an unsupported multi-disk or malformed EOCD")
    if total_entries < 1 or total_entries > _MAXIMUM_ZIP_ENTRIES or central_offset + central_size != position:
        raise SolarSystemDataError("wheel central directory is outside its cap")
    names: list[bytes] = []
    cursor = central_offset
    for _ in range(total_entries):
        if cursor + 46 > position or data[cursor : cursor + 4] != b"PK\x01\x02":
            raise SolarSystemDataError("wheel central directory is malformed")
        name_length, extra_length, member_comment_length = struct.unpack_from("<HHH", data, cursor + 28)
        start = cursor + 46
        end = start + name_length
        if end + extra_length + member_comment_length > position:
            raise SolarSystemDataError("wheel central member exceeds directory bounds")
        raw = data[start:end]
        if b"\x00" in raw or b"\\" in raw:
            raise SolarSystemDataError("wheel central member contains NUL or backslash")
        names.append(raw)
        cursor = end + extra_length + member_comment_length
    if cursor != position:
        raise SolarSystemDataError("wheel central directory has trailing data")
    return tuple(names)


def _wheel_catalog(artifact: ArtifactBinding, stage: _Stage) -> _WheelCatalog:
    fd = stage.open_file(artifact.logical_locator)
    try:
        length, digest, _ = _hash_fd(fd, artifact.byte_length)
        if digest != artifact.artifact_sha256:
            raise SolarSystemDataError("staged wheel bytes differ from exact binding")
        data = _read_fd(fd, length)
    finally:
        os.close(fd)
    raw_names = _raw_central_names(data)
    files: list[str] = []
    directories: list[str] = []
    expanded = 0
    with zipfile.ZipFile(io.BytesIO(data), "r") as archive:
        infos = archive.infolist()
        if len(infos) != len(raw_names) or len(infos) > _MAXIMUM_ZIP_ENTRIES:
            raise SolarSystemDataError("wheel entry roster differs from central directory")
        observed: set[str] = set()
        for info, raw_name in zip(infos, raw_names):
            try:
                decoded_raw = raw_name.decode("utf-8" if info.flag_bits & 0x800 else "cp437")
            except UnicodeDecodeError as exc:
                raise SolarSystemDataError("wheel member name cannot be decoded canonically") from exc
            if decoded_raw != info.orig_filename:
                raise SolarSystemDataError("wheel filename differs from raw central-directory bytes")
            is_directory = info.is_dir()
            name = _zip_name(info.filename, is_directory)
            if name in observed:
                raise SolarSystemDataError("wheel repeats a normalized member path")
            observed.add(name)
            mode = (info.external_attr >> 16) & 0xFFFF
            if not mode or ((is_directory and not stat.S_ISDIR(mode)) or (not is_directory and not stat.S_ISREG(mode))):
                raise SolarSystemDataError("wheel contains a symlink or special member")
            if info.flag_bits & 0x1:
                raise SolarSystemDataError("encrypted wheel members are unsupported")
            if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise SolarSystemDataError("wheel compression method is outside the profile")
            if info.file_size < 0 or info.file_size > _MAXIMUM_ZIP_MEMBER_BYTES:
                raise SolarSystemDataError("wheel member exceeds its expanded byte cap")
            if info.file_size and (info.compress_size <= 0 or info.file_size > info.compress_size * _MAXIMUM_ZIP_COMPRESSION_RATIO):
                raise SolarSystemDataError("wheel member exceeds its compression-ratio cap")
            expanded += info.file_size
            if expanded > _MAXIMUM_ZIP_EXPANDED_BYTES:
                raise SolarSystemDataError("wheel expanded bytes exceed their aggregate cap")
            (directories if is_directory else files).append(name)
        file_set = set(files)
        for name in files:
            parts = name.split("/")
            if any("/".join(parts[:index]) in file_set for index in range(1, len(parts))):
                raise SolarSystemDataError("wheel has a file/prefix collision")
        record_paths = [name for name in files if name.endswith(".dist-info/RECORD")]
        if len(record_paths) != 1:
            raise SolarSystemDataError("wheel requires exactly one dist-info RECORD")
        record_path = record_paths[0]
        record_info = archive.getinfo(record_path)
        if record_info.file_size > _MAXIMUM_REQUEST_BYTES * 2:
            raise SolarSystemDataError("wheel RECORD exceeds its byte cap")
        record_bytes = archive.read(record_info)
        try:
            rows = list(csv.reader(io.StringIO(record_bytes.decode("utf-8"), newline=""), strict=True))
        except (UnicodeDecodeError, csv.Error) as exc:
            raise SolarSystemDataError("wheel RECORD is not strict UTF-8 CSV") from exc
        if len(rows) != len(files):
            raise SolarSystemDataError("wheel RECORD row count differs from file roster")
        record_rows: dict[str, tuple[str, int] | None] = {}
        prospective = 0
        info_by_name = {item.filename: item for item in infos if not item.is_dir()}
        for row in rows:
            if len(row) != 3:
                raise SolarSystemDataError("wheel RECORD row must have three fields")
            name = _zip_name(row[0], False)
            if name in record_rows or name not in file_set:
                raise SolarSystemDataError("wheel RECORD repeats or invents a member")
            size = info_by_name[name].file_size
            prospective += size
            if prospective > _MAXIMUM_ZIP_EXPANDED_BYTES:
                raise SolarSystemDataError("wheel RECORD prospective aggregate exceeds cap")
            if name == record_path:
                if row[1] != "" or row[2] != "":
                    raise SolarSystemDataError("wheel RECORD self row must be unhashed")
                record_rows[name] = None
            else:
                if not row[1].startswith("sha256=") or not row[2].isdigit() or row[2] != str(size):
                    raise SolarSystemDataError("wheel RECORD member hash/size is noncanonical")
                encoded_digest = row[1][7:]
                try:
                    raw_digest = base64.urlsafe_b64decode(encoded_digest + "=" * (-len(encoded_digest) % 4))
                except Exception as exc:
                    raise SolarSystemDataError("wheel RECORD digest is invalid") from exc
                if len(raw_digest) != 32 or base64.urlsafe_b64encode(raw_digest).rstrip(b"=").decode("ascii") != encoded_digest:
                    raise SolarSystemDataError("wheel RECORD digest is noncanonical")
                record_rows[name] = (raw_digest.hex(), size)
        if set(record_rows) != file_set:
            raise SolarSystemDataError("wheel RECORD is not closed over all files")
    return _WheelCatalog(artifact.artifact_id, data, tuple(files), tuple(directories), record_path, record_rows)


def _validate_catalog_union(catalogs: tuple[_WheelCatalog, _WheelCatalog]) -> None:
    files: set[str] = set()
    directories: set[str] = set()
    for catalog in catalogs:
        for name in catalog.files:
            if name in files or name in directories:
                raise SolarSystemDataError("wheel union repeats a path")
            parts = name.split("/")
            if (
                any("/".join(parts[:index]) in files for index in range(1, len(parts)))
                or any(existing.startswith(name + "/") for existing in files | directories)
            ):
                raise SolarSystemDataError("wheel union has a file/prefix collision")
            files.add(name)
        for name in catalog.directories:
            if name in files:
                raise SolarSystemDataError("wheel union has a file/directory collision")
            directories.add(name)


def _extract_catalog(catalog: _WheelCatalog, stage: _Stage) -> tuple[tuple[str, int, str], ...]:
    rows: list[tuple[str, int, str]] = []
    with zipfile.ZipFile(io.BytesIO(catalog.data), "r") as archive:
        for name in sorted(catalog.files, key=lambda item: item.encode("utf-8")):
            info = archive.getinfo(name)
            destination = stage.create_file(f"runtime/{name}")
            digest = hashlib.sha256()
            observed = 0
            try:
                with archive.open(info, "r") as source:
                    while True:
                        block = source.read(min(_READ_CHUNK, info.file_size - observed + 1))
                        if not block:
                            break
                        observed += len(block)
                        if observed > info.file_size:
                            raise SolarSystemDataError("wheel member exceeds declared expanded size")
                        _write_all(destination, block)
                        digest.update(block)
                os.fsync(destination)
            finally:
                os.close(destination)
            if observed != info.file_size:
                raise SolarSystemDataError("wheel member ended before declared expanded size")
            observed_digest = digest.hexdigest()
            expected = catalog.record_rows[name]
            if expected is not None and expected != (observed_digest, observed):
                raise SolarSystemDataError("extracted member differs from RECORD")
            rows.append((name, observed, observed_digest))
    return tuple(rows)


def _manifest(rows: tuple[tuple[str, int, str], ...]) -> tuple[int, int, int, str]:
    ordered = sorted(rows, key=lambda item: item[0].encode("utf-8"))
    preimage = b"".join(path.encode("utf-8") + b"\0" + str(length).encode("ascii") + b"\0" + digest.encode("ascii") + b"\n" for path, length, digest in ordered)
    return len(ordered), sum(item[1] for item in ordered), len(preimage), hashlib.sha256(preimage).hexdigest()


def _read_stage_file(stage: _Stage, relative: str, expected_length: int) -> bytes:
    fd = stage.open_file(relative)
    try:
        length, _, _ = _hash_fd(fd, expected_length)
        return _read_fd(fd, length)
    finally:
        os.close(fd)


def _duplicate_rejecting_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SolarSystemDataError("JSON contains a duplicate key")
        result[key] = value
    return result


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(value, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    except (TypeError, ValueError, RecursionError, UnicodeError) as exc:
        raise SolarSystemDataError("protocol value cannot be canonically encoded") from exc


def _parse_protocol_int(value: str) -> int:
    if len(value) > 20:
        raise SolarSystemDataError("protocol integer literal exceeds its cap")
    result = int(value, 10)
    if result.bit_length() > 63:
        raise SolarSystemDataError("protocol integer exceeds its cap")
    return result


def _reject_protocol_number(value: str) -> object:
    raise SolarSystemDataError("protocol forbids floating-point and nonfinite JSON numbers")


def _validate_composite_and_license_leaves(stage: _Stage, abi: str, by_id: dict[str, ArtifactBinding]) -> None:
    raw = _read_stage_file(stage, by_id[_COMPOSITE_ID].logical_locator, _COMPOSITE_LENGTH)
    if hashlib.sha256(raw).hexdigest() != _COMPOSITE_SHA256 or not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise SolarSystemDataError("composite license/provenance leaf outer identity mismatch")
    try:
        leaf = json.loads(
            raw[:-1].decode("ascii"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_int=_parse_protocol_int,
            parse_float=_reject_protocol_number,
            parse_constant=_reject_protocol_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, SolarSystemDataError, RecursionError, ValueError) as exc:
        raise SolarSystemDataError("composite leaf is not strict canonical ASCII JSON") from exc
    if _canonical_json(leaf) + b"\n" != raw:
        raise SolarSystemDataError("composite leaf does not canonically re-encode")
    top_keys = {
        "applies_to", "artifact_binding_id", "canonicalization", "claim",
        "license_conclusion_status", "member_order_status", "members", "schema",
        "spdx_expression", "version",
    }
    if type(leaf) is not dict or set(leaf) != top_keys:
        raise SolarSystemDataError("composite leaf has an invalid top-level schema")
    if (
        leaf["artifact_binding_id"] != _COMPOSITE_ID
        or leaf["schema"] != "jxplanetx.spiceypy-cspice-composite-license-provenance-leaf.v1"
        or leaf["claim"] != "COMPOSITE_LICENSE_AND_PROVENANCE_EVIDENCE_ONLY"
        or leaf["spdx_expression"] != "MIT AND LicenseRef-NAIF-SPICE-Rules"
        or leaf["license_conclusion_status"] != "NOT_EVALUATED_NONAUTHORIZING_NO_LEGAL_OR_REDISTRIBUTION_APPROVAL"
    ):
        raise SolarSystemDataError("composite leaf fixed claims differ from the profile")
    members = leaf["members"]
    if type(members) is not list or len(members) != 3:
        raise SolarSystemDataError("composite leaf member roster must contain exactly three items")
    expected_members = (
        ("SPICEYPY_8_2_0_MIT_LICENSE", "LICENSE_TEXT", _MIT_LENGTH, _MIT_SHA256),
        ("NAIF_SPICE_RULES", "LICENSE_TEXT", _RULES_LENGTH, _RULES_SHA256),
        ("N0067_DERIVED_PATCHED_NON_LICENSE_PROVENANCE_NOTICE", "NON_LICENSE_PROVENANCE_NOTICE", _NOTICE_LENGTH, _NOTICE_SHA256),
    )
    decoded: list[bytes] = []
    member_keys = {"encoding", "evidence_locator", "kind", "media_type", "member_id", "raw_base64", "raw_byte_length", "raw_sha256"}
    for index, (member, expected) in enumerate(zip(members, expected_members)):
        if type(member) is not dict or set(member) != member_keys:
            raise SolarSystemDataError("composite leaf member schema mismatch")
        member_id, kind, length, digest = expected
        if (
            member["encoding"] != "BASE64_RFC4648_PADDED"
            or member["member_id"] != member_id
            or member["kind"] != kind
            or member["raw_byte_length"] != length
            or member["raw_sha256"] != digest
            or type(member["raw_base64"]) is not str
        ):
            raise SolarSystemDataError(f"composite leaf member {index} metadata mismatch")
        try:
            value = base64.b64decode(member["raw_base64"], validate=True)
        except Exception as exc:
            raise SolarSystemDataError("composite leaf member base64 is invalid") from exc
        if base64.b64encode(value).decode("ascii") != member["raw_base64"] or len(value) != length or hashlib.sha256(value).hexdigest() != digest:
            raise SolarSystemDataError("composite leaf member bytes do not match their declaration")
        decoded.append(value)
    rules = _read_stage_file(stage, by_id[_RULES_ID].logical_locator, _RULES_LENGTH)
    numpy_license = _read_stage_file(stage, by_id[_NUMPY_LICENSE_ID].logical_locator, _NUMPY_LICENSE_LENGTH)
    wheel_mit = _read_stage_file(stage, "runtime/spiceypy-8.2.0.dist-info/licenses/LICENSE", _MIT_LENGTH)
    wheel_numpy_license = _read_stage_file(stage, "runtime/numpy-2.3.5.dist-info/LICENSE.txt", _NUMPY_LICENSE_LENGTH)
    if decoded[0] != wheel_mit or decoded[1] != rules or numpy_license != wheel_numpy_license:
        raise SolarSystemDataError("license leaves do not cross-bind the extracted wheel members")
    applies = leaf["applies_to"]
    try:
        wheel_target = applies["license_expression_applies_to"]["spiceypy_wheel_variants"][abi]
        native_target = applies["provenance_notice_applies_to"]["bundled_native"]
    except (KeyError, TypeError) as exc:
        raise SolarSystemDataError("composite leaf applies-to schema mismatch") from exc
    wheel = by_id[_SPICEYPY_ID]
    expected_wheel_target = {
        "artifact_id": wheel.artifact_id,
        "artifact_logical_locator": wheel.logical_locator,
        "artifact_role": wheel.artifact_role,
        "byte_length": wheel.byte_length,
        "distribution_name": "spiceypy",
        "license_artifact_id": wheel.license_artifact_id,
        "license_spdx": wheel.license_spdx,
        "media_type": wheel.media_type,
        "provider_id": wheel.provider_id,
        "sha256": wheel.artifact_sha256,
        "version": wheel.version,
    }
    native = by_id[_NATIVE_ID]
    if wheel_target != expected_wheel_target:
        raise SolarSystemDataError("composite leaf wheel target differs from the selected ABI binding")
    if not (
        type(native_target) is dict
        and native_target.get("artifact_id") == native.artifact_id
        and native_target.get("artifact_logical_locator") == native.logical_locator
        and native_target.get("artifact_role") == native.artifact_role
        and native_target.get("byte_length") == native.byte_length
        and native_target.get("license_artifact_id") == native.license_artifact_id
        and native_target.get("license_spdx") == native.license_spdx
        and native_target.get("media_type") == native.media_type
        and native_target.get("provider_id") == native.provider_id
        and native_target.get("sha256") == native.artifact_sha256
        and native_target.get("version") == native.version
        and native_target.get("relative_path") == "spiceypy/utils/libcspice.so"
        and native_target.get("toolkit_version_readback") == "CSPICE_N0067"
    ):
        raise SolarSystemDataError("composite leaf native provenance target mismatch")


def _observe_source_artifacts(by_id: dict[str, ArtifactBinding], root_fd: int) -> tuple[LocalArtifactVerificationReceipt, ...]:
    receipts: list[LocalArtifactVerificationReceipt] = []
    for artifact_id in _EXPECTED_ARTIFACT_IDS:
        receipts.append(verify_local_artifact_bytes(by_id[artifact_id], root_fd))
    return tuple(receipts)


def _prepare_stage(stage: _Stage, by_id: dict[str, ArtifactBinding], source_root_fd: int, abi: str) -> tuple[tuple[LocalArtifactVerificationReceipt, ...], LocalArtifactVerificationReceipt]:
    for artifact_id in _EXPECTED_ARTIFACT_IDS:
        _copy_artifact(source_root_fd, by_id[artifact_id], stage)
    runtime_directory_fd = stage.create_runtime_directory()
    os.close(runtime_directory_fd)
    spice_catalog = _wheel_catalog(by_id[_SPICEYPY_ID], stage)
    numpy_catalog = _wheel_catalog(by_id[_NUMPY_ID], stage)
    catalogs = (spice_catalog, numpy_catalog)
    _validate_catalog_union(catalogs)
    spice_rows = _extract_catalog(spice_catalog, stage)
    numpy_rows = _extract_catalog(numpy_catalog, stage)
    profile = _evidence._ABI_PROFILES[abi]
    if _manifest(spice_rows) != profile["spiceypy_tree"] or _manifest(numpy_rows) != profile["numpy_tree"]:
        raise SolarSystemDataError("parent extracted distribution manifest differs from calibrated profile")
    numpy_native_rows = tuple(row for row in numpy_rows if row[0].endswith((".so", ".dll", ".dylib")) or ".so." in PurePosixPath(row[0]).name)
    if _manifest(numpy_native_rows) != profile["numpy_native_tree"]:
        raise SolarSystemDataError("parent extracted NumPy native manifest differs from calibrated profile")
    stage.replace_with_hardlink(by_id[_NATIVE_ID].logical_locator, "runtime/spiceypy/utils/libcspice.so")
    _validate_composite_and_license_leaves(stage, abi, by_id)
    staged_receipts = {artifact_id: verify_local_artifact_bytes(by_id[artifact_id], stage.root_fd) for artifact_id in _EXPECTED_ARTIFACT_IDS}
    implementation = tuple(staged_receipts[artifact_id] for artifact_id in (_SPICEYPY_ID, _NUMPY_ID, _NATIVE_ID))
    return implementation, staged_receipts[_SPK_ID]


def _protocol_shape(value: object, depth: int = 0, counter: list[int] | None = None) -> None:
    if counter is None:
        counter = [0]
    if depth > _MAXIMUM_PROTOCOL_DEPTH:
        raise SolarSystemDataError("worker protocol exceeds its depth cap")
    counter[0] += 1
    if counter[0] > _MAXIMUM_PROTOCOL_NODES:
        raise SolarSystemDataError("worker protocol exceeds its node cap")
    if value is None or type(value) is bool:
        return
    if type(value) is int:
        if value.bit_length() > 63:
            raise SolarSystemDataError("worker protocol integer exceeds its cap")
        return
    if type(value) is str:
        if len(value) > 4_096 or any(ord(character) > 0x7F for character in value):
            raise SolarSystemDataError("worker protocol string exceeds its ASCII cap")
        return
    if type(value) is list:
        if len(value) > 4_096:
            raise SolarSystemDataError("worker protocol list exceeds its count cap")
        for item in value:
            _protocol_shape(item, depth + 1, counter)
        return
    if type(value) is dict:
        if len(value) > 256 or any(type(key) is not str for key in value):
            raise SolarSystemDataError("worker protocol object exceeds its schema cap")
        for key, item in value.items():
            _protocol_shape(key, depth + 1, counter)
            _protocol_shape(item, depth + 1, counter)
        return
    raise SolarSystemDataError("worker protocol contains a forbidden scalar type")


def _request_frame(body: dict[str, object]) -> tuple[bytes, str]:
    _protocol_shape(body)
    body_bytes = _canonical_json(body)
    digest = hashlib.sha256(body_bytes).hexdigest()
    encoded = _canonical_json({"body": body, "sha256": digest})
    if len(encoded) > _MAXIMUM_REQUEST_BYTES:
        raise SolarSystemContractError("worker request exceeds its hard byte cap")
    return struct.pack(">I", len(encoded)) + encoded, digest


def _terminate_and_reap(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except OSError:
        pass
    deadline = _monotonic_time.monotonic() + _WORKER_TERM_GRACE_SECONDS
    while _monotonic_time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except OSError:
            break
        _monotonic_time.sleep(0.01)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        process.wait(timeout=_WORKER_TERM_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.kill(process.pid, signal.SIGKILL)
        except OSError:
            pass
        process.wait()


def _bounded_exchange(process: subprocess.Popen[bytes], request: bytes) -> tuple[bytes, bytes, int]:
    if process.stdin is None or process.stdout is None or process.stderr is None:
        _terminate_and_reap(process)
        raise SolarSystemDataError("worker pipes were not created")
    streams = (process.stdin, process.stdout, process.stderr)
    selector: selectors.BaseSelector | None = None
    written = 0
    stdout = bytearray()
    stderr = bytearray()
    deadline = _monotonic_time.monotonic() + _WORKER_TIMEOUT_SECONDS
    try:
        for stream in streams:
            os.set_blocking(stream.fileno(), False)
        selector = selectors.DefaultSelector()
        selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        while selector.get_map():
            remaining = deadline - _monotonic_time.monotonic()
            if remaining <= 0:
                raise SolarSystemDataError("isolated CSPICE worker exceeded its deadline")
            events = selector.select(min(remaining, 0.25))
            if not events and process.poll() is not None:
                events = selector.select(0)
            for key, mask in events:
                stream = key.fileobj
                label = key.data
                if label == "stdin" and mask & selectors.EVENT_WRITE:
                    try:
                        count = os.write(stream.fileno(), request[written:])
                    except BrokenPipeError:
                        count = 0
                    written += count
                    if written == len(request) or count == 0:
                        selector.unregister(stream)
                        stream.close()
                elif mask & selectors.EVENT_READ:
                    cap = _MAXIMUM_RESULT_BYTES + 4 if label == "stdout" else _MAXIMUM_STDERR_BYTES
                    target = stdout if label == "stdout" else stderr
                    block = os.read(stream.fileno(), min(65_536, cap - len(target) + 1))
                    if block:
                        target.extend(block)
                        if len(target) > cap:
                            raise SolarSystemDataError(f"worker {label} exceeded its hard byte cap")
                    else:
                        selector.unregister(stream)
                        stream.close()
        remaining = max(0.0, deadline - _monotonic_time.monotonic())
        try:
            return_code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            raise SolarSystemDataError("isolated CSPICE worker did not terminate by its deadline") from exc
        return bytes(stdout), bytes(stderr), return_code
    except Exception:
        _terminate_and_reap(process)
        raise
    finally:
        if selector is not None:
            selector.close()
        for stream in streams:
            try:
                stream.close()
            except Exception:
                pass


def _parse_response(stdout: bytes, stderr: bytes, return_code: int) -> dict[str, object]:
    if stderr != b"":
        raise SolarSystemDataError("isolated CSPICE worker emitted stderr")
    if len(stdout) < 4:
        raise SolarSystemDataError("worker response frame is truncated")
    length = struct.unpack(">I", stdout[:4])[0]
    if length < 2 or length > _MAXIMUM_RESULT_BYTES or len(stdout) != length + 4:
        raise SolarSystemDataError("worker response frame length is invalid")
    encoded = stdout[4:]
    try:
        value = json.loads(
            encoded.decode("ascii"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_int=_parse_protocol_int,
            parse_float=_reject_protocol_number,
            parse_constant=_reject_protocol_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, SolarSystemDataError, RecursionError, ValueError) as exc:
        raise SolarSystemDataError("worker response is not strict ASCII JSON") from exc
    _protocol_shape(value)
    if _canonical_json(value) != encoded or type(value) is not dict:
        raise SolarSystemDataError("worker response is not canonical JSON")
    if value.get("status") == "ERROR":
        if return_code != 70:
            raise SolarSystemDataError("worker failure response has an inconsistent exit status")
        if set(value) != {"error_kind", "error_status", "schema", "status"} or value.get("schema") != _RESULT_SCHEMA or value.get("error_status") != "FAIL_CLOSED":
            raise SolarSystemDataError("worker failure response schema is invalid")
        kind = value.get("error_kind")
        if kind == "COVERAGE":
            raise SolarSystemCoverageError("CSPICE could not supply the exact target/epoch profile")
        if kind == "DEPENDENCY":
            raise SolarSystemDependencyUnavailableError("isolated CSPICE worker dependency profile is unavailable")
        if kind == "DATA":
            raise SolarSystemDataError("isolated CSPICE worker failed closed")
        raise SolarSystemDataError("worker returned an unknown failure class")
    if return_code != 0 or set(value) != {"result", "status"} or value.get("status") != "OK" or type(value.get("result")) is not dict:
        raise SolarSystemDataError("worker success response or exit status is inconsistent")
    return value["result"]


_WORKER_RESULT_KEYS = {
    "abi_tag", "after_clear_pool_counts", "cspice_toolkit_version",
    "effective_et_be_hex", "fegetround_after", "fegetround_before",
    "fresh_pool_counts", "host_native_dependencies", "libc_name", "libc_version",
    "loaded_distribution_native_files", "loaded_handle", "loaded_pool_counts",
    "nonce", "numpy_cpu_feature_payload_byte_length",
    "numpy_cpu_feature_payload_sha256", "numpy_native_tree", "numpy_record_sha256",
    "numpy_tree", "post_query_pool_counts", "python_executable_byte_length",
    "python_executable_sha256", "python_implementation", "python_version",
    "request_sha256", "role", "schema", "spiceypy_record_sha256",
    "spiceypy_tree", "spk_byte_length", "spk_offset_after", "spk_offset_before",
    "spk_sha256", "targets", "worker_source_byte_length", "worker_source_sha256",
}


def _f64_bits(value: float) -> str:
    return struct.pack(">d", value).hex()


def _decode_f64(value: object, label: str) -> float:
    if type(value) is not str or len(value) != 16 or any(character not in "0123456789abcdef" for character in value):
        raise SolarSystemDataError(f"{label} is not exact 8-byte big-endian binary64 hex")
    result = struct.unpack(">d", bytes.fromhex(value))[0]
    if not math.isfinite(result):
        raise SolarSystemDataError(f"{label} is nonfinite")
    return result


def _exact_worker_int(value: object, label: str, minimum: int | None = None, maximum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise SolarSystemDataError(f"{label} is outside its exact integer profile")
    return value


def _exact_worker_text(value: object, label: str, maximum: int = 256) -> str:
    if type(value) is not str or not value or len(value) > maximum or value.strip() != value or any(ord(character) < 0x20 or ord(character) > 0x7E for character in value):
        raise SolarSystemDataError(f"{label} is outside its exact text profile")
    return value


def _exact_list(value: object, length: int, label: str) -> list[object]:
    if type(value) is not list or len(value) != length:
        raise SolarSystemDataError(f"{label} has an invalid exact list shape")
    return value


def _validate_worker_result(
    result: dict[str, object],
    *,
    abi: str,
    role: str,
    nonce: str,
    request_sha256: str,
    python_executable_byte_length: int,
    python_executable_sha256: str,
    worker_source_byte_length: int,
    worker_sha256: str,
    effective_et_bits: str,
    targets: tuple[int, ...],
) -> None:
    if set(result) != _WORKER_RESULT_KEYS:
        raise SolarSystemDataError("worker result has an invalid exact key roster")
    profile = _evidence._ABI_PROFILES[abi]
    fixed = (
        (result["schema"], _RESULT_SCHEMA, "schema"),
        (result["abi_tag"], abi, "abi_tag"),
        (result["role"], role, "role"),
        (result["nonce"], nonce, "nonce"),
        (result["request_sha256"], request_sha256, "request_sha256"),
        (result["worker_source_sha256"], worker_sha256, "worker_source_sha256"),
        (result["effective_et_be_hex"], effective_et_bits, "effective_et_be_hex"),
        (result["python_implementation"], "CPython", "python_implementation"),
        (result["python_version"], profile["python_version"], "python_version"),
        (result["cspice_toolkit_version"], "CSPICE_N0067", "cspice_toolkit_version"),
        (result["spiceypy_record_sha256"], profile["spiceypy_record_sha256"], "spiceypy_record_sha256"),
        (result["numpy_record_sha256"], profile["numpy_record_sha256"], "numpy_record_sha256"),
        (result["spk_sha256"], _SPK_SHA256, "spk_sha256"),
    )
    for actual, expected, label in fixed:
        if actual != expected:
            raise SolarSystemDataError(f"worker {label} differs from the exact request/profile")
    for key, expected in (("spiceypy_tree", profile["spiceypy_tree"]), ("numpy_tree", profile["numpy_tree"]), ("numpy_native_tree", profile["numpy_native_tree"])):
        observed = _exact_list(result[key], 4, key)
        checked = (
            _exact_worker_int(observed[0], f"{key}[0]", 1, 1 << 31),
            _exact_worker_int(observed[1], f"{key}[1]", 1, 1 << 40),
            _exact_worker_int(observed[2], f"{key}[2]", 1, 1 << 30),
            _exact_worker_sha256(observed[3], f"{key}[3]"),
        )
        if checked != expected:
            raise SolarSystemDataError(f"worker {key} differs from calibrated manifest")
    if _exact_worker_int(result["worker_source_byte_length"], "worker_source_byte_length", 1, 1 << 30) != worker_source_byte_length:
        raise SolarSystemDataError("worker source length differs from the held parent observation")
    if _exact_worker_int(result["python_executable_byte_length"], "python_executable_byte_length", 1, 1 << 30) != python_executable_byte_length:
        raise SolarSystemDataError("worker Python length differs from the held parent observation")
    if _exact_worker_sha256(result["python_executable_sha256"], "python_executable_sha256") != python_executable_sha256:
        raise SolarSystemDataError("worker Python digest differs from the held parent observation")
    if _exact_worker_int(result["numpy_cpu_feature_payload_byte_length"], "numpy_cpu_feature_payload_byte_length", 1, 1 << 20) != 1_086 or _exact_worker_sha256(result["numpy_cpu_feature_payload_sha256"], "numpy_cpu_feature_payload_sha256") != "b050bd98f1d010d3f10c7d685e2eb6875793ddb9909110de8474aba96740fe74":
        raise SolarSystemDataError("worker CPU feature payload differs from calibrated profile")
    for key, expected in (("fresh_pool_counts", [0, 0]), ("loaded_pool_counts", [1, 1]), ("post_query_pool_counts", [1, 1]), ("after_clear_pool_counts", [0, 0])):
        observed = _exact_list(result[key], 2, key)
        checked = [_exact_worker_int(item, f"{key}[{index}]", 0, 1) for index, item in enumerate(observed)]
        if checked != expected:
            raise SolarSystemDataError(f"worker {key} differs from exact pool profile")
    if (
        _exact_worker_int(result["spk_byte_length"], "spk_byte_length", 1, 1 << 40) != _SPK_LENGTH
        or _exact_worker_int(result["spk_offset_before"], "spk_offset_before", 0, (1 << 63) - 1) != 0
        or _exact_worker_int(result["spk_offset_after"], "spk_offset_after", 0, (1 << 63) - 1) != 0
    ):
        raise SolarSystemDataError("worker held-SPK observation differs from exact profile")
    if _exact_worker_int(result["fegetround_before"], "fegetround_before", 0, 0) != 0 or _exact_worker_int(result["fegetround_after"], "fegetround_after", 0, 0) != 0:
        raise SolarSystemDataError("worker floating-point rounding mode differs from FE_TONEAREST")
    _exact_worker_int(result["loaded_handle"], "loaded_handle", 1, (1 << 31) - 1)
    rows = result["targets"]
    if type(rows) is not list or len(rows) != len(targets):
        raise SolarSystemDataError("worker target row count differs from query")
    total_legs = 0
    for row in rows:
        if type(row) is not dict or type(row.get("legs")) is not list:
            raise SolarSystemDataError("worker target row is not shallowly bounded")
        total_legs += len(row["legs"])
        if total_legs > _MAXIMUM_TOTAL_LEGS:
            raise SolarSystemDataError("worker selected-leg graph exceeds its global cap")
    for index, (row, target) in enumerate(zip(rows, targets)):
        if type(row) is not dict or set(row) != {"legs", "spkgeo_light_time_be_hex", "spkgeo_state_be_hex", "target_naif_id"} or _exact_worker_int(row["target_naif_id"], "target_naif_id", -(1 << 31) + 1, (1 << 31) - 1) != target:
            raise SolarSystemDataError(f"worker target row {index} schema/order mismatch")
        _exact_list(row["spkgeo_state_be_hex"], 6, f"target[{index}].state")
        for component_index, component in enumerate(row["spkgeo_state_be_hex"]):
            _decode_f64(component, f"target[{index}].state[{component_index}]")
        light_time = _decode_f64(row["spkgeo_light_time_be_hex"], f"target[{index}].light_time")
        if light_time < 0.0 or (light_time == 0.0 and math.copysign(1.0, light_time) < 0.0):
            raise SolarSystemDataError("worker light time is negative or negative zero")
        legs = row["legs"]
        if type(legs) is not list or not 1 <= len(legs) <= 16:
            raise SolarSystemDataError("worker selected chain is outside its exact cap")
        expected_body = target
        seen: set[int] = set()
        for leg_index, leg in enumerate(legs):
            expected_keys = {
                "begin_daf_address", "center_naif_id", "end_daf_address",
                "first_et_be_hex", "frame_naif_id", "last_et_be_hex",
                "segment_identifier", "segment_type", "selected_body_naif_id",
                "spkpvn_returned_center_naif_id", "spkpvn_returned_frame_naif_id",
                "spkpvn_state_be_hex",
            }
            if type(leg) is not dict or set(leg) != expected_keys:
                raise SolarSystemDataError("worker selected leg schema mismatch")
            body = _exact_worker_int(leg["selected_body_naif_id"], "selected_body_naif_id", -(1 << 31) + 1, (1 << 31) - 1)
            center = _exact_worker_int(leg["center_naif_id"], "center_naif_id", -(1 << 31) + 1, (1 << 31) - 1)
            if body != expected_body or body in seen or body == 0 or ((center == 0) is not (leg_index == len(legs) - 1)):
                raise SolarSystemDataError("worker selected chain continuity is invalid")
            seen.add(body)
            expected_body = center
            frame = _exact_worker_int(leg["frame_naif_id"], "frame_naif_id", -(1 << 31) + 1, (1 << 31) - 1)
            segment_type = _exact_worker_int(leg["segment_type"], "segment_type", -(1 << 31) + 1, (1 << 31) - 1)
            returned_frame = _exact_worker_int(leg["spkpvn_returned_frame_naif_id"], "spkpvn_returned_frame_naif_id", -(1 << 31) + 1, (1 << 31) - 1)
            returned_center = _exact_worker_int(leg["spkpvn_returned_center_naif_id"], "spkpvn_returned_center_naif_id", -(1 << 31) + 1, (1 << 31) - 1)
            if frame != 1 or segment_type != 2 or returned_frame != 1 or returned_center != center:
                raise SolarSystemDataError("worker selected leg frame/type/readback mismatch")
            begin = _exact_worker_int(leg["begin_daf_address"], "begin_daf_address", 1, (1 << 31) - 1)
            end = _exact_worker_int(leg["end_daf_address"], "end_daf_address", 1, (1 << 31) - 1)
            if begin > end:
                raise SolarSystemDataError("worker selected leg has reversed DAF addresses")
            first = _decode_f64(leg["first_et_be_hex"], "first_et")
            last = _decode_f64(leg["last_et_be_hex"], "last_et")
            effective = _decode_f64(effective_et_bits, "effective_et")
            if first > effective or effective > last:
                raise SolarSystemDataError("successful worker evidence contains a segment outside effective ET")
            _exact_worker_text(leg["segment_identifier"], "segment_identifier", 40)
            state = _exact_list(leg["spkpvn_state_be_hex"], 6, "spkpvn_state")
            for component in state:
                _decode_f64(component, "spkpvn_state component")


def _run_worker(
    *,
    abi: str,
    role: str,
    nonce: str,
    runtime_fd: int,
    spk_fd: int,
    python_fd: int,
    worker_fd: int,
    worker_source_byte_length: int,
    worker_sha256: str,
    effective_et: float,
    targets: tuple[int, ...],
) -> dict[str, object]:
    body: dict[str, object] = {
        "abi_tag": abi,
        "effective_et_be_hex": _f64_bits(effective_et),
        "nonce": nonce,
        "python_fd": python_fd,
        "request_token": _REQUEST_TOKEN,
        "role": role,
        "schema": _REQUEST_SCHEMA,
        "spk_fd": spk_fd,
        "stage_fd": runtime_fd,
        "targets": list(targets),
        "worker_fd": worker_fd,
        "worker_source_sha256": worker_sha256,
    }
    request, request_sha = _request_frame(body)
    expected_executable = os.stat(sys.executable)
    if (expected_executable.st_dev, expected_executable.st_ino) != (os.fstat(python_fd).st_dev, os.fstat(python_fd).st_ino):
        raise SolarSystemDependencyUnavailableError("held Python FD differs from current sys.executable")
    python_executable_byte_length, python_executable_sha256, _ = _hash_fd(python_fd)
    observed_worker_length, observed_worker_sha256, _ = _hash_fd(worker_fd, worker_source_byte_length)
    if observed_worker_sha256 != worker_sha256:
        raise SolarSystemDependencyUnavailableError("held worker source changed before execution")
    environment = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    try:
        process = subprocess.Popen(
            [sys.executable, "-I", "-B", "-S", f"/proc/self/fd/{worker_fd}"],
            executable=f"/proc/self/fd/{python_fd}",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="/",
            env=environment,
            close_fds=True,
            pass_fds=tuple(sorted({runtime_fd, spk_fd, python_fd, worker_fd})),
            start_new_session=True,
        )
    except (OSError, ValueError) as exc:
        raise SolarSystemDependencyUnavailableError("isolated current-CPython worker could not start") from exc
    stdout, stderr, return_code = _bounded_exchange(process, request)
    result = _parse_response(stdout, stderr, return_code)
    _validate_worker_result(
        result,
        abi=abi,
        role=role,
        nonce=nonce,
        request_sha256=request_sha,
        python_executable_byte_length=python_executable_byte_length,
        python_executable_sha256=python_executable_sha256,
        worker_source_byte_length=observed_worker_length,
        worker_sha256=worker_sha256,
        effective_et_bits=_f64_bits(effective_et),
        targets=targets,
    )
    return result


def _build_runtime(
    result: dict[str, object],
    projection: CspiceBinary64QueryProjectionReceipt,
    worker_sha256: str,
) -> CspiceRuntimeObservation:
    abi = result["abi_tag"]
    profile = _evidence._ABI_PROFILES[abi]
    loaded_rows = result["loaded_distribution_native_files"]
    if type(loaded_rows) is not list or len(loaded_rows) != 6:
        raise SolarSystemDataError("worker loaded-native roster has invalid count")
    loaded: list[tuple[str, str, int, str]] = []
    for index, row in enumerate(loaded_rows):
        values = _exact_list(row, 3, f"loaded_native[{index}]")
        path = _exact_worker_text(values[0], f"loaded_native[{index}].path", 1_024)
        length = _exact_worker_int(values[1], f"loaded_native[{index}].length", 1, 1 << 30)
        digest = _exact_worker_sha256(values[2], f"loaded_native[{index}].sha256")
        owner = _SPICEYPY_ID if path == "spiceypy/utils/libcspice.so" else _NUMPY_ID
        loaded.append((owner, path, length, digest))
    host_rows = result["host_native_dependencies"]
    if type(host_rows) is not list or len(host_rows) != 3:
        raise SolarSystemDataError("worker host-native roster has invalid count")
    host: list[tuple[str, int, str]] = []
    for index, row in enumerate(host_rows):
        values = _exact_list(row, 3, f"host_native[{index}]")
        host.append((_exact_worker_text(values[0], "host native name"), _exact_worker_int(values[1], "host native length", 1, 1 << 30), _exact_worker_sha256(values[2], "host native digest")))
    return CspiceRuntimeObservation(
        provider_identity=projection.effective_query.provider.identity,
        worker_source_sha256=worker_sha256,
        worker_source_status=_evidence._RUNTIME_WORKER_SOURCE_STATUS,
        python_implementation="CPython",
        python_version=profile["python_version"],
        python_wheel_interpreter_abi_tag=abi,
        python_executable_byte_length=result["python_executable_byte_length"],
        python_executable_sha256=result["python_executable_sha256"],
        sys_platform="linux",
        machine="x86_64",
        byteorder="little",
        pointer_bits=64,
        libc_name=result["libc_name"],
        libc_version=result["libc_version"],
        binary64_format="IEEE, little-endian",
        fegetround_before=result["fegetround_before"],
        fegetround_after=result["fegetround_after"],
        floating_point_environment_status=_evidence._RUNTIME_FLOATING_POINT_STATUS,
        python_invocation_status=_evidence._RUNTIME_PYTHON_INVOCATION_STATUS,
        execution_environment_status=_evidence._RUNTIME_ENVIRONMENT_STATUS,
        numpy_cpu_feature_payload_byte_length=result["numpy_cpu_feature_payload_byte_length"],
        numpy_cpu_feature_payload_sha256=result["numpy_cpu_feature_payload_sha256"],
        numpy_cpu_feature_payload_status=_evidence._RUNTIME_CPU_FEATURE_STATUS,
        distribution_tree_manifest_recipe=_evidence._RUNTIME_DISTRIBUTION_MANIFEST_RECIPE,
        numpy_native_tree_selection_and_manifest_recipe=_evidence._RUNTIME_NATIVE_MANIFEST_RECIPE,
        spiceypy_distribution_artifact_id=_SPICEYPY_ID,
        spiceypy_version="8.2.0",
        spiceypy_module_relative_path="spiceypy/__init__.py",
        spiceypy_record_sha256=result["spiceypy_record_sha256"],
        spiceypy_content_tree_file_count=result["spiceypy_tree"][0],
        spiceypy_content_tree_total_byte_length=result["spiceypy_tree"][1],
        spiceypy_content_tree_manifest_byte_length=result["spiceypy_tree"][2],
        spiceypy_content_tree_sha256=result["spiceypy_tree"][3],
        spiceypy_record_validation_status=_evidence._RUNTIME_RECORD_STATUS,
        spiceypy_loaded_module_origin_status=_evidence._RUNTIME_MODULE_ORIGIN_STATUS,
        numpy_distribution_artifact_id=_NUMPY_ID,
        numpy_version="2.3.5",
        numpy_module_relative_path="numpy/__init__.py",
        numpy_record_sha256=result["numpy_record_sha256"],
        numpy_content_tree_file_count=result["numpy_tree"][0],
        numpy_content_tree_total_byte_length=result["numpy_tree"][1],
        numpy_content_tree_manifest_byte_length=result["numpy_tree"][2],
        numpy_content_tree_sha256=result["numpy_tree"][3],
        numpy_native_tree_file_count=result["numpy_native_tree"][0],
        numpy_native_tree_total_byte_length=result["numpy_native_tree"][1],
        numpy_native_tree_manifest_byte_length=result["numpy_native_tree"][2],
        numpy_native_tree_sha256=result["numpy_native_tree"][3],
        numpy_record_validation_status=_evidence._RUNTIME_RECORD_STATUS,
        numpy_loaded_module_origin_status=_evidence._RUNTIME_MODULE_ORIGIN_STATUS,
        loaded_cspice_artifact_id=_NATIVE_ID,
        loaded_cspice_relative_path="spiceypy/utils/libcspice.so",
        loaded_cspice_byte_length=_NATIVE_LENGTH,
        loaded_cspice_sha256=_NATIVE_SHA256,
        cspice_toolkit_version="CSPICE_N0067",
        cspice_loader_resolution_status=_evidence._RUNTIME_LOADER_STATUS,
        cyice_execution_status=_evidence._RUNTIME_CYICE_STATUS,
        loaded_distribution_native_files=tuple(sorted(loaded)),
        loaded_native_map_observation_status=_evidence._RUNTIME_LOADED_MAP_STATUS,
        host_native_dependencies=tuple(sorted(host)),
        transitive_native_dependency_custody_status=_evidence._RUNTIME_DEPENDENCY_CUSTODY_STATUS,
        runtime_scope=_evidence._RUNTIME_SCOPE,
        content_integrity_class=_CONTENT_INTEGRITY_CLASS,
    )


def _build_chains_and_batch(
    result: dict[str, object],
    projection: CspiceBinary64QueryProjectionReceipt,
    batch_id: str,
) -> tuple[tuple[CspiceTargetChainEvidence, ...], ProviderNativeStateBatch, bytes]:
    effective_et = projection.binary64_projection.rounded_value
    rows = result["targets"]
    chains: list[CspiceTargetChainEvidence] = []
    positions: list[float] = []
    velocities: list[float] = []
    for row_index, row in enumerate(rows):
        state = tuple(_decode_f64(item, f"targets[{row_index}].state") for item in row["spkgeo_state_be_hex"])
        positions.extend(state[:3])
        velocities.extend(state[3:])
        legs: list[CspiceSelectedSegmentRecord] = []
        for leg_index, raw in enumerate(row["legs"]):
            leg = CspiceSelectedSegmentRecord(
                kernel_artifact_id=_SPK_ID,
                kernel_artifact_sha256=_SPK_SHA256,
                kernel_load_index=0,
                selection_et=effective_et,
                selected_body_naif_id=raw["selected_body_naif_id"],
                center_naif_id=raw["center_naif_id"],
                frame_naif_id=raw["frame_naif_id"],
                segment_type=raw["segment_type"],
                first_et=_decode_f64(raw["first_et_be_hex"], f"targets[{row_index}].legs[{leg_index}].first_et"),
                last_et=_decode_f64(raw["last_et_be_hex"], f"targets[{row_index}].legs[{leg_index}].last_et"),
                begin_daf_address=raw["begin_daf_address"],
                end_daf_address=raw["end_daf_address"],
                segment_identifier=raw["segment_identifier"],
                spkpvn_returned_frame_naif_id=raw["spkpvn_returned_frame_naif_id"],
                spkpvn_returned_center_naif_id=raw["spkpvn_returned_center_naif_id"],
                spkpvn_state=tuple(_decode_f64(item, f"targets[{row_index}].legs[{leg_index}].state") for item in raw["spkpvn_state_be_hex"]),
                selection_status=_evidence._SEGMENT_SELECTION_STATUS,
                evidence_scope=_evidence._SEGMENT_EVIDENCE_SCOPE,
            )
            legs.append(leg)
        checked_legs = tuple(legs)
        reconstructed = _evidence._reconstruct_chain_state(checked_legs)
        chains.append(
            CspiceTargetChainEvidence(
                target_naif_id=row["target_naif_id"],
                observer_naif_id=0,
                reference_frame_name="J2000",
                effective_et=effective_et,
                spkgeo_state=state,
                spkgeo_light_time_seconds=_decode_f64(row["spkgeo_light_time_be_hex"], f"targets[{row_index}].light_time"),
                legs=checked_legs,
                terminal_center_naif_id=0,
                reconstructed_state=reconstructed,
                comparison_recipe=_evidence._CHAIN_COMPARISON_RECIPE,
                comparison_status=_evidence._CHAIN_COMPARISON_STATUS,
                evidence_scope=_evidence._CHAIN_EVIDENCE_SCOPE,
            )
        )
    query = projection.effective_query
    batch = ProviderNativeStateBatch(
        batch_id=batch_id,
        query=query,
        state_stage="DECLARED_PROVIDER_NATIVE_GEOMETRIC_TARGET_RELATIVE_OBSERVER_STATE_NO_JX_CONVERSION",
        execution_evidence_status="CALLER_SUPPLIED_UNAUTHENTICATED_NO_EXECUTION_RECEIPT",
        target_naif_ids=query.target_naif_ids,
        component_order=("X", "Y", "Z"),
        state_shape=(len(query.target_naif_ids), 3),
        position_unit_id=KILOMETRE.unit_id,
        velocity_length_unit_id=KILOMETRE.unit_id,
        velocity_time_unit_id=SECOND.unit_id,
        velocity_semantics="COORDINATE_DERIVATIVE_PER_PROVIDER_NATIVE_TIME_UNIT",
        positions=tuple(positions),
        velocities=tuple(velocities),
        target_availability_status=query.target_chain_availability_status,
        artifact_custody_status="NO_ARTIFACT_AT_USE_CUSTODY_EVIDENCE",
        semantic_replay_status="REQUIRES_PROVIDER_SPECIFIC_SEMANTIC_REPLAY",
        content_integrity_class=_CONTENT_INTEGRITY_CLASS,
    )
    payload = bytearray()
    for index in range(len(query.target_naif_ids)):
        offset = index * 3
        payload.extend(struct.pack(">6d", *batch.positions[offset : offset + 3], *batch.velocities[offset : offset + 3]))
    return tuple(chains), batch, bytes(payload)


def _build_lane(
    *,
    role: str,
    nonce: str,
    batch_id: str,
    projection: CspiceBinary64QueryProjectionReceipt,
    result: dict[str, object],
    implementation_receipts: tuple[LocalArtifactVerificationReceipt, ...],
    pre_spk_receipt: LocalArtifactVerificationReceipt,
    post_spk_receipt: LocalArtifactVerificationReceipt,
    worker_sha256: str,
) -> CspiceSpkgeoExecutionLane:
    runtime = _build_runtime(result, projection, worker_sha256)
    chains, batch, payload = _build_chains_and_batch(result, projection, batch_id)
    return CspiceSpkgeoExecutionLane(
        lane_role=role,
        execution_instance_nonce=nonce,
        execution_instance_evidence_status=_evidence._LANE_INSTANCE_STATUS,
        provider_profile=_evidence._PROVIDER_PROFILE,
        projection=projection,
        runtime=runtime,
        implementation_artifact_verifications=implementation_receipts,
        pre_spk_verifications=(pre_spk_receipt,),
        post_spk_verifications=(post_spk_receipt,),
        ordered_loaded_artifact_ids=(_SPK_ID,),
        kernel_pool_count_field_order=("ALL", "SPK"),
        fresh_pool_counts=tuple(result["fresh_pool_counts"]),
        loaded_pool_counts=tuple(result["loaded_pool_counts"]),
        post_query_pool_counts=tuple(result["post_query_pool_counts"]),
        after_clear_pool_counts=tuple(result["after_clear_pool_counts"]),
        loaded_kernel_inventory=((_SPK_ID, "DAF", "SPK", "DIRECT_FURNSH_SOURCE_EMPTY_NO_META_KERNEL"),),
        kernel_pool_status=_evidence._LANE_POOL_STATUS,
        cspice_error_state_status=_evidence._LANE_ERROR_STATUS,
        provider_call=_evidence._LANE_PROVIDER_CALL,
        state_batch=batch,
        target_chains=chains,
        state_payload_recipe=_evidence._LANE_STATE_PAYLOAD_RECIPE,
        state_payload_byte_length=len(payload),
        state_payload_sha256=hashlib.sha256(payload).hexdigest(),
        provider_execution_status=_evidence._LANE_EXECUTION_STATUS,
        nested_batch_evidence_status=_evidence._LANE_NESTED_BATCH_STATUS,
        atomicity_status=_evidence._LANE_ATOMICITY_STATUS,
        artifact_at_use_custody_status=_evidence._LANE_ARTIFACT_CUSTODY_STATUS,
        process_isolation_status=_evidence._LANE_PROCESS_STATUS,
        network_status=_evidence._LANE_NETWORK_STATUS,
        fallback_status=_evidence._LANE_FALLBACK_STATUS,
        extrapolation_status=_evidence._LANE_EXTRAPOLATION_STATUS,
        portability_scope=_evidence._LANE_PORTABILITY_SCOPE,
        content_integrity_class=_CONTENT_INTEGRITY_CLASS,
    )


def _run_worker_with_held_spk_observation(
    *,
    abi: str,
    role: str,
    nonce: str,
    runtime_fd: int,
    spk_fd: int,
    python_fd: int,
    worker_fd: int,
    worker_source_byte_length: int,
    worker_sha256: str,
    effective_et: float,
    targets: tuple[int, ...],
) -> dict[str, object]:
    before_length, before_sha, before_stat = _hash_fd(spk_fd, _SPK_LENGTH)
    if before_length != _SPK_LENGTH or before_sha != _SPK_SHA256 or os.lseek(spk_fd, 0, os.SEEK_CUR) != 0:
        raise SolarSystemDataError("held staged SPK differs from exact pre-use bytes")
    worker_failure: Exception | None = None
    result: dict[str, object] | None = None
    try:
        result = _run_worker(
            abi=abi,
            role=role,
            nonce=nonce,
            runtime_fd=runtime_fd,
            spk_fd=spk_fd,
            python_fd=python_fd,
            worker_fd=worker_fd,
            worker_source_byte_length=worker_source_byte_length,
            worker_sha256=worker_sha256,
            effective_et=effective_et,
            targets=targets,
        )
    except Exception as exc:
        worker_failure = exc
    after_length, after_sha, after_stat = _hash_fd(spk_fd, _SPK_LENGTH)
    if (
        after_length != before_length
        or after_sha != before_sha
        or after_stat != before_stat
        or os.lseek(spk_fd, 0, os.SEEK_CUR) != 0
    ):
        raise SolarSystemDataError("held staged SPK changed during worker use")
    if worker_failure is not None:
        raise worker_failure
    if result is None:
        raise SolarSystemDataError("worker returned no validated result")
    return result


def _execute_lane(
    *,
    role: str,
    nonce: str,
    batch_id: str,
    projection: CspiceBinary64QueryProjectionReceipt,
    by_id: dict[str, ArtifactBinding],
    source_root_fd: int,
    abi: str,
    python_fd: int,
    worker_fd: int,
    worker_source_byte_length: int,
    worker_sha256: str,
) -> CspiceSpkgeoExecutionLane:
    # These source observations are operational gates only.  The lane retains
    # the closer staged implementation and SPK observations.
    _observe_source_artifacts(by_id, source_root_fd)
    with _Stage() as stage:
        implementation, pre_spk = _prepare_stage(stage, by_id, source_root_fd, abi)
        runtime_fd = -1
        spk_fd = -1
        operation_failure: Exception | None = None
        result: dict[str, object] | None = None
        try:
            runtime_fd = os.open("runtime", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=stage.root_fd)
            spk_fd = stage.open_file(by_id[_SPK_ID].logical_locator)
            try:
                result = _run_worker_with_held_spk_observation(
                    abi=abi,
                    role=role,
                    nonce=nonce,
                    runtime_fd=runtime_fd,
                    spk_fd=spk_fd,
                    python_fd=python_fd,
                    worker_fd=worker_fd,
                    worker_source_byte_length=worker_source_byte_length,
                    worker_sha256=worker_sha256,
                    effective_et=projection.binary64_projection.rounded_value,
                    targets=projection.effective_query.target_naif_ids,
                )
            except Exception as exc:
                operation_failure = exc
        finally:
            if spk_fd >= 0:
                try:
                    os.close(spk_fd)
                finally:
                    if runtime_fd >= 0:
                        os.close(runtime_fd)
            elif runtime_fd >= 0:
                os.close(runtime_fd)
        post_spk = verify_local_artifact_bytes(by_id[_SPK_ID], stage.root_fd)
        if operation_failure is not None:
            raise operation_failure
        if result is None:
            raise SolarSystemDataError("worker returned no validated result")
        try:
            return _build_lane(
                role=role,
                nonce=nonce,
                batch_id=batch_id,
                projection=projection,
                result=result,
                implementation_receipts=implementation,
                pre_spk_receipt=pre_spk,
                post_spk_receipt=post_spk,
                worker_sha256=worker_sha256,
            )
        except SolarSystemContractError as exc:
            raise SolarSystemDataError("untrusted worker evidence could not form the frozen execution contracts") from exc


def execute_cspice_spkgeo_replay(
    projection: CspiceBinary64QueryProjectionReceipt,
    *,
    artifact_root_directory_fd: int,
    receipt_id: str,
    primary_batch_id: str,
    replay_batch_id: str,
    primary_execution_instance_nonce: str,
    replay_execution_instance_nonce: str,
) -> CspiceSpkgeoExecutionReceipt:
    """Execute the exact current-ABI CSPICE profile twice and seal the replay."""

    checked_receipt_id = _identifier(receipt_id, "receipt_id")
    checked_primary_batch = _identifier(primary_batch_id, "primary_batch_id")
    checked_replay_batch = _identifier(replay_batch_id, "replay_batch_id")
    checked_primary_nonce = _sha256_text(primary_execution_instance_nonce, "primary_execution_instance_nonce")
    checked_replay_nonce = _sha256_text(replay_execution_instance_nonce, "replay_execution_instance_nonce")
    checked_root_fd = _exact_fd(artifact_root_directory_fd, "artifact_root_directory_fd")
    if checked_primary_batch == checked_replay_batch:
        raise SolarSystemContractError("primary and replay batch IDs must be distinct")
    if checked_primary_nonce == checked_replay_nonce:
        raise SolarSystemContractError("primary and replay instance nonces must be distinct")
    if type(projection) is not CspiceBinary64QueryProjectionReceipt:
        raise SolarSystemContractError("projection must be an exact CspiceBinary64QueryProjectionReceipt")
    validate_cspice_binary64_query_projection_receipt(projection)
    abi = _current_abi()
    by_id = _require_exact_provider_roster(projection, abi)
    _require_platform()
    python_fd = -1
    worker_fd = -1
    execution_root_fd = -1
    try:
        execution_root_fd = os.dup(checked_root_fd)
        root_identity = _stable_directory_identity(os.fstat(execution_root_fd))
        python_fd = os.open("/proc/self/exe", os.O_RDONLY | os.O_CLOEXEC)
        python_info = os.fstat(python_fd)
        executable_info = os.stat(sys.executable)
        if not stat.S_ISREG(python_info.st_mode) or (python_info.st_dev, python_info.st_ino) != (executable_info.st_dev, executable_info.st_ino):
            raise SolarSystemDependencyUnavailableError("current sys.executable does not identify the running interpreter")
        worker_path = os.path.join(os.path.dirname(__file__), "_cspice_worker.py")
        worker_fd = os.open(worker_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        worker_length, worker_sha256, _ = _hash_fd(worker_fd, _EXPECTED_WORKER_BYTE_LENGTH)
        if worker_sha256 != _EXPECTED_WORKER_SHA256:
            raise SolarSystemDependencyUnavailableError("private worker source differs from the qualified M4C2B bytes")
        primary = _execute_lane(
            role="PRIMARY",
            nonce=checked_primary_nonce,
            batch_id=checked_primary_batch,
            projection=projection,
            by_id=by_id,
            source_root_fd=execution_root_fd,
            abi=abi,
            python_fd=python_fd,
            worker_fd=worker_fd,
            worker_source_byte_length=worker_length,
            worker_sha256=worker_sha256,
        )
        replay = _execute_lane(
            role="REPLAY",
            nonce=checked_replay_nonce,
            batch_id=checked_replay_batch,
            projection=projection,
            by_id=by_id,
            source_root_fd=execution_root_fd,
            abi=abi,
            python_fd=python_fd,
            worker_fd=worker_fd,
            worker_source_byte_length=worker_length,
            worker_sha256=worker_sha256,
        )
        if _stable_directory_identity(os.fstat(execution_root_fd)) != root_identity:
            raise SolarSystemDataError("borrowed artifact root capability identity changed during replay")
        try:
            return CspiceSpkgeoExecutionReceipt(
                receipt_id=checked_receipt_id,
                primary_lane=primary,
                replay_lane=replay,
                semantic_replay_comparison_scope=_evidence._REPLAY_COMPARISON_SCOPE,
                semantic_replay_status=_evidence._REPLAY_STATUS,
                claim_scope=_evidence._REPLAY_CLAIM_SCOPE,
                content_integrity_class=_CONTENT_INTEGRITY_CLASS,
            )
        except SolarSystemContractError as exc:
            raise SolarSystemDataError("generated primary/replay evidence failed the frozen receipt contract") from exc
    except (SolarSystemContractError, SolarSystemCoverageError, SolarSystemDependencyUnavailableError, SolarSystemDataError):
        raise
    except (OSError, OverflowError, zipfile.BadZipFile, csv.Error, subprocess.SubprocessError) as exc:
        raise SolarSystemDataError("M4C2B failed during bounded local staging or isolated execution") from exc
    finally:
        if worker_fd >= 0:
            os.close(worker_fd)
        if python_fd >= 0:
            os.close(python_fd)
        if execution_root_fd >= 0:
            os.close(execution_root_fd)


__all__ = ["execute_cspice_spkgeo_replay"]
