"""Fail-closed JPL-kernel ephemeris API candidate.

This module evaluates states from caller-supplied NAIF kernels.  It does not
integrate a JX dynamical model and must not be represented as an independently
generated JX ephemeris.  Every kernel is bound by an expected byte length and
SHA-256 digest, opened without following a final-component symlink, and checked
before and after an isolated CSPICE worker process executes the query.
The worker itself executes from a sealed in-memory copy, and the resulting
state binds the worker, Python executable, SpiceyPy module, and CSPICE library.

The API deliberately performs no download, kernel discovery, extrapolation,
or fallback.  CSPICE coverage failures remain failures.  Content digests are
integrity and provenance identifiers, not signatures or proof of origin.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from typing import Final

try:  # The execution boundary below is intentionally Linux-only.
    import fcntl
except ImportError:  # pragma: no cover - exercised by the platform guard.
    fcntl = None  # type: ignore[assignment]


_KERNEL_KINDS: Final = frozenset({"SPK", "LSK", "PCK", "FK", "CK", "SCLK", "IK"})
_CALENDAR_SCALES: Final = frozenset({"UTC", "TT", "TDB"})
_EPOCH_FORMATS: Final = frozenset({"CALENDAR", "SECONDS_PAST_J2000"})
_ABERRATION_CORRECTIONS: Final = frozenset(
    {"NONE", "LT", "LT+S", "CN", "CN+S", "XLT", "XLT+S", "XCN", "XCN+S"}
)
_MAX_TEXT = 512
_MAX_KERNELS = 64
_MAX_KERNEL_BYTES = 1 << 44
_MAX_WORKER_OUTPUT = 1 << 20
_MAX_WORKER_BYTES = 1 << 20
_MAX_RUNTIME_FILE_BYTES = 1 << 31
_WORKER_REQUEST_SCHEMA = "jx.ephemeris-worker-request.v2"
_WORKER_RESPONSE_SCHEMA = "jx.ephemeris-worker-response.v2"
_STATE_SCHEMA = "jx.ephemeris-state.v2"
_SOURCE_KIND = "JPL_KERNEL_INTERPOLATION"
_SCIENTIFIC_STATUS = "PROVIDER_DERIVED_STATE_NOT_JX_NATIVE_EPHEMERIS"
_COVERAGE_STATUS = "CSPICE_QUERY_SUCCEEDED_AT_REQUESTED_EPOCH"
_INTEGRITY_CLASS = "SHA256_CONTENT_INTEGRITY_NOT_SOURCE_AUTHENTICATION"
_ISO_CALENDAR = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,9})?"
)


class EphemerisError(RuntimeError):
    """Base class for JPL-kernel ephemeris failures."""


class EphemerisContractError(EphemerisError, ValueError):
    """The caller supplied an invalid or ambiguous request."""


class EphemerisDependencyError(EphemerisError):
    """The optional CSPICE runtime is unavailable."""


class EphemerisKernelError(EphemerisError):
    """Kernel custody, identity, loading, or time-realization failed."""


class EphemerisCoverageError(EphemerisError):
    """CSPICE could not form the requested target/observer state at the epoch."""


class EphemerisExecutionError(EphemerisError):
    """The isolated CSPICE execution failed outside the coverage boundary."""


def _ascii(value: object, label: str, *, maximum: int = _MAX_TEXT) -> str:
    if type(value) is not str:
        raise EphemerisContractError(f"{label} must be an exact string")
    if not value or len(value) > maximum or value.strip() != value:
        raise EphemerisContractError(f"{label} must be nonempty, trimmed, and at most {maximum} characters")
    if any(ord(character) < 0x20 or ord(character) > 0x7E for character in value):
        raise EphemerisContractError(f"{label} must contain printable ASCII only")
    return value


def _sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EphemerisContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _body_payload(value: int | str, label: str) -> dict[str, object]:
    if type(value) is int:
        if not -(1 << 31) <= value < (1 << 31):
            raise EphemerisContractError(f"{label} NAIF ID is outside signed 32-bit range")
        return {"kind": "NAIF_ID", "value": value}
    return {"kind": "NAIF_NAME", "value": _ascii(value, label, maximum=128)}


@dataclass(frozen=True, slots=True)
class EphemerisEpoch:
    """An explicit calendar scale or native CSPICE ET coordinate."""

    value: str | float
    time_scale: str
    format: str

    def __post_init__(self) -> None:
        if self.format not in _EPOCH_FORMATS:
            raise EphemerisContractError(f"epoch format must be one of {sorted(_EPOCH_FORMATS)}")
        if self.format == "SECONDS_PAST_J2000":
            if self.time_scale != "TDB":
                raise EphemerisContractError("seconds-past-J2000 epochs must use the TDB scale")
            if type(self.value) is not float or not math.isfinite(self.value):
                raise EphemerisContractError("seconds-past-J2000 epoch value must be a finite exact float")
        else:
            if self.time_scale not in _CALENDAR_SCALES:
                raise EphemerisContractError(f"calendar epoch scale must be one of {sorted(_CALENDAR_SCALES)}")
            calendar = _ascii(self.value, "calendar epoch", maximum=256)
            if _ISO_CALENDAR.fullmatch(calendar) is None:
                raise EphemerisContractError(
                    "calendar epoch must use YYYY-MM-DDTHH:MM:SS with optional fractional seconds and no scale suffix"
                )

    @classmethod
    def tdb_seconds(cls, value: float) -> EphemerisEpoch:
        return cls(value=value, time_scale="TDB", format="SECONDS_PAST_J2000")

    @classmethod
    def calendar(cls, value: str, *, time_scale: str) -> EphemerisEpoch:
        return cls(value=value, time_scale=time_scale, format="CALENDAR")

    def _payload(self) -> dict[str, object]:
        return {"format": self.format, "time_scale": self.time_scale, "value": self.value}


@dataclass(frozen=True, slots=True)
class KernelArtifact:
    """One explicitly located and byte-bound NAIF kernel."""

    logical_name: str
    kind: str
    path: str
    byte_length: int
    sha256: str

    def __post_init__(self) -> None:
        logical_name = _ascii(self.logical_name, "kernel logical_name", maximum=128)
        if "/" in logical_name or "\\" in logical_name or logical_name in {".", ".."}:
            raise EphemerisContractError("kernel logical_name must be one portable path component")
        if self.kind not in _KERNEL_KINDS:
            raise EphemerisContractError(f"kernel kind must be one of {sorted(_KERNEL_KINDS)}")
        path = _ascii(self.path, "kernel path", maximum=4096)
        if not os.path.isabs(path) or os.path.normpath(path) != path:
            raise EphemerisContractError("kernel path must be absolute and normalized")
        if type(self.byte_length) is not int or not 0 < self.byte_length <= _MAX_KERNEL_BYTES:
            raise EphemerisContractError("kernel byte_length is outside the accepted range")
        _sha256(self.sha256, "kernel sha256")

    def _portable_payload(self) -> dict[str, object]:
        return {
            "byte_length": self.byte_length,
            "kind": self.kind,
            "logical_name": self.logical_name,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class KernelBundle:
    """Ordered, portable identity for a caller-custodied JPL kernel set."""

    bundle_id: str
    ephemeris_name: str
    artifacts: tuple[KernelArtifact, ...]
    provider: str = "JPL"
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _ascii(self.bundle_id, "bundle_id", maximum=128)
        _ascii(self.ephemeris_name, "ephemeris_name", maximum=128)
        if self.provider != "JPL":
            raise EphemerisContractError("JplEphemeris accepts only the explicit JPL provider label")
        if type(self.artifacts) is not tuple or not 0 < len(self.artifacts) <= _MAX_KERNELS:
            raise EphemerisContractError("artifacts must be a nonempty exact tuple within the kernel-count cap")
        if any(type(item) is not KernelArtifact for item in self.artifacts):
            raise EphemerisContractError("every artifact must be an exact KernelArtifact")
        names = tuple(item.logical_name for item in self.artifacts)
        paths = tuple(item.path for item in self.artifacts)
        if len(set(names)) != len(names) or len(set(paths)) != len(paths):
            raise EphemerisContractError("kernel logical names and paths must be unique")
        if not any(item.kind == "SPK" for item in self.artifacts):
            raise EphemerisContractError("a JPL ephemeris bundle requires at least one SPK kernel")
        digest = _canonical_sha256(
            {
                "artifacts": [item._portable_payload() for item in self.artifacts],
                "bundle_id": self.bundle_id,
                "ephemeris_name": self.ephemeris_name,
                "provider": self.provider,
                "schema": "jx.jpl-kernel-bundle.v1",
            }
        )
        object.__setattr__(self, "content_sha256", digest)


@dataclass(frozen=True, slots=True)
class EphemerisRequest:
    """One explicit CSPICE target/observer state request."""

    request_id: str
    target: int | str
    observer: int | str
    epoch: EphemerisEpoch
    frame: str = "J2000"
    aberration_correction: str = "NONE"
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _ascii(self.request_id, "request_id", maximum=128)
        target = _body_payload(self.target, "target")
        observer = _body_payload(self.observer, "observer")
        if target == observer:
            raise EphemerisContractError("target and observer must differ")
        if type(self.epoch) is not EphemerisEpoch:
            raise EphemerisContractError("epoch must be an exact EphemerisEpoch")
        _ascii(self.frame, "frame", maximum=128)
        if self.aberration_correction not in _ABERRATION_CORRECTIONS:
            raise EphemerisContractError(
                f"aberration_correction must be one of {sorted(_ABERRATION_CORRECTIONS)}"
            )
        digest = _canonical_sha256(
            {
                "aberration_correction": self.aberration_correction,
                "epoch": self.epoch._payload(),
                "frame": self.frame,
                "observer": observer,
                "request_id": self.request_id,
                "schema": "jx.ephemeris-request.v1",
                "target": target,
            }
        )
        object.__setattr__(self, "content_sha256", digest)

    def _worker_payload(self) -> dict[str, object]:
        return {
            "aberration_correction": self.aberration_correction,
            "epoch": self.epoch._payload(),
            "frame": self.frame,
            "observer": str(self.observer),
            "target": str(self.target),
        }


@dataclass(frozen=True, slots=True)
class KernelIdentity:
    """Portable byte identity for a kernel actually used by the worker."""

    logical_name: str
    kind: str
    byte_length: int
    sha256: str

    def __post_init__(self) -> None:
        logical_name = _ascii(self.logical_name, "kernel identity logical_name", maximum=128)
        if "/" in logical_name or "\\" in logical_name or logical_name in {".", ".."}:
            raise EphemerisContractError(
                "kernel identity logical_name must be one portable path component"
            )
        if self.kind not in _KERNEL_KINDS:
            raise EphemerisContractError(
                f"kernel identity kind must be one of {sorted(_KERNEL_KINDS)}"
            )
        if type(self.byte_length) is not int or not 0 < self.byte_length <= _MAX_KERNEL_BYTES:
            raise EphemerisContractError("kernel identity byte_length is outside the accepted range")
        _sha256(self.sha256, "kernel identity sha256")

    def _portable_payload(self) -> dict[str, object]:
        return {
            "byte_length": self.byte_length,
            "kind": self.kind,
            "logical_name": self.logical_name,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class EphemerisState:
    """A state interpolated by CSPICE from the identified JPL kernel bundle."""

    request_id: str
    request_sha256: str
    bundle_id: str
    bundle_sha256: str
    ephemeris_name: str
    target: int | str
    target_naif_id: int
    observer: int | str
    observer_naif_id: int
    input_epoch: EphemerisEpoch
    ephemeris_time_seconds: float
    frame: str
    aberration_correction: str
    position_km: tuple[float, float, float]
    velocity_km_s: tuple[float, float, float]
    light_time_seconds: float
    kernels: tuple[KernelIdentity, ...]
    worker_sha256: str
    worker_byte_length: int
    python_implementation: str
    python_version: str
    python_cache_tag: str
    python_executable_path: str
    python_executable_byte_length: int
    python_executable_sha256: str
    spiceypy_version: str
    spiceypy_module_path: str
    spiceypy_module_byte_length: int
    spiceypy_module_sha256: str
    cspice_version: str
    cspice_library_path: str
    cspice_library_byte_length: int
    cspice_library_sha256: str
    source_kind: str = _SOURCE_KIND
    scientific_status: str = _SCIENTIFIC_STATUS
    coverage_status: str = _COVERAGE_STATUS
    integrity_class: str = _INTEGRITY_CLASS
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _ascii(self.request_id, "state request_id", maximum=128)
        _sha256(self.request_sha256, "state request_sha256")
        _ascii(self.bundle_id, "state bundle_id", maximum=128)
        _sha256(self.bundle_sha256, "state bundle_sha256")
        _ascii(self.ephemeris_name, "state ephemeris_name", maximum=128)
        target = _body_payload(self.target, "state target")
        observer = _body_payload(self.observer, "state observer")
        if target == observer:
            raise EphemerisContractError("state target and observer must differ")
        for value, label in (
            (self.target_naif_id, "state target_naif_id"),
            (self.observer_naif_id, "state observer_naif_id"),
        ):
            if type(value) is not int or not -(1 << 31) <= value < (1 << 31):
                raise EphemerisContractError(f"{label} is outside signed 32-bit range")
        if self.target_naif_id == self.observer_naif_id:
            raise EphemerisContractError("state target and observer NAIF IDs must differ")
        if type(self.target) is int and self.target != self.target_naif_id:
            raise EphemerisContractError("integer target does not match target_naif_id")
        if type(self.observer) is int and self.observer != self.observer_naif_id:
            raise EphemerisContractError("integer observer does not match observer_naif_id")
        if type(self.input_epoch) is not EphemerisEpoch:
            raise EphemerisContractError("state input_epoch must be an exact EphemerisEpoch")
        frame = _ascii(self.frame, "state frame", maximum=128)
        if self.aberration_correction not in _ABERRATION_CORRECTIONS:
            raise EphemerisContractError(
                f"state aberration_correction must be one of {sorted(_ABERRATION_CORRECTIONS)}"
            )
        for value, label in (
            (self.position_km, "state position_km"),
            (self.velocity_km_s, "state velocity_km_s"),
        ):
            if type(value) is not tuple or len(value) != 3:
                raise EphemerisExecutionError(f"{label} must be an exact three-element tuple")
        numeric = (
            self.ephemeris_time_seconds,
            *self.position_km,
            *self.velocity_km_s,
            self.light_time_seconds,
        )
        if any(type(value) is not float or not math.isfinite(value) for value in numeric):
            raise EphemerisExecutionError("CSPICE returned a non-finite or non-float state component")
        if self.light_time_seconds < 0.0:
            raise EphemerisExecutionError("CSPICE returned a negative light time")
        if (
            self.input_epoch.format == "SECONDS_PAST_J2000"
            and self.ephemeris_time_seconds != self.input_epoch.value
        ):
            raise EphemerisExecutionError(
                "native TDB result epoch does not equal the requested seconds-past-J2000 epoch"
            )
        if type(self.kernels) is not tuple or not 0 < len(self.kernels) <= _MAX_KERNELS:
            raise EphemerisContractError(
                "state kernels must be a nonempty exact tuple within the kernel-count cap"
            )
        if any(type(item) is not KernelIdentity for item in self.kernels):
            raise EphemerisContractError("every state kernel must be an exact KernelIdentity")
        kernel_names = tuple(item.logical_name for item in self.kernels)
        if len(set(kernel_names)) != len(kernel_names):
            raise EphemerisContractError("state kernel logical names must be unique")
        if not any(item.kind == "SPK" for item in self.kernels):
            raise EphemerisContractError("state kernel identities require at least one SPK")
        _sha256(self.worker_sha256, "state worker_sha256")
        if type(self.worker_byte_length) is not int or not 0 < self.worker_byte_length <= _MAX_WORKER_BYTES:
            raise EphemerisContractError("state worker_byte_length is outside the accepted range")
        _ascii(self.python_implementation, "state python_implementation", maximum=128)
        _ascii(self.python_version, "state python_version", maximum=128)
        _ascii(self.python_cache_tag, "state python_cache_tag", maximum=128)
        _ascii(self.spiceypy_version, "state spiceypy_version", maximum=128)
        _ascii(self.cspice_version, "state cspice_version", maximum=128)
        for value, label in (
            (self.python_executable_path, "state python_executable_path"),
            (self.spiceypy_module_path, "state spiceypy_module_path"),
            (self.cspice_library_path, "state cspice_library_path"),
        ):
            path = _ascii(value, label, maximum=4096)
            if not os.path.isabs(path) or os.path.normpath(path) != path:
                raise EphemerisContractError(f"{label} must be absolute and normalized")
        for value, label in (
            (self.python_executable_byte_length, "state python_executable_byte_length"),
            (self.spiceypy_module_byte_length, "state spiceypy_module_byte_length"),
            (self.cspice_library_byte_length, "state cspice_library_byte_length"),
        ):
            if type(value) is not int or not 0 < value <= _MAX_RUNTIME_FILE_BYTES:
                raise EphemerisContractError(f"{label} is outside the accepted range")
        _sha256(self.python_executable_sha256, "state python_executable_sha256")
        _sha256(self.spiceypy_module_sha256, "state spiceypy_module_sha256")
        _sha256(self.cspice_library_sha256, "state cspice_library_sha256")
        for observed, expected, label in (
            (self.source_kind, _SOURCE_KIND, "source_kind"),
            (self.scientific_status, _SCIENTIFIC_STATUS, "scientific_status"),
            (self.coverage_status, _COVERAGE_STATUS, "coverage_status"),
            (self.integrity_class, _INTEGRITY_CLASS, "integrity_class"),
        ):
            if observed != expected:
                raise EphemerisContractError(f"state {label} must be {expected!r}")

        expected_request_sha256 = _canonical_sha256(
            {
                "aberration_correction": self.aberration_correction,
                "epoch": self.input_epoch._payload(),
                "frame": frame,
                "observer": observer,
                "request_id": self.request_id,
                "schema": "jx.ephemeris-request.v1",
                "target": target,
            }
        )
        if self.request_sha256 != expected_request_sha256:
            raise EphemerisContractError("state request_sha256 does not bind the retained request")
        expected_bundle_sha256 = _canonical_sha256(
            {
                "artifacts": [item._portable_payload() for item in self.kernels],
                "bundle_id": self.bundle_id,
                "ephemeris_name": self.ephemeris_name,
                "provider": "JPL",
                "schema": "jx.jpl-kernel-bundle.v1",
            }
        )
        if self.bundle_sha256 != expected_bundle_sha256:
            raise EphemerisContractError("state bundle_sha256 does not bind the retained kernels")

        runtime = {
            "cspice_library": {
                "byte_length": self.cspice_library_byte_length,
                "path": self.cspice_library_path,
                "sha256": self.cspice_library_sha256,
            },
            "cspice_version": self.cspice_version,
            "python_cache_tag": self.python_cache_tag,
            "python_executable": {
                "byte_length": self.python_executable_byte_length,
                "path": self.python_executable_path,
                "sha256": self.python_executable_sha256,
            },
            "python_implementation": self.python_implementation,
            "python_version": self.python_version,
            "schema": "jx.ephemeris-runtime.v1",
            "spiceypy_module": {
                "byte_length": self.spiceypy_module_byte_length,
                "path": self.spiceypy_module_path,
                "sha256": self.spiceypy_module_sha256,
            },
            "spiceypy_version": self.spiceypy_version,
            "worker": {
                "byte_length": self.worker_byte_length,
                "sha256": self.worker_sha256,
            },
        }
        digest = _canonical_sha256(
            {
                "aberration_correction": self.aberration_correction,
                "bundle_id": self.bundle_id,
                "bundle_sha256": self.bundle_sha256,
                "coverage_status": self.coverage_status,
                "ephemeris_time_seconds": self.ephemeris_time_seconds.hex(),
                "ephemeris_name": self.ephemeris_name,
                "frame": self.frame,
                "input_epoch": self.input_epoch._payload(),
                "integrity_class": self.integrity_class,
                "kernels": [item._portable_payload() for item in self.kernels],
                "light_time_seconds": self.light_time_seconds.hex(),
                "observer": observer,
                "observer_naif_id": self.observer_naif_id,
                "position_km": [value.hex() for value in self.position_km],
                "request_id": self.request_id,
                "request_sha256": self.request_sha256,
                "runtime": runtime,
                "schema": _STATE_SCHEMA,
                "scientific_status": self.scientific_status,
                "source_kind": self.source_kind,
                "target": target,
                "target_naif_id": self.target_naif_id,
                "velocity_km_s": [value.hex() for value in self.velocity_km_s],
            }
        )
        object.__setattr__(self, "content_sha256", digest)


def _hash_fd(fd: int) -> tuple[int, str, tuple[int, int, int, int, int]]:
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
        raise EphemerisKernelError("kernel descriptor is not a regular file")
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    length = 0
    while True:
        block = os.read(fd, 1 << 20)
        if not block:
            break
        length += len(block)
        digest.update(block)
    os.lseek(fd, 0, os.SEEK_SET)
    after = os.fstat(fd)
    identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != identity:
        raise EphemerisKernelError("kernel identity changed while it was hashed")
    return length, digest.hexdigest(), identity


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number {value!r} is forbidden")


def _read_worker_source(path: Path) -> tuple[bytes, str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EphemerisDependencyError("the packaged ephemeris worker is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= _MAX_WORKER_BYTES:
            raise EphemerisDependencyError(
                "the packaged ephemeris worker is not a bounded regular file"
            )
        length, digest, identity = _hash_fd(descriptor)
        source = bytearray()
        while len(source) < length:
            block = os.read(descriptor, min(1 << 20, length - len(source)))
            if not block:
                break
            source.extend(block)
        observed_length, observed_digest, observed_identity = _hash_fd(descriptor)
        if (
            len(source) != length
            or observed_length != length
            or hashlib.sha256(source).hexdigest() != digest
            or observed_digest != digest
            or observed_identity != identity
        ):
            raise EphemerisDependencyError("the packaged ephemeris worker changed while read")
        return bytes(source), digest
    finally:
        os.close(descriptor)


def _sealed_worker_fd(source: bytes) -> int:
    required_names = (
        "F_ADD_SEALS",
        "F_GET_SEALS",
        "F_SEAL_GROW",
        "F_SEAL_SEAL",
        "F_SEAL_SHRINK",
        "F_SEAL_WRITE",
    )
    if (
        fcntl is None
        or not hasattr(os, "memfd_create")
        or not hasattr(os, "MFD_ALLOW_SEALING")
        or any(not hasattr(fcntl, name) for name in required_names)
    ):
        raise EphemerisDependencyError(
            "ephemeris worker custody requires Linux sealed-memory-file support"
        )
    flags = getattr(os, "MFD_CLOEXEC", 0) | os.MFD_ALLOW_SEALING
    try:
        descriptor = os.memfd_create("jxplanetx-ephemeris-worker", flags)
        view = memoryview(source)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("short write while sealing worker source")
            offset += written
        os.lseek(descriptor, 0, os.SEEK_SET)
        seals = (
            fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_SEAL
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_WRITE
        )
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
        if fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) != seals:
            raise OSError("worker memory file did not retain every required seal")
        return descriptor
    except OSError as exc:
        try:
            os.close(descriptor)
        except (OSError, UnboundLocalError):
            pass
        raise EphemerisDependencyError("could not create an immutable worker source copy") from exc


def _open_python_executable() -> tuple[int, str, int, str, tuple[int, int, int, int, int]]:
    executable_path = os.path.realpath(sys.executable)
    _ascii(executable_path, "Python executable path", maximum=4096)
    if not os.path.isabs(executable_path) or os.path.normpath(executable_path) != executable_path:
        raise EphemerisDependencyError("Python executable path is not absolute and normalized")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    try:
        descriptor = os.open(executable_path, flags)
    except OSError as exc:
        raise EphemerisDependencyError("the Python executable cannot be securely opened") from exc
    try:
        length, digest, identity = _hash_fd(descriptor)
        if not 0 < length <= _MAX_RUNTIME_FILE_BYTES:
            raise EphemerisDependencyError("the Python executable is outside the accepted size range")
        if not os.access(executable_path, os.X_OK):
            raise EphemerisDependencyError("the Python executable is not executable")
        return descriptor, executable_path, length, digest, identity
    except Exception:
        os.close(descriptor)
        raise


def _run_bounded_worker(
    *,
    argv: list[str],
    executable: str,
    encoded_request: bytes,
    pass_fds: tuple[int, ...],
    timeout_seconds: float,
) -> tuple[int, bytes, bytes]:
    """Run a worker without ever buffering an unbounded child stream in memory."""

    with tempfile.TemporaryFile(mode="w+b") as stdout_file, tempfile.TemporaryFile(
        mode="w+b"
    ) as stderr_file:
        try:
            process = subprocess.run(
                argv,
                executable=executable,
                input=encoded_request,
                stdout=stdout_file,
                stderr=stderr_file,
                check=False,
                timeout=timeout_seconds,
                pass_fds=pass_fds,
            )
        except subprocess.TimeoutExpired as exc:
            raise EphemerisExecutionError("isolated CSPICE query exceeded its timeout") from exc
        except OSError as exc:
            raise EphemerisDependencyError("isolated CSPICE worker could not start") from exc
        stdout_length = os.fstat(stdout_file.fileno()).st_size
        stderr_length = os.fstat(stderr_file.fileno()).st_size
        if stdout_length > _MAX_WORKER_OUTPUT or stderr_length > _MAX_WORKER_OUTPUT:
            raise EphemerisExecutionError("isolated CSPICE worker exceeded its output cap")
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read(_MAX_WORKER_OUTPUT + 1)
        stderr = stderr_file.read(_MAX_WORKER_OUTPUT + 1)
        if len(stdout) != stdout_length or len(stderr) != stderr_length:
            raise EphemerisExecutionError("isolated CSPICE worker output could not be read exactly")
        return process.returncode, stdout, stderr


class JplEphemeris:
    """Isolated, hash-bound CSPICE evaluator for one ordered JPL kernel bundle."""

    def __init__(self, bundle: KernelBundle, *, timeout_seconds: float = 30.0) -> None:
        if type(bundle) is not KernelBundle:
            raise EphemerisContractError("bundle must be an exact KernelBundle")
        if type(timeout_seconds) is not float or not math.isfinite(timeout_seconds) or not 0.0 < timeout_seconds <= 300.0:
            raise EphemerisContractError("timeout_seconds must be a finite exact float in (0, 300]")
        self._bundle = bundle
        self._timeout_seconds = timeout_seconds

    @property
    def bundle(self) -> KernelBundle:
        return self._bundle

    def state(self, request: EphemerisRequest) -> EphemerisState:
        """Evaluate one state, refusing missing dependencies, kernels, and coverage."""

        if type(request) is not EphemerisRequest:
            raise EphemerisContractError("request must be an exact EphemerisRequest")
        if request.epoch.format == "CALENDAR" and not any(
            artifact.kind == "LSK" for artifact in self._bundle.artifacts
        ):
            raise EphemerisKernelError("calendar epochs require an explicitly bound LSK kernel")
        if (
            sys.platform != "linux"
            or not hasattr(os, "O_NOFOLLOW")
            or not os.path.isdir("/proc/self/fd")
        ):
            raise EphemerisDependencyError(
                "ephemeris API requires Linux O_NOFOLLOW and /proc/self/fd custody"
            )
        if importlib.util.find_spec("spiceypy") is None:
            raise EphemerisDependencyError(
                "spiceypy is unavailable; install jxplanetx[ephemeris]"
            )

        worker_path = Path(__file__).with_name("_ephemeris_worker.py")
        worker_bytes, worker_sha256 = _read_worker_source(worker_path)
        worker_descriptor = _sealed_worker_fd(worker_bytes)
        python_descriptor = -1
        descriptors: list[int] = []
        identities: list[tuple[int, int, int, int, int]] = []
        kernel_rows: list[KernelIdentity] = []
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
        try:
            (
                python_descriptor,
                python_executable_path,
                python_executable_length,
                python_executable_sha256,
                python_executable_identity,
            ) = _open_python_executable()
            for artifact in self._bundle.artifacts:
                try:
                    descriptor = os.open(artifact.path, flags)
                except OSError as exc:
                    raise EphemerisKernelError(
                        f"cannot securely open kernel {artifact.logical_name!r}"
                    ) from exc
                descriptors.append(descriptor)
                length, digest, identity = _hash_fd(descriptor)
                if length != artifact.byte_length or digest != artifact.sha256:
                    raise EphemerisKernelError(
                        f"kernel {artifact.logical_name!r} differs from its expected byte identity"
                    )
                identities.append(identity)
                kernel_rows.append(
                    KernelIdentity(
                        logical_name=artifact.logical_name,
                        kind=artifact.kind,
                        byte_length=length,
                        sha256=digest,
                    )
                )

            payload = {
                "kernel_fds": descriptors,
                "python_executable_byte_length": python_executable_length,
                "python_executable_fd": python_descriptor,
                "python_executable_path": python_executable_path,
                "python_executable_sha256": python_executable_sha256,
                "request": request._worker_payload(),
                "schema": _WORKER_REQUEST_SCHEMA,
                "worker_source_byte_length": len(worker_bytes),
                "worker_source_fd": worker_descriptor,
                "worker_source_sha256": worker_sha256,
            }
            encoded = json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            return_code, stdout, stderr = _run_bounded_worker(
                argv=[sys.executable, "-I", f"/proc/self/fd/{worker_descriptor}"],
                executable=f"/proc/self/fd/{python_descriptor}",
                encoded_request=encoded,
                pass_fds=tuple(descriptors) + (worker_descriptor, python_descriptor),
                timeout_seconds=self._timeout_seconds,
            )

            for index, descriptor in enumerate(descriptors):
                length, digest, identity = _hash_fd(descriptor)
                artifact = self._bundle.artifacts[index]
                if (
                    identity != identities[index]
                    or length != artifact.byte_length
                    or digest != artifact.sha256
                ):
                    raise EphemerisKernelError(
                        f"kernel {artifact.logical_name!r} changed during CSPICE execution"
                    )
            worker_length_after, worker_sha256_after, _ = _hash_fd(worker_descriptor)
            if worker_length_after != len(worker_bytes) or worker_sha256_after != worker_sha256:
                raise EphemerisDependencyError("sealed worker source changed during execution")
            (
                python_length_after,
                python_sha256_after,
                python_identity_after,
            ) = _hash_fd(python_descriptor)
            if (
                python_length_after != python_executable_length
                or python_sha256_after != python_executable_sha256
                or python_identity_after != python_executable_identity
            ):
                raise EphemerisDependencyError("Python executable changed during execution")

            if stderr:
                raise EphemerisExecutionError("isolated CSPICE worker wrote unexpected stderr")
            try:
                response = json.loads(stdout.decode("ascii"), parse_constant=_reject_json_constant)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise EphemerisExecutionError("isolated CSPICE worker returned invalid JSON") from exc
            if type(response) is not dict or response.get("schema") != _WORKER_RESPONSE_SCHEMA:
                raise EphemerisExecutionError("isolated CSPICE worker returned the wrong schema")
            if return_code != 0 or response.get("ok") is not True:
                if set(response) != {"category", "message", "ok", "schema"}:
                    raise EphemerisExecutionError(
                        "isolated CSPICE failure response has unexpected fields"
                    )
                category = response.get("category")
                message = response.get("message")
                if type(message) is not str or not message:
                    message = "isolated CSPICE worker failed"
                if category == "COVERAGE":
                    raise EphemerisCoverageError(message)
                if category == "DEPENDENCY":
                    raise EphemerisDependencyError(message)
                if category in {"KERNEL", "TIME"}:
                    raise EphemerisKernelError(message)
                raise EphemerisExecutionError(message)

            expected_keys = {
                "ephemeris_time_seconds",
                "light_time_seconds",
                "observer_naif_id",
                "ok",
                "runtime",
                "schema",
                "state",
                "target_naif_id",
            }
            if set(response) != expected_keys:
                raise EphemerisExecutionError("isolated CSPICE success response has unexpected fields")
            raw_state = response["state"]
            if type(raw_state) is not list or len(raw_state) != 6:
                raise EphemerisExecutionError("isolated CSPICE state must contain six components")
            if any(type(value) is not float or not math.isfinite(value) for value in raw_state):
                raise EphemerisExecutionError("isolated CSPICE state contains a non-finite exact float")
            raw_et = response["ephemeris_time_seconds"]
            raw_light_time = response["light_time_seconds"]
            if (
                type(raw_et) is not float
                or type(raw_light_time) is not float
                or not math.isfinite(raw_et)
                or not math.isfinite(raw_light_time)
            ):
                raise EphemerisExecutionError("isolated CSPICE returned invalid exact-float time values")
            state = tuple(raw_state)
            target_id = response["target_naif_id"]
            observer_id = response["observer_naif_id"]
            if type(target_id) is not int or type(observer_id) is not int or target_id == observer_id:
                raise EphemerisExecutionError("isolated CSPICE returned invalid body identifiers")
            runtime = response["runtime"]
            expected_runtime_keys = {
                "cspice_library",
                "cspice_version",
                "python_cache_tag",
                "python_executable",
                "python_implementation",
                "python_version",
                "schema",
                "spiceypy_module",
                "spiceypy_version",
                "worker",
            }
            if type(runtime) is not dict or set(runtime) != expected_runtime_keys:
                raise EphemerisExecutionError("isolated CSPICE runtime identity is malformed")
            if runtime.get("schema") != "jx.ephemeris-runtime.v1":
                raise EphemerisExecutionError("isolated CSPICE runtime identity has the wrong schema")
            expected_file_keys = {"byte_length", "path", "sha256"}
            for key in ("python_executable", "spiceypy_module", "cspice_library"):
                if type(runtime[key]) is not dict or set(runtime[key]) != expected_file_keys:
                    raise EphemerisExecutionError(
                        f"isolated CSPICE runtime {key} identity is malformed"
                    )
            if type(runtime["worker"]) is not dict or set(runtime["worker"]) != {
                "byte_length",
                "sha256",
            }:
                raise EphemerisExecutionError("isolated CSPICE worker identity is malformed")
            if runtime["worker"] != {
                "byte_length": len(worker_bytes),
                "sha256": worker_sha256,
            }:
                raise EphemerisExecutionError("worker did not confirm its sealed source identity")
            if runtime["python_executable"] != {
                "byte_length": python_executable_length,
                "path": python_executable_path,
                "sha256": python_executable_sha256,
            }:
                raise EphemerisExecutionError("worker did not confirm the executed Python binary")
            return EphemerisState(
                request_id=request.request_id,
                request_sha256=request.content_sha256,
                bundle_id=self._bundle.bundle_id,
                bundle_sha256=self._bundle.content_sha256,
                ephemeris_name=self._bundle.ephemeris_name,
                target=request.target,
                target_naif_id=target_id,
                observer=request.observer,
                observer_naif_id=observer_id,
                input_epoch=request.epoch,
                ephemeris_time_seconds=float(raw_et),
                frame=request.frame,
                aberration_correction=request.aberration_correction,
                position_km=(state[0], state[1], state[2]),
                velocity_km_s=(state[3], state[4], state[5]),
                light_time_seconds=raw_light_time,
                kernels=tuple(kernel_rows),
                worker_sha256=worker_sha256,
                worker_byte_length=len(worker_bytes),
                python_implementation=runtime["python_implementation"],
                python_version=runtime["python_version"],
                python_cache_tag=runtime["python_cache_tag"],
                python_executable_path=runtime["python_executable"]["path"],
                python_executable_byte_length=runtime["python_executable"]["byte_length"],
                python_executable_sha256=runtime["python_executable"]["sha256"],
                spiceypy_version=runtime["spiceypy_version"],
                spiceypy_module_path=runtime["spiceypy_module"]["path"],
                spiceypy_module_byte_length=runtime["spiceypy_module"]["byte_length"],
                spiceypy_module_sha256=runtime["spiceypy_module"]["sha256"],
                cspice_version=runtime["cspice_version"],
                cspice_library_path=runtime["cspice_library"]["path"],
                cspice_library_byte_length=runtime["cspice_library"]["byte_length"],
                cspice_library_sha256=runtime["cspice_library"]["sha256"],
            )
        finally:
            for descriptor in descriptors:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if python_descriptor >= 0:
                try:
                    os.close(python_descriptor)
                except OSError:
                    pass
            try:
                os.close(worker_descriptor)
            except OSError:
                pass


__all__ = [
    "EphemerisContractError",
    "EphemerisCoverageError",
    "EphemerisDependencyError",
    "EphemerisEpoch",
    "EphemerisError",
    "EphemerisExecutionError",
    "EphemerisKernelError",
    "EphemerisRequest",
    "EphemerisState",
    "JplEphemeris",
    "KernelArtifact",
    "KernelBundle",
    "KernelIdentity",
]
