"""Fail-closed JPL-kernel ephemeris API candidate.

This module evaluates states from caller-supplied NAIF kernels.  It does not
integrate a JX dynamical model and must not be represented as an independently
generated JX ephemeris.  Every kernel is bound by an expected byte length and
SHA-256 digest, opened without following a final-component symlink, and checked
before and after an isolated CSPICE worker process executes the query.

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
from typing import Final


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
        if not -(1 << 31) < value < (1 << 31):
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
    logical_name: str
    kind: str
    byte_length: int
    sha256: str


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
    source_kind: str = "JPL_KERNEL_INTERPOLATION"
    scientific_status: str = "PROVIDER_DERIVED_STATE_NOT_JX_NATIVE_EPHEMERIS"
    coverage_status: str = "CSPICE_QUERY_SUCCEEDED_AT_REQUESTED_EPOCH"
    integrity_class: str = "SHA256_CONTENT_INTEGRITY_NOT_SOURCE_AUTHENTICATION"
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
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
        digest = _canonical_sha256(
            {
                "aberration_correction": self.aberration_correction,
                "bundle_sha256": self.bundle_sha256,
                "coverage_status": self.coverage_status,
                "ephemeris_time_seconds": self.ephemeris_time_seconds.hex(),
                "frame": self.frame,
                "integrity_class": self.integrity_class,
                "light_time_seconds": self.light_time_seconds.hex(),
                "observer_naif_id": self.observer_naif_id,
                "position_km": [value.hex() for value in self.position_km],
                "request_sha256": self.request_sha256,
                "schema": "jx.ephemeris-state.v1",
                "scientific_status": self.scientific_status,
                "source_kind": self.source_kind,
                "target_naif_id": self.target_naif_id,
                "velocity_km_s": [value.hex() for value in self.velocity_km_s],
                "worker_sha256": self.worker_sha256,
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
                "ephemeris API v1 requires Linux O_NOFOLLOW and /proc/self/fd custody"
            )
        if importlib.util.find_spec("spiceypy") is None:
            raise EphemerisDependencyError(
                "spiceypy is unavailable; install jxplanetx[ephemeris]"
            )

        worker_path = Path(__file__).with_name("_ephemeris_worker.py")
        try:
            worker_bytes = worker_path.read_bytes()
        except OSError as exc:
            raise EphemerisDependencyError("the packaged ephemeris worker is unavailable") from exc
        worker_sha256 = hashlib.sha256(worker_bytes).hexdigest()

        descriptors: list[int] = []
        identities: list[tuple[int, int, int, int, int]] = []
        kernel_rows: list[KernelIdentity] = []
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
        try:
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
                "request": request._worker_payload(),
                "schema": "jx.ephemeris-worker-request.v1",
            }
            encoded = json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            try:
                process = subprocess.run(
                    [sys.executable, "-I", str(worker_path)],
                    input=encoded,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=self._timeout_seconds,
                    pass_fds=tuple(descriptors),
                )
            except subprocess.TimeoutExpired as exc:
                raise EphemerisExecutionError("isolated CSPICE query exceeded its timeout") from exc
            except OSError as exc:
                raise EphemerisDependencyError("isolated CSPICE worker could not start") from exc

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

            if len(process.stdout) > _MAX_WORKER_OUTPUT or len(process.stderr) > _MAX_WORKER_OUTPUT:
                raise EphemerisExecutionError("isolated CSPICE worker exceeded its output cap")
            try:
                response = json.loads(process.stdout.decode("ascii"), parse_constant=_reject_json_constant)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise EphemerisExecutionError("isolated CSPICE worker returned invalid JSON") from exc
            if type(response) is not dict or response.get("schema") != "jx.ephemeris-worker-response.v1":
                raise EphemerisExecutionError("isolated CSPICE worker returned the wrong schema")
            if process.returncode != 0 or response.get("ok") is not True:
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
                "schema",
                "state",
                "target_naif_id",
            }
            if set(response) != expected_keys:
                raise EphemerisExecutionError("isolated CSPICE success response has unexpected fields")
            raw_state = response["state"]
            if type(raw_state) is not list or len(raw_state) != 6:
                raise EphemerisExecutionError("isolated CSPICE state must contain six components")
            if any(type(value) not in (int, float) for value in raw_state):
                raise EphemerisExecutionError("isolated CSPICE state contains a non-number")
            raw_et = response["ephemeris_time_seconds"]
            raw_light_time = response["light_time_seconds"]
            if type(raw_et) not in (int, float) or type(raw_light_time) not in (int, float):
                raise EphemerisExecutionError("isolated CSPICE returned invalid time values")
            state = tuple(float(value) for value in raw_state)
            target_id = response["target_naif_id"]
            observer_id = response["observer_naif_id"]
            if type(target_id) is not int or type(observer_id) is not int or target_id == observer_id:
                raise EphemerisExecutionError("isolated CSPICE returned invalid body identifiers")
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
                light_time_seconds=float(raw_light_time),
                kernels=tuple(kernel_rows),
                worker_sha256=worker_sha256,
            )
        finally:
            for descriptor in descriptors:
                try:
                    os.close(descriptor)
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
