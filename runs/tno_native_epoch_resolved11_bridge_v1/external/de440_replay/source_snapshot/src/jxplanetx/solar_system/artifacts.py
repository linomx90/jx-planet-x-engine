"""Fail-closed local artifact-byte observation without provider execution.

This unpublished POSIX capability profile verifies one exact, sealed M1
``ArtifactBinding`` against bytes read from one final open file description.
The caller supplies a directory file descriptor; this module duplicates it,
but neither authenticates how it was acquired nor retains it after return.

The result separates an exact declared-byte match from nonportable local
filesystem observations.  It is not an atomic filesystem snapshot, proof of
continuous pathname or byte immutability, exhaustive adversarial-TOCTOU
detection, trusted-clock or physical-local-storage evidence, exclusive
ancestry, or hard-link/mount/bind provenance.  The second pathname lookup is
only a later namespace observation.  POSIX offers no portable regular-only
open, so hostile device-node namespaces can still make even a nonblocking,
no-controlling-terminal open have implementation-specific side effects.
Successful directory walks and file reads may update access metadata; atime is
therefore deliberately absent from the retained stability tuple.
The byte and call caps do not bound latency or guarantee termination: regular
files and remote, FUSE, or other unusual filesystems may still block in POSIX
metadata or read operations despite ``O_NONBLOCK``.

No returned receipt binds future path or descriptor use, artifact-at-use
custody, provider consumption or execution, authenticity or signature,
license or redistribution authority, SPK validity or coverage, interpolation
accuracy, network/fallback/extrapolation enforcement, registry status, or
qualification.  Both receipt seals are forgeable unauthenticated content
integrity only.  A fresh current-state check requires a new call with a fresh
root capability.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import stat
from contextlib import ExitStack
from dataclasses import dataclass, fields
from typing import ClassVar

from .contracts import (
    ArtifactBinding,
    SolarSystemContractError,
    SolarSystemDataError,
    SolarSystemDependencyUnavailableError,
    validate_integrity,
)
from .serialization import domain_sha256, validate_sha256


MAXIMUM_LOCAL_ARTIFACT_BYTES = 1 << 30

_MAXIMUM_LOCATOR_COMPONENTS = 64
_READ_CHUNK_BYTE_LIMIT = 1 << 20
_MAXIMUM_DATA_READ_CALLS = 8_192
_MAXIMUM_ROOT_DIRECTORY_FD = (1 << 31) - 1
_MAXIMUM_STAT_INTEGER_BITS = 128
_MAXIMUM_LOCATOR_COMPONENT_CODEPOINTS = 256
_MAXIMUM_TOKEN_CODEPOINTS = 256

_CONTENT_INTEGRITY_CLASS = "UNAUTHENTICATED_CONTENT_INTEGRITY_ONLY"

_BYTE_VERIFICATION_SCOPE = "EXACT_DECLARED_ARTIFACT_BYTES_ONE_READ_STREAM"
_HASH_ALGORITHM = "SHA-256"
_BYTE_MATCH_STATUS = "OBSERVED_LENGTH_AND_SHA256_EQUAL_DECLARATION"

_VERIFICATION_STAGE = "LOCAL_BYTE_STREAM_VERIFICATION_NO_PROVIDER_EXECUTION"
_PLATFORM_SCOPE = "CAPABILITY_GATED_POSIX_PROFILE_NO_PORTABILITY_CLAIM"
_ROOT_CAPABILITY_STATUS = (
    "CALLER_DIRECTORY_FD_DUPLICATED_ROOT_PATH_ACQUISITION_UNATTESTED"
)
_LOCATOR_RESOLUTION_POLICY = "SEQUENTIAL_COMPONENTWISE_DIR_FD_O_NOFOLLOW"
_FINAL_OPEN_POLICY = "ONE_FINAL_OPEN_DESCRIPTION_READ_ONLY_NONBLOCKING_NOCTTY"
_READ_POLICY = "BOUNDED_OS_READ_DECLARED_LENGTH_PLUS_ONE_EOF_PROBE"
_METADATA_STABILITY_STATUS = (
    "SELECTED_PREOPEN_PRE_READ_POST_READ_AND_SECOND_STAT_FIELDS_IDENTICAL"
)
_SECOND_RESOLUTION_STATUS = (
    "SECOND_NAMESPACE_OBSERVATION_MATCHED_OPEN_FILE_IDENTITY_NOT_CONTINUITY_PROOF"
)
_LOCAL_METADATA_CLASSIFICATION = (
    "LOCAL_NONPORTABLE_OBSERVATION_NOT_CONTENT_IDENTITY"
)
_CUSTODY_SCOPE = "READ_STREAM_ONLY_NO_FUTURE_PATH_OR_FD_USE_AUTHORITY"
_EXECUTION_EVIDENCE_STATUS = (
    "LOCAL_VERIFIER_EXECUTION_ONLY_NO_PROVIDER_AT_USE_OR_PROVIDER_EXECUTION"
)
_SEMANTIC_REPLAY_STATUS = "REQUIRES_FRESH_VERIFICATION_WITH_NEW_ROOT_CAPABILITY"

_ROOT_STAT_FIELD_ORDER = ("st_dev", "st_ino", "st_mode")
_FILE_STAT_FIELD_ORDER = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_uid",
    "st_gid",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)

_BYTE_MATCH_DOMAIN = (
    "jxplanetx.solar-system.artifacts.artifact-byte-match-receipt."
    "content-integrity.v1"
)
_BYTE_MATCH_SCHEMA = "ArtifactByteMatchReceipt.v1"
_BYTE_MATCH_QUALIFIED_NAME = (
    "jxplanetx.solar_system.artifacts.ArtifactByteMatchReceipt"
)
_LOCAL_RECEIPT_DOMAIN = (
    "jxplanetx.solar-system.artifacts.local-artifact-verification-receipt."
    "content-integrity.v1"
)
_LOCAL_RECEIPT_SCHEMA = "LocalArtifactVerificationReceipt.v1"
_LOCAL_RECEIPT_QUALIFIED_NAME = (
    "jxplanetx.solar_system.artifacts.LocalArtifactVerificationReceipt"
)


def _preflight_optional_sha256(value: object, label: str) -> str:
    if type(value) is not str:
        raise SolarSystemContractError(f"{label} must be an exact string")
    if value != "":
        validate_sha256(value, label)
    return value


def _token(value: object, expected: str, label: str) -> str:
    if type(value) is not str:
        raise SolarSystemContractError(f"{label} must be an exact string")
    if len(value) > _MAXIMUM_TOKEN_CODEPOINTS:
        raise SolarSystemContractError(f"{label} exceeds the token cap")
    if value != expected:
        raise SolarSystemContractError(f"{label} is not the exact required token")
    return value


def _exact_integer(
    value: object,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int:
        raise SolarSystemContractError(f"{label} must be an exact integer")
    if value < minimum or value > maximum:
        raise SolarSystemContractError(f"{label} is outside its exact bounds")
    return value


def _stat_integer(value: object, label: str, *, signed: bool = False) -> int:
    if type(value) is not int:
        raise SolarSystemContractError(f"{label} must be an exact integer")
    if not signed and value < 0:
        raise SolarSystemContractError(f"{label} must be nonnegative")
    if value.bit_length() > _MAXIMUM_STAT_INTEGER_BITS:
        raise SolarSystemContractError(f"{label} exceeds the stat-integer cap")
    return value


def _locator_components(artifact: ArtifactBinding) -> tuple[str, ...]:
    locator = artifact.logical_locator
    components = tuple(locator.split("/"))
    if (
        not components
        or len(components) > _MAXIMUM_LOCATOR_COMPONENTS
        or any(
            not component
            or component in (".", "..")
            or "/" in component
            or len(component) > _MAXIMUM_LOCATOR_COMPONENT_CODEPOINTS
            or any(ord(character) < 0x20 or ord(character) > 0x7E for character in component)
            for component in components
        )
        or "/".join(components) != locator
    ):
        raise SolarSystemContractError(
            "logical_locator is outside the exact local component profile"
        )
    return components


def _validate_artifact_for_local_verification(
    artifact: object,
) -> tuple[ArtifactBinding, tuple[str, ...]]:
    if type(artifact) is not ArtifactBinding:
        raise SolarSystemContractError("artifact must be an exact ArtifactBinding")
    validate_integrity(artifact)
    if artifact.locator_kind != "LOCAL_REGULAR_FILE":
        raise SolarSystemContractError(
            "local artifact verification requires LOCAL_REGULAR_FILE"
        )
    if artifact.byte_length > MAXIMUM_LOCAL_ARTIFACT_BYTES:
        raise SolarSystemContractError(
            "artifact declared byte length exceeds the fixed local profile cap"
        )
    return artifact, _locator_components(artifact)


def _byte_match_payload(value: ArtifactByteMatchReceipt) -> tuple[object, ...]:
    return (
        _BYTE_MATCH_QUALIFIED_NAME,
        tuple(
            (descriptor.name, getattr(value, descriptor.name))
            for descriptor in fields(value)
            if descriptor.name != "content_sha256"
        ),
    )


def _byte_match_sealed_payload(
    value: ArtifactByteMatchReceipt,
) -> tuple[object, ...]:
    return (
        _byte_match_payload(value),
        ("content_sha256", value.content_sha256),
    )


def _byte_match_digest(value: ArtifactByteMatchReceipt) -> str:
    return domain_sha256(
        _BYTE_MATCH_DOMAIN,
        _BYTE_MATCH_SCHEMA,
        _byte_match_payload(value),
    )


def _local_receipt_payload(
    value: LocalArtifactVerificationReceipt,
) -> tuple[object, ...]:
    return (
        _LOCAL_RECEIPT_QUALIFIED_NAME,
        tuple(
            (
                descriptor.name,
                _byte_match_sealed_payload(value.byte_match)
                if descriptor.name == "byte_match"
                else getattr(value, descriptor.name),
            )
            for descriptor in fields(value)
            if descriptor.name != "content_sha256"
        ),
    )


def _local_receipt_digest(value: LocalArtifactVerificationReceipt) -> str:
    return domain_sha256(
        _LOCAL_RECEIPT_DOMAIN,
        _LOCAL_RECEIPT_SCHEMA,
        _local_receipt_payload(value),
    )


@dataclass(frozen=True, slots=True, eq=False)
class ArtifactByteMatchReceipt:
    """Exact declared-byte match, independent of local filesystem metadata."""

    artifact: ArtifactBinding
    verification_scope: str
    hash_algorithm: str
    expected_byte_length: int
    observed_byte_length: int
    expected_sha256: str
    observed_sha256: str
    byte_match_status: str
    content_integrity_class: str
    content_sha256: str = ""

    _DOMAIN: ClassVar[str] = _BYTE_MATCH_DOMAIN
    _SCHEMA: ClassVar[str] = _BYTE_MATCH_SCHEMA

    def __post_init__(self) -> None:
        if type(self) is not ArtifactByteMatchReceipt:
            raise SolarSystemContractError(
                "byte-match receipt must have its exact concrete type"
            )
        _preflight_optional_sha256(self.content_sha256, "content_sha256")
        self._validate()
        expected = _byte_match_digest(self)
        if self.content_sha256 == "":
            object.__setattr__(self, "content_sha256", expected)
        elif self.content_sha256 != expected:
            raise SolarSystemContractError(
                "content_sha256 does not bind the exact byte-match receipt"
            )

    def _validate(self) -> None:
        if type(self) is not ArtifactByteMatchReceipt:
            raise SolarSystemContractError(
                "byte-match receipt must have its exact concrete type"
            )
        artifact, _ = _validate_artifact_for_local_verification(self.artifact)
        _token(self.verification_scope, _BYTE_VERIFICATION_SCOPE, "verification_scope")
        _token(self.hash_algorithm, _HASH_ALGORITHM, "hash_algorithm")
        expected_length = _exact_integer(
            self.expected_byte_length,
            "expected_byte_length",
            minimum=1,
            maximum=MAXIMUM_LOCAL_ARTIFACT_BYTES,
        )
        observed_length = _exact_integer(
            self.observed_byte_length,
            "observed_byte_length",
            minimum=1,
            maximum=MAXIMUM_LOCAL_ARTIFACT_BYTES,
        )
        expected_digest = validate_sha256(self.expected_sha256, "expected_sha256")
        observed_digest = validate_sha256(self.observed_sha256, "observed_sha256")
        _token(self.byte_match_status, _BYTE_MATCH_STATUS, "byte_match_status")
        _token(
            self.content_integrity_class,
            _CONTENT_INTEGRITY_CLASS,
            "content_integrity_class",
        )
        if expected_length != artifact.byte_length or observed_length != expected_length:
            raise SolarSystemContractError(
                "receipt lengths must equal the exact artifact declaration"
            )
        if expected_digest != artifact.artifact_sha256 or observed_digest != expected_digest:
            raise SolarSystemContractError(
                "receipt digests must equal the exact artifact declaration"
            )

    def validate_integrity(self) -> None:
        """Revalidate schema, nested M1 seal, cross-bindings, and own seal."""

        if type(self) is not ArtifactByteMatchReceipt:
            raise SolarSystemContractError(
                "byte-match receipt must have its exact concrete type"
            )
        validate_sha256(self.content_sha256, "content_sha256")
        self._validate()
        if self.content_sha256 != _byte_match_digest(self):
            raise SolarSystemContractError(
                "content_sha256 does not bind the exact byte-match receipt"
            )


@dataclass(frozen=True, slots=True, eq=False)
class LocalArtifactVerificationReceipt:
    """One local observation containing a sealed declared-byte match."""

    byte_match: ArtifactByteMatchReceipt
    verification_stage: str
    platform_scope: str
    root_capability_status: str
    locator_resolution_policy: str
    locator_components: tuple[str, ...]
    final_open_policy: str
    read_policy: str
    maximum_byte_length: int
    read_chunk_byte_limit: int
    maximum_data_read_calls: int
    data_read_call_count: int
    eof_probe_call_count: int
    root_stat_field_order: tuple[str, str, str]
    pre_root_stat: tuple[int, int, int]
    post_root_stat: tuple[int, int, int]
    file_stat_field_order: tuple[str, ...]
    pre_open_path_file_stat: tuple[int, ...]
    pre_read_file_stat: tuple[int, ...]
    post_read_file_stat: tuple[int, ...]
    second_resolution_file_stat: tuple[int, ...]
    metadata_stability_status: str
    second_resolution_status: str
    local_metadata_classification: str
    custody_scope: str
    execution_evidence_status: str
    semantic_replay_status: str
    content_integrity_class: str
    content_sha256: str = ""

    _DOMAIN: ClassVar[str] = _LOCAL_RECEIPT_DOMAIN
    _SCHEMA: ClassVar[str] = _LOCAL_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if type(self) is not LocalArtifactVerificationReceipt:
            raise SolarSystemContractError(
                "local verification receipt must have its exact concrete type"
            )
        _preflight_optional_sha256(self.content_sha256, "content_sha256")
        self._validate()
        expected = _local_receipt_digest(self)
        if self.content_sha256 == "":
            object.__setattr__(self, "content_sha256", expected)
        elif self.content_sha256 != expected:
            raise SolarSystemContractError(
                "content_sha256 does not bind the exact local verification receipt"
            )

    def _validate(self) -> None:
        if type(self) is not LocalArtifactVerificationReceipt:
            raise SolarSystemContractError(
                "local verification receipt must have its exact concrete type"
            )
        if type(self.byte_match) is not ArtifactByteMatchReceipt:
            raise SolarSystemContractError(
                "byte_match must be an exact ArtifactByteMatchReceipt"
            )
        self.byte_match.validate_integrity()
        _token(self.verification_stage, _VERIFICATION_STAGE, "verification_stage")
        _token(self.platform_scope, _PLATFORM_SCOPE, "platform_scope")
        _token(
            self.root_capability_status,
            _ROOT_CAPABILITY_STATUS,
            "root_capability_status",
        )
        _token(
            self.locator_resolution_policy,
            _LOCATOR_RESOLUTION_POLICY,
            "locator_resolution_policy",
        )
        if type(self.locator_components) is not tuple:
            raise SolarSystemContractError("locator_components must be an exact tuple")
        if not self.locator_components or len(self.locator_components) > _MAXIMUM_LOCATOR_COMPONENTS:
            raise SolarSystemContractError(
                "locator_components is outside the exact component-count cap"
            )
        for index, component in enumerate(self.locator_components):
            if type(component) is not str:
                raise SolarSystemContractError(
                    f"locator_components[{index}] must be an exact string"
                )
            if len(component) > _MAXIMUM_LOCATOR_COMPONENT_CODEPOINTS:
                raise SolarSystemContractError(
                    f"locator_components[{index}] exceeds the component text cap"
                )
            if (
                not component
                or component in (".", "..")
                or "/" in component
                or any(
                    ord(character) < 0x20 or ord(character) > 0x7E
                    for character in component
                )
            ):
                raise SolarSystemContractError(
                    f"locator_components[{index}] is not a normalized ASCII component"
                )
        expected_components = _locator_components(self.byte_match.artifact)
        if self.locator_components != expected_components:
            raise SolarSystemContractError(
                "locator_components must exactly reconstruct the artifact locator"
            )
        _token(self.final_open_policy, _FINAL_OPEN_POLICY, "final_open_policy")
        _token(self.read_policy, _READ_POLICY, "read_policy")
        _exact_integer(
            self.maximum_byte_length,
            "maximum_byte_length",
            minimum=MAXIMUM_LOCAL_ARTIFACT_BYTES,
            maximum=MAXIMUM_LOCAL_ARTIFACT_BYTES,
        )
        _exact_integer(
            self.read_chunk_byte_limit,
            "read_chunk_byte_limit",
            minimum=_READ_CHUNK_BYTE_LIMIT,
            maximum=_READ_CHUNK_BYTE_LIMIT,
        )
        _exact_integer(
            self.maximum_data_read_calls,
            "maximum_data_read_calls",
            minimum=_MAXIMUM_DATA_READ_CALLS,
            maximum=_MAXIMUM_DATA_READ_CALLS,
        )
        data_calls = _exact_integer(
            self.data_read_call_count,
            "data_read_call_count",
            minimum=1,
            maximum=_MAXIMUM_DATA_READ_CALLS,
        )
        _exact_integer(
            self.eof_probe_call_count,
            "eof_probe_call_count",
            minimum=1,
            maximum=1,
        )
        minimum_calls = (
            self.byte_match.observed_byte_length + _READ_CHUNK_BYTE_LIMIT - 1
        ) // _READ_CHUNK_BYTE_LIMIT
        if data_calls < minimum_calls:
            raise SolarSystemContractError(
                "data_read_call_count is below the exact request-size lower bound"
            )
        if data_calls > self.byte_match.observed_byte_length:
            raise SolarSystemContractError(
                "data_read_call_count cannot exceed the nonempty observed byte count"
            )

        if (
            type(self.root_stat_field_order) is not tuple
            or len(self.root_stat_field_order) != len(_ROOT_STAT_FIELD_ORDER)
        ):
            raise SolarSystemContractError(
                "root_stat_field_order must be an exact fixed-length tuple"
            )
        for index, expected_name in enumerate(_ROOT_STAT_FIELD_ORDER):
            _token(
                self.root_stat_field_order[index],
                expected_name,
                f"root_stat_field_order[{index}]",
            )
        pre_root = _validate_stat_tuple(
            self.pre_root_stat,
            _ROOT_STAT_FIELD_ORDER,
            "pre_root_stat",
        )
        post_root = _validate_stat_tuple(
            self.post_root_stat,
            _ROOT_STAT_FIELD_ORDER,
            "post_root_stat",
        )
        if pre_root != post_root or not stat.S_ISDIR(pre_root[2]):
            raise SolarSystemContractError(
                "root stat observations must be identical directories"
            )

        if (
            type(self.file_stat_field_order) is not tuple
            or len(self.file_stat_field_order) != len(_FILE_STAT_FIELD_ORDER)
        ):
            raise SolarSystemContractError(
                "file_stat_field_order must be an exact fixed-length tuple"
            )
        for index, expected_name in enumerate(_FILE_STAT_FIELD_ORDER):
            _token(
                self.file_stat_field_order[index],
                expected_name,
                f"file_stat_field_order[{index}]",
            )
        file_stats = tuple(
            _validate_stat_tuple(value, _FILE_STAT_FIELD_ORDER, label)
            for value, label in (
                (self.pre_open_path_file_stat, "pre_open_path_file_stat"),
                (self.pre_read_file_stat, "pre_read_file_stat"),
                (self.post_read_file_stat, "post_read_file_stat"),
                (self.second_resolution_file_stat, "second_resolution_file_stat"),
            )
        )
        if any(value != file_stats[0] for value in file_stats[1:]):
            raise SolarSystemContractError(
                "all selected path and file stat observations must be identical"
            )
        if not stat.S_ISREG(file_stats[0][2]):
            raise SolarSystemContractError("all file observations must be regular files")
        if file_stats[0][3] < 1:
            raise SolarSystemContractError("a retained regular file requires st_nlink >= 1")
        if file_stats[0][6] != self.byte_match.observed_byte_length:
            raise SolarSystemContractError(
                "file stat byte length must equal the byte-match receipt"
            )
        if pre_root[:2] == file_stats[0][:2]:
            raise SolarSystemContractError(
                "root directory and regular file cannot share one live inode identity"
            )
        _token(
            self.metadata_stability_status,
            _METADATA_STABILITY_STATUS,
            "metadata_stability_status",
        )
        _token(
            self.second_resolution_status,
            _SECOND_RESOLUTION_STATUS,
            "second_resolution_status",
        )
        _token(
            self.local_metadata_classification,
            _LOCAL_METADATA_CLASSIFICATION,
            "local_metadata_classification",
        )
        _token(self.custody_scope, _CUSTODY_SCOPE, "custody_scope")
        _token(
            self.execution_evidence_status,
            _EXECUTION_EVIDENCE_STATUS,
            "execution_evidence_status",
        )
        _token(
            self.semantic_replay_status,
            _SEMANTIC_REPLAY_STATUS,
            "semantic_replay_status",
        )
        _token(
            self.content_integrity_class,
            _CONTENT_INTEGRITY_CLASS,
            "content_integrity_class",
        )

    def validate_integrity(self) -> None:
        """Revalidate schema, expanded child seal, cross-rules, and own seal."""

        if type(self) is not LocalArtifactVerificationReceipt:
            raise SolarSystemContractError(
                "local verification receipt must have its exact concrete type"
            )
        validate_sha256(self.content_sha256, "content_sha256")
        self._validate()
        if self.content_sha256 != _local_receipt_digest(self):
            raise SolarSystemContractError(
                "content_sha256 does not bind the exact local verification receipt"
            )


def _validate_stat_tuple(
    value: object,
    field_order: tuple[str, ...],
    label: str,
) -> tuple[int, ...]:
    if type(value) is not tuple or len(value) != len(field_order):
        raise SolarSystemContractError(
            f"{label} must be an exact tuple matching its stat-field roster"
        )
    result: list[int] = []
    for index, (component, field_name) in enumerate(zip(value, field_order)):
        signed = field_name in ("st_mtime_ns", "st_ctime_ns")
        result.append(
            _stat_integer(
                component,
                f"{label}[{index}]/{field_name}",
                signed=signed,
            )
        )
    return tuple(result)


def _selected_stat(value: os.stat_result, field_order: tuple[str, ...]) -> tuple[int, ...]:
    result = tuple(getattr(value, field_name) for field_name in field_order)
    try:
        return _validate_stat_tuple(result, field_order, "runtime stat observation")
    except SolarSystemContractError:
        raise SolarSystemDataError(
            "runtime stat observation is outside the retained profile"
        ) from None


def _require_platform_capabilities() -> tuple[int, int]:
    if os.name != "posix":
        raise SolarSystemDependencyUnavailableError(
            "local artifact verification requires the exact POSIX capability profile"
        )
    required_flags: dict[str, int] = {}
    for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK", "O_NOCTTY"):
        value = getattr(os, name, None)
        if type(value) is not int or value == 0:
            raise SolarSystemDependencyUnavailableError(
                "local artifact verification lacks required POSIX open flags"
            )
        required_flags[name] = value
    supports_dir_fd = getattr(os, "supports_dir_fd", ())
    supports_follow_symlinks = getattr(os, "supports_follow_symlinks", ())
    if (
        os.open not in supports_dir_fd
        or os.stat not in supports_dir_fd
        or os.stat not in supports_follow_symlinks
    ):
        raise SolarSystemDependencyUnavailableError(
            "local artifact verification lacks exact dir-fd/nofollow support"
        )
    directory_flags = (
        os.O_RDONLY
        | required_flags["O_CLOEXEC"]
        | required_flags["O_DIRECTORY"]
        | required_flags["O_NOFOLLOW"]
    )
    final_flags = (
        os.O_RDONLY
        | required_flags["O_CLOEXEC"]
        | required_flags["O_NOFOLLOW"]
        | required_flags["O_NONBLOCK"]
        | required_flags["O_NOCTTY"]
    )
    return directory_flags, final_flags


def _walk_parent_directory(
    root_directory_fd: int,
    parent_components: tuple[str, ...],
    directory_flags: int,
    stack: ExitStack,
) -> int:
    current = root_directory_fd
    for component in parent_components:
        current = os.open(component, directory_flags, dir_fd=current)
        stack.callback(os.close, current)
    return current


def _read_declared_bytes(file_fd: int, declared_length: int) -> tuple[int, str]:
    remaining = declared_length
    calls = 0
    digest = hashlib.sha256()
    while remaining:
        if calls >= _MAXIMUM_DATA_READ_CALLS:
            raise SolarSystemDataError(
                "local artifact verification exceeded the data-read call cap"
            )
        request = min(remaining, _READ_CHUNK_BYTE_LIMIT)
        block = os.read(file_fd, request)
        calls += 1
        if type(block) is not bytes or not block or len(block) > request:
            raise SolarSystemDataError(
                "local artifact verification observed an invalid or early read"
            )
        digest.update(block)
        remaining -= len(block)
    trailing = os.read(file_fd, 1)
    if type(trailing) is not bytes or trailing != b"":
        raise SolarSystemDataError(
            "local artifact verification observed trailing bytes"
        )
    return calls, digest.hexdigest()


def _verify_local_artifact_bytes(
    artifact: ArtifactBinding,
    components: tuple[str, ...],
    root_directory_fd: int,
    directory_flags: int,
    final_flags: int,
) -> LocalArtifactVerificationReceipt:
    with ExitStack() as root_stack:
        duplicated_root = os.dup(root_directory_fd)
        root_stack.callback(os.close, duplicated_root)
        os.set_inheritable(duplicated_root, False)
        pre_root = _selected_stat(os.fstat(duplicated_root), _ROOT_STAT_FIELD_ORDER)
        if not stat.S_ISDIR(pre_root[2]):
            raise SolarSystemDataError(
                "root capability does not identify a directory"
            )

        parent_components = components[:-1]
        leaf = components[-1]
        with ExitStack() as first_walk:
            parent_fd = _walk_parent_directory(
                duplicated_root,
                parent_components,
                directory_flags,
                first_walk,
            )
            pre_open = _selected_stat(
                os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False),
                _FILE_STAT_FIELD_ORDER,
            )
            if not stat.S_ISREG(pre_open[2]):
                raise SolarSystemDataError(
                    "local artifact locator does not name a regular file"
                )
            if pre_open[3] < 1:
                raise SolarSystemDataError(
                    "local artifact path has an invalid link count"
                )
            if pre_open[6] != artifact.byte_length:
                raise SolarSystemDataError(
                    "local artifact path size does not equal its declaration"
                )
            if pre_root[:2] == pre_open[:2]:
                raise SolarSystemDataError(
                    "root directory and local artifact share an invalid inode identity"
                )
            file_fd = os.open(leaf, final_flags, dir_fd=parent_fd)
            first_walk.callback(os.close, file_fd)
            pre_read = _selected_stat(os.fstat(file_fd), _FILE_STAT_FIELD_ORDER)
            if pre_read != pre_open or not stat.S_ISREG(pre_read[2]):
                raise SolarSystemDataError(
                    "local artifact changed between classification and open"
                )
            if pre_read[6] != artifact.byte_length:
                raise SolarSystemDataError(
                    "local artifact size does not equal its declaration"
                )
            data_calls, observed_digest = _read_declared_bytes(
                file_fd,
                artifact.byte_length,
            )
            post_read = _selected_stat(os.fstat(file_fd), _FILE_STAT_FIELD_ORDER)
            if post_read != pre_read or not stat.S_ISREG(post_read[2]):
                raise SolarSystemDataError(
                    "local artifact metadata changed during byte observation"
                )
            if not hmac.compare_digest(observed_digest, artifact.artifact_sha256):
                raise SolarSystemDataError(
                    "local artifact SHA-256 does not equal its declaration"
                )

            # Keep the verified open file description alive while making the
            # second namespace observation.  This still proves only two local
            # observations, not continuous pathname identity.
            with ExitStack() as second_walk:
                second_parent_fd = _walk_parent_directory(
                    duplicated_root,
                    parent_components,
                    directory_flags,
                    second_walk,
                )
                second_resolution = _selected_stat(
                    os.stat(leaf, dir_fd=second_parent_fd, follow_symlinks=False),
                    _FILE_STAT_FIELD_ORDER,
                )
                if second_resolution != post_read or not stat.S_ISREG(
                    second_resolution[2]
                ):
                    raise SolarSystemDataError(
                        "second local namespace observation does not match the read file"
                    )

        post_root = _selected_stat(os.fstat(duplicated_root), _ROOT_STAT_FIELD_ORDER)
        if post_root != pre_root or not stat.S_ISDIR(post_root[2]):
            raise SolarSystemDataError(
                "root capability metadata changed during verification"
            )

    byte_match = ArtifactByteMatchReceipt(
        artifact=artifact,
        verification_scope=_BYTE_VERIFICATION_SCOPE,
        hash_algorithm=_HASH_ALGORITHM,
        expected_byte_length=artifact.byte_length,
        observed_byte_length=artifact.byte_length,
        expected_sha256=artifact.artifact_sha256,
        observed_sha256=observed_digest,
        byte_match_status=_BYTE_MATCH_STATUS,
        content_integrity_class=_CONTENT_INTEGRITY_CLASS,
    )
    return LocalArtifactVerificationReceipt(
        byte_match=byte_match,
        verification_stage=_VERIFICATION_STAGE,
        platform_scope=_PLATFORM_SCOPE,
        root_capability_status=_ROOT_CAPABILITY_STATUS,
        locator_resolution_policy=_LOCATOR_RESOLUTION_POLICY,
        locator_components=components,
        final_open_policy=_FINAL_OPEN_POLICY,
        read_policy=_READ_POLICY,
        maximum_byte_length=MAXIMUM_LOCAL_ARTIFACT_BYTES,
        read_chunk_byte_limit=_READ_CHUNK_BYTE_LIMIT,
        maximum_data_read_calls=_MAXIMUM_DATA_READ_CALLS,
        data_read_call_count=data_calls,
        eof_probe_call_count=1,
        root_stat_field_order=_ROOT_STAT_FIELD_ORDER,
        pre_root_stat=pre_root,
        post_root_stat=post_root,
        file_stat_field_order=_FILE_STAT_FIELD_ORDER,
        pre_open_path_file_stat=pre_open,
        pre_read_file_stat=pre_read,
        post_read_file_stat=post_read,
        second_resolution_file_stat=second_resolution,
        metadata_stability_status=_METADATA_STABILITY_STATUS,
        second_resolution_status=_SECOND_RESOLUTION_STATUS,
        local_metadata_classification=_LOCAL_METADATA_CLASSIFICATION,
        custody_scope=_CUSTODY_SCOPE,
        execution_evidence_status=_EXECUTION_EVIDENCE_STATUS,
        semantic_replay_status=_SEMANTIC_REPLAY_STATUS,
        content_integrity_class=_CONTENT_INTEGRITY_CLASS,
    )


def verify_local_artifact_bytes(
    artifact: ArtifactBinding,
    root_directory_fd: int,
) -> LocalArtifactVerificationReceipt:
    """Verify one bounded local artifact from a borrowed directory capability."""

    checked_artifact, components = _validate_artifact_for_local_verification(artifact)
    checked_root_fd = _exact_integer(
        root_directory_fd,
        "root_directory_fd",
        minimum=0,
        maximum=_MAXIMUM_ROOT_DIRECTORY_FD,
    )
    directory_flags, final_flags = _require_platform_capabilities()
    try:
        return _verify_local_artifact_bytes(
            checked_artifact,
            components,
            checked_root_fd,
            directory_flags,
            final_flags,
        )
    except SolarSystemDataError:
        raise
    except (OSError, OverflowError):
        raise SolarSystemDataError(
            "local artifact verification failed during bounded POSIX access"
        ) from None


def validate_local_artifact_verification_receipt(
    value: LocalArtifactVerificationReceipt,
) -> None:
    """Validate a receipt seal and schema without touching the filesystem."""

    if type(value) is not LocalArtifactVerificationReceipt:
        raise SolarSystemContractError(
            "value must be an exact LocalArtifactVerificationReceipt"
        )
    value.validate_integrity()


__all__ = [
    "ArtifactByteMatchReceipt",
    "LocalArtifactVerificationReceipt",
    "MAXIMUM_LOCAL_ARTIFACT_BYTES",
    "validate_local_artifact_verification_receipt",
    "verify_local_artifact_bytes",
]
