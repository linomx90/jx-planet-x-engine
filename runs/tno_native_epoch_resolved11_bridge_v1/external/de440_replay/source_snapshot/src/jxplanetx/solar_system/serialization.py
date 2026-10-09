"""Bounded canonical serialization for unpublished Solar-System contracts.

The format is deliberately small and closed.  It supplies unauthenticated
content-integrity checks only; it does not establish publisher identity,
scientific authority, registry status, or qualification.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import fields
from typing import Any


MAXIMUM_CANONICAL_BYTES = 8_388_608
MAXIMUM_CANONICAL_DATACLASS_FIELDS = 64
MAXIMUM_CANONICAL_DEPTH = 32
MAXIMUM_CANONICAL_EXACT_INTEGER_BITS = 4_096
MAXIMUM_CANONICAL_MAPPING_ITEMS = 256
MAXIMUM_CANONICAL_NODES = 131_072
MAXIMUM_CANONICAL_SEQUENCE_ITEMS = 256
MAXIMUM_CANONICAL_STRING_CODEPOINTS = 4_096
MAXIMUM_CANONICAL_STRING_UTF8_BYTES = 16_384

_SHA256_HEX_DIGITS = frozenset("0123456789abcdef")
_CONTRACT_DATACLASS_NAMES = frozenset(
    (
        "ArtifactBinding",
        "BodyParameter",
        "CoordinateEpoch",
        "CoverageInterval",
        "EphemerisProviderSpec",
        "EphemerisQuerySpec",
        "ExactUnitScale",
        "FrameRealization",
        "PhysicalConstant",
        "ProviderIdentity",
        "UnitSystemDefinition",
    )
)


class SolarSystemContractError(ValueError):
    """A Solar-System contract or its retained content is inconsistent."""


class _Writer:
    __slots__ = ("buffer", "maximum_bytes")

    def __init__(self, maximum_bytes: int) -> None:
        if (
            type(maximum_bytes) is not int
            or maximum_bytes <= 0
            or maximum_bytes > MAXIMUM_CANONICAL_BYTES
        ):
            raise SolarSystemContractError(
                "maximum_bytes must be an exact positive integer within the hard cap"
            )
        self.buffer = bytearray()
        self.maximum_bytes = maximum_bytes

    @property
    def remaining(self) -> int:
        return self.maximum_bytes - len(self.buffer)

    def append(self, value: bytes) -> None:
        if type(value) is not bytes:
            raise SolarSystemContractError("canonical writer accepts exact bytes only")
        if len(value) > self.remaining:
            raise SolarSystemContractError("canonical content exceeds its byte cap")
        self.buffer.extend(value)

    def string(self, value: str) -> None:
        _validate_string(value, "canonical string")
        # ensure_ascii can require two six-byte surrogate escapes for one
        # non-BMP code point.  Reserve that worst case before invoking JSON.
        maximum_encoded = 2 + 12 * len(value)
        if maximum_encoded > self.remaining:
            raise SolarSystemContractError(
                "canonical content cannot reserve a bounded string encoding"
            )
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        self.append(encoded)


class _Traversal:
    __slots__ = ("active", "nodes", "writer")

    def __init__(self, maximum_bytes: int) -> None:
        self.active: set[int] = set()
        self.nodes = 0
        self.writer = _Writer(maximum_bytes)

    def enter_node(self, depth: int) -> None:
        if type(depth) is not int or depth < 0:
            raise SolarSystemContractError("canonical depth must be an exact integer")
        if depth > MAXIMUM_CANONICAL_DEPTH:
            raise SolarSystemContractError("canonical content exceeds its depth cap")
        if self.nodes >= MAXIMUM_CANONICAL_NODES:
            raise SolarSystemContractError("canonical content exceeds its node cap")
        self.nodes += 1

    def enter_container(self, value: object) -> None:
        identity = id(value)
        if identity in self.active:
            raise SolarSystemContractError("canonical content cannot contain a cycle")
        self.active.add(identity)

    def leave_container(self, value: object) -> None:
        self.active.remove(id(value))


def _validate_string(value: object, label: str) -> str:
    if type(value) is not str:
        raise SolarSystemContractError(f"{label} must be an exact string")
    if len(value) > MAXIMUM_CANONICAL_STRING_CODEPOINTS:
        raise SolarSystemContractError(f"{label} exceeds the code-point cap")
    # Four UTF-8 bytes per Unicode scalar is a prospective hard bound.  The
    # code-point test therefore runs before the bounded conversion.
    if 4 * len(value) > MAXIMUM_CANONICAL_STRING_UTF8_BYTES:
        raise SolarSystemContractError(f"{label} exceeds the UTF-8 reservation cap")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise SolarSystemContractError(f"{label} is not valid Unicode scalar text") from exc
    if len(encoded) > MAXIMUM_CANONICAL_STRING_UTF8_BYTES:
        raise SolarSystemContractError(f"{label} exceeds the UTF-8 byte cap")
    return value


def validate_sha256(value: object, label: str = "sha256") -> str:
    """Return one exact lowercase SHA-256 digest or fail closed."""

    if (
        type(label) is not str
        or not label
        or type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_HEX_DIGITS for character in value)
    ):
        raise SolarSystemContractError(
            f"{label if type(label) is str and label else 'sha256'} "
            "must be exact lowercase SHA-256 hex"
        )
    return value


def _write_sequence(
    value: tuple[object, ...] | list[object], traversal: _Traversal, depth: int
) -> None:
    if len(value) > MAXIMUM_CANONICAL_SEQUENCE_ITEMS:
        raise SolarSystemContractError("canonical sequence exceeds its item cap")
    traversal.enter_container(value)
    try:
        traversal.writer.append(b"[")
        for index, item in enumerate(value):
            if index:
                traversal.writer.append(b",")
            _write(item, traversal, depth + 1)
        traversal.writer.append(b"]")
    finally:
        traversal.leave_container(value)


def _write_mapping(value: dict[str, object], traversal: _Traversal, depth: int) -> None:
    if len(value) > MAXIMUM_CANONICAL_MAPPING_ITEMS:
        raise SolarSystemContractError("canonical mapping exceeds its item cap")
    for key in value:
        _validate_string(key, "canonical mapping key")
    ordered_keys = sorted(value)
    traversal.enter_container(value)
    try:
        traversal.writer.append(b"{")
        for index, key in enumerate(ordered_keys):
            if index:
                traversal.writer.append(b",")
            traversal.writer.string(key)
            traversal.writer.append(b":")
            _write(value[key], traversal, depth + 1)
        traversal.writer.append(b"}")
    finally:
        traversal.leave_container(value)


def _write_dataclass(value: object, traversal: _Traversal, depth: int) -> None:
    qualified_name = f"{type(value).__module__}.{type(value).__qualname__}"
    if (
        type(value).__module__ != "jxplanetx.solar_system.contracts"
        or type(value).__qualname__ not in _CONTRACT_DATACLASS_NAMES
    ):
        raise SolarSystemContractError(
            "canonical dataclasses must belong to jxplanetx.solar_system"
        )
    # Resolve exact identities lazily so this serializer remains importable
    # without importing contracts, while spoofed module/name dataclasses fail.
    from . import contracts as contract_module

    expected_type = getattr(contract_module, type(value).__qualname__, None)
    if type(value) is not expected_type:
        raise SolarSystemContractError(
            "canonical dataclass identity is not an approved exact contract type"
        )
    descriptors = fields(value)
    if len(descriptors) > MAXIMUM_CANONICAL_DATACLASS_FIELDS:
        raise SolarSystemContractError("canonical dataclass exceeds its field cap")
    traversal.enter_container(value)
    try:
        traversal.writer.append(b'{"dataclass":')
        traversal.writer.string(qualified_name)
        traversal.writer.append(b',"fields":[')
        for index, descriptor in enumerate(descriptors):
            if index:
                traversal.writer.append(b",")
            traversal.writer.append(b"[")
            traversal.writer.string(descriptor.name)
            traversal.writer.append(b",")
            _write(getattr(value, descriptor.name), traversal, depth + 1)
            traversal.writer.append(b"]")
        traversal.writer.append(b"]}")
    finally:
        traversal.leave_container(value)


def _is_approved_contract(value: object) -> bool:
    """Check exact identities without invoking arbitrary dataclass protocols."""

    from . import contracts as contract_module

    candidate_type = type(value)
    return any(
        candidate_type is getattr(contract_module, name)
        for name in _CONTRACT_DATACLASS_NAMES
    )


def _write(value: object, traversal: _Traversal, depth: int) -> None:
    traversal.enter_node(depth)
    writer = traversal.writer
    if value is None:
        writer.append(b"null")
    elif type(value) is bool:
        writer.append(b"true" if value else b"false")
    elif type(value) is int:
        if value.bit_length() > MAXIMUM_CANONICAL_EXACT_INTEGER_BITS:
            raise SolarSystemContractError("canonical integer exceeds its bit cap")
        writer.append(str(value).encode("ascii"))
    elif type(value) is float:
        if not math.isfinite(value):
            raise SolarSystemContractError("canonical floats must be finite")
        writer.append(b'{"float_hex":')
        writer.string(value.hex())
        writer.append(b"}")
    elif type(value) is str:
        writer.string(value)
    elif type(value) is tuple or type(value) is list:
        _write_sequence(value, traversal, depth)
    elif type(value) is dict:
        _write_mapping(value, traversal, depth)
    elif _is_approved_contract(value):
        _write_dataclass(value, traversal, depth)
    else:
        raise SolarSystemContractError(
            f"unsupported canonical content type {type(value).__name__!r}"
        )


def canonical_json(
    value: object, *, maximum_bytes: int = MAXIMUM_CANONICAL_BYTES
) -> bytes:
    """Return exact bounded canonical JSON bytes for supported content."""

    traversal = _Traversal(maximum_bytes)
    _write(value, traversal, 0)
    return bytes(traversal.writer.buffer)


def domain_sha256(
    domain: str,
    schema: str,
    value: object,
    *,
    maximum_bytes: int = MAXIMUM_CANONICAL_BYTES,
) -> str:
    """Hash one schema-bound payload with explicit domain separation."""

    _validate_string(domain, "checksum domain")
    _validate_string(schema, "checksum schema")
    if not domain or not schema or "\x00" in domain or "\x00" in schema:
        raise SolarSystemContractError(
            "checksum domain and schema must be nonempty and NUL-free"
        )
    if (
        type(maximum_bytes) is not int
        or maximum_bytes <= 0
        or maximum_bytes > MAXIMUM_CANONICAL_BYTES
    ):
        raise SolarSystemContractError(
            "maximum_bytes must be an exact positive integer within the hard cap"
        )
    domain_bytes = domain.encode("utf-8")
    schema_bytes = schema.encode("utf-8")
    prefix = domain_bytes + b"\x00" + schema_bytes + b"\x00"
    if len(prefix) >= maximum_bytes:
        raise SolarSystemContractError("checksum prefix exhausts its byte cap")
    payload = canonical_json(value, maximum_bytes=maximum_bytes - len(prefix))
    return hashlib.sha256(prefix + payload).hexdigest()


__all__ = [
    "MAXIMUM_CANONICAL_BYTES",
    "MAXIMUM_CANONICAL_DATACLASS_FIELDS",
    "MAXIMUM_CANONICAL_DEPTH",
    "MAXIMUM_CANONICAL_EXACT_INTEGER_BITS",
    "MAXIMUM_CANONICAL_MAPPING_ITEMS",
    "MAXIMUM_CANONICAL_NODES",
    "MAXIMUM_CANONICAL_SEQUENCE_ITEMS",
    "MAXIMUM_CANONICAL_STRING_CODEPOINTS",
    "MAXIMUM_CANONICAL_STRING_UTF8_BYTES",
    "SolarSystemContractError",
    "canonical_json",
    "domain_sha256",
    "validate_sha256",
]
