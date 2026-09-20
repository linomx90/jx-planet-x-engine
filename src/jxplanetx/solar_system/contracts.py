"""Exact immutable context contracts for future Solar-System support.

This module performs no time conversion, frame transformation, ephemeris
query, file access, network access, or force-model construction.  Its hashes
are unauthenticated content-integrity seals only.  They do not establish data
authenticity, legal authorization, registry authority, or qualification.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, fields
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from math import gcd
from pathlib import PurePosixPath
from typing import ClassVar

from .serialization import (
    MAXIMUM_CANONICAL_EXACT_INTEGER_BITS,
    SolarSystemContractError,
    domain_sha256,
    validate_sha256,
)


MAXIMUM_ARTIFACT_COUNT = 64
MAXIMUM_BODY_COUNT = 64
MAXIMUM_COVERAGE_INTERVAL_COUNT = 16
MAXIMUM_IDENTIFIER_CODEPOINTS = 256
MAXIMUM_LOGICAL_LOCATOR_CODEPOINTS = 1_024
MAXIMUM_NAIF_IDENTIFIER_MAGNITUDE = (1 << 31) - 1

_DEFINED_DAY_UNIT_ID = "unit.day.86400-si-seconds.defined-exact.v1"
_DEFINED_SECOND_UNIT_ID = "unit.second.defined-exact.v1"
_SPICE_TDB_OFFSET_ORIGIN_ID = "SPICE_J2000_TDB_ORIGIN"
_CONTENT_INTEGRITY_CLASS = "UNAUTHENTICATED_CONTENT_INTEGRITY_ONLY"
_EVIDENCE_CLASS = "MODEL_OUTPUT"
_DECIMAL = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?$")

_TIME_SCALES = frozenset(("TAI", "TCB", "TDB", "TT", "SYNTHETIC"))
_PHYSICAL_TIME_SCALES = frozenset(("TAI", "TCB", "TDB", "TT"))
_REPRESENTATIONS = frozenset(
    (
        "JD_TWO_PART",
        "SPICE_TDB_J2000_OFFSET_SECONDS_TWO_PART",
        "SYNTHETIC_OFFSET",
    )
)
_ENDPOINT_POLICIES = frozenset(
    (
        "CLOSED_CLOSED",
        "CLOSED_OPEN",
        "OPEN_CLOSED",
        "OPEN_OPEN",
    )
)
_ARTIFACT_ROLES = frozenset(
    (
        "CONSTANTS",
        "LICENSE",
        "NATIVE_LIBRARY",
        "SOFTWARE_DISTRIBUTION",
        "SOFTWARE_SOURCE",
        "SPK",
        "SYNTHETIC_FIXTURE",
    )
)
_ORDERED_LOAD_ROLES = frozenset(("SPK",))
_TIME_REALIZATION_ARTIFACT_ROLES = frozenset(
    (
        "CONSTANTS",
        "NATIVE_LIBRARY",
        "SOFTWARE_DISTRIBUTION",
        "SOFTWARE_SOURCE",
    )
)
_CONTEXT_COVERAGE_STATUSES = frozenset(
    ("RETAINED", "NOT_APPLICABLE", "NOT_RETAINED_BLOCKED")
)
_ARTIFACT_COVERAGE_STATUSES = frozenset(
    ("COARSE_ARTIFACT_TIME_ENVELOPE", "NOT_APPLICABLE", "NOT_RETAINED_BLOCKED")
)
_LICENSE_EVIDENCE_STATUSES = frozenset(
    (
        "RETAINED_HASH_BOUND_LICENSE_ARTIFACT",
        "RETAINED_LICENSE_TEXT_SELF_EVIDENCE_NONAUTHORIZING",
        "EXTERNAL_NOT_REDISTRIBUTED_LICENSE_NOT_RETAINED_NONAUTHORIZING",
        "NOT_APPLICABLE_INTERNAL_SYNTHETIC",
    )
)
_REDISTRIBUTION_STATUSES = frozenset(
    (
        "BUNDLED_WITH_RETAINED_LICENSE_EVIDENCE",
        "EXTERNAL_REFERENCE_NOT_REDISTRIBUTED",
        "NOT_EVALUATED_BLOCKED",
    )
)
_LOAD_ORDER_STATUSES = frozenset(("ORDERED_LOAD_MEMBER", "NOT_LOADABLE"))
_LOCATOR_KINDS = frozenset(("LOCAL_REGULAR_FILE", "EXTERNAL_REFERENCE_ONLY"))
_EXTRAPOLATION_POLICIES = frozenset(("FORBID",))
_CONTENT_INTEGRITY_CLASSES = frozenset((_CONTENT_INTEGRITY_CLASS,))
_EVIDENCE_CLASSES = frozenset((_EVIDENCE_CLASS,))
_PROVIDER_KINDS = frozenset(("LOCAL_OFFLINE_SPK", "SYNTHETIC_TEST"))
_FRAME_KINDS = frozenset(
    (
        "INERTIAL_ORIGIN_CENTERED",
        "SYNTHETIC",
        "TCB_BCRS",
        "TDB_COMPATIBLE_BARYCENTRIC_INERTIAL",
    )
)
_ORIGIN_KINDS = frozenset(
    (
        "BODY_CENTER",
        "NEWTONIAN_MODEL_BARYCENTER",
        "PLANETARY_SYSTEM_BARYCENTER",
        "SOLAR_SYSTEM_BARYCENTER",
        "SYNTHETIC",
    )
)
_ORIENTATION_TIME_DEPENDENCE = frozenset(("STATIC",))
_DIMENSIONS = frozenset(
    ("DIMENSIONLESS", "GRAVITATIONAL_PARAMETER", "LENGTH", "MASS", "SPEED", "TIME")
)
_SI_UNIT_BY_DIMENSION = {
    "DIMENSIONLESS": "si.one",
    "GRAVITATIONAL_PARAMETER": "si.metre3-per-second2",
    "LENGTH": "si.metre",
    "MASS": "si.kilogram",
    "SPEED": "si.metre-per-second",
    "TIME": "si.second",
}
_EXACT_DEFINITION_CLASSES = frozenset(("DEFINED_EXACT", "NOMINAL_EXACT"))
_CONSTANT_CLASSES = frozenset(
    ("DEFINED_EXACT", "ESTIMATED", "FITTED", "NOMINAL_EXACT")
)
_QUANTITY_KINDS = frozenset(
    (
        "DIMENSIONLESS",
        "GRAVITATIONAL_PARAMETER",
        "LENGTH",
        "MASS",
        "RADIUS",
        "SPEED",
        "TIME",
    )
)
_UNCERTAINTY_STATUSES = frozenset(
    ("EXACT_NOT_APPLICABLE", "NOT_PROVIDED", "PROVIDED")
)
_UNCERTAINTY_KINDS = frozenset(
    (
        "ABSOLUTE_BOUND_MAGNITUDE",
        "STANDARD_UNCERTAINTY",
        "SYMMETRIC_CONFIDENCE_HALF_WIDTH",
    )
)
_UNCERTAINTY_SCALES = frozenset(
    ("ABSOLUTE_SAME_UNIT_AS_VALUE", "NOT_APPLICABLE")
)
_COORDINATE_SCALE_APPLICABILITY = frozenset(
    ("NOT_APPLICABLE", "SCALE_INDEPENDENT", "TCB_COMPATIBLE", "TDB_COMPATIBLE")
)
_BODY_ROLES = frozenset(
    ("MASSIVE_BODY_CENTER", "MASSLESS_TEST_PARTICLE", "PLANETARY_SYSTEM_BARYCENTER")
)
_GM_STATUSES = frozenset(
    ("NOT_PROVIDED_MASSLESS_TARGET", "PROVIDED_DYNAMICS_PARAMETER")
)
_MASS_STATUSES = frozenset(
    ("NOT_PROVIDED_NOT_USED", "PROVIDED_NOT_USED_BY_GM_DYNAMICS")
)
_RADIUS_STATUSES = frozenset(
    ("POINT_MASS_NO_COLLISION_RADIUS", "PROVIDED_REFERENCE_RADIUS")
)
_RADIUS_KINDS = frozenset(
    (
        "COLLISION_EFFECTIVE",
        "EQUATORIAL",
        "HARMONIC_REFERENCE",
        "MEAN",
        "NOMINAL_REFERENCE",
    )
)
_PROVIDER_CAPABILITIES = frozenset(
    (
        "COARSE_ARTIFACT_TIME_ENVELOPE",
        "GEOMETRIC_CARTESIAN_STATE",
        "TDB_COORDINATE_EPOCH",
    )
)
_TARGET_CHAIN_AVAILABILITY_STATUS = (
    "REQUIRES_RUNTIME_PROVIDER_TARGET_AVAILABILITY_VALIDATION"
)
_TARGET_CHAIN_AVAILABILITY_STATUSES = frozenset(
    (_TARGET_CHAIN_AVAILABILITY_STATUS,)
)
_ABERRATION_CORRECTIONS = frozenset(("NONE",))
_STATE_KINDS = frozenset(("GEOMETRIC",))


class SolarSystemDataError(RuntimeError):
    """Retained Solar-System data cannot satisfy a declared context."""


class SolarSystemDependencyUnavailableError(SolarSystemDataError):
    """An exact optional provider dependency is unavailable or mismatched."""


class SolarSystemCoverageError(SolarSystemDataError):
    """An epoch is outside a declared retained temporal envelope."""


def _exact_type(value: object, expected: type[object], label: str) -> object:
    if type(value) is not expected:
        raise SolarSystemContractError(
            f"{label} must be exact {expected.__module__}.{expected.__qualname__}"
        )
    return value


def _text(
    value: object,
    label: str,
    *,
    maximum_codepoints: int = MAXIMUM_IDENTIFIER_CODEPOINTS,
) -> str:
    if type(value) is not str:
        raise SolarSystemContractError(f"{label} must be an exact string")
    if len(value) > maximum_codepoints:
        raise SolarSystemContractError(f"{label} exceeds its code-point cap")
    if not value or not _is_trimmed_contract_text(value):
        raise SolarSystemContractError(f"{label} must be a nonempty trimmed string")
    if any(ord(character) < 0x20 or ord(character) > 0x7E for character in value):
        raise SolarSystemContractError(
            f"{label} must use printable ASCII contract text only"
        )
    return value


def _is_trimmed_contract_text(value: str) -> bool:
    """Inspect only text that has already passed its hard code-point cap."""

    return value.strip() == value


def _token(value: object, allowed: frozenset[str], label: str) -> str:
    result = _text(value, label)
    if result not in allowed:
        raise SolarSystemContractError(f"{label} is not a closed supported token")
    return result


def _bool(value: object, expected: bool, label: str) -> bool:
    if type(value) is not bool or value is not expected:
        raise SolarSystemContractError(f"{label} must be exactly {expected}")
    return value


def _integer(
    value: object,
    label: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise SolarSystemContractError(f"{label} must be an exact integer")
    if value.bit_length() > MAXIMUM_CANONICAL_EXACT_INTEGER_BITS:
        raise SolarSystemContractError(f"{label} exceeds the exact-integer bit cap")
    if minimum is not None and value < minimum:
        raise SolarSystemContractError(f"{label} is below its minimum")
    if maximum is not None and value > maximum:
        raise SolarSystemContractError(f"{label} exceeds its maximum")
    return value


def _number(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise SolarSystemContractError(f"{label} must be an exact finite float")
    if minimum is not None and value < minimum:
        raise SolarSystemContractError(f"{label} is below its minimum")
    if maximum is not None and value > maximum:
        raise SolarSystemContractError(f"{label} exceeds its maximum")
    return value


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label)


def _identifier_tuple(
    value: object,
    label: str,
    *,
    maximum_count: int = MAXIMUM_ARTIFACT_COUNT,
    allow_empty: bool = True,
    sorted_unique: bool = True,
) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise SolarSystemContractError(f"{label} must be an exact tuple")
    if len(value) > maximum_count:
        raise SolarSystemContractError(f"{label} exceeds its count cap")
    if not allow_empty and not value:
        raise SolarSystemContractError(f"{label} must not be empty")
    result = tuple(_text(item, f"{label}[{index}]") for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise SolarSystemContractError(f"{label} cannot contain duplicates")
    if sorted_unique and result != tuple(sorted(result)):
        raise SolarSystemContractError(f"{label} must be sorted canonically")
    return result


def _contract_payload(value: object) -> dict[str, object]:
    return {
        "dataclass": f"{type(value).__module__}.{type(value).__qualname__}",
        "fields": tuple(
            (descriptor.name, getattr(value, descriptor.name))
            for descriptor in fields(value)
            if descriptor.name != "content_sha256"
        ),
    }


def _finish(value: object, slug: str, *, permit_empty: bool) -> None:
    domain = f"jxplanetx.solar-system.{slug}.content-integrity.v1"
    schema = f"jxplanetx.solar-system.{slug}.payload.v1"
    actual = getattr(value, "content_sha256")
    if type(actual) is not str:
        raise SolarSystemContractError("content_sha256 must be an exact string")
    expected = domain_sha256(domain, schema, _contract_payload(value))
    if actual == "" and permit_empty:
        object.__setattr__(value, "content_sha256", expected)
        return
    validate_sha256(actual, "content_sha256")
    if actual != expected:
        raise SolarSystemContractError("content_sha256 does not bind exact contract content")


def _require(value: object, expected: type[object], label: str) -> object:
    _exact_type(value, expected, label)
    validate_integrity(value)
    return value


def _coverage_tuple(
    status: object,
    value: object,
    label: str,
) -> tuple[CoverageInterval, ...] | None:
    checked_status = _token(status, _CONTEXT_COVERAGE_STATUSES, f"{label}_status")
    if checked_status == "RETAINED":
        if type(value) is not tuple:
            raise SolarSystemContractError(f"{label} must be an exact tuple when retained")
        if not value or len(value) > MAXIMUM_COVERAGE_INTERVAL_COUNT:
            raise SolarSystemContractError(f"{label} has an invalid retained count")
        for index, interval in enumerate(value):
            _require(interval, CoverageInterval, f"{label}[{index}]")
        first = value[0]
        assert type(first) is CoverageInterval
        for index, interval in enumerate(value[1:], start=1):
            assert type(interval) is CoverageInterval
            if not _same_coordinate_context(first.start, interval.start):
                raise SolarSystemContractError(
                    f"{label} intervals must share one exact coordinate context"
                )
            previous = value[index - 1]
            assert type(previous) is CoverageInterval
            if _epoch_value(previous.end) >= _epoch_value(interval.start):
                raise SolarSystemContractError(
                    f"{label} intervals must be strictly ordered and nonoverlapping"
                )
        return value
    if value is not None:
        raise SolarSystemContractError(f"{label} must be None unless retained")
    return None


def _artifact_coverage_tuple(
    status: object,
    value: object,
    label: str,
) -> tuple[CoverageInterval, ...] | None:
    checked_status = _token(status, _ARTIFACT_COVERAGE_STATUSES, f"{label}_status")
    translated = "RETAINED" if checked_status == "COARSE_ARTIFACT_TIME_ENVELOPE" else checked_status
    return _coverage_tuple(translated, value, label)


def _same_coordinate_context(left: CoordinateEpoch, right: CoordinateEpoch) -> bool:
    return (
        left.time_scale == right.time_scale
        and left.representation == right.representation
        and left.coordinate_unit_id == right.coordinate_unit_id
        and left.origin_id == right.origin_id
        and left.realization_id == right.realization_id
    )


def _epoch_value(value: CoordinateEpoch) -> Fraction:
    return Fraction(value.whole, 1) + Fraction.from_float(value.fraction)


def _coverage_contains(interval: CoverageInterval, epoch: CoordinateEpoch) -> bool:
    if not _same_coordinate_context(interval.start, epoch):
        return False
    item = _epoch_value(epoch)
    start = _epoch_value(interval.start)
    end = _epoch_value(interval.end)
    left_closed = interval.endpoint_policy in ("CLOSED_CLOSED", "CLOSED_OPEN")
    right_closed = interval.endpoint_policy in ("CLOSED_CLOSED", "OPEN_CLOSED")
    return (item >= start if left_closed else item > start) and (
        item <= end if right_closed else item < end
    )


@dataclass(frozen=True, eq=False)
class CoordinateEpoch:
    """One canonical coordinate epoch and only its time-realization evidence."""

    time_scale: str
    representation: str
    whole: int
    fraction: float
    coordinate_unit_id: str
    origin_id: str
    realization_id: str
    artifact_ids: tuple[str, ...]
    content_sha256: str = ""

    _SLUG: ClassVar[str] = "coordinate-epoch"

    def __post_init__(self) -> None:
        self._validate()
        _finish(self, self._SLUG, permit_empty=True)

    def _validate(self) -> None:
        _exact_type(self, CoordinateEpoch, "coordinate epoch")
        scale = _token(self.time_scale, _TIME_SCALES, "time_scale")
        representation = _token(self.representation, _REPRESENTATIONS, "representation")
        _integer(self.whole, "whole")
        fraction = _number(self.fraction, "fraction", minimum=-0.5)
        if fraction >= 0.5:
            raise SolarSystemContractError("fraction must be strictly below 0.5")
        if fraction == 0.0 and math.copysign(1.0, fraction) < 0.0:
            raise SolarSystemContractError("a canonical epoch requires positive zero fraction")
        unit_id = _text(self.coordinate_unit_id, "coordinate_unit_id")
        origin_id = _text(self.origin_id, "origin_id")
        _text(self.realization_id, "realization_id")
        _identifier_tuple(self.artifact_ids, "artifact_ids")
        if representation == "JD_TWO_PART":
            if scale not in _PHYSICAL_TIME_SCALES:
                raise SolarSystemContractError("JD_TWO_PART requires a physical uniform scale")
            if unit_id != _DEFINED_DAY_UNIT_ID or origin_id != "JULIAN_DATE":
                raise SolarSystemContractError(
                    "JD_TWO_PART requires the exact defined day and JULIAN_DATE origin"
                )
        elif representation == "SPICE_TDB_J2000_OFFSET_SECONDS_TWO_PART":
            if scale != "TDB":
                raise SolarSystemContractError(
                    "SPICE TDB offset representation requires TDB"
                )
            if (
                unit_id != _DEFINED_SECOND_UNIT_ID
                or origin_id != _SPICE_TDB_OFFSET_ORIGIN_ID
            ):
                raise SolarSystemContractError(
                    "SPICE TDB offsets require exact SI seconds and the explicit SPICE J2000 TDB origin"
                )
        else:
            if scale != "SYNTHETIC":
                raise SolarSystemContractError("SYNTHETIC_OFFSET requires SYNTHETIC")
            if unit_id in (_DEFINED_DAY_UNIT_ID, _DEFINED_SECOND_UNIT_ID) or origin_id in (
                "JULIAN_DATE",
                _SPICE_TDB_OFFSET_ORIGIN_ID,
            ):
                raise SolarSystemContractError(
                    "SYNTHETIC_OFFSET cannot inherit physical epoch semantics"
                )


@dataclass(frozen=True, eq=False)
class CoverageInterval:
    """One exact represented-coordinate coverage interval."""

    start: CoordinateEpoch
    end: CoordinateEpoch
    endpoint_policy: str
    content_sha256: str = ""

    _SLUG: ClassVar[str] = "coverage-interval"

    def __post_init__(self) -> None:
        self._validate()
        _finish(self, self._SLUG, permit_empty=True)

    def _validate(self) -> None:
        _exact_type(self, CoverageInterval, "coverage interval")
        start = _require(self.start, CoordinateEpoch, "start")
        end = _require(self.end, CoordinateEpoch, "end")
        assert type(start) is CoordinateEpoch and type(end) is CoordinateEpoch
        _token(self.endpoint_policy, _ENDPOINT_POLICIES, "endpoint_policy")
        if not _same_coordinate_context(start, end):
            raise SolarSystemContractError("coverage endpoints require identical context")
        if start.artifact_ids != end.artifact_ids:
            raise SolarSystemContractError(
                "coverage endpoints require identical time-realization evidence"
            )
        if _epoch_value(start) > _epoch_value(end):
            raise SolarSystemContractError("coverage start cannot exceed coverage end")
        if _epoch_value(start) == _epoch_value(end) and self.endpoint_policy != "CLOSED_CLOSED":
            raise SolarSystemContractError("zero-width coverage must include both endpoints")


@dataclass(frozen=True, eq=False)
class ArtifactBinding:
    """Byte identity, coverage, ordering, and nonauthorizing license evidence."""

    artifact_id: str
    artifact_role: str
    provider_id: str
    version: str
    logical_locator: str
    locator_kind: str
    byte_length: int
    artifact_sha256: str
    media_type: str
    coverage_status: str
    coverage: tuple[CoverageInterval, ...] | None
    license_evidence_status: str
    license_spdx: str | None
    license_artifact_id: str | None
    redistribution_status: str
    load_order_status: str
    load_order: int | None
    extrapolation_policy: str
    content_integrity_class: str
    content_sha256: str = ""

    _SLUG: ClassVar[str] = "artifact-binding"

    def __post_init__(self) -> None:
        self._validate()
        _finish(self, self._SLUG, permit_empty=True)

    def _validate(self) -> None:
        _exact_type(self, ArtifactBinding, "artifact binding")
        artifact_id = _text(self.artifact_id, "artifact_id")
        role = _token(self.artifact_role, _ARTIFACT_ROLES, "artifact_role")
        _text(self.provider_id, "provider_id")
        version = _text(self.version, "version")
        locator = _text(
            self.logical_locator,
            "logical_locator",
            maximum_codepoints=MAXIMUM_LOGICAL_LOCATOR_CODEPOINTS,
        )
        locator_kind = _token(self.locator_kind, _LOCATOR_KINDS, "locator_kind")
        redistribution = _token(
            self.redistribution_status,
            _REDISTRIBUTION_STATUSES,
            "redistribution_status",
        )
        if locator_kind == "LOCAL_REGULAR_FILE":
            path = PurePosixPath(locator)
            if path.is_absolute() or str(path) != locator or any(
                part in ("", ".", "..") for part in path.parts
            ):
                raise SolarSystemContractError(
                    "local logical_locator must be normalized relative POSIX form"
                )
        elif redistribution != "EXTERNAL_REFERENCE_NOT_REDISTRIBUTED":
            raise SolarSystemContractError(
                "external-reference locators require external-not-redistributed status"
            )
        _integer(self.byte_length, "byte_length", minimum=1, maximum=(1 << 63) - 1)
        digest = validate_sha256(self.artifact_sha256, "artifact_sha256")
        if digest == "0" * 64:
            raise SolarSystemContractError("artifact_sha256 cannot be an all-zero placeholder")
        _text(self.media_type, "media_type")
        coverage = _artifact_coverage_tuple(self.coverage_status, self.coverage, "coverage")
        if role == "SPK" and coverage is None:
            raise SolarSystemContractError("SPK artifacts require retained coverage")
        if role in ("LICENSE", "NATIVE_LIBRARY", "SOFTWARE_DISTRIBUTION", "SOFTWARE_SOURCE") and coverage is not None:
            raise SolarSystemContractError("software/license artifacts cannot claim time coverage")
        license_status = _token(
            self.license_evidence_status,
            _LICENSE_EVIDENCE_STATUSES,
            "license_evidence_status",
        )
        if license_status == "RETAINED_HASH_BOUND_LICENSE_ARTIFACT":
            _text(self.license_spdx, "license_spdx")
            license_id = _text(self.license_artifact_id, "license_artifact_id")
            if license_id == artifact_id:
                raise SolarSystemContractError("an artifact cannot be its own license artifact")
            if redistribution != "BUNDLED_WITH_RETAINED_LICENSE_EVIDENCE":
                raise SolarSystemContractError(
                    "retained license evidence requires the bundled redistribution status"
                )
            if locator_kind != "LOCAL_REGULAR_FILE":
                raise SolarSystemContractError(
                    "bundled bytes with retained license evidence must be local"
                )
        elif license_status == "RETAINED_LICENSE_TEXT_SELF_EVIDENCE_NONAUTHORIZING":
            if role != "LICENSE":
                raise SolarSystemContractError(
                    "self-evidencing retained license text requires LICENSE role"
                )
            _text(self.license_spdx, "license_spdx")
            if self.license_artifact_id is not None:
                raise SolarSystemContractError(
                    "a retained license text cannot name itself as evidence"
                )
            if redistribution != "BUNDLED_WITH_RETAINED_LICENSE_EVIDENCE":
                raise SolarSystemContractError(
                    "retained license text requires bundled redistribution status"
                )
            if locator_kind != "LOCAL_REGULAR_FILE":
                raise SolarSystemContractError("retained license text must be local")
        elif license_status == "EXTERNAL_NOT_REDISTRIBUTED_LICENSE_NOT_RETAINED_NONAUTHORIZING":
            _text(self.license_spdx, "license_spdx")
            if self.license_artifact_id is not None:
                raise SolarSystemContractError(
                    "external nonretained license evidence cannot name a retained artifact"
                )
            if redistribution != "EXTERNAL_REFERENCE_NOT_REDISTRIBUTED":
                raise SolarSystemContractError(
                    "external nonretained evidence requires external-not-redistributed status"
                )
        else:
            if role != "SYNTHETIC_FIXTURE":
                raise SolarSystemContractError(
                    "license evidence is not applicable only to internal synthetic fixtures"
                )
            if self.license_spdx is not None or self.license_artifact_id is not None:
                raise SolarSystemContractError(
                    "not-applicable license evidence requires exact None fields"
                )
            if redistribution != "NOT_EVALUATED_BLOCKED":
                raise SolarSystemContractError(
                    "internal synthetic evidence must remain legally unevaluated"
                )
        load_status = _token(
            self.load_order_status, _LOAD_ORDER_STATUSES, "load_order_status"
        )
        if load_status == "ORDERED_LOAD_MEMBER":
            _integer(self.load_order, "load_order", minimum=0, maximum=MAXIMUM_ARTIFACT_COUNT - 1)
            if role not in _ORDERED_LOAD_ROLES:
                raise SolarSystemContractError("only loadable data artifacts can have load order")
            if locator_kind != "LOCAL_REGULAR_FILE":
                raise SolarSystemContractError("ordered load artifacts must be retained locally")
            if coverage is None:
                raise SolarSystemContractError(
                    "ordered load artifacts require a retained coarse time envelope"
                )
        else:
            if self.load_order is not None:
                raise SolarSystemContractError("nonloadable artifacts require load_order=None")
            if role in _ORDERED_LOAD_ROLES:
                raise SolarSystemContractError("loadable data artifacts require explicit load order")
        _token(
            self.extrapolation_policy,
            _EXTRAPOLATION_POLICIES,
            "extrapolation_policy",
        )
        _token(
            self.content_integrity_class,
            _CONTENT_INTEGRITY_CLASSES,
            "content_integrity_class",
        )


@dataclass(frozen=True, eq=False)
class ProviderIdentity:
    """Exact implementation identity without resolving or importing it."""

    provider_id: str
    implementation_id: str
    provider_kind: str
    version: str
    runtime_id: str
    module_name: str
    distribution_name: str
    distribution_version: str
    ordered_implementation_artifact_ids: tuple[str, ...]
    content_sha256: str = ""

    _SLUG: ClassVar[str] = "provider-identity"

    def __post_init__(self) -> None:
        self._validate()
        _finish(self, self._SLUG, permit_empty=True)

    def _validate(self) -> None:
        _exact_type(self, ProviderIdentity, "provider identity")
        _text(self.provider_id, "provider_id")
        _text(self.implementation_id, "implementation_id")
        _token(self.provider_kind, _PROVIDER_KINDS, "provider_kind")
        version = _text(self.version, "version")
        _text(self.runtime_id, "runtime_id")
        _text(self.module_name, "module_name")
        _text(self.distribution_name, "distribution_name")
        distribution_version = _text(
            self.distribution_version, "distribution_version"
        )
        if distribution_version != version:
            raise SolarSystemContractError(
                "provider and distribution versions must be identical in v1"
            )
        _identifier_tuple(
            self.ordered_implementation_artifact_ids,
            "ordered_implementation_artifact_ids",
            allow_empty=False,
            sorted_unique=False,
        )


@dataclass(frozen=True, eq=False)
class FrameRealization:
    """A named axes and origin realization, not a transformation service."""

    frame_id: str
    frame_kind: str
    origin_kind: str
    origin_naif_id: int | None
    origin_realization_id: str
    axes_realization_id: str
    orientation_model_id: str
    orientation_time_dependence: str
    coordinate_time_scale: str
    coverage_status: str
    coverage: tuple[CoverageInterval, ...] | None
    artifact_ids: tuple[str, ...]
    content_sha256: str = ""

    _SLUG: ClassVar[str] = "frame-realization"

    def __post_init__(self) -> None:
        self._validate()
        _finish(self, self._SLUG, permit_empty=True)

    def _validate(self) -> None:
        _exact_type(self, FrameRealization, "frame realization")
        _text(self.frame_id, "frame_id")
        kind = _token(self.frame_kind, _FRAME_KINDS, "frame_kind")
        origin = _token(self.origin_kind, _ORIGIN_KINDS, "origin_kind")
        if self.origin_naif_id is not None:
            _integer(
                self.origin_naif_id,
                "origin_naif_id",
                minimum=-MAXIMUM_NAIF_IDENTIFIER_MAGNITUDE,
                maximum=MAXIMUM_NAIF_IDENTIFIER_MAGNITUDE,
            )
        _text(self.origin_realization_id, "origin_realization_id")
        _text(self.axes_realization_id, "axes_realization_id")
        _text(self.orientation_model_id, "orientation_model_id")
        _token(
            self.orientation_time_dependence,
            _ORIENTATION_TIME_DEPENDENCE,
            "orientation_time_dependence",
        )
        scale = _token(self.coordinate_time_scale, _TIME_SCALES, "coordinate_time_scale")
        coverage = _coverage_tuple(self.coverage_status, self.coverage, "coverage")
        artifact_ids = _identifier_tuple(self.artifact_ids, "artifact_ids")
        if coverage is not None:
            for interval in coverage:
                if interval.start.time_scale != scale:
                    raise SolarSystemContractError(
                        "frame coverage must use the frame coordinate time scale"
                    )
                if not set(interval.start.artifact_ids).issubset(artifact_ids):
                    raise SolarSystemContractError(
                        "frame artifacts must include coverage realization artifacts"
                    )
        if kind in ("TCB_BCRS", "TDB_COMPATIBLE_BARYCENTRIC_INERTIAL"):
            required_scale = "TCB" if kind == "TCB_BCRS" else "TDB"
            if scale != required_scale or origin != "SOLAR_SYSTEM_BARYCENTER":
                raise SolarSystemContractError(
                    "barycentric frame kind requires its exact scale and SSB origin"
                )
            if self.origin_naif_id != 0:
                raise SolarSystemContractError("solar-system barycenter requires NAIF observer 0")
            if not artifact_ids or coverage is None:
                raise SolarSystemContractError(
                    "physical barycentric realizations require artifacts and coverage"
                )
            if self.orientation_time_dependence != "STATIC":
                raise SolarSystemContractError("inertial barycentric axes must be STATIC")
        elif kind == "SYNTHETIC":
            if scale != "SYNTHETIC" or origin != "SYNTHETIC" or self.origin_naif_id is not None:
                raise SolarSystemContractError("synthetic frame context is inconsistent")
            if self.orientation_time_dependence != "STATIC":
                raise SolarSystemContractError("Milestone 1 synthetic inertial axes are STATIC")
        else:
            if origin == "NEWTONIAN_MODEL_BARYCENTER":
                if scale not in ("TCB", "TDB") or self.origin_naif_id is not None:
                    raise SolarSystemContractError(
                        "model-barycenter frames require TCB/TDB and no NAIF origin"
                    )
            else:
                if scale == "SYNTHETIC" or origin not in (
                    "BODY_CENTER",
                    "PLANETARY_SYSTEM_BARYCENTER",
                ):
                    raise SolarSystemContractError(
                        "origin-centered inertial frames require a physical body/system origin"
                    )
                if self.origin_naif_id is None:
                    raise SolarSystemContractError(
                        "physical body/system frames require a NAIF identifier"
                    )
                if origin == "PLANETARY_SYSTEM_BARYCENTER" and not (
                    1 <= self.origin_naif_id <= 9
                ):
                    raise SolarSystemContractError(
                        "planetary-system barycenter frames require NAIF IDs 1 through 9"
                    )
                if origin == "BODY_CENTER" and self.origin_naif_id <= 9:
                    raise SolarSystemContractError(
                        "body-center frames require canonical body IDs greater than 9"
                    )
            if self.orientation_time_dependence != "STATIC":
                raise SolarSystemContractError("Milestone 1 inertial frames are STATIC")


@dataclass(frozen=True, eq=False)
class ExactUnitScale:
    """One canonical exact rational scale to a named SI unit."""

    unit_id: str
    dimension: str
    si_unit_id: str
    numerator: int
    denominator: int
    definition_classification: str
    artifact_ids: tuple[str, ...]
    content_sha256: str = ""

    _SLUG: ClassVar[str] = "exact-unit-scale"

    def __post_init__(self) -> None:
        self._validate()
        _finish(self, self._SLUG, permit_empty=True)

    def _validate(self) -> None:
        _exact_type(self, ExactUnitScale, "exact unit scale")
        _text(self.unit_id, "unit_id")
        dimension = _token(self.dimension, _DIMENSIONS, "dimension")
        si_unit = _text(self.si_unit_id, "si_unit_id")
        if si_unit != _SI_UNIT_BY_DIMENSION[dimension]:
            raise SolarSystemContractError("dimension requires its canonical SI unit identifier")
        numerator = _integer(self.numerator, "numerator", minimum=1)
        denominator = _integer(self.denominator, "denominator", minimum=1)
        if gcd(numerator, denominator) != 1:
            raise SolarSystemContractError("exact unit scale must be a reduced fraction")
        definition = _token(
            self.definition_classification,
            _EXACT_DEFINITION_CLASSES,
            "definition_classification",
        )
        artifact_ids = _identifier_tuple(self.artifact_ids, "artifact_ids")
        if definition == "NOMINAL_EXACT" and not artifact_ids:
            raise SolarSystemContractError("nominal exact scales require source artifacts")

    @property
    def fraction(self) -> Fraction:
        """Return the exact scale without performing a unit conversion."""

        validate_integrity(self)
        return Fraction(self.numerator, self.denominator)


@dataclass(frozen=True, eq=False)
class UnitSystemDefinition:
    """Exact L/T/M/GM scale definitions for one coordinate-time convention."""

    unit_system_id: str
    length: ExactUnitScale
    time: ExactUnitScale
    mass: ExactUnitScale
    gravitational_parameter: ExactUnitScale
    coordinate_time_scale: str
    content_sha256: str = ""

    _SLUG: ClassVar[str] = "unit-system-definition"

    def __post_init__(self) -> None:
        self._validate()
        _finish(self, self._SLUG, permit_empty=True)

    def _validate(self) -> None:
        _exact_type(self, UnitSystemDefinition, "unit system definition")
        _text(self.unit_system_id, "unit_system_id")
        length = _require(self.length, ExactUnitScale, "length")
        time = _require(self.time, ExactUnitScale, "time")
        mass = _require(self.mass, ExactUnitScale, "mass")
        gm = _require(
            self.gravitational_parameter,
            ExactUnitScale,
            "gravitational_parameter",
        )
        assert all(type(item) is ExactUnitScale for item in (length, time, mass, gm))
        if (
            length.dimension != "LENGTH"
            or time.dimension != "TIME"
            or mass.dimension != "MASS"
            or gm.dimension != "GRAVITATIONAL_PARAMETER"
        ):
            raise SolarSystemContractError("unit-system dimensions are not canonical")
        if len({length.unit_id, time.unit_id, mass.unit_id, gm.unit_id}) != 4:
            raise SolarSystemContractError("unit-system unit identifiers must be distinct")
        expected_gm = length.fraction**3 / time.fraction**2
        if gm.fraction != expected_gm:
            raise SolarSystemContractError(
                "gravitational-parameter scale must equal length^3/time^2"
            )
        expected_definition = (
            "DEFINED_EXACT"
            if length.definition_classification == "DEFINED_EXACT"
            and time.definition_classification == "DEFINED_EXACT"
            else "NOMINAL_EXACT"
        )
        if gm.definition_classification != expected_definition:
            raise SolarSystemContractError(
                "GM scale classification must derive exactly from length and time scales"
            )
        _token(self.coordinate_time_scale, _TIME_SCALES, "coordinate_time_scale")


def _canonical_decimal(value: object, label: str) -> str:
    text = _text(value, label)
    if _DECIMAL.fullmatch(text) is None:
        raise SolarSystemContractError(f"{label} is not a canonical decimal literal")
    try:
        parsed = Decimal(text)
    except InvalidOperation as exc:
        raise SolarSystemContractError(f"{label} is not a finite decimal") from exc
    if not parsed.is_finite():
        raise SolarSystemContractError(f"{label} must be finite")
    if parsed.is_zero() and parsed.is_signed():
        raise SolarSystemContractError(f"{label} cannot use negative zero")
    return text


@dataclass(frozen=True, eq=False)
class PhysicalConstant:
    """One sourced decimal value and its exact rounded binary64 representation."""

    constant_id: str
    quantity_kind: str
    source_decimal: str
    operational_value: float
    unit: ExactUnitScale
    classification: str
    uncertainty_status: str
    uncertainty: float | None
    uncertainty_kind: str | None
    uncertainty_scale: str
    confidence_level: float | None
    covariance_group: str | None
    coordinate_scale_applicability: str
    coverage_status: str
    coverage: tuple[CoverageInterval, ...] | None
    artifact_ids: tuple[str, ...]
    content_sha256: str = ""

    _SLUG: ClassVar[str] = "physical-constant"

    def __post_init__(self) -> None:
        self._validate()
        _finish(self, self._SLUG, permit_empty=True)

    def _validate(self) -> None:
        _exact_type(self, PhysicalConstant, "physical constant")
        _text(self.constant_id, "constant_id")
        _token(self.quantity_kind, _QUANTITY_KINDS, "quantity_kind")
        source = _canonical_decimal(self.source_decimal, "source_decimal")
        operational = _number(self.operational_value, "operational_value")
        if operational == 0.0 and math.copysign(1.0, operational) < 0.0:
            raise SolarSystemContractError("operational constants cannot use negative zero")
        try:
            parsed_source = Decimal(source)
            rounded = float(parsed_source)
        except (OverflowError, ValueError) as exc:
            raise SolarSystemContractError("source_decimal does not round to binary64") from exc
        if parsed_source.is_zero() != (operational == 0.0):
            raise SolarSystemContractError(
                "source_decimal zero/nonzero status must survive binary64 rounding"
            )
        if not math.isfinite(rounded) or rounded.hex() != operational.hex():
            raise SolarSystemContractError(
                "operational_value must be the exact binary64 rounding of source_decimal"
            )
        unit = _require(self.unit, ExactUnitScale, "unit")
        assert type(unit) is ExactUnitScale
        expected_dimension = {
            "DIMENSIONLESS": "DIMENSIONLESS",
            "GRAVITATIONAL_PARAMETER": "GRAVITATIONAL_PARAMETER",
            "LENGTH": "LENGTH",
            "MASS": "MASS",
            "RADIUS": "LENGTH",
            "SPEED": "SPEED",
            "TIME": "TIME",
        }[self.quantity_kind]
        if unit.dimension != expected_dimension:
            raise SolarSystemContractError("physical constant unit dimension is inconsistent")
        classification = _token(self.classification, _CONSTANT_CLASSES, "classification")
        uncertainty_status = _token(
            self.uncertainty_status, _UNCERTAINTY_STATUSES, "uncertainty_status"
        )
        uncertainty_scale = _token(
            self.uncertainty_scale, _UNCERTAINTY_SCALES, "uncertainty_scale"
        )
        if classification in _EXACT_DEFINITION_CLASSES:
            if uncertainty_status != "EXACT_NOT_APPLICABLE":
                raise SolarSystemContractError(
                    "exact/nominal constants require exact-not-applicable uncertainty"
                )
            if any(
                item is not None
                for item in (
                    self.uncertainty,
                    self.uncertainty_kind,
                    self.confidence_level,
                    self.covariance_group,
                )
            ):
                raise SolarSystemContractError(
                    "exact/nominal constants cannot retain uncertainty fields"
                )
            if uncertainty_scale != "NOT_APPLICABLE":
                raise SolarSystemContractError(
                    "exact/nominal constants require uncertainty_scale NOT_APPLICABLE"
                )
        elif uncertainty_status == "PROVIDED":
            uncertainty = _number(self.uncertainty, "uncertainty", minimum=0.0)
            if uncertainty == 0.0:
                raise SolarSystemContractError("provided uncertainty must be positive")
            uncertainty_kind = _token(
                self.uncertainty_kind, _UNCERTAINTY_KINDS, "uncertainty_kind"
            )
            if uncertainty_scale != "ABSOLUTE_SAME_UNIT_AS_VALUE":
                raise SolarSystemContractError(
                    "provided uncertainty must be absolute in the constant's exact unit"
                )
            if uncertainty_kind == "SYMMETRIC_CONFIDENCE_HALF_WIDTH":
                if self.confidence_level is None:
                    raise SolarSystemContractError(
                        "symmetric confidence half-width requires an exact confidence level"
                    )
                confidence = _number(
                    self.confidence_level,
                    "confidence_level",
                    minimum=0.0,
                    maximum=1.0,
                )
                if confidence == 0.0 or confidence == 1.0:
                    raise SolarSystemContractError(
                        "confidence_level must be strictly between zero and one"
                    )
            elif self.confidence_level is not None:
                raise SolarSystemContractError(
                    "only symmetric confidence half-width can retain confidence_level"
                )
            if self.covariance_group is not None:
                _text(self.covariance_group, "covariance_group")
        elif uncertainty_status == "NOT_PROVIDED":
            if any(
                item is not None
                for item in (
                    self.uncertainty,
                    self.uncertainty_kind,
                    self.confidence_level,
                    self.covariance_group,
                )
            ):
                raise SolarSystemContractError(
                    "not-provided uncertainty requires exact None fields"
                )
            if uncertainty_scale != "NOT_APPLICABLE":
                raise SolarSystemContractError(
                    "not-provided uncertainty requires uncertainty_scale NOT_APPLICABLE"
                )
        else:
            raise SolarSystemContractError(
                "estimated/fitted constants require PROVIDED or NOT_PROVIDED uncertainty"
            )
        applicability = _token(
            self.coordinate_scale_applicability,
            _COORDINATE_SCALE_APPLICABILITY,
            "coordinate_scale_applicability",
        )
        coverage = _coverage_tuple(self.coverage_status, self.coverage, "coverage")
        artifact_ids = _identifier_tuple(
            self.artifact_ids, "artifact_ids", allow_empty=False
        )
        if not set(unit.artifact_ids).issubset(artifact_ids):
            raise SolarSystemContractError(
                "physical-constant artifacts must include unit-definition artifacts"
            )
        if coverage is not None:
            expected_scale = {
                "TCB_COMPATIBLE": "TCB",
                "TDB_COMPATIBLE": "TDB",
            }.get(applicability)
            if expected_scale is not None and any(
                interval.start.time_scale != expected_scale for interval in coverage
            ):
                raise SolarSystemContractError(
                    "constant coverage conflicts with coordinate-scale applicability"
                )
            for interval in coverage:
                if not set(interval.start.artifact_ids).issubset(artifact_ids):
                    raise SolarSystemContractError(
                        "constant artifacts must include coverage realization artifacts"
                    )
        if self.quantity_kind in (
            "GRAVITATIONAL_PARAMETER",
            "MASS",
            "RADIUS",
            "SPEED",
        ) and operational <= 0.0:
            raise SolarSystemContractError(f"{self.quantity_kind} constants must be positive")


@dataclass(frozen=True, eq=False)
class BodyParameter:
    """Explicit body-role parameters without deriving mass from GM or G."""

    body_id: str
    naif_id: int
    body_role: str
    gravitational_parameter_status: str
    gravitational_parameter: PhysicalConstant | None
    mass_status: str
    mass: PhysicalConstant | None
    radius_status: str
    radius_kind: str | None
    radius: PhysicalConstant | None
    artifact_ids: tuple[str, ...]
    content_sha256: str = ""

    _SLUG: ClassVar[str] = "body-parameter"

    def __post_init__(self) -> None:
        self._validate()
        _finish(self, self._SLUG, permit_empty=True)

    def _validate(self) -> None:
        _exact_type(self, BodyParameter, "body parameter")
        _text(self.body_id, "body_id")
        _integer(
            self.naif_id,
            "naif_id",
            minimum=-MAXIMUM_NAIF_IDENTIFIER_MAGNITUDE,
            maximum=MAXIMUM_NAIF_IDENTIFIER_MAGNITUDE,
        )
        role = _token(self.body_role, _BODY_ROLES, "body_role")
        gm_status = _token(
            self.gravitational_parameter_status,
            _GM_STATUSES,
            "gravitational_parameter_status",
        )
        retained_artifacts = _identifier_tuple(
            self.artifact_ids, "artifact_ids", allow_empty=False
        )
        if gm_status == "PROVIDED_DYNAMICS_PARAMETER":
            gm = _require(
                self.gravitational_parameter,
                PhysicalConstant,
                "gravitational_parameter",
            )
            assert type(gm) is PhysicalConstant
            if gm.quantity_kind != "GRAVITATIONAL_PARAMETER" or gm.operational_value <= 0.0:
                raise SolarSystemContractError("body GM must be a strictly positive GM constant")
            if gm.coordinate_scale_applicability not in (
                "TCB_COMPATIBLE",
                "TDB_COMPATIBLE",
            ):
                raise SolarSystemContractError("body GM requires explicit coordinate scaling")
            if gm.classification not in ("ESTIMATED", "FITTED"):
                raise SolarSystemContractError(
                    "physical body GM must be an estimated or fitted parameter"
                )
            if gm.coverage_status == "NOT_RETAINED_BLOCKED":
                raise SolarSystemContractError("body GM source coverage cannot be blocked")
            if not set(gm.artifact_ids).issubset(retained_artifacts):
                raise SolarSystemContractError("body artifacts must include GM source artifacts")
            if role == "MASSLESS_TEST_PARTICLE":
                raise SolarSystemContractError("massless targets cannot provide a dynamics GM")
        else:
            if self.gravitational_parameter is not None or role != "MASSLESS_TEST_PARTICLE":
                raise SolarSystemContractError(
                    "absent GM status is restricted to massless test particles"
                )
        if role == "PLANETARY_SYSTEM_BARYCENTER" and not (1 <= self.naif_id <= 9):
            raise SolarSystemContractError(
                "planetary-system barycenters require canonical NAIF IDs 1 through 9"
            )
        if role == "MASSIVE_BODY_CENTER" and self.naif_id <= 9:
            raise SolarSystemContractError(
                "massive body centers cannot use SSB/system-barycenter NAIF IDs"
            )
        if role == "MASSLESS_TEST_PARTICLE" and self.naif_id >= 0:
            raise SolarSystemContractError(
                "Milestone 1 massless test particles require explicit negative synthetic IDs"
            )
        mass_status = _token(self.mass_status, _MASS_STATUSES, "mass_status")
        if role == "MASSLESS_TEST_PARTICLE" and mass_status != "NOT_PROVIDED_NOT_USED":
            raise SolarSystemContractError(
                "massless test particles cannot retain a physical mass parameter"
            )
        if mass_status == "PROVIDED_NOT_USED_BY_GM_DYNAMICS":
            mass = _require(self.mass, PhysicalConstant, "mass")
            assert type(mass) is PhysicalConstant
            if mass.quantity_kind != "MASS" or mass.operational_value <= 0.0:
                raise SolarSystemContractError("mass must be a strictly positive mass constant")
            if mass.classification not in ("ESTIMATED", "FITTED"):
                raise SolarSystemContractError(
                    "physical body mass must be an estimated or fitted parameter"
                )
            if mass.coverage_status == "NOT_RETAINED_BLOCKED":
                raise SolarSystemContractError("body mass source coverage cannot be blocked")
            if not set(mass.artifact_ids).issubset(retained_artifacts):
                raise SolarSystemContractError("body artifacts must include mass source artifacts")
        elif self.mass is not None:
            raise SolarSystemContractError("not-provided mass requires mass=None")
        radius_status = _token(self.radius_status, _RADIUS_STATUSES, "radius_status")
        if (
            role == "PLANETARY_SYSTEM_BARYCENTER"
            and radius_status != "POINT_MASS_NO_COLLISION_RADIUS"
        ):
            raise SolarSystemContractError(
                "planetary-system barycenters cannot retain a surface/collision radius"
            )
        if radius_status == "PROVIDED_REFERENCE_RADIUS":
            radius_kind = _token(self.radius_kind, _RADIUS_KINDS, "radius_kind")
            radius = _require(self.radius, PhysicalConstant, "radius")
            assert type(radius) is PhysicalConstant
            if radius.quantity_kind != "RADIUS" or radius.operational_value <= 0.0:
                raise SolarSystemContractError("radius must be a strictly positive radius constant")
            if radius.coverage_status == "NOT_RETAINED_BLOCKED":
                raise SolarSystemContractError("body radius source coverage cannot be blocked")
            if radius_kind == "NOMINAL_REFERENCE":
                if radius.classification != "NOMINAL_EXACT":
                    raise SolarSystemContractError(
                        "nominal reference radius requires NOMINAL_EXACT classification"
                    )
            elif radius.classification not in ("ESTIMATED", "FITTED"):
                raise SolarSystemContractError(
                    "physical measured/model radius must be estimated or fitted"
                )
            if not set(radius.artifact_ids).issubset(retained_artifacts):
                raise SolarSystemContractError("body artifacts must include radius source artifacts")
        elif self.radius_kind is not None or self.radius is not None:
            raise SolarSystemContractError(
                "point-mass radius status requires radius_kind=None and radius=None"
            )


@dataclass(frozen=True, eq=False)
class EphemerisProviderSpec:
    """Declarative exact local/offline provider specification; never resolved here."""

    spec_id: str
    identity: ProviderIdentity
    artifacts: tuple[ArtifactBinding, ...]
    ordered_load_artifact_ids: tuple[str, ...]
    capabilities: tuple[str, ...]
    native_time_scale: str
    native_frame: FrameRealization
    native_output_unit_system: UnitSystemDefinition
    network_access: bool
    fallback_allowed: bool
    extrapolation_policy: str
    evidence_class: str
    registry_authorized: bool
    qualification_authorized: bool
    content_sha256: str = ""

    _SLUG: ClassVar[str] = "ephemeris-provider-spec"

    def __post_init__(self) -> None:
        self._validate()
        _finish(self, self._SLUG, permit_empty=True)

    def _validate(self) -> None:
        _exact_type(self, EphemerisProviderSpec, "ephemeris provider spec")
        _text(self.spec_id, "spec_id")
        identity = _require(self.identity, ProviderIdentity, "identity")
        assert type(identity) is ProviderIdentity
        if type(self.artifacts) is not tuple:
            raise SolarSystemContractError("artifacts must be an exact tuple")
        if not self.artifacts or len(self.artifacts) > MAXIMUM_ARTIFACT_COUNT:
            raise SolarSystemContractError("artifacts has an invalid count")
        by_id: dict[str, ArtifactBinding] = {}
        by_locator: dict[str, ArtifactBinding] = {}
        for index, artifact in enumerate(self.artifacts):
            checked = _require(artifact, ArtifactBinding, f"artifacts[{index}]")
            assert type(checked) is ArtifactBinding
            if checked.artifact_id in by_id:
                raise SolarSystemContractError("provider artifacts cannot repeat an ID")
            if checked.provider_id != identity.provider_id:
                raise SolarSystemContractError("provider artifact owner is inconsistent")
            prior_locator = by_locator.get(checked.logical_locator)
            if prior_locator is not None:
                raise SolarSystemContractError(
                    "provider artifacts cannot repeat a normalized logical locator"
                )
            by_id[checked.artifact_id] = checked
            by_locator[checked.logical_locator] = checked
        if tuple(by_id) != tuple(sorted(by_id)):
            raise SolarSystemContractError("provider artifacts must be sorted by artifact_id")
        implementation_ids = identity.ordered_implementation_artifact_ids
        if any(item not in by_id for item in implementation_ids):
            raise SolarSystemContractError("provider implementation artifact is missing")
        primary_implementation = by_id[implementation_ids[0]]
        if (
            primary_implementation.artifact_role
            not in ("SOFTWARE_DISTRIBUTION", "SOFTWARE_SOURCE")
            or primary_implementation.version != identity.version
        ):
            raise SolarSystemContractError(
                "the first implementation artifact must be the matching provider source/distribution"
            )
        if any(
            by_id[item].artifact_role
            not in ("LICENSE", "NATIVE_LIBRARY", "SOFTWARE_DISTRIBUTION", "SOFTWARE_SOURCE")
            for item in implementation_ids
        ):
            raise SolarSystemContractError("implementation roster contains a data artifact")
        if not any(
            by_id[item].artifact_role
            in ("NATIVE_LIBRARY", "SOFTWARE_DISTRIBUTION", "SOFTWARE_SOURCE")
            for item in implementation_ids
        ):
            raise SolarSystemContractError(
                "implementation roster requires executable/source implementation evidence"
            )
        if any(by_id[item].locator_kind != "LOCAL_REGULAR_FILE" for item in implementation_ids):
            raise SolarSystemContractError("implementation artifacts must be locally retained")
        if any(
            by_id[item].artifact_role != "LICENSE"
            and by_id[item].license_evidence_status
            != "RETAINED_HASH_BOUND_LICENSE_ARTIFACT"
            for item in implementation_ids
        ):
            raise SolarSystemContractError(
                "executable/source implementation bytes require retained license evidence"
            )
        for artifact in self.artifacts:
            if artifact.license_artifact_id is not None:
                license_artifact = by_id.get(artifact.license_artifact_id)
                if license_artifact is None or license_artifact.artifact_role != "LICENSE":
                    raise SolarSystemContractError("retained license artifact is missing")
                if license_artifact.license_spdx != artifact.license_spdx:
                    raise SolarSystemContractError("license SPDX evidence is inconsistent")
                if (
                    license_artifact.license_evidence_status
                    != "RETAINED_LICENSE_TEXT_SELF_EVIDENCE_NONAUTHORIZING"
                    or license_artifact.locator_kind != "LOCAL_REGULAR_FILE"
                ):
                    raise SolarSystemContractError(
                        "referenced license evidence must be a retained local license leaf"
                    )
        load_ids = _identifier_tuple(
            self.ordered_load_artifact_ids,
            "ordered_load_artifact_ids",
            allow_empty=identity.provider_kind == "SYNTHETIC_TEST",
            sorted_unique=False,
        )
        load_members = tuple(
            sorted(
                (
                    artifact
                    for artifact in self.artifacts
                    if artifact.load_order_status == "ORDERED_LOAD_MEMBER"
                ),
                key=lambda artifact: artifact.load_order,
            )
        )
        expected_load_ids = tuple(artifact.artifact_id for artifact in load_members)
        if load_ids != expected_load_ids:
            raise SolarSystemContractError("provider load roster/order is not exact")
        if tuple(artifact.load_order for artifact in load_members) != tuple(
            range(len(load_members))
        ):
            raise SolarSystemContractError("provider load order must be contiguous from zero")
        if type(self.capabilities) is not tuple or not self.capabilities:
            raise SolarSystemContractError("capabilities must be a nonempty exact tuple")
        if len(self.capabilities) > len(_PROVIDER_CAPABILITIES):
            raise SolarSystemContractError("capabilities exceeds its closed count")
        capabilities = tuple(
            _token(item, _PROVIDER_CAPABILITIES, f"capabilities[{index}]")
            for index, item in enumerate(self.capabilities)
        )
        if capabilities != tuple(sorted(set(capabilities))):
            raise SolarSystemContractError("capabilities must be sorted and unique")
        native_scale = _token(self.native_time_scale, _TIME_SCALES, "native_time_scale")
        frame = _require(self.native_frame, FrameRealization, "native_frame")
        units = _require(
            self.native_output_unit_system,
            UnitSystemDefinition,
            "native_output_unit_system",
        )
        assert type(frame) is FrameRealization and type(units) is UnitSystemDefinition
        if frame.coordinate_time_scale != native_scale or units.coordinate_time_scale != native_scale:
            raise SolarSystemContractError("provider native scale/frame/units are inconsistent")
        referenced = set(identity.ordered_implementation_artifact_ids)
        referenced.update(frame.artifact_ids)
        if frame.coverage is not None:
            for interval in frame.coverage:
                referenced.update(interval.start.artifact_ids)
        for scale in (units.length, units.time, units.mass, units.gravitational_parameter):
            referenced.update(scale.artifact_ids)
        for artifact in self.artifacts:
            if artifact.coverage is not None:
                for interval in artifact.coverage:
                    referenced.update(interval.start.artifact_ids)
        if not referenced.issubset(by_id):
            raise SolarSystemContractError("provider nested context references missing artifacts")
        time_evidence_ids = {
            artifact_id
            for artifact in self.artifacts
            if artifact.coverage is not None
            for interval in artifact.coverage
            for artifact_id in interval.start.artifact_ids
        }
        if frame.coverage is not None:
            time_evidence_ids.update(
                artifact_id
                for interval in frame.coverage
                for artifact_id in interval.start.artifact_ids
            )
        if any(
            by_id[artifact_id].artifact_role
            not in _TIME_REALIZATION_ARTIFACT_ROLES
            for artifact_id in time_evidence_ids
        ):
            raise SolarSystemContractError(
                "epoch artifact_ids are restricted to time-realization evidence"
            )
        if identity.provider_kind == "LOCAL_OFFLINE_SPK":
            if not any(artifact.artifact_role == "SPK" for artifact in self.artifacts):
                raise SolarSystemContractError("LOCAL_OFFLINE_SPK requires an SPK artifact")
            if "GEOMETRIC_CARTESIAN_STATE" not in capabilities:
                raise SolarSystemContractError("SPK provider lacks geometric-state capability")
            if capabilities != (
                "COARSE_ARTIFACT_TIME_ENVELOPE",
                "GEOMETRIC_CARTESIAN_STATE",
                "TDB_COORDINATE_EPOCH",
            ):
                raise SolarSystemContractError(
                    "LOCAL_OFFLINE_SPK requires the exact v1 capability roster"
                )
            if native_scale != "TDB" or frame.frame_kind != "TDB_COMPATIBLE_BARYCENTRIC_INERTIAL":
                raise SolarSystemContractError(
                    "LOCAL_OFFLINE_SPK v1 requires TDB-compatible SSB inertial context"
                )
            if frame.origin_kind != "SOLAR_SYSTEM_BARYCENTER" or frame.origin_naif_id != 0:
                raise SolarSystemContractError("LOCAL_OFFLINE_SPK v1 requires observer origin 0")
            spk_intervals = tuple(
                interval
                for artifact in self.artifacts
                if artifact.artifact_role == "SPK" and artifact.coverage is not None
                for interval in artifact.coverage
            )
            if not spk_intervals or any(
                interval.start.representation
                != "SPICE_TDB_J2000_OFFSET_SECONDS_TWO_PART"
                or interval.start.coordinate_unit_id != _DEFINED_SECOND_UNIT_ID
                or interval.start.origin_id != _SPICE_TDB_OFFSET_ORIGIN_ID
                for interval in spk_intervals
            ):
                raise SolarSystemContractError(
                    "LOCAL_OFFLINE_SPK coarse coverage must retain native TDB ET seconds"
                )
            if frame.coverage is None or any(
                interval.start.representation
                != "SPICE_TDB_J2000_OFFSET_SECONDS_TWO_PART"
                or interval.start.coordinate_unit_id != _DEFINED_SECOND_UNIT_ID
                or interval.start.origin_id != _SPICE_TDB_OFFSET_ORIGIN_ID
                for interval in frame.coverage
            ):
                raise SolarSystemContractError(
                    "LOCAL_OFFLINE_SPK frame coverage must retain native TDB ET seconds"
                )
            if any(
                artifact.locator_kind != "LOCAL_REGULAR_FILE"
                for artifact in self.artifacts
                if artifact.load_order_status == "ORDERED_LOAD_MEMBER"
            ):
                raise SolarSystemContractError("SPK load artifacts must be locally retained")
        else:
            if native_scale != "SYNTHETIC" or frame.frame_kind != "SYNTHETIC":
                raise SolarSystemContractError(
                    "SYNTHETIC_TEST cannot claim a physical time/frame realization"
                )
            if any(
                artifact.artifact_role in _ORDERED_LOAD_ROLES for artifact in self.artifacts
            ) or any(capability != "GEOMETRIC_CARTESIAN_STATE" for capability in capabilities):
                raise SolarSystemContractError(
                    "SYNTHETIC_TEST cannot claim physical kernel capabilities"
                )
            if frame.coverage_status != "NOT_APPLICABLE" or frame.coverage is not None:
                raise SolarSystemContractError(
                    "SYNTHETIC_TEST frame coverage is explicitly not applicable in v1"
                )
        _bool(self.network_access, False, "network_access")
        _bool(self.fallback_allowed, False, "fallback_allowed")
        _token(
            self.extrapolation_policy,
            _EXTRAPOLATION_POLICIES,
            "extrapolation_policy",
        )
        _token(self.evidence_class, _EVIDENCE_CLASSES, "evidence_class")
        _bool(self.registry_authorized, False, "registry_authorized")
        _bool(self.qualification_authorized, False, "qualification_authorized")


@dataclass(frozen=True, eq=False)
class EphemerisQuerySpec:
    """Declarative provider-native geometric-state query; it executes nothing."""

    query_id: str
    provider: EphemerisProviderSpec
    epoch: CoordinateEpoch
    target_naif_ids: tuple[int, ...]
    observer_naif_id: int | None
    frame: FrameRealization
    aberration_correction: str
    output_unit_system: UnitSystemDefinition
    state_kind: str
    target_chain_availability_status: str
    content_sha256: str = ""

    _SLUG: ClassVar[str] = "ephemeris-query-spec"

    def __post_init__(self) -> None:
        self._validate()
        _finish(self, self._SLUG, permit_empty=True)

    def _validate(self) -> None:
        _exact_type(self, EphemerisQuerySpec, "ephemeris query spec")
        _text(self.query_id, "query_id")
        provider = _require(self.provider, EphemerisProviderSpec, "provider")
        epoch = _require(self.epoch, CoordinateEpoch, "epoch")
        frame = _require(self.frame, FrameRealization, "frame")
        units = _require(self.output_unit_system, UnitSystemDefinition, "output_unit_system")
        assert type(provider) is EphemerisProviderSpec
        assert type(epoch) is CoordinateEpoch
        assert type(frame) is FrameRealization
        assert type(units) is UnitSystemDefinition
        if type(self.target_naif_ids) is not tuple:
            raise SolarSystemContractError("target_naif_ids must be an exact tuple")
        if not self.target_naif_ids or len(self.target_naif_ids) > MAXIMUM_BODY_COUNT:
            raise SolarSystemContractError("target_naif_ids has an invalid count")
        targets = tuple(
            _integer(
                item,
                f"target_naif_ids[{index}]",
                minimum=-MAXIMUM_NAIF_IDENTIFIER_MAGNITUDE,
                maximum=MAXIMUM_NAIF_IDENTIFIER_MAGNITUDE,
            )
            for index, item in enumerate(self.target_naif_ids)
        )
        if len(set(targets)) != len(targets):
            raise SolarSystemContractError("target_naif_ids cannot contain duplicates")
        if provider.identity.provider_kind == "LOCAL_OFFLINE_SPK":
            observer = _integer(
                self.observer_naif_id,
                "observer_naif_id",
                minimum=-MAXIMUM_NAIF_IDENTIFIER_MAGNITUDE,
                maximum=MAXIMUM_NAIF_IDENTIFIER_MAGNITUDE,
            )
            if observer in targets:
                raise SolarSystemContractError("observer cannot also be a target")
            if observer != 0 or frame.origin_naif_id != 0:
                raise SolarSystemContractError(
                    "LOCAL_OFFLINE_SPK v1 observer and frame origin must be NAIF 0"
                )
        elif self.observer_naif_id is not None or frame.origin_naif_id is not None:
            raise SolarSystemContractError(
                "SYNTHETIC_TEST query observer and frame origin must be None"
            )
        if provider.identity.provider_kind == "SYNTHETIC_TEST" and any(
            target >= 0 for target in targets
        ):
            raise SolarSystemContractError(
                "synthetic query targets require explicit negative synthetic IDs"
            )
        provider_artifact_ids = {artifact.artifact_id for artifact in provider.artifacts}
        if not set(epoch.artifact_ids).issubset(provider_artifact_ids):
            raise SolarSystemContractError("query epoch references unknown provider artifacts")
        provider_artifact_roles = {
            artifact.artifact_id: artifact.artifact_role
            for artifact in provider.artifacts
        }
        if any(
            provider_artifact_roles[artifact_id]
            not in _TIME_REALIZATION_ARTIFACT_ROLES
            for artifact_id in epoch.artifact_ids
        ):
            raise SolarSystemContractError(
                "query epoch artifacts must be time-realization evidence, not state data"
            )
        if provider.identity.provider_kind == "SYNTHETIC_TEST" and epoch.artifact_ids:
            raise SolarSystemContractError(
                "synthetic query epochs cannot claim physical realization artifacts"
            )
        if epoch.time_scale != provider.native_time_scale:
            raise SolarSystemContractError("query epoch must use provider-native time scale")
        if frame.content_sha256 != provider.native_frame.content_sha256:
            raise SolarSystemContractError("query frame must be the exact provider-native frame")
        if units.content_sha256 != provider.native_output_unit_system.content_sha256:
            raise SolarSystemContractError(
                "query output units must be the exact provider-native unit system"
            )
        if provider.identity.provider_kind == "LOCAL_OFFLINE_SPK":
            if (
                epoch.representation
                != "SPICE_TDB_J2000_OFFSET_SECONDS_TWO_PART"
                or epoch.coordinate_unit_id != _DEFINED_SECOND_UNIT_ID
                or epoch.origin_id != _SPICE_TDB_OFFSET_ORIGIN_ID
            ):
                raise SolarSystemContractError(
                    "SPK queries require exact native TDB ET seconds from SPICE J2000"
                )
        elif epoch.coordinate_unit_id != units.time.unit_id:
            raise SolarSystemContractError(
                "synthetic epoch coordinate unit must equal the provider-native time unit"
            )
        if provider.identity.provider_kind == "SYNTHETIC_TEST" and (
            epoch.origin_id != f"{provider.spec_id}.epoch-zero"
            or epoch.realization_id != f"{provider.spec_id}.epoch-realization"
        ):
            raise SolarSystemContractError(
                "synthetic query epoch context must be derived from provider spec_id"
            )
        _token(
            self.aberration_correction,
            _ABERRATION_CORRECTIONS,
            "aberration_correction",
        )
        _token(self.state_kind, _STATE_KINDS, "state_kind")
        _token(
            self.target_chain_availability_status,
            _TARGET_CHAIN_AVAILABILITY_STATUSES,
            "target_chain_availability_status",
        )
        if "GEOMETRIC_CARTESIAN_STATE" not in provider.capabilities:
            raise SolarSystemContractError("provider lacks geometric-state capability")
        if provider.identity.provider_kind == "LOCAL_OFFLINE_SPK":
            if frame.coverage is None or not any(
                _coverage_contains(interval, epoch) for interval in frame.coverage
            ):
                raise SolarSystemCoverageError(
                    "query epoch is outside provider-frame coverage"
                )
            covered = any(
                artifact.artifact_role == "SPK"
                and artifact.coverage is not None
                and any(_coverage_contains(interval, epoch) for interval in artifact.coverage)
                for artifact in provider.artifacts
            )
            if not covered:
                raise SolarSystemCoverageError(
                    "query epoch is outside the retained coarse SPK time envelope; "
                    "target/center-chain availability remains runtime-validated"
                )


_CONTRACT_TYPES = (
    ArtifactBinding,
    BodyParameter,
    CoordinateEpoch,
    CoverageInterval,
    EphemerisProviderSpec,
    EphemerisQuerySpec,
    ExactUnitScale,
    FrameRealization,
    PhysicalConstant,
    ProviderIdentity,
    UnitSystemDefinition,
)


def validate_integrity(value: object) -> None:
    """Revalidate exact schema, nested custody, and the self-excluding seal."""

    if type(value) not in _CONTRACT_TYPES:
        raise SolarSystemContractError("value is not an exact Solar-System contract type")
    value._validate()  # type: ignore[attr-defined]
    _finish(value, value._SLUG, permit_empty=False)  # type: ignore[attr-defined]


__all__ = [
    "ArtifactBinding",
    "BodyParameter",
    "CoordinateEpoch",
    "CoverageInterval",
    "EphemerisProviderSpec",
    "EphemerisQuerySpec",
    "ExactUnitScale",
    "FrameRealization",
    "MAXIMUM_ARTIFACT_COUNT",
    "MAXIMUM_BODY_COUNT",
    "MAXIMUM_COVERAGE_INTERVAL_COUNT",
    "PhysicalConstant",
    "ProviderIdentity",
    "SolarSystemContractError",
    "SolarSystemCoverageError",
    "SolarSystemDataError",
    "SolarSystemDependencyUnavailableError",
    "UnitSystemDefinition",
    "validate_integrity",
]
