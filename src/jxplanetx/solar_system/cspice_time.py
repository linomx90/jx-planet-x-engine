"""Exact requested-query projection onto CSPICE's binary64 ET interface.

An M1 ``EphemerisQuerySpec`` retains a two-part TDB coordinate epoch.  CSPICE
accepts one binary64 ephemeris-time argument.  This unpublished arithmetic
boundary therefore keeps the requested query intact, realizes its exact
two-part ET coordinate with the audited M2 nearest-ties-to-even operation,
and constructs a separately sealed effective query at that exact binary64
value.

The signed error retained here is ``effective ET - requested ET`` in defined
SI seconds.  It is only interface-realization loss.  It is not a time-scale
or origin conversion, an ephemeris interpolation or state-error bound, a
physical uncertainty, or evidence that rounding is acceptable for a caller's
purpose.  Exact-only callers validate the receipt and then require the nested
``binary64_projection.rounding_status`` to be ``"EXACT"``.

This module imports no astronomy provider and does not execute one.  It
performs no file, kernel, network, frame, unit, or state operation.  It proves no target-specific
coverage, provider availability or execution, artifact-at-use custody,
license or registry authority, physical accuracy, or qualification.  Future
provider-native state and execution receipts must bind ``effective_query``;
they must never relabel a rounded result as a state at ``requested_query``.
All digests are forgeable unauthenticated content integrity only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from fractions import Fraction
from math import gcd
from typing import ClassVar

from .contracts import (
    CoordinateEpoch,
    EphemerisQuerySpec,
    SolarSystemContractError,
    validate_integrity,
)
from .serialization import (
    MAXIMUM_CANONICAL_EXACT_INTEGER_BITS,
    MAXIMUM_CANONICAL_STRING_CODEPOINTS,
    domain_sha256,
    validate_sha256,
)
from .units import (
    SECOND,
    Binary64UnitConversionReceipt,
    convert_fraction_to_binary64,
)


_MAXIMUM_INTERMEDIATE_BITS = 16_384
_MAXIMUM_IDENTIFIER_CODEPOINTS = 256

_PROJECTION_STAGE = (
    "REQUESTED_SPICE_TDB_ET_SECONDS_TO_EFFECTIVE_CSPICE_BINARY64_ET"
)
_ERROR_UNIT_ID = "unit.second.defined-exact.v1"
_TIME_SCALE_OPERATION_STATUS = (
    "NO_TIME_SCALE_OR_ORIGIN_CONVERSION_SAME_TDB_ET_COORDINATE"
)
_PROVIDER_EXECUTION_STATUS = "NOT_EXECUTED_ARITHMETIC_INTERFACE_PROJECTION_ONLY"
_EVIDENCE_PRESERVATION_STATUS = (
    "REQUESTED_EPOCH_ARTIFACT_IDS_PRESERVED_UNCHANGED_NO_NEW_EVIDENCE"
)
_ARTIFACT_CUSTODY_STATUS = "NO_ARTIFACT_AT_USE_CUSTODY_EVIDENCE"
_CONTENT_INTEGRITY_CLASS = "UNAUTHENTICATED_CONTENT_INTEGRITY_ONLY"

_RECEIPT_DOMAIN = (
    "jxplanetx.solar-system.cspice-time.binary64-query-projection-receipt."
    "content-integrity.v1"
)
_RECEIPT_SCHEMA = "CspiceBinary64QueryProjectionReceipt.v1"
_RECEIPT_QUALIFIED_NAME = (
    "jxplanetx.solar_system.cspice_time.CspiceBinary64QueryProjectionReceipt"
)
_BINARY64_RECEIPT_QUALIFIED_NAME = (
    "jxplanetx.solar_system.units.Binary64UnitConversionReceipt"
)


def _preflight_optional_sha256(value: object, label: str) -> str:
    if type(value) is not str:
        raise SolarSystemContractError(f"{label} must be an exact string")
    if value != "":
        validate_sha256(value, label)
    return value


def _bounded_ascii_identifier(value: object, label: str) -> str:
    if type(value) is not str:
        raise SolarSystemContractError(f"{label} must be an exact string")
    if len(value) > _MAXIMUM_IDENTIFIER_CODEPOINTS:
        raise SolarSystemContractError(f"{label} exceeds its code-point cap")
    if (
        not value
        or value.strip() != value
        or any(ord(character) < 0x20 or ord(character) > 0x7E for character in value)
    ):
        raise SolarSystemContractError(
            f"{label} must be nonempty trimmed printable ASCII text"
        )
    return value


def _exact_token(value: object, expected: str, label: str) -> str:
    if type(value) is not str:
        raise SolarSystemContractError(f"{label} must be an exact string")
    if len(value) > MAXIMUM_CANONICAL_STRING_CODEPOINTS:
        raise SolarSystemContractError(f"{label} exceeds its code-point cap")
    if value != expected:
        raise SolarSystemContractError(f"{label} is not the exact required token")
    return value


def _preflight_product(left: int, right: int, label: str) -> None:
    if type(left) is not int or type(right) is not int:
        raise SolarSystemContractError(f"{label} operands must be exact integers")
    if left and right and left.bit_length() + right.bit_length() > _MAXIMUM_INTERMEDIATE_BITS:
        raise SolarSystemContractError(f"{label} exceeds the intermediate arithmetic cap")


def _retained_fraction(value: Fraction, label: str) -> Fraction:
    if type(value) is not Fraction:
        raise SolarSystemContractError(f"{label} must be an exact Fraction")
    if (
        value.numerator.bit_length() > MAXIMUM_CANONICAL_EXACT_INTEGER_BITS
        or value.denominator.bit_length() > MAXIMUM_CANONICAL_EXACT_INTEGER_BITS
    ):
        raise SolarSystemContractError(f"{label} exceeds the retained rational cap")
    return value


def _require_reduced_pair(
    numerator: object,
    denominator: object,
    label: str,
) -> tuple[int, int]:
    if type(numerator) is not int or type(denominator) is not int:
        raise SolarSystemContractError(f"{label} components must be exact integers")
    if (
        numerator.bit_length() > MAXIMUM_CANONICAL_EXACT_INTEGER_BITS
        or denominator.bit_length() > MAXIMUM_CANONICAL_EXACT_INTEGER_BITS
    ):
        raise SolarSystemContractError(f"{label} exceeds the retained integer cap")
    if denominator <= 0:
        raise SolarSystemContractError(f"{label} denominator must be strictly positive")
    if gcd(numerator, denominator) != 1:
        raise SolarSystemContractError(f"{label} components must be reduced")
    if numerator == 0 and denominator != 1:
        raise SolarSystemContractError(f"{label} zero must be canonical 0/1")
    return numerator, denominator


def _requested_et_fraction(query: EphemerisQuerySpec) -> Fraction:
    epoch = query.epoch
    fraction = Fraction.from_float(epoch.fraction)
    _preflight_product(epoch.whole, fraction.denominator, "requested ET whole product")
    product = epoch.whole * fraction.denominator
    if max(product.bit_length(), fraction.numerator.bit_length()) + 1 > _MAXIMUM_INTERMEDIATE_BITS:
        raise SolarSystemContractError(
            "requested ET numerator addition exceeds the intermediate arithmetic cap"
        )
    numerator = product + fraction.numerator
    if numerator.bit_length() > _MAXIMUM_INTERMEDIATE_BITS:
        raise SolarSystemContractError(
            "requested ET numerator exceeds the intermediate arithmetic cap"
        )
    return _retained_fraction(
        Fraction(numerator, fraction.denominator),
        "requested ET",
    )


def _exact_projection_error(
    requested: Fraction,
    effective: Fraction,
) -> Fraction:
    _preflight_product(
        effective.numerator,
        requested.denominator,
        "projection error effective numerator product",
    )
    _preflight_product(
        requested.numerator,
        effective.denominator,
        "projection error requested numerator product",
    )
    _preflight_product(
        effective.denominator,
        requested.denominator,
        "projection error denominator product",
    )
    return _retained_fraction(effective - requested, "exact projection error")


def _canonical_epoch_parts(value: Fraction) -> tuple[int, float]:
    if type(value) is not Fraction:
        raise SolarSystemContractError("effective ET must be an exact Fraction")
    if value.numerator.bit_length() + 1 > _MAXIMUM_INTERMEDIATE_BITS:
        raise SolarSystemContractError(
            "effective ET canonical split exceeds its numerator cap"
        )
    if value.denominator.bit_length() + 1 > _MAXIMUM_INTERMEDIATE_BITS:
        raise SolarSystemContractError(
            "effective ET canonical split exceeds its denominator cap"
        )
    doubled_numerator = value.numerator << 1
    doubled_denominator = value.denominator << 1
    whole = (doubled_numerator + value.denominator) // doubled_denominator
    remainder = value - whole
    if not Fraction(-1, 2) <= remainder < Fraction(1, 2):
        raise SolarSystemContractError(
            "effective ET canonical remainder is outside the half-open interval"
        )
    if remainder == 0:
        return whole, 0.0
    remainder_float = float(remainder)
    if (
        not math.isfinite(remainder_float)
        or Fraction.from_float(remainder_float) != remainder
        or (remainder_float == 0.0 and math.copysign(1.0, remainder_float) < 0.0)
    ):
        raise SolarSystemContractError(
            "effective ET canonical remainder is not exactly binary64-representable"
        )
    return whole, remainder_float


def _build_effective_query(
    requested_query: EphemerisQuerySpec,
    effective_query_id: str,
    effective_et: Fraction,
) -> EphemerisQuerySpec:
    whole, fraction = _canonical_epoch_parts(effective_et)
    requested_epoch = requested_query.epoch
    effective_epoch = CoordinateEpoch(
        time_scale=requested_epoch.time_scale,
        representation=requested_epoch.representation,
        whole=whole,
        fraction=fraction,
        coordinate_unit_id=requested_epoch.coordinate_unit_id,
        origin_id=requested_epoch.origin_id,
        realization_id=requested_epoch.realization_id,
        artifact_ids=requested_epoch.artifact_ids,
    )
    return EphemerisQuerySpec(
        query_id=effective_query_id,
        provider=requested_query.provider,
        epoch=effective_epoch,
        target_naif_ids=requested_query.target_naif_ids,
        observer_naif_id=requested_query.observer_naif_id,
        frame=requested_query.frame,
        aberration_correction=requested_query.aberration_correction,
        output_unit_system=requested_query.output_unit_system,
        state_kind=requested_query.state_kind,
        target_chain_availability_status=(
            requested_query.target_chain_availability_status
        ),
    )


def _expanded_binary64_projection_payload(
    value: Binary64UnitConversionReceipt,
) -> tuple[object, ...]:
    return (
        _BINARY64_RECEIPT_QUALIFIED_NAME,
        tuple(
            (descriptor.name, getattr(value, descriptor.name))
            for descriptor in fields(value)
            if descriptor.name != "content_sha256"
        ),
        ("content_sha256", value.content_sha256),
    )


def _projection_payload(
    value: CspiceBinary64QueryProjectionReceipt,
) -> tuple[object, ...]:
    return (
        _RECEIPT_QUALIFIED_NAME,
        tuple(
            (
                descriptor.name,
                _expanded_binary64_projection_payload(value.binary64_projection)
                if descriptor.name == "binary64_projection"
                else getattr(value, descriptor.name),
            )
            for descriptor in fields(value)
            if descriptor.name != "content_sha256"
        ),
    )


def _projection_digest(value: CspiceBinary64QueryProjectionReceipt) -> str:
    return domain_sha256(
        _RECEIPT_DOMAIN,
        _RECEIPT_SCHEMA,
        _projection_payload(value),
    )


@dataclass(frozen=True, slots=True, eq=False)
class CspiceBinary64QueryProjectionReceipt:
    """One exact requested-to-effective CSPICE ET projection."""

    requested_query: EphemerisQuerySpec
    effective_query: EphemerisQuerySpec
    binary64_projection: Binary64UnitConversionReceipt
    projection_stage: str
    exact_error_numerator: int
    exact_error_denominator: int
    exact_error_unit_id: str
    time_scale_operation_status: str
    provider_execution_status: str
    evidence_preservation_status: str
    artifact_custody_status: str
    content_integrity_class: str
    content_sha256: str = ""

    _DOMAIN: ClassVar[str] = _RECEIPT_DOMAIN
    _SCHEMA: ClassVar[str] = _RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if type(self) is not CspiceBinary64QueryProjectionReceipt:
            raise SolarSystemContractError(
                "CSPICE query projection receipt must have its exact concrete type"
            )
        _preflight_optional_sha256(self.content_sha256, "content_sha256")
        self._validate()
        expected = _projection_digest(self)
        if self.content_sha256 == "":
            object.__setattr__(self, "content_sha256", expected)
        elif self.content_sha256 != expected:
            raise SolarSystemContractError(
                "content_sha256 does not bind the exact CSPICE query projection receipt"
            )

    def _validate(self) -> None:
        if type(self) is not CspiceBinary64QueryProjectionReceipt:
            raise SolarSystemContractError(
                "CSPICE query projection receipt must have its exact concrete type"
            )
        if type(self.requested_query) is not EphemerisQuerySpec:
            raise SolarSystemContractError(
                "requested_query must be an exact EphemerisQuerySpec"
            )
        validate_integrity(self.requested_query)
        if self.requested_query.provider.identity.provider_kind != "LOCAL_OFFLINE_SPK":
            raise SolarSystemContractError(
                "requested_query must use the exact LOCAL_OFFLINE_SPK contract"
            )
        if type(self.effective_query) is not EphemerisQuerySpec:
            raise SolarSystemContractError(
                "effective_query must be an exact EphemerisQuerySpec"
            )
        validate_integrity(self.effective_query)
        effective_query_id = _bounded_ascii_identifier(
            self.effective_query.query_id,
            "effective_query.query_id",
        )
        if effective_query_id == self.requested_query.query_id:
            raise SolarSystemContractError(
                "effective query ID must be distinct from the requested query ID"
            )

        if type(self.binary64_projection) is not Binary64UnitConversionReceipt:
            raise SolarSystemContractError(
                "binary64_projection must be an exact Binary64UnitConversionReceipt"
            )
        self.binary64_projection.validate_integrity()

        _exact_token(self.projection_stage, _PROJECTION_STAGE, "projection_stage")
        error_numerator, error_denominator = _require_reduced_pair(
            self.exact_error_numerator,
            self.exact_error_denominator,
            "exact projection error",
        )
        _exact_token(
            self.exact_error_unit_id,
            _ERROR_UNIT_ID,
            "exact_error_unit_id",
        )
        _exact_token(
            self.time_scale_operation_status,
            _TIME_SCALE_OPERATION_STATUS,
            "time_scale_operation_status",
        )
        _exact_token(
            self.provider_execution_status,
            _PROVIDER_EXECUTION_STATUS,
            "provider_execution_status",
        )
        _exact_token(
            self.evidence_preservation_status,
            _EVIDENCE_PRESERVATION_STATUS,
            "evidence_preservation_status",
        )
        _exact_token(
            self.artifact_custody_status,
            _ARTIFACT_CUSTODY_STATUS,
            "artifact_custody_status",
        )
        _exact_token(
            self.content_integrity_class,
            _CONTENT_INTEGRITY_CLASS,
            "content_integrity_class",
        )

        requested_et = _requested_et_fraction(self.requested_query)
        expected_projection = convert_fraction_to_binary64(
            requested_et,
            SECOND,
            SECOND,
        )
        if self.binary64_projection.content_sha256 != expected_projection.content_sha256:
            raise SolarSystemContractError(
                "binary64_projection is not the exact requested-query ET realization"
            )
        effective_et = Fraction.from_float(expected_projection.rounded_value)
        expected_error = _exact_projection_error(requested_et, effective_et)
        if (error_numerator, error_denominator) != (
            expected_error.numerator,
            expected_error.denominator,
        ):
            raise SolarSystemContractError(
                "exact projection error does not equal effective ET minus requested ET"
            )
        direction = expected_projection.rounding_direction
        if (
            (direction == "EXACT" and expected_error != 0)
            or (direction == "BELOW_EXACT" and expected_error >= 0)
            or (direction == "ABOVE_EXACT" and expected_error <= 0)
        ):
            raise SolarSystemContractError(
                "projection error sign is inconsistent with the rounding receipt"
            )

        expected_effective_query = _build_effective_query(
            self.requested_query,
            effective_query_id,
            effective_et,
        )
        if self.effective_query.content_sha256 != expected_effective_query.content_sha256:
            raise SolarSystemContractError(
                "effective_query is not the exact canonical rounded-ET query"
            )

    def validate_integrity(self) -> None:
        """Recompute nested seals, exact arithmetic, cross-bindings, and own seal."""

        if type(self) is not CspiceBinary64QueryProjectionReceipt:
            raise SolarSystemContractError(
                "CSPICE query projection receipt must have its exact concrete type"
            )
        validate_sha256(self.content_sha256, "content_sha256")
        self._validate()
        if self.content_sha256 != _projection_digest(self):
            raise SolarSystemContractError(
                "content_sha256 does not bind the exact CSPICE query projection receipt"
            )


def project_cspice_binary64_query(
    requested_query: EphemerisQuerySpec,
    *,
    effective_query_id: str,
) -> CspiceBinary64QueryProjectionReceipt:
    """Project one exact two-part SPICE ET query onto CSPICE binary64 ET."""

    if type(requested_query) is not EphemerisQuerySpec:
        raise SolarSystemContractError(
            "requested_query must be an exact EphemerisQuerySpec"
        )
    checked_effective_query_id = _bounded_ascii_identifier(
        effective_query_id,
        "effective_query_id",
    )
    validate_integrity(requested_query)
    if requested_query.provider.identity.provider_kind != "LOCAL_OFFLINE_SPK":
        raise SolarSystemContractError(
            "requested_query must use the exact LOCAL_OFFLINE_SPK contract"
        )
    if checked_effective_query_id == requested_query.query_id:
        raise SolarSystemContractError(
            "effective query ID must be distinct from the requested query ID"
        )

    requested_et = _requested_et_fraction(requested_query)
    binary64_projection = convert_fraction_to_binary64(
        requested_et,
        SECOND,
        SECOND,
    )
    effective_et = Fraction.from_float(binary64_projection.rounded_value)
    exact_error = _exact_projection_error(requested_et, effective_et)
    effective_query = _build_effective_query(
        requested_query,
        checked_effective_query_id,
        effective_et,
    )
    return CspiceBinary64QueryProjectionReceipt(
        requested_query=requested_query,
        effective_query=effective_query,
        binary64_projection=binary64_projection,
        projection_stage=_PROJECTION_STAGE,
        exact_error_numerator=exact_error.numerator,
        exact_error_denominator=exact_error.denominator,
        exact_error_unit_id=_ERROR_UNIT_ID,
        time_scale_operation_status=_TIME_SCALE_OPERATION_STATUS,
        provider_execution_status=_PROVIDER_EXECUTION_STATUS,
        evidence_preservation_status=_EVIDENCE_PRESERVATION_STATUS,
        artifact_custody_status=_ARTIFACT_CUSTODY_STATUS,
        content_integrity_class=_CONTENT_INTEGRITY_CLASS,
    )


def validate_cspice_binary64_query_projection_receipt(
    value: CspiceBinary64QueryProjectionReceipt,
) -> None:
    """Validate one exact projection receipt without provider execution."""

    if type(value) is not CspiceBinary64QueryProjectionReceipt:
        raise SolarSystemContractError(
            "value must be an exact CspiceBinary64QueryProjectionReceipt"
        )
    value.validate_integrity()


__all__ = [
    "CspiceBinary64QueryProjectionReceipt",
    "project_cspice_binary64_query",
    "validate_cspice_binary64_query_projection_receipt",
]
