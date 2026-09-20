"""Defined-exact scalar unit arithmetic for Solar-System preparation.

This unpublished milestone performs dimension-preserving arithmetic on exact
rational unit definitions.  It does not convert coordinate time scales,
propagate measurement uncertainty, transform arrays or states, or establish
physical accuracy.  The astronomical unit is a defined length and the day is
exactly 86400 SI seconds; neither definition is an orbital or UTC-day claim.

Binary64 conversion is an explicit nearest-ties-to-even realization of an
exact rational result.  Its receipt is an unauthenticated content-integrity
record, not evidence of scientific or registry authority.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, fields
from fractions import Fraction
from math import gcd
from typing import ClassVar

from .contracts import (
    ExactUnitScale,
    SolarSystemContractError,
    validate_integrity,
)
from .serialization import (
    MAXIMUM_CANONICAL_EXACT_INTEGER_BITS,
    MAXIMUM_CANONICAL_STRING_CODEPOINTS,
    domain_sha256,
    validate_sha256,
)


_MAXIMUM_RETAINED_COMPONENT_BITS = MAXIMUM_CANONICAL_EXACT_INTEGER_BITS
_MAXIMUM_INTERMEDIATE_BITS = 16_384

_DEFINITION_SCOPE = "DEFINED_ONLY"
_ROUNDING_MODE = "NEAREST_TIES_TO_EVEN"
_ROUNDING_STATUSES = frozenset(("EXACT", "ROUNDED"))
_ROUNDING_DIRECTIONS = frozenset(("ABOVE_EXACT", "BELOW_EXACT", "EXACT"))
_RESULT_CLASSES = frozenset(("NORMAL", "SUBNORMAL", "ZERO"))

_RECEIPT_DOMAIN = (
    "jxplanetx.solar-system.units.binary64-unit-conversion-receipt.v1"
)
_RECEIPT_SCHEMA = "Binary64UnitConversionReceipt.v1"
_RECEIPT_QUALIFIED_NAME = (
    "jxplanetx.solar_system.units.Binary64UnitConversionReceipt"
)

_BINARY64_SIGNIFICAND_BITS = 53
_BINARY64_FRACTION_BITS = 52
_BINARY64_MIN_NORMAL_EXPONENT = -1022
_BINARY64_MAX_NORMAL_EXPONENT = 1023
_BINARY64_SUBNORMAL_EXPONENT = -1074
_BINARY64_HALF_MIN_SUBNORMAL = Fraction(1, 1 << 1075)
# H is the exact midpoint between the largest finite binary64 value and the
# next power of two.  A tie rounds to the even power-of-two significand, whose
# binary64 encoding is infinity; H and all larger magnitudes are rejected.
_BINARY64_OVERFLOW_THRESHOLD_H = Fraction(
    (1 << 1024) - (1 << 970),
    1,
)


def _unit(
    unit_id: str,
    dimension: str,
    si_unit_id: str,
    numerator: int,
) -> ExactUnitScale:
    return ExactUnitScale(
        unit_id=unit_id,
        dimension=dimension,
        si_unit_id=si_unit_id,
        numerator=numerator,
        denominator=1,
        definition_classification="DEFINED_EXACT",
        artifact_ids=(),
    )


METRE = _unit(
    "unit.metre.defined-exact.v1",
    "LENGTH",
    "si.metre",
    1,
)
SECOND = _unit(
    "unit.second.defined-exact.v1",
    "TIME",
    "si.second",
    1,
)
KILOGRAM = _unit(
    "unit.kilogram.defined-exact.v1",
    "MASS",
    "si.kilogram",
    1,
)
KILOMETRE = _unit(
    "unit.kilometre.1000-si-metres.defined-exact.v1",
    "LENGTH",
    "si.metre",
    1_000,
)
DAY = _unit(
    "unit.day.86400-si-seconds.defined-exact.v1",
    "TIME",
    "si.second",
    86_400,
)
ASTRONOMICAL_UNIT = _unit(
    "unit.astronomical-unit.149597870700-si-metres.defined-exact.v1",
    "LENGTH",
    "si.metre",
    149_597_870_700,
)


def _require_exact_integer(value: object, label: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise SolarSystemContractError(f"{label} must be an exact integer")
    if value.bit_length() > _MAXIMUM_RETAINED_COMPONENT_BITS:
        raise SolarSystemContractError(f"{label} exceeds the retained integer cap")
    if positive and value <= 0:
        raise SolarSystemContractError(f"{label} must be strictly positive")
    return value


def _require_reduced_pair(
    numerator: object,
    denominator: object,
    label: str,
) -> tuple[int, int]:
    checked_numerator = _require_exact_integer(numerator, f"{label}.numerator")
    checked_denominator = _require_exact_integer(
        denominator,
        f"{label}.denominator",
        positive=True,
    )
    if gcd(abs(checked_numerator), checked_denominator) != 1:
        raise SolarSystemContractError(f"{label} must be a reduced rational pair")
    if checked_numerator == 0 and checked_denominator != 1:
        raise SolarSystemContractError(f"{label} zero must be represented as 0/1")
    return checked_numerator, checked_denominator


def _require_fraction(value: object, label: str) -> Fraction:
    if type(value) is not Fraction:
        raise SolarSystemContractError(f"{label} must be an exact Fraction")
    _require_reduced_pair(value.numerator, value.denominator, label)
    return value


def _require_unit(value: object, label: str) -> ExactUnitScale:
    if type(value) is not ExactUnitScale:
        raise SolarSystemContractError(f"{label} must be an exact ExactUnitScale")
    validate_integrity(value)
    if value.definition_classification != "DEFINED_EXACT":
        raise SolarSystemContractError(
            f"{label} must have DEFINED_EXACT classification in this milestone"
        )
    return value


def _require_conversion_pair(
    source_unit: object,
    target_unit: object,
) -> tuple[ExactUnitScale, ExactUnitScale]:
    source = _require_unit(source_unit, "source_unit")
    target = _require_unit(target_unit, "target_unit")
    if source.dimension != target.dimension or source.si_unit_id != target.si_unit_id:
        raise SolarSystemContractError(
            "source and target units must share one exact dimension and canonical SI unit"
        )
    if source.unit_id == target.unit_id and source.content_sha256 != target.content_sha256:
        raise SolarSystemContractError(
            "one unit_id cannot name unequal exact unit definitions"
        )
    return source, target


def _prospective_product(value: int, factor: int, label: str) -> None:
    if type(value) is not int or type(factor) is not int:
        raise SolarSystemContractError(f"{label} operands must be exact integers")
    if value == 0 or factor == 0:
        return
    if abs(value).bit_length() + abs(factor).bit_length() > _MAXIMUM_INTERMEDIATE_BITS:
        raise SolarSystemContractError(f"{label} exceeds the intermediate arithmetic cap")


def _require_bounded_result(value: Fraction, label: str) -> Fraction:
    if (
        value.numerator.bit_length() > _MAXIMUM_RETAINED_COMPONENT_BITS
        or value.denominator.bit_length() > _MAXIMUM_RETAINED_COMPONENT_BITS
    ):
        raise SolarSystemContractError(f"{label} exceeds the retained rational cap")
    return value


def unit_ratio(source_unit: ExactUnitScale, target_unit: ExactUnitScale) -> Fraction:
    """Return the exact source-unit to target-unit multiplier.

    The result is ``source SI scale / target SI scale``.  Only defined-exact
    units of one dimension are accepted; nominal physical definitions are
    deliberately outside this milestone.
    """

    source, target = _require_conversion_pair(source_unit, target_unit)
    _prospective_product(source.numerator, target.denominator, "unit ratio numerator")
    _prospective_product(source.denominator, target.numerator, "unit ratio denominator")
    result = Fraction(
        source.numerator * target.denominator,
        source.denominator * target.numerator,
    )
    return _require_bounded_result(result, "unit ratio")


def _exact_conversion(
    value: Fraction,
    source_unit: ExactUnitScale,
    target_unit: ExactUnitScale,
) -> Fraction:
    ratio = unit_ratio(source_unit, target_unit)
    _prospective_product(value.numerator, ratio.numerator, "conversion numerator")
    _prospective_product(value.denominator, ratio.denominator, "conversion denominator")
    result = Fraction(
        value.numerator * ratio.numerator,
        value.denominator * ratio.denominator,
    )
    return _require_bounded_result(result, "exact conversion result")


def convert_fraction(
    value: Fraction,
    source_unit: ExactUnitScale,
    target_unit: ExactUnitScale,
) -> Fraction:
    """Convert one exact rational scalar between defined units."""

    exact_value = _require_fraction(value, "value")
    source, target = _require_conversion_pair(source_unit, target_unit)
    return _exact_conversion(exact_value, source, target)


@dataclass(frozen=True, slots=True)
class _Binary64Rounding:
    rounded_value: float
    rounding_status: str
    rounding_direction: str
    result_class: str


def _round_quotient_ties_to_even(
    quotient: int,
    remainder: int,
    denominator: int,
) -> int:
    if remainder.bit_length() + 1 > _MAXIMUM_INTERMEDIATE_BITS:
        raise SolarSystemContractError("rounding remainder exceeds its doubling cap")
    doubled = remainder << 1
    if doubled > denominator or (doubled == denominator and quotient & 1):
        return quotient + 1
    return quotient


def _floor_log2_ratio(numerator: int, denominator: int) -> int:
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if denominator.bit_length() + exponent > _MAXIMUM_INTERMEDIATE_BITS:
            raise SolarSystemContractError("binary64 exponent comparison exceeds its cap")
        if numerator < denominator << exponent:
            exponent -= 1
    else:
        shift = -exponent
        if numerator.bit_length() + shift > _MAXIMUM_INTERMEDIATE_BITS:
            raise SolarSystemContractError("binary64 exponent comparison exceeds its cap")
        if numerator << shift < denominator:
            exponent -= 1
    return exponent


def _pack_binary64(sign: int, exponent_bits: int, fraction_bits: int) -> float:
    bits = (sign << 63) | (exponent_bits << 52) | fraction_bits
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def _round_fraction_to_binary64(value: Fraction) -> _Binary64Rounding:
    """Return the exact RN-ties-even realization used by units and time.

    This private helper accepts only a bounded exact ``Fraction``.  It never
    creates a unit-conversion receipt and therefore can be reused by exact
    epoch-shift representability checks without inventing unit provenance.
    """

    exact_value = _require_fraction(value, "binary64 exact value")
    if exact_value.numerator == 0:
        return _Binary64Rounding(0.0, "EXACT", "EXACT", "ZERO")

    sign = 1 if exact_value.numerator < 0 else 0
    magnitude = -exact_value if sign else exact_value
    if magnitude >= _BINARY64_OVERFLOW_THRESHOLD_H:
        raise SolarSystemContractError(
            "exact value rounds to binary64 infinity at or beyond overflow threshold H"
        )
    if magnitude <= _BINARY64_HALF_MIN_SUBNORMAL:
        raise SolarSystemContractError(
            "nonzero exact value rounds to binary64 zero"
        )

    numerator = magnitude.numerator
    denominator = magnitude.denominator
    exponent = _floor_log2_ratio(numerator, denominator)
    result_class: str

    if exponent < _BINARY64_MIN_NORMAL_EXPONENT:
        if numerator.bit_length() + 1074 > _MAXIMUM_INTERMEDIATE_BITS:
            raise SolarSystemContractError("subnormal rounding exceeds its arithmetic cap")
        quotient, remainder = divmod(numerator << 1074, denominator)
        significand = _round_quotient_ties_to_even(
            quotient,
            remainder,
            denominator,
        )
        if significand == 0:
            raise SolarSystemContractError(
                "nonzero exact value rounds to binary64 zero"
            )
        if significand > (1 << _BINARY64_FRACTION_BITS):
            raise SolarSystemContractError("subnormal rounding produced an invalid carry")
        if significand == (1 << _BINARY64_FRACTION_BITS):
            rounded_value = _pack_binary64(sign, 1, 0)
            result_class = "NORMAL"
        else:
            rounded_value = _pack_binary64(sign, 0, significand)
            result_class = "SUBNORMAL"
    else:
        shift = _BINARY64_FRACTION_BITS - exponent
        if shift >= 0:
            if numerator.bit_length() + shift > _MAXIMUM_INTERMEDIATE_BITS:
                raise SolarSystemContractError("normal rounding exceeds its arithmetic cap")
            quotient, remainder = divmod(numerator << shift, denominator)
            rounding_denominator = denominator
        else:
            denominator_shift = -shift
            if denominator.bit_length() + denominator_shift > _MAXIMUM_INTERMEDIATE_BITS:
                raise SolarSystemContractError("normal rounding exceeds its arithmetic cap")
            rounding_denominator = denominator << denominator_shift
            quotient, remainder = divmod(numerator, rounding_denominator)
        significand = _round_quotient_ties_to_even(
            quotient,
            remainder,
            rounding_denominator,
        )
        if significand == (1 << _BINARY64_SIGNIFICAND_BITS):
            significand = 1 << _BINARY64_FRACTION_BITS
            exponent += 1
        if exponent > _BINARY64_MAX_NORMAL_EXPONENT:
            raise SolarSystemContractError("exact value rounds to binary64 infinity")
        if not (
            (1 << _BINARY64_FRACTION_BITS)
            <= significand
            < (1 << _BINARY64_SIGNIFICAND_BITS)
        ):
            raise SolarSystemContractError("normal rounding produced an invalid significand")
        rounded_value = _pack_binary64(
            sign,
            exponent + 1023,
            significand - (1 << _BINARY64_FRACTION_BITS),
        )
        result_class = "NORMAL"

    if not math.isfinite(rounded_value) or rounded_value == 0.0:
        raise SolarSystemContractError("binary64 rounding did not produce a finite nonzero value")
    rounded_exact = Fraction.from_float(rounded_value)
    if rounded_exact == exact_value:
        status = "EXACT"
        direction = "EXACT"
    elif rounded_exact < exact_value:
        status = "ROUNDED"
        direction = "BELOW_EXACT"
    else:
        status = "ROUNDED"
        direction = "ABOVE_EXACT"
    return _Binary64Rounding(rounded_value, status, direction, result_class)


def _receipt_payload(value: Binary64UnitConversionReceipt) -> tuple[object, ...]:
    return (
        _RECEIPT_QUALIFIED_NAME,
        tuple(
            (descriptor.name, getattr(value, descriptor.name))
            for descriptor in fields(value)
            if descriptor.name != "content_sha256"
        ),
    )


def _receipt_digest(value: Binary64UnitConversionReceipt) -> str:
    return domain_sha256(
        _RECEIPT_DOMAIN,
        _RECEIPT_SCHEMA,
        _receipt_payload(value),
    )


def _require_token(value: object, allowed: frozenset[str], label: str) -> str:
    if type(value) is not str:
        raise SolarSystemContractError(f"{label} must be an exact string")
    if len(value) > MAXIMUM_CANONICAL_STRING_CODEPOINTS:
        raise SolarSystemContractError(f"{label} exceeds the text cap")
    if value not in allowed:
        raise SolarSystemContractError(f"{label} is not an exact supported token")
    return value


@dataclass(frozen=True, slots=True, eq=False)
class Binary64UnitConversionReceipt:
    """A sealed, exactly recomputable defined-unit binary64 conversion."""

    source_unit: ExactUnitScale
    target_unit: ExactUnitScale
    input_numerator: int
    input_denominator: int
    exact_numerator: int
    exact_denominator: int
    rounded_value: float
    definition_scope: str
    rounding_mode: str
    rounding_status: str
    rounding_direction: str
    result_class: str
    content_sha256: str = ""

    _DOMAIN: ClassVar[str] = _RECEIPT_DOMAIN
    _SCHEMA: ClassVar[str] = _RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        self._validate()
        expected = _receipt_digest(self)
        if type(self.content_sha256) is not str:
            raise SolarSystemContractError("content_sha256 must be an exact string")
        if self.content_sha256 == "":
            object.__setattr__(self, "content_sha256", expected)
            return
        validate_sha256(self.content_sha256, "content_sha256")
        if self.content_sha256 != expected:
            raise SolarSystemContractError(
                "content_sha256 does not bind the exact conversion receipt"
            )

    def _validate(self) -> None:
        if type(self) is not Binary64UnitConversionReceipt:
            raise SolarSystemContractError(
                "conversion receipt must have its exact concrete type"
            )
        source, target = _require_conversion_pair(self.source_unit, self.target_unit)
        input_numerator, input_denominator = _require_reduced_pair(
            self.input_numerator,
            self.input_denominator,
            "input value",
        )
        exact_numerator, exact_denominator = _require_reduced_pair(
            self.exact_numerator,
            self.exact_denominator,
            "exact converted value",
        )
        if type(self.rounded_value) is not float or not math.isfinite(self.rounded_value):
            raise SolarSystemContractError("rounded_value must be an exact finite float")
        _require_token(
            self.definition_scope,
            frozenset((_DEFINITION_SCOPE,)),
            "definition_scope",
        )
        _require_token(
            self.rounding_mode,
            frozenset((_ROUNDING_MODE,)),
            "rounding_mode",
        )
        _require_token(self.rounding_status, _ROUNDING_STATUSES, "rounding_status")
        _require_token(
            self.rounding_direction,
            _ROUNDING_DIRECTIONS,
            "rounding_direction",
        )
        _require_token(self.result_class, _RESULT_CLASSES, "result_class")

        exact_input = Fraction(input_numerator, input_denominator)
        expected_exact = _exact_conversion(exact_input, source, target)
        if (
            exact_numerator != expected_exact.numerator
            or exact_denominator != expected_exact.denominator
        ):
            raise SolarSystemContractError(
                "receipt exact value does not match the defined unit conversion"
            )
        expected_rounding = _round_fraction_to_binary64(expected_exact)
        if self.rounded_value.hex() != expected_rounding.rounded_value.hex():
            raise SolarSystemContractError(
                "receipt rounded_value does not match exact binary64 rounding"
            )
        if (
            self.rounding_status != expected_rounding.rounding_status
            or self.rounding_direction != expected_rounding.rounding_direction
            or self.result_class != expected_rounding.result_class
        ):
            raise SolarSystemContractError(
                "receipt rounding classifications do not match exact recomputation"
            )

    def validate_integrity(self) -> None:
        """Revalidate exact fields, nested seals, arithmetic, and own seal."""

        self._validate()
        validate_sha256(self.content_sha256, "content_sha256")
        if self.content_sha256 != _receipt_digest(self):
            raise SolarSystemContractError(
                "content_sha256 does not bind the exact conversion receipt"
            )


def convert_fraction_to_binary64(
    value: Fraction,
    source_unit: ExactUnitScale,
    target_unit: ExactUnitScale,
) -> Binary64UnitConversionReceipt:
    """Convert an exact rational scalar and retain its binary64 realization."""

    exact_input = _require_fraction(value, "value")
    source, target = _require_conversion_pair(source_unit, target_unit)
    exact_result = _exact_conversion(exact_input, source, target)
    rounding = _round_fraction_to_binary64(exact_result)
    return Binary64UnitConversionReceipt(
        source_unit=source,
        target_unit=target,
        input_numerator=exact_input.numerator,
        input_denominator=exact_input.denominator,
        exact_numerator=exact_result.numerator,
        exact_denominator=exact_result.denominator,
        rounded_value=rounding.rounded_value,
        definition_scope=_DEFINITION_SCOPE,
        rounding_mode=_ROUNDING_MODE,
        rounding_status=rounding.rounding_status,
        rounding_direction=rounding.rounding_direction,
        result_class=rounding.result_class,
    )


__all__ = [
    "ASTRONOMICAL_UNIT",
    "Binary64UnitConversionReceipt",
    "DAY",
    "KILOGRAM",
    "KILOMETRE",
    "METRE",
    "SECOND",
    "convert_fraction",
    "convert_fraction_to_binary64",
    "unit_ratio",
]
