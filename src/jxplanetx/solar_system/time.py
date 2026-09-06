"""Exact same-context arithmetic for canonical two-part coordinate epochs.

This unpublished milestone compares, differences, and shifts already-labeled
coordinate epochs.  It performs no time-scale conversion, calendar or UTC
interpretation, leap-second handling, TDB/TT model, TCB/TDB scenario scaling,
SPICE ET/JD bridge, provider call, frame transformation, or event location.

Epoch evidence identifiers are deliberately not coordinate identity.  Binary
operations neither merge nor authenticate evidence; exact shifts preserve the
source epoch's ordered evidence tuple unchanged.
"""

from __future__ import annotations

from fractions import Fraction

from .contracts import (
    CoordinateEpoch,
    SolarSystemContractError,
    validate_integrity,
)
from .serialization import MAXIMUM_CANONICAL_EXACT_INTEGER_BITS
from .units import _round_fraction_to_binary64


_MAXIMUM_SHIFT_COMPONENT_BITS = MAXIMUM_CANONICAL_EXACT_INTEGER_BITS
_MAXIMUM_PUBLIC_RESULT_COMPONENT_BITS = MAXIMUM_CANONICAL_EXACT_INTEGER_BITS
_MAXIMUM_TIME_INTERMEDIATE_BITS = 16_384


def _require_epoch(value: object, label: str) -> CoordinateEpoch:
    if type(value) is not CoordinateEpoch:
        raise SolarSystemContractError(f"{label} must be an exact CoordinateEpoch")
    validate_integrity(value)
    return value


def _require_fraction(value: object, label: str) -> Fraction:
    if type(value) is not Fraction:
        raise SolarSystemContractError(f"{label} must be an exact Fraction")
    if (
        value.numerator.bit_length() > _MAXIMUM_SHIFT_COMPONENT_BITS
        or value.denominator.bit_length() > _MAXIMUM_SHIFT_COMPONENT_BITS
    ):
        raise SolarSystemContractError(f"{label} exceeds the retained rational cap")
    return value


def _coordinate_context(value: CoordinateEpoch) -> tuple[str, str, str, str, str]:
    return (
        value.time_scale,
        value.representation,
        value.coordinate_unit_id,
        value.origin_id,
        value.realization_id,
    )


def _require_same_coordinate_context(
    left: CoordinateEpoch,
    right: CoordinateEpoch,
) -> None:
    if _coordinate_context(left) != _coordinate_context(right):
        raise SolarSystemContractError(
            "coordinate epochs do not share one exact represented-coordinate context"
        )


def _product_bit_bound(value: int, factor: int) -> int:
    if value == 0 or factor == 0:
        return 0
    return abs(value).bit_length() + abs(factor).bit_length()


def _add_fractions(left: Fraction, right: Fraction, label: str) -> Fraction:
    left_bound = _product_bit_bound(left.numerator, right.denominator)
    right_bound = _product_bit_bound(right.numerator, left.denominator)
    denominator_bound = _product_bit_bound(left.denominator, right.denominator)
    if max(left_bound, right_bound) + 1 > _MAXIMUM_TIME_INTERMEDIATE_BITS:
        raise SolarSystemContractError(f"{label} numerator sum exceeds its prospective cap")
    if denominator_bound > _MAXIMUM_TIME_INTERMEDIATE_BITS:
        raise SolarSystemContractError(f"{label} denominator exceeds its prospective cap")
    result = Fraction(
        left.numerator * right.denominator
        + right.numerator * left.denominator,
        left.denominator * right.denominator,
    )
    if (
        result.numerator.bit_length() > _MAXIMUM_TIME_INTERMEDIATE_BITS
        or result.denominator.bit_length() > _MAXIMUM_TIME_INTERMEDIATE_BITS
    ):
        raise SolarSystemContractError(f"{label} exceeds the intermediate result cap")
    return result


def _compare_fractions(left: Fraction, right: Fraction, label: str) -> int:
    left_bound = _product_bit_bound(left.numerator, right.denominator)
    right_bound = _product_bit_bound(right.numerator, left.denominator)
    if max(left_bound, right_bound) > _MAXIMUM_TIME_INTERMEDIATE_BITS:
        raise SolarSystemContractError(f"{label} exceeds its prospective comparison cap")
    left_cross = left.numerator * right.denominator
    right_cross = right.numerator * left.denominator
    if left_cross < right_cross:
        return -1
    if left_cross > right_cross:
        return 1
    return 0


def _epoch_value(value: CoordinateEpoch) -> Fraction:
    return _add_fractions(
        Fraction(value.whole, 1),
        Fraction.from_float(value.fraction),
        "coordinate epoch value",
    )


def _require_public_result(value: Fraction, label: str) -> Fraction:
    if (
        value.numerator.bit_length() > _MAXIMUM_PUBLIC_RESULT_COMPONENT_BITS
        or value.denominator.bit_length() > _MAXIMUM_PUBLIC_RESULT_COMPONENT_BITS
    ):
        raise SolarSystemContractError(f"{label} exceeds the public rational cap")
    return value


def compare_coordinate_epochs(left: CoordinateEpoch, right: CoordinateEpoch) -> int:
    """Compare two epochs in one exact represented-coordinate context.

    Returns exactly ``-1``, ``0``, or ``1``.  Differing evidence identifiers
    do not affect coordinate equality and are not merged or authenticated.
    """

    checked_left = _require_epoch(left, "left")
    checked_right = _require_epoch(right, "right")
    _require_same_coordinate_context(checked_left, checked_right)
    left_value = _epoch_value(checked_left)
    right_value = _epoch_value(checked_right)
    return _compare_fractions(left_value, right_value, "coordinate epoch comparison")


def coordinate_epoch_difference(
    left: CoordinateEpoch,
    right: CoordinateEpoch,
) -> Fraction:
    """Return ``left - right`` exactly in the retained coordinate unit."""

    checked_left = _require_epoch(left, "left")
    checked_right = _require_epoch(right, "right")
    _require_same_coordinate_context(checked_left, checked_right)
    result = _add_fractions(
        _epoch_value(checked_left),
        -_epoch_value(checked_right),
        "coordinate epoch difference",
    )
    return _require_public_result(result, "coordinate epoch difference")


def _canonical_parts(value: Fraction) -> tuple[int, Fraction]:
    numerator = value.numerator
    denominator = value.denominator
    doubled_numerator_bound = numerator.bit_length() + 1
    doubled_denominator_bound = denominator.bit_length() + 1
    if max(doubled_numerator_bound, denominator.bit_length()) + 1 > (
        _MAXIMUM_TIME_INTERMEDIATE_BITS
    ):
        raise SolarSystemContractError("epoch canonicalization numerator sum exceeds its cap")
    if doubled_denominator_bound > _MAXIMUM_TIME_INTERMEDIATE_BITS:
        raise SolarSystemContractError("epoch canonicalization denominator exceeds its cap")
    whole = ((numerator << 1) + denominator) // (denominator << 1)
    product_bound = _product_bit_bound(whole, denominator)
    if max(numerator.bit_length(), product_bound) + 1 > _MAXIMUM_TIME_INTERMEDIATE_BITS:
        raise SolarSystemContractError("epoch canonicalization remainder exceeds its cap")
    remainder = Fraction(numerator - whole * denominator, denominator)
    if remainder < Fraction(-1, 2) or remainder >= Fraction(1, 2):
        raise SolarSystemContractError("epoch canonicalization left its half-open range")
    if whole.bit_length() > MAXIMUM_CANONICAL_EXACT_INTEGER_BITS:
        raise SolarSystemContractError("shifted epoch whole exceeds the retained integer cap")
    return whole, remainder


def _require_exact_binary64_remainder(value: Fraction) -> float:
    rounded = _round_fraction_to_binary64(value)
    if rounded.rounding_status != "EXACT":
        raise SolarSystemContractError(
            "shifted epoch fraction is not exactly representable as binary64"
        )
    if value == 0:
        return 0.0
    return rounded.rounded_value


def shift_coordinate_epoch(
    epoch: CoordinateEpoch,
    offset_in_coordinate_units: Fraction,
) -> CoordinateEpoch:
    """Shift an epoch exactly by a rational in its own coordinate unit.

    The result is produced only when its canonical half-open remainder is
    exactly binary64-representable.  The source evidence tuple is preserved;
    arithmetic neither adds evidence nor asserts its authenticity.
    """

    checked_epoch = _require_epoch(epoch, "epoch")
    checked_offset = _require_fraction(
        offset_in_coordinate_units,
        "offset_in_coordinate_units",
    )
    shifted = _add_fractions(
        _epoch_value(checked_epoch),
        checked_offset,
        "coordinate epoch shift",
    )
    whole, remainder = _canonical_parts(shifted)
    fraction = _require_exact_binary64_remainder(remainder)
    return CoordinateEpoch(
        time_scale=checked_epoch.time_scale,
        representation=checked_epoch.representation,
        whole=whole,
        fraction=fraction,
        coordinate_unit_id=checked_epoch.coordinate_unit_id,
        origin_id=checked_epoch.origin_id,
        realization_id=checked_epoch.realization_id,
        artifact_ids=checked_epoch.artifact_ids,
    )


__all__ = [
    "compare_coordinate_epochs",
    "coordinate_epoch_difference",
    "shift_coordinate_epoch",
]
