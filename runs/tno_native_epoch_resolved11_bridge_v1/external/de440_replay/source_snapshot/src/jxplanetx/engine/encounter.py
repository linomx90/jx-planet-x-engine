"""Standalone adaptive RKF78 solver for one guarded Newtonian encounter segment."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass, fields, is_dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np

from .backends import resolve_backend
from .contracts import (
    BackendSpec,
    ForcePlan,
    NewtonianPointMass,
    ParameterMetadata,
    Provenance,
    StateSnapshot,
)
from .encounter_contracts import (
    ADAPTIVE_ENCOUNTER_SEGMENT_METHOD_ID,
    AdaptiveEncounterSegmentSpec,
    ENCOUNTER_ALLOWED_AXES,
    ENCOUNTER_ALLOWED_TIME_SCALES,
    ENCOUNTER_ACCEPTED_ORDER,
    ENCOUNTER_BACKEND_SCOPE,
    ENCOUNTER_CERTIFICATE_CLAIM_SCOPE,
    ENCOUNTER_CERTIFICATE_NONCLAIMS,
    ENCOUNTER_CHECKSUM_BINDING_POLICY,
    ENCOUNTER_CONTROLLER_EXPONENT,
    ENCOUNTER_DEFECT_ORIENTATION,
    ENCOUNTER_DOMAIN_REJECTION_SCALE_FACTOR,
    ENCOUNTER_EMBEDDED_ORDER,
    ENCOUNTER_EVIDENCE_CLASS,
    ENCOUNTER_EXACT_ABSTRACT_WORK_WEIGHT_TABLE,
    ENCOUNTER_EXACT_ABSTRACT_WORK_WEIGHT_TABLE_ID,
    ENCOUNTER_EXACT_WITNESS_CHECKSUM_ALGORITHM,
    ENCOUNTER_EXACT_WITNESS_CHECKSUM_DOMAIN,
    ENCOUNTER_FORCE_EVALUATION_ACCOUNTING,
    ENCOUNTER_FORCE_PLAN_SCOPE,
    ENCOUNTER_HARD_MAXIMUM_BODY_COUNT,
    ENCOUNTER_HARD_MAXIMUM_BODY_ID_UTF8_BYTES,
    ENCOUNTER_INITIALIZATION_CHECKSUM_ALGORITHM,
    ENCOUNTER_INITIALIZATION_CHECKSUM_DOMAIN,
    ENCOUNTER_PAIR_ERROR_NORM,
    ENCOUNTER_PAIR_TABLE_CHECKSUM_ALGORITHM,
    ENCOUNTER_PAIR_TABLE_CHECKSUM_DOMAIN,
    ENCOUNTER_PROPOSAL_ACCOUNTING,
    ENCOUNTER_PUBLIC_EXECUTION_ACCOUNTING_SCOPE,
    ENCOUNTER_RESULT_CONTENT_CHECKSUM_ALGORITHM,
    ENCOUNTER_RESULT_CONTENT_CHECKSUM_DOMAIN,
    ENCOUNTER_REQUIRED_FRAME,
    ENCOUNTER_REQUIRED_ORIGIN,
    ENCOUNTER_SCHEDULE_CHECKSUM_ALGORITHM,
    ENCOUNTER_SCHEDULE_CHECKSUM_DOMAIN,
    ENCOUNTER_STAGE_COUNT,
    ENCOUNTER_STAGE_EPOCH_POLICY,
    ENCOUNTER_TABLEAU_ID,
    ENCOUNTER_VALIDATION_REPLAY_COUNT,
    ENCOUNTER_VALIDATION_REPLAY_POLICY,
    EncounterExactRationalResourceSpec,
)
from .evaluator import (
    EvaluationError,
    ForceLedgerEntry,
    StateMetadataBinding,
    _validate_dependencies,
    _validate_model_sequence,
    _validate_parameter_contracts,
    _validate_state,
    evaluate_force_plan,
)
from .forces import ForceDomainError
from .rkf78 import _C, _RKF78StepResult, _rkf78_step


ENCOUNTER_SEGMENT_SCOPE = "ADAPTIVE_NEWTONIAN_ENCOUNTER_SEGMENT_INTEGRATION"
_RESULT_SCHEMA = "jxplanetx.encounter-segment-result.v1"
_PAIR_TABLE_SCHEMA = "jxplanetx.encounter-pair-table.v1"
_SCHEDULE_SCHEMA = "jxplanetx.encounter-segment-schedule.v1"
_INITIALIZATION_SCHEMA = "jxplanetx.encounter-initialization.v1"
_WITNESS_SCHEMA = "jxplanetx.encounter-exact-clearance-witness.v1"

_ABSTRACT_WORK_WEIGHT_BY_PATH = MappingProxyType(
    {
        (lane, path): weight
        for lane, path, weight in ENCOUNTER_EXACT_ABSTRACT_WORK_WEIGHT_TABLE
    }
)
if len(_ABSTRACT_WORK_WEIGHT_BY_PATH) != len(
    ENCOUNTER_EXACT_ABSTRACT_WORK_WEIGHT_TABLE
):
    raise RuntimeError(
        f"duplicate path in {ENCOUNTER_EXACT_ABSTRACT_WORK_WEIGHT_TABLE_ID}"
    )

_DISPOSITIONS = (
    "CERTIFICATE_REJECTION",
    "STAGE_GUARD_ABORT",
    "DERIVATIVE_DOMAIN_ABORT",
    "CANDIDATE_DOMAIN_REJECTION",
    "ERROR_REJECTION",
    "ACCEPTED",
)


class EncounterError(EvaluationError):
    """An encounter segment cannot be completed as requested."""


class EncounterContractError(EncounterError):
    """Encounter inputs or retained output violate the frozen contract."""


class EncounterDomainError(EncounterError):
    """The accepted local IVP or a trial left the finite guarded domain."""


class EncounterResourceError(EncounterError):
    """A prospective exact-work or custody ceiling was reached."""


class EncounterStepLimitError(EncounterError):
    """A proposal, rejection, force, or endpoint-progress cap was reached."""


class _StageGuardAbort(Exception):
    pass


class _DerivativeDomainAbort(Exception):
    pass


def _sha256_hex(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EncounterContractError(f"{label} must be lowercase SHA-256 hex")
    return value


def _readonly_copy(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    copied = np.array(value, dtype=dtype, copy=True, order="C", subok=False)
    copied.setflags(write=False)
    return copied


def _float_bits(value: float) -> bytes:
    return struct.pack(">d", value)


def _canonical(value: object) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise EncounterContractError("checksum content cannot contain nonfinite floats")
        return {"float_hex": value.hex()}
    if type(value) is tuple:
        return [_canonical(item) for item in value]
    if type(value) is list:
        return [_canonical(item) for item in value]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise EncounterContractError("checksum dictionaries require string keys")
        return {key: _canonical(item) for key, item in value.items()}
    if type(value) is np.ndarray:
        if value.dtype == np.dtype(np.float64):
            flat: list[object] = [float(item).hex() for item in value.ravel(order="C")]
        elif value.dtype == np.dtype(np.bool_):
            flat = [bool(item) for item in value.ravel(order="C")]
        else:
            raise EncounterContractError("checksum arrays must be float64 or bool")
        return {"dtype": str(value.dtype), "shape": list(value.shape), "values": flat}
    if is_dataclass(value) and type(value).__module__.startswith("jxplanetx."):
        return {
            "dataclass": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                descriptor.name: _canonical(getattr(value, descriptor.name))
                for descriptor in fields(value)
            },
        }
    raise EncounterContractError(
        f"unsupported checksum content type {type(value).__name__!r}"
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _canonical(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _domain_sha256(domain: str, value: object) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + _canonical_json(value)).hexdigest()


@dataclass
class _MutableUsage:
    exact_operations: int = 0
    rational_operations: int = 0
    dyadic_operations: int = 0
    gcd_iterations: int = 0
    transcript_bytes: int = 0
    maximum_integer_bits: int = 0
    maximum_rational_exponent_magnitude: int = 0


class _ExactBudget:
    """One initialization/proposal budget sharing one whole-lane ledger."""

    def __init__(
        self,
        *,
        resources: EncounterExactRationalResourceSpec,
        segment: _MutableUsage,
        operation_limit: int,
        gcd_limit: int,
        transcript_limit: int,
        label: str,
    ) -> None:
        self.resources = resources
        self.segment = segment
        self.local = _MutableUsage()
        self.operation_limit = operation_limit
        self.gcd_limit = gcd_limit
        self.transcript_limit = transcript_limit
        self.label = label

    @staticmethod
    def _path_weight(lane: str, path: str) -> int:
        try:
            return _ABSTRACT_WORK_WEIGHT_BY_PATH[(lane, path)]
        except KeyError as exc:
            raise EncounterContractError(
                f"unknown {ENCOUNTER_EXACT_ABSTRACT_WORK_WEIGHT_TABLE_ID} path "
                f"{lane}.{path}"
            ) from exc

    def rational_work(self, path: str) -> None:
        count = self._path_weight("GENERAL_RATIONAL", path)
        if (
            self.local.exact_operations + count > self.operation_limit
            or self.segment.exact_operations + count
            > self.resources.maximum_operations_per_segment
        ):
            raise EncounterResourceError(
                f"{self.label} abstract exact-work cap exhausted"
            )
        self.local.exact_operations += count
        self.segment.exact_operations += count
        self.local.rational_operations += count
        self.segment.rational_operations += count

    def dyadic_work(self, path: str) -> None:
        count = self._path_weight("DYADIC", path)
        if (
            self.local.exact_operations + count > self.operation_limit
            or self.segment.exact_operations + count
            > self.resources.maximum_operations_per_segment
        ):
            raise EncounterResourceError(
                f"{self.label} abstract exact-work cap exhausted"
            )
        self.local.exact_operations += count
        self.segment.exact_operations += count
        self.local.dyadic_operations += count
        self.segment.dyadic_operations += count

    def gcd_iteration(self) -> None:
        count = self._path_weight("GCD", "EUCLIDEAN_ITERATION")
        if (
            self.local.gcd_iterations + count > self.gcd_limit
            or self.segment.gcd_iterations + count
            > self.resources.maximum_gcd_iterations_per_segment
        ):
            raise EncounterResourceError(f"{self.label} GCD-iteration cap exhausted")
        self.local.gcd_iterations += count
        self.segment.gcd_iterations += count

    def transcript(self, count: int) -> None:
        if type(count) is not int or count <= 0:
            raise EncounterContractError("transcript byte charge must be positive")
        if (
            self.local.transcript_bytes + count > self.transcript_limit
            or self.segment.transcript_bytes + count
            > self.resources.maximum_witness_transcript_bytes_per_segment
        ):
            raise EncounterResourceError(f"{self.label} transcript-byte cap exhausted")
        self.local.transcript_bytes += count
        self.segment.transcript_bytes += count

    def require_transcript_capacity(self, count: int) -> None:
        if type(count) is not int or count <= 0:
            raise EncounterContractError("transcript capacity must be positive")
        if (
            self.local.transcript_bytes + count > self.transcript_limit
            or self.segment.transcript_bytes + count
            > self.resources.maximum_witness_transcript_bytes_per_segment
        ):
            raise EncounterResourceError(
                f"{self.label} prospective transcript-byte cap exhausted"
            )

    def integer_bits(self, value: int, *, dyadic: bool = False) -> int:
        if dyadic:
            self.dyadic_work("INTEGER_WIDTH_OBSERVATION")
        else:
            self.rational_work("INTEGER_WIDTH_OBSERVATION")
        bits = abs(value).bit_length()
        if bits > self.resources.maximum_integer_bits:
            raise EncounterResourceError(f"{self.label} integer bit-length cap exceeded")
        self.local.maximum_integer_bits = max(self.local.maximum_integer_bits, bits)
        self.segment.maximum_integer_bits = max(self.segment.maximum_integer_bits, bits)
        return bits

    def observe_known_integer_bits(self, bits: int) -> None:
        """Retain an exactly derived intermediate width without re-running bit_length."""

        if type(bits) is not int or bits < 0:
            raise EncounterContractError("known integer width must be a nonnegative int")
        if bits > self.resources.maximum_integer_bits:
            raise EncounterResourceError(f"{self.label} integer bit-length cap exceeded")
        self.local.maximum_integer_bits = max(self.local.maximum_integer_bits, bits)
        self.segment.maximum_integer_bits = max(self.segment.maximum_integer_bits, bits)

    def rational_exponent(self, numerator: int, denominator: int) -> int:
        if numerator == 0:
            exponent = 0
        else:
            nbits = self.integer_bits(numerator)
            dbits = self.integer_bits(denominator)
            absolute_numerator = _int_absolute(numerator, self)
            exponent = nbits - dbits
            if exponent >= 0:
                shifted = _int_shift_left(denominator, exponent, self)
                if _int_compare(absolute_numerator, shifted, self) < 0:
                    exponent -= 1
            else:
                shifted = _int_shift_left(absolute_numerator, -exponent, self)
                if _int_compare(shifted, denominator, self) < 0:
                    exponent -= 1
        self.rational_work("RATIONAL_EXPONENT_DIAGNOSTIC")
        magnitude = abs(exponent)
        if magnitude > self.resources.maximum_rational_exponent_magnitude:
            raise EncounterResourceError(f"{self.label} rational exponent cap exceeded")
        self.local.maximum_rational_exponent_magnitude = max(
            self.local.maximum_rational_exponent_magnitude, magnitude
        )
        self.segment.maximum_rational_exponent_magnitude = max(
            self.segment.maximum_rational_exponent_magnitude, magnitude
        )
        return exponent


def _int_compare(left: int, right: int, budget: _ExactBudget) -> int:
    budget.rational_work("INTEGER_COMPARE")
    return (left > right) - (left < right)


def _int_absolute(value: int, budget: _ExactBudget) -> int:
    budget.rational_work("INTEGER_ABSOLUTE")
    result = abs(value)
    budget.integer_bits(result)
    return result


def _int_negate(value: int, budget: _ExactBudget) -> int:
    budget.rational_work("INTEGER_NEGATE")
    budget.integer_bits(value)
    return -value


def _int_shift_left(value: int, shift: int, budget: _ExactBudget) -> int:
    if type(shift) is not int or shift < 0:
        raise EncounterContractError("integer shift must be nonnegative")
    bits = budget.integer_bits(value)
    if value != 0 and bits + shift > budget.resources.maximum_integer_bits:
        raise EncounterResourceError(f"{budget.label} prospective shift cap exceeded")
    budget.rational_work("INTEGER_SHIFT_LEFT")
    shifted = value << shift
    budget.integer_bits(shifted)
    return shifted


def _int_add(left: int, right: int, budget: _ExactBudget) -> int:
    left_bits = budget.integer_bits(left)
    right_bits = budget.integer_bits(right)
    if max(left_bits, right_bits) + 1 > budget.resources.maximum_integer_bits:
        raise EncounterResourceError(f"{budget.label} prospective addition cap exceeded")
    budget.rational_work("INTEGER_ADD")
    result = left + right
    budget.integer_bits(result)
    return result


def _int_subtract(left: int, right: int, budget: _ExactBudget) -> int:
    left_bits = budget.integer_bits(left)
    right_bits = budget.integer_bits(right)
    if max(left_bits, right_bits) + 1 > budget.resources.maximum_integer_bits:
        raise EncounterResourceError(f"{budget.label} prospective subtraction cap exceeded")
    budget.rational_work("INTEGER_SUBTRACT")
    result = left - right
    budget.integer_bits(result)
    return result


def _int_multiply(left: int, right: int, budget: _ExactBudget) -> int:
    left_bits = budget.integer_bits(left)
    right_bits = budget.integer_bits(right)
    if left != 0 and right != 0 and left_bits + right_bits > budget.resources.maximum_integer_bits:
        raise EncounterResourceError(f"{budget.label} prospective multiplication cap exceeded")
    budget.rational_work("INTEGER_MULTIPLY")
    result = left * right
    budget.integer_bits(result)
    return result


def _int_floor_divide(numerator: int, denominator: int, budget: _ExactBudget) -> int:
    if denominator == 0:
        raise EncounterDomainError("exact integer division by zero")
    budget.integer_bits(numerator)
    budget.integer_bits(denominator)
    budget.rational_work("INTEGER_FLOOR_DIVIDE")
    result = numerator // denominator
    budget.integer_bits(result)
    return result


def _euclidean_gcd(left: int, right: int, budget: _ExactBudget) -> int:
    a = _int_absolute(left, budget)
    b = _int_absolute(right, budget)
    iterations = 0
    while b:
        if iterations >= budget.resources.maximum_gcd_iterations_per_reduction:
            raise EncounterResourceError(f"{budget.label} per-reduction GCD cap exhausted")
        budget.gcd_iteration()
        # The frozen abstract table assigns this selected Euclidean loop path
        # one separately reported GCD unit and no general-rational work weight.
        remainder = a % b
        a, b = b, remainder
        iterations += 1
    return a


def _authorize_binary64_ratio(
    value: float,
    budget: _ExactBudget,
    *,
    dyadic: bool,
) -> tuple[int, int]:
    """Authorize a finite binary64 ratio before creating its Python integers."""

    if type(value) is not float or not math.isfinite(value):
        raise EncounterDomainError("exact conversion requires a finite built-in float")
    charge = budget.dyadic_work if dyadic else budget.rational_work
    charge("BINARY64_ENVELOPE_BASE")
    bits = struct.unpack(">Q", _float_bits(value))[0]
    negative = bool(bits >> 63)
    exponent_field = (bits >> 52) & 0x7FF
    fraction = bits & ((1 << 52) - 1)
    if exponent_field == 0:
        mantissa = fraction
        exponent = -1074
    else:
        mantissa = (1 << 52) | fraction
        exponent = int(exponent_field) - 1023 - 52
    if mantissa == 0:
        budget.observe_known_integer_bits(0)
        budget.observe_known_integer_bits(1)
        return 0, 0
    charge("BINARY64_ENVELOPE_NONZERO")
    lowbit = mantissa & -mantissa
    trailing = lowbit.bit_length() - 1
    mantissa >>= trailing
    exponent += trailing
    if negative:
        mantissa = -mantissa
    mantissa_bits = abs(mantissa).bit_length()
    if exponent >= 0:
        numerator_bits = mantissa_bits + exponent
        denominator_bits = 1
    else:
        numerator_bits = mantissa_bits
        denominator_bits = -exponent + 1
    if (
        numerator_bits > budget.resources.maximum_integer_bits
        or denominator_bits > budget.resources.maximum_integer_bits
    ):
        raise EncounterResourceError(
            f"{budget.label} binary64 ratio exceeds the integer-bit cap"
        )
    budget.observe_known_integer_bits(numerator_bits)
    budget.observe_known_integer_bits(denominator_bits)
    value_exponent = mantissa_bits - 1 + exponent
    if abs(value_exponent) > budget.resources.maximum_rational_exponent_magnitude:
        raise EncounterResourceError(
            f"{budget.label} binary64 ratio exceeds the exponent cap"
        )
    return mantissa, exponent


class _Dyad:
    __slots__ = ("mantissa", "exponent", "mantissa_bits")

    def __init__(self, mantissa: int, exponent: int, budget: _ExactBudget) -> None:
        if type(mantissa) is not int or type(exponent) is not int:
            raise EncounterContractError("dyadic components must be built-in integers")
        budget.dyadic_work("DYADIC_CONSTRUCTION")
        input_bits = budget.integer_bits(mantissa, dyadic=True)
        budget.dyadic_work("INPUT_EXPONENT_CHECK")
        if abs(exponent) > budget.resources.maximum_rational_exponent_magnitude:
            raise EncounterResourceError(f"{budget.label} dyadic exponent cap exceeded")
        if mantissa == 0:
            self.mantissa = 0
            self.exponent = 0
            self.mantissa_bits = 0
            return
        budget.dyadic_work("NONZERO_ODDNESS_BRANCH")
        if mantissa & 1:
            canonical_mantissa = mantissa
            canonical_exponent = exponent
            canonical_bits = input_bits
        else:
            budget.dyadic_work("EVEN_LOWBIT_PATH")
            absolute = abs(mantissa)
            lowbit = absolute & -absolute
            lowbit_bits = budget.integer_bits(lowbit, dyadic=True)
            budget.dyadic_work("EVEN_CANONICALIZE_PATH")
            trailing = lowbit_bits - 1
            canonical_mantissa = mantissa >> trailing
            canonical_exponent = exponent + trailing
            canonical_bits = input_bits - trailing
        budget.dyadic_work("CANONICAL_EXPONENT_DIAGNOSTIC")
        value_exponent = canonical_bits - 1 + canonical_exponent
        if (
            abs(canonical_exponent)
            > budget.resources.maximum_rational_exponent_magnitude
            or abs(value_exponent)
            > budget.resources.maximum_rational_exponent_magnitude
        ):
            raise EncounterResourceError(f"{budget.label} dyadic exponent cap exceeded")
        magnitude = max(abs(canonical_exponent), abs(value_exponent))
        budget.local.maximum_rational_exponent_magnitude = max(
            budget.local.maximum_rational_exponent_magnitude, magnitude
        )
        budget.segment.maximum_rational_exponent_magnitude = max(
            budget.segment.maximum_rational_exponent_magnitude, magnitude
        )
        self.mantissa = canonical_mantissa
        self.exponent = canonical_exponent
        self.mantissa_bits = canonical_bits

    @classmethod
    def from_float(cls, value: float, budget: _ExactBudget) -> _Dyad:
        expected_mantissa, expected_exponent = _authorize_binary64_ratio(
            value, budget, dyadic=True
        )
        budget.dyadic_work("FROM_BINARY64_RATIO_EXTRACTION")
        numerator, denominator = value.as_integer_ratio()
        budget.dyadic_work("FROM_BINARY64_POWER_OF_TWO_CHECK")
        if denominator <= 0 or denominator & (denominator - 1):
            raise EncounterContractError("binary64 denominator is not a power of two")
        exponent = -(denominator.bit_length() - 1)
        result = cls(numerator, exponent, budget)
        if (
            result.mantissa != expected_mantissa
            or result.exponent != expected_exponent
        ):
            raise EncounterContractError("binary64 dyadic envelope disagrees with ratio")
        return result

    @classmethod
    def from_int(cls, value: int, budget: _ExactBudget) -> _Dyad:
        if type(value) is not int:
            raise EncounterContractError("dyadic integer conversion requires built-in int")
        return cls(value, 0, budget)

    def add(self, other: _Dyad, budget: _ExactBudget) -> _Dyad:
        budget.dyadic_work("ADD_OR_SUBTRACT_ALIGN")
        exponent = min(self.exponent, other.exponent)
        left_shift = self.exponent - exponent
        right_shift = other.exponent - exponent
        if (
            self.mantissa != 0
            and self.mantissa_bits + left_shift
            > budget.resources.maximum_integer_bits
        ) or (
            other.mantissa != 0
            and other.mantissa_bits + right_shift
            > budget.resources.maximum_integer_bits
        ):
            raise EncounterResourceError(f"{budget.label} prospective dyadic shift cap exceeded")
        budget.dyadic_work("ADD_OR_SUBTRACT_SHIFT_PAIR")
        left = self.mantissa << left_shift
        right = other.mantissa << right_shift
        left_bits = 0 if self.mantissa == 0 else self.mantissa_bits + left_shift
        right_bits = 0 if other.mantissa == 0 else other.mantissa_bits + right_shift
        budget.observe_known_integer_bits(left_bits)
        budget.observe_known_integer_bits(right_bits)
        carry_bits = 1 if self.mantissa != 0 and other.mantissa != 0 else 0
        if max(left_bits, right_bits) + carry_bits > budget.resources.maximum_integer_bits:
            raise EncounterResourceError(f"{budget.label} prospective dyadic addition cap exceeded")
        budget.dyadic_work("ADD_OR_SUBTRACT_COMBINE")
        return _Dyad(left + right, exponent, budget)

    def subtract(self, other: _Dyad, budget: _ExactBudget) -> _Dyad:
        budget.dyadic_work("ADD_OR_SUBTRACT_ALIGN")
        exponent = min(self.exponent, other.exponent)
        left_shift = self.exponent - exponent
        right_shift = other.exponent - exponent
        if (
            self.mantissa != 0
            and self.mantissa_bits + left_shift
            > budget.resources.maximum_integer_bits
        ) or (
            other.mantissa != 0
            and other.mantissa_bits + right_shift
            > budget.resources.maximum_integer_bits
        ):
            raise EncounterResourceError(f"{budget.label} prospective dyadic shift cap exceeded")
        budget.dyadic_work("ADD_OR_SUBTRACT_SHIFT_PAIR")
        left = self.mantissa << left_shift
        right = other.mantissa << right_shift
        left_bits = 0 if self.mantissa == 0 else self.mantissa_bits + left_shift
        right_bits = 0 if other.mantissa == 0 else other.mantissa_bits + right_shift
        budget.observe_known_integer_bits(left_bits)
        budget.observe_known_integer_bits(right_bits)
        carry_bits = 1 if self.mantissa != 0 and other.mantissa != 0 else 0
        if max(left_bits, right_bits) + carry_bits > budget.resources.maximum_integer_bits:
            raise EncounterResourceError(f"{budget.label} prospective dyadic subtraction cap exceeded")
        budget.dyadic_work("ADD_OR_SUBTRACT_COMBINE")
        return _Dyad(left - right, exponent, budget)

    def multiply(self, other: _Dyad, budget: _ExactBudget) -> _Dyad:
        if (
            self.mantissa != 0
            and other.mantissa != 0
            and self.mantissa_bits + other.mantissa_bits
            > budget.resources.maximum_integer_bits
        ):
            raise EncounterResourceError(
                f"{budget.label} prospective dyadic multiplication cap exceeded"
            )
        budget.dyadic_work("MULTIPLY_MANTISSA_AND_EXPONENT")
        exponent = self.exponent + other.exponent
        return _Dyad(
            self.mantissa * other.mantissa,
            exponent,
            budget,
        )

    def square(self, budget: _ExactBudget) -> _Dyad:
        return self.multiply(self, budget)

    def negate(self, budget: _ExactBudget) -> _Dyad:
        budget.dyadic_work("NEGATE")
        return _Dyad(-self.mantissa, self.exponent, budget)

    def compare(self, other: _Dyad, budget: _ExactBudget) -> int:
        budget.dyadic_work("COMPARE_ALIGN_AND_SHIFT_PAIR")
        exponent = min(self.exponent, other.exponent)
        left_shift = self.exponent - exponent
        right_shift = other.exponent - exponent
        if (
            self.mantissa != 0
            and self.mantissa_bits + left_shift
            > budget.resources.maximum_integer_bits
        ) or (
            other.mantissa != 0
            and other.mantissa_bits + right_shift
            > budget.resources.maximum_integer_bits
        ):
            raise EncounterResourceError(f"{budget.label} prospective dyadic compare cap exceeded")
        left = self.mantissa << left_shift
        right = other.mantissa << right_shift
        budget.observe_known_integer_bits(
            0 if self.mantissa == 0 else self.mantissa_bits + left_shift
        )
        budget.observe_known_integer_bits(
            0 if other.mantissa == 0 else other.mantissa_bits + right_shift
        )
        budget.dyadic_work("COMPARE_RESULT")
        return (left > right) - (left < right)

    def to_float(self, budget: _ExactBudget) -> float:
        if self.exponent >= 0:
            if (
                self.mantissa != 0
                and self.mantissa_bits + self.exponent
                > budget.resources.maximum_integer_bits
            ):
                raise EncounterResourceError(f"{budget.label} dyadic conversion cap exceeded")
            budget.dyadic_work("TO_BINARY64_RATIO_BUILD")
            numerator = self.mantissa << self.exponent
            denominator = 1
            budget.observe_known_integer_bits(
                0 if self.mantissa == 0 else self.mantissa_bits + self.exponent
            )
        else:
            numerator = self.mantissa
            if 1 - self.exponent > budget.resources.maximum_integer_bits:
                raise EncounterResourceError(f"{budget.label} dyadic conversion cap exceeded")
            budget.dyadic_work("TO_BINARY64_RATIO_BUILD")
            denominator = 1 << -self.exponent
            budget.observe_known_integer_bits(1 - self.exponent)
        budget.dyadic_work("TO_BINARY64_ROUND")
        try:
            return float(numerator / denominator)
        except OverflowError as exc:
            raise EncounterResourceError("dyadic binary64 conversion overflowed") from exc

    def record(self) -> dict[str, object]:
        return {"exponent": self.exponent, "mantissa": str(self.mantissa)}


class _Rat:
    __slots__ = ("numerator", "denominator")

    def __init__(self, numerator: int, denominator: int, budget: _ExactBudget) -> None:
        if type(numerator) is not int or type(denominator) is not int:
            raise EncounterContractError("exact rational components must be built-in integers")
        budget.rational_work("RATIONAL_CONSTRUCTION")
        if denominator == 0:
            raise EncounterDomainError("exact rational denominator cannot be zero")
        budget.integer_bits(numerator)
        budget.integer_bits(denominator)
        if denominator < 0:
            numerator = _int_negate(numerator, budget)
            denominator = _int_negate(denominator, budget)
        if numerator == 0:
            self.numerator = 0
            self.denominator = 1
        else:
            divisor = _euclidean_gcd(numerator, denominator, budget)
            self.numerator = _int_floor_divide(numerator, divisor, budget)
            self.denominator = _int_floor_divide(denominator, divisor, budget)
        budget.rational_exponent(self.numerator, self.denominator)

    @classmethod
    def from_float(cls, value: float, budget: _ExactBudget) -> _Rat:
        _authorize_binary64_ratio(value, budget, dyadic=False)
        budget.rational_work("FROM_BINARY64_RATIO_EXTRACTION")
        numerator, denominator = value.as_integer_ratio()
        return cls(numerator, denominator, budget)

    @classmethod
    def from_int(cls, value: int, budget: _ExactBudget) -> _Rat:
        if type(value) is not int:
            raise EncounterContractError("exact integer conversion requires built-in int")
        return cls(value, 1, budget)

    def add(self, other: _Rat, budget: _ExactBudget) -> _Rat:
        left = _int_multiply(self.numerator, other.denominator, budget)
        right = _int_multiply(other.numerator, self.denominator, budget)
        numerator = _int_add(left, right, budget)
        denominator = _int_multiply(self.denominator, other.denominator, budget)
        return _Rat(numerator, denominator, budget)

    def subtract(self, other: _Rat, budget: _ExactBudget) -> _Rat:
        left = _int_multiply(self.numerator, other.denominator, budget)
        right = _int_multiply(other.numerator, self.denominator, budget)
        numerator = _int_subtract(left, right, budget)
        denominator = _int_multiply(self.denominator, other.denominator, budget)
        return _Rat(numerator, denominator, budget)

    def multiply(self, other: _Rat, budget: _ExactBudget) -> _Rat:
        numerator = _int_multiply(self.numerator, other.numerator, budget)
        denominator = _int_multiply(self.denominator, other.denominator, budget)
        return _Rat(numerator, denominator, budget)

    def divide(self, other: _Rat, budget: _ExactBudget) -> _Rat:
        if other.numerator == 0:
            raise EncounterDomainError("exact rational division by zero")
        numerator = _int_multiply(self.numerator, other.denominator, budget)
        denominator = _int_multiply(self.denominator, other.numerator, budget)
        return _Rat(numerator, denominator, budget)

    def square(self, budget: _ExactBudget) -> _Rat:
        return self.multiply(self, budget)

    def negate(self, budget: _ExactBudget) -> _Rat:
        return _Rat(_int_negate(self.numerator, budget), self.denominator, budget)

    def compare(self, other: _Rat, budget: _ExactBudget) -> int:
        left = _int_multiply(self.numerator, other.denominator, budget)
        right = _int_multiply(other.numerator, self.denominator, budget)
        return _int_compare(left, right, budget)

    def record(self) -> dict[str, str]:
        return {"numerator": str(self.numerator), "denominator": str(self.denominator)}


class _Transcript:
    def __init__(
        self,
        phase: str,
        *,
        budget: _ExactBudget,
        domain: str,
        proposal_index: int | None,
    ) -> None:
        self.phase = phase
        self.budget = budget
        self.domain = domain
        self.proposal_index = proposal_index
        self.entries: list[dict[str, object]] = []
        self._entry_serialized_lengths: list[int] = []

    def _record_json_envelope(self, value: _Rat | _Dyad) -> int:
        if type(value) is _Rat:
            numerator_bits = self.budget.integer_bits(value.numerator)
            denominator_bits = self.budget.integer_bits(value.denominator)
            numerator_digits = max(1, (numerator_bits * 30103) // 100000 + 1)
            denominator_digits = max(1, (denominator_bits * 30103) // 100000 + 1)
            if value.numerator < 0:
                numerator_digits += 1
            return (
                len('{"denominator":"","numerator":""}')
                + numerator_digits
                + denominator_digits
            )
        if type(value) is _Dyad:
            mantissa_bits = self.budget.integer_bits(value.mantissa, dyadic=True)
            mantissa_digits = max(1, (mantissa_bits * 30103) // 100000 + 1)
            if value.mantissa < 0:
                mantissa_digits += 1
            # The exact exponent cap is four decimal digits in v1; reserve a
            # sign and five digits so this remains prospective if it tightens.
            exponent_digits = 6
            return (
                len('{"exponent":,"mantissa":""}')
                + exponent_digits
                + mantissa_digits
            )
        raise EncounterContractError("transcript operand has an invalid exact type")

    def _require_entry_capacity(self, maximum_entry_length: int) -> None:
        prospective_entries_length = (
            2
            + sum(self._entry_serialized_lengths)
            + len(self._entry_serialized_lengths)
            + maximum_entry_length
        )
        # Every possible fixed v1 header is below 512 ASCII bytes.  Reserving
        # that literal envelope avoids decimal conversion or JSON assembly
        # before the prospective byte check.
        header_length = 512
        prospective = (
            len(self.domain.encode("utf-8"))
            + 1
            + header_length
            - 2
            + prospective_entries_length
        )
        self.budget.require_transcript_capacity(prospective)

    def _append_prechecked(
        self,
        entry: dict[str, object],
        maximum_entry_length: int,
    ) -> None:
        serialized_entry = json.dumps(
            entry,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if len(serialized_entry) > maximum_entry_length:
            raise EncounterContractError("transcript entry exceeded its byte envelope")
        self.entries.append(entry)
        self._entry_serialized_lengths.append(len(serialized_entry))

    def comparison(
        self,
        *,
        kind: str,
        pair_index: int | None,
        left: _Rat | _Dyad,
        operator: str,
        right: _Rat | _Dyad,
        result: bool,
        branch: str | None = None,
    ) -> None:
        maximum_entry_length = (
            256
            + len(kind.encode("utf-8"))
            + (0 if branch is None else len(branch.encode("utf-8")))
            + self._record_json_envelope(left)
            + self._record_json_envelope(right)
        )
        self._require_entry_capacity(maximum_entry_length)
        self._append_prechecked(
            {
                "branch": branch,
                "kind": kind,
                "left": left.record(),
                "operator": operator,
                "pair_index": pair_index,
                "result": result,
                "right": right.record(),
            },
            maximum_entry_length,
        )

    def value(
        self,
        *,
        kind: str,
        value: _Rat | _Dyad,
        pair_index: int | None = None,
        branch: str | None = None,
    ) -> None:
        maximum_entry_length = (
            192
            + len(kind.encode("utf-8"))
            + (0 if branch is None else len(branch.encode("utf-8")))
            + self._record_json_envelope(value)
        )
        self._require_entry_capacity(maximum_entry_length)
        self._append_prechecked(
            {
                "branch": branch,
                "kind": kind,
                "pair_index": pair_index,
                "value": value.record(),
            },
            maximum_entry_length,
        )

    def finish(
        self,
        *,
        budget: _ExactBudget,
        domain: str,
        proposal_index: int | None,
        signed_step: float | None,
        disposition: str,
        force_evaluations: int,
    ) -> str:
        if budget is not self.budget or domain != self.domain:
            raise EncounterContractError("transcript budget/domain identity changed")
        if proposal_index != self.proposal_index:
            raise EncounterContractError("transcript proposal identity changed")
        payload = {
            "schema": _INITIALIZATION_SCHEMA if proposal_index is None else _WITNESS_SCHEMA,
            "method_id": ADAPTIVE_ENCOUNTER_SEGMENT_METHOD_ID,
            "phase": self.phase,
            "proposal_index": proposal_index,
            "signed_step_hex": None if signed_step is None else signed_step.hex(),
            "disposition": disposition,
            "actual_force_evaluations": force_evaluations,
            "entries": self.entries,
        }
        empty_payload = dict(payload)
        empty_payload["entries"] = []
        empty_length = len(
            json.dumps(
                empty_payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        entries_length = (
            2
            + sum(self._entry_serialized_lengths)
            + max(0, len(self._entry_serialized_lengths) - 1)
        )
        payload_length = empty_length - 2 + entries_length
        preimage_length = len(domain.encode("utf-8")) + 1 + payload_length
        budget.require_transcript_capacity(preimage_length)
        budget.transcript(preimage_length)
        serialized = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if len(serialized) != payload_length:
            raise EncounterContractError("canonical transcript sizing is inconsistent")
        return hashlib.sha256(domain.encode("utf-8") + b"\x00" + serialized).hexdigest()


@dataclass(frozen=True)
class EncounterInitializationRecord:
    exact_operations: int
    rational_operations: int
    dyadic_operations: int
    gcd_iterations: int
    transcript_bytes: int
    maximum_integer_bits: int
    maximum_rational_exponent_magnitude: int
    witness_content_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "exact_operations",
            "rational_operations",
            "dyadic_operations",
            "gcd_iterations",
            "transcript_bytes",
            "maximum_integer_bits",
            "maximum_rational_exponent_magnitude",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise EncounterContractError(f"{name} must be a nonnegative built-in integer")
        if self.exact_operations != self.rational_operations + self.dyadic_operations:
            raise EncounterContractError(
                "initialization abstract exact-work lane sum is invalid"
            )
        _sha256_hex(self.witness_content_sha256, "witness_content_sha256")


@dataclass(frozen=True)
class EncounterProposalRecord:
    index: int
    signed_step: float
    terminal_below_minimum: bool
    disposition: str
    certificate_passed: bool
    actual_force_evaluations: int
    normalized_error: float | None
    exact_operations: int
    rational_operations: int
    dyadic_operations: int
    gcd_iterations: int
    transcript_bytes: int
    maximum_integer_bits: int
    maximum_rational_exponent_magnitude: int
    witness_content_sha256: str

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index <= 0:
            raise EncounterContractError("proposal index must be positive")
        if type(self.signed_step) is not float or not math.isfinite(self.signed_step) or self.signed_step == 0.0:
            raise EncounterContractError("proposal signed_step must be finite and nonzero")
        if type(self.terminal_below_minimum) is not bool:
            raise EncounterContractError("terminal_below_minimum must be bool")
        if type(self.disposition) is not str or self.disposition not in _DISPOSITIONS:
            raise EncounterContractError("proposal disposition is not recognized")
        if type(self.certificate_passed) is not bool:
            raise EncounterContractError("certificate_passed must be bool")
        if type(self.actual_force_evaluations) is not int or not 0 <= self.actual_force_evaluations <= ENCOUNTER_STAGE_COUNT:
            raise EncounterContractError("actual_force_evaluations must be in [0,13]")
        if self.normalized_error is not None and (
            type(self.normalized_error) is not float
            or not math.isfinite(self.normalized_error)
            or self.normalized_error < 0.0
        ):
            raise EncounterContractError("normalized_error must be finite nonnegative float or None")
        for name in (
            "exact_operations",
            "rational_operations",
            "dyadic_operations",
            "gcd_iterations",
            "transcript_bytes",
            "maximum_integer_bits",
            "maximum_rational_exponent_magnitude",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise EncounterContractError(f"{name} must be a nonnegative built-in integer")
        if self.exact_operations != self.rational_operations + self.dyadic_operations:
            raise EncounterContractError(
                "proposal abstract exact-work lane sum is invalid"
            )
        _sha256_hex(self.witness_content_sha256, "witness_content_sha256")


@dataclass(frozen=True)
class EncounterExecutionCounts:
    substep_proposals: int
    accepted_substeps: int
    rejected_substeps: int
    certificate_rejections: int
    stage_guard_aborts: int
    derivative_domain_aborts: int
    completed_rk_attempts: int
    candidate_domain_rejections: int
    error_rejections: int
    force_evaluations: int
    initialization_exact_operations: int
    proposal_exact_operations: int
    total_exact_operations: int
    initialization_rational_operations: int
    proposal_rational_operations: int
    total_rational_operations: int
    initialization_dyadic_operations: int
    proposal_dyadic_operations: int
    total_dyadic_operations: int
    initialization_gcd_iterations: int
    proposal_gcd_iterations: int
    total_gcd_iterations: int
    initialization_transcript_bytes: int
    proposal_transcript_bytes: int
    total_transcript_bytes: int

    def __post_init__(self) -> None:
        for descriptor in fields(self):
            value = getattr(self, descriptor.name)
            if type(value) is not int or value < 0:
                raise EncounterContractError(
                    f"{descriptor.name} must be a nonnegative built-in integer"
                )
        if self.substep_proposals != (
            self.certificate_rejections
            + self.stage_guard_aborts
            + self.derivative_domain_aborts
            + self.completed_rk_attempts
        ):
            raise EncounterContractError("proposal accounting equation is inconsistent")
        if self.completed_rk_attempts != (
            self.candidate_domain_rejections
            + self.error_rejections
            + self.accepted_substeps
        ):
            raise EncounterContractError("completed-attempt accounting equation is inconsistent")
        if self.rejected_substeps != (
            self.certificate_rejections
            + self.stage_guard_aborts
            + self.derivative_domain_aborts
            + self.candidate_domain_rejections
            + self.error_rejections
        ):
            raise EncounterContractError("five-category rejection accounting is inconsistent")
        if self.total_exact_operations != self.initialization_exact_operations + self.proposal_exact_operations:
            raise EncounterContractError(
                "abstract exact-work lane total is inconsistent"
            )
        if self.total_rational_operations != (
            self.initialization_rational_operations
            + self.proposal_rational_operations
        ):
            raise EncounterContractError("rational-operation lane total is inconsistent")
        if self.total_dyadic_operations != (
            self.initialization_dyadic_operations
            + self.proposal_dyadic_operations
        ):
            raise EncounterContractError("dyadic-operation lane total is inconsistent")
        if self.total_exact_operations != (
            self.total_rational_operations + self.total_dyadic_operations
        ):
            raise EncounterContractError(
                "combined abstract exact-work lane total is inconsistent"
            )
        if self.total_gcd_iterations != self.initialization_gcd_iterations + self.proposal_gcd_iterations:
            raise EncounterContractError("GCD lane total is inconsistent")
        if self.total_transcript_bytes != self.initialization_transcript_bytes + self.proposal_transcript_bytes:
            raise EncounterContractError("transcript lane total is inconsistent")


def _sum_counts(left: EncounterExecutionCounts, right: EncounterExecutionCounts) -> EncounterExecutionCounts:
    return EncounterExecutionCounts(
        **{
            descriptor.name: getattr(left, descriptor.name) + getattr(right, descriptor.name)
            for descriptor in fields(EncounterExecutionCounts)
        }
    )


@dataclass(frozen=True, eq=False)
class _EncounterRun:
    final_positions: np.ndarray
    final_velocities: np.ndarray
    accepted_signed_substeps: tuple[float, ...]
    initialization_record: EncounterInitializationRecord
    proposal_ledger: tuple[EncounterProposalRecord, ...]
    force_model_ids: tuple[str, ...]
    force_ledger: tuple[ForceLedgerEntry, ...]
    counts: EncounterExecutionCounts


@dataclass(frozen=True, eq=False)
class EncounterSegmentResult:
    snapshot_id: str
    plan_id: str
    backend_id: str
    device: str
    dtype: str
    backend_spec: BackendSpec
    initial_snapshot: StateSnapshot
    force_plan: ForcePlan
    integration_spec: AdaptiveEncounterSegmentSpec
    final_positions: np.ndarray
    final_velocities: np.ndarray
    accepted_signed_substeps: tuple[float, ...]
    initialization_record: EncounterInitializationRecord
    proposal_ledger: tuple[EncounterProposalRecord, ...]
    force_model_ids: tuple[str, ...]
    force_ledger: tuple[ForceLedgerEntry, ...]
    primary_counts: EncounterExecutionCounts
    validation_replay_counts: EncounterExecutionCounts
    total_public_call_counts: EncounterExecutionCounts
    pair_table_content_sha256: str
    schedule_content_sha256: str
    result_content_sha256: str
    method_id: str = ADAPTIVE_ENCOUNTER_SEGMENT_METHOD_ID
    tableau_id: str = ENCOUNTER_TABLEAU_ID
    stage_count: int = ENCOUNTER_STAGE_COUNT
    accepted_order: int = ENCOUNTER_ACCEPTED_ORDER
    embedded_order: int = ENCOUNTER_EMBEDDED_ORDER
    defect_orientation: str = ENCOUNTER_DEFECT_ORIENTATION
    stage_epoch_policy: str = ENCOUNTER_STAGE_EPOCH_POLICY
    backend_scope: str = ENCOUNTER_BACKEND_SCOPE
    force_plan_scope: str = ENCOUNTER_FORCE_PLAN_SCOPE
    error_norm: str = ENCOUNTER_PAIR_ERROR_NORM
    certificate_claim_scope: str = ENCOUNTER_CERTIFICATE_CLAIM_SCOPE
    certificate_nonclaims: str = ENCOUNTER_CERTIFICATE_NONCLAIMS
    force_evaluation_accounting: str = ENCOUNTER_FORCE_EVALUATION_ACCOUNTING
    proposal_accounting: str = ENCOUNTER_PROPOSAL_ACCOUNTING
    public_execution_accounting_scope: str = ENCOUNTER_PUBLIC_EXECUTION_ACCOUNTING_SCOPE
    validation_replay_policy: str = ENCOUNTER_VALIDATION_REPLAY_POLICY
    validation_replay_count: int = ENCOUNTER_VALIDATION_REPLAY_COUNT
    checksum_binding_policy: str = ENCOUNTER_CHECKSUM_BINDING_POLICY
    pair_table_checksum_algorithm: str = ENCOUNTER_PAIR_TABLE_CHECKSUM_ALGORITHM
    pair_table_checksum_domain: str = ENCOUNTER_PAIR_TABLE_CHECKSUM_DOMAIN
    initialization_checksum_algorithm: str = ENCOUNTER_INITIALIZATION_CHECKSUM_ALGORITHM
    initialization_checksum_domain: str = ENCOUNTER_INITIALIZATION_CHECKSUM_DOMAIN
    exact_witness_checksum_algorithm: str = ENCOUNTER_EXACT_WITNESS_CHECKSUM_ALGORITHM
    exact_witness_checksum_domain: str = ENCOUNTER_EXACT_WITNESS_CHECKSUM_DOMAIN
    schedule_checksum_algorithm: str = ENCOUNTER_SCHEDULE_CHECKSUM_ALGORITHM
    schedule_checksum_domain: str = ENCOUNTER_SCHEDULE_CHECKSUM_DOMAIN
    result_content_checksum_algorithm: str = ENCOUNTER_RESULT_CONTENT_CHECKSUM_ALGORITHM
    result_content_checksum_domain: str = ENCOUNTER_RESULT_CONTENT_CHECKSUM_DOMAIN
    scope: str = ENCOUNTER_SEGMENT_SCOPE
    evidence_class: str = ENCOUNTER_EVIDENCE_CLASS
    adaptive: bool = True
    dense_output: bool = False
    event_detection: bool = False
    collision_response: bool = False
    globally_symplectic: bool = False
    exactly_time_reversible: bool = False
    registry_authorized: bool = False
    qualification_authorized: bool = False

    def __post_init__(self) -> None:
        _validate_result(self)

    @property
    def direction(self) -> str:
        return self.integration_spec.direction

    @property
    def initial_epoch(self) -> float:
        return self.integration_spec.initial_epoch

    @property
    def endpoint_epoch(self) -> float:
        return self.integration_spec.endpoint_epoch

    @property
    def duration(self) -> float:
        return self.integration_spec.duration

    @property
    def integrated(self) -> bool:
        return True

    @property
    def qualified(self) -> bool:
        return False


@dataclass(frozen=True)
class _StaticExact:
    pairs: tuple[tuple[int, int], ...]
    zero: _Dyad
    duration_magnitude: _Dyad
    initial_step_magnitude: _Dyad
    minimum_step_magnitude: _Dyad
    maximum_step_magnitude: _Dyad
    gravitational_parameters: tuple[_Rat, ...]
    floors: tuple[_Rat, ...]
    floor_squares: tuple[_Rat, ...]
    dyadic_floor_squares: tuple[_Dyad, ...]
    acceleration_bounds: tuple[_Rat, ...]
    initialization_record: EncounterInitializationRecord


def _exact_array(
    value: object,
    *,
    label: str,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise EncounterContractError(f"{label} must be an exact NumPy ndarray")
    array = value
    if array.dtype != dtype or array.shape != shape:
        raise EncounterContractError(
            f"{label} must have dtype {dtype} and shape {shape}"
        )
    if not array.flags.c_contiguous:
        raise EncounterContractError(f"{label} must be C-contiguous")
    if dtype == np.dtype(np.float64) and not bool(np.all(np.isfinite(array))):
        raise EncounterDomainError(f"{label} must contain only finite values")
    return array


def _utf8_bytes(value: str, label: str) -> bytes:
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise EncounterContractError(f"{label} must be valid UTF-8") from exc


def _exact_nonempty_text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise EncounterContractError(f"{label} must be an exact nonempty trimmed string")
    _utf8_bytes(value, label)
    return value


def _validate_provenance_schema(value: object, label: str) -> Provenance:
    if type(value) is not Provenance:
        raise EncounterContractError(f"{label} must be an exact Provenance")
    _exact_nonempty_text(value.source_id, f"{label}.source_id")
    _exact_nonempty_text(value.citation, f"{label}.citation")
    _exact_nonempty_text(value.version, f"{label}.version")
    _sha256_hex(value.sha256, f"{label}.sha256")
    if value.sha256 == "0" * 64:
        raise EncounterContractError(f"{label}.sha256 cannot be all zero")
    return value


def _validate_parameter_metadata_schema(
    value: object,
    label: str,
) -> ParameterMetadata:
    if type(value) is not ParameterMetadata:
        raise EncounterContractError(f"{label} must be exact ParameterMetadata")
    _exact_nonempty_text(value.parameter_id, f"{label}.parameter_id")
    _exact_nonempty_text(value.units, f"{label}.units")
    _validate_provenance_schema(value.provenance, f"{label}.provenance")
    if value.uncertainty is not None and (
        type(value.uncertainty) is not float
        or not math.isfinite(value.uncertainty)
        or value.uncertainty < 0.0
    ):
        raise EncounterContractError(
            f"{label}.uncertainty must be an exact finite nonnegative float or None"
        )
    if value.covariance_group is not None:
        _exact_nonempty_text(value.covariance_group, f"{label}.covariance_group")
    if (
        type(value.validity_start) is not float
        or type(value.validity_end) is not float
        or not math.isfinite(value.validity_start)
        or not math.isfinite(value.validity_end)
        or value.validity_start > value.validity_end
    ):
        raise EncounterContractError(
            f"{label} validity bounds must be ordered finite built-in floats"
        )
    return value


def _validate_backend_schema(value: object, label: str) -> BackendSpec:
    if type(value) is not BackendSpec:
        raise EncounterContractError(f"{label} must be an exact BackendSpec")
    for name in ("backend_id", "device", "dtype", "determinism_scope"):
        _exact_nonempty_text(getattr(value, name), f"{label}.{name}")
    if type(value.tile_size) is not int or value.tile_size <= 0:
        raise EncounterContractError(f"{label}.tile_size must be a positive built-in int")
    if (
        type(value.allow_fallback) is not bool
        or type(value.deterministic_reductions) is not bool
        or type(value.fast_math) is not bool
    ):
        raise EncounterContractError(f"{label} switches must be exact built-in bools")
    if (
        value.backend_id != "numpy"
        or value.device != "cpu"
        or value.dtype != "float64"
        or value.allow_fallback
        or not value.deterministic_reductions
        or value.fast_math
        or value.determinism_scope != "SAME_RUNTIME_DEVICE"
    ):
        raise EncounterContractError(f"{label} is outside exact NumPy CPU float64 scope")
    return value


def _validate_snapshot_schema(value: object) -> StateSnapshot:
    if type(value) is not StateSnapshot:
        raise EncounterContractError("snapshot must be an exact StateSnapshot")
    for name in (
        "snapshot_id",
        "time_scale",
        "frame",
        "origin",
        "axes",
        "length_unit",
        "time_unit",
        "mass_unit",
        "unit_system_id",
    ):
        _exact_nonempty_text(getattr(value, name), f"snapshot.{name}")
    if type(value.body_ids) is not tuple or not value.body_ids:
        raise EncounterContractError("snapshot.body_ids must be an exact string tuple")
    if len(value.body_ids) > ENCOUNTER_HARD_MAXIMUM_BODY_COUNT:
        raise EncounterContractError("snapshot.body_ids exceeds the hard body cap")
    for index, body_id in enumerate(value.body_ids):
        if type(body_id) is not str:
            raise EncounterContractError("snapshot.body_ids must contain exact strings")
        if len(body_id) > ENCOUNTER_HARD_MAXIMUM_BODY_ID_UTF8_BYTES:
            raise EncounterContractError("snapshot body ID exceeds its UTF-8 byte cap")
        _exact_nonempty_text(body_id, f"snapshot.body_ids[{index}]")
        if (
            len(_utf8_bytes(body_id, f"snapshot.body_ids[{index}]"))
            > ENCOUNTER_HARD_MAXIMUM_BODY_ID_UTF8_BYTES
        ):
            raise EncounterContractError("snapshot body ID exceeds its UTF-8 byte cap")
    if len(set(value.body_ids)) != len(value.body_ids):
        raise EncounterContractError("snapshot.body_ids must be unique")
    _validate_provenance_schema(value.provenance, "snapshot.provenance")
    return value


def _validate_newtonian_schema(value: object) -> NewtonianPointMass:
    if type(value) is not NewtonianPointMass:
        raise EncounterContractError("force model must be exact NewtonianPointMass")
    for name in ("source_ids", "target_ids"):
        identifiers = getattr(value, name)
        if type(identifiers) is not tuple or not identifiers:
            raise EncounterContractError(f"Newtonian {name} must be an exact string tuple")
        if len(identifiers) > ENCOUNTER_HARD_MAXIMUM_BODY_COUNT:
            raise EncounterContractError(f"Newtonian {name} exceeds the hard body cap")
        for index, identifier in enumerate(identifiers):
            if type(identifier) is not str:
                raise EncounterContractError(f"Newtonian {name} must contain exact strings")
            if len(identifier) > ENCOUNTER_HARD_MAXIMUM_BODY_ID_UTF8_BYTES:
                raise EncounterContractError(f"Newtonian {name} ID exceeds its UTF-8 byte cap")
            _exact_nonempty_text(identifier, f"Newtonian {name}[{index}]")
            if (
                len(_utf8_bytes(identifier, f"Newtonian {name}[{index}]"))
                > ENCOUNTER_HARD_MAXIMUM_BODY_ID_UTF8_BYTES
            ):
                raise EncounterContractError(f"Newtonian {name} ID exceeds its UTF-8 byte cap")
        if len(set(identifiers)) != len(identifiers):
            raise EncounterContractError(f"Newtonian {name} must be unique")
    _exact_nonempty_text(value.unit_system_id, "Newtonian unit_system_id")
    if type(value.parameter_metadata) is not tuple or len(value.parameter_metadata) != 1:
        raise EncounterContractError(
            "Newtonian parameter_metadata must be a one-entry exact tuple"
        )
    metadata = _validate_parameter_metadata_schema(
        value.parameter_metadata[0], "Newtonian parameter_metadata[0]"
    )
    if metadata.parameter_id != "state.gravitational_parameters":
        raise EncounterContractError("Newtonian parameter metadata identifier changed")
    return value


def _validate_force_plan_schema(value: object) -> ForcePlan:
    if type(value) is not ForcePlan:
        raise EncounterContractError("plan must be an exact ForcePlan")
    _exact_nonempty_text(value.plan_id, "plan.plan_id")
    _validate_backend_schema(value.backend, "plan.backend")
    if type(value.models) is not tuple or len(value.models) != 1:
        raise EncounterContractError("plan.models must be a one-entry exact tuple")
    _validate_newtonian_schema(value.models[0])
    if type(value.evidence_class) is not str or value.evidence_class != "MODEL_OUTPUT":
        raise EncounterContractError("plan.evidence_class must be exact MODEL_OUTPUT")
    if (
        type(value.registry_authorized) is not bool
        or value.registry_authorized
        or type(value.qualification_authorized) is not bool
        or value.qualification_authorized
    ):
        raise EncounterContractError("plan authorization controls are invalid")
    return value


def _preflight_request(
    snapshot: StateSnapshot,
    plan: ForcePlan,
    spec: AdaptiveEncounterSegmentSpec,
):
    if type(snapshot) is not StateSnapshot:
        raise EncounterContractError("snapshot must be an exact StateSnapshot")
    if type(plan) is not ForcePlan:
        raise EncounterContractError("plan must be an exact ForcePlan")
    if type(spec) is not AdaptiveEncounterSegmentSpec:
        raise EncounterContractError(
            "spec must be an exact AdaptiveEncounterSegmentSpec"
        )
    try:
        spec.__post_init__()
    except Exception as exc:
        if isinstance(exc, EncounterError):
            raise
        raise EncounterContractError("encounter specification is invalid") from exc
    _validate_snapshot_schema(snapshot)
    _validate_force_plan_schema(plan)
    if type(snapshot.epoch) is not float or not math.isfinite(snapshot.epoch):
        raise EncounterContractError("snapshot.epoch must be a finite built-in float")
    if _float_bits(snapshot.epoch) != _float_bits(spec.initial_epoch):
        raise EncounterContractError(
            "snapshot.epoch must be bit-identical to spec.initial_epoch"
        )
    if snapshot.body_ids != spec.body_order:
        raise EncounterContractError("snapshot.body_ids must equal spec.body_order")
    if snapshot.frame != ENCOUNTER_REQUIRED_FRAME:
        raise EncounterContractError(
            f"snapshot.frame must be {ENCOUNTER_REQUIRED_FRAME!r}"
        )
    if snapshot.origin != ENCOUNTER_REQUIRED_ORIGIN:
        raise EncounterContractError(
            f"snapshot.origin must be {ENCOUNTER_REQUIRED_ORIGIN!r}"
        )
    if snapshot.axes not in ENCOUNTER_ALLOWED_AXES:
        raise EncounterContractError("snapshot.axes is outside the encounter scope")
    if snapshot.time_scale not in ENCOUNTER_ALLOWED_TIME_SCALES:
        raise EncounterContractError("snapshot.time_scale is outside the encounter scope")
    if (
        plan.evidence_class != "MODEL_OUTPUT"
        or plan.registry_authorized
        or plan.qualification_authorized
    ):
        raise EncounterContractError("force plan must remain nonauthorizing MODEL_OUTPUT")
    if plan.backend.backend_id != "numpy" or plan.backend.device != "cpu":
        raise EncounterContractError("encounter integration requires NumPy CPU")
    backend = resolve_backend(plan.backend)
    if backend.name != "numpy" or backend.device != "cpu":
        raise EncounterContractError("encounter backend resolution changed scope")
    if type(plan.models) is not tuple or len(plan.models) != 1:
        raise EncounterContractError(
            "encounter force plan must contain exactly one force model"
        )
    model = plan.models[0]
    if type(model) is not NewtonianPointMass:
        raise EncounterContractError(
            "encounter force plan must contain only NewtonianPointMass"
        )
    if model.source_ids != spec.body_order or model.target_ids != spec.body_order:
        raise EncounterContractError(
            "Newtonian source_ids and target_ids must both equal body_order"
        )
    _validate_model_sequence(plan.models)
    _validate_dependencies(plan.models)
    _validate_parameter_contracts(snapshot, plan)
    lower_label = min(spec.initial_epoch, spec.endpoint_epoch)
    upper_label = max(spec.initial_epoch, spec.endpoint_epoch)
    for metadata in model.parameter_metadata:
        if (
            type(metadata.validity_start) is not float
            or type(metadata.validity_end) is not float
            or not math.isfinite(metadata.validity_start)
            or not math.isfinite(metadata.validity_end)
            or metadata.validity_start > lower_label
            or metadata.validity_end < upper_label
        ):
            raise EncounterContractError(
                "force metadata validity must cover the full public label interval"
            )

    body_count = len(spec.body_order)
    positions = _exact_array(
        snapshot.positions,
        label="positions",
        dtype=np.dtype(np.float64),
        shape=(body_count, 3),
    )
    velocities = _exact_array(
        snapshot.velocities,
        label="velocities",
        dtype=np.dtype(np.float64),
        shape=(body_count, 3),
    )
    gravitational_parameters = _exact_array(
        snapshot.gravitational_parameters,
        label="gravitational_parameters",
        dtype=np.dtype(np.float64),
        shape=(body_count,),
    )
    masses = _exact_array(
        snapshot.masses,
        label="masses",
        dtype=np.dtype(np.float64),
        shape=(body_count,),
    )
    radii = _exact_array(
        snapshot.radii,
        label="radii",
        dtype=np.dtype(np.float64),
        shape=(body_count,),
    )
    massive = _exact_array(
        snapshot.massive,
        label="massive",
        dtype=np.dtype(np.bool_),
        shape=(body_count,),
    )
    if not bool(np.all(gravitational_parameters > np.float64(0.0))):
        raise EncounterDomainError("every encounter body must have strictly positive GM")
    if not bool(np.all(masses >= np.float64(0.0))):
        raise EncounterDomainError("physical masses cannot be negative")
    if not bool(np.all(radii >= np.float64(0.0))):
        raise EncounterDomainError("radii cannot be negative")
    if not bool(np.all(massive)):
        raise EncounterContractError("every encounter body must be active and massive")
    _validate_state(backend, snapshot)
    return (
        backend,
        positions,
        velocities,
        gravitational_parameters,
        masses,
        radii,
        massive,
    )


def _copy_provenance(value: Any) -> Any:
    return replace(value)


def _owned_request(
    snapshot: StateSnapshot,
    plan: ForcePlan,
    spec: AdaptiveEncounterSegmentSpec,
    validated: tuple[Any, ...],
) -> tuple[StateSnapshot, ForcePlan, AdaptiveEncounterSegmentSpec]:
    _, positions, velocities, gm, masses, radii, massive = validated
    initial_snapshot = replace(
        snapshot,
        provenance=_copy_provenance(snapshot.provenance),
        positions=_readonly_copy(positions, np.dtype(np.float64)),
        velocities=_readonly_copy(velocities, np.dtype(np.float64)),
        gravitational_parameters=_readonly_copy(gm, np.dtype(np.float64)),
        masses=_readonly_copy(masses, np.dtype(np.float64)),
        radii=_readonly_copy(radii, np.dtype(np.float64)),
        massive=_readonly_copy(massive, np.dtype(np.bool_)),
    )
    model = plan.models[0]
    copied_metadata = tuple(
        replace(entry, provenance=_copy_provenance(entry.provenance))
        for entry in model.parameter_metadata
    )
    force_plan = replace(
        plan,
        backend=replace(plan.backend),
        models=(replace(model, parameter_metadata=copied_metadata),),
    )
    integration_spec = replace(
        spec,
        exact_rational_resources=replace(spec.exact_rational_resources),
    )
    return initial_snapshot, force_plan, integration_spec


def _canonical_pairs(body_count: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (left, right)
        for left in range(body_count - 1)
        for right in range(left + 1, body_count)
    )


def _rat_vector(value: np.ndarray, budget: _ExactBudget) -> tuple[_Rat, ...]:
    return tuple(_Rat.from_float(float(component), budget) for component in value)


def _dyadic_vector(value: np.ndarray, budget: _ExactBudget) -> tuple[_Dyad, ...]:
    return tuple(_Dyad.from_float(float(component), budget) for component in value)


def _dyadic_squared_distance(
    left: tuple[_Dyad, ...],
    right: tuple[_Dyad, ...],
    budget: _ExactBudget,
) -> _Dyad:
    differences = tuple(
        right_component.subtract(left_component, budget)
        for left_component, right_component in zip(left, right)
    )
    first_two = differences[0].square(budget).add(
        differences[1].square(budget), budget
    )
    return first_two.add(differences[2].square(budget), budget)


def _rat_subtract_vector(
    left: tuple[_Rat, ...],
    right: tuple[_Rat, ...],
    budget: _ExactBudget,
) -> tuple[_Rat, ...]:
    return tuple(a.subtract(b, budget) for a, b in zip(left, right))


def _rat_add_vector(
    left: tuple[_Rat, ...],
    right: tuple[_Rat, ...],
    budget: _ExactBudget,
) -> tuple[_Rat, ...]:
    return tuple(a.add(b, budget) for a, b in zip(left, right))


def _rat_scale_vector(
    value: tuple[_Rat, ...],
    scale: _Rat,
    budget: _ExactBudget,
) -> tuple[_Rat, ...]:
    return tuple(component.multiply(scale, budget) for component in value)


def _rat_dot(
    left: tuple[_Rat, ...],
    right: tuple[_Rat, ...],
    budget: _ExactBudget,
) -> _Rat:
    total = _Rat.from_int(0, budget)
    for a, b in zip(left, right):
        total = total.add(a.multiply(b, budget), budget)
    return total


def _initialize_exact(
    snapshot: StateSnapshot,
    spec: AdaptiveEncounterSegmentSpec,
) -> _StaticExact:
    resources = spec.exact_rational_resources
    segment = _MutableUsage()
    budget = _ExactBudget(
        resources=resources,
        segment=segment,
        operation_limit=resources.maximum_initialization_operations,
        gcd_limit=resources.maximum_initialization_gcd_iterations,
        transcript_limit=resources.maximum_initialization_transcript_bytes,
        label="initialization",
    )
    transcript = _Transcript(
        "INITIALIZATION",
        budget=budget,
        domain=ENCOUNTER_INITIALIZATION_CHECKSUM_DOMAIN,
        proposal_index=None,
    )
    pairs = _canonical_pairs(len(spec.body_order))
    budget.dyadic_work("DURATION_ABSOLUTE")
    duration = _Dyad.from_float(abs(spec.duration), budget)
    initial_step = _Dyad.from_float(spec.initial_step_magnitude, budget)
    minimum_step = _Dyad.from_float(spec.minimum_step_magnitude, budget)
    maximum_step = _Dyad.from_float(spec.maximum_step_magnitude, budget)
    transcript.value(kind="duration_magnitude", value=duration)
    transcript.value(kind="initial_step_magnitude", value=initial_step)
    transcript.value(kind="minimum_step_magnitude", value=minimum_step)
    transcript.value(kind="maximum_step_magnitude", value=maximum_step)
    zero = _Dyad.from_int(0, budget)
    rational_zero = _Rat.from_int(0, budget)
    gravitational_parameters = tuple(
        _Rat.from_float(float(value), budget)
        for value in snapshot.gravitational_parameters
    )
    radii = tuple(_Dyad.from_float(float(value), budget) for value in snapshot.radii)
    floors = tuple(
        _Rat.from_float(value, budget) for value in spec.pair_certification_floors
    )
    floor_squares = tuple(value.square(budget) for value in floors)
    dyadic_floors = tuple(
        _Dyad.from_float(value, budget) for value in spec.pair_certification_floors
    )
    dyadic_floor_squares = tuple(value.square(budget) for value in dyadic_floors)
    exact_positions = tuple(
        _dyadic_vector(snapshot.positions[index], budget)
        for index in range(len(spec.body_order))
    )
    for index, gm in enumerate(gravitational_parameters):
        positive = gm.compare(rational_zero, budget) > 0
        transcript.comparison(
            kind="positive_gm",
            pair_index=None,
            left=gm,
            operator=">",
            right=rational_zero,
            result=positive,
            branch=str(index),
        )
        if not positive:
            raise EncounterDomainError("every retained GM must be exactly positive")
    for pair_index, (left, right) in enumerate(pairs):
        radius_sum = radii[left].add(radii[right], budget)
        floor_valid = dyadic_floors[pair_index].compare(radius_sum, budget) >= 0
        transcript.comparison(
            kind="floor_at_least_radius_sum",
            pair_index=pair_index,
            left=dyadic_floors[pair_index],
            operator=">=",
            right=radius_sum,
            result=floor_valid,
        )
        if not floor_valid:
            raise EncounterContractError(
                "pair certification floor is below the exact radii sum"
            )
        distance_squared = _dyadic_squared_distance(
            exact_positions[left], exact_positions[right], budget
        )
        initial_clear = (
            distance_squared.compare(dyadic_floor_squares[pair_index], budget) > 0
        )
        transcript.comparison(
            kind="initial_squared_separation",
            pair_index=pair_index,
            left=distance_squared,
            operator=">",
            right=dyadic_floor_squares[pair_index],
            result=initial_clear,
        )
        if not initial_clear:
            raise EncounterDomainError(
                "initial pair separation must be strictly above its floor"
            )

    acceleration_bounds: list[_Rat] = []
    pair_lookup = {pair: index for index, pair in enumerate(pairs)}
    for body in range(len(spec.body_order)):
        total = _Rat.from_int(0, budget)
        for other in range(len(spec.body_order)):
            if body == other:
                continue
            pair = (body, other) if body < other else (other, body)
            pair_index = pair_lookup[pair]
            term = gravitational_parameters[other].divide(
                floor_squares[pair_index], budget
            )
            total = total.add(term, budget)
        acceleration_bounds.append(total)
        transcript.value(kind="acceleration_bound", value=total, branch=str(body))

    digest = transcript.finish(
        budget=budget,
        domain=ENCOUNTER_INITIALIZATION_CHECKSUM_DOMAIN,
        proposal_index=None,
        signed_step=None,
        disposition="INITIALIZED",
        force_evaluations=0,
    )
    record = EncounterInitializationRecord(
        exact_operations=budget.local.exact_operations,
        rational_operations=budget.local.rational_operations,
        dyadic_operations=budget.local.dyadic_operations,
        gcd_iterations=budget.local.gcd_iterations,
        transcript_bytes=budget.local.transcript_bytes,
        maximum_integer_bits=budget.local.maximum_integer_bits,
        maximum_rational_exponent_magnitude=(
            budget.local.maximum_rational_exponent_magnitude
        ),
        witness_content_sha256=digest,
    )
    return _StaticExact(
        pairs=pairs,
        zero=zero,
        duration_magnitude=duration,
        initial_step_magnitude=initial_step,
        minimum_step_magnitude=minimum_step,
        maximum_step_magnitude=maximum_step,
        gravitational_parameters=gravitational_parameters,
        floors=floors,
        floor_squares=floor_squares,
        dyadic_floor_squares=dyadic_floor_squares,
        acceleration_bounds=tuple(acceleration_bounds),
        initialization_record=record,
    )


def _record_compare(
    transcript: _Transcript,
    *,
    kind: str,
    pair_index: int | None,
    left: _Rat | _Dyad,
    operator: str,
    right: _Rat | _Dyad,
    comparison: int,
    branch: str | None = None,
) -> bool:
    if operator == ">":
        result = comparison > 0
    elif operator == ">=":
        result = comparison >= 0
    elif operator == "<":
        result = comparison < 0
    elif operator == "<=":
        result = comparison <= 0
    elif operator == "==":
        result = comparison == 0
    else:
        raise EncounterContractError("unsupported exact comparison operator")
    transcript.comparison(
        kind=kind,
        pair_index=pair_index,
        left=left,
        operator=operator,
        right=right,
        result=result,
        branch=branch,
    )
    return result


def _stage_guard(
    *,
    positions: np.ndarray,
    velocities: np.ndarray,
    static: _StaticExact,
    budget: _ExactBudget,
    transcript: _Transcript,
    phase: str,
) -> bool:
    if (
        type(positions) is not np.ndarray
        or type(velocities) is not np.ndarray
        or positions.dtype != np.dtype(np.float64)
        or velocities.dtype != np.dtype(np.float64)
        or positions.shape != velocities.shape
        or positions.ndim != 2
        or positions.shape[1:] != (3,)
        or not bool(np.all(np.isfinite(positions)))
        or not bool(np.all(np.isfinite(velocities)))
    ):
        return False
    exact_positions = tuple(
        _dyadic_vector(positions[index], budget)
        for index in range(positions.shape[0])
    )
    all_clear = True
    for pair_index, (left, right) in enumerate(static.pairs):
        distance_squared = _dyadic_squared_distance(
            exact_positions[left], exact_positions[right], budget
        )
        clear = _record_compare(
            transcript,
            kind=f"{phase}_squared_separation",
            pair_index=pair_index,
            left=distance_squared,
            operator=">",
            right=static.dyadic_floor_squares[pair_index],
            comparison=distance_squared.compare(
                static.dyadic_floor_squares[pair_index], budget
            ),
        )
        all_clear = all_clear and clear
    return all_clear


def _certificate(
    *,
    positions: np.ndarray,
    velocities: np.ndarray,
    signed_step: float,
    static: _StaticExact,
    budget: _ExactBudget,
    transcript: _Transcript,
) -> bool:
    exact_positions = tuple(
        _rat_vector(positions[index], budget) for index in range(positions.shape[0])
    )
    exact_velocities = tuple(
        _rat_vector(velocities[index], budget)
        for index in range(velocities.shape[0])
    )
    budget.rational_work("SIGNED_STEP_ABSOLUTE")
    delta = _Rat.from_float(abs(signed_step), budget)
    delta_squared = delta.square(budget)
    half = _Rat(1, 2, budget)
    zero = _Rat.from_int(0, budget)
    sign = _Rat.from_int(1 if signed_step > 0.0 else -1, budget)
    all_certified = True
    for pair_index, (left, right) in enumerate(static.pairs):
        displacement = _rat_subtract_vector(
            exact_positions[right], exact_positions[left], budget
        )
        relative_velocity = _rat_subtract_vector(
            exact_velocities[right], exact_velocities[left], budget
        )
        signed_relative_velocity = _rat_scale_vector(
            relative_velocity, sign, budget
        )
        b_value = _rat_dot(displacement, signed_relative_velocity, budget)
        c_value = _rat_dot(
            signed_relative_velocity, signed_relative_velocity, budget
        )
        q_value = _rat_dot(displacement, displacement, budget)
        c_zero = _record_compare(
            transcript,
            kind="linear_minimum_c_zero",
            pair_index=pair_index,
            left=c_value,
            operator="==",
            right=zero,
            comparison=c_value.compare(zero, budget),
        )
        b_nonnegative = _record_compare(
            transcript,
            kind="linear_minimum_b_nonnegative",
            pair_index=pair_index,
            left=b_value,
            operator=">=",
            right=zero,
            comparison=b_value.compare(zero, budget),
        )
        if c_zero or b_nonnegative:
            ell_squared = q_value
            branch = "START"
        else:
            negative_b = b_value.negate(budget)
            c_delta = c_value.multiply(delta, budget)
            endpoint_branch = _record_compare(
                transcript,
                kind="linear_minimum_endpoint_branch",
                pair_index=pair_index,
                left=negative_b,
                operator=">=",
                right=c_delta,
                comparison=negative_b.compare(c_delta, budget),
            )
            if endpoint_branch:
                endpoint_displacement = _rat_add_vector(
                    displacement,
                    _rat_scale_vector(signed_relative_velocity, delta, budget),
                    budget,
                )
                ell_squared = _rat_dot(
                    endpoint_displacement, endpoint_displacement, budget
                )
                branch = "ENDPOINT"
            else:
                ell_squared = q_value.subtract(
                    b_value.square(budget).divide(c_value, budget), budget
                )
                branch = "INTERIOR"
        transcript.value(
            kind="linear_minimum_squared",
            pair_index=pair_index,
            value=ell_squared,
            branch=branch,
        )
        relative_bound = static.acceleration_bounds[left].add(
            static.acceleration_bounds[right], budget
        )
        remainder = half.multiply(relative_bound, budget).multiply(
            delta_squared, budget
        )
        required_radius = static.floors[pair_index].add(remainder, budget)
        required_squared = required_radius.square(budget)
        certified = _record_compare(
            transcript,
            kind="first_crossing_clearance",
            pair_index=pair_index,
            left=ell_squared,
            operator=">",
            right=required_squared,
            comparison=ell_squared.compare(required_squared, budget),
            branch=branch,
        )
        all_certified = all_certified and certified
    return all_certified


def _scheduler_step(
    *,
    remaining: _Dyad,
    proposed_magnitude: float,
    static: _StaticExact,
    budget: _ExactBudget,
    transcript: _Transcript,
    direction: float,
) -> tuple[float, _Dyad, bool]:
    proposed = _Dyad.from_float(proposed_magnitude, budget)
    target = remaining
    if _record_compare(
        transcript,
        kind="scheduler_remaining_vs_proposed",
        pair_index=None,
        left=remaining,
        operator=">",
        right=proposed,
        comparison=remaining.compare(proposed, budget),
    ):
        target = proposed
    if _record_compare(
        transcript,
        kind="scheduler_target_vs_maximum",
        pair_index=None,
        left=target,
        operator=">",
        right=static.maximum_step_magnitude,
        comparison=target.compare(static.maximum_step_magnitude, budget),
    ):
        target = static.maximum_step_magnitude
    candidate = target.to_float(budget)
    if not math.isfinite(candidate) or candidate <= 0.0:
        raise EncounterStepLimitError("scheduler did not produce a finite positive chunk")
    candidate_exact = _Dyad.from_float(candidate, budget)
    if candidate_exact.compare(target, budget) > 0:
        budget.dyadic_work("SCHEDULER_NEXTAFTER_DOWN")
        candidate = math.nextafter(candidate, 0.0)
        if not math.isfinite(candidate) or candidate <= 0.0:
            raise EncounterStepLimitError(
                "scheduler cannot represent a positive chunk below the target"
            )
        candidate_exact = _Dyad.from_float(candidate, budget)
    no_overshoot = _record_compare(
        transcript,
        kind="scheduler_chunk_no_overshoot",
        pair_index=None,
        left=candidate_exact,
        operator="<=",
        right=target,
        comparison=candidate_exact.compare(target, budget),
    )
    if not no_overshoot:
        raise EncounterStepLimitError("scheduler chunk exceeds its exact target")
    budget.dyadic_work("SCHEDULER_SUCCESSOR_PROBE")
    successor = math.nextafter(candidate, math.inf)
    if math.isfinite(successor):
        successor_exact = _Dyad.from_float(successor, budget)
        maximal = _record_compare(
            transcript,
            kind="scheduler_successor_exceeds_target",
            pair_index=None,
            left=successor_exact,
            operator=">",
            right=target,
            comparison=successor_exact.compare(target, budget),
        )
        if not maximal:
            raise EncounterStepLimitError(
                "scheduler chunk is not the largest binary64 below its target"
            )
    terminal_below_minimum = remaining.compare(static.minimum_step_magnitude, budget) < 0
    signed = math.copysign(candidate, direction)
    if type(signed) is not float or not math.isfinite(signed) or signed == 0.0:
        raise EncounterStepLimitError("scheduler signed chunk is invalid")
    return signed, candidate_exact, terminal_below_minimum


def _validate_trial_arrays(
    trial: _RKF78StepResult,
    state_shape: tuple[int, int],
) -> bool:
    arrays = (
        trial.accepted_positions,
        trial.accepted_velocities,
        trial.embedded_positions,
        trial.embedded_velocities,
        trial.position_defect,
        trial.velocity_defect,
        trial.accepted_position_carry,
        trial.accepted_velocity_carry,
    )
    for array in arrays:
        if (
            type(array) is not np.ndarray
            or array.dtype != np.dtype(np.float64)
            or array.shape != state_shape
            or not array.flags.c_contiguous
            or not bool(np.all(np.isfinite(array)))
        ):
            return False
    return True


def _left_to_right_weighted_centroid(
    values: np.ndarray,
    gravitational_parameters: np.ndarray,
) -> np.ndarray:
    total_gm = np.float64(0.0)
    weighted = np.zeros(3, dtype=np.float64)
    for index in range(values.shape[0]):
        total_gm = np.float64(
            total_gm + np.float64(gravitational_parameters[index])
        )
        weighted = weighted + np.float64(gravitational_parameters[index]) * values[index]
    return weighted / total_gm


def _normalized_error(
    *,
    current_positions: np.ndarray,
    current_velocities: np.ndarray,
    trial: _RKF78StepResult,
    gravitational_parameters: np.ndarray,
    static: _StaticExact,
    spec: AdaptiveEncounterSegmentSpec,
) -> float:
    maximum = np.float64(0.0)
    for pair_index, (left, right) in enumerate(static.pairs):
        current_position = current_positions[right] - current_positions[left]
        accepted_position = (
            trial.accepted_positions[right] - trial.accepted_positions[left]
        )
        position_defect = trial.position_defect[right] - trial.position_defect[left]
        current_velocity = current_velocities[right] - current_velocities[left]
        accepted_velocity = (
            trial.accepted_velocities[right] - trial.accepted_velocities[left]
        )
        velocity_defect = trial.velocity_defect[right] - trial.velocity_defect[left]
        for component in range(3):
            position_scale = np.float64(spec.pair_position_atols[pair_index]) + np.float64(
                spec.pair_position_rtol
            ) * np.maximum(
                np.abs(current_position[component]),
                np.abs(accepted_position[component]),
            )
            velocity_scale = np.float64(spec.pair_velocity_atols[pair_index]) + np.float64(
                spec.pair_velocity_rtol
            ) * np.maximum(
                np.abs(current_velocity[component]),
                np.abs(accepted_velocity[component]),
            )
            if (
                not bool(np.isfinite(position_scale))
                or not bool(np.isfinite(velocity_scale))
                or position_scale <= np.float64(0.0)
                or velocity_scale <= np.float64(0.0)
            ):
                raise EncounterDomainError("pair error scale left the finite domain")
            maximum = np.maximum(
                maximum, np.abs(position_defect[component]) / position_scale
            )
            maximum = np.maximum(
                maximum, np.abs(velocity_defect[component]) / velocity_scale
            )

    current_position_centroid = _left_to_right_weighted_centroid(
        current_positions, gravitational_parameters
    )
    accepted_position_centroid = _left_to_right_weighted_centroid(
        trial.accepted_positions, gravitational_parameters
    )
    position_defect_centroid = _left_to_right_weighted_centroid(
        trial.position_defect, gravitational_parameters
    )
    current_velocity_centroid = _left_to_right_weighted_centroid(
        current_velocities, gravitational_parameters
    )
    accepted_velocity_centroid = _left_to_right_weighted_centroid(
        trial.accepted_velocities, gravitational_parameters
    )
    velocity_defect_centroid = _left_to_right_weighted_centroid(
        trial.velocity_defect, gravitational_parameters
    )
    for component in range(3):
        position_scale = np.float64(spec.gm_centroid_position_atol) + np.float64(
            spec.gm_centroid_position_rtol
        ) * np.maximum(
            np.abs(current_position_centroid[component]),
            np.abs(accepted_position_centroid[component]),
        )
        velocity_scale = np.float64(spec.gm_centroid_velocity_atol) + np.float64(
            spec.gm_centroid_velocity_rtol
        ) * np.maximum(
            np.abs(current_velocity_centroid[component]),
            np.abs(accepted_velocity_centroid[component]),
        )
        if (
            not bool(np.isfinite(position_scale))
            or not bool(np.isfinite(velocity_scale))
            or position_scale <= np.float64(0.0)
            or velocity_scale <= np.float64(0.0)
        ):
            raise EncounterDomainError("centroid error scale left the finite domain")
        maximum = np.maximum(
            maximum, np.abs(position_defect_centroid[component]) / position_scale
        )
        maximum = np.maximum(
            maximum, np.abs(velocity_defect_centroid[component]) / velocity_scale
        )
    result = float(maximum)
    if not math.isfinite(result) or result < 0.0:
        raise EncounterDomainError("normalized encounter error is nonfinite")
    return result


def _controller_factor(
    error: float,
    spec: AdaptiveEncounterSegmentSpec,
) -> float:
    if error == 0.0:
        return spec.maximum_scale_factor
    factor = spec.safety_factor * error ** (-ENCOUNTER_CONTROLLER_EXPONENT)
    if not math.isfinite(factor) or factor <= 0.0:
        raise EncounterDomainError("adaptive controller produced an invalid factor")
    return min(
        spec.maximum_scale_factor,
        max(spec.minimum_scale_factor, factor),
    )


def _reduced_proposal(
    *,
    current_magnitude: float,
    factor: float,
    minimum_magnitude: float,
) -> float:
    product = current_magnitude * factor
    if not math.isfinite(product) or product <= 0.0:
        raise EncounterStepLimitError("step reduction produced an invalid magnitude")
    reduced = max(minimum_magnitude, product)
    if reduced >= current_magnitude:
        next_lower = math.nextafter(current_magnitude, 0.0)
        if next_lower < minimum_magnitude or next_lower <= 0.0:
            raise EncounterStepLimitError("a rejected step cannot be reduced")
        reduced = next_lower
    return float(reduced)


def _accepted_proposal(
    *,
    current_magnitude: float,
    factor: float,
    spec: AdaptiveEncounterSegmentSpec,
) -> float:
    product = current_magnitude * factor
    if not math.isfinite(product) or product <= 0.0:
        raise EncounterStepLimitError("accepted-step controller left the finite domain")
    return float(
        min(
            spec.maximum_step_magnitude,
            max(spec.minimum_step_magnitude, product),
        )
    )


def _record_diagnostic_bytes(
    record: EncounterInitializationRecord | EncounterProposalRecord,
    resources: EncounterExactRationalResourceSpec,
    accumulated: int,
) -> int:
    prospective = resources.maximum_witness_diagnostic_bytes_per_proposal
    if accumulated + prospective > resources.maximum_witness_ledger_bytes:
        raise EncounterResourceError("retained witness ledger exceeds its prospective cap")
    size = len(_canonical_json(record))
    if size > resources.maximum_witness_diagnostic_bytes_per_proposal:
        raise EncounterResourceError("one retained witness diagnostic exceeds its cap")
    if accumulated + size > resources.maximum_witness_ledger_bytes:
        raise EncounterResourceError("retained witness ledger exceeds its cap")
    return accumulated + size


def _proposal_record(
    *,
    index: int,
    signed_step: float,
    terminal_below_minimum: bool,
    disposition: str,
    certificate_passed: bool,
    actual_force_evaluations: int,
    normalized_error: float | None,
    budget: _ExactBudget,
    transcript: _Transcript,
) -> EncounterProposalRecord:
    digest = transcript.finish(
        budget=budget,
        domain=ENCOUNTER_EXACT_WITNESS_CHECKSUM_DOMAIN,
        proposal_index=index,
        signed_step=signed_step,
        disposition=disposition,
        force_evaluations=actual_force_evaluations,
    )
    return EncounterProposalRecord(
        index=index,
        signed_step=signed_step,
        terminal_below_minimum=terminal_below_minimum,
        disposition=disposition,
        certificate_passed=certificate_passed,
        actual_force_evaluations=actual_force_evaluations,
        normalized_error=normalized_error,
        exact_operations=budget.local.exact_operations,
        rational_operations=budget.local.rational_operations,
        dyadic_operations=budget.local.dyadic_operations,
        gcd_iterations=budget.local.gcd_iterations,
        transcript_bytes=budget.local.transcript_bytes,
        maximum_integer_bits=budget.local.maximum_integer_bits,
        maximum_rational_exponent_magnitude=(
            budget.local.maximum_rational_exponent_magnitude
        ),
        witness_content_sha256=digest,
    )


def _execution_counts(
    counters: dict[str, int],
    initialization: EncounterInitializationRecord,
    proposals: tuple[EncounterProposalRecord, ...],
) -> EncounterExecutionCounts:
    proposal_operations = sum(record.exact_operations for record in proposals)
    proposal_rational_operations = sum(
        record.rational_operations for record in proposals
    )
    proposal_dyadic_operations = sum(
        record.dyadic_operations for record in proposals
    )
    proposal_gcd = sum(record.gcd_iterations for record in proposals)
    proposal_transcript = sum(record.transcript_bytes for record in proposals)
    return EncounterExecutionCounts(
        substep_proposals=counters["substep_proposals"],
        accepted_substeps=counters["accepted_substeps"],
        rejected_substeps=counters["rejected_substeps"],
        certificate_rejections=counters["certificate_rejections"],
        stage_guard_aborts=counters["stage_guard_aborts"],
        derivative_domain_aborts=counters["derivative_domain_aborts"],
        completed_rk_attempts=counters["completed_rk_attempts"],
        candidate_domain_rejections=counters["candidate_domain_rejections"],
        error_rejections=counters["error_rejections"],
        force_evaluations=counters["force_evaluations"],
        initialization_exact_operations=initialization.exact_operations,
        proposal_exact_operations=proposal_operations,
        total_exact_operations=initialization.exact_operations + proposal_operations,
        initialization_rational_operations=initialization.rational_operations,
        proposal_rational_operations=proposal_rational_operations,
        total_rational_operations=(
            initialization.rational_operations + proposal_rational_operations
        ),
        initialization_dyadic_operations=initialization.dyadic_operations,
        proposal_dyadic_operations=proposal_dyadic_operations,
        total_dyadic_operations=(
            initialization.dyadic_operations + proposal_dyadic_operations
        ),
        initialization_gcd_iterations=initialization.gcd_iterations,
        proposal_gcd_iterations=proposal_gcd,
        total_gcd_iterations=initialization.gcd_iterations + proposal_gcd,
        initialization_transcript_bytes=initialization.transcript_bytes,
        proposal_transcript_bytes=proposal_transcript,
        total_transcript_bytes=initialization.transcript_bytes + proposal_transcript,
    )


def _execute(
    snapshot: StateSnapshot,
    plan: ForcePlan,
    spec: AdaptiveEncounterSegmentSpec,
) -> _EncounterRun:
    validated = _preflight_request(snapshot, plan, spec)
    backend = validated[0]
    static = _initialize_exact(snapshot, spec)
    resources = spec.exact_rational_resources
    segment_usage = _MutableUsage(
        exact_operations=static.initialization_record.exact_operations,
        rational_operations=static.initialization_record.rational_operations,
        dyadic_operations=static.initialization_record.dyadic_operations,
        gcd_iterations=static.initialization_record.gcd_iterations,
        transcript_bytes=static.initialization_record.transcript_bytes,
        maximum_integer_bits=static.initialization_record.maximum_integer_bits,
        maximum_rational_exponent_magnitude=(
            static.initialization_record.maximum_rational_exponent_magnitude
        ),
    )
    ledger_bytes = _record_diagnostic_bytes(
        static.initialization_record, resources, 0
    )
    current_positions = np.array(
        snapshot.positions, dtype=np.float64, order="C", copy=True, subok=False
    )
    current_velocities = np.array(
        snapshot.velocities, dtype=np.float64, order="C", copy=True, subok=False
    )
    current_position_carry = np.zeros_like(current_positions, dtype=np.float64)
    current_velocity_carry = np.zeros_like(current_velocities, dtype=np.float64)
    remaining = static.duration_magnitude
    proposed_magnitude = spec.initial_step_magnitude
    direction = 1.0 if spec.duration > 0.0 else -1.0
    accepted_steps: list[float] = []
    proposal_records: list[EncounterProposalRecord] = []
    force_ledger: tuple[ForceLedgerEntry, ...] | None = None
    force_model_ids: tuple[str, ...] | None = None
    consecutive_rejections = 0
    counters = {
        "substep_proposals": 0,
        "accepted_substeps": 0,
        "rejected_substeps": 0,
        "certificate_rejections": 0,
        "stage_guard_aborts": 0,
        "derivative_domain_aborts": 0,
        "completed_rk_attempts": 0,
        "candidate_domain_rejections": 0,
        "error_rejections": 0,
        "force_evaluations": 0,
    }

    while remaining.mantissa != 0:
        if counters["accepted_substeps"] >= spec.maximum_accepted_substeps:
            raise EncounterStepLimitError("maximum_accepted_substeps exhausted")
        if counters["substep_proposals"] >= spec.maximum_substep_proposals:
            raise EncounterStepLimitError("maximum_substep_proposals exhausted")
        proposal_index = counters["substep_proposals"] + 1
        budget = _ExactBudget(
            resources=resources,
            segment=segment_usage,
            operation_limit=resources.maximum_operations_per_proposal,
            gcd_limit=resources.maximum_gcd_iterations_per_proposal,
            transcript_limit=resources.maximum_witness_transcript_bytes_per_proposal,
            label=f"proposal {proposal_index}",
        )
        transcript = _Transcript(
            "PROPOSAL",
            budget=budget,
            domain=ENCOUNTER_EXACT_WITNESS_CHECKSUM_DOMAIN,
            proposal_index=proposal_index,
        )
        local_offset_magnitude = static.duration_magnitude.subtract(remaining, budget)
        local_offset = (
            local_offset_magnitude
            if direction > 0.0
            else local_offset_magnitude.negate(budget)
        )
        transcript.value(kind="exact_local_offset_before", value=local_offset)
        for stage_index, coefficient in enumerate(_C):
            transcript.value(
                kind="rkf78_c_table",
                value=_Dyad.from_float(float(coefficient), budget),
                branch=str(stage_index),
            )
        signed_step, step_exact, terminal_below_minimum = _scheduler_step(
            remaining=remaining,
            proposed_magnitude=proposed_magnitude,
            static=static,
            budget=budget,
            transcript=transcript,
            direction=direction,
        )
        transcript.value(kind="exact_proposed_signed_step", value=(
            step_exact if direction > 0.0 else step_exact.negate(budget)
        ))
        counters["substep_proposals"] += 1
        actual_force_evaluations = 0
        certificate_passed = _certificate(
            positions=current_positions,
            velocities=current_velocities,
            signed_step=signed_step,
            static=static,
            budget=budget,
            transcript=transcript,
        )
        disposition: str
        normalized_error: float | None = None
        trial: _RKF78StepResult | None = None
        new_remaining: _Dyad | None = None

        if not certificate_passed:
            disposition = "CERTIFICATE_REJECTION"
        else:

            def derivative(
                _ignored_stage_epoch: float,
                stage_positions: np.ndarray,
                stage_velocities: np.ndarray,
            ) -> tuple[np.ndarray, np.ndarray]:
                nonlocal actual_force_evaluations, force_ledger, force_model_ids
                if not _stage_guard(
                    positions=stage_positions,
                    velocities=stage_velocities,
                    static=static,
                    budget=budget,
                    transcript=transcript,
                    phase=f"stage_{actual_force_evaluations}",
                ):
                    raise _StageGuardAbort
                if counters["force_evaluations"] >= spec.maximum_force_evaluations:
                    raise EncounterStepLimitError("maximum_force_evaluations exhausted")
                actual_force_evaluations += 1
                counters["force_evaluations"] += 1
                stage_snapshot = replace(
                    snapshot,
                    epoch=spec.initial_epoch,
                    positions=stage_positions,
                    velocities=stage_velocities,
                )
                try:
                    evaluation = evaluate_force_plan(stage_snapshot, plan)
                except ForceDomainError as exc:
                    raise _DerivativeDomainAbort from exc
                acceleration = evaluation.total_acceleration
                if (
                    type(acceleration) is not np.ndarray
                    or acceleration.dtype != np.dtype(np.float64)
                    or acceleration.shape != current_positions.shape
                    or not bool(np.all(np.isfinite(acceleration)))
                    or type(stage_velocities) is not np.ndarray
                    or not bool(np.all(np.isfinite(stage_velocities)))
                ):
                    raise _DerivativeDomainAbort
                if force_ledger is None:
                    force_ledger = evaluation.ledger
                    force_model_ids = evaluation.applied_model_ids
                elif (
                    evaluation.applied_model_ids != force_model_ids
                    or _canonical_json(evaluation.ledger)
                    != _canonical_json(force_ledger)
                ):
                    raise EncounterContractError(
                        "force identity or ledger changed between stages"
                    )
                return stage_velocities, acceleration

            try:
                trial = _rkf78_step(
                    backend=backend,
                    epoch=spec.initial_epoch,
                    endpoint_epoch=spec.initial_epoch,
                    step=signed_step,
                    positions=current_positions,
                    velocities=current_velocities,
                    position_carry=current_position_carry,
                    velocity_carry=current_velocity_carry,
                    derivative=derivative,
                )
            except _StageGuardAbort:
                disposition = "STAGE_GUARD_ABORT"
            except _DerivativeDomainAbort:
                disposition = "DERIVATIVE_DOMAIN_ABORT"
            else:
                counters["completed_rk_attempts"] += 1
                if actual_force_evaluations != ENCOUNTER_STAGE_COUNT:
                    raise EncounterContractError(
                        "a completed RKF78 attempt must have thirteen force calls"
                    )
                if not _validate_trial_arrays(trial, current_positions.shape):
                    disposition = "CANDIDATE_DOMAIN_REJECTION"
                elif not _stage_guard(
                    positions=trial.accepted_positions,
                    velocities=trial.accepted_velocities,
                    static=static,
                    budget=budget,
                    transcript=transcript,
                    phase="candidate",
                ):
                    disposition = "CANDIDATE_DOMAIN_REJECTION"
                else:
                    try:
                        normalized_error = _normalized_error(
                            current_positions=current_positions,
                            current_velocities=current_velocities,
                            trial=trial,
                            gravitational_parameters=snapshot.gravitational_parameters,
                            static=static,
                            spec=spec,
                        )
                    except EncounterDomainError:
                        disposition = "CANDIDATE_DOMAIN_REJECTION"
                    else:
                        if normalized_error > 1.0:
                            disposition = "ERROR_REJECTION"
                        else:
                            disposition = "ACCEPTED"
                            new_remaining = remaining.subtract(step_exact, budget)
                            if new_remaining.compare(static.zero, budget) < 0:
                                raise EncounterStepLimitError(
                                    "accepted chunk overran exact remaining duration"
                                )

        if disposition != "ACCEPTED":
            at_or_below_minimum = _record_compare(
                transcript,
                kind="rejected_chunk_at_or_below_minimum",
                pair_index=None,
                left=step_exact,
                operator="<=",
                right=static.minimum_step_magnitude,
                comparison=step_exact.compare(static.minimum_step_magnitude, budget),
            )
        else:
            at_or_below_minimum = False

        record = _proposal_record(
            index=proposal_index,
            signed_step=signed_step,
            terminal_below_minimum=terminal_below_minimum,
            disposition=disposition,
            certificate_passed=certificate_passed,
            actual_force_evaluations=actual_force_evaluations,
            normalized_error=normalized_error,
            budget=budget,
            transcript=transcript,
        )
        ledger_bytes = _record_diagnostic_bytes(record, resources, ledger_bytes)
        proposal_records.append(record)

        if disposition == "ACCEPTED":
            if trial is None or new_remaining is None:
                raise EncounterContractError("accepted proposal lacks a complete trial")
            if counters["accepted_substeps"] >= spec.maximum_accepted_substeps:
                raise EncounterStepLimitError("maximum_accepted_substeps exhausted")
            counters["accepted_substeps"] += 1
            consecutive_rejections = 0
            current_positions = np.array(
                trial.accepted_positions,
                dtype=np.float64,
                order="C",
                copy=True,
                subok=False,
            )
            current_velocities = np.array(
                trial.accepted_velocities,
                dtype=np.float64,
                order="C",
                copy=True,
                subok=False,
            )
            current_position_carry = np.array(
                trial.accepted_position_carry,
                dtype=np.float64,
                order="C",
                copy=True,
                subok=False,
            )
            current_velocity_carry = np.array(
                trial.accepted_velocity_carry,
                dtype=np.float64,
                order="C",
                copy=True,
                subok=False,
            )
            remaining = new_remaining
            accepted_steps.append(signed_step)
            proposed_magnitude = _accepted_proposal(
                current_magnitude=abs(signed_step),
                factor=_controller_factor(normalized_error or 0.0, spec),
                spec=spec,
            )
        else:
            counters["rejected_substeps"] += 1
            consecutive_rejections += 1
            category = {
                "CERTIFICATE_REJECTION": "certificate_rejections",
                "STAGE_GUARD_ABORT": "stage_guard_aborts",
                "DERIVATIVE_DOMAIN_ABORT": "derivative_domain_aborts",
                "CANDIDATE_DOMAIN_REJECTION": "candidate_domain_rejections",
                "ERROR_REJECTION": "error_rejections",
            }[disposition]
            counters[category] += 1
            if counters["rejected_substeps"] > spec.maximum_rejected_substeps:
                raise EncounterStepLimitError("maximum_rejected_substeps exhausted")
            if consecutive_rejections > spec.maximum_consecutive_rejections:
                raise EncounterStepLimitError(
                    "maximum_consecutive_rejections exhausted"
                )
            if terminal_below_minimum or at_or_below_minimum:
                raise EncounterStepLimitError(
                    "a rejected proposal cannot retry at or below minimum step"
                )
            if disposition == "ERROR_REJECTION" and normalized_error is not None:
                reduction_factor = _controller_factor(normalized_error, spec)
            else:
                reduction_factor = ENCOUNTER_DOMAIN_REJECTION_SCALE_FACTOR
            proposed_magnitude = _reduced_proposal(
                current_magnitude=abs(signed_step),
                factor=reduction_factor,
                minimum_magnitude=spec.minimum_step_magnitude,
            )

    if not accepted_steps:
        raise EncounterContractError("a completed encounter segment has no accepted step")
    if force_ledger is None or force_model_ids is None:
        raise EncounterContractError("a completed encounter segment has no force ledger")
    records = tuple(proposal_records)
    counts = _execution_counts(counters, static.initialization_record, records)
    if counts.force_evaluations != sum(
        record.actual_force_evaluations for record in records
    ):
        raise EncounterContractError("actual force-call accounting is inconsistent")
    if (
        counts.total_exact_operations != segment_usage.exact_operations
        or counts.total_rational_operations != segment_usage.rational_operations
        or counts.total_dyadic_operations != segment_usage.dyadic_operations
        or counts.total_gcd_iterations != segment_usage.gcd_iterations
        or counts.total_transcript_bytes != segment_usage.transcript_bytes
    ):
        raise EncounterContractError("exact-resource lane accounting is inconsistent")
    return _EncounterRun(
        final_positions=_readonly_copy(current_positions, np.dtype(np.float64)),
        final_velocities=_readonly_copy(current_velocities, np.dtype(np.float64)),
        accepted_signed_substeps=tuple(accepted_steps),
        initialization_record=static.initialization_record,
        proposal_ledger=records,
        force_model_ids=force_model_ids,
        force_ledger=force_ledger,
        counts=counts,
    )


def _pair_table_sha256(spec: AdaptiveEncounterSegmentSpec) -> str:
    return _domain_sha256(
        ENCOUNTER_PAIR_TABLE_CHECKSUM_DOMAIN,
        {
            "schema": _PAIR_TABLE_SCHEMA,
            "method_id": ADAPTIVE_ENCOUNTER_SEGMENT_METHOD_ID,
            "body_order": spec.body_order,
            "canonical_pairs": spec.canonical_pairs,
            "pair_certification_floors": spec.pair_certification_floors,
            "pair_position_atols": spec.pair_position_atols,
            "pair_position_rtol": spec.pair_position_rtol,
            "pair_velocity_atols": spec.pair_velocity_atols,
            "pair_velocity_rtol": spec.pair_velocity_rtol,
            "gm_centroid_position_atol": spec.gm_centroid_position_atol,
            "gm_centroid_position_rtol": spec.gm_centroid_position_rtol,
            "gm_centroid_velocity_atol": spec.gm_centroid_velocity_atol,
            "gm_centroid_velocity_rtol": spec.gm_centroid_velocity_rtol,
        },
    )


def _schedule_sha256(
    *,
    snapshot: StateSnapshot,
    plan: ForcePlan,
    spec: AdaptiveEncounterSegmentSpec,
    run: _EncounterRun,
    primary_counts: EncounterExecutionCounts,
    replay_counts: EncounterExecutionCounts,
    total_counts: EncounterExecutionCounts,
    pair_table_sha256: str,
) -> str:
    return _domain_sha256(
        ENCOUNTER_SCHEDULE_CHECKSUM_DOMAIN,
        {
            "schema": _SCHEDULE_SCHEMA,
            "method_id": ADAPTIVE_ENCOUNTER_SEGMENT_METHOD_ID,
            "tableau_id": ENCOUNTER_TABLEAU_ID,
            "rkf78_c_table": _C,
            "stage_epoch_policy": ENCOUNTER_STAGE_EPOCH_POLICY,
            "initial_epoch": spec.initial_epoch,
            "endpoint_epoch": spec.endpoint_epoch,
            "duration": spec.duration,
            "accepted_signed_substeps": run.accepted_signed_substeps,
            "initialization_record": run.initialization_record,
            "proposal_ledger": run.proposal_ledger,
            "force_model_ids": run.force_model_ids,
            "force_ledger": run.force_ledger,
            "snapshot_id": snapshot.snapshot_id,
            "snapshot_provenance": snapshot.provenance,
            "plan_id": plan.plan_id,
            "backend": plan.backend,
            "resource_spec": spec.exact_rational_resources,
            "primary_counts": primary_counts,
            "validation_replay_counts": replay_counts,
            "total_public_call_counts": total_counts,
            "pair_table_content_sha256": pair_table_sha256,
        },
    )


def _result_sha256(
    *,
    snapshot: StateSnapshot,
    plan: ForcePlan,
    spec: AdaptiveEncounterSegmentSpec,
    run: _EncounterRun,
    primary_counts: EncounterExecutionCounts,
    replay_counts: EncounterExecutionCounts,
    total_counts: EncounterExecutionCounts,
    pair_table_sha256: str,
    schedule_sha256: str,
) -> str:
    return _domain_sha256(
        ENCOUNTER_RESULT_CONTENT_CHECKSUM_DOMAIN,
        {
            "schema": _RESULT_SCHEMA,
            "snapshot_id": snapshot.snapshot_id,
            "plan_id": plan.plan_id,
            "backend_id": plan.backend.backend_id,
            "device": plan.backend.device,
            "dtype": plan.backend.dtype,
            "backend_spec": plan.backend,
            "method_id": ADAPTIVE_ENCOUNTER_SEGMENT_METHOD_ID,
            "tableau_id": ENCOUNTER_TABLEAU_ID,
            "stage_epoch_policy": ENCOUNTER_STAGE_EPOCH_POLICY,
            "initial_snapshot": snapshot,
            "force_plan": plan,
            "integration_spec": spec,
            "final_positions": run.final_positions,
            "final_velocities": run.final_velocities,
            "accepted_signed_substeps": run.accepted_signed_substeps,
            "initialization_record": run.initialization_record,
            "proposal_ledger": run.proposal_ledger,
            "force_model_ids": run.force_model_ids,
            "force_ledger": run.force_ledger,
            "primary_counts": primary_counts,
            "validation_replay_counts": replay_counts,
            "total_public_call_counts": total_counts,
            "pair_table_content_sha256": pair_table_sha256,
            "schedule_content_sha256": schedule_sha256,
            "controls": {
                "scope": ENCOUNTER_SEGMENT_SCOPE,
                "evidence_class": ENCOUNTER_EVIDENCE_CLASS,
                "adaptive": True,
                "dense_output": False,
                "event_detection": False,
                "collision_response": False,
                "globally_symplectic": False,
                "exactly_time_reversible": False,
                "registry_authorized": False,
                "qualification_authorized": False,
            },
        },
    )


def _readonly_owned_array(
    value: object,
    label: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> np.ndarray:
    array = _exact_array(value, label=label, dtype=dtype, shape=shape)
    if array.flags.writeable or not array.flags.owndata:
        raise EncounterContractError(f"{label} must be an owned read-only array")
    return array


def _dyadic_steps_equal_duration(steps: tuple[float, ...], duration: float) -> bool:
    values = (*steps, -duration)
    ratios = tuple(value.as_integer_ratio() for value in values)
    maximum_denominator = max(denominator for _, denominator in ratios)
    total = 0
    for numerator, denominator in ratios:
        if maximum_denominator % denominator != 0:
            return False
        total += numerator * (maximum_denominator // denominator)
    return total == 0


def _run_matches_result(run: _EncounterRun, result: EncounterSegmentResult) -> bool:
    return (
        np.array_equal(run.final_positions, result.final_positions)
        and np.array_equal(run.final_velocities, result.final_velocities)
        and run.final_positions.tobytes(order="C")
        == result.final_positions.tobytes(order="C")
        and run.final_velocities.tobytes(order="C")
        == result.final_velocities.tobytes(order="C")
        and _canonical_json(run.accepted_signed_substeps)
        == _canonical_json(result.accepted_signed_substeps)
        and _canonical_json(run.initialization_record)
        == _canonical_json(result.initialization_record)
        and _canonical_json(run.proposal_ledger)
        == _canonical_json(result.proposal_ledger)
        and _canonical_json(run.force_model_ids)
        == _canonical_json(result.force_model_ids)
        and _canonical_json(run.force_ledger) == _canonical_json(result.force_ledger)
    )


def _validate_force_ledger_schema(
    ledger: tuple[ForceLedgerEntry, ...],
    plan: ForcePlan,
    snapshot: StateSnapshot,
) -> None:
    if type(ledger) is not tuple or len(ledger) != 1:
        raise EncounterContractError("force ledger must be a one-entry exact tuple")
    entry = ledger[0]
    if type(entry) is not ForceLedgerEntry:
        raise EncounterContractError("force ledger entry has wrong exact type")
    if (
        type(entry.order) is not int
        or entry.order != 0
        or type(entry.model_id) is not str
        or entry.model_id != NewtonianPointMass.MODEL_ID
        or type(entry.role) is not str
        or entry.role != "NEWTONIAN_BASE"
        or type(entry.source_ids) is not tuple
        or type(entry.target_ids) is not tuple
        or entry.source_ids != snapshot.body_ids
        or entry.target_ids != snapshot.body_ids
        or any(type(value) is not str for value in entry.source_ids)
        or any(type(value) is not str for value in entry.target_ids)
        or type(entry.backend_spec) is not BackendSpec
        or entry.backend_spec != plan.backend
        or type(entry.tile_size) is not int
        or entry.tile_size != plan.backend.tile_size
        or type(entry.assumptions) is not tuple
        or any(type(value) is not str for value in entry.assumptions)
        or type(entry.determinism_scope) is not str
        or type(entry.evidence_class) is not str
        or entry.evidence_class != "MODEL_OUTPUT"
        or type(entry.registry_authorized) is not bool
        or entry.registry_authorized
        or type(entry.qualification_authorized) is not bool
        or entry.qualification_authorized
    ):
        raise EncounterContractError("force ledger entry schema is invalid")
    _validate_backend_schema(entry.backend_spec, "force ledger backend_spec")
    metadata = entry.state_metadata
    if type(metadata) is not StateMetadataBinding:
        raise EncounterContractError("force ledger metadata has wrong exact type")
    for descriptor in fields(StateMetadataBinding):
        value = getattr(metadata, descriptor.name)
        if descriptor.name == "epoch":
            if type(value) is not float or not math.isfinite(value):
                raise EncounterContractError("force ledger epoch must be finite float")
        elif descriptor.name == "body_ids":
            if (
                type(value) is not tuple
                or value != snapshot.body_ids
                or any(type(item) is not str for item in value)
            ):
                raise EncounterContractError("force ledger body_ids are invalid")
        elif type(value) is not str:
            raise EncounterContractError(
                f"force ledger metadata {descriptor.name} must be exact str"
            )
    if _float_bits(metadata.epoch) != _float_bits(snapshot.epoch):
        raise EncounterContractError("force ledger stage label differs from initial_epoch")


def _validate_resource_records(result: EncounterSegmentResult) -> None:
    resources = result.integration_spec.exact_rational_resources
    initialization = result.initialization_record
    if (
        initialization.exact_operations > resources.maximum_initialization_operations
        or initialization.gcd_iterations
        > resources.maximum_initialization_gcd_iterations
        or initialization.transcript_bytes
        > resources.maximum_initialization_transcript_bytes
        or initialization.maximum_integer_bits > resources.maximum_integer_bits
        or initialization.maximum_rational_exponent_magnitude
        > resources.maximum_rational_exponent_magnitude
    ):
        raise EncounterContractError("initialization record exceeds its resource caps")
    diagnostic_bytes = len(_canonical_json(initialization))
    if diagnostic_bytes > resources.maximum_witness_diagnostic_bytes_per_proposal:
        raise EncounterContractError("initialization diagnostic exceeds its custody cap")
    for record in result.proposal_ledger:
        if (
            record.exact_operations > resources.maximum_operations_per_proposal
            or record.gcd_iterations
            > resources.maximum_gcd_iterations_per_proposal
            or record.transcript_bytes
            > resources.maximum_witness_transcript_bytes_per_proposal
            or record.maximum_integer_bits > resources.maximum_integer_bits
            or record.maximum_rational_exponent_magnitude
            > resources.maximum_rational_exponent_magnitude
        ):
            raise EncounterContractError("proposal record exceeds its resource caps")
        record_bytes = len(_canonical_json(record))
        if record_bytes > resources.maximum_witness_diagnostic_bytes_per_proposal:
            raise EncounterContractError("proposal diagnostic exceeds its custody cap")
        diagnostic_bytes += record_bytes
        if diagnostic_bytes > resources.maximum_witness_ledger_bytes:
            raise EncounterContractError("retained witness ledger exceeds its custody cap")

    counts = result.primary_counts
    if (
        counts.initialization_exact_operations != initialization.exact_operations
        or counts.initialization_rational_operations
        != initialization.rational_operations
        or counts.initialization_dyadic_operations != initialization.dyadic_operations
        or counts.initialization_gcd_iterations != initialization.gcd_iterations
        or counts.initialization_transcript_bytes != initialization.transcript_bytes
        or counts.proposal_exact_operations
        != sum(record.exact_operations for record in result.proposal_ledger)
        or counts.proposal_rational_operations
        != sum(record.rational_operations for record in result.proposal_ledger)
        or counts.proposal_dyadic_operations
        != sum(record.dyadic_operations for record in result.proposal_ledger)
        or counts.proposal_gcd_iterations
        != sum(record.gcd_iterations for record in result.proposal_ledger)
        or counts.proposal_transcript_bytes
        != sum(record.transcript_bytes for record in result.proposal_ledger)
    ):
        raise EncounterContractError("resource records and primary counts disagree")


def _validate_result(result: EncounterSegmentResult) -> None:
    if type(result) is not EncounterSegmentResult:
        raise EncounterContractError("result must be an exact EncounterSegmentResult")
    fixed = {
        "method_id": ADAPTIVE_ENCOUNTER_SEGMENT_METHOD_ID,
        "tableau_id": ENCOUNTER_TABLEAU_ID,
        "stage_count": ENCOUNTER_STAGE_COUNT,
        "accepted_order": ENCOUNTER_ACCEPTED_ORDER,
        "embedded_order": ENCOUNTER_EMBEDDED_ORDER,
        "defect_orientation": ENCOUNTER_DEFECT_ORIENTATION,
        "stage_epoch_policy": ENCOUNTER_STAGE_EPOCH_POLICY,
        "backend_scope": ENCOUNTER_BACKEND_SCOPE,
        "force_plan_scope": ENCOUNTER_FORCE_PLAN_SCOPE,
        "error_norm": ENCOUNTER_PAIR_ERROR_NORM,
        "certificate_claim_scope": ENCOUNTER_CERTIFICATE_CLAIM_SCOPE,
        "certificate_nonclaims": ENCOUNTER_CERTIFICATE_NONCLAIMS,
        "force_evaluation_accounting": ENCOUNTER_FORCE_EVALUATION_ACCOUNTING,
        "proposal_accounting": ENCOUNTER_PROPOSAL_ACCOUNTING,
        "public_execution_accounting_scope": ENCOUNTER_PUBLIC_EXECUTION_ACCOUNTING_SCOPE,
        "validation_replay_policy": ENCOUNTER_VALIDATION_REPLAY_POLICY,
        "validation_replay_count": ENCOUNTER_VALIDATION_REPLAY_COUNT,
        "checksum_binding_policy": ENCOUNTER_CHECKSUM_BINDING_POLICY,
        "pair_table_checksum_algorithm": ENCOUNTER_PAIR_TABLE_CHECKSUM_ALGORITHM,
        "pair_table_checksum_domain": ENCOUNTER_PAIR_TABLE_CHECKSUM_DOMAIN,
        "initialization_checksum_algorithm": ENCOUNTER_INITIALIZATION_CHECKSUM_ALGORITHM,
        "initialization_checksum_domain": ENCOUNTER_INITIALIZATION_CHECKSUM_DOMAIN,
        "exact_witness_checksum_algorithm": ENCOUNTER_EXACT_WITNESS_CHECKSUM_ALGORITHM,
        "exact_witness_checksum_domain": ENCOUNTER_EXACT_WITNESS_CHECKSUM_DOMAIN,
        "schedule_checksum_algorithm": ENCOUNTER_SCHEDULE_CHECKSUM_ALGORITHM,
        "schedule_checksum_domain": ENCOUNTER_SCHEDULE_CHECKSUM_DOMAIN,
        "result_content_checksum_algorithm": ENCOUNTER_RESULT_CONTENT_CHECKSUM_ALGORITHM,
        "result_content_checksum_domain": ENCOUNTER_RESULT_CONTENT_CHECKSUM_DOMAIN,
        "scope": ENCOUNTER_SEGMENT_SCOPE,
        "evidence_class": ENCOUNTER_EVIDENCE_CLASS,
        "adaptive": True,
        "dense_output": False,
        "event_detection": False,
        "collision_response": False,
        "globally_symplectic": False,
        "exactly_time_reversible": False,
        "registry_authorized": False,
        "qualification_authorized": False,
    }
    for name, expected in fixed.items():
        value = getattr(result, name)
        if type(value) is not type(expected) or value != expected:
            raise EncounterContractError(f"result {name} differs from its fixed value")
    for name in ("snapshot_id", "plan_id", "backend_id", "device", "dtype"):
        if type(getattr(result, name)) is not str:
            raise EncounterContractError(f"result {name} must be an exact string")
    for name in (
        "pair_table_content_sha256",
        "schedule_content_sha256",
        "result_content_sha256",
    ):
        _sha256_hex(getattr(result, name), name)
    if type(result.backend_spec) is not BackendSpec:
        raise EncounterContractError("result backend_spec must be exact BackendSpec")
    _validate_backend_schema(result.backend_spec, "result.backend_spec")
    if (
        type(result.initial_snapshot) is not StateSnapshot
        or type(result.force_plan) is not ForcePlan
        or type(result.integration_spec) is not AdaptiveEncounterSegmentSpec
    ):
        raise EncounterContractError("result retained request types are invalid")
    if (
        result.snapshot_id != result.initial_snapshot.snapshot_id
        or result.plan_id != result.force_plan.plan_id
        or result.backend_id != "numpy"
        or result.device != "cpu"
        or result.dtype != "float64"
        or result.backend_spec != result.force_plan.backend
    ):
        raise EncounterContractError("result identity/backend binding is inconsistent")
    _preflight_request(
        result.initial_snapshot, result.force_plan, result.integration_spec
    )
    result.backend_spec.__post_init__()
    result.initial_snapshot.provenance.__post_init__()
    result.force_plan.backend.__post_init__()
    model = result.force_plan.models[0]
    model.__post_init__()
    for metadata in model.parameter_metadata:
        metadata.__post_init__()
        metadata.provenance.__post_init__()
    if (
        type(result.accepted_signed_substeps) is not tuple
        or not result.accepted_signed_substeps
        or len(result.accepted_signed_substeps)
        > result.integration_spec.maximum_accepted_substeps
    ):
        raise EncounterContractError("accepted-step tuple has invalid bounded length")
    if (
        type(result.proposal_ledger) is not tuple
        or len(result.proposal_ledger)
        > result.integration_spec.maximum_substep_proposals
    ):
        raise EncounterContractError("proposal ledger has invalid bounded length")
    if any(type(record) is not EncounterProposalRecord for record in result.proposal_ledger):
        raise EncounterContractError("proposal_ledger has wrong exact entry type")
    if type(result.initialization_record) is not EncounterInitializationRecord:
        raise EncounterContractError("initialization_record has wrong exact type")
    result.initialization_record.__post_init__()
    for record in result.proposal_ledger:
        record.__post_init__()
    if any(
        type(value) is not EncounterExecutionCounts
        for value in (
            result.primary_counts,
            result.validation_replay_counts,
            result.total_public_call_counts,
        )
    ):
        raise EncounterContractError("execution-count records have wrong exact type")
    for counts in (
        result.primary_counts,
        result.validation_replay_counts,
        result.total_public_call_counts,
    ):
        counts.__post_init__()
    _validate_resource_records(result)
    body_count = len(result.integration_spec.body_order)
    state_shape = (body_count, 3)
    final_positions = _readonly_owned_array(
        result.final_positions,
        "final_positions",
        state_shape,
        np.dtype(np.float64),
    )
    final_velocities = _readonly_owned_array(
        result.final_velocities,
        "final_velocities",
        state_shape,
        np.dtype(np.float64),
    )
    retained_arrays = (
        _readonly_owned_array(
            result.initial_snapshot.positions,
            "initial positions",
            state_shape,
            np.dtype(np.float64),
        ),
        _readonly_owned_array(
            result.initial_snapshot.velocities,
            "initial velocities",
            state_shape,
            np.dtype(np.float64),
        ),
        _readonly_owned_array(
            result.initial_snapshot.gravitational_parameters,
            "initial GM",
            (body_count,),
            np.dtype(np.float64),
        ),
        _readonly_owned_array(
            result.initial_snapshot.masses,
            "initial masses",
            (body_count,),
            np.dtype(np.float64),
        ),
        _readonly_owned_array(
            result.initial_snapshot.radii,
            "initial radii",
            (body_count,),
            np.dtype(np.float64),
        ),
        _readonly_owned_array(
            result.initial_snapshot.massive,
            "initial massive",
            (body_count,),
            np.dtype(np.bool_),
        ),
        final_positions,
        final_velocities,
    )
    for left in range(len(retained_arrays) - 1):
        for right in range(left + 1, len(retained_arrays)):
            if np.shares_memory(retained_arrays[left], retained_arrays[right]):
                raise EncounterContractError("retained result arrays must not alias")
    for step in result.accepted_signed_substeps:
        if (
            type(step) is not float
            or not math.isfinite(step)
            or step == 0.0
            or math.copysign(1.0, step) != math.copysign(1.0, result.duration)
        ):
            raise EncounterContractError("accepted signed step is invalid")
    if not _dyadic_steps_equal_duration(
        result.accepted_signed_substeps, result.duration
    ):
        raise EncounterContractError(
            "accepted signed steps do not sum exactly to retained duration"
        )
    if len(result.proposal_ledger) != result.primary_counts.substep_proposals:
        raise EncounterContractError("proposal ledger length is inconsistent")
    if (
        type(result.force_model_ids) is not tuple
        or result.force_model_ids != (NewtonianPointMass.MODEL_ID,)
        or any(type(value) is not str for value in result.force_model_ids)
        or type(result.force_ledger) is not tuple
    ):
        raise EncounterContractError("retained force identity/ledger is invalid")
    _validate_force_ledger_schema(
        result.force_ledger, result.force_plan, result.initial_snapshot
    )
    for expected_index, record in enumerate(result.proposal_ledger, start=1):
        if record.index != expected_index:
            raise EncounterContractError("proposal ledger indices are not canonical")
        if record.disposition == "CERTIFICATE_REJECTION":
            if record.certificate_passed or record.actual_force_evaluations != 0:
                raise EncounterContractError("certificate rejection accounting is invalid")
        else:
            if not record.certificate_passed:
                raise EncounterContractError("post-certificate disposition lacks certificate")
        if record.disposition == "STAGE_GUARD_ABORT" and not (
            0 <= record.actual_force_evaluations <= 12
        ):
            raise EncounterContractError("stage-guard force count is invalid")
        if record.disposition == "DERIVATIVE_DOMAIN_ABORT" and not (
            1 <= record.actual_force_evaluations <= 13
        ):
            raise EncounterContractError("derivative-abort force count is invalid")
        if record.disposition in {
            "CANDIDATE_DOMAIN_REJECTION",
            "ERROR_REJECTION",
            "ACCEPTED",
        } and record.actual_force_evaluations != ENCOUNTER_STAGE_COUNT:
            raise EncounterContractError("completed-attempt force count is invalid")
        if record.disposition in {"ERROR_REJECTION", "ACCEPTED"}:
            if record.normalized_error is None:
                raise EncounterContractError("error-tested proposal lacks its error value")
        elif record.normalized_error is not None:
            raise EncounterContractError("non-error-tested proposal retains an error value")
    if result.primary_counts.accepted_substeps != len(
        result.accepted_signed_substeps
    ):
        raise EncounterContractError("accepted-step count is inconsistent")
    if result.primary_counts.force_evaluations != sum(
        record.actual_force_evaluations for record in result.proposal_ledger
    ):
        raise EncounterContractError("primary force count is inconsistent")
    dispositions = {
        name: sum(record.disposition == value for record in result.proposal_ledger)
        for name, value in (
            ("certificate_rejections", "CERTIFICATE_REJECTION"),
            ("stage_guard_aborts", "STAGE_GUARD_ABORT"),
            ("derivative_domain_aborts", "DERIVATIVE_DOMAIN_ABORT"),
            ("candidate_domain_rejections", "CANDIDATE_DOMAIN_REJECTION"),
            ("error_rejections", "ERROR_REJECTION"),
            ("accepted_substeps", "ACCEPTED"),
        )
    }
    if any(
        getattr(result.primary_counts, name) != count
        for name, count in dispositions.items()
    ):
        raise EncounterContractError("proposal dispositions and primary counts disagree")
    if result.total_public_call_counts != _sum_counts(
        result.primary_counts, result.validation_replay_counts
    ):
        raise EncounterContractError("public primary-plus-replay accounting is invalid")
    if result.validation_replay_counts != result.primary_counts:
        raise EncounterContractError("deterministic replay accounting differs from primary")
    resources = result.integration_spec.exact_rational_resources
    for counts in (result.primary_counts, result.validation_replay_counts):
        if (
            counts.total_exact_operations > resources.maximum_operations_per_segment
            or counts.total_gcd_iterations > resources.maximum_gcd_iterations_per_segment
            or counts.total_transcript_bytes
            > resources.maximum_witness_transcript_bytes_per_segment
            or counts.force_evaluations > result.integration_spec.maximum_force_evaluations
        ):
            raise EncounterContractError("one execution lane exceeds a resource cap")
    expected_pair = _pair_table_sha256(result.integration_spec)
    if result.pair_table_content_sha256 != expected_pair:
        raise EncounterContractError("pair-table checksum mismatch")
    retained_run = _EncounterRun(
        final_positions=result.final_positions,
        final_velocities=result.final_velocities,
        accepted_signed_substeps=result.accepted_signed_substeps,
        initialization_record=result.initialization_record,
        proposal_ledger=result.proposal_ledger,
        force_model_ids=result.force_model_ids,
        force_ledger=result.force_ledger,
        counts=result.primary_counts,
    )
    expected_schedule = _schedule_sha256(
        snapshot=result.initial_snapshot,
        plan=result.force_plan,
        spec=result.integration_spec,
        run=retained_run,
        primary_counts=result.primary_counts,
        replay_counts=result.validation_replay_counts,
        total_counts=result.total_public_call_counts,
        pair_table_sha256=expected_pair,
    )
    if result.schedule_content_sha256 != expected_schedule:
        raise EncounterContractError("schedule checksum mismatch")
    expected_result = _result_sha256(
        snapshot=result.initial_snapshot,
        plan=result.force_plan,
        spec=result.integration_spec,
        run=retained_run,
        primary_counts=result.primary_counts,
        replay_counts=result.validation_replay_counts,
        total_counts=result.total_public_call_counts,
        pair_table_sha256=expected_pair,
        schedule_sha256=expected_schedule,
    )
    if result.result_content_sha256 != expected_result:
        raise EncounterContractError("result content checksum mismatch")

    replay = _execute(
        result.initial_snapshot, result.force_plan, result.integration_spec
    )
    if replay.counts != result.validation_replay_counts or not _run_matches_result(
        replay, result
    ):
        raise EncounterContractError("mandatory semantic replay does not match result")


def integrate_encounter_segment(
    snapshot: StateSnapshot,
    plan: ForcePlan,
    spec: AdaptiveEncounterSegmentSpec,
) -> EncounterSegmentResult:
    """Integrate one exact-duration guarded full-Cartesian encounter segment."""

    validated = _preflight_request(snapshot, plan, spec)
    initial_snapshot, force_plan, integration_spec = _owned_request(
        snapshot, plan, spec, validated
    )
    primary = _execute(initial_snapshot, force_plan, integration_spec)
    primary_counts = primary.counts
    replay_counts = primary_counts
    total_counts = _sum_counts(primary_counts, replay_counts)
    pair_table_sha256 = _pair_table_sha256(integration_spec)
    schedule_sha256 = _schedule_sha256(
        snapshot=initial_snapshot,
        plan=force_plan,
        spec=integration_spec,
        run=primary,
        primary_counts=primary_counts,
        replay_counts=replay_counts,
        total_counts=total_counts,
        pair_table_sha256=pair_table_sha256,
    )
    result_sha256 = _result_sha256(
        snapshot=initial_snapshot,
        plan=force_plan,
        spec=integration_spec,
        run=primary,
        primary_counts=primary_counts,
        replay_counts=replay_counts,
        total_counts=total_counts,
        pair_table_sha256=pair_table_sha256,
        schedule_sha256=schedule_sha256,
    )
    return EncounterSegmentResult(
        snapshot_id=initial_snapshot.snapshot_id,
        plan_id=force_plan.plan_id,
        backend_id="numpy",
        device="cpu",
        dtype="float64",
        backend_spec=force_plan.backend,
        initial_snapshot=initial_snapshot,
        force_plan=force_plan,
        integration_spec=integration_spec,
        final_positions=_readonly_copy(primary.final_positions, np.dtype(np.float64)),
        final_velocities=_readonly_copy(primary.final_velocities, np.dtype(np.float64)),
        accepted_signed_substeps=primary.accepted_signed_substeps,
        initialization_record=primary.initialization_record,
        proposal_ledger=primary.proposal_ledger,
        force_model_ids=primary.force_model_ids,
        force_ledger=primary.force_ledger,
        primary_counts=primary_counts,
        validation_replay_counts=replay_counts,
        total_public_call_counts=total_counts,
        pair_table_content_sha256=pair_table_sha256,
        schedule_content_sha256=schedule_sha256,
        result_content_sha256=result_sha256,
    )


__all__ = [
    "ENCOUNTER_SEGMENT_SCOPE",
    "EncounterContractError",
    "EncounterDomainError",
    "EncounterError",
    "EncounterExecutionCounts",
    "EncounterInitializationRecord",
    "EncounterProposalRecord",
    "EncounterResourceError",
    "EncounterSegmentResult",
    "EncounterStepLimitError",
    "integrate_encounter_segment",
]
