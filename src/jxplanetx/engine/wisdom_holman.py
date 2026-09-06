"""Public ordered-Jacobi Wisdom--Holman trajectory runtime and result.

The public integration boundary retains synchronized Cartesian checkpoints,
coordinate/result custody, and explicit execution accounting.  Ordered-Jacobi
transforms and the elliptic universal-variable solver remain private numerical
building blocks of that runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Any

import numpy as np

from .backends import resolve_backend
from .contracts import BackendSpec, ForcePlan, NewtonianPointMass, StateSnapshot
from .evaluator import (
    EvaluationError,
    ForceLedgerEntry,
    _validate_dependencies,
    _validate_model_sequence,
    _validate_parameter_contracts,
    _validate_state,
    evaluate_force_plan,
)
from .encounter_contracts import ENCOUNTER_HARD_MAXIMUM_BODY_COUNT
from .forces import ForceError
from .hybrid_contracts import (
    HYBRID_ALL_FAR_REASON,
    HYBRID_DYNAMIC_SWITCH_REASONS,
    HYBRID_KEPLER_PROBE_FAILURE_STATUS_BY_REASON,
    HybridFarProbeDecision,
    HybridFarProbeWork,
    HybridKeplerProbeRecord,
)
from .symplectic import (
    KDK_NEWTONIAN_LEDGER_ASSUMPTIONS,
    _readonly_snapshot_copy,
    _validate_backend_schema,
    _validate_checkpoint_schema,
    _validate_force_plan_schema,
    _validate_ledger_schema,
    _validate_parameter_metadata_schema,
    _validate_snapshot_schema,
)
from .trajectory import (
    TrajectoryCheckpoint,
    TrajectoryContractError,
    TrajectoryDomainError,
    TrajectoryStepLimitError,
)
from .wisdom_holman_contracts import (
    FIXED_STEP_WISDOM_HOLMAN_METHOD_ID,
    FixedStepWisdomHolmanSpec,
    UniversalKeplerSolverSpec,
    WH_ALLOWED_AXES,
    WH_ALLOWED_TIME_SCALES,
    WH_BACKEND_SCOPE,
    WH_BARYCENTER_ROUNDOFF_FACTOR,
    WH_CANONICAL_MOMENTUM_CONVENTION,
    WH_CANONICAL_WEIGHT_SOURCE,
    WH_CHECKPOINT_POLICY,
    WH_COMPOSITION,
    WH_ENCOUNTER_GUARD,
    WH_ENCOUNTER_ROUNDOFF_FACTOR,
    WH_EVIDENCE_CLASS,
    WH_FORCE_EVALUATION_ACCOUNTING,
    WH_FORCE_PLAN_SCOPE,
    WH_GUARD_CADENCE_POLICY,
    WH_HAMILTONIAN_SPLIT_POLICY,
    WH_HIERARCHY_GUARD,
    WH_INTERACTION_FORCE_POLICY,
    WH_INTERACTION_KICK_COEFFICIENTS,
    WH_INTERACTION_RATIO_POLICY,
    WH_JACOBI_BINDING_CHECKSUM_ALGORITHM,
    WH_JACOBI_BINDING_CHECKSUM_DOMAIN,
    WH_JACOBI_COORDINATE_SYSTEM,
    WH_JACOBI_MATRIX_POLICY,
    WH_JACOBI_SYMBOL_POLICY,
    WH_JACOBI_TRANSFORM_POLICY,
    WH_KEPLER_ANGULAR_MOMENTUM_ROUNDOFF_FACTOR,
    WH_KEPLER_BRACKET_ULP_FACTOR,
    WH_KEPLER_ENERGY_ROUNDOFF_FACTOR,
    WH_KEPLER_LAGRANGE_IDENTITY_ROUNDOFF_FACTOR,
    WH_KEPLER_MAXIMUM_BRACKET_EXPANSIONS,
    WH_KEPLER_MAXIMUM_ITERATIONS,
    WH_KEPLER_RESIDUAL_ROUNDOFF_FACTOR,
    WH_KEPLER_RESIDUAL_ULP_FACTOR,
    WH_KEPLER_SERIES_SWITCH_ABS_ARGUMENT,
    WH_KEPLER_SERIES_TERM_COUNT,
    WH_MAXIMUM_JACOBI_ECCENTRICITY,
    WH_MAXIMUM_INTERACTION_TO_KEPLER_FORCE_RATIO,
    WH_MAXIMUM_ORBIT_STEP_FRACTION,
    WH_MAXIMUM_PERIAPSE_STEP_FRACTION,
    WH_MAXIMUM_TOTAL_SECONDARY_TO_PRIMARY_GM_RATIO,
    WH_METHOD_CLASS,
    WH_MINIMUM_MUTUAL_HILL_SEPARATION_MULTIPLE,
    WH_MUTUAL_HILL_POLICY,
    WH_OSCULATING_JACOBI_POLICY,
    WH_PHYSICAL_MASS_POLICY,
    WH_PRINCIPAL_ORDER,
    WH_PUBLIC_EXECUTION_ACCOUNTING_SCOPE,
    WH_REQUIRED_FRAME,
    WH_REQUIRED_ORIGIN,
    WH_RESULT_CONTENT_CHECKSUM_ALGORITHM,
    WH_RESULT_CONTENT_CHECKSUM_DOMAIN,
    WH_SCHEDULE_CHECKSUM_ALGORITHM,
    WH_SCHEDULE_CHECKSUM_DOMAIN,
    WH_STEP_ENVELOPE_POLICY,
    WH_STEP_REPRESENTATION,
    WH_TIME_SEMANTICS,
    WH_VALIDATION_REPLAY_COUNT,
    WH_VALIDATION_REPLAY_POLICY,
)


_BINDING_SCHEMA = "jxplanetx.wisdom-holman-jacobi-binding.v1"
_FLOAT64_EPSILON = float(np.finfo(np.float64).eps)
WH_TRAJECTORY_SCOPE = "FIXED_STEP_WISDOM_HOLMAN_JACOBI_TRAJECTORY_INTEGRATION"


class _FiniteFarEnvelopeExit(TrajectoryDomainError):
    """Typed finite dynamic guard exit; never classify it by message text."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class _BodyNumericalFailure(TrajectoryDomainError):
    """Typed body-local numerical failure with canonical trigger indices."""

    def __init__(self, message: str, body_indices: tuple[int, ...]) -> None:
        super().__init__(message)
        self.body_indices = body_indices


class _InteractionAssemblyNumericalFailure(TrajectoryDomainError):
    """Typed incomplete interaction-assembly numerical failure."""

    def __init__(self, message: str, body_indices: tuple[int, ...] = ()) -> None:
        super().__init__(message)
        self.body_indices = body_indices


class _TranslationForceResidualFailure(TrajectoryDomainError):
    """Typed completed-assembly translation residual failure."""


class _KeplerSolverPostconditionFailure(TrajectoryDomainError):
    """Typed solver/postcondition failure retaining the frozen public base type."""


def _exact_body_ids(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not tuple or len(value) < 2:
        raise TrajectoryContractError(
            f"{label} must be an exact tuple containing at least two body IDs"
        )
    if any(
        type(body_id) is not str
        or not body_id
        or body_id.strip() != body_id
        for body_id in value
    ):
        raise TrajectoryContractError(
            f"{label} entries must be nonempty trimmed built-in strings"
        )
    if len(set(value)) != len(value):
        raise TrajectoryContractError(f"{label} must not repeat a body ID")
    return value


def _exact_text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise TrajectoryContractError(f"{label} must be a nonempty trimmed string")
    return value


def _validate_float64_array(
    value: object,
    label: str,
    shape: tuple[int, ...],
    *,
    owned_readonly: bool,
) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TrajectoryContractError(f"{label} must be an exact NumPy ndarray")
    if value.dtype != np.dtype(np.float64) or value.shape != shape:
        raise TrajectoryContractError(
            f"{label} must have dtype float64 and shape {shape!r}"
        )
    if not bool(np.all(np.isfinite(value))):
        raise TrajectoryDomainError(f"{label} must contain only finite values")
    if owned_readonly and (not value.flags.owndata or value.flags.writeable):
        raise TrajectoryContractError(
            f"{label} must own its memory and be read-only"
        )
    return value


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    copied = np.array(value, dtype=np.float64, copy=True, order="C", subok=False)
    copied.setflags(write=False)
    return copied


def _array_hex(value: np.ndarray) -> list[Any]:
    if value.ndim == 1:
        return [float(item).hex() for item in value]
    return [[float(item).hex() for item in row] for row in value]


def _binding_content_sha256(
    *,
    body_ids: tuple[str, ...],
    gravitational_parameters: np.ndarray,
    cumulative_gravitational_parameters: np.ndarray,
    canonical_inertias: np.ndarray,
    jacobi_from_cartesian_matrix: np.ndarray,
    cartesian_from_jacobi_matrix: np.ndarray,
) -> str:
    payload = {
        "schema": _BINDING_SCHEMA,
        "body_ids": list(body_ids),
        "primary_body_id": body_ids[0],
        "coordinate_system": WH_JACOBI_COORDINATE_SYSTEM,
        "canonical_weight_source": WH_CANONICAL_WEIGHT_SOURCE,
        "canonical_momentum_convention": WH_CANONICAL_MOMENTUM_CONVENTION,
        "physical_mass_policy": WH_PHYSICAL_MASS_POLICY,
        "symbol_policy": WH_JACOBI_SYMBOL_POLICY,
        "transform_policy": WH_JACOBI_TRANSFORM_POLICY,
        "matrix_policy": WH_JACOBI_MATRIX_POLICY,
        "gravitational_parameters_hex": _array_hex(gravitational_parameters),
        "cumulative_gravitational_parameters_hex": _array_hex(
            cumulative_gravitational_parameters
        ),
        "canonical_inertias_hex": _array_hex(canonical_inertias),
        "jacobi_from_cartesian_matrix_hex": _array_hex(
            jacobi_from_cartesian_matrix
        ),
        "cartesian_from_jacobi_matrix_hex": _array_hex(
            cartesian_from_jacobi_matrix
        ),
    }
    serialized = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    preimage = (
        WH_JACOBI_BINDING_CHECKSUM_DOMAIN.encode("utf-8")
        + b"\x00"
        + serialized
    )
    return hashlib.sha256(preimage).hexdigest()


def _derive_jacobi_arrays(
    gravitational_parameters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    body_count = gravitational_parameters.shape[0]
    cumulative = np.empty(body_count, dtype=np.float64)
    cumulative[0] = gravitational_parameters[0]
    for index in range(1, body_count):
        cumulative[index] = np.float64(
            cumulative[index - 1] + gravitational_parameters[index]
        )
    if not bool(np.all(np.isfinite(cumulative))) or bool(
        np.any(cumulative <= np.float64(0.0))
    ):
        raise TrajectoryDomainError(
            "ordered-Jacobi cumulative gravitational parameters must remain finite and positive"
        )
    if any(
        not float(cumulative[index]) > float(cumulative[index - 1])
        for index in range(1, body_count)
    ):
        raise TrajectoryDomainError(
            "each positive GM must make the ordered-Jacobi prefix sum advance"
        )

    inertias = np.empty(body_count, dtype=np.float64)
    inertias[0] = cumulative[-1]
    for index in range(1, body_count):
        product = np.float64(
            gravitational_parameters[index] * cumulative[index - 1]
        )
        inertias[index] = np.float64(product / cumulative[index])
    if not bool(np.all(np.isfinite(inertias))) or bool(
        np.any(inertias <= np.float64(0.0))
    ):
        raise TrajectoryDomainError(
            "ordered-Jacobi canonical inertias must remain finite and positive"
        )

    jacobi_from_cartesian = np.zeros(
        (body_count, body_count), dtype=np.float64
    )
    total = cumulative[-1]
    for column in range(body_count):
        jacobi_from_cartesian[0, column] = np.float64(
            gravitational_parameters[column] / total
        )
    for row in range(1, body_count):
        prefix = cumulative[row - 1]
        for column in range(row):
            jacobi_from_cartesian[row, column] = np.float64(
                -gravitational_parameters[column] / prefix
            )
        jacobi_from_cartesian[row, row] = np.float64(1.0)

    cartesian_from_jacobi = np.zeros(
        (body_count, body_count), dtype=np.float64
    )
    for row in range(body_count):
        cartesian_from_jacobi[row, 0] = np.float64(1.0)
    for column in range(1, body_count):
        coefficient = np.float64(
            -gravitational_parameters[column] / cumulative[column]
        )
        for row in range(column):
            cartesian_from_jacobi[row, column] = coefficient
        cartesian_from_jacobi[column, column] = np.float64(
            cumulative[column - 1] / cumulative[column]
        )
    return cumulative, inertias, jacobi_from_cartesian, cartesian_from_jacobi


@dataclass(frozen=True, eq=False)
class JacobiCoordinateBinding:
    """Retained exact coefficients for one immutable ordered body roster."""

    body_ids: tuple[str, ...]
    gravitational_parameters: np.ndarray
    cumulative_gravitational_parameters: np.ndarray
    canonical_inertias: np.ndarray
    jacobi_from_cartesian_matrix: np.ndarray
    cartesian_from_jacobi_matrix: np.ndarray
    binding_content_sha256: str
    coordinate_system: str = WH_JACOBI_COORDINATE_SYSTEM
    canonical_weight_source: str = WH_CANONICAL_WEIGHT_SOURCE
    canonical_momentum_convention: str = WH_CANONICAL_MOMENTUM_CONVENTION
    physical_mass_policy: str = WH_PHYSICAL_MASS_POLICY
    symbol_policy: str = WH_JACOBI_SYMBOL_POLICY
    transform_policy: str = WH_JACOBI_TRANSFORM_POLICY
    matrix_policy: str = WH_JACOBI_MATRIX_POLICY
    checksum_algorithm: str = WH_JACOBI_BINDING_CHECKSUM_ALGORITHM
    checksum_domain: str = WH_JACOBI_BINDING_CHECKSUM_DOMAIN
    evidence_class: str = "MODEL_OUTPUT"
    registry_authorized: bool = False
    qualification_authorized: bool = False

    def __post_init__(self) -> None:
        body_ids = _exact_body_ids(self.body_ids, "Jacobi binding body_ids")
        body_count = len(body_ids)
        arrays = (
            _validate_float64_array(
                self.gravitational_parameters,
                "Jacobi binding gravitational_parameters",
                (body_count,),
                owned_readonly=True,
            ),
            _validate_float64_array(
                self.cumulative_gravitational_parameters,
                "Jacobi binding cumulative_gravitational_parameters",
                (body_count,),
                owned_readonly=True,
            ),
            _validate_float64_array(
                self.canonical_inertias,
                "Jacobi binding canonical_inertias",
                (body_count,),
                owned_readonly=True,
            ),
            _validate_float64_array(
                self.jacobi_from_cartesian_matrix,
                "Jacobi binding jacobi_from_cartesian_matrix",
                (body_count, body_count),
                owned_readonly=True,
            ),
            _validate_float64_array(
                self.cartesian_from_jacobi_matrix,
                "Jacobi binding cartesian_from_jacobi_matrix",
                (body_count, body_count),
                owned_readonly=True,
            ),
        )
        if bool(np.any(arrays[0] <= np.float64(0.0))):
            raise TrajectoryDomainError(
                "Jacobi binding gravitational parameters must be strictly positive"
            )
        for left in range(len(arrays) - 1):
            for right in range(left + 1, len(arrays)):
                if np.shares_memory(arrays[left], arrays[right]):
                    raise TrajectoryContractError(
                        "Jacobi binding retained arrays must not share memory"
                    )

        expected = _derive_jacobi_arrays(arrays[0])
        for label, actual, derived in (
            (
                "cumulative_gravitational_parameters",
                arrays[1],
                expected[0],
            ),
            ("canonical_inertias", arrays[2], expected[1]),
            ("jacobi_from_cartesian_matrix", arrays[3], expected[2]),
            ("cartesian_from_jacobi_matrix", arrays[4], expected[3]),
        ):
            if actual.tobytes(order="C") != derived.tobytes(order="C"):
                raise TrajectoryContractError(
                    f"Jacobi binding {label} differs from the analytic ordered transform"
                )

        exact = {
            "coordinate_system": WH_JACOBI_COORDINATE_SYSTEM,
            "canonical_weight_source": WH_CANONICAL_WEIGHT_SOURCE,
            "canonical_momentum_convention": WH_CANONICAL_MOMENTUM_CONVENTION,
            "physical_mass_policy": WH_PHYSICAL_MASS_POLICY,
            "symbol_policy": WH_JACOBI_SYMBOL_POLICY,
            "transform_policy": WH_JACOBI_TRANSFORM_POLICY,
            "matrix_policy": WH_JACOBI_MATRIX_POLICY,
            "checksum_algorithm": WH_JACOBI_BINDING_CHECKSUM_ALGORITHM,
            "checksum_domain": WH_JACOBI_BINDING_CHECKSUM_DOMAIN,
            "evidence_class": "MODEL_OUTPUT",
            "registry_authorized": False,
            "qualification_authorized": False,
        }
        for name, expected_value in exact.items():
            value = getattr(self, name)
            if type(value) is not type(expected_value) or value != expected_value:
                raise TrajectoryContractError(
                    f"Jacobi binding {name} must equal {expected_value!r}"
                )
        if (
            type(self.binding_content_sha256) is not str
            or len(self.binding_content_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.binding_content_sha256
            )
        ):
            raise TrajectoryContractError(
                "binding_content_sha256 must be lowercase SHA-256 hexadecimal"
            )
        expected_checksum = _binding_content_sha256(
            body_ids=body_ids,
            gravitational_parameters=arrays[0],
            cumulative_gravitational_parameters=arrays[1],
            canonical_inertias=arrays[2],
            jacobi_from_cartesian_matrix=arrays[3],
            cartesian_from_jacobi_matrix=arrays[4],
        )
        if self.binding_content_sha256 != expected_checksum:
            raise TrajectoryContractError(
                "Jacobi binding checksum does not match retained coordinate content"
            )

    @property
    def primary_body_id(self) -> str:
        return self.body_ids[0]


def _build_jacobi_coordinate_binding(
    body_ids: tuple[str, ...], gravitational_parameters: np.ndarray
) -> JacobiCoordinateBinding:
    checked_ids = _exact_body_ids(body_ids, "body_ids")
    checked_gm = _validate_float64_array(
        gravitational_parameters,
        "gravitational_parameters",
        (len(checked_ids),),
        owned_readonly=False,
    )
    if bool(np.any(checked_gm <= np.float64(0.0))):
        raise TrajectoryDomainError(
            "ordered-Jacobi gravitational parameters must be strictly positive"
        )
    gm = _readonly_copy(checked_gm)
    cumulative, inertias, jacobi_matrix, cartesian_matrix = (
        _derive_jacobi_arrays(gm)
    )
    cumulative = _readonly_copy(cumulative)
    inertias = _readonly_copy(inertias)
    jacobi_matrix = _readonly_copy(jacobi_matrix)
    cartesian_matrix = _readonly_copy(cartesian_matrix)
    checksum = _binding_content_sha256(
        body_ids=checked_ids,
        gravitational_parameters=gm,
        cumulative_gravitational_parameters=cumulative,
        canonical_inertias=inertias,
        jacobi_from_cartesian_matrix=jacobi_matrix,
        cartesian_from_jacobi_matrix=cartesian_matrix,
    )
    return JacobiCoordinateBinding(
        body_ids=checked_ids,
        gravitational_parameters=gm,
        cumulative_gravitational_parameters=cumulative,
        canonical_inertias=inertias,
        jacobi_from_cartesian_matrix=jacobi_matrix,
        cartesian_from_jacobi_matrix=cartesian_matrix,
        binding_content_sha256=checksum,
    )


def _validate_cartesian_state_arrays(
    binding: JacobiCoordinateBinding,
    positions: object,
    velocities: object,
) -> tuple[np.ndarray, np.ndarray]:
    if type(binding) is not JacobiCoordinateBinding:
        raise TrajectoryContractError(
            "binding must be an exact JacobiCoordinateBinding"
        )
    binding.__post_init__()
    shape = (len(binding.body_ids), 3)
    return (
        _validate_float64_array(
            positions, "Cartesian positions", shape, owned_readonly=False
        ),
        _validate_float64_array(
            velocities, "Cartesian velocities", shape, owned_readonly=False
        ),
    )


def _cartesian_to_jacobi(
    binding: JacobiCoordinateBinding,
    positions: np.ndarray,
    velocities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the fixed O(N) forward recurrence to Cartesian x and v."""

    positions, velocities = _validate_cartesian_state_arrays(
        binding, positions, velocities
    )
    body_count = len(binding.body_ids)
    mu = binding.gravitational_parameters
    cumulative = binding.cumulative_gravitational_parameters
    inertias = binding.canonical_inertias
    coordinates = np.empty_like(positions)
    momenta = np.empty_like(velocities)

    weighted_position = np.empty(3, dtype=np.float64)
    weighted_velocity = np.empty(3, dtype=np.float64)
    for component in range(3):
        weighted_position[component] = float(mu[0]) * float(
            positions[0, component]
        )
        weighted_velocity[component] = float(mu[0]) * float(
            velocities[0, component]
        )
    for index in range(1, body_count):
        prefix = float(cumulative[index - 1])
        for component in range(3):
            prefix_position = float(weighted_position[component]) / prefix
            prefix_velocity = float(weighted_velocity[component]) / prefix
            coordinates[index, component] = float(
                positions[index, component]
            ) - prefix_position
            momenta[index, component] = float(inertias[index]) * (
                float(velocities[index, component]) - prefix_velocity
            )
            weighted_position[component] = float(
                weighted_position[component]
            ) + float(mu[index]) * float(positions[index, component])
            weighted_velocity[component] = float(
                weighted_velocity[component]
            ) + float(mu[index]) * float(velocities[index, component])
    total = float(cumulative[-1])
    for component in range(3):
        coordinates[0, component] = float(weighted_position[component]) / total
        momenta[0, component] = float(weighted_velocity[component])
    if not bool(np.all(np.isfinite(coordinates))) or not bool(
        np.all(np.isfinite(momenta))
    ):
        raise TrajectoryDomainError(
            "Cartesian-to-Jacobi recurrence produced a nonfinite value"
        )
    return coordinates, momenta


def _jacobi_to_cartesian(
    binding: JacobiCoordinateBinding,
    coordinates: np.ndarray,
    momenta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the fixed O(N) inverse recurrence to Jacobi Q and P."""

    if type(binding) is not JacobiCoordinateBinding:
        raise TrajectoryContractError(
            "binding must be an exact JacobiCoordinateBinding"
        )
    binding.__post_init__()
    body_count = len(binding.body_ids)
    shape = (body_count, 3)
    coordinates = _validate_float64_array(
        coordinates, "Jacobi coordinates", shape, owned_readonly=False
    )
    momenta = _validate_float64_array(
        momenta, "Jacobi momenta", shape, owned_readonly=False
    )
    mu = binding.gravitational_parameters
    cumulative = binding.cumulative_gravitational_parameters
    inertias = binding.canonical_inertias

    prefix_positions = np.empty_like(coordinates)
    prefix_velocities = np.empty_like(momenta)
    total = float(cumulative[-1])
    for component in range(3):
        prefix_positions[-1, component] = float(coordinates[0, component])
        prefix_velocities[-1, component] = (
            float(momenta[0, component]) / total
        )
    for index in range(body_count - 1, 0, -1):
        position_coefficient = float(mu[index]) / float(cumulative[index])
        velocity_divisor = float(cumulative[index - 1])
        for component in range(3):
            prefix_positions[index - 1, component] = float(
                prefix_positions[index, component]
            ) - position_coefficient * float(coordinates[index, component])
            prefix_velocities[index - 1, component] = float(
                prefix_velocities[index, component]
            ) - float(momenta[index, component]) / velocity_divisor

    positions = np.empty_like(coordinates)
    velocities = np.empty_like(momenta)
    for component in range(3):
        positions[0, component] = float(prefix_positions[0, component])
        velocities[0, component] = float(prefix_velocities[0, component])
    for index in range(1, body_count):
        inertia = float(inertias[index])
        for component in range(3):
            positions[index, component] = float(
                prefix_positions[index - 1, component]
            ) + float(coordinates[index, component])
            velocities[index, component] = float(
                prefix_velocities[index - 1, component]
            ) + float(momenta[index, component]) / inertia
    if not bool(np.all(np.isfinite(positions))) or not bool(
        np.all(np.isfinite(velocities))
    ):
        raise TrajectoryDomainError(
            "Jacobi-to-Cartesian recurrence produced a nonfinite value"
        )
    return positions, velocities


@dataclass(frozen=True)
class _UniversalGValues:
    g0: float
    g1: float
    g2: float
    g3: float
    used_series: bool
    term_count_per_function: int


@dataclass(frozen=True)
class _UniversalEvaluation:
    anomaly: float
    residual: float
    radius: float
    tolerance: float
    g_values: _UniversalGValues


@dataclass(frozen=True, eq=False)
class _UniversalKeplerStep:
    position: np.ndarray
    velocity: np.ndarray
    iterations: int
    bracket_expansions: int
    g_function_evaluations: int
    series_terms_evaluated: int
    time_residual: float
    residual_tolerance: float
    final_bracket_width: float
    universal_anomaly: float
    beta: float
    semimajor_axis: float
    eccentricity: float
    periapse: float
    period: float
    periapse_timescale: float
    lagrange_identity_error: float
    energy_error: float
    angular_momentum_error: float


@dataclass(frozen=True, eq=False)
class _StrictKeplerProbeRecord:
    """Unbounded-body public-WH counterpart of the hybrid probe record.

    The frozen public WH contract has no body-count ceiling.  Hybrid custody is
    therefore used only inside its qualified at-most-16-body envelope, while
    this private record preserves the same attempted-work cadence for larger
    strict public maps without narrowing their accepted input domain.
    """

    secondary_index: int
    call_completed: bool
    iterations_entered: int
    bracket_expansions_entered: int
    g_bundle_calls_entered: int
    g_bundle_calls_completed: int
    completed_series_bundle_count: int
    interrupted_series_terms: int
    terminal_status: str

    @property
    def series_terms_evaluated(self) -> int:
        return (
            4 * WH_KEPLER_SERIES_TERM_COUNT * self.completed_series_bundle_count
            + self.interrupted_series_terms
        )


@dataclass
class _KeplerProbeAccumulator:
    """Mutable private counter sink for one entered secondary subflow."""

    secondary_index: int
    iterations_entered: int = 0
    bracket_expansions_entered: int = 0
    g_bundle_calls_entered: int = 0
    g_bundle_calls_completed: int = 0
    completed_series_bundle_count: int = 0
    interrupted_series_terms: int = 0
    call_completed: bool = False

    def begin_g_bundle(self) -> None:
        self.g_bundle_calls_entered += 1
        self.interrupted_series_terms = 0

    def record_series_term(self) -> None:
        self.interrupted_series_terms += 1

    def complete_g_bundle(self, *, used_series: bool) -> None:
        if used_series:
            if self.interrupted_series_terms != 4 * WH_KEPLER_SERIES_TERM_COUNT:
                raise TrajectoryContractError(
                    "completed universal G series has impossible attempted-term work"
                )
            self.completed_series_bundle_count += 1
        elif self.interrupted_series_terms != 0:
            raise TrajectoryContractError(
                "trigonometric universal G bundle retained series work"
            )
        self.g_bundle_calls_completed += 1
        self.interrupted_series_terms = 0

    def freeze(
        self, terminal_status: str, *, hybrid_qualified: bool = True
    ) -> HybridKeplerProbeRecord | _StrictKeplerProbeRecord:
        values = {
            "secondary_index": self.secondary_index,
            "call_completed": self.call_completed,
            "iterations_entered": self.iterations_entered,
            "bracket_expansions_entered": self.bracket_expansions_entered,
            "g_bundle_calls_entered": self.g_bundle_calls_entered,
            "g_bundle_calls_completed": self.g_bundle_calls_completed,
            "completed_series_bundle_count": self.completed_series_bundle_count,
            "interrupted_series_terms": self.interrupted_series_terms,
            "terminal_status": terminal_status,
        }
        if hybrid_qualified:
            return HybridKeplerProbeRecord(**values)
        return _StrictKeplerProbeRecord(**values)


def _g_series(
    order: int,
    beta: float,
    anomaly: float,
    _probe: _KeplerProbeAccumulator | None = None,
) -> float:
    if order == 0:
        term = 1.0
    elif order == 1:
        term = anomaly
    elif order == 2:
        term = (anomaly * anomaly) / 2.0
    elif order == 3:
        term = ((anomaly * anomaly) * anomaly) / 6.0
    else:  # pragma: no cover - private caller fixes the four supported orders.
        raise TrajectoryContractError("universal G order must be 0, 1, 2, or 3")
    if _probe is not None:
        _probe.record_series_term()
    rho = ((-beta) * anomaly) * anomaly
    total = term
    for series_index in range(WH_KEPLER_SERIES_TERM_COUNT - 1):
        first = order + 2 * series_index + 1
        second = first + 1
        term = (term * rho) / float(first * second)
        total = total + term
        if _probe is not None:
            _probe.record_series_term()
        if not math.isfinite(term) or not math.isfinite(total):
            raise TrajectoryDomainError(
                "universal G series produced a nonfinite binary64 term"
            )
    return float(total)


def _universal_g_values(
    beta: float,
    anomaly: float,
    spec: UniversalKeplerSolverSpec,
    _probe: _KeplerProbeAccumulator | None = None,
) -> _UniversalGValues:
    if type(spec) is not UniversalKeplerSolverSpec:
        raise TrajectoryContractError(
            "Kepler solver spec must be an exact UniversalKeplerSolverSpec"
        )
    spec.__post_init__()
    if (
        type(beta) is not float
        or not math.isfinite(beta)
        or beta <= 0.0
        or type(anomaly) is not float
        or not math.isfinite(anomaly)
    ):
        raise TrajectoryDomainError(
            "universal G evaluation requires finite anomaly and positive beta"
        )
    root_beta = math.sqrt(beta)
    argument = root_beta * anomaly
    if not math.isfinite(argument):
        raise TrajectoryDomainError(
            "universal G argument must remain finite"
    )
    if abs(argument) <= WH_KEPLER_SERIES_SWITCH_ABS_ARGUMENT:
        values = tuple(
            _g_series(order, beta, anomaly, _probe) for order in range(4)
        )
        return _UniversalGValues(
            values[0], values[1], values[2], values[3], True, 64
        )
    theta = argument
    cosine = math.cos(theta)
    sine = math.sin(theta)
    g0 = cosine
    g1 = sine / root_beta
    g2 = (1.0 - cosine) / beta
    g3 = (anomaly - g1) / beta
    if not all(math.isfinite(value) for value in (g0, g1, g2, g3)):
        raise TrajectoryDomainError(
            "universal G trigonometric branch produced a nonfinite value"
        )
    return _UniversalGValues(g0, g1, g2, g3, False, 0)


def _evaluate_universal_equation(
    *,
    anomaly: float,
    beta: float,
    radius_initial: float,
    radial_moment: float,
    gravitational_parameter: float,
    signed_step: float,
    spec: UniversalKeplerSolverSpec,
    _probe: _KeplerProbeAccumulator | None = None,
) -> _UniversalEvaluation:
    if _probe is not None:
        _probe.begin_g_bundle()
    g_values = _universal_g_values(beta, anomaly, spec, _probe)
    if _probe is not None:
        _probe.complete_g_bundle(used_series=g_values.used_series)
    coefficient = gravitational_parameter - beta * radius_initial
    first = radius_initial * anomaly
    second = radial_moment * g_values.g2
    third = coefficient * g_values.g3
    residual = ((first + second) + third) - signed_step
    radius = (radius_initial + radial_moment * g_values.g1) + (
        coefficient * g_values.g2
    )
    scale = abs(first) + abs(second) + abs(third) + abs(signed_step)
    tolerance = max(
        WH_KEPLER_RESIDUAL_ULP_FACTOR * math.ulp(float(abs(signed_step))),
        WH_KEPLER_RESIDUAL_ROUNDOFF_FACTOR * _FLOAT64_EPSILON * scale,
    )
    if not all(
        math.isfinite(value) for value in (residual, radius, scale, tolerance)
    ):
        raise TrajectoryDomainError(
            "universal Kepler equation produced a nonfinite value"
        )
    if radius <= 0.0:
        raise TrajectoryDomainError(
            "universal Kepler root derivative radius must be strictly positive"
        )
    return _UniversalEvaluation(
        anomaly, residual, radius, tolerance, g_values
    )


def _dot3(left: np.ndarray, right: np.ndarray) -> float:
    """Three-component dot product with fixed scalar operation order."""

    first = float(left[0]) * float(right[0])
    second = float(left[1]) * float(right[1])
    third = float(left[2]) * float(right[2])
    return float((first + second) + third)


def _cross3(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Three-component cross product with fixed scalar operation order."""

    result = np.empty(3, dtype=np.float64)
    result[0] = float(left[1]) * float(right[2]) - float(left[2]) * float(
        right[1]
    )
    result[1] = float(left[2]) * float(right[0]) - float(left[0]) * float(
        right[2]
    )
    result[2] = float(left[0]) * float(right[1]) - float(left[1]) * float(
        right[0]
    )
    return result


def _sum_1d_fixed(values: np.ndarray, start: int = 0) -> float:
    total = 0.0
    for index in range(start, int(values.shape[0])):
        total = total + float(values[index])
    return float(total)


def _state_axpy(
    base: np.ndarray, coefficient: float, increment: np.ndarray
) -> np.ndarray:
    """Return base + coefficient*increment in body/component scalar order."""

    result = np.empty_like(base)
    for body_index in range(int(base.shape[0])):
        for component in range(3):
            result[body_index, component] = float(
                base[body_index, component]
            ) + coefficient * float(increment[body_index, component])
    return result


def _norm3(value: np.ndarray) -> float:
    squared = _dot3(value, value)
    if not math.isfinite(squared) or squared < 0.0:
        raise TrajectoryDomainError("three-vector norm became nonfinite")
    return math.sqrt(squared)


def _orbital_elements(
    position: np.ndarray, velocity: np.ndarray, gravitational_parameter: float
) -> tuple[float, float, float, float, float, float, float]:
    radius = _norm3(position)
    speed_squared = _dot3(velocity, velocity)
    radial_dot = _dot3(position, velocity)
    if (
        radius <= 0.0
        or not math.isfinite(speed_squared)
        or speed_squared < 0.0
        or not math.isfinite(radial_dot)
    ):
        raise TrajectoryDomainError(
            "elliptic Kepler input must have finite positive radius"
        )
    beta = 2.0 * gravitational_parameter / radius - speed_squared
    if not math.isfinite(beta):
        raise TrajectoryDomainError(
            "Wisdom--Holman v1 accepts only bound elliptic Kepler subflows"
        )
    if beta <= 0.0:
        raise _FiniteFarEnvelopeExit(
            "FINITE_JACOBI_STATE_OUTSIDE_BOUND_ELLIPTIC_FAR_DOMAIN",
            "Wisdom--Holman v1 accepts only bound elliptic Kepler subflows",
        )
    semimajor_axis = gravitational_parameter / beta
    eccentricity_vector = np.empty(3, dtype=np.float64)
    eccentricity_coefficient = speed_squared - gravitational_parameter / radius
    for component in range(3):
        eccentricity_vector[component] = (
            eccentricity_coefficient * float(position[component])
            - radial_dot * float(velocity[component])
        ) / gravitational_parameter
    eccentricity = _norm3(eccentricity_vector)
    if not math.isfinite(eccentricity) or eccentricity < 0.0:
        raise TrajectoryDomainError("elliptic eccentricity must be finite")
    if eccentricity > WH_MAXIMUM_JACOBI_ECCENTRICITY:
        raise _FiniteFarEnvelopeExit(
            "FINITE_JACOBI_ECCENTRICITY_ABOVE_FAR_LIMIT",
            "Jacobi eccentricity exceeds the Wisdom--Holman v1 envelope",
        )
    periapse = semimajor_axis * (1.0 - eccentricity)
    period = 2.0 * math.pi * math.sqrt(
        (semimajor_axis * semimajor_axis * semimajor_axis)
        / gravitational_parameter
    )
    periapse_timescale = 2.0 * math.pi * math.sqrt(
        (
            (semimajor_axis * semimajor_axis * semimajor_axis)
            / gravitational_parameter
        )
        * (((1.0 - eccentricity) ** 3) / (1.0 + eccentricity))
    )
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (semimajor_axis, periapse, period, periapse_timescale)
    ):
        raise TrajectoryDomainError(
            "elliptic osculating elements must remain finite and positive"
        )
    return (
        radius,
        beta,
        semimajor_axis,
        eccentricity,
        periapse,
        period,
        periapse_timescale,
    )


def _solve_elliptic_universal_kepler(
    position: np.ndarray,
    velocity: np.ndarray,
    gravitational_parameter: float,
    signed_step: float,
    spec: UniversalKeplerSolverSpec,
    _probe: _KeplerProbeAccumulator | None = None,
) -> _UniversalKeplerStep:
    """Advance one bound relative state with the fixed G-function solver."""

    position = _validate_float64_array(
        position, "Kepler position", (3,), owned_readonly=False
    )
    velocity = _validate_float64_array(
        velocity, "Kepler velocity", (3,), owned_readonly=False
    )
    if (
        type(gravitational_parameter) is not float
        or not math.isfinite(gravitational_parameter)
        or gravitational_parameter <= 0.0
    ):
        raise TrajectoryContractError(
            "Kepler gravitational_parameter must be a finite positive built-in float"
        )
    if (
        type(signed_step) is not float
        or not math.isfinite(signed_step)
        or signed_step == 0.0
    ):
        raise TrajectoryContractError(
            "Kepler signed_step must be a finite nonzero built-in float"
        )
    if type(spec) is not UniversalKeplerSolverSpec:
        raise TrajectoryContractError(
            "Kepler solver spec must be an exact UniversalKeplerSolverSpec"
        )
    spec.__post_init__()

    (
        radius_initial,
        beta,
        semimajor_axis,
        eccentricity,
        periapse,
        period,
        periapse_timescale,
    ) = _orbital_elements(position, velocity, gravitational_parameter)
    radial_moment = _dot3(position, velocity)
    if not math.isfinite(radial_moment):
        raise TrajectoryDomainError("Kepler radial moment must be finite")

    seed = signed_step / radius_initial
    if not math.isfinite(seed) or seed == 0.0:
        raise TrajectoryStepLimitError(
            "Kepler root seed cannot make representable progress from zero"
        )
    zero_residual = -signed_step
    bracket_expansions = 0
    g_function_evaluations = 0
    series_terms_evaluated = 0

    outer = seed
    outer_evaluation = _evaluate_universal_equation(
        anomaly=outer,
        beta=beta,
        radius_initial=radius_initial,
        radial_moment=radial_moment,
        gravitational_parameter=gravitational_parameter,
        signed_step=signed_step,
        spec=spec,
        _probe=_probe,
    )
    g_function_evaluations += 1
    if outer_evaluation.g_values.used_series:
        series_terms_evaluated += 4 * spec.series_term_count

    while (
        (signed_step > 0.0 and outer_evaluation.residual < 0.0)
        or (signed_step < 0.0 and outer_evaluation.residual > 0.0)
    ):
        if bracket_expansions >= WH_KEPLER_MAXIMUM_BRACKET_EXPANSIONS:
            raise TrajectoryStepLimitError(
                "elliptic Kepler root could not be bracketed within the fixed expansion cap"
            )
        candidate_outer = outer * 2.0
        if not math.isfinite(candidate_outer) or candidate_outer == outer:
            raise TrajectoryStepLimitError(
                "elliptic Kepler bracket expansion cannot make finite representable progress"
            )
        outer = candidate_outer
        bracket_expansions += 1
        if _probe is not None:
            _probe.bracket_expansions_entered += 1
        outer_evaluation = _evaluate_universal_equation(
            anomaly=outer,
            beta=beta,
            radius_initial=radius_initial,
            radial_moment=radial_moment,
            gravitational_parameter=gravitational_parameter,
            signed_step=signed_step,
            spec=spec,
            _probe=_probe,
        )
        g_function_evaluations += 1
        if outer_evaluation.g_values.used_series:
            series_terms_evaluated += 4 * spec.series_term_count

    if signed_step > 0.0:
        low, residual_low = 0.0, zero_residual
        high, residual_high = outer, outer_evaluation.residual
    else:
        low, residual_low = outer, outer_evaluation.residual
        high, residual_high = 0.0, zero_residual
    if not (
        low < high
        and residual_low <= 0.0
        and residual_high >= 0.0
        and all(
            math.isfinite(value)
            for value in (low, high, residual_low, residual_high)
        )
    ):
        raise TrajectoryStepLimitError(
            "elliptic Kepler root bracket does not preserve R(low)<=0<=R(high)"
        )

    if outer_evaluation.residual == 0.0:
        current = outer
    else:
        current = low + 0.5 * (high - low)
    if not math.isfinite(current) or not low <= current <= high:
        raise TrajectoryStepLimitError(
            "elliptic Kepler midpoint is not finite inside its bracket"
        )
    previous: float | None = None
    two_iterations_ago: float | None = None
    accepted: _UniversalEvaluation | None = None
    final_width = math.inf

    for iteration in range(1, WH_KEPLER_MAXIMUM_ITERATIONS + 1):
        if _probe is not None:
            _probe.iterations_entered += 1
        evaluation = _evaluate_universal_equation(
            anomaly=current,
            beta=beta,
            radius_initial=radius_initial,
            radial_moment=radial_moment,
            gravitational_parameter=gravitational_parameter,
            signed_step=signed_step,
            spec=spec,
            _probe=_probe,
        )
        g_function_evaluations += 1
        if evaluation.g_values.used_series:
            series_terms_evaluated += 4 * spec.series_term_count

        if evaluation.residual == 0.0:
            low = current
            high = current
            residual_low = 0.0
            residual_high = 0.0
        elif evaluation.residual < 0.0:
            low = current
            residual_low = evaluation.residual
        else:
            high = current
            residual_high = evaluation.residual
        if not (residual_low <= 0.0 <= residual_high and low <= high):
            raise TrajectoryStepLimitError(
                "elliptic Kepler iteration lost its signed root bracket"
            )

        final_width = high - low
        ulp_scale = max(
            math.ulp(float(abs(low))),
            math.ulp(float(abs(high))),
            math.ulp(float(abs(current))),
        )
        adjacent = low == high or math.nextafter(low, high) == high
        settled = adjacent or final_width <= (
            WH_KEPLER_BRACKET_ULP_FACTOR * ulp_scale
        )
        cycle = (
            (previous is not None and current == previous)
            or (
                two_iterations_ago is not None
                and current == two_iterations_ago
            )
        )
        if abs(evaluation.residual) <= evaluation.tolerance and (
            settled or cycle
        ):
            accepted = evaluation
            accepted_iterations = iteration
            break

        midpoint = low + 0.5 * (high - low)
        newton = current - evaluation.residual / evaluation.radius
        if math.isfinite(newton) and low < newton < high:
            candidate = newton
        else:
            candidate = midpoint
        if candidate == current:
            candidate = midpoint
        if (
            not math.isfinite(candidate)
            or not low <= candidate <= high
            or candidate == current
        ):
            raise TrajectoryStepLimitError(
                "elliptic Kepler iteration stagnated without satisfying convergence"
            )
        two_iterations_ago = previous
        previous = current
        current = candidate
    else:
        raise TrajectoryStepLimitError(
            "elliptic Kepler root exceeded the fixed iteration cap"
        )

    assert accepted is not None
    g_values = accepted.g_values
    f_value = 1.0 - (
        gravitational_parameter / radius_initial
    ) * g_values.g2
    g_value = signed_step - gravitational_parameter * g_values.g3
    candidate_position = np.empty(3, dtype=np.float64)
    for component in range(3):
        candidate_position[component] = (
            f_value * float(position[component])
            + g_value * float(velocity[component])
        )
    radius_final = _norm3(candidate_position)
    if radius_final <= 0.0:
        raise TrajectoryDomainError(
            "elliptic Kepler reconstruction produced a nonpositive radius"
        )
    fdot = -(
        gravitational_parameter / (radius_initial * radius_final)
    ) * g_values.g1
    gdot = 1.0 - (
        gravitational_parameter / radius_final
    ) * g_values.g2
    candidate_velocity = np.empty(3, dtype=np.float64)
    for component in range(3):
        candidate_velocity[component] = (
            fdot * float(position[component])
            + gdot * float(velocity[component])
        )
    if not bool(np.all(np.isfinite(candidate_position))) or not bool(
        np.all(np.isfinite(candidate_velocity))
    ):
        raise TrajectoryDomainError(
            "elliptic Kepler reconstruction produced a nonfinite vector"
        )
    if not all(
        math.isfinite(value)
        for value in (f_value, g_value, fdot, gdot, radius_final)
    ):
        raise TrajectoryDomainError(
            "elliptic Kepler reconstruction produced a nonfinite scalar"
        )

    lagrange_error = abs(f_value * gdot - fdot * g_value - 1.0)
    lagrange_scale = max(
        1.0, abs(f_value * gdot) + abs(fdot * g_value)
    )
    lagrange_limit = (
        WH_KEPLER_LAGRANGE_IDENTITY_ROUNDOFF_FACTOR
        * _FLOAT64_EPSILON
        * lagrange_scale
    )
    if not all(
        math.isfinite(value)
        for value in (lagrange_error, lagrange_scale, lagrange_limit)
    ) or lagrange_error > lagrange_limit:
        raise _KeplerSolverPostconditionFailure(
            "elliptic Kepler Lagrange identity postcondition failed"
        )

    speed_initial_squared = _dot3(velocity, velocity)
    speed_final_squared = _dot3(candidate_velocity, candidate_velocity)
    energy_initial = (
        0.5 * speed_initial_squared - gravitational_parameter / radius_initial
    )
    energy_final = (
        0.5 * speed_final_squared - gravitational_parameter / radius_final
    )
    energy_error = abs(energy_final - energy_initial)
    energy_scale = max(
        gravitational_parameter / radius_initial,
        gravitational_parameter / radius_final,
        0.5 * speed_initial_squared,
        0.5 * speed_final_squared,
    )
    energy_limit = (
        WH_KEPLER_ENERGY_ROUNDOFF_FACTOR
        * _FLOAT64_EPSILON
        * energy_scale
    )
    if not all(
        math.isfinite(value)
        for value in (energy_error, energy_scale, energy_limit)
    ) or energy_error > energy_limit:
        raise _KeplerSolverPostconditionFailure(
            "elliptic Kepler specific-energy postcondition failed"
        )

    angular_initial = _cross3(position, velocity)
    angular_final = _cross3(candidate_position, candidate_velocity)
    angular_error = _norm3(angular_final - angular_initial)
    angular_scale = max(
        radius_initial * _norm3(velocity),
        radius_final * _norm3(candidate_velocity),
    )
    angular_limit = (
        WH_KEPLER_ANGULAR_MOMENTUM_ROUNDOFF_FACTOR
        * _FLOAT64_EPSILON
        * angular_scale
    )
    if not all(
        math.isfinite(value)
        for value in (angular_error, angular_scale, angular_limit)
    ) or angular_error > angular_limit:
        raise _KeplerSolverPostconditionFailure(
            "elliptic Kepler angular-momentum postcondition failed"
        )

    return _UniversalKeplerStep(
        position=candidate_position,
        velocity=candidate_velocity,
        iterations=accepted_iterations,
        bracket_expansions=bracket_expansions,
        g_function_evaluations=g_function_evaluations,
        series_terms_evaluated=series_terms_evaluated,
        time_residual=abs(accepted.residual),
        residual_tolerance=accepted.tolerance,
        final_bracket_width=final_width,
        universal_anomaly=accepted.anomaly,
        beta=beta,
        semimajor_axis=semimajor_axis,
        eccentricity=eccentricity,
        periapse=periapse,
        period=period,
        periapse_timescale=periapse_timescale,
        lagrange_identity_error=lagrange_error,
        energy_error=energy_error,
        angular_momentum_error=angular_error,
    )


def _step_epochs(
    initial_epoch: float, spec: FixedStepWisdomHolmanSpec
) -> tuple[float, ...]:
    epochs: list[float] = []
    previous: float | None = None
    direction = 1.0 if spec.fixed_step > 0.0 else -1.0
    for step_index in range(spec.completed_steps + 1):
        epoch = float(initial_epoch + spec.fixed_step * step_index)
        if not math.isfinite(epoch):
            raise TrajectoryStepLimitError(
                "the fixed Wisdom--Holman schedule produced a nonfinite epoch"
            )
        if previous is not None and direction * (epoch - previous) <= 0.0:
            raise TrajectoryStepLimitError(
                "fixed_step cannot advance every binary64 Wisdom--Holman epoch"
            )
        epochs.append(epoch)
        previous = epoch
    return tuple(epochs)


def _validate_semantic_boundary(snapshot: StateSnapshot) -> None:
    if snapshot.time_scale not in WH_ALLOWED_TIME_SCALES:
        raise TrajectoryContractError(
            f"Wisdom--Holman time_scale must be one of {WH_ALLOWED_TIME_SCALES!r}"
        )
    if snapshot.frame != WH_REQUIRED_FRAME:
        raise TrajectoryContractError(
            f"Wisdom--Holman frame must equal {WH_REQUIRED_FRAME!r}"
        )
    if snapshot.origin != WH_REQUIRED_ORIGIN:
        raise TrajectoryContractError(
            f"Wisdom--Holman origin must equal {WH_REQUIRED_ORIGIN!r}"
        )
    if snapshot.axes not in WH_ALLOWED_AXES:
        raise TrajectoryContractError(
            f"Wisdom--Holman axes must be one of {WH_ALLOWED_AXES!r}"
        )


def _validate_force_plan_scope(
    snapshot: StateSnapshot, plan: ForcePlan, spec: FixedStepWisdomHolmanSpec
) -> NewtonianPointMass:
    if plan.evidence_class != WH_EVIDENCE_CLASS:
        raise TrajectoryContractError(
            "Wisdom--Holman accepts only MODEL_OUTPUT force plans"
        )
    if plan.registry_authorized or plan.qualification_authorized:
        raise TrajectoryContractError(
            "Wisdom--Holman force plans cannot carry authority"
        )
    if plan.backend.backend_id != "numpy" or plan.backend.device != "cpu":
        raise TrajectoryContractError(
            "Wisdom--Holman v1 requires the NumPy CPU backend"
        )
    if len(snapshot.body_ids) < 2:
        raise TrajectoryContractError(
            "Wisdom--Holman requires at least two mutually interacting bodies"
        )
    if spec.jacobi_body_order != snapshot.body_ids:
        raise TrajectoryContractError(
            "jacobi_body_order must equal snapshot.body_ids in exact central-first order"
        )
    if type(plan.models) is not tuple or len(plan.models) != 1:
        raise TrajectoryContractError(
            "Wisdom--Holman requires exactly one NewtonianPointMass configuration"
        )
    model = plan.models[0]
    if type(model) is not NewtonianPointMass:
        raise TrajectoryContractError(
            "Wisdom--Holman requires exactly one NewtonianPointMass configuration"
        )
    if model.source_ids != snapshot.body_ids or model.target_ids != snapshot.body_ids:
        raise TrajectoryContractError(
            "Wisdom--Holman requires source_ids == target_ids == snapshot.body_ids in exact order"
        )
    return model


def _validate_metadata_range(plan: ForcePlan, lower: float, upper: float) -> None:
    for model in plan.models:
        for metadata in model.parameter_metadata:  # type: ignore[union-attr]
            if metadata.validity_start > lower or metadata.validity_end < upper:
                raise TrajectoryContractError(
                    f"{metadata.parameter_id} validity interval does not cover the full Wisdom--Holman schedule"
                )


def _maximum_pairwise_distance(values: np.ndarray) -> float:
    maximum = 0.0
    for left in range(values.shape[0] - 1):
        for right in range(left + 1, values.shape[0]):
            maximum = max(maximum, _norm3(values[right] - values[left]))
    return maximum


def _validate_initial_barycenter(
    *,
    binding: JacobiCoordinateBinding,
    coordinates: np.ndarray,
    momenta: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    spec: FixedStepWisdomHolmanSpec,
) -> tuple[float, float, float, float]:
    body_count = len(binding.body_ids)
    position_scale = _maximum_pairwise_distance(positions)
    velocity_scale = _maximum_pairwise_distance(velocities)
    derived_position_cap = (
        spec.barycenter_roundoff_factor
        * body_count
        * _FLOAT64_EPSILON
        * position_scale
    )
    derived_velocity_cap = (
        spec.barycenter_roundoff_factor
        * body_count
        * _FLOAT64_EPSILON
        * velocity_scale
    )
    position_cap = min(
        spec.maximum_initial_barycenter_position_norm,
        derived_position_cap,
    )
    velocity_cap = min(
        spec.maximum_initial_barycenter_velocity_norm,
        derived_velocity_cap,
    )
    position_norm = _norm3(coordinates[0])
    barycenter_velocity = momenta[0] / binding.cumulative_gravitational_parameters[-1]
    velocity_norm = _norm3(barycenter_velocity)
    if position_norm > position_cap:
        raise TrajectoryDomainError(
            "initial GM-weighted barycenter position exceeds its fixed roundoff-scaled cap"
        )
    if velocity_norm > velocity_cap:
        raise TrajectoryDomainError(
            "initial GM-weighted barycenter velocity exceeds its fixed roundoff-scaled cap"
        )
    return position_norm, velocity_norm, position_cap, velocity_cap


@dataclass(frozen=True)
class _NodeGuardEvidence:
    semimajor_axes: tuple[float, ...]
    eccentricities: tuple[float, ...]
    periapses: tuple[float, ...]
    periods: tuple[float, ...]
    periapse_timescales: tuple[float, ...]
    interaction_force_ratios: tuple[float, ...]
    orbit_step_fractions: tuple[float, ...]
    periapse_step_fractions: tuple[float, ...]
    pair_separations: tuple[float, ...]
    secondary_hill_floor_ratios: tuple[float, ...]
    barycenter_position_norm: float
    barycenter_velocity_norm: float


@dataclass(frozen=True)
class _PathGuardEvidence:
    endpoint_minimum_separations: tuple[float, ...]
    path_lower_bounds: tuple[float, ...]
    clearance_after_margins: tuple[float, ...]


@dataclass(frozen=True, eq=False)
class _NodeGuardVerdict:
    evidence: _NodeGuardEvidence | None
    reason: str | None
    body_indices: tuple[int, ...]
    pair_indices: tuple[tuple[int, int], ...]
    public_error: TrajectoryDomainError | None


@dataclass(frozen=True, eq=False)
class _PathGuardVerdict:
    evidence: _PathGuardEvidence | None
    reason: str | None
    pair_indices: tuple[tuple[int, int], ...]
    public_error: TrajectoryDomainError | None


_DYNAMIC_GUARD_MESSAGES = {
    "FINITE_JACOBI_STATE_OUTSIDE_BOUND_ELLIPTIC_FAR_DOMAIN": (
        "Wisdom--Holman v1 accepts only bound elliptic Kepler subflows"
    ),
    "FINITE_JACOBI_ECCENTRICITY_ABOVE_FAR_LIMIT": (
        "Jacobi eccentricity exceeds the Wisdom--Holman v1 envelope"
    ),
    "FINITE_JACOBI_PERIAPSE_BELOW_FAR_FLOOR": (
        "Jacobi osculating periapse is below the mandatory caller floor"
    ),
    "FINITE_JACOBI_SEMIMAJOR_AXES_NOT_STRICTLY_INCREASING": (
        "osculating Jacobi semimajor axes must remain strictly increasing in body order"
    ),
    "FINITE_INTERACTION_TO_KEPLER_RATIO_ABOVE_FAR_LIMIT": (
        "interaction-to-Kepler force ratio exceeds the Wisdom--Holman v1 envelope"
    ),
    "FINITE_ORBIT_STEP_FRACTION_ABOVE_FAR_LIMIT": (
        "fixed step exceeds one twentieth of a Jacobi orbit period"
    ),
    "FINITE_PERIAPSE_STEP_FRACTION_ABOVE_FAR_LIMIT": (
        "fixed step exceeds one sixteenth of a Jacobi periapse timescale"
    ),
    "FINITE_NODE_PAIR_CLEARANCE_AT_OR_BELOW_FAR_FLOOR": (
        "Wisdom--Holman node reaches its contact, encounter, or mutual-Hill floor"
    ),
    "FINITE_KEPLER_DRIFT_CLEARANCE_UNCERTIFIED_BY_FAR_SCREEN": (
        "Wisdom--Holman Kepler drift reaches its conservative contact, encounter, or Hill screen"
    ),
}


def _interaction_force(
    *,
    binding: JacobiCoordinateBinding,
    coordinates: np.ndarray,
    cartesian_acceleration: np.ndarray,
) -> tuple[np.ndarray, float]:
    body_count = len(binding.body_ids)
    _validate_float64_array(
        coordinates,
        "interaction coordinates",
        (body_count, 3),
        owned_readonly=False,
    )
    _validate_float64_array(
        cartesian_acceleration,
        "interaction Cartesian acceleration",
        (body_count, 3),
        owned_readonly=False,
    )
    cartesian_canonical_force = np.empty_like(cartesian_acceleration)
    for body_index in range(body_count):
        weight = float(binding.gravitational_parameters[body_index])
        for component in range(3):
            cartesian_canonical_force[body_index, component] = weight * float(
                cartesian_acceleration[body_index, component]
            )
    full_jacobi_force = np.empty_like(cartesian_canonical_force)
    matrix = binding.cartesian_from_jacobi_matrix
    for coordinate_index in range(body_count):
        for component in range(3):
            total = 0.0
            for body_index in range(body_count):
                total = total + float(matrix[body_index, coordinate_index]) * float(
                    cartesian_canonical_force[body_index, component]
                )
            full_jacobi_force[coordinate_index, component] = total
    if not bool(np.all(np.isfinite(full_jacobi_force))):
        raise _InteractionAssemblyNumericalFailure(
            "full Cartesian-to-Jacobi canonical force transform became nonfinite"
        )
    translation_residual = _norm3(full_jacobi_force[0])
    force_scale = max(
        (_norm3(row) for row in cartesian_canonical_force), default=0.0
    )
    translation_cap = (
        WH_BARYCENTER_ROUNDOFF_FACTOR
        * body_count
        * _FLOAT64_EPSILON
        * force_scale
    )
    if translation_residual > translation_cap:
        raise _TranslationForceResidualFailure(
            "closed mutual force does not satisfy the Jacobi translation-force residual bound"
        )

    interaction = np.empty_like(full_jacobi_force)
    interaction[0] = np.zeros(3, dtype=np.float64)
    for index in range(1, body_count):
        try:
            radius = _norm3(coordinates[index])
        except TrajectoryDomainError as exc:
            raise _InteractionAssemblyNumericalFailure(
                str(exc), (index,)
            ) from exc
        if radius <= 0.0:
            raise _InteractionAssemblyNumericalFailure(
                "Jacobi Kepler coordinate reaches a singular zero radius",
                (index,),
            )
        central_coefficient = -(
            float(binding.canonical_inertias[index])
            * float(binding.cumulative_gravitational_parameters[index])
            / ((radius * radius) * radius)
        )
        for component in range(3):
            central = central_coefficient * float(coordinates[index, component])
            interaction[index, component] = float(
                full_jacobi_force[index, component]
            ) - central
    if not bool(np.all(np.isfinite(interaction))):
        raise _InteractionAssemblyNumericalFailure(
            "interaction canonical force became nonfinite"
        )
    return interaction, translation_residual


def _secondary_hill_radius(
    binding: JacobiCoordinateBinding,
    semimajor_axes: tuple[float, ...],
    left: int,
    right: int,
) -> float:
    if left == 0 or right == 0:
        return 0.0
    mu = binding.gravitational_parameters
    average_axis = 0.5 * (
        semimajor_axes[left - 1] + semimajor_axes[right - 1]
    )
    ratio = (mu[left] + mu[right]) / (3.0 * mu[0])
    hill = average_axis * math.pow(ratio, 1.0 / 3.0)
    if not math.isfinite(hill) or hill <= 0.0:
        raise TrajectoryDomainError(
            "secondary mutual-Hill radius must remain finite and positive"
        )
    return hill


def _node_guard_verdict(
    *,
    binding: JacobiCoordinateBinding,
    coordinates: np.ndarray,
    momenta: np.ndarray,
    interaction_force: np.ndarray,
    cartesian_positions: np.ndarray,
    radii: np.ndarray,
    spec: FixedStepWisdomHolmanSpec,
) -> _NodeGuardVerdict:
    body_count = len(binding.body_ids)
    body_offenders: dict[str, list[int]] = {
        reason: [] for reason in HYBRID_DYNAMIC_SWITCH_REASONS[:-2]
    }
    pair_offenders: dict[str, list[tuple[int, int]]] = {
        reason: [] for reason in HYBRID_DYNAMIC_SWITCH_REASONS[-2:]
    }
    semimajor_axes: list[float] = []
    eccentricities: list[float] = []
    periapses: list[float] = []
    periods: list[float] = []
    periapse_timescales: list[float] = []
    interaction_ratios: list[float] = []
    orbit_step_fractions: list[float] = []
    periapse_step_fractions: list[float] = []
    missing_orbital_elements = False
    for index in range(1, body_count):
        velocity = momenta[index] / binding.canonical_inertias[index]
        try:
            (
                radius,
                _beta,
                semimajor_axis,
                eccentricity,
                periapse,
                period,
                periapse_timescale,
            ) = _orbital_elements(
                coordinates[index],
                velocity,
                float(binding.cumulative_gravitational_parameters[index]),
            )
        except _FiniteFarEnvelopeExit as exc:
            body_offenders[exc.reason].append(index)
            missing_orbital_elements = True
            continue
        except TrajectoryDomainError as exc:
            raise _BodyNumericalFailure(str(exc), (index,)) from exc
        if periapse < spec.minimum_jacobi_periapse:
            body_offenders[
                "FINITE_JACOBI_PERIAPSE_BELOW_FAR_FLOOR"
            ].append(index)
        try:
            acceleration_scale = (
                binding.cumulative_gravitational_parameters[index]
                / (radius * radius)
            )
            interaction_acceleration = _norm3(interaction_force[index]) / float(
                binding.canonical_inertias[index]
            )
            interaction_ratio = interaction_acceleration / acceleration_scale
            orbit_fraction = abs(spec.fixed_step) / period
            periapse_fraction = abs(spec.fixed_step) / periapse_timescale
        except TrajectoryDomainError as exc:
            raise _BodyNumericalFailure(str(exc), (index,)) from exc
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in (
                interaction_ratio,
                orbit_fraction,
                periapse_fraction,
            )
        ):
            raise _BodyNumericalFailure(
                "Wisdom--Holman hierarchy guard produced a nonfinite ratio",
                (index,),
            )
        if interaction_ratio > WH_MAXIMUM_INTERACTION_TO_KEPLER_FORCE_RATIO:
            body_offenders[
                "FINITE_INTERACTION_TO_KEPLER_RATIO_ABOVE_FAR_LIMIT"
            ].append(index)
        if orbit_fraction > WH_MAXIMUM_ORBIT_STEP_FRACTION:
            body_offenders[
                "FINITE_ORBIT_STEP_FRACTION_ABOVE_FAR_LIMIT"
            ].append(index)
        if periapse_fraction > WH_MAXIMUM_PERIAPSE_STEP_FRACTION:
            body_offenders[
                "FINITE_PERIAPSE_STEP_FRACTION_ABOVE_FAR_LIMIT"
            ].append(index)
        semimajor_axes.append(semimajor_axis)
        eccentricities.append(eccentricity)
        periapses.append(periapse)
        periods.append(period)
        periapse_timescales.append(periapse_timescale)
        interaction_ratios.append(float(interaction_ratio))
        orbit_step_fractions.append(orbit_fraction)
        periapse_step_fractions.append(periapse_fraction)
    if not missing_orbital_elements:
        for offset, (left_axis, right_axis) in enumerate(
            zip(semimajor_axes, semimajor_axes[1:]), start=1
        ):
            if right_axis <= left_axis:
                offenders = body_offenders[
                    "FINITE_JACOBI_SEMIMAJOR_AXES_NOT_STRICTLY_INCREASING"
                ]
                offenders.extend((offset, offset + 1))

    pair_separations: list[float] = []
    hill_floor_ratios: list[float] = []
    if not missing_orbital_elements:
        for left in range(body_count - 1):
            for right in range(left + 1, body_count):
                separation = _norm3(
                    cartesian_positions[right] - cartesian_positions[left]
                )
                hill = _secondary_hill_radius(
                    binding, tuple(semimajor_axes), left, right
                )
                hill_floor = (
                    spec.minimum_mutual_hill_separation_multiple * hill
                    if hill > 0.0
                    else 0.0
                )
                floor = max(
                    float(radii[left] + radii[right]),
                    spec.minimum_encounter_pair_separation,
                    hill_floor,
                )
                margin = (
                    spec.encounter_roundoff_factor
                    * body_count
                    * _FLOAT64_EPSILON
                    * max(separation, floor)
                )
                if separation - margin <= floor:
                    pair_offenders[
                        "FINITE_NODE_PAIR_CLEARANCE_AT_OR_BELOW_FAR_FLOOR"
                    ].append((left, right))
                pair_separations.append(separation)
                if hill > 0.0:
                    hill_floor_ratios.append(float(separation / hill_floor))

    barycenter_position_norm = _norm3(coordinates[0])
    barycenter_velocity_norm = _norm3(
        momenta[0] / binding.cumulative_gravitational_parameters[-1]
    )
    for reason in HYBRID_DYNAMIC_SWITCH_REASONS:
        bodies = tuple(sorted(set(body_offenders.get(reason, ()))))
        pairs = tuple(pair_offenders.get(reason, ()))
        if bodies or pairs:
            return _NodeGuardVerdict(
                evidence=None,
                reason=reason,
                body_indices=bodies,
                pair_indices=pairs,
                public_error=TrajectoryDomainError(_DYNAMIC_GUARD_MESSAGES[reason]),
            )
    evidence = _NodeGuardEvidence(
        semimajor_axes=tuple(semimajor_axes),
        eccentricities=tuple(eccentricities),
        periapses=tuple(periapses),
        periods=tuple(periods),
        periapse_timescales=tuple(periapse_timescales),
        interaction_force_ratios=tuple(interaction_ratios),
        orbit_step_fractions=tuple(orbit_step_fractions),
        periapse_step_fractions=tuple(periapse_step_fractions),
        pair_separations=tuple(pair_separations),
        secondary_hill_floor_ratios=tuple(hill_floor_ratios),
        barycenter_position_norm=barycenter_position_norm,
        barycenter_velocity_norm=barycenter_velocity_norm,
    )
    return _NodeGuardVerdict(evidence, None, (), (), None)


def _node_guard(
    *,
    binding: JacobiCoordinateBinding,
    coordinates: np.ndarray,
    momenta: np.ndarray,
    interaction_force: np.ndarray,
    cartesian_positions: np.ndarray,
    radii: np.ndarray,
    spec: FixedStepWisdomHolmanSpec,
) -> _NodeGuardEvidence:
    verdict = _node_guard_verdict(
        binding=binding,
        coordinates=coordinates,
        momenta=momenta,
        interaction_force=interaction_force,
        cartesian_positions=cartesian_positions,
        radii=radii,
        spec=spec,
    )
    if verdict.public_error is not None:
        raise verdict.public_error
    assert verdict.evidence is not None
    return verdict.evidence


def _path_guard_verdict(
    *,
    binding: JacobiCoordinateBinding,
    start_positions: np.ndarray,
    end_positions: np.ndarray,
    radii: np.ndarray,
    drift_orbits: _NodeGuardEvidence,
    spec: FixedStepWisdomHolmanSpec,
) -> _PathGuardVerdict:
    body_count = len(binding.body_ids)
    periapse_speeds = tuple(
        math.sqrt(
            float(binding.cumulative_gravitational_parameters[index])
            * (1.0 + drift_orbits.eccentricities[index - 1])
            / (
                drift_orbits.semimajor_axes[index - 1]
                * (1.0 - drift_orbits.eccentricities[index - 1])
            )
        )
        for index in range(1, body_count)
    )
    endpoint_minima: list[float] = []
    lower_bounds: list[float] = []
    clearances: list[float] = []
    offending_pairs: list[tuple[int, int]] = []
    matrix = binding.cartesian_from_jacobi_matrix
    for left in range(body_count - 1):
        for right in range(left + 1, body_count):
            start_distance = _norm3(start_positions[right] - start_positions[left])
            end_distance = _norm3(end_positions[right] - end_positions[left])
            path_rate_bound = 0.0
            for index in range(1, body_count):
                path_rate_bound += abs(
                    float(matrix[left, index] - matrix[right, index])
                ) * periapse_speeds[index - 1]
            path_length_bound = abs(spec.fixed_step) * path_rate_bound
            lower_bound = max(start_distance, end_distance) - path_length_bound
            hill = _secondary_hill_radius(
                binding, drift_orbits.semimajor_axes, left, right
            )
            hill_floor = (
                spec.minimum_mutual_hill_separation_multiple * hill
                if hill > 0.0
                else 0.0
            )
            floor = max(
                float(radii[left] + radii[right]),
                spec.minimum_encounter_pair_separation,
                hill_floor,
            )
            margin = (
                spec.encounter_roundoff_factor
                * body_count
                * _FLOAT64_EPSILON
                * max(
                    start_distance,
                    end_distance,
                    path_length_bound,
                    floor,
                )
            )
            values = (
                start_distance - margin,
                end_distance - margin,
                lower_bound - margin,
            )
            if not all(math.isfinite(value) for value in values):
                raise TrajectoryDomainError(
                    "Wisdom--Holman Kepler drift reaches its conservative contact, encounter, or Hill screen"
                )
            if not all(value > floor for value in values):
                offending_pairs.append((left, right))
            endpoint_minima.append(min(start_distance, end_distance))
            lower_bounds.append(lower_bound)
            clearances.append(min(values) - floor)
    if offending_pairs:
        reason = "FINITE_KEPLER_DRIFT_CLEARANCE_UNCERTIFIED_BY_FAR_SCREEN"
        return _PathGuardVerdict(
            evidence=None,
            reason=reason,
            pair_indices=tuple(offending_pairs),
            public_error=TrajectoryDomainError(_DYNAMIC_GUARD_MESSAGES[reason]),
        )
    evidence = _PathGuardEvidence(
        endpoint_minimum_separations=tuple(endpoint_minima),
        path_lower_bounds=tuple(lower_bounds),
        clearance_after_margins=tuple(clearances),
    )
    return _PathGuardVerdict(evidence, None, (), None)


def _path_guard(
    *,
    binding: JacobiCoordinateBinding,
    start_positions: np.ndarray,
    end_positions: np.ndarray,
    radii: np.ndarray,
    drift_orbits: _NodeGuardEvidence,
    spec: FixedStepWisdomHolmanSpec,
) -> _PathGuardEvidence:
    verdict = _path_guard_verdict(
        binding=binding,
        start_positions=start_positions,
        end_positions=end_positions,
        radii=radii,
        drift_orbits=drift_orbits,
        spec=spec,
    )
    if verdict.public_error is not None:
        raise verdict.public_error
    assert verdict.evidence is not None
    return verdict.evidence


def _canonical_content(value: object) -> Any:
    """Return an exact, finite, JSON-safe representation of retained content."""

    if value is None:
        return None
    if type(value) is bool or type(value) is int or type(value) is str:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise TrajectoryContractError(
                "Wisdom--Holman checksum content must contain only finite floats"
            )
        return {"binary64_hex": value.hex()}
    if type(value) is tuple:
        return [_canonical_content(item) for item in value]
    if type(value) is list:
        return [_canonical_content(item) for item in value]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TrajectoryContractError(
                "Wisdom--Holman checksum mappings require built-in string keys"
            )
        return {key: _canonical_content(item) for key, item in value.items()}
    if type(value) is np.ndarray:
        if value.dtype == np.dtype(np.float64):
            values: Any = _array_hex(value)
        elif value.dtype == np.dtype(np.bool_):
            values = [bool(item) for item in value]
        else:
            raise TrajectoryContractError(
                "Wisdom--Holman checksum arrays must be float64 or bool"
            )
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": values,
        }
    if is_dataclass(value) and type(value).__module__.startswith("jxplanetx."):
        return {
            "dataclass": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                field.name: _canonical_content(getattr(value, field.name))
                for field in fields(value)
            },
        }
    raise TrajectoryContractError(
        f"unsupported retained checksum content type {type(value).__name__!r}"
    )


def _domain_separated_json_sha256(domain: str, payload: object) -> str:
    serialized = json.dumps(
        _canonical_content(payload),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + serialized).hexdigest()


_EXECUTION_COUNT_NAMES = (
    "force_evaluations",
    "interaction_force_assemblies",
    "kepler_subflow_solves",
    "universal_solver_iterations",
    "universal_solver_bracket_expansions",
    "universal_g_function_evaluations",
    "universal_series_terms_evaluated",
    "coordinate_forward_transforms",
    "coordinate_inverse_transforms",
    "node_guard_evaluations",
    "path_guard_evaluations",
)


def _execution_accounting_payload(
    primary_counts: tuple[int, ...],
) -> dict[str, object]:
    if (
        type(primary_counts) is not tuple
        or len(primary_counts) != len(_EXECUTION_COUNT_NAMES)
        or any(type(value) is not int or value < 0 for value in primary_counts)
    ):
        raise TrajectoryContractError(
            "primary execution counts must be one exact nonnegative integer per fixed counter"
        )
    return {
        "scope": WH_PUBLIC_EXECUTION_ACCOUNTING_SCOPE,
        "validation_replay_count": WH_VALIDATION_REPLAY_COUNT,
        "validation_replay_policy": WH_VALIDATION_REPLAY_POLICY,
        "counts": {
            name: {
                "primary_map": value,
                "validation_replay": value,
                "total_public_call": 2 * value,
            }
            for name, value in zip(_EXECUTION_COUNT_NAMES, primary_counts)
        },
    }


def _schedule_content_sha256(
    *,
    checkpoint_step_indices: tuple[int, ...],
    checkpoint_epochs: tuple[float, ...],
    fixed_step: float,
    direction: str,
    completed_steps: int,
    body_count: int,
) -> str:
    """Return the unauthenticated checksum of the exact fixed map schedule."""

    if type(body_count) is not int or body_count < 2:
        raise TrajectoryContractError(
            "schedule body_count must be an exact integer of at least two"
        )
    primary_static_counts = {
        "force_evaluations": completed_steps + 1,
        "interaction_force_assemblies": completed_steps + 1,
        "kepler_subflow_solves": completed_steps * (body_count - 1),
        "coordinate_forward_transforms": 1,
        "coordinate_inverse_transforms": 2 * completed_steps,
        "node_guard_evaluations": 1 + 2 * completed_steps,
        "path_guard_evaluations": completed_steps,
    }
    payload = {
        "body_count": body_count,
        "checkpoint_epochs_hex": [value.hex() for value in checkpoint_epochs],
        "checkpoint_step_indices": list(checkpoint_step_indices),
        "checksum_algorithm": WH_SCHEDULE_CHECKSUM_ALGORITHM,
        "completed_steps": completed_steps,
        "direction": direction,
        "fixed_step_hex": fixed_step.hex(),
        "method_id": FIXED_STEP_WISDOM_HOLMAN_METHOD_ID,
        "public_execution_accounting_scope": WH_PUBLIC_EXECUTION_ACCOUNTING_SCOPE,
        "static_expected_counts": {
            name: {
                "primary_map": value,
                "validation_replay": value,
                "total_public_call": 2 * value,
            }
            for name, value in primary_static_counts.items()
        },
        "validation_replay_count": WH_VALIDATION_REPLAY_COUNT,
        "validation_replay_policy": WH_VALIDATION_REPLAY_POLICY,
    }
    serialized = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    preimage = WH_SCHEDULE_CHECKSUM_DOMAIN.encode("utf-8") + b"\x00" + serialized
    return hashlib.sha256(preimage).hexdigest()


@dataclass(frozen=True, eq=False)
class _StrictFarProbeWork:
    """Private work custody for public WH systems outside the hybrid cap."""

    force_calls_entered: int
    force_evaluations_completed: int
    interaction_force_assemblies: int
    kepler_subflow_calls_entered: int
    kepler_subflow_solves_completed: int
    universal_solver_iterations: int
    universal_solver_bracket_expansions: int
    universal_g_bundle_calls_entered: int
    universal_g_bundle_calls_completed: int
    universal_series_terms_evaluated: int
    kepler_probe_records: tuple[_StrictKeplerProbeRecord, ...]
    cartesian_to_jacobi_calls_entered: int
    cartesian_to_jacobi_transforms_completed: int
    jacobi_to_cartesian_calls_entered: int
    jacobi_to_cartesian_transforms_completed: int
    node_guard_evaluations: int
    path_guard_evaluations: int
    first_half_kicks_completed: int
    center_of_mass_drifts_completed: int
    second_half_kicks_completed: int


@dataclass(frozen=True, eq=False)
class _StrictFarProbeDecision:
    """Private strict-WH decision when hybrid's 16-body schema cannot apply."""

    body_count: int
    outcome: str
    reason: str
    phase: str
    body_indices: tuple[int, ...]
    pair_indices: tuple[tuple[int, int], ...]
    work: _StrictFarProbeWork


_FarProbeRecord = HybridKeplerProbeRecord | _StrictKeplerProbeRecord
_FarProbeWork = HybridFarProbeWork | _StrictFarProbeWork
_FarProbeDecision = HybridFarProbeDecision | _StrictFarProbeDecision


@dataclass
class _WisdomHolmanProbeWorkBuilder:
    """Mutable transaction-local work sink frozen at one terminal outcome."""

    force_calls_entered: int = 0
    force_evaluations_completed: int = 0
    interaction_force_assemblies: int = 0
    kepler_probe_records: list[_FarProbeRecord] = field(default_factory=list)
    cartesian_to_jacobi_calls_entered: int = 0
    cartesian_to_jacobi_transforms_completed: int = 0
    jacobi_to_cartesian_calls_entered: int = 0
    jacobi_to_cartesian_transforms_completed: int = 0
    node_guard_evaluations: int = 0
    path_guard_evaluations: int = 0
    first_half_kicks_completed: int = 0
    center_of_mass_drifts_completed: int = 0
    second_half_kicks_completed: int = 0

    @classmethod
    def from_cache(
        cls, cache: _WisdomHolmanContinuousCache
    ) -> _WisdomHolmanProbeWorkBuilder:
        builder = cls()
        if cache.rebind_work_pending:
            builder.force_calls_entered = 1
            builder.force_evaluations_completed = 1
            builder.interaction_force_assemblies = 1
            builder.cartesian_to_jacobi_calls_entered = 1
            builder.cartesian_to_jacobi_transforms_completed = 1
            builder.node_guard_evaluations = 1
        return builder

    def freeze(self, *, body_count: int) -> _FarProbeWork:
        records = tuple(self.kepler_probe_records)
        values: dict[str, object] = {
            "force_calls_entered": self.force_calls_entered,
            "force_evaluations_completed": self.force_evaluations_completed,
            "interaction_force_assemblies": self.interaction_force_assemblies,
            "kepler_subflow_calls_entered": len(records),
            "kepler_subflow_solves_completed": sum(
                record.call_completed for record in records
            ),
            "universal_solver_iterations": sum(
                record.iterations_entered for record in records
            ),
            "universal_solver_bracket_expansions": sum(
                record.bracket_expansions_entered for record in records
            ),
            "universal_g_bundle_calls_entered": sum(
                record.g_bundle_calls_entered for record in records
            ),
            "universal_g_bundle_calls_completed": sum(
                record.g_bundle_calls_completed for record in records
            ),
            "universal_series_terms_evaluated": sum(
                record.series_terms_evaluated for record in records
            ),
            "kepler_probe_records": records,
            "cartesian_to_jacobi_calls_entered": (
                self.cartesian_to_jacobi_calls_entered
            ),
            "cartesian_to_jacobi_transforms_completed": (
                self.cartesian_to_jacobi_transforms_completed
            ),
            "jacobi_to_cartesian_calls_entered": (
                self.jacobi_to_cartesian_calls_entered
            ),
            "jacobi_to_cartesian_transforms_completed": (
                self.jacobi_to_cartesian_transforms_completed
            ),
            "node_guard_evaluations": self.node_guard_evaluations,
            "path_guard_evaluations": self.path_guard_evaluations,
            "first_half_kicks_completed": self.first_half_kicks_completed,
            "center_of_mass_drifts_completed": (
                self.center_of_mass_drifts_completed
            ),
            "second_half_kicks_completed": self.second_half_kicks_completed,
        }
        if body_count <= ENCOUNTER_HARD_MAXIMUM_BODY_COUNT:
            return HybridFarProbeWork(**values)  # type: ignore[arg-type]
        return _StrictFarProbeWork(**values)  # type: ignore[arg-type]


_WH_CAPABILITY_FACTORY_SENTINEL = object()
_WH_FORCE_CUSTODY_CONTENT_DOMAIN = "jxplanetx.private.wh-force-custody.v1"
_WH_FORCE_CUSTODY_SNAPSHOT_CONTEXT_DOMAIN = (
    "jxplanetx.private.wh-force-custody-snapshot-context.v1"
)
_WH_FORCE_CUSTODY_ORIGIN_SNAPSHOT_DOMAIN = (
    "jxplanetx.private.wh-force-custody-origin-snapshot.v1"
)
_WH_CACHE_CONTENT_DOMAIN = "jxplanetx.private.wh-continuous-cache.v1"
_WH_CANDIDATE_CONTENT_DOMAIN = "jxplanetx.private.wh-macrostep-candidate.v1"
# These digests are unauthenticated internal integrity seals, not adversarial
# authenticity claims.  They reject ordinary/stale capability mutation; exact
# schema, execution-context, lattice, and provenance checks below remain
# independently mandatory even if an internal caller recomputes a seal.


class _WisdomHolmanForceCustody:
    """Sealed first-evaluation identity retained across private WH rebinds."""

    __slots__ = (
        "force_model_ids",
        "force_ledger",
        "origin_snapshot",
        "origin_snapshot_id",
        "origin_snapshot_epoch",
        "origin_snapshot_context_sha256",
        "origin_snapshot_content_sha256",
        "origin_force_plan",
        "origin_integration_spec",
        "origin_binding",
        "content_sha256",
    )

    def __init__(
        self,
        *,
        factory_sentinel: object,
        force_model_ids: tuple[str, ...],
        force_ledger: tuple[ForceLedgerEntry, ...],
        origin_snapshot: StateSnapshot,
        origin_snapshot_id: str,
        origin_snapshot_epoch: float,
        origin_snapshot_context_sha256: str,
        origin_snapshot_content_sha256: str,
        origin_force_plan: ForcePlan,
        origin_integration_spec: FixedStepWisdomHolmanSpec,
        origin_binding: JacobiCoordinateBinding,
        content_sha256: str,
    ) -> None:
        if factory_sentinel is not _WH_CAPABILITY_FACTORY_SENTINEL:
            raise TrajectoryContractError(
                "WH force custody must be created by its private factory"
            )
        object.__setattr__(self, "force_model_ids", force_model_ids)
        object.__setattr__(self, "force_ledger", force_ledger)
        object.__setattr__(self, "origin_snapshot", origin_snapshot)
        object.__setattr__(self, "origin_snapshot_id", origin_snapshot_id)
        object.__setattr__(self, "origin_snapshot_epoch", origin_snapshot_epoch)
        object.__setattr__(
            self,
            "origin_snapshot_context_sha256",
            origin_snapshot_context_sha256,
        )
        object.__setattr__(
            self,
            "origin_snapshot_content_sha256",
            origin_snapshot_content_sha256,
        )
        object.__setattr__(self, "origin_force_plan", origin_force_plan)
        object.__setattr__(
            self, "origin_integration_spec", origin_integration_spec
        )
        object.__setattr__(self, "origin_binding", origin_binding)
        object.__setattr__(self, "content_sha256", content_sha256)

    def __setattr__(self, name: str, value: object) -> None:
        raise TrajectoryContractError("WH force custody is immutable")


class _WisdomHolmanContinuousCache:
    """Sealed accepted synchronized WH state used by one proposal chain."""

    __slots__ = (
        "positions",
        "velocities",
        "coordinates",
        "momenta",
        "interaction_force",
        "accepted_outer_step_index",
        "accepted_epoch",
        "force_custody",
        "rebind_work_pending",
        "accepted_start_guard",
        "accepted_start_translation_residual",
        "authoritative_snapshot",
        "force_plan",
        "integration_spec",
        "binding",
        "content_sha256",
    )

    def __init__(
        self,
        *,
        factory_sentinel: object,
        positions: np.ndarray,
        velocities: np.ndarray,
        coordinates: np.ndarray,
        momenta: np.ndarray,
        interaction_force: np.ndarray,
        accepted_outer_step_index: int,
        accepted_epoch: float,
        force_custody: _WisdomHolmanForceCustody,
        rebind_work_pending: bool,
        accepted_start_guard: _NodeGuardEvidence | None,
        accepted_start_translation_residual: float,
        authoritative_snapshot: StateSnapshot,
        force_plan: ForcePlan,
        integration_spec: FixedStepWisdomHolmanSpec,
        binding: JacobiCoordinateBinding,
        content_sha256: str,
    ) -> None:
        if factory_sentinel is not _WH_CAPABILITY_FACTORY_SENTINEL:
            raise TrajectoryContractError(
                "WH continuous cache must be created by its private factory"
            )
        for name, value in (
            ("positions", positions),
            ("velocities", velocities),
            ("coordinates", coordinates),
            ("momenta", momenta),
            ("interaction_force", interaction_force),
            ("accepted_outer_step_index", accepted_outer_step_index),
            ("accepted_epoch", accepted_epoch),
            ("force_custody", force_custody),
            ("rebind_work_pending", rebind_work_pending),
            ("accepted_start_guard", accepted_start_guard),
            (
                "accepted_start_translation_residual",
                accepted_start_translation_residual,
            ),
            ("authoritative_snapshot", authoritative_snapshot),
            ("force_plan", force_plan),
            ("integration_spec", integration_spec),
            ("binding", binding),
            ("content_sha256", content_sha256),
        ):
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: object) -> None:
        raise TrajectoryContractError("WH continuous cache is immutable")

    @property
    def force_model_ids(self) -> tuple[str, ...]:
        return self.force_custody.force_model_ids

    @property
    def force_ledger(self) -> tuple[ForceLedgerEntry, ...]:
        return self.force_custody.force_ledger


class _WisdomHolmanMacrostepCandidate:
    """Sealed fully checked payload, present only for a FAR_PASS."""

    __slots__ = (
        "positions",
        "velocities",
        "coordinates",
        "momenta",
        "interaction_force",
        "outer_step_index",
        "epoch",
        "drift_guard",
        "completed_guard",
        "path_evidence",
        "solver_records",
        "translation_residual",
        "source_cache",
        "signed_fixed_step",
        "content_sha256",
    )

    def __init__(
        self,
        *,
        factory_sentinel: object,
        positions: np.ndarray,
        velocities: np.ndarray,
        coordinates: np.ndarray,
        momenta: np.ndarray,
        interaction_force: np.ndarray,
        outer_step_index: int,
        epoch: float,
        drift_guard: _NodeGuardEvidence,
        completed_guard: _NodeGuardEvidence,
        path_evidence: _PathGuardEvidence,
        solver_records: tuple[_UniversalKeplerStep, ...],
        translation_residual: float,
        source_cache: _WisdomHolmanContinuousCache,
        signed_fixed_step: float,
        content_sha256: str,
    ) -> None:
        if factory_sentinel is not _WH_CAPABILITY_FACTORY_SENTINEL:
            raise TrajectoryContractError(
                "WH macrostep candidate must be created by its private factory"
            )
        for name, value in (
            ("positions", positions),
            ("velocities", velocities),
            ("coordinates", coordinates),
            ("momenta", momenta),
            ("interaction_force", interaction_force),
            ("outer_step_index", outer_step_index),
            ("epoch", epoch),
            ("drift_guard", drift_guard),
            ("completed_guard", completed_guard),
            ("path_evidence", path_evidence),
            ("solver_records", solver_records),
            ("translation_residual", translation_residual),
            ("source_cache", source_cache),
            ("signed_fixed_step", signed_fixed_step),
            ("content_sha256", content_sha256),
        ):
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: object) -> None:
        raise TrajectoryContractError("WH macrostep candidate is immutable")


@dataclass(frozen=True, eq=False)
class _WisdomHolmanMacrostepProposal:
    """Typed terminal decision plus an all-or-nothing candidate payload."""

    decision: _FarProbeDecision
    candidate: _WisdomHolmanMacrostepCandidate | None
    public_error: Exception | None


@dataclass(frozen=True, eq=False)
class _WisdomHolmanCachePreparation:
    """Initial/rebind cache preparation, or its typed terminal failure."""

    cache: _WisdomHolmanContinuousCache | None
    terminal: _WisdomHolmanMacrostepProposal | None
    initial_barycenter_position_norm: float
    initial_barycenter_velocity_norm: float
    initial_barycenter_position_cap: float
    initial_barycenter_velocity_cap: float


def _same_binary64(left: object, right: object) -> bool:
    return (
        type(left) is float
        and type(right) is float
        and left.hex() == right.hex()
    )


def _force_custody_snapshot_context_sha256(snapshot: StateSnapshot) -> str:
    return _domain_separated_json_sha256(
        _WH_FORCE_CUSTODY_SNAPSHOT_CONTEXT_DOMAIN,
        {
            "snapshot_id": snapshot.snapshot_id,
            "time_scale": snapshot.time_scale,
            "frame": snapshot.frame,
            "origin": snapshot.origin,
            "axes": snapshot.axes,
            "length_unit": snapshot.length_unit,
            "time_unit": snapshot.time_unit,
            "mass_unit": snapshot.mass_unit,
            "unit_system_id": snapshot.unit_system_id,
            "body_ids": snapshot.body_ids,
            "gravitational_parameters": snapshot.gravitational_parameters,
            "masses": snapshot.masses,
            "radii": snapshot.radii,
            "massive": snapshot.massive,
            "provenance": snapshot.provenance,
        },
    )


def _force_custody_origin_snapshot_sha256(snapshot: StateSnapshot) -> str:
    return _domain_separated_json_sha256(
        _WH_FORCE_CUSTODY_ORIGIN_SNAPSHOT_DOMAIN,
        snapshot,
    )


def _force_custody_content_sha256(
    *,
    force_model_ids: tuple[str, ...],
    force_ledger: tuple[ForceLedgerEntry, ...],
    origin_snapshot: StateSnapshot,
    origin_snapshot_id: str,
    origin_snapshot_epoch: float,
    origin_snapshot_context_sha256: str,
    origin_snapshot_content_sha256: str,
    origin_force_plan: ForcePlan,
    origin_integration_spec: FixedStepWisdomHolmanSpec,
    origin_binding: JacobiCoordinateBinding,
) -> str:
    return _domain_separated_json_sha256(
        _WH_FORCE_CUSTODY_CONTENT_DOMAIN,
        {
            "force_model_ids": force_model_ids,
            "force_ledger": force_ledger,
            "origin_snapshot_identity": id(origin_snapshot),
            "origin_snapshot_id": origin_snapshot_id,
            "origin_snapshot_epoch": origin_snapshot_epoch,
            "origin_snapshot_context_sha256": origin_snapshot_context_sha256,
            "origin_snapshot_content_sha256": origin_snapshot_content_sha256,
            "origin_force_plan_identity": id(origin_force_plan),
            "origin_integration_spec_identity": id(origin_integration_spec),
            "origin_binding_identity": id(origin_binding),
            "origin_binding_content_sha256": (
                origin_binding.binding_content_sha256
            ),
        },
    )


def _force_custody_matches_authoritative_evaluation(
    custody: _WisdomHolmanForceCustody,
    evaluation_ledger: object,
) -> bool:
    """Compare every ledger bit after restoring the retained origin epoch."""

    if type(evaluation_ledger) is not tuple or (
        len(evaluation_ledger) != len(custody.force_ledger)
    ):
        return False
    normalized: list[ForceLedgerEntry] = []
    for entry in evaluation_ledger:
        checked = _validate_ledger_schema(entry)
        normalized.append(
            replace(
                checked,
                state_metadata=replace(
                    checked.state_metadata,
                    epoch=custody.origin_snapshot_epoch,
                ),
            )
        )
    return _canonical_content(tuple(normalized)) == _canonical_content(
        custody.force_ledger
    )


def _validate_force_custody(
    value: object,
    *,
    initial_snapshot: StateSnapshot,
    force_plan: ForcePlan,
    integration_spec: FixedStepWisdomHolmanSpec,
    binding: JacobiCoordinateBinding,
) -> _WisdomHolmanForceCustody:
    if type(value) is not _WisdomHolmanForceCustody:
        raise TrajectoryContractError(
            "WH retained force custody must have its exact private type"
        )
    if (
        value.origin_force_plan is not force_plan
        or value.origin_integration_spec is not integration_spec
        or value.origin_binding is not binding
    ):
        raise TrajectoryContractError(
            "WH retained force custody belongs to a different execution context"
        )
    origin_snapshot = _validate_snapshot_schema(
        value.origin_snapshot, "WH force-custody origin_snapshot"
    )
    if (
        type(value.origin_snapshot_id) is not str
        or not value.origin_snapshot_id
        or value.origin_snapshot_id != origin_snapshot.snapshot_id
        or value.origin_snapshot_id != initial_snapshot.snapshot_id
        or type(value.origin_snapshot_epoch) is not float
        or not math.isfinite(value.origin_snapshot_epoch)
        or not _same_binary64(
            value.origin_snapshot_epoch, origin_snapshot.epoch
        )
    ):
        raise TrajectoryContractError(
            "WH retained force custody lost its originating snapshot label"
        )
    current_snapshot_context = _force_custody_snapshot_context_sha256(
        initial_snapshot
    )
    origin_snapshot_context = _force_custody_snapshot_context_sha256(
        origin_snapshot
    )
    if (
        type(value.origin_snapshot_context_sha256) is not str
        or len(value.origin_snapshot_context_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value.origin_snapshot_context_sha256
        )
        or value.origin_snapshot_context_sha256 != origin_snapshot_context
        or value.origin_snapshot_context_sha256 != current_snapshot_context
    ):
        raise TrajectoryContractError(
            "WH retained force custody belongs to a different snapshot context"
        )
    origin_snapshot_content = _force_custody_origin_snapshot_sha256(
        origin_snapshot
    )
    if (
        type(value.origin_snapshot_content_sha256) is not str
        or len(value.origin_snapshot_content_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value.origin_snapshot_content_sha256
        )
        or value.origin_snapshot_content_sha256 != origin_snapshot_content
    ):
        raise TrajectoryContractError(
            "WH retained force custody lost its authoritative origin snapshot"
        )
    if type(value.force_model_ids) is not tuple or not value.force_model_ids:
        raise TrajectoryContractError(
            "WH retained force-model identity must be a nonempty exact tuple"
        )
    if type(value.force_ledger) is not tuple or (
        len(value.force_ledger) != len(value.force_model_ids)
    ):
        raise TrajectoryContractError(
            "WH retained force ledger must be an aligned exact tuple"
        )
    for model_id in value.force_model_ids:
        if type(model_id) is not str or not model_id:
            raise TrajectoryContractError(
                "WH retained force-model IDs must be nonempty built-in strings"
            )
    for index, entry in enumerate(value.force_ledger):
        checked = _validate_ledger_schema(entry)
        if (
            checked.order != index
            or checked.model_id != value.force_model_ids[index]
            or checked.source_ids != binding.body_ids
            or checked.target_ids != binding.body_ids
            or checked.state_metadata.snapshot_id != value.origin_snapshot_id
            or not _same_binary64(
                checked.state_metadata.epoch, value.origin_snapshot_epoch
            )
        ):
            raise TrajectoryContractError(
                "WH retained force ledger is not bound to its canonical body/model order"
            )
    if not _same_binary64(
        value.origin_snapshot_epoch,
        value.force_ledger[0].state_metadata.epoch,
    ):
        raise TrajectoryContractError(
            "WH retained force custody lost its originating snapshot epoch"
        )
    expected_content_sha256 = _force_custody_content_sha256(
        force_model_ids=value.force_model_ids,
        force_ledger=value.force_ledger,
        origin_snapshot=value.origin_snapshot,
        origin_snapshot_id=value.origin_snapshot_id,
        origin_snapshot_epoch=value.origin_snapshot_epoch,
        origin_snapshot_context_sha256=value.origin_snapshot_context_sha256,
        origin_snapshot_content_sha256=(
            value.origin_snapshot_content_sha256
        ),
        origin_force_plan=value.origin_force_plan,
        origin_integration_spec=value.origin_integration_spec,
        origin_binding=value.origin_binding,
    )
    if (
        type(value.content_sha256) is not str
        or len(value.content_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value.content_sha256
        )
        or value.content_sha256 != expected_content_sha256
    ):
        raise TrajectoryContractError(
            "WH retained force custody does not match its sealed content"
        )
    return value


def _make_force_custody(
    *,
    force_model_ids: tuple[str, ...],
    force_ledger: tuple[ForceLedgerEntry, ...],
    initial_snapshot: StateSnapshot,
    force_plan: ForcePlan,
    integration_spec: FixedStepWisdomHolmanSpec,
    binding: JacobiCoordinateBinding,
) -> _WisdomHolmanForceCustody:
    snapshot_context_sha256 = _force_custody_snapshot_context_sha256(
        initial_snapshot
    )
    snapshot_content_sha256 = _force_custody_origin_snapshot_sha256(
        initial_snapshot
    )
    content_sha256 = _force_custody_content_sha256(
        force_model_ids=force_model_ids,
        force_ledger=force_ledger,
        origin_snapshot=initial_snapshot,
        origin_snapshot_id=initial_snapshot.snapshot_id,
        origin_snapshot_epoch=initial_snapshot.epoch,
        origin_snapshot_context_sha256=snapshot_context_sha256,
        origin_snapshot_content_sha256=snapshot_content_sha256,
        origin_force_plan=force_plan,
        origin_integration_spec=integration_spec,
        origin_binding=binding,
    )
    custody = _WisdomHolmanForceCustody(
        factory_sentinel=_WH_CAPABILITY_FACTORY_SENTINEL,
        force_model_ids=force_model_ids,
        force_ledger=force_ledger,
        origin_snapshot=initial_snapshot,
        origin_snapshot_id=initial_snapshot.snapshot_id,
        origin_snapshot_epoch=initial_snapshot.epoch,
        origin_snapshot_context_sha256=snapshot_context_sha256,
        origin_snapshot_content_sha256=snapshot_content_sha256,
        origin_force_plan=force_plan,
        origin_integration_spec=integration_spec,
        origin_binding=binding,
        content_sha256=content_sha256,
    )
    return _validate_force_custody(
        custody,
        initial_snapshot=initial_snapshot,
        force_plan=force_plan,
        integration_spec=integration_spec,
        binding=binding,
    )


def _validate_capability_array(
    value: object, label: str, shape: tuple[int, ...]
) -> np.ndarray:
    array = _validate_float64_array(
        value, label, shape, owned_readonly=True
    )
    if not array.flags.c_contiguous:
        raise TrajectoryContractError(f"{label} must be C-contiguous")
    return array


def _validate_disjoint_arrays(
    values: tuple[np.ndarray, ...], *, label: str
) -> None:
    for left_index, left in enumerate(values):
        for right in values[left_index + 1 :]:
            if np.shares_memory(left, right):
                raise TrajectoryContractError(
                    f"{label} arrays must not share memory"
                )


def _validate_capability_float_tuple(
    value: object,
    *,
    label: str,
    length: int,
    positive: bool,
) -> tuple[float, ...]:
    if type(value) is not tuple or len(value) != length:
        raise TrajectoryContractError(
            f"{label} must be an exact tuple with {length} entries"
        )
    for item in value:
        if (
            type(item) is not float
            or not math.isfinite(item)
            or (item <= 0.0 if positive else item < 0.0)
        ):
            qualifier = "positive" if positive else "nonnegative"
            raise TrajectoryContractError(
                f"{label} entries must be finite {qualifier} built-in floats"
            )
    return value


def _validate_node_guard_evidence(
    value: object, *, body_count: int, label: str
) -> _NodeGuardEvidence:
    if type(value) is not _NodeGuardEvidence:
        raise TrajectoryContractError(f"{label} must have exact guard-evidence type")
    secondary_count = body_count - 1
    pair_count = body_count * (body_count - 1) // 2
    hill_pair_count = secondary_count * (secondary_count - 1) // 2
    for name in (
        "semimajor_axes",
        "periapses",
        "periods",
        "periapse_timescales",
    ):
        _validate_capability_float_tuple(
            getattr(value, name),
            label=f"{label}.{name}",
            length=secondary_count,
            positive=True,
        )
    for name in (
        "eccentricities",
        "interaction_force_ratios",
        "orbit_step_fractions",
        "periapse_step_fractions",
    ):
        _validate_capability_float_tuple(
            getattr(value, name),
            label=f"{label}.{name}",
            length=secondary_count,
            positive=False,
        )
    _validate_capability_float_tuple(
        value.pair_separations,
        label=f"{label}.pair_separations",
        length=pair_count,
        positive=True,
    )
    _validate_capability_float_tuple(
        value.secondary_hill_floor_ratios,
        label=f"{label}.secondary_hill_floor_ratios",
        length=hill_pair_count,
        positive=True,
    )
    for name in ("barycenter_position_norm", "barycenter_velocity_norm"):
        candidate = getattr(value, name)
        if type(candidate) is not float or not math.isfinite(candidate) or candidate < 0.0:
            raise TrajectoryContractError(
                f"{label}.{name} must be a finite nonnegative built-in float"
            )
    return value


def _validate_path_guard_evidence(
    value: object, *, body_count: int
) -> _PathGuardEvidence:
    if type(value) is not _PathGuardEvidence:
        raise TrajectoryContractError(
            "WH candidate path evidence must have its exact private type"
        )
    pair_count = body_count * (body_count - 1) // 2
    for name in (
        "endpoint_minimum_separations",
        "path_lower_bounds",
        "clearance_after_margins",
    ):
        _validate_capability_float_tuple(
            getattr(value, name),
            label=f"candidate.path_evidence.{name}",
            length=pair_count,
            positive=True,
        )
    return value


def _cache_content_sha256(
    *,
    positions: np.ndarray,
    velocities: np.ndarray,
    coordinates: np.ndarray,
    momenta: np.ndarray,
    interaction_force: np.ndarray,
    accepted_outer_step_index: int,
    accepted_epoch: float,
    force_custody: _WisdomHolmanForceCustody,
    rebind_work_pending: bool,
    accepted_start_guard: _NodeGuardEvidence | None,
    accepted_start_translation_residual: float,
    authoritative_snapshot: StateSnapshot,
    force_plan: ForcePlan,
    integration_spec: FixedStepWisdomHolmanSpec,
    binding: JacobiCoordinateBinding,
) -> str:
    return _domain_separated_json_sha256(
        _WH_CACHE_CONTENT_DOMAIN,
        {
            "positions": positions,
            "velocities": velocities,
            "coordinates": coordinates,
            "momenta": momenta,
            "interaction_force": interaction_force,
            "accepted_outer_step_index": accepted_outer_step_index,
            "accepted_epoch": accepted_epoch,
            "force_model_ids": force_custody.force_model_ids,
            "force_ledger": force_custody.force_ledger,
            "force_custody_content_sha256": force_custody.content_sha256,
            "rebind_work_pending": rebind_work_pending,
            "accepted_start_guard": accepted_start_guard,
            "accepted_start_translation_residual": (
                accepted_start_translation_residual
            ),
            "authoritative_snapshot_identity": id(authoritative_snapshot),
            "force_plan_identity": id(force_plan),
            "integration_spec_identity": id(integration_spec),
            "binding_identity": id(binding),
            "binding_content_sha256": binding.binding_content_sha256,
        },
    )


def _make_continuous_cache(
    *,
    positions: np.ndarray,
    velocities: np.ndarray,
    coordinates: np.ndarray,
    momenta: np.ndarray,
    interaction_force: np.ndarray,
    accepted_outer_step_index: int,
    accepted_epoch: float,
    force_custody: _WisdomHolmanForceCustody,
    rebind_work_pending: bool,
    accepted_start_guard: _NodeGuardEvidence | None,
    accepted_start_translation_residual: float,
    authoritative_snapshot: StateSnapshot,
    force_plan: ForcePlan,
    integration_spec: FixedStepWisdomHolmanSpec,
    binding: JacobiCoordinateBinding,
) -> _WisdomHolmanContinuousCache:
    content_sha256 = _cache_content_sha256(
        positions=positions,
        velocities=velocities,
        coordinates=coordinates,
        momenta=momenta,
        interaction_force=interaction_force,
        accepted_outer_step_index=accepted_outer_step_index,
        accepted_epoch=accepted_epoch,
        force_custody=force_custody,
        rebind_work_pending=rebind_work_pending,
        accepted_start_guard=accepted_start_guard,
        accepted_start_translation_residual=accepted_start_translation_residual,
        authoritative_snapshot=authoritative_snapshot,
        force_plan=force_plan,
        integration_spec=integration_spec,
        binding=binding,
    )
    return _WisdomHolmanContinuousCache(
        factory_sentinel=_WH_CAPABILITY_FACTORY_SENTINEL,
        positions=positions,
        velocities=velocities,
        coordinates=coordinates,
        momenta=momenta,
        interaction_force=interaction_force,
        accepted_outer_step_index=accepted_outer_step_index,
        accepted_epoch=accepted_epoch,
        force_custody=force_custody,
        rebind_work_pending=rebind_work_pending,
        accepted_start_guard=accepted_start_guard,
        accepted_start_translation_residual=accepted_start_translation_residual,
        authoritative_snapshot=authoritative_snapshot,
        force_plan=force_plan,
        integration_spec=integration_spec,
        binding=binding,
        content_sha256=content_sha256,
    )


def _validate_continuous_cache(
    value: object,
    *,
    initial_snapshot: StateSnapshot,
    force_plan: ForcePlan,
    integration_spec: FixedStepWisdomHolmanSpec,
    binding: JacobiCoordinateBinding,
    expected_outer_step_index: int,
    expected_epoch: float,
    expected_rebind_work_pending: bool,
) -> _WisdomHolmanContinuousCache:
    if type(value) is not _WisdomHolmanContinuousCache:
        raise TrajectoryContractError(
            "WH proposal cache must have its exact private capability type"
        )
    if (
        value.authoritative_snapshot is not initial_snapshot
        or value.force_plan is not force_plan
        or value.integration_spec is not integration_spec
        or value.binding is not binding
    ):
        raise TrajectoryContractError(
            "WH proposal cache is not bound to the supplied execution context"
        )
    if (
        type(expected_outer_step_index) is not int
        or expected_outer_step_index < 0
        or type(value.accepted_outer_step_index) is not int
        or value.accepted_outer_step_index < 0
        or value.accepted_outer_step_index != expected_outer_step_index
    ):
        raise TrajectoryContractError(
            "WH proposal cache has the wrong accepted outer-step index"
        )
    if not _same_binary64(value.accepted_epoch, expected_epoch):
        raise TrajectoryContractError(
            "WH proposal cache epoch does not match the outer lattice label"
        )
    if (
        type(expected_rebind_work_pending) is not bool
        or type(value.rebind_work_pending) is not bool
        or value.rebind_work_pending is not expected_rebind_work_pending
    ):
        raise TrajectoryContractError(
            "WH proposal cache has the wrong explicit rebind-work state"
        )
    body_count = len(binding.body_ids)
    retained_arrays = tuple(
        _validate_capability_array(getattr(value, name), f"cache.{name}", (body_count, 3))
        for name in (
            "positions",
            "velocities",
            "coordinates",
            "momenta",
            "interaction_force",
        )
    )
    _validate_disjoint_arrays(retained_arrays, label="WH cache")
    for cache_array in retained_arrays:
        for source_array in (initial_snapshot.positions, initial_snapshot.velocities):
            if np.shares_memory(cache_array, source_array):
                raise TrajectoryContractError(
                    "WH cache must not alias its authoritative snapshot"
                )
    _validate_force_custody(
        value.force_custody,
        initial_snapshot=initial_snapshot,
        force_plan=force_plan,
        integration_spec=integration_spec,
        binding=binding,
    )
    if value.rebind_work_pending:
        _validate_node_guard_evidence(
            value.accepted_start_guard,
            body_count=body_count,
            label="cache.accepted_start_guard",
        )
        if (
            type(value.accepted_start_translation_residual) is not float
            or not math.isfinite(value.accepted_start_translation_residual)
            or value.accepted_start_translation_residual < 0.0
        ):
            raise TrajectoryContractError(
                "WH rebind translation residual must be finite and nonnegative"
            )
    elif (
        value.accepted_start_guard is not None
        or not _same_binary64(value.accepted_start_translation_residual, 0.0)
    ):
        raise TrajectoryContractError(
            "continuous FAR cache must not retain consumed rebind evidence"
        )
    if (
        type(value.content_sha256) is not str
        or len(value.content_sha256) != 64
        or any(character not in "0123456789abcdef" for character in value.content_sha256)
    ):
        raise TrajectoryContractError(
            "WH cache content digest must be an exact lowercase SHA-256"
        )
    expected_digest = _cache_content_sha256(
        positions=value.positions,
        velocities=value.velocities,
        coordinates=value.coordinates,
        momenta=value.momenta,
        interaction_force=value.interaction_force,
        accepted_outer_step_index=value.accepted_outer_step_index,
        accepted_epoch=value.accepted_epoch,
        force_custody=value.force_custody,
        rebind_work_pending=value.rebind_work_pending,
        accepted_start_guard=value.accepted_start_guard,
        accepted_start_translation_residual=(
            value.accepted_start_translation_residual
        ),
        authoritative_snapshot=value.authoritative_snapshot,
        force_plan=value.force_plan,
        integration_spec=value.integration_spec,
        binding=value.binding,
    )
    if value.content_sha256 != expected_digest:
        raise TrajectoryContractError(
            "WH cache content does not match its sealed digest"
        )
    return value


def _candidate_content_sha256(
    *,
    positions: np.ndarray,
    velocities: np.ndarray,
    coordinates: np.ndarray,
    momenta: np.ndarray,
    interaction_force: np.ndarray,
    outer_step_index: int,
    epoch: float,
    drift_guard: _NodeGuardEvidence,
    completed_guard: _NodeGuardEvidence,
    path_evidence: _PathGuardEvidence,
    solver_records: tuple[_UniversalKeplerStep, ...],
    translation_residual: float,
    source_cache: _WisdomHolmanContinuousCache,
    signed_fixed_step: float,
) -> str:
    return _domain_separated_json_sha256(
        _WH_CANDIDATE_CONTENT_DOMAIN,
        {
            "positions": positions,
            "velocities": velocities,
            "coordinates": coordinates,
            "momenta": momenta,
            "interaction_force": interaction_force,
            "outer_step_index": outer_step_index,
            "epoch": epoch,
            "drift_guard": drift_guard,
            "completed_guard": completed_guard,
            "path_evidence": path_evidence,
            "solver_records": solver_records,
            "translation_residual": translation_residual,
            "source_cache_content_sha256": source_cache.content_sha256,
            "signed_fixed_step": signed_fixed_step,
        },
    )


def _make_macrostep_candidate(
    *,
    positions: np.ndarray,
    velocities: np.ndarray,
    coordinates: np.ndarray,
    momenta: np.ndarray,
    interaction_force: np.ndarray,
    outer_step_index: int,
    epoch: float,
    drift_guard: _NodeGuardEvidence,
    completed_guard: _NodeGuardEvidence,
    path_evidence: _PathGuardEvidence,
    solver_records: tuple[_UniversalKeplerStep, ...],
    translation_residual: float,
    source_cache: _WisdomHolmanContinuousCache,
    signed_fixed_step: float,
) -> _WisdomHolmanMacrostepCandidate:
    content_sha256 = _candidate_content_sha256(
        positions=positions,
        velocities=velocities,
        coordinates=coordinates,
        momenta=momenta,
        interaction_force=interaction_force,
        outer_step_index=outer_step_index,
        epoch=epoch,
        drift_guard=drift_guard,
        completed_guard=completed_guard,
        path_evidence=path_evidence,
        solver_records=solver_records,
        translation_residual=translation_residual,
        source_cache=source_cache,
        signed_fixed_step=signed_fixed_step,
    )
    return _WisdomHolmanMacrostepCandidate(
        factory_sentinel=_WH_CAPABILITY_FACTORY_SENTINEL,
        positions=positions,
        velocities=velocities,
        coordinates=coordinates,
        momenta=momenta,
        interaction_force=interaction_force,
        outer_step_index=outer_step_index,
        epoch=epoch,
        drift_guard=drift_guard,
        completed_guard=completed_guard,
        path_evidence=path_evidence,
        solver_records=solver_records,
        translation_residual=translation_residual,
        source_cache=source_cache,
        signed_fixed_step=signed_fixed_step,
        content_sha256=content_sha256,
    )


def _validate_completed_solver_record(
    value: object, *, secondary_index: int
) -> _UniversalKeplerStep:
    if type(value) is not _UniversalKeplerStep:
        raise TrajectoryContractError(
            "WH candidate solver records must have their exact private type"
        )
    position = _validate_capability_array(
        value.position, "candidate solver position", (3,)
    )
    velocity = _validate_capability_array(
        value.velocity, "candidate solver velocity", (3,)
    )
    if np.shares_memory(position, velocity):
        raise TrajectoryContractError(
            "WH candidate solver position/velocity must not alias"
        )
    for name in (
        "iterations",
        "bracket_expansions",
        "g_function_evaluations",
        "series_terms_evaluated",
    ):
        candidate = getattr(value, name)
        minimum = 1 if name in ("iterations", "g_function_evaluations") else 0
        if type(candidate) is not int or candidate < minimum:
            raise TrajectoryContractError(
                f"WH candidate solver {name} has an impossible value"
            )
    if value.series_terms_evaluated % (4 * WH_KEPLER_SERIES_TERM_COUNT) != 0:
        raise TrajectoryContractError(
            "completed WH solver series work must contain whole G bundles"
        )
    positive_fields = (
        "residual_tolerance",
        "beta",
        "semimajor_axis",
        "periapse",
        "period",
        "periapse_timescale",
    )
    nonnegative_fields = (
        "final_bracket_width",
        "eccentricity",
        "lagrange_identity_error",
        "energy_error",
        "angular_momentum_error",
    )
    signed_fields = ("time_residual", "universal_anomaly")
    for name in positive_fields:
        candidate = getattr(value, name)
        if type(candidate) is not float or not math.isfinite(candidate) or candidate <= 0.0:
            raise TrajectoryContractError(
                f"WH candidate solver {name} must be finite and positive"
            )
    for name in nonnegative_fields:
        candidate = getattr(value, name)
        if type(candidate) is not float or not math.isfinite(candidate) or candidate < 0.0:
            raise TrajectoryContractError(
                f"WH candidate solver {name} must be finite and nonnegative"
            )
    for name in signed_fields:
        candidate = getattr(value, name)
        if type(candidate) is not float or not math.isfinite(candidate):
            raise TrajectoryContractError(
                f"WH candidate solver {name} must be a finite built-in float"
            )
    return value


def _validate_strict_far_probe_decision(
    value: object, *, body_count: int
) -> _StrictFarProbeDecision:
    """Validate the exact private schema used beyond the hybrid body cap."""

    if type(value) is not _StrictFarProbeDecision:
        raise TrajectoryContractError(
            "strict WH FAR decision must have its exact private type"
        )
    if type(value.body_count) is not int or value.body_count != body_count:
        raise TrajectoryContractError(
            "strict WH FAR decision has the wrong exact body count"
        )
    for name in ("outcome", "reason", "phase"):
        item = getattr(value, name)
        if type(item) is not str or not item:
            raise TrajectoryContractError(
                f"strict WH FAR decision {name} must be a nonempty built-in string"
            )
    if type(value.body_indices) is not tuple or len(value.body_indices) > body_count:
        raise TrajectoryContractError(
            "strict WH FAR body triggers must be a bounded exact tuple"
        )
    previous_body = -1
    for body_index in value.body_indices:
        if (
            type(body_index) is not int
            or body_index < 0
            or body_index >= body_count
            or body_index <= previous_body
        ):
            raise TrajectoryContractError(
                "strict WH FAR body triggers must be sorted unique exact indices"
            )
        previous_body = body_index
    pair_count = body_count * (body_count - 1) // 2
    if type(value.pair_indices) is not tuple or len(value.pair_indices) > pair_count:
        raise TrajectoryContractError(
            "strict WH FAR pair triggers must be a bounded exact tuple"
        )
    previous_pair = (-1, -1)
    for pair in value.pair_indices:
        if type(pair) is not tuple or len(pair) != 2:
            raise TrajectoryContractError(
                "strict WH FAR pair triggers must contain exact pairs"
            )
        left, right = pair
        if (
            type(left) is not int
            or type(right) is not int
            or left < 0
            or left >= right
            or right >= body_count
            or pair <= previous_pair
        ):
            raise TrajectoryContractError(
                "strict WH FAR pairs must be sorted unique canonical indices"
            )
        previous_pair = pair
    work = value.work
    if type(work) is not _StrictFarProbeWork:
        raise TrajectoryContractError(
            "strict WH FAR work must have its exact private type"
        )
    for work_field in fields(_StrictFarProbeWork):
        if work_field.name == "kepler_probe_records":
            continue
        counter = getattr(work, work_field.name)
        if type(counter) is not int or counter < 0:
            raise TrajectoryContractError(
                "strict WH FAR work counters must be nonnegative built-in integers"
            )
    if (
        type(work.kepler_probe_records) is not tuple
        or len(work.kepler_probe_records) != body_count - 1
    ):
        raise TrajectoryContractError(
            "strict WH FAR work must retain one exact probe record per secondary"
        )
    for secondary_index, record in enumerate(
        work.kepler_probe_records, start=1
    ):
        if type(record) is not _StrictKeplerProbeRecord:
            raise TrajectoryContractError(
                "strict WH Kepler probe records must have their exact private type"
            )
        if (
            type(record.secondary_index) is not int
            or record.secondary_index != secondary_index
            or type(record.call_completed) is not bool
            or record.call_completed is not True
            or type(record.terminal_status) is not str
            or record.terminal_status != "COMPLETED"
        ):
            raise TrajectoryContractError(
                "strict WH FAR Kepler records must be canonical completed records"
            )
        for name in (
            "iterations_entered",
            "bracket_expansions_entered",
            "g_bundle_calls_entered",
            "g_bundle_calls_completed",
            "completed_series_bundle_count",
            "interrupted_series_terms",
        ):
            counter = getattr(record, name)
            if type(counter) is not int or counter < 0:
                raise TrajectoryContractError(
                    "strict WH Kepler counters must be nonnegative built-in integers"
                )
        if (
            record.iterations_entered < 1
            or record.iterations_entered > WH_KEPLER_MAXIMUM_ITERATIONS
            or record.bracket_expansions_entered
            > WH_KEPLER_MAXIMUM_BRACKET_EXPANSIONS
            or record.g_bundle_calls_entered
            != 1
            + record.iterations_entered
            + record.bracket_expansions_entered
            or record.g_bundle_calls_completed
            != record.g_bundle_calls_entered
            or record.completed_series_bundle_count
            > record.g_bundle_calls_completed
            or record.interrupted_series_terms != 0
        ):
            raise TrajectoryContractError(
                "strict WH FAR Kepler work does not match the completed solver cadence"
            )
    return value


def _validate_far_candidate_and_decision(
    *,
    cache: _WisdomHolmanContinuousCache,
    decision: object,
    candidate: object,
    integration_spec: FixedStepWisdomHolmanSpec,
    binding: JacobiCoordinateBinding,
    expected_outer_step_index: int,
    expected_epoch: float,
) -> _WisdomHolmanMacrostepCandidate:
    if type(candidate) is not _WisdomHolmanMacrostepCandidate:
        raise TrajectoryContractError(
            "WH commit candidate must have its exact private capability type"
        )
    if candidate.source_cache is not cache:
        raise TrajectoryContractError(
            "WH proposal provenance does not match the cache being committed"
        )
    if integration_spec is not cache.integration_spec or binding is not cache.binding:
        raise TrajectoryContractError(
            "WH commit context does not match the candidate source cache"
        )
    if (
        type(expected_outer_step_index) is not int
        or type(candidate.outer_step_index) is not int
        or candidate.outer_step_index <= 0
        or candidate.outer_step_index != expected_outer_step_index
        or expected_outer_step_index != cache.accepted_outer_step_index + 1
    ):
        raise TrajectoryContractError(
            "WH committed candidate must match the expected next outer-step index"
        )
    if not _same_binary64(candidate.epoch, expected_epoch):
        raise TrajectoryContractError(
            "WH committed candidate epoch does not match the outer lattice label"
        )
    if not _same_binary64(candidate.signed_fixed_step, integration_spec.fixed_step):
        raise TrajectoryContractError(
            "WH committed candidate step does not match the fixed WH step"
        )
    body_count = len(binding.body_ids)
    arrays = tuple(
        _validate_capability_array(
            getattr(candidate, name), f"candidate.{name}", (body_count, 3)
        )
        for name in (
            "positions",
            "velocities",
            "coordinates",
            "momenta",
            "interaction_force",
        )
    )
    _validate_disjoint_arrays(arrays, label="WH candidate")
    for candidate_array in arrays:
        for cache_array in (
            cache.positions,
            cache.velocities,
            cache.coordinates,
            cache.momenta,
            cache.interaction_force,
        ):
            if np.shares_memory(candidate_array, cache_array):
                raise TrajectoryContractError(
                    "WH candidate must not alias its source cache"
                )
    _validate_node_guard_evidence(
        candidate.drift_guard,
        body_count=body_count,
        label="candidate.drift_guard",
    )
    _validate_node_guard_evidence(
        candidate.completed_guard,
        body_count=body_count,
        label="candidate.completed_guard",
    )
    _validate_path_guard_evidence(candidate.path_evidence, body_count=body_count)
    if (
        type(candidate.translation_residual) is not float
        or not math.isfinite(candidate.translation_residual)
        or candidate.translation_residual < 0.0
    ):
        raise TrajectoryContractError(
            "WH candidate translation residual must be finite and nonnegative"
        )
    if type(candidate.solver_records) is not tuple or len(candidate.solver_records) != body_count - 1:
        raise TrajectoryContractError(
            "WH candidate must retain one completed solver record per secondary"
        )
    solver_arrays: list[np.ndarray] = []
    for secondary_index, solver_record in enumerate(candidate.solver_records, start=1):
        checked = _validate_completed_solver_record(
            solver_record, secondary_index=secondary_index
        )
        solver_arrays.extend((checked.position, checked.velocity))
    _validate_disjoint_arrays(tuple(solver_arrays), label="WH candidate solver")
    for solver_array in solver_arrays:
        for retained in arrays + (
            cache.positions,
            cache.velocities,
            cache.coordinates,
            cache.momenta,
            cache.interaction_force,
        ):
            if np.shares_memory(solver_array, retained):
                raise TrajectoryContractError(
                    "WH candidate solver records must have independent custody"
                )
    expected_digest = _candidate_content_sha256(
        positions=candidate.positions,
        velocities=candidate.velocities,
        coordinates=candidate.coordinates,
        momenta=candidate.momenta,
        interaction_force=candidate.interaction_force,
        outer_step_index=candidate.outer_step_index,
        epoch=candidate.epoch,
        drift_guard=candidate.drift_guard,
        completed_guard=candidate.completed_guard,
        path_evidence=candidate.path_evidence,
        solver_records=candidate.solver_records,
        translation_residual=candidate.translation_residual,
        source_cache=candidate.source_cache,
        signed_fixed_step=candidate.signed_fixed_step,
    )
    if (
        type(candidate.content_sha256) is not str
        or len(candidate.content_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in candidate.content_sha256
        )
        or candidate.content_sha256 != expected_digest
    ):
        raise TrajectoryContractError(
            "WH candidate content does not match its sealed digest"
        )

    expected_decision_type: type[object] = (
        HybridFarProbeDecision
        if body_count <= ENCOUNTER_HARD_MAXIMUM_BODY_COUNT
        else _StrictFarProbeDecision
    )
    if type(decision) is not expected_decision_type:
        raise TrajectoryContractError(
            "WH FAR decision has the wrong exact schema for its body count"
        )
    if type(decision) is HybridFarProbeDecision:
        decision.__post_init__()
    else:
        decision = _validate_strict_far_probe_decision(
            decision, body_count=body_count
        )
    if (
        decision.body_count != body_count
        or decision.outcome != "FAR_PASS"
        or decision.reason != HYBRID_ALL_FAR_REASON
        or decision.phase != "ALL_GUARDS_COMPLETED"
        or decision.body_indices != ()
        or decision.pair_indices != ()
    ):
        raise TrajectoryContractError(
            "WH candidate is not paired with the canonical all-far decision"
        )
    work = decision.work
    expected_work_type: type[object] = (
        HybridFarProbeWork
        if body_count <= ENCOUNTER_HARD_MAXIMUM_BODY_COUNT
        else _StrictFarProbeWork
    )
    if type(work) is not expected_work_type:
        raise TrajectoryContractError("WH FAR work has the wrong exact schema")
    rebind = 1 if cache.rebind_work_pending else 0
    exact_counts = {
        "force_calls_entered": rebind + 1,
        "force_evaluations_completed": rebind + 1,
        "interaction_force_assemblies": rebind + 1,
        "kepler_subflow_calls_entered": body_count - 1,
        "kepler_subflow_solves_completed": body_count - 1,
        "cartesian_to_jacobi_calls_entered": rebind,
        "cartesian_to_jacobi_transforms_completed": rebind,
        "jacobi_to_cartesian_calls_entered": 2,
        "jacobi_to_cartesian_transforms_completed": 2,
        "node_guard_evaluations": rebind + 2,
        "path_guard_evaluations": 1,
        "first_half_kicks_completed": 1,
        "center_of_mass_drifts_completed": 1,
        "second_half_kicks_completed": 1,
    }
    if any(getattr(work, name) != expected for name, expected in exact_counts.items()):
        raise TrajectoryContractError(
            "WH all-far work does not match the source-cache rebind cadence"
        )
    if type(work.kepler_probe_records) is not tuple or len(work.kepler_probe_records) != body_count - 1:
        raise TrajectoryContractError(
            "WH all-far work must retain every completed Kepler probe record"
        )
    for secondary_index, (probe, solver) in enumerate(
        zip(work.kepler_probe_records, candidate.solver_records), start=1
    ):
        expected_probe_type: type[object] = (
            HybridKeplerProbeRecord
            if body_count <= ENCOUNTER_HARD_MAXIMUM_BODY_COUNT
            else _StrictKeplerProbeRecord
        )
        if (
            type(probe) is not expected_probe_type
            or probe.secondary_index != secondary_index
            or probe.call_completed is not True
            or probe.terminal_status != "COMPLETED"
            or probe.iterations_entered != solver.iterations
            or probe.bracket_expansions_entered != solver.bracket_expansions
            or probe.g_bundle_calls_entered != solver.g_function_evaluations
            or probe.g_bundle_calls_completed != solver.g_function_evaluations
            or probe.series_terms_evaluated != solver.series_terms_evaluated
            or probe.interrupted_series_terms != 0
        ):
            raise TrajectoryContractError(
                "WH all-far probe work does not match its solver candidate"
            )
    aggregate_fields = {
        "universal_solver_iterations": sum(
            record.iterations for record in candidate.solver_records
        ),
        "universal_solver_bracket_expansions": sum(
            record.bracket_expansions for record in candidate.solver_records
        ),
        "universal_g_bundle_calls_entered": sum(
            record.g_function_evaluations for record in candidate.solver_records
        ),
        "universal_g_bundle_calls_completed": sum(
            record.g_function_evaluations for record in candidate.solver_records
        ),
        "universal_series_terms_evaluated": sum(
            record.series_terms_evaluated for record in candidate.solver_records
        ),
    }
    if any(getattr(work, name) != expected for name, expected in aggregate_fields.items()):
        raise TrajectoryContractError(
            "WH all-far aggregate solver work does not match its candidate"
        )
    return candidate


def _far_probe_decision(
    *,
    body_count: int,
    outcome: str,
    reason: str,
    phase: str,
    work: _FarProbeWork,
    body_indices: tuple[int, ...] = (),
    pair_indices: tuple[tuple[int, int], ...] = (),
    require_hybrid_decision: bool = False,
) -> _FarProbeDecision:
    if require_hybrid_decision and body_count > ENCOUNTER_HARD_MAXIMUM_BODY_COUNT:
        raise TrajectoryContractError(
            "hybrid WH probes support at most sixteen active bodies"
        )
    values = {
        "body_count": body_count,
        "outcome": outcome,
        "reason": reason,
        "phase": phase,
        "body_indices": body_indices,
        "pair_indices": pair_indices,
        "work": work,
    }
    if body_count <= ENCOUNTER_HARD_MAXIMUM_BODY_COUNT:
        if type(work) is not HybridFarProbeWork:
            raise TrajectoryContractError(
                "hybrid-qualified probe work lost its exact frozen schema"
            )
        return HybridFarProbeDecision(**values)  # type: ignore[arg-type]
    if type(work) is not _StrictFarProbeWork:
        raise TrajectoryContractError(
            "large public WH probe work lost its private strict schema"
        )
    return _StrictFarProbeDecision(**values)  # type: ignore[arg-type]


def _terminal_probe(
    *,
    body_count: int,
    outcome: str,
    reason: str,
    phase: str,
    builder: _WisdomHolmanProbeWorkBuilder,
    public_error: Exception,
    body_indices: tuple[int, ...] = (),
    pair_indices: tuple[tuple[int, int], ...] = (),
    require_hybrid_decision: bool = False,
) -> _WisdomHolmanMacrostepProposal:
    work = builder.freeze(body_count=body_count)
    decision = _far_probe_decision(
        body_count=body_count,
        outcome=outcome,
        reason=reason,
        phase=phase,
        work=work,
        body_indices=body_indices,
        pair_indices=pair_indices,
        require_hybrid_decision=require_hybrid_decision,
    )
    return _WisdomHolmanMacrostepProposal(decision, None, public_error)


@dataclass(frozen=True, eq=False)
class _WisdomHolmanRun:
    checkpoints: tuple[TrajectoryCheckpoint, ...]
    force_model_ids: tuple[str, ...]
    force_ledger: tuple[ForceLedgerEntry, ...]
    force_evaluations: int
    interaction_force_assemblies: int
    kepler_subflow_solves: int
    universal_solver_iterations: int
    maximum_universal_solver_iterations: int
    universal_solver_bracket_expansions: int
    maximum_universal_solver_bracket_expansions: int
    universal_g_function_evaluations: int
    universal_series_terms_evaluated: int
    coordinate_forward_transforms: int
    coordinate_inverse_transforms: int
    node_guard_evaluations: int
    path_guard_evaluations: int
    maximum_kepler_time_residual: float
    maximum_kepler_residual_tolerance: float
    maximum_kepler_lagrange_identity_error: float
    maximum_kepler_energy_error: float
    maximum_kepler_angular_momentum_error: float
    maximum_translation_force_residual: float
    minimum_jacobi_periapses: tuple[float, ...]
    maximum_jacobi_eccentricities: tuple[float, ...]
    maximum_interaction_force_ratios: tuple[float, ...]
    maximum_orbit_step_fractions: tuple[float, ...]
    maximum_periapse_step_fractions: tuple[float, ...]
    minimum_pair_endpoint_separations: tuple[float, ...]
    minimum_pair_path_lower_bounds: tuple[float, ...]
    minimum_pair_clearance_after_margins: tuple[float, ...]
    minimum_secondary_hill_floor_ratios: tuple[float, ...]
    maximum_barycenter_position_norm: float
    maximum_barycenter_velocity_norm: float
    initial_barycenter_position_cap: float
    initial_barycenter_velocity_cap: float


def _run_execution_counts(run: _WisdomHolmanRun) -> tuple[int, ...]:
    if type(run) is not _WisdomHolmanRun:
        raise TrajectoryContractError("run must be an exact _WisdomHolmanRun")
    return tuple(getattr(run, name) for name in _EXECUTION_COUNT_NAMES)


def _merge_node_extrema(
    *,
    evidence: _NodeGuardEvidence,
    minimum_periapses: list[float],
    maximum_eccentricities: list[float],
    maximum_interaction_ratios: list[float],
    maximum_orbit_fractions: list[float],
    maximum_periapse_fractions: list[float],
    minimum_pair_endpoints: list[float],
    minimum_hill_ratios: list[float],
) -> tuple[float, float]:
    for target, observed in (
        (minimum_periapses, evidence.periapses),
        (minimum_pair_endpoints, evidence.pair_separations),
        (minimum_hill_ratios, evidence.secondary_hill_floor_ratios),
    ):
        if len(target) != len(observed):
            raise TrajectoryContractError(
                "Wisdom--Holman guard evidence changed its canonical pair/body roster"
            )
        for index, value in enumerate(observed):
            target[index] = min(target[index], value)
    for target, observed in (
        (maximum_eccentricities, evidence.eccentricities),
        (maximum_interaction_ratios, evidence.interaction_force_ratios),
        (maximum_orbit_fractions, evidence.orbit_step_fractions),
        (maximum_periapse_fractions, evidence.periapse_step_fractions),
    ):
        if len(target) != len(observed):
            raise TrajectoryContractError(
                "Wisdom--Holman guard evidence changed its canonical secondary roster"
            )
        for index, value in enumerate(observed):
            target[index] = max(target[index], value)
    return evidence.barycenter_position_norm, evidence.barycenter_velocity_norm


def _same_checkpoint(left: TrajectoryCheckpoint, right: TrajectoryCheckpoint) -> bool:
    return (
        type(left) is TrajectoryCheckpoint
        and type(right) is TrajectoryCheckpoint
        and all(
            getattr(left, name) == getattr(right, name)
            for name in (
                "index",
                "epoch",
                "body_ids",
                "backend_id",
                "device",
                "accepted_steps",
                "rejected_steps",
                "dtype",
                "evidence_class",
                "registry_authorized",
                "qualification_authorized",
            )
        )
        and left.positions.tobytes(order="C") == right.positions.tobytes(order="C")
        and left.velocities.tobytes(order="C") == right.velocities.tobytes(order="C")
    )


def _initialize_wisdom_holman_continuous_cache(
    *,
    initial_snapshot: StateSnapshot,
    force_plan: ForcePlan,
    integration_spec: FixedStepWisdomHolmanSpec,
    binding: JacobiCoordinateBinding,
    accepted_outer_step_index: int = 0,
    validate_initial_barycenter: bool = True,
    retained_force_custody: _WisdomHolmanForceCustody | None = None,
    retained_force_model_ids: tuple[str, ...] | None = None,
    retained_force_ledger: tuple[ForceLedgerEntry, ...] | None = None,
    require_hybrid_decision: bool = False,
) -> _WisdomHolmanCachePreparation:
    """Bind a synchronized Cartesian node into one continuous WH cache.

    The returned cache owns every numerical buffer.  Its successful transform,
    force, interaction assembly, and accepted-start guard are charged exactly
    once by the next proposal (R=1); subsequent committed FAR proposals use the
    continuous cache with R=0.
    """

    body_count = len(initial_snapshot.body_ids)
    if require_hybrid_decision and body_count > ENCOUNTER_HARD_MAXIMUM_BODY_COUNT:
        raise TrajectoryContractError(
            "hybrid WH probes support at most sixteen active bodies"
        )
    if retained_force_model_ids is not None or retained_force_ledger is not None:
        raise TrajectoryContractError(
            "raw rebind force IDs/ledger are forbidden; use sealed force custody"
        )
    if retained_force_custody is not None:
        _validate_force_custody(
            retained_force_custody,
            initial_snapshot=initial_snapshot,
            force_plan=force_plan,
            integration_spec=integration_spec,
            binding=binding,
        )
    if type(accepted_outer_step_index) is not int or accepted_outer_step_index < 0:
        raise TrajectoryContractError(
            "accepted_outer_step_index must be an exact nonnegative integer"
        )
    builder = _WisdomHolmanProbeWorkBuilder()
    positions = np.array(
        initial_snapshot.positions,
        dtype=np.float64,
        copy=True,
        order="C",
        subok=False,
    )
    velocities = np.array(
        initial_snapshot.velocities,
        dtype=np.float64,
        copy=True,
        order="C",
        subok=False,
    )

    builder.cartesian_to_jacobi_calls_entered = 1
    try:
        coordinates, momenta = _cartesian_to_jacobi(
            binding, positions, velocities
        )
    except TrajectoryDomainError as exc:
        terminal = _terminal_probe(
            body_count=body_count,
            outcome="FATAL_FAILURE",
            reason="NONFINITE_OR_SINGULAR_GLOBAL_NUMERICAL_FAILURE",
            phase="ACCEPTED_START_REBIND",
            builder=builder,
            public_error=exc,
            require_hybrid_decision=require_hybrid_decision,
        )
        return _WisdomHolmanCachePreparation(
            None, terminal, 0.0, 0.0, 0.0, 0.0
        )
    builder.cartesian_to_jacobi_transforms_completed = 1

    if validate_initial_barycenter:
        (
            barycenter_position_norm,
            barycenter_velocity_norm,
            barycenter_position_cap,
            barycenter_velocity_cap,
        ) = _validate_initial_barycenter(
            binding=binding,
            coordinates=coordinates,
            momenta=momenta,
            positions=positions,
            velocities=velocities,
            spec=integration_spec,
        )
    else:
        barycenter_position_norm = 0.0
        barycenter_velocity_norm = 0.0
        barycenter_position_cap = 0.0
        barycenter_velocity_cap = 0.0

    builder.force_calls_entered = 1
    try:
        evaluation = evaluate_force_plan(initial_snapshot, force_plan)
    except (EvaluationError, ForceError) as exc:
        terminal = _terminal_probe(
            body_count=body_count,
            outcome="FATAL_FAILURE",
            reason="FORCE_EVALUATION_OR_IDENTITY_FAILURE",
            phase="ACCEPTED_START_NODE",
            builder=builder,
            public_error=exc,
            require_hybrid_decision=require_hybrid_decision,
        )
        return _WisdomHolmanCachePreparation(
            None,
            terminal,
            barycenter_position_norm,
            barycenter_velocity_norm,
            barycenter_position_cap,
            barycenter_velocity_cap,
        )
    builder.force_evaluations_completed = 1
    expected_model_ids = (
        retained_force_custody.force_model_ids
        if retained_force_custody is not None
        else (NewtonianPointMass.MODEL_ID,)
    )
    if evaluation.applied_model_ids != expected_model_ids:
        error = TrajectoryContractError(
            "Wisdom--Holman force evaluation did not retain Newtonian-only identity"
            if retained_force_custody is None
            else "Wisdom--Holman force-model identity changed during integration"
        )
        terminal = _terminal_probe(
            body_count=body_count,
            outcome="FATAL_FAILURE",
            reason="FORCE_EVALUATION_OR_IDENTITY_FAILURE",
            phase="ACCEPTED_START_NODE",
            builder=builder,
            public_error=error,
            require_hybrid_decision=require_hybrid_decision,
        )
        return _WisdomHolmanCachePreparation(
            None,
            terminal,
            barycenter_position_norm,
            barycenter_velocity_norm,
            barycenter_position_cap,
            barycenter_velocity_cap,
        )
    if retained_force_custody is not None and not (
        _force_custody_matches_authoritative_evaluation(
            retained_force_custody, evaluation.ledger
        )
    ):
        raise TrajectoryContractError(
            "WH retained force ledger differs from the authoritative rebind evaluation"
        )
    acceleration = np.array(
        evaluation.total_acceleration,
        dtype=np.float64,
        copy=True,
        order="C",
        subok=False,
    )
    try:
        interaction_force, translation_residual = _interaction_force(
            binding=binding,
            coordinates=coordinates,
            cartesian_acceleration=acceleration,
        )
    except _TranslationForceResidualFailure as exc:
        builder.interaction_force_assemblies = 1
        terminal = _terminal_probe(
            body_count=body_count,
            outcome="FATAL_FAILURE",
            reason="TRANSLATION_FORCE_RESIDUAL_FAILURE",
            phase="ACCEPTED_START_NODE",
            builder=builder,
            public_error=exc,
            require_hybrid_decision=require_hybrid_decision,
        )
        return _WisdomHolmanCachePreparation(
            None,
            terminal,
            barycenter_position_norm,
            barycenter_velocity_norm,
            barycenter_position_cap,
            barycenter_velocity_cap,
        )
    except _InteractionAssemblyNumericalFailure as exc:
        reason = (
            "NONFINITE_OR_SINGULAR_BODY_NUMERICAL_FAILURE"
            if exc.body_indices
            else "NONFINITE_OR_SINGULAR_GLOBAL_NUMERICAL_FAILURE"
        )
        terminal = _terminal_probe(
            body_count=body_count,
            outcome="FATAL_FAILURE",
            reason=reason,
            phase="ACCEPTED_START_NODE",
            builder=builder,
            public_error=exc,
            body_indices=exc.body_indices,
            require_hybrid_decision=require_hybrid_decision,
        )
        return _WisdomHolmanCachePreparation(
            None,
            terminal,
            barycenter_position_norm,
            barycenter_velocity_norm,
            barycenter_position_cap,
            barycenter_velocity_cap,
        )
    except TrajectoryDomainError as exc:
        terminal = _terminal_probe(
            body_count=body_count,
            outcome="FATAL_FAILURE",
            reason="NONFINITE_OR_SINGULAR_GLOBAL_NUMERICAL_FAILURE",
            phase="ACCEPTED_START_NODE",
            builder=builder,
            public_error=exc,
            require_hybrid_decision=require_hybrid_decision,
        )
        return _WisdomHolmanCachePreparation(
            None,
            terminal,
            barycenter_position_norm,
            barycenter_velocity_norm,
            barycenter_position_cap,
            barycenter_velocity_cap,
        )
    builder.interaction_force_assemblies = 1

    builder.node_guard_evaluations = 1
    try:
        guard = _node_guard_verdict(
            binding=binding,
            coordinates=coordinates,
            momenta=momenta,
            interaction_force=interaction_force,
            cartesian_positions=positions,
            radii=initial_snapshot.radii,
            spec=integration_spec,
        )
    except _BodyNumericalFailure as exc:
        terminal = _terminal_probe(
            body_count=body_count,
            outcome="FATAL_FAILURE",
            reason="NONFINITE_OR_SINGULAR_BODY_NUMERICAL_FAILURE",
            phase="ACCEPTED_START_NODE",
            builder=builder,
            public_error=exc,
            body_indices=exc.body_indices,
            require_hybrid_decision=require_hybrid_decision,
        )
        return _WisdomHolmanCachePreparation(
            None,
            terminal,
            barycenter_position_norm,
            barycenter_velocity_norm,
            barycenter_position_cap,
            barycenter_velocity_cap,
        )
    except TrajectoryDomainError as exc:
        terminal = _terminal_probe(
            body_count=body_count,
            outcome="FATAL_FAILURE",
            reason="NONFINITE_OR_SINGULAR_GLOBAL_NUMERICAL_FAILURE",
            phase="ACCEPTED_START_NODE",
            builder=builder,
            public_error=exc,
            require_hybrid_decision=require_hybrid_decision,
        )
        return _WisdomHolmanCachePreparation(
            None,
            terminal,
            barycenter_position_norm,
            barycenter_velocity_norm,
            barycenter_position_cap,
            barycenter_velocity_cap,
        )
    if guard.reason is not None:
        assert guard.public_error is not None
        terminal = _terminal_probe(
            body_count=body_count,
            outcome="NEAR_SWITCH",
            reason=guard.reason,
            phase="ACCEPTED_START_NODE",
            builder=builder,
            public_error=guard.public_error,
            body_indices=guard.body_indices,
            pair_indices=guard.pair_indices,
            require_hybrid_decision=require_hybrid_decision,
        )
        return _WisdomHolmanCachePreparation(
            None,
            terminal,
            barycenter_position_norm,
            barycenter_velocity_norm,
            barycenter_position_cap,
            barycenter_velocity_cap,
        )
    assert guard.evidence is not None
    if retained_force_custody is None:
        force_custody = _make_force_custody(
            force_model_ids=evaluation.applied_model_ids,
            force_ledger=evaluation.ledger,
            initial_snapshot=initial_snapshot,
            force_plan=force_plan,
            integration_spec=integration_spec,
            binding=binding,
        )
    else:
        force_custody = retained_force_custody
    cache = _make_continuous_cache(
        positions=_readonly_copy(positions),
        velocities=_readonly_copy(velocities),
        coordinates=_readonly_copy(coordinates),
        momenta=_readonly_copy(momenta),
        interaction_force=_readonly_copy(interaction_force),
        accepted_outer_step_index=accepted_outer_step_index,
        accepted_epoch=float(initial_snapshot.epoch),
        force_custody=force_custody,
        rebind_work_pending=True,
        accepted_start_guard=guard.evidence,
        accepted_start_translation_residual=float(translation_residual),
        authoritative_snapshot=initial_snapshot,
        force_plan=force_plan,
        integration_spec=integration_spec,
        binding=binding,
    )
    return _WisdomHolmanCachePreparation(
        cache,
        None,
        float(barycenter_position_norm),
        float(barycenter_velocity_norm),
        float(barycenter_position_cap),
        float(barycenter_velocity_cap),
    )


def _propose_wisdom_holman_macrostep(
    *,
    cache: _WisdomHolmanContinuousCache,
    initial_snapshot: StateSnapshot,
    force_plan: ForcePlan,
    integration_spec: FixedStepWisdomHolmanSpec,
    binding: JacobiCoordinateBinding,
    expected_cache_epoch: float,
    expected_rebind_work_pending: bool,
    candidate_epoch: float,
    candidate_outer_step_index: int,
    require_hybrid_decision: bool = False,
) -> _WisdomHolmanMacrostepProposal:
    """Build one complete WH macrostep transaction without committing it."""

    body_count = len(binding.body_ids)
    if require_hybrid_decision and body_count > ENCOUNTER_HARD_MAXIMUM_BODY_COUNT:
        raise TrajectoryContractError(
            "hybrid WH probes support at most sixteen active bodies"
        )
    if (
        type(candidate_outer_step_index) is not int
        or candidate_outer_step_index <= 0
    ):
        raise TrajectoryContractError(
            "WH proposal candidate outer-step index must be an exact positive integer"
        )
    cache = _validate_continuous_cache(
        cache,
        initial_snapshot=initial_snapshot,
        force_plan=force_plan,
        integration_spec=integration_spec,
        binding=binding,
        expected_outer_step_index=candidate_outer_step_index - 1,
        expected_epoch=expected_cache_epoch,
        expected_rebind_work_pending=expected_rebind_work_pending,
    )
    if type(candidate_epoch) is not float or not math.isfinite(candidate_epoch):
        raise TrajectoryContractError(
            "WH proposal candidate_epoch must be a finite built-in float"
        )
    if (
        integration_spec.fixed_step > 0.0
        and candidate_epoch <= cache.accepted_epoch
    ) or (
        integration_spec.fixed_step < 0.0
        and candidate_epoch >= cache.accepted_epoch
    ):
        raise TrajectoryContractError(
            "WH proposal candidate_epoch must advance in the fixed-step direction"
        )
    builder = _WisdomHolmanProbeWorkBuilder.from_cache(cache)
    half_step = 0.5 * integration_spec.fixed_step

    half_momenta = _state_axpy(
        cache.momenta, half_step, cache.interaction_force
    )
    if not bool(np.all(np.isfinite(half_momenta))):
        error = TrajectoryDomainError(
            "Wisdom--Holman first interaction half-kick became nonfinite"
        )
        return _terminal_probe(
            body_count=body_count,
            outcome="FATAL_FAILURE",
            reason="NONFINITE_OR_SINGULAR_GLOBAL_NUMERICAL_FAILURE",
            phase="FIRST_HALF_KICK",
            builder=builder,
            public_error=error,
            require_hybrid_decision=require_hybrid_decision,
        )
    builder.first_half_kicks_completed = 1

    builder.node_guard_evaluations += 1
    try:
        drift_verdict = _node_guard_verdict(
            binding=binding,
            coordinates=cache.coordinates,
            momenta=half_momenta,
            interaction_force=cache.interaction_force,
            cartesian_positions=cache.positions,
            radii=initial_snapshot.radii,
            spec=integration_spec,
        )
    except _BodyNumericalFailure as exc:
        return _terminal_probe(
            body_count=body_count,
            outcome="FATAL_FAILURE",
            reason="NONFINITE_OR_SINGULAR_BODY_NUMERICAL_FAILURE",
            phase="POST_FIRST_KICK_PREDRIFT_NODE",
            builder=builder,
            public_error=exc,
            body_indices=exc.body_indices,
            require_hybrid_decision=require_hybrid_decision,
        )
    except TrajectoryDomainError as exc:
        return _terminal_probe(
            body_count=body_count,
            outcome="FATAL_FAILURE",
            reason="NONFINITE_OR_SINGULAR_GLOBAL_NUMERICAL_FAILURE",
            phase="POST_FIRST_KICK_PREDRIFT_NODE",
            builder=builder,
            public_error=exc,
            require_hybrid_decision=require_hybrid_decision,
        )
    if drift_verdict.reason is not None:
        assert drift_verdict.public_error is not None
        return _terminal_probe(
            body_count=body_count,
            outcome="NEAR_SWITCH",
            reason=drift_verdict.reason,
            phase="POST_FIRST_KICK_PREDRIFT_NODE",
            builder=builder,
            public_error=drift_verdict.public_error,
            body_indices=drift_verdict.body_indices,
            pair_indices=drift_verdict.pair_indices,
            require_hybrid_decision=require_hybrid_decision,
        )
    assert drift_verdict.evidence is not None
    drift_guard = drift_verdict.evidence

    candidate_coordinates = cache.coordinates.copy()
    candidate_drift_momenta = half_momenta.copy()
    center_divisor = float(binding.cumulative_gravitational_parameters[-1])
    for component in range(3):
        candidate_coordinates[0, component] = float(
            cache.coordinates[0, component]
        ) + integration_spec.fixed_step * float(
            half_momenta[0, component]
        ) / center_divisor
    if not bool(np.all(np.isfinite(candidate_coordinates[0]))):
        error = TrajectoryDomainError(
            "Wisdom--Holman analytic A flow produced a nonfinite canonical state"
        )
        return _terminal_probe(
            body_count=body_count,
            outcome="FATAL_FAILURE",
            reason="NONFINITE_OR_SINGULAR_GLOBAL_NUMERICAL_FAILURE",
            phase="KEPLER_DRIFT_FLOW",
            builder=builder,
            public_error=error,
            require_hybrid_decision=require_hybrid_decision,
        )
    builder.center_of_mass_drifts_completed = 1

    step_solver_records: list[_UniversalKeplerStep] = []
    hybrid_qualified = body_count <= ENCOUNTER_HARD_MAXIMUM_BODY_COUNT
    failure_statuses = dict(HYBRID_KEPLER_PROBE_FAILURE_STATUS_BY_REASON)
    for index in range(1, body_count):
        recorder = _KeplerProbeAccumulator(index)
        try:
            subflow = _solve_elliptic_universal_kepler(
                cache.coordinates[index],
                half_momenta[index] / binding.canonical_inertias[index],
                float(binding.cumulative_gravitational_parameters[index]),
                integration_spec.fixed_step,
                integration_spec.kepler_solver,
                recorder,
            )
        except (
            TrajectoryStepLimitError,
            _KeplerSolverPostconditionFailure,
        ) as exc:
            reason = (
                "KEPLER_SOLVER_BRACKET_ITERATION_STAGNATION_OR_POSTCONDITION_FAILURE"
            )
            builder.kepler_probe_records.append(
                recorder.freeze(
                    failure_statuses[reason],
                    hybrid_qualified=hybrid_qualified,
                )
            )
            return _terminal_probe(
                body_count=body_count,
                outcome="FATAL_FAILURE",
                reason=reason,
                phase="KEPLER_DRIFT_FLOW",
                builder=builder,
                public_error=exc,
                body_indices=(index,),
                require_hybrid_decision=require_hybrid_decision,
            )
        except TrajectoryDomainError as exc:
            reason = "NONFINITE_OR_SINGULAR_BODY_NUMERICAL_FAILURE"
            builder.kepler_probe_records.append(
                recorder.freeze(
                    failure_statuses[reason],
                    hybrid_qualified=hybrid_qualified,
                )
            )
            return _terminal_probe(
                body_count=body_count,
                outcome="FATAL_FAILURE",
                reason=reason,
                phase="KEPLER_DRIFT_FLOW",
                builder=builder,
                public_error=exc,
                body_indices=(index,),
                require_hybrid_decision=require_hybrid_decision,
            )
        recorder.call_completed = True
        builder.kepler_probe_records.append(
            recorder.freeze("COMPLETED", hybrid_qualified=hybrid_qualified)
        )
        candidate_coordinates[index] = subflow.position
        inertia = float(binding.canonical_inertias[index])
        for component in range(3):
            candidate_drift_momenta[index, component] = inertia * float(
                subflow.velocity[component]
            )
        if not bool(np.all(np.isfinite(candidate_coordinates[index]))) or not bool(
            np.all(np.isfinite(candidate_drift_momenta[index]))
        ):
            error = TrajectoryDomainError(
                "Wisdom--Holman analytic A flow produced a nonfinite canonical state"
            )
            return _terminal_probe(
                body_count=body_count,
                outcome="FATAL_FAILURE",
                reason="NONFINITE_OR_SINGULAR_BODY_NUMERICAL_FAILURE",
                phase="KEPLER_DRIFT_FLOW",
                builder=builder,
                public_error=error,
                body_indices=(index,),
                require_hybrid_decision=require_hybrid_decision,
            )
        step_solver_records.append(
            replace(
                subflow,
                position=_readonly_copy(subflow.position),
                velocity=_readonly_copy(subflow.velocity),
            )
        )
    if not bool(np.all(np.isfinite(candidate_coordinates))) or not bool(
        np.all(np.isfinite(candidate_drift_momenta))
    ):
        error = TrajectoryDomainError(
            "Wisdom--Holman analytic A flow produced a nonfinite canonical state"
        )
        return _terminal_probe(
            body_count=body_count,
            outcome="FATAL_FAILURE",
            reason="NONFINITE_OR_SINGULAR_GLOBAL_NUMERICAL_FAILURE",
            phase="KEPLER_DRIFT_FLOW",
            builder=builder,
            public_error=error,
            require_hybrid_decision=require_hybrid_decision,
        )

    builder.jacobi_to_cartesian_calls_entered = 1
    try:
        candidate_positions, drift_velocities = _jacobi_to_cartesian(
            binding, candidate_coordinates, candidate_drift_momenta
        )
    except TrajectoryDomainError as exc:
        return _terminal_probe(
            body_count=body_count,
            outcome="FATAL_FAILURE",
            reason="NONFINITE_OR_SINGULAR_GLOBAL_NUMERICAL_FAILURE",
            phase="POST_DRIFT_CARTESIAN_RECONSTRUCTION",
            builder=builder,
            public_error=exc,
            require_hybrid_decision=require_hybrid_decision,
        )
    builder.jacobi_to_cartesian_transforms_completed = 1

    builder.path_guard_evaluations = 1
    try:
        path_verdict = _path_guard_verdict(
            binding=binding,
            start_positions=cache.positions,
            end_positions=candidate_positions,
            radii=initial_snapshot.radii,
            drift_orbits=drift_guard,
            spec=integration_spec,
        )
    except TrajectoryDomainError as exc:
        return _terminal_probe(
            body_count=body_count,
            outcome="FATAL_FAILURE",
            reason="NONFINITE_OR_SINGULAR_GLOBAL_NUMERICAL_FAILURE",
            phase="POST_DRIFT_PRE_FORCE_PATH_SCREEN",
            builder=builder,
            public_error=exc,
            require_hybrid_decision=require_hybrid_decision,
        )
    if path_verdict.reason is not None:
        assert path_verdict.public_error is not None
        return _terminal_probe(
            body_count=body_count,
            outcome="NEAR_SWITCH",
            reason=path_verdict.reason,
            phase="POST_DRIFT_PRE_FORCE_PATH_SCREEN",
            builder=builder,
            public_error=path_verdict.public_error,
            pair_indices=path_verdict.pair_indices,
            require_hybrid_decision=require_hybrid_decision,
        )
    assert path_verdict.evidence is not None
    path_evidence = path_verdict.evidence

    stage_snapshot = replace(
        initial_snapshot,
        epoch=candidate_epoch,
        positions=candidate_positions,
        velocities=drift_velocities,
    )
    builder.force_calls_entered += 1
    try:
        candidate_evaluation = evaluate_force_plan(stage_snapshot, force_plan)
    except (EvaluationError, ForceError) as exc:
        return _terminal_probe(
            body_count=body_count,
            outcome="FATAL_FAILURE",
            reason="FORCE_EVALUATION_OR_IDENTITY_FAILURE",
            phase="POST_DRIFT_CANDIDATE_FORCE",
            builder=builder,
            public_error=exc,
            require_hybrid_decision=require_hybrid_decision,
        )
    builder.force_evaluations_completed += 1
    if candidate_evaluation.applied_model_ids != cache.force_model_ids:
        error = TrajectoryContractError(
            "Wisdom--Holman force-model identity changed during integration"
        )
        return _terminal_probe(
            body_count=body_count,
            outcome="FATAL_FAILURE",
            reason="FORCE_EVALUATION_OR_IDENTITY_FAILURE",
            phase="POST_DRIFT_CANDIDATE_FORCE",
            builder=builder,
            public_error=error,
            require_hybrid_decision=require_hybrid_decision,
        )
    candidate_acceleration = np.array(
        candidate_evaluation.total_acceleration,
        dtype=np.float64,
        copy=True,
        order="C",
        subok=False,
    )
    try:
        candidate_interaction_force, candidate_translation_residual = (
            _interaction_force(
                binding=binding,
                coordinates=candidate_coordinates,
                cartesian_acceleration=candidate_acceleration,
            )
        )
    except _TranslationForceResidualFailure as exc:
        builder.interaction_force_assemblies += 1
        return _terminal_probe(
            body_count=body_count,
            outcome="FATAL_FAILURE",
            reason="TRANSLATION_FORCE_RESIDUAL_FAILURE",
            phase="POST_DRIFT_CANDIDATE_FORCE",
            builder=builder,
            public_error=exc,
            require_hybrid_decision=require_hybrid_decision,
        )
    except _InteractionAssemblyNumericalFailure as exc:
        reason = (
            "NONFINITE_OR_SINGULAR_BODY_NUMERICAL_FAILURE"
            if exc.body_indices
            else "NONFINITE_OR_SINGULAR_GLOBAL_NUMERICAL_FAILURE"
        )
        return _terminal_probe(
            body_count=body_count,
            outcome="FATAL_FAILURE",
            reason=reason,
            phase="POST_DRIFT_CANDIDATE_FORCE",
            builder=builder,
            public_error=exc,
            body_indices=exc.body_indices,
            require_hybrid_decision=require_hybrid_decision,
        )
    except TrajectoryDomainError as exc:
        return _terminal_probe(
            body_count=body_count,
            outcome="FATAL_FAILURE",
            reason="NONFINITE_OR_SINGULAR_GLOBAL_NUMERICAL_FAILURE",
            phase="POST_DRIFT_CANDIDATE_FORCE",
            builder=builder,
            public_error=exc,
            require_hybrid_decision=require_hybrid_decision,
        )
    builder.interaction_force_assemblies += 1

    candidate_momenta = _state_axpy(
        candidate_drift_momenta,
        half_step,
        candidate_interaction_force,
    )
    if not bool(np.all(np.isfinite(candidate_momenta))):
        error = TrajectoryDomainError(
            "Wisdom--Holman second interaction half-kick became nonfinite"
        )
        return _terminal_probe(
            body_count=body_count,
            outcome="FATAL_FAILURE",
            reason="NONFINITE_OR_SINGULAR_GLOBAL_NUMERICAL_FAILURE",
            phase="SECOND_HALF_KICK",
            builder=builder,
            public_error=error,
            require_hybrid_decision=require_hybrid_decision,
        )
    builder.second_half_kicks_completed = 1

    builder.jacobi_to_cartesian_calls_entered = 2
    try:
        synchronized_positions, synchronized_velocities = _jacobi_to_cartesian(
            binding, candidate_coordinates, candidate_momenta
        )
    except TrajectoryDomainError as exc:
        return _terminal_probe(
            body_count=body_count,
            outcome="FATAL_FAILURE",
            reason="NONFINITE_OR_SINGULAR_GLOBAL_NUMERICAL_FAILURE",
            phase="POST_SECOND_KICK_CARTESIAN_RECONSTRUCTION",
            builder=builder,
            public_error=exc,
            require_hybrid_decision=require_hybrid_decision,
        )
    builder.jacobi_to_cartesian_transforms_completed = 2

    builder.node_guard_evaluations += 1
    try:
        completed_verdict = _node_guard_verdict(
            binding=binding,
            coordinates=candidate_coordinates,
            momenta=candidate_momenta,
            interaction_force=candidate_interaction_force,
            cartesian_positions=synchronized_positions,
            radii=initial_snapshot.radii,
            spec=integration_spec,
        )
    except _BodyNumericalFailure as exc:
        return _terminal_probe(
            body_count=body_count,
            outcome="FATAL_FAILURE",
            reason="NONFINITE_OR_SINGULAR_BODY_NUMERICAL_FAILURE",
            phase="COMPLETED_CANDIDATE_NODE",
            builder=builder,
            public_error=exc,
            body_indices=exc.body_indices,
            require_hybrid_decision=require_hybrid_decision,
        )
    except TrajectoryDomainError as exc:
        return _terminal_probe(
            body_count=body_count,
            outcome="FATAL_FAILURE",
            reason="NONFINITE_OR_SINGULAR_GLOBAL_NUMERICAL_FAILURE",
            phase="COMPLETED_CANDIDATE_NODE",
            builder=builder,
            public_error=exc,
            require_hybrid_decision=require_hybrid_decision,
        )
    if completed_verdict.reason is not None:
        assert completed_verdict.public_error is not None
        return _terminal_probe(
            body_count=body_count,
            outcome="NEAR_SWITCH",
            reason=completed_verdict.reason,
            phase="COMPLETED_CANDIDATE_NODE",
            builder=builder,
            public_error=completed_verdict.public_error,
            body_indices=completed_verdict.body_indices,
            pair_indices=completed_verdict.pair_indices,
            require_hybrid_decision=require_hybrid_decision,
        )
    assert completed_verdict.evidence is not None
    work = builder.freeze(body_count=body_count)
    decision = _far_probe_decision(
        body_count=body_count,
        outcome="FAR_PASS",
        reason=HYBRID_ALL_FAR_REASON,
        phase="ALL_GUARDS_COMPLETED",
        work=work,
        require_hybrid_decision=require_hybrid_decision,
    )
    candidate = _make_macrostep_candidate(
        positions=_readonly_copy(synchronized_positions),
        velocities=_readonly_copy(synchronized_velocities),
        coordinates=_readonly_copy(candidate_coordinates),
        momenta=_readonly_copy(candidate_momenta),
        interaction_force=_readonly_copy(candidate_interaction_force),
        outer_step_index=candidate_outer_step_index,
        epoch=candidate_epoch,
        drift_guard=drift_guard,
        completed_guard=completed_verdict.evidence,
        path_evidence=path_evidence,
        solver_records=tuple(step_solver_records),
        translation_residual=float(candidate_translation_residual),
        source_cache=cache,
        signed_fixed_step=integration_spec.fixed_step,
    )
    return _WisdomHolmanMacrostepProposal(decision, candidate, None)


def _commit_wisdom_holman_macrostep(
    *,
    cache: _WisdomHolmanContinuousCache,
    proposal: _WisdomHolmanMacrostepProposal,
    initial_snapshot: StateSnapshot,
    force_plan: ForcePlan,
    integration_spec: FixedStepWisdomHolmanSpec,
    binding: JacobiCoordinateBinding,
    expected_cache_epoch: float,
    expected_rebind_work_pending: bool,
    expected_candidate_epoch: float,
    expected_outer_step_index: int,
) -> _WisdomHolmanContinuousCache:
    """Atomically adopt a fully checked FAR candidate with independent custody."""

    if type(proposal) is not _WisdomHolmanMacrostepProposal:
        raise TrajectoryContractError(
            "WH commit proposal must have its exact private type"
        )
    if type(expected_outer_step_index) is not int or expected_outer_step_index <= 0:
        raise TrajectoryContractError(
            "WH commit expected outer-step index must be an exact positive integer"
        )
    cache = _validate_continuous_cache(
        cache,
        initial_snapshot=initial_snapshot,
        force_plan=force_plan,
        integration_spec=integration_spec,
        binding=binding,
        expected_outer_step_index=expected_outer_step_index - 1,
        expected_epoch=expected_cache_epoch,
        expected_rebind_work_pending=expected_rebind_work_pending,
    )
    if proposal.candidate is None:
        raise TrajectoryContractError("only a FAR_PASS candidate can be committed")
    if proposal.public_error is not None:
        raise TrajectoryContractError("a FAR_PASS cannot retain a public failure")
    candidate = _validate_far_candidate_and_decision(
        cache=cache,
        decision=proposal.decision,
        candidate=proposal.candidate,
        integration_spec=integration_spec,
        binding=binding,
        expected_outer_step_index=expected_outer_step_index,
        expected_epoch=expected_candidate_epoch,
    )
    return _make_continuous_cache(
        positions=_readonly_copy(candidate.positions),
        velocities=_readonly_copy(candidate.velocities),
        coordinates=_readonly_copy(candidate.coordinates),
        momenta=_readonly_copy(candidate.momenta),
        interaction_force=_readonly_copy(candidate.interaction_force),
        accepted_outer_step_index=candidate.outer_step_index,
        accepted_epoch=candidate.epoch,
        force_custody=cache.force_custody,
        rebind_work_pending=False,
        accepted_start_guard=None,
        accepted_start_translation_residual=0.0,
        authoritative_snapshot=initial_snapshot,
        force_plan=force_plan,
        integration_spec=integration_spec,
        binding=binding,
    )


def _execute_wisdom_holman_map(
    *,
    initial_snapshot: StateSnapshot,
    force_plan: ForcePlan,
    integration_spec: FixedStepWisdomHolmanSpec,
    binding: JacobiCoordinateBinding,
    all_epochs: tuple[float, ...],
    checkpoint_epochs: tuple[float, ...],
) -> _WisdomHolmanRun:
    """Execute the complete map without constructing its public result."""

    body_count = len(initial_snapshot.body_ids)
    prepared = _initialize_wisdom_holman_continuous_cache(
        initial_snapshot=initial_snapshot,
        force_plan=force_plan,
        integration_spec=integration_spec,
        binding=binding,
    )
    if prepared.terminal is not None:
        assert prepared.terminal.public_error is not None
        raise prepared.terminal.public_error
    assert prepared.cache is not None
    cache = prepared.cache
    assert cache.accepted_start_guard is not None
    initial_guard = cache.accepted_start_guard

    minimum_periapses = list(initial_guard.periapses)
    maximum_eccentricities = list(initial_guard.eccentricities)
    maximum_interaction_ratios = list(initial_guard.interaction_force_ratios)
    maximum_orbit_fractions = list(initial_guard.orbit_step_fractions)
    maximum_periapse_fractions = list(initial_guard.periapse_step_fractions)
    minimum_pair_endpoints = list(initial_guard.pair_separations)
    minimum_hill_ratios = list(initial_guard.secondary_hill_floor_ratios)
    pair_count = body_count * (body_count - 1) // 2
    minimum_path_lower_bounds = [math.inf] * pair_count
    minimum_path_clearances = [math.inf] * pair_count
    maximum_barycenter_position_norm = prepared.initial_barycenter_position_norm
    maximum_barycenter_velocity_norm = prepared.initial_barycenter_velocity_norm

    force_evaluations = 0
    interaction_force_assemblies = 0
    kepler_subflow_solves = 0
    total_solver_iterations = 0
    maximum_solver_iterations = 0
    total_bracket_expansions = 0
    maximum_bracket_expansions = 0
    g_function_evaluations = 0
    series_terms_evaluated = 0
    coordinate_forward_transforms = 0
    coordinate_inverse_transforms = 0
    node_guard_evaluations = 0
    path_guard_evaluations = 0
    maximum_time_residual = 0.0
    maximum_residual_tolerance = 0.0
    maximum_lagrange_error = 0.0
    maximum_energy_error = 0.0
    maximum_angular_error = 0.0
    maximum_translation_residual = cache.accepted_start_translation_residual

    checkpoints: list[TrajectoryCheckpoint] = [
        TrajectoryCheckpoint(
            index=0,
            epoch=checkpoint_epochs[0],
            body_ids=initial_snapshot.body_ids,
            backend_id="numpy",
            device="cpu",
            positions=_readonly_copy(cache.positions),
            velocities=_readonly_copy(cache.velocities),
            accepted_steps=0,
            rejected_steps=0,
        )
    ]
    next_checkpoint = 1

    for step_index in range(1, integration_spec.completed_steps + 1):
        expected_rebind_work_pending = step_index == 1
        proposal = _propose_wisdom_holman_macrostep(
            cache=cache,
            initial_snapshot=initial_snapshot,
            force_plan=force_plan,
            integration_spec=integration_spec,
            binding=binding,
            expected_cache_epoch=all_epochs[step_index - 1],
            expected_rebind_work_pending=expected_rebind_work_pending,
            candidate_epoch=all_epochs[step_index],
            candidate_outer_step_index=step_index,
        )
        if proposal.decision.outcome != "FAR_PASS":
            assert proposal.public_error is not None
            raise proposal.public_error
        assert proposal.candidate is not None
        candidate = proposal.candidate
        work = proposal.decision.work
        cache = _commit_wisdom_holman_macrostep(
            cache=cache,
            proposal=proposal,
            initial_snapshot=initial_snapshot,
            force_plan=force_plan,
            integration_spec=integration_spec,
            binding=binding,
            expected_cache_epoch=all_epochs[step_index - 1],
            expected_rebind_work_pending=expected_rebind_work_pending,
            expected_candidate_epoch=all_epochs[step_index],
            expected_outer_step_index=step_index,
        )

        force_evaluations += work.force_evaluations_completed
        interaction_force_assemblies += work.interaction_force_assemblies
        kepler_subflow_solves += work.kepler_subflow_solves_completed
        total_solver_iterations += work.universal_solver_iterations
        total_bracket_expansions += work.universal_solver_bracket_expansions
        g_function_evaluations += work.universal_g_bundle_calls_completed
        series_terms_evaluated += work.universal_series_terms_evaluated
        coordinate_forward_transforms += (
            work.cartesian_to_jacobi_transforms_completed
        )
        coordinate_inverse_transforms += (
            work.jacobi_to_cartesian_transforms_completed
        )
        node_guard_evaluations += work.node_guard_evaluations
        path_guard_evaluations += work.path_guard_evaluations
        maximum_translation_residual = max(
            maximum_translation_residual, candidate.translation_residual
        )

        for evidence in (candidate.drift_guard, candidate.completed_guard):
            bary_position, bary_velocity = _merge_node_extrema(
                evidence=evidence,
                minimum_periapses=minimum_periapses,
                maximum_eccentricities=maximum_eccentricities,
                maximum_interaction_ratios=maximum_interaction_ratios,
                maximum_orbit_fractions=maximum_orbit_fractions,
                maximum_periapse_fractions=maximum_periapse_fractions,
                minimum_pair_endpoints=minimum_pair_endpoints,
                minimum_hill_ratios=minimum_hill_ratios,
            )
            maximum_barycenter_position_norm = max(
                maximum_barycenter_position_norm, bary_position
            )
            maximum_barycenter_velocity_norm = max(
                maximum_barycenter_velocity_norm, bary_velocity
            )
        for pair_index, value in enumerate(
            candidate.path_evidence.endpoint_minimum_separations
        ):
            minimum_pair_endpoints[pair_index] = min(
                minimum_pair_endpoints[pair_index], value
            )
        for pair_index, value in enumerate(candidate.path_evidence.path_lower_bounds):
            minimum_path_lower_bounds[pair_index] = min(
                minimum_path_lower_bounds[pair_index], value
            )
        for pair_index, value in enumerate(
            candidate.path_evidence.clearance_after_margins
        ):
            minimum_path_clearances[pair_index] = min(
                minimum_path_clearances[pair_index], value
            )
        for subflow in candidate.solver_records:
            maximum_solver_iterations = max(
                maximum_solver_iterations, subflow.iterations
            )
            maximum_bracket_expansions = max(
                maximum_bracket_expansions, subflow.bracket_expansions
            )
            maximum_time_residual = max(
                maximum_time_residual, subflow.time_residual
            )
            maximum_residual_tolerance = max(
                maximum_residual_tolerance, subflow.residual_tolerance
            )
            maximum_lagrange_error = max(
                maximum_lagrange_error, subflow.lagrange_identity_error
            )
            maximum_energy_error = max(
                maximum_energy_error, subflow.energy_error
            )
            maximum_angular_error = max(
                maximum_angular_error, subflow.angular_momentum_error
            )

        if (
            next_checkpoint < len(integration_spec.checkpoint_step_indices)
            and step_index
            == integration_spec.checkpoint_step_indices[next_checkpoint]
        ):
            checkpoints.append(
                TrajectoryCheckpoint(
                    index=next_checkpoint,
                    epoch=checkpoint_epochs[next_checkpoint],
                    body_ids=initial_snapshot.body_ids,
                    backend_id="numpy",
                    device="cpu",
                    positions=_readonly_copy(cache.positions),
                    velocities=_readonly_copy(cache.velocities),
                    accepted_steps=step_index,
                    rejected_steps=0,
                )
            )
            next_checkpoint += 1

    if next_checkpoint != len(integration_spec.checkpoint_step_indices):
        raise TrajectoryContractError(
            "Wisdom--Holman did not materialize every integer-node checkpoint"
        )
    expected_force_evaluations = integration_spec.completed_steps + 1
    expected_solves = integration_spec.completed_steps * (body_count - 1)
    if force_evaluations != expected_force_evaluations:
        raise TrajectoryContractError(
            "Wisdom--Holman force evaluations must equal completed steps plus one"
        )
    if kepler_subflow_solves != expected_solves:
        raise TrajectoryContractError(
            "Wisdom--Holman Kepler solve accounting is inconsistent"
        )

    return _WisdomHolmanRun(
        checkpoints=tuple(checkpoints),
        force_model_ids=cache.force_model_ids,
        force_ledger=cache.force_ledger,
        force_evaluations=force_evaluations,
        interaction_force_assemblies=interaction_force_assemblies,
        kepler_subflow_solves=kepler_subflow_solves,
        universal_solver_iterations=total_solver_iterations,
        maximum_universal_solver_iterations=maximum_solver_iterations,
        universal_solver_bracket_expansions=total_bracket_expansions,
        maximum_universal_solver_bracket_expansions=maximum_bracket_expansions,
        universal_g_function_evaluations=g_function_evaluations,
        universal_series_terms_evaluated=series_terms_evaluated,
        coordinate_forward_transforms=coordinate_forward_transforms,
        coordinate_inverse_transforms=coordinate_inverse_transforms,
        node_guard_evaluations=node_guard_evaluations,
        path_guard_evaluations=path_guard_evaluations,
        maximum_kepler_time_residual=float(maximum_time_residual),
        maximum_kepler_residual_tolerance=float(maximum_residual_tolerance),
        maximum_kepler_lagrange_identity_error=float(maximum_lagrange_error),
        maximum_kepler_energy_error=float(maximum_energy_error),
        maximum_kepler_angular_momentum_error=float(maximum_angular_error),
        maximum_translation_force_residual=float(maximum_translation_residual),
        minimum_jacobi_periapses=tuple(float(value) for value in minimum_periapses),
        maximum_jacobi_eccentricities=tuple(
            float(value) for value in maximum_eccentricities
        ),
        maximum_interaction_force_ratios=tuple(
            float(value) for value in maximum_interaction_ratios
        ),
        maximum_orbit_step_fractions=tuple(
            float(value) for value in maximum_orbit_fractions
        ),
        maximum_periapse_step_fractions=tuple(
            float(value) for value in maximum_periapse_fractions
        ),
        minimum_pair_endpoint_separations=tuple(
            float(value) for value in minimum_pair_endpoints
        ),
        minimum_pair_path_lower_bounds=tuple(
            float(value) for value in minimum_path_lower_bounds
        ),
        minimum_pair_clearance_after_margins=tuple(
            float(value) for value in minimum_path_clearances
        ),
        minimum_secondary_hill_floor_ratios=tuple(
            float(value) for value in minimum_hill_ratios
        ),
        maximum_barycenter_position_norm=float(maximum_barycenter_position_norm),
        maximum_barycenter_velocity_norm=float(maximum_barycenter_velocity_norm),
        initial_barycenter_position_cap=float(
            prepared.initial_barycenter_position_cap
        ),
        initial_barycenter_velocity_cap=float(
            prepared.initial_barycenter_velocity_cap
        ),
    )


def _result_content_sha256(
    *,
    initial_snapshot: StateSnapshot,
    force_plan: ForcePlan,
    integration_spec: FixedStepWisdomHolmanSpec,
    coordinate_binding: JacobiCoordinateBinding,
    checkpoint_step_indices: tuple[int, ...],
    checkpoint_epochs: tuple[float, ...],
    run: _WisdomHolmanRun,
    direction: str,
    schedule_content_sha256: str,
) -> str:
    """Hash all retained WH result content; this is not authentication."""

    components = {
        "initial_snapshot": initial_snapshot,
        "force_plan": force_plan,
        "integration_spec": integration_spec,
        "coordinate_binding": coordinate_binding,
        "schedule_binding": {
            "checkpoint_step_indices": checkpoint_step_indices,
            "checkpoint_epochs": checkpoint_epochs,
            "direction": direction,
            "schedule_content_sha256": schedule_content_sha256,
        },
        "runtime": run,
        "public_execution_accounting": _execution_accounting_payload(
            _run_execution_counts(run)
        ),
        "control": {
            "scope": WH_TRAJECTORY_SCOPE,
            "evidence_class": WH_EVIDENCE_CLASS,
            "formal_exact_kepler_subflow_symplectic": True,
            "floating_point_symplectic": False,
            "formal_exact_kepler_subflow_time_reversible": True,
            "floating_point_exactly_reversible": False,
            "dense_output": False,
            "adaptive": False,
            "registry_authorized": False,
            "qualification_authorized": False,
        },
    }
    component_hashes = {
        name: _domain_separated_json_sha256(
            f"{WH_RESULT_CONTENT_CHECKSUM_DOMAIN}.component.{name}", value
        )
        for name, value in components.items()
    }
    payload = {
        "checksum_algorithm": WH_RESULT_CONTENT_CHECKSUM_ALGORITHM,
        "component_hashes": component_hashes,
        "component_order": tuple(components),
        "schema": "jxplanetx.wisdom-holman-result.v1",
    }
    return _domain_separated_json_sha256(
        WH_RESULT_CONTENT_CHECKSUM_DOMAIN, payload
    )


@dataclass(frozen=True, eq=False)
class WisdomHolmanTrajectoryResult:
    """Copied synchronized Cartesian states from the ordered-Jacobi map."""

    snapshot_id: str
    plan_id: str
    backend_id: str
    device: str
    dtype: str
    backend_spec: BackendSpec
    initial_snapshot: StateSnapshot
    force_plan: ForcePlan
    integration_spec: FixedStepWisdomHolmanSpec
    coordinate_binding: JacobiCoordinateBinding
    checkpoint_step_indices: tuple[int, ...]
    checkpoint_epochs: tuple[float, ...]
    checkpoints: tuple[TrajectoryCheckpoint, ...]
    force_model_ids: tuple[str, ...]
    force_ledger: tuple[ForceLedgerEntry, ...]
    completed_steps: int
    primary_map_force_evaluations: int
    validation_replay_force_evaluations: int
    total_public_call_force_evaluations: int
    primary_map_interaction_force_assemblies: int
    validation_replay_interaction_force_assemblies: int
    total_public_call_interaction_force_assemblies: int
    primary_map_kepler_subflow_solves: int
    validation_replay_kepler_subflow_solves: int
    total_public_call_kepler_subflow_solves: int
    primary_map_universal_solver_iterations: int
    validation_replay_universal_solver_iterations: int
    total_public_call_universal_solver_iterations: int
    maximum_universal_solver_iterations: int
    primary_map_universal_solver_bracket_expansions: int
    validation_replay_universal_solver_bracket_expansions: int
    total_public_call_universal_solver_bracket_expansions: int
    maximum_universal_solver_bracket_expansions: int
    primary_map_universal_g_function_evaluations: int
    validation_replay_universal_g_function_evaluations: int
    total_public_call_universal_g_function_evaluations: int
    primary_map_universal_series_terms_evaluated: int
    validation_replay_universal_series_terms_evaluated: int
    total_public_call_universal_series_terms_evaluated: int
    primary_map_coordinate_forward_transforms: int
    validation_replay_coordinate_forward_transforms: int
    total_public_call_coordinate_forward_transforms: int
    primary_map_coordinate_inverse_transforms: int
    validation_replay_coordinate_inverse_transforms: int
    total_public_call_coordinate_inverse_transforms: int
    primary_map_node_guard_evaluations: int
    validation_replay_node_guard_evaluations: int
    total_public_call_node_guard_evaluations: int
    primary_map_path_guard_evaluations: int
    validation_replay_path_guard_evaluations: int
    total_public_call_path_guard_evaluations: int
    maximum_kepler_time_residual: float
    maximum_kepler_residual_tolerance: float
    maximum_kepler_lagrange_identity_error: float
    maximum_kepler_energy_error: float
    maximum_kepler_angular_momentum_error: float
    maximum_translation_force_residual: float
    minimum_jacobi_periapses: tuple[float, ...]
    maximum_jacobi_eccentricities: tuple[float, ...]
    maximum_interaction_force_ratios: tuple[float, ...]
    maximum_orbit_step_fractions: tuple[float, ...]
    maximum_periapse_step_fractions: tuple[float, ...]
    minimum_pair_endpoint_separations: tuple[float, ...]
    minimum_pair_path_lower_bounds: tuple[float, ...]
    minimum_pair_clearance_after_margins: tuple[float, ...]
    minimum_secondary_hill_floor_ratios: tuple[float, ...]
    maximum_barycenter_position_norm: float
    maximum_barycenter_velocity_norm: float
    initial_barycenter_position_cap: float
    initial_barycenter_velocity_cap: float
    direction: str
    schedule_content_sha256: str
    result_content_sha256: str
    method_id: str = FIXED_STEP_WISDOM_HOLMAN_METHOD_ID
    method_class: str = WH_METHOD_CLASS
    principal_order: int = WH_PRINCIPAL_ORDER
    interaction_kick_coefficients: tuple[float, ...] = (
        WH_INTERACTION_KICK_COEFFICIENTS
    )
    kepler_drift_coefficients: tuple[float, ...] = (1.0,)
    composition: str = WH_COMPOSITION
    hamiltonian_split_policy: str = WH_HAMILTONIAN_SPLIT_POLICY
    step_representation: str = WH_STEP_REPRESENTATION
    checkpoint_policy: str = WH_CHECKPOINT_POLICY
    force_plan_scope: str = WH_FORCE_PLAN_SCOPE
    backend_scope: str = WH_BACKEND_SCOPE
    time_semantics: str = WH_TIME_SEMANTICS
    jacobi_coordinate_system: str = WH_JACOBI_COORDINATE_SYSTEM
    canonical_momentum_convention: str = WH_CANONICAL_MOMENTUM_CONVENTION
    physical_mass_policy: str = WH_PHYSICAL_MASS_POLICY
    force_evaluation_accounting: str = WH_FORCE_EVALUATION_ACCOUNTING
    public_execution_accounting_scope: str = (
        WH_PUBLIC_EXECUTION_ACCOUNTING_SCOPE
    )
    validation_replay_policy: str = WH_VALIDATION_REPLAY_POLICY
    validation_replay_count: int = WH_VALIDATION_REPLAY_COUNT
    schedule_checksum_algorithm: str = WH_SCHEDULE_CHECKSUM_ALGORITHM
    schedule_checksum_domain: str = WH_SCHEDULE_CHECKSUM_DOMAIN
    result_content_checksum_algorithm: str = WH_RESULT_CONTENT_CHECKSUM_ALGORITHM
    result_content_checksum_domain: str = WH_RESULT_CONTENT_CHECKSUM_DOMAIN
    scope: str = WH_TRAJECTORY_SCOPE
    evidence_class: str = WH_EVIDENCE_CLASS
    formal_exact_kepler_subflow_symplectic: bool = True
    floating_point_symplectic: bool = False
    formal_exact_kepler_subflow_time_reversible: bool = True
    floating_point_exactly_reversible: bool = False
    dense_output: bool = False
    adaptive: bool = False
    registry_authorized: bool = False
    qualification_authorized: bool = False

    def __post_init__(self) -> None:
        for name in (
            "interaction_kick_coefficients",
            "kepler_drift_coefficients",
        ):
            value = getattr(self, name)
            if type(value) is not tuple or any(
                type(component) is not float for component in value
            ):
                raise TrajectoryContractError(
                    f"{name} must be an exact tuple of built-in floats"
                )
        exact = {
            "backend_id": "numpy",
            "device": "cpu",
            "dtype": "float64",
            "method_id": FIXED_STEP_WISDOM_HOLMAN_METHOD_ID,
            "method_class": WH_METHOD_CLASS,
            "principal_order": WH_PRINCIPAL_ORDER,
            "interaction_kick_coefficients": WH_INTERACTION_KICK_COEFFICIENTS,
            "kepler_drift_coefficients": (1.0,),
            "composition": WH_COMPOSITION,
            "hamiltonian_split_policy": WH_HAMILTONIAN_SPLIT_POLICY,
            "step_representation": WH_STEP_REPRESENTATION,
            "checkpoint_policy": WH_CHECKPOINT_POLICY,
            "force_plan_scope": WH_FORCE_PLAN_SCOPE,
            "backend_scope": WH_BACKEND_SCOPE,
            "time_semantics": WH_TIME_SEMANTICS,
            "jacobi_coordinate_system": WH_JACOBI_COORDINATE_SYSTEM,
            "canonical_momentum_convention": WH_CANONICAL_MOMENTUM_CONVENTION,
            "physical_mass_policy": WH_PHYSICAL_MASS_POLICY,
            "force_evaluation_accounting": WH_FORCE_EVALUATION_ACCOUNTING,
            "public_execution_accounting_scope": WH_PUBLIC_EXECUTION_ACCOUNTING_SCOPE,
            "validation_replay_policy": WH_VALIDATION_REPLAY_POLICY,
            "validation_replay_count": WH_VALIDATION_REPLAY_COUNT,
            "schedule_checksum_algorithm": WH_SCHEDULE_CHECKSUM_ALGORITHM,
            "schedule_checksum_domain": WH_SCHEDULE_CHECKSUM_DOMAIN,
            "result_content_checksum_algorithm": WH_RESULT_CONTENT_CHECKSUM_ALGORITHM,
            "result_content_checksum_domain": WH_RESULT_CONTENT_CHECKSUM_DOMAIN,
            "scope": WH_TRAJECTORY_SCOPE,
            "evidence_class": WH_EVIDENCE_CLASS,
            "formal_exact_kepler_subflow_symplectic": True,
            "floating_point_symplectic": False,
            "formal_exact_kepler_subflow_time_reversible": True,
            "floating_point_exactly_reversible": False,
            "dense_output": False,
            "adaptive": False,
            "registry_authorized": False,
            "qualification_authorized": False,
        }
        for name, expected in exact.items():
            value = getattr(self, name)
            if type(value) is not type(expected) or value != expected:
                raise TrajectoryContractError(
                    f"{name} must equal the fixed Wisdom--Holman result value {expected!r}"
                )
        _validate_wisdom_holman_result(self)

    @property
    def checkpoint_count(self) -> int:
        return len(self.checkpoints)

    @property
    def final_epoch(self) -> float:
        return self.checkpoint_epochs[-1]

    @property
    def positions(self) -> tuple[np.ndarray, ...]:
        return tuple(checkpoint.positions for checkpoint in self.checkpoints)

    @property
    def velocities(self) -> tuple[np.ndarray, ...]:
        return tuple(checkpoint.velocities for checkpoint in self.checkpoints)

    @property
    def final_positions(self) -> np.ndarray:
        return self.checkpoints[-1].positions

    @property
    def final_velocities(self) -> np.ndarray:
        return self.checkpoints[-1].velocities

    @property
    def integrated(self) -> bool:
        return True

    @property
    def qualified(self) -> bool:
        return False


_RUN_FIELD_NAMES = tuple(field.name for field in fields(_WisdomHolmanRun))
_RUN_NONCOUNT_FIELD_NAMES = tuple(
    name for name in _RUN_FIELD_NAMES if name not in _EXECUTION_COUNT_NAMES
)


def _public_execution_result_kwargs(run: _WisdomHolmanRun) -> dict[str, int]:
    values: dict[str, int] = {}
    for name, primary in zip(_EXECUTION_COUNT_NAMES, _run_execution_counts(run)):
        values[f"primary_map_{name}"] = primary
        values[f"validation_replay_{name}"] = primary
        values[f"total_public_call_{name}"] = 2 * primary
    return values


def _run_from_result(result: WisdomHolmanTrajectoryResult) -> _WisdomHolmanRun:
    values = {
        name: getattr(result, name) for name in _RUN_NONCOUNT_FIELD_NAMES
    }
    values.update(
        {
            name: getattr(result, f"primary_map_{name}")
            for name in _EXECUTION_COUNT_NAMES
        }
    )
    return _WisdomHolmanRun(**values)


def _validate_lowercase_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TrajectoryContractError(
            f"{label} must be lowercase SHA-256 hexadecimal"
        )
    return value


def _validate_float_tuple(
    value: object,
    label: str,
    length: int,
    *,
    allow_zero: bool,
) -> tuple[float, ...]:
    if type(value) is not tuple or len(value) != length:
        raise TrajectoryContractError(
            f"{label} must be an exact tuple with {length} entries"
        )
    for item in value:
        if (
            type(item) is not float
            or not math.isfinite(item)
            or (item < 0.0 if allow_zero else item <= 0.0)
        ):
            qualifier = "nonnegative" if allow_zero else "positive"
            raise TrajectoryContractError(
                f"{label} entries must be finite {qualifier} built-in floats"
            )
    return value


def _validate_owned_readonly_arrays(
    initial_snapshot: StateSnapshot,
    binding: JacobiCoordinateBinding,
    checkpoints: tuple[TrajectoryCheckpoint, ...],
) -> None:
    retained: list[np.ndarray] = []
    for label, array, dtype, shape in (
        (
            "initial_snapshot.positions",
            initial_snapshot.positions,
            np.dtype(np.float64),
            (len(initial_snapshot.body_ids), 3),
        ),
        (
            "initial_snapshot.velocities",
            initial_snapshot.velocities,
            np.dtype(np.float64),
            (len(initial_snapshot.body_ids), 3),
        ),
        (
            "initial_snapshot.gravitational_parameters",
            initial_snapshot.gravitational_parameters,
            np.dtype(np.float64),
            (len(initial_snapshot.body_ids),),
        ),
        (
            "initial_snapshot.masses",
            initial_snapshot.masses,
            np.dtype(np.float64),
            (len(initial_snapshot.body_ids),),
        ),
        (
            "initial_snapshot.radii",
            initial_snapshot.radii,
            np.dtype(np.float64),
            (len(initial_snapshot.body_ids),),
        ),
        (
            "initial_snapshot.massive",
            initial_snapshot.massive,
            np.dtype(np.bool_),
            (len(initial_snapshot.body_ids),),
        ),
    ):
        if (
            type(array) is not np.ndarray
            or array.dtype != dtype
            or array.shape != shape
            or not array.flags.owndata
            or array.flags.writeable
        ):
            raise TrajectoryContractError(
                f"retained {label} must be an exact owned read-only NumPy array"
            )
        retained.append(array)
    for label, array in (
        ("binding.gravitational_parameters", binding.gravitational_parameters),
        (
            "binding.cumulative_gravitational_parameters",
            binding.cumulative_gravitational_parameters,
        ),
        ("binding.canonical_inertias", binding.canonical_inertias),
        (
            "binding.jacobi_from_cartesian_matrix",
            binding.jacobi_from_cartesian_matrix,
        ),
        (
            "binding.cartesian_from_jacobi_matrix",
            binding.cartesian_from_jacobi_matrix,
        ),
    ):
        if type(array) is not np.ndarray or not array.flags.owndata or array.flags.writeable:
            raise TrajectoryContractError(
                f"retained {label} must be an exact owned read-only NumPy array"
            )
        retained.append(array)
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        for name in ("positions", "velocities"):
            array = getattr(checkpoint, name)
            if (
                type(array) is not np.ndarray
                or array.dtype != np.dtype(np.float64)
                or array.shape != (len(initial_snapshot.body_ids), 3)
                or not array.flags.owndata
                or array.flags.writeable
                or not bool(np.all(np.isfinite(array)))
            ):
                raise TrajectoryContractError(
                    f"checkpoint[{checkpoint_index}].{name} must be an exact owned read-only finite NumPy array"
                )
            retained.append(array)
    for left in range(len(retained) - 1):
        for right in range(left + 1, len(retained)):
            try:
                overlaps = bool(np.shares_memory(retained[left], retained[right]))
            except Exception as exc:  # pragma: no cover - NumPy exact-base path.
                raise TrajectoryContractError(
                    "Wisdom--Holman retained array overlap could not be resolved"
                ) from exc
            if overlaps:
                raise TrajectoryContractError(
                    "Wisdom--Holman retained arrays must not overlap memory"
                )


def _validate_force_ledger_binding(result: WisdomHolmanTrajectoryResult) -> None:
    if result.force_model_ids != (NewtonianPointMass.MODEL_ID,):
        raise TrajectoryContractError(
            "Wisdom--Holman force_model_ids must contain only Newtonian gravity"
        )
    if (
        type(result.force_ledger) is not tuple
        or len(result.force_ledger) != 1
        or type(result.force_ledger[0]) is not ForceLedgerEntry
    ):
        raise TrajectoryContractError(
            "Wisdom--Holman force_ledger must contain one exact ForceLedgerEntry"
        )
    entry = _validate_ledger_schema(result.force_ledger[0])
    model = result.force_plan.models[0]
    metadata = entry.state_metadata
    snapshot = result.initial_snapshot
    if (
        entry.order != 0
        or entry.model_id != NewtonianPointMass.MODEL_ID
        or entry.role != "NEWTONIAN_BASE"
        or entry.source_ids != snapshot.body_ids
        or entry.target_ids != snapshot.body_ids
        or entry.backend_spec is not result.backend_spec
        or entry.tile_size != result.backend_spec.tile_size
        or entry.determinism_scope != result.backend_spec.determinism_scope
        or entry.assumptions != KDK_NEWTONIAN_LEDGER_ASSUMPTIONS
        or entry.evidence_class != WH_EVIDENCE_CLASS
        or entry.registry_authorized is not False
        or entry.qualification_authorized is not False
        or model.source_ids != entry.source_ids  # type: ignore[union-attr]
        or model.target_ids != entry.target_ids  # type: ignore[union-attr]
        or metadata.snapshot_id != snapshot.snapshot_id
        or metadata.epoch != snapshot.epoch
        or metadata.unit_system_id != snapshot.unit_system_id
        or metadata.time_scale != snapshot.time_scale
        or metadata.frame != snapshot.frame
        or metadata.origin != snapshot.origin
        or metadata.axes != snapshot.axes
        or metadata.length_unit != snapshot.length_unit
        or metadata.time_unit != snapshot.time_unit
        or metadata.mass_unit != snapshot.mass_unit
        or metadata.body_ids != snapshot.body_ids
        or metadata.provenance_source_id != snapshot.provenance.source_id
        or metadata.provenance_citation != snapshot.provenance.citation
        or metadata.provenance_version != snapshot.provenance.version
        or metadata.provenance_sha256 != snapshot.provenance.sha256
    ):
        raise TrajectoryContractError(
            "Wisdom--Holman force-ledger binding is inconsistent"
        )


def _validate_wisdom_holman_result(
    result: WisdomHolmanTrajectoryResult,
) -> None:
    if type(result) is not WisdomHolmanTrajectoryResult:
        raise TrajectoryContractError(
            "result must be an exact WisdomHolmanTrajectoryResult"
        )
    for name in (
        "snapshot_id",
        "plan_id",
        "direction",
    ):
        _exact_text(getattr(result, name), name)
    _validate_lowercase_sha256(
        result.schedule_content_sha256, "schedule_content_sha256"
    )
    _validate_lowercase_sha256(
        result.result_content_sha256, "result_content_sha256"
    )
    if (
        type(result.checkpoint_step_indices) is not tuple
        or any(type(value) is not int for value in result.checkpoint_step_indices)
    ):
        raise TrajectoryContractError(
            "checkpoint_step_indices must be an exact tuple of built-in integers"
        )
    if (
        type(result.checkpoint_epochs) is not tuple
        or any(
            type(value) is not float or not math.isfinite(value)
            for value in result.checkpoint_epochs
        )
    ):
        raise TrajectoryContractError(
            "checkpoint_epochs must be an exact tuple of finite built-in floats"
        )
    if (
        type(result.checkpoints) is not tuple
        or any(
            type(value) is not TrajectoryCheckpoint for value in result.checkpoints
        )
    ):
        raise TrajectoryContractError(
            "checkpoints must be an exact tuple of TrajectoryCheckpoint values"
        )
    if (
        type(result.force_model_ids) is not tuple
        or not result.force_model_ids
        or any(
            type(value) is not str or not value or value.strip() != value
            for value in result.force_model_ids
        )
    ):
        raise TrajectoryContractError(
            "force_model_ids must be an exact nonempty tuple of built-in strings"
        )
    for name in (
        "completed_steps",
        "maximum_universal_solver_iterations",
        "maximum_universal_solver_bracket_expansions",
    ) + tuple(
        f"{prefix}_{counter}"
        for counter in _EXECUTION_COUNT_NAMES
        for prefix in (
            "primary_map",
            "validation_replay",
            "total_public_call",
        )
    ):
        value = getattr(result, name)
        if type(value) is not int or value < 0:
            raise TrajectoryContractError(
                f"{name} must be an exact nonnegative built-in integer"
            )

    _validate_snapshot_schema(result.initial_snapshot, "initial_snapshot")
    _validate_force_plan_schema(result.force_plan, "force_plan")
    _validate_backend_schema(result.backend_spec, "backend_spec")
    if type(result.integration_spec) is not FixedStepWisdomHolmanSpec:
        raise TrajectoryContractError(
            "integration_spec must be an exact FixedStepWisdomHolmanSpec"
        )
    result.integration_spec.__post_init__()
    if type(result.coordinate_binding) is not JacobiCoordinateBinding:
        raise TrajectoryContractError(
            "coordinate_binding must be an exact JacobiCoordinateBinding"
        )
    result.coordinate_binding.__post_init__()
    if result.backend_spec is not result.force_plan.backend:
        raise TrajectoryContractError(
            "backend_spec must be the retained force-plan backend"
        )
    if result.snapshot_id != result.initial_snapshot.snapshot_id:
        raise TrajectoryContractError(
            "snapshot_id does not match retained initial_snapshot"
        )
    if result.plan_id != result.force_plan.plan_id:
        raise TrajectoryContractError("plan_id does not match retained force_plan")
    if result.coordinate_binding.body_ids != result.initial_snapshot.body_ids:
        raise TrajectoryContractError(
            "coordinate binding body order differs from initial_snapshot"
        )
    if (
        result.coordinate_binding.gravitational_parameters.tobytes(order="C")
        != result.initial_snapshot.gravitational_parameters.tobytes(order="C")
    ):
        raise TrajectoryContractError(
            "coordinate binding GM bytes differ from retained initial_snapshot"
        )

    _validate_semantic_boundary(result.initial_snapshot)
    retained_model = _validate_force_plan_scope(
        result.initial_snapshot, result.force_plan, result.integration_spec
    )
    _validate_parameter_metadata_schema(retained_model)
    _validate_model_sequence(result.force_plan.models)
    _validate_dependencies(result.force_plan.models)
    _validate_parameter_contracts(result.initial_snapshot, result.force_plan)
    all_epochs = _step_epochs(
        result.initial_snapshot.epoch, result.integration_spec
    )
    expected_checkpoint_epochs = tuple(
        all_epochs[index]
        for index in result.integration_spec.checkpoint_step_indices
    )
    if (
        result.checkpoint_step_indices
        != result.integration_spec.checkpoint_step_indices
    ):
        raise TrajectoryContractError(
            "checkpoint_step_indices differ from integration_spec"
        )
    if result.checkpoint_epochs != expected_checkpoint_epochs:
        raise TrajectoryContractError(
            "checkpoint_epochs differ from the fixed integer map lattice"
        )
    _validate_metadata_range(
        result.force_plan,
        min(all_epochs[0], all_epochs[-1]),
        max(all_epochs[0], all_epochs[-1]),
    )
    if result.direction != result.integration_spec.direction:
        raise TrajectoryContractError("direction differs from signed fixed_step")
    if result.completed_steps != result.integration_spec.completed_steps:
        raise TrajectoryContractError("completed_steps differs from the schedule")

    expected_force_evaluations = result.completed_steps + 1
    expected_solves = result.completed_steps * (
        len(result.initial_snapshot.body_ids) - 1
    )
    primary_exact_counts = {
        "force_evaluations": expected_force_evaluations,
        "interaction_force_assemblies": expected_force_evaluations,
        "kepler_subflow_solves": expected_solves,
        "coordinate_forward_transforms": 1,
        "coordinate_inverse_transforms": 2 * result.completed_steps,
        "node_guard_evaluations": 1 + 2 * result.completed_steps,
        "path_guard_evaluations": result.completed_steps,
    }
    for name, expected in primary_exact_counts.items():
        if getattr(result, f"primary_map_{name}") != expected:
            raise TrajectoryContractError(
                f"primary_map_{name} does not match fixed Wisdom--Holman accounting"
            )
    for name in _EXECUTION_COUNT_NAMES:
        primary = getattr(result, f"primary_map_{name}")
        replay = getattr(result, f"validation_replay_{name}")
        total = getattr(result, f"total_public_call_{name}")
        if replay != primary or total != primary + replay:
            raise TrajectoryContractError(
                f"{name} does not bind primary map, mandatory replay, and total public-call work"
            )
    if (
        result.maximum_universal_solver_iterations
        > result.integration_spec.kepler_solver.maximum_iterations
        or result.maximum_universal_solver_bracket_expansions
        > result.integration_spec.kepler_solver.maximum_bracket_expansions
        or result.maximum_universal_solver_iterations
        > result.primary_map_universal_solver_iterations
        or result.maximum_universal_solver_bracket_expansions
        > result.primary_map_universal_solver_bracket_expansions
        or result.primary_map_universal_g_function_evaluations
        < result.primary_map_kepler_subflow_solves
        or result.primary_map_universal_series_terms_evaluated
        % (4 * result.integration_spec.kepler_solver.series_term_count)
        != 0
    ):
        raise TrajectoryContractError(
            "Wisdom--Holman universal-solver accounting is inconsistent"
        )

    body_count = len(result.initial_snapshot.body_ids)
    secondary_count = body_count - 1
    pair_count = body_count * (body_count - 1) // 2
    secondary_pair_count = secondary_count * (secondary_count - 1) // 2
    for name, length, allow_zero in (
        ("minimum_jacobi_periapses", secondary_count, False),
        ("maximum_jacobi_eccentricities", secondary_count, True),
        ("maximum_interaction_force_ratios", secondary_count, True),
        ("maximum_orbit_step_fractions", secondary_count, False),
        ("maximum_periapse_step_fractions", secondary_count, False),
        ("minimum_pair_endpoint_separations", pair_count, False),
        ("minimum_pair_path_lower_bounds", pair_count, False),
        ("minimum_pair_clearance_after_margins", pair_count, False),
        ("minimum_secondary_hill_floor_ratios", secondary_pair_count, False),
    ):
        _validate_float_tuple(
            getattr(result, name), name, length, allow_zero=allow_zero
        )
    for name in (
        "maximum_kepler_time_residual",
        "maximum_kepler_residual_tolerance",
        "maximum_kepler_lagrange_identity_error",
        "maximum_kepler_energy_error",
        "maximum_kepler_angular_momentum_error",
        "maximum_translation_force_residual",
        "maximum_barycenter_position_norm",
        "maximum_barycenter_velocity_norm",
        "initial_barycenter_position_cap",
        "initial_barycenter_velocity_cap",
    ):
        value = getattr(result, name)
        if type(value) is not float or not math.isfinite(value) or value < 0.0:
            raise TrajectoryContractError(
                f"{name} must be a finite nonnegative built-in float"
            )
    if result.maximum_kepler_residual_tolerance <= 0.0:
        raise TrajectoryContractError(
            "maximum_kepler_residual_tolerance must be positive"
        )
    if result.maximum_kepler_time_residual > result.maximum_kepler_residual_tolerance:
        raise TrajectoryContractError(
            "retained maximum Kepler residual exceeds its retained tolerance"
        )
    if any(
        value < result.integration_spec.minimum_jacobi_periapse
        for value in result.minimum_jacobi_periapses
    ):
        raise TrajectoryContractError(
            "retained Jacobi periapse evidence violates its caller floor"
        )
    if any(
        value > WH_MAXIMUM_JACOBI_ECCENTRICITY
        for value in result.maximum_jacobi_eccentricities
    ) or any(
        value > WH_MAXIMUM_INTERACTION_TO_KEPLER_FORCE_RATIO
        for value in result.maximum_interaction_force_ratios
    ) or any(
        value > WH_MAXIMUM_ORBIT_STEP_FRACTION
        for value in result.maximum_orbit_step_fractions
    ) or any(
        value > WH_MAXIMUM_PERIAPSE_STEP_FRACTION
        for value in result.maximum_periapse_step_fractions
    ):
        raise TrajectoryContractError(
            "retained Wisdom--Holman hierarchy evidence violates the v1 envelope"
        )
    if any(value <= 1.0 for value in result.minimum_secondary_hill_floor_ratios):
        raise TrajectoryContractError(
            "retained secondary pair does not clear its three-Hill-radius floor"
        )

    if len(result.checkpoints) != len(expected_checkpoint_epochs):
        raise TrajectoryContractError(
            "Wisdom--Holman checkpoints do not align with the requested lattice"
        )
    for output_index, (checkpoint, step_index, epoch) in enumerate(
        zip(
            result.checkpoints,
            result.checkpoint_step_indices,
            result.checkpoint_epochs,
        )
    ):
        _validate_checkpoint_schema(checkpoint, f"checkpoints[{output_index}]")
        if (
            checkpoint.index != output_index
            or checkpoint.epoch != epoch
            or checkpoint.accepted_steps != step_index
            or checkpoint.rejected_steps != 0
            or checkpoint.body_ids != result.initial_snapshot.body_ids
            or checkpoint.backend_id != "numpy"
            or checkpoint.device != "cpu"
            or checkpoint.dtype != "float64"
            or checkpoint.evidence_class != WH_EVIDENCE_CLASS
            or checkpoint.registry_authorized is not False
            or checkpoint.qualification_authorized is not False
        ):
            raise TrajectoryContractError(
                "Wisdom--Holman checkpoint binding is inconsistent"
            )

    backend = resolve_backend(result.backend_spec)
    with backend.activate():
        initial_arrays = _validate_state(backend, result.initial_snapshot)
        if bool(np.any(~initial_arrays[5])):
            raise TrajectoryContractError(
                "every retained Wisdom--Holman body must remain active"
            )
        if bool(np.any(initial_arrays[2] <= np.float64(0.0))):
            raise TrajectoryContractError(
                "every retained Wisdom--Holman body must retain positive GM"
            )
    total_secondary_ratio = _sum_1d_fixed(initial_arrays[2], 1) / float(
        initial_arrays[2][0]
    )
    if (
        not math.isfinite(total_secondary_ratio)
        or total_secondary_ratio
        > WH_MAXIMUM_TOTAL_SECONDARY_TO_PRIMARY_GM_RATIO
    ):
        raise TrajectoryContractError(
            "retained total secondary-to-primary GM ratio exceeds the v1 envelope"
        )
    _validate_owned_readonly_arrays(
        result.initial_snapshot, result.coordinate_binding, result.checkpoints
    )
    if (
        result.checkpoints[0].positions.tobytes(order="C")
        != result.initial_snapshot.positions.tobytes(order="C")
        or result.checkpoints[0].velocities.tobytes(order="C")
        != result.initial_snapshot.velocities.tobytes(order="C")
    ):
        raise TrajectoryContractError(
            "initial Wisdom--Holman checkpoint differs from initial_snapshot"
        )
    _validate_force_ledger_binding(result)

    run = _run_from_result(result)
    expected_schedule_checksum = _schedule_content_sha256(
        checkpoint_step_indices=result.checkpoint_step_indices,
        checkpoint_epochs=result.checkpoint_epochs,
        fixed_step=result.integration_spec.fixed_step,
        direction=result.direction,
        completed_steps=result.completed_steps,
        body_count=len(result.initial_snapshot.body_ids),
    )
    if result.schedule_content_sha256 != expected_schedule_checksum:
        raise TrajectoryContractError(
            "Wisdom--Holman schedule checksum does not match retained metadata"
        )
    expected_result_checksum = _result_content_sha256(
        initial_snapshot=result.initial_snapshot,
        force_plan=result.force_plan,
        integration_spec=result.integration_spec,
        coordinate_binding=result.coordinate_binding,
        checkpoint_step_indices=result.checkpoint_step_indices,
        checkpoint_epochs=result.checkpoint_epochs,
        run=run,
        direction=result.direction,
        schedule_content_sha256=result.schedule_content_sha256,
    )
    if result.result_content_sha256 != expected_result_checksum:
        raise TrajectoryContractError(
            "Wisdom--Holman result checksum does not match retained content"
        )

    replay = _execute_wisdom_holman_map(
        initial_snapshot=result.initial_snapshot,
        force_plan=result.force_plan,
        integration_spec=result.integration_spec,
        binding=result.coordinate_binding,
        all_epochs=all_epochs,
        checkpoint_epochs=expected_checkpoint_epochs,
    )
    if _canonical_content(run) != _canonical_content(replay):
        raise TrajectoryContractError(
            "Wisdom--Holman retained runtime content differs from deterministic semantic replay"
        )


def integrate_wisdom_holman_trajectory(
    snapshot: StateSnapshot,
    plan: ForcePlan,
    spec: FixedStepWisdomHolmanSpec,
) -> WisdomHolmanTrajectoryResult:
    """Integrate one closed Newtonian hierarchy on the fixed Jacobi map."""

    if type(snapshot) is not StateSnapshot:
        raise TrajectoryContractError("snapshot must be an exact StateSnapshot")
    if type(plan) is not ForcePlan:
        raise TrajectoryContractError("plan must be an exact ForcePlan")
    if type(spec) is not FixedStepWisdomHolmanSpec:
        raise TrajectoryContractError(
            "spec must be an exact FixedStepWisdomHolmanSpec"
        )
    _validate_snapshot_schema(snapshot, "snapshot")
    _validate_force_plan_schema(plan, "plan")
    spec.__post_init__()
    _validate_semantic_boundary(snapshot)
    model = _validate_force_plan_scope(snapshot, plan, spec)
    _validate_parameter_metadata_schema(model)
    _validate_model_sequence(plan.models)
    _validate_dependencies(plan.models)
    _validate_parameter_contracts(snapshot, plan)
    backend = resolve_backend(plan.backend)
    if backend.name != "numpy" or backend.device != "cpu":
        raise TrajectoryContractError(
            "Wisdom--Holman v1 requires the exact NumPy CPU backend"
        )
    for name in (
        "positions",
        "velocities",
        "gravitational_parameters",
        "masses",
        "radii",
        "massive",
    ):
        if type(getattr(snapshot, name)) is not np.ndarray:
            raise TrajectoryContractError(
                f"Wisdom--Holman input {name} must be an exact NumPy ndarray"
            )

    all_epochs = _step_epochs(snapshot.epoch, spec)
    checkpoint_epochs = tuple(
        all_epochs[index] for index in spec.checkpoint_step_indices
    )
    _validate_metadata_range(
        plan,
        min(all_epochs[0], all_epochs[-1]),
        max(all_epochs[0], all_epochs[-1]),
    )
    with backend.activate():
        validated_state = _validate_state(backend, snapshot)
        if bool(np.any(~validated_state[5])):
            raise TrajectoryContractError(
                "Wisdom--Holman requires every body to be marked massive/active"
            )
        if bool(np.any(validated_state[2] <= np.float64(0.0))):
            raise TrajectoryContractError(
                "Wisdom--Holman requires positive GM for every body"
            )
        total_secondary_ratio = _sum_1d_fixed(validated_state[2], 1) / float(
            validated_state[2][0]
        )
        if (
            not math.isfinite(total_secondary_ratio)
            or total_secondary_ratio
            > WH_MAXIMUM_TOTAL_SECONDARY_TO_PRIMARY_GM_RATIO
        ):
            raise TrajectoryDomainError(
                "total secondary-to-primary GM ratio exceeds the Wisdom--Holman v1 envelope"
            )

        initial_snapshot = _readonly_snapshot_copy(snapshot, validated_state)
        force_plan = replace(
            plan,
            backend=replace(plan.backend),
            models=(replace(plan.models[0]),),
        )
        integration_spec = replace(
            spec, kepler_solver=replace(spec.kepler_solver)
        )
        binding = _build_jacobi_coordinate_binding(
            initial_snapshot.body_ids,
            initial_snapshot.gravitational_parameters,
        )
        run = _execute_wisdom_holman_map(
            initial_snapshot=initial_snapshot,
            force_plan=force_plan,
            integration_spec=integration_spec,
            binding=binding,
            all_epochs=all_epochs,
            checkpoint_epochs=checkpoint_epochs,
        )

    schedule_checksum = _schedule_content_sha256(
        checkpoint_step_indices=integration_spec.checkpoint_step_indices,
        checkpoint_epochs=checkpoint_epochs,
        fixed_step=integration_spec.fixed_step,
        direction=integration_spec.direction,
        completed_steps=integration_spec.completed_steps,
        body_count=len(initial_snapshot.body_ids),
    )
    result_checksum = _result_content_sha256(
        initial_snapshot=initial_snapshot,
        force_plan=force_plan,
        integration_spec=integration_spec,
        coordinate_binding=binding,
        checkpoint_step_indices=integration_spec.checkpoint_step_indices,
        checkpoint_epochs=checkpoint_epochs,
        run=run,
        direction=integration_spec.direction,
        schedule_content_sha256=schedule_checksum,
    )
    return WisdomHolmanTrajectoryResult(
        snapshot_id=initial_snapshot.snapshot_id,
        plan_id=force_plan.plan_id,
        backend_id="numpy",
        device="cpu",
        dtype="float64",
        backend_spec=force_plan.backend,
        initial_snapshot=initial_snapshot,
        force_plan=force_plan,
        integration_spec=integration_spec,
        coordinate_binding=binding,
        checkpoint_step_indices=integration_spec.checkpoint_step_indices,
        checkpoint_epochs=checkpoint_epochs,
        completed_steps=integration_spec.completed_steps,
        direction=integration_spec.direction,
        schedule_content_sha256=schedule_checksum,
        result_content_sha256=result_checksum,
        **{
            name: getattr(run, name) for name in _RUN_NONCOUNT_FIELD_NAMES
        },
        **_public_execution_result_kwargs(run),
    )


__all__ = [
    "JacobiCoordinateBinding",
    "WH_TRAJECTORY_SCOPE",
    "WisdomHolmanTrajectoryResult",
    "integrate_wisdom_holman_trajectory",
]
