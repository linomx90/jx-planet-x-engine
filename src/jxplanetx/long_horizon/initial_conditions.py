"""Content-addressed observed-elements to Cartesian genesis contracts.

This module performs one deliberately narrow operation: it converts the
predeclared heliocentric osculating-element benchmarks into primary-relative
Cartesian binary64 states.  The conversion is a two-body numerical genesis
model, not an orbit fit, an ephemeris evaluation, or scientific inference.

Every state binds the population, retained raw-source digest, perturbation
ensemble, materialized offsets, numerical model, and converter implementation.
Construction also performs an independent Cartesian-to-elements round trip and
fails closed when the recovered elements exceed the locked tolerances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping

from .science_inputs import (
    DETACHED_TNO_SCREEN_V1,
    DETACHED_TNO_SENSITIVITY_V1,
    MaterializedPerturbationEnsemble,
    OrbitalElements,
    PerturbedOrbit,
    SCATTERING_DISK_CONTROL_V1,
    SCATTERING_DISK_SENSITIVITY_V1,
    SCIENCE_INPUT_CLAIM_CEILING,
    SCIENCE_INPUT_NONCLAIMS,
    ScienceInputContractError,
    ScientificPopulationBlock,
    SeededPerturbationEnsembleSpec,
    materialize_perturbation_ensemble,
)


GENESIS_MODEL_SCHEMA = "jx.long-horizon.cartesian-genesis-model.v1"
GENESIS_STATE_SCHEMA = "jx.long-horizon.cartesian-initial-state.v1"
GENESIS_BUNDLE_SCHEMA = "jx.long-horizon.initial-condition-bundle.v1"
GENESIS_INVENTORY_SCHEMA = "jx.long-horizon.initial-condition-inventory.v1"
GENESIS_STATUS = "NUMERICAL_INITIAL_CONDITION_ONLY"
GENESIS_NONCLAIMS = SCIENCE_INPUT_NONCLAIMS + (
    "not an ephemeris evaluation or observational orbit fit",
    "not an exact heliocenter-to-inner-system-monopole ephemeris translation",
    "not cross-runtime bitwise reproducibility evidence",
    "not a many-body trajectory or long-duration qualification result",
)

BINARY64_SEMANTICS = (
    "IEEE_754_BINARY64_PYTHON_FLOAT_ROUND_TO_NEAREST_TIES_TO_EVEN"
)
KEPLER_SOLVER_ID = "SAFEGUARDED_NEWTON_ELLIPTIC_BINARY64_V1"
ROTATION_POLICY = "R3_NODE_R1_INCLINATION_R3_ARGUMENT_BINARY64_V1"
ROUND_TRIP_POLICY = "INDEPENDENT_VECTOR_INVARIANTS_TO_ELEMENTS_BINARY64_V1"
STATE_ORDER_POLICY = "POPULATION_NOMINAL_ORDER_THEN_MATERIALIZED_ENSEMBLE_ORDER"

SOURCE_ELEMENT_FRAME = "HELIOCENTRIC_IAU76_80_ECLIPTIC_J2000_OSCULATING"
OUTPUT_CARTESIAN_FRAME = "PRIMARY_RELATIVE_IAU76_80_ECLIPTIC_J2000"
OUTPUT_COORDINATE_SYSTEM = "PRIMARY_RELATIVE_CARTESIAN_RIGHT_HANDED"
EPOCH_SCALE = "TDB"
EPOCH_UNIT = "JULIAN_DATE_DAY"
LENGTH_UNIT = "AU"
TIME_UNIT = "JULIAN_YEAR_365.25_D"
VELOCITY_UNIT = "AU_PER_JULIAN_YEAR_365.25_D"
GM_UNIT = "AU3_PER_JULIAN_YEAR_365.25_D2"

# NAIF DE440 BODY10_GM = 132712440041.27942 km^3/s^2, converted using the
# defined exact astronomical unit (149597870.7 km), SI day (86400 s), and
# Julian year (365.25 d), with the final operation rounded to binary64.
DE440_SOLAR_GM_AU3_PER_JULIAN_YEAR2 = float.fromhex(
    "0x1.3bd0beccac6afp+5"
)
DE440_GM_ARTIFACT_SHA256 = (
    "924ddf4fb9ead9fe8a1aa55780bcabde40b09d00065d58226e24b68d8092f140"
)
# Exact binary64 value retained in row 0 of the frozen canonical restricted-WH
# DE441 control source.  That row is the effective inner-system monopole used
# by the actual force model, not the standalone NAIF body-10 solar GM.
DE441_CONTROL_PRIMARY_GM_AU3_PER_JULIAN_YEAR2 = float.fromhex(
    "0x1.3bd13a81d2262p+5"
)
DE441_CONTROL_STATE_SHA256 = (
    "edfbfd50ca29b28b46fe43cae8cb99bd73a9303a5f2e5babf7bf9ad2a820ff14"
)
DE441_CONTROL_STATE_PACKAGE_PATH = (
    "jxplanetx/data/long_horizon_source_model_v1/de441_control_state.csv"
)
DE441_CONTROL_SOURCE_MODEL_MANIFEST_PACKAGE_PATH = (
    "jxplanetx/data/long_horizon_source_model_v1/MANIFEST.json"
)
DE441_CONTROL_SOURCE_MODEL_MANIFEST_SHA256 = (
    "3695a886bae1b5cbb30daefea6a09dea6305a5b24cd0ccd6aae5d919633de9a1"
)

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SHA256_HEX = frozenset("0123456789abcdef")
_ELEMENT_NAMES = frozenset(
    (
        "semimajor_axis_AU",
        "eccentricity",
        "inclination_rad",
        "longitude_of_ascending_node_rad",
        "argument_of_periapsis_rad",
        "mean_anomaly_rad",
    )
)


class InitialConditionContractError(ValueError):
    """Observed-state genesis input or output violates its closed contract."""


def _identifier(value: object, label: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise InitialConditionContractError(
            f"{label} must match {_IDENTIFIER.pattern!r}"
        )
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise InitialConditionContractError(
            f"{label} must be a nonempty, trimmed built-in string"
        )
    return value


def _finite_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise InitialConditionContractError(
            f"{label} must be a finite built-in float"
        )
    return value


def _positive_float(value: object, label: str) -> float:
    result = _finite_float(value, label)
    if result <= 0.0:
        raise InitialConditionContractError(f"{label} must be positive")
    return result


def _sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise InitialConditionContractError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _text_tuple(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not tuple or any(
        type(item) is not str or not item or item.strip() != item
        for item in value
    ):
        raise InitialConditionContractError(
            f"{label} must be an exact tuple of nonempty trimmed strings"
        )
    return value


def _vec3(value: object, label: str) -> tuple[float, float, float]:
    if (
        type(value) is not tuple
        or len(value) != 3
        or any(type(item) is not float or not math.isfinite(item) for item in value)
    ):
        raise InitialConditionContractError(
            f"{label} must be an exact tuple of three finite built-in floats"
        )
    return value


def _canonical_sha256(domain: bytes, record: Mapping[str, Any]) -> str:
    payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(domain + b"\0" + payload).hexdigest()


def _float_hex(value: float) -> str:
    return value.hex()


def _require_binary64_runtime() -> None:
    if (
        sys.float_info.radix != 2
        or sys.float_info.mant_dig != 53
        or sys.float_info.max_exp != 1024
        or not float.__getformat__("double").startswith("IEEE")
    ):
        raise InitialConditionContractError(
            "observed-state genesis requires IEEE-754 binary64 Python floats"
        )


def _solve_eccentric_anomaly(
    mean_anomaly: float,
    eccentricity: float,
    maximum_iterations: int,
    residual_tolerance: float,
) -> float:
    """Solve E - e*sin(E) = M with a fixed safeguarded binary64 policy."""

    lower = 0.0
    upper = math.tau
    estimate = mean_anomaly if eccentricity < 0.8 else math.pi
    for _ in range(maximum_iterations):
        sine = math.sin(estimate)
        cosine = math.cos(estimate)
        residual = estimate - eccentricity * sine - mean_anomaly
        if residual > 0.0:
            upper = estimate
        elif residual < 0.0:
            lower = estimate
        else:
            return estimate
        derivative = 1.0 - eccentricity * cosine
        candidate = estimate - residual / derivative
        if (
            not math.isfinite(candidate)
            or candidate <= lower
            or candidate >= upper
        ):
            candidate = 0.5 * (lower + upper)
        if candidate == estimate:
            break
        estimate = candidate
    residual = estimate - eccentricity * math.sin(estimate) - mean_anomaly
    if not math.isfinite(residual) or abs(residual) > residual_tolerance:
        raise InitialConditionContractError(
            "binary64 Kepler solve did not meet its locked residual tolerance"
        )
    return estimate


def _elements_to_components(
    elements: OrbitalElements,
    primary_gm: float,
    maximum_iterations: int,
    residual_tolerance: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Apply the locked perifocal state and explicit 3-1-3 rotation."""

    semimajor_axis = elements.semimajor_axis_AU
    eccentricity = elements.eccentricity
    eccentric_anomaly = _solve_eccentric_anomaly(
        elements.mean_anomaly_rad,
        eccentricity,
        maximum_iterations,
        residual_tolerance,
    )
    sine_e = math.sin(eccentric_anomaly)
    cosine_e = math.cos(eccentric_anomaly)
    beta = math.sqrt((1.0 - eccentricity) * (1.0 + eccentricity))
    denominator = 1.0 - eccentricity * cosine_e
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise InitialConditionContractError(
            "elliptic anomaly produced a nonpositive radial denominator"
        )

    perifocal_x = semimajor_axis * (cosine_e - eccentricity)
    perifocal_y = semimajor_axis * beta * sine_e
    speed_scale = math.sqrt(primary_gm / semimajor_axis) / denominator
    perifocal_vx = -speed_scale * sine_e
    perifocal_vy = speed_scale * beta * cosine_e

    node = elements.longitude_of_ascending_node_rad
    inclination = elements.inclination_rad
    argument = elements.argument_of_periapsis_rad
    cosine_node = math.cos(node)
    sine_node = math.sin(node)
    cosine_inclination = math.cos(inclination)
    sine_inclination = math.sin(inclination)
    cosine_argument = math.cos(argument)
    sine_argument = math.sin(argument)

    r11 = cosine_node * cosine_argument - (
        sine_node * sine_argument * cosine_inclination
    )
    r12 = -cosine_node * sine_argument - (
        sine_node * cosine_argument * cosine_inclination
    )
    r21 = sine_node * cosine_argument + (
        cosine_node * sine_argument * cosine_inclination
    )
    r22 = -sine_node * sine_argument + (
        cosine_node * cosine_argument * cosine_inclination
    )
    r31 = sine_argument * sine_inclination
    r32 = cosine_argument * sine_inclination

    position = (
        r11 * perifocal_x + r12 * perifocal_y,
        r21 * perifocal_x + r22 * perifocal_y,
        r31 * perifocal_x + r32 * perifocal_y,
    )
    velocity = (
        r11 * perifocal_vx + r12 * perifocal_vy,
        r21 * perifocal_vx + r22 * perifocal_vy,
        r31 * perifocal_vx + r32 * perifocal_vy,
    )
    _vec3(position, "generated position")
    _vec3(velocity, "generated velocity")
    return position, velocity


def _dot(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _cross(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm(value: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(value, value))


def _clamp_unit(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _components_to_element_values(
    position: tuple[float, float, float],
    velocity: tuple[float, float, float],
    primary_gm: float,
) -> tuple[float, float, float, float, float, float]:
    """Recover nonsingular elliptic elements from vector invariants."""

    radius = _norm(position)
    speed_squared = _dot(velocity, velocity)
    if radius <= 0.0:
        raise InitialConditionContractError("Cartesian radius must be positive")
    angular_momentum = _cross(position, velocity)
    angular_momentum_norm = _norm(angular_momentum)
    if angular_momentum_norm <= 0.0:
        raise InitialConditionContractError(
            "Cartesian state has zero specific angular momentum"
        )
    node = (-angular_momentum[1], angular_momentum[0], 0.0)
    node_norm = _norm(node)

    velocity_cross_h = _cross(velocity, angular_momentum)
    eccentricity_vector = tuple(
        velocity_cross_h[index] / primary_gm - position[index] / radius
        for index in range(3)
    )
    eccentricity = _norm(eccentricity_vector)
    specific_energy = 0.5 * speed_squared - primary_gm / radius
    if specific_energy >= 0.0 or not 0.0 < eccentricity < 1.0:
        raise InitialConditionContractError(
            "Cartesian state is not a nonsingular bound elliptic orbit"
        )
    semimajor_axis = -primary_gm / (2.0 * specific_energy)
    inclination = math.acos(
        _clamp_unit(angular_momentum[2] / angular_momentum_norm)
    )
    if node_norm <= 0.0:
        raise InitialConditionContractError(
            "Cartesian state has a singular equatorial ascending node"
        )
    ascending_node = math.atan2(node[1], node[0]) % math.tau

    node_cross_eccentricity = _cross(node, eccentricity_vector)
    argument_sine = _dot(
        node_cross_eccentricity, angular_momentum
    ) / (node_norm * eccentricity * angular_momentum_norm)
    argument_cosine = _dot(node, eccentricity_vector) / (
        node_norm * eccentricity
    )
    argument = math.atan2(argument_sine, argument_cosine) % math.tau

    eccentric_anomaly_cosine = _clamp_unit(
        (1.0 - radius / semimajor_axis) / eccentricity
    )
    eccentric_anomaly_sine = _dot(position, velocity) / (
        eccentricity * math.sqrt(primary_gm * semimajor_axis)
    )
    eccentric_anomaly_sine = _clamp_unit(eccentric_anomaly_sine)
    eccentric_anomaly = math.atan2(
        eccentric_anomaly_sine, eccentric_anomaly_cosine
    ) % math.tau
    mean_anomaly = (
        eccentric_anomaly - eccentricity * math.sin(eccentric_anomaly)
    ) % math.tau

    values = (
        semimajor_axis,
        eccentricity,
        inclination,
        ascending_node,
        argument,
        mean_anomaly,
    )
    if any(not math.isfinite(value) for value in values):
        raise InitialConditionContractError(
            "Cartesian-to-elements recovery produced a nonfinite value"
        )
    return values


def _angular_distance(left: float, right: float) -> float:
    return abs((left - right + math.pi) % math.tau - math.pi)


def _validate_round_trip(
    original: OrbitalElements,
    recovered: tuple[float, float, float, float, float, float],
    semimajor_axis_relative_tolerance: float,
    scalar_absolute_tolerance: float,
    angle_absolute_tolerance: float,
) -> None:
    names = (
        "semimajor axis",
        "eccentricity",
        "inclination",
        "ascending node",
        "argument of periapsis",
        "mean anomaly",
    )
    expected = (
        original.semimajor_axis_AU,
        original.eccentricity,
        original.inclination_rad,
        original.longitude_of_ascending_node_rad,
        original.argument_of_periapsis_rad,
        original.mean_anomaly_rad,
    )
    semimajor_error = abs(recovered[0] - expected[0])
    if semimajor_error > max(
        scalar_absolute_tolerance,
        semimajor_axis_relative_tolerance * expected[0],
    ):
        raise InitialConditionContractError(
            "Cartesian round trip exceeded the semimajor-axis tolerance"
        )
    if abs(recovered[1] - expected[1]) > scalar_absolute_tolerance:
        raise InitialConditionContractError(
            "Cartesian round trip exceeded the eccentricity tolerance"
        )
    if abs(recovered[2] - expected[2]) > angle_absolute_tolerance:
        raise InitialConditionContractError(
            "Cartesian round trip exceeded the inclination tolerance"
        )
    for index in range(3, 6):
        if _angular_distance(recovered[index], expected[index]) > (
            angle_absolute_tolerance
        ):
            raise InitialConditionContractError(
                f"Cartesian round trip exceeded the {names[index]} tolerance"
            )


def _implementation_sha256() -> str:
    """Bind the complete converter, inverse validator, and build closure.

    A selected-function digest is unsafe here: validation and construction
    also depend on private vector helpers and the public wrapper/build logic.
    The exact installed module bytes are therefore the implementation unit.
    """

    source = Path(__file__).read_bytes()
    return hashlib.sha256(
        b"jx.long-horizon.cartesian-genesis-implementation.v1\0"
        + source
    ).hexdigest()


CONVERTER_IMPLEMENTATION_SHA256 = _implementation_sha256()


@dataclass(frozen=True, slots=True)
class GenesisModel:
    """Fully explicit two-body model used only to generate initial states."""

    model_id: str
    epoch_jd_tdb: float
    source_element_frame: str
    output_cartesian_frame: str
    output_coordinate_system: str
    primary_id: str
    primary_definition: str
    primary_gm_AU3_per_year2: float
    primary_gm_source_id: str
    primary_gm_source_artifact_path: str
    primary_gm_source_artifact_sha256: str
    source_model_manifest_path: str | None
    source_model_manifest_sha256: str | None
    element_origin_to_output_origin_policy: str
    epoch_scale: str = EPOCH_SCALE
    epoch_unit: str = EPOCH_UNIT
    length_unit: str = LENGTH_UNIT
    time_unit: str = TIME_UNIT
    velocity_unit: str = VELOCITY_UNIT
    gm_unit: str = GM_UNIT
    binary64_semantics: str = BINARY64_SEMANTICS
    kepler_solver_id: str = KEPLER_SOLVER_ID
    kepler_maximum_iterations: int = 64
    kepler_residual_tolerance_rad: float = 1.0e-14
    minimum_nonsingular_eccentricity: float = 1.0e-12
    minimum_nonsingular_sin_inclination: float = 1.0e-12
    roundtrip_semimajor_axis_relative_tolerance: float = 2.0e-13
    roundtrip_scalar_absolute_tolerance: float = 3.0e-14
    roundtrip_angle_absolute_tolerance_rad: float = 3.0e-13
    converter_implementation_sha256: str = CONVERTER_IMPLEMENTATION_SHA256
    status: str = GENESIS_STATUS
    claim_ceiling: str = SCIENCE_INPUT_CLAIM_CEILING
    nonclaims: tuple[str, ...] = GENESIS_NONCLAIMS
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_binary64_runtime()
        _identifier(self.model_id, "model_id")
        if _positive_float(self.epoch_jd_tdb, "epoch_jd_tdb") <= 2_400_000.0:
            raise InitialConditionContractError(
                "epoch_jd_tdb must be an explicit modern Julian date"
            )
        for value, label in (
            (self.source_element_frame, "source_element_frame"),
            (self.output_cartesian_frame, "output_cartesian_frame"),
            (self.output_coordinate_system, "output_coordinate_system"),
            (self.primary_id, "primary_id"),
            (self.primary_definition, "primary_definition"),
            (self.primary_gm_source_id, "primary_gm_source_id"),
            (
                self.primary_gm_source_artifact_path,
                "primary_gm_source_artifact_path",
            ),
            (
                self.element_origin_to_output_origin_policy,
                "element_origin_to_output_origin_policy",
            ),
            (self.epoch_scale, "epoch_scale"),
            (self.epoch_unit, "epoch_unit"),
            (self.length_unit, "length_unit"),
            (self.time_unit, "time_unit"),
            (self.velocity_unit, "velocity_unit"),
            (self.gm_unit, "gm_unit"),
            (self.binary64_semantics, "binary64_semantics"),
            (self.kepler_solver_id, "kepler_solver_id"),
            (self.status, "status"),
            (self.claim_ceiling, "claim_ceiling"),
        ):
            _text(value, label)
        _positive_float(
            self.primary_gm_AU3_per_year2,
            "primary_gm_AU3_per_year2",
        )
        _sha256(
            self.primary_gm_source_artifact_sha256,
            "primary_gm_source_artifact_sha256",
        )
        if (self.source_model_manifest_path is None) != (
            self.source_model_manifest_sha256 is None
        ):
            raise InitialConditionContractError(
                "source-model manifest path and digest must be present together"
            )
        if self.source_model_manifest_path is not None:
            _text(self.source_model_manifest_path, "source_model_manifest_path")
            _sha256(
                self.source_model_manifest_sha256,
                "source_model_manifest_sha256",
            )
        _sha256(
            self.converter_implementation_sha256,
            "converter_implementation_sha256",
        )
        _text_tuple(self.nonclaims, "nonclaims")
        if (
            self.source_element_frame != SOURCE_ELEMENT_FRAME
            or self.output_cartesian_frame != OUTPUT_CARTESIAN_FRAME
            or self.output_coordinate_system != OUTPUT_COORDINATE_SYSTEM
            or self.epoch_scale != EPOCH_SCALE
            or self.epoch_unit != EPOCH_UNIT
            or self.length_unit != LENGTH_UNIT
            or self.time_unit != TIME_UNIT
            or self.velocity_unit != VELOCITY_UNIT
            or self.gm_unit != GM_UNIT
            or self.binary64_semantics != BINARY64_SEMANTICS
            or self.kepler_solver_id != KEPLER_SOLVER_ID
            or self.converter_implementation_sha256
            != CONVERTER_IMPLEMENTATION_SHA256
            or self.status != GENESIS_STATUS
            or self.claim_ceiling != SCIENCE_INPUT_CLAIM_CEILING
            or self.nonclaims != GENESIS_NONCLAIMS
        ):
            raise InitialConditionContractError(
                "genesis semantics, units, implementation, or claims changed"
            )
        if (
            type(self.kepler_maximum_iterations) is not int
            or not 8 <= self.kepler_maximum_iterations <= 256
        ):
            raise InitialConditionContractError(
                "kepler_maximum_iterations must be an exact integer in [8, 256]"
            )
        for value, label in (
            (self.kepler_residual_tolerance_rad, "Kepler residual tolerance"),
            (
                self.minimum_nonsingular_eccentricity,
                "minimum nonsingular eccentricity",
            ),
            (
                self.minimum_nonsingular_sin_inclination,
                "minimum nonsingular sin(inclination)",
            ),
            (
                self.roundtrip_semimajor_axis_relative_tolerance,
                "round-trip semimajor-axis tolerance",
            ),
            (
                self.roundtrip_scalar_absolute_tolerance,
                "round-trip scalar tolerance",
            ),
            (
                self.roundtrip_angle_absolute_tolerance_rad,
                "round-trip angle tolerance",
            ),
        ):
            tolerance = _positive_float(value, label)
            if tolerance >= 1.0e-9:
                raise InitialConditionContractError(
                    f"{label} is too loose for precision-first genesis"
                )
        object.__setattr__(
            self,
            "content_sha256",
            _canonical_sha256(
                b"jx.long-horizon.cartesian-genesis-model.v1",
                self._canonical_record(),
            ),
        )

    def _canonical_record(self) -> dict[str, Any]:
        return {
            "schema": GENESIS_MODEL_SCHEMA,
            "model_id": self.model_id,
            "epoch_jd_tdb_hex": _float_hex(self.epoch_jd_tdb),
            "epoch_scale": self.epoch_scale,
            "epoch_unit": self.epoch_unit,
            "source_element_frame": self.source_element_frame,
            "output_cartesian_frame": self.output_cartesian_frame,
            "output_coordinate_system": self.output_coordinate_system,
            "primary_id": self.primary_id,
            "primary_definition": self.primary_definition,
            "primary_gm_AU3_per_year2_hex": _float_hex(
                self.primary_gm_AU3_per_year2
            ),
            "primary_gm_source_id": self.primary_gm_source_id,
            "primary_gm_source_artifact_path": (
                self.primary_gm_source_artifact_path
            ),
            "primary_gm_source_artifact_sha256": (
                self.primary_gm_source_artifact_sha256
            ),
            "source_model_manifest_path": self.source_model_manifest_path,
            "source_model_manifest_sha256": self.source_model_manifest_sha256,
            "element_origin_to_output_origin_policy": (
                self.element_origin_to_output_origin_policy
            ),
            "length_unit": self.length_unit,
            "time_unit": self.time_unit,
            "velocity_unit": self.velocity_unit,
            "gm_unit": self.gm_unit,
            "binary64_semantics": self.binary64_semantics,
            "kepler_solver_id": self.kepler_solver_id,
            "kepler_maximum_iterations": self.kepler_maximum_iterations,
            "kepler_residual_tolerance_rad_hex": _float_hex(
                self.kepler_residual_tolerance_rad
            ),
            "minimum_nonsingular_eccentricity_hex": _float_hex(
                self.minimum_nonsingular_eccentricity
            ),
            "minimum_nonsingular_sin_inclination_hex": _float_hex(
                self.minimum_nonsingular_sin_inclination
            ),
            "roundtrip_semimajor_axis_relative_tolerance_hex": _float_hex(
                self.roundtrip_semimajor_axis_relative_tolerance
            ),
            "roundtrip_scalar_absolute_tolerance_hex": _float_hex(
                self.roundtrip_scalar_absolute_tolerance
            ),
            "roundtrip_angle_absolute_tolerance_rad_hex": _float_hex(
                self.roundtrip_angle_absolute_tolerance_rad
            ),
            "rotation_policy": ROTATION_POLICY,
            "round_trip_policy": ROUND_TRIP_POLICY,
            "converter_implementation_sha256": (
                self.converter_implementation_sha256
            ),
            "status": self.status,
            "claim_ceiling": self.claim_ceiling,
            "nonclaims": list(self.nonclaims),
        }

    @property
    def schema(self) -> str:
        return GENESIS_MODEL_SCHEMA


DE440_SUN_GENESIS_MODEL = GenesisModel(
    model_id="de440-sun-two-body-genesis-v1",
    epoch_jd_tdb=2_461_200.5,
    source_element_frame=SOURCE_ELEMENT_FRAME,
    output_cartesian_frame=OUTPUT_CARTESIAN_FRAME,
    output_coordinate_system=OUTPUT_COORDINATE_SYSTEM,
    primary_id="NAIF-10-SUN",
    primary_definition="STANDALONE_NAIF_BODY_10_SUN",
    primary_gm_AU3_per_year2=DE440_SOLAR_GM_AU3_PER_JULIAN_YEAR2,
    primary_gm_source_id=(
        "NAIF_GM_DE440_TPC_BODY10_GM_DEFINED_AU_JULIAN_YEAR_CONVERSION"
    ),
    primary_gm_source_artifact_path="NAIF/generic_kernels/pck/gm_de440.tpc",
    primary_gm_source_artifact_sha256=DE440_GM_ARTIFACT_SHA256,
    source_model_manifest_path=None,
    source_model_manifest_sha256=None,
    element_origin_to_output_origin_policy=(
        "SOURCE_AND_OUTPUT_PRIMARY_ARE_NAIF_BODY_10_SUN_NO_ORIGIN_TRANSLATION"
    ),
)


SOURCE_ALIGNED_GENESIS_MODEL = GenesisModel(
    model_id="de441-control-primary-two-body-genesis-v1",
    epoch_jd_tdb=2_461_200.5,
    source_element_frame=SOURCE_ELEMENT_FRAME,
    output_cartesian_frame=OUTPUT_CARTESIAN_FRAME,
    output_coordinate_system=OUTPUT_COORDINATE_SYSTEM,
    primary_id="DE441-INNER-SYSTEM-MONOPOLE",
    primary_definition=(
        "FROZEN_DE441_OUTER_CONTROL_ROW_0_EFFECTIVE_INNER_SYSTEM_MONOPOLE"
    ),
    primary_gm_AU3_per_year2=(
        DE441_CONTROL_PRIMARY_GM_AU3_PER_JULIAN_YEAR2
    ),
    primary_gm_source_id=(
        "JX_DE441_CONTROL_STATE_ROW_0_BINARY64_PRIMARY_GM"
    ),
    primary_gm_source_artifact_path=DE441_CONTROL_STATE_PACKAGE_PATH,
    primary_gm_source_artifact_sha256=DE441_CONTROL_STATE_SHA256,
    source_model_manifest_path=(
        DE441_CONTROL_SOURCE_MODEL_MANIFEST_PACKAGE_PATH
    ),
    source_model_manifest_sha256=(
        DE441_CONTROL_SOURCE_MODEL_MANIFEST_SHA256
    ),
    element_origin_to_output_origin_policy=(
        "NO_EPHEMERIS_ORIGIN_TRANSLATION_REINTERPRET_ELEMENTS_IN_FROZEN_"
        "PRIMARY_FORCE_MODEL_AT_SHARED_EPOCH"
    ),
)


DEFAULT_GENESIS_MODEL = SOURCE_ALIGNED_GENESIS_MODEL


def _elements_sha256(elements: OrbitalElements) -> str:
    return _canonical_sha256(
        b"jx.long-horizon.genesis-source-elements.v1",
        elements._canonical_record(),
    )


def _offsets_record(
    offsets: tuple[tuple[str, float], ...],
) -> list[dict[str, str]]:
    return [
        {"element_name": name, "offset_hex": _float_hex(value)}
        for name, value in offsets
    ]


def _validate_offsets(value: object) -> tuple[tuple[str, float], ...]:
    if type(value) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not str
        or item[0] not in _ELEMENT_NAMES
        or type(item[1]) is not float
        or not math.isfinite(item[1])
        for item in value
    ):
        raise InitialConditionContractError(
            "perturbation_offsets must be exact immutable finite pairs"
        )
    names = tuple(item[0] for item in value)
    if len(set(names)) != len(names):
        raise InitialConditionContractError(
            "perturbation_offsets element names must be unique"
        )
    return value


@dataclass(frozen=True, slots=True)
class CartesianInitialState:
    """One self-validating primary-relative Cartesian initial state."""

    object_id: str
    parent_object_id: str
    source_kind: str
    realization_index: int | None
    source_elements: OrbitalElements
    perturbation_offsets: tuple[tuple[str, float], ...]
    observed_source_sha256: str
    population_id: str
    population_sha256: str
    raw_source_artifact_sha256: str
    ensemble_id: str
    ensemble_sha256: str
    materialized_ensemble_sha256: str
    model: GenesisModel
    position_AU: tuple[float, float, float]
    velocity_AU_per_year: tuple[float, float, float]
    status: str = GENESIS_STATUS
    inference_authorized: bool = False
    claim_ceiling: str = SCIENCE_INPUT_CLAIM_CEILING
    nonclaims: tuple[str, ...] = GENESIS_NONCLAIMS
    source_elements_sha256: str = field(init=False)
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.object_id, "object_id")
        _identifier(self.parent_object_id, "parent_object_id")
        _identifier(self.population_id, "population_id")
        _identifier(self.ensemble_id, "ensemble_id")
        if (
            type(self.source_kind) is not str
            or self.source_kind not in ("NOMINAL", "PERTURBED")
        ):
            raise InitialConditionContractError(
                "source_kind must be NOMINAL or PERTURBED"
            )
        if type(self.source_elements) is not OrbitalElements:
            raise InitialConditionContractError(
                "source_elements must be an exact OrbitalElements"
            )
        if self.source_elements.object_id != self.object_id:
            raise InitialConditionContractError(
                "state object_id must equal source-elements object_id"
            )
        offsets = _validate_offsets(self.perturbation_offsets)
        if self.source_kind == "NOMINAL":
            if (
                self.realization_index is not None
                or self.parent_object_id != self.object_id
                or offsets
                or self.source_elements.observed_source is None
            ):
                raise InitialConditionContractError(
                    "nominal state lineage is incomplete or inconsistent"
                )
            if (
                self.observed_source_sha256
                != self.source_elements.observed_source.content_sha256
            ):
                raise InitialConditionContractError(
                    "nominal observed-source digest is inconsistent"
                )
        else:
            if (
                type(self.realization_index) is not int
                or self.realization_index < 0
                or self.parent_object_id == self.object_id
                or not offsets
                or self.source_elements.observed_source is not None
            ):
                raise InitialConditionContractError(
                    "perturbed state lineage is incomplete or inconsistent"
                )
        for value, label in (
            (self.observed_source_sha256, "observed_source_sha256"),
            (self.population_sha256, "population_sha256"),
            (self.raw_source_artifact_sha256, "raw_source_artifact_sha256"),
            (self.ensemble_sha256, "ensemble_sha256"),
            (
                self.materialized_ensemble_sha256,
                "materialized_ensemble_sha256",
            ),
        ):
            _sha256(value, label)
        if type(self.model) is not GenesisModel:
            raise InitialConditionContractError(
                "model must be an exact GenesisModel"
            )
        position = _vec3(self.position_AU, "position_AU")
        velocity = _vec3(self.velocity_AU_per_year, "velocity_AU_per_year")
        _text_tuple(self.nonclaims, "nonclaims")
        if (
            type(self.status) is not str
            or type(self.claim_ceiling) is not str
            or type(self.nonclaims) is not tuple
            or self.status != GENESIS_STATUS
            or type(self.inference_authorized) is not bool
            or self.inference_authorized
            or self.claim_ceiling != SCIENCE_INPUT_CLAIM_CEILING
            or self.nonclaims != GENESIS_NONCLAIMS
        ):
            raise InitialConditionContractError(
                "initial state attempted to widen its numerical-only claims"
            )

        eccentricity = self.source_elements.eccentricity
        sine_inclination = abs(math.sin(self.source_elements.inclination_rad))
        if (
            eccentricity < self.model.minimum_nonsingular_eccentricity
            or sine_inclination
            < self.model.minimum_nonsingular_sin_inclination
        ):
            raise InitialConditionContractError(
                "source elements are singular under the locked round-trip model"
            )
        generated_position, generated_velocity = _elements_to_components(
            self.source_elements,
            self.model.primary_gm_AU3_per_year2,
            self.model.kepler_maximum_iterations,
            self.model.kepler_residual_tolerance_rad,
        )
        if position != generated_position or velocity != generated_velocity:
            raise InitialConditionContractError(
                "Cartesian components do not exactly match locked genesis"
            )
        recovered = _components_to_element_values(
            position,
            velocity,
            self.model.primary_gm_AU3_per_year2,
        )
        _validate_round_trip(
            self.source_elements,
            recovered,
            self.model.roundtrip_semimajor_axis_relative_tolerance,
            self.model.roundtrip_scalar_absolute_tolerance,
            self.model.roundtrip_angle_absolute_tolerance_rad,
        )
        elements_sha256 = _elements_sha256(self.source_elements)
        object.__setattr__(
            self, "source_elements_sha256", elements_sha256
        )
        object.__setattr__(
            self,
            "content_sha256",
            _canonical_sha256(
                b"jx.long-horizon.cartesian-initial-state.v1",
                self._canonical_record(elements_sha256),
            ),
        )

    @property
    def schema(self) -> str:
        return GENESIS_STATE_SCHEMA

    @property
    def epoch_jd_tdb(self) -> float:
        return self.model.epoch_jd_tdb

    @property
    def epoch_scale(self) -> str:
        return self.model.epoch_scale

    @property
    def epoch_unit(self) -> str:
        return self.model.epoch_unit

    @property
    def frame(self) -> str:
        return self.model.output_cartesian_frame

    @property
    def coordinate_system(self) -> str:
        return self.model.output_coordinate_system

    @property
    def primary_id(self) -> str:
        return self.model.primary_id

    @property
    def primary_gm_AU3_per_year2(self) -> float:
        return self.model.primary_gm_AU3_per_year2

    @property
    def gm_unit(self) -> str:
        return self.model.gm_unit

    @property
    def length_unit(self) -> str:
        return self.model.length_unit

    @property
    def time_unit(self) -> str:
        return self.model.time_unit

    @property
    def velocity_unit(self) -> str:
        return self.model.velocity_unit

    @property
    def model_sha256(self) -> str:
        return self.model.content_sha256

    @property
    def converter_implementation_sha256(self) -> str:
        return self.model.converter_implementation_sha256

    def _canonical_record(self, elements_sha256: str) -> dict[str, Any]:
        return {
            "schema": GENESIS_STATE_SCHEMA,
            "object_id": self.object_id,
            "parent_object_id": self.parent_object_id,
            "source_kind": self.source_kind,
            "realization_index": self.realization_index,
            "source_elements_sha256": elements_sha256,
            "source_elements": self.source_elements._canonical_record(),
            "perturbation_offsets": _offsets_record(
                self.perturbation_offsets
            ),
            "observed_source_sha256": self.observed_source_sha256,
            "population_id": self.population_id,
            "population_sha256": self.population_sha256,
            "raw_source_artifact_sha256": (
                self.raw_source_artifact_sha256
            ),
            "ensemble_id": self.ensemble_id,
            "ensemble_sha256": self.ensemble_sha256,
            "materialized_ensemble_sha256": (
                self.materialized_ensemble_sha256
            ),
            "model_sha256": self.model.content_sha256,
            "converter_implementation_sha256": (
                self.model.converter_implementation_sha256
            ),
            "epoch_jd_tdb_hex": _float_hex(self.model.epoch_jd_tdb),
            "epoch_scale": self.model.epoch_scale,
            "epoch_unit": self.model.epoch_unit,
            "frame": self.model.output_cartesian_frame,
            "coordinate_system": self.model.output_coordinate_system,
            "primary_id": self.model.primary_id,
            "primary_gm_AU3_per_year2_hex": _float_hex(
                self.model.primary_gm_AU3_per_year2
            ),
            "gm_unit": self.model.gm_unit,
            "length_unit": self.model.length_unit,
            "time_unit": self.model.time_unit,
            "velocity_unit": self.model.velocity_unit,
            "position_AU_hex": [_float_hex(value) for value in self.position_AU],
            "velocity_AU_per_year_hex": [
                _float_hex(value) for value in self.velocity_AU_per_year
            ],
            "status": self.status,
            "inference_authorized": self.inference_authorized,
            "claim_ceiling": self.claim_ceiling,
            "nonclaims": list(self.nonclaims),
        }


def _state_from_nominal(
    elements: OrbitalElements,
    population: ScientificPopulationBlock,
    ensemble_spec: SeededPerturbationEnsembleSpec,
    materialized: MaterializedPerturbationEnsemble,
    model: GenesisModel,
) -> CartesianInitialState:
    if elements.observed_source is None:
        raise InitialConditionContractError(
            "nominal genesis requires observed-source provenance"
        )
    position, velocity = _elements_to_components(
        elements,
        model.primary_gm_AU3_per_year2,
        model.kepler_maximum_iterations,
        model.kepler_residual_tolerance_rad,
    )
    return CartesianInitialState(
        object_id=elements.object_id,
        parent_object_id=elements.object_id,
        source_kind="NOMINAL",
        realization_index=None,
        source_elements=elements,
        perturbation_offsets=(),
        observed_source_sha256=elements.observed_source.content_sha256,
        population_id=population.population_id,
        population_sha256=population.content_sha256,
        raw_source_artifact_sha256=population.source_artifact_sha256,
        ensemble_id=ensemble_spec.ensemble_id,
        ensemble_sha256=ensemble_spec.content_sha256,
        materialized_ensemble_sha256=materialized.content_sha256,
        model=model,
        position_AU=position,
        velocity_AU_per_year=velocity,
    )


def _state_from_perturbed(
    member: PerturbedOrbit,
    parent: OrbitalElements,
    population: ScientificPopulationBlock,
    ensemble_spec: SeededPerturbationEnsembleSpec,
    materialized: MaterializedPerturbationEnsemble,
    model: GenesisModel,
) -> CartesianInitialState:
    if parent.observed_source is None:
        raise InitialConditionContractError(
            "perturbed genesis parent lacks observed-source provenance"
        )
    position, velocity = _elements_to_components(
        member.elements,
        model.primary_gm_AU3_per_year2,
        model.kepler_maximum_iterations,
        model.kepler_residual_tolerance_rad,
    )
    return CartesianInitialState(
        object_id=member.elements.object_id,
        parent_object_id=member.parent_object_id,
        source_kind="PERTURBED",
        realization_index=member.realization_index,
        source_elements=member.elements,
        perturbation_offsets=member.offsets,
        observed_source_sha256=parent.observed_source.content_sha256,
        population_id=population.population_id,
        population_sha256=population.content_sha256,
        raw_source_artifact_sha256=population.source_artifact_sha256,
        ensemble_id=ensemble_spec.ensemble_id,
        ensemble_sha256=ensemble_spec.content_sha256,
        materialized_ensemble_sha256=materialized.content_sha256,
        model=model,
        position_AU=position,
        velocity_AU_per_year=velocity,
    )


@dataclass(frozen=True, slots=True)
class InitialConditionBundle:
    """A fully bound nominal-plus-perturbed population genesis bundle."""

    population: ScientificPopulationBlock
    ensemble_spec: SeededPerturbationEnsembleSpec
    materialized_ensemble: MaterializedPerturbationEnsemble
    model: GenesisModel
    states: tuple[CartesianInitialState, ...]
    order_policy: str = STATE_ORDER_POLICY
    status: str = GENESIS_STATUS
    inference_authorized: bool = False
    claim_ceiling: str = SCIENCE_INPUT_CLAIM_CEILING
    nonclaims: tuple[str, ...] = GENESIS_NONCLAIMS
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.population) is not ScientificPopulationBlock:
            raise InitialConditionContractError(
                "population must be an exact ScientificPopulationBlock"
            )
        if type(self.ensemble_spec) is not SeededPerturbationEnsembleSpec:
            raise InitialConditionContractError(
                "ensemble_spec must be an exact SeededPerturbationEnsembleSpec"
            )
        if type(self.materialized_ensemble) is not MaterializedPerturbationEnsemble:
            raise InitialConditionContractError(
                "materialized_ensemble must be exact"
            )
        if type(self.model) is not GenesisModel:
            raise InitialConditionContractError("model must be exact")
        if (
            self.population.epoch_jd_tdb != self.model.epoch_jd_tdb
            or self.population.frame != self.model.source_element_frame
        ):
            raise InitialConditionContractError(
                "population epoch or element frame differs from genesis model"
            )
        if (
            self.ensemble_spec.population_id != self.population.population_id
            or self.ensemble_spec.population_sha256
            != self.population.content_sha256
            or self.materialized_ensemble.ensemble_id
            != self.ensemble_spec.ensemble_id
            or self.materialized_ensemble.ensemble_sha256
            != self.ensemble_spec.content_sha256
            or self.materialized_ensemble.population_sha256
            != self.population.content_sha256
        ):
            raise InitialConditionContractError(
                "population, ensemble spec, and materialized ensemble are unbound"
            )
        if type(self.states) is not tuple or any(
            type(state) is not CartesianInitialState for state in self.states
        ):
            raise InitialConditionContractError(
                "states must be an exact CartesianInitialState tuple"
            )
        expected_count = len(self.population.members) + len(
            self.materialized_ensemble.members
        )
        if len(self.states) != expected_count:
            raise InitialConditionContractError(
                "bundle state count does not cover nominal plus perturbed inputs"
            )
        expected_descriptors: list[
            tuple[str, str, int | None, OrbitalElements, tuple[tuple[str, float], ...]]
        ] = [
            (member.object_id, member.object_id, None, member, ())
            for member in self.population.members
        ]
        expected_descriptors.extend(
            (
                member.elements.object_id,
                member.parent_object_id,
                member.realization_index,
                member.elements,
                member.offsets,
            )
            for member in self.materialized_ensemble.members
        )
        parent_by_id = {
            member.object_id: member for member in self.population.members
        }
        for index, (state, descriptor) in enumerate(
            zip(self.states, expected_descriptors, strict=True)
        ):
            object_id, parent_id, realization_index, elements, offsets = descriptor
            expected_kind = "NOMINAL" if realization_index is None else "PERTURBED"
            parent = parent_by_id[parent_id]
            if parent.observed_source is None:
                raise InitialConditionContractError(
                    "bundle parent lacks observed-source provenance"
                )
            if (
                state.object_id != object_id
                or state.parent_object_id != parent_id
                or state.source_kind != expected_kind
                or state.realization_index != realization_index
                or state.source_elements != elements
                or state.perturbation_offsets != offsets
                or state.observed_source_sha256
                != parent.observed_source.content_sha256
                or state.population_id != self.population.population_id
                or state.population_sha256 != self.population.content_sha256
                or state.raw_source_artifact_sha256
                != self.population.source_artifact_sha256
                or state.ensemble_id != self.ensemble_spec.ensemble_id
                or state.ensemble_sha256 != self.ensemble_spec.content_sha256
                or state.materialized_ensemble_sha256
                != self.materialized_ensemble.content_sha256
                or state.model != self.model
            ):
                raise InitialConditionContractError(
                    f"bundle state {index} is not bound to its exact source lineage"
                )
        state_ids = tuple(state.object_id for state in self.states)
        if len(set(state_ids)) != len(state_ids):
            raise InitialConditionContractError("bundle state IDs are not unique")
        _text_tuple(self.nonclaims, "nonclaims")
        if (
            type(self.order_policy) is not str
            or type(self.status) is not str
            or type(self.claim_ceiling) is not str
            or type(self.nonclaims) is not tuple
            or self.order_policy != STATE_ORDER_POLICY
            or self.status != GENESIS_STATUS
            or type(self.inference_authorized) is not bool
            or self.inference_authorized
            or self.claim_ceiling != SCIENCE_INPUT_CLAIM_CEILING
            or self.nonclaims != GENESIS_NONCLAIMS
        ):
            raise InitialConditionContractError(
                "bundle ordering or numerical-only claims changed"
            )
        object.__setattr__(
            self,
            "content_sha256",
            _canonical_sha256(
                b"jx.long-horizon.initial-condition-bundle.v1",
                self._canonical_record(),
            ),
        )

    @property
    def schema(self) -> str:
        return GENESIS_BUNDLE_SCHEMA

    @property
    def population_id(self) -> str:
        return self.population.population_id

    @property
    def population_sha256(self) -> str:
        return self.population.content_sha256

    @property
    def raw_source_artifact_sha256(self) -> str:
        return self.population.source_artifact_sha256

    @property
    def ensemble_id(self) -> str:
        return self.ensemble_spec.ensemble_id

    @property
    def ensemble_sha256(self) -> str:
        return self.ensemble_spec.content_sha256

    @property
    def materialized_ensemble_sha256(self) -> str:
        return self.materialized_ensemble.content_sha256

    @property
    def state_count(self) -> int:
        return len(self.states)

    @property
    def model_sha256(self) -> str:
        return self.model.content_sha256

    @property
    def converter_implementation_sha256(self) -> str:
        return self.model.converter_implementation_sha256

    def _canonical_record(self) -> dict[str, Any]:
        return {
            "schema": GENESIS_BUNDLE_SCHEMA,
            "population_id": self.population.population_id,
            "population_sha256": self.population.content_sha256,
            "raw_source_artifact_sha256": (
                self.population.source_artifact_sha256
            ),
            "ensemble_id": self.ensemble_spec.ensemble_id,
            "ensemble_sha256": self.ensemble_spec.content_sha256,
            "materialized_ensemble_sha256": (
                self.materialized_ensemble.content_sha256
            ),
            "model_sha256": self.model.content_sha256,
            "converter_implementation_sha256": (
                self.model.converter_implementation_sha256
            ),
            "state_count": len(self.states),
            "state_sha256": [state.content_sha256 for state in self.states],
            "order_policy": self.order_policy,
            "status": self.status,
            "inference_authorized": self.inference_authorized,
            "claim_ceiling": self.claim_ceiling,
            "nonclaims": list(self.nonclaims),
        }


@dataclass(frozen=True, slots=True)
class ObservedInitialConditionInventory:
    """The exact 108-state predeclared observed benchmark inventory."""

    inventory_id: str
    model: GenesisModel
    bundles: tuple[InitialConditionBundle, ...]
    expected_state_count: int
    status: str = GENESIS_STATUS
    inference_authorized: bool = False
    claim_ceiling: str = SCIENCE_INPUT_CLAIM_CEILING
    nonclaims: tuple[str, ...] = GENESIS_NONCLAIMS
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.inventory_id, "inventory_id")
        if type(self.model) is not GenesisModel:
            raise InitialConditionContractError("inventory model must be exact")
        if type(self.bundles) is not tuple or not self.bundles or any(
            type(bundle) is not InitialConditionBundle for bundle in self.bundles
        ):
            raise InitialConditionContractError(
                "bundles must be a nonempty exact InitialConditionBundle tuple"
            )
        if (
            type(self.expected_state_count) is not int
            or self.expected_state_count <= 0
            or self.expected_state_count != sum(
                bundle.state_count for bundle in self.bundles
            )
        ):
            raise InitialConditionContractError(
                "expected_state_count does not match the complete inventory"
            )
        if any(bundle.model != self.model for bundle in self.bundles):
            raise InitialConditionContractError(
                "inventory bundles use different genesis models"
            )
        population_ids = tuple(bundle.population_id for bundle in self.bundles)
        ensemble_ids = tuple(bundle.ensemble_id for bundle in self.bundles)
        state_ids = tuple(
            state.object_id for bundle in self.bundles for state in bundle.states
        )
        if (
            len(set(population_ids)) != len(population_ids)
            or len(set(ensemble_ids)) != len(ensemble_ids)
            or len(set(state_ids)) != len(state_ids)
        ):
            raise InitialConditionContractError(
                "inventory population, ensemble, or state IDs are not unique"
            )
        _text_tuple(self.nonclaims, "nonclaims")
        if (
            type(self.status) is not str
            or type(self.claim_ceiling) is not str
            or type(self.nonclaims) is not tuple
            or self.status != GENESIS_STATUS
            or type(self.inference_authorized) is not bool
            or self.inference_authorized
            or self.claim_ceiling != SCIENCE_INPUT_CLAIM_CEILING
            or self.nonclaims != GENESIS_NONCLAIMS
        ):
            raise InitialConditionContractError(
                "inventory attempted to widen its numerical-only claims"
            )
        object.__setattr__(
            self,
            "content_sha256",
            _canonical_sha256(
                b"jx.long-horizon.initial-condition-inventory.v1",
                {
                    "schema": GENESIS_INVENTORY_SCHEMA,
                    "inventory_id": self.inventory_id,
                    "model_sha256": self.model.content_sha256,
                    "converter_implementation_sha256": (
                        self.model.converter_implementation_sha256
                    ),
                    "bundle_sha256": [
                        bundle.content_sha256 for bundle in self.bundles
                    ],
                    "expected_state_count": self.expected_state_count,
                    "status": self.status,
                    "inference_authorized": self.inference_authorized,
                    "claim_ceiling": self.claim_ceiling,
                    "nonclaims": list(self.nonclaims),
                },
            ),
        )

    @property
    def schema(self) -> str:
        return GENESIS_INVENTORY_SCHEMA

    @property
    def state_count(self) -> int:
        return sum(bundle.state_count for bundle in self.bundles)

    @property
    def states(self) -> tuple[CartesianInitialState, ...]:
        return tuple(state for bundle in self.bundles for state in bundle.states)


def orbital_elements_to_primary_relative_cartesian(
    elements: OrbitalElements,
    *,
    model: GenesisModel = DEFAULT_GENESIS_MODEL,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Convert one exact nonsingular elliptic element record."""

    if type(elements) is not OrbitalElements:
        raise InitialConditionContractError(
            "elements must be an exact OrbitalElements"
        )
    if type(model) is not GenesisModel:
        raise InitialConditionContractError("model must be an exact GenesisModel")
    if (
        elements.eccentricity < model.minimum_nonsingular_eccentricity
        or abs(math.sin(elements.inclination_rad))
        < model.minimum_nonsingular_sin_inclination
    ):
        raise InitialConditionContractError(
            "source elements are singular under the locked conversion model"
        )
    position, velocity = _elements_to_components(
        elements,
        model.primary_gm_AU3_per_year2,
        model.kepler_maximum_iterations,
        model.kepler_residual_tolerance_rad,
    )
    recovered = _components_to_element_values(
        position, velocity, model.primary_gm_AU3_per_year2
    )
    _validate_round_trip(
        elements,
        recovered,
        model.roundtrip_semimajor_axis_relative_tolerance,
        model.roundtrip_scalar_absolute_tolerance,
        model.roundtrip_angle_absolute_tolerance_rad,
    )
    return position, velocity


def primary_relative_cartesian_to_orbital_elements(
    object_id: str,
    position_AU: tuple[float, float, float],
    velocity_AU_per_year: tuple[float, float, float],
    *,
    model: GenesisModel = DEFAULT_GENESIS_MODEL,
) -> OrbitalElements:
    """Independently recover nonsingular elliptic elements for validation."""

    _identifier(object_id, "object_id")
    position = _vec3(position_AU, "position_AU")
    velocity = _vec3(velocity_AU_per_year, "velocity_AU_per_year")
    if type(model) is not GenesisModel:
        raise InitialConditionContractError("model must be an exact GenesisModel")
    values = _components_to_element_values(
        position, velocity, model.primary_gm_AU3_per_year2
    )
    return OrbitalElements(object_id, *values)


def build_initial_condition_bundle(
    population: ScientificPopulationBlock,
    ensemble_spec: SeededPerturbationEnsembleSpec,
    *,
    model: GenesisModel = DEFAULT_GENESIS_MODEL,
) -> InitialConditionBundle:
    """Build one complete nominal-plus-sensitivity Cartesian bundle."""

    if type(population) is not ScientificPopulationBlock:
        raise InitialConditionContractError(
            "population must be an exact ScientificPopulationBlock"
        )
    if type(ensemble_spec) is not SeededPerturbationEnsembleSpec:
        raise InitialConditionContractError(
            "ensemble_spec must be exact"
        )
    if type(model) is not GenesisModel:
        raise InitialConditionContractError("model must be exact")
    if (
        population.epoch_jd_tdb != model.epoch_jd_tdb
        or population.frame != model.source_element_frame
    ):
        raise InitialConditionContractError(
            "population epoch/frame is outside the locked genesis model"
        )
    try:
        materialized = materialize_perturbation_ensemble(
            ensemble_spec, population
        )
    except ScienceInputContractError as exc:
        raise InitialConditionContractError(
            "perturbation ensemble failed its source-content binding"
        ) from exc
    parent_by_id = {member.object_id: member for member in population.members}
    states = tuple(
        _state_from_nominal(
            member, population, ensemble_spec, materialized, model
        )
        for member in population.members
    ) + tuple(
        _state_from_perturbed(
            member,
            parent_by_id[member.parent_object_id],
            population,
            ensemble_spec,
            materialized,
            model,
        )
        for member in materialized.members
    )
    return InitialConditionBundle(
        population=population,
        ensemble_spec=ensemble_spec,
        materialized_ensemble=materialized,
        model=model,
        states=states,
    )


def build_predeclared_initial_condition_inventory(
    *, model: GenesisModel = DEFAULT_GENESIS_MODEL
) -> ObservedInitialConditionInventory:
    """Build the locked two-population inventory and require exactly 108 states."""

    if type(model) is not GenesisModel:
        raise InitialConditionContractError("model must be exact")
    bundles = (
        build_initial_condition_bundle(
            DETACHED_TNO_SCREEN_V1,
            DETACHED_TNO_SENSITIVITY_V1,
            model=model,
        ),
        build_initial_condition_bundle(
            SCATTERING_DISK_CONTROL_V1,
            SCATTERING_DISK_SENSITIVITY_V1,
            model=model,
        ),
    )
    state_count = sum(bundle.state_count for bundle in bundles)
    if state_count != 108:
        raise InitialConditionContractError(
            f"predeclared observed inventory must contain 108 states, got {state_count}"
        )
    return ObservedInitialConditionInventory(
        inventory_id="jpl-observed-two-population-cartesian-genesis-v1",
        model=model,
        bundles=bundles,
        expected_state_count=108,
    )


__all__ = [
    "BINARY64_SEMANTICS",
    "CONVERTER_IMPLEMENTATION_SHA256",
    "CartesianInitialState",
    "DE440_GM_ARTIFACT_SHA256",
    "DE440_SOLAR_GM_AU3_PER_JULIAN_YEAR2",
    "DE440_SUN_GENESIS_MODEL",
    "DE441_CONTROL_PRIMARY_GM_AU3_PER_JULIAN_YEAR2",
    "DE441_CONTROL_SOURCE_MODEL_MANIFEST_PACKAGE_PATH",
    "DE441_CONTROL_SOURCE_MODEL_MANIFEST_SHA256",
    "DE441_CONTROL_STATE_PACKAGE_PATH",
    "DE441_CONTROL_STATE_SHA256",
    "DEFAULT_GENESIS_MODEL",
    "GENESIS_BUNDLE_SCHEMA",
    "GENESIS_INVENTORY_SCHEMA",
    "GENESIS_MODEL_SCHEMA",
    "GENESIS_NONCLAIMS",
    "GENESIS_STATE_SCHEMA",
    "GENESIS_STATUS",
    "GenesisModel",
    "InitialConditionBundle",
    "InitialConditionContractError",
    "ObservedInitialConditionInventory",
    "SOURCE_ALIGNED_GENESIS_MODEL",
    "build_initial_condition_bundle",
    "build_predeclared_initial_condition_inventory",
    "orbital_elements_to_primary_relative_cartesian",
    "primary_relative_cartesian_to_orbital_elements",
]
