"""Sealed DE440 resolved-Earth/Moon Newtonian preparation contracts.

This unpublished boundary translates one exact M4C2B geometric-state replay
into a reduced eleven-body Newtonian model.  It explicitly converts units and
recenters the selected state at an operational-GM-weighted model barycenter;
it does not claim that this model origin is NAIF body 0, that the engine frame
labels are a registry-authorized alias, or that the reduced Newtonian model is
DE440 itself.  Relativity, nonspherical gravity, tides, radiation, mass loss,
collisions, parameter uncertainty propagation, and all omitted bodies remain
outside this profile.  In particular, this resolved Earth--Moon state is not
Wisdom-Holman or ordered-Jacobi domain qualification; that requires a separate
system-barycenter trajectory profile.

The records are forgeable unauthenticated content-integrity evidence.  They do
not establish source authenticity, license or redistribution authority,
physical accuracy, trajectory qualification, continuous array custody, or
independent-process replay.  The engine epoch is canonical positive zero in a
local TDB-second coordinate whose affine origin is retained separately.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, fields
from decimal import Decimal
from fractions import Fraction
from math import gcd
from typing import ClassVar

from .artifacts import LocalArtifactVerificationReceipt
from .contracts import (
    ArtifactBinding,
    BodyParameter,
    CoordinateEpoch,
    ExactUnitScale,
    FrameRealization,
    PhysicalConstant,
    SolarSystemContractError,
    SolarSystemDataError,
    UnitSystemDefinition,
    validate_integrity,
)
from .cspice_execution_contracts import (
    CspiceSpkgeoExecutionLane,
    CspiceSpkgeoExecutionReceipt,
    validate_cspice_spkgeo_execution_receipt,
)
from .serialization import (
    MAXIMUM_CANONICAL_EXACT_INTEGER_BITS,
    domain_sha256,
    validate_sha256,
)
from .units import (
    Binary64UnitConversionReceipt,
    KILOGRAM,
    KILOMETRE,
    METRE,
    SECOND,
    convert_fraction_to_binary64,
)


_CONTENT_INTEGRITY_CLASS = "UNAUTHENTICATED_CONTENT_INTEGRITY_ONLY"
_EVIDENCE_CLASS = "MODEL_OUTPUT"
_MAXIMUM_TEXT_CODEPOINTS = 256
_MAXIMUM_COMPONENTS = 33
_PAIR_COUNT = 3

_GM_PROVIDER_ID = "provider.naif.gm-de440-parameter-source.v1"
_GM_ARTIFACT_ID = "artifact.naif.gm-de440.constants"
_RULES_ARTIFACT_ID = "artifact.naif.gm-de440.rules-license"
_RULES_SPDX = "LicenseRef-NAIF-SPICE-Rules"
_GM_LENGTH = 12_406
_GM_SHA256 = "924ddf4fb9ead9fe8a1aa55780bcabde40b09d00065d58226e24b68d8092f140"
_RULES_LENGTH = 23_487
_RULES_SHA256 = "ae85f851646e7c4f0a762db852907bc090a2ab50c815eb2a6cd8639e96b7e047"
_GM_ARTIFACT_SEAL = "554b17547700e75e4c2907c726037f9aac09428adc3d21ac10118c1ce264b546"
_RULES_ARTIFACT_SEAL = "8dfac4b97cc999b8a4c30146115b3abf52f474f4dd62c85120961cf9bf3b39d7"

_BODY_ROWS = (
    (10, "SUN", "MASSIVE_BODY_CENTER", "BODY10_GM", "1.3271244004127942E+11", "132712440041.27942", "132712440041279420000", "0x1.cc705344ebf26p+66"),
    (199, "MERCURY", "MASSIVE_BODY_CENTER", "BODY199_GM", "2.2031868551400003D+04", "22031.868551400003", "22031868551400.003", "0x1.409b1b2e0e801p+44"),
    (299, "VENUS", "MASSIVE_BODY_CENTER", "BODY299_GM", "3.2485859200000000D+05", "324858.592", "324858592000000", "0x1.277508fa78000p+48"),
    (399, "EARTH", "MASSIVE_BODY_CENTER", "BODY399_GM", "3.9860043550702266D+05", "398600.43550702266", "398600435507022.66", "0x1.6a86655d9f4ebp+48"),
    (301, "MOON", "MASSIVE_BODY_CENTER", "BODY301_GM", "4.9028001184575496D+03", "4902.8001184575496", "4902800118457.5496", "0x1.1d616a34ae633p+42"),
    (4, "MARS_BARYCENTER", "PLANETARY_SYSTEM_BARYCENTER", "BODY4_GM", "4.2828375815756102E+04", "42828.375815756102", "42828375815756.102", "0x1.379e1103b260dp+45"),
    (5, "JUPITER_BARYCENTER", "PLANETARY_SYSTEM_BARYCENTER", "BODY5_GM", "1.2671276409999998E+08", "126712764.09999998", "126712764099999980", "0x1.c22c966cb038fp+56"),
    (6, "SATURN_BARYCENTER", "PLANETARY_SYSTEM_BARYCENTER", "BODY6_GM", "3.7940584841799997E+07", "37940584.841799997", "37940584841799997", "0x1.0d95859421d28p+55"),
    (7, "URANUS_BARYCENTER", "PLANETARY_SYSTEM_BARYCENTER", "BODY7_GM", "5.7945563999999985E+06", "5794556.3999999985", "5794556399999998.5", "0x1.4961e4bda5bfep+52"),
    (8, "NEPTUNE_BARYCENTER", "PLANETARY_SYSTEM_BARYCENTER", "BODY8_GM", "6.8365271005803989D+06", "6836527.1005803989", "6836527100580398.9", "0x1.849c90153d22fp+52"),
    (9, "PLUTO_BARYCENTER", "PLANETARY_SYSTEM_BARYCENTER", "BODY9_GM", "9.7550000000000000D+02", "975.5", "975500000000", "0x1.c640a97600000p+39"),
)
_BODY_NAIF_IDS = tuple(row[0] for row in _BODY_ROWS)
_BODY_IDS = tuple(row[1] for row in _BODY_ROWS)
_BODY_COUNT = len(_BODY_ROWS)

_PROJECTION_POLICY = "EXACT_DE440_TPC_DECIMAL_TIMES_DEFINED_1E9_RN_EVEN_BINARY64"
_ROSTER_POLICY = "EXACT_RESOLVED_EARTH_MOON_11_BODY_NO_DUPLICATE_SYSTEM_PROXY"
_ORIGIN_POLICY = (
    "OPERATIONAL_GM_WEIGHTED_SELECTED_MODEL_BARYCENTER_"
    "NOT_DE440_SOLAR_SYSTEM_BARYCENTER"
)
_FORCE_SCOPE = "MUTUAL_ALL_BODY_DIRECT_UNSOFTENED_NEWTONIAN_POINT_MASS_ONLY"
_INTEGRATOR_SCOPE = (
    "INITIAL_FORCE_INPUT_ONLY_NOT_WISDOM_HOLMAN_OR_ORDERED_JACOBI_QUALIFIED_"
    "RESOLVED_EARTH_MOON_REQUIRES_SEPARATE_TRAJECTORY_PROFILE"
)
_MASS_POLICY = "POSITIVE_ZERO_PLACEHOLDER_NOT_PROVIDED_NOT_USED_BY_GM_DYNAMICS"
_RADIUS_POLICY = "POSITIVE_ZERO_POINT_MASS_NO_COLLISION_RADIUS"
_ENGINE_ALIAS_POLICY = (
    "MODEL_BARYCENTER_J2000_AXES_TO_ENGINE_BARYCENTRIC_INERTIAL_BARYCENTER_"
    "CARTESIAN_RIGHT_HANDED_LABELS_NO_ROTATION_NONAUTHORIZING_CROSSWALK"
)
_OMITTED_PHYSICS = (
    "COLLISIONS_AND_FINITE_RADII",
    "DE440_NATIVE_EPHEMERIS_FORCE_MODEL",
    "MASS_PARAMETER_UNCERTAINTY_PROPAGATION",
    "NON_GRAVITATIONAL_FORCES",
    "NONSPHERICAL_GRAVITY_AND_TIDES",
    "RELATIVITY",
)
_STATE_CONVERSION_POLICY = (
    "EXACT_BINARY64_DYAD_TIMES_DEFINED_1000_RN_EVEN_SIGNED_ZERO_PRESERVED"
)
_RECENTER_POLICY = (
    "EXACT_DYAD_OPERATIONAL_GM_WEIGHTED_CENTROID_DIRECT_SUBTRACTION_ONE_RN_EVEN"
)
_SELECTED_LANE_POLICY = "PRIMARY_VALUES_AFTER_EXACT_PRIMARY_REPLAY_SEMANTIC_EQUALITY"
_PARAMETER_READ_STATUS = (
    "M4B_POINT_VERIFIED_THEN_SEPARATE_HELD_FD_EXACT_HASH_PARSE_NO_CONTINUOUS_CUSTODY"
)
_PREPARATION_STATUS = "DE440S_INITIAL_STATE_REDUCED_NEWTONIAN_MODEL_PREPARED_UNQUALIFIED"
_PREPARATION_REPLAY_SCOPE = (
    "PRIMARY_AND_REPLAY_VALUES_PREPARED_SEPARATELY_SAME_CALLER_NO_PROCESS_INDEPENDENCE_PROOF"
)
_PREPARATION_REPLAY_STATUS = "PRIMARY_AND_REPLAY_PREPARATION_SEMANTIC_SHA256_EQUAL"
_STATE_STAGE = "RECENTERED_TDB_COMPATIBLE_MODEL_BARYCENTER_INITIAL_STATE"

_EPOCH_MAPPING_POLICY = "SPICE_TDB_ET_EQUALS_EFFECTIVE_ET_PLUS_ENGINE_LOCAL_TDB_SECONDS"
_EPOCH_FUTURE_STATUS = (
    "INITIAL_ORIGIN_EXACT_FUTURE_BINARY64_SUM_REQUIRES_SEPARATE_ROUNDING_RECEIPT"
)

_SOURCE_GM_UNIT = ExactUnitScale(
    unit_id="unit.kilometre3-per-second2.defined-exact.v1",
    dimension="GRAVITATIONAL_PARAMETER",
    si_unit_id="si.metre3-per-second2",
    numerator=1_000_000_000,
    denominator=1,
    definition_classification="DEFINED_EXACT",
    artifact_ids=(),
)
_TARGET_GM_UNIT = ExactUnitScale(
    unit_id="unit.metre3-per-second2.defined-exact.v1",
    dimension="GRAVITATIONAL_PARAMETER",
    si_unit_id="si.metre3-per-second2",
    numerator=1,
    denominator=1,
    definition_classification="DEFINED_EXACT",
    artifact_ids=(),
)
_SOURCE_SPEED_UNIT = ExactUnitScale(
    unit_id="unit.kilometre-per-second.defined-exact.v1",
    dimension="SPEED",
    si_unit_id="si.metre-per-second",
    numerator=1_000,
    denominator=1,
    definition_classification="DEFINED_EXACT",
    artifact_ids=(),
)
_TARGET_SPEED_UNIT = ExactUnitScale(
    unit_id="unit.metre-per-second.defined-exact.v1",
    dimension="SPEED",
    si_unit_id="si.metre-per-second",
    numerator=1,
    denominator=1,
    definition_classification="DEFINED_EXACT",
    artifact_ids=(),
)
_TARGET_UNIT_SYSTEM = UnitSystemDefinition(
    unit_system_id="units.tdb-compatible.metre-second-kilogram.v1",
    length=METRE,
    time=SECOND,
    mass=KILOGRAM,
    gravitational_parameter=_TARGET_GM_UNIT,
    coordinate_time_scale="TDB",
)

_PROJECTION_DOMAIN = "jxplanetx.solar-system.engine-preparation.de440-gm-projection.content-integrity.v1"
_PROJECTION_SCHEMA = "De440GmProjection.v1"
_PROJECTION_NAME = "jxplanetx.solar_system.engine_preparation_contracts.De440GmProjection"
_EPOCH_DOMAIN = "jxplanetx.solar-system.engine-preparation.tdb-engine-epoch-binding.content-integrity.v1"
_EPOCH_SCHEMA = "TdbEngineEpochBinding.v1"
_EPOCH_NAME = "jxplanetx.solar_system.engine_preparation_contracts.TdbEngineEpochBinding"
_SPEC_DOMAIN = "jxplanetx.solar-system.engine-preparation.de440-resolved11-newtonian-spec.content-integrity.v1"
_SPEC_SCHEMA = "De440ResolvedEarthMoonNewtonianSpec.v1"
_SPEC_NAME = "jxplanetx.solar_system.engine_preparation_contracts.De440ResolvedEarthMoonNewtonianSpec"
_PREPARATION_DOMAIN = "jxplanetx.solar-system.engine-preparation.de440-newtonian-preparation-receipt.content-integrity.v1"
_PREPARATION_SCHEMA = "De440NewtonianPreparationReceipt.v1"
_PREPARATION_NAME = "jxplanetx.solar_system.engine_preparation_contracts.De440NewtonianPreparationReceipt"
_PREPARATION_SEMANTIC_DOMAIN = "jxplanetx.solar-system.engine-preparation.de440-newtonian-preparation-semantic.v1"
_PREPARATION_SEMANTIC_SCHEMA = "De440NewtonianPreparationSemantic.v1"
_LANE_PREPARATION_DOMAIN = "jxplanetx.solar-system.engine-preparation.de440-newtonian-lane-semantic.v1"
_LANE_PREPARATION_SCHEMA = "De440NewtonianLaneSemantic.v1"
_STATE_DOMAIN = "jxplanetx.solar-system.engine-preparation.de440-newtonian-initial-state.content-integrity.v1"
_STATE_SCHEMA = "De440NewtonianInitialState.v1"
_STATE_NAME = "jxplanetx.solar_system.engine_preparation_contracts.De440NewtonianInitialState"
_STATE_SEMANTIC_DOMAIN = "jxplanetx.solar-system.engine-preparation.de440-newtonian-initial-state-semantic.v1"
_STATE_SEMANTIC_SCHEMA = "De440NewtonianInitialStateSemantic.v1"


def _text(value: object, label: str, *, maximum: int = _MAXIMUM_TEXT_CODEPOINTS) -> str:
    if type(value) is not str:
        raise SolarSystemContractError(f"{label} must be an exact string")
    if not value or len(value) > maximum:
        raise SolarSystemContractError(f"{label} must be bounded nonempty text")
    if value.strip() != value:
        raise SolarSystemContractError(f"{label} must be bounded nonempty trimmed text")
    if any(ord(character) < 0x20 or ord(character) > 0x7E for character in value):
        raise SolarSystemContractError(f"{label} must use printable ASCII only")
    return value


def _token(value: object, expected: str, label: str) -> str:
    checked = _text(value, label, maximum=max(_MAXIMUM_TEXT_CODEPOINTS, len(expected)))
    if checked != expected:
        raise SolarSystemContractError(f"{label} is not the exact required token")
    return checked


def _integer(value: object, label: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if type(value) is not int or value.bit_length() > MAXIMUM_CANONICAL_EXACT_INTEGER_BITS:
        raise SolarSystemContractError(f"{label} must be an exact bounded integer")
    if minimum is not None and value < minimum:
        raise SolarSystemContractError(f"{label} is below its minimum")
    if maximum is not None and value > maximum:
        raise SolarSystemContractError(f"{label} exceeds its maximum")
    return value


def _float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise SolarSystemContractError(f"{label} must be an exact finite binary64 value")
    return value


def _same_float(left: float, right: float) -> bool:
    return left.hex() == right.hex()


def _sha(value: object, label: str) -> str:
    checked = validate_sha256(value, label)
    if checked == "0" * 64:
        raise SolarSystemContractError(f"{label} cannot be an all-zero placeholder")
    return checked


def _optional_sha(value: object, label: str) -> str:
    if type(value) is not str:
        raise SolarSystemContractError(f"{label} must be an exact string")
    if value:
        _sha(value, label)
    return value


def _record_payload(value: object, name: str, overrides: dict[str, object] | None = None) -> tuple[object, ...]:
    replacements = {} if overrides is None else overrides
    return (
        name,
        tuple(
            (descriptor.name, replacements.get(descriptor.name, getattr(value, descriptor.name)))
            for descriptor in fields(value)
            if descriptor.name != "content_sha256"
        ),
    )


def _child_ref(value: object, expected: type[object], validator: object, name: str, schema: str, label: str) -> tuple[str, str, str]:
    if type(value) is not expected:
        raise SolarSystemContractError(f"{label} has the wrong exact type")
    if not callable(validator):
        raise SolarSystemContractError("internal child validator is not callable")
    validator(value)
    return (name, schema, _sha(getattr(value, "content_sha256"), f"{label}.content_sha256"))


_M1_SCHEMA_BY_TYPE = {
    ArtifactBinding: "jxplanetx.solar-system.artifact-binding.payload.v1",
    BodyParameter: "jxplanetx.solar-system.body-parameter.payload.v1",
    CoordinateEpoch: "jxplanetx.solar-system.coordinate-epoch.payload.v1",
    FrameRealization: "jxplanetx.solar-system.frame-realization.payload.v1",
    PhysicalConstant: "jxplanetx.solar-system.physical-constant.payload.v1",
    UnitSystemDefinition: "jxplanetx.solar-system.unit-system-definition.payload.v1",
}


def _m1_ref(value: object, expected: type[object], label: str) -> tuple[str, str, str]:
    if type(value) is not expected:
        raise SolarSystemContractError(f"{label} has the wrong exact M1 type")
    validate_integrity(value)
    name = f"{type(value).__module__}.{type(value).__qualname__}"
    return (
        name,
        _M1_SCHEMA_BY_TYPE[expected],
        _sha(getattr(value, "content_sha256"), f"{label}.content_sha256"),
    )


def _binary64_ref(value: Binary64UnitConversionReceipt) -> tuple[str, str, str]:
    if type(value) is not Binary64UnitConversionReceipt:
        raise SolarSystemContractError("si_conversion must be an exact Binary64UnitConversionReceipt")
    value.validate_integrity()
    return (
        "jxplanetx.solar_system.units.Binary64UnitConversionReceipt",
        "Binary64UnitConversionReceipt.v1",
        value.content_sha256,
    )


def _local_ref(value: LocalArtifactVerificationReceipt, label: str) -> tuple[str, str, str]:
    if type(value) is not LocalArtifactVerificationReceipt:
        raise SolarSystemContractError(f"{label} must be an exact LocalArtifactVerificationReceipt")
    value.validate_integrity()
    return (
        "jxplanetx.solar_system.artifacts.LocalArtifactVerificationReceipt",
        "LocalArtifactVerificationReceipt.v1",
        value.content_sha256,
    )


def _execution_ref(value: CspiceSpkgeoExecutionReceipt) -> tuple[str, str, str]:
    validate_cspice_spkgeo_execution_receipt(value)
    return (
        "jxplanetx.solar_system.cspice_execution_contracts.CspiceSpkgeoExecutionReceipt",
        "CspiceSpkgeoExecutionReceipt.v1",
        value.content_sha256,
    )


def _lane_semantic_ref(value: CspiceSpkgeoExecutionLane, label: str) -> tuple[str, str, str]:
    if type(value) is not CspiceSpkgeoExecutionLane:
        raise SolarSystemContractError(f"{label} has the wrong exact lane type")
    value.validate_integrity()
    return (
        "jxplanetx.solar_system.cspice_execution_contracts.CspiceSpkgeoExecutionLane",
        "CspiceSpkgeoExecutionLane.semantic-result.v1",
        _sha(value.semantic_content_sha256, f"{label}.semantic_content_sha256"),
    )


def _byte_match_ref(value: LocalArtifactVerificationReceipt, label: str) -> tuple[str, str, str]:
    _local_ref(value, label)
    value.byte_match.validate_integrity()
    return (
        "jxplanetx.solar_system.artifacts.ArtifactByteMatchReceipt",
        "ArtifactByteMatchReceipt.v1",
        _sha(value.byte_match.content_sha256, f"{label}.byte_match.content_sha256"),
    )


def _pair(value: object, label: str) -> tuple[int, int]:
    if type(value) is not tuple or len(value) != 2:
        raise SolarSystemContractError(f"{label} must be an exact numerator/denominator pair")
    numerator = _integer(value[0], f"{label}[0]")
    denominator = _integer(value[1], f"{label}[1]", minimum=1)
    if gcd(numerator, denominator) != 1:
        raise SolarSystemContractError(f"{label} must be reduced")
    return (numerator, denominator)


def _pair3(value: object, label: str) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    if type(value) is not tuple or len(value) != _PAIR_COUNT:
        raise SolarSystemContractError(f"{label} must contain exactly three reduced pairs")
    checked = tuple(_pair(item, f"{label}[{index}]") for index, item in enumerate(value))
    return checked  # type: ignore[return-value]


def _float_tuple(value: object, count: int, label: str) -> tuple[float, ...]:
    if type(value) is not tuple or len(value) != count:
        raise SolarSystemContractError(f"{label} must be an exact {count}-item tuple")
    return tuple(_float(item, f"{label}[{index}]") for index, item in enumerate(value))


def _artifact_binding(role: str) -> ArtifactBinding:
    if role == "CONSTANTS":
        return ArtifactBinding(
            artifact_id=_GM_ARTIFACT_ID,
            artifact_role="CONSTANTS",
            provider_id=_GM_PROVIDER_ID,
            version="NAIF-gm_de440.tpc-2022-12-14",
            logical_locator="constants/gm_de440.tpc",
            locator_kind="LOCAL_REGULAR_FILE",
            byte_length=_GM_LENGTH,
            artifact_sha256=_GM_SHA256,
            media_type="text/plain",
            coverage_status="NOT_APPLICABLE",
            coverage=None,
            license_evidence_status="RETAINED_HASH_BOUND_LICENSE_ARTIFACT",
            license_spdx=_RULES_SPDX,
            license_artifact_id=_RULES_ARTIFACT_ID,
            redistribution_status="BUNDLED_WITH_RETAINED_LICENSE_EVIDENCE",
            load_order_status="NOT_LOADABLE",
            load_order=None,
            extrapolation_policy="FORBID",
            content_integrity_class=_CONTENT_INTEGRITY_CLASS,
        )
    if role == "LICENSE":
        return ArtifactBinding(
            artifact_id=_RULES_ARTIFACT_ID,
            artifact_role="LICENSE",
            provider_id=_GM_PROVIDER_ID,
            version="NAIF-SPICE-Rules-ae85f851.v1",
            logical_locator="licenses/naif-spice-rules.html",
            locator_kind="LOCAL_REGULAR_FILE",
            byte_length=_RULES_LENGTH,
            artifact_sha256=_RULES_SHA256,
            media_type="text/html",
            coverage_status="NOT_APPLICABLE",
            coverage=None,
            license_evidence_status="RETAINED_LICENSE_TEXT_SELF_EVIDENCE_NONAUTHORIZING",
            license_spdx=_RULES_SPDX,
            license_artifact_id=None,
            redistribution_status="BUNDLED_WITH_RETAINED_LICENSE_EVIDENCE",
            load_order_status="NOT_LOADABLE",
            load_order=None,
            extrapolation_policy="FORBID",
            content_integrity_class=_CONTENT_INTEGRITY_CLASS,
        )
    raise SolarSystemContractError("unknown parameter artifact role")


def _physical_constant(row: tuple[object, ...], *, target: bool) -> PhysicalConstant:
    naif_id, _body_id, _role, _keyword, _raw, km_decimal, si_decimal, expected_hex = row
    decimal_text = si_decimal if target else km_decimal
    unit = _TARGET_GM_UNIT if target else _SOURCE_GM_UNIT
    operational = float.fromhex(expected_hex) if target else float(Decimal(str(km_decimal)))
    return PhysicalConstant(
        constant_id=(
            f"constant.naif.gm-de440.body-{naif_id}.si-operational.v1"
            if target
            else f"constant.naif.gm-de440.body-{naif_id}.kernel-km3-s2.v1"
        ),
        quantity_kind="GRAVITATIONAL_PARAMETER",
        source_decimal=str(decimal_text),
        operational_value=operational,
        unit=unit,
        classification="ESTIMATED",
        uncertainty_status="NOT_PROVIDED",
        uncertainty=None,
        uncertainty_kind=None,
        uncertainty_scale="NOT_APPLICABLE",
        confidence_level=None,
        covariance_group=None,
        coordinate_scale_applicability="TDB_COMPATIBLE",
        coverage_status="NOT_APPLICABLE",
        coverage=None,
        artifact_ids=(_GM_ARTIFACT_ID,),
    )


def _projection_payload(value: "De440GmProjection") -> tuple[object, ...]:
    return _record_payload(
        value,
        _PROJECTION_NAME,
        {
            "source_gravitational_parameter": _m1_ref(value.source_gravitational_parameter, PhysicalConstant, "source_gravitational_parameter"),
            "si_conversion": _binary64_ref(value.si_conversion),
            "target_gravitational_parameter": _m1_ref(value.target_gravitational_parameter, PhysicalConstant, "target_gravitational_parameter"),
        },
    )


@dataclass(frozen=True, slots=True, eq=False)
class De440GmProjection:
    naif_id: int
    kernel_keyword: str
    kernel_value_lexeme: str
    source_gravitational_parameter: PhysicalConstant
    si_conversion: Binary64UnitConversionReceipt
    target_gravitational_parameter: PhysicalConstant
    projection_policy: str
    content_integrity_class: str
    content_sha256: str = ""

    _SCHEMA: ClassVar[str] = _PROJECTION_SCHEMA

    def __post_init__(self) -> None:
        _optional_sha(self.content_sha256, "content_sha256")
        self._validate()
        expected = domain_sha256(_PROJECTION_DOMAIN, _PROJECTION_SCHEMA, _projection_payload(self))
        if not self.content_sha256:
            object.__setattr__(self, "content_sha256", expected)
        elif self.content_sha256 != expected:
            raise SolarSystemContractError("content_sha256 does not bind the GM projection")

    def _validate(self) -> None:
        if type(self) is not De440GmProjection:
            raise SolarSystemContractError("GM projection must have its exact concrete type")
        naif_id = _integer(self.naif_id, "naif_id", minimum=1, maximum=(1 << 31) - 1)
        row = next((item for item in _BODY_ROWS if item[0] == naif_id), None)
        if row is None:
            raise SolarSystemContractError("naif_id is outside the resolved-eleven roster")
        _token(self.kernel_keyword, str(row[3]), "kernel_keyword")
        _token(self.kernel_value_lexeme, str(row[4]), "kernel_value_lexeme")
        expected_source = _physical_constant(row, target=False)
        expected_target = _physical_constant(row, target=True)
        _m1_ref(self.source_gravitational_parameter, PhysicalConstant, "source_gravitational_parameter")
        _m1_ref(self.target_gravitational_parameter, PhysicalConstant, "target_gravitational_parameter")
        if self.source_gravitational_parameter.content_sha256 != expected_source.content_sha256:
            raise SolarSystemContractError("source GM does not match its exact kernel decimal")
        if self.target_gravitational_parameter.content_sha256 != expected_target.content_sha256:
            raise SolarSystemContractError("target GM does not match its exact SI decimal")
        expected_conversion = convert_fraction_to_binary64(
            Fraction(Decimal(str(row[5]))), _SOURCE_GM_UNIT, _TARGET_GM_UNIT
        )
        if _binary64_ref(self.si_conversion)[2] != expected_conversion.content_sha256:
            raise SolarSystemContractError("SI conversion is not the exact M2 projection")
        if not _same_float(self.si_conversion.rounded_value, self.target_gravitational_parameter.operational_value):
            raise SolarSystemContractError("target operational GM differs from the conversion result")
        _token(self.projection_policy, _PROJECTION_POLICY, "projection_policy")
        _token(self.content_integrity_class, _CONTENT_INTEGRITY_CLASS, "content_integrity_class")

    def validate_integrity(self) -> None:
        _sha(self.content_sha256, "content_sha256")
        self._validate()
        if self.content_sha256 != domain_sha256(_PROJECTION_DOMAIN, _PROJECTION_SCHEMA, _projection_payload(self)):
            raise SolarSystemContractError("content_sha256 does not bind the GM projection")


def validate_de440_gm_projection(value: object) -> None:
    if type(value) is not De440GmProjection:
        raise SolarSystemContractError("value must be an exact De440GmProjection")
    value.validate_integrity()


def _epoch_payload(value: "TdbEngineEpochBinding") -> tuple[object, ...]:
    return _record_payload(
        value,
        _EPOCH_NAME,
        {"effective_epoch": _m1_ref(value.effective_epoch, CoordinateEpoch, "effective_epoch")},
    )


@dataclass(frozen=True, slots=True, eq=False)
class TdbEngineEpochBinding:
    binding_id: str
    effective_epoch: CoordinateEpoch
    effective_binary64_et: float
    engine_time_scale: str
    engine_time_unit_id: str
    engine_epoch_at_origin: float
    mapping_policy: str
    future_mapping_status: str
    content_integrity_class: str
    content_sha256: str = ""

    _SCHEMA: ClassVar[str] = _EPOCH_SCHEMA

    def __post_init__(self) -> None:
        _optional_sha(self.content_sha256, "content_sha256")
        self._validate()
        expected = domain_sha256(_EPOCH_DOMAIN, _EPOCH_SCHEMA, _epoch_payload(self))
        if not self.content_sha256:
            object.__setattr__(self, "content_sha256", expected)
        elif self.content_sha256 != expected:
            raise SolarSystemContractError("content_sha256 does not bind the epoch binding")

    def _validate(self) -> None:
        if type(self) is not TdbEngineEpochBinding:
            raise SolarSystemContractError("epoch binding must have its exact concrete type")
        _text(self.binding_id, "binding_id")
        _m1_ref(self.effective_epoch, CoordinateEpoch, "effective_epoch")
        if (
            self.effective_epoch.time_scale != "TDB"
            or self.effective_epoch.representation != "SPICE_TDB_J2000_OFFSET_SECONDS_TWO_PART"
            or self.effective_epoch.coordinate_unit_id != SECOND.unit_id
            or self.effective_epoch.origin_id != "SPICE_J2000_TDB_ORIGIN"
        ):
            raise SolarSystemContractError("effective epoch is outside exact SPICE TDB ET v1")
        effective = _float(self.effective_binary64_et, "effective_binary64_et")
        if effective == 0.0 and math.copysign(1.0, effective) < 0.0:
            raise SolarSystemContractError("effective_binary64_et must use canonical positive zero")
        exact_epoch = Fraction(self.effective_epoch.whole, 1) + Fraction.from_float(self.effective_epoch.fraction)
        if exact_epoch != Fraction.from_float(effective):
            raise SolarSystemContractError("effective binary64 ET does not equal the exact effective epoch")
        _token(self.engine_time_scale, "TDB", "engine_time_scale")
        _token(self.engine_time_unit_id, SECOND.unit_id, "engine_time_unit_id")
        origin = _float(self.engine_epoch_at_origin, "engine_epoch_at_origin")
        if origin != 0.0 or math.copysign(1.0, origin) < 0.0:
            raise SolarSystemContractError("engine epoch origin must be canonical positive zero")
        _token(self.mapping_policy, _EPOCH_MAPPING_POLICY, "mapping_policy")
        _token(self.future_mapping_status, _EPOCH_FUTURE_STATUS, "future_mapping_status")
        _token(self.content_integrity_class, _CONTENT_INTEGRITY_CLASS, "content_integrity_class")

    def validate_integrity(self) -> None:
        _sha(self.content_sha256, "content_sha256")
        self._validate()
        if self.content_sha256 != domain_sha256(_EPOCH_DOMAIN, _EPOCH_SCHEMA, _epoch_payload(self)):
            raise SolarSystemContractError("content_sha256 does not bind the epoch binding")


def validate_tdb_engine_epoch_binding(value: object) -> None:
    if type(value) is not TdbEngineEpochBinding:
        raise SolarSystemContractError("value must be an exact TdbEngineEpochBinding")
    value.validate_integrity()


def _projection_ref(value: De440GmProjection, label: str) -> tuple[str, str, str]:
    return _child_ref(
        value,
        De440GmProjection,
        validate_de440_gm_projection,
        _PROJECTION_NAME,
        _PROJECTION_SCHEMA,
        label,
    )


def _epoch_ref(value: TdbEngineEpochBinding) -> tuple[str, str, str]:
    return _child_ref(
        value,
        TdbEngineEpochBinding,
        validate_tdb_engine_epoch_binding,
        _EPOCH_NAME,
        _EPOCH_SCHEMA,
        "epoch_binding",
    )


def _body_parameter(row: tuple[object, ...], projection: De440GmProjection) -> BodyParameter:
    return BodyParameter(
        body_id=str(row[1]),
        naif_id=int(row[0]),
        body_role=str(row[2]),
        gravitational_parameter_status="PROVIDED_DYNAMICS_PARAMETER",
        gravitational_parameter=projection.target_gravitational_parameter,
        mass_status="NOT_PROVIDED_NOT_USED",
        mass=None,
        radius_status="POINT_MASS_NO_COLLISION_RADIUS",
        radius_kind=None,
        radius=None,
        artifact_ids=(_GM_ARTIFACT_ID,),
    )


def _expected_source_unit_system(value: UnitSystemDefinition) -> None:
    validate_integrity(value)
    if (
        value.unit_system_id != "units.cspice.kilometre-second-kilogram.v1"
        or value.length.content_sha256 != KILOMETRE.content_sha256
        or value.time.content_sha256 != SECOND.content_sha256
        or value.mass.content_sha256 != KILOGRAM.content_sha256
        or value.gravitational_parameter.content_sha256 != _SOURCE_GM_UNIT.content_sha256
        or value.coordinate_time_scale != "TDB"
    ):
        raise SolarSystemContractError("source unit system is outside the exact CSPICE km/s profile")


def _expected_source_frame(value: FrameRealization) -> None:
    validate_integrity(value)
    if (
        value.frame_id != "frame.naif-j2000.ssb.tdb-compatible.v1"
        or value.frame_kind != "TDB_COMPATIBLE_BARYCENTRIC_INERTIAL"
        or value.origin_kind != "SOLAR_SYSTEM_BARYCENTER"
        or value.origin_naif_id != 0
        or value.origin_realization_id != "NAIF_BODY_0_SOLAR_SYSTEM_BARYCENTER"
        or value.axes_realization_id != "NAIF_BUILTIN_INERTIAL_FRAME_1_J2000"
        or value.orientation_model_id != "NAIF_BUILTIN_J2000_STATIC_ORIENTATION"
        or value.orientation_time_dependence != "STATIC"
        or value.coordinate_time_scale != "TDB"
        or value.coverage is None
    ):
        raise SolarSystemContractError("source frame is outside the exact NAIF J2000/SSB profile")


def _target_frame(source: FrameRealization) -> FrameRealization:
    artifact_ids = tuple(sorted(set(source.artifact_ids) | {_GM_ARTIFACT_ID}))
    return FrameRealization(
        frame_id="frame.jxplanetx.de440-resolved11.operational-gm-barycenter.j2000.v1",
        frame_kind="INERTIAL_ORIGIN_CENTERED",
        origin_kind="NEWTONIAN_MODEL_BARYCENTER",
        origin_naif_id=None,
        origin_realization_id=(
            "JXPLANETX_DE440_RESOLVED11_OPERATIONAL_GM_WEIGHTED_MODEL_BARYCENTER"
        ),
        axes_realization_id="NAIF_BUILTIN_INERTIAL_FRAME_1_J2000",
        orientation_model_id="NAIF_BUILTIN_J2000_STATIC_ORIENTATION",
        orientation_time_dependence="STATIC",
        coordinate_time_scale="TDB",
        coverage_status="NOT_RETAINED_BLOCKED",
        coverage=None,
        artifact_ids=artifact_ids,
    )


def _spec_payload(value: "De440ResolvedEarthMoonNewtonianSpec") -> tuple[object, ...]:
    return _record_payload(
        value,
        _SPEC_NAME,
        {
            "constants_artifact": _m1_ref(value.constants_artifact, ArtifactBinding, "constants_artifact"),
            "constants_license_artifact": _m1_ref(value.constants_license_artifact, ArtifactBinding, "constants_license_artifact"),
            "ordered_gm_projections": tuple(
                _projection_ref(item, f"ordered_gm_projections[{index}]")
                for index, item in enumerate(value.ordered_gm_projections)
            ),
            "ordered_body_parameters": tuple(
                _m1_ref(item, BodyParameter, f"ordered_body_parameters[{index}]")
                for index, item in enumerate(value.ordered_body_parameters)
            ),
            "source_frame": _m1_ref(value.source_frame, FrameRealization, "source_frame"),
            "target_frame": _m1_ref(value.target_frame, FrameRealization, "target_frame"),
            "source_unit_system": _m1_ref(value.source_unit_system, UnitSystemDefinition, "source_unit_system"),
            "target_unit_system": _m1_ref(value.target_unit_system, UnitSystemDefinition, "target_unit_system"),
        },
    )


@dataclass(frozen=True, slots=True, eq=False)
class De440ResolvedEarthMoonNewtonianSpec:
    spec_id: str
    constants_artifact: ArtifactBinding
    constants_license_artifact: ArtifactBinding
    ordered_gm_projections: tuple[De440GmProjection, ...]
    ordered_body_parameters: tuple[BodyParameter, ...]
    source_frame: FrameRealization
    target_frame: FrameRealization
    source_unit_system: UnitSystemDefinition
    target_unit_system: UnitSystemDefinition
    roster_policy: str
    origin_policy: str
    force_scope: str
    integrator_scope: str
    mass_policy: str
    radius_policy: str
    engine_alias_policy: str
    omitted_physics: tuple[str, ...]
    evidence_class: str
    registry_authorized: bool
    qualification_authorized: bool
    content_integrity_class: str
    content_sha256: str = ""

    _SCHEMA: ClassVar[str] = _SPEC_SCHEMA

    def __post_init__(self) -> None:
        _optional_sha(self.content_sha256, "content_sha256")
        self._validate()
        expected = domain_sha256(_SPEC_DOMAIN, _SPEC_SCHEMA, _spec_payload(self))
        if not self.content_sha256:
            object.__setattr__(self, "content_sha256", expected)
        elif self.content_sha256 != expected:
            raise SolarSystemContractError("content_sha256 does not bind the model spec")

    def _validate(self) -> None:
        if type(self) is not De440ResolvedEarthMoonNewtonianSpec:
            raise SolarSystemContractError("model spec must have its exact concrete type")
        _token(self.spec_id, "spec.de440-resolved-earth-moon-newtonian-initial-state.v1", "spec_id")
        gm_ref = _m1_ref(self.constants_artifact, ArtifactBinding, "constants_artifact")
        rules_ref = _m1_ref(self.constants_license_artifact, ArtifactBinding, "constants_license_artifact")
        if gm_ref[2] != _GM_ARTIFACT_SEAL or rules_ref[2] != _RULES_ARTIFACT_SEAL:
            raise SolarSystemContractError("parameter artifacts differ from the exact two-member roster")
        if (
            self.constants_artifact.license_artifact_id != self.constants_license_artifact.artifact_id
            or self.constants_artifact.license_spdx != self.constants_license_artifact.license_spdx
            or self.constants_artifact.provider_id != self.constants_license_artifact.provider_id
        ):
            raise SolarSystemContractError("parameter artifact license backedge is inconsistent")
        if type(self.ordered_gm_projections) is not tuple or len(self.ordered_gm_projections) != _BODY_COUNT:
            raise SolarSystemContractError("ordered_gm_projections must have exactly eleven items")
        if type(self.ordered_body_parameters) is not tuple or len(self.ordered_body_parameters) != _BODY_COUNT:
            raise SolarSystemContractError("ordered_body_parameters must have exactly eleven items")
        for index, (row, projection, body) in enumerate(
            zip(_BODY_ROWS, self.ordered_gm_projections, self.ordered_body_parameters)
        ):
            _projection_ref(projection, f"ordered_gm_projections[{index}]")
            if projection.naif_id != row[0]:
                raise SolarSystemContractError("GM projections do not preserve the closed NAIF order")
            expected_body = _body_parameter(row, projection)
            _m1_ref(body, BodyParameter, f"ordered_body_parameters[{index}]")
            if body.content_sha256 != expected_body.content_sha256:
                raise SolarSystemContractError("body parameter differs from its exact resolved-eleven row")
        _expected_source_frame(self.source_frame)
        validate_integrity(self.target_frame)
        expected_target_frame = _target_frame(self.source_frame)
        if self.target_frame.content_sha256 != expected_target_frame.content_sha256:
            raise SolarSystemContractError("target frame is not the exact operational-GM model barycenter")
        _expected_source_unit_system(self.source_unit_system)
        validate_integrity(self.target_unit_system)
        if self.target_unit_system.content_sha256 != _TARGET_UNIT_SYSTEM.content_sha256:
            raise SolarSystemContractError("target units are not the exact TDB-compatible SI system")
        _token(self.roster_policy, _ROSTER_POLICY, "roster_policy")
        _token(self.origin_policy, _ORIGIN_POLICY, "origin_policy")
        _token(self.force_scope, _FORCE_SCOPE, "force_scope")
        _token(self.integrator_scope, _INTEGRATOR_SCOPE, "integrator_scope")
        _token(self.mass_policy, _MASS_POLICY, "mass_policy")
        _token(self.radius_policy, _RADIUS_POLICY, "radius_policy")
        _token(self.engine_alias_policy, _ENGINE_ALIAS_POLICY, "engine_alias_policy")
        if (
            type(self.omitted_physics) is not tuple
            or len(self.omitted_physics) != len(_OMITTED_PHYSICS)
        ):
            raise SolarSystemContractError("omitted_physics must be the exact bounded tuple")
        for index, item in enumerate(self.omitted_physics):
            _text(item, f"omitted_physics[{index}]")
        if self.omitted_physics != _OMITTED_PHYSICS:
            raise SolarSystemContractError("omitted_physics differs from the closed model scope")
        _token(self.evidence_class, _EVIDENCE_CLASS, "evidence_class")
        if type(self.registry_authorized) is not bool or self.registry_authorized:
            raise SolarSystemContractError("registry_authorized must be exact false")
        if type(self.qualification_authorized) is not bool or self.qualification_authorized:
            raise SolarSystemContractError("qualification_authorized must be exact false")
        _token(self.content_integrity_class, _CONTENT_INTEGRITY_CLASS, "content_integrity_class")

    def validate_integrity(self) -> None:
        _sha(self.content_sha256, "content_sha256")
        self._validate()
        if self.content_sha256 != domain_sha256(_SPEC_DOMAIN, _SPEC_SCHEMA, _spec_payload(self)):
            raise SolarSystemContractError("content_sha256 does not bind the model spec")


def validate_de440_resolved_earth_moon_newtonian_spec(value: object) -> None:
    if type(value) is not De440ResolvedEarthMoonNewtonianSpec:
        raise SolarSystemContractError(
            "value must be an exact De440ResolvedEarthMoonNewtonianSpec"
        )
    value.validate_integrity()


def _spec_ref(value: De440ResolvedEarthMoonNewtonianSpec) -> tuple[str, str, str]:
    return _child_ref(
        value,
        De440ResolvedEarthMoonNewtonianSpec,
        validate_de440_resolved_earth_moon_newtonian_spec,
        _SPEC_NAME,
        _SPEC_SCHEMA,
        "model_spec",
    )


def _fraction_pair(value: Fraction) -> tuple[int, int]:
    return (value.numerator, value.denominator)


def _state_payload(positions: tuple[float, ...], velocities: tuple[float, ...]) -> bytes:
    payload = bytearray()
    for index in range(_BODY_COUNT):
        offset = index * 3
        payload.extend(struct.pack(">6d", *positions[offset : offset + 3], *velocities[offset : offset + 3]))
    return bytes(payload)


def _gm_payload(values: tuple[float, ...]) -> bytes:
    return struct.pack(f">{_BODY_COUNT}d", *values)


def _convert_state_component(value: float, *, velocity: bool) -> float:
    if value == 0.0:
        return math.copysign(0.0, value)
    source = _SOURCE_SPEED_UNIT if velocity else KILOMETRE
    target = _TARGET_SPEED_UNIT if velocity else METRE
    return convert_fraction_to_binary64(Fraction.from_float(value), source, target).rounded_value


def _round_target_component(value: Fraction, *, velocity: bool) -> float:
    unit = _TARGET_SPEED_UNIT if velocity else METRE
    return convert_fraction_to_binary64(value, unit, unit).rounded_value


def _lane_preparation_values(
    lane: CspiceSpkgeoExecutionLane,
    spec: De440ResolvedEarthMoonNewtonianSpec,
) -> dict[str, object]:
    if type(lane) is not CspiceSpkgeoExecutionLane:
        raise SolarSystemContractError("source lane has the wrong exact type")
    lane.validate_integrity()
    batch = lane.state_batch
    if batch.target_naif_ids != _BODY_NAIF_IDS or batch.state_shape != (_BODY_COUNT, 3):
        raise SolarSystemContractError("source lane does not contain the exact resolved-eleven state")
    positions_source = _float_tuple(batch.positions, _MAXIMUM_COMPONENTS, "source positions")
    velocities_source = _float_tuple(batch.velocities, _MAXIMUM_COMPONENTS, "source velocities")
    source_payload = _state_payload(positions_source, velocities_source)
    if (
        lane.state_payload_byte_length != len(source_payload)
        or lane.state_payload_sha256 != hashlib.sha256(source_payload).hexdigest()
    ):
        raise SolarSystemContractError("source lane payload does not bind its exact state values")

    converted_positions = tuple(
        _convert_state_component(value, velocity=False) for value in positions_source
    )
    converted_velocities = tuple(
        _convert_state_component(value, velocity=True) for value in velocities_source
    )
    converted_payload = _state_payload(converted_positions, converted_velocities)
    gm_values = tuple(
        projection.target_gravitational_parameter.operational_value
        for projection in spec.ordered_gm_projections
    )
    _float_tuple(gm_values, _BODY_COUNT, "operational GMs")
    if any(value <= 0.0 for value in gm_values):
        raise SolarSystemContractError("every resolved-eleven operational GM must be positive")
    gm_fractions = tuple(Fraction.from_float(value) for value in gm_values)
    total_gm = sum(gm_fractions, Fraction(0, 1))
    if total_gm <= 0:
        raise SolarSystemContractError("total operational GM must be positive")

    centroid_position: list[Fraction] = []
    centroid_velocity: list[Fraction] = []
    for component in range(3):
        centroid_position.append(
            sum(
                gm_fractions[index]
                * Fraction.from_float(converted_positions[index * 3 + component])
                for index in range(_BODY_COUNT)
            )
            / total_gm
        )
        centroid_velocity.append(
            sum(
                gm_fractions[index]
                * Fraction.from_float(converted_velocities[index * 3 + component])
                for index in range(_BODY_COUNT)
            )
            / total_gm
        )

    recentered_positions: list[float] = []
    recentered_velocities: list[float] = []
    for index in range(_BODY_COUNT):
        for component in range(3):
            offset = index * 3 + component
            recentered_positions.append(
                _round_target_component(
                    Fraction.from_float(converted_positions[offset]) - centroid_position[component],
                    velocity=False,
                )
            )
            recentered_velocities.append(
                _round_target_component(
                    Fraction.from_float(converted_velocities[offset]) - centroid_velocity[component],
                    velocity=True,
                )
            )

    normalized_position_residual: list[Fraction] = []
    normalized_velocity_residual: list[Fraction] = []
    for component in range(3):
        normalized_position_residual.append(
            sum(
                gm_fractions[index]
                * Fraction.from_float(recentered_positions[index * 3 + component])
                for index in range(_BODY_COUNT)
            )
            / total_gm
        )
        normalized_velocity_residual.append(
            sum(
                gm_fractions[index]
                * Fraction.from_float(recentered_velocities[index * 3 + component])
                for index in range(_BODY_COUNT)
            )
            / total_gm
        )
    recentered_positions_tuple = tuple(recentered_positions)
    recentered_velocities_tuple = tuple(recentered_velocities)
    recentered_payload = _state_payload(recentered_positions_tuple, recentered_velocities_tuple)
    gm_bytes = _gm_payload(gm_values)
    return {
        "source_payload_byte_length": len(source_payload),
        "source_payload_sha256": hashlib.sha256(source_payload).hexdigest(),
        "converted_payload_sha256": hashlib.sha256(converted_payload).hexdigest(),
        "centroid_position_pairs": tuple(_fraction_pair(value) for value in centroid_position),
        "centroid_velocity_pairs": tuple(_fraction_pair(value) for value in centroid_velocity),
        "position_residual_pairs": tuple(_fraction_pair(value) for value in normalized_position_residual),
        "velocity_residual_pairs": tuple(_fraction_pair(value) for value in normalized_velocity_residual),
        "recentered_payload_sha256": hashlib.sha256(recentered_payload).hexdigest(),
        "gm_payload_sha256": hashlib.sha256(gm_bytes).hexdigest(),
        "positions": recentered_positions_tuple,
        "velocities": recentered_velocities_tuple,
        "gm_values": gm_values,
    }


def _require_source_execution(value: CspiceSpkgeoExecutionReceipt) -> None:
    validate_cspice_spkgeo_execution_receipt(value)
    primary = value.primary_lane
    replay = value.replay_lane
    for label, lane in (("primary", primary), ("replay", replay)):
        query = lane.projection.effective_query
        if (
            query.target_naif_ids != _BODY_NAIF_IDS
            or query.observer_naif_id != 0
            or query.aberration_correction != "NONE"
            or query.state_kind != "GEOMETRIC"
        ):
            raise SolarSystemContractError(f"{label} source query is outside the resolved-eleven geometric profile")
        if lane.provider_profile != "SPICEYPY_8_2_0_N0067_DERIVED_CSPICE_SPKGEO_J2000_TYPE2_KM_PER_S_V1":
            raise SolarSystemContractError(f"{label} source lane provider profile is not exact")
    if primary.projection.content_sha256 != replay.projection.content_sha256:
        raise SolarSystemContractError("source lanes must bind the same exact M4C1 projection")
    # M4C2B's private frozen gate is imported lazily to avoid any provider import.
    from .cspice_execution import _require_exact_provider_roster

    for lane in (primary, replay):
        abi = lane.runtime.python_wheel_interpreter_abi_tag
        _require_exact_provider_roster(lane.projection, abi)


def _lane_preparation_semantic_digest(
    lane: CspiceSpkgeoExecutionLane,
    spec: De440ResolvedEarthMoonNewtonianSpec,
    epoch_binding: TdbEngineEpochBinding,
    gm_artifact_verification: LocalArtifactVerificationReceipt,
    rules_artifact_verification: LocalArtifactVerificationReceipt,
    values: dict[str, object],
) -> str:
    payload = (
        ("source_lane_semantic_ref", _lane_semantic_ref(lane, "source_lane")),
        ("model_spec", _spec_ref(spec)),
        ("epoch_binding", _epoch_ref(epoch_binding)),
        (
            "gm_artifact_byte_match_ref",
            _byte_match_ref(gm_artifact_verification, "gm_artifact_verification"),
        ),
        (
            "rules_artifact_byte_match_ref",
            _byte_match_ref(rules_artifact_verification, "rules_artifact_verification"),
        ),
        ("source_payload_byte_length", values["source_payload_byte_length"]),
        ("source_payload_sha256", values["source_payload_sha256"]),
        ("converted_payload_sha256", values["converted_payload_sha256"]),
        ("centroid_position_pairs", values["centroid_position_pairs"]),
        ("centroid_velocity_pairs", values["centroid_velocity_pairs"]),
        ("position_residual_pairs", values["position_residual_pairs"]),
        ("velocity_residual_pairs", values["velocity_residual_pairs"]),
        ("recentered_payload_sha256", values["recentered_payload_sha256"]),
        ("gm_payload_sha256", values["gm_payload_sha256"]),
        ("state_conversion_policy", _STATE_CONVERSION_POLICY),
        ("recenter_policy", _RECENTER_POLICY),
    )
    return domain_sha256(_LANE_PREPARATION_DOMAIN, _LANE_PREPARATION_SCHEMA, payload)


def _preparation_payload(value: "De440NewtonianPreparationReceipt") -> tuple[object, ...]:
    return _record_payload(
        value,
        _PREPARATION_NAME,
        {
            "source_execution": _execution_ref(value.source_execution),
            "model_spec": _spec_ref(value.model_spec),
            "epoch_binding": _epoch_ref(value.epoch_binding),
            "gm_artifact_verification": _local_ref(value.gm_artifact_verification, "gm_artifact_verification"),
            "gm_license_verification": _local_ref(value.gm_license_verification, "gm_license_verification"),
        },
    )


def _preparation_semantic_payload(value: "De440NewtonianPreparationReceipt") -> tuple[object, ...]:
    return (
        (
            "source_lane_semantic_ref",
            _lane_semantic_ref(value.source_execution.primary_lane, "source_execution.primary_lane"),
        ),
        ("model_spec", _spec_ref(value.model_spec)),
        ("selected_lane_policy", value.selected_lane_policy),
        ("source_state_payload_byte_length", value.source_state_payload_byte_length),
        ("source_state_payload_sha256", value.source_state_payload_sha256),
        ("converted_ssb_payload_sha256", value.converted_ssb_payload_sha256),
        ("state_conversion_policy", value.state_conversion_policy),
        ("centroid_position_exact_pairs", value.centroid_position_exact_pairs),
        ("centroid_velocity_exact_pairs", value.centroid_velocity_exact_pairs),
        ("recenter_policy", value.recenter_policy),
        ("normalized_position_residual_exact_pairs", value.normalized_position_residual_exact_pairs),
        ("normalized_velocity_residual_exact_pairs", value.normalized_velocity_residual_exact_pairs),
        ("recentered_payload_sha256", value.recentered_payload_sha256),
        ("gm_payload_sha256", value.gm_payload_sha256),
        ("epoch_binding", _epoch_ref(value.epoch_binding)),
        (
            "gm_artifact_byte_match_ref",
            _byte_match_ref(value.gm_artifact_verification, "gm_artifact_verification"),
        ),
        (
            "rules_artifact_byte_match_ref",
            _byte_match_ref(value.gm_license_verification, "gm_license_verification"),
        ),
        ("parameter_read_status", value.parameter_read_status),
        ("primary_preparation_semantic_sha256", value.primary_preparation_semantic_sha256),
        ("replay_preparation_semantic_sha256", value.replay_preparation_semantic_sha256),
        ("semantic_replay_scope", value.semantic_replay_scope),
        ("semantic_replay_status", value.semantic_replay_status),
        ("preparation_status", value.preparation_status),
        ("evidence_class", value.evidence_class),
        ("content_integrity_class", value.content_integrity_class),
    )


@dataclass(frozen=True, slots=True, eq=False)
class De440NewtonianPreparationReceipt:
    receipt_id: str
    source_execution: CspiceSpkgeoExecutionReceipt
    model_spec: De440ResolvedEarthMoonNewtonianSpec
    selected_lane_policy: str
    source_state_payload_byte_length: int
    source_state_payload_sha256: str
    converted_ssb_payload_sha256: str
    state_conversion_policy: str
    centroid_position_exact_pairs: tuple[tuple[int, int], ...]
    centroid_velocity_exact_pairs: tuple[tuple[int, int], ...]
    recenter_policy: str
    normalized_position_residual_exact_pairs: tuple[tuple[int, int], ...]
    normalized_velocity_residual_exact_pairs: tuple[tuple[int, int], ...]
    recentered_payload_sha256: str
    gm_payload_sha256: str
    epoch_binding: TdbEngineEpochBinding
    gm_artifact_verification: LocalArtifactVerificationReceipt
    gm_license_verification: LocalArtifactVerificationReceipt
    parameter_read_status: str
    primary_preparation_semantic_sha256: str
    replay_preparation_semantic_sha256: str
    semantic_replay_scope: str
    semantic_replay_status: str
    preparation_status: str
    evidence_class: str
    content_integrity_class: str
    semantic_content_sha256: str = ""
    content_sha256: str = ""

    _SCHEMA: ClassVar[str] = _PREPARATION_SCHEMA

    def __post_init__(self) -> None:
        _optional_sha(self.semantic_content_sha256, "semantic_content_sha256")
        _optional_sha(self.content_sha256, "content_sha256")
        self._validate()
        semantic = domain_sha256(
            _PREPARATION_SEMANTIC_DOMAIN,
            _PREPARATION_SEMANTIC_SCHEMA,
            _preparation_semantic_payload(self),
        )
        if not self.semantic_content_sha256:
            object.__setattr__(self, "semantic_content_sha256", semantic)
        elif self.semantic_content_sha256 != semantic:
            raise SolarSystemContractError("semantic_content_sha256 does not bind the preparation core")
        content = domain_sha256(_PREPARATION_DOMAIN, _PREPARATION_SCHEMA, _preparation_payload(self))
        if not self.content_sha256:
            object.__setattr__(self, "content_sha256", content)
        elif self.content_sha256 != content:
            raise SolarSystemContractError("content_sha256 does not bind the preparation receipt")

    def _validate(self) -> None:
        if type(self) is not De440NewtonianPreparationReceipt:
            raise SolarSystemContractError("preparation receipt must have its exact concrete type")
        _text(self.receipt_id, "receipt_id")
        _require_source_execution(self.source_execution)
        _spec_ref(self.model_spec)
        primary = self.source_execution.primary_lane
        replay = self.source_execution.replay_lane
        query = primary.projection.effective_query
        if (
            self.model_spec.source_frame.content_sha256 != query.frame.content_sha256
            or self.model_spec.source_unit_system.content_sha256 != query.output_unit_system.content_sha256
        ):
            raise SolarSystemContractError("model spec source frame/units differ from the executed query")
        _token(self.selected_lane_policy, _SELECTED_LANE_POLICY, "selected_lane_policy")
        length = _integer(
            self.source_state_payload_byte_length,
            "source_state_payload_byte_length",
            minimum=_BODY_COUNT * 48,
            maximum=_BODY_COUNT * 48,
        )
        for label in (
            "source_state_payload_sha256",
            "converted_ssb_payload_sha256",
            "recentered_payload_sha256",
            "gm_payload_sha256",
            "primary_preparation_semantic_sha256",
            "replay_preparation_semantic_sha256",
        ):
            _sha(getattr(self, label), label)
        _token(self.state_conversion_policy, _STATE_CONVERSION_POLICY, "state_conversion_policy")
        centroid_position = _pair3(self.centroid_position_exact_pairs, "centroid_position_exact_pairs")
        centroid_velocity = _pair3(self.centroid_velocity_exact_pairs, "centroid_velocity_exact_pairs")
        _token(self.recenter_policy, _RECENTER_POLICY, "recenter_policy")
        position_residual = _pair3(
            self.normalized_position_residual_exact_pairs,
            "normalized_position_residual_exact_pairs",
        )
        velocity_residual = _pair3(
            self.normalized_velocity_residual_exact_pairs,
            "normalized_velocity_residual_exact_pairs",
        )
        _epoch_ref(self.epoch_binding)
        projection = primary.projection
        if (
            self.epoch_binding.effective_epoch.content_sha256
            != projection.effective_query.epoch.content_sha256
            or not _same_float(
                self.epoch_binding.effective_binary64_et,
                projection.binary64_projection.rounded_value,
            )
        ):
            raise SolarSystemContractError("epoch binding differs from the exact M4C1 effective ET")
        gm_verification = self.gm_artifact_verification
        rules_verification = self.gm_license_verification
        _local_ref(gm_verification, "gm_artifact_verification")
        _local_ref(rules_verification, "gm_license_verification")
        if (
            gm_verification.byte_match.artifact.content_sha256 != self.model_spec.constants_artifact.content_sha256
            or rules_verification.byte_match.artifact.content_sha256
            != self.model_spec.constants_license_artifact.content_sha256
        ):
            raise SolarSystemContractError("parameter verifications do not bind the model artifacts")
        _token(self.parameter_read_status, _PARAMETER_READ_STATUS, "parameter_read_status")
        primary_values = _lane_preparation_values(primary, self.model_spec)
        replay_values = _lane_preparation_values(replay, self.model_spec)
        for key in (
            "source_payload_byte_length",
            "source_payload_sha256",
            "converted_payload_sha256",
            "centroid_position_pairs",
            "centroid_velocity_pairs",
            "position_residual_pairs",
            "velocity_residual_pairs",
            "recentered_payload_sha256",
            "gm_payload_sha256",
        ):
            if primary_values[key] != replay_values[key]:
                raise SolarSystemContractError("primary and replay prepared values differ")
        for key in ("positions", "velocities", "gm_values"):
            primary_block = primary_values[key]
            replay_block = replay_values[key]
            if (
                type(primary_block) is not tuple
                or type(replay_block) is not tuple
                or len(primary_block) != len(replay_block)
                or any(
                    not _same_float(left, right)
                    for left, right in zip(primary_block, replay_block)
                )
            ):
                raise SolarSystemContractError(
                    "primary and replay prepared binary64 values differ"
                )
        expected_fields = {
            "source_payload_byte_length": length,
            "source_payload_sha256": self.source_state_payload_sha256,
            "converted_payload_sha256": self.converted_ssb_payload_sha256,
            "centroid_position_pairs": centroid_position,
            "centroid_velocity_pairs": centroid_velocity,
            "position_residual_pairs": position_residual,
            "velocity_residual_pairs": velocity_residual,
            "recentered_payload_sha256": self.recentered_payload_sha256,
            "gm_payload_sha256": self.gm_payload_sha256,
        }
        for key, expected in expected_fields.items():
            if primary_values[key] != expected:
                raise SolarSystemContractError(f"{key} differs from exact preparation recomputation")
        expected_primary = _lane_preparation_semantic_digest(
            primary,
            self.model_spec,
            self.epoch_binding,
            gm_verification,
            rules_verification,
            primary_values,
        )
        expected_replay = _lane_preparation_semantic_digest(
            replay,
            self.model_spec,
            self.epoch_binding,
            gm_verification,
            rules_verification,
            replay_values,
        )
        if (
            self.primary_preparation_semantic_sha256 != expected_primary
            or self.replay_preparation_semantic_sha256 != expected_replay
            or expected_primary != expected_replay
        ):
            raise SolarSystemContractError("preparation semantic replay digests do not match")
        _token(self.semantic_replay_scope, _PREPARATION_REPLAY_SCOPE, "semantic_replay_scope")
        _token(self.semantic_replay_status, _PREPARATION_REPLAY_STATUS, "semantic_replay_status")
        _token(self.preparation_status, _PREPARATION_STATUS, "preparation_status")
        _token(self.evidence_class, _EVIDENCE_CLASS, "evidence_class")
        _token(self.content_integrity_class, _CONTENT_INTEGRITY_CLASS, "content_integrity_class")

    def validate_integrity(self) -> None:
        _sha(self.semantic_content_sha256, "semantic_content_sha256")
        _sha(self.content_sha256, "content_sha256")
        self._validate()
        expected_semantic = domain_sha256(
            _PREPARATION_SEMANTIC_DOMAIN,
            _PREPARATION_SEMANTIC_SCHEMA,
            _preparation_semantic_payload(self),
        )
        if self.semantic_content_sha256 != expected_semantic:
            raise SolarSystemContractError("semantic_content_sha256 does not bind the preparation core")
        if self.content_sha256 != domain_sha256(_PREPARATION_DOMAIN, _PREPARATION_SCHEMA, _preparation_payload(self)):
            raise SolarSystemContractError("content_sha256 does not bind the preparation receipt")


def validate_de440_newtonian_preparation_receipt(value: object) -> None:
    if type(value) is not De440NewtonianPreparationReceipt:
        raise SolarSystemContractError("value must be an exact De440NewtonianPreparationReceipt")
    value.validate_integrity()


def _preparation_ref(value: De440NewtonianPreparationReceipt) -> tuple[str, str, str]:
    return _child_ref(
        value,
        De440NewtonianPreparationReceipt,
        validate_de440_newtonian_preparation_receipt,
        _PREPARATION_NAME,
        _PREPARATION_SCHEMA,
        "preparation_receipt",
    )


def _preparation_semantic_ref(
    value: De440NewtonianPreparationReceipt,
) -> tuple[str, str, str]:
    validate_de440_newtonian_preparation_receipt(value)
    return (
        _PREPARATION_NAME,
        _PREPARATION_SEMANTIC_SCHEMA,
        _sha(value.semantic_content_sha256, "preparation_receipt.semantic_content_sha256"),
    )


def _initial_state_payload(value: "De440NewtonianInitialState") -> tuple[object, ...]:
    return _record_payload(
        value,
        _STATE_NAME,
        {"preparation_receipt": _preparation_ref(value.preparation_receipt)},
    )


def _initial_state_semantic_payload(value: "De440NewtonianInitialState") -> tuple[object, ...]:
    return (
        ("preparation_semantic_ref", _preparation_semantic_ref(value.preparation_receipt)),
        ("body_ids", value.body_ids),
        ("naif_ids", value.naif_ids),
        ("component_order", value.component_order),
        ("state_shape", value.state_shape),
        ("positions", value.positions),
        ("velocities", value.velocities),
        ("gravitational_parameters", value.gravitational_parameters),
        ("masses", value.masses),
        ("radii", value.radii),
        ("massive", value.massive),
        ("engine_epoch", value.engine_epoch),
        ("engine_time_scale", value.engine_time_scale),
        ("engine_frame", value.engine_frame),
        ("engine_origin", value.engine_origin),
        ("engine_axes", value.engine_axes),
        ("engine_length_unit", value.engine_length_unit),
        ("engine_time_unit", value.engine_time_unit),
        ("engine_mass_unit", value.engine_mass_unit),
        ("engine_unit_system_id", value.engine_unit_system_id),
        ("state_stage", value.state_stage),
        ("evidence_class", value.evidence_class),
        ("registry_authorized", value.registry_authorized),
        ("qualification_authorized", value.qualification_authorized),
        ("content_integrity_class", value.content_integrity_class),
    )


@dataclass(frozen=True, slots=True, eq=False)
class De440NewtonianInitialState:
    state_id: str
    force_plan_id: str
    preparation_receipt: De440NewtonianPreparationReceipt
    body_ids: tuple[str, ...]
    naif_ids: tuple[int, ...]
    component_order: tuple[str, str, str]
    state_shape: tuple[int, int]
    positions: tuple[float, ...]
    velocities: tuple[float, ...]
    gravitational_parameters: tuple[float, ...]
    masses: tuple[float, ...]
    radii: tuple[float, ...]
    massive: tuple[bool, ...]
    engine_epoch: float
    engine_time_scale: str
    engine_frame: str
    engine_origin: str
    engine_axes: str
    engine_length_unit: str
    engine_time_unit: str
    engine_mass_unit: str
    engine_unit_system_id: str
    state_stage: str
    evidence_class: str
    registry_authorized: bool
    qualification_authorized: bool
    semantic_content_sha256: str = ""
    content_integrity_class: str = _CONTENT_INTEGRITY_CLASS
    content_sha256: str = ""

    _SCHEMA: ClassVar[str] = _STATE_SCHEMA

    def __post_init__(self) -> None:
        _optional_sha(self.semantic_content_sha256, "semantic_content_sha256")
        _optional_sha(self.content_sha256, "content_sha256")
        self._validate()
        semantic = domain_sha256(
            _STATE_SEMANTIC_DOMAIN,
            _STATE_SEMANTIC_SCHEMA,
            _initial_state_semantic_payload(self),
        )
        if not self.semantic_content_sha256:
            object.__setattr__(self, "semantic_content_sha256", semantic)
        elif self.semantic_content_sha256 != semantic:
            raise SolarSystemContractError("semantic_content_sha256 does not bind the initial state")
        content = domain_sha256(_STATE_DOMAIN, _STATE_SCHEMA, _initial_state_payload(self))
        if not self.content_sha256:
            object.__setattr__(self, "content_sha256", content)
        elif self.content_sha256 != content:
            raise SolarSystemContractError("content_sha256 does not bind the initial state")

    def _validate(self) -> None:
        if type(self) is not De440NewtonianInitialState:
            raise SolarSystemContractError("initial state must have its exact concrete type")
        _text(self.state_id, "state_id")
        _text(self.force_plan_id, "force_plan_id")
        if self.force_plan_id == self.state_id:
            raise SolarSystemContractError("force_plan_id must differ from state_id")
        _preparation_ref(self.preparation_receipt)
        if type(self.body_ids) is not tuple or len(self.body_ids) != _BODY_COUNT:
            raise SolarSystemContractError("body_ids must contain the exact eleven-body roster")
        for index, body_id in enumerate(self.body_ids):
            _text(body_id, f"body_ids[{index}]")
        if self.body_ids != _BODY_IDS:
            raise SolarSystemContractError("body_ids differ from the closed resolved-eleven order")
        if type(self.naif_ids) is not tuple or len(self.naif_ids) != _BODY_COUNT:
            raise SolarSystemContractError("naif_ids must contain exactly eleven items")
        for index, naif_id in enumerate(self.naif_ids):
            _integer(naif_id, f"naif_ids[{index}]", minimum=1, maximum=(1 << 31) - 1)
        if self.naif_ids != _BODY_NAIF_IDS:
            raise SolarSystemContractError("naif_ids differ from the closed resolved-eleven order")
        if type(self.component_order) is not tuple or len(self.component_order) != 3:
            raise SolarSystemContractError("component_order must be an exact three-item tuple")
        for index, item in enumerate(self.component_order):
            _text(item, f"component_order[{index}]")
        if self.component_order != ("X", "Y", "Z"):
            raise SolarSystemContractError("component_order must be exactly X,Y,Z")
        if type(self.state_shape) is not tuple or len(self.state_shape) != 2:
            raise SolarSystemContractError("state_shape must be an exact two-item tuple")
        for index, extent in enumerate(self.state_shape):
            _integer(extent, f"state_shape[{index}]", minimum=1, maximum=_BODY_COUNT)
        if self.state_shape != (_BODY_COUNT, 3):
            raise SolarSystemContractError("state_shape must be exactly (11,3)")
        positions = _float_tuple(self.positions, _MAXIMUM_COMPONENTS, "positions")
        velocities = _float_tuple(self.velocities, _MAXIMUM_COMPONENTS, "velocities")
        gm_values = _float_tuple(self.gravitational_parameters, _BODY_COUNT, "gravitational_parameters")
        masses = _float_tuple(self.masses, _BODY_COUNT, "masses")
        radii = _float_tuple(self.radii, _BODY_COUNT, "radii")
        if any(value <= 0.0 for value in gm_values):
            raise SolarSystemContractError("every gravitational parameter must be positive")
        for label, values in (("masses", masses), ("radii", radii)):
            if any(value != 0.0 or math.copysign(1.0, value) < 0.0 for value in values):
                raise SolarSystemContractError(f"{label} must contain canonical positive-zero placeholders")
        if type(self.massive) is not tuple or len(self.massive) != _BODY_COUNT:
            raise SolarSystemContractError("massive must be an exact eleven-item tuple")
        if any(type(value) is not bool or not value for value in self.massive):
            raise SolarSystemContractError("every resolved-eleven body must be exactly massive=true")
        engine_epoch = _float(self.engine_epoch, "engine_epoch")
        if engine_epoch != 0.0 or math.copysign(1.0, engine_epoch) < 0.0:
            raise SolarSystemContractError("engine_epoch must be canonical positive zero")
        _token(self.engine_time_scale, "TDB", "engine_time_scale")
        _token(self.engine_frame, "BARYCENTRIC_INERTIAL", "engine_frame")
        _token(self.engine_origin, "BARYCENTER", "engine_origin")
        _token(self.engine_axes, "CARTESIAN_RIGHT_HANDED", "engine_axes")
        _token(self.engine_length_unit, METRE.unit_id, "engine_length_unit")
        _token(self.engine_time_unit, SECOND.unit_id, "engine_time_unit")
        _token(self.engine_mass_unit, KILOGRAM.unit_id, "engine_mass_unit")
        _token(self.engine_unit_system_id, _TARGET_UNIT_SYSTEM.unit_system_id, "engine_unit_system_id")
        _token(self.state_stage, _STATE_STAGE, "state_stage")
        _token(self.evidence_class, _EVIDENCE_CLASS, "evidence_class")
        if type(self.registry_authorized) is not bool or self.registry_authorized:
            raise SolarSystemContractError("registry_authorized must be exact false")
        if type(self.qualification_authorized) is not bool or self.qualification_authorized:
            raise SolarSystemContractError("qualification_authorized must be exact false")
        _token(self.content_integrity_class, _CONTENT_INTEGRITY_CLASS, "content_integrity_class")
        expected = _lane_preparation_values(
            self.preparation_receipt.source_execution.primary_lane,
            self.preparation_receipt.model_spec,
        )
        for label, actual, expected_values in (
            ("positions", positions, expected["positions"]),
            ("velocities", velocities, expected["velocities"]),
            ("gravitational_parameters", gm_values, expected["gm_values"]),
        ):
            if any(
                not _same_float(left, right)
                for left, right in zip(actual, expected_values)  # type: ignore[arg-type]
            ):
                raise SolarSystemContractError(f"{label} differs from exact preparation output")
        if hashlib.sha256(_state_payload(positions, velocities)).hexdigest() != self.preparation_receipt.recentered_payload_sha256:
            raise SolarSystemContractError("initial-state payload differs from preparation receipt")
        if hashlib.sha256(_gm_payload(gm_values)).hexdigest() != self.preparation_receipt.gm_payload_sha256:
            raise SolarSystemContractError("initial-state GM payload differs from preparation receipt")

    def validate_integrity(self) -> None:
        _sha(self.semantic_content_sha256, "semantic_content_sha256")
        _sha(self.content_sha256, "content_sha256")
        self._validate()
        expected_semantic = domain_sha256(
            _STATE_SEMANTIC_DOMAIN,
            _STATE_SEMANTIC_SCHEMA,
            _initial_state_semantic_payload(self),
        )
        if self.semantic_content_sha256 != expected_semantic:
            raise SolarSystemContractError("semantic_content_sha256 does not bind the initial state")
        if self.content_sha256 != domain_sha256(_STATE_DOMAIN, _STATE_SCHEMA, _initial_state_payload(self)):
            raise SolarSystemContractError("content_sha256 does not bind the initial state")


def validate_de440_newtonian_initial_state(value: object) -> None:
    if type(value) is not De440NewtonianInitialState:
        raise SolarSystemContractError("value must be an exact De440NewtonianInitialState")
    value.validate_integrity()


__all__ = [
    "De440GmProjection",
    "De440NewtonianInitialState",
    "De440NewtonianPreparationReceipt",
    "De440ResolvedEarthMoonNewtonianSpec",
    "TdbEngineEpochBinding",
    "validate_de440_gm_projection",
    "validate_de440_newtonian_initial_state",
    "validate_de440_newtonian_preparation_receipt",
    "validate_de440_resolved_earth_moon_newtonian_spec",
    "validate_tdb_engine_epoch_binding",
]
