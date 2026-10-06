"""Content-addressed physical inputs for experimental long-horizon work.

The known-system loader is deliberately fail-closed.  It verifies retained
NASA/JPL Horizons DE441 rows and NAIF DE440 gravitational parameters, repeats
the declared unit/frame conversion, and requires bitwise equality with a
packaged float-hex state before exposing the physical Sun plus nine planetary
system barycenters.

The compact-planet contracts do not authorize inference.  They only bind a
candidate's mass, gravitational parameter, elliptic elements, weight, and
provenance to matched physical arms whose known-system prefixes are bitwise
identical.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, replace
from decimal import Decimal, localcontext
import hashlib
from importlib import resources
import io
import json
import math
import re
import sys
from typing import Any, Mapping


KNOWN_SYSTEM_SCHEMA = "jx.long-horizon.known-solar-system-source-model.v2"
COMPACT_MEMBER_SCHEMA = "jx.long-horizon.bound-compact-planet-member.v1"
COMPACT_ENSEMBLE_SCHEMA = "jx.long-horizon.compact-planet-ensemble.v1"
MATCHED_ARM_SET_SCHEMA = "jx.long-horizon.matched-physical-arm-set.v1"

KNOWN_SYSTEM_MODEL_ID = "de441-de440-full-known-solar-system-jd2461200p5-v2"
KNOWN_SYSTEM_EPOCH_JD_TDB = 2_461_200.5
KNOWN_SYSTEM_BODY_IDS = (10, 1, 2, 3, 4, 5, 6, 7, 8, 9)
KNOWN_SYSTEM_BODY_NAMES = (
    "Sun",
    "Mercury Barycenter",
    "Venus Barycenter",
    "Earth-Moon Barycenter",
    "Mars Barycenter",
    "Jupiter Barycenter",
    "Saturn Barycenter",
    "Uranus Barycenter",
    "Neptune Barycenter",
    "Pluto Barycenter",
)

EPOCH_SCALE = "TDB"
SOURCE_FRAME = "SOLAR_SYSTEM_BARYCENTRIC_ICRF"
OUTPUT_FRAME = "PRIMARY_RELATIVE_IAU76_80_ECLIPTIC_J2000"
COMPACT_ELEMENT_FRAME = "HELIOCENTRIC_IAU76_80_ECLIPTIC_J2000_OSCULATING"
OUTPUT_COORDINATE_SYSTEM = "PRIMARY_RELATIVE_CARTESIAN_RIGHT_HANDED"
PRIMARY_ID = "NAIF-10-SUN"
LENGTH_UNIT = "AU"
SOURCE_VELOCITY_UNIT = "AU_PER_DAY"
TIME_UNIT = "JULIAN_YEAR_365.25_D"
VELOCITY_UNIT = "AU_PER_JULIAN_YEAR_365.25_D"
GM_UNIT = "AU3_PER_JULIAN_YEAR_365.25_D2"
BINARY64_SEMANTICS = (
    "IEEE_754_BINARY64_PYTHON_FLOAT_ROUND_TO_NEAREST_TIES_TO_EVEN"
)

JULIAN_YEAR_DAYS = float.fromhex("0x1.6d40000000000p+8")  # 365.25 exactly.
J2000_MEAN_OBLIQUITY_ARCSEC_EXACT = "84381.448"
J2000_MEAN_OBLIQUITY_RAD = float.fromhex("0x1.a2e9395ca7c3fp-2")
J2000_MEAN_OBLIQUITY_SIN = float.fromhex("0x1.9752e50f4b398p-2")
J2000_MEAN_OBLIQUITY_COS = float.fromhex("0x1.d5c0357681ef3p-1")
ROTATION_POLICY = (
    "IAU1976_J2000_MEAN_ECLIPTIC_ROTATE_EACH_BARYCENTRIC_STATE_THEN_"
    "SUBTRACT_ROTATED_SUN_84381.448_ARCSEC_SEPARATE_BINARY64_OPERATIONS_V1"
)
GM_CONVERSION_POLICY = (
    "EXACT_DECIMAL_KM3_S2_TO_AU3_PER_JULIAN_YEAR2_THEN_ONE_BINARY64_ROUND_V1"
)
CANDIDATE_STATE_POLICY = (
    "DE440_BODY10_SUN_PLUS_CANDIDATE_RELATIVE_TWO_BODY_OSCULATING_GENESIS_V2"
)
DE440_EARTH_GM_KM3_S2_EXACT = "398600.43550702266"
MASS_GM_POLICY = (
    "EARTH_MASS_TIMES_EXACT_DE440_BODY399_GM_ONE_BINARY64_ROUND_V1"
)
ENSEMBLE_WEIGHT_POLICY = (
    "EXPLICIT_POSITIVE_FINITE_UNIT_INTERVAL_RELATIVE_WEIGHTS_NOT_REQUIRED_TO_SUM_ONE"
)

PHYSICAL_CLAIM_FLAGS = (
    ("complete_de441_force_model", False),
    ("detection_claim", False),
    ("exclusion_claim", False),
    ("inference_authorized", False),
    ("long_duration_accuracy_qualified", False),
    ("observationally_fitted", False),
    ("science_result", False),
    ("uncertainty_calibrated", False),
)
PHYSICAL_MODEL_NONCLAIMS = (
    "not a retained DE441 SPK kernel or direct ephemeris evaluator",
    "not the complete DE441 force model",
    "not a resolved Earth-Moon or planetary-satellite model",
    "not long-duration accuracy or stability evidence",
    "not an observational fit or uncertainty model",
    "not evidence for detection, exclusion, orbit, mass, or sky position",
)

_DATA_DIRECTORY = "data/long_horizon_known_system_v2"
_SOURCE_ARTIFACT = "horizons_de441_barycentric_icrf_jd2461200p5.csv"
_GM_ARTIFACT = "gm_de440_major_barycenters.csv"
_DERIVED_ARTIFACT = "known_system_primary_relative_ecliptic_float_hex.csv"
_MANIFEST_ARTIFACT = "MANIFEST.json"
_README_ARTIFACT = "README.md"
_CHECKSUM_ARTIFACT = "SHA256SUMS"

SOURCE_ARTIFACT_SHA256 = (
    "97af9f841445a26d1c69dca46cfde73aa1a44c0651f19a1269a3facce6af8c9b"
)
GM_ARTIFACT_SHA256 = (
    "11fe03048f125e5283c3ef0d8054bc366baaf74dd86093ac3e61798e93769217"
)
DERIVED_ARTIFACT_SHA256 = (
    "6b3a2b1fbcee18821783c581a2e81ecdb9198a6da6550c81fbd03d579011bb8f"
)
MANIFEST_ARTIFACT_SHA256 = (
    "40d3e77c9ab5081f08354fe8e565e92008755059ef48ed9b7f43848c4f6f8231"
)
README_ARTIFACT_SHA256 = (
    "af0ae85eca293bc5d7f3b5d77cb913560adc363ce5c215dd29c300d67a1662e0"
)
CHECKSUM_ARTIFACT_SHA256 = (
    "bf4eb9ec805f450e3928a841c377b9584404486556fe1bf041f36df5b88175dd"
)
UPSTREAM_HORIZONS_VECTORS_SHA256 = (
    "18308086c9191333448e07a6ab6942a6b0fc7f0100ad2b3abaffbb1a0fd3993e"
)
UPSTREAM_HORIZONS_MANIFEST_SHA256 = (
    "87bf405df5d333dde237b8a6f63ba57a67baff60e74f752b5d1e77cc094d4565"
)
UPSTREAM_GM_DE440_KERNEL_SHA256 = (
    "924ddf4fb9ead9fe8a1aa55780bcabde40b09d00065d58226e24b68d8092f140"
)

_FROZEN_KNOWN_SYSTEM_SOURCE_STATE_SHA256 = (
    "d545472608abc19360b5fd020918c48bcd8cb9fe95da4b8adc315ba8a1513e29"
)
_FROZEN_KNOWN_SYSTEM_CONVERTED_STATE_SHA256 = (
    "3b6de264608aa07fca7281c28c2dfbccc66bd9bf7bcb84c098cb0b81decbdbee"
)
_FROZEN_KNOWN_SYSTEM_CONTENT_SHA256 = (
    "19d33011f45d07d15a2aff8acf56a48c57e726828d8a34f1354f070987b86387"
)

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SHA256_HEX = frozenset("0123456789abcdef")
_ZERO_HEX = "0x0.0p+0"


class PhysicalModelContractError(ValueError):
    """A physical-model artifact or immutable contract is inconsistent."""


def _require_binary64_runtime() -> None:
    if (
        sys.float_info.radix != 2
        or sys.float_info.mant_dig != 53
        or sys.float_info.max_exp != 1024
        or not float.__getformat__("double").startswith("IEEE")
    ):
        raise PhysicalModelContractError(
            "physical models require IEEE-754 binary64 Python floats"
        )


def _identifier(value: object, label: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise PhysicalModelContractError(
            f"{label} must match {_IDENTIFIER.pattern!r}"
        )
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise PhysicalModelContractError(
            f"{label} must be a nonempty trimmed built-in string"
        )
    return value


def _finite_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise PhysicalModelContractError(
            f"{label} must be a finite built-in float"
        )
    return value


def _positive_float(value: object, label: str) -> float:
    result = _finite_float(value, label)
    if result <= 0.0:
        raise PhysicalModelContractError(f"{label} must be positive")
    return result


def _sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise PhysicalModelContractError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _vec3(value: object, label: str) -> tuple[float, float, float]:
    if (
        type(value) is not tuple
        or len(value) != 3
        or any(type(item) is not float or not math.isfinite(item) for item in value)
    ):
        raise PhysicalModelContractError(
            f"{label} must be an exact tuple of three finite built-in floats"
        )
    return value


def _vec3_tuple(
    value: object, label: str, expected_count: int
) -> tuple[tuple[float, float, float], ...]:
    if type(value) is not tuple or len(value) != expected_count:
        raise PhysicalModelContractError(
            f"{label} must be an exact {expected_count}-element tuple"
        )
    for index, vector in enumerate(value):
        _vec3(vector, f"{label}[{index}]")
    return value


def _claims(value: object) -> tuple[tuple[str, bool], ...]:
    if type(value) is not tuple or value != PHYSICAL_CLAIM_FLAGS:
        raise PhysicalModelContractError(
            "claim_flags must be the exact closed all-false physical claim set"
        )
    for item in value:
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not bool
            or item[1]
        ):
            raise PhysicalModelContractError("every physical claim must be false")
    return value


def _nonclaims(value: object) -> tuple[str, ...]:
    if type(value) is not tuple or value != PHYSICAL_MODEL_NONCLAIMS:
        raise PhysicalModelContractError(
            "nonclaims must be the exact closed physical-model nonclaim set"
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


def _vector_record(value: tuple[float, float, float]) -> list[str]:
    return [_float_hex(component) for component in value]


def _same_float(left: float, right: float) -> bool:
    return left.hex() == right.hex()


def _same_vectors(
    left: tuple[tuple[float, float, float], ...],
    right: tuple[tuple[float, float, float], ...],
) -> bool:
    return len(left) == len(right) and all(
        all(_same_float(a, b) for a, b in zip(x, y, strict=True))
        for x, y in zip(left, right, strict=True)
    )


@dataclass(frozen=True, slots=True)
class KnownSolarSystemSourceModel:
    """Verified physical Sun plus nine DE441 planetary-system barycenters."""

    model_id: str
    epoch_jd_tdb: float
    body_ids: tuple[int, ...]
    body_names: tuple[str, ...]
    gm_AU3_per_julian_year2: tuple[float, ...]
    source_positions_barycentric_icrf_AU: tuple[
        tuple[float, float, float], ...
    ]
    source_velocities_barycentric_icrf_AU_per_day: tuple[
        tuple[float, float, float], ...
    ]
    positions_primary_relative_ecliptic_AU: tuple[
        tuple[float, float, float], ...
    ]
    velocities_primary_relative_ecliptic_AU_per_julian_year: tuple[
        tuple[float, float, float], ...
    ]
    source_artifact_path: str
    source_artifact_sha256: str
    gm_artifact_path: str
    gm_artifact_sha256: str
    derived_artifact_path: str
    derived_artifact_sha256: str
    manifest_artifact_path: str
    manifest_artifact_sha256: str
    upstream_horizons_vectors_sha256: str
    upstream_horizons_manifest_sha256: str
    upstream_gm_de440_kernel_sha256: str
    epoch_scale: str = EPOCH_SCALE
    source_frame: str = SOURCE_FRAME
    output_frame: str = OUTPUT_FRAME
    output_coordinate_system: str = OUTPUT_COORDINATE_SYSTEM
    primary_id: str = PRIMARY_ID
    length_unit: str = LENGTH_UNIT
    source_velocity_unit: str = SOURCE_VELOCITY_UNIT
    time_unit: str = TIME_UNIT
    velocity_unit: str = VELOCITY_UNIT
    gm_unit: str = GM_UNIT
    binary64_semantics: str = BINARY64_SEMANTICS
    rotation_policy: str = ROTATION_POLICY
    gm_conversion_policy: str = GM_CONVERSION_POLICY
    claim_flags: tuple[tuple[str, bool], ...] = PHYSICAL_CLAIM_FLAGS
    nonclaims: tuple[str, ...] = PHYSICAL_MODEL_NONCLAIMS
    source_barycentric_state_sha256: str = field(init=False)
    converted_state_sha256: str = field(init=False)
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_binary64_runtime()
        _identifier(self.model_id, "model_id")
        if not _same_float(
            _finite_float(self.epoch_jd_tdb, "epoch_jd_tdb"),
            KNOWN_SYSTEM_EPOCH_JD_TDB,
        ):
            raise PhysicalModelContractError("known-system epoch changed")
        if type(self.body_ids) is not tuple or self.body_ids != KNOWN_SYSTEM_BODY_IDS:
            raise PhysicalModelContractError(
                "body_ids must be Sun then NAIF barycenters 1 through 9"
            )
        if any(type(body_id) is not int for body_id in self.body_ids):
            raise PhysicalModelContractError("body IDs must be exact built-in integers")
        if (
            type(self.body_names) is not tuple
            or self.body_names != KNOWN_SYSTEM_BODY_NAMES
            or any(type(name) is not str for name in self.body_names)
        ):
            raise PhysicalModelContractError("known-system body names or order changed")
        count = len(KNOWN_SYSTEM_BODY_IDS)
        if type(self.gm_AU3_per_julian_year2) is not tuple or len(
            self.gm_AU3_per_julian_year2
        ) != count:
            raise PhysicalModelContractError("GM tuple must cover the exact source roster")
        for index, gm in enumerate(self.gm_AU3_per_julian_year2):
            _positive_float(gm, f"gm[{index}]")
        source_positions = _vec3_tuple(
            self.source_positions_barycentric_icrf_AU,
            "source positions",
            count,
        )
        source_velocities = _vec3_tuple(
            self.source_velocities_barycentric_icrf_AU_per_day,
            "source velocities",
            count,
        )
        positions = _vec3_tuple(
            self.positions_primary_relative_ecliptic_AU,
            "converted positions",
            count,
        )
        velocities = _vec3_tuple(
            self.velocities_primary_relative_ecliptic_AU_per_julian_year,
            "converted velocities",
            count,
        )
        if any(component.hex() != _ZERO_HEX for component in (*positions[0], *velocities[0])):
            raise PhysicalModelContractError(
                "the actual Sun must be exact positive-zero primary at slot 0"
            )
        for value, label in (
            (self.source_artifact_path, "source_artifact_path"),
            (self.gm_artifact_path, "gm_artifact_path"),
            (self.derived_artifact_path, "derived_artifact_path"),
            (self.manifest_artifact_path, "manifest_artifact_path"),
            (self.epoch_scale, "epoch_scale"),
            (self.source_frame, "source_frame"),
            (self.output_frame, "output_frame"),
            (self.output_coordinate_system, "output_coordinate_system"),
            (self.primary_id, "primary_id"),
            (self.length_unit, "length_unit"),
            (self.source_velocity_unit, "source_velocity_unit"),
            (self.time_unit, "time_unit"),
            (self.velocity_unit, "velocity_unit"),
            (self.gm_unit, "gm_unit"),
            (self.binary64_semantics, "binary64_semantics"),
            (self.rotation_policy, "rotation_policy"),
            (self.gm_conversion_policy, "gm_conversion_policy"),
        ):
            _text(value, label)
        for value, label in (
            (self.source_artifact_sha256, "source_artifact_sha256"),
            (self.gm_artifact_sha256, "gm_artifact_sha256"),
            (self.derived_artifact_sha256, "derived_artifact_sha256"),
            (self.manifest_artifact_sha256, "manifest_artifact_sha256"),
            (
                self.upstream_horizons_vectors_sha256,
                "upstream_horizons_vectors_sha256",
            ),
            (
                self.upstream_horizons_manifest_sha256,
                "upstream_horizons_manifest_sha256",
            ),
            (
                self.upstream_gm_de440_kernel_sha256,
                "upstream_gm_de440_kernel_sha256",
            ),
        ):
            _sha256(value, label)
        package_prefix = f"jxplanetx/{_DATA_DIRECTORY}"
        if (
            self.model_id != KNOWN_SYSTEM_MODEL_ID
            or self.source_artifact_path
            != f"{package_prefix}/{_SOURCE_ARTIFACT}"
            or self.source_artifact_sha256 != SOURCE_ARTIFACT_SHA256
            or self.gm_artifact_path != f"{package_prefix}/{_GM_ARTIFACT}"
            or self.gm_artifact_sha256 != GM_ARTIFACT_SHA256
            or self.derived_artifact_path
            != f"{package_prefix}/{_DERIVED_ARTIFACT}"
            or self.derived_artifact_sha256 != DERIVED_ARTIFACT_SHA256
            or self.manifest_artifact_path
            != f"{package_prefix}/{_MANIFEST_ARTIFACT}"
            or self.manifest_artifact_sha256 != MANIFEST_ARTIFACT_SHA256
            or self.upstream_horizons_vectors_sha256
            != UPSTREAM_HORIZONS_VECTORS_SHA256
            or self.upstream_horizons_manifest_sha256
            != UPSTREAM_HORIZONS_MANIFEST_SHA256
            or self.upstream_gm_de440_kernel_sha256
            != UPSTREAM_GM_DE440_KERNEL_SHA256
        ):
            raise PhysicalModelContractError(
                "known-system model or artifact custody changed"
            )
        if (
            self.epoch_scale != EPOCH_SCALE
            or self.source_frame != SOURCE_FRAME
            or self.output_frame != OUTPUT_FRAME
            or self.output_coordinate_system != OUTPUT_COORDINATE_SYSTEM
            or self.primary_id != PRIMARY_ID
            or self.length_unit != LENGTH_UNIT
            or self.source_velocity_unit != SOURCE_VELOCITY_UNIT
            or self.time_unit != TIME_UNIT
            or self.velocity_unit != VELOCITY_UNIT
            or self.gm_unit != GM_UNIT
            or self.binary64_semantics != BINARY64_SEMANTICS
            or self.rotation_policy != ROTATION_POLICY
            or self.gm_conversion_policy != GM_CONVERSION_POLICY
        ):
            raise PhysicalModelContractError(
                "known-system frame, origin, units, or conversion semantics changed"
            )
        _claims(self.claim_flags)
        _nonclaims(self.nonclaims)
        source_digest = _canonical_sha256(
            b"jx.long-horizon.known-system-barycentric-source-state.v2",
            {
                "epoch_jd_tdb_hex": self.epoch_jd_tdb.hex(),
                "body_ids": list(self.body_ids),
                "body_names": list(self.body_names),
                "positions_AU_hex": [_vector_record(v) for v in source_positions],
                "velocities_AU_per_day_hex": [
                    _vector_record(v) for v in source_velocities
                ],
            },
        )
        converted_digest = _canonical_sha256(
            b"jx.long-horizon.known-system-converted-state.v2",
            {
                "body_ids": list(self.body_ids),
                "gm_AU3_per_julian_year2_hex": [
                    value.hex() for value in self.gm_AU3_per_julian_year2
                ],
                "positions_AU_hex": [_vector_record(v) for v in positions],
                "velocities_AU_per_julian_year_hex": [
                    _vector_record(v) for v in velocities
                ],
            },
        )
        if (
            source_digest != _FROZEN_KNOWN_SYSTEM_SOURCE_STATE_SHA256
            or converted_digest != _FROZEN_KNOWN_SYSTEM_CONVERTED_STATE_SHA256
        ):
            raise PhysicalModelContractError(
                "known-system retained source or converted state changed"
            )
        object.__setattr__(self, "source_barycentric_state_sha256", source_digest)
        object.__setattr__(self, "converted_state_sha256", converted_digest)
        content_digest = _canonical_sha256(
            b"jx.long-horizon.known-solar-system-source-model.v2",
            self._canonical_record(source_digest, converted_digest),
        )
        if content_digest != _FROZEN_KNOWN_SYSTEM_CONTENT_SHA256:
            raise PhysicalModelContractError(
                "known-system content identity changed"
            )
        object.__setattr__(
            self,
            "content_sha256",
            content_digest,
        )

    def _canonical_record(
        self, source_digest: str | None = None, converted_digest: str | None = None
    ) -> dict[str, Any]:
        return {
            "schema": KNOWN_SYSTEM_SCHEMA,
            "model_id": self.model_id,
            "epoch_jd_tdb_hex": self.epoch_jd_tdb.hex(),
            "epoch_scale": self.epoch_scale,
            "body_ids": list(self.body_ids),
            "body_names": list(self.body_names),
            "source_frame": self.source_frame,
            "output_frame": self.output_frame,
            "output_coordinate_system": self.output_coordinate_system,
            "primary_id": self.primary_id,
            "length_unit": self.length_unit,
            "source_velocity_unit": self.source_velocity_unit,
            "time_unit": self.time_unit,
            "velocity_unit": self.velocity_unit,
            "gm_unit": self.gm_unit,
            "binary64_semantics": self.binary64_semantics,
            "rotation_policy": self.rotation_policy,
            "gm_conversion_policy": self.gm_conversion_policy,
            "source_barycentric_state_sha256": source_digest
            or self.source_barycentric_state_sha256,
            "converted_state_sha256": converted_digest or self.converted_state_sha256,
            "artifacts": {
                "source": [self.source_artifact_path, self.source_artifact_sha256],
                "gm": [self.gm_artifact_path, self.gm_artifact_sha256],
                "derived": [self.derived_artifact_path, self.derived_artifact_sha256],
                "manifest": [
                    self.manifest_artifact_path,
                    self.manifest_artifact_sha256,
                ],
            },
            "upstream": {
                "horizons_vectors": self.upstream_horizons_vectors_sha256,
                "horizons_manifest": self.upstream_horizons_manifest_sha256,
                "gm_de440_kernel": self.upstream_gm_de440_kernel_sha256,
            },
            "claim_flags": [list(item) for item in self.claim_flags],
            "nonclaims": list(self.nonclaims),
        }

    @property
    def schema(self) -> str:
        return KNOWN_SYSTEM_SCHEMA

    @property
    def source_count(self) -> int:
        return len(self.body_ids)

    @property
    def primary_gm_AU3_per_julian_year2(self) -> float:
        return self.gm_AU3_per_julian_year2[0]


@dataclass(frozen=True, slots=True)
class BoundCompactPlanetMember:
    """One provenance-bound, weighted, bound compact-planet hypothesis."""

    member_id: str
    epoch_jd_tdb: float
    mass_earth_masses: float
    gm_AU3_per_julian_year2: float
    semimajor_axis_AU: float
    eccentricity: float
    inclination_rad: float
    longitude_of_ascending_node_rad: float
    argument_of_periapsis_rad: float
    mean_anomaly_rad: float
    prior_weight: float
    provenance_id: str
    provenance_artifact_path: str
    provenance_artifact_sha256: str
    element_frame: str = COMPACT_ELEMENT_FRAME
    primary_id: str = PRIMARY_ID
    mass_gm_policy: str = MASS_GM_POLICY
    candidate_state_policy: str = CANDIDATE_STATE_POLICY
    claim_flags: tuple[tuple[str, bool], ...] = PHYSICAL_CLAIM_FLAGS
    nonclaims: tuple[str, ...] = PHYSICAL_MODEL_NONCLAIMS
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.member_id, "member_id")
        if not _same_float(
            _finite_float(self.epoch_jd_tdb, "epoch_jd_tdb"),
            KNOWN_SYSTEM_EPOCH_JD_TDB,
        ):
            raise PhysicalModelContractError(
                "compact-planet epoch must equal the known-system epoch"
            )
        _positive_float(self.mass_earth_masses, "mass_earth_masses")
        _positive_float(
            self.gm_AU3_per_julian_year2, "gm_AU3_per_julian_year2"
        )
        expected_gm = compact_planet_gm_from_earth_masses(
            self.mass_earth_masses
        )
        if not _same_float(self.gm_AU3_per_julian_year2, expected_gm):
            raise PhysicalModelContractError(
                "compact-planet GM does not exactly match its Earth-mass value"
            )
        semimajor_axis = _positive_float(self.semimajor_axis_AU, "semimajor_axis_AU")
        eccentricity = _finite_float(self.eccentricity, "eccentricity")
        if not 0.0 <= eccentricity < 1.0:
            raise PhysicalModelContractError(
                "compact-planet candidate must be a bound elliptic orbit"
            )
        if eccentricity > 0.9:
            raise PhysicalModelContractError(
                "compact-planet candidate eccentricity may not exceed 0.9"
            )
        if semimajor_axis * (1.0 - eccentricity) <= 0.0:
            raise PhysicalModelContractError("candidate perihelion must be positive")
        inclination = _finite_float(self.inclination_rad, "inclination_rad")
        if not 0.0 <= inclination <= math.pi:
            raise PhysicalModelContractError("inclination_rad must be in [0, pi]")
        for value, label in (
            (
                self.longitude_of_ascending_node_rad,
                "longitude_of_ascending_node_rad",
            ),
            (self.argument_of_periapsis_rad, "argument_of_periapsis_rad"),
            (self.mean_anomaly_rad, "mean_anomaly_rad"),
        ):
            angle = _finite_float(value, label)
            if not 0.0 <= angle < math.tau:
                raise PhysicalModelContractError(f"{label} must be in [0, 2*pi)")
        weight = _positive_float(self.prior_weight, "prior_weight")
        if weight > 1.0:
            raise PhysicalModelContractError(
                "prior_weight must be normalized-compatible in (0, 1]"
            )
        for value, label in (
            (self.provenance_id, "provenance_id"),
            (self.provenance_artifact_path, "provenance_artifact_path"),
            (self.element_frame, "element_frame"),
            (self.primary_id, "primary_id"),
            (self.mass_gm_policy, "mass_gm_policy"),
            (self.candidate_state_policy, "candidate_state_policy"),
        ):
            _text(value, label)
        _sha256(self.provenance_artifact_sha256, "provenance_artifact_sha256")
        if (
            self.element_frame != COMPACT_ELEMENT_FRAME
            or self.primary_id != PRIMARY_ID
            or self.mass_gm_policy != MASS_GM_POLICY
            or self.candidate_state_policy != CANDIDATE_STATE_POLICY
        ):
            raise PhysicalModelContractError(
                "candidate epoch, frame, primary, or mass/GM semantics changed"
            )
        _claims(self.claim_flags)
        _nonclaims(self.nonclaims)
        object.__setattr__(
            self,
            "content_sha256",
            _canonical_sha256(
                b"jx.long-horizon.bound-compact-planet-member.v1",
                self._canonical_record(),
            ),
        )

    def _canonical_record(self) -> dict[str, Any]:
        return {
            "schema": COMPACT_MEMBER_SCHEMA,
            "member_id": self.member_id,
            "epoch_jd_tdb_hex": self.epoch_jd_tdb.hex(),
            "element_frame": self.element_frame,
            "primary_id": self.primary_id,
            "mass_earth_masses_hex": self.mass_earth_masses.hex(),
            "gm_AU3_per_julian_year2_hex": self.gm_AU3_per_julian_year2.hex(),
            "elements_hex": {
                "semimajor_axis_AU": self.semimajor_axis_AU.hex(),
                "eccentricity": self.eccentricity.hex(),
                "inclination_rad": self.inclination_rad.hex(),
                "longitude_of_ascending_node_rad": (
                    self.longitude_of_ascending_node_rad.hex()
                ),
                "argument_of_periapsis_rad": self.argument_of_periapsis_rad.hex(),
                "mean_anomaly_rad": self.mean_anomaly_rad.hex(),
            },
            "prior_weight_hex": self.prior_weight.hex(),
            "provenance": {
                "id": self.provenance_id,
                "path": self.provenance_artifact_path,
                "sha256": self.provenance_artifact_sha256,
            },
            "mass_gm_policy": self.mass_gm_policy,
            "candidate_state_policy": self.candidate_state_policy,
            "claim_flags": [list(item) for item in self.claim_flags],
            "nonclaims": list(self.nonclaims),
        }

    @property
    def schema(self) -> str:
        return COMPACT_MEMBER_SCHEMA

    @property
    def perihelion_AU(self) -> float:
        return self.semimajor_axis_AU * (1.0 - self.eccentricity)

    def as_orbital_elements(self):
        """Return the exact existing JX element contract without provenance widening."""

        from .science_inputs import OrbitalElements

        return OrbitalElements(
            object_id=self.member_id,
            semimajor_axis_AU=self.semimajor_axis_AU,
            eccentricity=self.eccentricity,
            inclination_rad=self.inclination_rad,
            longitude_of_ascending_node_rad=self.longitude_of_ascending_node_rad,
            argument_of_periapsis_rad=self.argument_of_periapsis_rad,
            mean_anomaly_rad=self.mean_anomaly_rad,
            observed_source=None,
        )


@dataclass(frozen=True, slots=True)
class CompactPlanetEnsemble:
    """An ordered provenance-bound set of weighted compact-planet members."""

    ensemble_id: str
    members: tuple[BoundCompactPlanetMember, ...]
    provenance_id: str
    provenance_artifact_path: str
    provenance_artifact_sha256: str
    weight_policy: str = ENSEMBLE_WEIGHT_POLICY
    claim_flags: tuple[tuple[str, bool], ...] = PHYSICAL_CLAIM_FLAGS
    nonclaims: tuple[str, ...] = PHYSICAL_MODEL_NONCLAIMS
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.ensemble_id, "ensemble_id")
        if (
            type(self.members) is not tuple
            or not self.members
            or any(type(member) is not BoundCompactPlanetMember for member in self.members)
        ):
            raise PhysicalModelContractError(
                "members must be a nonempty exact BoundCompactPlanetMember tuple"
            )
        member_ids = tuple(member.member_id for member in self.members)
        if len(set(member_ids)) != len(member_ids):
            raise PhysicalModelContractError("compact-planet member IDs must be unique")
        if any(
            not _same_float(member.epoch_jd_tdb, KNOWN_SYSTEM_EPOCH_JD_TDB)
            or member.element_frame != COMPACT_ELEMENT_FRAME
            for member in self.members
        ):
            raise PhysicalModelContractError(
                "ensemble members must share the exact known-system epoch and frame"
            )
        for value, label in (
            (self.provenance_id, "provenance_id"),
            (self.provenance_artifact_path, "provenance_artifact_path"),
            (self.weight_policy, "weight_policy"),
        ):
            _text(value, label)
        _sha256(self.provenance_artifact_sha256, "provenance_artifact_sha256")
        if self.weight_policy != ENSEMBLE_WEIGHT_POLICY:
            raise PhysicalModelContractError("ensemble weight semantics changed")
        _claims(self.claim_flags)
        _nonclaims(self.nonclaims)
        object.__setattr__(
            self,
            "content_sha256",
            _canonical_sha256(
                b"jx.long-horizon.compact-planet-ensemble.v1",
                {
                    "schema": COMPACT_ENSEMBLE_SCHEMA,
                    "ensemble_id": self.ensemble_id,
                    "member_sha256": [
                        member.content_sha256 for member in self.members
                    ],
                    "weight_policy": self.weight_policy,
                    "provenance": {
                        "id": self.provenance_id,
                        "path": self.provenance_artifact_path,
                        "sha256": self.provenance_artifact_sha256,
                    },
                    "claim_flags": [list(item) for item in self.claim_flags],
                    "nonclaims": list(self.nonclaims),
                },
            ),
        )

    @property
    def schema(self) -> str:
        return COMPACT_ENSEMBLE_SCHEMA

    @property
    def weights(self) -> tuple[float, ...]:
        return tuple(member.prior_weight for member in self.members)

    @property
    def total_weight(self) -> float:
        return math.fsum(self.weights)


def _known_prefix_is_bitwise_identical(
    reference: KnownSolarSystemSourceModel,
    candidate: KnownSolarSystemSourceModel,
) -> bool:
    return (
        candidate.body_ids == reference.body_ids
        and candidate.body_names == reference.body_names
        and len(candidate.gm_AU3_per_julian_year2)
        == len(reference.gm_AU3_per_julian_year2)
        and all(
            _same_float(left, right)
            for left, right in zip(
                candidate.gm_AU3_per_julian_year2,
                reference.gm_AU3_per_julian_year2,
                strict=True,
            )
        )
        and _same_vectors(
            candidate.positions_primary_relative_ecliptic_AU,
            reference.positions_primary_relative_ecliptic_AU,
        )
        and _same_vectors(
            candidate.velocities_primary_relative_ecliptic_AU_per_julian_year,
            reference.velocities_primary_relative_ecliptic_AU_per_julian_year,
        )
        and candidate.converted_state_sha256 == reference.converted_state_sha256
        and candidate.content_sha256 == reference.content_sha256
    )


def _candidate_cartesian_state(
    member: BoundCompactPlanetMember,
    known_source_model: KnownSolarSystemSourceModel,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    from .initial_conditions import (
        DE440_SUN_GENESIS_MODEL,
        InitialConditionContractError,
        orbital_elements_to_primary_relative_cartesian,
    )

    if not _same_float(
        known_source_model.primary_gm_AU3_per_julian_year2,
        DE440_SUN_GENESIS_MODEL.primary_gm_AU3_per_year2,
    ):
        raise PhysicalModelContractError(
            "known-system Sun GM differs from the locked DE440 Sun genesis model"
        )
    relative_parameter_sha256 = _canonical_sha256(
        b"jx.long-horizon.compact-planet-relative-two-body-parameter.v1",
        {
            "known_source_model_sha256": known_source_model.content_sha256,
            "candidate_member_sha256": member.content_sha256,
            "sun_gm_hex": known_source_model.primary_gm_AU3_per_julian_year2.hex(),
            "candidate_gm_hex": member.gm_AU3_per_julian_year2.hex(),
            "relative_parameter_gm_hex": (
                known_source_model.primary_gm_AU3_per_julian_year2
                + member.gm_AU3_per_julian_year2
            ).hex(),
        },
    )
    candidate_model = replace(
        DE440_SUN_GENESIS_MODEL,
        model_id=(
            "de440-sun-plus-candidate-" + member.content_sha256[:16]
        ),
        primary_definition=(
            "RELATIVE_TWO_BODY_PARAMETER_GM_DE440_SUN_PLUS_CANDIDATE"
        ),
        primary_gm_AU3_per_year2=(
            known_source_model.primary_gm_AU3_per_julian_year2
            + member.gm_AU3_per_julian_year2
        ),
        primary_gm_source_id=(
            "DE440_BODY10_PLUS_PROVENANCE_BOUND_CANDIDATE_GM"
        ),
        primary_gm_source_artifact_path=(
            known_source_model.gm_artifact_path
            + "+"
            + member.provenance_artifact_path
        ),
        primary_gm_source_artifact_sha256=relative_parameter_sha256,
        source_model_manifest_path=known_source_model.manifest_artifact_path,
        source_model_manifest_sha256=known_source_model.manifest_artifact_sha256,
    )
    try:
        return orbital_elements_to_primary_relative_cartesian(
            member.as_orbital_elements(), model=candidate_model
        )
    except InitialConditionContractError as exc:
        raise PhysicalModelContractError(
            "candidate cannot be converted by the locked precision-first genesis model"
        ) from exc


@dataclass(frozen=True, slots=True)
class MatchedPhysicalArmSet:
    """Candidate arms with bitwise-identical known-source prefixes."""

    arm_set_id: str
    known_source_model: KnownSolarSystemSourceModel
    candidate_ensemble: CompactPlanetEnsemble
    arm_known_source_prefixes: tuple[KnownSolarSystemSourceModel, ...]
    candidate_positions_primary_relative_ecliptic_AU: tuple[
        tuple[float, float, float], ...
    ]
    candidate_velocities_primary_relative_ecliptic_AU_per_julian_year: tuple[
        tuple[float, float, float], ...
    ]
    provenance_id: str
    provenance_artifact_path: str
    provenance_artifact_sha256: str
    candidate_state_policy: str = CANDIDATE_STATE_POLICY
    append_policy: str = "APPEND_ONE_CANDIDATE_AFTER_PLUTO_BARYCENTER"
    claim_flags: tuple[tuple[str, bool], ...] = PHYSICAL_CLAIM_FLAGS
    nonclaims: tuple[str, ...] = PHYSICAL_MODEL_NONCLAIMS
    arm_state_sha256: tuple[str, ...] = field(init=False)
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.arm_set_id, "arm_set_id")
        if type(self.known_source_model) is not KnownSolarSystemSourceModel:
            raise PhysicalModelContractError(
                "known_source_model must be an exact KnownSolarSystemSourceModel"
            )
        if type(self.candidate_ensemble) is not CompactPlanetEnsemble:
            raise PhysicalModelContractError(
                "candidate_ensemble must be an exact CompactPlanetEnsemble"
            )
        count = len(self.candidate_ensemble.members)
        if (
            type(self.arm_known_source_prefixes) is not tuple
            or len(self.arm_known_source_prefixes) != count
            or any(
                type(prefix) is not KnownSolarSystemSourceModel
                for prefix in self.arm_known_source_prefixes
            )
        ):
            raise PhysicalModelContractError(
                "every candidate arm requires one exact known-source prefix"
            )
        if any(
            not _known_prefix_is_bitwise_identical(self.known_source_model, prefix)
            for prefix in self.arm_known_source_prefixes
        ):
            raise PhysicalModelContractError(
                "all matched arms must share a bitwise exact known-source prefix"
            )
        positions = _vec3_tuple(
            self.candidate_positions_primary_relative_ecliptic_AU,
            "candidate positions",
            count,
        )
        velocities = _vec3_tuple(
            self.candidate_velocities_primary_relative_ecliptic_AU_per_julian_year,
            "candidate velocities",
            count,
        )
        for index, (member, position, velocity) in enumerate(
            zip(self.candidate_ensemble.members, positions, velocities, strict=True)
        ):
            expected_position, expected_velocity = _candidate_cartesian_state(
                member, self.known_source_model
            )
            if not all(
                _same_float(left, right)
                for left, right in zip(position, expected_position, strict=True)
            ) or not all(
                _same_float(left, right)
                for left, right in zip(velocity, expected_velocity, strict=True)
            ):
                raise PhysicalModelContractError(
                    f"candidate arm {index} is not the locked element conversion"
                )
        if self.known_source_model.body_names[-1] != "Pluto Barycenter":
            raise PhysicalModelContractError(
                "known-source prefix must end with Pluto Barycenter"
            )
        for value, label in (
            (self.provenance_id, "provenance_id"),
            (self.provenance_artifact_path, "provenance_artifact_path"),
            (self.candidate_state_policy, "candidate_state_policy"),
            (self.append_policy, "append_policy"),
        ):
            _text(value, label)
        _sha256(self.provenance_artifact_sha256, "provenance_artifact_sha256")
        if (
            self.candidate_state_policy != CANDIDATE_STATE_POLICY
            or self.append_policy != "APPEND_ONE_CANDIDATE_AFTER_PLUTO_BARYCENTER"
        ):
            raise PhysicalModelContractError("matched-arm construction semantics changed")
        _claims(self.claim_flags)
        _nonclaims(self.nonclaims)
        arm_digests = tuple(
            _canonical_sha256(
                b"jx.long-horizon.matched-physical-arm-state.v1",
                {
                    "known_source_converted_state_sha256": (
                        prefix.converted_state_sha256
                    ),
                    "candidate_member_sha256": member.content_sha256,
                    "candidate_position_AU_hex": _vector_record(position),
                    "candidate_velocity_AU_per_julian_year_hex": (
                        _vector_record(velocity)
                    ),
                    "candidate_gm_AU3_per_julian_year2_hex": (
                        member.gm_AU3_per_julian_year2.hex()
                    ),
                    "append_policy": self.append_policy,
                },
            )
            for prefix, member, position, velocity in zip(
                self.arm_known_source_prefixes,
                self.candidate_ensemble.members,
                positions,
                velocities,
                strict=True,
            )
        )
        object.__setattr__(self, "arm_state_sha256", arm_digests)
        object.__setattr__(
            self,
            "content_sha256",
            _canonical_sha256(
                b"jx.long-horizon.matched-physical-arm-set.v1",
                {
                    "schema": MATCHED_ARM_SET_SCHEMA,
                    "arm_set_id": self.arm_set_id,
                    "known_source_model_sha256": self.known_source_model.content_sha256,
                    "candidate_ensemble_sha256": (
                        self.candidate_ensemble.content_sha256
                    ),
                    "arm_state_sha256": list(arm_digests),
                    "candidate_state_policy": self.candidate_state_policy,
                    "append_policy": self.append_policy,
                    "provenance": {
                        "id": self.provenance_id,
                        "path": self.provenance_artifact_path,
                        "sha256": self.provenance_artifact_sha256,
                    },
                    "claim_flags": [list(item) for item in self.claim_flags],
                    "nonclaims": list(self.nonclaims),
                },
            ),
        )

    @property
    def schema(self) -> str:
        return MATCHED_ARM_SET_SCHEMA

    @property
    def arm_count(self) -> int:
        return len(self.candidate_ensemble.members)

    @property
    def arm_body_names(self) -> tuple[tuple[str, ...], ...]:
        return tuple(
            prefix.body_names + (member.member_id,)
            for prefix, member in zip(
                self.arm_known_source_prefixes,
                self.candidate_ensemble.members,
                strict=True,
            )
        )

    @property
    def arm_gm_AU3_per_julian_year2(self) -> tuple[tuple[float, ...], ...]:
        return tuple(
            prefix.gm_AU3_per_julian_year2
            + (member.gm_AU3_per_julian_year2,)
            for prefix, member in zip(
                self.arm_known_source_prefixes,
                self.candidate_ensemble.members,
                strict=True,
            )
        )

    @property
    def arm_positions_primary_relative_ecliptic_AU(
        self,
    ) -> tuple[tuple[tuple[float, float, float], ...], ...]:
        return tuple(
            prefix.positions_primary_relative_ecliptic_AU + (position,)
            for prefix, position in zip(
                self.arm_known_source_prefixes,
                self.candidate_positions_primary_relative_ecliptic_AU,
                strict=True,
            )
        )

    @property
    def arm_velocities_primary_relative_ecliptic_AU_per_julian_year(
        self,
    ) -> tuple[tuple[tuple[float, float, float], ...], ...]:
        return tuple(
            prefix.velocities_primary_relative_ecliptic_AU_per_julian_year
            + (velocity,)
            for prefix, velocity in zip(
                self.arm_known_source_prefixes,
                self.candidate_velocities_primary_relative_ecliptic_AU_per_julian_year,
                strict=True,
            )
        )


def _resource_bytes(name: str, expected_sha256: str) -> bytes:
    root = resources.files("jxplanetx")
    resource = root
    for part in (*_DATA_DIRECTORY.split("/"), name):
        resource = resource.joinpath(part)
    try:
        payload = resource.read_bytes()
    except (FileNotFoundError, OSError) as exc:
        raise PhysicalModelContractError(
            f"required packaged physical-model artifact {name!r} is unavailable"
        ) from exc
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_sha256:
        raise PhysicalModelContractError(
            f"packaged physical-model artifact {name!r} failed SHA-256 custody"
        )
    return payload


def _decode_ascii_csv(payload: bytes, label: str) -> list[dict[str, str]]:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise PhysicalModelContractError(f"{label} is not exact ASCII") from exc
    with io.StringIO(text, newline="") as stream:
        return list(csv.DictReader(stream))


def _rotate_icrf_to_ecliptic(
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    x, y, z = vector
    return (
        x,
        J2000_MEAN_OBLIQUITY_COS * y + J2000_MEAN_OBLIQUITY_SIN * z,
        -J2000_MEAN_OBLIQUITY_SIN * y + J2000_MEAN_OBLIQUITY_COS * z,
    )


def _converted_gm(source_decimal: str) -> float:
    try:
        with localcontext() as context:
            context.prec = 80
            seconds_per_year = Decimal(86_400) * Decimal("365.25")
            factor = seconds_per_year**2 / Decimal("149597870.7") ** 3
            return float(Decimal(source_decimal) * factor)
    except Exception as exc:
        raise PhysicalModelContractError("invalid exact-decimal GM source") from exc


def compact_planet_gm_from_earth_masses(mass_earth_masses: float) -> float:
    """Convert one binary64 Earth-mass value using exact DE440 body-399 GM."""

    mass = _positive_float(mass_earth_masses, "mass_earth_masses")
    try:
        with localcontext() as context:
            context.prec = 80
            seconds_per_year = Decimal(86_400) * Decimal("365.25")
            factor = seconds_per_year**2 / Decimal("149597870.7") ** 3
            value = (
                Decimal.from_float(mass)
                * Decimal(DE440_EARTH_GM_KM3_S2_EXACT)
                * factor
            )
            result = float(value)
    except Exception as exc:
        raise PhysicalModelContractError(
            "invalid compact-planet Earth-mass conversion"
        ) from exc
    return _positive_float(result, "converted compact-planet GM")


DE440_EARTH_GM_AU3_PER_JULIAN_YEAR2 = (
    compact_planet_gm_from_earth_masses(1.0)
)


def load_known_solar_system_source_model() -> KnownSolarSystemSourceModel:
    """Load and independently verify the packaged ten-body source model."""

    _require_binary64_runtime()
    manifest_payload = _resource_bytes(_MANIFEST_ARTIFACT, MANIFEST_ARTIFACT_SHA256)
    _resource_bytes(_README_ARTIFACT, README_ARTIFACT_SHA256)
    checksum_payload = _resource_bytes(_CHECKSUM_ARTIFACT, CHECKSUM_ARTIFACT_SHA256)
    source_payload = _resource_bytes(_SOURCE_ARTIFACT, SOURCE_ARTIFACT_SHA256)
    gm_payload = _resource_bytes(_GM_ARTIFACT, GM_ARTIFACT_SHA256)
    derived_payload = _resource_bytes(_DERIVED_ARTIFACT, DERIVED_ARTIFACT_SHA256)
    try:
        manifest = json.loads(manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PhysicalModelContractError("known-system manifest is invalid JSON") from exc
    if (
        type(manifest) is not dict
        or manifest.get("schema") != "jx.long-horizon.known-system-source-manifest.v2"
        or manifest.get("artifact_id")
        != "jx.long-horizon.full-known-solar-system-source.v2"
        or tuple(manifest.get("body_ids_ordered", ())) != KNOWN_SYSTEM_BODY_IDS
        or tuple(manifest.get("body_names_ordered", ())) != KNOWN_SYSTEM_BODY_NAMES
        or manifest.get("epoch_jd_tdb") != KNOWN_SYSTEM_EPOCH_JD_TDB
        or tuple(manifest.get("claim_flags", {}).items()) != PHYSICAL_CLAIM_FLAGS
        or tuple(manifest.get("nonclaims", ())) != PHYSICAL_MODEL_NONCLAIMS
    ):
        raise PhysicalModelContractError(
            "known-system manifest semantics or all-false claims changed"
        )
    expected_checksum_lines = {
        f"{MANIFEST_ARTIFACT_SHA256}  {_MANIFEST_ARTIFACT}",
        f"{README_ARTIFACT_SHA256}  {_README_ARTIFACT}",
        f"{GM_ARTIFACT_SHA256}  {_GM_ARTIFACT}",
        f"{SOURCE_ARTIFACT_SHA256}  {_SOURCE_ARTIFACT}",
        f"{DERIVED_ARTIFACT_SHA256}  {_DERIVED_ARTIFACT}",
    }
    if set(checksum_payload.decode("ascii").splitlines()) != expected_checksum_lines:
        raise PhysicalModelContractError("known-system checksum inventory changed")

    source_rows = _decode_ascii_csv(source_payload, "Horizons source rows")
    if len(source_rows) != len(KNOWN_SYSTEM_BODY_IDS):
        raise PhysicalModelContractError("source artifact must contain exactly ten rows")
    source_positions: list[tuple[float, float, float]] = []
    source_velocities: list[tuple[float, float, float]] = []
    for index, (row, body_id, body_name) in enumerate(
        zip(source_rows, KNOWN_SYSTEM_BODY_IDS, KNOWN_SYSTEM_BODY_NAMES, strict=True)
    ):
        if (
            row.get("body_id") != str(body_id)
            or row.get("body_name") != body_name
            or row.get("jd_tdb") != "2461200.500000000"
            or row.get("calendar_tdb") != "A.D. 2026-Jun-09 00:00:00.0000"
        ):
            raise PhysicalModelContractError(
                f"source row {index} identity, epoch, or order changed"
            )
        try:
            position = tuple(float(row[name]) for name in ("x_au", "y_au", "z_au"))
            velocity = tuple(
                float(row[name])
                for name in ("vx_au_per_day", "vy_au_per_day", "vz_au_per_day")
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PhysicalModelContractError(
                f"source row {index} has invalid Cartesian values"
            ) from exc
        source_positions.append(_vec3(position, f"source position {index}"))
        source_velocities.append(_vec3(velocity, f"source velocity {index}"))

    gm_rows = _decode_ascii_csv(gm_payload, "DE440 GM rows")
    if len(gm_rows) != len(KNOWN_SYSTEM_BODY_IDS):
        raise PhysicalModelContractError("GM artifact must contain exactly ten rows")
    gm_by_id: dict[int, tuple[str, str]] = {}
    for row in gm_rows:
        try:
            body_id = int(row["body_id"])
            name = row["body_name"]
            decimal_value = row["gm_km3_s2"]
        except (KeyError, TypeError, ValueError) as exc:
            raise PhysicalModelContractError("GM row is malformed") from exc
        if body_id in gm_by_id:
            raise PhysicalModelContractError("GM body IDs must be unique")
        gm_by_id[body_id] = (name, decimal_value)
    if tuple(sorted(gm_by_id)) != tuple(range(1, 11)):
        raise PhysicalModelContractError("GM table must contain body IDs 1 through 10")

    derived_rows = _decode_ascii_csv(derived_payload, "derived known-system state")
    if len(derived_rows) != len(KNOWN_SYSTEM_BODY_IDS):
        raise PhysicalModelContractError("derived state must contain exactly ten rows")
    retained_gm: list[float] = []
    retained_positions: list[tuple[float, float, float]] = []
    retained_velocities: list[tuple[float, float, float]] = []
    for index, (row, body_id, body_name) in enumerate(
        zip(derived_rows, KNOWN_SYSTEM_BODY_IDS, KNOWN_SYSTEM_BODY_NAMES, strict=True)
    ):
        if (
            row.get("slot") != str(index)
            or row.get("body_id") != str(body_id)
            or row.get("body_name") != body_name
        ):
            raise PhysicalModelContractError(
                f"derived row {index} identity or order changed"
            )
        columns = (
            "gm_au3_per_julian_year2_hex",
            "x_au_hex",
            "y_au_hex",
            "z_au_hex",
            "vx_au_per_julian_year_hex",
            "vy_au_per_julian_year_hex",
            "vz_au_per_julian_year_hex",
        )
        try:
            parsed = tuple(float.fromhex(row[name]) for name in columns)
        except (KeyError, TypeError, ValueError) as exc:
            raise PhysicalModelContractError(
                f"derived row {index} has invalid float-hex data"
            ) from exc
        if any(
            not math.isfinite(value) or value.hex() != row[name]
            for value, name in zip(parsed, columns, strict=True)
        ):
            raise PhysicalModelContractError(
                f"derived row {index} float-hex data is not canonical"
            )
        retained_gm.append(parsed[0])
        retained_positions.append(parsed[1:4])
        retained_velocities.append(parsed[4:7])

    rotated_source_positions = [
        _rotate_icrf_to_ecliptic(position) for position in source_positions
    ]
    rotated_source_velocities = [
        _rotate_icrf_to_ecliptic(velocity) for velocity in source_velocities
    ]
    sun_position = rotated_source_positions[0]
    sun_velocity = rotated_source_velocities[0]
    recomputed_gm: list[float] = []
    recomputed_positions: list[tuple[float, float, float]] = []
    recomputed_velocities: list[tuple[float, float, float]] = []
    for index, (body_id, body_name, position, velocity) in enumerate(
        zip(
            KNOWN_SYSTEM_BODY_IDS,
            KNOWN_SYSTEM_BODY_NAMES,
            rotated_source_positions,
            rotated_source_velocities,
            strict=True,
        )
    ):
        gm_name, gm_decimal = gm_by_id[body_id]
        if gm_name != body_name:
            raise PhysicalModelContractError(f"GM name mismatch for body {body_id}")
        recomputed_gm.append(_converted_gm(gm_decimal))
        relative_position = tuple(
            value - primary
            for value, primary in zip(position, sun_position, strict=True)
        )
        relative_velocity = tuple(
            value - primary
            for value, primary in zip(velocity, sun_velocity, strict=True)
        )
        recomputed_positions.append(relative_position)
        recomputed_velocities.append(
            tuple(
                component * JULIAN_YEAR_DAYS
                for component in relative_velocity
            )
        )
        if not _same_float(recomputed_gm[index], retained_gm[index]):
            raise PhysicalModelContractError(f"GM conversion mismatch at slot {index}")
        if not all(
            _same_float(left, right)
            for left, right in zip(
                recomputed_positions[index], retained_positions[index], strict=True
            )
        ):
            raise PhysicalModelContractError(
                f"position conversion mismatch at slot {index}"
            )
        if not all(
            _same_float(left, right)
            for left, right in zip(
                recomputed_velocities[index], retained_velocities[index], strict=True
            )
        ):
            raise PhysicalModelContractError(
                f"velocity conversion mismatch at slot {index}"
            )

    package_prefix = f"jxplanetx/{_DATA_DIRECTORY}"
    return KnownSolarSystemSourceModel(
        model_id=KNOWN_SYSTEM_MODEL_ID,
        epoch_jd_tdb=KNOWN_SYSTEM_EPOCH_JD_TDB,
        body_ids=KNOWN_SYSTEM_BODY_IDS,
        body_names=KNOWN_SYSTEM_BODY_NAMES,
        gm_AU3_per_julian_year2=tuple(retained_gm),
        source_positions_barycentric_icrf_AU=tuple(source_positions),
        source_velocities_barycentric_icrf_AU_per_day=tuple(source_velocities),
        positions_primary_relative_ecliptic_AU=tuple(retained_positions),
        velocities_primary_relative_ecliptic_AU_per_julian_year=tuple(
            retained_velocities
        ),
        source_artifact_path=f"{package_prefix}/{_SOURCE_ARTIFACT}",
        source_artifact_sha256=SOURCE_ARTIFACT_SHA256,
        gm_artifact_path=f"{package_prefix}/{_GM_ARTIFACT}",
        gm_artifact_sha256=GM_ARTIFACT_SHA256,
        derived_artifact_path=f"{package_prefix}/{_DERIVED_ARTIFACT}",
        derived_artifact_sha256=DERIVED_ARTIFACT_SHA256,
        manifest_artifact_path=f"{package_prefix}/{_MANIFEST_ARTIFACT}",
        manifest_artifact_sha256=MANIFEST_ARTIFACT_SHA256,
        upstream_horizons_vectors_sha256=UPSTREAM_HORIZONS_VECTORS_SHA256,
        upstream_horizons_manifest_sha256=UPSTREAM_HORIZONS_MANIFEST_SHA256,
        upstream_gm_de440_kernel_sha256=UPSTREAM_GM_DE440_KERNEL_SHA256,
    )


def build_matched_physical_arm_set(
    arm_set_id: str,
    known_source_model: KnownSolarSystemSourceModel,
    candidate_ensemble: CompactPlanetEnsemble,
    *,
    provenance_id: str,
    provenance_artifact_path: str,
    provenance_artifact_sha256: str,
) -> MatchedPhysicalArmSet:
    """Build exact known-prefix-plus-one-candidate physical arms."""

    if type(known_source_model) is not KnownSolarSystemSourceModel:
        raise PhysicalModelContractError(
            "known_source_model must be an exact KnownSolarSystemSourceModel"
        )
    if type(candidate_ensemble) is not CompactPlanetEnsemble:
        raise PhysicalModelContractError(
            "candidate_ensemble must be an exact CompactPlanetEnsemble"
        )
    states = tuple(
        _candidate_cartesian_state(member, known_source_model)
        for member in candidate_ensemble.members
    )
    return MatchedPhysicalArmSet(
        arm_set_id=arm_set_id,
        known_source_model=known_source_model,
        candidate_ensemble=candidate_ensemble,
        arm_known_source_prefixes=tuple(
            known_source_model for _ in candidate_ensemble.members
        ),
        candidate_positions_primary_relative_ecliptic_AU=tuple(
            position for position, _ in states
        ),
        candidate_velocities_primary_relative_ecliptic_AU_per_julian_year=tuple(
            velocity for _, velocity in states
        ),
        provenance_id=provenance_id,
        provenance_artifact_path=provenance_artifact_path,
        provenance_artifact_sha256=provenance_artifact_sha256,
    )


KNOWN_SOLAR_SYSTEM_V2 = load_known_solar_system_source_model()
KNOWN_SOLAR_SYSTEM_CONTENT_SHA256 = KNOWN_SOLAR_SYSTEM_V2.content_sha256
KNOWN_SOLAR_SYSTEM_SOURCE_STATE_SHA256 = (
    KNOWN_SOLAR_SYSTEM_V2.source_barycentric_state_sha256
)
KNOWN_SOLAR_SYSTEM_CONVERTED_STATE_SHA256 = (
    KNOWN_SOLAR_SYSTEM_V2.converted_state_sha256
)


__all__ = [
    "BINARY64_SEMANTICS",
    "COMPACT_ELEMENT_FRAME",
    "COMPACT_ENSEMBLE_SCHEMA",
    "COMPACT_MEMBER_SCHEMA",
    "CANDIDATE_STATE_POLICY",
    "DE440_EARTH_GM_AU3_PER_JULIAN_YEAR2",
    "DE440_EARTH_GM_KM3_S2_EXACT",
    "BoundCompactPlanetMember",
    "CompactPlanetEnsemble",
    "ENSEMBLE_WEIGHT_POLICY",
    "GM_ARTIFACT_SHA256",
    "KNOWN_SOLAR_SYSTEM_CONTENT_SHA256",
    "KNOWN_SOLAR_SYSTEM_CONVERTED_STATE_SHA256",
    "KNOWN_SOLAR_SYSTEM_SOURCE_STATE_SHA256",
    "KNOWN_SOLAR_SYSTEM_V2",
    "KNOWN_SYSTEM_BODY_IDS",
    "KNOWN_SYSTEM_BODY_NAMES",
    "KNOWN_SYSTEM_EPOCH_JD_TDB",
    "KNOWN_SYSTEM_SCHEMA",
    "KnownSolarSystemSourceModel",
    "MATCHED_ARM_SET_SCHEMA",
    "MANIFEST_ARTIFACT_SHA256",
    "MASS_GM_POLICY",
    "MatchedPhysicalArmSet",
    "PHYSICAL_CLAIM_FLAGS",
    "PHYSICAL_MODEL_NONCLAIMS",
    "PhysicalModelContractError",
    "README_ARTIFACT_SHA256",
    "SOURCE_ARTIFACT_SHA256",
    "DERIVED_ARTIFACT_SHA256",
    "build_matched_physical_arm_set",
    "compact_planet_gm_from_earth_masses",
    "load_known_solar_system_source_model",
]
