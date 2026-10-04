"""Closed source schedules for CUDA outer-Solar-System mechanism screens.

The CUDA massless-tracer engine already implements the force semantics needed
by the first mechanism screen: every positive-GM source mutually accelerates
every other source, and every massless tracer feels every source without
backreaction.  This module deliberately does *not* add callbacks, force
switches, prescribed ephemerides, or source insertion/removal.  It freezes four
constant-roster arms instead:

* the known outer-system control;
* the same control plus one circular synthetic distant-planet sensitivity arm;
* the same control plus one synthetic passing star; and
* the same control plus one synthetic unbound rogue planet.

In the flyby and rogue arms the appended source is active for the complete
integration.  The nominal passage window is an analysis annotation only.  The
actual trajectory must emerge from the mutual source integration, including
the perturber's backreaction on the known bodies.

The frozen DE441 control CSV used here is a current-epoch, reduced outer-system
state.  Its ``Sun`` row is an explicitly assumed inner-system monopole and the
state is not primordial.  Every object emitted here remains
``MODEL_OUTPUT / SCREENING_ONLY`` and supports no Planet Nine, stellar-flyby,
rogue-flyby, or 4.5-Gyr inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

import numpy as np


SCHEDULE_SCHEMA = "jxplanetx.cuda-closed-force-schedule.v1"
MATCHED_SET_SCHEMA = "jxplanetx.cuda-matched-mechanism-schedules.v1"
SOURCE_STATE_SCHEMA = "jxplanetx.cuda-mechanism-source-state.v1"

SCHEDULE_HASH_DOMAIN = b"jxplanetx.cuda-closed-force-schedule.v1\x00"
MATCHED_SET_HASH_DOMAIN = b"jxplanetx.cuda-matched-mechanism-schedules.v1\x00"
SOURCE_STATE_HASH_DOMAIN = b"jxplanetx.cuda-mechanism-source-state.v1\x00"
SOURCE_PREFIX_HASH_DOMAIN = b"jxplanetx.cuda-mechanism-source-prefix.v1\x00"

SCIENTIFIC_CLASSIFICATION = "MODEL_OUTPUT"
PROJECT_SCOPE = "SCREENING_ONLY"
INITIAL_CONDITION_SCOPE = "CURRENT_EPOCH_INTERVENTION_SCREEN_NOT_PRIMORDIAL"

CONTROL_ARM = "KNOWN_PLANETS_ONLY"
DISTANT_PLANET_ARM = "CIRCULAR_DISTANT_PLANET_SENSITIVITY"
STELLAR_FLYBY_ARM = "STELLAR_FLYBY"
TEMPORARY_ROGUE_ARM = "UNBOUND_ROGUE_FLYBY"
ARM_KINDS = (
    CONTROL_ARM,
    DISTANT_PLANET_ARM,
    STELLAR_FLYBY_ARM,
    TEMPORARY_ROGUE_ARM,
)

KNOWN_SOURCE_ROLE = "KNOWN_OUTER_SYSTEM"
DISTANT_PLANET_ROLE = "SYNTHETIC_DISTANT_PLANET"
STELLAR_FLYBY_ROLE = "SYNTHETIC_FLYBY_STAR"
TEMPORARY_ROGUE_ROLE = "SYNTHETIC_UNBOUND_ROGUE_FLYBY"
PERTURBER_ROLE_FOR_ARM = MappingProxyType(
    {
        DISTANT_PLANET_ARM: DISTANT_PLANET_ROLE,
        STELLAR_FLYBY_ARM: STELLAR_FLYBY_ROLE,
        TEMPORARY_ROGUE_ARM: TEMPORARY_ROGUE_ROLE,
    }
)

MASSIVE_FORCE_SEMANTICS = "FULL_MUTUAL_NEWTONIAN_ALL_ACTIVE_SOURCES"
TRACER_FORCE_SEMANTICS = "DIRECT_NEWTONIAN_FROM_ALL_ACTIVE_SOURCES"
TRACER_BACKREACTION_SEMANTICS = "EXACTLY_MASSLESS_NO_BACKREACTION"
SOURCE_ACTIVITY_SEMANTICS = "FIXED_ROSTER_FULL_HORIZON_NO_FORCE_SWITCHES"
SOURCE_INTEGRATION_SEMANTICS = "FIXED_STEP_INERTIAL_KICK_DRIFT_KICK"
PERTURBER_TRAJECTORY_SEMANTICS = "DYNAMIC_MUTUAL_SOURCE_NOT_PRESCRIBED"

DE441_CONTROL_ARTIFACT_ID = "jx.de441-population.outer-control-state.v1"
DE441_CONTROL_STATE_SHA256 = (
    "edfbfd50ca29b28b46fe43cae8cb99bd73a9303a5f2e5babf7bf9ad2a820ff14"
)
DE441_CONTROL_SOURCE_IDS = (
    "Sun",
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune",
    "Pluto",
)
DE441_CONTROL_EPOCH_JD_TDB = 2_461_200.5
DE441_CONTROL_FRAME = "J2000_ECLIPTIC_ROTATED_FROM_HORIZONS_ICRF"
DE441_CONTROL_ORIGIN = "INNER_SYSTEM_MONOPOLE_CENTER_AT_T0"
DE441_CONTROL_CLASSIFICATION = (
    "RECONSTRUCTED_DE441_BACKBONE_PLUS_ASSUMED_INNER_SYSTEM_MONOPOLE"
)

SOLAR_GM_AU3_PER_YEAR2 = 4.0 * math.pi * math.pi
EARTH_TO_SUN_MASS_RATIO = 3.0034896149156e-6

RESULT_CLAIMS = MappingProxyType(
    {
        "four_point_five_gyr_accuracy_qualified": False,
        "planet_nine_detected": False,
        "planet_nine_excluded": False,
        "stellar_flyby_inferred": False,
        "unbound_rogue_flyby_inferred": False,
        "primordial_history_reconstructed": False,
        "rubin_data_fitted": False,
    }
)

UNSUPPORTED_CAPABILITIES = (
    "ARBITRARY_FORCE_CALLBACKS",
    "SOURCE_ACTIVATION_DEACTIVATION_OR_GM_RAMP",
    "PRESCRIBED_PERTURBER_EPHEMERIS",
    "MASSIVE_SOURCE_COLLISION_MERGER_OR_REMOVAL",
    "MASSIVE_SOURCE_ADAPTIVE_ENCOUNTER_ROUTING",
    "BINARY_OR_MULTIPLE_SIMULTANEOUS_PERTURBERS",
    "GENERAL_RELATIVITY",
    "GALACTIC_TIDE_OR_STELLAR_HISTORY_ENSEMBLE",
    "PRIMORDIAL_HISTORY_OR_4P5_GYR_QUALIFICATION",
)

_STATE_COLUMNS = ("index", "name", "mass", "x", "y", "z", "vx", "vy", "vz")
_SHA256_HEX = frozenset("0123456789abcdef")


class MechanismScheduleError(ValueError):
    """A closed mechanism schedule or source state violates its contract."""


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise MechanismScheduleError(f"{label} must be a nonempty trimmed string")
    return value


def _sha256_text(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
        or value == "0" * 64
    ):
        raise MechanismScheduleError(
            f"{label} must be a nonzero lowercase SHA-256 digest"
        )
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MechanismScheduleError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise MechanismScheduleError(f"{label} must be a finite number")
    return result


def _positive(value: object, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise MechanismScheduleError(f"{label} must be positive")
    return result


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise MechanismScheduleError(f"{label} must be a nonnegative integer")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise MechanismScheduleError(f"{label} must be a positive integer")
    return value


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, RecursionError) as exc:
        raise MechanismScheduleError("manifest is not canonical-JSON encodable") from exc


def _domain_hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise MechanismScheduleError(f"cannot read source artifact: {path}") from exc
    return digest.hexdigest()


def _float_hex(value: object, label: str) -> str:
    return _finite(value, label).hex()


def _vec3(value: object, label: str) -> tuple[float, float, float]:
    if type(value) is not tuple or len(value) != 3:
        raise MechanismScheduleError(f"{label} must be an immutable length-three tuple")
    checked = tuple(
        _finite(item, f"{label}[{index}]") for index, item in enumerate(value)
    )
    return checked[0], checked[1], checked[2]


def _readonly_f64(value: object, label: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.array(value, dtype="<f8", order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MechanismScheduleError(f"{label} cannot be converted to float64") from exc
    if array.shape != shape:
        raise MechanismScheduleError(
            f"{label} must have shape {shape}; received {array.shape}"
        )
    if not np.isfinite(array).all():
        raise MechanismScheduleError(f"{label} must contain only finite values")
    array.setflags(write=False)
    return array


def _array_record(array: np.ndarray) -> dict[str, object]:
    value = np.ascontiguousarray(array, dtype="<f8")
    return {
        "dtype": "<f8",
        "shape": list(value.shape),
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


@dataclass(frozen=True, slots=True, eq=False)
class MechanismSourceState:
    """One immutable CPU source-state artifact for a closed schedule arm."""

    snapshot_id: str
    epoch_jd_tdb: float
    source_ids: tuple[str, ...]
    source_roles: tuple[str, ...]
    primary_source_id: str
    positions_AU: np.ndarray
    velocities_AU_per_year: np.ndarray
    gm_AU3_per_year2: np.ndarray
    base_source_count: int
    base_artifact_id: str
    base_artifact_sha256: str
    state_classification: str
    frame: str = DE441_CONTROL_FRAME
    origin: str = DE441_CONTROL_ORIGIN
    time_scale: str = "TDB"
    unit_system_id: str = "AU-YEAR"
    scientific_classification: str = SCIENTIFIC_CLASSIFICATION
    project_scope: str = PROJECT_SCOPE
    initial_condition_scope: str = INITIAL_CONDITION_SCOPE
    primordial_initial_conditions: bool = False

    def __post_init__(self) -> None:
        _text(self.snapshot_id, "snapshot_id")
        _finite(self.epoch_jd_tdb, "epoch_jd_tdb")
        if type(self.source_ids) is not tuple or not self.source_ids:
            raise MechanismScheduleError("source_ids must be a nonempty tuple")
        if any(type(value) is not str or not value for value in self.source_ids):
            raise MechanismScheduleError("source_ids must contain nonempty strings")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise MechanismScheduleError("source_ids must be unique")
        if type(self.source_roles) is not tuple or len(self.source_roles) != len(
            self.source_ids
        ):
            raise MechanismScheduleError("source_roles must align with source_ids")
        for index, role in enumerate(self.source_roles):
            _text(role, f"source_roles[{index}]")
        _text(self.primary_source_id, "primary_source_id")
        if self.primary_source_id != self.source_ids[0]:
            raise MechanismScheduleError("the primary source must occupy slot zero")
        _positive_int(self.base_source_count, "base_source_count")
        if self.base_source_count > len(self.source_ids):
            raise MechanismScheduleError("base_source_count exceeds the source roster")
        if any(
            role != KNOWN_SOURCE_ROLE
            for role in self.source_roles[: self.base_source_count]
        ):
            raise MechanismScheduleError("the base source prefix has an invalid role")
        count = len(self.source_ids)
        for label, array, shape in (
            ("positions_AU", self.positions_AU, (count, 3)),
            ("velocities_AU_per_year", self.velocities_AU_per_year, (count, 3)),
            ("gm_AU3_per_year2", self.gm_AU3_per_year2, (count,)),
        ):
            if type(array) is not np.ndarray:
                raise MechanismScheduleError(f"{label} must be an exact numpy.ndarray")
            if array.dtype.str not in {"<f8", "=f8"} or array.shape != shape:
                raise MechanismScheduleError(
                    f"{label} must be C-contiguous little-endian float64 with shape {shape}"
                )
            if (
                not array.flags.c_contiguous
                or not array.flags.owndata
                or array.flags.writeable
            ):
                raise MechanismScheduleError(
                    f"{label} must own C-contiguous read-only storage"
                )
            if not np.isfinite(array).all():
                raise MechanismScheduleError(f"{label} must contain only finite values")
        if np.any(self.gm_AU3_per_year2 <= 0.0):
            raise MechanismScheduleError("every source GM must be strictly positive")
        _text(self.base_artifact_id, "base_artifact_id")
        _sha256_text(self.base_artifact_sha256, "base_artifact_sha256")
        _text(self.state_classification, "state_classification")
        for label in ("frame", "origin", "time_scale", "unit_system_id"):
            _text(getattr(self, label), label)
        if self.unit_system_id != "AU-YEAR":
            raise MechanismScheduleError("unit_system_id must equal 'AU-YEAR'")
        if self.scientific_classification != SCIENTIFIC_CLASSIFICATION:
            raise MechanismScheduleError("scientific classification cannot be widened")
        if self.project_scope != PROJECT_SCOPE:
            raise MechanismScheduleError("project scope cannot be widened")
        if self.initial_condition_scope != INITIAL_CONDITION_SCOPE:
            raise MechanismScheduleError("initial-condition scope cannot be widened")
        if (
            type(self.primordial_initial_conditions) is not bool
            or self.primordial_initial_conditions
        ):
            raise MechanismScheduleError("this state must remain explicitly nonprimordial")

    def canonical_metadata(self) -> dict[str, object]:
        return {
            "schema": SOURCE_STATE_SCHEMA,
            "snapshot_id": self.snapshot_id,
            "epoch_jd_tdb_hex": self.epoch_jd_tdb.hex(),
            "source_ids": list(self.source_ids),
            "source_roles": list(self.source_roles),
            "primary_source_id": self.primary_source_id,
            "base_source_count": self.base_source_count,
            "base_artifact_id": self.base_artifact_id,
            "base_artifact_sha256": self.base_artifact_sha256,
            "state_classification": self.state_classification,
            "frame": self.frame,
            "origin": self.origin,
            "time_scale": self.time_scale,
            "unit_system_id": self.unit_system_id,
            "scientific_classification": self.scientific_classification,
            "project_scope": self.project_scope,
            "initial_condition_scope": self.initial_condition_scope,
            "primordial_initial_conditions": self.primordial_initial_conditions,
            "arrays": {
                "positions_AU": _array_record(self.positions_AU),
                "velocities_AU_per_year": _array_record(
                    self.velocities_AU_per_year
                ),
                "gm_AU3_per_year2": _array_record(self.gm_AU3_per_year2),
            },
        }

    @property
    def state_sha256(self) -> str:
        return _domain_hash(SOURCE_STATE_HASH_DOMAIN, self.canonical_metadata())

    @property
    def base_prefix_sha256(self) -> str:
        count = self.base_source_count
        record = {
            "epoch_jd_tdb_hex": self.epoch_jd_tdb.hex(),
            "source_ids": list(self.source_ids[:count]),
            "source_roles": list(self.source_roles[:count]),
            "primary_source_id": self.primary_source_id,
            "frame": self.frame,
            "origin": self.origin,
            "time_scale": self.time_scale,
            "unit_system_id": self.unit_system_id,
            "base_artifact_id": self.base_artifact_id,
            "base_artifact_sha256": self.base_artifact_sha256,
            "positions_AU": _array_record(self.positions_AU[:count]),
            "velocities_AU_per_year": _array_record(
                self.velocities_AU_per_year[:count]
            ),
            "gm_AU3_per_year2": _array_record(self.gm_AU3_per_year2[:count]),
        }
        return _domain_hash(SOURCE_PREFIX_HASH_DOMAIN, record)


@dataclass(frozen=True, slots=True)
class SourceActivityInterval:
    """One source's full-horizon active interval on the integer lattice."""

    source_id: str
    start_global_step: int
    end_global_step_exclusive: int

    def __post_init__(self) -> None:
        _text(self.source_id, "activity source_id")
        _nonnegative_int(self.start_global_step, "activity start_global_step")
        _positive_int(
            self.end_global_step_exclusive,
            "activity end_global_step_exclusive",
        )
        if self.start_global_step >= self.end_global_step_exclusive:
            raise MechanismScheduleError("activity interval must be nonempty")

    def canonical_record(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "start_global_step": self.start_global_step,
            "end_global_step_exclusive": self.end_global_step_exclusive,
        }


@dataclass(frozen=True, slots=True)
class DiagnosticWindow:
    """A lattice-aligned analysis annotation with no effect on dynamics."""

    event_kind: str
    nominal_event_global_step: int
    start_global_step: int
    end_global_step_exclusive: int
    dynamics_effect: str = "NONE"

    def __post_init__(self) -> None:
        _text(self.event_kind, "event_kind")
        nominal = _nonnegative_int(
            self.nominal_event_global_step,
            "nominal_event_global_step",
        )
        start = _nonnegative_int(self.start_global_step, "window start_global_step")
        end = _positive_int(
            self.end_global_step_exclusive,
            "window end_global_step_exclusive",
        )
        if not start <= nominal < end:
            raise MechanismScheduleError(
                "diagnostic window must contain its nominal event step"
            )
        if self.dynamics_effect != "NONE":
            raise MechanismScheduleError(
                "a diagnostic window cannot activate, disable, or modify forces"
            )

    def canonical_record(self) -> dict[str, object]:
        return {
            "event_kind": self.event_kind,
            "nominal_event_global_step": self.nominal_event_global_step,
            "start_global_step": self.start_global_step,
            "end_global_step_exclusive": self.end_global_step_exclusive,
            "dynamics_effect": self.dynamics_effect,
        }


@dataclass(frozen=True, slots=True)
class SyntheticPerturber:
    """One explicit Cartesian synthetic assumption appended to a base state."""

    source_id: str
    role: str
    gm_AU3_per_year2: float
    initial_position_AU: tuple[float, float, float]
    initial_velocity_AU_per_year: tuple[float, float, float]
    synthetic_model_id: str
    parameter_record: tuple[tuple[str, str], ...]
    provenance_classification: str = "SYNTHETIC_ASSUMPTION_NOT_OBSERVATIONAL_FIT"
    trajectory_semantics: str = PERTURBER_TRAJECTORY_SEMANTICS

    def __post_init__(self) -> None:
        _text(self.source_id, "perturber source_id")
        if self.role not in PERTURBER_ROLE_FOR_ARM.values():
            raise MechanismScheduleError("perturber role is outside the closed roster")
        _positive(self.gm_AU3_per_year2, "perturber GM")
        _vec3(self.initial_position_AU, "perturber initial_position_AU")
        _vec3(self.initial_velocity_AU_per_year, "perturber initial_velocity_AU_per_year")
        _text(self.synthetic_model_id, "synthetic_model_id")
        if type(self.parameter_record) is not tuple:
            raise MechanismScheduleError("parameter_record must be an immutable tuple")
        keys: list[str] = []
        for index, item in enumerate(self.parameter_record):
            if type(item) is not tuple or len(item) != 2:
                raise MechanismScheduleError(
                    f"parameter_record[{index}] must be a (name, value) tuple"
                )
            key, value = item
            keys.append(_text(key, f"parameter_record[{index}].name"))
            _text(value, f"parameter_record[{index}].value")
        if keys != sorted(keys) or len(set(keys)) != len(keys):
            raise MechanismScheduleError(
                "parameter_record names must be unique and sorted"
            )
        if self.provenance_classification != "SYNTHETIC_ASSUMPTION_NOT_OBSERVATIONAL_FIT":
            raise MechanismScheduleError("perturber provenance cannot be widened")
        if self.trajectory_semantics != PERTURBER_TRAJECTORY_SEMANTICS:
            raise MechanismScheduleError("perturber trajectory cannot be prescribed")

    def canonical_record(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "role": self.role,
            "gm_AU3_per_year2_hex": self.gm_AU3_per_year2.hex(),
            "initial_position_AU_hex": [value.hex() for value in self.initial_position_AU],
            "initial_velocity_AU_per_year_hex": [
                value.hex() for value in self.initial_velocity_AU_per_year
            ],
            "synthetic_model_id": self.synthetic_model_id,
            "parameter_record": [list(item) for item in self.parameter_record],
            "provenance_classification": self.provenance_classification,
            "trajectory_semantics": self.trajectory_semantics,
        }


@dataclass(frozen=True, slots=True)
class ClosedForceSchedule:
    """A callback-free, constant-roster force schedule for one matched arm."""

    schedule_id: str
    matched_set_id: str
    arm_kind: str
    step_years: float
    total_steps: int
    source_ids: tuple[str, ...]
    base_source_ids: tuple[str, ...]
    primary_source_id: str
    source_state_sha256: str
    base_source_prefix_sha256: str
    source_activity: tuple[SourceActivityInterval, ...]
    perturber: SyntheticPerturber | None
    diagnostic_window: DiagnosticWindow | None
    force_transition_steps: tuple[int, ...] = ()
    massive_force_semantics: str = MASSIVE_FORCE_SEMANTICS
    tracer_force_semantics: str = TRACER_FORCE_SEMANTICS
    tracer_backreaction_semantics: str = TRACER_BACKREACTION_SEMANTICS
    source_activity_semantics: str = SOURCE_ACTIVITY_SEMANTICS
    source_integration_semantics: str = SOURCE_INTEGRATION_SEMANTICS
    scientific_classification: str = SCIENTIFIC_CLASSIFICATION
    project_scope: str = PROJECT_SCOPE
    initial_condition_scope: str = INITIAL_CONDITION_SCOPE
    primordial_initial_conditions: bool = False

    def __post_init__(self) -> None:
        _text(self.schedule_id, "schedule_id")
        _text(self.matched_set_id, "matched_set_id")
        if self.arm_kind not in ARM_KINDS:
            raise MechanismScheduleError("arm_kind is outside the closed four-arm set")
        _positive(self.step_years, "step_years")
        _positive_int(self.total_steps, "total_steps")
        for label, values in (
            ("source_ids", self.source_ids),
            ("base_source_ids", self.base_source_ids),
        ):
            if type(values) is not tuple or not values:
                raise MechanismScheduleError(f"{label} must be a nonempty tuple")
            if len(set(values)) != len(values):
                raise MechanismScheduleError(f"{label} must contain unique identities")
            for index, value in enumerate(values):
                _text(value, f"{label}[{index}]")
        if self.source_ids[: len(self.base_source_ids)] != self.base_source_ids:
            raise MechanismScheduleError("the base source roster must be an exact prefix")
        if self.primary_source_id != self.source_ids[0]:
            raise MechanismScheduleError("primary_source_id must name source slot zero")
        _sha256_text(self.source_state_sha256, "source_state_sha256")
        _sha256_text(self.base_source_prefix_sha256, "base_source_prefix_sha256")
        if type(self.source_activity) is not tuple or len(self.source_activity) != len(
            self.source_ids
        ):
            raise MechanismScheduleError("every source requires one activity interval")
        expected_activity = tuple(
            SourceActivityInterval(source_id, 0, self.total_steps)
            for source_id in self.source_ids
        )
        if self.source_activity != expected_activity:
            raise MechanismScheduleError(
                "every source must remain active over the complete modeled horizon"
            )
        if self.arm_kind == CONTROL_ARM:
            if self.perturber is not None or self.diagnostic_window is not None:
                raise MechanismScheduleError("the control arm cannot contain a perturber")
            if self.source_ids != self.base_source_ids:
                raise MechanismScheduleError("the control roster must equal the base roster")
        else:
            if type(self.perturber) is not SyntheticPerturber:
                raise MechanismScheduleError("candidate arms require one perturber")
            if len(self.source_ids) != len(self.base_source_ids) + 1:
                raise MechanismScheduleError(
                    "candidate arms must append exactly one perturber"
                )
            if self.source_ids[-1] != self.perturber.source_id:
                raise MechanismScheduleError("the perturber must occupy the final source slot")
            if self.perturber.role != PERTURBER_ROLE_FOR_ARM[self.arm_kind]:
                raise MechanismScheduleError("perturber role does not match arm_kind")
            if self.arm_kind in (STELLAR_FLYBY_ARM, TEMPORARY_ROGUE_ARM):
                if type(self.diagnostic_window) is not DiagnosticWindow:
                    raise MechanismScheduleError(
                        "passing-source arms require a diagnostic window"
                    )
                expected_event_kind = {
                    STELLAR_FLYBY_ARM: "NOMINAL_LINEAR_STELLAR_PASSAGE_EPOCH",
                    TEMPORARY_ROGUE_ARM: "NOMINAL_LINEAR_ROGUE_PASSAGE_EPOCH",
                }[self.arm_kind]
                if self.diagnostic_window.event_kind != expected_event_kind:
                    raise MechanismScheduleError(
                        "diagnostic event kind does not match arm_kind"
                    )
                if self.diagnostic_window.end_global_step_exclusive > self.total_steps:
                    raise MechanismScheduleError(
                        "diagnostic window exceeds the modeled horizon"
                    )
            elif self.diagnostic_window is not None:
                raise MechanismScheduleError(
                    "the persistent-planet arm has no force or passage window"
                )
        if type(self.force_transition_steps) is not tuple or self.force_transition_steps:
            raise MechanismScheduleError("M2 schedules forbid force-transition steps")
        fixed = {
            "massive_force_semantics": MASSIVE_FORCE_SEMANTICS,
            "tracer_force_semantics": TRACER_FORCE_SEMANTICS,
            "tracer_backreaction_semantics": TRACER_BACKREACTION_SEMANTICS,
            "source_activity_semantics": SOURCE_ACTIVITY_SEMANTICS,
            "source_integration_semantics": SOURCE_INTEGRATION_SEMANTICS,
            "scientific_classification": SCIENTIFIC_CLASSIFICATION,
            "project_scope": PROJECT_SCOPE,
            "initial_condition_scope": INITIAL_CONDITION_SCOPE,
        }
        for label, expected in fixed.items():
            if getattr(self, label) != expected:
                raise MechanismScheduleError(f"{label} cannot be widened or changed")
        if (
            type(self.primordial_initial_conditions) is not bool
            or self.primordial_initial_conditions
        ):
            raise MechanismScheduleError("the schedule must remain explicitly nonprimordial")

    @property
    def end_global_step_exclusive(self) -> int:
        return self.total_steps

    def canonical_record(self) -> dict[str, object]:
        return {
            "schema": SCHEDULE_SCHEMA,
            "schedule_id": self.schedule_id,
            "matched_set_id": self.matched_set_id,
            "arm_kind": self.arm_kind,
            "clock": {
                "start_global_step": 0,
                "end_global_step_exclusive": self.total_steps,
                "step_years_hex": self.step_years.hex(),
                "direction": "FORWARD",
            },
            "source_ids": list(self.source_ids),
            "base_source_ids": list(self.base_source_ids),
            "primary_source_id": self.primary_source_id,
            "source_state_sha256": self.source_state_sha256,
            "base_source_prefix_sha256": self.base_source_prefix_sha256,
            "source_activity": [row.canonical_record() for row in self.source_activity],
            "perturber": (
                None if self.perturber is None else self.perturber.canonical_record()
            ),
            "diagnostic_window": (
                None
                if self.diagnostic_window is None
                else self.diagnostic_window.canonical_record()
            ),
            "force_transition_steps": list(self.force_transition_steps),
            "force_semantics": {
                "massive_sources": self.massive_force_semantics,
                "tracers": self.tracer_force_semantics,
                "tracer_backreaction": self.tracer_backreaction_semantics,
                "source_activity": self.source_activity_semantics,
                "source_integration": self.source_integration_semantics,
            },
            "scientific_classification": self.scientific_classification,
            "project_scope": self.project_scope,
            "initial_condition_scope": self.initial_condition_scope,
            "primordial_initial_conditions": self.primordial_initial_conditions,
            "claims": dict(RESULT_CLAIMS),
            "unsupported_capabilities": list(UNSUPPORTED_CAPABILITIES),
        }

    @property
    def schedule_sha256(self) -> str:
        return _domain_hash(SCHEDULE_HASH_DOMAIN, self.canonical_record())

    @property
    def manifest_bytes(self) -> bytes:
        return _canonical_json(self.canonical_record())


@dataclass(frozen=True, slots=True, eq=False)
class MechanismArm:
    """One closed schedule bound to the exact source state it describes."""

    schedule: ClosedForceSchedule
    initial_state: MechanismSourceState

    def __post_init__(self) -> None:
        if type(self.schedule) is not ClosedForceSchedule:
            raise MechanismScheduleError("schedule must be ClosedForceSchedule")
        if type(self.initial_state) is not MechanismSourceState:
            raise MechanismScheduleError("initial_state must be MechanismSourceState")
        if self.schedule.source_ids != self.initial_state.source_ids:
            raise MechanismScheduleError("schedule/source-state rosters differ")
        if self.schedule.primary_source_id != self.initial_state.primary_source_id:
            raise MechanismScheduleError("schedule/source-state primary differs")
        if self.schedule.source_state_sha256 != self.initial_state.state_sha256:
            raise MechanismScheduleError("schedule/source-state semantic hash differs")
        if self.schedule.base_source_prefix_sha256 != self.initial_state.base_prefix_sha256:
            raise MechanismScheduleError("schedule/source-state base prefix differs")
        perturber = self.schedule.perturber
        if perturber is not None:
            if self.initial_state.source_roles[-1] != perturber.role:
                raise MechanismScheduleError("perturber role differs from source state")
            if self.initial_state.gm_AU3_per_year2[-1:].tobytes() != np.asarray(
                [perturber.gm_AU3_per_year2], dtype="<f8"
            ).tobytes():
                raise MechanismScheduleError("perturber GM differs from source state")
            expected_position = np.asarray(perturber.initial_position_AU, dtype="<f8")
            expected_velocity = np.asarray(
                perturber.initial_velocity_AU_per_year, dtype="<f8"
            )
            if (
                self.initial_state.positions_AU[-1].tobytes()
                != expected_position.tobytes()
                or self.initial_state.velocities_AU_per_year[-1].tobytes()
                != expected_velocity.tobytes()
            ):
                raise MechanismScheduleError(
                    "perturber Cartesian state differs from source state"
                )


@dataclass(frozen=True, slots=True, eq=False)
class MatchedMechanismScheduleSet:
    """The exact four-arm, same-base current-epoch screening design."""

    matched_set_id: str
    control: MechanismArm
    persistent_distant_planet: MechanismArm
    stellar_flyby: MechanismArm
    temporary_rogue: MechanismArm

    def __post_init__(self) -> None:
        _text(self.matched_set_id, "matched_set_id")
        arms = self.arms
        if any(type(arm) is not MechanismArm for arm in arms):
            raise MechanismScheduleError("all matched entries must be MechanismArm")
        observed = tuple(arm.schedule.arm_kind for arm in arms)
        if observed != ARM_KINDS:
            raise MechanismScheduleError(
                f"matched arm order must be exactly {ARM_KINDS!r}"
            )
        if any(arm.schedule.matched_set_id != self.matched_set_id for arm in arms):
            raise MechanismScheduleError("an arm is bound to a different matched set")
        control_state = self.control.initial_state
        base_count = control_state.base_source_count
        if len(control_state.source_ids) != base_count:
            raise MechanismScheduleError("control state must contain only base sources")
        reference_clock = (
            self.control.schedule.step_years,
            self.control.schedule.total_steps,
        )
        reference_prefix = control_state.base_prefix_sha256
        for arm in arms:
            if (
                arm.schedule.step_years,
                arm.schedule.total_steps,
            ) != reference_clock:
                raise MechanismScheduleError("matched arms must share one lattice clock")
            if arm.initial_state.base_prefix_sha256 != reference_prefix:
                raise MechanismScheduleError("matched arms have different base prefixes")
            if not _base_prefix_bytes_equal(control_state, arm.initial_state):
                raise MechanismScheduleError(
                    "matched arms must retain byte-identical base source arrays"
                )

    @property
    def arms(self) -> tuple[MechanismArm, ...]:
        return (
            self.control,
            self.persistent_distant_planet,
            self.stellar_flyby,
            self.temporary_rogue,
        )

    def canonical_record(self) -> dict[str, object]:
        return {
            "schema": MATCHED_SET_SCHEMA,
            "matched_set_id": self.matched_set_id,
            "base_source_prefix_sha256": self.control.initial_state.base_prefix_sha256,
            "base_artifact_id": self.control.initial_state.base_artifact_id,
            "base_artifact_sha256": self.control.initial_state.base_artifact_sha256,
            "arm_schedule_sha256": {
                arm.schedule.arm_kind: arm.schedule.schedule_sha256
                for arm in self.arms
            },
            "arm_initial_state_sha256": {
                arm.schedule.arm_kind: arm.initial_state.state_sha256
                for arm in self.arms
            },
            "scientific_classification": SCIENTIFIC_CLASSIFICATION,
            "project_scope": PROJECT_SCOPE,
            "initial_condition_scope": INITIAL_CONDITION_SCOPE,
            "primordial_initial_conditions": False,
            "claims": dict(RESULT_CLAIMS),
        }

    @property
    def matched_set_sha256(self) -> str:
        return _domain_hash(MATCHED_SET_HASH_DOMAIN, self.canonical_record())

    @property
    def manifest_bytes(self) -> bytes:
        return _canonical_json(self.canonical_record())


def _base_prefix_bytes_equal(
    left: MechanismSourceState, right: MechanismSourceState
) -> bool:
    count = left.base_source_count
    if right.base_source_count != count:
        return False
    if (
        left.source_ids[:count] != right.source_ids[:count]
        or left.source_roles[:count] != right.source_roles[:count]
    ):
        return False
    return all(
        left_array[:count].tobytes(order="C")
        == right_array[:count].tobytes(order="C")
        for left_array, right_array in (
            (left.positions_AU, right.positions_AU),
            (left.velocities_AU_per_year, right.velocities_AU_per_year),
            (left.gm_AU3_per_year2, right.gm_AU3_per_year2),
        )
    )


def load_frozen_de441_outer_control(path: str | Path) -> MechanismSourceState:
    """Load only the exact frozen current-epoch outer-system control CSV."""

    source = Path(path)
    observed_sha256 = _sha256_file(source)
    if observed_sha256 != DE441_CONTROL_STATE_SHA256:
        raise MechanismScheduleError(
            "DE441 control-state artifact SHA-256 does not match the frozen input"
        )
    try:
        with source.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != _STATE_COLUMNS:
                raise MechanismScheduleError(
                    f"DE441 control CSV must have columns {_STATE_COLUMNS!r}"
                )
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise MechanismScheduleError("DE441 control CSV is unreadable") from exc
    if len(rows) != len(DE441_CONTROL_SOURCE_IDS):
        raise MechanismScheduleError("DE441 control CSV has the wrong source count")
    positions: list[tuple[float, float, float]] = []
    velocities: list[tuple[float, float, float]] = []
    gm: list[float] = []
    observed_ids: list[str] = []
    for expected_index, row in enumerate(rows):
        try:
            index = int(row["index"])
            name = row["name"]
            local_gm = float(row["mass"])
            position = tuple(float(row[key]) for key in ("x", "y", "z"))
            velocity = tuple(float(row[key]) for key in ("vx", "vy", "vz"))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise MechanismScheduleError("DE441 control CSV contains an invalid row") from exc
        if index != expected_index:
            raise MechanismScheduleError("DE441 control indices must be contiguous")
        observed_ids.append(name)
        gm.append(_positive(local_gm, f"source GM {name}"))
        positions.append(_vec3(position, f"source position {name}"))
        velocities.append(_vec3(velocity, f"source velocity {name}"))
    if tuple(observed_ids) != DE441_CONTROL_SOURCE_IDS:
        raise MechanismScheduleError(
            "DE441 control source roster differs from the frozen outer system"
        )
    return MechanismSourceState(
        snapshot_id="de441-current-epoch-outer-control-jd2461200.5",
        epoch_jd_tdb=DE441_CONTROL_EPOCH_JD_TDB,
        source_ids=DE441_CONTROL_SOURCE_IDS,
        source_roles=(KNOWN_SOURCE_ROLE,) * len(DE441_CONTROL_SOURCE_IDS),
        primary_source_id=DE441_CONTROL_SOURCE_IDS[0],
        positions_AU=_readonly_f64(
            positions,
            "positions_AU",
            (len(DE441_CONTROL_SOURCE_IDS), 3),
        ),
        velocities_AU_per_year=_readonly_f64(
            velocities,
            "velocities_AU_per_year",
            (len(DE441_CONTROL_SOURCE_IDS), 3),
        ),
        gm_AU3_per_year2=_readonly_f64(
            gm,
            "gm_AU3_per_year2",
            (len(DE441_CONTROL_SOURCE_IDS),),
        ),
        base_source_count=len(DE441_CONTROL_SOURCE_IDS),
        base_artifact_id=DE441_CONTROL_ARTIFACT_ID,
        base_artifact_sha256=observed_sha256,
        state_classification=DE441_CONTROL_CLASSIFICATION,
    )


def _parameters(**values: object) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    for key, value in values.items():
        if isinstance(value, bool):
            encoded = "true" if value else "false"
        elif isinstance(value, int):
            encoded = str(value)
        elif isinstance(value, float):
            encoded = _float_hex(value, key)
        elif isinstance(value, str):
            encoded = _text(value, key)
        else:
            raise MechanismScheduleError(f"unsupported synthetic parameter {key}")
        rows.append((_text(key, "parameter name"), encoded))
    return tuple(sorted(rows))


def build_synthetic_persistent_distant_planet(
    base_state: MechanismSourceState,
    *,
    source_id: str = "P9_SYNTHETIC",
    mass_earth: float = 5.0,
    orbital_radius_AU: float = 500.0,
    inclination_deg: float = 20.0,
    phase_deg: float = 0.0,
) -> SyntheticPerturber:
    """Build a circular synthetic distant-planet Cartesian assumption."""

    if type(base_state) is not MechanismSourceState:
        raise MechanismScheduleError("base_state must be MechanismSourceState")
    _text(source_id, "source_id")
    if source_id in base_state.source_ids:
        raise MechanismScheduleError("synthetic source_id duplicates a base source")
    mass = _positive(mass_earth, "mass_earth")
    radius = _positive(orbital_radius_AU, "orbital_radius_AU")
    inclination = math.radians(_finite(inclination_deg, "inclination_deg"))
    phase = math.radians(_finite(phase_deg, "phase_deg"))
    gm = SOLAR_GM_AU3_PER_YEAR2 * EARTH_TO_SUN_MASS_RATIO * mass
    speed = math.sqrt((float(base_state.gm_AU3_per_year2[0]) + gm) / radius)
    cos_phase, sin_phase = math.cos(phase), math.sin(phase)
    cos_i, sin_i = math.cos(inclination), math.sin(inclination)
    relative_position = np.asarray(
        (radius * cos_phase, radius * sin_phase * cos_i, radius * sin_phase * sin_i),
        dtype=np.float64,
    )
    relative_velocity = np.asarray(
        (-speed * sin_phase, speed * cos_phase * cos_i, speed * cos_phase * sin_i),
        dtype=np.float64,
    )
    absolute_position = base_state.positions_AU[0] + relative_position
    absolute_velocity = base_state.velocities_AU_per_year[0] + relative_velocity
    return SyntheticPerturber(
        source_id=source_id,
        role=DISTANT_PLANET_ROLE,
        gm_AU3_per_year2=gm,
        initial_position_AU=tuple(float(value) for value in absolute_position),
        initial_velocity_AU_per_year=tuple(float(value) for value in absolute_velocity),
        synthetic_model_id="jx.synthetic-persistent-distant-planet.circular.v1",
        parameter_record=_parameters(
            inclination_deg=float(inclination_deg),
            mass_earth=float(mass_earth),
            orbital_radius_AU=float(orbital_radius_AU),
            phase_deg=float(phase_deg),
        ),
    )


def _build_synthetic_linear_passage(
    base_state: MechanismSourceState,
    *,
    source_id: str,
    role: str,
    synthetic_model_id: str,
    gm_AU3_per_year2: float,
    speed_AU_per_year: float,
    impact_parameter_AU: float,
    trajectory_inclination_deg: float,
    nominal_event_global_step: int,
    step_years: float,
    mass_label: str,
    mass_value: float,
) -> SyntheticPerturber:
    if type(base_state) is not MechanismSourceState:
        raise MechanismScheduleError("base_state must be MechanismSourceState")
    _text(source_id, "source_id")
    if source_id in base_state.source_ids:
        raise MechanismScheduleError("synthetic source_id duplicates a base source")
    gm = _positive(gm_AU3_per_year2, "perturber GM")
    speed = _positive(speed_AU_per_year, "speed_AU_per_year")
    impact = _positive(impact_parameter_AU, "impact_parameter_AU")
    nominal = _positive_int(
        nominal_event_global_step,
        "nominal_event_global_step",
    )
    step = _positive(step_years, "step_years")
    inclination_value = _finite(
        trajectory_inclination_deg,
        "trajectory_inclination_deg",
    )
    inclination = math.radians(inclination_value)
    direction = np.asarray(
        (0.0, math.cos(inclination), math.sin(inclination)), dtype=np.float64
    )
    impact_vector = np.asarray((impact, 0.0, 0.0), dtype=np.float64)
    velocity = speed * direction
    relative_position = impact_vector - velocity * (nominal * step)
    absolute_position = base_state.positions_AU[0] + relative_position
    absolute_velocity = base_state.velocities_AU_per_year[0] + velocity
    parameter_values = {
        "asymptotic_linear_impact_parameter_AU": impact,
        mass_label: mass_value,
        "nominal_event_global_step": nominal,
        "speed_AU_per_year": speed,
        "step_years": step,
        "trajectory_inclination_deg": inclination_value,
    }
    return SyntheticPerturber(
        source_id=source_id,
        role=role,
        gm_AU3_per_year2=gm,
        initial_position_AU=tuple(float(value) for value in absolute_position),
        initial_velocity_AU_per_year=tuple(float(value) for value in absolute_velocity),
        synthetic_model_id=synthetic_model_id,
        parameter_record=_parameters(**parameter_values),
    )


def build_synthetic_stellar_flyby(
    base_state: MechanismSourceState,
    *,
    nominal_event_global_step: int,
    step_years: float,
    source_id: str = "FLYBY_STAR_SYNTHETIC",
    mass_solar: float = 0.5,
    speed_AU_per_year: float = 4.218949842,
    impact_parameter_AU: float = 1_000.0,
    trajectory_inclination_deg: float = 35.0,
) -> SyntheticPerturber:
    """Build a finite-start passing-star assumption, not a prescribed path."""

    mass = _positive(mass_solar, "mass_solar")
    return _build_synthetic_linear_passage(
        base_state,
        source_id=source_id,
        role=STELLAR_FLYBY_ROLE,
        synthetic_model_id="jx.synthetic-stellar-flyby.finite-start-cartesian.v1",
        gm_AU3_per_year2=SOLAR_GM_AU3_PER_YEAR2 * mass,
        speed_AU_per_year=speed_AU_per_year,
        impact_parameter_AU=impact_parameter_AU,
        trajectory_inclination_deg=trajectory_inclination_deg,
        nominal_event_global_step=nominal_event_global_step,
        step_years=step_years,
        mass_label="mass_solar",
        mass_value=mass,
    )


def build_synthetic_temporary_rogue(
    base_state: MechanismSourceState,
    *,
    nominal_event_global_step: int,
    step_years: float,
    source_id: str = "ROGUE_PLANET_SYNTHETIC",
    mass_earth: float = 10.0,
    speed_AU_per_year: float = 1.0,
    impact_parameter_AU: float = 100.0,
    trajectory_inclination_deg: float = 15.0,
) -> SyntheticPerturber:
    """Build an unbound synthetic rogue whose gravity is never switched off."""

    mass = _positive(mass_earth, "mass_earth")
    return _build_synthetic_linear_passage(
        base_state,
        source_id=source_id,
        role=TEMPORARY_ROGUE_ROLE,
        synthetic_model_id="jx.synthetic-unbound-rogue-flyby.finite-start-cartesian.v1",
        gm_AU3_per_year2=(
            SOLAR_GM_AU3_PER_YEAR2 * EARTH_TO_SUN_MASS_RATIO * mass
        ),
        speed_AU_per_year=speed_AU_per_year,
        impact_parameter_AU=impact_parameter_AU,
        trajectory_inclination_deg=trajectory_inclination_deg,
        nominal_event_global_step=nominal_event_global_step,
        step_years=step_years,
        mass_label="mass_earth",
        mass_value=mass,
    )


def _append_perturber_state(
    base_state: MechanismSourceState,
    perturber: SyntheticPerturber,
    *,
    snapshot_id: str,
) -> MechanismSourceState:
    if perturber.source_id in base_state.source_ids:
        raise MechanismScheduleError("perturber duplicates a base source identity")
    source_ids = (*base_state.source_ids, perturber.source_id)
    source_roles = (*base_state.source_roles, perturber.role)
    positions = _readonly_f64(
        np.vstack((base_state.positions_AU, perturber.initial_position_AU)),
        "candidate positions_AU",
        (len(source_ids), 3),
    )
    velocities = _readonly_f64(
        np.vstack(
            (
                base_state.velocities_AU_per_year,
                perturber.initial_velocity_AU_per_year,
            )
        ),
        "candidate velocities_AU_per_year",
        (len(source_ids), 3),
    )
    gm = _readonly_f64(
        np.concatenate(
            (
                base_state.gm_AU3_per_year2,
                np.asarray([perturber.gm_AU3_per_year2], dtype="<f8"),
            )
        ),
        "candidate gm_AU3_per_year2",
        (len(source_ids),),
    )
    return MechanismSourceState(
        snapshot_id=snapshot_id,
        epoch_jd_tdb=base_state.epoch_jd_tdb,
        source_ids=source_ids,
        source_roles=source_roles,
        primary_source_id=base_state.primary_source_id,
        positions_AU=positions,
        velocities_AU_per_year=velocities,
        gm_AU3_per_year2=gm,
        base_source_count=base_state.base_source_count,
        base_artifact_id=base_state.base_artifact_id,
        base_artifact_sha256=base_state.base_artifact_sha256,
        state_classification=(
            f"{base_state.state_classification}_PLUS_{perturber.role}"
        ),
        frame=base_state.frame,
        origin=base_state.origin,
        time_scale=base_state.time_scale,
        unit_system_id=base_state.unit_system_id,
    )


def _diagnostic_window(
    event_kind: str,
    *,
    nominal_step: int,
    total_steps: int,
    half_window_steps: int,
) -> DiagnosticWindow:
    _positive_int(half_window_steps, "diagnostic_half_window_steps")
    start = max(0, nominal_step - half_window_steps)
    end = min(total_steps, nominal_step + half_window_steps + 1)
    return DiagnosticWindow(event_kind, nominal_step, start, end)


def _arm(
    base_state: MechanismSourceState,
    *,
    matched_set_id: str,
    arm_kind: str,
    step_years: float,
    total_steps: int,
    perturber: SyntheticPerturber | None = None,
    diagnostic_window: DiagnosticWindow | None = None,
) -> MechanismArm:
    if perturber is None:
        state = base_state
    else:
        state = _append_perturber_state(
            base_state,
            perturber,
            snapshot_id=f"{base_state.snapshot_id}:{arm_kind.lower()}",
        )
    schedule = ClosedForceSchedule(
        schedule_id=f"{matched_set_id}:{arm_kind.lower()}",
        matched_set_id=matched_set_id,
        arm_kind=arm_kind,
        step_years=float(step_years),
        total_steps=total_steps,
        source_ids=state.source_ids,
        base_source_ids=base_state.source_ids,
        primary_source_id=state.primary_source_id,
        source_state_sha256=state.state_sha256,
        base_source_prefix_sha256=state.base_prefix_sha256,
        source_activity=tuple(
            SourceActivityInterval(source_id, 0, total_steps)
            for source_id in state.source_ids
        ),
        perturber=perturber,
        diagnostic_window=diagnostic_window,
    )
    return MechanismArm(schedule, state)


def build_de441_four_arm_schedule_set(
    control_state_path: str | Path,
    *,
    step_years: float,
    total_steps: int,
    matched_set_id: str = "jx-de441-current-epoch-four-mechanism-screen-v1",
    nominal_passage_global_step: int | None = None,
    diagnostic_half_window_steps: int | None = None,
) -> MatchedMechanismScheduleSet:
    """Build the closed, byte-matched four-arm current-epoch screen.

    The function accepts only the frozen DE441 control bytes.  The extra
    sources are deterministic synthetic assumptions; their defaults are
    engineering fixtures, not estimates or recommended astrophysical priors.
    """

    _text(matched_set_id, "matched_set_id")
    step = _positive(step_years, "step_years")
    count = _positive_int(total_steps, "total_steps")
    base = load_frozen_de441_outer_control(control_state_path)
    if nominal_passage_global_step is None:
        nominal = count // 2
    else:
        nominal = _positive_int(
            nominal_passage_global_step,
            "nominal_passage_global_step",
        )
    if not 0 < nominal < count:
        raise MechanismScheduleError(
            "nominal passage step must be strictly inside the modeled horizon"
        )
    if diagnostic_half_window_steps is None:
        half_window = max(1, count // 100)
    else:
        half_window = _positive_int(
            diagnostic_half_window_steps,
            "diagnostic_half_window_steps",
        )

    planet = build_synthetic_persistent_distant_planet(base)
    star = build_synthetic_stellar_flyby(
        base,
        nominal_event_global_step=nominal,
        step_years=step,
    )
    rogue = build_synthetic_temporary_rogue(
        base,
        nominal_event_global_step=nominal,
        step_years=step,
    )

    control = _arm(
        base,
        matched_set_id=matched_set_id,
        arm_kind=CONTROL_ARM,
        step_years=step,
        total_steps=count,
    )
    distant = _arm(
        base,
        matched_set_id=matched_set_id,
        arm_kind=DISTANT_PLANET_ARM,
        step_years=step,
        total_steps=count,
        perturber=planet,
    )
    flyby = _arm(
        base,
        matched_set_id=matched_set_id,
        arm_kind=STELLAR_FLYBY_ARM,
        step_years=step,
        total_steps=count,
        perturber=star,
        diagnostic_window=_diagnostic_window(
            "NOMINAL_LINEAR_STELLAR_PASSAGE_EPOCH",
            nominal_step=nominal,
            total_steps=count,
            half_window_steps=half_window,
        ),
    )
    temporary = _arm(
        base,
        matched_set_id=matched_set_id,
        arm_kind=TEMPORARY_ROGUE_ARM,
        step_years=step,
        total_steps=count,
        perturber=rogue,
        diagnostic_window=_diagnostic_window(
            "NOMINAL_LINEAR_ROGUE_PASSAGE_EPOCH",
            nominal_step=nominal,
            total_steps=count,
            half_window_steps=half_window,
        ),
    )
    return MatchedMechanismScheduleSet(
        matched_set_id=matched_set_id,
        control=control,
        persistent_distant_planet=distant,
        stellar_flyby=flyby,
        temporary_rogue=temporary,
    )


__all__ = [
    "ARM_KINDS",
    "CONTROL_ARM",
    "DE441_CONTROL_ARTIFACT_ID",
    "DE441_CONTROL_EPOCH_JD_TDB",
    "DE441_CONTROL_SOURCE_IDS",
    "DE441_CONTROL_STATE_SHA256",
    "DISTANT_PLANET_ARM",
    "DiagnosticWindow",
    "ClosedForceSchedule",
    "INITIAL_CONDITION_SCOPE",
    "MASSIVE_FORCE_SEMANTICS",
    "MATCHED_SET_SCHEMA",
    "MechanismArm",
    "MechanismScheduleError",
    "MechanismSourceState",
    "MatchedMechanismScheduleSet",
    "PROJECT_SCOPE",
    "RESULT_CLAIMS",
    "SCHEDULE_SCHEMA",
    "SCIENTIFIC_CLASSIFICATION",
    "SOURCE_ACTIVITY_SEMANTICS",
    "SOURCE_STATE_SCHEMA",
    "STELLAR_FLYBY_ARM",
    "SourceActivityInterval",
    "SyntheticPerturber",
    "TEMPORARY_ROGUE_ARM",
    "TRACER_BACKREACTION_SEMANTICS",
    "TRACER_FORCE_SEMANTICS",
    "UNSUPPORTED_CAPABILITIES",
    "build_de441_four_arm_schedule_set",
    "build_synthetic_persistent_distant_planet",
    "build_synthetic_stellar_flyby",
    "build_synthetic_temporary_rogue",
    "load_frozen_de441_outer_control",
]
