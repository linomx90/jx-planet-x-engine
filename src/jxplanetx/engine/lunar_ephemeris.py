"""Public screening-only eleven-body Earth--Moon physical model.

This component promotes the retained CPU equations that passed the lunar EIH
screen into the engine package.  It advances the resolved eleven-body
translation state together with lunar mantle attitude, mantle angular
velocity, and fluid-core angular velocity.  Every RKF78 stage evaluates:

* mutual eleven-body Newtonian point-mass gravity;
* mutual point-mass EIH 1PN gravity (beta = gamma = 1);
* the reacting static lunar quadrupole for the Sun and Earth;
* the reacting fixed-axis Earth J2 term; and
* mantle--core pressure and viscous coupling.

The component intentionally does not include Earth J3--J5, lunar degree
three, time-variable deformation, delayed tides, minor bodies, observation
reduction, or fitted initial conditions.  Its result is screening-only model
output, not a production ephemeris.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import struct
from typing import Any

from jxplanetx.solar_system import eih_1pn
from jxplanetx.solar_system import lunar_coupled as _lunar
from jxplanetx.solar_system import lunar_coupled_solar as _solar
from jxplanetx.solar_system.lunar_fluid_core import (
    LunarFluidCoreError,
    lunar_mantle_core_angular_accelerations,
)

from .backends import resolve_backend
from .contracts import Provenance
from .lunar_ephemeris_contracts import (
    LUNAR_EPHEMERIS_V1_METHOD_ID,
    LUNAR_EPHEMERIS_V1_MODEL_ID,
)
from .rkf78 import RKF78_STAGE_COUNT, _rkf78_step


LUNAR_EPHEMERIS_V1_STATE_ABI_VERSION = 1
LUNAR_EPHEMERIS_V1_RESULT_DIGEST_DOMAIN = (
    "jxplanetx.engine.lunar-ephemeris-v1-result.v2"
)
LUNAR_EPHEMERIS_V1_EPOCH_UNIT = "SI_SECOND"
LUNAR_EPHEMERIS_V1_EPOCH_ORIGIN = "J2000_TDB_JD_2451545.0"
LUNAR_EPHEMERIS_V1_FORCE_ROSTER = (
    "MUTUAL_NEWTONIAN_POINT_MASS",
    "MUTUAL_EIH_1PN_GR",
    "LUNAR_STATIC_DEGREE2_SUN_REACTING",
    "LUNAR_STATIC_DEGREE2_EARTH_REACTING",
    "EARTH_J2_FIXED_AXIS_REACTING",
    "LUNAR_MANTLE_FLUID_CORE_COUPLING",
)
LUNAR_EPHEMERIS_V1_BODY_IDS = (
    "SUN",
    "MERCURY",
    "VENUS",
    "EARTH",
    "MOON",
    "MARS_BARYCENTER",
    "JUPITER_BARYCENTER",
    "SATURN_BARYCENTER",
    "URANUS_BARYCENTER",
    "NEPTUNE_BARYCENTER",
    "PLUTO_BARYCENTER",
)
LUNAR_EPHEMERIS_V1_OMITTED_PHYSICS = (
    "EARTH_J3_THROUGH_J5",
    "EARTH_TESSERALS_AND_TIME_VARIABLE_GRAVITY",
    "LUNAR_DEGREE3_AND_HIGHER",
    "TIME_VARIABLE_LUNAR_DEFORMATION",
    "DELAYED_TIDES",
    "MINOR_BODIES_AND_PLANETARY_SATELLITES",
    "OBSERVATION_REDUCTION_AND_PARAMETER_FITTING",
)
SCIENTIFIC_CLAIM_STATE = "SCREENING_ONLY"
SUN_INDEX = 0
EARTH_INDEX = 3
MOON_INDEX = 4
BODY_COUNT = len(LUNAR_EPHEMERIS_V1_BODY_IDS)
_POSITION_COMPONENTS = BODY_COUNT * 3 + 4
_VELOCITY_COMPONENTS = BODY_COUNT * 3 + 6
_LEDGER_DOMAIN = b"jxplanetx.engine.lunar-ephemeris-v1-ledger.v2\0"


class LunarEphemerisV1Error(ValueError):
    """The public physical-model contract or integration failed."""


class LunarEphemerisV1StepLimitError(LunarEphemerisV1Error):
    """The declared fixed-step work budget was exhausted."""


def _numpy():
    try:
        import numpy as np
    except (ImportError, OSError) as exc:  # pragma: no cover - engine extra
        raise LunarEphemerisV1Error(
            "lunar ephemeris v1 requires NumPy"
        ) from exc
    return np


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise LunarEphemerisV1Error(
            f"{label} must be a nonempty, trimmed string"
        )
    return value


def _finite_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise LunarEphemerisV1Error(
            f"{label} must be a finite built-in float"
        )
    return value


def _positive_float(value: object, label: str) -> float:
    result = _finite_float(value, label)
    if result <= 0.0:
        raise LunarEphemerisV1Error(f"{label} must be positive")
    return result


def _owned_array(value: object, shape: tuple[int, ...], label: str):
    np = _numpy()
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.float64)
        or value.shape != shape
        or not value.flags.c_contiguous
        or not np.all(np.isfinite(value))
    ):
        raise LunarEphemerisV1Error(
            f"{label} must be finite C-contiguous NumPy float64 with shape {shape}"
        )
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    result[result == 0.0] = 0.0
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, eq=False)
class LunarEphemerisV1State:
    """Owned resolved-eleven translation and lunar rotation state.

    ``epoch`` is an elapsed count of SI seconds on the TDB coordinate-time
    scale from J2000 TDB (JD 2451545.0).  The two explicit metadata fields are
    appended defaults so callers using the original v1 constructor continue
    to construct the same interpretation without a positional-argument shift.
    """

    snapshot_id: str
    epoch: float
    positions_km: object
    velocities_km_s: object
    mantle_quaternion_body_to_inertial: object
    mantle_angular_velocity_body_s: object
    core_angular_velocity_body_s: object
    provenance: Provenance
    time_scale: str = "TDB"
    frame: str = "BARYCENTRIC_INERTIAL"
    origin: str = "SYSTEM_BARYCENTER"
    axes: str = "J2000"
    body_ids: tuple[str, ...] = LUNAR_EPHEMERIS_V1_BODY_IDS
    state_abi_version: int = LUNAR_EPHEMERIS_V1_STATE_ABI_VERSION
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    production_authorized: bool = False
    epoch_unit: str = LUNAR_EPHEMERIS_V1_EPOCH_UNIT
    epoch_origin: str = LUNAR_EPHEMERIS_V1_EPOCH_ORIGIN

    def __post_init__(self) -> None:
        _text(self.snapshot_id, "snapshot_id")
        _finite_float(self.epoch, "epoch")
        if type(self.provenance) is not Provenance:
            raise LunarEphemerisV1Error("provenance must be an exact Provenance")
        expected_context = (
            (self.time_scale, "TDB", "time_scale"),
            (self.frame, "BARYCENTRIC_INERTIAL", "frame"),
            (self.origin, "SYSTEM_BARYCENTER", "origin"),
            (self.axes, "J2000", "axes"),
            (self.epoch_unit, LUNAR_EPHEMERIS_V1_EPOCH_UNIT, "epoch_unit"),
            (
                self.epoch_origin,
                LUNAR_EPHEMERIS_V1_EPOCH_ORIGIN,
                "epoch_origin",
            ),
        )
        for value, expected, label in expected_context:
            if value != expected:
                raise LunarEphemerisV1Error(
                    f"{label} must equal {expected!r}"
                )
        if self.body_ids != LUNAR_EPHEMERIS_V1_BODY_IDS:
            raise LunarEphemerisV1Error(
                "body_ids must equal the resolved-eleven roster in exact order"
            )
        positions = _owned_array(
            self.positions_km, (BODY_COUNT, 3), "positions_km"
        )
        velocities = _owned_array(
            self.velocities_km_s, (BODY_COUNT, 3), "velocities_km_s"
        )
        quaternion = _owned_array(
            self.mantle_quaternion_body_to_inertial,
            (4,),
            "mantle_quaternion_body_to_inertial",
        )
        try:
            quaternion = _lunar._unit_quaternion(quaternion)
        except _lunar.CoupledLunarError as exc:
            raise LunarEphemerisV1Error(str(exc)) from exc
        quaternion.setflags(write=False)
        mantle_rate = _owned_array(
            self.mantle_angular_velocity_body_s,
            (3,),
            "mantle_angular_velocity_body_s",
        )
        core_rate = _owned_array(
            self.core_angular_velocity_body_s,
            (3,),
            "core_angular_velocity_body_s",
        )
        object.__setattr__(self, "positions_km", positions)
        object.__setattr__(self, "velocities_km_s", velocities)
        object.__setattr__(
            self, "mantle_quaternion_body_to_inertial", quaternion
        )
        object.__setattr__(self, "mantle_angular_velocity_body_s", mantle_rate)
        object.__setattr__(self, "core_angular_velocity_body_s", core_rate)
        if (
            self.state_abi_version != LUNAR_EPHEMERIS_V1_STATE_ABI_VERSION
            or self.scientific_claim_state != SCIENTIFIC_CLAIM_STATE
            or type(self.production_authorized) is not bool
            or self.production_authorized
        ):
            raise LunarEphemerisV1Error(
                "state ABI or screening-only claim controls changed"
            )


@dataclass(frozen=True, slots=True, eq=False)
class LunarEphemerisV1Parameters:
    """Physical constants bound to the promoted v1 equation set."""

    coupled_lunar: _solar.SolarCoupledLunarParameters
    gravitational_parameters_km3_s2: object
    eih: eih_1pn.EIH1PNParameters
    model_id: str = LUNAR_EPHEMERIS_V1_MODEL_ID
    force_roster: tuple[str, ...] = LUNAR_EPHEMERIS_V1_FORCE_ROSTER
    omitted_physics: tuple[str, ...] = LUNAR_EPHEMERIS_V1_OMITTED_PHYSICS
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    production_authorized: bool = False
    ephemeris_qualified: bool = False

    def __post_init__(self) -> None:
        if type(self.coupled_lunar) is not _solar.SolarCoupledLunarParameters:
            raise LunarEphemerisV1Error(
                "coupled_lunar must be exact SolarCoupledLunarParameters"
            )
        if type(self.eih) is not eih_1pn.EIH1PNParameters:
            raise LunarEphemerisV1Error("eih must be exact EIH1PNParameters")
        gm = _owned_array(
            self.gravitational_parameters_km3_s2,
            (BODY_COUNT,),
            "gravitational_parameters_km3_s2",
        )
        np = _numpy()
        if np.any(gm <= 0.0):
            raise LunarEphemerisV1Error(
                "all eleven gravitational parameters must be positive"
            )
        expected = self.coupled_lunar.gravitational_parameters_km3_s2
        if tuple(float(gm[index]) for index in (SUN_INDEX, EARTH_INDEX, MOON_INDEX)) != expected:
            raise LunarEphemerisV1Error(
                "Sun, Earth, and Moon GM values must exactly match coupled_lunar"
            )
        object.__setattr__(self, "gravitational_parameters_km3_s2", gm)
        if (
            self.model_id != LUNAR_EPHEMERIS_V1_MODEL_ID
            or self.force_roster != LUNAR_EPHEMERIS_V1_FORCE_ROSTER
            or self.omitted_physics != LUNAR_EPHEMERIS_V1_OMITTED_PHYSICS
            or self.scientific_claim_state != SCIENTIFIC_CLAIM_STATE
            or type(self.production_authorized) is not bool
            or self.production_authorized
            or type(self.ephemeris_qualified) is not bool
            or self.ephemeris_qualified
        ):
            raise LunarEphemerisV1Error(
                "parameter identity or screening-only claim controls changed"
            )


@dataclass(frozen=True, slots=True)
class LunarEphemerisV1IntegrationSpec:
    """Exact TDB-second checkpoint schedule and fixed maximum RKF78 step."""

    checkpoint_epochs: tuple[float, ...]
    maximum_step: float
    maximum_steps: int = 1_000_000

    def __post_init__(self) -> None:
        if type(self.checkpoint_epochs) is not tuple or len(self.checkpoint_epochs) < 2:
            raise LunarEphemerisV1Error(
                "checkpoint_epochs must contain at least two exact epochs"
            )
        epochs = tuple(
            _finite_float(value, f"checkpoint_epochs[{index}]")
            for index, value in enumerate(self.checkpoint_epochs)
        )
        differences = tuple(
            right - left for left, right in zip(epochs, epochs[1:])
        )
        if not (
            all(value > 0.0 for value in differences)
            or all(value < 0.0 for value in differences)
        ):
            raise LunarEphemerisV1Error(
                "checkpoint epochs must be strictly monotone in one direction"
            )
        _positive_float(self.maximum_step, "maximum_step")
        if type(self.maximum_steps) is not int or self.maximum_steps <= 0:
            raise LunarEphemerisV1Error(
                "maximum_steps must be a positive built-in integer"
            )


@dataclass(frozen=True, slots=True, eq=False)
class LunarEphemerisV1Run:
    """Hashed result of one coupled physical-model execution."""

    initial_state: LunarEphemerisV1State
    parameters: LunarEphemerisV1Parameters
    integration_spec: LunarEphemerisV1IntegrationSpec
    checkpoints: tuple[LunarEphemerisV1State, ...]
    accepted_step_epochs: tuple[float, ...]
    accepted_step_magnitudes: tuple[float, ...]
    accepted_steps: int
    force_evaluations: int
    accepted_step_ledger_sha256: str
    result_content_sha256: str
    maximum_eih_compactness: float
    maximum_eih_speed_fraction_squared: float
    maximum_quaternion_preprojection_norm_error: float
    method_id: str = LUNAR_EPHEMERIS_V1_METHOD_ID
    model_id: str = LUNAR_EPHEMERIS_V1_MODEL_ID
    force_roster: tuple[str, ...] = LUNAR_EPHEMERIS_V1_FORCE_ROSTER
    omitted_physics: tuple[str, ...] = LUNAR_EPHEMERIS_V1_OMITTED_PHYSICS
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    production_authorized: bool = False
    ephemeris_qualified: bool = False
    general_superiority_claimed: bool = False

    def __post_init__(self) -> None:
        if type(self.initial_state) is not LunarEphemerisV1State:
            raise LunarEphemerisV1Error("initial_state type changed")
        if type(self.parameters) is not LunarEphemerisV1Parameters:
            raise LunarEphemerisV1Error("parameters type changed")
        if type(self.integration_spec) is not LunarEphemerisV1IntegrationSpec:
            raise LunarEphemerisV1Error("integration_spec type changed")
        if (
            type(self.checkpoints) is not tuple
            or len(self.checkpoints) != len(self.integration_spec.checkpoint_epochs)
            or any(type(item) is not LunarEphemerisV1State for item in self.checkpoints)
        ):
            raise LunarEphemerisV1Error("checkpoint roster changed")
        if self.initial_state.epoch != self.integration_spec.checkpoint_epochs[0]:
            raise LunarEphemerisV1Error(
                "initial state epoch must equal the first checkpoint epoch"
            )
        initial_context = (
            self.initial_state.time_scale,
            self.initial_state.frame,
            self.initial_state.origin,
            self.initial_state.axes,
            self.initial_state.epoch_unit,
            self.initial_state.epoch_origin,
            self.initial_state.body_ids,
            self.initial_state.state_abi_version,
            self.initial_state.scientific_claim_state,
            self.initial_state.production_authorized,
        )
        for index, (checkpoint, scheduled_epoch) in enumerate(
            zip(
                self.checkpoints,
                self.integration_spec.checkpoint_epochs,
                strict=True,
            )
        ):
            if checkpoint.epoch != scheduled_epoch:
                raise LunarEphemerisV1Error(
                    f"checkpoints[{index}].epoch does not match the schedule"
                )
            checkpoint_context = (
                checkpoint.time_scale,
                checkpoint.frame,
                checkpoint.origin,
                checkpoint.axes,
                checkpoint.epoch_unit,
                checkpoint.epoch_origin,
                checkpoint.body_ids,
                checkpoint.state_abi_version,
                checkpoint.scientific_claim_state,
                checkpoint.production_authorized,
            )
            if checkpoint_context != initial_context:
                raise LunarEphemerisV1Error(
                    f"checkpoints[{index}] coordinate or claim context changed"
                )
            if checkpoint.provenance != self.initial_state.provenance:
                raise LunarEphemerisV1Error(
                    f"checkpoints[{index}] provenance changed"
                )
            expected_snapshot_id = (
                self.initial_state.snapshot_id
                if index == 0
                else f"{self.initial_state.snapshot_id}:checkpoint:{index}"
            )
            if checkpoint.snapshot_id != expected_snapshot_id:
                raise LunarEphemerisV1Error(
                    f"checkpoints[{index}] snapshot identity changed"
                )
        if _state_digest(self.checkpoints[0]) != _state_digest(self.initial_state):
            raise LunarEphemerisV1Error(
                "the first checkpoint must exactly reproduce the initial state"
            )
        if (
            type(self.accepted_step_epochs) is not tuple
            or type(self.accepted_step_magnitudes) is not tuple
        ):
            raise LunarEphemerisV1Error(
                "accepted step epochs and magnitudes must be exact tuples"
            )
        for value, label in (
            (self.accepted_steps, "accepted_steps"),
            (self.force_evaluations, "force_evaluations"),
        ):
            if type(value) is not int or value <= 0:
                raise LunarEphemerisV1Error(f"{label} must be positive")
        if (
            len(self.accepted_step_epochs) != self.accepted_steps
            or len(self.accepted_step_magnitudes) != self.accepted_steps
        ):
            raise LunarEphemerisV1Error(
                "accepted step tuple lengths must equal accepted_steps"
            )
        if self.accepted_steps > self.integration_spec.maximum_steps:
            raise LunarEphemerisV1Error(
                "accepted_steps exceeds the declared maximum_steps"
            )
        _validate_step_roster(
            self.integration_spec,
            self.accepted_step_epochs,
            self.accepted_step_magnitudes,
        )
        if self.force_evaluations != self.accepted_steps * RKF78_STAGE_COUNT:
            raise LunarEphemerisV1Error("force-evaluation accounting changed")
        for value, label in (
            (self.maximum_eih_compactness, "maximum_eih_compactness"),
            (
                self.maximum_eih_speed_fraction_squared,
                "maximum_eih_speed_fraction_squared",
            ),
            (
                self.maximum_quaternion_preprojection_norm_error,
                "maximum_quaternion_preprojection_norm_error",
            ),
        ):
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise LunarEphemerisV1Error(f"{label} must be finite and nonnegative")
        if (
            self.maximum_eih_compactness
            > self.parameters.eih.maximum_compactness
            or self.maximum_eih_speed_fraction_squared
            > self.parameters.eih.maximum_speed_fraction_squared
        ):
            raise LunarEphemerisV1Error(
                "EIH diagnostic exceeds the declared weak-field domain"
            )
        for value, label in (
            (self.accepted_step_ledger_sha256, "accepted_step_ledger_sha256"),
            (self.result_content_sha256, "result_content_sha256"),
        ):
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise LunarEphemerisV1Error(f"{label} must be lowercase SHA-256 hex")
        if (
            self.method_id != LUNAR_EPHEMERIS_V1_METHOD_ID
            or self.model_id != LUNAR_EPHEMERIS_V1_MODEL_ID
            or self.force_roster != LUNAR_EPHEMERIS_V1_FORCE_ROSTER
            or self.omitted_physics != LUNAR_EPHEMERIS_V1_OMITTED_PHYSICS
            or self.scientific_claim_state != SCIENTIFIC_CLAIM_STATE
            or type(self.production_authorized) is not bool
            or self.production_authorized
            or type(self.ephemeris_qualified) is not bool
            or self.ephemeris_qualified
            or type(self.general_superiority_claimed) is not bool
            or self.general_superiority_claimed
        ):
            raise LunarEphemerisV1Error("run identity or claim controls changed")
        expected_ledger = _accepted_step_ledger_digest(
            self.accepted_step_epochs,
            self.accepted_step_magnitudes,
        )
        if self.accepted_step_ledger_sha256 != expected_ledger:
            raise LunarEphemerisV1Error(
                "accepted_step_ledger_sha256 does not match the retained step roster"
            )
        expected_result = _result_digest(
            self.initial_state,
            self.parameters,
            self.integration_spec,
            self.checkpoints,
            self.accepted_step_epochs,
            self.accepted_step_magnitudes,
            self.accepted_steps,
            self.force_evaluations,
            self.accepted_step_ledger_sha256,
            self.maximum_eih_compactness,
            self.maximum_eih_speed_fraction_squared,
            self.maximum_quaternion_preprojection_norm_error,
        )
        if self.result_content_sha256 != expected_result:
            raise LunarEphemerisV1Error(
                "result_content_sha256 does not match the complete run content"
            )

    @property
    def final_state(self) -> LunarEphemerisV1State:
        return self.checkpoints[-1]


def _pack_state(state: LunarEphemerisV1State):
    np = _numpy()
    return (
        np.ascontiguousarray(
            np.concatenate(
                (
                    state.positions_km.reshape(BODY_COUNT * 3),
                    state.mantle_quaternion_body_to_inertial,
                )
            ),
            dtype=np.float64,
        ),
        np.ascontiguousarray(
            np.concatenate(
                (
                    state.velocities_km_s.reshape(BODY_COUNT * 3),
                    state.mantle_angular_velocity_body_s,
                    state.core_angular_velocity_body_s,
                )
            ),
            dtype=np.float64,
        ),
    )


def _state_from_packed(
    template: LunarEphemerisV1State,
    checkpoint_index: int,
    epoch: float,
    position: object,
    velocity: object,
) -> LunarEphemerisV1State:
    np = _numpy()
    p = np.asarray(position, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    return LunarEphemerisV1State(
        snapshot_id=f"{template.snapshot_id}:checkpoint:{checkpoint_index}",
        epoch=float(epoch),
        positions_km=np.ascontiguousarray(
            p[: BODY_COUNT * 3].reshape(BODY_COUNT, 3)
        ),
        velocities_km_s=np.ascontiguousarray(
            v[: BODY_COUNT * 3].reshape(BODY_COUNT, 3)
        ),
        mantle_quaternion_body_to_inertial=np.ascontiguousarray(
            p[BODY_COUNT * 3 : BODY_COUNT * 3 + 4]
        ),
        mantle_angular_velocity_body_s=np.ascontiguousarray(
            v[BODY_COUNT * 3 : BODY_COUNT * 3 + 3]
        ),
        core_angular_velocity_body_s=np.ascontiguousarray(
            v[BODY_COUNT * 3 + 3 : BODY_COUNT * 3 + 6]
        ),
        provenance=template.provenance,
        time_scale=template.time_scale,
        frame=template.frame,
        origin=template.origin,
        axes=template.axes,
        body_ids=template.body_ids,
        state_abi_version=template.state_abi_version,
        scientific_claim_state=template.scientific_claim_state,
        production_authorized=template.production_authorized,
        epoch_unit=template.epoch_unit,
        epoch_origin=template.epoch_origin,
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _framed_update(digest: Any, label: str, value: bytes) -> None:
    label_bytes = label.encode("ascii")
    digest.update(struct.pack(">Q", len(label_bytes)))
    digest.update(label_bytes)
    digest.update(struct.pack(">Q", len(value)))
    digest.update(value)


def _state_metadata(state: LunarEphemerisV1State) -> dict[str, object]:
    provenance = state.provenance
    return {
        "axes": state.axes,
        "body_ids": list(state.body_ids),
        "epoch": state.epoch.hex(),
        "epoch_origin": state.epoch_origin,
        "epoch_unit": state.epoch_unit,
        "frame": state.frame,
        "origin": state.origin,
        "production_authorized": state.production_authorized,
        "provenance": {
            "citation": provenance.citation,
            "sha256": provenance.sha256,
            "source_id": provenance.source_id,
            "version": provenance.version,
        },
        "scientific_claim_state": state.scientific_claim_state,
        "snapshot_id": state.snapshot_id,
        "state_abi_version": state.state_abi_version,
        "time_scale": state.time_scale,
    }


def _update_state_digest(
    digest: Any, label: str, state: LunarEphemerisV1State
) -> None:
    _framed_update(
        digest,
        f"{label}.metadata",
        _canonical_json_bytes(_state_metadata(state)),
    )
    for array_label, value in (
        ("positions_km", state.positions_km),
        ("velocities_km_s", state.velocities_km_s),
        (
            "mantle_quaternion_body_to_inertial",
            state.mantle_quaternion_body_to_inertial,
        ),
        (
            "mantle_angular_velocity_body_s",
            state.mantle_angular_velocity_body_s,
        ),
        ("core_angular_velocity_body_s", state.core_angular_velocity_body_s),
    ):
        _framed_update(
            digest,
            f"{label}.{array_label}.float64-c-bytes",
            value.tobytes(order="C"),
        )


def _state_digest(state: LunarEphemerisV1State) -> str:
    digest = hashlib.sha256(b"jxplanetx.engine.lunar-ephemeris-v1-state.v2\0")
    _update_state_digest(digest, "state", state)
    return digest.hexdigest()


def _parameter_digest(parameters: LunarEphemerisV1Parameters) -> str:
    """Bind every public and nested physical/claim parameter field."""

    base = parameters.coupled_lunar
    lunar = base.lunar
    core = lunar.core_coupling
    eih = parameters.eih
    payload = {
        "coupled_lunar": {
            "earth_j2": base.earth_j2.hex(),
            "earth_pole_inertial": [
                value.hex() for value in base.earth_pole_inertial
            ],
            "earth_reference_radius_km": base.earth_reference_radius_km.hex(),
            "lunar": {
                "core_coupling": {
                    "core_inertia_over_mr2": [
                        value.hex() for value in core.core_inertia_over_mr2
                    ],
                    "evidence_class": core.evidence_class,
                    "mantle_frame": core.mantle_frame,
                    "model_id": core.model_id,
                    "parameter_source_id": core.parameter_source_id,
                    "parameter_status": core.parameter_status,
                    "physical_status": core.physical_status,
                    "pressure_torque_formula": core.pressure_torque_formula,
                    "qualification_authorized": core.qualification_authorized,
                    "registry_authorized": core.registry_authorized,
                    "viscous_coefficient_over_mr2_per_second": (
                        core.viscous_coefficient_over_mr2_per_second.hex()
                    ),
                },
                "earth_gm_km3_s2": lunar.earth_gm_km3_s2.hex(),
                "figure_inertia_over_mr2": [
                    value.hex() for value in lunar.figure_inertia_over_mr2
                ],
                "lunar_reference_radius_km": (
                    lunar.lunar_reference_radius_km.hex()
                ),
                "mantle_inertia_over_mr2": [
                    value.hex() for value in lunar.mantle_inertia_over_mr2
                ],
                "model_id": lunar.model_id,
                "moon_gm_km3_s2": lunar.moon_gm_km3_s2.hex(),
                "qualification_authorized": lunar.qualification_authorized,
                "registry_authorized": lunar.registry_authorized,
                "scientific_claim_state": lunar.scientific_claim_state,
                "scope": lunar.scope,
            },
            "model_id": base.model_id,
            "qualification_authorized": base.qualification_authorized,
            "registry_authorized": base.registry_authorized,
            "scientific_claim_state": base.scientific_claim_state,
            "scope": base.scope,
            "sun_gm_km3_s2": base.sun_gm_km3_s2.hex(),
        },
        "eih": {
            "beta": eih.beta.hex(),
            "coordinate_scope": eih.coordinate_scope,
            "correction_only": eih.correction_only,
            "gamma": eih.gamma.hex(),
            "maximum_compactness": eih.maximum_compactness.hex(),
            "maximum_speed_fraction_squared": (
                eih.maximum_speed_fraction_squared.hex()
            ),
            "model_id": eih.model_id,
            "production_ready": eih.production_ready,
            "qualification_authorized": eih.qualification_authorized,
            "registry_authorized": eih.registry_authorized,
            "scientific_claim_state": eih.scientific_claim_state,
            "speed_of_light_km_s": eih.speed_of_light_km_s.hex(),
        },
        "ephemeris_qualified": parameters.ephemeris_qualified,
        "force_roster": list(parameters.force_roster),
        "gravitational_parameters_km3_s2": [
            float(value).hex()
            for value in parameters.gravitational_parameters_km3_s2
        ],
        "model_id": parameters.model_id,
        "omitted_physics": list(parameters.omitted_physics),
        "production_authorized": parameters.production_authorized,
        "scientific_claim_state": parameters.scientific_claim_state,
    }
    return hashlib.sha256(
        b"jxplanetx.engine.lunar-ephemeris-v1-parameters.v2\0"
        + _canonical_json_bytes(payload)
    ).hexdigest()


def _accepted_step_ledger_digest(
    epochs: tuple[float, ...], magnitudes: tuple[float, ...]
) -> str:
    """Hash the retained step roster; checkpoint states use the result digest."""

    digest = hashlib.sha256(_LEDGER_DOMAIN)
    for epoch, magnitude in zip(epochs, magnitudes, strict=True):
        digest.update(struct.pack(">dd", epoch, magnitude))
    return digest.hexdigest()


def _validate_step_roster(
    spec: LunarEphemerisV1IntegrationSpec,
    epochs: tuple[float, ...],
    magnitudes: tuple[float, ...],
) -> None:
    if len(epochs) != len(magnitudes):
        raise LunarEphemerisV1Error("accepted step tuple lengths differ")
    direction = (
        1.0 if spec.checkpoint_epochs[-1] > spec.checkpoint_epochs[0] else -1.0
    )
    current = spec.checkpoint_epochs[0]
    checkpoint_index = 1
    for index, (epoch, magnitude) in enumerate(
        zip(epochs, magnitudes, strict=True)
    ):
        _finite_float(epoch, f"accepted_step_epochs[{index}]")
        _positive_float(magnitude, f"accepted_step_magnitudes[{index}]")
        if checkpoint_index >= len(spec.checkpoint_epochs):
            raise LunarEphemerisV1Error(
                "accepted step roster extends past the schedule"
            )
        target = spec.checkpoint_epochs[checkpoint_index]
        remaining = target - current
        if direction * remaining <= 0.0 or not math.isfinite(remaining):
            raise LunarEphemerisV1Error("accepted step roster lost monotone progress")
        expected_magnitude = min(spec.maximum_step, abs(remaining))
        expected_epoch = (
            target
            if expected_magnitude >= abs(remaining)
            else current + math.copysign(expected_magnitude, direction)
        )
        if magnitude != expected_magnitude or epoch != expected_epoch:
            raise LunarEphemerisV1Error(
                f"accepted step {index} does not match the fixed-step schedule"
            )
        if expected_epoch == current or not math.isfinite(expected_epoch):
            raise LunarEphemerisV1Error(
                "accepted step cannot advance the binary64 epoch"
            )
        current = expected_epoch
        if current == target:
            checkpoint_index += 1
    if checkpoint_index != len(spec.checkpoint_epochs):
        raise LunarEphemerisV1Error(
            "accepted step roster does not reach every checkpoint"
        )


def _result_digest(
    initial: LunarEphemerisV1State,
    parameters: LunarEphemerisV1Parameters,
    spec: LunarEphemerisV1IntegrationSpec,
    checkpoints: tuple[LunarEphemerisV1State, ...],
    accepted_step_epochs: tuple[float, ...],
    accepted_step_magnitudes: tuple[float, ...],
    accepted_steps: int,
    force_evaluations: int,
    ledger_sha256: str,
    maximum_eih_compactness: float,
    maximum_eih_speed_fraction_squared: float,
    maximum_quaternion_preprojection_norm_error: float,
) -> str:
    payload = {
        "accepted_step_epochs": [value.hex() for value in accepted_step_epochs],
        "accepted_step_magnitudes": [
            value.hex() for value in accepted_step_magnitudes
        ],
        "accepted_steps": accepted_steps,
        "checkpoint_epochs": [value.hex() for value in spec.checkpoint_epochs],
        "ephemeris_qualified": False,
        "force_evaluations": force_evaluations,
        "force_roster": list(LUNAR_EPHEMERIS_V1_FORCE_ROSTER),
        "general_superiority_claimed": False,
        "ledger_sha256": ledger_sha256,
        "maximum_eih_compactness": maximum_eih_compactness.hex(),
        "maximum_eih_speed_fraction_squared": (
            maximum_eih_speed_fraction_squared.hex()
        ),
        "maximum_quaternion_preprojection_norm_error": (
            maximum_quaternion_preprojection_norm_error.hex()
        ),
        "maximum_step": spec.maximum_step.hex(),
        "maximum_steps": spec.maximum_steps,
        "method_id": LUNAR_EPHEMERIS_V1_METHOD_ID,
        "model_id": LUNAR_EPHEMERIS_V1_MODEL_ID,
        "omitted_physics": list(LUNAR_EPHEMERIS_V1_OMITTED_PHYSICS),
        "parameter_sha256": _parameter_digest(parameters),
        "production_authorized": False,
        "schema": LUNAR_EPHEMERIS_V1_RESULT_DIGEST_DOMAIN,
        "scientific_claim_state": SCIENTIFIC_CLAIM_STATE,
    }
    digest = hashlib.sha256(
        LUNAR_EPHEMERIS_V1_RESULT_DIGEST_DOMAIN.encode("ascii") + b"\0"
    )
    _framed_update(digest, "run", _canonical_json_bytes(payload))
    # Bind the initial bytes explicitly even though the first checkpoint is
    # required to reproduce them.  That duplication is intentional: initial
    # conditions and checkpoint evidence are separate contract roles.
    _update_state_digest(digest, "initial_state", initial)
    for index, checkpoint in enumerate(checkpoints):
        _update_state_digest(digest, f"checkpoints[{index}]", checkpoint)
    return digest.hexdigest()


def integrate_lunar_ephemeris_v1(
    initial_state: LunarEphemerisV1State,
    parameters: LunarEphemerisV1Parameters,
    integration_spec: LunarEphemerisV1IntegrationSpec,
) -> LunarEphemerisV1Run:
    """Advance the promoted coupled eleven-body EIH equation set."""

    if type(initial_state) is not LunarEphemerisV1State:
        raise LunarEphemerisV1Error(
            "initial_state must be exact LunarEphemerisV1State"
        )
    if type(parameters) is not LunarEphemerisV1Parameters:
        raise LunarEphemerisV1Error(
            "parameters must be exact LunarEphemerisV1Parameters"
        )
    if type(integration_spec) is not LunarEphemerisV1IntegrationSpec:
        raise LunarEphemerisV1Error(
            "integration_spec must be exact LunarEphemerisV1IntegrationSpec"
        )
    if initial_state.epoch != integration_spec.checkpoint_epochs[0]:
        raise LunarEphemerisV1Error(
            "initial state epoch must equal the first checkpoint"
        )

    np = _numpy()
    backend = resolve_backend("numpy")
    position, velocity = _pack_state(initial_state)
    position_carry = np.zeros(_POSITION_COMPONENTS, dtype=np.float64)
    velocity_carry = np.zeros(_VELOCITY_COMPONENTS, dtype=np.float64)
    gm = np.asarray(parameters.gravitational_parameters_km3_s2)
    base = parameters.coupled_lunar
    mantle_inertia = np.ascontiguousarray(
        np.diag(base.lunar.mantle_inertia_over_mr2), dtype=np.float64
    )
    zero_matrix = np.zeros((3, 3), dtype=np.float64)
    zero_vector = np.zeros(3, dtype=np.float64)
    accepted_epochs: list[float] = []
    accepted_magnitudes: list[float] = []
    checkpoints: list[LunarEphemerisV1State] = [initial_state]
    maximum_compactness = 0.0
    maximum_speed_fraction = 0.0
    maximum_quaternion_error = 0.0
    epoch = initial_state.epoch
    direction = (
        1.0 if integration_spec.checkpoint_epochs[-1] > epoch else -1.0
    )

    for checkpoint_index, checkpoint_epoch in enumerate(
        integration_spec.checkpoint_epochs[1:], 1
    ):
        while epoch != checkpoint_epoch:
            if len(accepted_epochs) >= integration_spec.maximum_steps:
                raise LunarEphemerisV1StepLimitError("maximum_steps exhausted")
            remaining = checkpoint_epoch - epoch
            if direction * remaining <= 0.0 or not math.isfinite(remaining):
                raise LunarEphemerisV1Error(
                    "checkpoint schedule lost monotone progress"
                )
            magnitude = min(integration_spec.maximum_step, abs(remaining))
            endpoint = (
                checkpoint_epoch
                if magnitude >= abs(remaining)
                else epoch + math.copysign(magnitude, direction)
            )
            if endpoint == epoch or not math.isfinite(endpoint):
                raise LunarEphemerisV1Error(
                    "fixed step cannot advance the binary64 epoch"
                )
            step = endpoint - epoch

            def derivative(
                _stage_epoch: float,
                stage_position: object,
                stage_velocity: object,
            ) -> tuple[object, object]:
                nonlocal maximum_compactness, maximum_speed_fraction
                state = np.asarray(stage_position, dtype=np.float64)
                rate = np.asarray(stage_velocity, dtype=np.float64)
                xyz = np.ascontiguousarray(
                    state[: BODY_COUNT * 3].reshape(BODY_COUNT, 3)
                )
                uvw = np.ascontiguousarray(
                    rate[: BODY_COUNT * 3].reshape(BODY_COUNT, 3)
                )
                quaternion = _lunar._unit_quaternion(
                    state[BODY_COUNT * 3 : BODY_COUNT * 3 + 4]
                )
                mantle_rate = np.ascontiguousarray(
                    rate[BODY_COUNT * 3 : BODY_COUNT * 3 + 3]
                )
                core_rate = np.ascontiguousarray(
                    rate[BODY_COUNT * 3 + 3 : BODY_COUNT * 3 + 6]
                )
                try:
                    acceleration = eih_1pn.newtonian_point_mass_accelerations(
                        xyz, gm
                    )
                    relativity = eih_1pn.evaluate_eih_1pn_correction(
                        xyz, uvw, gm, parameters.eih
                    )
                except eih_1pn.EIH1PNError as exc:
                    raise LunarEphemerisV1Error(str(exc)) from exc
                acceleration += relativity.correction_accelerations_km_s2
                maximum_compactness = max(
                    maximum_compactness, relativity.maximum_compactness
                )
                maximum_speed_fraction = max(
                    maximum_speed_fraction,
                    relativity.maximum_speed_fraction_squared,
                )

                external_torque = np.zeros(3, dtype=np.float64)
                for source_index, source_gm in (
                    (SUN_INDEX, base.sun_gm_km3_s2),
                    (EARTH_INDEX, base.lunar.earth_gm_km3_s2),
                ):
                    try:
                        interaction = _solar._source_lunar_quadrupole(
                            quaternion,
                            xyz[source_index],
                            xyz[MOON_INDEX],
                            source_gm_km3_s2=source_gm,
                            parameters=base,
                        )
                    except _solar.SolarCoupledLunarError as exc:
                        raise LunarEphemerisV1Error(str(exc)) from exc
                    acceleration[source_index] += (
                        interaction.source_acceleration_km_s2
                    )
                    acceleration[MOON_INDEX] += (
                        interaction.moon_acceleration_km_s2
                    )
                    external_torque += interaction.torque_on_mantle_over_mr2

                try:
                    earth_j2 = _solar._earth_j2_interaction(
                        xyz[EARTH_INDEX], xyz[MOON_INDEX], base
                    )
                except _solar.SolarCoupledLunarError as exc:
                    raise LunarEphemerisV1Error(str(exc)) from exc
                acceleration[EARTH_INDEX] += earth_j2.earth_acceleration_km_s2
                acceleration[MOON_INDEX] += earth_j2.moon_acceleration_km_s2
                try:
                    rotation = lunar_mantle_core_angular_accelerations(
                        base.lunar.core_coupling,
                        mantle_angular_velocity_body=mantle_rate,
                        core_angular_velocity_body=core_rate,
                        mantle_inertia_over_mr2=mantle_inertia,
                        mantle_inertia_rate_over_mr2_per_second=zero_matrix,
                        external_torque_over_mr2=np.ascontiguousarray(
                            external_torque
                        ),
                        geodetic_precession_body=zero_vector,
                    )
                except LunarFluidCoreError as exc:
                    raise LunarEphemerisV1Error(str(exc)) from exc
                return (
                    np.ascontiguousarray(
                        np.concatenate(
                            (
                                uvw.reshape(BODY_COUNT * 3),
                                _lunar._quaternion_derivative(
                                    quaternion, mantle_rate
                                ),
                            )
                        ),
                        dtype=np.float64,
                    ),
                    np.ascontiguousarray(
                        np.concatenate(
                            (
                                acceleration.reshape(BODY_COUNT * 3),
                                rotation.mantle_angular_acceleration_body,
                                rotation.core_angular_acceleration_body,
                            )
                        ),
                        dtype=np.float64,
                    ),
                )

            trial = _rkf78_step(
                backend=backend,
                epoch=epoch,
                endpoint_epoch=endpoint,
                step=step,
                positions=position,
                velocities=velocity,
                position_carry=position_carry,
                velocity_carry=velocity_carry,
                derivative=derivative,
            )
            accepted_position = np.ascontiguousarray(
                trial.accepted_positions, dtype=np.float64
            )
            quaternion_slice = slice(BODY_COUNT * 3, BODY_COUNT * 3 + 4)
            quaternion_norm = float(
                np.linalg.norm(accepted_position[quaternion_slice])
            )
            maximum_quaternion_error = max(
                maximum_quaternion_error, abs(quaternion_norm - 1.0)
            )
            accepted_position[quaternion_slice] = _lunar._unit_quaternion(
                accepted_position[quaternion_slice]
            )
            position = accepted_position
            velocity = np.ascontiguousarray(
                trial.accepted_velocities, dtype=np.float64
            )
            position_carry = np.ascontiguousarray(
                trial.accepted_position_carry, dtype=np.float64
            )
            position_carry[quaternion_slice] = 0.0
            velocity_carry = np.ascontiguousarray(
                trial.accepted_velocity_carry, dtype=np.float64
            )
            epoch = endpoint
            accepted_epochs.append(epoch)
            accepted_magnitudes.append(abs(step))

        checkpoints.append(
            _state_from_packed(
                initial_state,
                checkpoint_index,
                epoch,
                position,
                velocity,
            )
        )

    checkpoint_tuple = tuple(checkpoints)
    accepted_epoch_tuple = tuple(accepted_epochs)
    accepted_magnitude_tuple = tuple(accepted_magnitudes)
    ledger_sha256 = _accepted_step_ledger_digest(
        accepted_epoch_tuple,
        accepted_magnitude_tuple,
    )
    content_sha256 = _result_digest(
        initial_state,
        parameters,
        integration_spec,
        checkpoint_tuple,
        accepted_epoch_tuple,
        accepted_magnitude_tuple,
        len(accepted_epoch_tuple),
        len(accepted_epoch_tuple) * RKF78_STAGE_COUNT,
        ledger_sha256,
        float(maximum_compactness),
        float(maximum_speed_fraction),
        float(maximum_quaternion_error),
    )
    return LunarEphemerisV1Run(
        initial_state=initial_state,
        parameters=parameters,
        integration_spec=integration_spec,
        checkpoints=checkpoint_tuple,
        accepted_step_epochs=accepted_epoch_tuple,
        accepted_step_magnitudes=accepted_magnitude_tuple,
        accepted_steps=len(accepted_epoch_tuple),
        force_evaluations=len(accepted_epoch_tuple) * RKF78_STAGE_COUNT,
        accepted_step_ledger_sha256=ledger_sha256,
        result_content_sha256=content_sha256,
        maximum_eih_compactness=float(maximum_compactness),
        maximum_eih_speed_fraction_squared=float(maximum_speed_fraction),
        maximum_quaternion_preprojection_norm_error=float(
            maximum_quaternion_error
        ),
    )


__all__ = [
    "BODY_COUNT",
    "EARTH_INDEX",
    "LUNAR_EPHEMERIS_V1_BODY_IDS",
    "LUNAR_EPHEMERIS_V1_EPOCH_ORIGIN",
    "LUNAR_EPHEMERIS_V1_EPOCH_UNIT",
    "LUNAR_EPHEMERIS_V1_FORCE_ROSTER",
    "LUNAR_EPHEMERIS_V1_METHOD_ID",
    "LUNAR_EPHEMERIS_V1_MODEL_ID",
    "LUNAR_EPHEMERIS_V1_OMITTED_PHYSICS",
    "LUNAR_EPHEMERIS_V1_RESULT_DIGEST_DOMAIN",
    "LUNAR_EPHEMERIS_V1_STATE_ABI_VERSION",
    "LunarEphemerisV1Error",
    "LunarEphemerisV1IntegrationSpec",
    "LunarEphemerisV1Parameters",
    "LunarEphemerisV1Run",
    "LunarEphemerisV1State",
    "LunarEphemerisV1StepLimitError",
    "MOON_INDEX",
    "SCIENTIFIC_CLAIM_STATE",
    "SUN_INDEX",
    "integrate_lunar_ephemeris_v1",
]
