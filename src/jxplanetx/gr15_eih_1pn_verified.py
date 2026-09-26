"""Operationally verified CPU propagation for the supported EIH 1PN model.

This boundary makes the state convention and source provenance explicit, runs
the packaged GR15-EIH1PN implementation twice under distinct numerical
contracts, and fails closed unless every retained state agrees with the
declared absolute gates.  The confirmation is a same-implementation numerical
consistency check, not an independent physical model or external integrator.

A passing result verifies one execution of the declared mutual point-mass EIH
1PN model.  It is not a production ephemeris, observational reduction,
navigation product, or claim of exact general relativity.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
import re
from typing import Any

import numpy as np

from .gr15 import GR15ContractError, GR15Error, GR15Spec
from .gr15_eih_1pn import (
    GR15EIH1PNResult,
    integrate_gr15_eih_1pn,
)
from .solar_system.eih_1pn import EIH1PNParameters, SCIENTIFIC_CLAIM_STATE


API_ID = "JX_GR15_EIH1PN_VERIFIED_EXECUTION_V1"
VERIFICATION_STATUS = "PASSED_DUAL_EXECUTION_GATE"
REFERENCE_FRAME = "J2000"
TIME_SCALE = "TDB"
EPOCH_DEFINITION = "SECONDS_PAST_J2000"
COORDINATE_ORIGIN = "SYSTEM_BARYCENTER"
LENGTH_UNIT = "km"
VELOCITY_UNIT = "km/s"
GM_UNIT = "km^3/s^2"
_SHA256 = re.compile(r"[0-9a-f]{64}").fullmatch


class GR15EIH1PNVerificationError(GR15Error):
    """Base error for the verified-execution boundary."""


class GR15EIH1PNVerificationContractError(
    GR15EIH1PNVerificationError, GR15ContractError
):
    """The operational metadata or confirmation contract is invalid."""


class GR15EIH1PNVerificationFailure(GR15EIH1PNVerificationError):
    """Both executions completed but disagreed beyond a declared gate."""

    def __init__(
        self,
        message: str,
        *,
        maximum_position_difference_km: float,
        maximum_velocity_difference_km_s: float,
    ) -> None:
        super().__init__(message)
        self.maximum_position_difference_km = maximum_position_difference_km
        self.maximum_velocity_difference_km_s = maximum_velocity_difference_km_s


def _nonempty(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise GR15EIH1PNVerificationContractError(
            f"{label} must be a nonempty built-in string"
        )
    return value


def _source_hash(value: Any, label: str) -> str:
    checked = _nonempty(value, label)
    if _SHA256(checked) is None:
        raise GR15EIH1PNVerificationContractError(
            f"{label} must be a lowercase SHA-256 hexadecimal digest"
        )
    return checked


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GR15EIH1PNVerificationContractError(
            f"{label} must be a finite nonnegative real scalar"
        )
    checked = float(value)
    if not math.isfinite(checked) or checked < 0.0:
        raise GR15EIH1PNVerificationContractError(
            f"{label} must be a finite nonnegative real scalar"
        )
    return checked


@dataclass(frozen=True, slots=True)
class GR15EIH1PNOperationalContext:
    """Mandatory convention, identity, and provenance for one state."""

    body_ids: tuple[str, ...]
    state_source: str
    state_source_sha256: str
    gm_source: str
    gm_source_sha256: str
    reference_frame: str = REFERENCE_FRAME
    time_scale: str = TIME_SCALE
    epoch_definition: str = EPOCH_DEFINITION
    coordinate_origin: str = COORDINATE_ORIGIN
    length_unit: str = LENGTH_UNIT
    velocity_unit: str = VELOCITY_UNIT
    gravitational_parameter_unit: str = GM_UNIT

    def __post_init__(self) -> None:
        if type(self.body_ids) is not tuple or not 2 <= len(self.body_ids) <= 32:
            raise GR15EIH1PNVerificationContractError(
                "body_ids must be an exact tuple containing 2 through 32 identifiers"
            )
        checked_ids = tuple(
            _nonempty(value, f"body_ids[{index}]")
            for index, value in enumerate(self.body_ids)
        )
        if len(set(checked_ids)) != len(checked_ids):
            raise GR15EIH1PNVerificationContractError("body_ids must be unique")
        _nonempty(self.state_source, "state_source")
        _source_hash(self.state_source_sha256, "state_source_sha256")
        _nonempty(self.gm_source, "gm_source")
        _source_hash(self.gm_source_sha256, "gm_source_sha256")
        required = {
            "reference_frame": REFERENCE_FRAME,
            "time_scale": TIME_SCALE,
            "epoch_definition": EPOCH_DEFINITION,
            "coordinate_origin": COORDINATE_ORIGIN,
            "length_unit": LENGTH_UNIT,
            "velocity_unit": VELOCITY_UNIT,
            "gravitational_parameter_unit": GM_UNIT,
        }
        for name, expected in required.items():
            if getattr(self, name) != expected:
                raise GR15EIH1PNVerificationContractError(
                    f"{name} must be exactly {expected!r}"
                )


@dataclass(frozen=True, slots=True)
class GR15EIH1PNVerificationGates:
    """Absolute full-trajectory agreement limits for the two executions."""

    maximum_position_difference_km: float
    maximum_velocity_difference_km_s: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "maximum_position_difference_km",
            _finite_nonnegative(
                self.maximum_position_difference_km,
                "maximum_position_difference_km",
            ),
        )
        object.__setattr__(
            self,
            "maximum_velocity_difference_km_s",
            _finite_nonnegative(
                self.maximum_velocity_difference_km_s,
                "maximum_velocity_difference_km_s",
            ),
        )


@dataclass(frozen=True, slots=True)
class GR15EIH1PNVerifiedResult:
    """A primary trajectory that passed its declared confirmation gate."""

    primary: GR15EIH1PNResult
    context: GR15EIH1PNOperationalContext
    gates: GR15EIH1PNVerificationGates
    confirmation_replay_digest: str
    input_digest: str
    verification_digest: str
    maximum_position_difference_km: float
    maximum_velocity_difference_km_s: float
    primary_epsilon: float
    confirmation_epsilon: float
    primary_maximum_step_seconds: float
    confirmation_maximum_step_seconds: float
    api_id: str = API_ID
    verification_status: str = VERIFICATION_STATUS
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    verified_numerical_execution: bool = True
    independent_implementation_confirmation: bool = False
    production_ephemeris_authorized: bool = False
    navigation_authorized: bool = False
    exact_general_relativity_claimed: bool = False
    de440_equivalence_claimed: bool = False

    def __post_init__(self) -> None:
        if self.api_id != API_ID or self.verification_status != VERIFICATION_STATUS:
            raise GR15EIH1PNVerificationContractError(
                "verified result identity changed"
            )
        if self.scientific_claim_state != SCIENTIFIC_CLAIM_STATE:
            raise GR15EIH1PNVerificationContractError(
                "verified result scientific claim state changed"
            )
        if (
            not self.verified_numerical_execution
            or self.independent_implementation_confirmation
            or self.production_ephemeris_authorized
            or self.navigation_authorized
            or self.exact_general_relativity_claimed
            or self.de440_equivalence_claimed
        ):
            raise GR15EIH1PNVerificationContractError(
                "verified result elevated an unsupported claim"
            )
        for value, label in (
            (self.confirmation_replay_digest, "confirmation_replay_digest"),
            (self.input_digest, "input_digest"),
            (self.verification_digest, "verification_digest"),
        ):
            _source_hash(value, label)
        if (
            self.maximum_position_difference_km
            > self.gates.maximum_position_difference_km
            or self.maximum_velocity_difference_km_s
            > self.gates.maximum_velocity_difference_km_s
        ):
            raise GR15EIH1PNVerificationContractError(
                "verified result contains an out-of-gate comparison"
            )

    @property
    def positions(self) -> np.ndarray:
        """Return the verified primary final positions."""

        return self.primary.positions

    @property
    def velocities(self) -> np.ndarray:
        """Return the verified primary final velocities."""

        return self.primary.velocities

    @property
    def checkpoint_positions(self) -> np.ndarray:
        """Return the verified primary position trajectory."""

        return self.primary.checkpoint_positions

    @property
    def checkpoint_velocities(self) -> np.ndarray:
        """Return the verified primary velocity trajectory."""

        return self.primary.checkpoint_velocities


def _same_checkpoint_schedule(primary: GR15Spec, confirmation: GR15Spec) -> None:
    if primary.checkpoint_epochs != confirmation.checkpoint_epochs:
        raise GR15EIH1PNVerificationContractError(
            "primary and confirmation checkpoint schedules must match exactly"
        )


def _validate_confirmation(primary: GR15Spec, confirmation: GR15Spec) -> None:
    if type(primary) is not GR15Spec or type(confirmation) is not GR15Spec:
        raise GR15EIH1PNVerificationContractError(
            "primary_spec and confirmation_spec must be exact GR15Spec instances"
        )
    _same_checkpoint_schedule(primary, confirmation)
    if confirmation.epsilon > primary.epsilon:
        raise GR15EIH1PNVerificationContractError(
            "confirmation epsilon must not exceed primary epsilon"
        )
    if confirmation.initial_step > primary.initial_step:
        raise GR15EIH1PNVerificationContractError(
            "confirmation initial step must not exceed the primary initial step"
        )
    if confirmation.minimum_step > primary.minimum_step:
        raise GR15EIH1PNVerificationContractError(
            "confirmation minimum step must not exceed the primary minimum step"
        )
    if confirmation.maximum_step > primary.maximum_step:
        raise GR15EIH1PNVerificationContractError(
            "confirmation maximum step must not exceed the primary maximum step"
        )
    for name in (
        "safety_factor",
        "minimum_scale_factor",
        "maximum_scale_factor",
        "convergence_factor",
    ):
        if getattr(confirmation, name) != getattr(primary, name):
            raise GR15EIH1PNVerificationContractError(
                f"confirmation {name} must equal the primary value"
            )
    for name in ("maximum_iterations", "maximum_steps", "maximum_rejections"):
        if getattr(confirmation, name) < getattr(primary, name):
            raise GR15EIH1PNVerificationContractError(
                f"confirmation {name} must not be smaller than the primary value"
            )
    distinct = (
        confirmation.epsilon < primary.epsilon
        or confirmation.initial_step < primary.initial_step
        or confirmation.minimum_step < primary.minimum_step
        or confirmation.maximum_step < primary.maximum_step
    )
    if not distinct:
        raise GR15EIH1PNVerificationContractError(
            "confirmation must tighten epsilon or at least one step bound"
        )


def _validate_operational_arrays(
    positions: Any,
    velocities: Any,
    gm: Any,
    body_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected_state_shape = (body_count, 3)
    for value, label in (
        (positions, "positions_km"),
        (velocities, "velocities_km_s"),
    ):
        if (
            type(value) is not np.ndarray
            or value.dtype != np.dtype(np.float64)
            or value.shape != expected_state_shape
            or not value.flags.c_contiguous
            or not np.all(np.isfinite(value))
        ):
            raise GR15EIH1PNVerificationContractError(
                f"{label} must be a finite C-contiguous NumPy float64 "
                f"array with shape {expected_state_shape}"
            )
    if (
        type(gm) is not np.ndarray
        or gm.dtype != np.dtype(np.float64)
        or gm.shape != (body_count,)
        or not gm.flags.c_contiguous
        or not np.all(np.isfinite(gm))
        or np.any(gm <= 0.0)
    ):
        raise GR15EIH1PNVerificationContractError(
            "gravitational_parameters_km3_s2 must be a positive finite "
            f"C-contiguous NumPy float64 vector with shape {(body_count,)}"
        )
    return positions, velocities, gm


def _json_digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _spec_record(spec: GR15Spec) -> dict[str, Any]:
    return {
        field.name: (
            [float(value).hex() for value in spec.intermediate_epochs]
            if field.name == "intermediate_epochs"
            else (
                float(getattr(spec, field.name)).hex()
                if isinstance(getattr(spec, field.name), float)
                else getattr(spec, field.name)
            )
        )
        for field in fields(GR15Spec)
    }


def _input_digest(
    positions: np.ndarray,
    velocities: np.ndarray,
    gm: np.ndarray,
    parameters: EIH1PNParameters,
    primary: GR15Spec,
    confirmation: GR15Spec,
    context: GR15EIH1PNOperationalContext,
    gates: GR15EIH1PNVerificationGates,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"jx.gr15-eih-1pn.verified-input.v1\0")
    for value in (positions, velocities, gm):
        array = np.asarray(value, dtype=np.float64)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    record = {
        "confirmation": _spec_record(confirmation),
        "context": {
            field.name: getattr(context, field.name) for field in fields(context)
        },
        "gates": {
            "maximum_position_difference_km": (
                gates.maximum_position_difference_km.hex()
            ),
            "maximum_velocity_difference_km_s": (
                gates.maximum_velocity_difference_km_s.hex()
            ),
        },
        "parameters": {
            field.name: (
                float(getattr(parameters, field.name)).hex()
                if type(getattr(parameters, field.name)) is float
                else getattr(parameters, field.name)
            )
            for field in fields(parameters)
        },
        "primary": _spec_record(primary),
    }
    digest.update(
        json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )
    return digest.hexdigest()


def integrate_verified_gr15_eih_1pn(
    positions_km: np.ndarray,
    velocities_km_s: np.ndarray,
    gravitational_parameters_km3_s2: np.ndarray,
    primary_spec: GR15Spec,
    confirmation_spec: GR15Spec,
    parameters: EIH1PNParameters,
    context: GR15EIH1PNOperationalContext,
    gates: GR15EIH1PNVerificationGates,
) -> GR15EIH1PNVerifiedResult:
    """Run and verify one explicitly scoped GR15-EIH1PN propagation.

    The confirmation uses the same packaged implementation with a distinct,
    no-looser numerical contract. Inputs are validated again by the underlying
    public integrator and are never modified.
    """

    if type(context) is not GR15EIH1PNOperationalContext:
        raise GR15EIH1PNVerificationContractError(
            "context must be an exact GR15EIH1PNOperationalContext"
        )
    if type(gates) is not GR15EIH1PNVerificationGates:
        raise GR15EIH1PNVerificationContractError(
            "gates must be exact GR15EIH1PNVerificationGates"
        )
    if type(parameters) is not EIH1PNParameters:
        raise GR15EIH1PNVerificationContractError(
            "parameters must be an exact EIH1PNParameters"
        )
    _validate_confirmation(primary_spec, confirmation_spec)
    body_count = len(context.body_ids)
    positions, velocities, gm = _validate_operational_arrays(
        positions_km,
        velocities_km_s,
        gravitational_parameters_km3_s2,
        body_count,
    )
    if body_count != positions.shape[0]:
        raise GR15EIH1PNVerificationContractError(
            "body_ids count must match the state body count"
        )
    digest = _input_digest(
        positions,
        velocities,
        gm,
        parameters,
        primary_spec,
        confirmation_spec,
        context,
        gates,
    )
    primary = integrate_gr15_eih_1pn(
        positions,
        velocities,
        gm,
        primary_spec,
        parameters,
    )
    confirmation = integrate_gr15_eih_1pn(
        positions,
        velocities,
        gm,
        confirmation_spec,
        parameters,
    )
    position_differences = np.linalg.norm(
        primary.checkpoint_positions - confirmation.checkpoint_positions,
        axis=2,
    )
    velocity_differences = np.linalg.norm(
        primary.checkpoint_velocities - confirmation.checkpoint_velocities,
        axis=2,
    )
    maximum_position = float(np.max(position_differences))
    maximum_velocity = float(np.max(velocity_differences))
    if (
        maximum_position > gates.maximum_position_difference_km
        or maximum_velocity > gates.maximum_velocity_difference_km_s
    ):
        raise GR15EIH1PNVerificationFailure(
            "primary and confirmation trajectories exceeded a verification gate",
            maximum_position_difference_km=maximum_position,
            maximum_velocity_difference_km_s=maximum_velocity,
        )
    verification_digest = _json_digest(
        {
            "api_id": API_ID,
            "confirmation_replay_digest": confirmation.replay_digest,
            "input_digest": digest,
            "maximum_position_difference_km": maximum_position.hex(),
            "maximum_velocity_difference_km_s": maximum_velocity.hex(),
            "primary_replay_digest": primary.replay_digest,
            "verification_status": VERIFICATION_STATUS,
        }
    )
    return GR15EIH1PNVerifiedResult(
        primary=primary,
        context=context,
        gates=gates,
        confirmation_replay_digest=confirmation.replay_digest,
        input_digest=digest,
        verification_digest=verification_digest,
        maximum_position_difference_km=maximum_position,
        maximum_velocity_difference_km_s=maximum_velocity,
        primary_epsilon=primary_spec.epsilon,
        confirmation_epsilon=confirmation_spec.epsilon,
        primary_maximum_step_seconds=primary_spec.maximum_step,
        confirmation_maximum_step_seconds=confirmation_spec.maximum_step,
    )


__all__ = [
    "API_ID",
    "COORDINATE_ORIGIN",
    "EPOCH_DEFINITION",
    "GM_UNIT",
    "GR15EIH1PNOperationalContext",
    "GR15EIH1PNVerificationContractError",
    "GR15EIH1PNVerificationError",
    "GR15EIH1PNVerificationFailure",
    "GR15EIH1PNVerificationGates",
    "GR15EIH1PNVerifiedResult",
    "LENGTH_UNIT",
    "REFERENCE_FRAME",
    "TIME_SCALE",
    "VELOCITY_UNIT",
    "VERIFICATION_STATUS",
    "integrate_verified_gr15_eih_1pn",
]
