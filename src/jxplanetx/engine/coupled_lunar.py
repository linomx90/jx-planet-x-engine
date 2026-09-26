"""Typed unified-engine boundary for retained coupled lunar dynamics.

This module does not duplicate the lunar equations.  It gives the retained
Sun--Earth--Moon delayed-deformation solver an explicit state-block ABI,
coordinate context, provenance custody, common result digest, and public
engine entry point.  The native solver still owns its fixed-delay RKF78
physics bundle; it has not yet been converted to force ABI v1 callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Callable, Mapping

from jxplanetx.solar_system import lunar_coupled_deformable as _deformable
from jxplanetx.solar_system import lunar_coupled_solar as _solar

from .contracts import Provenance
from .coupled_lunar_contracts import COUPLED_LUNAR_RKF78_METHOD_ID


COUPLED_LUNAR_STATE_ABI_VERSION = 1
COUPLED_LUNAR_RESULT_DIGEST_ALGORITHM = (
    "SHA256_CANONICAL_JSON_AND_ARRAY_BYTES_V1"
)
COUPLED_LUNAR_RESULT_DIGEST_DOMAIN = (
    "jxplanetx.engine.coupled-lunar-result.v1"
)
COUPLED_LUNAR_FORCE_DISPATCH_SCOPE = (
    "NATIVE_COUPLED_PHYSICS_BUNDLE_NOT_FORCE_ABI_V1"
)
COUPLED_LUNAR_BODY_IDS = ("SUN", "EARTH", "MOON")
SCIENTIFIC_CLAIM_STATE = "SCREENING_ONLY"


class CoupledLunarContractError(ValueError):
    """The coupled lunar state or execution contract is incomplete."""


@dataclass(frozen=True, slots=True)
class StateBlockSpec:
    """One logical block in a versioned first-order state layout."""

    block_id: str
    field_name: str
    shape: tuple[int | str, ...]
    units: str
    frame: str
    derivative_block_id: str

    def __post_init__(self) -> None:
        for label in (
            "block_id",
            "field_name",
            "units",
            "frame",
            "derivative_block_id",
        ):
            value = getattr(self, label)
            if type(value) is not str or not value or value.strip() != value:
                raise CoupledLunarContractError(
                    f"{label} must be a nonempty, trimmed string"
                )
        if type(self.shape) is not tuple or not self.shape:
            raise CoupledLunarContractError("shape must be a nonempty tuple")
        for dimension in self.shape:
            if type(dimension) is int:
                if dimension < 1:
                    raise CoupledLunarContractError(
                        "integer state-block dimensions must be positive"
                    )
            elif dimension != "BODY_COUNT":
                raise CoupledLunarContractError(
                    "symbolic state-block dimensions must equal BODY_COUNT"
                )


_STATE_BLOCK_SPECS = (
    StateBlockSpec(
        "translation.position",
        "positions_km",
        ("BODY_COUNT", 3),
        "KM",
        "INERTIAL",
        "translation.velocity",
    ),
    StateBlockSpec(
        "translation.velocity",
        "velocities_km_s",
        ("BODY_COUNT", 3),
        "KM/S",
        "INERTIAL",
        "translation.acceleration",
    ),
    StateBlockSpec(
        "lunar.mantle.attitude_quaternion",
        "mantle_quaternion_body_to_inertial",
        (4,),
        "1",
        "MANTLE_BODY_TO_INERTIAL",
        "lunar.mantle.quaternion_rate",
    ),
    StateBlockSpec(
        "lunar.mantle.angular_velocity",
        "mantle_angular_velocity_body_s",
        (3,),
        "1/S",
        "LUNAR_MANTLE_PRINCIPAL_AXIS",
        "lunar.mantle.angular_acceleration",
    ),
    StateBlockSpec(
        "lunar.fluid_core.angular_velocity",
        "core_angular_velocity_body_s",
        (3,),
        "1/S",
        "LUNAR_MANTLE_PRINCIPAL_AXIS",
        "lunar.fluid_core.angular_acceleration",
    ),
)

STATE_BLOCK_REGISTRY: Mapping[str, StateBlockSpec] = MappingProxyType(
    {spec.block_id: spec for spec in _STATE_BLOCK_SPECS}
)


def list_state_blocks() -> tuple[StateBlockSpec, ...]:
    """Return the stable coupled-state ABI roster."""

    return _STATE_BLOCK_SPECS


def get_state_block(block_id: str) -> StateBlockSpec:
    """Return one exact state-block declaration or fail closed."""

    if type(block_id) is not str or not block_id or block_id.strip() != block_id:
        raise CoupledLunarContractError(
            "block_id must be a nonempty, trimmed string"
        )
    try:
        return STATE_BLOCK_REGISTRY[block_id]
    except KeyError as exc:
        raise CoupledLunarContractError(
            f"unknown coupled state block {block_id!r}"
        ) from exc


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise CoupledLunarContractError(
            f"{label} must be a nonempty, trimmed string"
        )
    return value


def _owned_float64(value: object, shape: tuple[int, ...], label: str):
    try:
        import numpy as np
    except (ImportError, OSError) as exc:  # pragma: no cover - engine extra
        raise CoupledLunarContractError(
            "coupled lunar state blocks require NumPy"
        ) from exc
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.float64)
        or value.shape != shape
        or not value.flags.c_contiguous
        or not np.all(np.isfinite(value))
    ):
        raise CoupledLunarContractError(
            f"{label} must be finite C-contiguous NumPy float64 with shape {shape}"
        )
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    result[result == 0.0] = 0.0
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, eq=False)
class CoupledLunarStateSnapshot:
    """Owned translation, mantle-attitude, mantle-rate, and core-rate state."""

    snapshot_id: str
    epoch: float
    time_scale: str
    frame: str
    origin: str
    axes: str
    body_ids: tuple[str, ...]
    positions_km: object
    velocities_km_s: object
    mantle_quaternion_body_to_inertial: object
    mantle_angular_velocity_body_s: object
    core_angular_velocity_body_s: object
    provenance: Provenance
    state_abi_version: int = COUPLED_LUNAR_STATE_ABI_VERSION
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    production_authorized: bool = False

    def __post_init__(self) -> None:
        _text(self.snapshot_id, "snapshot_id")
        if type(self.epoch) is not float or not math.isfinite(self.epoch):
            raise CoupledLunarContractError("epoch must be a finite built-in float")
        expected_context = (
            (self.time_scale, "TDB", "time_scale"),
            (self.frame, "BARYCENTRIC_INERTIAL", "frame"),
            (self.origin, "SYSTEM_BARYCENTER", "origin"),
            (self.axes, "J2000", "axes"),
        )
        for value, expected, label in expected_context:
            if value != expected:
                raise CoupledLunarContractError(
                    f"{label} must equal {expected!r} for the public coupled route"
                )
        if self.body_ids != COUPLED_LUNAR_BODY_IDS:
            raise CoupledLunarContractError(
                f"body_ids must exactly equal {COUPLED_LUNAR_BODY_IDS!r}"
            )
        if type(self.provenance) is not Provenance:
            raise CoupledLunarContractError("provenance must be an exact Provenance")
        positions = _owned_float64(self.positions_km, (3, 3), "positions_km")
        velocities = _owned_float64(
            self.velocities_km_s, (3, 3), "velocities_km_s"
        )
        quaternion = _owned_float64(
            self.mantle_quaternion_body_to_inertial,
            (4,),
            "mantle_quaternion_body_to_inertial",
        )
        mantle_rate = _owned_float64(
            self.mantle_angular_velocity_body_s,
            (3,),
            "mantle_angular_velocity_body_s",
        )
        core_rate = _owned_float64(
            self.core_angular_velocity_body_s,
            (3,),
            "core_angular_velocity_body_s",
        )
        # Use the retained state constructor as the sole attitude-domain
        # authority, then retain its normalized, owned representation.
        try:
            native = _solar.SolarCoupledLunarState(
                epoch=self.epoch,
                positions_km=positions,
                velocities_km_s=velocities,
                mantle_quaternion_body_to_inertial=quaternion,
                mantle_angular_velocity_body_s=mantle_rate,
                core_angular_velocity_body_s=core_rate,
            )
        except _solar.SolarCoupledLunarError as exc:
            raise CoupledLunarContractError(str(exc)) from exc
        object.__setattr__(self, "positions_km", native.positions_km)
        object.__setattr__(self, "velocities_km_s", native.velocities_km_s)
        object.__setattr__(
            self,
            "mantle_quaternion_body_to_inertial",
            native.mantle_quaternion_body_to_inertial,
        )
        object.__setattr__(
            self,
            "mantle_angular_velocity_body_s",
            native.mantle_angular_velocity_body_s,
        )
        object.__setattr__(
            self,
            "core_angular_velocity_body_s",
            native.core_angular_velocity_body_s,
        )
        if (
            type(self.state_abi_version) is not int
            or self.state_abi_version != COUPLED_LUNAR_STATE_ABI_VERSION
            or self.scientific_claim_state != SCIENTIFIC_CLAIM_STATE
            or type(self.production_authorized) is not bool
            or self.production_authorized
        ):
            raise CoupledLunarContractError(
                "coupled state ABI and screening-only claim controls changed"
            )

    @property
    def block_ids(self) -> tuple[str, ...]:
        return tuple(spec.block_id for spec in _STATE_BLOCK_SPECS)

    def to_native(self) -> _solar.SolarCoupledLunarState:
        """Return an owned retained-solver state with identical block values."""

        return _solar.SolarCoupledLunarState(
            epoch=self.epoch,
            positions_km=self.positions_km,
            velocities_km_s=self.velocities_km_s,
            mantle_quaternion_body_to_inertial=(
                self.mantle_quaternion_body_to_inertial
            ),
            mantle_angular_velocity_body_s=self.mantle_angular_velocity_body_s,
            core_angular_velocity_body_s=self.core_angular_velocity_body_s,
        )


def bind_coupled_lunar_state(
    snapshot_id: str,
    state: _solar.SolarCoupledLunarState,
    provenance: Provenance,
    *,
    time_scale: str = "TDB",
    frame: str = "BARYCENTRIC_INERTIAL",
    origin: str = "SYSTEM_BARYCENTER",
    axes: str = "J2000",
) -> CoupledLunarStateSnapshot:
    """Bind one retained lunar state to explicit public-engine context."""

    if type(state) is not _solar.SolarCoupledLunarState:
        raise CoupledLunarContractError(
            "state must be an exact SolarCoupledLunarState"
        )
    return CoupledLunarStateSnapshot(
        snapshot_id=snapshot_id,
        epoch=state.epoch,
        time_scale=time_scale,
        frame=frame,
        origin=origin,
        axes=axes,
        body_ids=COUPLED_LUNAR_BODY_IDS,
        positions_km=state.positions_km,
        velocities_km_s=state.velocities_km_s,
        mantle_quaternion_body_to_inertial=(
            state.mantle_quaternion_body_to_inertial
        ),
        mantle_angular_velocity_body_s=state.mantle_angular_velocity_body_s,
        core_angular_velocity_body_s=state.core_angular_velocity_body_s,
        provenance=provenance,
    )


def _checkpoint_digest(
    initial: CoupledLunarStateSnapshot,
    native: _deformable.DeformableSolarCoupledLunarResult,
) -> str:
    payload = {
        "algorithm": COUPLED_LUNAR_RESULT_DIGEST_ALGORITHM,
        "axes": initial.axes,
        "body_ids": list(initial.body_ids),
        "checkpoint_step_indices": list(
            native.integration_spec.checkpoint_step_indices
        ),
        "force_dispatch_scope": COUPLED_LUNAR_FORCE_DISPATCH_SCOPE,
        "frame": initial.frame,
        "method_id": COUPLED_LUNAR_RKF78_METHOD_ID,
        "model_id": native.model_id,
        "origin": initial.origin,
        "provenance_sha256": initial.provenance.sha256,
        "schema": COUPLED_LUNAR_RESULT_DIGEST_DOMAIN,
        "stage_ledger_sha256": native.stage_ledger_sha256,
        "state_abi_version": COUPLED_LUNAR_STATE_ABI_VERSION,
        "time_scale": initial.time_scale,
    }
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    digest = hashlib.sha256(
        COUPLED_LUNAR_RESULT_DIGEST_DOMAIN.encode("ascii") + b"\x00" + canonical
    )
    for index, checkpoint in enumerate(native.checkpoints):
        digest.update(index.to_bytes(8, "big", signed=False))
        digest.update(float(checkpoint.epoch).hex().encode("ascii"))
        for label, value in (
            ("positions_km", checkpoint.positions_km),
            ("velocities_km_s", checkpoint.velocities_km_s),
            (
                "mantle_quaternion_body_to_inertial",
                checkpoint.mantle_quaternion_body_to_inertial,
            ),
            (
                "mantle_angular_velocity_body_s",
                checkpoint.mantle_angular_velocity_body_s,
            ),
            (
                "core_angular_velocity_body_s",
                checkpoint.core_angular_velocity_body_s,
            ),
        ):
            digest.update(label.encode("ascii") + b"\x00")
            digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class CoupledLunarRun:
    """Common engine envelope around one retained coupled lunar execution."""

    initial_state: CoupledLunarStateSnapshot
    checkpoints: tuple[CoupledLunarStateSnapshot, ...]
    parameters: _deformable.DeformableSolarCoupledLunarParameters
    integration_spec: _deformable.DeformableSolarCoupledLunarIntegrationSpec
    native_result: _deformable.DeformableSolarCoupledLunarResult
    result_content_sha256: str
    method_id: str = COUPLED_LUNAR_RKF78_METHOD_ID
    force_dispatch_scope: str = COUPLED_LUNAR_FORCE_DISPATCH_SCOPE
    state_abi_version: int = COUPLED_LUNAR_STATE_ABI_VERSION
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    production_authorized: bool = False
    ephemeris_qualified: bool = False
    general_superiority_claimed: bool = False

    def __post_init__(self) -> None:
        if type(self.initial_state) is not CoupledLunarStateSnapshot:
            raise CoupledLunarContractError(
                "initial_state must be an exact CoupledLunarStateSnapshot"
            )
        if (
            type(self.checkpoints) is not tuple
            or len(self.checkpoints) < 2
            or any(type(item) is not CoupledLunarStateSnapshot for item in self.checkpoints)
        ):
            raise CoupledLunarContractError(
                "checkpoints must contain at least two exact coupled states"
            )
        if type(self.parameters) is not _deformable.DeformableSolarCoupledLunarParameters:
            raise CoupledLunarContractError("parameters type changed")
        if (
            type(self.integration_spec)
            is not _deformable.DeformableSolarCoupledLunarIntegrationSpec
        ):
            raise CoupledLunarContractError("integration_spec type changed")
        if type(self.native_result) is not _deformable.DeformableSolarCoupledLunarResult:
            raise CoupledLunarContractError("native_result type changed")
        if (
            type(self.result_content_sha256) is not str
            or len(self.result_content_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.result_content_sha256)
        ):
            raise CoupledLunarContractError(
                "result_content_sha256 must be lowercase SHA-256 hex"
            )
        if (
            self.method_id != COUPLED_LUNAR_RKF78_METHOD_ID
            or self.force_dispatch_scope != COUPLED_LUNAR_FORCE_DISPATCH_SCOPE
            or self.state_abi_version != COUPLED_LUNAR_STATE_ABI_VERSION
            or self.scientific_claim_state != SCIENTIFIC_CLAIM_STATE
            or type(self.production_authorized) is not bool
            or self.production_authorized
            or type(self.ephemeris_qualified) is not bool
            or self.ephemeris_qualified
            or type(self.general_superiority_claimed) is not bool
            or self.general_superiority_claimed
        ):
            raise CoupledLunarContractError(
                "coupled run method, ABI, or claim controls changed"
            )

    @property
    def final_state(self) -> CoupledLunarStateSnapshot:
        return self.checkpoints[-1]

    @property
    def accepted_steps(self) -> int:
        return self.native_result.accepted_steps

    @property
    def force_evaluations(self) -> int:
        return self.native_result.rkf78_stage_evaluations


# Keep the runtime alias dependency-light: the retained deformable module
# imports low-level engine helpers, so dereferencing one of its classes while
# that import is still in progress would make public import order observable.
PrehistoryProvider = Callable[[float], object]


def integrate_deformable_coupled_lunar_state(
    initial_state: CoupledLunarStateSnapshot,
    parameters: _deformable.DeformableSolarCoupledLunarParameters,
    integration_spec: _deformable.DeformableSolarCoupledLunarIntegrationSpec,
    prehistory_provider: PrehistoryProvider,
) -> CoupledLunarRun:
    """Advance all five state blocks with the retained simultaneous solver."""

    if type(initial_state) is not CoupledLunarStateSnapshot:
        raise CoupledLunarContractError(
            "initial_state must be an exact CoupledLunarStateSnapshot"
        )
    if type(parameters) is not _deformable.DeformableSolarCoupledLunarParameters:
        raise CoupledLunarContractError(
            "parameters must be exact DeformableSolarCoupledLunarParameters"
        )
    if type(integration_spec) is not _deformable.DeformableSolarCoupledLunarIntegrationSpec:
        raise CoupledLunarContractError(
            "integration_spec must be exact DeformableSolarCoupledLunarIntegrationSpec"
        )
    if not callable(prehistory_provider):
        raise CoupledLunarContractError("prehistory_provider must be callable")
    native = _deformable.integrate_deformable_solar_coupled_lunar_trajectory(
        initial_state.to_native(),
        parameters,
        integration_spec,
        prehistory_provider,
    )
    checkpoints = tuple(
        bind_coupled_lunar_state(
            f"{initial_state.snapshot_id}:checkpoint:{step_index}",
            checkpoint,
            initial_state.provenance,
            time_scale=initial_state.time_scale,
            frame=initial_state.frame,
            origin=initial_state.origin,
            axes=initial_state.axes,
        )
        for step_index, checkpoint in zip(
            integration_spec.checkpoint_step_indices,
            native.checkpoints,
            strict=True,
        )
    )
    digest = _checkpoint_digest(initial_state, native)
    return CoupledLunarRun(
        initial_state=initial_state,
        checkpoints=checkpoints,
        parameters=parameters,
        integration_spec=integration_spec,
        native_result=native,
        result_content_sha256=digest,
    )


__all__ = [
    "COUPLED_LUNAR_BODY_IDS",
    "COUPLED_LUNAR_FORCE_DISPATCH_SCOPE",
    "COUPLED_LUNAR_RESULT_DIGEST_ALGORITHM",
    "COUPLED_LUNAR_RESULT_DIGEST_DOMAIN",
    "COUPLED_LUNAR_RKF78_METHOD_ID",
    "COUPLED_LUNAR_STATE_ABI_VERSION",
    "CoupledLunarContractError",
    "CoupledLunarRun",
    "CoupledLunarStateSnapshot",
    "SCIENTIFIC_CLAIM_STATE",
    "STATE_BLOCK_REGISTRY",
    "StateBlockSpec",
    "bind_coupled_lunar_state",
    "get_state_block",
    "integrate_deformable_coupled_lunar_state",
    "list_state_blocks",
]
