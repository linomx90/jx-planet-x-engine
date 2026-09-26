"""One explicit coordinator for existing JX force and integrator components.

This module does not introduce another numerical solver.  ``JXSimulation``
owns one state/force definition and dispatches only to existing public JX
integrators whose declared physics exactly match that definition.  The first
native adapter joins the generic ``ForcePlan`` representation to the supported
Newtonian GR15 implementation.  Incompatible requests fail closed rather than
dropping forces, changing backends, or changing collision semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from threading import RLock
from typing import Any

from .backends import BackendUnavailableError, resolve_backend
from .contracts import (
    ForcePlan,
    MutualEIH1PN,
    NewtonianPointMass,
    Provenance,
    StateSnapshot,
)
from .evaluator import _validate_state
from .trajectory import (
    TrajectoryResult,
    _copy_force_plan,
    _copy_initial_snapshot,
    integrate_trajectory,
)
from .trajectory_contracts import ADAPTIVE_RKF78_METHOD_ID, AdaptiveRKF78Spec


GR15_NEWTONIAN_METHOD_ID = "JX_GAUSS_RADAU15_V3"
GR15_EIH1PN_METHOD_ID = "JX_GR15_EIH1PN_V1"
GR15_EIH1PN_CUDA_METHOD_ID = "JX_GR15_EIH1PN_CUDA_V1"
SCIENTIFIC_CLAIM_STATE = "SCREENING_ONLY"
CONTINUATION_DIGEST_ALGORITHM = "SHA256_CANONICAL_JSON_AND_ARRAY_BYTES_V1"
CONTINUATION_DIGEST_DOMAIN = "jxplanetx.simulation.continuation-state.v1"


class JXSimulationError(RuntimeError):
    """Base error for the unified simulation coordinator."""


class JXSimulationContractError(JXSimulationError, ValueError):
    """The simulation identity, state, plan, or integrator request is invalid."""


class JXSimulationCompatibilityError(JXSimulationError):
    """An integrator cannot preserve the requested force-plan semantics."""


@dataclass(frozen=True, slots=True, eq=False)
class JXSimulationRun:
    """Common envelope around one successful existing JX integrator result."""

    simulation_id: str
    run_index: int
    integrator_id: str
    force_model_ids: tuple[str, ...]
    checkpoint_epochs: tuple[float, ...]
    checkpoint_positions: tuple[Any, ...]
    checkpoint_velocities: tuple[Any, ...]
    attempted_steps: int
    accepted_steps: int
    rejected_steps: int
    force_evaluations: int
    backend_id: str
    device: str
    native_result: object
    native_result_digest: str | None
    native_result_digest_scope: str
    parent_run_index: int | None = None
    parent_native_result_digest: str | None = None
    continuation_state_sha256: str | None = None
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    production_authorized: bool = False
    general_superiority_claimed: bool = False

    def __post_init__(self) -> None:
        if type(self.simulation_id) is not str or not self.simulation_id:
            raise JXSimulationContractError("simulation_id must be nonempty")
        if type(self.run_index) is not int or self.run_index < 0:
            raise JXSimulationContractError("run_index must be nonnegative")
        if type(self.integrator_id) is not str or not self.integrator_id:
            raise JXSimulationContractError("integrator_id must be nonempty")
        if (
            type(self.force_model_ids) is not tuple
            or not self.force_model_ids
            or any(type(value) is not str or not value for value in self.force_model_ids)
        ):
            raise JXSimulationContractError("force_model_ids must be a nonempty tuple")
        if (
            type(self.checkpoint_epochs) is not tuple
            or len(self.checkpoint_epochs) < 2
            or len(self.checkpoint_positions) != len(self.checkpoint_epochs)
            or len(self.checkpoint_velocities) != len(self.checkpoint_epochs)
        ):
            raise JXSimulationContractError("checkpoint result lengths are inconsistent")
        if any(not math.isfinite(value) for value in self.checkpoint_epochs):
            raise JXSimulationContractError("checkpoint epochs must be finite")
        for label in (
            "attempted_steps",
            "accepted_steps",
            "rejected_steps",
            "force_evaluations",
        ):
            value = getattr(self, label)
            if type(value) is not int or value < 0:
                raise JXSimulationContractError(f"{label} must be nonnegative")
        if self.attempted_steps != self.accepted_steps + self.rejected_steps:
            raise JXSimulationContractError("attempted-step accounting is inconsistent")
        for label in ("backend_id", "device", "native_result_digest_scope"):
            value = getattr(self, label)
            if type(value) is not str or not value:
                raise JXSimulationContractError(f"{label} must be nonempty")
        if self.native_result_digest is not None and (
            type(self.native_result_digest) is not str
            or len(self.native_result_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.native_result_digest)
        ):
            raise JXSimulationContractError(
                "native_result_digest must be lowercase SHA-256 hex or None"
            )
        continuation_values = (
            self.parent_run_index,
            self.parent_native_result_digest,
            self.continuation_state_sha256,
        )
        if self.parent_run_index is None:
            if any(value is not None for value in continuation_values[1:]):
                raise JXSimulationContractError(
                    "an initial run cannot retain continuation identities"
                )
        else:
            if (
                type(self.parent_run_index) is not int
                or self.parent_run_index < 0
                or self.parent_run_index >= self.run_index
            ):
                raise JXSimulationContractError(
                    "parent_run_index must identify an earlier run"
                )
            for value, label in (
                (self.parent_native_result_digest, "parent_native_result_digest"),
                (self.continuation_state_sha256, "continuation_state_sha256"),
            ):
                if (
                    type(value) is not str
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise JXSimulationContractError(
                        f"{label} must be lowercase SHA-256 hex for a continuation"
                    )
        if (
            self.scientific_claim_state != SCIENTIFIC_CLAIM_STATE
            or type(self.production_authorized) is not bool
            or self.production_authorized
            or type(self.general_superiority_claimed) is not bool
            or self.general_superiority_claimed
        ):
            raise JXSimulationContractError(
                "unified simulation runs remain nonproduction screening results"
            )

    @property
    def final_positions(self) -> Any:
        return self.checkpoint_positions[-1]

    @property
    def final_velocities(self) -> Any:
        return self.checkpoint_velocities[-1]

    @property
    def is_continuation(self) -> bool:
        return self.parent_run_index is not None


@dataclass(frozen=True, slots=True, eq=False)
class JXSimulationContinuation:
    """One provenance-bound NumPy endpoint prepared for explicit continuation."""

    simulation_id: str
    parent_run_index: int
    parent_integrator_id: str
    parent_native_result_digest: str
    continuation_state_sha256: str
    snapshot: StateSnapshot
    force_model_ids: tuple[str, ...]
    digest_algorithm: str = CONTINUATION_DIGEST_ALGORITHM
    digest_domain: str = CONTINUATION_DIGEST_DOMAIN
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    production_authorized: bool = False

    def __post_init__(self) -> None:
        _require_simulation_id(self.simulation_id)
        if type(self.parent_run_index) is not int or self.parent_run_index < 0:
            raise JXSimulationContractError("parent_run_index must be nonnegative")
        if type(self.parent_integrator_id) is not str or not self.parent_integrator_id:
            raise JXSimulationContractError("parent_integrator_id must be nonempty")
        for value, label in (
            (self.parent_native_result_digest, "parent_native_result_digest"),
            (self.continuation_state_sha256, "continuation_state_sha256"),
        ):
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise JXSimulationContractError(f"{label} must be lowercase SHA-256 hex")
        if type(self.snapshot) is not StateSnapshot:
            raise JXSimulationContractError("snapshot must be an exact StateSnapshot")
        if type(self.force_model_ids) is not tuple or not self.force_model_ids:
            raise JXSimulationContractError("force_model_ids must be a nonempty tuple")
        if (
            self.digest_algorithm != CONTINUATION_DIGEST_ALGORITHM
            or self.digest_domain != CONTINUATION_DIGEST_DOMAIN
            or self.scientific_claim_state != SCIENTIFIC_CLAIM_STATE
            or type(self.production_authorized) is not bool
            or self.production_authorized
        ):
            raise JXSimulationContractError(
                "continuations retain the fixed digest and screening-only contract"
            )


def _require_simulation_id(value: object) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise JXSimulationContractError(
            "simulation_id must be a nonempty, trimmed string"
        )
    return value


def _continuation_state_sha256(
    *,
    simulation_id: str,
    parent: JXSimulationRun,
    snapshot: StateSnapshot,
) -> str:
    """Bind one NumPy endpoint and all retained state constants to its parent."""

    if parent.native_result_digest is None:
        raise JXSimulationCompatibilityError(
            "continuation requires a parent with a retained content digest"
        )
    return _continuation_state_identity_sha256(
        simulation_id=simulation_id,
        parent_run_index=parent.run_index,
        parent_integrator_id=parent.integrator_id,
        parent_native_result_digest=parent.native_result_digest,
        force_model_ids=parent.force_model_ids,
        device=parent.device,
        snapshot=snapshot,
    )


def _continuation_state_identity_sha256(
    *,
    simulation_id: str,
    parent_run_index: int,
    parent_integrator_id: str,
    parent_native_result_digest: str,
    force_model_ids: tuple[str, ...],
    device: str,
    snapshot: StateSnapshot,
) -> str:
    """Hash explicit continuation lineage and exact NumPy state bytes."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - engine extra owns NumPy
        raise JXSimulationCompatibilityError(
            "NumPy is required to prepare a continuation state"
        ) from exc
    arrays = (
        ("positions", snapshot.positions, "float64", (len(snapshot.body_ids), 3)),
        ("velocities", snapshot.velocities, "float64", (len(snapshot.body_ids), 3)),
        (
            "gravitational_parameters",
            snapshot.gravitational_parameters,
            "float64",
            (len(snapshot.body_ids),),
        ),
        ("masses", snapshot.masses, "float64", (len(snapshot.body_ids),)),
        ("radii", snapshot.radii, "float64", (len(snapshot.body_ids),)),
        ("massive", snapshot.massive, "bool", (len(snapshot.body_ids),)),
    )
    array_records: list[dict[str, object]] = []
    for label, value, dtype, shape in arrays:
        if (
            type(value) is not np.ndarray
            or str(value.dtype) != dtype
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise JXSimulationCompatibilityError(
                f"continuation {label} must be C-contiguous NumPy {dtype} with shape {shape}"
            )
        array_records.append(
            {
                "label": label,
                "dtype": dtype,
                "shape": list(shape),
                "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
            }
        )
    payload = {
        "arrays": array_records,
        "axes": snapshot.axes,
        "body_ids": list(snapshot.body_ids),
        "device": device,
        "epoch_hex": float(snapshot.epoch).hex(),
        "force_model_ids": list(force_model_ids),
        "frame": snapshot.frame,
        "length_unit": snapshot.length_unit,
        "mass_unit": snapshot.mass_unit,
        "origin": snapshot.origin,
        "parent_integrator_id": parent_integrator_id,
        "parent_native_result_digest": parent_native_result_digest,
        "parent_run_index": parent_run_index,
        "schema": CONTINUATION_DIGEST_DOMAIN,
        "simulation_id": simulation_id,
        "time_scale": snapshot.time_scale,
        "time_unit": snapshot.time_unit,
        "unit_system_id": snapshot.unit_system_id,
    }
    serialized = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(
        CONTINUATION_DIGEST_DOMAIN.encode("ascii") + b"\x00" + serialized
    ).hexdigest()


def _gr15_newtonian_inputs(
    snapshot: StateSnapshot,
    plan: ForcePlan,
    spec: object,
) -> tuple[Any, Any, Any]:
    """Return exact native inputs only for a semantics-preserving GR15 route."""

    from jxplanetx.gr15 import GR15Spec

    if type(spec) is not GR15Spec:
        raise JXSimulationContractError(
            "the GR15 route requires an exact jxplanetx.GR15Spec"
        )
    if spec.checkpoint_epochs[0] != snapshot.epoch:
        raise JXSimulationContractError(
            "GR15 initial_epoch must exactly equal StateSnapshot.epoch"
        )
    if plan.backend.backend_id != "numpy" or plan.backend.device != "cpu":
        raise JXSimulationCompatibilityError(
            "native GR15 currently requires the explicit NumPy CPU backend"
        )
    if len(plan.models) != 1 or type(plan.models[0]) is not NewtonianPointMass:
        raise JXSimulationCompatibilityError(
            "native GR15 accepts exactly one NewtonianPointMass force term"
        )
    gravity = plan.models[0]
    if (
        gravity.source_ids != snapshot.body_ids
        or gravity.target_ids != snapshot.body_ids
    ):
        raise JXSimulationCompatibilityError(
            "native GR15 requires every body to be a mutual source and target "
            "in StateSnapshot body order"
        )
    if gravity.unit_system_id != snapshot.unit_system_id:
        raise JXSimulationCompatibilityError(
            "Newtonian force and state unit-system identifiers differ"
        )
    metadata = gravity.parameter_metadata[0]
    expected_units = f"{snapshot.length_unit}^3/{snapshot.time_unit}^2"
    if metadata.units != expected_units:
        raise JXSimulationCompatibilityError(
            "Newtonian GM metadata units do not match the state units"
        )
    lower = min(spec.initial_epoch, spec.final_epoch)
    upper = max(spec.initial_epoch, spec.final_epoch)
    if metadata.validity_start > lower or metadata.validity_end < upper:
        raise JXSimulationCompatibilityError(
            "Newtonian GM metadata validity does not cover the GR15 trajectory"
        )

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - engine extra owns NumPy
        raise JXSimulationCompatibilityError(
            "the NumPy CPU backend required by GR15 is unavailable"
        ) from exc

    body_count = len(snapshot.body_ids)
    for label, value in (
        ("positions", snapshot.positions),
        ("velocities", snapshot.velocities),
    ):
        if (
            type(value) is not np.ndarray
            or value.dtype != np.dtype(np.float64)
            or value.shape != (body_count, 3)
            or not value.flags.c_contiguous
            or not np.all(np.isfinite(value))
        ):
            raise JXSimulationCompatibilityError(
                f"native GR15 requires {label} to be finite C-contiguous NumPy "
                "float64 with shape (body_count, 3)"
            )
    gm = snapshot.gravitational_parameters
    if (
        type(gm) is not np.ndarray
        or gm.dtype != np.dtype(np.float64)
        or gm.shape != (body_count,)
        or not gm.flags.c_contiguous
        or not np.all(np.isfinite(gm))
        or np.any(gm <= 0.0)
    ):
        raise JXSimulationCompatibilityError(
            "native GR15 requires positive finite C-contiguous NumPy float64 GM"
        )
    massive = snapshot.massive
    if (
        type(massive) is not np.ndarray
        or massive.dtype != np.dtype(np.bool_)
        or massive.shape != (body_count,)
        or not massive.flags.c_contiguous
        or not np.all(massive)
    ):
        raise JXSimulationCompatibilityError(
            "native GR15 requires every body to be marked massive"
        )
    masses = snapshot.masses
    if (
        type(masses) is not np.ndarray
        or masses.dtype != np.dtype(np.float64)
        or masses.shape != (body_count,)
        or not masses.flags.c_contiguous
        or not np.all(np.isfinite(masses))
        or np.any(masses <= 0.0)
    ):
        raise JXSimulationCompatibilityError(
            "native GR15 requires positive finite masses for all mutual bodies"
        )
    radii = snapshot.radii
    if (
        type(radii) is not np.ndarray
        or radii.dtype != np.dtype(np.float64)
        or radii.shape != (body_count,)
        or not radii.flags.c_contiguous
        or not np.all(radii == 0.0)
    ):
        raise JXSimulationCompatibilityError(
            "native GR15 requires zero radii because it has no finite-radius "
            "collision detector equivalent to the generic force runtime"
        )
    return snapshot.positions, snapshot.velocities, gm


def _gr15_eih_plan_parameters(
    snapshot: StateSnapshot,
    plan: ForcePlan,
    spec: object,
) -> tuple[NewtonianPointMass, object]:
    """Validate the common CPU/CUDA EIH plan and construct native parameters."""

    from jxplanetx.gr15 import GR15Spec

    if type(spec) is not GR15Spec:
        raise JXSimulationContractError(
            "a native GR15-EIH1PN route requires an exact jxplanetx.GR15Spec"
        )
    if spec.checkpoint_epochs[0] != snapshot.epoch:
        raise JXSimulationContractError(
            "GR15-EIH1PN initial_epoch must exactly equal StateSnapshot.epoch"
        )

    if (
        len(plan.models) != 2
        or type(plan.models[0]) is not NewtonianPointMass
        or type(plan.models[1]) is not MutualEIH1PN
    ):
        raise JXSimulationCompatibilityError(
            "native GR15-EIH1PN requires exactly NewtonianPointMass followed "
            "by MutualEIH1PN"
        )
    gravity = plan.models[0]
    relativity = plan.models[1]
    if (
        gravity.source_ids != snapshot.body_ids
        or gravity.target_ids != snapshot.body_ids
    ):
        raise JXSimulationCompatibilityError(
            "native GR15-EIH1PN requires every body to be a mutual Newtonian "
            "source and target in StateSnapshot order"
        )
    if relativity.body_ids != snapshot.body_ids:
        raise JXSimulationCompatibilityError(
            "native GR15-EIH1PN requires EIH body_ids to equal StateSnapshot.body_ids"
        )
    if relativity.unit_system_id != snapshot.unit_system_id:
        raise JXSimulationCompatibilityError(
            "EIH force and state unit-system identifiers differ"
        )
    if gravity.unit_system_id != snapshot.unit_system_id:
        raise JXSimulationCompatibilityError(
            "Newtonian force and state unit-system identifiers differ"
        )
    if (
        snapshot.time_scale != "TDB"
        or snapshot.frame != "BARYCENTRIC_INERTIAL"
        or snapshot.origin != "SYSTEM_BARYCENTER"
    ):
        raise JXSimulationCompatibilityError(
            "native GR15-EIH1PN requires a TDB barycentric-inertial state "
            "with SYSTEM_BARYCENTER origin"
        )
    if snapshot.length_unit != "KM" or snapshot.time_unit != "S":
        raise JXSimulationCompatibilityError(
            "native GR15-EIH1PN requires length_unit='KM' and time_unit='S'"
        )
    expected = (
        ("speed_of_light", "KM/S"),
        ("maximum_compactness", "1"),
        ("maximum_speed_fraction_squared", "1"),
    )
    lower = min(spec.initial_epoch, spec.final_epoch)  # type: ignore[attr-defined]
    upper = max(spec.initial_epoch, spec.final_epoch)  # type: ignore[attr-defined]
    gravity_metadata = gravity.parameter_metadata[0]
    if gravity_metadata.units != "KM^3/S^2":
        raise JXSimulationCompatibilityError(
            "Newtonian GM metadata does not match the native KM/S unit contract"
        )
    if (
        gravity_metadata.validity_start > lower
        or gravity_metadata.validity_end < upper
    ):
        raise JXSimulationCompatibilityError(
            "Newtonian GM metadata validity does not cover the GR15 trajectory"
        )
    if len(relativity.parameter_metadata) != len(expected):
        raise JXSimulationCompatibilityError("EIH parameter metadata is incomplete")
    for metadata, (parameter_id, units) in zip(
        relativity.parameter_metadata,
        expected,
    ):
        if metadata.parameter_id != parameter_id or metadata.units != units:
            raise JXSimulationCompatibilityError(
                f"EIH {parameter_id} metadata does not match the native unit contract"
            )
        if metadata.validity_start > lower or metadata.validity_end < upper:
            raise JXSimulationCompatibilityError(
                f"EIH {parameter_id} validity does not cover the GR15 trajectory"
            )

    from jxplanetx.solar_system.eih_1pn import EIH1PNParameters

    parameters = EIH1PNParameters(
        speed_of_light_km_s=float(relativity.speed_of_light),
        maximum_compactness=float(relativity.maximum_compactness),
        maximum_speed_fraction_squared=float(
            relativity.maximum_speed_fraction_squared
        ),
    )
    return gravity, parameters


def _gr15_eih_inputs(
    snapshot: StateSnapshot,
    plan: ForcePlan,
    spec: object,
) -> tuple[Any, Any, Any, object]:
    """Return exact CPU-native inputs for Newtonian plus full mutual EIH 1PN."""

    gravity, parameters = _gr15_eih_plan_parameters(snapshot, plan, spec)
    positions, velocities, gm = _gr15_newtonian_inputs(
        snapshot,
        ForcePlan(
            plan_id=plan.plan_id,
            backend=plan.backend,
            models=(gravity,),
        ),
        spec,
    )
    return positions, velocities, gm, parameters


def _gr15_eih_cuda_inputs(
    snapshot: StateSnapshot,
    plan: ForcePlan,
    spec: object,
) -> tuple[Any, Any, Any, object]:
    """Return a one-system device batch for the persistent CUDA GR15 route."""

    _gravity, parameters = _gr15_eih_plan_parameters(snapshot, plan, spec)
    if plan.backend.backend_id != "cupy" or not plan.backend.device.startswith(
        ("cuda", "gpu")
    ):
        raise JXSimulationCompatibilityError(
            "native CUDA GR15-EIH1PN requires an explicit CuPy CUDA backend"
        )
    try:
        backend = resolve_backend(plan.backend)
    except BackendUnavailableError as exc:
        raise JXSimulationCompatibilityError(
            "the explicitly requested CUDA backend is unavailable"
        ) from exc
    with backend.activate():
        positions, velocities, gm, masses, radii, massive = _validate_state(
            backend,
            snapshot,
        )
        body_count = len(snapshot.body_ids)
        if not 2 <= body_count <= 32:
            raise JXSimulationCompatibilityError(
                "native CUDA GR15-EIH1PN supports exactly 2--32 bodies"
            )
        if backend.scalar_bool(
            backend.xp.any(~massive),
            "CUDA GR15-EIH massive-body check",
        ):
            raise JXSimulationCompatibilityError(
                "native CUDA GR15-EIH1PN requires every body to be marked massive"
            )
        if backend.scalar_bool(
            backend.xp.any(masses <= backend.xp.float64(0.0)),
            "CUDA GR15-EIH mass check",
        ):
            raise JXSimulationCompatibilityError(
                "native CUDA GR15-EIH1PN requires positive masses for all bodies"
            )
        if backend.scalar_bool(
            backend.xp.any(radii != backend.xp.float64(0.0)),
            "CUDA GR15-EIH radius check",
        ):
            raise JXSimulationCompatibilityError(
                "native CUDA GR15-EIH1PN requires zero radii because its kernel "
                "has no generic finite-radius collision semantics"
            )
        return (
            positions[backend.xp.newaxis, :, :],
            velocities[backend.xp.newaxis, :, :],
            gm,
            parameters,
        )


class JXSimulation:
    """Persistent owner of one state/force definition and successful run log.

    The object serializes integrations and appends to its run log only after a
    complete successful result. ``integrate`` always starts from the retained
    initial snapshot. ``continue_from`` instead requires an exact parent run
    and constructs a provenance-bound endpoint snapshot; continuation is never
    inferred from call order.
    """

    __slots__ = (
        "_simulation_id",
        "_snapshot",
        "_force_plan",
        "_runs",
        "_lock",
    )

    def __init__(
        self,
        simulation_id: str,
        snapshot: StateSnapshot,
        force_plan: ForcePlan,
    ) -> None:
        self._simulation_id = _require_simulation_id(simulation_id)
        if type(snapshot) is not StateSnapshot:
            raise JXSimulationContractError("snapshot must be an exact StateSnapshot")
        if type(force_plan) is not ForcePlan:
            raise JXSimulationContractError("force_plan must be an exact ForcePlan")
        try:
            backend = resolve_backend(force_plan.backend)
        except BackendUnavailableError:
            # Preserve construction for an explicitly unavailable backend so
            # route compatibility can still fail for the requested reason.
            # No execution can occur, so no mutable caller buffer can enter a
            # successful run. A usable backend always receives owned copies.
            self._snapshot = snapshot
            self._force_plan = force_plan
        except Exception as exc:
            raise JXSimulationContractError(
                f"simulation backend contract is invalid: {exc}"
            ) from exc
        else:
            try:
                with backend.activate():
                    validated_state = _validate_state(backend, snapshot)
                    self._snapshot = _copy_initial_snapshot(
                        backend,
                        snapshot,
                        validated_state,
                    )
                    self._force_plan = _copy_force_plan(backend, force_plan)
            except Exception as exc:
                raise JXSimulationContractError(
                    "simulation inputs could not be retained with backend-native "
                    f"custody: {exc}"
                ) from exc
        self._runs: list[JXSimulationRun] = []
        self._lock = RLock()

    @property
    def simulation_id(self) -> str:
        return self._simulation_id

    @property
    def snapshot(self) -> StateSnapshot:
        return self._snapshot

    @property
    def force_plan(self) -> ForcePlan:
        return self._force_plan

    @property
    def runs(self) -> tuple[JXSimulationRun, ...]:
        with self._lock:
            return tuple(self._runs)

    @property
    def run_count(self) -> int:
        with self._lock:
            return len(self._runs)

    @property
    def last_run(self) -> JXSimulationRun | None:
        with self._lock:
            return self._runs[-1] if self._runs else None

    def integrate(self, integrator_id: str, spec: object) -> JXSimulationRun:
        """Run one explicitly selected compatible integrator from the initial state."""

        if type(integrator_id) is not str or not integrator_id:
            raise JXSimulationContractError("integrator_id must be a nonempty string")
        with self._lock:
            run = self._integrate_locked(
                self._snapshot,
                integrator_id,
                spec,
                continuation=None,
            )
            self._runs.append(run)
            return run

    def prepare_continuation(self, parent_run_index: int) -> JXSimulationContinuation:
        """Return a read-only, provenance-bound NumPy endpoint snapshot."""

        with self._lock:
            return self._prepare_continuation_locked(parent_run_index)

    def continue_from(
        self,
        parent_run_index: int,
        integrator_id: str,
        spec: object,
    ) -> JXSimulationRun:
        """Integrate explicitly from one retained run endpoint.

        The new run is appended only after complete success. Its initial state
        is bound to the parent result digest and endpoint bytes.
        """

        if type(integrator_id) is not str or not integrator_id:
            raise JXSimulationContractError("integrator_id must be a nonempty string")
        with self._lock:
            continuation = self._prepare_continuation_locked(parent_run_index)
            run = self._integrate_locked(
                continuation.snapshot,
                integrator_id,
                spec,
                continuation=continuation,
            )
            self._runs.append(run)
            return run

    def _prepare_continuation_locked(
        self,
        parent_run_index: int,
    ) -> JXSimulationContinuation:
        if type(parent_run_index) is not int or parent_run_index < 0:
            raise JXSimulationContractError(
                "parent_run_index must be a nonnegative built-in integer"
            )
        if parent_run_index >= len(self._runs):
            raise JXSimulationContractError(
                "parent_run_index does not identify a successful retained run"
            )
        parent = self._runs[parent_run_index]
        if parent.backend_id != "numpy" or self._force_plan.backend.backend_id != "numpy":
            raise JXSimulationCompatibilityError(
                "continuation v1 supports only NumPy/CPU results with host content digests"
            )
        if parent.native_result_digest is None:
            raise JXSimulationCompatibilityError(
                "continuation requires a parent with a retained content digest"
            )
        if parent.force_model_ids != tuple(
            model.model_id for model in self._force_plan.models
        ):
            raise JXSimulationContractError(
                "parent force-model identity differs from the retained force plan"
            )
        backend = resolve_backend(self._force_plan.backend)
        with backend.activate():
            provisional = replace(
                self._snapshot,
                snapshot_id=(
                    f"{self._snapshot.snapshot_id}.continuation.{parent_run_index}"
                ),
                epoch=parent.checkpoint_epochs[-1],
                positions=backend.xp.ascontiguousarray(
                    parent.final_positions, dtype=backend.xp.float64
                ),
                velocities=backend.xp.ascontiguousarray(
                    parent.final_velocities, dtype=backend.xp.float64
                ),
            )
            validated = _validate_state(backend, provisional)
            owned = _copy_initial_snapshot(backend, provisional, validated)
        state_sha256 = _continuation_state_sha256(
            simulation_id=self._simulation_id,
            parent=parent,
            snapshot=owned,
        )
        bound = replace(
            owned,
            snapshot_id=(
                f"{self._snapshot.snapshot_id}.continuation."
                f"{parent_run_index}.{state_sha256[:16]}"
            ),
            provenance=Provenance(
                source_id=(
                    f"jx-continuation:{self._simulation_id}:{parent_run_index}"
                ),
                citation=(
                    f"JX continuation of run {parent_run_index} "
                    f"({parent.integrator_id})"
                ),
                version="1",
                sha256=state_sha256,
            ),
        )
        return JXSimulationContinuation(
            simulation_id=self._simulation_id,
            parent_run_index=parent_run_index,
            parent_integrator_id=parent.integrator_id,
            parent_native_result_digest=parent.native_result_digest,
            continuation_state_sha256=state_sha256,
            snapshot=bound,
            force_model_ids=parent.force_model_ids,
        )

    def _integrate_locked(
        self,
        snapshot: StateSnapshot,
        integrator_id: str,
        spec: object,
        *,
        continuation: JXSimulationContinuation | None,
    ) -> JXSimulationRun:
        run_index = len(self._runs)
        if integrator_id == ADAPTIVE_RKF78_METHOD_ID:
            if type(spec) is not AdaptiveRKF78Spec:
                raise JXSimulationContractError(
                    "the adaptive RKF78 route requires an exact AdaptiveRKF78Spec"
                )
            native = integrate_trajectory(snapshot, self._force_plan, spec)
            return self._wrap_rkf78(run_index, native, continuation)
        if integrator_id == GR15_NEWTONIAN_METHOD_ID:
            positions, velocities, gm = _gr15_newtonian_inputs(
                snapshot, self._force_plan, spec
            )
            from jxplanetx.gr15 import integrate_gr15

            native = integrate_gr15(positions, velocities, gm, spec)
            return self._wrap_gr15(run_index, native, continuation)
        if integrator_id == GR15_EIH1PN_METHOD_ID:
            positions, velocities, gm, parameters = _gr15_eih_inputs(
                snapshot, self._force_plan, spec
            )
            from jxplanetx.gr15_eih_1pn import integrate_gr15_eih_1pn

            native = integrate_gr15_eih_1pn(
                positions, velocities, gm, spec, parameters
            )
            return self._wrap_gr15(run_index, native, continuation)
        if integrator_id == GR15_EIH1PN_CUDA_METHOD_ID:
            positions, velocities, gm, parameters = _gr15_eih_cuda_inputs(
                snapshot, self._force_plan, spec
            )
            from jxplanetx.gr15_eih_1pn_cuda import (
                integrate_gr15_eih_1pn_cuda_batch,
            )

            native = integrate_gr15_eih_1pn_cuda_batch(
                positions, velocities, gm, spec, parameters
            )
            return self._wrap_gr15_cuda(run_index, native, continuation)
        raise JXSimulationCompatibilityError(
            f"unsupported unified integrator route {integrator_id!r}"
        )

    def _wrap_rkf78(
        self,
        run_index: int,
        native: TrajectoryResult,
        continuation: JXSimulationContinuation | None,
    ) -> JXSimulationRun:
        return JXSimulationRun(
            simulation_id=self._simulation_id,
            run_index=run_index,
            integrator_id=native.method_id,
            force_model_ids=native.force_model_ids,
            checkpoint_epochs=native.checkpoint_epochs,
            checkpoint_positions=tuple(
                checkpoint.positions for checkpoint in native.checkpoints
            ),
            checkpoint_velocities=tuple(
                checkpoint.velocities for checkpoint in native.checkpoints
            ),
            attempted_steps=native.attempted_steps,
            accepted_steps=native.accepted_steps,
            rejected_steps=native.rejected_steps,
            force_evaluations=native.force_evaluations,
            backend_id=native.backend_id,
            device=native.device,
            native_result=native,
            native_result_digest=native.result_content_sha256,
            native_result_digest_scope=(
                "FULL_RETAINED_RESULT_CONTENT"
                if native.result_content_sha256 is not None
                else "NO_HOST_CONTENT_DIGEST_DEVICE_RESULT"
            ),
            parent_run_index=(
                None if continuation is None else continuation.parent_run_index
            ),
            parent_native_result_digest=(
                None
                if continuation is None
                else continuation.parent_native_result_digest
            ),
            continuation_state_sha256=(
                None
                if continuation is None
                else continuation.continuation_state_sha256
            ),
        )

    def _wrap_gr15(
        self,
        run_index: int,
        native: object,
        continuation: JXSimulationContinuation | None,
    ) -> JXSimulationRun:
        return JXSimulationRun(
            simulation_id=self._simulation_id,
            run_index=run_index,
            integrator_id=native.method_id,  # type: ignore[attr-defined]
            force_model_ids=tuple(
                model.model_id for model in self._force_plan.models
            ),
            checkpoint_epochs=native.checkpoint_epochs,  # type: ignore[attr-defined]
            checkpoint_positions=tuple(
                native.checkpoint_positions[index]  # type: ignore[attr-defined]
                for index in range(len(native.checkpoint_epochs))  # type: ignore[attr-defined]
            ),
            checkpoint_velocities=tuple(
                native.checkpoint_velocities[index]  # type: ignore[attr-defined]
                for index in range(len(native.checkpoint_epochs))  # type: ignore[attr-defined]
            ),
            attempted_steps=native.attempted_steps,  # type: ignore[attr-defined]
            accepted_steps=native.accepted_steps,  # type: ignore[attr-defined]
            rejected_steps=native.rejected_steps,  # type: ignore[attr-defined]
            force_evaluations=native.force_evaluations,  # type: ignore[attr-defined]
            backend_id="numpy",
            device="cpu",
            native_result=native,
            native_result_digest=native.replay_digest,  # type: ignore[attr-defined]
            native_result_digest_scope="GR15_NATIVE_REPLAY_CONTENT",
            parent_run_index=(
                None if continuation is None else continuation.parent_run_index
            ),
            parent_native_result_digest=(
                None
                if continuation is None
                else continuation.parent_native_result_digest
            ),
            continuation_state_sha256=(
                None
                if continuation is None
                else continuation.continuation_state_sha256
            ),
        )

    def _wrap_gr15_cuda(
        self,
        run_index: int,
        native: object,
        continuation: JXSimulationContinuation | None,
    ) -> JXSimulationRun:
        audit = native.system_audits[0]  # type: ignore[attr-defined]
        checkpoint_count = len(native.checkpoint_epochs)  # type: ignore[attr-defined]
        device_index = int(native.positions_km.device.id)  # type: ignore[attr-defined]
        return JXSimulationRun(
            simulation_id=self._simulation_id,
            run_index=run_index,
            integrator_id=native.method_id,  # type: ignore[attr-defined]
            force_model_ids=tuple(
                model.model_id for model in self._force_plan.models
            ),
            checkpoint_epochs=native.checkpoint_epochs,  # type: ignore[attr-defined]
            checkpoint_positions=tuple(
                native.checkpoint_positions_km[0, index]  # type: ignore[attr-defined]
                for index in range(checkpoint_count)
            ),
            checkpoint_velocities=tuple(
                native.checkpoint_velocities_km_s[0, index]  # type: ignore[attr-defined]
                for index in range(checkpoint_count)
            ),
            attempted_steps=audit.attempted_steps,
            accepted_steps=audit.accepted_steps,
            rejected_steps=audit.rejected_steps,
            force_evaluations=audit.force_evaluations,
            backend_id="cupy",
            device=f"cuda:{device_index}",
            native_result=native,
            native_result_digest=None,
            native_result_digest_scope="NO_HOST_CONTENT_DIGEST_DEVICE_RESULT",
            parent_run_index=(
                None if continuation is None else continuation.parent_run_index
            ),
            parent_native_result_digest=(
                None
                if continuation is None
                else continuation.parent_native_result_digest
            ),
            continuation_state_sha256=(
                None
                if continuation is None
                else continuation.continuation_state_sha256
            ),
        )


__all__ = [
    "CONTINUATION_DIGEST_ALGORITHM",
    "CONTINUATION_DIGEST_DOMAIN",
    "GR15_EIH1PN_CUDA_METHOD_ID",
    "GR15_EIH1PN_METHOD_ID",
    "GR15_NEWTONIAN_METHOD_ID",
    "JXSimulation",
    "JXSimulationCompatibilityError",
    "JXSimulationContinuation",
    "JXSimulationContractError",
    "JXSimulationError",
    "JXSimulationRun",
]
