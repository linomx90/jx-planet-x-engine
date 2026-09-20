"""Ordered, nonauthorizing force-plan evaluation for JX engine snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .backends import ArrayBackend, resolve_backend
from .contracts import (
    CannonballSRP,
    ForcePlan,
    NewtonianPointMass,
    RestrictedStaticCentral1PN,
    StateSnapshot,
)
from .forces import (
    COLLISION_POLICY_ERROR,
    SINGULARITY_POLICY_ERROR,
    ForceContractError,
    ForceDomainError,
    cannonball_srp_acceleration,
    newtonian_point_mass_acceleration,
    restricted_static_central_1pn_acceleration,
)


MODEL_OUTPUT = "MODEL_OUTPUT"
FORCE_EVALUATION_ONLY = "FORCE_EVALUATION_ONLY"
DETERMINISM_SCOPE = "SAME_RUNTIME_DEVICE"
DETERMINISM_LIMITATION = (
    "repeatability is limited to the same backend, device, software stack, "
    "force order, body order, and tile size; no cross-device bitwise guarantee"
)
_CANONICAL_MODEL_ORDER = {
    NewtonianPointMass.MODEL_ID: 0,
    RestrictedStaticCentral1PN.MODEL_ID: 1,
    CannonballSRP.MODEL_ID: 2,
}


class EvaluationError(ValueError):
    """A snapshot and force plan cannot be evaluated together."""


class DuplicateForceError(EvaluationError):
    """A model identifier appears more than once in one evaluation."""


class UnsupportedForceError(EvaluationError):
    """A plan contains a declaration without an implemented runtime kernel."""


@dataclass(frozen=True)
class StateMetadataBinding:
    """Non-array state metadata bound into every result and ledger entry."""

    snapshot_id: str
    epoch: float
    unit_system_id: str
    time_scale: str
    frame: str
    origin: str
    axes: str
    length_unit: str
    time_unit: str
    mass_unit: str
    body_ids: tuple[str, ...]
    provenance_source_id: str
    provenance_citation: str
    provenance_version: str
    provenance_sha256: str


@dataclass(frozen=True)
class ForceLedgerEntry:
    """One force term in exact accumulation order."""

    order: int
    model_id: str
    role: str
    source_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    state_metadata: StateMetadataBinding
    backend_spec: Any
    tile_size: int
    assumptions: tuple[str, ...]
    determinism_scope: str = DETERMINISM_SCOPE
    evidence_class: str = MODEL_OUTPUT
    registry_authorized: bool = False
    qualification_authorized: bool = False

    @property
    def qualified(self) -> bool:
        return False


@dataclass(frozen=True, eq=False)
class ForceContribution:
    """Backend-native acceleration from one ledger entry."""

    ledger: ForceLedgerEntry
    acceleration: Any

    @property
    def model_id(self) -> str:
        return self.ledger.model_id

    @property
    def evidence_class(self) -> str:
        return MODEL_OUTPUT

    @property
    def registry_authorized(self) -> bool:
        return False

    @property
    def qualification_authorized(self) -> bool:
        return False

    @property
    def qualified(self) -> bool:
        return False


@dataclass(frozen=True, eq=False)
class ForceEvaluationResult:
    """Acceleration-only result; no integration or trajectory is implied."""

    snapshot_id: str
    plan_id: str
    backend_id: str
    device: str
    dtype: str
    state_metadata: StateMetadataBinding
    backend_spec: Any
    tile_size: int
    total_acceleration: Any
    contributions: tuple[ForceContribution, ...]
    ledger: tuple[ForceLedgerEntry, ...]
    singularity_policy: str = SINGULARITY_POLICY_ERROR
    collision_policy: str = COLLISION_POLICY_ERROR
    scope: str = FORCE_EVALUATION_ONLY
    evidence_class: str = MODEL_OUTPUT
    registry_authorized: bool = False
    qualification_authorized: bool = False
    determinism_scope: str = DETERMINISM_SCOPE

    @property
    def acceleration(self) -> Any:
        return self.total_acceleration

    @property
    def accelerations(self) -> Any:
        return self.total_acceleration

    @property
    def applied_model_ids(self) -> tuple[str, ...]:
        return tuple(entry.model_id for entry in self.ledger)

    @property
    def integrated(self) -> bool:
        return False

    @property
    def qualified(self) -> bool:
        return False


def _has_any(backend: ArrayBackend, expression: Any, label: str) -> bool:
    return backend.scalar_bool(backend.xp.any(expression), label)


def _validate_vector_buffer(
    backend: ArrayBackend,
    value: object,
    label: str,
    body_count: int,
) -> Any:
    array = backend.require_native_array(value, label)
    if array.dtype != backend.float64:
        raise EvaluationError(f"{label} must have dtype float64")
    if array.ndim != 2 or array.shape != (body_count, 3):
        raise EvaluationError(f"{label} must have shape (body_count, 3)")
    if _has_any(backend, ~backend.xp.isfinite(array), f"{label} finiteness check"):
        raise ForceDomainError(f"{label} must contain only finite values")
    return array


def _validate_scalar_buffer(
    backend: ArrayBackend,
    value: object,
    label: str,
    body_count: int,
    *,
    nonnegative: bool,
) -> Any:
    array = backend.require_native_array(value, label)
    if array.dtype != backend.float64:
        raise EvaluationError(f"{label} must have dtype float64")
    if array.ndim != 1 or array.shape != (body_count,):
        raise EvaluationError(f"{label} must have shape (body_count,)")
    if _has_any(backend, ~backend.xp.isfinite(array), f"{label} finiteness check"):
        raise ForceDomainError(f"{label} must contain only finite values")
    if nonnegative and _has_any(backend, array < 0.0, f"{label} sign check"):
        raise ForceDomainError(f"{label} cannot contain negative values")
    return array


def _validate_state(
    backend: ArrayBackend,
    snapshot: StateSnapshot,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    body_count = len(snapshot.body_ids)
    if body_count == 0:
        raise EvaluationError("StateSnapshot.body_ids cannot be empty")
    positions = _validate_vector_buffer(
        backend, snapshot.positions, "positions", body_count
    )
    velocities = _validate_vector_buffer(
        backend, snapshot.velocities, "velocities", body_count
    )
    gravitational_parameters = _validate_scalar_buffer(
        backend,
        snapshot.gravitational_parameters,
        "gravitational_parameters",
        body_count,
        nonnegative=True,
    )
    masses = _validate_scalar_buffer(
        backend, snapshot.masses, "masses", body_count, nonnegative=True
    )
    radii = _validate_scalar_buffer(
        backend, snapshot.radii, "radii", body_count, nonnegative=True
    )
    massive = backend.require_native_array(snapshot.massive, "massive")
    if massive.dtype != backend.bool_:
        raise EvaluationError("massive must have boolean dtype")
    if massive.ndim != 1 or massive.shape != (body_count,):
        raise EvaluationError("massive must have shape (body_count,)")
    if _has_any(
        backend,
        massive & (gravitational_parameters <= 0.0),
        "massive-body GM check",
    ):
        raise ForceDomainError("every body marked massive must have positive GM")
    return positions, velocities, gravitational_parameters, masses, radii, massive


def _indices_for_ids(
    snapshot: StateSnapshot,
    body_ids: tuple[str, ...],
    label: str,
) -> tuple[int, ...]:
    if type(body_ids) is not tuple or not body_ids:
        raise EvaluationError(f"{label} must be a nonempty immutable tuple")
    if len(set(body_ids)) != len(body_ids):
        raise EvaluationError(f"{label} contains duplicate body identifiers")
    index_by_id = {
        body_id: index for index, body_id in enumerate(snapshot.body_ids)
    }
    try:
        return tuple(index_by_id[body_id] for body_id in body_ids)
    except KeyError as exc:
        raise EvaluationError(f"{label} references a body absent from the snapshot") from exc


def _mask_for_indices(
    backend: ArrayBackend,
    body_count: int,
    indices: tuple[int, ...],
) -> Any:
    mask = backend.xp.zeros(body_count, dtype=backend.xp.bool_)
    for index in indices:
        mask[index] = True
    return mask


def _require_massive_sources(
    backend: ArrayBackend,
    massive: Any,
    indices: tuple[int, ...],
    label: str,
) -> None:
    for index in indices:
        if not backend.scalar_bool(massive[index], f"{label} massive-body check"):
            raise EvaluationError(f"{label} must identify bodies marked massive")


def _require_massless_targets(
    backend: ArrayBackend,
    massive: Any,
    indices: tuple[int, ...],
    label: str,
) -> None:
    for index in indices:
        if backend.scalar_bool(massive[index], f"{label} massless-target check"):
            raise EvaluationError(f"{label} must identify massless/no-backreaction targets")


def _validate_model_sequence(models: tuple[object, ...]) -> None:
    exact_types = (NewtonianPointMass, RestrictedStaticCentral1PN, CannonballSRP)
    seen: set[str] = set()
    ranks: list[int] = []
    for index, model in enumerate(models):
        if type(model) not in exact_types:
            raise UnsupportedForceError(
                f"models[{index}] is not an exact implemented JX force-config type"
            )
        model_id = model.model_id  # type: ignore[union-attr]
        if model_id in seen:
            raise DuplicateForceError(f"force model already applied: {model_id}")
        seen.add(model_id)
        ranks.append(_CANONICAL_MODEL_ORDER[model_id])
    if ranks != sorted(ranks):
        raise EvaluationError(
            "force models must follow canonical order: Newtonian, restricted 1PN, SRP"
        )
    if any(rank > 0 for rank in ranks) and NewtonianPointMass.MODEL_ID not in seen:
        raise EvaluationError("correction-only force models require a Newtonian base term")


def _validate_dependencies(models: tuple[object, ...]) -> NewtonianPointMass:
    base = next((model for model in models if type(model) is NewtonianPointMass), None)
    if type(base) is not NewtonianPointMass:
        raise EvaluationError("a force plan requires one NewtonianPointMass base term")
    base_sources = set(base.source_ids)
    base_targets = set(base.target_ids)
    for model in models:
        if type(model) is RestrictedStaticCentral1PN:
            if model.central_source_id not in base_sources:
                raise EvaluationError("the 1PN central source must be a Newtonian source")
            if not set(model.target_ids).issubset(base_targets):
                raise EvaluationError("all 1PN targets must also be Newtonian targets")
        elif type(model) is CannonballSRP:
            if model.radiation_source_id not in base_sources:
                raise EvaluationError("the SRP radiation source must be a Newtonian source")
            if not set(model.target_ids).issubset(base_targets):
                raise EvaluationError("all SRP targets must also be Newtonian targets")
    return base


def _expected_metadata(
    snapshot: StateSnapshot,
    model: object,
) -> tuple[tuple[str, str], ...]:
    length = snapshot.length_unit
    time = snapshot.time_unit
    mass = snapshot.mass_unit
    if type(model) is NewtonianPointMass:
        return (("state.gravitational_parameters", f"{length}^3/{time}^2"),)
    if type(model) is RestrictedStaticCentral1PN:
        return (
            ("speed_of_light", f"{length}/{time}"),
            ("maximum_compactness", "1"),
            ("maximum_speed_fraction_squared", "1"),
        )
    if type(model) is CannonballSRP:
        return (
            ("reference_pressure", f"{mass}/({length}*{time}^2)"),
            ("reference_distance", length),
            ("area_to_mass", f"{length}^2/{mass}"),
            ("radiation_pressure_coefficient", "1"),
        )
    raise UnsupportedForceError("force model has no parameter-unit contract")


def _validate_parameter_contracts(snapshot: StateSnapshot, plan: ForcePlan) -> None:
    """Close unit-system, parameter-unit, provenance, and epoch validity before arithmetic."""

    for model in plan.models:
        if model.unit_system_id != snapshot.unit_system_id:  # type: ignore[union-attr]
            raise EvaluationError(
                f"{model.model_id} unit_system_id differs from the StateSnapshot"  # type: ignore[union-attr]
            )
        expected = _expected_metadata(snapshot, model)
        metadata = model.parameter_metadata  # type: ignore[union-attr]
        if len(metadata) != len(expected):
            raise EvaluationError(f"{model.model_id} parameter metadata is incomplete")  # type: ignore[union-attr]
        for index, (entry, (parameter_id, units)) in enumerate(zip(metadata, expected)):
            if entry.parameter_id != parameter_id:
                raise EvaluationError(
                    f"{model.model_id} parameter_metadata[{index}] has the wrong identifier"  # type: ignore[union-attr]
                )
            if entry.units != units:
                raise EvaluationError(
                    f"{parameter_id} units must be exactly {units!r} for this StateSnapshot"
                )
            if entry.validity_start is None or entry.validity_end is None:
                raise EvaluationError(
                    f"{parameter_id} requires a closed validity interval"
                )
            if not entry.validity_start <= snapshot.epoch <= entry.validity_end:
                raise EvaluationError(
                    f"{parameter_id} validity interval does not include the snapshot epoch"
                )
            # ParameterMetadata construction already requires exact Provenance.
            if entry.provenance.sha256 == "0" * 64:
                raise EvaluationError(f"{parameter_id} provenance digest is a placeholder")


def _validate_restricted_frame(snapshot: StateSnapshot, models: tuple[object, ...]) -> None:
    for model in models:
        if type(model) is RestrictedStaticCentral1PN:
            if snapshot.origin != model.central_source_id:
                raise EvaluationError(
                    "restricted 1PN requires StateSnapshot.origin equal to central_source_id"
                )
            if snapshot.frame != "CENTRAL_BODY_INERTIAL":
                raise EvaluationError(
                    "restricted 1PN requires frame='CENTRAL_BODY_INERTIAL'"
                )


def _validate_restricted_central_position(
    backend: ArrayBackend,
    snapshot: StateSnapshot,
    positions: Any,
    models: tuple[object, ...],
) -> None:
    """Require coordinate zero to agree with a declared central-body origin."""

    for model in models:
        if type(model) is RestrictedStaticCentral1PN:
            central_index = snapshot.index_of(model.central_source_id)
            if _has_any(
                backend,
                positions[central_index] != backend.xp.float64(0.0),
                "restricted 1PN central-position check",
            ):
                raise EvaluationError(
                    "restricted 1PN central_source_id must be exactly at coordinate zero"
                )


def _validate_target_parameter(
    backend: ArrayBackend,
    value: object,
    label: str,
    target_count: int,
) -> Any:
    array = backend.require_native_array(value, label)
    if array.dtype != backend.float64:
        raise EvaluationError(f"{label} must have dtype float64")
    if array.ndim != 1 or array.shape != (target_count,):
        raise EvaluationError(f"{label} must have shape (len(target_ids),)")
    if _has_any(backend, ~backend.xp.isfinite(array), f"{label} finiteness check"):
        raise ForceDomainError(f"{label} must contain only finite values")
    if _has_any(backend, array < 0.0, f"{label} sign check"):
        raise ForceDomainError(f"{label} cannot contain negative values")
    return array


def _expand_target_parameter(
    backend: ArrayBackend,
    body_count: int,
    target_indices: tuple[int, ...],
    values: Any,
) -> Any:
    expanded = backend.xp.zeros(body_count, dtype=backend.xp.float64)
    for parameter_index, body_index in enumerate(target_indices):
        expanded[body_index] = values[parameter_index]
    return expanded


def _ensure_finite_acceleration(
    backend: ArrayBackend,
    acceleration: Any,
    label: str,
    body_count: int,
) -> None:
    if (
        acceleration.dtype != backend.float64
        or acceleration.ndim != 2
        or acceleration.shape != (body_count, 3)
    ):
        raise EvaluationError(f"{label} kernel returned an invalid acceleration array")
    if _has_any(
        backend,
        ~backend.xp.isfinite(acceleration),
        f"{label} result finiteness check",
    ):
        raise ForceDomainError(f"{label} produced nonfinite acceleration")


def evaluate_force_plan(
    snapshot: StateSnapshot,
    plan: ForcePlan,
) -> ForceEvaluationResult:
    """Evaluate one immutable force plan without advancing the snapshot.

    Input numerical buffers must already be native arrays of the explicitly
    requested backend.  They are validated in place and never converted,
    copied to another device, or mutated.
    """

    if type(snapshot) is not StateSnapshot:
        raise EvaluationError("snapshot must be an exact StateSnapshot")
    if type(plan) is not ForcePlan:
        raise EvaluationError("plan must be an exact ForcePlan")
    if (
        plan.evidence_class != MODEL_OUTPUT
        or plan.registry_authorized
        or plan.qualification_authorized
    ):
        raise EvaluationError("the force evaluator accepts only nonauthorizing MODEL_OUTPUT plans")
    _validate_model_sequence(plan.models)
    _validate_dependencies(plan.models)
    _validate_parameter_contracts(snapshot, plan)
    _validate_restricted_frame(snapshot, plan.models)
    backend = resolve_backend(plan.backend)
    tile_size = plan.backend.tile_size
    state_metadata = StateMetadataBinding(
        snapshot_id=snapshot.snapshot_id,
        epoch=snapshot.epoch,
        unit_system_id=snapshot.unit_system_id,
        time_scale=snapshot.time_scale,
        frame=snapshot.frame,
        origin=snapshot.origin,
        axes=snapshot.axes,
        length_unit=snapshot.length_unit,
        time_unit=snapshot.time_unit,
        mass_unit=snapshot.mass_unit,
        body_ids=snapshot.body_ids,
        provenance_source_id=snapshot.provenance.source_id,
        provenance_citation=snapshot.provenance.citation,
        provenance_version=snapshot.provenance.version,
        provenance_sha256=snapshot.provenance.sha256,
    )

    with backend.activate():
        (
            positions,
            velocities,
            gravitational_parameters,
            _masses,
            radii,
            massive,
        ) = _validate_state(backend, snapshot)
        body_count = len(snapshot.body_ids)
        _validate_restricted_central_position(
            backend, snapshot, positions, plan.models
        )
        total = backend.xp.zeros((body_count, 3), dtype=backend.xp.float64)
        ledger: list[ForceLedgerEntry] = []
        contributions: list[ForceContribution] = []
        applied: set[str] = set()

        for order, model in enumerate(plan.models):
            if model.model_id in applied:  # type: ignore[union-attr]
                raise DuplicateForceError(f"force model already applied: {model.model_id}")  # type: ignore[union-attr]
            applied.add(model.model_id)  # type: ignore[union-attr]

            if type(model) is NewtonianPointMass:
                source_indices = _indices_for_ids(snapshot, model.source_ids, "Newtonian source_ids")
                target_indices = _indices_for_ids(snapshot, model.target_ids, "Newtonian target_ids")
                _require_massive_sources(backend, massive, source_indices, "Newtonian source_ids")
                source_mask = _mask_for_indices(backend, body_count, source_indices)
                target_mask = _mask_for_indices(backend, body_count, target_indices)
                acceleration = newtonian_point_mass_acceleration(
                    backend=backend,
                    positions=positions,
                    gravitational_parameters=gravitational_parameters,
                    radii=radii,
                    source_mask=source_mask,
                    target_mask=target_mask,
                    tile_size=tile_size,
                    singularity_policy=SINGULARITY_POLICY_ERROR,
                    collision_policy=COLLISION_POLICY_ERROR,
                )
                role = "NEWTONIAN_BASE"
                source_ids = model.source_ids
                target_ids = model.target_ids
                assumptions = (
                    "direct unsoftened point masses",
                    "finite-radius contact and singular coincidence are errors",
                    DETERMINISM_LIMITATION,
                )
            elif type(model) is RestrictedStaticCentral1PN:
                central_index = _indices_for_ids(
                    snapshot, (model.central_source_id,), "1PN central_source_id"
                )[0]
                target_indices = _indices_for_ids(snapshot, model.target_ids, "1PN target_ids")
                _require_massive_sources(
                    backend, massive, (central_index,), "1PN central_source_id"
                )
                _require_massless_targets(backend, massive, target_indices, "1PN target_ids")
                source_mask = _mask_for_indices(backend, body_count, (central_index,))
                target_mask = _mask_for_indices(backend, body_count, target_indices)
                acceleration = restricted_static_central_1pn_acceleration(
                    backend=backend,
                    positions=positions,
                    velocities=velocities,
                    gravitational_parameters=gravitational_parameters,
                    radii=radii,
                    source_mask=source_mask,
                    target_mask=target_mask,
                    central_index=central_index,
                    speed_of_light=model.speed_of_light,
                    maximum_compactness=model.maximum_compactness,
                    maximum_speed_fraction_squared=model.maximum_speed_fraction_squared,
                    singularity_policy=SINGULARITY_POLICY_ERROR,
                    collision_policy=COLLISION_POLICY_ERROR,
                )
                role = "CORRECTION_ONLY"
                source_ids = (model.central_source_id,)
                target_ids = model.target_ids
                assumptions = (
                    "static central source in CENTRAL_BODY_INERTIAL coordinates",
                    "massless targets with no backreaction",
                    "restricted Schwarzschild 1PN correction only",
                    DETERMINISM_LIMITATION,
                )
            elif type(model) is CannonballSRP:
                source_index = _indices_for_ids(
                    snapshot, (model.radiation_source_id,), "SRP radiation_source_id"
                )[0]
                target_indices = _indices_for_ids(snapshot, model.target_ids, "SRP target_ids")
                _require_massive_sources(
                    backend, massive, (source_index,), "SRP radiation_source_id"
                )
                _require_massless_targets(backend, massive, target_indices, "SRP target_ids")
                source_mask = _mask_for_indices(backend, body_count, (source_index,))
                target_mask = _mask_for_indices(backend, body_count, target_indices)
                area_to_mass = _validate_target_parameter(
                    backend, model.area_to_mass, "area_to_mass", len(target_indices)
                )
                coefficient = _validate_target_parameter(
                    backend,
                    model.radiation_pressure_coefficient,
                    "radiation_pressure_coefficient",
                    len(target_indices),
                )
                acceleration = cannonball_srp_acceleration(
                    backend=backend,
                    positions=positions,
                    gravitational_parameters=gravitational_parameters,
                    radii=radii,
                    source_mask=source_mask,
                    target_mask=target_mask,
                    radiation_source_index=source_index,
                    reference_pressure=model.reference_pressure,
                    reference_distance=model.reference_distance,
                    area_to_mass=_expand_target_parameter(
                        backend, body_count, target_indices, area_to_mass
                    ),
                    radiation_pressure_coefficient=_expand_target_parameter(
                        backend, body_count, target_indices, coefficient
                    ),
                    singularity_policy=SINGULARITY_POLICY_ERROR,
                    collision_policy=COLLISION_POLICY_ERROR,
                )
                role = "CORRECTION_ONLY"
                source_ids = (model.radiation_source_id,)
                target_ids = model.target_ids
                assumptions = (
                    "radiation source is required to be marked massive",
                    "isotropic cannonball targets are massless with no backreaction",
                    "unshadowed radial pressure only",
                    DETERMINISM_LIMITATION,
                )
            else:  # The exact-type preflight makes this unreachable.
                raise UnsupportedForceError("force model has no runtime implementation")

            _ensure_finite_acceleration(
                backend, acceleration, model.model_id, body_count
            )
            entry = ForceLedgerEntry(
                order=order,
                model_id=model.model_id,
                role=role,
                source_ids=source_ids,
                target_ids=target_ids,
                state_metadata=state_metadata,
                backend_spec=plan.backend,
                tile_size=tile_size,
                assumptions=assumptions,
            )
            ledger.append(entry)
            contributions.append(ForceContribution(entry, acceleration))
            total = total + acceleration

        _ensure_finite_acceleration(backend, total, "total", body_count)
        return ForceEvaluationResult(
            snapshot_id=snapshot.snapshot_id,
            plan_id=plan.plan_id,
            backend_id=backend.name,
            device=backend.device,
            dtype="float64",
            state_metadata=state_metadata,
            backend_spec=plan.backend,
            tile_size=tile_size,
            total_acceleration=total,
            contributions=tuple(contributions),
            ledger=tuple(ledger),
        )


__all__ = [
    "DuplicateForceError",
    "DETERMINISM_LIMITATION",
    "DETERMINISM_SCOPE",
    "EvaluationError",
    "FORCE_EVALUATION_ONLY",
    "ForceContribution",
    "ForceEvaluationResult",
    "ForceLedgerEntry",
    "MODEL_OUTPUT",
    "StateMetadataBinding",
    "UnsupportedForceError",
    "evaluate_force_plan",
]
