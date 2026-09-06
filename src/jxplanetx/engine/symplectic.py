"""Fixed-step, mutual-Newtonian kick-drift-kick trajectories."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Any, Callable

from .backends import ArrayBackend, resolve_backend
from .contracts import (
    BackendSpec,
    ForcePlan,
    NewtonianPointMass,
    ParameterMetadata,
    Provenance,
    StateSnapshot,
)
from .evaluator import (
    DETERMINISM_LIMITATION,
    ForceLedgerEntry,
    StateMetadataBinding,
    _validate_dependencies,
    _validate_model_sequence,
    _validate_parameter_contracts,
    _validate_state,
    evaluate_force_plan,
)
from .symplectic_contracts import (
    FIXED_STEP_KDK_METHOD_ID,
    FixedStepKDKSpec,
    KDK_ALLOWED_AXES,
    KDK_ALLOWED_TIME_SCALES,
    KDK_BACKEND_SCOPE,
    KDK_CHECKPOINT_POLICY,
    KDK_COMPOSITION,
    KDK_DRIFT_COEFFICIENTS,
    KDK_ENCOUNTER_GUARD,
    KDK_EVIDENCE_CLASS,
    KDK_FORCE_EVALUATION_ACCOUNTING,
    KDK_FORCE_PLAN_SCOPE,
    KDK_KICK_COEFFICIENTS,
    KDK_METHOD_CLASS,
    KDK_PAIR_FREQUENCY_GUARD,
    KDK_PRINCIPAL_ORDER,
    KDK_REQUIRED_FRAME,
    KDK_REQUIRED_ORIGIN,
    KDK_RESULT_CONTENT_CHECKSUM_ALGORITHM,
    KDK_RESULT_CONTENT_CHECKSUM_DOMAIN,
    KDK_SCHEDULE_CHECKSUM_ALGORITHM,
    KDK_SCHEDULE_CHECKSUM_DOMAIN,
    KDK_STEP_REPRESENTATION,
    KDK_TIME_SEMANTICS,
)
from .trajectory import (
    TrajectoryCheckpoint,
    TrajectoryContractError,
    TrajectoryDomainError,
    TrajectoryStepLimitError,
)


KDK_TRAJECTORY_SCOPE = "FIXED_STEP_KDK_TRAJECTORY_INTEGRATION"
KDK_NEWTONIAN_LEDGER_ASSUMPTIONS = (
    "direct unsoftened point masses",
    "finite-radius contact and singular coincidence are errors",
    DETERMINISM_LIMITATION,
)


def _exact_text(value: object, label: str) -> None:
    if type(value) is not str or not value or value.strip() != value:
        raise TrajectoryContractError(f"{label} must be an exact nonempty trimmed string")


def _exact_string_tuple(value: object, label: str) -> None:
    if type(value) is not tuple or any(type(item) is not str for item in value):
        raise TrajectoryContractError(f"{label} must be an exact tuple of built-in strings")


def _validate_backend_schema(value: object, label: str) -> BackendSpec:
    if type(value) is not BackendSpec:
        raise TrajectoryContractError(f"{label} must be an exact BackendSpec")
    for field in ("backend_id", "device", "dtype", "determinism_scope"):
        _exact_text(getattr(value, field), f"{label}.{field}")
    if type(value.tile_size) is not int or value.tile_size <= 0:
        raise TrajectoryContractError(f"{label}.tile_size must be an exact positive integer")
    for field in (
        "allow_fallback",
        "deterministic_reductions",
        "fast_math",
    ):
        if type(getattr(value, field)) is not bool:
            raise TrajectoryContractError(f"{label}.{field} must be exact bool")
    value.__post_init__()
    return value


def _validate_provenance_schema(value: object, label: str) -> Provenance:
    if type(value) is not Provenance:
        raise TrajectoryContractError(f"{label} must be an exact Provenance")
    for field in ("source_id", "citation", "version", "sha256"):
        _exact_text(getattr(value, field), f"{label}.{field}")
    value.__post_init__()
    return value


def _validate_parameter_metadata_schema(model: NewtonianPointMass) -> None:
    _exact_string_tuple(model.source_ids, "Newtonian source_ids")
    _exact_string_tuple(model.target_ids, "Newtonian target_ids")
    _exact_text(model.unit_system_id, "Newtonian unit_system_id")
    if type(model.parameter_metadata) is not tuple:
        raise TrajectoryContractError("parameter_metadata must be an exact tuple")
    for index, metadata in enumerate(model.parameter_metadata):
        if type(metadata) is not ParameterMetadata:
            raise TrajectoryContractError(
                f"parameter_metadata[{index}] must be exact ParameterMetadata"
            )
        if type(metadata.provenance) is not Provenance:
            raise TrajectoryContractError(
                f"parameter_metadata[{index}].provenance must be exact Provenance"
            )
        _exact_text(metadata.parameter_id, f"parameter_metadata[{index}].parameter_id")
        _exact_text(metadata.units, f"parameter_metadata[{index}].units")
        if type(metadata.validity_start) is not float or type(metadata.validity_end) is not float:
            raise TrajectoryContractError(
                f"parameter_metadata[{index}] validity bounds must be exact floats"
            )
        if metadata.uncertainty is not None and type(metadata.uncertainty) is not float:
            raise TrajectoryContractError(
                f"parameter_metadata[{index}].uncertainty must be None or exact float"
            )
        if metadata.covariance_group is not None and type(metadata.covariance_group) is not str:
            raise TrajectoryContractError(
                f"parameter_metadata[{index}].covariance_group must be None or exact str"
            )
        if metadata.covariance_group is not None:
            _exact_text(
                metadata.covariance_group,
                f"parameter_metadata[{index}].covariance_group",
            )
        _validate_provenance_schema(
            metadata.provenance, f"parameter_metadata[{index}].provenance"
        )
        metadata.__post_init__()
    model.__post_init__()


def _validate_snapshot_schema(value: object, label: str) -> StateSnapshot:
    if type(value) is not StateSnapshot:
        raise TrajectoryContractError(f"{label} must be an exact StateSnapshot")
    for field in (
        "snapshot_id",
        "time_scale",
        "frame",
        "origin",
        "axes",
        "length_unit",
        "time_unit",
        "mass_unit",
        "unit_system_id",
    ):
        _exact_text(getattr(value, field), f"{label}.{field}")
    if type(value.epoch) is not float or not math.isfinite(value.epoch):
        raise TrajectoryContractError(f"{label}.epoch must be an exact finite float")
    _exact_string_tuple(value.body_ids, f"{label}.body_ids")
    _validate_provenance_schema(value.provenance, f"{label}.provenance")
    value.__post_init__()
    return value


def _validate_force_plan_schema(value: object, label: str) -> ForcePlan:
    if type(value) is not ForcePlan:
        raise TrajectoryContractError(f"{label} must be an exact ForcePlan")
    _exact_text(value.plan_id, f"{label}.plan_id")
    _exact_text(value.evidence_class, f"{label}.evidence_class")
    if type(value.registry_authorized) is not bool:
        raise TrajectoryContractError(f"{label}.registry_authorized must be exact bool")
    if type(value.qualification_authorized) is not bool:
        raise TrajectoryContractError(
            f"{label}.qualification_authorized must be exact bool"
        )
    _validate_backend_schema(value.backend, f"{label}.backend")
    if type(value.models) is not tuple:
        raise TrajectoryContractError(f"{label}.models must be an exact tuple")
    value.__post_init__()
    return value


def _validate_ledger_schema(value: object) -> ForceLedgerEntry:
    if type(value) is not ForceLedgerEntry:
        raise TrajectoryContractError("force_ledger entries must be exact ForceLedgerEntry")
    if type(value.order) is not int or value.order < 0:
        raise TrajectoryContractError("force-ledger order must be an exact nonnegative int")
    if type(value.tile_size) is not int or value.tile_size <= 0:
        raise TrajectoryContractError("force-ledger tile_size must be an exact positive int")
    for field in ("model_id", "role", "determinism_scope", "evidence_class"):
        _exact_text(getattr(value, field), f"force-ledger {field}")
    for field in ("source_ids", "target_ids", "assumptions"):
        _exact_string_tuple(getattr(value, field), f"force-ledger {field}")
    for field in ("registry_authorized", "qualification_authorized"):
        if type(getattr(value, field)) is not bool:
            raise TrajectoryContractError(f"force-ledger {field} must be exact bool")
    _validate_backend_schema(value.backend_spec, "force-ledger backend_spec")
    _validate_state_metadata_schema(value.state_metadata)
    return value


def _validate_checkpoint_schema(value: object, label: str) -> TrajectoryCheckpoint:
    if type(value) is not TrajectoryCheckpoint:
        raise TrajectoryContractError(f"{label} must be an exact TrajectoryCheckpoint")
    for field in ("index", "accepted_steps", "rejected_steps"):
        candidate = getattr(value, field)
        if type(candidate) is not int or candidate < 0:
            raise TrajectoryContractError(f"{label}.{field} must be exact nonnegative int")
    if type(value.epoch) is not float or not math.isfinite(value.epoch):
        raise TrajectoryContractError(f"{label}.epoch must be an exact finite float")
    _exact_string_tuple(value.body_ids, f"{label}.body_ids")
    for field in ("backend_id", "device", "dtype", "evidence_class"):
        _exact_text(getattr(value, field), f"{label}.{field}")
    for field in ("registry_authorized", "qualification_authorized"):
        if type(getattr(value, field)) is not bool:
            raise TrajectoryContractError(f"{label}.{field} must be exact bool")
    value.__post_init__()
    return value


def _validate_state_metadata_schema(value: object) -> StateMetadataBinding:
    if type(value) is not StateMetadataBinding:
        raise TrajectoryContractError("force-ledger state_metadata must be exact StateMetadataBinding")
    for field in (
        "snapshot_id",
        "unit_system_id",
        "time_scale",
        "frame",
        "origin",
        "axes",
        "length_unit",
        "time_unit",
        "mass_unit",
        "provenance_source_id",
        "provenance_citation",
        "provenance_version",
        "provenance_sha256",
    ):
        _exact_text(getattr(value, field), f"force-ledger state_metadata.{field}")
    if type(value.epoch) is not float or not math.isfinite(value.epoch):
        raise TrajectoryContractError("force-ledger state_metadata.epoch must be exact finite float")
    _exact_string_tuple(value.body_ids, "force-ledger state_metadata.body_ids")
    return value


def _has_any(backend: ArrayBackend, expression: Any, label: str) -> bool:
    return backend.scalar_bool(backend.xp.any(expression), label)


def _schedule_content_sha256(
    *,
    checkpoint_step_indices: tuple[int, ...],
    checkpoint_epochs: tuple[float, ...],
    fixed_step: float,
    direction: str,
    completed_steps: int,
    force_evaluations: int,
) -> str:
    """Return the unauthenticated V1 checksum of KDK schedule metadata."""

    payload = {
        "checkpoint_epochs_hex": [epoch.hex() for epoch in checkpoint_epochs],
        "checkpoint_step_indices": list(checkpoint_step_indices),
        "checksum_algorithm": KDK_SCHEDULE_CHECKSUM_ALGORITHM,
        "completed_steps": completed_steps,
        "direction": direction,
        "fixed_step_hex": fixed_step.hex(),
        "force_evaluations": force_evaluations,
        "method_id": FIXED_STEP_KDK_METHOD_ID,
    }
    serialized = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    preimage = KDK_SCHEDULE_CHECKSUM_DOMAIN.encode("utf-8") + b"\x00" + serialized
    return hashlib.sha256(preimage).hexdigest()


def _array_hex(array: Any) -> list[Any]:
    if array.ndim == 1:
        return [float(value).hex() for value in array]
    return [[float(value).hex() for value in row] for row in array]


def _result_content_sha256(
    *,
    initial_snapshot: StateSnapshot,
    force_plan: ForcePlan,
    integration_spec: FixedStepKDKSpec,
    checkpoint_step_indices: tuple[int, ...],
    checkpoint_epochs: tuple[float, ...],
    checkpoints: tuple[TrajectoryCheckpoint, ...],
    force_model_ids: tuple[str, ...],
    force_ledger: tuple[ForceLedgerEntry, ...],
    completed_steps: int,
    force_evaluations: int,
    direction: str,
    schedule_content_sha256: str,
    minimum_observed_pair_separations: tuple[float, ...],
    minimum_observed_swept_pair_separation: float,
    maximum_observed_pair_frequency_step: float,
) -> str:
    """Hash retained KDK content; this is integrity metadata, not authority."""

    model = force_plan.models[0]
    parameter_metadata = []
    for item in model.parameter_metadata:  # type: ignore[union-attr]
        parameter_metadata.append(
            {
                "covariance_group": item.covariance_group,
                "parameter_id": item.parameter_id,
                "provenance": {
                    "citation": item.provenance.citation,
                    "sha256": item.provenance.sha256,
                    "source_id": item.provenance.source_id,
                    "version": item.provenance.version,
                },
                "uncertainty_hex": (
                    None if item.uncertainty is None else float(item.uncertainty).hex()
                ),
                "units": item.units,
                "validity_end_hex": float(item.validity_end).hex(),
                "validity_start_hex": float(item.validity_start).hex(),
            }
        )
    payload = {
        "backend": {
            "allow_fallback": force_plan.backend.allow_fallback,
            "backend_id": force_plan.backend.backend_id,
            "determinism_scope": force_plan.backend.determinism_scope,
            "deterministic_reductions": force_plan.backend.deterministic_reductions,
            "device": force_plan.backend.device,
            "dtype": force_plan.backend.dtype,
            "fast_math": force_plan.backend.fast_math,
            "tile_size": force_plan.backend.tile_size,
        },
        "checkpoint_states": [
            {
                "accepted_steps": checkpoint.accepted_steps,
                "epoch_hex": float(checkpoint.epoch).hex(),
                "index": checkpoint.index,
                "positions_hex": _array_hex(checkpoint.positions),
                "rejected_steps": checkpoint.rejected_steps,
                "step_index": step_index,
                "velocities_hex": _array_hex(checkpoint.velocities),
            }
            for checkpoint, step_index in zip(checkpoints, checkpoint_step_indices)
        ],
        "checksum_algorithm": KDK_RESULT_CONTENT_CHECKSUM_ALGORITHM,
        "control": {
            "adaptive": False,
            "dense_output": False,
            "evidence_class": KDK_EVIDENCE_CLASS,
            "exact_arithmetic_symplectic": True,
            "exact_arithmetic_time_reversible": True,
            "floating_point_symplectic": False,
            "qualification_authorized": False,
            "registry_authorized": False,
            "scope": KDK_TRAJECTORY_SCOPE,
        },
        "force_ledger": [
            {
                "assumptions": list(entry.assumptions),
                "backend": {
                    "backend_id": entry.backend_spec.backend_id,
                    "device": entry.backend_spec.device,
                    "dtype": entry.backend_spec.dtype,
                    "tile_size": entry.backend_spec.tile_size,
                },
                "determinism_scope": entry.determinism_scope,
                "evidence_class": entry.evidence_class,
                "model_id": entry.model_id,
                "order": entry.order,
                "qualification_authorized": entry.qualification_authorized,
                "registry_authorized": entry.registry_authorized,
                "role": entry.role,
                "source_ids": list(entry.source_ids),
                "state_metadata": {
                    "axes": entry.state_metadata.axes,
                    "body_ids": list(entry.state_metadata.body_ids),
                    "epoch_hex": float(entry.state_metadata.epoch).hex(),
                    "frame": entry.state_metadata.frame,
                    "length_unit": entry.state_metadata.length_unit,
                    "mass_unit": entry.state_metadata.mass_unit,
                    "origin": entry.state_metadata.origin,
                    "provenance_citation": entry.state_metadata.provenance_citation,
                    "provenance_sha256": entry.state_metadata.provenance_sha256,
                    "provenance_source_id": entry.state_metadata.provenance_source_id,
                    "provenance_version": entry.state_metadata.provenance_version,
                    "snapshot_id": entry.state_metadata.snapshot_id,
                    "time_scale": entry.state_metadata.time_scale,
                    "time_unit": entry.state_metadata.time_unit,
                    "unit_system_id": entry.state_metadata.unit_system_id,
                },
                "target_ids": list(entry.target_ids),
                "tile_size": entry.tile_size,
            }
            for entry in force_ledger
        ],
        "force_plan": {
            "evidence_class": force_plan.evidence_class,
            "model": {
                "model_id": model.model_id,  # type: ignore[union-attr]
                "parameter_metadata": parameter_metadata,
                "source_ids": list(model.source_ids),  # type: ignore[union-attr]
                "target_ids": list(model.target_ids),  # type: ignore[union-attr]
                "unit_system_id": model.unit_system_id,  # type: ignore[union-attr]
            },
            "plan_id": force_plan.plan_id,
            "qualification_authorized": force_plan.qualification_authorized,
            "registry_authorized": force_plan.registry_authorized,
        },
        "guards": {
            "maximum_observed_pair_frequency_step_hex": maximum_observed_pair_frequency_step.hex(),
            "minimum_observed_pair_separations_hex": [
                value.hex() for value in minimum_observed_pair_separations
            ],
            "minimum_observed_swept_pair_separation_hex": minimum_observed_swept_pair_separation.hex(),
        },
        "integration_spec": {
            "checkpoint_step_indices": list(integration_spec.checkpoint_step_indices),
            "encounter_guard": integration_spec.encounter_guard,
            "fixed_step_hex": integration_spec.fixed_step.hex(),
            "force_plan_scope": integration_spec.force_plan_scope,
            "maximum_pair_frequency_step_hex": integration_spec.maximum_pair_frequency_step.hex(),
            "maximum_steps": integration_spec.maximum_steps,
            "minimum_swept_pair_separation_hex": integration_spec.minimum_swept_pair_separation.hex(),
            "pair_frequency_guard": integration_spec.pair_frequency_guard,
        },
        "method": {
            "backend_scope": KDK_BACKEND_SCOPE,
            "checkpoint_policy": KDK_CHECKPOINT_POLICY,
            "composition": KDK_COMPOSITION,
            "drift_coefficients_hex": [value.hex() for value in KDK_DRIFT_COEFFICIENTS],
            "force_evaluation_accounting": KDK_FORCE_EVALUATION_ACCOUNTING,
            "kick_coefficients_hex": [value.hex() for value in KDK_KICK_COEFFICIENTS],
            "method_class": KDK_METHOD_CLASS,
            "method_id": FIXED_STEP_KDK_METHOD_ID,
            "principal_order": KDK_PRINCIPAL_ORDER,
            "step_representation": KDK_STEP_REPRESENTATION,
            "time_semantics": KDK_TIME_SEMANTICS,
        },
        "runtime": {
            "checkpoint_epochs_hex": [value.hex() for value in checkpoint_epochs],
            "completed_steps": completed_steps,
            "direction": direction,
            "force_evaluations": force_evaluations,
            "force_model_ids": list(force_model_ids),
            "schedule_content_sha256": schedule_content_sha256,
        },
        "snapshot": {
            "axes": initial_snapshot.axes,
            "body_ids": list(initial_snapshot.body_ids),
            "epoch_hex": float(initial_snapshot.epoch).hex(),
            "frame": initial_snapshot.frame,
            "gravitational_parameters_hex": _array_hex(initial_snapshot.gravitational_parameters),
            "length_unit": initial_snapshot.length_unit,
            "mass_unit": initial_snapshot.mass_unit,
            "masses_hex": _array_hex(initial_snapshot.masses),
            "massive": [bool(value) for value in initial_snapshot.massive],
            "origin": initial_snapshot.origin,
            "positions_hex": _array_hex(initial_snapshot.positions),
            "provenance": {
                "citation": initial_snapshot.provenance.citation,
                "sha256": initial_snapshot.provenance.sha256,
                "source_id": initial_snapshot.provenance.source_id,
                "version": initial_snapshot.provenance.version,
            },
            "radii_hex": _array_hex(initial_snapshot.radii),
            "snapshot_id": initial_snapshot.snapshot_id,
            "time_scale": initial_snapshot.time_scale,
            "time_unit": initial_snapshot.time_unit,
            "unit_system_id": initial_snapshot.unit_system_id,
            "velocities_hex": _array_hex(initial_snapshot.velocities),
        },
    }
    serialized = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    preimage = KDK_RESULT_CONTENT_CHECKSUM_DOMAIN.encode("utf-8") + b"\x00" + serialized
    return hashlib.sha256(preimage).hexdigest()


def _step_epochs(initial_epoch: float, spec: FixedStepKDKSpec) -> tuple[float, ...]:
    """Preflight every binary64 time label before touching state arithmetic."""

    epochs: list[float] = []
    previous: float | None = None
    direction = 1.0 if spec.fixed_step > 0.0 else -1.0
    for index in range(spec.completed_steps + 1):
        epoch = float(initial_epoch + spec.fixed_step * index)
        if not math.isfinite(epoch):
            raise TrajectoryStepLimitError("the fixed KDK schedule produced a nonfinite epoch")
        if previous is not None and direction * (epoch - previous) <= 0.0:
            raise TrajectoryStepLimitError(
                "fixed_step cannot advance every binary64 epoch on the requested lattice"
            )
        epochs.append(epoch)
        previous = epoch
    return tuple(epochs)


def _validate_semantic_boundary(snapshot: StateSnapshot) -> None:
    if snapshot.time_scale not in KDK_ALLOWED_TIME_SCALES:
        raise TrajectoryContractError(
            f"KDK time_scale must be one of {KDK_ALLOWED_TIME_SCALES!r}"
        )
    if snapshot.frame != KDK_REQUIRED_FRAME:
        raise TrajectoryContractError(
            f"KDK frame must equal {KDK_REQUIRED_FRAME!r}"
        )
    if snapshot.origin != KDK_REQUIRED_ORIGIN:
        raise TrajectoryContractError(
            f"KDK origin must equal {KDK_REQUIRED_ORIGIN!r}"
        )
    if snapshot.axes not in KDK_ALLOWED_AXES:
        raise TrajectoryContractError(
            f"KDK axes must be one of {KDK_ALLOWED_AXES!r}"
        )


def _validate_force_plan_scope(snapshot: StateSnapshot, plan: ForcePlan) -> NewtonianPointMass:
    if plan.evidence_class != KDK_EVIDENCE_CLASS:
        raise TrajectoryContractError("KDK accepts only MODEL_OUTPUT force plans")
    if plan.registry_authorized or plan.qualification_authorized:
        raise TrajectoryContractError("KDK force plans cannot carry authority")
    if plan.backend.backend_id != "numpy" or plan.backend.device != "cpu":
        raise TrajectoryContractError("KDK v1 requires the NumPy CPU backend")
    if len(snapshot.body_ids) < 2:
        raise TrajectoryContractError("KDK requires at least two mutually interacting bodies")
    if type(plan.models) is not tuple or len(plan.models) != 1:
        raise TrajectoryContractError(
            "KDK requires exactly one NewtonianPointMass force configuration"
        )
    model = plan.models[0]
    if type(model) is not NewtonianPointMass:
        raise TrajectoryContractError(
            "KDK requires exactly one NewtonianPointMass force configuration"
        )
    if model.source_ids != snapshot.body_ids or model.target_ids != snapshot.body_ids:
        raise TrajectoryContractError(
            "KDK requires source_ids == target_ids == snapshot.body_ids in exact order"
        )
    return model


def _validate_metadata_range(plan: ForcePlan, lower: float, upper: float) -> None:
    for model in plan.models:
        for metadata in model.parameter_metadata:  # type: ignore[union-attr]
            if metadata.validity_start > lower or metadata.validity_end < upper:
                raise TrajectoryContractError(
                    f"{metadata.parameter_id} validity interval does not cover the full KDK schedule"
                )


def _validate_map_arrays(
    backend: ArrayBackend,
    arrays: tuple[tuple[str, Any], ...],
    shape: tuple[int, int],
) -> None:
    for label, array in arrays:
        backend.require_native_array(array, label)
        if array.dtype != backend.float64 or array.ndim != 2 or array.shape != shape:
            raise TrajectoryContractError(
                f"{label} must remain a native float64 (body_count, 3) array"
            )
        if _has_any(backend, ~backend.xp.isfinite(array), f"{label} finiteness check"):
            raise TrajectoryDomainError(f"{label} contains a nonfinite value")


def _pair_guard(
    *,
    positions: Any,
    candidate_positions: Any,
    gravitational_parameters: Any,
    radii: Any,
    spec: FixedStepKDKSpec,
) -> tuple[tuple[float, ...], float]:
    """Validate every swept linear drift segment and return observed extrema."""

    pair_distances: list[float] = []
    maximum_frequency_step = 0.0
    body_count = int(positions.shape[0])
    for left in range(body_count - 1):
        for right in range(left + 1, body_count):
            relative_start = positions[right] - positions[left]
            relative_delta = (
                candidate_positions[right]
                - positions[right]
                - candidate_positions[left]
                + positions[left]
            )
            delta_squared = float(relative_delta @ relative_delta)
            if not math.isfinite(delta_squared) or delta_squared < 0.0:
                raise TrajectoryDomainError("KDK swept-drift geometry became nonfinite")
            if delta_squared == 0.0:
                fraction = 0.0
            else:
                fraction = min(
                    1.0,
                    max(0.0, -float(relative_start @ relative_delta) / delta_squared),
                )
            closest = relative_start + fraction * relative_delta
            distance_squared = float(closest @ closest)
            if not math.isfinite(distance_squared) or distance_squared <= 0.0:
                raise TrajectoryDomainError(
                    "KDK swept drift reaches a singular pair coincidence"
                )
            distance = math.sqrt(distance_squared)
            contact = float(radii[left] + radii[right])
            required = max(float(spec.minimum_swept_pair_separation), contact)
            if distance <= required:
                raise TrajectoryDomainError(
                    "KDK swept pair separation reaches the contact/caller encounter floor"
                )
            pair_gm = float(
                gravitational_parameters[left] + gravitational_parameters[right]
            )
            frequency_step = abs(spec.fixed_step) * math.sqrt(
                pair_gm / (distance * distance * distance)
            )
            if not math.isfinite(frequency_step):
                raise TrajectoryDomainError("KDK pair-frequency guard became nonfinite")
            if frequency_step > spec.maximum_pair_frequency_step:
                raise TrajectoryDomainError(
                    "KDK pair-frequency resolution limit is exceeded"
                )
            pair_distances.append(distance)
            maximum_frequency_step = max(maximum_frequency_step, frequency_step)
    return tuple(pair_distances), maximum_frequency_step


def _kdk_step(
    *,
    backend: ArrayBackend,
    signed_step: float,
    positions: Any,
    velocities: Any,
    acceleration: Any,
    acceleration_at_drifted_state: Callable[[Any, Any], Any],
) -> tuple[Any, Any, Any]:
    """Apply one KDK map with the locked binary64 coefficient order."""

    if type(signed_step) is not float or not math.isfinite(signed_step) or signed_step == 0.0:
        raise TrajectoryContractError("signed_step must be one finite nonzero built-in float")
    shape = positions.shape
    _validate_map_arrays(
        backend,
        (("positions", positions), ("velocities", velocities), ("acceleration", acceleration)),
        shape,
    )
    xp = backend.xp
    h = xp.float64(signed_step)
    half = xp.float64(0.5) * h
    half_velocities = velocities + half * acceleration
    candidate_positions = positions + h * half_velocities
    _validate_map_arrays(
        backend,
        (("half-step velocities", half_velocities), ("candidate positions", candidate_positions)),
        shape,
    )
    candidate_acceleration = acceleration_at_drifted_state(
        candidate_positions, half_velocities
    )
    _validate_map_arrays(
        backend,
        (("candidate acceleration", candidate_acceleration),),
        shape,
    )
    candidate_velocities = half_velocities + half * candidate_acceleration
    _validate_map_arrays(
        backend,
        (("candidate velocities", candidate_velocities),),
        shape,
    )
    return candidate_positions, candidate_velocities, candidate_acceleration


def _readonly_copy(array: Any) -> Any:
    copied = array.copy()
    copied.setflags(write=False)
    return copied


def _readonly_snapshot_copy(
    snapshot: StateSnapshot,
    validated_state: tuple[Any, Any, Any, Any, Any, Any],
) -> StateSnapshot:
    return replace(
        snapshot,
        positions=_readonly_copy(validated_state[0]),
        velocities=_readonly_copy(validated_state[1]),
        gravitational_parameters=_readonly_copy(validated_state[2]),
        masses=_readonly_copy(validated_state[3]),
        radii=_readonly_copy(validated_state[4]),
        massive=_readonly_copy(validated_state[5]),
    )


@dataclass(frozen=True, eq=False)
class KDKTrajectoryResult:
    """Copied NumPy states on an exact integer KDK map lattice."""

    snapshot_id: str
    plan_id: str
    backend_id: str
    device: str
    dtype: str
    backend_spec: BackendSpec
    initial_snapshot: StateSnapshot
    force_plan: ForcePlan
    integration_spec: FixedStepKDKSpec
    checkpoint_step_indices: tuple[int, ...]
    checkpoint_epochs: tuple[float, ...]
    checkpoints: tuple[TrajectoryCheckpoint, ...]
    force_model_ids: tuple[str, ...]
    force_ledger: tuple[ForceLedgerEntry, ...]
    completed_steps: int
    force_evaluations: int
    direction: str
    schedule_content_sha256: str
    result_content_sha256: str
    minimum_observed_pair_separations: tuple[float, ...]
    minimum_observed_swept_pair_separation: float
    maximum_observed_pair_frequency_step: float
    method_id: str = FIXED_STEP_KDK_METHOD_ID
    method_class: str = KDK_METHOD_CLASS
    principal_order: int = KDK_PRINCIPAL_ORDER
    kick_coefficients: tuple[float, ...] = KDK_KICK_COEFFICIENTS
    drift_coefficients: tuple[float, ...] = KDK_DRIFT_COEFFICIENTS
    composition: str = KDK_COMPOSITION
    step_representation: str = KDK_STEP_REPRESENTATION
    checkpoint_policy: str = KDK_CHECKPOINT_POLICY
    force_plan_scope: str = KDK_FORCE_PLAN_SCOPE
    backend_scope: str = KDK_BACKEND_SCOPE
    time_semantics: str = KDK_TIME_SEMANTICS
    encounter_guard: str = KDK_ENCOUNTER_GUARD
    pair_frequency_guard: str = KDK_PAIR_FREQUENCY_GUARD
    force_evaluation_accounting: str = KDK_FORCE_EVALUATION_ACCOUNTING
    schedule_checksum_algorithm: str = KDK_SCHEDULE_CHECKSUM_ALGORITHM
    schedule_checksum_domain: str = KDK_SCHEDULE_CHECKSUM_DOMAIN
    result_content_checksum_algorithm: str = KDK_RESULT_CONTENT_CHECKSUM_ALGORITHM
    result_content_checksum_domain: str = KDK_RESULT_CONTENT_CHECKSUM_DOMAIN
    scope: str = KDK_TRAJECTORY_SCOPE
    evidence_class: str = KDK_EVIDENCE_CLASS
    exact_arithmetic_symplectic: bool = True
    floating_point_symplectic: bool = False
    exact_arithmetic_time_reversible: bool = True
    registry_authorized: bool = False
    qualification_authorized: bool = False
    dense_output: bool = False
    adaptive: bool = False

    def __post_init__(self) -> None:
        for field in ("kick_coefficients", "drift_coefficients"):
            values = getattr(self, field)
            if type(values) is not tuple or any(type(value) is not float for value in values):
                raise TrajectoryContractError(
                    f"{field} must be an exact tuple of built-in floats"
                )
        exact = {
            "backend_id": "numpy",
            "device": "cpu",
            "dtype": "float64",
            "method_id": FIXED_STEP_KDK_METHOD_ID,
            "method_class": KDK_METHOD_CLASS,
            "principal_order": KDK_PRINCIPAL_ORDER,
            "kick_coefficients": KDK_KICK_COEFFICIENTS,
            "drift_coefficients": KDK_DRIFT_COEFFICIENTS,
            "composition": KDK_COMPOSITION,
            "step_representation": KDK_STEP_REPRESENTATION,
            "checkpoint_policy": KDK_CHECKPOINT_POLICY,
            "force_plan_scope": KDK_FORCE_PLAN_SCOPE,
            "backend_scope": KDK_BACKEND_SCOPE,
            "time_semantics": KDK_TIME_SEMANTICS,
            "encounter_guard": KDK_ENCOUNTER_GUARD,
            "pair_frequency_guard": KDK_PAIR_FREQUENCY_GUARD,
            "force_evaluation_accounting": KDK_FORCE_EVALUATION_ACCOUNTING,
            "schedule_checksum_algorithm": KDK_SCHEDULE_CHECKSUM_ALGORITHM,
            "schedule_checksum_domain": KDK_SCHEDULE_CHECKSUM_DOMAIN,
            "result_content_checksum_algorithm": KDK_RESULT_CONTENT_CHECKSUM_ALGORITHM,
            "result_content_checksum_domain": KDK_RESULT_CONTENT_CHECKSUM_DOMAIN,
            "scope": KDK_TRAJECTORY_SCOPE,
            "evidence_class": KDK_EVIDENCE_CLASS,
            "exact_arithmetic_symplectic": True,
            "floating_point_symplectic": False,
            "exact_arithmetic_time_reversible": True,
            "registry_authorized": False,
            "qualification_authorized": False,
            "dense_output": False,
            "adaptive": False,
        }
        for field, expected in exact.items():
            value = getattr(self, field)
            if type(value) is not type(expected) or value != expected:
                raise TrajectoryContractError(
                    f"{field} must equal the fixed KDK result value {expected!r}"
                )
        _validate_result(self)

    @property
    def checkpoint_count(self) -> int:
        return len(self.checkpoints)

    @property
    def final_epoch(self) -> float:
        return self.checkpoint_epochs[-1]

    @property
    def positions(self) -> tuple[Any, ...]:
        return tuple(checkpoint.positions for checkpoint in self.checkpoints)

    @property
    def velocities(self) -> tuple[Any, ...]:
        return tuple(checkpoint.velocities for checkpoint in self.checkpoints)

    @property
    def final_positions(self) -> Any:
        return self.checkpoints[-1].positions

    @property
    def final_velocities(self) -> Any:
        return self.checkpoints[-1].velocities

    @property
    def integrated(self) -> bool:
        return True

    @property
    def qualified(self) -> bool:
        return False


def _validate_result(result: KDKTrajectoryResult) -> None:
    if type(result) is not KDKTrajectoryResult:
        raise TrajectoryContractError("result must be an exact KDKTrajectoryResult")
    for field in (
        "snapshot_id",
        "plan_id",
        "direction",
        "schedule_content_sha256",
        "result_content_sha256",
    ):
        _exact_text(getattr(result, field), field)
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
        or any(type(value) is not TrajectoryCheckpoint for value in result.checkpoints)
    ):
        raise TrajectoryContractError(
            "checkpoints must be an exact tuple of TrajectoryCheckpoint values"
        )
    _exact_string_tuple(result.force_model_ids, "force_model_ids")
    if (
        type(result.force_ledger) is not tuple
        or any(type(value) is not ForceLedgerEntry for value in result.force_ledger)
    ):
        raise TrajectoryContractError(
            "force_ledger must be an exact tuple of ForceLedgerEntry values"
        )
    for field in ("completed_steps", "force_evaluations"):
        value = getattr(result, field)
        if type(value) is not int or value < 0:
            raise TrajectoryContractError(f"{field} must be an exact nonnegative int")
    if type(result.minimum_observed_pair_separations) is not tuple:
        raise TrajectoryContractError(
            "minimum_observed_pair_separations must be an exact tuple"
        )

    _validate_snapshot_schema(result.initial_snapshot, "initial_snapshot")
    _validate_force_plan_schema(result.force_plan, "force_plan")
    if type(result.integration_spec) is not FixedStepKDKSpec:
        raise TrajectoryContractError("integration_spec must be an exact FixedStepKDKSpec")
    result.integration_spec.__post_init__()
    _validate_backend_schema(result.backend_spec, "backend_spec")
    if result.backend_spec is not result.force_plan.backend:
        raise TrajectoryContractError("backend_spec must be the retained force-plan backend")
    if result.snapshot_id != result.initial_snapshot.snapshot_id:
        raise TrajectoryContractError("snapshot_id does not match initial_snapshot")
    if result.plan_id != result.force_plan.plan_id:
        raise TrajectoryContractError("plan_id does not match force_plan")
    _validate_semantic_boundary(result.initial_snapshot)
    retained_model = _validate_force_plan_scope(
        result.initial_snapshot, result.force_plan
    )
    _validate_parameter_metadata_schema(retained_model)
    _validate_model_sequence(result.force_plan.models)
    _validate_dependencies(result.force_plan.models)
    _validate_parameter_contracts(result.initial_snapshot, result.force_plan)
    expected_epochs_all = _step_epochs(result.initial_snapshot.epoch, result.integration_spec)
    expected_epochs = tuple(
        expected_epochs_all[index]
        for index in result.integration_spec.checkpoint_step_indices
    )
    if result.checkpoint_step_indices != result.integration_spec.checkpoint_step_indices:
        raise TrajectoryContractError("checkpoint_step_indices do not match integration_spec")
    if result.checkpoint_epochs != expected_epochs:
        raise TrajectoryContractError("checkpoint_epochs do not match the fixed-step lattice")
    _validate_metadata_range(
        result.force_plan,
        min(expected_epochs_all[0], expected_epochs_all[-1]),
        max(expected_epochs_all[0], expected_epochs_all[-1]),
    )
    if result.direction != result.integration_spec.direction:
        raise TrajectoryContractError("direction does not match fixed_step")
    if result.completed_steps != result.integration_spec.completed_steps:
        raise TrajectoryContractError("completed_steps does not match the schedule")
    if result.force_evaluations != result.completed_steps + 1:
        raise TrajectoryContractError("KDK force evaluations must equal completed_steps + 1")
    if result.force_model_ids != (NewtonianPointMass.MODEL_ID,):
        raise TrajectoryContractError("KDK force_model_ids must contain only Newtonian gravity")
    if len(result.force_ledger) != 1:
        raise TrajectoryContractError("KDK force_ledger must contain one entry")
    entry = _validate_ledger_schema(result.force_ledger[0])
    model = result.force_plan.models[0]
    if (
        entry.order != 0
        or entry.model_id != NewtonianPointMass.MODEL_ID
        or entry.role != "NEWTONIAN_BASE"
        or entry.source_ids != result.initial_snapshot.body_ids
        or entry.target_ids != result.initial_snapshot.body_ids
        or entry.backend_spec is not result.backend_spec
        or entry.tile_size != result.backend_spec.tile_size
        or entry.determinism_scope != result.backend_spec.determinism_scope
        or entry.state_metadata.snapshot_id != result.snapshot_id
        or entry.state_metadata.epoch != result.initial_snapshot.epoch
        or entry.state_metadata.unit_system_id != result.initial_snapshot.unit_system_id
        or entry.state_metadata.time_scale != result.initial_snapshot.time_scale
        or entry.state_metadata.frame != result.initial_snapshot.frame
        or entry.state_metadata.origin != result.initial_snapshot.origin
        or entry.state_metadata.axes != result.initial_snapshot.axes
        or entry.state_metadata.length_unit != result.initial_snapshot.length_unit
        or entry.state_metadata.time_unit != result.initial_snapshot.time_unit
        or entry.state_metadata.mass_unit != result.initial_snapshot.mass_unit
        or entry.state_metadata.body_ids != result.initial_snapshot.body_ids
        or entry.state_metadata.provenance_source_id
        != result.initial_snapshot.provenance.source_id
        or entry.state_metadata.provenance_citation
        != result.initial_snapshot.provenance.citation
        or entry.state_metadata.provenance_version
        != result.initial_snapshot.provenance.version
        or entry.state_metadata.provenance_sha256
        != result.initial_snapshot.provenance.sha256
        or model.source_ids != entry.source_ids  # type: ignore[union-attr]
        or model.target_ids != entry.target_ids  # type: ignore[union-attr]
        or entry.evidence_class != KDK_EVIDENCE_CLASS
        or entry.assumptions != KDK_NEWTONIAN_LEDGER_ASSUMPTIONS
        or entry.registry_authorized is not False
        or entry.qualification_authorized is not False
    ):
        raise TrajectoryContractError("KDK force-ledger binding is inconsistent")
    if len(result.checkpoints) != len(expected_epochs):
        raise TrajectoryContractError("KDK checkpoints must align with the output lattice")

    backend = resolve_backend(result.backend_spec)
    with backend.activate():
        initial_arrays = _validate_state(backend, result.initial_snapshot)
        if _has_any(backend, ~initial_arrays[5], "retained KDK active-body check"):
            raise TrajectoryContractError("every retained KDK body must remain active")
        if _has_any(
            backend,
            initial_arrays[2] <= backend.xp.float64(0.0),
            "retained KDK positive-GM check",
        ):
            raise TrajectoryContractError("every retained KDK body must have positive GM")
        retained_arrays: list[Any] = []
        for field, array in zip(
            (
                "positions",
                "velocities",
                "gravitational_parameters",
                "masses",
                "radii",
                "massive",
            ),
            initial_arrays,
        ):
            if type(array) is not backend.array_type:
                raise TrajectoryContractError(
                    f"retained KDK {field} must be an exact NumPy ndarray"
                )
            if not array.flags.owndata:
                raise TrajectoryContractError(
                    f"retained KDK {field} must own its memory"
                )
            if array.flags.writeable:
                raise TrajectoryContractError("retained KDK state arrays must be read-only")
            for previous in retained_arrays:
                try:
                    overlaps = bool(backend.xp.shares_memory(array, previous))
                except Exception as exc:
                    raise TrajectoryContractError(
                        "retained KDK array overlap could not be resolved exactly"
                    ) from exc
                if overlaps:
                    raise TrajectoryContractError(
                        "retained KDK state arrays must not overlap memory"
                    )
            retained_arrays.append(array)
        for output_index, (checkpoint, step_index, epoch) in enumerate(
            zip(result.checkpoints, result.checkpoint_step_indices, result.checkpoint_epochs)
        ):
            _validate_checkpoint_schema(
                checkpoint, f"checkpoints[{output_index}]"
            )
            if (
                checkpoint.index != output_index
                or checkpoint.epoch != epoch
                or checkpoint.accepted_steps != step_index
                or checkpoint.rejected_steps != 0
                or checkpoint.body_ids != result.initial_snapshot.body_ids
                or checkpoint.backend_id != "numpy"
                or checkpoint.device != "cpu"
                or checkpoint.dtype != "float64"
                or checkpoint.evidence_class != KDK_EVIDENCE_CLASS
                or checkpoint.registry_authorized is not False
                or checkpoint.qualification_authorized is not False
            ):
                raise TrajectoryContractError("KDK checkpoint binding is inconsistent")
            _validate_map_arrays(
                backend,
                (("checkpoint positions", checkpoint.positions), ("checkpoint velocities", checkpoint.velocities)),
                (len(checkpoint.body_ids), 3),
            )
            for array in (checkpoint.positions, checkpoint.velocities):
                if type(array) is not backend.array_type:
                    raise TrajectoryContractError(
                        "KDK checkpoint arrays must be exact NumPy ndarrays"
                    )
                if array.flags.writeable:
                    raise TrajectoryContractError("KDK checkpoint arrays must be read-only")
                for previous in retained_arrays:
                    try:
                        overlaps = bool(backend.xp.shares_memory(array, previous))
                    except Exception as exc:
                        raise TrajectoryContractError(
                            "KDK checkpoint overlap could not be resolved exactly"
                        ) from exc
                    if overlaps:
                        raise TrajectoryContractError(
                            "KDK checkpoint arrays must not overlap retained memory"
                        )
                if not array.flags.owndata:
                    raise TrajectoryContractError(
                        "KDK checkpoint arrays must own their memory"
                    )
                retained_arrays.append(array)
        if _has_any(
            backend,
            result.checkpoints[0].positions != result.initial_snapshot.positions,
            "initial KDK position binding",
        ) or _has_any(
            backend,
            result.checkpoints[0].velocities != result.initial_snapshot.velocities,
            "initial KDK velocity binding",
        ):
            raise TrajectoryContractError("initial KDK checkpoint differs from initial_snapshot")

    for label, value in (
        ("minimum_observed_swept_pair_separation", result.minimum_observed_swept_pair_separation),
        ("maximum_observed_pair_frequency_step", result.maximum_observed_pair_frequency_step),
    ):
        if type(value) is not float or not math.isfinite(value) or value <= 0.0:
            raise TrajectoryContractError(f"{label} must be a finite positive float")
    pair_count = len(result.initial_snapshot.body_ids) * (
        len(result.initial_snapshot.body_ids) - 1
    ) // 2
    if (
        type(result.minimum_observed_pair_separations) is not tuple
        or len(result.minimum_observed_pair_separations) != pair_count
        or any(
            type(value) is not float or not math.isfinite(value) or value <= 0.0
            for value in result.minimum_observed_pair_separations
        )
    ):
        raise TrajectoryContractError(
            "minimum_observed_pair_separations must be one finite positive float "
            "per canonical i<j pair"
        )
    expected_frequency = 0.0
    pair_index = 0
    for left in range(len(result.initial_snapshot.body_ids) - 1):
        for right in range(left + 1, len(result.initial_snapshot.body_ids)):
            separation = result.minimum_observed_pair_separations[pair_index]
            required = max(
                result.integration_spec.minimum_swept_pair_separation,
                float(initial_arrays[4][left] + initial_arrays[4][right]),
            )
            if separation <= required:
                raise TrajectoryContractError(
                    "observed KDK pair separation does not clear its retained contact/caller floor"
                )
            pair_gm = float(initial_arrays[2][left] + initial_arrays[2][right])
            expected_frequency = max(
                expected_frequency,
                abs(result.integration_spec.fixed_step)
                * math.sqrt(pair_gm / (separation * separation * separation)),
            )
            pair_index += 1
    if result.minimum_observed_swept_pair_separation != min(
        result.minimum_observed_pair_separations
    ):
        raise TrajectoryContractError(
            "minimum observed KDK separation does not match the per-pair ledger"
        )
    if result.maximum_observed_pair_frequency_step != expected_frequency:
        raise TrajectoryContractError(
            "maximum observed KDK frequency does not match the per-pair ledger"
        )
    if expected_frequency > result.integration_spec.maximum_pair_frequency_step:
        raise TrajectoryContractError("observed KDK frequency does not satisfy its guard")
    expected_checksum = _schedule_content_sha256(
        checkpoint_step_indices=result.checkpoint_step_indices,
        checkpoint_epochs=result.checkpoint_epochs,
        fixed_step=result.integration_spec.fixed_step,
        direction=result.direction,
        completed_steps=result.completed_steps,
        force_evaluations=result.force_evaluations,
    )
    if result.schedule_content_sha256 != expected_checksum:
        raise TrajectoryContractError(
            "KDK schedule content checksum does not match its fixed metadata"
        )
    if (
        type(result.result_content_sha256) is not str
        or len(result.result_content_sha256) != 64
        or any(character not in "0123456789abcdef" for character in result.result_content_sha256)
    ):
        raise TrajectoryContractError(
            "result_content_sha256 must be lowercase SHA-256 hexadecimal"
        )
    expected_result_checksum = _result_content_sha256(
        initial_snapshot=result.initial_snapshot,
        force_plan=result.force_plan,
        integration_spec=result.integration_spec,
        checkpoint_step_indices=result.checkpoint_step_indices,
        checkpoint_epochs=result.checkpoint_epochs,
        checkpoints=result.checkpoints,
        force_model_ids=result.force_model_ids,
        force_ledger=result.force_ledger,
        completed_steps=result.completed_steps,
        force_evaluations=result.force_evaluations,
        direction=result.direction,
        schedule_content_sha256=result.schedule_content_sha256,
        minimum_observed_pair_separations=result.minimum_observed_pair_separations,
        minimum_observed_swept_pair_separation=result.minimum_observed_swept_pair_separation,
        maximum_observed_pair_frequency_step=result.maximum_observed_pair_frequency_step,
    )
    if result.result_content_sha256 != expected_result_checksum:
        raise TrajectoryContractError(
            "KDK result content checksum does not match the retained trajectory"
        )


def integrate_kdk_trajectory(
    snapshot: StateSnapshot,
    plan: ForcePlan,
    spec: FixedStepKDKSpec,
) -> KDKTrajectoryResult:
    """Apply a CPU/NumPy KDK map to a fully mutual Newtonian force plan."""

    if type(snapshot) is not StateSnapshot:
        raise TrajectoryContractError("snapshot must be an exact StateSnapshot")
    if type(plan) is not ForcePlan:
        raise TrajectoryContractError("plan must be an exact ForcePlan")
    if type(spec) is not FixedStepKDKSpec:
        raise TrajectoryContractError("spec must be an exact FixedStepKDKSpec")

    _validate_snapshot_schema(snapshot, "snapshot")
    _validate_force_plan_schema(plan, "plan")
    spec.__post_init__()

    _validate_semantic_boundary(snapshot)
    model = _validate_force_plan_scope(snapshot, plan)
    _validate_parameter_metadata_schema(model)
    _validate_model_sequence(plan.models)
    _validate_dependencies(plan.models)
    _validate_parameter_contracts(snapshot, plan)
    backend = resolve_backend(plan.backend)
    for field in (
        "positions",
        "velocities",
        "gravitational_parameters",
        "masses",
        "radii",
        "massive",
    ):
        if type(getattr(snapshot, field)) is not backend.array_type:
            raise TrajectoryContractError(
                f"KDK input {field} must be an exact NumPy ndarray"
            )
    all_epochs = _step_epochs(snapshot.epoch, spec)
    checkpoint_epochs = tuple(all_epochs[index] for index in spec.checkpoint_step_indices)
    _validate_metadata_range(plan, min(all_epochs[0], all_epochs[-1]), max(all_epochs[0], all_epochs[-1]))

    with backend.activate():
        validated_state = _validate_state(backend, snapshot)
        if _has_any(backend, ~validated_state[5], "KDK active-body check"):
            raise TrajectoryContractError("KDK requires every body to be marked massive/active")
        if _has_any(
            backend,
            validated_state[2] <= backend.xp.float64(0.0),
            "KDK positive-GM check",
        ):
            raise TrajectoryContractError("KDK requires positive GM for every body")

        initial_snapshot = _readonly_snapshot_copy(snapshot, validated_state)
        force_plan = replace(
            plan,
            backend=replace(plan.backend),
            models=(replace(plan.models[0]),),
        )
        integration_spec = replace(spec)
        current_positions = initial_snapshot.positions.copy()
        current_velocities = initial_snapshot.velocities.copy()
        gravitational_parameters = initial_snapshot.gravitational_parameters
        radii = initial_snapshot.radii
        state_shape = (len(snapshot.body_ids), 3)

        initial_pair_separations, initial_frequency = _pair_guard(
            positions=current_positions,
            candidate_positions=current_positions,
            gravitational_parameters=gravitational_parameters,
            radii=radii,
            spec=integration_spec,
        )
        minimum_observed_pairs = initial_pair_separations
        maximum_observed_frequency = initial_frequency

        initial_evaluation = evaluate_force_plan(initial_snapshot, force_plan)
        current_acceleration = initial_evaluation.total_acceleration
        force_evaluations = 1
        force_ledger = initial_evaluation.ledger
        force_model_ids = initial_evaluation.applied_model_ids

        checkpoints: list[TrajectoryCheckpoint] = [
            TrajectoryCheckpoint(
                index=0,
                epoch=checkpoint_epochs[0],
                body_ids=initial_snapshot.body_ids,
                backend_id="numpy",
                device="cpu",
                positions=_readonly_copy(current_positions),
                velocities=_readonly_copy(current_velocities),
                accepted_steps=0,
                rejected_steps=0,
            )
        ]
        next_checkpoint = 1

        for step_index in range(1, integration_spec.completed_steps + 1):
            epoch = all_epochs[step_index]

            def acceleration_at_drifted_state(candidate_positions: Any, half_velocities: Any) -> Any:
                nonlocal force_evaluations, minimum_observed_pairs, maximum_observed_frequency
                observed_pairs, observed_frequency = _pair_guard(
                    positions=current_positions,
                    candidate_positions=candidate_positions,
                    gravitational_parameters=gravitational_parameters,
                    radii=radii,
                    spec=integration_spec,
                )
                minimum_observed_pairs = tuple(
                    min(previous, observed)
                    for previous, observed in zip(
                        minimum_observed_pairs, observed_pairs
                    )
                )
                maximum_observed_frequency = max(
                    maximum_observed_frequency, observed_frequency
                )
                stage_snapshot = replace(
                    initial_snapshot,
                    epoch=epoch,
                    positions=candidate_positions,
                    velocities=half_velocities,
                )
                evaluation = evaluate_force_plan(stage_snapshot, force_plan)
                if evaluation.applied_model_ids != force_model_ids:
                    raise TrajectoryContractError("KDK force-model identity changed during integration")
                force_evaluations += 1
                return evaluation.total_acceleration

            current_positions, current_velocities, current_acceleration = _kdk_step(
                backend=backend,
                signed_step=integration_spec.fixed_step,
                positions=current_positions,
                velocities=current_velocities,
                acceleration=current_acceleration,
                acceleration_at_drifted_state=acceleration_at_drifted_state,
            )
            if (
                next_checkpoint < len(integration_spec.checkpoint_step_indices)
                and step_index == integration_spec.checkpoint_step_indices[next_checkpoint]
            ):
                checkpoints.append(
                    TrajectoryCheckpoint(
                        index=next_checkpoint,
                        epoch=checkpoint_epochs[next_checkpoint],
                        body_ids=initial_snapshot.body_ids,
                        backend_id="numpy",
                        device="cpu",
                        positions=_readonly_copy(current_positions),
                        velocities=_readonly_copy(current_velocities),
                        accepted_steps=step_index,
                        rejected_steps=0,
                    )
                )
                next_checkpoint += 1

        if next_checkpoint != len(integration_spec.checkpoint_step_indices):
            raise TrajectoryContractError("KDK did not materialize every checkpoint")
        if force_evaluations != integration_spec.completed_steps + 1:
            raise TrajectoryContractError("KDK force-evaluation accounting is inconsistent")

        checkpoint_records = tuple(checkpoints)
        checksum = _schedule_content_sha256(
            checkpoint_step_indices=integration_spec.checkpoint_step_indices,
            checkpoint_epochs=checkpoint_epochs,
            fixed_step=integration_spec.fixed_step,
            direction=integration_spec.direction,
            completed_steps=integration_spec.completed_steps,
            force_evaluations=force_evaluations,
        )
        minimum_observed = float(min(minimum_observed_pairs))
        maximum_observed_frequency = float(maximum_observed_frequency)
        result_checksum = _result_content_sha256(
            initial_snapshot=initial_snapshot,
            force_plan=force_plan,
            integration_spec=integration_spec,
            checkpoint_step_indices=integration_spec.checkpoint_step_indices,
            checkpoint_epochs=checkpoint_epochs,
            checkpoints=checkpoint_records,
            force_model_ids=force_model_ids,
            force_ledger=force_ledger,
            completed_steps=integration_spec.completed_steps,
            force_evaluations=force_evaluations,
            direction=integration_spec.direction,
            schedule_content_sha256=checksum,
            minimum_observed_pair_separations=minimum_observed_pairs,
            minimum_observed_swept_pair_separation=minimum_observed,
            maximum_observed_pair_frequency_step=maximum_observed_frequency,
        )
        return KDKTrajectoryResult(
            snapshot_id=initial_snapshot.snapshot_id,
            plan_id=force_plan.plan_id,
            backend_id="numpy",
            device="cpu",
            dtype="float64",
            backend_spec=force_plan.backend,
            initial_snapshot=initial_snapshot,
            force_plan=force_plan,
            integration_spec=integration_spec,
            checkpoint_step_indices=integration_spec.checkpoint_step_indices,
            checkpoint_epochs=checkpoint_epochs,
            checkpoints=checkpoint_records,
            force_model_ids=force_model_ids,
            force_ledger=force_ledger,
            completed_steps=integration_spec.completed_steps,
            force_evaluations=force_evaluations,
            direction=integration_spec.direction,
            schedule_content_sha256=checksum,
            result_content_sha256=result_checksum,
            minimum_observed_pair_separations=minimum_observed_pairs,
            minimum_observed_swept_pair_separation=minimum_observed,
            maximum_observed_pair_frequency_step=maximum_observed_frequency,
        )


__all__ = [
    "KDK_TRAJECTORY_SCOPE",
    "KDKTrajectoryResult",
    "integrate_kdk_trajectory",
]
