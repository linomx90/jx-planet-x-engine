"""Adaptive, checkpoint-clipped trajectories for the experimental JX engine.

The runtime advances the classic Fehlberg RK7(8) hatted eighth-order solution.
Every trial performs all 13 force evaluations on the selected array backend.
There is no dense output: a step that would cross a requested checkpoint is
shortened so its accepted endpoint is that checkpoint exactly.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any

from .backends import ArrayBackend, resolve_backend
from .catalog import CodeStatus, get_capability
from .contracts import (
    BackendSpec,
    CannonballSRP,
    ForcePlan,
    NewtonianPointMass,
    RestrictedStaticCentral1PN,
    StateSnapshot,
)
from .evaluator import (
    EvaluationError,
    ForceLedgerEntry,
    _validate_dependencies,
    _validate_model_sequence,
    _validate_parameter_contracts,
    _validate_restricted_central_position,
    _validate_restricted_frame,
    _validate_state,
    evaluate_force_plan,
)
from .rkf78 import (
    RKF78_ACCEPTED_ORDER,
    RKF78_DEFECT_ORIENTATION,
    RKF78_EMBEDDED_ORDER,
    RKF78_METHOD_ID,
    RKF78_STAGE_COUNT,
    RKF78_TABLEAU_ID,
    RKF78_TABLEAU_SOURCE,
    _rkf78_step,
)
from .trajectory_contracts import (
    AdaptiveRKF78Spec,
    RKF78_ACCEPTED_STATE_ACCUMULATION,
    RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_ALGORITHM,
    RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_DOMAIN,
    RKF78_ACCEPTED_STEP_MAGNITUDE_SOURCE,
    RKF78_CHECKPOINT_PROPOSAL_POLICY,
    RKF78_CHECKPOINT_POLICY,
    RKF78_CONTROLLER_EXPONENT,
    RKF78_ERROR_NORM,
    RKF78_ERROR_SCALE,
    RKF78_TIME_STEP_REPRESENTATION,
)


TRAJECTORY_SCOPE = "ADAPTIVE_TRAJECTORY_INTEGRATION"
MODEL_OUTPUT = "MODEL_OUTPUT"
RKF78_RESULT_CONTENT_CHECKSUM_ALGORITHM = (
    "SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1"
)
RKF78_RESULT_CONTENT_CHECKSUM_DOMAIN = (
    "jxplanetx.adaptive-rkf78-result.content-integrity.v1"
)
RKF78_NUMPY_RESULT_CONTENT_INTEGRITY_STATUS = (
    "FULL_RETAINED_CONTENT_SHA256_READONLY_ARRAYS"
)
RKF78_CUPY_RESULT_CONTENT_INTEGRITY_STATUS = (
    "DEVICE_RESIDENT_OWNED_DISJOINT_NO_HOST_CONTENT_HASH"
)


class TrajectoryError(EvaluationError):
    """An adaptive trajectory request cannot be completed as specified."""


class TrajectoryContractError(TrajectoryError):
    """Trajectory inputs are ambiguous or mutually inconsistent."""


class TrajectoryDomainError(TrajectoryError):
    """A trial step left the finite numerical domain."""


class TrajectoryStepLimitError(TrajectoryError):
    """A fail-closed step or rejection limit was reached."""


@dataclass(frozen=True, eq=False)
class TrajectoryCheckpoint:
    """One exact checkpoint with backend-native state copies."""

    index: int
    epoch: float
    body_ids: tuple[str, ...]
    backend_id: str
    device: str
    positions: Any
    velocities: Any
    accepted_steps: int
    rejected_steps: int
    dtype: str = "float64"
    evidence_class: str = MODEL_OUTPUT
    registry_authorized: bool = False
    qualification_authorized: bool = False

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise TrajectoryContractError("checkpoint index must be nonnegative")
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, (int, float)):
            raise TrajectoryContractError("checkpoint epoch must be a finite real number")
        if not math.isfinite(float(self.epoch)):
            raise TrajectoryContractError("checkpoint epoch must be finite")
        if (
            type(self.body_ids) is not tuple
            or not self.body_ids
            or len(set(self.body_ids)) != len(self.body_ids)
            or any(type(body_id) is not str or not body_id for body_id in self.body_ids)
        ):
            raise TrajectoryContractError("checkpoint body_ids must be an exact nonempty tuple")
        for label in ("backend_id", "device"):
            value = getattr(self, label)
            if type(value) is not str or not value or value.strip() != value:
                raise TrajectoryContractError(f"checkpoint {label} must be explicit")
        for label in ("positions", "velocities"):
            array = getattr(self, label)
            if array is None:
                raise TrajectoryContractError(f"checkpoint {label} cannot be null")
            if (
                getattr(array, "ndim", None) != 2
                or getattr(array, "shape", None) != (len(self.body_ids), 3)
                or str(getattr(array, "dtype", "")) != "float64"
            ):
                raise TrajectoryContractError(
                    f"checkpoint {label} must be float64 with shape (body_count, 3)"
                )
        for label in ("accepted_steps", "rejected_steps"):
            value = getattr(self, label)
            if type(value) is not int or value < 0:
                raise TrajectoryContractError(f"checkpoint {label} must be nonnegative")
        if (
            self.dtype != "float64"
            or self.evidence_class != MODEL_OUTPUT
            or type(self.registry_authorized) is not bool
            or self.registry_authorized
            or type(self.qualification_authorized) is not bool
            or self.qualification_authorized
        ):
            raise TrajectoryContractError("checkpoints remain nonauthorizing MODEL_OUTPUT")

    @property
    def qualified(self) -> bool:
        return False


@dataclass(frozen=True, eq=False)
class TrajectoryResult:
    """Backend-native copies at exact requested checkpoint epochs.

    ``accepted_step_epochs`` and ``accepted_step_magnitudes`` are complete,
    untruncated host ledgers.  Magnitudes retain ``abs(signed_step)`` from the
    actual RKF78 call; the fixed representation makes that argument
    bit-exactly equal to the corresponding representable endpoint delta.

    NumPy arrays retained by a result are disjoint, owned, C-contiguous,
    read-only copies and their complete retained content is bound by
    ``result_content_sha256``.  CuPy does not expose NumPy's write-disable
    mechanism, and the engine's no-implicit-transfer contract forbids copying
    numerical results back to the host merely to hash them.  CuPy results
    therefore retain disjoint owned device copies and set
    ``result_content_sha256`` to ``None``; direct caller mutation remains a
    documented limitation rather than a hidden transfer.
    """

    snapshot_id: str
    plan_id: str
    backend_id: str
    device: str
    dtype: str
    backend_spec: BackendSpec
    initial_snapshot: StateSnapshot
    force_plan: ForcePlan
    integration_spec: AdaptiveRKF78Spec
    checkpoint_epochs: tuple[float, ...]
    checkpoints: tuple[TrajectoryCheckpoint, ...]
    force_model_ids: tuple[str, ...]
    force_ledger: tuple[ForceLedgerEntry, ...]
    accepted_step_epochs: tuple[float, ...]
    accepted_step_magnitudes: tuple[float, ...]
    attempted_steps: int
    accepted_steps: int
    rejected_steps: int
    force_evaluations: int
    minimum_accepted_step: float
    maximum_accepted_step: float
    last_accepted_step: float
    direction: str
    accepted_step_ledger_content_sha256: str
    result_content_sha256: str | None
    accepted_state_accumulation: str = RKF78_ACCEPTED_STATE_ACCUMULATION
    accepted_step_magnitude_source: str = RKF78_ACCEPTED_STEP_MAGNITUDE_SOURCE
    accepted_step_ledger_checksum_algorithm: str = (
        RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_ALGORITHM
    )
    accepted_step_ledger_checksum_domain: str = (
        RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_DOMAIN
    )
    time_step_representation: str = RKF78_TIME_STEP_REPRESENTATION
    checkpoint_proposal_policy: str = RKF78_CHECKPOINT_PROPOSAL_POLICY
    result_content_checksum_algorithm: str = (
        RKF78_RESULT_CONTENT_CHECKSUM_ALGORITHM
    )
    result_content_checksum_domain: str = RKF78_RESULT_CONTENT_CHECKSUM_DOMAIN
    method_id: str = RKF78_METHOD_ID
    tableau_id: str = RKF78_TABLEAU_ID
    tableau_source: str = RKF78_TABLEAU_SOURCE
    stage_count: int = RKF78_STAGE_COUNT
    accepted_order: int = RKF78_ACCEPTED_ORDER
    embedded_order: int = RKF78_EMBEDDED_ORDER
    defect_orientation: str = RKF78_DEFECT_ORIENTATION
    controller_exponent: float = RKF78_CONTROLLER_EXPONENT
    error_norm: str = RKF78_ERROR_NORM
    error_scale: str = RKF78_ERROR_SCALE
    checkpoint_policy: str = RKF78_CHECKPOINT_POLICY
    scope: str = TRAJECTORY_SCOPE
    evidence_class: str = MODEL_OUTPUT
    registry_authorized: bool = False
    qualification_authorized: bool = False
    dense_output: bool = False

    def __post_init__(self) -> None:
        exact = {
            "dtype": "float64",
            "method_id": RKF78_METHOD_ID,
            "tableau_id": RKF78_TABLEAU_ID,
            "tableau_source": RKF78_TABLEAU_SOURCE,
            "stage_count": RKF78_STAGE_COUNT,
            "accepted_order": RKF78_ACCEPTED_ORDER,
            "embedded_order": RKF78_EMBEDDED_ORDER,
            "defect_orientation": RKF78_DEFECT_ORIENTATION,
            "controller_exponent": RKF78_CONTROLLER_EXPONENT,
            "error_norm": RKF78_ERROR_NORM,
            "error_scale": RKF78_ERROR_SCALE,
            "checkpoint_policy": RKF78_CHECKPOINT_POLICY,
            "accepted_state_accumulation": RKF78_ACCEPTED_STATE_ACCUMULATION,
            "accepted_step_magnitude_source": RKF78_ACCEPTED_STEP_MAGNITUDE_SOURCE,
            "accepted_step_ledger_checksum_algorithm": (
                RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_ALGORITHM
            ),
            "accepted_step_ledger_checksum_domain": (
                RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_DOMAIN
            ),
            "time_step_representation": RKF78_TIME_STEP_REPRESENTATION,
            "checkpoint_proposal_policy": RKF78_CHECKPOINT_PROPOSAL_POLICY,
            "result_content_checksum_algorithm": (
                RKF78_RESULT_CONTENT_CHECKSUM_ALGORITHM
            ),
            "result_content_checksum_domain": RKF78_RESULT_CONTENT_CHECKSUM_DOMAIN,
            "scope": TRAJECTORY_SCOPE,
            "evidence_class": MODEL_OUTPUT,
            "registry_authorized": False,
            "qualification_authorized": False,
            "dense_output": False,
        }
        for field, expected in exact.items():
            value = getattr(self, field)
            if type(value) is not type(expected) or value != expected:
                raise TrajectoryContractError(
                    f"{field} must equal the fixed trajectory value {expected!r}"
                )
        for field in ("accepted_step_ledger_content_sha256",):
            value = getattr(self, field)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise TrajectoryContractError(
                    f"{field} must be lowercase SHA-256 hex"
                )
        if self.backend_id == "numpy":
            value = self.result_content_sha256
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise TrajectoryContractError(
                    "NumPy result_content_sha256 must be lowercase SHA-256 hex"
                )
        elif self.backend_id == "cupy" and self.result_content_sha256 is not None:
            raise TrajectoryContractError(
                "CuPy result_content_sha256 must be None under the no-transfer contract"
            )
        _validate_result_bindings(self)

    @property
    def checkpoint_count(self) -> int:
        return len(self.checkpoint_epochs)

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
    def result_content_integrity_status(self) -> str:
        if self.backend_id == "numpy":
            return RKF78_NUMPY_RESULT_CONTENT_INTEGRITY_STATUS
        return RKF78_CUPY_RESULT_CONTENT_INTEGRITY_STATUS

    @property
    def qualified(self) -> bool:
        return False


def _has_any(backend: ArrayBackend, expression: Any, label: str) -> bool:
    return backend.scalar_bool(backend.xp.any(expression), label)


def _native_scalar_float(value: object, label: str) -> float:
    """Synchronize exactly one controller scalar from the selected backend."""

    try:
        item = value.item()  # NumPy/CuPy zero-dimensional float scalar.
    except (AttributeError, TypeError, ValueError) as exc:
        raise TrajectoryDomainError(f"{label} did not produce a backend scalar") from exc
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise TrajectoryDomainError(f"{label} did not produce a real scalar")
    checked = float(item)
    if not math.isfinite(checked):
        raise TrajectoryDomainError(f"{label} must be finite")
    return checked


def _accepted_step_ledger_content_sha256(
    *,
    accepted_step_epochs: tuple[float, ...],
    accepted_step_magnitudes: tuple[float, ...],
    checkpoints: tuple[TrajectoryCheckpoint, ...],
    direction: str,
    attempted_steps: int,
    accepted_steps: int,
    rejected_steps: int,
    force_evaluations: int,
    minimum_accepted_step: float,
    maximum_accepted_step: float,
    last_accepted_step: float,
    magnitude_source: str,
) -> str:
    """Return an unauthenticated, domain-separated content checksum."""

    payload = {
        "accepted_step_epochs_hex": [
            value.hex() for value in accepted_step_epochs
        ],
        "accepted_step_magnitudes_hex": [
            value.hex() for value in accepted_step_magnitudes
        ],
        "checkpoint_endpoint_bindings": [
            {
                "accepted_steps": checkpoint.accepted_steps,
                "epoch_hex": float(checkpoint.epoch).hex(),
                "index": checkpoint.index,
                "rejected_steps": checkpoint.rejected_steps,
            }
            for checkpoint in checkpoints
        ],
        "checksum_algorithm": RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_ALGORITHM,
        "counters": {
            "accepted_steps": accepted_steps,
            "attempted_steps": attempted_steps,
            "force_evaluations": force_evaluations,
            "rejected_steps": rejected_steps,
        },
        "direction": direction,
        "extrema_hex": {
            "last": last_accepted_step.hex(),
            "maximum": maximum_accepted_step.hex(),
            "minimum": minimum_accepted_step.hex(),
        },
        "magnitude_source": magnitude_source,
    }
    serialized = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    preimage = (
        RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_DOMAIN.encode("utf-8")
        + b"\x00"
        + serialized
    )
    return hashlib.sha256(preimage).hexdigest()


def _canonical_result_content(backend: ArrayBackend, value: object) -> Any:
    """Return exact JSON-safe result content without implicit array coercion.

    The content digest is a NumPy-only contract. CuPy arrays are rejected so
    this helper can never introduce an implicit device-to-host transfer.
    """

    if value is None:
        return None
    if type(value) in (bool, int, str):
        return value
    if isinstance(value, float):
        canonical_float = float(value)
        if not math.isfinite(canonical_float):
            raise TrajectoryContractError(
                "RKF78 result checksum content must contain only finite floats"
            )
        return {"binary64_hex": canonical_float.hex()}
    if type(value) is tuple:
        return [_canonical_result_content(backend, item) for item in value]
    if type(value) is list:
        return [_canonical_result_content(backend, item) for item in value]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TrajectoryContractError(
                "RKF78 result checksum mappings require built-in string keys"
            )
        return {
            key: _canonical_result_content(backend, item)
            for key, item in value.items()
        }
    if type(value) is backend.array_type:
        if value.dtype not in (backend.xp.dtype("float64"), backend.xp.dtype("bool")):
            raise TrajectoryContractError(
                "RKF78 result checksum arrays must be float64 or bool"
            )
        if backend.name != "numpy":
            raise TrajectoryContractError(
                "CuPy result content is not host-hashed implicitly"
            )
        host_value = value
        if str(value.dtype) == "float64":
            values = [float(item).hex() for item in host_value.reshape(-1)]
        else:
            values = [bool(item) for item in host_value.reshape(-1)]
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": values,
        }
    if is_dataclass(value) and type(value).__module__.startswith("jxplanetx."):
        return {
            "dataclass": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                field.name: _canonical_result_content(
                    backend, getattr(value, field.name)
                )
                for field in fields(value)
            },
        }
    raise TrajectoryContractError(
        f"unsupported RKF78 result checksum content type {type(value).__name__!r}"
    )


def _result_content_sha256(
    *,
    backend: ArrayBackend,
    initial_snapshot: StateSnapshot,
    force_plan: ForcePlan,
    integration_spec: AdaptiveRKF78Spec,
    checkpoints: tuple[TrajectoryCheckpoint, ...],
    force_model_ids: tuple[str, ...],
    force_ledger: tuple[ForceLedgerEntry, ...],
    accepted_step_epochs: tuple[float, ...],
    accepted_step_magnitudes: tuple[float, ...],
    attempted_steps: int,
    accepted_steps: int,
    rejected_steps: int,
    force_evaluations: int,
    minimum_accepted_step: float,
    maximum_accepted_step: float,
    last_accepted_step: float,
    direction: str,
    accepted_step_ledger_content_sha256: str,
) -> str:
    """Hash all retained adaptive-RKF78 result content; not authentication."""

    if backend.name != "numpy":
        raise TrajectoryContractError(
            "full RKF78 result-content hashing is available only for NumPy; "
            "CuPy numerical results remain device resident"
        )

    payload = {
        "accepted_step_epochs": accepted_step_epochs,
        "accepted_step_ledger_content_sha256": (
            accepted_step_ledger_content_sha256
        ),
        "accepted_step_magnitudes": accepted_step_magnitudes,
        "accounting": {
            "accepted_steps": accepted_steps,
            "attempted_steps": attempted_steps,
            "force_evaluations": force_evaluations,
            "rejected_steps": rejected_steps,
        },
        "checkpoints": checkpoints,
        "checksum_algorithm": RKF78_RESULT_CONTENT_CHECKSUM_ALGORITHM,
        "control": {
            "dense_output": False,
            "evidence_class": MODEL_OUTPUT,
            "qualification_authorized": False,
            "registry_authorized": False,
            "scope": TRAJECTORY_SCOPE,
        },
        "direction": direction,
        "extrema": {
            "last_accepted_step": last_accepted_step,
            "maximum_accepted_step": maximum_accepted_step,
            "minimum_accepted_step": minimum_accepted_step,
        },
        "force_ledger": force_ledger,
        "force_model_ids": force_model_ids,
        "force_plan": force_plan,
        "initial_snapshot": initial_snapshot,
        "integration_spec": integration_spec,
        "schema": "jxplanetx.adaptive-rkf78-result.v1",
    }
    serialized = json.dumps(
        _canonical_result_content(backend, payload),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    preimage = (
        RKF78_RESULT_CONTENT_CHECKSUM_DOMAIN.encode("utf-8")
        + b"\x00"
        + serialized
    )
    return hashlib.sha256(preimage).hexdigest()


def _validate_atol(
    backend: ArrayBackend,
    value: object,
    label: str,
    state_shape: tuple[int, int],
) -> Any:
    array = backend.require_native_array(value, label)
    if array.dtype != backend.float64:
        raise TrajectoryContractError(f"{label} must have dtype float64")
    if array.ndim != 2 or array.shape != state_shape:
        raise TrajectoryContractError(f"{label} must have shape (body_count, 3)")
    if _has_any(backend, ~backend.xp.isfinite(array), f"{label} finiteness check"):
        raise TrajectoryDomainError(f"{label} must contain only finite values")
    if _has_any(backend, array <= backend.xp.float64(0.0), f"{label} positivity check"):
        raise TrajectoryDomainError(f"{label} must be strictly positive componentwise")
    return array


def _validate_force_compatibility(plan: ForcePlan) -> None:
    for model in plan.models:
        capability = get_capability(model.model_id)  # type: ignore[union-attr]
        if capability.code_status is not CodeStatus.IMPLEMENTED:
            raise TrajectoryContractError(
                f"{capability.model_id} is not an implemented force model"
            )
        if "GENERAL_FIRST_ORDER" not in capability.compatible_integrators:
            raise TrajectoryContractError(
                f"{capability.model_id} is incompatible with a general first-order RK method"
            )


def _validate_restricted_central_velocity(
    backend: ArrayBackend,
    snapshot: StateSnapshot,
    velocities: Any,
    models: tuple[object, ...],
) -> None:
    """Bind the restricted static-central assumption to retained state."""

    for model in models:
        if type(model) is RestrictedStaticCentral1PN:
            central_index = snapshot.index_of(model.central_source_id)
            if _has_any(
                backend,
                velocities[central_index] != backend.xp.float64(0.0),
                "restricted 1PN central-velocity check",
            ):
                raise TrajectoryContractError(
                    "restricted 1PN central_source_id must be exactly static"
                )


def _validate_metadata_range(
    plan: ForcePlan,
    lower_epoch: float,
    upper_epoch: float,
) -> None:
    for model in plan.models:
        for metadata in model.parameter_metadata:  # type: ignore[union-attr]
            if metadata.validity_start > lower_epoch or metadata.validity_end < upper_epoch:
                raise TrajectoryContractError(
                    f"{metadata.parameter_id} validity interval does not cover the full trajectory"
                )


def _expected_ledger_scope(model: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if type(model) is NewtonianPointMass:
        return model.source_ids, model.target_ids
    if type(model) is RestrictedStaticCentral1PN:
        return (model.central_source_id,), model.target_ids
    if type(model) is CannonballSRP:
        return (model.radiation_source_id,), model.target_ids
    raise TrajectoryContractError("force plan contains an unsupported model binding")


def _validate_result_bindings(result: TrajectoryResult) -> None:
    """Revalidate every public result binding and reject replacement tampering."""

    if type(result.snapshot_id) is not str or not result.snapshot_id:
        raise TrajectoryContractError("snapshot_id must be nonempty")
    if type(result.plan_id) is not str or not result.plan_id:
        raise TrajectoryContractError("plan_id must be nonempty")
    if type(result.backend_spec) is not BackendSpec:
        raise TrajectoryContractError("backend_spec must be BackendSpec")
    if type(result.initial_snapshot) is not StateSnapshot:
        raise TrajectoryContractError("initial_snapshot must be StateSnapshot")
    if type(result.force_plan) is not ForcePlan:
        raise TrajectoryContractError("force_plan must be ForcePlan")
    if type(result.integration_spec) is not AdaptiveRKF78Spec:
        raise TrajectoryContractError("integration_spec must be AdaptiveRKF78Spec")
    if result.snapshot_id != result.initial_snapshot.snapshot_id:
        raise TrajectoryContractError("snapshot_id does not match initial_snapshot")
    if result.plan_id != result.force_plan.plan_id:
        raise TrajectoryContractError("plan_id does not match force_plan")
    if result.backend_spec is not result.force_plan.backend:
        raise TrajectoryContractError("backend_spec must be the retained force-plan backend")
    if result.initial_snapshot.epoch != result.integration_spec.checkpoint_epochs[0]:
        raise TrajectoryContractError("initial snapshot epoch does not match integration_spec")
    if result.checkpoint_epochs != result.integration_spec.checkpoint_epochs:
        raise TrajectoryContractError("checkpoint_epochs do not match integration_spec")
    if result.direction != result.integration_spec.direction:
        raise TrajectoryContractError("direction does not match the checkpoint schedule")

    _validate_model_sequence(result.force_plan.models)
    _validate_dependencies(result.force_plan.models)
    _validate_parameter_contracts(result.initial_snapshot, result.force_plan)
    _validate_restricted_frame(result.initial_snapshot, result.force_plan.models)
    _validate_force_compatibility(result.force_plan)
    _validate_metadata_range(
        result.force_plan,
        min(result.checkpoint_epochs[0], result.checkpoint_epochs[-1]),
        max(result.checkpoint_epochs[0], result.checkpoint_epochs[-1]),
    )

    backend = resolve_backend(result.backend_spec)
    if result.backend_id != backend.name or result.device != backend.device:
        raise TrajectoryContractError("result backend identity does not match backend_spec")
    with backend.activate():
        initial_state = _validate_state(backend, result.initial_snapshot)
        _validate_restricted_central_position(
            backend,
            result.initial_snapshot,
            initial_state[0],
            result.force_plan.models,
        )
        _validate_restricted_central_velocity(
            backend,
            result.initial_snapshot,
            initial_state[1],
            result.force_plan.models,
        )
        body_count = len(result.initial_snapshot.body_ids)
        state_shape = (body_count, 3)
        position_atol = _validate_atol(
            backend,
            result.integration_spec.position_atol,
            "retained position_atol",
            state_shape,
        )
        velocity_atol = _validate_atol(
            backend,
            result.integration_spec.velocity_atol,
            "retained velocity_atol",
            state_shape,
        )

        retained_arrays: list[Any] = []
        for label, array in zip(
            (
                "initial_snapshot.positions",
                "initial_snapshot.velocities",
                "initial_snapshot.gravitational_parameters",
                "initial_snapshot.masses",
                "initial_snapshot.radii",
                "initial_snapshot.massive",
            ),
            initial_state,
        ):
            _validate_retained_array_custody(
                backend, array, label, retained_arrays
            )
        _validate_retained_array_custody(
            backend,
            position_atol,
            "integration_spec.position_atol",
            retained_arrays,
        )
        _validate_retained_array_custody(
            backend,
            velocity_atol,
            "integration_spec.velocity_atol",
            retained_arrays,
        )

        for model in result.force_plan.models:
            if type(model) is not CannonballSRP:
                continue
            for label, value in (
                ("retained area_to_mass", model.area_to_mass),
                (
                    "retained radiation_pressure_coefficient",
                    model.radiation_pressure_coefficient,
                ),
            ):
                array = backend.require_native_array(value, label)
                if (
                    array.dtype != backend.float64
                    or array.ndim != 1
                    or array.shape != (len(model.target_ids),)
                ):
                    raise TrajectoryContractError(
                        f"{label} must be float64 with shape (len(target_ids),)"
                    )
                if _has_any(
                    backend,
                    (~backend.xp.isfinite(array)) | (array < backend.xp.float64(0.0)),
                    f"{label} domain check",
                ):
                    raise TrajectoryDomainError(f"{label} must be finite and nonnegative")
                _validate_retained_array_custody(
                    backend, array, f"force_plan.{label}", retained_arrays
                )

        if type(result.checkpoints) is not tuple or len(result.checkpoints) != len(
            result.checkpoint_epochs
        ):
            raise TrajectoryContractError("checkpoints must align exactly with checkpoint_epochs")
        previous_accepted = 0
        previous_rejected = 0
        for index, (checkpoint, epoch) in enumerate(
            zip(result.checkpoints, result.checkpoint_epochs)
        ):
            if type(checkpoint) is not TrajectoryCheckpoint:
                raise TrajectoryContractError("checkpoints must contain TrajectoryCheckpoint")
            if checkpoint.index != index or checkpoint.epoch != epoch:
                raise TrajectoryContractError("checkpoint index or epoch binding is inconsistent")
            if (
                checkpoint.body_ids != result.initial_snapshot.body_ids
                or checkpoint.backend_id != result.backend_id
                or checkpoint.device != result.device
                or checkpoint.dtype != result.dtype
            ):
                raise TrajectoryContractError("checkpoint state/backend binding is inconsistent")
            _validate_trial_arrays(
                backend,
                (
                    (f"checkpoint[{index}] positions", checkpoint.positions),
                    (f"checkpoint[{index}] velocities", checkpoint.velocities),
                ),
                state_shape,
            )
            for name, array in (
                ("positions", checkpoint.positions),
                ("velocities", checkpoint.velocities),
            ):
                _validate_retained_array_custody(
                    backend,
                    array,
                    f"checkpoint[{index}].{name}",
                    retained_arrays,
                )
            if index == 0:
                if checkpoint.accepted_steps != 0 or checkpoint.rejected_steps != 0:
                    raise TrajectoryContractError("initial checkpoint counters must be zero")
                if _has_any(
                    backend,
                    checkpoint.positions != result.initial_snapshot.positions,
                    "initial checkpoint position binding check",
                ) or _has_any(
                    backend,
                    checkpoint.velocities != result.initial_snapshot.velocities,
                    "initial checkpoint velocity binding check",
                ):
                    raise TrajectoryContractError(
                        "initial checkpoint state differs from initial_snapshot"
                    )
            else:
                if checkpoint.accepted_steps <= previous_accepted:
                    raise TrajectoryContractError(
                        "every checkpoint interval requires an accepted clipped step"
                    )
                if checkpoint.rejected_steps < previous_rejected:
                    raise TrajectoryContractError("checkpoint rejection counters are not monotone")
            previous_accepted = checkpoint.accepted_steps
            previous_rejected = checkpoint.rejected_steps

    expected_model_ids = tuple(model.model_id for model in result.force_plan.models)
    if result.force_model_ids != expected_model_ids:
        raise TrajectoryContractError("force_model_ids do not match the retained force plan")
    if type(result.force_ledger) is not tuple or len(result.force_ledger) != len(
        expected_model_ids
    ):
        raise TrajectoryContractError("force_ledger does not match the retained force plan")
    for order, (entry, model_id, model) in enumerate(
        zip(result.force_ledger, expected_model_ids, result.force_plan.models)
    ):
        if type(entry) is not ForceLedgerEntry:
            raise TrajectoryContractError("force_ledger contains an invalid entry")
        expected_sources, expected_targets = _expected_ledger_scope(model)
        if (
            entry.order != order
            or entry.model_id != model_id
            or entry.source_ids != expected_sources
            or entry.target_ids != expected_targets
            or entry.backend_spec is not result.backend_spec
            or entry.tile_size != result.backend_spec.tile_size
            or entry.state_metadata.snapshot_id != result.snapshot_id
            or entry.state_metadata.epoch != result.checkpoint_epochs[0]
            or entry.state_metadata.body_ids != result.initial_snapshot.body_ids
            or entry.state_metadata.provenance_sha256
            != result.initial_snapshot.provenance.sha256
            or entry.evidence_class != MODEL_OUTPUT
            or entry.registry_authorized
            or entry.qualification_authorized
        ):
            raise TrajectoryContractError("force-ledger binding is inconsistent")

    for label in (
        "attempted_steps",
        "accepted_steps",
        "rejected_steps",
        "force_evaluations",
    ):
        value = getattr(result, label)
        if type(value) is not int or value < 0:
            raise TrajectoryContractError(f"{label} must be a nonnegative integer")
    if result.attempted_steps != result.accepted_steps + result.rejected_steps:
        raise TrajectoryContractError("attempted step accounting is inconsistent")
    if result.accepted_steps < len(result.checkpoint_epochs) - 1:
        raise TrajectoryContractError("accepted steps cannot cover every checkpoint interval")
    if result.attempted_steps > result.integration_spec.maximum_steps:
        raise TrajectoryContractError("attempted steps exceed maximum_steps")
    if result.rejected_steps > result.integration_spec.maximum_rejections:
        raise TrajectoryContractError("rejected steps exceed maximum_rejections")
    if result.force_evaluations != result.attempted_steps * RKF78_STAGE_COUNT:
        raise TrajectoryContractError("force-evaluation accounting is inconsistent")
    if (
        result.checkpoints[-1].accepted_steps != result.accepted_steps
        or result.checkpoints[-1].rejected_steps != result.rejected_steps
    ):
        raise TrajectoryContractError("final checkpoint counters do not match the result")

    if (
        type(result.accepted_step_epochs) is not tuple
        or len(result.accepted_step_epochs) != result.accepted_steps
    ):
        raise TrajectoryContractError(
            "accepted_step_epochs must contain every accepted endpoint exactly once"
        )
    if (
        type(result.accepted_step_magnitudes) is not tuple
        or len(result.accepted_step_magnitudes) != result.accepted_steps
    ):
        raise TrajectoryContractError(
            "accepted_step_magnitudes must contain every accepted RKF78 step argument"
        )
    for index, magnitude in enumerate(result.accepted_step_magnitudes):
        if type(magnitude) is not float or not math.isfinite(magnitude) or magnitude <= 0.0:
            raise TrajectoryContractError(
                f"accepted_step_magnitudes[{index}] must be a finite positive float"
            )

    previous_epoch = float(result.checkpoint_epochs[0])
    direction_sign = 1.0 if result.direction == "FORWARD" else -1.0
    for index, (endpoint, magnitude) in enumerate(
        zip(result.accepted_step_epochs, result.accepted_step_magnitudes)
    ):
        if type(endpoint) is not float or not math.isfinite(endpoint):
            raise TrajectoryContractError(
                f"accepted_step_epochs[{index}] must be a finite float"
            )
        signed_step = endpoint - previous_epoch
        if direction_sign * signed_step <= 0.0:
            raise TrajectoryContractError(
                "accepted_step_epochs must be strictly monotone in result direction"
            )
        observed_magnitude = abs(signed_step)
        if observed_magnitude != magnitude:
            raise TrajectoryContractError(
                "accepted step magnitude must bit-exactly equal its representable endpoint delta"
            )
        previous_epoch = endpoint

    previous_count = 0
    for checkpoint in result.checkpoints[1:]:
        if checkpoint.accepted_steps <= previous_count:
            raise TrajectoryContractError(
                "checkpoint accepted-step counters must advance at every interval"
            )
        endpoint_index = checkpoint.accepted_steps - 1
        if (
            endpoint_index >= len(result.accepted_step_epochs)
            or result.accepted_step_epochs[endpoint_index] != float(checkpoint.epoch)
        ):
            raise TrajectoryContractError(
                "accepted-step endpoint ledger does not hit a checkpoint exactly"
            )
        previous_count = checkpoint.accepted_steps
    if previous_count != result.accepted_steps:
        raise TrajectoryContractError(
            "accepted-step endpoint ledger does not end at the final checkpoint"
        )

    for label in (
        "minimum_accepted_step",
        "maximum_accepted_step",
        "last_accepted_step",
    ):
        value = getattr(result, label)
        if type(value) is not float or not math.isfinite(value) or value <= 0.0:
            raise TrajectoryContractError(f"{label} must be a finite positive float")
    exact_extrema = (
        float(min(result.accepted_step_magnitudes)),
        float(max(result.accepted_step_magnitudes)),
        float(result.accepted_step_magnitudes[-1]),
    )
    observed_extrema = (
        result.minimum_accepted_step,
        result.maximum_accepted_step,
        result.last_accepted_step,
    )
    if observed_extrema != exact_extrema:
        raise TrajectoryContractError(
            "accepted step extrema do not match the complete endpoint ledger"
        )
    if result.maximum_accepted_step > float(result.integration_spec.maximum_step):
        raise TrajectoryContractError("an accepted step exceeds maximum_step")
    expected_content_sha256 = _accepted_step_ledger_content_sha256(
        accepted_step_epochs=result.accepted_step_epochs,
        accepted_step_magnitudes=result.accepted_step_magnitudes,
        checkpoints=result.checkpoints,
        direction=result.direction,
        attempted_steps=result.attempted_steps,
        accepted_steps=result.accepted_steps,
        rejected_steps=result.rejected_steps,
        force_evaluations=result.force_evaluations,
        minimum_accepted_step=result.minimum_accepted_step,
        maximum_accepted_step=result.maximum_accepted_step,
        last_accepted_step=result.last_accepted_step,
        magnitude_source=result.accepted_step_magnitude_source,
    )
    if result.accepted_step_ledger_content_sha256 != expected_content_sha256:
        raise TrajectoryContractError(
            "accepted-step ledger content checksum does not match its exact float-hex content"
        )
    if backend.name == "numpy":
        with backend.activate():
            expected_result_content_sha256 = _result_content_sha256(
                backend=backend,
                initial_snapshot=result.initial_snapshot,
                force_plan=result.force_plan,
                integration_spec=result.integration_spec,
                checkpoints=result.checkpoints,
                force_model_ids=result.force_model_ids,
                force_ledger=result.force_ledger,
                accepted_step_epochs=result.accepted_step_epochs,
                accepted_step_magnitudes=result.accepted_step_magnitudes,
                attempted_steps=result.attempted_steps,
                accepted_steps=result.accepted_steps,
                rejected_steps=result.rejected_steps,
                force_evaluations=result.force_evaluations,
                minimum_accepted_step=result.minimum_accepted_step,
                maximum_accepted_step=result.maximum_accepted_step,
                last_accepted_step=result.last_accepted_step,
                direction=result.direction,
                accepted_step_ledger_content_sha256=(
                    result.accepted_step_ledger_content_sha256
                ),
            )
        if result.result_content_sha256 != expected_result_content_sha256:
            raise TrajectoryContractError(
                "RKF78 result content checksum does not match the retained trajectory"
            )
    elif result.result_content_sha256 is not None:
        raise TrajectoryContractError(
            "CuPy result content hash would violate the no-transfer contract"
        )


def validate_trajectory_result_integrity(result: TrajectoryResult) -> None:
    """Revalidate one retained result without granting scientific authority.

    NumPy content is checked against its full retained-content digest.  CuPy
    validation remains structural because device arrays deliberately stay on
    device and remain caller-mutable under the documented backend contract.
    """

    if type(result) is not TrajectoryResult:
        raise TrajectoryContractError("result must be an exact TrajectoryResult")
    _validate_result_bindings(result)


def _validate_trial_arrays(
    backend: ArrayBackend,
    arrays: tuple[tuple[str, Any], ...],
    state_shape: tuple[int, int],
) -> None:
    for label, array in arrays:
        backend.require_native_array(array, label)
        if array.dtype != backend.float64 or array.ndim != 2 or array.shape != state_shape:
            raise TrajectoryContractError(
                f"{label} must remain a native float64 (body_count, 3) array"
            )
        if _has_any(backend, ~backend.xp.isfinite(array), f"{label} finiteness check"):
            raise TrajectoryDomainError(f"{label} contains a nonfinite value")


def _owned_retained_copy(backend: ArrayBackend, array: Any) -> Any:
    """Copy one retained buffer and apply the backend's custody guarantee."""

    copied = backend.xp.copy(array, order="C")
    if backend.name == "numpy":
        copied.setflags(write=False)
    # CuPy 14 has neither ndarray.setflags nor a writeable flag. A disjoint
    # owning device copy is its strongest no-transfer custody mechanism;
    # TrajectoryResult documents the residual mutability and absent host hash.
    return copied


def _validate_retained_array_custody(
    backend: ArrayBackend,
    array: Any,
    label: str,
    retained_arrays: list[Any],
) -> None:
    """Require exact owned, disjoint result buffers and NumPy read-only state."""

    if type(array) is not backend.array_type:
        raise TrajectoryContractError(
            f"{label} must be an exact {backend.name} ndarray"
        )
    if not bool(array.flags.c_contiguous):
        raise TrajectoryContractError(f"{label} must be C-contiguous")
    if backend.name == "numpy":
        if not bool(array.flags.owndata) or bool(array.flags.writeable):
            raise TrajectoryContractError(
                f"{label} must own its memory and be read-only"
            )
    elif array.base is not None:
        raise TrajectoryContractError(
            f"{label} must be an owning CuPy array rather than a view"
        )
    for previous in retained_arrays:
        try:
            overlaps = bool(backend.xp.shares_memory(array, previous))
        except Exception as exc:
            raise TrajectoryContractError(
                "retained RKF78 array overlap could not be resolved exactly"
            ) from exc
        if overlaps:
            raise TrajectoryContractError(
                "retained RKF78 arrays must not overlap memory"
            )
    retained_arrays.append(array)


def _copy_initial_snapshot(
    backend: ArrayBackend,
    snapshot: StateSnapshot,
    validated_state: tuple[Any, Any, Any, Any, Any, Any],
) -> StateSnapshot:
    positions, velocities, gravitational_parameters, masses, radii, massive = (
        validated_state
    )
    return replace(
        snapshot,
        positions=_owned_retained_copy(backend, positions),
        velocities=_owned_retained_copy(backend, velocities),
        gravitational_parameters=_owned_retained_copy(
            backend, gravitational_parameters
        ),
        masses=_owned_retained_copy(backend, masses),
        radii=_owned_retained_copy(backend, radii),
        massive=_owned_retained_copy(backend, massive),
    )


def _copy_force_plan(backend: ArrayBackend, plan: ForcePlan) -> ForcePlan:
    copied_models: list[object] = []
    for model in plan.models:
        if type(model) is CannonballSRP:
            area_to_mass = backend.require_native_array(
                model.area_to_mass, "area_to_mass"
            )
            coefficient = backend.require_native_array(
                model.radiation_pressure_coefficient,
                "radiation_pressure_coefficient",
            )
            copied_models.append(
                replace(
                    model,
                    area_to_mass=_owned_retained_copy(backend, area_to_mass),
                    radiation_pressure_coefficient=_owned_retained_copy(
                        backend, coefficient
                    ),
                )
            )
        else:
            copied_models.append(replace(model))
    return replace(
        plan,
        backend=replace(plan.backend),
        models=tuple(copied_models),
    )


def _copy_integration_spec(
    backend: ArrayBackend,
    spec: AdaptiveRKF78Spec,
    position_atol: Any,
    velocity_atol: Any,
) -> AdaptiveRKF78Spec:
    return replace(
        spec,
        position_atol=_owned_retained_copy(backend, position_atol),
        velocity_atol=_owned_retained_copy(backend, velocity_atol),
    )


class _StageEvaluator:
    """Build complete stage snapshots and delegate every force evaluation."""

    def __init__(
        self,
        snapshot: StateSnapshot,
        plan: ForcePlan,
        backend: ArrayBackend,
        state_shape: tuple[int, int],
        lower_epoch: float,
        upper_epoch: float,
    ) -> None:
        self.snapshot = snapshot
        self.plan = plan
        self.backend = backend
        self.state_shape = state_shape
        self.lower_epoch = lower_epoch
        self.upper_epoch = upper_epoch
        self.force_evaluations = 0
        self.force_ledger: tuple[ForceLedgerEntry, ...] | None = None
        self.force_model_ids: tuple[str, ...] | None = None

    def __call__(self, epoch: float, positions: Any, velocities: Any) -> tuple[Any, Any]:
        if not math.isfinite(epoch) or not self.lower_epoch <= epoch <= self.upper_epoch:
            raise TrajectoryDomainError("an RKF78 stage epoch left the requested range")
        _validate_trial_arrays(
            self.backend,
            (("stage positions", positions), ("stage velocities", velocities)),
            self.state_shape,
        )
        stage_snapshot = replace(
            self.snapshot,
            epoch=epoch,
            positions=positions,
            velocities=velocities,
        )
        force_result = evaluate_force_plan(stage_snapshot, self.plan)
        self.force_evaluations += 1
        if self.force_ledger is None:
            self.force_ledger = force_result.ledger
            self.force_model_ids = force_result.applied_model_ids
        elif force_result.applied_model_ids != self.force_model_ids:
            raise TrajectoryContractError("force-plan identity changed between RKF78 stages")
        return velocities, force_result.total_acceleration


def _normalized_max_error(
    *,
    backend: ArrayBackend,
    current_positions: Any,
    current_velocities: Any,
    candidate_positions: Any,
    candidate_velocities: Any,
    position_defect: Any,
    velocity_defect: Any,
    position_atol: Any,
    position_rtol: float,
    velocity_atol: Any,
    velocity_rtol: float,
    state_shape: tuple[int, int],
) -> float:
    _validate_trial_arrays(
        backend,
        (
            ("accepted positions", candidate_positions),
            ("accepted velocities", candidate_velocities),
            ("position defect", position_defect),
            ("velocity defect", velocity_defect),
        ),
        state_shape,
    )
    xp = backend.xp
    position_scale = position_atol + xp.float64(position_rtol) * xp.maximum(
        xp.abs(current_positions), xp.abs(candidate_positions)
    )
    velocity_scale = velocity_atol + xp.float64(velocity_rtol) * xp.maximum(
        xp.abs(current_velocities), xp.abs(candidate_velocities)
    )
    if _has_any(
        backend,
        (~xp.isfinite(position_scale)) | (position_scale <= xp.float64(0.0)),
        "position error-scale domain check",
    ):
        raise TrajectoryDomainError("position error scale must remain finite and positive")
    if _has_any(
        backend,
        (~xp.isfinite(velocity_scale)) | (velocity_scale <= xp.float64(0.0)),
        "velocity error-scale domain check",
    ):
        raise TrajectoryDomainError("velocity error scale must remain finite and positive")
    position_error = xp.max(xp.abs(position_defect) / position_scale)
    velocity_error = xp.max(xp.abs(velocity_defect) / velocity_scale)
    normalized = xp.maximum(position_error, velocity_error)
    checked = _native_scalar_float(normalized, "normalized maximum error")
    if checked < 0.0:
        raise TrajectoryDomainError("normalized maximum error cannot be negative")
    return checked


def _controller_factor(error: float, spec: AdaptiveRKF78Spec) -> float:
    if error == 0.0:
        return float(spec.maximum_scale_factor)
    raw = float(spec.safety_factor) * error ** (-float(spec.controller_exponent))
    return min(
        float(spec.maximum_scale_factor),
        max(float(spec.minimum_scale_factor), raw),
    )


def _trial_endpoint(
    current_epoch: float,
    checkpoint_epoch: float,
    direction: float,
    proposed_step: float,
) -> tuple[float, float, bool]:
    remaining = checkpoint_epoch - current_epoch
    if direction * remaining <= 0.0:
        raise TrajectoryContractError("checkpoint direction changed during integration")
    if proposed_step >= abs(remaining):
        return checkpoint_epoch, remaining, True
    proposed_signed_step = math.copysign(proposed_step, direction)
    endpoint = current_epoch + proposed_signed_step
    if endpoint == current_epoch:
        raise TrajectoryStepLimitError("step size cannot advance the binary64 epoch")
    if not math.isfinite(endpoint):
        raise TrajectoryStepLimitError("step size produced a nonfinite binary64 epoch")
    actual_step = endpoint - current_epoch
    if direction * actual_step <= 0.0:
        raise TrajectoryStepLimitError("step size advanced in the wrong direction")
    if abs(actual_step) > proposed_step:
        endpoint = math.nextafter(endpoint, current_epoch)
        if endpoint == current_epoch:
            raise TrajectoryStepLimitError("step size cannot advance the binary64 epoch")
        actual_step = endpoint - current_epoch
        if direction * actual_step <= 0.0 or abs(actual_step) > proposed_step:
            raise TrajectoryStepLimitError(
                "no representable endpoint satisfies the proposed step bound"
            )
    if direction * (checkpoint_epoch - endpoint) <= 0.0:
        if abs(remaining) > proposed_step:
            raise TrajectoryStepLimitError(
                "no representable checkpoint endpoint satisfies the proposed step bound"
            )
        return checkpoint_epoch, remaining, True
    return endpoint, actual_step, False


def integrate_trajectory(
    snapshot: StateSnapshot,
    plan: ForcePlan,
    spec: AdaptiveRKF78Spec,
) -> TrajectoryResult:
    """Integrate one force plan to exact monotone checkpoint epochs.

    All state, tolerance, stage, defect, accepted-state, and result arrays stay
    native to the explicitly requested backend.  Host observation is limited
    to scalar domain decisions and the adaptive controller.
    """

    if type(snapshot) is not StateSnapshot:
        raise TrajectoryContractError("snapshot must be an exact StateSnapshot")
    if type(plan) is not ForcePlan:
        raise TrajectoryContractError("plan must be an exact ForcePlan")
    if type(spec) is not AdaptiveRKF78Spec:
        raise TrajectoryContractError("spec must be an exact AdaptiveRKF78Spec")
    if spec.checkpoint_epochs[0] != snapshot.epoch:
        raise TrajectoryContractError(
            "checkpoint_epochs[0] must exactly equal StateSnapshot.epoch"
        )

    _validate_model_sequence(plan.models)
    _validate_dependencies(plan.models)
    _validate_parameter_contracts(snapshot, plan)
    _validate_restricted_frame(snapshot, plan.models)
    _validate_force_compatibility(plan)
    lower_epoch = min(spec.checkpoint_epochs[0], spec.checkpoint_epochs[-1])
    upper_epoch = max(spec.checkpoint_epochs[0], spec.checkpoint_epochs[-1])
    _validate_metadata_range(plan, lower_epoch, upper_epoch)

    backend = resolve_backend(plan.backend)
    direction = 1.0 if spec.direction == "FORWARD" else -1.0
    with backend.activate():
        validated_state = _validate_state(backend, snapshot)
        _validate_restricted_central_position(
            backend,
            snapshot,
            validated_state[0],
            plan.models,
        )
        _validate_restricted_central_velocity(
            backend,
            snapshot,
            validated_state[1],
            plan.models,
        )
        state_shape = (len(snapshot.body_ids), 3)
        caller_position_atol = _validate_atol(
            backend, spec.position_atol, "position_atol", state_shape
        )
        caller_velocity_atol = _validate_atol(
            backend, spec.velocity_atol, "velocity_atol", state_shape
        )
        initial_snapshot = _copy_initial_snapshot(backend, snapshot, validated_state)
        force_plan = _copy_force_plan(backend, plan)
        integration_spec = _copy_integration_spec(
            backend,
            spec,
            caller_position_atol,
            caller_velocity_atol,
        )
        position_atol = integration_spec.position_atol
        velocity_atol = integration_spec.velocity_atol

        # Integration owns evolving copies independent from both caller buffers
        # and the immutable initial-state binding retained in the result.
        current_positions = backend.xp.copy(initial_snapshot.positions)
        current_velocities = backend.xp.copy(initial_snapshot.velocities)
        current_position_carry = backend.xp.zeros_like(
            current_positions, dtype=backend.xp.float64
        )
        current_velocity_carry = backend.xp.zeros_like(
            current_velocities, dtype=backend.xp.float64
        )
        initial_checkpoint_positions = _owned_retained_copy(
            backend, current_positions
        )
        initial_checkpoint_velocities = _owned_retained_copy(
            backend, current_velocities
        )
        checkpoints = [
            TrajectoryCheckpoint(
                index=0,
                epoch=integration_spec.checkpoint_epochs[0],
                body_ids=initial_snapshot.body_ids,
                backend_id=backend.name,
                device=backend.device,
                positions=initial_checkpoint_positions,
                velocities=initial_checkpoint_velocities,
                accepted_steps=0,
                rejected_steps=0,
            )
        ]

        stage_evaluator = _StageEvaluator(
            initial_snapshot,
            force_plan,
            backend,
            state_shape,
            lower_epoch,
            upper_epoch,
        )
        current_epoch = integration_spec.checkpoint_epochs[0]
        proposed_step = float(integration_spec.initial_step)
        minimum_step = float(integration_spec.minimum_step)
        maximum_step = float(integration_spec.maximum_step)
        attempted_steps = 0
        accepted_steps = 0
        rejected_steps = 0
        accepted_step_epochs: list[float] = []
        accepted_step_magnitudes: list[float] = []

        for checkpoint_index, checkpoint_epoch in enumerate(
            integration_spec.checkpoint_epochs[1:], start=1
        ):
            while current_epoch != checkpoint_epoch:
                if attempted_steps >= integration_spec.maximum_steps:
                    raise TrajectoryStepLimitError(
                        "maximum_steps exhausted before all checkpoints"
                    )
                proposed_step = min(maximum_step, max(minimum_step, proposed_step))
                endpoint_epoch, signed_step, checkpoint_clipped = _trial_endpoint(
                    current_epoch,
                    checkpoint_epoch,
                    direction,
                    proposed_step,
                )
                step_magnitude = abs(signed_step)
                trial_lower = min(current_epoch, endpoint_epoch)
                trial_upper = max(current_epoch, endpoint_epoch)

                def derivative(epoch: float, stage_positions: Any, stage_velocities: Any):
                    if not trial_lower <= epoch <= trial_upper:
                        raise TrajectoryDomainError(
                            "an RKF78 stage epoch left its clipped trial interval"
                        )
                    return stage_evaluator(epoch, stage_positions, stage_velocities)

                trial = _rkf78_step(
                    backend=backend,
                    epoch=current_epoch,
                    endpoint_epoch=endpoint_epoch,
                    step=signed_step,
                    positions=current_positions,
                    velocities=current_velocities,
                    position_carry=current_position_carry,
                    velocity_carry=current_velocity_carry,
                    derivative=derivative,
                )
                attempted_steps += 1
                _validate_trial_arrays(
                    backend,
                    (
                        ("accepted position carry", trial.accepted_position_carry),
                        ("accepted velocity carry", trial.accepted_velocity_carry),
                    ),
                    state_shape,
                )
                normalized_error = _normalized_max_error(
                    backend=backend,
                    current_positions=current_positions,
                    current_velocities=current_velocities,
                    candidate_positions=trial.accepted_positions,
                    candidate_velocities=trial.accepted_velocities,
                    position_defect=trial.position_defect,
                    velocity_defect=trial.velocity_defect,
                    position_atol=position_atol,
                    position_rtol=float(integration_spec.position_rtol),
                    velocity_atol=velocity_atol,
                    velocity_rtol=float(integration_spec.velocity_rtol),
                    state_shape=state_shape,
                )
                scale_factor = _controller_factor(normalized_error, integration_spec)

                if normalized_error <= 1.0:
                    current_positions = trial.accepted_positions
                    current_velocities = trial.accepted_velocities
                    current_position_carry = trial.accepted_position_carry
                    current_velocity_carry = trial.accepted_velocity_carry
                    current_epoch = endpoint_epoch
                    accepted_steps += 1
                    accepted_step_epochs.append(float(endpoint_epoch))
                    accepted_step_magnitudes.append(float(step_magnitude))
                    if not checkpoint_clipped:
                        proposed_step = step_magnitude * scale_factor
                else:
                    rejected_steps += 1
                    if rejected_steps > integration_spec.maximum_rejections:
                        raise TrajectoryStepLimitError(
                            "maximum_rejections exhausted before all checkpoints"
                        )
                    if proposed_step <= minimum_step or (
                        checkpoint_clipped and step_magnitude < minimum_step
                    ):
                        raise TrajectoryStepLimitError(
                            "a rejected step cannot be reduced below minimum_step"
                        )
                    reduced = max(minimum_step, step_magnitude * scale_factor)
                    if reduced >= step_magnitude:
                        raise TrajectoryStepLimitError(
                            "the bounded controller cannot reduce a rejected step"
                        )
                    proposed_step = reduced

            # The loop reaches each checkpoint only by an accepted clipped step.
            copied_positions = _owned_retained_copy(backend, current_positions)
            copied_velocities = _owned_retained_copy(backend, current_velocities)
            checkpoints.append(
                TrajectoryCheckpoint(
                    index=checkpoint_index,
                    epoch=checkpoint_epoch,
                    body_ids=initial_snapshot.body_ids,
                    backend_id=backend.name,
                    device=backend.device,
                    positions=copied_positions,
                    velocities=copied_velocities,
                    accepted_steps=accepted_steps,
                    rejected_steps=rejected_steps,
                )
            )

        if stage_evaluator.force_evaluations != attempted_steps * RKF78_STAGE_COUNT:
            raise TrajectoryContractError("RKF78 force-evaluation count is inconsistent")
        if stage_evaluator.force_ledger is None or stage_evaluator.force_model_ids is None:
            raise TrajectoryContractError("trajectory completed without a force ledger")

        checkpoint_records = tuple(checkpoints)
        accepted_epoch_records = tuple(accepted_step_epochs)
        accepted_magnitude_records = tuple(accepted_step_magnitudes)
        minimum_accepted_step = float(min(accepted_magnitude_records))
        maximum_accepted_step = float(max(accepted_magnitude_records))
        last_accepted_step = float(accepted_magnitude_records[-1])
        accepted_step_ledger_content_sha256 = (
            _accepted_step_ledger_content_sha256(
                accepted_step_epochs=accepted_epoch_records,
                accepted_step_magnitudes=accepted_magnitude_records,
                checkpoints=checkpoint_records,
                direction=integration_spec.direction,
                attempted_steps=attempted_steps,
                accepted_steps=accepted_steps,
                rejected_steps=rejected_steps,
                force_evaluations=stage_evaluator.force_evaluations,
                minimum_accepted_step=minimum_accepted_step,
                maximum_accepted_step=maximum_accepted_step,
                last_accepted_step=last_accepted_step,
                magnitude_source=RKF78_ACCEPTED_STEP_MAGNITUDE_SOURCE,
            )
        )
        result_content_sha256 = None
        if backend.name == "numpy":
            result_content_sha256 = _result_content_sha256(
                backend=backend,
                initial_snapshot=initial_snapshot,
                force_plan=force_plan,
                integration_spec=integration_spec,
                checkpoints=checkpoint_records,
                force_model_ids=stage_evaluator.force_model_ids,
                force_ledger=stage_evaluator.force_ledger,
                accepted_step_epochs=accepted_epoch_records,
                accepted_step_magnitudes=accepted_magnitude_records,
                attempted_steps=attempted_steps,
                accepted_steps=accepted_steps,
                rejected_steps=rejected_steps,
                force_evaluations=stage_evaluator.force_evaluations,
                minimum_accepted_step=minimum_accepted_step,
                maximum_accepted_step=maximum_accepted_step,
                last_accepted_step=last_accepted_step,
                direction=integration_spec.direction,
                accepted_step_ledger_content_sha256=(
                    accepted_step_ledger_content_sha256
                ),
            )

        return TrajectoryResult(
            snapshot_id=snapshot.snapshot_id,
            plan_id=plan.plan_id,
            backend_id=backend.name,
            device=backend.device,
            dtype="float64",
            backend_spec=force_plan.backend,
            initial_snapshot=initial_snapshot,
            force_plan=force_plan,
            integration_spec=integration_spec,
            checkpoint_epochs=integration_spec.checkpoint_epochs,
            checkpoints=checkpoint_records,
            force_model_ids=stage_evaluator.force_model_ids,
            force_ledger=stage_evaluator.force_ledger,
            accepted_step_epochs=accepted_epoch_records,
            accepted_step_magnitudes=accepted_magnitude_records,
            attempted_steps=attempted_steps,
            accepted_steps=accepted_steps,
            rejected_steps=rejected_steps,
            force_evaluations=stage_evaluator.force_evaluations,
            minimum_accepted_step=minimum_accepted_step,
            maximum_accepted_step=maximum_accepted_step,
            last_accepted_step=last_accepted_step,
            direction=integration_spec.direction,
            accepted_step_ledger_content_sha256=(
                accepted_step_ledger_content_sha256
            ),
            result_content_sha256=result_content_sha256,
        )


__all__ = [
    "RKF78_CUPY_RESULT_CONTENT_INTEGRITY_STATUS",
    "RKF78_NUMPY_RESULT_CONTENT_INTEGRITY_STATUS",
    "RKF78_RESULT_CONTENT_CHECKSUM_ALGORITHM",
    "RKF78_RESULT_CONTENT_CHECKSUM_DOMAIN",
    "TRAJECTORY_SCOPE",
    "TrajectoryContractError",
    "TrajectoryCheckpoint",
    "TrajectoryDomainError",
    "TrajectoryError",
    "TrajectoryResult",
    "TrajectoryStepLimitError",
    "integrate_trajectory",
    "validate_trajectory_result_integrity",
]
