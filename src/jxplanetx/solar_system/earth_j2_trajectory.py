"""Opt-in Step 5 RKF78 composition for Newtonian gravity plus Earth J2.

The published V5 engine remains byte-for-byte unchanged.  This additive
Solar-System adapter derives a trajectory-scoped Newtonian plan from one
validated DE440 resolved-eleven initial-state receipt, then evaluates the
optional Earth-J2 correction at every Fehlberg RK7(8) stage.

The input GM values are point observations.  Extending them over a requested
trajectory is therefore an explicit caller-selected constant-parameter model,
not a claim that the source artifact supplied an interval-valid force model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

from jxplanetx.engine import trajectory as _trajectory
from jxplanetx.engine.backends import resolve_backend
from jxplanetx.engine.contracts import ForcePlan, NewtonianPointMass, StateSnapshot
from jxplanetx.engine.evaluator import evaluate_force_plan
from jxplanetx.engine.rkf78 import RKF78_STAGE_COUNT, _rkf78_step
from jxplanetx.engine.trajectory import TrajectoryCheckpoint
from jxplanetx.engine.trajectory_contracts import AdaptiveRKF78Spec

from .earth_j2 import (
    EARTH_J2_PAIR_MODEL_ID,
    EarthJ2Error,
    EarthJ2PairForce,
    evaluate_earth_j2_pair_correction,
)
from .earth_pole import FIXED_J2000_POSITIVE_Z
from .engine_preparation import (
    PreparedDe440NewtonianInputs,
    validate_prepared_de440_newtonian_inputs,
)


MODEL_OUTPUT = "MODEL_OUTPUT"
TRAJECTORY_SCOPE = "STEP5_DE440_NEWTONIAN_PLUS_OPTIONAL_EARTH_J2_RKF78"
PARAMETER_EXTENSION_POLICY = (
    "CALLER_OPT_IN_CONSTANT_DE440_GM_VALUES_OVER_REQUESTED_TRAJECTORY_INTERVAL"
)


class EarthJ2TrajectoryError(ValueError):
    """A Step 5 Earth-J2 trajectory request violated its narrow contract."""


class EarthJ2TrajectoryStepLimitError(EarthJ2TrajectoryError):
    """The adaptive RKF78 work or rejection cap was exhausted."""


def _identifier(value: object, label: str) -> str:
    if type(value) is not str or not value or len(value) > 256:
        raise EarthJ2TrajectoryError(f"{label} must be bounded nonempty text")
    if value.strip() != value or any(
        ord(character) < 0x21 or ord(character) > 0x7E for character in value
    ):
        raise EarthJ2TrajectoryError(
            f"{label} must be trimmed printable ASCII without spaces"
        )
    return value


@dataclass(frozen=True, slots=True, eq=False)
class PreparedDe440EarthJ2RKF78:
    """Validated trajectory-scoped composition derived from Step 5 inputs."""

    source_inputs: PreparedDe440NewtonianInputs
    snapshot: StateSnapshot
    force_plan: ForcePlan
    integration_spec: AdaptiveRKF78Spec
    earth_j2_force: EarthJ2PairForce | None
    source_force_plan_id: str
    trajectory_plan_id: str
    parameter_extension_policy: str = PARAMETER_EXTENSION_POLICY
    scope: str = TRAJECTORY_SCOPE
    evidence_class: str = MODEL_OUTPUT
    registry_authorized: bool = False
    qualification_authorized: bool = False


@dataclass(frozen=True, slots=True, eq=False)
class EarthJ2RKF78Result:
    """Owned checkpoint states and complete adaptive-step accounting."""

    configuration: PreparedDe440EarthJ2RKF78
    checkpoints: tuple[TrajectoryCheckpoint, ...]
    accepted_step_epochs: tuple[float, ...]
    accepted_step_magnitudes: tuple[float, ...]
    attempted_steps: int
    accepted_steps: int
    rejected_steps: int
    rkf78_stage_evaluations: int
    baseline_force_evaluations: int
    earth_j2_correction_evaluations: int
    baseline_force_model_ids: tuple[str, ...]
    applied_model_ids: tuple[str, ...]
    accepted_step_ledger_content_sha256: str
    direction: str
    earth_j2_enabled: bool
    pole_mode: str | None
    scope: str = TRAJECTORY_SCOPE
    evidence_class: str = MODEL_OUTPUT
    registry_authorized: bool = False
    qualification_authorized: bool = False

    @property
    def checkpoint_epochs(self) -> tuple[float, ...]:
        return self.configuration.integration_spec.checkpoint_epochs


def _trajectory_plan(
    source: PreparedDe440NewtonianInputs,
    spec: AdaptiveRKF78Spec,
    trajectory_plan_id: str,
) -> ForcePlan:
    model = source.force_plan.models[0]
    if type(model) is not NewtonianPointMass:
        raise EarthJ2TrajectoryError("source plan must contain exact Newtonian gravity")
    lower = min(spec.checkpoint_epochs[0], spec.checkpoint_epochs[-1])
    upper = max(spec.checkpoint_epochs[0], spec.checkpoint_epochs[-1])
    source_metadata = model.parameter_metadata[0]
    extended_metadata = replace(
        source_metadata,
        validity_start=float(lower),
        validity_end=float(upper),
    )
    trajectory_model = replace(model, parameter_metadata=(extended_metadata,))
    return replace(
        source.force_plan,
        plan_id=trajectory_plan_id,
        models=(trajectory_model,),
    )


def _owned_integration_spec(
    source: PreparedDe440NewtonianInputs,
    spec: AdaptiveRKF78Spec,
) -> AdaptiveRKF78Spec:
    """Validate and retain private, read-only NumPy tolerance buffers."""

    backend = resolve_backend(source.force_plan.backend)
    if backend.name != "numpy" or backend.device != "cpu":
        raise EarthJ2TrajectoryError(
            "the Step 5 Earth-J2 trajectory supports exact NumPy CPU only"
        )
    state_shape = (len(source.snapshot.body_ids), 3)
    with backend.activate():
        position_atol = _trajectory._validate_atol(
            backend,
            spec.position_atol,
            "position_atol",
            state_shape,
        )
        velocity_atol = _trajectory._validate_atol(
            backend,
            spec.velocity_atol,
            "velocity_atol",
            state_shape,
        )
        retained = _trajectory._copy_integration_spec(
            backend,
            spec,
            position_atol,
            velocity_atol,
        )
        retained.position_atol.setflags(write=False)
        retained.velocity_atol.setflags(write=False)
    return retained


def _validate_owned_integration_spec(
    source: PreparedDe440NewtonianInputs,
    spec: AdaptiveRKF78Spec,
) -> None:
    backend = resolve_backend(source.force_plan.backend)
    if backend.name != "numpy" or backend.device != "cpu":
        raise EarthJ2TrajectoryError(
            "the Step 5 Earth-J2 trajectory supports exact NumPy CPU only"
        )
    state_shape = (len(source.snapshot.body_ids), 3)
    with backend.activate():
        try:
            position_atol = _trajectory._validate_atol(
                backend,
                spec.position_atol,
                "retained position_atol",
                state_shape,
            )
            velocity_atol = _trajectory._validate_atol(
                backend,
                spec.velocity_atol,
                "retained velocity_atol",
                state_shape,
            )
        except ValueError as exc:
            raise EarthJ2TrajectoryError(
                "retained RKF78 tolerance arrays are malformed"
            ) from exc
        for array in (position_atol, velocity_atol):
            if (
                not array.flags.c_contiguous
                or not array.flags.owndata
                or array.flags.writeable
            ):
                raise EarthJ2TrajectoryError(
                    "retained RKF78 tolerance arrays must be owned, C-contiguous, and read-only"
                )
        if backend.xp.shares_memory(position_atol, velocity_atol):
            raise EarthJ2TrajectoryError(
                "retained RKF78 tolerance arrays must use separate buffers"
            )


def _validate_force_binding(
    source: PreparedDe440NewtonianInputs,
    force: EarthJ2PairForce | None,
) -> None:
    if force is None:
        return
    if type(force) is not EarthJ2PairForce:
        raise EarthJ2TrajectoryError(
            "earth_j2_force must be an exact EarthJ2PairForce or None"
        )
    body_ids = source.snapshot.body_ids
    if (
        force.source_index >= len(body_ids)
        or force.target_index >= len(body_ids)
        or body_ids[force.source_index] != force.source_id
        or body_ids[force.target_index] != force.target_id
    ):
        raise EarthJ2TrajectoryError(
            "Earth-J2 identifiers and indices do not match the prepared body roster"
        )
    if force.pole_mode != FIXED_J2000_POSITIVE_Z:
        expected_start = (
            source.preparation_receipt.epoch_binding.effective_binary64_et
        )
        if (
            type(force.pole_policy.absolute_start_et) is not float
            or force.pole_policy.absolute_start_et.hex() != expected_start.hex()
        ):
            raise EarthJ2TrajectoryError(
                "non-fixed pole policy must bind the exact prepared absolute start ET"
            )
    try:
        evaluate_earth_j2_pair_correction(
            force,
            body_ids=body_ids,
            positions=source.snapshot.positions,
            gravitational_parameters=source.snapshot.gravitational_parameters,
            elapsed_time=0.0,
        )
    except EarthJ2Error as exc:
        raise EarthJ2TrajectoryError(str(exc)) from exc


def prepare_de440_earth_j2_rkf78(
    source_inputs: PreparedDe440NewtonianInputs,
    integration_spec: AdaptiveRKF78Spec,
    *,
    trajectory_plan_id: str,
    earth_j2_force: EarthJ2PairForce | None = None,
) -> PreparedDe440EarthJ2RKF78:
    """Derive one explicit trajectory configuration from point-valid inputs.

    ``earth_j2_force=None`` is the default and selects the unchanged Newtonian
    engine path.  Supplying a force enables exactly one correction-only
    Earth-J2 pair term at every RKF78 stage.
    """

    if type(source_inputs) is not PreparedDe440NewtonianInputs:
        raise EarthJ2TrajectoryError(
            "source_inputs must be exact PreparedDe440NewtonianInputs"
        )
    validate_prepared_de440_newtonian_inputs(source_inputs)
    if type(integration_spec) is not AdaptiveRKF78Spec:
        raise EarthJ2TrajectoryError("integration_spec must be exact AdaptiveRKF78Spec")
    if integration_spec.checkpoint_epochs[0] != source_inputs.snapshot.epoch:
        raise EarthJ2TrajectoryError(
            "the first RKF78 checkpoint must equal the prepared engine epoch"
        )
    checked_plan_id = _identifier(trajectory_plan_id, "trajectory_plan_id")
    if checked_plan_id == source_inputs.force_plan.plan_id:
        raise EarthJ2TrajectoryError(
            "trajectory_plan_id must distinguish the interval model from the point plan"
        )
    _validate_force_binding(source_inputs, earth_j2_force)
    retained_spec = _owned_integration_spec(source_inputs, integration_spec)
    result = PreparedDe440EarthJ2RKF78(
        source_inputs=source_inputs,
        snapshot=source_inputs.snapshot,
        force_plan=_trajectory_plan(
            source_inputs,
            retained_spec,
            checked_plan_id,
        ),
        integration_spec=retained_spec,
        earth_j2_force=earth_j2_force,
        source_force_plan_id=source_inputs.force_plan.plan_id,
        trajectory_plan_id=checked_plan_id,
    )
    validate_prepared_de440_earth_j2_rkf78(result)
    return result


def validate_prepared_de440_earth_j2_rkf78(value: object) -> None:
    """Revalidate the complete derived trajectory configuration."""

    if type(value) is not PreparedDe440EarthJ2RKF78:
        raise EarthJ2TrajectoryError(
            "value must be exact PreparedDe440EarthJ2RKF78"
        )
    source = value.source_inputs
    if type(source) is not PreparedDe440NewtonianInputs:
        raise EarthJ2TrajectoryError("configuration lost its exact source inputs")
    validate_prepared_de440_newtonian_inputs(source)
    if value.snapshot is not source.snapshot:
        raise EarthJ2TrajectoryError("trajectory snapshot must retain the source snapshot")
    if type(value.integration_spec) is not AdaptiveRKF78Spec:
        raise EarthJ2TrajectoryError("configuration lost its exact RKF78 spec")
    _validate_owned_integration_spec(source, value.integration_spec)
    if value.integration_spec.checkpoint_epochs[0] != value.snapshot.epoch:
        raise EarthJ2TrajectoryError("configuration starts at the wrong engine epoch")
    if type(value.force_plan) is not ForcePlan:
        raise EarthJ2TrajectoryError("configuration force_plan must be exact ForcePlan")
    _identifier(value.source_force_plan_id, "source_force_plan_id")
    _identifier(value.trajectory_plan_id, "trajectory_plan_id")
    if value.source_force_plan_id != source.force_plan.plan_id:
        raise EarthJ2TrajectoryError("source force-plan identity changed")
    if value.trajectory_plan_id != value.force_plan.plan_id:
        raise EarthJ2TrajectoryError("trajectory force-plan identity changed")
    expected_plan = _trajectory_plan(
        source,
        value.integration_spec,
        value.trajectory_plan_id,
    )
    observed_model = value.force_plan.models
    expected_model = expected_plan.models
    if (
        value.force_plan.plan_id != expected_plan.plan_id
        or value.force_plan.backend != expected_plan.backend
        or type(observed_model) is not tuple
        or observed_model != expected_model
        or value.force_plan.evidence_class != expected_plan.evidence_class
        or value.force_plan.registry_authorized
        or value.force_plan.qualification_authorized
    ):
        raise EarthJ2TrajectoryError(
            "trajectory plan is not the exact interval-scoped Newtonian derivation"
        )
    _validate_force_binding(source, value.earth_j2_force)
    exact = (
        (value.parameter_extension_policy, PARAMETER_EXTENSION_POLICY),
        (value.scope, TRAJECTORY_SCOPE),
        (value.evidence_class, MODEL_OUTPUT),
        (value.registry_authorized, False),
        (value.qualification_authorized, False),
    )
    for observed, expected in exact:
        if type(observed) is not type(expected) or observed != expected:
            raise EarthJ2TrajectoryError(
                "trajectory authority or parameter-extension contract changed"
            )


def _result_from_engine(
    configuration: PreparedDe440EarthJ2RKF78,
    result: object,
) -> EarthJ2RKF78Result:
    if type(result) is not _trajectory.TrajectoryResult:
        raise EarthJ2TrajectoryError("disabled engine path returned the wrong result type")
    wrapped = EarthJ2RKF78Result(
        configuration=configuration,
        checkpoints=result.checkpoints,
        accepted_step_epochs=result.accepted_step_epochs,
        accepted_step_magnitudes=result.accepted_step_magnitudes,
        attempted_steps=result.attempted_steps,
        accepted_steps=result.accepted_steps,
        rejected_steps=result.rejected_steps,
        rkf78_stage_evaluations=result.force_evaluations,
        baseline_force_evaluations=result.force_evaluations,
        earth_j2_correction_evaluations=0,
        baseline_force_model_ids=result.force_model_ids,
        applied_model_ids=result.force_model_ids,
        accepted_step_ledger_content_sha256=(
            result.accepted_step_ledger_content_sha256
        ),
        direction=result.direction,
        earth_j2_enabled=False,
        pole_mode=None,
    )
    validate_earth_j2_rkf78_result(wrapped)
    return wrapped


class _ComposedStageEvaluator:
    def __init__(
        self,
        configuration: PreparedDe440EarthJ2RKF78,
        snapshot: StateSnapshot,
        plan: ForcePlan,
    ) -> None:
        self.configuration = configuration
        self.snapshot = snapshot
        self.plan = plan
        self.initial_epoch = float(snapshot.epoch)
        self.baseline_force_evaluations = 0
        self.earth_j2_correction_evaluations = 0
        self.baseline_force_model_ids: tuple[str, ...] | None = None

    def __call__(self, epoch: float, positions: Any, velocities: Any):
        stage_snapshot = replace(
            self.snapshot,
            epoch=epoch,
            positions=positions,
            velocities=velocities,
        )
        baseline = evaluate_force_plan(stage_snapshot, self.plan)
        self.baseline_force_evaluations += 1
        if self.baseline_force_model_ids is None:
            self.baseline_force_model_ids = baseline.applied_model_ids
        elif baseline.applied_model_ids != self.baseline_force_model_ids:
            raise EarthJ2TrajectoryError(
                "baseline force-plan identity changed between RKF78 stages"
            )
        force = self.configuration.earth_j2_force
        if force is None:
            raise EarthJ2TrajectoryError("composed evaluator lost its Earth-J2 force")
        elapsed_time = float(epoch - self.initial_epoch)
        correction = evaluate_earth_j2_pair_correction(
            force,
            body_ids=stage_snapshot.body_ids,
            positions=positions,
            gravitational_parameters=stage_snapshot.gravitational_parameters,
            elapsed_time=elapsed_time,
        )
        self.earth_j2_correction_evaluations += 1
        total = baseline.total_acceleration + correction
        if not self.plan.backend.backend_id == "numpy":
            raise EarthJ2TrajectoryError("Earth-J2 trajectory supports NumPy only")
        return velocities, total


def _integrate_enabled(
    configuration: PreparedDe440EarthJ2RKF78,
) -> EarthJ2RKF78Result:
    spec = configuration.integration_spec
    snapshot = configuration.snapshot
    plan = configuration.force_plan
    backend = resolve_backend(plan.backend)
    if backend.name != "numpy" or backend.device != "cpu":
        raise EarthJ2TrajectoryError(
            "the Step 5 Earth-J2 trajectory supports exact NumPy CPU only"
        )
    direction = 1.0 if spec.direction == "FORWARD" else -1.0
    with backend.activate():
        validated_state = _trajectory._validate_state(backend, snapshot)
        state_shape = (len(snapshot.body_ids), 3)
        position_atol = _trajectory._validate_atol(
            backend, spec.position_atol, "position_atol", state_shape
        )
        velocity_atol = _trajectory._validate_atol(
            backend, spec.velocity_atol, "velocity_atol", state_shape
        )
        initial_snapshot = _trajectory._copy_initial_snapshot(
            backend, snapshot, validated_state
        )
        force_plan = _trajectory._copy_force_plan(backend, plan)
        integration_spec = _trajectory._copy_integration_spec(
            backend,
            spec,
            position_atol,
            velocity_atol,
        )
        current_positions = backend.xp.copy(initial_snapshot.positions)
        current_velocities = backend.xp.copy(initial_snapshot.velocities)
        current_position_carry = backend.xp.zeros_like(
            current_positions, dtype=backend.xp.float64
        )
        current_velocity_carry = backend.xp.zeros_like(
            current_velocities, dtype=backend.xp.float64
        )
        checkpoints = [
            TrajectoryCheckpoint(
                index=0,
                epoch=integration_spec.checkpoint_epochs[0],
                body_ids=initial_snapshot.body_ids,
                backend_id=backend.name,
                device=backend.device,
                positions=backend.xp.copy(current_positions),
                velocities=backend.xp.copy(current_velocities),
                accepted_steps=0,
                rejected_steps=0,
            )
        ]
        evaluator = _ComposedStageEvaluator(
            configuration,
            initial_snapshot,
            force_plan,
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
                    raise EarthJ2TrajectoryStepLimitError(
                        "maximum_steps exhausted before all checkpoints"
                    )
                proposed_step = min(maximum_step, max(minimum_step, proposed_step))
                endpoint_epoch, signed_step, checkpoint_clipped = (
                    _trajectory._trial_endpoint(
                        current_epoch,
                        checkpoint_epoch,
                        direction,
                        proposed_step,
                    )
                )
                step_magnitude = abs(signed_step)
                trial_lower = min(current_epoch, endpoint_epoch)
                trial_upper = max(current_epoch, endpoint_epoch)

                def derivative(epoch: float, positions: Any, velocities: Any):
                    if not trial_lower <= epoch <= trial_upper:
                        raise EarthJ2TrajectoryError(
                            "an RKF78 stage epoch left its clipped trial interval"
                        )
                    return evaluator(epoch, positions, velocities)

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
                _trajectory._validate_trial_arrays(
                    backend,
                    (
                        ("accepted position carry", trial.accepted_position_carry),
                        ("accepted velocity carry", trial.accepted_velocity_carry),
                    ),
                    state_shape,
                )
                normalized_error = _trajectory._normalized_max_error(
                    backend=backend,
                    current_positions=current_positions,
                    current_velocities=current_velocities,
                    candidate_positions=trial.accepted_positions,
                    candidate_velocities=trial.accepted_velocities,
                    position_defect=trial.position_defect,
                    velocity_defect=trial.velocity_defect,
                    position_atol=integration_spec.position_atol,
                    position_rtol=float(integration_spec.position_rtol),
                    velocity_atol=integration_spec.velocity_atol,
                    velocity_rtol=float(integration_spec.velocity_rtol),
                    state_shape=state_shape,
                )
                scale_factor = _trajectory._controller_factor(
                    normalized_error, integration_spec
                )
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
                        raise EarthJ2TrajectoryStepLimitError(
                            "maximum_rejections exhausted before all checkpoints"
                        )
                    if proposed_step <= minimum_step or (
                        checkpoint_clipped and step_magnitude < minimum_step
                    ):
                        raise EarthJ2TrajectoryStepLimitError(
                            "a rejected step cannot be reduced below minimum_step"
                        )
                    reduced = max(minimum_step, step_magnitude * scale_factor)
                    if reduced >= step_magnitude:
                        raise EarthJ2TrajectoryStepLimitError(
                            "the bounded controller cannot reduce a rejected step"
                        )
                    proposed_step = reduced

            checkpoints.append(
                TrajectoryCheckpoint(
                    index=checkpoint_index,
                    epoch=checkpoint_epoch,
                    body_ids=initial_snapshot.body_ids,
                    backend_id=backend.name,
                    device=backend.device,
                    positions=backend.xp.copy(current_positions),
                    velocities=backend.xp.copy(current_velocities),
                    accepted_steps=accepted_steps,
                    rejected_steps=rejected_steps,
                )
            )

        expected_stages = attempted_steps * RKF78_STAGE_COUNT
        if (
            evaluator.baseline_force_evaluations != expected_stages
            or evaluator.earth_j2_correction_evaluations != expected_stages
            or evaluator.baseline_force_model_ids is None
        ):
            raise EarthJ2TrajectoryError(
                "composed force-evaluation accounting is inconsistent"
            )
        checkpoint_records = tuple(checkpoints)
        epoch_records = tuple(accepted_step_epochs)
        magnitude_records = tuple(accepted_step_magnitudes)
        if not magnitude_records:
            raise EarthJ2TrajectoryError("trajectory completed without an accepted step")
        ledger_sha256 = _trajectory._accepted_step_ledger_content_sha256(
            accepted_step_epochs=epoch_records,
            accepted_step_magnitudes=magnitude_records,
            checkpoints=checkpoint_records,
            direction=integration_spec.direction,
            attempted_steps=attempted_steps,
            accepted_steps=accepted_steps,
            rejected_steps=rejected_steps,
            force_evaluations=expected_stages,
            minimum_accepted_step=float(min(magnitude_records)),
            maximum_accepted_step=float(max(magnitude_records)),
            last_accepted_step=float(magnitude_records[-1]),
            magnitude_source=_trajectory.RKF78_ACCEPTED_STEP_MAGNITUDE_SOURCE,
        )
        result = EarthJ2RKF78Result(
            configuration=configuration,
            checkpoints=checkpoint_records,
            accepted_step_epochs=epoch_records,
            accepted_step_magnitudes=magnitude_records,
            attempted_steps=attempted_steps,
            accepted_steps=accepted_steps,
            rejected_steps=rejected_steps,
            rkf78_stage_evaluations=expected_stages,
            baseline_force_evaluations=evaluator.baseline_force_evaluations,
            earth_j2_correction_evaluations=(
                evaluator.earth_j2_correction_evaluations
            ),
            baseline_force_model_ids=evaluator.baseline_force_model_ids,
            applied_model_ids=(
                evaluator.baseline_force_model_ids + (EARTH_J2_PAIR_MODEL_ID,)
            ),
            accepted_step_ledger_content_sha256=ledger_sha256,
            direction=integration_spec.direction,
            earth_j2_enabled=True,
            pole_mode=configuration.earth_j2_force.pole_mode,  # type: ignore[union-attr]
        )
    validate_earth_j2_rkf78_result(result)
    return result


def integrate_de440_earth_j2_rkf78(
    configuration: PreparedDe440EarthJ2RKF78,
) -> EarthJ2RKF78Result:
    """Run the configured Newtonian plus optional Earth-J2 RKF78 trajectory."""

    validate_prepared_de440_earth_j2_rkf78(configuration)
    if configuration.earth_j2_force is None:
        baseline = _trajectory.integrate_trajectory(
            configuration.snapshot,
            configuration.force_plan,
            configuration.integration_spec,
        )
        return _result_from_engine(configuration, baseline)
    return _integrate_enabled(configuration)


def validate_earth_j2_rkf78_result(value: object) -> None:
    """Revalidate result bindings, checkpoint states, and accounting."""

    if type(value) is not EarthJ2RKF78Result:
        raise EarthJ2TrajectoryError("value must be exact EarthJ2RKF78Result")
    validate_prepared_de440_earth_j2_rkf78(value.configuration)
    spec = value.configuration.integration_spec
    body_ids = value.configuration.snapshot.body_ids
    backend = resolve_backend(value.configuration.force_plan.backend)
    if (
        type(value.checkpoints) is not tuple
        or len(value.checkpoints) != len(spec.checkpoint_epochs)
    ):
        raise EarthJ2TrajectoryError("result checkpoints do not match the RKF78 spec")
    previous_accepted = 0
    previous_rejected = 0
    for index, (checkpoint, epoch) in enumerate(
        zip(value.checkpoints, spec.checkpoint_epochs)
    ):
        if (
            type(checkpoint) is not TrajectoryCheckpoint
            or checkpoint.index != index
            or checkpoint.epoch != epoch
            or checkpoint.body_ids != body_ids
            or checkpoint.backend_id != "numpy"
            or checkpoint.device != "cpu"
        ):
            raise EarthJ2TrajectoryError("checkpoint metadata binding changed")
        try:
            _trajectory._validate_trial_arrays(
                backend,
                (
                    ("checkpoint positions", checkpoint.positions),
                    ("checkpoint velocities", checkpoint.velocities),
                ),
                (len(body_ids), 3),
            )
        except ValueError as exc:
            raise EarthJ2TrajectoryError("checkpoint state array is malformed") from exc
        for array in (checkpoint.positions, checkpoint.velocities):
            if not array.flags.c_contiguous or not array.flags.owndata:
                raise EarthJ2TrajectoryError(
                    "checkpoint state arrays must be owned C-contiguous storage"
                )
        if index == 0:
            if checkpoint.accepted_steps != 0 or checkpoint.rejected_steps != 0:
                raise EarthJ2TrajectoryError("initial checkpoint counters must be zero")
            if backend.scalar_bool(
                backend.xp.any(
                    checkpoint.positions
                    != value.configuration.snapshot.positions
                ),
                "initial checkpoint position binding",
            ) or backend.scalar_bool(
                backend.xp.any(
                    checkpoint.velocities
                    != value.configuration.snapshot.velocities
                ),
                "initial checkpoint velocity binding",
            ):
                raise EarthJ2TrajectoryError(
                    "initial checkpoint differs from the prepared state"
                )
        else:
            if (
                checkpoint.accepted_steps <= previous_accepted
                or checkpoint.rejected_steps < previous_rejected
            ):
                raise EarthJ2TrajectoryError("checkpoint counters are not monotone")
        previous_accepted = checkpoint.accepted_steps
        previous_rejected = checkpoint.rejected_steps
    for label in (
        "attempted_steps",
        "accepted_steps",
        "rejected_steps",
        "rkf78_stage_evaluations",
        "baseline_force_evaluations",
        "earth_j2_correction_evaluations",
    ):
        observed = getattr(value, label)
        if type(observed) is not int or observed < 0:
            raise EarthJ2TrajectoryError(f"{label} must be a nonnegative exact integer")
    if value.attempted_steps != value.accepted_steps + value.rejected_steps:
        raise EarthJ2TrajectoryError("attempted-step accounting is inconsistent")
    expected_stages = value.attempted_steps * RKF78_STAGE_COUNT
    if (
        value.rkf78_stage_evaluations != expected_stages
        or value.baseline_force_evaluations != expected_stages
        or value.accepted_steps < len(spec.checkpoint_epochs) - 1
        or value.checkpoints[-1].accepted_steps != value.accepted_steps
        or value.checkpoints[-1].rejected_steps != value.rejected_steps
    ):
        raise EarthJ2TrajectoryError("trajectory work accounting is inconsistent")
    enabled = value.configuration.earth_j2_force is not None
    expected_j2 = expected_stages if enabled else 0
    expected_ids = tuple(
        model.model_id for model in value.configuration.force_plan.models
    )
    if enabled:
        expected_applied = expected_ids + (EARTH_J2_PAIR_MODEL_ID,)
        expected_mode = value.configuration.earth_j2_force.pole_mode  # type: ignore[union-attr]
    else:
        expected_applied = expected_ids
        expected_mode = None
    if (
        type(value.earth_j2_enabled) is not bool
        or value.earth_j2_enabled is not enabled
        or value.earth_j2_correction_evaluations != expected_j2
        or value.baseline_force_model_ids != expected_ids
        or value.applied_model_ids != expected_applied
        or value.pole_mode != expected_mode
        or value.direction != spec.direction
        or type(value.accepted_step_epochs) is not tuple
        or type(value.accepted_step_magnitudes) is not tuple
        or len(value.accepted_step_epochs) != value.accepted_steps
        or len(value.accepted_step_magnitudes) != value.accepted_steps
    ):
        raise EarthJ2TrajectoryError("result force or step-ledger binding changed")
    previous_epoch = float(spec.checkpoint_epochs[0])
    direction = 1.0 if spec.direction == "FORWARD" else -1.0
    for epoch, magnitude in zip(
        value.accepted_step_epochs, value.accepted_step_magnitudes
    ):
        if (
            type(epoch) is not float
            or type(magnitude) is not float
            or not math.isfinite(epoch)
            or not math.isfinite(magnitude)
            or magnitude <= 0.0
            or direction * (epoch - previous_epoch) <= 0.0
            or abs(epoch - previous_epoch) != magnitude
        ):
            raise EarthJ2TrajectoryError("accepted-step ledger is malformed")
        previous_epoch = epoch
    for checkpoint in value.checkpoints[1:]:
        endpoint_index = checkpoint.accepted_steps - 1
        if (
            endpoint_index < 0
            or endpoint_index >= len(value.accepted_step_epochs)
            or value.accepted_step_epochs[endpoint_index] != float(checkpoint.epoch)
        ):
            raise EarthJ2TrajectoryError(
                "accepted-step ledger does not terminate at every checkpoint"
            )
    if (
        type(value.accepted_step_ledger_content_sha256) is not str
        or len(value.accepted_step_ledger_content_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value.accepted_step_ledger_content_sha256
        )
    ):
        raise EarthJ2TrajectoryError("accepted-step ledger checksum is malformed")
    expected_ledger_sha256 = _trajectory._accepted_step_ledger_content_sha256(
        accepted_step_epochs=value.accepted_step_epochs,
        accepted_step_magnitudes=value.accepted_step_magnitudes,
        checkpoints=value.checkpoints,
        direction=value.direction,
        attempted_steps=value.attempted_steps,
        accepted_steps=value.accepted_steps,
        rejected_steps=value.rejected_steps,
        force_evaluations=value.rkf78_stage_evaluations,
        minimum_accepted_step=float(min(value.accepted_step_magnitudes)),
        maximum_accepted_step=float(max(value.accepted_step_magnitudes)),
        last_accepted_step=float(value.accepted_step_magnitudes[-1]),
        magnitude_source=_trajectory.RKF78_ACCEPTED_STEP_MAGNITUDE_SOURCE,
    )
    if value.accepted_step_ledger_content_sha256 != expected_ledger_sha256:
        raise EarthJ2TrajectoryError("accepted-step ledger checksum changed")
    exact = (
        (value.scope, TRAJECTORY_SCOPE),
        (value.evidence_class, MODEL_OUTPUT),
        (value.registry_authorized, False),
        (value.qualification_authorized, False),
    )
    for observed, expected in exact:
        if type(observed) is not type(expected) or observed != expected:
            raise EarthJ2TrajectoryError("result authority controls changed")


__all__ = [
    "EarthJ2RKF78Result",
    "EarthJ2TrajectoryError",
    "EarthJ2TrajectoryStepLimitError",
    "MODEL_OUTPUT",
    "PARAMETER_EXTENSION_POLICY",
    "PreparedDe440EarthJ2RKF78",
    "TRAJECTORY_SCOPE",
    "integrate_de440_earth_j2_rkf78",
    "prepare_de440_earth_j2_rkf78",
    "validate_earth_j2_rkf78_result",
    "validate_prepared_de440_earth_j2_rkf78",
]
