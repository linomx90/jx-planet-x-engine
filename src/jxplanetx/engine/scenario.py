"""Explicit scenario and matched-comparison contracts for the JX engine.

The scenario layer is intentionally small.  It assembles one existing
``StateSnapshot``, ``ForcePlan``, and ``AdaptiveRKF78Spec`` without weakening
any of their contracts.  Its first comparison mode is an additive-force
ablation: the candidate retains the exact control state, integrator request,
backend, and ordered force prefix, then appends one or more declared terms.

Checkpoint differences are model-to-model distances.  They are not errors,
accuracy estimates, improvements, physical detections, or qualification
evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .backends import resolve_backend
from .contracts import ContractError, ForcePlan, StateSnapshot
from .trajectory import TrajectoryResult, integrate_trajectory
from .trajectory_contracts import AdaptiveRKF78Spec


DYNAMICS_SCENARIO_SCOPE = "ADAPTIVE_RKF78_FORCE_PLAN_SCENARIO_V1"
MATCHED_SCENARIO_SCOPE = "ADDITIVE_FORCE_ABLATION_SAME_INITIAL_STATE_AND_SOLVER_V1"
SCENARIO_COMPARISON_SEMANTICS = (
    "MODEL_TO_MODEL_CHECKPOINT_DIFFERENCE_NOT_ACCURACY_OR_IMPROVEMENT"
)
SCENARIO_EXECUTION_ORDER = "CONTROL_THEN_CANDIDATE"
SCENARIO_EVIDENCE_CLASS = "MODEL_OUTPUT"

_SCENARIO_ROLES = frozenset({"STANDALONE", "CONTROL", "CANDIDATE"})


class ScenarioContractError(ContractError):
    """A scenario or matched comparison is incomplete or ambiguous."""


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ScenarioContractError(f"{label} must be a nonempty, trimmed string")
    return value


def _ids(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not tuple or not value:
        raise ScenarioContractError(f"{label} must be a nonempty immutable tuple")
    checked = tuple(
        _text(item, f"{label}[{index}]") for index, item in enumerate(value)
    )
    if len(set(checked)) != len(checked):
        raise ScenarioContractError(f"{label} must not contain duplicates")
    return checked


def _fixed_claim_controls(value: object) -> None:
    exact = {
        "evidence_class": SCENARIO_EVIDENCE_CLASS,
        "registry_authorized": False,
        "qualification_authorized": False,
        "accuracy_claimed": False,
        "improvement_claimed": False,
        "physics_claimed": False,
    }
    for field, expected in exact.items():
        actual = getattr(value, field)
        if type(actual) is not type(expected) or actual != expected:
            raise ScenarioContractError(
                f"{field} must equal the fixed scenario value {expected!r}"
            )


@dataclass(frozen=True, eq=False)
class DynamicsScenario:
    """One explicit adaptive-RKF78 dynamics request.

    ``initial_snapshot`` and ``integration_spec`` compare by identity in a
    matched experiment.  This prevents an apparently matched pair from hiding
    different state buffers, tolerances, or checkpoint schedules.
    """

    scenario_id: str
    role: str
    description: str
    comparison_id: str | None
    initial_snapshot: StateSnapshot
    force_plan: ForcePlan
    integration_spec: AdaptiveRKF78Spec
    scope: str = DYNAMICS_SCENARIO_SCOPE
    evidence_class: str = SCENARIO_EVIDENCE_CLASS
    registry_authorized: bool = False
    qualification_authorized: bool = False
    accuracy_claimed: bool = False
    improvement_claimed: bool = False
    physics_claimed: bool = False

    def __post_init__(self) -> None:
        _text(self.scenario_id, "scenario_id")
        if type(self.role) is not str or self.role not in _SCENARIO_ROLES:
            raise ScenarioContractError(
                "role must be exactly STANDALONE, CONTROL, or CANDIDATE"
            )
        _text(self.description, "description")
        if self.role == "STANDALONE":
            if self.comparison_id is not None:
                raise ScenarioContractError(
                    "a STANDALONE scenario cannot declare comparison_id"
                )
        else:
            _text(self.comparison_id, "comparison_id")
        if type(self.initial_snapshot) is not StateSnapshot:
            raise ScenarioContractError("initial_snapshot must be StateSnapshot")
        if type(self.force_plan) is not ForcePlan:
            raise ScenarioContractError("force_plan must be ForcePlan")
        if type(self.integration_spec) is not AdaptiveRKF78Spec:
            raise ScenarioContractError(
                "integration_spec must be AdaptiveRKF78Spec in scenario V1"
            )
        if self.integration_spec.checkpoint_epochs[0] != self.initial_snapshot.epoch:
            raise ScenarioContractError(
                "the first checkpoint epoch must equal the initial snapshot epoch"
            )
        if self.scope != DYNAMICS_SCENARIO_SCOPE:
            raise ScenarioContractError(
                f"scope must equal {DYNAMICS_SCENARIO_SCOPE!r}"
            )
        _fixed_claim_controls(self)

    @property
    def force_model_ids(self) -> tuple[str, ...]:
        return tuple(model.model_id for model in self.force_plan.models)

    @property
    def qualified(self) -> bool:
        return False


@dataclass(frozen=True, eq=False)
class MatchedScenarioComparison:
    """A same-state, same-solver, additive-force control/candidate pair."""

    comparison_id: str
    scientific_question: str
    control: DynamicsScenario
    candidate: DynamicsScenario
    compared_body_ids: tuple[str, ...]
    added_force_model_ids: tuple[str, ...]
    scope: str = MATCHED_SCENARIO_SCOPE
    comparison_semantics: str = SCENARIO_COMPARISON_SEMANTICS
    evidence_class: str = SCENARIO_EVIDENCE_CLASS
    registry_authorized: bool = False
    qualification_authorized: bool = False
    accuracy_claimed: bool = False
    improvement_claimed: bool = False
    physics_claimed: bool = False

    def __post_init__(self) -> None:
        _text(self.comparison_id, "comparison_id")
        _text(self.scientific_question, "scientific_question")
        if type(self.control) is not DynamicsScenario:
            raise ScenarioContractError("control must be DynamicsScenario")
        if type(self.candidate) is not DynamicsScenario:
            raise ScenarioContractError("candidate must be DynamicsScenario")
        if self.control.role != "CONTROL" or self.candidate.role != "CANDIDATE":
            raise ScenarioContractError(
                "matched scenarios require explicit CONTROL and CANDIDATE roles"
            )
        if (
            self.control.comparison_id != self.comparison_id
            or self.candidate.comparison_id != self.comparison_id
        ):
            raise ScenarioContractError(
                "comparison_id must match both scenario bindings"
            )
        if self.control.scenario_id == self.candidate.scenario_id:
            raise ScenarioContractError("control and candidate scenario_id must differ")
        if self.control.initial_snapshot is not self.candidate.initial_snapshot:
            raise ScenarioContractError(
                "control and candidate must share the exact initial_snapshot object"
            )
        if self.control.integration_spec is not self.candidate.integration_spec:
            raise ScenarioContractError(
                "control and candidate must share the exact integration_spec object"
            )
        if self.control.force_plan.backend != self.candidate.force_plan.backend:
            raise ScenarioContractError(
                "control and candidate must request the same backend contract"
            )
        if self.control.force_plan.plan_id == self.candidate.force_plan.plan_id:
            raise ScenarioContractError("control and candidate plan_id must differ")

        compared_ids = _ids(self.compared_body_ids, "compared_body_ids")
        body_roster = self.control.initial_snapshot.body_ids
        unknown = tuple(body_id for body_id in compared_ids if body_id not in body_roster)
        if unknown:
            raise ScenarioContractError(
                f"compared_body_ids contains unknown bodies {unknown!r}"
            )

        declared_added = _ids(
            self.added_force_model_ids,
            "added_force_model_ids",
        )
        control_models = self.control.force_plan.models
        candidate_models = self.candidate.force_plan.models
        if len(candidate_models) <= len(control_models):
            raise ScenarioContractError(
                "candidate must append at least one force model to the control"
            )
        for index, control_model in enumerate(control_models):
            if candidate_models[index] is not control_model:
                raise ScenarioContractError(
                    "candidate must preserve the exact ordered control force prefix"
                )
        actual_added = tuple(
            model.model_id for model in candidate_models[len(control_models) :]
        )
        if declared_added != actual_added:
            raise ScenarioContractError(
                "added_force_model_ids must exactly name the appended candidate models"
            )
        if self.scope != MATCHED_SCENARIO_SCOPE:
            raise ScenarioContractError(
                f"scope must equal {MATCHED_SCENARIO_SCOPE!r}"
            )
        if self.comparison_semantics != SCENARIO_COMPARISON_SEMANTICS:
            raise ScenarioContractError(
                "comparison_semantics must retain the model-difference claim ceiling"
            )
        _fixed_claim_controls(self)

    @property
    def qualified(self) -> bool:
        return False


def _validate_trajectory_binding(
    scenario: DynamicsScenario,
    trajectory: TrajectoryResult,
) -> None:
    if type(trajectory) is not TrajectoryResult:
        raise ScenarioContractError("trajectory must be TrajectoryResult")
    if trajectory.snapshot_id != scenario.initial_snapshot.snapshot_id:
        raise ScenarioContractError("trajectory snapshot_id does not match scenario")
    if trajectory.plan_id != scenario.force_plan.plan_id:
        raise ScenarioContractError("trajectory plan_id does not match scenario")
    if trajectory.checkpoint_epochs != scenario.integration_spec.checkpoint_epochs:
        raise ScenarioContractError(
            "trajectory checkpoints do not match the scenario integration_spec"
        )
    if trajectory.force_model_ids != scenario.force_model_ids:
        raise ScenarioContractError("trajectory force roster does not match scenario")
    if trajectory.backend_spec != scenario.force_plan.backend:
        raise ScenarioContractError("trajectory backend does not match scenario")
    if trajectory.evidence_class != SCENARIO_EVIDENCE_CLASS or trajectory.qualified:
        raise ScenarioContractError("trajectory must remain unqualified MODEL_OUTPUT")


@dataclass(frozen=True, eq=False)
class DynamicsScenarioResult:
    """The trajectory produced by one explicit scenario."""

    scenario: DynamicsScenario
    trajectory: TrajectoryResult
    scope: str = DYNAMICS_SCENARIO_SCOPE
    evidence_class: str = SCENARIO_EVIDENCE_CLASS
    registry_authorized: bool = False
    qualification_authorized: bool = False
    accuracy_claimed: bool = False
    improvement_claimed: bool = False
    physics_claimed: bool = False

    def __post_init__(self) -> None:
        if type(self.scenario) is not DynamicsScenario:
            raise ScenarioContractError("scenario must be DynamicsScenario")
        _validate_trajectory_binding(self.scenario, self.trajectory)
        if self.scope != DYNAMICS_SCENARIO_SCOPE:
            raise ScenarioContractError(
                f"scope must equal {DYNAMICS_SCENARIO_SCOPE!r}"
            )
        _fixed_claim_controls(self)

    @property
    def final_positions(self) -> Any:
        return self.trajectory.final_positions

    @property
    def final_velocities(self) -> Any:
        return self.trajectory.final_velocities

    @property
    def qualified(self) -> bool:
        return False


@dataclass(frozen=True)
class ScenarioCheckpointDelta:
    """One fixed-order model-to-model state-distance record."""

    checkpoint_index: int
    epoch: float
    body_id: str
    position_difference_norm: float
    velocity_difference_norm: float
    position_unit: str
    velocity_unit: str
    semantics: str = SCENARIO_COMPARISON_SEMANTICS
    evidence_class: str = SCENARIO_EVIDENCE_CLASS
    accuracy_claimed: bool = False
    improvement_claimed: bool = False
    physics_claimed: bool = False

    def __post_init__(self) -> None:
        if type(self.checkpoint_index) is not int or self.checkpoint_index < 0:
            raise ScenarioContractError(
                "checkpoint_index must be a nonnegative integer"
            )
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, (int, float)):
            raise ScenarioContractError("epoch must be a finite real number")
        if not math.isfinite(float(self.epoch)):
            raise ScenarioContractError("epoch must be a finite real number")
        _text(self.body_id, "body_id")
        for field in ("position_difference_norm", "velocity_difference_norm"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ScenarioContractError(f"{field} must be a finite real number")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ScenarioContractError(f"{field} must be finite and nonnegative")
        _text(self.position_unit, "position_unit")
        _text(self.velocity_unit, "velocity_unit")
        if self.semantics != SCENARIO_COMPARISON_SEMANTICS:
            raise ScenarioContractError(
                "semantics must retain the model-difference claim ceiling"
            )
        exact = {
            "evidence_class": SCENARIO_EVIDENCE_CLASS,
            "accuracy_claimed": False,
            "improvement_claimed": False,
            "physics_claimed": False,
        }
        for field, expected in exact.items():
            value = getattr(self, field)
            if type(value) is not type(expected) or value != expected:
                raise ScenarioContractError(
                    f"{field} must equal the fixed delta value {expected!r}"
                )


def _component_float(value: object, label: str) -> float:
    try:
        item = value.item()
    except (AttributeError, TypeError, ValueError) as exc:
        raise ScenarioContractError(f"{label} did not produce a backend scalar") from exc
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise ScenarioContractError(f"{label} did not produce a real scalar")
    result = float(item)
    if not math.isfinite(result):
        raise ScenarioContractError(f"{label} must be finite")
    return result


def _fixed_vec3_norm(value: Any, label: str) -> float:
    components = tuple(
        _component_float(value[index], f"{label} component {index}")
        for index in range(3)
    )
    result = math.hypot(*components)
    if not math.isfinite(result):
        raise ScenarioContractError(f"{label} must be finite")
    return result


def _checkpoint_deltas(
    comparison: MatchedScenarioComparison,
    control_result: DynamicsScenarioResult,
    candidate_result: DynamicsScenarioResult,
) -> tuple[ScenarioCheckpointDelta, ...]:
    control_trajectory = control_result.trajectory
    candidate_trajectory = candidate_result.trajectory
    if control_trajectory.checkpoint_epochs != candidate_trajectory.checkpoint_epochs:
        raise ScenarioContractError("matched trajectories have different checkpoints")
    if control_trajectory.initial_snapshot.body_ids != candidate_trajectory.initial_snapshot.body_ids:
        raise ScenarioContractError("matched trajectories have different body rosters")
    if control_trajectory.backend_spec != candidate_trajectory.backend_spec:
        raise ScenarioContractError("matched trajectories have different backends")

    backend = resolve_backend(control_trajectory.backend_spec)
    snapshot = comparison.control.initial_snapshot
    velocity_unit = f"{snapshot.length_unit}/{snapshot.time_unit}"
    records: list[ScenarioCheckpointDelta] = []
    with backend.activate():
        for checkpoint_index, (control_checkpoint, candidate_checkpoint) in enumerate(
            zip(control_trajectory.checkpoints, candidate_trajectory.checkpoints)
        ):
            if (
                control_checkpoint.index != checkpoint_index
                or candidate_checkpoint.index != checkpoint_index
                or control_checkpoint.epoch != candidate_checkpoint.epoch
            ):
                raise ScenarioContractError(
                    "matched trajectory checkpoint bindings are inconsistent"
                )
            control_positions = backend.require_native_array(
                control_checkpoint.positions,
                "control checkpoint positions",
            )
            candidate_positions = backend.require_native_array(
                candidate_checkpoint.positions,
                "candidate checkpoint positions",
            )
            control_velocities = backend.require_native_array(
                control_checkpoint.velocities,
                "control checkpoint velocities",
            )
            candidate_velocities = backend.require_native_array(
                candidate_checkpoint.velocities,
                "candidate checkpoint velocities",
            )
            for body_id in comparison.compared_body_ids:
                body_index = snapshot.index_of(body_id)
                position_delta = candidate_positions[body_index] - control_positions[body_index]
                velocity_delta = candidate_velocities[body_index] - control_velocities[body_index]
                records.append(
                    ScenarioCheckpointDelta(
                        checkpoint_index=checkpoint_index,
                        epoch=control_checkpoint.epoch,
                        body_id=body_id,
                        position_difference_norm=_fixed_vec3_norm(
                            position_delta,
                            "position difference norm",
                        ),
                        velocity_difference_norm=_fixed_vec3_norm(
                            velocity_delta,
                            "velocity difference norm",
                        ),
                        position_unit=snapshot.length_unit,
                        velocity_unit=velocity_unit,
                    )
                )
    return tuple(records)


@dataclass(frozen=True, eq=False)
class MatchedScenarioResult:
    """Two retained runs plus fixed-order model-to-model checkpoint deltas."""

    comparison: MatchedScenarioComparison
    control_result: DynamicsScenarioResult
    candidate_result: DynamicsScenarioResult
    checkpoint_deltas: tuple[ScenarioCheckpointDelta, ...]
    execution_order: str = SCENARIO_EXECUTION_ORDER
    scope: str = MATCHED_SCENARIO_SCOPE
    comparison_semantics: str = SCENARIO_COMPARISON_SEMANTICS
    evidence_class: str = SCENARIO_EVIDENCE_CLASS
    registry_authorized: bool = False
    qualification_authorized: bool = False
    accuracy_claimed: bool = False
    improvement_claimed: bool = False
    physics_claimed: bool = False

    def __post_init__(self) -> None:
        if type(self.comparison) is not MatchedScenarioComparison:
            raise ScenarioContractError(
                "comparison must be MatchedScenarioComparison"
            )
        if type(self.control_result) is not DynamicsScenarioResult:
            raise ScenarioContractError(
                "control_result must be DynamicsScenarioResult"
            )
        if type(self.candidate_result) is not DynamicsScenarioResult:
            raise ScenarioContractError(
                "candidate_result must be DynamicsScenarioResult"
            )
        if self.control_result.scenario is not self.comparison.control:
            raise ScenarioContractError("control_result does not bind the control")
        if self.candidate_result.scenario is not self.comparison.candidate:
            raise ScenarioContractError("candidate_result does not bind the candidate")
        if type(self.checkpoint_deltas) is not tuple:
            raise ScenarioContractError("checkpoint_deltas must be an immutable tuple")
        expected = _checkpoint_deltas(
            self.comparison,
            self.control_result,
            self.candidate_result,
        )
        if len(self.checkpoint_deltas) != len(expected):
            raise ScenarioContractError("checkpoint_deltas has the wrong length")
        fields = (
            "checkpoint_index",
            "epoch",
            "body_id",
            "position_difference_norm",
            "velocity_difference_norm",
            "position_unit",
            "velocity_unit",
            "semantics",
            "evidence_class",
            "accuracy_claimed",
            "improvement_claimed",
            "physics_claimed",
        )
        for index, (actual, required) in enumerate(
            zip(self.checkpoint_deltas, expected)
        ):
            if type(actual) is not ScenarioCheckpointDelta:
                raise ScenarioContractError(
                    f"checkpoint_deltas[{index}] must be ScenarioCheckpointDelta"
                )
            if any(
                type(getattr(actual, field)) is not type(getattr(required, field))
                or getattr(actual, field) != getattr(required, field)
                for field in fields
            ):
                raise ScenarioContractError(
                    f"checkpoint_deltas[{index}] does not match the retained trajectories"
                )
        if self.execution_order != SCENARIO_EXECUTION_ORDER:
            raise ScenarioContractError(
                f"execution_order must equal {SCENARIO_EXECUTION_ORDER!r}"
            )
        if self.scope != MATCHED_SCENARIO_SCOPE:
            raise ScenarioContractError(
                f"scope must equal {MATCHED_SCENARIO_SCOPE!r}"
            )
        if self.comparison_semantics != SCENARIO_COMPARISON_SEMANTICS:
            raise ScenarioContractError(
                "comparison_semantics must retain the model-difference claim ceiling"
            )
        _fixed_claim_controls(self)

    @property
    def qualified(self) -> bool:
        return False


def run_dynamics_scenario(scenario: DynamicsScenario) -> DynamicsScenarioResult:
    """Execute one validated scenario through the public RKF78 path."""

    if type(scenario) is not DynamicsScenario:
        raise ScenarioContractError("scenario must be an exact DynamicsScenario")
    trajectory = integrate_trajectory(
        scenario.initial_snapshot,
        scenario.force_plan,
        scenario.integration_spec,
    )
    return DynamicsScenarioResult(scenario=scenario, trajectory=trajectory)


def run_matched_scenario_comparison(
    comparison: MatchedScenarioComparison,
) -> MatchedScenarioResult:
    """Run the control then candidate and retain model-to-model deltas only."""

    if type(comparison) is not MatchedScenarioComparison:
        raise ScenarioContractError(
            "comparison must be an exact MatchedScenarioComparison"
        )
    control_result = run_dynamics_scenario(comparison.control)

    # Start the candidate from the control run's retained copies of the state,
    # controller request, backend, and force prefix.  Besides documenting a
    # matched request, this makes the executed baseline identical even though
    # public input arrays are caller-owned and mutable.  Candidate-only force
    # buffers remain caller-owned for the duration of this complete call.
    control_model_count = len(comparison.control.force_plan.models)
    candidate_only_models = comparison.candidate.force_plan.models[
        control_model_count:
    ]
    retained_control_trajectory = control_result.trajectory
    candidate_runtime_plan = ForcePlan(
        plan_id=comparison.candidate.force_plan.plan_id,
        backend=retained_control_trajectory.force_plan.backend,
        models=(
            retained_control_trajectory.force_plan.models
            + candidate_only_models
        ),
    )
    candidate_trajectory = integrate_trajectory(
        retained_control_trajectory.initial_snapshot,
        candidate_runtime_plan,
        retained_control_trajectory.integration_spec,
    )
    candidate_result = DynamicsScenarioResult(
        scenario=comparison.candidate,
        trajectory=candidate_trajectory,
    )
    deltas = _checkpoint_deltas(comparison, control_result, candidate_result)
    return MatchedScenarioResult(
        comparison=comparison,
        control_result=control_result,
        candidate_result=candidate_result,
        checkpoint_deltas=deltas,
    )


__all__ = [
    "DYNAMICS_SCENARIO_SCOPE",
    "DynamicsScenario",
    "DynamicsScenarioResult",
    "MATCHED_SCENARIO_SCOPE",
    "MatchedScenarioComparison",
    "MatchedScenarioResult",
    "SCENARIO_COMPARISON_SEMANTICS",
    "SCENARIO_EVIDENCE_CLASS",
    "SCENARIO_EXECUTION_ORDER",
    "ScenarioCheckpointDelta",
    "ScenarioContractError",
    "run_dynamics_scenario",
    "run_matched_scenario_comparison",
]
