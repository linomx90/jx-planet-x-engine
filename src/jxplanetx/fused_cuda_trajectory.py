#!/usr/bin/env python3
"""Fail-closed package-owned fused CUDA RKF78 trajectory implementation.

The bridge accepts the exact public ``StateSnapshot``, ``ForcePlan``, and
``AdaptiveRKF78Spec`` types and emits an exact public ``TrajectoryResult``.
Its numerical scope is intentionally narrow: one fully mutual Newtonian model,
all bodies massive, CuPy float64 state, two through 32 bodies, and
``tile_size=1``.  Those restrictions make the fused kernel's ordered source
sum agree with the declared public force-plan accumulation order.

This module is package-owned experimental code. It is not registered as a public
backend and does not expand the engine's scientific qualification.
"""

from __future__ import annotations

from typing import Any

from jxplanetx.engine import (
    AdaptiveRKF78Spec,
    ForceCollisionError,
    ForcePlan,
    ForceSingularityError,
    NewtonianPointMass,
    StateSnapshot,
    TrajectoryCheckpoint,
    TrajectoryContractError,
    TrajectoryDomainError,
    TrajectoryResult,
    TrajectoryStepLimitError,
)
from jxplanetx.engine.backends import resolve_backend
from jxplanetx.engine.evaluator import (
    DETERMINISM_LIMITATION,
    ForceLedgerEntry,
    StateMetadataBinding,
    _validate_dependencies,
    _validate_model_sequence,
    _validate_parameter_contracts,
    _validate_restricted_frame,
    _validate_state,
)
from jxplanetx.engine.rkf78 import RKF78_STAGE_COUNT
from jxplanetx.engine.trajectory_contracts import (
    RKF78_ACCEPTED_STEP_MAGNITUDE_SOURCE,
)
from jxplanetx.engine.trajectory import (
    _accepted_step_ledger_content_sha256,
    _copy_force_plan,
    _copy_initial_snapshot,
    _copy_integration_spec,
    _owned_retained_copy,
    _validate_atol,
    _validate_force_compatibility,
    _validate_metadata_range,
    validate_trajectory_result_integrity,
)

from jxplanetx import fused_cuda_adaptive_rkf78 as adaptive


class FusedTrajectoryContractError(TrajectoryContractError):
    """A public request is outside the fused bridge's exact contract."""


MAX_PUBLIC_LEDGER_ENTRIES = 1_000_000


def _force_ledger(
    snapshot: StateSnapshot,
    plan: ForcePlan,
    model: NewtonianPointMass,
) -> tuple[ForceLedgerEntry, ...]:
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
    return (
        ForceLedgerEntry(
            order=0,
            model_id=model.model_id,
            role="NEWTONIAN_BASE",
            source_ids=model.source_ids,
            target_ids=model.target_ids,
            state_metadata=state_metadata,
            backend_spec=plan.backend,
            tile_size=plan.backend.tile_size,
            assumptions=(
                "direct unsoftened point masses",
                "finite-radius contact and singular coincidence are errors",
                DETERMINISM_LIMITATION,
            ),
        ),
    )


def _raise_status(code: int) -> None:
    message = adaptive.STATUS_MESSAGES.get(code, f"UNKNOWN_STATUS_{code}")
    if code == 1:
        raise ForceSingularityError(f"fused CUDA trajectory failed: {message}")
    if code == 10:
        raise ForceCollisionError(f"fused CUDA trajectory failed: {message}")
    if code in {4, 6, 7, 8, 9, 11}:
        raise TrajectoryStepLimitError(f"fused CUDA trajectory failed: {message}")
    raise TrajectoryDomainError(f"fused CUDA trajectory failed: {message}")


def integrate_fused_newtonian_trajectory(
    snapshot: StateSnapshot,
    plan: ForcePlan,
    spec: AdaptiveRKF78Spec,
) -> TrajectoryResult:
    """Run one public-contract trajectory through the restricted fused path."""

    if type(snapshot) is not StateSnapshot:
        raise FusedTrajectoryContractError("snapshot must be an exact StateSnapshot")
    if type(plan) is not ForcePlan:
        raise FusedTrajectoryContractError("plan must be an exact ForcePlan")
    if type(spec) is not AdaptiveRKF78Spec:
        raise FusedTrajectoryContractError("spec must be an exact AdaptiveRKF78Spec")
    if spec.checkpoint_epochs[0] != snapshot.epoch:
        raise FusedTrajectoryContractError(
            "checkpoint_epochs[0] must exactly equal StateSnapshot.epoch"
        )
    if plan.backend.backend_id != "cupy":
        raise FusedTrajectoryContractError("fused bridge requires the CuPy backend")
    if plan.backend.tile_size != 1:
        raise FusedTrajectoryContractError(
            "fused bridge requires tile_size=1 to preserve source-sum order"
        )
    if len(plan.models) != 1 or type(plan.models[0]) is not NewtonianPointMass:
        raise FusedTrajectoryContractError(
            "fused bridge supports exactly one NewtonianPointMass model"
        )
    model = plan.models[0]
    if model.source_ids != snapshot.body_ids or model.target_ids != snapshot.body_ids:
        raise FusedTrajectoryContractError(
            "fused bridge requires source_ids and target_ids in exact body order"
        )
    body_count = len(snapshot.body_ids)
    if not 2 <= body_count <= adaptive.MAX_BODIES:
        raise FusedTrajectoryContractError("fused bridge requires 2--32 bodies")
    if spec.maximum_steps > MAX_PUBLIC_LEDGER_ENTRIES:
        raise FusedTrajectoryContractError(
            "maximum_steps exceeds the fused public ledger allocation limit"
        )

    _validate_model_sequence(plan.models)
    _validate_dependencies(plan.models)
    _validate_parameter_contracts(snapshot, plan)
    _validate_restricted_frame(snapshot, plan.models)
    _validate_force_compatibility(plan)
    _validate_metadata_range(
        plan,
        min(spec.checkpoint_epochs[0], spec.checkpoint_epochs[-1]),
        max(spec.checkpoint_epochs[0], spec.checkpoint_epochs[-1]),
    )

    backend = resolve_backend(plan.backend)
    with backend.activate():
        validated_state = _validate_state(backend, snapshot)
        positions, velocities, gm, _masses, radii, massive = validated_state
        if backend.scalar_bool(backend.xp.any(~massive), "fused massive-body check"):
            raise FusedTrajectoryContractError(
                "fused bridge requires every body to be marked massive"
            )
        state_shape = (body_count, 3)
        caller_position_atol = _validate_atol(
            backend, spec.position_atol, "position_atol", state_shape
        )
        caller_velocity_atol = _validate_atol(
            backend, spec.velocity_atol, "velocity_atol", state_shape
        )
        initial_snapshot = _copy_initial_snapshot(
            backend, snapshot, validated_state
        )
        force_plan = _copy_force_plan(backend, plan)
        integration_spec = _copy_integration_spec(
            backend,
            spec,
            caller_position_atol,
            caller_velocity_atol,
        )
        retained_ledger = _force_ledger(initial_snapshot, force_plan, force_plan.models[0])

        core_spec = adaptive.AdaptiveFusedSpec(
            initial_epoch=integration_spec.checkpoint_epochs[0],
            final_epoch=integration_spec.checkpoint_epochs[-1],
            intermediate_epochs=integration_spec.checkpoint_epochs[1:-1],
            initial_step=float(integration_spec.initial_step),
            minimum_step=float(integration_spec.minimum_step),
            maximum_step=float(integration_spec.maximum_step),
            position_atol=float(caller_position_atol.reshape(-1)[0].item()),
            position_rtol=float(integration_spec.position_rtol),
            velocity_atol=float(caller_velocity_atol.reshape(-1)[0].item()),
            velocity_rtol=float(integration_spec.velocity_rtol),
            maximum_steps=integration_spec.maximum_steps,
            maximum_rejections=integration_spec.maximum_rejections,
            safety_factor=float(integration_spec.safety_factor),
            minimum_scale_factor=float(integration_spec.minimum_scale_factor),
            maximum_scale_factor=float(integration_spec.maximum_scale_factor),
            ledger_capacity=integration_spec.maximum_steps,
        )
        core_result = adaptive.fused_adaptive_integrate(
            positions[backend.xp.newaxis, :, :],
            velocities[backend.xp.newaxis, :, :],
            gm,
            core_spec,
            position_atol=caller_position_atol,
            velocity_atol=caller_velocity_atol,
            radii=radii,
        )
        status = int(core_result.status[0].item())
        if status != 0:
            _raise_status(status)
        if len(core_result.checkpoint_positions) != len(
            integration_spec.checkpoint_epochs
        ):
            raise FusedTrajectoryContractError(
                "fused trajectory did not retain every requested checkpoint"
            )

        checkpoints: list[TrajectoryCheckpoint] = []
        for index, epoch in enumerate(integration_spec.checkpoint_epochs):
            checkpoints.append(
                TrajectoryCheckpoint(
                    index=index,
                    epoch=epoch,
                    body_ids=initial_snapshot.body_ids,
                    backend_id=backend.name,
                    device=backend.device,
                    positions=_owned_retained_copy(
                        backend, core_result.checkpoint_positions[index][0]
                    ),
                    velocities=_owned_retained_copy(
                        backend, core_result.checkpoint_velocities[index][0]
                    ),
                    accepted_steps=int(
                        core_result.checkpoint_accepted_steps[index][0]
                    ),
                    rejected_steps=int(
                        core_result.checkpoint_rejected_steps[index][0]
                    ),
                )
            )

        checkpoint_records = tuple(checkpoints)
        accepted_epoch_records = core_result.accepted_step_epochs[0]
        accepted_magnitude_records = core_result.accepted_step_magnitudes[0]
        attempted_steps = int(core_result.attempted_steps[0].item())
        accepted_steps = int(core_result.accepted_steps[0].item())
        rejected_steps = int(core_result.rejected_steps[0].item())
        force_evaluations = attempted_steps * RKF78_STAGE_COUNT
        minimum_accepted_step = float(min(accepted_magnitude_records))
        maximum_accepted_step = float(max(accepted_magnitude_records))
        last_accepted_step = float(accepted_magnitude_records[-1])
        ledger_sha256 = _accepted_step_ledger_content_sha256(
            accepted_step_epochs=accepted_epoch_records,
            accepted_step_magnitudes=accepted_magnitude_records,
            checkpoints=checkpoint_records,
            direction=integration_spec.direction,
            attempted_steps=attempted_steps,
            accepted_steps=accepted_steps,
            rejected_steps=rejected_steps,
            force_evaluations=force_evaluations,
            minimum_accepted_step=minimum_accepted_step,
            maximum_accepted_step=maximum_accepted_step,
            last_accepted_step=last_accepted_step,
            magnitude_source=RKF78_ACCEPTED_STEP_MAGNITUDE_SOURCE,
        )
        result = TrajectoryResult(
            snapshot_id=initial_snapshot.snapshot_id,
            plan_id=force_plan.plan_id,
            backend_id=backend.name,
            device=backend.device,
            dtype="float64",
            backend_spec=force_plan.backend,
            initial_snapshot=initial_snapshot,
            force_plan=force_plan,
            integration_spec=integration_spec,
            checkpoint_epochs=integration_spec.checkpoint_epochs,
            checkpoints=checkpoint_records,
            force_model_ids=(model.model_id,),
            force_ledger=retained_ledger,
            accepted_step_epochs=accepted_epoch_records,
            accepted_step_magnitudes=accepted_magnitude_records,
            attempted_steps=attempted_steps,
            accepted_steps=accepted_steps,
            rejected_steps=rejected_steps,
            force_evaluations=force_evaluations,
            minimum_accepted_step=minimum_accepted_step,
            maximum_accepted_step=maximum_accepted_step,
            last_accepted_step=last_accepted_step,
            direction=integration_spec.direction,
            accepted_step_ledger_content_sha256=ledger_sha256,
            result_content_sha256=None,
        )
        validate_trajectory_result_integrity(result)
        return result


__all__ = [
    "FusedTrajectoryContractError",
    "integrate_fused_newtonian_trajectory",
]
