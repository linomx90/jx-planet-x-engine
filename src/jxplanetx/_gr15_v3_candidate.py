"""Private qualification boundary for the packaged GR15 V3 candidate."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import sys
from typing import Any

import numpy as np

from .gr15 import (
    GR15ContractError,
    GR15IntegrationError,
    GR15Spec,
    MAXIMUM_BODY_COUNT,
    MINIMUM_BODY_COUNT,
    SCIENTIFIC_CLAIM_STATE,
    STATUS_NAMES,
    STATUS_SUCCESS,
)


METHOD_ID = "JX_GAUSS_RADAU15_V3"


@dataclass(slots=True)
class GR15V3Workspace:
    body_count: int
    checkpoint_epochs: tuple[float, ...]
    checkpoint_positions: np.ndarray
    checkpoint_velocities: np.ndarray
    checkpoint_accepted_steps: np.ndarray
    checkpoint_rejected_steps: np.ndarray
    counters: np.ndarray
    metrics: np.ndarray
    checkpoint_epoch_array: np.ndarray = field(repr=False)


@dataclass(frozen=True, slots=True)
class GR15V3Result:
    checkpoint_positions: np.ndarray
    checkpoint_velocities: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    checkpoint_accepted_steps: tuple[int, ...]
    checkpoint_rejected_steps: tuple[int, ...]
    attempted_steps: int
    accepted_steps: int
    rejected_steps: int
    force_evaluations: int
    predictor_corrector_iterations: int
    nonconverged_retries: int
    polynomial_predictor_trials: int
    constant_predictor_trials: int
    history_commits: int
    terminal_force_reuses: int
    terminal_force_sweeps: int
    maximum_error_ratio: float
    minimum_accepted_step: float
    maximum_accepted_step: float
    final_proposed_step: float
    maximum_corrector_residual: float
    corrector_threshold: float
    replay_digest: str
    method_id: str = METHOD_ID
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    production_authorized: bool = False


def prepare_workspace(body_count: int, spec: GR15Spec) -> GR15V3Workspace:
    if type(body_count) is not int or not (
        MINIMUM_BODY_COUNT <= body_count <= MAXIMUM_BODY_COUNT
    ):
        raise GR15ContractError("GR15 V3 supports exactly 2--32 bodies")
    if type(spec) is not GR15Spec:
        raise GR15ContractError("spec must be an exact GR15Spec")
    epochs = np.ascontiguousarray(spec.checkpoint_epochs, dtype=np.float64)
    epochs.setflags(write=False)
    checkpoint_count = len(spec.checkpoint_epochs)
    shape = (checkpoint_count, body_count, 3)
    return GR15V3Workspace(
        body_count=body_count,
        checkpoint_epochs=spec.checkpoint_epochs,
        checkpoint_positions=np.empty(shape, dtype=np.float64),
        checkpoint_velocities=np.empty(shape, dtype=np.float64),
        checkpoint_accepted_steps=np.empty(checkpoint_count, dtype=np.int64),
        checkpoint_rejected_steps=np.empty(checkpoint_count, dtype=np.int64),
        counters=np.empty(11, dtype=np.int64),
        metrics=np.empty(5, dtype=np.float64),
        checkpoint_epoch_array=epochs,
    )


def _state(value: Any, body_count: int, label: str) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.float64)
        or value.shape != (body_count, 3)
        or not value.flags.c_contiguous
        or not np.all(np.isfinite(value))
    ):
        raise GR15ContractError(
            f"{label} must be a finite C-contiguous float64 array with shape "
            "(body_count, 3)"
        )
    return value


def _digest(workspace: GR15V3Workspace) -> str:
    digest = hashlib.sha256()
    digest.update(b"jx.gr15-v3.qualification-result.v1\0")
    for value in (
        workspace.checkpoint_epoch_array,
        workspace.checkpoint_positions,
        workspace.checkpoint_velocities,
        workspace.counters,
        workspace.metrics,
    ):
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def runtime_identity() -> dict[str, str]:
    from . import _gr15_v3_cpu

    return {
        "method_id": _gr15_v3_cpu.METHOD_ID,
        "source_sha256": _gr15_v3_cpu.CORE_SOURCE_SHA256,
        "compiler_version": _gr15_v3_cpu.COMPILER_VERSION,
        "compiler_flags": _gr15_v3_cpu.COMPILER_FLAGS,
        "scientific_claim_state": _gr15_v3_cpu.SCIENTIFIC_CLAIM_STATE,
    }


def integrate(
    positions: np.ndarray,
    velocities: np.ndarray,
    gravitational_parameters: np.ndarray,
    spec: GR15Spec,
    *,
    workspace: GR15V3Workspace | None = None,
) -> GR15V3Result:
    from . import _gr15_v3_cpu

    if type(spec) is not GR15Spec:
        raise GR15ContractError("spec must be an exact GR15Spec")
    if type(positions) is not np.ndarray or positions.ndim != 2:
        raise GR15ContractError("positions must be a two-dimensional NumPy array")
    body_count = positions.shape[0]
    if not MINIMUM_BODY_COUNT <= body_count <= MAXIMUM_BODY_COUNT:
        raise GR15ContractError("GR15 V3 supports exactly 2--32 bodies")
    positions = _state(positions, body_count, "positions")
    velocities = _state(velocities, body_count, "velocities")
    if (
        type(gravitational_parameters) is not np.ndarray
        or gravitational_parameters.dtype != np.dtype(np.float64)
        or gravitational_parameters.shape != (body_count,)
        or not gravitational_parameters.flags.c_contiguous
        or not np.all(np.isfinite(gravitational_parameters))
        or np.any(gravitational_parameters <= 0.0)
    ):
        raise GR15ContractError(
            "gravitational_parameters must be a positive finite C-contiguous "
            "float64 vector"
        )
    if workspace is None:
        workspace = prepare_workspace(body_count, spec)
    if (
        type(workspace) is not GR15V3Workspace
        or workspace.body_count != body_count
        or workspace.checkpoint_epochs != spec.checkpoint_epochs
    ):
        raise GR15ContractError("GR15 V3 workspace does not match the request")
    status = int(
        _gr15_v3_cpu.integrate(
            body_count,
            len(spec.checkpoint_epochs),
            positions,
            velocities,
            gravitational_parameters,
            workspace.checkpoint_epoch_array,
            spec.initial_step,
            spec.minimum_step,
            spec.maximum_step,
            spec.epsilon,
            spec.safety_factor,
            spec.minimum_scale_factor,
            spec.maximum_scale_factor,
            spec.convergence_factor,
            spec.maximum_iterations,
            spec.maximum_steps,
            spec.maximum_rejections,
            workspace.checkpoint_positions,
            workspace.checkpoint_velocities,
            workspace.checkpoint_accepted_steps,
            workspace.checkpoint_rejected_steps,
            workspace.counters,
            workspace.metrics,
        )
    )
    if status != STATUS_SUCCESS:
        raise GR15IntegrationError(
            f"GR15 V3 failed with status {status} "
            f"({STATUS_NAMES.get(status, 'UNKNOWN')})",
            status=status,
        )
    counters = workspace.counters
    metrics = workspace.metrics
    if (
        counters[0] != counters[1] + counters[2]
        or counters[8] != counters[1]
        or counters[9] + counters[10] != counters[0] - counters[5]
    ):
        raise GR15ContractError("GR15 V3 native accounting is inconsistent")
    threshold = max(
        spec.convergence_factor * sys.float_info.epsilon,
        spec.epsilon * 2.0**-20,
    )
    if not (0.0 <= metrics[4] <= threshold):
        raise GR15ContractError("GR15 V3 corrector residual exceeded its threshold")
    checkpoint_positions = workspace.checkpoint_positions.view()
    checkpoint_velocities = workspace.checkpoint_velocities.view()
    checkpoint_positions.setflags(write=False)
    checkpoint_velocities.setflags(write=False)
    return GR15V3Result(
        checkpoint_positions=checkpoint_positions,
        checkpoint_velocities=checkpoint_velocities,
        positions=checkpoint_positions[-1],
        velocities=checkpoint_velocities[-1],
        checkpoint_accepted_steps=tuple(
            int(value) for value in workspace.checkpoint_accepted_steps
        ),
        checkpoint_rejected_steps=tuple(
            int(value) for value in workspace.checkpoint_rejected_steps
        ),
        attempted_steps=int(counters[0]),
        accepted_steps=int(counters[1]),
        rejected_steps=int(counters[2]),
        force_evaluations=int(counters[3]),
        predictor_corrector_iterations=int(counters[4]),
        nonconverged_retries=int(counters[5]),
        polynomial_predictor_trials=int(counters[6]),
        constant_predictor_trials=int(counters[7]),
        history_commits=int(counters[8]),
        terminal_force_reuses=int(counters[9]),
        terminal_force_sweeps=int(counters[10]),
        maximum_error_ratio=float(metrics[0]),
        minimum_accepted_step=float(metrics[1]),
        maximum_accepted_step=float(metrics[2]),
        final_proposed_step=float(metrics[3]),
        maximum_corrector_residual=float(metrics[4]),
        corrector_threshold=threshold,
        replay_digest=_digest(workspace),
    )


__all__ = [
    "GR15V3Result",
    "GR15V3Workspace",
    "METHOD_ID",
    "integrate",
    "prepare_workspace",
    "runtime_identity",
]
