"""Packaged CPU GR15 integration with mutual point-mass EIH 1PN gravity.

This boundary advances the Newtonian plus first post-Newtonian EIH equations
with the qualified GR15 V3 corrector/controller algorithm in a distinct native
component.  It intentionally does not modify the qualified Newtonian
``integrate_gr15`` source or runtime identity.

Inputs use kilometres, seconds, kilometres per second, and km^3/s^2.  The
model is screening-only and is not a production ephemeris, exact general
relativity, collision handler, or observational reduction system.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import sys
import time
from typing import Any

import numpy as np

from .gr15 import (
    GR15ContractError,
    GR15IntegrationError,
    GR15Spec,
    GR15UnavailableError,
    GR15Workspace,
    MAXIMUM_BODY_COUNT,
    MINIMUM_BODY_COUNT,
    SCIENTIFIC_CLAIM_STATE,
    STAGE_COUNT,
    STATUS_CHECKPOINT_SCHEDULE,
    STATUS_FORCE_SINGULARITY,
    STATUS_INPUT_DOMAIN,
    STATUS_MINIMUM_STEP,
    STATUS_NONFINITE,
    STATUS_REJECTION_LIMIT,
    STATUS_STEP_LIMIT,
    STATUS_SUCCESS,
    _readonly_view,
    _require_state,
    _validate_workspace,
    prepare_gr15_workspace,
)
from .solar_system.eih_1pn import COORDINATE_SCOPE, EIH1PNParameters


METHOD_ID = "JX_GR15_EIH1PN_V1"
MODEL_ID = "jx.gr15.eih-1pn-mutual-point-mass.v1"
EXPECTED_CORE_SOURCE_SHA256 = (
    "bef3a607c4c3674d342745570c68b462a6178e789089c84419d6f23dd810fad1"
)
EXPECTED_FORCE_CORE_SOURCE_SHA256 = (
    "de4b173ff26aab1a5d012aca79b2687c81ce8ef97db290e48b00ca3d21f40746"
)
STATUS_WEAK_FIELD_DOMAIN = 8
STATUS_NAMES = {
    STATUS_SUCCESS: "SUCCESS",
    STATUS_INPUT_DOMAIN: "INPUT_DOMAIN",
    STATUS_NONFINITE: "NONFINITE",
    STATUS_FORCE_SINGULARITY: "FORCE_SINGULARITY",
    STATUS_STEP_LIMIT: "STEP_LIMIT",
    STATUS_REJECTION_LIMIT: "REJECTION_LIMIT",
    STATUS_MINIMUM_STEP: "MINIMUM_STEP",
    STATUS_CHECKPOINT_SCHEDULE: "CHECKPOINT_SCHEDULE",
    STATUS_WEAK_FIELD_DOMAIN: "WEAK_FIELD_DOMAIN",
}

_NATIVE_EXTENSION: Any | None = None


class GR15EIH1PNIntegrationError(GR15IntegrationError):
    """The native GR15-EIH1PN solver failed closed with a named status."""

    def __init__(self, message: str, *, status: int) -> None:
        super().__init__(message, status=status)
        self.status_name = STATUS_NAMES.get(status, "UNKNOWN")


@dataclass(slots=True)
class GR15EIH1PNWorkspace:
    """Reusable trajectory and force-diagnostic buffers for one request shape."""

    trajectory: GR15Workspace
    force_diagnostics: np.ndarray


@dataclass(frozen=True, slots=True)
class GR15EIH1PNResult:
    """Successful screening-only EIH 1PN trajectory and audited accounting."""

    method_id: str
    model_id: str
    coordinate_scope: str
    scientific_claim_state: str
    checkpoint_epochs: tuple[float, ...]
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
    speed_of_light_km_s: float
    maximum_allowed_compactness: float
    maximum_allowed_speed_fraction_squared: float
    observed_maximum_compactness: float
    observed_maximum_speed_fraction_squared: float
    elapsed_seconds: float
    replay_digest: str
    status: int
    production_authorized: bool = False
    exact_general_relativity_claimed: bool = False
    ephemeris_equivalence_claimed: bool = False
    general_superiority_claimed: bool = False

    def __post_init__(self) -> None:
        if self.method_id != METHOD_ID or self.model_id != MODEL_ID:
            raise GR15ContractError("GR15-EIH1PN result identity changed")
        if self.coordinate_scope != COORDINATE_SCOPE:
            raise GR15ContractError("GR15-EIH1PN coordinate scope changed")
        if self.scientific_claim_state != SCIENTIFIC_CLAIM_STATE:
            raise GR15ContractError("GR15-EIH1PN claim state changed")
        if (
            self.production_authorized
            or self.exact_general_relativity_claimed
            or self.ephemeris_equivalence_claimed
            or self.general_superiority_claimed
        ):
            raise GR15ContractError("GR15-EIH1PN screening output elevated a claim")
        if self.status != STATUS_SUCCESS:
            raise GR15ContractError("only successful native results may be constructed")
        if self.attempted_steps != self.accepted_steps + self.rejected_steps:
            raise GR15ContractError("GR15-EIH1PN step accounting is inconsistent")
        if self.accepted_steps <= 0 or self.rejected_steps < 0:
            raise GR15ContractError("GR15-EIH1PN accepted/rejected accounting is invalid")
        if self.history_commits != self.accepted_steps:
            raise GR15ContractError("GR15-EIH1PN history accounting is inconsistent")
        if (
            self.terminal_force_reuses + self.terminal_force_sweeps
            != self.attempted_steps - self.nonconverged_retries
        ):
            raise GR15ContractError("GR15-EIH1PN terminal-force accounting is inconsistent")
        if (
            self.polynomial_predictor_trials + self.constant_predictor_trials
            != self.attempted_steps
        ):
            raise GR15ContractError("GR15-EIH1PN predictor accounting is inconsistent")
        if not 0.0 <= self.maximum_corrector_residual <= self.corrector_threshold:
            raise GR15ContractError("GR15-EIH1PN corrector residual exceeded its threshold")
        if not (
            0.0 <= self.observed_maximum_compactness
            <= self.maximum_allowed_compactness
        ):
            raise GR15ContractError("GR15-EIH1PN compactness diagnostic is invalid")
        if not (
            0.0 <= self.observed_maximum_speed_fraction_squared
            <= self.maximum_allowed_speed_fraction_squared
        ):
            raise GR15ContractError("GR15-EIH1PN speed diagnostic is invalid")


def _native_extension() -> Any:
    global _NATIVE_EXTENSION
    if _NATIVE_EXTENSION is not None:
        return _NATIVE_EXTENSION
    try:
        from . import _gr15_eih_1pn_cpu
    except ImportError as exc:
        raise GR15UnavailableError(
            "the packaged jxplanetx._gr15_eih_1pn_cpu extension is required; "
            "install a platform wheel or build JX from source with a C11 compiler"
        ) from exc
    if (
        _gr15_eih_1pn_cpu.MAX_BODIES != MAXIMUM_BODY_COUNT
        or _gr15_eih_1pn_cpu.STAGE_COUNT != STAGE_COUNT
        or _gr15_eih_1pn_cpu.METHOD_ID != METHOD_ID
        or _gr15_eih_1pn_cpu.MODEL_ID != MODEL_ID
        or _gr15_eih_1pn_cpu.SCIENTIFIC_CLAIM_STATE != SCIENTIFIC_CLAIM_STATE
        or _gr15_eih_1pn_cpu.CORE_SOURCE_SHA256
        != EXPECTED_CORE_SOURCE_SHA256
        or _gr15_eih_1pn_cpu.FORCE_CORE_SOURCE_SHA256
        != EXPECTED_FORCE_CORE_SOURCE_SHA256
    ):
        raise GR15UnavailableError(
            "the packaged GR15-EIH1PN native identity is invalid"
        )
    _NATIVE_EXTENSION = _gr15_eih_1pn_cpu
    return _NATIVE_EXTENSION


def gr15_eih_1pn_runtime_identity() -> dict[str, str]:
    """Return the exact native integrator and force-kernel build identity."""

    extension = _native_extension()
    return {
        "method_id": extension.METHOD_ID,
        "model_id": extension.MODEL_ID,
        "core_source_sha256": extension.CORE_SOURCE_SHA256,
        "force_core_source_sha256": extension.FORCE_CORE_SOURCE_SHA256,
        "compiler_version": extension.COMPILER_VERSION,
        "compiler_flags": extension.COMPILER_FLAGS,
        "scientific_claim_state": extension.SCIENTIFIC_CLAIM_STATE,
    }


def prepare_gr15_eih_1pn_workspace(
    body_count: int,
    spec: GR15Spec,
) -> GR15EIH1PNWorkspace:
    """Allocate reusable buffers for one GR15-EIH1PN integration shape."""

    return GR15EIH1PNWorkspace(
        trajectory=prepare_gr15_workspace(body_count, spec),
        force_diagnostics=np.empty(2, dtype=np.float64),
    )


def _validate_eih_workspace(
    workspace: GR15EIH1PNWorkspace,
    body_count: int,
    spec: GR15Spec,
) -> None:
    if type(workspace) is not GR15EIH1PNWorkspace:
        raise GR15ContractError("workspace must be an exact GR15EIH1PNWorkspace")
    _validate_workspace(workspace.trajectory, body_count, spec)
    diagnostics = workspace.force_diagnostics
    if (
        type(diagnostics) is not np.ndarray
        or diagnostics.dtype != np.dtype(np.float64)
        or diagnostics.shape != (2,)
        or not diagnostics.flags.c_contiguous
        or not diagnostics.flags.writeable
    ):
        raise GR15ContractError("workspace force_diagnostics buffer is invalid")


def _replay_digest(
    parameters: EIH1PNParameters,
    workspace: GR15EIH1PNWorkspace,
) -> str:
    trajectory = workspace.trajectory
    digest = hashlib.sha256()
    digest.update(b"jx.gr15-eih-1pn.supported-result.v1\0")
    parameter_values = np.asarray(
        (
            parameters.speed_of_light_km_s,
            parameters.maximum_compactness,
            parameters.maximum_speed_fraction_squared,
        ),
        dtype=np.float64,
    )
    for value in (
        parameter_values,
        trajectory._checkpoint_epoch_array,
        trajectory.checkpoint_positions,
        trajectory.checkpoint_velocities,
        trajectory.counters,
        trajectory.metrics,
        workspace.force_diagnostics,
    ):
        digest.update(np.asarray(value).tobytes(order="C"))
    return digest.hexdigest()


def integrate_gr15_eih_1pn(
    positions_km: np.ndarray,
    velocities_km_s: np.ndarray,
    gravitational_parameters_km3_s2: np.ndarray,
    spec: GR15Spec,
    parameters: EIH1PNParameters,
    *,
    workspace: GR15EIH1PNWorkspace | None = None,
) -> GR15EIH1PNResult:
    """Advance a 2--32 body mutual Newtonian-plus-EIH-1PN system.

    Epochs and steps in ``spec`` are seconds. Inputs are never modified.
    Unsupported weak-field states and all numerical failures fail closed.
    """

    if type(spec) is not GR15Spec:
        raise GR15ContractError("spec must be an exact GR15Spec")
    if type(parameters) is not EIH1PNParameters:
        raise GR15ContractError("parameters must be an exact EIH1PNParameters")
    if type(positions_km) is not np.ndarray or positions_km.ndim != 2:
        raise GR15ContractError("positions_km must be a two-dimensional NumPy array")
    body_count = positions_km.shape[0]
    if not MINIMUM_BODY_COUNT <= body_count <= MAXIMUM_BODY_COUNT:
        raise GR15ContractError("GR15-EIH1PN supports exactly 2--32 bodies")
    positions = _require_state(positions_km, body_count, "positions_km")
    velocities = _require_state(velocities_km_s, body_count, "velocities_km_s")
    gm = gravitational_parameters_km3_s2
    if (
        type(gm) is not np.ndarray
        or gm.dtype != np.dtype(np.float64)
        or gm.shape != (body_count,)
        or not gm.flags.c_contiguous
        or not np.all(np.isfinite(gm))
        or np.any(gm <= 0.0)
    ):
        raise GR15ContractError(
            "gravitational_parameters_km3_s2 must be a positive finite "
            "C-contiguous NumPy float64 vector"
        )
    if workspace is None:
        workspace = prepare_gr15_eih_1pn_workspace(body_count, spec)
    _validate_eih_workspace(workspace, body_count, spec)
    trajectory = workspace.trajectory

    started = time.perf_counter_ns()
    try:
        status = int(
            _native_extension().integrate(
                body_count,
                len(spec.checkpoint_epochs),
                positions,
                velocities,
                gm,
                trajectory._checkpoint_epoch_array,
                parameters.speed_of_light_km_s,
                parameters.maximum_compactness,
                parameters.maximum_speed_fraction_squared,
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
                trajectory.checkpoint_positions,
                trajectory.checkpoint_velocities,
                trajectory.checkpoint_accepted_steps,
                trajectory.checkpoint_rejected_steps,
                trajectory.counters,
                trajectory.metrics,
                workspace.force_diagnostics,
            )
        )
    except (BufferError, OverflowError, TypeError, ValueError) as exc:
        raise GR15ContractError(
            f"the native GR15-EIH1PN boundary rejected its buffers: {exc}"
        ) from exc
    elapsed_seconds = (time.perf_counter_ns() - started) * 1.0e-9
    if status != STATUS_SUCCESS:
        status_name = STATUS_NAMES.get(status, "UNKNOWN")
        raise GR15EIH1PNIntegrationError(
            f"GR15-EIH1PN integration failed with status {status} ({status_name})",
            status=status,
        )

    counters = trajectory.counters
    metrics = trajectory.metrics
    diagnostics = workspace.force_diagnostics
    if not np.all(np.isfinite(diagnostics)) or np.any(diagnostics < 0.0):
        raise GR15ContractError("native GR15-EIH1PN diagnostics are invalid")
    return GR15EIH1PNResult(
        method_id=METHOD_ID,
        model_id=MODEL_ID,
        coordinate_scope=COORDINATE_SCOPE,
        scientific_claim_state=SCIENTIFIC_CLAIM_STATE,
        checkpoint_epochs=spec.checkpoint_epochs,
        checkpoint_positions=_readonly_view(trajectory.checkpoint_positions),
        checkpoint_velocities=_readonly_view(trajectory.checkpoint_velocities),
        positions=_readonly_view(trajectory.checkpoint_positions[-1]),
        velocities=_readonly_view(trajectory.checkpoint_velocities[-1]),
        checkpoint_accepted_steps=tuple(
            int(value) for value in trajectory.checkpoint_accepted_steps
        ),
        checkpoint_rejected_steps=tuple(
            int(value) for value in trajectory.checkpoint_rejected_steps
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
        corrector_threshold=max(
            spec.convergence_factor * sys.float_info.epsilon,
            spec.epsilon * 2.0**-20,
        ),
        speed_of_light_km_s=parameters.speed_of_light_km_s,
        maximum_allowed_compactness=parameters.maximum_compactness,
        maximum_allowed_speed_fraction_squared=(
            parameters.maximum_speed_fraction_squared
        ),
        observed_maximum_compactness=float(diagnostics[0]),
        observed_maximum_speed_fraction_squared=float(diagnostics[1]),
        elapsed_seconds=elapsed_seconds,
        replay_digest=_replay_digest(parameters, workspace),
        status=status,
    )


__all__ = [
    "EXPECTED_CORE_SOURCE_SHA256",
    "EXPECTED_FORCE_CORE_SOURCE_SHA256",
    "GR15EIH1PNIntegrationError",
    "GR15EIH1PNResult",
    "GR15EIH1PNWorkspace",
    "METHOD_ID",
    "MODEL_ID",
    "STATUS_WEAK_FIELD_DOMAIN",
    "gr15_eih_1pn_runtime_identity",
    "integrate_gr15_eih_1pn",
    "prepare_gr15_eih_1pn_workspace",
]
