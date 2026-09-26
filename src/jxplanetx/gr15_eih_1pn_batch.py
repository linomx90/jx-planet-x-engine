"""Supported NumPy batch and conservative CPU/CUDA dispatch for JX GR15.

The CPU path runs independent systems in a thread pool.  The packaged native
GR15-EIH1PN boundary releases the Python GIL for the complete integration, so
threads execute concurrently without subprocess startup or state transport.

The automatic path is deliberately conservative.  It selects CUDA only when
the active CPU, CUDA device, body count, duration, worker policy, and ensemble
size match a recorded crossover profile.  Every uncalibrated case stays on
CPU unless the caller explicitly requests CUDA.  Backend selection changes
performance only; both paths retain the same screening-only scientific claim
ceiling and are returned through one NumPy-facing result contract.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import math
import os
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np

from .gr15 import GR15ContractError, GR15Spec, MAXIMUM_BODY_COUNT, MINIMUM_BODY_COUNT
from .gr15_eih_1pn import (
    GR15EIH1PNIntegrationError,
    GR15EIH1PNResult,
    integrate_gr15_eih_1pn,
)
from .solar_system.eih_1pn import COORDINATE_SCOPE, EIH1PNParameters, SCIENTIFIC_CLAIM_STATE


CPU_BACKEND_ID = "JX_GR15_EIH1PN_CPU_THREADS_V1"
CUDA_BACKEND_ID = "JX_GR15_EIH1PN_CUDA_NUMPY_V1"
AUTO_API_ID = "JX_GR15_EIH1PN_BATCH_V1"
MAXIMUM_SYSTEM_COUNT = 65_535
DEFAULT_CPU_WORKERS = 8


class GR15EIH1PNBatchError(RuntimeError):
    """Base error for the supported NumPy batch boundary."""


class GR15EIH1PNBatchContractError(GR15EIH1PNBatchError, GR15ContractError):
    """A caller violated the supported NumPy batch contract."""


class GR15EIH1PNBatchIntegrationError(GR15EIH1PNBatchError):
    """One CPU batch system failed closed without releasing partial output."""

    def __init__(self, message: str, *, system: int, cause: BaseException) -> None:
        super().__init__(message)
        self.system = system
        self.cause = cause
        self.status = getattr(cause, "status", None)
        self.status_name = getattr(cause, "status_name", None)


@dataclass(frozen=True, slots=True)
class GR15EIH1PNDispatchPolicy:
    """One explicit, machine-bound automatic-routing calibration."""

    calibration_id: str
    cpu_model: str
    cuda_device_name: str
    body_count: int
    absolute_duration_seconds: float
    cpu_workers: int
    cuda_minimum_system_count: int
    largest_measured_system_count: int

    def __post_init__(self) -> None:
        for value, label in (
            (self.calibration_id, "calibration_id"),
            (self.cpu_model, "cpu_model"),
            (self.cuda_device_name, "cuda_device_name"),
        ):
            if type(value) is not str or not value.strip():
                raise GR15EIH1PNBatchContractError(f"{label} must be a nonempty string")
        if type(self.body_count) is not int or not MINIMUM_BODY_COUNT <= self.body_count <= MAXIMUM_BODY_COUNT:
            raise GR15EIH1PNBatchContractError("body_count must be an integer from 2 through 32")
        if (
            type(self.absolute_duration_seconds) is not float
            or not math.isfinite(self.absolute_duration_seconds)
            or self.absolute_duration_seconds <= 0.0
        ):
            raise GR15EIH1PNBatchContractError(
                "absolute_duration_seconds must be a positive finite built-in float"
            )
        for value, label in (
            (self.cpu_workers, "cpu_workers"),
            (self.cuda_minimum_system_count, "cuda_minimum_system_count"),
            (self.largest_measured_system_count, "largest_measured_system_count"),
        ):
            if type(value) is not int or value <= 0:
                raise GR15EIH1PNBatchContractError(f"{label} must be a positive integer")
        if self.cuda_minimum_system_count > self.largest_measured_system_count:
            raise GR15EIH1PNBatchContractError(
                "the CUDA threshold must not exceed the largest measured count"
            )


RTX_5060_TI_TEN_YEAR_ELEVEN_BODY_POLICY = GR15EIH1PNDispatchPolicy(
    calibration_id="jx.gr15.dispatch.ryzen-5500.rtx-5060-ti.11-body.10-year.v1",
    cpu_model="AMD Ryzen 5 5500",
    cuda_device_name="NVIDIA GeForce RTX 5060 Ti",
    body_count=11,
    absolute_duration_seconds=315_576_000.0,
    cpu_workers=8,
    cuda_minimum_system_count=128,
    largest_measured_system_count=128,
)
DEFAULT_DISPATCH_POLICY = RTX_5060_TI_TEN_YEAR_ELEVEN_BODY_POLICY


@dataclass(frozen=True, slots=True)
class GR15EIH1PNDispatchDecision:
    """Auditable backend choice made before integration begins."""

    requested_backend: str
    backend_id: str
    reason: str
    calibration_id: str | None
    performance_calibrated: bool
    threshold_extrapolated: bool
    cpu_model: str
    cuda_device_name: str | None
    system_count: int
    body_count: int
    absolute_duration_seconds: float
    cpu_workers: int

    def __post_init__(self) -> None:
        if self.requested_backend not in {"auto", "cpu", "cuda"}:
            raise GR15EIH1PNBatchContractError("decision requested_backend is invalid")
        if self.backend_id not in {CPU_BACKEND_ID, CUDA_BACKEND_ID}:
            raise GR15EIH1PNBatchContractError("decision backend_id is invalid")
        if type(self.reason) is not str or not self.reason:
            raise GR15EIH1PNBatchContractError("decision reason is missing")
        if type(self.performance_calibrated) is not bool or type(self.threshold_extrapolated) is not bool:
            raise GR15EIH1PNBatchContractError("decision calibration flags must be booleans")


@dataclass(frozen=True, slots=True)
class GR15EIH1PNSystemAudit:
    """Backend-neutral adaptive and force-domain accounting for one system."""

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
    observed_maximum_compactness: float
    observed_maximum_speed_fraction_squared: float
    replay_digest: str | None

    def __post_init__(self) -> None:
        if self.attempted_steps != self.accepted_steps + self.rejected_steps:
            raise GR15EIH1PNBatchContractError("system attempted-step accounting is inconsistent")
        if self.accepted_steps <= 0 or self.rejected_steps < 0:
            raise GR15EIH1PNBatchContractError("system accepted/rejected accounting is invalid")
        if self.history_commits != self.accepted_steps:
            raise GR15EIH1PNBatchContractError("system history accounting is inconsistent")


@dataclass(frozen=True, slots=True)
class GR15EIH1PNBatchResult:
    """Owned NumPy trajectories with backend-neutral accounting."""

    checkpoint_positions_km: np.ndarray
    checkpoint_velocities_km_s: np.ndarray
    positions_km: np.ndarray
    velocities_km_s: np.ndarray
    checkpoint_epochs: tuple[float, ...]
    system_audits: tuple[GR15EIH1PNSystemAudit, ...]
    decision: GR15EIH1PNDispatchDecision
    system_count: int
    body_count: int
    elapsed_seconds: float
    input_transfer_seconds: float
    output_transfer_seconds: float
    worker_count: int
    kernel_launch_count: int
    api_id: str = AUTO_API_ID
    coordinate_scope: str = COORDINATE_SCOPE
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    production_authorized: bool = False
    exact_general_relativity_claimed: bool = False
    ephemeris_equivalence_claimed: bool = False
    general_superiority_claimed: bool = False

    def __post_init__(self) -> None:
        expected = (self.system_count, len(self.checkpoint_epochs), self.body_count, 3)
        for value, label in (
            (self.checkpoint_positions_km, "checkpoint_positions_km"),
            (self.checkpoint_velocities_km_s, "checkpoint_velocities_km_s"),
        ):
            if (
                type(value) is not np.ndarray
                or value.dtype != np.dtype(np.float64)
                or value.shape != expected
                or value.flags.writeable
                or not value.flags.c_contiguous
                or not np.all(np.isfinite(value))
            ):
                raise GR15EIH1PNBatchContractError(f"{label} is invalid")
        if self.positions_km.shape != expected[0:1] + expected[2:] or self.velocities_km_s.shape != self.positions_km.shape:
            raise GR15EIH1PNBatchContractError("final batch state shape is invalid")
        if self.positions_km.flags.writeable or self.velocities_km_s.flags.writeable:
            raise GR15EIH1PNBatchContractError("final batch states must be read-only")
        if len(self.system_audits) != self.system_count:
            raise GR15EIH1PNBatchContractError("system audit count changed")
        if self.api_id != AUTO_API_ID or self.coordinate_scope != COORDINATE_SCOPE:
            raise GR15EIH1PNBatchContractError("batch identity changed")
        if self.scientific_claim_state != SCIENTIFIC_CLAIM_STATE:
            raise GR15EIH1PNBatchContractError("batch scientific claim state changed")
        if (
            self.production_authorized
            or self.exact_general_relativity_claimed
            or self.ephemeris_equivalence_claimed
            or self.general_superiority_claimed
        ):
            raise GR15EIH1PNBatchContractError("batch result elevated a scientific claim")
        for value, label in (
            (self.elapsed_seconds, "elapsed_seconds"),
            (self.input_transfer_seconds, "input_transfer_seconds"),
            (self.output_transfer_seconds, "output_transfer_seconds"),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise GR15EIH1PNBatchContractError(f"{label} is invalid")
        if self.decision.backend_id == CPU_BACKEND_ID:
            if not 1 <= self.worker_count <= self.system_count or self.kernel_launch_count != 0:
                raise GR15EIH1PNBatchContractError("CPU execution accounting is invalid")
        else:
            if self.worker_count != 0 or self.kernel_launch_count != 1:
                raise GR15EIH1PNBatchContractError("CUDA execution accounting is invalid")


def _readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def _require_batch_inputs(
    positions_km: np.ndarray,
    velocities_km_s: np.ndarray,
    gravitational_parameters_km3_s2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    for value, label in (
        (positions_km, "positions_km"),
        (velocities_km_s, "velocities_km_s"),
    ):
        if (
            type(value) is not np.ndarray
            or value.dtype != np.dtype(np.float64)
            or value.ndim != 3
            or value.shape[2] != 3
            or not value.flags.c_contiguous
            or not np.all(np.isfinite(value))
        ):
            raise GR15EIH1PNBatchContractError(
                f"{label} must be a finite C-contiguous NumPy float64 array with shape (systems,bodies,3)"
            )
    if velocities_km_s.shape != positions_km.shape:
        raise GR15EIH1PNBatchContractError("position and velocity shapes must match")
    system_count, body_count, _ = positions_km.shape
    if not 1 <= system_count <= MAXIMUM_SYSTEM_COUNT:
        raise GR15EIH1PNBatchContractError(
            f"system count must be from 1 through {MAXIMUM_SYSTEM_COUNT}"
        )
    if not MINIMUM_BODY_COUNT <= body_count <= MAXIMUM_BODY_COUNT:
        raise GR15EIH1PNBatchContractError("body count must be from 2 through 32")
    gm = gravitational_parameters_km3_s2
    if (
        type(gm) is not np.ndarray
        or gm.dtype != np.dtype(np.float64)
        or gm.shape not in {(body_count,), (system_count, body_count)}
        or not gm.flags.c_contiguous
        or not np.all(np.isfinite(gm))
        or np.any(gm <= 0.0)
    ):
        raise GR15EIH1PNBatchContractError(
            "gravitational_parameters_km3_s2 must be positive finite C-contiguous NumPy float64 with shape (bodies,) or (systems,bodies)"
        )
    return positions_km, velocities_km_s, gm, system_count, body_count


def _require_spec_parameters(spec: GR15Spec, parameters: EIH1PNParameters) -> None:
    if type(spec) is not GR15Spec:
        raise GR15EIH1PNBatchContractError("spec must be an exact GR15Spec")
    if type(parameters) is not EIH1PNParameters:
        raise GR15EIH1PNBatchContractError("parameters must be an exact EIH1PNParameters")


def _worker_count(value: int | None, system_count: int) -> int:
    if value is None:
        value = min(DEFAULT_CPU_WORKERS, system_count)
    if type(value) is not int or not 1 <= value <= system_count:
        raise GR15EIH1PNBatchContractError(
            "cpu_workers must be an integer from 1 through system_count"
        )
    return value


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="ascii").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, UnicodeError):
        pass
    return "UNKNOWN"


def _audit(value: Any) -> GR15EIH1PNSystemAudit:
    return GR15EIH1PNSystemAudit(
        checkpoint_accepted_steps=tuple(value.checkpoint_accepted_steps),
        checkpoint_rejected_steps=tuple(value.checkpoint_rejected_steps),
        attempted_steps=int(value.attempted_steps),
        accepted_steps=int(value.accepted_steps),
        rejected_steps=int(value.rejected_steps),
        force_evaluations=int(value.force_evaluations),
        predictor_corrector_iterations=int(value.predictor_corrector_iterations),
        nonconverged_retries=int(value.nonconverged_retries),
        polynomial_predictor_trials=int(value.polynomial_predictor_trials),
        constant_predictor_trials=int(value.constant_predictor_trials),
        history_commits=int(value.history_commits),
        terminal_force_reuses=int(value.terminal_force_reuses),
        terminal_force_sweeps=int(value.terminal_force_sweeps),
        maximum_error_ratio=float(value.maximum_error_ratio),
        minimum_accepted_step=float(value.minimum_accepted_step),
        maximum_accepted_step=float(value.maximum_accepted_step),
        final_proposed_step=float(value.final_proposed_step),
        maximum_corrector_residual=float(value.maximum_corrector_residual),
        observed_maximum_compactness=float(value.observed_maximum_compactness),
        observed_maximum_speed_fraction_squared=float(
            value.observed_maximum_speed_fraction_squared
        ),
        replay_digest=getattr(value, "replay_digest", None),
    )


def _result(
    results: tuple[GR15EIH1PNResult, ...],
    decision: GR15EIH1PNDispatchDecision,
    elapsed_seconds: float,
    worker_count: int,
) -> GR15EIH1PNBatchResult:
    checkpoint_positions = _readonly(
        np.ascontiguousarray(
            np.stack([item.checkpoint_positions for item in results]),
            dtype=np.float64,
        )
    )
    checkpoint_velocities = _readonly(
        np.ascontiguousarray(
            np.stack([item.checkpoint_velocities for item in results]),
            dtype=np.float64,
        )
    )
    return GR15EIH1PNBatchResult(
        checkpoint_positions_km=checkpoint_positions,
        checkpoint_velocities_km_s=checkpoint_velocities,
        positions_km=_readonly(checkpoint_positions[:, -1]),
        velocities_km_s=_readonly(checkpoint_velocities[:, -1]),
        checkpoint_epochs=results[0].checkpoint_epochs,
        system_audits=tuple(_audit(item) for item in results),
        decision=decision,
        system_count=len(results),
        body_count=checkpoint_positions.shape[2],
        elapsed_seconds=elapsed_seconds,
        input_transfer_seconds=0.0,
        output_transfer_seconds=0.0,
        worker_count=worker_count,
        kernel_launch_count=0,
    )


def _integrate_cpu_validated(
    positions: np.ndarray,
    velocities: np.ndarray,
    gm: np.ndarray,
    spec: GR15Spec,
    parameters: EIH1PNParameters,
    workers: int,
    decision: GR15EIH1PNDispatchDecision,
) -> GR15EIH1PNBatchResult:
    system_count = positions.shape[0]

    def run(system: int) -> GR15EIH1PNResult:
        system_gm = gm if gm.ndim == 1 else gm[system]
        try:
            return integrate_gr15_eih_1pn(
                positions[system], velocities[system], system_gm, spec, parameters
            )
        except GR15EIH1PNIntegrationError as exc:
            raise GR15EIH1PNBatchIntegrationError(
                f"CPU GR15-EIH1PN batch system {system} failed: {exc}",
                system=system,
                cause=exc,
            ) from exc
        except GR15ContractError as exc:
            raise GR15EIH1PNBatchIntegrationError(
                f"CPU GR15-EIH1PN batch system {system} violated the native contract: {exc}",
                system=system,
                cause=exc,
            ) from exc

    started = time.perf_counter_ns()
    if workers == 1:
        results = tuple(run(system) for system in range(system_count))
    else:
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="jx-gr15-eih",
        ) as executor:
            futures: tuple[Future[GR15EIH1PNResult], ...] = tuple(
                executor.submit(run, system) for system in range(system_count)
            )
            results = tuple(future.result() for future in futures)
    elapsed = (time.perf_counter_ns() - started) * 1.0e-9
    return _result(results, decision, elapsed, workers)


def integrate_gr15_eih_1pn_cpu_batch(
    positions_km: np.ndarray,
    velocities_km_s: np.ndarray,
    gravitational_parameters_km3_s2: np.ndarray,
    spec: GR15Spec,
    parameters: EIH1PNParameters,
    *,
    workers: int | None = None,
) -> GR15EIH1PNBatchResult:
    """Advance independent NumPy systems concurrently with native CPU GR15."""

    _require_spec_parameters(spec, parameters)
    positions, velocities, gm, systems, bodies = _require_batch_inputs(
        positions_km, velocities_km_s, gravitational_parameters_km3_s2
    )
    worker_count = _worker_count(workers, systems)
    decision = GR15EIH1PNDispatchDecision(
        requested_backend="cpu",
        backend_id=CPU_BACKEND_ID,
        reason="caller selected the supported CPU batch API",
        calibration_id=None,
        performance_calibrated=False,
        threshold_extrapolated=False,
        cpu_model=_cpu_model(),
        cuda_device_name=None,
        system_count=systems,
        body_count=bodies,
        absolute_duration_seconds=abs(spec.final_epoch - spec.initial_epoch),
        cpu_workers=worker_count,
    )
    return _integrate_cpu_validated(
        positions, velocities, gm, spec, parameters, worker_count, decision
    )


def _cuda_identity() -> dict[str, Any]:
    from .gr15_eih_1pn_cuda import gr15_eih_1pn_cuda_runtime_identity

    return gr15_eih_1pn_cuda_runtime_identity()


def select_gr15_eih_1pn_backend(
    system_count: int,
    body_count: int,
    spec: GR15Spec,
    *,
    backend: Literal["auto", "cpu", "cuda"] = "auto",
    cpu_workers: int | None = None,
    policy: GR15EIH1PNDispatchPolicy = DEFAULT_DISPATCH_POLICY,
) -> GR15EIH1PNDispatchDecision:
    """Select a backend without executing an integration."""

    if type(system_count) is not int or not 1 <= system_count <= MAXIMUM_SYSTEM_COUNT:
        raise GR15EIH1PNBatchContractError("system_count is outside the supported range")
    if type(body_count) is not int or not MINIMUM_BODY_COUNT <= body_count <= MAXIMUM_BODY_COUNT:
        raise GR15EIH1PNBatchContractError("body_count is outside the supported range")
    if type(spec) is not GR15Spec:
        raise GR15EIH1PNBatchContractError("spec must be an exact GR15Spec")
    if backend not in {"auto", "cpu", "cuda"}:
        raise GR15EIH1PNBatchContractError("backend must be 'auto', 'cpu', or 'cuda'")
    if type(policy) is not GR15EIH1PNDispatchPolicy:
        raise GR15EIH1PNBatchContractError("policy must be an exact GR15EIH1PNDispatchPolicy")
    workers = _worker_count(cpu_workers, system_count)
    duration = abs(spec.final_epoch - spec.initial_epoch)
    cpu_model = _cpu_model()
    common = {
        "requested_backend": backend,
        "cpu_model": cpu_model,
        "system_count": system_count,
        "body_count": body_count,
        "absolute_duration_seconds": duration,
        "cpu_workers": workers,
    }
    if backend == "cpu":
        return GR15EIH1PNDispatchDecision(
            backend_id=CPU_BACKEND_ID,
            reason="caller forced CPU",
            calibration_id=None,
            performance_calibrated=False,
            threshold_extrapolated=False,
            cuda_device_name=None,
            **common,
        )
    if backend == "cuda":
        identity = _cuda_identity()
        return GR15EIH1PNDispatchDecision(
            backend_id=CUDA_BACKEND_ID,
            reason="caller forced CUDA; automatic calibration checks were bypassed",
            calibration_id=None,
            performance_calibrated=False,
            threshold_extrapolated=False,
            cuda_device_name=str(identity["device_name"]),
            **common,
        )
    scope_matches = (
        cpu_model == policy.cpu_model
        and body_count == policy.body_count
        and duration == policy.absolute_duration_seconds
        and workers == min(policy.cpu_workers, system_count)
    )
    if not scope_matches:
        return GR15EIH1PNDispatchDecision(
            backend_id=CPU_BACKEND_ID,
            reason="workload or CPU runtime is outside the registered CUDA crossover profile",
            calibration_id=policy.calibration_id,
            performance_calibrated=False,
            threshold_extrapolated=False,
            cuda_device_name=None,
            **common,
        )
    if system_count < policy.cuda_minimum_system_count:
        return GR15EIH1PNDispatchDecision(
            backend_id=CPU_BACKEND_ID,
            reason="system count is below the measured CUDA crossover",
            calibration_id=policy.calibration_id,
            performance_calibrated=system_count <= 64,
            threshold_extrapolated=64 < system_count < policy.cuda_minimum_system_count,
            cuda_device_name=None,
            **common,
        )
    try:
        identity = _cuda_identity()
    except Exception as exc:
        return GR15EIH1PNDispatchDecision(
            backend_id=CPU_BACKEND_ID,
            reason=f"CUDA profile probe unavailable; CPU fallback: {type(exc).__name__}",
            calibration_id=policy.calibration_id,
            performance_calibrated=False,
            threshold_extrapolated=False,
            cuda_device_name=None,
            **common,
        )
    device_name = str(identity["device_name"])
    if device_name != policy.cuda_device_name:
        return GR15EIH1PNDispatchDecision(
            backend_id=CPU_BACKEND_ID,
            reason="CUDA device is outside the registered crossover profile",
            calibration_id=policy.calibration_id,
            performance_calibrated=False,
            threshold_extrapolated=False,
            cuda_device_name=device_name,
            **common,
        )
    extrapolated = system_count > policy.largest_measured_system_count
    return GR15EIH1PNDispatchDecision(
        backend_id=CUDA_BACKEND_ID,
        reason=(
            "matched the measured CUDA crossover"
            if not extrapolated
            else "system count exceeds the measured CUDA crossover; monotonic scaling is extrapolated"
        ),
        calibration_id=policy.calibration_id,
        performance_calibrated=not extrapolated,
        threshold_extrapolated=extrapolated,
        cuda_device_name=device_name,
        **common,
    )


def _integrate_cuda_validated(
    positions: np.ndarray,
    velocities: np.ndarray,
    gm: np.ndarray,
    spec: GR15Spec,
    parameters: EIH1PNParameters,
    decision: GR15EIH1PNDispatchDecision,
) -> GR15EIH1PNBatchResult:
    try:
        import cupy as cp

        from .gr15_eih_1pn_cuda import integrate_gr15_eih_1pn_cuda_batch
    except (ImportError, OSError) as exc:
        raise GR15EIH1PNBatchError(f"CUDA runtime unavailable: {exc}") from exc
    started = time.perf_counter_ns()
    transfer_started = time.perf_counter_ns()
    device_positions = cp.asarray(positions)
    device_velocities = cp.asarray(velocities)
    device_gm = cp.asarray(gm)
    cp.cuda.get_current_stream().synchronize()
    input_transfer = (time.perf_counter_ns() - transfer_started) * 1.0e-9
    cuda_result = integrate_gr15_eih_1pn_cuda_batch(
        device_positions, device_velocities, device_gm, spec, parameters
    )
    output_started = time.perf_counter_ns()
    checkpoint_positions = np.ascontiguousarray(
        cp.asnumpy(cuda_result.checkpoint_positions_km), dtype=np.float64
    )
    checkpoint_velocities = np.ascontiguousarray(
        cp.asnumpy(cuda_result.checkpoint_velocities_km_s), dtype=np.float64
    )
    output_transfer = (time.perf_counter_ns() - output_started) * 1.0e-9
    elapsed = (time.perf_counter_ns() - started) * 1.0e-9
    checkpoint_positions = _readonly(checkpoint_positions)
    checkpoint_velocities = _readonly(checkpoint_velocities)
    return GR15EIH1PNBatchResult(
        checkpoint_positions_km=checkpoint_positions,
        checkpoint_velocities_km_s=checkpoint_velocities,
        positions_km=_readonly(checkpoint_positions[:, -1]),
        velocities_km_s=_readonly(checkpoint_velocities[:, -1]),
        checkpoint_epochs=cuda_result.checkpoint_epochs,
        system_audits=tuple(_audit(item) for item in cuda_result.system_audits),
        decision=decision,
        system_count=positions.shape[0],
        body_count=positions.shape[1],
        elapsed_seconds=elapsed,
        input_transfer_seconds=input_transfer,
        output_transfer_seconds=output_transfer,
        worker_count=0,
        kernel_launch_count=1,
    )


def integrate_gr15_eih_1pn_batch(
    positions_km: np.ndarray,
    velocities_km_s: np.ndarray,
    gravitational_parameters_km3_s2: np.ndarray,
    spec: GR15Spec,
    parameters: EIH1PNParameters,
    *,
    backend: Literal["auto", "cpu", "cuda"] = "auto",
    cpu_workers: int | None = None,
    policy: GR15EIH1PNDispatchPolicy = DEFAULT_DISPATCH_POLICY,
) -> GR15EIH1PNBatchResult:
    """Advance independent NumPy systems with audited CPU/CUDA routing.

    ``auto`` uses CUDA only inside the registered performance profile and
    otherwise falls back to CPU.  ``cuda`` is an explicit caller override and
    raises if CUDA is unavailable.  Returned trajectories are always owned,
    read-only NumPy arrays regardless of the selected backend.
    """

    _require_spec_parameters(spec, parameters)
    positions, velocities, gm, systems, bodies = _require_batch_inputs(
        positions_km, velocities_km_s, gravitational_parameters_km3_s2
    )
    decision = select_gr15_eih_1pn_backend(
        systems,
        bodies,
        spec,
        backend=backend,
        cpu_workers=cpu_workers,
        policy=policy,
    )
    if decision.backend_id == CUDA_BACKEND_ID:
        return _integrate_cuda_validated(
            positions, velocities, gm, spec, parameters, decision
        )
    return _integrate_cpu_validated(
        positions,
        velocities,
        gm,
        spec,
        parameters,
        decision.cpu_workers,
        decision,
    )


__all__ = [
    "AUTO_API_ID",
    "CPU_BACKEND_ID",
    "CUDA_BACKEND_ID",
    "DEFAULT_CPU_WORKERS",
    "DEFAULT_DISPATCH_POLICY",
    "GR15EIH1PNBatchContractError",
    "GR15EIH1PNBatchError",
    "GR15EIH1PNBatchIntegrationError",
    "GR15EIH1PNBatchResult",
    "GR15EIH1PNDispatchDecision",
    "GR15EIH1PNDispatchPolicy",
    "GR15EIH1PNSystemAudit",
    "MAXIMUM_SYSTEM_COUNT",
    "RTX_5060_TI_TEN_YEAR_ELEVEN_BODY_POLICY",
    "integrate_gr15_eih_1pn_batch",
    "integrate_gr15_eih_1pn_cpu_batch",
    "select_gr15_eih_1pn_backend",
]
