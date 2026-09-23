"""Supported CPU Gauss--Radau order-15 integration boundary.

The backend advances 2--32 fully mutual, unsoftened Newtonian point masses in
binary64 arithmetic.  It packages the validated V2 cross-step predictor and
native collocation core; it does not compile code at runtime or import the
benchmark tree.

This is a supported software interface with a deliberately narrow scientific
scope.  Results remain ``SCREENING_ONLY`` and are not production ephemerides.
All units are caller selected but must be mutually consistent: epochs and
steps share one time unit, positions one length unit, velocities length/time,
and each gravitational parameter length^3/time^2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import time
from typing import Any

import numpy as np


METHOD_ID = "JX_GAUSS_RADAU15_V2"
MODEL_ID = "jx.gr15.mutual-unsoftened-newtonian.v1"
SCIENTIFIC_CLAIM_STATE = "SCREENING_ONLY"
MINIMUM_BODY_COUNT = 2
MAXIMUM_BODY_COUNT = 32
STAGE_COUNT = 8
EXPECTED_CORE_SOURCE_SHA256 = (
    "9a6cb532e88c732f6653c2763d9510036432d9930d930c19a5d259a03b3b514c"
)

STATUS_SUCCESS = 0
STATUS_INPUT_DOMAIN = 1
STATUS_NONFINITE = 2
STATUS_FORCE_SINGULARITY = 3
STATUS_STEP_LIMIT = 4
STATUS_REJECTION_LIMIT = 5
STATUS_MINIMUM_STEP = 6
STATUS_CHECKPOINT_SCHEDULE = 7

STATUS_NAMES = {
    STATUS_SUCCESS: "SUCCESS",
    STATUS_INPUT_DOMAIN: "INPUT_DOMAIN",
    STATUS_NONFINITE: "NONFINITE",
    STATUS_FORCE_SINGULARITY: "FORCE_SINGULARITY",
    STATUS_STEP_LIMIT: "STEP_LIMIT",
    STATUS_REJECTION_LIMIT: "REJECTION_LIMIT",
    STATUS_MINIMUM_STEP: "MINIMUM_STEP",
    STATUS_CHECKPOINT_SCHEDULE: "CHECKPOINT_SCHEDULE",
}

_NATIVE_EXTENSION: Any | None = None


class GR15Error(RuntimeError):
    """Base class for supported GR15 runtime failures."""


class GR15ContractError(GR15Error, ValueError):
    """A request violates the documented GR15 input contract."""


class GR15UnavailableError(GR15Error):
    """The packaged native GR15 extension is unavailable or has wrong identity."""


class GR15IntegrationError(GR15Error):
    """The native solver failed closed with an explicit status."""

    def __init__(self, message: str, *, status: int) -> None:
        super().__init__(message)
        self.status = status
        self.status_name = STATUS_NAMES.get(status, "UNKNOWN")


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GR15ContractError(f"{label} must be a finite real scalar")
    checked = float(value)
    if not math.isfinite(checked):
        raise GR15ContractError(f"{label} must be finite")
    return checked


@dataclass(frozen=True, slots=True)
class GR15Spec:
    """Adaptive controller and exact-checkpoint schedule.

    ``epsilon`` controls the ratio of the highest acceleration-polynomial
    coefficient to the maximum sampled acceleration.  It is not a conventional
    position ``rtol`` or ``atol``.
    """

    initial_epoch: float
    final_epoch: float
    intermediate_epochs: tuple[float, ...] = ()
    initial_step: float = 1.0e-2
    minimum_step: float = 1.0e-15
    maximum_step: float = 1.0
    epsilon: float = 1.0e-9
    safety_factor: float = 0.9
    minimum_scale_factor: float = 0.25
    maximum_scale_factor: float = 4.0
    convergence_factor: float = 16.0
    maximum_iterations: int = 12
    maximum_steps: int = 10_000_000
    maximum_rejections: int = 100_000

    def __post_init__(self) -> None:
        for name in (
            "initial_epoch",
            "final_epoch",
            "initial_step",
            "minimum_step",
            "maximum_step",
            "epsilon",
            "safety_factor",
            "minimum_scale_factor",
            "maximum_scale_factor",
            "convergence_factor",
        ):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))
        if type(self.intermediate_epochs) is not tuple:
            raise GR15ContractError("intermediate_epochs must be an exact tuple")
        intermediate = tuple(
            _finite_float(value, f"intermediate_epochs[{index}]")
            for index, value in enumerate(self.intermediate_epochs)
        )
        object.__setattr__(self, "intermediate_epochs", intermediate)
        if self.initial_epoch == self.final_epoch:
            raise GR15ContractError("initial_epoch and final_epoch must differ")
        direction = 1.0 if self.final_epoch > self.initial_epoch else -1.0
        previous = self.initial_epoch
        for epoch in (*intermediate, self.final_epoch):
            if direction * (epoch - previous) <= 0.0:
                raise GR15ContractError(
                    "checkpoint epochs must be strictly monotonic in integration order"
                )
            previous = epoch
        if not (
            self.minimum_step > 0.0
            and self.initial_step >= self.minimum_step
            and self.maximum_step >= self.initial_step
        ):
            raise GR15ContractError(
                "steps must satisfy 0 < minimum_step <= initial_step <= maximum_step"
            )
        if self.epsilon <= 0.0:
            raise GR15ContractError("epsilon must be positive")
        if not 0.0 < self.safety_factor < 1.0:
            raise GR15ContractError("safety_factor must be in the open interval (0, 1)")
        if not 0.0 < self.minimum_scale_factor <= 1.0:
            raise GR15ContractError("minimum_scale_factor must be in (0, 1]")
        if self.maximum_scale_factor < 1.0:
            raise GR15ContractError("maximum_scale_factor must be at least 1")
        if self.convergence_factor < 1.0:
            raise GR15ContractError("convergence_factor must be at least 1")
        if type(self.maximum_iterations) is not int or not (
            1 <= self.maximum_iterations <= 12
        ):
            raise GR15ContractError("maximum_iterations must be an integer in [1, 12]")
        if type(self.maximum_steps) is not int or self.maximum_steps < 1:
            raise GR15ContractError("maximum_steps must be a positive integer")
        if type(self.maximum_rejections) is not int or self.maximum_rejections < 0:
            raise GR15ContractError("maximum_rejections must be a nonnegative integer")

    @property
    def checkpoint_epochs(self) -> tuple[float, ...]:
        """Return the complete, exact output schedule."""

        return (self.initial_epoch, *self.intermediate_epochs, self.final_epoch)


@dataclass(slots=True)
class GR15Workspace:
    """Reusable output storage for one body count and checkpoint schedule.

    A result returned from an explicit workspace borrows read-only views of
    these arrays and remains numerically valid only until that workspace is
    reused.
    """

    body_count: int
    checkpoint_epochs: tuple[float, ...]
    checkpoint_positions: np.ndarray
    checkpoint_velocities: np.ndarray
    checkpoint_accepted_steps: np.ndarray
    checkpoint_rejected_steps: np.ndarray
    counters: np.ndarray
    metrics: np.ndarray
    _checkpoint_epoch_array: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        checkpoints = np.ascontiguousarray(
            np.asarray(self.checkpoint_epochs, dtype=np.float64)
        )
        checkpoints.setflags(write=False)
        self._checkpoint_epoch_array = checkpoints


@dataclass(frozen=True, slots=True)
class GR15Result:
    """Successful GR15 trajectory plus adaptive and predictor accounting."""

    method_id: str
    model_id: str
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
    maximum_error_ratio: float
    minimum_accepted_step: float
    maximum_accepted_step: float
    final_proposed_step: float
    elapsed_seconds: float
    replay_digest: str
    status: int
    production_authorized: bool = False
    scientific_qualification_claimed: bool = False

    def __post_init__(self) -> None:
        if self.method_id != METHOD_ID or self.model_id != MODEL_ID:
            raise GR15ContractError("GR15 result identity changed")
        if self.scientific_claim_state != SCIENTIFIC_CLAIM_STATE:
            raise GR15ContractError("GR15 scientific claim state changed")
        if self.production_authorized or self.scientific_qualification_claimed:
            raise GR15ContractError("GR15 screening results cannot elevate claims")
        if self.status != STATUS_SUCCESS:
            raise GR15ContractError("only successful native results may be constructed")
        if self.attempted_steps != self.accepted_steps + self.rejected_steps:
            raise GR15ContractError("GR15 step accounting is inconsistent")
        if self.accepted_steps <= 0 or self.rejected_steps < 0:
            raise GR15ContractError("GR15 accepted/rejected accounting is invalid")
        if self.history_commits != self.accepted_steps:
            raise GR15ContractError("predictor history was not committed exactly once per step")
        if (
            self.polynomial_predictor_trials + self.constant_predictor_trials
            != self.attempted_steps
        ):
            raise GR15ContractError("GR15 predictor accounting is inconsistent")


def _native_extension() -> Any:
    global _NATIVE_EXTENSION
    if _NATIVE_EXTENSION is not None:
        return _NATIVE_EXTENSION
    try:
        from . import _gr15_cpu
    except ImportError as exc:
        raise GR15UnavailableError(
            "the packaged jxplanetx._gr15_cpu extension is required; install a "
            "platform wheel or build JX from source with a C11 compiler"
        ) from exc
    if (
        _gr15_cpu.MAX_BODIES != MAXIMUM_BODY_COUNT
        or _gr15_cpu.STAGE_COUNT != STAGE_COUNT
        or _gr15_cpu.METHOD_ID != METHOD_ID
        or _gr15_cpu.SCIENTIFIC_CLAIM_STATE != SCIENTIFIC_CLAIM_STATE
        or _gr15_cpu.CORE_SOURCE_SHA256 != EXPECTED_CORE_SOURCE_SHA256
    ):
        raise GR15UnavailableError("the packaged GR15 native identity is invalid")
    _NATIVE_EXTENSION = _gr15_cpu
    return _NATIVE_EXTENSION


def gr15_runtime_identity() -> dict[str, str]:
    """Return the exact packaged native-core build identity."""

    extension = _native_extension()
    return {
        "method_id": extension.METHOD_ID,
        "source_sha256": extension.CORE_SOURCE_SHA256,
        "compiler_version": extension.COMPILER_VERSION,
        "compiler_flags": extension.COMPILER_FLAGS,
        "scientific_claim_state": extension.SCIENTIFIC_CLAIM_STATE,
    }


def gr15_tableau_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return read-only copies of the packaged collocation coefficients."""

    nodes = np.empty(STAGE_COUNT, dtype=np.float64)
    weights = np.empty(STAGE_COUNT, dtype=np.float64)
    matrix = np.empty((STAGE_COUNT, STAGE_COUNT), dtype=np.float64)
    try:
        status = int(_native_extension().tableau(nodes, weights, matrix))
    except (BufferError, OverflowError, TypeError, ValueError) as exc:
        raise GR15UnavailableError(f"the packaged tableau export failed: {exc}") from exc
    if status != STATUS_SUCCESS:
        raise GR15UnavailableError(f"the packaged tableau export returned status {status}")
    for value in (nodes, weights, matrix):
        value.setflags(write=False)
    return nodes, weights, matrix


def prepare_gr15_workspace(body_count: int, spec: GR15Spec) -> GR15Workspace:
    """Allocate reusable output buffers for one exact integration shape."""

    if type(body_count) is not int or not (
        MINIMUM_BODY_COUNT <= body_count <= MAXIMUM_BODY_COUNT
    ):
        raise GR15ContractError("GR15 supports exactly 2--32 bodies")
    if type(spec) is not GR15Spec:
        raise GR15ContractError("spec must be an exact GR15Spec")
    checkpoint_count = len(spec.checkpoint_epochs)
    shape = (checkpoint_count, body_count, 3)
    return GR15Workspace(
        body_count=body_count,
        checkpoint_epochs=spec.checkpoint_epochs,
        checkpoint_positions=np.empty(shape, dtype=np.float64),
        checkpoint_velocities=np.empty(shape, dtype=np.float64),
        checkpoint_accepted_steps=np.empty(checkpoint_count, dtype=np.int64),
        checkpoint_rejected_steps=np.empty(checkpoint_count, dtype=np.int64),
        counters=np.empty(9, dtype=np.int64),
        metrics=np.empty(4, dtype=np.float64),
    )


def _require_state(value: Any, body_count: int, label: str) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.float64)
        or value.shape != (body_count, 3)
        or not value.flags.c_contiguous
        or not np.all(np.isfinite(value))
    ):
        raise GR15ContractError(
            f"{label} must be a finite C-contiguous NumPy float64 "
            "array with shape (body_count, 3)"
        )
    return value


def _require_workspace_array(
    value: Any,
    *,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
    label: str,
) -> None:
    if (
        type(value) is not np.ndarray
        or value.dtype != dtype
        or value.shape != shape
        or not value.flags.c_contiguous
        or not value.flags.writeable
    ):
        raise GR15ContractError(f"workspace {label} buffer is invalid")


def _validate_workspace(
    workspace: GR15Workspace,
    body_count: int,
    spec: GR15Spec,
) -> None:
    if type(workspace) is not GR15Workspace:
        raise GR15ContractError("workspace must be an exact GR15Workspace")
    if (
        workspace.body_count != body_count
        or workspace.checkpoint_epochs != spec.checkpoint_epochs
    ):
        raise GR15ContractError("workspace does not match the request")
    checkpoint_count = len(spec.checkpoint_epochs)
    shape = (checkpoint_count, body_count, 3)
    _require_workspace_array(
        workspace.checkpoint_positions,
        dtype=np.dtype(np.float64),
        shape=shape,
        label="checkpoint_positions",
    )
    _require_workspace_array(
        workspace.checkpoint_velocities,
        dtype=np.dtype(np.float64),
        shape=shape,
        label="checkpoint_velocities",
    )
    for name in ("checkpoint_accepted_steps", "checkpoint_rejected_steps"):
        _require_workspace_array(
            getattr(workspace, name),
            dtype=np.dtype(np.int64),
            shape=(checkpoint_count,),
            label=name,
        )
    _require_workspace_array(
        workspace.counters,
        dtype=np.dtype(np.int64),
        shape=(9,),
        label="counters",
    )
    _require_workspace_array(
        workspace.metrics,
        dtype=np.dtype(np.float64),
        shape=(4,),
        label="metrics",
    )
    checkpoints = workspace._checkpoint_epoch_array
    if (
        type(checkpoints) is not np.ndarray
        or checkpoints.dtype != np.dtype(np.float64)
        or checkpoints.shape != (checkpoint_count,)
        or not checkpoints.flags.c_contiguous
        or checkpoints.flags.writeable
        or not np.array_equal(checkpoints, spec.checkpoint_epochs)
    ):
        raise GR15ContractError("workspace checkpoint cache is invalid")


def _readonly_view(value: np.ndarray) -> np.ndarray:
    view = value.view()
    view.setflags(write=False)
    return view


def _replay_digest(
    checkpoints: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    counters: np.ndarray,
    metrics: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"jx.gr15.supported-result.v1\0")
    for value in (checkpoints, positions, velocities, counters, metrics):
        digest.update(np.asarray(value).tobytes(order="C"))
    return digest.hexdigest()


def integrate_gr15(
    positions: np.ndarray,
    velocities: np.ndarray,
    gravitational_parameters: np.ndarray,
    spec: GR15Spec,
    *,
    workspace: GR15Workspace | None = None,
) -> GR15Result:
    """Integrate one supported Newtonian point-mass system with GR15.

    Inputs are never modified.  Exact collisions, exhausted controller limits,
    non-finite arithmetic, and unsupported inputs fail with explicit errors.
    """

    if type(spec) is not GR15Spec:
        raise GR15ContractError("spec must be an exact GR15Spec")
    if type(positions) is not np.ndarray or positions.ndim != 2:
        raise GR15ContractError("positions must be a two-dimensional NumPy array")
    body_count = positions.shape[0]
    if not MINIMUM_BODY_COUNT <= body_count <= MAXIMUM_BODY_COUNT:
        raise GR15ContractError("GR15 supports exactly 2--32 bodies")
    positions = _require_state(positions, body_count, "positions")
    velocities = _require_state(velocities, body_count, "velocities")
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
            "NumPy float64 vector"
        )
    if workspace is None:
        workspace = prepare_gr15_workspace(body_count, spec)
    _validate_workspace(workspace, body_count, spec)

    started = time.perf_counter_ns()
    try:
        status = int(
            _native_extension().integrate(
                body_count,
                len(spec.checkpoint_epochs),
                positions,
                velocities,
                gravitational_parameters,
                workspace._checkpoint_epoch_array,
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
    except (BufferError, OverflowError, TypeError, ValueError) as exc:
        raise GR15ContractError(f"the native GR15 boundary rejected its buffers: {exc}") from exc
    elapsed_seconds = (time.perf_counter_ns() - started) * 1.0e-9
    if status != STATUS_SUCCESS:
        status_name = STATUS_NAMES.get(status, "UNKNOWN")
        raise GR15IntegrationError(
            f"GR15 integration failed with status {status} ({status_name})",
            status=status,
        )

    checkpoint_positions = _readonly_view(workspace.checkpoint_positions)
    checkpoint_velocities = _readonly_view(workspace.checkpoint_velocities)
    counters = workspace.counters
    metrics = workspace.metrics
    return GR15Result(
        method_id=METHOD_ID,
        model_id=MODEL_ID,
        scientific_claim_state=SCIENTIFIC_CLAIM_STATE,
        checkpoint_epochs=spec.checkpoint_epochs,
        checkpoint_positions=checkpoint_positions,
        checkpoint_velocities=checkpoint_velocities,
        positions=_readonly_view(workspace.checkpoint_positions[-1]),
        velocities=_readonly_view(workspace.checkpoint_velocities[-1]),
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
        maximum_error_ratio=float(metrics[0]),
        minimum_accepted_step=float(metrics[1]),
        maximum_accepted_step=float(metrics[2]),
        final_proposed_step=float(metrics[3]),
        elapsed_seconds=elapsed_seconds,
        replay_digest=_replay_digest(
            workspace._checkpoint_epoch_array,
            workspace.checkpoint_positions,
            workspace.checkpoint_velocities,
            counters,
            metrics,
        ),
        status=status,
    )


integrate = integrate_gr15


__all__ = [
    "EXPECTED_CORE_SOURCE_SHA256",
    "GR15ContractError",
    "GR15Error",
    "GR15IntegrationError",
    "GR15Result",
    "GR15Spec",
    "GR15UnavailableError",
    "GR15Workspace",
    "MAXIMUM_BODY_COUNT",
    "METHOD_ID",
    "MINIMUM_BODY_COUNT",
    "MODEL_ID",
    "SCIENTIFIC_CLAIM_STATE",
    "STAGE_COUNT",
    "STATUS_CHECKPOINT_SCHEDULE",
    "STATUS_FORCE_SINGULARITY",
    "STATUS_INPUT_DOMAIN",
    "STATUS_MINIMUM_STEP",
    "STATUS_NAMES",
    "STATUS_NONFINITE",
    "STATUS_REJECTION_LIMIT",
    "STATUS_STEP_LIMIT",
    "STATUS_SUCCESS",
    "gr15_runtime_identity",
    "gr15_tableau_arrays",
    "integrate",
    "integrate_gr15",
    "prepare_gr15_workspace",
]
