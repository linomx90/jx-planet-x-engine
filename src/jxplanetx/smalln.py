"""Measured small-N CPU/CUDA routing for screening workloads.

This supported package interface advances independent 2--32-body, fully
mutual, unsoftened Newtonian systems with the JX adaptive RKF78 controller.
Measured low-lane routes use the packaged compiled CPU extension; measured
batch routes use the package-owned fused CUDA implementation.  The dispatcher
never transfers state implicitly between host and device.

The interface remains ``SCREENING_ONLY``.  It is not a production backend,
scientific qualification, collision handler, or general replacement for the
full JX engine or REBOUND solver portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from . import fused_cuda_adaptive_rkf78 as adaptive
from . import fused_cuda_rkf78 as fused


MODEL_ID = "jx.smalln.mutual-newtonian-adaptive-rkf78.v1"
SCIENTIFIC_CLAIM_STATE = "SCREENING_ONLY"
CPU_BACKEND = "jx-compiled-cpu-rkf78"
CUDA_BACKEND = "jx-fused-cuda-rkf78"
MINIMUM_BODY_COUNT = 2
MAXIMUM_BODY_COUNT = 32
QUALIFIED_LANE_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
EIGHT_LANE_CUDA_BODY_COUNTS = frozenset((32,))
EXPECTED_NATIVE_CORE_SOURCE_SHA256 = (
    "3952fb198482261d2ac210f3a9ef3f941af60c56c9e97bcf403b3b937c73397c"
)

SmallNSpec = adaptive.AdaptiveFusedSpec
SmallNCUDAWorkspace = adaptive.AdaptiveFusedWorkspace
SmallNCUDAResult = adaptive.AdaptiveFusedResult

_CPU_TABLEAU = fused.tableau_arrays()
for _tableau_array in _CPU_TABLEAU:
    _tableau_array.setflags(write=False)
_NATIVE_EXTENSION: Any | None = None


class SmallNError(RuntimeError):
    """A small-N request violates the supported screening contract."""


class SmallNUnavailableError(SmallNError):
    """The selected compiled CPU or CUDA runtime is unavailable."""


@dataclass(frozen=True, slots=True)
class SmallNCPUWorkspace:
    """Reusable CPU output storage for one integration shape.

    Results created with an explicit workspace borrow read-only views of these
    arrays and remain numerically valid only until the workspace is reused.
    """

    body_count: int
    checkpoint_epochs: tuple[float, ...]
    ledger_capacity: int
    checkpoint_positions: np.ndarray
    checkpoint_velocities: np.ndarray
    checkpoint_accepted_steps: np.ndarray
    checkpoint_rejected_steps: np.ndarray
    accepted_step_epochs: np.ndarray
    accepted_step_magnitudes: np.ndarray
    _checkpoint_epoch_array: np.ndarray = field(init=False, repr=False)
    _checkpoint_position_views: tuple[np.ndarray, ...] = field(
        init=False,
        repr=False,
    )
    _checkpoint_velocity_views: tuple[np.ndarray, ...] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        checkpoint_array = np.ascontiguousarray(
            np.asarray(self.checkpoint_epochs, dtype=np.float64)
        )
        checkpoint_array.setflags(write=False)
        object.__setattr__(self, "_checkpoint_epoch_array", checkpoint_array)
        object.__setattr__(
            self,
            "_checkpoint_position_views",
            self._readonly_checkpoint_views(self.checkpoint_positions),
        )
        object.__setattr__(
            self,
            "_checkpoint_velocity_views",
            self._readonly_checkpoint_views(self.checkpoint_velocities),
        )

    @staticmethod
    def _readonly_checkpoint_views(value: Any) -> tuple[np.ndarray, ...]:
        if type(value) is not np.ndarray or value.ndim < 1:
            return ()
        result = tuple(value[index].view() for index in range(value.shape[0]))
        for view in result:
            view.setflags(write=False)
        return result

@dataclass(frozen=True, slots=True)
class SmallNCPUResult:
    """Compiled-CPU result with complete checkpoint and accepted-step ledgers."""

    model_id: str
    scientific_claim_state: str
    positions: np.ndarray
    velocities: np.ndarray
    checkpoint_positions: tuple[np.ndarray, ...]
    checkpoint_velocities: tuple[np.ndarray, ...]
    checkpoint_accepted_steps: tuple[int, ...]
    checkpoint_rejected_steps: tuple[int, ...]
    accepted_step_epochs: tuple[float, ...]
    accepted_step_magnitudes: tuple[float, ...]
    attempted_steps: int
    accepted_steps: int
    rejected_steps: int
    maximum_normalized_error: float
    status: int
    production_authorized: bool = False
    scientific_qualification_claimed: bool = False

    def __post_init__(self) -> None:
        if self.model_id != MODEL_ID:
            raise SmallNError("small-N result model identity changed")
        if self.scientific_claim_state != SCIENTIFIC_CLAIM_STATE:
            raise SmallNError("small-N result claim state changed")
        if self.production_authorized or self.scientific_qualification_claimed:
            raise SmallNError("small-N result cannot authorize elevated claims")
        if self.status != 0:
            raise SmallNError("only successful native results may be constructed")
        if (
            type(self.attempted_steps) is not int
            or type(self.accepted_steps) is not int
            or type(self.rejected_steps) is not int
            or self.attempted_steps != self.accepted_steps + self.rejected_steps
            or self.accepted_steps <= 0
            or self.rejected_steps < 0
        ):
            raise SmallNError("small-N step accounting is invalid")
        if (
            type(self.maximum_normalized_error) is not float
            or not math.isfinite(self.maximum_normalized_error)
            or self.maximum_normalized_error < 0.0
        ):
            raise SmallNError("small-N maximum normalized error is invalid")
        if len(self.accepted_step_epochs) != self.accepted_steps or len(
            self.accepted_step_magnitudes
        ) != self.accepted_steps:
            raise SmallNError("small-N accepted-step ledger length changed")


@dataclass(frozen=True, slots=True)
class SmallNRoute:
    """One route selected only from the measured crossover table."""

    backend: str
    independent_systems: int
    body_count: int
    implicit_transfer: bool = False
    production_authorized: bool = False
    scientific_qualification_claimed: bool = False

    def __post_init__(self) -> None:
        if self.backend not in (CPU_BACKEND, CUDA_BACKEND):
            raise SmallNError("small-N route backend changed")
        if self.independent_systems not in QUALIFIED_LANE_COUNTS:
            raise SmallNError("small-N route lane count is not measured")
        if not MINIMUM_BODY_COUNT <= self.body_count <= MAXIMUM_BODY_COUNT:
            raise SmallNError("small-N route body count is unsupported")
        if (
            self.implicit_transfer
            or self.production_authorized
            or self.scientific_qualification_claimed
        ):
            raise SmallNError("small-N route cannot elevate claims or transfer state")


@dataclass(frozen=True, slots=True)
class SmallNDispatchResult:
    """Selected route and backend-specific screening result."""

    route: SmallNRoute
    payload: SmallNCPUResult | tuple[SmallNCPUResult, ...] | SmallNCUDAResult

    def __post_init__(self) -> None:
        if type(self.route) is not SmallNRoute:
            raise SmallNError("dispatch result route has the wrong type")


def select_smalln_route(independent_systems: int, body_count: int) -> SmallNRoute:
    """Select only a measured CPU or CUDA route; never interpolate."""

    if (
        type(independent_systems) is not int
        or independent_systems not in QUALIFIED_LANE_COUNTS
    ):
        raise SmallNError(
            "independent_systems must be a measured lane count: "
            f"{QUALIFIED_LANE_COUNTS}"
        )
    if (
        type(body_count) is not int
        or not MINIMUM_BODY_COUNT <= body_count <= MAXIMUM_BODY_COUNT
    ):
        raise SmallNError("small-N routing supports exactly 2--32 bodies")
    cuda_crossover = 8 if body_count in EIGHT_LANE_CUDA_BODY_COUNTS else 16
    return SmallNRoute(
        backend=(
            CUDA_BACKEND
            if independent_systems >= cuda_crossover
            else CPU_BACKEND
        ),
        independent_systems=independent_systems,
        body_count=body_count,
    )


def _native_extension() -> Any:
    global _NATIVE_EXTENSION
    if _NATIVE_EXTENSION is not None:
        return _NATIVE_EXTENSION
    try:
        from . import _smalln_cpu
    except ImportError as exc:
        raise SmallNUnavailableError(
            "the packaged jxplanetx._smalln_cpu extension is required; "
            "install a platform wheel or build JX from source with a C11 compiler"
        ) from exc
    if (
        _smalln_cpu.MAX_BODIES != MAXIMUM_BODY_COUNT
        or _smalln_cpu.STAGE_COUNT != fused.STAGE_COUNT
        or _smalln_cpu.SCIENTIFIC_CLAIM_STATE != SCIENTIFIC_CLAIM_STATE
        or _smalln_cpu.CORE_SOURCE_SHA256 != EXPECTED_NATIVE_CORE_SOURCE_SHA256
    ):
        raise SmallNUnavailableError("the packaged small-N CPU identity is invalid")
    _NATIVE_EXTENSION = _smalln_cpu
    return _NATIVE_EXTENSION


def smalln_cpu_runtime_identity() -> dict[str, str]:
    """Return the exact packaged native-core build identity."""

    extension = _native_extension()
    return {
        "source_sha256": extension.CORE_SOURCE_SHA256,
        "compiler_version": extension.COMPILER_VERSION,
        "compiler_flags": extension.COMPILER_FLAGS,
        "scientific_claim_state": extension.SCIENTIFIC_CLAIM_STATE,
    }


def prepare_smalln_cpu_workspace(
    body_count: int,
    spec: SmallNSpec,
) -> SmallNCPUWorkspace:
    """Allocate reusable CPU storage for one exact state and schedule shape."""

    if type(body_count) is not int or not 2 <= body_count <= MAXIMUM_BODY_COUNT:
        raise SmallNError("CPU workspace requires 2--32 bodies")
    if type(spec) is not SmallNSpec:
        raise SmallNError("spec must be an exact SmallNSpec")
    checkpoint_count = len(spec.checkpoint_epochs)
    state_shape = (checkpoint_count, body_count, 3)
    return SmallNCPUWorkspace(
        body_count=body_count,
        checkpoint_epochs=spec.checkpoint_epochs,
        ledger_capacity=spec.ledger_capacity,
        checkpoint_positions=np.empty(state_shape, dtype=np.float64),
        checkpoint_velocities=np.empty(state_shape, dtype=np.float64),
        checkpoint_accepted_steps=np.empty(checkpoint_count, dtype=np.int64),
        checkpoint_rejected_steps=np.empty(checkpoint_count, dtype=np.int64),
        accepted_step_epochs=np.empty(spec.ledger_capacity, dtype=np.float64),
        accepted_step_magnitudes=np.empty(spec.ledger_capacity, dtype=np.float64),
    )


def prepare_smalln_cpu_workspaces(
    independent_systems: int,
    body_count: int,
    spec: SmallNSpec,
) -> tuple[SmallNCPUWorkspace, ...]:
    """Allocate one CPU workspace per system for a measured CPU batch route."""

    route = select_smalln_route(independent_systems, body_count)
    if route.backend != CPU_BACKEND:
        raise SmallNError("the measured route is CUDA, not sequential CPU")
    return tuple(
        prepare_smalln_cpu_workspace(body_count, spec)
        for _ in range(independent_systems)
    )


def _require_cpu_state(value: Any, body_count: int, label: str) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.float64)
        or value.shape != (body_count, 3)
        or not value.flags.c_contiguous
        or not np.all(np.isfinite(value))
    ):
        raise SmallNError(
            f"{label} must be a finite C-contiguous NumPy float64 (bodies, 3) array"
        )
    return value


def _require_workspace_array(
    value: Any,
    *,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != dtype
        or value.shape != shape
        or not value.flags.c_contiguous
        or not value.flags.writeable
    ):
        raise SmallNError(f"workspace {label} buffer is invalid")
    return value


def _validate_cpu_workspace(
    workspace: SmallNCPUWorkspace,
    body_count: int,
    spec: SmallNSpec,
) -> None:
    if type(workspace) is not SmallNCPUWorkspace:
        raise SmallNError("workspace must be an exact SmallNCPUWorkspace")
    if (
        workspace.body_count != body_count
        or workspace.checkpoint_epochs != spec.checkpoint_epochs
        or workspace.ledger_capacity != spec.ledger_capacity
    ):
        raise SmallNError("CPU workspace does not match the request")
    checkpoint_count = len(spec.checkpoint_epochs)
    state_shape = (checkpoint_count, body_count, 3)
    checkpoint_array = workspace._checkpoint_epoch_array
    if (
        type(checkpoint_array) is not np.ndarray
        or checkpoint_array.dtype != np.dtype(np.float64)
        or checkpoint_array.shape != (checkpoint_count,)
        or not checkpoint_array.flags.c_contiguous
        or checkpoint_array.flags.writeable
        or not np.array_equal(checkpoint_array, spec.checkpoint_epochs)
    ):
        raise SmallNError("workspace checkpoint epoch cache is invalid")
    _require_workspace_array(
        workspace.checkpoint_positions,
        dtype=np.dtype(np.float64),
        shape=state_shape,
        label="checkpoint_positions",
    )
    _require_workspace_array(
        workspace.checkpoint_velocities,
        dtype=np.dtype(np.float64),
        shape=state_shape,
        label="checkpoint_velocities",
    )
    for name in ("checkpoint_accepted_steps", "checkpoint_rejected_steps"):
        _require_workspace_array(
            getattr(workspace, name),
            dtype=np.dtype(np.int64),
            shape=(checkpoint_count,),
            label=name,
        )
    for name in ("accepted_step_epochs", "accepted_step_magnitudes"):
        _require_workspace_array(
            getattr(workspace, name),
            dtype=np.dtype(np.float64),
            shape=(spec.ledger_capacity,),
            label=name,
        )
    for views, storage, label in (
        (
            workspace._checkpoint_position_views,
            workspace.checkpoint_positions,
            "checkpoint position views",
        ),
        (
            workspace._checkpoint_velocity_views,
            workspace.checkpoint_velocities,
            "checkpoint velocity views",
        ),
    ):
        if type(views) is not tuple or len(views) != checkpoint_count:
            raise SmallNError(f"workspace {label} cache is invalid")
        for view in views:
            if (
                type(view) is not np.ndarray
                or view.dtype != np.dtype(np.float64)
                or view.shape != (body_count, 3)
                or not view.flags.c_contiguous
                or view.flags.writeable
                or view.base is not storage
            ):
                raise SmallNError(f"workspace {label} cache is invalid")


def integrate_smalln_cpu(
    positions: np.ndarray,
    velocities: np.ndarray,
    gravitational_parameters: np.ndarray,
    spec: SmallNSpec,
    *,
    workspace: SmallNCPUWorkspace | None = None,
) -> SmallNCPUResult:
    """Integrate one small-N system with the packaged compiled CPU core."""

    if type(spec) is not SmallNSpec:
        raise SmallNError("spec must be an exact SmallNSpec")
    if type(positions) is not np.ndarray or positions.ndim != 2:
        raise SmallNError("CPU positions must be a two-dimensional NumPy array")
    body_count = positions.shape[0]
    positions = _require_cpu_state(positions, body_count, "positions")
    velocities = _require_cpu_state(velocities, body_count, "velocities")
    if not MINIMUM_BODY_COUNT <= body_count <= MAXIMUM_BODY_COUNT:
        raise SmallNError("compiled CPU integration supports 2--32 bodies")
    if (
        type(gravitational_parameters) is not np.ndarray
        or gravitational_parameters.dtype != np.dtype(np.float64)
        or gravitational_parameters.shape != (body_count,)
        or not gravitational_parameters.flags.c_contiguous
        or not np.all(np.isfinite(gravitational_parameters))
        or np.any(gravitational_parameters <= 0.0)
    ):
        raise SmallNError("GM must be a positive C-contiguous NumPy float64 vector")
    if workspace is None:
        workspace = prepare_smalln_cpu_workspace(body_count, spec)
    _validate_cpu_workspace(workspace, body_count, spec)

    coefficients, eighth, defect = _CPU_TABLEAU
    checkpoints = workspace._checkpoint_epoch_array
    extension = _native_extension()
    try:
        status, attempted, accepted, rejected, maximum_error = extension.integrate(
            body_count,
            len(checkpoints),
            positions,
            velocities,
            gravitational_parameters,
            checkpoints,
            coefficients,
            eighth,
            defect,
            spec.initial_step,
            spec.minimum_step,
            spec.maximum_step,
            spec.position_atol,
            spec.position_rtol,
            spec.velocity_atol,
            spec.velocity_rtol,
            spec.safety_factor,
            spec.minimum_scale_factor,
            spec.maximum_scale_factor,
            spec.maximum_steps,
            spec.maximum_rejections,
            spec.ledger_capacity,
            workspace.checkpoint_positions,
            workspace.checkpoint_velocities,
            workspace.checkpoint_accepted_steps,
            workspace.checkpoint_rejected_steps,
            workspace.accepted_step_epochs,
            workspace.accepted_step_magnitudes,
        )
    except (BufferError, OverflowError, TypeError, ValueError) as exc:
        raise SmallNError(f"compiled CPU integration rejected its buffers: {exc}") from exc
    status = int(status)
    if status != 0:
        raise SmallNError(f"compiled CPU integration failed with status {status}")
    accepted = int(accepted)
    checkpoint_positions = workspace._checkpoint_position_views
    checkpoint_velocities = workspace._checkpoint_velocity_views
    return SmallNCPUResult(
        model_id=MODEL_ID,
        scientific_claim_state=SCIENTIFIC_CLAIM_STATE,
        positions=checkpoint_positions[-1],
        velocities=checkpoint_velocities[-1],
        checkpoint_positions=checkpoint_positions,
        checkpoint_velocities=checkpoint_velocities,
        checkpoint_accepted_steps=tuple(
            int(value) for value in workspace.checkpoint_accepted_steps
        ),
        checkpoint_rejected_steps=tuple(
            int(value) for value in workspace.checkpoint_rejected_steps
        ),
        accepted_step_epochs=tuple(
            float(value) for value in workspace.accepted_step_epochs[:accepted]
        ),
        accepted_step_magnitudes=tuple(
            float(value) for value in workspace.accepted_step_magnitudes[:accepted]
        ),
        attempted_steps=int(attempted),
        accepted_steps=accepted,
        rejected_steps=int(rejected),
        maximum_normalized_error=float(maximum_error),
        status=status,
    )


def integrate_smalln(
    positions: Any,
    velocities: Any,
    gravitational_parameters: Any,
    spec: SmallNSpec,
    *,
    cpu_workspace: (
        SmallNCPUWorkspace | tuple[SmallNCPUWorkspace, ...] | None
    ) = None,
    cuda_workspace: SmallNCUDAWorkspace | None = None,
) -> SmallNDispatchResult:
    """Execute the measured route without implicit host/device transfers."""

    if type(spec) is not SmallNSpec:
        raise SmallNError("spec must be an exact SmallNSpec")
    if type(positions) is np.ndarray:
        if positions.ndim == 2:
            lane_count, body_count = 1, positions.shape[0]
            cpu_positions = positions
            cpu_velocities = velocities
            cpu_gm = gravitational_parameters
        elif positions.ndim == 3:
            lane_count, body_count = positions.shape[:2]
            route = select_smalln_route(lane_count, body_count)
            if route.backend != CPU_BACKEND:
                raise SmallNError(
                    "CUDA-selected routing requires caller-provided CuPy arrays"
                )
            if cpu_workspace is None:
                workspaces = prepare_smalln_cpu_workspaces(
                    lane_count,
                    body_count,
                    spec,
                )
            elif type(cpu_workspace) is tuple and len(cpu_workspace) == lane_count:
                workspaces = cpu_workspace
            else:
                raise SmallNError(
                    "CPU batch routing requires one workspace per system"
                )
            payload = tuple(
                integrate_smalln_cpu(
                    positions[lane],
                    velocities[lane],
                    (
                        gravitational_parameters[lane]
                        if getattr(gravitational_parameters, "ndim", None) == 2
                        else gravitational_parameters
                    ),
                    spec,
                    workspace=workspaces[lane],
                )
                for lane in range(lane_count)
            )
            return SmallNDispatchResult(route=route, payload=payload)
        else:
            raise SmallNError("NumPy state must have two or three dimensions")
        route = select_smalln_route(lane_count, body_count)
        if route.backend != CPU_BACKEND:
            raise SmallNError("single-system route identity changed")
        if type(cpu_workspace) is tuple:
            raise SmallNError("single-system CPU routing requires one workspace")
        return SmallNDispatchResult(
            route=route,
            payload=integrate_smalln_cpu(
                cpu_positions,
                cpu_velocities,
                cpu_gm,
                spec,
                workspace=cpu_workspace,
            ),
        )

    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SmallNUnavailableError(f"CuPy is unavailable: {exc}") from exc
    if not isinstance(positions, cp.ndarray) or positions.ndim != 3:
        raise SmallNError("state must be a NumPy CPU or CuPy CUDA array")
    lane_count, body_count, components = positions.shape
    if components != 3:
        raise SmallNError("CuPy state must have shape (lanes, bodies, 3)")
    route = select_smalln_route(lane_count, body_count)
    if route.backend != CUDA_BACKEND:
        raise SmallNError(
            "CPU-selected routing requires caller-provided NumPy arrays"
        )
    payload = adaptive.fused_adaptive_integrate(
        positions,
        velocities,
        gravitational_parameters,
        spec,
        workspace=cuda_workspace,
    )
    return SmallNDispatchResult(route=route, payload=payload)


__all__ = [
    "CPU_BACKEND",
    "CUDA_BACKEND",
    "EIGHT_LANE_CUDA_BODY_COUNTS",
    "EXPECTED_NATIVE_CORE_SOURCE_SHA256",
    "MAXIMUM_BODY_COUNT",
    "MINIMUM_BODY_COUNT",
    "MODEL_ID",
    "QUALIFIED_LANE_COUNTS",
    "SCIENTIFIC_CLAIM_STATE",
    "SmallNCPUResult",
    "SmallNCPUWorkspace",
    "SmallNCUDAResult",
    "SmallNCUDAWorkspace",
    "SmallNDispatchResult",
    "SmallNError",
    "SmallNRoute",
    "SmallNSpec",
    "SmallNUnavailableError",
    "integrate_smalln",
    "integrate_smalln_cpu",
    "prepare_smalln_cpu_workspace",
    "prepare_smalln_cpu_workspaces",
    "select_smalln_route",
    "smalln_cpu_runtime_identity",
]
