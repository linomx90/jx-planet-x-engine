#!/usr/bin/env python3
"""Benchmark-private native adaptive Gauss--Radau order-15 prototype.

This module compiles the independent C11 collocation kernel on demand.  It is
an engineering experiment, is not a supported engine backend, and authorizes
only ``SCREENING_ONLY`` scientific claims.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any

import numpy as np


MAX_BODIES = 32
STAGE_COUNT = 8
SCIENTIFIC_CLAIM_STATE = "SCREENING_ONLY"
METHOD_ID = "JX_NATIVE_GAUSS_RADAU15_PROTOTYPE_V1"

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


class GaussRadau15Error(RuntimeError):
    """The benchmark-private native prototype rejected or failed a request."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _finite_float(value: Any, label: str) -> float:
    if type(value) not in (int, float):
        raise GaussRadau15Error(f"{label} must be a finite real scalar")
    checked = float(value)
    if not np.isfinite(checked):
        raise GaussRadau15Error(f"{label} must be finite")
    return checked


@dataclass(frozen=True)
class GaussRadau15Spec:
    """Validated adaptive-controller and checkpoint schedule configuration."""

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
        scalar_names = (
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
        )
        for name in scalar_names:
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))
        if type(self.intermediate_epochs) is not tuple:
            raise GaussRadau15Error("intermediate_epochs must be an exact tuple")
        intermediate = tuple(
            _finite_float(value, "intermediate epoch")
            for value in self.intermediate_epochs
        )
        object.__setattr__(self, "intermediate_epochs", intermediate)
        if self.initial_epoch == self.final_epoch:
            raise GaussRadau15Error("initial and final epochs must differ")
        direction = 1.0 if self.final_epoch > self.initial_epoch else -1.0
        previous = self.initial_epoch
        for epoch in (*intermediate, self.final_epoch):
            if direction * (epoch - previous) <= 0.0:
                raise GaussRadau15Error(
                    "checkpoint epochs must be strictly monotonic"
                )
            previous = epoch
        if not (
            self.minimum_step > 0.0
            and self.initial_step >= self.minimum_step
            and self.maximum_step >= self.initial_step
        ):
            raise GaussRadau15Error(
                "steps must satisfy 0 < minimum <= initial <= maximum"
            )
        if self.epsilon <= 0.0:
            raise GaussRadau15Error("epsilon must be positive")
        if not 0.0 < self.safety_factor < 1.0:
            raise GaussRadau15Error("safety_factor must lie strictly between 0 and 1")
        if not 0.0 < self.minimum_scale_factor <= 1.0:
            raise GaussRadau15Error(
                "minimum_scale_factor must lie in (0, 1]"
            )
        if self.maximum_scale_factor < 1.0:
            raise GaussRadau15Error("maximum_scale_factor must be at least 1")
        if self.convergence_factor < 1.0:
            raise GaussRadau15Error("convergence_factor must be at least 1")
        if type(self.maximum_iterations) is not int or not (
            1 <= self.maximum_iterations <= 12
        ):
            raise GaussRadau15Error("maximum_iterations must be an integer in [1, 12]")
        if type(self.maximum_steps) is not int or self.maximum_steps < 1:
            raise GaussRadau15Error("maximum_steps must be a positive integer")
        if type(self.maximum_rejections) is not int or self.maximum_rejections < 0:
            raise GaussRadau15Error(
                "maximum_rejections must be a nonnegative integer"
            )

    @property
    def checkpoint_epochs(self) -> tuple[float, ...]:
        return (self.initial_epoch, *self.intermediate_epochs, self.final_epoch)


@dataclass
class GaussRadau15Workspace:
    """Reusable native output storage; returned views remain valid until reuse."""

    body_count: int
    checkpoint_epochs: tuple[float, ...]
    checkpoint_positions: np.ndarray
    checkpoint_velocities: np.ndarray
    checkpoint_accepted_steps: np.ndarray
    checkpoint_rejected_steps: np.ndarray
    counters: np.ndarray
    metrics: np.ndarray


@dataclass(frozen=True)
class GaussRadau15Result:
    """One successful trajectory and its complete adaptive accounting."""

    method_id: str
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
    maximum_error_ratio: float
    minimum_accepted_step: float
    maximum_accepted_step: float
    final_proposed_step: float
    elapsed_seconds: float
    replay_digest: str
    status: int


_BUILD_DIRECTORY: tempfile.TemporaryDirectory[str] | None = None
_LIBRARY: Any | None = None
_SOURCE_SHA256: str | None = None
_COMPILER_VERSION: str | None = None


def _native_function() -> Any:
    global _BUILD_DIRECTORY, _LIBRARY, _SOURCE_SHA256, _COMPILER_VERSION
    if _LIBRARY is not None:
        return _LIBRARY.jx_native_gauss_radau15_integrate
    source = Path(__file__).with_name("jx_native_gauss_radau15.c").resolve()
    raw = source.read_bytes()
    _SOURCE_SHA256 = hashlib.sha256(raw).hexdigest()
    compiler = shutil.which("gcc")
    if compiler is None:
        raise GaussRadau15Error("gcc is unavailable for the native prototype")
    _BUILD_DIRECTORY = tempfile.TemporaryDirectory(prefix="jx-gauss-radau15-")
    output = Path(_BUILD_DIRECTORY.name) / "libjx_native_gauss_radau15.so"
    command = [
        compiler,
        "-O3",
        "-g0",
        "-std=c11",
        "-fPIC",
        "-shared",
        "-fno-fast-math",
        "-ffp-contract=off",
        str(source),
        "-lm",
        "-o",
        str(output),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise GaussRadau15Error(
            f"native Gauss--Radau compilation failed: {completed.stderr.strip()}"
        )
    version = subprocess.run(
        [compiler, "--version"], check=True, capture_output=True, text=True
    )
    _COMPILER_VERSION = version.stdout.splitlines()[0]
    _LIBRARY = ctypes.CDLL(str(output))
    function = _LIBRARY.jx_native_gauss_radau15_integrate
    pointer = ctypes.POINTER(ctypes.c_double)
    integer_pointer = ctypes.POINTER(ctypes.c_int64)
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        pointer,
        pointer,
        pointer,
        pointer,
        *([ctypes.c_double] * 8),
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_int64,
        pointer,
        pointer,
        integer_pointer,
        integer_pointer,
        integer_pointer,
        pointer,
    ]
    function.restype = ctypes.c_int
    return function


def runtime_identity() -> dict[str, str]:
    """Return the exact source and compiler identity of this local prototype."""

    _native_function()
    assert _SOURCE_SHA256 is not None and _COMPILER_VERSION is not None
    return {
        "source_sha256": _SOURCE_SHA256,
        "compiler_version": _COMPILER_VERSION,
        "compiler_flags": (
            "-O3 -g0 -std=c11 -fPIC -shared -fno-fast-math "
            "-ffp-contract=off -lm"
        ),
    }


def tableau_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return read-only copies of the exact coefficients embedded in C."""

    _native_function()
    assert _LIBRARY is not None
    function = _LIBRARY.jx_native_gauss_radau15_tableau
    pointer = ctypes.POINTER(ctypes.c_double)
    function.argtypes = [pointer, pointer, pointer]
    function.restype = ctypes.c_int
    nodes = np.empty(STAGE_COUNT, dtype=np.float64)
    weights = np.empty(STAGE_COUNT, dtype=np.float64)
    matrix = np.empty((STAGE_COUNT, STAGE_COUNT), dtype=np.float64)
    status = int(
        function(
            nodes.ctypes.data_as(pointer),
            weights.ctypes.data_as(pointer),
            matrix.ctypes.data_as(pointer),
        )
    )
    if status != STATUS_SUCCESS:
        raise GaussRadau15Error("native tableau export failed", status=status)
    for value in (nodes, weights, matrix):
        value.setflags(write=False)
    return nodes, weights, matrix


def prepare_workspace(
    body_count: int, spec: GaussRadau15Spec
) -> GaussRadau15Workspace:
    if type(body_count) is not int or not 2 <= body_count <= MAX_BODIES:
        raise GaussRadau15Error("native prototype requires 2--32 bodies")
    if type(spec) is not GaussRadau15Spec:
        raise GaussRadau15Error("spec must be an exact GaussRadau15Spec")
    count = len(spec.checkpoint_epochs)
    shape = (count, body_count, 3)
    return GaussRadau15Workspace(
        body_count=body_count,
        checkpoint_epochs=spec.checkpoint_epochs,
        checkpoint_positions=np.empty(shape, dtype=np.float64),
        checkpoint_velocities=np.empty(shape, dtype=np.float64),
        checkpoint_accepted_steps=np.empty(count, dtype=np.int64),
        checkpoint_rejected_steps=np.empty(count, dtype=np.int64),
        counters=np.empty(6, dtype=np.int64),
        metrics=np.empty(4, dtype=np.float64),
    )


def _state(value: Any, body_count: int, label: str) -> np.ndarray:
    if type(value) is not np.ndarray or value.dtype != np.float64:
        raise GaussRadau15Error(f"{label} must be an exact NumPy float64 array")
    if value.shape != (body_count, 3) or not value.flags.c_contiguous:
        raise GaussRadau15Error(
            f"{label} must be C-contiguous with shape (body_count, 3)"
        )
    if not np.all(np.isfinite(value)):
        raise GaussRadau15Error(f"{label} must be finite")
    return value


def _readonly_view(value: np.ndarray) -> np.ndarray:
    view = value.view()
    view.setflags(write=False)
    return view


def _digest(
    checkpoints: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    counters: np.ndarray,
    metrics: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"jx.native-gauss-radau15.replay.v1\0")
    for value in (checkpoints, positions, velocities, counters, metrics):
        digest.update(np.asarray(value).tobytes(order="C"))
    return digest.hexdigest()


def integrate(
    positions: np.ndarray,
    velocities: np.ndarray,
    gravitational_parameters: np.ndarray,
    spec: GaussRadau15Spec,
    *,
    workspace: GaussRadau15Workspace | None = None,
) -> GaussRadau15Result:
    """Integrate one point-mass system or fail closed with a native status."""

    if type(spec) is not GaussRadau15Spec:
        raise GaussRadau15Error("spec must be an exact GaussRadau15Spec")
    if type(positions) is not np.ndarray or positions.ndim != 2:
        raise GaussRadau15Error("positions must be a two-dimensional NumPy array")
    body_count = positions.shape[0]
    if not 2 <= body_count <= MAX_BODIES:
        raise GaussRadau15Error("native prototype requires 2--32 bodies")
    positions = _state(positions, body_count, "positions")
    velocities = _state(velocities, body_count, "velocities")
    if (
        type(gravitational_parameters) is not np.ndarray
        or gravitational_parameters.dtype != np.float64
        or gravitational_parameters.shape != (body_count,)
        or not gravitational_parameters.flags.c_contiguous
        or not np.all(np.isfinite(gravitational_parameters))
        or np.any(gravitational_parameters <= 0.0)
    ):
        raise GaussRadau15Error(
            "GM must be a positive finite C-contiguous NumPy float64 vector"
        )
    if workspace is None:
        workspace = prepare_workspace(body_count, spec)
    if type(workspace) is not GaussRadau15Workspace:
        raise GaussRadau15Error("workspace must be an exact GaussRadau15Workspace")
    if (
        workspace.body_count != body_count
        or workspace.checkpoint_epochs != spec.checkpoint_epochs
    ):
        raise GaussRadau15Error("workspace does not match the request")

    function = _native_function()
    checkpoints = np.asarray(spec.checkpoint_epochs, dtype=np.float64)
    pointer = ctypes.POINTER(ctypes.c_double)
    integer_pointer = ctypes.POINTER(ctypes.c_int64)
    started = time.perf_counter_ns()
    status = int(
        function(
            ctypes.c_int(body_count),
            ctypes.c_int(len(checkpoints)),
            positions.ctypes.data_as(pointer),
            velocities.ctypes.data_as(pointer),
            gravitational_parameters.ctypes.data_as(pointer),
            checkpoints.ctypes.data_as(pointer),
            ctypes.c_double(spec.initial_step),
            ctypes.c_double(spec.minimum_step),
            ctypes.c_double(spec.maximum_step),
            ctypes.c_double(spec.epsilon),
            ctypes.c_double(spec.safety_factor),
            ctypes.c_double(spec.minimum_scale_factor),
            ctypes.c_double(spec.maximum_scale_factor),
            ctypes.c_double(spec.convergence_factor),
            ctypes.c_int(spec.maximum_iterations),
            ctypes.c_int64(spec.maximum_steps),
            ctypes.c_int64(spec.maximum_rejections),
            workspace.checkpoint_positions.ctypes.data_as(pointer),
            workspace.checkpoint_velocities.ctypes.data_as(pointer),
            workspace.checkpoint_accepted_steps.ctypes.data_as(integer_pointer),
            workspace.checkpoint_rejected_steps.ctypes.data_as(integer_pointer),
            workspace.counters.ctypes.data_as(integer_pointer),
            workspace.metrics.ctypes.data_as(pointer),
        )
    )
    elapsed = (time.perf_counter_ns() - started) * 1.0e-9
    if status != STATUS_SUCCESS:
        name = STATUS_NAMES.get(status, "UNKNOWN")
        raise GaussRadau15Error(
            f"native Gauss--Radau integration failed with status {status} ({name})",
            status=status,
        )

    checkpoint_positions = _readonly_view(workspace.checkpoint_positions)
    checkpoint_velocities = _readonly_view(workspace.checkpoint_velocities)
    replay_digest = _digest(
        checkpoints,
        workspace.checkpoint_positions,
        workspace.checkpoint_velocities,
        workspace.counters,
        workspace.metrics,
    )
    return GaussRadau15Result(
        method_id=METHOD_ID,
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
        attempted_steps=int(workspace.counters[0]),
        accepted_steps=int(workspace.counters[1]),
        rejected_steps=int(workspace.counters[2]),
        force_evaluations=int(workspace.counters[3]),
        predictor_corrector_iterations=int(workspace.counters[4]),
        nonconverged_retries=int(workspace.counters[5]),
        maximum_error_ratio=float(workspace.metrics[0]),
        minimum_accepted_step=float(workspace.metrics[1]),
        maximum_accepted_step=float(workspace.metrics[2]),
        final_proposed_step=float(workspace.metrics[3]),
        elapsed_seconds=elapsed,
        replay_digest=replay_digest,
        status=status,
    )
