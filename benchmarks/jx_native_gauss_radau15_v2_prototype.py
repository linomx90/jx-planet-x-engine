#!/usr/bin/env python3
"""Benchmark-private cross-step-predicted Gauss--Radau order-15 prototype."""

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

from benchmarks import jx_native_gauss_radau15_prototype as base


MAX_BODIES = base.MAX_BODIES
STAGE_COUNT = base.STAGE_COUNT
SCIENTIFIC_CLAIM_STATE = "SCREENING_ONLY"
METHOD_ID = "JX_NATIVE_GAUSS_RADAU15_CROSS_STEP_PREDICTOR_V2"
GaussRadau15Spec = base.GaussRadau15Spec
GaussRadau15Error = base.GaussRadau15Error


@dataclass
class GaussRadau15V2Workspace:
    """Reusable V2 output storage; returned views remain valid until reuse."""

    body_count: int
    checkpoint_epochs: tuple[float, ...]
    checkpoint_positions: np.ndarray
    checkpoint_velocities: np.ndarray
    checkpoint_accepted_steps: np.ndarray
    checkpoint_rejected_steps: np.ndarray
    counters: np.ndarray
    metrics: np.ndarray


@dataclass(frozen=True)
class GaussRadau15V2Result:
    """One successful V2 trajectory and predictor accounting."""

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


_BUILD_DIRECTORY: tempfile.TemporaryDirectory[str] | None = None
_LIBRARY: Any | None = None
_SOURCE_SHA256: str | None = None
_BASE_SOURCE_SHA256: str | None = None
_COMPILER_VERSION: str | None = None


def _native_function() -> Any:
    global _BUILD_DIRECTORY, _LIBRARY, _SOURCE_SHA256, _BASE_SOURCE_SHA256
    global _COMPILER_VERSION
    if _LIBRARY is not None:
        return _LIBRARY.jx_native_gauss_radau15_v2_integrate
    source = Path(__file__).with_name("jx_native_gauss_radau15_v2.c").resolve()
    base_source = Path(__file__).with_name("jx_native_gauss_radau15.c").resolve()
    _SOURCE_SHA256 = hashlib.sha256(source.read_bytes()).hexdigest()
    _BASE_SOURCE_SHA256 = hashlib.sha256(base_source.read_bytes()).hexdigest()
    compiler = shutil.which("gcc")
    if compiler is None:
        raise GaussRadau15Error("gcc is unavailable for the native V2 prototype")
    _BUILD_DIRECTORY = tempfile.TemporaryDirectory(prefix="jx-gauss-radau15-v2-")
    output = Path(_BUILD_DIRECTORY.name) / "libjx_native_gauss_radau15_v2.so"
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
            f"native Gauss--Radau V2 compilation failed: {completed.stderr.strip()}"
        )
    version = subprocess.run(
        [compiler, "--version"], check=True, capture_output=True, text=True
    )
    _COMPILER_VERSION = version.stdout.splitlines()[0]
    _LIBRARY = ctypes.CDLL(str(output))
    function = _LIBRARY.jx_native_gauss_radau15_v2_integrate
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
    _native_function()
    assert _SOURCE_SHA256 is not None
    assert _BASE_SOURCE_SHA256 is not None
    assert _COMPILER_VERSION is not None
    return {
        "source_sha256": _SOURCE_SHA256,
        "base_v1_source_sha256": _BASE_SOURCE_SHA256,
        "compiler_version": _COMPILER_VERSION,
        "compiler_flags": (
            "-O3 -g0 -std=c11 -fPIC -shared -fno-fast-math "
            "-ffp-contract=off -lm"
        ),
    }


def prepare_workspace(
    body_count: int, spec: GaussRadau15Spec
) -> GaussRadau15V2Workspace:
    if type(body_count) is not int or not 2 <= body_count <= MAX_BODIES:
        raise GaussRadau15Error("native V2 prototype requires 2--32 bodies")
    if type(spec) is not GaussRadau15Spec:
        raise GaussRadau15Error("spec must be an exact GaussRadau15Spec")
    count = len(spec.checkpoint_epochs)
    shape = (count, body_count, 3)
    return GaussRadau15V2Workspace(
        body_count=body_count,
        checkpoint_epochs=spec.checkpoint_epochs,
        checkpoint_positions=np.empty(shape, dtype=np.float64),
        checkpoint_velocities=np.empty(shape, dtype=np.float64),
        checkpoint_accepted_steps=np.empty(count, dtype=np.int64),
        checkpoint_rejected_steps=np.empty(count, dtype=np.int64),
        counters=np.empty(9, dtype=np.int64),
        metrics=np.empty(4, dtype=np.float64),
    )


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
    digest.update(b"jx.native-gauss-radau15-v2.replay.v1\0")
    for value in (checkpoints, positions, velocities, counters, metrics):
        digest.update(np.asarray(value).tobytes(order="C"))
    return digest.hexdigest()


def integrate(
    positions: np.ndarray,
    velocities: np.ndarray,
    gravitational_parameters: np.ndarray,
    spec: GaussRadau15Spec,
    *,
    workspace: GaussRadau15V2Workspace | None = None,
) -> GaussRadau15V2Result:
    """Integrate one point-mass system with the V2 cross-step predictor."""

    if type(spec) is not GaussRadau15Spec:
        raise GaussRadau15Error("spec must be an exact GaussRadau15Spec")
    if type(positions) is not np.ndarray or positions.ndim != 2:
        raise GaussRadau15Error("positions must be a two-dimensional NumPy array")
    body_count = positions.shape[0]
    if not 2 <= body_count <= MAX_BODIES:
        raise GaussRadau15Error("native V2 prototype requires 2--32 bodies")
    positions = base._state(positions, body_count, "positions")
    velocities = base._state(velocities, body_count, "velocities")
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
    if type(workspace) is not GaussRadau15V2Workspace:
        raise GaussRadau15Error(
            "workspace must be an exact GaussRadau15V2Workspace"
        )
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
    if status != base.STATUS_SUCCESS:
        name = base.STATUS_NAMES.get(status, "UNKNOWN")
        raise GaussRadau15Error(
            f"native Gauss--Radau V2 failed with status {status} ({name})",
            status=status,
        )
    checkpoint_positions = _readonly_view(workspace.checkpoint_positions)
    checkpoint_velocities = _readonly_view(workspace.checkpoint_velocities)
    return GaussRadau15V2Result(
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
        polynomial_predictor_trials=int(workspace.counters[6]),
        constant_predictor_trials=int(workspace.counters[7]),
        history_commits=int(workspace.counters[8]),
        maximum_error_ratio=float(workspace.metrics[0]),
        minimum_accepted_step=float(workspace.metrics[1]),
        maximum_accepted_step=float(workspace.metrics[2]),
        final_proposed_step=float(workspace.metrics[3]),
        elapsed_seconds=elapsed,
        replay_digest=_digest(
            checkpoints,
            workspace.checkpoint_positions,
            workspace.checkpoint_velocities,
            workspace.counters,
            workspace.metrics,
        ),
        status=status,
    )
