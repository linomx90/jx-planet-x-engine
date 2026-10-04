#!/usr/bin/env python3
"""RTX engineering acceptance run for the restricted massless-tracer backend.

The generated population and circular giant-planet state are deterministic
synthetic fixtures.  They measure implementation behavior and throughput; they
are not an observed or primordial TNO population and support no Planet Nine,
flyby, rogue-body, or 4.5-Gyr scientific claim.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

from jxplanetx.cuda_massless_tracer import (
    CUDA_OPTIONS,
    IMPLEMENTATION_SOURCE_SHA256,
    KERNEL_SOURCE_SHA256,
    METHOD_ID,
    CUDAMasslessTracerState,
    FixedStepCUDAMasslessTracerSpec,
    advance_cuda_massless_tracers,
    cuda_massless_tracer_runtime_identity,
)
from jxplanetx.engine import BackendSpec, Provenance


SCHEMA = "jxplanetx.cuda-massless-tracer-acceptance.v3"
PROTOCOL_SCHEMA = "jxplanetx.cuda-massless-tracer-acceptance-protocol.v1"
FAILURE_SCHEMA = "jxplanetx.cuda-massless-tracer-acceptance-failure.v1"
RUN_ID = "jx-cuda-massless-tracer-rtx-acceptance-v3"
REPORT_DOMAIN = b"jxplanetx.cuda-massless-tracer-acceptance.v3\x00"
PROTOCOL_DOMAIN = b"jxplanetx.cuda-massless-tracer-acceptance-protocol.v1\x00"
FIXTURE_SEED = 0x4A58475550555452
SOLAR_GM = 4.0 * math.pi * math.pi
EARTH_TO_SUN = 3.0034896149156e-6
SOURCE_SPECS = (
    ("Sun", 1.0, 0.0, 0.0),
    ("Jupiter", 317.828, 5.2038, 0.1),
    ("Saturn", 95.159, 9.5826, 1.7),
    ("Uranus", 14.536, 19.2184, 3.0),
    ("Neptune", 17.147, 30.1104, 4.4),
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256(b"jx.cuda-massless-tracer-acceptance.arrays.v1\x00")
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii") + b"\x00")
        digest.update(str(array.shape).encode("ascii") + b"\x00")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _active_state() -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    names = tuple(row[0] for row in SOURCE_SPECS)
    gm = np.asarray(
        [SOLAR_GM]
        + [SOLAR_GM * EARTH_TO_SUN * row[1] for row in SOURCE_SPECS[1:]],
        dtype=np.float64,
    )
    position = np.zeros((len(names), 3), dtype=np.float64)
    velocity = np.zeros_like(position)
    for index, (_, _, axis, phase) in enumerate(SOURCE_SPECS[1:], start=1):
        angle = phase
        speed = math.sqrt((gm[0] + gm[index]) / axis)
        position[index] = (
            axis * math.cos(angle),
            axis * math.sin(angle),
            0.0,
        )
        velocity[index] = (
            -speed * math.sin(angle),
            speed * math.cos(angle),
            0.0,
        )
    total = float(np.sum(gm))
    position -= np.sum(gm[:, None] * position, axis=0) / total
    velocity -= np.sum(gm[:, None] * velocity, axis=0) / total
    encounter_radii = np.empty(len(names), dtype=np.float64)
    encounter_radii[0] = 5.0
    for index, (_, _, axis, _) in enumerate(SOURCE_SPECS[1:], start=1):
        encounter_radii[index] = 3.0 * axis * (gm[index] / (3.0 * gm[0])) ** (1.0 / 3.0)
    return names, position, velocity, gm, encounter_radii


def _synthetic_population(
    count: int,
    sun_position: np.ndarray,
    sun_velocity: np.ndarray,
    *,
    seed: int = FIXTURE_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(seed))
    coordinate = (np.arange(count, dtype=np.float64) + 0.5) / count
    rng.shuffle(coordinate)
    semimajor_axis = np.exp(math.log(100.0) + coordinate * math.log(10.0))
    perihelion = rng.uniform(31.0, 80.0, count)
    eccentricity = 1.0 - perihelion / semimajor_axis
    inclination = np.radians(rng.uniform(0.0, 40.0, count))
    node = rng.uniform(0.0, 2.0 * math.pi, count)
    argument = rng.uniform(0.0, 2.0 * math.pi, count)
    mean = rng.uniform(0.0, 2.0 * math.pi, count)
    anomaly = mean.copy()
    for _ in range(64):
        correction = (
            anomaly - eccentricity * np.sin(anomaly) - mean
        ) / (1.0 - eccentricity * np.cos(anomaly))
        anomaly -= correction
        if float(np.max(np.abs(correction))) <= 2.0e-14:
            break
    root = np.sqrt(1.0 - eccentricity * eccentricity)
    denominator = 1.0 - eccentricity * np.cos(anomaly)
    mean_motion = np.sqrt(SOLAR_GM / semimajor_axis**3)
    x = semimajor_axis * (np.cos(anomaly) - eccentricity)
    y = semimajor_axis * root * np.sin(anomaly)
    vx = -semimajor_axis * mean_motion * np.sin(anomaly) / denominator
    vy = semimajor_axis * mean_motion * root * np.cos(anomaly) / denominator
    cn, sn = np.cos(node), np.sin(node)
    ca, sa = np.cos(argument), np.sin(argument)
    ci, si = np.cos(inclination), np.sin(inclination)
    r00, r01 = cn * ca - sn * sa * ci, -cn * sa - sn * ca * ci
    r10, r11 = sn * ca + cn * sa * ci, -sn * sa + cn * ca * ci
    r20, r21 = sa * si, ca * si
    position = np.column_stack((r00 * x + r01 * y, r10 * x + r11 * y, r20 * x + r21 * y))
    velocity = np.column_stack((r00 * vx + r01 * vy, r10 * vx + r11 * vy, r20 * vx + r21 * vy))
    position += sun_position
    velocity += sun_velocity
    return np.ascontiguousarray(position), np.ascontiguousarray(velocity)


def _commit_create_only(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError:
        raise FileExistsError(
            f"refusing to overwrite acceptance artifact: {destination}"
        ) from None
    else:
        temporary.unlink()
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(destination.parent, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite acceptance artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _commit_create_only(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite acceptance artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        _commit_create_only(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite acceptance artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        _commit_create_only(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _artifact_identity(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _git_value(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run(
    *,
    tracer_count: int,
    duration_years: float,
    step_years: float,
    run_directory: Path,
) -> dict[str, Any]:
    import cupy as cp

    if tracer_count <= 0:
        raise ValueError("tracer_count must be positive")
    ratio = duration_years / step_years
    step_count = round(ratio)
    if step_count <= 0 or not math.isclose(
        ratio, step_count, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("duration_years/step_years must be a positive integer")
    run_directory.mkdir(parents=True, exist_ok=True)
    if any(run_directory.iterdir()):
        raise FileExistsError(
            f"acceptance directory must be new and empty: {run_directory}"
        )

    backend = BackendSpec("cupy", "cuda:0", 128)
    spec = FixedStepCUDAMasslessTracerSpec(step_years, step_count, backend)
    names, source_position, source_velocity, gm, encounter_radii = _active_state()
    tracer_position, tracer_velocity = _synthetic_population(
        tracer_count, source_position[0], source_velocity[0]
    )
    acceleration_bounds = np.full(len(names), 10.0, dtype=np.float64)
    tracer_indices = np.arange(tracer_count, dtype=np.uint64)
    state_metadata = {
        "snapshot_id": "cuda-massless-tracer-acceptance-t0",
        "epoch": 0.0,
        "global_step": 0,
        "time_scale": "SYNTHETIC_YEAR",
        "frame": "SYNTHETIC_BARYCENTRIC_INERTIAL",
        "origin": "ACTIVE_SOURCE_BARYCENTER",
        "axes": "SYNTHETIC_FIXED_AXES",
        "length_unit": "AU",
        "time_unit": "year",
        "unit_system_id": "AU-YEAR",
        "source_ids": list(names),
        "primary_source_id": "Sun",
    }
    input_array_sha = _array_sha256(
        source_position,
        source_velocity,
        gm,
        encounter_radii,
        acceleration_bounds,
        tracer_position,
        tracer_velocity,
        tracer_indices,
    )
    input_sha = hashlib.sha256(
        b"jx.cuda-massless-tracer-acceptance.input.v2\x00"
        + _canonical(state_metadata)
        + bytes.fromhex(input_array_sha)
    ).hexdigest()
    provenance = Provenance(
        source_id="jx.synthetic.cuda-massless-tracer-acceptance",
        citation="Deterministic engineering fixture; not a scientific population",
        version="v2",
        sha256=input_sha,
    )
    bindings = {
        "runner_sha256": _sha256_file(Path(__file__)),
        "module_sha256": IMPLEMENTATION_SOURCE_SHA256,
        "focused_test_sha256": _sha256_file(
            ROOT / "tests/test_cuda_massless_tracer.py"
        ),
    }
    git_state = {
        "branch": _git_value("branch", "--show-current"),
        "head": _git_value("rev-parse", "HEAD"),
        "main": _git_value("rev-parse", "main"),
        "status_short_before_run": _git_value("status", "--short"),
    }
    test_command = [
        sys.executable,
        "-m",
        "unittest",
        "tests.test_cuda_massless_tracer",
        "-v",
    ]
    protocol: dict[str, Any] = {
        "schema": PROTOCOL_SCHEMA,
        "run_id": RUN_ID,
        "registered_utc": _utc_now(),
        "classification": "MODEL_OUTPUT / SCREENING_ONLY",
        "method_id": METHOD_ID,
        "kernel_source_sha256": KERNEL_SOURCE_SHA256,
        "cuda_options": list(CUDA_OPTIONS),
        "bindings": bindings,
        "git": git_state,
        "state_metadata": state_metadata,
        "input_state_semantic_sha256": input_sha,
        "fixture": {
            "seed": FIXTURE_SEED,
            "source_specs": [list(row) for row in SOURCE_SPECS],
            "population": (
                "deterministic synthetic a=100..1000 AU, q=31..80 AU; "
                "not observed, primordial, Rubin-selected, or survey-debiased"
            ),
            "tracer_count": tracer_count,
        },
        "integration": {
            "duration_years": duration_years,
            "step_years": spec.step_years,
            "step_count": spec.step_count,
            "maximum_encounter_routes_per_tracer": (
                spec.maximum_encounter_routes_per_tracer
            ),
            "encounter_route_method": spec.encounter_route_method,
            "encounter_route_rtol": spec.encounter_route_rtol,
            "encounter_route_atol": spec.encounter_route_atol,
            "encounter_route_maximum_substep_fraction": (
                spec.encounter_route_maximum_substep_fraction
            ),
            "source_path_steps_per_chunk": spec.source_path_steps_per_chunk,
            "threads_per_block": spec.threads_per_block,
            "backend": {
                "backend_id": backend.backend_id,
                "device": backend.device,
                "tile_size": backend.tile_size,
            },
        },
        "focused_test_command": test_command,
        "gates": {
            "focused_tests_pass": True,
            "cuda_runtime_observed": True,
            "all_endpoint_states_finite": True,
            "all_guarded_steps_transactionally_resolved": True,
            "runtime_kernel_and_implementation_bound": True,
            "minimum_tracer_steps_per_second": 1_000_000.0,
        },
        "claims": {
            "four_point_five_gyr_accuracy_qualified": False,
            "real_tno_population_modeled": False,
            "planet_nine_detected_or_excluded": False,
            "flyby_or_rogue_inferred": False,
            "independently_reproduced": False,
        },
    }
    protocol["content_sha256"] = hashlib.sha256(
        PROTOCOL_DOMAIN + _canonical(protocol)
    ).hexdigest()
    protocol_path = run_directory / "PROTOCOL.json"
    initial_path = run_directory / "INITIAL_STATE.npz"
    tests_path = run_directory / "TESTS.log"
    environment_path = run_directory / "ENVIRONMENT.json"
    endpoint_path = run_directory / "ENDPOINT_STATE.npz"
    routes_path = run_directory / "ROUTES.json"
    result_path = run_directory / "RESULT.json"
    checksums_path = run_directory / "SHA256SUMS"
    _atomic_json(protocol_path, protocol)
    _atomic_npz(
        initial_path,
        {
            "source_positions": source_position,
            "source_velocities": source_velocity,
            "source_gm": gm,
            "source_encounter_radii": encounter_radii,
            "relative_acceleration_bounds": acceleration_bounds,
            "tracer_positions": tracer_position,
            "tracer_velocities": tracer_velocity,
            "tracer_indices": tracer_indices,
        },
    )

    test_environment = dict(os.environ)
    python_path = str(ROOT / "src")
    if test_environment.get("PYTHONPATH"):
        python_path += os.pathsep + test_environment["PYTHONPATH"]
    test_environment["PYTHONPATH"] = python_path
    test_started = _utc_now()
    test_clock = time.perf_counter()
    test_process = subprocess.run(
        test_command,
        cwd=ROOT,
        env=test_environment,
        capture_output=True,
        text=True,
    )
    test_elapsed = time.perf_counter() - test_clock
    test_finished = _utc_now()
    test_log = (
        f"run_id: {RUN_ID}\n"
        f"started_utc: {test_started}\n"
        f"finished_utc: {test_finished}\n"
        f"elapsed_seconds: {test_elapsed:.9f}\n"
        f"command_json: {json.dumps(test_command)}\n"
        f"exit_status: {test_process.returncode}\n"
        "--- stdout ---\n"
        f"{test_process.stdout}"
        "--- stderr ---\n"
        f"{test_process.stderr}"
    )
    _atomic_text(tests_path, test_log)
    if test_process.returncode != 0:
        raise RuntimeError("focused CUDA tests failed; see TESTS.log")

    runtime_before = dict(cuda_massless_tracer_runtime_identity(backend))
    environment = {
        "schema": "jxplanetx.cuda-massless-tracer-environment.v1",
        "run_id": RUN_ID,
        "captured_utc": _utc_now(),
        "runtime_identity": runtime_before,
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "focused_test_exit_status": test_process.returncode,
        "focused_test_elapsed_seconds": test_elapsed,
    }
    _atomic_json(environment_path, environment)

    state = CUDAMasslessTracerState(
        snapshot_id=state_metadata["snapshot_id"],
        epoch=state_metadata["epoch"],
        global_step=state_metadata["global_step"],
        time_scale=state_metadata["time_scale"],
        frame=state_metadata["frame"],
        origin=state_metadata["origin"],
        axes=state_metadata["axes"],
        length_unit=state_metadata["length_unit"],
        time_unit=state_metadata["time_unit"],
        unit_system_id=state_metadata["unit_system_id"],
        source_ids=names,
        primary_source_id=state_metadata["primary_source_id"],
        source_positions=cp.asarray(source_position),
        source_velocities=cp.asarray(source_velocity),
        source_gm=cp.asarray(gm),
        source_encounter_radii=cp.asarray(encounter_radii),
        relative_acceleration_bounds=cp.asarray(acceleration_bounds),
        tracer_positions=cp.asarray(tracer_position),
        tracer_velocities=cp.asarray(tracer_velocity),
        tracer_indices=cp.asarray(tracer_indices),
        provenance=provenance,
    )
    warm = advance_cuda_massless_tracers(
        CUDAMasslessTracerState(
            **{
                **state.__dict__,
                "snapshot_id": "cuda-massless-tracer-acceptance-warmup",
                "tracer_positions": state.tracer_positions[:32].copy(),
                "tracer_velocities": state.tracer_velocities[:32].copy(),
                "tracer_indices": state.tracer_indices[:32].copy(),
            }
        ),
        FixedStepCUDAMasslessTracerSpec(step_years, 1, backend),
    )
    del warm
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.runtime.deviceSynchronize()
    started = time.perf_counter()
    result = advance_cuda_massless_tracers(state, spec)
    cp.cuda.runtime.deviceSynchronize()
    elapsed = time.perf_counter() - started

    endpoint_position = cp.asnumpy(result.state.tracer_positions)
    endpoint_velocity = cp.asnumpy(result.state.tracer_velocities)
    source_endpoint_position = cp.asnumpy(result.state.source_positions)
    source_endpoint_velocity = cp.asnumpy(result.state.source_velocities)
    route_counts = cp.asnumpy(result.encounter_route_counts)
    ever_unbound = cp.asnumpy(result.ever_unbound)
    minimum_distance = cp.asnumpy(result.minimum_source_distance_AU)
    endpoint_arrays = {
        "source_positions": source_endpoint_position,
        "source_velocities": source_endpoint_velocity,
        "source_gm": cp.asnumpy(result.state.source_gm),
        "source_encounter_radii": cp.asnumpy(
            result.state.source_encounter_radii
        ),
        "relative_acceleration_bounds": cp.asnumpy(
            result.state.relative_acceleration_bounds
        ),
        "tracer_positions": endpoint_position,
        "tracer_velocities": endpoint_velocity,
        "tracer_indices": cp.asnumpy(result.state.tracer_indices),
        "minimum_source_distance_AU": minimum_distance,
        "ever_unbound": ever_unbound,
        "encounter_route_counts": route_counts,
    }
    _atomic_npz(endpoint_path, endpoint_arrays)
    route_records = [asdict(row) for row in result.encounter_routes]
    routes_document = {
        "schema": "jxplanetx.cuda-massless-tracer-route-ledger.v1",
        "run_id": RUN_ID,
        "count": len(route_records),
        "minimum_semantics": (
            "minimum over GPU lattice nodes and CPU DOP853 derivative-evaluation "
            "stages; not a continuous event distance"
        ),
        "records": route_records,
    }
    _atomic_json(routes_path, routes_document)

    runtime_after = dict(result.runtime_identity)
    finite = bool(all(np.isfinite(value).all() for value in endpoint_arrays.values()))
    tracer_steps = tracer_count * step_count
    reason_counts = Counter(row.reason for row in result.encounter_routes)
    source_counts = Counter(row.source_id for row in result.encounter_routes)
    pre_result_artifacts = {
        path.name: _artifact_identity(path, run_directory)
        for path in (
            protocol_path,
            initial_path,
            tests_path,
            environment_path,
            endpoint_path,
            routes_path,
        )
    }
    gates = {
        "focused_tests_passed": test_process.returncode == 0,
        "cuda_runtime_observed": (
            runtime_after.get("device_uuid") is not None
            and runtime_after.get("cupy_version") is not None
            and runtime_after.get("cuda_local_runtime_version") is not None
            and runtime_after.get("nvrtc_version") is not None
            and runtime_after.get("runtime_binary_sha256", {}).get(
                "cupy_nvrtc_extension"
            )
            is not None
            and runtime_after == runtime_before
        ),
        "all_endpoint_arrays_finite": finite,
        "all_guarded_steps_transactionally_resolved": (
            len(result.encounter_routes) == int(np.sum(route_counts))
        ),
        "runtime_kernel_and_implementation_bound": (
            runtime_after.get("kernel_source_sha256") == KERNEL_SOURCE_SHA256
            and runtime_after.get("implementation_source_sha256")
            == bindings["module_sha256"]
        ),
        "all_required_pre_result_artifacts_present": (
            len(pre_result_artifacts) == 6
            and all(row["size_bytes"] > 0 for row in pre_result_artifacts.values())
        ),
        "minimum_one_million_tracer_steps_per_second": (
            tracer_steps / elapsed >= 1_000_000.0
        ),
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "run_id": RUN_ID,
        "completed_utc": _utc_now(),
        "status": (
            "PASSED_ENGINEERING_ACCEPTANCE_NOT_SCIENCE"
            if all(gates.values())
            else "FAILED_ENGINEERING_ACCEPTANCE"
        ),
        "classification": "MODEL_OUTPUT / SCREENING_ONLY",
        "method_id": METHOD_ID,
        "protocol_content_sha256": protocol["content_sha256"],
        "input_state_semantic_sha256": input_sha,
        "bindings": bindings,
        "git": git_state,
        "design": {
            "source_ids": list(names),
            "tracer_count": tracer_count,
            "duration_years": duration_years,
            "step_years": step_years,
            "step_count": step_count,
            "fixture_seed": FIXTURE_SEED,
        },
        "measurements": {
            "elapsed_seconds": elapsed,
            "tracer_steps": tracer_steps,
            "tracer_steps_per_second": tracer_steps / elapsed,
            "encounter_route_steps": int(np.sum(route_counts)),
            "encounter_route_reason_counts": dict(sorted(reason_counts.items())),
            "encounter_route_source_counts": dict(sorted(source_counts.items())),
            "ever_unbound_at_lattice_node_count": int(
                np.count_nonzero(ever_unbound)
            ),
            "minimum_evaluated_source_distance_AU": float(
                np.min(minimum_distance)
            ),
            "endpoint_state_semantic_sha256": _array_sha256(
                *[endpoint_arrays[key] for key in sorted(endpoint_arrays)]
            ),
            "engine_segment_input_state_sha256": result.input_state_sha256,
        },
        "runtime": runtime_after,
        "artifacts": pre_result_artifacts,
        "gates": {**gates, "all_pass": all(gates.values())},
        "interpretation_limits": [
            "Engineering throughput and replay evidence only.",
            "No 4.5-Gyr accuracy or stability qualification.",
            "Synthetic fixtures are not a real or survey-selected TNO population.",
            "Guard routes are not confirmed encounters, collisions, or events.",
            "Unbound is a primary-relative two-body diagnostic sampled at lattice nodes.",
            "CPU route minima are evaluated-stage samples, not continuous minima.",
        ],
        "claims": protocol["claims"],
    }
    report["content_sha256"] = hashlib.sha256(
        REPORT_DOMAIN + _canonical(report)
    ).hexdigest()
    _atomic_json(result_path, report)
    checksum_files = sorted(
        path for path in run_directory.iterdir() if path.is_file()
    )
    checksum_text = "".join(
        f"{_sha256_file(path)}  {path.name}\n" for path in checksum_files
    )
    _atomic_text(checksums_path, checksum_text)
    for line in checksum_text.splitlines():
        expected, name = line.split("  ", 1)
        if _sha256_file(run_directory / name) != expected:
            raise RuntimeError(f"post-write checksum verification failed for {name}")
    return report


def _write_failure(
    run_directory: Path,
    arguments: argparse.Namespace,
    exception: BaseException,
) -> None:
    if (run_directory / "RESULT.json").exists() or (
        run_directory / "FAILURE.json"
    ).exists():
        return
    run_directory.mkdir(parents=True, exist_ok=True)
    existing = {
        path.name: _artifact_identity(path, run_directory)
        for path in sorted(run_directory.iterdir())
        if path.is_file()
    }
    failure = {
        "schema": FAILURE_SCHEMA,
        "run_id": RUN_ID,
        "failed_utc": _utc_now(),
        "classification": "ENGINEERING_ACCEPTANCE_FAILURE",
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
        "traceback": traceback.format_exc(),
        "arguments": {
            "tracers": arguments.tracers,
            "duration_years": arguments.duration_years,
            "step_years": arguments.step_years,
            "output_directory": str(run_directory),
        },
        "bindings": {
            "runner_sha256": _sha256_file(Path(__file__)),
            "module_sha256": IMPLEMENTATION_SOURCE_SHA256,
            "focused_test_sha256": _sha256_file(
                ROOT / "tests/test_cuda_massless_tracer.py"
            ),
        },
        "completed_artifacts": existing,
        "claims": {
            "engineering_acceptance_passed": False,
            "scientific_qualification_claimed": False,
        },
    }
    _atomic_json(run_directory / "FAILURE.json", failure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracers", type=int, default=16_800)
    parser.add_argument("--duration-years", type=float, default=5_000.0)
    parser.add_argument("--step-years", type=float, default=0.5)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "runs/cuda_massless_tracer_acceptance_v3",
    )
    arguments = parser.parse_args()
    run_directory = arguments.output_directory.resolve()
    try:
        report = run(
            tracer_count=arguments.tracers,
            duration_years=arguments.duration_years,
            step_years=arguments.step_years,
            run_directory=run_directory,
        )
    except BaseException as exception:
        _write_failure(run_directory, arguments, exception)
        print(
            json.dumps(
                {
                    "output_directory": str(run_directory),
                    "status": "FAILED_ENGINEERING_ACCEPTANCE",
                    "exception_type": type(exception).__name__,
                    "exception_message": str(exception),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "output_directory": str(run_directory),
                "status": report["status"],
                "tracer_steps_per_second": report["measurements"][
                    "tracer_steps_per_second"
                ],
                "content_sha256": report["content_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["gates"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
