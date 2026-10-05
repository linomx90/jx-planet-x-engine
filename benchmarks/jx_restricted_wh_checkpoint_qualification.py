#!/usr/bin/env python3
"""Bounded replay/checkpoint qualification for the restricted-WH prototype."""

from __future__ import annotations

import os


NUMERICAL_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
for _thread_variable in NUMERICAL_THREAD_ENVIRONMENT:
    os.environ[_thread_variable] = "1"

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any

import numpy as np

import jxplanetx.restricted_wh_checkpoint as checkpoint_module
from jxplanetx.cuda_restricted_wh import (
    COORDINATE_SYSTEM,
    CUDARestrictedWHState,
    FixedStepCUDARestrictedWHSpec,
    FrozenSourcePathTile,
)
from jxplanetx.engine import BackendSpec, Provenance
from jxplanetx.restricted_wh_checkpoint import (
    CHECKPOINT_CLAIMS,
    checkpoint_implementation_identity,
    checkpoint_runtime_identity,
    dump_restricted_wh_lifecycle_checkpoint,
    load_latest_restricted_wh_lifecycle_checkpoint,
    restricted_wh_lifecycle_state_sha256,
)
from jxplanetx.restricted_wh_lifecycle import (
    ACTIVE,
    advance_cuda_restricted_wh_lifecycle,
    initialize_restricted_wh_lifecycle,
)


SCHEMA = "jx.restricted-wh.checkpoint-qualification.v1"
PROTOCOL_SCHEMA = "jx.restricted-wh.checkpoint-qualification-protocol.v1"
RUN_ID = "restricted-wh-checkpoint-qualification-v1"
PROTOCOL_DOMAIN = b"jx.restricted-wh.checkpoint-qualification.protocol.v1\0"
RESULT_DOMAIN = b"jx.restricted-wh.checkpoint-qualification.result.v1\0"
SOLAR_GM = 4.0 * math.pi * math.pi
ROUTED_LANES = 256
SEGMENT_STEPS = 2
SEGMENT_COUNT = 2
TOTAL_STEPS = SEGMENT_STEPS * SEGMENT_COUNT
STEP_YEARS = 0.001
ROUTE_WORKER_COUNT = 6
PHYSICAL_CPU_AFFINITY = (0, 1, 2, 3, 4, 5)
SCALING_LANES = 12
SCALING_STEPS = 8
SCALING_REPEATS = 3
MINIMUM_SIX_WORKER_SPEEDUP = 2.0
EXPECTED_ROUTES_PER_SEGMENT = ROUTED_LANES * SEGMENT_STEPS
EXPECTED_ROUTE_COUNT = ROUTED_LANES * TOTAL_STEPS
SCHEDULE_SHA256 = hashlib.sha256(
    b"jx-restricted-wh-checkpoint-qualification-source-v1"
).hexdigest()
CLAIMS = {
    "production_qualified": False,
    "science_result": False,
    "long_horizon_accuracy_validated": False,
    "long_horizon_completion_projected": False,
    "power_loss_durability_validated": False,
    "cross_runtime_determinism_validated": False,
}


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _content_record(value: dict[str, Any], domain: bytes) -> dict[str, Any]:
    if "content_sha256" in value:
        raise ValueError("content hash field must be derived")
    record = dict(value)
    record["content_sha256"] = hashlib.sha256(
        domain + _canonical_json(value)
    ).hexdigest()
    return record


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_create_only(path: Path, record: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        record, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _write_text_create_only(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _git_head(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10.0,
    )
    value = completed.stdout.strip()
    if len(value) != 40:
        raise RuntimeError("git HEAD is not a full commit digest")
    return value


def _provenance(label: str) -> Provenance:
    return Provenance(
        source_id=f"jx-checkpoint-qualification.{label}",
        citation="Synthetic route-heavy checkpoint/replay fixture",
        version="v1",
        sha256=hashlib.sha256(label.encode("ascii")).hexdigest(),
    )


def _source_tile(start: int, steps: int) -> FrozenSourcePathTile:
    zeros = np.zeros((steps + 1, 1, 3), dtype=np.float64)
    return FrozenSourcePathTile(
        tile_id=f"checkpoint-qualification-{start}-{steps}",
        epoch_origin_years=0.0,
        global_source_step=start,
        source_step_years=STEP_YEARS,
        time_scale="TDB",
        frame="ICRF",
        axes="J2000_ECLIPTIC",
        coordinate_system=COORDINATE_SYSTEM,
        length_unit="AU",
        time_unit="year",
        unit_system_id="AU-YEAR",
        source_ids=("Sun",),
        primary_source_id="Sun",
        source_schedule_sha256=SCHEDULE_SHA256,
        source_relative_positions_AU=zeros,
        source_relative_velocities_AU_per_year=zeros.copy(),
        source_gm_AU3_per_year2=np.asarray((SOLAR_GM,), dtype=np.float64),
        source_encounter_radii_AU=np.asarray((0.997,), dtype=np.float64),
        certified_relative_acceleration_bounds_AU_per_year2=np.asarray(
            (100.0,), dtype=np.float64
        ),
        provenance=_provenance("source-tile"),
    )


def _initial_state(cp: Any, lane_count: int = ROUTED_LANES) -> Any:
    positions = cp.zeros((lane_count, 3), dtype=cp.float64)
    velocities = cp.zeros_like(positions)
    positions[:, 0] = cp.float64(1.0)
    velocities[:, 1] = cp.sqrt(cp.float64(SOLAR_GM))
    dynamics = CUDARestrictedWHState(
        snapshot_id="checkpoint-qualification-initial",
        epoch_origin_years=0.0,
        global_source_step=0,
        source_step_years=STEP_YEARS,
        time_scale="TDB",
        frame="ICRF",
        axes="J2000_ECLIPTIC",
        coordinate_system=COORDINATE_SYSTEM,
        length_unit="AU",
        time_unit="year",
        unit_system_id="AU-YEAR",
        source_ids=("Sun",),
        primary_source_id="Sun",
        source_schedule_sha256=SCHEDULE_SHA256,
        tracer_relative_positions_AU=positions,
        tracer_relative_velocities_AU_per_year=velocities,
        tracer_indices=cp.arange(lane_count, dtype=cp.uint64),
        provenance=_provenance("tracer-state"),
    )
    return initialize_restricted_wh_lifecycle(dynamics)


def _spec(steps: int) -> FixedStepCUDARestrictedWHSpec:
    return FixedStepCUDARestrictedWHSpec(
        steps,
        1,
        BackendSpec("cupy", "cuda:0", 128),
    )


def _normalized_route_rows(route_records: tuple[Any, ...]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in route_records:
        row = checkpoint_module._route_record(item)
        row.pop("map_step_index")
        result = row["route_result"]
        result.pop("source_tile_sha256")
        result["runtime_identity"].pop("source_tile_sha256")
        normalized.append(row)
    return normalized


def _route_digest(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        b"jx.restricted-wh.checkpoint-qualification.routes.v1\0"
        + _canonical_json(rows)
    ).hexdigest()


def _timed(cp: Any, operation: Any) -> tuple[Any, float]:
    cp.cuda.get_current_stream().synchronize()
    started = time.perf_counter()
    value = operation()
    cp.cuda.get_current_stream().synchronize()
    return value, time.perf_counter() - started


def _array_equal(cp: Any, left: Any, right: Any) -> bool:
    return bool(cp.array_equal(left, right))


def _pin_physical_cpu_affinity() -> tuple[int, ...]:
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("qualification requires Linux CPU-affinity support")
    available = set(os.sched_getaffinity(0))
    required = set(PHYSICAL_CPU_AFFINITY)
    if not required.issubset(available):
        raise RuntimeError("the six declared physical CPUs are unavailable")
    os.sched_setaffinity(0, required)
    actual = tuple(sorted(os.sched_getaffinity(0)))
    if actual != PHYSICAL_CPU_AFFINITY:
        raise RuntimeError("physical CPU affinity did not bind exactly")
    return actual


def _scaling_screen(cp: Any, backend: BackendSpec) -> dict[str, Any]:
    state = _initial_state(cp, SCALING_LANES)
    tile = _source_tile(0, SCALING_STEPS)
    spec = _spec(SCALING_STEPS)
    rows: dict[str, Any] = {}
    reference_state_sha256: str | None = None
    reference_route_sha256: str | None = None
    for workers in (1, ROUTE_WORKER_COUNT):
        samples: list[float] = []
        state_hashes: list[str] = []
        route_hashes: list[str] = []
        for _repeat in range(SCALING_REPEATS):
            result, elapsed = _timed(
                cp,
                lambda workers=workers: advance_cuda_restricted_wh_lifecycle(
                    state,
                    tile,
                    spec,
                    route_worker_count=workers,
                ),
            )
            if len(result.route_records) != SCALING_LANES * SCALING_STEPS:
                raise RuntimeError("scaling screen route count changed")
            if result.new_removal_records:
                raise RuntimeError("scaling screen unexpectedly removed a lane")
            samples.append(elapsed)
            state_hashes.append(
                restricted_wh_lifecycle_state_sha256(result.state, backend)
            )
            route_hashes.append(
                _route_digest(_normalized_route_rows(result.route_records))
            )
        if len(set(state_hashes)) != 1 or len(set(route_hashes)) != 1:
            raise RuntimeError("scaling screen repeatability gate failed")
        if reference_state_sha256 is None:
            reference_state_sha256 = state_hashes[0]
            reference_route_sha256 = route_hashes[0]
        elif (
            state_hashes[0] != reference_state_sha256
            or route_hashes[0] != reference_route_sha256
        ):
            raise RuntimeError("one-worker and six-worker precision evidence differs")
        median = statistics.median(samples)
        rows[str(workers)] = {
            "samples_seconds": samples,
            "median_seconds": median,
            "mad_seconds": statistics.median(
                abs(value - median) for value in samples
            ),
            "state_sha256": state_hashes[0],
            "route_ledger_sha256": route_hashes[0],
            "route_count": SCALING_LANES * SCALING_STEPS,
        }
    speedup = rows["1"]["median_seconds"] / rows[str(ROUTE_WORKER_COUNT)][
        "median_seconds"
    ]
    return {
        "lanes": SCALING_LANES,
        "steps": SCALING_STEPS,
        "repeats": SCALING_REPEATS,
        "rows": rows,
        "six_worker_speedup": speedup,
        "precision_gate_passed": True,
        "speedup_gate_passed": speedup >= MINIMUM_SIX_WORKER_SPEEDUP,
    }


def _artifact_records(root: Path, excluded: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded or path.is_symlink() or not path.is_file():
            continue
        rows.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return rows


def run(output_root: Path) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    cpu_affinity = _pin_physical_cpu_affinity()
    output_root.mkdir(parents=True, exist_ok=False)
    _fsync_directory(output_root.parent)

    import cupy as cp

    backend = _spec(1).backend
    initial = _initial_state(cp)
    full_tile = _source_tile(0, TOTAL_STEPS)
    segment_tiles = tuple(
        _source_tile(index * SEGMENT_STEPS, SEGMENT_STEPS)
        for index in range(SEGMENT_COUNT)
    )
    genesis_sha256 = restricted_wh_lifecycle_state_sha256(initial, backend)
    implementation_start = _plain(checkpoint_implementation_identity())
    runtime_start = _plain(checkpoint_runtime_identity(backend))
    runner_sha256 = _sha256_file(Path(__file__).resolve())
    protocol = _content_record(
        {
            "schema": PROTOCOL_SCHEMA,
            "run_id": RUN_ID,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "classification": "ENGINEERING_QUALIFICATION_ONLY",
            "profile": {
                "routed_lanes": ROUTED_LANES,
                "segment_steps": SEGMENT_STEPS,
                "segment_count": SEGMENT_COUNT,
                "total_steps": TOTAL_STEPS,
                "step_years_hex": STEP_YEARS.hex(),
                "route_worker_count": ROUTE_WORKER_COUNT,
                "physical_cpu_affinity": list(PHYSICAL_CPU_AFFINITY),
                "numerical_thread_environment": {
                    name: "1" for name in NUMERICAL_THREAD_ENVIRONMENT
                },
                "scaling_lanes": SCALING_LANES,
                "scaling_steps": SCALING_STEPS,
                "scaling_repeats": SCALING_REPEATS,
                "minimum_six_worker_speedup_hex": (
                    MINIMUM_SIX_WORKER_SPEEDUP.hex()
                ),
                "expected_routes_per_segment": EXPECTED_ROUTES_PER_SEGMENT,
                "expected_route_count": EXPECTED_ROUTE_COUNT,
            },
            "genesis_state_sha256": genesis_sha256,
            "source_schedule_sha256": SCHEDULE_SHA256,
            "full_source_tile_sha256": full_tile.content_sha256,
            "segment_source_tile_sha256": [
                tile.content_sha256 for tile in segment_tiles
            ],
            "code_identity": {
                "git_head": _git_head(repository),
                "runner_sha256": runner_sha256,
                "checkpoint_implementation_identity": implementation_start,
            },
            "runtime_identity": runtime_start,
            "required_gates": [
                "all_routes_rejoined",
                "exact_route_count",
                "bitwise_segmented_equals_uninterrupted",
                "normalized_route_ledger_exact",
                "checkpoint_chain_full_replay",
                "checkpoint_external_head_exact",
                "runtime_identity_stable",
                "six_worker_bitwise_and_speedup",
                "checkpoint_artifact_size_bounded",
                "operation_time_bounds_passed",
                "replay_time_bounds_passed",
                "all_claims_false",
            ],
            "claims": dict(CLAIMS),
        },
        PROTOCOL_DOMAIN,
    )
    _write_json_create_only(output_root / "PROTOCOL.json", protocol)

    scaling = _scaling_screen(cp, backend)
    if not scaling["speedup_gate_passed"]:
        raise RuntimeError("six-worker speedup gate failed")

    full_result, full_seconds = _timed(
        cp,
        lambda: advance_cuda_restricted_wh_lifecycle(
            initial,
            full_tile,
            _spec(TOTAL_STEPS),
            route_worker_count=ROUTE_WORKER_COUNT,
        ),
    )

    checkpoints = output_root / "checkpoints"
    checkpoints.mkdir()
    _fsync_directory(output_root)
    state = initial
    previous: str | None = None
    segment_results = []
    checkpoint_records = []
    timings: dict[str, Any] = {
        "uninterrupted_advance_seconds": full_seconds,
        "segments": [],
    }
    for index, tile in enumerate(segment_tiles):
        spec = _spec(SEGMENT_STEPS)
        segment_result, advance_seconds = _timed(
            cp,
            lambda state=state, tile=tile, spec=spec: (
                advance_cuda_restricted_wh_lifecycle(
                    state,
                    tile,
                    spec,
                    route_worker_count=ROUTE_WORKER_COUNT,
                )
            ),
        )
        record, dump_seconds = _timed(
            cp,
            lambda: dump_restricted_wh_lifecycle_checkpoint(
                checkpoints,
                run_id=RUN_ID,
                protocol_sha256=protocol["content_sha256"],
                genesis_state_sha256=genesis_sha256,
                checkpoint_index=index,
                input_state=state,
                result=segment_result,
                source_tile=tile,
                spec=spec,
                route_worker_count=ROUTE_WORKER_COUNT,
                previous_checkpoint_content_sha256=previous,
            ),
        )
        segment_results.append(segment_result)
        checkpoint_records.append(record)
        timings["segments"].append(
            {
                "checkpoint_index": index,
                "advance_seconds": advance_seconds,
                "dump_and_replay_seconds": dump_seconds,
            }
        )
        state = segment_result.state
        previous = record["checkpoint_content_sha256"]

    final_record = checkpoint_records[-1]
    loaded, load_seconds = _timed(
        cp,
        lambda: load_latest_restricted_wh_lifecycle_checkpoint(
            checkpoints,
            run_id=RUN_ID,
            protocol_sha256=protocol["content_sha256"],
            genesis_state_sha256=genesis_sha256,
            expected_checkpoint_index=final_record["checkpoint_index"],
            expected_checkpoint_content_sha256=final_record[
                "checkpoint_content_sha256"
            ],
            backend=backend,
        ),
    )
    timings["final_full_chain_load_and_replay_seconds"] = load_seconds
    loaded_again, second_load_seconds = _timed(
        cp,
        lambda: load_latest_restricted_wh_lifecycle_checkpoint(
            checkpoints,
            run_id=RUN_ID,
            protocol_sha256=protocol["content_sha256"],
            genesis_state_sha256=genesis_sha256,
            expected_checkpoint_index=final_record["checkpoint_index"],
            expected_checkpoint_content_sha256=final_record[
                "checkpoint_content_sha256"
            ],
            backend=backend,
        ),
    )
    timings["second_full_chain_load_and_replay_seconds"] = second_load_seconds

    segmented_rows = [
        row
        for result in segment_results
        for row in _normalized_route_rows(result.route_records)
    ]
    uninterrupted_rows = _normalized_route_rows(full_result.route_records)
    stored_rows = []
    for record in checkpoint_records:
        for row in record["result"]["route_records"]:
            mutable = _plain(row)
            mutable.pop("map_step_index")
            mutable["route_result"].pop("source_tile_sha256")
            mutable["route_result"]["runtime_identity"].pop(
                "source_tile_sha256"
            )
            stored_rows.append(mutable)

    final_state_sha256 = restricted_wh_lifecycle_state_sha256(state, backend)
    uninterrupted_state_sha256 = restricted_wh_lifecycle_state_sha256(
        full_result.state, backend
    )
    loaded_state_sha256 = restricted_wh_lifecycle_state_sha256(
        loaded.state, backend
    )
    second_loaded_state_sha256 = restricted_wh_lifecycle_state_sha256(
        loaded_again.state, backend
    )
    route_counts = [len(result.route_records) for result in segment_results]
    runtime_end = _plain(checkpoint_runtime_identity(backend))
    implementation_end = _plain(checkpoint_implementation_identity())
    namespace_files = [
        path for path in checkpoints.rglob("*") if path.is_file()
    ]
    json_sizes = [path.stat().st_size for path in namespace_files if path.suffix == ".json"]
    npz_sizes = [path.stat().st_size for path in namespace_files if path.suffix == ".npz"]
    checkpoint_bytes = sum(path.stat().st_size for path in namespace_files)
    advance_total = sum(row["advance_seconds"] for row in timings["segments"])
    operation_times = [
        full_seconds,
        load_seconds,
        second_load_seconds,
        *(
            value
            for row in timings["segments"]
            for value in (row["advance_seconds"], row["dump_and_replay_seconds"])
        ),
    ]
    gates = {
        "all_routes_rejoined": (
            all(not row.route_result.events for row in full_result.route_records)
            and all(
                not row.route_result.events
                for result in segment_results
                for row in result.route_records
            )
            and not full_result.new_removal_records
            and all(not result.new_removal_records for result in segment_results)
        ),
        "exact_route_count": (
            len(full_result.route_records) == EXPECTED_ROUTE_COUNT
            and route_counts
            == [EXPECTED_ROUTES_PER_SEGMENT] * SEGMENT_COUNT
        ),
        "all_lanes_active": (
            bool(np.all(state.tracer_lifecycle == ACTIVE))
            and bool(np.all(loaded.state.tracer_lifecycle == ACTIVE))
            and not state.removal_records
            and not loaded.state.removal_records
        ),
        "bitwise_segmented_equals_uninterrupted": (
            final_state_sha256 == uninterrupted_state_sha256
            and _array_equal(
                cp,
                state.dynamics_state.tracer_relative_positions_AU,
                full_result.state.dynamics_state.tracer_relative_positions_AU,
            )
            and _array_equal(
                cp,
                state.dynamics_state.tracer_relative_velocities_AU_per_year,
                full_result.state.dynamics_state.tracer_relative_velocities_AU_per_year,
            )
        ),
        "loaded_endpoint_bitwise_exact": (
            loaded_state_sha256 == final_state_sha256
            and _array_equal(
                cp,
                loaded.state.dynamics_state.tracer_relative_positions_AU,
                state.dynamics_state.tracer_relative_positions_AU,
            )
            and _array_equal(
                cp,
                loaded.state.dynamics_state.tracer_relative_velocities_AU_per_year,
                state.dynamics_state.tracer_relative_velocities_AU_per_year,
            )
        ),
        "repeat_load_bitwise_exact": (
            second_loaded_state_sha256 == loaded_state_sha256
            and loaded_again.checkpoint_content_sha256
            == loaded.checkpoint_content_sha256
            and _array_equal(
                cp,
                loaded_again.state.dynamics_state.tracer_relative_positions_AU,
                loaded.state.dynamics_state.tracer_relative_positions_AU,
            )
            and _array_equal(
                cp,
                loaded_again.state.dynamics_state.tracer_relative_velocities_AU_per_year,
                loaded.state.dynamics_state.tracer_relative_velocities_AU_per_year,
            )
        ),
        "normalized_route_ledger_exact": (
            segmented_rows == uninterrupted_rows == stored_rows
        ),
        "checkpoint_chain_full_replay": (
            loaded.checkpoint_index == SEGMENT_COUNT - 1
            and loaded.checkpoint_content_sha256
            == final_record["checkpoint_content_sha256"]
        ),
        "checkpoint_external_head_exact": (
            loaded.checkpoint_index == final_record["checkpoint_index"]
            and loaded.checkpoint_content_sha256
            == final_record["checkpoint_content_sha256"]
        ),
        "source_boundary_chain_exact": (
            checkpoint_records[0]["source_final_node_sha256"]
            == checkpoint_records[1]["source_initial_node_sha256"]
        ),
        "runtime_identity_stable": runtime_start == runtime_end,
        "implementation_identity_stable": (
            implementation_start == implementation_end
        ),
        "six_worker_bitwise_and_speedup": (
            scaling["precision_gate_passed"]
            and scaling["speedup_gate_passed"]
            and scaling["six_worker_speedup"] >= MINIMUM_SIX_WORKER_SPEEDUP
        ),
        "six_physical_cpu_affinity_exact": (
            cpu_affinity == PHYSICAL_CPU_AFFINITY
            and tuple(runtime_end["cpu_runtime"]["process_cpu_affinity"])
            == PHYSICAL_CPU_AFFINITY
        ),
        "numerical_thread_environment_exact": all(
            os.environ.get(name) == "1"
            and runtime_end["cpu_runtime"]["numerical_thread_environment"][name]
            == "1"
            for name in NUMERICAL_THREAD_ENVIRONMENT
        ),
        "checkpoint_artifact_size_bounded": (
            len(json_sizes) == SEGMENT_COUNT
            and len(npz_sizes) == SEGMENT_COUNT
            and max(json_sizes) <= 18 * 1024 * 1024
            and max(npz_sizes) <= 4 * 1024 * 1024
            and checkpoint_bytes <= 32 * 1024 * 1024
        ),
        "operation_time_bounds_passed": (
            max(operation_times) <= 120.0
            and sum(operation_times) <= 15.0 * 60.0
        ),
        "replay_time_bounds_passed": (
            timings["segments"][0]["dump_and_replay_seconds"]
            <= 1.35 * timings["segments"][0]["advance_seconds"] + 2.0
            and timings["segments"][1]["dump_and_replay_seconds"]
            <= 1.35 * advance_total + 3.0
            and load_seconds <= 1.35 * advance_total + 3.0
            and second_load_seconds <= 1.35 * advance_total + 3.0
        ),
        "all_claims_false": (
            not any(CLAIMS.values())
            and not any(CHECKPOINT_CLAIMS.values())
            and not any(full_result.claims.values())
            and all(
                not any(result.claims.values()) for result in segment_results
            )
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"qualification gate failed: {gates}")

    artifacts = _artifact_records(
        output_root, {"RESULT.json", "SHA256SUMS"}
    )
    result = _content_record(
        {
            "schema": SCHEMA,
            "run_id": RUN_ID,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASSED_ENGINEERING_QUALIFICATION_NOT_SCIENCE",
            "classification": "ENGINEERING_QUALIFICATION_ONLY",
            "protocol_sha256": protocol["content_sha256"],
            "profile": protocol["profile"],
            "timings": timings,
            "worker_scaling": scaling,
            "route_counts_per_segment": route_counts,
            "route_count_total": len(segmented_rows),
            "route_ledger_sha256": _route_digest(segmented_rows),
            "uninterrupted_route_ledger_sha256": _route_digest(
                uninterrupted_rows
            ),
            "final_state_sha256": final_state_sha256,
            "uninterrupted_state_sha256": uninterrupted_state_sha256,
            "loaded_state_sha256": loaded_state_sha256,
            "second_loaded_state_sha256": second_loaded_state_sha256,
            "checkpoint_storage": {
                "json_sizes_bytes": json_sizes,
                "npz_sizes_bytes": npz_sizes,
                "namespace_size_bytes": checkpoint_bytes,
                "bytes_per_route_json_max": max(json_sizes) / EXPECTED_ROUTES_PER_SEGMENT,
            },
            "checkpoint_heads": [
                {
                    "checkpoint_index": row["checkpoint_index"],
                    "checkpoint_content_sha256": row[
                        "checkpoint_content_sha256"
                    ],
                    "array_file_sha256": row["array_file_sha256"],
                    "segment_sha256": row["segment_sha256"],
                }
                for row in checkpoint_records
            ],
            "gates": gates,
            "artifacts": artifacts,
            "runtime_identity": runtime_end,
            "implementation_identity": implementation_end,
            "claims": dict(CLAIMS),
            "limitations": [
                "synthetic one-source route-heavy fixture",
                "same-runtime deterministic replay only",
                "process-crash recovery tested separately; power-loss durability not claimed",
                "no long-horizon accuracy, completion, or science inference",
            ],
        },
        RESULT_DOMAIN,
    )
    _write_json_create_only(output_root / "RESULT.json", result)
    manifest_paths = [
        path
        for path in sorted(output_root.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and path.name != "SHA256SUMS"
    ]
    manifest = "".join(
        f"{_sha256_file(path)}  {path.relative_to(output_root).as_posix()}\n"
        for path in manifest_paths
    )
    _write_text_create_only(output_root / "SHA256SUMS", manifest)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/restricted_wh_checkpoint_qualification_v1"),
    )
    args = parser.parse_args()
    result = run(args.output_root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
