#!/usr/bin/env python3
"""Deterministic spawned-process scaling screen for guarded route waves."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np

from jxplanetx.cuda_restricted_wh import (
    COORDINATE_SYSTEM,
    CUDARestrictedWHState,
    FixedStepCUDARestrictedWHSpec,
    FrozenSourcePathTile,
)
from jxplanetx.cuda_restricted_wh_hybrid import (
    advance_cuda_restricted_wh_hybrid,
)
from jxplanetx.engine import BackendSpec, Provenance


SOLAR_GM = 4.0 * math.pi * math.pi
SCHEDULE_SHA256 = hashlib.sha256(
    b"jx-restricted-wh-route-wave-benchmark-v1"
).hexdigest()


def _provenance(label: str) -> Provenance:
    return Provenance(
        source_id=f"jx-benchmark.{label}",
        citation="Synthetic guarded-route throughput fixture",
        version="v1",
        sha256=hashlib.sha256(label.encode("ascii")).hexdigest(),
    )


def _tile(step_count: int, step_years: float) -> FrozenSourcePathTile:
    zeros = np.zeros((step_count + 1, 1, 3), dtype=np.float64)
    return FrozenSourcePathTile(
        tile_id="synthetic-route-wave",
        epoch_origin_years=0.0,
        global_source_step=0,
        source_step_years=step_years,
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


def _state(cp, routed_lanes: int, quiet_lanes: int, step_years: float):
    lane_count = routed_lanes + quiet_lanes
    radii = cp.concatenate(
        (
            cp.ones(routed_lanes, dtype=cp.float64),
            cp.full(quiet_lanes, 1.5, dtype=cp.float64),
        )
    )
    positions = cp.zeros((lane_count, 3), dtype=cp.float64)
    velocities = cp.zeros_like(positions)
    positions[:, 0] = radii
    velocities[:, 1] = cp.sqrt(cp.float64(SOLAR_GM) / radii)
    return CUDARestrictedWHState(
        snapshot_id="synthetic-route-wave",
        epoch_origin_years=0.0,
        global_source_step=0,
        source_step_years=step_years,
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


def _run(state, tile, spec, workers):
    started = time.perf_counter()
    result = advance_cuda_restricted_wh_hybrid(
        state,
        tile,
        spec,
        route_worker_count=workers,
    )
    return result, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routed-lanes", type=int, default=12)
    parser.add_argument("--quiet-lanes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--step-years", type=float, default=0.001)
    parser.add_argument("--workers", default="1,6,12")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    worker_counts = tuple(int(value) for value in args.workers.split(","))
    if (
        args.routed_lanes <= 0
        or args.quiet_lanes < 0
        or args.steps <= 0
        or args.repeats <= 0
        or not worker_counts
        or any(not 1 <= value <= 32 for value in worker_counts)
        or len(set(worker_counts)) != len(worker_counts)
    ):
        parser.error("route-wave benchmark arguments are outside their domain")

    import cupy as cp

    tile = _tile(args.steps, args.step_years)
    state = _state(cp, args.routed_lanes, args.quiet_lanes, args.step_years)
    spec = FixedStepCUDARestrictedWHSpec(
        args.steps,
        1,
        BackendSpec("cupy", "cuda:0", 128),
    )

    rows = {}
    reference = None
    reference_endpoint = None
    for workers in worker_counts:
        samples = []
        last = None
        for _repeat in range(args.repeats):
            last, elapsed = _run(state, tile, spec, workers)
            samples.append(elapsed)
        assert last is not None
        expected_routes = args.routed_lanes * args.steps
        if len(last.route_records) != expected_routes:
            raise RuntimeError("route count differs from the declared fixture")
        endpoint = (
            cp.asnumpy(last.state.tracer_relative_positions_AU).tobytes()
            + cp.asnumpy(
                last.state.tracer_relative_velocities_AU_per_year
            ).tobytes()
        )
        endpoint_sha256 = hashlib.sha256(endpoint).hexdigest()
        if reference is None:
            reference = last
            reference_endpoint = endpoint_sha256
        else:
            for left, right, label in (
                (
                    reference.state.tracer_relative_positions_AU,
                    last.state.tracer_relative_positions_AU,
                    "positions",
                ),
                (
                    reference.state.tracer_relative_velocities_AU_per_year,
                    last.state.tracer_relative_velocities_AU_per_year,
                    "velocities",
                ),
                (
                    reference.minimum_source_distance_AU,
                    last.minimum_source_distance_AU,
                    "minimum distances",
                ),
            ):
                if not bool(cp.array_equal(left, right)):
                    raise RuntimeError(f"worker-count result differs for {label}")
            if endpoint_sha256 != reference_endpoint:
                raise RuntimeError("worker-count endpoint hash changed")
        rows[str(workers)] = {
            "samples_seconds": samples,
            "median_seconds": statistics.median(samples),
            "route_count": len(last.route_records),
            "endpoint_sha256": endpoint_sha256,
        }

    serial_median = rows["1"]["median_seconds"] if "1" in rows else None
    if serial_median is not None:
        for row in rows.values():
            row["speedup_vs_one_worker"] = serial_median / row["median_seconds"]
    report = {
        "schema": "jx.cuda-restricted-wh.route-wave-benchmark.v1",
        "classification": "ENGINEERING_THROUGHPUT_ONLY",
        "precision_gate": "BITWISE_EQUAL_ACROSS_FIXED_WORKER_COUNTS",
        "precision_gate_passed": True,
        "routed_lanes": args.routed_lanes,
        "quiet_lanes": args.quiet_lanes,
        "steps": args.steps,
        "expected_route_count": args.routed_lanes * args.steps,
        "step_years_hex": args.step_years.hex(),
        "worker_counts": worker_counts,
        "repeats": args.repeats,
        "rows": rows,
        "source_tile_sha256": tile.content_sha256,
        "claims": {
            "production_qualified": False,
            "real_encounter_rate_represented": False,
            "long_horizon_completion_projected": False,
            "science_result": False,
        },
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
