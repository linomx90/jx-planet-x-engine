#!/usr/bin/env python3
"""Bounded exact-parity benchmark for the experimental fused CUDA WH map.

The fixture is deliberately numerical, not scientific: 16,800 circular
primary-relative tracers and five distant static secondary sources exercise
the full binary64 kick machinery without creating encounter routes.  Each
timed implementation starts from identical device arrays, includes source-tile
transfer and result synchronization, and must return bitwise-identical state
and diagnostics before a speedup is reported.
"""

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
    advance_cuda_restricted_wh,
    advance_cuda_restricted_wh_fused,
)
from jxplanetx.engine import BackendSpec, Provenance


SOLAR_GM = 4.0 * math.pi * math.pi
SOURCE_IDS = ("Sun", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto")
SCHEDULE_SHA256 = hashlib.sha256(
    b"jx-restricted-wh-fused-throughput-fixture-v1"
).hexdigest()


def _provenance(label: str) -> Provenance:
    return Provenance(
        source_id=f"jx-benchmark.{label}",
        citation="Synthetic throughput fixture; no scientific interpretation",
        version="v1",
        sha256=hashlib.sha256(label.encode("ascii")).hexdigest(),
    )


def _tile(step_count: int, step_years: float) -> FrozenSourcePathTile:
    positions = np.zeros((step_count + 1, len(SOURCE_IDS), 3), dtype=np.float64)
    velocities = np.zeros_like(positions)
    # Keep the five positive-GM secondaries safely distant.  This is a stable
    # tidal-force throughput fixture, not a model of the Solar System.
    scale = 1.0e6
    positions[:, 1, :] = (scale, 0.0, 0.0)
    positions[:, 2, :] = (0.0, scale, 0.0)
    positions[:, 3, :] = (-scale, 0.0, 0.0)
    positions[:, 4, :] = (0.0, -scale, 0.0)
    positions[:, 5, :] = (0.0, 0.0, scale)
    gm_ratios = np.asarray(
        (1.0, 9.547919384243266e-4, 2.858859806661308e-4,
         4.366244043351563e-5, 5.151389020466114e-5, 7.350478973e-9),
        dtype=np.float64,
    )
    return FrozenSourcePathTile(
        tile_id=f"throughput-{step_count}",
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
        source_ids=SOURCE_IDS,
        primary_source_id="Sun",
        source_schedule_sha256=SCHEDULE_SHA256,
        source_relative_positions_AU=positions,
        source_relative_velocities_AU_per_year=velocities,
        source_gm_AU3_per_year2=SOLAR_GM * gm_ratios,
        source_encounter_radii_AU=np.full(len(SOURCE_IDS), 0.01, dtype=np.float64),
        certified_relative_acceleration_bounds_AU_per_year2=np.full(
            len(SOURCE_IDS), 1.0, dtype=np.float64
        ),
        provenance=_provenance("source-tile"),
    )


def _state(cp, lane_count: int, step_years: float) -> CUDARestrictedWHState:
    radii = cp.exp(
        cp.linspace(
            math.log(40.0),
            math.log(1000.0),
            lane_count,
            dtype=cp.float64,
        )
    )
    positions = cp.zeros((lane_count, 3), dtype=cp.float64)
    velocities = cp.zeros_like(positions)
    if lane_count:
        positions[:, 0] = radii
        velocities[:, 1] = cp.sqrt(cp.float64(SOLAR_GM) / radii)
    return CUDARestrictedWHState(
        snapshot_id="synthetic-fused-throughput",
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
        source_ids=SOURCE_IDS,
        primary_source_id="Sun",
        source_schedule_sha256=SCHEDULE_SHA256,
        tracer_relative_positions_AU=positions,
        tracer_relative_velocities_AU_per_year=velocities,
        tracer_indices=cp.arange(lane_count, dtype=cp.uint64),
        provenance=_provenance("tracer-state"),
    )


def _require_equal(cp, reference, fused) -> None:
    pairs = (
        (
            reference.state.tracer_relative_positions_AU,
            fused.state.tracer_relative_positions_AU,
            "positions",
        ),
        (
            reference.state.tracer_relative_velocities_AU_per_year,
            fused.state.tracer_relative_velocities_AU_per_year,
            "velocities",
        ),
        (
            reference.state.tracer_indices,
            fused.state.tracer_indices,
            "indices",
        ),
        (
            reference.minimum_endpoint_source_distance_AU,
            fused.minimum_endpoint_source_distance_AU,
            "minimum distances",
        ),
    )
    for expected, actual, label in pairs:
        if not bool(cp.array_equal(expected, actual)):
            raise RuntimeError(f"fused result is not bitwise equal for {label}")
    if reference.maximum_kepler_iterations != fused.maximum_kepler_iterations:
        raise RuntimeError("fused maximum Kepler iteration diagnostic differs")


def _time_call(callable_, state, tile, spec):
    started = time.perf_counter()
    result = callable_(state, tile, spec)
    elapsed = time.perf_counter() - started
    return result, elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lanes", type=int, default=16_800)
    parser.add_argument("--steps", type=int, default=8_000)
    parser.add_argument("--ratio", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--step-years", type=float, default=0.125)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.lanes <= 0 or args.steps <= 0 or args.repeats <= 0 or args.ratio <= 0:
        parser.error("lanes, steps, ratio, and repeats must be positive")
    if args.steps % args.ratio:
        parser.error("steps must be exactly divisible by ratio")

    import cupy as cp

    backend = BackendSpec("cupy", "cuda:0", args.threads)
    map_steps = args.steps // args.ratio
    spec = FixedStepCUDARestrictedWHSpec(
        map_steps,
        args.ratio,
        backend,
        threads_per_block=args.threads,
    )
    tile = _tile(args.steps, args.step_years)
    initial = _state(cp, args.lanes, args.step_years)

    # Compile and warm both paths outside the measured region.
    warm_map_steps = min(4, map_steps)
    warm_source_steps = warm_map_steps * args.ratio
    warm_tile = _tile(warm_source_steps, args.step_years)
    warm_spec = FixedStepCUDARestrictedWHSpec(
        warm_map_steps,
        args.ratio,
        backend,
        threads_per_block=args.threads,
    )
    advance_cuda_restricted_wh(initial, warm_tile, warm_spec)
    advance_cuda_restricted_wh_fused(initial, warm_tile, warm_spec)

    stepwise_seconds: list[float] = []
    fused_seconds: list[float] = []
    last_reference = None
    last_fused = None
    for repeat in range(args.repeats):
        # Alternate order so thermal/order bias does not consistently favor one.
        if repeat % 2 == 0:
            last_reference, reference_elapsed = _time_call(
                advance_cuda_restricted_wh, initial, tile, spec
            )
            last_fused, fused_elapsed = _time_call(
                advance_cuda_restricted_wh_fused, initial, tile, spec
            )
        else:
            last_fused, fused_elapsed = _time_call(
                advance_cuda_restricted_wh_fused, initial, tile, spec
            )
            last_reference, reference_elapsed = _time_call(
                advance_cuda_restricted_wh, initial, tile, spec
            )
        _require_equal(cp, last_reference, last_fused)
        stepwise_seconds.append(reference_elapsed)
        fused_seconds.append(fused_elapsed)

    tracer_steps = args.lanes * map_steps
    fine_lattice_tracer_steps = args.lanes * args.steps
    reference_median = statistics.median(stepwise_seconds)
    fused_median = statistics.median(fused_seconds)
    endpoint_bytes = (
        cp.asnumpy(last_fused.state.tracer_relative_positions_AU).tobytes()
        + cp.asnumpy(
            last_fused.state.tracer_relative_velocities_AU_per_year
        ).tobytes()
    )
    report = {
        "schema": "jx.cuda-restricted-wh.fused-benchmark.v2",
        "classification": "ENGINEERING_THROUGHPUT_ONLY",
        "scientific_fixture": False,
        "precision_gate": "BITWISE_EQUAL_TO_STEPWISE_REFERENCE",
        "precision_gate_passed": True,
        "lanes": args.lanes,
        "source_steps": args.steps,
        "source_steps_per_map_step": args.ratio,
        "map_steps": map_steps,
        "map_tracer_steps": tracer_steps,
        "fine_lattice_tracer_steps": fine_lattice_tracer_steps,
        "simulated_years": args.steps * args.step_years,
        "step_years_hex": float(args.step_years).hex(),
        "source_count": len(SOURCE_IDS),
        "threads_per_block": args.threads,
        "stepwise_seconds": stepwise_seconds,
        "fused_seconds": fused_seconds,
        "stepwise_median_seconds": reference_median,
        "fused_median_seconds": fused_median,
        "stepwise_tracer_steps_per_second": tracer_steps / reference_median,
        "stepwise_fine_lattice_equivalent_steps_per_second": (
            fine_lattice_tracer_steps / reference_median
        ),
        "fused_tracer_steps_per_second": tracer_steps / fused_median,
        "fused_fine_lattice_equivalent_steps_per_second": (
            fine_lattice_tracer_steps / fused_median
        ),
        "fused_speedup": reference_median / fused_median,
        "endpoint_sha256": hashlib.sha256(endpoint_bytes).hexdigest(),
        "maximum_kepler_iterations": last_fused.maximum_kepler_iterations,
        "source_tile_sha256": tile.content_sha256,
        "runtime_identity": dict(last_fused.runtime_identity),
        "claims": {
            "production_qualified": False,
            "long_horizon_accuracy_qualified": False,
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
