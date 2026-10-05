#!/usr/bin/env python3
"""Bounded quiet-overhead benchmark for the guarded M1 scheduler.

The fixture is the same non-scientific distant-source workload used by the
fused-kernel benchmark.  The hybrid result must be bitwise identical to the
quiet fused result and contain no route rows before timing is reported.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import statistics
import time

from jx_cuda_restricted_wh_fused_benchmark import _state, _tile

from jxplanetx.cuda_restricted_wh import (
    FixedStepCUDARestrictedWHSpec,
    advance_cuda_restricted_wh_fused,
)
from jxplanetx.cuda_restricted_wh_hybrid import (
    advance_cuda_restricted_wh_hybrid,
)
from jxplanetx.engine import BackendSpec


def _plain(value):
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _elapsed(callable_, state, tile, spec):
    started = time.perf_counter()
    result = callable_(state, tile, spec)
    return result, time.perf_counter() - started


def _require_equal(cp, fused, hybrid) -> None:
    if hybrid.route_records:
        raise RuntimeError("quiet fixture unexpectedly produced route records")
    for left, right, label in (
        (
            fused.state.tracer_relative_positions_AU,
            hybrid.state.tracer_relative_positions_AU,
            "positions",
        ),
        (
            fused.state.tracer_relative_velocities_AU_per_year,
            hybrid.state.tracer_relative_velocities_AU_per_year,
            "velocities",
        ),
        (fused.state.tracer_indices, hybrid.state.tracer_indices, "indices"),
        (
            fused.minimum_endpoint_source_distance_AU,
            hybrid.minimum_source_distance_AU,
            "minimum distances",
        ),
    ):
        if not bool(cp.array_equal(left, right)):
            raise RuntimeError(f"hybrid quiet result differs for {label}")
    if fused.maximum_kepler_iterations != hybrid.maximum_kepler_iterations:
        raise RuntimeError("hybrid maximum Kepler iterations differ")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lanes", type=int, default=16_800)
    parser.add_argument("--steps", type=int, default=8_000)
    parser.add_argument("--step-years", type=float, default=0.125)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.lanes <= 0 or args.steps <= 0 or args.repeats <= 0:
        parser.error("lanes, steps, and repeats must be positive")

    import cupy as cp

    backend = BackendSpec("cupy", "cuda:0", args.threads)
    spec = FixedStepCUDARestrictedWHSpec(
        args.steps,
        1,
        backend,
        threads_per_block=args.threads,
    )
    tile = _tile(args.steps, args.step_years)
    state = _state(cp, args.lanes, args.step_years)
    warm_tile = _tile(min(4, args.steps), args.step_years)
    warm_spec = FixedStepCUDARestrictedWHSpec(
        min(4, args.steps),
        1,
        backend,
        threads_per_block=args.threads,
    )
    advance_cuda_restricted_wh_fused(state, warm_tile, warm_spec)
    advance_cuda_restricted_wh_hybrid(state, warm_tile, warm_spec)

    fused_seconds: list[float] = []
    hybrid_seconds: list[float] = []
    fused = None
    hybrid = None
    for repeat in range(args.repeats):
        if repeat % 2:
            hybrid, hybrid_elapsed = _elapsed(
                advance_cuda_restricted_wh_hybrid, state, tile, spec
            )
            fused, fused_elapsed = _elapsed(
                advance_cuda_restricted_wh_fused, state, tile, spec
            )
        else:
            fused, fused_elapsed = _elapsed(
                advance_cuda_restricted_wh_fused, state, tile, spec
            )
            hybrid, hybrid_elapsed = _elapsed(
                advance_cuda_restricted_wh_hybrid, state, tile, spec
            )
        _require_equal(cp, fused, hybrid)
        fused_seconds.append(fused_elapsed)
        hybrid_seconds.append(hybrid_elapsed)

    assert fused is not None and hybrid is not None
    fused_median = statistics.median(fused_seconds)
    hybrid_median = statistics.median(hybrid_seconds)
    endpoint = (
        cp.asnumpy(hybrid.state.tracer_relative_positions_AU).tobytes()
        + cp.asnumpy(
            hybrid.state.tracer_relative_velocities_AU_per_year
        ).tobytes()
    )
    tracer_steps = args.lanes * args.steps
    report = {
        "schema": "jx.cuda-restricted-wh.hybrid-quiet-benchmark.v1",
        "classification": "ENGINEERING_THROUGHPUT_ONLY",
        "scientific_fixture": False,
        "precision_gate": "BITWISE_EQUAL_TO_FUSED_CUDA_AND_ZERO_ROUTES",
        "precision_gate_passed": True,
        "lanes": args.lanes,
        "steps": args.steps,
        "simulated_years": args.steps * args.step_years,
        "step_years_hex": args.step_years.hex(),
        "threads_per_block": args.threads,
        "repeats": args.repeats,
        "fused_seconds": fused_seconds,
        "hybrid_seconds": hybrid_seconds,
        "fused_median_seconds": fused_median,
        "hybrid_median_seconds": hybrid_median,
        "hybrid_overhead_fraction": hybrid_median / fused_median - 1.0,
        "hybrid_tracer_steps_per_second": tracer_steps / hybrid_median,
        "endpoint_sha256": hashlib.sha256(endpoint).hexdigest(),
        "source_tile_sha256": tile.content_sha256,
        "route_count": len(hybrid.route_records),
        "runtime_identity": _plain(hybrid.runtime_identity),
        "claims": {
            "production_qualified": False,
            "encounter_throughput_qualified": False,
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
