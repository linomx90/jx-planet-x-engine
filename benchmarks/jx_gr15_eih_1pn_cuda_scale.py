#!/usr/bin/env python3
"""Measure supported CPU/CUDA GR15-EIH1PN batch throughput and parity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time
from typing import Any

import numpy as np

import jxplanetx
from jxplanetx.gr15 import GR15Spec
from jxplanetx.gr15_eih_1pn import integrate_gr15_eih_1pn
from jxplanetx.gr15_eih_1pn_cuda import (
    CUDA_SOURCE_SHA256,
    gr15_eih_1pn_cuda_runtime_identity,
    integrate_gr15_eih_1pn_cuda_batch,
)
from jxplanetx.solar_system.eih_1pn import EIH1PNParameters


def _parse_counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(item) for item in value.split(","))
    if not counts or any(item < 1 for item in counts):
        raise argparse.ArgumentTypeError("system counts must be positive integers")
    return counts


def _fixture(
    system_count: int, body_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    body_index = np.arange(body_count, dtype=np.float64)
    positions = np.empty((system_count, body_count, 3), dtype=np.float64)
    velocities = np.empty_like(positions)
    gm = np.empty((system_count, body_count), dtype=np.float64)
    base_x = np.linspace(-20.0, 20.0, body_count)
    base_gm = np.linspace(1.0, 0.2, body_count)
    for system in range(system_count):
        phase = (system % 97) * 0.013
        positions[system, :, 0] = base_x + phase
        positions[system, :, 1] = 0.35 * np.sin(body_index + phase)
        positions[system, :, 2] = 0.2 * np.cos(0.7 * body_index - phase)
        velocities[system, :, 0] = 0.003 * np.cos(body_index + phase)
        velocities[system, :, 1] = 0.02 * np.sin(0.3 * body_index - phase)
        velocities[system, :, 2] = 0.004 * np.cos(0.5 * body_index + phase)
        gm[system] = base_gm * (1.0 + (system % 17) * 1.0e-4)
    return (
        np.ascontiguousarray(positions),
        np.ascontiguousarray(velocities),
        np.ascontiguousarray(gm),
    )


def _median_runtime(function: Any, repeats: int) -> tuple[float, Any]:
    samples: list[float] = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter_ns()
        result = function()
        samples.append((time.perf_counter_ns() - started) * 1.0e-9)
    return statistics.median(samples), result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", type=_parse_counts, default=(1, 8, 64, 512))
    parser.add_argument("--bodies", type=int, default=11)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 2 <= args.bodies <= 32:
        parser.error("--bodies must be in [2, 32]")
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    import cupy as cp

    spec = GR15Spec(
        0.0,
        0.2,
        intermediate_epochs=(0.07, 0.13),
        initial_step=0.005,
        maximum_step=0.025,
        epsilon=1.0e-9,
    )
    parameters = EIH1PNParameters(
        speed_of_light_km_s=100.0,
        maximum_compactness=0.1,
        maximum_speed_fraction_squared=0.1,
    )

    warm_position, warm_velocity, warm_gm = _fixture(1, args.bodies)
    integrate_gr15_eih_1pn_cuda_batch(
        cp.asarray(warm_position),
        cp.asarray(warm_velocity),
        cp.asarray(warm_gm),
        spec,
        parameters,
    )

    rows: list[dict[str, Any]] = []
    for system_count in args.systems:
        positions, velocities, gm = _fixture(system_count, args.bodies)
        device_positions = cp.asarray(positions)
        device_velocities = cp.asarray(velocities)
        device_gm = cp.asarray(gm)

        def run_cpu():
            return tuple(
                integrate_gr15_eih_1pn(
                    positions[system],
                    velocities[system],
                    gm[system],
                    spec,
                    parameters,
                )
                for system in range(system_count)
            )

        def run_cuda():
            return integrate_gr15_eih_1pn_cuda_batch(
                device_positions,
                device_velocities,
                device_gm,
                spec,
                parameters,
            )

        cpu_seconds, cpu_results = _median_runtime(run_cpu, args.repeats)
        cuda_seconds, cuda_result = _median_runtime(run_cuda, args.repeats)
        cuda_positions = cp.asnumpy(cuda_result.checkpoint_positions_km)
        cuda_velocities = cp.asnumpy(cuda_result.checkpoint_velocities_km_s)
        maximum_position_difference = 0.0
        maximum_velocity_difference = 0.0
        accounting_mismatch_count = 0
        for system, cpu_result in enumerate(cpu_results):
            maximum_position_difference = max(
                maximum_position_difference,
                float(
                    np.max(
                        np.abs(
                            cuda_positions[system]
                            - cpu_result.checkpoint_positions
                        )
                    )
                ),
            )
            maximum_velocity_difference = max(
                maximum_velocity_difference,
                float(
                    np.max(
                        np.abs(
                            cuda_velocities[system]
                            - cpu_result.checkpoint_velocities
                        )
                    )
                ),
            )
            audit = cuda_result.system_audits[system]
            if (
                audit.attempted_steps != cpu_result.attempted_steps
                or audit.accepted_steps != cpu_result.accepted_steps
                or audit.rejected_steps != cpu_result.rejected_steps
            ):
                accounting_mismatch_count += 1
        rows.append(
            {
                "system_count": system_count,
                "body_count": args.bodies,
                "cpu_sequential_median_seconds": cpu_seconds,
                "cuda_batch_median_seconds": cuda_seconds,
                "cuda_speedup_over_sequential_cpu": cpu_seconds / cuda_seconds,
                "cpu_trajectories_per_second": system_count / cpu_seconds,
                "cuda_trajectories_per_second": system_count / cuda_seconds,
                "maximum_absolute_position_difference_km": (
                    maximum_position_difference
                ),
                "maximum_absolute_velocity_difference_km_s": (
                    maximum_velocity_difference
                ),
                "step_accounting_mismatch_count": accounting_mismatch_count,
            }
        )

    report: dict[str, Any] = {
        "schema": "jx.gr15-eih-1pn-cuda-scale.v1",
        "jx_version": jxplanetx.__version__,
        "comparison_scope": (
            "supported single-thread sequential CPU API versus supported "
            "single-GPU batch API; includes Python result construction and "
            "CUDA host audit, excludes initial device input transfer"
        ),
        "scientific_claim_state": "SCREENING_ONLY",
        "general_superiority_claimed": False,
        "production_ephemeris_claimed": False,
        "cuda_source_sha256": CUDA_SOURCE_SHA256,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cuda": gr15_eih_1pn_cuda_runtime_identity(),
        },
        "configuration": {
            "system_counts": list(args.systems),
            "body_count": args.bodies,
            "repeats": args.repeats,
            "checkpoint_epochs": list(spec.checkpoint_epochs),
            "initial_step": spec.initial_step,
            "maximum_step": spec.maximum_step,
            "epsilon": spec.epsilon,
        },
        "rows": rows,
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["semantic_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
