#!/usr/bin/env python3
"""Refine the massive-source KDK lattice under a fixed tracer map step.

All tracer runs use a 0.125-year restricted-WH map.  Only the source KDK path
changes from 0.125 down to 0.0078125 year, so endpoint differences isolate
source-path discretization within this same-model engineering experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import jx_cuda_restricted_wh_refinement as base
from jxplanetx.cuda_restricted_wh import (
    COORDINATE_SYSTEM,
    CUDARestrictedWHState,
    FixedStepCUDARestrictedWHSpec,
    advance_cuda_restricted_wh_fused,
)
from jxplanetx.engine import BackendSpec, Provenance
from jxplanetx.engine.backends import resolve_backend


MAP_STEP_YEARS = 0.125
SOURCE_STEP_YEARS = (0.125, 0.0625, 0.03125, 0.015625, 0.0078125)


def _state(
    cp,
    *,
    metadata: dict,
    schedule_sha256: str,
    source_step_years: float,
    positions: np.ndarray,
    velocities: np.ndarray,
    indices: np.ndarray,
    provenance: Provenance,
) -> CUDARestrictedWHState:
    return CUDARestrictedWHState(
        snapshot_id=metadata["snapshot_id"],
        epoch_origin_years=0.0,
        global_source_step=0,
        source_step_years=source_step_years,
        time_scale=metadata["time_scale"],
        frame=metadata["frame"],
        axes="J2000_ECLIPTIC",
        coordinate_system=COORDINATE_SYSTEM,
        length_unit="AU",
        time_unit="year",
        unit_system_id=metadata["unit_system_id"],
        source_ids=tuple(metadata["source_ids"]),
        primary_source_id=metadata["primary_source_id"],
        source_schedule_sha256=schedule_sha256,
        tracer_relative_positions_AU=cp.asarray(positions),
        tracer_relative_velocities_AU_per_year=cp.asarray(velocities),
        tracer_indices=cp.asarray(indices),
        provenance=provenance,
    )


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0.0:
        return None
    return float(numerator / denominator)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=float, default=1_000.0)
    parser.add_argument("--lanes", type=int, default=2_048)
    parser.add_argument("--minimum-initial-perihelion", type=float, default=60.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.years <= 0.0 or args.lanes <= 0:
        parser.error("years and lanes must be positive")
    map_steps_float = args.years / MAP_STEP_YEARS
    map_steps = int(round(map_steps_float))
    if map_steps != map_steps_float:
        parser.error("years must be an exact multiple of the map step")
    for source_step in SOURCE_STEP_YEARS:
        source_count = args.years / source_step
        if int(round(source_count)) != source_count:
            parser.error("years must be an exact multiple of every source step")

    import cupy as cp

    root = Path(__file__).resolve().parents[1]
    (
        tracer_position,
        tracer_velocity,
        tracer_indices,
        source_position,
        source_velocity,
        source_gm,
        source_radii,
        acceleration_bounds,
        metadata,
        schedule_sha256,
        input_hashes,
    ) = base._load_inputs(root)
    initial_elements = base._osculating(
        tracer_position, tracer_velocity, float(source_gm[0])
    )
    eligible = np.flatnonzero(
        initial_elements["bound"]
        & (
            initial_elements["perihelion_AU"]
            >= args.minimum_initial_perihelion
        )
    )
    if eligible.size < args.lanes:
        raise RuntimeError(
            f"only {eligible.size} tracers satisfy the outcome-blind selection"
        )
    selected = eligible[: args.lanes]
    selected_position = np.ascontiguousarray(tracer_position[selected])
    selected_velocity = np.ascontiguousarray(tracer_velocity[selected])
    selected_indices = np.ascontiguousarray(tracer_indices[selected])

    backend_spec = BackendSpec("cupy", "cuda:0", 128)
    resolved = resolve_backend(backend_spec)
    provenance = Provenance(
        source_id="jx.de441-control-source-step-refinement",
        citation="Tracked JX qualification inputs; engineering refinement only",
        version="v1",
        sha256=input_hashes["SOURCE_STATE.npz"],
    )
    records = {}
    with resolved.activate():
        for source_step in SOURCE_STEP_YEARS:
            source_steps = int(round(args.years / source_step))
            started = time.perf_counter()
            tile = base._build_tile(
                cp,
                resolved,
                source_position,
                source_velocity,
                source_gm,
                source_radii,
                acceleration_bounds,
                metadata,
                schedule_sha256,
                source_steps,
                source_step,
                provenance,
            )
            tile_seconds = time.perf_counter() - started
            ratio = int(round(MAP_STEP_YEARS / source_step))
            if ratio * source_step != MAP_STEP_YEARS:
                raise RuntimeError("map/source step ratio is not exact")
            state = _state(
                cp,
                metadata=metadata,
                schedule_sha256=schedule_sha256,
                source_step_years=source_step,
                positions=selected_position,
                velocities=selected_velocity,
                indices=selected_indices,
                provenance=provenance,
            )
            started = time.perf_counter()
            result = advance_cuda_restricted_wh_fused(
                state,
                tile,
                FixedStepCUDARestrictedWHSpec(
                    map_steps,
                    ratio,
                    backend_spec,
                    threads_per_block=128,
                ),
            )
            execution_seconds = time.perf_counter() - started
            records[source_step] = {
                "tile": tile,
                "result": result,
                "positions": cp.asnumpy(
                    result.state.tracer_relative_positions_AU
                ),
                "velocities": cp.asnumpy(
                    result.state.tracer_relative_velocities_AU_per_year
                ),
                "tile_seconds": tile_seconds,
                "execution_seconds": execution_seconds,
                "ratio": ratio,
            }

        finest_step = SOURCE_STEP_YEARS[-1]
        finest = records[finest_step]
        repeat_state = _state(
            cp,
            metadata=metadata,
            schedule_sha256=schedule_sha256,
            source_step_years=finest_step,
            positions=selected_position,
            velocities=selected_velocity,
            indices=selected_indices,
            provenance=provenance,
        )
        repeat = advance_cuda_restricted_wh_fused(
            repeat_state,
            finest["tile"],
            FixedStepCUDARestrictedWHSpec(
                map_steps,
                finest["ratio"],
                backend_spec,
                threads_per_block=128,
            ),
        )
        if not bool(
            cp.array_equal(
                repeat.state.tracer_relative_positions_AU,
                cp.asarray(finest["positions"]),
            )
        ) or not bool(
            cp.array_equal(
                repeat.state.tracer_relative_velocities_AU_per_year,
                cp.asarray(finest["velocities"]),
            )
        ):
            raise RuntimeError("finest source-step refinement did not repeat bitwise")

    finest = records[SOURCE_STEP_YEARS[-1]]
    finest_elements = base._osculating(
        finest["positions"], finest["velocities"], float(source_gm[0])
    )
    comparisons = {}
    for source_step in SOURCE_STEP_YEARS[:-1]:
        record = records[source_step]
        elements = base._osculating(
            record["positions"], record["velocities"], float(source_gm[0])
        )
        source_position_error = np.linalg.norm(
            record["tile"].source_relative_positions_AU[-1, 1:, :]
            - finest["tile"].source_relative_positions_AU[-1, 1:, :],
            axis=1,
        )
        source_velocity_error = np.linalg.norm(
            record["tile"].source_relative_velocities_AU_per_year[-1, 1:, :]
            - finest["tile"].source_relative_velocities_AU_per_year[-1, 1:, :],
            axis=1,
        )
        comparisons[source_step.hex()] = {
            "source_step_years": source_step,
            "map_source_ratio": record["ratio"],
            "maximum_secondary_source_position_error_AU_vs_finest": float(
                np.max(source_position_error)
            ),
            "maximum_secondary_source_velocity_error_AU_per_year_vs_finest": float(
                np.max(source_velocity_error)
            ),
            "tracer_position_error_AU_vs_finest": base._quantiles(
                np.linalg.norm(
                    record["positions"] - finest["positions"], axis=1
                )
            ),
            "tracer_velocity_error_AU_per_year_vs_finest": base._quantiles(
                np.linalg.norm(
                    record["velocities"] - finest["velocities"], axis=1
                )
            ),
            "absolute_perihelion_delta_AU_vs_finest": base._quantiles(
                np.abs(
                    elements["perihelion_AU"]
                    - finest_elements["perihelion_AU"]
                )
            ),
            "absolute_semimajor_delta_AU_vs_finest": base._quantiles(
                np.abs(
                    elements["semimajor_AU"]
                    - finest_elements["semimajor_AU"]
                )
            ),
        }

    # For a second-order method, h=4*h_ref versus h=2*h_ref, both compared
    # with h_ref, has the exact asymptotic error ratio (16-1)/(4-1) = 5.
    coarse = comparisons[SOURCE_STEP_YEARS[-3].hex()]
    medium = comparisons[SOURCE_STEP_YEARS[-2].hex()]
    observed_refinement_ratios = {
        "maximum_source_position": _safe_ratio(
            coarse["maximum_secondary_source_position_error_AU_vs_finest"],
            medium["maximum_secondary_source_position_error_AU_vs_finest"],
        ),
        "maximum_source_velocity": _safe_ratio(
            coarse["maximum_secondary_source_velocity_error_AU_per_year_vs_finest"],
            medium["maximum_secondary_source_velocity_error_AU_per_year_vs_finest"],
        ),
        "median_tracer_position": _safe_ratio(
            coarse["tracer_position_error_AU_vs_finest"]["median"],
            medium["tracer_position_error_AU_vs_finest"]["median"],
        ),
        "q95_absolute_perihelion": _safe_ratio(
            coarse["absolute_perihelion_delta_AU_vs_finest"]["q95"],
            medium["absolute_perihelion_delta_AU_vs_finest"]["q95"],
        ),
    }

    endpoint_bytes = finest["positions"].tobytes() + finest["velocities"].tobytes()
    report = {
        "schema": "jx.cuda-restricted-wh.source-refinement.v1",
        "classification": "ENGINEERING_REFINEMENT_ONLY",
        "independent_reference": False,
        "years": args.years,
        "map_step_years": MAP_STEP_YEARS,
        "source_step_years": SOURCE_STEP_YEARS,
        "lane_count": args.lanes,
        "eligible_lane_count": int(eligible.size),
        "selection_rule": (
            "first tracer IDs in frozen order with initial primary-relative "
            f"two-body osculating perihelion >= {args.minimum_initial_perihelion} AU"
        ),
        "selection_sha256": hashlib.sha256(selected_indices.tobytes()).hexdigest(),
        "input_hashes": input_hashes,
        "tile_sha256_by_source_step": {
            step.hex(): records[step]["tile"].content_sha256
            for step in SOURCE_STEP_YEARS
        },
        "tile_build_seconds_by_source_step": {
            step.hex(): records[step]["tile_seconds"]
            for step in SOURCE_STEP_YEARS
        },
        "execution_seconds_by_source_step": {
            step.hex(): records[step]["execution_seconds"]
            for step in SOURCE_STEP_YEARS
        },
        "comparisons_to_finest_source_step": comparisons,
        "expected_second_order_three-level_error_ratio": 5.0,
        "observed_refinement_ratios": observed_refinement_ratios,
        "finest_endpoint_sha256": hashlib.sha256(endpoint_bytes).hexdigest(),
        "same_runtime_finest_repeat_bitwise": True,
        "runtime_identity": dict(finest["result"].runtime_identity),
        "claims": {
            "production_qualified": False,
            "source_timestep_qualified": False,
            "independent_accuracy_validated": False,
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
