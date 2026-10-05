#!/usr/bin/env python3
"""End-to-end fixed-step refinement for the JX long-horizon prototype.

Each lane advances both the six-body source system and the same outcome-blind
tracer subset at one common step.  Candidate ordered-Jacobi source paths are
compared with a GR15 source path, while the finest 0.03125-year restricted CUDA
map is the tracer reference.  This is an engineering refinement, not a science
or production qualification result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import jx_cuda_restricted_wh_refinement as base
import jx_cuda_restricted_wh_source_provider_refinement as provider_benchmark
import jx_source_integrator_refinement as source_refinement
from jxplanetx.cuda_restricted_wh import (
    CUDARestrictedWHState,
    FixedStepCUDARestrictedWHSpec,
    advance_cuda_restricted_wh_fused,
)
from jxplanetx.engine import BackendSpec, Provenance
from jxplanetx.engine.backends import resolve_backend
from jxplanetx.restricted_wh_source import (
    WisdomHolmanSourceTileSpec,
    build_wisdom_holman_source_tile,
)


STEPS_YEARS = (0.25, 0.125, 0.0625, 0.03125)
REFERENCE_STEP_YEARS = STEPS_YEARS[-1]


def _state(
    cp,
    *,
    step: float,
    metadata: dict,
    schedule_sha256: str,
    positions: np.ndarray,
    velocities: np.ndarray,
    indices: np.ndarray,
    provenance: Provenance,
) -> CUDARestrictedWHState:
    return CUDARestrictedWHState(
        snapshot_id=metadata["snapshot_id"],
        epoch_origin_years=0.0,
        global_source_step=0,
        source_step_years=step,
        time_scale=metadata["time_scale"],
        frame=metadata["frame"],
        axes="J2000_ECLIPTIC",
        coordinate_system="PRIMARY_RELATIVE_NONROTATING",
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
    if denominator == 0.0 or not np.isfinite(numerator / denominator):
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
    for step in STEPS_YEARS:
        count = args.years / step
        if int(round(count)) != count:
            parser.error("years must be an exact multiple of every step")

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
    provenance = Provenance(
        "jx.de441-control-end-to-end-refinement",
        "Tracked JX qualification inputs; engineering refinement only",
        "v1",
        input_hashes["SOURCE_STATE.npz"],
    )

    bary_position, bary_velocity = source_refinement._barycentric(
        source_position, source_velocity, source_gm
    )
    reference_steps = int(round(args.years / REFERENCE_STEP_YEARS))
    reference_checkpoints = tuple(
        REFERENCE_STEP_YEARS * index
        for index in range(1, reference_steps)
    )
    from jxplanetx._gr15_v2_reference import GR15Spec, integrate_gr15

    gr15_started = time.perf_counter()
    gr15 = integrate_gr15(
        bary_position,
        bary_velocity,
        source_gm,
        GR15Spec(
            0.0,
            args.years,
            intermediate_epochs=reference_checkpoints,
            initial_step=0.01,
            minimum_step=1.0e-15,
            maximum_step=0.25,
            epsilon=source_refinement.GR15_EPSILONS[-1],
            maximum_steps=2_000_000,
            maximum_rejections=200_000,
        ),
    )
    gr15_seconds = time.perf_counter() - gr15_started
    reference_tile = provider_benchmark._reference_tile(
        source_step_years=REFERENCE_STEP_YEARS,
        metadata=metadata,
        schedule_sha256=schedule_sha256,
        source_gm=source_gm,
        source_radii=source_radii,
        acceleration_bounds=acceleration_bounds,
        positions=gr15.checkpoint_positions,
        velocities=gr15.checkpoint_velocities,
        provenance=provenance,
    )

    backend_spec = BackendSpec("cupy", "cuda:0", 128)
    resolved = resolve_backend(backend_spec)
    import cupy as cp

    with resolved.activate():
        reference_state = _state(
            cp,
            step=REFERENCE_STEP_YEARS,
            metadata=metadata,
            schedule_sha256=schedule_sha256,
            positions=selected_position,
            velocities=selected_velocity,
            indices=selected_indices,
            provenance=provenance,
        )
        started = time.perf_counter()
        reference_result = advance_cuda_restricted_wh_fused(
            reference_state,
            reference_tile,
            FixedStepCUDARestrictedWHSpec(
                reference_steps, 1, backend_spec, threads_per_block=128
            ),
        )
        reference_seconds = time.perf_counter() - started
        reference_position = cp.asnumpy(
            reference_result.state.tracer_relative_positions_AU
        )
        reference_velocity = cp.asnumpy(
            reference_result.state.tracer_relative_velocities_AU_per_year
        )

        records: dict[float, dict[str, object]] = {}
        for step in STEPS_YEARS:
            count = int(round(args.years / step))
            build_started = time.perf_counter()
            source_build = build_wisdom_holman_source_tile(
                tile_id=f"de441-control-wh-{step.hex()}-source-tile",
                epoch_origin_years=0.0,
                global_source_step=0,
                time_scale=metadata["time_scale"],
                frame=metadata["frame"],
                axes="J2000_ECLIPTIC",
                source_ids=tuple(metadata["source_ids"]),
                primary_source_id=metadata["primary_source_id"],
                source_schedule_sha256=schedule_sha256,
                initial_source_relative_positions_AU=source_position,
                initial_source_relative_velocities_AU_per_year=source_velocity,
                source_gm_AU3_per_year2=source_gm,
                source_encounter_radii_AU=source_radii,
                certified_relative_acceleration_bounds_AU_per_year2=(
                    acceleration_bounds
                ),
                provenance=provenance,
                spec=WisdomHolmanSourceTileSpec(step, count),
            )
            build_seconds = time.perf_counter() - build_started
            state = _state(
                cp,
                step=step,
                metadata=metadata,
                schedule_sha256=schedule_sha256,
                positions=selected_position,
                velocities=selected_velocity,
                indices=selected_indices,
                provenance=provenance,
            )
            started = time.perf_counter()
            result = advance_cuda_restricted_wh_fused(
                state,
                source_build.tile,
                FixedStepCUDARestrictedWHSpec(
                    count, 1, backend_spec, threads_per_block=128
                ),
            )
            execution_seconds = time.perf_counter() - started
            position = cp.asnumpy(result.state.tracer_relative_positions_AU)
            velocity = cp.asnumpy(
                result.state.tracer_relative_velocities_AU_per_year
            )
            comparison = provider_benchmark._endpoint_record(
                position,
                velocity,
                reference_position,
                reference_velocity,
                float(source_gm[0]),
            )
            records[step] = {
                "comparison_to_gr15_source_and_finest_tracer_map": comparison,
                "source_build_seconds_including_exact_replay": build_seconds,
                "tracer_execution_seconds": execution_seconds,
                "source_tile_sha256": source_build.tile.content_sha256,
                "source_map_result_content_sha256": (
                    source_build.source_map_result_content_sha256
                ),
                "maximum_kepler_iterations": result.maximum_kepler_iterations,
            }

    def value(step: float, field: str, statistic: str) -> float:
        return float(
            records[step]["comparison_to_gr15_source_and_finest_tracer_map"][
                field
            ][statistic]
        )

    refinement_ratios = {
        "q95_perihelion_0.25_to_0.125": _safe_ratio(
            value(0.25, "absolute_perihelion_delta_AU", "q95"),
            value(0.125, "absolute_perihelion_delta_AU", "q95"),
        ),
        "q95_perihelion_0.125_to_0.0625": _safe_ratio(
            value(0.125, "absolute_perihelion_delta_AU", "q95"),
            value(0.0625, "absolute_perihelion_delta_AU", "q95"),
        ),
        "median_position_0.125_to_0.0625": _safe_ratio(
            value(0.125, "position_error_AU", "median"),
            value(0.0625, "position_error_AU", "median"),
        ),
    }
    report = {
        "schema": "jx.cuda-restricted-wh.end-to-end-refinement.v1",
        "classification": "ENGINEERING_REFINEMENT_ONLY",
        "years": args.years,
        "steps_years": STEPS_YEARS,
        "reference": {
            "source_method": "JX_GR15_V2",
            "source_epsilon": source_refinement.GR15_EPSILONS[-1],
            "tracer_method": "JX_CUDA_RESTRICTED_WH_PRIMARY_RELATIVE_V1",
            "step_years": REFERENCE_STEP_YEARS,
            "source_seconds": gr15_seconds,
            "tracer_seconds": reference_seconds,
            "endpoint_sha256": hashlib.sha256(
                reference_position.tobytes() + reference_velocity.tobytes()
            ).hexdigest(),
        },
        "lane_count": args.lanes,
        "eligible_lane_count": int(eligible.size),
        "selection_sha256": hashlib.sha256(selected_indices.tobytes()).hexdigest(),
        "input_hashes": input_hashes,
        "records": {step.hex(): record for step, record in records.items()},
        "refinement_ratios": refinement_ratios,
        "claims": {
            "production_step_selected": False,
            "independent_tracer_accuracy_qualified": False,
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
