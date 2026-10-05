#!/usr/bin/env python3
"""End-to-end source-provider refinement for the restricted CUDA tracer map.

The identical frozen tracer subset is advanced against three source paths:
the replay-verified JX Wisdom--Holman provider, an independently integrated
GR15 reference sampled on the same lattice, and the legacy direct Cartesian
KDK source path.  All tracer maps use the same 0.125-year CUDA step.  This is
an engineering precision screen, not a science or long-horizon result.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import jx_cuda_restricted_wh_refinement as base
import jx_source_integrator_refinement as source_refinement
from jxplanetx.cuda_restricted_wh import (
    COORDINATE_SYSTEM,
    CUDARestrictedWHState,
    FixedStepCUDARestrictedWHSpec,
    FrozenSourcePathTile,
    advance_cuda_restricted_wh_fused,
)
from jxplanetx.engine import BackendSpec, Provenance
from jxplanetx.engine.backends import resolve_backend
from jxplanetx.restricted_wh_source import (
    WisdomHolmanSourceTileSpec,
    build_wisdom_holman_source_tile,
)


SOURCE_STEP_YEARS = 0.125


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _state(
    cp,
    *,
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
        source_step_years=SOURCE_STEP_YEARS,
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


def _reference_tile(
    *,
    source_step_years: float = SOURCE_STEP_YEARS,
    metadata: dict,
    schedule_sha256: str,
    source_gm: np.ndarray,
    source_radii: np.ndarray,
    acceleration_bounds: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    provenance: Provenance,
) -> FrozenSourcePathTile:
    relative_positions = np.ascontiguousarray(positions - positions[:, 0:1, :])
    relative_velocities = np.ascontiguousarray(
        velocities - velocities[:, 0:1, :]
    )
    return FrozenSourcePathTile(
        tile_id="de441-control-gr15-reference-source-tile",
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
        source_relative_positions_AU=relative_positions,
        source_relative_velocities_AU_per_year=relative_velocities,
        source_gm_AU3_per_year2=source_gm,
        source_encounter_radii_AU=source_radii,
        certified_relative_acceleration_bounds_AU_per_year2=acceleration_bounds,
        provenance=provenance,
    )


def _endpoint_record(
    position: np.ndarray,
    velocity: np.ndarray,
    reference_position: np.ndarray,
    reference_velocity: np.ndarray,
    gm_primary: float,
) -> dict[str, object]:
    position_error = np.linalg.norm(position - reference_position, axis=1)
    velocity_error = np.linalg.norm(velocity - reference_velocity, axis=1)
    elements = base._osculating(position, velocity, gm_primary)
    reference_elements = base._osculating(
        reference_position, reference_velocity, gm_primary
    )
    return {
        "position_error_AU": base._quantiles(position_error),
        "velocity_error_AU_per_year": base._quantiles(velocity_error),
        "absolute_perihelion_delta_AU": base._quantiles(
            np.abs(
                elements["perihelion_AU"]
                - reference_elements["perihelion_AU"]
            )
        ),
        "absolute_semimajor_delta_AU": base._quantiles(
            np.abs(
                elements["semimajor_AU"]
                - reference_elements["semimajor_AU"]
            )
        ),
        "endpoint_sha256": hashlib.sha256(
            position.tobytes() + velocity.tobytes()
        ).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=float, default=1_000.0)
    parser.add_argument("--lanes", type=int, default=2_048)
    parser.add_argument("--minimum-initial-perihelion", type=float, default=60.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source_steps_float = args.years / SOURCE_STEP_YEARS
    source_steps = int(round(source_steps_float))
    if (
        args.years <= 0.0
        or source_steps <= 0
        or float(source_steps) * SOURCE_STEP_YEARS != args.years
        or args.lanes <= 0
    ):
        parser.error("years must be a positive 0.125-year multiple and lanes positive")

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
        "jx.de441-control-source-provider-refinement",
        "Tracked JX qualification inputs; engineering precision screen only",
        "v1",
        input_hashes["SOURCE_STATE.npz"],
    )

    wh_started = time.perf_counter()
    wh_build = build_wisdom_holman_source_tile(
        tile_id="de441-control-wh-source-tile",
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
        spec=WisdomHolmanSourceTileSpec(SOURCE_STEP_YEARS, source_steps),
    )
    wh_build_seconds = time.perf_counter() - wh_started

    bary_position, bary_velocity = source_refinement._barycentric(
        source_position, source_velocity, source_gm
    )
    gr15, gr15_seconds = source_refinement._run_gr15(
        bary_position,
        bary_velocity,
        source_gm,
        args.years,
        source_refinement.GR15_EPSILONS[-1],
    )
    gr15_tile = _reference_tile(
        metadata=metadata,
        schedule_sha256=schedule_sha256,
        source_gm=source_gm,
        source_radii=source_radii,
        acceleration_bounds=acceleration_bounds,
        positions=gr15.checkpoint_positions,
        velocities=gr15.checkpoint_velocities,
        provenance=provenance,
    )

    import cupy as cp

    backend_spec = BackendSpec("cupy", "cuda:0", 128)
    resolved = resolve_backend(backend_spec)
    with resolved.activate():
        kdk_started = time.perf_counter()
        kdk_tile = base._build_tile(
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
            SOURCE_STEP_YEARS,
            provenance,
        )
        kdk_build_seconds = time.perf_counter() - kdk_started

        records: dict[str, dict[str, object]] = {}
        for label, tile in (
            ("gr15_reference", gr15_tile),
            ("wisdom_holman", wh_build.tile),
            ("direct_cartesian_kdk", kdk_tile),
        ):
            state = _state(
                cp,
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
                tile,
                FixedStepCUDARestrictedWHSpec(
                    source_steps,
                    1,
                    backend_spec,
                    threads_per_block=128,
                ),
            )
            records[label] = {
                "positions": cp.asnumpy(
                    result.state.tracer_relative_positions_AU
                ),
                "velocities": cp.asnumpy(
                    result.state.tracer_relative_velocities_AU_per_year
                ),
                "seconds": time.perf_counter() - started,
                "runtime_identity": dict(result.runtime_identity),
                "maximum_kepler_iterations": result.maximum_kepler_iterations,
                "tile_sha256": tile.content_sha256,
            }

    reference = records["gr15_reference"]
    comparisons = {
        label: _endpoint_record(
            record["positions"],
            record["velocities"],
            reference["positions"],
            reference["velocities"],
            float(source_gm[0]),
        )
        for label, record in records.items()
        if label != "gr15_reference"
    }
    reference_endpoint_sha256 = hashlib.sha256(
        reference["positions"].tobytes() + reference["velocities"].tobytes()
    ).hexdigest()
    report = {
        "schema": "jx.cuda-restricted-wh.source-provider-refinement.v1",
        "classification": "ENGINEERING_PRECISION_SCREEN_ONLY",
        "years": args.years,
        "source_step_years": SOURCE_STEP_YEARS,
        "source_steps": source_steps,
        "lane_count": args.lanes,
        "eligible_lane_count": int(eligible.size),
        "selection_rule": (
            "first tracer IDs in frozen order with initial primary-relative "
            f"two-body perihelion >= {args.minimum_initial_perihelion} AU"
        ),
        "selection_sha256": hashlib.sha256(selected_indices.tobytes()).hexdigest(),
        "input_hashes": input_hashes,
        "source_schedule_sha256": schedule_sha256,
        "source_build_seconds": {
            "wisdom_holman_including_exact_replay": wh_build_seconds,
            "gr15_reference": gr15_seconds,
            "direct_cartesian_kdk": kdk_build_seconds,
        },
        "wisdom_holman_source_build": {
            "input_content_sha256": wh_build.input_content_sha256,
            "source_map_result_content_sha256": (
                wh_build.source_map_result_content_sha256
            ),
            "source_map_verification_content_sha256": (
                wh_build.source_map_verification_content_sha256
            ),
            "runtime_identity": _jsonable(wh_build.runtime_identity),
        },
        "tracer_execution": {
            label: {
                "seconds": record["seconds"],
                "maximum_kepler_iterations": record["maximum_kepler_iterations"],
                "tile_sha256": record["tile_sha256"],
            }
            for label, record in records.items()
        },
        "gr15_reference_endpoint_sha256": reference_endpoint_sha256,
        "comparisons_to_gr15_source_path": comparisons,
        "claims": {
            "production_qualified": False,
            "long_horizon_accuracy_qualified": False,
            "science_result": False,
            "source_provider_selected_for_next_milestone": True,
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
