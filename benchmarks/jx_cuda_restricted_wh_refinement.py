#!/usr/bin/env python3
"""Precision-first timestep refinement on JX's frozen outer-system fixture.

This is an engineering characterization, not a science result.  It builds one
fine 0.125-year massive-source KDK path from the tracked qualification input,
selects an outcome-blind quiet tracer subset by initial osculating perihelion,
and advances the identical source/tracer bytes with restricted-WH map ratios
1, 2, 4, and optionally 8.  The finest map is a same-model reference, not an
independent truth solution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from jxplanetx import cuda_massless_tracer as cuda_tracer
from jxplanetx.cuda_restricted_wh import (
    COORDINATE_SYSTEM,
    CUDARestrictedWHState,
    FixedStepCUDARestrictedWHSpec,
    FrozenSourcePathTile,
    advance_cuda_restricted_wh_fused,
)
from jxplanetx.engine import BackendSpec, Provenance
from jxplanetx.engine.backends import resolve_backend


SOURCE_STEP_YEARS = 0.125
PRIMARY_GM = 39.47716239462883


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def _osculating(position: np.ndarray, velocity: np.ndarray, mu: float):
    radius = np.linalg.norm(position, axis=1)
    speed_squared = np.sum(velocity * velocity, axis=1)
    angular = np.cross(position, velocity)
    eccentricity_vector = np.cross(velocity, angular) / mu - position / radius[:, None]
    eccentricity = np.linalg.norm(eccentricity_vector, axis=1)
    energy = 0.5 * speed_squared - mu / radius
    semimajor = np.full(radius.shape, np.nan, dtype=np.float64)
    bound = energy < 0.0
    semimajor[bound] = -mu / (2.0 * energy[bound])
    perihelion = np.full(radius.shape, np.nan, dtype=np.float64)
    perihelion[bound] = semimajor[bound] * (1.0 - eccentricity[bound])
    inclination = np.arccos(
        np.clip(
            angular[:, 2] / np.linalg.norm(angular, axis=1),
            -1.0,
            1.0,
        )
    )
    return {
        "bound": bound,
        "semimajor_AU": semimajor,
        "eccentricity": eccentricity,
        "perihelion_AU": perihelion,
        "inclination_radians": inclination,
    }


def _load_inputs(root: Path):
    arm = root / "runs/cuda_four_arm_screen_v1/qualification/arms/control"
    population_path = (
        root / "runs/cuda_four_arm_screen_v1/qualification/TRANSFORMED_TRACERS.npz"
    )
    source_path = arm / "SOURCE_STATE.npz"
    initial_path = arm / "INITIAL_STATE.npz"
    metadata_path = arm / "SOURCE_STATE.json"
    schedule_path = arm / "FORCE_SCHEDULE.json"
    required = (
        population_path,
        source_path,
        initial_path,
        metadata_path,
        schedule_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required frozen inputs are missing: {missing}")
    with np.load(population_path, allow_pickle=False) as archive:
        tracer_position = np.ascontiguousarray(archive["tracer_positions"])
        tracer_velocity = np.ascontiguousarray(archive["tracer_velocities"])
        tracer_indices = np.ascontiguousarray(archive["tracer_indices"])
    with np.load(source_path, allow_pickle=False) as archive:
        source_position = np.ascontiguousarray(archive["source_positions"])
        source_velocity = np.ascontiguousarray(archive["source_velocities"])
        source_gm = np.ascontiguousarray(archive["source_gm"])
    with np.load(initial_path, allow_pickle=False) as archive:
        source_radii = np.ascontiguousarray(archive["source_encounter_radii"])
        acceleration_bounds = np.ascontiguousarray(
            archive["relative_acceleration_bounds"]
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    hashes = {path.name: _sha256_file(path) for path in required}
    return (
        tracer_position,
        tracer_velocity,
        tracer_indices,
        source_position,
        source_velocity,
        source_gm,
        source_radii,
        acceleration_bounds,
        metadata,
        hashlib.sha256(schedule_path.read_bytes()).hexdigest(),
        hashes,
    )


def _build_tile(
    cp,
    resolved,
    source_position: np.ndarray,
    source_velocity: np.ndarray,
    source_gm: np.ndarray,
    source_radii: np.ndarray,
    acceleration_bounds: np.ndarray,
    metadata: dict,
    schedule_sha256: str,
    source_steps: int,
    source_step_years: float,
    provenance: Provenance,
) -> FrozenSourcePathTile:
    source_kernel, _ = cuda_tracer._path_kernels(resolved)
    path = cuda_tracer._build_source_path(
        resolved,
        source_kernel,
        cp.asarray(source_position),
        cp.asarray(source_velocity),
        cp.asarray(source_gm),
        step_count=source_steps,
        step=source_step_years,
    )
    absolute_position = cp.asnumpy(path.positions)
    absolute_velocity = cp.asnumpy(path.velocities)
    relative_position = np.ascontiguousarray(
        absolute_position - absolute_position[:, 0:1, :]
    )
    relative_velocity = np.ascontiguousarray(
        absolute_velocity - absolute_velocity[:, 0:1, :]
    )
    return FrozenSourcePathTile(
        tile_id=f"de441-control-{source_steps}-fine-steps",
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
        source_relative_positions_AU=relative_position,
        source_relative_velocities_AU_per_year=relative_velocity,
        source_gm_AU3_per_year2=source_gm,
        source_encounter_radii_AU=source_radii,
        certified_relative_acceleration_bounds_AU_per_year2=acceleration_bounds,
        provenance=provenance,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=float, default=1_000.0)
    parser.add_argument("--lanes", type=int, default=2_048)
    parser.add_argument("--minimum-initial-perihelion", type=float, default=60.0)
    parser.add_argument("--ratios", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source_steps_float = args.years / SOURCE_STEP_YEARS
    source_steps = int(round(source_steps_float))
    if args.years <= 0.0 or source_steps <= 0 or source_steps != source_steps_float:
        parser.error("years must be a positive exact multiple of 0.125")
    if args.lanes <= 0 or args.minimum_initial_perihelion <= 0.0:
        parser.error("lanes and minimum initial perihelion must be positive")
    ratios = tuple(args.ratios)
    if not ratios or ratios[0] != 1 or any(
        type(ratio) is not int or ratio <= 0 or source_steps % ratio
        for ratio in ratios
    ):
        parser.error("ratios must start with 1 and divide the source-step count")

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
    ) = _load_inputs(root)
    initial_elements = _osculating(tracer_position, tracer_velocity, float(source_gm[0]))
    eligible = np.flatnonzero(
        initial_elements["bound"]
        & (initial_elements["perihelion_AU"] >= args.minimum_initial_perihelion)
    )
    if eligible.size < args.lanes:
        raise RuntimeError(
            f"only {eligible.size} tracers satisfy the outcome-blind selection"
        )
    selected = eligible[: args.lanes]
    selected_position = np.ascontiguousarray(tracer_position[selected])
    selected_velocity = np.ascontiguousarray(tracer_velocity[selected])
    selected_indices = np.ascontiguousarray(tracer_indices[selected])
    selection_sha256 = hashlib.sha256(selected_indices.tobytes()).hexdigest()

    backend_spec = BackendSpec("cupy", "cuda:0", 128)
    resolved = resolve_backend(backend_spec)
    provenance = Provenance(
        source_id="jx.de441-control-restricted-wh-refinement",
        citation="Tracked JX qualification inputs; engineering refinement only",
        version="v1",
        sha256=input_hashes["SOURCE_STATE.npz"],
    )
    with resolved.activate():
        tile_started = time.perf_counter()
        tile = _build_tile(
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
        tile_seconds = time.perf_counter() - tile_started
        state = CUDARestrictedWHState(
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
            tracer_relative_positions_AU=cp.asarray(selected_position),
            tracer_relative_velocities_AU_per_year=cp.asarray(selected_velocity),
            tracer_indices=cp.asarray(selected_indices),
            provenance=provenance,
        )

        results = {}
        timings = {}
        for ratio in ratios:
            spec = FixedStepCUDARestrictedWHSpec(
                source_steps // ratio,
                ratio,
                backend_spec,
                threads_per_block=128,
            )
            started = time.perf_counter()
            result = advance_cuda_restricted_wh_fused(state, tile, spec)
            timings[str(ratio)] = time.perf_counter() - started
            results[ratio] = (
                cp.asnumpy(result.state.tracer_relative_positions_AU),
                cp.asnumpy(result.state.tracer_relative_velocities_AU_per_year),
                result,
            )

        # Same-runtime repeatability of the finest reference is mandatory.
        repeat = advance_cuda_restricted_wh_fused(
            state,
            tile,
            FixedStepCUDARestrictedWHSpec(source_steps, 1, backend_spec),
        )
        if not bool(
            cp.array_equal(
                repeat.state.tracer_relative_positions_AU,
                cp.asarray(results[1][0]),
            )
        ) or not bool(
            cp.array_equal(
                repeat.state.tracer_relative_velocities_AU_per_year,
                cp.asarray(results[1][1]),
            )
        ):
            raise RuntimeError("finest restricted-WH run did not repeat bitwise")

    reference_position, reference_velocity, reference_result = results[1]
    reference_elements = _osculating(
        reference_position, reference_velocity, float(source_gm[0])
    )
    comparisons = {}
    for ratio in ratios[1:]:
        position, velocity, _ = results[ratio]
        elements = _osculating(position, velocity, float(source_gm[0]))
        comparisons[str(ratio)] = {
            "map_step_years": ratio * SOURCE_STEP_YEARS,
            "position_error_AU_vs_ratio1": _quantiles(
                np.linalg.norm(position - reference_position, axis=1)
            ),
            "velocity_error_AU_per_year_vs_ratio1": _quantiles(
                np.linalg.norm(velocity - reference_velocity, axis=1)
            ),
            "absolute_perihelion_delta_AU_vs_ratio1": _quantiles(
                np.abs(elements["perihelion_AU"] - reference_elements["perihelion_AU"])
            ),
            "absolute_semimajor_delta_AU_vs_ratio1": _quantiles(
                np.abs(elements["semimajor_AU"] - reference_elements["semimajor_AU"])
            ),
            "absolute_eccentricity_delta_vs_ratio1": _quantiles(
                np.abs(elements["eccentricity"] - reference_elements["eccentricity"])
            ),
            "absolute_inclination_delta_radians_vs_ratio1": _quantiles(
                np.abs(
                    elements["inclination_radians"]
                    - reference_elements["inclination_radians"]
                )
            ),
        }

    endpoint_bytes = reference_position.tobytes() + reference_velocity.tobytes()
    report = {
        "schema": "jx.cuda-restricted-wh.refinement.v1",
        "classification": "ENGINEERING_REFINEMENT_ONLY",
        "independent_reference": False,
        "finest_same_model_reference_ratio": 1,
        "finest_map_step_years": SOURCE_STEP_YEARS,
        "years": args.years,
        "source_steps": source_steps,
        "ratios": ratios,
        "lane_count": args.lanes,
        "eligible_lane_count": int(eligible.size),
        "selection_rule": (
            "first tracer IDs in frozen order with initial primary-relative "
            f"two-body osculating perihelion >= {args.minimum_initial_perihelion} AU"
        ),
        "selection_sha256": selection_sha256,
        "selected_first_id": int(selected_indices[0]),
        "selected_last_id": int(selected_indices[-1]),
        "input_hashes": input_hashes,
        "source_tile_sha256": tile.content_sha256,
        "source_tile_build_seconds": tile_seconds,
        "execution_seconds_by_ratio": timings,
        "maximum_kepler_iterations_by_ratio": {
            str(ratio): results[ratio][2].maximum_kepler_iterations
            for ratio in ratios
        },
        "finest_endpoint_sha256": hashlib.sha256(endpoint_bytes).hexdigest(),
        "same_runtime_finest_repeat_bitwise": True,
        "comparisons": comparisons,
        "runtime_identity": dict(reference_result.runtime_identity),
        "claims": {
            "production_qualified": False,
            "timestep_qualified": False,
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
