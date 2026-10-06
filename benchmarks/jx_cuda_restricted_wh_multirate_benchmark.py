#!/usr/bin/env python3
"""Bounded RTX throughput benchmark for static-rung CUDA restricted WH.

This is a synthetic engineering fixture, not an accuracy experiment.  It
compares an all-fine fused execution with one static mixed-rung execution
whose equal contiguous lane groups use source-step ratios 1, 2, 4, and 8.
Every mixed group must be bitwise identical to a separate fixed-ratio fused
execution, and every timed path must repeat bitwise on the same CUDA runtime
and device before any timing is reported.

The independent fixed-group gate establishes implementation parity only.  It
does not compare coarse rungs with an external numerical reference and cannot
authorize their accuracy, production use, or scientific interpretation.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import replace
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
    advance_cuda_restricted_wh_fused,
)
from jxplanetx.cuda_restricted_wh_multirate import (
    FrozenStaticRungPlan,
    StaticMultirateCUDARestrictedWHSpec,
    advance_cuda_restricted_wh_static_multirate,
    static_multirate_input_state_sha256,
    static_multirate_spec_sha256,
)
from jxplanetx.engine import BackendSpec, Provenance


SOLAR_GM = 4.0 * math.pi * math.pi
SOURCE_IDS = ("Sun", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto")
RUNG_RATIOS = (1, 2, 4, 8)
MAXIMUM_LANES = 16_800
MAXIMUM_SOURCE_STEPS = 8_000
MAXIMUM_REPEATS = 5
MAXIMUM_STEP_YEARS = 0.125
SCHEDULE_SHA256 = hashlib.sha256(
    b"jx-restricted-wh-static-multirate-throughput-fixture-v1"
).hexdigest()
NONAUTHORIZATION_DECLARATION = (
    "Synthetic fixed-kernel parity and timing only; no external accuracy "
    "reference was executed and no coarse rung is accuracy-authorized."
)
NONAUTHORIZATION_SHA256 = hashlib.sha256(
    NONAUTHORIZATION_DECLARATION.encode("ascii")
).hexdigest()


def _provenance(label: str) -> Provenance:
    return Provenance(
        source_id=f"jx-benchmark.static-multirate.{label}",
        citation="Synthetic engineering throughput fixture; no accuracy claim",
        version="v1",
        sha256=hashlib.sha256(label.encode("ascii")).hexdigest(),
    )


def _tile(step_count: int, step_years: float) -> FrozenSourcePathTile:
    positions = np.zeros((step_count + 1, len(SOURCE_IDS), 3), dtype=np.float64)
    velocities = np.zeros_like(positions)
    # Static distant secondaries exercise the differential kick while keeping
    # this bounded throughput fixture away from encounter-routing regimes.
    scale = 1.0e6
    positions[:, 1, :] = (scale, 0.0, 0.0)
    positions[:, 2, :] = (0.0, scale, 0.0)
    positions[:, 3, :] = (-scale, 0.0, 0.0)
    positions[:, 4, :] = (0.0, -scale, 0.0)
    positions[:, 5, :] = (0.0, 0.0, scale)
    gm_ratios = np.asarray(
        (
            1.0,
            9.547919384243266e-4,
            2.858859806661308e-4,
            4.366244043351563e-5,
            5.151389020466114e-5,
            7.350478973e-9,
        ),
        dtype=np.float64,
    )
    return FrozenSourcePathTile(
        tile_id=f"synthetic-static-multirate-{step_count}",
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
        source_encounter_radii_AU=np.full(
            len(SOURCE_IDS), 0.01, dtype=np.float64
        ),
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
    positions[:, 0] = radii
    velocities[:, 1] = cp.sqrt(cp.float64(SOLAR_GM) / radii)
    return CUDARestrictedWHState(
        snapshot_id="synthetic-static-multirate-throughput",
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


def _ratios(lane_count: int) -> np.ndarray:
    lanes_per_rung = lane_count // len(RUNG_RATIOS)
    return np.repeat(
        np.asarray(RUNG_RATIOS, dtype=np.uint32), lanes_per_rung
    )


def _plan(
    state: CUDARestrictedWHState,
    tile: FrozenSourcePathTile,
    spec: StaticMultirateCUDARestrictedWHSpec,
    ratios: np.ndarray,
) -> FrozenStaticRungPlan:
    return FrozenStaticRungPlan(
        plan_id=(
            "synthetic-equal-contiguous-rungs-"
            f"{ratios.size}-{spec.total_source_steps}"
        ),
        global_source_step=state.global_source_step,
        total_source_steps=spec.total_source_steps,
        source_schedule_sha256=state.source_schedule_sha256,
        source_tile_sha256=tile.content_sha256 or "",
        tracer_indices=np.arange(ratios.size, dtype=np.uint64),
        source_steps_per_map_step=ratios,
        allowed_source_steps_per_map_step=RUNG_RATIOS,
        selection_basis=(
            "PREDECLARED_EQUAL_CONTIGUOUS_SYNTHETIC_GROUPS_BY_TRACER_INDEX"
        ),
        accuracy_policy_id=(
            "ENGINEERING_PARITY_ONLY_NO_ACCURACY_AUTHORIZATION_V1"
        ),
        accuracy_validation_sha256=NONAUTHORIZATION_SHA256,
        input_state_sha256=static_multirate_input_state_sha256(
            state, spec.backend
        ),
        execution_policy_sha256=static_multirate_spec_sha256(spec),
        provenance=_provenance("static-rung-plan"),
    )


def _plain(value):
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _elapsed(callable_, *args):
    started = time.perf_counter()
    result = callable_(*args)
    return result, time.perf_counter() - started


def _array_equal(cp, expected, actual, label: str) -> None:
    if not bool(cp.array_equal(expected, actual)):
        raise RuntimeError(f"bitwise equality failed for {label}")


def _require_result_equal(cp, expected, actual, label: str) -> None:
    if expected.state.global_source_step != actual.state.global_source_step:
        raise RuntimeError(f"{label} ended at a different source step")
    if expected.source_tile_sha256 != actual.source_tile_sha256:
        raise RuntimeError(f"{label} changed source-tile identity")
    for expected_array, actual_array, array_label in (
        (
            expected.state.tracer_relative_positions_AU,
            actual.state.tracer_relative_positions_AU,
            "positions",
        ),
        (
            expected.state.tracer_relative_velocities_AU_per_year,
            actual.state.tracer_relative_velocities_AU_per_year,
            "velocities",
        ),
        (
            expected.state.tracer_indices,
            actual.state.tracer_indices,
            "indices",
        ),
        (
            expected.minimum_endpoint_source_distance_AU,
            actual.minimum_endpoint_source_distance_AU,
            "minimum distances",
        ),
    ):
        _array_equal(
            cp,
            expected_array,
            actual_array,
            f"{label} {array_label}",
        )
    if expected.maximum_kepler_iterations != actual.maximum_kepler_iterations:
        raise RuntimeError(f"{label} changed maximum Kepler iterations")
    expected_plan = getattr(expected, "rung_plan_sha256", None)
    actual_plan = getattr(actual, "rung_plan_sha256", None)
    if expected_plan != actual_plan:
        raise RuntimeError(f"{label} changed rung-plan identity")
    expected_counts = getattr(expected, "rung_lane_counts", None)
    actual_counts = getattr(actual, "rung_lane_counts", None)
    if expected_counts != actual_counts:
        raise RuntimeError(f"{label} changed rung lane counts")
    if _plain(expected.runtime_identity) != _plain(actual.runtime_identity):
        raise RuntimeError(f"{label} changed runtime identity")


def _require_group_equal(cp, mixed, lanes, fixed, ratio: int) -> None:
    if mixed.state.global_source_step != fixed.state.global_source_step:
        raise RuntimeError(f"ratio-{ratio} fixed group ended at another step")
    for mixed_array, fixed_array, label in (
        (
            mixed.state.tracer_relative_positions_AU[lanes],
            fixed.state.tracer_relative_positions_AU,
            "positions",
        ),
        (
            mixed.state.tracer_relative_velocities_AU_per_year[lanes],
            fixed.state.tracer_relative_velocities_AU_per_year,
            "velocities",
        ),
        (
            mixed.state.tracer_indices[lanes],
            fixed.state.tracer_indices,
            "indices",
        ),
        (
            mixed.minimum_endpoint_source_distance_AU[lanes],
            fixed.minimum_endpoint_source_distance_AU,
            "minimum distances",
        ),
    ):
        _array_equal(cp, mixed_array, fixed_array, f"ratio-{ratio} {label}")


def _digest_arrays(cp, arrays, metadata: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    encoded = json.dumps(
        dict(metadata), sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)
    for label, value in arrays:
        host = np.ascontiguousarray(cp.asnumpy(value))
        encoded_label = label.encode("ascii")
        encoded_dtype = host.dtype.str.encode("ascii")
        digest.update(len(encoded_label).to_bytes(4, "little"))
        digest.update(encoded_label)
        digest.update(len(encoded_dtype).to_bytes(4, "little"))
        digest.update(encoded_dtype)
        digest.update(host.ndim.to_bytes(4, "little"))
        for dimension in host.shape:
            digest.update(int(dimension).to_bytes(8, "little"))
        raw = host.tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _result_sha256(cp, result) -> str:
    return _digest_arrays(
        cp,
        (
            ("positions", result.state.tracer_relative_positions_AU),
            ("velocities", result.state.tracer_relative_velocities_AU_per_year),
            ("indices", result.state.tracer_indices),
            (
                "minimum_distances",
                result.minimum_endpoint_source_distance_AU,
            ),
        ),
        {
            "global_source_step": result.state.global_source_step,
            "maximum_kepler_iterations": result.maximum_kepler_iterations,
            "source_tile_sha256": result.source_tile_sha256,
            "rung_plan_sha256": getattr(result, "rung_plan_sha256", None),
        },
    )


def _endpoint_sha256(cp, result) -> str:
    return _digest_arrays(
        cp,
        (
            ("positions", result.state.tracer_relative_positions_AU),
            ("velocities", result.state.tracer_relative_velocities_AU_per_year),
        ),
        {"global_source_step": result.state.global_source_step},
    )


def _require_all_false(claims: Mapping[str, bool], label: str) -> None:
    if not claims or any(value is not False for value in claims.values()):
        raise RuntimeError(f"{label} claims are not an all-false map")


def _subset_state(cp, state: CUDARestrictedWHState, lanes):
    return replace(
        state,
        tracer_relative_positions_AU=cp.ascontiguousarray(
            state.tracer_relative_positions_AU[lanes]
        ),
        tracer_relative_velocities_AU_per_year=cp.ascontiguousarray(
            state.tracer_relative_velocities_AU_per_year[lanes]
        ),
        tracer_indices=cp.ascontiguousarray(state.tracer_indices[lanes]),
    )


def _validate_arguments(parser: argparse.ArgumentParser, args) -> None:
    if (
        not 4 <= args.lanes <= MAXIMUM_LANES
        or args.lanes % len(RUNG_RATIOS)
    ):
        parser.error(
            f"lanes must be a multiple of four in [4, {MAXIMUM_LANES}]"
        )
    if (
        not RUNG_RATIOS[-1] <= args.steps <= MAXIMUM_SOURCE_STEPS
        or args.steps % RUNG_RATIOS[-1]
    ):
        parser.error(
            "steps must be a multiple of eight in "
            f"[8, {MAXIMUM_SOURCE_STEPS}]"
        )
    if not 2 <= args.repeats <= MAXIMUM_REPEATS:
        parser.error(f"repeats must be in [2, {MAXIMUM_REPEATS}]")
    if (
        not 32 <= args.threads <= 128
        or args.threads % 32
    ):
        parser.error("threads must be a warp multiple in [32, 128]")
    if (
        not math.isfinite(args.step_years)
        or not 0.0 < args.step_years <= MAXIMUM_STEP_YEARS
    ):
        parser.error(
            f"step-years must be finite and in (0, {MAXIMUM_STEP_YEARS}]"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lanes", type=int, default=16_800)
    parser.add_argument("--steps", type=int, default=8_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--step-years", type=float, default=0.125)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    _validate_arguments(parser, args)

    import cupy as cp

    backend = BackendSpec("cupy", "cuda:0", args.threads)
    tile = _tile(args.steps, args.step_years)
    initial = _state(cp, args.lanes, args.step_years)
    ratios = _ratios(args.lanes)
    all_fine_spec = FixedStepCUDARestrictedWHSpec(
        args.steps,
        1,
        backend,
        threads_per_block=args.threads,
    )
    mixed_spec = StaticMultirateCUDARestrictedWHSpec(
        args.steps,
        backend,
        threads_per_block=args.threads,
    )
    plan = _plan(initial, tile, mixed_spec, ratios)

    # Compile and warm both timed kernels outside the measured region.
    warm_steps = RUNG_RATIOS[-1]
    warm_lanes = min(args.lanes, 1_024)
    warm_state = _state(cp, warm_lanes, args.step_years)
    warm_tile = _tile(warm_steps, args.step_years)
    warm_fixed_spec = FixedStepCUDARestrictedWHSpec(
        warm_steps,
        1,
        backend,
        threads_per_block=args.threads,
    )
    warm_mixed_spec = StaticMultirateCUDARestrictedWHSpec(
        warm_steps,
        backend,
        threads_per_block=args.threads,
    )
    warm_plan = _plan(
        warm_state,
        warm_tile,
        warm_mixed_spec,
        _ratios(warm_lanes),
    )
    advance_cuda_restricted_wh_fused(
        warm_state, warm_tile, warm_fixed_spec
    )
    advance_cuda_restricted_wh_static_multirate(
        warm_state, warm_tile, warm_mixed_spec, warm_plan
    )

    all_fine_seconds: list[float] = []
    mixed_seconds: list[float] = []
    all_fine_result_sha256: list[str] = []
    mixed_result_sha256: list[str] = []
    all_fine_reference = None
    mixed_reference = None
    for repeat in range(args.repeats):
        if repeat % 2 == 0:
            all_fine, all_fine_elapsed = _elapsed(
                advance_cuda_restricted_wh_fused,
                initial,
                tile,
                all_fine_spec,
            )
            mixed, mixed_elapsed = _elapsed(
                advance_cuda_restricted_wh_static_multirate,
                initial,
                tile,
                mixed_spec,
                plan,
            )
        else:
            mixed, mixed_elapsed = _elapsed(
                advance_cuda_restricted_wh_static_multirate,
                initial,
                tile,
                mixed_spec,
                plan,
            )
            all_fine, all_fine_elapsed = _elapsed(
                advance_cuda_restricted_wh_fused,
                initial,
                tile,
                all_fine_spec,
            )
        if all_fine_reference is None:
            all_fine_reference = all_fine
            mixed_reference = mixed
        else:
            _require_result_equal(
                cp,
                all_fine_reference,
                all_fine,
                "all-fine repeat",
            )
            _require_result_equal(
                cp,
                mixed_reference,
                mixed,
                "static-multirate repeat",
            )
        all_fine_seconds.append(all_fine_elapsed)
        mixed_seconds.append(mixed_elapsed)
        all_fine_result_sha256.append(_result_sha256(cp, all_fine))
        mixed_result_sha256.append(_result_sha256(cp, mixed))

    assert all_fine_reference is not None and mixed_reference is not None
    if len(set(all_fine_result_sha256)) != 1:
        raise RuntimeError("all-fine repeat digest changed")
    if len(set(mixed_result_sha256)) != 1:
        raise RuntimeError("static-multirate repeat digest changed")
    _require_all_false(all_fine_reference.claims, "all-fine engine")
    _require_all_false(mixed_reference.claims, "static-multirate engine")

    independent_groups = {}
    group_maximum_iterations: list[int] = []
    lanes_per_rung = args.lanes // len(RUNG_RATIOS)
    for group, ratio in enumerate(RUNG_RATIOS):
        start = group * lanes_per_rung
        stop = start + lanes_per_rung
        lanes = cp.arange(start, stop, dtype=cp.int64)
        subset = _subset_state(cp, initial, lanes)
        fixed_group_spec = FixedStepCUDARestrictedWHSpec(
            args.steps // ratio,
            ratio,
            backend,
            maximum_primary_dynamical_phase=(
                mixed_spec.maximum_source_dynamical_phase
            ),
            threads_per_block=args.threads,
            primary_outer_boundary_AU=mixed_spec.primary_outer_boundary_AU,
        )
        fixed_group = advance_cuda_restricted_wh_fused(
            subset, tile, fixed_group_spec
        )
        _require_all_false(fixed_group.claims, f"ratio-{ratio} fixed engine")
        _require_group_equal(
            cp, mixed_reference, lanes, fixed_group, ratio
        )
        group_maximum_iterations.append(fixed_group.maximum_kepler_iterations)
        independent_groups[str(ratio)] = {
            "bitwise_equal_to_mixed_group": True,
            "lane_start": start,
            "lane_stop_exclusive": stop,
            "lane_count": lanes_per_rung,
            "map_steps_per_lane": args.steps // ratio,
            "fixed_group_result_sha256": _result_sha256(cp, fixed_group),
            "maximum_kepler_iterations": (
                fixed_group.maximum_kepler_iterations
            ),
        }
    if mixed_reference.maximum_kepler_iterations != max(
        group_maximum_iterations
    ):
        raise RuntimeError(
            "mixed maximum Kepler iterations differ from fixed-group maximum"
        )

    all_fine_median = statistics.median(all_fine_seconds)
    mixed_median = statistics.median(mixed_seconds)
    all_fine_map_tracer_steps = args.lanes * args.steps
    mixed_map_tracer_steps = lanes_per_rung * sum(
        args.steps // ratio for ratio in RUNG_RATIOS
    )
    all_fine_endpoint = _endpoint_sha256(cp, all_fine_reference)
    mixed_endpoint = _endpoint_sha256(cp, mixed_reference)
    all_fine_matches_mixed = all(
        bool(cp.array_equal(left, right))
        for left, right in (
            (
                all_fine_reference.state.tracer_relative_positions_AU,
                mixed_reference.state.tracer_relative_positions_AU,
            ),
            (
                all_fine_reference.state.tracer_relative_velocities_AU_per_year,
                mixed_reference.state.tracer_relative_velocities_AU_per_year,
            ),
        )
    )
    fixed_runtime = _plain(all_fine_reference.runtime_identity)
    mixed_runtime = _plain(mixed_reference.runtime_identity)
    mixed_execution_runtime = mixed_runtime["execution_runtime_identity"]
    claims = dict(mixed_reference.claims)
    claims.update(
        {
            "benchmark_authorizes_accuracy": False,
            "coarse_rung_accuracy_authorized": False,
            "throughput_authorizes_production": False,
        }
    )
    _require_all_false(claims, "benchmark")

    report = {
        "schema": "jx.cuda-restricted-wh.static-multirate-benchmark.v1",
        "classification": "ENGINEERING_THROUGHPUT_ONLY",
        "scientific_fixture": False,
        "target_hardware_class": "SINGLE_NVIDIA_RTX_GPU",
        "precision_gate": (
            "BITWISE_REPEAT_DETERMINISM_AND_INDEPENDENT_FIXED_GROUP_PARITY"
        ),
        "precision_gate_passed": True,
        "lanes": args.lanes,
        "source_steps": args.steps,
        "simulated_years": args.steps * args.step_years,
        "step_years_hex": args.step_years.hex(),
        "threads_per_block": args.threads,
        "repeats": args.repeats,
        "rung_ratios": list(RUNG_RATIOS),
        "rung_lane_counts": _plain(mixed_reference.rung_lane_counts),
        "all_fine_map_tracer_steps": all_fine_map_tracer_steps,
        "mixed_map_tracer_steps": mixed_map_tracer_steps,
        "fine_lattice_tracer_steps": all_fine_map_tracer_steps,
        "all_fine_seconds": all_fine_seconds,
        "static_multirate_seconds": mixed_seconds,
        "all_fine_median_seconds": all_fine_median,
        "static_multirate_median_seconds": mixed_median,
        "static_multirate_speedup_vs_all_fine": (
            all_fine_median / mixed_median
        ),
        "all_fine_map_tracer_steps_per_second": (
            all_fine_map_tracer_steps / all_fine_median
        ),
        "static_multirate_map_tracer_steps_per_second": (
            mixed_map_tracer_steps / mixed_median
        ),
        "static_multirate_fine_lattice_equivalent_steps_per_second": (
            all_fine_map_tracer_steps / mixed_median
        ),
        "repeat_determinism": {
            "scope": "SAME_CUPY_CUDA_RUNTIME_AND_DEVICE",
            "all_fine_bitwise": True,
            "static_multirate_bitwise": True,
            "all_fine_result_sha256": all_fine_result_sha256,
            "static_multirate_result_sha256": mixed_result_sha256,
        },
        "endpoint_comparison": {
            "all_fine_endpoint_sha256": all_fine_endpoint,
            "static_multirate_endpoint_sha256": mixed_endpoint,
            "all_fine_and_static_multirate_bitwise_equal": (
                all_fine_matches_mixed
            ),
            "accuracy_inference_permitted": False,
        },
        "independent_fixed_groups": independent_groups,
        "source_identity": {
            "source_ids": list(SOURCE_IDS),
            "primary_source_id": "Sun",
            "source_schedule_sha256": SCHEDULE_SHA256,
            "source_tile_sha256": tile.content_sha256,
        },
        "kernel_identities": {
            "all_fine_fixed": {
                "method_id": all_fine_reference.method_id,
                "implementation_source_sha256": fixed_runtime[
                    "implementation_source_sha256"
                ],
                "kernel_id": fixed_runtime["kernel_id"],
                "kernel_source_sha256": fixed_runtime[
                    "execution_kernel_source_sha256"
                ],
            },
            "static_multirate": {
                "method_id": mixed_reference.method_id,
                "implementation_source_sha256": mixed_runtime[
                    "implementation_source_sha256"
                ],
                "fixed_implementation_source_sha256": mixed_runtime[
                    "fixed_implementation_source_sha256"
                ],
                "kernel_id": mixed_execution_runtime["kernel_id"],
                "kernel_source_sha256": mixed_execution_runtime[
                    "execution_kernel_source_sha256"
                ],
            },
        },
        "plan_identity": {
            "plan_id": plan.plan_id,
            "rung_plan_sha256": plan.content_sha256,
            "input_state_sha256": plan.input_state_sha256,
            "execution_policy_sha256": plan.execution_policy_sha256,
            "selection_basis": plan.selection_basis,
            "accuracy_policy_id": plan.accuracy_policy_id,
            "accuracy_validation_sha256": plan.accuracy_validation_sha256,
        },
        "benchmark_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "all_fine_runtime_identity": fixed_runtime,
        "static_multirate_runtime_identity": mixed_runtime,
        "accuracy_authorization": {
            "authorized": False,
            "coarse_rungs_authorized": False,
            "independent_accuracy_reference_executed": False,
            "declaration": NONAUTHORIZATION_DECLARATION,
            "declaration_sha256": NONAUTHORIZATION_SHA256,
        },
        "engine_claims": {
            "all_fine_fixed": _plain(all_fine_reference.claims),
            "static_multirate": _plain(mixed_reference.claims),
        },
        "claims": claims,
        "workload_bounds": {
            "maximum_lanes": MAXIMUM_LANES,
            "maximum_source_steps": MAXIMUM_SOURCE_STEPS,
            "maximum_repeats": MAXIMUM_REPEATS,
            "maximum_step_years_hex": MAXIMUM_STEP_YEARS.hex(),
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
