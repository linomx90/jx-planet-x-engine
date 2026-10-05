#!/usr/bin/env python3
"""Compare JX massive-source integrators against an independent GR15 reference.

This is a precision-screening benchmark, not a science result.  It consumes the
tracked six-source control state, removes the initial barycentre exactly in
binary64, and compares the native Wisdom--Holman and direct Cartesian KDK maps
at a common endpoint.  Two independently controlled GR15 runs establish that
the retained reference is numerically converged at the scale being measured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

import jx_cuda_restricted_wh_refinement as inputs
from jxplanetx import cuda_massless_tracer as cuda_tracer
from jxplanetx._gr15_v2_reference import (
    GR15Spec,
    gr15_runtime_identity,
    integrate_gr15,
)
from jxplanetx.engine import (
    BackendSpec,
    FixedStepWisdomHolmanSpec,
    ForcePlan,
    NewtonianPointMass,
    ParameterMetadata,
    Provenance,
    StateSnapshot,
)
from jxplanetx.engine.backends import resolve_backend
from jxplanetx.fast_wisdom_holman import (
    FastWisdomHolmanLoopError,
    fast_wisdom_holman_runtime_identity,
    integrate_fast_wisdom_holman_trajectory,
)


WH_STEPS_YEARS = (0.5, 0.25, 0.125, 0.0625, 0.03125)
KDK_STEPS_YEARS = (0.125, 0.0625, 0.03125, 0.015625, 0.0078125)
# GR15's epsilon controls its highest acceleration-polynomial coefficient; it
# is not a conventional position tolerance.  The validated binary64 envelope
# is around 1e-9.  A 3x tighter companion lane establishes retained-reference
# convergence without driving the controller below its binary64 step floor.
GR15_EPSILONS = (1.0e-9, 3.0e-10)
COMMON_PATH_STEP_YEARS = 0.125


def _sha256_bytes(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _barycentric(
    position: np.ndarray,
    velocity: np.ndarray,
    gm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    total = np.sum(gm, dtype=np.float64)
    position_center = np.sum(position * gm[:, None], axis=0) / total
    velocity_center = np.sum(velocity * gm[:, None], axis=0) / total
    return (
        np.ascontiguousarray(position - position_center),
        np.ascontiguousarray(velocity - velocity_center),
    )


def _primary_relative(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values - values[:, 0:1, :])


def _energy_and_angular(
    position: np.ndarray,
    velocity: np.ndarray,
    gm: np.ndarray,
) -> tuple[float, np.ndarray]:
    kinetic = 0.5 * np.sum(gm[:, None] * velocity * velocity, dtype=np.float64)
    potential = np.float64(0.0)
    for left in range(gm.size):
        for right in range(left + 1, gm.size):
            separation = float(np.linalg.norm(position[right] - position[left]))
            potential -= gm[left] * gm[right] / separation
    angular = np.sum(gm[:, None] * np.cross(position, velocity), axis=0)
    return float(kinetic + potential), np.asarray(angular, dtype=np.float64)


def _error_record(
    position: np.ndarray,
    velocity: np.ndarray,
    reference_position: np.ndarray,
    reference_velocity: np.ndarray,
) -> dict[str, float]:
    relative_position = _primary_relative(position[None, :, :])[0]
    relative_velocity = _primary_relative(velocity[None, :, :])[0]
    reference_relative_position = _primary_relative(reference_position[None, :, :])[0]
    reference_relative_velocity = _primary_relative(reference_velocity[None, :, :])[0]
    position_error = np.linalg.norm(
        relative_position[1:] - reference_relative_position[1:], axis=1
    )
    velocity_error = np.linalg.norm(
        relative_velocity[1:] - reference_relative_velocity[1:], axis=1
    )
    return {
        "maximum_secondary_position_error_AU": float(np.max(position_error)),
        "rms_secondary_position_error_AU": float(
            np.sqrt(np.mean(position_error * position_error))
        ),
        "maximum_secondary_velocity_error_AU_per_year": float(
            np.max(velocity_error)
        ),
        "rms_secondary_velocity_error_AU_per_year": float(
            np.sqrt(np.mean(velocity_error * velocity_error))
        ),
    }


def _state_and_plan(
    body_ids: tuple[str, ...],
    position: np.ndarray,
    velocity: np.ndarray,
    gm: np.ndarray,
    years: float,
    provenance: Provenance,
) -> tuple[StateSnapshot, ForcePlan]:
    snapshot = StateSnapshot(
        "jx.de441.outer-source.wh-refinement.snapshot.v1",
        0.0,
        "TDB",
        "BARYCENTRIC_INERTIAL",
        "BARYCENTER",
        "CARTESIAN_RIGHT_HANDED",
        "AU",
        "year",
        "GM_EQUIVALENT",
        "AU-YEAR",
        body_ids,
        position,
        velocity,
        gm.copy(),
        gm.copy(),
        np.zeros(gm.size, dtype=np.float64),
        np.ones(gm.size, dtype=np.bool_),
        provenance,
    )
    metadata = ParameterMetadata(
        "state.gravitational_parameters",
        "AU^3/year^2",
        provenance,
        None,
        None,
        -1.0,
        years + 1.0,
    )
    plan = ForcePlan(
        "jx.de441.outer-source.wh-refinement.plan.v1",
        BackendSpec("numpy", "cpu", 2),
        (NewtonianPointMass(body_ids, body_ids, "AU-YEAR", (metadata,)),),
    )
    return snapshot, plan


def _run_gr15(
    position: np.ndarray,
    velocity: np.ndarray,
    gm: np.ndarray,
    years: float,
    epsilon: float,
) -> tuple[object, float]:
    count = int(round(years / COMMON_PATH_STEP_YEARS))
    checkpoints = tuple(COMMON_PATH_STEP_YEARS * index for index in range(1, count))
    spec = GR15Spec(
        initial_epoch=0.0,
        final_epoch=years,
        intermediate_epochs=checkpoints,
        initial_step=0.01,
        minimum_step=1.0e-15,
        maximum_step=0.25,
        epsilon=epsilon,
        maximum_steps=2_000_000,
        maximum_rejections=200_000,
    )
    started = time.perf_counter()
    result = integrate_gr15(position, velocity, gm, spec)
    return result, time.perf_counter() - started


def _run_wh(
    snapshot: StateSnapshot,
    plan: ForcePlan,
    body_ids: tuple[str, ...],
    years: float,
    step: float,
) -> tuple[object, float]:
    count = int(round(years / step))
    if float(count) * step != years:
        raise RuntimeError("WH horizon is not an exact step multiple")
    spec = FixedStepWisdomHolmanSpec(
        checkpoint_step_indices=tuple(range(count + 1)),
        fixed_step=step,
        maximum_steps=count,
        jacobi_body_order=body_ids,
        minimum_encounter_pair_separation=0.01,
        minimum_jacobi_periapse=0.1,
        maximum_initial_barycenter_position_norm=1.0e-12,
        maximum_initial_barycenter_velocity_norm=1.0e-12,
    )
    started = time.perf_counter()
    result = integrate_fast_wisdom_holman_trajectory(
        snapshot,
        plan,
        spec,
        exact_replay=(step == 0.125),
    )
    return result, time.perf_counter() - started


def _run_kdk(
    source_position: np.ndarray,
    source_velocity: np.ndarray,
    source_gm: np.ndarray,
    years: float,
    step: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    import cupy as cp

    count = int(round(years / step))
    if float(count) * step != years:
        raise RuntimeError("KDK horizon is not an exact step multiple")
    resolved = resolve_backend(BackendSpec("cupy", "cuda:0", 128))
    started = time.perf_counter()
    with resolved.activate():
        source_kernel, _ = cuda_tracer._path_kernels(resolved)
        path = cuda_tracer._build_source_path(
            resolved,
            source_kernel,
            cp.asarray(source_position),
            cp.asarray(source_velocity),
            cp.asarray(source_gm),
            step_count=count,
            step=step,
        )
        endpoint_position = cp.asnumpy(path.positions[-1])
        endpoint_velocity = cp.asnumpy(path.velocities[-1])
    return endpoint_position, endpoint_velocity, time.perf_counter() - started


def _refinement_ratio(
    records: dict[float, dict[str, object]],
    coarse: float,
    medium: float,
    key: str,
) -> float | None:
    if "error" not in records[coarse] or "error" not in records[medium]:
        return None
    numerator = float(records[coarse]["error"][key])
    denominator = float(records[medium]["error"][key])
    if denominator == 0.0 or not math.isfinite(numerator / denominator):
        return None
    return numerator / denominator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=float, default=100.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.years <= 0.0:
        parser.error("years must be positive")
    for step in (*WH_STEPS_YEARS, *KDK_STEPS_YEARS, COMMON_PATH_STEP_YEARS):
        count = args.years / step
        if int(round(count)) != count:
            parser.error("years must be an exact multiple of every benchmark step")

    root = Path(__file__).resolve().parents[1]
    (
        _,
        _,
        _,
        source_position,
        source_velocity,
        source_gm,
        _,
        _,
        metadata,
        schedule_sha256,
        input_hashes,
    ) = inputs._load_inputs(root)
    body_ids = tuple(metadata["source_ids"])
    bary_position, bary_velocity = _barycentric(
        source_position, source_velocity, source_gm
    )
    source_digest = _sha256_bytes(bary_position, bary_velocity, source_gm)
    provenance = Provenance(
        "jx.de441.outer-source.integrator-refinement",
        "Tracked JX qualification state; engineering precision screen only",
        "v1",
        source_digest,
    )
    snapshot, plan = _state_and_plan(
        body_ids,
        bary_position,
        bary_velocity,
        source_gm,
        args.years,
        provenance,
    )
    initial_energy, initial_angular = _energy_and_angular(
        bary_position, bary_velocity, source_gm
    )

    gr15_runs: dict[float, dict[str, object]] = {}
    for epsilon in GR15_EPSILONS:
        result, seconds = _run_gr15(
            bary_position, bary_velocity, source_gm, args.years, epsilon
        )
        final_energy, final_angular = _energy_and_angular(
            result.positions, result.velocities, source_gm
        )
        gr15_runs[epsilon] = {
            "result": result,
            "seconds": seconds,
            "accepted_steps": result.accepted_steps,
            "rejected_steps": result.rejected_steps,
            "force_evaluations": result.force_evaluations,
            "minimum_accepted_step": result.minimum_accepted_step,
            "maximum_accepted_step": result.maximum_accepted_step,
            "relative_energy_drift": (final_energy - initial_energy)
            / abs(initial_energy),
            "relative_angular_vector_drift": float(
                np.linalg.norm(final_angular - initial_angular)
                / np.linalg.norm(initial_angular)
            ),
            "endpoint_sha256": _sha256_bytes(result.positions, result.velocities),
        }
    reference = gr15_runs[GR15_EPSILONS[-1]]["result"]
    gr15_convergence = _error_record(
        gr15_runs[GR15_EPSILONS[0]]["result"].positions,
        gr15_runs[GR15_EPSILONS[0]]["result"].velocities,
        reference.positions,
        reference.velocities,
    )

    wh_records: dict[float, dict[str, object]] = {}
    for step in WH_STEPS_YEARS:
        try:
            result, seconds = _run_wh(snapshot, plan, body_ids, args.years, step)
        except FastWisdomHolmanLoopError as exc:
            wh_records[step] = {
                "status": "FAILED_CLOSED",
                "error_type": type(exc).__name__,
                "failure_message": str(exc),
            }
            continue
        final_position = result.positions[-1]
        final_velocity = result.velocities[-1]
        final_energy, final_angular = _energy_and_angular(
            final_position, final_velocity, source_gm
        )
        wh_records[step] = {
            "status": "SUCCESS",
            "error": _error_record(
                final_position,
                final_velocity,
                reference.positions,
                reference.velocities,
            ),
            "seconds": seconds,
            "relative_energy_drift": (final_energy - initial_energy)
            / abs(initial_energy),
            "relative_angular_vector_drift": float(
                np.linalg.norm(final_angular - initial_angular)
                / np.linalg.norm(initial_angular)
            ),
            "result_content_sha256": result.result_content_sha256,
            "exact_replay_verified": result.exact_replay_verified,
            "metrics": dict(result.metrics),
        }

    kdk_records: dict[float, dict[str, object]] = {}
    for step in KDK_STEPS_YEARS:
        final_position, final_velocity, seconds = _run_kdk(
            source_position, source_velocity, source_gm, args.years, step
        )
        # The direct KDK starts primary-centric.  Shift its endpoint to a
        # barycentric representation only for invariant accounting; relative
        # errors are translation invariant.
        final_bary_position, final_bary_velocity = _barycentric(
            final_position, final_velocity, source_gm
        )
        final_energy, final_angular = _energy_and_angular(
            final_bary_position, final_bary_velocity, source_gm
        )
        kdk_records[step] = {
            "error": _error_record(
                final_position,
                final_velocity,
                reference.positions,
                reference.velocities,
            ),
            "seconds": seconds,
            "relative_energy_drift": (final_energy - initial_energy)
            / abs(initial_energy),
            "relative_angular_vector_drift": float(
                np.linalg.norm(final_angular - initial_angular)
                / np.linalg.norm(initial_angular)
            ),
            "endpoint_sha256": _sha256_bytes(final_position, final_velocity),
        }

    error_key = "maximum_secondary_position_error_AU"
    report = {
        "schema": "jx.source-integrator-refinement.v1",
        "classification": "ENGINEERING_PRECISION_SCREEN_ONLY",
        "years": args.years,
        "body_ids": body_ids,
        "input_hashes": input_hashes,
        "source_schedule_sha256": schedule_sha256,
        "barycentric_input_sha256": source_digest,
        "gr15": {
            "runtime_identity": gr15_runtime_identity(),
            "epsilons": GR15_EPSILONS,
            "runs": {
                epsilon.hex(): {
                    key: value
                    for key, value in record.items()
                    if key != "result"
                }
                for epsilon, record in gr15_runs.items()
            },
            "coarser_vs_retained_reference": gr15_convergence,
            "retained_reference_epsilon": GR15_EPSILONS[-1],
        },
        "wisdom_holman": {
            "runtime_identity": fast_wisdom_holman_runtime_identity(),
            "records": {step.hex(): record for step, record in wh_records.items()},
            "position_error_ratio_0.125_to_0.0625": _refinement_ratio(
                wh_records, 0.125, 0.0625, error_key
            ),
            "position_error_ratio_0.0625_to_0.03125": _refinement_ratio(
                wh_records, 0.0625, 0.03125, error_key
            ),
        },
        "direct_cartesian_kdk": {
            "records": {step.hex(): record for step, record in kdk_records.items()},
            "position_error_ratio_0.03125_to_0.015625": _refinement_ratio(
                kdk_records, 0.03125, 0.015625, error_key
            ),
            "position_error_ratio_0.015625_to_0.0078125": _refinement_ratio(
                kdk_records, 0.015625, 0.0078125, error_key
            ),
        },
        "claims": {
            "production_qualified": False,
            "long_horizon_accuracy_qualified": False,
            "science_result": False,
            "gr15_is_independent_numerical_reference": True,
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
