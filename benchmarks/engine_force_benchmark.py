#!/usr/bin/env python3
"""Opt-in probe for the public JX force and trajectory API.

The probe runs the complete three-model force plan and the adaptive RKF78
trajectory path on one explicitly selected backend. It is not part of the
test suite, does not compare backends, and makes no speedup, correctness, or
scientific-qualification claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

from jxplanetx import __version__
from jxplanetx.engine import (
    AdaptiveRKF78Spec,
    BackendSpec,
    CannonballSRP,
    ForcePlan,
    NewtonianPointMass,
    ParameterMetadata,
    Provenance,
    RestrictedStaticCentral1PN,
    StateSnapshot,
    evaluate,
    integrate_trajectory,
)
from jxplanetx.engine.backends import resolve_backend


UNIT_SYSTEM_ID = "jx.benchmark.synthetic_ltm.v1"
MODEL_INTERACTIONS_PER_TARGET = 3


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value cannot be negative")
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if not 0.0 < value < float("inf"):
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure one explicit JX three-force and RKF78 path."
    )
    parser.add_argument("--backend", choices=("numpy", "cupy"), required=True)
    parser.add_argument(
        "--device",
        required=True,
        help="Use 'cpu' for NumPy or an explicit CUDA device such as 'cuda:0'.",
    )
    parser.add_argument("--targets", type=_positive_int, default=100_000)
    parser.add_argument("--tile-size", type=_positive_int, default=256)
    parser.add_argument("--warmup", type=_nonnegative_int, default=2)
    parser.add_argument("--iterations", type=_positive_int, default=10)
    parser.add_argument("--trajectory-targets", type=_positive_int, default=64)
    parser.add_argument("--trajectory-warmup", type=_nonnegative_int, default=1)
    parser.add_argument("--trajectory-iterations", type=_positive_int, default=1)
    parser.add_argument(
        "--trajectory-duration",
        type=_positive_float,
        default=1.0e-3,
        help="Synthetic duration in the benchmark's declared time unit.",
    )
    return parser


def _provenance() -> Provenance:
    source_path = Path(__file__).resolve()
    return Provenance(
        source_id="jx.benchmark.engine_dynamics.v1",
        citation="Local opt-in synthetic JX engine dynamics benchmark",
        version=__version__,
        sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
    )


def _metadata(
    parameter_id: str,
    units: str,
    provenance: Provenance,
) -> ParameterMetadata:
    return ParameterMetadata(
        parameter_id=parameter_id,
        units=units,
        provenance=provenance,
        uncertainty=None,
        covariance_group=None,
        validity_start=-1.0,
        validity_end=1.0,
    )


def _build_problem(
    backend: Any,
    backend_spec: BackendSpec,
    target_count: int,
    provenance: Provenance,
    label: str,
) -> tuple[StateSnapshot, ForcePlan]:
    xp = backend.xp
    body_count = target_count + 1
    body_ids = ("SOURCE",) + tuple(f"TARGET_{index}" for index in range(target_count))
    target_ids = body_ids[1:]

    with backend.activate():
        positions = xp.zeros((body_count, 3), dtype=xp.float64)
        radial_positions = xp.linspace(1.0, 2.0, target_count, dtype=xp.float64)
        positions[1:, 0] = radial_positions
        velocities = xp.zeros((body_count, 3), dtype=xp.float64)
        velocities[1:, 1] = xp.sqrt(xp.float64(1.0) / radial_positions)
        gravitational_parameters = xp.zeros(body_count, dtype=xp.float64)
        gravitational_parameters[0] = xp.float64(1.0)
        masses = xp.zeros(body_count, dtype=xp.float64)
        masses[0] = xp.float64(1.0)
        radii = xp.zeros(body_count, dtype=xp.float64)
        massive = xp.zeros(body_count, dtype=xp.bool_)
        massive[0] = True
        area_to_mass = xp.ones(target_count, dtype=xp.float64)
        radiation_pressure_coefficient = xp.ones(target_count, dtype=xp.float64)

    snapshot = StateSnapshot(
        snapshot_id=f"jx.benchmark.{label}.snapshot.v1",
        epoch=0.0,
        time_scale="SYNTHETIC",
        frame="CENTRAL_BODY_INERTIAL",
        origin="SOURCE",
        axes="CARTESIAN_RIGHT_HANDED",
        length_unit="L",
        time_unit="T",
        mass_unit="M",
        unit_system_id=UNIT_SYSTEM_ID,
        body_ids=body_ids,
        positions=positions,
        velocities=velocities,
        gravitational_parameters=gravitational_parameters,
        masses=masses,
        radii=radii,
        massive=massive,
        provenance=provenance,
    )
    newtonian = NewtonianPointMass(
        source_ids=("SOURCE",),
        target_ids=target_ids,
        unit_system_id=UNIT_SYSTEM_ID,
        parameter_metadata=(
            _metadata("state.gravitational_parameters", "L^3/T^2", provenance),
        ),
    )
    relativity = RestrictedStaticCentral1PN(
        central_source_id="SOURCE",
        target_ids=target_ids,
        speed_of_light=1_000.0,
        maximum_compactness=0.01,
        maximum_speed_fraction_squared=0.01,
        unit_system_id=UNIT_SYSTEM_ID,
        parameter_metadata=(
            _metadata("speed_of_light", "L/T", provenance),
            _metadata("maximum_compactness", "1", provenance),
            _metadata("maximum_speed_fraction_squared", "1", provenance),
        ),
    )
    radiation_pressure = CannonballSRP(
        radiation_source_id="SOURCE",
        target_ids=target_ids,
        reference_pressure=1.0e-8,
        reference_distance=1.0,
        area_to_mass=area_to_mass,
        radiation_pressure_coefficient=radiation_pressure_coefficient,
        coefficient_convention="QPR",
        attitude_model="ISOTROPIC_CANNONBALL",
        shadow_model="NONE",
        unit_system_id=UNIT_SYSTEM_ID,
        parameter_metadata=(
            _metadata("reference_pressure", "M/(L*T^2)", provenance),
            _metadata("reference_distance", "L", provenance),
            _metadata("area_to_mass", "L^2/M", provenance),
            _metadata("radiation_pressure_coefficient", "1", provenance),
        ),
    )
    return snapshot, ForcePlan(
        plan_id=f"jx.benchmark.{label}.plan.v1",
        backend=backend_spec,
        models=(newtonian, relativity, radiation_pressure),
    )


def _trajectory_spec(
    backend: Any,
    body_count: int,
    duration: float,
) -> AdaptiveRKF78Spec:
    with backend.activate():
        position_atol = backend.xp.full(
            (body_count, 3), 1.0e-12, dtype=backend.xp.float64
        )
        velocity_atol = backend.xp.full(
            (body_count, 3), 1.0e-12, dtype=backend.xp.float64
        )
    return AdaptiveRKF78Spec(
        checkpoint_epochs=(0.0, duration / 2.0, duration),
        initial_step=duration / 4.0,
        minimum_step=duration * 1.0e-12,
        maximum_step=duration / 2.0,
        position_atol=position_atol,
        position_rtol=1.0e-10,
        velocity_atol=velocity_atol,
        velocity_rtol=1.0e-10,
        maximum_steps=100_000,
        maximum_rejections=10_000,
        safety_factor=0.9,
        minimum_scale_factor=0.2,
        maximum_scale_factor=5.0,
    )


def _synchronize(backend: Any) -> None:
    if backend.name == "cupy":
        backend.xp.cuda.get_current_stream().synchronize()


def main() -> int:
    args = _parser().parse_args()
    backend_spec = BackendSpec(
        backend_id=args.backend,
        device=args.device,
        tile_size=args.tile_size,
        dtype="float64",
        allow_fallback=False,
        deterministic_reductions=True,
        fast_math=False,
        determinism_scope="SAME_RUNTIME_DEVICE",
    )
    backend = resolve_backend(backend_spec)
    provenance = _provenance()
    force_snapshot, force_plan = _build_problem(
        backend, backend_spec, args.targets, provenance, "force"
    )
    trajectory_snapshot, trajectory_plan = _build_problem(
        backend,
        backend_spec,
        args.trajectory_targets,
        provenance,
        "trajectory",
    )
    trajectory_spec = _trajectory_spec(
        backend,
        args.trajectory_targets + 1,
        args.trajectory_duration,
    )

    force_result = None
    for _ in range(args.warmup):
        force_result = evaluate(force_snapshot, force_plan)
    _synchronize(backend)
    force_started = time.perf_counter()
    for _ in range(args.iterations):
        force_result = evaluate(force_snapshot, force_plan)
    _synchronize(backend)
    force_elapsed = time.perf_counter() - force_started

    trajectory_result = None
    for _ in range(args.trajectory_warmup):
        trajectory_result = integrate_trajectory(
            trajectory_snapshot, trajectory_plan, trajectory_spec
        )
    _synchronize(backend)
    trajectory_force_evaluations = 0
    trajectory_started = time.perf_counter()
    for _ in range(args.trajectory_iterations):
        trajectory_result = integrate_trajectory(
            trajectory_snapshot, trajectory_plan, trajectory_spec
        )
        trajectory_force_evaluations += trajectory_result.force_evaluations
    _synchronize(backend)
    trajectory_elapsed = time.perf_counter() - trajectory_started

    force_interactions_per_evaluation = (
        args.targets * MODEL_INTERACTIONS_PER_TARGET
    )
    force_interactions = force_interactions_per_evaluation * args.iterations
    trajectory_interactions = (
        args.trajectory_targets
        * MODEL_INTERACTIONS_PER_TARGET
        * trajectory_force_evaluations
    )
    report = {
        "schema": "jx-engine-dynamics-benchmark/v1",
        "jx_version": __version__,
        "scope": "PUBLIC_COMPLETE_THREE_FORCE_PLAN_AND_RKF78_TRAJECTORY",
        "backend": backend.name,
        "backend_runtime_version": str(
            getattr(backend.xp, "__version__", "unknown")
        ),
        "device": backend.device,
        "dtype": "float64",
        "determinism_scope": "SAME_RUNTIME_DEVICE",
        "tile_size": args.tile_size,
        "force_evaluation": {
            "targets": args.targets,
            "model_ids": force_result.applied_model_ids,
            "warmup_evaluations": args.warmup,
            "timed_evaluations": args.iterations,
            "model_interactions_per_evaluation": force_interactions_per_evaluation,
            "elapsed_seconds": force_elapsed,
            "model_interactions_per_second": force_interactions / force_elapsed,
            "result_evidence_class": force_result.evidence_class,
        },
        "trajectory": {
            "method_id": trajectory_result.method_id,
            "targets": args.trajectory_targets,
            "checkpoint_epochs": trajectory_result.checkpoint_epochs,
            "warmup_trajectories": args.trajectory_warmup,
            "timed_trajectories": args.trajectory_iterations,
            "elapsed_seconds": trajectory_elapsed,
            "attempted_steps_last_trajectory": trajectory_result.attempted_steps,
            "accepted_steps_last_trajectory": trajectory_result.accepted_steps,
            "rejected_steps_last_trajectory": trajectory_result.rejected_steps,
            "force_evaluations_total": trajectory_force_evaluations,
            "model_interactions_total": trajectory_interactions,
            "model_interactions_per_second": (
                trajectory_interactions / trajectory_elapsed
            ),
            "result_evidence_class": trajectory_result.evidence_class,
        },
        "interaction_definition": (
            "one selected model-source/target application; synthetic plan has "
            "one Newtonian, one restricted-1PN, and one SRP application per target"
        ),
        "python": platform.python_version(),
        "backend_fallback_allowed": False,
        "gpu_release_validation_claimed": False,
        "performance_claimed": False,
        "speedup_claimed": False,
        "scientific_qualification_claimed": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
