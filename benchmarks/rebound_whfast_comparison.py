#!/usr/bin/env python3
"""Source-tree weak-three-body comparison of JX WH, WHFast, and IAS15.

REBOUND 5.1.1 is an optional external GPL-v3-family comparator.  Raw timings
and finite sampled errors are diagnostics only; no authority, qualification,
production, reference-truth, or superiority claim is authorized.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from jxplanetx import __version__ as JX_VERSION
from jxplanetx.engine import (
    BackendSpec,
    FIXED_STEP_WISDOM_HOLMAN_METHOD_ID,
    FixedStepWisdomHolmanSpec,
    ForcePlan,
    NewtonianPointMass,
    ParameterMetadata,
    Provenance,
    StateSnapshot,
    TrajectoryDomainError,
    WH_CHECKPOINT_POLICY,
    WH_COMPOSITION,
    WH_FORCE_EVALUATION_ACCOUNTING,
    WH_JACOBI_BINDING_CHECKSUM_ALGORITHM,
    WH_JACOBI_BINDING_CHECKSUM_DOMAIN,
    WH_PUBLIC_EXECUTION_ACCOUNTING_SCOPE,
    WH_RESULT_CONTENT_CHECKSUM_ALGORITHM,
    WH_RESULT_CONTENT_CHECKSUM_DOMAIN,
    WH_SCHEDULE_CHECKSUM_ALGORITHM,
    WH_SCHEDULE_CHECKSUM_DOMAIN,
    WH_VALIDATION_REPLAY_COUNT,
    WH_VALIDATION_REPLAY_POLICY,
    integrate_wisdom_holman_trajectory,
)


SCHEMA = "jx.rebound_whfast.weak_three_body.v1"
BENCHMARK_ID = "jx.rebound_whfast.ordered_jacobi_weak_three_body.v1"
REQUIRED_REBOUND_VERSION = "5.1.1"
REBOUND_MODES = ("auto", "required", "disabled")
PERIOD = float.fromhex("0x1.91ec4658ac4a3p+2")
IAS15_INITIAL_DT = PERIOD / 128.0
BODY_IDS = ("STAR", "INNER", "OUTER")
GRAVITATIONAL_PARAMETERS = np.array((1.0, 0.001, 0.002), dtype=np.float64)
INITIAL_CARTESIAN_HEX = (
    (
        "0x1.56f40f71a182dp-9",
        "-0x1.3a5c50c2d6a40p-11",
        "0x0p+0",
        "0x1.3d399c60c981cp-11",
        "0x1.2b6efeed14806p-11",
        "0x0p+0",
    ),
    (
        "0x1.ad1b090b71629p-1",
        "0x1.d2c3e5372a878p-2",
        "0x0p+0",
        "-0x1.f55cc646a3929p-2",
        "0x1.db8934f99b592p-1",
        "0x0p+0",
    ),
    (
        "-0x1.ba311957d4125p+0",
        "0x1.2670d10ac1618p-4",
        "0x0p+0",
        "-0x1.d8defc9997fb1p-5",
        "-0x1.7ff9ccf690af4p-1",
        "0x0p+0",
    ),
)
INITIAL_STATE_HASH_DOMAIN = "jx.rebound-whfast.initial-state.v1"
INITIAL_STATE_SHA256 = (
    "9ae0797e88e65d6f49c48e6ca24e13d0845e44bbfa5ee19e73d97dfc1f8ecc08"
)
LANE_CONTENT_CHECKSUM_ALGORITHM = "SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1"
LANE_CONTENT_CHECKSUM_DOMAIN = "jx.rebound-whfast.lane-content.v1"
JX_STEP_DIVISORS = (64, 128)
JX_NEGATIVE_STEP_DIVISOR = 32
SAMPLES_PER_PERIOD = 4
SHORT_CONVERGENCE_RATIO_WINDOW = (3.7, 4.3)
LONG_DESCRIPTIVE_ENVELOPE = {
    "position_vector_l2_max": 1.0e-3,
    "velocity_vector_l2_max": 1.0e-3,
    "phase_proxy_max_abs_radians": 2.0e-3,
    "relative_total_energy_max": 5.0e-6,
    "relative_angular_momentum_max": 1.0e-12,
    "center_of_mass_position_max": 1.0e-12,
    "total_momentum_max": 1.0e-12,
}
CLAIM_CONTROLS = {
    "timing_comparable": False,
    "superiority_claimed": False,
    "qualification_claimed": False,
    "production_use_authorized": False,
    "reference_truth_claimed": False,
    "registry_authorized": False,
    "external_comparator_authorizes_jx": False,
    "long_term_boundedness_claimed": False,
    "finite_step_map_equivalence_claimed": False,
}
FORBIDDEN_RESULT_KEYS = frozenset(("winner", "defeated", "passed", "speedup"))


class BenchmarkError(RuntimeError):
    """The comparison contract could not be satisfied exactly."""


class ReboundUnavailable(BenchmarkError):
    """The optional exact-version external comparator is unavailable."""


def _load_provenance_support() -> Any:
    path = Path(__file__).resolve().with_name("rebound_leapfrog_comparison.py")
    if not path.is_file() or path.is_symlink():
        raise BenchmarkError("the frozen provenance-support source is unavailable")
    support_bytes = path.read_bytes()
    support_digest = hashlib.sha256(support_bytes).hexdigest()
    name = f"jx_rebound_leapfrog_provenance_support_{support_digest}"
    if name in sys.modules:
        raise BenchmarkError("preloaded provenance-support substitution detected")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise BenchmarkError("the provenance-support module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    required_callables = (
        "_exact_tree_equal",
        "_file_identity",
        "_engine_sources_manifest",
        "_numpy_runtime_identity",
        "_load_rebound",
        "_rebound_runtime_provenance",
        "_validate_rebound_runtime_provenance",
    )
    if (
        Path(module.__file__).resolve() != path
        or module.__loader__ is not spec.loader
        or module.__spec__ is not spec
        or Path(spec.origin).resolve() != path
        or module.REQUIRED_REBOUND_VERSION != REQUIRED_REBOUND_VERSION
    ):
        raise BenchmarkError("provenance-support REBOUND version is inconsistent")
    for callable_name in required_callables:
        value = getattr(module, callable_name, None)
        source = inspect.getsourcefile(value) if callable(value) else None
        if source is None or Path(source).resolve() != path:
            raise BenchmarkError(
                "provenance-support callable substitution detected"
            )
    return module


_PROVENANCE_SUPPORT = _load_provenance_support()


@dataclass(frozen=True)
class Profile:
    short_periods: int = 1
    long_periods: int = 2
    samples_per_period: int = SAMPLES_PER_PERIOD

    def __post_init__(self) -> None:
        for name in ("short_periods", "long_periods", "samples_per_period"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive built-in integer")
        if self.long_periods < self.short_periods:
            raise ValueError("long_periods cannot be smaller than short_periods")
        if any(divisor % self.samples_per_period for divisor in JX_STEP_DIVISORS):
            raise ValueError("samples_per_period must divide both fixed-step divisors")

    @classmethod
    def full(cls) -> "Profile":
        return cls(10, 100, 4)

    @property
    def name(self) -> str:
        if (self.short_periods, self.long_periods, self.samples_per_period) == (
            10,
            100,
            4,
        ):
            return "full-10-and-100-period"
        if (self.short_periods, self.long_periods, self.samples_per_period) == (
            1,
            2,
            4,
        ):
            return "smoke"
        return "custom"

    @property
    def profile_id(self) -> str:
        return f"{BENCHMARK_ID}.{self.name}.v1"

    @property
    def named_full_profile(self) -> bool:
        return self.name == "full-10-and-100-period"

    def periods_for(self, horizon: str) -> int:
        if horizon == "short":
            return self.short_periods
        if horizon == "long":
            return self.long_periods
        raise ValueError("horizon must be 'short' or 'long'")

    def checkpoint_step_indices(
        self, horizon: str, step_divisor: int
    ) -> tuple[int, ...]:
        if type(step_divisor) is not int or step_divisor <= 0:
            raise ValueError("step_divisor must be a positive built-in integer")
        if step_divisor % self.samples_per_period:
            raise ValueError("samples_per_period must divide step_divisor")
        stride = step_divisor // self.samples_per_period
        count = self.periods_for(horizon) * self.samples_per_period
        return tuple(index * stride for index in range(count + 1))

    def declared_epochs(self, horizon: str, step_divisor: int) -> tuple[float, ...]:
        step = PERIOD / step_divisor
        return tuple(
            float(step * index)
            for index in self.checkpoint_step_indices(horizon, step_divisor)
        )


@dataclass(frozen=True, eq=False)
class Lane:
    engine_id: str
    horizon: str
    step_divisor: int | None
    declared_epochs: np.ndarray
    observed_epochs: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    setup_seconds: float
    integration_seconds: float
    settings: dict[str, Any]
    runtime: dict[str, Any]
    accounting: dict[str, Any]

    def __post_init__(self) -> None:
        _validate_lane_arrays(self)


@dataclass(frozen=True, eq=False)
class StudyRun:
    profile: Profile
    rebound_mode: str
    jx_lanes: tuple[Lane, ...]
    whfast_lanes: tuple[Lane, ...]
    ias15_lanes: tuple[Lane, ...]
    negative_control: dict[str, Any]
    external_status: dict[str, Any]


def _owned_readonly_float64(value: Any) -> np.ndarray:
    array = np.array(value, dtype=np.float64, order="C", copy=True, subok=False)
    array.setflags(write=False)
    return array


def _validate_lane_arrays(lane: Any) -> None:
    if type(lane) is not Lane:
        raise BenchmarkError("lane array custody requires an exact Lane")
    arrays = {
        "declared_epochs": lane.declared_epochs,
        "observed_epochs": lane.observed_epochs,
        "positions": lane.positions,
        "velocities": lane.velocities,
    }
    for label, array in arrays.items():
        if type(array) is not np.ndarray:
            raise BenchmarkError(f"{label} must be an exact base ndarray")
        if array.dtype != np.dtype(np.float64):
            raise BenchmarkError(f"{label} must have dtype float64")
        if not array.flags.owndata or array.flags.writeable:
            raise BenchmarkError(f"{label} must be owned and read-only")
        if not array.flags.c_contiguous:
            raise BenchmarkError(f"{label} must be C-contiguous")
        if not np.all(np.isfinite(array)):
            raise BenchmarkError(f"{label} must contain only finite values")
    if lane.declared_epochs.ndim != 1 or len(lane.declared_epochs) == 0:
        raise BenchmarkError("declared_epochs must be a nonempty vector")
    if lane.observed_epochs.shape != lane.declared_epochs.shape:
        raise BenchmarkError("observed_epochs must match the declared epoch shape")
    expected_state_shape = (len(lane.declared_epochs), 3, 3)
    if lane.positions.shape != expected_state_shape:
        raise BenchmarkError("positions must have shape (checkpoints,3,3)")
    if lane.velocities.shape != expected_state_shape:
        raise BenchmarkError("velocities must have shape (checkpoints,3,3)")
    retained = tuple(arrays.items())
    for first_index, (first_label, first) in enumerate(retained):
        for second_label, second in retained[first_index + 1 :]:
            if np.shares_memory(first, second):
                raise BenchmarkError(
                    f"{first_label} and {second_label} must not share memory"
                )


def _initial_state_payload() -> dict[str, Any]:
    return {
        "schema": "jx.rebound_whfast.weak_three_body.initial_state.v1",
        "body_ids": BODY_IDS,
        "gm_hex": tuple(float(value).hex() for value in GRAVITATIONAL_PARAMETERS),
        "cartesian_rows_hex": INITIAL_CARTESIAN_HEX,
    }


def _initial_state_sha256() -> str:
    payload = json.dumps(
        _initial_state_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(
        INITIAL_STATE_HASH_DOMAIN.encode("ascii") + b"\0" + payload
    ).hexdigest()


def _initial_arrays() -> tuple[np.ndarray, np.ndarray]:
    if _initial_state_sha256() != INITIAL_STATE_SHA256:
        raise BenchmarkError("the locked weak-three-body input bytes changed")
    state = np.array(
        [[float.fromhex(value) for value in row] for row in INITIAL_CARTESIAN_HEX],
        dtype=np.float64,
    )
    return state[:, :3].copy(), state[:, 3:].copy()


def _file_identity(path: Any) -> dict[str, Any]:
    return _PROVENANCE_SUPPORT._file_identity(path)


def _jx_runtime_provenance() -> dict[str, Any]:
    return {
        "classification": "PROVENANCE_DIAGNOSTIC",
        "authority_authorized": False,
        "jxplanetx_version": JX_VERSION,
        "engine_sources": _PROVENANCE_SUPPORT._engine_sources_manifest(),
        "benchmark_script": _file_identity(__file__),
        "provenance_support_script": _file_identity(_PROVENANCE_SUPPORT.__file__),
        "numpy": _PROVENANCE_SUPPORT._numpy_runtime_identity(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def _validate_jx_runtime_provenance(runtime: Any) -> None:
    if not _PROVENANCE_SUPPORT._exact_tree_equal(runtime, _jx_runtime_provenance()):
        raise BenchmarkError("JX runtime provenance differs from loaded source bytes")


def _rebound_runtime_provenance(rebound: Any) -> dict[str, Any]:
    return _PROVENANCE_SUPPORT._rebound_runtime_provenance(rebound)


def _validate_rebound_runtime_provenance(runtime: Any) -> None:
    _PROVENANCE_SUPPORT._validate_rebound_runtime_provenance(runtime)


def _load_rebound() -> Any:
    try:
        return _PROVENANCE_SUPPORT._load_rebound()
    except _PROVENANCE_SUPPORT.ReboundUnavailable as exc:
        raise ReboundUnavailable(str(exc)) from exc
    except _PROVENANCE_SUPPORT.BenchmarkError as exc:
        raise BenchmarkError(str(exc)) from exc


def _state_and_plan(profile: Profile) -> tuple[StateSnapshot, ForcePlan]:
    positions, velocities = _initial_arrays()
    source = _file_identity(__file__)["sha256"]
    provenance = Provenance(
        "jx.benchmark.whfast.weak-three-body.v1",
        "benchmarks/rebound_whfast_comparison.py",
        "1",
        source,
    )
    snapshot = StateSnapshot(
        "jx.benchmark.whfast.weak-three-body.snapshot.v1",
        0.0,
        "TDB",
        "BARYCENTRIC_INERTIAL",
        "BARYCENTER",
        "ICRS_ALIGNED",
        "L",
        "T",
        "M",
        "jx.benchmark.whfast.synthetic.v1",
        BODY_IDS,
        positions,
        velocities,
        GRAVITATIONAL_PARAMETERS.copy(),
        GRAVITATIONAL_PARAMETERS.copy(),
        np.zeros(3, dtype=np.float64),
        np.ones(3, dtype=np.bool_),
        provenance,
    )
    validity_end = float(PERIOD * (profile.long_periods + 1))
    metadata = ParameterMetadata(
        "state.gravitational_parameters",
        "L^3/T^2",
        provenance,
        None,
        None,
        -1.0,
        validity_end,
    )
    plan = ForcePlan(
        "jx.benchmark.whfast.weak-three-body.plan.v1",
        BackendSpec("numpy", "cpu", 2),
        (
            NewtonianPointMass(
                BODY_IDS,
                BODY_IDS,
                "jx.benchmark.whfast.synthetic.v1",
                (metadata,),
            ),
        ),
    )
    return snapshot, plan


def _wh_spec(
    profile: Profile, horizon: str, step_divisor: int
) -> FixedStepWisdomHolmanSpec:
    indices = profile.checkpoint_step_indices(horizon, step_divisor)
    return FixedStepWisdomHolmanSpec(
        checkpoint_step_indices=indices,
        fixed_step=PERIOD / step_divisor,
        maximum_steps=indices[-1],
        jacobi_body_order=BODY_IDS,
        minimum_encounter_pair_separation=0.01,
        minimum_jacobi_periapse=0.05,
        maximum_initial_barycenter_position_norm=1.0e-12,
        maximum_initial_barycenter_velocity_norm=1.0e-12,
    )


_PUBLIC_COUNT_FIELDS = tuple(
    field.name
    for field in fields(
        __import__(
            "jxplanetx.engine.wisdom_holman",
            fromlist=["WisdomHolmanTrajectoryResult"],
        ).WisdomHolmanTrajectoryResult
    )
    if field.name.startswith("primary_map_")
    or field.name.startswith("validation_replay_")
    or field.name.startswith("total_public_call_")
)


def _jx_accounting(result: Any) -> dict[str, Any]:
    accounting = {
        name: getattr(result, name) for name in _PUBLIC_COUNT_FIELDS
    }
    accounting.update(
        {
            "custody_source": "WisdomHolmanTrajectoryResult",
            "completed_steps": result.completed_steps,
            "checkpoint_count": result.checkpoint_count,
            "method_id": result.method_id,
            "composition": result.composition,
            "checkpoint_policy": result.checkpoint_policy,
            "force_evaluation_accounting": result.force_evaluation_accounting,
            "public_execution_accounting_scope": (
                result.public_execution_accounting_scope
            ),
            "validation_replay_policy": result.validation_replay_policy,
            "validation_replay_count": result.validation_replay_count,
            "coordinate_binding_sha256": (
                result.coordinate_binding.binding_content_sha256
            ),
            "coordinate_binding_checksum_algorithm": (
                result.coordinate_binding.checksum_algorithm
            ),
            "coordinate_binding_checksum_domain": (
                result.coordinate_binding.checksum_domain
            ),
            "schedule_checksum_algorithm": result.schedule_checksum_algorithm,
            "schedule_checksum_domain": result.schedule_checksum_domain,
            "schedule_content_sha256": result.schedule_content_sha256,
            "result_content_checksum_algorithm": (
                result.result_content_checksum_algorithm
            ),
            "result_content_checksum_domain": result.result_content_checksum_domain,
            "result_content_sha256": result.result_content_sha256,
            "maximum_universal_solver_iterations": (
                result.maximum_universal_solver_iterations
            ),
            "maximum_universal_solver_bracket_expansions": (
                result.maximum_universal_solver_bracket_expansions
            ),
            "maximum_kepler_time_residual": result.maximum_kepler_time_residual,
            "maximum_kepler_residual_tolerance": (
                result.maximum_kepler_residual_tolerance
            ),
            "maximum_kepler_lagrange_identity_error": (
                result.maximum_kepler_lagrange_identity_error
            ),
            "maximum_kepler_energy_error": result.maximum_kepler_energy_error,
            "maximum_kepler_angular_momentum_error": (
                result.maximum_kepler_angular_momentum_error
            ),
            "maximum_translation_force_residual": (
                result.maximum_translation_force_residual
            ),
            "minimum_jacobi_periapses": list(result.minimum_jacobi_periapses),
            "maximum_jacobi_eccentricities": list(
                result.maximum_jacobi_eccentricities
            ),
            "maximum_interaction_force_ratios": list(
                result.maximum_interaction_force_ratios
            ),
            "maximum_orbit_step_fractions": list(
                result.maximum_orbit_step_fractions
            ),
            "maximum_periapse_step_fractions": list(
                result.maximum_periapse_step_fractions
            ),
            "minimum_pair_path_lower_bounds": list(
                result.minimum_pair_path_lower_bounds
            ),
            "minimum_pair_clearance_after_margins": list(
                result.minimum_pair_clearance_after_margins
            ),
            "minimum_secondary_hill_floor_ratios": list(
                result.minimum_secondary_hill_floor_ratios
            ),
            "maximum_barycenter_position_norm": (
                result.maximum_barycenter_position_norm
            ),
            "maximum_barycenter_velocity_norm": (
                result.maximum_barycenter_velocity_norm
            ),
            "formal_exact_kepler_subflow_symplectic": (
                result.formal_exact_kepler_subflow_symplectic
            ),
            "floating_point_symplectic": result.floating_point_symplectic,
            "formal_exact_kepler_subflow_time_reversible": (
                result.formal_exact_kepler_subflow_time_reversible
            ),
            "floating_point_exactly_reversible": (
                result.floating_point_exactly_reversible
            ),
            "evidence_class": result.evidence_class,
            "registry_authorized": result.registry_authorized,
            "qualification_authorized": result.qualification_authorized,
        }
    )
    return accounting


def _jx_lane(profile: Profile, horizon: str, step_divisor: int) -> Lane:
    started = time.perf_counter()
    snapshot, plan = _state_and_plan(profile)
    spec = _wh_spec(profile, horizon, step_divisor)
    setup_seconds = time.perf_counter() - started
    integrated = time.perf_counter()
    result = integrate_wisdom_holman_trajectory(snapshot, plan, spec)
    integration_seconds = time.perf_counter() - integrated
    epochs = np.asarray(result.checkpoint_epochs, dtype=np.float64)
    settings = {
        "method_id": FIXED_STEP_WISDOM_HOLMAN_METHOD_ID,
        "fixed_step": spec.fixed_step,
        "step_divisor": step_divisor,
        "checkpoint_step_indices": list(spec.checkpoint_step_indices),
        "maximum_steps": spec.maximum_steps,
        "jacobi_body_order": list(spec.jacobi_body_order),
        "minimum_encounter_pair_separation": (
            spec.minimum_encounter_pair_separation
        ),
        "minimum_jacobi_periapse": spec.minimum_jacobi_periapse,
        "maximum_initial_barycenter_position_norm": (
            spec.maximum_initial_barycenter_position_norm
        ),
        "maximum_initial_barycenter_velocity_norm": (
            spec.maximum_initial_barycenter_velocity_norm
        ),
        "composition": WH_COMPOSITION,
        "checkpoint_policy": WH_CHECKPOINT_POLICY,
        "force_evaluation_accounting": WH_FORCE_EVALUATION_ACCOUNTING,
        "public_execution_accounting_scope": WH_PUBLIC_EXECUTION_ACCOUNTING_SCOPE,
        "validation_replay_policy": WH_VALIDATION_REPLAY_POLICY,
        "validation_replay_count": WH_VALIDATION_REPLAY_COUNT,
        "backend": "numpy",
        "device": "cpu",
        "dtype": "float64",
        "force_plan": "MUTUAL_ALL_BODY_NEWTONIAN_POINT_MASS_ONLY",
        "physical_masses_used": False,
        "all_bodies_active": True,
    }
    return Lane(
        f"jx_wh_{horizon}_p{step_divisor}",
        horizon,
        step_divisor,
        _owned_readonly_float64(epochs),
        _owned_readonly_float64(epochs),
        _owned_readonly_float64(np.stack(result.positions)),
        _owned_readonly_float64(np.stack(result.velocities)),
        float(setup_seconds),
        float(integration_seconds),
        settings,
        _jx_runtime_provenance(),
        _jx_accounting(result),
    )


def _jx_p32_negative_control(profile: Profile) -> dict[str, Any]:
    snapshot, plan = _state_and_plan(profile)
    spec = _wh_spec(Profile(1, 1, 4), "short", JX_NEGATIVE_STEP_DIVISOR)
    try:
        integrate_wisdom_holman_trajectory(snapshot, plan, spec)
    except TrajectoryDomainError as exc:
        return {
            "engine_id": "jx_wh_negative_p32",
            "step_divisor": JX_NEGATIVE_STEP_DIVISOR,
            "requested_periods": 1,
            "domain_rejection_observed": True,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "expected_screen": "CONSERVATIVE_CONTACT_ENCOUNTER_OR_HILL_SCREEN",
            "fallback_attempted": False,
            "qualification_authorized": False,
        }
    raise BenchmarkError("the locked JX P/32 negative-domain case was not rejected")


def _particle_readback(simulation: Any) -> list[dict[str, float]]:
    return [
        {
            name: float(getattr(particle, name))
            for name in ("m", "r", "x", "y", "z", "vx", "vy", "vz")
        }
        for particle in simulation.particles
    ]


def _simulation_state(simulation: Any) -> tuple[np.ndarray, np.ndarray]:
    positions = np.array(
        [[particle.x, particle.y, particle.z] for particle in simulation.particles],
        dtype=np.float64,
    )
    velocities = np.array(
        [[particle.vx, particle.vy, particle.vz] for particle in simulation.particles],
        dtype=np.float64,
    )
    if positions.shape != (3, 3) or velocities.shape != (3, 3):
        raise BenchmarkError("REBOUND returned an unexpected particle roster")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
        raise BenchmarkError("REBOUND returned a nonfinite state")
    return positions, velocities


def _add_locked_particles(simulation: Any) -> None:
    positions, velocities = _initial_arrays()
    for index in range(3):
        simulation.add(
            m=float(GRAVITATIONAL_PARAMETERS[index]),
            r=0.0,
            x=float(positions[index, 0]),
            y=float(positions[index, 1]),
            z=float(positions[index, 2]),
            vx=float(velocities[index, 0]),
            vy=float(velocities[index, 1]),
            vz=float(velocities[index, 2]),
        )
    simulation.N_active = 3
    simulation.testparticle_type = 0


def _configured_whfast_simulation(rebound: Any, fixed_step: float) -> Any:
    simulation = rebound.Simulation()
    simulation.G = 1.0
    simulation.gravity = "basic"
    simulation.collision = "none"
    simulation.boundary = "none"
    simulation.softening = 0.0
    simulation.integrator = "whfast"
    simulation.dt = fixed_step
    integrator = simulation.integrator
    integrator.coordinates = "jacobi"
    integrator.corrector = 0
    integrator.corrector2 = 0
    integrator.kernel = "default"
    integrator.safe_mode = 1
    integrator.keep_unsynchronized = 0
    _add_locked_particles(simulation)
    if not callable(getattr(simulation, "steps", None)):
        raise BenchmarkError("REBOUND 5.1.1 does not expose Simulation.steps")
    return simulation


def _configured_ias15_simulation(rebound: Any) -> Any:
    simulation = rebound.Simulation()
    simulation.G = 1.0
    simulation.gravity = "basic"
    simulation.collision = "none"
    simulation.boundary = "none"
    simulation.softening = 0.0
    simulation.integrator = "ias15"
    simulation.dt = IAS15_INITIAL_DT
    simulation.integrator.epsilon = 1.0e-13
    simulation.integrator.min_dt = 0.0
    simulation.integrator.adaptive_mode = "PRS23"
    _add_locked_particles(simulation)
    return simulation


def _clock_replay(checkpoint_indices: tuple[int, ...], fixed_step: float) -> np.ndarray:
    values = np.empty(len(checkpoint_indices), dtype=np.float64)
    values[0] = np.float64(0.0)
    current = 0.0
    completed = 0
    half = float(0.5 * fixed_step)
    for output_index, target in enumerate(checkpoint_indices[1:], start=1):
        for _ in range(target - completed):
            current = float(float(current + half) + half)
        completed = target
        values[output_index] = current
    return values


def _whfast_settings(simulation: Any) -> dict[str, Any]:
    integrator = simulation.integrator
    name = integrator.name
    if type(name) is bytes:
        name = name.decode("ascii")
    return {
        "provenance_categories": {
            "persistent_effective_readback": [
                "integrator",
                "integrator_name",
                "coordinates",
                "corrector",
                "corrector2",
                "kernel",
                "safe_mode",
                "keep_unsynchronized",
                "G",
                "gravity",
                "collision",
                "boundary",
                "softening",
                "dt",
                "N",
                "N_active",
                "testparticle_type",
                "particles_initial_readback",
            ],
            "call_argument": ["Simulation.steps(integer_delta)"],
            "environment_observation": ["OMP_NUM_THREADS"],
        },
        "integrator": str(simulation.integrator),
        "integrator_name": name,
        "coordinates": str(integrator.coordinates),
        "corrector": int(integrator.corrector),
        "corrector2": int(integrator.corrector2),
        "kernel": str(integrator.kernel),
        "safe_mode": int(integrator.safe_mode),
        "keep_unsynchronized": int(integrator.keep_unsynchronized),
        "G": float(simulation.G),
        "gravity": str(simulation.gravity),
        "collision": str(simulation.collision),
        "boundary": str(simulation.boundary),
        "softening": float(simulation.softening),
        "dt": float(simulation.dt),
        "N": int(simulation.N),
        "N_active": int(simulation.N_active),
        "testparticle_type": int(simulation.testparticle_type),
        "particles_initial_readback": _particle_readback(simulation),
        "step_api": "Simulation.steps(integer_delta)",
        "callbacks": "NONE",
        "thread_observation": {"OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"]},
    }


def _whfast_one_step_evidence(rebound: Any, fixed_step: float) -> dict[str, Any]:
    simulation = _configured_whfast_simulation(rebound, fixed_step)
    before_positions, before_velocities = _simulation_state(simulation)
    simulation.steps(1)
    after_positions, after_velocities = _simulation_state(simulation)
    digest = hashlib.sha256()
    for array in (
        before_positions,
        before_velocities,
        after_positions,
        after_velocities,
    ):
        digest.update(array.tobytes(order="C"))
    expected_clock = _clock_replay((0, 1), fixed_step)[-1]
    if float(simulation.t).hex() != float(expected_clock).hex():
        raise BenchmarkError("WHFast one-step clock differs from two-half replay")
    if int(simulation.steps_done) != 1 or float(simulation.dt_last_done) != fixed_step:
        raise BenchmarkError("WHFast one-step accounting readback is inconsistent")
    if int(simulation.is_synchronized) != 1:
        raise BenchmarkError("WHFast one-step output is not synchronized")
    return {
        "classification": "IMPLEMENTATION_BEHAVIOR_EVIDENCE",
        "authority_authorized": False,
        "state_sha256": digest.hexdigest(),
        "observed_epoch_hex": float(simulation.t).hex(),
        "two_half_clock_replay_hex": float(expected_clock).hex(),
        "steps_done": int(simulation.steps_done),
        "dt_last_done_hex": float(simulation.dt_last_done).hex(),
        "is_synchronized": int(simulation.is_synchronized),
        "official_integrator_documentation": (
            "https://rebound.hanno-rein.de/integrators/whfast/"
        ),
        "official_steps_api_documentation": (
            "https://rebound.hanno-rein.de/python_api/Simulation.html"
        ),
        "finite_step_map_equivalence_claimed": False,
    }


def _whfast_lane(
    profile: Profile,
    horizon: str,
    step_divisor: int,
    rebound: Any,
    runtime: dict[str, Any],
) -> Lane:
    fixed_step = PERIOD / step_divisor
    indices = profile.checkpoint_step_indices(horizon, step_divisor)
    declared = np.asarray(profile.declared_epochs(horizon, step_divisor))
    started = time.perf_counter()
    simulation = _configured_whfast_simulation(rebound, fixed_step)
    settings = _whfast_settings(simulation)
    setup_seconds = time.perf_counter() - started
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    observed: list[float] = []
    steps_done_readbacks: list[int] = []
    synchronization_readbacks: list[int] = []
    initial_position, initial_velocity = _simulation_state(simulation)
    positions.append(initial_position)
    velocities.append(initial_velocity)
    observed.append(float(simulation.t))
    steps_done_readbacks.append(int(simulation.steps_done))
    synchronization_readbacks.append(int(simulation.is_synchronized))
    call_deltas: list[int] = []
    previous = 0
    integrated = time.perf_counter()
    for target in indices[1:]:
        delta = target - previous
        simulation.steps(delta)
        previous = target
        call_deltas.append(delta)
        state_position, state_velocity = _simulation_state(simulation)
        positions.append(state_position)
        velocities.append(state_velocity)
        observed.append(float(simulation.t))
        steps_done_readbacks.append(int(simulation.steps_done))
        synchronization_readbacks.append(int(simulation.is_synchronized))
    integration_seconds = time.perf_counter() - integrated
    observed_array = np.asarray(observed, dtype=np.float64)
    replay = _clock_replay(indices, fixed_step)
    if not np.array_equal(observed_array, replay):
        raise BenchmarkError("WHFast observed clock differs from exact two-half replay")
    if tuple(steps_done_readbacks) != indices:
        raise BenchmarkError("WHFast steps_done readbacks differ from integer lattice")
    if any(value != 1 for value in synchronization_readbacks):
        raise BenchmarkError("WHFast checkpoint output is not synchronized")
    if float(simulation.dt_last_done) != fixed_step:
        raise BenchmarkError("WHFast dt_last_done differs from the fixed step")
    accounting = {
        "requested_integer_steps": indices[-1],
        "completed_integer_steps": int(simulation.steps_done),
        "step_api_call_count": len(call_deltas),
        "step_api_call_deltas": call_deltas,
        "steps_done_readbacks": steps_done_readbacks,
        "dt_last_done": float(simulation.dt_last_done),
        "dt_last_done_hex": float(simulation.dt_last_done).hex(),
        "is_synchronized_readbacks": synchronization_readbacks,
        "observed_epoch_hex": [float(value).hex() for value in observed_array],
        "two_half_clock_replay_hex": [float(value).hex() for value in replay],
        "declared_epoch_hex": [float(value).hex() for value in declared],
        "maximum_abs_clock_drift": float(np.max(np.abs(observed_array - declared))),
        "final_clock_drift": float(observed_array[-1] - declared[-1]),
        "final_particles_readback": _particle_readback(simulation),
        "one_step_evidence": _whfast_one_step_evidence(rebound, fixed_step),
        "state_synchronized_at_every_retained_checkpoint": True,
        "exact_finish_time_used": False,
        "integrate_to_endpoint_used": False,
        "evidence_class": "MODEL_OUTPUT",
        "registry_authorized": False,
        "qualification_authorized": False,
    }
    return Lane(
        f"rebound_whfast_{horizon}_p{step_divisor}",
        horizon,
        step_divisor,
        _owned_readonly_float64(declared),
        _owned_readonly_float64(observed_array),
        _owned_readonly_float64(np.stack(positions)),
        _owned_readonly_float64(np.stack(velocities)),
        float(setup_seconds),
        float(integration_seconds),
        settings,
        runtime,
        accounting,
    )


def _ias15_settings(simulation: Any, requested_epochs: np.ndarray) -> dict[str, Any]:
    integrator = simulation.integrator
    effective_initial_dt = float(simulation.dt)
    if effective_initial_dt.hex() != IAS15_INITIAL_DT.hex():
        raise BenchmarkError("IAS15 initial-step readback differs from the locked value")
    return {
        "provenance_categories": {
            "configured_value": [
                "requested_initial_dt",
                "requested_initial_dt_hex",
            ],
            "persistent_effective_readback": [
                "integrator",
                "effective_initial_dt",
                "effective_initial_dt_hex",
                "epsilon",
                "min_dt",
                "adaptive_mode",
                "G",
                "gravity",
                "collision",
                "boundary",
                "softening",
                "N",
                "N_active",
                "testparticle_type",
                "particles_initial_readback",
            ],
            "call_argument": ["integrate(target,exact_finish_time=1)"],
            "environment_observation": ["OMP_NUM_THREADS"],
        },
        "integrator": str(simulation.integrator),
        "requested_initial_dt": float(IAS15_INITIAL_DT),
        "requested_initial_dt_hex": IAS15_INITIAL_DT.hex(),
        "effective_initial_dt": effective_initial_dt,
        "effective_initial_dt_hex": effective_initial_dt.hex(),
        "epsilon": float(integrator.epsilon),
        "min_dt": float(integrator.min_dt),
        "adaptive_mode": str(integrator.adaptive_mode),
        "G": float(simulation.G),
        "gravity": str(simulation.gravity),
        "collision": str(simulation.collision),
        "boundary": str(simulation.boundary),
        "softening": float(simulation.softening),
        "N": int(simulation.N),
        "N_active": int(simulation.N_active),
        "testparticle_type": int(simulation.testparticle_type),
        "particles_initial_readback": _particle_readback(simulation),
        "requested_epoch_hex": [float(value).hex() for value in requested_epochs],
        "exact_finish_time": 1,
        "callbacks": "NONE",
        "thread_observation": {"OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"]},
    }


def _ias15_lane(
    engine_id: str,
    horizon: str,
    requested_epochs: np.ndarray,
    rebound: Any,
    runtime: dict[str, Any],
) -> Lane:
    started = time.perf_counter()
    simulation = _configured_ias15_simulation(rebound)
    settings = _ias15_settings(simulation, requested_epochs)
    setup_seconds = time.perf_counter() - started
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    observed: list[float] = []
    initial_position, initial_velocity = _simulation_state(simulation)
    positions.append(initial_position)
    velocities.append(initial_velocity)
    observed.append(float(simulation.t))
    integrated = time.perf_counter()
    for epoch in requested_epochs[1:]:
        simulation.integrate(float(epoch), exact_finish_time=1)
        state_position, state_velocity = _simulation_state(simulation)
        positions.append(state_position)
        velocities.append(state_velocity)
        observed.append(float(simulation.t))
    integration_seconds = time.perf_counter() - integrated
    observed_array = np.asarray(observed, dtype=np.float64)
    if not np.array_equal(observed_array, requested_epochs):
        raise BenchmarkError("IAS15 did not retain the exact requested epochs")
    accounting = {
        "checkpoint_count": len(requested_epochs),
        "integrate_target_call_count": len(requested_epochs) - 1,
        "exact_finish_time": 1,
        "final_observed_epoch_hex": float(observed_array[-1]).hex(),
        "final_particles_readback": _particle_readback(simulation),
        "evidence_class": "NUMERICAL_REFERENCE_MODEL_OUTPUT",
        "reference_truth_claimed": False,
        "registry_authorized": False,
        "qualification_authorized": False,
    }
    return Lane(
        engine_id,
        horizon,
        None,
        _owned_readonly_float64(requested_epochs),
        _owned_readonly_float64(observed_array),
        _owned_readonly_float64(np.stack(positions)),
        _owned_readonly_float64(np.stack(velocities)),
        float(setup_seconds),
        float(integration_seconds),
        settings,
        runtime,
        accounting,
    )


def _execute_study(profile: Profile, rebound_mode: str) -> StudyRun:
    if type(profile) is not Profile:
        raise BenchmarkError("profile must be an exact Profile")
    profile.__post_init__()
    if type(rebound_mode) is not str or rebound_mode not in REBOUND_MODES:
        raise BenchmarkError(f"rebound_mode must be one of {REBOUND_MODES!r}")
    jx_lanes = (
        _jx_lane(profile, "short", 64),
        _jx_lane(profile, "short", 128),
        _jx_lane(profile, "long", 64),
    )
    negative = _jx_p32_negative_control(profile)
    if rebound_mode == "disabled":
        return StudyRun(
            profile,
            rebound_mode,
            jx_lanes,
            (),
            (),
            negative,
            {
                "requested_mode": rebound_mode,
                "external_lanes_executed": False,
                "reason": "EXPLICITLY_DISABLED",
                "required_version": REQUIRED_REBOUND_VERSION,
                "authority_authorized": False,
            },
        )
    try:
        rebound = _load_rebound()
    except ReboundUnavailable:
        if rebound_mode == "required":
            raise
        return StudyRun(
            profile,
            rebound_mode,
            jx_lanes,
            (),
            (),
            negative,
            {
                "requested_mode": rebound_mode,
                "external_lanes_executed": False,
                "reason": "EXACT_REBOUND_5_1_1_UNAVAILABLE",
                "required_version": REQUIRED_REBOUND_VERSION,
                "authority_authorized": False,
            },
        )
    runtime = _rebound_runtime_provenance(rebound)
    whfast_lanes = (
        _whfast_lane(profile, "short", 64, rebound, runtime),
        _whfast_lane(profile, "short", 128, rebound, runtime),
        _whfast_lane(profile, "long", 64, rebound, runtime),
    )
    if not np.array_equal(
        jx_lanes[0].declared_epochs, jx_lanes[1].declared_epochs
    ):
        raise BenchmarkError("short declared output grids differ by step divisor")
    ias15_lanes = (
        _ias15_lane(
            "rebound_ias15_short_declared_reference",
            "short",
            jx_lanes[0].declared_epochs,
            rebound,
            runtime,
        ),
        _ias15_lane(
            "rebound_ias15_long_declared_reference",
            "long",
            jx_lanes[2].declared_epochs,
            rebound,
            runtime,
        ),
        _ias15_lane(
            "rebound_ias15_short_whfast_p64_observed_reference",
            "short",
            whfast_lanes[0].observed_epochs,
            rebound,
            runtime,
        ),
        _ias15_lane(
            "rebound_ias15_short_whfast_p128_observed_reference",
            "short",
            whfast_lanes[1].observed_epochs,
            rebound,
            runtime,
        ),
        _ias15_lane(
            "rebound_ias15_long_whfast_p64_observed_reference",
            "long",
            whfast_lanes[2].observed_epochs,
            rebound,
            runtime,
        ),
    )
    return StudyRun(
        profile,
        rebound_mode,
        jx_lanes,
        whfast_lanes,
        ias15_lanes,
        negative,
        {
            "requested_mode": rebound_mode,
            "external_lanes_executed": True,
            "reason": "EXACT_REBOUND_5_1_1_LOADED",
            "required_version": REQUIRED_REBOUND_VERSION,
            "authority_authorized": False,
        },
    )


def _lane_equal_ignoring_timing(left: Lane, right: Lane) -> bool:
    if type(left) is not Lane or type(right) is not Lane:
        return False
    try:
        _validate_lane_arrays(left)
        _validate_lane_arrays(right)
    except BenchmarkError:
        return False
    for name in ("engine_id", "horizon", "step_divisor"):
        if type(getattr(left, name)) is not type(getattr(right, name)):
            return False
        if getattr(left, name) != getattr(right, name):
            return False
    for name in ("declared_epochs", "observed_epochs", "positions", "velocities"):
        first = getattr(left, name)
        second = getattr(right, name)
        if (
            type(first) is not np.ndarray
            or type(second) is not np.ndarray
            or first.dtype != second.dtype
            or first.shape != second.shape
            or first.tobytes(order="C") != second.tobytes(order="C")
        ):
            return False
    for name in ("settings", "runtime", "accounting"):
        if not _PROVENANCE_SUPPORT._exact_tree_equal(
            getattr(left, name), getattr(right, name)
        ):
            return False
    for name in ("setup_seconds", "integration_seconds"):
        value = getattr(left, name)
        if type(value) is not float or not math.isfinite(value) or value < 0.0:
            return False
    return True


def _float_hex_json_value(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise BenchmarkError("lane content checksum requires finite floats")
        return {"binary64_hex": value.hex()}
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise BenchmarkError("lane content checksum requires built-in string keys")
        return {key: _float_hex_json_value(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [_float_hex_json_value(item) for item in value]
    raise BenchmarkError(
        f"unsupported lane content checksum type {type(value).__name__!r}"
    )


def _array_float_hex(value: np.ndarray, label: str) -> list[Any]:
    if type(value) is not np.ndarray or value.dtype != np.dtype(np.float64):
        raise BenchmarkError(f"{label} must be an exact float64 ndarray")
    if not np.all(np.isfinite(value)):
        raise BenchmarkError(f"{label} must be finite")
    return _float_hex_json_value(value.tolist())


def _lane_content_payload(lane: Lane) -> dict[str, Any]:
    if type(lane) is not Lane:
        raise BenchmarkError("lane content checksum requires an exact Lane")
    _validate_lane_arrays(lane)
    return {
        "schema": "jx.rebound_whfast.lane_content.v1",
        "engine_id": lane.engine_id,
        "horizon": lane.horizon,
        "step_divisor": lane.step_divisor,
        "declared_epochs": _array_float_hex(
            lane.declared_epochs, "declared_epochs"
        ),
        "observed_epochs": _array_float_hex(
            lane.observed_epochs, "observed_epochs"
        ),
        "positions": _array_float_hex(lane.positions, "positions"),
        "velocities": _array_float_hex(lane.velocities, "velocities"),
        "settings": _float_hex_json_value(lane.settings),
        "runtime": _float_hex_json_value(lane.runtime),
        "accounting": _float_hex_json_value(lane.accounting),
        "timing_fields_excluded": ["setup_seconds", "integration_seconds"],
    }


def _lane_content_sha256(lane: Lane) -> str:
    payload = json.dumps(
        _lane_content_payload(lane),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(
        LANE_CONTENT_CHECKSUM_DOMAIN.encode("ascii") + b"\0" + payload
    ).hexdigest()


def _validate_study_against_replay(primary: StudyRun, replay: StudyRun) -> None:
    if type(primary) is not StudyRun or type(replay) is not StudyRun:
        raise BenchmarkError("study run must be an exact StudyRun")
    if primary.profile != replay.profile or primary.rebound_mode != replay.rebound_mode:
        raise BenchmarkError("study profile or external mode differs from replay")
    if not _PROVENANCE_SUPPORT._exact_tree_equal(
        primary.negative_control, replay.negative_control
    ):
        raise BenchmarkError("JX P/32 negative-domain evidence differs from replay")
    if not _PROVENANCE_SUPPORT._exact_tree_equal(
        primary.external_status, replay.external_status
    ):
        raise BenchmarkError("external lane status differs from replay")
    for label in ("jx_lanes", "whfast_lanes", "ias15_lanes"):
        left = getattr(primary, label)
        right = getattr(replay, label)
        if type(left) is not tuple or type(right) is not tuple or len(left) != len(right):
            raise BenchmarkError(f"{label} roster differs from replay")
        for first, second in zip(left, right):
            if not _lane_equal_ignoring_timing(first, second):
                raise BenchmarkError(f"{label} content differs from independent replay")


def _cartesian_metrics(candidate: Any, reference: Any) -> dict[str, float]:
    first = np.asarray(candidate, dtype=np.float64)
    second = np.asarray(reference, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 3 or first.shape[1:] != (3, 3):
        raise ValueError("Cartesian trajectories must share shape (K,3,3)")
    difference = first - second
    norms = np.linalg.norm(difference, axis=2)
    return {
        "vector_l2_max": float(np.max(norms)),
        "vector_l2_rms": float(np.sqrt(np.mean(norms * norms))),
        "component_abs_max": float(np.max(np.abs(difference))),
        "component_rms": float(np.sqrt(np.mean(difference * difference))),
    }


def _phase_proxy_metrics(
    candidate_positions: Any, reference_positions: Any
) -> dict[str, Any]:
    candidate = np.asarray(candidate_positions, dtype=np.float64)
    reference = np.asarray(reference_positions, dtype=np.float64)
    if candidate.shape != reference.shape or candidate.shape[1:] != (3, 3):
        raise ValueError("phase-proxy trajectories must share shape (K,3,3)")
    bodies: dict[str, dict[str, float]] = {}
    all_differences: list[np.ndarray] = []
    for body_index, body_id in ((1, "INNER"), (2, "OUTER")):
        candidate_relative = candidate[:, body_index, :2] - candidate[:, 0, :2]
        reference_relative = reference[:, body_index, :2] - reference[:, 0, :2]
        candidate_phase = np.arctan2(
            candidate_relative[:, 1], candidate_relative[:, 0]
        )
        reference_phase = np.arctan2(
            reference_relative[:, 1], reference_relative[:, 0]
        )
        difference = np.arctan2(
            np.sin(candidate_phase - reference_phase),
            np.cos(candidate_phase - reference_phase),
        )
        all_differences.append(difference)
        bodies[body_id] = {
            "max_abs_radians": float(np.max(np.abs(difference))),
            "rms_radians": float(np.sqrt(np.mean(difference * difference))),
            "final_wrapped_radians": float(difference[-1]),
        }
    combined = np.concatenate(all_differences)
    return {
        "definition": (
            "WRAPPED_PLANAR_POLAR_ANGLE_OF_BODY_MINUS_PRIMARY_"
            "CANDIDATE_MINUS_IAS15_REFERENCE"
        ),
        "not_osculating_mean_longitude": True,
        "bodies": bodies,
        "max_abs_radians": float(np.max(np.abs(combined))),
        "rms_radians": float(np.sqrt(np.mean(combined * combined))),
    }


def _invariants(positions: np.ndarray, velocities: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    energy = 0.0
    angular = np.zeros(3, dtype=np.float64)
    momentum = np.zeros(3, dtype=np.float64)
    weighted_position = np.zeros(3, dtype=np.float64)
    total_gm = float(np.sum(GRAVITATIONAL_PARAMETERS))
    for body in range(3):
        gm = float(GRAVITATIONAL_PARAMETERS[body])
        energy += 0.5 * gm * float(np.dot(velocities[body], velocities[body]))
        angular += gm * np.cross(positions[body], velocities[body])
        momentum += gm * velocities[body]
        weighted_position += gm * positions[body]
    for left in range(2):
        for right in range(left + 1, 3):
            distance = float(np.linalg.norm(positions[right] - positions[left]))
            energy -= float(
                GRAVITATIONAL_PARAMETERS[left]
                * GRAVITATIONAL_PARAMETERS[right]
            ) / distance
    return energy, angular, weighted_position / total_gm, momentum


def _invariant_metrics(positions: Any, velocities: Any) -> dict[str, float]:
    position_array = np.asarray(positions, dtype=np.float64)
    velocity_array = np.asarray(velocities, dtype=np.float64)
    if position_array.shape != velocity_array.shape or position_array.shape[1:] != (3, 3):
        raise ValueError("invariant trajectories must share shape (K,3,3)")
    energy_0, angular_0, _, _ = _invariants(position_array[0], velocity_array[0])
    angular_scale = float(np.linalg.norm(angular_0))
    energy_errors: list[float] = []
    angular_errors: list[float] = []
    center_norms: list[float] = []
    momentum_norms: list[float] = []
    for position, velocity in zip(position_array, velocity_array):
        energy, angular, center, momentum = _invariants(position, velocity)
        energy_errors.append(abs((energy - energy_0) / energy_0))
        angular_errors.append(float(np.linalg.norm(angular - angular_0) / angular_scale))
        center_norms.append(float(np.linalg.norm(center)))
        momentum_norms.append(float(np.linalg.norm(momentum)))
    return {
        "initial_total_energy": energy_0,
        "initial_angular_momentum_norm": angular_scale,
        "relative_total_energy_max": max(energy_errors),
        "relative_angular_momentum_max": max(angular_errors),
        "center_of_mass_position_max": max(center_norms),
        "total_momentum_max": max(momentum_norms),
    }


def _within_long_envelope(
    position: dict[str, float],
    velocity: dict[str, float],
    phase: dict[str, Any],
    invariants: dict[str, float],
) -> bool:
    return (
        position["vector_l2_max"] <= LONG_DESCRIPTIVE_ENVELOPE["position_vector_l2_max"]
        and velocity["vector_l2_max"] <= LONG_DESCRIPTIVE_ENVELOPE["velocity_vector_l2_max"]
        and phase["max_abs_radians"] <= LONG_DESCRIPTIVE_ENVELOPE["phase_proxy_max_abs_radians"]
        and invariants["relative_total_energy_max"] <= LONG_DESCRIPTIVE_ENVELOPE["relative_total_energy_max"]
        and invariants["relative_angular_momentum_max"] <= LONG_DESCRIPTIVE_ENVELOPE["relative_angular_momentum_max"]
        and invariants["center_of_mass_position_max"] <= LONG_DESCRIPTIVE_ENVELOPE["center_of_mass_position_max"]
        and invariants["total_momentum_max"] <= LONG_DESCRIPTIVE_ENVELOPE["total_momentum_max"]
    )


def _lane_summary(
    lane: Lane,
    observed_reference: Lane | None,
    declared_reference: Lane | None = None,
) -> dict[str, Any]:
    is_jx = lane.engine_id.startswith("jx_wh_")
    timing_scope = (
        "JX_PUBLIC_INTEGRATE_INCLUDES_CORE_PRIMARY_MAP_AND_MANDATORY_"
        "SEMANTIC_REPLAY_EXCLUDES_OUTER_REPORT_VALIDATION_REPLAY"
        if is_jx
        else "EXTERNAL_PRIMARY_EXECUTION_EXCLUDES_OUTER_REPORT_VALIDATION_REPLAY"
    )
    summary: dict[str, Any] = {
        "engine_id": lane.engine_id,
        "horizon": lane.horizon,
        "step_divisor": lane.step_divisor,
        "checkpoint_count": len(lane.declared_epochs),
        "declared_initial_epoch": float(lane.declared_epochs[0]),
        "declared_final_epoch": float(lane.declared_epochs[-1]),
        "observed_initial_epoch": float(lane.observed_epochs[0]),
        "observed_final_epoch": float(lane.observed_epochs[-1]),
        "maximum_abs_clock_drift": float(
            np.max(np.abs(lane.observed_epochs - lane.declared_epochs))
        ),
        "invariant_checkpoint_samples": _invariant_metrics(
            lane.positions, lane.velocities
        ),
        "settings": lane.settings,
        "accounting": lane.accounting,
        "lane_content_integrity": {
            "algorithm": LANE_CONTENT_CHECKSUM_ALGORITHM,
            "domain": LANE_CONTENT_CHECKSUM_DOMAIN,
            "sha256": _lane_content_sha256(lane),
            "scope": (
                "DECLARED_AND_OBSERVED_EPOCH_BINARY64_FLOAT_HEX_VALUES_ALL_"
                "POSITION_VELOCITY_BINARY64_FLOAT_HEX_VALUES_SETTINGS_"
                "RUNTIME_ACCOUNTING_EXCLUDES_RAW_TIMINGS"
            ),
            "unauthenticated": True,
            "authority_authorized": False,
        },
        "raw_timing_seconds": {
            "setup": lane.setup_seconds,
            "integration": lane.integration_seconds,
            "scope": timing_scope,
            "jx_core_mandatory_semantic_replay_included": is_jx,
            "outer_report_validation_replay_included": False,
            "timing_comparable": False,
        },
        "authority_authorized": False,
        "qualification_authorized": False,
    }
    if observed_reference is None:
        summary["ias15_observed_epoch_reference_available"] = False
        summary["observed_epoch_state_error"] = None
        summary["observed_epoch_phase_proxy"] = None
        summary["within_long_descriptive_envelope"] = None
    else:
        if not np.array_equal(lane.observed_epochs, observed_reference.observed_epochs):
            raise BenchmarkError("lane and IAS15 observed-reference epochs differ")
        position = _cartesian_metrics(lane.positions, observed_reference.positions)
        velocity = _cartesian_metrics(lane.velocities, observed_reference.velocities)
        phase = _phase_proxy_metrics(lane.positions, observed_reference.positions)
        summary["ias15_observed_epoch_reference_available"] = True
        summary["observed_epoch_state_error"] = {
            "position": position,
            "velocity": velocity,
            "reference_engine_id": observed_reference.engine_id,
            "reference_truth_claimed": False,
        }
        summary["observed_epoch_phase_proxy"] = phase
        if lane.horizon == "long":
            summary["within_long_descriptive_envelope"] = _within_long_envelope(
                position,
                velocity,
                phase,
                summary["invariant_checkpoint_samples"],
            )
        else:
            summary["within_long_descriptive_envelope"] = None
    if declared_reference is not None:
        if not np.array_equal(lane.declared_epochs, declared_reference.observed_epochs):
            raise BenchmarkError("declared lane grid and IAS15 declared reference differ")
        summary["declared_grid_state_mismatch_without_epoch_relabeling"] = {
            "position": _cartesian_metrics(
                lane.positions, declared_reference.positions
            ),
            "velocity": _cartesian_metrics(
                lane.velocities, declared_reference.velocities
            ),
            "candidate_state_epoch_basis": "OBSERVED_REBOUND_SIMULATION_TIME",
            "reference_epoch_basis": "DECLARED_INTEGER_LATTICE",
            "states_are_not_at_identical_epochs": not np.array_equal(
                lane.observed_epochs, lane.declared_epochs
            ),
            "reference_truth_claimed": False,
        }
    return summary


def _reference_summary(lane: Lane) -> dict[str, Any]:
    return {
        "engine_id": lane.engine_id,
        "horizon": lane.horizon,
        "checkpoint_count": len(lane.observed_epochs),
        "initial_epoch": float(lane.observed_epochs[0]),
        "final_epoch": float(lane.observed_epochs[-1]),
        "invariant_checkpoint_samples": _invariant_metrics(
            lane.positions, lane.velocities
        ),
        "settings": lane.settings,
        "accounting": lane.accounting,
        "lane_content_integrity": {
            "algorithm": LANE_CONTENT_CHECKSUM_ALGORITHM,
            "domain": LANE_CONTENT_CHECKSUM_DOMAIN,
            "sha256": _lane_content_sha256(lane),
            "scope": (
                "DECLARED_AND_OBSERVED_EPOCH_BINARY64_FLOAT_HEX_VALUES_ALL_"
                "POSITION_VELOCITY_BINARY64_FLOAT_HEX_VALUES_SETTINGS_"
                "RUNTIME_ACCOUNTING_EXCLUDES_RAW_TIMINGS"
            ),
            "unauthenticated": True,
            "authority_authorized": False,
        },
        "raw_timing_seconds": {
            "setup": lane.setup_seconds,
            "integration": lane.integration_seconds,
            "scope": (
                "EXTERNAL_PRIMARY_EXECUTION_EXCLUDES_OUTER_REPORT_"
                "VALIDATION_REPLAY"
            ),
            "jx_core_mandatory_semantic_replay_included": False,
            "outer_report_validation_replay_included": False,
            "timing_comparable": False,
        },
        "reference_truth_claimed": False,
        "authority_authorized": False,
        "qualification_authorized": False,
    }


def _ratio(numerator: float, denominator: float) -> float:
    if (
        type(numerator) is not float
        or type(denominator) is not float
        or not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or numerator <= 0.0
        or denominator <= 0.0
    ):
        raise BenchmarkError("convergence ratio inputs must be finite and positive")
    return numerator / denominator


def _convergence_summary(
    coarse: dict[str, Any], fine: dict[str, Any]
) -> dict[str, Any]:
    coarse_state = coarse["observed_epoch_state_error"]
    fine_state = fine["observed_epoch_state_error"]
    if coarse_state is None or fine_state is None:
        raise BenchmarkError("convergence summary requires IAS15 state references")
    ratios = {
        "position_vector_l2_max_ratio_p64_over_p128": _ratio(
            coarse_state["position"]["vector_l2_max"],
            fine_state["position"]["vector_l2_max"],
        ),
        "velocity_vector_l2_max_ratio_p64_over_p128": _ratio(
            coarse_state["velocity"]["vector_l2_max"],
            fine_state["velocity"]["vector_l2_max"],
        ),
        "phase_proxy_max_abs_ratio_p64_over_p128": _ratio(
            coarse["observed_epoch_phase_proxy"]["max_abs_radians"],
            fine["observed_epoch_phase_proxy"]["max_abs_radians"],
        ),
    }
    low, high = SHORT_CONVERGENCE_RATIO_WINDOW
    return {
        "horizon_periods": None,
        "coarse_step": "P/64",
        "fine_step": "P/128",
        "expected_second_order_ratio_window": [low, high],
        "ratios": ratios,
        "each_ratio_within_expected_second_order_window": all(
            low <= value <= high for value in ratios.values()
        ),
        "single_workload_observation_only": True,
        "qualification_authorized": False,
    }


def _lane_by_id(lanes: tuple[Lane, ...], engine_id: str) -> Lane:
    matches = tuple(lane for lane in lanes if lane.engine_id == engine_id)
    if len(matches) != 1:
        raise BenchmarkError(f"lane roster does not contain exactly one {engine_id!r}")
    return matches[0]


def _assert_no_forbidden_keys(value: Any) -> None:
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise BenchmarkError("report mappings require built-in string keys")
            if key.lower() in FORBIDDEN_RESULT_KEYS:
                raise BenchmarkError(f"forbidden comparative result key {key!r}")
            _assert_no_forbidden_keys(item)
    elif type(value) in (list, tuple):
        for item in value:
            _assert_no_forbidden_keys(item)


def _validate_finite_json(value: Any) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise BenchmarkError("report contains a nonfinite float")
        return
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise BenchmarkError("report mappings require built-in string keys")
        for item in value.values():
            _validate_finite_json(item)
        return
    if type(value) in (list, tuple):
        for item in value:
            _validate_finite_json(item)
        return
    raise BenchmarkError(f"report contains unsupported value type {type(value).__name__}")


def build_report(primary: StudyRun) -> dict[str, Any]:
    if type(primary) is not StudyRun:
        raise BenchmarkError("build_report requires an exact StudyRun")
    replay = _execute_study(primary.profile, primary.rebound_mode)
    _validate_study_against_replay(primary, replay)
    for lane in primary.jx_lanes:
        _validate_jx_runtime_provenance(lane.runtime)
    if primary.whfast_lanes or primary.ias15_lanes:
        for lane in primary.whfast_lanes + primary.ias15_lanes:
            _validate_rebound_runtime_provenance(lane.runtime)

    external = primary.external_status["external_lanes_executed"]
    jx_short64 = _lane_by_id(primary.jx_lanes, "jx_wh_short_p64")
    jx_short128 = _lane_by_id(primary.jx_lanes, "jx_wh_short_p128")
    jx_long64 = _lane_by_id(primary.jx_lanes, "jx_wh_long_p64")
    if external:
        wh_short64 = _lane_by_id(
            primary.whfast_lanes, "rebound_whfast_short_p64"
        )
        wh_short128 = _lane_by_id(
            primary.whfast_lanes, "rebound_whfast_short_p128"
        )
        wh_long64 = _lane_by_id(
            primary.whfast_lanes, "rebound_whfast_long_p64"
        )
        ref_short_declared = _lane_by_id(
            primary.ias15_lanes, "rebound_ias15_short_declared_reference"
        )
        ref_long_declared = _lane_by_id(
            primary.ias15_lanes, "rebound_ias15_long_declared_reference"
        )
        ref_short_wh64 = _lane_by_id(
            primary.ias15_lanes,
            "rebound_ias15_short_whfast_p64_observed_reference",
        )
        ref_short_wh128 = _lane_by_id(
            primary.ias15_lanes,
            "rebound_ias15_short_whfast_p128_observed_reference",
        )
        ref_long_wh64 = _lane_by_id(
            primary.ias15_lanes,
            "rebound_ias15_long_whfast_p64_observed_reference",
        )
        lane_summaries = {
            jx_short64.engine_id: _lane_summary(
                jx_short64, ref_short_declared
            ),
            jx_short128.engine_id: _lane_summary(
                jx_short128, ref_short_declared
            ),
            jx_long64.engine_id: _lane_summary(jx_long64, ref_long_declared),
            wh_short64.engine_id: _lane_summary(
                wh_short64, ref_short_wh64, ref_short_declared
            ),
            wh_short128.engine_id: _lane_summary(
                wh_short128, ref_short_wh128, ref_short_declared
            ),
            wh_long64.engine_id: _lane_summary(
                wh_long64, ref_long_wh64, ref_long_declared
            ),
        }
        convergence = {
            "jx_wisdom_holman": _convergence_summary(
                lane_summaries[jx_short64.engine_id],
                lane_summaries[jx_short128.engine_id],
            ),
            "rebound_whfast": _convergence_summary(
                lane_summaries[wh_short64.engine_id],
                lane_summaries[wh_short128.engine_id],
            ),
        }
        for value in convergence.values():
            value["horizon_periods"] = primary.profile.short_periods
        references = {
            lane.engine_id: _reference_summary(lane)
            for lane in primary.ias15_lanes
        }
        rebound_runtime = primary.whfast_lanes[0].runtime
    else:
        lane_summaries = {
            lane.engine_id: _lane_summary(lane, None) for lane in primary.jx_lanes
        }
        convergence = None
        references = {}
        rebound_runtime = None

    initial_payload = _initial_state_payload()
    report = {
        "schema": SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "profile": {
            "profile_id": primary.profile.profile_id,
            "name": primary.profile.name,
            "named_full_profile": primary.profile.named_full_profile,
            "short_periods": primary.profile.short_periods,
            "long_periods": primary.profile.long_periods,
            "samples_per_period": primary.profile.samples_per_period,
            "full_profile_contract": {
                "short_convergence_periods": 10,
                "long_arc_periods": 100,
                "samples_per_period": 4,
                "short_step_divisors": [64, 128],
                "long_step_divisor": 64,
            },
        },
        "problem": {
            "model": "LOCKED_WEAK_THREE_BODY_MUTUAL_NEWTONIAN",
            "G": 1.0,
            "body_order": BODY_IDS,
            "gravitational_parameters": [
                float(value) for value in GRAVITATIONAL_PARAMETERS
            ],
            "period_scale": PERIOD,
            "period_scale_hex": PERIOD.hex(),
            "initial_state": initial_payload,
            "initial_state_hash_algorithm": "SHA256_DOMAIN_SEPARATED_SORTED_JSON_V1",
            "initial_state_hash_domain": INITIAL_STATE_HASH_DOMAIN,
            "initial_state_sha256": INITIAL_STATE_SHA256,
            "initial_state_generated_by_rebound_at_report_time": False,
            "all_bodies_active": True,
            "massless_test_particles": False,
            "physical_radii": [0.0, 0.0, 0.0],
            "frame": "BARYCENTRIC_INERTIAL",
            "origin": "BARYCENTER",
            "time_scale": "TDB",
        },
        "method_provenance": {
            "jx_method_id": FIXED_STEP_WISDOM_HOLMAN_METHOD_ID,
            "rebound_whfast": {
                "classification": "OPTIONAL_EXTERNAL_GPL_V3_FAMILY_COMPARATOR",
                "official_documentation": (
                    "https://rebound.hanno-rein.de/integrators/whfast/"
                ),
                "official_tag": "https://github.com/hannorein/rebound/tree/5.1.1",
                "official_license": (
                    "https://github.com/hannorein/rebound/blob/5.1.1/LICENSE"
                ),
                "installed_metadata_and_project_text_license_inconsistency_is_reported": True,
                "legal_or_redistribution_conclusion_authorized": False,
                "rebound_code_vendored": False,
                "rebound_code_copied": False,
                "jx_core_linked_to_rebound": False,
                "finite_step_map_equivalence_claimed": False,
            },
            "ias15": {
                "role": "HIGH_ACCURACY_EXTERNAL_NUMERICAL_REFERENCE",
                "reference_truth_claimed": False,
            },
        },
        "runtime_provenance": {
            "jx": primary.jx_lanes[0].runtime,
            "rebound": rebound_runtime,
            "external_status": primary.external_status,
        },
        "lanes": lane_summaries,
        "ias15_numerical_references": references,
        "short_convergence": convergence,
        "long_descriptive_envelope": LONG_DESCRIPTIVE_ENVELOPE,
        "negative_domain_control": primary.negative_control,
        "timing_disclosure": {
            "lane_timings_are_raw_single_requested_lane_executions": True,
            "mandatory_report_validation_replay_execution_count_per_requested_lane": 1,
            "report_validation_replay_timings_included_in_lane_timings": False,
            "jx_internal_semantic_replay_is_included_in_jx_integration_timing": True,
            "lane_timings_are_report_generation_wall_time": False,
            "timing_comparable": False,
        },
        "interpretation": {
            "whfast_state_error_uses_ias15_at_observed_simulation_time": True,
            "whfast_declared_grid_mismatch_is_separately_disclosed": True,
            "whfast_state_is_not_relabelled_to_declared_epoch": True,
            "small_sampled_energy_error_does_not_imply_small_phase_or_state_error": True,
            "one_sampled_100_period_run_establishes_general_long_term_boundedness": False,
            "step_ratio_observation_establishes_general_order_or_accuracy": False,
            "step_size_resonance_excluded": False,
            "scientific_use_requires_preregistered_neighboring_step_scan": True,
        },
        "claim_controls": dict(CLAIM_CONTROLS),
        "serialization": {
            "format": "SORTED_FINITE_JSON",
            "byte_determinism_claimed": False,
            "reason": "RAW_TIMINGS_AND_RUNTIME_PATH_DIAGNOSTICS_ARE_INCLUDED",
        },
    }
    _assert_no_forbidden_keys(report)
    _validate_finite_json(report)
    json.dumps(
        report,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    )
    return report


def run_comparison(profile: Profile, rebound_mode: str = "auto") -> dict[str, Any]:
    return build_report(_execute_study(profile, rebound_mode))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare JX ordered-Jacobi Wisdom--Holman with optional exact "
            "REBOUND 5.1.1 WHFast and IAS15 on a locked weak three-body state."
        )
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "full"),
        default="smoke",
        help="full selects the named 10-period convergence and 100-period arc",
    )
    parser.add_argument(
        "--rebound-mode",
        choices=REBOUND_MODES,
        default="auto",
        help="required fails closed unless exact REBOUND 5.1.1 is available",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    profile = Profile.full() if arguments.profile == "full" else Profile()
    report = run_comparison(profile, arguments.rebound_mode)
    text = json.dumps(
        report,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    )
    if arguments.output is None:
        print(text)
    else:
        arguments.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
