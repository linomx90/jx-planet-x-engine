#!/usr/bin/env python3
"""Source-tree-only equal-binary comparison of KDK, RKF78, and REBOUND.

The optional REBOUND 5.1.1 lane is an external GPL-v3-family comparator.
Every lane is measured against an independent Decimal analytic orbit.  Raw
single-run timings are diagnostics only and authorize no comparison or claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import stat
import sys
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from jxplanetx import __version__ as JX_VERSION
import jxplanetx.engine as jx_engine
from jxplanetx.engine import (
    ADAPTIVE_RKF78_METHOD_ID,
    AdaptiveRKF78Spec,
    BackendSpec,
    FIXED_STEP_KDK_METHOD_ID,
    FixedStepKDKSpec,
    ForcePlan,
    KDK_CHECKPOINT_POLICY,
    KDK_COMPOSITION,
    KDK_FORCE_EVALUATION_ACCOUNTING,
    KDK_RESULT_CONTENT_CHECKSUM_ALGORITHM,
    KDK_RESULT_CONTENT_CHECKSUM_DOMAIN,
    KDK_SCHEDULE_CHECKSUM_ALGORITHM,
    KDK_SCHEDULE_CHECKSUM_DOMAIN,
    NewtonianPointMass,
    ParameterMetadata,
    Provenance,
    RKF78_ACCEPTED_STATE_ACCUMULATION,
    RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_ALGORITHM,
    RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_DOMAIN,
    RKF78_ACCEPTED_STEP_MAGNITUDE_SOURCE,
    RKF78_CHECKPOINT_POLICY,
    RKF78_CHECKPOINT_PROPOSAL_POLICY,
    RKF78_TABLEAU_ID,
    RKF78_TIME_STEP_REPRESENTATION,
    StateSnapshot,
    integrate_kdk_trajectory,
    integrate_trajectory,
)


SCHEMA = "jx.rebound_leapfrog.equal_mass_binary.v1"
BENCHMARK_ID = "jx.rebound_leapfrog.newtonian_equal_mass_binary.v1"
REQUIRED_REBOUND_VERSION = "5.1.1"
REBOUND_MODES = ("auto", "required", "disabled")
ENGINE_SOURCE_MANIFEST_SCHEMA = "jxplanetx.engine-source-manifest.v1"
ENGINE_SOURCE_TREE_HASH_DOMAIN = "jxplanetx.engine-source-tree.v1"
REBOUND_SOURCE_MANIFEST_SCHEMA = "rebound.distribution-python-source-manifest.v1"
REBOUND_SOURCE_TREE_HASH_DOMAIN = "rebound.distribution-python-source-tree.v1"
REQUIRED_ENGINE_SOURCE_PATHS = (
    "jxplanetx/engine/__init__.py",
    "jxplanetx/engine/api.py",
    "jxplanetx/engine/backends.py",
    "jxplanetx/engine/catalog.py",
    "jxplanetx/engine/contracts.py",
    "jxplanetx/engine/encounter.py",
    "jxplanetx/engine/encounter_contracts.py",
    "jxplanetx/engine/evaluator.py",
    "jxplanetx/engine/forces.py",
    "jxplanetx/engine/hybrid.py",
    "jxplanetx/engine/hybrid_contracts.py",
    "jxplanetx/engine/rkf78.py",
    "jxplanetx/engine/scenario.py",
    "jxplanetx/engine/scenario_manifest.py",
    "jxplanetx/engine/symplectic.py",
    "jxplanetx/engine/symplectic_contracts.py",
    "jxplanetx/engine/trajectory.py",
    "jxplanetx/engine/trajectory_contracts.py",
    "jxplanetx/engine/wisdom_holman.py",
    "jxplanetx/engine/wisdom_holman_contracts.py",
)
DECIMAL_PRECISION = 100
DECIMAL_EPOCH_ABS_MAX = 1.0e20
DECIMAL_PI_TEXT = (
    "3.141592653589793238462643383279502884197169399375105820974944592307816406"
    "2862089986280348253421170679821480865132823066470938446"
)
PERIOD = 2.0 * float(DECIMAL_PI_TEXT)
FIXED_STEPS_PER_PERIOD = 256
FIXED_STEP = PERIOD / FIXED_STEPS_PER_PERIOD
BODY_IDS = ("BODY_A", "BODY_B")
MASSES = np.array((0.5, 0.5), dtype=np.float64)
ENERGY_0 = -0.125
ANGULAR_MOMENTUM_0 = np.array((0.0, 0.0, 0.25), dtype=np.float64)
FIXED_MAP_ANALYTIC_GATE = {
    "position_vector_l2_max": 0.07,
    "velocity_vector_l2_max": 0.07,
    "phase_error_max_abs_radians": 0.15,
    "relative_total_energy_max": 1.2e-7,
    "relative_angular_momentum_max": 3.0e-14,
    "center_of_mass_position_max": 1.0e-12,
    "total_momentum_max": 1.0e-12,
}
RKF78_REFERENCE_ANALYTIC_GATE = {
    "position_vector_l2_max": 1.0e-7,
    "velocity_vector_l2_max": 1.0e-7,
    "phase_error_max_abs_radians": 2.0e-7,
    "relative_total_energy_max": 1.0e-9,
    "relative_angular_momentum_max": 5.0e-10,
    "center_of_mass_position_max": 1.0e-12,
    "total_momentum_max": 1.0e-12,
}
CLAIM_CONTROLS = {
    "timing_comparable": False,
    "superiority_claimed": False,
    "qualification_claimed": False,
    "reference_truth_claimed": False,
    "production_use_authorized": False,
    "registry_authorized": False,
    "external_comparator_authorizes_jx": False,
}
FORBIDDEN_RESULT_KEYS = frozenset(("winner", "defeated", "passed", "speedup"))


class BenchmarkError(RuntimeError):
    """The comparison contract could not be satisfied exactly."""


class ReboundUnavailable(BenchmarkError):
    """The optional exact-version comparator is not installed."""


@dataclass(frozen=True)
class Profile:
    periods: int = 1
    samples_per_period: int = 16

    def __post_init__(self) -> None:
        if type(self.periods) is not int or self.periods <= 0:
            raise ValueError("periods must be a positive built-in integer")
        if type(self.samples_per_period) is not int or self.samples_per_period <= 0:
            raise ValueError("samples_per_period must be a positive built-in integer")
        if FIXED_STEPS_PER_PERIOD % self.samples_per_period:
            raise ValueError("samples_per_period must divide 256 for the fixed map lattice")

    @property
    def checkpoint_step_indices(self) -> tuple[int, ...]:
        stride = FIXED_STEPS_PER_PERIOD // self.samples_per_period
        count = self.periods * self.samples_per_period
        return tuple(index * stride for index in range(count + 1))

    @property
    def epochs(self) -> tuple[float, ...]:
        return tuple(float(FIXED_STEP * index) for index in self.checkpoint_step_indices)

    @property
    def name(self) -> str:
        if (self.periods, self.samples_per_period) == (100, 4):
            return "full-100-period"
        if (self.periods, self.samples_per_period) == (1, 16):
            return "smoke"
        return "custom"

    @property
    def named_full_profile(self) -> bool:
        return self.name == "full-100-period"

    @property
    def profile_id(self) -> str:
        if self.named_full_profile:
            return "jx.rebound_leapfrog.equal_mass_binary.full_100_period.v1"
        if self.name == "smoke":
            return "jx.rebound_leapfrog.equal_mass_binary.smoke.v1"
        return "jx.rebound_leapfrog.equal_mass_binary.custom.v1"


@dataclass(frozen=True, eq=False)
class Lane:
    engine_id: str
    declared_epochs: np.ndarray
    observed_epochs: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    setup_seconds: float
    integration_seconds: float
    settings: dict[str, Any]
    runtime: dict[str, Any]
    accounting: dict[str, Any]


def _decimal_sin_cos(epoch: float) -> tuple[float, float]:
    if not np.isfinite(epoch):
        raise ValueError("epoch must be finite")
    if abs(float(epoch)) > DECIMAL_EPOCH_ABS_MAX:
        raise ValueError(f"abs(epoch) must not exceed {DECIMAL_EPOCH_ABS_MAX:.1e}")
    try:
        with localcontext() as context:
            context.prec = DECIMAL_PRECISION
            x = Decimal.from_float(float(epoch))
            pi = Decimal(DECIMAL_PI_TEXT)
            two_pi = Decimal(2) * pi
            x %= two_pi
            if x > pi:
                x -= two_pi
            if x < -pi:
                x += two_pi
            cosine_sign = Decimal(1)
            half_pi = pi / Decimal(2)
            if x > half_pi:
                x = pi - x
                cosine_sign = Decimal(-1)
            elif x < -half_pi:
                x = -pi - x
                cosine_sign = Decimal(-1)
            squared = x * x
            sine_term = sine = x
            cosine_term = cosine = Decimal(1)
            index = 1
            while True:
                previous = (sine, cosine)
                sine_term *= -squared / Decimal((2 * index) * (2 * index + 1))
                cosine_term *= -squared / Decimal((2 * index - 1) * (2 * index))
                sine += sine_term
                cosine += cosine_term
                if previous == (sine, cosine):
                    break
                index += 1
            return float(sine), float(cosine_sign * cosine)
    except InvalidOperation as exc:
        raise ValueError("Decimal argument reduction failed for epoch") from exc


def analytic_trajectory(epochs: Any) -> tuple[np.ndarray, np.ndarray]:
    checked = np.asarray(epochs, dtype=np.float64)
    if checked.ndim != 1 or checked.size < 1 or not np.all(np.isfinite(checked)):
        raise ValueError("epochs must be a nonempty finite one-dimensional array")
    values = tuple(_decimal_sin_cos(float(epoch)) for epoch in checked)
    sine = np.array(tuple(value[0] for value in values), dtype=np.float64)[:, None]
    cosine = np.array(tuple(value[1] for value in values), dtype=np.float64)[:, None]
    signs = np.array((-1.0, 1.0), dtype=np.float64)[None, :]
    positions = np.zeros((checked.size, 2, 3), dtype=np.float64)
    velocities = np.zeros_like(positions)
    positions[:, :, 0] = 0.5 * signs * cosine
    positions[:, :, 1] = 0.5 * signs * sine
    velocities[:, :, 0] = -0.5 * signs * sine
    velocities[:, :, 1] = 0.5 * signs * cosine
    positions[positions == 0.0] = 0.0
    velocities[velocities == 0.0] = 0.0
    return positions, velocities


def _trajectory_arrays(positions: Any, velocities: Any) -> tuple[np.ndarray, np.ndarray]:
    checked = (
        np.asarray(positions, dtype=np.float64),
        np.asarray(velocities, dtype=np.float64),
    )
    if (
        checked[0].shape != checked[1].shape
        or checked[0].ndim != 3
        or checked[0].shape[1:] != (2, 3)
        or not all(np.all(np.isfinite(value)) for value in checked)
    ):
        raise ValueError("trajectory arrays must be finite with shape (count, 2, 3)")
    return checked


def cartesian_metrics(candidate: Any, reference: Any) -> dict[str, float]:
    candidate_array = np.asarray(candidate, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if (
        candidate_array.shape != reference_array.shape
        or candidate_array.ndim != 3
        or candidate_array.shape[1:] != (2, 3)
    ):
        raise ValueError("Cartesian arrays must share shape (checkpoint_count, 2, 3)")
    difference = candidate_array - reference_array
    if not np.all(np.isfinite(difference)):
        raise ValueError("Cartesian differences must be finite")
    vector_norm = np.linalg.norm(difference, axis=2)
    return {
        "component_max_abs": float(np.max(np.abs(difference))),
        "component_rms": float(np.sqrt(np.mean(difference * difference))),
        "vector_l2_max": float(np.max(vector_norm)),
        "vector_l2_rms": float(np.sqrt(np.mean(vector_norm * vector_norm))),
    }


def phase_error_metrics(
    candidate: Any,
    reference: Any,
    *,
    candidate_label: str = "candidate",
    reference_label: str = "reference",
) -> dict[str, Any]:
    for value, label in (
        (candidate_label, "candidate_label"),
        (reference_label, "reference_label"),
    ):
        if type(value) is not str or not value or value.strip() != value:
            raise ValueError(f"{label} must be an explicit trimmed string")
    candidate_array = np.asarray(candidate, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if candidate_array.shape != reference_array.shape or candidate_array.shape[1:] != (2, 3):
        raise ValueError("phase arrays must share shape (checkpoint_count, 2, 3)")
    candidate_relative = candidate_array[:, 1, :2] - candidate_array[:, 0, :2]
    reference_relative = reference_array[:, 1, :2] - reference_array[:, 0, :2]
    if (
        not np.all(np.isfinite(candidate_relative))
        or not np.all(np.isfinite(reference_relative))
        or np.any(np.linalg.norm(candidate_relative, axis=1) <= 0.0)
        or np.any(np.linalg.norm(reference_relative, axis=1) <= 0.0)
    ):
        raise ValueError("relative planar positions must be finite and nonzero")
    cross = reference_relative[:, 0] * candidate_relative[:, 1]
    cross -= reference_relative[:, 1] * candidate_relative[:, 0]
    dot = np.sum(reference_relative * candidate_relative, axis=1)
    unwrapped = np.unwrap(np.arctan2(cross, dot))
    return {
        "interpretation": (
            "UNWRAPPED_SIGNED_CANDIDATE_MINUS_REFERENCE_RELATIVE_ORBIT_ANGLE_"
            "IN_XY_PLANE_FOR_THIS_NONCOLLIDING_WORKLOAD"
        ),
        "candidate_label": candidate_label,
        "reference_label": reference_label,
        "maximum_abs_radians": float(np.max(np.abs(unwrapped))),
        "rms_radians": float(np.sqrt(np.mean(unwrapped * unwrapped))),
        "final_signed_radians": float(unwrapped[-1]),
    }


def invariant_metrics(positions: Any, velocities: Any) -> dict[str, float]:
    position, velocity = _trajectory_arrays(positions, velocities)
    separation = np.linalg.norm(position[:, 1] - position[:, 0], axis=1)
    if np.any(separation <= 0.0):
        raise ValueError("binary separation must remain positive")
    energy = 0.5 * np.sum(MASSES[None, :, None] * velocity * velocity, axis=(1, 2))
    energy -= MASSES[0] * MASSES[1] / separation
    angular = np.sum(MASSES[None, :, None] * np.cross(position, velocity), axis=1)
    center = np.sum(MASSES[None, :, None] * position, axis=1) / np.sum(MASSES)
    momentum = np.sum(MASSES[None, :, None] * velocity, axis=1)
    series = (
        ("relative_total_energy", np.abs((energy - ENERGY_0) / abs(ENERGY_0))),
        (
            "relative_angular_momentum",
            np.linalg.norm(angular - ANGULAR_MOMENTUM_0, axis=1)
            / np.linalg.norm(ANGULAR_MOMENTUM_0),
        ),
        ("center_of_mass_position", np.linalg.norm(center, axis=1)),
        ("total_momentum", np.linalg.norm(momentum, axis=1)),
    )
    result: dict[str, float] = {}
    for label, values in series:
        result[f"{label}_max"] = float(np.max(values))
        result[f"{label}_rms"] = float(np.sqrt(np.mean(values * values)))
        result[f"{label}_final"] = float(values[-1])
    return result


def _digest(*arrays: Any) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        checked = np.ascontiguousarray(value, dtype="<f8")
        digest.update(str(checked.shape).encode("ascii"))
        digest.update(checked.tobytes())
    return digest.hexdigest()


def _exact_tree_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return all(type(key) is str for key in actual) and set(actual) == set(expected) and all(
            type(key) is str and _exact_tree_equal(actual[key], expected[key])
            for key in expected
        )
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            _exact_tree_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    if type(expected) is tuple:
        return len(actual) == len(expected) and all(
            _exact_tree_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    if type(expected) is float:
        return actual.hex() == expected.hex()
    return bool(actual == expected)


def _file_identity(path: Any) -> dict[str, Any]:
    if path is None:
        raise BenchmarkError("runtime source path is unavailable")
    checked = Path(os.path.abspath(os.fspath(path)))
    try:
        mode = checked.lstat().st_mode
    except OSError as exc:
        raise BenchmarkError(f"runtime source is unavailable: {checked}") from exc
    if stat.S_ISLNK(mode):
        raise BenchmarkError(f"runtime source must not be a symlink: {checked}")
    if not stat.S_ISREG(mode):
        raise BenchmarkError(f"runtime source must be a regular file: {checked}")
    data = checked.read_bytes()
    return {
        "path": str(checked),
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _engine_source_tree_sha256(files: Any) -> str:
    checked = list(files)
    for entry in checked:
        if type(entry) is not dict or set(entry) != {"path", "size_bytes", "sha256"}:
            raise BenchmarkError("engine source manifest entry schema is invalid")
    paths = [entry["path"] for entry in checked]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise BenchmarkError("engine source paths must be unique, normalized, and sorted")
    payload = json.dumps(
        checked,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(
        ENGINE_SOURCE_TREE_HASH_DOMAIN.encode("ascii") + b"\0" + payload
    ).hexdigest()


def _engine_sources_manifest() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    expected_directory = (root / "src" / "jxplanetx" / "engine").resolve()
    package_init = Path(os.path.abspath(os.fspath(jx_engine.__file__)))
    package_directory = package_init.parent
    if package_directory != expected_directory:
        raise BenchmarkError("the comparison must load jxplanetx.engine from this source tree")
    files: list[dict[str, Any]] = []
    for source in sorted(package_directory.glob("*.py"), key=lambda path: path.name):
        identity = _file_identity(source)
        files.append(
            {
                "path": f"jxplanetx/engine/{source.name}",
                "size_bytes": identity["size_bytes"],
                "sha256": identity["sha256"],
            }
        )
    roster = tuple(entry["path"] for entry in files)
    if roster != REQUIRED_ENGINE_SOURCE_PATHS:
        raise BenchmarkError("the source-tree engine roster differs from the locked manifest")
    loaded_paths: list[str] = []
    for name, module in tuple(sys.modules.items()):
        if name != "jxplanetx.engine" and not name.startswith("jxplanetx.engine."):
            continue
        module_path = getattr(module, "__file__", None)
        if module_path is None:
            raise BenchmarkError(f"loaded engine module has no source path: {name}")
        path = Path(os.path.abspath(os.fspath(module_path)))
        if path.suffix == ".py" and path.parent == package_directory:
            loaded_paths.append(f"jxplanetx/engine/{path.name}")
    loaded_paths = sorted(set(loaded_paths))
    if not set(loaded_paths).issubset(roster):
        raise BenchmarkError("a loaded engine source is absent from the manifest")
    return {
        "schema": ENGINE_SOURCE_MANIFEST_SCHEMA,
        "selection": "EXACT_TOP_LEVEL_SOURCE_TREE_PYTHON_ROSTER",
        "classification": "PROVENANCE_DIAGNOSTIC",
        "authority_authorized": False,
        "source_tree_root": str(root),
        "package_directory": str(package_directory),
        "files": files,
        "file_count": len(files),
        "loaded_source_paths": loaded_paths,
        "tree_hash_algorithm": "SHA256_DOMAIN_SEPARATED_SORTED_JSON_V1",
        "tree_hash_domain": ENGINE_SOURCE_TREE_HASH_DOMAIN,
        "tree_sha256": _engine_source_tree_sha256(files),
    }


def _validate_engine_sources_manifest(manifest: Any) -> None:
    expected = _engine_sources_manifest()
    if manifest != expected:
        raise BenchmarkError("engine source manifest does not match current source-tree bytes")


def _numpy_runtime_identity() -> dict[str, Any]:
    multiarray = importlib.import_module("numpy._core._multiarray_umath")
    linalg = importlib.import_module("numpy.linalg._umath_linalg")
    return {
        "classification": "PROVENANCE_DIAGNOSTIC",
        "authority_authorized": False,
        "identity_scope": (
            "PACKAGE_MODULE_MULTIARRAY_UMATH_AND_LINALG_NATIVE_BINARIES"
        ),
        "version": np.__version__,
        "package_module": _file_identity(np.__file__),
        "multiarray_umath_native_binary": _file_identity(multiarray.__file__),
        "linalg_native_binary": _file_identity(linalg.__file__),
    }


def _jx_runtime_provenance() -> dict[str, Any]:
    return {
        "classification": "PROVENANCE_DIAGNOSTIC",
        "authority_authorized": False,
        "jxplanetx_version": JX_VERSION,
        "engine_sources": _engine_sources_manifest(),
        "benchmark_script": _file_identity(__file__),
        "numpy": _numpy_runtime_identity(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def _validate_jx_runtime_provenance(runtime: Any) -> None:
    if not _exact_tree_equal(runtime, _jx_runtime_provenance()):
        raise BenchmarkError("JX runtime provenance does not match loaded source bytes")


def _state_and_plan(profile: Profile) -> tuple[StateSnapshot, ForcePlan]:
    script_digest = _file_identity(__file__)["sha256"]
    provenance = Provenance(
        "jx.benchmark.rebound_leapfrog.equal_binary.v1",
        "benchmarks/rebound_leapfrog_comparison.py",
        "1",
        script_digest,
    )
    positions, velocities = analytic_trajectory((0.0,))
    snapshot = StateSnapshot(
        "jx.benchmark.rebound_leapfrog.equal_binary.snapshot.v1",
        0.0,
        "SYNTHETIC",
        "BARYCENTRIC_INERTIAL",
        "BARYCENTER",
        "CARTESIAN_RIGHT_HANDED",
        "L",
        "T",
        "M",
        "jx.benchmark.synthetic.v1",
        BODY_IDS,
        positions[0].copy(),
        velocities[0].copy(),
        MASSES.copy(),
        MASSES.copy(),
        np.zeros(2, dtype=np.float64),
        np.ones(2, dtype=np.bool_),
        provenance,
    )
    metadata = ParameterMetadata(
        "state.gravitational_parameters",
        "L^3/T^2",
        provenance,
        None,
        None,
        0.0,
        profile.epochs[-1],
    )
    plan = ForcePlan(
        "jx.benchmark.rebound_leapfrog.equal_binary.plan.v1",
        BackendSpec("numpy", "cpu", 2),
        (
            NewtonianPointMass(
                BODY_IDS,
                BODY_IDS,
                "jx.benchmark.synthetic.v1",
                (metadata,),
            ),
        ),
    )
    return snapshot, plan


def _kdk_settings() -> dict[str, Any]:
    return {
        "method_classification": "FIXED_STEP_KDK_MAP",
        "method_variant": "JX_KICK_DRIFT_KICK",
        "equivalent_to_rebound_leapfrog_map_claimed": False,
        "backend": "numpy",
        "device": "cpu",
        "dtype": "float64",
        "force_model": NewtonianPointMass.MODEL_ID,
        "fixed_step": FIXED_STEP,
        "fixed_steps_per_period": FIXED_STEPS_PER_PERIOD,
        "minimum_swept_pair_separation": 0.1,
        "maximum_pair_frequency_step": 0.05,
        "checkpoint_policy": KDK_CHECKPOINT_POLICY,
    }


def _rkf78_settings() -> dict[str, Any]:
    return {
        "lane_role": "NUMERICAL_REFERENCE_LANE_NOT_REFERENCE_TRUTH",
        "method_classification": "ADAPTIVE_RKF78",
        "backend": "numpy",
        "device": "cpu",
        "dtype": "float64",
        "force_model": NewtonianPointMass.MODEL_ID,
        "initial_step": PERIOD / 128.0,
        "minimum_step": 1.0e-15,
        "maximum_step": PERIOD / 16.0,
        "position_atol": 1.0e-13,
        "position_rtol": 1.0e-12,
        "velocity_atol": 1.0e-13,
        "velocity_rtol": 1.0e-12,
        "maximum_steps": 2_000_000,
        "maximum_rejections": 200_000,
        "safety_factor": 0.9,
        "minimum_scale_factor": 0.2,
        "maximum_scale_factor": 5.0,
        "checkpoint_policy": RKF78_CHECKPOINT_POLICY,
    }


def _jx_kdk_lane(profile: Profile) -> Lane:
    started = time.perf_counter()
    snapshot, plan = _state_and_plan(profile)
    spec = FixedStepKDKSpec(
        checkpoint_step_indices=profile.checkpoint_step_indices,
        fixed_step=FIXED_STEP,
        maximum_steps=profile.checkpoint_step_indices[-1],
        minimum_swept_pair_separation=0.1,
        maximum_pair_frequency_step=0.05,
    )
    setup_seconds = time.perf_counter() - started
    integrated = time.perf_counter()
    result = integrate_kdk_trajectory(snapshot, plan, spec)
    integration_seconds = time.perf_counter() - integrated
    settings = _kdk_settings()
    accounting = {
        "custody_source": "KDKTrajectoryResult",
        "completed_steps": result.completed_steps,
        "force_evaluations": result.force_evaluations,
        "checkpoint_count": result.checkpoint_count,
        "method_id": result.method_id,
        "composition": result.composition,
        "force_evaluation_accounting": result.force_evaluation_accounting,
        "schedule_checksum_algorithm": result.schedule_checksum_algorithm,
        "schedule_checksum_domain": result.schedule_checksum_domain,
        "schedule_content_sha256": result.schedule_content_sha256,
        "result_content_checksum_algorithm": result.result_content_checksum_algorithm,
        "result_content_checksum_domain": result.result_content_checksum_domain,
        "result_content_sha256": result.result_content_sha256,
        "minimum_observed_pair_separations": list(
            result.minimum_observed_pair_separations
        ),
        "minimum_observed_swept_pair_separation": (
            result.minimum_observed_swept_pair_separation
        ),
        "maximum_observed_pair_frequency_step": (
            result.maximum_observed_pair_frequency_step
        ),
        "exact_arithmetic_symplectic": result.exact_arithmetic_symplectic,
        "floating_point_symplectic": result.floating_point_symplectic,
        "evidence_class": result.evidence_class,
        "registry_authorized": result.registry_authorized,
        "qualification_authorized": result.qualification_authorized,
    }
    epochs = np.asarray(result.checkpoint_epochs, dtype=np.float64)
    return Lane(
        "jx_kdk",
        epochs,
        epochs.copy(),
        np.stack(result.positions),
        np.stack(result.velocities),
        setup_seconds,
        integration_seconds,
        settings,
        _jx_runtime_provenance(),
        accounting,
    )


def _jx_rkf78_lane(profile: Profile) -> Lane:
    started = time.perf_counter()
    snapshot, plan = _state_and_plan(profile)
    position_atol = np.full((2, 3), 1.0e-13, dtype=np.float64)
    velocity_atol = position_atol.copy()
    spec = AdaptiveRKF78Spec(
        profile.epochs,
        initial_step=PERIOD / 128.0,
        minimum_step=1.0e-15,
        maximum_step=PERIOD / 16.0,
        position_atol=position_atol,
        position_rtol=1.0e-12,
        velocity_atol=velocity_atol,
        velocity_rtol=1.0e-12,
        maximum_steps=2_000_000,
        maximum_rejections=200_000,
        safety_factor=0.9,
        minimum_scale_factor=0.2,
        maximum_scale_factor=5.0,
    )
    setup_seconds = time.perf_counter() - started
    integrated = time.perf_counter()
    result = integrate_trajectory(snapshot, plan, spec)
    integration_seconds = time.perf_counter() - integrated
    settings = _rkf78_settings()
    accounting = {
        "custody_source": "TrajectoryResult",
        "attempted_steps": result.attempted_steps,
        "accepted_steps": result.accepted_steps,
        "rejected_steps": result.rejected_steps,
        "force_evaluations": result.force_evaluations,
        "checkpoint_count": result.checkpoint_count,
        "method_id": result.method_id,
        "tableau_id": result.tableau_id,
        "accepted_state_accumulation": result.accepted_state_accumulation,
        "time_step_representation": result.time_step_representation,
        "accepted_step_magnitude_source": result.accepted_step_magnitude_source,
        "checkpoint_proposal_policy": result.checkpoint_proposal_policy,
        "accepted_step_ledger": {
            "epoch_count": len(result.accepted_step_epochs),
            "magnitude_count": len(result.accepted_step_magnitudes),
            "minimum_magnitude": result.minimum_accepted_step,
            "maximum_magnitude": result.maximum_accepted_step,
            "last_magnitude": result.last_accepted_step,
            "magnitude_fsum": math.fsum(result.accepted_step_magnitudes),
            "epoch_span": abs(result.checkpoint_epochs[-1] - result.checkpoint_epochs[0]),
            "checksum_algorithm": result.accepted_step_ledger_checksum_algorithm,
            "checksum_domain": result.accepted_step_ledger_checksum_domain,
            "content_integrity_sha256": result.accepted_step_ledger_content_sha256,
        },
        "evidence_class": result.evidence_class,
        "registry_authorized": result.registry_authorized,
        "qualification_authorized": result.qualification_authorized,
    }
    epochs = np.asarray(result.checkpoint_epochs, dtype=np.float64)
    return Lane(
        "jx_rkf78_adaptive_reference",
        epochs,
        epochs.copy(),
        np.stack(result.positions),
        np.stack(result.velocities),
        setup_seconds,
        integration_seconds,
        settings,
        _jx_runtime_provenance(),
        accounting,
    )


def _validate_rebound_module(module: Any) -> None:
    version = getattr(module, "__version__", None)
    if version != REQUIRED_REBOUND_VERSION:
        raise BenchmarkError(
            f"REBOUND {REQUIRED_REBOUND_VERSION} is required exactly; found {version!r}"
        )
    if not callable(getattr(module, "Simulation", None)):
        raise BenchmarkError("REBOUND 5.1.1 does not expose Simulation")


def _load_rebound() -> Any:
    if os.environ.get("OMP_NUM_THREADS") not in (None, "1"):
        raise BenchmarkError("OMP_NUM_THREADS must be unset or exactly '1'")
    if "rebound" in sys.modules and os.environ.get("OMP_NUM_THREADS") != "1":
        raise BenchmarkError("REBOUND was imported before single-thread mode was established")
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        module = importlib.import_module("rebound")
    except ModuleNotFoundError as exc:
        if exc.name == "rebound":
            raise ReboundUnavailable("REBOUND 5.1.1 is not installed") from exc
        raise BenchmarkError("the REBOUND installation has a missing dependency") from exc
    _validate_rebound_module(module)
    return module


def _distribution_file(distribution: Any, suffix: str) -> Path:
    matches = [
        Path(distribution.locate_file(entry))
        for entry in distribution.files or ()
        if str(entry).endswith(suffix)
    ]
    if len(matches) != 1:
        raise BenchmarkError(f"REBOUND distribution must expose exactly one {suffix}")
    return matches[0]


def _distribution_recorded_file(distribution: Any, relative_path: str) -> Path:
    matches = [
        Path(distribution.locate_file(entry))
        for entry in distribution.files or ()
        if str(entry) == relative_path
    ]
    if len(matches) != 1:
        raise BenchmarkError(
            f"REBOUND distribution must record exactly one {relative_path}"
        )
    return Path(os.path.abspath(os.fspath(matches[0])))


def _rebound_distribution_binding(rebound: Any) -> tuple[Any, dict[str, Path]]:
    distribution = importlib.metadata.distribution("rebound")
    if distribution.metadata.get("Name", "").lower() != "rebound":
        raise BenchmarkError("the installed distribution name is not REBOUND")
    if distribution.version != REQUIRED_REBOUND_VERSION:
        raise BenchmarkError("the installed REBOUND distribution version is not 5.1.1")
    module_path = _distribution_recorded_file(distribution, "rebound/__init__.py")
    library_entries = [
        str(entry)
        for entry in distribution.files or ()
        if PurePosixPath(str(entry)).parent == PurePosixPath(".")
        and PurePosixPath(str(entry)).name.startswith("librebound.")
    ]
    if len(library_entries) != 1:
        raise BenchmarkError("REBOUND distribution must record one root librebound binary")
    library_path = _distribution_recorded_file(distribution, library_entries[0])
    imported_module_path = Path(os.path.abspath(os.fspath(rebound.__file__)))
    library_name = getattr(getattr(rebound, "clibrebound", None), "_name", None)
    if library_name is None:
        library_name = getattr(rebound, "__libpath__", None)
    if library_name is None:
        raise BenchmarkError("the loaded REBOUND native library path is unavailable")
    imported_library_path = Path(os.path.abspath(os.fspath(library_name)))
    if imported_module_path != module_path:
        raise BenchmarkError(
            "the imported REBOUND module is not the distribution-recorded module"
        )
    if imported_library_path != library_path:
        raise BenchmarkError(
            "the loaded REBOUND library is not the distribution-recorded native binary"
        )
    return distribution, {
        "module": module_path,
        "library": library_path,
        "metadata": _distribution_file(distribution, ".dist-info/METADATA"),
        "license": _distribution_file(
            distribution, ".dist-info/licenses/LICENSE"
        ),
    }


def _rebound_source_tree_sha256(files: Any) -> str:
    checked = list(files)
    for entry in checked:
        if type(entry) is not dict or set(entry) != {"path", "size_bytes", "sha256"}:
            raise BenchmarkError("REBOUND source manifest entry schema is invalid")
    paths = [entry["path"] for entry in checked]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise BenchmarkError("REBOUND source paths must be unique and sorted")
    payload = json.dumps(
        checked,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(
        REBOUND_SOURCE_TREE_HASH_DOMAIN.encode("ascii") + b"\0" + payload
    ).hexdigest()


def _rebound_python_sources_manifest(distribution: Any) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    absolute_by_relative: dict[str, Path] = {}
    for entry in distribution.files or ():
        relative = str(entry)
        normalized = PurePosixPath(relative)
        if (
            relative.startswith("/")
            or "\\" in relative
            or ".." in normalized.parts
            or normalized.as_posix() != relative
        ):
            raise BenchmarkError("REBOUND distribution contains a nonnormalized path")
        if not relative.startswith("rebound/") or not relative.endswith(".py"):
            continue
        path = Path(os.path.abspath(os.fspath(distribution.locate_file(entry))))
        identity = _file_identity(path)
        entries.append(
            {
                "path": relative,
                "size_bytes": identity["size_bytes"],
                "sha256": identity["sha256"],
            }
        )
        absolute_by_relative[relative] = path
    entries.sort(key=lambda item: item["path"])
    required = (
        "rebound/__init__.py",
        "rebound/integrator.py",
        "rebound/particle.py",
        "rebound/particles.py",
        "rebound/simulation.py",
    )
    roster = tuple(entry["path"] for entry in entries)
    if not set(required).issubset(roster):
        raise BenchmarkError("REBOUND distribution omits a required Python wrapper")
    recorded_absolute = {path: relative for relative, path in absolute_by_relative.items()}
    for name, module in tuple(sys.modules.items()):
        if name != "rebound" and not name.startswith("rebound."):
            continue
        source = getattr(module, "__file__", None)
        if source is None or not str(source).endswith(".py"):
            continue
        loaded = Path(os.path.abspath(os.fspath(source)))
        if loaded not in recorded_absolute:
            raise BenchmarkError(
                f"loaded REBOUND wrapper is outside the recorded distribution: {name}"
            )
    return {
        "schema": REBOUND_SOURCE_MANIFEST_SCHEMA,
        "selection": "ALL_DISTRIBUTION_RECORDED_REBOUND_PACKAGE_PYTHON_SOURCES",
        "classification": "PROVENANCE_DIAGNOSTIC",
        "authority_authorized": False,
        "required_execution_wrappers": list(required),
        "files": entries,
        "file_count": len(entries),
        "tree_hash_algorithm": "SHA256_DOMAIN_SEPARATED_SORTED_JSON_V1",
        "tree_hash_domain": REBOUND_SOURCE_TREE_HASH_DOMAIN,
        "tree_sha256": _rebound_source_tree_sha256(entries),
    }


def _rebound_runtime_provenance(rebound: Any) -> dict[str, Any]:
    distribution, paths = _rebound_distribution_binding(rebound)
    metadata_path = paths["metadata"]
    license_path = paths["license"]
    metadata_text = metadata_path.read_text(encoding="utf-8")
    runtime = {
        "classification": "OPTIONAL_EXTERNAL_GPL_V3_FAMILY_COMPARATOR",
        "authority_authorized": False,
        "distribution_name": distribution.metadata.get("Name"),
        "distribution_version": distribution.version,
        "distribution_root": str(
            Path(os.path.abspath(os.fspath(distribution.locate_file(""))))
        ),
        "rebound_version": rebound.__version__,
        "rebound_githash": getattr(rebound, "__githash__", None),
        "rebound_build": getattr(rebound, "__build__", None),
        "module_source": _file_identity(paths["module"]),
        "c_library": _file_identity(paths["library"]),
        "python_sources": _rebound_python_sources_manifest(distribution),
        "distribution_metadata": _file_identity(metadata_path),
        "distribution_license_file": _file_identity(license_path),
        "installed_license_expression": distribution.metadata.get(
            "License-Expression"
        ),
        "project_description_mentions_v3_or_later": (
            "either version 3" in metadata_text and "any later version" in metadata_text
        ),
        "license_text_inconsistency_observed": (
            distribution.metadata.get("License-Expression") == "GPL-3.0-only"
            and "either version 3" in metadata_text
            and "any later version" in metadata_text
        ),
        "official_repository": "https://github.com/hannorein/rebound",
        "official_license_page": (
            "https://github.com/hannorein/rebound/blob/5.1.1/LICENSE"
        ),
        "legal_or_redistribution_conclusion_authorized": False,
        "rebound_code_vendored": False,
        "rebound_code_copied": False,
        "jx_core_linked_to_rebound": False,
        "omp_num_threads": os.environ["OMP_NUM_THREADS"],
    }
    for field in ("rebound_githash", "rebound_build", "installed_license_expression"):
        if type(runtime[field]) is not str or not runtime[field]:
            raise BenchmarkError(f"REBOUND runtime field {field} is unavailable")
    return runtime


def _validate_rebound_runtime_provenance(runtime: Any) -> None:
    if type(runtime) is not dict:
        raise BenchmarkError("REBOUND runtime provenance schema is invalid")
    expected = _rebound_runtime_provenance(_load_rebound())
    if not _exact_tree_equal(runtime, expected):
        raise BenchmarkError(
            "REBOUND runtime provenance does not match the loaded distribution bytes"
        )


def _configured_rebound_simulation(rebound: Any) -> Any:
    simulation = rebound.Simulation()
    simulation.G = 1.0
    simulation.gravity = "basic"
    simulation.collision = "none"
    simulation.boundary = "none"
    simulation.softening = 0.0
    simulation.integrator = "leapfrog"
    simulation.dt = FIXED_STEP
    simulation.add(
        m=0.5, r=0.0, x=-0.5, y=0.0, z=0.0, vx=0.0, vy=-0.5, vz=0.0
    )
    simulation.add(
        m=0.5, r=0.0, x=0.5, y=0.0, z=0.0, vx=0.0, vy=0.5, vz=0.0
    )
    simulation.N_active = 2
    simulation.testparticle_type = 0
    if not callable(getattr(simulation, "steps", None)):
        raise BenchmarkError("REBOUND 5.1.1 does not expose Simulation.steps")
    return simulation


def _rebound_particles_readback(simulation: Any) -> list[dict[str, float]]:
    return [
        {
            field: float(getattr(particle, field))
            for field in ("m", "r", "x", "y", "z", "vx", "vy", "vz")
        }
        for particle in simulation.particles[:2]
    ]


def _independent_rebound_dkd_evidence(rebound: Any) -> dict[str, Any]:
    simulation = _configured_rebound_simulation(rebound)
    positions = np.array(((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)), dtype=np.float64)
    velocities = np.array(((0.0, -0.5, 0.0), (0.0, 0.5, 0.0)), dtype=np.float64)
    half_positions = positions + (0.5 * FIXED_STEP) * velocities
    displacement = half_positions[1] - half_positions[0]
    squared_distance = float(np.dot(displacement, displacement))
    inverse_cube = 1.0 / (squared_distance * math.sqrt(squared_distance))
    acceleration = np.empty_like(positions)
    acceleration[0] = 0.5 * displacement * inverse_cube
    acceleration[1] = -0.5 * displacement * inverse_cube
    expected_velocities = velocities + FIXED_STEP * acceleration
    expected_positions = half_positions + (0.5 * FIXED_STEP) * expected_velocities
    simulation.steps(1)
    observed_positions = np.array(
        tuple((particle.x, particle.y, particle.z) for particle in simulation.particles[:2]),
        dtype=np.float64,
    )
    observed_velocities = np.array(
        tuple((particle.vx, particle.vy, particle.vz) for particle in simulation.particles[:2]),
        dtype=np.float64,
    )
    position_equal = bool(np.array_equal(expected_positions, observed_positions))
    velocity_equal = bool(np.array_equal(expected_velocities, observed_velocities))
    if not position_equal or not velocity_equal:
        raise BenchmarkError("REBOUND leapfrog failed the independent one-step DKD oracle")
    return {
        "classification": "INDEPENDENT_ONE_STEP_BINARY64_DKD_ORACLE",
        "implementation": "BENCHMARK_LOCAL_EQUATIONS_NO_REBOUND_SOURCE_COPIED",
        "step_count": 1,
        "bitwise_position_equal": position_equal,
        "bitwise_velocity_equal": velocity_equal,
        "expected_state_sha256": _digest(expected_positions, expected_velocities),
        "observed_state_sha256": _digest(observed_positions, observed_velocities),
        "claim_authorized": False,
    }


def _rebound_settings(simulation: Any, evidence: dict[str, Any]) -> dict[str, Any]:
    integrator = simulation.integrator
    return {
        "G": float(simulation.G),
        "gravity": str(simulation.gravity),
        "collision": str(simulation.collision),
        "boundary": str(simulation.boundary),
        "softening": float(simulation.softening),
        "integrator": str(integrator),
        "integrator_order": int(integrator.order),
        "method_variant": "REBOUND_LEAPFROG_DRIFT_KICK_DRIFT",
        "equivalent_to_jx_kdk_map_claimed": False,
        "dt": float(simulation.dt),
        "fixed_steps_per_period": FIXED_STEPS_PER_PERIOD,
        "N": int(simulation.N),
        "N_active": int(simulation.N_active),
        "testparticle_type": int(simulation.testparticle_type),
        "initial_particle_effective_readback": _rebound_particles_readback(simulation),
        "advance_api": "Simulation.steps(integer_count)",
        "official_api_index": "https://rebound.hanno-rein.de/api/",
        "official_simulation_api_documentation": (
            "https://rebound.hanno-rein.de/simulation/"
        ),
        "official_integrator_documentation": (
            "https://rebound.hanno-rein.de/integrators/leapfrog/"
        ),
        "independent_one_step_method_evidence": evidence,
        "thread_count": int(os.environ["OMP_NUM_THREADS"]),
        "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
        "provenance": {
            "G": "effective_readback",
            "gravity": "effective_readback",
            "collision": "effective_readback",
            "boundary": "effective_readback",
            "softening": "effective_readback",
            "integrator": "effective_readback",
            "integrator_order": "effective_readback",
            "dt": "effective_readback",
            "N": "effective_readback",
            "N_active": "effective_readback",
            "testparticle_type": "effective_readback",
            "initial_particle_effective_readback": "effective_readback",
            "advance_api": "call_contract",
            "official_api_index": "external_official_documentation_url",
            "official_simulation_api_documentation": (
                "external_official_documentation_url"
            ),
            "official_integrator_documentation": (
                "external_official_documentation_url"
            ),
            "method_variant": "external_documentation_and_independent_one_step_oracle",
            "independent_one_step_method_evidence": (
                "independent_benchmark_equations_and_effective_readback"
            ),
            "thread_count": "environment_observation",
            "OMP_NUM_THREADS": "environment_observation",
        },
    }


def _rebound_leapfrog_lane(profile: Profile) -> Lane:
    started = time.perf_counter()
    rebound = _load_rebound()
    simulation = _configured_rebound_simulation(rebound)
    settings = _rebound_settings(
        simulation, _independent_rebound_dkd_evidence(rebound)
    )
    setup_seconds = time.perf_counter() - started
    checkpoint_count = len(profile.epochs)
    positions = np.empty((checkpoint_count, 2, 3), dtype=np.float64)
    velocities = np.empty_like(positions)
    observed_epochs = np.empty(checkpoint_count, dtype=np.float64)

    def capture(index: int) -> None:
        observed_epochs[index] = float(simulation.t)
        for body_index, particle in enumerate(simulation.particles[:2]):
            positions[index, body_index] = (particle.x, particle.y, particle.z)
            velocities[index, body_index] = (particle.vx, particle.vy, particle.vz)

    capture(0)
    step_arguments: list[int] = []
    integrated = time.perf_counter()
    previous = profile.checkpoint_step_indices[0]
    for output_index, current in enumerate(profile.checkpoint_step_indices[1:], 1):
        count = current - previous
        if type(count) is not int or count <= 0:
            raise BenchmarkError("REBOUND step-call arguments must be positive integers")
        simulation.steps(count)
        step_arguments.append(count)
        capture(output_index)
        previous = current
    integration_seconds = time.perf_counter() - integrated
    requested_steps = sum(step_arguments)
    if requested_steps != profile.checkpoint_step_indices[-1]:
        raise BenchmarkError("REBOUND integer step accounting is inconsistent")
    accounting = {
        "custody_source": "BENCHMARK_RECORDED_INTEGER_SIMULATION_STEPS_ARGUMENTS",
        "step_api": "Simulation.steps",
        "step_call_count": len(step_arguments),
        "step_call_arguments": step_arguments,
        "requested_integer_steps": requested_steps,
        "completed_integer_steps": requested_steps,
        "checkpoint_count": checkpoint_count,
        "force_evaluations_exposed": False,
        "observed_clock_source": "Simulation.t_effective_readback_after_each_steps_call",
        "registry_authorized": False,
        "qualification_authorized": False,
    }
    return Lane(
        "rebound_leapfrog",
        np.asarray(profile.epochs, dtype=np.float64),
        observed_epochs,
        positions,
        velocities,
        setup_seconds,
        integration_seconds,
        settings,
        _rebound_runtime_provenance(rebound),
        accounting,
    )


def _is_lower_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_lane_arrays(lane: Any, profile: Profile, engine_id: str) -> None:
    if type(lane) is not Lane or type(lane.engine_id) is not str:
        raise BenchmarkError(f"{engine_id} lane container is invalid")
    if lane.engine_id != engine_id:
        raise BenchmarkError(f"{engine_id} lane engine identity is invalid")
    expected_epochs = np.asarray(profile.epochs, dtype=np.float64)
    expected_shape = (len(expected_epochs), 2, 3)
    for field in ("declared_epochs", "observed_epochs", "positions", "velocities"):
        value = getattr(lane, field)
        if type(value) is not np.ndarray or value.dtype != np.dtype(np.float64):
            raise BenchmarkError(f"{engine_id} lane {field} must be a float64 ndarray")
        if not np.all(np.isfinite(value)):
            raise BenchmarkError(f"{engine_id} lane {field} must be finite")
    if lane.declared_epochs.shape != expected_epochs.shape or not np.array_equal(
        lane.declared_epochs, expected_epochs
    ):
        raise BenchmarkError(f"{engine_id} lane declared checkpoint grid is invalid")
    if lane.observed_epochs.shape != expected_epochs.shape:
        raise BenchmarkError(f"{engine_id} lane observed epoch shape is invalid")
    if lane.positions.shape != expected_shape or lane.velocities.shape != expected_shape:
        raise BenchmarkError(f"{engine_id} lane trajectory shape is invalid")
    initial_positions, initial_velocities = analytic_trajectory((0.0,))
    if _digest(lane.positions[0]) != _digest(initial_positions[0]):
        raise BenchmarkError(f"{engine_id} lane initial positions are invalid")
    if _digest(lane.velocities[0]) != _digest(initial_velocities[0]):
        raise BenchmarkError(f"{engine_id} lane initial velocities are invalid")
    for field in ("setup_seconds", "integration_seconds"):
        value = getattr(lane, field)
        if type(value) is not float or not math.isfinite(value) or value < 0.0:
            raise BenchmarkError(f"{engine_id} lane timing diagnostic is invalid")
    for field in ("settings", "runtime", "accounting"):
        if type(getattr(lane, field)) is not dict:
            raise BenchmarkError(f"{engine_id} lane {field} schema is invalid")


def _validate_kdk_lane(lane: Lane, profile: Profile) -> None:
    _validate_lane_arrays(lane, profile, "jx_kdk")
    if not np.array_equal(lane.observed_epochs, lane.declared_epochs):
        raise BenchmarkError("JX KDK observed epochs must equal the integer lattice")
    if not _exact_tree_equal(lane.settings, _kdk_settings()):
        raise BenchmarkError("JX KDK settings differ from the locked lane")
    accounting = lane.accounting
    expected_keys = {
        "custody_source",
        "completed_steps",
        "force_evaluations",
        "checkpoint_count",
        "method_id",
        "composition",
        "force_evaluation_accounting",
        "schedule_checksum_algorithm",
        "schedule_checksum_domain",
        "schedule_content_sha256",
        "result_content_checksum_algorithm",
        "result_content_checksum_domain",
        "result_content_sha256",
        "minimum_observed_pair_separations",
        "minimum_observed_swept_pair_separation",
        "maximum_observed_pair_frequency_step",
        "exact_arithmetic_symplectic",
        "floating_point_symplectic",
        "evidence_class",
        "registry_authorized",
        "qualification_authorized",
    }
    if set(accounting) != expected_keys:
        raise BenchmarkError("JX KDK accounting schema is invalid")
    expected_fixed = {
        "custody_source": "KDKTrajectoryResult",
        "completed_steps": profile.checkpoint_step_indices[-1],
        "force_evaluations": profile.checkpoint_step_indices[-1] + 1,
        "checkpoint_count": len(profile.epochs),
        "method_id": FIXED_STEP_KDK_METHOD_ID,
        "composition": KDK_COMPOSITION,
        "force_evaluation_accounting": KDK_FORCE_EVALUATION_ACCOUNTING,
        "schedule_checksum_algorithm": KDK_SCHEDULE_CHECKSUM_ALGORITHM,
        "schedule_checksum_domain": KDK_SCHEDULE_CHECKSUM_DOMAIN,
        "result_content_checksum_algorithm": KDK_RESULT_CONTENT_CHECKSUM_ALGORITHM,
        "result_content_checksum_domain": KDK_RESULT_CONTENT_CHECKSUM_DOMAIN,
        "exact_arithmetic_symplectic": True,
        "floating_point_symplectic": False,
        "evidence_class": "MODEL_OUTPUT",
        "registry_authorized": False,
        "qualification_authorized": False,
    }
    if any(
        not _exact_tree_equal(accounting[field], value)
        for field, value in expected_fixed.items()
    ):
        raise BenchmarkError("JX KDK accounting values are invalid")
    if not _is_lower_sha256(accounting["schedule_content_sha256"]) or not _is_lower_sha256(
        accounting["result_content_sha256"]
    ):
        raise BenchmarkError("JX KDK content checksums are invalid")
    separations = accounting["minimum_observed_pair_separations"]
    if (
        type(separations) is not list
        or len(separations) != 1
        or type(separations[0]) is not float
        or not math.isfinite(separations[0])
        or separations[0] <= lane.settings["minimum_swept_pair_separation"]
    ):
        raise BenchmarkError("JX KDK pair-separation accounting is invalid")
    swept = accounting["minimum_observed_swept_pair_separation"]
    frequency = accounting["maximum_observed_pair_frequency_step"]
    if (
        type(swept) is not float
        or swept.hex() != separations[0].hex()
        or type(frequency) is not float
        or not math.isfinite(frequency)
        or frequency < 0.0
        or frequency > lane.settings["maximum_pair_frequency_step"]
    ):
        raise BenchmarkError("JX KDK guard accounting is invalid")


def _validate_rkf78_lane(lane: Lane, profile: Profile) -> None:
    _validate_lane_arrays(lane, profile, "jx_rkf78_adaptive_reference")
    if not np.array_equal(lane.observed_epochs, lane.declared_epochs):
        raise BenchmarkError("JX RKF78 observed epochs must equal its checkpoints")
    if not _exact_tree_equal(lane.settings, _rkf78_settings()):
        raise BenchmarkError("JX RKF78 settings differ from the locked lane")
    accounting = lane.accounting
    expected_keys = {
        "custody_source",
        "attempted_steps",
        "accepted_steps",
        "rejected_steps",
        "force_evaluations",
        "checkpoint_count",
        "method_id",
        "tableau_id",
        "accepted_state_accumulation",
        "time_step_representation",
        "accepted_step_magnitude_source",
        "checkpoint_proposal_policy",
        "accepted_step_ledger",
        "evidence_class",
        "registry_authorized",
        "qualification_authorized",
    }
    if set(accounting) != expected_keys:
        raise BenchmarkError("JX RKF78 accounting schema is invalid")
    for field in (
        "attempted_steps",
        "accepted_steps",
        "rejected_steps",
        "force_evaluations",
        "checkpoint_count",
    ):
        if type(accounting[field]) is not int:
            raise BenchmarkError(f"JX RKF78 {field} must be a built-in integer")
    if (
        accounting["accepted_steps"] <= 0
        or accounting["rejected_steps"] < 0
        or accounting["attempted_steps"]
        != accounting["accepted_steps"] + accounting["rejected_steps"]
        or accounting["force_evaluations"] != 13 * accounting["attempted_steps"]
        or accounting["checkpoint_count"] != len(profile.epochs)
    ):
        raise BenchmarkError("JX RKF78 step accounting is invalid")
    expected_fixed = {
        "custody_source": "TrajectoryResult",
        "method_id": ADAPTIVE_RKF78_METHOD_ID,
        "tableau_id": RKF78_TABLEAU_ID,
        "accepted_state_accumulation": RKF78_ACCEPTED_STATE_ACCUMULATION,
        "time_step_representation": RKF78_TIME_STEP_REPRESENTATION,
        "accepted_step_magnitude_source": RKF78_ACCEPTED_STEP_MAGNITUDE_SOURCE,
        "checkpoint_proposal_policy": RKF78_CHECKPOINT_PROPOSAL_POLICY,
        "evidence_class": "MODEL_OUTPUT",
        "registry_authorized": False,
        "qualification_authorized": False,
    }
    if any(
        not _exact_tree_equal(accounting[field], value)
        for field, value in expected_fixed.items()
    ):
        raise BenchmarkError("JX RKF78 fixed accounting provenance is invalid")
    ledger = accounting["accepted_step_ledger"]
    ledger_keys = {
        "epoch_count",
        "magnitude_count",
        "minimum_magnitude",
        "maximum_magnitude",
        "last_magnitude",
        "magnitude_fsum",
        "epoch_span",
        "checksum_algorithm",
        "checksum_domain",
        "content_integrity_sha256",
    }
    if type(ledger) is not dict or set(ledger) != ledger_keys:
        raise BenchmarkError("JX RKF78 accepted-step ledger schema is invalid")
    for field in ("epoch_count", "magnitude_count"):
        if type(ledger[field]) is not int or ledger[field] != accounting["accepted_steps"]:
            raise BenchmarkError("JX RKF78 accepted-step ledger counts are invalid")
    for field in (
        "minimum_magnitude",
        "maximum_magnitude",
        "last_magnitude",
        "magnitude_fsum",
        "epoch_span",
    ):
        if type(ledger[field]) is not float or not math.isfinite(ledger[field]):
            raise BenchmarkError("JX RKF78 accepted-step ledger values are invalid")
    expected_span = abs(profile.epochs[-1] - profile.epochs[0])
    if (
        ledger["minimum_magnitude"] <= 0.0
        or ledger["minimum_magnitude"] > ledger["maximum_magnitude"]
        or not ledger["minimum_magnitude"]
        <= ledger["last_magnitude"]
        <= ledger["maximum_magnitude"]
        or ledger["maximum_magnitude"] > lane.settings["maximum_step"]
        or ledger["epoch_span"].hex() != expected_span.hex()
        or abs(ledger["magnitude_fsum"] - expected_span)
        > 2.0 * math.ulp(expected_span)
        or ledger["checksum_algorithm"]
        != RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_ALGORITHM
        or ledger["checksum_domain"] != RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_DOMAIN
        or not _is_lower_sha256(ledger["content_integrity_sha256"])
    ):
        raise BenchmarkError("JX RKF78 accepted-step ledger is invalid")


def _validate_rebound_lane(lane: Lane, profile: Profile) -> None:
    _validate_lane_arrays(lane, profile, "rebound_leapfrog")
    if lane.observed_epochs[0].hex() != 0.0.hex() or np.any(
        np.diff(lane.observed_epochs) <= 0.0
    ):
        raise BenchmarkError("REBOUND observed simulation epochs are invalid")
    rebound = _load_rebound()
    expected_settings = _rebound_settings(
        _configured_rebound_simulation(rebound),
        _independent_rebound_dkd_evidence(rebound),
    )
    if not _exact_tree_equal(lane.settings, expected_settings):
        raise BenchmarkError("REBOUND settings differ from effective readback")
    _validate_rebound_runtime_provenance(lane.runtime)
    arguments = [
        right - left
        for left, right in zip(
            profile.checkpoint_step_indices[:-1],
            profile.checkpoint_step_indices[1:],
            strict=True,
        )
    ]
    expected_accounting = {
        "custody_source": "BENCHMARK_RECORDED_INTEGER_SIMULATION_STEPS_ARGUMENTS",
        "step_api": "Simulation.steps",
        "step_call_count": len(arguments),
        "step_call_arguments": arguments,
        "requested_integer_steps": profile.checkpoint_step_indices[-1],
        "completed_integer_steps": profile.checkpoint_step_indices[-1],
        "checkpoint_count": len(profile.epochs),
        "force_evaluations_exposed": False,
        "observed_clock_source": (
            "Simulation.t_effective_readback_after_each_steps_call"
        ),
        "registry_authorized": False,
        "qualification_authorized": False,
    }
    if not _exact_tree_equal(lane.accounting, expected_accounting):
        raise BenchmarkError("REBOUND integer step accounting is invalid")
    replay = _configured_rebound_simulation(rebound)
    expected_observed_epochs = np.empty(len(profile.epochs), dtype=np.float64)
    expected_observed_epochs[0] = float(replay.t)
    for index, count in enumerate(arguments, 1):
        replay.steps(count)
        expected_observed_epochs[index] = float(replay.t)
    if _digest(lane.observed_epochs) != _digest(expected_observed_epochs):
        raise BenchmarkError("REBOUND observed clock is not the exact steps replay readback")


def _validate_lane_against_independent_replay(
    claimed: Lane, replayed: Lane, label: str
) -> None:
    if type(claimed) is not Lane or type(replayed) is not Lane:
        raise BenchmarkError(f"{label} replay lane container is invalid")
    if type(claimed.engine_id) is not str or claimed.engine_id != replayed.engine_id:
        raise BenchmarkError(f"{label} replay engine identity differs")
    for field in ("declared_epochs", "observed_epochs", "positions", "velocities"):
        left = getattr(claimed, field)
        right = getattr(replayed, field)
        if (
            type(left) is not np.ndarray
            or type(right) is not np.ndarray
            or left.dtype != right.dtype
            or left.shape != right.shape
            or left.tobytes(order="C") != right.tobytes(order="C")
        ):
            raise BenchmarkError(f"{label} {field} differs from independent replay")
    for field in ("settings", "runtime", "accounting"):
        if not _exact_tree_equal(getattr(claimed, field), getattr(replayed, field)):
            raise BenchmarkError(f"{label} {field} differs from independent replay")


def _within_analytic_workload_gate(
    position: dict[str, float],
    velocity: dict[str, float],
    phase: dict[str, Any],
    invariants: dict[str, float],
    thresholds: dict[str, float],
) -> bool:
    observed = {
        "position_vector_l2_max": position["vector_l2_max"],
        "velocity_vector_l2_max": velocity["vector_l2_max"],
        "phase_error_max_abs_radians": phase["maximum_abs_radians"],
        "relative_total_energy_max": invariants["relative_total_energy_max"],
        "relative_angular_momentum_max": invariants[
            "relative_angular_momentum_max"
        ],
        "center_of_mass_position_max": invariants["center_of_mass_position_max"],
        "total_momentum_max": invariants["total_momentum_max"],
    }
    return all(
        observed[field] <= threshold
        for field, threshold in thresholds.items()
    )


def _lane_report(
    lane: Lane,
    declared_oracle_positions: np.ndarray,
    declared_oracle_velocities: np.ndarray,
) -> dict[str, Any]:
    declared = np.asarray(lane.declared_epochs, dtype=np.float64)
    observed = np.asarray(lane.observed_epochs, dtype=np.float64)
    if (
        declared.ndim != 1
        or observed.shape != declared.shape
        or not np.all(np.isfinite(declared))
        or not np.all(np.isfinite(observed))
    ):
        raise BenchmarkError("lane epoch arrays are invalid")
    _trajectory_arrays(lane.positions, lane.velocities)
    observed_oracle_positions, observed_oracle_velocities = analytic_trajectory(observed)
    declared_position = cartesian_metrics(lane.positions, declared_oracle_positions)
    declared_velocity = cartesian_metrics(lane.velocities, declared_oracle_velocities)
    declared_phase = phase_error_metrics(
        lane.positions,
        declared_oracle_positions,
        candidate_label=lane.engine_id,
        reference_label="analytic_declared_grid_orbit",
    )
    observed_position = cartesian_metrics(lane.positions, observed_oracle_positions)
    observed_velocity = cartesian_metrics(lane.velocities, observed_oracle_velocities)
    observed_phase = phase_error_metrics(
        lane.positions,
        observed_oracle_positions,
        candidate_label=lane.engine_id,
        reference_label="analytic_observed_clock_orbit",
    )
    invariants = invariant_metrics(lane.positions, lane.velocities)
    clock_residual = observed - declared
    if lane.engine_id == "jx_rkf78_adaptive_reference":
        gate_id = "RKF78_ADAPTIVE_REFERENCE_EQUAL_BINARY_V1"
        gate_thresholds = RKF78_REFERENCE_ANALYTIC_GATE
    else:
        gate_id = "FIXED_MAP_EQUAL_BINARY_P_OVER_256_V1"
        gate_thresholds = FIXED_MAP_ANALYTIC_GATE
    return {
        "engine_id": lane.engine_id,
        "settings": lane.settings,
        "runtime": lane.runtime,
        "accounting": lane.accounting,
        "timing_seconds": {
            "measurement_count": 1,
            "scope": "PRIMARY_LANE_EXECUTION_ONLY",
            "setup": lane.setup_seconds,
            "integration": lane.integration_seconds,
            "total": lane.setup_seconds + lane.integration_seconds,
            "mandatory_validation_replay_execution_count": 1,
            "mandatory_validation_replay_timing_included": False,
            "represents_report_generation_wall_time": False,
            "comparability_authorized": False,
        },
        "trajectory_sha256": _digest(
            lane.declared_epochs,
            lane.observed_epochs,
            lane.positions,
            lane.velocities,
        ),
        "clock_alignment": {
            "declared_grid_epochs": declared.tolist(),
            "observed_simulation_epochs": observed.tolist(),
            "declared_grid_sha256": _digest(declared),
            "observed_simulation_sha256": _digest(observed),
            "maximum_abs_observed_minus_declared": float(
                np.max(np.abs(clock_residual))
            ),
            "final_observed_minus_declared": float(clock_residual[-1]),
        },
        "declared_grid_oracle_errors": {
            "position": declared_position,
            "velocity": declared_velocity,
            "phase": declared_phase,
        },
        "observed_clock_oracle_errors": {
            "position": observed_position,
            "velocity": observed_velocity,
            "phase": observed_phase,
        },
        "invariant_envelopes": invariants,
        "analytic_workload_gate_basis": (
            "COMMON_DECLARED_OUTPUT_CHECKPOINT_GRID_AND_ANALYTIC_EQUAL_BINARY_ONLY"
            if lane.engine_id == "jx_rkf78_adaptive_reference"
            else "DECLARED_INTEGER_MAP_LATTICE_AND_ANALYTIC_EQUAL_BINARY_ONLY"
        ),
        "analytic_workload_gate_id": gate_id,
        "within_analytic_workload_gate": _within_analytic_workload_gate(
            declared_position,
            declared_velocity,
            declared_phase,
            invariants,
            gate_thresholds,
        ),
    }


def _pair_disagreement(left: Lane, right: Lane) -> dict[str, Any]:
    return {
        "classification": (
            "SAME_DECLARED_OUTPUT_CHECKPOINT_INDEX_STATE_DISAGREEMENT_"
            "NO_MAP_EQUIVALENCE_CLAIM"
        ),
        "position": cartesian_metrics(left.positions, right.positions),
        "velocity": cartesian_metrics(left.velocities, right.velocities),
        "relative_orbit_phase": phase_error_metrics(
            left.positions,
            right.positions,
            candidate_label=left.engine_id,
            reference_label=right.engine_id,
        ),
    }


def build_report(
    profile: Profile,
    kdk_lane: Lane,
    rkf78_lane: Lane,
    rebound_lane: Lane | None,
    *,
    rebound_unavailability_reason: str | None = None,
) -> dict[str, Any]:
    declared_epochs = np.asarray(profile.epochs, dtype=np.float64)
    _validate_kdk_lane(kdk_lane, profile)
    _validate_rkf78_lane(rkf78_lane, profile)
    _validate_jx_runtime_provenance(kdk_lane.runtime)
    _validate_jx_runtime_provenance(rkf78_lane.runtime)
    _validate_lane_against_independent_replay(
        kdk_lane, _jx_kdk_lane(profile), "JX KDK"
    )
    _validate_lane_against_independent_replay(
        rkf78_lane, _jx_rkf78_lane(profile), "JX RKF78"
    )
    if rebound_lane is not None:
        _validate_rebound_lane(rebound_lane, profile)
        _validate_lane_against_independent_replay(
            rebound_lane, _rebound_leapfrog_lane(profile), "REBOUND leapfrog"
        )
    oracle_positions, oracle_velocities = analytic_trajectory(declared_epochs)
    lanes: dict[str, Any] = {
        "jx_kdk": _lane_report(
            kdk_lane, oracle_positions, oracle_velocities
        ),
        "jx_rkf78_adaptive_reference": _lane_report(
            rkf78_lane, oracle_positions, oracle_velocities
        ),
    }
    disagreements: dict[str, Any] = {
        "jx_kdk_vs_jx_rkf78_adaptive_reference": _pair_disagreement(
            kdk_lane, rkf78_lane
        )
    }
    if rebound_lane is None:
        lanes["rebound_leapfrog"] = {
            "engine_id": "rebound_leapfrog",
            "available": False,
            "classification": "OPTIONAL_EXTERNAL_GPL_V3_FAMILY_COMPARATOR",
            "unavailability_reason": rebound_unavailability_reason,
            "authority_authorized": False,
        }
    else:
        rebound_report = _lane_report(
            rebound_lane, oracle_positions, oracle_velocities
        )
        rebound_report["available"] = True
        rebound_report["classification"] = (
            "OPTIONAL_EXTERNAL_GPL_V3_FAMILY_COMPARATOR"
        )
        lanes["rebound_leapfrog"] = rebound_report
        disagreements["jx_kdk_vs_rebound_leapfrog"] = _pair_disagreement(
            kdk_lane, rebound_lane
        )
        disagreements[
            "jx_rkf78_adaptive_reference_vs_rebound_leapfrog"
        ] = _pair_disagreement(rkf78_lane, rebound_lane)
    report = {
        "schema": SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "profile": {
            "name": profile.name,
            "profile_id": profile.profile_id,
            "periods": profile.periods,
            "samples_per_period": profile.samples_per_period,
            "named_full_profile": profile.named_full_profile,
            "fixed_steps_per_period": FIXED_STEPS_PER_PERIOD,
        },
        "problem": {
            "model": "pure_newtonian_equal_mass_barycentric_binary",
            "G": 1.0,
            "body_ids": list(BODY_IDS),
            "masses_and_gravitational_parameters": MASSES.tolist(),
            "initial_positions": oracle_positions[0].tolist(),
            "initial_velocities": oracle_velocities[0].tolist(),
            "period": PERIOD,
            "analytic_total_energy": ENERGY_0,
            "analytic_total_angular_momentum": ANGULAR_MOMENTUM_0.tolist(),
            "analytic_oracle": {
                "implementation": "STDLIB_DECIMAL_ARGUMENT_REDUCED_TAYLOR",
                "decimal_precision": DECIMAL_PRECISION,
                "maximum_absolute_epoch": DECIMAL_EPOCH_ABS_MAX,
                "epoch_input": "EXACT_BINARY64_VIA_DECIMAL_FROM_FLOAT",
                "zero_sign_policy": (
                    "CANONICAL_POSITIVE_ZERO_FOR_EXACT_MATHEMATICAL_ZERO"
                ),
                "pi": DECIMAL_PI_TEXT,
                "reference_truth_claimed": False,
            },
        },
        "checkpoints": {
            "count": len(declared_epochs),
            "fixed_map_step_indices": list(profile.checkpoint_step_indices),
            "declared_grid_epochs": declared_epochs.tolist(),
            "declared_grid_sha256": _digest(declared_epochs),
            "declared_output_epochs_shared_by_all_executed_lanes": True,
            "fixed_map_step_indices_shared_by_jx_kdk_and_rebound": (
                rebound_lane is not None
            ),
            "adaptive_rkf78_integration_steps_share_fixed_map_indices": False,
            "rebound_clock_is_separately_observed": rebound_lane is not None,
        },
        "lanes": lanes,
        "cross_lane_disagreement": disagreements,
        "analytic_workload_gates": {
            "fixed_map_p_over_256": {
                "scope": "THIS_ANALYTIC_EQUAL_MASS_BINARY_WORKLOAD_ONLY",
                "applies_to": ["jx_kdk", "rebound_leapfrog"],
                "thresholds": dict(FIXED_MAP_ANALYTIC_GATE),
                "basis": (
                    "DECLARED_INTEGER_MAP_LATTICE_ORACLE_ERRORS_AND_INVARIANTS"
                ),
                "claim_authorized": False,
            },
            "rkf78_adaptive_reference": {
                "scope": "THIS_ANALYTIC_EQUAL_MASS_BINARY_WORKLOAD_ONLY",
                "applies_to": ["jx_rkf78_adaptive_reference"],
                "thresholds": dict(RKF78_REFERENCE_ANALYTIC_GATE),
                "basis": "DECLARED_CHECKPOINT_ORACLE_ERRORS_AND_INVARIANTS",
                "claim_authorized": False,
            },
        },
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "benchmark_script": _file_identity(__file__),
            "engine_source_tree_sha256": kdk_lane.runtime["engine_sources"][
                "tree_sha256"
            ],
        },
        "scientific_interpretation": {
            "invariant_maxima_sampling_scope": "RETAINED_OUTPUT_CHECKPOINTS_ONLY",
            "small_sampled_energy_envelope_implies_small_phase_error": False,
            "small_sampled_energy_envelope_implies_small_trajectory_error": False,
            "general_long_term_boundedness_claimed": False,
            "single_sampled_100_period_run_establishes_general_behavior": False,
            "phase_and_state_errors_are_reported_separately_from_invariants": True,
        },
        "claim_controls": dict(CLAIM_CONTROLS),
        "serialization": {
            "format": "JSON",
            "object_keys_sorted": True,
            "nonfinite_numbers_allowed": False,
            "includes_timing_and_path_diagnostics": True,
            "deterministic_report_bytes_claimed": False,
        },
    }
    keys: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            keys.update(value)
            for nested in value.values():
                collect(nested)
        elif isinstance(value, list):
            for nested in value:
                collect(nested)

    collect(report)
    if keys & FORBIDDEN_RESULT_KEYS:
        raise BenchmarkError("report contains forbidden comparative vocabulary")
    json.dumps(report, sort_keys=True, allow_nan=False)
    return report


def run(profile: Profile, rebound_mode: str = "auto") -> dict[str, Any]:
    if rebound_mode not in REBOUND_MODES:
        raise ValueError(f"rebound_mode must be one of {REBOUND_MODES!r}")
    kdk_lane = _jx_kdk_lane(profile)
    rkf78_lane = _jx_rkf78_lane(profile)
    rebound_lane: Lane | None = None
    unavailable: str | None = None
    if rebound_mode != "disabled":
        try:
            rebound_lane = _rebound_leapfrog_lane(profile)
        except ReboundUnavailable as exc:
            if rebound_mode == "required":
                raise
            unavailable = str(exc)
    else:
        unavailable = "disabled_by_cli"
    return build_report(
        profile,
        kdk_lane,
        rkf78_lane,
        rebound_lane,
        rebound_unavailability_reason=unavailable,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure JX KDK, JX adaptive RKF78, and optional exact REBOUND "
            "5.1.1 leapfrog on an analytic equal-mass binary."
        )
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=1,
        help="Positive orbit count; use 100 for the named full profile.",
    )
    parser.add_argument(
        "--samples-per-period",
        type=int,
        default=16,
        help="Positive divisor of 256; use 4 with 100 periods for full profile.",
    )
    parser.add_argument(
        "--rebound-mode",
        choices=REBOUND_MODES,
        default="auto",
        help="Auto-detect, require, or disable the exact 5.1.1 external comparator.",
    )
    parser.add_argument(
        "--output", type=Path, help="Write sorted finite JSON here instead of stdout."
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        rendered = json.dumps(
            run(Profile(args.periods, args.samples_per_period), args.rebound_mode),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
    except (BenchmarkError, ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
