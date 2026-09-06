#!/usr/bin/env python3
"""Opt-in Newtonian equal-mass binary comparison with REBOUND 5.1.1.

Both numerical lanes are measured independently against the closed-form orbit.
The report is measurement-only: timings are raw and no comparative or
qualification claim is authorized.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import stat
import sys
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any

import numpy as np

from jxplanetx import __version__ as JX_VERSION
import jxplanetx.engine as jx_engine
from jxplanetx.engine import (
    AdaptiveRKF78Spec,
    BackendSpec,
    ForcePlan,
    NewtonianPointMass,
    ParameterMetadata,
    Provenance,
    StateSnapshot,
    integrate_trajectory,
)


SCHEMA = "jx.rebound_ias15.equal_mass_binary.v2"
REQUIRED_REBOUND_VERSION = "5.1.1"
JX_ADAPTIVE_BASELINE_MODE = "adaptive_baseline"
JX_MATCHED_ACCURACY_MODE = "matched_accuracy_compensated_fixed_cap"
JX_CONFIGURATION_MODES = (
    JX_ADAPTIVE_BASELINE_MODE,
    JX_MATCHED_ACCURACY_MODE,
)
DEFAULT_JX_CONFIGURATION_MODE = JX_MATCHED_ACCURACY_MODE
ENGINE_SOURCE_MANIFEST_SCHEMA = "jxplanetx.engine-source-manifest.v1"
ENGINE_SOURCE_TREE_HASH_DOMAIN = "jxplanetx.engine-source-tree.v1"
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
BODY_IDS = ("BODY_A", "BODY_B")
MASSES = np.array((0.5, 0.5), dtype=np.float64)
ENERGY_0 = -0.125
ANGULAR_MOMENTUM_0 = np.array((0.0, 0.0, 0.25), dtype=np.float64)
FORBIDDEN_RESULT_KEYS = frozenset(("winner", "defeated", "passed", "speedup"))
CLAIM_CONTROLS = {
    "timing_comparable": False,
    "superiority_claimed": False,
    "qualification_claimed": False,
    "reference_truth_claimed": False,
    "production_gate_authorized": False,
}
COMMON_ACCURACY_GATE = {
    "position_vector_l2_max": 2.0e-12,
    "velocity_vector_l2_max": 2.0e-12,
    "relative_total_energy_max": 1.0e-14,
    "relative_angular_momentum_max": 5.0e-15,
    "center_of_mass_position_max": 1.0e-12,
    "total_momentum_max": 1.0e-12,
}


class BenchmarkError(RuntimeError):
    """The comparison contract could not be satisfied exactly."""


@dataclass(frozen=True)
class Profile:
    periods: int = 1
    samples_per_period: int = 16

    def __post_init__(self) -> None:
        if type(self.periods) is not int or self.periods <= 0:
            raise ValueError("periods must be a positive integer")
        if type(self.samples_per_period) is not int or self.samples_per_period <= 0:
            raise ValueError("samples_per_period must be a positive integer")

    @property
    def epochs(self) -> tuple[float, ...]:
        count = self.periods * self.samples_per_period
        return tuple(float(index * PERIOD / self.samples_per_period) for index in range(count + 1))

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
            return "jx.rebound_ias15.equal_mass_binary.full_100_period.v1"
        if self.name == "smoke":
            return "jx.rebound_ias15.equal_mass_binary.smoke.v1"
        return "jx.rebound_ias15.equal_mass_binary.custom.v1"


@dataclass(frozen=True, eq=False)
class Lane:
    engine_id: str
    epochs: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    setup_seconds: float
    integration_seconds: float
    settings: dict[str, Any]
    runtime: dict[str, Any]
    accounting: dict[str, Any]


def _decimal_sin_cos(epoch: float) -> tuple[float, float]:
    """Evaluate sin/cos for a binary64 epoch within the documented safe bound."""

    if not np.isfinite(epoch):
        raise ValueError("epoch must be finite")
    if abs(float(epoch)) > DECIMAL_EPOCH_ABS_MAX:
        raise ValueError(f"abs(epoch) must not exceed {DECIMAL_EPOCH_ABS_MAX:.1e}")
    try:
        with localcontext() as context:
            context.prec = DECIMAL_PRECISION
            x = Decimal.from_float(float(epoch))
            pi = Decimal(DECIMAL_PI_TEXT)
            x %= Decimal(2) * pi
            if x > pi:
                x -= Decimal(2) * pi
            if x < -pi:
                x += Decimal(2) * pi
            cosine_sign = Decimal(1)
            half_pi = pi / Decimal(2)
            if x > half_pi:
                x = pi - x
                cosine_sign = Decimal(-1)
            elif x < -half_pi:
                x = -pi - x
                cosine_sign = Decimal(-1)

            x_squared = x * x
            sine_term = sine = x
            cosine_term = cosine = Decimal(1)
            index = 1
            while True:
                previous_sine, previous_cosine = sine, cosine
                sine_term *= -x_squared / Decimal((2 * index) * (2 * index + 1))
                cosine_term *= -x_squared / Decimal((2 * index - 1) * (2 * index))
                sine += sine_term
                cosine += cosine_term
                if sine == previous_sine and cosine == previous_cosine:
                    break
                index += 1
            return float(sine), float(cosine_sign * cosine)
    except InvalidOperation as exc:
        raise ValueError("Decimal argument reduction failed for epoch") from exc


def analytic_trajectory(epochs: Any) -> tuple[np.ndarray, np.ndarray]:
    checked = np.asarray(epochs, dtype=np.float64)
    if checked.ndim != 1 or checked.size < 1 or not np.all(np.isfinite(checked)):
        raise ValueError("epochs must be a nonempty finite one-dimensional array")
    sine_cosine = tuple(_decimal_sin_cos(float(epoch)) for epoch in checked)
    sine = np.array(tuple(value[0] for value in sine_cosine), dtype=np.float64)[:, None]
    cosine = np.array(tuple(value[1] for value in sine_cosine), dtype=np.float64)[:, None]
    signs = np.array((-1.0, 1.0), dtype=np.float64)[None, :]
    positions = np.zeros((checked.size, 2, 3), dtype=np.float64)
    velocities = np.zeros_like(positions)
    positions[:, :, 0] = 0.5 * signs * cosine
    positions[:, :, 1] = 0.5 * signs * sine
    velocities[:, :, 0] = -0.5 * signs * sine
    velocities[:, :, 1] = 0.5 * signs * cosine
    return positions, velocities


def _trajectory_arrays(positions: Any, velocities: Any) -> tuple[np.ndarray, np.ndarray]:
    checked = (np.asarray(positions, dtype=np.float64), np.asarray(velocities, dtype=np.float64))
    if checked[0].shape != checked[1].shape or checked[0].ndim != 3 or checked[0].shape[1:] != (2, 3):
        raise ValueError("trajectory arrays must share shape (checkpoint_count, 2, 3)")
    if not all(np.all(np.isfinite(value)) for value in checked):
        raise ValueError("trajectory arrays must be finite")
    return checked


def cartesian_metrics(candidate: Any, reference: Any) -> dict[str, float]:
    candidate_array = np.asarray(candidate, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if candidate_array.shape != reference_array.shape or candidate_array.ndim != 3 or candidate_array.shape[1:] != (2, 3):
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
    energy_drift = np.abs((energy - ENERGY_0) / abs(ENERGY_0))
    angular_drift = np.linalg.norm(angular - ANGULAR_MOMENTUM_0, axis=1) / np.linalg.norm(ANGULAR_MOMENTUM_0)
    center_drift = np.linalg.norm(center, axis=1)
    momentum_drift = np.linalg.norm(momentum, axis=1)
    result: dict[str, float] = {}
    for label, values in (("relative_total_energy", energy_drift), ("relative_angular_momentum", angular_drift), ("center_of_mass_position", center_drift), ("total_momentum", momentum_drift)):
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
    return {"path": str(checked), "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def _engine_source_tree_sha256(files: Any) -> str:
    try:
        checked = list(files)
    except TypeError as exc:
        raise BenchmarkError("engine source manifest files must be iterable") from exc
    for entry in checked:
        if type(entry) is not dict or set(entry) != {"path", "size_bytes", "sha256"}:
            raise BenchmarkError("engine source manifest entry schema is invalid")
        path = entry["path"]
        size = entry["size_bytes"]
        digest = entry["sha256"]
        if (
            type(path) is not str
            or not path.startswith("jxplanetx/engine/")
            or "\\" in path
            or type(size) is not int
            or size < 0
            or type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise BenchmarkError("engine source manifest entry value is invalid")
    paths = [entry["path"] for entry in checked]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise BenchmarkError(
            "engine source manifest must have unique normalized paths in sorted order"
        )
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
    package_init = Path(os.path.abspath(os.fspath(jx_engine.__file__)))
    _file_identity(package_init)
    package_directory = package_init.parent
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
    missing = tuple(path for path in REQUIRED_ENGINE_SOURCE_PATHS if path not in roster)
    if missing:
        raise BenchmarkError(f"required engine sources are missing: {missing!r}")
    loaded_paths = set()
    for name, module in tuple(sys.modules.items()):
        if name != "jxplanetx.engine" and not name.startswith("jxplanetx.engine."):
            continue
        module_path = getattr(module, "__file__", None)
        if module_path is None:
            raise BenchmarkError(f"loaded engine module has no source path: {name}")
        path = Path(os.path.abspath(os.fspath(module_path)))
        if path.suffix == ".py" and path.parent == package_directory:
            loaded_paths.add(f"jxplanetx/engine/{path.name}")
    if not loaded_paths.issubset(roster):
        raise BenchmarkError("loaded engine source is absent from the package manifest")
    return {
        "schema": ENGINE_SOURCE_MANIFEST_SCHEMA,
        "package": "jxplanetx.engine",
        "package_directory": str(package_directory),
        "selection": "ALL_TOP_LEVEL_REGULAR_NONSYMLINK_PYTHON_SOURCES",
        "classification": "PROVENANCE_DIAGNOSTIC",
        "authority_authorized": False,
        "files": files,
        "file_count": len(files),
        "loaded_source_paths": sorted(loaded_paths),
        "tree_hash_algorithm": "SHA256_DOMAIN_SEPARATED_SORTED_JSON_V1",
        "tree_hash_domain": ENGINE_SOURCE_TREE_HASH_DOMAIN,
        "tree_sha256": _engine_source_tree_sha256(files),
    }


def _validate_engine_sources_manifest(manifest: Any) -> None:
    if type(manifest) is not dict or set(manifest) != {
        "schema",
        "package",
        "package_directory",
        "selection",
        "classification",
        "authority_authorized",
        "files",
        "file_count",
        "loaded_source_paths",
        "tree_hash_algorithm",
        "tree_hash_domain",
        "tree_sha256",
    }:
        raise BenchmarkError("engine source manifest schema is invalid")
    if (
        manifest["schema"] != ENGINE_SOURCE_MANIFEST_SCHEMA
        or manifest["package"] != "jxplanetx.engine"
        or manifest["selection"]
        != "ALL_TOP_LEVEL_REGULAR_NONSYMLINK_PYTHON_SOURCES"
        or manifest["classification"] != "PROVENANCE_DIAGNOSTIC"
        or manifest["authority_authorized"] is not False
        or manifest["tree_hash_algorithm"]
        != "SHA256_DOMAIN_SEPARATED_SORTED_JSON_V1"
        or manifest["tree_hash_domain"] != ENGINE_SOURCE_TREE_HASH_DOMAIN
        or type(manifest["files"]) is not list
        or type(manifest["loaded_source_paths"]) is not list
    ):
        raise BenchmarkError("engine source manifest fixed provenance is invalid")
    package_init = Path(os.path.abspath(os.fspath(jx_engine.__file__)))
    package_directory = package_init.parent
    if manifest["package_directory"] != str(package_directory):
        raise BenchmarkError("engine source manifest package path is inconsistent")
    files = manifest["files"]
    if type(manifest["file_count"]) is not int or manifest["file_count"] != len(files):
        raise BenchmarkError("engine source manifest count is inconsistent")
    roster = tuple(entry["path"] for entry in files)
    expected_roster = tuple(
        f"jxplanetx/engine/{source.name}"
        for source in sorted(package_directory.glob("*.py"), key=lambda path: path.name)
    )
    if roster != expected_roster or not set(REQUIRED_ENGINE_SOURCE_PATHS).issubset(roster):
        raise BenchmarkError("engine source manifest roster is inconsistent")
    for entry in files:
        source = package_directory / Path(entry["path"]).name
        identity = _file_identity(source)
        if entry["size_bytes"] != identity["size_bytes"] or entry["sha256"] != identity["sha256"]:
            raise BenchmarkError("engine source manifest content identity is inconsistent")
    loaded = manifest["loaded_source_paths"]
    if loaded != sorted(set(loaded)) or not set(loaded).issubset(roster):
        raise BenchmarkError("engine source manifest loaded-source roster is invalid")
    expected_tree = _engine_source_tree_sha256(files)
    if manifest["tree_sha256"] != expected_tree:
        raise BenchmarkError("engine source tree content hash is inconsistent")


def _numpy_runtime_identity() -> dict[str, Any]:
    native = importlib.import_module("numpy._core._multiarray_umath")
    return {
        "classification": "PROVENANCE_DIAGNOSTIC",
        "authority_authorized": False,
        "identity_scope": "PACKAGE_MODULE_AND_MULTIARRAY_UMATH_NATIVE_BINARY",
        "version": np.__version__,
        "package_module": _file_identity(np.__file__),
        "multiarray_umath_native_binary": _file_identity(native.__file__),
    }


def _jx_runtime_provenance() -> dict[str, Any]:
    return {
        "jxplanetx_version": JX_VERSION,
        "engine_sources": _engine_sources_manifest(),
        "benchmark_script": _file_identity(__file__),
        "numpy": _numpy_runtime_identity(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def _validate_jx_runtime_provenance(runtime: Any) -> None:
    if type(runtime) is not dict or set(runtime) != {
        "jxplanetx_version",
        "engine_sources",
        "benchmark_script",
        "numpy",
        "python_executable",
        "python_version",
        "platform",
    }:
        raise BenchmarkError("JX runtime provenance schema is invalid")
    _validate_engine_sources_manifest(runtime["engine_sources"])
    expected = _jx_runtime_provenance()
    if runtime != expected:
        raise BenchmarkError("JX runtime provenance does not match loaded bytes")


def _jx_spec(profile: Profile, mode: str) -> tuple[AdaptiveRKF78Spec, dict[str, Any]]:
    if mode not in JX_CONFIGURATION_MODES:
        raise ValueError(
            f"jx configuration mode must be one of {JX_CONFIGURATION_MODES!r}"
        )
    if mode == JX_MATCHED_ACCURACY_MODE:
        step = PERIOD / 256.0
        position_atol = np.full((2, 3), 1.0e6, dtype=np.float64)
        velocity_atol = position_atol.copy()
        values = {
            "initial_step": step,
            "minimum_step": 1.0e-15,
            "maximum_step": step,
            "position_atol": position_atol,
            "position_rtol": 1.0e-6,
            "velocity_atol": velocity_atol,
            "velocity_rtol": 1.0e-6,
            "maximum_steps": 30_000,
            "maximum_rejections": 0,
            "safety_factor": 0.9,
            "minimum_scale_factor": 0.2,
            "maximum_scale_factor": 5.0,
        }
        description = (
            "compensated capped-adaptive RKF78 configuration matched to the "
            "common workload accuracy gate"
        )
    else:
        position_atol = np.full((2, 3), 1.0e-13, dtype=np.float64)
        velocity_atol = position_atol.copy()
        values = {
            "initial_step": PERIOD / 128.0,
            "minimum_step": 1.0e-15,
            "maximum_step": PERIOD / 16.0,
            "position_atol": position_atol,
            "position_rtol": 1.0e-12,
            "velocity_atol": velocity_atol,
            "velocity_rtol": 1.0e-12,
            "maximum_steps": 2_000_000,
            "maximum_rejections": 200_000,
            "safety_factor": 0.9,
            "minimum_scale_factor": 0.2,
            "maximum_scale_factor": 5.0,
        }
        description = "prior adaptive RKF78 baseline retained for continuity"
    spec = AdaptiveRKF78Spec(profile.epochs, **values)
    settings = {
        "configuration_mode": mode,
        "configuration_description": description,
        "method_classification": "capped_adaptive_rkf78",
        "backend": "numpy",
        "device": "cpu",
        "dtype": "float64",
        "force_model": "force.newtonian.point_mass",
        "initial_step": spec.initial_step,
        "minimum_step": spec.minimum_step,
        "maximum_step": spec.maximum_step,
        "position_atol": float(position_atol[0, 0]),
        "position_rtol": spec.position_rtol,
        "velocity_atol": float(velocity_atol[0, 0]),
        "velocity_rtol": spec.velocity_rtol,
        "maximum_steps": spec.maximum_steps,
        "maximum_rejections": spec.maximum_rejections,
        "safety_factor": spec.safety_factor,
        "minimum_scale_factor": spec.minimum_scale_factor,
        "maximum_scale_factor": spec.maximum_scale_factor,
        "checkpoint_policy": spec.checkpoint_policy,
    }
    return spec, settings


def _jx_lane(profile: Profile, mode: str = DEFAULT_JX_CONFIGURATION_MODE) -> Lane:
    started = time.perf_counter()
    script = Path(__file__).resolve()
    provenance = Provenance("jx.benchmark.equal_mass_binary.v1", "docs/JX_PUBLIC_RESEARCH_STUDY.md", "1", hashlib.sha256(script.read_bytes()).hexdigest())
    backend = BackendSpec("numpy", "cpu", 2)
    positions, velocities = analytic_trajectory((0.0,))
    snapshot = StateSnapshot(
        "jx.benchmark.equal_mass_binary.snapshot.v1", 0.0, "SYNTHETIC", "BARYCENTRIC_INERTIAL", "BARYCENTER", "CARTESIAN_RIGHT_HANDED", "L", "T", "M", "jx.benchmark.synthetic.v1", BODY_IDS,
        positions[0].copy(), velocities[0].copy(), MASSES.copy(), MASSES.copy(), np.zeros(2), np.ones(2, dtype=np.bool_), provenance,
    )
    metadata = ParameterMetadata("state.gravitational_parameters", "L^3/T^2", provenance, None, None, 0.0, profile.epochs[-1])
    plan = ForcePlan("jx.benchmark.equal_mass_binary.plan.v1", backend, (NewtonianPointMass(BODY_IDS, BODY_IDS, "jx.benchmark.synthetic.v1", (metadata,)),))
    spec, settings = _jx_spec(profile, mode)
    setup_seconds = time.perf_counter() - started
    integrated = time.perf_counter()
    result = integrate_trajectory(snapshot, plan, spec)
    integration_seconds = time.perf_counter() - integrated
    return Lane(
        "jx_rkf78", np.asarray(result.checkpoint_epochs), np.stack(result.positions), np.stack(result.velocities), setup_seconds, integration_seconds,
        settings,
        _jx_runtime_provenance(),
        {
            "custody_source": "TrajectoryResult",
            "attempted_steps": result.attempted_steps,
            "accepted_steps": result.accepted_steps,
            "rejected_steps": result.rejected_steps,
            "force_evaluations": result.force_evaluations,
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
                "epoch_span": abs(
                    result.checkpoint_epochs[-1] - result.checkpoint_epochs[0]
                ),
                "checksum_algorithm": (
                    result.accepted_step_ledger_checksum_algorithm
                ),
                "checksum_domain": result.accepted_step_ledger_checksum_domain,
                "content_integrity_sha256": (
                    result.accepted_step_ledger_content_sha256
                ),
            },
            "evidence_class": result.evidence_class,
            "registry_authorized": result.registry_authorized,
            "qualification_authorized": result.qualification_authorized,
        },
    )


def _validate_rebound_module(module: Any) -> None:
    version = getattr(module, "__version__", None)
    if version != REQUIRED_REBOUND_VERSION:
        raise BenchmarkError(f"REBOUND {REQUIRED_REBOUND_VERSION} is required exactly; found {version!r}")
    if not callable(getattr(module, "Simulation", None)):
        raise BenchmarkError("REBOUND 5.1.1 does not expose the required Simulation API")


def _load_rebound() -> Any:
    if os.environ.get("OMP_NUM_THREADS") not in (None, "1"):
        raise BenchmarkError("OMP_NUM_THREADS must be unset or exactly '1'")
    if "rebound" in sys.modules and os.environ.get("OMP_NUM_THREADS") != "1":
        raise BenchmarkError("REBOUND was imported before single-thread mode was established")
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        module = importlib.import_module("rebound")
    except ModuleNotFoundError as exc:
        raise BenchmarkError("install REBOUND 5.1.1 in an isolated benchmark environment") from exc
    _validate_rebound_module(module)
    return module


def _rebound_lane(profile: Profile) -> Lane:
    started = time.perf_counter()
    rebound = _load_rebound()
    simulation = rebound.Simulation()
    simulation.G, simulation.gravity, simulation.collision, simulation.softening = 1.0, "basic", "none", 0.0
    simulation.integrator = "ias15"
    simulation.integrator.epsilon = 1.0e-12
    simulation.integrator.min_dt = 0.0
    simulation.integrator.adaptive_mode = "PRS23"
    simulation.dt = PERIOD / 128.0
    simulation.add(m=0.5, x=-0.5, y=0.0, z=0.0, vx=0.0, vy=-0.5, vz=0.0)
    simulation.add(m=0.5, x=0.5, y=0.0, z=0.0, vx=0.0, vy=0.5, vz=0.0)
    simulation.N_active, simulation.testparticle_type = 2, 0
    effective_settings = {
        "G": float(simulation.G),
        "gravity": str(simulation.gravity),
        "collision": str(simulation.collision),
        "softening": float(simulation.softening),
        "integrator": str(simulation.integrator),
        "epsilon": float(simulation.integrator.epsilon),
        "min_dt": float(simulation.integrator.min_dt),
        "adaptive_mode": str(simulation.integrator.adaptive_mode),
        "initial_dt": float(simulation.dt),
        "N_active": int(simulation.N_active),
        "testparticle_type": int(simulation.testparticle_type),
        "exact_finish_time": 1,
        "thread_count": int(os.environ["OMP_NUM_THREADS"]),
        "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
        "provenance": {
            "G": "effective_readback",
            "gravity": "effective_readback",
            "collision": "effective_readback",
            "softening": "effective_readback",
            "integrator": "effective_readback",
            "epsilon": "effective_readback",
            "min_dt": "effective_readback",
            "adaptive_mode": "effective_readback",
            "initial_dt": "effective_readback",
            "N_active": "effective_readback",
            "testparticle_type": "effective_readback",
            "exact_finish_time": "call_argument",
            "thread_count": "environment_observation",
            "OMP_NUM_THREADS": "environment_observation",
        },
    }
    setup_seconds = time.perf_counter() - started
    positions = np.empty((len(profile.epochs), 2, 3), dtype=np.float64)
    velocities = np.empty_like(positions)

    def capture(index: int) -> None:
        for body_index, particle in enumerate(simulation.particles[:2]):
            positions[index, body_index] = (particle.x, particle.y, particle.z)
            velocities[index, body_index] = (particle.vx, particle.vy, particle.vz)

    capture(0)
    integrated = time.perf_counter()
    for index, epoch in enumerate(profile.epochs[1:], 1):
        simulation.integrate(epoch, exact_finish_time=1)
        if float(simulation.t) != epoch:
            raise BenchmarkError("REBOUND did not stop at an exact requested checkpoint")
        capture(index)
    integration_seconds = time.perf_counter() - integrated
    module_identity = _file_identity(rebound.__file__)
    library_name = getattr(getattr(rebound, "clibrebound", None), "_name", None)
    runtime: dict[str, Any] = {"rebound_version": rebound.__version__, "module_source": module_identity, "omp_num_threads": os.environ["OMP_NUM_THREADS"]}
    if library_name and Path(library_name).is_file():
        runtime["c_library"] = _file_identity(library_name)
    return Lane(
        "rebound_ias15", np.asarray(profile.epochs), positions, velocities, setup_seconds, integration_seconds,
        effective_settings,
        runtime, {},
    )


def _within_common_accuracy_gate(
    position_metrics: dict[str, float],
    velocity_metrics: dict[str, float],
    drifts: dict[str, float],
) -> bool:
    observed = {
        "position_vector_l2_max": position_metrics["vector_l2_max"],
        "velocity_vector_l2_max": velocity_metrics["vector_l2_max"],
        "relative_total_energy_max": drifts["relative_total_energy_max"],
        "relative_angular_momentum_max": drifts[
            "relative_angular_momentum_max"
        ],
        "center_of_mass_position_max": drifts["center_of_mass_position_max"],
        "total_momentum_max": drifts["total_momentum_max"],
    }
    return all(observed[key] <= limit for key, limit in COMMON_ACCURACY_GATE.items())


def _lane_report(lane: Lane, oracle_positions: np.ndarray, oracle_velocities: np.ndarray) -> dict[str, Any]:
    if not np.array_equal(lane.epochs, np.asarray(lane.epochs, dtype=np.float64)):
        raise BenchmarkError("lane epochs are invalid")
    position_metrics = cartesian_metrics(lane.positions, oracle_positions)
    velocity_metrics = cartesian_metrics(lane.velocities, oracle_velocities)
    drifts = invariant_metrics(lane.positions, lane.velocities)
    return {
        "engine_id": lane.engine_id,
        "settings": lane.settings,
        "runtime": lane.runtime,
        "accounting": lane.accounting,
        "timing_seconds": {
            "measurement_count": 1,
            "setup": lane.setup_seconds,
            "integration": lane.integration_seconds,
            "total": lane.setup_seconds + lane.integration_seconds,
        },
        "trajectory_sha256": _digest(lane.epochs, lane.positions, lane.velocities),
        "analytic_oracle_errors": {
            "position": position_metrics,
            "velocity": velocity_metrics,
        },
        "invariant_drifts": drifts,
        "within_common_accuracy_gate": _within_common_accuracy_gate(
            position_metrics, velocity_metrics, drifts
        ),
    }


def build_report(profile: Profile, jx_lane: Lane, rebound_lane: Lane, source_details: dict[str, Any]) -> dict[str, Any]:
    epochs = np.asarray(profile.epochs, dtype=np.float64)
    if not np.array_equal(jx_lane.epochs, epochs) or not np.array_equal(rebound_lane.epochs, epochs):
        raise BenchmarkError("both lanes must use the exact declared checkpoint array")
    jx_mode = jx_lane.settings.get("configuration_mode")
    if jx_mode not in JX_CONFIGURATION_MODES:
        raise BenchmarkError("JX lane does not declare a supported configuration mode")
    _validate_jx_runtime_provenance(jx_lane.runtime)
    oracle_positions, oracle_velocities = analytic_trajectory(epochs)
    report = {
        "schema": SCHEMA,
        "benchmark_id": "jx.rebound_ias15.newtonian_equal_mass_binary.v2",
        "profile": {"name": profile.name, "profile_id": profile.profile_id, "periods": profile.periods, "samples_per_period": profile.samples_per_period, "named_full_profile": profile.named_full_profile, "jx_configuration_mode": jx_mode},
        "problem": {"model": "pure_newtonian_equal_mass_barycentric_binary", "G": 1.0, "body_ids": list(BODY_IDS), "masses_and_gravitational_parameters": MASSES.tolist(), "initial_positions": oracle_positions[0].tolist(), "initial_velocities": oracle_velocities[0].tolist(), "period": PERIOD, "analytic_total_energy": ENERGY_0, "analytic_total_angular_momentum": ANGULAR_MOMENTUM_0.tolist(), "analytic_oracle": {"implementation": "stdlib_decimal_argument_reduced_taylor", "decimal_precision": DECIMAL_PRECISION, "maximum_absolute_epoch": DECIMAL_EPOCH_ABS_MAX, "epoch_input": "exact_binary64_via_Decimal.from_float", "pi": DECIMAL_PI_TEXT}},
        "checkpoints": {"count": len(epochs), "epochs": epochs.tolist(), "sha256": _digest(epochs), "identical_for_both_lanes": True},
        "runtime": {"python_executable": sys.executable, "python_version": platform.python_version(), "python_implementation": platform.python_implementation(), "numpy_version": np.__version__, "platform": platform.platform(), "sources": source_details},
        "lanes": {"jx_rkf78": _lane_report(jx_lane, oracle_positions, oracle_velocities), "rebound_ias15": _lane_report(rebound_lane, oracle_positions, oracle_velocities)},
        "cross_engine_disagreement": {"position": cartesian_metrics(jx_lane.positions, rebound_lane.positions), "velocity": cartesian_metrics(jx_lane.velocities, rebound_lane.velocities)},
        "common_accuracy_gate": {
            "scope": "THIS_ANALYTIC_EQUAL_MASS_BINARY_WORKLOAD_ONLY",
            "thresholds": dict(COMMON_ACCURACY_GATE),
            "claim_authorized": False,
        },
        "claim_controls": dict(CLAIM_CONTROLS),
        "serialization": {"format": "JSON", "object_keys_sorted": True, "nonfinite_numbers_allowed": False, "includes_timing_and_path_diagnostics": True},
    }
    keys: set[str] = set()
    def collect(value: Any) -> None:
        if isinstance(value, dict):
            keys.update(value)
            for nested in value.values(): collect(nested)
        elif isinstance(value, list):
            for nested in value: collect(nested)
    collect(report)
    if keys & FORBIDDEN_RESULT_KEYS:
        raise BenchmarkError("report contains forbidden comparative vocabulary")
    json.dumps(report, allow_nan=False, sort_keys=True)
    return report


def run(
    profile: Profile,
    jx_configuration_mode: str = DEFAULT_JX_CONFIGURATION_MODE,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    details = {"benchmark_script": _file_identity(__file__), "public_research_study": _file_identity(root / "docs" / "JX_PUBLIC_RESEARCH_STUDY.md")}
    return build_report(
        profile,
        _jx_lane(profile, jx_configuration_mode),
        _rebound_lane(profile),
        details,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare JX RKF78 and exact REBOUND 5.1.1 IAS15 against an analytic Newtonian equal-mass binary.")
    parser.add_argument("--periods", type=int, default=1, help="Positive orbit count; use 100 for the named full profile.")
    parser.add_argument("--samples-per-period", type=int, default=16, help="Positive checkpoint intervals per orbit; use 4 with 100 periods for the full profile.")
    parser.add_argument(
        "--jx-mode",
        choices=JX_CONFIGURATION_MODES,
        default=DEFAULT_JX_CONFIGURATION_MODE,
        help=(
            "JX capped-adaptive RKF78 configuration; the matched-accuracy "
            "compensated fixed-cap lane is the default."
        ),
    )
    parser.add_argument("--output", type=Path, help="Write sorted finite JSON here instead of stdout.")
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        profile = Profile(args.periods, args.samples_per_period)
        rendered = json.dumps(
            run(profile, args.jx_mode),
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
