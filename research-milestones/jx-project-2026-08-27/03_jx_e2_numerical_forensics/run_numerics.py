#!/usr/bin/env python3
"""Run the locked JX-E2 active-body numerical-method forensic matrix."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import resource
import secrets
import shutil
import struct
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any


EXPECTED_CONTRACT_SHA256 = "d4edc6e17df40c3eeb6a72c7c55ad3bb530e6c79a40017f6dffe9ec553bc3d8f"
RESULT_SCHEMA = "jx-e2-numerics-result/v1"
BUNDLE_SCHEMA = "jx-e2-numerics-bundle/v1"
ARM_SCHEMA = "jx-e2-numerics-arm/v1"
REBOUND_PYTHON_SOURCE_SHA256 = (
    "2c40b16571d57049cbf4bb8329a0c58342f3dc0f0cf49d860ca77fda5a73ae3a"
)
REBOUND_BINARY_SHA256 = (
    "fe7a23bcece1c3f1f869089e9e8d806bedb4727d893d2e551339adbb6665c28a"
)
_SOURCE_CACHE_HOLDER: tempfile.TemporaryDirectory[str] | None = None


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant: {token}")


def _finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"non-finite JSON number: {token}")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
        parse_float=_finite_float,
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def load_source_module(
    name: str, path: Path, expected_sha256: str
) -> types.ModuleType:
    """Compile the exact hash-bound source bytes without consulting __pycache__."""
    source = path.read_bytes()
    if sha256_bytes(source) != expected_sha256:
        raise RuntimeError(f"source-module hash mismatch: {path}")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    code = compile(source, str(path), "exec", dont_inherit=True)
    exec(code, module.__dict__)
    return module


def rebound_python_tree_sha256(root: Path) -> str:
    files = sorted(root.rglob("*.py"), key=lambda path: path.relative_to(root).as_posix())
    digest = hashlib.sha256()
    digest.update(b"jx-e2-rebound-python-sources/v1\0")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def rebound_python_source_sha256(module: Any) -> str:
    return rebound_python_tree_sha256(Path(module.__file__).resolve().parent)


def get_rebound() -> Any:
    global _SOURCE_CACHE_HOLDER
    module = sys.modules.get("rebound")
    if module is None:
        specification = importlib.util.find_spec("rebound")
        if specification is None or not specification.submodule_search_locations:
            raise RuntimeError("cannot resolve REBOUND package before import")
        source_root = Path(next(iter(specification.submodule_search_locations))).resolve()
        if rebound_python_tree_sha256(source_root) != REBOUND_PYTHON_SOURCE_SHA256:
            raise RuntimeError("REBOUND Python source-tree hash mismatch before import")
        binary_candidates = sorted(source_root.parent.glob("librebound*.so"))
        if (
            len(binary_candidates) != 1
            or sha256_file(binary_candidates[0]) != REBOUND_BINARY_SHA256
        ):
            raise RuntimeError("REBOUND native binary hash mismatch before import")
        _SOURCE_CACHE_HOLDER = tempfile.TemporaryDirectory(
            prefix="jx-e2-source-import-"
        )
        sys.pycache_prefix = _SOURCE_CACHE_HOLDER.name
        sys.dont_write_bytecode = True
        importlib.invalidate_caches()
        module = importlib.import_module("rebound")
        setattr(module, "_jx_e2_source_only_import", True)
        setattr(module, "_jx_e2_source_cache_holder", _SOURCE_CACHE_HOLDER)
    elif getattr(module, "_jx_e2_source_only_import", False) is not True:
        raise RuntimeError("REBOUND was imported before the source-only import guard")
    if rebound_python_source_sha256(module) != REBOUND_PYTHON_SOURCE_SHA256:
        raise RuntimeError("REBOUND Python source-tree hash mismatch")
    return module


def serialized_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def atomic_json(path: Path, value: Any) -> None:
    payload = serialized_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def vector_norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(math.fsum(component * component for component in vector))


def vector_subtract(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return tuple(left[index] - right[index] for index in range(3))  # type: ignore[return-value]


def cross(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def physical_state_digest(simulation: Any) -> str:
    digest = hashlib.sha256()
    digest.update(b"jx-e2-physical-active-state/v1\0")
    digest.update(
        struct.pack(
            "!2d3I",
            float(simulation.t),
            float(simulation.G),
            int(simulation.N),
            int(simulation.N_active),
            int(simulation.testparticle_type),
        )
    )
    for index, particle in enumerate(simulation.particles):
        digest.update(
            struct.pack(
                "!II8d",
                index,
                int(particle.hash.value),
                float(particle.m),
                float(particle.r),
                float(particle.x),
                float(particle.y),
                float(particle.z),
                float(particle.vx),
                float(particle.vy),
                float(particle.vz),
            )
        )
    return digest.hexdigest()


def particle_identity(simulation: Any) -> tuple[tuple[int, float, float], ...]:
    return tuple(
        (
            int(simulation.particles[index].hash.value),
            float(simulation.particles[index].m),
            float(simulation.particles[index].r),
        )
        for index in range(simulation.N)
    )


def decoded_state_digest(simulation: Any) -> str:
    digest = hashlib.sha256()
    digest.update(b"jx-e2-decoded-active-state/v1\0")
    configuration: dict[str, Any] = {
        "physical_state_sha256": physical_state_digest(simulation),
        "integrator": str(simulation.integrator),
        "gravity": str(simulation.gravity),
        "collision": str(simulation.collision),
        "boundary": str(simulation.boundary),
        "dt_hex": float(simulation.dt).hex(),
    }
    if str(simulation.integrator) == "mercurius":
        configuration["mercurius"] = {
            "r_crit_hill_hex": float(simulation.ri_mercurius.r_crit_hill).hex(),
            "safe_mode": int(simulation.ri_mercurius.safe_mode),
            "is_synchronized": int(simulation.ri_mercurius.is_synchronized),
            "recalculate_coordinates_this_timestep": int(
                simulation.ri_mercurius.recalculate_coordinates_this_timestep
            ),
        }
    elif str(simulation.integrator) == "ias15":
        configuration["ias15"] = {
            "epsilon_hex": float(simulation.ri_ias15.epsilon).hex(),
            "min_dt_hex": float(simulation.ri_ias15.min_dt).hex(),
            "adaptive_mode": str(simulation.ri_ias15.adaptive_mode),
            "dt_last_done_hex": float(simulation.dt_last_done).hex(),
        }
    else:
        raise ValueError("unexpected integrator")
    digest.update(canonical_bytes(configuration))
    return digest.hexdigest()


def active_projection_digest(simulation: Any) -> str:
    digest = hashlib.sha256()
    digest.update(b"jx-e2-e1-active-projection/v1\0")
    digest.update(struct.pack("!2dI", float(simulation.t), float(simulation.G), int(simulation.N_active)))
    for index in range(simulation.N_active):
        particle = simulation.particles[index]
        digest.update(
            struct.pack(
                "!II8d",
                index,
                int(particle.hash.value),
                float(particle.m),
                float(particle.r),
                float(particle.x),
                float(particle.y),
                float(particle.z),
                float(particle.vx),
                float(particle.vy),
                float(particle.vz),
            )
        )
    return digest.hexdigest()


def load_e1(contract: dict[str, Any], contract_path: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    boundary = contract["e1_immutable_boundary"]
    base = contract_path.parent
    paths = {
        "contract": (base / boundary["contract_path"]).resolve(),
        "runner": (base / boundary["runner_path"]).resolve(),
        "verifier": (base / boundary["verifier_path"]).resolve(),
        "result": (base / boundary["result_path"]).resolve(),
        "post_failure_audit": (base / boundary["post_failure_audit_path"]).resolve(),
        "audit_script": (base / boundary["audit_script_path"]).resolve(),
    }
    expected_hashes = {
        "contract": boundary["contract_sha256"],
        "runner": boundary["runner_sha256"],
        "verifier": boundary["verifier_sha256"],
        "result": boundary["result_sha256"],
        "post_failure_audit": boundary["post_failure_audit_sha256"],
        "audit_script": boundary["audit_script_sha256"],
    }
    for key, path in paths.items():
        if not path.is_file() or sha256_file(path) != expected_hashes[key]:
            raise RuntimeError(f"immutable E1 binding mismatch: {key}")
    e1_contract = strict_json(paths["contract"])
    e1_result = strict_json(paths["result"])
    e1_audit = strict_json(paths["post_failure_audit"])
    if e1_result.get("verdict") != boundary["required_e1_verdict"]:
        raise RuntimeError("E1 invalid verdict changed")
    if e1_result.get("semantic_sha256") != boundary["result_semantic_sha256"]:
        raise RuntimeError("E1 semantic hash changed")
    if sha256_bytes(canonical_bytes(e1_result["semantic"])) != boundary["result_semantic_sha256"]:
        raise RuntimeError("E1 embedded semantic hash does not recompute")
    if e1_audit.get("audit_state") != boundary["required_e1_audit_state"]:
        raise RuntimeError("E1 post-failure audit state changed")
    if e1_audit.get("thresholds_changed") is not False:
        raise RuntimeError("E1 thresholds were changed")
    if e1_audit.get("execution_b_authorized_or_started") is not False:
        raise RuntimeError("E1 B is no longer forbidden/unstarted")
    if e1_audit.get("new_dynamics_executed") is not False:
        raise RuntimeError("E1 post-failure receipt scope changed")
    if e1_audit.get("integrity") != {
        "all_duplicate_tracer_blocks_have_identical_checkpointed_active_states": True,
        "audit_arm_records_verified": 16,
        "stored_checkpoints_verified": 160,
    }:
        raise RuntimeError("E1 post-failure integrity receipt changed")
    e1_root = paths["contract"].parent
    if (e1_root / "long_b").exists():
        raise RuntimeError("canonical JX-E1 long_b artifact now exists")
    for manifest_path in e1_root.rglob("run_manifest.json"):
        manifest = strict_json(manifest_path)
        if (
            manifest.get("experiment_id") == e1_contract.get("experiment_id")
            and manifest.get("execution_label") == "B"
        ):
            raise RuntimeError("a JX-E1 execution-B manifest now exists")

    module = load_source_module(
        "jx_e1_locked_runner", paths["runner"], expected_hashes["runner"]
    )
    return module, e1_contract, e1_result


def model_and_angle(
    e1_contract: dict[str, Any], configuration: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if configuration["model_id"] is None:
        return None, None
    models = {item["id"]: item for item in e1_contract["model_grid"]}
    angles = {item["id"]: item for item in e1_contract["angle_grid"]}
    return models[configuration["model_id"]], angles[configuration["angle_id"]]


def apply_frame(simulation: Any, frame_id: str) -> None:
    if frame_id == "F0_E1_UNSHIFTED":
        return
    if frame_id != "FCM_ACTIVE_BARYCENTRIC":
        raise ValueError("unknown frame")
    particles = [simulation.particles[index] for index in range(simulation.N_active)]
    total_mass = math.fsum(particle.m for particle in particles)
    r_com = tuple(
        math.fsum(particle.m * getattr(particle, field) for particle in particles) / total_mass
        for field in ("x", "y", "z")
    )
    v_com = tuple(
        math.fsum(particle.m * getattr(particle, field) for particle in particles) / total_mass
        for field in ("vx", "vy", "vz")
    )
    for particle in particles:
        particle.x -= r_com[0]
        particle.y -= r_com[1]
        particle.z -= r_com[2]
        particle.vx -= v_com[0]
        particle.vy -= v_com[1]
        particle.vz -= v_com[2]


def configure_regime(simulation: Any, regime: dict[str, Any]) -> None:
    simulation.collision = "none"
    simulation.boundary = "none"
    if regime["integrator"] == "mercurius":
        simulation.integrator = "mercurius"
        simulation.dt = float(regime["dt_years"])
        simulation.ri_mercurius.r_crit_hill = float(regime["r_crit_hill"])
        simulation.ri_mercurius.safe_mode = int(regime["safe_mode"])
    elif regime["integrator"] == "ias15":
        simulation.integrator = "ias15"
        simulation.reset_integrator()
        simulation.dt = float(regime["initial_dt_years"])
        simulation.ri_ias15.epsilon = float(regime["epsilon"])
        simulation.ri_ias15.min_dt = float(regime["min_dt_years"])
        simulation.ri_ias15.adaptive_mode = regime["adaptive_mode"]
    else:
        raise ValueError("unknown integrator regime")
    verify_regime(simulation, regime)


def verify_regime(simulation: Any, regime: dict[str, Any]) -> None:
    if str(simulation.integrator) != regime["integrator"]:
        raise RuntimeError("integrator readback mismatch")
    if str(simulation.collision) != "none" or str(simulation.boundary) != "none":
        raise RuntimeError("collision/boundary readback mismatch")
    if regime["integrator"] == "mercurius":
        if (
            float(simulation.dt) != float(regime["dt_years"])
            or
            float(simulation.ri_mercurius.r_crit_hill) != float(regime["r_crit_hill"])
            or int(simulation.ri_mercurius.safe_mode) != int(regime["safe_mode"])
        ):
            raise RuntimeError("MERCURIUS setting readback mismatch")
    else:
        if (
            float(simulation.ri_ias15.epsilon) != float(regime["epsilon"])
            or float(simulation.ri_ias15.min_dt) != float(regime["min_dt_years"])
            or str(simulation.ri_ias15.adaptive_mode) != regime["adaptive_mode"]
        ):
            raise RuntimeError("IAS15 setting readback mismatch")


def build_active_simulation(
    e1_module: Any,
    e1_contract: dict[str, Any],
    configuration: dict[str, Any],
    frame_id: str,
    regime: dict[str, Any],
) -> tuple[Any, list[str], dict[str, float]]:
    get_rebound()
    model, angle = model_and_angle(e1_contract, configuration)
    simulation, tracer_start, common_names = e1_module.build_simulation(
        e1_contract, [], model, angle, 0.125
    )
    if tracer_start != simulation.N or simulation.N != simulation.N_active:
        raise RuntimeError("active-only builder layout mismatch")
    active_names = list(common_names)
    scales = {
        item["name"]: float(item["a_AU"])
        for item in e1_contract["analytic_benchmark"]["giants"]
    }
    if model is not None:
        active_names.append(f"P9_{model['id']}_{angle['id']}")
        scales[active_names[-1]] = float(model["a_AU"])
    before = physical_state_digest(simulation)
    apply_frame(simulation, frame_id)
    after_frame = physical_state_digest(simulation)
    configure_regime(simulation, regime)
    after_regime = physical_state_digest(simulation)
    if after_frame != after_regime:
        raise RuntimeError("integrator configuration changed particle state")
    if frame_id == "F0_E1_UNSHIFTED" and before != after_frame:
        raise RuntimeError("unshifted frame changed particle state")
    return simulation, active_names, scales


def state_components(simulation: Any, names: list[str]) -> dict[str, tuple[float, ...]]:
    result = {}
    sun = simulation.particles["Sun"]
    for name in names:
        particle = simulation.particles[name]
        result[name] = (
            float(particle.x - sun.x),
            float(particle.y - sun.y),
            float(particle.z - sun.z),
            float(particle.vx - sun.vx),
            float(particle.vy - sun.vy),
            float(particle.vz - sun.vz),
        )
    return result


def state_discrepancy(
    left: Any,
    right: Any,
    names: list[str],
    scales: dict[str, float],
    gravitational_constant: float,
    sun_mass: float,
) -> dict[str, float]:
    left_state = state_components(left, names)
    right_state = state_components(right, names)
    max_position = 0.0
    max_velocity = 0.0
    maximum_dimensionless = 0.0
    for name in names:
        if name == "Sun":
            continue
        first = left_state[name]
        second = right_state[name]
        position = vector_norm(tuple(first[index] - second[index] for index in range(3)))
        velocity = vector_norm(tuple(first[index] - second[index] for index in range(3, 6)))
        a_scale = scales[name]
        v_scale = math.sqrt(gravitational_constant * sun_mass / a_scale)
        max_position = max(max_position, position)
        max_velocity = max(max_velocity, velocity)
        maximum_dimensionless = max(
            maximum_dimensionless, position / a_scale, velocity / v_scale
        )
    return {
        "maximum_dimensionless_state_discrepancy": maximum_dimensionless,
        "maximum_position_separation_AU": max_position,
        "maximum_velocity_separation_AU_per_year": max_velocity,
    }


def invariant_snapshot(simulation: Any) -> dict[str, Any]:
    particles = [simulation.particles[index] for index in range(simulation.N_active)]
    for index, particle in enumerate(particles):
        for field in ("m", "r", "x", "y", "z", "vx", "vy", "vz"):
            if not math.isfinite(float(getattr(particle, field))):
                raise RuntimeError(f"non-finite particle state: {index}/{field}")
    total_mass = math.fsum(float(particle.m) for particle in particles)
    position = [(float(p.x), float(p.y), float(p.z)) for p in particles]
    velocity = [(float(p.vx), float(p.vy), float(p.vz)) for p in particles]
    masses = [float(p.m) for p in particles]
    momentum = tuple(
        math.fsum(masses[index] * velocity[index][component] for index in range(len(particles)))
        for component in range(3)
    )
    r_com = tuple(
        math.fsum(masses[index] * position[index][component] for index in range(len(particles)))
        / total_mass
        for component in range(3)
    )
    v_com = tuple(component / total_mass for component in momentum)
    origin_terms = [cross(position[index], velocity[index]) for index in range(len(particles))]
    origin_angular = tuple(
        math.fsum(masses[index] * origin_terms[index][component] for index in range(len(particles)))
        for component in range(3)
    )
    relative_positions = [vector_subtract(item, r_com) for item in position]
    relative_velocities = [vector_subtract(item, v_com) for item in velocity]
    com_terms = [cross(relative_positions[index], relative_velocities[index]) for index in range(len(particles))]
    com_angular = tuple(
        math.fsum(masses[index] * com_terms[index][component] for index in range(len(particles)))
        for component in range(3)
    )
    kinetic_internal = 0.5 * math.fsum(
        masses[index] * vector_norm(relative_velocities[index]) ** 2
        for index in range(len(particles))
    )
    potential_terms = []
    for left in range(len(particles)):
        for right in range(left + 1, len(particles)):
            separation = vector_norm(vector_subtract(position[left], position[right]))
            potential_terms.append(
                -float(simulation.G) * masses[left] * masses[right] / separation
            )
    potential = math.fsum(potential_terms)
    intrinsic_energy = kinetic_internal + potential
    linear_scale = math.fsum(
        masses[index] * vector_norm(relative_velocities[index])
        for index in range(len(particles))
    )
    angular_scale = math.fsum(
        masses[index] * vector_norm(com_terms[index]) for index in range(len(particles))
    )
    energy_scale = kinetic_internal + abs(potential)
    engine_energy = float(simulation.energy())
    values = (
        total_mass,
        *momentum,
        *r_com,
        *v_com,
        *origin_angular,
        *com_angular,
        intrinsic_energy,
        engine_energy,
        linear_scale,
        angular_scale,
        energy_scale,
    )
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("non-finite invariant")
    if min(linear_scale, angular_scale, energy_scale) <= 0.0:
        raise RuntimeError("non-positive invariant normalization scale")
    decomposition = vector_subtract(
        origin_angular,
        tuple(
            com_angular[index] + cross(r_com, momentum)[index] for index in range(3)
        ),
    )
    return {
        "total_mass": total_mass,
        "momentum": momentum,
        "r_com": r_com,
        "v_com": v_com,
        "origin_angular": origin_angular,
        "com_angular": com_angular,
        "intrinsic_energy": intrinsic_energy,
        "engine_energy": engine_energy,
        "linear_scale": linear_scale,
        "angular_scale": angular_scale,
        "energy_scale": energy_scale,
        "decomposition_residual": vector_norm(decomposition),
    }


def blank_metric(initial: dict[str, Any]) -> dict[str, Any]:
    return {
        "initial": initial,
        "maximum": {
            "relative_engine_energy_drift": 0.0,
            "scale_normalized_intrinsic_energy_residual": 0.0,
            "scale_normalized_linear_momentum_residual": 0.0,
            "scale_normalized_origin_angular_momentum_residual": 0.0,
            "scale_normalized_com_angular_momentum_residual": 0.0,
            "relative_initial_linear_momentum_residual": 0.0,
            "relative_initial_origin_angular_momentum_residual": 0.0,
            "com_ballistic_position_residual_AU": 0.0,
            "angular_decomposition_residual_over_scale": 0.0,
        },
        "endpoint": None,
    }


def update_metric(metric: dict[str, Any], current: dict[str, Any], time_year: float) -> None:
    initial = metric["initial"]
    p_delta = vector_norm(vector_subtract(current["momentum"], initial["momentum"]))
    origin_delta = vector_norm(
        vector_subtract(current["origin_angular"], initial["origin_angular"])
    )
    com_delta = vector_norm(vector_subtract(current["com_angular"], initial["com_angular"]))
    expected_r = tuple(
        initial["r_com"][index] + initial["v_com"][index] * time_year
        for index in range(3)
    )
    values = {
        "relative_engine_energy_drift": abs(
            current["engine_energy"] - initial["engine_energy"]
        )
        / max(abs(initial["engine_energy"]), sys.float_info.min),
        "scale_normalized_intrinsic_energy_residual": abs(
            current["intrinsic_energy"] - initial["intrinsic_energy"]
        )
        / initial["energy_scale"],
        "scale_normalized_linear_momentum_residual": p_delta / initial["linear_scale"],
        "scale_normalized_origin_angular_momentum_residual": origin_delta
        / initial["angular_scale"],
        "scale_normalized_com_angular_momentum_residual": com_delta
        / initial["angular_scale"],
        "relative_initial_linear_momentum_residual": p_delta
        / max(vector_norm(initial["momentum"]), sys.float_info.min),
        "relative_initial_origin_angular_momentum_residual": origin_delta
        / max(vector_norm(initial["origin_angular"]), sys.float_info.min),
        "com_ballistic_position_residual_AU": vector_norm(
            vector_subtract(current["r_com"], expected_r)
        ),
        "angular_decomposition_residual_over_scale": current["decomposition_residual"]
        / initial["angular_scale"],
    }
    for key, value in values.items():
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite derived invariant metric: {key}")
        metric["maximum"][key] = max(metric["maximum"][key], value)
    metric["endpoint"] = values


def pair_key(kind: str, left: str, right: str) -> str:
    return f"{kind}:{left}__{right}"


def pair_definitions(contract: dict[str, Any]) -> list[dict[str, str]]:
    frames = [item["id"] for item in contract["frames"]]
    regimes = [item["id"] for item in contract["numerical_regimes"]]
    definitions: list[dict[str, str]] = []
    for regime in regimes:
        definitions.append({
            "kind": "FRAME",
            "left": f"F0_E1_UNSHIFTED/{regime}",
            "right": f"FCM_ACTIVE_BARYCENTRIC/{regime}",
        })
    for frame in frames:
        for left, right in contract["comparison_policy"]["mercurius_refinement_pairs"]:
            definitions.append({"kind": "MERCURIUS_REFINEMENT", "left": f"{frame}/{left}", "right": f"{frame}/{right}"})
        for left, right in contract["comparison_policy"]["ias15_reference_pairs"]:
            definitions.append({"kind": "IAS15_REFERENCE", "left": f"{frame}/{left}", "right": f"{frame}/{right}"})
        for left, right in contract["comparison_policy"]["mercurius_to_tight_reference"]:
            definitions.append({"kind": "MERCURIUS_TO_IAS15", "left": f"{frame}/{left}", "right": f"{frame}/{right}"})
    if len(definitions) != 20:
        raise RuntimeError("comparison definition cardinality changed")
    return definitions


def update_pair(
    accumulator: dict[str, Any],
    values: dict[str, float],
    time_year: float,
) -> None:
    for name, value in values.items():
        accumulator["maximum"][name] = max(accumulator["maximum"][name], value)
    accumulator["endpoint"] = {"time_year": time_year, **values}


def frame_identity_error(
    unshifted: Any, barycentric: Any, names: list[str], epsilon_factor: float
) -> dict[str, Any]:
    left = state_components(unshifted, names)
    right = state_components(barycentric, names)
    maximum_ratio = 0.0
    within = True
    for name in names:
        for first, second in zip(left[name], right[name], strict=True):
            scale = max(1.0, abs(first), abs(second))
            ratio = abs(first - second) / (sys.float_info.epsilon * scale)
            maximum_ratio = max(maximum_ratio, ratio)
            within = within and ratio <= epsilon_factor
    return {"maximum_binary64_epsilon_units": maximum_ratio, "within_locked_bound": within}


def atomic_checkpoint(path: Path, simulation: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        simulation.save_to_file(str(temporary))
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def enforce_resources(
    contract: dict[str, Any], output_dir: Path, execution_started: float, bundle_started: float
) -> None:
    caps = contract["resource_caps"]
    if time.perf_counter() - execution_started > float(
        caps["max_wall_seconds_total_per_clean_execution"]
    ):
        raise RuntimeError("total wall-time cap exceeded")
    if time.perf_counter() - bundle_started > float(
        caps["max_wall_seconds_per_configuration_bundle"]
    ):
        raise RuntimeError("configuration-bundle wall-time cap exceeded")
    if peak_rss_bytes() > int(caps["max_peak_rss_bytes"]):
        raise RuntimeError("peak-RSS cap exceeded")
    if shutil.disk_usage(output_dir).free < int(caps["minimum_free_disk_bytes"]):
        raise RuntimeError("free-disk floor violated")
    if directory_bytes(output_dir) > int(caps["max_output_bytes"]):
        raise RuntimeError("output-size cap exceeded")


def enforce_final_resources(
    contract: dict[str, Any], output_dir: Path, execution_started: float
) -> None:
    caps = contract["resource_caps"]
    if time.perf_counter() - execution_started > float(
        caps["max_wall_seconds_total_per_clean_execution"]
    ):
        raise RuntimeError("total wall-time cap exceeded")
    if peak_rss_bytes() > int(caps["max_peak_rss_bytes"]):
        raise RuntimeError("peak-RSS cap exceeded")
    if shutil.disk_usage(output_dir).free < int(caps["minimum_free_disk_bytes"]):
        raise RuntimeError("free-disk floor violated")
    if directory_bytes(output_dir) > int(caps["max_output_bytes"]):
        raise RuntimeError("output-size cap exceeded")


def e1_checkpoint_path(
    e1_contract_path: Path,
    e1_result: dict[str, Any],
    configuration_id: str,
    regime_id: str,
    checkpoint_index: int,
) -> Path:
    if regime_id == "MERCURIUS_0125":
        run_key = f"{configuration_id}-b00"
    elif regime_id == "MERCURIUS_00625":
        run_key = f"AUDIT-{configuration_id}-b00"
    else:
        raise ValueError("E1 context requested for unsupported regime")
    if configuration_id == "M0":
        run_key = "M0-b00" if regime_id == "MERCURIUS_0125" else "AUDIT-M0-b00"
    containers = e1_result["provenance"]["arm_records"][run_key]["checkpoint_containers"]
    record = containers[checkpoint_index - 1]
    if record["checkpoint_index"] != checkpoint_index:
        raise RuntimeError("E1 checkpoint ordering changed")
    result_directory = (e1_contract_path.parent / "long_a").resolve()
    path = (result_directory / record["relative_path"]).resolve()
    if not path.is_relative_to(result_directory):
        raise RuntimeError("E1 checkpoint path escapes result directory")
    if sha256_file(path) != record["container_sha256_provenance_only"]:
        raise RuntimeError("E1 checkpoint raw hash changed")
    return path


def run_bundle(
    contract: dict[str, Any],
    contract_path: Path,
    contract_sha256: str,
    output_dir: Path,
    configuration: dict[str, Any],
    e1_module: Any,
    e1_contract: dict[str, Any],
    e1_result: dict[str, Any],
    execution_started: float,
) -> dict[str, Any]:
    rebound = get_rebound()

    bundle_started = time.perf_counter()
    contexts: dict[str, dict[str, Any]] = {}
    names: list[str] | None = None
    scales: dict[str, float] | None = None
    for frame in contract["frames"]:
        for regime in contract["numerical_regimes"]:
            key = f"{frame['id']}/{regime['id']}"
            direct, direct_names, direct_scales = build_active_simulation(
                e1_module, e1_contract, configuration, frame["id"], regime
            )
            chained, chained_names, chained_scales = build_active_simulation(
                e1_module, e1_contract, configuration, frame["id"], regime
            )
            if direct_names != chained_names or direct_scales != chained_scales:
                raise RuntimeError("independent builders disagree on active identity")
            if decoded_state_digest(direct) != decoded_state_digest(chained):
                raise RuntimeError("independent initial decoded states differ")
            if names is None:
                names, scales = direct_names, direct_scales
            elif names != direct_names or scales != direct_scales:
                raise RuntimeError("configuration identity changed across regimes")
            initial_invariant = invariant_snapshot(chained)
            metric = blank_metric(initial_invariant)
            update_metric(metric, initial_invariant, 0.0)
            stream = hashlib.sha256()
            stream.update(b"jx-e2-sampled-state-stream/v1\0")
            stream.update(configuration["id"].encode("ascii"))
            stream.update(key.encode("ascii"))
            stream.update(bytes.fromhex(decoded_state_digest(chained)))
            contexts[key] = {
                "frame": frame,
                "regime": regime,
                "direct": direct,
                "chained": chained,
                "metric": metric,
                "stream": stream,
                "checkpoints_semantic": [],
                "checkpoints_provenance": [],
                "minimum_dt_last_done": None,
                "maximum_dt_last_done": None,
                "ias15_iteration_limit_events": 0,
                "particle_identity": particle_identity(chained),
            }
    assert names is not None and scales is not None

    identity = frame_identity_error(
        contexts["F0_E1_UNSHIFTED/MERCURIUS_0125"]["chained"],
        contexts["FCM_ACTIVE_BARYCENTRIC/MERCURIUS_0125"]["chained"],
        names,
        float(contract["frame_identity_rule"]["absolute_relative_bound_in_binary64_eps"]),
    )
    if not identity["within_locked_bound"]:
        raise RuntimeError("frame construction identity bound failed")

    definitions = pair_definitions(contract)
    pair_accumulators: dict[str, dict[str, Any]] = {}
    for definition in definitions:
        key = pair_key(definition["kind"], definition["left"], definition["right"])
        pair_accumulators[key] = {
            **definition,
            "maximum": {
                "maximum_dimensionless_state_discrepancy": 0.0,
                "maximum_position_separation_AU": 0.0,
                "maximum_velocity_separation_AU_per_year": 0.0,
            },
            "endpoint": None,
        }

    def compare_all(time_year: float) -> None:
        for definition in definitions:
            left = contexts[definition["left"]]["chained"]
            right = contexts[definition["right"]]["chained"]
            values = state_discrepancy(
                left,
                right,
                names,
                scales,
                float(contract["analytic_benchmark"]["G_AU3_Msun_yr2"]),
                float(contract["analytic_benchmark"]["sun_mass_Msun"]),
            )
            key = pair_key(definition["kind"], definition["left"], definition["right"])
            update_pair(pair_accumulators[key], values, time_year)

    compare_all(0.0)
    duration = float(contract["dynamics"]["duration_years"])
    cadence = float(contract["dynamics"]["sample_cadence_years"])
    checkpoint_cadence = float(contract["dynamics"]["checkpoint_cadence_years"])
    samples = int(round(duration / cadence))
    checkpoint_stride = int(round(checkpoint_cadence / cadence))
    e1_context: list[dict[str, Any]] = []

    for sample_index in range(1, samples + 1):
        target = cadence * sample_index
        for key, context in contexts.items():
            context["direct"].integrate(target, exact_finish_time=int(contract["dynamics"]["exact_finish_time"]))
            context["chained"].integrate(target, exact_finish_time=int(contract["dynamics"]["exact_finish_time"]))
            if (
                particle_identity(context["direct"]) != context["particle_identity"]
                or particle_identity(context["chained"]) != context["particle_identity"]
                or context["direct"].N != context["direct"].N_active
                or context["chained"].N != context["chained"].N_active
            ):
                raise RuntimeError("active particle identity/layout changed")
            if decoded_state_digest(context["direct"]) != decoded_state_digest(context["chained"]):
                raise RuntimeError(f"direct/chained state mismatch: {configuration['id']}/{key}/{target}")
            verify_regime(context["direct"], context["regime"])
            verify_regime(context["chained"], context["regime"])
            snapshot = invariant_snapshot(context["chained"])
            update_metric(context["metric"], snapshot, target)
            context["stream"].update(struct.pack("!d", target))
            context["stream"].update(bytes.fromhex(decoded_state_digest(context["chained"])))
            direct_dt_last = abs(float(context["direct"].dt_last_done))
            dt_last = abs(float(context["chained"].dt_last_done))
            if (
                not math.isfinite(direct_dt_last)
                or direct_dt_last <= 0.0
                or not math.isfinite(dt_last)
                or dt_last <= 0.0
            ):
                raise RuntimeError("non-positive/non-finite completed timestep")
            context["minimum_dt_last_done"] = (
                dt_last
                if context["minimum_dt_last_done"] is None
                else min(context["minimum_dt_last_done"], dt_last)
            )
            context["maximum_dt_last_done"] = (
                dt_last
                if context["maximum_dt_last_done"] is None
                else max(context["maximum_dt_last_done"], dt_last)
            )
            if context["regime"]["integrator"] == "ias15":
                events = max(
                    int(context["direct"].ri_ias15._iterations_max_exceeded),
                    int(context["chained"].ri_ias15._iterations_max_exceeded),
                )
                context["ias15_iteration_limit_events"] = max(
                    context["ias15_iteration_limit_events"], events
                )
                if events != 0:
                    raise RuntimeError("IAS15 iteration-limit event")
        compare_all(target)
        enforce_resources(contract, output_dir, execution_started, bundle_started)

        if sample_index % checkpoint_stride == 0:
            checkpoint_index = sample_index // checkpoint_stride
            for key, context in contexts.items():
                checkpoint_path = (
                    output_dir
                    / "checkpoints"
                    / configuration["id"]
                    / key.replace("/", "__")
                    / f"checkpoint_{checkpoint_index:02d}.bin"
                )
                before = decoded_state_digest(context["chained"])
                projected_bytes = directory_bytes(output_dir) + 262144
                if projected_bytes > int(contract["resource_caps"]["max_output_bytes"]):
                    raise RuntimeError("checkpoint pre-write output cap would be exceeded")
                atomic_checkpoint(checkpoint_path, context["chained"])
                if checkpoint_path.stat().st_size > int(
                    contract["resource_caps"]["max_checkpoint_bytes"]
                ):
                    raise RuntimeError("checkpoint size cap exceeded")
                loaded = rebound.Simulation(str(checkpoint_path))
                verify_regime(loaded, context["regime"])
                after = decoded_state_digest(loaded)
                if before != after:
                    raise RuntimeError("checkpoint decoded state changed")
                if decoded_state_digest(context["direct"]) != after:
                    raise RuntimeError("direct/reloaded state mismatch")
                context["chained"] = loaded
                semantic_checkpoint = {
                    "checkpoint_index": checkpoint_index,
                    "time_year": target,
                    "decoded_state_sha256": after,
                }
                provenance_checkpoint = {
                    **semantic_checkpoint,
                    "relative_path": str(checkpoint_path.relative_to(output_dir)),
                    "container_bytes": checkpoint_path.stat().st_size,
                    "container_sha256_provenance_only": sha256_file(checkpoint_path),
                }
                context["checkpoints_semantic"].append(semantic_checkpoint)
                context["checkpoints_provenance"].append(provenance_checkpoint)

            for regime_id in ("MERCURIUS_0125", "MERCURIUS_00625"):
                context = contexts[f"F0_E1_UNSHIFTED/{regime_id}"]
                e1_path = e1_checkpoint_path(
                    (contract_path.parent / contract["e1_immutable_boundary"]["contract_path"]).resolve(),
                    e1_result,
                    configuration["id"],
                    regime_id,
                    checkpoint_index,
                )
                e1_loaded = rebound.Simulation(str(e1_path))
                e2_digest = active_projection_digest(context["chained"])
                e1_digest = active_projection_digest(e1_loaded)
                e1_context.append({
                    "regime_id": regime_id,
                    "checkpoint_index": checkpoint_index,
                    "time_year": target,
                    "e2_active_projection_sha256": e2_digest,
                    "e1_active_projection_sha256": e1_digest,
                    "exact": e2_digest == e1_digest,
                })

    arms = []
    checkpoint_provenance: dict[str, Any] = {}
    inherited = contract["inherited_reference_flags"]
    for key, context in sorted(contexts.items()):
        if len(context["checkpoints_semantic"]) != int(
            contract["dynamics"]["expected_checkpoints_per_arm"]
        ):
            raise RuntimeError("checkpoint count mismatch")
        metric = context["metric"]
        legacy_applicable = context["frame"]["id"] == "F0_E1_UNSHIFTED"
        reference_flags = {
            "legacy_E1_reference_applicable": legacy_applicable,
            "relative_energy_within_legacy_E1_reference": (
                metric["maximum"]["relative_engine_energy_drift"]
                <= float(inherited["maximum_relative_energy_drift"])
                if legacy_applicable
                else None
            ),
            "compensated_origin_angular_within_legacy_numeric_value_illustration": (
                metric["maximum"]["relative_initial_origin_angular_momentum_residual"]
                <= float(inherited["maximum_relative_origin_angular_momentum_drift"])
                if legacy_applicable
                else None
            ),
            "origin_angular_is_not_exact_E1_evaluator": True,
            "Pstar_and_Lstar_metrics_are_not_legacy_E1_thresholds": True,
        }
        arm_semantic = {
            "schema": ARM_SCHEMA,
            "arm_key": key,
            "frame_id": context["frame"]["id"],
            "regime_id": context["regime"]["id"],
            "integrator": context["regime"]["integrator"],
            "sample_count": samples + 1,
            "final_time_year": float(context["chained"].t),
            "initial_physical_state_sha256": physical_state_digest(
                build_active_simulation(
                    e1_module,
                    e1_contract,
                    configuration,
                    context["frame"]["id"],
                    context["regime"],
                )[0]
            ),
            "endpoint_decoded_state_sha256": decoded_state_digest(context["chained"]),
            "sampled_state_stream_sha256": context["stream"].hexdigest(),
            "checkpoints": context["checkpoints_semantic"],
            "minimum_dt_last_done_years": context["minimum_dt_last_done"],
            "maximum_dt_last_done_years": context["maximum_dt_last_done"],
            "ias15_iteration_limit_events": context["ias15_iteration_limit_events"],
            "invariant_metrics": metric,
            "inherited_reference_flags": reference_flags,
            "integrity": {
                "direct_and_chained_exact_every_sample": True,
                "checkpoint_decoded_states_exact": True,
                "states_and_invariants_finite": True,
                "particle_identity_unchanged": True,
                "integrator_settings_readback_exact": True,
            },
        }
        arms.append({
            "semantic": arm_semantic,
            "semantic_sha256": sha256_bytes(canonical_bytes(arm_semantic)),
        })
        checkpoint_provenance[key] = context["checkpoints_provenance"]

    bundle_semantic = {
        "schema": BUNDLE_SCHEMA,
        "contract_sha256": contract_sha256,
        "configuration_id": configuration["id"],
        "frame_identity": identity,
        "arms": arms,
        "comparisons": [pair_accumulators[key] for key in sorted(pair_accumulators)],
        "e1_context_comparisons": e1_context,
        "integrity": {
            "arm_count_exact": len(arms) == 12,
            "comparison_count_exact": len(pair_accumulators) == 20,
            "e1_context_count_exact": len(e1_context) == 20,
            "all_frame_identity_bounds_met": identity["within_locked_bound"],
            "all_direct_chained_and_checkpoint_checks_exact": True,
        },
    }
    if not all(bundle_semantic["integrity"].values()):
        raise RuntimeError("bundle integrity check failed")
    enforce_resources(contract, output_dir, execution_started, bundle_started)
    return {
        "schema": BUNDLE_SCHEMA,
        "semantic": bundle_semantic,
        "semantic_sha256": sha256_bytes(canonical_bytes(bundle_semantic)),
        "provenance": {
            "elapsed_seconds": time.perf_counter() - bundle_started,
            "peak_rss_bytes": peak_rss_bytes(),
            "checkpoint_containers": checkpoint_provenance,
        },
    }


def comparison_lookup(bundle_semantic: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (item["kind"], item["left"], item["right"]): item
        for item in bundle_semantic["comparisons"]
    }


def arm_lookup(bundle_semantic: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["semantic"]["arm_key"]: item["semantic"] for item in bundle_semantic["arms"]}


def configuration_classification(
    contract: dict[str, Any], bundle_semantic: dict[str, Any]
) -> dict[str, Any]:
    comparisons = comparison_lookup(bundle_semantic)
    arms = arm_lookup(bundle_semantic)
    floor = float(contract["classification_policy"]["state_floor"])
    reference_ceiling = float(
        contract["classification_policy"]["ias15_reference_absolute_ceiling"]
    )
    frame_results: dict[str, Any] = {}
    for frame in ("F0_E1_UNSHIFTED", "FCM_ACTIVE_BARYCENTRIC"):
        d10_12 = comparisons[(
            "IAS15_REFERENCE",
            f"{frame}/IAS15_1E10",
            f"{frame}/IAS15_1E12",
        )]["maximum"]["maximum_dimensionless_state_discrepancy"]
        d12_14 = comparisons[(
            "IAS15_REFERENCE",
            f"{frame}/IAS15_1E12",
            f"{frame}/IAS15_1E14",
        )]["maximum"]["maximum_dimensionless_state_discrepancy"]
        reference = d12_14 <= reference_ceiling and (
            d12_14 <= 0.25 * d10_12 or (d10_12 <= floor and d12_14 <= floor)
        )
        tau = max(floor, 10.0 * d12_14)
        errors = []
        for regime in ("MERCURIUS_0125", "MERCURIUS_00625", "MERCURIUS_003125"):
            errors.append(comparisons[(
                "MERCURIUS_TO_IAS15",
                f"{frame}/{regime}",
                f"{frame}/IAS15_1E14",
            )]["maximum"]["maximum_dimensionless_state_discrepancy"])
        refinement = (errors[1] <= errors[0] / 2.0 or errors[1] <= tau) and (
            errors[2] <= errors[1] / 2.0 or errors[2] <= tau
        )
        divergent = (
            errors[1] > 2.0 * errors[0]
            and errors[2] > 2.0 * errors[1]
            and errors[2] > 10.0 * tau
        )
        frame_results[frame] = {
            "ias15_reference_established": reference,
            "state_equivalence_band": tau,
            "mercurius_errors_to_ias15_1e14": errors,
            "mercurius_refinement_consistent": refinement,
            "two_step_refinement_divergence": divergent,
        }

    all_reference = all(
        item["ias15_reference_established"] for item in frame_results.values()
    )
    tau_frame = max(item["state_equivalence_band"] for item in frame_results.values())
    max_frame_error = max(
        item["maximum"]["maximum_dimensionless_state_discrepancy"]
        for item in bundle_semantic["comparisons"]
        if item["kind"] == "FRAME"
    )
    frame_equivalent = all_reference and max_frame_error <= tau_frame
    arm_frames = {
        frame: {
            regime: arms[f"{frame}/{regime}"]
            for regime in (
                "MERCURIUS_0125",
                "MERCURIUS_00625",
                "MERCURIUS_003125",
                "IAS15_1E14",
            )
        }
        for frame in ("F0_E1_UNSHIFTED", "FCM_ACTIVE_BARYCENTRIC")
    }
    p_values_by_frame = {
        frame: [
            frame_arms[regime]["invariant_metrics"]["maximum"][
                "scale_normalized_linear_momentum_residual"
            ]
            for regime in (
                "MERCURIUS_0125",
                "MERCURIUS_00625",
                "MERCURIUS_003125",
            )
        ]
        for frame, frame_arms in arm_frames.items()
    }
    invariant_floor = float(contract["classification_policy"]["invariant_floor"])
    step_count_by_frame = {
        frame: (
            values[2] > invariant_floor
            and 1.5 * values[0] <= values[1] <= 3.0 * values[0]
            and 1.5 * values[1] <= values[2] <= 3.0 * values[1]
        )
        for frame, values in p_values_by_frame.items()
    }
    step_count_indicator = all(step_count_by_frame.values())
    native_arms = arm_frames["F0_E1_UNSHIFTED"]

    origin_native = native_arms["MERCURIUS_003125"]["invariant_metrics"]["maximum"][
        "scale_normalized_origin_angular_momentum_residual"
    ]
    origin_bary = arms["FCM_ACTIVE_BARYCENTRIC/MERCURIUS_003125"]["invariant_metrics"]["maximum"][
        "scale_normalized_origin_angular_momentum_residual"
    ]
    com_native = native_arms["MERCURIUS_003125"]["invariant_metrics"]["maximum"][
        "scale_normalized_com_angular_momentum_residual"
    ]
    com_bary = arms["FCM_ACTIVE_BARYCENTRIC/MERCURIUS_003125"]["invariant_metrics"]["maximum"][
        "scale_normalized_com_angular_momentum_residual"
    ]
    def symmetric_ratio(first: float, second: float) -> float:
        return max(first, second, invariant_floor) / max(min(first, second), invariant_floor)

    frame_indicator = (
        frame_equivalent
        and symmetric_ratio(origin_native, origin_bary) >= 4.0
        and symmetric_ratio(com_native, com_bary) < 4.0
    )
    ias_quarter_by_frame = {}
    for frame, frame_arms in arm_frames.items():
        fine_linear = p_values_by_frame[frame][2]
        fine_com = frame_arms["MERCURIUS_003125"]["invariant_metrics"]["maximum"][
            "scale_normalized_com_angular_momentum_residual"
        ]
        ias_linear = frame_arms["IAS15_1E14"]["invariant_metrics"]["maximum"][
            "scale_normalized_linear_momentum_residual"
        ]
        ias_com = frame_arms["IAS15_1E14"]["invariant_metrics"]["maximum"][
            "scale_normalized_com_angular_momentum_residual"
        ]
        linear_within_quarter = (
            fine_linear > invariant_floor and ias_linear <= 0.25 * fine_linear
        )
        com_within_quarter = (
            fine_com > invariant_floor and ias_com <= 0.25 * fine_com
        )
        ias_quarter_by_frame[frame] = {
            "linear_momentum": linear_within_quarter,
            "com_angular_momentum": com_within_quarter,
            "both_metrics": linear_within_quarter and com_within_quarter,
        }
    ias_indicator = all(
        item["both_metrics"] for item in ias_quarter_by_frame.values()
    )
    indicators = {
        "step_count_scaling": step_count_indicator,
        "frame_intrinsic_metric_sensitivity": frame_indicator,
        "ias15_residual_at_most_quarter_fine_mercurius": ias_indicator,
    }
    refinement = all(item["mercurius_refinement_consistent"] for item in frame_results.values())
    instability = all_reference and all(
        item["two_step_refinement_divergence"] for item in frame_results.values()
    )
    origin_frame_effect = frame_indicator
    linear_step_signature = (
        all_reference
        and refinement
        and step_count_indicator
        and ias_indicator
    )
    if instability:
        classification = "MERCURIUS_REFINEMENT_DIVERGENCE_SUSPECTED"
    elif origin_frame_effect and linear_step_signature:
        classification = "FRAME_EFFECT_AND_LINEAR_STEP_SIGNATURE_CONSISTENT"
    elif all_reference and refinement:
        classification = "MERCURIUS_REFINEMENT_CONSISTENT"
    else:
        classification = "MIXED_OR_INCONCLUSIVE"
    return {
        "configuration_id": bundle_semantic["configuration_id"],
        "classification": classification,
        "frame_results": frame_results,
        "maximum_frame_state_discrepancy": max_frame_error,
        "frame_state_equivalent": frame_equivalent,
        "forensic_signature_indicators": indicators,
        "mechanism_details": {
            "origin_angular_frame_effect_consistent": origin_frame_effect,
            "linear_momentum_step_signature_consistent": linear_step_signature,
            "linear_step_signature_by_frame": step_count_by_frame,
            "ias15_quarter_residual_by_frame": ias_quarter_by_frame,
        },
        "e1_active_context_exact_count": sum(
            item["exact"] for item in bundle_semantic["e1_context_comparisons"]
        ),
        "e1_active_context_total": len(bundle_semantic["e1_context_comparisons"]),
    }


def overall_classification(configuration_results: list[dict[str, Any]]) -> str:
    if (
        len(configuration_results) != 8
        or len({item.get("configuration_id") for item in configuration_results}) != 8
    ):
        raise ValueError("overall classification requires the exact eight configurations")
    if any(
        item["classification"] == "MERCURIUS_REFINEMENT_DIVERGENCE_SUSPECTED"
        for item in configuration_results
    ):
        return "MERCURIUS_REFINEMENT_DIVERGENCE_SUSPECTED"
    if all(
        item["classification"]
        == "FRAME_EFFECT_AND_LINEAR_STEP_SIGNATURE_CONSISTENT"
        for item in configuration_results
    ):
        return "FRAME_EFFECT_AND_LINEAR_STEP_SIGNATURE_CONSISTENT"
    return "MIXED_OR_INCONCLUSIVE"


def validate_contract(contract: dict[str, Any], contract_path: Path) -> None:
    if sha256_file(contract_path) != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("contract hash does not match the frozen runner binding")
    if contract["schema"] != "jx-e2-numerics-contract/v1":
        raise ValueError("contract schema changed")
    if contract["experiment_id"] != "jx-e2-active-frame-integrator-50k-v1":
        raise ValueError("experiment identity changed")
    if contract["claim_ceiling"] != "NUMERICAL_METHOD_FORENSICS_ONLY":
        raise ValueError("claim ceiling changed")
    expected_permissions = {
        "local_cpu_numerical_diagnostic_authorized": True,
        "gpu_execution_authorized": False,
        "network_access_authorized": False,
        "observed_data_access_authorized": False,
        "jx_e1_reclassification_authorized": False,
        "jx_e1_execution_b_authorized": False,
        "jx_o2_execution_authorized": False,
        "scientific_planet_x_claim_authorized": False,
    }
    if contract["permissions"] != expected_permissions:
        raise ValueError("permission boundary changed")
    if len(contract["configuration_set"]) != 8 or len(
        {item["id"] for item in contract["configuration_set"]}
    ) != 8:
        raise ValueError("configuration set changed")
    if len(contract["frames"]) != 2 or len(contract["numerical_regimes"]) != 6:
        raise ValueError("frame/regime matrix changed")
    if int(contract["dynamics"]["expected_arm_count"]) != 96:
        raise ValueError("expected arm count changed")
    if int(contract["dynamics"]["expected_pairwise_comparison_count"]) != 160:
        raise ValueError("expected pairwise-comparison count changed")
    if int(contract["dynamics"]["expected_e1_context_record_count"]) != 160:
        raise ValueError("expected E1-context count changed")
    if contract["outcomes_generated_at_registration"] is not False:
        raise ValueError("contract contains outcome status")
    if contract["inherited_reference_flags"]["affect_e2_validity"] is not False:
        raise ValueError("reference flags were promoted to validity gates")


def validate_runtime() -> dict[str, Any]:
    rebound = get_rebound()

    binary_path = Path(rebound.clibrebound._name).resolve()
    if not binary_path.is_file():
        raise RuntimeError("cannot resolve REBOUND binary")
    runtime = {
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "rebound_version": rebound.__version__,
        "rebound_build": rebound.__build__,
        "rebound_binary_sha256": sha256_file(binary_path),
        "rebound_python_source_sha256": rebound_python_source_sha256(rebound),
    }
    expected = {
        "python_version": "3.12.13",
        "rebound_version": "4.4.11",
        "rebound_build": "Nov 13 2025 14:44:51",
        "rebound_binary_sha256": REBOUND_BINARY_SHA256,
        "rebound_python_source_sha256": REBOUND_PYTHON_SOURCE_SHA256,
    }
    if runtime != expected:
        raise RuntimeError(f"runtime mismatch: {runtime}")
    return runtime


def validate_registration(
    registration_path: Path, contract_path: Path, runner_path: Path
) -> tuple[dict[str, Any], str]:
    registration = strict_json(registration_path)
    contract = strict_json(contract_path)
    if registration.get("schema") != "jx-e2-numerics-local-registration/v1":
        raise ValueError("registration schema changed")
    if set(registration) != {
        "schema",
        "experiment_id",
        "artifact_class",
        "registration_state",
        "recorded_at_utc",
        "timestamp_authority",
        "externally_timestamped",
        "scientific_evidence_artifact",
        "outcomes_generated",
        "execution_permissions",
        "locked_files",
        "mandatory_nonclaim",
    }:
        raise ValueError("registration top-level shape changed")
    expected_execution_permissions = {
        "execution_a_authorized": True,
        "execution_b_authorized_only_after_verified_a": True,
        "gpu_execution_authorized": False,
        "network_access_authorized": False,
        "observed_data_access_authorized": False,
        "jx_o2_execution_authorized": False,
        "scientific_planet_x_claim_authorized": False,
    }
    if (
        registration["experiment_id"] != contract["experiment_id"]
        or registration["artifact_class"] != "LOCAL_CONTENT_HASH_REGISTRATION_ONLY"
        or registration["registration_state"]
        != "LOCAL_CONTENT_HASH_LOCK_COMPLETE_BEFORE_ANY_E2_50KYR_OUTPUT"
        or registration["timestamp_authority"]
        != "LOCAL_CONTENT_HASH_REGISTRATION_ONLY_NO_EXTERNAL_TIMESTAMP"
        or registration["externally_timestamped"] is not False
        or registration["scientific_evidence_artifact"] is not False
        or registration["outcomes_generated"] is not False
        or registration["execution_permissions"] != expected_execution_permissions
        or registration["mandatory_nonclaim"] != contract["mandatory_nonclaim"]
        or not isinstance(registration["recorded_at_utc"], str)
        or not registration["recorded_at_utc"].endswith("Z")
    ):
        raise ValueError("registration identity/permission boundary changed")
    locked = registration.get("locked_files")
    if not isinstance(locked, dict):
        raise ValueError("registration locked-file map missing")
    expected_locked = {
        "README.md",
        "contract_v1.json",
        "run_numerics.py",
        "verify_replay.py",
        "test_jx_e2.py",
    }
    if set(locked) != expected_locked:
        raise ValueError("registration locked-file set changed")
    root = registration_path.parent.resolve()
    if (
        contract_path.resolve() != root / "contract_v1.json"
        or runner_path.resolve() != root / "run_numerics.py"
        or registration_path.resolve() != root / "registration_v1.json"
    ):
        raise ValueError("registration inputs are not canonical package paths")
    expected_contract = locked.get(contract_path.name)
    expected_runner = locked.get(runner_path.name)
    if expected_contract != sha256_file(contract_path):
        raise RuntimeError("registration contract hash mismatch")
    if expected_runner != sha256_file(runner_path):
        raise RuntimeError("registration runner hash mismatch")
    for relative, expected in locked.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"registration locked file mismatch: {relative}")
    return registration, sha256_file(registration_path)


def validate_a_for_b(
    contract: dict[str, Any],
    registration_path: Path,
    registration_sha256: str,
    a_result_path: Path,
) -> None:
    a_result_path = a_result_path.resolve()
    if a_result_path.name != "result_v1.json" or not a_result_path.is_file():
        raise ValueError("E2-A result path is not canonical")
    a_directory = a_result_path.parent
    if (a_directory / "failure_receipt.json").exists():
        raise RuntimeError("E2-A directory has a permanent failure receipt")
    registration = strict_json(registration_path)
    verifier_path = registration_path.parent / "verify_replay.py"
    verifier = load_source_module(
        "jx_e2_locked_verifier_for_b_authorization",
        verifier_path,
        registration["locked_files"]["verify_replay.py"],
    )
    verifier.validate_actual_runtime()
    verified = verifier.verify_output(contract, registration_sha256, a_directory)
    if verified["manifest"]["execution_label"] != "E2-A":
        raise RuntimeError("E2-A execution label mismatch")


def execute(
    contract_path: Path,
    registration_path: Path,
    output_dir: Path,
    execution_label: str,
    a_result: Path | None,
    command_started: float | None = None,
) -> dict[str, Any]:
    started = time.perf_counter() if command_started is None else command_started
    contract_path = contract_path.resolve()
    registration_path = registration_path.resolve()
    output_dir = output_dir.resolve()
    runner_path = Path(__file__).resolve()
    contract = strict_json(contract_path)
    validate_contract(contract, contract_path)
    registration, registration_sha256 = validate_registration(
        registration_path, contract_path, runner_path
    )
    runtime = validate_runtime()
    contract_sha256 = sha256_file(contract_path)
    labels = contract["result_policy"]["clean_execution_labels"]
    if execution_label not in labels:
        raise ValueError("execution label is not predeclared")
    if execution_label == "E2-B":
        if a_result is None:
            raise ValueError("E2-B requires the E2-A result path")
        validate_a_for_b(
            contract,
            registration_path,
            registration_sha256,
            a_result,
        )
        a_directory = a_result.resolve().parent
        if (
            output_dir == a_directory
            or output_dir in a_directory.parents
            or a_directory in output_dir.parents
        ):
            raise ValueError("E2-A and E2-B output roots must be disjoint")
    e1_directory = (contract_path.parent / contract["e1_immutable_boundary"]["directory"]).resolve()
    if output_dir == e1_directory or e1_directory in output_dir.parents:
        raise ValueError("E2 output may not be inside E1")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("clean E2 output directory must be absent or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result_v1.json"
    failure_path = output_dir / "failure_receipt.json"
    manifest = {
        "schema": "jx-e2-numerics-run-manifest/v1",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": contract_sha256,
        "registration_sha256": registration_sha256,
        "runner_sha256": sha256_file(runner_path),
        "execution_label": execution_label,
        "execution_instance_id": secrets.token_hex(16),
        "runtime": runtime,
    }
    atomic_json(output_dir / "run_manifest.json", manifest)
    bundles = []
    try:
        e1_module, e1_contract, e1_result = load_e1(contract, contract_path)
        for index, configuration in enumerate(contract["configuration_set"], start=1):
            bundle = run_bundle(
                contract,
                contract_path,
                contract_sha256,
                output_dir,
                configuration,
                e1_module,
                e1_contract,
                e1_result,
                started,
            )
            bundle_path = output_dir / "bundles" / f"{configuration['id']}.json"
            projected = directory_bytes(output_dir) + len(serialized_json(bundle))
            if projected > int(contract["resource_caps"]["max_output_bytes"]):
                raise RuntimeError("bundle write would exceed output cap")
            atomic_json(bundle_path, bundle)
            bundles.append(bundle)
            print(f"[E2 {index:02d}/08] {configuration['id']} complete", flush=True)

        classifications = [
            configuration_classification(contract, bundle["semantic"]) for bundle in bundles
        ]
        semantic = {
            "schema": RESULT_SCHEMA,
            "experiment_id": contract["experiment_id"],
            "contract_sha256": contract_sha256,
            "registration_sha256": registration_sha256,
            "claim_ceiling": contract["claim_ceiling"],
            "execution_integrity": {
                "bundle_count_exact": len(bundles) == 8,
                "arm_count_exact": sum(len(bundle["semantic"]["arms"]) for bundle in bundles)
                == int(contract["dynamics"]["expected_arm_count"]),
                "comparison_count_exact": sum(
                    len(bundle["semantic"]["comparisons"]) for bundle in bundles
                )
                == int(contract["dynamics"]["expected_pairwise_comparison_count"]),
                "e1_context_count_exact": sum(
                    len(bundle["semantic"]["e1_context_comparisons"])
                    for bundle in bundles
                )
                == int(contract["dynamics"]["expected_e1_context_record_count"]),
                "all_bundle_integrity_checks_true": all(
                    all(value is True for value in bundle["semantic"]["integrity"].values())
                    for bundle in bundles
                ),
                "all_ias15_iteration_limit_events_zero": all(
                    arm["semantic"]["ias15_iteration_limit_events"] == 0
                    for bundle in bundles
                    for arm in bundle["semantic"]["arms"]
                ),
                "inherited_reference_flags_do_not_control_validity": contract["inherited_reference_flags"]["affect_e2_validity"] is False,
            },
            "bundle_semantic_hashes": {
                bundle["semantic"]["configuration_id"]: bundle["semantic_sha256"]
                for bundle in bundles
            },
            "configuration_classifications": classifications,
            "overall_numerical_classification": overall_classification(classifications),
            "mandatory_nonclaim": contract["mandatory_nonclaim"],
        }
        if not all(semantic["execution_integrity"].values()):
            raise RuntimeError("result integrity check failed")
        result = {
            "schema": RESULT_SCHEMA,
            "experiment_id": contract["experiment_id"],
            "verdict": contract["result_policy"]["first_execution_verdict"],
            "claim_ceiling": contract["claim_ceiling"],
            "semantic": semantic,
            "semantic_sha256": sha256_bytes(canonical_bytes(semantic)),
            "provenance": {
                "execution_label": execution_label,
                "execution_instance_id": manifest["execution_instance_id"],
                "elapsed_seconds": time.perf_counter() - started,
                "peak_rss_bytes": peak_rss_bytes(),
                "output_bytes_before_result": directory_bytes(output_dir),
                "output_directory": str(output_dir),
            },
            "nonclaim": contract["mandatory_nonclaim"],
        }
        projected = directory_bytes(output_dir) + len(serialized_json(result))
        if projected > int(contract["resource_caps"]["max_output_bytes"]):
            raise RuntimeError("result write would exceed output cap")
        enforce_final_resources(contract, output_dir, started)
        atomic_json(result_path, result)
        enforce_final_resources(contract, output_dir, started)
        return result
    except BaseException as exc:
        failure = {
            "schema": "jx-e2-numerics-failure/v1",
            "experiment_id": contract["experiment_id"],
            "contract_sha256": contract_sha256,
            "verdict": contract["result_policy"]["invalid_execution_verdict"],
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "elapsed_seconds": time.perf_counter() - started,
            "completed_bundles": [bundle["semantic"]["configuration_id"] for bundle in bundles],
            "nonclaim": contract["mandatory_nonclaim"],
        }
        atomic_json(failure_path, failure)
        raise


def main() -> int:
    command_started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execution-label", required=True)
    parser.add_argument("--a-result", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args()
    contract_path = arguments.contract.resolve()
    registration_path = arguments.registration.resolve()
    contract = strict_json(contract_path)
    validate_contract(contract, contract_path)
    validate_registration(registration_path, contract_path, Path(__file__).resolve())
    runtime = validate_runtime()
    load_e1(contract, contract_path)
    if arguments.execution_label not in contract["result_policy"]["clean_execution_labels"]:
        raise ValueError("execution label is not predeclared")
    if arguments.validate_only:
        print(json.dumps({
            "status": "JX_E2_VALIDATE_ONLY_OK",
            "claim_ceiling": contract["claim_ceiling"],
            "contract_sha256": sha256_file(contract_path),
            "registration_sha256": sha256_file(registration_path),
            "runtime": runtime,
            "arm_count": 96,
            "dynamics_executed": False,
        }, sort_keys=True))
        return 0
    if arguments.output is None:
        parser.error("--output is required unless --validate-only is used")
    result = execute(
        contract_path,
        registration_path,
        arguments.output,
        arguments.execution_label,
        arguments.a_result,
        command_started,
    )
    print(json.dumps({
        "verdict": result["verdict"],
        "claim_ceiling": result["claim_ceiling"],
        "classification": result["semantic"]["overall_numerical_classification"],
        "semantic_sha256": result["semantic_sha256"],
        "elapsed_seconds": result["provenance"]["elapsed_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
