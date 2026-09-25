#!/usr/bin/env python3
"""Verify two clean JX-E2 executions without running new dynamics."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import secrets
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any


EXPECTED_CONTRACT_SHA256 = "d4edc6e17df40c3eeb6a72c7c55ad3bb530e6c79a40017f6dffe9ec553bc3d8f"
EXPECTED_RUNNER_SHA256 = "223735e8256ae86ee00bc0105a33b88a6a225ba556414201b27cacb0a7ab7f9d"
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
    result = {}
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
        raise ValueError("JSON root must be an object")
    return value


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(serialized_json(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def recursive_file_manifest_sha256(root: Path) -> str:
    paths = list(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("content-addressed output contains a symlink")
    if any(not path.is_file() and not path.is_dir() for path in paths):
        raise ValueError("content-addressed output has a non-regular entry")
    manifest = []
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            manifest.append({"relative_path": relative, "kind": "directory"})
        else:
            manifest.append(
                {
                    "relative_path": relative,
                    "kind": "file",
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return sha256_bytes(canonical_bytes(manifest))


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


def decoded_state_digest(simulation: Any) -> str:
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
        raise ValueError("unexpected checkpoint integrator")
    digest = hashlib.sha256()
    digest.update(b"jx-e2-decoded-active-state/v1\0")
    digest.update(canonical_bytes(configuration))
    return digest.hexdigest()


def comparison_lookup(bundle: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    result = {}
    for item in bundle["comparisons"]:
        key = (item["kind"], item["left"], item["right"])
        if key in result:
            raise ValueError("duplicate comparison key")
        result[key] = item
    return result


def arm_lookup(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for item in bundle["arms"]:
        if set(item) != {"semantic", "semantic_sha256"}:
            raise ValueError("arm wrapper shape mismatch")
        arm = item["semantic"]
        if sha256_bytes(canonical_bytes(arm)) != item["semantic_sha256"]:
            raise ValueError("arm semantic hash mismatch")
        if arm["arm_key"] in result:
            raise ValueError("duplicate arm key")
        result[arm["arm_key"]] = arm
    return result


def recompute_configuration_classification(
    contract: dict[str, Any], bundle: dict[str, Any]
) -> dict[str, Any]:
    comparisons = comparison_lookup(bundle)
    arms = arm_lookup(bundle)
    floor = float(contract["classification_policy"]["state_floor"])
    reference_ceiling = float(
        contract["classification_policy"]["ias15_reference_absolute_ceiling"]
    )
    frame_results = {}
    for frame in ("F0_E1_UNSHIFTED", "FCM_ACTIVE_BARYCENTRIC"):
        d10_12 = comparisons[("IAS15_REFERENCE", f"{frame}/IAS15_1E10", f"{frame}/IAS15_1E12")]["maximum"]["maximum_dimensionless_state_discrepancy"]
        d12_14 = comparisons[("IAS15_REFERENCE", f"{frame}/IAS15_1E12", f"{frame}/IAS15_1E14")]["maximum"]["maximum_dimensionless_state_discrepancy"]
        reference = d12_14 <= reference_ceiling and (
            d12_14 <= 0.25 * d10_12 or (d10_12 <= floor and d12_14 <= floor)
        )
        tau = max(floor, 10.0 * d12_14)
        errors = [
            comparisons[("MERCURIUS_TO_IAS15", f"{frame}/{regime}", f"{frame}/IAS15_1E14")]["maximum"]["maximum_dimensionless_state_discrepancy"]
            for regime in ("MERCURIUS_0125", "MERCURIUS_00625", "MERCURIUS_003125")
        ]
        refinement = (errors[1] <= errors[0] / 2.0 or errors[1] <= tau) and (
            errors[2] <= errors[1] / 2.0 or errors[2] <= tau
        )
        divergent = errors[1] > 2.0 * errors[0] and errors[2] > 2.0 * errors[1] and errors[2] > 10.0 * tau
        frame_results[frame] = {
            "ias15_reference_established": reference,
            "state_equivalence_band": tau,
            "mercurius_errors_to_ias15_1e14": errors,
            "mercurius_refinement_consistent": refinement,
            "two_step_refinement_divergence": divergent,
        }
    all_reference = all(
        value["ias15_reference_established"] for value in frame_results.values()
    )
    tau_frame = max(value["state_equivalence_band"] for value in frame_results.values())
    max_frame_error = max(
        item["maximum"]["maximum_dimensionless_state_discrepancy"]
        for item in bundle["comparisons"]
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
    step_by_frame = {
        frame: (
            values[2] > invariant_floor
            and 1.5 * values[0] <= values[1] <= 3.0 * values[0]
            and 1.5 * values[1] <= values[2] <= 3.0 * values[1]
        )
        for frame, values in p_values_by_frame.items()
    }
    step_indicator = all(step_by_frame.values())
    native = arm_frames["F0_E1_UNSHIFTED"]
    origin_native = native["MERCURIUS_003125"]["invariant_metrics"]["maximum"]["scale_normalized_origin_angular_momentum_residual"]
    origin_bary = arms["FCM_ACTIVE_BARYCENTRIC/MERCURIUS_003125"]["invariant_metrics"]["maximum"]["scale_normalized_origin_angular_momentum_residual"]
    com_native = native["MERCURIUS_003125"]["invariant_metrics"]["maximum"]["scale_normalized_com_angular_momentum_residual"]
    com_bary = arms["FCM_ACTIVE_BARYCENTRIC/MERCURIUS_003125"]["invariant_metrics"]["maximum"]["scale_normalized_com_angular_momentum_residual"]
    def ratio(first: float, second: float) -> float:
        return max(first, second, invariant_floor) / max(min(first, second), invariant_floor)

    frame_indicator = frame_equivalent and ratio(origin_native, origin_bary) >= 4.0 and ratio(com_native, com_bary) < 4.0
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
        "step_count_scaling": step_indicator,
        "frame_intrinsic_metric_sensitivity": frame_indicator,
        "ias15_residual_at_most_quarter_fine_mercurius": ias_indicator,
    }
    refinement = all(value["mercurius_refinement_consistent"] for value in frame_results.values())
    instability = all_reference and all(
        value["two_step_refinement_divergence"] for value in frame_results.values()
    )
    origin_frame_effect = frame_indicator
    linear_step_signature = (
        all_reference
        and refinement
        and step_indicator
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
        "configuration_id": bundle["configuration_id"],
        "classification": classification,
        "frame_results": frame_results,
        "maximum_frame_state_discrepancy": max_frame_error,
        "frame_state_equivalent": frame_equivalent,
        "forensic_signature_indicators": indicators,
        "mechanism_details": {
            "origin_angular_frame_effect_consistent": origin_frame_effect,
            "linear_momentum_step_signature_consistent": linear_step_signature,
            "linear_step_signature_by_frame": step_by_frame,
            "ias15_quarter_residual_by_frame": ias_quarter_by_frame,
        },
        "e1_active_context_exact_count": sum(item["exact"] for item in bundle["e1_context_comparisons"]),
        "e1_active_context_total": len(bundle["e1_context_comparisons"]),
    }


def recompute_overall(results: list[dict[str, Any]]) -> str:
    if (
        len(results) != 8
        or len({item.get("configuration_id") for item in results}) != 8
    ):
        raise ValueError("overall classification requires the exact eight configurations")
    if any(
        item["classification"] == "MERCURIUS_REFINEMENT_DIVERGENCE_SUSPECTED"
        for item in results
    ):
        return "MERCURIUS_REFINEMENT_DIVERGENCE_SUSPECTED"
    if all(
        item["classification"]
        == "FRAME_EFFECT_AND_LINEAR_STEP_SIGNATURE_CONSISTENT"
        for item in results
    ):
        return "FRAME_EFFECT_AND_LINEAR_STEP_SIGNATURE_CONSISTENT"
    return "MIXED_OR_INCONCLUSIVE"


def validate_registration(registration_path: Path, contract_path: Path) -> dict[str, Any]:
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
    expected_locked = {
        "README.md",
        "contract_v1.json",
        "run_numerics.py",
        "verify_replay.py",
        "test_jx_e2.py",
    }
    if not isinstance(locked, dict) or set(locked) != expected_locked:
        raise ValueError("registration locked-file set changed")
    root = registration_path.parent.resolve()
    if (
        contract_path.resolve() != root / "contract_v1.json"
        or registration_path.resolve() != root / "registration_v1.json"
        or Path(__file__).resolve() != root / "verify_replay.py"
    ):
        raise ValueError("registration inputs are not canonical package paths")
    if locked.get("contract_v1.json") != sha256_file(contract_path):
        raise RuntimeError("registered contract hash mismatch")
    if locked.get("run_numerics.py") != EXPECTED_RUNNER_SHA256:
        raise RuntimeError("registered runner hash mismatch")
    if locked.get("verify_replay.py") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("registered verifier hash mismatch")
    for relative, expected in locked.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"registered file mismatch: {relative}")
    if sha256_file(contract_path) != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("verifier contract binding mismatch")
    return registration


def validate_actual_runtime() -> None:
    rebound = get_rebound()

    if (
        ".".join(map(str, sys.version_info[:3])) != "3.12.13"
        or rebound.__version__ != "4.4.11"
        or rebound.__build__ != "Nov 13 2025 14:44:51"
    ):
        raise RuntimeError("verifier runtime mismatch")
    binary_path = Path(rebound.clibrebound._name).resolve()
    if not binary_path.is_file() or sha256_file(binary_path) != REBOUND_BINARY_SHA256:
        raise RuntimeError("verifier REBOUND binary mismatch")
    if rebound_python_source_sha256(rebound) != REBOUND_PYTHON_SOURCE_SHA256:
        raise RuntimeError("verifier REBOUND Python source mismatch")


def expected_runtime() -> dict[str, Any]:
    return {
        "python_version": "3.12.13",
        "rebound_version": "4.4.11",
        "rebound_build": "Nov 13 2025 14:44:51",
        "rebound_binary_sha256": REBOUND_BINARY_SHA256,
        "rebound_python_source_sha256": REBOUND_PYTHON_SOURCE_SHA256,
    }


def load_bound_e1_contract(contract: dict[str, Any]) -> dict[str, Any]:
    boundary = contract["e1_immutable_boundary"]
    root = Path(__file__).resolve().parent
    bindings = {
        "contract": (boundary["contract_path"], boundary["contract_sha256"]),
        "runner": (boundary["runner_path"], boundary["runner_sha256"]),
        "verifier": (boundary["verifier_path"], boundary["verifier_sha256"]),
        "result": (boundary["result_path"], boundary["result_sha256"]),
        "audit": (
            boundary["post_failure_audit_path"],
            boundary["post_failure_audit_sha256"],
        ),
        "audit_script": (
            boundary["audit_script_path"],
            boundary["audit_script_sha256"],
        ),
    }
    paths = {}
    for name, (relative, expected) in bindings.items():
        path = (root / relative).resolve()
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"verifier E1 binding mismatch: {name}")
        paths[name] = path
    e1_contract = strict_json(paths["contract"])
    e1_result = strict_json(paths["result"])
    e1_audit = strict_json(paths["audit"])
    if (
        e1_result.get("verdict") != boundary["required_e1_verdict"]
        or e1_result.get("semantic_sha256") != boundary["result_semantic_sha256"]
        or sha256_bytes(canonical_bytes(e1_result["semantic"]))
        != boundary["result_semantic_sha256"]
        or e1_audit.get("audit_state") != boundary["required_e1_audit_state"]
        or e1_audit.get("thresholds_changed") is not False
        or e1_audit.get("execution_b_authorized_or_started") is not False
        or e1_audit.get("new_dynamics_executed") is not False
    ):
        raise RuntimeError("verifier E1 invalid/frozen boundary changed")
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
    return e1_contract


def expected_initial_simulation(
    contract: dict[str, Any],
    e1_contract: dict[str, Any],
    configuration_id: str,
    frame_id: str,
    rebound: Any,
) -> Any:
    configurations = {item["id"]: item for item in contract["configuration_set"]}
    configuration = configurations[configuration_id]
    benchmark = contract["analytic_benchmark"]
    simulation = rebound.Simulation()
    simulation.G = float(benchmark["G_AU3_Msun_yr2"])
    simulation.add(m=float(benchmark["sun_mass_Msun"]), hash="Sun")
    for body in benchmark["giants"]:
        simulation.add(
            primary=simulation.particles[0],
            m=float(body["mass_Msun"]),
            a=float(body["a_AU"]),
            e=0.0,
            inc=0.0,
            Omega=0.0,
            omega=0.0,
            M=math.radians(float(body["initial_longitude_deg"])),
            hash=body["name"],
        )
    if configuration["model_id"] is not None:
        models = {item["id"]: item for item in e1_contract["model_grid"]}
        angles = {item["id"]: item for item in e1_contract["angle_grid"]}
        model = models[configuration["model_id"]]
        angle = angles[configuration["angle_id"]]
        omega_deg = (float(angle["varpi_deg"]) - float(angle["Omega_deg"])) % 360.0
        simulation.add(
            primary=simulation.particles[0],
            m=float(model["mass_Mearth"])
            * float(benchmark["earth_to_sun_mass_ratio"]),
            a=float(model["a_AU"]),
            e=float(model["e"]),
            inc=math.radians(float(model["i_deg"])),
            Omega=math.radians(float(angle["Omega_deg"])),
            omega=math.radians(omega_deg),
            M=math.radians(float(angle["M_deg"])),
            hash=f"P9_{model['id']}_{angle['id']}",
        )
    simulation.N_active = simulation.N
    simulation.testparticle_type = 0
    if frame_id == "FCM_ACTIVE_BARYCENTRIC":
        particles = [simulation.particles[index] for index in range(simulation.N_active)]
        total_mass = math.fsum(float(particle.m) for particle in particles)
        r_com = tuple(
            math.fsum(float(p.m) * float(getattr(p, field)) for p in particles)
            / total_mass
            for field in ("x", "y", "z")
        )
        v_com = tuple(
            math.fsum(float(p.m) * float(getattr(p, field)) for p in particles)
            / total_mass
            for field in ("vx", "vy", "vz")
        )
        for particle in particles:
            particle.x -= r_com[0]
            particle.y -= r_com[1]
            particle.z -= r_com[2]
            particle.vx -= v_com[0]
            particle.vy -= v_com[1]
            particle.vz -= v_com[2]
    elif frame_id != "F0_E1_UNSHIFTED":
        raise ValueError("unknown frame in verifier initial-state reconstruction")
    return simulation


def particle_identity(simulation: Any) -> tuple[tuple[int, float, float], ...]:
    return tuple(
        (
            int(simulation.particles[index].hash.value),
            float(simulation.particles[index].m),
            float(simulation.particles[index].r),
        )
        for index in range(simulation.N)
    )


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


def independent_invariant_snapshot(simulation: Any) -> dict[str, Any]:
    particles = [simulation.particles[index] for index in range(simulation.N_active)]
    total_mass = math.fsum(float(particle.m) for particle in particles)
    positions = [
        (float(p.x), float(p.y), float(p.z)) for p in particles
    ]
    velocities = [
        (float(p.vx), float(p.vy), float(p.vz)) for p in particles
    ]
    masses = [float(p.m) for p in particles]
    momentum = tuple(
        math.fsum(
            masses[index] * velocities[index][component]
            for index in range(len(particles))
        )
        for component in range(3)
    )
    r_com = tuple(
        math.fsum(
            masses[index] * positions[index][component]
            for index in range(len(particles))
        )
        / total_mass
        for component in range(3)
    )
    v_com = tuple(component / total_mass for component in momentum)
    origin_terms = [cross(positions[index], velocities[index]) for index in range(len(particles))]
    origin_angular = tuple(
        math.fsum(
            masses[index] * origin_terms[index][component]
            for index in range(len(particles))
        )
        for component in range(3)
    )
    relative_positions = [vector_subtract(item, r_com) for item in positions]
    relative_velocities = [vector_subtract(item, v_com) for item in velocities]
    com_terms = [
        cross(relative_positions[index], relative_velocities[index])
        for index in range(len(particles))
    ]
    com_angular = tuple(
        math.fsum(
            masses[index] * com_terms[index][component]
            for index in range(len(particles))
        )
        for component in range(3)
    )
    kinetic_internal = 0.5 * math.fsum(
        masses[index] * vector_norm(relative_velocities[index]) ** 2
        for index in range(len(particles))
    )
    potential = math.fsum(
        -float(simulation.G)
        * masses[left]
        * masses[right]
        / vector_norm(vector_subtract(positions[left], positions[right]))
        for left in range(len(particles))
        for right in range(left + 1, len(particles))
    )
    intrinsic_energy = kinetic_internal + potential
    linear_scale = math.fsum(
        masses[index] * vector_norm(relative_velocities[index])
        for index in range(len(particles))
    )
    angular_scale = math.fsum(
        masses[index] * vector_norm(com_terms[index]) for index in range(len(particles))
    )
    energy_scale = kinetic_internal + abs(potential)
    decomposition = vector_subtract(
        origin_angular,
        tuple(
            com_angular[index] + cross(r_com, momentum)[index]
            for index in range(3)
        ),
    )
    return {
        "total_mass": total_mass,
        "momentum": list(momentum),
        "r_com": list(r_com),
        "v_com": list(v_com),
        "origin_angular": list(origin_angular),
        "com_angular": list(com_angular),
        "intrinsic_energy": intrinsic_energy,
        "engine_energy": float(simulation.energy()),
        "linear_scale": linear_scale,
        "angular_scale": angular_scale,
        "energy_scale": energy_scale,
        "decomposition_residual": vector_norm(decomposition),
    }


def independent_invariant_endpoint(
    initial: dict[str, Any], current: dict[str, Any], time_year: float
) -> dict[str, float]:
    initial_momentum = tuple(initial["momentum"])
    current_momentum = tuple(current["momentum"])
    initial_origin = tuple(initial["origin_angular"])
    current_origin = tuple(current["origin_angular"])
    initial_com = tuple(initial["com_angular"])
    current_com = tuple(current["com_angular"])
    p_delta = vector_norm(vector_subtract(current_momentum, initial_momentum))
    origin_delta = vector_norm(vector_subtract(current_origin, initial_origin))
    com_delta = vector_norm(vector_subtract(current_com, initial_com))
    expected_r = tuple(
        initial["r_com"][index] + initial["v_com"][index] * time_year
        for index in range(3)
    )
    return {
        "relative_engine_energy_drift": abs(
            current["engine_energy"] - initial["engine_energy"]
        )
        / max(abs(initial["engine_energy"]), sys.float_info.min),
        "scale_normalized_intrinsic_energy_residual": abs(
            current["intrinsic_energy"] - initial["intrinsic_energy"]
        )
        / initial["energy_scale"],
        "scale_normalized_linear_momentum_residual": p_delta
        / initial["linear_scale"],
        "scale_normalized_origin_angular_momentum_residual": origin_delta
        / initial["angular_scale"],
        "scale_normalized_com_angular_momentum_residual": com_delta
        / initial["angular_scale"],
        "relative_initial_linear_momentum_residual": p_delta
        / max(vector_norm(initial_momentum), sys.float_info.min),
        "relative_initial_origin_angular_momentum_residual": origin_delta
        / max(vector_norm(initial_origin), sys.float_info.min),
        "com_ballistic_position_residual_AU": vector_norm(
            vector_subtract(tuple(current["r_com"]), expected_r)
        ),
        "angular_decomposition_residual_over_scale": current[
            "decomposition_residual"
        ]
        / initial["angular_scale"],
    }


def independent_state_discrepancy(
    left: Any,
    right: Any,
    names: list[str],
    scales: dict[str, float],
    gravitational_constant: float,
    sun_mass: float,
) -> dict[str, float]:
    left_sun = left.particles["Sun"]
    right_sun = right.particles["Sun"]
    maximum_position = 0.0
    maximum_velocity = 0.0
    maximum_dimensionless = 0.0
    for name in names:
        if name == "Sun":
            continue
        first = left.particles[name]
        second = right.particles[name]
        first_position = (
            float(first.x - left_sun.x),
            float(first.y - left_sun.y),
            float(first.z - left_sun.z),
        )
        second_position = (
            float(second.x - right_sun.x),
            float(second.y - right_sun.y),
            float(second.z - right_sun.z),
        )
        first_velocity = (
            float(first.vx - left_sun.vx),
            float(first.vy - left_sun.vy),
            float(first.vz - left_sun.vz),
        )
        second_velocity = (
            float(second.vx - right_sun.vx),
            float(second.vy - right_sun.vy),
            float(second.vz - right_sun.vz),
        )
        position = vector_norm(vector_subtract(first_position, second_position))
        velocity = vector_norm(vector_subtract(first_velocity, second_velocity))
        a_scale = scales[name]
        v_scale = math.sqrt(gravitational_constant * sun_mass / a_scale)
        maximum_position = max(maximum_position, position)
        maximum_velocity = max(maximum_velocity, velocity)
        maximum_dimensionless = max(
            maximum_dimensionless, position / a_scale, velocity / v_scale
        )
    return {
        "maximum_dimensionless_state_discrepancy": maximum_dimensionless,
        "maximum_position_separation_AU": maximum_position,
        "maximum_velocity_separation_AU_per_year": maximum_velocity,
    }


def expected_frame_identity(
    unshifted: Any, barycentric: Any, epsilon_factor: float
) -> dict[str, Any]:
    maximum_ratio = 0.0
    within = True
    left_sun = unshifted.particles[0]
    right_sun = barycentric.particles[0]
    for index in range(unshifted.N_active):
        left = unshifted.particles[index]
        right = barycentric.particles[index]
        for position_or_velocity, sun_field in (
            ("x", "x"),
            ("y", "y"),
            ("z", "z"),
            ("vx", "vx"),
            ("vy", "vy"),
            ("vz", "vz"),
        ):
            first = float(getattr(left, position_or_velocity)) - float(
                getattr(left_sun, sun_field)
            )
            second = float(getattr(right, position_or_velocity)) - float(
                getattr(right_sun, sun_field)
            )
            scale = max(1.0, abs(first), abs(second))
            ratio = abs(first - second) / (sys.float_info.epsilon * scale)
            maximum_ratio = max(maximum_ratio, ratio)
            within = within and ratio <= epsilon_factor
    return {
        "maximum_binary64_epsilon_units": maximum_ratio,
        "within_locked_bound": within,
    }


def expected_pair_keys(contract: dict[str, Any]) -> set[tuple[str, str, str]]:
    frames = [item["id"] for item in contract["frames"]]
    regimes = [item["id"] for item in contract["numerical_regimes"]]
    keys = {
        ("FRAME", f"F0_E1_UNSHIFTED/{regime}", f"FCM_ACTIVE_BARYCENTRIC/{regime}")
        for regime in regimes
    }
    for frame in frames:
        keys.update(
            ("MERCURIUS_REFINEMENT", f"{frame}/{left}", f"{frame}/{right}")
            for left, right in contract["comparison_policy"]["mercurius_refinement_pairs"]
        )
        keys.update(
            ("IAS15_REFERENCE", f"{frame}/{left}", f"{frame}/{right}")
            for left, right in contract["comparison_policy"]["ias15_reference_pairs"]
        )
        keys.update(
            ("MERCURIUS_TO_IAS15", f"{frame}/{left}", f"{frame}/{right}")
            for left, right in contract["comparison_policy"]["mercurius_to_tight_reference"]
        )
    if len(keys) != 20:
        raise RuntimeError("expected comparison matrix changed")
    return keys


INVARIANT_MAXIMUM_KEYS = {
    "relative_engine_energy_drift",
    "scale_normalized_intrinsic_energy_residual",
    "scale_normalized_linear_momentum_residual",
    "scale_normalized_origin_angular_momentum_residual",
    "scale_normalized_com_angular_momentum_residual",
    "relative_initial_linear_momentum_residual",
    "relative_initial_origin_angular_momentum_residual",
    "com_ballistic_position_residual_AU",
    "angular_decomposition_residual_over_scale",
}
PAIR_METRIC_KEYS = {
    "maximum_dimensionless_state_discrepancy",
    "maximum_position_separation_AU",
    "maximum_velocity_separation_AU_per_year",
}


def finite_nonnegative(value: Any) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def validate_invariant_metrics(
    contract: dict[str, Any], arm: dict[str, Any]
) -> None:
    metrics = arm.get("invariant_metrics")
    if not isinstance(metrics, dict) or set(metrics) != {"initial", "maximum", "endpoint"}:
        raise ValueError("arm invariant-metric shape mismatch")
    initial = metrics["initial"]
    if set(initial) != {
        "total_mass",
        "momentum",
        "r_com",
        "v_com",
        "origin_angular",
        "com_angular",
        "intrinsic_energy",
        "engine_energy",
        "linear_scale",
        "angular_scale",
        "energy_scale",
        "decomposition_residual",
    }:
        raise ValueError("initial invariant shape mismatch")
    vector_keys = {"momentum", "r_com", "v_com", "origin_angular", "com_angular"}
    for key, value in initial.items():
        values = value if key in vector_keys else (value,)
        if key in vector_keys and (not isinstance(value, list) or len(value) != 3):
            raise ValueError("initial invariant vector shape mismatch")
        if any(type(item) not in (int, float) or not math.isfinite(float(item)) for item in values):
            raise ValueError("non-finite initial invariant")
    if (
        float(initial["total_mass"]) <= 0.0
        or float(initial["linear_scale"]) <= 0.0
        or float(initial["angular_scale"]) <= 0.0
        or float(initial["energy_scale"]) <= 0.0
        or float(initial["decomposition_residual"]) < 0.0
    ):
        raise ValueError("invalid invariant scale/residual")
    maximum = metrics["maximum"]
    endpoint = metrics["endpoint"]
    if set(maximum) != INVARIANT_MAXIMUM_KEYS or set(endpoint) != INVARIANT_MAXIMUM_KEYS:
        raise ValueError("invariant maximum/endpoint shape mismatch")
    for key in INVARIANT_MAXIMUM_KEYS:
        if (
            not finite_nonnegative(maximum[key])
            or not finite_nonnegative(endpoint[key])
            or float(maximum[key]) < float(endpoint[key])
        ):
            raise ValueError("invalid invariant maximum/endpoint value")
    legacy = arm["frame_id"] == "F0_E1_UNSHIFTED"
    expected_flags = {
        "legacy_E1_reference_applicable": legacy,
        "relative_energy_within_legacy_E1_reference": (
            float(maximum["relative_engine_energy_drift"])
            <= float(contract["inherited_reference_flags"]["maximum_relative_energy_drift"])
            if legacy
            else None
        ),
        "compensated_origin_angular_within_legacy_numeric_value_illustration": (
            float(maximum["relative_initial_origin_angular_momentum_residual"])
            <= float(
                contract["inherited_reference_flags"][
                    "maximum_relative_origin_angular_momentum_drift"
                ]
            )
            if legacy
            else None
        ),
        "origin_angular_is_not_exact_E1_evaluator": True,
        "Pstar_and_Lstar_metrics_are_not_legacy_E1_thresholds": True,
    }
    if arm.get("inherited_reference_flags") != expected_flags:
        raise ValueError("arm inherited-reference flags do not recompute")
    for key in (
        "initial_physical_state_sha256",
        "endpoint_decoded_state_sha256",
        "sampled_state_stream_sha256",
    ):
        value = arm.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("arm state digest format mismatch")
    minimum_dt = arm.get("minimum_dt_last_done_years")
    maximum_dt = arm.get("maximum_dt_last_done_years")
    if (
        not finite_nonnegative(minimum_dt)
        or not finite_nonnegative(maximum_dt)
        or float(minimum_dt) <= 0.0
        or float(minimum_dt) > float(maximum_dt)
    ):
        raise ValueError("arm sampled timestep bounds invalid")


def validate_comparison_record(contract: dict[str, Any], item: dict[str, Any]) -> None:
    if set(item) != {"kind", "left", "right", "maximum", "endpoint"}:
        raise ValueError("comparison record shape mismatch")
    maximum = item["maximum"]
    endpoint = item["endpoint"]
    if set(maximum) != PAIR_METRIC_KEYS or set(endpoint) != {
        "time_year",
        *PAIR_METRIC_KEYS,
    }:
        raise ValueError("comparison metric shape mismatch")
    if endpoint["time_year"] != float(contract["dynamics"]["duration_years"]):
        raise ValueError("comparison endpoint time mismatch")
    for key in PAIR_METRIC_KEYS:
        if (
            not finite_nonnegative(maximum[key])
            or not finite_nonnegative(endpoint[key])
            or float(maximum[key]) < float(endpoint[key])
        ):
            raise ValueError("comparison maximum/endpoint value invalid")


def validate_checkpoint_configuration(
    contract: dict[str, Any],
    bundle: dict[str, Any],
    arm: dict[str, Any],
    checkpoint: dict[str, Any],
    simulation: Any,
    expected_identity: tuple[tuple[int, float, float], ...],
) -> None:
    regimes = {item["id"]: item for item in contract["numerical_regimes"]}
    regime = regimes[arm["regime_id"]]
    for index, particle in enumerate(simulation.particles):
        for field in ("m", "r", "x", "y", "z", "vx", "vy", "vz"):
            if not math.isfinite(float(getattr(particle, field))):
                raise ValueError(f"non-finite checkpoint particle field: {index}/{field}")
    if str(simulation.integrator) != regime["integrator"]:
        raise ValueError("checkpoint integrator mismatch")
    if float(simulation.t) != float(checkpoint["time_year"]):
        raise ValueError("checkpoint time mismatch")
    expected_n = 5 if bundle["configuration_id"] == "M0" else 6
    if simulation.N != expected_n or simulation.N_active != expected_n:
        raise ValueError("checkpoint active layout mismatch")
    if (
        float(simulation.G) != float(contract["analytic_benchmark"]["G_AU3_Msun_yr2"])
        or int(simulation.testparticle_type) != 0
        or particle_identity(simulation) != expected_identity
    ):
        raise ValueError("checkpoint physical identity/configuration mismatch")
    if str(simulation.collision) != "none" or str(simulation.boundary) != "none":
        raise ValueError("checkpoint collision/boundary mismatch")
    if regime["integrator"] == "mercurius":
        if (
            float(simulation.dt) != float(regime["dt_years"])
            or not math.isfinite(float(simulation.dt_last_done))
            or float(simulation.dt_last_done) != float(regime["dt_years"])
            or float(simulation.ri_mercurius.r_crit_hill) != float(regime["r_crit_hill"])
            or int(simulation.ri_mercurius.safe_mode) != int(regime["safe_mode"])
            or str(simulation.gravity) != "mercurius"
        ):
            raise ValueError("checkpoint MERCURIUS setting mismatch")
    else:
        if (
            float(simulation.ri_ias15.epsilon) != float(regime["epsilon"])
            or float(simulation.ri_ias15.min_dt) != float(regime["min_dt_years"])
            or str(simulation.ri_ias15.adaptive_mode) != regime["adaptive_mode"]
            or str(simulation.gravity) != "basic"
            or not math.isfinite(float(simulation.dt))
            or not math.isfinite(float(simulation.dt_last_done))
            or float(simulation.dt) <= 0.0
            or float(simulation.dt_last_done) <= 0.0
            or int(simulation.ri_ias15._iterations_max_exceeded) != 0
        ):
            raise ValueError("checkpoint IAS15 setting mismatch")


def verify_output(
    contract: dict[str, Any], registration_sha256: str, output_dir: Path
) -> dict[str, Any]:
    rebound = get_rebound()

    output_dir = output_dir.resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError("execution output directory is missing")
    if (output_dir / "failure_receipt.json").exists():
        raise RuntimeError("execution directory has a permanent failure receipt")
    manifest = strict_json(output_dir / "run_manifest.json")
    result = strict_json(output_dir / "result_v1.json")
    if set(manifest) != {
        "schema",
        "experiment_id",
        "contract_sha256",
        "registration_sha256",
        "runner_sha256",
        "execution_label",
        "execution_instance_id",
        "runtime",
    }:
        raise RuntimeError("run-manifest schema shape mismatch")
    if (
        manifest["schema"] != "jx-e2-numerics-run-manifest/v1"
        or manifest["experiment_id"] != contract["experiment_id"]
        or manifest["contract_sha256"] != EXPECTED_CONTRACT_SHA256
        or manifest["registration_sha256"] != registration_sha256
        or manifest["runner_sha256"] != EXPECTED_RUNNER_SHA256
        or manifest["execution_label"] not in contract["result_policy"]["clean_execution_labels"]
        or not isinstance(manifest["execution_instance_id"], str)
        or len(manifest["execution_instance_id"]) != 32
        or any(character not in "0123456789abcdef" for character in manifest["execution_instance_id"])
    ):
        raise RuntimeError("run-manifest identity/binding mismatch")
    if set(result) != {
        "schema",
        "experiment_id",
        "verdict",
        "claim_ceiling",
        "semantic",
        "semantic_sha256",
        "provenance",
        "nonclaim",
    }:
        raise RuntimeError("result schema shape mismatch")
    if (
        result["schema"] != "jx-e2-numerics-result/v1"
        or result["experiment_id"] != contract["experiment_id"]
    ):
        raise RuntimeError("result identity mismatch")
    if result.get("verdict") != contract["result_policy"]["first_execution_verdict"]:
        raise RuntimeError("execution is not provisionally valid")
    if result.get("semantic_sha256") != sha256_bytes(canonical_bytes(result["semantic"])):
        raise RuntimeError("result semantic hash mismatch")
    if result.get("claim_ceiling") != contract["claim_ceiling"] or result.get("nonclaim") != contract["mandatory_nonclaim"]:
        raise RuntimeError("result claim boundary mismatch")
    if set(result["semantic"]["execution_integrity"]) != {
        "bundle_count_exact",
        "arm_count_exact",
        "comparison_count_exact",
        "e1_context_count_exact",
        "all_bundle_integrity_checks_true",
        "all_ias15_iteration_limit_events_zero",
        "inherited_reference_flags_do_not_control_validity",
    } or not all(value is True for value in result["semantic"]["execution_integrity"].values()):
        raise RuntimeError("result integrity map is not exact/all true")
    if result["semantic"].get("contract_sha256") != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("result contract binding mismatch")
    if result["semantic"].get("registration_sha256") != registration_sha256:
        raise RuntimeError("result registration binding mismatch")
    if (
        result["semantic"].get("schema") != "jx-e2-numerics-result/v1"
        or result["semantic"].get("experiment_id") != contract["experiment_id"]
        or result["semantic"].get("claim_ceiling") != contract["claim_ceiling"]
        or result["semantic"].get("mandatory_nonclaim") != contract["mandatory_nonclaim"]
    ):
        raise RuntimeError("result semantic identity/claim boundary mismatch")
    if set(result["semantic"]) != {
        "schema",
        "experiment_id",
        "contract_sha256",
        "registration_sha256",
        "claim_ceiling",
        "execution_integrity",
        "bundle_semantic_hashes",
        "configuration_classifications",
        "overall_numerical_classification",
        "mandatory_nonclaim",
    }:
        raise RuntimeError("result semantic shape mismatch")
    if set(result["provenance"]) != {
        "execution_label",
        "execution_instance_id",
        "elapsed_seconds",
        "peak_rss_bytes",
        "output_bytes_before_result",
        "output_directory",
    }:
        raise RuntimeError("result provenance shape mismatch")
    if Path(result["provenance"]["output_directory"]).resolve() != output_dir:
        raise RuntimeError("result output-directory provenance mismatch")
    if manifest["execution_label"] != result["provenance"]["execution_label"]:
        raise RuntimeError("manifest/result label mismatch")
    if manifest["execution_instance_id"] != result["provenance"]["execution_instance_id"]:
        raise RuntimeError("manifest/result instance mismatch")
    if manifest["runtime"] != expected_runtime():
        raise RuntimeError("manifest runtime mismatch")
    caps = contract["resource_caps"]
    if (
        not isinstance(result["provenance"]["elapsed_seconds"], (int, float))
        or not math.isfinite(float(result["provenance"]["elapsed_seconds"]))
        or not 0.0 <= float(result["provenance"]["elapsed_seconds"])
        <= float(caps["max_wall_seconds_total_per_clean_execution"])
        or not isinstance(result["provenance"]["peak_rss_bytes"], int)
        or not 0 <= result["provenance"]["peak_rss_bytes"] <= int(caps["max_peak_rss_bytes"])
        or not isinstance(result["provenance"]["output_bytes_before_result"], int)
        or not 0 <= result["provenance"]["output_bytes_before_result"]
        <= int(caps["max_output_bytes"])
    ):
        raise RuntimeError("result resource provenance violates locked caps")
    e1_contract = load_bound_e1_contract(contract)
    expected_configurations = [item["id"] for item in contract["configuration_set"]]
    if set(result["semantic"]["bundle_semantic_hashes"]) != set(expected_configurations):
        raise RuntimeError("result bundle set mismatch")
    stored_classifications = result["semantic"].get("configuration_classifications")
    if (
        not isinstance(stored_classifications, list)
        or len(stored_classifications) != len(expected_configurations)
        or [item.get("configuration_id") for item in stored_classifications]
        != expected_configurations
    ):
        raise RuntimeError("result classification set/order mismatch")
    expected_files = {
        "run_manifest.json",
        "result_v1.json",
        *(f"bundles/{configuration_id}.json" for configuration_id in expected_configurations),
    }
    seen_checkpoint_paths: set[str] = set()
    bundles = []
    for configuration_id in expected_configurations:
        bundle = strict_json(output_dir / "bundles" / f"{configuration_id}.json")
        if set(bundle) != {"schema", "semantic", "semantic_sha256", "provenance"}:
            raise ValueError("bundle top-level shape mismatch")
        if bundle["schema"] != "jx-e2-numerics-bundle/v1":
            raise ValueError("bundle schema mismatch")
        if set(bundle["semantic"]) != {
            "schema",
            "contract_sha256",
            "configuration_id",
            "frame_identity",
            "arms",
            "comparisons",
            "e1_context_comparisons",
            "integrity",
        }:
            raise ValueError("bundle semantic shape mismatch")
        if (
            bundle["semantic"].get("schema") != "jx-e2-numerics-bundle/v1"
            or bundle["semantic"].get("contract_sha256") != EXPECTED_CONTRACT_SHA256
            or bundle["semantic"].get("configuration_id") != configuration_id
        ):
            raise ValueError("bundle identity mismatch")
        if bundle["semantic_sha256"] != sha256_bytes(canonical_bytes(bundle["semantic"])):
            raise ValueError("bundle semantic hash mismatch")
        if result["semantic"]["bundle_semantic_hashes"][configuration_id] != bundle["semantic_sha256"]:
            raise ValueError("result/bundle hash mismatch")
        if set(bundle["provenance"]) != {
            "elapsed_seconds",
            "peak_rss_bytes",
            "checkpoint_containers",
        }:
            raise ValueError("bundle provenance shape mismatch")
        if (
            not isinstance(bundle["provenance"]["elapsed_seconds"], (int, float))
            or not math.isfinite(float(bundle["provenance"]["elapsed_seconds"]))
            or not 0.0 <= float(bundle["provenance"]["elapsed_seconds"])
            <= float(caps["max_wall_seconds_per_configuration_bundle"])
            or not isinstance(bundle["provenance"]["peak_rss_bytes"], int)
            or not 0 <= bundle["provenance"]["peak_rss_bytes"]
            <= int(caps["max_peak_rss_bytes"])
        ):
            raise ValueError("bundle resource provenance violates locked caps")
        arms = arm_lookup(bundle["semantic"])
        if len(arms) != 12 or len(bundle["semantic"]["comparisons"]) != 20:
            raise ValueError("bundle matrix cardinality mismatch")
        if set(bundle["semantic"]["integrity"]) != {
            "arm_count_exact",
            "comparison_count_exact",
            "e1_context_count_exact",
            "all_frame_identity_bounds_met",
            "all_direct_chained_and_checkpoint_checks_exact",
        } or not all(value is True for value in bundle["semantic"]["integrity"].values()):
            raise ValueError("bundle integrity map is not exact/all true")
        if set(comparison_lookup(bundle["semantic"])) != expected_pair_keys(contract):
            raise ValueError("bundle comparison key set mismatch")
        for comparison in bundle["semantic"]["comparisons"]:
            validate_comparison_record(contract, comparison)
        expected_arm_keys = {
            f"{frame['id']}/{regime['id']}"
            for frame in contract["frames"]
            for regime in contract["numerical_regimes"]
        }
        if set(arms) != expected_arm_keys:
            raise ValueError("bundle arm key set mismatch")
        expected_initial_by_frame = {
            frame["id"]: expected_initial_simulation(
                contract, e1_contract, configuration_id, frame["id"], rebound
            )
            for frame in contract["frames"]
        }
        recomputed_frame_identity = expected_frame_identity(
            expected_initial_by_frame["F0_E1_UNSHIFTED"],
            expected_initial_by_frame["FCM_ACTIVE_BARYCENTRIC"],
            float(
                contract["frame_identity_rule"][
                    "absolute_relative_bound_in_binary64_eps"
                ]
            ),
        )
        if bundle["semantic"]["frame_identity"] != recomputed_frame_identity:
            raise ValueError("bundle frame-identity diagnostic does not recompute")
        for arm_key, arm in arms.items():
            expected_arm_key = f"{arm.get('frame_id')}/{arm.get('regime_id')}"
            regimes = {item["id"]: item for item in contract["numerical_regimes"]}
            regime = regimes.get(arm.get("regime_id"))
            if (
                set(arm) != {
                    "schema",
                    "arm_key",
                    "frame_id",
                    "regime_id",
                    "integrator",
                    "sample_count",
                    "final_time_year",
                    "initial_physical_state_sha256",
                    "endpoint_decoded_state_sha256",
                    "sampled_state_stream_sha256",
                    "checkpoints",
                    "minimum_dt_last_done_years",
                    "maximum_dt_last_done_years",
                    "ias15_iteration_limit_events",
                    "invariant_metrics",
                    "inherited_reference_flags",
                    "integrity",
                }
                or arm.get("schema") != "jx-e2-numerics-arm/v1"
                or arm_key != expected_arm_key
                or regime is None
                or arm.get("integrator") != regime["integrator"]
                or arm["sample_count"]
                != int(contract["dynamics"]["expected_samples_per_arm_including_t0"])
                or arm["final_time_year"] != float(contract["dynamics"]["duration_years"])
                or arm.get("initial_physical_state_sha256")
                != physical_state_digest(expected_initial_by_frame[arm["frame_id"]])
                or type(arm["ias15_iteration_limit_events"]) is not int
                or arm["ias15_iteration_limit_events"] != 0
                or set(arm["integrity"]) != {
                    "direct_and_chained_exact_every_sample",
                    "checkpoint_decoded_states_exact",
                    "states_and_invariants_finite",
                    "particle_identity_unchanged",
                    "integrator_settings_readback_exact",
                }
                or not all(value is True for value in arm["integrity"].values())
            ):
                raise ValueError("arm integrity/count fields invalid")
            validate_invariant_metrics(contract, arm)
            if regime["integrator"] == "mercurius" and (
                arm["minimum_dt_last_done_years"] != float(regime["dt_years"])
                or arm["maximum_dt_last_done_years"] != float(regime["dt_years"])
            ):
                raise ValueError("MERCURIUS sampled timestep bounds mismatch")
        context_records = bundle["semantic"].get("e1_context_comparisons")
        expected_context_keys = {
            (regime_id, checkpoint_index)
            for regime_id in ("MERCURIUS_0125", "MERCURIUS_00625")
            for checkpoint_index in range(
                1, int(contract["dynamics"]["expected_checkpoints_per_arm"]) + 1
            )
        }
        if not isinstance(context_records, list) or len(context_records) != len(
            expected_context_keys
        ):
            raise ValueError("E1-context record count mismatch")
        actual_context_keys = set()
        for record in context_records:
            if set(record) != {
                "regime_id",
                "checkpoint_index",
                "time_year",
                "e2_active_projection_sha256",
                "e1_active_projection_sha256",
                "exact",
            }:
                raise ValueError("E1-context record shape mismatch")
            key = (record["regime_id"], record["checkpoint_index"])
            if key in actual_context_keys:
                raise ValueError("duplicate E1-context record")
            actual_context_keys.add(key)
            expected_time = (
                float(contract["dynamics"]["checkpoint_cadence_years"])
                * record["checkpoint_index"]
            )
            hashes = (
                record["e2_active_projection_sha256"],
                record["e1_active_projection_sha256"],
            )
            if (
                record["time_year"] != expected_time
                or any(
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                    for value in hashes
                )
                or record["exact"] is not (hashes[0] == hashes[1])
            ):
                raise ValueError("E1-context record value mismatch")
        if actual_context_keys != expected_context_keys:
            raise ValueError("E1-context key set mismatch")
        provenance = bundle["provenance"]["checkpoint_containers"]
        if set(provenance) != set(arms):
            raise ValueError("checkpoint provenance arm set mismatch")
        final_simulations: dict[str, Any] = {}
        for arm_key, arm in arms.items():
            containers = provenance[arm_key]
            semantic_checkpoints = arm["checkpoints"]
            expected_checkpoint_count = int(
                contract["dynamics"]["expected_checkpoints_per_arm"]
            )
            if (
                len(containers) != expected_checkpoint_count
                or len(semantic_checkpoints) != expected_checkpoint_count
            ):
                raise ValueError("checkpoint count mismatch")
            for index, (container, semantic_checkpoint) in enumerate(
                zip(containers, semantic_checkpoints, strict=True), start=1
            ):
                if set(semantic_checkpoint) != {
                    "checkpoint_index",
                    "time_year",
                    "decoded_state_sha256",
                } or set(container) != {
                    "checkpoint_index",
                    "time_year",
                    "decoded_state_sha256",
                    "relative_path",
                    "container_bytes",
                    "container_sha256_provenance_only",
                }:
                    raise ValueError("checkpoint record shape mismatch")
                expected_time = (
                    float(contract["dynamics"]["checkpoint_cadence_years"]) * index
                )
                expected_relative = (
                    f"checkpoints/{configuration_id}/{arm_key.replace('/', '__')}"
                    f"/checkpoint_{index:02d}.bin"
                )
                if (
                    container["checkpoint_index"] != index
                    or semantic_checkpoint["checkpoint_index"] != index
                    or container["time_year"] != expected_time
                    or semantic_checkpoint["time_year"] != expected_time
                    or container["relative_path"] != expected_relative
                    or expected_relative in seen_checkpoint_paths
                ):
                    raise ValueError("checkpoint identity/order/time/path mismatch")
                seen_checkpoint_paths.add(expected_relative)
                expected_files.add(expected_relative)
                if {k: container[k] for k in semantic_checkpoint} != semantic_checkpoint:
                    raise ValueError("checkpoint semantic/provenance mismatch")
                for digest_value in (
                    semantic_checkpoint["decoded_state_sha256"],
                    container["container_sha256_provenance_only"],
                ):
                    if (
                        not isinstance(digest_value, str)
                        or len(digest_value) != 64
                        or any(
                            character not in "0123456789abcdef"
                            for character in digest_value
                        )
                    ):
                        raise ValueError("checkpoint digest format mismatch")
                if (
                    not isinstance(container["container_bytes"], int)
                    or not 0 < container["container_bytes"]
                    <= int(caps["max_checkpoint_bytes"])
                ):
                    raise ValueError("checkpoint byte count violates cap")
                path = (output_dir / container["relative_path"]).resolve()
                if not path.is_relative_to(output_dir) or not path.is_file():
                    raise ValueError("checkpoint path invalid")
                if path.stat().st_size != container["container_bytes"]:
                    raise ValueError("checkpoint byte count mismatch")
                if sha256_file(path) != container["container_sha256_provenance_only"]:
                    raise ValueError("checkpoint raw hash mismatch")
                loaded = rebound.Simulation(str(path))
                validate_checkpoint_configuration(
                    contract,
                    bundle["semantic"],
                    arm,
                    semantic_checkpoint,
                    loaded,
                    particle_identity(expected_initial_by_frame[arm["frame_id"]]),
                )
                if decoded_state_digest(loaded) != semantic_checkpoint["decoded_state_sha256"]:
                    raise ValueError("checkpoint decoded state mismatch")
                if (
                    index == expected_checkpoint_count
                    and semantic_checkpoint["decoded_state_sha256"]
                    != arm["endpoint_decoded_state_sha256"]
                ):
                    raise ValueError("final checkpoint/arm endpoint mismatch")
                if index == expected_checkpoint_count:
                    final_simulations[arm_key] = loaded
        if set(final_simulations) != expected_arm_keys:
            raise ValueError("final checkpoint simulation set mismatch")
        for arm_key, arm in arms.items():
            initial_snapshot = independent_invariant_snapshot(
                expected_initial_by_frame[arm["frame_id"]]
            )
            endpoint_snapshot = independent_invariant_snapshot(
                final_simulations[arm_key]
            )
            endpoint_metrics = independent_invariant_endpoint(
                initial_snapshot,
                endpoint_snapshot,
                float(contract["dynamics"]["duration_years"]),
            )
            if arm["invariant_metrics"]["initial"] != initial_snapshot:
                raise ValueError("arm initial invariant snapshot does not recompute")
            if arm["invariant_metrics"]["endpoint"] != endpoint_metrics:
                raise ValueError("arm endpoint invariant metrics do not recompute")
        names = ["Sun", *(item["name"] for item in contract["analytic_benchmark"]["giants"])]
        scales = {
            item["name"]: float(item["a_AU"])
            for item in contract["analytic_benchmark"]["giants"]
        }
        configuration = next(
            item for item in contract["configuration_set"] if item["id"] == configuration_id
        )
        if configuration["model_id"] is not None:
            models = {item["id"]: item for item in e1_contract["model_grid"]}
            p9_name = f"P9_{configuration['model_id']}_{configuration['angle_id']}"
            names.append(p9_name)
            scales[p9_name] = float(models[configuration["model_id"]]["a_AU"])
        for comparison in bundle["semantic"]["comparisons"]:
            recomputed_endpoint = {
                "time_year": float(contract["dynamics"]["duration_years"]),
                **independent_state_discrepancy(
                    final_simulations[comparison["left"]],
                    final_simulations[comparison["right"]],
                    names,
                    scales,
                    float(contract["analytic_benchmark"]["G_AU3_Msun_yr2"]),
                    float(contract["analytic_benchmark"]["sun_mass_Msun"]),
                ),
            }
            if comparison["endpoint"] != recomputed_endpoint:
                raise ValueError("comparison endpoint does not recompute")
        recomputed = recompute_configuration_classification(contract, bundle["semantic"])
        stored = next(
            item
            for item in result["semantic"]["configuration_classifications"]
            if item["configuration_id"] == configuration_id
        )
        if recomputed != stored:
            raise ValueError("configuration classification does not recompute")
        bundles.append(bundle)
    recomputed_results = [
        recompute_configuration_classification(contract, bundle["semantic"])
        for bundle in bundles
    ]
    if recompute_overall(recomputed_results) != result["semantic"]["overall_numerical_classification"]:
        raise ValueError("overall classification does not recompute")
    expected_execution_integrity = {
        "bundle_count_exact": len(bundles) == len(expected_configurations),
        "arm_count_exact": sum(
            len(bundle["semantic"]["arms"]) for bundle in bundles
        )
        == int(contract["dynamics"]["expected_arm_count"]),
        "comparison_count_exact": sum(
            len(bundle["semantic"]["comparisons"]) for bundle in bundles
        )
        == int(contract["dynamics"]["expected_pairwise_comparison_count"]),
        "e1_context_count_exact": sum(
            len(bundle["semantic"]["e1_context_comparisons"]) for bundle in bundles
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
        "inherited_reference_flags_do_not_control_validity": contract[
            "inherited_reference_flags"
        ]["affect_e2_validity"]
        is False,
    }
    if result["semantic"]["execution_integrity"] != expected_execution_integrity:
        raise ValueError("result execution-integrity map does not recompute")
    all_output_paths = list(output_dir.rglob("*"))
    if any(path.is_symlink() for path in all_output_paths):
        raise ValueError("execution output contains a symlink")
    if any(not path.is_file() and not path.is_dir() for path in all_output_paths):
        raise ValueError("execution output contains a non-regular filesystem entry")
    actual_files = {
        path.relative_to(output_dir).as_posix()
        for path in all_output_paths
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("execution output file set is not exact")
    expected_directories = set()
    for relative in expected_files:
        parent = Path(relative).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    actual_directories = {
        path.relative_to(output_dir).as_posix()
        for path in all_output_paths
        if path.is_dir()
    }
    if actual_directories != expected_directories:
        raise ValueError("execution output directory set is not exact")
    actual_output_bytes = sum(
        path.stat().st_size for path in all_output_paths if path.is_file()
    )
    if actual_output_bytes > int(caps["max_output_bytes"]):
        raise ValueError("execution output currently exceeds locked size cap")
    if (
        actual_output_bytes - (output_dir / "result_v1.json").stat().st_size
        != result["provenance"]["output_bytes_before_result"]
    ):
        raise ValueError("result pre-write output byte count does not recompute")
    output_manifest_sha256 = recursive_file_manifest_sha256(output_dir)
    return {
        "manifest": manifest,
        "result": result,
        "bundle_semantics": {
            bundle["semantic"]["configuration_id"]: bundle["semantic"] for bundle in bundles
        },
        "output_manifest_sha256": output_manifest_sha256,
        "result_file_sha256": sha256_file(output_dir / "result_v1.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--a", required=True, type=Path)
    parser.add_argument("--b", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    contract_path = arguments.contract.resolve()
    registration_path = arguments.registration.resolve()
    a_directory = arguments.a.resolve()
    b_directory = arguments.b.resolve()
    receipt_path = arguments.output.resolve()
    contract = strict_json(contract_path)
    if (
        a_directory == b_directory
        or a_directory in b_directory.parents
        or b_directory in a_directory.parents
    ):
        raise ValueError("A and B execution roots must be disjoint")
    e1_directory = (
        contract_path.parent / contract["e1_immutable_boundary"]["directory"]
    ).resolve()
    for protected in (a_directory, b_directory, e1_directory):
        if receipt_path == protected or protected in receipt_path.parents:
            raise ValueError("replay receipt may not be written inside a protected tree")
    if receipt_path.exists():
        raise FileExistsError("refusing to overwrite replay receipt")
    validate_actual_runtime()
    validate_registration(registration_path, contract_path)
    registration_sha256 = sha256_file(registration_path)
    first = verify_output(contract, registration_sha256, a_directory)
    second = verify_output(contract, registration_sha256, b_directory)
    if first["manifest"]["execution_label"] != "E2-A" or second["manifest"]["execution_label"] != "E2-B":
        raise RuntimeError("clean execution labels mismatch")
    if first["manifest"]["execution_instance_id"] == second["manifest"]["execution_instance_id"]:
        raise RuntimeError("execution instance IDs are not distinct")
    if first["result"]["semantic"] != second["result"]["semantic"]:
        verdict = contract["result_policy"]["replay_mismatch_verdict"]
    elif first["bundle_semantics"] != second["bundle_semantics"]:
        verdict = contract["result_policy"]["replay_mismatch_verdict"]
    else:
        verdict = contract["result_policy"]["replay_verdict"]
    receipt = {
        "schema": "jx-e2-numerics-replay-receipt/v1",
        "experiment_id": contract["experiment_id"],
        "verdict": verdict,
        "claim_ceiling": contract["claim_ceiling"],
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "registration_sha256": registration_sha256,
        "runner_sha256": EXPECTED_RUNNER_SHA256,
        "verifier_sha256": sha256_file(Path(__file__).resolve()),
        "execution_a_result_sha256": first["result_file_sha256"],
        "execution_b_result_sha256": second["result_file_sha256"],
        "execution_a_output_manifest_sha256": first["output_manifest_sha256"],
        "execution_b_output_manifest_sha256": second["output_manifest_sha256"],
        "semantic_sha256": first["result"]["semantic_sha256"],
        "overall_numerical_classification": first["result"]["semantic"]["overall_numerical_classification"],
        "verification_scope": "The verifier independently checks locked identities, checkpoint containers and decoded states, structural completeness, stored classification arithmetic, and exact A/B semantic replay. Raw 10-year state samples are not retained, so sampled maxima are content-bound by A/B equality but cannot be independently reconstructed from trajectories. This is not an independent scientific implementation.",
        "mandatory_nonclaim": contract["mandatory_nonclaim"],
    }
    if (
        recursive_file_manifest_sha256(a_directory)
        != first["output_manifest_sha256"]
        or recursive_file_manifest_sha256(b_directory)
        != second["output_manifest_sha256"]
        or sha256_file(a_directory / "result_v1.json") != first["result_file_sha256"]
        or sha256_file(b_directory / "result_v1.json") != second["result_file_sha256"]
    ):
        raise RuntimeError("execution outputs changed during replay verification")
    atomic_json(receipt_path, receipt)
    print(json.dumps({
        "verdict": verdict,
        "claim_ceiling": contract["claim_ceiling"],
        "classification": receipt["overall_numerical_classification"],
        "semantic_sha256": receipt["semantic_sha256"],
    }, sort_keys=True))
    return 0 if verdict == contract["result_policy"]["replay_verdict"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
