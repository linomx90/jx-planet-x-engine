#!/usr/bin/env python3
"""Standalone runner for the registered JX-XP1 synthetic response pilot."""

from __future__ import annotations

import argparse
import decimal
import hashlib
import importlib
import importlib.util
import json
import math
import os
import resource
import select
import shutil
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable


CONTRACT_SCHEMA = "jx-xp1-synthetic-response-contract/v1"
REGISTRATION_SCHEMA = "jx-xp1-local-registration/v1"
MANIFEST_SCHEMA = "jx-xp1-run-manifest/v1"
RESULT_SCHEMA = "jx-xp1-synthetic-response-result/v1"
SEMANTIC_SCHEMA = "jx-xp1-synthetic-response-semantic/v1"
FAILURE_SCHEMA = "jx-xp1-synthetic-response-failure/v1"
REDACTED_FAILURE_MESSAGE = "REDACTED_NON_SEMANTIC_FAILURE_DETAIL"

EXPECTED_EXPERIMENT_ID = "jx-xp1-public-synthetic-response-v1"
EXPECTED_CONTRACT_SHA256 = "dd4527ef2b7d61bda93395d9dec7107b57c962c88aee3f1f1032af60dd055d63"
EXPECTED_SEED_MANIFEST_SHA256 = "92de9fae8c32f322c58216c64355739917dddee2881e823541b7fbad791e1ac7"
EXPECTED_DESIGN_CORE_SHA256 = "0865266fa46b3cdf080d783f366f4988a76fb1667bf334bd79b005e9ad68380c"
EXPECTED_TRACER_ROWS_SHA256 = "b98c8c27f3301f54afff72a0b71847e1508d6ed51dc2ce566c4ca9daec7133ab"
EXPECTED_PYTHON_SHA256 = "021044895e95be79dc2f110367607e684119afbc8ce75f6f0eec94844e0acec7"
EXPECTED_REBOUND_BINARY_SHA256 = "fe7a23bcece1c3f1f869089e9e8d806bedb4727d893d2e551339adbb6665c28a"
EXPECTED_REBOUND_SOURCE_SHA256 = "2c40b16571d57049cbf4bb8329a0c58342f3dc0f0cf49d860ca77fda5a73ae3a"
EXPECTED_REBOUND_SOURCE_COUNT = 29

LOCKED_FILES = {
    "README.md",
    "contract_v1.json",
    "seed_manifest_v1.json",
    "run_exploratory.py",
    "verify_replay.py",
    "test_exploratory.py",
}
PRIMARY_ARM_IDS = ("M0", "CI01-A", "CI01-B", "CI05-A", "CI05-B", "CI09-A", "CI09-B")
AUDIT_ARM_IDS = tuple(f"AUDIT-{arm_id}" for arm_id in PRIMARY_ARM_IDS)
AUDIT_PRIMARY = {f"AUDIT-{arm_id}": arm_id for arm_id in PRIMARY_ARM_IDS}
STREAM_SUFFIXES = ("LOG_A", "Q", "COS_I", "OMEGA", "OMEGA_ARGUMENT", "MEAN_ANOMALY")
LHS_DOMAIN = b"jx-xp1-lhs-u64/v1\0"
TRACER_DIGEST_DOMAIN = b"jx-xp1-canonical-tracer-design/v1\0"
REBOUND_TREE_DIGEST_DOMAIN = b"jx-e2-rebound-python-sources/v1\0"
FLOAT_EPSILON = sys.float_info.epsilon
NANOSECONDS_PER_SECOND = 1_000_000_000

_REBOUND_CACHE: tempfile.TemporaryDirectory[str] | None = None

ANALYSIS_COMPLETE_STATUS = "COMPLETE_AT_BOTH_RESOLUTIONS"
ANALYSIS_SUPPRESSED_STATUS = "SUPPRESSED_REQUIRED_FINITE_ORBIT_METRICS_INCOMPLETE"
INDEPENDENT_RECOMPUTATION_KEYS = {
    "seeds_and_64_tracers",
    "all_14_initial_states_without_integration",
    "particle_summaries",
    "two_resolution_integer_effects_and_block_effects_or_locked_suppression",
    "wasserstein1_or_locked_suppression",
    "all_seven_timestep_pairs_or_locked_suppression",
    "raw_class_agreement_and_emitted_classification_or_locked_suppression",
}


class IntegrityError(RuntimeError):
    """The frozen design or generated artifact failed an integrity condition."""


class ResourceLimitError(RuntimeError):
    """A predeclared local resource cap was reached."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def serialized_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite_float(value: str) -> float:
    parsed = float(value)
    exact = decimal.Decimal(value)
    if not math.isfinite(parsed) or not exact.is_finite() or (parsed == 0.0 and exact != 0):
        raise ValueError(f"non-finite JSON number: {value}")
    return parsed


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def strict_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f"JSON input is not a regular non-symlink file: {path}")
    parsed = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_float=_finite_float,
        parse_constant=_reject_constant,
    )
    if not isinstance(parsed, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return parsed


def strict_json_bytes(payload: bytes, source: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"non-UTF8 JSON payload: {source}") from exc
    parsed = json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_float=_finite_float,
        parse_constant=_reject_constant,
    )
    if not isinstance(parsed, dict):
        raise ValueError(f"JSON root must be an object: {source}")
    return parsed


def atomic_json(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    payload = serialized_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.pending")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"stale atomic-write file exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def elapsed_seconds(started_ns: int, now_ns: int | None = None) -> float:
    if now_ns is None:
        now_ns = time.monotonic_ns()
    return (now_ns - started_ns) / NANOSECONDS_PER_SECOND


def deadline_ns(started_ns: int, seconds: float) -> int:
    return started_ns + int(seconds * NANOSECONDS_PER_SECOND)


def deadline_expired(now_ns: int, locked_deadline_ns: int) -> bool:
    return now_ns >= locked_deadline_ns


def directory_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            total += path.stat().st_size
    return total


def protected_tree_roots(contract: dict[str, Any], package_root: Path) -> list[Path]:
    roots = [package_root.resolve()]
    roots.extend(
        (package_root / binding["path"]).resolve().parent
        for binding in contract["excluded_context_bindings"]
    )
    return roots


def trees_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def validate_clean_output_directory(
    output_dir: Path, contract: dict[str, Any], package_root: Path
) -> Path:
    if output_dir.is_symlink():
        raise ValueError("clean output directory must not be a symlink")
    resolved = output_dir.resolve()
    if resolved.exists():
        raise FileExistsError("clean output directory must be absent")
    if not resolved.parent.is_dir():
        raise ValueError("clean output directory parent must already exist")
    if any(
        trees_overlap(resolved, protected)
        for protected in protected_tree_roots(contract, package_root)
    ):
        raise ValueError("output directory overlaps a package or bound-context tree")
    return resolved


def rebound_python_tree(root: Path) -> tuple[int, str]:
    files = sorted(root.rglob("*.py"), key=lambda path: path.relative_to(root).as_posix())
    digest = hashlib.sha256()
    digest.update(REBOUND_TREE_DIGEST_DOMAIN)
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return len(files), digest.hexdigest()


def get_rebound() -> Any:
    """Import only the hash-bound REBOUND source tree and native library."""
    global _REBOUND_CACHE
    existing = sys.modules.get("rebound")
    if existing is None:
        specification = importlib.util.find_spec("rebound")
        if specification is None or not specification.submodule_search_locations:
            raise RuntimeError("cannot resolve REBOUND source package")
        source_root = Path(next(iter(specification.submodule_search_locations))).resolve()
        count, digest = rebound_python_tree(source_root)
        if (count, digest) != (EXPECTED_REBOUND_SOURCE_COUNT, EXPECTED_REBOUND_SOURCE_SHA256):
            raise RuntimeError(f"REBOUND source-tree mismatch: {(count, digest)}")
        binary_candidates = sorted(source_root.parent.glob("librebound*.so"))
        if len(binary_candidates) != 1:
            raise RuntimeError("expected exactly one REBOUND native library")
        if sha256_file(binary_candidates[0]) != EXPECTED_REBOUND_BINARY_SHA256:
            raise RuntimeError("REBOUND native-library hash mismatch before import")
        _REBOUND_CACHE = tempfile.TemporaryDirectory(prefix="jx-xp1-source-import-")
        sys.pycache_prefix = _REBOUND_CACHE.name
        sys.dont_write_bytecode = True
        importlib.invalidate_caches()
        existing = importlib.import_module("rebound")
        setattr(existing, "_jx_xp1_source_only_import", True)
        setattr(existing, "_jx_xp1_source_cache_holder", _REBOUND_CACHE)
    elif getattr(existing, "_jx_xp1_source_only_import", False) is not True:
        raise RuntimeError("REBOUND was imported before the XP1 source-only guard")
    count, digest = rebound_python_tree(Path(existing.__file__).resolve().parent)
    if (count, digest) != (EXPECTED_REBOUND_SOURCE_COUNT, EXPECTED_REBOUND_SOURCE_SHA256):
        raise RuntimeError("REBOUND source-tree mismatch after import")
    return existing


def validate_runtime(contract: dict[str, Any]) -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    rebound = get_rebound()
    binary = Path(rebound.clibrebound._name).resolve()
    source_count, source_digest = rebound_python_tree(Path(rebound.__file__).resolve().parent)
    actual = {
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "python_executable_sha256": sha256_file(executable),
        "rebound_version": rebound.__version__,
        "rebound_build": rebound.__build__,
        "rebound_binary_sha256": sha256_file(binary),
        "rebound_python_source_file_count": source_count,
        "rebound_python_source_sha256": source_digest,
    }
    if actual != contract["runtime_lock"]:
        raise RuntimeError(f"runtime lock mismatch: {actual}")
    return actual


def derive_seed(domain: str, design_core_sha256: str, label: str, counter: int) -> bytes:
    domain_bytes = domain.encode("ascii")
    label_bytes = label.encode("ascii")
    payload = (
        len(domain_bytes).to_bytes(4, "big")
        + domain_bytes
        + bytes.fromhex(design_core_sha256)
        + len(label_bytes).to_bytes(4, "big")
        + label_bytes
        + counter.to_bytes(8, "big")
    )
    return hashlib.sha256(payload).digest()


def validate_seed_manifest(
    contract: dict[str, Any], manifest_path: Path
) -> tuple[dict[str, Any], dict[str, bytes]]:
    if manifest_path.is_symlink():
        raise IntegrityError("seed manifest must not be a symlink")
    if sha256_file(manifest_path) != EXPECTED_SEED_MANIFEST_SHA256:
        raise IntegrityError("seed-manifest byte hash mismatch")
    manifest = strict_json(manifest_path)
    policy = contract["seed_policy"]
    expected_keys = {
        "schema", "experiment_id", "artifact_class", "design_core_sha256",
        "domain_ascii", "derivation", "seed_bytes_used", "encoding",
        "external_randomness_used", "outcome_or_prior_trajectory_used",
        "override_allowed", "streams", "mandatory_nonclaim",
    }
    if set(manifest) != expected_keys:
        raise IntegrityError("seed-manifest top-level shape changed")
    if (
        manifest["schema"] != "jx-xp1-local-seed-manifest/v1"
        or manifest["experiment_id"] != EXPECTED_EXPERIMENT_ID
        or manifest["design_core_sha256"] != EXPECTED_DESIGN_CORE_SHA256
        or manifest["domain_ascii"] != policy["domain_ascii"]
        or manifest["derivation"] != policy["stream_formula"]
        or manifest["seed_bytes_used"] != 16
        or manifest["encoding"] != "LOWERCASE_HEX_BIG_ENDIAN"
        or manifest["external_randomness_used"] is not False
        or manifest["outcome_or_prior_trajectory_used"] is not False
        or manifest["override_allowed"] is not False
    ):
        raise IntegrityError("seed-manifest identity or policy changed")
    expected_labels = [
        f"LHS_BLOCK_{block}_{suffix}"
        for block in range(4)
        for suffix in STREAM_SUFFIXES
    ]
    streams = manifest["streams"]
    if not isinstance(streams, list) or len(streams) != 24:
        raise IntegrityError("seed stream count changed")
    seeds: dict[str, bytes] = {}
    for expected_label, row in zip(expected_labels, streams, strict=True):
        if set(row) != {"stream_label", "counter", "seed_hex_128"}:
            raise IntegrityError("seed stream row shape changed")
        if row["stream_label"] != expected_label or row["counter"] != 0:
            raise IntegrityError("seed stream identity/order changed")
        digest = derive_seed(
            manifest["domain_ascii"], manifest["design_core_sha256"], expected_label, 0
        )
        expected_hex = digest[:16].hex()
        if row["seed_hex_128"] != expected_hex:
            raise IntegrityError(f"seed derivation mismatch: {expected_label}")
        seeds[expected_label] = bytes.fromhex(expected_hex)
    return manifest, seeds


def lhs_values(seed: bytes) -> tuple[list[int], list[float]]:
    if len(seed) != 16:
        raise ValueError("LHS seed must be exactly 16 bytes")
    counter = 0

    def next_u64() -> int:
        nonlocal counter
        digest = hashlib.sha256(LHS_DOMAIN + seed + counter.to_bytes(8, "big")).digest()
        counter += 1
        return int.from_bytes(digest[:8], "big")

    permutation = list(range(16))
    modulus = 1 << 64
    for index in range(15, 0, -1):
        divisor = index + 1
        limit = modulus - modulus % divisor
        while True:
            word = next_u64()
            if word < limit:
                break
        swap = word % divisor
        permutation[index], permutation[swap] = permutation[swap], permutation[index]
    values = [
        (stratum + (next_u64() >> 11) / float(1 << 53)) / 16.0
        for stratum in permutation
    ]
    return permutation, values


def make_tracers(contract: dict[str, Any], seeds: dict[str, bytes]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    canonical_rows: list[dict[str, Any]] = []
    block_zero_log_a: tuple[list[int], list[float]] | None = None
    for block in range(4):
        dimensions: dict[str, list[float]] = {}
        for suffix in STREAM_SUFFIXES:
            permutation, values = lhs_values(seeds[f"LHS_BLOCK_{block}_{suffix}"])
            dimensions[suffix] = values
            if block == 0 and suffix == "LOG_A":
                block_zero_log_a = permutation, values
        for index in range(16):
            a_au = math.exp(
                math.log(150.0)
                + dimensions["LOG_A"][index] * (math.log(800.0) - math.log(150.0))
            )
            q_au = 35.0 + 45.0 * dimensions["Q"][index]
            eccentricity = 1.0 - q_au / a_au
            cos_i = math.cos(math.radians(40.0)) + dimensions["COS_I"][index] * (
                1.0 - math.cos(math.radians(40.0))
            )
            inclination = math.acos(cos_i)
            row = {
                "logical_id": f"XP1-B{block:02d}-T{index:02d}",
                "block_index": block,
                "index_within_block": index,
                "a_AU": a_au,
                "q_AU": q_au,
                "e": eccentricity,
                "i_rad": inclination,
                "Omega_rad": 2.0 * math.pi * dimensions["OMEGA"][index],
                "omega_rad": 2.0 * math.pi * dimensions["OMEGA_ARGUMENT"][index],
                "M_rad": 2.0 * math.pi * dimensions["MEAN_ANOMALY"][index],
            }
            if not (
                150.0 <= a_au <= 800.0
                and 35.0 <= q_au <= 80.0
                and 0.0 <= eccentricity < 1.0
                and 0.0 <= inclination <= math.radians(40.0)
            ):
                raise IntegrityError("realized tracer is outside the frozen design bounds")
            rows.append(row)
            canonical_rows.append({
                "logical_id": row["logical_id"],
                "block_index": block,
                "index_within_block": index,
                "a_AU_hex": a_au.hex(),
                "q_AU_hex": q_au.hex(),
                "e_hex": eccentricity.hex(),
                "i_rad_hex": inclination.hex(),
                "Omega_rad_hex": row["Omega_rad"].hex(),
                "omega_rad_hex": row["omega_rad"].hex(),
                "M_rad_hex": row["M_rad"].hex(),
            })
    if block_zero_log_a is None:
        raise AssertionError("missing LHS test vector")
    test = contract["seed_policy"]["lhs_construction"]
    if block_zero_log_a[0] != test["block_0_log_a_permutation_test_vector"]:
        raise IntegrityError("LHS permutation test vector mismatch")
    if [value.hex() for value in block_zero_log_a[1][:4]] != test[
        "block_0_log_a_first_four_lhs_float_hex"
    ]:
        raise IntegrityError("LHS jitter test vector mismatch")
    digest = sha256_bytes(TRACER_DIGEST_DOMAIN + canonical_bytes(canonical_rows))
    if digest != EXPECTED_TRACER_ROWS_SHA256 or digest != test["canonical_rows_sha256"]:
        raise IntegrityError("canonical realized-tracer digest mismatch")
    return rows, canonical_rows, digest


def validate_contract(contract: dict[str, Any], contract_path: Path) -> None:
    if contract_path.name != "contract_v1.json" or sha256_file(contract_path) != EXPECTED_CONTRACT_SHA256:
        raise IntegrityError("frozen contract byte hash mismatch")
    if contract.get("schema") != CONTRACT_SCHEMA or contract.get("experiment_id") != EXPECTED_EXPERIMENT_ID:
        raise IntegrityError("contract identity changed")
    if contract.get("claim_ceiling") != "SYNTHETIC_250KYR_RESPONSE_ONLY":
        raise IntegrityError("claim ceiling changed")
    if contract.get("permissions") != {
        "local_cpu_execution_authorized": True,
        "network_access_authorized": False,
        "gpu_execution_authorized": False,
        "observed_data_access_authorized": False,
        "survey_adapter_execution_authorized": False,
        "jx_o2_execution_or_g0_evidence_authorized": False,
        "planet_x_detection_exclusion_constraint_or_preference_claim_authorized": False,
    }:
        raise IntegrityError("permission boundary changed")
    design_digest = sha256_bytes(canonical_bytes(contract["design_core"]))
    if design_digest != EXPECTED_DESIGN_CORE_SHA256 or design_digest != contract["seed_policy"]["design_core_sha256"]:
        raise IntegrityError("design-core canonical digest mismatch")
    shape = contract["execution_shape_lock"]
    if shape != {
        "primary_arm_count": 7,
        "audit_arm_count": 7,
        "total_arm_count": 14,
        "tracers_in_every_arm": 64,
        "analysis_blocks": 4,
        "particles_per_analysis_block": 16,
        "analysis_blocks_are_not_arm_multipliers": True,
        "checkpoints_or_restart_files": "FORBIDDEN",
        "raw_trajectories": "FORBIDDEN",
    }:
        raise IntegrityError("execution shape changed")
    core = contract["design_core"]
    if tuple(core["primary_arm_ids"]) != PRIMARY_ARM_IDS or tuple(core["audit_arm_ids"]) != AUDIT_ARM_IDS:
        raise IntegrityError("arm matrix changed")
    if core["tracer_design"]["total_tracers"] != 64 or core["tracer_design"]["block_count"] != 4:
        raise IntegrityError("tracer cardinality changed")
    dynamics = core["dynamics"]
    if (
        dynamics["duration_years"] != 250000.0
        or dynamics["sample_cadence_years"] != 50.0
        or dynamics["sample_count_including_t0"] != 5001
        or dynamics["primary_dt_years"] != 0.125
        or dynamics["audit_dt_years"] != 0.0625
        or dynamics["r_crit_hill"] != 3.0
        or dynamics["safe_mode"] != 1
        or dynamics["testparticle_type"] != 0
    ):
        raise IntegrityError("dynamics changed")
    if int(dynamics["duration_years"] / dynamics["sample_cadence_years"]) + 1 != dynamics[
        "sample_count_including_t0"
    ]:
        raise IntegrityError("sample cardinality is inconsistent")
    caps = contract["resource_caps_per_execution"]
    if caps != {
        "workers": 1,
        "worker_interpretation": "ONE_NUMERICAL_CHILD_AT_A_TIME_PLUS_NONNUMERICAL_PARENT_COORDINATOR",
        "max_wall_seconds_total": 3600.0,
        "max_wall_seconds_per_arm": 600.0,
        "max_peak_rss_bytes": 2147483648,
        "max_output_bytes": 134217728,
        "minimum_free_disk_bytes": 1073741824,
        "gpu_used": False,
        "scope": "EACH_A_OR_B_EXECUTION_INDEPENDENTLY",
        "aggregate_a_b_verification_wall_limit_promised": False,
        "native_dynamics_watchdog": "ONE_POSIX_FORKED_CHILD_PER_ARM_WITH_PARENT_MONOTONIC_NS_POLL_AND_IMMEDIATE_KILL_ON_CAP_OR_FAILURE",
        "watchdog_platform_lock": "LINUX_POSIX_FORK_AND_PROCFS",
        "watchdog_clock": "PYTHON_TIME_MONOTONIC_NS",
        "deadline_expiry_rule": "NOW_NS_GREATER_THAN_OR_EQUAL_TO_DEADLINE_NS",
        "arm_wall_scope": "BEFORE_PROCESS_START_THROUGH_VALIDATED_RESPONSE_RECEIPT_AND_WORKER_EXIT",
        "fork_precondition": "COORDINATOR_PROCFS_THREAD_COUNT_EXACTLY_ONE_BEFORE_EVERY_FORK",
        "watchdog_poll_seconds": 0.25,
        "watchdog_live_rss_source": "LINUX_PROC_PID_STATUS_VM_HWM",
        "watchdog_terminal_rss_source": "POSIX_WAIT4_RUSAGE_RU_MAXRSS",
        "worker_response_transport": "PARENT_NONBLOCKING_UINT64_BE_LENGTH_FRAMED_UTF8_JSON_PIPE_WITH_DEADLINE_CHECKS",
        "rss_cap_scope": "PARENT_AND_EACH_SINGLE_ACTIVE_CHILD_PROCESS_PEAK_CHECKED_INDIVIDUALLY_NOT_SUMMED",
        "watchdog_scope": "NATIVE_ARM_DYNAMICS_HARD_PROCESS_BOUNDARY_WITH_SYNCHRONOUS_CHECKS_AROUND_NONINTEGRATING_SETUP_ANALYSIS_AND_FINALIZATION",
        "preflight_contains_no_integration_and_resource_checks_occur_before_and_after": True,
        "finalization_wall_reserve_seconds": 5.0,
        "final_result_is_written_via_fsynced_pending_file_then_size_and_resource_checked_before_atomic_publish": True,
    }:
        raise IntegrityError("resource caps changed")
    for binding in contract["excluded_context_bindings"]:
        candidate = contract_path.parent / binding["path"]
        path = candidate.resolve()
        if (
            candidate.is_symlink() or not path.is_file() or path.stat().st_nlink != 1
            or sha256_file(path) != binding["sha256"]
        ):
            raise IntegrityError(f"excluded-context binding mismatch: {binding['id']}")
    models = core["m1_physical_cases"]
    forbidden = contract["retired_candidate_nonuse_lock"]
    forbidden_tuple = forbidden["forbidden_orbit_tuple"]
    for model in models:
        if model["id"] == forbidden["forbidden_historical_name"]:
            raise IntegrityError("retired candidate identifier entered the model set")
        comparable = {
            "mass_Mearth": model["mass_Mearth"], "a_AU": model["a_AU"],
            "e": model["e"], "i_deg": model["i_deg"],
        }
        if all(comparable[key] == forbidden_tuple[key] for key in comparable):
            raise IntegrityError("retired candidate physical tuple entered the model set")


def validate_registration(
    registration_path: Path, contract_path: Path, runner_path: Path
) -> tuple[dict[str, Any], str]:
    if registration_path.is_symlink():
        raise ValueError("registration path must not be a symlink")
    registration_path = registration_path.resolve()
    root = registration_path.parent
    if registration_path != root / "registration_v1.json":
        raise ValueError("registration path must be canonical registration_v1.json")
    registration = strict_json(registration_path)
    contract = strict_json(contract_path)
    expected_keys = {
        "schema", "experiment_id", "artifact_class", "registration_state",
        "recorded_at_utc", "timestamp_authority", "externally_timestamped",
        "scientific_evidence_artifact", "outcomes_generated",
        "execution_permissions", "locked_files", "mandatory_nonclaim",
    }
    if set(registration) != expected_keys:
        raise IntegrityError("registration top-level shape changed")
    expected_permissions = {
        "execution_a_authorized": True,
        "execution_b_authorized_only_after_verified_a": True,
        "local_cpu_only": True,
        "network_access_authorized": False,
        "gpu_execution_authorized": False,
        "observed_data_access_authorized": False,
        "survey_adapter_execution_authorized": False,
        "jx_o2_execution_or_g0_evidence_authorized": False,
        "planet_x_claim_authorized": False,
    }
    if (
        registration["schema"] != REGISTRATION_SCHEMA
        or registration["experiment_id"] != EXPECTED_EXPERIMENT_ID
        or registration["artifact_class"] != "LOCAL_CONTENT_HASH_REGISTRATION_ONLY"
        or registration["registration_state"] != "LOCAL_CONTENT_HASH_LOCK_COMPLETE_BEFORE_ANY_XP1_NUMERICAL_OUTPUT"
        or registration["timestamp_authority"] != "LOCAL_CONTENT_HASH_ONLY_NO_EXTERNAL_TIMESTAMP"
        or registration["externally_timestamped"] is not False
        or registration["scientific_evidence_artifact"] is not False
        or registration["outcomes_generated"] is not False
        or registration["execution_permissions"] != expected_permissions
        or registration["mandatory_nonclaim"] != contract["mandatory_nonclaim"]
        or not isinstance(registration["recorded_at_utc"], str)
        or not registration["recorded_at_utc"].endswith("Z")
    ):
        raise IntegrityError("registration identity or permission boundary changed")
    locked = registration["locked_files"]
    if not isinstance(locked, dict) or set(locked) != LOCKED_FILES:
        raise IntegrityError("registration locked-file inventory changed")
    expected_package_inventory = set(contract["result_policy"]["registered_package_inventory"])
    if expected_package_inventory != LOCKED_FILES | {"registration_v1.json"}:
        raise IntegrityError("contract registered-package inventory lock changed")
    if {path.name for path in root.iterdir()} != expected_package_inventory:
        raise IntegrityError("registered package has an extra or missing filesystem entry")
    if contract_path.resolve() != root / "contract_v1.json" or runner_path.resolve() != root / "run_exploratory.py":
        raise IntegrityError("registration inputs are not canonical package files")
    forbidden_hashes = {
        contract["retired_candidate_nonuse_lock"]["forbidden_metadata_sha256"],
        contract["retired_candidate_nonuse_lock"]["forbidden_historical_source_state_sha256"],
        contract["retired_candidate_nonuse_lock"]["forbidden_catalog_sha256"],
    }
    for relative, expected in locked.items():
        candidate = root / relative
        path = candidate.resolve()
        if (
            candidate.is_symlink() or path != candidate.absolute()
            or path.parent != root or not path.is_file() or path.stat().st_nlink != 1
        ):
            raise IntegrityError(f"locked file is unavailable or escaped: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise IntegrityError(f"registration locked-file mismatch: {relative}")
        if actual in forbidden_hashes:
            raise IntegrityError(f"retired candidate payload entered package: {relative}")
    return registration, sha256_file(registration_path)


def vector_norm(values: Iterable[float]) -> float:
    return math.sqrt(math.fsum(value * value for value in values))


def vector_subtract(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(a - b for a, b in zip(left, right, strict=True))


def cross(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def active_snapshot(simulation: Any) -> dict[str, Any]:
    particles = [simulation.particles[index] for index in range(simulation.N_active)]
    masses = [float(particle.m) for particle in particles]
    positions = [(float(p.x), float(p.y), float(p.z)) for p in particles]
    velocities = [(float(p.vx), float(p.vy), float(p.vz)) for p in particles]
    total_mass = math.fsum(masses)
    momentum = tuple(
        math.fsum(masses[index] * velocities[index][axis] for index in range(len(particles)))
        for axis in range(3)
    )
    r_com = tuple(
        math.fsum(masses[index] * positions[index][axis] for index in range(len(particles))) / total_mass
        for axis in range(3)
    )
    v_com = tuple(value / total_mass for value in momentum)
    relative_positions = [vector_subtract(position, r_com) for position in positions]
    relative_velocities = [vector_subtract(velocity, v_com) for velocity in velocities]
    angular_terms = [cross(relative_positions[index], relative_velocities[index]) for index in range(len(particles))]
    com_angular = tuple(
        math.fsum(masses[index] * angular_terms[index][axis] for index in range(len(particles)))
        for axis in range(3)
    )
    linear_scale = math.fsum(
        masses[index] * vector_norm(relative_velocities[index]) for index in range(len(particles))
    )
    kinetic_internal = 0.5 * math.fsum(
        masses[index] * vector_norm(relative_velocities[index]) ** 2
        for index in range(len(particles))
    )
    potential_terms = []
    for left in range(len(particles)):
        for right in range(left + 1, len(particles)):
            separation = vector_norm(vector_subtract(positions[left], positions[right]))
            if not math.isfinite(separation) or separation <= 0.0:
                raise IntegrityError("invalid active-particle separation")
            potential_terms.append(
                -float(simulation.G) * masses[left] * masses[right] / separation
            )
    intrinsic_energy = math.fsum((kinetic_internal, math.fsum(potential_terms)))
    values = (
        *momentum, *r_com, *v_com, *com_angular, linear_scale,
        intrinsic_energy,
    )
    if not all(math.isfinite(value) for value in values):
        raise IntegrityError("non-finite active invariant")
    if (
        linear_scale <= 0.0
        or vector_norm(com_angular) <= 0.0
        or abs(intrinsic_energy) <= 0.0
    ):
        raise IntegrityError("non-positive active invariant denominator")
    return {
        "momentum": momentum,
        "r_com": r_com,
        "v_com": v_com,
        "com_angular": com_angular,
        "linear_internal_scale": linear_scale,
        "intrinsic_energy": intrinsic_energy,
    }


def update_invariant_maximum(maximum: dict[str, float], initial: dict[str, Any], current: dict[str, Any]) -> None:
    values = {
        "relative_active_energy_drift": abs(
            current["intrinsic_energy"] - initial["intrinsic_energy"]
        ) / abs(initial["intrinsic_energy"]),
        "relative_active_com_angular_momentum_vector_drift": vector_norm(
            vector_subtract(current["com_angular"], initial["com_angular"])
        ) / vector_norm(initial["com_angular"]),
        "scale_normalized_active_linear_momentum_residual": vector_norm(
            vector_subtract(current["momentum"], initial["momentum"])
        ) / initial["linear_internal_scale"],
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise IntegrityError("non-finite active invariant metric")
    for key, value in values.items():
        maximum[key] = max(maximum[key], value)


def active_com_shift(simulation: Any) -> None:
    snapshot = active_snapshot(simulation)
    for particle in simulation.particles:
        particle.x -= snapshot["r_com"][0]
        particle.y -= snapshot["r_com"][1]
        particle.z -= snapshot["r_com"][2]
        particle.vx -= snapshot["v_com"][0]
        particle.vy -= snapshot["v_com"][1]
        particle.vz -= snapshot["v_com"][2]


def relative_components(simulation: Any, common_names: list[str]) -> list[dict[str, Any]]:
    sun = simulation.particles["Sun"]
    rows = []
    for name in common_names:
        particle = simulation.particles[name]
        rows.append({
            "logical_id": name,
            "components_hex": [
                (float(particle.x) - float(sun.x)).hex(),
                (float(particle.y) - float(sun.y)).hex(),
                (float(particle.z) - float(sun.z)).hex(),
                (float(particle.vx) - float(sun.vx)).hex(),
                (float(particle.vy) - float(sun.vy)).hex(),
                (float(particle.vz) - float(sun.vz)).hex(),
            ],
        })
    return rows


def decoded_state_sha256(simulation: Any) -> str:
    digest = hashlib.sha256(b"jx-xp1-decoded-state/v1\0")
    configuration = {
        "t_hex": float(simulation.t).hex(), "G_hex": float(simulation.G).hex(),
        "dt_hex": float(simulation.dt).hex(), "N": simulation.N,
        "N_active": simulation.N_active, "integrator": simulation.integrator,
        "testparticle_type": simulation.testparticle_type,
        "r_crit_hill_hex": float(simulation.ri_mercurius.r_crit_hill).hex(),
        "safe_mode": int(simulation.ri_mercurius.safe_mode),
    }
    digest.update(canonical_bytes(configuration))
    for index, particle in enumerate(simulation.particles):
        digest.update(struct.pack(
            "!II8d", index, int(particle.hash.value), float(particle.m), float(particle.r),
            float(particle.x), float(particle.y), float(particle.z),
            float(particle.vx), float(particle.vy), float(particle.vz),
        ))
    return digest.hexdigest()


def arm_specifications(contract: dict[str, Any]) -> list[dict[str, Any]]:
    models = {item["id"]: item for item in contract["design_core"]["m1_physical_cases"]}
    angles = {item["id"]: item for item in contract["design_core"]["orientation_probes"]}

    def parts(base: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if base == "M0":
            return None, None
        model_id, angle_id = base.split("-")
        return models[model_id], angles[angle_id]

    rows = []
    for arm_id in PRIMARY_ARM_IDS:
        model, angle = parts(arm_id)
        rows.append({
            "arm_id": arm_id, "arm_class": "PRIMARY", "primary_arm_id": None,
            "model": model, "angle": angle,
            "dt_years": contract["design_core"]["dynamics"]["primary_dt_years"],
        })
    for arm_id in AUDIT_ARM_IDS:
        primary = AUDIT_PRIMARY[arm_id]
        model, angle = parts(primary)
        rows.append({
            "arm_id": arm_id, "arm_class": "TIMESTEP_SENTINEL", "primary_arm_id": primary,
            "model": model, "angle": angle,
            "dt_years": contract["design_core"]["dynamics"]["audit_dt_years"],
        })
    return rows


def build_simulation(
    contract: dict[str, Any], tracers: list[dict[str, Any]], spec: dict[str, Any]
) -> tuple[Any, list[str], str, float]:
    rebound = get_rebound()
    core = contract["design_core"]
    units = core["units_and_frame"]
    active = core["common_active_system"]
    simulation = rebound.Simulation()
    simulation.G = float(units["G_AU3_Msun_yr2"])
    simulation.add(m=float(active["sun_mass_Msun"]), hash="Sun")
    common_names = ["Sun"]
    for body in active["giants"]:
        simulation.add(
            primary=simulation.particles["Sun"], m=float(body["mass_Msun"]),
            a=float(body["a_AU"]), e=0.0, inc=0.0, Omega=0.0, omega=0.0,
            M=math.radians(float(body["initial_longitude_deg"])), hash=body["name"],
        )
        common_names.append(body["name"])
    model = spec["model"]
    angle = spec["angle"]
    if model is not None:
        if angle is None:
            raise IntegrityError("M1 arm is missing its orientation")
        omega_deg = (float(angle["varpi_deg"]) - float(angle["Omega_deg"])) % 360.0
        simulation.add(
            primary=simulation.particles["Sun"],
            m=float(model["mass_Mearth"]) * float(active["earth_to_sun_mass_ratio"]),
            a=float(model["a_AU"]), e=float(model["e"]), inc=math.radians(float(model["i_deg"])),
            Omega=math.radians(float(angle["Omega_deg"])), omega=math.radians(omega_deg),
            M=math.radians(float(angle["M_deg"])), hash=f"XP1-{model['id']}-{angle['id']}",
        )
    simulation.N_active = simulation.N
    for tracer in tracers:
        simulation.add(
            primary=simulation.particles["Sun"], m=0.0,
            a=tracer["a_AU"], e=tracer["e"], inc=tracer["i_rad"],
            Omega=tracer["Omega_rad"], omega=tracer["omega_rad"], M=tracer["M_rad"],
            hash=tracer["logical_id"],
        )
        common_names.append(tracer["logical_id"])
    pre_rows = relative_components(simulation, common_names)
    pre_digest = sha256_bytes(b"jx-xp1-pre-com-common-relative-state/v1\0" + canonical_bytes(pre_rows))
    active_com_shift(simulation)
    post_rows = relative_components(simulation, common_names)
    maximum_epsilon_units = 0.0
    for before, after in zip(pre_rows, post_rows, strict=True):
        if before["logical_id"] != after["logical_id"]:
            raise IntegrityError("common-particle identity changed during COM shift")
        for reference_hex, actual_hex in zip(before["components_hex"], after["components_hex"], strict=True):
            reference = float.fromhex(reference_hex)
            actual = float.fromhex(actual_hex)
            ratio = abs(actual - reference) / (FLOAT_EPSILON * max(1.0, abs(reference)))
            maximum_epsilon_units = max(maximum_epsilon_units, ratio)
    if maximum_epsilon_units > float(
        contract["initial_pairing_lock"]["post_translation_max_binary64_epsilon_units_per_component"]
    ):
        raise IntegrityError("post-COM common-state tolerance exceeded")
    dynamics = core["dynamics"]
    simulation.integrator = "mercurius"
    simulation.dt = float(spec["dt_years"])
    simulation.testparticle_type = int(dynamics["testparticle_type"])
    simulation.ri_mercurius.r_crit_hill = float(dynamics["r_crit_hill"])
    simulation.ri_mercurius.safe_mode = int(dynamics["safe_mode"])
    simulation.collision = "none"
    if (
        simulation.integrator != "mercurius"
        or simulation.dt != float(spec["dt_years"])
        or simulation.testparticle_type != 0
        or simulation.ri_mercurius.r_crit_hill != 3.0
        or int(simulation.ri_mercurius.safe_mode) != 1
    ):
        raise IntegrityError("REBOUND configuration readback mismatch")
    return simulation, common_names, pre_digest, maximum_epsilon_units


def preflight_initial_states(
    contract: dict[str, Any], tracers: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    reference_digest: str | None = None
    rows: dict[str, dict[str, Any]] = {}
    for spec in arm_specifications(contract):
        simulation, common_names, digest, epsilon_units = build_simulation(contract, tracers, spec)
        if reference_digest is None:
            reference_digest = digest
        elif digest != reference_digest:
            raise IntegrityError("pre-translation common relative state is not bitwise exact")
        expected_active = 5 if spec["model"] is None else 6
        if simulation.N_active != expected_active or simulation.N != expected_active + 64:
            raise IntegrityError("initial particle cardinality changed")
        rows[spec["arm_id"]] = {
            "pre_translation_common_relative_state_sha256": digest,
            "post_com_max_binary64_epsilon_units": epsilon_units,
            "decoded_initial_state_sha256": decoded_state_sha256(simulation),
            "particle_count": simulation.N,
            "active_particle_count": simulation.N_active,
            "common_particle_count": len(common_names),
        }
    if len(rows) != 14:
        raise IntegrityError("preflight arm count changed")
    return rows


def blank_tracker(tracers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "logical_id": row["logical_id"], "block_index": row["block_index"],
        "index_within_block": row["index_within_block"],
        "minimum_sampled_q_AU": None,
        "first_sampled_q_below_35_time_year": None,
        "first_sampled_q_below_30_time_year": None,
        "final_a_AU": None, "final_e": None, "final_i_deg": None,
        "final_q_AU": None, "final_distance_AU": None,
        "final_finite_and_bound": False,
        "all_samples_finite_osculating_orbit": True,
    } for row in tracers]


def cartesian_finite(simulation: Any) -> bool:
    return all(
        math.isfinite(float(getattr(particle, field)))
        for particle in simulation.particles
        for field in ("x", "y", "z", "vx", "vy", "vz")
    )


def sample_tracers(simulation: Any, tracker: list[dict[str, Any]], time_year: float) -> None:
    if not cartesian_finite(simulation):
        raise IntegrityError("non-finite Cartesian particle state")
    sun = simulation.particles["Sun"]
    tracer_start = simulation.N_active
    for offset, row in enumerate(tracker):
        particle = simulation.particles[tracer_start + offset]
        finite_orbit = False
        bound = False
        a_au = eccentricity = inclination_deg = q_au = None
        try:
            orbit = particle.orbit(primary=sun)
            a_au = float(orbit.a)
            eccentricity = float(orbit.e)
            inclination_deg = math.degrees(float(orbit.inc))
            q_au = a_au * (1.0 - eccentricity)
            finite_orbit = (
                all(math.isfinite(value) for value in (a_au, eccentricity, inclination_deg, q_au))
                and eccentricity >= 0.0 and q_au >= 0.0
            )
            bound = bool(finite_orbit and a_au > 0.0 and eccentricity < 1.0)
        except (ValueError, ZeroDivisionError, OverflowError):
            finite_orbit = False
            bound = False
        if not finite_orbit:
            row["all_samples_finite_osculating_orbit"] = False
        else:
            old = row["minimum_sampled_q_AU"]
            if old is None or q_au < old:
                row["minimum_sampled_q_AU"] = q_au
            if q_au < 35.0 and row["first_sampled_q_below_35_time_year"] is None:
                row["first_sampled_q_below_35_time_year"] = time_year
            if q_au < 30.0 and row["first_sampled_q_below_30_time_year"] is None:
                row["first_sampled_q_below_30_time_year"] = time_year
        if time_year == 250000.0:
            distance = math.hypot(
                float(particle.x) - float(sun.x),
                float(particle.y) - float(sun.y),
                float(particle.z) - float(sun.z),
            )
            if not math.isfinite(distance):
                raise IntegrityError("non-finite final Sun-relative distance")
            row["final_a_AU"] = a_au if finite_orbit else None
            row["final_e"] = eccentricity if finite_orbit else None
            row["final_i_deg"] = inclination_deg if finite_orbit else None
            row["final_q_AU"] = q_au if finite_orbit else None
            row["final_distance_AU"] = distance
            row["final_finite_and_bound"] = bound


def summarize_particles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 64:
        raise IntegrityError("particle metric count changed")
    q30 = sum(row["minimum_sampled_q_AU"] is not None and row["minimum_sampled_q_AU"] < 30.0 for row in rows)
    q35 = sum(row["minimum_sampled_q_AU"] is not None and row["minimum_sampled_q_AU"] < 35.0 for row in rows)
    bound = sum(row["final_finite_and_bound"] for row in rows)
    return {
        "particle_count": 64,
        "q_below_30_hit_count": q30,
        "q_below_30_fraction": q30 / 64.0,
        "q_below_35_hit_count": q35,
        "q_below_35_fraction": q35 / 64.0,
        "final_finite_bound_count": bound,
        "final_finite_bound_fraction": bound / 64.0,
        "all_particles_have_complete_finite_osculating_history": all(
            row["all_samples_finite_osculating_orbit"] for row in rows
        ),
    }


def update_sampled_state_stream(
    digest: Any, simulation: Any, sample_index: int, time_year: float
) -> None:
    digest.update(struct.pack("!QdII", sample_index, time_year, simulation.N, simulation.N_active))
    for particle_index, particle in enumerate(simulation.particles):
        digest.update(struct.pack(
            "!II8d", particle_index, int(particle.hash.value),
            float(particle.m), float(particle.r), float(particle.x), float(particle.y),
            float(particle.z), float(particle.vx), float(particle.vy), float(particle.vz),
        ))


def enforce_resources(
    contract: dict[str, Any], output_dir: Path, execution_started_ns: int,
    arm_started_ns: int | None = None,
) -> None:
    caps = contract["resource_caps_per_execution"]
    now_ns = time.monotonic_ns()
    if deadline_expired(
        now_ns, deadline_ns(execution_started_ns, float(caps["max_wall_seconds_total"]))
    ):
        raise ResourceLimitError("per-execution wall-time cap exceeded")
    if arm_started_ns is not None and deadline_expired(
        now_ns, deadline_ns(arm_started_ns, float(caps["max_wall_seconds_per_arm"]))
    ):
        raise ResourceLimitError("per-arm wall-time cap exceeded")
    if peak_rss_bytes() > int(caps["max_peak_rss_bytes"]):
        raise ResourceLimitError("peak-RSS cap exceeded")
    if output_dir.exists() and directory_bytes(output_dir) > int(caps["max_output_bytes"]):
        raise ResourceLimitError("output byte cap exceeded")
    disk_root = output_dir if output_dir.exists() else output_dir.parent
    if shutil.disk_usage(disk_root).free < int(caps["minimum_free_disk_bytes"]):
        raise ResourceLimitError("free-disk floor violated")


def run_arm(
    contract: dict[str, Any], tracers: list[dict[str, Any]], spec: dict[str, Any],
    preflight: dict[str, Any], output_dir: Path, execution_started_ns: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    arm_started_ns = time.monotonic_ns()
    simulation, _, common_digest, epsilon_units = build_simulation(contract, tracers, spec)
    if (
        common_digest != preflight["pre_translation_common_relative_state_sha256"]
        or decoded_state_sha256(simulation) != preflight["decoded_initial_state_sha256"]
        or epsilon_units != preflight["post_com_max_binary64_epsilon_units"]
    ):
        raise IntegrityError("execution initial state differs from preflight")
    tracker = blank_tracker(tracers)
    initial = active_snapshot(simulation)
    maximum = {
        "relative_active_energy_drift": 0.0,
        "relative_active_com_angular_momentum_vector_drift": 0.0,
        "scale_normalized_active_linear_momentum_residual": 0.0,
    }
    sampled_state_stream = hashlib.sha256(b"jx-xp1-sampled-state-stream/v1\0")
    dynamics = contract["design_core"]["dynamics"]
    sample_count = int(dynamics["sample_count_including_t0"])
    cadence = float(dynamics["sample_cadence_years"])
    for sample_index in range(sample_count):
        target = sample_index * cadence
        if sample_index:
            simulation.integrate(target, exact_finish_time=1)
        if float(simulation.t) != target:
            raise IntegrityError("sample time readback mismatch")
        update_sampled_state_stream(sampled_state_stream, simulation, sample_index, target)
        sample_tracers(simulation, tracker, target)
        update_invariant_maximum(maximum, initial, active_snapshot(simulation))
        enforce_resources(contract, output_dir, execution_started_ns, arm_started_ns)
    gates = contract["numerical_gates"]
    summary = summarize_particles(tracker)
    checks = {
        "sample_count_exact": sample_count == 5001,
        "final_time_exact": float(simulation.t) == 250000.0,
        "particle_count_unchanged": simulation.N == preflight["particle_count"],
        "active_particle_count_unchanged": simulation.N_active == preflight["active_particle_count"],
        "integrator_and_dt_readback_exact": simulation.integrator == "mercurius" and simulation.dt == float(spec["dt_years"]),
        "r_crit_hill_and_safe_mode_readback_exact": simulation.ri_mercurius.r_crit_hill == 3.0 and int(simulation.ri_mercurius.safe_mode) == 1,
        "pre_translation_common_state_exact": common_digest == preflight["pre_translation_common_relative_state_sha256"],
        "post_com_common_state_within_epsilon_gate": epsilon_units <= 64.0,
        "all_samples_cartesian_finite": True,
        "all_particles_have_complete_finite_osculating_history": summary[
            "all_particles_have_complete_finite_osculating_history"
        ],
        "active_energy_drift_within_gate": maximum["relative_active_energy_drift"] <= float(gates["max_relative_active_energy_drift"]),
        "active_com_angular_momentum_drift_within_gate": maximum["relative_active_com_angular_momentum_vector_drift"] <= float(gates["max_relative_active_com_angular_momentum_vector_drift"]),
        "active_linear_momentum_residual_within_gate": maximum["scale_normalized_active_linear_momentum_residual"] <= float(gates["max_scale_normalized_active_linear_momentum_residual"]),
    }
    semantic = {
        "arm_id": spec["arm_id"], "arm_class": spec["arm_class"],
        "primary_arm_id": spec["primary_arm_id"],
        "model_id": None if spec["model"] is None else spec["model"]["id"],
        "orientation_id": None if spec["angle"] is None else spec["angle"]["id"],
        "dt_years": float(spec["dt_years"]), "duration_years": 250000.0,
        "sample_count_including_t0": sample_count,
        "pre_translation_common_relative_state_sha256": common_digest,
        "post_com_max_binary64_epsilon_units": epsilon_units,
        "decoded_initial_state_sha256": preflight["decoded_initial_state_sha256"],
        "decoded_final_state_sha256": decoded_state_sha256(simulation),
        "sampled_state_stream_sha256": sampled_state_stream.hexdigest(),
        "maximum_invariant_metrics": maximum,
        "summary": summary,
        "checks": checks,
        "particle_metrics": tracker,
    }
    provenance = {
        "arm_id": spec["arm_id"],
        "elapsed_seconds": elapsed_seconds(arm_started_ns),
        "peak_rss_bytes": peak_rss_bytes(),
    }
    return semantic, provenance


def parse_proc_status_integer(status_text: str, field: str, unit: str | None) -> int:
    lines = status_text.splitlines()
    for line in lines:
        if line.startswith(f"{field}:"):
            fields = line.split()
            expected_length = 3 if unit is not None else 2
            if len(fields) != expected_length or (unit is not None and fields[2] != unit):
                raise IntegrityError(f"unexpected /proc {field} format")
            return int(fields[1])
    raise IntegrityError(f"process status has no {field} record")


def process_peak_rss_bytes(pid: int) -> int:
    """Read the supervised worker's peak RSS from the registered Linux source."""
    try:
        status_text = Path(f"/proc/{pid}/status").read_text(encoding="ascii")
    except FileNotFoundError:
        return 0
    return peak_rss_from_proc_status(status_text)


def peak_rss_from_proc_status(status_text: str) -> int:
    if not any(line.startswith("VmHWM:") for line in status_text.splitlines()):
        if any(line.startswith("State:") for line in status_text.splitlines()):
            return 0
        raise IntegrityError("process status has neither State nor VmHWM")
    return parse_proc_status_integer(status_text, "VmHWM", "kB") * 1024


def coordinator_thread_count() -> int:
    status_text = Path("/proc/self/status").read_text(encoding="ascii")
    return parse_proc_status_integer(status_text, "Threads", None)


def _kill_and_reap_worker(pid: int | None) -> None:
    if pid is None:
        return
    try:
        os.kill(pid, 9)
    except ProcessLookupError:
        pass
    try:
        os.wait4(pid, 0)
    except ChildProcessError:
        pass


def write_framed_worker_response(file_descriptor: int, value: dict[str, Any]) -> None:
    payload = serialized_json(value)
    frame = len(payload).to_bytes(8, "big") + payload
    view = memoryview(frame)
    while view:
        written = os.write(file_descriptor, view)
        if written <= 0:
            raise OSError("worker response pipe made no progress")
        view = view[written:]


def _arm_worker(
    write_file_descriptor: int, contract: dict[str, Any], tracers: list[dict[str, Any]],
    spec: dict[str, Any], preflight: dict[str, Any], output_dir: Path,
    execution_started_ns: int,
) -> None:
    try:
        semantic, provenance = run_arm(
            contract, tracers, spec, preflight, output_dir, execution_started_ns
        )
        response = {"ok": True, "semantic": semantic, "provenance": provenance}
    except BaseException as exc:
        response = {
            "ok": False,
            "kind": type(exc).__name__,
            "message": REDACTED_FAILURE_MESSAGE,
        }
    try:
        write_framed_worker_response(write_file_descriptor, response)
    finally:
        os.close(write_file_descriptor)


def run_arm_supervised(
    contract: dict[str, Any], tracers: list[dict[str, Any]], spec: dict[str, Any],
    preflight: dict[str, Any], output_dir: Path, execution_started_ns: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run every native integration in a killable one-arm worker process."""
    caps = contract["resource_caps_per_execution"]
    if not hasattr(os, "fork") or not hasattr(os, "wait4"):
        raise IntegrityError("registered POSIX-fork watchdog is unavailable")
    if coordinator_thread_count() != 1:
        raise IntegrityError("coordinator is not single-threaded before POSIX fork")
    arm_started_ns = time.monotonic_ns()
    execution_deadline_ns = deadline_ns(
        execution_started_ns, float(caps["max_wall_seconds_total"])
    )
    arm_deadline_ns = deadline_ns(
        arm_started_ns, float(caps["max_wall_seconds_per_arm"])
    )
    if deadline_expired(arm_started_ns, execution_deadline_ns):
        raise ResourceLimitError("per-execution wall-time cap reached before arm start")
    read_file_descriptor, write_file_descriptor = os.pipe()
    response: dict[str, Any] | None = None
    write_descriptor_open = True
    framed = bytearray()
    expected_payload_bytes: int | None = None
    pipe_closed = False
    child_pid: int | None = None
    child_exit_status: int | None = None
    child_exit_ns: int | None = None
    child_peak_rss_bytes = 0
    try:
        child_pid = os.fork()
        if child_pid == 0:
            try:
                os.close(read_file_descriptor)
                _arm_worker(
                    write_file_descriptor, contract, tracers, spec, preflight,
                    output_dir, execution_started_ns,
                )
                os._exit(0)
            except BaseException:
                try:
                    os.close(write_file_descriptor)
                except OSError:
                    pass
                os._exit(70)
        os.close(write_file_descriptor)
        write_descriptor_open = False
        os.set_blocking(read_file_descriptor, False)
        while response is None or child_exit_status is None:
            enforce_resources(contract, output_dir, execution_started_ns, arm_started_ns)
            now_ns = time.monotonic_ns()
            remaining_ns = min(execution_deadline_ns, arm_deadline_ns) - now_ns
            if remaining_ns <= 0:
                raise ResourceLimitError("supervised arm watchdog deadline exceeded")
            poll_seconds = min(
                float(caps["watchdog_poll_seconds"]),
                remaining_ns / NANOSECONDS_PER_SECOND,
            )
            try:
                ready, _, _ = select.select(
                    [] if pipe_closed else [read_file_descriptor], [], [], poll_seconds
                )
            except InterruptedError:
                continue
            if ready:
                try:
                    chunk = os.read(read_file_descriptor, 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    pipe_closed = True
                    if response is None:
                        raise IntegrityError("arm worker closed with an incomplete response frame")
                else:
                    framed.extend(chunk)
                    if expected_payload_bytes is None and len(framed) >= 8:
                        expected_payload_bytes = int.from_bytes(framed[:8], "big")
                        if not 0 < expected_payload_bytes <= int(caps["max_output_bytes"]):
                            raise IntegrityError("arm worker response length is outside the cap")
                    if expected_payload_bytes is not None:
                        received_payload_bytes = len(framed) - 8
                        if received_payload_bytes > expected_payload_bytes:
                            raise IntegrityError("arm worker sent bytes beyond its declared frame")
                        if received_payload_bytes == expected_payload_bytes and response is None:
                            response = strict_json_bytes(
                                bytes(framed[8:]), f"arm-worker:{spec['arm_id']}"
                            )
            if child_exit_status is None:
                waited_pid, status, usage = os.wait4(child_pid, os.WNOHANG)
                if waited_pid:
                    child_exit_status = status
                    child_exit_ns = time.monotonic_ns()
                    child_peak_rss_bytes = int(usage.ru_maxrss) * 1024
                    if child_peak_rss_bytes > int(caps["max_peak_rss_bytes"]):
                        raise ResourceLimitError("supervised arm terminal peak-RSS cap exceeded")
                else:
                    worker_peak_rss = process_peak_rss_bytes(child_pid)
                    if worker_peak_rss > int(caps["max_peak_rss_bytes"]):
                        raise ResourceLimitError("supervised arm peak-RSS cap exceeded")
            if child_exit_status is not None and os.waitstatus_to_exitcode(child_exit_status) != 0:
                raise IntegrityError(
                    f"arm worker nonzero exit: {os.waitstatus_to_exitcode(child_exit_status)}"
                )
        os.set_blocking(read_file_descriptor, True)
        trailing = b"" if pipe_closed else os.read(read_file_descriptor, 65536)
        if trailing:
            raise IntegrityError("arm worker sent trailing bytes after its response frame")
        if deadline_expired(
            time.monotonic_ns(), min(execution_deadline_ns, arm_deadline_ns)
        ):
            raise ResourceLimitError("arm worker result reached parent at or after deadline")
        if child_exit_ns is None or child_peak_rss_bytes <= 0:
            raise IntegrityError("arm worker terminal resource usage was not captured")
    except BaseException:
        if child_exit_status is None:
            _kill_and_reap_worker(child_pid)
        raise
    finally:
        os.close(read_file_descriptor)
        if write_descriptor_open:
            os.close(write_file_descriptor)
    if response.get("ok") is not True:
        if set(response) != {"ok", "kind", "message"}:
            raise IntegrityError("malformed arm-worker failure response")
        if response["kind"] == "ResourceLimitError":
            raise ResourceLimitError(response["message"])
        if response["kind"] == "IntegrityError":
            raise IntegrityError(response["message"])
        raise IntegrityError(
            f"unexpected arm-worker failure {response['kind']}: {response['message']}"
        )
    if set(response) != {"ok", "semantic", "provenance"}:
        raise IntegrityError("malformed arm-worker success response")
    semantic = response["semantic"]
    provenance = response["provenance"]
    provenance["elapsed_seconds"] = elapsed_seconds(arm_started_ns, child_exit_ns)
    provenance["peak_rss_bytes"] = max(
        int(provenance.get("peak_rss_bytes", 0)), child_peak_rss_bytes
    )
    if (
        provenance.get("arm_id") != spec["arm_id"]
        or float(provenance.get("elapsed_seconds", math.inf))
        > float(caps["max_wall_seconds_per_arm"])
        or int(provenance.get("peak_rss_bytes", 0)) <= 0
        or int(provenance.get("peak_rss_bytes", 0)) > int(caps["max_peak_rss_bytes"])
    ):
        raise ResourceLimitError("completed arm exceeded a registered resource cap")
    enforce_resources(contract, output_dir, execution_started_ns)
    return semantic, provenance


def empirical_w1(left: list[float], right: list[float]) -> float:
    if not left or not right or not all(math.isfinite(value) for value in left + right):
        raise IntegrityError("W1 requires nonempty finite populations")
    x = sorted(left)
    y = sorted(right)
    i = j = 0
    remaining_x = 1.0 / len(x)
    remaining_y = 1.0 / len(y)
    cost = 0.0
    tolerance = 4.0 * FLOAT_EPSILON
    while i < len(x) and j < len(y):
        weight = min(remaining_x, remaining_y)
        cost += weight * abs(x[i] - y[j])
        remaining_x -= weight
        remaining_y -= weight
        if remaining_x <= tolerance:
            i += 1
            remaining_x = 1.0 / len(x) if i < len(x) else 0.0
        if remaining_y <= tolerance:
            j += 1
            remaining_y = 1.0 / len(y) if j < len(y) else 0.0
    if i != len(x) or j != len(y):
        raise IntegrityError("W1 mass accounting failed")
    return cost


def metric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = [row[key] for row in rows]
    if any(value is None or not math.isfinite(float(value)) for value in values):
        raise IntegrityError(f"metric population is incomplete: {key}")
    return [float(value) for value in values]


def arm_effect(source: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    left = source["particle_metrics"]
    right = control["particle_metrics"]
    if [row["logical_id"] for row in left] != [row["logical_id"] for row in right]:
        raise IntegrityError("paired particle identities differ")
    result = {"configuration_id": source["arm_id"]}
    for name, field in (
        ("q_below_30", "q_below_30_hit_count"),
        ("q_below_35", "q_below_35_hit_count"),
        ("final_finite_bound", "final_finite_bound_count"),
    ):
        numerator = int(source["summary"][field]) - int(control["summary"][field])
        result[f"{name}_effect_numerator"] = numerator
        result[f"{name}_effect_denominator"] = 64
        result[f"{name}_effect"] = numerator / 64.0
    result["w1_minimum_sampled_q_AU"] = empirical_w1(
        metric_values(left, "minimum_sampled_q_AU"), metric_values(right, "minimum_sampled_q_AU")
    )
    result["w1_final_q_AU"] = empirical_w1(metric_values(left, "final_q_AU"), metric_values(right, "final_q_AU"))
    result["w1_final_i_deg"] = empirical_w1(metric_values(left, "final_i_deg"), metric_values(right, "final_i_deg"))
    return result


def mixture_analysis(primary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    control = primary["M0"]
    source_ids = list(PRIMARY_ARM_IDS[1:])
    configuration_effects = [arm_effect(primary[arm_id], control) for arm_id in source_ids]
    mixture: dict[str, Any] = {}
    for name, field in (
        ("q_below_30", "q_below_30_hit_count"),
        ("q_below_35", "q_below_35_hit_count"),
        ("final_finite_bound", "final_finite_bound_count"),
    ):
        numerator = sum(int(primary[arm_id]["summary"][field]) for arm_id in source_ids) - 6 * int(control["summary"][field])
        mixture[f"{name}_effect_numerator"] = numerator
        mixture[f"{name}_effect_denominator"] = 384
        mixture[f"{name}_effect"] = numerator / 384.0
    for output_name, field in (
        ("w1_minimum_sampled_q_AU", "minimum_sampled_q_AU"),
        ("w1_final_q_AU", "final_q_AU"),
        ("w1_final_i_deg", "final_i_deg"),
    ):
        source_values = [
            value for arm_id in source_ids for value in metric_values(primary[arm_id]["particle_metrics"], field)
        ]
        control_values = metric_values(control["particle_metrics"], field) * 6
        mixture[output_name] = empirical_w1(source_values, control_values)
    blocks = []
    for block in range(4):
        start = block * 16
        stop = start + 16
        row: dict[str, Any] = {"block_index": block}
        for name, metric in (
            ("q_below_30", "minimum_sampled_q_AU"),
            ("q_below_35", "minimum_sampled_q_AU"),
        ):
            threshold = 30.0 if name.endswith("30") else 35.0
            source_hits = sum(
                primary[arm_id]["particle_metrics"][index][metric] < threshold
                for arm_id in source_ids for index in range(start, stop)
            )
            control_hits = sum(
                control["particle_metrics"][index][metric] < threshold for index in range(start, stop)
            )
            numerator = source_hits - 6 * control_hits
            row[f"{name}_effect_numerator"] = numerator
            row[f"{name}_effect_denominator"] = 96
            row[f"{name}_effect"] = numerator / 96.0
        blocks.append(row)
    return {
        "configuration_effects": configuration_effects,
        "mixture_effects": mixture,
        "block_effects": blocks,
    }


def timestep_comparison(audit: dict[str, Any], primary: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    if audit["primary_arm_id"] != primary["arm_id"]:
        raise IntegrityError("audit-to-primary pairing changed")
    left, right = audit["particle_metrics"], primary["particle_metrics"]
    if [row["logical_id"] for row in left] != [row["logical_id"] for row in right]:
        raise IntegrityError("audit particle identities changed")
    differences = {
        "q_below_30_fraction_difference": abs(audit["summary"]["q_below_30_fraction"] - primary["summary"]["q_below_30_fraction"]),
        "q_below_35_fraction_difference": abs(audit["summary"]["q_below_35_fraction"] - primary["summary"]["q_below_35_fraction"]),
        "final_finite_bound_fraction_difference": abs(audit["summary"]["final_finite_bound_fraction"] - primary["summary"]["final_finite_bound_fraction"]),
        "w1_minimum_sampled_q_AU": empirical_w1(metric_values(left, "minimum_sampled_q_AU"), metric_values(right, "minimum_sampled_q_AU")),
        "w1_final_q_AU": empirical_w1(metric_values(left, "final_q_AU"), metric_values(right, "final_q_AU")),
        "w1_final_i_deg": empirical_w1(metric_values(left, "final_i_deg"), metric_values(right, "final_i_deg")),
    }
    checks = {
        "q_below_30_fraction_within_gate": differences["q_below_30_fraction_difference"] <= gates["max_dt_half_q_below_30_fraction_difference"],
        "q_below_35_fraction_within_gate": differences["q_below_35_fraction_difference"] <= gates["max_dt_half_q_below_35_fraction_difference"],
        "final_finite_bound_fraction_within_gate": differences["final_finite_bound_fraction_difference"] <= gates["max_dt_half_final_finite_bound_fraction_difference"],
        "w1_minimum_sampled_q_within_gate": differences["w1_minimum_sampled_q_AU"] <= gates["max_dt_half_w1_minimum_sampled_q_AU"],
        "w1_final_q_within_gate": differences["w1_final_q_AU"] <= gates["max_dt_half_w1_final_q_AU"],
        "w1_final_i_within_gate": differences["w1_final_i_deg"] <= gates["max_dt_half_w1_final_i_deg"],
    }
    return {
        "audit_arm_id": audit["arm_id"], "primary_arm_id": primary["arm_id"],
        "differences": differences, "checks": checks,
    }


def raw_classification(analysis: dict[str, Any]) -> str:
    numerator = int(analysis["mixture_effects"]["q_below_30_effect_numerator"])
    block_numerators = [
        int(row["q_below_30_effect_numerator"]) for row in analysis["block_effects"]
    ]
    if numerator >= 20 and all(value >= 1 for value in block_numerators):
        return "DIRECTIONALLY_STABLE_INCREASE"
    if numerator <= -20 and all(value <= -1 for value in block_numerators):
        return "DIRECTIONALLY_STABLE_DECREASE"
    if abs(numerator) <= 7 and max(abs(value) for value in block_numerators) <= 4:
        return "PRACTICALLY_SMALL"
    return "INCONCLUSIVE"


def two_resolution_analysis(
    arm_semantics: dict[str, dict[str, Any]], gates: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], bool, bool]:
    """Compute complete effects only when all required orbit metrics exist."""
    if set(arm_semantics) != set(PRIMARY_ARM_IDS + AUDIT_ARM_IDS):
        raise IntegrityError("two-resolution analysis received the wrong arm inventory")
    metrics_complete = all(
        arm["summary"]["all_particles_have_complete_finite_osculating_history"]
        for arm in arm_semantics.values()
    )
    if not metrics_complete:
        return ({
            "status": ANALYSIS_SUPPRESSED_STATUS,
            "primary_dt": None,
            "audit_dt_half": None,
            "primary_raw_classification": None,
            "audit_raw_classification": None,
        }, [], False, False)
    primary = {arm_id: arm_semantics[arm_id] for arm_id in PRIMARY_ARM_IDS}
    audits = {arm_id: arm_semantics[arm_id] for arm_id in AUDIT_ARM_IDS}
    primary_analysis = mixture_analysis(primary)
    audit_resolution_view = {
        primary_id: {**audits[f"AUDIT-{primary_id}"], "arm_id": primary_id}
        for primary_id in PRIMARY_ARM_IDS
    }
    audit_analysis = mixture_analysis(audit_resolution_view)
    timestep = [
        timestep_comparison(audits[arm_id], primary[AUDIT_PRIMARY[arm_id]], gates)
        for arm_id in AUDIT_ARM_IDS
    ]
    timestep_pass = all(
        value for row in timestep for value in row["checks"].values()
    )
    primary_raw = raw_classification(primary_analysis)
    audit_raw = raw_classification(audit_analysis)
    class_agreement = primary_raw == audit_raw
    return ({
        "status": ANALYSIS_COMPLETE_STATUS,
        "primary_dt": primary_analysis,
        "audit_dt_half": audit_analysis,
        "primary_raw_classification": primary_raw,
        "audit_raw_classification": audit_raw,
    }, timestep, timestep_pass, class_agreement)


def publish_final_result(
    path: Path, result: dict[str, Any], contract: dict[str, Any],
    output_dir: Path, execution_started_ns: int,
) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite final result: {path}")
    caps = contract["resource_caps_per_execution"]
    reserve = float(caps["finalization_wall_reserve_seconds"])
    if time.monotonic_ns() + int(reserve * NANOSECONDS_PER_SECOND) >= deadline_ns(
        execution_started_ns, float(caps["max_wall_seconds_total"])
    ):
        raise ResourceLimitError("insufficient locked wall reserve for final publication")
    payload = serialized_json(result)
    pending = path.with_name(f".{path.name}.pending")
    if pending.exists() or pending.is_symlink():
        raise FileExistsError("stale final-result pending file exists")
    try:
        with pending.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        enforce_resources(contract, output_dir, execution_started_ns)
        os.replace(pending, path)
    except BaseException:
        try:
            pending.unlink()
        except FileNotFoundError:
            pass
        raise


def output_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256(b"jx-xp1-output-tree/v1\0")
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink() or not path.is_file():
            raise IntegrityError("output tree contains a nonregular entry")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def validate_a_for_b(
    a_result_path: Path, a_verification_receipt_path: Path,
    registration_sha256: str, contract_sha256: str, verifier_sha256: str,
    output_b: Path, package_root: Path, contract: dict[str, Any],
) -> None:
    if a_result_path.is_symlink() or a_verification_receipt_path.is_symlink():
        raise ValueError("A result and verification receipt must not be symlinks")
    path = a_result_path.resolve()
    if path.name != "result_v1.json" or not path.is_file() or path.is_symlink():
        raise ValueError("B authorization requires canonical A/result_v1.json")
    output_a = path.parent
    if output_b == output_a or output_b in output_a.parents or output_a in output_b.parents:
        raise ValueError("A and B output directories must be disjoint")
    protected = protected_tree_roots(contract, package_root)
    if any(
        trees_overlap(candidate, protected_root)
        for candidate in (output_a, output_b)
        for protected_root in protected
    ):
        raise ValueError("A or B output overlaps a package or bound-context tree")
    if {item.name for item in output_a.iterdir()} != {"run_manifest.json", "result_v1.json"}:
        raise IntegrityError("A output inventory is not a clean successful execution")
    manifest = strict_json(output_a / "run_manifest.json")
    result = strict_json(path)
    if set(manifest) != {
        "schema", "experiment_id", "contract_sha256", "seed_manifest_sha256",
        "registration_sha256", "runner_sha256", "verifier_sha256",
        "execution_label", "execution_instance_id", "runtime",
    } or set(result) != {
        "schema", "experiment_id", "state", "claim_ceiling", "mandatory_nonclaim",
        "semantic_sha256", "semantic", "provenance",
    }:
        raise IntegrityError("A manifest/result shape mismatch")
    if manifest.get("execution_label") != "A":
        raise IntegrityError("A manifest label mismatch")
    if manifest.get("registration_sha256") != registration_sha256 or manifest.get("contract_sha256") != contract_sha256:
        raise IntegrityError("A design binding mismatch")
    if result.get("semantic_sha256") != sha256_bytes(canonical_bytes(result.get("semantic"))):
        raise IntegrityError("A semantic hash does not recompute")
    if result.get("state") not in {"NUMERICALLY_UNRESOLVED", "EXPLORATORY_COMPLETE"}:
        raise IntegrityError("A did not reach a completed semantic result")
    semantic = result["semantic"]
    if (
        semantic.get("schema") != SEMANTIC_SCHEMA
        or semantic.get("experiment_id") != EXPECTED_EXPERIMENT_ID
        or semantic.get("execution_state") != result["state"]
        or semantic.get("claim_ceiling") != result["claim_ceiling"]
        or semantic.get("mandatory_nonclaim") != result["mandatory_nonclaim"]
        or semantic.get("matrix", {}).get("primary_arm_ids") != list(PRIMARY_ARM_IDS)
        or semantic.get("matrix", {}).get("audit_arm_ids") != list(AUDIT_ARM_IDS)
        or set(semantic.get("arms", {})) != set(PRIMARY_ARM_IDS + AUDIT_ARM_IDS)
        or semantic.get("gate_summary", {}).get("integrity_pass") is not True
    ):
        raise IntegrityError("A semantic identity/matrix/gate binding mismatch")
    receipt_path = a_verification_receipt_path.resolve()
    if (
        receipt_path.is_relative_to(output_a)
        or receipt_path.is_relative_to(output_b)
        or receipt_path.is_relative_to(package_root)
        or any(trees_overlap(receipt_path, protected_root) for protected_root in protected)
    ):
        raise ValueError("A verification receipt must be outside package and execution trees")
    receipt = strict_json(receipt_path)
    if set(receipt) != {
        "schema", "experiment_id", "state", "contract_sha256",
        "seed_manifest_sha256", "registration_sha256", "verifier_sha256",
        "execution_a", "independent_recomputation", "mandatory_nonclaim",
    }:
        raise IntegrityError("A verification receipt shape mismatch")
    expected_execution = {
        "semantic_sha256": result["semantic_sha256"],
        "result_sha256": sha256_file(path),
        "output_tree_sha256": output_tree_sha256(output_a),
        "execution_instance_id": manifest["execution_instance_id"],
    }
    if (
        receipt["schema"] != "jx-xp1-a-verification-receipt/v1"
        or receipt["experiment_id"] != EXPECTED_EXPERIMENT_ID
        or receipt["state"] != "XP1_A_OUTPUT_VERIFIED_FOR_B"
        or receipt["contract_sha256"] != contract_sha256
        or receipt["seed_manifest_sha256"] != EXPECTED_SEED_MANIFEST_SHA256
        or receipt["registration_sha256"] != registration_sha256
        or receipt["verifier_sha256"] != verifier_sha256
        or receipt["execution_a"] != expected_execution
        or receipt["mandatory_nonclaim"] != result["mandatory_nonclaim"]
        or receipt["independent_recomputation"]
        != {key: True for key in INDEPENDENT_RECOMPUTATION_KEYS}
    ):
        raise IntegrityError("A verification receipt content mismatch")


def execute(
    contract_path: Path, seed_manifest_path: Path, registration_path: Path,
    output_dir: Path, execution_label: str, a_result: Path | None,
    a_verification_receipt: Path | None,
    command_started_ns: int,
) -> dict[str, Any]:
    if contract_path.is_symlink() or seed_manifest_path.is_symlink() or registration_path.is_symlink():
        raise ValueError("canonical execution inputs must not be symlinks")
    contract_path = contract_path.resolve()
    seed_manifest_path = seed_manifest_path.resolve()
    registration_path = registration_path.resolve()
    runner_path = Path(__file__).resolve()
    root = runner_path.parent
    if contract_path != root / "contract_v1.json" or seed_manifest_path != root / "seed_manifest_v1.json":
        raise ValueError("contract and seed manifest must be canonical XP1 package files")
    contract = strict_json(contract_path)
    validate_contract(contract, contract_path)
    registration, registration_sha256 = validate_registration(registration_path, contract_path, runner_path)
    _, seeds = validate_seed_manifest(contract, seed_manifest_path)
    tracers, _, tracer_digest = make_tracers(contract, seeds)
    if execution_label not in contract["result_policy"]["clean_execution_labels"]:
        raise ValueError("execution label is not predeclared")
    output_dir = validate_clean_output_directory(output_dir, contract, root)
    contract_sha256 = sha256_file(contract_path)
    if execution_label == "B":
        if a_result is None or a_verification_receipt is None:
            raise ValueError("execution B requires --a-result and --a-verification-receipt")
        validate_a_for_b(
            a_result, a_verification_receipt, registration_sha256, contract_sha256,
            registration["locked_files"]["verify_replay.py"], output_dir, root, contract,
        )
    elif a_result is not None or a_verification_receipt is not None:
        raise ValueError("execution A must not receive A-result verification inputs")
    runtime = dict(contract["runtime_lock"])
    instance_id = sha256_bytes(
        b"jx-xp1-execution-instance/v1\0"
        + bytes.fromhex(registration_sha256)
        + execution_label.encode("ascii")
    )[:32]
    manifest = {
        "schema": MANIFEST_SCHEMA, "experiment_id": EXPECTED_EXPERIMENT_ID,
        "contract_sha256": contract_sha256,
        "seed_manifest_sha256": sha256_file(seed_manifest_path),
        "registration_sha256": registration_sha256,
        "runner_sha256": registration["locked_files"]["run_exploratory.py"],
        "verifier_sha256": registration["locked_files"]["verify_replay.py"],
        "execution_label": execution_label, "execution_instance_id": instance_id,
        "runtime": runtime,
    }
    completed: list[str] = []
    arm_semantics: dict[str, dict[str, Any]] = {}
    arm_provenance: list[dict[str, Any]] = []
    output_created = False
    try:
        output_dir.mkdir()
        output_created = True
        atomic_json(output_dir / "run_manifest.json", manifest)
        enforce_resources(contract, output_dir, command_started_ns)
        if validate_runtime(contract) != runtime:
            raise IntegrityError("validated runtime differs from manifest runtime lock")
        enforce_resources(contract, output_dir, command_started_ns)
        preflight = preflight_initial_states(contract, tracers)
        enforce_resources(contract, output_dir, command_started_ns)
        for index, spec in enumerate(arm_specifications(contract), start=1):
            semantic, provenance = run_arm_supervised(
                contract, tracers, spec, preflight[spec["arm_id"]], output_dir,
                command_started_ns,
            )
            arm_semantics[spec["arm_id"]] = semantic
            arm_provenance.append(provenance)
            completed.append(spec["arm_id"])
            print(f"[arm {index:02d}/14] {spec['arm_id']} complete", flush=True)
        integrity_keys = {
            "sample_count_exact", "final_time_exact", "particle_count_unchanged",
            "active_particle_count_unchanged", "integrator_and_dt_readback_exact",
            "r_crit_hill_and_safe_mode_readback_exact", "pre_translation_common_state_exact",
            "post_com_common_state_within_epsilon_gate", "all_samples_cartesian_finite",
        }
        numerical_keys = {
            "all_particles_have_complete_finite_osculating_history",
            "active_energy_drift_within_gate",
            "active_com_angular_momentum_drift_within_gate",
            "active_linear_momentum_residual_within_gate",
        }
        integrity_pass = all(
            arm["checks"][key] for arm in arm_semantics.values() for key in integrity_keys
        )
        if not integrity_pass:
            raise IntegrityError("one or more arm integrity checks failed")
        conservation_pass = all(
            arm["checks"][key] for arm in arm_semantics.values() for key in numerical_keys
        )
        analysis, timestep, timestep_pass, class_agreement = two_resolution_analysis(
            arm_semantics, contract["numerical_gates"]
        )
        numerical_pass = conservation_pass and timestep_pass and class_agreement
        state = "EXPLORATORY_COMPLETE" if numerical_pass else "NUMERICALLY_UNRESOLVED"
        classification = (
            analysis["primary_raw_classification"]
            if numerical_pass else "SUPPRESSED_NUMERICALLY_UNRESOLVED"
        )
        semantic = {
            "schema": SEMANTIC_SCHEMA, "experiment_id": EXPECTED_EXPERIMENT_ID,
            "claim_ceiling": contract["claim_ceiling"],
            "mandatory_nonclaim": contract["mandatory_nonclaim"],
            "design_bindings": {
                "contract_sha256": contract_sha256,
                "seed_manifest_sha256": sha256_file(seed_manifest_path),
                "registration_sha256": registration_sha256,
                "tracer_rows_sha256": tracer_digest,
                "runner_sha256": registration["locked_files"]["run_exploratory.py"],
                "verifier_sha256": registration["locked_files"]["verify_replay.py"],
                "test_sha256": registration["locked_files"]["test_exploratory.py"],
            },
            "matrix": {
                "primary_arm_ids": list(PRIMARY_ARM_IDS),
                "audit_arm_ids": list(AUDIT_ARM_IDS),
                "tracer_count_in_every_arm": 64,
                "analysis_block_count": 4,
                "samples_including_t0": 5001,
            },
            "arms": arm_semantics,
            "analysis": analysis,
            "timestep_all_arm_comparisons": timestep,
            "gate_summary": {
                "integrity_pass": integrity_pass,
                "conservation_and_finite_orbit_history_pass": conservation_pass,
                "all_seven_timestep_pairs_pass": timestep_pass,
                "primary_and_audit_raw_classifications_exact": class_agreement,
                "all_numerical_gates_pass": numerical_pass,
                "timestep_scope": contract["numerical_gates"]["timestep_scope"],
            },
            "exploratory_classification": classification,
            "execution_state": state,
        }
        semantic_sha256 = sha256_bytes(canonical_bytes(semantic))
        provenance = {
            "execution_label": execution_label,
            "execution_instance_id": instance_id,
            "runtime": runtime,
            "arm_records": arm_provenance,
            "elapsed_seconds": elapsed_seconds(command_started_ns),
            "peak_rss_bytes": max(
                [peak_rss_bytes()]
                + [int(row["peak_rss_bytes"]) for row in arm_provenance]
            ),
            "output_bytes_before_result": directory_bytes(output_dir),
        }
        result = {
            "schema": RESULT_SCHEMA, "experiment_id": EXPECTED_EXPERIMENT_ID,
            "state": state, "claim_ceiling": contract["claim_ceiling"],
            "mandatory_nonclaim": contract["mandatory_nonclaim"],
            "semantic_sha256": semantic_sha256, "semantic": semantic,
            "provenance": provenance,
        }
        projected = directory_bytes(output_dir) + len(serialized_json(result))
        if projected > int(contract["resource_caps_per_execution"]["max_output_bytes"]):
            raise ResourceLimitError("final result would exceed output cap")
        enforce_resources(contract, output_dir, command_started_ns)
        publish_final_result(
            output_dir / "result_v1.json", result, contract, output_dir,
            command_started_ns,
        )
        return result
    except BaseException as exc:
        if output_created:
            manifest_path = output_dir / "run_manifest.json"
            failure_path = output_dir / "failure_receipt.json"
            try:
                if not manifest_path.exists() and not manifest_path.is_symlink():
                    atomic_json(manifest_path, manifest)
                if not failure_path.exists() and not (output_dir / "result_v1.json").exists():
                    state = (
                        "RESOURCE_LIMIT" if isinstance(exc, ResourceLimitError)
                        else "INVALID_INTEGRITY"
                    )
                    atomic_json(failure_path, {
                        "schema": FAILURE_SCHEMA, "experiment_id": EXPECTED_EXPERIMENT_ID,
                        "state": state, "execution_label": execution_label,
                        "contract_sha256": contract_sha256,
                        "registration_sha256": registration_sha256,
                        "exception_type": type(exc).__name__,
                        "message": REDACTED_FAILURE_MESSAGE,
                        "completed_arm_ids": completed,
                        "elapsed_seconds": elapsed_seconds(command_started_ns),
                        "mandatory_nonclaim": contract["mandatory_nonclaim"],
                    })
            except BaseException as receipt_error:
                raise IntegrityError(
                    f"execution failed and failure receipt could not be finalized: {receipt_error}"
                ) from exc
        raise


def validate_only(contract_path: Path, seed_manifest_path: Path, registration_path: Path) -> dict[str, Any]:
    if contract_path.is_symlink() or seed_manifest_path.is_symlink() or registration_path.is_symlink():
        raise ValueError("canonical validation inputs must not be symlinks")
    contract_path = contract_path.resolve()
    seed_manifest_path = seed_manifest_path.resolve()
    runner_path = Path(__file__).resolve()
    root = runner_path.parent
    if contract_path != root / "contract_v1.json" or seed_manifest_path != root / "seed_manifest_v1.json":
        raise ValueError("validation inputs must be canonical package files")
    contract = strict_json(contract_path)
    validate_contract(contract, contract_path)
    registration, registration_sha256 = validate_registration(
        registration_path.resolve(), contract_path, runner_path
    )
    _, seeds = validate_seed_manifest(contract, seed_manifest_path)
    tracers, _, tracer_digest = make_tracers(contract, seeds)
    runtime = validate_runtime(contract)
    initial = preflight_initial_states(contract, tracers)
    return {
        "contract_sha256": sha256_file(contract_path),
        "registration_sha256": registration_sha256,
        "seed_manifest_sha256": sha256_file(seed_manifest_path),
        "tracer_rows_sha256": tracer_digest,
        "runtime": runtime,
        "registration_runner_sha256": registration["locked_files"]["run_exploratory.py"],
        "preflight_arm_count": len(initial),
        "pre_translation_common_relative_state_sha256": initial["M0"][
            "pre_translation_common_relative_state_sha256"
        ],
        "maximum_post_com_binary64_epsilon_units": max(
            row["post_com_max_binary64_epsilon_units"] for row in initial.values()
        ),
        "dynamics_executed": False,
    }


def main() -> int:
    command_started_ns = time.monotonic_ns()
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execution-label", choices=("A", "B"), default="A")
    parser.add_argument("--a-result", type=Path)
    parser.add_argument("--a-verification-receipt", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    if arguments.validate_only:
        if (
            arguments.output_dir is not None
            or arguments.a_result is not None
            or arguments.a_verification_receipt is not None
            or arguments.execution_label != "A"
        ):
            raise ValueError("validate-only does not accept execution/output arguments")
        print(json.dumps(validate_only(
            arguments.contract, arguments.seed_manifest, arguments.registration
        ), indent=2, sort_keys=True, allow_nan=False))
        return 0
    if arguments.output_dir is None:
        parser.error("--output-dir is required for execution")
    result = execute(
        arguments.contract, arguments.seed_manifest, arguments.registration,
        arguments.output_dir, arguments.execution_label, arguments.a_result,
        arguments.a_verification_receipt,
        command_started_ns,
    )
    print(json.dumps({
        "state": result["state"], "semantic_sha256": result["semantic_sha256"],
        "result_file": "result_v1.json",
    }, indent=2, sort_keys=True))
    return 0 if result["state"] == "EXPLORATORY_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
