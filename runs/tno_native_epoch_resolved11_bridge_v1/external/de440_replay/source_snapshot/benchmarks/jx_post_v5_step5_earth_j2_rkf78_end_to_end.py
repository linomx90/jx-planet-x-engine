#!/usr/bin/env python3
"""One-day real-DE440 end-to-end exercise of the Step 5 J2 RKF78 adapter.

This narrow post-V5 probe crosses the retained CSPICE execution boundary,
prepares the resolved-eleven Newtonian inputs, and integrates three otherwise
matched lanes: Newtonian control, fixed-+Z Earth--Moon J2, and the DE440
long-term dynamic Earth-pole prescription.  It is deterministic MODEL_OUTPUT,
not an ephemeris or integrator qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import struct
import sys
from typing import Any

import numpy as np

from jxplanetx.engine import trajectory as engine_trajectory
from jxplanetx.engine.trajectory import TrajectoryCheckpoint
from jxplanetx.engine.trajectory_contracts import AdaptiveRKF78Spec
from jxplanetx.solar_system import cspice_execution
from jxplanetx.solar_system import engine_preparation
from jxplanetx.solar_system import engine_preparation_contracts
from jxplanetx.solar_system import earth_j2
from jxplanetx.solar_system import earth_j2_trajectory
from jxplanetx.solar_system import earth_pole
from tests import test_solar_system_engine_preparation as retained_fixture


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "jx.post-v5.step5-earth-j2-rkf78-end-to-end.v1"
BENCHMARK_ID = "jx.post-v5.step5.earth-j2-rkf78-end-to-end-occam.v1"
SEMANTIC_DOMAIN = b"jx.post-v5.step5-earth-j2-rkf78-end-to-end.semantic.v1\0"
MATRIX_DOMAIN = b"jx.post-v5.step5-earth-j2-rkf78-end-to-end.matrix.v1\0"
STEP_DOMAIN = b"jx.post-v5.step5-earth-j2-rkf78-end-to-end.steps.v1\0"
NUMERICAL_KAT_DOMAIN = b"jx.post-v5.step5-earth-j2-rkf78-end-to-end.kat.v1\0"

SECONDS_PER_DAY = 86_400
START_ET = 0
END_ET = SECONDS_PER_DAY
MAXIMUM_STEP = 900.0
POSITION_ATOL = 1.0e-3
VELOCITY_ATOL = 1.0e-9
RELATIVE_TOLERANCE = 1.0e-13
MAXIMUM_STEPS = 10_000
MAXIMUM_REJECTIONS = 2_000
EXPECTED_ACCEPTED_STEPS = 96
MAXIMUM_REPORT_BYTES = 4 << 20
EXPECTED_NUMERICAL_KAT_SHA256 = (
    "2f9a5e3674c4d9516334e9bc8b165b9c95c7f3d39d3dae530a7ecac430af6841"
)

BODY_IDS = (
    "SUN",
    "MERCURY",
    "VENUS",
    "EARTH",
    "MOON",
    "MARS_BARYCENTER",
    "JUPITER_BARYCENTER",
    "SATURN_BARYCENTER",
    "URANUS_BARYCENTER",
    "NEPTUNE_BARYCENTER",
    "PLUTO_BARYCENTER",
)
EARTH_INDEX = 3
MOON_INDEX = 4

ARTIFACTS = {
    "artifact-root/kernels/de440s.bsp": (
        32_726_016,
        "c1c7feeab882263fc493a9d5a5b2ddd71b54826cdf65d8d17a76126b260a49f2",
    ),
    "artifact-root/licenses/naif-spice-rules.html": (
        23_487,
        "ae85f851646e7c4f0a762db852907bc090a2ab50c815eb2a6cd8639e96b7e047",
    ),
    "artifact-root/licenses/numpy-2.3.5-LICENSE.txt": (
        47_768,
        "2046a3130e50b11c01659b3a0d963e6ae0b7436ff8e89cbcfd9e87bc6112d595",
    ),
    "artifact-root/licenses/spiceypy-cspice-composite-license-provenance.v1.json": (
        42_166,
        "285cdb7a837de375dbaecf7578949449262bc5ecb1625e8f4972e5dd82233a9b",
    ),
    "artifact-root/native/libcspice.so": (
        3_561_056,
        "1d9273fc9afce5201e904569a9439a0b010d2461dfc88c0de549d2a1b5ecdf84",
    ),
    "artifact-root/wheels/numpy-2.3.5-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl": (
        16_597_350,
        "fffe29a1ef00883599d1dc2c51aa2e5d80afe49523c261a74933df395c15c520",
    ),
    "artifact-root/wheels/spiceypy-8.2.0-cp314-cp314-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl": (
        2_442_346,
        "00c2714d3dd67180794ad0b66735d6cac470e19627d6e99f9628a50b52133d90",
    ),
    "parameter-root/constants/gm_de440.tpc": (
        12_406,
        "924ddf4fb9ead9fe8a1aa55780bcabde40b09d00065d58226e24b68d8092f140",
    ),
    "parameter-root/licenses/naif-spice-rules.html": (
        23_487,
        "ae85f851646e7c4f0a762db852907bc090a2ab50c815eb2a6cd8639e96b7e047",
    ),
}

SOURCE_PATHS = (
    "benchmarks/jx_post_v5_step5_earth_j2_rkf78_end_to_end.py",
    "src/jxplanetx/engine/trajectory.py",
    "src/jxplanetx/engine/rkf78.py",
    "src/jxplanetx/solar_system/cspice_execution.py",
    "src/jxplanetx/solar_system/engine_preparation.py",
    "src/jxplanetx/solar_system/earth_j2.py",
    "src/jxplanetx/solar_system/earth_j2_trajectory.py",
    "src/jxplanetx/solar_system/earth_pole.py",
    "tests/test_solar_system_engine_preparation.py",
    "tests/test_solar_system_cspice_execution.py",
)

CLAIM_CONTROLS = {
    "complete_de440_force_model_claimed": False,
    "eih_1pn_included": False,
    "ephemeris_qualification_claimed": False,
    "general_accuracy_claimed": False,
    "integrator_qualification_claimed": False,
    "long_term_accuracy_claimed": False,
    "performance_or_superiority_claimed": False,
    "production_use_authorized": False,
    "rebound_superiority_claimed": False,
    "frozen_v5_or_kdk_modified": False,
}


class EndToEndProbeError(ValueError):
    """The narrow end-to-end probe or its evidence violated its contract."""


def _sha256_path(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _identity(path: Path, locator: str) -> dict[str, Any]:
    size, sha256 = _sha256_path(path)
    return {"logical_locator": locator, "size_bytes": size, "sha256": sha256}


def _validate_roots(artifact_root: Path, parameter_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for locator, expected in sorted(ARTIFACTS.items()):
        prefix, relative = locator.split("/", 1)
        root = artifact_root if prefix == "artifact-root" else parameter_root
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise EndToEndProbeError(f"required retained artifact is unavailable: {locator}")
        observed = _sha256_path(path)
        if observed != expected:
            raise EndToEndProbeError(f"retained artifact identity changed: {locator}")
        records.append(
            {"logical_locator": locator, "size_bytes": observed[0], "sha256": observed[1]}
        )
    return records


def _source_records() -> list[dict[str, Any]]:
    records = []
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise EndToEndProbeError(f"source dependency is unavailable: {relative}")
        records.append(_identity(path, relative))
    return records


def _runtime_record() -> dict[str, Any]:
    try:
        import erfa
        from numpy._core import _multiarray_umath
    except (ImportError, OSError) as exc:
        raise EndToEndProbeError("the retained NumPy/PyERFA runtime is unavailable") from exc
    executable = Path(sys.executable).resolve()
    runtime_files = (
        (executable, "python-executable"),
        (Path(np.__file__).resolve(), "numpy-package-init"),
        (Path(_multiarray_umath.__file__).resolve(), "numpy-multiarray-native"),
        (Path(erfa.__file__).resolve(), "pyerfa-package-init"),
        (Path(erfa.ufunc.__file__).resolve(), "pyerfa-native-ufunc"),
    )
    return {
        "implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pyerfa_version": erfa.__version__,
        "files": [_identity(path, locator) for path, locator in runtime_files],
    }


def canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def strict_json_bytes(data: bytes, maximum_bytes: int = MAXIMUM_REPORT_BYTES) -> dict[str, Any]:
    if type(data) is not bytes or not data or len(data) > maximum_bytes:
        raise EndToEndProbeError("report bytes are empty or exceed the fixed cap")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise EndToEndProbeError("duplicate JSON object key")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("ascii"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                EndToEndProbeError(f"invalid JSON numeric token: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        if isinstance(exc, EndToEndProbeError):
            raise
        raise EndToEndProbeError("report is not strict ASCII JSON") from exc
    if type(value) is not dict or canonical(value) != data:
        raise EndToEndProbeError("report is not canonical JSON")
    return value


def _matrix_digest(matrix: np.ndarray, columns: str) -> str:
    exact = np.ascontiguousarray(matrix, dtype="<f8")
    digest = hashlib.sha256(MATRIX_DOMAIN)
    digest.update(struct.pack(">Q", exact.ndim))
    for dimension in exact.shape:
        digest.update(struct.pack(">Q", dimension))
    encoded_columns = columns.encode("ascii")
    digest.update(struct.pack(">Q", len(encoded_columns)))
    digest.update(encoded_columns)
    digest.update(exact.tobytes(order="C"))
    return digest.hexdigest()


def _matrix_evidence(matrix: np.ndarray, columns: str) -> dict[str, Any]:
    exact = np.ascontiguousarray(matrix, dtype="<f8")
    if exact.ndim != 2 or not np.all(np.isfinite(exact)):
        raise EndToEndProbeError("matrix evidence must be a finite rank-two array")
    return {
        "algorithm": "SHA256_DOMAIN_SHAPE_COLUMNS_LITTLE_ENDIAN_FLOAT64_V1",
        "columns": columns,
        "dtype": "<f8",
        "matrix_hex": [[float(item).hex() for item in row] for row in exact.tolist()],
        "sha256": _matrix_digest(exact, columns),
        "shape": list(exact.shape),
    }


def _decode_matrix(value: object, columns: str, shape: tuple[int, int]) -> np.ndarray:
    if type(value) is not dict or set(value) != {
        "algorithm", "columns", "dtype", "matrix_hex", "sha256", "shape"
    }:
        raise EndToEndProbeError("matrix evidence schema changed")
    if (
        value["algorithm"] != "SHA256_DOMAIN_SHAPE_COLUMNS_LITTLE_ENDIAN_FLOAT64_V1"
        or value["columns"] != columns
        or value["dtype"] != "<f8"
        or value["shape"] != list(shape)
        or type(value["matrix_hex"]) is not list
    ):
        raise EndToEndProbeError("matrix evidence contract changed")
    rows = value["matrix_hex"]
    if any(type(row) is not list for row in rows):
        raise EndToEndProbeError("matrix evidence rows must be exact lists")
    try:
        decoded_rows = []
        for row in rows:
            decoded_row = []
            for item in row:
                if type(item) is not str:
                    raise EndToEndProbeError("matrix cells must be canonical float hex strings")
                number = float.fromhex(item)
                if number.hex() != item:
                    raise EndToEndProbeError("matrix cell is not canonical float hex")
                decoded_row.append(number)
            decoded_rows.append(decoded_row)
        matrix = np.ascontiguousarray(decoded_rows, dtype="<f8")
    except (TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, EndToEndProbeError):
            raise
        raise EndToEndProbeError("matrix evidence contains invalid float hex") from exc
    if matrix.shape != shape or not np.all(np.isfinite(matrix)):
        raise EndToEndProbeError("matrix evidence shape or finiteness changed")
    if value["sha256"] != _matrix_digest(matrix, columns):
        raise EndToEndProbeError("matrix evidence digest changed")
    return matrix


def _step_digest(values: tuple[float, ...]) -> str:
    digest = hashlib.sha256(STEP_DOMAIN)
    digest.update(struct.pack(">Q", len(values)))
    for value in values:
        digest.update(struct.pack(">d", value))
    return digest.hexdigest()


def _execute_at(artifact_root: Path, absolute_et: int, label: str) -> Any:
    projection = retained_fixture.live_fixtures.projection_at(
        "cp314-cp314",
        targets=engine_preparation_contracts._BODY_NAIF_IDS,
        whole=absolute_et,
        fraction=0.0,
    )
    if projection.binary64_projection.rounded_value.hex() != float(absolute_et).hex():
        raise EndToEndProbeError("DE440 query did not project to the exact requested ET")
    root_fd = os.open(artifact_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        return cspice_execution.execute_cspice_spkgeo_replay(
            projection,
            artifact_root_directory_fd=root_fd,
            receipt_id=f"receipt.post-v5.step5.j2-rkf78-e2e.{label}",
            primary_batch_id=f"batch.post-v5.step5.j2-rkf78-e2e.{label}.primary",
            replay_batch_id=f"batch.post-v5.step5.j2-rkf78-e2e.{label}.replay",
            primary_execution_instance_nonce=hashlib.sha256(
                f"j2-rkf78-e2e.{label}.primary".encode("ascii")
            ).hexdigest(),
            replay_execution_instance_nonce=hashlib.sha256(
                f"j2-rkf78-e2e.{label}.replay".encode("ascii")
            ).hexdigest(),
        )
    finally:
        os.close(root_fd)


def _receipt_values(receipt: Any, model_spec: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    primary = engine_preparation_contracts._lane_preparation_values(
        receipt.primary_lane, model_spec
    )
    replay = engine_preparation_contracts._lane_preparation_values(
        receipt.replay_lane, model_spec
    )
    keys = (
        "source_payload_sha256",
        "converted_payload_sha256",
        "recentered_payload_sha256",
        "gm_payload_sha256",
    )
    if any(primary[key] != replay[key] for key in keys):
        raise EndToEndProbeError("CSPICE primary and replay state projections differ")
    positions = np.ascontiguousarray(
        np.asarray(primary["positions"], dtype=np.float64).reshape(len(BODY_IDS), 3)
    )
    velocities = np.ascontiguousarray(
        np.asarray(primary["velocities"], dtype=np.float64).reshape(len(BODY_IDS), 3)
    )
    return positions, velocities, {key: primary[key] for key in keys}


def _integration_matrix(result: earth_j2_trajectory.EarthJ2RKF78Result) -> np.ndarray:
    rows = []
    for checkpoint in result.checkpoints:
        rows.append(
            np.concatenate(
                (
                    np.asarray([checkpoint.epoch], dtype=np.float64),
                    checkpoint.positions.reshape(-1),
                    checkpoint.velocities.reshape(-1),
                )
            )
        )
    return np.ascontiguousarray(rows, dtype=np.float64)


def _result_record(
    lane_id: str,
    result: earth_j2_trajectory.EarthJ2RKF78Result,
    replay: earth_j2_trajectory.EarthJ2RKF78Result,
) -> dict[str, Any]:
    earth_j2_trajectory.validate_earth_j2_rkf78_result(result)
    earth_j2_trajectory.validate_earth_j2_rkf78_result(replay)
    matrix = _integration_matrix(result)
    replay_matrix = _integration_matrix(replay)
    if (
        matrix.tobytes(order="C") != replay_matrix.tobytes(order="C")
        or result.accepted_step_epochs != replay.accepted_step_epochs
        or result.accepted_step_magnitudes != replay.accepted_step_magnitudes
        or result.accepted_step_ledger_content_sha256
        != replay.accepted_step_ledger_content_sha256
        or result.attempted_steps != replay.attempted_steps
        or result.accepted_steps != replay.accepted_steps
        or result.rejected_steps != replay.rejected_steps
    ):
        raise EndToEndProbeError("fresh trajectory replay changed")
    steps = result.accepted_step_magnitudes
    return {
        "lane_id": lane_id,
        "earth_j2_enabled": result.earth_j2_enabled,
        "pole_mode": result.pole_mode,
        "applied_model_ids": list(result.applied_model_ids),
        "raw_state_evidence": _matrix_evidence(
            matrix,
            "elapsed_epoch_seconds,body_major_position_xyz_metres,body_major_velocity_xyz_metres_per_second",
        ),
        "accounting": {
            "accepted_step_hex": [value.hex() for value in steps],
            "accepted_step_sha256": _step_digest(steps),
            "accepted_step_ledger_content_sha256": (
                result.accepted_step_ledger_content_sha256
            ),
            "attempted_steps": result.attempted_steps,
            "accepted_steps": result.accepted_steps,
            "rejected_steps": result.rejected_steps,
            "rkf78_stage_evaluations": result.rkf78_stage_evaluations,
            "baseline_force_evaluations": result.baseline_force_evaluations,
            "earth_j2_correction_evaluations": (
                result.earth_j2_correction_evaluations
            ),
            "checkpoint_counters": [
                [checkpoint.accepted_steps, checkpoint.rejected_steps]
                for checkpoint in result.checkpoints
            ],
        },
        "fresh_replay": {
            "raw_state_sha256": _matrix_digest(
                replay_matrix,
                "elapsed_epoch_seconds,body_major_position_xyz_metres,body_major_velocity_xyz_metres_per_second",
            ),
            "accepted_step_sha256": _step_digest(replay.accepted_step_magnitudes),
            "accepted_step_ledger_content_sha256": (
                replay.accepted_step_ledger_content_sha256
            ),
            "exact": True,
        },
    }


def _errors(candidate: np.ndarray, reference: np.ndarray) -> dict[str, str]:
    candidate_positions = candidate[1:34].reshape(len(BODY_IDS), 3)
    candidate_velocities = candidate[34:67].reshape(len(BODY_IDS), 3)
    reference_positions = reference[1:34].reshape(len(BODY_IDS), 3)
    reference_velocities = reference[34:67].reshape(len(BODY_IDS), 3)
    relative_position_error = (
        candidate_positions[MOON_INDEX]
        - candidate_positions[EARTH_INDEX]
        - (reference_positions[MOON_INDEX] - reference_positions[EARTH_INDEX])
    )
    relative_velocity_error = (
        candidate_velocities[MOON_INDEX]
        - candidate_velocities[EARTH_INDEX]
        - (reference_velocities[MOON_INDEX] - reference_velocities[EARTH_INDEX])
    )
    return {
        "moon_minus_earth_position_error_metres_hex": float(
            np.linalg.norm(relative_position_error)
        ).hex(),
        "moon_minus_earth_velocity_error_metres_per_second_hex": float(
            np.linalg.norm(relative_velocity_error)
        ).hex(),
    }


def _comparisons(lane_matrices: dict[str, np.ndarray], reference: np.ndarray) -> dict[str, Any]:
    errors = {
        lane_id: _errors(matrix[-1], reference[-1])
        for lane_id, matrix in lane_matrices.items()
    }
    control_position = float.fromhex(
        errors["newtonian_control"]["moon_minus_earth_position_error_metres_hex"]
    )
    control_velocity = float.fromhex(
        errors["newtonian_control"][
            "moon_minus_earth_velocity_error_metres_per_second_hex"
        ]
    )
    improvements: dict[str, Any] = {}
    for lane_id in ("fixed_j2000_positive_z", "dynamic_de440_long_term_pole"):
        position = float.fromhex(
            errors[lane_id]["moon_minus_earth_position_error_metres_hex"]
        )
        velocity = float.fromhex(
            errors[lane_id]["moon_minus_earth_velocity_error_metres_per_second_hex"]
        )
        improvements[lane_id] = {
            "position_error_reduction_metres_hex": (control_position - position).hex(),
            "position_improvement_factor_hex": (control_position / position).hex(),
            "velocity_error_reduction_metres_per_second_hex": (
                control_velocity - velocity
            ).hex(),
            "velocity_improvement_factor_hex": (control_velocity / velocity).hex(),
        }
    fixed = lane_matrices["fixed_j2000_positive_z"][-1]
    dynamic = lane_matrices["dynamic_de440_long_term_pole"][-1]
    fixed_positions = fixed[1:34].reshape(len(BODY_IDS), 3)
    dynamic_positions = dynamic[1:34].reshape(len(BODY_IDS), 3)
    fixed_velocities = fixed[34:67].reshape(len(BODY_IDS), 3)
    dynamic_velocities = dynamic[34:67].reshape(len(BODY_IDS), 3)
    return {
        "endpoint_errors": errors,
        "control_relative_improvements": improvements,
        "dynamic_minus_fixed": {
            "moon_minus_earth_position_delta_metres_hex": float(
                np.linalg.norm(
                    (dynamic_positions[MOON_INDEX] - dynamic_positions[EARTH_INDEX])
                    - (fixed_positions[MOON_INDEX] - fixed_positions[EARTH_INDEX])
                )
            ).hex(),
            "moon_minus_earth_velocity_delta_metres_per_second_hex": float(
                np.linalg.norm(
                    (dynamic_velocities[MOON_INDEX] - dynamic_velocities[EARTH_INDEX])
                    - (fixed_velocities[MOON_INDEX] - fixed_velocities[EARTH_INDEX])
                )
            ).hex(),
        },
    }


def _protocol() -> dict[str, Any]:
    return {
        "start_absolute_et_seconds": START_ET,
        "end_absolute_et_seconds": END_ET,
        "elapsed_checkpoint_seconds_hex": [(0.0).hex(), float(END_ET).hex()],
        "maximum_step_seconds_hex": MAXIMUM_STEP.hex(),
        "position_atol_metres_hex": POSITION_ATOL.hex(),
        "velocity_atol_metres_per_second_hex": VELOCITY_ATOL.hex(),
        "relative_tolerance_hex": RELATIVE_TOLERANCE.hex(),
        "maximum_steps": MAXIMUM_STEPS,
        "maximum_rejections": MAXIMUM_REJECTIONS,
        "body_ids": list(BODY_IDS),
        "force_scope": "RESOLVED_ELEVEN_MUTUAL_NEWTONIAN_PLUS_OPTIONAL_EARTH_MOON_J2",
        "parameter_extension_policy": earth_j2_trajectory.PARAMETER_EXTENSION_POLICY,
    }


def _pole_samples() -> dict[str, Any]:
    dynamic = earth_pole.prepare_earth_pole_policy(
        earth_pole.DYNAMIC,
        absolute_start_et=float(START_ET),
        pole_provider=earth_pole.de440_long_term_earth_pole,
    )
    return {
        "mode": earth_pole.DYNAMIC,
        "samples": [
            {
                "absolute_et": absolute_et,
                "raw_hex": [value.hex() for value in dynamic.raw_pole_at_elapsed_time(float(absolute_et))],
                "effective_hex": [value.hex() for value in dynamic.pole_at_elapsed_time(float(absolute_et))],
            }
            for absolute_et in (START_ET, END_ET)
        ],
    }


def _numerical_kat_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "reference_raw_state_sha256": report["reference"]["raw_state_evidence"]["sha256"],
        "reference_query_kats": report["reference"]["query_kats"],
        "pole_samples": report["pole_samples"],
        "lanes": [
            {
                "lane_id": lane["lane_id"],
                "raw_state_sha256": lane["raw_state_evidence"]["sha256"],
                "accepted_step_sha256": lane["accounting"]["accepted_step_sha256"],
                "accepted_step_ledger_content_sha256": lane["accounting"][
                    "accepted_step_ledger_content_sha256"
                ],
                "attempted_steps": lane["accounting"]["attempted_steps"],
                "accepted_steps": lane["accounting"]["accepted_steps"],
                "rejected_steps": lane["accounting"]["rejected_steps"],
                "earth_j2_correction_evaluations": lane["accounting"][
                    "earth_j2_correction_evaluations"
                ],
            }
            for lane in report["lanes"]
        ],
    }


def _numerical_kat(report: dict[str, Any]) -> str:
    return hashlib.sha256(NUMERICAL_KAT_DOMAIN + canonical(_numerical_kat_payload(report))).hexdigest()


def _semantic_sha256(report: dict[str, Any]) -> str:
    core = {key: value for key, value in report.items() if key != "semantic_sha256"}
    return hashlib.sha256(SEMANTIC_DOMAIN + canonical(core)).hexdigest()


def _make_spec(shape: tuple[int, int]) -> AdaptiveRKF78Spec:
    return AdaptiveRKF78Spec(
        checkpoint_epochs=(0.0, float(END_ET)),
        initial_step=MAXIMUM_STEP,
        minimum_step=1.0e-9,
        maximum_step=MAXIMUM_STEP,
        position_atol=np.full(shape, POSITION_ATOL, dtype=np.float64),
        position_rtol=RELATIVE_TOLERANCE,
        velocity_atol=np.full(shape, VELOCITY_ATOL, dtype=np.float64),
        velocity_rtol=RELATIVE_TOLERANCE,
        maximum_steps=MAXIMUM_STEPS,
        maximum_rejections=MAXIMUM_REJECTIONS,
        safety_factor=0.9,
        minimum_scale_factor=0.2,
        maximum_scale_factor=5.0,
    )


def build_report(
    artifact_root: Path,
    parameter_root: Path,
    *,
    calibration: bool,
) -> dict[str, Any]:
    artifacts_before = _validate_roots(artifact_root, parameter_root)
    sources_before = _source_records()
    runtime_before = _runtime_record()
    initial_receipt = _execute_at(artifact_root, START_ET, "initial")
    final_receipt = _execute_at(artifact_root, END_ET, "final")
    parameter_fd = os.open(parameter_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        prepared = engine_preparation.prepare_de440_newtonian_engine_inputs(
            initial_receipt,
            parameter_root_directory_fd=parameter_fd,
            receipt_id="receipt.post-v5.step5.j2-rkf78-e2e.prepared",
            state_id="state.post-v5.step5.j2-rkf78-e2e.initial",
            plan_id="plan.post-v5.step5.j2-rkf78-e2e.point",
        )
    finally:
        os.close(parameter_fd)
    if prepared.snapshot.body_ids != BODY_IDS:
        raise EndToEndProbeError("prepared resolved-eleven body roster changed")
    initial_positions, initial_velocities, initial_kat = _receipt_values(
        initial_receipt, prepared.preparation_receipt.model_spec
    )
    final_positions, final_velocities, final_kat = _receipt_values(
        final_receipt, prepared.preparation_receipt.model_spec
    )
    if (
        prepared.snapshot.positions.tobytes(order="C") != initial_positions.tobytes(order="C")
        or prepared.snapshot.velocities.tobytes(order="C") != initial_velocities.tobytes(order="C")
    ):
        raise EndToEndProbeError("prepared initial state differs from retained DE440 replay")
    reference_matrix = np.ascontiguousarray(
        [
            np.concatenate(([float(START_ET)], initial_positions.reshape(-1), initial_velocities.reshape(-1))),
            np.concatenate(([float(END_ET)], final_positions.reshape(-1), final_velocities.reshape(-1))),
        ],
        dtype=np.float64,
    )
    spec = _make_spec(prepared.snapshot.positions.shape)
    fixed_force = earth_j2.prepare_earth_j2_pair_force(
        source_id="EARTH",
        target_id="MOON",
        source_index=EARTH_INDEX,
        target_index=MOON_INDEX,
    )
    dynamic_policy = earth_pole.prepare_earth_pole_policy(
        earth_pole.DYNAMIC,
        absolute_start_et=float(START_ET),
        pole_provider=earth_pole.de440_long_term_earth_pole,
    )
    dynamic_force = earth_j2.prepare_earth_j2_pair_force(
        source_id="EARTH",
        target_id="MOON",
        source_index=EARTH_INDEX,
        target_index=MOON_INDEX,
        pole_policy=dynamic_policy,
    )
    lane_specs = (
        ("newtonian_control", None),
        ("fixed_j2000_positive_z", fixed_force),
        ("dynamic_de440_long_term_pole", dynamic_force),
    )
    lane_records = []
    lane_matrices: dict[str, np.ndarray] = {}
    source_metadata_before = prepared.force_plan.models[0].parameter_metadata[0]
    for lane_id, force in lane_specs:
        configuration = earth_j2_trajectory.prepare_de440_earth_j2_rkf78(
            prepared,
            spec,
            trajectory_plan_id=f"plan.post-v5.step5.j2-rkf78-e2e.{lane_id}",
            earth_j2_force=force,
        )
        primary = earth_j2_trajectory.integrate_de440_earth_j2_rkf78(configuration)
        replay = earth_j2_trajectory.integrate_de440_earth_j2_rkf78(configuration)
        record = _result_record(lane_id, primary, replay)
        lane_records.append(record)
        lane_matrices[lane_id] = _integration_matrix(primary)
    source_metadata_after = prepared.force_plan.models[0].parameter_metadata[0]
    reference = {
        "raw_state_evidence": _matrix_evidence(
            reference_matrix,
            "absolute_et_seconds,body_major_position_xyz_metres,body_major_velocity_xyz_metres_per_second",
        ),
        "query_kats": [
            {"absolute_et": START_ET, **initial_kat},
            {"absolute_et": END_ET, **final_kat},
        ],
        "primary_replay_exact": True,
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "evidence_class": "MODEL_OUTPUT",
        "calibration": calibration,
        "release_ready": not calibration and EXPECTED_NUMERICAL_KAT_SHA256 != "CALIBRATE",
        "protocol": _protocol(),
        "artifacts": artifacts_before,
        "sources": sources_before,
        "runtime": runtime_before,
        "reference": reference,
        "pole_samples": _pole_samples(),
        "lanes": lane_records,
        "comparisons": _comparisons(lane_matrices, reference_matrix),
        "checks": {
            "retained_artifacts_stable": artifacts_before
            == _validate_roots(artifact_root, parameter_root),
            "source_bytes_stable": sources_before == _source_records(),
            "runtime_identity_stable": runtime_before == _runtime_record(),
            "prepared_initial_state_equals_de440": True,
            "source_point_plan_unchanged": (
                source_metadata_before == source_metadata_after
                and source_metadata_after.validity_start.hex() == "0x0.0p+0"
                and source_metadata_after.validity_end.hex() == "0x0.0p+0"
            ),
            "all_fresh_trajectory_replays_exact": all(
                lane["fresh_replay"]["exact"] for lane in lane_records
            ),
            "all_lanes_zero_rejections": all(
                lane["accounting"]["rejected_steps"] == 0 for lane in lane_records
            ),
            "all_lanes_expected_step_count": all(
                lane["accounting"]["accepted_steps"] == EXPECTED_ACCEPTED_STEPS
                for lane in lane_records
            ),
            "disabled_lane_has_no_j2_evaluations": (
                lane_records[0]["accounting"]["earth_j2_correction_evaluations"] == 0
            ),
            "enabled_lanes_evaluate_j2_at_every_stage": all(
                lane["accounting"]["earth_j2_correction_evaluations"]
                == lane["accounting"]["rkf78_stage_evaluations"]
                for lane in lane_records[1:]
            ),
            "dynamic_and_fixed_endpoint_states_differ": (
                lane_records[1]["raw_state_evidence"]["sha256"]
                != lane_records[2]["raw_state_evidence"]["sha256"]
            ),
        },
        "claims": CLAIM_CONTROLS,
        "expected_numerical_kat_sha256": EXPECTED_NUMERICAL_KAT_SHA256,
    }
    report["observed_numerical_kat_sha256"] = _numerical_kat(report)
    report["semantic_sha256"] = _semantic_sha256(report)
    validate_report(report, enforce_kat=not calibration)
    return report


def _validate_step_accounting(lane: dict[str, Any], matrix: np.ndarray) -> None:
    accounting = lane["accounting"]
    required = {
        "accepted_step_hex", "accepted_step_sha256",
        "accepted_step_ledger_content_sha256", "attempted_steps", "accepted_steps",
        "rejected_steps", "rkf78_stage_evaluations", "baseline_force_evaluations",
        "earth_j2_correction_evaluations", "checkpoint_counters",
    }
    if type(accounting) is not dict or set(accounting) != required:
        raise EndToEndProbeError("lane accounting schema changed")
    for label in (
        "attempted_steps", "accepted_steps", "rejected_steps",
        "rkf78_stage_evaluations", "baseline_force_evaluations",
        "earth_j2_correction_evaluations",
    ):
        if type(accounting[label]) is not int or accounting[label] < 0:
            raise EndToEndProbeError("lane accounting contains an invalid integer")
    if type(accounting["accepted_step_hex"]) is not list:
        raise EndToEndProbeError("accepted steps must be an exact list")
    steps = []
    for item in accounting["accepted_step_hex"]:
        if type(item) is not str:
            raise EndToEndProbeError("accepted step is not canonical float hex")
        value = float.fromhex(item)
        if value.hex() != item or not math.isfinite(value) or value <= 0.0:
            raise EndToEndProbeError("accepted step is invalid")
        steps.append(value)
    step_tuple = tuple(steps)
    if (
        len(step_tuple) != accounting["accepted_steps"]
        or accounting["attempted_steps"]
        != accounting["accepted_steps"] + accounting["rejected_steps"]
        or accounting["accepted_step_sha256"] != _step_digest(step_tuple)
        or accounting["rkf78_stage_evaluations"]
        != accounting["attempted_steps"] * 13
        or accounting["baseline_force_evaluations"]
        != accounting["rkf78_stage_evaluations"]
    ):
        raise EndToEndProbeError("lane step accounting changed")
    counters = accounting["checkpoint_counters"]
    if counters != [[0, 0], [accounting["accepted_steps"], accounting["rejected_steps"]]]:
        raise EndToEndProbeError("lane checkpoint counters changed")
    checkpoints = (
        TrajectoryCheckpoint(
            index=0, epoch=float(matrix[0, 0]), body_ids=BODY_IDS,
            backend_id="numpy", device="cpu",
            positions=np.ascontiguousarray(matrix[0, 1:34].reshape(len(BODY_IDS), 3)),
            velocities=np.ascontiguousarray(matrix[0, 34:67].reshape(len(BODY_IDS), 3)),
            accepted_steps=0, rejected_steps=0,
        ),
        TrajectoryCheckpoint(
            index=1, epoch=float(matrix[1, 0]), body_ids=BODY_IDS,
            backend_id="numpy", device="cpu",
            positions=np.ascontiguousarray(matrix[1, 1:34].reshape(len(BODY_IDS), 3)),
            velocities=np.ascontiguousarray(matrix[1, 34:67].reshape(len(BODY_IDS), 3)),
            accepted_steps=accounting["accepted_steps"],
            rejected_steps=accounting["rejected_steps"],
        ),
    )
    epochs = tuple(float((index + 1) * MAXIMUM_STEP) for index in range(len(step_tuple)))
    expected_ledger = engine_trajectory._accepted_step_ledger_content_sha256(
        accepted_step_epochs=epochs,
        accepted_step_magnitudes=step_tuple,
        checkpoints=checkpoints,
        direction="FORWARD",
        attempted_steps=accounting["attempted_steps"],
        accepted_steps=accounting["accepted_steps"],
        rejected_steps=accounting["rejected_steps"],
        force_evaluations=accounting["rkf78_stage_evaluations"],
        minimum_accepted_step=float(min(step_tuple)),
        maximum_accepted_step=float(max(step_tuple)),
        last_accepted_step=float(step_tuple[-1]),
        magnitude_source=engine_trajectory.RKF78_ACCEPTED_STEP_MAGNITUDE_SOURCE,
    )
    if accounting["accepted_step_ledger_content_sha256"] != expected_ledger:
        raise EndToEndProbeError("accepted-step engine ledger checksum changed")


def validate_report(report: object, *, enforce_kat: bool = True) -> None:
    if type(report) is not dict or set(report) != {
        "schema", "benchmark_id", "evidence_class", "calibration", "release_ready",
        "protocol", "artifacts", "sources", "runtime", "reference", "pole_samples",
        "lanes", "comparisons", "checks", "claims", "expected_numerical_kat_sha256",
        "observed_numerical_kat_sha256", "semantic_sha256",
    }:
        raise EndToEndProbeError("report top-level schema changed")
    if (
        report["schema"] != SCHEMA
        or report["benchmark_id"] != BENCHMARK_ID
        or report["evidence_class"] != "MODEL_OUTPUT"
        or type(report["calibration"]) is not bool
        or type(report["release_ready"]) is not bool
        or report["protocol"] != _protocol()
        or report["artifacts"]
        != [
            {"logical_locator": locator, "size_bytes": size, "sha256": sha256}
            for locator, (size, sha256) in sorted(ARTIFACTS.items())
        ]
        or report["sources"] != _source_records()
        or report["runtime"] != _runtime_record()
        or report["claims"] != CLAIM_CONTROLS
    ):
        raise EndToEndProbeError("report protocol, provenance, or claim boundary changed")
    reference = report["reference"]
    if type(reference) is not dict or set(reference) != {
        "raw_state_evidence", "query_kats", "primary_replay_exact"
    } or reference["primary_replay_exact"] is not True:
        raise EndToEndProbeError("reference evidence schema changed")
    reference_matrix = _decode_matrix(
        reference["raw_state_evidence"],
        "absolute_et_seconds,body_major_position_xyz_metres,body_major_velocity_xyz_metres_per_second",
        (2, 67),
    )
    if tuple(reference_matrix[:, 0]) != (float(START_ET), float(END_ET)):
        raise EndToEndProbeError("reference absolute epochs changed")
    query_kats = reference["query_kats"]
    if (
        type(query_kats) is not list
        or len(query_kats) != 2
        or [item.get("absolute_et") for item in query_kats] != [START_ET, END_ET]
    ):
        raise EndToEndProbeError("reference query KAT roster changed")
    expected_poles = _pole_samples()
    if report["pole_samples"] != expected_poles:
        raise EndToEndProbeError("Earth-pole samples changed")
    lanes = report["lanes"]
    expected_lane_contracts = (
        ("newtonian_control", False, None, 0),
        ("fixed_j2000_positive_z", True, earth_pole.FIXED_J2000_POSITIVE_Z, None),
        ("dynamic_de440_long_term_pole", True, earth_pole.DYNAMIC, None),
    )
    if type(lanes) is not list or len(lanes) != 3:
        raise EndToEndProbeError("lane roster changed")
    lane_matrices: dict[str, np.ndarray] = {}
    for lane, (lane_id, enabled, mode, correction_count) in zip(lanes, expected_lane_contracts):
        if type(lane) is not dict or set(lane) != {
            "lane_id", "earth_j2_enabled", "pole_mode", "applied_model_ids",
            "raw_state_evidence", "accounting", "fresh_replay",
        }:
            raise EndToEndProbeError("lane schema changed")
        if (
            lane["lane_id"] != lane_id
            or type(lane["earth_j2_enabled"]) is not bool
            or lane["earth_j2_enabled"] is not enabled
            or lane["pole_mode"] != mode
            or type(lane["applied_model_ids"]) is not list
        ):
            raise EndToEndProbeError("lane role or force binding changed")
        matrix = _decode_matrix(
            lane["raw_state_evidence"],
            "elapsed_epoch_seconds,body_major_position_xyz_metres,body_major_velocity_xyz_metres_per_second",
            (2, 67),
        )
        if tuple(matrix[:, 0]) != (0.0, float(END_ET)):
            raise EndToEndProbeError("lane elapsed epochs changed")
        if matrix[0, 1:].tobytes(order="C") != reference_matrix[0, 1:].tobytes(order="C"):
            raise EndToEndProbeError("lane initial state differs from DE440")
        _validate_step_accounting(lane, matrix)
        accounting = lane["accounting"]
        expected_corrections = (
            0 if correction_count == 0 else accounting["rkf78_stage_evaluations"]
        )
        if (
            accounting["earth_j2_correction_evaluations"] != expected_corrections
            or accounting["accepted_steps"] != EXPECTED_ACCEPTED_STEPS
            or accounting["attempted_steps"] != EXPECTED_ACCEPTED_STEPS
            or accounting["rejected_steps"] != 0
        ):
            raise EndToEndProbeError("lane force accounting changed")
        replay = lane["fresh_replay"]
        if type(replay) is not dict or set(replay) != {
            "raw_state_sha256", "accepted_step_sha256",
            "accepted_step_ledger_content_sha256", "exact"
        } or replay["exact"] is not True:
            raise EndToEndProbeError("fresh replay evidence changed")
        if (
            replay["raw_state_sha256"] != lane["raw_state_evidence"]["sha256"]
            or replay["accepted_step_sha256"] != accounting["accepted_step_sha256"]
            or replay["accepted_step_ledger_content_sha256"]
            != accounting["accepted_step_ledger_content_sha256"]
        ):
            raise EndToEndProbeError("fresh replay does not match the primary lane")
        lane_matrices[lane_id] = matrix
    if report["comparisons"] != _comparisons(lane_matrices, reference_matrix):
        raise EndToEndProbeError("derived endpoint comparisons changed")
    expected_checks = {
        "retained_artifacts_stable": True,
        "source_bytes_stable": True,
        "runtime_identity_stable": True,
        "prepared_initial_state_equals_de440": True,
        "source_point_plan_unchanged": True,
        "all_fresh_trajectory_replays_exact": True,
        "all_lanes_zero_rejections": True,
        "all_lanes_expected_step_count": True,
        "disabled_lane_has_no_j2_evaluations": True,
        "enabled_lanes_evaluate_j2_at_every_stage": True,
        "dynamic_and_fixed_endpoint_states_differ": True,
    }
    if report["checks"] != expected_checks:
        raise EndToEndProbeError("validity checks changed")
    observed_kat = _numerical_kat(report)
    if report["observed_numerical_kat_sha256"] != observed_kat:
        raise EndToEndProbeError("observed numerical KAT changed")
    if report["expected_numerical_kat_sha256"] != EXPECTED_NUMERICAL_KAT_SHA256:
        raise EndToEndProbeError("expected numerical KAT field changed")
    if enforce_kat:
        if EXPECTED_NUMERICAL_KAT_SHA256 == "CALIBRATE":
            raise EndToEndProbeError("numerical KAT remains unfrozen")
        if observed_kat != EXPECTED_NUMERICAL_KAT_SHA256:
            raise EndToEndProbeError("numerical KAT does not match the frozen value")
    expected_release_ready = (
        not report["calibration"] and EXPECTED_NUMERICAL_KAT_SHA256 != "CALIBRATE"
    )
    if report["release_ready"] is not expected_release_ready:
        raise EndToEndProbeError("release-ready state changed")
    if report["semantic_sha256"] != _semantic_sha256(report):
        raise EndToEndProbeError("semantic report checksum changed")


def _write_new(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        written = 0
        while written < len(data):
            count = os.write(descriptor, data[written:])
            if count <= 0:
                raise EndToEndProbeError("report write made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--parameter-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibrate", action="store_true")
    arguments = parser.parse_args()
    if arguments.calibrate and EXPECTED_NUMERICAL_KAT_SHA256 != "CALIBRATE":
        raise EndToEndProbeError("calibration is disabled after the KAT is frozen")
    report = build_report(
        arguments.artifact_root.resolve(),
        arguments.parameter_root.resolve(),
        calibration=arguments.calibrate,
    )
    payload = canonical(report)
    if len(payload) > MAXIMUM_REPORT_BYTES:
        raise EndToEndProbeError("report exceeds the fixed byte cap")
    _write_new(arguments.output.resolve(), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
