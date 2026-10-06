#!/usr/bin/env python3
"""Bounded M2 qualification of the full known-system massive-source model.

This executable is deliberately limited to the Sun plus the nine planetary
system barycenters in ``KNOWN_SOLAR_SYSTEM_V2`` over ten Julian years.  It
compares two exact-replay JX ordered-Jacobi Wisdom--Holman lanes with an
independently written Cartesian SciPy DOP853 integration, REBOUND 5.1.1
IAS15, and the retained annual DE441/Horizons vectors.  It also compares one
ten-year JX tile with ten chained one-year tiles.

There are no massless tracers, compact-planet hypotheses, observational
weights, or science inference in this program.  Every claim remains false.
The run directory and its protocol, state, result, and artifact manifests are
create-only.  A failed numerical gate is retained as ``INVALID``; it is never
silently relaxed after execution.
"""

from __future__ import annotations

import os


_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
for _name in _THREAD_ENVIRONMENT:
    os.environ[_name] = "1"

import argparse
from collections.abc import Mapping, Sequence
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import platform
import stat
import subprocess
import sys
import time
from typing import Any
import zipfile

import numpy as np

from jxplanetx.engine import Provenance
from jxplanetx.engine.wisdom_holman_contracts import (
    WH_MAXIMUM_ORBIT_STEP_FRACTION,
    WH_MAXIMUM_PERIAPSE_STEP_FRACTION,
)
from jxplanetx.fast_wisdom_holman import fast_wisdom_holman_runtime_identity
from jxplanetx.long_horizon.physical_models import (
    J2000_MEAN_OBLIQUITY_COS,
    J2000_MEAN_OBLIQUITY_SIN,
    JULIAN_YEAR_DAYS,
    KNOWN_SOLAR_SYSTEM_V2,
    KNOWN_SYSTEM_BODY_IDS,
    KNOWN_SYSTEM_BODY_NAMES,
)
from jxplanetx.restricted_wh_source import (
    SOURCE_PROVIDER_CLAIMS,
    WisdomHolmanSourceTileSpec,
    build_wisdom_holman_source_tile,
)


def _capture_import_time_local_files() -> dict[str, dict[str, str]]:
    root = Path(__file__).resolve().parents[1]
    records: dict[str, dict[str, str]] = {}
    candidates: list[tuple[str, Path]] = [("__benchmark__", Path(__file__))]
    for module_name, module in tuple(sys.modules.items()):
        path_text = getattr(module, "__file__", None)
        if module_name.startswith("jxplanetx") and isinstance(path_text, str):
            candidates.append((module_name, Path(path_text)))
    for module_name, unresolved in candidates:
        if unresolved.is_symlink():
            raise RuntimeError(f"imported local module is a symlink: {module_name}")
        path = unresolved.resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            continue
        if not path.is_file():
            raise RuntimeError(f"imported local module is absent: {module_name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        previous = records.get(relative)
        if previous is not None and previous["sha256"] != digest:
            raise RuntimeError("one imported local path has inconsistent bytes")
        records[relative] = {"module": module_name, "sha256": digest}
    return dict(sorted(records.items()))


_IMPORT_TIME_LOCAL_FILES = _capture_import_time_local_files()


PROTOCOL_SCHEMA = "jx.long-horizon.m2-known-source-protocol.v2"
RESULT_SCHEMA = "jx.long-horizon.m2-known-source-result.v2"
ARTIFACT_SCHEMA = "jx.long-horizon.m2-known-source-artifacts.v2"
COMPLETION_SCHEMA = "jx.long-horizon.m2-known-source-completion.v2"
RUN_ID = "jx-long-horizon-m2-known-source-10y-v2"
CLASSIFICATION = "BOUNDED_NUMERICAL_ENGINEERING_QUALIFICATION_ONLY"

PROTOCOL_DOMAIN = b"jx.long-horizon.m2-known-source.protocol.v2\0"
RESULT_DOMAIN = b"jx.long-horizon.m2-known-source.result.v2\0"
ARTIFACT_DOMAIN = b"jx.long-horizon.m2-known-source.artifacts.v2\0"
COMPLETION_DOMAIN = b"jx.long-horizon.m2-known-source.completion.v2\0"
STATE_DOMAIN = b"jx.long-horizon.m2-known-source.state.v2\0"
CODE_FILES_DOMAIN = b"jx.m2.code-files.v2\0"
RUNTIME_DISTRIBUTION_DOMAIN = b"jx.m2.runtime-distribution.v2\0"
SOURCE_SCHEDULE_DOMAIN = b"jx.long-horizon.m2-source-schedule.v2\0"
SOURCE_ARCHIVE_SCHEMA = "jx.long-horizon.m2-executed-sources.v2"
PROVENANCE_VERSION = "v2"
DEFAULT_OUTPUT_ROOT = Path("runs/long_horizon_m2_source_preflight_v2")

HORIZON_YEARS = 10
ANNUAL_SAMPLE_YEARS = tuple(float(index) for index in range(HORIZON_YEARS + 1))
COARSE_STEP_YEARS = 1.0 / 4096.0
FINE_STEP_YEARS = 1.0 / 8192.0
COARSE_STEP_COUNT = HORIZON_YEARS * 4096
FINE_STEP_COUNT = HORIZON_YEARS * 8192
PARTITION_COUNT = 10
PARTITION_STEP_COUNT = 4096

REQUIRED_REBOUND_VERSION = "5.1.1"
DOP853_RTOL = 1.0e-13
DOP853_ATOL = 1.0e-15
DOP853_MAX_STEP_YEARS = 1.0 / 128.0
IAS15_EPSILON = 1.0e-15

REFERENCE_VECTOR_PATH = Path(
    "runs/de441_horizons_10yr/reference/horizons_de441_vectors.csv"
)
REFERENCE_MANIFEST_PATH = Path(
    "runs/de441_horizons_10yr/reference/horizons_reference_manifest.json"
)
REFERENCE_CONTRACT_PATH = Path("runs/de441_horizons_10yr/contract_v1.json")
GM_TABLE_PATH = Path("runs/de441_horizons_10yr/gm_de440_major_barycenters.csv")
REFERENCE_VECTOR_SHA256 = (
    "18308086c9191333448e07a6ab6942a6b0fc7f0100ad2b3abaffbb1a0fd3993e"
)
REFERENCE_MANIFEST_SHA256 = (
    "87bf405df5d333dde237b8a6f63ba57a67baff60e74f752b5d1e77cc094d4565"
)
REFERENCE_CONTRACT_SHA256 = (
    "5112c90e71317a5cae54d17ea82296296af68812f18c663429dc773ac161053f"
)
GM_TABLE_SHA256 = (
    "11fe03048f125e5283c3ef0d8054bc366baaf74dd86093ac3e61798e93769217"
)

# Jupiter, Saturn, Uranus, Neptune, and Pluto barycenters.  Inner-body DE441
# residuals are recorded but deliberately excluded from ephemeris gates.
OUTER_GATE_BODY_IDS = (5, 6, 7, 8, 9)
OUTER_GATE_INDICES = tuple(
    KNOWN_SYSTEM_BODY_IDS.index(body_id) for body_id in OUTER_GATE_BODY_IDS
)

# These values are immutable pre-execution commitments.  All velocity gates
# use AU per Julian year, including the converted DE441 comparison.
GATE_THRESHOLDS = {
    "initial_reference_position_AU": 0.0,
    "initial_reference_velocity_AU_per_year": 0.0,
    "coarse_all_body_vs_dop853_position_AU": 8.0e-8,
    "coarse_all_body_vs_dop853_velocity_AU_per_year": 8.0e-7,
    "fine_all_body_vs_dop853_position_AU": 2.0e-8,
    "fine_all_body_vs_dop853_velocity_AU_per_year": 2.0e-7,
    "fine_outer_vs_dop853_position_AU": 3.0e-9,
    "fine_outer_vs_dop853_velocity_AU_per_year": 3.0e-9,
    "minimum_position_refinement_ratio": 2.0,
    "minimum_velocity_refinement_ratio": 2.0,
    "dop853_vs_ias15_position_AU": 5.0e-11,
    "dop853_vs_ias15_velocity_AU_per_year": 5.0e-10,
    "outer_de441_position_AU": 1.0e-5,
    "outer_de441_velocity_AU_per_year": 4.0e-6,
    "partition_position_AU": 1.0e-9,
    "partition_velocity_AU_per_year": 1.0e-8,
    "coarse_relative_energy_drift": 1.0e-8,
    "fine_relative_energy_drift": 3.0e-9,
    "jx_relative_angular_momentum_drift": 5.0e-12,
    "reference_relative_energy_drift": 1.0e-10,
    "reference_relative_angular_momentum_drift": 1.0e-10,
    "independent_minimum_pair_path_lower_bound_AU": 1.0e-2,
    "independent_minimum_primary_periapse_AU": 5.0e-2,
    "independent_maximum_primary_phase_radians": 1.0 / 64.0,
    "independent_maximum_primary_periapse_phase_radians": 1.0 / 64.0,
    "native_minimum_pair_path_lower_bound_AU": 1.0e-4,
    "native_minimum_jacobi_periapse_AU": 5.0e-2,
    "native_maximum_orbit_step_fraction": WH_MAXIMUM_ORBIT_STEP_FRACTION,
    "native_maximum_periapse_step_fraction": (
        WH_MAXIMUM_PERIAPSE_STEP_FRACTION
    ),
}

SOURCE_ENCOUNTER_RADIUS_AU = 1.0e-6
INDEPENDENT_PAIR_SEPARATION_FLOOR_AU = GATE_THRESHOLDS[
    "independent_minimum_pair_path_lower_bound_AU"
]
LOCAL_CODE_SUFFIXES = frozenset((".c", ".cc", ".cpp", ".cu", ".h", ".hpp", ".py", ".pyx"))
SOURCE_ARCHIVE_NAME = "EXECUTED_SOURCES.zip"
COMPLETION_MARKER_NAME = "COMPLETED.json"

CLAIMS = {
    "complete_de441_force_model": False,
    "detection_claim": False,
    "ephemeris_reconstruction": False,
    "exclusion_claim": False,
    "four_point_five_gyr_accuracy_qualified": False,
    "inference_authorized": False,
    "long_duration_accuracy_qualified": False,
    "observationally_fitted": False,
    "planet_nine_modeled": False,
    "production_qualified": False,
    "science_result": False,
    "tracer_dynamics_qualified": False,
    "uncertainty_calibrated": False,
}

CODE_PATHS = (
    Path("benchmarks/jx_long_horizon_m2_source_preflight.py"),
    Path("src/jxplanetx/long_horizon/physical_models.py"),
    Path("src/jxplanetx/restricted_wh_source.py"),
    Path("src/jxplanetx/fast_wisdom_holman.py"),
    Path("src/jxplanetx/_all_conic_kepler.py"),
    Path("src/jxplanetx/_wisdom_holman_cpu_v3_module.c"),
    Path("src/jxplanetx/_wisdom_holman_loop.c"),
    Path("src/jxplanetx/_wisdom_holman_loop_v2.c"),
    Path("src/jxplanetx/_wisdom_holman_loop_v3.c"),
)


class M2SourceQualificationError(RuntimeError):
    """The predeclared M2 qualification could not be executed validly."""


def _plain(value: Any) -> Any:
    """Convert frozen runtime trees into strict JSON-compatible values."""

    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError) as exc:
        raise M2SourceQualificationError("record is not strict canonical JSON") from exc


def _content_record(
    value: Mapping[str, Any], domain: bytes
) -> dict[str, Any]:
    if "content_sha256" in value:
        raise M2SourceQualificationError("content hash must be derived")
    plain = dict(value)
    plain["content_sha256"] = hashlib.sha256(
        domain + _canonical_json(value)
    ).hexdigest()
    return plain


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_repository_file_no_follow(root: Path, relative: Path) -> bytes:
    resolved_root = root.resolve()
    if relative.is_absolute() or ".." in relative.parts:
        raise M2SourceQualificationError("repository-relative path is invalid")
    current = resolved_root
    for component in relative.parts:
        current = current / component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise M2SourceQualificationError(
                f"required repository file is absent: {relative}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise M2SourceQualificationError(
                f"repository file path contains a symlink: {relative}"
            )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved_root / relative, flags)
    except OSError as exc:
        raise M2SourceQualificationError(
            f"cannot open required repository file: {relative}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise M2SourceQualificationError(
                f"repository input is not a regular file: {relative}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        finished = os.fstat(descriptor)
        if (
            opened.st_dev != finished.st_dev
            or opened.st_ino != finished.st_ino
            or opened.st_size != finished.st_size
        ):
            raise M2SourceQualificationError(
                f"repository input changed while reading: {relative}"
            )
        payload = b"".join(chunks)
        if len(payload) != finished.st_size:
            raise M2SourceQualificationError(
                f"repository input size changed while reading: {relative}"
            )
        return payload
    finally:
        os.close(descriptor)


def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256(STATE_DOMAIN)
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _array_bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left_value = np.asarray(left)
    right_value = np.asarray(right)
    return bool(
        left_value.dtype == right_value.dtype
        and left_value.shape == right_value.shape
        and np.array_equal(
            np.ascontiguousarray(left_value).view(np.uint8),
            np.ascontiguousarray(right_value).view(np.uint8),
        )
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    _fsync_directory(path.parent)


def _write_npz_create_only(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w+b") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    _fsync_directory(path.parent)


def _verified_file(root: Path, relative: Path, expected: str) -> dict[str, Any]:
    payload = _read_repository_file_no_follow(root, relative)
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise M2SourceQualificationError(
            f"input digest changed for {relative}: {observed}"
        )
    return {
        "path": relative.as_posix(),
        "sha256": observed,
        "size_bytes": len(payload),
    }


def _local_code_paths(root: Path) -> tuple[Path, ...]:
    paths = {path for path in CODE_PATHS}
    paths.update(Path(relative) for relative in _IMPORT_TIME_LOCAL_FILES)
    package_root = root / "src" / "jxplanetx"
    if package_root.is_symlink() or not package_root.is_dir():
        raise M2SourceQualificationError("local package source tree is absent")
    for path in package_root.rglob("*"):
        if path.is_symlink():
            raise M2SourceQualificationError(
                f"local package source contains a symlink: {path}"
            )
        if path.is_file() and path.suffix in LOCAL_CODE_SUFFIXES:
            paths.add(path.relative_to(root))
    packaged_model_root = package_root / "data" / "long_horizon_known_system_v2"
    if packaged_model_root.is_symlink() or not packaged_model_root.is_dir():
        raise M2SourceQualificationError("packaged known-system inputs are absent")
    for path in packaged_model_root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise M2SourceQualificationError(
                f"packaged known-system input is not a regular file: {path}"
            )
        paths.add(path.relative_to(root))
    paths.update(
        (
            REFERENCE_VECTOR_PATH,
            REFERENCE_MANIFEST_PATH,
            REFERENCE_CONTRACT_PATH,
            GM_TABLE_PATH,
        )
    )
    return tuple(sorted(paths, key=lambda value: value.as_posix()))


def _code_identity(root: Path) -> dict[str, Any]:
    files: dict[str, str] = {}
    for relative in _local_code_paths(root):
        payload = _read_repository_file_no_follow(root, relative)
        files[relative.as_posix()] = hashlib.sha256(payload).hexdigest()
    for relative, imported in _IMPORT_TIME_LOCAL_FILES.items():
        if files.get(relative) != imported["sha256"]:
            raise M2SourceQualificationError(
                f"imported local bytes differ from archived disk bytes: {relative}"
            )
    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10.0,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise M2SourceQualificationError("cannot bind git HEAD") from exc
    if len(git_head) != 40 or any(c not in "0123456789abcdef" for c in git_head):
        raise M2SourceQualificationError("git HEAD is not a full lowercase digest")
    return {
        "git_head": git_head,
        "files": files,
        "file_count": len(files),
        "import_time_local_files": _IMPORT_TIME_LOCAL_FILES,
        "files_content_sha256": hashlib.sha256(
            CODE_FILES_DOMAIN + _canonical_json(files)
        ).hexdigest(),
    }


def _write_source_archive_create_only(
    path: Path, root: Path, identity: Mapping[str, Any]
) -> dict[str, Any]:
    files = identity.get("files")
    if not isinstance(files, Mapping) or not files:
        raise M2SourceQualificationError("code identity has no source roster")
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w+b") as stream:
            with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_STORED) as archive:
                manifest = {
                    "schema": SOURCE_ARCHIVE_SCHEMA,
                    "git_head": identity["git_head"],
                    "files": dict(files),
                    "import_time_local_files": identity[
                        "import_time_local_files"
                    ],
                    "files_content_sha256": identity["files_content_sha256"],
                }
                info = zipfile.ZipInfo("MANIFEST.json", date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, _canonical_json(manifest) + b"\n")
                for relative, expected_sha256 in sorted(files.items()):
                    if not isinstance(relative, str) or not isinstance(expected_sha256, str):
                        raise M2SourceQualificationError("code identity roster is malformed")
                    relative_path = Path(relative)
                    payload = _read_repository_file_no_follow(root, relative_path)
                    if hashlib.sha256(payload).hexdigest() != expected_sha256:
                        raise M2SourceQualificationError(
                            f"source changed before archival: {relative}"
                        )
                    info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                    info.create_system = 3
                    info.external_attr = 0o100644 << 16
                    archive.writestr(info, payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    _fsync_directory(path.parent)
    return {
        "path": path.name,
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "file_count": len(files),
        "files_content_sha256": identity["files_content_sha256"],
    }


def _module_file_identity(module: Any) -> dict[str, Any]:
    path_text = getattr(module, "__file__", None)
    if not isinstance(path_text, str):
        raise M2SourceQualificationError(
            f"cannot identify binary/module path for {module!r}"
        )
    path = Path(path_text).resolve()
    if not path.is_file():
        raise M2SourceQualificationError(f"runtime module file is absent: {path}")
    return {"path": str(path), "sha256": _sha256_file(path), "size_bytes": path.stat().st_size}


def _distribution_identity(
    name: str, required_modules: Sequence[Any]
) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise M2SourceQualificationError(
            f"required runtime distribution is absent: {name}"
        ) from exc
    rows: list[dict[str, Any]] = []
    hashed_paths: dict[Path, str] = {}
    for item in sorted(distribution.files or (), key=lambda value: str(value)):
        relative = str(item).replace(os.sep, "/")
        if "__pycache__" in relative or relative.endswith((".pyc", ".pyo")):
            continue
        unresolved_path = Path(distribution.locate_file(item))
        if unresolved_path.is_symlink():
            raise M2SourceQualificationError(
                f"runtime distribution contains a symlink: {name}:{relative}"
            )
        path = unresolved_path.resolve()
        if not path.is_file():
            raise M2SourceQualificationError(
                f"runtime distribution member is absent: {name}:{relative}"
            )
        hashed_paths[path] = relative
        rows.append(
            {
                "path": relative,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    if not rows:
        raise M2SourceQualificationError(
            f"runtime distribution has no hashable files: {name}"
        )
    required_members: dict[str, str] = {}
    for module in required_modules:
        path_text = getattr(module, "__file__", None)
        module_name = getattr(module, "__name__", repr(module))
        if not isinstance(path_text, str):
            raise M2SourceQualificationError(
                f"required imported module has no file: {module_name}"
            )
        path = Path(path_text).resolve()
        try:
            required_members[module_name] = hashed_paths[path]
        except KeyError as exc:
            raise M2SourceQualificationError(
                f"imported module is outside hashed distribution: {module_name}"
            ) from exc
    return {
        "name": distribution.metadata["Name"],
        "version": distribution.version,
        "file_count": len(rows),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "content_sha256": hashlib.sha256(
            RUNTIME_DISTRIBUTION_DOMAIN + _canonical_json(rows)
        ).hexdigest(),
        "required_imported_members": required_members,
    }


def _runtime_identity() -> dict[str, Any]:
    import scipy
    import scipy.integrate._ivp.base as scipy_base
    import scipy.integrate._ivp.common as scipy_common
    import scipy.integrate._ivp.ivp as scipy_ivp
    import scipy.integrate._ivp.rk as scipy_rk
    import rebound
    import rebound.simulation as rebound_simulation

    if rebound.__version__ != REQUIRED_REBOUND_VERSION:
        raise M2SourceQualificationError(
            f"REBOUND must be exactly {REQUIRED_REBOUND_VERSION}, found {rebound.__version__}"
        )
    numpy_core = np._core._multiarray_umath  # type: ignore[attr-defined]
    rebound_identity = _module_file_identity(rebound)
    library_candidates: list[Path] = []
    declared_library = getattr(rebound, "__libpath__", None)
    if isinstance(declared_library, str):
        library_candidates.append(Path(declared_library))
    if isinstance(getattr(rebound, "__file__", None), str):
        library_candidates.extend(
            Path(rebound.__file__).resolve().parent.glob("librebound*")
        )
    libraries = []
    for candidate in sorted({path.resolve() for path in library_candidates}):
        if candidate.is_file():
            libraries.append(
                {
                    "path": str(candidate),
                    "sha256": _sha256_file(candidate),
                    "size_bytes": candidate.stat().st_size,
                }
            )
    if not libraries:
        # Some wheels expose the shared object as the rebound module itself.
        libraries = [rebound_identity]
    return {
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_executable_identity": {
            "sha256": _sha256_file(Path(sys.executable).resolve()),
            "size_bytes": Path(sys.executable).resolve().stat().st_size,
        },
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "numpy_core": _module_file_identity(numpy_core),
        "numpy_distribution": _distribution_identity("numpy", (np, numpy_core)),
        "scipy_version": scipy.__version__,
        "scipy_dop853_module": _module_file_identity(scipy_rk),
        "scipy_distribution": _distribution_identity(
            "scipy", (scipy, scipy_base, scipy_common, scipy_ivp, scipy_rk)
        ),
        "rebound_version": rebound.__version__,
        "rebound_module": rebound_identity,
        "rebound_libraries": libraries,
        "rebound_distribution": _distribution_identity(
            "rebound", (rebound, rebound_simulation)
        ),
        "numerical_thread_environment": {
            name: os.environ.get(name) for name in _THREAD_ENVIRONMENT
        },
        "jx_wisdom_holman": fast_wisdom_holman_runtime_identity(),
    }


def _schedule_sha256(step: float) -> str:
    return hashlib.sha256(
        SOURCE_SCHEDULE_DOMAIN
        + KNOWN_SOLAR_SYSTEM_V2.content_sha256.encode("ascii")
        + b"\0"
        + step.hex().encode("ascii")
        + b"\0"
        + str(HORIZON_YEARS).encode("ascii")
    ).hexdigest()


def _verified_input_receipts(root: Path) -> dict[str, Any]:
    return {
        "de441_vectors": _verified_file(
            root, REFERENCE_VECTOR_PATH, REFERENCE_VECTOR_SHA256
        ),
        "de441_manifest": _verified_file(
            root, REFERENCE_MANIFEST_PATH, REFERENCE_MANIFEST_SHA256
        ),
        "de441_contract": _verified_file(
            root, REFERENCE_CONTRACT_PATH, REFERENCE_CONTRACT_SHA256
        ),
        "de440_gm_table": _verified_file(root, GM_TABLE_PATH, GM_TABLE_SHA256),
        "known_system_model_sha256": KNOWN_SOLAR_SYSTEM_V2.content_sha256,
        "known_system_converted_state_sha256": (
            KNOWN_SOLAR_SYSTEM_V2.converted_state_sha256
        ),
        "known_system_source_state_sha256": (
            KNOWN_SOLAR_SYSTEM_V2.source_barycentric_state_sha256
        ),
    }


def build_protocol(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    inputs = _verified_input_receipts(root)
    _, _, known_gm = _known_arrays()
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "run_id": RUN_ID,
        "status": "PREDECLARED_BEFORE_EXECUTION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": CLASSIFICATION,
        "model": {
            "force_law": "NEWTONIAN_POINT_MASS_GRAVITY_G_EQUALS_ONE",
            "known_system_schema": KNOWN_SOLAR_SYSTEM_V2.schema,
            "known_system_model_id": KNOWN_SOLAR_SYSTEM_V2.model_id,
            "epoch_jd_tdb_hex": KNOWN_SOLAR_SYSTEM_V2.epoch_jd_tdb.hex(),
            "body_ids": list(KNOWN_SYSTEM_BODY_IDS),
            "body_names": list(KNOWN_SYSTEM_BODY_NAMES),
            "outer_de441_gate_body_ids": list(OUTER_GATE_BODY_IDS),
            "inner_de441_residuals": "DESCRIPTIVE_ONLY",
        },
        "profile": {
            "horizon_years": HORIZON_YEARS,
            "annual_sample_years_hex": [value.hex() for value in ANNUAL_SAMPLE_YEARS],
            "jx_lanes": [
                {
                    "name": "h",
                    "step_years_hex": COARSE_STEP_YEARS.hex(),
                    "step_count": COARSE_STEP_COUNT,
                    "exact_replay_required": True,
                    "schedule_sha256": _schedule_sha256(COARSE_STEP_YEARS),
                },
                {
                    "name": "h_over_2",
                    "step_years_hex": FINE_STEP_YEARS.hex(),
                    "step_count": FINE_STEP_COUNT,
                    "exact_replay_required": True,
                    "schedule_sha256": _schedule_sha256(FINE_STEP_YEARS),
                },
            ],
            "partition_check": {
                "full_tile_years": 10,
                "segment_count": PARTITION_COUNT,
                "segment_years": 1,
                "step_years_hex": COARSE_STEP_YEARS.hex(),
                "steps_per_segment": PARTITION_STEP_COUNT,
            },
            "dop853": {
                "implementation": "scipy.integrate.solve_ivp(method='DOP853')",
                "independent_cartesian_rhs": True,
                "rtol_hex": DOP853_RTOL.hex(),
                "atol_hex": DOP853_ATOL.hex(),
                "maximum_step_years_hex": DOP853_MAX_STEP_YEARS.hex(),
            },
            "ias15": {
                "implementation": "REBOUND_5.1.1_IAS15",
                "epsilon_hex": IAS15_EPSILON.hex(),
                "exact_finish_time": 1,
            },
            "independent_source_guard": {
                "all_jx_nodes_checked": True,
                "pair_separation_floor_AU_hex": (
                    INDEPENDENT_PAIR_SEPARATION_FLOOR_AU.hex()
                ),
                "primary_periapse_floor_AU_hex": GATE_THRESHOLDS[
                    "independent_minimum_primary_periapse_AU"
                ].hex(),
                "primary_phase_ceiling_radians_hex": GATE_THRESHOLDS[
                    "independent_maximum_primary_phase_radians"
                ].hex(),
                "primary_periapse_phase_ceiling_radians_hex": GATE_THRESHOLDS[
                    "independent_maximum_primary_periapse_phase_radians"
                ].hex(),
                "bootstrap_argument": (
                    "FIRST_CROSSING_CONTRADICTION_USING_2_SUM_GM_OVER_FLOOR_SQUARED"
                ),
            },
            "tracer_acceleration_bootstrap": (
                _tracer_acceleration_bootstrap_certificate(known_gm)
            ),
        },
        "gate_thresholds_float64_hex": {
            key: value.hex() for key, value in GATE_THRESHOLDS.items()
        },
        "required_gates": [
            "input_epoch_and_initial_state_exact",
            "jx_both_lanes_exact_replay",
            "jx_all_body_absolute_accuracy",
            "jx_outer_absolute_accuracy",
            "jx_timestep_refinement",
            "native_source_map_encounter_and_timestep_guards",
            "independent_all_node_encounter_and_timestep_guards",
            "dop853_ias15_agreement",
            "outer_only_de441_compatibility",
            "ten_year_vs_ten_partition_agreement",
            "energy_and_angular_momentum_conservation",
            "all_node_conservation_for_jx_lanes",
            "input_bytes_stable_through_execution",
            "executed_local_source_archive_retained",
            "runtime_and_code_identity_stable",
            "all_claims_false",
        ],
        "inputs": inputs,
        "code_identity": _code_identity(root),
        "runtime_identity": _runtime_identity(),
        "artifacts": {
            "protocol": "PROTOCOL.json",
            "states": "STATES.npz",
            "result": "RESULT.json",
            "manifest": "ARTIFACTS.json",
            "executed_sources": SOURCE_ARCHIVE_NAME,
            "completion_marker": COMPLETION_MARKER_NAME,
            "create_only": True,
        },
        "claims": dict(CLAIMS),
        "limitations": [
            "Sun plus nine planetary-system barycenters only",
            "Newtonian point masses; no relativity, oblateness, resolved satellites, asteroids, Galactic tide, or stellar passages",
            "ten Julian years only",
            "DE441 inner-body residuals are descriptive and cannot fail the ephemeris gate",
            "no tracer, compact planet, survey model, posterior, detection, exclusion, or scientific inference",
        ],
    }
    if any(protocol["claims"].values()):
        raise M2SourceQualificationError("protocol widened a claim")
    return _content_record(protocol, PROTOCOL_DOMAIN)


def _known_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = KNOWN_SOLAR_SYSTEM_V2
    return (
        np.ascontiguousarray(
            model.positions_primary_relative_ecliptic_AU, dtype=np.float64
        ),
        np.ascontiguousarray(
            model.velocities_primary_relative_ecliptic_AU_per_julian_year,
            dtype=np.float64,
        ),
        np.ascontiguousarray(model.gm_AU3_per_julian_year2, dtype=np.float64),
    )


def _barycentric(
    positions: np.ndarray, velocities: np.ndarray, gm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    total = np.sum(gm, dtype=np.float64)
    position_center = np.sum(positions * gm[:, None], axis=0, dtype=np.float64) / total
    velocity_center = np.sum(velocities * gm[:, None], axis=0, dtype=np.float64) / total
    return (
        np.ascontiguousarray(positions - position_center),
        np.ascontiguousarray(velocities - velocity_center),
    )


def _barycentric_series(
    positions: np.ndarray, velocities: np.ndarray, gm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    position_values = np.asarray(positions, dtype=np.float64)
    velocity_values = np.asarray(velocities, dtype=np.float64)
    if (
        position_values.ndim != 3
        or velocity_values.shape != position_values.shape
        or position_values.shape[1:] != (gm.size, 3)
    ):
        raise M2SourceQualificationError("source series has invalid barycentric shape")
    total = np.sum(gm, dtype=np.float64)
    position_center = np.sum(
        position_values * gm[None, :, None], axis=1, dtype=np.float64
    ) / total
    velocity_center = np.sum(
        velocity_values * gm[None, :, None], axis=1, dtype=np.float64
    ) / total
    return (
        np.ascontiguousarray(position_values - position_center[:, None, :]),
        np.ascontiguousarray(velocity_values - velocity_center[:, None, :]),
    )


def _primary_relative(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values - values[:, 0:1, :])


def _rotate_icrf(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    result[..., 0] = values[..., 0]
    result[..., 1] = (
        J2000_MEAN_OBLIQUITY_COS * values[..., 1]
        + J2000_MEAN_OBLIQUITY_SIN * values[..., 2]
    )
    result[..., 2] = (
        -J2000_MEAN_OBLIQUITY_SIN * values[..., 1]
        + J2000_MEAN_OBLIQUITY_COS * values[..., 2]
    )
    return np.ascontiguousarray(result)


def _read_verified_input_bytes(
    root: Path, relative: Path, receipt: Mapping[str, Any]
) -> bytes:
    payload = _read_repository_file_no_follow(root, relative)
    if (
        receipt.get("path") != relative.as_posix()
        or receipt.get("sha256") != hashlib.sha256(payload).hexdigest()
        or receipt.get("size_bytes") != len(payload)
    ):
        raise M2SourceQualificationError(
            f"input bytes no longer match the protocol: {relative}"
        )
    return payload


def _load_de441_reference(
    root: Path, receipt: Mapping[str, Any] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if receipt is None:
        receipt = _verified_file(root, REFERENCE_VECTOR_PATH, REFERENCE_VECTOR_SHA256)
    payload = _read_verified_input_bytes(root, REFERENCE_VECTOR_PATH, receipt)
    records: dict[tuple[float, int], tuple[np.ndarray, np.ndarray]] = {}
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise M2SourceQualificationError("DE441 reference is not UTF-8") from exc
    with io.StringIO(text, newline="") as stream:
        rows = csv.DictReader(stream)
        for row in rows:
            epoch = float(row["jd_tdb"])
            body_id = int(row["body_id"])
            key = (epoch, body_id)
            if key in records:
                raise M2SourceQualificationError("duplicate DE441 reference row")
            records[key] = (
                np.asarray(
                    [float(row[name]) for name in ("x_au", "y_au", "z_au")],
                    dtype=np.float64,
                ),
                np.asarray(
                    [
                        float(row[name])
                        for name in (
                            "vx_au_per_day",
                            "vy_au_per_day",
                            "vz_au_per_day",
                        )
                    ],
                    dtype=np.float64,
                ),
            )
    epochs_jd = np.asarray(
        [KNOWN_SOLAR_SYSTEM_V2.epoch_jd_tdb + value * JULIAN_YEAR_DAYS for value in ANNUAL_SAMPLE_YEARS],
        dtype=np.float64,
    )
    positions = np.empty((len(epochs_jd), len(KNOWN_SYSTEM_BODY_IDS), 3), dtype=np.float64)
    velocities = np.empty_like(positions)
    for epoch_index, epoch in enumerate(epochs_jd):
        for body_index, body_id in enumerate(KNOWN_SYSTEM_BODY_IDS):
            try:
                position, velocity = records[(float(epoch), body_id)]
            except KeyError as exc:
                raise M2SourceQualificationError(
                    f"missing DE441 row for body {body_id} at {epoch}"
                ) from exc
            positions[epoch_index, body_index] = position
            velocities[epoch_index, body_index] = velocity
    if len(records) != len(epochs_jd) * len(KNOWN_SYSTEM_BODY_IDS):
        raise M2SourceQualificationError("DE441 reference roster is not exact")
    positions = _primary_relative(_rotate_icrf(positions))
    velocities = _primary_relative(_rotate_icrf(velocities)) * JULIAN_YEAR_DAYS
    return epochs_jd, np.ascontiguousarray(positions), np.ascontiguousarray(velocities)


def _provenance() -> Provenance:
    return Provenance(
        "jx.long-horizon.m2-known-source-preflight",
        "Retained NASA/JPL Horizons DE441 state and NAIF DE440 GM known-system model",
        PROVENANCE_VERSION,
        KNOWN_SOLAR_SYSTEM_V2.content_sha256,
    )


def _tracer_acceleration_bootstrap_certificate(gm: np.ndarray) -> dict[str, Any]:
    minimum_radius = float(SOURCE_ENCOUNTER_RADIUS_AU)
    gm_values = np.asarray(gm, dtype=np.float64)
    total_gm_exact = sum(
        (Fraction.from_float(float(value)) for value in gm_values),
        Fraction(0, 1),
    )
    total_gm = float(total_gm_exact)
    exact_bound = (2 * total_gm_exact) / (
        Fraction.from_float(minimum_radius) ** 2
    )
    bound = float(exact_bound)
    if Fraction.from_float(bound) < exact_bound:
        bound = math.nextafter(bound, math.inf)
    if not np.isfinite(bound) or bound <= 0.0:
        raise M2SourceQualificationError("tracer acceleration bootstrap is invalid")
    return {
        "scope": "OUTSIDE_EVERY_DECLARED_SOURCE_ENCOUNTER_SPHERE",
        "derivation": "2*sum(source_gm)/minimum_source_encounter_radius^2",
        "minimum_source_encounter_radius_AU": minimum_radius,
        "sum_source_gm_AU3_per_year2": total_gm,
        "uniform_relative_acceleration_bound_AU_per_year2": bound,
        "qualification_effect": (
            "FAIL_CLOSED_SWEPT_GUARD_ONLY; NOT_TRACER_DYNAMICS_QUALIFICATION"
        ),
    }


def _build_jx_tile(
    *,
    tile_id: str,
    step: float,
    step_count: int,
    global_step: int,
    positions: np.ndarray,
    velocities: np.ndarray,
) -> Any:
    _, _, gm = _known_arrays()
    acceleration_certificate = _tracer_acceleration_bootstrap_certificate(gm)
    acceleration_bound = acceleration_certificate[
        "uniform_relative_acceleration_bound_AU_per_year2"
    ]
    return build_wisdom_holman_source_tile(
        tile_id=tile_id,
        epoch_origin_years=0.0,
        global_source_step=global_step,
        time_scale="TDB",
        frame=KNOWN_SOLAR_SYSTEM_V2.output_frame,
        axes="IAU76_80_ECLIPTIC_J2000",
        source_ids=KNOWN_SYSTEM_BODY_NAMES,
        primary_source_id=KNOWN_SYSTEM_BODY_NAMES[0],
        source_schedule_sha256=_schedule_sha256(step),
        initial_source_relative_positions_AU=np.ascontiguousarray(positions),
        initial_source_relative_velocities_AU_per_year=np.ascontiguousarray(velocities),
        source_gm_AU3_per_year2=gm,
        source_encounter_radii_AU=np.full(
            gm.size, SOURCE_ENCOUNTER_RADIUS_AU, dtype=np.float64
        ),
        certified_relative_acceleration_bounds_AU_per_year2=np.full(
            gm.size, acceleration_bound, dtype=np.float64
        ),
        provenance=_provenance(),
        spec=WisdomHolmanSourceTileSpec(
            step,
            step_count,
            minimum_encounter_pair_separation_AU=1.0e-4,
            minimum_jacobi_periapse_AU=0.05,
        ),
    )


def _sample_tile(build: Any, steps_per_year: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(
        [index * steps_per_year for index in range(HORIZON_YEARS + 1)],
        dtype=np.int64,
    )
    return (
        np.ascontiguousarray(build.tile.source_relative_positions_AU[indices]),
        np.ascontiguousarray(build.tile.source_relative_velocities_AU_per_year[indices]),
    )


def _run_partitioned_jx(
    positions: np.ndarray, velocities: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], float]:
    position_rows = [positions.copy()]
    velocity_rows = [velocities.copy()]
    receipts: list[dict[str, Any]] = []
    current_position = positions
    current_velocity = velocities
    started = time.perf_counter()
    for index in range(PARTITION_COUNT):
        build = _build_jx_tile(
            tile_id=f"m2-known-source-partition-{index:02d}",
            step=COARSE_STEP_YEARS,
            step_count=PARTITION_STEP_COUNT,
            global_step=index * PARTITION_STEP_COUNT,
            positions=current_position,
            velocities=current_velocity,
        )
        current_position = np.ascontiguousarray(
            build.tile.source_relative_positions_AU[-1]
        )
        current_velocity = np.ascontiguousarray(
            build.tile.source_relative_velocities_AU_per_year[-1]
        )
        position_rows.append(current_position)
        velocity_rows.append(current_velocity)
        receipts.append(_jx_receipt(build))
    return (
        np.ascontiguousarray(position_rows),
        np.ascontiguousarray(velocity_rows),
        receipts,
        time.perf_counter() - started,
    )


def _newtonian_rhs(_time_value: float, state: np.ndarray, gm: np.ndarray) -> np.ndarray:
    count = gm.size
    positions = state[: count * 3].reshape(count, 3)
    velocities = state[count * 3 :].reshape(count, 3)
    delta = positions[None, :, :] - positions[:, None, :]
    radius_squared = np.sum(delta * delta, axis=2)
    np.fill_diagonal(radius_squared, np.inf)
    inverse_cube = radius_squared ** -1.5
    acceleration = np.sum(
        delta * inverse_cube[:, :, None] * gm[None, :, None],
        axis=1,
        dtype=np.float64,
    )
    return np.concatenate((velocities.reshape(-1), acceleration.reshape(-1)))


def _run_dop853(
    positions: np.ndarray, velocities: np.ndarray, gm: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], float]:
    from scipy.integrate import solve_ivp

    initial = np.concatenate((positions.reshape(-1), velocities.reshape(-1)))
    started = time.perf_counter()
    solution = solve_ivp(
        lambda t, y: _newtonian_rhs(t, y, gm),
        (0.0, float(HORIZON_YEARS)),
        initial,
        method="DOP853",
        t_eval=np.asarray(ANNUAL_SAMPLE_YEARS, dtype=np.float64),
        rtol=DOP853_RTOL,
        atol=DOP853_ATOL,
        max_step=DOP853_MAX_STEP_YEARS,
    )
    seconds = time.perf_counter() - started
    if not solution.success or solution.status != 0:
        raise M2SourceQualificationError(
            f"DOP853 failed: status={solution.status}, message={solution.message}"
        )
    count = gm.size
    states = solution.y.T
    result_positions = np.ascontiguousarray(states[:, : count * 3].reshape(-1, count, 3))
    result_velocities = np.ascontiguousarray(states[:, count * 3 :].reshape(-1, count, 3))
    return result_positions, result_velocities, {
        "success": bool(solution.success),
        "status": int(solution.status),
        "function_evaluations": int(solution.nfev),
        "jacobian_evaluations": int(solution.njev),
        "lu_decompositions": int(solution.nlu),
    }, seconds


def _run_ias15(
    positions: np.ndarray, velocities: np.ndarray, gm: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], float]:
    import rebound

    if rebound.__version__ != REQUIRED_REBOUND_VERSION:
        raise M2SourceQualificationError("REBOUND version changed during execution")
    simulation = rebound.Simulation()
    simulation.G = 1.0
    simulation.integrator = "ias15"
    simulation.integrator.epsilon = IAS15_EPSILON
    for index in range(gm.size):
        simulation.add(
            m=float(gm[index]),
            x=float(positions[index, 0]),
            y=float(positions[index, 1]),
            z=float(positions[index, 2]),
            vx=float(velocities[index, 0]),
            vy=float(velocities[index, 1]),
            vz=float(velocities[index, 2]),
        )
    output_positions = np.empty((len(ANNUAL_SAMPLE_YEARS), gm.size, 3), dtype=np.float64)
    output_velocities = np.empty_like(output_positions)
    started = time.perf_counter()
    for epoch_index, epoch in enumerate(ANNUAL_SAMPLE_YEARS):
        if epoch_index:
            simulation.integrate(epoch, exact_finish_time=1)
        if simulation.t != epoch:
            raise M2SourceQualificationError("IAS15 did not stop at an exact annual epoch")
        for body_index, particle in enumerate(simulation.particles):
            output_positions[epoch_index, body_index] = (particle.x, particle.y, particle.z)
            output_velocities[epoch_index, body_index] = (particle.vx, particle.vy, particle.vz)
    seconds = time.perf_counter() - started
    if not np.all(np.isfinite(output_positions)) or not np.all(np.isfinite(output_velocities)):
        raise M2SourceQualificationError("IAS15 produced a non-finite state")
    return output_positions, output_velocities, {
        "final_time_hex": float(simulation.t).hex(),
        "ias15_epsilon_hex": float(simulation.integrator.epsilon).hex(),
        "steps_done": int(simulation.steps_done),
    }, seconds


def _invariants(
    positions: np.ndarray,
    velocities: np.ndarray,
    gm: np.ndarray,
    *,
    primary_relative: bool = False,
    sample_scope: str = "RETAINED_REFERENCE_EPOCHS",
) -> dict[str, Any]:
    if type(sample_scope) is not str or not sample_scope:
        raise M2SourceQualificationError("invariant sample_scope is invalid")
    position_values = np.asarray(positions, dtype=np.float64)
    velocity_values = np.asarray(velocities, dtype=np.float64)
    if (
        position_values.ndim != 3
        or velocity_values.shape != position_values.shape
        or position_values.shape[1:] != (gm.size, 3)
        or position_values.shape[0] == 0
        or not np.all(np.isfinite(position_values))
        or not np.all(np.isfinite(velocity_values))
    ):
        raise M2SourceQualificationError("invariant series is invalid")
    energy_chunks: list[np.ndarray] = []
    angular_chunks: list[np.ndarray] = []
    chunk_size = 4096
    for start in range(0, position_values.shape[0], chunk_size):
        stop = min(start + chunk_size, position_values.shape[0])
        position = position_values[start:stop]
        velocity = velocity_values[start:stop]
        if primary_relative:
            position, velocity = _barycentric_series(position, velocity, gm)
        kinetic = 0.5 * np.sum(
            gm[None, :, None] * velocity * velocity,
            axis=(1, 2),
            dtype=np.float64,
        )
        potential = np.zeros(stop - start, dtype=np.float64)
        for left in range(gm.size):
            for right in range(left + 1, gm.size):
                separation = np.linalg.norm(
                    position[:, right, :] - position[:, left, :], axis=1
                )
                if not np.all(np.isfinite(separation)) or np.any(separation <= 0.0):
                    raise M2SourceQualificationError(
                        "invariant series contains a singular pair"
                    )
                potential -= gm[left] * gm[right] / separation
        energy_chunks.append(np.ascontiguousarray(kinetic + potential))
        angular_chunks.append(
            np.ascontiguousarray(
                np.sum(
                    gm[None, :, None] * np.cross(position, velocity),
                    axis=1,
                    dtype=np.float64,
                )
            )
        )
    energy = np.concatenate(energy_chunks)
    angular_array = np.concatenate(angular_chunks, axis=0)
    energy_scale = abs(float(energy[0]))
    angular_scale = float(np.linalg.norm(angular_array[0]))
    if energy_scale == 0.0 or angular_scale == 0.0:
        raise M2SourceQualificationError("conservation scale is zero")
    return {
        "sample_scope": sample_scope,
        "sample_count": int(position_values.shape[0]),
        "initial_energy_hex": float(energy[0]).hex(),
        "initial_angular_momentum_hex": [value.hex() for value in angular_array[0]],
        "maximum_relative_energy_drift": float(
            np.max(np.abs(energy - energy[0])) / energy_scale
        ),
        "maximum_relative_angular_momentum_drift": float(
            np.max(np.linalg.norm(angular_array - angular_array[0], axis=1))
            / angular_scale
        ),
    }


def _independent_source_guard_metrics(
    positions: np.ndarray,
    velocities: np.ndarray,
    gm: np.ndarray,
    step_years: float,
) -> dict[str, Any]:
    """Independently bound pair reachability and primary two-body resolution.

    The source map's native guards are deliberately not reused.  The pair
    bound uses a first-crossing contradiction: while every pair remains above
    the declared floor, each body's acceleration is at most ``sum(GM)/d^2``
    and a pair's relative acceleration is at most twice that value.
    """

    position = np.asarray(positions, dtype=np.float64)
    velocity = np.asarray(velocities, dtype=np.float64)
    if (
        position.ndim != 3
        or velocity.shape != position.shape
        or position.shape[1:] != (gm.size, 3)
        or position.shape[0] < 2
        or not np.all(np.isfinite(position))
        or not np.all(np.isfinite(velocity))
        or not np.isfinite(step_years)
        or step_years == 0.0
    ):
        raise M2SourceQualificationError("independent source-guard input is invalid")
    absolute_step = abs(float(step_years))
    floor = float(INDEPENDENT_PAIR_SEPARATION_FLOOR_AU)
    total_gm_exact = sum(
        (Fraction.from_float(float(value)) for value in gm), Fraction(0, 1)
    )
    exact_pair_bound = (2 * total_gm_exact) / (Fraction.from_float(floor) ** 2)
    pair_acceleration_bound = float(exact_pair_bound)
    if Fraction.from_float(pair_acceleration_bound) < exact_pair_bound:
        pair_acceleration_bound = math.nextafter(pair_acceleration_bound, math.inf)
    acceleration_reach = 0.5 * pair_acceleration_bound * absolute_step**2
    minimum_endpoint = float("inf")
    minimum_path_lower = float("inf")
    pair_count = 0
    for left in range(gm.size):
        for right in range(left + 1, gm.size):
            delta = position[:, right, :] - position[:, left, :]
            relative_velocity = velocity[:, right, :] - velocity[:, left, :]
            distance = np.linalg.norm(delta, axis=1)
            speed = np.linalg.norm(relative_velocity[:-1], axis=1)
            if (
                not np.all(np.isfinite(distance))
                or not np.all(np.isfinite(speed))
                or np.any(distance <= 0.0)
            ):
                raise M2SourceQualificationError(
                    "independent source guard encountered a singular pair"
                )
            minimum_endpoint = min(minimum_endpoint, float(np.min(distance)))
            path_lower = distance[:-1] - speed * absolute_step - acceleration_reach
            minimum_path_lower = min(minimum_path_lower, float(np.min(path_lower)))
            pair_count += 1

    minimum_periapse = float("inf")
    maximum_phase = 0.0
    maximum_periapse_phase = 0.0
    for body in range(1, gm.size):
        relative_position = position[:, body, :] - position[:, 0, :]
        relative_velocity = velocity[:, body, :] - velocity[:, 0, :]
        radius = np.linalg.norm(relative_position, axis=1)
        if not np.all(np.isfinite(radius)) or np.any(radius <= 0.0):
            raise M2SourceQualificationError(
                "independent primary-relative guard encountered a singular radius"
            )
        mu = float(gm[0] + gm[body])
        angular = np.cross(relative_position, relative_velocity)
        angular_squared = np.sum(angular * angular, axis=1, dtype=np.float64)
        velocity_squared = np.sum(
            relative_velocity * relative_velocity, axis=1, dtype=np.float64
        )
        radial_velocity = np.sum(
            relative_position * relative_velocity, axis=1, dtype=np.float64
        )
        eccentricity_vector = (
            (velocity_squared - mu / radius)[:, None] * relative_position
            - radial_velocity[:, None] * relative_velocity
        ) / mu
        eccentricity = np.linalg.norm(eccentricity_vector, axis=1)
        periapse = angular_squared / (mu * (1.0 + eccentricity))
        if not np.all(np.isfinite(periapse)) or np.any(periapse <= 0.0):
            raise M2SourceQualificationError(
                "independent primary-relative periapse is invalid"
            )
        phase = absolute_step * np.sqrt(mu / (radius * radius * radius))
        periapse_phase = absolute_step * np.sqrt(
            mu / (periapse * periapse * periapse)
        )
        if (
            not np.all(np.isfinite(phase))
            or not np.all(np.isfinite(periapse_phase))
        ):
            raise M2SourceQualificationError(
                "independent primary-relative guard is non-finite"
            )
        minimum_periapse = min(minimum_periapse, float(np.min(periapse)))
        maximum_phase = max(maximum_phase, float(np.max(phase)))
        maximum_periapse_phase = max(
            maximum_periapse_phase, float(np.max(periapse_phase))
        )

    return {
        "method": "INDEPENDENT_NUMPY_FIRST_CROSSING_AND_TWO_BODY_GEOMETRY_V1",
        "node_count": int(position.shape[0]),
        "interval_count": int(position.shape[0] - 1),
        "pair_count": pair_count,
        "pair_floor_AU": floor,
        "pair_relative_acceleration_bound_AU_per_year2": pair_acceleration_bound,
        "pair_acceleration_reach_per_interval_AU": acceleration_reach,
        "minimum_pair_endpoint_AU": minimum_endpoint,
        "minimum_pair_path_lower_bound_AU": minimum_path_lower,
        "minimum_primary_two_body_periapse_AU": minimum_periapse,
        "maximum_primary_phase_radians": maximum_phase,
        "maximum_primary_periapse_phase_radians": maximum_periapse_phase,
    }


def _residual_summary(
    left_position: np.ndarray,
    left_velocity: np.ndarray,
    right_position: np.ndarray,
    right_velocity: np.ndarray,
    body_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    indices = tuple(range(len(KNOWN_SYSTEM_BODY_IDS))) if body_indices is None else tuple(body_indices)
    position = np.linalg.norm(
        left_position[:, indices, :] - right_position[:, indices, :], axis=2
    )
    velocity = np.linalg.norm(
        left_velocity[:, indices, :] - right_velocity[:, indices, :], axis=2
    )
    by_body: dict[str, Any] = {}
    for local_index, body_index in enumerate(indices):
        by_body[str(KNOWN_SYSTEM_BODY_IDS[body_index])] = {
            "body_name": KNOWN_SYSTEM_BODY_NAMES[body_index],
            "maximum_position_AU": float(np.max(position[:, local_index])),
            "rms_position_AU": float(np.sqrt(np.mean(position[:, local_index] ** 2))),
            "final_position_AU": float(position[-1, local_index]),
            "maximum_velocity_AU_per_year": float(np.max(velocity[:, local_index])),
            "rms_velocity_AU_per_year": float(np.sqrt(np.mean(velocity[:, local_index] ** 2))),
            "final_velocity_AU_per_year": float(velocity[-1, local_index]),
        }
    return {
        "body_ids": [KNOWN_SYSTEM_BODY_IDS[index] for index in indices],
        "maximum_position_AU": float(np.max(position)),
        "rms_position_AU": float(np.sqrt(np.mean(position * position))),
        "maximum_velocity_AU_per_year": float(np.max(velocity)),
        "rms_velocity_AU_per_year": float(np.sqrt(np.mean(velocity * velocity))),
        "by_body": by_body,
    }


def _jx_receipt(build: Any) -> dict[str, Any]:
    if dict(build.claims) != dict(SOURCE_PROVIDER_CLAIMS) or any(build.claims.values()):
        raise M2SourceQualificationError("source-provider claim ceiling widened")
    acceleration_certificate = _tracer_acceleration_bootstrap_certificate(
        np.ascontiguousarray(build.tile.source_gm_AU3_per_year2)
    )
    expected_bound = acceleration_certificate[
        "uniform_relative_acceleration_bound_AU_per_year2"
    ]
    if not np.array_equal(
        build.tile.source_encounter_radii_AU,
        np.full(
            build.tile.source_gm_AU3_per_year2.size,
            SOURCE_ENCOUNTER_RADIUS_AU,
            dtype=np.float64,
        ),
    ) or not np.array_equal(
        build.tile.certified_relative_acceleration_bounds_AU_per_year2,
        np.full(
            build.tile.source_gm_AU3_per_year2.size,
            expected_bound,
            dtype=np.float64,
        ),
    ):
        raise M2SourceQualificationError(
            "source tile is not bound to the fail-closed acceleration certificate"
        )
    return {
        "tile_sha256": build.tile.content_sha256,
        "input_sha256": build.input_content_sha256,
        "map_input_sha256": build.source_map_input_content_sha256,
        "map_result_sha256": build.source_map_result_content_sha256,
        "map_verification_sha256": build.source_map_verification_content_sha256,
        "native_source_map_metrics": {
            name: value for name, value in build.source_map_metrics
        },
        "tracer_acceleration_bootstrap": acceleration_certificate,
        "primary_seconds": build.source_map_primary_seconds,
        "replay_seconds": build.source_map_replay_seconds,
        "exact_replay_required": bool(build.runtime_identity["exact_replay_required"]),
        "runtime_identity": _plain(build.runtime_identity),
        "claims": dict(build.claims),
    }


def _gate(value: float | bool, threshold: float | bool, operator: str) -> dict[str, Any]:
    if operator == "<=":
        passed = float(value) <= float(threshold)
    elif operator == ">=":
        passed = float(value) >= float(threshold)
    elif operator == "==":
        passed = value == threshold
    else:
        raise M2SourceQualificationError(f"unsupported gate operator: {operator}")
    return {"value": value, "operator": operator, "threshold": threshold, "passed": bool(passed)}


def _evaluate_gates(metrics: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    t = GATE_THRESHOLDS
    coarse = metrics["jx_coarse_vs_dop853_all"]
    fine = metrics["jx_fine_vs_dop853_all"]
    fine_outer = metrics["jx_fine_vs_dop853_outer"]
    cross = metrics["dop853_vs_ias15_all"]
    ephemeris = metrics["jx_fine_vs_de441_outer"]
    partition = metrics["jx_full_vs_partition"]
    invariants = metrics["invariants"]
    native = metrics["native_source_map_guards"]
    independent = metrics["independent_source_guards"]
    position_ratio = coarse["maximum_position_AU"] / max(
        fine["maximum_position_AU"], np.finfo(np.float64).tiny
    )
    velocity_ratio = coarse["maximum_velocity_AU_per_year"] / max(
        fine["maximum_velocity_AU_per_year"], np.finfo(np.float64).tiny
    )
    return {
        "initial_reference_bitwise_exact": _gate(
            metrics["initial_reference_bitwise_exact"], True, "=="
        ),
        "initial_reference_position": _gate(
            metrics["initial_reference"]["maximum_position_AU"],
            t["initial_reference_position_AU"], "<="
        ),
        "initial_reference_velocity": _gate(
            metrics["initial_reference"]["maximum_velocity_AU_per_year"],
            t["initial_reference_velocity_AU_per_year"], "<="
        ),
        "jx_coarse_exact_replay": _gate(metrics["jx_exact_replay"]["coarse"], True, "=="),
        "jx_fine_exact_replay": _gate(metrics["jx_exact_replay"]["fine"], True, "=="),
        "jx_partitions_exact_replay": _gate(metrics["jx_exact_replay"]["partitions"], True, "=="),
        "coarse_native_pair_path": _gate(
            native["coarse"]["minimum_pair_path_lower_bound"],
            t["native_minimum_pair_path_lower_bound_AU"], ">="
        ),
        "fine_native_pair_path": _gate(
            native["fine"]["minimum_pair_path_lower_bound"],
            t["native_minimum_pair_path_lower_bound_AU"], ">="
        ),
        "coarse_native_jacobi_periapse": _gate(
            native["coarse"]["minimum_jacobi_periapse"],
            t["native_minimum_jacobi_periapse_AU"], ">="
        ),
        "fine_native_jacobi_periapse": _gate(
            native["fine"]["minimum_jacobi_periapse"],
            t["native_minimum_jacobi_periapse_AU"], ">="
        ),
        "coarse_native_orbit_step_fraction": _gate(
            native["coarse"]["maximum_orbit_step_fraction"],
            t["native_maximum_orbit_step_fraction"], "<="
        ),
        "fine_native_orbit_step_fraction": _gate(
            native["fine"]["maximum_orbit_step_fraction"],
            t["native_maximum_orbit_step_fraction"], "<="
        ),
        "coarse_native_periapse_step_fraction": _gate(
            native["coarse"]["maximum_periapse_step_fraction"],
            t["native_maximum_periapse_step_fraction"], "<="
        ),
        "fine_native_periapse_step_fraction": _gate(
            native["fine"]["maximum_periapse_step_fraction"],
            t["native_maximum_periapse_step_fraction"], "<="
        ),
        "coarse_independent_pair_path": _gate(
            independent["coarse"]["minimum_pair_path_lower_bound_AU"],
            t["independent_minimum_pair_path_lower_bound_AU"], ">="
        ),
        "fine_independent_pair_path": _gate(
            independent["fine"]["minimum_pair_path_lower_bound_AU"],
            t["independent_minimum_pair_path_lower_bound_AU"], ">="
        ),
        "coarse_independent_primary_periapse": _gate(
            independent["coarse"]["minimum_primary_two_body_periapse_AU"],
            t["independent_minimum_primary_periapse_AU"], ">="
        ),
        "fine_independent_primary_periapse": _gate(
            independent["fine"]["minimum_primary_two_body_periapse_AU"],
            t["independent_minimum_primary_periapse_AU"], ">="
        ),
        "coarse_independent_primary_phase": _gate(
            independent["coarse"]["maximum_primary_phase_radians"],
            t["independent_maximum_primary_phase_radians"], "<="
        ),
        "fine_independent_primary_phase": _gate(
            independent["fine"]["maximum_primary_phase_radians"],
            t["independent_maximum_primary_phase_radians"], "<="
        ),
        "coarse_independent_primary_periapse_phase": _gate(
            independent["coarse"][
                "maximum_primary_periapse_phase_radians"
            ],
            t["independent_maximum_primary_periapse_phase_radians"],
            "<=",
        ),
        "fine_independent_primary_periapse_phase": _gate(
            independent["fine"]["maximum_primary_periapse_phase_radians"],
            t["independent_maximum_primary_periapse_phase_radians"],
            "<=",
        ),
        "tracer_acceleration_bootstrap_exact": _gate(
            metrics["tracer_acceleration_bootstrap_exact"], True, "=="
        ),
        "coarse_all_body_position": _gate(coarse["maximum_position_AU"], t["coarse_all_body_vs_dop853_position_AU"], "<="),
        "coarse_all_body_velocity": _gate(coarse["maximum_velocity_AU_per_year"], t["coarse_all_body_vs_dop853_velocity_AU_per_year"], "<="),
        "fine_all_body_position": _gate(fine["maximum_position_AU"], t["fine_all_body_vs_dop853_position_AU"], "<="),
        "fine_all_body_velocity": _gate(fine["maximum_velocity_AU_per_year"], t["fine_all_body_vs_dop853_velocity_AU_per_year"], "<="),
        "fine_outer_position": _gate(fine_outer["maximum_position_AU"], t["fine_outer_vs_dop853_position_AU"], "<="),
        "fine_outer_velocity": _gate(fine_outer["maximum_velocity_AU_per_year"], t["fine_outer_vs_dop853_velocity_AU_per_year"], "<="),
        "position_refinement_ratio": _gate(position_ratio, t["minimum_position_refinement_ratio"], ">="),
        "velocity_refinement_ratio": _gate(velocity_ratio, t["minimum_velocity_refinement_ratio"], ">="),
        "dop853_ias15_position": _gate(cross["maximum_position_AU"], t["dop853_vs_ias15_position_AU"], "<="),
        "dop853_ias15_velocity": _gate(cross["maximum_velocity_AU_per_year"], t["dop853_vs_ias15_velocity_AU_per_year"], "<="),
        "outer_de441_position": _gate(ephemeris["maximum_position_AU"], t["outer_de441_position_AU"], "<="),
        "outer_de441_velocity": _gate(ephemeris["maximum_velocity_AU_per_year"], t["outer_de441_velocity_AU_per_year"], "<="),
        "partition_position": _gate(partition["maximum_position_AU"], t["partition_position_AU"], "<="),
        "partition_velocity": _gate(partition["maximum_velocity_AU_per_year"], t["partition_velocity_AU_per_year"], "<="),
        "coarse_energy": _gate(invariants["jx_coarse"]["maximum_relative_energy_drift"], t["coarse_relative_energy_drift"], "<="),
        "fine_energy": _gate(invariants["jx_fine"]["maximum_relative_energy_drift"], t["fine_relative_energy_drift"], "<="),
        "coarse_angular": _gate(invariants["jx_coarse"]["maximum_relative_angular_momentum_drift"], t["jx_relative_angular_momentum_drift"], "<="),
        "fine_angular": _gate(invariants["jx_fine"]["maximum_relative_angular_momentum_drift"], t["jx_relative_angular_momentum_drift"], "<="),
        "dop853_energy": _gate(invariants["dop853"]["maximum_relative_energy_drift"], t["reference_relative_energy_drift"], "<="),
        "dop853_angular": _gate(invariants["dop853"]["maximum_relative_angular_momentum_drift"], t["reference_relative_angular_momentum_drift"], "<="),
        "ias15_energy": _gate(invariants["ias15"]["maximum_relative_energy_drift"], t["reference_relative_energy_drift"], "<="),
        "ias15_angular": _gate(invariants["ias15"]["maximum_relative_angular_momentum_drift"], t["reference_relative_angular_momentum_drift"], "<="),
        "runtime_identity_stable": _gate(metrics["runtime_identity_stable"], True, "=="),
        "code_identity_stable": _gate(metrics["code_identity_stable"], True, "=="),
        "input_identity_stable": _gate(metrics["input_identity_stable"], True, "=="),
        "executed_source_archive_retained": _gate(
            metrics["executed_source_archive_retained"], True, "=="
        ),
        "all_claims_false": _gate(not any(CLAIMS.values()), True, "=="),
    }


def _artifact_inventory(root: Path, expected: Sequence[str]) -> dict[str, Any]:
    actual = sorted(
        path.relative_to(root).as_posix() for path in root.iterdir()
    )
    if actual != sorted(expected):
        raise M2SourceQualificationError(f"artifact roster changed: {actual}")
    files = []
    for relative in sorted(expected):
        payload = _read_repository_file_no_follow(root, Path(relative))
        files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    return _content_record(
        {
            "schema": ARTIFACT_SCHEMA,
            "run_id": RUN_ID,
            "files": files,
            "file_count": len(files),
            "claims": dict(CLAIMS),
        },
        ARTIFACT_DOMAIN,
    )


def _completion_record(
    output: Path,
    protocol: Mapping[str, Any],
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return _content_record(
        {
            "schema": COMPLETION_SCHEMA,
            "run_id": RUN_ID,
            "protocol_content_sha256": protocol["content_sha256"],
            "protocol_file_sha256": _sha256_file(output / "PROTOCOL.json"),
            "result_content_sha256": result["content_sha256"],
            "result_file_sha256": _sha256_file(output / "RESULT.json"),
            "artifact_manifest_content_sha256": manifest["content_sha256"],
            "artifact_manifest_file_sha256": _sha256_file(
                output / "ARTIFACTS.json"
            ),
            "result_status": result["status"],
            "all_gates_passed": result["all_gates_passed"],
            "claims": dict(CLAIMS),
            "meaning": (
                "THIS_CREATE_ONLY_ATTEMPT_REACHED_FINAL_PUBLICATION; "
                "ABSENCE_OF_THIS_MARKER_MEANS_INCOMPLETE"
            ),
        },
        COMPLETION_DOMAIN,
    )


def _execute_after_protocol(
    output: Path, root: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    source_archive = _write_source_archive_create_only(
        output / SOURCE_ARCHIVE_NAME, root, protocol["code_identity"]
    )
    relative_position, relative_velocity, gm = _known_arrays()
    barycentric_position, barycentric_velocity = _barycentric(
        relative_position, relative_velocity, gm
    )
    epochs_jd, de441_position, de441_velocity = _load_de441_reference(
        root, protocol["inputs"]["de441_vectors"]
    )

    timings: dict[str, float] = {}
    started = time.perf_counter()
    coarse_build = _build_jx_tile(
        tile_id="m2-known-source-h",
        step=COARSE_STEP_YEARS,
        step_count=COARSE_STEP_COUNT,
        global_step=0,
        positions=relative_position,
        velocities=relative_velocity,
    )
    timings["jx_coarse_seconds"] = time.perf_counter() - started
    coarse_position, coarse_velocity = _sample_tile(coarse_build, 4096)

    started = time.perf_counter()
    fine_build = _build_jx_tile(
        tile_id="m2-known-source-h-over-2",
        step=FINE_STEP_YEARS,
        step_count=FINE_STEP_COUNT,
        global_step=0,
        positions=relative_position,
        velocities=relative_velocity,
    )
    timings["jx_fine_seconds"] = time.perf_counter() - started
    fine_position, fine_velocity = _sample_tile(fine_build, 8192)

    partition_position, partition_velocity, partition_receipts, partition_seconds = _run_partitioned_jx(
        relative_position, relative_velocity
    )
    timings["jx_partition_seconds"] = partition_seconds

    dop_position, dop_velocity, dop_receipt, dop_seconds = _run_dop853(
        barycentric_position, barycentric_velocity, gm
    )
    timings["dop853_seconds"] = dop_seconds
    ias_position, ias_velocity, ias_receipt, ias_seconds = _run_ias15(
        barycentric_position, barycentric_velocity, gm
    )
    timings["ias15_seconds"] = ias_seconds

    dop_relative_position = _primary_relative(dop_position)
    dop_relative_velocity = _primary_relative(dop_velocity)
    ias_relative_position = _primary_relative(ias_position)
    ias_relative_velocity = _primary_relative(ias_velocity)

    initial_reference = _residual_summary(
        relative_position[None, :, :], relative_velocity[None, :, :],
        de441_position[:1], de441_velocity[:1]
    )
    initial_reference_bitwise_exact = bool(
        epochs_jd[0].tobytes()
        == np.float64(KNOWN_SOLAR_SYSTEM_V2.epoch_jd_tdb).tobytes()
        and _array_bitwise_equal(relative_position, de441_position[0])
        and _array_bitwise_equal(relative_velocity, de441_velocity[0])
        and tuple(KNOWN_SYSTEM_BODY_IDS) == KNOWN_SOLAR_SYSTEM_V2.body_ids
    )
    coarse_native_metrics = dict(coarse_build.source_map_metrics)
    fine_native_metrics = dict(fine_build.source_map_metrics)
    coarse_independent_guards = _independent_source_guard_metrics(
        coarse_build.tile.source_relative_positions_AU,
        coarse_build.tile.source_relative_velocities_AU_per_year,
        gm,
        COARSE_STEP_YEARS,
    )
    fine_independent_guards = _independent_source_guard_metrics(
        fine_build.tile.source_relative_positions_AU,
        fine_build.tile.source_relative_velocities_AU_per_year,
        gm,
        FINE_STEP_YEARS,
    )
    acceleration_certificate = _tracer_acceleration_bootstrap_certificate(gm)
    expected_acceleration_bound = acceleration_certificate[
        "uniform_relative_acceleration_bound_AU_per_year2"
    ]
    tracer_acceleration_bootstrap_exact = bool(
        np.array_equal(
            coarse_build.tile.certified_relative_acceleration_bounds_AU_per_year2,
            np.full(gm.size, expected_acceleration_bound, dtype=np.float64),
        )
        and np.array_equal(
            fine_build.tile.certified_relative_acceleration_bounds_AU_per_year2,
            np.full(gm.size, expected_acceleration_bound, dtype=np.float64),
        )
    )
    metrics: dict[str, Any] = {
        "initial_reference": initial_reference,
        "initial_reference_bitwise_exact": initial_reference_bitwise_exact,
        "jx_coarse_vs_dop853_all": _residual_summary(
            coarse_position, coarse_velocity, dop_relative_position, dop_relative_velocity
        ),
        "jx_fine_vs_dop853_all": _residual_summary(
            fine_position, fine_velocity, dop_relative_position, dop_relative_velocity
        ),
        "jx_fine_vs_dop853_outer": _residual_summary(
            fine_position, fine_velocity, dop_relative_position, dop_relative_velocity,
            OUTER_GATE_INDICES,
        ),
        "dop853_vs_ias15_all": _residual_summary(
            dop_relative_position, dop_relative_velocity,
            ias_relative_position, ias_relative_velocity,
        ),
        "jx_fine_vs_de441_outer": _residual_summary(
            fine_position, fine_velocity, de441_position, de441_velocity,
            OUTER_GATE_INDICES,
        ),
        "jx_fine_vs_de441_all_descriptive": _residual_summary(
            fine_position, fine_velocity, de441_position, de441_velocity
        ),
        "jx_full_vs_partition": _residual_summary(
            coarse_position, coarse_velocity, partition_position, partition_velocity
        ),
        "invariants": {
            "jx_coarse": _invariants(
                coarse_build.tile.source_relative_positions_AU,
                coarse_build.tile.source_relative_velocities_AU_per_year,
                gm,
                primary_relative=True,
                sample_scope="ALL_SOURCE_MAP_NODES",
            ),
            "jx_fine": _invariants(
                fine_build.tile.source_relative_positions_AU,
                fine_build.tile.source_relative_velocities_AU_per_year,
                gm,
                primary_relative=True,
                sample_scope="ALL_SOURCE_MAP_NODES",
            ),
            "dop853": _invariants(dop_position, dop_velocity, gm),
            "ias15": _invariants(ias_position, ias_velocity, gm),
        },
        "native_source_map_guards": {
            "coarse": coarse_native_metrics,
            "fine": fine_native_metrics,
        },
        "independent_source_guards": {
            "coarse": coarse_independent_guards,
            "fine": fine_independent_guards,
        },
        "tracer_acceleration_bootstrap": acceleration_certificate,
        "tracer_acceleration_bootstrap_exact": (
            tracer_acceleration_bootstrap_exact
        ),
        "jx_exact_replay": {
            "coarse": bool(coarse_build.runtime_identity["exact_replay_required"]),
            "fine": bool(fine_build.runtime_identity["exact_replay_required"]),
            "partitions": all(row["exact_replay_required"] for row in partition_receipts),
        },
    }

    runtime_end = _runtime_identity()
    code_end = _code_identity(root)
    metrics["runtime_identity_stable"] = runtime_end == protocol["runtime_identity"]
    metrics["code_identity_stable"] = code_end == protocol["code_identity"]
    metrics["input_identity_stable"] = (
        _verified_input_receipts(root) == protocol["inputs"]
    )
    metrics["executed_source_archive_retained"] = bool(
        (output / SOURCE_ARCHIVE_NAME).is_file()
        and _sha256_file(output / SOURCE_ARCHIVE_NAME) == source_archive["sha256"]
        and source_archive["files_content_sha256"]
        == protocol["code_identity"]["files_content_sha256"]
    )
    gates = _evaluate_gates(metrics)
    all_gates_passed = all(record["passed"] for record in gates.values())

    arrays = {
        "sample_years": np.asarray(ANNUAL_SAMPLE_YEARS, dtype=np.float64),
        "epochs_jd_tdb": epochs_jd,
        "gm_AU3_per_year2": gm,
        "jx_coarse_all_node_position_AU": np.ascontiguousarray(
            coarse_build.tile.source_relative_positions_AU
        ),
        "jx_coarse_all_node_velocity_AU_per_year": np.ascontiguousarray(
            coarse_build.tile.source_relative_velocities_AU_per_year
        ),
        "jx_fine_all_node_position_AU": np.ascontiguousarray(
            fine_build.tile.source_relative_positions_AU
        ),
        "jx_fine_all_node_velocity_AU_per_year": np.ascontiguousarray(
            fine_build.tile.source_relative_velocities_AU_per_year
        ),
        "jx_coarse_position_AU": coarse_position,
        "jx_coarse_velocity_AU_per_year": coarse_velocity,
        "jx_fine_position_AU": fine_position,
        "jx_fine_velocity_AU_per_year": fine_velocity,
        "jx_partition_position_AU": partition_position,
        "jx_partition_velocity_AU_per_year": partition_velocity,
        "dop853_barycentric_position_AU": dop_position,
        "dop853_barycentric_velocity_AU_per_year": dop_velocity,
        "ias15_barycentric_position_AU": ias_position,
        "ias15_barycentric_velocity_AU_per_year": ias_velocity,
        "de441_primary_relative_position_AU": de441_position,
        "de441_primary_relative_velocity_AU_per_year": de441_velocity,
    }
    _write_npz_create_only(output / "STATES.npz", arrays)
    state_hashes = {
        key: _sha256_arrays(value) for key, value in arrays.items()
    }
    result = _content_record(
        {
            "schema": RESULT_SCHEMA,
            "run_id": RUN_ID,
            "protocol_content_sha256": protocol["content_sha256"],
            "protocol_file_sha256": _sha256_file(output / "PROTOCOL.json"),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "classification": CLASSIFICATION,
            "status": "PASSED" if all_gates_passed else "INVALID",
            "all_gates_passed": all_gates_passed,
            "gates": gates,
            "metrics": metrics,
            "timings": timings,
            "jx_receipts": {
                "coarse": _jx_receipt(coarse_build),
                "fine": _jx_receipt(fine_build),
                "partitions": partition_receipts,
            },
            "independent_reference_receipts": {
                "dop853": dop_receipt,
                "ias15": ias_receipt,
            },
            "executed_source_archive": source_archive,
            "state_arrays": {
                "path": "STATES.npz",
                "sha256": _sha256_file(output / "STATES.npz"),
                "array_content_sha256": state_hashes,
            },
            "runtime_identity_end": runtime_end,
            "code_identity_end": code_end,
            "claims": dict(CLAIMS),
            "interpretation": (
                "A pass qualifies only this ten-year Newtonian known-source numerical "
                "preflight. It does not qualify tracers, long duration, Planet Nine, "
                "observational inference, detection, or exclusion."
            ),
        },
        RESULT_DOMAIN,
    )
    _write_json_create_only(output / "RESULT.json", result)
    manifest = _artifact_inventory(
        output, (SOURCE_ARCHIVE_NAME, "PROTOCOL.json", "RESULT.json", "STATES.npz")
    )
    _write_json_create_only(output / "ARTIFACTS.json", manifest)
    _write_json_create_only(
        output / COMPLETION_MARKER_NAME,
        _completion_record(output, protocol, result, manifest),
    )
    return result


def _failed_execution_result(
    protocol: Mapping[str, Any], output: Path, error: Exception
) -> dict[str, Any]:
    return _content_record(
        {
            "schema": RESULT_SCHEMA,
            "run_id": RUN_ID,
            "protocol_content_sha256": protocol["content_sha256"],
            "protocol_file_sha256": _sha256_file(output / "PROTOCOL.json"),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "classification": CLASSIFICATION,
            "status": "FAILED_EXECUTION",
            "all_gates_passed": False,
            "gates": {},
            "execution_error": {
                "exception_module": type(error).__module__,
                "exception_type": type(error).__name__,
                "message": str(error),
            },
            "claims": dict(CLAIMS),
            "interpretation": (
                "The bounded execution failed before a numerical verdict. "
                "This retained result authorizes no qualification or science claim."
            ),
        },
        RESULT_DOMAIN,
    )


def run(output_root: Path, project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    protocol = build_protocol(root)
    output = output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    _fsync_directory(output.parent)
    _write_json_create_only(output / "PROTOCOL.json", protocol)
    try:
        return _execute_after_protocol(output, root, protocol)
    except Exception as error:
        if (output / "RESULT.json").exists():
            raise
        failure = _failed_execution_result(protocol, output, error)
        _write_json_create_only(output / "RESULT.json", failure)
        retained = ["PROTOCOL.json", "RESULT.json"]
        if (output / SOURCE_ARCHIVE_NAME).exists():
            retained.append(SOURCE_ARCHIVE_NAME)
        if (output / "STATES.npz").exists():
            retained.append("STATES.npz")
        manifest = _artifact_inventory(output, tuple(retained))
        _write_json_create_only(output / "ARTIFACTS.json", manifest)
        _write_json_create_only(
            output / COMPLETION_MARKER_NAME,
            _completion_record(output, protocol, failure, manifest),
        )
        return failure


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute the bounded 10-year M2 known-source qualification."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        required=True,
        help="required acknowledgement; without it no integration can start",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args(argv)
    result = run(args.output_root, args.project_root)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if result["all_gates_passed"]:
        return 0
    return 2 if result["status"] == "INVALID" else 3


if __name__ == "__main__":
    raise SystemExit(main())
