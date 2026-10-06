#!/usr/bin/env python3
"""Bounded M3 qualification of 108 physical massless tracer states.

M3 is deliberately limited to ten Julian years.  It consumes the retained
M2 fine massive-source path without rebuilding or modifying that path, then
compares two fixed restricted-WH tracer maps with an independent REBOUND
5.1.1 IAS15 integration.  Every tracer is exactly massless and therefore has
no source back-reaction.

There are three intentionally separate modes:

``--prepare-only``
    Emits a deterministic protocol with numerical thresholds unset and every
    execution/claim authorization false.  It performs no integration.

``--calibrate``
    Executes the bounded experiment only to measure errors.  Its caller-
    selected create-only evidence directory is explicitly non-authorizing and
    cannot be placed in the canonical qualification namespace.

``--official``
    Requires the completed calibration evidence, an independently supplied
    content-addressed threshold commitment, and a new unique attempt
    directory.  Evidence is create-only; failures and interruptions remain
    visible, and the completion marker is always the final published artifact.

Importing this module never initializes CUDA or REBOUND.  No mode authorizes
scientific inference, production, long-duration accuracy, Planet Nine claims,
or a staged 1,000-year run.
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
import base64
from collections.abc import Callable, Mapping, Sequence
import ctypes
from datetime import datetime, timezone
import errno
from fractions import Fraction
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any
import zipfile

import numpy as np


PROTOCOL_SCHEMA = "jx.long-horizon.m3-tracer-preflight.protocol.v1"
CALIBRATION_SCHEMA = "jx.long-horizon.m3-tracer-preflight.calibration.v1"
CALIBRATION_COMPLETION_SCHEMA = (
    "jx.long-horizon.m3-tracer-preflight.calibration-completion.v1"
)
THRESHOLD_SCHEMA = "jx.long-horizon.m3-tracer-preflight.thresholds.v1"
RESULT_SCHEMA = "jx.long-horizon.m3-tracer-preflight.result.v1"
ARTIFACT_SCHEMA = "jx.long-horizon.m3-tracer-preflight.artifacts.v1"
COMPLETION_SCHEMA = "jx.long-horizon.m3-tracer-preflight.completion.v1"
RUN_ID = "jx-long-horizon-m3-tracer-preflight-10y-v1"
CLASSIFICATION = "BOUNDED_NUMERICAL_ENGINEERING_QUALIFICATION_ONLY"

PROTOCOL_DOMAIN = b"jx.long-horizon.m3-tracer-preflight.protocol.v1\0"
CALIBRATION_DOMAIN = b"jx.long-horizon.m3-tracer-preflight.calibration.v1\0"
CALIBRATION_COMPLETION_DOMAIN = (
    b"jx.long-horizon.m3-tracer-preflight.calibration-completion.v1\0"
)
THRESHOLD_DOMAIN = b"jx.long-horizon.m3-tracer-preflight.thresholds.v1\0"
RESULT_DOMAIN = b"jx.long-horizon.m3-tracer-preflight.result.v1\0"
ARTIFACT_DOMAIN = b"jx.long-horizon.m3-tracer-preflight.artifacts.v1\0"
COMPLETION_DOMAIN = b"jx.long-horizon.m3-tracer-preflight.completion.v1\0"
ARRAY_DOMAIN = b"jx.long-horizon.m3-tracer-preflight.arrays.v1\0"
EVENT_DOMAIN = b"jx.long-horizon.m3-tracer-preflight.events.v1\0"
SOURCE_CUSTODY_DOMAIN = b"jx.long-horizon.m3-tracer-preflight.source-custody.v1\0"
TRACE_NODE_DOMAIN = b"jx.long-horizon.m3-tracer-preflight.trace-node.v1\0"
TRACE_ROSTER_DOMAIN = b"jx.long-horizon.m3-tracer-preflight.trace-roster.v1\0"
TRACE_CUSTODY_DOMAIN = b"jx.long-horizon.m3-tracer-preflight.trace-custody.v1\0"
CODE_ARCHIVE_SCHEMA = "jx.long-horizon.m3-tracer-preflight.executed-sources.v1"
CODE_FILES_DOMAIN = b"jx.long-horizon.m3-tracer-preflight.code-files.v2\0"
RUNTIME_DISTRIBUTION_DOMAIN = (
    b"jx.long-horizon.m3-tracer-preflight.runtime-distribution.v1\0"
)
RUNTIME_IDENTITY_DOMAIN = (
    b"jx.long-horizon.m3-tracer-preflight.runtime-identity.v1\0"
)

HORIZON_YEARS = 10
SPLIT_EPOCH_YEARS = 5
STATE_COUNT = 108
SOURCE_COUNT = 10
SOURCE_STEP_YEARS = 1.0 / 8192.0
SOURCE_STEPS_PER_YEAR = 8192
SOURCE_STEP_COUNT = HORIZON_YEARS * SOURCE_STEPS_PER_YEAR
CANDIDATE_SOURCE_STEPS_PER_MAP = 16
REFINEMENT_SOURCE_STEPS_PER_MAP = 8
CANDIDATE_STEP_YEARS = 1.0 / 512.0
REFINEMENT_STEP_YEARS = 1.0 / 1024.0
CANDIDATE_MAP_STEP_COUNT = HORIZON_YEARS * 512
REFINEMENT_MAP_STEP_COUNT = HORIZON_YEARS * 1024
SPLIT_CANDIDATE_MAP_STEP = SPLIT_EPOCH_YEARS * 512
SPLIT_REFINEMENT_MAP_STEP = SPLIT_EPOCH_YEARS * 1024

# Retained only as non-authorizing context for the precision-first step
# decision.  These values came from the caller-approved /tmp calibration and
# are not thresholds, qualification evidence, or an execution authorization.
PRECURSOR_CALIBRATION_CONTEXT = {
    "status": "NON_AUTHORIZING_EPHEMERAL_CONTEXT_ONLY",
    "legacy_candidate_step_years_hex": (1.0 / 256.0).hex(),
    "legacy_refinement_step_years_hex": (1.0 / 512.0).hex(),
    "legacy_pair_max_position_AU_hex": (3.256583475914143e-08).hex(),
    "legacy_pair_max_velocity_AU_per_year_hex": (8.908323450418738e-09).hex(),
    "legacy_refinement_vs_ias15": {
        "position_vector_l2_AU_hex": (1.0855568637512103e-08).hex(),
        "velocity_vector_l2_AU_per_year_hex": (2.9687615298070667e-09).hex(),
        "perihelion_AU_hex": (3.033371882565916e-07).hex(),
        "eccentricity_hex": (9.725124483495051e-09).hex(),
        "inclination_rad_hex": (7.953671610216873e-10).hex(),
    },
    "selected_refinement_step_years_hex": REFINEMENT_STEP_YEARS.hex(),
    "selected_refinement_vs_ias15": {
        "position_vector_l2_AU_hex": (2.71416719739234e-09).hex(),
        "velocity_vector_l2_AU_per_year_hex": (7.421643268353379e-10).hex(),
        "perihelion_AU_hex": (7.582949734796784e-08).hex(),
        "eccentricity_hex": (2.431165269101143e-09).hex(),
        "inclination_rad_hex": (1.988305631250853e-10).hex(),
    },
    "observed_routes": 0,
    "observed_removals": 0,
    "official_thresholds_selected": False,
    "official_execution_authorized": False,
}

REQUIRED_REBOUND_VERSION = "5.1.1"
IAS15_EPSILON = 1.0e-15
IAS15_ADAPTIVE_MODE = "PRS23"
IAS15_INITIAL_STEP_YEARS = 1.0e-3
ALGORITHMIC_GUARD_RADIUS_AU = 0.1
PHYSICAL_TERMINAL_RADIUS_AU = 0.0
PRIMARY_OUTER_BOUNDARY_AU = 10_000.0
MAXIMUM_PRIMARY_DYNAMICAL_PHASE = 1.0 / 64.0
OUTWARD_RADIAL_VELOCITY_TOLERANCE_AU_PER_YEAR = 1.0e-13
ROUTE_WORKER_COUNT = 6
THREADS_PER_BLOCK = 128

DEFAULT_CANONICAL_ROOT = Path("runs/long_horizon_m3_tracer_preflight_v1")
SOURCE_ARCHIVE_NAME = "EXECUTED_SOURCES.zip"
ARRAY_ARCHIVE_NAME = "STATES.npz"
EVENT_LEDGER_NAME = "EVENTS.json"
PROTOCOL_NAME = "PROTOCOL.json"
RESULT_NAME = "RESULT.json"
ARTIFACT_NAME = "ARTIFACTS.json"
COMPLETION_NAME = "COMPLETED.json"
CALIBRATION_RECORD_NAME = "CALIBRATION.json"
CALIBRATION_COMPLETION_NAME = "CALIBRATION_COMPLETED.json"
OFFICIAL_CALIBRATION_COPY_NAMES = (
    (PROTOCOL_NAME, "CALIBRATION_PROTOCOL.json"),
    (ARRAY_ARCHIVE_NAME, "CALIBRATION_STATES.npz"),
    (EVENT_LEDGER_NAME, "CALIBRATION_EVENTS.json"),
    (SOURCE_ARCHIVE_NAME, "CALIBRATION_EXECUTED_SOURCES.zip"),
    (CALIBRATION_RECORD_NAME, "CALIBRATION_RECEIPT.json"),
    (CALIBRATION_COMPLETION_NAME, "CALIBRATION_COMPLETED.json"),
)

COMPARISON_METRICS = (
    "position_vector_l2_AU",
    "velocity_vector_l2_AU_per_year",
    "semimajor_axis_AU",
    "perihelion_AU",
    "eccentricity",
    "inclination_rad",
)
COMPARISON_IDS = (
    "candidate_vs_refinement",
    "refinement_vs_rebound_ias15",
)

# Conservative binary64 enclosure for the short, explicit three-component
# norm/reach expressions used by the independent guard.  The factor is much
# wider than the operation-count gamma bound (subtractions, 3 squares, 2 adds,
# sqrt and scale operations) and is applied in the adverse direction before a
# final nextafter.  Inputs outside the finite physical scale fail closed.
_GUARD_ROUNDING_PAD = 256.0 * sys.float_info.epsilon
_GUARD_ORBITAL_ELEMENT_PAD = 4096.0 * sys.float_info.epsilon
_GUARD_MAXIMUM_ABSOLUTE_COMPONENT = 1.0e8

CLAIMS = {
    "complete_de441_force_model": False,
    "detection_claim": False,
    "encounter_physics_qualified": False,
    "exclusion_claim": False,
    "four_point_five_gyr_accuracy_qualified": False,
    "inference_authorized": False,
    "long_duration_accuracy_qualified": False,
    "observationally_fitted": False,
    "planet_nine_modeled": False,
    "production_qualified": False,
    "science_result": False,
    "staged_one_thousand_year_run_authorized": False,
    "tracer_dynamics_qualified": False,
    "uncertainty_calibrated": False,
}

_THRESHOLD_KEYS = {
    "schema",
    "protocol_content_sha256",
    "calibration_content_sha256",
    "review_id",
    "authorized_attempt_id",
    "comparisons",
    "fixed_gates",
    "authorized_for_one_official_attempt",
    "staged_one_thousand_year_run_authorized",
    "claims",
    "content_sha256",
}
_ATTEMPT_ID_PATTERN = re.compile(r"attempt-[a-z0-9][a-z0-9-]{0,62}\Z")
_REVIEW_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")

CODE_PATHS = (
    Path("benchmarks/jx_long_horizon_m3_tracer_preflight.py"),
    Path("src/jxplanetx/__init__.py"),
    Path("src/jxplanetx/long_horizon/known_system_inputs.py"),
    Path("src/jxplanetx/long_horizon/m2_source_path.py"),
    Path("src/jxplanetx/long_horizon/initial_conditions.py"),
    Path("src/jxplanetx/long_horizon/physical_models.py"),
    Path("src/jxplanetx/cuda_restricted_wh.py"),
    Path("src/jxplanetx/cuda_restricted_wh_hybrid.py"),
    Path("src/jxplanetx/restricted_wh_lifecycle.py"),
    Path("src/jxplanetx/restricted_wh_route.py"),
    Path("src/jxplanetx/restricted_wh_source.py"),
    Path("src/jxplanetx/restricted_wh_checkpoint.py"),
    Path("src/jxplanetx/cuda_tracer_events.py"),
    Path("src/jxplanetx/_all_conic_kepler.py"),
)

LOCAL_EVIDENCE_SUFFIXES = frozenset(
    (".c", ".cc", ".cpp", ".cu", ".h", ".hpp", ".json", ".md", ".csv", ".py", ".pyx", ".so")
)
LOCAL_EVIDENCE_BASENAMES = frozenset(("SHA256SUMS",))
NATIVE_DISTRIBUTION_MARKERS = (".so", ".pyd", ".dll", ".dylib")
NUMERICAL_LOCAL_MODULES = (
    "jxplanetx._all_conic_kepler",
    "jxplanetx.cuda_restricted_wh",
    "jxplanetx.cuda_restricted_wh_hybrid",
    "jxplanetx.cuda_tracer_events",
    "jxplanetx.engine.backends",
    "jxplanetx.engine.contracts",
    "jxplanetx.fast_wisdom_holman",
    "jxplanetx.long_horizon",
    "jxplanetx.long_horizon.known_system_inputs",
    "jxplanetx.long_horizon.m2_source_path",
    "jxplanetx.restricted_wh_checkpoint",
    "jxplanetx.restricted_wh_lifecycle",
    "jxplanetx.restricted_wh_route",
    "jxplanetx.restricted_wh_source",
)


class M3TracerPreflightError(RuntimeError):
    """The M3 preflight failed closed."""


class M3ProtocolError(M3TracerPreflightError, ValueError):
    """A protocol, threshold, or evidence request is invalid."""


class M3NumericalError(M3TracerPreflightError):
    """A bounded numerical lane could not produce valid evidence."""


def _require_nonzero_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value == "0" * 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise M3ProtocolError(f"{label} must be one nonzero lowercase SHA-256")
    return value


def _require_attempt_id(value: object) -> str:
    if type(value) is not str or _ATTEMPT_ID_PATTERN.fullmatch(value) is None:
        raise M3ProtocolError(
            "attempt ID must match exact attempt-[a-z0-9-] syntax"
        )
    return value


def _require_review_id(value: object) -> str:
    if type(value) is not str or _REVIEW_ID_PATTERN.fullmatch(value) is None:
        raise M3ProtocolError("review_id is not one normalized explicit identifier")
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if type(value) is float:
        if not math.isfinite(value):
            raise M3ProtocolError("evidence contains a nonfinite float")
        return value
    if value is None or type(value) in (str, bool, int):
        return value
    raise M3ProtocolError(
        f"evidence contains unsupported {type(value).__name__}"
    )


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            _plain(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError) as exc:
        raise M3ProtocolError("value is not strict canonical JSON") from exc


def _content_sha256(domain: bytes, value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("content_sha256", None)
    return hashlib.sha256(domain + _canonical_json(unsigned)).hexdigest()


def _content_record(
    value: Mapping[str, Any], domain: bytes
) -> dict[str, Any]:
    result = dict(_plain(value))
    result["content_sha256"] = _content_sha256(domain, result)
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(ARRAY_DOMAIN)
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.ndim.to_bytes(4, "little"))
    for dimension in array.shape:
        digest.update(int(dimension).to_bytes(8, "little"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _arrays_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256(ARRAY_DOMAIN + b"bundle\0")
    for name in sorted(arrays):
        if type(name) is not str or not name:
            raise M3ProtocolError("array names must be nonempty strings")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(bytes.fromhex(_array_sha256(arrays[name])))
    return digest.hexdigest()


def _trace_node_custody(
    *,
    lane_id: str,
    position_AU: np.ndarray,
    velocity_AU_per_year: np.ndarray,
    lifecycle: np.ndarray,
    state_ids: Sequence[str],
    source_steps_per_map_step: int,
) -> dict[str, Any]:
    """Hash every retained map node from independently replayable arrays."""

    if lane_id not in ("candidate", "refinement"):
        raise M3ProtocolError("trace lane is invalid")
    if (
        type(position_AU) is not np.ndarray
        or type(velocity_AU_per_year) is not np.ndarray
        or type(lifecycle) is not np.ndarray
        or position_AU.dtype != np.dtype(np.float64)
        or velocity_AU_per_year.dtype != np.dtype(np.float64)
        or lifecycle.dtype != np.dtype(np.uint8)
        or position_AU.shape != velocity_AU_per_year.shape
        or position_AU.ndim != 3
        or position_AU.shape[1:] != (STATE_COUNT, 3)
        or lifecycle.shape != position_AU.shape[:2]
        or type(source_steps_per_map_step) is not int
        or source_steps_per_map_step <= 0
        or len(state_ids) != STATE_COUNT
        or any(type(value) is not str or not value for value in state_ids)
        or len(set(state_ids)) != STATE_COUNT
    ):
        raise M3ProtocolError("trace-node custody input is malformed")
    roster_sha256 = hashlib.sha256(
        TRACE_ROSTER_DOMAIN + _canonical_json(list(state_ids))
    ).hexdigest()
    node_sha256: list[str] = []
    for map_index in range(position_AU.shape[0]):
        node_sha256.append(
            hashlib.sha256(
                TRACE_NODE_DOMAIN
                + _canonical_json(
                    {
                        "lane_id": lane_id,
                        "map_index": map_index,
                        "global_source_step": (
                            map_index * source_steps_per_map_step
                        ),
                        "state_roster_sha256": roster_sha256,
                        "position_AU_sha256": _array_sha256(
                            position_AU[map_index]
                        ),
                        "velocity_AU_per_year_sha256": _array_sha256(
                            velocity_AU_per_year[map_index]
                        ),
                        "lifecycle_sha256": _array_sha256(
                            lifecycle[map_index]
                        ),
                    }
                )
            ).hexdigest()
        )
    return _content_record(
        {
            "lane_id": lane_id,
            "node_count": len(node_sha256),
            "state_roster_sha256": roster_sha256,
            "source_steps_per_map_step": source_steps_per_map_step,
            "first_global_source_step": 0,
            "final_global_source_step": (
                (len(node_sha256) - 1) * source_steps_per_map_step
            ),
            "node_sha256": node_sha256,
        },
        TRACE_CUSTODY_DOMAIN,
    )


def _bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(
        left.shape == right.shape
        and left.dtype == right.dtype
        and np.array_equal(
            np.ascontiguousarray(left).view(np.uint8),
            np.ascontiguousarray(right).view(np.uint8),
        )
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_absolute_no_follow_create_only_open(
    path: Path, label: str
) -> tuple[Path, int]:
    """Atomically publish and retain one new directory through its parent FD."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    if not absolute.name or absolute.name in (".", ".."):
        raise M3ProtocolError(f"invalid directory destination: {label}")
    parent_descriptor = _open_absolute_directory_no_follow(
        absolute.parent, f"{label} parent"
    )
    try:
        try:
            created = _create_directory_at_no_replace_open(
                parent_descriptor, absolute.name, label
            )
        except FileExistsError:
            raise M3ProtocolError(f"{label} must be a new path") from None
        except M3ProtocolError:
            raise
        except OSError as exc:
            raise M3ProtocolError(f"cannot create {label}") from exc
    finally:
        os.close(parent_descriptor)
    return absolute, created


def _rename_directory_no_replace_at(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
    label: str,
) -> None:
    """Atomically rename one directory without replacing an existing target."""

    try:
        function = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:
        raise M3ProtocolError(
            f"atomic no-replace directory publication is unavailable: {label}"
        ) from exc
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_int
    result = function(
        parent_descriptor,
        os.fsencode(source_name),
        parent_descriptor,
        os.fsencode(destination_name),
        1,  # Linux RENAME_NOREPLACE.
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(destination_name)
    raise M3ProtocolError(
        f"cannot atomically publish directory: {label}"
    ) from OSError(error_number, os.strerror(error_number))


def _create_directory_at_no_replace_open(
    parent_descriptor: int, destination_name: str, label: str
) -> int:
    """Create privately, open, and atomically publish one child directory."""

    if (
        type(destination_name) is not str
        or not destination_name
        or Path(destination_name).name != destination_name
        or destination_name in (".", "..")
    ):
        raise M3ProtocolError(f"invalid directory destination: {label}")
    staging_name = f".{destination_name}.{os.urandom(16).hex()}.staging"
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor: int | None = None
    published = False
    try:
        os.mkdir(staging_name, mode=0o700, dir_fd=parent_descriptor)
        descriptor = os.open(staging_name, flags, dir_fd=parent_descriptor)
        _rename_directory_no_replace_at(
            parent_descriptor,
            staging_name,
            destination_name,
            label,
        )
        published = True
        visible = os.open(destination_name, flags, dir_fd=parent_descriptor)
        try:
            held = os.fstat(descriptor)
            current = os.fstat(visible)
            if held.st_dev != current.st_dev or held.st_ino != current.st_ino:
                raise M3ProtocolError(
                    f"published directory identity changed: {label}"
                )
        finally:
            os.close(visible)
        os.fchmod(descriptor, 0o755)
        os.fsync(descriptor)
        os.fsync(parent_descriptor)
        result = descriptor
        descriptor = None
        return result
    except FileExistsError:
        raise
    except M3ProtocolError:
        raise
    except OSError as exc:
        raise M3ProtocolError(
            f"cannot create and publish directory: {label}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if not published:
            try:
                os.rmdir(staging_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
            except OSError:
                # A hostile replacement is deliberately left visible rather
                # than recursively removed; the caller fails closed.
                pass


def _mkdir_absolute_no_follow_create_only(path: Path, label: str) -> Path:
    """Create one directory through a no-follow parent descriptor."""

    absolute, descriptor = _mkdir_absolute_no_follow_create_only_open(path, label)
    os.close(descriptor)
    return absolute


def _require_directory_path_identity(
    path: Path, directory_descriptor: int, label: str
) -> None:
    """Require a canonical path to still name one retained directory inode."""

    path_descriptor = _open_absolute_directory_no_follow(path, label)
    try:
        held = os.fstat(directory_descriptor)
        resolved = os.fstat(path_descriptor)
        if (
            held.st_dev != resolved.st_dev
            or held.st_ino != resolved.st_ino
        ):
            raise M3ProtocolError(f"{label} directory identity changed")
    finally:
        os.close(path_descriptor)


def _commit_create_only(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError:
        raise M3ProtocolError(
            f"refusing to overwrite evidence artifact: {destination}"
        ) from None
    temporary.unlink()
    _fsync_directory(destination.parent)


def _write_bytes_create_only(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise M3ProtocolError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _commit_create_only(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _write_bytes_at_create_only(
    directory_descriptor: int, name: str, payload: bytes
) -> None:
    """Atomically publish one flat artifact through a retained directory FD."""

    if (
        type(name) is not str
        or not name
        or Path(name).name != name
        or name in (".", "..")
        or type(payload) is not bytes
    ):
        raise M3ProtocolError("invalid descriptor-relative artifact publication")
    temporary_name = f".{name}.{os.urandom(16).hex()}.tmp"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary_name, flags, 0o600, dir_fd=directory_descriptor
        )
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise M3ProtocolError("artifact publication made no progress")
            written += count
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(
                temporary_name,
                name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            raise M3ProtocolError(
                f"refusing to overwrite evidence artifact: {name}"
            ) from None
        os.unlink(temporary_name, dir_fd=directory_descriptor)
        os.fsync(directory_descriptor)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass
        raise


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _plain(value), indent=2, sort_keys=True, allow_nan=False
    ).encode("ascii") + b"\n"


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        **{
            name: np.ascontiguousarray(value)
            for name, value in arrays.items()
        },
    )
    return buffer.getvalue()


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes_create_only(path, _json_bytes(value))


def _write_npz_create_only(
    path: Path, arrays: Mapping[str, np.ndarray]
) -> None:
    _write_bytes_create_only(path, _npz_bytes(arrays))


def _decode_json_object(payload: bytes, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise M3ProtocolError(f"duplicate JSON key is forbidden: {key}")
            result[key] = item
        return result

    def reject_nonfinite(token: str) -> None:
        raise M3ProtocolError(f"nonfinite JSON token is forbidden: {token}")

    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise M3ProtocolError(f"cannot parse strict JSON object: {label}") from exc
    if type(value) is not dict:
        raise M3ProtocolError("JSON evidence root must be an object")
    return value


def _read_json(path: Path, *, maximum_bytes: int = 8_000_000) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise M3ProtocolError(f"required regular JSON file is absent: {path}")
    size = path.stat().st_size
    if not 0 < size <= maximum_bytes:
        raise M3ProtocolError(f"JSON file has an invalid bounded size: {path}")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise M3ProtocolError(f"cannot read JSON file: {path}") from exc
    if len(payload) != size:
        raise M3ProtocolError(f"JSON file changed while reading: {path}")
    return _decode_json_object(payload, str(path))


def _read_npz_arrays(
    path: Path, *, maximum_bytes: int = 1_000_000_000
) -> dict[str, np.ndarray]:
    if path.is_symlink() or not path.is_file():
        raise M3ProtocolError(f"required regular NPZ file is absent: {path}")
    size = path.stat().st_size
    if not 0 < size <= maximum_bytes:
        raise M3ProtocolError(f"NPZ file has an invalid bounded size: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise M3ProtocolError("NPZ contains duplicate member names")
            arrays = {
                name: np.ascontiguousarray(archive[name])
                for name in archive.files
            }
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise M3ProtocolError(f"cannot parse strict NPZ file: {path}") from exc
    return arrays


def _git_value(project_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _open_absolute_directory_no_follow(path: Path, label: str) -> int:
    """Open an absolute directory by descriptor without following any link."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    if not absolute.is_absolute() or ".." in absolute.parts:
        raise M3ProtocolError(f"invalid absolute directory path: {label}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(os.sep, flags)
    except OSError as exc:
        raise M3ProtocolError(f"cannot open filesystem root: {label}") from exc
    try:
        for component in absolute.parts[1:]:
            if component in ("", ".", ".."):
                raise M3ProtocolError(
                    f"invalid directory component in path: {label}"
                )
            try:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                raise M3ProtocolError(
                    f"directory path is absent or linked: {label}"
                ) from exc
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_regular_file_at_no_follow(
    directory_descriptor: int, relative: Path, label: str
) -> tuple[bytes, os.stat_result]:
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise M3ProtocolError(f"invalid relative regular-file path: {label}")
    current = os.dup(directory_descriptor)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        for component in relative.parts[:-1]:
            if component in ("", ".", ".."):
                raise M3ProtocolError(
                    f"invalid directory component in file path: {label}"
                )
            try:
                next_descriptor = os.open(
                    component, directory_flags, dir_fd=current
                )
            except OSError as exc:
                raise M3ProtocolError(
                    f"file path is absent or linked: {label}"
                ) from exc
            os.close(current)
            current = next_descriptor
        final_name = relative.parts[-1]
        if final_name in ("", ".", ".."):
            raise M3ProtocolError(f"invalid regular-file name: {label}")
        try:
            descriptor = os.open(
                final_name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=current,
            )
        except OSError as exc:
            raise M3ProtocolError(
                f"regular file is absent or linked: {label}"
            ) from exc
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise M3ProtocolError(f"path is not a regular file: {label}")
            chunks: list[bytes] = []
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                chunks.append(block)
            finished = os.fstat(descriptor)
            if (
                opened.st_dev != finished.st_dev
                or opened.st_ino != finished.st_ino
                or opened.st_size != finished.st_size
                or opened.st_mtime_ns != finished.st_mtime_ns
                or opened.st_ctime_ns != finished.st_ctime_ns
            ):
                raise M3ProtocolError(f"file changed while reading: {label}")
            payload = b"".join(chunks)
            if len(payload) != finished.st_size:
                raise M3ProtocolError(f"file size changed while reading: {label}")
            return payload, finished
        finally:
            os.close(descriptor)
    finally:
        os.close(current)


def _read_absolute_file_no_follow(
    path: Path, label: str
) -> tuple[bytes, os.stat_result]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    parent_descriptor = _open_absolute_directory_no_follow(
        absolute.parent, label
    )
    try:
        return _read_regular_file_at_no_follow(
            parent_descriptor, Path(absolute.name), label
        )
    finally:
        os.close(parent_descriptor)


def _read_exact_directory_payloads_no_follow(
    path: Path,
    expected_names: set[str] | frozenset[str],
    label: str,
    *,
    maximum_file_bytes: int = 2_000_000_000,
) -> dict[str, bytes]:
    """Read an exact flat evidence directory through one stable directory FD."""

    expected = set(expected_names)
    if (
        not expected
        or type(maximum_file_bytes) is not int
        or maximum_file_bytes <= 0
        or any(
            type(name) is not str
            or not name
            or Path(name).name != name
            or name in (".", "..")
            for name in expected
        )
    ):
        raise M3ProtocolError(f"invalid expected directory roster: {label}")
    descriptor = _open_absolute_directory_no_follow(path, label)
    try:
        opened_directory = os.fstat(descriptor)
        try:
            before = os.listdir(descriptor)
        except OSError as exc:
            raise M3ProtocolError(
                f"cannot enumerate evidence directory: {label}"
            ) from exc
        if len(before) != len(set(before)) or set(before) != expected:
            raise M3ProtocolError(f"evidence directory roster changed: {label}")
        payloads: dict[str, bytes] = {}
        for name in sorted(expected):
            payload, metadata = _read_regular_file_at_no_follow(
                descriptor, Path(name), f"{label}/{name}"
            )
            if not 0 < metadata.st_size <= maximum_file_bytes:
                raise M3ProtocolError(
                    f"evidence file has an invalid bounded size: {label}/{name}"
                )
            payloads[name] = payload
        try:
            after = os.listdir(descriptor)
        except OSError as exc:
            raise M3ProtocolError(
                f"cannot re-enumerate evidence directory: {label}"
            ) from exc
        finished_directory = os.fstat(descriptor)
        if (
            set(before) != set(after)
            or opened_directory.st_dev != finished_directory.st_dev
            or opened_directory.st_ino != finished_directory.st_ino
            or opened_directory.st_mtime_ns != finished_directory.st_mtime_ns
            or opened_directory.st_ctime_ns != finished_directory.st_ctime_ns
        ):
            raise M3ProtocolError(
                f"evidence directory changed while reading: {label}"
            )
        return payloads
    finally:
        os.close(descriptor)


def _read_allowed_directory_payloads_at_no_follow(
    directory_descriptor: int,
    allowed_names: set[str] | frozenset[str],
    label: str,
    *,
    maximum_file_bytes: int = 2_000_000_000,
) -> dict[str, bytes]:
    """Read a flat directory through one retained, no-follow descriptor."""

    allowed = set(allowed_names)
    if not allowed or any(
        type(name) is not str
        or not name
        or Path(name).name != name
        or name in (".", "..")
        for name in allowed
    ):
        raise M3ProtocolError(f"invalid allowed directory roster: {label}")
    descriptor = os.dup(directory_descriptor)
    try:
        opened_directory = os.fstat(descriptor)
        try:
            before = os.listdir(descriptor)
        except OSError as exc:
            raise M3ProtocolError(
                f"cannot enumerate evidence directory: {label}"
            ) from exc
        observed = set(before)
        if (
            len(before) != len(observed)
            or not observed
            or not observed <= allowed
        ):
            raise M3ProtocolError(f"evidence directory roster is invalid: {label}")
        payloads: dict[str, bytes] = {}
        for name in sorted(observed):
            payload, metadata = _read_regular_file_at_no_follow(
                descriptor, Path(name), f"{label}/{name}"
            )
            if not 0 < metadata.st_size <= maximum_file_bytes:
                raise M3ProtocolError(
                    f"evidence file has an invalid bounded size: {label}/{name}"
                )
            payloads[name] = payload
        try:
            after = os.listdir(descriptor)
        except OSError as exc:
            raise M3ProtocolError(
                f"cannot re-enumerate evidence directory: {label}"
            ) from exc
        finished_directory = os.fstat(descriptor)
        if (
            set(before) != set(after)
            or opened_directory.st_dev != finished_directory.st_dev
            or opened_directory.st_ino != finished_directory.st_ino
            or opened_directory.st_mtime_ns != finished_directory.st_mtime_ns
            or opened_directory.st_ctime_ns != finished_directory.st_ctime_ns
        ):
            raise M3ProtocolError(
                f"evidence directory changed while reading: {label}"
            )
        return payloads
    finally:
        os.close(descriptor)


def _read_allowed_directory_payloads_no_follow(
    path: Path,
    allowed_names: set[str] | frozenset[str],
    label: str,
    *,
    maximum_file_bytes: int = 2_000_000_000,
) -> dict[str, bytes]:
    """Read a flat directory whose exact status-dependent roster is checked later."""

    descriptor = _open_absolute_directory_no_follow(path, label)
    try:
        return _read_allowed_directory_payloads_at_no_follow(
            descriptor,
            allowed_names,
            label,
            maximum_file_bytes=maximum_file_bytes,
        )
    finally:
        os.close(descriptor)


def _strict_zip_infos(
    archive: zipfile.ZipFile,
    label: str,
    *,
    maximum_total_uncompressed_bytes: int = 8_000_000_000,
) -> dict[str, zipfile.ZipInfo]:
    """Return a path-safe, duplicate-free regular-file ZIP roster."""

    infos = archive.infolist()
    names = [info.filename for info in infos]
    if not infos or len(names) != len(set(names)):
        raise M3ProtocolError(f"ZIP member roster is empty or duplicated: {label}")
    total = 0
    result: dict[str, zipfile.ZipInfo] = {}
    for info in infos:
        name = info.filename
        pure = Path(name)
        mode = (info.external_attr >> 16) & 0o170000
        if (
            type(name) is not str
            or not name
            or name.endswith("/")
            or pure.is_absolute()
            or ".." in pure.parts
            or "\\" in name
            or info.flag_bits & 0x1
            or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
            or mode == stat.S_IFLNK
            or info.file_size < 0
        ):
            raise M3ProtocolError(f"ZIP member is unsafe or unsupported: {label}")
        total += info.file_size
        if total > maximum_total_uncompressed_bytes:
            raise M3ProtocolError(f"ZIP expansion exceeds custody bound: {label}")
        result[name] = info
    return result


def _decode_npz_arrays_payload(
    payload: bytes,
    label: str,
    *,
    expected_names: set[str] | frozenset[str] | None = None,
) -> dict[str, np.ndarray]:
    """Decode an NPZ only after validating its exact raw ZIP-member roster."""

    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as raw_archive:
            infos = _strict_zip_infos(raw_archive, label)
            raw_names = set(infos)
            if any(
                "/" in name or not name.endswith(".npy") or len(name) <= 4
                for name in raw_names
            ):
                raise M3ProtocolError(f"NPZ member roster is invalid: {label}")
            logical_names = {name[:-4] for name in raw_names}
            if len(logical_names) != len(raw_names):
                raise M3ProtocolError(f"NPZ logical names are duplicated: {label}")
            if expected_names is not None and logical_names != set(expected_names):
                raise M3ProtocolError(f"NPZ member roster changed: {label}")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if (
                len(archive.files) != len(set(archive.files))
                or set(archive.files) != logical_names
            ):
                raise M3ProtocolError(f"NPZ logical roster changed: {label}")
            arrays = {
                name: np.ascontiguousarray(np.array(archive[name], copy=True))
                for name in archive.files
            }
    except M3ProtocolError:
        raise
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise M3ProtocolError(f"cannot parse strict NPZ payload: {label}") from exc
    return arrays


def _read_repository_file_no_follow(project_root: Path, relative: Path) -> bytes:
    """Read one repository file through a stable no-follow directory chain."""

    root_descriptor = _open_absolute_directory_no_follow(
        Path(os.path.abspath(os.fspath(project_root))), "project root"
    )
    try:
        payload, _metadata = _read_regular_file_at_no_follow(
            root_descriptor, relative, relative.as_posix()
        )
        return payload
    finally:
        os.close(root_descriptor)


def _executed_source_paths(project_root: Path) -> tuple[Path, ...]:
    """Return the complete local source, data, and native evidence roster."""

    root = Path(os.path.abspath(os.fspath(project_root)))
    paths = set(CODE_PATHS)
    package_root = root / "src" / "jxplanetx"
    if package_root.is_symlink() or not package_root.is_dir():
        raise M3ProtocolError("local jxplanetx package tree is absent")
    for path in package_root.rglob("*"):
        if path.is_symlink():
            raise M3ProtocolError(
                f"local package evidence contains a symlink: {path}"
            )
        if path.is_file() and (
            path.suffix in LOCAL_EVIDENCE_SUFFIXES
            or path.name in LOCAL_EVIDENCE_BASENAMES
        ):
            paths.add(path.relative_to(root))
    return tuple(sorted(paths, key=lambda path: path.as_posix()))


def _imported_local_module_identity(
    project_root: Path,
    expected_files: Mapping[str, Mapping[str, Any]],
    *,
    module_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Bind imported JX module origins to the exact archived local bytes."""

    root = Path(os.path.abspath(os.fspath(project_root)))
    modules: list[tuple[str, Any]] = []
    if module_names is None:
        modules.append(("__benchmark__", sys.modules[__name__]))
        modules.extend(
            (name, module)
            for name, module in tuple(sys.modules.items())
            if name == "jxplanetx" or name.startswith("jxplanetx.")
        )
    else:
        modules.extend(
            (name, importlib.import_module(name)) for name in module_names
        )
    rows: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for module_name, module in sorted(modules, key=lambda item: item[0]):
        if module_name in seen_names:
            raise M3ProtocolError("imported local module roster is not unique")
        seen_names.add(module_name)
        path_text = getattr(module, "__file__", None)
        if not isinstance(path_text, str):
            raise M3ProtocolError(
                f"imported local module has no file origin: {module_name}"
            )
        unresolved = Path(path_text)
        if unresolved.is_symlink():
            raise M3ProtocolError(
                f"imported local module origin is a symlink: {module_name}"
            )
        path = Path(os.path.abspath(os.fspath(unresolved)))
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise M3ProtocolError(
                f"imported JX module is outside the bound project: {module_name}"
            ) from exc
        payload = _read_repository_file_no_follow(root, relative)
        observed = hashlib.sha256(payload).hexdigest()
        expected = expected_files.get(relative.as_posix())
        if (
            type(expected) is not dict
            or expected.get("sha256") != observed
            or expected.get("size_bytes") != len(payload)
        ):
            raise M3ProtocolError(
                f"imported local bytes differ from the archive roster: {module_name}"
            )
        rows.append(
            {
                "module": module_name,
                "path": relative.as_posix(),
                "size_bytes": len(payload),
                "sha256": observed,
            }
        )
    if not rows:
        raise M3ProtocolError("imported local module roster is empty")
    return rows


def _code_identity(project_root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for relative in _executed_source_paths(project_root):
        payload = _read_repository_file_no_follow(project_root, relative)
        files.append(
            {
                "path": relative.as_posix(),
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    expected_files = {row["path"]: row for row in files}
    imported_modules = _imported_local_module_identity(
        project_root, expected_files
    )
    digest = hashlib.sha256(CODE_FILES_DOMAIN)
    for row in files:
        digest.update(_canonical_json(row))
    try:
        commit = _git_value(project_root, "rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "git_commit": commit,
        "files": files,
        "file_count": len(files),
        "imported_local_modules": imported_modules,
        "files_content_sha256": digest.hexdigest(),
    }


def _verify_local_file_identity(
    project_root: Path, expected_code_identity: Mapping[str, Any]
) -> dict[str, Any]:
    expected_rows = expected_code_identity.get("files")
    if type(expected_rows) is not list or any(
        type(row) is not dict
        or type(row.get("path")) is not str
        or type(row.get("size_bytes")) is not int
        or type(row.get("sha256")) is not str
        for row in expected_rows
    ):
        raise M3ProtocolError("protocol code identity file roster is malformed")
    expected = {row["path"]: row for row in expected_rows}
    paths = _executed_source_paths(project_root)
    if len(expected) != len(expected_rows) or set(expected) != {
        path.as_posix() for path in paths
    }:
        raise M3ProtocolError("local source/data/native roster changed")
    digest = hashlib.sha256(CODE_FILES_DOMAIN)
    for row in expected_rows:
        relative = Path(row["path"])
        payload = _read_repository_file_no_follow(project_root, relative)
        if (
            len(payload) != row["size_bytes"]
            or hashlib.sha256(payload).hexdigest() != row["sha256"]
        ):
            raise M3ProtocolError(
                f"local source/data/native bytes changed: {relative}"
            )
        digest.update(_canonical_json(row))
    imported = expected_code_identity.get("imported_local_modules")
    if type(imported) is not list or any(
        type(row) is not dict
        or row.get("path") not in expected
        or row.get("sha256") != expected[row["path"]]["sha256"]
        or row.get("size_bytes") != expected[row["path"]]["size_bytes"]
        for row in imported
    ):
        raise M3ProtocolError("protocol imported local module roster is malformed")
    content_sha256 = digest.hexdigest()
    if (
        expected_code_identity.get("file_count") != len(expected_rows)
        or expected_code_identity.get("files_content_sha256") != content_sha256
    ):
        raise M3ProtocolError("protocol local file identity digest is inconsistent")
    return {
        "file_count": len(expected_rows),
        "files_content_sha256": content_sha256,
        "imported_module_count_at_preparation": len(imported),
    }


def _sha256_path_no_follow(path: Path, label: str) -> dict[str, Any]:
    payload, metadata = _read_absolute_file_no_follow(path, label)
    return {
        "path": str(Path(os.path.abspath(os.fspath(path)))),
        "size_bytes": metadata.st_size,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _module_file_identity(module: Any) -> dict[str, Any]:
    module_name = getattr(module, "__name__", repr(module))
    path_text = getattr(module, "__file__", None)
    if not isinstance(path_text, str):
        raise M3ProtocolError(
            f"required imported module has no file origin: {module_name}"
        )
    identity = _sha256_path_no_follow(
        Path(os.path.abspath(path_text)), module_name
    )
    return {"module": module_name, **identity}


def _record_hash_matches(actual_sha256: str, item: Any, label: str) -> None:
    declared = getattr(item, "hash", None)
    if declared is None or declared.mode != "sha256":
        raise M3ProtocolError(
            f"runtime distribution member lacks a SHA-256 RECORD entry: {label}"
        )
    encoded = (
        base64.urlsafe_b64encode(bytes.fromhex(actual_sha256))
        .rstrip(b"=")
        .decode("ascii")
    )
    if encoded != declared.value:
        raise M3ProtocolError(
            f"runtime distribution member differs from RECORD: {label}"
        )


def _distribution_identity(
    name: str, required_modules: Sequence[Any]
) -> dict[str, Any]:
    """Bind a wheel RECORD plus every required module and native binary."""

    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise M3ProtocolError(
            f"required runtime distribution is absent: {name}"
        ) from exc
    files = tuple(distribution.files or ())
    if not files:
        raise M3ProtocolError(f"runtime distribution has no RECORD roster: {name}")
    inventory_rows: list[dict[str, Any]] = []
    located: dict[Path, tuple[str, Any, dict[str, Any]]] = {}
    record_path: Path | None = None
    for item in sorted(files, key=lambda value: str(value)):
        relative = str(item).replace(os.sep, "/")
        if "__pycache__" in relative or relative.endswith((".pyc", ".pyo")):
            continue
        unresolved = Path(distribution.locate_file(item))
        path = Path(os.path.abspath(os.fspath(unresolved)))
        payload, metadata = _read_absolute_file_no_follow(
            path, f"{name}:{relative}"
        )
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        declared_size = getattr(item, "size", None)
        if declared_size is not None and int(declared_size) != metadata.st_size:
            raise M3ProtocolError(
                f"runtime distribution member size differs from RECORD: {name}:{relative}"
            )
        declared_hash = getattr(item, "hash", None)
        is_record = relative.endswith(".dist-info/RECORD")
        if is_record:
            if declared_hash is not None or declared_size is not None:
                raise M3ProtocolError(
                    f"runtime RECORD unexpectedly self-hashes: {name}:{relative}"
                )
        else:
            _record_hash_matches(actual_sha256, item, f"{name}:{relative}")
        row = {
            "path": relative,
            "size_bytes": metadata.st_size,
            "sha256": actual_sha256,
            "record_hash": (
                None
                if declared_hash is None
                else f"{declared_hash.mode}={declared_hash.value}"
            ),
            "record_size_bytes": declared_size,
        }
        inventory_rows.append(row)
        if path in located:
            raise M3ProtocolError(
                f"runtime distribution RECORD aliases one path: {name}:{relative}"
            )
        located[path] = (relative, item, row)
        if is_record:
            if record_path is not None:
                raise M3ProtocolError(
                    f"runtime distribution has multiple RECORD files: {name}"
                )
            record_path = path
    if record_path is None:
        raise M3ProtocolError(f"runtime distribution RECORD is absent: {name}")
    required_rows: list[dict[str, Any]] = []
    for module in required_modules:
        identity = _module_file_identity(module)
        path = Path(identity["path"])
        try:
            relative, item, distribution_row = located[path]
        except KeyError as exc:
            raise M3ProtocolError(
                f"imported module is outside the hashed distribution: {identity['module']}"
            ) from exc
        _record_hash_matches(
            distribution_row["sha256"], item, f"{name}:{relative}"
        )
        if (
            identity["sha256"] != distribution_row["sha256"]
            or identity["size_bytes"] != distribution_row["size_bytes"]
        ):
            raise M3ProtocolError(
                f"imported module changed during distribution hashing: {identity['module']}"
            )
        required_rows.append({**identity, "distribution_path": relative})
    native_rows: list[dict[str, Any]] = []
    for path, (relative, item, distribution_row) in sorted(
        located.items(), key=lambda row: row[1][0]
    ):
        filename = Path(relative).name.lower()
        if not any(marker in filename for marker in NATIVE_DISTRIBUTION_MARKERS):
            continue
        _record_hash_matches(
            distribution_row["sha256"], item, f"{name}:{relative}"
        )
        native_rows.append(
            {
                "distribution_path": relative,
                "size_bytes": distribution_row["size_bytes"],
                "sha256": distribution_row["sha256"],
            }
        )
    _record_relative, _record_item, record_row = located[record_path]
    record_identity = {
        "path": str(record_path),
        "size_bytes": record_row["size_bytes"],
        "sha256": record_row["sha256"],
    }
    return {
        "name": distribution.metadata["Name"],
        "version": distribution.version,
        "record_file": record_identity,
        "record_member_count": len(inventory_rows),
        "total_size_bytes": sum(
            int(row["size_bytes"]) for row in inventory_rows
        ),
        "files": inventory_rows,
        "record_inventory_sha256": hashlib.sha256(
            RUNTIME_DISTRIBUTION_DOMAIN + _canonical_json(inventory_rows)
        ).hexdigest(),
        "required_imported_members": required_rows,
        "native_binaries": native_rows,
        "native_binary_count": len(native_rows),
        "native_binaries_sha256": hashlib.sha256(
            RUNTIME_DISTRIBUTION_DOMAIN + _canonical_json(native_rows)
        ).hexdigest(),
    }


def _rebound_native_libraries(rebound_module: Any) -> list[dict[str, Any]]:
    candidates: set[Path] = set()
    declared = getattr(rebound_module, "__libpath__", None)
    if isinstance(declared, str):
        candidates.add(Path(os.path.abspath(declared)))
    module_path = getattr(rebound_module, "__file__", None)
    if isinstance(module_path, str):
        candidates.update(
            Path(os.path.abspath(os.fspath(path)))
            for path in Path(os.path.abspath(module_path)).parent.glob("librebound*")
        )
    rows = [
        _sha256_path_no_follow(path, f"REBOUND native library {path.name}")
        for path in sorted(candidates)
        if path.is_file()
    ]
    if not rows:
        rows = [_module_file_identity(rebound_module)]
    return rows


def _gpu_identity(cp: Any, cuda_module: Any) -> dict[str, Any]:
    try:
        device_index = int(cp.cuda.Device().id)
        properties = cp.cuda.runtime.getDeviceProperties(device_index)
        raw_name = properties["name"]
        device_name = (
            raw_name.decode("ascii")
            if isinstance(raw_name, bytes)
            else str(raw_name)
        )
        device_uuid = bytes(properties["uuid"]).hex()
        identity = {
            "backend": "cupy",
            "device": f"cuda:{device_index}",
            "device_index": device_index,
            "device_name": device_name,
            "device_uuid": device_uuid,
            "device_compute_capability": (
                f"{int(properties['major'])}.{int(properties['minor'])}"
            ),
            "device_pci_domain_bus_device": (
                f"{int(properties['pciDomainID']):04x}:"
                f"{int(properties['pciBusID']):02x}:"
                f"{int(properties['pciDeviceID']):02x}"
            ),
            "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_local_runtime_version": int(
                cp.cuda.get_local_runtime_version()
            ),
            "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "nvrtc_version": [
                int(value) for value in cp.cuda.nvrtc.getVersion()
            ],
            "cupy_version": str(cp.__version__),
            "method_id": cuda_module.METHOD_ID,
            "kernel_id": cuda_module.KERNEL_ID,
            "fused_kernel_id": cuda_module.FUSED_KERNEL_ID,
            "kick_cuda_source_sha256": cuda_module.KICK_CUDA_SOURCE_SHA256,
            "fused_cuda_source_sha256": cuda_module.FUSED_CUDA_SOURCE_SHA256,
            "implementation_source_sha256": (
                cuda_module.IMPLEMENTATION_SOURCE_SHA256
            ),
            "all_conic_implementation_source_sha256": (
                cuda_module.ALL_CONIC_IMPLEMENTATION_SOURCE_SHA256
            ),
            "cuda_options": list(cuda_module.CUDA_OPTIONS),
            "threads_per_block": THREADS_PER_BLOCK,
            "determinism_scope": (
                "SAME_BOUND_PACKAGE_FILES_REPORTED_CUDA_VERSIONS_DEVICE_"
                "AND_KERNEL_SOURCE_OPTIONS"
            ),
        }
    except Exception as exc:
        raise M3ProtocolError("exact CUDA runtime identity is unavailable") from exc
    for label in (
        "device_name",
        "device_uuid",
        "device_compute_capability",
        "method_id",
        "kernel_id",
        "fused_kernel_id",
    ):
        if not identity[label]:
            raise M3ProtocolError(f"CUDA runtime identity omits {label}")
    return identity


def _numerical_runtime_modules() -> dict[str, Any]:
    """Import the reviewed numerical closure only in execution modes."""

    import cupy as cp
    import rebound
    import rebound.simulation as rebound_simulation
    import scipy
    import scipy.integrate
    import scipy.integrate._ivp.rk as scipy_rk
    import scipy.optimize

    local = {
        name: importlib.import_module(name) for name in NUMERICAL_LOCAL_MODULES
    }
    return {
        "cupy": cp,
        "rebound": rebound,
        "rebound_simulation": rebound_simulation,
        "scipy": scipy,
        "scipy_integrate": scipy.integrate,
        "scipy_rk": scipy_rk,
        "scipy_optimize": scipy.optimize,
        "local": local,
    }


def _runtime_identity(
    *,
    include_numerical_packages: bool,
    project_root: Path | None = None,
    expected_code_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if project_root is None or type(expected_code_identity) is not dict:
        raise M3ProtocolError(
            "runtime identity requires the protocol local file roster"
        )
    executable = Path(sys.executable).resolve()
    numpy_core = np._core._multiarray_umath  # type: ignore[attr-defined]
    identity: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "python_executable": _sha256_path_no_follow(
            executable, "Python executable"
        ),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "float_mantissa_bits": sys.float_info.mant_dig,
        "numpy_distribution": _distribution_identity(
            "numpy", (np, numpy_core)
        ),
        "local_files": _verify_local_file_identity(
            project_root, expected_code_identity
        ),
        "thread_environment": {
            name: os.environ.get(name) for name in _THREAD_ENVIRONMENT
        },
        "numerical_packages_included": include_numerical_packages,
    }
    if not include_numerical_packages:
        return identity
    expected_rows = expected_code_identity.get("files")
    if type(expected_rows) is not list:
        raise M3ProtocolError("protocol code identity has no exact file roster")
    expected_files = {
        row.get("path"): row for row in expected_rows if type(row) is dict
    }
    if len(expected_files) != len(expected_rows) or any(
        type(path) is not str for path in expected_files
    ):
        raise M3ProtocolError("protocol code identity file roster is malformed")
    modules = _numerical_runtime_modules()
    cp = modules["cupy"]
    rebound = modules["rebound"]
    scipy = modules["scipy"]
    identity.update(
        {
            "local_imported_modules": _imported_local_module_identity(
                project_root,
                expected_files,
                module_names=NUMERICAL_LOCAL_MODULES,
            ),
            "scipy_distribution": _distribution_identity(
                "scipy",
                (
                    scipy,
                    modules["scipy_integrate"],
                    modules["scipy_rk"],
                    modules["scipy_optimize"],
                ),
            ),
            "cupy_distribution": _distribution_identity(
                "cupy-cuda13x", (cp, cp.cuda.runtime, cp.cuda.nvrtc)
            ),
            "rebound_distribution": _distribution_identity(
                "rebound", (rebound, modules["rebound_simulation"])
            ),
            "rebound_native_libraries": _rebound_native_libraries(rebound),
            "gpu": _gpu_identity(
                cp, modules["local"]["jxplanetx.cuda_restricted_wh"]
            ),
        }
    )
    return identity


def _uniform_relative_acceleration_bound(
    gm_AU3_per_year2: np.ndarray,
    guard_radius_AU: float = ALGORITHMIC_GUARD_RADIUS_AU,
) -> float:
    gm = np.ascontiguousarray(gm_AU3_per_year2, dtype=np.float64)
    if gm.shape != (SOURCE_COUNT,) or not bool(np.all(gm > 0.0)):
        raise M3ProtocolError("source GM vector is not the exact positive roster")
    exact = (
        2
        * sum((Fraction.from_float(float(value)) for value in gm), Fraction())
        / Fraction.from_float(guard_radius_AU) ** 2
    )
    value = float(exact)
    if Fraction.from_float(value) < exact:
        value = math.nextafter(value, math.inf)
    if not math.isfinite(value) or value <= 0.0:
        raise M3ProtocolError("uniform acceleration certificate is invalid")
    return value


_SOURCE_CUSTODY_EXPECTATION_CACHE: dict[str, dict[str, Any]] = {}


def _source_custody_expectations(source: Any) -> dict[str, Any]:
    """Predeclare the exact retained-M2 slice and overlay hash chains."""

    from jxplanetx.long_horizon import (
        M2_DERIVED_ACCELERATION_BOUND_AU_PER_YEAR2,
        build_m2_frozen_source_path_tile,
        seal_m2_guard_overlay,
        verify_m2_source_path,
    )

    parent_sha256 = verify_m2_source_path(source)
    cache_key = hashlib.sha256(
        SOURCE_CUSTODY_DOMAIN
        + bytes.fromhex(parent_sha256)
        + ALGORITHMIC_GUARD_RADIUS_AU.hex().encode("ascii")
        + M2_DERIVED_ACCELERATION_BOUND_AU_PER_YEAR2.hex().encode("ascii")
    ).hexdigest()
    cached = _SOURCE_CUSTODY_EXPECTATION_CACHE.get(cache_key)
    if cached is not None:
        return cached
    overlay = seal_m2_guard_overlay(
        source,
        source_encounter_radii_AU=np.full(
            SOURCE_COUNT, ALGORITHMIC_GUARD_RADIUS_AU, dtype=np.float64
        ),
        certified_relative_acceleration_bounds_AU_per_year2=np.full(
            SOURCE_COUNT,
            M2_DERIVED_ACCELERATION_BOUND_AU_PER_YEAR2,
            dtype=np.float64,
        ),
    )

    def lane_expectation(
        lane_id: str, source_steps_per_map: int, map_step_count: int
    ) -> dict[str, Any]:
        slice_digest = hashlib.sha256(
            SOURCE_CUSTODY_DOMAIN + lane_id.encode("ascii")
        )
        binding_digest = hashlib.sha256(
            SOURCE_CUSTODY_DOMAIN
            + lane_id.encode("ascii")
            + b".bindings"
        )
        for map_index in range(map_step_count):
            start = map_index * source_steps_per_map
            source_slice = source.slice_steps(
                start, start + source_steps_per_map
            )
            binding = build_m2_frozen_source_path_tile(source_slice, overlay)
            slice_digest.update(bytes.fromhex(source_slice.dynamic_node_sha256))
            binding_digest.update(bytes.fromhex(binding.content_sha256))
        return {
            "map_slice_count": map_step_count,
            "slice_dynamic_digest": slice_digest.hexdigest(),
            "tile_binding_digest": binding_digest.hexdigest(),
        }

    annual_slice_sha256: list[str] = []
    annual_binding_sha256: list[str] = []
    for year_index in range(HORIZON_YEARS):
        start = year_index * SOURCE_STEPS_PER_YEAR
        source_slice = source.slice_steps(
            start, start + SOURCE_STEPS_PER_YEAR
        )
        binding = build_m2_frozen_source_path_tile(source_slice, overlay)
        annual_slice_sha256.append(source_slice.dynamic_node_sha256)
        annual_binding_sha256.append(binding.content_sha256)
    result = {
        "parent_path_content_sha256": parent_sha256,
        "parent_dynamic_node_sha256": source.dynamic_node_sha256,
        "guard_overlay_sha256": overlay.content_sha256,
        "candidate": lane_expectation(
            "candidate",
            CANDIDATE_SOURCE_STEPS_PER_MAP,
            CANDIDATE_MAP_STEP_COUNT,
        ),
        "refinement": lane_expectation(
            "refinement",
            REFINEMENT_SOURCE_STEPS_PER_MAP,
            REFINEMENT_MAP_STEP_COUNT,
        ),
        "annual_source_slice_dynamic_node_sha256": annual_slice_sha256,
        "annual_source_tile_binding_sha256": annual_binding_sha256,
    }
    _SOURCE_CUSTODY_EXPECTATION_CACHE[cache_key] = result
    return result


def _input_record(project_root: Path) -> tuple[dict[str, Any], Any, Any]:
    """Load only sealed host inputs and retained M2 evidence; no dynamics."""

    from jxplanetx.long_horizon import (
        KnownSystemRunProfile,
        prepare_known_system_observed_fixed_run,
        COMPLETED_CONTENT_SHA256,
        COMPLETED_FILE_SHA256,
        FINE_SOURCE_SCHEDULE_SHA256,
        FINE_SOURCE_STEP_COUNT,
        FINE_SOURCE_STEP_YEARS,
        OFFICIAL_RUN_RELATIVE_PATH,
        RESULT_CONTENT_SHA256,
        RESULT_FILE_SHA256,
        STATES_FILE_SHA256,
        load_verified_m2_source_path,
    )

    source = load_verified_m2_source_path(
        project_root / OFFICIAL_RUN_RELATIVE_PATH
    )
    # The sibling bridge provides two closed retained-M2-fine profiles.  It
    # seals the tracer genesis, roster, epoch adapter, and numerical policies;
    # source dynamics still come only from the independently verified M2 path.
    candidate_prepared = prepare_known_system_observed_fixed_run(
        profile_id=KnownSystemRunProfile.M2_FINE_CANDIDATE_10Y_ANNUAL
    )
    refinement_prepared = prepare_known_system_observed_fixed_run(
        profile_id=KnownSystemRunProfile.M2_FINE_REFINEMENT_10Y_ANNUAL
    )
    prepared = candidate_prepared
    if FINE_SOURCE_STEP_YEARS.hex() != SOURCE_STEP_YEARS.hex():
        raise M3ProtocolError("retained M2 source step changed")
    if FINE_SOURCE_STEP_COUNT != SOURCE_STEP_COUNT:
        raise M3ProtocolError("retained M2 source node count changed")
    if source.source_schedule_sha256 != FINE_SOURCE_SCHEDULE_SHA256:
        raise M3ProtocolError("retained M2 source schedule changed")
    if prepared.inventory.state_count != STATE_COUNT:
        raise M3ProtocolError("sealed tracer bridge is not exactly 108 states")
    if (
        refinement_prepared.inventory.content_sha256
        != prepared.inventory.content_sha256
        or refinement_prepared.roster.content_sha256
        != prepared.roster.content_sha256
        or not _bitwise_equal(
            refinement_prepared.tracer_positions_AU,
            prepared.tracer_positions_AU,
        )
        or not _bitwise_equal(
            refinement_prepared.tracer_velocities_AU_per_year,
            prepared.tracer_velocities_AU_per_year,
        )
        or prepared.profile.source_schedule_sha256
        != FINE_SOURCE_SCHEDULE_SHA256
        or refinement_prepared.profile.source_schedule_sha256
        != FINE_SOURCE_SCHEDULE_SHA256
    ):
        raise M3ProtocolError("candidate/refinement sealed bridges diverged")
    bound = _uniform_relative_acceleration_bound(
        source.source_gm_AU3_per_year2
    )
    source_custody_expectations = _source_custody_expectations(source)
    family_ids = tuple(
        state.parent_object_id for state in prepared.inventory.states
    )
    family_counts = {
        family_id: family_ids.count(family_id)
        for family_id in dict.fromkeys(family_ids)
    }
    if len(family_counts) != 12 or set(family_counts.values()) != {9}:
        raise M3ProtocolError("sealed tracer families are not twelve groups of nine")
    model = prepared.known_system_model
    model_position = np.ascontiguousarray(
        model.positions_primary_relative_ecliptic_AU, dtype=np.float64
    )
    model_velocity = np.ascontiguousarray(
        model.velocities_primary_relative_ecliptic_AU_per_julian_year,
        dtype=np.float64,
    )
    model_gm = np.ascontiguousarray(
        model.gm_AU3_per_julian_year2, dtype=np.float64
    )
    if (
        source.source_body_ids != model.body_ids
        or source.source_ids != model.body_names
        or source.source_names != model.body_names
        or source.epoch_jd_tdb.hex() != model.epoch_jd_tdb.hex()
        or not _bitwise_equal(source.source_relative_positions_AU[0], model_position)
        or not _bitwise_equal(
            source.source_relative_velocities_AU_per_year[0], model_velocity
        )
        or not _bitwise_equal(source.source_gm_AU3_per_year2, model_gm)
    ):
        raise M3ProtocolError(
            "sealed tracer model and retained M2 node-zero source differ"
        )
    record = {
        "known_system_model_sha256": (
            prepared.known_system_model.content_sha256
        ),
        "genesis_model_sha256": prepared.inventory.model.content_sha256,
        "inventory_content_sha256": prepared.inventory.content_sha256,
        "roster_content_sha256": prepared.roster.content_sha256,
        "tracer_genesis_bridge_content_sha256": {
            "candidate": candidate_prepared.content_sha256,
            "refinement": refinement_prepared.content_sha256,
        },
        "profile_receipt_sha256": {
            "candidate": candidate_prepared.profile.content_sha256,
            "refinement": refinement_prepared.profile.content_sha256,
        },
        "tracer_genesis_bridge_scope": (
            "TRACER_ARRAYS_ROSTER_AND_EPOCH_ADAPTER_ONLY_"
            "OLD_SOURCE_PROFILE_EXPLICITLY_NOT_REUSED"
        ),
        "state_count": prepared.inventory.state_count,
        "state_ids": list(prepared.roster.object_ids),
        "family_ids": list(family_ids),
        "family_state_counts": family_counts,
        "tracer_positions_sha256": _array_sha256(prepared.tracer_positions_AU),
        "tracer_velocities_sha256": _array_sha256(
            prepared.tracer_velocities_AU_per_year
        ),
        "source_initial_positions_sha256": _array_sha256(
            source.source_relative_positions_AU[0]
        ),
        "source_initial_velocities_sha256": _array_sha256(
            source.source_relative_velocities_AU_per_year[0]
        ),
        "source_gm_sha256": _array_sha256(
            source.source_gm_AU3_per_year2
        ),
        "m2_source": {
            "content_sha256": source.content_sha256,
            "dynamic_node_sha256": source.dynamic_node_sha256,
            "source_ids": list(source.source_ids),
            "source_body_ids": list(source.source_body_ids),
            "primary_source_id": source.primary_source_id,
            "epoch_jd_tdb_hex": source.epoch_jd_tdb.hex(),
            "source_step_years_hex": source.source_step_years.hex(),
            "source_schedule_sha256": source.source_schedule_sha256,
            "node_count": source.node_count,
            "parent_file_sha256": dict(source.parent_file_sha256),
            "parent_content_sha256": dict(source.parent_content_sha256),
            "official_result_file_sha256": RESULT_FILE_SHA256,
            "official_result_content_sha256": RESULT_CONTENT_SHA256,
            "official_completion_file_sha256": COMPLETED_FILE_SHA256,
            "official_completion_content_sha256": COMPLETED_CONTENT_SHA256,
            "official_states_file_sha256": STATES_FILE_SHA256,
        },
        "m2_execution_source_custody": source_custody_expectations,
        "fine_source_schedule_sha256": FINE_SOURCE_SCHEDULE_SHA256,
        "uniform_relative_acceleration_bound_AU_per_year2_hex": bound.hex(),
    }
    return record, {
        "candidate": candidate_prepared,
        "refinement": refinement_prepared,
    }, source


def build_protocol(project_root: str | Path) -> dict[str, Any]:
    """Build the deterministic, explicitly non-authorizing M3 protocol."""

    root = Path(project_root).resolve()
    inputs, _, _ = _input_record(root)
    code_identity = _code_identity(root)
    record = {
        "schema": PROTOCOL_SCHEMA,
        "run_id": RUN_ID,
        "classification": CLASSIFICATION,
        "status": "PREPARED_THRESHOLDS_UNSET_NOT_EXECUTED",
        "horizon": {
            "years": HORIZON_YEARS,
            "split_epoch_years": SPLIT_EPOCH_YEARS,
            "source_step_years_hex": SOURCE_STEP_YEARS.hex(),
            "source_step_count": SOURCE_STEP_COUNT,
        },
        "inputs": inputs,
        "lanes": {
            "jx_candidate": {
                "integrator": "JX_RESTRICTED_WH_FIXED_LIFECYCLE_V1",
                "step_years_hex": CANDIDATE_STEP_YEARS.hex(),
                "map_step_count": CANDIDATE_MAP_STEP_COUNT,
                "source_steps_per_map_step": CANDIDATE_SOURCE_STEPS_PER_MAP,
                "retain_every_map_endpoint": True,
            },
            "jx_refinement": {
                "integrator": "JX_RESTRICTED_WH_FIXED_LIFECYCLE_V1",
                "step_years_hex": REFINEMENT_STEP_YEARS.hex(),
                "map_step_count": REFINEMENT_MAP_STEP_COUNT,
                "source_steps_per_map_step": REFINEMENT_SOURCE_STEPS_PER_MAP,
                "retain_every_map_endpoint": True,
            },
            "independent_comparator": {
                "implementation": "REBOUND",
                "required_exact_version": REQUIRED_REBOUND_VERSION,
                "integrator": "ias15",
                "epsilon_hex": IAS15_EPSILON.hex(),
                "adaptive_mode": IAS15_ADAPTIVE_MODE,
                "initial_step_years_hex": IAS15_INITIAL_STEP_YEARS.hex(),
                "minimum_step_years_hex": (0.0).hex(),
                "active_massive_source_count": SOURCE_COUNT,
                "exactly_massless_tracer_count": STATE_COUNT,
                "output_every_refinement_node": True,
                "output_frame": "SUN_RELATIVE_NONROTATING",
                "reference_truth": False,
            },
        },
        "routing_and_events": {
            "algorithmic_guard_radius_AU_hex": (
                ALGORITHMIC_GUARD_RADIUS_AU.hex()
            ),
            "guard_radius_role": "ALGORITHMIC_ROUTING_SURFACE_NOT_PHYSICAL_RADIUS",
            "terminal_source_radii_AU_hex": [
                PHYSICAL_TERMINAL_RADIUS_AU.hex()
            ]
            * SOURCE_COUNT,
            "primary_outer_boundary_AU_hex": PRIMARY_OUTER_BOUNDARY_AU.hex(),
            "maximum_primary_dynamical_phase_hex": (
                MAXIMUM_PRIMARY_DYNAMICAL_PHASE.hex()
            ),
            "outward_radial_velocity_tolerance_AU_per_year_hex": (
                OUTWARD_RADIAL_VELOCITY_TOLERANCE_AU_PER_YEAR.hex()
            ),
            "route_worker_count": ROUTE_WORKER_COUNT,
            "independent_interval_clearance_required": True,
            "instantaneous_primary_phase_gate_required": True,
            "osculating_periapse_phase_gate_required": True,
            "zero_unserialized_or_dropped_events_required": True,
        },
        "comparisons": {
            "ids": list(COMPARISON_IDS),
            "metrics": list(COMPARISON_METRICS),
            "individual_state_gates_required": True,
            "family_gates_required": True,
            "global_gates_required": True,
            "gate_thresholds": None,
            "threshold_status": "UNSET_CALIBRATION_REQUIRED",
        },
        "precision_selection": {
            "policy": "PRECISION_FIRST_FINER_SECOND_ORDER_PAIR_V1",
            "legacy_ratio_32_lane_is_official": False,
            "candidate_source_steps_per_map": CANDIDATE_SOURCE_STEPS_PER_MAP,
            "refinement_source_steps_per_map": REFINEMENT_SOURCE_STEPS_PER_MAP,
            "literal_calibration_metrics_in_prepare_protocol": False,
        },
        "source_custody": {
            "retained_m2_parent_path_only": True,
            "source_reintegration_by_jx_lanes": False,
            "exact_parent_slice_hashes_required": True,
            "exact_source_no_backreaction_required": True,
            "independent_source_only_rebound_crosscheck_required": True,
        },
        "split_resume": {
            "split_epoch_years": SPLIT_EPOCH_YEARS,
            "candidate_split_map_step": SPLIT_CANDIDATE_MAP_STEP,
            "refinement_split_map_step": SPLIT_REFINEMENT_MAP_STEP,
            "checkpoint_segment_years": 1,
            "checkpoint_count_per_lane": HORIZON_YEARS,
            "replay_load_at_split_and_final_head": True,
            "identical_retained_source_slices_required": True,
            "bitwise_terminal_state_equality_required": True,
            "lifecycle_and_event_ledger_equality_required": True,
        },
        "resources": {
            "physical_cpu_route_workers": ROUTE_WORKER_COUNT,
            "gpu_devices": 1,
            "concurrent_numerical_lanes": 1,
            "blas_threads_per_process": 1,
        },
        "evidence_policy": {
            "official_attempt_directory": "UNSET_UNIQUE_CHILD_REQUIRED",
            "create_only": True,
            "failed_and_incomplete_attempts_retained": True,
            "arrays_required": True,
            "event_ledger_required": True,
            "executed_code_and_runtime_archive_required": True,
            "completion_marker_written_last": True,
        },
        "execution_authorized": False,
        "official_attempt_authorized": False,
        "staged_one_thousand_year_run_authorized": False,
        "code_identity": code_identity,
        "preparation_runtime_identity": _runtime_identity(
            include_numerical_packages=False,
            project_root=root,
            expected_code_identity=code_identity,
        ),
        "claims": dict(CLAIMS),
    }
    return _content_record(record, PROTOCOL_DOMAIN)


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if type(protocol) is not dict:
        raise M3ProtocolError("protocol must be an exact dictionary")
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise M3ProtocolError("protocol schema changed")
    if protocol.get("content_sha256") != _content_sha256(
        PROTOCOL_DOMAIN, protocol
    ):
        raise M3ProtocolError("protocol content digest changed")
    if (
        protocol.get("execution_authorized") is not False
        or protocol.get("official_attempt_authorized") is not False
        or protocol.get("staged_one_thousand_year_run_authorized") is not False
        or protocol.get("comparisons", {}).get("gate_thresholds") is not None
        or protocol.get("comparisons", {}).get("threshold_status")
        != "UNSET_CALIBRATION_REQUIRED"
        or protocol.get("claims") != CLAIMS
        or any(protocol["claims"].values())
    ):
        raise M3ProtocolError("prepared protocol widened authorization or claims")


def validate_threshold_commitment(
    value: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    expected_attempt_id: str | None = None,
) -> dict[str, Any]:
    """Validate a separately reviewed one-attempt threshold commitment."""

    if type(value) is not dict:
        raise M3ProtocolError("threshold commitment must be an exact dictionary")
    if set(value) != _THRESHOLD_KEYS:
        raise M3ProtocolError("threshold commitment root roster changed")
    if value.get("schema") != THRESHOLD_SCHEMA:
        raise M3ProtocolError("threshold commitment schema changed")
    _require_nonzero_sha256(value.get("content_sha256"), "threshold content digest")
    if value.get("content_sha256") != _content_sha256(THRESHOLD_DOMAIN, value):
        raise M3ProtocolError("threshold commitment content digest changed")
    protocol_digest = _require_nonzero_sha256(
        value.get("protocol_content_sha256"), "threshold protocol digest"
    )
    if protocol_digest != protocol.get("content_sha256"):
        raise M3ProtocolError("threshold commitment names a different protocol")
    if value.get("authorized_for_one_official_attempt") is not True:
        raise M3ProtocolError("threshold commitment is not authorized")
    if value.get("staged_one_thousand_year_run_authorized") is not False:
        raise M3ProtocolError("threshold commitment widened duration authorization")
    attempt_id = _require_attempt_id(value.get("authorized_attempt_id"))
    if expected_attempt_id is not None and attempt_id != expected_attempt_id:
        raise M3ProtocolError("threshold commitment authorizes another attempt")
    _require_review_id(value.get("review_id"))
    _require_nonzero_sha256(
        value.get("calibration_content_sha256"), "calibration custody digest"
    )
    comparisons = value.get("comparisons")
    if type(comparisons) is not dict or set(comparisons) != set(COMPARISON_IDS):
        raise M3ProtocolError("threshold comparison roster changed")
    for comparison_id in COMPARISON_IDS:
        limits = comparisons[comparison_id]
        if type(limits) is not dict or set(limits) != set(COMPARISON_METRICS):
            raise M3ProtocolError("threshold metric roster changed")
        for metric in COMPARISON_METRICS:
            encoded = limits[metric]
            if type(encoded) is not str:
                raise M3ProtocolError("thresholds must use exact float hex")
            try:
                threshold = float.fromhex(encoded)
            except ValueError as exc:
                raise M3ProtocolError("threshold is not valid float hex") from exc
            if not math.isfinite(threshold) or threshold <= 0.0:
                raise M3ProtocolError("thresholds must be finite and positive")
            if encoded != threshold.hex():
                raise M3ProtocolError("threshold float hex is not canonical")
    fixed = value.get("fixed_gates")
    expected_fixed = {
        "minimum_interval_clearance_AU_hex": (
            ALGORITHMIC_GUARD_RADIUS_AU.hex()
        ),
        "maximum_instantaneous_primary_phase_rad_hex": (
            MAXIMUM_PRIMARY_DYNAMICAL_PHASE.hex()
        ),
        "maximum_periapse_primary_phase_rad_hex": (
            MAXIMUM_PRIMARY_DYNAMICAL_PHASE.hex()
        ),
        "unmatched_or_dropped_event_budget": 0,
        "source_backreaction_bitwise_budget": 0,
        "split_resume_bitwise_budget": 0,
    }
    if type(fixed) is not dict or set(fixed) != set(expected_fixed):
        raise M3ProtocolError("fixed safety gate roster changed")
    if any(
        (
            type(fixed[key]) is not type(expected)
            or fixed[key] != expected
        )
        for key, expected in expected_fixed.items()
    ):
        raise M3ProtocolError("fixed safety gates cannot be calibrated or relaxed")
    claims = value.get("claims")
    if (
        type(claims) is not dict
        or set(claims) != set(CLAIMS)
        or any(type(claims[key]) is not bool or claims[key] for key in CLAIMS)
    ):
        raise M3ProtocolError("threshold commitment widened claims")
    return dict(value)


def build_threshold_commitment(
    *,
    protocol_content_sha256: str,
    calibration_content_sha256: str,
    comparisons: Mapping[str, Mapping[str, str]],
    review_id: str,
    attempt_id: str,
) -> dict[str, Any]:
    """Construct a reviewable commitment; this function performs no run.

    The explicit review identifier prevents a calibration result from being
    mistaken for authorization merely because it contains measured values.
    """

    _require_nonzero_sha256(protocol_content_sha256, "protocol content digest")
    _require_nonzero_sha256(calibration_content_sha256, "calibration content digest")
    _require_review_id(review_id)
    _require_attempt_id(attempt_id)
    record = _content_record(
        {
            "schema": THRESHOLD_SCHEMA,
            "protocol_content_sha256": protocol_content_sha256,
            "calibration_content_sha256": calibration_content_sha256,
            "review_id": review_id,
            "authorized_attempt_id": attempt_id,
            "comparisons": _plain(comparisons),
            "fixed_gates": {
                "minimum_interval_clearance_AU_hex": (
                    ALGORITHMIC_GUARD_RADIUS_AU.hex()
                ),
                "maximum_instantaneous_primary_phase_rad_hex": (
                    MAXIMUM_PRIMARY_DYNAMICAL_PHASE.hex()
                ),
                "maximum_periapse_primary_phase_rad_hex": (
                    MAXIMUM_PRIMARY_DYNAMICAL_PHASE.hex()
                ),
                "unmatched_or_dropped_event_budget": 0,
                "source_backreaction_bitwise_budget": 0,
                "split_resume_bitwise_budget": 0,
            },
            "authorized_for_one_official_attempt": True,
            "staged_one_thousand_year_run_authorized": False,
            "claims": dict(CLAIMS),
        },
        THRESHOLD_DOMAIN,
    )
    # A minimal synthetic protocol is enough for structural validation here;
    # official execution revalidates against the complete protocol.
    validate_threshold_commitment(
        record,
        {"content_sha256": protocol_content_sha256},
    )
    return record


def _orbital_elements_arrays(
    positions_AU: np.ndarray,
    velocities_AU_per_year: np.ndarray,
    primary_gm_AU3_per_year2: float,
) -> dict[str, np.ndarray]:
    positions = np.asarray(positions_AU, dtype=np.float64)
    velocities = np.asarray(velocities_AU_per_year, dtype=np.float64)
    if (
        positions.shape != velocities.shape
        or positions.ndim != 3
        or positions.shape[1:] != (STATE_COUNT, 3)
        or not bool(np.all(np.isfinite(positions)))
        or not bool(np.all(np.isfinite(velocities)))
        or type(primary_gm_AU3_per_year2) is not float
        or not math.isfinite(primary_gm_AU3_per_year2)
        or primary_gm_AU3_per_year2 <= 0.0
    ):
        raise M3NumericalError("orbital-element input arrays are invalid")
    radius = np.linalg.norm(positions, axis=2)
    speed2 = np.sum(velocities * velocities, axis=2)
    angular_momentum = np.cross(positions, velocities)
    angular_norm = np.linalg.norm(angular_momentum, axis=2)
    energy = 0.5 * speed2 - primary_gm_AU3_per_year2 / radius
    semimajor_axis = -primary_gm_AU3_per_year2 / (2.0 * energy)
    eccentricity_vector = (
        np.cross(velocities, angular_momentum) / primary_gm_AU3_per_year2
        - positions / radius[:, :, None]
    )
    eccentricity = np.linalg.norm(eccentricity_vector, axis=2)
    cosine_inclination = np.clip(
        angular_momentum[:, :, 2] / angular_norm, -1.0, 1.0
    )
    inclination = np.arccos(cosine_inclination)
    perihelion = semimajor_axis * (1.0 - eccentricity)
    if (
        bool(np.any(radius <= 0.0))
        or bool(np.any(angular_norm <= 0.0))
        or bool(np.any(energy >= 0.0))
        or bool(np.any(semimajor_axis <= 0.0))
        or bool(np.any(eccentricity < 0.0))
        or bool(np.any(eccentricity >= 1.0))
        or bool(np.any(perihelion <= 0.0))
        or not all(
            bool(np.all(np.isfinite(array)))
            for array in (semimajor_axis, eccentricity, inclination, perihelion)
        )
    ):
        raise M3NumericalError(
            "a retained tracer state is outside the bound nonsingular element domain"
        )
    return {
        "semimajor_axis_AU": np.ascontiguousarray(semimajor_axis),
        "perihelion_AU": np.ascontiguousarray(perihelion),
        "eccentricity": np.ascontiguousarray(eccentricity),
        "inclination_rad": np.ascontiguousarray(inclination),
    }


def _comparison_per_state(
    left_position: np.ndarray,
    left_velocity: np.ndarray,
    right_position: np.ndarray,
    right_velocity: np.ndarray,
    primary_gm: float,
) -> dict[str, np.ndarray]:
    arrays = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (
            left_position,
            left_velocity,
            right_position,
            right_velocity,
        )
    )
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise M3NumericalError("comparison state arrays have different shapes")
    if arrays[0].ndim != 3 or arrays[0].shape[1:] != (STATE_COUNT, 3):
        raise M3NumericalError("comparison state arrays have an invalid shape")
    if not all(bool(np.all(np.isfinite(value))) for value in arrays):
        raise M3NumericalError("comparison state arrays contain nonfinite values")
    left_elements = _orbital_elements_arrays(arrays[0], arrays[1], primary_gm)
    right_elements = _orbital_elements_arrays(arrays[2], arrays[3], primary_gm)
    result = {
        "position_vector_l2_AU": np.max(
            np.linalg.norm(arrays[0] - arrays[2], axis=2), axis=0
        ),
        "velocity_vector_l2_AU_per_year": np.max(
            np.linalg.norm(arrays[1] - arrays[3], axis=2), axis=0
        ),
    }
    for name in (
        "semimajor_axis_AU",
        "perihelion_AU",
        "eccentricity",
        "inclination_rad",
    ):
        result[name] = np.max(
            np.abs(left_elements[name] - right_elements[name]), axis=0
        )
    return {
        name: np.ascontiguousarray(value, dtype=np.float64)
        for name, value in result.items()
    }


def _comparison_record(
    per_state: Mapping[str, np.ndarray],
    state_ids: Sequence[str],
    family_ids: Sequence[str],
) -> dict[str, Any]:
    if len(state_ids) != STATE_COUNT or len(family_ids) != STATE_COUNT:
        raise M3NumericalError("state/family roster length changed")
    unique_families = tuple(dict.fromkeys(family_ids))
    record: dict[str, Any] = {"metrics": {}}
    for metric in COMPARISON_METRICS:
        values = np.ascontiguousarray(per_state[metric], dtype=np.float64)
        if values.shape != (STATE_COUNT,) or not bool(np.all(np.isfinite(values))):
            raise M3NumericalError(f"invalid per-state metric: {metric}")
        family_maxima = {
            family: float(
                np.max(
                    values[
                        np.asarray(
                            [item == family for item in family_ids], dtype=np.bool_
                        )
                    ]
                )
            )
            for family in unique_families
        }
        record["metrics"][metric] = {
            "global_maximum_hex": float(np.max(values)).hex(),
            "per_state_hex": {
                str(state_id): float(value).hex()
                for state_id, value in zip(state_ids, values, strict=True)
            },
            "per_family_maximum_hex": {
                family: value.hex() for family, value in family_maxima.items()
            },
        }
    return record


def _norm3_enclosure(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape[-1:] != (3,) or not bool(np.all(np.isfinite(array))):
        raise M3NumericalError("guard vector is not a finite three-vector")
    if bool(np.any(np.abs(array) > _GUARD_MAXIMUM_ABSOLUTE_COMPONENT)):
        raise M3NumericalError("guard vector exceeds the reviewed physical scale")
    observed = np.sqrt(np.sum(array * array, axis=-1, dtype=np.float64))
    if not bool(np.all(np.isfinite(observed))):
        raise M3NumericalError("guard vector norm overflowed")
    lower = np.maximum(
        0.0,
        np.nextafter(
            observed * (1.0 - _GUARD_ROUNDING_PAD), -math.inf
        ),
    )
    upper = np.nextafter(
        observed * (1.0 + _GUARD_ROUNDING_PAD), math.inf
    )
    return lower, upper


def _difference_norm3_enclosure(
    left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    try:
        difference = left_array - right_array
        scale = np.nextafter(
            np.abs(left_array) + np.abs(right_array), math.inf
        )
    except ValueError as exc:
        raise M3NumericalError("guard difference arrays do not broadcast") from exc
    difference_lower, difference_upper = _norm3_enclosure(difference)
    # For correctly rounded binary64 subtraction, u*(|a|+|b|) bounds the
    # component error even under catastrophic cancellation.  Round the bound
    # itself upward and combine it with the norm enclosure by the triangle
    # inequality.
    component_error = np.nextafter(
        scale * sys.float_info.epsilon, math.inf
    )
    _, error_upper = _norm3_enclosure(component_error)
    lower = np.maximum(
        0.0,
        np.nextafter(difference_lower - error_upper, -math.inf),
    )
    upper = np.nextafter(difference_upper + error_upper, math.inf)
    return lower, upper


def _certified_swept_clearance(
    position_left: np.ndarray,
    position_right: np.ndarray,
    velocity_left: np.ndarray,
    velocity_right: np.ndarray,
    *,
    step_years: float,
    acceleration_bound: float,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        type(step_years) is not float
        or not math.isfinite(step_years)
        or step_years <= 0.0
        or type(acceleration_bound) is not float
        or not math.isfinite(acceleration_bound)
        or acceleration_bound <= 0.0
    ):
        raise M3NumericalError("guard sweep scalars are invalid")
    distance_lower, _ = _difference_norm3_enclosure(
        position_left, position_right
    )
    _, speed_upper = _difference_norm3_enclosure(
        velocity_left, velocity_right
    )
    linear = np.nextafter(speed_upper * step_years, math.inf)
    step_squared_upper = math.nextafter(step_years * step_years, math.inf)
    acceleration_product_upper = math.nextafter(
        acceleration_bound * step_squared_upper, math.inf
    )
    quadratic = math.nextafter(0.5 * acceleration_product_upper, math.inf)
    swept_upper = np.nextafter(linear + quadratic, math.inf)
    clearance_lower = np.nextafter(distance_lower - swept_upper, -math.inf)
    return clearance_lower, distance_lower


def _independent_source_pair_guards(
    source_position: np.ndarray,
    source_velocity: np.ndarray,
    source_gm: np.ndarray,
    step_years: float,
) -> dict[str, Any]:
    node_count = source_position.shape[0]
    if (
        source_position.shape != (node_count, SOURCE_COUNT, 3)
        or source_velocity.shape != source_position.shape
        or source_gm.shape != (SOURCE_COUNT,)
        or node_count < 2
    ):
        raise M3NumericalError("source-pair guard arrays have invalid shapes")
    pair_left, pair_right = np.triu_indices(SOURCE_COUNT, k=1)
    acceleration_bound = _uniform_relative_acceleration_bound(source_gm)
    minimum_clearance = math.inf
    minimum_instantaneous = math.inf
    all_clear = True
    interval_chunk = 256
    for start in range(0, node_count - 1, interval_chunk):
        stop = min(start + interval_chunk, node_count - 1)
        clearance, distance_lower = _certified_swept_clearance(
            source_position[start:stop, pair_left],
            source_position[start:stop, pair_right],
            source_velocity[start:stop, pair_left],
            source_velocity[start:stop, pair_right],
            step_years=step_years,
            acceleration_bound=acceleration_bound,
        )
        minimum_clearance = min(minimum_clearance, float(np.min(clearance)))
        minimum_instantaneous = min(
            minimum_instantaneous, float(np.min(distance_lower))
        )
        all_clear = bool(
            all_clear and np.all(clearance > ALGORITHMIC_GUARD_RADIUS_AU)
        )
    final_lower, _ = _difference_norm3_enclosure(
        source_position[-1, pair_left], source_position[-1, pair_right]
    )
    minimum_instantaneous = min(
        minimum_instantaneous, float(np.min(final_lower))
    )
    return {
        "pair_count": int(pair_left.size),
        "interval_count": node_count - 1,
        "minimum_instantaneous_source_pair_distance_AU": minimum_instantaneous,
        "minimum_certified_source_pair_interval_clearance_AU": minimum_clearance,
        "all_source_pair_intervals_clear": all_clear,
        "rounding_enclosure_relative_pad_hex": _GUARD_ROUNDING_PAD.hex(),
    }


def _independent_interval_guards(
    tracer_position: np.ndarray,
    tracer_velocity: np.ndarray,
    source_position: np.ndarray,
    source_velocity: np.ndarray,
    source_gm: np.ndarray,
    step_years: float,
) -> dict[str, Any]:
    tracer_position = np.asarray(tracer_position, dtype=np.float64)
    tracer_velocity = np.asarray(tracer_velocity, dtype=np.float64)
    source_position = np.asarray(source_position, dtype=np.float64)
    source_velocity = np.asarray(source_velocity, dtype=np.float64)
    source_gm = np.asarray(source_gm, dtype=np.float64)
    node_count = tracer_position.shape[0]
    if (
        tracer_position.shape != (node_count, STATE_COUNT, 3)
        or tracer_velocity.shape != tracer_position.shape
        or source_position.shape != (node_count, SOURCE_COUNT, 3)
        or source_velocity.shape != source_position.shape
        or source_gm.shape != (SOURCE_COUNT,)
        or node_count < 2
    ):
        raise M3NumericalError("independent guard arrays have invalid shapes")
    if not all(
        bool(np.all(np.isfinite(value)))
        for value in (
            tracer_position,
            tracer_velocity,
            source_position,
            source_velocity,
            source_gm,
        )
    ):
        raise M3NumericalError("independent guard arrays contain nonfinite values")
    acceleration_bound = _uniform_relative_acceleration_bound(source_gm)
    source_pair = _independent_source_pair_guards(
        source_position, source_velocity, source_gm, step_years
    )
    minimum_clearance = math.inf
    minimum_instantaneous = math.inf
    all_intervals_clear = True
    # A full 10,240 x 108 x 10 x 3 tensor would waste several gigabytes.
    # Chunking changes no arithmetic within an interval and keeps the
    # independent validation bounded to a few megabytes of scratch memory.
    interval_chunk = 128
    for start in range(0, node_count - 1, interval_chunk):
        stop = min(start + interval_chunk, node_count - 1)
        clearance, start_distance_lower = _certified_swept_clearance(
            tracer_position[start:stop, :, None, :],
            source_position[start:stop, None, :, :],
            tracer_velocity[start:stop, :, None, :],
            source_velocity[start:stop, None, :, :],
            step_years=step_years,
            acceleration_bound=acceleration_bound,
        )
        minimum_clearance = min(minimum_clearance, float(np.min(clearance)))
        minimum_instantaneous = min(
            minimum_instantaneous, float(np.min(start_distance_lower))
        )
        all_intervals_clear = bool(
            all_intervals_clear
            and np.all(clearance > ALGORITHMIC_GUARD_RADIUS_AU)
        )
    final_distance, _ = _difference_norm3_enclosure(
        tracer_position[-1, :, None, :], source_position[-1, None, :, :]
    )
    minimum_instantaneous = min(
        minimum_instantaneous, float(np.min(final_distance))
    )
    primary_radius, _ = _norm3_enclosure(tracer_position)
    if bool(np.any(primary_radius <= 0.0)):
        raise M3NumericalError("primary radius has no positive lower bound")
    instantaneous_phase = np.nextafter(
        step_years * np.sqrt(float(source_gm[0]) / primary_radius**3),
        math.inf,
    )
    elements = _orbital_elements_arrays(
        tracer_position, tracer_velocity, float(source_gm[0])
    )
    periapse_lower = np.maximum(
        0.0,
        np.nextafter(
            elements["perihelion_AU"]
            * (1.0 - _GUARD_ORBITAL_ELEMENT_PAD),
            -math.inf,
        ),
    )
    if bool(np.any(periapse_lower <= 0.0)):
        raise M3NumericalError("osculating periapse has no positive lower bound")
    periapse_phase = np.nextafter(
        step_years * np.sqrt(float(source_gm[0]) / periapse_lower**3),
        math.inf,
    )
    return {
        "interval_count": node_count - 1,
        "minimum_instantaneous_source_distance_AU": float(
            minimum_instantaneous
        ),
        "minimum_certified_interval_clearance_AU": float(
            minimum_clearance
        ),
        "maximum_instantaneous_primary_phase_rad": float(
            np.max(instantaneous_phase)
        ),
        "maximum_periapse_primary_phase_rad": float(np.max(periapse_phase)),
        "uniform_relative_acceleration_bound_AU_per_year2": (
            acceleration_bound
        ),
        "bootstrap_policy": (
            "FIRST_CROSSING_CONTRADICTION_FROM_START_DISTANCE_MINUS_"
            "RELATIVE_SPEED_H_MINUS_HALF_UNIFORM_ACCELERATION_H2_V1"
        ),
        "algorithmic_guard_radius_AU": ALGORITHMIC_GUARD_RADIUS_AU,
        "all_intervals_clear": all_intervals_clear,
        "source_pair_safety": source_pair,
        "instantaneous_phase_passed": bool(
            np.all(instantaneous_phase <= MAXIMUM_PRIMARY_DYNAMICAL_PHASE)
        ),
        "periapse_phase_passed": bool(
            np.all(periapse_phase <= MAXIMUM_PRIMARY_DYNAMICAL_PHASE)
        ),
    }


def _validate_numerical_bundle(bundle: Mapping[str, Any]) -> None:
    required_arrays = {
        "candidate_position_AU": (CANDIDATE_MAP_STEP_COUNT + 1, STATE_COUNT, 3),
        "candidate_velocity_AU_per_year": (
            CANDIDATE_MAP_STEP_COUNT + 1,
            STATE_COUNT,
            3,
        ),
        "candidate_lifecycle": (CANDIDATE_MAP_STEP_COUNT + 1, STATE_COUNT),
        "candidate_minimum_source_distance_AU": (
            CANDIDATE_MAP_STEP_COUNT,
            STATE_COUNT,
        ),
        "refinement_position_AU": (
            REFINEMENT_MAP_STEP_COUNT + 1,
            STATE_COUNT,
            3,
        ),
        "refinement_velocity_AU_per_year": (
            REFINEMENT_MAP_STEP_COUNT + 1,
            STATE_COUNT,
            3,
        ),
        "refinement_lifecycle": (REFINEMENT_MAP_STEP_COUNT + 1, STATE_COUNT),
        "refinement_minimum_source_distance_AU": (
            REFINEMENT_MAP_STEP_COUNT,
            STATE_COUNT,
        ),
        "ias15_tracer_position_AU": (
            REFINEMENT_MAP_STEP_COUNT + 1,
            STATE_COUNT,
            3,
        ),
        "ias15_tracer_velocity_AU_per_year": (
            REFINEMENT_MAP_STEP_COUNT + 1,
            STATE_COUNT,
            3,
        ),
        "ias15_source_position_AU": (
            REFINEMENT_MAP_STEP_COUNT + 1,
            SOURCE_COUNT,
            3,
        ),
        "ias15_source_velocity_AU_per_year": (
            REFINEMENT_MAP_STEP_COUNT + 1,
            SOURCE_COUNT,
            3,
        ),
        "ias15_source_only_position_AU": (
            REFINEMENT_MAP_STEP_COUNT + 1,
            SOURCE_COUNT,
            3,
        ),
        "ias15_source_only_velocity_AU_per_year": (
            REFINEMENT_MAP_STEP_COUNT + 1,
            SOURCE_COUNT,
            3,
        ),
        "candidate_source_position_AU": (
            CANDIDATE_MAP_STEP_COUNT + 1,
            SOURCE_COUNT,
            3,
        ),
        "candidate_source_velocity_AU_per_year": (
            CANDIDATE_MAP_STEP_COUNT + 1,
            SOURCE_COUNT,
            3,
        ),
        "refinement_source_position_AU": (
            REFINEMENT_MAP_STEP_COUNT + 1,
            SOURCE_COUNT,
            3,
        ),
        "refinement_source_velocity_AU_per_year": (
            REFINEMENT_MAP_STEP_COUNT + 1,
            SOURCE_COUNT,
            3,
        ),
        "source_gm_AU3_per_year2": (SOURCE_COUNT,),
    }
    arrays = bundle.get("arrays")
    if type(arrays) is not dict or set(arrays) != set(required_arrays):
        raise M3NumericalError("numerical bundle array roster changed")
    for name, shape in required_arrays.items():
        value = arrays[name]
        if type(value) is not np.ndarray or value.shape != shape:
            raise M3NumericalError(f"numerical array shape changed: {name}")
        if name.endswith("lifecycle"):
            if value.dtype != np.dtype(np.uint8):
                raise M3NumericalError(f"lifecycle dtype changed: {name}")
            if bool(np.any(value > 2)):
                raise M3NumericalError(f"lifecycle code changed: {name}")
        elif value.dtype != np.dtype(np.float64):
            raise M3NumericalError(f"numerical array dtype changed: {name}")
        if not bool(np.all(np.isfinite(value))):
            raise M3NumericalError(f"numerical array is nonfinite: {name}")
        if name.endswith("minimum_source_distance_AU") and not bool(
            np.all(value > 0.0)
        ):
            raise M3NumericalError(
                f"minimum-source-distance evidence is not strictly positive: {name}"
            )
    for name in ("candidate_events", "refinement_events"):
        if type(bundle.get(name)) is not list:
            raise M3NumericalError(f"numerical bundle omits event ledger: {name}")
    receipts = bundle.get("receipts")
    if type(receipts) is not dict:
        raise M3NumericalError("numerical bundle omits receipts")
    runtime = receipts.get("runtime")
    if (
        type(runtime) is not dict
        or runtime.get("runtime_identity_stable") is not True
        or type(runtime.get("runtime_identity_start")) is not dict
        or runtime.get("runtime_identity_end")
        != runtime.get("runtime_identity_start")
        or runtime.get("runtime_identity_sha256")
        != hashlib.sha256(
            RUNTIME_IDENTITY_DOMAIN
            + _canonical_json(runtime.get("runtime_identity_start"))
        ).hexdigest()
    ):
        raise M3NumericalError("numerical runtime custody is absent or unstable")


def analyze_numerical_bundle(
    bundle: Mapping[str, Any],
    *,
    state_ids: Sequence[str],
    family_ids: Sequence[str],
) -> dict[str, Any]:
    _validate_numerical_bundle(bundle)
    arrays = bundle["arrays"]
    endpoint_custody = bundle["receipts"].get("endpoint_state_custody")
    if type(endpoint_custody) is not dict:
        raise M3NumericalError("endpoint-state custody receipt is absent")
    expected_trace_custody = {
        "candidate": _trace_node_custody(
            lane_id="candidate",
            position_AU=arrays["candidate_position_AU"],
            velocity_AU_per_year=arrays[
                "candidate_velocity_AU_per_year"
            ],
            lifecycle=arrays["candidate_lifecycle"],
            state_ids=state_ids,
            source_steps_per_map_step=CANDIDATE_SOURCE_STEPS_PER_MAP,
        ),
        "refinement": _trace_node_custody(
            lane_id="refinement",
            position_AU=arrays["refinement_position_AU"],
            velocity_AU_per_year=arrays[
                "refinement_velocity_AU_per_year"
            ],
            lifecycle=arrays["refinement_lifecycle"],
            state_ids=state_ids,
            source_steps_per_map_step=REFINEMENT_SOURCE_STEPS_PER_MAP,
        ),
    }
    if (
        endpoint_custody.get("candidate_trace_node_custody")
        != expected_trace_custody["candidate"]
        or endpoint_custody.get("refinement_trace_node_custody")
        != expected_trace_custody["refinement"]
    ):
        raise M3NumericalError(
            "endpoint trace hashes do not recompute from retained arrays"
        )
    gm = arrays["source_gm_AU3_per_year2"]
    candidate_per_state = _comparison_per_state(
        arrays["candidate_position_AU"],
        arrays["candidate_velocity_AU_per_year"],
        arrays["refinement_position_AU"][::2],
        arrays["refinement_velocity_AU_per_year"][::2],
        float(gm[0]),
    )
    reference_per_state = _comparison_per_state(
        arrays["refinement_position_AU"],
        arrays["refinement_velocity_AU_per_year"],
        arrays["ias15_tracer_position_AU"],
        arrays["ias15_tracer_velocity_AU_per_year"],
        float(gm[0]),
    )
    comparisons = {
        "candidate_vs_refinement": _comparison_record(
            candidate_per_state, state_ids, family_ids
        ),
        "refinement_vs_rebound_ias15": _comparison_record(
            reference_per_state, state_ids, family_ids
        ),
    }
    candidate_source_position = arrays["candidate_source_position_AU"]
    candidate_source_velocity = arrays["candidate_source_velocity_AU_per_year"]
    refinement_source_position = arrays["refinement_source_position_AU"]
    refinement_source_velocity = arrays[
        "refinement_source_velocity_AU_per_year"
    ]
    encounter = {
        "jx_candidate": _independent_interval_guards(
            arrays["candidate_position_AU"],
            arrays["candidate_velocity_AU_per_year"],
            candidate_source_position,
            candidate_source_velocity,
            gm,
            CANDIDATE_STEP_YEARS,
        ),
        "jx_refinement": _independent_interval_guards(
            arrays["refinement_position_AU"],
            arrays["refinement_velocity_AU_per_year"],
            refinement_source_position,
            refinement_source_velocity,
            gm,
            REFINEMENT_STEP_YEARS,
        ),
        "rebound_ias15": _independent_interval_guards(
            arrays["ias15_tracer_position_AU"],
            arrays["ias15_tracer_velocity_AU_per_year"],
            arrays["ias15_source_position_AU"],
            arrays["ias15_source_velocity_AU_per_year"],
            gm,
            REFINEMENT_STEP_YEARS,
        ),
    }
    source_no_backreaction = bool(
        _bitwise_equal(
            arrays["ias15_source_position_AU"],
            arrays["ias15_source_only_position_AU"],
        )
        and _bitwise_equal(
            arrays["ias15_source_velocity_AU_per_year"],
            arrays["ias15_source_only_velocity_AU_per_year"],
        )
    )
    initial_state_custody = {
        "candidate": {
            "tracer_positions_sha256": _array_sha256(
                arrays["candidate_position_AU"][0]
            ),
            "tracer_velocities_sha256": _array_sha256(
                arrays["candidate_velocity_AU_per_year"][0]
            ),
            "lifecycle_sha256": _array_sha256(
                arrays["candidate_lifecycle"][0]
            ),
        },
        "refinement": {
            "tracer_positions_sha256": _array_sha256(
                arrays["refinement_position_AU"][0]
            ),
            "tracer_velocities_sha256": _array_sha256(
                arrays["refinement_velocity_AU_per_year"][0]
            ),
            "lifecycle_sha256": _array_sha256(
                arrays["refinement_lifecycle"][0]
            ),
        },
        "rebound_ias15": {
            "tracer_positions_sha256": _array_sha256(
                arrays["ias15_tracer_position_AU"][0]
            ),
            "tracer_velocities_sha256": _array_sha256(
                arrays["ias15_tracer_velocity_AU_per_year"][0]
            ),
            "source_positions_sha256": _array_sha256(
                arrays["ias15_source_position_AU"][0]
            ),
            "source_velocities_sha256": _array_sha256(
                arrays["ias15_source_velocity_AU_per_year"][0]
            ),
        },
        "rebound_ias15_source_only": {
            "source_positions_sha256": _array_sha256(
                arrays["ias15_source_only_position_AU"][0]
            ),
            "source_velocities_sha256": _array_sha256(
                arrays["ias15_source_only_velocity_AU_per_year"][0]
            ),
        },
        "candidate_source": {
            "source_positions_sha256": _array_sha256(
                arrays["candidate_source_position_AU"][0]
            ),
            "source_velocities_sha256": _array_sha256(
                arrays["candidate_source_velocity_AU_per_year"][0]
            ),
        },
        "refinement_source": {
            "source_positions_sha256": _array_sha256(
                arrays["refinement_source_position_AU"][0]
            ),
            "source_velocities_sha256": _array_sha256(
                arrays["refinement_source_velocity_AU_per_year"][0]
            ),
        },
        "source_gm_sha256": _array_sha256(gm),
        "candidate_node_zero_index": 0,
        "refinement_node_zero_index": 0,
        "rebound_node_zero_index": 0,
    }
    return {
        "comparisons": comparisons,
        "encounter_and_timestep": encounter,
        "initial_state_custody": initial_state_custody,
        "source_no_backreaction_bitwise_exact": source_no_backreaction,
        "source_parent_custody": _plain(bundle["receipts"]["source_parent_custody"]),
        "split_resume": _plain(bundle["receipts"]["split_resume"]),
        "event_accounting": _plain(bundle["receipts"]["event_accounting"]),
        "endpoint_state_custody": _plain(
            endpoint_custody
        ),
        "runtime": _plain(bundle["receipts"]["runtime"]),
    }


def evaluate_gates(
    analysis: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate authenticated analysis against exact rosters and thresholds.

    This is not a standalone evidence trust boundary.  Calibration and
    official bundle validators first recompute array, event, source, split,
    checkpoint, and runtime custody before calling this evaluator.
    """

    validate_protocol(protocol)
    if type(analysis) is not dict or type(thresholds) is not dict:
        raise M3ProtocolError("gate inputs must be exact dictionaries")
    inputs = protocol.get("inputs")
    if type(inputs) is not dict:
        raise M3ProtocolError("protocol input roster is absent")
    state_ids = inputs.get("state_ids")
    family_ids = inputs.get("family_ids")
    if (
        type(state_ids) is not list
        or len(state_ids) != STATE_COUNT
        or any(type(value) is not str or not value for value in state_ids)
        or len(set(state_ids)) != STATE_COUNT
        or type(family_ids) is not list
        or len(family_ids) != STATE_COUNT
        or any(type(value) is not str or not value for value in family_ids)
    ):
        raise M3ProtocolError("protocol state/family roster is invalid")
    unique_families = tuple(dict.fromkeys(family_ids))
    if len(unique_families) != 12 or any(
        family_ids.count(family) != 9 for family in unique_families
    ):
        raise M3ProtocolError("protocol family roster is not twelve groups of nine")

    def canonical_nonnegative_hex(value: object, label: str) -> float:
        if type(value) is not str:
            raise M3ProtocolError(f"{label} is not float hex")
        try:
            parsed = float.fromhex(value)
        except ValueError as exc:
            raise M3ProtocolError(f"{label} is not float hex") from exc
        if (
            not math.isfinite(parsed)
            or parsed < 0.0
            or value != parsed.hex()
        ):
            raise M3ProtocolError(f"{label} is not canonical finite nonnegative hex")
        return parsed

    def finite_measurement(value: object, label: str) -> float:
        if type(value) is not float or not math.isfinite(value):
            raise M3ProtocolError(f"{label} is not a finite built-in float")
        return value

    gates: dict[str, Any] = {}
    initial = analysis.get("initial_state_custody")
    expected_initial_keys = {
        "candidate",
        "refinement",
        "rebound_ias15",
        "rebound_ias15_source_only",
        "candidate_source",
        "refinement_source",
        "source_gm_sha256",
        "candidate_node_zero_index",
        "refinement_node_zero_index",
        "rebound_node_zero_index",
    }
    if type(initial) is not dict or set(initial) != expected_initial_keys:
        raise M3ProtocolError("initial-state custody roster changed")
    tracer_expected = {
        "tracer_positions_sha256": inputs.get("tracer_positions_sha256"),
        "tracer_velocities_sha256": inputs.get("tracer_velocities_sha256"),
    }
    source_expected = {
        "source_positions_sha256": inputs.get(
            "source_initial_positions_sha256"
        ),
        "source_velocities_sha256": inputs.get(
            "source_initial_velocities_sha256"
        ),
    }
    for value in (*tracer_expected.values(), *source_expected.values()):
        _require_nonzero_sha256(value, "protocol genesis array digest")
    active_lifecycle_sha256 = _array_sha256(
        np.zeros(STATE_COUNT, dtype=np.uint8)
    )
    exact_initial_rows: dict[str, bool] = {}
    for lane in ("candidate", "refinement"):
        row = initial[lane]
        if type(row) is not dict or set(row) != {
            *tracer_expected,
            "lifecycle_sha256",
        }:
            raise M3ProtocolError("JX initial-state custody row changed")
        exact_initial_rows[lane] = bool(
            all(row[key] == value for key, value in tracer_expected.items())
            and row["lifecycle_sha256"] == active_lifecycle_sha256
        )
    rebound = initial["rebound_ias15"]
    if type(rebound) is not dict or set(rebound) != {
        *tracer_expected,
        *source_expected,
    }:
        raise M3ProtocolError("IAS15 combined genesis custody row changed")
    exact_initial_rows["rebound_ias15"] = bool(
        all(rebound[key] == value for key, value in tracer_expected.items())
        and all(rebound[key] == value for key, value in source_expected.items())
    )
    for lane in (
        "rebound_ias15_source_only",
        "candidate_source",
        "refinement_source",
    ):
        row = initial[lane]
        if type(row) is not dict or set(row) != set(source_expected):
            raise M3ProtocolError("source genesis custody row changed")
        exact_initial_rows[lane] = all(
            row[key] == value for key, value in source_expected.items()
        )
    exact_source_gm = bool(
        initial["source_gm_sha256"] == inputs.get("source_gm_sha256")
    )
    exact_node_zero = bool(
        type(initial["candidate_node_zero_index"]) is int
        and initial["candidate_node_zero_index"] == 0
        and type(initial["refinement_node_zero_index"]) is int
        and initial["refinement_node_zero_index"] == 0
        and type(initial["rebound_node_zero_index"]) is int
        and initial["rebound_node_zero_index"] == 0
    )
    exact_rosters_and_epoch = bool(
        inputs.get("state_count") == STATE_COUNT
        and len(inputs.get("m2_source", {}).get("source_ids", [])) == SOURCE_COUNT
        and inputs.get("m2_source", {}).get("source_ids", [None])[0]
        == inputs.get("m2_source", {}).get("primary_source_id")
        and inputs.get("m2_source", {}).get("epoch_jd_tdb_hex")
        == float.fromhex(inputs.get("m2_source", {}).get("epoch_jd_tdb_hex")).hex()
    )
    gates["initial_state_bitwise_exact"] = {
        "passed": bool(
            all(exact_initial_rows.values())
            and exact_source_gm
            and exact_node_zero
            and exact_rosters_and_epoch
        ),
        "lane_exact": exact_initial_rows,
        "source_gm_exact": exact_source_gm,
        "node_zero_exact": exact_node_zero,
        "rosters_and_epoch_exact": exact_rosters_and_epoch,
    }
    comparison_limits = thresholds["comparisons"]
    comparisons = analysis.get("comparisons")
    if type(comparisons) is not dict or set(comparisons) != set(COMPARISON_IDS):
        raise M3ProtocolError("analysis comparison roster changed")
    for comparison_id in COMPARISON_IDS:
        comparison = comparisons[comparison_id]
        if type(comparison) is not dict or set(comparison) != {"metrics"}:
            raise M3ProtocolError("analysis comparison record changed")
        metrics = comparison["metrics"]
        if type(metrics) is not dict or set(metrics) != set(COMPARISON_METRICS):
            raise M3ProtocolError("analysis metric roster changed")
        for metric in COMPARISON_METRICS:
            limit = float.fromhex(comparison_limits[comparison_id][metric])
            record = metrics[metric]
            if type(record) is not dict or set(record) != {
                "global_maximum_hex",
                "per_state_hex",
                "per_family_maximum_hex",
            }:
                raise M3ProtocolError("analysis metric record changed")
            state_values = record["per_state_hex"]
            family_values = record["per_family_maximum_hex"]
            if (
                type(state_values) is not dict
                or set(state_values) != set(state_ids)
                or type(family_values) is not dict
                or set(family_values) != set(unique_families)
            ):
                raise M3ProtocolError("analysis state/family metric roster changed")
            parsed_states = {
                state_id: canonical_nonnegative_hex(
                    state_values[state_id], f"{comparison_id}.{metric}.{state_id}"
                )
                for state_id in state_ids
            }
            parsed_families = {
                family: canonical_nonnegative_hex(
                    family_values[family], f"{comparison_id}.{metric}.{family}"
                )
                for family in unique_families
            }
            expected_global = max(parsed_states.values())
            observed_global = canonical_nonnegative_hex(
                record["global_maximum_hex"], f"{comparison_id}.{metric}.global"
            )
            if observed_global.hex() != expected_global.hex():
                raise M3ProtocolError("analysis global maximum is not derived from states")
            for family in unique_families:
                expected_family = max(
                    parsed_states[state_id]
                    for state_id, state_family in zip(
                        state_ids, family_ids, strict=True
                    )
                    if state_family == family
                )
                if parsed_families[family].hex() != expected_family.hex():
                    raise M3ProtocolError(
                        "analysis family maximum is not derived from states"
                    )
            per_state = {
                state_id: value <= limit
                for state_id, value in parsed_states.items()
            }
            per_family = {
                family_id: value <= limit
                for family_id, value in parsed_families.items()
            }
            passed = (
                observed_global <= limit
                and all(per_state.values())
                and all(per_family.values())
            )
            gates[f"{comparison_id}.{metric}"] = {
                "passed": passed,
                "threshold_hex": limit.hex(),
                "global_maximum_hex": record["global_maximum_hex"],
                "individual_state_passed": per_state,
                "family_passed": per_family,
            }
    encounter = analysis.get("encounter_and_timestep")
    expected_lanes = ("jx_candidate", "jx_refinement", "rebound_ias15")
    if type(encounter) is not dict or set(encounter) != set(expected_lanes):
        raise M3ProtocolError("encounter/timestep lane roster changed")
    expected_intervals = {
        "jx_candidate": CANDIDATE_MAP_STEP_COUNT,
        "jx_refinement": REFINEMENT_MAP_STEP_COUNT,
        "rebound_ias15": REFINEMENT_MAP_STEP_COUNT,
    }
    for lane_id in expected_lanes:
        record = encounter[lane_id]
        guard_keys = {
            "interval_count",
            "minimum_instantaneous_source_distance_AU",
            "minimum_certified_interval_clearance_AU",
            "maximum_instantaneous_primary_phase_rad",
            "maximum_periapse_primary_phase_rad",
            "uniform_relative_acceleration_bound_AU_per_year2",
            "bootstrap_policy",
            "algorithmic_guard_radius_AU",
            "all_intervals_clear",
            "source_pair_safety",
            "instantaneous_phase_passed",
            "periapse_phase_passed",
        }
        if type(record) is not dict or set(record) != guard_keys:
            raise M3ProtocolError("encounter/timestep record is malformed")
        if type(record.get("interval_count")) is not int or record[
            "interval_count"
        ] != expected_intervals[lane_id]:
            raise M3ProtocolError("encounter interval count changed")
        clearance = finite_measurement(
            record.get("minimum_certified_interval_clearance_AU"),
            f"{lane_id} tracer-source clearance",
        )
        instantaneous_distance = finite_measurement(
            record.get("minimum_instantaneous_source_distance_AU"),
            f"{lane_id} instantaneous tracer-source distance",
        )
        acceleration_bound = finite_measurement(
            record.get("uniform_relative_acceleration_bound_AU_per_year2"),
            f"{lane_id} acceleration bound",
        )
        instantaneous_phase = finite_measurement(
            record.get("maximum_instantaneous_primary_phase_rad"),
            f"{lane_id} instantaneous phase",
        )
        periapse_phase = finite_measurement(
            record.get("maximum_periapse_primary_phase_rad"),
            f"{lane_id} periapse phase",
        )
        if (
            instantaneous_phase < 0.0
            or periapse_phase < 0.0
            or instantaneous_distance <= 0.0
            or instantaneous_distance < clearance
            or record.get("algorithmic_guard_radius_AU")
            != ALGORITHMIC_GUARD_RADIUS_AU
            or record.get("bootstrap_policy")
            != (
                "FIRST_CROSSING_CONTRADICTION_FROM_START_DISTANCE_MINUS_"
                "RELATIVE_SPEED_H_MINUS_HALF_UNIFORM_ACCELERATION_H2_V1"
            )
            or acceleration_bound.hex()
            != inputs[
                "uniform_relative_acceleration_bound_AU_per_year2_hex"
            ]
        ):
            raise M3ProtocolError("encounter/timestep phase is negative")
        clearance_passed = clearance > ALGORITHMIC_GUARD_RADIUS_AU
        instantaneous_passed = (
            instantaneous_phase <= MAXIMUM_PRIMARY_DYNAMICAL_PHASE
        )
        periapse_passed = periapse_phase <= MAXIMUM_PRIMARY_DYNAMICAL_PHASE
        if (
            record.get("all_intervals_clear") is not clearance_passed
            or record.get("instantaneous_phase_passed") is not instantaneous_passed
            or record.get("periapse_phase_passed") is not periapse_passed
        ):
            raise M3ProtocolError("encounter/timestep pass flags are not derived")
        source_pair = record.get("source_pair_safety")
        source_pair_keys = {
            "pair_count",
            "interval_count",
            "minimum_instantaneous_source_pair_distance_AU",
            "minimum_certified_source_pair_interval_clearance_AU",
            "all_source_pair_intervals_clear",
            "rounding_enclosure_relative_pad_hex",
        }
        if type(source_pair) is not dict or set(source_pair) != source_pair_keys:
            raise M3ProtocolError("source-pair safety receipt is absent")
        pair_instantaneous = finite_measurement(
            source_pair.get("minimum_instantaneous_source_pair_distance_AU"),
            f"{lane_id} instantaneous source-pair distance",
        )
        pair_clearance = finite_measurement(
            source_pair.get(
                "minimum_certified_source_pair_interval_clearance_AU"
            ),
            f"{lane_id} source-pair clearance",
        )
        pair_passed = pair_clearance > ALGORITHMIC_GUARD_RADIUS_AU
        if (
            source_pair.get("pair_count") != 45
            or type(source_pair.get("pair_count")) is not int
            or source_pair.get("interval_count") != expected_intervals[lane_id]
            or type(source_pair.get("interval_count")) is not int
            or source_pair.get("all_source_pair_intervals_clear") is not pair_passed
            or pair_instantaneous <= 0.0
            or pair_instantaneous < pair_clearance
            or source_pair.get("rounding_enclosure_relative_pad_hex")
            != _GUARD_ROUNDING_PAD.hex()
        ):
            raise M3ProtocolError("source-pair safety receipt is inconsistent")
        gates[f"{lane_id}.interval_clearance"] = {
            "passed": clearance_passed,
            "minimum_certified_interval_clearance_AU_hex": clearance.hex(),
            "threshold_AU_hex": ALGORITHMIC_GUARD_RADIUS_AU.hex(),
        }
        gates[f"{lane_id}.source_pair_clearance"] = {
            "passed": pair_passed,
            "minimum_certified_source_pair_interval_clearance_AU_hex": (
                pair_clearance.hex()
            ),
            "threshold_AU_hex": ALGORITHMIC_GUARD_RADIUS_AU.hex(),
            "pair_count": 45,
        }
        gates[f"{lane_id}.instantaneous_phase"] = {
            "passed": instantaneous_passed,
            "maximum_hex": instantaneous_phase.hex(),
            "threshold_hex": MAXIMUM_PRIMARY_DYNAMICAL_PHASE.hex(),
        }
        gates[f"{lane_id}.periapse_phase"] = {
            "passed": periapse_passed,
            "maximum_hex": periapse_phase.hex(),
            "threshold_hex": MAXIMUM_PRIMARY_DYNAMICAL_PHASE.hex(),
        }
    valid_hash = lambda value: (
        type(value) is str
        and len(value) == 64
        and value != "0" * 64
        and all(character in "0123456789abcdef" for character in value)
    )
    custody = analysis["source_parent_custody"]
    custody_keys = {
        "parent_path_content_sha256",
        "parent_dynamic_node_sha256",
        "candidate_exact_parent_slices",
        "refinement_exact_parent_slices",
        "parent_unchanged_after_execution",
        "candidate",
        "refinement",
        "guard_overlay_sha256",
    }
    child_custody_keys = {
        "parent_path_content_sha256_before",
        "parent_path_content_sha256_after",
        "parent_unchanged",
        "map_slice_count",
        "slice_dynamic_digest",
        "tile_binding_digest",
        "first_global_source_step",
        "final_global_source_step",
        "every_slice_verified_by_constructor",
    }
    if type(custody) is not dict or set(custody) != custody_keys:
        raise M3ProtocolError("source-parent custody schema changed")
    parent_expected = inputs["m2_source"]["content_sha256"]
    dynamic_expected = inputs["m2_source"]["dynamic_node_sha256"]
    source_expectations = inputs.get("m2_execution_source_custody")
    if (
        type(source_expectations) is not dict
        or set(source_expectations)
        != {
            "parent_path_content_sha256",
            "parent_dynamic_node_sha256",
            "guard_overlay_sha256",
            "candidate",
            "refinement",
            "annual_source_slice_dynamic_node_sha256",
            "annual_source_tile_binding_sha256",
        }
        or source_expectations.get("parent_path_content_sha256")
        != parent_expected
        or source_expectations.get("parent_dynamic_node_sha256")
        != dynamic_expected
        or not valid_hash(source_expectations.get("guard_overlay_sha256"))
    ):
        raise M3ProtocolError("predeclared M2 execution custody is invalid")
    parent_exact = bool(
        custody["parent_path_content_sha256"] == parent_expected
        and custody["parent_dynamic_node_sha256"] == dynamic_expected
        and custody["guard_overlay_sha256"]
        == source_expectations["guard_overlay_sha256"]
    )
    custody_child_pass: dict[str, bool] = {}
    for lane, expected_maps in (
        ("candidate", CANDIDATE_MAP_STEP_COUNT),
        ("refinement", REFINEMENT_MAP_STEP_COUNT),
    ):
        child = custody[lane]
        expected_child = source_expectations[lane]
        if type(child) is not dict or set(child) != child_custody_keys:
            raise M3ProtocolError("source-parent lane custody schema changed")
        if (
            type(expected_child) is not dict
            or set(expected_child)
            != {
                "map_slice_count",
                "slice_dynamic_digest",
                "tile_binding_digest",
            }
            or expected_child.get("map_slice_count") != expected_maps
            or not valid_hash(expected_child.get("slice_dynamic_digest"))
            or not valid_hash(expected_child.get("tile_binding_digest"))
        ):
            raise M3ProtocolError("predeclared M2 lane custody is invalid")
        derived = bool(
            child["parent_path_content_sha256_before"] == parent_expected
            and child["parent_path_content_sha256_after"] == parent_expected
            and child["parent_unchanged"] is True
            and type(child["map_slice_count"]) is int
            and child["map_slice_count"] == expected_maps
            and child["slice_dynamic_digest"]
            == expected_child["slice_dynamic_digest"]
            and child["tile_binding_digest"]
            == expected_child["tile_binding_digest"]
            and type(child["first_global_source_step"]) is int
            and child["first_global_source_step"] == 0
            and type(child["final_global_source_step"]) is int
            and child["final_global_source_step"] == SOURCE_STEP_COUNT
            and child["every_slice_verified_by_constructor"] is True
        )
        top_flag = custody[f"{lane}_exact_parent_slices"]
        if type(top_flag) is not bool or top_flag is not derived:
            raise M3ProtocolError("source-parent lane custody flag is not derived")
        custody_child_pass[lane] = derived
    derived_parent_unchanged = bool(
        custody["candidate"]["parent_unchanged"] is True
        and custody["refinement"]["parent_unchanged"] is True
    )
    if (
        type(custody["parent_unchanged_after_execution"]) is not bool
        or custody["parent_unchanged_after_execution"]
        is not derived_parent_unchanged
    ):
        raise M3ProtocolError("source-parent final custody flag is not derived")
    gates["source_parent_custody"] = {
        "passed": bool(
            parent_exact
            and all(custody_child_pass.values())
            and derived_parent_unchanged
        )
    }
    if type(analysis.get("source_no_backreaction_bitwise_exact")) is not bool:
        raise M3ProtocolError("source backreaction receipt is not boolean")
    gates["source_no_backreaction"] = {
        "passed": analysis["source_no_backreaction_bitwise_exact"]
    }
    split = analysis["split_resume"]
    if type(split) is not dict or set(split) != {
        "candidate",
        "refinement",
        "candidate_bitwise_exact",
        "refinement_bitwise_exact",
    }:
        raise M3ProtocolError("split/resume receipt schema changed")
    endpoint_custody = analysis.get("endpoint_state_custody")
    if type(endpoint_custody) is not dict or set(endpoint_custody) != {
        "candidate_state_anchors",
        "refinement_state_anchors",
        "candidate_trace_node_custody",
        "refinement_trace_node_custody",
    }:
        raise M3ProtocolError("endpoint-state custody schema changed")
    endpoint_hashes = {
        "candidate": endpoint_custody["candidate_state_anchors"],
        "refinement": endpoint_custody["refinement_state_anchors"],
    }
    split_required_hashes = (
        "genesis_state_sha256",
        "first_half_state_sha256",
        "expected_split_state_sha256",
        "split_checkpoint_content_sha256",
        "final_checkpoint_content_sha256",
        "loaded_split_checkpoint_state_sha256",
        "loaded_final_checkpoint_state_sha256",
        "terminal_state_sha256",
        "expected_terminal_state_sha256",
        "expected_event_ledger_sha256",
        "split_event_ledger_sha256",
    )
    split_pass: dict[str, bool] = {}
    split_child_keys = {
        "lane_id",
        *split_required_hashes,
        "first_half_matches_one_map_trace",
        "bitwise_exact",
        "lifecycle_state_bitwise_exact",
        "event_ledger_bitwise_exact",
        "annual_checkpoint_count",
        "annual_source_slice_dynamic_node_sha256",
        "annual_source_tile_binding_sha256",
    }
    for lane in ("candidate", "refinement"):
        child = split[lane]
        hashes = endpoint_hashes[lane]
        if (
            type(child) is not dict
            or set(child) != split_child_keys
            or type(hashes) is not dict
            or set(hashes) != {"genesis", "split", "terminal"}
            or any(not valid_hash(value) for value in hashes.values())
            or any(
            not valid_hash(child.get(key)) for key in split_required_hashes
            )
        ):
            raise M3ProtocolError("split/resume digest custody changed")
        derived = bool(
            child["lane_id"] == lane
            and child["genesis_state_sha256"] == hashes["genesis"]
            and child["first_half_state_sha256"]
            == child["expected_split_state_sha256"]
            == child["loaded_split_checkpoint_state_sha256"]
            == hashes["split"]
            and child["terminal_state_sha256"]
            == child["expected_terminal_state_sha256"]
            == child["loaded_final_checkpoint_state_sha256"]
            == hashes["terminal"]
            and child["expected_event_ledger_sha256"]
            == child["split_event_ledger_sha256"]
            and type(child.get("annual_checkpoint_count")) is int
            and child["annual_checkpoint_count"] == HORIZON_YEARS
            and type(child.get("annual_source_slice_dynamic_node_sha256"))
            is list
            and len(child["annual_source_slice_dynamic_node_sha256"])
            == HORIZON_YEARS
            and all(
                valid_hash(value)
                for value in child["annual_source_slice_dynamic_node_sha256"]
            )
            and child["annual_source_slice_dynamic_node_sha256"]
            == source_expectations[
                "annual_source_slice_dynamic_node_sha256"
            ]
            and type(child.get("annual_source_tile_binding_sha256")) is list
            and len(child["annual_source_tile_binding_sha256"])
            == HORIZON_YEARS
            and all(
                valid_hash(value)
                for value in child["annual_source_tile_binding_sha256"]
            )
            and child["annual_source_tile_binding_sha256"]
            == source_expectations["annual_source_tile_binding_sha256"]
        )
        for flag in (
            "first_half_matches_one_map_trace",
            "bitwise_exact",
            "lifecycle_state_bitwise_exact",
            "event_ledger_bitwise_exact",
        ):
            if type(child.get(flag)) is not bool or child[flag] is not derived:
                raise M3ProtocolError("split/resume flag is not derived")
        top_flag = split[f"{lane}_bitwise_exact"]
        if type(top_flag) is not bool or top_flag is not derived:
            raise M3ProtocolError("split/resume top-level flag is not derived")
        split_pass[lane] = derived
    gates["candidate_split_resume"] = {"passed": split_pass["candidate"]}
    gates["refinement_split_resume"] = {"passed": split_pass["refinement"]}
    events = analysis["event_accounting"]
    if type(events) is not dict or set(events) != {
        "candidate",
        "refinement",
        "candidate_exact",
        "refinement_exact",
    }:
        raise M3ProtocolError("event-accounting receipt schema changed")
    event_pass: dict[str, bool] = {}
    event_keys = {
        "exact",
        "route_record_count",
        "removal_record_count",
        "terminal_lane_count",
        "lifecycle_transition_count",
        "no_reactivation",
        "ledger_content_sha256",
    }
    for lane in ("candidate", "refinement"):
        child = events[lane]
        if type(child) is not dict or set(child) != event_keys:
            raise M3ProtocolError("event-accounting lane schema changed")
        for count_key in (
            "route_record_count",
            "removal_record_count",
            "terminal_lane_count",
            "lifecycle_transition_count",
        ):
            if type(child[count_key]) is not int or child[count_key] < 0:
                raise M3ProtocolError("event-accounting count is invalid")
        derived = bool(
            child["exact"] is True
            and child["no_reactivation"] is True
            and child["removal_record_count"] == child["terminal_lane_count"]
            == child["lifecycle_transition_count"]
            and valid_hash(child["ledger_content_sha256"])
        )
        top_flag = events[f"{lane}_exact"]
        if type(top_flag) is not bool or top_flag is not derived:
            raise M3ProtocolError("event-accounting top-level flag is not derived")
        event_pass[lane] = derived
    gates["event_accounting"] = {
        "passed": all(event_pass.values())
    }
    engine_anchors_retained = bool(
        all(
            type(anchors) is dict
            and set(anchors) == {"genesis", "split", "terminal"}
            and all(valid_hash(value) for value in anchors.values())
            and len(set(anchors.values())) == 3
            for anchors in endpoint_hashes.values()
        )
    )
    expected_roster_sha256 = hashlib.sha256(
        TRACE_ROSTER_DOMAIN + _canonical_json(state_ids)
    ).hexdigest()
    trace_pass: dict[str, bool] = {}
    trace_keys = {
        "lane_id",
        "node_count",
        "state_roster_sha256",
        "source_steps_per_map_step",
        "first_global_source_step",
        "final_global_source_step",
        "node_sha256",
        "content_sha256",
    }
    for lane, map_count, ratio in (
        (
            "candidate",
            CANDIDATE_MAP_STEP_COUNT,
            CANDIDATE_SOURCE_STEPS_PER_MAP,
        ),
        (
            "refinement",
            REFINEMENT_MAP_STEP_COUNT,
            REFINEMENT_SOURCE_STEPS_PER_MAP,
        ),
    ):
        trace = endpoint_custody[f"{lane}_trace_node_custody"]
        node_hashes = trace.get("node_sha256") if type(trace) is dict else None
        derived = bool(
            type(trace) is dict
            and set(trace) == trace_keys
            and trace.get("lane_id") == lane
            and type(trace.get("node_count")) is int
            and trace.get("node_count") == map_count + 1
            and trace.get("state_roster_sha256") == expected_roster_sha256
            and type(trace.get("source_steps_per_map_step")) is int
            and trace.get("source_steps_per_map_step") == ratio
            and trace.get("first_global_source_step") == 0
            and trace.get("final_global_source_step") == SOURCE_STEP_COUNT
            and type(node_hashes) is list
            and len(node_hashes) == map_count + 1
            and all(valid_hash(value) for value in node_hashes)
            and len(set(node_hashes)) == len(node_hashes)
            and trace.get("content_sha256")
            == _content_sha256(TRACE_CUSTODY_DOMAIN, trace)
        )
        trace_pass[lane] = derived
    gates["every_map_endpoint_trace_hash_recomputed"] = {
        "passed": bool(engine_anchors_retained and all(trace_pass.values())),
        "candidate_trace_content_sha256": endpoint_custody[
            "candidate_trace_node_custody"
        ].get("content_sha256"),
        "refinement_trace_content_sha256": endpoint_custody[
            "refinement_trace_node_custody"
        ].get("content_sha256"),
    }
    runtime = analysis.get("runtime")
    if type(runtime) is not dict:
        raise M3ProtocolError("runtime custody receipt is absent")
    start_identity = runtime.get("runtime_identity_start")
    end_identity = runtime.get("runtime_identity_end")
    stable = runtime.get("runtime_identity_stable")
    if (
        type(start_identity) is not dict
        or type(end_identity) is not dict
        or type(stable) is not bool
    ):
        raise M3ProtocolError("runtime custody receipt is malformed")
    derived_sha256 = hashlib.sha256(
        RUNTIME_IDENTITY_DOMAIN + _canonical_json(start_identity)
    ).hexdigest()
    derived_stable = start_identity == end_identity
    if (
        stable is not derived_stable
        or runtime.get("runtime_identity_sha256") != derived_sha256
    ):
        raise M3ProtocolError("runtime custody receipt is inconsistent")
    gates["runtime_identity_stable"] = {
        "passed": derived_stable,
        "runtime_identity_sha256": derived_sha256,
    }
    return gates


def _primary_relative_to_barycentric(
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    gm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    position = np.ascontiguousarray(relative_position, dtype=np.float64)
    velocity = np.ascontiguousarray(relative_velocity, dtype=np.float64)
    gm = np.ascontiguousarray(gm, dtype=np.float64)
    if (
        position.shape != (SOURCE_COUNT, 3)
        or velocity.shape != position.shape
        or gm.shape != (SOURCE_COUNT,)
        or not bool(np.all(position[0] == 0.0))
        or not bool(np.all(velocity[0] == 0.0))
    ):
        raise M3NumericalError("source genesis is not primary-relative")
    total_gm = float(np.sum(gm, dtype=np.float64))
    position_center = (
        np.sum(position * gm[:, None], axis=0, dtype=np.float64) / total_gm
    )
    velocity_center = (
        np.sum(velocity * gm[:, None], axis=0, dtype=np.float64) / total_gm
    )
    return (
        np.ascontiguousarray(position - position_center[None, :]),
        np.ascontiguousarray(velocity - velocity_center[None, :]),
    )


def _load_exact_rebound() -> Any:
    """Import only the exact independently pinned REBOUND runtime."""

    try:
        version = importlib.metadata.version("rebound")
    except importlib.metadata.PackageNotFoundError as exc:
        raise M3NumericalError("REBOUND is not installed") from exc
    if version != REQUIRED_REBOUND_VERSION:
        raise M3NumericalError(
            f"REBOUND must be exactly {REQUIRED_REBOUND_VERSION}, got {version}"
        )
    import rebound

    if str(rebound.__version__) != REQUIRED_REBOUND_VERSION:
        raise M3NumericalError("REBOUND module/distribution versions disagree")
    probe = rebound.Simulation()
    probe.integrator = "ias15"
    try:
        float(probe.integrator.epsilon)
        str(probe.integrator.adaptive_mode)
        float(probe.integrator.min_dt)
    except (AttributeError, TypeError, ValueError) as exc:
        raise M3NumericalError(
            "REBOUND 5.1.1 IAS15 configuration surface is unavailable"
        ) from exc
    return rebound


def _new_rebound_simulation(
    rebound: Any,
    source_position: np.ndarray,
    source_velocity: np.ndarray,
    gm: np.ndarray,
    tracer_position: np.ndarray | None,
    tracer_velocity: np.ndarray | None,
) -> Any:
    simulation = rebound.Simulation()
    # With G=1, REBOUND's masses are the exact retained GM values and time is
    # the Julian-year coordinate used by every other M3 lane.
    simulation.G = 1.0
    for index in range(SOURCE_COUNT):
        simulation.add(
            m=float(gm[index]),
            x=float(source_position[index, 0]),
            y=float(source_position[index, 1]),
            z=float(source_position[index, 2]),
            vx=float(source_velocity[index, 0]),
            vy=float(source_velocity[index, 1]),
            vz=float(source_velocity[index, 2]),
        )
    if tracer_position is not None or tracer_velocity is not None:
        if (
            tracer_position is None
            or tracer_velocity is None
            or tracer_position.shape != (STATE_COUNT, 3)
            or tracer_velocity.shape != (STATE_COUNT, 3)
        ):
            raise M3NumericalError("REBOUND tracer genesis has an invalid shape")
        primary_position = source_position[0]
        primary_velocity = source_velocity[0]
        for index in range(STATE_COUNT):
            absolute_position = tracer_position[index] + primary_position
            absolute_velocity = tracer_velocity[index] + primary_velocity
            simulation.add(
                m=0.0,
                x=float(absolute_position[0]),
                y=float(absolute_position[1]),
                z=float(absolute_position[2]),
                vx=float(absolute_velocity[0]),
                vy=float(absolute_velocity[1]),
                vz=float(absolute_velocity[2]),
            )
    simulation.N_active = SOURCE_COUNT
    simulation.testparticle_type = 0
    simulation.integrator = "ias15"
    # REBOUND 5.1.1 exposes IAS15's configuration through the generic
    # IntegratorConfiguration object.  The pre-5 API (``ri_ias15``) is
    # intentionally not accepted by this exact-version comparator.
    simulation.integrator.epsilon = IAS15_EPSILON
    simulation.integrator.adaptive_mode = IAS15_ADAPTIVE_MODE
    simulation.integrator.min_dt = 0.0
    simulation.dt = IAS15_INITIAL_STEP_YEARS
    simulation.collision = "none"
    if (
        simulation.N_active != SOURCE_COUNT
        or simulation.testparticle_type != 0
        or str(simulation.integrator) != "ias15"
        or float(simulation.integrator.epsilon).hex() != IAS15_EPSILON.hex()
        or str(simulation.integrator.adaptive_mode) != IAS15_ADAPTIVE_MODE
        or float(simulation.integrator.min_dt).hex() != (0.0).hex()
        or float(simulation.dt).hex() != IAS15_INITIAL_STEP_YEARS.hex()
        or any(
            float(simulation.particles[index].m).hex() != (0.0).hex()
            for index in range(SOURCE_COUNT, simulation.N)
        )
    ):
        raise M3NumericalError(
            "REBOUND did not retain the exact active/massless IAS15 contract"
        )
    return simulation


def _rebound_node(simulation: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    primary = simulation.particles[0]
    source_position = np.empty((SOURCE_COUNT, 3), dtype=np.float64)
    source_velocity = np.empty_like(source_position)
    for index in range(SOURCE_COUNT):
        particle = simulation.particles[index]
        source_position[index] = (
            particle.x - primary.x,
            particle.y - primary.y,
            particle.z - primary.z,
        )
        source_velocity[index] = (
            particle.vx - primary.vx,
            particle.vy - primary.vy,
            particle.vz - primary.vz,
        )
    tracer_position = np.empty((max(simulation.N - SOURCE_COUNT, 0), 3), dtype=np.float64)
    tracer_velocity = np.empty_like(tracer_position)
    for lane in range(simulation.N - SOURCE_COUNT):
        particle = simulation.particles[SOURCE_COUNT + lane]
        tracer_position[lane] = (
            particle.x - primary.x,
            particle.y - primary.y,
            particle.z - primary.z,
        )
        tracer_velocity[lane] = (
            particle.vx - primary.vx,
            particle.vy - primary.vy,
            particle.vz - primary.vz,
        )
    return source_position, source_velocity, tracer_position, tracer_velocity


def _run_rebound_ias15(
    *,
    initial_source_position: np.ndarray,
    initial_source_velocity: np.ndarray,
    source_gm: np.ndarray,
    initial_tracer_position: np.ndarray,
    initial_tracer_velocity: np.ndarray,
) -> dict[str, np.ndarray]:
    """Run combined and source-only exact-version IAS15 comparators."""

    rebound = _load_exact_rebound()
    bary_position, bary_velocity = _primary_relative_to_barycentric(
        initial_source_position, initial_source_velocity, source_gm
    )
    combined = _new_rebound_simulation(
        rebound,
        bary_position,
        bary_velocity,
        source_gm,
        initial_tracer_position,
        initial_tracer_velocity,
    )
    source_only = _new_rebound_simulation(
        rebound,
        bary_position,
        bary_velocity,
        source_gm,
        None,
        None,
    )
    nodes = REFINEMENT_MAP_STEP_COUNT + 1
    source_position = np.empty((nodes, SOURCE_COUNT, 3), dtype=np.float64)
    source_velocity = np.empty_like(source_position)
    source_only_position = np.empty_like(source_position)
    source_only_velocity = np.empty_like(source_position)
    tracer_position = np.empty((nodes, STATE_COUNT, 3), dtype=np.float64)
    tracer_velocity = np.empty_like(tracer_position)
    for node in range(nodes):
        if node:
            target = node * REFINEMENT_STEP_YEARS
            combined.integrate(target, exact_finish_time=1)
            source_only.integrate(target, exact_finish_time=1)
            if (
                float(combined.t).hex() != float(target).hex()
                or float(source_only.t).hex() != float(target).hex()
            ):
                raise M3NumericalError(
                    "IAS15 did not stop on the exact refinement node"
                )
        (
            source_position[node],
            source_velocity[node],
            tracer_position[node],
            tracer_velocity[node],
        ) = _rebound_node(combined)
        (
            source_only_position[node],
            source_only_velocity[node],
            ignored_position,
            ignored_velocity,
        ) = _rebound_node(source_only)
        if ignored_position.shape != (0, 3) or ignored_velocity.shape != (0, 3):
            raise M3NumericalError("source-only REBOUND lane contains tracers")
        if node == 0 and (
            not _bitwise_equal(source_position[0], initial_source_position)
            or not _bitwise_equal(source_velocity[0], initial_source_velocity)
            or not _bitwise_equal(
                source_only_position[0], initial_source_position
            )
            or not _bitwise_equal(
                source_only_velocity[0], initial_source_velocity
            )
            or not _bitwise_equal(tracer_position[0], initial_tracer_position)
            or not _bitwise_equal(tracer_velocity[0], initial_tracer_velocity)
        ):
            raise M3NumericalError(
                "REBOUND barycentric construction changed the exact node-zero genesis"
            )
    return {
        "ias15_tracer_position_AU": tracer_position,
        "ias15_tracer_velocity_AU_per_year": tracer_velocity,
        "ias15_source_position_AU": source_position,
        "ias15_source_velocity_AU_per_year": source_velocity,
        "ias15_source_only_position_AU": source_only_position,
        "ias15_source_only_velocity_AU_per_year": source_only_velocity,
    }


def _routed_event_record(event: Any) -> dict[str, Any]:
    """Serialize one route event without discarding any physical witness."""

    return {
        "event_type": event.event_type,
        "source_index": event.source_index,
        "source_id": event.source_id,
        "map_step_fraction_hex": event.map_step_fraction.hex(),
        "source_interval_offset": event.source_interval_offset,
        "interval_fraction_hex": event.interval_fraction.hex(),
        "relative_position_AU_hex": [
            float(value).hex() for value in event.relative_position_AU
        ],
        "relative_velocity_AU_per_year_hex": [
            float(value).hex()
            for value in event.relative_velocity_AU_per_year
        ],
        "surface_distance_AU_hex": event.surface_distance_AU.hex(),
        "surface_threshold_AU_hex": event.surface_threshold_AU.hex(),
        "surface_residual_AU_hex": event.surface_residual_AU.hex(),
        "surface_tolerance_AU_hex": event.surface_tolerance_AU.hex(),
        "primary_specific_energy_AU2_per_year2_hex": (
            event.primary_specific_energy_AU2_per_year2.hex()
        ),
        "primary_radial_velocity_AU_per_year_hex": (
            event.primary_radial_velocity_AU_per_year.hex()
        ),
        "surface_radial_velocity_AU_per_year_hex": (
            event.surface_radial_velocity_AU_per_year.hex()
        ),
        "location_method": event.location_method,
    }


def _route_record(row: Any, *, global_map_step: int) -> dict[str, Any]:
    route = row.route_result
    runtime_identity = dict(_plain(route.runtime_identity))
    runtime_identity.pop("source_tile_sha256", None)
    event_content = [_routed_event_record(event) for event in route.events]
    return {
        "record_kind": "ROUTED_REJOIN",
        "global_map_step": global_map_step,
        "lane_index": row.lane_index,
        "tracer_index": row.tracer_index,
        "guard_phase": row.guard_phase,
        "guard_reason": row.guard_reason,
        "guard_source_index": row.guard_source_index,
        "source_tile_sha256": route.source_tile_sha256,
        "source_segment_sha256": route.source_segment_sha256,
        "global_source_step": route.global_source_step,
        "source_steps_per_map_step": route.source_steps_per_map_step,
        "accepted_step_count": route.accepted_step_count,
        "derivative_evaluation_count": route.derivative_evaluation_count,
        "endpoint_relative_position_AU_hex": [
            float(value).hex()
            for value in route.endpoint_relative_position_AU
        ],
        "endpoint_relative_velocity_AU_per_year_hex": [
            float(value).hex()
            for value in route.endpoint_relative_velocity_AU_per_year
        ],
        "minimum_evaluated_stage_distance_AU_hex": (
            route.minimum_evaluated_stage_distance_AU.hex()
        ),
        "minimum_committed_stage_distance_AU_hex": (
            route.minimum_committed_stage_distance_AU.hex()
        ),
        "maximum_accepted_primary_dynamical_phase_radians_hex": (
            route.maximum_accepted_primary_dynamical_phase_radians.hex()
        ),
        "maximum_committed_primary_dynamical_phase_radians_hex": (
            route.maximum_committed_primary_dynamical_phase_radians.hex()
        ),
        "completed_map_fraction_hex": route.completed_map_fraction.hex(),
        "event_count": len(event_content),
        "event_content": event_content,
        "runtime_identity_sha256": hashlib.sha256(
            EVENT_DOMAIN + _canonical_json(runtime_identity)
        ).hexdigest(),
    }


def _removal_record(row: Any) -> dict[str, Any]:
    event_content = [_routed_event_record(event) for event in row.events]
    return {
        "record_kind": "TERMINAL_REMOVAL",
        "content_sha256": row.content_sha256,
        "lane_index": row.lane_index,
        "tracer_index": row.tracer_index,
        "lifecycle_code": row.lifecycle_code,
        "global_source_step": row.global_source_step,
        "source_steps_per_map_step": row.source_steps_per_map_step,
        "guard_phase": row.guard_phase,
        "guard_reason": row.guard_reason,
        "guard_source_index": row.guard_source_index,
        "source_segment_sha256": row.source_segment_sha256,
        "route_runtime_identity_sha256": row.route_runtime_identity_sha256,
        "accepted_step_count": row.accepted_step_count,
        "derivative_evaluation_count": row.derivative_evaluation_count,
        "minimum_committed_stage_distance_AU_hex": (
            row.minimum_committed_stage_distance_AU.hex()
        ),
        "maximum_committed_primary_dynamical_phase_radians_hex": (
            row.maximum_committed_primary_dynamical_phase_radians.hex()
        ),
        "event_count": len(event_content),
        "event_content": event_content,
    }


def _run_jx_trace(
    *,
    lane_id: str,
    initial_state: Any,
    backend: Any,
    source: Any,
    overlay: Any,
    source_steps_per_map: int,
    map_step_count: int,
    route_spec: Any,
) -> dict[str, Any]:
    """Advance exactly one map per call and retain every committed endpoint."""

    from jxplanetx.cuda_restricted_wh import FixedStepCUDARestrictedWHSpec
    from jxplanetx.engine.backends import resolve_backend
    from jxplanetx.long_horizon import (
        build_m2_frozen_source_path_tile,
        verify_m2_source_path,
    )
    from jxplanetx.restricted_wh_checkpoint import (
        restricted_wh_lifecycle_state_sha256,
    )
    from jxplanetx.restricted_wh_lifecycle import (
        RestrictedWHLifecycleState,
        advance_cuda_restricted_wh_lifecycle,
    )

    lane_contract = {
        "candidate": (
            CANDIDATE_SOURCE_STEPS_PER_MAP,
            CANDIDATE_MAP_STEP_COUNT,
        ),
        "refinement": (
            REFINEMENT_SOURCE_STEPS_PER_MAP,
            REFINEMENT_MAP_STEP_COUNT,
        ),
    }.get(lane_id)
    if lane_contract is None:
        raise M3ProtocolError("unknown JX lane")
    if (source_steps_per_map, map_step_count) != lane_contract:
        raise M3ProtocolError("JX lane step lattice changed")
    if type(initial_state) is not RestrictedWHLifecycleState:
        raise M3ProtocolError("sealed lane factory returned the wrong state type")
    if (
        initial_state.dynamics_state.global_source_step != 0
        or initial_state.dynamics_state.source_step_years.hex()
        != SOURCE_STEP_YEARS.hex()
        or initial_state.dynamics_state.source_schedule_sha256
        != source.source_schedule_sha256
    ):
        raise M3ProtocolError("sealed lane genesis differs from retained M2 lattice")
    before_parent_digest = verify_m2_source_path(source)
    resolved = resolve_backend(backend)
    cp = resolved.xp
    position = np.empty((map_step_count + 1, STATE_COUNT, 3), dtype=np.float64)
    velocity = np.empty_like(position)
    lifecycle = np.empty((map_step_count + 1, STATE_COUNT), dtype=np.uint8)
    minimum_source_distance = np.empty(
        (map_step_count, STATE_COUNT), dtype=np.float64
    )
    endpoint_state_sha256: list[str] = []
    route_records: list[dict[str, Any]] = []
    removal_records: list[dict[str, Any]] = []
    slice_digest = hashlib.sha256(SOURCE_CUSTODY_DOMAIN + lane_id.encode("ascii"))
    binding_digest = hashlib.sha256(
        SOURCE_CUSTODY_DOMAIN + lane_id.encode("ascii") + b".bindings"
    )
    maximum_kepler_iterations = 0
    state = initial_state
    spec = FixedStepCUDARestrictedWHSpec(
        1,
        source_steps_per_map,
        backend,
        maximum_primary_dynamical_phase=MAXIMUM_PRIMARY_DYNAMICAL_PHASE,
        threads_per_block=THREADS_PER_BLOCK,
        primary_outer_boundary_AU=PRIMARY_OUTER_BOUNDARY_AU,
    )
    with resolved.activate():
        position[0] = cp.asnumpy(
            state.dynamics_state.tracer_relative_positions_AU
        )
        velocity[0] = cp.asnumpy(
            state.dynamics_state.tracer_relative_velocities_AU_per_year
        )
        lifecycle[0] = state.tracer_lifecycle
        endpoint_state_sha256.append(
            restricted_wh_lifecycle_state_sha256(state, backend)
        )
        for map_index in range(map_step_count):
            start = map_index * source_steps_per_map
            final = start + source_steps_per_map
            source_slice = source.slice_steps(start, final)
            binding = build_m2_frozen_source_path_tile(source_slice, overlay)
            slice_digest.update(bytes.fromhex(source_slice.dynamic_node_sha256))
            binding_digest.update(bytes.fromhex(binding.content_sha256))
            result = advance_cuda_restricted_wh_lifecycle(
                state,
                binding.tile,
                spec,
                route_spec=route_spec,
                route_worker_count=ROUTE_WORKER_COUNT,
            )
            state = result.state
            if state.dynamics_state.global_source_step != final:
                raise M3NumericalError("JX trace left the retained source lattice")
            position[map_index + 1] = cp.asnumpy(
                state.dynamics_state.tracer_relative_positions_AU
            )
            velocity[map_index + 1] = cp.asnumpy(
                state.dynamics_state.tracer_relative_velocities_AU_per_year
            )
            lifecycle[map_index + 1] = state.tracer_lifecycle
            minimum_source_distance[map_index] = cp.asnumpy(
                result.minimum_source_distance_AU
            )
            endpoint_state_sha256.append(
                restricted_wh_lifecycle_state_sha256(state, backend)
            )
            maximum_kepler_iterations = max(
                maximum_kepler_iterations, result.maximum_kepler_iterations
            )
            route_records.extend(
                _route_record(row, global_map_step=map_index)
                for row in result.route_records
            )
            removal_records.extend(
                _removal_record(row) for row in result.new_removal_records
            )
    after_parent_digest = verify_m2_source_path(source)
    return {
        "position_AU": position,
        "velocity_AU_per_year": velocity,
        "lifecycle": lifecycle,
        "minimum_source_distance_AU": minimum_source_distance,
        "terminal_state": state,
        "endpoint_state_sha256": endpoint_state_sha256,
        "route_records": route_records,
        "removal_records": removal_records,
        "maximum_kepler_iterations": maximum_kepler_iterations,
        "source_parent_custody": {
            "parent_path_content_sha256_before": before_parent_digest,
            "parent_path_content_sha256_after": after_parent_digest,
            "parent_unchanged": before_parent_digest == after_parent_digest,
            "map_slice_count": map_step_count,
            "slice_dynamic_digest": slice_digest.hexdigest(),
            "tile_binding_digest": binding_digest.hexdigest(),
            "first_global_source_step": 0,
            "final_global_source_step": map_step_count * source_steps_per_map,
            "every_slice_verified_by_constructor": True,
        },
    }


def _run_split_resume(
    *,
    lane_id: str,
    initial_state: Any,
    expected_terminal_state_sha256: str,
    expected_split_state_sha256: str,
    expected_route_records: Sequence[Mapping[str, Any]],
    expected_removal_records: Sequence[Mapping[str, Any]],
    backend: Any,
    source: Any,
    overlay: Any,
    source_steps_per_map: int,
    half_map_step_count: int,
    route_spec: Any,
    protocol_sha256: str,
    checkpoint_root: Path,
) -> dict[str, Any]:
    """Build ten annual checkpoints and resume from the verified 5y head."""

    from jxplanetx.cuda_restricted_wh import FixedStepCUDARestrictedWHSpec
    from jxplanetx.long_horizon import (
        build_m2_frozen_source_path_tile,
    )
    from jxplanetx.restricted_wh_checkpoint import (
        dump_restricted_wh_lifecycle_checkpoint,
        load_latest_restricted_wh_lifecycle_checkpoint,
        restricted_wh_lifecycle_state_sha256,
    )
    from jxplanetx.restricted_wh_lifecycle import (
        advance_cuda_restricted_wh_lifecycle,
    )

    lane_contract = {
        "candidate": (
            CANDIDATE_SOURCE_STEPS_PER_MAP,
            SPLIT_CANDIDATE_MAP_STEP,
            512,
        ),
        "refinement": (
            REFINEMENT_SOURCE_STEPS_PER_MAP,
            SPLIT_REFINEMENT_MAP_STEP,
            1024,
        ),
    }.get(lane_id)
    if lane_contract is None:
        raise M3ProtocolError("unknown split/resume JX lane")
    expected_source_steps_per_map, expected_half_maps, expected_annual_maps = (
        lane_contract
    )
    if (
        source_steps_per_map != expected_source_steps_per_map
        or half_map_step_count != expected_half_maps
    ):
        raise M3ProtocolError("split/resume lane step lattice changed")
    split_source_step = half_map_step_count * source_steps_per_map
    if split_source_step != SPLIT_EPOCH_YEARS * SOURCE_STEPS_PER_YEAR:
        raise M3ProtocolError("split map count is not exactly five years")
    annual_map_step_count = half_map_step_count // SPLIT_EPOCH_YEARS
    if annual_map_step_count * SPLIT_EPOCH_YEARS != half_map_step_count:
        raise M3ProtocolError("split map count is not divisible into annual segments")
    if annual_map_step_count != expected_annual_maps:
        raise M3ProtocolError("annual checkpoint map count changed")
    spec = FixedStepCUDARestrictedWHSpec(
        annual_map_step_count,
        source_steps_per_map,
        backend,
        maximum_primary_dynamical_phase=MAXIMUM_PRIMARY_DYNAMICAL_PHASE,
        threads_per_block=THREADS_PER_BLOCK,
        primary_outer_boundary_AU=PRIMARY_OUTER_BOUNDARY_AU,
    )
    genesis_sha = restricted_wh_lifecycle_state_sha256(initial_state, backend)
    run_id = f"{RUN_ID}-{lane_id}-split-v1"
    current_state = initial_state
    previous_checkpoint_sha: str | None = None
    checkpoint_records: list[Mapping[str, Any]] = []
    bindings: list[Any] = []
    segment_results: list[Any] = []
    first_state_sha = ""
    loaded_split = None
    for year in range(HORIZON_YEARS):
        source_slice = source.slice_years(year, year + 1)
        binding = build_m2_frozen_source_path_tile(source_slice, overlay)
        bindings.append(binding)
        segment_input = current_state
        segment_result = advance_cuda_restricted_wh_lifecycle(
            segment_input,
            binding.tile,
            spec,
            route_spec=route_spec,
            route_worker_count=ROUTE_WORKER_COUNT,
        )
        segment_results.append(segment_result)
        record = dump_restricted_wh_lifecycle_checkpoint(
            checkpoint_root,
            run_id=run_id,
            protocol_sha256=protocol_sha256,
            genesis_state_sha256=genesis_sha,
            checkpoint_index=year,
            input_state=segment_input,
            result=segment_result,
            source_tile=binding.tile,
            spec=spec,
            route_spec=route_spec,
            route_worker_count=ROUTE_WORKER_COUNT,
            previous_checkpoint_content_sha256=previous_checkpoint_sha,
        )
        checkpoint_records.append(record)
        previous_checkpoint_sha = record["checkpoint_content_sha256"]
        current_state = segment_result.state
        if year == SPLIT_EPOCH_YEARS - 1:
            first_state_sha = restricted_wh_lifecycle_state_sha256(
                current_state, backend
            )
            loaded_split = load_latest_restricted_wh_lifecycle_checkpoint(
                checkpoint_root,
                run_id=run_id,
                protocol_sha256=protocol_sha256,
                genesis_state_sha256=genesis_sha,
                expected_checkpoint_index=year,
                expected_checkpoint_content_sha256=previous_checkpoint_sha,
                backend=backend,
            )
            if loaded_split.state_sha256 != first_state_sha:
                raise M3NumericalError(
                    "checkpoint replay changed the five-year state"
                )
            # The post-split execution must begin from the replay-released
            # state, not the in-memory pre-publication state.
            current_state = loaded_split.state
    if loaded_split is None or not checkpoint_records:
        raise M3NumericalError("split checkpoint chain was not constructed")
    final_record = checkpoint_records[-1]
    loaded_final = load_latest_restricted_wh_lifecycle_checkpoint(
        checkpoint_root,
        run_id=run_id,
        protocol_sha256=protocol_sha256,
        genesis_state_sha256=genesis_sha,
        expected_checkpoint_index=HORIZON_YEARS - 1,
        expected_checkpoint_content_sha256=final_record[
            "checkpoint_content_sha256"
        ],
        backend=backend,
    )
    terminal_sha = loaded_final.state_sha256
    split_route_records: list[dict[str, Any]] = []
    split_removal_records: list[dict[str, Any]] = []
    for year, segment_result in enumerate(segment_results):
        offset = year * annual_map_step_count
        split_route_records.extend(
            _route_record(row, global_map_step=offset + row.map_step_index)
            for row in segment_result.route_records
        )
        split_removal_records.extend(
            _removal_record(row) for row in segment_result.new_removal_records
        )
    # An annual tile and a one-map tile intentionally have different
    # container hashes. Compare the physical route/event ledger after
    # removing only that container-size identity; each exact source segment
    # hash, lattice step, located event, tracer, and diagnostic remains bound.
    expected_ledger = {
        "route_records": [
            {key: value for key, value in row.items() if key != "source_tile_sha256"}
            for row in expected_route_records
        ],
        "removal_records": list(expected_removal_records),
    }
    split_ledger = {
        "route_records": [
            {key: value for key, value in row.items() if key != "source_tile_sha256"}
            for row in split_route_records
        ],
        "removal_records": split_removal_records,
    }
    expected_ledger_sha = hashlib.sha256(
        EVENT_DOMAIN + _canonical_json(expected_ledger)
    ).hexdigest()
    split_ledger_sha = hashlib.sha256(
        EVENT_DOMAIN + _canonical_json(split_ledger)
    ).hexdigest()
    ledger_exact = split_ledger_sha == expected_ledger_sha
    return {
        "lane_id": lane_id,
        "genesis_state_sha256": genesis_sha,
        "first_half_state_sha256": first_state_sha,
        "expected_split_state_sha256": expected_split_state_sha256,
        "split_checkpoint_content_sha256": checkpoint_records[
            SPLIT_EPOCH_YEARS - 1
        ]["checkpoint_content_sha256"],
        "final_checkpoint_content_sha256": final_record[
            "checkpoint_content_sha256"
        ],
        "loaded_split_checkpoint_state_sha256": loaded_split.state_sha256,
        "loaded_final_checkpoint_state_sha256": loaded_final.state_sha256,
        "terminal_state_sha256": terminal_sha,
        "expected_terminal_state_sha256": expected_terminal_state_sha256,
        "first_half_matches_one_map_trace": (
            first_state_sha == expected_split_state_sha256
        ),
        "bitwise_exact": bool(
            terminal_sha == expected_terminal_state_sha256
            and first_state_sha == expected_split_state_sha256
            and ledger_exact
        ),
        "lifecycle_state_bitwise_exact": (
            terminal_sha == expected_terminal_state_sha256
        ),
        "event_ledger_bitwise_exact": ledger_exact,
        "expected_event_ledger_sha256": expected_ledger_sha,
        "split_event_ledger_sha256": split_ledger_sha,
        "annual_checkpoint_count": len(checkpoint_records),
        "annual_source_slice_dynamic_node_sha256": [
            binding.dynamic_node_sha256 for binding in bindings
        ],
        "annual_source_tile_binding_sha256": [
            binding.content_sha256 for binding in bindings
        ],
    }


_ROUTED_EVENT_KEYS = frozenset(
    {
        "event_type",
        "source_index",
        "source_id",
        "map_step_fraction_hex",
        "source_interval_offset",
        "interval_fraction_hex",
        "relative_position_AU_hex",
        "relative_velocity_AU_per_year_hex",
        "surface_distance_AU_hex",
        "surface_threshold_AU_hex",
        "surface_residual_AU_hex",
        "surface_tolerance_AU_hex",
        "primary_specific_energy_AU2_per_year2_hex",
        "primary_radial_velocity_AU_per_year_hex",
        "surface_radial_velocity_AU_per_year_hex",
        "location_method",
    }
)
_ROUTE_LEDGER_KEYS = frozenset(
    {
        "record_kind",
        "global_map_step",
        "lane_index",
        "tracer_index",
        "guard_phase",
        "guard_reason",
        "guard_source_index",
        "source_tile_sha256",
        "source_segment_sha256",
        "global_source_step",
        "source_steps_per_map_step",
        "accepted_step_count",
        "derivative_evaluation_count",
        "endpoint_relative_position_AU_hex",
        "endpoint_relative_velocity_AU_per_year_hex",
        "minimum_evaluated_stage_distance_AU_hex",
        "minimum_committed_stage_distance_AU_hex",
        "maximum_accepted_primary_dynamical_phase_radians_hex",
        "maximum_committed_primary_dynamical_phase_radians_hex",
        "completed_map_fraction_hex",
        "event_count",
        "event_content",
        "runtime_identity_sha256",
    }
)
_REMOVAL_LEDGER_KEYS = frozenset(
    {
        "record_kind",
        "content_sha256",
        "lane_index",
        "tracer_index",
        "lifecycle_code",
        "global_source_step",
        "source_steps_per_map_step",
        "guard_phase",
        "guard_reason",
        "guard_source_index",
        "source_segment_sha256",
        "route_runtime_identity_sha256",
        "accepted_step_count",
        "derivative_evaluation_count",
        "minimum_committed_stage_distance_AU_hex",
        "maximum_committed_primary_dynamical_phase_radians_hex",
        "event_count",
        "event_content",
    }
)
_TERMINAL_EVENT_LIFECYCLE = {
    "SOURCE_RADIUS_CONTACT": 1,
    "PRIMARY_BOUNDARY_OUTWARD": 2,
}
_TERMINAL_EVENT_PRIORITY = {
    "SOURCE_RADIUS_CONTACT": 0,
    "PRIMARY_BOUNDARY_OUTWARD": 1,
}


def _event_ledger_error(message: str) -> M3NumericalError:
    return M3NumericalError(f"event ledger is invalid: {message}")


def _ledger_integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise _event_ledger_error(f"{label} is not an integer >= {minimum}")
    return value


def _ledger_digest(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value == "0" * 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _event_ledger_error(f"{label} is not a nonzero lowercase SHA-256")
    return value


def _ledger_hex_float(value: object, label: str) -> float:
    if type(value) is not str:
        raise _event_ledger_error(f"{label} is not a hexadecimal binary64")
    try:
        parsed = float.fromhex(value)
    except (TypeError, ValueError) as exc:
        raise _event_ledger_error(
            f"{label} is not a hexadecimal binary64"
        ) from exc
    if not math.isfinite(parsed) or parsed.hex() != value:
        raise _event_ledger_error(
            f"{label} is not a canonical finite hexadecimal binary64"
        )
    return parsed


def _ledger_hex_vector(value: object, label: str) -> tuple[float, float, float]:
    if type(value) is not list or len(value) != 3:
        raise _event_ledger_error(f"{label} is not a three-vector")
    parsed = tuple(
        _ledger_hex_float(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    return parsed  # type: ignore[return-value]


def _validate_routed_events(
    value: object,
    *,
    source_steps_per_map_step: int,
    label: str,
) -> list[dict[str, Any]]:
    from jxplanetx.cuda_tracer_events import (
        LOCATION_BRACKETED_ROOT,
        LOCATION_DENSE_MINIMUM,
        LOCATION_EXACT_DENSE_NODE,
        LOCATION_INITIAL_SURFACE,
    )
    from jxplanetx.long_horizon.m2_source_path import SOURCE_IDS

    location_methods = {
        LOCATION_BRACKETED_ROOT,
        LOCATION_DENSE_MINIMUM,
        LOCATION_EXACT_DENSE_NODE,
        LOCATION_INITIAL_SURFACE,
    }
    if len(SOURCE_IDS) != SOURCE_COUNT:
        raise _event_ledger_error("the canonical M2 source roster changed")
    if type(value) is not list:
        raise _event_ledger_error(f"{label} is not an exact event list")
    parsed: list[dict[str, Any]] = []
    order: list[tuple[float, int, int]] = []
    for event_index, event in enumerate(value):
        event_label = f"{label}[{event_index}]"
        if type(event) is not dict or set(event) != _ROUTED_EVENT_KEYS:
            raise _event_ledger_error(f"{event_label} has the wrong schema")
        event_type = event["event_type"]
        if event_type not in _TERMINAL_EVENT_LIFECYCLE:
            raise _event_ledger_error(f"{event_label}.event_type is not terminal")
        source_index = _ledger_integer(
            event["source_index"], f"{event_label}.source_index"
        )
        if source_index >= SOURCE_COUNT:
            raise _event_ledger_error(
                f"{event_label}.source_index is outside the source roster"
            )
        if event_type == "PRIMARY_BOUNDARY_OUTWARD" and source_index != 0:
            raise _event_ledger_error(
                f"{event_label} assigns the primary boundary to a nonprimary source"
            )
        if event["source_id"] != SOURCE_IDS[source_index]:
            raise _event_ledger_error(
                f"{event_label}.source_id disagrees with the canonical M2 roster"
            )
        source_interval_offset = _ledger_integer(
            event["source_interval_offset"],
            f"{event_label}.source_interval_offset",
        )
        if source_interval_offset >= source_steps_per_map_step:
            raise _event_ledger_error(
                f"{event_label}.source_interval_offset is outside the map step"
            )
        interval_fraction = _ledger_hex_float(
            event["interval_fraction_hex"],
            f"{event_label}.interval_fraction_hex",
        )
        map_fraction = _ledger_hex_float(
            event["map_step_fraction_hex"],
            f"{event_label}.map_step_fraction_hex",
        )
        if not 0.0 <= interval_fraction <= 1.0 or not 0.0 <= map_fraction <= 1.0:
            raise _event_ledger_error(f"{event_label} fraction is outside [0, 1]")
        expected_map_fraction = (
            source_interval_offset + interval_fraction
        ) / source_steps_per_map_step
        if expected_map_fraction.hex() != event["map_step_fraction_hex"]:
            raise _event_ledger_error(
                f"{event_label} map and source-interval fractions disagree"
            )
        position = _ledger_hex_vector(
            event["relative_position_AU_hex"],
            f"{event_label}.relative_position_AU_hex",
        )
        velocity = _ledger_hex_vector(
            event["relative_velocity_AU_per_year_hex"],
            f"{event_label}.relative_velocity_AU_per_year_hex",
        )
        surface_distance = _ledger_hex_float(
            event["surface_distance_AU_hex"],
            f"{event_label}.surface_distance_AU_hex",
        )
        surface_threshold = _ledger_hex_float(
            event["surface_threshold_AU_hex"],
            f"{event_label}.surface_threshold_AU_hex",
        )
        surface_residual = _ledger_hex_float(
            event["surface_residual_AU_hex"],
            f"{event_label}.surface_residual_AU_hex",
        )
        surface_tolerance = _ledger_hex_float(
            event["surface_tolerance_AU_hex"],
            f"{event_label}.surface_tolerance_AU_hex",
        )
        _ledger_hex_float(
            event["primary_specific_energy_AU2_per_year2_hex"],
            f"{event_label}.primary_specific_energy_AU2_per_year2_hex",
        )
        primary_radial_velocity = _ledger_hex_float(
            event["primary_radial_velocity_AU_per_year_hex"],
            f"{event_label}.primary_radial_velocity_AU_per_year_hex",
        )
        surface_radial_velocity = _ledger_hex_float(
            event["surface_radial_velocity_AU_per_year_hex"],
            f"{event_label}.surface_radial_velocity_AU_per_year_hex",
        )
        if event_type == "SOURCE_RADIUS_CONTACT":
            # M3 deliberately has no physical contact/removal radius.  The
            # 0.1-AU spheres are routing guards only and must never be
            # serialized as physical terminal contacts.
            if PHYSICAL_TERMINAL_RADIUS_AU.hex() != (0.0).hex():
                raise _event_ledger_error("M3 terminal-radius policy changed")
            raise _event_ledger_error(
                f"{event_label} reports a disabled physical contact event"
            )
        if (
            surface_threshold.hex() != PRIMARY_OUTER_BOUNDARY_AU.hex()
            or primary_radial_velocity
            <= OUTWARD_RADIAL_VELOCITY_TOLERANCE_AU_PER_YEAR
            or surface_radial_velocity.hex() != primary_radial_velocity.hex()
        ):
            raise _event_ledger_error(
                f"{event_label} is not the exact outward primary boundary"
            )
        if (
            surface_distance < 0.0
            or surface_threshold <= 0.0
            or surface_tolerance <= 0.0
        ):
            raise _event_ledger_error(
                f"{event_label} contains an invalid surface magnitude"
            )
        if (
            abs(surface_residual) > surface_tolerance
            or (surface_distance - surface_threshold).hex()
            != event["surface_residual_AU_hex"]
        ):
            raise _event_ledger_error(
                f"{event_label} surface residual is not a valid located root"
            )
        if event["location_method"] not in location_methods:
            raise _event_ledger_error(
                f"{event_label}.location_method is not recognized"
            )
        order.append(
            (
                map_fraction,
                _TERMINAL_EVENT_PRIORITY[event_type],
                source_index,
            )
        )
        parsed.append(
            {
                "event_type": event_type,
                "map_fraction": map_fraction,
                "position": position,
                "velocity": velocity,
            }
        )
    if order != sorted(order) or len(set(order)) != len(order):
        raise _event_ledger_error(f"{label} is not canonically ordered and unique")
    if parsed and any(
        item["position"] != parsed[0]["position"]
        or item["velocity"] != parsed[0]["velocity"]
        for item in parsed[1:]
    ):
        raise _event_ledger_error(
            f"{label} simultaneous events do not share one terminal state"
        )
    return parsed


def _removal_ledger_sha256(row: Mapping[str, Any]) -> str:
    events = []
    for event in row["event_content"]:
        events.append(
            {
                "event_type": event["event_type"],
                "source_index": event["source_index"],
                "source_id": event["source_id"],
                "map_step_fraction": event["map_step_fraction_hex"],
                "source_interval_offset": event["source_interval_offset"],
                "interval_fraction": event["interval_fraction_hex"],
                "relative_position_AU": event["relative_position_AU_hex"],
                "relative_velocity_AU_per_year": event[
                    "relative_velocity_AU_per_year_hex"
                ],
                "surface_distance_AU": event["surface_distance_AU_hex"],
                "surface_threshold_AU": event["surface_threshold_AU_hex"],
                "surface_residual_AU": event["surface_residual_AU_hex"],
                "surface_tolerance_AU": event["surface_tolerance_AU_hex"],
                "primary_specific_energy_AU2_per_year2": event[
                    "primary_specific_energy_AU2_per_year2_hex"
                ],
                "primary_radial_velocity_AU_per_year": event[
                    "primary_radial_velocity_AU_per_year_hex"
                ],
                "surface_radial_velocity_AU_per_year": event[
                    "surface_radial_velocity_AU_per_year_hex"
                ],
                "location_method": event["location_method"],
            }
        )
    payload = {
        "schema": "jx.restricted-wh.removal-record.v1",
        "lane_index": row["lane_index"],
        "tracer_index": row["tracer_index"],
        "lifecycle_code": row["lifecycle_code"],
        "global_source_step": row["global_source_step"],
        "source_steps_per_map_step": row["source_steps_per_map_step"],
        "guard_phase": row["guard_phase"],
        "guard_reason": row["guard_reason"],
        "guard_source_index": row["guard_source_index"],
        "source_segment_sha256": row["source_segment_sha256"],
        "route_runtime_identity_sha256": row[
            "route_runtime_identity_sha256"
        ],
        "accepted_step_count": row["accepted_step_count"],
        "derivative_evaluation_count": row[
            "derivative_evaluation_count"
        ],
        "minimum_committed_stage_distance_AU": row[
            "minimum_committed_stage_distance_AU_hex"
        ],
        "maximum_committed_primary_dynamical_phase_radians": row[
            "maximum_committed_primary_dynamical_phase_radians_hex"
        ],
        "events": events,
    }
    return hashlib.sha256(
        b"jx.restricted-wh.removal-record.v1\0" + _canonical_json(payload)
    ).hexdigest()


def _event_accounting(trace: Mapping[str, Any]) -> dict[str, Any]:
    lifecycle = trace["lifecycle"]
    position = trace["position_AU"]
    velocity = trace["velocity_AU_per_year"]
    routes = trace["route_records"]
    removals = trace["removal_records"]
    if (
        type(lifecycle) is not np.ndarray
        or lifecycle.dtype != np.dtype(np.uint8)
        or lifecycle.ndim != 2
        or lifecycle.shape[0] < 1
        or lifecycle.shape[1] != STATE_COUNT
    ):
        raise M3NumericalError("lifecycle ledger has an invalid shape or dtype")
    expected_state_shape = (*lifecycle.shape, 3)
    if (
        type(position) is not np.ndarray
        or type(velocity) is not np.ndarray
        or position.dtype != np.dtype(np.float64)
        or velocity.dtype != np.dtype(np.float64)
        or position.shape != expected_state_shape
        or velocity.shape != expected_state_shape
        or not bool(np.all(np.isfinite(position)))
        or not bool(np.all(np.isfinite(velocity)))
    ):
        raise _event_ledger_error(
            "retained trace positions/velocities have an invalid shape, dtype, or value"
        )
    if type(routes) is not list or type(removals) is not list:
        raise _event_ledger_error("route and removal ledgers must be exact lists")
    if np.any(lifecycle[0] != 0) or np.any(lifecycle > 2):
        raise _event_ledger_error("lifecycle codes or genesis are invalid")
    prior = lifecycle[:-1]
    following = lifecycle[1:]
    no_reactivation = bool(np.all((prior == 0) | (following == prior)))
    if not no_reactivation:
        raise _event_ledger_error("a terminal lifecycle lane changed or reactivated")
    transition_rows = np.argwhere((prior == 0) & (following != 0))
    transitions = {
        int(lane): (int(map_step), int(lifecycle[map_step + 1, lane]))
        for map_step, lane in transition_rows
    }
    if len(transitions) != len(transition_rows):
        raise _event_ledger_error("one lane has more than one terminal transition")
    map_step_count = lifecycle.shape[0] - 1
    lane_to_tracer: dict[int, int] = {}
    tracer_to_lane: dict[int, int] = {}

    def bind_identity(lane: int, tracer: int, label: str) -> None:
        if lane in lane_to_tracer and lane_to_tracer[lane] != tracer:
            raise _event_ledger_error(f"{label} changes the tracer bound to a lane")
        if tracer in tracer_to_lane and tracer_to_lane[tracer] != lane:
            raise _event_ledger_error(f"{label} binds one tracer to two lanes")
        lane_to_tracer[lane] = tracer
        tracer_to_lane[tracer] = lane

    route_order: list[tuple[int, int, int]] = []
    route_map_lanes: set[tuple[int, int]] = set()
    for index, row in enumerate(routes):
        label = f"route_records[{index}]"
        if type(row) is not dict or set(row) != _ROUTE_LEDGER_KEYS:
            raise _event_ledger_error(f"{label} has the wrong schema")
        if row["record_kind"] != "ROUTED_REJOIN":
            raise _event_ledger_error(f"{label} has the wrong record kind")
        map_step = _ledger_integer(row["global_map_step"], f"{label}.global_map_step")
        lane = _ledger_integer(row["lane_index"], f"{label}.lane_index")
        tracer = _ledger_integer(row["tracer_index"], f"{label}.tracer_index")
        if (
            map_step >= map_step_count
            or lane >= STATE_COUNT
            or tracer >= STATE_COUNT
            or tracer != lane
        ):
            raise _event_ledger_error(f"{label} is outside the map/lane lattice")
        ratio = _ledger_integer(
            row["source_steps_per_map_step"],
            f"{label}.source_steps_per_map_step",
            minimum=1,
        )
        if ratio not in {
            CANDIDATE_SOURCE_STEPS_PER_MAP,
            REFINEMENT_SOURCE_STEPS_PER_MAP,
        }:
            raise _event_ledger_error(f"{label} has an unknown source/map ratio")
        global_source_step = _ledger_integer(
            row["global_source_step"], f"{label}.global_source_step"
        )
        if global_source_step != map_step * ratio:
            raise _event_ledger_error(f"{label} source and map steps disagree")
        for field in (
            "source_tile_sha256",
            "source_segment_sha256",
            "runtime_identity_sha256",
        ):
            _ledger_digest(row[field], f"{label}.{field}")
        for field in ("accepted_step_count", "derivative_evaluation_count"):
            _ledger_integer(row[field], f"{label}.{field}")
        if type(row["guard_phase"]) is not str or not row["guard_phase"]:
            raise _event_ledger_error(f"{label}.guard_phase is empty")
        if type(row["guard_reason"]) is not str or not row["guard_reason"]:
            raise _event_ledger_error(f"{label}.guard_reason is empty")
        guard_source = row["guard_source_index"]
        if guard_source is not None and (
            type(guard_source) is not int
            or guard_source < 0
            or guard_source >= SOURCE_COUNT
        ):
            raise _event_ledger_error(f"{label}.guard_source_index is invalid")
        endpoint_position = _ledger_hex_vector(
            row["endpoint_relative_position_AU_hex"],
            f"{label}.endpoint_relative_position_AU_hex",
        )
        endpoint_velocity = _ledger_hex_vector(
            row["endpoint_relative_velocity_AU_per_year_hex"],
            f"{label}.endpoint_relative_velocity_AU_per_year_hex",
        )
        if (
            tuple(
                float(value).hex()
                for value in position[map_step + 1, lane]
            )
            != tuple(value.hex() for value in endpoint_position)
            or tuple(
                float(value).hex()
                for value in velocity[map_step + 1, lane]
            )
            != tuple(value.hex() for value in endpoint_velocity)
        ):
            raise _event_ledger_error(
                f"{label} endpoint differs from the retained trace node"
            )
        minimum_evaluated = _ledger_hex_float(
            row["minimum_evaluated_stage_distance_AU_hex"],
            f"{label}.minimum_evaluated_stage_distance_AU_hex",
        )
        minimum_committed = _ledger_hex_float(
            row["minimum_committed_stage_distance_AU_hex"],
            f"{label}.minimum_committed_stage_distance_AU_hex",
        )
        maximum_accepted = _ledger_hex_float(
            row["maximum_accepted_primary_dynamical_phase_radians_hex"],
            f"{label}.maximum_accepted_primary_dynamical_phase_radians_hex",
        )
        maximum_committed = _ledger_hex_float(
            row["maximum_committed_primary_dynamical_phase_radians_hex"],
            f"{label}.maximum_committed_primary_dynamical_phase_radians_hex",
        )
        completed = _ledger_hex_float(
            row["completed_map_fraction_hex"],
            f"{label}.completed_map_fraction_hex",
        )
        if (
            minimum_evaluated <= 0.0
            or minimum_committed <= 0.0
            or minimum_evaluated
            > minimum_committed
            + 64.0
            * np.finfo(np.float64).eps
            * max(1.0, minimum_evaluated, minimum_committed)
            or maximum_accepted < 0.0
            or maximum_committed < 0.0
            or maximum_committed
            > maximum_accepted
            + 64.0
            * np.finfo(np.float64).eps
            * max(1.0, maximum_accepted, maximum_committed)
            or completed != 1.0
        ):
            raise _event_ledger_error(f"{label} route diagnostics are inconsistent")
        event_count = _ledger_integer(row["event_count"], f"{label}.event_count")
        events = _validate_routed_events(
            row["event_content"],
            source_steps_per_map_step=ratio,
            label=f"{label}.event_content",
        )
        if event_count != len(events):
            raise _event_ledger_error(f"{label}.event_count disagrees with its events")
        if events:
            raise _event_ledger_error(
                f"{label} is a committed rejoin but contains terminal route events"
            )
        order_key = (map_step, lane, tracer)
        if (map_step, lane) in route_map_lanes:
            raise _event_ledger_error("route map/lane pairs are not unique")
        route_map_lanes.add((map_step, lane))
        route_order.append(order_key)
        if lifecycle[map_step, lane] != 0 or lifecycle[map_step + 1, lane] != 0:
            raise _event_ledger_error(
                f"{label} is not active across its committed rejoin"
            )
        bind_identity(lane, tracer, label)
    if route_order != sorted(route_order):
        raise _event_ledger_error("route records are not canonically ordered")

    removal_order: list[tuple[int, float, int, int]] = []
    removal_lanes: set[int] = set()
    removal_hashes: set[str] = set()
    witnessed_transitions: dict[int, tuple[int, int]] = {}
    for index, row in enumerate(removals):
        label = f"removal_records[{index}]"
        if type(row) is not dict or set(row) != _REMOVAL_LEDGER_KEYS:
            raise _event_ledger_error(f"{label} has the wrong schema")
        if row["record_kind"] != "TERMINAL_REMOVAL":
            raise _event_ledger_error(f"{label} has the wrong record kind")
        lane = _ledger_integer(row["lane_index"], f"{label}.lane_index")
        tracer = _ledger_integer(row["tracer_index"], f"{label}.tracer_index")
        lifecycle_code = _ledger_integer(
            row["lifecycle_code"], f"{label}.lifecycle_code", minimum=1
        )
        if (
            lane >= STATE_COUNT
            or tracer >= STATE_COUNT
            or tracer != lane
            or lifecycle_code not in {1, 2}
        ):
            raise _event_ledger_error(f"{label} lane or lifecycle code is invalid")
        ratio = _ledger_integer(
            row["source_steps_per_map_step"],
            f"{label}.source_steps_per_map_step",
            minimum=1,
        )
        if ratio not in {
            CANDIDATE_SOURCE_STEPS_PER_MAP,
            REFINEMENT_SOURCE_STEPS_PER_MAP,
        }:
            raise _event_ledger_error(f"{label} has an unknown source/map ratio")
        global_source_step = _ledger_integer(
            row["global_source_step"], f"{label}.global_source_step"
        )
        if global_source_step % ratio:
            raise _event_ledger_error(f"{label} does not start on a map boundary")
        map_step = global_source_step // ratio
        if map_step >= map_step_count:
            raise _event_ledger_error(f"{label} is outside the map lattice")
        content_sha256 = _ledger_digest(
            row["content_sha256"], f"{label}.content_sha256"
        )
        for field in ("source_segment_sha256", "route_runtime_identity_sha256"):
            _ledger_digest(row[field], f"{label}.{field}")
        for field in ("accepted_step_count", "derivative_evaluation_count"):
            _ledger_integer(row[field], f"{label}.{field}")
        if type(row["guard_phase"]) is not str or not row["guard_phase"]:
            raise _event_ledger_error(f"{label}.guard_phase is empty")
        if type(row["guard_reason"]) is not str or not row["guard_reason"]:
            raise _event_ledger_error(f"{label}.guard_reason is empty")
        guard_source = row["guard_source_index"]
        if guard_source is not None and (
            type(guard_source) is not int
            or guard_source < 0
            or guard_source >= SOURCE_COUNT
        ):
            raise _event_ledger_error(f"{label}.guard_source_index is invalid")
        minimum_committed = _ledger_hex_float(
            row["minimum_committed_stage_distance_AU_hex"],
            f"{label}.minimum_committed_stage_distance_AU_hex",
        )
        maximum_committed = _ledger_hex_float(
            row["maximum_committed_primary_dynamical_phase_radians_hex"],
            f"{label}.maximum_committed_primary_dynamical_phase_radians_hex",
        )
        if minimum_committed <= 0.0 or maximum_committed < 0.0:
            raise _event_ledger_error(f"{label} route diagnostics are inconsistent")
        event_count = _ledger_integer(row["event_count"], f"{label}.event_count")
        events = _validate_routed_events(
            row["event_content"],
            source_steps_per_map_step=ratio,
            label=f"{label}.event_content",
        )
        if event_count != len(events) or not events:
            raise _event_ledger_error(
                f"{label}.event_count does not identify terminal route events"
            )
        expected_lifecycle_code = _TERMINAL_EVENT_LIFECYCLE[
            events[0]["event_type"]
        ]
        if lifecycle_code != expected_lifecycle_code:
            raise _event_ledger_error(
                f"{label} first terminal event and lifecycle code disagree"
            )
        transition = transitions.get(lane)
        if transition != (map_step, lifecycle_code):
            raise _event_ledger_error(
                f"{label} does not match its lane's lifecycle transition"
            )
        terminal_event = events[0]
        if (
            tuple(
                float(value).hex()
                for value in position[map_step + 1, lane]
            )
            != tuple(value.hex() for value in terminal_event["position"])
            or tuple(
                float(value).hex()
                for value in velocity[map_step + 1, lane]
            )
            != tuple(value.hex() for value in terminal_event["velocity"])
        ):
            raise _event_ledger_error(
                f"{label} terminal event differs from the retained frozen trace"
            )
        if lane in removal_lanes or content_sha256 in removal_hashes:
            raise _event_ledger_error("removal lanes and content hashes must be unique")
        if (map_step, lane) in route_map_lanes:
            raise _event_ledger_error(
                "one map/lane is both a committed rejoin and terminal removal"
            )
        if _removal_ledger_sha256(row) != content_sha256:
            raise _event_ledger_error(f"{label}.content_sha256 does not bind the row")
        removal_lanes.add(lane)
        removal_hashes.add(content_sha256)
        witnessed_transitions[lane] = (map_step, lifecycle_code)
        removal_order.append(
            (global_source_step, events[0]["map_fraction"], lane, tracer)
        )
        bind_identity(lane, tracer, label)
    if removal_order != sorted(removal_order):
        raise _event_ledger_error("removal records are not canonically ordered")
    if witnessed_transitions != transitions:
        raise _event_ledger_error(
            "terminal route events and lifecycle transitions are not bijective"
        )
    final_terminal_lanes = set(np.flatnonzero(lifecycle[-1] != 0).tolist())
    if removal_lanes != final_terminal_lanes:
        raise _event_ledger_error(
            "terminal route events and final lifecycle lanes disagree"
        )
    transition_count = len(transitions)
    final_terminal_count = len(final_terminal_lanes)
    ledger = {
        "route_records": routes,
        "removal_records": removals,
    }
    return {
        "exact": True,
        "route_record_count": len(routes),
        "removal_record_count": len(removals),
        "terminal_lane_count": final_terminal_count,
        "lifecycle_transition_count": transition_count,
        "no_reactivation": True,
        "ledger_content_sha256": hashlib.sha256(
            EVENT_DOMAIN + _canonical_json(ledger)
        ).hexdigest(),
    }


def execute_numerics(
    *,
    project_root: Path,
    prepared_profiles: Mapping[str, Any],
    source: Any,
    protocol: Mapping[str, Any],
    work_directory: Path,
) -> dict[str, Any]:
    """Execute all bounded lanes sequentially; callers control publication."""

    from jxplanetx.long_horizon import (
        KnownSystemRunProfile,
        build_known_system_observed_fixed_run_inputs,
        M2_DERIVED_ACCELERATION_BOUND_AU_PER_YEAR2,
        seal_m2_guard_overlay,
        verify_m2_source_path,
    )

    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise M3ProtocolError("numerical execution received a different protocol")
    # Fail before any expensive GPU work if the independent comparator is not
    # exactly the reviewed runtime/API.
    _load_exact_rebound()
    runtime_identity_start = _runtime_identity(
        include_numerical_packages=True,
        project_root=project_root,
        expected_code_identity=protocol.get("code_identity"),
    )
    parent_before = verify_m2_source_path(source)
    gm = np.ascontiguousarray(source.source_gm_AU3_per_year2)
    radii = np.full(SOURCE_COUNT, ALGORITHMIC_GUARD_RADIUS_AU, dtype=np.float64)
    bounds = np.full(
        SOURCE_COUNT,
        M2_DERIVED_ACCELERATION_BOUND_AU_PER_YEAR2,
        dtype=np.float64,
    )
    overlay = seal_m2_guard_overlay(
        source,
        source_encounter_radii_AU=radii,
        certified_relative_acceleration_bounds_AU_per_year2=bounds,
    )

    def verify_materialized_genesis(
        materialized: Any, prepared: Any, lane_id: str
    ) -> None:
        """Recheck the actual CUDA arrays before allowing a numerical lane."""

        from jxplanetx.engine.backends import resolve_backend

        if materialized.contract.content_sha256 != prepared.content_sha256:
            raise M3ProtocolError(f"{lane_id} materialization changed its contract")
        lifecycle = materialized.lifecycle_state
        state = lifecycle.dynamics_state
        resolved = resolve_backend(materialized.contract.config.tracer_spec.backend)
        with resolved.activate():
            host_position = np.ascontiguousarray(
                resolved.xp.asnumpy(state.tracer_relative_positions_AU),
                dtype=np.float64,
            )
            host_velocity = np.ascontiguousarray(
                resolved.xp.asnumpy(
                    state.tracer_relative_velocities_AU_per_year
                ),
                dtype=np.float64,
            )
            host_indices = np.ascontiguousarray(
                resolved.xp.asnumpy(state.tracer_indices), dtype=np.uint64
            )
            resolved.xp.cuda.get_current_stream().synchronize()
        if (
            not _bitwise_equal(host_position, prepared.tracer_positions_AU)
            or not _bitwise_equal(host_velocity, prepared.tracer_velocities_AU_per_year)
            or not _bitwise_equal(
                host_indices, np.arange(STATE_COUNT, dtype=np.uint64)
            )
            or not _bitwise_equal(
                lifecycle.tracer_lifecycle,
                np.zeros(STATE_COUNT, dtype=np.uint8),
            )
            or lifecycle.removal_records
            or state.global_source_step != 0
        ):
            raise M3ProtocolError(
                f"{lane_id} materialized CUDA genesis differs bitwise from protocol"
            )

    timings: dict[str, float] = {}
    started = time.perf_counter()
    candidate_inputs = build_known_system_observed_fixed_run_inputs(
        profile_id=KnownSystemRunProfile.M2_FINE_CANDIDATE_10Y_ANNUAL
    )
    verify_materialized_genesis(
        candidate_inputs, prepared_profiles["candidate"], "candidate"
    )
    candidate = _run_jx_trace(
        lane_id="candidate",
        initial_state=candidate_inputs.lifecycle_state,
        backend=candidate_inputs.contract.config.tracer_spec.backend,
        source=source,
        overlay=overlay,
        source_steps_per_map=CANDIDATE_SOURCE_STEPS_PER_MAP,
        map_step_count=CANDIDATE_MAP_STEP_COUNT,
        route_spec=candidate_inputs.contract.config.route_spec,
    )
    timings["jx_candidate_trace_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    candidate_split = _run_split_resume(
        lane_id="candidate",
        initial_state=candidate_inputs.lifecycle_state,
        expected_terminal_state_sha256=candidate["endpoint_state_sha256"][-1],
        expected_split_state_sha256=candidate["endpoint_state_sha256"][
            SPLIT_CANDIDATE_MAP_STEP
        ],
        expected_route_records=candidate["route_records"],
        expected_removal_records=candidate["removal_records"],
        backend=candidate_inputs.contract.config.tracer_spec.backend,
        source=source,
        overlay=overlay,
        source_steps_per_map=CANDIDATE_SOURCE_STEPS_PER_MAP,
        half_map_step_count=SPLIT_CANDIDATE_MAP_STEP,
        route_spec=candidate_inputs.contract.config.route_spec,
        protocol_sha256=protocol["content_sha256"],
        checkpoint_root=work_directory / "candidate_split_checkpoint",
    )
    timings["jx_candidate_split_resume_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    refinement_inputs = build_known_system_observed_fixed_run_inputs(
        profile_id=KnownSystemRunProfile.M2_FINE_REFINEMENT_10Y_ANNUAL
    )
    verify_materialized_genesis(
        refinement_inputs, prepared_profiles["refinement"], "refinement"
    )
    refinement = _run_jx_trace(
        lane_id="refinement",
        initial_state=refinement_inputs.lifecycle_state,
        backend=refinement_inputs.contract.config.tracer_spec.backend,
        source=source,
        overlay=overlay,
        source_steps_per_map=REFINEMENT_SOURCE_STEPS_PER_MAP,
        map_step_count=REFINEMENT_MAP_STEP_COUNT,
        route_spec=refinement_inputs.contract.config.route_spec,
    )
    timings["jx_refinement_trace_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    refinement_split = _run_split_resume(
        lane_id="refinement",
        initial_state=refinement_inputs.lifecycle_state,
        expected_terminal_state_sha256=refinement["endpoint_state_sha256"][-1],
        expected_split_state_sha256=refinement["endpoint_state_sha256"][
            SPLIT_REFINEMENT_MAP_STEP
        ],
        expected_route_records=refinement["route_records"],
        expected_removal_records=refinement["removal_records"],
        backend=refinement_inputs.contract.config.tracer_spec.backend,
        source=source,
        overlay=overlay,
        source_steps_per_map=REFINEMENT_SOURCE_STEPS_PER_MAP,
        half_map_step_count=SPLIT_REFINEMENT_MAP_STEP,
        route_spec=refinement_inputs.contract.config.route_spec,
        protocol_sha256=protocol["content_sha256"],
        checkpoint_root=work_directory / "refinement_split_checkpoint",
    )
    timings["jx_refinement_split_resume_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    rebound_arrays = _run_rebound_ias15(
        initial_source_position=source.source_relative_positions_AU[0],
        initial_source_velocity=source.source_relative_velocities_AU_per_year[0],
        source_gm=gm,
        initial_tracer_position=prepared_profiles[
            "candidate"
        ].tracer_positions_AU,
        initial_tracer_velocity=prepared_profiles[
            "candidate"
        ].tracer_velocities_AU_per_year,
    )
    timings["rebound_ias15_combined_and_source_only_seconds"] = (
        time.perf_counter() - started
    )
    parent_after = verify_m2_source_path(source)
    candidate_events = _event_accounting(candidate)
    refinement_events = _event_accounting(refinement)
    runtime_identity_end = _runtime_identity(
        include_numerical_packages=True,
        project_root=project_root,
        expected_code_identity=protocol.get("code_identity"),
    )
    if runtime_identity_end != runtime_identity_start:
        raise M3NumericalError(
            "code, package, native, or GPU runtime identity changed during execution"
        )
    arrays = {
        "candidate_position_AU": candidate["position_AU"],
        "candidate_velocity_AU_per_year": candidate[
            "velocity_AU_per_year"
        ],
        "candidate_lifecycle": candidate["lifecycle"],
        "candidate_minimum_source_distance_AU": candidate[
            "minimum_source_distance_AU"
        ],
        "refinement_position_AU": refinement["position_AU"],
        "refinement_velocity_AU_per_year": refinement[
            "velocity_AU_per_year"
        ],
        "refinement_lifecycle": refinement["lifecycle"],
        "refinement_minimum_source_distance_AU": refinement[
            "minimum_source_distance_AU"
        ],
        "candidate_source_position_AU": np.ascontiguousarray(
            source.source_relative_positions_AU[
                ::CANDIDATE_SOURCE_STEPS_PER_MAP
            ]
        ),
        "candidate_source_velocity_AU_per_year": np.ascontiguousarray(
            source.source_relative_velocities_AU_per_year[
                ::CANDIDATE_SOURCE_STEPS_PER_MAP
            ]
        ),
        "refinement_source_position_AU": np.ascontiguousarray(
            source.source_relative_positions_AU[
                ::REFINEMENT_SOURCE_STEPS_PER_MAP
            ]
        ),
        "refinement_source_velocity_AU_per_year": np.ascontiguousarray(
            source.source_relative_velocities_AU_per_year[
                ::REFINEMENT_SOURCE_STEPS_PER_MAP
            ]
        ),
        "source_gm_AU3_per_year2": gm,
        **rebound_arrays,
    }
    state_ids = protocol["inputs"]["state_ids"]
    candidate_trace_node_custody = _trace_node_custody(
        lane_id="candidate",
        position_AU=arrays["candidate_position_AU"],
        velocity_AU_per_year=arrays["candidate_velocity_AU_per_year"],
        lifecycle=arrays["candidate_lifecycle"],
        state_ids=state_ids,
        source_steps_per_map_step=CANDIDATE_SOURCE_STEPS_PER_MAP,
    )
    refinement_trace_node_custody = _trace_node_custody(
        lane_id="refinement",
        position_AU=arrays["refinement_position_AU"],
        velocity_AU_per_year=arrays["refinement_velocity_AU_per_year"],
        lifecycle=arrays["refinement_lifecycle"],
        state_ids=state_ids,
        source_steps_per_map_step=REFINEMENT_SOURCE_STEPS_PER_MAP,
    )
    return {
        "arrays": arrays,
        "candidate_events": candidate["route_records"]
        + candidate["removal_records"],
        "refinement_events": refinement["route_records"]
        + refinement["removal_records"],
        "receipts": {
            "source_parent_custody": {
                "parent_path_content_sha256": parent_before,
                "parent_dynamic_node_sha256": source.dynamic_node_sha256,
                "candidate_exact_parent_slices": bool(
                    candidate["source_parent_custody"][
                        "every_slice_verified_by_constructor"
                    ]
                    and candidate["source_parent_custody"][
                        "final_global_source_step"
                    ]
                    == SOURCE_STEP_COUNT
                ),
                "refinement_exact_parent_slices": bool(
                    refinement["source_parent_custody"][
                        "every_slice_verified_by_constructor"
                    ]
                    and refinement["source_parent_custody"][
                        "final_global_source_step"
                    ]
                    == SOURCE_STEP_COUNT
                ),
                "parent_unchanged_after_execution": bool(
                    parent_before == parent_after
                ),
                "candidate": candidate["source_parent_custody"],
                "refinement": refinement["source_parent_custody"],
                "guard_overlay_sha256": overlay.content_sha256,
            },
            "split_resume": {
                "candidate": candidate_split,
                "refinement": refinement_split,
                "candidate_bitwise_exact": candidate_split["bitwise_exact"],
                "refinement_bitwise_exact": refinement_split["bitwise_exact"],
            },
            "event_accounting": {
                "candidate": candidate_events,
                "refinement": refinement_events,
                "candidate_exact": candidate_events["exact"],
                "refinement_exact": refinement_events["exact"],
            },
            "runtime": {
                "timings_seconds": timings,
                "candidate_maximum_kepler_iterations": candidate[
                    "maximum_kepler_iterations"
                ],
                "refinement_maximum_kepler_iterations": refinement[
                    "maximum_kepler_iterations"
                ],
                "runtime_identity_start": runtime_identity_start,
                "runtime_identity_end": runtime_identity_end,
                "runtime_identity_stable": True,
                "runtime_identity_sha256": hashlib.sha256(
                    RUNTIME_IDENTITY_DOMAIN
                    + _canonical_json(runtime_identity_start)
                ).hexdigest(),
            },
            "endpoint_state_custody": {
                "candidate_state_anchors": {
                    "genesis": candidate["endpoint_state_sha256"][0],
                    "split": candidate["endpoint_state_sha256"][
                        SPLIT_CANDIDATE_MAP_STEP
                    ],
                    "terminal": candidate["endpoint_state_sha256"][-1],
                },
                "refinement_state_anchors": {
                    "genesis": refinement["endpoint_state_sha256"][0],
                    "split": refinement["endpoint_state_sha256"][
                        SPLIT_REFINEMENT_MAP_STEP
                    ],
                    "terminal": refinement["endpoint_state_sha256"][-1],
                },
                "candidate_trace_node_custody": (
                    candidate_trace_node_custody
                ),
                "refinement_trace_node_custody": (
                    refinement_trace_node_custody
                ),
            },
        },
    }


NumericalExecutor = Callable[..., Mapping[str, Any]]
NumericalAnalyzer = Callable[..., Mapping[str, Any]]

# Capture the reviewed callables once.  Public calibration/official entry
# points deliberately expose no dependency-injection surface; mocked helpers
# cannot publish an eligible calibration or a PASSED official result.
_EXACT_NUMERICAL_EXECUTOR: NumericalExecutor = execute_numerics
_EXACT_NUMERICAL_ANALYZER: NumericalAnalyzer = analyze_numerical_bundle


def _calibration_output_allowed(
    output_path: Path, project_root: Path
) -> Path:
    target = Path(os.path.abspath(os.fspath(output_path)))
    canonical = Path(
        os.path.abspath(os.fspath(project_root / DEFAULT_CANONICAL_ROOT))
    )
    if target == canonical or canonical in target.parents:
        raise M3ProtocolError(
            "calibration output cannot enter the canonical qualification namespace"
        )
    parent_descriptor = _open_absolute_directory_no_follow(
        target.parent, "calibration output parent"
    )
    try:
        try:
            os.stat(target.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise M3ProtocolError("cannot inspect calibration output") from exc
        else:
            raise M3ProtocolError("calibration output must be a new path")
    finally:
        os.close(parent_descriptor)
    return target


def run_calibration(
    *,
    project_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Execute and retain a self-contained non-authorizing calibration bundle."""

    root = Path(project_root).resolve()
    target = _calibration_output_allowed(Path(output_path), root)
    protocol = build_protocol(root)
    validate_protocol(protocol)
    target, directory_descriptor = _mkdir_absolute_no_follow_create_only_open(
        target, "calibration output"
    )
    try:
        protocol_payload = _json_bytes(protocol)
        _write_bytes_at_create_only(
            directory_descriptor, PROTOCOL_NAME, protocol_payload
        )
        _, prepared_profiles, source = _input_record(root)
        with tempfile.TemporaryDirectory(
            prefix="jx-m3-calibration-"
        ) as temporary:
            work = Path(temporary)
            bundle = _EXACT_NUMERICAL_EXECUTOR(
                project_root=root,
                prepared_profiles=prepared_profiles,
                source=source,
                protocol=protocol,
                work_directory=work,
            )
            _validate_numerical_bundle(bundle)
            analysis = _EXACT_NUMERICAL_ANALYZER(
                bundle,
                state_ids=protocol["inputs"]["state_ids"],
                family_ids=protocol["inputs"]["family_ids"],
            )
            arrays = bundle.get("arrays", {})
            if type(arrays) is not dict or any(
                type(name) is not str or type(value) is not np.ndarray
                for name, value in arrays.items()
            ):
                raise M3NumericalError(
                    "calibration executor returned invalid arrays"
                )
            array_payload = _npz_bytes(arrays)
            _write_bytes_at_create_only(
                directory_descriptor, ARRAY_ARCHIVE_NAME, array_payload
            )
            event_ledger = _content_record(
                {
                    "schema": "jx.long-horizon.m3-tracer-preflight.events.v1",
                    "candidate": bundle["candidate_events"],
                    "refinement": bundle["refinement_events"],
                    "claims": dict(CLAIMS),
                },
                EVENT_DOMAIN,
            )
            event_payload = _json_bytes(event_ledger)
            _write_bytes_at_create_only(
                directory_descriptor, EVENT_LEDGER_NAME, event_payload
            )
            runtime_identity = bundle["receipts"]["runtime"][
                "runtime_identity_end"
            ]
            archive_payload = _create_executed_archive(
                project_root=root,
                work_directory=work,
                expected_code_identity=protocol["code_identity"],
                expected_runtime_identity=runtime_identity,
            )
            _write_bytes_at_create_only(
                directory_descriptor, SOURCE_ARCHIVE_NAME, archive_payload
            )
            record = _content_record(
                {
                    "schema": CALIBRATION_SCHEMA,
                    "run_id": RUN_ID,
                    "classification": CLASSIFICATION,
                    "status": "CALIBRATED_NON_AUTHORIZING",
                    "completed_utc": _utc_now(),
                    "protocol_content_sha256": protocol["content_sha256"],
                    "protocol_file_sha256": hashlib.sha256(
                        protocol_payload
                    ).hexdigest(),
                    "precision_selection_precursor": dict(
                        PRECURSOR_CALIBRATION_CONTEXT
                    ),
                    "measured_analysis": analysis,
                    "state_arrays": {
                        "path": ARRAY_ARCHIVE_NAME,
                        "file_sha256": hashlib.sha256(
                            array_payload
                        ).hexdigest(),
                        "array_content_sha256": {
                            name: _array_sha256(value)
                            for name, value in sorted(arrays.items())
                        },
                        "array_bundle_sha256": _arrays_sha256(arrays),
                    },
                    "event_ledger": {
                        "path": EVENT_LEDGER_NAME,
                        "file_sha256": hashlib.sha256(
                            event_payload
                        ).hexdigest(),
                        "content_sha256": event_ledger["content_sha256"],
                    },
                    "executed_sources": {
                        "path": SOURCE_ARCHIVE_NAME,
                        "file_sha256": hashlib.sha256(
                            archive_payload
                        ).hexdigest(),
                        "runtime_identity_sha256": bundle["receipts"]
                        ["runtime"]["runtime_identity_sha256"],
                    },
                    "gate_thresholds_selected": False,
                    "official_attempt_authorized": False,
                    "staged_one_thousand_year_run_authorized": False,
                    "claims": dict(CLAIMS),
                    "interpretation": (
                        "Calibration measures this bounded implementation only. "
                        "It is not an acceptance threshold, qualification pass, "
                        "or science result."
                    ),
                },
                CALIBRATION_DOMAIN,
            )
            record_payload = _json_bytes(record)
            _write_bytes_at_create_only(
                directory_descriptor,
                CALIBRATION_RECORD_NAME,
                record_payload,
            )
        completion = _content_record(
            {
                "schema": CALIBRATION_COMPLETION_SCHEMA,
                "run_id": RUN_ID,
                "status": "CALIBRATED_NON_AUTHORIZING",
                "completed_utc": _utc_now(),
                "protocol_content_sha256": protocol["content_sha256"],
                "protocol_file_sha256": hashlib.sha256(
                    protocol_payload
                ).hexdigest(),
                "calibration_content_sha256": record["content_sha256"],
                "calibration_file_sha256": hashlib.sha256(
                    record_payload
                ).hexdigest(),
                "states_file_sha256": hashlib.sha256(
                    array_payload
                ).hexdigest(),
                "events_file_sha256": hashlib.sha256(
                    event_payload
                ).hexdigest(),
                "executed_sources_file_sha256": hashlib.sha256(
                    archive_payload
                ).hexdigest(),
                "completion_marker_written_last": True,
                "official_attempt_authorized": False,
                "staged_one_thousand_year_run_authorized": False,
                "claims": dict(CLAIMS),
            },
            CALIBRATION_COMPLETION_DOMAIN,
        )
        completion_payload = _json_bytes(completion)
        expected_names = {
            PROTOCOL_NAME,
            ARRAY_ARCHIVE_NAME,
            EVENT_LEDGER_NAME,
            SOURCE_ARCHIVE_NAME,
            CALIBRATION_RECORD_NAME,
            CALIBRATION_COMPLETION_NAME,
        }
        snapshot = _read_allowed_directory_payloads_at_no_follow(
            directory_descriptor,
            expected_names,
            "calibration output before completion",
        )
        if set(snapshot) != expected_names - {CALIBRATION_COMPLETION_NAME}:
            raise M3ProtocolError(
                "calibration output is not ready for its final seal"
            )
        snapshot[CALIBRATION_COMPLETION_NAME] = completion_payload
        verified_snapshot = _validate_calibration_payload_snapshot(
            snapshot, protocol
        )
        if verified_snapshot["content_sha256"] != record["content_sha256"]:
            raise M3ProtocolError(
                "calibration payload snapshot changed before publication"
            )
        _require_directory_path_identity(
            target, directory_descriptor, "calibration output before completion"
        )
        _write_bytes_at_create_only(
            directory_descriptor,
            CALIBRATION_COMPLETION_NAME,
            completion_payload,
        )
        verified = validate_calibration_bundle(
            target,
            protocol,
            directory_descriptor=directory_descriptor,
        )
        _require_directory_path_identity(
            target, directory_descriptor, "calibration output after completion"
        )
        if verified["content_sha256"] != record["content_sha256"]:
            raise M3ProtocolError(
                "published calibration changed during verification"
            )
        return record
    finally:
        os.close(directory_descriptor)


def _create_executed_archive(
    *,
    project_root: Path,
    work_directory: Path,
    expected_code_identity: Mapping[str, Any],
    expected_runtime_identity: Mapping[str, Any] | None = None,
) -> bytes:
    expected_rows = expected_code_identity.get("files")
    if type(expected_rows) is not list or any(
        type(row) is not dict
        or type(row.get("path")) is not str
        or type(row.get("size_bytes")) is not int
        or type(row.get("sha256")) is not str
        for row in expected_rows
    ):
        raise M3ProtocolError("protocol code identity is malformed")
    expected = {row["path"]: row for row in expected_rows}
    source_paths = _executed_source_paths(project_root)
    if len(expected) != len(expected_rows) or set(expected) != {
        path.as_posix() for path in source_paths
    }:
        raise M3ProtocolError("executed source inventory changed after preparation")
    code_digest = hashlib.sha256(CODE_FILES_DOMAIN)
    for row in expected_rows:
        code_digest.update(_canonical_json(row))
    if (
        expected_code_identity.get("file_count") != len(expected_rows)
        or expected_code_identity.get("files_content_sha256")
        != code_digest.hexdigest()
    ):
        raise M3ProtocolError("protocol code identity digest is inconsistent")
    imported_modules = expected_code_identity.get("imported_local_modules")
    if type(imported_modules) is not list or any(
        type(row) is not dict
        or row.get("path") not in expected
        or row.get("sha256") != expected[row["path"]]["sha256"]
        or row.get("size_bytes") != expected[row["path"]]["size_bytes"]
        for row in imported_modules
    ):
        raise M3ProtocolError("protocol imported-module identity is malformed")
    archive_runtime_identity = _runtime_identity(
        include_numerical_packages=True,
        project_root=project_root,
        expected_code_identity=expected_code_identity,
    )
    if (
        expected_runtime_identity is not None
        and archive_runtime_identity != expected_runtime_identity
    ):
        raise M3ProtocolError(
            "archive runtime differs from the numerical execution receipt"
        )
    runtime_sha256 = hashlib.sha256(
        RUNTIME_IDENTITY_DOMAIN + _canonical_json(archive_runtime_identity)
    ).hexdigest()
    manifest: dict[str, Any] = {
        "schema": CODE_ARCHIVE_SCHEMA,
        "created_utc": _utc_now(),
        "runtime_identity": archive_runtime_identity,
        "runtime_identity_sha256": runtime_sha256,
        "runtime_matches_execution_receipt": (
            expected_runtime_identity is not None
        ),
        "protocol_code_files_content_sha256": expected_code_identity.get(
            "files_content_sha256"
        ),
        "protocol_imported_local_modules": imported_modules,
        "protocol_git_commit": expected_code_identity.get("git_commit"),
        "files": [],
        "claims": dict(CLAIMS),
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for relative in source_paths:
            payload = _read_repository_file_no_follow(project_root, relative)
            payload_sha256 = hashlib.sha256(payload).hexdigest()
            expected_row = expected[relative.as_posix()]
            if (
                len(payload) != expected_row["size_bytes"]
                or payload_sha256 != expected_row["sha256"]
            ):
                raise M3ProtocolError(
                    f"executed source changed after preparation: {relative}"
                )
            member = f"source/{relative.as_posix()}"
            archive.writestr(member, payload)
            manifest["files"].append(
                {
                    "path": member,
                    "size_bytes": len(payload),
                    "sha256": payload_sha256,
                }
            )
        if work_directory.exists():
            for path in sorted(work_directory.rglob("*")):
                if path.is_dir():
                    continue
                if path.is_symlink() or not path.is_file():
                    raise M3ProtocolError("checkpoint work tree contains a link")
                payload = path.read_bytes()
                relative = path.relative_to(work_directory).as_posix()
                member = f"split-checkpoints/{relative}"
                archive.writestr(member, payload)
                manifest["files"].append(
                    {
                        "path": member,
                        "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
        manifest["archived_files_content_sha256"] = hashlib.sha256(
            CODE_FILES_DOMAIN + _canonical_json(manifest["files"])
        ).hexdigest()
        manifest_payload = json.dumps(
            _plain(manifest), indent=2, sort_keys=True, allow_nan=False
        ).encode("ascii") + b"\n"
        archive.writestr("MANIFEST.json", manifest_payload)
    return buffer.getvalue()


_M3_ARRAY_NAMES = frozenset(
    {
        "candidate_position_AU",
        "candidate_velocity_AU_per_year",
        "candidate_lifecycle",
        "candidate_minimum_source_distance_AU",
        "refinement_position_AU",
        "refinement_velocity_AU_per_year",
        "refinement_lifecycle",
        "refinement_minimum_source_distance_AU",
        "ias15_tracer_position_AU",
        "ias15_tracer_velocity_AU_per_year",
        "ias15_source_position_AU",
        "ias15_source_velocity_AU_per_year",
        "ias15_source_only_position_AU",
        "ias15_source_only_velocity_AU_per_year",
        "candidate_source_position_AU",
        "candidate_source_velocity_AU_per_year",
        "refinement_source_position_AU",
        "refinement_source_velocity_AU_per_year",
        "source_gm_AU3_per_year2",
    }
)


def _archive_member_payload(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    row: Mapping[str, Any],
    label: str,
) -> bytes:
    if (
        type(row) is not dict
        or set(row) != {"path", "size_bytes", "sha256"}
        or row.get("path") != info.filename
        or type(row.get("size_bytes")) is not int
        or row.get("size_bytes") != info.file_size
        or type(row.get("sha256")) is not str
    ):
        raise M3ProtocolError(f"archive manifest row is malformed: {label}")
    try:
        payload = archive.read(info)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise M3ProtocolError(f"archive member is unreadable: {label}") from exc
    if (
        len(payload) != row["size_bytes"]
        or hashlib.sha256(payload).hexdigest() != row["sha256"]
    ):
        raise M3ProtocolError(f"archive member custody failed: {label}")
    return payload


def _rebuild_expected_m2_annual_bindings(
    protocol: Mapping[str, Any], checkpoint: Any
) -> tuple[Any, ...]:
    """Independently reconstruct the ten exact M2 annual source bindings."""

    try:
        from jxplanetx.long_horizon import (
            M2_DERIVED_ACCELERATION_BOUND_AU_PER_YEAR2,
            OFFICIAL_RUN_RELATIVE_PATH,
            build_m2_frozen_source_path_tile,
            load_verified_m2_source_path,
            seal_m2_guard_overlay,
        )

        project_root = Path(__file__).resolve().parents[1]
        source = load_verified_m2_source_path(
            project_root / OFFICIAL_RUN_RELATIVE_PATH
        )
        expectations = protocol["inputs"]["m2_execution_source_custody"]
        if (
            type(expectations) is not dict
            or source.content_sha256
            != expectations.get("parent_path_content_sha256")
            or source.dynamic_node_sha256
            != expectations.get("parent_dynamic_node_sha256")
        ):
            raise M3ProtocolError("protocol and verified M2 source path differ")
        overlay = seal_m2_guard_overlay(
            source,
            source_encounter_radii_AU=np.full(
                SOURCE_COUNT, ALGORITHMIC_GUARD_RADIUS_AU, dtype=np.float64
            ),
            certified_relative_acceleration_bounds_AU_per_year2=np.full(
                SOURCE_COUNT,
                M2_DERIVED_ACCELERATION_BOUND_AU_PER_YEAR2,
                dtype=np.float64,
            ),
        )
        if overlay.content_sha256 != expectations.get("guard_overlay_sha256"):
            raise M3ProtocolError("protocol M2 guard overlay differs")
        dynamic_hashes = expectations.get(
            "annual_source_slice_dynamic_node_sha256"
        )
        binding_hashes = expectations.get("annual_source_tile_binding_sha256")
        if (
            type(dynamic_hashes) is not list
            or type(binding_hashes) is not list
            or len(dynamic_hashes) != HORIZON_YEARS
            or len(binding_hashes) != HORIZON_YEARS
        ):
            raise M3ProtocolError("protocol annual M2 custody roster changed")
        bindings = []
        for year in range(HORIZON_YEARS):
            source_slice = source.slice_years(year, year + 1)
            binding = build_m2_frozen_source_path_tile(source_slice, overlay)
            if (
                binding.dynamic_node_sha256 != dynamic_hashes[year]
                or binding.content_sha256 != binding_hashes[year]
                or checkpoint._tile_metadata(binding.tile).get("content_sha256")
                != binding.tile.content_sha256
            ):
                raise M3ProtocolError(
                    "independently rebuilt annual M2 source custody changed"
                )
            bindings.append(binding)
        return tuple(bindings)
    except M3ProtocolError:
        raise
    except Exception as exc:
        raise M3ProtocolError(
            "cannot independently rebuild annual M2 source custody"
        ) from exc


def _checkpoint_route_event_row(
    value: object, *, annual_map_offset: int
) -> dict[str, Any]:
    """Convert one archived checkpoint route row to the retained EVENTS form."""

    row_keys = {
        "map_step_index",
        "lane_index",
        "tracer_index",
        "guard_phase",
        "guard_reason",
        "guard_source_index",
        "route_result",
    }
    route_keys = {
        "method_id",
        "scientific_classification",
        "project_scope",
        "claims",
        "tracer_index",
        "source_tile_sha256",
        "source_segment_sha256",
        "global_source_step",
        "source_steps_per_map_step",
        "events",
        "endpoint_relative_position_AU_hex",
        "endpoint_relative_velocity_AU_per_year_hex",
        "accepted_step_count",
        "derivative_evaluation_count",
        "minimum_evaluated_stage_distance_AU_hex",
        "minimum_committed_stage_distance_AU_hex",
        "maximum_accepted_primary_dynamical_phase_radians_hex",
        "maximum_committed_primary_dynamical_phase_radians_hex",
        "completed_map_fraction_hex",
        "runtime_identity",
    }
    if type(value) is not dict or set(value) != row_keys:
        raise M3ProtocolError("checkpoint route row schema changed")
    route = value.get("route_result")
    events = route.get("events") if type(route) is dict else None
    runtime = route.get("runtime_identity") if type(route) is dict else None
    if (
        type(route) is not dict
        or set(route) != route_keys
        or type(events) is not list
        or any(
            type(event) is not dict or set(event) != _ROUTED_EVENT_KEYS
            for event in events
        )
        or type(runtime) is not dict
        or route.get("tracer_index") != value.get("tracer_index")
        or type(value.get("map_step_index")) is not int
        or value["map_step_index"] < 0
    ):
        raise M3ProtocolError("checkpoint route-result schema changed")
    normalized_runtime = dict(runtime)
    normalized_runtime.pop("source_tile_sha256", None)
    return {
        "record_kind": "ROUTED_REJOIN",
        "global_map_step": annual_map_offset + value["map_step_index"],
        "lane_index": value["lane_index"],
        "tracer_index": value["tracer_index"],
        "guard_phase": value["guard_phase"],
        "guard_reason": value["guard_reason"],
        "guard_source_index": value["guard_source_index"],
        "source_tile_sha256": route["source_tile_sha256"],
        "source_segment_sha256": route["source_segment_sha256"],
        "global_source_step": route["global_source_step"],
        "source_steps_per_map_step": route["source_steps_per_map_step"],
        "accepted_step_count": route["accepted_step_count"],
        "derivative_evaluation_count": route["derivative_evaluation_count"],
        "endpoint_relative_position_AU_hex": route[
            "endpoint_relative_position_AU_hex"
        ],
        "endpoint_relative_velocity_AU_per_year_hex": route[
            "endpoint_relative_velocity_AU_per_year_hex"
        ],
        "minimum_evaluated_stage_distance_AU_hex": route[
            "minimum_evaluated_stage_distance_AU_hex"
        ],
        "minimum_committed_stage_distance_AU_hex": route[
            "minimum_committed_stage_distance_AU_hex"
        ],
        "maximum_accepted_primary_dynamical_phase_radians_hex": route[
            "maximum_accepted_primary_dynamical_phase_radians_hex"
        ],
        "maximum_committed_primary_dynamical_phase_radians_hex": route[
            "maximum_committed_primary_dynamical_phase_radians_hex"
        ],
        "completed_map_fraction_hex": route["completed_map_fraction_hex"],
        "event_count": len(events),
        "event_content": events,
        "runtime_identity_sha256": hashlib.sha256(
            EVENT_DOMAIN + _canonical_json(normalized_runtime)
        ).hexdigest(),
    }


def _checkpoint_removal_event_row(value: object) -> dict[str, Any]:
    """Convert one archived checkpoint removal delta to retained EVENTS form."""

    keys = {
        "lane_index",
        "tracer_index",
        "lifecycle_code",
        "global_source_step",
        "source_steps_per_map_step",
        "guard_phase",
        "guard_reason",
        "guard_source_index",
        "source_segment_sha256",
        "route_runtime_identity_sha256",
        "accepted_step_count",
        "derivative_evaluation_count",
        "minimum_committed_stage_distance_AU_hex",
        "maximum_committed_primary_dynamical_phase_radians_hex",
        "events",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        raise M3ProtocolError("checkpoint removal row schema changed")
    events = value.get("events")
    if (
        type(events) is not list
        or any(
            type(event) is not dict or set(event) != _ROUTED_EVENT_KEYS
            for event in events
        )
    ):
        raise M3ProtocolError("checkpoint removal-event schema changed")
    row = {
        "record_kind": "TERMINAL_REMOVAL",
        "content_sha256": value["content_sha256"],
        "lane_index": value["lane_index"],
        "tracer_index": value["tracer_index"],
        "lifecycle_code": value["lifecycle_code"],
        "global_source_step": value["global_source_step"],
        "source_steps_per_map_step": value["source_steps_per_map_step"],
        "guard_phase": value["guard_phase"],
        "guard_reason": value["guard_reason"],
        "guard_source_index": value["guard_source_index"],
        "source_segment_sha256": value["source_segment_sha256"],
        "route_runtime_identity_sha256": value[
            "route_runtime_identity_sha256"
        ],
        "accepted_step_count": value["accepted_step_count"],
        "derivative_evaluation_count": value["derivative_evaluation_count"],
        "minimum_committed_stage_distance_AU_hex": value[
            "minimum_committed_stage_distance_AU_hex"
        ],
        "maximum_committed_primary_dynamical_phase_radians_hex": value[
            "maximum_committed_primary_dynamical_phase_radians_hex"
        ],
        "event_count": len(events),
        "event_content": events,
    }
    if _removal_ledger_sha256(row) != row["content_sha256"]:
        raise M3ProtocolError("checkpoint removal content digest failed")
    return row


def _require_checkpoint_event_ledger_crosslink(
    receipt: Mapping[str, Any],
    retained_rows: object,
    checkpoint_route_rows: Sequence[Mapping[str, Any]],
    checkpoint_removal_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    """Bind both split-ledger hashes to independently retained row content."""

    if type(receipt) is not dict or type(retained_rows) is not list:
        raise M3ProtocolError("split event-ledger receipt is malformed")
    one_map_routes = [
        row
        for row in retained_rows
        if type(row) is dict and row.get("record_kind") == "ROUTED_REJOIN"
    ]
    one_map_removals = [
        row
        for row in retained_rows
        if type(row) is dict and row.get("record_kind") == "TERMINAL_REMOVAL"
    ]
    if len(one_map_routes) + len(one_map_removals) != len(retained_rows):
        raise M3ProtocolError("retained split event roster changed")

    def ledger_sha256(
        routes: Sequence[Mapping[str, Any]],
        removals: Sequence[Mapping[str, Any]],
    ) -> str:
        ledger = {
            "route_records": [
                {
                    key: item
                    for key, item in row.items()
                    if key != "source_tile_sha256"
                }
                for row in routes
            ],
            "removal_records": list(removals),
        }
        return hashlib.sha256(
            EVENT_DOMAIN + _canonical_json(ledger)
        ).hexdigest()

    expected_sha256 = ledger_sha256(one_map_routes, one_map_removals)
    checkpoint_sha256 = ledger_sha256(
        checkpoint_route_rows, checkpoint_removal_rows
    )
    if (
        receipt.get("expected_event_ledger_sha256") != expected_sha256
        or receipt.get("split_event_ledger_sha256") != checkpoint_sha256
        or expected_sha256 != checkpoint_sha256
    ):
        raise M3ProtocolError(
            "split event ledgers do not match retained events and checkpoints"
        )
    return expected_sha256, checkpoint_sha256


def _validate_checkpoint_archive_members(
    archive: zipfile.ZipFile,
    infos: Mapping[str, zipfile.ZipInfo],
    rows: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
    split_receipt: Mapping[str, Any] | None,
    expected_events: Mapping[str, Any] | None,
) -> None:
    """Statically verify every retained restricted-WH checkpoint chain."""

    checkpoint_names = sorted(
        name for name in infos if name.startswith("split-checkpoints/")
    )
    if not checkpoint_names:
        if split_receipt is not None:
            raise M3ProtocolError("executed-source archive omits checkpoint chains")
        return
    try:
        from jxplanetx import restricted_wh_checkpoint as checkpoint
    except Exception as exc:
        raise M3ProtocolError("checkpoint verifier is unavailable") from exc

    if split_receipt is not None and (
        type(split_receipt) is not dict
        or set(split_receipt) < {"candidate", "refinement"}
        or type(expected_events) is not dict
        or set(expected_events) != {"candidate", "refinement"}
    ):
        raise M3ProtocolError("split/resume receipt is malformed")
    implementation_identity = dict(
        checkpoint._plain_tree(checkpoint.checkpoint_implementation_identity())
    )
    expected_bindings = _rebuild_expected_m2_annual_bindings(
        protocol, checkpoint
    )
    consumed: set[str] = set()
    for lane in ("candidate", "refinement"):
        run_id = f"{RUN_ID}-{lane}-split-v1"
        namespace = "run-" + hashlib.sha256(
            checkpoint.CHECKPOINT_NAMESPACE_DOMAIN
            + run_id.encode("utf-8")
            + b"\0"
            + protocol["content_sha256"].encode("ascii")
        ).hexdigest()
        prefix = (
            f"split-checkpoints/{lane}_split_checkpoint/{namespace}/"
        )
        lane_names = [name for name in checkpoint_names if name.startswith(prefix)]
        if not lane_names:
            if split_receipt is not None:
                raise M3ProtocolError(f"{lane} checkpoint chain is absent")
            continue
        consumed.update(lane_names)
        suffixes = {name[len(prefix) :] for name in lane_names}
        if any("/" in suffix or not suffix for suffix in suffixes):
            raise M3ProtocolError(f"{lane} checkpoint archive path is invalid")
        lock_name = prefix + ".append.lock"
        if lock_name not in infos:
            raise M3ProtocolError(f"{lane} checkpoint append lock is absent")
        if _archive_member_payload(
            archive, infos[lock_name], rows[lock_name], lock_name
        ) != b"":
            raise M3ProtocolError(f"{lane} checkpoint append lock is not empty")

        record_pattern = re.compile(r"checkpoint_([0-9]{20})\.json")
        array_pattern = re.compile(
            r"checkpoint_([0-9]{20})\.array-([0-9a-f]{64})\.npz"
        )
        record_suffixes = sorted(
            suffix for suffix in suffixes if record_pattern.fullmatch(suffix)
        )
        array_suffixes = {
            suffix for suffix in suffixes if array_pattern.fullmatch(suffix)
        }
        if suffixes != {".append.lock", *record_suffixes, *array_suffixes}:
            raise M3ProtocolError(f"{lane} checkpoint roster contains debris")
        indices: list[int] = []
        for name in record_suffixes:
            matched = record_pattern.fullmatch(name)
            if matched is None:
                raise M3ProtocolError(f"{lane} checkpoint record name is invalid")
            indices.append(int(matched.group(1)))
        if indices != list(range(len(indices))):
            raise M3ProtocolError(f"{lane} checkpoint indices are not contiguous")
        if split_receipt is not None and indices != list(range(HORIZON_YEARS)):
            raise M3ProtocolError(f"{lane} checkpoint chain is incomplete")

        records: list[dict[str, Any]] = []
        checkpoint_route_rows: list[dict[str, Any]] = []
        checkpoint_removal_rows: list[dict[str, Any]] = []
        predecessor_content: str | None = None
        predecessor_state: str | None = None
        predecessor_lineage: str | None = None
        predecessor_source_final: str | None = None
        predecessor_source_lineage: str | None = None
        removal_rows: list[dict[str, Any]] | None = None
        runtime_identity: object = None
        referenced_arrays: set[str] = set()
        for index, suffix in zip(indices, record_suffixes, strict=True):
            member = prefix + suffix
            record = _decode_json_object(
                _archive_member_payload(
                    archive, infos[member], rows[member], member
                ),
                member,
            )
            unsigned = dict(record)
            recorded_content = unsigned.pop("checkpoint_content_sha256", None)
            if (
                set(record) != set(checkpoint._RECORD_KEYS)
                or recorded_content
                != hashlib.sha256(
                    checkpoint.CHECKPOINT_DOMAIN
                    + checkpoint._canonical_json(unsigned)
                ).hexdigest()
                or record.get("schema") != checkpoint.CHECKPOINT_SCHEMA
                or record.get("method_id") != checkpoint.CHECKPOINT_METHOD_ID
                or record.get("run_id") != run_id
                or record.get("protocol_sha256") != protocol["content_sha256"]
                or record.get("checkpoint_namespace") != namespace
                or record.get("checkpoint_index") != index
                or record.get("previous_checkpoint_content_sha256")
                != predecessor_content
                or record.get("implementation_identity")
                != implementation_identity
            ):
                raise M3ProtocolError(f"{lane} checkpoint record custody failed")
            if index == 0:
                runtime_identity = record.get("checkpoint_runtime_identity")
            elif record.get("checkpoint_runtime_identity") != runtime_identity:
                raise M3ProtocolError(f"{lane} checkpoint runtime changed")
            required_genesis = (
                record["genesis_state_sha256"]
                if index == 0
                else records[0]["genesis_state_sha256"]
            )
            try:
                checkpoint._verify_record_static(
                    record,
                    implementation_identity,
                    runtime_identity,
                    required_genesis,
                )
            except Exception as exc:
                raise M3ProtocolError(
                    f"{lane} checkpoint static contract failed"
                ) from exc

            array_name = record.get("array_file")
            array_digest = record.get("array_file_sha256")
            expected_array_name = (
                f"checkpoint_{index:020d}.array-{array_digest}.npz"
            )
            array_member = prefix + str(array_name)
            if (
                type(array_name) is not str
                or array_name != expected_array_name
                or array_name not in array_suffixes
                or array_member not in infos
                or rows[array_member].get("size_bytes")
                != record.get("array_file_size_bytes")
                or rows[array_member].get("sha256") != array_digest
            ):
                raise M3ProtocolError(f"{lane} checkpoint array anchor failed")
            array_payload = _archive_member_payload(
                archive, infos[array_member], rows[array_member], array_member
            )
            arrays = _decode_npz_arrays_payload(
                array_payload,
                array_member,
                expected_names=set(checkpoint._ARRAY_KEYS),
            )
            referenced_arrays.add(array_name)
            result = record.get("result")
            input_metadata = record.get("input_state_metadata")
            state_metadata = record.get("state_metadata")
            if (
                type(result) is not dict
                or type(input_metadata) is not dict
                or type(state_metadata) is not dict
                or type(result.get("route_records")) is not list
                or type(result.get("new_removal_records")) is not list
            ):
                raise M3ProtocolError(f"{lane} checkpoint semantics are malformed")
            annual_map_count = (
                CANDIDATE_MAP_STEP_COUNT
                if lane == "candidate"
                else REFINEMENT_MAP_STEP_COUNT
            ) // HORIZON_YEARS
            checkpoint_route_rows.extend(
                _checkpoint_route_event_row(
                    row, annual_map_offset=index * annual_map_count
                )
                for row in result["route_records"]
            )
            checkpoint_removal_rows.extend(
                _checkpoint_removal_event_row(row)
                for row in result["new_removal_records"]
            )
            genesis_removals = record.get("genesis_removal_records")
            if index == 0:
                if type(genesis_removals) is not list:
                    raise M3ProtocolError(
                        f"{lane} checkpoint genesis ledger is absent"
                    )
                input_removals = list(genesis_removals)
            else:
                if genesis_removals is not None or removal_rows is None:
                    raise M3ProtocolError(
                        f"{lane} checkpoint successor ledger is invalid"
                    )
                input_removals = removal_rows
            output_removals = input_removals + result["new_removal_records"]
            try:
                input_hash = checkpoint._state_sha256(
                    input_metadata, arrays, input_removals, prefix="input"
                )
                state_hash = checkpoint._state_sha256(
                    state_metadata, arrays, output_removals, prefix="output"
                )
                lineage_hash = checkpoint._lineage_sha256(
                    input_metadata, arrays, prefix="input"
                )
                output_lineage = checkpoint._lineage_sha256(
                    state_metadata, arrays, prefix="output"
                )
                tile = checkpoint._tile_from_storage(
                    record.get("source_tile_metadata"), arrays
                )
                binding = expected_bindings[index]
                expected_tile_metadata = checkpoint._tile_metadata(binding.tile)
                expected_source_arrays = checkpoint._tile_arrays(binding.tile)
                source_lineage = checkpoint._source_lineage_sha256(tile)
                source_initial = checkpoint._source_boundary_sha256(tile, 0)
                source_final = checkpoint._source_boundary_sha256(
                    tile, tile.node_count - 1
                )
                minimum_digest = hashlib.sha256()
                checkpoint._array_digest_update(
                    minimum_digest,
                    "minimum_source_distance_AU",
                    arrays["minimum_source_distance_AU"],
                )
            except Exception as exc:
                raise M3ProtocolError(
                    f"{lane} checkpoint semantic reconstruction failed"
                ) from exc
            if (
                record.get("source_tile_metadata") != expected_tile_metadata
                or any(
                    not _bitwise_equal(arrays[name], expected_value)
                    for name, expected_value in expected_source_arrays.items()
                )
                or input_hash != record.get("input_state_sha256")
                or state_hash != record.get("state_sha256")
                or lineage_hash != record.get("lineage_sha256")
                or output_lineage != lineage_hash
                or (index == 0 and input_hash != record.get("genesis_state_sha256"))
                or (index > 0 and input_hash != predecessor_state)
                or (index > 0 and lineage_hash != predecessor_lineage)
                or source_lineage != record.get("source_lineage_sha256")
                or source_initial != record.get("source_initial_node_sha256")
                or source_final != record.get("source_final_node_sha256")
                or (index > 0 and source_initial != predecessor_source_final)
                or (index > 0 and source_lineage != predecessor_source_lineage)
                or result.get("source_tile_sha256") != tile.content_sha256
                or result.get("minimum_source_distance_array_sha256")
                != minimum_digest.hexdigest()
            ):
                raise M3ProtocolError(f"{lane} checkpoint semantic hash failed")
            segment = {
                "input_state_sha256": input_hash,
                "state_sha256": state_hash,
                "lineage_sha256": lineage_hash,
                "source_tile_sha256": tile.content_sha256,
                "spec": record.get("spec"),
                "route_policy": record.get("route_policy"),
                "route_worker_count": record.get("route_worker_count"),
                "result": result,
            }
            if checkpoint._segment_sha256(segment) != record.get("segment_sha256"):
                raise M3ProtocolError(f"{lane} checkpoint segment hash failed")
            records.append(record)
            predecessor_content = record["checkpoint_content_sha256"]
            predecessor_state = state_hash
            predecessor_lineage = lineage_hash
            predecessor_source_final = source_final
            predecessor_source_lineage = source_lineage
            removal_rows = output_removals
        if referenced_arrays != array_suffixes:
            raise M3ProtocolError(f"{lane} checkpoint arrays are orphaned or absent")
        if split_receipt is not None:
            receipt = split_receipt[lane]
            expectations = protocol["inputs"]["m2_execution_source_custody"]
            one_map_rows = expected_events[lane]  # type: ignore[index]
            _require_checkpoint_event_ledger_crosslink(
                receipt,
                one_map_rows,
                checkpoint_route_rows,
                checkpoint_removal_rows,
            )
            if (
                type(receipt) is not dict
                or receipt.get("annual_checkpoint_count") != HORIZON_YEARS
                or receipt.get("genesis_state_sha256")
                != records[0]["genesis_state_sha256"]
                or receipt.get("split_checkpoint_content_sha256")
                != records[SPLIT_EPOCH_YEARS - 1]["checkpoint_content_sha256"]
                or receipt.get("final_checkpoint_content_sha256")
                != records[-1]["checkpoint_content_sha256"]
                or receipt.get("first_half_state_sha256")
                != records[SPLIT_EPOCH_YEARS - 1]["state_sha256"]
                or receipt.get("loaded_split_checkpoint_state_sha256")
                != records[SPLIT_EPOCH_YEARS - 1]["state_sha256"]
                or receipt.get("terminal_state_sha256")
                != records[-1]["state_sha256"]
                or receipt.get("loaded_final_checkpoint_state_sha256")
                != records[-1]["state_sha256"]
                or receipt.get("annual_source_slice_dynamic_node_sha256")
                != expectations["annual_source_slice_dynamic_node_sha256"]
                or receipt.get("annual_source_tile_binding_sha256")
                != expectations["annual_source_tile_binding_sha256"]
            ):
                raise M3ProtocolError(
                    f"{lane} checkpoint chain differs from split receipt"
                )
    if consumed != set(checkpoint_names):
        raise M3ProtocolError("checkpoint archive contains an unknown lane or path")


def _validate_executed_source_archive_payload(
    payload: bytes,
    protocol: Mapping[str, Any],
    *,
    expected_runtime_identity: Mapping[str, Any] | None,
    split_receipt: Mapping[str, Any] | None,
    expected_events: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify every manifest row and every archived source/checkpoint byte."""

    expected_code = protocol.get("code_identity")
    if type(expected_code) is not dict or type(expected_code.get("files")) is not list:
        raise M3ProtocolError("protocol code identity is absent")
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            infos = _strict_zip_infos(archive, "executed-source archive")
            if "MANIFEST.json" not in infos:
                raise M3ProtocolError("executed-source manifest is absent")
            manifest_payload = archive.read(infos["MANIFEST.json"])
            manifest = _decode_json_object(
                manifest_payload, "executed-source MANIFEST.json"
            )
            manifest_keys = {
                "schema",
                "created_utc",
                "runtime_identity",
                "runtime_identity_sha256",
                "runtime_matches_execution_receipt",
                "protocol_code_files_content_sha256",
                "protocol_imported_local_modules",
                "protocol_git_commit",
                "files",
                "claims",
                "archived_files_content_sha256",
            }
            file_rows = manifest.get("files")
            if (
                set(manifest) != manifest_keys
                or manifest.get("schema") != CODE_ARCHIVE_SCHEMA
                or type(manifest.get("created_utc")) is not str
                or not manifest["created_utc"]
                or type(manifest.get("runtime_identity")) is not dict
                or manifest.get("runtime_identity_sha256")
                != hashlib.sha256(
                    RUNTIME_IDENTITY_DOMAIN
                    + _canonical_json(manifest["runtime_identity"])
                ).hexdigest()
                or manifest.get("runtime_matches_execution_receipt")
                is not (expected_runtime_identity is not None)
                or (
                    expected_runtime_identity is not None
                    and manifest.get("runtime_identity")
                    != expected_runtime_identity
                )
                or manifest.get("protocol_code_files_content_sha256")
                != expected_code.get("files_content_sha256")
                or manifest.get("protocol_imported_local_modules")
                != expected_code.get("imported_local_modules")
                or manifest.get("protocol_git_commit")
                != expected_code.get("git_commit")
                or manifest.get("claims") != CLAIMS
                or type(file_rows) is not list
            ):
                raise M3ProtocolError("executed-source manifest contract failed")
            if any(
                type(row) is not dict
                or set(row) != {"path", "size_bytes", "sha256"}
                or type(row.get("path")) is not str
                or type(row.get("size_bytes")) is not int
                or row["size_bytes"] < 0
                or type(row.get("sha256")) is not str
                for row in file_rows
            ):
                raise M3ProtocolError("executed-source file roster is malformed")
            row_map = {row["path"]: row for row in file_rows}
            if (
                len(row_map) != len(file_rows)
                or set(infos) != {"MANIFEST.json", *row_map}
                or manifest.get("archived_files_content_sha256")
                != hashlib.sha256(
                    CODE_FILES_DOMAIN + _canonical_json(file_rows)
                ).hexdigest()
            ):
                raise M3ProtocolError("executed-source archive roster changed")
            expected_source_rows = [
                {
                    "path": f"source/{row['path']}",
                    "size_bytes": row["size_bytes"],
                    "sha256": row["sha256"],
                }
                for row in expected_code["files"]
            ]
            if file_rows[: len(expected_source_rows)] != expected_source_rows:
                raise M3ProtocolError("archived local-code roster changed")
            checkpoint_rows = file_rows[len(expected_source_rows) :]
            if [row["path"] for row in checkpoint_rows] != sorted(
                row["path"] for row in checkpoint_rows
            ) or any(
                not row["path"].startswith("split-checkpoints/")
                for row in checkpoint_rows
            ):
                raise M3ProtocolError("archived checkpoint roster is noncanonical")
            for row in expected_source_rows:
                _archive_member_payload(
                    archive, infos[row["path"]], row_map[row["path"]], row["path"]
                )
            runtime = manifest["runtime_identity"]
            local_files = runtime.get("local_files")
            expected_local_files = {
                "file_count": expected_code.get("file_count"),
                "files_content_sha256": expected_code.get("files_content_sha256"),
                "imported_module_count_at_preparation": len(
                    expected_code.get("imported_local_modules", [])
                ),
            }
            if local_files != expected_local_files:
                raise M3ProtocolError("archive runtime is not bound to local code")
            runtime_imports = runtime.get("local_imported_modules", [])
            expected_by_path = {
                row["path"]: row for row in expected_code["files"]
            }
            if type(runtime_imports) is not list or any(
                type(row) is not dict
                or row.get("path") not in expected_by_path
                or row.get("size_bytes")
                != expected_by_path[row["path"]]["size_bytes"]
                or row.get("sha256") != expected_by_path[row["path"]]["sha256"]
                for row in runtime_imports
            ):
                raise M3ProtocolError("archive imported-module identity is unbound")
            _validate_checkpoint_archive_members(
                archive,
                infos,
                row_map,
                protocol,
                split_receipt,
                expected_events,
            )
    except M3ProtocolError:
        raise
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise M3ProtocolError("executed-source archive is invalid") from exc
    return manifest


def validate_calibration_bundle(
    calibration_directory: str | Path,
    protocol: Mapping[str, Any],
    *,
    directory_descriptor: int | None = None,
) -> dict[str, Any]:
    """Recompute a completed calibration from its retained arrays and ledgers."""

    validate_protocol(protocol)
    expected_names = {
        PROTOCOL_NAME,
        ARRAY_ARCHIVE_NAME,
        EVENT_LEDGER_NAME,
        SOURCE_ARCHIVE_NAME,
        CALIBRATION_RECORD_NAME,
        CALIBRATION_COMPLETION_NAME,
    }
    if directory_descriptor is None:
        payloads = _read_allowed_directory_payloads_no_follow(
            Path(calibration_directory), expected_names, "calibration bundle"
        )
    else:
        payloads = _read_allowed_directory_payloads_at_no_follow(
            directory_descriptor, expected_names, "calibration bundle"
        )
    if CALIBRATION_COMPLETION_NAME not in payloads:
        raise M3ProtocolError(
            "calibration bundle is INCOMPLETE: completion marker is absent"
        )
    if set(payloads) != expected_names:
        raise M3ProtocolError("calibration bundle roster changed")
    retained_protocol = _decode_json_object(
        payloads[PROTOCOL_NAME], "calibration protocol"
    )
    validate_protocol(retained_protocol)
    if (
        retained_protocol != protocol
        or retained_protocol["content_sha256"] != protocol["content_sha256"]
    ):
        raise M3ProtocolError("calibration protocol differs from current protocol")
    record = _decode_json_object(
        payloads[CALIBRATION_RECORD_NAME], "calibration receipt"
    )
    record_keys = {
        "schema",
        "run_id",
        "classification",
        "status",
        "completed_utc",
        "protocol_content_sha256",
        "protocol_file_sha256",
        "precision_selection_precursor",
        "measured_analysis",
        "state_arrays",
        "event_ledger",
        "executed_sources",
        "gate_thresholds_selected",
        "official_attempt_authorized",
        "staged_one_thousand_year_run_authorized",
        "claims",
        "interpretation",
        "content_sha256",
    }
    if (
        type(record) is not dict
        or set(record) != record_keys
        or record.get("schema") != CALIBRATION_SCHEMA
        or record.get("run_id") != RUN_ID
        or record.get("classification") != CLASSIFICATION
        or record.get("status") != "CALIBRATED_NON_AUTHORIZING"
        or record.get("protocol_content_sha256") != protocol["content_sha256"]
        or record.get("protocol_file_sha256")
        != hashlib.sha256(payloads[PROTOCOL_NAME]).hexdigest()
        or record.get("content_sha256")
        != _content_sha256(CALIBRATION_DOMAIN, record)
        or record.get("precision_selection_precursor")
        != PRECURSOR_CALIBRATION_CONTEXT
        or record.get("gate_thresholds_selected") is not False
        or record.get("official_attempt_authorized") is not False
        or record.get("staged_one_thousand_year_run_authorized") is not False
        or record.get("claims") != CLAIMS
        or any(record["claims"].values())
        or record.get("interpretation")
        != (
            "Calibration measures this bounded implementation only. It is "
            "not an acceptance threshold, qualification pass, or science "
            "result."
        )
    ):
        raise M3ProtocolError("calibration receipt is invalid or authorizing")
    arrays = _decode_npz_arrays_payload(
        payloads[ARRAY_ARCHIVE_NAME],
        "calibration states",
        expected_names=_M3_ARRAY_NAMES,
    )
    events = _decode_json_object(
        payloads[EVENT_LEDGER_NAME], "calibration events"
    )
    if (
        set(events) != {
            "schema",
            "candidate",
            "refinement",
            "claims",
            "content_sha256",
        }
        or events.get("schema")
        != "jx.long-horizon.m3-tracer-preflight.events.v1"
        or type(events.get("candidate")) is not list
        or type(events.get("refinement")) is not list
        or events.get("claims") != CLAIMS
        or events.get("content_sha256") != _content_sha256(EVENT_DOMAIN, events)
    ):
        raise M3ProtocolError("calibration event ledger is invalid")
    measured = record.get("measured_analysis")
    if type(measured) is not dict:
        raise M3ProtocolError("calibration measured analysis is absent")
    receipt_names = (
        "source_parent_custody",
        "split_resume",
        "event_accounting",
        "endpoint_state_custody",
        "runtime",
    )
    if any(name not in measured for name in receipt_names):
        raise M3ProtocolError("calibration analysis omits execution receipts")
    bundle = {
        "arrays": arrays,
        "candidate_events": events["candidate"],
        "refinement_events": events["refinement"],
        "receipts": {name: measured[name] for name in receipt_names},
    }
    _validate_numerical_bundle(bundle)
    recomputed = _EXACT_NUMERICAL_ANALYZER(
        bundle,
        state_ids=protocol["inputs"]["state_ids"],
        family_ids=protocol["inputs"]["family_ids"],
    )
    if _canonical_json(recomputed) != _canonical_json(measured):
        raise M3ProtocolError("calibration analysis does not recompute from arrays")
    permissive_thresholds = {
        "comparisons": {
            comparison: {
                metric: sys.float_info.max.hex()
                for metric in COMPARISON_METRICS
            }
            for comparison in COMPARISON_IDS
        }
    }
    calibration_gates = evaluate_gates(
        recomputed, permissive_thresholds, protocol
    )
    if not calibration_gates or not all(
        type(gate) is dict and gate.get("passed") is True
        for gate in calibration_gates.values()
    ):
        raise M3ProtocolError(
            "calibration structural or safety gates did not all pass"
        )
    for lane, prefix in (
        ("candidate", "candidate"),
        ("refinement", "refinement"),
    ):
        rows = events[lane]
        route_rows = [
            row for row in rows if row.get("record_kind") == "ROUTED_REJOIN"
        ]
        removal_rows = [
            row for row in rows if row.get("record_kind") == "TERMINAL_REMOVAL"
        ]
        if len(route_rows) + len(removal_rows) != len(rows):
            raise M3ProtocolError("calibration event ledger has an unknown row")
        recomputed_events = _event_accounting(
            {
                "position_AU": arrays[f"{prefix}_position_AU"],
                "velocity_AU_per_year": arrays[
                    f"{prefix}_velocity_AU_per_year"
                ],
                "lifecycle": arrays[f"{prefix}_lifecycle"],
                "route_records": route_rows,
                "removal_records": removal_rows,
            }
        )
        if recomputed_events != measured["event_accounting"][lane]:
            raise M3ProtocolError(
                "calibration event receipt does not recompute from retained ledgers"
            )
    state_record = record["state_arrays"]
    expected_array_hashes = {
        name: _array_sha256(value) for name, value in sorted(arrays.items())
    }
    if (
        type(state_record) is not dict
        or set(state_record)
        != {
            "path",
            "file_sha256",
            "array_content_sha256",
            "array_bundle_sha256",
        }
        or state_record.get("path") != ARRAY_ARCHIVE_NAME
        or state_record.get("file_sha256")
        != hashlib.sha256(payloads[ARRAY_ARCHIVE_NAME]).hexdigest()
        or state_record.get("array_content_sha256") != expected_array_hashes
        or state_record.get("array_bundle_sha256") != _arrays_sha256(arrays)
    ):
        raise M3ProtocolError("calibration array custody is invalid")
    event_record = record["event_ledger"]
    if (
        type(event_record) is not dict
        or set(event_record) != {"path", "file_sha256", "content_sha256"}
        or event_record.get("path") != EVENT_LEDGER_NAME
        or event_record.get("file_sha256")
        != hashlib.sha256(payloads[EVENT_LEDGER_NAME]).hexdigest()
        or event_record.get("content_sha256") != events["content_sha256"]
    ):
        raise M3ProtocolError("calibration event-file custody is invalid")
    archive_record = record["executed_sources"]
    if (
        type(archive_record) is not dict
        or set(archive_record)
        != {"path", "file_sha256", "runtime_identity_sha256"}
        or archive_record.get("path") != SOURCE_ARCHIVE_NAME
        or archive_record.get("file_sha256")
        != hashlib.sha256(payloads[SOURCE_ARCHIVE_NAME]).hexdigest()
        or archive_record.get("runtime_identity_sha256")
        != measured["runtime"]["runtime_identity_sha256"]
    ):
        raise M3ProtocolError("calibration executed-source custody is invalid")
    manifest = _validate_executed_source_archive_payload(
        payloads[SOURCE_ARCHIVE_NAME],
        protocol,
        expected_runtime_identity=measured["runtime"]["runtime_identity_end"],
        split_receipt=measured["split_resume"],
        expected_events={
            "candidate": events["candidate"],
            "refinement": events["refinement"],
        },
    )
    if (
        manifest.get("runtime_identity_sha256")
        != measured["runtime"]["runtime_identity_sha256"]
        or manifest.get("runtime_matches_execution_receipt") is not True
    ):
        raise M3ProtocolError("calibration source archive runtime is unbound")
    completion = _decode_json_object(
        payloads[CALIBRATION_COMPLETION_NAME], "calibration completion"
    )
    completion_keys = {
        "schema",
        "run_id",
        "status",
        "completed_utc",
        "protocol_content_sha256",
        "protocol_file_sha256",
        "calibration_content_sha256",
        "calibration_file_sha256",
        "states_file_sha256",
        "events_file_sha256",
        "executed_sources_file_sha256",
        "completion_marker_written_last",
        "official_attempt_authorized",
        "staged_one_thousand_year_run_authorized",
        "claims",
        "content_sha256",
    }
    if (
        type(completion) is not dict
        or set(completion) != completion_keys
        or completion.get("schema") != CALIBRATION_COMPLETION_SCHEMA
        or completion.get("run_id") != RUN_ID
        or completion.get("status") != "CALIBRATED_NON_AUTHORIZING"
        or completion.get("protocol_content_sha256")
        != protocol["content_sha256"]
        or completion.get("protocol_file_sha256")
        != hashlib.sha256(payloads[PROTOCOL_NAME]).hexdigest()
        or completion.get("calibration_content_sha256")
        != record["content_sha256"]
        or completion.get("calibration_file_sha256")
        != hashlib.sha256(payloads[CALIBRATION_RECORD_NAME]).hexdigest()
        or completion.get("states_file_sha256")
        != hashlib.sha256(payloads[ARRAY_ARCHIVE_NAME]).hexdigest()
        or completion.get("events_file_sha256")
        != hashlib.sha256(payloads[EVENT_LEDGER_NAME]).hexdigest()
        or completion.get("executed_sources_file_sha256")
        != hashlib.sha256(payloads[SOURCE_ARCHIVE_NAME]).hexdigest()
        or completion.get("completion_marker_written_last") is not True
        or completion.get("official_attempt_authorized") is not False
        or completion.get("staged_one_thousand_year_run_authorized") is not False
        or completion.get("claims") != CLAIMS
        or completion.get("content_sha256")
        != _content_sha256(CALIBRATION_COMPLETION_DOMAIN, completion)
    ):
        raise M3ProtocolError("calibration completion seal is invalid")
    return record


def _validate_calibration_payload_snapshot(
    payloads: Mapping[str, bytes], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate an owned six-file calibration snapshot before publication."""

    expected_names = {
        PROTOCOL_NAME,
        ARRAY_ARCHIVE_NAME,
        EVENT_LEDGER_NAME,
        SOURCE_ARCHIVE_NAME,
        CALIBRATION_RECORD_NAME,
        CALIBRATION_COMPLETION_NAME,
    }
    if set(payloads) != expected_names or any(
        type(name) is not str or type(payload) is not bytes
        for name, payload in payloads.items()
    ):
        raise M3ProtocolError("calibration payload snapshot roster changed")
    with tempfile.TemporaryDirectory(
        prefix="jx-m3-calibration-snapshot-"
    ) as temporary:
        root = Path(temporary)
        for name in sorted(expected_names):
            _write_bytes_create_only(root / name, payloads[name])
        return validate_calibration_bundle(root, protocol)


def _validate_retained_calibration_payloads(
    payloads: Mapping[str, bytes], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Revalidate the exact flat-name calibration copy retained in an attempt."""

    retained_names = {retained for _, retained in OFFICIAL_CALIBRATION_COPY_NAMES}
    if set(payloads) != retained_names:
        raise M3ProtocolError("retained calibration roster changed")
    with tempfile.TemporaryDirectory(prefix="jx-m3-retained-calibration-") as temporary:
        root = Path(temporary)
        for source_name, retained_name in OFFICIAL_CALIBRATION_COPY_NAMES:
            _write_bytes_create_only(root / source_name, payloads[retained_name])
        return validate_calibration_bundle(root, protocol)


def _validate_official_events(
    payload: bytes,
    arrays: Mapping[str, np.ndarray],
    measured: Mapping[str, Any],
) -> dict[str, Any]:
    events = _decode_json_object(payload, "official event ledger")
    if (
        set(events)
        != {"schema", "candidate", "refinement", "claims", "content_sha256"}
        or events.get("schema")
        != "jx.long-horizon.m3-tracer-preflight.events.v1"
        or type(events.get("candidate")) is not list
        or type(events.get("refinement")) is not list
        or events.get("claims") != CLAIMS
        or events.get("content_sha256") != _content_sha256(EVENT_DOMAIN, events)
    ):
        raise M3ProtocolError("official event ledger is invalid")
    for lane in ("candidate", "refinement"):
        rows = events[lane]
        route_rows = [
            row for row in rows if row.get("record_kind") == "ROUTED_REJOIN"
        ]
        removal_rows = [
            row for row in rows if row.get("record_kind") == "TERMINAL_REMOVAL"
        ]
        if len(route_rows) + len(removal_rows) != len(rows):
            raise M3ProtocolError("official event ledger has an unknown row")
        recomputed = _event_accounting(
            {
                "position_AU": arrays[f"{lane}_position_AU"],
                "velocity_AU_per_year": arrays[
                    f"{lane}_velocity_AU_per_year"
                ],
                "lifecycle": arrays[f"{lane}_lifecycle"],
                "route_records": route_rows,
                "removal_records": removal_rows,
            }
        )
        if recomputed != measured["event_accounting"][lane]:
            raise M3ProtocolError(
                "official event receipt does not recompute from retained evidence"
            )
    return events


def validate_official_attempt(
    attempt_directory: str | Path,
    protocol: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    *,
    require_completed: bool,
    directory_descriptor: int | None = None,
) -> dict[str, Any]:
    """Recompute and cross-link a complete or explicitly incomplete attempt."""

    if type(require_completed) is not bool:
        raise M3ProtocolError("completion requirement must be boolean")
    validate_protocol(protocol)
    attempt = Path(attempt_directory)
    validated_thresholds = validate_threshold_commitment(
        thresholds, protocol, expected_attempt_id=attempt.name
    )
    calibration_retained_names = {
        retained for _, retained in OFFICIAL_CALIBRATION_COPY_NAMES
    }
    allowed_names = {
        PROTOCOL_NAME,
        "THRESHOLDS.json",
        *calibration_retained_names,
        ARRAY_ARCHIVE_NAME,
        EVENT_LEDGER_NAME,
        SOURCE_ARCHIVE_NAME,
        RESULT_NAME,
        ARTIFACT_NAME,
        COMPLETION_NAME,
    }
    if directory_descriptor is None:
        payloads = _read_allowed_directory_payloads_no_follow(
            attempt, allowed_names, "official attempt"
        )
    else:
        payloads = _read_allowed_directory_payloads_at_no_follow(
            directory_descriptor, allowed_names, "official attempt"
        )
    required_base = {
        PROTOCOL_NAME,
        "THRESHOLDS.json",
        *calibration_retained_names,
        SOURCE_ARCHIVE_NAME,
        RESULT_NAME,
        ARTIFACT_NAME,
    }
    if not required_base <= set(payloads):
        raise M3ProtocolError("official attempt is missing required evidence")
    retained_protocol = _decode_json_object(
        payloads[PROTOCOL_NAME], "official protocol"
    )
    validate_protocol(retained_protocol)
    if retained_protocol != protocol:
        raise M3ProtocolError("official protocol differs from the execution protocol")
    retained_thresholds = validate_threshold_commitment(
        _decode_json_object(payloads["THRESHOLDS.json"], "official thresholds"),
        protocol,
        expected_attempt_id=attempt.name,
    )
    if retained_thresholds != validated_thresholds:
        raise M3ProtocolError("official thresholds differ from execution thresholds")

    retained_calibration_payloads = {
        retained: payloads[retained]
        for retained in calibration_retained_names
    }
    calibration = _validate_retained_calibration_payloads(
        retained_calibration_payloads, protocol
    )
    if (
        calibration.get("content_sha256")
        != thresholds.get("calibration_content_sha256")
    ):
        raise M3ProtocolError("retained calibration differs from threshold custody")
    calibration_rows = [
        {
            "path": retained,
            "size_bytes": len(payloads[retained]),
            "sha256": hashlib.sha256(payloads[retained]).hexdigest(),
        }
        for _, retained in OFFICIAL_CALIBRATION_COPY_NAMES
    ]

    result = _decode_json_object(payloads[RESULT_NAME], "official result")
    common_result_valid = bool(
        result.get("schema") == RESULT_SCHEMA
        and result.get("run_id") == RUN_ID
        and result.get("classification") == CLASSIFICATION
        and result.get("protocol_content_sha256") == protocol["content_sha256"]
        and result.get("threshold_content_sha256")
        == thresholds["content_sha256"]
        and result.get("calibration")
        == {
            "content_sha256": calibration["content_sha256"],
            "retained_files": calibration_rows,
        }
        and result.get("staged_one_thousand_year_run_authorized") is False
        and result.get("claims") == CLAIMS
        and not any(result.get("claims", {}).values())
        and result.get("content_sha256")
        == _content_sha256(RESULT_DOMAIN, result)
    )
    if not common_result_valid:
        raise M3ProtocolError("official result custody failed")

    successful_shape = "analysis" in result
    if successful_shape:
        success_keys = {
            "schema",
            "run_id",
            "classification",
            "status",
            "all_gates_passed",
            "completed_utc",
            "protocol_content_sha256",
            "threshold_content_sha256",
            "calibration",
            "analysis",
            "gates",
            "state_arrays",
            "event_ledger",
            "staged_one_thousand_year_run_authorized",
            "claims",
            "interpretation",
            "content_sha256",
        }
        expected_names = required_base | {ARRAY_ARCHIVE_NAME, EVENT_LEDGER_NAME}
        if require_completed:
            expected_names.add(COMPLETION_NAME)
        if (
            set(result) != success_keys
            or type(result.get("completed_utc")) is not str
            or not result["completed_utc"]
            or result.get("status") not in ("PASSED", "FAILED")
            or type(result.get("all_gates_passed")) is not bool
            or result.get("interpretation")
            != (
                "A pass qualifies only this bounded ten-year M3 tracer "
                "engineering preflight. It does not authorize a longer run "
                "or any scientific inference."
            )
        ):
            raise M3ProtocolError("official numerical result schema failed")
        arrays = _decode_npz_arrays_payload(
            payloads[ARRAY_ARCHIVE_NAME],
            "official states",
            expected_names=_M3_ARRAY_NAMES,
        )
        measured = result["analysis"]
        if type(measured) is not dict:
            raise M3ProtocolError("official measured analysis is absent")
        receipt_names = (
            "source_parent_custody",
            "split_resume",
            "event_accounting",
            "endpoint_state_custody",
            "runtime",
        )
        if any(name not in measured for name in receipt_names):
            raise M3ProtocolError("official execution receipts are incomplete")
        events = _validate_official_events(
            payloads[EVENT_LEDGER_NAME], arrays, measured
        )
        bundle = {
            "arrays": arrays,
            "candidate_events": events["candidate"],
            "refinement_events": events["refinement"],
            "receipts": {name: measured[name] for name in receipt_names},
        }
        _validate_numerical_bundle(bundle)
        recomputed = _EXACT_NUMERICAL_ANALYZER(
            bundle,
            state_ids=protocol["inputs"]["state_ids"],
            family_ids=protocol["inputs"]["family_ids"],
        )
        recomputed_gates = evaluate_gates(recomputed, thresholds, protocol)
        calibration_runtime_sha256 = _require_exact_calibration_runtime(
            calibration, recomputed["runtime"]
        )
        recomputed_gates["calibration_runtime_identity_exact"] = {
            "passed": True,
            "runtime_identity_sha256": calibration_runtime_sha256,
        }
        recomputed_pass = all(
            type(gate) is dict and gate.get("passed") is True
            for gate in recomputed_gates.values()
        )
        if (
            _canonical_json(recomputed) != _canonical_json(measured)
            or result.get("gates") != recomputed_gates
            or result.get("all_gates_passed") is not recomputed_pass
            or result.get("status")
            != ("PASSED" if recomputed_pass else "FAILED")
        ):
            raise M3ProtocolError("official analysis or gates do not recompute")
        state_record = result.get("state_arrays")
        array_hashes = {
            name: _array_sha256(value) for name, value in sorted(arrays.items())
        }
        if (
            type(state_record) is not dict
            or set(state_record)
            != {"path", "sha256", "bundle_content_sha256", "array_content_sha256"}
            or state_record.get("path") != ARRAY_ARCHIVE_NAME
            or state_record.get("sha256")
            != hashlib.sha256(payloads[ARRAY_ARCHIVE_NAME]).hexdigest()
            or state_record.get("bundle_content_sha256") != _arrays_sha256(arrays)
            or state_record.get("array_content_sha256") != array_hashes
        ):
            raise M3ProtocolError("official state-array custody failed")
        event_record = result.get("event_ledger")
        if (
            type(event_record) is not dict
            or set(event_record) != {"path", "sha256", "content_sha256"}
            or event_record.get("path") != EVENT_LEDGER_NAME
            or event_record.get("sha256")
            != hashlib.sha256(payloads[EVENT_LEDGER_NAME]).hexdigest()
            or event_record.get("content_sha256") != events["content_sha256"]
        ):
            raise M3ProtocolError("official event-file custody failed")
        _validate_executed_source_archive_payload(
            payloads[SOURCE_ARCHIVE_NAME],
            protocol,
            expected_runtime_identity=measured["runtime"]["runtime_identity_end"],
            split_receipt=measured["split_resume"],
            expected_events={
                "candidate": events["candidate"],
                "refinement": events["refinement"],
            },
        )
        retained_order = [
            PROTOCOL_NAME,
            "THRESHOLDS.json",
            *(retained for _, retained in OFFICIAL_CALIBRATION_COPY_NAMES),
            ARRAY_ARCHIVE_NAME,
            EVENT_LEDGER_NAME,
            SOURCE_ARCHIVE_NAME,
            RESULT_NAME,
        ]
    else:
        failure_keys = {
            "schema",
            "run_id",
            "classification",
            "status",
            "all_gates_passed",
            "protocol_content_sha256",
            "threshold_content_sha256",
            "calibration",
            "failure",
            "official_attempt_authorized_only_for_this_failed_attempt",
            "staged_one_thousand_year_run_authorized",
            "claims",
            "content_sha256",
        }
        expected_names = set(required_base)
        if require_completed:
            expected_names.add(COMPLETION_NAME)
        failure = result.get("failure")
        if (
            set(result) != failure_keys
            or result.get("status") not in ("FAILED", "INCOMPLETE")
            or result.get("all_gates_passed") is not False
            or result.get("official_attempt_authorized_only_for_this_failed_attempt")
            is not True
            or type(failure) is not dict
            or set(failure) != {"type", "message"}
            or type(failure.get("type")) is not str
            or not failure["type"]
            or type(failure.get("message")) is not str
        ):
            raise M3ProtocolError("official failure result schema failed")
        _validate_executed_source_archive_payload(
            payloads[SOURCE_ARCHIVE_NAME],
            protocol,
            expected_runtime_identity=None,
            split_receipt=None,
            expected_events=None,
        )
        retained_order = [
            PROTOCOL_NAME,
            "THRESHOLDS.json",
            *(retained for _, retained in OFFICIAL_CALIBRATION_COPY_NAMES),
            SOURCE_ARCHIVE_NAME,
            RESULT_NAME,
        ]
    if set(payloads) != expected_names:
        raise M3ProtocolError("official attempt status-dependent roster changed")

    artifacts = _decode_json_object(payloads[ARTIFACT_NAME], "official artifacts")
    expected_artifact_rows = [
        {
            "path": name,
            "size_bytes": len(payloads[name]),
            "sha256": hashlib.sha256(payloads[name]).hexdigest(),
        }
        for name in retained_order
    ]
    if (
        set(artifacts)
        != {
            "schema",
            "run_id",
            "artifacts",
            "completion_marker_excluded_because_written_last",
            "claims",
            "content_sha256",
        }
        or artifacts.get("schema") != ARTIFACT_SCHEMA
        or artifacts.get("run_id") != RUN_ID
        or artifacts.get("artifacts") != expected_artifact_rows
        or artifacts.get("completion_marker_excluded_because_written_last")
        is not True
        or artifacts.get("claims") != CLAIMS
        or artifacts.get("content_sha256")
        != _content_sha256(ARTIFACT_DOMAIN, artifacts)
    ):
        raise M3ProtocolError("official artifact inventory custody failed")

    if not require_completed:
        return {
            "status": "INCOMPLETE",
            "result_status": result["status"],
            "completion_present": False,
            "result_content_sha256": result["content_sha256"],
            "artifacts_content_sha256": artifacts["content_sha256"],
        }
    completion = _decode_json_object(
        payloads[COMPLETION_NAME], "official completion"
    )
    completion_keys = {
        "schema",
        "run_id",
        "status",
        "completed_utc",
        "protocol_content_sha256",
        "protocol_file_sha256",
        "result_content_sha256",
        "result_file_sha256",
        "artifacts_content_sha256",
        "artifacts_file_sha256",
        "completion_marker_written_last",
        "claims",
        "content_sha256",
    }
    if (
        set(completion) != completion_keys
        or completion.get("schema") != COMPLETION_SCHEMA
        or completion.get("run_id") != RUN_ID
        or completion.get("status") != result["status"]
        or type(completion.get("completed_utc")) is not str
        or not completion["completed_utc"]
        or completion.get("protocol_content_sha256")
        != protocol["content_sha256"]
        or completion.get("protocol_file_sha256")
        != hashlib.sha256(payloads[PROTOCOL_NAME]).hexdigest()
        or completion.get("result_content_sha256")
        != result["content_sha256"]
        or completion.get("result_file_sha256")
        != hashlib.sha256(payloads[RESULT_NAME]).hexdigest()
        or completion.get("artifacts_content_sha256")
        != artifacts["content_sha256"]
        or completion.get("artifacts_file_sha256")
        != hashlib.sha256(payloads[ARTIFACT_NAME]).hexdigest()
        or completion.get("completion_marker_written_last") is not True
        or completion.get("claims") != CLAIMS
        or completion.get("content_sha256")
        != _content_sha256(COMPLETION_DOMAIN, completion)
    ):
        raise M3ProtocolError("official completion seal custody failed")
    return {
        "status": "COMPLETED",
        "result_status": result["status"],
        "completion_present": True,
        "result_content_sha256": result["content_sha256"],
        "artifacts_content_sha256": artifacts["content_sha256"],
        "completion_content_sha256": completion["content_sha256"],
    }


def _artifact_inventory(
    attempt_directory: Path,
    names: Sequence[str],
    *,
    directory_descriptor: int | None = None,
) -> dict[str, Any]:
    artifacts = []
    for name in names:
        if directory_descriptor is None:
            path = attempt_directory / name
            if path.is_symlink() or not path.is_file():
                raise M3ProtocolError(f"retained artifact is absent: {name}")
            payload, metadata = _read_absolute_file_no_follow(
                path, f"retained artifact {name}"
            )
        else:
            payload, metadata = _read_regular_file_at_no_follow(
                directory_descriptor, Path(name), f"retained artifact {name}"
            )
        artifacts.append(
            {
                "path": name,
                "size_bytes": metadata.st_size,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return _content_record(
        {
            "schema": ARTIFACT_SCHEMA,
            "run_id": RUN_ID,
            "artifacts": artifacts,
            "completion_marker_excluded_because_written_last": True,
            "claims": dict(CLAIMS),
        },
        ARTIFACT_DOMAIN,
    )


def _completion_record(
    attempt_directory: Path,
    protocol: Mapping[str, Any],
    result: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    *,
    directory_descriptor: int | None = None,
) -> dict[str, Any]:
    if directory_descriptor is None:
        protocol_payload, _ = _read_absolute_file_no_follow(
            attempt_directory / PROTOCOL_NAME, "official protocol"
        )
        result_payload, _ = _read_absolute_file_no_follow(
            attempt_directory / RESULT_NAME, "official result"
        )
        artifacts_payload, _ = _read_absolute_file_no_follow(
            attempt_directory / ARTIFACT_NAME, "official artifacts"
        )
    else:
        protocol_payload, _ = _read_regular_file_at_no_follow(
            directory_descriptor, Path(PROTOCOL_NAME), "official protocol"
        )
        result_payload, _ = _read_regular_file_at_no_follow(
            directory_descriptor, Path(RESULT_NAME), "official result"
        )
        artifacts_payload, _ = _read_regular_file_at_no_follow(
            directory_descriptor, Path(ARTIFACT_NAME), "official artifacts"
        )
    return _content_record(
        {
            "schema": COMPLETION_SCHEMA,
            "run_id": RUN_ID,
            "status": result["status"],
            "completed_utc": _utc_now(),
            "protocol_content_sha256": protocol["content_sha256"],
            "protocol_file_sha256": hashlib.sha256(
                protocol_payload
            ).hexdigest(),
            "result_content_sha256": result["content_sha256"],
            "result_file_sha256": hashlib.sha256(result_payload).hexdigest(),
            "artifacts_content_sha256": artifacts["content_sha256"],
            "artifacts_file_sha256": hashlib.sha256(
                artifacts_payload
            ).hexdigest(),
            "completion_marker_written_last": True,
            "claims": dict(CLAIMS),
        },
        COMPLETION_DOMAIN,
    )


def _official_attempt_directory_open(
    project_root: Path, attempt_directory: Path
) -> tuple[Path, int]:
    canonical_root = Path(
        os.path.abspath(os.fspath(project_root / DEFAULT_CANONICAL_ROOT))
    )
    target = Path(os.path.abspath(os.fspath(attempt_directory)))
    if target.parent != canonical_root or not target.name.startswith("attempt-"):
        raise M3ProtocolError(
            "official output must be a unique attempt-* child of the canonical root"
        )
    parent_descriptor = _open_absolute_directory_no_follow(
        canonical_root.parent, "official canonical-root parent"
    )
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        try:
            canonical_descriptor = _create_directory_at_no_replace_open(
                parent_descriptor,
                canonical_root.name,
                "official canonical root",
            )
        except FileExistsError:
            try:
                canonical_descriptor = os.open(
                    canonical_root.name,
                    directory_flags,
                    dir_fd=parent_descriptor,
                )
            except OSError as exc:
                raise M3ProtocolError(
                    "official canonical root is absent or linked"
                ) from exc
    finally:
        os.close(parent_descriptor)
    try:
        try:
            attempt_descriptor = _create_directory_at_no_replace_open(
                canonical_descriptor,
                target.name,
                "official attempt directory",
            )
        except FileExistsError:
            raise M3ProtocolError(
                "official attempt directory must be a new path"
            ) from None
    finally:
        os.close(canonical_descriptor)
    return target, attempt_descriptor


def _official_attempt_directory(
    project_root: Path, attempt_directory: Path
) -> Path:
    target, descriptor = _official_attempt_directory_open(
        project_root, attempt_directory
    )
    os.close(descriptor)
    return target


def _failure_result(
    *,
    protocol: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    calibration: Mapping[str, Any],
    calibration_files: Sequence[Mapping[str, Any]],
    error: BaseException,
) -> dict[str, Any]:
    status = "INCOMPLETE" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "FAILED"
    return _content_record(
        {
            "schema": RESULT_SCHEMA,
            "run_id": RUN_ID,
            "classification": CLASSIFICATION,
            "status": status,
            "all_gates_passed": False,
            "protocol_content_sha256": protocol["content_sha256"],
            "threshold_content_sha256": thresholds["content_sha256"],
            "calibration": {
                "content_sha256": calibration["content_sha256"],
                "retained_files": list(calibration_files),
            },
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "official_attempt_authorized_only_for_this_failed_attempt": True,
            "staged_one_thousand_year_run_authorized": False,
            "claims": dict(CLAIMS),
        },
        RESULT_DOMAIN,
    )


def _require_exact_calibration_runtime(
    calibration: Mapping[str, Any], official_runtime: Mapping[str, Any]
) -> str:
    """Require official execution on the exact reviewed calibration runtime."""

    measured = calibration.get("measured_analysis")
    calibration_runtime = (
        measured.get("runtime") if type(measured) is dict else None
    )
    if type(calibration_runtime) is not dict or type(official_runtime) is not dict:
        raise M3ProtocolError("calibration or official runtime receipt is absent")
    required = {
        "runtime_identity_start",
        "runtime_identity_end",
        "runtime_identity_sha256",
    }
    if any(key not in calibration_runtime for key in required) or any(
        key not in official_runtime for key in required
    ):
        raise M3ProtocolError("calibration or official runtime receipt is incomplete")
    calibration_start = calibration_runtime["runtime_identity_start"]
    official_start = official_runtime["runtime_identity_start"]
    calibration_digest = calibration_runtime["runtime_identity_sha256"]
    official_digest = official_runtime["runtime_identity_sha256"]
    if (
        type(calibration_start) is not dict
        or type(official_start) is not dict
        or calibration_runtime["runtime_identity_end"] != calibration_start
        or official_runtime["runtime_identity_end"] != official_start
        or calibration_digest
        != hashlib.sha256(
            RUNTIME_IDENTITY_DOMAIN + _canonical_json(calibration_start)
        ).hexdigest()
        or official_digest
        != hashlib.sha256(
            RUNTIME_IDENTITY_DOMAIN + _canonical_json(official_start)
        ).hexdigest()
        or official_start != calibration_start
        or official_digest != calibration_digest
    ):
        raise M3ProtocolError(
            "official numerical runtime differs from the reviewed calibration "
            "runtime"
        )
    return official_digest


def _run_official_in_open_attempt(
    *,
    root: Path,
    attempt: Path,
    attempt_descriptor: int,
    protocol: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    threshold_payload: bytes,
    calibration_root: Path,
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish an official attempt only through its retained directory FD."""

    _write_bytes_at_create_only(
        attempt_descriptor, PROTOCOL_NAME, _json_bytes(protocol)
    )
    _write_bytes_at_create_only(
        attempt_descriptor, "THRESHOLDS.json", threshold_payload
    )
    calibration_files: list[dict[str, Any]] = []
    retained_calibration_payloads: dict[str, bytes] = {}
    for source_name, retained_name in OFFICIAL_CALIBRATION_COPY_NAMES:
        payload, metadata = _read_absolute_file_no_follow(
            calibration_root / source_name,
            f"calibration artifact {source_name}",
        )
        _write_bytes_at_create_only(attempt_descriptor, retained_name, payload)
        retained_calibration_payloads[retained_name] = payload
        calibration_files.append(
            {
                "path": retained_name,
                "size_bytes": metadata.st_size,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    copied_calibration = _validate_retained_calibration_payloads(
        retained_calibration_payloads, protocol
    )
    if (
        copied_calibration["content_sha256"]
        != calibration["content_sha256"]
        or copied_calibration["content_sha256"]
        != thresholds["calibration_content_sha256"]
    ):
        raise M3ProtocolError(
            "retained calibration snapshot differs from the reviewed bundle"
        )
    calibration = copied_calibration
    result: dict[str, Any]
    archive_runtime_identity: Mapping[str, Any] | None = None
    retained = [
        PROTOCOL_NAME,
        "THRESHOLDS.json",
        *(row["path"] for row in calibration_files),
    ]
    with tempfile.TemporaryDirectory(prefix="jx-m3-official-") as temporary:
        work = Path(temporary)
        try:
            _, prepared_profiles, source = _input_record(root)
            bundle = _EXACT_NUMERICAL_EXECUTOR(
                project_root=root,
                prepared_profiles=prepared_profiles,
                source=source,
                protocol=protocol,
                work_directory=work,
            )
            _validate_numerical_bundle(bundle)
            archive_runtime_identity = bundle["receipts"]["runtime"][
                "runtime_identity_end"
            ]
            official_runtime = bundle["receipts"]["runtime"]
            calibration_runtime_sha256 = _require_exact_calibration_runtime(
                calibration, official_runtime
            )
            analysis = _EXACT_NUMERICAL_ANALYZER(
                bundle,
                state_ids=protocol["inputs"]["state_ids"],
                family_ids=protocol["inputs"]["family_ids"],
            )
            gates = evaluate_gates(analysis, thresholds, protocol)
            gates["calibration_runtime_identity_exact"] = {
                "passed": True,
                "runtime_identity_sha256": calibration_runtime_sha256,
            }
            all_passed = all(bool(row["passed"]) for row in gates.values())
            arrays = bundle["arrays"]
            array_payload = _npz_bytes(arrays)
            _write_bytes_at_create_only(
                attempt_descriptor, ARRAY_ARCHIVE_NAME, array_payload
            )
            retained.append(ARRAY_ARCHIVE_NAME)
            event_ledger = _content_record(
                {
                    "schema": "jx.long-horizon.m3-tracer-preflight.events.v1",
                    "candidate": bundle["candidate_events"],
                    "refinement": bundle["refinement_events"],
                    "claims": dict(CLAIMS),
                },
                EVENT_DOMAIN,
            )
            event_payload = _json_bytes(event_ledger)
            _write_bytes_at_create_only(
                attempt_descriptor, EVENT_LEDGER_NAME, event_payload
            )
            retained.append(EVENT_LEDGER_NAME)
            result = _content_record(
                {
                    "schema": RESULT_SCHEMA,
                    "run_id": RUN_ID,
                    "classification": CLASSIFICATION,
                    "status": "PASSED" if all_passed else "FAILED",
                    "all_gates_passed": all_passed,
                    "completed_utc": _utc_now(),
                    "protocol_content_sha256": protocol["content_sha256"],
                    "threshold_content_sha256": thresholds["content_sha256"],
                    "calibration": {
                        "content_sha256": calibration["content_sha256"],
                        "retained_files": calibration_files,
                    },
                    "analysis": analysis,
                    "gates": gates,
                    "state_arrays": {
                        "path": ARRAY_ARCHIVE_NAME,
                        "sha256": hashlib.sha256(array_payload).hexdigest(),
                        "bundle_content_sha256": _arrays_sha256(arrays),
                        "array_content_sha256": {
                            name: _array_sha256(value)
                            for name, value in sorted(arrays.items())
                        },
                    },
                    "event_ledger": {
                        "path": EVENT_LEDGER_NAME,
                        "sha256": hashlib.sha256(event_payload).hexdigest(),
                        "content_sha256": event_ledger["content_sha256"],
                    },
                    "staged_one_thousand_year_run_authorized": False,
                    "claims": dict(CLAIMS),
                    "interpretation": (
                        "A pass qualifies only this bounded ten-year M3 tracer "
                        "engineering preflight. It does not authorize a longer run "
                        "or any scientific inference."
                    ),
                },
                RESULT_DOMAIN,
            )
        except BaseException as error:
            archive_runtime_identity = None
            result = _failure_result(
                protocol=protocol,
                thresholds=thresholds,
                calibration=calibration,
                calibration_files=calibration_files,
                error=error,
            )
        archive_payload = _create_executed_archive(
            project_root=root,
            work_directory=work,
            expected_code_identity=protocol["code_identity"],
            expected_runtime_identity=archive_runtime_identity,
        )
        _write_bytes_at_create_only(
            attempt_descriptor, SOURCE_ARCHIVE_NAME, archive_payload
        )
        retained.append(SOURCE_ARCHIVE_NAME)
    _write_bytes_at_create_only(
        attempt_descriptor, RESULT_NAME, _json_bytes(result)
    )
    retained.append(RESULT_NAME)
    artifacts = _artifact_inventory(
        attempt, retained, directory_descriptor=attempt_descriptor
    )
    _write_bytes_at_create_only(
        attempt_descriptor, ARTIFACT_NAME, _json_bytes(artifacts)
    )
    incomplete_verification = validate_official_attempt(
        attempt,
        protocol,
        thresholds,
        require_completed=False,
        directory_descriptor=attempt_descriptor,
    )
    if (
        incomplete_verification.get("status") != "INCOMPLETE"
        or incomplete_verification.get("completion_present") is not False
        or incomplete_verification.get("result_content_sha256")
        != result["content_sha256"]
        or incomplete_verification.get("artifacts_content_sha256")
        != artifacts["content_sha256"]
    ):
        raise M3ProtocolError("official attempt is not ready for its final seal")
    completion = _completion_record(
        attempt,
        protocol,
        result,
        artifacts,
        directory_descriptor=attempt_descriptor,
    )
    _require_directory_path_identity(
        attempt,
        attempt_descriptor,
        "official attempt before completion",
    )
    _write_bytes_at_create_only(
        attempt_descriptor, COMPLETION_NAME, _json_bytes(completion)
    )
    completed_verification = validate_official_attempt(
        attempt,
        protocol,
        thresholds,
        require_completed=True,
        directory_descriptor=attempt_descriptor,
    )
    if (
        completed_verification.get("status") != "COMPLETED"
        or completed_verification.get("completion_present") is not True
        or completed_verification.get("completion_content_sha256")
        != completion["content_sha256"]
    ):
        raise M3ProtocolError("official completion seal did not reverify")
    _require_directory_path_identity(
        attempt,
        attempt_descriptor,
        "official attempt after completion",
    )
    return result


def run_official(
    *,
    project_root: str | Path,
    threshold_path: str | Path,
    calibration_path: str | Path,
    attempt_directory: str | Path,
) -> dict[str, Any]:
    """Run one content-bound attempt and retain success or failure evidence."""

    root = Path(project_root).resolve()
    protocol = build_protocol(root)
    validate_protocol(protocol)
    threshold_payload, _threshold_metadata = _read_absolute_file_no_follow(
        Path(threshold_path), "reviewed threshold commitment"
    )
    thresholds = validate_threshold_commitment(
        _decode_json_object(threshold_payload, "reviewed threshold commitment"),
        protocol,
        expected_attempt_id=Path(attempt_directory).name,
    )
    calibration_root = Path(calibration_path)
    calibration = validate_calibration_bundle(calibration_root, protocol)
    if calibration["content_sha256"] != thresholds["calibration_content_sha256"]:
        raise M3ProtocolError(
            "reviewed thresholds do not identify the supplied calibration"
        )
    attempt, attempt_descriptor = _official_attempt_directory_open(
        root, Path(attempt_directory)
    )
    try:
        return _run_official_in_open_attempt(
            root=root,
            attempt=attempt,
            attempt_descriptor=attempt_descriptor,
            protocol=protocol,
            thresholds=thresholds,
            threshold_payload=threshold_payload,
            calibration_root=calibration_root,
            calibration=calibration,
        )
    finally:
        os.close(attempt_descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare, calibrate, or execute the bounded M3 tracer preflight."
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--calibrate", action="store_true")
    modes.add_argument("--official", action="store_true")
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--attempt-directory", type=Path)
    args = parser.parse_args(argv)
    if args.prepare_only:
        protocol = build_protocol(args.project_root)
        validate_protocol(protocol)
        if args.output is None:
            parser.error("--prepare-only requires --output")
        _write_json_create_only(args.output.resolve(), protocol)
        print(json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.calibrate:
        if args.output is None:
            parser.error("--calibrate requires --output")
        record = run_calibration(
            project_root=args.project_root,
            output_path=args.output,
        )
        print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if (
        args.thresholds is None
        or args.calibration is None
        or args.attempt_directory is None
    ):
        parser.error(
            "--official requires --thresholds, --calibration, and "
            "--attempt-directory"
        )
    result = run_official(
        project_root=args.project_root,
        threshold_path=args.thresholds,
        calibration_path=args.calibration,
        attempt_directory=args.attempt_directory,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if result["status"] == "PASSED":
        return 0
    return 2 if result["status"] == "FAILED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
