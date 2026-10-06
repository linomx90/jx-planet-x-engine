"""Real-epoch, outcome-blind, large-population JX source/control screen."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import math
import multiprocessing
import os
import re
import statistics
import tempfile
import time
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .encounter_tail import (
    _active_state,
    _bootstrap_ci,
    _canonical_bytes,
    _cartesian_state_is_finite,
    _simulation_digest,
)
from .ensemble_validation import wasserstein_1d
from .population_scale import (
    _by_name,
    _canonical_sha256,
    _fraction,
    _load_rows,
    _relative_state,
    _sha256_file,
)


CONTRACT_SCHEMA = "jx-de441-population-contract/v1"
RESULT_SCHEMA = "jx-de441-population-result/v2"
JOB_SCHEMA = "jx-de441-population-job/v2"
BLOCK_SCHEMA = "jx-de441-population-block/v2"
CHECKPOINT_SCHEMA = "jx-de441-population-checkpoint/v2"
POPULATION_COLUMNS = (
    "block_index",
    "local_index",
    "logical_id",
    "a0_AU",
    "q0_AU",
    "e0",
    "i0_deg",
    "Omega0_rad",
    "omega0_rad",
    "M0_rad",
)
TRACER_COLUMNS = POPULATION_COLUMNS + (
    "minimum_sampled_q_AU",
    "sampled_injection",
    "first_sampled_low_q_year",
    "ever_unbound_at_sample",
    "first_unbound_year",
    "bound_final",
    "final_q_AU",
    "final_i_deg",
    "minimum_sampled_neptune_distance_AU",
    "minimum_sampled_neptune_hill_ratio",
    "minimum_sampled_source_distance_AU",
    "minimum_sampled_source_hill_ratio",
)
COMMON_NAMES = ("Sun", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto")
SOURCE_NAME = "P9_BB21_idx9118"

_HASH_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_CHECKPOINT_PATTERN = re.compile(r"checkpoint_(\d{3,})\.(bin|json)\Z")
_FILE_IDENTITY_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
_JOB_CORE_FIELDS = (
    "contract_sha256",
    "arm",
    "stage",
    "state_sha256",
    "population_sha256",
    "block_index",
    "local_count",
    "dt_years",
    "sample_years",
    "duration_years",
    "aggregate_years",
    "checkpoint_years",
    "total_samples",
    "aggregate_stride_samples",
    "checkpoint_stride_samples",
    "q_lower",
)
_TRACKER_FIELDS = {
    "minimum_q",
    "first_low_q_year",
    "ever_unbound",
    "first_unbound_year",
    "minimum_neptune_distance_AU",
    "minimum_neptune_hill_ratio",
    "minimum_source_distance_AU",
    "minimum_source_hill_ratio",
    "neptune_index",
    "source_index",
    "neptune_hill_radius_AU",
    "source_hill_radius_AU",
    "sample_count",
    "timeseries",
}
_TIMESERIES_FIELDS = {
    "time_year",
    "bound",
    "survival_fraction",
    "cumulative_sampled_injections",
    "cumulative_sampled_injection_fraction",
    "mean_current_bound_q_AU",
    "current_bound_inclination_width_deg",
}
_BLOCK_FIELDS = {
    "schema",
    "job_sha256",
    "block_content_sha256",
    "arm",
    "stage",
    "block_index",
    "tracers",
    "particles",
    "N_active",
    "dt_years",
    "duration_years",
    "sample_years",
    "sample_count_including_t0",
    "checkpoint_epochs_including_t0",
    "restart_replay_state_hash_exact",
    "cartesian_state_finite",
    "active_endpoint_state",
    "active_endpoint_state_sha256",
    "tracer_csv",
    "tracer_csv_sha256",
    "summary_json",
    "timeseries",
    "elapsed_seconds",
}

FileIdentity = tuple[int, int, int, int, int]
_POPULATION_CACHE: dict[str, tuple[FileIdentity, dict[int, list[dict[str, Any]]]]] = {}
_STATE_CACHE: dict[str, tuple[FileIdentity, list[dict[str, str]]]] = {}
_FILE_SHA256_CACHE: dict[str, tuple[FileIdentity, str]] = {}


def _resolve(contract_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (contract_path.parent / path).resolve()


def _file_identity(path: Path) -> FileIdentity:
    status = path.stat()
    return tuple(int(getattr(status, field)) for field in _FILE_IDENTITY_FIELDS)  # type: ignore[return-value]


def _stable_read_bytes(path: Path) -> tuple[bytes, FileIdentity]:
    """Read one immutable view and reject replacement or mutation during the read."""
    before = _file_identity(path)
    data = path.read_bytes()
    after = _file_identity(path)
    if before != after or len(data) != after[2]:
        raise RuntimeError(f"file changed while it was being read: {path}")
    return data, after


def _no_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _decode_json_object(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_no_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _cached_sha256(path: Path) -> str:
    """Hash a locked input once per stable filesystem identity, never by path alone."""
    key = str(path.resolve())
    identity = _file_identity(path)
    cached = _FILE_SHA256_CACHE.get(key)
    if cached is not None and cached[0] == identity:
        return cached[1]
    data, observed_identity = _stable_read_bytes(path)
    digest = hashlib.sha256(data).hexdigest()
    _FILE_SHA256_CACHE[key] = (observed_identity, digest)
    return digest


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _HASH_PATTERN.fullmatch(value) is not None


def _strict_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _strict_number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _strict_float(value: Any, label: str, *, minimum: float | None = None) -> float:
    if type(value) is not float:
        raise ValueError(f"{label} must be a JSON floating-point number")
    return _strict_number(value, label, minimum=minimum)


def _strict_csv_int(value: Any, label: str) -> int:
    if not isinstance(value, str) or not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise ValueError(f"{label} must be a canonical non-negative integer")
    return int(value)


def _strict_csv_float(value: Any, label: str, *, optional: bool = False) -> float | None:
    if optional and value == "":
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a finite decimal")
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"{label} must be a finite decimal") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite decimal")
    return result


def _time_is_sample_epoch(value: float, sample_years: float) -> bool:
    ratio = value / sample_years
    return math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=2e-10)


def _write_bytes_create_only(path: Path, data: bytes) -> None:
    """Atomically publish bytes without ever replacing an existing custody artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_create_only(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _write_bytes_create_only(path, payload.encode("utf-8"))


def _write_csv_create_only(path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...]) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _write_bytes_create_only(path, output.getvalue().encode("utf-8"))


@contextmanager
def _block_lock(directory: Path) -> Iterator[None]:
    """Hold one non-blocking advisory writer lock for a block directory."""
    lock_path = directory / ".block.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another worker owns block directory: {directory}") from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _positive_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _integer_ratio(numerator: float, denominator: float, label: str) -> int:
    ratio = numerator / denominator
    rounded = round(ratio)
    if not math.isclose(ratio, rounded, rel_tol=0.0, abs_tol=2e-10):
        raise ValueError(f"{label} must be an integer ratio")
    return int(rounded)


def _job_digest_payload(job: Mapping[str, Any]) -> dict[str, Any]:
    """Return the path-independent job identity derivable from the locked contract."""
    return {
        "schema": JOB_SCHEMA,
        **{field: job[field] for field in _JOB_CORE_FIELDS},
    }


def _job_sha256(job: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(_job_digest_payload(job))).hexdigest()


def _validate_job(job: Mapping[str, Any]) -> None:
    expected_fields = {*_JOB_CORE_FIELDS, "state_csv", "population_csv", "job_sha256", "directory"}
    if set(job) != expected_fields:
        raise ValueError(
            f"job fields differ from the {JOB_SCHEMA} contract: "
            f"missing={sorted(expected_fields - set(job))}, extra={sorted(set(job) - expected_fields)}"
        )
    for field in ("contract_sha256", "state_sha256", "population_sha256", "job_sha256"):
        if not _is_sha256(job[field]):
            raise ValueError(f"job {field} must be a lowercase SHA-256")
    if job["arm"] not in {"control", "source"}:
        raise ValueError("job arm must be control or source")
    if not isinstance(job["stage"], str) or not job["stage"]:
        raise ValueError("job stage must be a non-empty string")
    for field in ("state_csv", "population_csv", "directory"):
        if not isinstance(job[field], str) or not job[field]:
            raise ValueError(f"job {field} must be a non-empty path string")
    _strict_int(job["block_index"], "job block_index", minimum=0)
    _strict_int(job["local_count"], "job local_count", minimum=1)
    _strict_int(job["total_samples"], "job total_samples", minimum=1)
    _strict_int(job["aggregate_stride_samples"], "job aggregate stride", minimum=1)
    _strict_int(job["checkpoint_stride_samples"], "job checkpoint stride", minimum=1)
    for field in (
        "dt_years",
        "sample_years",
        "duration_years",
        "aggregate_years",
        "checkpoint_years",
        "q_lower",
    ):
        _strict_float(job[field], f"job {field}", minimum=0.0)
    if float(job["dt_years"]) <= 0.0 or float(job["sample_years"]) <= 0.0:
        raise ValueError("job integration and sample steps must be positive")
    if float(job["aggregate_years"]) <= 0.0 or float(job["checkpoint_years"]) <= 0.0:
        raise ValueError("job aggregate and checkpoint cadences must be positive")
    if job["total_samples"] != _integer_ratio(
        float(job["duration_years"]), float(job["sample_years"]), "job duration/sample"
    ):
        raise ValueError("job total_samples does not match duration/sample")
    if job["aggregate_stride_samples"] != _integer_ratio(
        float(job["aggregate_years"]), float(job["sample_years"]), "job aggregate/sample"
    ):
        raise ValueError("job aggregate stride does not match its cadence")
    if job["checkpoint_stride_samples"] != _integer_ratio(
        float(job["checkpoint_years"]), float(job["sample_years"]), "job checkpoint/sample"
    ):
        raise ValueError("job checkpoint stride does not match its cadence")
    if job["total_samples"] % job["checkpoint_stride_samples"] != 0:
        raise ValueError("job endpoint must be a checkpoint epoch")
    observed = _job_sha256(job)
    if job["job_sha256"] != observed:
        raise ValueError(f"job_sha256 mismatch: expected {observed}")


def _load_population(path: Path) -> dict[int, list[dict[str, Any]]]:
    key = str(path.resolve())
    identity = _file_identity(path)
    cached = _POPULATION_CACHE.get(key)
    if cached is not None and cached[0] == identity:
        return cached[1]
    blocks: dict[int, list[dict[str, Any]]] = {}
    data, observed_identity = _stable_read_bytes(path)
    with io.StringIO(data.decode("utf-8"), newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != POPULATION_COLUMNS:
            raise ValueError(f"population CSV must have columns {POPULATION_COLUMNS}")
        for raw in reader:
            block = _strict_csv_int(raw["block_index"], "population block_index")
            local = _strict_csv_int(raw["local_index"], "population local_index")
            row = {
                "block_index": block,
                "local_index": local,
                "logical_id": raw["logical_id"],
                **{key: float(raw[key]) for key in POPULATION_COLUMNS[3:]},
            }
            numbers = [row[key] for key in POPULATION_COLUMNS[3:]]
            if not all(math.isfinite(value) for value in numbers):
                raise ValueError("population contains a non-finite element")
            if not (
                row["a0_AU"] > row["q0_AU"] > 30.0
                and math.isclose(row["e0"], 1.0 - row["q0_AU"] / row["a0_AU"], abs_tol=2e-15)
                and 0.0 <= row["i0_deg"] <= 40.0
                and all(0.0 <= row[key] < 2.0 * math.pi for key in ("Omega0_rad", "omega0_rad", "M0_rad"))
            ):
                raise ValueError(f"invalid population row {row['logical_id']}")
            if row["logical_id"] != f"b{block:03d}-j{local:04d}":
                raise ValueError(f"population logical ID does not match its indices: {row['logical_id']}")
            blocks.setdefault(block, []).append(row)
    identities: set[str] = set()
    for block, rows in blocks.items():
        rows.sort(key=lambda row: row["local_index"])
        if [row["local_index"] for row in rows] != list(range(len(rows))):
            raise ValueError(f"block {block} local indices are not contiguous from zero")
        for row in rows:
            if row["logical_id"] in identities:
                raise ValueError(f"duplicate logical ID {row['logical_id']}")
            identities.add(row["logical_id"])
    _POPULATION_CACHE[key] = (observed_identity, blocks)
    return blocks


def _state_rows(path: Path) -> list[dict[str, str]]:
    key = str(path.resolve())
    identity = _file_identity(path)
    cached = _STATE_CACHE.get(key)
    if cached is not None and cached[0] == identity:
        return cached[1]
    rows = _load_rows(path)
    after = _file_identity(path)
    if after != identity:
        raise RuntimeError(f"state CSV changed while it was being read: {path}")
    _STATE_CACHE[key] = (after, rows)
    return rows


def _verify_matched_states(source_rows: list[dict[str, str]], control_rows: list[dict[str, str]]) -> dict[str, Any]:
    source, control = _by_name(source_rows), _by_name(control_rows)
    if tuple(row["name"] for row in control_rows) != COMMON_NAMES:
        raise ValueError(f"control active bodies must be {COMMON_NAMES}")
    if tuple(row["name"] for row in source_rows) != (*COMMON_NAMES, SOURCE_NAME):
        raise ValueError("source state must add exactly candidate 9118 after the common bodies")
    for rows in (source_rows, control_rows):
        if any(Decimal(row["mass"]) <= 0 for row in rows):
            raise ValueError("all state rows must contain positive gravitational parameters")
    mismatches = []
    digest_rows = []
    for name in COMMON_NAMES:
        left, right = source[name], control[name]
        if _fraction(left["mass"]) != _fraction(right["mass"]):
            mismatches.append(f"{name}:gm")
        left_relative = _relative_state(left, source["Sun"])
        right_relative = _relative_state(right, control["Sun"])
        if left_relative != right_relative:
            mismatches.append(f"{name}:relative_state")
        digest_rows.append(
            {
                "name": name,
                "gm": str(_fraction(right["mass"])),
                "relative_state": [str(value) for value in right_relative],
            }
        )
    if mismatches:
        raise ValueError(f"source/control common state mismatch: {mismatches}")
    return {
        "common_active_bodies": list(COMMON_NAMES),
        "source_only_body": SOURCE_NAME,
        "maximum_common_relative_state_difference": 0.0,
        "common_relative_state_sha256": _canonical_sha256(digest_rows),
        "mass_column_interpretation": "GM in AU^3/year^2; simulation G=1",
    }


def _build_simulation(
    rows: list[dict[str, str]],
    elements: list[dict[str, Any]],
    dt_years: float,
) -> tuple[Any, list[str]]:
    import rebound

    simulation = rebound.Simulation()
    simulation.G = 1.0
    sun_row = _by_name(rows)["Sun"]
    names = [row["name"] for row in rows]
    for row in rows:
        relative = _relative_state(row, sun_row)
        simulation.add(
            m=float(row["mass"]),
            x=float(relative[0]),
            y=float(relative[1]),
            z=float(relative[2]),
            vx=float(relative[3]),
            vy=float(relative[4]),
            vz=float(relative[5]),
            hash=row["name"],
        )
    simulation.N_active = len(rows)
    simulation.testparticle_type = 0
    simulation.integrator = "mercurius"
    simulation.dt = dt_years
    simulation.ri_mercurius.r_crit_hill = 3.0
    simulation.collision = "none"
    for element in elements:
        simulation.add(
            primary=simulation.particles[0],
            m=0.0,
            a=element["a0_AU"],
            e=element["e0"],
            inc=math.radians(element["i0_deg"]),
            Omega=element["Omega0_rad"],
            omega=element["omega0_rad"],
            M=element["M0_rad"],
        )
    return simulation, names


def _hill_radius(simulation: Any, body_index: int) -> float:
    sun = simulation.particles[0]
    body = simulation.particles[body_index]
    orbit = body.orbit(primary=sun)
    return orbit.a * (body.m / (3.0 * sun.m)) ** (1.0 / 3.0)


def _blank_tracker(simulation: Any, names: list[str], elements: list[dict[str, Any]]) -> dict[str, Any]:
    neptune_index = names.index("Neptune")
    source_index = names.index(SOURCE_NAME) if SOURCE_NAME in names else None
    return {
        "minimum_q": [element["q0_AU"] for element in elements],
        "first_low_q_year": [None] * len(elements),
        "ever_unbound": [False] * len(elements),
        "first_unbound_year": [None] * len(elements),
        "minimum_neptune_distance_AU": [None] * len(elements),
        "minimum_neptune_hill_ratio": [None] * len(elements),
        "minimum_source_distance_AU": [None] * len(elements),
        "minimum_source_hill_ratio": [None] * len(elements),
        "neptune_index": neptune_index,
        "source_index": source_index,
        "neptune_hill_radius_AU": _hill_radius(simulation, neptune_index),
        "source_hill_radius_AU": _hill_radius(simulation, source_index) if source_index is not None else None,
        "sample_count": 0,
        "timeseries": [],
    }


def _update_minimum(values: list[float | None], index: int, candidate: float) -> None:
    current = values[index]
    if math.isfinite(candidate) and (current is None or candidate < current):
        values[index] = candidate


def _validate_timeseries(
    values: Any,
    job: Mapping[str, Any],
    maximum_sample_index: int,
) -> None:
    if not isinstance(values, list):
        raise ValueError("tracker timeseries must be a list")
    aggregate_stride = int(job["aggregate_stride_samples"])
    expected_count = maximum_sample_index // aggregate_stride + 1
    if len(values) != expected_count:
        raise ValueError(
            f"tracker timeseries has {len(values)} rows; expected {expected_count}"
        )
    previous_injections = -1
    local_count = int(job["local_count"])
    for index, row in enumerate(values):
        if not isinstance(row, dict) or set(row) != _TIMESERIES_FIELDS:
            raise ValueError(f"tracker timeseries row {index} has the wrong schema")
        expected_time = index * float(job["aggregate_years"])
        observed_time = _strict_float(row["time_year"], f"timeseries[{index}].time_year")
        if not math.isclose(observed_time, expected_time, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError(f"tracker timeseries row {index} has the wrong epoch")
        bound = _strict_int(row["bound"], f"timeseries[{index}].bound", minimum=0)
        injections = _strict_int(
            row["cumulative_sampled_injections"],
            f"timeseries[{index}].cumulative_sampled_injections",
            minimum=0,
        )
        if bound > local_count or injections > local_count or injections < previous_injections:
            raise ValueError(f"tracker timeseries row {index} violates count bounds or monotonicity")
        previous_injections = injections
        survival = _strict_float(
            row["survival_fraction"], f"timeseries[{index}].survival_fraction", minimum=0.0
        )
        injection_fraction = _strict_float(
            row["cumulative_sampled_injection_fraction"],
            f"timeseries[{index}].cumulative_sampled_injection_fraction",
            minimum=0.0,
        )
        if survival > 1.0 or injection_fraction > 1.0:
            raise ValueError(f"tracker timeseries row {index} contains a fraction above one")
        if survival != bound / local_count or injection_fraction != injections / local_count:
            raise ValueError(f"tracker timeseries row {index} fraction does not match its count")
        mean_q = row["mean_current_bound_q_AU"]
        width = row["current_bound_inclination_width_deg"]
        if bound == 0:
            if mean_q is not None or width is not None:
                raise ValueError(f"tracker timeseries row {index} has bound statistics without bodies")
        else:
            _strict_float(mean_q, f"timeseries[{index}].mean_current_bound_q_AU")
            _strict_float(
                width,
                f"timeseries[{index}].current_bound_inclination_width_deg",
                minimum=0.0,
            )


def _validate_tracker(
    tracker: Any,
    job: Mapping[str, Any],
    elements: list[dict[str, Any]],
    names: list[str],
    checkpoint_index: int,
) -> None:
    if not isinstance(tracker, dict) or set(tracker) != _TRACKER_FIELDS:
        raise ValueError("checkpoint tracker has the wrong schema")
    local_count = int(job["local_count"])
    if len(elements) != local_count:
        raise ValueError("checkpoint validation received the wrong element count")
    sample_index = checkpoint_index * int(job["checkpoint_stride_samples"])
    if sample_index > int(job["total_samples"]):
        raise ValueError("checkpoint index lies after the contracted endpoint")
    if _strict_int(tracker["sample_count"], "tracker sample_count", minimum=1) != sample_index + 1:
        raise ValueError("checkpoint tracker sample_count does not match its index")
    neptune_index = names.index("Neptune")
    source_index = names.index(SOURCE_NAME) if SOURCE_NAME in names else None
    observed_neptune_index = _strict_int(
        tracker["neptune_index"], "tracker neptune_index", minimum=0
    )
    observed_source_index = tracker["source_index"]
    if observed_source_index is not None:
        observed_source_index = _strict_int(
            observed_source_index, "tracker source_index", minimum=0
        )
    if observed_neptune_index != neptune_index or observed_source_index != source_index:
        raise ValueError("checkpoint tracker body indices do not match the contracted arm")
    _strict_float(tracker["neptune_hill_radius_AU"], "tracker Neptune Hill radius", minimum=0.0)
    if float(tracker["neptune_hill_radius_AU"]) <= 0.0:
        raise ValueError("tracker Neptune Hill radius must be positive")
    if source_index is None:
        if tracker["source_hill_radius_AU"] is not None:
            raise ValueError("control tracker must not contain a source Hill radius")
    else:
        _strict_float(tracker["source_hill_radius_AU"], "tracker source Hill radius", minimum=0.0)
        if float(tracker["source_hill_radius_AU"]) <= 0.0:
            raise ValueError("tracker source Hill radius must be positive")

    list_fields = _TRACKER_FIELDS - {
        "neptune_index",
        "source_index",
        "neptune_hill_radius_AU",
        "source_hill_radius_AU",
        "sample_count",
        "timeseries",
    }
    for field in list_fields:
        if not isinstance(tracker[field], list) or len(tracker[field]) != local_count:
            raise ValueError(f"checkpoint tracker {field} must have {local_count} entries")
    maximum_time = sample_index * float(job["sample_years"])
    for index, element in enumerate(elements):
        minimum_q = _strict_float(tracker["minimum_q"][index], f"tracker minimum_q[{index}]")
        if minimum_q > float(element["q0_AU"]):
            raise ValueError("checkpoint minimum_q exceeds its initial value")
        first_low = tracker["first_low_q_year"][index]
        injected = minimum_q < float(job["q_lower"])
        if injected != (first_low is not None):
            raise ValueError("checkpoint low-q flag and epoch disagree")
        if first_low is not None:
            first_low_value = _strict_float(
                first_low, f"tracker first_low_q_year[{index}]", minimum=0.0
            )
            if first_low_value > maximum_time or not _time_is_sample_epoch(
                first_low_value, float(job["sample_years"])
            ):
                raise ValueError("checkpoint first low-q epoch is outside the sampled grid")
        unbound = tracker["ever_unbound"][index]
        if not isinstance(unbound, bool):
            raise ValueError("checkpoint ever_unbound entries must be booleans")
        first_unbound = tracker["first_unbound_year"][index]
        if unbound != (first_unbound is not None):
            raise ValueError("checkpoint unbound flag and epoch disagree")
        if first_unbound is not None:
            first_unbound_value = _strict_float(
                first_unbound, f"tracker first_unbound_year[{index}]", minimum=0.0
            )
            if first_unbound_value > maximum_time or not _time_is_sample_epoch(
                first_unbound_value, float(job["sample_years"])
            ):
                raise ValueError("checkpoint first-unbound epoch is outside the sampled grid")
        for field in ("minimum_neptune_distance_AU", "minimum_neptune_hill_ratio"):
            _strict_float(tracker[field][index], f"tracker {field}[{index}]", minimum=0.0)
        for field in ("minimum_source_distance_AU", "minimum_source_hill_ratio"):
            value = tracker[field][index]
            if source_index is None:
                if value is not None:
                    raise ValueError(f"control checkpoint tracker must leave {field} empty")
            else:
                _strict_float(value, f"tracker {field}[{index}]", minimum=0.0)
    _validate_timeseries(tracker["timeseries"], job, sample_index)


def _validate_tracker_progress(previous: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    if current["timeseries"][: len(previous["timeseries"])] != previous["timeseries"]:
        raise ValueError("checkpoint timeseries does not preserve its prior prefix")
    for field in (
        "minimum_q",
        "minimum_neptune_distance_AU",
        "minimum_neptune_hill_ratio",
        "minimum_source_distance_AU",
        "minimum_source_hill_ratio",
    ):
        for before, after in zip(previous[field], current[field], strict=True):
            if before is None:
                if after is not None and field.startswith("minimum_source"):
                    raise ValueError("control checkpoint unexpectedly acquired source distances")
            elif after is None or float(after) > float(before):
                raise ValueError(f"checkpoint {field} is not monotonically non-increasing")
    for flag_field, epoch_field in (
        ("ever_unbound", "first_unbound_year"),
        (None, "first_low_q_year"),
    ):
        if flag_field is not None:
            for before, after in zip(previous[flag_field], current[flag_field], strict=True):
                if before and not after:
                    raise ValueError(f"checkpoint {flag_field} reverted from true to false")
        for before, after in zip(previous[epoch_field], current[epoch_field], strict=True):
            if before is not None and after != before:
                raise ValueError(f"checkpoint {epoch_field} changed after first assignment")


def _sample(
    simulation: Any,
    elements: list[dict[str, Any]],
    tracker: dict[str, Any],
    q_lower: float,
    record_timeseries: bool,
) -> None:
    particles = simulation.particles
    sun = particles[0]
    neptune = particles[tracker["neptune_index"]]
    source = particles[tracker["source_index"]] if tracker["source_index"] is not None else None
    bound_count = 0
    current_q: list[float] = []
    current_i: list[float] = []
    for index, _element in enumerate(elements):
        particle = particles[simulation.N_active + index]
        try:
            orbit = particle.orbit(primary=sun)
            q = orbit.a * (1.0 - orbit.e)
            inclination = math.degrees(orbit.inc)
            bound = orbit.a > 0.0 and orbit.e < 1.0 and math.isfinite(q) and math.isfinite(inclination)
        except (ValueError, ZeroDivisionError, OverflowError):
            bound = False
            q = inclination = math.nan
        if bound:
            bound_count += 1
            current_q.append(q)
            current_i.append(inclination)
            tracker["minimum_q"][index] = min(tracker["minimum_q"][index], q)
            if q < q_lower and tracker["first_low_q_year"][index] is None:
                tracker["first_low_q_year"][index] = float(simulation.t)
        else:
            tracker["ever_unbound"][index] = True
            if tracker["first_unbound_year"][index] is None:
                tracker["first_unbound_year"][index] = float(simulation.t)
        dx, dy, dz = particle.x - neptune.x, particle.y - neptune.y, particle.z - neptune.z
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        _update_minimum(tracker["minimum_neptune_distance_AU"], index, distance)
        _update_minimum(
            tracker["minimum_neptune_hill_ratio"],
            index,
            distance / tracker["neptune_hill_radius_AU"],
        )
        if source is not None:
            dx, dy, dz = particle.x - source.x, particle.y - source.y, particle.z - source.z
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            _update_minimum(tracker["minimum_source_distance_AU"], index, distance)
            _update_minimum(
                tracker["minimum_source_hill_ratio"],
                index,
                distance / tracker["source_hill_radius_AU"],
            )
    tracker["sample_count"] += 1
    if record_timeseries:
        injections = sum(value < q_lower for value in tracker["minimum_q"])
        tracker["timeseries"].append(
            {
                "time_year": float(simulation.t),
                "bound": bound_count,
                "survival_fraction": bound_count / len(elements),
                "cumulative_sampled_injections": injections,
                "cumulative_sampled_injection_fraction": injections / len(elements),
                "mean_current_bound_q_AU": statistics.fmean(current_q) if current_q else None,
                "current_bound_inclination_width_deg": statistics.pstdev(current_i) if current_i else None,
            }
        )


def _checkpoint_paths(directory: Path, index: int) -> tuple[Path, Path]:
    return directory / f"checkpoint_{index:03d}.bin", directory / f"checkpoint_{index:03d}.json"


def _write_simulation_create_only(path: Path, simulation: Any) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        simulation.save_to_file(str(temporary), delete_file=True)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _write_checkpoint(
    directory: Path,
    simulation: Any,
    tracker: dict[str, Any],
    index: int,
    job: Mapping[str, Any],
    elements: list[dict[str, Any]],
    names: list[str],
) -> bool:
    binary_path, state_path = _checkpoint_paths(directory, index)
    if binary_path.exists() or state_path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint {index} in {directory}")
    _validate_tracker(tracker, job, elements, names, index)
    expected_time = index * int(job["checkpoint_stride_samples"]) * float(job["sample_years"])
    if not math.isclose(float(simulation.t), expected_time, rel_tol=0.0, abs_tol=1e-10):
        raise RuntimeError("checkpoint simulation time does not match its index")
    before = _simulation_digest(simulation)
    _write_simulation_create_only(binary_path, simulation)
    import rebound

    replay = rebound.Simulation(str(binary_path))
    after = _simulation_digest(replay)
    if before != after:
        raise RuntimeError("REBOUND checkpoint/replay changed the binary64 state")
    state = {
        "schema": CHECKPOINT_SCHEMA,
        "job_sha256": job["job_sha256"],
        "checkpoint_index": index,
        "time_year": float(simulation.t),
        "simulation_archive_sha256": _sha256_file(binary_path),
        "simulation_state_sha256": before,
        "tracker": tracker,
    }
    _write_json_create_only(state_path, state)
    return True


def _load_checkpoint(
    directory: Path,
    job: Mapping[str, Any],
    elements: list[dict[str, Any]],
    names: list[str],
) -> tuple[Any, dict[str, Any], int] | None:
    import rebound

    artifacts: dict[str, set[int]] = {"bin": set(), "json": set()}
    for path in directory.iterdir():
        if not path.name.startswith("checkpoint_"):
            continue
        match = _CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is None or not path.is_file():
            raise RuntimeError(f"invalid checkpoint artifact name or type: {path}")
        index = int(match.group(1))
        kind = match.group(2)
        expected_path = _checkpoint_paths(directory, index)[0 if kind == "bin" else 1]
        if path != expected_path or index in artifacts[kind]:
            raise RuntimeError(f"noncanonical or duplicate checkpoint artifact: {path}")
        artifacts[kind].add(index)
    if not artifacts["bin"] and not artifacts["json"]:
        return None
    if artifacts["bin"] != artifacts["json"]:
        raise RuntimeError("checkpoint archive/metadata pairs are incomplete")
    maximum = max(artifacts["json"])
    if artifacts["json"] != set(range(maximum + 1)):
        raise RuntimeError("checkpoint indices are not contiguous from zero")

    latest: tuple[Any, dict[str, Any], int] | None = None
    state_fields = {
        "schema",
        "job_sha256",
        "checkpoint_index",
        "time_year",
        "simulation_archive_sha256",
        "simulation_state_sha256",
        "tracker",
    }
    for index in range(maximum + 1):
        binary_path, state_path = _checkpoint_paths(directory, index)
        data, _identity = _stable_read_bytes(state_path)
        try:
            state = _decode_json_object(data, f"checkpoint {index} metadata")
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        if set(state) != state_fields:
            raise RuntimeError(f"checkpoint {index} metadata has the wrong exact schema")
        if state["schema"] != CHECKPOINT_SCHEMA or state["job_sha256"] != job["job_sha256"]:
            raise RuntimeError(f"checkpoint {index} schema or job identity mismatch")
        if _strict_int(state["checkpoint_index"], "checkpoint index", minimum=0) != index:
            raise RuntimeError(f"checkpoint {index} metadata index mismatch")
        expected_time = index * int(job["checkpoint_stride_samples"]) * float(job["sample_years"])
        observed_time = _strict_float(state["time_year"], "checkpoint time", minimum=0.0)
        if not math.isclose(observed_time, expected_time, rel_tol=0.0, abs_tol=1e-10):
            raise RuntimeError(f"checkpoint {index} metadata time mismatch")
        if not _is_sha256(state["simulation_archive_sha256"]) or not _is_sha256(
            state["simulation_state_sha256"]
        ):
            raise RuntimeError(f"checkpoint {index} contains an invalid digest")
        if _sha256_file(binary_path) != state["simulation_archive_sha256"]:
            raise RuntimeError(f"checkpoint {index} archive digest mismatch")
        simulation = rebound.Simulation(str(binary_path))
        if _simulation_digest(simulation) != state["simulation_state_sha256"]:
            raise RuntimeError(f"checkpoint {index} state digest mismatch")
        if simulation.integrator != "mercurius" or simulation.testparticle_type != 0:
            raise RuntimeError(f"checkpoint {index} integrator contract mismatch")
        if simulation.N_active != len(names) or simulation.N != len(names) + len(elements):
            raise RuntimeError(f"checkpoint {index} particle-count mismatch")
        if not math.isclose(float(simulation.t), expected_time, rel_tol=0.0, abs_tol=1e-10):
            raise RuntimeError(f"checkpoint {index} simulation time mismatch")
        if float(simulation.dt) != float(job["dt_years"]) or float(simulation.G) != 1.0:
            raise RuntimeError(f"checkpoint {index} step or gravity constant mismatch")
        if float(simulation.ri_mercurius.r_crit_hill) != 3.0:
            raise RuntimeError(f"checkpoint {index} MERCURIUS critical-radius mismatch")
        _validate_tracker(state["tracker"], job, elements, names, index)
        if latest is not None:
            _validate_tracker_progress(latest[1], state["tracker"])
        latest = simulation, state["tracker"], index
    return latest


def _final_rows(
    simulation: Any,
    elements: list[dict[str, Any]],
    tracker: dict[str, Any],
    q_lower: float,
) -> list[dict[str, Any]]:
    result = []
    sun = simulation.particles[0]
    for index, element in enumerate(elements):
        particle = simulation.particles[simulation.N_active + index]
        try:
            orbit = particle.orbit(primary=sun)
            q = orbit.a * (1.0 - orbit.e)
            bound = orbit.a > 0.0 and orbit.e < 1.0 and math.isfinite(q) and math.isfinite(orbit.inc)
            final_q = q if bound else None
            final_i = math.degrees(orbit.inc) if bound else None
        except (ValueError, ZeroDivisionError, OverflowError):
            bound, final_q, final_i = False, None, None
        result.append(
            {
                **element,
                "minimum_sampled_q_AU": tracker["minimum_q"][index],
                "sampled_injection": int(tracker["minimum_q"][index] < q_lower),
                "first_sampled_low_q_year": tracker["first_low_q_year"][index],
                "ever_unbound_at_sample": int(tracker["ever_unbound"][index]),
                "first_unbound_year": tracker["first_unbound_year"][index],
                "bound_final": int(bound),
                "final_q_AU": final_q,
                "final_i_deg": final_i,
                "minimum_sampled_neptune_distance_AU": tracker["minimum_neptune_distance_AU"][index],
                "minimum_sampled_neptune_hill_ratio": tracker["minimum_neptune_hill_ratio"][index],
                "minimum_sampled_source_distance_AU": tracker["minimum_source_distance_AU"][index],
                "minimum_sampled_source_hill_ratio": tracker["minimum_source_hill_ratio"][index],
            }
        )
    return result


def _read_tracer_csv(path: Path) -> tuple[list[dict[str, str]], str]:
    data, _identity = _stable_read_bytes(path)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("tracer CSV must be UTF-8") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != TRACER_COLUMNS:
        raise ValueError(f"tracer CSV must have exactly these columns: {TRACER_COLUMNS}")
    rows = list(reader)
    if reader.restkey is not None or any(None in row for row in rows):
        raise ValueError("tracer CSV contains fields outside its locked schema")
    return rows, hashlib.sha256(data).hexdigest()


def _validate_tracer_csv(
    path: Path,
    job: Mapping[str, Any],
    elements: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], str]:
    rows, digest = _read_tracer_csv(path)
    if len(rows) != len(elements) or len(rows) != int(job["local_count"]):
        raise ValueError("tracer CSV row count does not match the job")
    identities: set[str] = set()
    for offset, (row, element) in enumerate(zip(rows, elements, strict=True)):
        block = _strict_csv_int(row["block_index"], f"tracer[{offset}].block_index")
        local = _strict_csv_int(row["local_index"], f"tracer[{offset}].local_index")
        if block != int(job["block_index"]) or local != offset:
            raise ValueError("tracer CSV indices are not in the contracted block order")
        if row["logical_id"] != element["logical_id"] or row["logical_id"] in identities:
            raise ValueError("tracer CSV logical identities are missing, duplicated, or reordered")
        identities.add(row["logical_id"])
        for field in POPULATION_COLUMNS[3:]:
            value = _strict_csv_float(row[field], f"tracer[{offset}].{field}")
            if value != float(element[field]):
                raise ValueError(f"tracer CSV changed locked initial field {field}")

        minimum_q = _strict_csv_float(
            row["minimum_sampled_q_AU"], f"tracer[{offset}].minimum_sampled_q_AU"
        )
        if minimum_q is None or minimum_q > float(element["q0_AU"]):
            raise ValueError("tracer minimum sampled perihelion is invalid")
        if row["sampled_injection"] not in {"0", "1"}:
            raise ValueError("tracer sampled_injection must be 0 or 1")
        injected = row["sampled_injection"] == "1"
        if injected != (minimum_q < float(job["q_lower"])):
            raise ValueError("tracer injection flag does not match its minimum perihelion")
        first_low = _strict_csv_float(
            row["first_sampled_low_q_year"],
            f"tracer[{offset}].first_sampled_low_q_year",
            optional=True,
        )
        if injected != (first_low is not None):
            raise ValueError("tracer injection flag and first low-q epoch disagree")
        if first_low is not None and (
            first_low < 0.0
            or first_low > float(job["duration_years"])
            or not _time_is_sample_epoch(first_low, float(job["sample_years"]))
        ):
            raise ValueError("tracer first low-q epoch is outside the contracted sampled grid")

        if row["ever_unbound_at_sample"] not in {"0", "1"}:
            raise ValueError("tracer ever_unbound_at_sample must be 0 or 1")
        ever_unbound = row["ever_unbound_at_sample"] == "1"
        first_unbound = _strict_csv_float(
            row["first_unbound_year"], f"tracer[{offset}].first_unbound_year", optional=True
        )
        if ever_unbound != (first_unbound is not None):
            raise ValueError("tracer unbound flag and first-unbound epoch disagree")
        if first_unbound is not None and (
            first_unbound < 0.0
            or first_unbound > float(job["duration_years"])
            or not _time_is_sample_epoch(first_unbound, float(job["sample_years"]))
        ):
            raise ValueError("tracer first-unbound epoch is outside the contracted sampled grid")

        if row["bound_final"] not in {"0", "1"}:
            raise ValueError("tracer bound_final must be 0 or 1")
        bound_final = row["bound_final"] == "1"
        final_q = _strict_csv_float(
            row["final_q_AU"], f"tracer[{offset}].final_q_AU", optional=True
        )
        final_i = _strict_csv_float(
            row["final_i_deg"], f"tracer[{offset}].final_i_deg", optional=True
        )
        if bound_final != (final_q is not None and final_i is not None):
            raise ValueError("tracer final-bound flag and final elements disagree")
        if final_i is not None and not 0.0 <= final_i <= 180.0:
            raise ValueError("tracer final inclination is outside [0, 180] degrees")
        for field in (
            "minimum_sampled_neptune_distance_AU",
            "minimum_sampled_neptune_hill_ratio",
        ):
            value = _strict_csv_float(row[field], f"tracer[{offset}].{field}")
            if value is None or value < 0.0:
                raise ValueError(f"tracer {field} must be non-negative")
        for field in (
            "minimum_sampled_source_distance_AU",
            "minimum_sampled_source_hill_ratio",
        ):
            value = _strict_csv_float(row[field], f"tracer[{offset}].{field}", optional=True)
            if job["arm"] == "control":
                if value is not None:
                    raise ValueError(f"control tracer must leave {field} empty")
            elif value is None or value < 0.0:
                raise ValueError(f"source tracer {field} must be non-negative")
    return rows, digest


def _block_content_sha256(summary: Mapping[str, Any]) -> str:
    payload = {key: summary[key] for key in sorted(_BLOCK_FIELDS - {"block_content_sha256"})}
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _validate_active_endpoint_state(value: Any, expected_count: int) -> None:
    fields = {"index", "m", "x", "y", "z", "vx", "vy", "vz"}
    if not isinstance(value, list) or len(value) != expected_count:
        raise ValueError("block active endpoint state has the wrong length")
    for index, row in enumerate(value):
        if not isinstance(row, dict) or set(row) != fields or row["index"] != str(index):
            raise ValueError("block active endpoint state has the wrong schema or order")
        for field in fields - {"index"}:
            if not isinstance(row[field], str):
                raise ValueError("block active endpoint components must be hexadecimal strings")
            try:
                number = float.fromhex(row[field])
            except ValueError as error:
                raise ValueError("block active endpoint contains invalid hexadecimal binary64") from error
            if not math.isfinite(number):
                raise ValueError("block active endpoint contains a non-finite component")


def _validate_cached_block(
    directory: Path,
    job: Mapping[str, Any],
    elements: list[dict[str, Any]],
    expected_active: int,
) -> dict[str, Any]:
    summary_path = directory / "summary.json"
    tracer_path = directory / "tracers.csv"
    data, _identity = _stable_read_bytes(summary_path)
    cached = _decode_json_object(data, "cached block summary")
    if set(cached) != _BLOCK_FIELDS:
        raise ValueError("cached block summary has the wrong exact schema")
    if cached["schema"] != BLOCK_SCHEMA or cached["job_sha256"] != job["job_sha256"]:
        raise ValueError("cached block schema or job identity does not match")
    if cached["arm"] != job["arm"] or cached["stage"] != job["stage"]:
        raise ValueError("cached block arm or stage does not match")
    if _strict_int(cached["block_index"], "cached block index", minimum=0) != job["block_index"]:
        raise ValueError("cached block index does not match")
    tracers = _strict_int(cached["tracers"], "cached block tracers", minimum=1)
    active = _strict_int(cached["N_active"], "cached block N_active", minimum=1)
    particles = _strict_int(cached["particles"], "cached block particles", minimum=1)
    if tracers != job["local_count"] or active != expected_active:
        raise ValueError("cached block particle counts do not match")
    if particles != expected_active + int(job["local_count"]):
        raise ValueError("cached block total particle count does not match")
    for field in ("dt_years", "duration_years", "sample_years"):
        _strict_float(cached[field], f"cached block {field}")
        if float(cached[field]) != float(job[field]):
            raise ValueError(f"cached block {field} does not match the job")
    samples = _strict_int(
        cached["sample_count_including_t0"], "cached block sample count", minimum=1
    )
    if samples != int(job["total_samples"]) + 1:
        raise ValueError("cached block sample count does not match the job")
    expected_checkpoints = int(job["total_samples"]) // int(job["checkpoint_stride_samples"]) + 1
    checkpoints = _strict_int(
        cached["checkpoint_epochs_including_t0"], "cached block checkpoint count", minimum=1
    )
    if checkpoints != expected_checkpoints:
        raise ValueError("cached block checkpoint count does not match the job")
    if cached["restart_replay_state_hash_exact"] is not True or cached["cartesian_state_finite"] is not True:
        raise ValueError("cached block does not attest exact replay and finite state")
    if cached["tracer_csv"] != "tracers.csv" or cached["summary_json"] != "summary.json":
        raise ValueError("cached block artifact paths must be portable block-relative names")
    if not _is_sha256(cached["tracer_csv_sha256"]) or not _is_sha256(
        cached["active_endpoint_state_sha256"]
    ):
        raise ValueError("cached block contains an invalid artifact digest")
    _validate_active_endpoint_state(cached["active_endpoint_state"], expected_active)
    if _canonical_sha256(cached["active_endpoint_state"]) != cached["active_endpoint_state_sha256"]:
        raise ValueError("cached block active endpoint digest mismatch")
    rows, tracer_digest = _validate_tracer_csv(tracer_path, job, elements)
    if tracer_digest != cached["tracer_csv_sha256"]:
        raise ValueError("cached block tracer CSV digest mismatch")
    _validate_timeseries(cached["timeseries"], job, int(job["total_samples"]))
    endpoint = cached["timeseries"][-1]
    injections = sum(row["sampled_injection"] == "1" for row in rows)
    bound = sum(row["bound_final"] == "1" for row in rows)
    if endpoint["cumulative_sampled_injections"] != injections or endpoint["bound"] != bound:
        raise ValueError("cached block endpoint timeseries does not match tracer rows")
    final_q = [float(row["final_q_AU"]) for row in rows if row["bound_final"] == "1"]
    final_i = [float(row["final_i_deg"]) for row in rows if row["bound_final"] == "1"]
    expected_mean_q = statistics.fmean(final_q) if final_q else None
    expected_width_i = statistics.pstdev(final_i) if final_i else None
    if (
        endpoint["mean_current_bound_q_AU"] != expected_mean_q
        or endpoint["current_bound_inclination_width_deg"] != expected_width_i
    ):
        raise ValueError("cached block endpoint moments do not match tracer rows")
    _strict_float(cached["elapsed_seconds"], "cached block elapsed_seconds", minimum=0.0)
    expected_digest = _block_content_sha256(cached)
    if cached["block_content_sha256"] != expected_digest:
        raise ValueError("cached block content digest mismatch")
    result = dict(cached)
    result["resumed_from_complete_block"] = True
    result["summary_json_sha256"] = hashlib.sha256(data).hexdigest()
    return result


def _validate_completed_checkpoint_custody(
    cached: Mapping[str, Any],
    directory: Path,
    job: Mapping[str, Any],
    elements: list[dict[str, Any]],
    names: list[str],
) -> None:
    loaded = _load_checkpoint(directory, job, elements, names)
    if loaded is None:
        raise RuntimeError("completed block has no retained checkpoint chain")
    simulation, tracker, checkpoint_index = loaded
    expected_index = int(job["total_samples"]) // int(job["checkpoint_stride_samples"])
    if checkpoint_index != expected_index:
        raise RuntimeError("completed block checkpoint chain does not reach the endpoint")
    active_state = _active_state(simulation)
    if active_state != cached["active_endpoint_state"]:
        raise RuntimeError("completed block endpoint state differs from its final checkpoint")
    if tracker["timeseries"] != cached["timeseries"]:
        raise RuntimeError("completed block timeseries differs from its final checkpoint")
    stored_rows, _digest = _read_tracer_csv(directory / "tracers.csv")
    replay_rows = _final_rows(simulation, elements, tracker, float(job["q_lower"]))
    for index, (stored, replay) in enumerate(zip(stored_rows, replay_rows, strict=True)):
        for field in TRACER_COLUMNS:
            expected = "" if replay[field] is None else str(replay[field])
            if stored[field] != expected:
                raise RuntimeError(
                    f"completed tracer row {index} field {field} differs from the final checkpoint"
                )


def _run_block(job: dict[str, Any]) -> dict[str, Any]:
    _validate_job(job)
    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    with _block_lock(directory):
        return _run_block_locked(job, directory)


def _run_block_locked(job: dict[str, Any], directory: Path) -> dict[str, Any]:
    started = time.perf_counter()
    summary_path = directory / "summary.json"
    tracer_path = directory / "tracers.csv"
    if _cached_sha256(Path(job["state_csv"])) != job["state_sha256"]:
        raise RuntimeError("worker state hash mismatch")
    if _cached_sha256(Path(job["population_csv"])) != job["population_sha256"]:
        raise RuntimeError("worker population hash mismatch")
    states = _state_rows(Path(job["state_csv"]))
    population = _load_population(Path(job["population_csv"]))
    if _cached_sha256(Path(job["state_csv"])) != job["state_sha256"]:
        raise RuntimeError("worker state changed while it was being loaded")
    if _cached_sha256(Path(job["population_csv"])) != job["population_sha256"]:
        raise RuntimeError("worker population changed while it was being loaded")
    try:
        block_elements = population[job["block_index"]]
    except KeyError as error:
        raise RuntimeError("job block is absent from the locked population") from error
    if len(block_elements) < job["local_count"]:
        raise RuntimeError("job requests more elements than its locked population block")
    elements = block_elements[: job["local_count"]]
    expected_active = len(states)
    if summary_path.exists():
        if not summary_path.is_file():
            raise RuntimeError(f"block summary path is not a regular file: {summary_path}")
        try:
            cached = _validate_cached_block(directory, job, elements, expected_active)
            _validate_completed_checkpoint_custody(
                cached,
                directory,
                job,
                elements,
                [row["name"] for row in states],
            )
            return cached
        except (OSError, ValueError, RuntimeError) as error:
            raise RuntimeError(f"locked block conflicts with requested job: {directory}: {error}") from error
    if tracer_path.exists():
        raise RuntimeError(f"orphan tracer CSV requires custody review before resume: {tracer_path}")
    built, names = _build_simulation(states, elements, job["dt_years"])
    if built.N != len(names) + len(elements) or built.N_active != len(names):
        raise RuntimeError("particle-count invariant failed")
    loaded = _load_checkpoint(directory, job, elements, names)
    replay_exact = True
    if loaded is None:
        simulation = built
        tracker = _blank_tracker(simulation, names, elements)
        checkpoint_index = 0
        _sample(simulation, elements, tracker, job["q_lower"], True)
        replay_exact = _write_checkpoint(directory, simulation, tracker, 0, job, elements, names)
        import rebound

        simulation = rebound.Simulation(str(_checkpoint_paths(directory, 0)[0]))
    else:
        simulation, tracker, checkpoint_index = loaded
        if simulation.N != len(names) + len(elements) or simulation.N_active != len(names):
            raise RuntimeError("checkpoint particle-count invariant failed")
    del built
    sample_stride = job["checkpoint_stride_samples"]
    aggregate_stride = job["aggregate_stride_samples"]
    for sample_index in range(checkpoint_index * sample_stride + 1, job["total_samples"] + 1):
        simulation.integrate(sample_index * job["sample_years"], exact_finish_time=1)
        _sample(
            simulation,
            elements,
            tracker,
            job["q_lower"],
            sample_index % aggregate_stride == 0,
        )
        if sample_index % sample_stride == 0:
            checkpoint_index = sample_index // sample_stride
            replay_exact = _write_checkpoint(
                directory, simulation, tracker, checkpoint_index, job, elements, names
            ) and replay_exact
            import rebound

            simulation = rebound.Simulation(str(_checkpoint_paths(directory, checkpoint_index)[0]))
    if not math.isclose(simulation.t, job["duration_years"], rel_tol=0.0, abs_tol=1e-8):
        raise RuntimeError("block did not reach the contracted endpoint")
    _validate_tracker(tracker, job, elements, names, checkpoint_index)
    final_rows = _final_rows(simulation, elements, tracker, job["q_lower"])
    if not _cartesian_state_is_finite(simulation):
        raise RuntimeError("block produced a non-finite Cartesian state")
    _write_csv_create_only(tracer_path, final_rows, TRACER_COLUMNS)
    _rows, tracer_sha256 = _validate_tracer_csv(tracer_path, job, elements)
    active_state = _active_state(simulation)
    result = {
        "schema": BLOCK_SCHEMA,
        "job_sha256": job["job_sha256"],
        "arm": job["arm"],
        "stage": job["stage"],
        "block_index": job["block_index"],
        "tracers": len(elements),
        "particles": simulation.N,
        "N_active": simulation.N_active,
        "dt_years": job["dt_years"],
        "duration_years": job["duration_years"],
        "sample_years": job["sample_years"],
        "sample_count_including_t0": tracker["sample_count"],
        "checkpoint_epochs_including_t0": checkpoint_index + 1,
        "restart_replay_state_hash_exact": replay_exact,
        "cartesian_state_finite": True,
        "active_endpoint_state": active_state,
        "active_endpoint_state_sha256": _canonical_sha256(active_state),
        "tracer_csv": "tracers.csv",
        "tracer_csv_sha256": tracer_sha256,
        "summary_json": "summary.json",
        "timeseries": tracker["timeseries"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    result["block_content_sha256"] = _block_content_sha256(result)
    _write_json_create_only(summary_path, result)
    _validate_cached_block(directory, job, elements, expected_active)
    result["summary_json_sha256"] = _sha256_file(summary_path)
    return result


def _float_values(rows: Iterable[Mapping[str, str]], field: str, condition: str | None = None) -> list[float]:
    result = []
    for row in rows:
        if condition is not None and int(row[condition]) == 0:
            continue
        raw = row.get(field, "")
        if raw in (None, ""):
            continue
        value = float(raw)
        if math.isfinite(value):
            result.append(value)
    return result


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _population_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    count = len(rows)
    injections = sum(int(row["sampled_injection"]) for row in rows)
    bound = sum(int(row["bound_final"]) for row in rows)
    min_q = _float_values(rows, "minimum_sampled_q_AU")
    final_q = _float_values(rows, "final_q_AU", "bound_final")
    final_i = _float_values(rows, "final_i_deg", "bound_final")
    neptune_ratio = _float_values(rows, "minimum_sampled_neptune_hill_ratio")
    source_ratio = _float_values(rows, "minimum_sampled_source_hill_ratio")
    return {
        "tracers": count,
        "sampled_injections": injections,
        "sampled_injection_fraction": injections / count,
        "bound_final": bound,
        "survival_fraction": bound / count,
        "ever_unbound_at_sample": sum(int(row["ever_unbound_at_sample"]) for row in rows),
        "minimum_sampled_q_AU": {
            "mean": statistics.fmean(min_q),
            "p01": _quantile(min_q, 0.01),
            "p05": _quantile(min_q, 0.05),
            "median": _quantile(min_q, 0.5),
            "minimum": min(min_q),
        },
        "final_bound_q_AU": {
            "mean": statistics.fmean(final_q) if final_q else None,
            "median": _quantile(final_q, 0.5),
        },
        "final_bound_inclination_width_deg": statistics.pstdev(final_i) if final_i else None,
        "sampled_neptune_entries": {
            "lt_1_initial_hill_radius": sum(value < 1.0 for value in neptune_ratio),
            "lt_3_initial_hill_radii": sum(value < 3.0 for value in neptune_ratio),
        },
        "sampled_source_entries": (
            {
                "lt_1_initial_hill_radius": sum(value < 1.0 for value in source_ratio),
                "lt_3_initial_hill_radii": sum(value < 3.0 for value in source_ratio),
            }
            if source_ratio
            else None
        ),
    }


def _paired_effect(
    control: list[dict[str, str]],
    source: list[dict[str, str]],
    bootstrap_seed: str,
    repetitions: int,
    equivalence_margin: float,
) -> dict[str, Any]:
    control_by_id = {row["logical_id"]: row for row in control}
    source_by_id = {row["logical_id"]: row for row in source}
    if set(control_by_id) != set(source_by_id):
        raise ValueError("source/control logical tracer IDs differ")
    initial_fields = POPULATION_COLUMNS
    for identity, left in control_by_id.items():
        right = source_by_id[identity]
        if any(left[field] != right[field] for field in initial_fields):
            raise ValueError(f"paired initial metadata mismatch for {identity}")
    control_summary = _population_summary(control)
    source_summary = _population_summary(source)
    blocks = sorted({int(row["block_index"]) for row in control})
    block_effects = []
    for block in blocks:
        c = [row for row in control if int(row["block_index"]) == block]
        s = [row for row in source if int(row["block_index"]) == block]
        cs, ss = _population_summary(c), _population_summary(s)
        block_effects.append(
            {
                "block_index": block,
                "sampled_injection_fraction_difference": ss["sampled_injection_fraction"]
                - cs["sampled_injection_fraction"],
                "survival_fraction_difference": ss["survival_fraction"] - cs["survival_fraction"],
            }
        )
    injection_effects = [row["sampled_injection_fraction_difference"] for row in block_effects]
    point = statistics.fmean(injection_effects)
    interval = _bootstrap_ci(injection_effects, bootstrap_seed, repetitions)
    if interval[0] > 0.0:
        classification = "RESOLVED_POSITIVE_SOURCE_EFFECT"
    elif interval[1] < 0.0:
        classification = "RESOLVED_NEGATIVE_SOURCE_EFFECT"
    elif interval[0] >= -equivalence_margin and interval[1] <= equivalence_margin:
        classification = "EQUIVALENT_WITHIN_LOCKED_MARGIN"
    else:
        classification = "NO_RESOLVED_EFFECT"
    return {
        "control": control_summary,
        "source": source_summary,
        "source_minus_control": {
            "sampled_injections": source_summary["sampled_injections"] - control_summary["sampled_injections"],
            "sampled_injection_fraction": source_summary["sampled_injection_fraction"]
            - control_summary["sampled_injection_fraction"],
            "paired_block_mean_sampled_injection_fraction": point,
            "paired_block_bootstrap_95_percent_CI": interval,
            "equivalence_margin": equivalence_margin,
            "effect_classification": classification,
            "survival_fraction": source_summary["survival_fraction"] - control_summary["survival_fraction"],
            "wasserstein_minimum_sampled_q_AU": wasserstein_1d(
                _float_values(control, "minimum_sampled_q_AU"),
                _float_values(source, "minimum_sampled_q_AU"),
            ),
            "wasserstein_final_bound_q_AU": wasserstein_1d(
                _float_values(control, "final_q_AU", "bound_final"),
                _float_values(source, "final_q_AU", "bound_final"),
            ),
            "wasserstein_final_bound_i_deg": wasserstein_1d(
                _float_values(control, "final_i_deg", "bound_final"),
                _float_values(source, "final_i_deg", "bound_final"),
            ),
        },
        "block_effects": block_effects,
        "bootstrap_repetitions": repetitions,
        "paired_initial_metadata_sha256": _canonical_sha256(
            [{field: control_by_id[key][field] for field in initial_fields} for key in sorted(control_by_id)]
        ),
    }


def _audit_comparison(
    primary: dict[str, list[dict[str, str]]],
    audit: dict[str, list[dict[str, str]]],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    arms = {}
    primary_selected: dict[str, list[dict[str, str]]] = {}
    all_checks = []
    for arm in ("control", "source"):
        first_by_id = {row["logical_id"]: row for row in primary[arm]}
        second_by_id = {row["logical_id"]: row for row in audit[arm]}
        if set(first_by_id) != set(second_by_id):
            raise ValueError(f"{arm} audit identity mismatch")
        first = [first_by_id[key] for key in sorted(first_by_id)]
        second = [second_by_id[key] for key in sorted(second_by_id)]
        primary_selected[arm] = first
        fs, ss = _population_summary(first), _population_summary(second)
        disagreement = sum(
            first_by_id[key]["sampled_injection"] != second_by_id[key]["sampled_injection"]
            for key in first_by_id
        ) / len(first_by_id)
        metrics = {
            "tracers": len(first),
            "absolute_injection_fraction_difference": abs(
                ss["sampled_injection_fraction"] - fs["sampled_injection_fraction"]
            ),
            "injection_identity_disagreement_fraction": disagreement,
            "absolute_survival_fraction_difference": abs(ss["survival_fraction"] - fs["survival_fraction"]),
            "wasserstein_minimum_sampled_q_AU": wasserstein_1d(
                _float_values(first, "minimum_sampled_q_AU"),
                _float_values(second, "minimum_sampled_q_AU"),
            ),
            "wasserstein_final_bound_q_AU": wasserstein_1d(
                _float_values(first, "final_q_AU", "bound_final"),
                _float_values(second, "final_q_AU", "bound_final"),
            ),
            "wasserstein_final_bound_i_deg": wasserstein_1d(
                _float_values(first, "final_i_deg", "bound_final"),
                _float_values(second, "final_i_deg", "bound_final"),
            ),
        }
        checks = {
            "injection_fraction": metrics["absolute_injection_fraction_difference"]
            <= float(gates["max_injection_fraction_difference"]),
            "injection_identity": metrics["injection_identity_disagreement_fraction"]
            <= float(gates["max_injection_identity_disagreement_fraction"]),
            "survival_fraction": metrics["absolute_survival_fraction_difference"]
            <= float(gates["max_survival_fraction_difference"]),
            "minimum_q_wasserstein": metrics["wasserstein_minimum_sampled_q_AU"]
            <= float(gates["max_wasserstein_minimum_q_AU"]),
            "final_q_wasserstein": metrics["wasserstein_final_bound_q_AU"]
            <= float(gates["max_wasserstein_final_q_AU"]),
            "final_i_wasserstein": metrics["wasserstein_final_bound_i_deg"]
            <= float(gates["max_wasserstein_final_i_deg"]),
        }
        all_checks.extend(checks.values())
        arms[arm] = {"metrics": metrics, "checks": checks, "passed": all(checks.values())}
    primary_effect = _population_summary(primary_selected["source"])["sampled_injection_fraction"] - _population_summary(primary_selected["control"])["sampled_injection_fraction"]
    audit_effect = _population_summary(audit["source"])["sampled_injection_fraction"] - _population_summary(audit["control"])["sampled_injection_fraction"]
    effect_difference = abs(audit_effect - primary_effect)
    effect_passed = effect_difference <= float(gates["max_source_control_effect_difference"])
    all_checks.append(effect_passed)
    return {
        "arms": arms,
        "source_control_effect": {
            "primary_subset": primary_effect,
            "audit": audit_effect,
            "absolute_difference": effect_difference,
            "threshold": float(gates["max_source_control_effect_difference"]),
            "passed": effect_passed,
        },
        "passed": all(all_checks),
    }


def _active_audit(rows: list[dict[str, str]], dt: float, duration: float) -> dict[str, Any]:
    simulation, names = _build_simulation(rows, [], dt)
    initial_energy = simulation.energy()
    initial_angular = simulation.angular_momentum()
    initial_vector = (float(initial_angular.x), float(initial_angular.y), float(initial_angular.z))
    initial_norm = math.sqrt(sum(value * value for value in initial_vector))
    steps = _integer_ratio(duration, dt, "active audit duration/dt")
    max_energy = max_angular = 0.0
    source_index = names.index(SOURCE_NAME) if SOURCE_NAME in names else None
    if source_index is not None:
        source_orbit = simulation.particles[source_index].orbit(primary=simulation.particles[0])
        initial_a = source_orbit.a
        min_q = source_orbit.a * (1.0 - source_orbit.e)
        min_a = max_a = source_orbit.a
        source_bound = True
    else:
        initial_a = min_q = min_a = max_a = None
        source_bound = None
    started = time.perf_counter()
    for _ in range(steps):
        simulation.step()
        max_energy = max(max_energy, abs((simulation.energy() - initial_energy) / initial_energy))
        angular = simulation.angular_momentum()
        delta = (
            float(angular.x) - initial_vector[0],
            float(angular.y) - initial_vector[1],
            float(angular.z) - initial_vector[2],
        )
        max_angular = max(max_angular, math.sqrt(sum(value * value for value in delta)) / initial_norm)
        if source_index is not None:
            orbit = simulation.particles[source_index].orbit(primary=simulation.particles[0])
            bound = orbit.a > 0.0 and orbit.e < 1.0 and math.isfinite(orbit.a) and math.isfinite(orbit.e)
            source_bound = bool(source_bound and bound)
            if bound:
                min_q = min(float(min_q), orbit.a * (1.0 - orbit.e))
                min_a, max_a = min(float(min_a), orbit.a), max(float(max_a), orbit.a)
    excursion = (
        max(abs(float(min_a) - float(initial_a)), abs(float(max_a) - float(initial_a))) / float(initial_a)
        if source_index is not None
        else None
    )
    return {
        "active_bodies": names,
        "steps": steps,
        "dt_years": dt,
        "duration_years": duration,
        "maximum_every_step_relative_energy_drift": max_energy,
        "maximum_every_step_relative_angular_momentum_vector_drift": max_angular,
        "endpoint_active_state": _active_state(simulation),
        "endpoint_active_state_sha256": _canonical_sha256(_active_state(simulation)),
        "cartesian_state_finite": _cartesian_state_is_finite(simulation),
        "source_bound_at_every_step": source_bound,
        "source_minimum_q_AU": min_q,
        "source_maximum_fractional_a_excursion": excursion,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _prepare(contract_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract_data, _identity = _stable_read_bytes(contract_path)
    contract = _decode_json_object(contract_data, "population contract")
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError(f"contract schema must be {CONTRACT_SCHEMA}")
    if contract.get("registration_status") != "PRELOCKED_BEFORE_MODEL_OUTCOMES":
        raise ValueError("contract is not prelocked")
    verified_files = {}
    for label, specification in contract["locked_files"].items():
        path = _resolve(contract_path, specification["path"])
        observed = _sha256_file(path)
        if observed != specification["sha256"]:
            raise ValueError(f"locked file hash mismatch for {label}: {observed}")
        verified_files[label] = {"path": specification["path"], "sha256": observed}
    source_path = _resolve(contract_path, contract["states"]["source_path"])
    control_path = _resolve(contract_path, contract["states"]["control_path"])
    population_path = _resolve(contract_path, contract["population"]["elements_path"])
    for label, path in (("source", source_path), ("control", control_path)):
        if _sha256_file(path) != contract["states"][f"{label}_sha256"]:
            raise ValueError(f"{label} state hash mismatch")
    if _sha256_file(population_path) != contract["population"]["elements_sha256"]:
        raise ValueError("population elements hash mismatch")
    source_rows, control_rows = _state_rows(source_path), _state_rows(control_path)
    for label, path in (("source", source_path), ("control", control_path)):
        if _cached_sha256(path) != contract["states"][f"{label}_sha256"]:
            raise ValueError(f"{label} state changed while it was being loaded")
    matched = _verify_matched_states(source_rows, control_rows)
    population = _load_population(population_path)
    if _cached_sha256(population_path) != contract["population"]["elements_sha256"]:
        raise ValueError("population elements changed while they were being loaded")
    block_count = _positive_int(contract["population"]["blocks"], "population blocks")
    tracers_per_block = _positive_int(
        contract["population"]["tracers_per_block"], "tracers_per_block"
    )
    if sorted(population) != list(range(100)) or any(len(rows) != 1000 for rows in population.values()):
        raise ValueError("locked population file must contain 100 blocks of 1000")
    if block_count > 100 or tracers_per_block > 1000:
        raise ValueError("contract requests more tracers than the locked population")
    import rebound

    wheel = _resolve(contract_path, contract["backend"]["wheel_path"])
    runtime = {
        "version": rebound.__version__,
        "build": rebound.__build__,
        "binary_sha256": _sha256_file(Path(rebound.clibrebound._name)),
        "wheel_sha256": _sha256_file(wheel),
    }
    expected = {
        "version": contract["backend"]["version"],
        "build": contract["backend"]["build"],
        "binary_sha256": contract["backend"]["binary_sha256"],
        "wheel_sha256": contract["backend"]["wheel_sha256"],
    }
    if runtime != expected:
        raise ValueError(f"REBOUND runtime mismatch: {runtime}")
    dynamics = contract["dynamics"]
    if dynamics.get("integrator") != "mercurius" or dynamics.get("G") != 1:
        raise ValueError("runner requires MERCURIUS and G=1")
    for stage_name, stage in contract["stages"].items():
        dt = _positive_float(stage["dt_years"], f"{stage_name} dt")
        sample = _positive_float(stage["sample_years"], f"{stage_name} sample")
        _integer_ratio(sample, dt, f"{stage_name} sample/dt")
    duration = _positive_float(dynamics["duration_years"], "duration")
    aggregate = _positive_float(dynamics["aggregate_years"], "aggregate cadence")
    checkpoint = _positive_float(dynamics["checkpoint_years"], "checkpoint cadence")
    for stage_name, stage in contract["stages"].items():
        sample = float(stage["sample_years"])
        _integer_ratio(duration, sample, f"{stage_name} duration/sample")
        _integer_ratio(aggregate, sample, f"{stage_name} aggregate/sample")
        _integer_ratio(checkpoint, sample, f"{stage_name} checkpoint/sample")
    return contract, {
        "source_path": source_path,
        "control_path": control_path,
        "source_rows": source_rows,
        "control_rows": control_rows,
        "population_path": population_path,
        "matched": matched,
        "runtime": runtime,
        "verified_files": verified_files,
        "contract_sha256": _sha256_file(contract_path),
    }


def _jobs(
    contract: dict[str, Any],
    context: dict[str, Any],
    run_dir: Path,
    stage_name: str,
) -> list[dict[str, Any]]:
    stage = contract["stages"][stage_name]
    block_count = int(
        contract["population"]["blocks"] if stage_name == "primary" else stage["blocks"]
    )
    local_count = int(
        contract["population"]["tracers_per_block"]
        if stage_name == "primary"
        else stage["tracers_per_block"]
    )
    duration = float(contract["dynamics"]["duration_years"])
    aggregate = float(contract["dynamics"]["aggregate_years"])
    checkpoint = float(contract["dynamics"]["checkpoint_years"])
    dt, sample = float(stage["dt_years"]), float(stage["sample_years"])
    q_lower = float(contract["classification"]["q_threshold_AU"]) - float(
        contract["classification"]["q_hysteresis_AU"]
    )
    jobs = []
    for arm, path in (("control", context["control_path"]), ("source", context["source_path"])):
        for block in range(block_count):
            core = {
                "contract_sha256": context["contract_sha256"],
                "arm": arm,
                "stage": stage_name,
                "state_csv": str(path),
                "state_sha256": contract["states"][f"{arm}_sha256"],
                "population_csv": str(context["population_path"]),
                "population_sha256": contract["population"]["elements_sha256"],
                "block_index": block,
                "local_count": local_count,
                "dt_years": dt,
                "sample_years": sample,
                "duration_years": duration,
                "aggregate_years": aggregate,
                "checkpoint_years": checkpoint,
                "total_samples": _integer_ratio(duration, sample, "duration/sample"),
                "aggregate_stride_samples": _integer_ratio(aggregate, sample, "aggregate/sample"),
                "checkpoint_stride_samples": _integer_ratio(checkpoint, sample, "checkpoint/sample"),
                "q_lower": q_lower,
            }
            job = {
                **core,
                "job_sha256": _job_sha256(core),
                "directory": str(run_dir / stage_name / arm / f"block_{block:03d}"),
            }
            _validate_job(job)
            jobs.append(job)
    return jobs


def _initialize_worker() -> None:
    import rebound

    simulation = rebound.Simulation()
    simulation.add(m=1.0)
    simulation.add(primary=simulation.particles[0], m=0.0, a=1.0)


def _execute(jobs: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    results = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_initialize_worker) as executor:
        futures = {executor.submit(_run_block, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            result = future.result()
            result["_runtime_directory"] = job["directory"]
            result["_runtime_job"] = job
            results.append(result)
            print(
                f"[{job['stage']}] {job['arm']} block {job['block_index']:03d} "
                f"complete in {result['elapsed_seconds']:.1f}s",
                flush=True,
            )
    return sorted(results, key=lambda row: (row["stage"], row["arm"], row["block_index"]))


def run_de441_population(
    contract_path: str | Path,
    run_dir: str | Path,
    output_path: str | Path,
    workers: int | None = None,
) -> dict[str, Any]:
    contract_file = Path(contract_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite locked result: {output}")
    contract, context = _prepare(contract_file)
    configured_workers = int(contract["execution"]["workers"])
    worker_count = configured_workers if workers is None else workers
    if worker_count <= 0 or worker_count > configured_workers:
        raise ValueError("workers must be positive and no greater than the prelocked maximum")
    run_root = Path(run_dir).resolve()
    started = time.perf_counter()
    stage_records = {}
    for stage_name in ("primary", "dt_half", "sample_fine"):
        stage_records[stage_name] = _execute(_jobs(contract, context, run_root, stage_name), worker_count)
    rows: dict[str, dict[str, list[dict[str, str]]]] = {}
    for stage_name, records in stage_records.items():
        rows[stage_name] = {"control": [], "source": []}
        for record in records:
            tracer_path = Path(record["_runtime_directory"]) / record["tracer_csv"]
            runtime_job = record["_runtime_job"]
            elements = _load_population(context["population_path"])[runtime_job["block_index"]][
                : runtime_job["local_count"]
            ]
            verified_rows, observed_digest = _validate_tracer_csv(
                tracer_path, runtime_job, elements
            )
            if observed_digest != record["tracer_csv_sha256"]:
                raise RuntimeError("block tracer changed after worker validation")
            rows[stage_name][record["arm"]].extend(verified_rows)
    expected_primary = int(contract["population"]["blocks"]) * int(
        contract["population"]["tracers_per_block"]
    )
    if any(len(rows["primary"][arm]) != expected_primary for arm in ("control", "source")):
        raise RuntimeError("primary population row count is incomplete")
    effects = _paired_effect(
        rows["primary"]["control"],
        rows["primary"]["source"],
        contract["statistics"]["bootstrap_seed"],
        int(contract["statistics"]["bootstrap_repetitions"]),
        float(contract["statistics"]["equivalence_margin"]),
    )
    audit_blocks = int(contract["stages"]["dt_half"]["blocks"])
    primary_subset = {
        arm: [row for row in rows["primary"][arm] if int(row["block_index"]) < audit_blocks]
        for arm in ("control", "source")
    }
    dt_comparison = _audit_comparison(
        primary_subset, rows["dt_half"], contract["gates"]["dt_half_convergence"]
    )
    sample_comparison = _audit_comparison(
        primary_subset, rows["sample_fine"], contract["gates"]["sample_cadence_convergence"]
    )
    active_audits = {
        "control": _active_audit(
            context["control_rows"],
            float(contract["stages"]["primary"]["dt_years"]),
            float(contract["dynamics"]["duration_years"]),
        ),
        "source": _active_audit(
            context["source_rows"],
            float(contract["stages"]["primary"]["dt_years"]),
            float(contract["dynamics"]["duration_years"]),
        ),
    }
    primary_records = stage_records["primary"]
    active_twin_exact = all(
        record["active_endpoint_state_sha256"] == active_audits[record["arm"]]["endpoint_active_state_sha256"]
        for record in primary_records
    )
    all_records = [record for stage in stage_records.values() for record in stage]
    max_energy = max(audit["maximum_every_step_relative_energy_drift"] for audit in active_audits.values())
    max_angular = max(
        audit["maximum_every_step_relative_angular_momentum_vector_drift"]
        for audit in active_audits.values()
    )
    source_audit = active_audits["source"]
    gates = contract["gates"]
    checks = {
        "complete_finite_block_outputs": all(record["cartesian_state_finite"] for record in all_records),
        "checkpoint_restart_exact": all(record["restart_replay_state_hash_exact"] for record in all_records),
        "massless_active_twin_endpoint_exact": active_twin_exact,
        "massive_energy_drift": max_energy <= float(gates["max_relative_massive_energy_drift"]),
        "massive_angular_momentum_drift": max_angular
        <= float(gates["max_relative_massive_angular_momentum_vector_drift"]),
        "source_bound": bool(source_audit["source_bound_at_every_step"]),
        "source_minimum_q": float(source_audit["source_minimum_q_AU"])
        >= float(gates["source_minimum_q_AU"]),
        "source_semimajor_axis_excursion": float(source_audit["source_maximum_fractional_a_excursion"])
        <= float(gates["source_maximum_fractional_a_excursion"]),
        "dt_half_population_convergence": dt_comparison["passed"],
        "sample_cadence_convergence": sample_comparison["passed"],
    }
    verdict = "PASSED" if all(checks.values()) else "INVALID"
    result = {
        "schema": RESULT_SCHEMA,
        "science_verdict": verdict,
        "numerical_status": "VALID" if verdict == "PASSED" else "INVALID",
        "science_status": "SCREENING_ONLY",
        "experiment_id": contract["experiment_id"],
        "contract_path": os.path.relpath(contract_file, output.parent),
        "contract_sha256": context["contract_sha256"],
        "classification": "REAL_EPOCH_DE441_BACKBONE_OUTCOME_BLIND_PROPOSAL",
        "design": {
            "blocks": contract["population"]["blocks"],
            "tracers_per_block": contract["population"]["tracers_per_block"],
            "tracers_per_arm": expected_primary,
            "paired_primary_trajectories": 2 * expected_primary,
            "duration_years": contract["dynamics"]["duration_years"],
            "primary_dt_years": contract["stages"]["primary"]["dt_years"],
            "primary_sample_years": contract["stages"]["primary"]["sample_years"],
        },
        "verified_locked_files": context["verified_files"],
        "matched_state_audit": context["matched"],
        "rebound_runtime": context["runtime"],
        "population_screening": effects,
        "dt_half_convergence": dt_comparison,
        "sample_cadence_convergence": sample_comparison,
        "active_only_every_step_audits": active_audits,
        "maximum_every_step_relative_massive_energy_drift": max_energy,
        "maximum_every_step_relative_massive_angular_momentum_vector_drift": max_angular,
        "checks": checks,
        "all_gates_passed": verdict == "PASSED",
        "block_records": [
            {
                **{
                    key: record[key]
                    for key in (
                        "stage",
                        "arm",
                        "block_index",
                        "tracers",
                        "dt_years",
                        "sample_years",
                        "job_sha256",
                        "block_content_sha256",
                        "tracer_csv_sha256",
                        "summary_json_sha256",
                        "restart_replay_state_hash_exact",
                        "elapsed_seconds",
                    )
                },
                "tracer_csv": os.path.relpath(
                    Path(record["_runtime_directory"]) / record["tracer_csv"], output.parent
                ),
                "summary_json": os.path.relpath(
                    Path(record["_runtime_directory"]) / record["summary_json"], output.parent
                ),
            }
            for record in all_records
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "claim_decision": "SCREENING_ONLY" if verdict == "PASSED" else "INVALID",
        "scientific_scope": contract["scientific_scope"],
        "limitations": contract["limitations"],
        "nonclaim": contract["nonclaim"],
        "next_required_gate": "independent-software replication and an observed-population/selection-function model",
    }
    _write_json_create_only(output, result)
    return result
