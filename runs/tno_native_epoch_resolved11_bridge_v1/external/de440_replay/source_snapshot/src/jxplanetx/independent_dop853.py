"""Independent SciPy DOP853 replication of the JX population screen.

This module deliberately shares no trajectory integrator with the REBOUND
production run.  It implements the Newtonian force function, Kepler conversion,
sampling, checkpoints, comparison, and fail-closed verdict logic locally.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import multiprocessing
import os
import platform
import statistics
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import scipy
from scipy.integrate import DOP853


CONTRACT_SCHEMA = "jx-independent-dop853-contract/v1"
RESULT_SCHEMA = "jx-independent-dop853-result/v1"
BLOCK_SCHEMA = "jx-independent-dop853-block/v1"
CHECKPOINT_SCHEMA = "jx-independent-dop853-checkpoint/v1"
STATE_COLUMNS = ("index", "name", "mass", "x", "y", "z", "vx", "vy", "vz")
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
TRACER_COLUMNS = (
    *POPULATION_COLUMNS,
    "minimum_sampled_q_AU",
    "sampled_injection",
    "first_sampled_low_q_year",
    "ever_unbound_at_sample",
    "first_unbound_year",
    "bound_final",
    "final_q_AU",
    "final_i_deg",
)
COMMON_NAMES = ("Sun", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto")
SOURCE_NAME = "P9_BB21_idx9118"


_FILE_HASH_CACHE: dict[str, str] = {}
_POPULATION_CACHE: dict[str, dict[int, list[dict[str, Any]]]] = {}
_STATE_CACHE: dict[str, list[dict[str, str]]] = {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cached_sha256(path: Path) -> str:
    key = str(path.resolve())
    if key not in _FILE_HASH_CACHE:
        _FILE_HASH_CACHE[key] = _sha256_file(path)
    return _FILE_HASH_CACHE[key]


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _resolve(contract_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (contract_path.parent / path).resolve()


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)


def _atomic_json(path: Path, value: Mapping[str, Any], replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"refusing to overwrite locked JSON: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
                default=_json_default,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_csv(path: Path, rows: list[Mapping[str, Any]], columns: Iterable[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite locked CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_npz(path: Path, state: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint array: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        np.savez(temporary, state=np.ascontiguousarray(state, dtype=np.float64))
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


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


def _load_state(path: Path) -> list[dict[str, str]]:
    key = str(path.resolve())
    if key in _STATE_CACHE:
        return _STATE_CACHE[key]
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != STATE_COLUMNS:
            raise ValueError(f"state CSV must have columns {STATE_COLUMNS}")
        rows = list(reader)
    names = tuple(row["name"] for row in rows)
    if names not in (COMMON_NAMES, (*COMMON_NAMES, SOURCE_NAME)):
        raise ValueError("state body ordering is not the locked control/source model")
    for index, row in enumerate(rows):
        values = [float(row[field]) for field in ("mass", "x", "y", "z", "vx", "vy", "vz")]
        if int(row["index"]) != index or not all(math.isfinite(value) for value in values):
            raise ValueError("invalid state row")
        if float(row["mass"]) <= 0.0:
            raise ValueError("active gravitational parameters must be positive")
    _STATE_CACHE[key] = rows
    return rows


def _load_population(path: Path) -> dict[int, list[dict[str, Any]]]:
    key = str(path.resolve())
    if key in _POPULATION_CACHE:
        return _POPULATION_CACHE[key]
    blocks: dict[int, list[dict[str, Any]]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != POPULATION_COLUMNS:
            raise ValueError(f"population CSV must have columns {POPULATION_COLUMNS}")
        for raw in reader:
            row = {
                "block_index": int(raw["block_index"]),
                "local_index": int(raw["local_index"]),
                "logical_id": raw["logical_id"],
                **{field: float(raw[field]) for field in POPULATION_COLUMNS[3:]},
            }
            if not all(math.isfinite(row[field]) for field in POPULATION_COLUMNS[3:]):
                raise ValueError("population contains a non-finite element")
            if not (
                row["a0_AU"] > row["q0_AU"] > 30.0
                and math.isclose(
                    row["e0"], 1.0 - row["q0_AU"] / row["a0_AU"], rel_tol=0.0, abs_tol=2e-15
                )
            ):
                raise ValueError(f"invalid population element {row['logical_id']}")
            blocks.setdefault(row["block_index"], []).append(row)
    identities: set[str] = set()
    for block, rows in blocks.items():
        rows.sort(key=lambda row: row["local_index"])
        if [row["local_index"] for row in rows] != list(range(len(rows))):
            raise ValueError(f"noncontiguous local indices in block {block}")
        for row in rows:
            if row["logical_id"] in identities:
                raise ValueError(f"duplicate logical ID {row['logical_id']}")
            identities.add(row["logical_id"])
    if sorted(blocks) != list(range(100)) or any(len(rows) != 1000 for rows in blocks.values()):
        raise ValueError("locked population must contain 100 blocks of 1000")
    _POPULATION_CACHE[key] = blocks
    return blocks


def _state_arrays(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masses = np.array([float(row["mass"]) for row in rows], dtype=np.float64)
    positions = np.array([[float(row[field]) for field in ("x", "y", "z")] for row in rows])
    velocities = np.array([[float(row[field]) for field in ("vx", "vy", "vz")] for row in rows])
    return masses, positions, velocities


def _solve_kepler(mean_anomaly: np.ndarray, eccentricity: np.ndarray) -> np.ndarray:
    eccentric_anomaly = np.array(mean_anomaly, dtype=np.float64, copy=True)
    for _ in range(32):
        correction = (
            eccentric_anomaly
            - eccentricity * np.sin(eccentric_anomaly)
            - mean_anomaly
        ) / (1.0 - eccentricity * np.cos(eccentric_anomaly))
        eccentric_anomaly -= correction
        if float(np.max(np.abs(correction))) < 2e-15:
            break
    residual = np.max(
        np.abs(eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - mean_anomaly)
    )
    if not math.isfinite(float(residual)) or float(residual) > 2e-14:
        raise ArithmeticError(f"Kepler solve did not converge: residual={residual}")
    return eccentric_anomaly


def _elements_to_cartesian(
    elements: list[dict[str, Any]],
    primary_position: np.ndarray,
    primary_velocity: np.ndarray,
    primary_gm: float,
) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([row["a0_AU"] for row in elements])
    e = np.array([row["e0"] for row in elements])
    inclination = np.radians([row["i0_deg"] for row in elements])
    ascending = np.array([row["Omega0_rad"] for row in elements])
    periapse = np.array([row["omega0_rad"] for row in elements])
    mean = np.array([row["M0_rad"] for row in elements])
    anomaly = _solve_kepler(mean, e)
    cosine, sine = np.cos(anomaly), np.sin(anomaly)
    root = np.sqrt(1.0 - e * e)
    x_orbit = a * (cosine - e)
    y_orbit = a * root * sine
    mean_motion = np.sqrt(primary_gm / (a * a * a))
    denominator = 1.0 - e * cosine
    vx_orbit = -a * mean_motion * sine / denominator
    vy_orbit = a * mean_motion * root * cosine / denominator
    c_node, s_node = np.cos(ascending), np.sin(ascending)
    c_peri, s_peri = np.cos(periapse), np.sin(periapse)
    c_inc, s_inc = np.cos(inclination), np.sin(inclination)
    p_vector = np.column_stack(
        (
            c_node * c_peri - s_node * s_peri * c_inc,
            s_node * c_peri + c_node * s_peri * c_inc,
            s_peri * s_inc,
        )
    )
    q_vector = np.column_stack(
        (
            -c_node * s_peri - s_node * c_peri * c_inc,
            -s_node * s_peri + c_node * c_peri * c_inc,
            c_peri * s_inc,
        )
    )
    positions = x_orbit[:, None] * p_vector + y_orbit[:, None] * q_vector
    velocities = vx_orbit[:, None] * p_vector + vy_orbit[:, None] * q_vector
    return positions + primary_position, velocities + primary_velocity


def _orbital_elements(
    state: np.ndarray, active_count: int, primary_gm: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    particle_count = state.size // 6
    positions = state[: 3 * particle_count].reshape(particle_count, 3)
    velocities = state[3 * particle_count :].reshape(particle_count, 3)
    relative_position = positions[active_count:] - positions[0]
    relative_velocity = velocities[active_count:] - velocities[0]
    radius = np.linalg.norm(relative_position, axis=1)
    speed2 = np.sum(relative_velocity * relative_velocity, axis=1)
    specific_energy = 0.5 * speed2 - primary_gm / radius
    semimajor = np.full_like(specific_energy, np.nan)
    bound_energy = specific_energy < 0.0
    semimajor[bound_energy] = -primary_gm / (2.0 * specific_energy[bound_energy])
    angular = np.cross(relative_position, relative_velocity)
    angular_norm = np.linalg.norm(angular, axis=1)
    eccentricity_vector = (
        np.cross(relative_velocity, angular) / primary_gm
        - relative_position / radius[:, None]
    )
    eccentricity = np.linalg.norm(eccentricity_vector, axis=1)
    bound = (
        bound_energy
        & (eccentricity < 1.0)
        & np.isfinite(semimajor)
        & np.isfinite(eccentricity)
        & (angular_norm > 0.0)
    )
    perihelion = np.full_like(semimajor, np.nan)
    perihelion[bound] = semimajor[bound] * (1.0 - eccentricity[bound])
    inclination = np.full_like(semimajor, np.nan)
    cosine = np.clip(angular[:, 2] / angular_norm, -1.0, 1.0)
    inclination[bound] = np.degrees(np.arccos(cosine[bound]))
    return perihelion, inclination, bound


def _rhs_factory(masses: np.ndarray, tracer_count: int):
    active_count = len(masses)
    particle_count = active_count + tracer_count
    position_size = 3 * particle_count

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        positions = state[:position_size].reshape(particle_count, 3)
        velocities = state[position_size:].reshape(particle_count, 3)
        active = positions[:active_count]
        delta_active = active[None, :, :] - active[:, None, :]
        radius2_active = np.sum(delta_active * delta_active, axis=2)
        np.fill_diagonal(radius2_active, np.inf)
        inverse_active = radius2_active ** -1.5
        acceleration_active = np.sum(
            delta_active * inverse_active[:, :, None] * masses[None, :, None], axis=1
        )
        acceleration = np.empty_like(positions)
        acceleration[:active_count] = acceleration_active
        if tracer_count:
            tracer = positions[active_count:]
            delta_tracer = active[None, :, :] - tracer[:, None, :]
            radius2_tracer = np.sum(delta_tracer * delta_tracer, axis=2)
            if np.any(radius2_tracer <= 0.0):
                raise ZeroDivisionError("active/tracer collision in independent force function")
            inverse_tracer = radius2_tracer ** -1.5
            acceleration[active_count:] = np.sum(
                delta_tracer * inverse_tracer[:, :, None] * masses[None, :, None], axis=1
            )
        derivative = np.empty_like(state)
        derivative[:position_size] = velocities.ravel()
        derivative[position_size:] = acceleration.ravel()
        return derivative

    return rhs


def _active_invariants(state: np.ndarray, masses: np.ndarray) -> tuple[float, np.ndarray]:
    count = len(masses)
    positions = state[: 3 * count].reshape(count, 3)
    velocities = state[3 * count : 6 * count].reshape(count, 3)
    energy = 0.5 * float(np.sum(masses[:, None] * velocities * velocities))
    angular = np.sum(masses[:, None] * np.cross(positions, velocities), axis=0)
    for left in range(count):
        for right in range(left + 1, count):
            energy -= masses[left] * masses[right] / float(
                np.linalg.norm(positions[right] - positions[left])
            )
    return energy, angular


def _active_slice(state: np.ndarray, active_count: int) -> np.ndarray:
    particle_count = state.size // 6
    positions = state[: 3 * particle_count].reshape(particle_count, 3)[:active_count]
    velocities = state[3 * particle_count :].reshape(particle_count, 3)[:active_count]
    return np.concatenate((positions.ravel(), velocities.ravel()))


def _active_state_record(state: np.ndarray, active_count: int) -> list[dict[str, Any]]:
    active = _active_slice(state, active_count)
    positions = active[: 3 * active_count].reshape(active_count, 3)
    velocities = active[3 * active_count :].reshape(active_count, 3)
    return [
        {
            "index": index,
            "position_AU": [float(value) for value in positions[index]],
            "velocity_AU_per_year": [float(value) for value in velocities[index]],
        }
        for index in range(active_count)
    ]


def _source_orbit(state: np.ndarray, active_count: int, masses: np.ndarray) -> tuple[float, float, bool]:
    if active_count != len(COMMON_NAMES) + 1:
        return math.nan, math.nan, False
    positions = state[: 3 * active_count].reshape(active_count, 3)
    velocities = state[3 * active_count : 6 * active_count].reshape(active_count, 3)
    relative_position = positions[-1] - positions[0]
    relative_velocity = velocities[-1] - velocities[0]
    radius = float(np.linalg.norm(relative_position))
    speed2 = float(np.dot(relative_velocity, relative_velocity))
    mu = float(masses[0] + masses[-1])
    energy = 0.5 * speed2 - mu / radius
    if not math.isfinite(energy) or energy >= 0.0:
        return math.nan, math.nan, False
    semimajor = -mu / (2.0 * energy)
    angular = np.cross(relative_position, relative_velocity)
    eccentricity_vector = np.cross(relative_velocity, angular) / mu - relative_position / radius
    eccentricity = float(np.linalg.norm(eccentricity_vector))
    bound = eccentricity < 1.0 and math.isfinite(eccentricity)
    return semimajor, semimajor * (1.0 - eccentricity), bound


def _initial_state(
    rows: list[dict[str, str]], elements: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray, int, dict[str, float]]:
    masses, positions, velocities = _state_arrays(rows)
    tracer_positions, tracer_velocities = _elements_to_cartesian(
        elements, positions[0], velocities[0], float(masses[0])
    )
    all_positions = np.vstack((positions, tracer_positions))
    all_velocities = np.vstack((velocities, tracer_velocities))
    state = np.concatenate((all_positions.ravel(), all_velocities.ravel()))
    perihelion, inclination, bound = _orbital_elements(state, len(rows), float(masses[0]))
    expected_q = np.array([row["q0_AU"] for row in elements])
    expected_i = np.array([row["i0_deg"] for row in elements])
    expected_a = np.array([row["a0_AU"] for row in elements])
    relative_position = tracer_positions - positions[0]
    relative_velocity = tracer_velocities - velocities[0]
    radius = np.linalg.norm(relative_position, axis=1)
    energy = 0.5 * np.sum(relative_velocity * relative_velocity, axis=1) - masses[0] / radius
    reconstructed_a = -masses[0] / (2.0 * energy)
    roundtrip = {
        "maximum_absolute_a_error_AU": float(np.max(np.abs(reconstructed_a - expected_a))),
        "maximum_absolute_q_error_AU": float(np.max(np.abs(perihelion - expected_q))),
        "maximum_absolute_i_error_deg": float(np.max(np.abs(inclination - expected_i))),
    }
    if not bool(np.all(bound)) or not np.all(np.isfinite(state)):
        raise ValueError("initial independent state is not finite and bound")
    return state, masses, len(rows), roundtrip


def _blank_tracker(elements: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "minimum_q": np.array([row["q0_AU"] for row in elements], dtype=np.float64),
        "first_low_q_year": [None] * len(elements),
        "ever_unbound": np.zeros(len(elements), dtype=bool),
        "first_unbound_year": [None] * len(elements),
        "sample_count": 0,
    }


def _sample_state(
    state: np.ndarray,
    sample_time: float,
    active_count: int,
    primary_gm: float,
    tracker: dict[str, Any],
    q_lower: float,
) -> None:
    perihelion, _inclination, bound = _orbital_elements(state, active_count, primary_gm)
    bound_indices = np.flatnonzero(bound)
    tracker["minimum_q"][bound_indices] = np.minimum(
        tracker["minimum_q"][bound_indices], perihelion[bound_indices]
    )
    newly_low = bound & (perihelion < q_lower)
    for index in np.flatnonzero(newly_low):
        if tracker["first_low_q_year"][int(index)] is None:
            tracker["first_low_q_year"][int(index)] = sample_time
    newly_unbound = ~bound
    tracker["ever_unbound"] |= newly_unbound
    for index in np.flatnonzero(newly_unbound):
        if tracker["first_unbound_year"][int(index)] is None:
            tracker["first_unbound_year"][int(index)] = sample_time
    tracker["sample_count"] += 1


def _tracker_json(tracker: dict[str, Any]) -> dict[str, Any]:
    return {
        "minimum_q": [float(value) for value in tracker["minimum_q"]],
        "first_low_q_year": tracker["first_low_q_year"],
        "ever_unbound": [bool(value) for value in tracker["ever_unbound"]],
        "first_unbound_year": tracker["first_unbound_year"],
        "sample_count": int(tracker["sample_count"]),
    }


def _tracker_from_json(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "minimum_q": np.array(value["minimum_q"], dtype=np.float64),
        "first_low_q_year": list(value["first_low_q_year"]),
        "ever_unbound": np.array(value["ever_unbound"], dtype=bool),
        "first_unbound_year": list(value["first_unbound_year"]),
        "sample_count": int(value["sample_count"]),
    }


def _checkpoint_paths(directory: Path, index: int) -> tuple[Path, Path]:
    return directory / f"checkpoint_{index:03d}.npz", directory / f"checkpoint_{index:03d}.json"


def _write_checkpoint(
    directory: Path,
    index: int,
    time_year: float,
    state: np.ndarray,
    tracker: dict[str, Any],
    segment_records: list[dict[str, Any]],
    job_sha256: str,
) -> bool:
    array_path, json_path = _checkpoint_paths(directory, index)
    _atomic_npz(array_path, state)
    with np.load(array_path, allow_pickle=False) as archive:
        replay = np.array(archive["state"], dtype=np.float64, copy=True)
    exact = np.array_equal(state, replay) and _array_sha256(state) == _array_sha256(replay)
    if not exact:
        raise RuntimeError("independent checkpoint replay changed the binary64 state")
    record = {
        "schema": CHECKPOINT_SCHEMA,
        "checkpoint_index": index,
        "job_sha256": job_sha256,
        "time_year": time_year,
        "state_npz_sha256": _sha256_file(array_path),
        "state_array_sha256": _array_sha256(state),
        "state_shape": list(state.shape),
        "tracker": _tracker_json(tracker),
        "completed_segment_records": segment_records,
    }
    _atomic_json(json_path, record)
    return exact


def _load_checkpoint(
    directory: Path, job_sha256: str
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]], int, float] | None:
    for json_path in sorted(directory.glob("checkpoint_*.json"), reverse=True):
        try:
            record = _read_json(json_path)
            index = int(record["checkpoint_index"])
            array_path, expected_json = _checkpoint_paths(directory, index)
            if (
                record["schema"] != CHECKPOINT_SCHEMA
                or record["job_sha256"] != job_sha256
                or json_path != expected_json
                or not array_path.is_file()
                or _sha256_file(array_path) != record["state_npz_sha256"]
            ):
                continue
            with np.load(array_path, allow_pickle=False) as archive:
                state = np.array(archive["state"], dtype=np.float64, copy=True)
            if _array_sha256(state) != record["state_array_sha256"]:
                continue
            return (
                state,
                _tracker_from_json(record["tracker"]),
                list(record["completed_segment_records"]),
                index,
                float(record["time_year"]),
            )
        except (KeyError, ValueError, OSError, json.JSONDecodeError):
            continue
    return None


def _integrate_segment(
    state: np.ndarray,
    tracker: dict[str, Any],
    start_time: float,
    end_time: float,
    sample_years: float,
    rhs: Any,
    masses: np.ndarray,
    active_count: int,
    q_lower: float,
    settings: Mapping[str, Any],
    invariant_reference: tuple[float, np.ndarray],
    source_reference_a: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    solver = DOP853(
        rhs,
        start_time,
        state,
        end_time,
        rtol=float(settings["rtol"]),
        atol=float(settings["atol"]),
        max_step=float(settings["max_step_years"]),
    )
    expected_samples_at_start = _integer_ratio(start_time, sample_years, "segment start/sample") + 1
    if tracker["sample_count"] != expected_samples_at_start:
        raise RuntimeError("tracker sample count does not match segment start")
    next_sample = tracker["sample_count"] * sample_years
    initial_energy, initial_angular = invariant_reference
    angular_norm = float(np.linalg.norm(initial_angular))
    maximum_energy = maximum_angular = 0.0
    minimum_step = math.inf
    maximum_step = 0.0
    accepted_steps = 0
    source_minimum_q = math.inf
    source_minimum_a = math.inf
    source_maximum_a = -math.inf
    source_bound = True if source_reference_a is not None else None
    while solver.status == "running":
        message = solver.step()
        if solver.status == "failed":
            raise RuntimeError(f"DOP853 failed: {message}")
        accepted_steps += 1
        step_size = float(solver.step_size)
        minimum_step = min(minimum_step, step_size)
        maximum_step = max(maximum_step, step_size)
        active_state = _active_slice(solver.y, active_count)
        energy, angular = _active_invariants(active_state, masses)
        maximum_energy = max(maximum_energy, abs((energy - initial_energy) / initial_energy))
        maximum_angular = max(
            maximum_angular, float(np.linalg.norm(angular - initial_angular)) / angular_norm
        )
        if source_reference_a is not None:
            source_a, source_q, bound = _source_orbit(active_state, active_count, masses)
            source_bound = bool(source_bound and bound)
            if bound:
                source_minimum_q = min(source_minimum_q, source_q)
                source_minimum_a = min(source_minimum_a, source_a)
                source_maximum_a = max(source_maximum_a, source_a)
        if next_sample <= solver.t + 2e-12:
            dense = solver.dense_output()
            while next_sample <= solver.t + 2e-12 and next_sample <= end_time + 2e-12:
                sampled_state = np.asarray(dense(next_sample), dtype=np.float64)
                _sample_state(
                    sampled_state,
                    next_sample,
                    active_count,
                    float(masses[0]),
                    tracker,
                    q_lower,
                )
                next_sample += sample_years
    if not math.isclose(float(solver.t), end_time, rel_tol=0.0, abs_tol=2e-10):
        raise RuntimeError("DOP853 segment did not reach its contracted endpoint")
    source_excursion = None
    if source_reference_a is not None:
        source_excursion = max(
            abs(source_minimum_a - source_reference_a), abs(source_maximum_a - source_reference_a)
        ) / source_reference_a
    return np.asarray(solver.y, dtype=np.float64), {
        "accepted_steps": accepted_steps,
        "rhs_evaluations": int(solver.nfev),
        "minimum_accepted_step_years": minimum_step,
        "maximum_accepted_step_years": maximum_step,
        "maximum_relative_active_energy_drift": maximum_energy,
        "maximum_relative_active_angular_momentum_vector_drift": maximum_angular,
        "source_bound_at_every_accepted_step": source_bound,
        "source_minimum_q_AU": source_minimum_q if source_reference_a is not None else None,
        "source_maximum_fractional_a_excursion": source_excursion,
    }


def _final_rows(
    state: np.ndarray,
    active_count: int,
    elements: list[dict[str, Any]],
    tracker: dict[str, Any],
    primary_gm: float,
    q_lower: float,
) -> list[dict[str, Any]]:
    perihelion, inclination, bound = _orbital_elements(state, active_count, primary_gm)
    rows = []
    for index, element in enumerate(elements):
        rows.append(
            {
                **element,
                "minimum_sampled_q_AU": float(tracker["minimum_q"][index]),
                "sampled_injection": int(tracker["minimum_q"][index] < q_lower),
                "first_sampled_low_q_year": tracker["first_low_q_year"][index],
                "ever_unbound_at_sample": int(tracker["ever_unbound"][index]),
                "first_unbound_year": tracker["first_unbound_year"][index],
                "bound_final": int(bound[index]),
                "final_q_AU": float(perihelion[index]) if bound[index] else None,
                "final_i_deg": float(inclination[index]) if bound[index] else None,
            }
        )
    return rows


def _run_block(job: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / "summary.json"
    tracer_path = directory / "tracers.csv"
    if summary_path.is_file():
        summary = _read_json(summary_path)
        if (
            summary.get("schema") == BLOCK_SCHEMA
            and summary.get("job_sha256") == job["job_sha256"]
            and tracer_path.is_file()
            and _sha256_file(tracer_path) == summary.get("tracer_csv_sha256")
        ):
            summary["resumed_from_complete_block"] = True
            summary["summary_json_sha256"] = _sha256_file(summary_path)
            return summary
        raise RuntimeError(f"locked block conflicts with requested job: {directory}")
    if _cached_sha256(Path(job["state_csv"])) != job["state_sha256"]:
        raise RuntimeError("worker state hash mismatch")
    if _cached_sha256(Path(job["population_csv"])) != job["population_sha256"]:
        raise RuntimeError("worker population hash mismatch")
    rows = _load_state(Path(job["state_csv"]))
    elements = _load_population(Path(job["population_csv"]))[job["block_index"]]
    initial, masses, active_count, roundtrip = _initial_state(rows, elements)
    rhs = _rhs_factory(masses, len(elements))
    initial_active = _active_slice(initial, active_count)
    invariant_reference = _active_invariants(initial_active, masses)
    source_initial_a = None
    if job["arm"] == "source":
        source_initial_a, _source_q, source_bound = _source_orbit(initial_active, active_count, masses)
        if not source_bound:
            raise RuntimeError("source is unbound at the initial epoch")
    loaded = _load_checkpoint(directory, job["job_sha256"])
    replay_exact = True
    if loaded is None:
        state = initial
        tracker = _blank_tracker(elements)
        _sample_state(state, 0.0, active_count, float(masses[0]), tracker, job["q_lower"])
        checkpoint_index = 0
        current_time = 0.0
        segment_records: list[dict[str, Any]] = []
        replay_exact = _write_checkpoint(
            directory,
            checkpoint_index,
            current_time,
            state,
            tracker,
            segment_records,
            job["job_sha256"],
        )
    else:
        state, tracker, segment_records, checkpoint_index, current_time = loaded
        if state.shape != initial.shape or tracker["minimum_q"].shape != (len(elements),):
            raise RuntimeError("checkpoint shape mismatch")
    del initial
    checkpoint_count = _integer_ratio(
        job["duration_years"], job["checkpoint_years"], "duration/checkpoint"
    )
    for target_index in range(checkpoint_index + 1, checkpoint_count + 1):
        target_time = target_index * job["checkpoint_years"]
        state, segment = _integrate_segment(
            state,
            tracker,
            current_time,
            target_time,
            job["sample_years"],
            rhs,
            masses,
            active_count,
            job["q_lower"],
            job["solver"],
            invariant_reference,
            source_initial_a,
        )
        segment_records.append({"segment_index": target_index, **segment})
        current_time = target_time
        replay_exact = _write_checkpoint(
            directory,
            target_index,
            current_time,
            state,
            tracker,
            segment_records,
            job["job_sha256"],
        ) and replay_exact
    expected_samples = _integer_ratio(
        job["duration_years"], job["sample_years"], "duration/sample"
    ) + 1
    if tracker["sample_count"] != expected_samples:
        raise RuntimeError("independent block sample count is incomplete")
    final_rows = _final_rows(
        state,
        active_count,
        elements,
        tracker,
        float(masses[0]),
        job["q_lower"],
    )
    if len(final_rows) != 1000 or not np.all(np.isfinite(state)):
        raise RuntimeError("independent block is incomplete or non-finite")
    _atomic_csv(tracer_path, final_rows, TRACER_COLUMNS)
    maximum_energy = max(
        (record["maximum_relative_active_energy_drift"] for record in segment_records),
        default=0.0,
    )
    maximum_angular = max(
        (
            record["maximum_relative_active_angular_momentum_vector_drift"]
            for record in segment_records
        ),
        default=0.0,
    )
    source_q_values = [
        record["source_minimum_q_AU"]
        for record in segment_records
        if record["source_minimum_q_AU"] is not None
    ]
    source_excursions = [
        record["source_maximum_fractional_a_excursion"]
        for record in segment_records
        if record["source_maximum_fractional_a_excursion"] is not None
    ]
    summary = {
        "schema": BLOCK_SCHEMA,
        "job_sha256": job["job_sha256"],
        "arm": job["arm"],
        "block_index": job["block_index"],
        "tracers": len(elements),
        "active_bodies": [row["name"] for row in rows],
        "duration_years": job["duration_years"],
        "sample_years": job["sample_years"],
        "solver": job["solver"],
        "sample_count_including_t0": tracker["sample_count"],
        "checkpoint_epochs_including_t0": checkpoint_count + 1,
        "checkpoint_replay_binary64_exact": replay_exact,
        "initial_element_roundtrip": roundtrip,
        "state_finite": True,
        "maximum_relative_active_energy_drift": maximum_energy,
        "maximum_relative_active_angular_momentum_vector_drift": maximum_angular,
        "source_bound_at_every_accepted_step": (
            all(record["source_bound_at_every_accepted_step"] for record in segment_records)
            if job["arm"] == "source"
            else None
        ),
        "source_minimum_q_AU": min(source_q_values) if source_q_values else None,
        "source_maximum_fractional_a_excursion": (
            max(source_excursions) if source_excursions else None
        ),
        "active_endpoint_state": _active_state_record(state, active_count),
        "active_endpoint_state_sha256": _canonical_sha256(
            _active_state_record(state, active_count)
        ),
        "segments": segment_records,
        "tracer_csv": str(tracer_path),
        "tracer_csv_sha256": _sha256_file(tracer_path),
        "summary_json": str(summary_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(summary_path, summary)
    summary["summary_json_sha256"] = _sha256_file(summary_path)
    return summary


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _population_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    count = len(rows)
    injections = sum(int(row["sampled_injection"]) for row in rows)
    bound = sum(int(row["bound_final"]) for row in rows)
    return {
        "tracers": count,
        "sampled_injections": injections,
        "sampled_injection_fraction": injections / count,
        "bound_final": bound,
        "survival_fraction": bound / count,
    }


def _float_values(rows: list[dict[str, str]], field: str, bound_only: bool = False) -> list[float]:
    result = []
    for row in rows:
        if bound_only and not int(row["bound_final"]):
            continue
        value = float(row[field])
        if math.isfinite(value):
            result.append(value)
    return result


def _wasserstein(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return math.inf
    mass_left: dict[float, float] = {}
    mass_right: dict[float, float] = {}
    for value in left:
        mass_left[value] = mass_left.get(value, 0.0) + 1.0 / len(left)
    for value in right:
        mass_right[value] = mass_right.get(value, 0.0) + 1.0 / len(right)
    support = sorted(set(mass_left) | set(mass_right))
    cdf_left = cdf_right = distance = 0.0
    previous = support[0]
    for value in support:
        distance += abs(cdf_left - cdf_right) * (value - previous)
        cdf_left += mass_left.get(value, 0.0)
        cdf_right += mass_right.get(value, 0.0)
        previous = value
    return distance


def _compare_rows(
    reference: list[dict[str, str]], independent: list[dict[str, str]]
) -> dict[str, Any]:
    reference_by_id = {row["logical_id"]: row for row in reference}
    independent_by_id = {row["logical_id"]: row for row in independent}
    if set(reference_by_id) != set(independent_by_id):
        raise ValueError("reference and independent identity sets differ")
    initial_fields = POPULATION_COLUMNS
    for identity in reference_by_id:
        if any(
            reference_by_id[identity][field] != independent_by_id[identity][field]
            for field in initial_fields
        ):
            raise ValueError(f"paired initial metadata differs for {identity}")
    ref_summary = _population_summary(reference)
    ind_summary = _population_summary(independent)
    disagreements = sum(
        reference_by_id[identity]["sampled_injection"]
        != independent_by_id[identity]["sampled_injection"]
        for identity in reference_by_id
    )
    count = len(reference_by_id)
    return {
        "reference": ref_summary,
        "independent": ind_summary,
        "metrics": {
            "absolute_injection_fraction_difference": abs(
                ind_summary["sampled_injection_fraction"]
                - ref_summary["sampled_injection_fraction"]
            ),
            "injection_identity_disagreement_fraction": disagreements / count,
            "absolute_survival_fraction_difference": abs(
                ind_summary["survival_fraction"] - ref_summary["survival_fraction"]
            ),
            "wasserstein_minimum_sampled_q_AU": _wasserstein(
                _float_values(reference, "minimum_sampled_q_AU"),
                _float_values(independent, "minimum_sampled_q_AU"),
            ),
            "wasserstein_final_bound_q_AU": _wasserstein(
                _float_values(reference, "final_q_AU", True),
                _float_values(independent, "final_q_AU", True),
            ),
            "wasserstein_final_bound_i_deg": _wasserstein(
                _float_values(reference, "final_i_deg", True),
                _float_values(independent, "final_i_deg", True),
            ),
        },
    }


def _bootstrap_ci(values: list[float], seed: str, repetitions: int) -> list[float]:
    estimates = []
    for repetition in range(repetitions):
        draw = []
        for index in range(len(values)):
            message = f"jx-independent-paired-bootstrap/v1\x1f{seed}\x1f{repetition}\x1f{index}".encode()
            selected = int.from_bytes(hashlib.sha256(message).digest()[:8], "big") % len(values)
            draw.append(values[selected])
        estimates.append(statistics.fmean(draw))
    estimates.sort()

    def quantile(probability: float) -> float:
        position = probability * (len(estimates) - 1)
        lower, upper = math.floor(position), math.ceil(position)
        if lower == upper:
            return estimates[lower]
        fraction = position - lower
        return estimates[lower] * (1.0 - fraction) + estimates[upper] * fraction

    return [quantile(0.025), quantile(0.975)]


def _effect(
    control: list[dict[str, str]],
    source: list[dict[str, str]],
    blocks: list[int],
    seed: str,
    repetitions: int,
    margin: float,
) -> dict[str, Any]:
    control_by_id = {row["logical_id"]: row for row in control}
    source_by_id = {row["logical_id"]: row for row in source}
    if set(control_by_id) != set(source_by_id):
        raise ValueError("source/control identity sets differ")
    block_effects = []
    for block in blocks:
        c = [row for row in control if int(row["block_index"]) == block]
        s = [row for row in source if int(row["block_index"]) == block]
        block_effects.append(
            _population_summary(s)["sampled_injection_fraction"]
            - _population_summary(c)["sampled_injection_fraction"]
        )
    interval = _bootstrap_ci(block_effects, seed, repetitions)
    if interval[0] > 0.0:
        classification = "RESOLVED_POSITIVE_SOURCE_EFFECT"
    elif interval[1] < 0.0:
        classification = "RESOLVED_NEGATIVE_SOURCE_EFFECT"
    elif interval[0] >= -margin and interval[1] <= margin:
        classification = "EQUIVALENT_WITHIN_LOCKED_MARGIN"
    else:
        classification = "NO_RESOLVED_EFFECT"
    control_summary = _population_summary(control)
    source_summary = _population_summary(source)
    return {
        "control": control_summary,
        "source": source_summary,
        "source_minus_control_injection_fraction": source_summary["sampled_injection_fraction"]
        - control_summary["sampled_injection_fraction"],
        "block_effects": block_effects,
        "paired_block_bootstrap_95_percent_CI": interval,
        "equivalence_margin": margin,
        "classification": classification,
    }


def _reference_rows(
    reference_result: dict[str, Any],
    reference_root: Path,
    blocks: list[int],
    arm: str,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    records = {
        (record["stage"], record["arm"], int(record["block_index"])): record
        for record in reference_result["block_records"]
    }
    rows = []
    verified = []
    for block in blocks:
        record = records.get(("primary", arm, block))
        if record is None:
            raise ValueError(f"missing reference record for {arm} block {block}")
        path = Path(record["tracer_csv"]).resolve()
        expected = reference_root / "primary" / arm / f"block_{block:03d}" / "tracers.csv"
        if path != expected or not path.is_file():
            raise ValueError(f"reference path mismatch for {arm} block {block}")
        observed = _sha256_file(path)
        if observed != record["tracer_csv_sha256"]:
            raise ValueError(f"reference tracer hash mismatch for {arm} block {block}")
        block_rows = _read_csv(path)
        if len(block_rows) != 1000:
            raise ValueError("reference block row count mismatch")
        rows.extend(block_rows)
        verified.append({"block_index": block, "path": str(path), "sha256": observed})
    return rows, verified


def _active_only_audit(
    rows: list[dict[str, str]], duration: float, settings: Mapping[str, Any]
) -> dict[str, Any]:
    masses, positions, velocities = _state_arrays(rows)
    active_count = len(rows)
    state = np.concatenate((positions.ravel(), velocities.ravel()))
    rhs = _rhs_factory(masses, 0)
    energy0, angular0 = _active_invariants(state, masses)
    angular_norm = float(np.linalg.norm(angular0))
    source_initial_a = None
    if active_count == len(COMMON_NAMES) + 1:
        source_initial_a, _q, bound = _source_orbit(state, active_count, masses)
        if not bound:
            raise RuntimeError("source is initially unbound in active audit")
    maximum_energy = maximum_angular = 0.0
    source_minimum_q = math.inf
    source_minimum_a = math.inf
    source_maximum_a = -math.inf
    source_bound = True if source_initial_a is not None else None
    accepted_steps = rhs_evaluations = 0
    started = time.perf_counter()
    checkpoint = float(settings["active_audit_segment_years"])
    segment_count = _integer_ratio(duration, checkpoint, "active duration/segment")
    current = 0.0
    for segment in range(1, segment_count + 1):
        target = segment * checkpoint
        solver = DOP853(
            rhs,
            current,
            state,
            target,
            rtol=float(settings["rtol"]),
            atol=float(settings["atol"]),
            max_step=float(settings["max_step_years"]),
        )
        while solver.status == "running":
            message = solver.step()
            if solver.status == "failed":
                raise RuntimeError(f"active DOP853 audit failed: {message}")
            accepted_steps += 1
            energy, angular = _active_invariants(solver.y, masses)
            maximum_energy = max(maximum_energy, abs((energy - energy0) / energy0))
            maximum_angular = max(
                maximum_angular, float(np.linalg.norm(angular - angular0)) / angular_norm
            )
            if source_initial_a is not None:
                source_a, source_q, bound = _source_orbit(solver.y, active_count, masses)
                source_bound = bool(source_bound and bound)
                if bound:
                    source_minimum_q = min(source_minimum_q, source_q)
                    source_minimum_a = min(source_minimum_a, source_a)
                    source_maximum_a = max(source_maximum_a, source_a)
        rhs_evaluations += int(solver.nfev)
        state = np.asarray(solver.y, dtype=np.float64)
        current = target
    source_excursion = None
    if source_initial_a is not None:
        source_excursion = max(
            abs(source_minimum_a - source_initial_a), abs(source_maximum_a - source_initial_a)
        ) / source_initial_a
    return {
        "active_bodies": [row["name"] for row in rows],
        "duration_years": duration,
        "accepted_steps": accepted_steps,
        "rhs_evaluations": rhs_evaluations,
        "maximum_relative_energy_drift": maximum_energy,
        "maximum_relative_angular_momentum_vector_drift": maximum_angular,
        "source_bound_at_every_accepted_step": source_bound,
        "source_minimum_q_AU": source_minimum_q if source_initial_a is not None else None,
        "source_maximum_fractional_a_excursion": source_excursion,
        "endpoint_state": _active_state_record(state, active_count),
        "endpoint_state_sha256": _canonical_sha256(_active_state_record(state, active_count)),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _endpoint_disagreement(
    records: list[dict[str, Any]], active_audit: dict[str, Any]
) -> dict[str, float]:
    reference = active_audit["endpoint_state"]
    maximum_position = maximum_velocity = 0.0
    for record in records:
        state = record["active_endpoint_state"]
        if len(state) != len(reference):
            raise ValueError("active endpoint body count differs")
        for left, right in zip(state, reference):
            position = math.sqrt(
                sum(
                    (float(left["position_AU"][axis]) - float(right["position_AU"][axis])) ** 2
                    for axis in range(3)
                )
            )
            velocity = math.sqrt(
                sum(
                    (
                        float(left["velocity_AU_per_year"][axis])
                        - float(right["velocity_AU_per_year"][axis])
                    )
                    ** 2
                    for axis in range(3)
                )
            )
            maximum_position = max(maximum_position, position)
            maximum_velocity = max(maximum_velocity, velocity)
    return {
        "maximum_active_endpoint_position_disagreement_AU": maximum_position,
        "maximum_active_endpoint_velocity_disagreement_AU_per_year": maximum_velocity,
    }


def _runtime_manifest(contract_path: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    import scipy.integrate._ivp.dop853_coefficients as coefficients
    import scipy.integrate._ivp.rk as rk

    numpy_binary = Path(np._core._multiarray_umath.__file__).resolve()
    files = {
        "scipy_rk_source": Path(rk.__file__).resolve(),
        "scipy_dop853_coefficients": Path(coefficients.__file__).resolve(),
        "numpy_multiarray_binary": numpy_binary,
    }
    observed = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "files": {
            label: {"path": str(path), "sha256": _sha256_file(path)}
            for label, path in files.items()
        },
    }
    expected = contract["backend"]
    if (
        observed["python_version"] != expected["python_version"]
        or observed["python_implementation"] != expected["python_implementation"]
        or observed["numpy_version"] != expected["numpy_version"]
        or observed["scipy_version"] != expected["scipy_version"]
    ):
        raise ValueError(f"independent runtime version mismatch: {observed}")
    for label, record in observed["files"].items():
        if record["sha256"] != expected["files"][label]["sha256"]:
            raise ValueError(f"independent runtime file mismatch for {label}")
    return observed


def _prepare(contract_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _read_json(contract_path)
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError(f"contract schema must be {CONTRACT_SCHEMA}")
    if contract.get("registration_status") != "PRELOCKED_BEFORE_INDEPENDENT_OUTCOMES":
        raise ValueError("independent contract is not prelocked")
    verified_files = {}
    for label, specification in contract["locked_files"].items():
        path = _resolve(contract_path, specification["path"])
        observed = _sha256_file(path)
        if observed != specification["sha256"]:
            raise ValueError(f"locked file hash mismatch for {label}: {observed}")
        verified_files[label] = {"path": str(path), "sha256": observed}
    source_path = _resolve(contract_path, contract["states"]["source_path"])
    control_path = _resolve(contract_path, contract["states"]["control_path"])
    population_path = _resolve(contract_path, contract["population"]["elements_path"])
    selection_path = _resolve(contract_path, contract["selection"]["path"])
    reference_result_path = _resolve(contract_path, contract["reference"]["result_path"])
    reference_root = _resolve(contract_path, contract["reference"]["execution_root"])
    for label, path in (("source", source_path), ("control", control_path)):
        if _sha256_file(path) != contract["states"][f"{label}_sha256"]:
            raise ValueError(f"{label} state hash mismatch")
    if _sha256_file(population_path) != contract["population"]["elements_sha256"]:
        raise ValueError("population hash mismatch")
    if _sha256_file(selection_path) != contract["selection"]["sha256"]:
        raise ValueError("selection hash mismatch")
    if _sha256_file(reference_result_path) != contract["reference"]["result_sha256"]:
        raise ValueError("reference result hash mismatch")
    selection = _read_json(selection_path)
    blocks = [int(value) for value in selection["selected_blocks"]]
    if len(blocks) != int(contract["selection"]["expected_blocks"]) or len(set(blocks)) != len(blocks):
        raise ValueError("selected block count is invalid")
    population = _load_population(population_path)
    if any(block not in population for block in blocks):
        raise ValueError("selection references a missing population block")
    source_rows, control_rows = _load_state(source_path), _load_state(control_path)
    if tuple(row["name"] for row in control_rows) != COMMON_NAMES:
        raise ValueError("control state names differ")
    if tuple(row["name"] for row in source_rows) != (*COMMON_NAMES, SOURCE_NAME):
        raise ValueError("source state names differ")
    for left, right in zip(source_rows[: len(COMMON_NAMES)], control_rows):
        if any(left[field] != right[field] for field in STATE_COLUMNS):
            raise ValueError("source/control common active states are not byte-identical")
    dynamics = contract["dynamics"]
    duration = _positive_float(dynamics["duration_years"], "duration")
    sample = _positive_float(dynamics["sample_years"], "sample cadence")
    checkpoint = _positive_float(dynamics["checkpoint_years"], "checkpoint cadence")
    _integer_ratio(duration, sample, "duration/sample")
    _integer_ratio(duration, checkpoint, "duration/checkpoint")
    solver = contract["solver"]
    if solver.get("method") != "scipy.integrate.DOP853":
        raise ValueError("independent solver must be SciPy DOP853")
    for key in ("rtol", "atol", "max_step_years", "active_audit_segment_years"):
        _positive_float(solver[key], f"solver {key}")
    runtime = _runtime_manifest(contract_path, contract)
    reference_result = _read_json(reference_result_path)
    reference_rows = {}
    reference_verified = {}
    for arm in ("control", "source"):
        reference_rows[arm], reference_verified[arm] = _reference_rows(
            reference_result, reference_root, blocks, arm
        )
    return contract, {
        "contract_sha256": _sha256_file(contract_path),
        "verified_files": verified_files,
        "source_path": source_path,
        "control_path": control_path,
        "source_rows": source_rows,
        "control_rows": control_rows,
        "population_path": population_path,
        "selection": selection,
        "blocks": blocks,
        "reference_result_path": reference_result_path,
        "reference_result": reference_result,
        "reference_rows": reference_rows,
        "reference_verified": reference_verified,
        "runtime": runtime,
    }


def _jobs(
    contract: dict[str, Any], context: dict[str, Any], run_dir: Path
) -> list[dict[str, Any]]:
    q_lower = float(contract["classification"]["q_threshold_AU"]) - float(
        contract["classification"]["q_hysteresis_AU"]
    )
    jobs = []
    for arm, state_path in (("control", context["control_path"]), ("source", context["source_path"])):
        for block in context["blocks"]:
            core = {
                "contract_sha256": context["contract_sha256"],
                "arm": arm,
                "block_index": block,
                "state_csv": str(state_path),
                "state_sha256": contract["states"][f"{arm}_sha256"],
                "population_csv": str(context["population_path"]),
                "population_sha256": contract["population"]["elements_sha256"],
                "duration_years": float(contract["dynamics"]["duration_years"]),
                "sample_years": float(contract["dynamics"]["sample_years"]),
                "checkpoint_years": float(contract["dynamics"]["checkpoint_years"]),
                "q_lower": q_lower,
                "solver": contract["solver"],
            }
            job_sha256 = _canonical_sha256(core)
            jobs.append(
                {
                    **core,
                    "job_sha256": job_sha256,
                    "directory": str(run_dir / arm / f"block_{block:03d}"),
                }
            )
    return jobs


def _initialize_worker() -> None:
    state = np.array([1.0, 0.0], dtype=np.float64)

    def oscillator(_time: float, value: np.ndarray) -> np.ndarray:
        return np.array([value[1], -value[0]])

    solver = DOP853(oscillator, 0.0, state, 0.01, rtol=1e-10, atol=1e-12)
    solver.step()


def _execute(jobs: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    results = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=context, initializer=_initialize_worker
    ) as executor:
        futures = {executor.submit(_run_block, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            result = future.result()
            results.append(result)
            print(
                f"[dop853] {job['arm']} block {job['block_index']:03d} "
                f"complete in {result['elapsed_seconds']:.1f}s",
                flush=True,
            )
    return sorted(results, key=lambda row: (row["arm"], row["block_index"]))


def run_independent_replication(
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
        raise ValueError("worker count is outside the locked range")
    run_root = Path(run_dir).resolve()
    started = time.perf_counter()
    block_records = _execute(_jobs(contract, context, run_root), worker_count)
    independent_rows = {"control": [], "source": []}
    for record in block_records:
        independent_rows[record["arm"]].extend(_read_csv(Path(record["tracer_csv"])))
    expected = 1000 * len(context["blocks"])
    if any(len(independent_rows[arm]) != expected for arm in ("control", "source")):
        raise RuntimeError("independent population output is incomplete")
    comparisons = {
        arm: _compare_rows(context["reference_rows"][arm], independent_rows[arm])
        for arm in ("control", "source")
    }
    statistics_contract = contract["statistics"]
    independent_effect = _effect(
        independent_rows["control"],
        independent_rows["source"],
        context["blocks"],
        statistics_contract["bootstrap_seed"],
        int(statistics_contract["bootstrap_repetitions"]),
        float(statistics_contract["equivalence_margin"]),
    )
    reference_effect = _effect(
        context["reference_rows"]["control"],
        context["reference_rows"]["source"],
        context["blocks"],
        statistics_contract["bootstrap_seed"],
        int(statistics_contract["bootstrap_repetitions"]),
        float(statistics_contract["equivalence_margin"]),
    )
    effect_difference = abs(
        independent_effect["source_minus_control_injection_fraction"]
        - reference_effect["source_minus_control_injection_fraction"]
    )
    active_audits = {
        "control": _active_only_audit(
            context["control_rows"],
            float(contract["dynamics"]["duration_years"]),
            contract["solver"],
        ),
        "source": _active_only_audit(
            context["source_rows"],
            float(contract["dynamics"]["duration_years"]),
            contract["solver"],
        ),
    }
    endpoint_disagreement = {
        arm: _endpoint_disagreement(
            [record for record in block_records if record["arm"] == arm], active_audits[arm]
        )
        for arm in ("control", "source")
    }
    gates = contract["gates"]
    comparison_checks = {}
    for arm in ("control", "source"):
        metrics = comparisons[arm]["metrics"]
        comparison_checks[arm] = {
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
    maximum_roundtrip = max(
        max(record["initial_element_roundtrip"].values()) for record in block_records
    )
    maximum_energy = max(
        [record["maximum_relative_active_energy_drift"] for record in block_records]
        + [audit["maximum_relative_energy_drift"] for audit in active_audits.values()]
    )
    maximum_angular = max(
        [record["maximum_relative_active_angular_momentum_vector_drift"] for record in block_records]
        + [
            audit["maximum_relative_angular_momentum_vector_drift"]
            for audit in active_audits.values()
        ]
    )
    maximum_endpoint_position = max(
        value["maximum_active_endpoint_position_disagreement_AU"]
        for value in endpoint_disagreement.values()
    )
    maximum_endpoint_velocity = max(
        value["maximum_active_endpoint_velocity_disagreement_AU_per_year"]
        for value in endpoint_disagreement.values()
    )
    source_records = [record for record in block_records if record["arm"] == "source"]
    numerical_checks = {
        "complete_finite_outputs": all(record["state_finite"] for record in block_records),
        "checkpoint_replay_exact": all(
            record["checkpoint_replay_binary64_exact"] for record in block_records
        ),
        "initial_element_roundtrip": maximum_roundtrip
        <= float(gates["max_initial_element_roundtrip_error"]),
        "active_energy_drift": maximum_energy <= float(gates["max_relative_active_energy_drift"]),
        "active_angular_momentum_drift": maximum_angular
        <= float(gates["max_relative_active_angular_momentum_vector_drift"]),
        "active_endpoint_position_consistency": maximum_endpoint_position
        <= float(gates["max_active_endpoint_position_disagreement_AU"]),
        "active_endpoint_velocity_consistency": maximum_endpoint_velocity
        <= float(gates["max_active_endpoint_velocity_disagreement_AU_per_year"]),
        "source_bound": all(record["source_bound_at_every_accepted_step"] for record in source_records)
        and bool(active_audits["source"]["source_bound_at_every_accepted_step"]),
        "source_minimum_q": min(
            [record["source_minimum_q_AU"] for record in source_records]
            + [active_audits["source"]["source_minimum_q_AU"]]
        )
        >= float(gates["source_minimum_q_AU"]),
        "source_semimajor_axis_excursion": max(
            [record["source_maximum_fractional_a_excursion"] for record in source_records]
            + [active_audits["source"]["source_maximum_fractional_a_excursion"]]
        )
        <= float(gates["source_maximum_fractional_a_excursion"]),
    }
    cross_software_checks = {
        "control_population_comparison": all(comparison_checks["control"].values()),
        "source_population_comparison": all(comparison_checks["source"].values()),
        "source_control_effect_difference": effect_difference
        <= float(gates["max_source_control_effect_difference"]),
    }
    if not all(numerical_checks.values()):
        verdict = "INVALID"
    elif not all(cross_software_checks.values()):
        verdict = "CONFLICT"
    else:
        verdict = "PASSED"
    result = {
        "schema": RESULT_SCHEMA,
        "verdict": verdict,
        "numerical_status": "VALID" if all(numerical_checks.values()) else "INVALID",
        "science_status": "SCREENING_ONLY" if verdict == "PASSED" else verdict,
        "experiment_id": contract["experiment_id"],
        "contract_path": str(contract_file),
        "contract_sha256": context["contract_sha256"],
        "classification": "INDEPENDENT_SCIPY_DOP853_REPLICATION",
        "selected_blocks": context["blocks"],
        "tracers_per_arm": expected,
        "duration_years": contract["dynamics"]["duration_years"],
        "runtime": context["runtime"],
        "verified_locked_files": context["verified_files"],
        "verified_reference_tracer_files": context["reference_verified"],
        "comparisons": comparisons,
        "comparison_checks": comparison_checks,
        "independent_effect": independent_effect,
        "reference_effect": reference_effect,
        "absolute_source_control_effect_difference": effect_difference,
        "active_only_audits": active_audits,
        "active_endpoint_disagreement": endpoint_disagreement,
        "maximum_initial_element_roundtrip_error": maximum_roundtrip,
        "maximum_relative_active_energy_drift": maximum_energy,
        "maximum_relative_active_angular_momentum_vector_drift": maximum_angular,
        "numerical_checks": numerical_checks,
        "cross_software_checks": cross_software_checks,
        "all_gates_passed": verdict == "PASSED",
        "block_records": [
            {
                key: record[key]
                for key in (
                    "arm",
                    "block_index",
                    "tracers",
                    "tracer_csv",
                    "tracer_csv_sha256",
                    "summary_json",
                    "summary_json_sha256",
                    "checkpoint_replay_binary64_exact",
                    "elapsed_seconds",
                )
            }
            for record in block_records
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "claim_decision": "SCREENING_ONLY" if verdict == "PASSED" else verdict,
        "scientific_scope": contract["scientific_scope"],
        "limitations": contract["limitations"],
        "nonclaim": contract["nonclaim"],
        "next_required_gate": "observed-population and explicit survey-selection model",
    }
    _atomic_json(output, result)
    return result
