"""Real-epoch, outcome-blind, large-population JX source/control screen."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import multiprocessing
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

from .encounter_tail import (
    _active_state,
    _atomic_csv,
    _atomic_json_replace,
    _atomic_simulation,
    _bootstrap_ci,
    _canonical_bytes,
    _cartesian_state_is_finite,
    _simulation_digest,
)
from .ensemble_validation import wasserstein_1d
from .population_scale import (
    STATE_COLUMNS,
    VECTOR_COLUMNS,
    _by_name,
    _canonical_sha256,
    _fraction,
    _load_rows,
    _relative_state,
    _sha256_file,
)


CONTRACT_SCHEMA = "jx-de441-population-contract/v1"
RESULT_SCHEMA = "jx-de441-population-result/v1"
BLOCK_SCHEMA = "jx-de441-population-block/v1"
CHECKPOINT_SCHEMA = "jx-de441-population-checkpoint/v1"
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
COMMON_NAMES = ("Sun", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto")
SOURCE_NAME = "P9_BB21_idx9118"

_POPULATION_CACHE: dict[str, dict[int, list[dict[str, Any]]]] = {}
_STATE_CACHE: dict[str, list[dict[str, str]]] = {}
_FILE_SHA256_CACHE: dict[str, str] = {}


def _resolve(contract_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (contract_path.parent / path).resolve()


def _cached_sha256(path: Path) -> str:
    """Hash immutable locked inputs once per worker process."""
    key = str(path.resolve())
    if key not in _FILE_SHA256_CACHE:
        _FILE_SHA256_CACHE[key] = _sha256_file(path)
    return _FILE_SHA256_CACHE[key]


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
            block = int(raw["block_index"])
            local = int(raw["local_index"])
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
    _POPULATION_CACHE[key] = blocks
    return blocks


def _state_rows(path: Path) -> list[dict[str, str]]:
    key = str(path.resolve())
    if key not in _STATE_CACHE:
        _STATE_CACHE[key] = _load_rows(path)
    return _STATE_CACHE[key]


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


def _write_checkpoint(
    directory: Path,
    simulation: Any,
    tracker: dict[str, Any],
    index: int,
    job_sha256: str,
) -> bool:
    binary_path, state_path = _checkpoint_paths(directory, index)
    before = _simulation_digest(simulation)
    _atomic_simulation(binary_path, simulation)
    import rebound

    replay = rebound.Simulation(str(binary_path))
    after = _simulation_digest(replay)
    if before != after:
        raise RuntimeError("REBOUND checkpoint/replay changed the binary64 state")
    state = {
        "schema": CHECKPOINT_SCHEMA,
        "job_sha256": job_sha256,
        "checkpoint_index": index,
        "time_year": float(simulation.t),
        "simulation_archive_sha256": _sha256_file(binary_path),
        "simulation_state_sha256": before,
        "tracker": tracker,
    }
    _atomic_json_replace(state_path, state)
    return True


def _load_checkpoint(directory: Path, job_sha256: str) -> tuple[Any, dict[str, Any], int] | None:
    import rebound

    for state_path in sorted(directory.glob("checkpoint_*.json"), reverse=True):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            index = int(state["checkpoint_index"])
            binary_path, expected = _checkpoint_paths(directory, index)
            if expected != state_path or state.get("schema") != CHECKPOINT_SCHEMA:
                continue
            if state.get("job_sha256") != job_sha256 or not binary_path.is_file():
                continue
            if _sha256_file(binary_path) != state["simulation_archive_sha256"]:
                continue
            simulation = rebound.Simulation(str(binary_path))
            if _simulation_digest(simulation) != state["simulation_state_sha256"]:
                continue
            if simulation.integrator != "mercurius" or simulation.testparticle_type != 0:
                continue
            return simulation, state["tracker"], index
        except (KeyError, ValueError, OSError, RuntimeError, json.JSONDecodeError):
            continue
    return None


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


def _rows_complete(rows: list[dict[str, Any]], expected: int) -> bool:
    if len(rows) != expected or len({row["logical_id"] for row in rows}) != expected:
        return False
    for row in rows:
        for field in ("a0_AU", "q0_AU", "i0_deg", "minimum_sampled_q_AU"):
            if not math.isfinite(float(row[field])):
                return False
        if int(row["bound_final"]):
            if not math.isfinite(float(row["final_q_AU"])) or not math.isfinite(float(row["final_i_deg"])):
                return False
    return True


def _run_block(job: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / "summary.json"
    tracer_path = directory / "tracers.csv"
    if summary_path.is_file():
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            cached.get("schema") == BLOCK_SCHEMA
            and cached.get("job_sha256") == job["job_sha256"]
            and tracer_path.is_file()
            and _sha256_file(tracer_path) == cached.get("tracer_csv_sha256")
        ):
            cached["resumed_from_complete_block"] = True
            cached["summary_json_sha256"] = _sha256_file(summary_path)
            return cached
        raise RuntimeError(f"locked block conflicts with requested job: {directory}")
    if _cached_sha256(Path(job["state_csv"])) != job["state_sha256"]:
        raise RuntimeError("worker state hash mismatch")
    if _cached_sha256(Path(job["population_csv"])) != job["population_sha256"]:
        raise RuntimeError("worker population hash mismatch")
    states = _state_rows(Path(job["state_csv"]))
    population = _load_population(Path(job["population_csv"]))
    elements = population[job["block_index"]][: job["local_count"]]
    built, names = _build_simulation(states, elements, job["dt_years"])
    if built.N != len(names) + len(elements) or built.N_active != len(names):
        raise RuntimeError("particle-count invariant failed")
    loaded = _load_checkpoint(directory, job["job_sha256"])
    replay_exact = True
    if loaded is None:
        simulation = built
        tracker = _blank_tracker(simulation, names, elements)
        checkpoint_index = 0
        _sample(simulation, elements, tracker, job["q_lower"], True)
        replay_exact = _write_checkpoint(directory, simulation, tracker, 0, job["job_sha256"])
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
                directory, simulation, tracker, checkpoint_index, job["job_sha256"]
            ) and replay_exact
            import rebound

            simulation = rebound.Simulation(str(_checkpoint_paths(directory, checkpoint_index)[0]))
    if not math.isclose(simulation.t, job["duration_years"], rel_tol=0.0, abs_tol=1e-8):
        raise RuntimeError("block did not reach the contracted endpoint")
    final_rows = _final_rows(simulation, elements, tracker, job["q_lower"])
    if not _rows_complete(final_rows, len(elements)):
        raise RuntimeError("block produced incomplete or non-finite tracer rows")
    if not _cartesian_state_is_finite(simulation):
        raise RuntimeError("block produced a non-finite Cartesian state")
    _atomic_csv(tracer_path, final_rows, list(final_rows[0]))
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
        "active_endpoint_state": _active_state(simulation),
        "active_endpoint_state_sha256": _canonical_sha256(_active_state(simulation)),
        "tracer_csv": str(tracer_path),
        "tracer_csv_sha256": _sha256_file(tracer_path),
        "summary_json": str(summary_path),
        "timeseries": tracker["timeseries"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json_replace(summary_path, result)
    result["summary_json_sha256"] = _sha256_file(summary_path)
    return result


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


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
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
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
        verified_files[label] = {"path": str(path), "sha256": observed}
    source_path = _resolve(contract_path, contract["states"]["source_path"])
    control_path = _resolve(contract_path, contract["states"]["control_path"])
    population_path = _resolve(contract_path, contract["population"]["elements_path"])
    for label, path in (("source", source_path), ("control", control_path)):
        if _sha256_file(path) != contract["states"][f"{label}_sha256"]:
            raise ValueError(f"{label} state hash mismatch")
    if _sha256_file(population_path) != contract["population"]["elements_sha256"]:
        raise ValueError("population elements hash mismatch")
    source_rows, control_rows = _state_rows(source_path), _state_rows(control_path)
    matched = _verify_matched_states(source_rows, control_rows)
    population = _load_population(population_path)
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
            digest = hashlib.sha256(_canonical_bytes(core)).hexdigest()
            jobs.append(
                {
                    **core,
                    "job_sha256": digest,
                    "directory": str(run_dir / stage_name / arm / f"block_{block:03d}"),
                }
            )
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
            rows[stage_name][record["arm"]].extend(_load_csv_rows(Path(record["tracer_csv"])))
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
        "contract_path": str(contract_file),
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
                key: record[key]
                for key in (
                    "stage",
                    "arm",
                    "block_index",
                    "tracers",
                    "dt_years",
                    "sample_years",
                    "tracer_csv",
                    "tracer_csv_sha256",
                    "summary_json",
                    "summary_json_sha256",
                    "restart_replay_state_hash_exact",
                    "elapsed_seconds",
                )
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
    _atomic_json_replace(output, result)
    return result
