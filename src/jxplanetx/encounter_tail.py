"""Checkpointed encounter-tail pilot for a controlled synthetic benchmark.

This is deliberately a screening calculation.  It tracks online sampled
perihelia and sampled Hill-sphere entries for a physically eligible synthetic
population, but it does not turn the archived approximate-element benchmark
into a real-epoch Solar-System state or an ephemeris validation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import multiprocessing
import os
import statistics
import struct
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

from .ensemble_validation import wasserstein_1d
from .population_scale import (
    CORE_NAMES,
    G_AU3_MSUN_YR2,
    _binary64_massive_state,
    _by_name,
    _canonical_sha256,
    _fraction,
    _load_rows,
    _phase_angles,
    _relative_state,
    _sha256_file,
    _verify_matched_states,
)
from .provenance import runtime_source_manifest


CONTRACT_SCHEMA = "jx-encounter-tail-contract/v1"
RESULT_SCHEMA = "jx-encounter-tail-result/v1"
BLOCK_SCHEMA = "jx-encounter-tail-block/v1"
CHECKPOINT_SCHEMA = "jx-encounter-tail-checkpoint/v1"
FRAME_LABEL = "CONTROLLED_SYNTHETIC_BENCHMARK"


@dataclass(frozen=True)
class GridCell:
    index: int
    a_AU: float
    q0_AU: float
    i_deg: float

    @property
    def e(self) -> float:
        return 1.0 - self.q0_AU / self.a_AU


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _atomic_json_replace(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
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


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
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


def _atomic_simulation(path: Path, simulation: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    os.unlink(temporary)
    try:
        simulation.save_to_file(str(temporary), delete_file=True)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _resolve(contract_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (contract_path.parent / path).resolve()


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


def _grid(design: Mapping[str, Any]) -> list[GridCell]:
    axes = design["grid"]
    semimajor = [float(value) for value in axes["a_AU"]]
    perihelia = [float(value) for value in axes["q0_AU"]]
    inclinations = [float(value) for value in axes["i_deg"]]
    for label, values in (("a_AU", semimajor), ("q0_AU", perihelia), ("i_deg", inclinations)):
        if not values or len(values) != len(set(values)) or any(not math.isfinite(value) for value in values):
            raise ValueError(f"grid {label} must contain unique finite values")
    cells: list[GridCell] = []
    # Contracted lexicographic order: a, then q0, then inclination.
    for a in semimajor:
        for q0 in perihelia:
            for inclination in inclinations:
                if not (a > q0 > 0.0 and 0.0 <= inclination < 90.0):
                    raise ValueError("every grid cell must be bound with a>q0>0 and 0<=i<90")
                cells.append(GridCell(len(cells), a, q0, inclination))
    if len(cells) != int(design["required_grid_cells"]):
        raise ValueError("grid cardinality does not equal required_grid_cells")
    return cells


def _block_seed(base_seed: str, block_index: int) -> str:
    return hashlib.sha256(f"jx-tail-block/v1\x1f{base_seed}\x1f{block_index}".encode()).hexdigest()


def _cell_for(cells: list[GridCell], block_index: int, local_index: int, stride: int) -> GridCell:
    return cells[(local_index + stride * block_index) % len(cells)]


def _massive_names(rows: Iterable[Mapping[str, str]]) -> list[str]:
    return [row["name"] for row in rows if Decimal(row["mass"]) > 0]


def _safe_name(name: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in name).strip("_")


def _simulation_digest(simulation: Any) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("!dii", float(simulation.t), int(simulation.N), int(simulation.N_active)))
    digest.update(str(simulation.integrator).encode("ascii"))
    digest.update(struct.pack("!d", float(simulation.dt)))
    for particle in simulation.particles:
        digest.update(struct.pack("!7d", particle.m, particle.x, particle.y, particle.z, particle.vx, particle.vy, particle.vz))
    return digest.hexdigest()


def _active_state(simulation: Any) -> list[dict[str, str]]:
    result = []
    for index in range(simulation.N_active):
        particle = simulation.particles[index]
        result.append(
            {
                "index": str(index),
                **{field: float(getattr(particle, field)).hex() for field in ("m", "x", "y", "z", "vx", "vy", "vz")},
            }
        )
    return result


def _cartesian_state_is_finite(simulation: Any) -> bool:
    return all(
        math.isfinite(float(getattr(particle, field)))
        for particle in simulation.particles
        for field in ("m", "x", "y", "z", "vx", "vy", "vz")
    )


def _build_simulation(
    rows: list[dict[str, str]],
    cells: list[GridCell],
    block_index: int,
    local_indices: list[int],
    base_seed: str,
    stratum_stride: int,
    dt_years: float,
) -> tuple[Any, list[dict[str, Any]], list[str]]:
    import rebound

    simulation = rebound.Simulation()
    simulation.G = G_AU3_MSUN_YR2
    sun_row = _by_name(rows)["Sun"]
    names = _massive_names(rows)
    for row in rows:
        if Decimal(row["mass"]) <= 0:
            continue
        relative = _relative_state(row, sun_row)
        simulation.add(
            m=float(row["mass"]),
            x=float(relative[0]), y=float(relative[1]), z=float(relative[2]),
            vx=float(relative[3]), vy=float(relative[4]), vz=float(relative[5]),
            hash=row["name"],
        )
    simulation.N_active = len(names)
    simulation.testparticle_type = 0
    simulation.integrator = "mercurius"
    simulation.dt = dt_years
    simulation.ri_mercurius.r_crit_hill = 3.0
    simulation.collision = "none"
    metadata: list[dict[str, Any]] = []
    seed = _block_seed(base_seed, block_index)
    for local_index in local_indices:
        cell = _cell_for(cells, block_index, local_index, stratum_stride)
        node, periapse, mean_anomaly = _phase_angles(seed, local_index)
        simulation.add(
            # Fetch particle 0 after every possible array reallocation.  A
            # cached Particle pointer can become stale as REBOUND grows the
            # particle array, intermittently appearing to have zero mass.
            primary=simulation.particles[0],
            m=0.0,
            a=cell.a_AU,
            e=cell.e,
            inc=math.radians(cell.i_deg),
            Omega=node,
            omega=periapse,
            M=mean_anomaly,
        )
        metadata.append(
            {
                "block_index": block_index,
                "local_index": local_index,
                "logical_id": f"b{block_index:02d}-j{local_index:04d}",
                "diagnostic_replicate": local_index // 100,
                "cell_index": cell.index,
                "a0_AU": cell.a_AU,
                "q0_AU": cell.q0_AU,
                "i0_deg": cell.i_deg,
                "Omega0_rad": node,
                "omega0_rad": periapse,
                "M0_rad": mean_anomaly,
            }
        )
    return simulation, metadata, names


def _initial_hill_radii(simulation: Any, massive_names: list[str]) -> dict[str, float]:
    sun = simulation.particles[0]
    result: dict[str, float] = {}
    for index, name in enumerate(massive_names):
        if name == "Sun":
            continue
        body = simulation.particles[index]
        orbit = body.orbit(primary=sun)
        result[name] = orbit.a * (body.m / (3.0 * sun.m)) ** (1.0 / 3.0)
    return result


def _blank_tracker(metadata: list[dict[str, Any]], perturber_names: list[str], hill_radii: dict[str, float]) -> dict[str, Any]:
    count = len(metadata)
    return {
        "minimum_q": [None] * count,
        "ever_unbound": [False] * count,
        "first_sampled_low_q_year": [None] * count,
        "first_unbound_year": [None] * count,
        "fixed_initial_hill_radius_AU": hill_radii,
        "minimum_distance": {name: [None] * count for name in perturber_names},
        "minimum_hill_ratio": {name: [None] * count for name in perturber_names},
        "previous_lt1": {name: [False] * count for name in perturber_names},
        "previous_lt3": {name: [False] * count for name in perturber_names},
        "entries_lt1": {name: [0] * count for name in perturber_names},
        "entries_lt3": {name: [0] * count for name in perturber_names},
        "first_entry_lt1_year": {name: [None] * count for name in perturber_names},
        "first_entry_lt3_year": {name: [None] * count for name in perturber_names},
        "timeseries": [],
        "sample_count": 0,
    }


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _sample(
    simulation: Any,
    metadata: list[dict[str, Any]],
    massive_names: list[str],
    tracker: dict[str, Any],
    record_timeseries: bool,
    q_lower: float,
    q_upper: float,
) -> None:
    particles = simulation.particles
    sun = particles[0]
    perturber_indices = [(index, name) for index, name in enumerate(massive_names) if name != "Sun"]
    current_q: list[float | None] = []
    current_i: list[float | None] = []
    for tracer_index, item in enumerate(metadata):
        particle = particles[simulation.N_active + tracer_index]
        try:
            orbit = particle.orbit(primary=sun)
            q = orbit.a * (1.0 - orbit.e)
            inclination = math.degrees(orbit.inc)
            bound = orbit.a > 0.0 and orbit.e < 1.0 and math.isfinite(q) and math.isfinite(inclination)
        except (ValueError, ZeroDivisionError, OverflowError):
            q, inclination, bound = math.nan, math.nan, False
        if bound:
            old_q = tracker["minimum_q"][tracer_index]
            tracker["minimum_q"][tracer_index] = q if old_q is None else min(old_q, q)
            if q < q_lower and tracker["first_sampled_low_q_year"][tracer_index] is None:
                tracker["first_sampled_low_q_year"][tracer_index] = float(simulation.t)
            current_q.append(q)
            current_i.append(inclination)
        else:
            tracker["ever_unbound"][tracer_index] = True
            if tracker["first_unbound_year"][tracer_index] is None:
                tracker["first_unbound_year"][tracer_index] = float(simulation.t)
            current_q.append(None)
            current_i.append(None)
        for active_index, name in perturber_indices:
            body = particles[active_index]
            dx, dy, dz = particle.x - body.x, particle.y - body.y, particle.z - body.z
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            hill_radius = tracker["fixed_initial_hill_radius_AU"][name]
            ratio = distance / hill_radius if hill_radius > 0.0 else math.inf
            old_distance = tracker["minimum_distance"][name][tracer_index]
            old_ratio = tracker["minimum_hill_ratio"][name][tracer_index]
            if math.isfinite(distance) and (old_distance is None or distance < old_distance):
                tracker["minimum_distance"][name][tracer_index] = distance
            if math.isfinite(ratio) and (old_ratio is None or ratio < old_ratio):
                tracker["minimum_hill_ratio"][name][tracer_index] = ratio
            inside1, inside3 = ratio < 1.0, ratio < 3.0
            initial_observation = tracker["sample_count"] == 0
            if inside1 and not initial_observation and not tracker["previous_lt1"][name][tracer_index]:
                tracker["entries_lt1"][name][tracer_index] += 1
                if tracker["first_entry_lt1_year"][name][tracer_index] is None:
                    tracker["first_entry_lt1_year"][name][tracer_index] = float(simulation.t)
            if inside3 and not initial_observation and not tracker["previous_lt3"][name][tracer_index]:
                tracker["entries_lt3"][name][tracer_index] += 1
                if tracker["first_entry_lt3_year"][name][tracer_index] is None:
                    tracker["first_entry_lt3_year"][name][tracer_index] = float(simulation.t)
            tracker["previous_lt1"][name][tracer_index] = inside1
            tracker["previous_lt3"][name][tracer_index] = inside3
    tracker["sample_count"] += 1
    if record_timeseries:
        bound_q = [value for value in current_q if value is not None]
        bound_i = [value for value in current_i if value is not None]
        injections = sum(
            item["q0_AU"] > q_upper
            and tracker["minimum_q"][index] is not None
            and tracker["minimum_q"][index] < q_lower
            for index, item in enumerate(metadata)
        )
        tracker["timeseries"].append(
            {
                "time_year": float(simulation.t),
                "bound": len(bound_q),
                "survival_fraction": len(bound_q) / len(metadata),
                "cumulative_sampled_injections": injections,
                "cumulative_sampled_injection_fraction": injections / len(metadata),
                "mean_current_bound_q_AU": statistics.fmean(bound_q) if bound_q else None,
                "current_bound_inclination_width_deg": statistics.pstdev(bound_i) if bound_i else None,
            }
        )


def _checkpoint_paths(directory: Path, checkpoint_index: int) -> tuple[Path, Path]:
    stem = f"checkpoint_{checkpoint_index:04d}"
    return directory / f"{stem}.bin", directory / f"{stem}.json"


def _load_latest_checkpoint(directory: Path, job_sha256: str) -> tuple[Any, dict[str, Any], int] | None:
    import rebound

    for state_path in sorted(directory.glob("checkpoint_*.json"), reverse=True):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            index = int(state["checkpoint_index"])
            binary_path, expected_state_path = _checkpoint_paths(directory, index)
            if expected_state_path != state_path or state.get("schema") != CHECKPOINT_SCHEMA:
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
            if not math.isclose(simulation.ri_mercurius.r_crit_hill, 3.0, rel_tol=0.0, abs_tol=0.0):
                continue
            return simulation, state["tracker"], index
        except (KeyError, ValueError, OSError, json.JSONDecodeError, RuntimeError):
            continue
    return None


def _write_checkpoint(
    directory: Path,
    simulation: Any,
    tracker: dict[str, Any],
    checkpoint_index: int,
    job_sha256: str,
) -> bool:
    binary_path, state_path = _checkpoint_paths(directory, checkpoint_index)
    before = _simulation_digest(simulation)
    _atomic_simulation(binary_path, simulation)
    import rebound

    replay = rebound.Simulation(str(binary_path))
    after = _simulation_digest(replay)
    exact = before == after
    if not exact:
        raise RuntimeError("REBOUND checkpoint/replay changed the binary64 state")
    state = {
        "schema": CHECKPOINT_SCHEMA,
        "job_sha256": job_sha256,
        "checkpoint_index": checkpoint_index,
        "time_year": float(simulation.t),
        "simulation_archive_sha256": _sha256_file(binary_path),
        "simulation_state_sha256": before,
        "exact_replay": exact,
        "tracker": tracker,
    }
    _atomic_json_replace(state_path, state)
    return exact


def _final_rows(
    simulation: Any,
    metadata: list[dict[str, Any]],
    massive_names: list[str],
    tracker: dict[str, Any],
    q_lower: float,
    q_upper: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sun = simulation.particles[0]
    for index, item in enumerate(metadata):
        particle = simulation.particles[simulation.N_active + index]
        try:
            orbit = particle.orbit(primary=sun)
            q = orbit.a * (1.0 - orbit.e)
            bound = orbit.a > 0.0 and orbit.e < 1.0 and math.isfinite(q) and math.isfinite(orbit.inc)
            final_q = q if bound else None
            final_i = math.degrees(orbit.inc) if bound else None
        except (ValueError, ZeroDivisionError, OverflowError):
            bound, final_q, final_i = False, None, None
        row: dict[str, Any] = {
            **item,
            "minimum_sampled_q_AU": tracker["minimum_q"][index],
            "sampled_injection": int(
                item["q0_AU"] > q_upper
                and tracker["minimum_q"][index] is not None
                and tracker["minimum_q"][index] < q_lower
            ),
            "first_sampled_low_q_year": tracker["first_sampled_low_q_year"][index],
            "ever_unbound_at_sample": int(tracker["ever_unbound"][index]),
            "first_unbound_year": tracker["first_unbound_year"][index],
            "bound_final": int(bound),
            "final_q_AU": final_q,
            "final_i_deg": final_i,
        }
        for name in massive_names:
            if name == "Sun":
                continue
            prefix = _safe_name(name)
            row[f"minimum_sampled_{prefix}_distance_AU"] = tracker["minimum_distance"][name][index]
            row[f"minimum_sampled_{prefix}_hill_ratio"] = tracker["minimum_hill_ratio"][name][index]
            row[f"fixed_initial_{prefix}_hill_radius_AU"] = tracker["fixed_initial_hill_radius_AU"][name]
            row[f"sampled_{prefix}_entries_lt1RH"] = tracker["entries_lt1"][name][index]
            row[f"sampled_{prefix}_entries_lt3RH"] = tracker["entries_lt3"][name][index]
            row[f"first_sampled_{prefix}_entry_lt1RH_year"] = tracker["first_entry_lt1_year"][name][index]
            row[f"first_sampled_{prefix}_entry_lt3RH_year"] = tracker["first_entry_lt3_year"][name][index]
        rows.append(row)
    return rows


def _rows_finite_complete(rows: list[dict[str, Any]], required_count: int) -> bool:
    if len(rows) != required_count:
        return False
    for row in rows:
        for required in ("a0_AU", "q0_AU", "i0_deg", "minimum_sampled_q_AU"):
            try:
                if not math.isfinite(float(row[required])):
                    return False
            except (KeyError, TypeError, ValueError):
                return False
        if int(row["bound_final"]):
            for required in ("final_q_AU", "final_i_deg"):
                try:
                    if not math.isfinite(float(row[required])):
                        return False
                except (KeyError, TypeError, ValueError):
                    return False
    return True


def _run_block(job: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / "summary.json"
    csv_path = directory / "tracers.csv"
    if summary_path.is_file():
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            cached.get("schema") == BLOCK_SCHEMA
            and cached.get("job_sha256") == job["job_sha256"]
            and csv_path.is_file()
            and _sha256_file(csv_path) == cached.get("tracer_csv_sha256")
        ):
            cached["resumed_from_complete_block"] = True
            cached["summary_json_sha256"] = _sha256_file(summary_path)
            return cached
        raise RuntimeError(f"locked block output conflicts with requested job: {directory}")
    rows = _load_rows(Path(job["state_csv"]))
    cells = [GridCell(**cell) for cell in job["cells"]]
    local_indices = list(range(job["local_start"], job["local_stop"]))
    built, metadata, names = _build_simulation(
        rows, cells, job["block_index"], local_indices, job["base_seed"],
        job["stratum_stride"], job["dt_years"],
    )
    expected_particles = len(names) + len(metadata)
    if built.N != expected_particles or built.N_active != len(names):
        raise RuntimeError("particle-count invariant failed")
    loaded = _load_latest_checkpoint(directory, job["job_sha256"])
    replay_exact = True
    if loaded is None:
        simulation = built
        tracker = _blank_tracker(metadata, names[1:], _initial_hill_radii(simulation, names))
        checkpoint_index = 0
        _sample(simulation, metadata, names, tracker, True, job["q_lower"], job["q_upper"])
        replay_exact = _write_checkpoint(
            directory, simulation, tracker, checkpoint_index, job["job_sha256"]
        )
        import rebound

        simulation = rebound.Simulation(str(_checkpoint_paths(directory, checkpoint_index)[0]))
    else:
        simulation, tracker, checkpoint_index = loaded
        if simulation.N != expected_particles or simulation.N_active != len(names):
            raise RuntimeError("checkpoint particle-count invariant failed")
    del built
    sample_steps = job["sample_steps"]
    aggregate_stride = job["aggregate_stride_samples"]
    checkpoint_stride = job["checkpoint_stride_samples"]
    total_samples = job["total_samples"]
    for sample_index in range(checkpoint_index * checkpoint_stride + 1, total_samples + 1):
        target_time = sample_index * job["sample_years"]
        simulation.integrate(target_time, exact_finish_time=1)
        _sample(
            simulation, metadata, names, tracker,
            sample_index % aggregate_stride == 0,
            job["q_lower"], job["q_upper"],
        )
        if sample_index % checkpoint_stride == 0:
            checkpoint_index = sample_index // checkpoint_stride
            replay_exact = _write_checkpoint(
                directory, simulation, tracker, checkpoint_index, job["job_sha256"]
            ) and replay_exact
            # Continue from the verified replay, exercising real restart semantics.
            import rebound

            simulation = rebound.Simulation(str(_checkpoint_paths(directory, checkpoint_index)[0]))
    if not math.isclose(simulation.t, job["duration_years"], rel_tol=0.0, abs_tol=1e-8):
        raise RuntimeError("block did not reach the contracted endpoint")
    final_rows = _final_rows(
        simulation, metadata, names, tracker, job["q_lower"], job["q_upper"]
    )
    cartesian_finite = _cartesian_state_is_finite(simulation)
    if not _rows_finite_complete(final_rows, len(metadata)):
        raise RuntimeError("block produced incomplete or nonfinite required tracer rows")
    if not cartesian_finite:
        raise RuntimeError("block produced a nonfinite Cartesian particle state")
    fieldnames = list(final_rows[0])
    _atomic_csv(csv_path, final_rows, fieldnames)
    result = {
        "schema": BLOCK_SCHEMA,
        "job_sha256": job["job_sha256"],
        "arm": job["arm"],
        "stage": job["stage"],
        "block_index": job["block_index"],
        "local_start": job["local_start"],
        "local_stop": job["local_stop"],
        "tracers": len(metadata),
        "particles": simulation.N,
        "N_active": simulation.N_active,
        "dt_years": job["dt_years"],
        "duration_years": job["duration_years"],
        "online_sample_cadence_years": job["sample_years"],
        "online_samples_including_t0": tracker["sample_count"],
        "aggregate_timeseries_cadence_years": job["aggregate_years"],
        "checkpoint_cadence_years": job["checkpoint_years"],
        "checkpoint_epochs_written_or_verified_including_t0": checkpoint_index + 1,
        "restart_replay_state_hash_exact": replay_exact,
        "integrator": simulation.integrator,
        "testparticle_type": simulation.testparticle_type,
        "mercurius_r_crit_hill": simulation.ri_mercurius.r_crit_hill,
        "collision_mode": simulation.collision,
        "active_endpoint_state": _active_state(simulation),
        "active_endpoint_state_sha256": _canonical_sha256(_active_state(simulation)),
        "tracer_csv": str(csv_path),
        "tracer_csv_sha256": _sha256_file(csv_path),
        "summary_json": str(summary_path),
        "timeseries": tracker["timeseries"],
        "elapsed_seconds": time.perf_counter() - started,
        "rows_finite_complete": True,
        "cartesian_state_finite": cartesian_finite,
        "sample_steps": sample_steps,
    }
    _atomic_json_replace(summary_path, result)
    result["summary_json_sha256"] = _sha256_file(summary_path)
    return result


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _float_values(rows: list[dict[str, str]], field: str, condition: str | None = None) -> list[float]:
    result = []
    for row in rows:
        if condition is not None and int(row[condition]) == 0:
            continue
        raw = row.get(field, "")
        if raw == "":
            continue
        value = float(raw)
        if math.isfinite(value):
            result.append(value)
    return result


def _population_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    count = len(rows)
    injections = sum(int(row["sampled_injection"]) for row in rows)
    bound = sum(int(row["bound_final"]) for row in rows)
    ever_unbound = sum(int(row["ever_unbound_at_sample"]) for row in rows)
    min_q = _float_values(rows, "minimum_sampled_q_AU")
    final_q = _float_values(rows, "final_q_AU", "bound_final")
    final_i = _float_values(rows, "final_i_deg", "bound_final")
    by_q0: dict[str, Any] = {}
    for q0 in sorted({float(row["q0_AU"]) for row in rows}):
        subset = [row for row in rows if float(row["q0_AU"]) == q0]
        subtotal = sum(int(row["sampled_injection"]) for row in subset)
        by_q0[format(q0, "g")] = {
            "tracers": len(subset),
            "sampled_injections": subtotal,
            "sampled_injection_fraction": subtotal / len(subset),
        }
    by_cell: dict[str, Any] = {}
    for cell in sorted({int(row["cell_index"]) for row in rows}):
        subset = [row for row in rows if int(row["cell_index"]) == cell]
        cell_injections = sum(int(row["sampled_injection"]) for row in subset)
        cell_bound = sum(int(row["bound_final"]) for row in subset)
        by_cell[str(cell)] = {
            "tracers": len(subset),
            "sampled_injections": cell_injections,
            "sampled_injection_fraction": cell_injections / len(subset),
            "survival_fraction": cell_bound / len(subset),
        }
    entry_totals: dict[str, dict[str, int]] = {}
    if rows:
        for field in rows[0]:
            if field.startswith("sampled_") and field.endswith("_entries_lt1RH"):
                body = field[len("sampled_") : -len("_entries_lt1RH")]
                field3 = f"sampled_{body}_entries_lt3RH"
                entry_totals[body] = {
                    "sampled_entries_lt1RH": sum(int(row[field]) for row in rows),
                    "sampled_entries_lt3RH": sum(int(row[field3]) for row in rows),
                }
    return {
        "tracers": count,
        "sampled_injections": injections,
        "sampled_injection_fraction": injections / count,
        "bound_final": bound,
        "survival_fraction": bound / count,
        "ever_unbound_at_sample": ever_unbound,
        "minimum_sampled_q_mean_AU": statistics.fmean(min_q),
        "final_bound_q_mean_AU": statistics.fmean(final_q) if final_q else None,
        "final_bound_inclination_width_deg": statistics.pstdev(final_i) if final_i else None,
        "by_initial_q_AU": by_q0,
        "by_grid_cell": by_cell,
        "equal_grid_cell_weighted_sampled_injection_fraction": statistics.fmean(
            item["sampled_injection_fraction"] for item in by_cell.values()
        ),
        "equal_grid_cell_weighted_survival_fraction": statistics.fmean(
            item["survival_fraction"] for item in by_cell.values()
        ),
        "sampled_hill_sphere_entries": entry_totals,
    }


def _bootstrap_ci(values: list[float], seed: str, repetitions: int) -> list[float]:
    estimates: list[float] = []
    for repetition in range(repetitions):
        draw = []
        for index in range(len(values)):
            message = f"jx-paired-block-bootstrap/v1\x1f{seed}\x1f{repetition}\x1f{index}".encode()
            selected = int.from_bytes(hashlib.sha256(message).digest()[:8], "big") % len(values)
            draw.append(values[selected])
        estimates.append(statistics.fmean(draw))
    estimates.sort()

    def quantile(probability: float) -> float:
        position = probability * (len(estimates) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return estimates[lower]
        fraction = position - lower
        return estimates[lower] * (1.0 - fraction) + estimates[upper] * fraction

    return [quantile(0.025), quantile(0.975)]


def _effect(
    control: list[dict[str, str]],
    source: list[dict[str, str]],
    bootstrap_seed: str,
    bootstrap_repetitions: int,
) -> dict[str, Any]:
    control_by_id = {row["logical_id"]: row for row in control}
    source_by_id = {row["logical_id"]: row for row in source}
    if set(control_by_id) != set(source_by_id):
        raise ValueError("source and control logical tracer IDs do not match")
    ids = sorted(control_by_id)
    for identity in ids:
        for field in ("cell_index", "a0_AU", "q0_AU", "i0_deg", "Omega0_rad", "omega0_rad", "M0_rad"):
            if control_by_id[identity][field] != source_by_id[identity][field]:
                raise ValueError(f"paired initial tracer metadata mismatch for {identity}")
    control_summary, source_summary = _population_summary(control), _population_summary(source)
    block_effects = []
    for block in sorted({int(row["block_index"]) for row in control}):
        c = [row for row in control if int(row["block_index"]) == block]
        s = [row for row in source if int(row["block_index"]) == block]
        cs, ss = _population_summary(c), _population_summary(s)
        block_effects.append(
            {
                "block_index": block,
                "sampled_injection_fraction_difference": ss["sampled_injection_fraction"] - cs["sampled_injection_fraction"],
                "equal_grid_cell_weighted_sampled_injection_fraction_difference": (
                    ss["equal_grid_cell_weighted_sampled_injection_fraction"]
                    - cs["equal_grid_cell_weighted_sampled_injection_fraction"]
                ),
                "survival_fraction_difference": ss["survival_fraction"] - cs["survival_fraction"],
            }
        )
    injection_effects = [row["equal_grid_cell_weighted_sampled_injection_fraction_difference"] for row in block_effects]
    return {
        "control": control_summary,
        "source": source_summary,
        "source_minus_control": {
            "sampled_injections": source_summary["sampled_injections"] - control_summary["sampled_injections"],
            "sampled_injection_fraction": source_summary["sampled_injection_fraction"] - control_summary["sampled_injection_fraction"],
            "primary_equal_grid_cell_weighted_sampled_injection_fraction": (
                source_summary["equal_grid_cell_weighted_sampled_injection_fraction"]
                - control_summary["equal_grid_cell_weighted_sampled_injection_fraction"]
            ),
            "survival_fraction": source_summary["survival_fraction"] - control_summary["survival_fraction"],
            "wasserstein_minimum_sampled_q_AU": wasserstein_1d(
                _float_values(control, "minimum_sampled_q_AU"),
                _float_values(source, "minimum_sampled_q_AU"),
            ),
            "wasserstein_final_bound_q_AU": wasserstein_1d(
                _float_values(control, "final_q_AU", "bound_final"),
                _float_values(source, "final_q_AU", "bound_final"),
            ),
            "block_mean_sampled_injection_effect": statistics.fmean(injection_effects),
            "block_sample_standard_deviation": statistics.stdev(injection_effects) if len(injection_effects) > 1 else None,
            "normal_95_percent_CI": (
                [
                    statistics.fmean(injection_effects) - 1.96 * statistics.stdev(injection_effects) / math.sqrt(len(injection_effects)),
                    statistics.fmean(injection_effects) + 1.96 * statistics.stdev(injection_effects) / math.sqrt(len(injection_effects)),
                ]
                if len(injection_effects) > 1 else None
            ),
            "paired_seed_block_bootstrap_repetitions": bootstrap_repetitions,
            "paired_seed_block_bootstrap_95_percent_CI": _bootstrap_ci(
                injection_effects, bootstrap_seed, bootstrap_repetitions
            ),
        },
        "block_effects": block_effects,
        "paired_initial_metadata_sha256": _canonical_sha256(
            [{field: control_by_id[identity][field] for field in ("logical_id", "cell_index", "a0_AU", "q0_AU", "i0_deg", "Omega0_rad", "omega0_rad", "M0_rad")} for identity in ids]
        ),
    }


def _convergence_comparison(
    primary: dict[str, list[dict[str, str]]],
    half: dict[str, list[dict[str, str]]],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    selected_primary: dict[str, list[dict[str, str]]] = {}
    all_checks: list[bool] = []
    for arm in ("control", "source"):
        first_by_id = {row["logical_id"]: row for row in primary[arm] if int(row["local_index"]) < 100}
        second_by_id = {row["logical_id"]: row for row in half[arm]}
        if set(first_by_id) != set(second_by_id):
            raise ValueError(f"{arm} dt/2 audit identity mismatch")
        first, second = list(first_by_id.values()), [second_by_id[key] for key in first_by_id]
        selected_primary[arm] = first
        first_summary, second_summary = _population_summary(first), _population_summary(second)
        q0_disagreement = max(
            abs(
                second_summary["by_initial_q_AU"][q0]["sampled_injection_fraction"]
                - first_summary["by_initial_q_AU"][q0]["sampled_injection_fraction"]
            )
            for q0 in first_summary["by_initial_q_AU"]
        )
        metrics = {
            "tracers": len(first),
            "absolute_sampled_injection_fraction_difference": abs(second_summary["sampled_injection_fraction"] - first_summary["sampled_injection_fraction"]),
            "absolute_survival_fraction_difference": abs(second_summary["survival_fraction"] - first_summary["survival_fraction"]),
            "wasserstein_minimum_sampled_q_AU": wasserstein_1d(_float_values(first, "minimum_sampled_q_AU"), _float_values(second, "minimum_sampled_q_AU")),
            "wasserstein_final_bound_q_AU": wasserstein_1d(_float_values(first, "final_q_AU", "bound_final"), _float_values(second, "final_q_AU", "bound_final")),
            "wasserstein_final_bound_i_deg": wasserstein_1d(_float_values(first, "final_i_deg", "bound_final"), _float_values(second, "final_i_deg", "bound_final")),
            "maximum_q0_group_injection_fraction_difference": q0_disagreement,
        }
        checks = {
            "sampled_injection_fraction": metrics["absolute_sampled_injection_fraction_difference"] <= float(gates["max_injection_fraction_difference"]),
            "survival_fraction": metrics["absolute_survival_fraction_difference"] <= float(gates["max_survival_fraction_difference"]),
            "minimum_sampled_q_wasserstein": metrics["wasserstein_minimum_sampled_q_AU"] <= float(gates["max_wasserstein_minimum_q_AU"]),
            "final_q_wasserstein": metrics["wasserstein_final_bound_q_AU"] <= float(gates["max_wasserstein_final_q_AU"]),
            "final_i_wasserstein": metrics["wasserstein_final_bound_i_deg"] <= float(gates["max_wasserstein_final_i_deg"]),
            "q0_group_injection_fraction": metrics["maximum_q0_group_injection_fraction_difference"] <= float(gates["max_q0_group_injection_fraction_difference"]),
        }
        all_checks.extend(checks.values())
        arms[arm] = {"metrics": metrics, "checks": checks, "passed": all(checks.values())}
    primary_effect = _population_summary(selected_primary["source"])["sampled_injection_fraction"] - _population_summary(selected_primary["control"])["sampled_injection_fraction"]
    half_effect = _population_summary(half["source"])["sampled_injection_fraction"] - _population_summary(half["control"])["sampled_injection_fraction"]
    effect_difference = abs(half_effect - primary_effect)
    effect_passed = effect_difference <= float(gates["max_source_control_injection_effect_difference"])
    all_checks.append(effect_passed)
    q0_effect_differences: dict[str, float] = {}
    for q0 in _population_summary(half["source"])["by_initial_q_AU"]:
        primary_source = _population_summary(selected_primary["source"])["by_initial_q_AU"][q0]["sampled_injection_fraction"]
        primary_control = _population_summary(selected_primary["control"])["by_initial_q_AU"][q0]["sampled_injection_fraction"]
        half_source = _population_summary(half["source"])["by_initial_q_AU"][q0]["sampled_injection_fraction"]
        half_control = _population_summary(half["control"])["by_initial_q_AU"][q0]["sampled_injection_fraction"]
        q0_effect_differences[q0] = abs((half_source - half_control) - (primary_source - primary_control))
    q0_effect_maximum = max(q0_effect_differences.values())
    q0_effect_passed = q0_effect_maximum <= float(gates["max_q0_group_injection_fraction_difference"])
    all_checks.append(q0_effect_passed)
    return {
        "arms": arms,
        "source_control_sampled_injection_effect": {
            "dt_primary": primary_effect,
            "dt_half": half_effect,
            "absolute_difference": effect_difference,
            "threshold": float(gates["max_source_control_injection_effect_difference"]),
            "passed": effect_passed,
        },
        "source_control_q0_group_injection_effect": {
            "absolute_differences": q0_effect_differences,
            "maximum_absolute_difference": q0_effect_maximum,
            "threshold": float(gates["max_q0_group_injection_fraction_difference"]),
            "passed": q0_effect_passed,
        },
        "passed": all(all_checks),
    }


def _active_audit(rows: list[dict[str, str]], dt: float, duration: float) -> dict[str, Any]:
    cells = [GridCell(0, 100.0, 35.0, 5.0)]
    simulation, _, names = _build_simulation(rows, cells, 0, [], "audit", 8, dt)
    initial_energy = simulation.energy()
    initial_l = simulation.angular_momentum()
    initial_l_values = (float(initial_l.x), float(initial_l.y), float(initial_l.z))
    initial_l_norm = math.sqrt(sum(value * value for value in initial_l_values))
    steps = _integer_ratio(duration, dt, "active audit duration/dt")
    max_energy = 0.0
    max_l = 0.0
    energy_endpoint = 0.0
    l_endpoint = 0.0
    started = time.perf_counter()
    source_index = names.index(next((name for name in names if name not in CORE_NAMES), "")) if len(names) > len(CORE_NAMES) else None
    if source_index is not None:
        source_initial = simulation.particles[source_index].orbit(primary=simulation.particles[0])
        source_initial_a = source_initial.a
        source_min_q = source_initial.a * (1.0 - source_initial.e)
        source_min_a = source_initial.a
        source_max_a = source_initial.a
        source_bound_all_steps = source_initial.a > 0.0 and source_initial.e < 1.0
    else:
        source_initial_a = source_min_q = source_min_a = source_max_a = None
        source_bound_all_steps = None
    for _ in range(steps):
        simulation.step()
        energy_endpoint = abs((simulation.energy() - initial_energy) / initial_energy)
        angular = simulation.angular_momentum()
        differences = (
            float(angular.x) - initial_l_values[0],
            float(angular.y) - initial_l_values[1],
            float(angular.z) - initial_l_values[2],
        )
        l_endpoint = math.sqrt(sum(value * value for value in differences)) / initial_l_norm
        max_energy = max(max_energy, energy_endpoint)
        max_l = max(max_l, l_endpoint)
        if source_index is not None:
            source_orbit = simulation.particles[source_index].orbit(primary=simulation.particles[0])
            source_bound = source_orbit.a > 0.0 and source_orbit.e < 1.0 and math.isfinite(source_orbit.a) and math.isfinite(source_orbit.e)
            source_bound_all_steps = bool(source_bound_all_steps and source_bound)
            if source_bound:
                source_q = source_orbit.a * (1.0 - source_orbit.e)
                source_min_q = min(float(source_min_q), source_q)
                source_min_a = min(float(source_min_a), source_orbit.a)
                source_max_a = max(float(source_max_a), source_orbit.a)
    source_a_excursion = (
        max(abs(float(source_min_a) - float(source_initial_a)), abs(float(source_max_a) - float(source_initial_a))) / float(source_initial_a)
        if source_index is not None else None
    )
    return {
        "massive_bodies": names,
        "nominal_steps": steps,
        "dt_years": dt,
        "duration_years": duration,
        "maximum_every_step_relative_energy_drift": max_energy,
        "endpoint_relative_energy_drift": energy_endpoint,
        "maximum_every_step_relative_angular_momentum_vector_drift": max_l,
        "endpoint_relative_angular_momentum_vector_drift": l_endpoint,
        "endpoint_active_state": _active_state(simulation),
        "endpoint_active_state_sha256": _canonical_sha256(_active_state(simulation)),
        "cartesian_state_finite": _cartesian_state_is_finite(simulation),
        "source_bound_at_every_nominal_step": source_bound_all_steps,
        "source_minimum_sampled_q_AU": source_min_q,
        "source_maximum_fractional_semimajor_axis_excursion": source_a_excursion,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _prepare(contract_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError(f"contract schema must be {CONTRACT_SCHEMA}")
    manifest = runtime_source_manifest()
    if contract.get("runner_source_tree_sha256") != manifest["tree_sha256"]:
        raise ValueError("active JX source tree does not match the prelocked contract")
    if contract.get("benchmark_class") != FRAME_LABEL:
        raise ValueError(f"benchmark_class must be {FRAME_LABEL}")
    source_path = _resolve(contract_path, contract["source_state_csv"])
    control_path = _resolve(contract_path, contract["control_state_csv"])
    actual_hashes = {"source": _sha256_file(source_path), "control": _sha256_file(control_path)}
    if actual_hashes != contract["state_sha256"]:
        raise ValueError("state SHA-256 mismatch")
    verified_support: dict[str, dict[str, str]] = {}
    for label, specification in contract.get("support_files", {}).items():
        support_path = _resolve(contract_path, specification["path"])
        actual = _sha256_file(support_path)
        if actual != specification["sha256"]:
            raise ValueError(f"support file SHA-256 mismatch for {label}")
        verified_support[label] = {"path": str(support_path), "sha256": actual}
    source_rows, control_rows = _load_rows(source_path), _load_rows(control_path)
    tolerances = contract["archive_match_tolerances"]
    match = _verify_matched_states(
        source_rows, control_rows,
        _fraction(str(tolerances["position_AU"])),
        _fraction(str(tolerances["velocity_AU_per_year"])),
        _fraction(str(tolerances["mass_Msun"])),
    )
    source_binary, control_binary = _binary64_massive_state(source_rows), _binary64_massive_state(control_rows)
    difference = max(
        abs(left - right)
        for name in CORE_NAMES
        for left, right in zip(source_binary[name], control_binary[name])
    )
    if difference != 0.0:
        raise ValueError("canonical common binary64 massive states are not identical")
    import rebound

    wheel = _resolve(contract_path, contract["dynamics"]["rebound_wheel_file"])
    actual_rebound = {
        "version": rebound.__version__,
        "build": rebound.__build__,
        "binary_sha256": _sha256_file(Path(rebound.clibrebound._name)),
        "wheel_sha256": _sha256_file(wheel),
    }
    expected_rebound = {key: contract["dynamics"][f"rebound_{key}"] for key in actual_rebound}
    if actual_rebound != expected_rebound:
        raise ValueError(f"REBOUND runtime mismatch: {actual_rebound}")
    design = contract["population_design"]
    cells = _grid(design)
    if design.get("phase_generator") != "sha256-counter-open-uniform/v1":
        raise ValueError("unsupported phase generator")
    if design.get("cell_assignment") != "(local_index + 8*block_index) mod 72":
        raise ValueError("unsupported cell assignment")
    blocks = _positive_int(design["independent_blocks"], "independent_blocks")
    tracers_per_block = _positive_int(design["tracers_per_block"], "tracers_per_block")
    if blocks != 10 or tracers_per_block != 1000 or len(cells) != 72:
        raise ValueError("v1 pilot requires 10 blocks, 1000 tracers/block, and 72 cells")
    dynamics = contract["dynamics"]
    if dynamics.get("integrator") != "mercurius" or dynamics.get("testparticle_type") != 0:
        raise ValueError("v1 pilot requires MERCURIUS and testparticle_type=0")
    dt = _positive_float(dynamics["dt_years"], "dt_years")
    duration = _positive_float(dynamics["duration_years"], "duration_years")
    sample = _positive_float(dynamics["online_sample_cadence_years"], "online sample cadence")
    aggregate = _positive_float(dynamics["aggregate_timeseries_cadence_years"], "aggregate cadence")
    checkpoint = _positive_float(dynamics["checkpoint_cadence_years"], "checkpoint cadence")
    for numerator, denominator, label in (
        (sample, dt, "sample/dt"),
        (duration, sample, "duration/sample"),
        (aggregate, sample, "aggregate/sample"),
        (checkpoint, sample, "checkpoint/sample"),
    ):
        _integer_ratio(numerator, denominator, label)
    context = {
        "contract": contract,
        "contract_sha256": _sha256_file(contract_path),
        "software_manifest": manifest,
        "source_path": source_path,
        "control_path": control_path,
        "source_rows": source_rows,
        "control_rows": control_rows,
        "cells": cells,
        "match": {**match, "canonical_binary64_common_state_max_abs_difference": difference},
        "rebound": actual_rebound,
        "verified_support_files": verified_support,
    }
    return contract, context


def _jobs(contract: dict[str, Any], context: dict[str, Any], run_dir: Path, stage: str) -> list[dict[str, Any]]:
    design, dynamics = contract["population_design"], contract["dynamics"]
    audit = contract["timestep_audit"]
    if stage == "primary":
        dt = float(dynamics["dt_years"])
        start, stop = 0, int(design["tracers_per_block"])
    elif stage == "dt_half":
        dt = float(audit["dt_years"])
        start, stop = 0, int(audit["logical_replicate_size_per_block"])
    else:
        raise ValueError(f"unknown stage {stage}")
    sample = float(dynamics["online_sample_cadence_years"])
    duration = float(dynamics["duration_years"])
    aggregate = float(dynamics["aggregate_timeseries_cadence_years"])
    checkpoint = float(dynamics["checkpoint_cadence_years"])
    q_threshold = float(contract["classification"]["q_threshold_AU"])
    hysteresis = float(contract["classification"]["q_hysteresis_AU"])
    cells = [cell.__dict__ for cell in context["cells"]]
    result = []
    for arm, path in (("control", context["control_path"]), ("source", context["source_path"])):
        for block in range(int(design["independent_blocks"])):
            core = {
                "contract_sha256": context["contract_sha256"],
                "arm": arm,
                "stage": stage,
                "state_csv": str(path),
                "state_sha256": contract["state_sha256"][arm],
                "cells": cells,
                "block_index": block,
                "local_start": start,
                "local_stop": stop,
                "base_seed": design["base_seed"],
                "stratum_stride": int(design["block_stratum_stride"]),
                "dt_years": dt,
                "duration_years": duration,
                "sample_years": sample,
                "aggregate_years": aggregate,
                "checkpoint_years": checkpoint,
                "sample_steps": _integer_ratio(sample, dt, "sample/dt"),
                "total_samples": _integer_ratio(duration, sample, "duration/sample"),
                "aggregate_stride_samples": _integer_ratio(aggregate, sample, "aggregate/sample"),
                "checkpoint_stride_samples": _integer_ratio(checkpoint, sample, "checkpoint/sample"),
                "q_lower": q_threshold - hysteresis,
                "q_upper": q_threshold + hysteresis,
            }
            job_sha = hashlib.sha256(_canonical_bytes(core)).hexdigest()
            directory = run_dir / stage / arm / f"block_{block:02d}"
            result.append({**core, "job_sha256": job_sha, "directory": str(directory)})
    return result


def _execute_jobs(jobs: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_initialize_worker,
    ) as executor:
        futures = {executor.submit(_run_block, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            result = future.result()
            results.append(result)
            print(
                f"[{job['stage']}] {job['arm']} block {job['block_index']:02d} "
                f"complete in {result['elapsed_seconds']:.1f}s",
                flush=True,
            )
    return sorted(results, key=lambda row: (row["stage"], row["arm"], row["block_index"]))


def _initialize_worker() -> None:
    """Force REBOUND's process-local first-allocation path before real work."""
    import rebound

    simulation = rebound.Simulation()
    simulation.add(m=1.0)
    simulation.add(primary=simulation.particles[0], m=0.0, a=1.0)


def run_encounter_tail_pilot(
    contract_path: str | Path,
    run_dir: str | Path,
    output_path: str | Path,
    workers: int | None = None,
) -> dict[str, Any]:
    """Run the primary pilot and its prelocked dt/2 audit, then fail closed."""
    contract_file = Path(contract_path).resolve()
    run_root = Path(run_dir).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite locked result: {output}")
    contract, context = _prepare(contract_file)
    configured_workers = int(contract["execution"]["workers"])
    worker_count = workers if workers is not None else configured_workers
    if worker_count <= 0 or worker_count > configured_workers:
        raise ValueError("workers must be positive and may not exceed the prelocked maximum")
    started = time.perf_counter()
    primary_blocks = _execute_jobs(_jobs(contract, context, run_root, "primary"), worker_count)
    half_blocks = _execute_jobs(_jobs(contract, context, run_root, "dt_half"), worker_count)
    primary_rows: dict[str, list[dict[str, str]]] = {"control": [], "source": []}
    half_rows: dict[str, list[dict[str, str]]] = {"control": [], "source": []}
    for block in primary_blocks:
        primary_rows[block["arm"]].extend(_load_csv_rows(Path(block["tracer_csv"])))
    for block in half_blocks:
        half_rows[block["arm"]].extend(_load_csv_rows(Path(block["tracer_csv"])))
    expected_primary = int(contract["population_design"]["independent_blocks"]) * int(contract["population_design"]["tracers_per_block"])
    expected_half = int(contract["population_design"]["independent_blocks"]) * int(contract["timestep_audit"]["logical_replicate_size_per_block"])
    if any(len(primary_rows[arm]) != expected_primary for arm in primary_rows):
        raise RuntimeError("primary row count is incomplete")
    if any(len(half_rows[arm]) != expected_half for arm in half_rows):
        raise RuntimeError("dt/2 audit row count is incomplete")
    effects = _effect(
        primary_rows["control"],
        primary_rows["source"],
        contract["population_design"]["base_seed"],
        int(contract["statistics"]["paired_block_bootstrap_repetitions"]),
    )
    convergence = _convergence_comparison(primary_rows, half_rows, contract["gates"]["timestep_convergence"])
    active_audits = {
        "control": _active_audit(context["control_rows"], float(contract["dynamics"]["dt_years"]), float(contract["dynamics"]["duration_years"])),
        "source": _active_audit(context["source_rows"], float(contract["dynamics"]["dt_years"]), float(contract["dynamics"]["duration_years"])),
    }
    all_blocks = primary_blocks + half_blocks
    restart_exact = all(block["restart_replay_state_hash_exact"] for block in all_blocks)
    rows_complete = all(block["rows_finite_complete"] for block in all_blocks)
    cartesian_finite = all(block["cartesian_state_finite"] for block in all_blocks)
    active_twin_exact = True
    for arm in ("control", "source"):
        expected_state = active_audits[arm]["endpoint_active_state_sha256"]
        for block in primary_blocks:
            if block["arm"] == arm and block["active_endpoint_state_sha256"] != expected_state:
                active_twin_exact = False
    max_energy = max(item["maximum_every_step_relative_energy_drift"] for item in active_audits.values())
    max_angular = max(item["maximum_every_step_relative_angular_momentum_vector_drift"] for item in active_audits.values())
    source_audit = active_audits["source"]
    source_bound = bool(source_audit["source_bound_at_every_nominal_step"])
    source_q_passed = float(source_audit["source_minimum_sampled_q_AU"]) >= float(contract["gates"]["source_minimum_q_AU"])
    source_a_passed = float(source_audit["source_maximum_fractional_semimajor_axis_excursion"]) <= float(contract["gates"]["source_maximum_fractional_semimajor_axis_excursion"])
    checks = {
        "all_required_rows_finite_and_complete": rows_complete,
        "all_endpoint_cartesian_states_finite": cartesian_finite and all(item["cartesian_state_finite"] for item in active_audits.values()),
        "checkpoint_restart_replay_state_hash_exact": restart_exact,
        "massless_active_twin_endpoint_state_exact": active_twin_exact,
        "massive_energy_drift": max_energy <= float(contract["gates"]["max_relative_massive_energy_drift"]),
        "massive_angular_momentum_vector_drift": max_angular <= float(contract["gates"]["max_relative_massive_angular_momentum_vector_drift"]),
        "source_bound_at_every_nominal_step": source_bound,
        "source_minimum_q": source_q_passed,
        "source_semimajor_axis_excursion": source_a_passed,
        "dt_half_population_convergence": convergence["passed"],
    }
    if all(checks.values()):
        verdict = "ENCOUNTER_TAIL_PILOT_PASSED"
    else:
        verdict = "ENCOUNTER_TAIL_PILOT_INVALID"
    result = {
        "schema": RESULT_SCHEMA,
        "verdict": verdict,
        "science_status": "SCREENING_ONLY",
        "benchmark_class": FRAME_LABEL,
        "nonclaim": (
            "The archived common state is a synthetic J2000-ecliptic-like benchmark generated from approximate elements. "
            "This result is not a real-epoch Solar-System population inference, a Planet X detection, or ephemeris validation."
        ),
        "limitations": [
            "No Planet X detection or exclusion and no observational population likelihood.",
            "The 72-cell population is a nearly equally weighted synthetic grid, not an observed population prior.",
            "The planetary state is not DE441 and is not a real common epoch.",
            "Ten thousand years does not test a 100-Myr steady state or 4-Gyr survival.",
            "Candidate 9118 does not represent the full candidate posterior family.",
            "Sampled osculating q below 30 AU is not an observed physical-radius crossing.",
            "The 0.25-year distance stream cannot guarantee complete close-encounter detection.",
            "Massless tracers do not test collective or self gravity.",
            "No sky location, composition, ephemeris refit, survey completeness, or resonance-occupancy claim.",
        ],
        "contract": str(contract_file),
        "contract_sha256": context["contract_sha256"],
        "runner_source_tree_sha256": context["software_manifest"]["tree_sha256"],
        "state_sha256": contract["state_sha256"],
        "rebound_runtime": context["rebound"],
        "verified_support_files": context["verified_support_files"],
        "matched_state_audit": context["match"],
        "design": {
            "grid_cells": len(context["cells"]),
            "independent_blocks": contract["population_design"]["independent_blocks"],
            "primary_tracers_per_arm": expected_primary,
            "dt_half_audit_tracers_per_arm": expected_half,
            "online_sample_cadence_years": contract["dynamics"]["online_sample_cadence_years"],
            "checkpoint_cadence_years": contract["dynamics"]["checkpoint_cadence_years"],
        },
        "population_screening": effects,
        "timestep_convergence": convergence,
        "active_only_every_step_audits": active_audits,
        "maximum_every_step_relative_massive_energy_drift": max_energy,
        "maximum_every_step_relative_massive_angular_momentum_vector_drift": max_angular,
        "checks": checks,
        "block_records": [
            {
                key: block[key]
                for key in (
                    "stage", "arm", "block_index", "tracers", "dt_years", "tracer_csv",
                    "tracer_csv_sha256", "summary_json", "summary_json_sha256",
                    "restart_replay_state_hash_exact", "elapsed_seconds",
                )
            }
            for block in all_blocks
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "next_required_gate": "tail-biased IAS15 epsilon-pair audit under a separate prelocked contract",
    }
    _atomic_json_replace(output, result)
    return result
