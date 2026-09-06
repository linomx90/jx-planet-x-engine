#!/usr/bin/env python3
"""Dedicated 50 kyr JX-E1 synthetic paired-dynamics engineering runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import secrets
import shutil
import statistics
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


CONTRACT_SCHEMA = "jx-e1-long-engineering-contract/v1"
RESULT_SCHEMA = "jx-e1-long-engineering-result/v1"
ARM_SCHEMA = "jx-e1-long-engineering-arm/v1"
SEMANTIC_SCHEMA = "jx-e1-long-engineering-semantic/v1"
PROGRESS_SCHEMA = "jx-e1-long-engineering-progress/v1"
EXPECTED_CONTRACT_POLICY_SHA256 = "b2675b4b8f4baa360abd3706d40d07a88fbaaf5453dda29e8f92a98343964c37"

ARM_CHECK_KEYS = {
    "independent_initial_decoded_states_exact",
    "checkpoint_count_exact",
    "sample_count_exact",
    "final_time_exact",
    "particle_count_unchanged",
    "checkpoint_serialization_decoded_state_exact",
    "direct_vs_chained_restart_exact_at_every_sample",
    "all_sampled_states_and_invariants_finite",
    "no_orbit_conversion_failure",
    "active_energy_drift_within_gate",
    "active_angular_momentum_drift_within_gate",
    "active_linear_momentum_drift_within_gate",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    result = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=constant,
    )
    if not isinstance(result, dict):
        raise ValueError("JSON root must be an object")

    def validate(node: Any) -> None:
        if isinstance(node, dict):
            for child in node.values():
                validate(child)
        elif isinstance(node, list):
            for child in node:
                validate(child)
        elif isinstance(node, float) and not math.isfinite(node):
            raise ValueError("non-finite JSON number")

    validate(result)
    return result


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
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


def serialized_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def contract_policy_sha256(contract: dict[str, Any]) -> str:
    normalized = json.loads(json.dumps(contract, allow_nan=False))
    normalized["runtime"]["runner_sha256"] = "<BOUND_RUNNER_SHA256>"
    normalized["runtime"]["verifier_sha256"] = "<BOUND_VERIFIER_SHA256>"
    return sha256_bytes(canonical_bytes(normalized))


def peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def open_uniform(domain: str, *parts: object) -> float:
    message = "\x1f".join((domain, *(str(part) for part in parts))).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(message).digest()[:8], "big") >> 11
    return (integer + 0.5) / float(1 << 53)


def lhs_values(domain: str, block: int, dimension: str, count: int) -> list[float]:
    order = sorted(
        range(count),
        key=lambda index: hashlib.sha256(
            f"{domain}\x1fperm\x1f{block}\x1f{dimension}\x1f{index}".encode("utf-8")
        ).digest(),
    )
    values = [0.0] * count
    for rank, index in enumerate(order):
        values[index] = (
            rank + open_uniform(domain, "jitter", block, dimension, index)
        ) / count
    return values


def make_tracers(contract: dict[str, Any], block: int) -> list[dict[str, Any]]:
    design = contract["tracer_design"]
    count = int(design["tracers_per_block"])
    domain = design["hash_domain"]
    dimensions = {
        name: lhs_values(domain, block, name, count)
        for name in ("log_a", "q", "cos_i", "Omega", "omega", "M")
    }
    a_min, a_max = map(float, design["a_AU"])
    q_min, q_max = map(float, design["q_AU"])
    i_min, i_max = map(math.radians, map(float, design["i_deg"]))
    cos_min, cos_max = math.cos(i_max), math.cos(i_min)
    result = []
    for index in range(count):
        a = math.exp(math.log(a_min) + dimensions["log_a"][index] * math.log(a_max / a_min))
        q = q_min + dimensions["q"][index] * (q_max - q_min)
        inclination = math.acos(
            max(-1.0, min(1.0, cos_max - dimensions["cos_i"][index] * (cos_max - cos_min)))
        )
        result.append({
            "logical_id": f"b{block:02d}-t{index:03d}",
            "a_AU": a,
            "q_AU": q,
            "e": 1.0 - q / a,
            "i_rad": inclination,
            "Omega_rad": 2.0 * math.pi * dimensions["Omega"][index],
            "omega_rad": 2.0 * math.pi * dimensions["omega"][index],
            "M_rad": 2.0 * math.pi * dimensions["M"][index],
        })
    return result


def decoded_state_digest(simulation: Any) -> str:
    digest = hashlib.sha256()
    digest.update(b"jx-e1-long-decoded-state/v1\0")
    configuration = {
        "t_hex": float(simulation.t).hex(),
        "G_hex": float(simulation.G).hex(),
        "dt_hex": float(simulation.dt).hex(),
        "N": int(simulation.N),
        "N_active": int(simulation.N_active),
        "integrator": str(simulation.integrator),
        "gravity": str(simulation.gravity),
        "collision": str(simulation.collision),
        "boundary": str(simulation.boundary),
        "testparticle_type": int(simulation.testparticle_type),
        "mercurius_r_crit_hill_hex": float(simulation.ri_mercurius.r_crit_hill).hex(),
        "mercurius_safe_mode": int(simulation.ri_mercurius.safe_mode),
        "mercurius_is_synchronized": int(simulation.ri_mercurius.is_synchronized),
        "mercurius_recalculate_coordinates_this_timestep": int(
            simulation.ri_mercurius.recalculate_coordinates_this_timestep
        ),
    }
    digest.update(canonical_bytes(configuration))
    for index, particle in enumerate(simulation.particles):
        digest.update(
            struct.pack(
                "!II8d",
                index,
                int(particle.hash.value),
                particle.m,
                particle.r,
                particle.x,
                particle.y,
                particle.z,
                particle.vx,
                particle.vy,
                particle.vz,
            )
        )
    return digest.hexdigest()


def build_simulation(
    contract: dict[str, Any],
    tracers: list[dict[str, Any]],
    model: dict[str, Any] | None,
    angle: dict[str, Any] | None,
    dt_years: float,
) -> tuple[Any, int, list[str]]:
    import rebound

    benchmark = contract["analytic_benchmark"]
    simulation = rebound.Simulation()
    simulation.G = float(benchmark["G_AU3_Msun_yr2"])
    simulation.add(m=float(benchmark["sun_mass_Msun"]), hash="Sun")
    common_names = ["Sun"]
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
        common_names.append(body["name"])
    if model is not None:
        if angle is None:
            raise ValueError("M1 requires an angular completion")
        omega = (float(angle["varpi_deg"]) - float(angle["Omega_deg"])) % 360.0
        simulation.add(
            primary=simulation.particles[0],
            m=float(model["mass_Mearth"]) * float(benchmark["earth_to_sun_mass_ratio"]),
            a=float(model["a_AU"]),
            e=float(model["e"]),
            inc=math.radians(float(model["i_deg"])),
            Omega=math.radians(float(angle["Omega_deg"])),
            omega=math.radians(omega),
            M=math.radians(float(angle["M_deg"])),
            hash=f"P9_{model['id']}_{angle['id']}",
        )
    simulation.N_active = simulation.N
    simulation.testparticle_type = 0
    simulation.integrator = "mercurius"
    simulation.dt = dt_years
    simulation.ri_mercurius.r_crit_hill = float(contract["dynamics"]["mercurius_r_crit_hill"])
    if simulation.ri_mercurius.r_crit_hill != float(contract["dynamics"]["mercurius_r_crit_hill"]):
        raise RuntimeError("r_crit_hill readback mismatch")
    simulation.collision = "none"
    tracer_start = simulation.N
    for tracer in tracers:
        simulation.add(
            primary=simulation.particles[0],
            m=0.0,
            a=tracer["a_AU"],
            e=tracer["e"],
            inc=tracer["i_rad"],
            Omega=tracer["Omega_rad"],
            omega=tracer["omega_rad"],
            M=tracer["M_rad"],
            hash=tracer["logical_id"],
        )
    return simulation, tracer_start, common_names


def particle_state_bytes(particle: Any, sun: Any) -> bytes:
    return struct.pack(
        "!8d",
        particle.m,
        particle.r,
        particle.x - sun.x,
        particle.y - sun.y,
        particle.z - sun.z,
        particle.vx - sun.vx,
        particle.vy - sun.vy,
        particle.vz - sun.vz,
    )


def common_initial_digest(
    simulation: Any,
    tracer_start: int,
    common_names: list[str],
    tracers: list[dict[str, Any]],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"jx-e1-long-common-initial/v1\0")
    sun = simulation.particles[0]
    for name in common_names:
        digest.update(name.encode("ascii"))
        digest.update(particle_state_bytes(simulation.particles[name], sun))
    for offset, tracer in enumerate(tracers):
        digest.update(tracer["logical_id"].encode("ascii"))
        digest.update(particle_state_bytes(simulation.particles[tracer_start + offset], sun))
    return digest.hexdigest()


def vector_norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def angular_tuple(simulation: Any) -> tuple[float, float, float]:
    vector = simulation.angular_momentum()
    return float(vector.x), float(vector.y), float(vector.z)


def active_linear_momentum_tuple(simulation: Any) -> tuple[float, float, float]:
    return tuple(
        sum(
            simulation.particles[index].m
            * getattr(simulation.particles[index], velocity)
            for index in range(simulation.N_active)
        )
        for velocity in ("vx", "vy", "vz")
    )


def scalar_drift(current: float, initial: float) -> float:
    return abs(current - initial) / max(abs(initial), sys.float_info.min)


def vector_drift(
    current: tuple[float, float, float],
    initial: tuple[float, float, float],
) -> float:
    return vector_norm(tuple(current[index] - initial[index] for index in range(3))) / max(
        vector_norm(initial), sys.float_info.min
    )


def validate_cartesian_and_invariants(
    simulation: Any,
) -> tuple[float, tuple[float, float, float], tuple[float, float, float]]:
    for index, particle in enumerate(simulation.particles):
        for field in ("m", "r", "x", "y", "z", "vx", "vy", "vz"):
            if not math.isfinite(float(getattr(particle, field))):
                raise RuntimeError(f"non-finite particle field at index {index}: {field}")
    energy = float(simulation.energy())
    angular = angular_tuple(simulation)
    linear = active_linear_momentum_tuple(simulation)
    if (
        not math.isfinite(energy)
        or not all(math.isfinite(value) for value in angular)
        or not all(math.isfinite(value) for value in linear)
    ):
        raise RuntimeError("non-finite active energy, angular momentum, or linear momentum")
    return energy, angular, linear


def blank_tracker(tracers: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(tracers)
    return {
        "logical_ids": [tracer["logical_id"] for tracer in tracers],
        "ever_q_lt_30_sampled": [False] * count,
        "ever_i_gt_40_sampled": [False] * count,
        "ever_i_gt_60_sampled": [False] * count,
        "ever_unbound_sampled": [False] * count,
        "orbit_conversion_failure": [False] * count,
        "minimum_neptune_distance_AU_sampled": [None] * count,
        "minimum_p9_distance_AU_sampled": [None] * count,
        "final_bound": [False] * count,
        "final_q_AU": [None] * count,
        "final_i_deg": [None] * count,
        "sample_count": 0,
    }


def sample_tracker(
    simulation: Any,
    tracer_start: int,
    tracker: dict[str, Any],
) -> None:
    sun = simulation.particles[0]
    neptune = simulation.particles["Neptune"]
    p9 = simulation.particles[5] if simulation.N_active == 6 else None
    count = len(tracker["logical_ids"])
    for offset in range(count):
        particle = simulation.particles[tracer_start + offset]
        try:
            orbit = particle.orbit(primary=sun)
            q = orbit.a * (1.0 - orbit.e)
            inclination = math.degrees(orbit.inc)
            if not all(math.isfinite(value) for value in (orbit.a, orbit.e, q, inclination)):
                raise ValueError("non-finite osculating element")
            bound = orbit.a > 0.0 and orbit.e < 1.0
            if not bound:
                tracker["ever_unbound_sampled"][offset] = True
        except (ValueError, ZeroDivisionError, OverflowError):
            tracker["orbit_conversion_failure"][offset] = True
            bound, q, inclination = False, None, None
        tracker["final_bound"][offset] = bound
        tracker["final_q_AU"][offset] = q if bound else None
        tracker["final_i_deg"][offset] = inclination if bound else None
        if bound:
            tracker["ever_q_lt_30_sampled"][offset] |= bool(q < 30.0)
            tracker["ever_i_gt_40_sampled"][offset] |= bool(inclination > 40.0)
            tracker["ever_i_gt_60_sampled"][offset] |= bool(inclination > 60.0)
        for name, body in (("neptune", neptune), ("p9", p9)):
            if body is None:
                continue
            distance = math.sqrt(
                (particle.x - body.x) ** 2
                + (particle.y - body.y) ** 2
                + (particle.z - body.z) ** 2
            )
            key = f"minimum_{name}_distance_AU_sampled"
            old = tracker[key][offset]
            if old is None or distance < old:
                tracker[key][offset] = distance
    tracker["sample_count"] += 1


def particle_diagnostics(tracker: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for index, logical_id in enumerate(tracker["logical_ids"]):
        result.append({
            "logical_id": logical_id,
            "bound_final": tracker["final_bound"][index],
            "final_q_AU": tracker["final_q_AU"][index],
            "final_i_deg": tracker["final_i_deg"][index],
            "ever_q_lt_30_sampled": tracker["ever_q_lt_30_sampled"][index],
            "ever_i_gt_40_sampled": tracker["ever_i_gt_40_sampled"][index],
            "ever_i_gt_60_sampled": tracker["ever_i_gt_60_sampled"][index],
            "ever_unbound_sampled": tracker["ever_unbound_sampled"][index],
            "orbit_conversion_failure": tracker["orbit_conversion_failure"][index],
            "minimum_neptune_distance_AU_sampled": tracker[
                "minimum_neptune_distance_AU_sampled"
            ][index],
            "minimum_p9_distance_AU_sampled": tracker["minimum_p9_distance_AU_sampled"][index],
        })
    return result


def summary_from_particles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    q = [row["final_q_AU"] for row in rows if row["final_q_AU"] is not None]
    inclination = [row["final_i_deg"] for row in rows if row["final_i_deg"] is not None]
    neptune = [
        row["minimum_neptune_distance_AU_sampled"]
        for row in rows
        if row["minimum_neptune_distance_AU_sampled"] is not None
    ]
    p9 = [
        row["minimum_p9_distance_AU_sampled"]
        for row in rows
        if row["minimum_p9_distance_AU_sampled"] is not None
    ]
    return {
        "tracer_count": count,
        "bound_fraction_final": sum(row["bound_final"] for row in rows) / count,
        "ever_q_lt_30_fraction_sampled": sum(row["ever_q_lt_30_sampled"] for row in rows) / count,
        "ever_i_gt_40_fraction_sampled": sum(row["ever_i_gt_40_sampled"] for row in rows) / count,
        "ever_i_gt_60_fraction_sampled": sum(row["ever_i_gt_60_sampled"] for row in rows) / count,
        "ever_unbound_fraction_sampled": sum(row["ever_unbound_sampled"] for row in rows) / count,
        "orbit_conversion_failure_count": sum(row["orbit_conversion_failure"] for row in rows),
        "median_final_q_AU": statistics.median(q) if q else None,
        "median_final_i_deg": statistics.median(inclination) if inclination else None,
        "minimum_neptune_distance_AU_sampled": min(neptune) if neptune else None,
        "minimum_p9_distance_AU_sampled": min(p9) if p9 else None,
    }


def atomic_checkpoint(
    path: Path,
    simulation: Any,
    output_dir: Path,
    contract: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    caps = contract["resource_caps"]
    maximum_checkpoint_bytes = int(caps["max_checkpoint_bytes"])
    if output_bytes(output_dir) + maximum_checkpoint_bytes > int(caps["max_output_bytes"]):
        raise RuntimeError("insufficient locked output budget for another checkpoint")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    os.unlink(temporary)
    try:
        simulation.save_to_file(temporary, delete_file=True)
        if Path(temporary).stat().st_size > maximum_checkpoint_bytes:
            raise RuntimeError("checkpoint byte cap exceeded")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def enforce_resources(
    contract: dict[str, Any],
    output_dir: Path,
    arm_started: float,
    cumulative_elapsed_before_arm: float,
) -> None:
    caps = contract["resource_caps"]
    current_arm_elapsed = time.perf_counter() - arm_started
    if current_arm_elapsed > float(caps["max_wall_seconds_per_arm"]):
        raise RuntimeError("per-arm wall-time cap exceeded")
    if cumulative_elapsed_before_arm + current_arm_elapsed > float(
        caps["max_wall_seconds_total"]
    ):
        raise RuntimeError("total wall-time cap exceeded")
    if peak_rss_bytes() > int(caps["max_peak_rss_bytes"]):
        raise RuntimeError("peak-RSS cap exceeded")
    if shutil.disk_usage(output_dir).free < int(caps["minimum_free_disk_bytes"]):
        raise RuntimeError("free-disk floor violated")


def enforce_output_size(contract: dict[str, Any], output_dir: Path, extra_bytes: int = 0) -> None:
    size = output_bytes(output_dir) + int(extra_bytes)
    if size > int(contract["resource_caps"]["max_output_bytes"]):
        raise RuntimeError("output-size cap exceeded")


def progress_path(output_dir: Path) -> Path:
    return output_dir / "progress.json"


def load_or_initialize_progress(
    output_dir: Path,
    contract_sha256: str,
) -> dict[str, Any]:
    path = progress_path(output_dir)
    if not path.exists():
        progress = {
            "schema": PROGRESS_SCHEMA,
            "contract_sha256": contract_sha256,
            "cumulative_elapsed_seconds": 0.0,
            "active_attempt": None,
        }
        atomic_json(path, progress)
        return progress
    progress = strict_json(path)
    if set(progress) != {
        "schema",
        "contract_sha256",
        "cumulative_elapsed_seconds",
        "active_attempt",
    }:
        raise ValueError("progress schema keys changed")
    if progress["schema"] != PROGRESS_SCHEMA:
        raise ValueError("progress schema changed")
    if progress["contract_sha256"] != contract_sha256:
        raise ValueError("progress contract hash mismatch")
    if type(progress["cumulative_elapsed_seconds"]) not in (int, float):
        raise ValueError("invalid cumulative progress type")
    cumulative = float(progress["cumulative_elapsed_seconds"])
    if not math.isfinite(cumulative) or cumulative < 0.0:
        raise ValueError("invalid cumulative progress time")
    active = progress["active_attempt"]
    if active is not None:
        if set(active) != {"run_key", "started_unix_seconds"}:
            raise ValueError("invalid active-attempt schema")
        if type(active["started_unix_seconds"]) not in (int, float):
            raise ValueError("invalid active-attempt start-time type")
        started_unix = float(active["started_unix_seconds"])
        now_unix = time.time()
        if not math.isfinite(started_unix) or started_unix < 0.0 or started_unix > now_unix + 1.0:
            raise ValueError("invalid active-attempt start time")
        recovered = max(0.0, now_unix - started_unix)
        progress["cumulative_elapsed_seconds"] = cumulative + recovered
        progress["active_attempt"] = None
        atomic_json(path, progress)
    return progress


def start_progress_attempt(
    output_dir: Path,
    progress: dict[str, Any],
    run_key: str,
) -> None:
    if progress["active_attempt"] is not None:
        raise RuntimeError("another arm attempt is already active")
    progress["active_attempt"] = {
        "run_key": run_key,
        "started_unix_seconds": time.time(),
    }
    atomic_json(progress_path(output_dir), progress)


def finish_progress_attempt(
    output_dir: Path,
    progress: dict[str, Any],
    run_key: str,
    elapsed_seconds: float,
) -> None:
    if not math.isfinite(float(elapsed_seconds)) or float(elapsed_seconds) < 0.0:
        raise ValueError("invalid attempt elapsed time")
    active = progress["active_attempt"]
    if active is None or active.get("run_key") != run_key:
        raise RuntimeError("progress active-attempt mismatch")
    progress["cumulative_elapsed_seconds"] = (
        float(progress["cumulative_elapsed_seconds"]) + float(elapsed_seconds)
    )
    progress["active_attempt"] = None
    atomic_json(progress_path(output_dir), progress)


def arm_key(model: dict[str, Any] | None, angle: dict[str, Any] | None, block: int) -> str:
    if model is None:
        return f"M0-b{block:02d}"
    return f"{model['id']}-{angle['id']}-b{block:02d}"


def run_arm(
    contract: dict[str, Any],
    contract_sha256: str,
    output_dir: Path,
    run_key: str,
    arm_class: str,
    block: int,
    model: dict[str, Any] | None,
    angle: dict[str, Any] | None,
    dt_years: float,
    primary_run_key: str | None,
    cumulative_elapsed_before_arm: float,
) -> dict[str, Any]:
    import rebound

    arm_started = time.perf_counter()
    tracers = make_tracers(contract, block)
    direct, direct_start, common_names = build_simulation(contract, tracers, model, angle, dt_years)
    resumed, resumed_start, resumed_names = build_simulation(contract, tracers, model, angle, dt_years)
    if common_names != resumed_names or direct_start != resumed_start:
        raise RuntimeError("independent builders disagree on layout")
    initial_direct = decoded_state_digest(direct)
    initial_resumed = decoded_state_digest(resumed)
    if initial_direct != initial_resumed:
        raise RuntimeError("independent initial decoded states differ")
    common_initial = common_initial_digest(resumed, resumed_start, common_names, tracers)
    initial_energy, initial_angular, initial_linear = validate_cartesian_and_invariants(resumed)
    direct_energy, direct_angular, direct_linear = validate_cartesian_and_invariants(direct)
    if (
        initial_energy != direct_energy
        or initial_angular != direct_angular
        or initial_linear != direct_linear
    ):
        raise RuntimeError("independent initial invariants differ")
    tracker = blank_tracker(tracers)
    sample_tracker(resumed, resumed_start, tracker)
    maximum_energy_drift = 0.0
    maximum_angular_drift = 0.0
    maximum_linear_drift = 0.0
    sampled_stream = hashlib.sha256()
    sampled_stream.update(b"jx-e1-long-sampled-stream/v1\0")
    sampled_stream.update(run_key.encode("ascii"))
    sampled_stream.update(bytes.fromhex(initial_resumed))
    duration = float(contract["dynamics"]["duration_years"])
    sample_cadence = float(contract["dynamics"]["sample_cadence_years"])
    checkpoint_cadence = float(contract["dynamics"]["checkpoint_cadence_years"])
    sample_count = int(round(duration / sample_cadence))
    checkpoint_stride = int(round(checkpoint_cadence / sample_cadence))
    checkpoint_records = []
    restart_exact_every_sample = True
    first_restart_mismatch_year = None
    expected_particles = resumed.N

    for sample_index in range(1, sample_count + 1):
        target = sample_cadence * sample_index
        direct.integrate(target, exact_finish_time=1)
        resumed.integrate(target, exact_finish_time=1)
        direct_state = decoded_state_digest(direct)
        resumed_state = decoded_state_digest(resumed)
        if direct_state != resumed_state:
            restart_exact_every_sample = False
            first_restart_mismatch_year = target
            raise RuntimeError(f"direct/resumed decoded-state mismatch at {target} yr")
        energy, angular, linear = validate_cartesian_and_invariants(resumed)
        direct_energy, direct_angular, direct_linear = validate_cartesian_and_invariants(direct)
        if energy != direct_energy or angular != direct_angular or linear != direct_linear:
            raise RuntimeError(f"direct/resumed active invariants mismatch at {target} yr")
        maximum_energy_drift = max(maximum_energy_drift, scalar_drift(energy, initial_energy))
        maximum_angular_drift = max(maximum_angular_drift, vector_drift(angular, initial_angular))
        maximum_linear_drift = max(maximum_linear_drift, vector_drift(linear, initial_linear))
        sample_tracker(resumed, resumed_start, tracker)
        sampled_stream.update(struct.pack("!d", target))
        sampled_stream.update(bytes.fromhex(resumed_state))
        enforce_resources(
            contract,
            output_dir,
            arm_started,
            cumulative_elapsed_before_arm,
        )
        if sample_index % checkpoint_stride == 0:
            checkpoint_index = sample_index // checkpoint_stride
            checkpoint_path = output_dir / "checkpoints" / run_key / f"checkpoint_{checkpoint_index:02d}.bin"
            state_before = decoded_state_digest(resumed)
            atomic_checkpoint(checkpoint_path, resumed, output_dir, contract)
            loaded = rebound.Simulation(str(checkpoint_path))
            state_after = decoded_state_digest(loaded)
            exact = state_before == state_after
            if not exact:
                raise RuntimeError(f"checkpoint decoded state changed at {target} yr")
            if loaded.ri_mercurius.r_crit_hill != float(contract["dynamics"]["mercurius_r_crit_hill"]):
                raise RuntimeError("checkpoint r_crit_hill readback mismatch")
            checkpoint_records.append({
                "checkpoint_index": checkpoint_index,
                "time_year": target,
                "decoded_state_sha256": state_after,
                "container_sha256_provenance_only": sha256_file(checkpoint_path),
                "container_bytes": checkpoint_path.stat().st_size,
                "relative_path": str(checkpoint_path.relative_to(output_dir)),
            })
            enforce_output_size(contract, output_dir)
            resumed = loaded
    if resumed.N != expected_particles or direct.N != expected_particles:
        raise RuntimeError("particle count changed")
    particles = particle_diagnostics(tracker)
    arm_checks = {
        "independent_initial_decoded_states_exact": initial_direct == initial_resumed,
        "checkpoint_count_exact": len(checkpoint_records)
        == int(round(duration / checkpoint_cadence)),
        "sample_count_exact": tracker["sample_count"] == sample_count + 1,
        "final_time_exact": float(resumed.t) == duration and float(direct.t) == duration,
        "particle_count_unchanged": resumed.N == direct.N == expected_particles,
        "checkpoint_serialization_decoded_state_exact": True,
        "direct_vs_chained_restart_exact_at_every_sample": restart_exact_every_sample,
        "all_sampled_states_and_invariants_finite": True,
        "no_orbit_conversion_failure": not any(
            row["orbit_conversion_failure"] for row in particles
        ),
        "active_energy_drift_within_gate": maximum_energy_drift
        <= float(contract["gates"]["max_relative_active_energy_drift"]),
        "active_angular_momentum_drift_within_gate": maximum_angular_drift
        <= float(contract["gates"]["max_relative_active_angular_momentum_vector_drift"]),
        "active_linear_momentum_drift_within_gate": maximum_linear_drift
        <= float(contract["gates"]["max_relative_active_linear_momentum_vector_drift"]),
    }
    if set(arm_checks) != ARM_CHECK_KEYS:
        raise RuntimeError("internal arm check-key set changed")
    semantic = {
        "schema": ARM_SCHEMA,
        "contract_sha256": contract_sha256,
        "run_key": run_key,
        "arm_class": arm_class,
        "block": block,
        "model_id": None if model is None else model["id"],
        "angle_id": None if angle is None else angle["id"],
        "dt_years": dt_years,
        "duration_years": duration,
        "primary_run_key": primary_run_key,
        "initial_common_state_sha256": common_initial,
        "initial_decoded_state_sha256": initial_resumed,
        "endpoint_decoded_state_sha256": decoded_state_digest(resumed),
        "sampled_state_stream_sha256": sampled_stream.hexdigest(),
        "tracker_sha256": sha256_bytes(canonical_bytes(particles)),
        "sample_count": tracker["sample_count"],
        "checkpoint_decoded_states": [
            {
                "checkpoint_index": row["checkpoint_index"],
                "time_year": row["time_year"],
                "decoded_state_sha256": row["decoded_state_sha256"],
            }
            for row in checkpoint_records
        ],
        "maximum_relative_active_energy_drift": maximum_energy_drift,
        "maximum_relative_active_angular_momentum_vector_drift": maximum_angular_drift,
        "maximum_relative_active_linear_momentum_vector_drift": maximum_linear_drift,
        "first_restart_mismatch_year": first_restart_mismatch_year,
        "particle_diagnostics": particles,
        "summary": summary_from_particles(particles),
        "checks": arm_checks,
    }
    record = {
        "schema": ARM_SCHEMA,
        "semantic": semantic,
        "semantic_sha256": sha256_bytes(canonical_bytes(semantic)),
        "provenance": {
            "elapsed_seconds": time.perf_counter() - arm_started,
            "peak_rss_bytes": peak_rss_bytes(),
            "checkpoint_containers": checkpoint_records,
        },
    }
    enforce_resources(
        contract,
        output_dir,
        arm_started,
        cumulative_elapsed_before_arm,
    )
    enforce_output_size(contract, output_dir, len(serialized_json_bytes(record)))
    return record


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_summary(source: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    source_rows = source["semantic"]["particle_diagnostics"]
    control_rows = control["semantic"]["particle_diagnostics"]
    if [row["logical_id"] for row in source_rows] != [row["logical_id"] for row in control_rows]:
        raise ValueError("paired particle identities differ")
    q, inclination = [], []
    for left, right in zip(source_rows, control_rows):
        if left["bound_final"] and right["bound_final"]:
            q.append(abs(left["final_q_AU"] - right["final_q_AU"]))
            inclination.append(abs(left["final_i_deg"] - right["final_i_deg"]))
    count = len(source_rows)
    boolean_keys = (
        "bound_final",
        "ever_q_lt_30_sampled",
        "ever_i_gt_40_sampled",
        "ever_i_gt_60_sampled",
        "ever_unbound_sampled",
    )
    result = {
        "source_run_key": source["semantic"]["run_key"],
        "control_run_key": control["semantic"]["run_key"],
        "paired_bound_count": len(q),
        "median_absolute_final_q_change_AU": statistics.median(q) if q else None,
        "p90_absolute_final_q_change_AU": percentile(q, 0.9),
        "median_absolute_final_i_change_deg": statistics.median(inclination) if inclination else None,
        "p90_absolute_final_i_change_deg": percentile(inclination, 0.9),
    }
    for key in boolean_keys:
        result[f"source_minus_control_{key}_fraction"] = (
            sum(row[key] for row in source_rows) - sum(row[key] for row in control_rows)
        ) / count
    return result


def timestep_comparison(audit: dict[str, Any], primary: dict[str, Any]) -> dict[str, Any]:
    left = audit["semantic"]["particle_diagnostics"]
    right = primary["semantic"]["particle_diagnostics"]
    if [row["logical_id"] for row in left] != [row["logical_id"] for row in right]:
        raise ValueError("timestep particle identities differ")
    boolean_keys = (
        "bound_final",
        "ever_q_lt_30_sampled",
        "ever_i_gt_40_sampled",
        "ever_i_gt_60_sampled",
        "ever_unbound_sampled",
        "orbit_conversion_failure",
    )
    identities = all(a[key] == b[key] for a, b in zip(left, right) for key in boolean_keys)
    q, inclination = [], []
    for a, b in zip(left, right):
        if a["bound_final"] and b["bound_final"]:
            q.append(abs(a["final_q_AU"] - b["final_q_AU"]))
            inclination.append(abs(a["final_i_deg"] - b["final_i_deg"]))
    return {
        "audit_run_key": audit["semantic"]["run_key"],
        "primary_run_key": primary["semantic"]["run_key"],
        "paired_bound_count": len(q),
        "sampled_event_and_bound_identities_exact": identities,
        "maximum_absolute_final_q_difference_AU": max(q) if q else None,
        "maximum_absolute_final_i_difference_deg": max(inclination) if inclination else None,
    }


def validate_contract(contract: dict[str, Any], contract_path: Path) -> None:
    if contract_policy_sha256(contract) != EXPECTED_CONTRACT_POLICY_SHA256:
        raise ValueError("contract policy digest mismatch")
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("unexpected contract schema")
    if contract.get("experiment_id") != "jx-e1-p9-9x4-50k-v1":
        raise ValueError("unexpected experiment ID")
    if contract.get("claim_ceiling") != "ENGINEERING_SURROGATE_ONLY":
        raise ValueError("claim ceiling changed")
    if contract.get("permissions") != {
        "local_cpu_engineering_execution_authorized": True,
        "jx_o2_execution_authorized": False,
        "observed_data_access_authorized": False,
        "scientific_planet_x_claim_authorized": False,
        "gpu_execution_authorized": False,
    }:
        raise ValueError("permission boundary changed")
    parent = contract["parent_smoke_gate"]
    parent_contract_path = (contract_path.parent / parent["contract_path"]).resolve()
    parent_receipt_path = (contract_path.parent / parent["replay_receipt_path"]).resolve()
    if sha256_file(parent_contract_path) != parent["contract_sha256"]:
        raise ValueError("parent smoke contract hash mismatch")
    if sha256_file(parent_receipt_path) != parent["replay_receipt_sha256"]:
        raise ValueError("parent smoke replay-receipt hash mismatch")
    parent_receipt = strict_json(parent_receipt_path)
    if parent_receipt.get("verdict") != parent["required_replay_verdict"]:
        raise ValueError("parent smoke replay verdict is not eligible")
    if parent_receipt.get("semantic_sha256") != parent["smoke_semantic_sha256"]:
        raise ValueError("parent smoke semantic hash mismatch")
    if parent.get("effect_direction_or_favorable_row_used_for_scaling") is not False:
        raise ValueError("scaling must be outcome-direction blind")
    expected_models = [
        ("CI01", 5.0, 367.0, 0.20, 20.0),
        ("CI02", 5.0, 420.0, 0.35, 20.0),
        ("CI03", 5.0, 480.0, 0.50, 20.0),
        ("CI04", 7.07, 356.0, 0.20, 20.0),
        ("CI05", 7.07, 433.0, 0.35, 20.0),
        ("CI06", 7.07, 497.0, 0.50, 20.0),
        ("CI07", 10.0, 356.0, 0.20, 20.0),
        ("CI08", 10.0, 433.0, 0.35, 20.0),
        ("CI09", 10.0, 540.0, 0.50, 20.0),
    ]
    actual_models = [
        (item["id"], float(item["mass_Mearth"]), float(item["a_AU"]), float(item["e"]), float(item["i_deg"]))
        for item in contract["model_grid"]
    ]
    if actual_models != expected_models:
        raise ValueError("physical grid changed")
    expected_angles = [
        ("A", 225.0, 280.0, 14.4),
        ("B", 135.0, 80.0, 86.4),
        ("C", 315.0, 200.0, 158.4),
        ("D", 22.5, 320.0, 230.4),
    ]
    actual_angles = [
        (item["id"], float(item["Omega_deg"]), float(item["varpi_deg"]), float(item["M_deg"]))
        for item in contract["angle_grid"]
    ]
    if actual_angles != expected_angles:
        raise ValueError("angle grid changed")
    dynamics = contract["dynamics"]
    if dynamics != {
        "integrator": "mercurius",
        "dt_years": 0.125,
        "duration_years": 50000.0,
        "sample_cadence_years": 10.0,
        "checkpoint_cadence_years": 5000.0,
        "mercurius_r_crit_hill": 3.0,
        "testparticle_type": 0,
    }:
        raise ValueError("long-stage dynamics changed")
    if contract["tracer_design"]["blocks"] != 2 or contract["tracer_design"]["tracers_per_block"] != 32:
        raise ValueError("tracer design cardinality changed")
    if contract["timestep_audit"] != {
        "dt_years": 0.0625,
        "both_blocks": True,
        "base_run_keys_without_block": [
            "M0",
            "CI01-A",
            "CI03-C",
            "CI05-C",
            "CI06-C",
            "CI07-B",
            "CI09-A",
            "CI09-D",
        ],
        "selection_basis": (
            "Predeclared coverage and numerical-stress sentinels, including both "
            "lowest-perihelion physical rows; never effect direction."
        ),
    }:
        raise ValueError("timestep audit selection changed")
    runtime = contract["runtime"]
    if runtime["runner_sha256"] != sha256_file(Path(__file__).resolve()):
        raise ValueError("runner hash mismatch")
    verifier_path = Path(__file__).resolve().with_name("verify_long_replay.py")
    if runtime["verifier_sha256"] != sha256_file(verifier_path):
        raise ValueError("verifier hash mismatch")
    if contract_path.name != "contract_v1.json":
        raise ValueError("expected contract_v1.json")
    if contract["resource_caps"] != {
        "workers": 1,
        "max_wall_seconds_per_arm": 600.0,
        "max_wall_seconds_total": 3600.0,
        "max_peak_rss_bytes": 2147483648,
        "max_output_bytes": 67108864,
        "max_checkpoint_bytes": 1048576,
        "minimum_free_disk_bytes": 1073741824,
        "gpu_used": False,
    }:
        raise ValueError("resource caps changed")


def validate_runtime(contract: dict[str, Any]) -> dict[str, Any]:
    import rebound

    binary = Path(rebound.clibrebound._name).resolve()
    actual = {
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "rebound_version": rebound.__version__,
        "rebound_build": rebound.__build__,
        "rebound_binary_sha256": sha256_file(binary),
    }
    expected = {key: contract["runtime"][key] for key in actual}
    if actual != expected:
        raise ValueError(f"runtime mismatch: {actual}")
    return {**actual, "rebound_binary_path": str(binary)}


def record_path(output_dir: Path, run_key: str) -> Path:
    return output_dir / "arm_records" / f"{run_key}.json"


def load_completed_record(
    path: Path,
    output_dir: Path,
    contract: dict[str, Any],
    contract_sha256: str,
    expected_run_key: str,
    expected_arm_class: str,
    expected_block: int,
    expected_model: dict[str, Any] | None,
    expected_angle: dict[str, Any] | None,
    expected_dt_years: float,
    expected_primary_run_key: str | None,
) -> dict[str, Any] | None:
    import rebound

    if not path.is_file():
        return None
    if path != record_path(output_dir, expected_run_key):
        raise ValueError("arm record path does not match expected run key")
    record = strict_json(path)
    if set(record) != {"schema", "semantic", "semantic_sha256", "provenance"}:
        raise ValueError(f"arm top-level keys changed: {path}")
    if record.get("schema") != ARM_SCHEMA:
        raise ValueError(f"invalid arm schema: {path}")
    semantic = record["semantic"]
    expected_semantic_keys = {
        "schema",
        "contract_sha256",
        "run_key",
        "arm_class",
        "block",
        "model_id",
        "angle_id",
        "dt_years",
        "duration_years",
        "primary_run_key",
        "initial_common_state_sha256",
        "initial_decoded_state_sha256",
        "endpoint_decoded_state_sha256",
        "sampled_state_stream_sha256",
        "tracker_sha256",
        "sample_count",
        "checkpoint_decoded_states",
        "maximum_relative_active_energy_drift",
        "maximum_relative_active_angular_momentum_vector_drift",
        "maximum_relative_active_linear_momentum_vector_drift",
        "first_restart_mismatch_year",
        "particle_diagnostics",
        "summary",
        "checks",
    }
    if set(semantic) != expected_semantic_keys:
        raise ValueError(f"arm semantic keys changed: {path}")
    expected_identity = {
        "schema": ARM_SCHEMA,
        "contract_sha256": contract_sha256,
        "run_key": expected_run_key,
        "arm_class": expected_arm_class,
        "block": expected_block,
        "model_id": None if expected_model is None else expected_model["id"],
        "angle_id": None if expected_angle is None else expected_angle["id"],
        "dt_years": expected_dt_years,
        "duration_years": float(contract["dynamics"]["duration_years"]),
        "primary_run_key": expected_primary_run_key,
    }
    for key, value in expected_identity.items():
        if semantic.get(key) != value:
            raise ValueError(f"arm identity mismatch for {key}: {path}")
    if semantic.get("contract_sha256") != contract_sha256:
        raise ValueError(f"arm contract hash mismatch: {path}")
    if record["semantic_sha256"] != sha256_bytes(canonical_bytes(semantic)):
        raise ValueError(f"arm semantic hash mismatch: {path}")
    checks = semantic["checks"]
    if set(checks) != ARM_CHECK_KEYS or any(value is not True for value in checks.values()):
        raise ValueError(f"stored arm is invalid: {path}")
    if semantic["first_restart_mismatch_year"] is not None:
        raise ValueError(f"stored arm has restart mismatch: {path}")
    tracers = make_tracers(contract, expected_block)
    expected_ids = [item["logical_id"] for item in tracers]
    expected_simulation, expected_tracer_start, expected_common_names = build_simulation(
        contract,
        tracers,
        expected_model,
        expected_angle,
        expected_dt_years,
    )
    expected_initial_decoded = decoded_state_digest(expected_simulation)
    expected_common_initial = common_initial_digest(
        expected_simulation,
        expected_tracer_start,
        expected_common_names,
        tracers,
    )
    if semantic["initial_decoded_state_sha256"] != expected_initial_decoded:
        raise ValueError(f"stored arm initial decoded state mismatch: {path}")
    if semantic["initial_common_state_sha256"] != expected_common_initial:
        raise ValueError(f"stored arm initial common state mismatch: {path}")
    particles = semantic["particle_diagnostics"]
    if len(particles) != len(expected_ids) or [item.get("logical_id") for item in particles] != expected_ids:
        raise ValueError(f"stored arm tracer identities changed: {path}")
    expected_particle_keys = {
        "logical_id",
        "bound_final",
        "final_q_AU",
        "final_i_deg",
        "ever_q_lt_30_sampled",
        "ever_i_gt_40_sampled",
        "ever_i_gt_60_sampled",
        "ever_unbound_sampled",
        "orbit_conversion_failure",
        "minimum_neptune_distance_AU_sampled",
        "minimum_p9_distance_AU_sampled",
    }
    boolean_particle_keys = {
        "bound_final",
        "ever_q_lt_30_sampled",
        "ever_i_gt_40_sampled",
        "ever_i_gt_60_sampled",
        "ever_unbound_sampled",
        "orbit_conversion_failure",
    }
    numeric_particle_keys = {
        "final_q_AU",
        "final_i_deg",
        "minimum_neptune_distance_AU_sampled",
        "minimum_p9_distance_AU_sampled",
    }
    for particle in particles:
        if set(particle) != expected_particle_keys:
            raise ValueError(f"stored particle diagnostic keys changed: {path}")
        if any(type(particle[key]) is not bool for key in boolean_particle_keys):
            raise ValueError(f"stored particle diagnostic boolean type changed: {path}")
        for key in numeric_particle_keys:
            value = particle[key]
            if value is not None and (
                type(value) not in (int, float) or not math.isfinite(float(value))
            ):
                raise ValueError(f"stored particle diagnostic numeric value invalid: {path}")
        if particle["bound_final"] != (
            particle["final_q_AU"] is not None and particle["final_i_deg"] is not None
        ):
            raise ValueError(f"stored bound/final-element fields disagree: {path}")
        if particle["bound_final"] and (
            float(particle["final_q_AU"]) < 0.0
            or not 0.0 <= float(particle["final_i_deg"]) <= 180.0
        ):
            raise ValueError(f"stored bound final elements are outside physical domains: {path}")
        if particle["minimum_neptune_distance_AU_sampled"] is None:
            raise ValueError(f"stored Neptune sampled distance is missing: {path}")
        if (expected_model is None) != (
            particle["minimum_p9_distance_AU_sampled"] is None
        ):
            raise ValueError(f"stored P9 sampled-distance applicability changed: {path}")
        if (
            particle["bound_final"]
            and float(particle["final_q_AU"]) < 30.0
            and not particle["ever_q_lt_30_sampled"]
        ):
            raise ValueError(f"stored final q contradicts sampled-event history: {path}")
        if (
            particle["bound_final"]
            and float(particle["final_i_deg"]) > 40.0
            and not particle["ever_i_gt_40_sampled"]
        ):
            raise ValueError(f"stored final inclination contradicts sampled-event history: {path}")
        if (
            particle["bound_final"]
            and float(particle["final_i_deg"]) > 60.0
            and not particle["ever_i_gt_60_sampled"]
        ):
            raise ValueError(f"stored final inclination contradicts sampled-event history: {path}")
        if particle["ever_i_gt_60_sampled"] and not particle["ever_i_gt_40_sampled"]:
            raise ValueError(f"stored inclination-event nesting is inconsistent: {path}")
        if not particle["bound_final"] and not particle["ever_unbound_sampled"]:
            raise ValueError(f"stored final unbound state is absent from sampled history: {path}")
        for key in (
            "minimum_neptune_distance_AU_sampled",
            "minimum_p9_distance_AU_sampled",
        ):
            if particle[key] is not None and float(particle[key]) < 0.0:
                raise ValueError(f"stored sampled distance is negative: {path}")
    if semantic["tracker_sha256"] != sha256_bytes(canonical_bytes(particles)):
        raise ValueError(f"stored arm tracker hash mismatch: {path}")
    if semantic["summary"] != summary_from_particles(particles):
        raise ValueError(f"stored arm summary mismatch: {path}")
    if any(particle["orbit_conversion_failure"] for particle in particles):
        raise ValueError(f"stored arm contains orbit conversion failure: {path}")
    for field, gate in (
        ("maximum_relative_active_energy_drift", "max_relative_active_energy_drift"),
        (
            "maximum_relative_active_angular_momentum_vector_drift",
            "max_relative_active_angular_momentum_vector_drift",
        ),
        (
            "maximum_relative_active_linear_momentum_vector_drift",
            "max_relative_active_linear_momentum_vector_drift",
        ),
    ):
        value = semantic[field]
        if (
            type(value) not in (int, float)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            or float(value) > float(contract["gates"][gate])
        ):
            raise ValueError(f"stored arm drift violates gate for {field}: {path}")
    expected_checkpoints = int(
        round(
            float(contract["dynamics"]["duration_years"])
            / float(contract["dynamics"]["checkpoint_cadence_years"])
        )
    )
    if len(semantic["checkpoint_decoded_states"]) != expected_checkpoints:
        raise ValueError(f"stored arm checkpoint count changed: {path}")
    if semantic["sample_count"] != int(
        round(
            float(contract["dynamics"]["duration_years"])
            / float(contract["dynamics"]["sample_cadence_years"])
        )
    ) + 1:
        raise ValueError(f"stored arm sample count changed: {path}")
    provenance = record["provenance"]
    if set(provenance) != {"elapsed_seconds", "peak_rss_bytes", "checkpoint_containers"}:
        raise ValueError(f"arm provenance keys changed: {path}")
    if type(provenance["elapsed_seconds"]) not in (int, float):
        raise ValueError(f"stored arm elapsed-time type changed: {path}")
    if type(provenance["peak_rss_bytes"]) is not int:
        raise ValueError(f"stored arm peak-RSS type changed: {path}")
    if (
        not math.isfinite(float(provenance["elapsed_seconds"]))
        or float(provenance["elapsed_seconds"]) < 0.0
        or float(provenance["elapsed_seconds"])
        > float(contract["resource_caps"]["max_wall_seconds_per_arm"])
    ):
        raise ValueError(f"stored arm elapsed time exceeds cap: {path}")
    if (
        int(provenance["peak_rss_bytes"]) < 0
        or int(provenance["peak_rss_bytes"])
        > int(contract["resource_caps"]["max_peak_rss_bytes"])
    ):
        raise ValueError(f"stored arm peak RSS exceeds cap: {path}")
    containers = provenance["checkpoint_containers"]
    if len(containers) != expected_checkpoints:
        raise ValueError(f"stored arm checkpoint-container count changed: {path}")
    semantic_by_index = {
        item["checkpoint_index"]: item for item in semantic["checkpoint_decoded_states"]
    }
    if len(semantic_by_index) != expected_checkpoints:
        raise ValueError(f"duplicate semantic checkpoint index: {path}")
    resolved_output = output_dir.resolve()
    for index, container in enumerate(containers, start=1):
        if set(container) != {
            "checkpoint_index",
            "time_year",
            "decoded_state_sha256",
            "container_sha256_provenance_only",
            "container_bytes",
            "relative_path",
        }:
            raise ValueError(f"checkpoint provenance keys changed: {path}")
        expected_time = float(contract["dynamics"]["checkpoint_cadence_years"]) * index
        expected_relative = f"checkpoints/{expected_run_key}/checkpoint_{index:02d}.bin"
        if (
            container["checkpoint_index"] != index
            or container["time_year"] != expected_time
            or container["relative_path"] != expected_relative
        ):
            raise ValueError(f"checkpoint identity mismatch: {path}")
        semantic_checkpoint = semantic_by_index.get(index)
        if semantic_checkpoint != {
            "checkpoint_index": index,
            "time_year": expected_time,
            "decoded_state_sha256": container["decoded_state_sha256"],
        }:
            raise ValueError(f"semantic/provenance checkpoint mismatch: {path}")
        checkpoint_path = (output_dir / expected_relative).resolve()
        if not checkpoint_path.is_relative_to(resolved_output) or not checkpoint_path.is_file():
            raise ValueError(f"checkpoint path unavailable or escaped output: {path}")
        if checkpoint_path.stat().st_size != container["container_bytes"]:
            raise ValueError(f"checkpoint byte count mismatch: {path}")
        if checkpoint_path.stat().st_size > int(contract["resource_caps"]["max_checkpoint_bytes"]):
            raise ValueError(f"checkpoint byte cap exceeded: {path}")
        if sha256_file(checkpoint_path) != container["container_sha256_provenance_only"]:
            raise ValueError(f"checkpoint container hash mismatch: {path}")
        loaded = rebound.Simulation(str(checkpoint_path))
        if decoded_state_digest(loaded) != container["decoded_state_sha256"]:
            raise ValueError(f"checkpoint decoded-state hash mismatch: {path}")
        if float(loaded.t) != expected_time:
            raise ValueError(f"checkpoint time mismatch: {path}")
        expected_active = 5 if expected_model is None else 6
        expected_total = expected_active + int(contract["tracer_design"]["tracers_per_block"])
        if loaded.N_active != expected_active or loaded.N != expected_total:
            raise ValueError(f"checkpoint particle layout mismatch: {path}")
        expected_hash_names = ["Sun", "Jupiter", "Saturn", "Uranus", "Neptune"]
        if expected_model is not None:
            expected_hash_names.append(
                f"P9_{expected_model['id']}_{expected_angle['id']}"
            )
        expected_hash_names.extend(expected_ids)
        actual_hashes = [int(particle.hash.value) for particle in loaded.particles]
        expected_hashes = [int(rebound.hash(name).value) for name in expected_hash_names]
        if actual_hashes != expected_hashes:
            raise ValueError(f"checkpoint logical particle identity mismatch: {path}")
        if float(loaded.dt) != expected_dt_years:
            raise ValueError(f"checkpoint timestep mismatch: {path}")
        if loaded.testparticle_type != int(contract["dynamics"]["testparticle_type"]):
            raise ValueError(f"checkpoint test-particle mode mismatch: {path}")
        if loaded.ri_mercurius.r_crit_hill != float(
            contract["dynamics"]["mercurius_r_crit_hill"]
        ):
            raise ValueError(f"checkpoint Mercurius radius mismatch: {path}")
    final_checkpoint = semantic_by_index[expected_checkpoints]["decoded_state_sha256"]
    if semantic["endpoint_decoded_state_sha256"] != final_checkpoint:
        raise ValueError(f"stored arm endpoint/final-checkpoint mismatch: {path}")
    return record


def output_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def build_matrix_specs(contract: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    models = contract["model_grid"]
    angles = contract["angle_grid"]
    blocks = range(int(contract["tracer_design"]["blocks"]))
    primary_specs: list[dict[str, Any]] = []
    for block in blocks:
        primary_specs.append({
            "run_key": arm_key(None, None, block),
            "arm_class": "PRIMARY",
            "block": block,
            "model": None,
            "angle": None,
            "dt_years": float(contract["dynamics"]["dt_years"]),
            "primary_run_key": None,
        })
    for model in models:
        for angle in angles:
            for block in blocks:
                primary_specs.append({
                    "run_key": arm_key(model, angle, block),
                    "arm_class": "PRIMARY",
                    "block": block,
                    "model": model,
                    "angle": angle,
                    "dt_years": float(contract["dynamics"]["dt_years"]),
                    "primary_run_key": None,
                })
    audit_specs: list[dict[str, Any]] = []
    for base in contract["timestep_audit"]["base_run_keys_without_block"]:
        for block in blocks:
            if base == "M0":
                model, angle = None, None
            else:
                model_id, angle_id = base.split("-")
                model = next(item for item in models if item["id"] == model_id)
                angle = next(item for item in angles if item["id"] == angle_id)
            base_key = arm_key(model, angle, block)
            audit_specs.append({
                "run_key": f"AUDIT-{base_key}",
                "arm_class": "TIMESTEP_AUDIT",
                "block": block,
                "model": model,
                "angle": angle,
                "dt_years": float(contract["timestep_audit"]["dt_years"]),
                "primary_run_key": base_key,
            })
    if len(primary_specs) != 74 or len(audit_specs) != 16:
        raise RuntimeError("internal matrix cardinality changed")
    all_keys = [item["run_key"] for item in primary_specs + audit_specs]
    if len(all_keys) != len(set(all_keys)):
        raise RuntimeError("internal matrix run keys are not unique")
    return primary_specs, audit_specs


def execute(
    contract_path: Path,
    output_dir: Path,
    execution_label: str,
) -> dict[str, Any]:
    contract = strict_json(contract_path)
    validate_contract(contract, contract_path)
    runtime = validate_runtime(contract)
    contract_sha256 = sha256_file(contract_path)
    allowed_labels = contract["replay_policy"]["clean_execution_labels"]
    if execution_label not in allowed_labels:
        raise ValueError("execution label is not predeclared")
    result_path = output_dir / "result_v1.json"
    failure_path = output_dir / "failure_receipt.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite final result: {result_path}")
    if failure_path.exists():
        raise FileExistsError(
            "this output directory already has a permanent failure receipt; use a new clean directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = strict_json(manifest_path)
        if set(manifest) != {
            "schema",
            "experiment_id",
            "contract_sha256",
            "runner_sha256",
            "execution_label",
            "execution_instance_id",
            "runtime",
        }:
            raise ValueError("existing run manifest schema changed")
        if manifest["schema"] != "jx-e1-long-run-manifest/v1":
            raise ValueError("existing run manifest schema changed")
        if manifest["execution_label"] != execution_label:
            raise ValueError("existing run manifest execution label mismatch")
        instance_id = manifest["execution_instance_id"]
        if (
            not isinstance(instance_id, str)
            or len(instance_id) != 32
            or any(character not in "0123456789abcdef" for character in instance_id)
        ):
            raise ValueError("existing run manifest instance ID invalid")
        expected_manifest = {
            "schema": "jx-e1-long-run-manifest/v1",
            "experiment_id": contract["experiment_id"],
            "contract_sha256": contract_sha256,
            "runner_sha256": contract["runtime"]["runner_sha256"],
            "execution_label": execution_label,
            "execution_instance_id": instance_id,
            "runtime": runtime,
        }
        if manifest != expected_manifest:
            raise ValueError("existing run manifest mismatch")
    else:
        manifest = {
            "schema": "jx-e1-long-run-manifest/v1",
            "experiment_id": contract["experiment_id"],
            "contract_sha256": contract_sha256,
            "runner_sha256": contract["runtime"]["runner_sha256"],
            "execution_label": execution_label,
            "execution_instance_id": secrets.token_hex(16),
            "runtime": runtime,
        }
        atomic_json(manifest_path, manifest)
    if shutil.disk_usage(output_dir).free < int(contract["resource_caps"]["minimum_free_disk_bytes"]):
        raise RuntimeError("insufficient free disk at start")
    progress = load_or_initialize_progress(output_dir, contract_sha256)
    max_total = float(contract["resource_caps"]["max_wall_seconds_total"])
    if float(progress["cumulative_elapsed_seconds"]) > max_total:
        raise RuntimeError("persistent cumulative wall-time cap already exceeded")
    primary_specs, audit_specs = build_matrix_specs(contract)
    primary: dict[str, dict[str, Any]] = {}
    audits: dict[str, dict[str, Any]] = {}
    current_attempt_started: float | None = None
    current_attempt_key: str | None = None

    def acquire(spec: dict[str, Any]) -> dict[str, Any]:
        nonlocal current_attempt_started, current_attempt_key
        key = spec["run_key"]
        start_progress_attempt(output_dir, progress, key)
        current_attempt_started = time.perf_counter()
        current_attempt_key = key
        path = record_path(output_dir, key)
        record = load_completed_record(
            path,
            output_dir,
            contract,
            contract_sha256,
            key,
            spec["arm_class"],
            spec["block"],
            spec["model"],
            spec["angle"],
            spec["dt_years"],
            spec["primary_run_key"],
        )
        if record is None:
            record = run_arm(
                contract,
                contract_sha256,
                output_dir,
                key,
                spec["arm_class"],
                spec["block"],
                spec["model"],
                spec["angle"],
                spec["dt_years"],
                spec["primary_run_key"],
                float(progress["cumulative_elapsed_seconds"]),
            )
            projected_record_bytes = len(serialized_json_bytes(record))
            enforce_output_size(contract, output_dir, projected_record_bytes)
            atomic_json(path, record)
        elapsed = time.perf_counter() - current_attempt_started
        finish_progress_attempt(output_dir, progress, key, elapsed)
        current_attempt_started = None
        current_attempt_key = None
        if elapsed > float(contract["resource_caps"]["max_wall_seconds_per_arm"]):
            raise RuntimeError("arm acquisition wall-time cap exceeded")
        if peak_rss_bytes() > int(contract["resource_caps"]["max_peak_rss_bytes"]):
            raise RuntimeError("arm acquisition peak-RSS cap exceeded")
        if shutil.disk_usage(output_dir).free < int(
            contract["resource_caps"]["minimum_free_disk_bytes"]
        ):
            raise RuntimeError("arm acquisition free-disk floor violated")
        if float(progress["cumulative_elapsed_seconds"]) > max_total:
            raise RuntimeError("persistent cumulative wall-time cap exceeded")
        enforce_output_size(contract, output_dir)
        return record

    try:
        for index, spec in enumerate(primary_specs, start=1):
            key = spec["run_key"]
            record = acquire(spec)
            primary[key] = record
            print(f"[primary {index:02d}/{len(primary_specs)}] {key} complete", flush=True)
        for index, spec in enumerate(audit_specs, start=1):
            key = spec["run_key"]
            record = acquire(spec)
            audits[key] = record
            print(f"[audit {index:02d}/{len(audit_specs)}] {key} complete", flush=True)
    except BaseException as exc:
        if current_attempt_started is not None and current_attempt_key is not None:
            elapsed = time.perf_counter() - current_attempt_started
            finish_progress_attempt(output_dir, progress, current_attempt_key, elapsed)
            current_attempt_started = None
            current_attempt_key = None
        failure = {
            "schema": "jx-e1-long-engineering-failure/v1",
            "experiment_id": contract["experiment_id"],
            "contract_sha256": contract_sha256,
            "verdict": "ENGINEERING_LONG_INVALID",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "completed_primary_arms": sorted(primary),
            "completed_audit_arms": sorted(audits),
            "cumulative_elapsed_seconds": progress["cumulative_elapsed_seconds"],
            "nonclaim": contract["mandatory_nonclaim"],
        }
        atomic_json(failure_path, failure)
        raise

    blocks = range(int(contract["tracer_design"]["blocks"]))
    control = {block: primary[f"M0-b{block:02d}"] for block in blocks}
    paired = [
        paired_summary(record, control[record["semantic"]["block"]])
        for key, record in sorted(primary.items())
        if record["semantic"]["model_id"] is not None
    ]
    comparisons = []
    for key, audit in sorted(audits.items()):
        base_key = audit["semantic"]["primary_run_key"]
        comparisons.append(timestep_comparison(audit, primary[base_key]))
    gates = contract["gates"]
    final_peak_rss = peak_rss_bytes()
    final_free_disk = shutil.disk_usage(output_dir).free
    timestep_q_differences = [
        item["maximum_absolute_final_q_difference_AU"]
        for item in comparisons
        if item["maximum_absolute_final_q_difference_AU"] is not None
    ]
    timestep_i_differences = [
        item["maximum_absolute_final_i_difference_deg"]
        for item in comparisons
        if item["maximum_absolute_final_i_difference_deg"] is not None
    ]
    all_records = list(primary.values()) + list(audits.values())
    checks = {
        "complete_primary_matrix": len(primary) == 74,
        "complete_timestep_audit_matrix": len(audits) == 16,
        "paired_diagnostic_count_exact": len(paired) == 72,
        "timestep_comparison_count_exact": len(comparisons) == 16,
        "all_arm_checks_true": all(
            set(record["semantic"]["checks"]) == ARM_CHECK_KEYS
            and all(value is True for value in record["semantic"]["checks"].values())
            for record in all_records
        ),
        "all_M1_common_initial_states_match_block_M0": all(
            record["semantic"]["initial_common_state_sha256"]
            == control[record["semantic"]["block"]]["semantic"]["initial_common_state_sha256"]
            for record in primary.values()
            if record["semantic"]["model_id"] is not None
        ),
        "all_timestep_audit_common_initial_states_match_primary": all(
            audit["semantic"]["initial_common_state_sha256"]
            == primary[audit["semantic"]["primary_run_key"]]["semantic"]["initial_common_state_sha256"]
            for audit in audits.values()
        ),
        "timestep_bound_and_sampled_event_identities_exact": all(
            item["sampled_event_and_bound_identities_exact"] for item in comparisons
        ),
        "timestep_minimum_paired_bound_count_met": min(
            item["paired_bound_count"] for item in comparisons
        ) >= int(gates["minimum_dt_half_paired_bound_count"]),
        "timestep_max_final_q_difference_within_gate": bool(timestep_q_differences)
        and max(timestep_q_differences)
        <= float(gates["max_dt_half_final_q_difference_AU"]),
        "timestep_max_final_i_difference_within_gate": bool(timestep_i_differences)
        and max(timestep_i_differences)
        <= float(gates["max_dt_half_final_i_difference_deg"]),
        "persistent_cumulative_wall_time_within_cap": float(
            progress["cumulative_elapsed_seconds"]
        ) <= max_total,
        "final_peak_rss_within_cap": final_peak_rss
        <= int(contract["resource_caps"]["max_peak_rss_bytes"]),
        "final_free_disk_floor_met": final_free_disk
        >= int(contract["resource_caps"]["minimum_free_disk_bytes"]),
        "output_size_within_cap_before_result": output_bytes(output_dir)
        < int(contract["resource_caps"]["max_output_bytes"]),
        "no_failure_receipt_present": not failure_path.exists(),
    }
    semantic = {
        "schema": SEMANTIC_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "claim_ceiling": contract["claim_ceiling"],
        "contract_sha256": contract_sha256,
        "runtime": {key: value for key, value in runtime.items() if key != "rebound_binary_path"},
        "primary_arms": [primary[key]["semantic"] for key in sorted(primary)],
        "timestep_audit_arms": [audits[key]["semantic"] for key in sorted(audits)],
        "paired_M1_minus_M0_diagnostics": paired,
        "timestep_comparisons": comparisons,
        "checks": checks,
        "replay_status": "PENDING_SEPARATE_CLEAN_EXECUTION_AND_LOCKED_VERIFIER",
        "mandatory_nonclaim": contract["mandatory_nonclaim"],
    }
    semantic_sha256 = sha256_bytes(canonical_bytes(semantic))
    provenance = {
        "execution_label": execution_label,
        "execution_instance_id": manifest["execution_instance_id"],
        "cumulative_elapsed_seconds": progress["cumulative_elapsed_seconds"],
        "peak_rss_bytes": final_peak_rss,
        "free_disk_bytes_before_result": final_free_disk,
        "output_directory": str(output_dir),
        "arm_records": {
            key: record["provenance"] for key, record in sorted({**primary, **audits}.items())
        },
    }
    result = {
        "schema": RESULT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "verdict": (
            "ENGINEERING_LONG_PROVISIONAL_VALID"
            if all(value is True for value in checks.values())
            else "ENGINEERING_LONG_INVALID"
        ),
        "claim_ceiling": contract["claim_ceiling"],
        "semantic": semantic,
        "semantic_sha256": semantic_sha256,
        "provenance": provenance,
        "nonclaim": contract["mandatory_nonclaim"],
    }
    projected_result_bytes = len(serialized_json_bytes(result))
    if output_bytes(output_dir) + projected_result_bytes > int(
        contract["resource_caps"]["max_output_bytes"]
    ):
        failure = {
            "schema": "jx-e1-long-engineering-failure/v1",
            "experiment_id": contract["experiment_id"],
            "contract_sha256": contract_sha256,
            "verdict": "ENGINEERING_LONG_INVALID",
            "exception_type": "ResourceLimitError",
            "message": "final result would exceed the predeclared output-size cap",
            "completed_primary_arms": sorted(primary),
            "completed_audit_arms": sorted(audits),
            "cumulative_elapsed_seconds": progress["cumulative_elapsed_seconds"],
            "nonclaim": contract["mandatory_nonclaim"],
        }
        atomic_json(failure_path, failure)
        raise RuntimeError("final result would exceed output-size cap")
    atomic_json(result_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-label", required=True)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args()
    contract_path = arguments.contract.resolve()
    contract = strict_json(contract_path)
    validate_contract(contract, contract_path)
    runtime = validate_runtime(contract)
    if arguments.execution_label not in contract["replay_policy"]["clean_execution_labels"]:
        raise ValueError("execution label is not predeclared")
    if arguments.validate_only:
        simulation, _, _ = build_simulation(
            contract,
            make_tracers(contract, 0),
            None,
            None,
            float(contract["dynamics"]["dt_years"]),
        )
        print(json.dumps({
            "contract_sha256": sha256_file(contract_path),
            "runtime": runtime,
            "r_crit_hill_readback": simulation.ri_mercurius.r_crit_hill,
            "initial_decoded_state_sha256": decoded_state_digest(simulation),
        }, indent=2))
        return 0
    output_dir = arguments.output_dir.resolve()
    try:
        result = execute(contract_path, output_dir, arguments.execution_label)
    except BaseException as exc:
        result_path = output_dir / "result_v1.json"
        failure_path = output_dir / "failure_receipt.json"
        if output_dir.is_dir() and not result_path.exists() and not failure_path.exists():
            progress = None
            progress_file = progress_path(output_dir)
            if progress_file.is_file():
                try:
                    progress = strict_json(progress_file)
                except Exception:
                    progress = None
            atomic_json(failure_path, {
                "schema": "jx-e1-long-engineering-failure/v1",
                "experiment_id": contract["experiment_id"],
                "contract_sha256": sha256_file(contract_path),
                "verdict": "ENGINEERING_LONG_INVALID",
                "execution_label": arguments.execution_label,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "completed_primary_arms": [],
                "completed_audit_arms": [],
                "cumulative_elapsed_seconds": (
                    None if progress is None else progress.get("cumulative_elapsed_seconds")
                ),
                "nonclaim": contract["mandatory_nonclaim"],
            })
        raise
    print(json.dumps({
        "verdict": result["verdict"],
        "semantic_sha256": result["semantic_sha256"],
        "cumulative_elapsed_seconds": result["provenance"][
            "cumulative_elapsed_seconds"
        ],
        "output": str(arguments.output_dir.resolve() / "result_v1.json"),
    }, indent=2))
    return 0 if result["verdict"] == "ENGINEERING_LONG_PROVISIONAL_VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
