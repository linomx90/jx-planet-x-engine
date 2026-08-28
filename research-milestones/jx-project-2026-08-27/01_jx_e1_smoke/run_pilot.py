#!/usr/bin/env python3
"""Run the locked JX-E1 synthetic paired-dynamics engineering smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import struct
import sys
import time
from pathlib import Path
from typing import Any


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
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("contract root must be an object")

    def finite(node: Any) -> None:
        if isinstance(node, dict):
            for child in node.values():
                finite(child)
        elif isinstance(node, list):
            for child in node:
                finite(child)
        elif isinstance(node, float) and not math.isfinite(node):
            raise ValueError("contract contains a non-finite number")

    finite(value)
    return value


def open_uniform(domain: str, *parts: object) -> float:
    payload = "\x1f".join((domain, *(str(part) for part in parts))).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") >> 11
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
        jitter = open_uniform(domain, "jitter", block, dimension, index)
        values[index] = (rank + jitter) / count
    return values


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def norm3(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def angular_momentum_tuple(simulation: Any) -> tuple[float, float, float]:
    vector = simulation.angular_momentum()
    return float(vector.x), float(vector.y), float(vector.z)


def relative_drift(current: float, initial: float) -> float:
    scale = max(abs(initial), sys.float_info.min)
    return abs(current - initial) / scale


def vector_relative_drift(
    current: tuple[float, float, float], initial: tuple[float, float, float]
) -> float:
    scale = max(norm3(initial), sys.float_info.min)
    return norm3(tuple(current[index] - initial[index] for index in range(3))) / scale


def simulation_digest(simulation: Any) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("!dii", float(simulation.t), int(simulation.N), int(simulation.N_active)))
    digest.update(str(simulation.integrator).encode("ascii"))
    digest.update(struct.pack("!d", float(simulation.dt)))
    for particle in simulation.particles:
        digest.update(
            struct.pack(
                "!7d",
                particle.m,
                particle.x,
                particle.y,
                particle.z,
                particle.vx,
                particle.vy,
                particle.vz,
            )
        )
    return digest.hexdigest()


def particle_state_bytes(particle: Any, sun: Any) -> bytes:
    return struct.pack(
        "!7d",
        particle.m,
        particle.x - sun.x,
        particle.y - sun.y,
        particle.z - sun.z,
        particle.vx - sun.vx,
        particle.vy - sun.vy,
        particle.vz - sun.vz,
    )


def make_tracers(contract: dict[str, Any], block: int, count: int | None = None) -> list[dict[str, Any]]:
    design = contract["tracer_design"]
    total = int(design["tracers_per_block"])
    if count is not None:
        total = count
    domain = design["hash_domain"]
    dimensions = {
        name: lhs_values(domain, block, name, total)
        for name in ("log_a", "q", "cos_i", "Omega", "omega", "M")
    }
    a_min, a_max = map(float, design["a_AU"])
    q_min, q_max = map(float, design["q_AU"])
    i_min, i_max = map(math.radians, map(float, design["i_deg"]))
    cos_min, cos_max = math.cos(i_max), math.cos(i_min)
    result = []
    for index in range(total):
        a = math.exp(math.log(a_min) + dimensions["log_a"][index] * math.log(a_max / a_min))
        q = q_min + dimensions["q"][index] * (q_max - q_min)
        cos_i = cos_max - dimensions["cos_i"][index] * (cos_max - cos_min)
        inclination = math.acos(max(-1.0, min(1.0, cos_i)))
        result.append(
            {
                "logical_id": f"b{block:02d}-t{index:03d}",
                "a_AU": a,
                "q_AU": q,
                "e": 1.0 - q / a,
                "i_rad": inclination,
                "Omega_rad": 2.0 * math.pi * dimensions["Omega"][index],
                "omega_rad": 2.0 * math.pi * dimensions["omega"][index],
                "M_rad": 2.0 * math.pi * dimensions["M"][index],
            }
        )
    return result


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
            raise ValueError("an M1 model requires an angular completion")
        omega_deg = (float(angle["varpi_deg"]) - float(angle["Omega_deg"])) % 360.0
        simulation.add(
            primary=simulation.particles[0],
            m=float(model["mass_Mearth"]) * float(benchmark["earth_to_sun_mass_ratio"]),
            a=float(model["a_AU"]),
            e=float(model["e"]),
            inc=math.radians(float(model["i_deg"])),
            Omega=math.radians(float(angle["Omega_deg"])),
            omega=math.radians(omega_deg),
            M=math.radians(float(angle["M_deg"])),
            hash="P9_ENGINEERING_SURROGATE",
        )
    simulation.N_active = simulation.N
    simulation.testparticle_type = 0
    simulation.integrator = contract["dynamics"]["integrator"]
    simulation.dt = dt_years
    simulation.ri_mercurius.hillfac = float(contract["dynamics"]["mercurius_hillfac"])
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
        )
    return simulation, tracer_start, common_names


def common_initial_digest(
    simulation: Any, tracer_start: int, common_names: list[str], tracer_count: int
) -> str:
    digest = hashlib.sha256()
    sun = simulation.particles[0]
    for name in common_names:
        digest.update(name.encode("ascii"))
        digest.update(particle_state_bytes(simulation.particles[name], sun))
    for offset in range(tracer_count):
        digest.update(struct.pack("!I", offset))
        digest.update(particle_state_bytes(simulation.particles[tracer_start + offset], sun))
    return digest.hexdigest()


def blank_tracker(tracer_count: int) -> dict[str, Any]:
    return {
        "ever_q_lt_30": [False] * tracer_count,
        "ever_i_gt_40": [False] * tracer_count,
        "ever_i_gt_60": [False] * tracer_count,
        "minimum_neptune_distance_AU": [None] * tracer_count,
        "minimum_p9_distance_AU": [None] * tracer_count,
        "final_bound": [False] * tracer_count,
        "final_q_AU": [None] * tracer_count,
        "final_i_deg": [None] * tracer_count,
        "all_sampled_states_finite": True,
        "sample_count": 0,
    }


def update_tracker(simulation: Any, tracer_start: int, tracker: dict[str, Any]) -> None:
    sun = simulation.particles[0]
    neptune = simulation.particles["Neptune"]
    p9 = None
    try:
        p9 = simulation.particles["P9_ENGINEERING_SURROGATE"]
    except Exception:
        pass
    for offset in range(len(tracker["final_bound"])):
        particle = simulation.particles[tracer_start + offset]
        numbers = (particle.x, particle.y, particle.z, particle.vx, particle.vy, particle.vz)
        if not all(math.isfinite(value) for value in numbers):
            tracker["all_sampled_states_finite"] = False
            continue
        try:
            orbit = particle.orbit(primary=sun)
            q = orbit.a * (1.0 - orbit.e)
            inclination = math.degrees(orbit.inc)
            bound = orbit.a > 0.0 and orbit.e < 1.0 and math.isfinite(q) and math.isfinite(inclination)
        except (ValueError, ZeroDivisionError, OverflowError):
            q, inclination, bound = math.nan, math.nan, False
        tracker["final_bound"][offset] = bound
        tracker["final_q_AU"][offset] = q if bound else None
        tracker["final_i_deg"][offset] = inclination if bound else None
        if bound:
            tracker["ever_q_lt_30"][offset] |= q < 30.0
            tracker["ever_i_gt_40"][offset] |= inclination > 40.0
            tracker["ever_i_gt_60"][offset] |= inclination > 60.0
        for label, body in (("neptune", neptune), ("p9", p9)):
            if body is None:
                continue
            distance = math.sqrt(
                (particle.x - body.x) ** 2
                + (particle.y - body.y) ** 2
                + (particle.z - body.z) ** 2
            )
            key = f"minimum_{label}_distance_AU"
            old = tracker[key][offset]
            if math.isfinite(distance) and (old is None or distance < old):
                tracker[key][offset] = distance
    tracker["sample_count"] += 1


def summarize_tracker(tracker: dict[str, Any]) -> dict[str, Any]:
    count = len(tracker["final_bound"])
    bound_count = sum(tracker["final_bound"])
    q_values = [value for value in tracker["final_q_AU"] if value is not None]
    i_values = [value for value in tracker["final_i_deg"] if value is not None]
    neptune = [value for value in tracker["minimum_neptune_distance_AU"] if value is not None]
    p9 = [value for value in tracker["minimum_p9_distance_AU"] if value is not None]
    return {
        "tracer_count": count,
        "sample_count": tracker["sample_count"],
        "all_sampled_states_finite": tracker["all_sampled_states_finite"],
        "bound_final": bound_count,
        "bound_fraction_final": bound_count / count,
        "ever_q_lt_30_fraction_sampled": sum(tracker["ever_q_lt_30"]) / count,
        "ever_i_gt_40_fraction_sampled": sum(tracker["ever_i_gt_40"]) / count,
        "ever_i_gt_60_fraction_sampled": sum(tracker["ever_i_gt_60"]) / count,
        "median_final_q_AU": statistics.median(q_values) if q_values else None,
        "median_final_i_deg": statistics.median(i_values) if i_values else None,
        "minimum_sampled_neptune_distance_AU": min(neptune) if neptune else None,
        "minimum_sampled_p9_distance_AU": min(p9) if p9 else None,
    }


def run_arm(
    contract: dict[str, Any],
    output_dir: Path,
    label: str,
    block: int,
    tracers: list[dict[str, Any]],
    model: dict[str, Any] | None,
    angle: dict[str, Any] | None,
    dt_years: float,
) -> dict[str, Any]:
    import rebound

    started = time.perf_counter()
    simulation, tracer_start, common_names = build_simulation(contract, tracers, model, angle, dt_years)
    initial_common = common_initial_digest(simulation, tracer_start, common_names, len(tracers))
    initial_energy = float(simulation.energy())
    initial_angular = angular_momentum_tuple(simulation)
    maximum_energy_drift = 0.0
    maximum_angular_drift = 0.0
    tracker = blank_tracker(len(tracers))
    update_tracker(simulation, tracer_start, tracker)

    duration = float(contract["dynamics"]["duration_years"])
    sample = float(contract["dynamics"]["sample_cadence_years"])
    checkpoint_time = float(contract["dynamics"]["checkpoint_years"])
    times = [sample * index for index in range(1, int(round(duration / sample)) + 1)]
    before_times = [value for value in times if value <= checkpoint_time]
    after_times = [value for value in times if value > checkpoint_time]
    for target in before_times:
        simulation.integrate(target, exact_finish_time=1)
        update_tracker(simulation, tracer_start, tracker)
        maximum_energy_drift = max(maximum_energy_drift, relative_drift(float(simulation.energy()), initial_energy))
        maximum_angular_drift = max(
            maximum_angular_drift,
            vector_relative_drift(angular_momentum_tuple(simulation), initial_angular),
        )

    checkpoint_path = output_dir / "checkpoints" / f"{label}.bin"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.unlink(missing_ok=True)
    state_before = simulation_digest(simulation)
    simulation.save_to_file(str(checkpoint_path), delete_file=True)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    resumed = rebound.Simulation(str(checkpoint_path))
    serialization_exact = simulation_digest(resumed) == state_before
    direct = simulation.copy()
    for target in after_times:
        direct.integrate(target, exact_finish_time=1)
        resumed.integrate(target, exact_finish_time=1)
        update_tracker(resumed, tracer_start, tracker)
        maximum_energy_drift = max(maximum_energy_drift, relative_drift(float(resumed.energy()), initial_energy))
        maximum_angular_drift = max(
            maximum_angular_drift,
            vector_relative_drift(angular_momentum_tuple(resumed), initial_angular),
        )
    direct_digest = simulation_digest(direct)
    resumed_digest = simulation_digest(resumed)
    continuation_exact = direct_digest == resumed_digest
    all_endpoint_finite = all(
        math.isfinite(float(getattr(particle, field)))
        for particle in resumed.particles
        for field in ("m", "x", "y", "z", "vx", "vy", "vz")
    )
    return {
        "label": label,
        "block": block,
        "model_id": None if model is None else model["id"],
        "angle_id": None if angle is None else angle["id"],
        "dt_years": dt_years,
        "initial_common_state_sha256": initial_common,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_serialization_state_exact": serialization_exact,
        "checkpoint_continuation_state_exact": continuation_exact,
        "endpoint_state_sha256": resumed_digest,
        "all_endpoint_states_finite": all_endpoint_finite,
        "maximum_relative_active_energy_drift": maximum_energy_drift,
        "maximum_relative_active_angular_momentum_vector_drift": maximum_angular_drift,
        "diagnostics": summarize_tracker(tracker),
        "elapsed_seconds": time.perf_counter() - started,
        "_tracker": tracker,
    }


def paired_diagnostics(source: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    left, right = source["_tracker"], control["_tracker"]
    q_differences: list[float] = []
    i_differences: list[float] = []
    for index, (source_bound, control_bound) in enumerate(zip(left["final_bound"], right["final_bound"])):
        if source_bound and control_bound:
            q_differences.append(abs(left["final_q_AU"][index] - right["final_q_AU"][index]))
            i_differences.append(abs(left["final_i_deg"][index] - right["final_i_deg"][index]))
    count = len(left["final_bound"])
    return {
        "paired_bound_count": len(q_differences),
        "median_absolute_final_q_change_AU": statistics.median(q_differences) if q_differences else None,
        "p90_absolute_final_q_change_AU": percentile(q_differences, 0.9),
        "median_absolute_final_i_change_deg": statistics.median(i_differences) if i_differences else None,
        "p90_absolute_final_i_change_deg": percentile(i_differences, 0.9),
        "source_minus_control_bound_fraction": (
            sum(left["final_bound"]) - sum(right["final_bound"])
        ) / count,
        "source_minus_control_ever_q_lt_30_fraction_sampled": (
            sum(left["ever_q_lt_30"]) - sum(right["ever_q_lt_30"])
        ) / count,
        "source_minus_control_ever_i_gt_40_fraction_sampled": (
            sum(left["ever_i_gt_40"]) - sum(right["ever_i_gt_40"])
        ) / count,
        "source_minus_control_ever_i_gt_60_fraction_sampled": (
            sum(left["ever_i_gt_60"]) - sum(right["ever_i_gt_60"])
        ) / count,
    }


def public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def compare_timestep(audit: dict[str, Any], primary: dict[str, Any]) -> dict[str, Any]:
    left, right = audit["_tracker"], primary["_tracker"]
    q: list[float] = []
    i: list[float] = []
    bound_agree = True
    event_agree = True
    for index in range(len(left["final_bound"])):
        bound_agree &= left["final_bound"][index] == right["final_bound"][index]
        event_agree &= all(
            left[key][index] == right[key][index]
            for key in ("ever_q_lt_30", "ever_i_gt_40", "ever_i_gt_60")
        )
        if left["final_bound"][index] and right["final_bound"][index]:
            q.append(abs(left["final_q_AU"][index] - right["final_q_AU"][index]))
            i.append(abs(left["final_i_deg"][index] - right["final_i_deg"][index]))
    return {
        "label": audit["label"],
        "bound_identity_exact": bound_agree,
        "sampled_threshold_identities_exact": event_agree,
        "maximum_absolute_final_q_difference_AU": max(q, default=0.0),
        "maximum_absolute_final_i_difference_deg": max(i, default=0.0),
        "checkpoint_serialization_state_exact": audit["checkpoint_serialization_state_exact"],
        "checkpoint_continuation_state_exact": audit["checkpoint_continuation_state_exact"],
        "all_endpoint_states_finite": audit["all_endpoint_states_finite"],
        "maximum_relative_active_energy_drift": audit["maximum_relative_active_energy_drift"],
        "maximum_relative_active_angular_momentum_vector_drift": audit[
            "maximum_relative_active_angular_momentum_vector_drift"
        ],
    }


def validate_contract(contract: dict[str, Any], contract_path: Path) -> None:
    if contract.get("schema") != "jx-e1-synthetic-engineering-pilot-contract/v1":
        raise ValueError("unexpected contract schema")
    if contract.get("experiment_id") != "jx-e1-p9-9x4-smoke-v1":
        raise ValueError("unexpected experiment ID")
    permissions = contract["permissions"]
    if permissions != {
        "local_engineering_execution_authorized": True,
        "jx_o2_execution_authorized": False,
        "observed_data_access_authorized": False,
        "scientific_claim_authorized": False,
        "gpu_execution_authorized": False,
    }:
        raise ValueError("permission boundary changed")
    if contract.get("claim_ceiling") != "ENGINEERING_SURROGATE_ONLY":
        raise ValueError("claim ceiling changed")
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
        raise ValueError("the nine-row physical grid changed")
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
        raise ValueError("the four fixed angular completions changed")
    dynamics = contract["dynamics"]
    if dynamics["integrator"] != "mercurius" or dynamics["duration_years"] != 100.0:
        raise ValueError("the smoke-test dynamics changed")
    if dynamics["duration_years"] / dynamics["sample_cadence_years"] % 1 != 0:
        raise ValueError("duration must be an integer multiple of sample cadence")
    if dynamics["checkpoint_years"] / dynamics["sample_cadence_years"] % 1 != 0:
        raise ValueError("checkpoint must align with sample cadence")
    if contract["tracer_design"]["blocks"] != 2 or contract["tracer_design"]["tracers_per_block"] != 32:
        raise ValueError("the smoke-test tracer cardinality changed")
    if sha256_file(Path(__file__).resolve()) != contract["runtime"]["runner_sha256"]:
        raise ValueError("runner hash does not match contract")
    if contract_path.name != "contract_v1.json":
        raise ValueError("expected contract_v1.json")


def validate_runtime(contract: dict[str, Any]) -> dict[str, Any]:
    import rebound

    binary = Path(rebound.clibrebound._name).resolve()
    actual = {
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "rebound_version": rebound.__version__,
        "rebound_build": rebound.__build__,
        "rebound_binary_sha256": sha256_file(binary),
    }
    expected = {
        key: contract["runtime"][key]
        for key in ("python_version", "rebound_version", "rebound_build", "rebound_binary_sha256")
    }
    if actual != expected:
        raise ValueError(f"runtime mismatch: {actual}")
    return {**actual, "rebound_binary_path": str(binary)}


def execute(contract_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    contract = strict_json(contract_path)
    validate_contract(contract, contract_path)
    runtime = validate_runtime(contract)
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    models = contract["model_grid"]
    angles = contract["angle_grid"]
    blocks = int(contract["tracer_design"]["blocks"])
    primary_dt = float(contract["dynamics"]["dt_years"])
    controls: dict[int, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for block in range(blocks):
        tracers = make_tracers(contract, block)
        control = run_arm(contract, output_dir, f"M0-b{block:02d}", block, tracers, None, None, primary_dt)
        controls[block] = control
        records.append(control)
    for model in models:
        for angle in angles:
            for block in range(blocks):
                tracers = make_tracers(contract, block)
                label = f"{model['id']}-{angle['id']}-b{block:02d}"
                record = run_arm(contract, output_dir, label, block, tracers, model, angle, primary_dt)
                record["common_initial_state_matches_M0"] = (
                    record["initial_common_state_sha256"] == controls[block]["initial_common_state_sha256"]
                )
                record["paired_M1_minus_M0_diagnostics"] = paired_diagnostics(record, controls[block])
                records.append(record)

    audit_count = int(contract["timestep_audit"]["tracers"])
    audit_dt = float(contract["timestep_audit"]["dt_years"])
    block = int(contract["timestep_audit"]["block"])
    audit_tracers = make_tracers(contract, block, audit_count)
    selected: list[tuple[dict[str, Any] | None, dict[str, Any] | None, str]] = [(None, None, "M0")]
    for model_id in contract["timestep_audit"]["model_ids"]:
        model = next(item for item in models if item["id"] == model_id)
        angle = next(item for item in angles if item["id"] == contract["timestep_audit"]["angle_id"])
        selected.append((model, angle, f"{model_id}-{angle['id']}"))
    audits = []
    for model, angle, stem in selected:
        audit = run_arm(
            contract,
            output_dir,
            f"dt-half-{stem}-b{block:02d}",
            block,
            audit_tracers,
            model,
            angle,
            audit_dt,
        )
        if model is None:
            primary = controls[block]
        else:
            primary = next(
                item
                for item in records
                if item["model_id"] == model["id"] and item["angle_id"] == angle["id"] and item["block"] == block
            )
        truncated_tracker = {
            key: value[:audit_count] if isinstance(value, list) else value
            for key, value in primary["_tracker"].items()
        }
        primary_subset = {**primary, "_tracker": truncated_tracker}
        audits.append(compare_timestep(audit, primary_subset))

    thresholds = contract["gates"]
    all_records = records
    checks = {
        "complete_expected_run_matrix": len(records) == blocks + len(models) * len(angles) * blocks,
        "every_M1_common_initial_state_matches_M0": all(
            item.get("common_initial_state_matches_M0", True) for item in records
        ),
        "every_checkpoint_serialization_state_exact": all(
            item["checkpoint_serialization_state_exact"] for item in all_records
        ),
        "every_checkpoint_continuation_state_exact": all(
            item["checkpoint_continuation_state_exact"] for item in all_records
        ),
        "all_sampled_and_endpoint_states_finite": all(
            item["diagnostics"]["all_sampled_states_finite"] and item["all_endpoint_states_finite"]
            for item in all_records
        ),
        "active_energy_drift_within_gate": max(
            item["maximum_relative_active_energy_drift"] for item in all_records
        ) <= float(thresholds["max_relative_active_energy_drift"]),
        "active_angular_momentum_drift_within_gate": max(
            item["maximum_relative_active_angular_momentum_vector_drift"] for item in all_records
        ) <= float(thresholds["max_relative_active_angular_momentum_vector_drift"]),
        "timestep_audit_within_gate": all(
            item["bound_identity_exact"]
            and item["sampled_threshold_identities_exact"]
            and item["checkpoint_serialization_state_exact"]
            and item["checkpoint_continuation_state_exact"]
            and item["all_endpoint_states_finite"]
            and item["maximum_absolute_final_q_difference_AU"]
            <= float(thresholds["max_dt_half_final_q_difference_AU"])
            and item["maximum_absolute_final_i_difference_deg"]
            <= float(thresholds["max_dt_half_final_i_difference_deg"])
            for item in audits
        ),
    }
    public_records = [public_record(item) for item in records]
    deterministic_payload = {
        "contract_sha256": sha256_file(contract_path),
        "runtime": {key: runtime[key] for key in runtime if key != "rebound_binary_path"},
        "records": [
            {key: value for key, value in public_record(item).items() if key != "elapsed_seconds"}
            for item in records
        ],
        "timestep_audit": audits,
        "checks": checks,
    }
    result = {
        "schema": "jx-e1-synthetic-engineering-pilot-result/v1",
        "experiment_id": contract["experiment_id"],
        "verdict": "ENGINEERING_SMOKE_VALID" if all(checks.values()) else "ENGINEERING_SMOKE_INVALID",
        "claim_ceiling": contract["claim_ceiling"],
        "nonclaim": contract["mandatory_nonclaim"],
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "runtime": runtime,
        "design_counts": {
            "published_physical_rows": len(models),
            "fixed_angular_completions": len(angles),
            "synthetic_blocks": blocks,
            "tracers_per_block": contract["tracer_design"]["tracers_per_block"],
            "M0_runs": blocks,
            "M1_runs": len(models) * len(angles) * blocks,
            "timestep_audit_runs": len(audits),
        },
        "checks": checks,
        "maximum_observed_drifts": {
            "relative_active_energy": max(item["maximum_relative_active_energy_drift"] for item in records),
            "relative_active_angular_momentum_vector": max(
                item["maximum_relative_active_angular_momentum_vector_drift"] for item in records
            ),
        },
        "timestep_audit": audits,
        "run_records": public_records,
        "deterministic_payload_sha256": sha256_bytes(canonical_bytes(deterministic_payload)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    result_path = output_dir / "result_v1.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args()
    contract_path = arguments.contract.resolve()
    contract = strict_json(contract_path)
    validate_contract(contract, contract_path)
    runtime = validate_runtime(contract)
    if arguments.validate_only:
        print(json.dumps({"contract_sha256": sha256_file(contract_path), "runtime": runtime}, indent=2))
        return 0
    result = execute(contract_path, arguments.output_dir.resolve())
    print(json.dumps({
        "verdict": result["verdict"],
        "deterministic_payload_sha256": result["deterministic_payload_sha256"],
        "elapsed_seconds": result["elapsed_seconds"],
        "output": str(arguments.output_dir.resolve() / "result_v1.json"),
    }, indent=2))
    return 0 if result["verdict"] == "ENGINEERING_SMOKE_VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
