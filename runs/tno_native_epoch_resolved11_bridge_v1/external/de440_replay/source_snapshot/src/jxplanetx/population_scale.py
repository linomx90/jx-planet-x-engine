"""Measured scale gate for large, paired massless-tracer populations.

This module closes a narrow gap between the ensemble contract layer and an
external production backend.  It deliberately implements a *scale gate*, not
an astronomical population inference: a locked set of seed orbit shapes is
expanded only in angular phase, then propagated in matched source and control
arms.  Tracers are addressed by particle index because REBOUND string hashes
are 32-bit and collide at 100,000 members.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import resource
import statistics
import tempfile
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping

from .ensemble_validation import wasserstein_1d
from .provenance import runtime_source_manifest


CONTRACT_SCHEMA = "jx-population-scale-contract/v1"
RESULT_SCHEMA = "jx-population-scale-result/v1"
G_AU3_MSUN_YR2 = 4.0 * math.pi * math.pi
CORE_NAMES = ("Sun", "Jupiter", "Saturn", "Uranus", "Neptune")
STATE_COLUMNS = ("index", "name", "mass", "x", "y", "z", "vx", "vy", "vz")
VECTOR_COLUMNS = ("x", "y", "z", "vx", "vy", "vz")


@dataclass(frozen=True)
class OrbitShape:
    template_name: str
    a_AU: float
    e: float
    i_rad: float
    q_AU: float


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite locked result: {path}")
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


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != STATE_COLUMNS:
            raise ValueError(f"{path} must have columns {STATE_COLUMNS}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no bodies")
    names = [row["name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"{path} contains duplicate body names")
    for expected, row in enumerate(rows):
        if int(row["index"]) != expected:
            raise ValueError(f"{path} indices must be contiguous and ordered")
        try:
            numbers = [Decimal(row[key]) for key in ("mass", *VECTOR_COLUMNS)]
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"{path} contains an invalid numeric field") from exc
        if any(not number.is_finite() for number in numbers):
            raise ValueError(f"{path} contains a nonfinite numeric field")
    return rows


def _by_name(rows: Iterable[Mapping[str, str]]) -> dict[str, Mapping[str, str]]:
    return {row["name"]: row for row in rows}


def _fraction(value: str) -> Fraction:
    return Fraction(Decimal(value))


def _relative_state(row: Mapping[str, str], sun: Mapping[str, str]) -> tuple[Fraction, ...]:
    return tuple(_fraction(row[key]) - _fraction(sun[key]) for key in VECTOR_COLUMNS)


def _canonical_fraction(value: Fraction) -> str:
    if value == 0:
        return "0"
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _verify_matched_states(
    source_rows: list[dict[str, str]],
    control_rows: list[dict[str, str]],
    position_tolerance: Fraction = Fraction(0),
    velocity_tolerance: Fraction = Fraction(0),
    mass_tolerance: Fraction = Fraction(0),
) -> dict[str, Any]:
    source = _by_name(source_rows)
    control = _by_name(control_rows)
    if source_rows[0]["name"] != "Sun" or control_rows[0]["name"] != "Sun":
        raise ValueError("Sun must be particle index 0 in both matched states")
    for name in CORE_NAMES:
        if name not in source or name not in control:
            raise ValueError(f"matched states require {name}")
    for label, rows in (("source", source_rows), ("control", control_rows)):
        negative = [row["name"] for row in rows if Decimal(row["mass"]) < 0]
        if negative:
            raise ValueError(f"{label} state contains negative masses: {negative}")
    control_massive = {row["name"] for row in control_rows if Decimal(row["mass"]) > 0}
    if control_massive != set(CORE_NAMES):
        raise ValueError("control massive bodies must be exactly the Sun and four giant planets")
    common = sorted(set(source) & set(control))
    if set(control) != set(common):
        raise ValueError("control state may not contain bodies absent from source state")
    source_only = [row for row in source_rows if row["name"] not in control]
    massive_source_only = [row for row in source_only if Decimal(row["mass"]) > 0]
    if len(source_only) != 1 or len(massive_source_only) != 1:
        raise ValueError("source state must add exactly one massive source body")
    source_sun, control_sun = source["Sun"], control["Sun"]
    mismatches: list[str] = []
    digest_rows = []
    maximum_position_residual = Fraction(0)
    maximum_velocity_residual = Fraction(0)
    maximum_mass_residual = Fraction(0)
    for name in common:
        left, right = source[name], control[name]
        left_mass, right_mass = _fraction(left["mass"]), _fraction(right["mass"])
        mass_residual = abs(left_mass - right_mass)
        maximum_mass_residual = max(maximum_mass_residual, mass_residual)
        if mass_residual > mass_tolerance:
            mismatches.append(f"{name}:mass")
        left_relative = _relative_state(left, source_sun)
        right_relative = _relative_state(right, control_sun)
        residuals = [abs(left_relative[index] - right_relative[index]) for index in range(6)]
        maximum_position_residual = max(maximum_position_residual, *residuals[:3])
        maximum_velocity_residual = max(maximum_velocity_residual, *residuals[3:])
        if any(value > position_tolerance for value in residuals[:3]) or any(
            value > velocity_tolerance for value in residuals[3:]
        ):
            mismatches.append(f"{name}:heliocentric_state")
        digest_rows.append(
            {
                "name": name,
                "mass": _canonical_fraction(right_mass),
                "heliocentric_state": [_canonical_fraction(value) for value in right_relative],
            }
        )
    if mismatches:
        raise ValueError(f"source/control relative-state mismatch: {mismatches[:8]}")
    tracers = [row for row in control_rows if Decimal(row["mass"]) == 0]
    if not tracers or any(not row["name"].startswith("t") for row in tracers):
        raise ValueError("control state must contain named massless tracer templates")
    return {
        "source_name": massive_source_only[0]["name"],
        "common_body_count": len(common),
        "template_count": len(tracers),
        "control_relative_state_sha256": _canonical_sha256(digest_rows),
        "maximum_archive_relative_position_residual_AU": float(maximum_position_residual),
        "maximum_archive_relative_velocity_residual_AU_per_year": float(maximum_velocity_residual),
        "maximum_archive_mass_residual_Msun": float(maximum_mass_residual),
        "archive_tolerances": {
            "position_AU": float(position_tolerance),
            "velocity_AU_per_year": float(velocity_tolerance),
            "mass_Msun": float(mass_tolerance),
        },
    }


def _binary64_massive_state(rows: list[dict[str, str]]) -> dict[str, tuple[float, ...]]:
    sun = _by_name(rows)["Sun"]
    result = {}
    for row in rows:
        if Decimal(row["mass"]) <= 0:
            continue
        relative = _relative_state(row, sun)
        result[row["name"]] = (float(_fraction(row["mass"])), *(float(value) for value in relative))
    return result


def _cross(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _orbit_shape(row: Mapping[str, str], sun: Mapping[str, str]) -> OrbitShape:
    relative = _relative_state(row, sun)
    position = tuple(float(value) for value in relative[:3])
    velocity = tuple(float(value) for value in relative[3:])
    radius = _norm(position)
    speed2 = sum(value * value for value in velocity)
    mu = G_AU3_MSUN_YR2 * float(_fraction(sun["mass"]) + _fraction(row["mass"]))
    energy = speed2 / 2.0 - mu / radius
    if not math.isfinite(energy) or energy >= 0.0:
        raise ValueError(f"template {row['name']} is not on a bound two-body orbit")
    a = -mu / (2.0 * energy)
    angular_momentum = _cross(position, velocity)
    hnorm = _norm(angular_momentum)
    vxh = _cross(velocity, angular_momentum)
    eccentricity_vector = tuple(vxh[index] / mu - position[index] / radius for index in range(3))
    eccentricity = _norm(eccentricity_vector)
    inclination = math.acos(max(-1.0, min(1.0, angular_momentum[2] / hnorm)))
    q = a * (1.0 - eccentricity)
    if not all(math.isfinite(value) for value in (a, eccentricity, inclination, q)) or not (a > 0 and 0 <= eccentricity < 1 and q > 0):
        raise ValueError(f"template {row['name']} has invalid orbital elements")
    return OrbitShape(row["name"], a, eccentricity, inclination, q)


def _open_uniform(seed: str, tracer_index: int, stream: str) -> float:
    message = f"jx-phase-expand/v1\x1f{seed}\x1f{tracer_index}\x1f{stream}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(message).digest()[:8], "big") >> 11
    return (integer + 0.5) / float(1 << 53)


def _phase_angles(seed: str, tracer_index: int) -> tuple[float, float, float]:
    turn = 2.0 * math.pi
    return tuple(turn * _open_uniform(seed, tracer_index, stream) for stream in ("Omega", "omega", "M"))  # type: ignore[return-value]


def _host_memory_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _cgroup_value(paths: tuple[str, ...]) -> int:
    for raw_path in paths:
        try:
            raw = Path(raw_path).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if raw == "max":
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value > 0:
            return value
    return 0


def _effective_memory_limit_bytes() -> int:
    host = _host_memory_bytes()
    cgroup = _cgroup_value(
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes")
    )
    values = [value for value in (host, cgroup) if value > 0]
    return min(values) if values else 0


def _cgroup_memory_current_bytes() -> int:
    return _cgroup_value(
        ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes")
    )


def _peak_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _build_simulation(
    rows: list[dict[str, str]],
    templates: list[OrbitShape],
    tracer_count: int,
    seed: str,
    integrator: str,
    dt_years: float,
):
    import rebound

    simulation = rebound.Simulation()
    simulation.G = G_AU3_MSUN_YR2
    massive = [row for row in rows if Decimal(row["mass"]) > 0]
    sun_row = _by_name(rows)["Sun"]
    for row in massive:
        relative = _relative_state(row, sun_row)
        simulation.add(
            m=float(row["mass"]),
            x=float(relative[0]), y=float(relative[1]), z=float(relative[2]),
            vx=float(relative[3]), vy=float(relative[4]), vz=float(relative[5]),
            hash=row["name"],
        )
    simulation.N_active = len(massive)
    simulation.testparticle_type = 0
    simulation.integrator = integrator
    simulation.dt = dt_years
    if integrator == "mercurius":
        simulation.ri_mercurius.hillfac = 3.0
    sun = simulation.particles[0]
    for tracer_index in range(tracer_count):
        template = templates[tracer_index % len(templates)]
        node, periapse, mean_anomaly = _phase_angles(seed, tracer_index)
        # Intentionally omit a hash: 100,000 human-readable IDs collide in
        # REBOUND's 32-bit hash space.  Particle order plus side metadata is
        # the stable identity contract.
        simulation.add(
            primary=sun,
            m=0.0,
            a=template.a_AU,
            e=template.e,
            inc=template.i_rad,
            Omega=node,
            omega=periapse,
            M=mean_anomaly,
        )
    return simulation, len(massive)


def _population_metrics(
    simulation,
    tracer_offset: int,
    templates: list[OrbitShape],
    tracer_count: int,
    q_threshold: float,
    q_hysteresis: float,
    replicate_size: int,
) -> tuple[dict[str, Any], list[float], list[float], list[dict[str, Any]]]:
    sun = simulation.particles[0]
    q_values: list[float] = []
    i_values: list[float] = []
    low_q = boundary_q = injections = initially_high_eligible = 0
    lower_q = q_threshold - q_hysteresis
    upper_q = q_threshold + q_hysteresis
    replicate_accumulators: list[dict[str, Any]] = []
    replicate_count = math.ceil(tracer_count / replicate_size)
    for replicate in range(replicate_count):
        replicate_accumulators.append(
            {
                "replicate_index": replicate,
                "planned": 0,
                "bound": 0,
                "low_q": 0,
                "boundary_q": 0,
                "initially_high_q_eligible": 0,
                "injections": 0,
                "q": [],
            }
        )
    for tracer_index in range(tracer_count):
        accumulator = replicate_accumulators[tracer_index // replicate_size]
        accumulator["planned"] += 1
        template_q = templates[tracer_index % len(templates)].q_AU
        initially_high = template_q > upper_q
        if initially_high:
            initially_high_eligible += 1
            accumulator["initially_high_q_eligible"] += 1
        particle = simulation.particles[tracer_offset + tracer_index]
        orbit = particle.orbit(primary=sun)
        bound = orbit.a > 0.0 and orbit.e < 1.0 and math.isfinite(orbit.a) and math.isfinite(orbit.e)
        if not bound:
            continue
        q = orbit.a * (1.0 - orbit.e)
        inclination = math.degrees(orbit.inc)
        if not math.isfinite(q) or not math.isfinite(inclination):
            continue
        q_values.append(q)
        i_values.append(inclination)
        accumulator["bound"] += 1
        accumulator["q"].append(q)
        if q < lower_q:
            low_q += 1
            accumulator["low_q"] += 1
            if initially_high:
                injections += 1
                accumulator["injections"] += 1
        elif q <= upper_q:
            boundary_q += 1
            accumulator["boundary_q"] += 1
    replicate_rows: list[dict[str, Any]] = []
    for accumulator in replicate_accumulators:
        planned = accumulator["planned"]
        bound = accumulator["bound"]
        values = accumulator.pop("q")
        replicate_rows.append(
            {
                **accumulator,
                "survival_fraction": bound / planned,
                "classified_low_q_fraction": accumulator["low_q"] / planned,
                "q_boundary_fraction": accumulator["boundary_q"] / planned,
                "endpoint_injection_fraction": accumulator["injections"] / planned,
                "endpoint_injection_fraction_of_eligible": (
                    accumulator["injections"] / accumulator["initially_high_q_eligible"]
                    if accumulator["initially_high_q_eligible"]
                    else None
                ),
                "mean_bound_q_AU": statistics.fmean(values) if values else None,
            }
        )
    metrics = {
        "planned_tracers": tracer_count,
        "bound_final": len(q_values),
        "survival_fraction": len(q_values) / tracer_count,
        "q_below_lower_band_final": low_q,
        "classified_low_q_fraction": low_q / tracer_count,
        "q_boundary_final": boundary_q,
        "q_boundary_fraction": boundary_q / tracer_count,
        "q_classification_lower_AU": lower_q,
        "q_classification_upper_AU": upper_q,
        "initially_high_q_eligible": initially_high_eligible,
        "endpoint_injections": injections,
        "endpoint_injection_fraction": injections / tracer_count,
        "endpoint_injection_fraction_of_eligible": (
            injections / initially_high_eligible if initially_high_eligible else None
        ),
        "mean_bound_q_AU": statistics.fmean(q_values) if q_values else None,
        "inclination_width_deg": statistics.pstdev(i_values) if i_values else None,
    }
    return metrics, q_values, i_values, replicate_rows


def _resolve(contract_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (contract_path.parent / path).resolve()


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _max_active_state_difference(full, audit, active_count: int) -> float:
    maximum = 0.0
    for index in range(active_count):
        left, right = full.particles[index], audit.particles[index]
        for attribute in ("x", "y", "z", "vx", "vy", "vz", "m"):
            maximum = max(maximum, abs(getattr(left, attribute) - getattr(right, attribute)))
    return maximum


def run_population_scale_gate(contract_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    contract_file = Path(contract_path).resolve()
    if Path(output_path).exists():
        raise FileExistsError(f"refusing to overwrite locked result: {output_path}")
    contract = json.loads(contract_file.read_text(encoding="utf-8"))
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError(f"contract schema must be {CONTRACT_SCHEMA}")
    software_manifest = runtime_source_manifest()
    if contract.get("runner_source_tree_sha256") != software_manifest["tree_sha256"]:
        raise ValueError("active JX source tree does not match the prelocked contract")
    source_path = _resolve(contract_file, contract["source_state_csv"])
    control_path = _resolve(contract_file, contract["control_state_csv"])
    expected_hashes = contract["state_sha256"]
    actual_hashes = {"source": _sha256_file(source_path), "control": _sha256_file(control_path)}
    if actual_hashes != expected_hashes:
        raise ValueError(f"state SHA-256 mismatch: {actual_hashes}")
    design = contract["population_design"]
    if design.get("phase_generator") != "sha256-counter-open-uniform/v1":
        raise ValueError("contract phase_generator is not supported by this runner")
    blocks = _require_positive_int(design["seed_blocks"], "seed_blocks")
    replicates_per_block = _require_positive_int(design["replicates_per_block"], "replicates_per_block")
    tracers_per_replicate = _require_positive_int(design["tracers_per_replicate"], "tracers_per_replicate")
    tracer_count = blocks * replicates_per_block * tracers_per_replicate
    dynamics = contract["dynamics"]
    if dynamics.get("testparticle_type") != 0:
        raise ValueError("contract must lock REBOUND testparticle_type=0")
    gate_years = float(dynamics["gate_years"])
    target_years = float(dynamics["target_years"])
    dt_years = float(dynamics["dt_years"])
    energy_check_interval = float(dynamics.get("energy_check_interval_years", gate_years))
    if not all(math.isfinite(value) and value > 0 for value in (gate_years, target_years, dt_years, energy_check_interval)):
        raise ValueError("gate_years, target_years, dt_years, and energy_check_interval_years must be finite and positive")
    if energy_check_interval > gate_years:
        raise ValueError("energy_check_interval_years may not exceed gate_years")
    step_ratio = gate_years / dt_years
    if not math.isclose(step_ratio, round(step_ratio), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("gate_years must be an integer multiple of dt_years")
    energy_stride = energy_check_interval / dt_years
    if not math.isclose(energy_stride, round(energy_stride), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("energy_check_interval_years must be an integer multiple of dt_years")
    integrator = dynamics["integrator"]
    if integrator not in ("mercurius", "whfast"):
        raise ValueError("scale gate supports mercurius or whfast")
    import rebound

    rebound_build = getattr(rebound, "__build__", "")
    rebound_binary = Path(rebound.clibrebound._name).resolve()
    wheel_path = _resolve(contract_file, dynamics["rebound_wheel_file"])
    rebound_integrity = {
        "version": rebound.__version__,
        "build": rebound_build,
        "binary_sha256": _sha256_file(rebound_binary),
        "wheel_sha256": _sha256_file(wheel_path),
    }
    expected_rebound = {
        "version": dynamics["rebound_version"],
        "build": dynamics["rebound_build"],
        "binary_sha256": dynamics["rebound_binary_sha256"],
        "wheel_sha256": dynamics["rebound_wheel_sha256"],
    }
    if rebound_integrity != expected_rebound:
        raise ValueError(f"REBOUND runtime does not match the prelocked contract: {rebound_integrity}")
    source_rows, control_rows = _load_rows(source_path), _load_rows(control_path)
    archive_tolerances = contract["archive_match_tolerances"]
    match = _verify_matched_states(
        source_rows,
        control_rows,
        _fraction(str(archive_tolerances["position_AU"])),
        _fraction(str(archive_tolerances["velocity_AU_per_year"])),
        _fraction(str(archive_tolerances["mass_Msun"])),
    )
    source_binary_state = _binary64_massive_state(source_rows)
    control_binary_state = _binary64_massive_state(control_rows)
    binary_differences = [
        abs(left - right)
        for name in CORE_NAMES
        for left, right in zip(source_binary_state[name], control_binary_state[name])
    ]
    maximum_binary_difference = max(binary_differences, default=0.0)
    if maximum_binary_difference != 0.0:
        raise ValueError("canonical heliocentric binary64 common massive states are not identical")
    match["canonical_binary64_common_massive_state_max_abs_difference"] = maximum_binary_difference
    match["canonical_binary64_common_massive_state_sha256"] = _canonical_sha256(
        {name: list(control_binary_state[name]) for name in CORE_NAMES}
    )
    control_sun = _by_name(control_rows)["Sun"]
    templates = [_orbit_shape(row, control_sun) for row in control_rows if Decimal(row["mass"]) == 0]
    q_hysteresis = float(contract.get("q_hysteresis_AU", 0.0))
    if not math.isfinite(q_hysteresis) or q_hysteresis < 0.0 or q_hysteresis >= float(contract["q_threshold_AU"]):
        raise ValueError("q_hysteresis_AU must be finite, nonnegative, and below q_threshold_AU")
    template_digest = _canonical_sha256(
        [
            {
                "name": item.template_name,
                "a_AU": format(item.a_AU, ".17g"),
                "e": format(item.e, ".17g"),
                "i_rad": format(item.i_rad, ".17g"),
                "q_AU": format(item.q_AU, ".17g"),
            }
            for item in templates
        ]
    )
    phase_digest = _canonical_sha256(
        [
            {
                "index": index,
                "template": templates[index % len(templates)].template_name,
                "angles": [format(value, ".17g") for value in _phase_angles(design["seed"], index)],
            }
            for index in range(tracer_count)
        ]
    )
    arms: dict[str, Any] = {}
    arm_values: dict[str, tuple[list[float], list[float], list[dict[str, Any]]]] = {}
    memory_limit = _effective_memory_limit_bytes()
    run_started = time.perf_counter()
    memory_current_samples = [_cgroup_memory_current_bytes()]

    for arm, rows in (("control", control_rows), ("source", source_rows)):
        build_started = time.perf_counter()
        simulation, tracer_offset = _build_simulation(
            rows,
            templates,
            tracer_count,
            design["seed"],
            integrator,
            dt_years,
        )
        build_seconds = time.perf_counter() - build_started
        if simulation.N != tracer_offset + tracer_count or simulation.N_active != tracer_offset:
            raise RuntimeError(f"{arm} particle count contract failed")
        integrate_started = time.perf_counter()
        simulation.integrate(gate_years, exact_finish_time=1)
        integrate_seconds = time.perf_counter() - integrate_started
        energy_audit, audit_active_count = _build_simulation(
            rows,
            templates,
            0,
            design["seed"],
            integrator,
            dt_years,
        )
        if audit_active_count != tracer_offset:
            raise RuntimeError(f"{arm} active-only audit body count mismatch")
        audit_initial_energy = energy_audit.energy()
        energy_samples = [{"time_year": 0.0, "relative_drift": 0.0}]
        energy_audit_started = time.perf_counter()
        nominal_steps = int(round(step_ratio))
        sample_stride = int(round(energy_stride))
        for step in range(1, nominal_steps + 1):
            target_time = step * dt_years
            energy_audit.integrate(target_time, exact_finish_time=1)
            if step % sample_stride == 0 or step == nominal_steps:
                drift = abs((energy_audit.energy() - audit_initial_energy) / audit_initial_energy)
                energy_samples.append({"time_year": target_time, "relative_drift": drift})
        energy_audit_seconds = time.perf_counter() - energy_audit_started
        active_twin_difference = _max_active_state_difference(simulation, energy_audit, tracer_offset)
        endpoint_energy_drift = energy_samples[-1]["relative_drift"]
        metrics_started = time.perf_counter()
        metrics, q_values, i_values, replicate_rows = _population_metrics(
            simulation,
            tracer_offset,
            templates,
            tracer_count,
            float(contract["q_threshold_AU"]),
            q_hysteresis,
            tracers_per_replicate,
        )
        metrics_seconds = time.perf_counter() - metrics_started
        maximum_energy_sample = max(energy_samples, key=lambda item: item["relative_drift"])
        arms[arm] = {
            "massive_bodies": tracer_offset,
            "particles": simulation.N,
            "N_active": simulation.N_active,
            "testparticle_type": simulation.testparticle_type,
            "build_seconds": build_seconds,
            "integrate_seconds": integrate_seconds,
            "energy_audit_seconds": energy_audit_seconds,
            "metrics_seconds": metrics_seconds,
            "tracer_steps": tracer_count * int(round(step_ratio)),
            "tracer_steps_per_second": tracer_count * step_ratio / integrate_seconds,
            "relative_massive_energy_drift_endpoint": endpoint_energy_drift,
            "maximum_sampled_relative_massive_energy_drift": maximum_energy_sample["relative_drift"],
            "maximum_energy_drift_sample_year": maximum_energy_sample["time_year"],
            "energy_drift_sampling_cadence_years": energy_check_interval,
            "energy_drift_samples": energy_samples,
            "active_only_twin_max_abs_state_difference": active_twin_difference,
            "collision_mode": simulation.collision,
            "population": metrics,
            "replicates": replicate_rows,
            "peak_process_rss_bytes_after_arm": _peak_rss_bytes(),
            "cgroup_memory_current_bytes_after_arm": _cgroup_memory_current_bytes(),
        }
        memory_current_samples.append(arms[arm]["cgroup_memory_current_bytes_after_arm"])
        arm_values[arm] = (q_values, i_values, replicate_rows)
        del simulation, energy_audit
        gc.collect()
    control_q, control_i, control_replicates = arm_values["control"]
    source_q, source_i, source_replicates = arm_values["source"]
    replicate_effects = []
    for control_row, source_row in zip(control_replicates, source_replicates):
        control_eligible_injection = control_row["endpoint_injection_fraction_of_eligible"]
        source_eligible_injection = source_row["endpoint_injection_fraction_of_eligible"]
        replicate_effects.append(
            {
                "replicate_index": control_row["replicate_index"],
                "source_minus_control_classified_low_q_fraction": source_row["classified_low_q_fraction"] - control_row["classified_low_q_fraction"],
                "source_minus_control_endpoint_injection_fraction": source_row["endpoint_injection_fraction"] - control_row["endpoint_injection_fraction"],
                "source_minus_control_endpoint_injection_fraction_of_eligible": (
                    source_eligible_injection - control_eligible_injection
                    if source_eligible_injection is not None and control_eligible_injection is not None
                    else None
                ),
                "source_minus_control_survival_fraction": source_row["survival_fraction"] - control_row["survival_fraction"],
            }
        )
    control_eligible_injection = arms["control"]["population"]["endpoint_injection_fraction_of_eligible"]
    source_eligible_injection = arms["source"]["population"]["endpoint_injection_fraction_of_eligible"]
    comparison = {
        "wasserstein_final_q_AU": wasserstein_1d(control_q, source_q) if control_q and source_q else None,
        "wasserstein_final_i_deg": wasserstein_1d(control_i, source_i) if control_i and source_i else None,
        "source_minus_control_classified_low_q_fraction": arms["source"]["population"]["classified_low_q_fraction"] - arms["control"]["population"]["classified_low_q_fraction"],
        "source_minus_control_endpoint_injection_fraction": arms["source"]["population"]["endpoint_injection_fraction"] - arms["control"]["population"]["endpoint_injection_fraction"],
        "source_minus_control_endpoint_injection_fraction_of_eligible": (
            source_eligible_injection - control_eligible_injection
            if source_eligible_injection is not None and control_eligible_injection is not None
            else None
        ),
        "source_minus_control_survival_fraction": arms["source"]["population"]["survival_fraction"] - arms["control"]["population"]["survival_fraction"],
        "replicate_effects": replicate_effects,
    }
    projected_seconds = sum(arm["integrate_seconds"] for arm in arms.values()) * target_years / gate_years
    peak_rss = _peak_rss_bytes()
    maximum_observed_memory = max([peak_rss, *memory_current_samples])
    gates = contract["operational_gates"]
    checks = {
        "exact_particle_counts": all(arm["particles"] == tracer_count + arm["massive_bodies"] for arm in arms.values()),
        "massless_no_backreaction_configuration": all(
            arm["N_active"] == arm["massive_bodies"]
            and arm["testparticle_type"] == 0
            and arm["active_only_twin_max_abs_state_difference"] == 0.0
            and arm["collision_mode"] == "none"
            for arm in arms.values()
        ),
        "massive_energy_drift": all(arm["maximum_sampled_relative_massive_energy_drift"] <= float(gates["max_relative_energy_drift"]) for arm in arms.values()),
        "memory": memory_limit > 0 and maximum_observed_memory / memory_limit <= float(gates["max_peak_rss_fraction"]),
        "short_horizon_linear_projection_within_budget": projected_seconds / 3600.0 <= float(gates["max_projected_paired_hours"]),
        "projection_evidence_sufficient": contract.get("projection_validation_status") == "LOCKED_REPEAT_AND_LONG_HORIZON_TAIL_COMPLETE",
    }
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "verdict": "SCALE_GATE_PASSED" if all(checks.values()) else "SCALE_GATE_BLOCKED",
        "science_status": "BLOCKED_PENDING_PHYSICAL_POPULATION_PRIOR_TIMESTEP_CONVERGENCE_AND_INDEPENDENT_METHOD",
        "contract_sha256": _sha256_file(contract_file),
        "contract": contract,
        "runtime": {
            "rebound_version": rebound.__version__,
            "rebound_build": rebound_build,
            "rebound_binary_sha256": rebound_integrity["binary_sha256"],
            "rebound_wheel_sha256": rebound_integrity["wheel_sha256"],
            "runner_source_manifest": software_manifest,
            "python_phase_generator": "sha256-counter-open-uniform/v1",
            "wall_seconds": time.perf_counter() - run_started,
            "peak_process_rss_bytes": peak_rss,
            "effective_memory_limit_bytes": memory_limit,
            "maximum_observed_memory_bytes": maximum_observed_memory,
            "maximum_observed_memory_fraction": maximum_observed_memory / memory_limit if memory_limit else None,
        },
        "input_integrity": {
            "state_sha256": actual_hashes,
            **match,
            "template_orbit_shape_sha256": template_digest,
            "expanded_relative_initial_state_sha256": phase_digest,
        },
        "design_counts": {
            "seed_blocks": blocks,
            "replicates_per_block": replicates_per_block,
            "tracers_per_replicate": tracers_per_replicate,
            "replicates": blocks * replicates_per_block,
            "tracers_per_arm": tracer_count,
            "paired_tracer_trajectories": 2 * tracer_count,
        },
        "arms": arms,
        "source_control_comparison_at_gate_endpoint": comparison,
        "projection": {
            "target_years": target_years,
            "linear_projected_paired_seconds": projected_seconds,
            "linear_projected_paired_hours": projected_seconds / 3600.0,
            "classification": "SHORT_HORIZON_ESTIMATE_ONLY",
            "warning": "This is not a validated wall-time bound. It excludes encounter-tail slowdown, checkpoint I/O, convergence reruns, and independent-method validation; repeated timing and a long-horizon tail pilot remain required.",
        },
        "operational_checks": checks,
        "nonclaims": [
            "This scale gate is not a Planet X detection or exclusion.",
            "Phase-expanding 15 seed shapes is not an observational population model.",
            "Endpoint injection counts are not minimum-perihelion event counts.",
            "A full scientific verdict requires a locked physical prior, timestep convergence, close-encounter audits, and an independent algorithm.",
        ],
    }
    _atomic_json(Path(output_path), result)
    return result
