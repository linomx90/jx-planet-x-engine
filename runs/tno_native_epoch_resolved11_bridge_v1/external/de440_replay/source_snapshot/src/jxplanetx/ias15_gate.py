"""Adaptive IAS15 convergence gate for the preserved anomaly-zone states.

This module deliberately imports REBOUND only inside run functions so the
stdlib-only JX core remains usable without the optional validated dependency.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any


AU_KM = 149_597_870.700
YEAR_D = 365.25
SELECTED_TRACERS = tuple(
    f"t{index:03d}"
    for base in (0, 9, 18, 27, 36)
    for index in (base, base + 4, base + 8)
)

IAS15_GATES = {
    "massive_position_AU": 1e-8,
    "massive_velocity_AU_per_yr": 1e-10,
    "tracer_q_AU": 1e-2,
    "tracer_i_deg": 1e-2,
    "bound_mismatches": 0,
}

POPULATION_GATES = {
    "max_q_lt30_count_difference": 0,
    "max_mean_q_difference_AU": 0.1,
    "max_wasserstein_q_AU": 0.1,
    "bound_mismatches": 0,
}

SOURCE_EFFECT_GATES = {
    "max_count_effect_disagreement": 0,
    "max_mean_q_effect_disagreement_AU": 0.2,
    "max_wasserstein_effect_disagreement_AU": 0.2,
    "minimum_resolved_wasserstein_effect_AU": 0.2,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _solar_g(gm_path: Path) -> float:
    import re

    text = gm_path.read_text(encoding="utf-8")
    match = re.search(r"BODY10_GM\s*=\s*\(\s*([0-9.+\-EDed]+)\s*\)", text)
    if match is None:
        raise ValueError("official solar GM is missing")
    gm_km3_s2 = float(match.group(1).replace("D", "E").replace("d", "e"))
    return gm_km3_s2 * (86400.0 * YEAR_D) ** 2 / AU_KM**3


def _configuration_rows(state_path: Path, family: str, inclination: float, phase: int) -> list[dict[str, str]]:
    rows = list(csv.DictReader(state_path.open(newline="", encoding="utf-8")))
    wanted_i = "0.0" if family == "baseline" else str(float(inclination))
    chosen = [
        row for row in rows
        if row["family"] == family
        and row["source_i_deg"] == wanted_i
        and int(row["phase_index"]) == phase
        and (row["label"] in ("Sun", "Jupiter", "Saturn", "Uranus", "Neptune", "source") or row["label"] in SELECTED_TRACERS)
    ]
    expected = 20 if family == "baseline" else 21
    if len(chosen) != expected:
        raise ValueError(f"expected {expected} selected rows, found {len(chosen)}")
    chosen.sort(key=lambda row: int(row["particle_index"]))
    return chosen


def run_ias15_member(
    state_path: str | Path,
    gm_path: str | Path,
    output_csv: str | Path,
    summary_json: str | Path,
    family: str,
    inclination: float,
    phase: int,
    epsilon: float,
    duration_years: int = 100_000,
    output_interval_years: int = 1_000,
) -> dict[str, Any]:
    import rebound

    state = Path(state_path)
    rows = _configuration_rows(state, family, inclination, phase)
    sim = rebound.Simulation()
    sim.G = _solar_g(Path(gm_path))
    for row in rows:
        sim.add(
            m=float(row["mass_Msun"]),
            x=float(row["x_AU"]), y=float(row["y_AU"]), z=float(row["z_AU"]),
            vx=float(row["vx_AU_per_year"]), vy=float(row["vy_AU_per_year"]), vz=float(row["vz_AU_per_year"]),
            hash=row["label"],
        )
    sim.N_active = 5 if family == "baseline" else 6
    sim.testparticle_type = 0
    sim.integrator = "ias15"
    sim.ri_ias15.epsilon = epsilon
    initial_energy = sim.energy()
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    fields = ("time_year", "body", "name", "x", "y", "z", "vx", "vy", "vz", "a", "e", "q", "i_deg", "bound")
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for output_time in range(0, duration_years + 1, output_interval_years):
            if output_time:
                sim.integrate(float(output_time), exact_finish_time=1)
            sun = sim.particles["Sun"]
            for body, row in enumerate(rows):
                particle = sim.particles[row["label"]]
                if row["label"] == "Sun":
                    a = e = q = i_deg = math.nan
                    bound = 1
                else:
                    orbit = particle.orbit(primary=sun)
                    a, e = orbit.a, orbit.e
                    bound = int(a > 0.0 and e < 1.0)
                    q = a * (1.0 - e) if bound else math.nan
                    i_deg = math.degrees(orbit.inc)
                writer.writerow(
                    {
                        "time_year": output_time, "body": body, "name": row["label"],
                        "x": repr(particle.x), "y": repr(particle.y), "z": repr(particle.z),
                        "vx": repr(particle.vx), "vy": repr(particle.vy), "vz": repr(particle.vz),
                        "a": repr(a), "e": repr(e), "q": repr(q), "i_deg": repr(i_deg), "bound": bound,
                    }
                )
    expected_rows = (duration_years // output_interval_years + 1) * len(rows)
    with output.open(newline="", encoding="utf-8") as stream:
        actual_rows = sum(1 for _ in csv.DictReader(stream))
    if actual_rows != expected_rows:
        raise IOError(f"IAS15 trajectory row-count failure: expected {expected_rows}, found {actual_rows}")
    wall = time.perf_counter() - start
    final_energy = sim.energy()
    result = {
        "method": "REBOUND IAS15 adaptive 15th-order integrator",
        "rebound_version": rebound.__version__,
        "family": family, "inclination_deg": inclination, "phase": phase,
        "epsilon": epsilon, "duration_years": duration_years,
        "output_interval_years": output_interval_years,
        "tracers": list(SELECTED_TRACERS),
        "state_sha256": _sha256(state),
        "trajectory_rows": actual_rows,
        "trajectory_sha256": _sha256(output),
        "wall_seconds": wall,
        "relative_energy_drift": abs((final_energy - initial_energy) / initial_energy),
    }
    Path(summary_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _load(path: Path) -> dict[tuple[int, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {(int(row["time_year"]), row["name"]): row for row in csv.DictReader(stream)}


def compare_ias15_members(loose_path: str | Path, tight_path: str | Path, label: str) -> dict[str, Any]:
    loose, tight = _load(Path(loose_path)), _load(Path(tight_path))
    if set(loose) != set(tight):
        raise ValueError("IAS15 trajectory support mismatch")
    maxima = {name: {"max": 0.0, "time_year": 0, "body": ""} for name in IAS15_GATES if name != "bound_mismatches"}
    bound_mismatches = 0
    for key in sorted(loose):
        left, right = loose[key], tight[key]
        massive = left["name"] in ("Sun", "Jupiter", "Saturn", "Uranus", "Neptune", "source")
        position = math.sqrt(sum((float(left[k]) - float(right[k])) ** 2 for k in ("x", "y", "z")))
        velocity = math.sqrt(sum((float(left[k]) - float(right[k])) ** 2 for k in ("vx", "vy", "vz")))
        candidates: list[tuple[str, float]] = []
        if massive:
            candidates.extend((("massive_position_AU", position), ("massive_velocity_AU_per_yr", velocity)))
        elif left["bound"] == right["bound"] == "1":
            candidates.extend((("tracer_q_AU", abs(float(left["q"]) - float(right["q"]))), ("tracer_i_deg", abs(float(left["i_deg"]) - float(right["i_deg"])))))
        for metric, value in candidates:
            if value > maxima[metric]["max"]:
                maxima[metric] = {"max": value, "time_year": key[0], "body": key[1]}
        bound_mismatches += left["bound"] != right["bound"]
    metrics: dict[str, Any] = {}
    for name, maximum in maxima.items():
        gate = IAS15_GATES[name]
        metrics[name] = {**maximum, "gate": gate, "passed": maximum["max"] <= gate}
    metrics["bound_mismatches"] = {"count": bound_mismatches, "gate": 0, "passed": bound_mismatches == 0}
    return {"label": label, "rows": len(loose), "metrics": metrics, "passed": all(item["passed"] for item in metrics.values())}


def compare_ias15_population(
    path_a: str | Path,
    path_b: str | Path,
    label: str,
    classification: str,
) -> dict[str, Any]:
    """Compare population observables after individual chaotic separation."""
    left, right = _load(Path(path_a)), _load(Path(path_b))
    if set(left) != set(right):
        raise ValueError("IAS15 population trajectory support mismatch")
    times = sorted({key[0] for key in left})
    tracers = sorted({key[1] for key in left if key[1].startswith("t")})
    tracer_diagnostics: dict[str, Any] = {}
    for tracer in tracers:
        first_failure = None
        max_dq = max_di = 0.0
        for time_year in times:
            a, b = left[(time_year, tracer)], right[(time_year, tracer)]
            if a["bound"] != b["bound"] or a["bound"] != "1":
                continue
            dq = abs(float(a["q"]) - float(b["q"]))
            di = abs(float(a["i_deg"]) - float(b["i_deg"]))
            max_dq, max_di = max(max_dq, dq), max(max_di, di)
            if first_failure is None and (dq > IAS15_GATES["tracer_q_AU"] or di > IAS15_GATES["tracer_i_deg"]):
                first_failure = time_year
        tracer_diagnostics[tracer] = {
            "first_pointwise_failure_year": first_failure,
            "max_abs_q_difference_AU": max_dq,
            "max_abs_i_difference_deg": max_di,
            "pointwise_passed": first_failure is None,
        }
    maximum_count = 0
    maximum_mean = 0.0
    maximum_wasserstein = 0.0
    bound_mismatches = 0
    worst_epochs = {"count": 0, "mean_q": 0, "wasserstein_q": 0}
    for time_year in times:
        qa, qb = [], []
        for tracer in tracers:
            a, b = left[(time_year, tracer)], right[(time_year, tracer)]
            bound_mismatches += a["bound"] != b["bound"]
            if a["bound"] == b["bound"] == "1":
                qa.append(float(a["q"]))
                qb.append(float(b["q"]))
        if len(qa) != len(tracers):
            continue
        count_difference = abs(sum(q < 30.0 for q in qa) - sum(q < 30.0 for q in qb))
        mean_difference = abs(sum(qa) / len(qa) - sum(qb) / len(qb))
        wasserstein = sum(abs(a - b) for a, b in zip(sorted(qa), sorted(qb))) / len(qa)
        if count_difference > maximum_count:
            maximum_count, worst_epochs["count"] = count_difference, time_year
        if mean_difference > maximum_mean:
            maximum_mean, worst_epochs["mean_q"] = mean_difference, time_year
        if wasserstein > maximum_wasserstein:
            maximum_wasserstein, worst_epochs["wasserstein_q"] = wasserstein, time_year
    values = {
        "max_q_lt30_count_difference": maximum_count,
        "max_mean_q_difference_AU": maximum_mean,
        "max_wasserstein_q_AU": maximum_wasserstein,
        "bound_mismatches": bound_mismatches,
    }
    metrics = {
        name: {
            "value": value,
            "gate": POPULATION_GATES[name],
            "passed": value <= POPULATION_GATES[name],
        }
        for name, value in values.items()
    }
    return {
        "label": label,
        "classification": classification,
        "population_equation": "f_q<30(t)=(1/N) sum_i I[q_i(t)<30 AU]",
        "tracer_count": len(tracers),
        "pointwise_passing_tracers": sum(item["pointwise_passed"] for item in tracer_diagnostics.values()),
        "tracer_diagnostics": tracer_diagnostics,
        "worst_epochs_year": worst_epochs,
        "metrics": metrics,
        "population_gate_passed": all(item["passed"] for item in metrics.values()),
        "pointwise_gate_remains_failed": any(not item["pointwise_passed"] for item in tracer_diagnostics.values()),
    }


def _source_control_series(control_path: Path, source_path: Path) -> list[dict[str, float | int]]:
    control, source = _load(control_path), _load(source_path)
    times = sorted({key[0] for key in control})
    tracers = sorted({key[1] for key in control if key[1].startswith("t")})
    if any((time_year, tracer) not in source for time_year in times for tracer in tracers):
        raise ValueError("source/control tracer support mismatch")
    series = []
    for time_year in times:
        qc = [float(control[(time_year, tracer)]["q"]) for tracer in tracers]
        qs = [float(source[(time_year, tracer)]["q"]) for tracer in tracers]
        series.append(
            {
                "time_year": time_year,
                "delta_count_q_lt30": sum(q < 30.0 for q in qs) - sum(q < 30.0 for q in qc),
                "delta_mean_q_AU": sum(qs) / len(qs) - sum(qc) / len(qc),
                "wasserstein_q_AU": sum(abs(a - b) for a, b in zip(sorted(qc), sorted(qs))) / len(qc),
            }
        )
    return series


def compare_source_control_effect(
    control_first: str | Path,
    source_first: str | Path,
    control_second: str | Path,
    source_second: str | Path,
    label: str,
    classification: str,
) -> dict[str, Any]:
    first = _source_control_series(Path(control_first), Path(source_first))
    second = _source_control_series(Path(control_second), Path(source_second))
    if [row["time_year"] for row in first] != [row["time_year"] for row in second]:
        raise ValueError("source-effect epoch mismatch")
    count_disagreement = max(abs(int(a["delta_count_q_lt30"]) - int(b["delta_count_q_lt30"])) for a, b in zip(first, second))
    mean_disagreement = max(abs(float(a["delta_mean_q_AU"]) - float(b["delta_mean_q_AU"])) for a, b in zip(first, second))
    wasserstein_disagreement = max(abs(float(a["wasserstein_q_AU"]) - float(b["wasserstein_q_AU"])) for a, b in zip(first, second))
    first_max_wasserstein = max(float(row["wasserstein_q_AU"]) for row in first)
    second_max_wasserstein = max(float(row["wasserstein_q_AU"]) for row in second)
    first_max_count = max(abs(int(row["delta_count_q_lt30"])) for row in first)
    second_max_count = max(abs(int(row["delta_count_q_lt30"])) for row in second)
    metrics = {
        "max_count_effect_disagreement": {
            "value": count_disagreement, "gate": 0, "passed": count_disagreement == 0,
        },
        "max_mean_q_effect_disagreement_AU": {
            "value": mean_disagreement, "gate": 0.2, "passed": mean_disagreement <= 0.2,
        },
        "max_wasserstein_effect_disagreement_AU": {
            "value": wasserstein_disagreement, "gate": 0.2, "passed": wasserstein_disagreement <= 0.2,
        },
        "resolved_wasserstein_effect_both": {
            "first_max_AU": first_max_wasserstein,
            "second_max_AU": second_max_wasserstein,
            "gate": "> 0.2 AU in both",
            "passed": min(first_max_wasserstein, second_max_wasserstein) > 0.2,
        },
        "nonzero_low_q_count_effect_both": {
            "first_max_abs_count": first_max_count,
            "second_max_abs_count": second_max_count,
            "gate": ">= 1 tracer in both",
            "passed": min(first_max_count, second_max_count) >= 1,
        },
    }
    return {
        "label": label,
        "classification": classification,
        "effect_equation": "D_epsilon(t)=W1({q_source,i(t)},{q_control,i(t)})",
        "metrics": metrics,
        "source_effect_gate_passed": all(metric["passed"] for metric in metrics.values()),
        "scope": "numerically resolved population sensitivity; not explanatory sufficiency or detection",
    }


def finalize_ias15_equation_gate(run_root: str | Path) -> dict[str, Any]:
    root = Path(run_root)
    paths = {
        "phase0_pointwise": root / "baseline" / "comparison_tight_tighter.json",
        "phase0_population_exploratory": root / "baseline" / "population_exploratory.json",
        "phase1_control_population": root / "baseline_phase1" / "population_confirmation.json",
        "phase1_source_population": root / "middle_i25_phase1" / "population_convergence.json",
        "phase1_effect_exploratory": root / "phase1_source_effect_exploratory.json",
        "phase2_control_population": root / "phase2_control" / "population_convergence.json",
        "phase2_source_population": root / "phase2_middle_i25" / "population_convergence.json",
        "phase2_effect_confirmation": root / "phase2_source_effect.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing IAS15 gate records: " + ", ".join(missing))
    records = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    confirmation_checks = {
        "phase2_control_population_converged": records["phase2_control_population"]["population_gate_passed"],
        "phase2_source_population_converged": records["phase2_source_population"]["population_gate_passed"],
        "phase2_source_effect_resolved": records["phase2_effect_confirmation"]["source_effect_gate_passed"],
    }
    passed = all(confirmation_checks.values())
    return {
        "test_name": "JX 100-kyr adaptive IAS15 equation and population-sensitivity gate",
        "classification": "MODEL_OUTPUT numerical test",
        "equations": {
            "dynamics": "r_i_ddot=G sum_(j!=i) m_j (r_j-r_i)/|r_j-r_i|^3",
            "low_q_fraction": "f_q<30(t)=(1/N) sum_i I[q_i(t)<30 AU]",
            "population_distance": "D_epsilon(t)=W1({q_source,i(t)},{q_control,i(t)})",
        },
        "design": {
            "integrator": "REBOUND 4.4.11 IAS15",
            "duration_years": 100000,
            "epsilons": [1e-12, 1e-14],
            "tracers": list(SELECTED_TRACERS),
            "source_case": "equal-5-K500 middle family, inclination 25 deg",
            "phase2_status": "untouched confirmation; thresholds fixed before integrations",
        },
        "confirmation_checks": confirmation_checks,
        "all_required_gates_passed": passed,
        "governing_verdict": "PASS_POPULATION_SENSITIVITY" if passed else "BLOCKED_SOURCE_POPULATION_NONCONVERGENCE",
        "claim_decision": "SCREENING_ONLY" if passed else "INVALID_FOR_SOURCE_INFERENCE",
        "critical_result": records["phase2_source_population"]["metrics"],
        "resolved_effect_but_blocked": records["phase2_effect_confirmation"]["metrics"],
        "pointwise_status": "FAILED; chaotic tracer identities are not reproducible to 100 kyr",
        "scientific_scope": "No detection, explanatory sufficiency, mass, distance, orbit, or sky direction is established.",
        "next_valid_method": "A preregistered phase/uncertainty ensemble that compares converged distributions, not repeated deterministic tracer identities.",
        "record_sha256": {name: _sha256(path) for name, path in paths.items()},
    }
