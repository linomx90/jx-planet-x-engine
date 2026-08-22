"""Fail-closed DE441/Horizons compatibility validation for the JX engine.

The validation intentionally compares a ten-body Newtonian point-mass model
with DE441 only over a short, predeclared arc.  It is a force-model
compatibility check, not an ephemeris reconstruction or orbit fit.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


AU_KM = 149_597_870.700
DAY_SECONDS = 86_400.0
EXPECTED_BODY_IDS = tuple(range(1, 11))
OUTER_GATE_IDS = (5, 6, 7, 8)
STATE_KEYS = ("x", "y", "z", "vx", "vy", "vz")


class ValidationBlocked(RuntimeError):
    """Raised when a locked prerequisite cannot be verified."""


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_reference(path: str | Path) -> tuple[list[float], dict[float, dict[int, dict[str, Any]]]]:
    reference: dict[float, dict[int, dict[str, Any]]] = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "body_id",
        "body_name",
        "jd_tdb",
        "x_au",
        "y_au",
        "z_au",
        "vx_au_per_day",
        "vy_au_per_day",
        "vz_au_per_day",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValidationBlocked("normalized Horizons reference schema is incomplete")
    for row in rows:
        body_id = int(row["body_id"])
        epoch = float(row["jd_tdb"])
        state = {
            "body_id": body_id,
            "body_name": row["body_name"],
            "x": float(row["x_au"]),
            "y": float(row["y_au"]),
            "z": float(row["z_au"]),
            "vx": float(row["vx_au_per_day"]),
            "vy": float(row["vy_au_per_day"]),
            "vz": float(row["vz_au_per_day"]),
        }
        if not all(math.isfinite(state[key]) for key in STATE_KEYS):
            raise ValidationBlocked("normalized Horizons reference contains a non-finite value")
        if body_id in reference.setdefault(epoch, {}):
            raise ValidationBlocked(f"duplicate body {body_id} at JD {epoch}")
        reference[epoch][body_id] = state
    epochs = sorted(reference)
    if any(tuple(sorted(reference[epoch])) != EXPECTED_BODY_IDS for epoch in epochs):
        raise ValidationBlocked("each reference epoch must contain barycenters 1 through 10 exactly once")
    return epochs, reference


def load_gm(path: str | Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            body_id = int(row["body_id"])
            gm_km3_s2 = float(row["gm_km3_s2"])
            gm_au3_day2 = gm_km3_s2 * DAY_SECONDS**2 / AU_KM**3
            if body_id in result or not math.isfinite(gm_au3_day2) or gm_au3_day2 <= 0.0:
                raise ValidationBlocked("GM table contains a duplicate, non-finite, or non-positive value")
            result[body_id] = {
                "body_name": row["body_name"],
                "gm_km3_s2": gm_km3_s2,
                "gm_au3_day2": gm_au3_day2,
            }
    if tuple(sorted(result)) != EXPECTED_BODY_IDS:
        raise ValidationBlocked("GM table must contain barycenters 1 through 10 exactly once")
    return result


def _resolve_locked_path(project_root: Path, relative: str) -> Path:
    candidate = (project_root / relative).resolve()
    try:
        candidate.relative_to(project_root.resolve())
    except ValueError as error:
        raise ValidationBlocked(f"locked path leaves project root: {relative}") from error
    return candidate


def _verify_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise ValidationBlocked(f"locked {label} is missing: {path}")
    observed = sha256(path)
    if observed != expected:
        raise ValidationBlocked(f"locked {label} hash changed: expected {expected}, observed {observed}")
    return observed


def verify_locked_inputs(contract: dict[str, Any], project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if contract.get("schema") != "jx-de441-horizons-contract/v1":
        raise ValidationBlocked("unknown validation contract schema")
    if contract.get("lock_state") != "PRELOCKED_BEFORE_MODEL_EXECUTION":
        raise ValidationBlocked("contract is not marked prelocked")

    verified: dict[str, Any] = {"files": {}}
    for label, record in contract["locked_files"].items():
        path = _resolve_locked_path(root, record["path"])
        observed = _verify_hash(path, record["sha256"], label)
        verified["files"][label] = {"path": str(path), "sha256": observed}

    manifest_path = Path(verified["files"]["reference_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    vector_path = Path(verified["files"]["reference_vectors"]["path"])
    if manifest.get("normalized_sha256") != sha256(vector_path):
        raise ValidationBlocked("reference manifest does not identify the locked normalized vector file")
    if manifest.get("ephemeris_source_required") != "DE441":
        raise ValidationBlocked("reference manifest no longer requires DE441")
    if manifest.get("reference_frame") != "ICRF" or manifest.get("time_scale") != "TDB":
        raise ValidationBlocked("reference frame or time scale changed")
    raw_root = manifest_path.parent.resolve()
    for record in manifest.get("raw_responses", []):
        raw_path = (raw_root / record["raw_path"]).resolve()
        try:
            raw_path.relative_to(raw_root)
        except ValueError as error:
            raise ValidationBlocked("raw response path leaves the reference directory") from error
        _verify_hash(raw_path, record["raw_sha256"], f"raw Horizons body {record['body_id']}")
        raw_text = raw_path.read_text(encoding="utf-8")
        markers = (
            "{source: DE441}",
            "Solar System Barycenter (0)",
            "Output units    : AU-D",
            "Output type     : GEOMETRIC cartesian states",
            "Reference frame : ICRF",
        )
        if any(marker not in raw_text for marker in markers):
            raise ValidationBlocked(f"raw Horizons response {raw_path.name} lost a required source marker")

    epochs, reference = load_reference(vector_path)
    expected_epochs = [float(value) for value in contract["epochs_jd_tdb"]]
    if epochs != expected_epochs:
        raise ValidationBlocked(f"reference epoch grid changed: {epochs}")
    gm = load_gm(Path(verified["files"]["gm_table"]["path"]))
    if tuple(contract["model"]["active_body_ids"]) != EXPECTED_BODY_IDS:
        raise ValidationBlocked("active-body declaration changed")
    if tuple(contract["model"]["science_gate_body_ids"]) != OUTER_GATE_IDS:
        raise ValidationBlocked("science-gate body declaration changed")
    verified.update({"manifest": manifest, "epochs": epochs, "reference": reference, "gm": gm})
    return verified


def verify_rebound_backend(contract: dict[str, Any]) -> dict[str, str]:
    try:
        import rebound
    except ImportError as error:
        raise ValidationBlocked("the locked REBOUND backend is unavailable") from error
    expected = contract["backend"]
    if rebound.__version__ != expected["rebound_version"]:
        raise ValidationBlocked(
            f"REBOUND version changed: expected {expected['rebound_version']}, observed {rebound.__version__}"
        )
    specification = importlib.util.find_spec("librebound")
    if specification is None or specification.origin is None:
        raise ValidationBlocked("cannot identify the REBOUND shared library")
    library = Path(specification.origin).resolve()
    observed_hash = _verify_hash(library, expected["librebound_sha256"], "REBOUND shared library")
    return {"version": rebound.__version__, "library_path": str(library), "library_sha256": observed_hash}


def _angular_momentum(particles: Iterable[Any]) -> tuple[float, float, float]:
    result = [0.0, 0.0, 0.0]
    for particle in particles:
        result[0] += particle.m * (particle.y * particle.vz - particle.z * particle.vy)
        result[1] += particle.m * (particle.z * particle.vx - particle.x * particle.vz)
        result[2] += particle.m * (particle.x * particle.vy - particle.y * particle.vx)
    return tuple(result)


def _norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _capture_state(simulation: Any, epoch: float, body_ids: list[int]) -> dict[int, dict[str, float]]:
    captured: dict[int, dict[str, float]] = {}
    for body_id, particle in zip(body_ids, simulation.particles, strict=True):
        captured[body_id] = {
            "x": particle.x,
            "y": particle.y,
            "z": particle.z,
            "vx": particle.vx,
            "vy": particle.vy,
            "vz": particle.vz,
        }
    if not all(math.isfinite(value) for state in captured.values() for value in state.values()):
        raise RuntimeError(f"non-finite JX state at JD {epoch}")
    return captured


def run_fixed_ias15(
    reference: dict[float, dict[int, dict[str, Any]]],
    epochs: list[float],
    gm: dict[int, dict[str, Any]],
    dt_days: float,
) -> dict[str, Any]:
    import rebound

    body_ids = list(EXPECTED_BODY_IDS)
    simulation = rebound.Simulation()
    simulation.G = 1.0
    simulation.integrator = "ias15"
    simulation.dt = dt_days
    simulation.ri_ias15.adaptive_mode = 2
    simulation.ri_ias15.min_dt = 0.0
    simulation.ri_ias15.epsilon = 0.0
    initial = reference[epochs[0]]
    for body_id in body_ids:
        state = initial[body_id]
        simulation.add(
            m=gm[body_id]["gm_au3_day2"],
            x=state["x"],
            y=state["y"],
            z=state["z"],
            vx=state["vx"],
            vy=state["vy"],
            vz=state["vz"],
        )

    interval_days = epochs[1] - epochs[0]
    steps_per_interval = round(interval_days / dt_days)
    if not math.isclose(steps_per_interval * dt_days, interval_days, rel_tol=0.0, abs_tol=1e-13):
        raise ValidationBlocked("fixed timestep does not divide the locked output interval")
    initial_energy = simulation.energy()
    initial_angular = _angular_momentum(simulation.particles)
    initial_angular_norm = _norm(initial_angular)
    states: dict[float, dict[int, dict[str, float]]] = {}
    invariant_rows = []
    started = time.perf_counter()
    for index, epoch in enumerate(epochs):
        if index:
            simulation.steps(steps_per_interval)
        expected_time = epoch - epochs[0]
        if not math.isclose(simulation.t, expected_time, rel_tol=0.0, abs_tol=2e-10):
            raise RuntimeError(f"JX integration time mismatch at output {index}: {simulation.t} vs {expected_time}")
        states[epoch] = _capture_state(simulation, epoch, body_ids)
        energy = simulation.energy()
        angular = _angular_momentum(simulation.particles)
        invariant_rows.append(
            {
                "jd_tdb": epoch,
                "relative_energy_drift": abs((energy - initial_energy) / initial_energy),
                "relative_angular_momentum_drift": _norm(
                    angular[axis] - initial_angular[axis] for axis in range(3)
                )
                / initial_angular_norm,
            }
        )
    wall_seconds = time.perf_counter() - started
    expected_steps = steps_per_interval * (len(epochs) - 1)
    steps_done = int(simulation.steps_done)
    if steps_done != expected_steps:
        raise RuntimeError(f"fixed-step count mismatch: {steps_done} vs {expected_steps}")
    return {
        "dt_days": dt_days,
        "steps_per_output_interval": steps_per_interval,
        "expected_steps": expected_steps,
        "steps_done": steps_done,
        "wall_seconds": wall_seconds,
        "iterations_max_exceeded": int(simulation.ri_ias15._iterations_max_exceeded),
        "max_relative_energy_drift": max(row["relative_energy_drift"] for row in invariant_rows),
        "max_relative_angular_momentum_drift": max(
            row["relative_angular_momentum_drift"] for row in invariant_rows
        ),
        "invariants": invariant_rows,
        "states": states,
    }


def _heliocentric(state: dict[int, dict[str, Any]], body_id: int) -> tuple[float, ...]:
    body = state[body_id]
    sun = state[10]
    return tuple(body[key] - sun[key] for key in STATE_KEYS)


def compare_state_sets(
    left: dict[float, dict[int, dict[str, Any]]],
    right: dict[float, dict[int, dict[str, Any]]],
    epochs: list[float],
    body_ids: Iterable[int] = EXPECTED_BODY_IDS,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, float]]]:
    rows: list[dict[str, Any]] = []
    aggregates: dict[int, dict[str, list[float]]] = {}
    for epoch in epochs:
        for body_id in body_ids:
            left_state = _heliocentric(left[epoch], body_id)
            right_state = _heliocentric(right[epoch], body_id)
            position = _norm(left_state[index] - right_state[index] for index in range(3))
            velocity = _norm(left_state[index] - right_state[index] for index in range(3, 6))
            rows.append(
                {
                    "jd_tdb": epoch,
                    "body_id": body_id,
                    "position_residual_au": position,
                    "velocity_residual_au_per_day": velocity,
                }
            )
            aggregate = aggregates.setdefault(body_id, {"position": [], "velocity": []})
            aggregate["position"].append(position)
            aggregate["velocity"].append(velocity)
    summary = {}
    for body_id, values in aggregates.items():
        summary[body_id] = {
            "max_position_residual_au": max(values["position"]),
            "rms_position_residual_au": math.sqrt(
                sum(value * value for value in values["position"]) / len(values["position"])
            ),
            "final_position_residual_au": values["position"][-1],
            "max_velocity_residual_au_per_day": max(values["velocity"]),
            "rms_velocity_residual_au_per_day": math.sqrt(
                sum(value * value for value in values["velocity"]) / len(values["velocity"])
            ),
            "final_velocity_residual_au_per_day": values["velocity"][-1],
        }
    return rows, summary


def evaluate_gates(
    contract: dict[str, Any],
    coarse: dict[str, Any],
    tight: dict[str, Any],
    convergence_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    thresholds = {key: float(value) for key, value in contract["acceptance_gates"].items()}
    outer_reference = [row for row in reference_rows if row["body_id"] in OUTER_GATE_IDS]
    outer_convergence = [row for row in convergence_rows if row["body_id"] in OUTER_GATE_IDS]
    initial_rows = [row for row in reference_rows if row["jd_tdb"] == float(contract["epochs_jd_tdb"][0])]
    values = {
        "initial_state_position_residual_au": max(row["position_residual_au"] for row in initial_rows),
        "initial_state_velocity_residual_au_per_day": max(
            row["velocity_residual_au_per_day"] for row in initial_rows
        ),
        "outer_convergence_position_au": max(row["position_residual_au"] for row in outer_convergence),
        "outer_convergence_velocity_au_per_day": max(
            row["velocity_residual_au_per_day"] for row in outer_convergence
        ),
        "outer_de441_position_residual_au": max(row["position_residual_au"] for row in outer_reference),
        "outer_de441_velocity_residual_au_per_day": max(
            row["velocity_residual_au_per_day"] for row in outer_reference
        ),
        "coarse_relative_energy_drift": coarse["max_relative_energy_drift"],
        "tight_relative_energy_drift": tight["max_relative_energy_drift"],
        "coarse_relative_angular_momentum_drift": coarse["max_relative_angular_momentum_drift"],
        "tight_relative_angular_momentum_drift": tight["max_relative_angular_momentum_drift"],
        "ias15_iterations_max_exceeded": max(
            coarse["iterations_max_exceeded"], tight["iterations_max_exceeded"]
        ),
    }
    gate_to_threshold = {
        "initial_state_position_residual_au": "max_initial_position_residual_au",
        "initial_state_velocity_residual_au_per_day": "max_initial_velocity_residual_au_per_day",
        "outer_convergence_position_au": "max_outer_convergence_position_au",
        "outer_convergence_velocity_au_per_day": "max_outer_convergence_velocity_au_per_day",
        "outer_de441_position_residual_au": "max_outer_de441_position_residual_au",
        "outer_de441_velocity_residual_au_per_day": "max_outer_de441_velocity_residual_au_per_day",
        "coarse_relative_energy_drift": "max_relative_energy_drift",
        "tight_relative_energy_drift": "max_relative_energy_drift",
        "coarse_relative_angular_momentum_drift": "max_relative_angular_momentum_drift",
        "tight_relative_angular_momentum_drift": "max_relative_angular_momentum_drift",
        "ias15_iterations_max_exceeded": "max_ias15_iterations_exceeded",
    }
    gates = {}
    for name, value in values.items():
        threshold_name = gate_to_threshold[name]
        threshold = thresholds[threshold_name]
        gates[name] = {
            "value": value,
            "threshold": threshold,
            "operator": "<=",
            "passed": value <= threshold,
        }
    gates["completed_exact_step_counts"] = {
        "value": coarse["steps_done"] == coarse["expected_steps"]
        and tight["steps_done"] == tight["expected_steps"],
        "threshold": True,
        "operator": "==",
        "passed": coarse["steps_done"] == coarse["expected_steps"]
        and tight["steps_done"] == tight["expected_steps"],
    }
    return gates


def verdict_from_gates(gates: dict[str, dict[str, Any]]) -> str:
    return "PASSED" if gates and all(gate["passed"] for gate in gates.values()) else "INVALID"


def render_state_csv(runs: dict[str, dict[str, Any]], names: dict[int, str]) -> str:
    import io

    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("run", "jd_tdb", "body_id", "body_name", *STATE_KEYS))
    for run_name, run in runs.items():
        for epoch in sorted(run["states"]):
            for body_id in EXPECTED_BODY_IDS:
                state = run["states"][epoch][body_id]
                writer.writerow(
                    (
                        run_name,
                        format(epoch, ".9f"),
                        body_id,
                        names[body_id],
                        *(format(state[key], ".17e") for key in STATE_KEYS),
                    )
                )
    return output.getvalue()


def render_residual_csv(rows: list[dict[str, Any]], names: dict[int, str]) -> str:
    import io

    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "comparison",
            "jd_tdb",
            "body_id",
            "body_name",
            "position_residual_au",
            "position_residual_km",
            "velocity_residual_au_per_day",
            "velocity_residual_m_per_s",
        )
    )
    for row in rows:
        writer.writerow(
            (
                row["comparison"],
                format(row["jd_tdb"], ".9f"),
                row["body_id"],
                names[row["body_id"]],
                format(row["position_residual_au"], ".17e"),
                format(row["position_residual_au"] * AU_KM, ".17e"),
                format(row["velocity_residual_au_per_day"], ".17e"),
                format(row["velocity_residual_au_per_day"] * AU_KM * 1000.0 / DAY_SECONDS, ".17e"),
            )
        )
    return output.getvalue()


def run_validation(
    contract_path: str | Path,
    project_root: str | Path,
    output_path: str | Path,
    state_csv_path: str | Path,
    residual_csv_path: str | Path,
) -> dict[str, Any]:
    started_utc = datetime.now(timezone.utc).isoformat()
    contract_file = Path(contract_path).resolve()
    contract = json.loads(contract_file.read_text(encoding="utf-8"))
    verified = verify_locked_inputs(contract, project_root)
    backend = verify_rebound_backend(contract)
    epochs = verified["epochs"]
    reference = verified["reference"]
    gm = verified["gm"]
    run_specs = contract["runs"]
    coarse = run_fixed_ias15(reference, epochs, gm, float(run_specs["coarse"]["dt_days"]))
    tight = run_fixed_ias15(reference, epochs, gm, float(run_specs["tight"]["dt_days"]))

    convergence_rows, convergence_by_body = compare_state_sets(
        coarse["states"], tight["states"], epochs
    )
    reference_rows, reference_by_body = compare_state_sets(tight["states"], reference, epochs)
    for row in convergence_rows:
        row["comparison"] = "coarse_vs_tight"
    for row in reference_rows:
        row["comparison"] = "tight_jx_vs_horizons_de441"
    gates = evaluate_gates(contract, coarse, tight, convergence_rows, reference_rows)
    verdict = verdict_from_gates(gates)
    names = {body_id: reference[epochs[0]][body_id]["body_name"] for body_id in EXPECTED_BODY_IDS}
    atomic_write_text(state_csv_path, render_state_csv({"coarse": coarse, "tight": tight}, names))
    atomic_write_text(residual_csv_path, render_residual_csv(convergence_rows + reference_rows, names))

    def public_run(run: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in run.items() if key != "states"}

    result = {
        "schema": "jx-de441-horizons-result/v1",
        "science_verdict": verdict,
        "status": "DE441_NEWTONIAN_COMPATIBILITY_PASSED" if verdict == "PASSED" else "INVALID",
        "classification": "EXTERNAL_REFERENCE_COMPATIBILITY_TEST",
        "started_at_utc": started_utc,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_path": str(contract_file),
        "contract_sha256": sha256(contract_file),
        "verified_locked_inputs": {"files": verified["files"]},
        "backend": backend,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "model": contract["model"],
        "runs": {"coarse": public_run(coarse), "tight": public_run(tight)},
        "gates": gates,
        "all_gates_passed": verdict == "PASSED",
        "reference_residuals_by_body": {str(key): value for key, value in reference_by_body.items()},
        "timestep_convergence_by_body": {str(key): value for key, value in convergence_by_body.items()},
        "derived_outer_gate_units": {
            "max_position_residual_km": gates["outer_de441_position_residual_au"]["value"] * AU_KM,
            "max_velocity_residual_m_per_s": gates["outer_de441_velocity_residual_au_per_day"]["value"]
            * AU_KM
            * 1000.0
            / DAY_SECONDS,
        },
        "state_csv": {"path": str(Path(state_csv_path).resolve()), "sha256": sha256(state_csv_path)},
        "residual_csv": {
            "path": str(Path(residual_csv_path).resolve()),
            "sha256": sha256(residual_csv_path),
        },
        "claim_decision": "SCREENING_ONLY" if verdict == "PASSED" else "INVALID",
        "scientific_scope": contract["scientific_scope"],
        "nonclaim": contract["nonclaim"],
    }
    atomic_write_text(output_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def write_blocked_result(
    output_path: str | Path,
    contract_path: str | Path,
    error: BaseException,
) -> dict[str, Any]:
    contract_file = Path(contract_path).resolve()
    result = {
        "schema": "jx-de441-horizons-result/v1",
        "science_verdict": "BLOCKED",
        "status": "BLOCKED",
        "classification": "VALIDATION_NOT_COMPLETED",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_path": str(contract_file),
        "contract_sha256": sha256(contract_file) if contract_file.is_file() else None,
        "error_type": type(error).__name__,
        "error": str(error),
        "all_gates_passed": False,
        "claim_decision": "BLOCKED",
        "nonclaim": "No scientific compatibility conclusion can be drawn from a blocked execution.",
    }
    atomic_write_text(output_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
