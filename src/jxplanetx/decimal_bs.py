"""Independent arbitrary-precision Bulirsch–Stoer reference integrator."""

from __future__ import annotations

import csv
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Callable

from .decimal_math import D, precision_context, sin_cos

PI = D("3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628")
G = D(4) * PI * PI
SEQUENCE = (2, 4, 6, 8, 10, 12)


@dataclass
class BSStats:
    accepted_steps: int = 0
    rejected_steps: int = 0
    rhs_calls: int = 0
    maximum_level: int = 0


class NBodySystem:
    def __init__(self, names: list[str], masses: list[Decimal]):
        self.names = names
        self.masses = masses
        self.nb = len(names)
        self.nm = sum(mass > 0 for mass in masses)
        if self.nb < 5 or self.nm != 5:
            raise ValueError("DE441 reference blocks require exactly 5 massive bodies")
        self.stats = BSStats()

    def rhs(self, state: list[Decimal]) -> list[Decimal]:
        self.stats.rhs_calls += 1
        n3 = self.nb * 3
        result = list(state[n3:]) + [D(0)] * n3
        for i in range(self.nm):
            for j in range(i + 1, self.nm):
                dx = state[3 * j] - state[3 * i]
                dy = state[3 * j + 1] - state[3 * i + 1]
                dz = state[3 * j + 2] - state[3 * i + 2]
                r2 = dx * dx + dy * dy + dz * dz
                inv_r3 = D(1) / (r2 * r2.sqrt())
                fi = G * self.masses[j] * inv_r3
                fj = G * self.masses[i] * inv_r3
                result[n3 + 3 * i] += fi * dx
                result[n3 + 3 * i + 1] += fi * dy
                result[n3 + 3 * i + 2] += fi * dz
                result[n3 + 3 * j] -= fj * dx
                result[n3 + 3 * j + 1] -= fj * dy
                result[n3 + 3 * j + 2] -= fj * dz
        for i in range(self.nm, self.nb):
            for j in range(self.nm):
                dx = state[3 * j] - state[3 * i]
                dy = state[3 * j + 1] - state[3 * i + 1]
                dz = state[3 * j + 2] - state[3 * i + 2]
                r2 = dx * dx + dy * dy + dz * dz
                factor = G * self.masses[j] / (r2 * r2.sqrt())
                result[n3 + 3 * i] += factor * dx
                result[n3 + 3 * i + 1] += factor * dy
                result[n3 + 3 * i + 2] += factor * dz
        return result


class _OscillatorSystem:
    """Analytic q''=-q problem used to validate the independent BS kernel."""

    def rhs(self, state: list[Decimal]) -> list[Decimal]:
        return [state[1], -state[0]]


def validate_bs_oscillator(decimal_digits: int = 78) -> dict:
    """Integrate one radian and compare to independently evaluated sin/cos."""
    with localcontext(precision_context(decimal_digits)):
        system = _OscillatorSystem()
        state = [D(1), D(0)]
        time_value = D(0)
        step = D("0.1")
        accepted_steps = rejected_steps = 0
        while time_value < 1:
            interval = min(step, D(1) - time_value)
            accepted, candidate, error, level = bs_attempt(system, state, interval, D("1e-33"), D("1e-30"))  # type: ignore[arg-type]
            if accepted:
                state = candidate
                time_value += interval
                accepted_steps += 1
                factor = 4.0 if error == 0 else max(0.5, min(2.0, 0.9 * float(error) ** (-1.0 / (2 * level + 1))))
                step = min(D("0.25"), interval * D(str(factor)))
            else:
                rejected_steps += 1
                step = interval / D(2)
                if step < D("1e-10"):
                    raise ArithmeticError("oscillator validation step underflow")
        sine, cosine = sin_cos(D(1))
        maximum_error = max(abs(state[0] - cosine), abs(state[1] + sine))
        gate = D("1e-25")
        return {
            "problem": "harmonic oscillator q''=-q, q(0)=1, p(0)=0, t=1",
            "maximum_absolute_state_error": str(maximum_error),
            "gate": str(gate),
            "passed": maximum_error <= gate,
            "accepted_steps": accepted_steps,
            "rejected_steps": rejected_steps,
            "decimal_digits": decimal_digits,
        }


def load_state(path: str | Path) -> tuple[NBodySystem, list[Decimal]]:
    rows = list(csv.DictReader(Path(path).open(newline="", encoding="utf-8")))
    if len(rows) < 5:
        raise ValueError("expected at least five state rows")
    rows.sort(key=lambda row: int(row["index"]))
    names = [row["name"] for row in rows]
    masses = [D(row["mass"]) for row in rows]
    positions = [D(row[key]) for row in rows for key in ("x", "y", "z")]
    velocities = [D(row[key]) for row in rows for key in ("vx", "vy", "vz")]
    return NBodySystem(names, masses), positions + velocities


def modified_midpoint(rhs: Callable[[list[Decimal]], list[Decimal]], y: list[Decimal], interval: Decimal, subdivisions: int) -> list[Decimal]:
    h = interval / D(subdivisions)
    previous = list(y)
    derivative = rhs(previous)
    current = [value + h * slope for value, slope in zip(previous, derivative)]
    two_h = D(2) * h
    for _ in range(1, subdivisions):
        derivative = rhs(current)
        following = [old + two_h * slope for old, slope in zip(previous, derivative)]
        previous, current = current, following
    derivative = rhs(current)
    return [(old + now + h * slope) / D(2) for old, now, slope in zip(previous, current, derivative)]


def _normalized_error(a: list[Decimal], b: list[Decimal], original: list[Decimal], atol: Decimal, rtol: Decimal) -> Decimal:
    maximum = D(0)
    for left, right, start in zip(a, b, original):
        scale = atol + rtol * max(abs(left), abs(start))
        value = abs(left - right) / scale
        if value > maximum:
            maximum = value
    return maximum


def bs_attempt(system: NBodySystem, state: list[Decimal], interval: Decimal, atol: Decimal, rtol: Decimal) -> tuple[bool, list[Decimal], Decimal, int]:
    table: list[list[list[Decimal]]] = []
    last_error = D("Infinity")
    for k, subdivisions in enumerate(SEQUENCE):
        row = [modified_midpoint(system.rhs, state, interval, subdivisions)]
        for j in range(1, k + 1):
            # Neville extrapolation uses x_k=(H/n_k)^2. The denominator is the
            # ratio of abscissae, not that ratio raised again for each column.
            ratio = (D(subdivisions) / D(SEQUENCE[k - j])) ** 2
            prior = table[k - 1][j - 1]
            base = row[j - 1]
            row.append([value + (value - old) / (ratio - D(1)) for value, old in zip(base, prior)])
        table.append(row)
        if k >= 2:
            last_error = _normalized_error(row[-1], row[-2], state, atol, rtol)
            if last_error <= 1:
                return True, row[-1], last_error, k + 1
    return False, table[-1][-1], last_error, len(SEQUENCE)


def invariants(system: NBodySystem, state: list[Decimal]) -> tuple[Decimal, tuple[Decimal, Decimal, Decimal]]:
    n3 = system.nb * 3
    energy = D(0)
    lx = ly = lz = D(0)
    for i in range(system.nm):
        vx, vy, vz = state[n3 + 3 * i:n3 + 3 * i + 3]
        x, y, z = state[3 * i:3 * i + 3]
        mass = system.masses[i]
        energy += mass * (vx * vx + vy * vy + vz * vz) / D(2)
        lx += mass * (y * vz - z * vy)
        ly += mass * (z * vx - x * vz)
        lz += mass * (x * vy - y * vx)
    for i in range(system.nm):
        for j in range(i + 1, system.nm):
            dx = state[3 * j] - state[3 * i]
            dy = state[3 * j + 1] - state[3 * i + 1]
            dz = state[3 * j + 2] - state[3 * i + 2]
            distance = (dx * dx + dy * dy + dz * dz).sqrt()
            energy -= G * system.masses[i] * system.masses[j] / distance
    return energy, (lx, ly, lz)


def _elements(system: NBodySystem, state: list[Decimal], i: int) -> tuple[str, str, str, str, int]:
    if i == 0:
        return "0", "NaN", "NaN", "NaN", 1
    n3 = system.nb * 3
    rx, ry, rz = (state[3 * i + k] - state[k] for k in range(3))
    vx, vy, vz = (state[n3 + 3 * i + k] - state[n3 + k] for k in range(3))
    radius = (rx * rx + ry * ry + rz * rz).sqrt()
    v2 = vx * vx + vy * vy + vz * vz
    mu = G * (system.masses[0] + system.masses[i])
    specific = v2 / D(2) - mu / radius
    bound = int(specific < 0)
    semimajor = -mu / (D(2) * specific) if bound else D(-1)
    hx, hy, hz = ry * vz - rz * vy, rz * vx - rx * vz, rx * vy - ry * vx
    hnorm = (hx * hx + hy * hy + hz * hz).sqrt()
    ex = (vy * hz - vz * hy) / mu - rx / radius
    ey = (vz * hx - vx * hz) / mu - ry / radius
    ez = (vx * hy - vy * hx) / mu - rz / radius
    eccentricity = (ex * ex + ey * ey + ez * ez).sqrt()
    perihelion = semimajor * (D(1) - eccentricity) if bound else D(-1)
    cosine = max(-1.0, min(1.0, float(hz / hnorm)))
    inclination = math.degrees(math.acos(cosine))
    return str(semimajor), str(eccentricity), str(perihelion), repr(inclination), bound


def _write_epoch(writer: csv.writer, year: int, system: NBodySystem, state: list[Decimal]) -> None:
    n3 = system.nb * 3
    for i, name in enumerate(system.names):
        elements = _elements(system, state, i)
        writer.writerow(
            (year, i, name, *[str(x) for x in state[3 * i:3 * i + 3]],
             *[str(x) for x in state[n3 + 3 * i:n3 + 3 * i + 3]], *elements)
        )


def run_reference(
    initial_state: str | Path,
    trajectory_path: str | Path,
    summary_path: str | Path,
    years: int,
    decimal_digits: int = 78,
    rtol: Decimal = D("1e-30"),
    atol: Decimal = D("1e-33"),
) -> dict:
    with localcontext(precision_context(decimal_digits)):
        system, state = load_state(initial_state)
        initial_energy, initial_l = invariants(system, state)
        trajectory = Path(trajectory_path)
        trajectory.parent.mkdir(parents=True, exist_ok=True)
        start_wall = time.perf_counter()
        current_time = D(0)
        step = D("0.05")
        with trajectory.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("time", "body", "name", "x", "y", "z", "vx", "vy", "vz", "a", "e", "q", "i_deg", "bound"))
            _write_epoch(writer, 0, system, state)
            for target_year in range(1, years + 1):
                target = D(target_year)
                while current_time < target:
                    remaining = target - current_time
                    trial_step = step
                    interval = min(trial_step, remaining)
                    clipped_to_output = interval < trial_step
                    accepted, candidate, error, level = bs_attempt(system, state, interval, atol, rtol)
                    system.stats.maximum_level = max(system.stats.maximum_level, level)
                    if accepted:
                        state = candidate
                        current_time += interval
                        system.stats.accepted_steps += 1
                        factor = 4.0 if error == 0 else max(0.5, min(2.0, 0.9 * float(error) ** (-1.0 / (2 * level + 1))))
                        # A short interval used only to land exactly on an output
                        # epoch must not permanently shrink the adaptive step.
                        # It provides no evidence that the previously accepted
                        # (larger) trial step is unsafe.
                        proposed = interval * D(str(factor))
                        step = trial_step if clipped_to_output else min(D("0.25"), proposed)
                    else:
                        system.stats.rejected_steps += 1
                        step = interval / D(2)
                        if step < D("1e-7"):
                            raise ArithmeticError("Bulirsch-Stoer step underflow")
                _write_epoch(writer, target_year, system, state)
                elapsed = time.perf_counter() - start_wall
                print(f"BS_REFERENCE year={target_year}/{years} wall={elapsed:.3f}s accepted={system.stats.accepted_steps} rejected={system.stats.rejected_steps} rhs={system.stats.rhs_calls}", flush=True)
        wall = time.perf_counter() - start_wall
        final_energy, final_l = invariants(system, state)
        energy_drift = abs((final_energy - initial_energy) / initial_energy)
        dl = tuple(final_l[i] - initial_l[i] for i in range(3))
        lnorm = sum(value * value for value in initial_l).sqrt()
        ldrift = sum(value * value for value in dl).sqrt() / lnorm
        summary = {
            "method": "independent Decimal Bulirsch-Stoer modified-midpoint extrapolation",
            "decimal_digits": decimal_digits,
            "rtol": str(rtol), "atol": str(atol), "duration_years": years,
            "wall_seconds": wall,
            "accepted_steps": system.stats.accepted_steps,
            "rejected_steps": system.stats.rejected_steps,
            "rhs_calls": system.stats.rhs_calls,
            "maximum_extrapolation_level": system.stats.maximum_level,
            "relative_energy_drift": str(energy_drift),
            "relative_angular_momentum_vector_drift": str(ldrift),
        }
        Path(summary_path).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary


def _run_block_job(arguments: tuple[str, str, str, int, int]) -> dict:
    state, trajectory, summary, years, digits = arguments
    return run_reference(state, trajectory, summary, years, digits)


def _write_subset(source_rows: list[dict[str, str]], tracer: dict[str, str], path: Path) -> None:
    chosen = source_rows[:5] + [tracer]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("index", "name", "mass", "x", "y", "z", "vx", "vy", "vz"), lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(chosen):
            item = dict(row)
            item["index"] = str(index)
            writer.writerow(item)


def _trajectory_by_name(path: Path) -> dict[tuple[int, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {(int(row["time"]), row["name"]): row for row in csv.DictReader(stream)}


def run_block_reference(
    initial_state: str | Path,
    output_dir: str | Path,
    merged_trajectory: str | Path,
    summary_path: str | Path,
    years: int = 100,
    decimal_digits: int = 78,
    workers: int = 4,
) -> dict:
    """Exploit the exact massless-tracer block structure for independent BS runs."""
    source_rows = list(csv.DictReader(Path(initial_state).open(newline="", encoding="utf-8")))
    if len(source_rows) != 20 or any(D(row["mass"]) != 0 for row in source_rows[5:]):
        raise ValueError("block decomposition requires five massive plus fifteen massless bodies")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, str, str, int, int]] = []
    job_meta: list[tuple[str, Path, Path]] = []
    for tracer in source_rows[5:]:
        name = tracer["name"]
        state_path = output / f"state_{name}.csv"
        trajectory = output / f"bs_{name}.csv"
        block_summary = output / f"bs_{name}_summary.json"
        _write_subset(source_rows, tracer, state_path)
        jobs.append((str(state_path), str(trajectory), str(block_summary), years, decimal_digits))
        job_meta.append((name, trajectory, block_summary))
    start = time.perf_counter()
    completed_summaries: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_names = {pool.submit(_run_block_job, job): meta[0] for job, meta in zip(jobs, job_meta)}
        for future in as_completed(future_names):
            name = future_names[future]
            completed_summaries[name] = future.result()
            print(f"BS_BLOCK_DONE tracer={name} blocks={len(completed_summaries)}/15 wall={time.perf_counter()-start:.3f}s", flush=True)

    blocks = {name: _trajectory_by_name(path) for name, path, _ in job_meta}
    master_name = source_rows[5]["name"]
    master = blocks[master_name]
    state_fields = ("x", "y", "z", "vx", "vy", "vz")
    with localcontext(precision_context(100)):
        massive_path_spread = D(0)
        for name, block in blocks.items():
            if name == master_name:
                continue
            for year in range(years + 1):
                for massive in ("Sun", "Jupiter", "Saturn", "Uranus", "Neptune"):
                    for field in state_fields:
                        massive_path_spread = max(
                            massive_path_spread,
                            abs(D(block[(year, massive)][field]) - D(master[(year, massive)][field])),
                        )

    merged = Path(merged_trajectory)
    merged.parent.mkdir(parents=True, exist_ok=True)
    header = ("time", "body", "name", "x", "y", "z", "vx", "vy", "vz", "a", "e", "q", "i_deg", "bound")
    with merged.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        names = [row["name"] for row in source_rows]
        for year in range(years + 1):
            for index, name in enumerate(names):
                row = dict(master[(year, name)]) if index < 5 else dict(blocks[name][(year, name)])
                row["body"] = str(index)
                writer.writerow(row)
    wall = time.perf_counter() - start
    block_invariant_pass = all(
        D(summary["relative_energy_drift"]) <= D("1e-9")
        and D(summary["relative_angular_momentum_vector_drift"]) <= D("1e-10")
        for summary in completed_summaries.values()
    )
    result = {
        "method": "independent Decimal Bulirsch-Stoer; exact massless-tracer block decomposition",
        "years": years,
        "decimal_digits": decimal_digits,
        "workers": workers,
        "wall_seconds": wall,
        "tracer_blocks": 15,
        "all_block_invariants_passed": block_invariant_pass,
        "maximum_massive_path_spread_across_blocks": str(massive_path_spread),
        "block_summaries": completed_summaries,
    }
    Path(summary_path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
