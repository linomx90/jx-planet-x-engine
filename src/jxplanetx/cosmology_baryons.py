"""Narrow dark-matter--baryon cosmology in supercomoving variables.

The module couples collisionless particle-mesh dark matter to a periodic,
adiabatic, monatomic ideal gas.  Baryons are advanced with a conservative
first-order local Lax--Friedrichs (Rusanov) finite-volume method.  Both matter
components source one spectral Poisson solve and receive the same acceleration.

For gamma = 5/3, the supercomoving gas variables obey the ordinary Euler
equations between gravitational kicks.  The supercomoving velocity is the same
canonical velocity used by :mod:`jxplanetx.cosmology_pm`, and the drift and kick
integrals therefore remain shared exactly.

This is a validation implementation.  It has no radiative cooling, chemistry,
magnetic field, star formation, feedback, adaptive mesh, or production shock
solver.  Its first-order flux is deliberately simple and diffusive.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal

import numpy as np

from . import cosmology_pm


BackendName = Literal["numpy", "cupy"]
MONATOMIC_GAMMA = 5.0 / 3.0


class CosmologyBaryonError(RuntimeError):
    """The coupled dark-matter--baryon contract was violated."""


def _backend(name: BackendName) -> Any:
    if name == "numpy":
        return np
    if name == "cupy":
        try:
            import cupy as cp
        except ImportError as exc:
            raise CosmologyBaryonError("CuPy is unavailable") from exc
        return cp
    raise CosmologyBaryonError(f"unknown array backend: {name!r}")


def _bool(value: Any) -> bool:
    item = getattr(value, "item", None)
    return bool(item() if item is not None else value)


def _finite_positive(value: float, label: str) -> float:
    if type(value) not in (float, int):
        raise TypeError(f"{label} must be a real scalar")
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return scalar


def _gamma(value: float) -> float:
    scalar = _finite_positive(value, "gamma")
    if scalar != MONATOMIC_GAMMA:
        raise ValueError("source-free supercomoving gas requires gamma == 5/3")
    return scalar


@dataclass(frozen=True)
class MatterComposition:
    """Physical-density parameters used only to form clustering fractions."""

    omega_cdm_h2: float
    omega_baryon_h2: float

    def __post_init__(self) -> None:
        _finite_positive(self.omega_cdm_h2, "omega_cdm_h2")
        _finite_positive(self.omega_baryon_h2, "omega_baryon_h2")

    @property
    def dark_matter_fraction(self) -> float:
        total = self.omega_cdm_h2 + self.omega_baryon_h2
        return self.omega_cdm_h2 / total

    @property
    def baryon_fraction(self) -> float:
        total = self.omega_cdm_h2 + self.omega_baryon_h2
        return self.omega_baryon_h2 / total

    @property
    def dark_matter_to_baryon_ratio(self) -> float:
        return self.omega_cdm_h2 / self.omega_baryon_h2


PLANCK_2018_MATTER = MatterComposition(
    omega_cdm_h2=0.1200,
    omega_baryon_h2=0.0224,
)


def conserved_from_primitive(
    density: Any,
    velocity: Any,
    pressure: Any,
    *,
    gamma: float = MONATOMIC_GAMMA,
    backend: BackendName = "numpy",
) -> Any:
    """Return ``[rho, rho*v_x, rho*v_y, rho*v_z, E]`` on a cubic mesh."""

    ratio = _gamma(gamma)
    xp = _backend(backend)
    rho = xp.asarray(density, dtype=xp.float64)
    flow = xp.asarray(velocity, dtype=xp.float64)
    gas_pressure = xp.asarray(pressure, dtype=xp.float64)
    if rho.ndim != 3 or not (rho.shape[0] == rho.shape[1] == rho.shape[2]):
        raise ValueError("density must be a cubic rank-three mesh")
    if flow.shape != (3, *rho.shape):
        raise ValueError("velocity must have shape (3, mesh, mesh, mesh)")
    if gas_pressure.shape != rho.shape:
        raise ValueError("pressure must match density")
    if (
        not _bool(xp.all(xp.isfinite(rho)))
        or not _bool(xp.all(xp.isfinite(flow)))
        or not _bool(xp.all(xp.isfinite(gas_pressure)))
        or not _bool(xp.all(rho > 0.0))
        or not _bool(xp.all(gas_pressure > 0.0))
    ):
        raise ValueError("density and pressure must be finite and positive; velocity finite")
    conserved = xp.empty((5, *rho.shape), dtype=xp.float64)
    conserved[0] = rho
    conserved[1:4] = rho[None, ...] * flow
    kinetic = 0.5 * rho * xp.sum(flow * flow, axis=0)
    conserved[4] = gas_pressure / (ratio - 1.0) + kinetic
    return conserved


def primitive_from_conserved(
    conserved: Any,
    *,
    gamma: float = MONATOMIC_GAMMA,
    backend: BackendName = "numpy",
) -> tuple[Any, Any, Any]:
    """Validate a conserved gas state and return density, velocity, pressure."""

    ratio = _gamma(gamma)
    xp = _backend(backend)
    state = xp.asarray(conserved, dtype=xp.float64)
    if (
        state.ndim != 4
        or state.shape[0] != 5
        or not (state.shape[1] == state.shape[2] == state.shape[3])
        or state.shape[1] < 4
    ):
        raise ValueError("conserved state must have shape (5, mesh, mesh, mesh)")
    if not _bool(xp.all(xp.isfinite(state))):
        raise ValueError("conserved state must be finite")
    density = state[0]
    if not _bool(xp.all(density > 0.0)):
        raise CosmologyBaryonError("nonpositive baryon density")
    velocity = state[1:4] / density[None, ...]
    kinetic = 0.5 * xp.sum(state[1:4] * state[1:4], axis=0) / density
    pressure = (ratio - 1.0) * (state[4] - kinetic)
    if not _bool(xp.all(xp.isfinite(pressure))) or not _bool(xp.all(pressure > 0.0)):
        raise CosmologyBaryonError("nonpositive baryon pressure")
    return density, velocity, pressure


def _euler_flux(state: Any, axis: int, gamma: float, xp: Any) -> tuple[Any, Any]:
    density = state[0]
    velocity = state[1:4] / density[None, ...]
    kinetic = 0.5 * xp.sum(state[1:4] * state[1:4], axis=0) / density
    pressure = (gamma - 1.0) * (state[4] - kinetic)
    if not _bool(xp.all(density > 0.0)) or not _bool(xp.all(pressure > 0.0)):
        raise CosmologyBaryonError("Rusanov flux received a nonphysical state")
    normal_velocity = velocity[axis]
    flux = xp.empty_like(state)
    flux[0] = state[axis + 1]
    flux[1:4] = state[1:4] * normal_velocity[None, ...]
    flux[axis + 1] += pressure
    flux[4] = (state[4] + pressure) * normal_velocity
    signal_speed = xp.abs(normal_velocity) + xp.sqrt(gamma * pressure / density)
    return flux, signal_speed


def maximum_cfl_timestep(
    conserved: Any,
    box_size: float = 1.0,
    *,
    cfl: float = 0.35,
    gamma: float = MONATOMIC_GAMMA,
    backend: BackendName = "numpy",
) -> float:
    """Return a conservative unsplit three-dimensional CFL timestep."""

    ratio = _gamma(gamma)
    courant = _finite_positive(cfl, "cfl")
    if courant > 0.5:
        raise ValueError("cfl must not exceed 0.5 for this validation solver")
    length = _finite_positive(box_size, "box_size")
    xp = _backend(backend)
    density, velocity, pressure = primitive_from_conserved(
        conserved, gamma=ratio, backend=backend
    )
    sound_speed = xp.sqrt(ratio * pressure / density)
    # The Rusanov flux chooses its signal bound at each face.  A cell-centred
    # maximum can underestimate the update when six different neighbours
    # supply the six face maxima.  Reproduce those exact face bounds and use
    # half their incident sum, which is the coefficient multiplying the cell
    # state in the unsplit local-Lax--Friedrichs dissipation operator.
    incident_bound = xp.zeros_like(density)
    for axis in range(3):
        cell_speed = xp.abs(velocity[axis]) + sound_speed
        positive_face = xp.maximum(
            cell_speed,
            xp.roll(cell_speed, -1, axis=axis),
        )
        negative_face = xp.roll(positive_face, 1, axis=axis)
        incident_bound += 0.5 * (positive_face + negative_face)
    maximum = float(xp.max(incident_bound).item())
    if maximum == 0.0:
        return math.inf
    return courant * (length / density.shape[0]) / maximum


def rusanov_step(
    conserved: Any,
    duration: float,
    box_size: float = 1.0,
    *,
    gamma: float = MONATOMIC_GAMMA,
    backend: BackendName = "numpy",
) -> Any:
    """Advance one unsplit periodic Rusanov step in supercomoving time."""

    ratio = _gamma(gamma)
    interval = _finite_positive(duration, "duration")
    length = _finite_positive(box_size, "box_size")
    xp = _backend(backend)
    state = xp.asarray(conserved, dtype=xp.float64)
    density, _, _ = primitive_from_conserved(state, gamma=ratio, backend=backend)
    spacing = length / density.shape[0]
    updated = state.copy()
    for axis in range(3):
        spatial_axis = axis + 1
        right = xp.roll(state, -1, axis=spatial_axis)
        left_flux, left_speed = _euler_flux(state, axis, ratio, xp)
        right_flux, right_speed = _euler_flux(right, axis, ratio, xp)
        speed = xp.maximum(left_speed, right_speed)
        face_flux = 0.5 * (left_flux + right_flux) - 0.5 * speed[None, ...] * (
            right - state
        )
        updated -= (interval / spacing) * (
            face_flux - xp.roll(face_flux, 1, axis=spatial_axis)
        )
    primitive_from_conserved(updated, gamma=ratio, backend=backend)
    return updated


def evolve_adiabatic_gas(
    conserved: Any,
    duration: float,
    box_size: float = 1.0,
    *,
    cfl: float = 0.35,
    gamma: float = MONATOMIC_GAMMA,
    maximum_substeps: int = 100000,
    backend: BackendName = "numpy",
) -> tuple[Any, int]:
    """Advance gas with CFL subcycling and return state plus substep count."""

    interval = _finite_positive(duration, "duration")
    if type(maximum_substeps) is not int or maximum_substeps < 1:
        raise ValueError("maximum_substeps must be a positive integer")
    state = conserved
    elapsed = 0.0
    substeps = 0
    while elapsed < interval:
        if substeps >= maximum_substeps:
            raise CosmologyBaryonError("hydrodynamic substep ceiling exceeded")
        allowed = maximum_cfl_timestep(
            state,
            box_size,
            cfl=cfl,
            gamma=gamma,
            backend=backend,
        )
        step = min(allowed, interval - elapsed)
        if not math.isfinite(step) or step <= 0.0:
            step = interval - elapsed
        state = rusanov_step(
            state,
            step,
            box_size,
            gamma=gamma,
            backend=backend,
        )
        elapsed += step
        substeps += 1
    return state, substeps


def total_matter_gravity(
    dark_matter_positions: Any,
    baryon_conserved: Any,
    mesh_size: int,
    composition: MatterComposition = PLANCK_2018_MATTER,
    box_size: float = 1.0,
    *,
    dark_matter_weights: Any | None = None,
    backend: BackendName = "numpy",
) -> tuple[Any, Any, Any, Any, Any]:
    """Return particle acceleration and the shared total-matter gravity fields."""

    if type(composition) is not MatterComposition:
        raise TypeError("composition must be MatterComposition")
    xp = _backend(backend)
    baryon_density, _, _ = primitive_from_conserved(
        baryon_conserved, backend=backend
    )
    if baryon_density.shape != (mesh_size, mesh_size, mesh_size):
        raise ValueError("baryon mesh and gravity mesh must match")
    # The finite-volume baryon value at index i lives at the cell centre
    # (i + 1/2) dx, whereas cosmology_pm's CIC index i lives at coordinate
    # i dx.  Translate particle coordinates by half a cell for both deposit
    # and gather so the shared array has one physical centring convention.
    # Applying the identical translation to both operations preserves CIC's
    # adjoint pairing and periodicity.
    gravity_positions = xp.mod(
        xp.asarray(dark_matter_positions, dtype=xp.float64)
        - 0.5 * float(box_size) / mesh_size,
        float(box_size),
    )
    dark_contrast = cosmology_pm.cic_density_contrast(
        gravity_positions,
        mesh_size,
        box_size,
        dark_matter_weights,
        backend,
    )
    baryon_mean = xp.mean(baryon_density, dtype=xp.float64)
    baryon_contrast = baryon_density / baryon_mean - 1.0
    total_contrast = (
        composition.dark_matter_fraction * dark_contrast
        + composition.baryon_fraction * baryon_contrast
    )
    field = cosmology_pm.spectral_acceleration(total_contrast, box_size, backend)
    particle_acceleration = cosmology_pm.cic_gather(
        field, gravity_positions, box_size, backend
    )
    return particle_acceleration, field, total_contrast, dark_contrast, baryon_contrast


def gravity_kick_baryons(
    conserved: Any,
    acceleration_field: Any,
    kick: float,
    *,
    backend: BackendName = "numpy",
) -> Any:
    """Apply a gravity impulse while preserving baryon internal energy exactly."""

    if type(kick) not in (float, int) or not math.isfinite(float(kick)):
        raise TypeError("kick must be a finite real scalar")
    xp = _backend(backend)
    state = xp.asarray(conserved, dtype=xp.float64)
    density, velocity, _ = primitive_from_conserved(state, backend=backend)
    acceleration = xp.asarray(acceleration_field, dtype=xp.float64)
    if acceleration.shape != (3, *density.shape) or not _bool(
        xp.all(xp.isfinite(acceleration))
    ):
        raise ValueError("acceleration_field must be finite and match the gas mesh")
    delta_velocity = float(kick) * acceleration
    updated = state.copy()
    updated[1:4] += density[None, ...] * delta_velocity
    updated[4] += density * (
        xp.sum(velocity * delta_velocity, axis=0)
        + 0.5 * xp.sum(delta_velocity * delta_velocity, axis=0)
    )
    primitive_from_conserved(updated, backend=backend)
    return updated


def coupled_kdk_step(
    dark_matter_positions: Any,
    dark_matter_momenta: Any,
    baryon_conserved: Any,
    a0: float,
    a1: float,
    cosmology: cosmology_pm.FlatLambdaCDM,
    mesh_size: int,
    composition: MatterComposition = PLANCK_2018_MATTER,
    box_size: float = 1.0,
    *,
    dark_matter_weights: Any | None = None,
    cfl: float = 0.35,
    backend: BackendName = "numpy",
) -> tuple[Any, Any, Any, int]:
    """Advance one shared-gravity dark-matter--baryon KDK interval."""

    if type(cosmology) is not cosmology_pm.FlatLambdaCDM:
        raise TypeError("cosmology must be FlatLambdaCDM")
    start = _finite_positive(a0, "a0")
    finish = _finite_positive(a1, "a1")
    if finish <= start:
        raise ValueError("a1 must be greater than a0")
    length = _finite_positive(box_size, "box_size")
    xp = _backend(backend)
    positions = xp.mod(xp.asarray(dark_matter_positions, dtype=xp.float64), length)
    momenta = xp.asarray(dark_matter_momenta, dtype=xp.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or momenta.shape != positions.shape:
        raise ValueError("dark-matter positions and momenta must have shape (N, 3)")
    if not _bool(xp.all(xp.isfinite(positions))) or not _bool(
        xp.all(xp.isfinite(momenta))
    ):
        raise ValueError("dark-matter state must be finite")
    primitive_from_conserved(baryon_conserved, backend=backend)

    midpoint = 0.5 * (start + finish)
    first_particle_force, first_field, _, _, _ = total_matter_gravity(
        positions,
        baryon_conserved,
        mesh_size,
        composition,
        length,
        dark_matter_weights=dark_matter_weights,
        backend=backend,
    )
    first_kick = cosmology.kick_integral(start, midpoint)
    half_momenta = momenta + first_kick * first_particle_force
    half_baryons = gravity_kick_baryons(
        baryon_conserved, first_field, first_kick, backend=backend
    )

    drift = cosmology.drift_integral(start, finish)
    new_positions = xp.mod(positions + drift * half_momenta, length)
    drifted_baryons, substeps = evolve_adiabatic_gas(
        half_baryons,
        drift,
        length,
        cfl=cfl,
        backend=backend,
    )

    second_particle_force, second_field, _, _, _ = total_matter_gravity(
        new_positions,
        drifted_baryons,
        mesh_size,
        composition,
        length,
        dark_matter_weights=dark_matter_weights,
        backend=backend,
    )
    second_kick = cosmology.kick_integral(midpoint, finish)
    new_momenta = half_momenta + second_kick * second_particle_force
    new_baryons = gravity_kick_baryons(
        drifted_baryons, second_field, second_kick, backend=backend
    )
    return new_positions, new_momenta, new_baryons, substeps


def integrate_coupled_kdk(
    dark_matter_positions: Any,
    dark_matter_momenta: Any,
    baryon_conserved: Any,
    scale_factors: Any,
    cosmology: cosmology_pm.FlatLambdaCDM,
    mesh_size: int,
    composition: MatterComposition = PLANCK_2018_MATTER,
    box_size: float = 1.0,
    *,
    dark_matter_weights: Any | None = None,
    cfl: float = 0.35,
    backend: BackendName = "numpy",
) -> tuple[Any, Any, Any, int]:
    """Advance a coupled system over increasing scale factors."""

    factors = np.asarray(scale_factors, dtype=np.float64)
    if (
        factors.ndim != 1
        or factors.size < 2
        or not np.all(np.isfinite(factors))
        or np.any(factors <= 0.0)
        or np.any(factors[1:] <= factors[:-1])
    ):
        raise ValueError("scale_factors must be finite, positive, and increasing")
    positions = dark_matter_positions
    momenta = dark_matter_momenta
    gas = baryon_conserved
    total_substeps = 0
    for start, finish in zip(factors[:-1], factors[1:]):
        positions, momenta, gas, substeps = coupled_kdk_step(
            positions,
            momenta,
            gas,
            float(start),
            float(finish),
            cosmology,
            mesh_size,
            composition,
            box_size,
            dark_matter_weights=dark_matter_weights,
            cfl=cfl,
            backend=backend,
        )
        total_substeps += substeps
    return positions, momenta, gas, total_substeps


__all__ = [
    "BackendName",
    "CosmologyBaryonError",
    "MONATOMIC_GAMMA",
    "MatterComposition",
    "PLANCK_2018_MATTER",
    "conserved_from_primitive",
    "coupled_kdk_step",
    "evolve_adiabatic_gas",
    "gravity_kick_baryons",
    "integrate_coupled_kdk",
    "maximum_cfl_timestep",
    "primitive_from_conserved",
    "rusanov_step",
    "total_matter_gravity",
]
