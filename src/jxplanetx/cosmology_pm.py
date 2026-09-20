"""Periodic particle-mesh gravity for controlled cosmology benchmarks.

This module is intentionally narrow.  It advances collisionless particles in
an expanding, flat Lambda-CDM background.  It does not implement baryonic
hydrodynamics, radiation, plasma physics, nuclear reactions, star formation,
modified gravity, or a microscopic dark-sector model.

The dimensionless canonical variables are

    p = a**2 dx / d(H0 t)

and the mesh force ``f`` is derived from ``laplacian(chi) = density_contrast``.
For matter density parameter ``Omega_m``, the split equations are

    dx/da = p / (a**3 E(a))
    dp/da = (3 Omega_m / 2) f(x) / (a**2 E(a)).

The implementation uses cloud-in-cell deposition and gathering, a periodic
spectral Poisson solve, and kick-drift-kick stepping in scale factor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterator, Literal

import numpy as np


BackendName = Literal["numpy", "cupy"]


_CUDA_SEGMENT_SUM_SOURCE = r"""
extern "C" __global__
void jx_ordered_segment_sum(
    const long long* starts,
    const long long segment_count,
    const long long value_count,
    const double* values,
    double* totals
) {
    const long long segment =
        (long long)blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (segment >= segment_count) {
        return;
    }
    const long long begin = starts[segment];
    const long long end =
        segment + 1 < segment_count ? starts[segment + 1] : value_count;
    double total = 0.0;
    for (long long index = begin; index < end; ++index) {
        total += values[index];
    }
    totals[segment] = total;
}
"""
_CUDA_SEGMENT_SUM_KERNEL: Any | None = None


class CosmologyPMError(RuntimeError):
    """The particle-mesh contract was violated."""


def _backend(name: BackendName) -> Any:
    if name == "numpy":
        return np
    if name == "cupy":
        try:
            import cupy as cp
        except ImportError as exc:
            raise CosmologyPMError("CuPy is unavailable") from exc
        return cp
    raise CosmologyPMError(f"unknown array backend: {name!r}")


def _bool(value: Any) -> bool:
    item = getattr(value, "item", None)
    return bool(item() if item is not None else value)


def _strict_positive_float(value: float, label: str) -> float:
    if type(value) not in (float, int):
        raise TypeError(f"{label} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _strict_mesh_size(value: int) -> int:
    if type(value) is not int or value < 4 or value & (value - 1):
        raise ValueError("mesh_size must be a power of two no smaller than four")
    return value


@dataclass(frozen=True)
class FlatLambdaCDM:
    """Dimensionless flat background with constant dark-energy equation of state."""

    omega_m: float
    omega_lambda: float
    omega_r: float = 0.0
    dark_energy_w: float = -1.0

    def __post_init__(self) -> None:
        values = {
            "omega_m": self.omega_m,
            "omega_lambda": self.omega_lambda,
            "omega_r": self.omega_r,
            "dark_energy_w": self.dark_energy_w,
        }
        for label, value in values.items():
            if type(value) not in (float, int) or not math.isfinite(float(value)):
                raise TypeError(f"{label} must be a finite real scalar")
        if self.omega_m <= 0.0 or self.omega_lambda < 0.0 or self.omega_r < 0.0:
            raise ValueError("density parameters must be nonnegative and omega_m positive")
        if abs((self.omega_m + self.omega_lambda + self.omega_r) - 1.0) > 1.0e-14:
            raise ValueError("FlatLambdaCDM density parameters must sum to one")

    def expansion_rate(self, scale_factor: Any) -> Any:
        """Return E(a) = H(a)/H0 for scalar or array scale factor."""

        a = np.asarray(scale_factor, dtype=np.float64)
        if not np.all(np.isfinite(a)) or np.any(a <= 0.0):
            raise ValueError("scale factor must be finite and positive")
        dark = self.omega_lambda * np.power(a, -3.0 * (1.0 + self.dark_energy_w))
        result = np.sqrt(self.omega_r / a**4 + self.omega_m / a**3 + dark)
        return float(result) if result.ndim == 0 else result

    def _integral(self, a0: float, a1: float, power: int, order: int) -> float:
        left = _strict_positive_float(a0, "a0")
        right = _strict_positive_float(a1, "a1")
        if right <= left:
            raise ValueError("a1 must be greater than a0")
        if type(order) is not int or order < 8 or order > 128:
            raise ValueError("quadrature order must be an integer in [8, 128]")
        nodes, weights = np.polynomial.legendre.leggauss(order)
        midpoint = 0.5 * (left + right)
        radius = 0.5 * (right - left)
        samples = midpoint + radius * nodes
        integrand = 1.0 / (samples**power * self.expansion_rate(samples))
        return float(radius * np.dot(weights, integrand))

    def drift_integral(self, a0: float, a1: float, order: int = 32) -> float:
        """Return the dimensionless drift integral int da / (a**3 E(a))."""

        return self._integral(a0, a1, 3, order)

    def kick_integral(self, a0: float, a1: float, order: int = 32) -> float:
        """Return (3 Omega_m / 2) int da / (a**2 E(a))."""

        return 1.5 * self.omega_m * self._integral(a0, a1, 2, order)


PLANCK_2018_BASELINE = FlatLambdaCDM(
    omega_m=0.315,
    omega_lambda=0.685,
    omega_r=0.0,
    dark_energy_w=-1.0,
)


def _positions(
    positions: Any,
    box_size: float,
    backend: BackendName,
) -> tuple[Any, Any, float]:
    xp = _backend(backend)
    length = _strict_positive_float(box_size, "box_size")
    array = xp.asarray(positions, dtype=xp.float64)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] == 0:
        raise ValueError("positions must have shape (particle_count, 3)")
    if not _bool(xp.all(xp.isfinite(array))):
        raise ValueError("positions must be finite")
    return xp.mod(array, length), xp, length


def _weights(weights: Any | None, count: int, xp: Any) -> Any:
    if weights is None:
        return xp.ones(count, dtype=xp.float64)
    array = xp.asarray(weights, dtype=xp.float64)
    if array.shape != (count,):
        raise ValueError("weights must have shape (particle_count,)")
    if not _bool(xp.all(xp.isfinite(array))) or not _bool(xp.all(array > 0.0)):
        raise ValueError("weights must be finite and positive")
    return array


def _cic_corner_data(
    positions: Any,
    mesh_size: int,
    box_size: float,
    xp: Any,
) -> Iterator[tuple[Any, Any]]:
    """Yield one CIC corner at a time to bound accelerator memory."""

    coordinates = positions * (mesh_size / box_size)
    lower = xp.floor(coordinates).astype(xp.int64)
    fraction = coordinates - lower
    for dx in (0, 1):
        wx = fraction[:, 0] if dx else 1.0 - fraction[:, 0]
        ix = (lower[:, 0] + dx) % mesh_size
        for dy in (0, 1):
            wy = fraction[:, 1] if dy else 1.0 - fraction[:, 1]
            iy = (lower[:, 1] + dy) % mesh_size
            for dz in (0, 1):
                wz = fraction[:, 2] if dz else 1.0 - fraction[:, 2]
                iz = (lower[:, 2] + dz) % mesh_size
                flat_index = (ix * mesh_size + iy) * mesh_size + iz
                yield flat_index, wx * wy * wz


def _ordered_segment_sum(indices: Any, values: Any, size: int, xp: Any) -> Any:
    """Aggregate values by integer index after imposing a total input order."""

    sequence = xp.arange(indices.size, dtype=xp.int64)
    keys = (sequence, indices) if xp is np else xp.stack((sequence, indices), axis=0)
    order = xp.lexsort(keys)
    sorted_indices = indices[order]
    sorted_values = values[order]
    starts = xp.concatenate(
        (
            xp.asarray([0], dtype=xp.int64),
            xp.nonzero(sorted_indices[1:] != sorted_indices[:-1])[0].astype(xp.int64) + 1,
        )
    )
    unique_indices = sorted_indices[starts]
    if xp is np:
        totals = xp.add.reduceat(sorted_values, starts)
    else:
        global _CUDA_SEGMENT_SUM_KERNEL
        if _CUDA_SEGMENT_SUM_KERNEL is None:
            _CUDA_SEGMENT_SUM_KERNEL = xp.RawKernel(
                _CUDA_SEGMENT_SUM_SOURCE,
                "jx_ordered_segment_sum",
                options=("--std=c++11",),
            )
        totals = xp.empty(starts.size, dtype=xp.float64)
        threads = 256
        blocks = (int(starts.size) + threads - 1) // threads
        _CUDA_SEGMENT_SUM_KERNEL(
            (blocks,),
            (threads,),
            (
                starts,
                np.int64(starts.size),
                np.int64(sorted_values.size),
                sorted_values,
                totals,
            ),
        )
    result = xp.zeros(size, dtype=xp.float64)
    result[unique_indices] = totals
    return result


def cic_density_contrast(
    positions: Any,
    mesh_size: int,
    box_size: float = 1.0,
    weights: Any | None = None,
    backend: BackendName = "numpy",
) -> Any:
    """Deposit particles by CIC and return a periodic density-contrast mesh."""

    mesh = _strict_mesh_size(mesh_size)
    wrapped, xp, length = _positions(positions, box_size, backend)
    particle_weights = _weights(weights, wrapped.shape[0], xp)
    mass_flat = xp.zeros(mesh**3, dtype=xp.float64)
    for indices, coefficient in _cic_corner_data(wrapped, mesh, length, xp):
        contributions = particle_weights * coefficient
        mass_flat += _ordered_segment_sum(indices, contributions, mesh**3, xp)
    mass_grid = mass_flat.reshape(mesh, mesh, mesh)
    total_mass = xp.sum(particle_weights, dtype=xp.float64)
    mean_cell_mass = total_mass / float(mesh**3)
    contrast = mass_grid / mean_cell_mass - 1.0
    return contrast


def spectral_acceleration(
    density_contrast: Any,
    box_size: float = 1.0,
    backend: BackendName = "numpy",
) -> Any:
    """Solve laplacian(chi)=delta and return -gradient(chi) spectrally."""

    xp = _backend(backend)
    length = _strict_positive_float(box_size, "box_size")
    density = xp.asarray(density_contrast, dtype=xp.float64)
    if density.ndim != 3 or not (density.shape[0] == density.shape[1] == density.shape[2]):
        raise ValueError("density_contrast must be a cubic rank-three mesh")
    mesh = _strict_mesh_size(int(density.shape[0]))
    if not _bool(xp.all(xp.isfinite(density))):
        raise ValueError("density_contrast must be finite")

    density = density - xp.mean(density, dtype=xp.float64)
    transform_axes = (0, 1, 2)
    spectrum = xp.fft.rfftn(density, axes=transform_axes)
    spacing = length / mesh
    kx = (2.0 * math.pi * xp.fft.fftfreq(mesh, d=spacing)).reshape(mesh, 1, 1)
    ky = (2.0 * math.pi * xp.fft.fftfreq(mesh, d=spacing)).reshape(1, mesh, 1)
    kz = (2.0 * math.pi * xp.fft.rfftfreq(mesh, d=spacing)).reshape(1, 1, mesh // 2 + 1)
    k_squared = kx * kx + ky * ky + kz * kz
    inverse_k_squared = xp.zeros_like(k_squared, dtype=xp.float64)
    inverse_k_squared[k_squared > 0.0] = 1.0 / k_squared[k_squared > 0.0]

    fields = []
    for axis, wave_number in enumerate((kx, ky, kz)):
        acceleration_spectrum = 1j * wave_number * spectrum * inverse_k_squared
        if axis == 0:
            acceleration_spectrum[mesh // 2, :, :] = 0.0
        elif axis == 1:
            acceleration_spectrum[:, mesh // 2, :] = 0.0
        else:
            acceleration_spectrum[:, :, mesh // 2] = 0.0
        fields.append(
            xp.fft.irfftn(
                acceleration_spectrum,
                s=density.shape,
                axes=transform_axes,
            ).real
        )
    return xp.stack(fields, axis=0)


def cic_gather(
    vector_field: Any,
    positions: Any,
    box_size: float = 1.0,
    backend: BackendName = "numpy",
) -> Any:
    """Interpolate a three-component periodic mesh field to particle positions."""

    wrapped, xp, length = _positions(positions, box_size, backend)
    field = xp.asarray(vector_field, dtype=xp.float64)
    if field.ndim != 4 or field.shape[0] != 3:
        raise ValueError("vector_field must have shape (3, mesh, mesh, mesh)")
    mesh = _strict_mesh_size(int(field.shape[1]))
    if field.shape[1:] != (mesh, mesh, mesh):
        raise ValueError("vector_field must be cubic")
    if not _bool(xp.all(xp.isfinite(field))):
        raise ValueError("vector_field must be finite")
    result = xp.zeros_like(wrapped)
    for flat_index, coefficient in _cic_corner_data(wrapped, mesh, length, xp):
        ix = flat_index // (mesh * mesh)
        remainder = flat_index % (mesh * mesh)
        iy = remainder // mesh
        iz = remainder % mesh
        result += coefficient[:, None] * field[:, ix, iy, iz].T
    return result


def particle_mesh_acceleration(
    positions: Any,
    mesh_size: int,
    box_size: float = 1.0,
    weights: Any | None = None,
    backend: BackendName = "numpy",
) -> tuple[Any, Any, Any]:
    """Return particle acceleration, density contrast, and acceleration mesh."""

    density = cic_density_contrast(positions, mesh_size, box_size, weights, backend)
    field = spectral_acceleration(density, box_size, backend)
    acceleration = cic_gather(field, positions, box_size, backend)
    return acceleration, density, field


def kdk_step(
    positions: Any,
    canonical_momenta: Any,
    a0: float,
    a1: float,
    cosmology: FlatLambdaCDM,
    mesh_size: int,
    box_size: float = 1.0,
    weights: Any | None = None,
    backend: BackendName = "numpy",
) -> tuple[Any, Any]:
    """Advance one second-order kick-drift-kick step in scale factor."""

    if type(cosmology) is not FlatLambdaCDM:
        raise TypeError("cosmology must be FlatLambdaCDM")
    start = _strict_positive_float(a0, "a0")
    finish = _strict_positive_float(a1, "a1")
    if finish <= start:
        raise ValueError("a1 must be greater than a0")
    wrapped, xp, length = _positions(positions, box_size, backend)
    momenta = xp.asarray(canonical_momenta, dtype=xp.float64)
    if momenta.shape != wrapped.shape or not _bool(xp.all(xp.isfinite(momenta))):
        raise ValueError("canonical_momenta must be finite and match positions")

    middle = 0.5 * (start + finish)
    first_force, _, _ = particle_mesh_acceleration(
        wrapped, mesh_size, length, weights, backend
    )
    half_momenta = momenta + cosmology.kick_integral(start, middle) * first_force
    new_positions = xp.mod(
        wrapped + cosmology.drift_integral(start, finish) * half_momenta,
        length,
    )
    second_force, _, _ = particle_mesh_acceleration(
        new_positions, mesh_size, length, weights, backend
    )
    new_momenta = half_momenta + cosmology.kick_integral(middle, finish) * second_force
    return new_positions, new_momenta


def integrate_kdk(
    positions: Any,
    canonical_momenta: Any,
    scale_factors: Any,
    cosmology: FlatLambdaCDM,
    mesh_size: int,
    box_size: float = 1.0,
    weights: Any | None = None,
    backend: BackendName = "numpy",
) -> tuple[Any, Any]:
    """Advance over a strictly increasing sequence of scale factors."""

    factors = np.asarray(scale_factors, dtype=np.float64)
    if (
        factors.ndim != 1
        or factors.size < 2
        or not np.all(np.isfinite(factors))
        or np.any(factors <= 0.0)
        or np.any(factors[1:] <= factors[:-1])
    ):
        raise ValueError("scale_factors must be finite, positive, and strictly increasing")
    current_positions = positions
    current_momenta = canonical_momenta
    for start, finish in zip(factors[:-1], factors[1:]):
        current_positions, current_momenta = kdk_step(
            current_positions,
            current_momenta,
            float(start),
            float(finish),
            cosmology,
            mesh_size,
            box_size,
            weights,
            backend,
        )
    return current_positions, current_momenta


__all__ = [
    "BackendName",
    "CosmologyPMError",
    "FlatLambdaCDM",
    "PLANCK_2018_BASELINE",
    "cic_density_contrast",
    "cic_gather",
    "integrate_kdk",
    "kdk_step",
    "particle_mesh_acceleration",
    "spectral_acceleration",
]
