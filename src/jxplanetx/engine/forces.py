"""Backend-neutral force kernels used by the JX evaluation runtime.

These functions evaluate accelerations only.  They do not advance time,
mutate a state snapshot, write trajectories, or claim that a model is
qualified.  NumPy and CuPy follow the same array expressions wherever their
public APIs permit it.
"""

from __future__ import annotations

import math
from typing import Any

from .backends import ArrayBackend


NEWTONIAN_POINT_MASS_MODEL_ID = "force.newtonian.point_mass"
RESTRICTED_STATIC_CENTRAL_1PN_MODEL_ID = (
    "relativity.solar_schwarzschild_test_particle_1pn"
)
MUTUAL_EIH_1PN_MODEL_ID = "force.relativity.eih_1pn_gr"
CANNONBALL_SRP_MODEL_ID = "force.nongrav.srp_cannonball"

SINGULARITY_POLICY_ERROR = "error"
COLLISION_POLICY_ERROR = "error"


class ForceError(ValueError):
    """Base class for force-evaluation failures."""


class ForceContractError(ForceError):
    """A force request is ambiguous or internally inconsistent."""


class ForceDomainError(ForceError):
    """A validated request lies outside a model's numerical domain."""


class ForceSingularityError(ForceDomainError):
    """A selected source and target occupy the same position."""


class ForceCollisionError(ForceDomainError):
    """Selected finite-radius bodies overlap or touch."""


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ForceContractError(f"{label} must be a positive integer")
    return value


def _finite_positive_scalar(value: object, label: str) -> float:
    if type(value) not in (float, int) or type(value) is bool:
        raise ForceContractError(f"{label} must be a finite numeric scalar")
    checked = float(value)
    if checked != checked or checked in (float("inf"), float("-inf")):
        raise ForceDomainError(f"{label} must be finite")
    if checked <= 0.0:
        raise ForceDomainError(f"{label} must be strictly positive")
    return checked


def _selected_indices(backend: ArrayBackend, mask: Any) -> Any:
    # ``nonzero`` returns a one-tuple consistently in NumPy and CuPy.
    return backend.xp.nonzero(mask)[0]


def _has_any(backend: ArrayBackend, expression: Any, label: str) -> bool:
    return backend.scalar_bool(backend.xp.any(expression), label)


def _validate_common_arrays(
    backend: ArrayBackend,
    positions: Any,
    gravitational_parameters: Any,
    radii: Any,
    source_mask: Any,
    target_mask: Any,
    *,
    require_positive_source_gm: bool,
) -> int:
    positions = backend.require_native_array(positions, "positions")
    gravitational_parameters = backend.require_native_array(
        gravitational_parameters, "gravitational_parameters"
    )
    radii = backend.require_native_array(radii, "radii")
    source_mask = backend.require_native_array(source_mask, "source_mask")
    target_mask = backend.require_native_array(target_mask, "target_mask")
    if positions.dtype != backend.float64:
        raise ForceContractError("positions must have dtype float64")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ForceContractError("positions must have shape (body_count, 3)")
    body_count = positions.shape[0]
    if gravitational_parameters.dtype != backend.float64:
        raise ForceContractError("gravitational_parameters must have dtype float64")
    if gravitational_parameters.ndim != 1 or gravitational_parameters.shape != (body_count,):
        raise ForceContractError(
            "gravitational_parameters must have shape (body_count,)"
        )
    if radii.dtype != backend.float64:
        raise ForceContractError("radii must have dtype float64")
    if radii.ndim != 1 or radii.shape != (body_count,):
        raise ForceContractError("radii must have shape (body_count,)")
    for label, mask in (("source_mask", source_mask), ("target_mask", target_mask)):
        if mask.dtype != backend.bool_:
            raise ForceContractError(f"{label} must have boolean dtype")
        if mask.ndim != 1 or mask.shape != (body_count,):
            raise ForceContractError(f"{label} must have shape (body_count,)")
    if body_count == 0:
        raise ForceContractError("a force evaluation requires at least one body")
    if not _has_any(backend, source_mask, "source mask population check"):
        raise ForceContractError("source_mask must select at least one source")
    if not _has_any(backend, target_mask, "target mask population check"):
        raise ForceContractError("target_mask must select at least one target")
    if _has_any(
        backend,
        ~backend.xp.isfinite(positions),
        "position finiteness check",
    ):
        raise ForceDomainError("positions must contain only finite values")
    if _has_any(
        backend,
        ~backend.xp.isfinite(gravitational_parameters),
        "gravitational-parameter finiteness check",
    ):
        raise ForceDomainError("gravitational_parameters must contain only finite values")
    if _has_any(
        backend,
        gravitational_parameters < 0.0,
        "gravitational-parameter sign check",
    ):
        raise ForceDomainError("gravitational_parameters cannot be negative")
    if _has_any(backend, ~backend.xp.isfinite(radii), "radius finiteness check"):
        raise ForceDomainError("radii must contain only finite values")
    if _has_any(backend, radii < 0.0, "radius sign check"):
        raise ForceDomainError("radii cannot be negative")
    if require_positive_source_gm and _has_any(
        backend,
        source_mask & (gravitational_parameters <= 0.0),
        "active-source gravitational-parameter check",
    ):
        raise ForceDomainError("every selected gravitational source must have positive GM")
    return body_count


def newtonian_point_mass_acceleration(
    *,
    backend: ArrayBackend,
    positions: Any,
    gravitational_parameters: Any,
    radii: Any,
    source_mask: Any,
    target_mask: Any,
    tile_size: int,
    singularity_policy: str,
    collision_policy: str,
) -> Any:
    """Evaluate tiled all-pairs point-mass gravity in float64.

    Selected sources exert gravity.  Bodies not selected as sources have no
    backreaction and therefore serve as massless targets when selected by
    ``target_mask``.  A source may also be a target; its diagonal self-pair is
    excluded by identity, while any distinct coincident pair is an error.
    """

    if singularity_policy != SINGULARITY_POLICY_ERROR:
        raise ForceContractError(
            "the point-mass kernel supports only singularity_policy='error'; "
            "softening is a different physical model"
        )
    if collision_policy != COLLISION_POLICY_ERROR:
        raise ForceContractError("the point-mass kernel supports only collision_policy='error'")
    checked_tile_size = _positive_int(tile_size, "tile_size")
    body_count = _validate_common_arrays(
        backend,
        positions,
        gravitational_parameters,
        radii,
        source_mask,
        target_mask,
        require_positive_source_gm=True,
    )
    xp = backend.xp
    result = xp.zeros((body_count, 3), dtype=xp.float64)
    source_indices = _selected_indices(backend, source_mask)
    target_indices = _selected_indices(backend, target_mask)

    source_count = int(source_indices.shape[0])
    target_count = int(target_indices.shape[0])
    one = xp.float64(1.0)
    zero = xp.float64(0.0)

    for target_start in range(0, target_count, checked_tile_size):
        target_tile = target_indices[target_start : target_start + checked_tile_size]
        target_positions = positions[target_tile]
        tile_acceleration = xp.zeros((target_tile.shape[0], 3), dtype=xp.float64)
        for source_start in range(0, source_count, checked_tile_size):
            source_tile = source_indices[source_start : source_start + checked_tile_size]
            displacement = (
                positions[source_tile][xp.newaxis, :, :]
                - target_positions[:, xp.newaxis, :]
            )
            distance_squared = xp.sum(displacement * displacement, axis=2)
            self_pair = target_tile[:, xp.newaxis] == source_tile[xp.newaxis, :]
            singular = (distance_squared == zero) & ~self_pair
            if _has_any(backend, singular, "Newtonian singularity check"):
                raise ForceSingularityError(
                    "distinct selected Newtonian source and target positions coincide"
                )
            collision_distance = (
                radii[target_tile][:, xp.newaxis]
                + radii[source_tile][xp.newaxis, :]
            )
            collision = (
                (distance_squared <= collision_distance * collision_distance)
                & (distance_squared > zero)
                & ~self_pair
            )
            if _has_any(backend, collision, "Newtonian collision check"):
                raise ForceCollisionError(
                    "selected finite-radius Newtonian source and target bodies overlap or touch"
                )
            excluded = self_pair
            safe_distance_squared = xp.where(excluded, one, distance_squared)
            inverse_distance_cubed = one / (
                safe_distance_squared * xp.sqrt(safe_distance_squared)
            )
            weights = xp.where(
                excluded,
                zero,
                gravitational_parameters[source_tile][xp.newaxis, :]
                * inverse_distance_cubed,
            )
            tile_acceleration += xp.sum(
                displacement * weights[:, :, xp.newaxis], axis=1
            )
        result[target_tile] = tile_acceleration
    return result


def restricted_static_central_1pn_acceleration(
    *,
    backend: ArrayBackend,
    positions: Any,
    velocities: Any,
    gravitational_parameters: Any,
    radii: Any,
    source_mask: Any,
    target_mask: Any,
    central_index: int,
    speed_of_light: object,
    maximum_compactness: object,
    maximum_speed_fraction_squared: object,
    singularity_policy: str,
    collision_policy: str,
) -> Any:
    """Evaluate the restricted static-central Schwarzschild 1PN correction.

    The result is correction-only.  Targets must be massless/no-backreaction,
    the central source must be selected and exactly static, and the supplied
    weak-field and slow-motion bounds are enforced for every selected target.
    """

    if singularity_policy != SINGULARITY_POLICY_ERROR:
        raise ForceContractError("the restricted 1PN kernel requires singularity_policy='error'")
    if collision_policy != COLLISION_POLICY_ERROR:
        raise ForceContractError("the restricted 1PN kernel requires collision_policy='error'")
    body_count = _validate_common_arrays(
        backend,
        positions,
        gravitational_parameters,
        radii,
        source_mask,
        target_mask,
        require_positive_source_gm=True,
    )
    velocities = backend.require_native_array(velocities, "velocities")
    if velocities.dtype != backend.float64:
        raise ForceContractError("velocities must have dtype float64")
    if velocities.shape != (body_count, 3):
        raise ForceContractError("velocities must have shape (body_count, 3)")
    if _has_any(backend, ~backend.xp.isfinite(velocities), "velocity finiteness check"):
        raise ForceDomainError("velocities must contain only finite values")
    if type(central_index) is not int or not 0 <= central_index < body_count:
        raise ForceContractError("central_index is outside the state body range")
    c = _finite_positive_scalar(speed_of_light, "speed_of_light")
    compactness_limit = _finite_positive_scalar(
        maximum_compactness, "maximum_compactness"
    )
    speed_limit = _finite_positive_scalar(
        maximum_speed_fraction_squared, "maximum_speed_fraction_squared"
    )
    if compactness_limit >= 1.0:
        raise ForceDomainError("maximum_compactness must be less than one")
    if speed_limit >= 1.0:
        raise ForceDomainError("maximum_speed_fraction_squared must be less than one")

    xp = backend.xp
    if not backend.scalar_bool(source_mask[central_index], "central source-mask check"):
        raise ForceContractError("the restricted 1PN central body must be a selected source")
    if backend.scalar_bool(target_mask[central_index], "central target-mask check"):
        raise ForceContractError("the restricted 1PN central body cannot be a target")
    if _has_any(
        backend,
        target_mask & source_mask,
        "restricted 1PN massless-target check",
    ):
        raise ForceContractError(
            "restricted 1PN targets must be massless/no-backreaction bodies"
        )
    if _has_any(
        backend,
        velocities[central_index] != xp.float64(0.0),
        "static-central velocity check",
    ):
        raise ForceDomainError("the restricted 1PN central source must be exactly static")

    target_indices = _selected_indices(backend, target_mask)
    relative_position = positions[target_indices] - positions[central_index]
    relative_velocity = velocities[target_indices] - velocities[central_index]
    radius_squared = xp.sum(relative_position * relative_position, axis=1)
    if _has_any(
        backend,
        radius_squared == xp.float64(0.0),
        "restricted 1PN singularity check",
    ):
        raise ForceSingularityError("a restricted 1PN target coincides with its central source")
    collision_distance = radii[target_indices] + radii[central_index]
    if _has_any(
        backend,
        (radius_squared <= collision_distance * collision_distance)
        & (radius_squared > xp.float64(0.0)),
        "restricted 1PN collision check",
    ):
        raise ForceCollisionError(
            "a restricted 1PN target overlaps or touches the finite-radius central source"
        )

    radius = xp.sqrt(radius_squared)
    speed_squared = xp.sum(relative_velocity * relative_velocity, axis=1)
    mu = gravitational_parameters[central_index]
    c_squared = xp.float64(c) * xp.float64(c)
    compactness = mu / (radius * c_squared)
    speed_fraction_squared = speed_squared / c_squared
    if _has_any(
        backend,
        compactness > xp.float64(compactness_limit),
        "restricted 1PN compactness check",
    ):
        raise ForceDomainError("a target exceeds the declared 1PN weak-field bound")
    if _has_any(
        backend,
        speed_fraction_squared > xp.float64(speed_limit),
        "restricted 1PN slow-motion check",
    ):
        raise ForceDomainError("a target exceeds the declared 1PN slow-motion bound")

    radial_velocity_product = xp.sum(relative_position * relative_velocity, axis=1)
    radial_coefficient = xp.float64(4.0) * mu / radius - speed_squared
    common = mu / (c_squared * radius_squared * radius)
    correction = common[:, xp.newaxis] * (
        radial_coefficient[:, xp.newaxis] * relative_position
        + xp.float64(4.0)
        * radial_velocity_product[:, xp.newaxis]
        * relative_velocity
    )
    result = xp.zeros((body_count, 3), dtype=xp.float64)
    result[target_indices] = correction
    return result


def mutual_eih_1pn_acceleration(
    *,
    backend: ArrayBackend,
    positions: Any,
    velocities: Any,
    gravitational_parameters: Any,
    radii: Any,
    body_mask: Any,
    tile_size: int,
    speed_of_light: object,
    maximum_compactness: object,
    maximum_speed_fraction_squared: object,
    singularity_policy: str,
    collision_policy: str,
) -> Any:
    """Evaluate the correction-only mutual point-mass EIH 1PN acceleration.

    The expression is equation 27 of the JPL DE440/DE441 description with
    beta=gamma=1. Acceleration-dependent source terms use simultaneous
    Newtonian accelerations, which is equivalent through retained 1PN order.
    All bodies must participate as massive mutual sources and targets.
    """

    if singularity_policy != SINGULARITY_POLICY_ERROR:
        raise ForceContractError("the mutual EIH 1PN kernel requires singularity_policy='error'")
    if collision_policy != COLLISION_POLICY_ERROR:
        raise ForceContractError("the mutual EIH 1PN kernel requires collision_policy='error'")
    body_count = _validate_common_arrays(
        backend,
        positions,
        gravitational_parameters,
        radii,
        body_mask,
        body_mask,
        require_positive_source_gm=True,
    )
    velocities = backend.require_native_array(velocities, "velocities")
    if velocities.dtype != backend.float64:
        raise ForceContractError("velocities must have dtype float64")
    if velocities.shape != (body_count, 3):
        raise ForceContractError("velocities must have shape (body_count, 3)")
    if _has_any(backend, ~backend.xp.isfinite(velocities), "velocity finiteness check"):
        raise ForceDomainError("velocities must contain only finite values")
    if body_count < 2:
        raise ForceContractError("mutual EIH 1PN requires at least two bodies")
    if _has_any(backend, ~body_mask, "mutual EIH body-mask completeness check"):
        raise ForceContractError("mutual EIH 1PN requires every state body")

    c = _finite_positive_scalar(speed_of_light, "speed_of_light")
    compactness_limit = _finite_positive_scalar(
        maximum_compactness, "maximum_compactness"
    )
    speed_limit = _finite_positive_scalar(
        maximum_speed_fraction_squared, "maximum_speed_fraction_squared"
    )
    if compactness_limit >= 1.0:
        raise ForceDomainError("maximum_compactness must be less than one")
    if speed_limit >= 1.0:
        raise ForceDomainError("maximum_speed_fraction_squared must be less than one")
    if not math.isfinite(c * c):
        raise ForceDomainError("speed_of_light squared must be finite")

    xp = backend.xp
    one = xp.float64(1.0)
    zero = xp.float64(0.0)
    c_squared = xp.float64(c) * xp.float64(c)
    diagonal = xp.eye(body_count, dtype=xp.bool_)
    difference = positions[:, xp.newaxis, :] - positions[xp.newaxis, :, :]
    distance_squared = xp.sum(difference * difference, axis=2)
    if _has_any(
        backend,
        (distance_squared == zero) & ~diagonal,
        "mutual EIH singularity check",
    ):
        raise ForceSingularityError("distinct mutual EIH point masses coincide")
    collision_distance = radii[:, xp.newaxis] + radii[xp.newaxis, :]
    if _has_any(
        backend,
        (distance_squared <= collision_distance * collision_distance)
        & (distance_squared > zero)
        & ~diagonal,
        "mutual EIH collision check",
    ):
        raise ForceCollisionError("mutual EIH finite-radius bodies overlap or touch")

    safe_distance_squared = xp.where(diagonal, one, distance_squared)
    distance = xp.sqrt(safe_distance_squared)
    inverse_distance = one / distance
    inverse_distance_cubed = inverse_distance / safe_distance_squared
    pair_weight = xp.where(
        diagonal,
        zero,
        gravitational_parameters[xp.newaxis, :] * inverse_distance,
    )
    potentials = xp.sum(pair_weight, axis=1)
    speed_squared = xp.sum(velocities * velocities, axis=1)
    if _has_any(
        backend,
        potentials / c_squared > xp.float64(compactness_limit),
        "mutual EIH compactness check",
    ):
        raise ForceDomainError("a body exceeds the declared EIH 1PN weak-field bound")
    if _has_any(
        backend,
        speed_squared / c_squared > xp.float64(speed_limit),
        "mutual EIH slow-motion check",
    ):
        raise ForceDomainError("a body exceeds the declared EIH 1PN slow-motion bound")

    newtonian = newtonian_point_mass_acceleration(
        backend=backend,
        positions=positions,
        gravitational_parameters=gravitational_parameters,
        radii=radii,
        source_mask=body_mask,
        target_mask=body_mask,
        tile_size=tile_size,
        singularity_policy=singularity_policy,
        collision_policy=collision_policy,
    )
    velocity_body = velocities[:, xp.newaxis, :]
    velocity_source = velocities[xp.newaxis, :, :]
    radial_source_velocity = xp.sum(difference * velocity_source, axis=2)
    bracket = (
        xp.float64(4.0) * potentials[:, xp.newaxis]
        + potentials[xp.newaxis, :]
        - speed_squared[:, xp.newaxis]
        - xp.float64(2.0) * speed_squared[xp.newaxis, :]
        + xp.float64(4.0) * xp.sum(velocity_body * velocity_source, axis=2)
        + xp.float64(1.5)
        * radial_source_velocity
        * radial_source_velocity
        / safe_distance_squared
        + xp.float64(0.5)
        * xp.sum(difference * newtonian[xp.newaxis, :, :], axis=2)
    ) / c_squared
    velocity_bracket = xp.sum(
        difference
        * (xp.float64(4.0) * velocity_body - xp.float64(3.0) * velocity_source),
        axis=2,
    )
    pair_correction = gravitational_parameters[xp.newaxis, :, xp.newaxis] * (
        difference * (bracket * inverse_distance_cubed)[:, :, xp.newaxis]
        + (
            velocity_bracket[:, :, xp.newaxis]
            * (velocity_body - velocity_source)
            * inverse_distance_cubed[:, :, xp.newaxis]
            + xp.float64(3.5)
            * newtonian[xp.newaxis, :, :]
            * inverse_distance[:, :, xp.newaxis]
        )
        / c_squared
    )
    pair_correction = xp.where(
        diagonal[:, :, xp.newaxis],
        zero,
        pair_correction,
    )
    return xp.sum(pair_correction, axis=1)


def cannonball_srp_acceleration(
    *,
    backend: ArrayBackend,
    positions: Any,
    gravitational_parameters: Any,
    radii: Any,
    source_mask: Any,
    target_mask: Any,
    radiation_source_index: int,
    reference_pressure: object,
    reference_distance: object,
    area_to_mass: Any,
    radiation_pressure_coefficient: Any,
    singularity_policy: str,
    collision_policy: str,
) -> Any:
    """Evaluate unshadowed radial cannonball solar-radiation pressure.

    ``reference_pressure`` is force per area at ``reference_distance``.
    ``area_to_mass`` and ``radiation_pressure_coefficient`` are per-body
    float64 arrays.  The acceleration points away from the radiation source.
    No eclipse, penumbra, attitude, absorption, or thermal-recoil model is
    hidden in this term.
    """

    if singularity_policy != SINGULARITY_POLICY_ERROR:
        raise ForceContractError("the cannonball SRP kernel requires singularity_policy='error'")
    if collision_policy != COLLISION_POLICY_ERROR:
        raise ForceContractError("the cannonball SRP kernel requires collision_policy='error'")
    body_count = _validate_common_arrays(
        backend,
        positions,
        gravitational_parameters,
        radii,
        source_mask,
        target_mask,
        require_positive_source_gm=False,
    )
    if type(radiation_source_index) is not int or not 0 <= radiation_source_index < body_count:
        raise ForceContractError("radiation_source_index is outside the state body range")
    pressure = _finite_positive_scalar(reference_pressure, "reference_pressure")
    distance_reference = _finite_positive_scalar(reference_distance, "reference_distance")
    area_to_mass = backend.require_native_array(area_to_mass, "area_to_mass")
    radiation_pressure_coefficient = backend.require_native_array(
        radiation_pressure_coefficient, "radiation_pressure_coefficient"
    )
    for label, values in (
        ("area_to_mass", area_to_mass),
        ("radiation_pressure_coefficient", radiation_pressure_coefficient),
    ):
        if values.dtype != backend.float64:
            raise ForceContractError(f"{label} must have dtype float64")
        if values.ndim != 1 or values.shape != (body_count,):
            raise ForceContractError(f"{label} must have shape (body_count,)")
        if _has_any(backend, ~backend.xp.isfinite(values), f"{label} finiteness check"):
            raise ForceDomainError(f"{label} must contain only finite values")
    xp = backend.xp
    if backend.scalar_bool(target_mask[radiation_source_index], "SRP source target-mask check"):
        raise ForceContractError("the radiation source cannot be an SRP target")
    if _has_any(
        backend,
        target_mask & source_mask,
        "SRP massless-target check",
    ):
        raise ForceContractError("cannonball SRP targets must be massless/no-backreaction bodies")
    if _has_any(
        backend,
        target_mask & (area_to_mass < xp.float64(0.0)),
        "SRP area-to-mass domain check",
    ):
        raise ForceDomainError("selected SRP targets require nonnegative area_to_mass")
    if _has_any(
        backend,
        target_mask & (radiation_pressure_coefficient < xp.float64(0.0)),
        "SRP coefficient domain check",
    ):
        raise ForceDomainError(
            "selected SRP targets require nonnegative radiation_pressure_coefficient"
        )

    target_indices = _selected_indices(backend, target_mask)
    displacement = positions[target_indices] - positions[radiation_source_index]
    radius_squared = xp.sum(displacement * displacement, axis=1)
    if _has_any(
        backend,
        radius_squared == xp.float64(0.0),
        "SRP singularity check",
    ):
        raise ForceSingularityError("an SRP target coincides with its radiation source")
    collision_distance = radii[target_indices] + radii[radiation_source_index]
    if _has_any(
        backend,
        (radius_squared <= collision_distance * collision_distance)
        & (radius_squared > xp.float64(0.0)),
        "SRP collision check",
    ):
        raise ForceCollisionError(
            "an SRP target overlaps or touches the finite-radius radiation source"
        )
    radius = xp.sqrt(radius_squared)
    magnitude = (
        xp.float64(pressure)
        * radiation_pressure_coefficient[target_indices]
        * area_to_mass[target_indices]
        * (xp.float64(distance_reference) * xp.float64(distance_reference))
        / radius_squared
    )
    acceleration = magnitude[:, xp.newaxis] * displacement / radius[:, xp.newaxis]
    result = xp.zeros((body_count, 3), dtype=xp.float64)
    result[target_indices] = acceleration
    return result


__all__ = [
    "CANNONBALL_SRP_MODEL_ID",
    "COLLISION_POLICY_ERROR",
    "ForceCollisionError",
    "ForceContractError",
    "ForceDomainError",
    "ForceError",
    "ForceSingularityError",
    "MUTUAL_EIH_1PN_MODEL_ID",
    "NEWTONIAN_POINT_MASS_MODEL_ID",
    "RESTRICTED_STATIC_CENTRAL_1PN_MODEL_ID",
    "SINGULARITY_POLICY_ERROR",
    "cannonball_srp_acceleration",
    "newtonian_point_mass_acceleration",
    "mutual_eih_1pn_acceleration",
    "restricted_static_central_1pn_acceleration",
]
