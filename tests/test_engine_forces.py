import unittest
from decimal import Decimal, localcontext

import numpy as np

from jxplanetx.engine.backends import resolve_backend
from jxplanetx.engine.forces import (
    COLLISION_POLICY_ERROR,
    SINGULARITY_POLICY_ERROR,
    ForceCollisionError,
    ForceContractError,
    ForceDomainError,
    ForceSingularityError,
    cannonball_srp_acceleration,
    mutual_eih_1pn_acceleration,
    newtonian_point_mass_acceleration,
    restricted_static_central_1pn_acceleration,
)
from jxplanetx.solar_system.eih_1pn import (
    EIH1PNParameters,
    evaluate_eih_1pn_correction,
)


BACKEND = resolve_backend("numpy")
F64 = np.float64


def newtonian(
    positions: np.ndarray,
    gravitational_parameters: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    radii: np.ndarray | None = None,
    tile_size: int = 2,
) -> np.ndarray:
    return newtonian_point_mass_acceleration(
        backend=BACKEND,
        positions=positions,
        gravitational_parameters=gravitational_parameters,
        radii=np.zeros(len(positions), dtype=F64) if radii is None else radii,
        source_mask=source_mask,
        target_mask=target_mask,
        tile_size=tile_size,
        singularity_policy=SINGULARITY_POLICY_ERROR,
        collision_policy=COLLISION_POLICY_ERROR,
    )


def one_pn(
    positions: np.ndarray,
    velocities: np.ndarray,
    gravitational_parameters: np.ndarray,
    *,
    radii: np.ndarray | None = None,
    speed_of_light: float = 100.0,
    maximum_compactness: float = 0.1,
    maximum_speed_fraction_squared: float = 0.1,
) -> np.ndarray:
    body_count = len(positions)
    return restricted_static_central_1pn_acceleration(
        backend=BACKEND,
        positions=positions,
        velocities=velocities,
        gravitational_parameters=gravitational_parameters,
        radii=np.zeros(body_count, dtype=F64) if radii is None else radii,
        source_mask=np.array((True,) + (False,) * (body_count - 1)),
        target_mask=np.array((False,) + (True,) * (body_count - 1)),
        central_index=0,
        speed_of_light=speed_of_light,
        maximum_compactness=maximum_compactness,
        maximum_speed_fraction_squared=maximum_speed_fraction_squared,
        singularity_policy=SINGULARITY_POLICY_ERROR,
        collision_policy=COLLISION_POLICY_ERROR,
    )


def srp(
    positions: np.ndarray,
    area_to_mass: np.ndarray,
    coefficient: np.ndarray,
    *,
    reference_pressure: float = 5.0,
    reference_distance: float = 2.0,
    radii: np.ndarray | None = None,
) -> np.ndarray:
    body_count = len(positions)
    return cannonball_srp_acceleration(
        backend=BACKEND,
        positions=positions,
        gravitational_parameters=np.zeros(body_count, dtype=F64),
        radii=np.zeros(body_count, dtype=F64) if radii is None else radii,
        source_mask=np.array((True,) + (False,) * (body_count - 1)),
        target_mask=np.array((False,) + (True,) * (body_count - 1)),
        radiation_source_index=0,
        reference_pressure=reference_pressure,
        reference_distance=reference_distance,
        area_to_mass=area_to_mass,
        radiation_pressure_coefficient=coefficient,
        singularity_policy=SINGULARITY_POLICY_ERROR,
        collision_policy=COLLISION_POLICY_ERROR,
    )


class MutualEIH1PNTests(unittest.TestCase):
    def test_backend_neutral_kernel_matches_independent_three_body_evaluator(self):
        positions = np.ascontiguousarray(
            ((-0.4, 0.2, 0.1), (0.8, -0.1, 0.3), (0.1, 1.1, -0.2)),
            dtype=F64,
        )
        velocities = np.ascontiguousarray(
            ((0.1, -0.3, 0.05), (-0.2, 0.4, 0.1), (0.3, 0.1, -0.2)),
            dtype=F64,
        )
        gm = np.ascontiguousarray((1.0, 0.2, 0.05), dtype=F64)
        selected = np.ones(3, dtype=np.bool_)
        observed = mutual_eih_1pn_acceleration(
            backend=BACKEND,
            positions=positions,
            velocities=velocities,
            gravitational_parameters=gm,
            radii=np.zeros(3, dtype=F64),
            body_mask=selected,
            tile_size=2,
            speed_of_light=31.0,
            maximum_compactness=1.0e-2,
            maximum_speed_fraction_squared=1.0e-2,
            singularity_policy=SINGULARITY_POLICY_ERROR,
            collision_policy=COLLISION_POLICY_ERROR,
        )
        expected = evaluate_eih_1pn_correction(
            positions,
            velocities,
            gm,
            EIH1PNParameters(
                speed_of_light_km_s=31.0,
                maximum_compactness=1.0e-2,
                maximum_speed_fraction_squared=1.0e-2,
            ),
        ).correction_accelerations_km_s2
        np.testing.assert_allclose(observed, expected, rtol=3.0e-15, atol=3.0e-15)

        translated = mutual_eih_1pn_acceleration(
            backend=BACKEND,
            positions=np.ascontiguousarray(positions + (7.0, -5.0, 2.0)),
            velocities=velocities,
            gravitational_parameters=gm,
            radii=np.zeros(3, dtype=F64),
            body_mask=selected,
            tile_size=3,
            speed_of_light=31.0,
            maximum_compactness=1.0e-2,
            maximum_speed_fraction_squared=1.0e-2,
            singularity_policy=SINGULARITY_POLICY_ERROR,
            collision_policy=COLLISION_POLICY_ERROR,
        )
        np.testing.assert_allclose(translated, observed, rtol=3.0e-14, atol=3.0e-15)

    def test_incomplete_roster_and_weak_field_violation_fail_closed(self):
        positions = np.ascontiguousarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
        velocities = np.ascontiguousarray(((0.0, 0.0, 0.0), (0.0, 0.5, 0.0)))
        gm = np.ascontiguousarray((1.0, 0.2))
        common = dict(
            backend=BACKEND,
            positions=positions,
            velocities=velocities,
            gravitational_parameters=gm,
            radii=np.zeros(2, dtype=F64),
            tile_size=2,
            speed_of_light=31.0,
            maximum_compactness=1.0e-2,
            maximum_speed_fraction_squared=1.0e-2,
            singularity_policy=SINGULARITY_POLICY_ERROR,
            collision_policy=COLLISION_POLICY_ERROR,
        )
        with self.assertRaisesRegex(ForceContractError, "every state body"):
            mutual_eih_1pn_acceleration(
                **common,
                body_mask=np.array((True, False), dtype=np.bool_),
            )
        with self.assertRaisesRegex(ForceDomainError, "weak-field"):
            mutual_eih_1pn_acceleration(
                **{**common, "maximum_compactness": 1.0e-5},
                body_mask=np.ones(2, dtype=np.bool_),
            )


def decimal_one_pn_oracle(
    position: tuple[str, str, str],
    velocity: tuple[str, str, str],
    mu: str,
    speed_of_light: str,
    exact_radius: str,
) -> tuple[Decimal, Decimal, Decimal]:
    """Independently expressed high-precision oracle for rational-radius cases."""

    with localcontext() as context:
        context.prec = 80
        r = tuple(Decimal(value) for value in position)
        v = tuple(Decimal(value) for value in velocity)
        checked_mu = Decimal(mu)
        c = Decimal(speed_of_light)
        radius = Decimal(exact_radius)
        speed_squared = sum((value * value for value in v), Decimal(0))
        radial_velocity_product = sum(
            (r[index] * v[index] for index in range(3)), Decimal(0)
        )
        scale = checked_mu / (c * c * radius * radius * radius)
        radial = Decimal(4) * checked_mu / radius - speed_squared
        return tuple(
            scale
            * (
                radial * r[index]
                + Decimal(4) * radial_velocity_product * v[index]
            )
            for index in range(3)
        )  # type: ignore[return-value]


class NewtonianPointMassTests(unittest.TestCase):
    def test_two_body_analytic_acceleration_and_massless_no_backreaction(self):
        positions = np.array(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)), dtype=F64)
        gm = np.array((4.0, 0.0), dtype=F64)
        result = newtonian(
            positions,
            gm,
            np.array((True, False)),
            np.array((True, True)),
        )
        np.testing.assert_array_equal(
            result,
            np.array(((0.0, 0.0, 0.0), (-1.0, 0.0, 0.0)), dtype=F64),
        )

    def test_two_mass_momentum_reaction_and_translation_invariance(self):
        positions = np.array(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)), dtype=F64)
        gm = np.array((4.0, 1.0), dtype=F64)
        selected = np.array((True, True))
        original = newtonian(positions, gm, selected, selected)
        translated = newtonian(
            positions + np.array((7.0, -3.0, 2.0)), gm, selected, selected
        )
        np.testing.assert_array_equal(
            original,
            np.array(((0.25, 0.0, 0.0), (-1.0, 0.0, 0.0)), dtype=F64),
        )
        np.testing.assert_array_equal(translated, original)
        np.testing.assert_array_equal(gm @ original, np.zeros(3, dtype=F64))

    def test_rotation_covariance(self):
        positions = np.array(
            ((0.0, 0.0, 0.0), (2.0, 1.0, 0.0), (-1.0, 3.0, 0.0)),
            dtype=F64,
        )
        gm = np.array((4.0, 1.0, 0.5), dtype=F64)
        selected = np.ones(3, dtype=np.bool_)
        rotation = np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
        original = newtonian(positions, gm, selected, selected)
        rotated = newtonian(positions @ rotation.T, gm, selected, selected)
        np.testing.assert_allclose(rotated, original @ rotation.T, rtol=2e-15, atol=0.0)

    def test_tile_sizes_preserve_the_same_force_to_roundoff(self):
        positions = np.array(
            (
                (0.0, 0.0, 0.0),
                (1.0, 2.0, 0.5),
                (-2.0, 0.25, 1.0),
                (0.75, -1.0, 2.0),
                (3.0, 0.5, -1.0),
            ),
            dtype=F64,
        )
        gm = np.array((1.0, 0.25, 0.5, 0.75, 0.125), dtype=F64)
        selected = np.ones(5, dtype=np.bool_)
        reference = newtonian(positions, gm, selected, selected, tile_size=1)
        for tile_size in (2, 3, 5, 8):
            with self.subTest(tile_size=tile_size):
                candidate = newtonian(
                    positions, gm, selected, selected, tile_size=tile_size
                )
                np.testing.assert_allclose(candidate, reference, rtol=3e-15, atol=2e-16)

    def test_inputs_are_not_mutated(self):
        positions = np.array(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)), dtype=F64)
        gm = np.array((4.0, 0.0), dtype=F64)
        sources = np.array((True, False))
        targets = np.array((False, True))
        originals = tuple(value.copy() for value in (positions, gm, sources, targets))
        newtonian(positions, gm, sources, targets)
        for value, original in zip((positions, gm, sources, targets), originals):
            np.testing.assert_array_equal(value, original)

    def test_singularity_collision_and_softening_requests_fail_closed(self):
        positions = np.array(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), dtype=F64)
        gm = np.array((1.0, 0.0), dtype=F64)
        with self.assertRaises(ForceSingularityError):
            newtonian(
                positions,
                gm,
                np.array((True, False)),
                np.array((False, True)),
            )

        positions[1, 0] = 1.0
        with self.assertRaises(ForceCollisionError):
            newtonian(
                positions,
                gm,
                np.array((True, False)),
                np.array((False, True)),
                radii=np.array((0.5, 0.5)),
            )

        with self.assertRaises(ForceContractError):
            newtonian_point_mass_acceleration(
                backend=BACKEND,
                positions=positions,
                gravitational_parameters=gm,
                radii=np.zeros(2),
                source_mask=np.array((True, False)),
                target_mask=np.array((False, True)),
                tile_size=2,
                singularity_policy="soften",
                collision_policy=COLLISION_POLICY_ERROR,
            )


class RestrictedStaticCentral1PNTests(unittest.TestCase):
    def test_matches_independent_decimal_oracle(self):
        positions = np.array(((0.0, 0.0, 0.0), (3.0, 4.0, 0.0)), dtype=F64)
        velocities = np.array(((0.0, 0.0, 0.0), (2.0, -1.0, 1.0)), dtype=F64)
        result = one_pn(
            positions,
            velocities,
            np.array((7.0, 0.0), dtype=F64),
        )
        oracle = decimal_one_pn_oracle(
            ("3", "4", "0"), ("2", "-1", "1"), "7", "100", "5"
        )
        expected = np.array((0.0, 0.0, 0.0, *map(float, oracle))).reshape(2, 3)
        np.testing.assert_allclose(result, expected, rtol=2e-15, atol=0.0)

    def test_inverse_c_squared_and_large_c_limit(self):
        positions = np.array(((0.0, 0.0, 0.0), (3.0, 4.0, 0.0)), dtype=F64)
        velocities = np.array(((0.0, 0.0, 0.0), (2.0, -1.0, 1.0)), dtype=F64)
        gm = np.array((7.0, 0.0), dtype=F64)
        base = one_pn(positions, velocities, gm, speed_of_light=100.0)
        doubled = one_pn(positions, velocities, gm, speed_of_light=200.0)
        distant_limit = one_pn(positions, velocities, gm, speed_of_light=1e30)
        np.testing.assert_allclose(doubled, base / 4.0, rtol=2e-15, atol=0.0)
        self.assertLess(float(np.max(np.abs(distant_limit))), 1e-58)

    def test_zero_velocity_correction_is_radially_outward(self):
        result = one_pn(
            np.array(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)), dtype=F64),
            np.zeros((2, 3), dtype=F64),
            np.array((3.0, 0.0), dtype=F64),
        )
        self.assertGreater(result[1, 0], 0.0)
        np.testing.assert_array_equal(result[1, 1:], np.zeros(2))

    def test_domain_bounds_static_source_and_collision_fail_closed(self):
        positions = np.array(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)), dtype=F64)
        velocities = np.zeros((2, 3), dtype=F64)
        gm = np.array((3.0, 0.0), dtype=F64)
        with self.assertRaises(ForceDomainError):
            one_pn(positions, velocities, gm, maximum_compactness=1e-5)
        velocities[1, 0] = 50.0
        with self.assertRaises(ForceDomainError):
            one_pn(
                positions,
                velocities,
                gm,
                maximum_speed_fraction_squared=0.1,
            )
        velocities[:] = 0.0
        velocities[0, 0] = 1.0
        with self.assertRaises(ForceDomainError):
            one_pn(positions, velocities, gm)
        velocities[:] = 0.0
        with self.assertRaises(ForceCollisionError):
            one_pn(positions, velocities, gm, radii=np.array((1.0, 1.0)))


class CannonballSRPTests(unittest.TestCase):
    def test_direction_scaling_and_exact_zero_controls(self):
        positions = np.array(
            ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 3.0, 0.0)),
            dtype=F64,
        )
        result = srp(
            positions,
            np.array((0.0, 3.0, 0.0), dtype=F64),
            np.array((0.0, 2.0, 1.0), dtype=F64),
        )
        np.testing.assert_array_equal(result[0], np.zeros(3))
        np.testing.assert_array_equal(result[1], np.array((30.0, 0.0, 0.0)))
        np.testing.assert_array_equal(result[2], np.zeros(3))

        twice_as_far = positions.copy()
        twice_as_far[1] *= 2.0
        scaled = srp(
            twice_as_far,
            np.array((0.0, 3.0, 0.0), dtype=F64),
            np.array((0.0, 2.0, 1.0), dtype=F64),
        )
        np.testing.assert_allclose(scaled[1], result[1] / 4.0, rtol=0.0, atol=0.0)

    def test_linear_pressure_area_and_coefficient_scaling(self):
        positions = np.array(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)), dtype=F64)
        area = np.array((0.0, 1.0), dtype=F64)
        coefficient = np.array((0.0, 1.0), dtype=F64)
        base = srp(positions, area, coefficient)
        np.testing.assert_array_equal(srp(positions, area * 2.0, coefficient), base * 2.0)
        np.testing.assert_array_equal(srp(positions, area, coefficient * 3.0), base * 3.0)
        np.testing.assert_array_equal(
            srp(positions, area, coefficient, reference_pressure=10.0), base * 2.0
        )

    def test_negative_nonfinite_singular_and_collision_inputs_fail_closed(self):
        positions = np.array(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)), dtype=F64)
        for area, coefficient in (
            ((0.0, -1.0), (0.0, 1.0)),
            ((0.0, 1.0), (0.0, -1.0)),
            ((0.0, np.nan), (0.0, 1.0)),
        ):
            with self.subTest(area=area, coefficient=coefficient), self.assertRaises(ForceDomainError):
                srp(positions, np.array(area), np.array(coefficient))

        coincident = positions.copy()
        coincident[1] = coincident[0]
        with self.assertRaises(ForceSingularityError):
            srp(coincident, np.array((0.0, 1.0)), np.array((0.0, 1.0)))
        with self.assertRaises(ForceCollisionError):
            srp(
                positions,
                np.array((0.0, 1.0)),
                np.array((0.0, 1.0)),
                radii=np.array((1.0, 1.0)),
            )


if __name__ == "__main__":
    unittest.main()
