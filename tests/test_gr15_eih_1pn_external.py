from __future__ import annotations

from importlib import metadata
import unittest

import numpy as np
import rebound
import reboundx

from jxplanetx.gr15 import GR15Spec
from jxplanetx.gr15_eih_1pn import integrate_gr15_eih_1pn
from jxplanetx.solar_system.eih_1pn import (
    EIH1PNParameters,
    evaluate_eih_1pn_correction,
)


END_EPOCH = 0.1
REBOUND_IAS15_EPSILON = 1.0e-15


def _finite_mass_state() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gm = np.ascontiguousarray((1.0, 0.2), dtype=np.float64)
    relative_position = np.ascontiguousarray((1.2, 0.3, -0.1), dtype=np.float64)
    relative_velocity = np.ascontiguousarray((-0.2, 0.9, 0.15), dtype=np.float64)
    total = float(np.sum(gm))
    positions = np.ascontiguousarray(
        (
            -gm[1] / total * relative_position,
            gm[0] / total * relative_position,
        ),
        dtype=np.float64,
    )
    velocities = np.ascontiguousarray(
        (
            -gm[1] / total * relative_velocity,
            gm[0] / total * relative_velocity,
        ),
        dtype=np.float64,
    )
    return positions, velocities, gm


def _physical_sun_earth_state() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gm = np.ascontiguousarray(
        (132_712_440_041.279419, 398_600.435507),
        dtype=np.float64,
    )
    total = float(np.sum(gm))
    semimajor_axis_km = 149_597_870.7
    eccentricity = 0.0167
    relative_position = np.asarray(
        (semimajor_axis_km * (1.0 - eccentricity), 0.0, 0.0),
        dtype=np.float64,
    )
    relative_velocity = np.asarray(
        (
            0.0,
            np.sqrt(
                total
                * (1.0 + eccentricity)
                / (semimajor_axis_km * (1.0 - eccentricity))
            ),
            0.0,
        ),
        dtype=np.float64,
    )
    positions = np.ascontiguousarray(
        (
            -gm[1] / total * relative_position,
            gm[0] / total * relative_position,
        ),
        dtype=np.float64,
    )
    velocities = np.ascontiguousarray(
        (
            -gm[1] / total * relative_velocity,
            gm[0] / total * relative_velocity,
        ),
        dtype=np.float64,
    )
    return positions, velocities, gm


def _simulation(
    positions: np.ndarray,
    velocities: np.ndarray,
    gm: np.ndarray,
) -> rebound.Simulation:
    simulation = rebound.Simulation()
    simulation.G = 1.0
    for index in range(gm.size):
        simulation.add(
            m=float(gm[index]),
            x=float(positions[index, 0]),
            y=float(positions[index, 1]),
            z=float(positions[index, 2]),
            vx=float(velocities[index, 0]),
            vy=float(velocities[index, 1]),
            vz=float(velocities[index, 2]),
        )
    simulation.integrator = "ias15"
    simulation.integrator.epsilon = REBOUND_IAS15_EPSILON
    return simulation


def _state(simulation: rebound.Simulation) -> tuple[np.ndarray, np.ndarray]:
    particles = simulation.particles
    positions = np.ascontiguousarray(
        tuple((particle.x, particle.y, particle.z) for particle in particles),
        dtype=np.float64,
    )
    velocities = np.ascontiguousarray(
        tuple((particle.vx, particle.vy, particle.vz) for particle in particles),
        dtype=np.float64,
    )
    return positions, velocities


def _jx_result(speed_of_light: float):
    positions, velocities, gm = _finite_mass_state()
    return integrate_gr15_eih_1pn(
        positions,
        velocities,
        gm,
        GR15Spec(
            0.0,
            END_EPOCH,
            initial_step=1.0e-4,
            minimum_step=1.0e-14,
            maximum_step=1.0e-3,
            epsilon=1.0e-10,
        ),
        EIH1PNParameters(
            speed_of_light_km_s=speed_of_light,
            maximum_compactness=1.0e-2,
            maximum_speed_fraction_squared=1.0e-2,
        ),
    )


def _reboundx_gr_full(speed_of_light: float) -> tuple[np.ndarray, np.ndarray]:
    positions, velocities, gm = _finite_mass_state()
    simulation = _simulation(positions, velocities, gm)
    extras = reboundx.Extras(simulation)
    force = extras.load_force("gr_full")
    force.params["c"] = speed_of_light
    extras.add_force(force)
    simulation.integrate(END_EPOCH, exact_finish_time=1)
    return _state(simulation)


def _rebound_same_model_state(
    positions: np.ndarray,
    velocities: np.ndarray,
    gm: np.ndarray,
    parameters: EIH1PNParameters,
    final_epoch: float,
) -> tuple[np.ndarray, np.ndarray]:
    simulation = _simulation(positions, velocities, gm)

    def add_eih_correction(simulation_pointer) -> None:
        active = simulation_pointer.contents
        particles = active.particles
        stage_positions = np.ascontiguousarray(
            tuple((particle.x, particle.y, particle.z) for particle in particles),
            dtype=np.float64,
        )
        stage_velocities = np.ascontiguousarray(
            tuple((particle.vx, particle.vy, particle.vz) for particle in particles),
            dtype=np.float64,
        )
        correction = evaluate_eih_1pn_correction(
            stage_positions,
            stage_velocities,
            gm,
            parameters,
        ).correction_accelerations_km_s2
        for index, particle in enumerate(particles):
            particle.ax += float(correction[index, 0])
            particle.ay += float(correction[index, 1])
            particle.az += float(correction[index, 2])

    simulation.force_is_velocity_dependent = 1
    simulation.additional_forces = add_eih_correction
    simulation.integrate(final_epoch, exact_finish_time=1)
    return _state(simulation)


def _rebound_same_model(speed_of_light: float) -> tuple[np.ndarray, np.ndarray]:
    positions, velocities, gm = _finite_mass_state()
    return _rebound_same_model_state(
        positions,
        velocities,
        gm,
        EIH1PNParameters(
            speed_of_light_km_s=speed_of_light,
            maximum_compactness=1.0e-2,
            maximum_speed_fraction_squared=1.0e-2,
        ),
        END_EPOCH,
    )


class GR15EIH1PNExternalTests(unittest.TestCase):
    def test_reference_versions_are_exact(self) -> None:
        self.assertEqual(metadata.version("rebound"), "5.1.1")
        self.assertEqual(metadata.version("reboundx"), "5.1.0")
        self.assertEqual(rebound.__version__, "5.1.1")
        self.assertEqual(reboundx.__version__, "5.1.0")

    def test_reboundx_difference_has_inverse_c_fourth_limit(self) -> None:
        differences: dict[float, tuple[float, float]] = {}
        for speed_of_light in (31.0, 62.0):
            jx = _jx_result(speed_of_light)
            reference_positions, reference_velocities = _reboundx_gr_full(
                speed_of_light
            )
            differences[speed_of_light] = (
                float(np.linalg.norm(jx.positions - reference_positions)),
                float(np.linalg.norm(jx.velocities - reference_velocities)),
            )
        for component in range(2):
            low_c = differences[31.0][component]
            high_c = differences[62.0][component]
            self.assertGreater(high_c, 1.0e-15)
            self.assertGreaterEqual(low_c / high_c, 12.0)

    def test_same_equations_match_independent_ias15_trajectory(self) -> None:
        speed_of_light = 31.0
        jx = _jx_result(speed_of_light)
        reference_positions, reference_velocities = _rebound_same_model(
            speed_of_light
        )
        np.testing.assert_allclose(
            jx.positions,
            reference_positions,
            rtol=0.0,
            atol=5.0e-13,
        )
        np.testing.assert_allclose(
            jx.velocities,
            reference_velocities,
            rtol=0.0,
            atol=5.0e-13,
        )

    def test_physical_sun_earth_same_model_100_year_gate(self) -> None:
        positions, velocities, gm = _physical_sun_earth_state()
        final_epoch = 100.0 * 365.25 * 86_400.0
        parameters = EIH1PNParameters(speed_of_light_km_s=299_792.458)
        jx = integrate_gr15_eih_1pn(
            positions,
            velocities,
            gm,
            GR15Spec(
                0.0,
                final_epoch,
                initial_step=86_400.0,
                minimum_step=1.0e-6,
                maximum_step=10.0 * 86_400.0,
                epsilon=1.0e-10,
            ),
            parameters,
        )
        reference_positions, reference_velocities = _rebound_same_model_state(
            positions,
            velocities,
            gm,
            parameters,
            final_epoch,
        )
        maximum_position_difference_metres = 1_000.0 * float(
            np.max(np.linalg.norm(jx.positions - reference_positions, axis=1))
        )
        maximum_velocity_difference_metres_per_second = 1_000.0 * float(
            np.max(np.linalg.norm(jx.velocities - reference_velocities, axis=1))
        )
        self.assertLessEqual(maximum_position_difference_metres, 10_000.0)
        self.assertLessEqual(
            maximum_velocity_difference_metres_per_second,
            0.01,
        )


if __name__ == "__main__":
    unittest.main()
