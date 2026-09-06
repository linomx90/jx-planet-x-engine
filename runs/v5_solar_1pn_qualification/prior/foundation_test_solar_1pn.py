import math
import unittest
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction

from jxplanetx.solar_1pn import (
    AccelerationSemantics,
    CentralSourceModel,
    CoherentUnitSystem,
    CoordinateTimeScale,
    DuplicateForceApplicationError,
    ForceEvaluationContext,
    InertialFrame,
    RelativeState,
    RelativityContractError,
    RelativityDomainError,
    SOLAR_SCHWARZSCHILD_1PN_MODEL_ID,
    SolarSchwarzschild1PNContract,
    TargetTreatment,
    solar_schwarzschild_1pn_correction,
)


D = Decimal
FRAME = InertialFrame.BCRS_DERIVED_SUN_RELATIVE_ICRS_ALIGNED_RESTRICTED
SCOPE = TargetTreatment.MASSLESS_NO_BACKREACTION
EMPTY_CONTEXT = ForceEvaluationContext(frozenset())


def contract(
    mu: str = "3",
    c: str = "10",
    *,
    units: CoherentUnitSystem = CoherentUnitSystem.AU_DAY,
    time_scale: CoordinateTimeScale = CoordinateTimeScale.TDB_COMPATIBLE,
) -> SolarSchwarzschild1PNContract:
    return SolarSchwarzschild1PNContract(
        central_source="SUN",
        gravitational_parameter=D(mu),
        speed_of_light=D(c),
        units=units,
        frame=FRAME,
        time_scale=time_scale,
        central_source_model=CentralSourceModel.STATIC_SPHERICAL_SOLAR_MONOPOLE,
        target_treatment=SCOPE,
        output_semantics=AccelerationSemantics.CORRECTION_ONLY,
        maximum_compactness=D("0.1"),
        maximum_speed_fraction_squared=D("0.1"),
    )


def state(
    position: tuple[Decimal, Decimal, Decimal],
    velocity: tuple[Decimal, Decimal, Decimal],
    *,
    target: str = "TEST_PARTICLE",
    units: CoherentUnitSystem = CoherentUnitSystem.AU_DAY,
    time_scale: CoordinateTimeScale = CoordinateTimeScale.TDB_COMPATIBLE,
) -> RelativeState:
    return RelativeState(
        central_source="SUN",
        target=target,
        position=position,
        velocity=velocity,
        units=units,
        frame=FRAME,
        time_scale=time_scale,
        target_treatment=SCOPE,
    )


def independent_fraction_oracle(
    r: tuple[Fraction, Fraction, Fraction],
    v: tuple[Fraction, Fraction, Fraction],
    mu: Fraction,
    c: Fraction,
    exact_radius: Fraction,
) -> tuple[Fraction, Fraction, Fraction]:
    """Separately expressed rational oracle for fixtures with rational |r|."""
    r_dot_v = sum((r[i] * v[i] for i in range(3)), Fraction(0))
    speed2 = sum((item * item for item in v), Fraction(0))
    radial = 4 * mu / exact_radius - speed2
    scale = mu / (c * c * exact_radius * exact_radius * exact_radius)
    return tuple(scale * (radial * r[i] + 4 * r_dot_v * v[i]) for i in range(3))  # type: ignore[return-value]


class SolarSchwarzschild1PNEquationTests(unittest.TestCase):
    def evaluate(
        self,
        position: tuple[Decimal, Decimal, Decimal],
        velocity: tuple[Decimal, Decimal, Decimal],
        *,
        model: SolarSchwarzschild1PNContract | None = None,
        context: ForceEvaluationContext = EMPTY_CONTEXT,
    ):
        return solar_schwarzschild_1pn_correction(
            model or contract(), state(position, velocity), context
        )

    def test_exact_fixture_matches_independently_coded_fraction_oracle(self):
        r_fraction = (Fraction(3), Fraction(4), Fraction(0))
        v_fraction = (Fraction(2), Fraction(-1), Fraction(1))
        oracle = independent_fraction_oracle(
            r_fraction, v_fraction, Fraction(7), Fraction(100), Fraction(5)
        )
        result = self.evaluate(
            tuple(D(item.numerator) / D(item.denominator) for item in r_fraction),  # type: ignore[arg-type]
            tuple(D(item.numerator) / D(item.denominator) for item in v_fraction),  # type: ignore[arg-type]
            model=contract("7", "100"),
        )
        expected = tuple(D(item.numerator) / D(item.denominator) for item in oracle)
        self.assertEqual(result.acceleration, expected)
        self.assertTrue(all(type(item) is Decimal for item in result.acceleration))
        self.assertEqual(result.semantics, AccelerationSemantics.CORRECTION_ONLY)
        self.assertFalse(result.registry_authorized)

    def test_locked_sign_and_component_fixture_is_correction_only(self):
        result = self.evaluate((D(2), D(0), D(0)), (D(1), D(1), D(0)))
        self.assertEqual(result.acceleration, (D("0.06"), D("0.03"), D(0)))
        newtonian = (D("-0.75"), D(0), D(0))
        total = tuple(newtonian[i] + result.acceleration[i] for i in range(3))
        self.assertEqual(total, (D("-0.69"), D("0.03"), D(0)))
        self.assertNotEqual(result.acceleration, total)

    def test_zero_velocity_is_radially_outward(self):
        result = self.evaluate((D(2), D(0), D(0)), (D(0), D(0), D(0)))
        self.assertEqual(result.acceleration, (D("0.045"), D(0), D(0)))

    def test_radial_motion_generates_tangential_term_with_expected_sign(self):
        outward = self.evaluate((D(2), D(0), D(0)), (D(1), D(1), D(0))).acceleration
        inward = self.evaluate((D(2), D(0), D(0)), (D(-1), D(1), D(0))).acceleration
        self.assertGreater(outward[1], 0)
        self.assertLess(inward[1], 0)
        self.assertEqual(outward[0], inward[0])

    def test_circular_orbit_analytic_component(self):
        with localcontext() as decimal_context:
            decimal_context.prec = 60
            mu, radius, c = D(3), D(2), D(100)
            circular_speed = (mu / radius).sqrt()
            result = self.evaluate(
                (radius, D(0), D(0)),
                (D(0), circular_speed, D(0)),
                model=contract(str(mu), str(c)),
            )
            expected_radial = 3 * mu * mu / (c * c * radius * radius * radius)
            self.assertLess(abs(result.acceleration[0] - expected_radial), D("1e-60"))
            self.assertEqual(result.acceleration[1:], (D(0), D(0)))

    def test_velocity_dependence_is_not_hidden(self):
        resting = self.evaluate((D(2), D(0), D(0)), (D(0), D(0), D(0))).acceleration
        moving = self.evaluate((D(2), D(0), D(0)), (D(1), D(1), D(0))).acceleration
        self.assertNotEqual(resting, moving)

    def test_rotation_covariance(self):
        # Proper 90-degree rotation about z: (x,y,z) -> (-y,x,z).
        rotate = lambda q: (-q[1], q[0], q[2])
        r = (D(2), D(1), D(0))
        v = (D("0.5"), D("-0.25"), D("0.1"))
        original = self.evaluate(r, v, model=contract("3", "100")).acceleration
        rotated = self.evaluate(rotate(r), rotate(v), model=contract("3", "100")).acceleration
        self.assertEqual(rotated, rotate(original))

    def test_coherent_length_time_rescaling_and_inverse_c_squared(self):
        with localcontext() as decimal_context:
            decimal_context.prec = 70
            r = (D(2), D(0), D(0))
            v = (D(1), D(1), D(0))
            base = self.evaluate(r, v, model=contract("3", "100")).acceleration

            # Same quantities expressed from AU/day in km/s.
            length_scale = D("149597870.7")
            time_scale = D(86400)
            converted_model = contract(
                str(D(3) * length_scale**3 / time_scale**2),
                str(D(100) * length_scale / time_scale),
                units=CoherentUnitSystem.KILOMETRE_SECOND,
            )
            converted_state = state(
                tuple(length_scale * item for item in r),  # type: ignore[arg-type]
                tuple(length_scale / time_scale * item for item in v),  # type: ignore[arg-type]
                units=CoherentUnitSystem.KILOMETRE_SECOND,
            )
            scaled = solar_schwarzschild_1pn_correction(
                converted_model, converted_state, EMPTY_CONTEXT
            ).acceleration
            expected_scale = length_scale / time_scale**2
            self.assertLess(
                max(abs(scaled[i] - expected_scale * base[i]) for i in range(3)),
                D("1e-60"),
            )

            doubled_c = self.evaluate(r, v, model=contract("3", "200")).acceleration
            self.assertLess(
                max(abs(doubled_c[i] - base[i] / 4) for i in range(3)),
                D("1e-60"),
            )

    def test_large_finite_c_limit_recovers_newtonian_total(self):
        with localcontext() as decimal_context:
            decimal_context.prec = 70
            correction = self.evaluate(
                (D(2), D(0), D(0)),
                (D(1), D(1), D(0)),
                model=contract("3", "1e30"),
            ).acceleration
            newtonian = (D("-0.75"), D(0), D(0))
            total = tuple(newtonian[i] + correction[i] for i in range(3))
            self.assertLess(max(abs(item) for item in correction), D("1e-59"))
            self.assertLess(max(abs(total[i] - newtonian[i]) for i in range(3)), D("1e-59"))

    def test_solar_system_fixture_is_small_relative_to_newtonian(self):
        with localcontext() as decimal_context:
            decimal_context.prec = 60
            mu = D("0.0002959122082855911025")
            c = D("173.144632674240")
            circular_speed = mu.sqrt()
            correction = self.evaluate(
                (D(1), D(0), D(0)),
                (D(0), circular_speed, D(0)),
                model=contract(str(mu), str(c)),
            ).acceleration
            newtonian_magnitude = mu
            self.assertLess(abs(correction[0]) / newtonian_magnitude, D("1e-7"))

    def test_equation_level_perihelion_relation(self):
        # Integrate Gauss' periapsis equation over one unperturbed Kepler ellipse.
        # This is an equation-level quadrature, not a trajectory propagation.
        mu, c, semimajor, eccentricity = 1.0, 1000.0, 2.0, 0.2
        p = semimajor * (1.0 - eccentricity * eccentricity)
        samples = 4096
        accumulated = 0.0
        model = contract(str(mu), str(c))
        for index in range(samples):
            anomaly = 2.0 * math.pi * index / samples
            cosine, sine = math.cos(anomaly), math.sin(anomaly)
            radius = p / (1.0 + eccentricity * cosine)
            radial_speed = math.sqrt(mu / p) * eccentricity * sine
            transverse_speed = math.sqrt(mu / p) * (1.0 + eccentricity * cosine)
            radial_unit = (cosine, sine, 0.0)
            transverse_unit = (-sine, cosine, 0.0)
            position = tuple(D(str(radius * item)) for item in radial_unit)
            velocity = tuple(
                D(str(radial_speed * radial_unit[i] + transverse_speed * transverse_unit[i]))
                for i in range(3)
            )
            delta = self.evaluate(position, velocity, model=model).acceleration  # type: ignore[arg-type]
            radial_accel = sum(float(delta[i]) * radial_unit[i] for i in range(3))
            transverse_accel = sum(float(delta[i]) * transverse_unit[i] for i in range(3))
            gauss_domega_df = radius * radius / (eccentricity * mu) * (
                -radial_accel * cosine
                + transverse_accel * (1.0 + radius / p) * sine
            )
            accumulated += gauss_domega_df
        quadrature = accumulated * (2.0 * math.pi / samples)
        analytic = 6.0 * math.pi * mu / (p * c * c)
        self.assertAlmostEqual(quadrature, analytic, delta=analytic * 2e-12)


class SolarSchwarzschild1PNFailureTests(unittest.TestCase):
    def test_rejects_duplicate_application_before_evaluation(self):
        duplicate = ForceEvaluationContext(frozenset((SOLAR_SCHWARZSCHILD_1PN_MODEL_ID,)))
        # A coincident state would also fail, but duplicate detection must win.
        with self.assertRaises(DuplicateForceApplicationError):
            solar_schwarzschild_1pn_correction(
                contract(), state((D(0), D(0), D(0)), (D(0), D(0), D(0))), duplicate
            )

    def test_rejects_total_output_request_and_non_solar_source(self):
        values = dict(
            central_source="SUN",
            gravitational_parameter=D(3),
            speed_of_light=D(10),
            units=CoherentUnitSystem.AU_DAY,
            frame=FRAME,
            time_scale=CoordinateTimeScale.TDB_COMPATIBLE,
            central_source_model=CentralSourceModel.STATIC_SPHERICAL_SOLAR_MONOPOLE,
            target_treatment=SCOPE,
            output_semantics=AccelerationSemantics.TOTAL_ACCELERATION,
            maximum_compactness=D("0.1"),
            maximum_speed_fraction_squared=D("0.1"),
        )
        with self.assertRaises(RelativityContractError):
            SolarSchwarzschild1PNContract(**values)
        values["central_source"] = "EARTH"
        values["output_semantics"] = AccelerationSemantics.CORRECTION_ONLY
        with self.assertRaises(RelativityContractError):
            SolarSchwarzschild1PNContract(**values)

    def test_rejects_binary_float_bool_and_nonfinite_decimal(self):
        with self.assertRaises(RelativityContractError):
            replace(contract(), gravitational_parameter=3.0)  # type: ignore[arg-type]
        with self.assertRaises(RelativityContractError):
            replace(contract(), speed_of_light=True)  # type: ignore[arg-type]
        with self.assertRaises(RelativityDomainError):
            contract(mu="NaN")
        with self.assertRaises(RelativityDomainError):
            state((D("Infinity"), D(0), D(0)), (D(0), D(0), D(0)))
        with self.assertRaises(RelativityContractError):
            state((1.0, D(0), D(0)), (D(0), D(0), D(0)))  # type: ignore[arg-type]

    def test_rejects_nonpositive_parameters_and_zero_separation(self):
        for mu in ("0", "-1"):
            with self.assertRaises(RelativityDomainError):
                contract(mu=mu)
        for c in ("0", "-1"):
            with self.assertRaises(RelativityDomainError):
                contract(c=c)
        with self.assertRaises(RelativityDomainError):
            solar_schwarzschild_1pn_correction(
                contract(), state((D(0), D(0), D(0)), (D(0), D(0), D(0))), EMPTY_CONTEXT
            )

    def test_rejects_metadata_mismatch_and_massive_target_treatment(self):
        mismatched_state = state(
            (D(2), D(0), D(0)),
            (D(0), D(0), D(0)),
            units=CoherentUnitSystem.METRE_SECOND,
        )
        with self.assertRaises(RelativityContractError):
            solar_schwarzschild_1pn_correction(contract(), mismatched_state, EMPTY_CONTEXT)

        mismatched_time = state(
            (D(2), D(0), D(0)),
            (D(0), D(0), D(0)),
            time_scale=CoordinateTimeScale.TCB_COMPATIBLE,
        )
        with self.assertRaises(RelativityContractError):
            solar_schwarzschild_1pn_correction(contract(), mismatched_time, EMPTY_CONTEXT)

        values = state((D(2), D(0), D(0)), (D(0), D(0), D(0)))
        with self.assertRaises(RelativityContractError):
            RelativeState(
                central_source=values.central_source,
                target=values.target,
                position=values.position,
                velocity=values.velocity,
                units=values.units,
                frame=values.frame,
                time_scale=values.time_scale,
                target_treatment="MASSIVE",  # type: ignore[arg-type]
            )

    def test_rejects_states_outside_declared_1pn_bounds(self):
        with self.assertRaises(RelativityDomainError):
            solar_schwarzschild_1pn_correction(
                contract("3", "10"),
                state((D("0.1"), D(0), D(0)), (D(0), D(0), D(0))),
                EMPTY_CONTEXT,
            )
        with self.assertRaises(RelativityDomainError):
            solar_schwarzschild_1pn_correction(
                contract("3", "10"),
                state((D(2), D(0), D(0)), (D(4), D(0), D(0))),
                EMPTY_CONTEXT,
            )

    def test_context_and_sequences_must_be_immutable(self):
        with self.assertRaises(RelativityContractError):
            ForceEvaluationContext(set())  # type: ignore[arg-type]
        with self.assertRaises(RelativityContractError):
            state([D(2), D(0), D(0)], (D(0), D(0), D(0)))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
