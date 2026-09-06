import ast
import hashlib
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError
from decimal import ROUND_DOWN, Decimal, FloatOperation, localcontext
from fractions import Fraction
from pathlib import Path

from oracle.v5_solar_1pn.candidate import (
    ACCEPTED_EIGHTH_WEIGHTS_RATIONAL,
    CONTROLLER_ERROR_EXPONENT_RATIONAL,
    EMBEDDED_SEVENTH_WEIGHTS_RATIONAL,
    METHOD_ID,
    NEWTONIAN_MODEL_ID,
    REGISTRY_AUTHORIZED,
    REVIEW_STATUS,
    SOLAR_1PN_MODEL_ID,
    SOURCE_SHA256,
    SOURCE_SIZE_BYTES,
    TABLEAU_ALPHA_RATIONAL,
    TABLEAU_A_RATIONAL,
    AdaptiveControllerSpec,
    CentralSourceModel,
    CoherentUnitSystem,
    CoordinateTimeScale,
    DecimalContextSpec,
    ExactCheckpointIntegrationResult,
    ForceMode,
    InertialFrame,
    OracleCheckpoint,
    OracleContract,
    OracleContractError,
    OracleDomainError,
    OracleIntegrationError,
    OracleIntegrationResult,
    OracleResultMetadata,
    OracleState,
    TargetTreatment,
    VectorCheckpoint,
    adaptive_step_factor,
    controller_sha256,
    evaluate_acceleration,
    integrate_exact_checkpoints,
    integrate_rhs_exact_checkpoints,
    max_scaled_checkpoint_discrepancy,
    normalized_max_error,
    rk78_step,
)


D = Decimal
FRAME = InertialFrame.BCRS_DERIVED_SUN_RELATIVE_ICRS_ALIGNED_RESTRICTED
SCOPE = TargetTreatment.MASSLESS_NO_BACKREACTION
CTX50 = DecimalContextSpec("decimal.context.precision_50", 50)
CTX70 = DecimalContextSpec("decimal.context.precision_70", 70)
CTX90 = DecimalContextSpec("decimal.context.precision_90", 90)


def contract(*, target: str = "TOY_TARGET", mu: str = "3", c: str = "10") -> OracleContract:
    return OracleContract(
        central_source="SUN",
        target=target,
        gravitational_parameter=D(mu),
        speed_of_light=D(c),
        units=CoherentUnitSystem.AU_DAY,
        frame=FRAME,
        time_scale=CoordinateTimeScale.TDB_COMPATIBLE,
        central_source_model=CentralSourceModel.STATIC_SPHERICAL_SOLAR_MONOPOLE,
        target_treatment=SCOPE,
        maximum_compactness=D("0.1"),
        maximum_speed_fraction_squared=D("0.1"),
    )


def state(
    position=(D(2), D(0), D(0)),
    velocity=(D(1), D(1), D(0)),
    *,
    target: str = "TOY_TARGET",
) -> OracleState:
    return OracleState(
        central_source="SUN",
        target=target,
        position=position,
        velocity=velocity,
        units=CoherentUnitSystem.AU_DAY,
        frame=FRAME,
        time_scale=CoordinateTimeScale.TDB_COMPATIBLE,
        target_treatment=SCOPE,
    )


def controller(
    tolerance: str,
    *,
    dimension: int = 1,
    initial: str = "0.1",
    maximum: str = "0.3",
):
    return AdaptiveControllerSpec(
        absolute_tolerances=tuple(D(tolerance) for _ in range(dimension)),
        relative_tolerance=D(tolerance),
        initial_step=D(initial),
        minimum_step=D("1e-30"),
        maximum_step=D(maximum),
    )


class FehlbergTableauTests(unittest.TestCase):
    def test_every_table_x_rational_is_locked(self):
        expected_alpha = (
            (0, 1), (2, 27), (1, 9), (1, 6), (5, 12), (1, 2), (5, 6),
            (1, 6), (2, 3), (1, 3), (1, 1), (0, 1), (1, 1),
        )
        expected_a = (
            (),
            ((2, 27),),
            ((1, 36), (1, 12)),
            ((1, 24), (0, 1), (1, 8)),
            ((5, 12), (0, 1), (-25, 16), (25, 16)),
            ((1, 20), (0, 1), (0, 1), (1, 4), (1, 5)),
            ((-25, 108), (0, 1), (0, 1), (125, 108), (-65, 27), (125, 54)),
            ((31, 300), (0, 1), (0, 1), (0, 1), (61, 225), (-2, 9), (13, 900)),
            ((2, 1), (0, 1), (0, 1), (-53, 6), (704, 45), (-107, 9), (67, 90), (3, 1)),
            ((-91, 108), (0, 1), (0, 1), (23, 108), (-976, 135), (311, 54), (-19, 60), (17, 6), (-1, 12)),
            ((2383, 4100), (0, 1), (0, 1), (-341, 164), (4496, 1025), (-301, 82), (2133, 4100), (45, 82), (45, 164), (18, 41)),
            ((3, 205), (0, 1), (0, 1), (0, 1), (0, 1), (-6, 41), (-3, 205), (-3, 41), (3, 41), (6, 41), (0, 1)),
            ((-1777, 4100), (0, 1), (0, 1), (-341, 164), (4496, 1025), (-289, 82), (2193, 4100), (51, 82), (33, 164), (12, 41), (0, 1), (1, 1)),
        )
        expected_seven = (
            (41, 840), (0, 1), (0, 1), (0, 1), (0, 1), (34, 105),
            (9, 35), (9, 35), (9, 280), (9, 280), (41, 840), (0, 1), (0, 1),
        )
        expected_eight = (
            (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (34, 105),
            (9, 35), (9, 35), (9, 280), (9, 280), (0, 1), (41, 840), (41, 840),
        )
        self.assertEqual(TABLEAU_ALPHA_RATIONAL, expected_alpha)
        self.assertEqual(TABLEAU_A_RATIONAL, expected_a)
        self.assertEqual(EMBEDDED_SEVENTH_WEIGHTS_RATIONAL, expected_seven)
        self.assertEqual(ACCEPTED_EIGHTH_WEIGHTS_RATIONAL, expected_eight)
        self.assertEqual(CONTROLLER_ERROR_EXPONENT_RATIONAL, (1, 8))

    def test_stage_rows_and_both_weights_are_consistent(self):
        q = lambda value: Fraction(value[0], value[1])
        self.assertEqual(len(TABLEAU_A_RATIONAL), 13)
        for stage, (alpha, row) in enumerate(
            zip(TABLEAU_ALPHA_RATIONAL, TABLEAU_A_RATIONAL, strict=True)
        ):
            self.assertEqual(len(row), stage)
            self.assertEqual(sum(map(q, row), Fraction(0)), q(alpha))
        self.assertEqual(sum(map(q, EMBEDDED_SEVENTH_WEIGHTS_RATIONAL)), 1)
        self.assertEqual(sum(map(q, ACCEPTED_EIGHTH_WEIGHTS_RATIONAL)), 1)
        defect = tuple(
            q(seven) - q(eight)
            for seven, eight in zip(
                EMBEDDED_SEVENTH_WEIGHTS_RATIONAL,
                ACCEPTED_EIGHTH_WEIGHTS_RATIONAL,
                strict=True,
            )
        )
        expected = [Fraction(0)] * 13
        for stage, sign in ((0, 1), (10, 1), (11, -1), (12, -1)):
            expected[stage] = sign * Fraction(41, 840)
        self.assertEqual(defect, tuple(expected))

    def test_source_and_method_provenance_are_pinned(self):
        self.assertEqual(METHOD_ID, "fehlberg.nasa-tr-r-287.rk7-8.table-x.decimal.v1")
        self.assertEqual(
            SOURCE_SHA256,
            "5553a2a3eb53785a461762cc2b29428015f1b32c3ad0a5cb57f85a421256a0c8",
        )
        self.assertEqual(SOURCE_SIZE_BYTES, 2_625_098)
        source_path = (
            Path(__file__).resolve().parents[1]
            / "oracle/v5_solar_1pn/sources/nasa-tr-r-287-fehlberg.pdf"
        )
        source_bytes = source_path.read_bytes()
        self.assertEqual(len(source_bytes), SOURCE_SIZE_BYTES)
        self.assertEqual(hashlib.sha256(source_bytes).hexdigest(), SOURCE_SHA256)

    def test_reversed_seventh_and_eighth_weights_are_detected(self):
        def exponential_rhs(_epoch, vector):
            return vector

        result = rk78_step(exponential_rhs, D(0), (D(1),), D("0.1"), CTX90)
        self.assertEqual(
            result.accepted_eighth[0],
            D("1.10517091807564721368644359356968667668256145622400766433688244387832865199120343153264964"),
        )
        self.assertEqual(
            result.embedded_seventh[0],
            D("1.10517091807563085100653746366516428244823306551701613430008491736886798615193676922071983"),
        )
        with localcontext(CTX90.build()):
            exact = D("0.1").exp()
            self.assertLess(
                abs(result.accepted_eighth[0] - exact),
                abs(result.embedded_seventh[0] - exact),
            )
            self.assertEqual(
                result.error_embedded_minus_accepted[0],
                result.embedded_seventh[0] - result.accepted_eighth[0],
            )

    def test_accepted_formula_has_eighth_order_global_convergence(self):
        def exponential_rhs(_epoch, vector):
            return vector

        with localcontext(CTX90.build()):
            exact = D(1).exp()

        def fixed_propagation(step_size: Decimal) -> Decimal:
            epoch = D(0)
            vector = (D(1),)
            for _ in range(int(D(1) / step_size)):
                result = rk78_step(exponential_rhs, epoch, vector, step_size, CTX90)
                vector = result.accepted_eighth
                epoch += step_size
            return abs(vector[0] - exact)

        errors = [fixed_propagation(D(value)) for value in ("0.25", "0.125", "0.0625")]
        self.assertGreater(errors[0] / errors[1], D(200))
        self.assertGreater(errors[1] / errors[2], D(200))
        self.assertLess(errors[0] / errors[1], D(300))
        self.assertLess(errors[1] / errors[2], D(300))


class EquationAndContractTests(unittest.TestCase):
    def test_total_equation_sign_components_and_frozen_ledger(self):
        newtonian = evaluate_acceleration(contract(), state(), ForceMode.NEWTONIAN, CTX70)
        one_pn = evaluate_acceleration(
            contract(), state(), ForceMode.NEWTONIAN_PLUS_SOLAR_1PN, CTX70
        )
        self.assertEqual(newtonian.acceleration, (D("-0.75"), D(0), D(0)))
        self.assertEqual(one_pn.acceleration, (D("-0.69"), D("0.03"), D(0)))
        self.assertEqual(newtonian.component_model_ids, (NEWTONIAN_MODEL_ID,))
        self.assertEqual(
            one_pn.component_model_ids,
            (NEWTONIAN_MODEL_ID, SOLAR_1PN_MODEL_ID),
        )
        self.assertEqual(NEWTONIAN_MODEL_ID, "force.newtonian.point_mass")
        self.assertEqual(
            SOLAR_1PN_MODEL_ID,
            "relativity.solar_schwarzschild_test_particle_1pn",
        )

    def test_rotation_covariance_of_separately_transcribed_total_rhs(self):
        rotate = lambda vector: (-vector[1], vector[0], vector[2])
        model = contract(c="100")
        original_state = state(
            (D(2), D(1), D(0)),
            (D("0.5"), D("-0.25"), D("0.1")),
        )
        rotated_state = state(rotate(original_state.position), rotate(original_state.velocity))
        original = evaluate_acceleration(
            model, original_state, ForceMode.NEWTONIAN_PLUS_SOLAR_1PN, CTX70
        ).acceleration
        rotated = evaluate_acceleration(
            model, rotated_state, ForceMode.NEWTONIAN_PLUS_SOLAR_1PN, CTX70
        ).acceleration
        with localcontext(CTX70.build()):
            self.assertEqual(rotated, rotate(original))

    def test_state_metadata_mismatch_and_domain_violations_fail_closed(self):
        with self.assertRaises(OracleContractError):
            evaluate_acceleration(contract(), state(target="OTHER"), ForceMode.NEWTONIAN, CTX50)
        with self.assertRaises(OracleDomainError):
            evaluate_acceleration(
                contract(), state((D(0), D(0), D(0)), (D(0), D(0), D(0))),
                ForceMode.NEWTONIAN, CTX50,
            )
        with self.assertRaises(OracleDomainError):
            evaluate_acceleration(
                contract(), state((D("0.01"), D(0), D(0)), (D(0), D(0), D(0))),
                ForceMode.NEWTONIAN, CTX50,
            )
        with self.assertRaises(OracleContractError):
            state((1.0, D(0), D(0)), (D(0), D(0), D(0)))

    def test_decimal_overflow_is_trapped_and_wrapped(self):
        huge_c = contract(c="1e999999")
        with self.assertRaisesRegex(OracleDomainError, "trapped Decimal signal"):
            evaluate_acceleration(huge_c, state(), ForceMode.NEWTONIAN, CTX50)

    def test_frozen_context_rejects_ambient_or_unsupported_variants(self):
        with self.assertRaises(OracleContractError):
            DecimalContextSpec("decimal.context.precision_60", 60)
        with self.assertRaises(OracleContractError):
            DecimalContextSpec("decimal.context.precision_50", 50, rounding=ROUND_DOWN)
        with self.assertRaises(OracleContractError):
            DecimalContextSpec("wrong", 50)


class AdaptiveIntegrationTests(unittest.TestCase):
    @staticmethod
    def exponential_rhs(_epoch, vector):
        return vector

    def test_controller_uses_exact_one_eighth_exponent(self):
        spec = controller("1e-12")
        self.assertEqual(adaptive_step_factor(D(0), spec, CTX50), D(5))
        self.assertEqual(adaptive_step_factor(D(256), spec, CTX50), D("0.45"))
        self.assertEqual(adaptive_step_factor(D("1e100"), spec, CTX50), D("0.2"))

    def test_component_scales_cannot_hide_velocity_error(self):
        dimensioned = AdaptiveControllerSpec(
            absolute_tolerances=(D(1), D("1e-6")),
            relative_tolerance=D("1e-30"),
            initial_step=D("0.1"),
            minimum_step=D("1e-20"),
            maximum_step=D("0.2"),
        )
        position_only = normalized_max_error(
            (D(0), D(0)),
            (D(0), D(0)),
            (D("1e-4"), D(0)),
            dimensioned,
            CTX50,
        )
        with_velocity = normalized_max_error(
            (D(0), D(0)),
            (D(0), D(0)),
            (D("1e-4"), D("1e-4")),
            dimensioned,
            CTX50,
        )
        self.assertLess(position_only, D("0.001"))
        self.assertGreater(with_velocity, D(99))

    def test_wrong_length_absolute_tolerance_roster_fails_before_arithmetic(self):
        with self.assertRaisesRegex(OracleContractError, "roster"):
            integrate_rhs_exact_checkpoints(
                self.exponential_rhs,
                D(0),
                (D(1), D(1)),
                (D(1),),
                controller("1e-12", dimension=1),
                CTX50,
            )
        with self.assertRaises(OracleContractError):
            AdaptiveControllerSpec(
                absolute_tolerances=[D("1e-12")],  # type: ignore[arg-type]
                relative_tolerance=D("1e-12"),
                initial_step=D("0.1"),
                minimum_step=D("1e-20"),
                maximum_step=D("0.2"),
            )

    def test_exact_checkpoint_clipping_is_in_memory_and_monotone(self):
        result = integrate_rhs_exact_checkpoints(
            self.exponential_rhs,
            D(0),
            (D(1),),
            (D(0), D("0.3"), D(1)),
            controller("1e-18", initial="0.17"),
            CTX50,
        )
        self.assertEqual(tuple(item.epoch for item in result.checkpoints), (D(0), D("0.3"), D(1)))
        self.assertEqual(
            result.checkpoint_policy,
            "CLIP_EACH_STEP_TO_EXACT_CHECKPOINT_NO_INTERPOLATION",
        )
        self.assertTrue(
            all(
                left.accepted_steps <= right.accepted_steps
                and left.rejected_steps <= right.rejected_steps
                for left, right in zip(result.checkpoints, result.checkpoints[1:])
            )
        )
        self.assertFalse(result.metadata.qualification_outcomes_generated)
        self.assertFalse(result.metadata.registry_authorized)
        self.assertEqual(result.initial_epoch, D(0))
        self.assertEqual(result.initial_state, (D(1),))

    def test_tighter_tolerance_self_converges(self):
        def run(tolerance):
            return integrate_rhs_exact_checkpoints(
                self.exponential_rhs,
                D(0),
                (D(1),),
                (D("0.5"), D(1)),
                controller(tolerance, initial="0.2", maximum="0.4"),
                CTX70,
            )

        loose, tight, tighter = run("1e-10"), run("1e-16"), run("1e-22")
        loose_to_tight = max_scaled_checkpoint_discrepancy(
            loose, tight, D("1e-30"), D("1e-30"), CTX70
        )
        tight_to_tighter = max_scaled_checkpoint_discrepancy(
            tight, tighter, D("1e-30"), D("1e-30"), CTX70
        )
        self.assertLess(tight_to_tighter, loose_to_tight)

    def test_higher_decimal_precision_self_converges_on_second_axis(self):
        same_controller = controller("1e-20", initial="0.1", maximum="0.3")

        def run(context):
            return integrate_rhs_exact_checkpoints(
                self.exponential_rhs,
                D(0),
                (D(1),),
                (D("0.3"),),
                same_controller,
                context,
            )

        precision_50, precision_70, precision_90 = run(CTX50), run(CTX70), run(CTX90)
        discrepancy_50_70 = max_scaled_checkpoint_discrepancy(
            precision_50, precision_70, D("1e-80"), D("1e-80"), CTX90
        )
        discrepancy_70_90 = max_scaled_checkpoint_discrepancy(
            precision_70, precision_90, D("1e-80"), D("1e-80"), CTX90
        )
        self.assertLess(discrepancy_70_90, discrepancy_50_70)
        self.assertEqual(
            {
                precision_50.controller_sha256,
                precision_70.controller_sha256,
                precision_90.controller_sha256,
            },
            {controller_sha256(same_controller)},
        )

    def test_decimal_context_is_isolated_from_ambient_process_state(self):
        with localcontext() as ambient:
            ambient.prec = 6
            ambient.rounding = ROUND_DOWN
            ambient.traps[FloatOperation] = False
            before = (ambient.prec, ambient.rounding, dict(ambient.traps))
            result = rk78_step(self.exponential_rhs, D(0), (D(1),), D("0.1"), CTX90)
            after = (ambient.prec, ambient.rounding, dict(ambient.traps))
            self.assertEqual(before, after)
        self.assertEqual(
            result.accepted_eighth[0],
            D("1.10517091807564721368644359356968667668256145622400766433688244387832865199120343153264964"),
        )

    def test_unattainable_tolerance_fails_at_minimum_step(self):
        impossible = AdaptiveControllerSpec(
            absolute_tolerances=(D("1e-90"),),
            relative_tolerance=D("1e-90"),
            initial_step=D("0.1"),
            minimum_step=D("0.1"),
            maximum_step=D("0.1"),
        )
        with self.assertRaisesRegex(OracleIntegrationError, "minimum_step"):
            integrate_rhs_exact_checkpoints(
                self.exponential_rhs, D(0), (D(1),), (D(1),), impossible, CTX90
            )

    def test_toy_model_wrapper_retains_nonauthorizing_metadata(self):
        toy_contract = contract(mu="1", c="100")
        toy_state = state((D(1), D(0), D(0)), (D(0), D(1), D(0)))
        result = integrate_exact_checkpoints(
            toy_contract,
            D(0),
            toy_state,
            (D("0.05"), D("0.1")),
            ForceMode.NEWTONIAN_PLUS_SOLAR_1PN,
            controller("1e-14", dimension=6, initial="0.03", maximum="0.05"),
            CTX50,
        )
        self.assertEqual(tuple(item.epoch for item in result.checkpoints), (D("0.05"), D("0.1")))
        self.assertEqual(result.force_mode, ForceMode.NEWTONIAN_PLUS_SOLAR_1PN)
        self.assertEqual(result.contract, toy_contract)
        self.assertEqual(result.initial_epoch, D(0))
        self.assertEqual(result.initial_state, toy_state)
        self.assertFalse(result.metadata.registry_authorized)
        self.assertEqual(result.metadata.review_status, REVIEW_STATUS)
        with self.assertRaises(FrozenInstanceError):
            result.metadata.registry_authorized = True


class TamperResistanceAndIsolationTests(unittest.TestCase):
    def test_exported_generic_result_rejects_mixed_series(self):
        first = VectorCheckpoint(D(1), (D(1),), 2, 1)
        bound_controller = controller("1e-12")
        bound_sha256 = controller_sha256(bound_controller)
        with self.assertRaises(OracleContractError):
            ExactCheckpointIntegrationResult(
                checkpoints=(first, VectorCheckpoint(D(2), (D(1), D(2)), 3, 1)),
                context_id=CTX50.context_id,
                initial_epoch=D(0),
                initial_state=(D(1),),
                controller=bound_controller,
                controller_sha256=bound_sha256,
            )
        with self.assertRaises(OracleContractError):
            ExactCheckpointIntegrationResult(
                checkpoints=(first, VectorCheckpoint(D(2), (D(2),), 1, 1)),
                context_id=CTX50.context_id,
                initial_epoch=D(0),
                initial_state=(D(1),),
                controller=bound_controller,
                controller_sha256=bound_sha256,
            )
        with self.assertRaises(OracleContractError):
            ExactCheckpointIntegrationResult(
                checkpoints=(first, VectorCheckpoint(D("0.5"), (D(2),), 3, 1)),
                context_id=CTX50.context_id,
                initial_epoch=D(0),
                initial_state=(D(1),),
                controller=bound_controller,
                controller_sha256=bound_sha256,
            )

    def test_exported_oracle_result_rejects_changed_state_metadata(self):
        first = OracleCheckpoint(D(1), state(target="A"), 1, 0)
        second = OracleCheckpoint(D(2), state(target="B"), 2, 0)
        bound_controller = controller("1e-12", dimension=6)
        bound_contract = contract(target="A")
        with self.assertRaises(OracleContractError):
            OracleIntegrationResult(
                checkpoints=(first, second),
                force_mode=ForceMode.NEWTONIAN,
                context_id=CTX50.context_id,
                contract=bound_contract,
                initial_epoch=D(0),
                initial_state=state(target="A"),
                controller=bound_controller,
                controller_sha256=controller_sha256(bound_controller),
            )

    def test_controller_record_and_digest_cannot_be_mixed(self):
        first = VectorCheckpoint(D(1), (D(1),), 1, 0)
        loose = controller("1e-12")
        tight = controller("1e-18")
        self.assertNotEqual(controller_sha256(loose), controller_sha256(tight))
        with self.assertRaisesRegex(OracleContractError, "controller_sha256"):
            ExactCheckpointIntegrationResult(
                checkpoints=(first,),
                context_id=CTX50.context_id,
                initial_epoch=D(0),
                initial_state=(D(1),),
                controller=loose,
                controller_sha256=controller_sha256(tight),
            )

    def test_initial_condition_binding_cannot_be_lost_or_mixed(self):
        bound_controller = controller("1e-12")
        with self.assertRaisesRegex(OracleContractError, "initial condition"):
            ExactCheckpointIntegrationResult(
                checkpoints=(VectorCheckpoint(D(0), (D(2),), 0, 0),),
                context_id=CTX50.context_id,
                initial_epoch=D(0),
                initial_state=(D(1),),
                controller=bound_controller,
                controller_sha256=controller_sha256(bound_controller),
            )
        model_controller = controller("1e-12", dimension=6)
        with self.assertRaises(OracleContractError):
            OracleIntegrationResult(
                checkpoints=(OracleCheckpoint(D(1), state(target="A"), 1, 0),),
                force_mode=ForceMode.NEWTONIAN,
                context_id=CTX50.context_id,
                contract=contract(target="B"),
                initial_epoch=D(0),
                initial_state=state(target="A"),
                controller=model_controller,
                controller_sha256=controller_sha256(model_controller),
            )

    def test_metadata_cannot_promote_candidate(self):
        self.assertFalse(REGISTRY_AUTHORIZED)
        self.assertEqual(REVIEW_STATUS, "CANDIDATE_PENDING_INDEPENDENT_REVIEW")
        with self.assertRaises(OracleContractError):
            OracleResultMetadata(registry_authorized=True)
        with self.assertRaises(OracleContractError):
            OracleResultMetadata(qualification_outcomes_generated=True)

    def test_ast_has_no_jx_import_io_cli_or_holdout_binding(self):
        source_path = Path(__file__).resolve().parents[1] / "oracle/v5_solar_1pn/candidate.py"
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        allowed_roots = {
            "__future__", "collections", "dataclasses", "decimal", "enum", "hashlib", "json", "typing"
        }
        imported_roots = set()
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_roots.add((node.module or "").split(".")[0])
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                calls.append(node.func.id)
        self.assertLessEqual(imported_roots, allowed_roots)
        self.assertNotIn("open", calls)
        self.assertNotIn("__main__", source)
        self.assertNotIn("qualification_inputs_v1", source)
        self.assertNotIn("fixture.holdout", source)

    def test_clean_runtime_import_does_not_load_jxplanetx(self):
        repository = Path(__file__).resolve().parents[1]
        script = (
            "import sys; "
            f"sys.path.insert(0, {str(repository)!r}); "
            "import oracle.v5_solar_1pn.candidate; "
            "assert not any(n == 'jxplanetx' or n.startswith('jxplanetx.') for n in sys.modules)"
        )
        completed = subprocess.run(
            [sys.executable, "-S", "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
