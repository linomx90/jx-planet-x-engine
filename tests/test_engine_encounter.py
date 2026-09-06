import dataclasses
import hashlib
import math
import unittest
from fractions import Fraction
from unittest import mock

import numpy as np

import jxplanetx.engine.encounter_contracts as encounter_contracts
from jxplanetx.engine.contracts import (
    BackendSpec,
    ContractError,
    ForcePlan,
    NewtonianPointMass,
    ParameterMetadata,
    Provenance,
    StateSnapshot,
)
from jxplanetx.engine.encounter import (
    EncounterContractError,
    EncounterDomainError,
    EncounterProposalRecord,
    EncounterResourceError,
    EncounterSegmentResult,
    EncounterStepLimitError,
    _Dyad,
    _EncounterRun,
    _ExactBudget,
    _MutableUsage,
    _Rat,
    _Transcript,
    _canonical_json,
    _domain_sha256,
    _normalized_error,
    _record_diagnostic_bytes,
    _result_sha256,
    _schedule_sha256,
    integrate_encounter_segment,
)
from jxplanetx.engine.encounter_contracts import (
    AdaptiveEncounterSegmentSpec,
    ENCOUNTER_EXACT_ABSTRACT_WORK_WEIGHT_TABLE,
    ENCOUNTER_EXACT_ABSTRACT_WORK_WEIGHT_TABLE_ID,
    ENCOUNTER_EXACT_WITNESS_CHECKSUM_DOMAIN,
    ENCOUNTER_HARD_MAXIMUM_OPERATIONS_PER_PROPOSAL,
    ENCOUNTER_HARD_MAXIMUM_WITNESS_TRANSCRIPT_BYTES_PER_PROPOSAL,
    EncounterExactRationalResourceSpec,
)
from jxplanetx.engine.rkf78 import (
    RKF78_ACCEPTED_ORDER,
    RKF78_DEFECT_ORIENTATION,
    RKF78_EMBEDDED_ORDER,
    RKF78_STAGE_COUNT,
    RKF78_TABLEAU_ID,
)


UNIT_SYSTEM_ID = "fixture.encounter.units"


def _provenance() -> Provenance:
    return Provenance(
        source_id="fixture.encounter",
        citation="Synthetic encounter fixture",
        version="1",
        sha256="a" * 64,
    )


def _metadata(start: float = -100.0, end: float = 100.0) -> ParameterMetadata:
    return ParameterMetadata(
        parameter_id="state.gravitational_parameters",
        units="L^3/T^2",
        provenance=_provenance(),
        uncertainty=None,
        covariance_group=None,
        validity_start=start,
        validity_end=end,
    )


def _resources(**changes: int) -> EncounterExactRationalResourceSpec:
    values = {
        "maximum_body_count": 16,
        "maximum_pair_count": 120,
        "maximum_integer_bits": 8192,
        "maximum_rational_exponent_magnitude": 4096,
        "maximum_gcd_iterations_per_reduction": 16384,
        "maximum_initialization_operations": 100_000,
        "maximum_initialization_gcd_iterations": 500_000,
        "maximum_initialization_transcript_bytes": 1_048_576,
        "maximum_gcd_iterations_per_proposal": 500_000,
        "maximum_gcd_iterations_per_segment": 64_500_000,
        "maximum_operations_per_proposal": 650_000,
        "maximum_operations_per_segment": 83_450_000,
        "maximum_witness_transcript_bytes_per_proposal": 1_310_720,
        "maximum_witness_transcript_bytes_per_segment": 168_820_736,
        "maximum_witness_diagnostic_bytes_per_proposal": 4096,
        "maximum_witness_ledger_bytes": 16_777_216,
    }
    values.update(changes)
    return EncounterExactRationalResourceSpec(**values)


def _state(
    *,
    positions: tuple[tuple[float, float, float], ...] = (
        (-0.5, 0.0, 0.0),
        (0.5, 0.0, 0.0),
    ),
    velocities: tuple[tuple[float, float, float], ...] = (
        (0.0, -0.5, 0.0),
        (0.0, 0.5, 0.0),
    ),
    gm: tuple[float, ...] = (0.5, 0.5),
    radii: tuple[float, ...] = (0.0, 0.0),
    epoch: float = 0.0,
) -> StateSnapshot:
    body_ids = tuple(chr(ord("A") + index) for index in range(len(gm)))
    return StateSnapshot(
        snapshot_id="fixture.encounter.snapshot",
        epoch=epoch,
        time_scale="SYNTHETIC",
        frame="BARYCENTRIC_INERTIAL",
        origin="BARYCENTER",
        axes="CARTESIAN_RIGHT_HANDED",
        length_unit="L",
        time_unit="T",
        mass_unit="M",
        unit_system_id=UNIT_SYSTEM_ID,
        body_ids=body_ids,
        positions=np.array(positions, dtype=np.float64),
        velocities=np.array(velocities, dtype=np.float64),
        gravitational_parameters=np.array(gm, dtype=np.float64),
        masses=np.array(gm, dtype=np.float64),
        radii=np.array(radii, dtype=np.float64),
        massive=np.ones(len(gm), dtype=np.bool_),
        provenance=_provenance(),
    )


def _plan(body_ids: tuple[str, ...], *, start: float = -100.0, end: float = 100.0) -> ForcePlan:
    return ForcePlan(
        plan_id="fixture.encounter.plan",
        backend=BackendSpec(
            backend_id="numpy",
            device="cpu",
            tile_size=2,
            dtype="float64",
            allow_fallback=False,
            deterministic_reductions=True,
            fast_math=False,
            determinism_scope="SAME_RUNTIME_DEVICE",
        ),
        models=(
            NewtonianPointMass(
                source_ids=body_ids,
                target_ids=body_ids,
                unit_system_id=UNIT_SYSTEM_ID,
                parameter_metadata=(_metadata(start, end),),
            ),
        ),
    )


def _spec(
    state: StateSnapshot,
    *,
    duration: float = 1.0,
    endpoint_epoch: float | None = None,
    step: float = 0.0625,
    minimum_step: float | None = None,
    floor: float = 2.0 / 3.0,
    atol: float = 1.0e-12,
    rtol: float = 1.0e-12,
    maximum_substep_proposals: int = 128,
    maximum_accepted_substeps: int = 128,
    maximum_rejected_substeps: int = 64,
    maximum_consecutive_rejections: int = 32,
    maximum_force_evaluations: int = 1664,
    resources: EncounterExactRationalResourceSpec | None = None,
) -> AdaptiveEncounterSegmentSpec:
    pair_count = len(state.body_ids) * (len(state.body_ids) - 1) // 2
    if endpoint_epoch is None:
        endpoint_epoch = state.epoch + math.copysign(10.0, duration)
    return AdaptiveEncounterSegmentSpec(
        body_order=state.body_ids,
        initial_epoch=float(state.epoch),
        endpoint_epoch=float(endpoint_epoch),
        duration=float(duration),
        initial_step_magnitude=float(step),
        minimum_step_magnitude=float(minimum_step if minimum_step is not None else step),
        maximum_step_magnitude=float(step),
        pair_certification_floors=(float(floor),) * pair_count,
        pair_position_atols=(float(atol),) * pair_count,
        pair_position_rtol=float(rtol),
        pair_velocity_atols=(float(atol),) * pair_count,
        pair_velocity_rtol=float(rtol),
        gm_centroid_position_atol=float(atol),
        gm_centroid_position_rtol=float(rtol),
        gm_centroid_velocity_atol=float(atol),
        gm_centroid_velocity_rtol=float(rtol),
        maximum_substep_proposals=maximum_substep_proposals,
        maximum_accepted_substeps=maximum_accepted_substeps,
        maximum_rejected_substeps=maximum_rejected_substeps,
        maximum_consecutive_rejections=maximum_consecutive_rejections,
        maximum_force_evaluations=maximum_force_evaluations,
        safety_factor=0.9,
        minimum_scale_factor=0.2,
        maximum_scale_factor=5.0,
        exact_rational_resources=resources or _resources(),
    )


class EncounterRuntimeTests(unittest.TestCase):
    def test_tableau_contract_identity(self) -> None:
        state = _state()
        spec = _spec(state)
        self.assertEqual(spec.tableau_id, RKF78_TABLEAU_ID)
        self.assertEqual(spec.stage_count, RKF78_STAGE_COUNT)
        self.assertEqual(spec.accepted_order, RKF78_ACCEPTED_ORDER)
        self.assertEqual(spec.embedded_order, RKF78_EMBEDDED_ORDER)
        self.assertEqual(spec.defect_orientation, RKF78_DEFECT_ORIENTATION)

    def test_abstract_exact_work_weight_table_known_answer(self) -> None:
        expected = (
            ("GENERAL_RATIONAL", "INTEGER_WIDTH_OBSERVATION", 2),
            ("GENERAL_RATIONAL", "RATIONAL_EXPONENT_DIAGNOSTIC", 1),
            ("GENERAL_RATIONAL", "INTEGER_COMPARE", 1),
            ("GENERAL_RATIONAL", "INTEGER_ABSOLUTE", 1),
            ("GENERAL_RATIONAL", "INTEGER_NEGATE", 1),
            ("GENERAL_RATIONAL", "INTEGER_SHIFT_LEFT", 1),
            ("GENERAL_RATIONAL", "INTEGER_ADD", 1),
            ("GENERAL_RATIONAL", "INTEGER_SUBTRACT", 1),
            ("GENERAL_RATIONAL", "INTEGER_MULTIPLY", 1),
            ("GENERAL_RATIONAL", "INTEGER_FLOOR_DIVIDE", 1),
            ("GENERAL_RATIONAL", "BINARY64_ENVELOPE_BASE", 8),
            ("GENERAL_RATIONAL", "BINARY64_ENVELOPE_NONZERO", 4),
            ("GENERAL_RATIONAL", "RATIONAL_CONSTRUCTION", 1),
            ("GENERAL_RATIONAL", "FROM_BINARY64_RATIO_EXTRACTION", 1),
            ("GENERAL_RATIONAL", "SIGNED_STEP_ABSOLUTE", 1),
            ("DYADIC", "INTEGER_WIDTH_OBSERVATION", 2),
            ("DYADIC", "BINARY64_ENVELOPE_BASE", 8),
            ("DYADIC", "BINARY64_ENVELOPE_NONZERO", 4),
            ("DYADIC", "DYADIC_CONSTRUCTION", 1),
            ("DYADIC", "INPUT_EXPONENT_CHECK", 1),
            ("DYADIC", "NONZERO_ODDNESS_BRANCH", 1),
            ("DYADIC", "EVEN_LOWBIT_PATH", 3),
            ("DYADIC", "EVEN_CANONICALIZE_PATH", 3),
            ("DYADIC", "CANONICAL_EXPONENT_DIAGNOSTIC", 4),
            ("DYADIC", "FROM_BINARY64_RATIO_EXTRACTION", 1),
            ("DYADIC", "FROM_BINARY64_POWER_OF_TWO_CHECK", 2),
            ("DYADIC", "ADD_OR_SUBTRACT_ALIGN", 3),
            ("DYADIC", "ADD_OR_SUBTRACT_SHIFT_PAIR", 2),
            ("DYADIC", "ADD_OR_SUBTRACT_COMBINE", 1),
            ("DYADIC", "MULTIPLY_MANTISSA_AND_EXPONENT", 2),
            ("DYADIC", "NEGATE", 1),
            ("DYADIC", "COMPARE_ALIGN_AND_SHIFT_PAIR", 5),
            ("DYADIC", "COMPARE_RESULT", 1),
            ("DYADIC", "TO_BINARY64_RATIO_BUILD", 1),
            ("DYADIC", "TO_BINARY64_ROUND", 1),
            ("DYADIC", "DURATION_ABSOLUTE", 1),
            ("DYADIC", "SCHEDULER_NEXTAFTER_DOWN", 1),
            ("DYADIC", "SCHEDULER_SUCCESSOR_PROBE", 1),
            ("GCD", "EUCLIDEAN_ITERATION", 1),
        )
        self.assertEqual(
            ENCOUNTER_EXACT_ABSTRACT_WORK_WEIGHT_TABLE_ID,
            "ENCOUNTER_EXACT_ABSTRACT_WORK_WEIGHTS_V1",
        )
        self.assertEqual(ENCOUNTER_EXACT_ABSTRACT_WORK_WEIGHT_TABLE, expected)
        resources = _resources()
        self.assertEqual(
            resources.work_weight_table_id,
            ENCOUNTER_EXACT_ABSTRACT_WORK_WEIGHT_TABLE_ID,
        )
        self.assertEqual(resources.work_weight_table, expected)
        with self.assertRaises(ContractError):
            dataclasses.replace(resources, work_weight_table_id="changed")

        class TextSubclass(str):
            pass

        malformed = (
            (TextSubclass(expected[0][0]), expected[0][1], expected[0][2]),
        ) + expected[1:]
        with self.assertRaises(ContractError):
            dataclasses.replace(resources, work_weight_table=malformed)
        with self.assertRaises(ContractError):
            dataclasses.replace(resources, work_weight_table=expected * 1000)

    def test_abstract_exact_work_constants_have_exact_contract_roster(self) -> None:
        self.assertEqual(
            tuple(
                name
                for name in encounter_contracts.__all__
                if name.startswith("ENCOUNTER_EXACT_ABSTRACT_WORK_WEIGHT_TABLE")
            ),
            (
                "ENCOUNTER_EXACT_ABSTRACT_WORK_WEIGHT_TABLE",
                "ENCOUNTER_EXACT_ABSTRACT_WORK_WEIGHT_TABLE_ID",
            ),
        )
        self.assertEqual(
            len(encounter_contracts.__all__),
            len(set(encounter_contracts.__all__)),
        )

    def test_canonical_serializer_literal_known_answer(self) -> None:
        domain = "jxplanetx.encounter-kat.content-integrity.v1"
        expected = (
            b"jxplanetx.encounter-kat.content-integrity.v1\x00"
            b'{"array":{"dtype":"float64","shape":[1,2],"values":'
            b'["0x0.0p+0","-0x0.0p+0"]},"bools":{"dtype":"bool",'
            b'"shape":[2],"values":[true,false]},"float":{"float_hex":'
            b'"-0x0.0p+0"},"tuple":[1,"x"]}'
        )
        expected_digest = (
            "669fb2216194fccd8725d85330c662ca2bd5b852977719fa98c2587f49a06cd0"
        )
        value = {
            "array": np.array(((+0.0, -0.0),), dtype=np.float64),
            "bools": np.array((True, False), dtype=np.bool_),
            "float": -0.0,
            "tuple": (1, "x"),
        }
        self.assertEqual(hashlib.sha256(expected).hexdigest(), expected_digest)
        self.assertEqual(domain.encode("utf-8") + b"\x00" + _canonical_json(value), expected)
        self.assertEqual(_domain_sha256(domain, value), expected_digest)
        mutated = dict(value)
        mutated["float"] = +0.0
        self.assertNotEqual(_domain_sha256(domain, mutated), expected_digest)

    def test_exact_witness_literal_known_answer(self) -> None:
        expected = (
            b"jxplanetx.encounter-exact-clearance-witness.content-integrity.v1\x00"
            b'{"actual_force_evaluations":0,"disposition":"CERTIFICATE_REJECTION",'
            b'"entries":[{"branch":"START","kind":"kat_exact_compare","left":'
            b'{"denominator":"1","numerator":"1"},"operator":">","pair_index":0,'
            b'"result":true,"right":{"denominator":"1","numerator":"0"}}],'
            b'"method_id":"integrator.adaptive.rkf78.encounter_segment_newtonian_v1",'
            b'"phase":"PROPOSAL","proposal_index":1,"schema":'
            b'"jxplanetx.encounter-exact-clearance-witness.v1","signed_step_hex":'
            b'"0x1.0000000000000p-1"}'
        )
        expected_digest = (
            "045020a5da910116d54edec3302cfc960e7ec7483c8ffa45c02c6bd94cf6dada"
        )
        self.assertEqual(len(expected), 530)
        self.assertEqual(hashlib.sha256(expected).hexdigest(), expected_digest)
        resources = _resources()
        usage = _MutableUsage()
        budget = _ExactBudget(
            resources=resources,
            segment=usage,
            operation_limit=resources.maximum_operations_per_proposal,
            gcd_limit=resources.maximum_gcd_iterations_per_proposal,
            transcript_limit=resources.maximum_witness_transcript_bytes_per_proposal,
            label="witness KAT",
        )
        transcript = _Transcript(
            "PROPOSAL",
            budget=budget,
            domain=ENCOUNTER_EXACT_WITNESS_CHECKSUM_DOMAIN,
            proposal_index=1,
        )
        transcript.comparison(
            kind="kat_exact_compare",
            pair_index=0,
            left=_Rat(1, 1, budget),
            operator=">",
            right=_Rat(0, 1, budget),
            result=True,
            branch="START",
        )
        digest = transcript.finish(
            budget=budget,
            domain=ENCOUNTER_EXACT_WITNESS_CHECKSUM_DOMAIN,
            proposal_index=1,
            signed_step=0.5,
            disposition="CERTIFICATE_REJECTION",
            force_evaluations=0,
        )
        self.assertEqual(digest, expected_digest)
        self.assertEqual(budget.local.transcript_bytes, len(expected))
        self.assertEqual(budget.local.exact_operations, 53)
        self.assertEqual(budget.local.rational_operations, 53)
        self.assertEqual(budget.local.dyadic_operations, 0)
        self.assertEqual(budget.local.gcd_iterations, 1)

        mutated_usage = _MutableUsage()
        mutated_budget = _ExactBudget(
            resources=resources,
            segment=mutated_usage,
            operation_limit=resources.maximum_operations_per_proposal,
            gcd_limit=resources.maximum_gcd_iterations_per_proposal,
            transcript_limit=resources.maximum_witness_transcript_bytes_per_proposal,
            label="mutated witness KAT",
        )
        mutated_transcript = _Transcript(
            "PROPOSAL",
            budget=mutated_budget,
            domain=ENCOUNTER_EXACT_WITNESS_CHECKSUM_DOMAIN,
            proposal_index=1,
        )
        mutated_transcript.comparison(
            kind="kat_exact_compare",
            pair_index=0,
            left=_Rat(1, 1, mutated_budget),
            operator=">",
            right=_Rat(0, 1, mutated_budget),
            result=False,
            branch="START",
        )
        self.assertNotEqual(
            mutated_transcript.finish(
                budget=mutated_budget,
                domain=ENCOUNTER_EXACT_WITNESS_CHECKSUM_DOMAIN,
                proposal_index=1,
                signed_step=0.5,
                disposition="CERTIFICATE_REJECTION",
                force_evaluations=0,
            ),
            expected_digest,
        )

    def test_dyadic_known_answers_and_strict_floor_equality(self) -> None:
        resources = _resources()
        usage = _MutableUsage()
        budget = _ExactBudget(
            resources=resources,
            segment=usage,
            operation_limit=resources.maximum_operations_per_proposal,
            gcd_limit=resources.maximum_gcd_iterations_per_proposal,
            transcript_limit=resources.maximum_witness_transcript_bytes_per_proposal,
            label="dyadic KAT",
        )

        def components(value: _Dyad) -> tuple[int, int]:
            return value.mantissa, value.exponent

        one = _Dyad.from_float(1.0, budget)
        self.assertEqual(components(_Dyad.from_float(1.5, budget)), (3, -1))
        self.assertEqual(components(_Dyad.from_float(2.0, budget)), (1, 1))
        self.assertEqual(components(_Dyad.from_float(-0.0, budget)), (0, 0))
        successor = _Dyad.from_float(math.nextafter(1.0, math.inf), budget)
        self.assertEqual(components(successor), (4503599627370497, -52))
        ulp = successor.subtract(one, budget)
        self.assertEqual(components(ulp), (1, -52))
        self.assertEqual(components(ulp.square(budget)), (1, -104))
        tenth = _Dyad.from_float(0.1, budget)
        self.assertEqual(components(tenth), (3602879701896397, -55))
        self.assertEqual(
            components(tenth.square(budget)),
            (12980742146337070512478121581609, -110),
        )
        self.assertEqual(components(one.add(one.negate(budget), budget)), (0, 0))
        self.assertEqual(components(_Dyad.from_float(5.0e-324, budget)), (1, -1074))
        maximum = float.fromhex("0x1.fffffffffffffp+1023")
        maximum_dyad = _Dyad.from_float(maximum, budget)
        self.assertEqual(components(maximum_dyad), (9007199254740991, 971))
        opposite_distance = maximum_dyad.subtract(
            _Dyad.from_float(-maximum, budget), budget
        )
        self.assertEqual(components(opposite_distance), (9007199254740991, 972))
        self.assertEqual(budget.local.gcd_iterations, 0)
        self.assertEqual(budget.local.rational_operations, 0)
        self.assertEqual(budget.local.exact_operations, budget.local.dyadic_operations)
        self.assertEqual(budget.local.exact_operations, 311)
        self.assertEqual(budget.local.maximum_rational_exponent_magnitude, 1074)

        diagnostic_usage = _MutableUsage()
        diagnostic_budget = _ExactBudget(
            resources=resources,
            segment=diagnostic_usage,
            operation_limit=resources.maximum_operations_per_proposal,
            gcd_limit=resources.maximum_gcd_iterations_per_proposal,
            transcript_limit=resources.maximum_witness_transcript_bytes_per_proposal,
            label="dyadic exponent diagnostic KAT",
        )
        self.assertEqual(
            components(_Dyad(1 << 100, -100, diagnostic_budget)),
            (1, 0),
        )
        self.assertEqual(components(_Dyad(0, -4096, diagnostic_budget)), (0, 0))
        self.assertEqual(
            diagnostic_budget.local.maximum_rational_exponent_magnitude,
            0,
        )

        equality = _state(
            positions=((-0.05, 0.0, 0.0), (0.05, 0.0, 0.0)),
            velocities=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        )
        with mock.patch("jxplanetx.engine.encounter.evaluate_force_plan") as evaluate:
            with self.assertRaises(EncounterDomainError):
                integrate_encounter_segment(
                    equality,
                    _plan(equality.body_ids),
                    _spec(equality, duration=0.1, step=0.1, floor=0.1),
                )
            evaluate.assert_not_called()

    def test_forward_exact_duration_external_label_and_accounting(self) -> None:
        state = _state(epoch=1.0e16)
        spec = _spec(
            state,
            duration=1.0,
            endpoint_epoch=math.nextafter(state.epoch, math.inf),
        )
        result = integrate_encounter_segment(state, _plan(state.body_ids, start=-1.0e17, end=1.0e17), spec)
        self.assertIsInstance(result, EncounterSegmentResult)
        self.assertEqual(result.endpoint_epoch, spec.endpoint_epoch)
        self.assertEqual(
            sum(Fraction.from_float(step) for step in result.accepted_signed_substeps),
            Fraction.from_float(spec.duration),
        )
        self.assertEqual(result.primary_counts.accepted_substeps, 16)
        self.assertEqual(result.primary_counts.force_evaluations, 208)
        self.assertEqual(result.validation_replay_counts, result.primary_counts)
        self.assertEqual(
            result.total_public_call_counts.force_evaluations,
            2 * result.primary_counts.force_evaluations,
        )
        self.assertTrue(all(entry.state_metadata.epoch == state.epoch for entry in result.force_ledger))

    def test_backward_and_analytic_equal_binary_orbit(self) -> None:
        state = _state()
        forward = integrate_encounter_segment(state, _plan(state.body_ids), _spec(state))
        expected = np.array(
            (
                (-0.5 * math.cos(1.0), -0.5 * math.sin(1.0), 0.0),
                (0.5 * math.cos(1.0), 0.5 * math.sin(1.0), 0.0),
            ),
            dtype=np.float64,
        )
        self.assertLess(float(np.max(np.abs(forward.final_positions - expected))), 2.0e-13)
        backward_state = _state(
            positions=tuple(tuple(float(x) for x in row) for row in forward.final_positions),
            velocities=tuple(tuple(float(x) for x in row) for row in forward.final_velocities),
            epoch=10.0,
        )
        backward = integrate_encounter_segment(
            backward_state,
            _plan(backward_state.body_ids),
            _spec(backward_state, duration=-1.0, endpoint_epoch=0.0),
        )
        self.assertTrue(all(step < 0.0 for step in backward.accepted_signed_substeps))
        self.assertLess(float(np.max(np.abs(backward.final_positions - state.positions))), 4.0e-13)
        self.assertLess(float(np.max(np.abs(backward.final_velocities - state.velocities))), 4.0e-13)

    def test_exact_tunnel_rejected_before_force_and_equality_is_uncertified(self) -> None:
        tunnel = _state(
            positions=((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            velocities=((2.0, 0.0, 0.0), (-2.0, 0.0, 0.0)),
            gm=(1.0e-12, 1.0e-12),
            radii=(0.05, 0.05),
        )
        spec = _spec(
            tunnel,
            duration=1.0,
            step=1.0,
            minimum_step=0.01,
            floor=0.1,
            atol=1.0e-6,
            rtol=1.0e-6,
            maximum_substep_proposals=1,
            maximum_accepted_substeps=1,
            maximum_rejected_substeps=1,
            maximum_consecutive_rejections=1,
            maximum_force_evaluations=13,
        )
        with mock.patch("jxplanetx.engine.encounter.evaluate_force_plan") as evaluate:
            with self.assertRaises(EncounterStepLimitError):
                integrate_encounter_segment(tunnel, _plan(tunnel.body_ids), spec)
            evaluate.assert_not_called()

        equality = _state(
            positions=((-0.75, 0.0, 0.0), (0.75, 0.0, 0.0)),
            velocities=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            gm=(0.5, 0.5),
        )
        equality_spec = _spec(
            equality,
            duration=1.0,
            step=1.0,
            minimum_step=0.01,
            floor=1.0,
            atol=1.0,
            rtol=1.0,
            maximum_substep_proposals=1,
            maximum_accepted_substeps=1,
            maximum_rejected_substeps=1,
            maximum_consecutive_rejections=1,
            maximum_force_evaluations=13,
        )
        with mock.patch("jxplanetx.engine.encounter.evaluate_force_plan") as evaluate:
            with self.assertRaises(EncounterStepLimitError):
                integrate_encounter_segment(
                    equality, _plan(equality.body_ids), equality_spec
                )
            evaluate.assert_not_called()

    def test_near_miss_one_step(self) -> None:
        state = _state(
            positions=((-1.0, -0.1, 0.0), (1.0, 0.1, 0.0)),
            velocities=((2.0, 0.0, 0.0), (-2.0, 0.0, 0.0)),
            gm=(1.0e-12, 1.0e-12),
            radii=(0.05, 0.05),
        )
        spec = _spec(state, duration=1.0, step=1.0, minimum_step=0.01, floor=0.1, atol=1.0e-6, rtol=1.0e-6)
        result = integrate_encounter_segment(state, _plan(state.body_ids), spec)
        self.assertEqual(result.primary_counts.accepted_substeps, 1)
        self.assertEqual(result.primary_counts.force_evaluations, 13)
        self.assertEqual(result.proposal_ledger[0].disposition, "ACCEPTED")

    def test_arrays_are_owned_readonly_and_caller_independent(self) -> None:
        state = _state()
        original = state.positions.copy()
        result = integrate_encounter_segment(state, _plan(state.body_ids), _spec(state))
        state.positions[0, 0] = 99.0
        self.assertTrue(np.array_equal(result.initial_snapshot.positions, original))
        for array in (
            result.initial_snapshot.positions,
            result.initial_snapshot.velocities,
            result.initial_snapshot.gravitational_parameters,
            result.initial_snapshot.masses,
            result.initial_snapshot.radii,
            result.initial_snapshot.massive,
            result.final_positions,
            result.final_velocities,
        ):
            self.assertTrue(array.flags.owndata)
            self.assertFalse(array.flags.writeable)
        with self.assertRaises(ValueError):
            result.final_positions[0, 0] = 0.0
        with self.assertRaises(EncounterContractError):
            dataclasses.replace(result, schedule_content_sha256="0" * 64)

    def test_exact_public_schema_rejects_subclasses_and_fake_backend(self) -> None:
        class TextSubclass(str):
            pass

        class EqualToEverything:
            def __eq__(self, _other: object) -> bool:
                return True

        state = _state()
        result = integrate_encounter_segment(
            state,
            _plan(state.body_ids),
            _spec(state, duration=0.0625, step=0.0625, minimum_step=0.0625),
        )
        for name in (
            "snapshot_id",
            "plan_id",
            "backend_id",
            "device",
            "dtype",
            "pair_table_content_sha256",
            "schedule_content_sha256",
            "result_content_sha256",
        ):
            with self.subTest(name=name):
                with self.assertRaises(EncounterContractError):
                    dataclasses.replace(
                        result,
                        **{name: TextSubclass(getattr(result, name))},
                    )
        with self.assertRaises(EncounterContractError):
            dataclasses.replace(result, backend_spec=EqualToEverything())
        with self.assertRaises(EncounterContractError):
            dataclasses.replace(
                result.proposal_ledger[0],
                disposition=TextSubclass(result.proposal_ledger[0].disposition),
            )

        retained_backend = result.backend_spec
        malformed_backend = object.__new__(BackendSpec)
        for descriptor in dataclasses.fields(BackendSpec):
            object.__setattr__(
                malformed_backend,
                descriptor.name,
                getattr(retained_backend, descriptor.name),
            )
        object.__setattr__(
            malformed_backend,
            "determinism_scope",
            TextSubclass(retained_backend.determinism_scope),
        )
        with self.assertRaises(EncounterContractError):
            dataclasses.replace(result, backend_spec=malformed_backend)
        malformed_plan = dataclasses.replace(
            result.force_plan,
            backend=malformed_backend,
        )
        with self.assertRaises(EncounterContractError):
            dataclasses.replace(
                result,
                backend_spec=malformed_backend,
                force_plan=malformed_plan,
            )

    def test_invalid_utf8_identifiers_fail_with_contract_errors(self) -> None:
        state = _state()
        spec = _spec(state)
        plan = _plan(state.body_ids)
        lone_surrogate = "\ud800"

        with self.assertRaisesRegex(ContractError, "valid UTF-8"):
            dataclasses.replace(spec, body_order=("A", lone_surrogate))

        invalid_snapshot = dataclasses.replace(
            state,
            snapshot_id=lone_surrogate,
        )
        with self.assertRaisesRegex(EncounterContractError, "valid UTF-8"):
            integrate_encounter_segment(invalid_snapshot, plan, spec)

        invalid_plan = dataclasses.replace(plan, plan_id=lone_surrogate)
        with self.assertRaisesRegex(EncounterContractError, "valid UTF-8"):
            integrate_encounter_segment(state, invalid_plan, spec)

        invalid_model = dataclasses.replace(
            plan.models[0],
            source_ids=(state.body_ids[0], lone_surrogate),
        )
        invalid_model_plan = dataclasses.replace(plan, models=(invalid_model,))
        with self.assertRaisesRegex(EncounterContractError, "valid UTF-8"):
            integrate_encounter_segment(state, invalid_model_plan, spec)

    def test_semantic_replay_distinguishes_positive_and_negative_zero(self) -> None:
        state = _state(gm=(5.0e-324, 5.0e-324))
        result = integrate_encounter_segment(
            state,
            _plan(state.body_ids),
            _spec(
                state,
                duration=0.0625,
                step=0.0625,
                minimum_step=0.0625,
                atol=1.0,
            ),
        )
        self.assertEqual(result.proposal_ledger[0].normalized_error.hex(), "0x0.0p+0")
        mutated_record = dataclasses.replace(
            result.proposal_ledger[0], normalized_error=-0.0
        )
        mutated_ledger = (mutated_record,)
        mutated_run = _EncounterRun(
            final_positions=result.final_positions,
            final_velocities=result.final_velocities,
            accepted_signed_substeps=result.accepted_signed_substeps,
            initialization_record=result.initialization_record,
            proposal_ledger=mutated_ledger,
            force_model_ids=result.force_model_ids,
            force_ledger=result.force_ledger,
            counts=result.primary_counts,
        )
        schedule = _schedule_sha256(
            snapshot=result.initial_snapshot,
            plan=result.force_plan,
            spec=result.integration_spec,
            run=mutated_run,
            primary_counts=result.primary_counts,
            replay_counts=result.validation_replay_counts,
            total_counts=result.total_public_call_counts,
            pair_table_sha256=result.pair_table_content_sha256,
        )
        content = _result_sha256(
            snapshot=result.initial_snapshot,
            plan=result.force_plan,
            spec=result.integration_spec,
            run=mutated_run,
            primary_counts=result.primary_counts,
            replay_counts=result.validation_replay_counts,
            total_counts=result.total_public_call_counts,
            pair_table_sha256=result.pair_table_content_sha256,
            schedule_sha256=schedule,
        )
        with self.assertRaisesRegex(EncounterContractError, "semantic replay"):
            dataclasses.replace(
                result,
                proposal_ledger=mutated_ledger,
                schedule_content_sha256=schedule,
                result_content_sha256=content,
            )

    def test_prospective_custody_and_result_length_caps(self) -> None:
        state = _state()
        result = integrate_encounter_segment(
            state,
            _plan(state.body_ids),
            _spec(state, duration=0.0625, step=0.0625, minimum_step=0.0625),
        )
        resources = _resources(
            maximum_witness_transcript_bytes_per_proposal=128,
        )
        usage = _MutableUsage()
        budget = _ExactBudget(
            resources=resources,
            segment=usage,
            operation_limit=resources.maximum_operations_per_proposal,
            gcd_limit=resources.maximum_gcd_iterations_per_proposal,
            transcript_limit=resources.maximum_witness_transcript_bytes_per_proposal,
            label="prospective transcript",
        )
        transcript = _Transcript(
            "PROPOSAL",
            budget=budget,
            domain=ENCOUNTER_EXACT_WITNESS_CHECKSUM_DOMAIN,
            proposal_index=1,
        )
        left = _Rat(1, 1, budget)
        right = _Rat(0, 1, budget)
        with mock.patch.object(_Rat, "record", side_effect=AssertionError):
            with self.assertRaises(EncounterResourceError):
                transcript.comparison(
                    kind="capacity_precheck",
                    pair_index=0,
                    left=left,
                    operator=">",
                    right=right,
                    result=True,
                )

        diagnostic_resources = _resources(
            maximum_witness_ledger_bytes=4096,
        )
        with mock.patch(
            "jxplanetx.engine.encounter._canonical_json",
            side_effect=AssertionError,
        ):
            with self.assertRaises(EncounterResourceError):
                _record_diagnostic_bytes(
                    result.proposal_ledger[0],
                    diagnostic_resources,
                    1,
                )

        tiny_integer_resources = _resources(maximum_integer_bits=1)
        tiny_usage = _MutableUsage()
        tiny_budget = _ExactBudget(
            resources=tiny_integer_resources,
            segment=tiny_usage,
            operation_limit=tiny_integer_resources.maximum_operations_per_proposal,
            gcd_limit=tiny_integer_resources.maximum_gcd_iterations_per_proposal,
            transcript_limit=(
                tiny_integer_resources.maximum_witness_transcript_bytes_per_proposal
            ),
            label="binary64 envelope",
        )
        with self.assertRaises(EncounterResourceError):
            _Rat.from_float(1.5, tiny_budget)

        too_many_proposals = (object(),) * (
            result.integration_spec.maximum_substep_proposals + 1
        )
        with self.assertRaisesRegex(EncounterContractError, "bounded length"):
            dataclasses.replace(result, proposal_ledger=too_many_proposals)
        too_many_steps = (0.0625,) * (
            result.integration_spec.maximum_accepted_substeps + 1
        )
        with self.assertRaisesRegex(EncounterContractError, "bounded length"):
            dataclasses.replace(result, accepted_signed_substeps=too_many_steps)

    def test_error_rejection_then_transactional_acceptance(self) -> None:
        state = _state()
        spec = _spec(
            state,
            duration=0.25,
            step=0.125,
            minimum_step=0.001953125,
            atol=1.0e-14,
            rtol=1.0e-14,
        )
        result = integrate_encounter_segment(state, _plan(state.body_ids), spec)
        self.assertEqual(result.proposal_ledger[0].disposition, "ERROR_REJECTION")
        self.assertEqual(result.primary_counts.error_rejections, 1)
        self.assertEqual(result.primary_counts.rejected_substeps, 1)
        self.assertEqual(
            result.primary_counts.force_evaluations,
            13 * result.primary_counts.completed_rk_attempts,
        )
        self.assertEqual(
            sum(Fraction.from_float(step) for step in result.accepted_signed_substeps),
            Fraction.from_float(spec.duration),
        )

    def test_recoverable_stage_guard_uses_actual_force_count_and_discards_trial(self) -> None:
        import jxplanetx.engine.encounter as encounter

        state = _state()
        spec = _spec(
            state,
            duration=0.25,
            step=0.125,
            minimum_step=0.015625,
            atol=1.0e-8,
            rtol=1.0e-8,
        )
        real_guard = encounter._stage_guard

        def deterministic_guard(**kwargs: object) -> bool:
            positions = kwargs["positions"]
            phase = kwargs["phase"]
            if (
                phase == "stage_1"
                and 0.004 < abs(float(positions[0, 1])) < 0.005
            ):
                return False
            return real_guard(**kwargs)

        with mock.patch.object(encounter, "_stage_guard", side_effect=deterministic_guard):
            result = integrate_encounter_segment(state, _plan(state.body_ids), spec)
        self.assertEqual(result.proposal_ledger[0].disposition, "STAGE_GUARD_ABORT")
        self.assertEqual(result.proposal_ledger[0].actual_force_evaluations, 1)
        self.assertEqual(result.primary_counts.stage_guard_aborts, 1)
        self.assertEqual(
            result.primary_counts.force_evaluations,
            1 + 13 * result.primary_counts.completed_rk_attempts,
        )
        clean_spec = dataclasses.replace(spec, initial_step_magnitude=0.0625)
        clean = integrate_encounter_segment(state, _plan(state.body_ids), clean_spec)
        self.assertEqual(
            result.final_positions.tobytes(), clean.final_positions.tobytes()
        )
        self.assertEqual(
            result.final_velocities.tobytes(), clean.final_velocities.tobytes()
        )

    def test_pair_and_centroid_norm_use_returned_defects(self) -> None:
        from jxplanetx.engine.rkf78 import _RKF78StepResult

        state = _state()
        spec = _spec(state, atol=1.0, rtol=1.0e-12)
        shape = state.positions.shape
        defect = np.ones(shape, dtype=np.float64) * 2.0
        trial = _RKF78StepResult(
            accepted_positions=state.positions.copy(),
            accepted_velocities=state.velocities.copy(),
            accepted_position_carry=np.zeros(shape, dtype=np.float64),
            accepted_velocity_carry=np.zeros(shape, dtype=np.float64),
            embedded_positions=np.ones(shape, dtype=np.float64) * 1.0e100,
            embedded_velocities=np.ones(shape, dtype=np.float64) * -1.0e100,
            position_defect=defect,
            velocity_defect=np.zeros(shape, dtype=np.float64),
        )
        static = mock.Mock(pairs=((0, 1),))
        error = _normalized_error(
            current_positions=state.positions,
            current_velocities=state.velocities,
            trial=trial,
            gravitational_parameters=state.gravitational_parameters,
            static=static,
            spec=spec,
        )
        self.assertAlmostEqual(error, 2.0)

        pair_only_defect = np.zeros(shape, dtype=np.float64)
        pair_only_defect[0, 0] = -1.0
        pair_only_defect[1, 0] = 1.0
        pair_only_trial = dataclasses.replace(
            trial,
            position_defect=pair_only_defect,
        )
        pair_only_error = _normalized_error(
            current_positions=state.positions,
            current_velocities=state.velocities,
            trial=pair_only_trial,
            gravitational_parameters=state.gravitational_parameters,
            static=static,
            spec=spec,
        )
        self.assertAlmostEqual(pair_only_error, 2.0, places=10)

    def test_one_period_eighth_order_convergence(self) -> None:
        state = _state()
        period = 2.0 * math.pi
        errors = []
        for step_count in (16, 32, 64):
            step = period / step_count
            spec = _spec(
                state,
                duration=period,
                endpoint_epoch=10.0,
                step=step,
                minimum_step=step,
                atol=1.0e-6,
                rtol=1.0e-12,
                maximum_substep_proposals=step_count + 4,
                maximum_accepted_substeps=step_count + 4,
                maximum_rejected_substeps=0,
                maximum_consecutive_rejections=0,
                maximum_force_evaluations=13 * (step_count + 4),
            )
            result = integrate_encounter_segment(state, _plan(state.body_ids), spec)
            error = max(
                float(np.max(np.abs(result.final_positions - state.positions))),
                float(np.max(np.abs(result.final_velocities - state.velocities))),
            )
            errors.append(error)
            self.assertEqual(result.primary_counts.accepted_substeps, step_count)
            self.assertEqual(result.primary_counts.force_evaluations, 13 * step_count)
        self.assertGreater(errors[0] / errors[1], 200.0)
        self.assertGreater(errors[1] / errors[2], 200.0)
        self.assertLess(errors[2], 3.0e-12)

    def test_n16_endpoint_exponent_resource_envelope(self) -> None:
        x_coordinates = (
            -128.0,
            -64.0,
            -32.0,
            -16.0,
            -8.0,
            -4.0,
            -2.0,
            -1.0,
            1.0,
            2.0,
            4.0,
            8.0,
            16.0,
            32.0,
            64.0,
            128.0,
        )
        positions = tuple(
            (
                x,
                math.copysign(
                    2.0 ** (-200 + int(math.log2(abs(x)))),
                    x,
                ),
                0.0,
            )
            for x in x_coordinates
        )
        velocities = tuple((-100.0 * x, -100.0 * y, 0.0) for x, y, _ in positions)
        state = _state(
            positions=positions,
            velocities=velocities,
            gm=(1.0e-12,) * 16,
            radii=(0.05,) * 16,
        )
        spec = _spec(
            state,
            duration=0.001,
            endpoint_epoch=1.0,
            step=0.001,
            minimum_step=0.001,
            floor=0.1,
            atol=1.0,
            rtol=1.0e-12,
            maximum_substep_proposals=1,
            maximum_accepted_substeps=1,
            maximum_rejected_substeps=0,
            maximum_consecutive_rejections=0,
            maximum_force_evaluations=13,
        )
        result = integrate_encounter_segment(state, _plan(state.body_ids), spec)
        record = result.proposal_ledger[0]
        self.assertEqual(record.disposition, "ACCEPTED")
        self.assertEqual(record.actual_force_evaluations, 13)
        self.assertEqual(
            (
                record.exact_operations,
                record.rational_operations,
                record.dyadic_operations,
                record.gcd_iterations,
                record.transcript_bytes,
            ),
            (510683, 312638, 198045, 128361, 930833),
        )
        self.assertGreater(record.rational_operations, 0)
        self.assertGreater(record.dyadic_operations, 0)
        self.assertEqual(
            record.exact_operations,
            record.rational_operations + record.dyadic_operations,
        )
        self.assertLessEqual(
            5 * record.exact_operations,
            4 * ENCOUNTER_HARD_MAXIMUM_OPERATIONS_PER_PROPOSAL,
        )
        self.assertLessEqual(
            5 * record.transcript_bytes,
            4 * ENCOUNTER_HARD_MAXIMUM_WITNESS_TRANSCRIPT_BYTES_PER_PROPOSAL,
        )
        self.assertEqual(result.validation_replay_counts, result.primary_counts)

    def test_unequal_three_body_reference_order_and_invariants(self) -> None:
        positions = np.array(
            (
                (
                    float.fromhex("-0x1.055c23bb98e2bp-10"),
                    float.fromhex("-0x1.bc4fd65883e7cp-9"),
                    0.0,
                ),
                (
                    float.fromhex("0x1.ff7d51ee22339p-1"),
                    float.fromhex("-0x1.bc4fd65883e7cp-9"),
                    0.0,
                ),
                (
                    float.fromhex("-0x1.055c23bb98e2bp-10"),
                    float.fromhex("0x1.b2550b4806f14p+0"),
                    0.0,
                ),
            ),
            dtype=np.float64,
        )
        velocities = np.array(
            (
                (
                    float.fromhex("0x1.914efb70e84afp-10"),
                    float.fromhex("-0x1.057d95d5ad6afp-10"),
                    0.0,
                ),
                (
                    float.fromhex("0x1.914efb70e84afp-10"),
                    float.fromhex("0x1.ffbec63b2c623p-1"),
                    0.0,
                ),
                (
                    float.fromhex("-0x1.884b754b1f133p-1"),
                    float.fromhex("-0x1.057d95d5ad6afp-10"),
                    0.0,
                ),
            ),
            dtype=np.float64,
        )
        # Data-only oracle; there is no REBOUND import or runtime dependency.
        # REBOUND 5.1.1 githash 33549d1d50d616a95a6d6a79e5e2c9c3b3730b1f;
        # IAS15 epsilon=1e-14, adaptive_mode=PRS23, G=1,
        # exact_finish_time=1, evaluated at t=10.
        oracle_positions = np.array(
            (
                (
                    float.fromhex("-0x1.45eea50122554p-9"),
                    float.fromhex("0x1.278240c609c7cp-10"),
                    0.0,
                ),
                (
                    float.fromhex("-0x1.b504f052eeddfp-1"),
                    float.fromhex("-0x1.16d23b31c50b4p-1"),
                    0.0,
                ),
                (
                    float.fromhex("0x1.ab8c4937d73f0p+0"),
                    float.fromhex("-0x1.2a582b5106102p-2"),
                    0.0,
                ),
            ),
            dtype=np.float64,
        )
        oracle_velocities = np.array(
            (
                (
                    float.fromhex("-0x1.a1571a71b0354p-11"),
                    float.fromhex("-0x1.6058ded4274d3p-11"),
                    0.0,
                ),
                (
                    float.fromhex("0x1.0e503e6db246ap-1"),
                    float.fromhex("-0x1.ace3dfdb1636dp-1"),
                    0.0,
                ),
                (
                    float.fromhex("0x1.127da2caa79b1p-3"),
                    float.fromhex("0x1.827d54bb224c0p-1"),
                    0.0,
                ),
            ),
            dtype=np.float64,
        )
        gm = np.array((1.0, 0.001, 0.002), dtype=np.float64)
        state = _state(
            positions=tuple(tuple(float(value) for value in row) for row in positions),
            velocities=tuple(tuple(float(value) for value in row) for row in velocities),
            gm=tuple(float(value) for value in gm),
            radii=(0.05, 0.05, 0.05),
        )

        def invariants(
            current_positions: np.ndarray,
            current_velocities: np.ndarray,
        ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            energy = 0.5 * sum(
                gm[index] * np.dot(current_velocities[index], current_velocities[index])
                for index in range(3)
            )
            for left in range(2):
                for right in range(left + 1, 3):
                    energy -= (
                        gm[left]
                        * gm[right]
                        / np.linalg.norm(current_positions[right] - current_positions[left])
                    )
            angular = sum(
                (
                    gm[index]
                    * np.cross(current_positions[index], current_velocities[index])
                    for index in range(3)
                ),
                start=np.zeros(3, dtype=np.float64),
            )
            momentum = sum(
                (gm[index] * current_velocities[index] for index in range(3)),
                start=np.zeros(3, dtype=np.float64),
            )
            centroid = sum(
                (gm[index] * current_positions[index] for index in range(3)),
                start=np.zeros(3, dtype=np.float64),
            ) / np.sum(gm)
            centroid_velocity = momentum / np.sum(gm)
            return float(energy), angular, momentum, centroid, centroid_velocity

        initial = invariants(positions, velocities)
        position_errors: list[float] = []
        velocity_errors: list[float] = []
        energy_limits = (4.1e-9, 8.0e-12, 2.0e-14)
        angular_limits = (1.3e-9, 2.5e-12, 6.0e-15)
        for refinement_index, step_count in enumerate((32, 64, 128)):
            step = 10.0 / step_count
            spec = dataclasses.replace(
                _spec(
                    state,
                    duration=10.0,
                    endpoint_epoch=10.0,
                    step=step,
                    minimum_step=step,
                    floor=0.3,
                    atol=1.0e-7,
                    rtol=1.0e-12,
                    maximum_substep_proposals=step_count,
                    maximum_accepted_substeps=step_count,
                    maximum_rejected_substeps=0,
                    maximum_consecutive_rejections=0,
                    maximum_force_evaluations=13 * step_count,
                ),
                pair_certification_floors=(0.5, 1.0, 0.3),
            )
            result = integrate_encounter_segment(state, _plan(state.body_ids), spec)
            position_errors.append(
                float(
                    np.max(
                        np.linalg.norm(
                            result.final_positions - oracle_positions, axis=1
                        )
                    )
                )
            )
            velocity_errors.append(
                float(
                    np.max(
                        np.linalg.norm(
                            result.final_velocities - oracle_velocities, axis=1
                        )
                    )
                )
            )
            final = invariants(result.final_positions, result.final_velocities)
            self.assertLessEqual(
                abs(final[0] - initial[0]) / abs(initial[0]),
                energy_limits[refinement_index],
            )
            self.assertLessEqual(
                float(np.linalg.norm(final[1] - initial[1]))
                / float(np.linalg.norm(initial[1])),
                angular_limits[refinement_index],
            )
            self.assertLessEqual(float(np.linalg.norm(final[2] - initial[2])), 1.0e-18)
            expected_centroid = initial[3] + 10.0 * initial[4]
            self.assertLessEqual(
                float(np.linalg.norm(final[3] - expected_centroid)), 1.0e-17
            )
            self.assertEqual(result.primary_counts.accepted_substeps, step_count)
            self.assertEqual(result.primary_counts.rejected_substeps, 0)
            self.assertEqual(result.primary_counts.force_evaluations, 13 * step_count)

        for errors in (position_errors, velocity_errors):
            self.assertGreater(errors[0] / errors[1], 200.0)
            self.assertGreater(errors[1] / errors[2], 200.0)
            self.assertLessEqual(errors[2], 2.0e-12)

    def test_initialization_resource_cap_fails_closed(self) -> None:
        state = _state()
        resources = _resources(maximum_initialization_operations=1)
        spec = _spec(state, resources=resources)
        with self.assertRaises(EncounterResourceError):
            integrate_encounter_segment(state, _plan(state.body_ids), spec)

    def test_proposal_resource_cap_fails_closed(self) -> None:
        state = _state()
        resources = _resources(maximum_operations_per_proposal=1000)
        spec = _spec(state, resources=resources)
        with self.assertRaises(EncounterResourceError):
            integrate_encounter_segment(state, _plan(state.body_ids), spec)

    def test_accepted_step_cap_is_prospective(self) -> None:
        import jxplanetx.engine.encounter as encounter

        state = _state()
        spec = _spec(
            state,
            duration=0.125,
            step=0.0625,
            minimum_step=0.0625,
            maximum_substep_proposals=2,
            maximum_accepted_substeps=1,
            maximum_rejected_substeps=0,
            maximum_consecutive_rejections=0,
            maximum_force_evaluations=26,
        )
        with mock.patch.object(
            encounter,
            "_certificate",
            wraps=encounter._certificate,
        ) as certificate:
            with self.assertRaises(EncounterStepLimitError):
                integrate_encounter_segment(state, _plan(state.body_ids), spec)
        self.assertEqual(certificate.call_count, 1)


if __name__ == "__main__":
    unittest.main()
