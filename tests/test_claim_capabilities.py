from __future__ import annotations

import hashlib
import unittest
from dataclasses import replace

from jxplanetx.claims import (
    LOCKED_OBSERVATION_GATES,
    OBSERVATION_CAPABILITY_API_MIGRATION_REASON,
    OBSERVATION_CAPABILITY_SCHEMA,
    REQUIRED_OBSERVATION_CAPABILITIES,
    REQUIRED_OBSERVATION_CAPABILITIES_V1,
    Decision,
    ObservationCapability,
    ObservationCapabilityEvidence,
    assess_claim,
    observation_gate_results_sha256,
)
from jxplanetx.gates import GateResult


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _passed_observation_gates() -> list[GateResult]:
    return [
        GateResult(
            name,
            True,
            "locked validation metric",
            "pass",
            "predeclared threshold",
            "RECONSTRUCTED",
        )
        for name in sorted(LOCKED_OBSERVATION_GATES)
    ]


def _capability_evidence(gates: list[GateResult]) -> ObservationCapabilityEvidence:
    return ObservationCapabilityEvidence(
        schema=OBSERVATION_CAPABILITY_SCHEMA,
        validated_capabilities=REQUIRED_OBSERVATION_CAPABILITIES,
        capability_manifest_sha256=_digest("capability manifest"),
        implementation_manifest_sha256=_digest("implementation manifest"),
        observation_data_manifest_sha256=_digest("observation data manifest"),
        validation_record_sha256=_digest("validation record"),
        gate_results_sha256=observation_gate_results_sha256(gates),
    )


class ObservationCapabilityClaimTests(unittest.TestCase):
    def test_v1_capability_roster_is_explicitly_pinned(self):
        expected = frozenset(
            {
                ObservationCapability.FULL_SOLAR_SYSTEM_DYNAMICS,
                ObservationCapability.MEASUREMENT_MODEL,
                ObservationCapability.OBSERVER_STATION_GEOMETRY,
                ObservationCapability.TIME_SCALE_CONVERSION,
                ObservationCapability.RELATIVISTIC_SIGNAL_PROPAGATION,
                ObservationCapability.MEDIA_CLOCK_CALIBRATION,
                ObservationCapability.SIMULTANEOUS_PARAMETER_ESTIMATION,
                ObservationCapability.COVARIANCE_RANK_DIAGNOSTICS,
                ObservationCapability.RECOVERY_AND_HOLDOUTS,
                ObservationCapability.INDEPENDENT_REPLICATION,
            }
        )

        self.assertEqual(REQUIRED_OBSERVATION_CAPABILITIES_V1, expected)
        self.assertIs(REQUIRED_OBSERVATION_CAPABILITIES, REQUIRED_OBSERVATION_CAPABILITIES_V1)

    def test_name_matched_gate_results_alone_cannot_authorize_claim(self):
        gates = _passed_observation_gates()

        decision = assess_claim(gates, observational=True)

        self.assertEqual(decision.decision, Decision.SCREENING_ONLY)
        self.assertIn("capability evidence is absent", decision.reason)
        self.assertIn("API v1 migration", decision.reason)
        self.assertIn("artifact-loading verifier", decision.reason)

    def test_incomplete_observation_call_reports_api_migration(self):
        decision = assess_claim([], observational=True)

        self.assertEqual(decision.decision, Decision.SCREENING_ONLY)
        self.assertEqual(set(decision.missing_gates), set(LOCKED_OBSERVATION_GATES))
        self.assertIn("API v1 migration", decision.reason)
        self.assertIn("artifact-loading verifier", decision.reason)

    def test_untyped_capability_mapping_cannot_authorize_claim(self):
        gates = _passed_observation_gates()
        untyped = {
            "schema": OBSERVATION_CAPABILITY_SCHEMA,
            "validated_capabilities": REQUIRED_OBSERVATION_CAPABILITIES,
            "gate_results_sha256": observation_gate_results_sha256(gates),
        }

        decision = assess_claim(  # type: ignore[arg-type]
            gates,
            observational=True,
            observation_capability=untyped,
        )

        self.assertEqual(decision.decision, Decision.SCREENING_ONLY)
        self.assertIn("typed observation capability evidence is absent", decision.reason)

    def test_capability_record_requires_every_typed_capability(self):
        gates = _passed_observation_gates()
        evidence = _capability_evidence(gates)
        evidence = replace(
            evidence,
            validated_capabilities=frozenset(list(REQUIRED_OBSERVATION_CAPABILITIES)[:-1]),
        )

        decision = assess_claim(gates, observational=True, observation_capability=evidence)

        self.assertEqual(decision.decision, Decision.SCREENING_ONLY)
        self.assertIn("required observation capabilities are missing", decision.reason)

    def test_placeholder_provenance_digest_cannot_authorize_claim(self):
        gates = _passed_observation_gates()
        evidence = replace(
            _capability_evidence(gates),
            validation_record_sha256="0" * 64,
        )

        decision = assess_claim(gates, observational=True, observation_capability=evidence)

        self.assertEqual(decision.decision, Decision.SCREENING_ONLY)
        self.assertIn("placeholder digest", decision.reason)

    def test_capability_record_is_bound_to_complete_gate_rows(self):
        gates = _passed_observation_gates()
        evidence = _capability_evidence(gates)
        changed = list(gates)
        changed[0] = replace(changed[0], value="different passing result")

        decision = assess_claim(changed, observational=True, observation_capability=evidence)

        self.assertEqual(decision.decision, Decision.SCREENING_ONLY)
        self.assertIn("not bound to the supplied observation gates", decision.reason)

    def test_duplicate_gate_names_cannot_authorize_claim(self):
        gates = _passed_observation_gates()
        gates.append(gates[0])
        evidence = _capability_evidence(gates)

        decision = assess_claim(gates, observational=True, observation_capability=evidence)

        self.assertEqual(decision.decision, Decision.SCREENING_ONLY)
        self.assertIn("gate names are not unique", decision.reason)

    def test_failed_duplicate_cannot_be_hidden_by_later_passing_duplicate(self):
        gates = _passed_observation_gates()
        failed = replace(gates[0], passed=False, value="fail")
        gates = [failed, *gates]
        evidence = _capability_evidence(gates)

        decision = assess_claim(gates, observational=True, observation_capability=evidence)

        self.assertEqual(decision.decision, Decision.INVALID)
        self.assertEqual(decision.failed_gates, (failed.name,))

    def test_unknown_gate_evidence_class_cannot_authorize_claim(self):
        gates = _passed_observation_gates()
        gates[0] = replace(gates[0], evidence_class="OBSERVED")
        evidence = _capability_evidence(gates)

        decision = assess_claim(gates, observational=True, observation_capability=evidence)

        self.assertEqual(decision.decision, Decision.SCREENING_ONLY)
        self.assertIn("unknown evidence class", decision.reason)

    def test_speculative_gate_evidence_cannot_authorize_claim(self):
        gates = _passed_observation_gates()
        gates[0] = replace(gates[0], evidence_class="SPECULATION")
        evidence = _capability_evidence(gates)

        decision = assess_claim(gates, observational=True, observation_capability=evidence)

        self.assertEqual(decision.decision, Decision.SCREENING_ONLY)
        self.assertIn("ineligible evidence class SPECULATION", decision.reason)

    def test_complete_caller_constructed_evidence_remains_screening_only(self):
        gates = _passed_observation_gates()
        evidence = _capability_evidence(gates)

        decision = assess_claim(
            reversed(gates),
            observational=True,
            observation_capability=evidence,
        )

        self.assertEqual(decision.decision, Decision.SCREENING_ONLY)
        self.assertEqual(decision.reason, OBSERVATION_CAPABILITY_API_MIGRATION_REASON)
        self.assertIn("caller-constructed", decision.reason)

    def test_failed_gate_remains_invalid_with_capability_evidence(self):
        gates = _passed_observation_gates()
        gates[0] = replace(gates[0], passed=False, value="fail")
        evidence = _capability_evidence(gates)

        decision = assess_claim(gates, observational=True, observation_capability=evidence)

        self.assertEqual(decision.decision, Decision.INVALID)
        self.assertEqual(decision.failed_gates, (gates[0].name,))

    def test_nonobservational_call_remains_screening_only(self):
        gate = GateResult("numerical", True, "metric", "1", "<= 1")

        decision = assess_claim(iter([gate]))

        self.assertEqual(decision.decision, Decision.SCREENING_ONLY)
        self.assertEqual(
            decision.reason,
            "numerical/model gates cannot establish an observed source",
        )


if __name__ == "__main__":
    unittest.main()
