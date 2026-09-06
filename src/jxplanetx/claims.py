"""Evidence labels and a deliberately conservative claim state machine."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .gates import GateResult


class EvidenceClass(str, Enum):
    MEASURED = "MEASURED"
    RECONSTRUCTED = "RECONSTRUCTED"
    MODEL_OUTPUT = "MODEL_OUTPUT"
    ASSUMPTION = "ASSUMPTION"
    FORECAST = "FORECAST"
    SPECULATION = "SPECULATION"


class Decision(str, Enum):
    SCREENING_ONLY = "SCREENING_ONLY"
    INVALID = "INVALID"
    CONFLICT = "CONFLICT"
    ELIGIBLE_FOR_REVIEW = "ELIGIBLE_FOR_REVIEW"


class ObservationCapability(str, Enum):
    """Implemented and validated components required for observation claims."""

    FULL_SOLAR_SYSTEM_DYNAMICS = "full_solar_system_dynamics"
    MEASUREMENT_MODEL = "measurement_model"
    OBSERVER_STATION_GEOMETRY = "observer_station_geometry"
    TIME_SCALE_CONVERSION = "time_scale_conversion"
    RELATIVISTIC_SIGNAL_PROPAGATION = "relativistic_signal_propagation"
    MEDIA_CLOCK_CALIBRATION = "media_clock_calibration"
    SIMULTANEOUS_PARAMETER_ESTIMATION = "simultaneous_parameter_estimation"
    COVARIANCE_RANK_DIAGNOSTICS = "covariance_rank_diagnostics"
    RECOVERY_AND_HOLDOUTS = "recovery_and_holdouts"
    INDEPENDENT_REPLICATION = "independent_replication"


OBSERVATION_CAPABILITY_SCHEMA = "jx-observation-capability-evidence/v1"
OBSERVATION_GATE_RESULTS_SCHEMA = "jx-observation-gate-results/v1"
REQUIRED_OBSERVATION_CAPABILITIES_V1 = frozenset(
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
# Compatibility name for the v1 API. The explicit v1 roster above must not
# change merely because a later enum member is added.
REQUIRED_OBSERVATION_CAPABILITIES = REQUIRED_OBSERVATION_CAPABILITIES_V1
ALLOWED_OBSERVATION_GATE_EVIDENCE_CLASSES_V1 = frozenset(
    {
        EvidenceClass.MEASURED,
        EvidenceClass.RECONSTRUCTED,
        EvidenceClass.MODEL_OUTPUT,
    }
)
OBSERVATION_CAPABILITY_API_MIGRATION_REASON = (
    "observation capability API v1 migration: ELIGIBLE_FOR_REVIEW is unavailable "
    "until a future artifact-loading verifier authenticates the referenced "
    "capability, implementation, observation-data, validation, and gate artifacts; "
    "caller-constructed observation capability evidence is non-authoritative"
)


LOCKED_OBSERVATION_GATES = frozenset(
    {
        "zero_signal",
        "injection_recovery",
        "rank",
        "covariance",
        "chronological_holdout",
        "planet_holdout",
        "ephemeris_replication",
        "residual_precision",
        "source_precision",
        "independent_implementation",
        "look_elsewhere",
        "known_force_nuisance_fit",
    }
)


@dataclass(frozen=True)
class ObservationCapabilityEvidence:
    """Caller-supplied v1 descriptor of observation-capability provenance.

    The claim controller can check this descriptor's structure and exact gate
    binding, but it cannot authenticate or load the referenced artifacts.
    Consequently this type is diagnostic input, not current authorization for
    an observational claim.
    """

    schema: str
    validated_capabilities: frozenset[ObservationCapability]
    capability_manifest_sha256: str
    implementation_manifest_sha256: str
    observation_data_manifest_sha256: str
    validation_record_sha256: str
    gate_results_sha256: str


@dataclass(frozen=True)
class ClaimDecision:
    decision: Decision
    reason: str
    missing_gates: tuple[str, ...] = ()
    failed_gates: tuple[str, ...] = ()


def observation_gate_results_sha256(gates: Iterable[GateResult]) -> str:
    """Return an order-independent digest of complete observation-gate rows."""

    rows: list[dict[str, str | bool]] = []
    for gate in gates:
        if not isinstance(gate, GateResult):
            raise TypeError("observation gates must be GateResult instances")
        evidence_class = gate.evidence_class
        if isinstance(evidence_class, Enum):
            evidence_class = evidence_class.value
        row = {
            "name": gate.name,
            "passed": gate.passed,
            "metric": gate.metric,
            "value": gate.value,
            "threshold": gate.threshold,
            "evidence_class": evidence_class,
        }
        if (
            not isinstance(row["name"], str)
            or not isinstance(row["passed"], bool)
            or not all(
                isinstance(row[field], str)
                for field in ("metric", "value", "threshold", "evidence_class")
            )
        ):
            raise TypeError("observation gate fields must use their declared string/bool types")
        rows.append(row)

    rows.sort(
        key=lambda row: json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )
    payload = {
        "schema": OBSERVATION_GATE_RESULTS_SCHEMA,
        "gates": rows,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CAPABILITY_PROVENANCE_FIELDS = (
    "capability_manifest_sha256",
    "implementation_manifest_sha256",
    "observation_data_manifest_sha256",
    "validation_record_sha256",
)


def _observation_capability_issue(
    evidence: ObservationCapabilityEvidence | None,
    gates: tuple[GateResult, ...],
) -> str | None:
    if not isinstance(evidence, ObservationCapabilityEvidence):
        return "typed observation capability evidence is absent"
    if evidence.schema != OBSERVATION_CAPABILITY_SCHEMA:
        return "observation capability evidence schema is unsupported"
    if not isinstance(evidence.validated_capabilities, frozenset) or any(
        not isinstance(capability, ObservationCapability)
        for capability in evidence.validated_capabilities
    ):
        return "validated capabilities are not a typed frozen set"
    missing_capabilities = REQUIRED_OBSERVATION_CAPABILITIES_V1 - evidence.validated_capabilities
    if missing_capabilities:
        names = ", ".join(sorted(capability.value for capability in missing_capabilities))
        return f"required observation capabilities are missing: {names}"

    for field in _CAPABILITY_PROVENANCE_FIELDS:
        digest = getattr(evidence, field)
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            return f"{field} is not a lowercase SHA-256 digest"
        if digest == "0" * 64:
            return f"{field} is a placeholder digest"

    if len({gate.name for gate in gates}) != len(gates):
        return "observation gate names are not unique"
    try:
        expected_gate_digest = observation_gate_results_sha256(gates)
    except (TypeError, ValueError):
        return "observation gate rows are not canonical"
    if evidence.gate_results_sha256 != expected_gate_digest:
        return "capability evidence is not bound to the supplied observation gates"
    return None


def _observation_gate_row_issue(gates: tuple[GateResult, ...]) -> str | None:
    if len({gate.name for gate in gates}) != len(gates):
        return "observation gate names are not unique"
    for gate in gates:
        try:
            evidence_class = EvidenceClass(gate.evidence_class)
        except (TypeError, ValueError):
            return f"observation gate {gate.name!r} has an unknown evidence class"
        if evidence_class not in ALLOWED_OBSERVATION_GATE_EVIDENCE_CLASSES_V1:
            return (
                f"observation gate {gate.name!r} uses ineligible evidence class "
                f"{evidence_class.value}"
            )
    return None


def assess_claim(
    gates: Iterable[GateResult],
    observational: bool = False,
    *,
    observation_capability: ObservationCapabilityEvidence | None = None,
) -> ClaimDecision:
    gate_rows = tuple(gates)
    # Inspect every row before collapsing names. A later duplicate PASS must
    # never conceal an earlier failed row with the same name.
    failed = tuple(sorted({gate.name for gate in gate_rows if not gate.passed}))
    if failed:
        return ClaimDecision(Decision.INVALID, "one or more supplied validity gates failed", failed_gates=failed)
    if not observational:
        return ClaimDecision(Decision.SCREENING_ONLY, "numerical/model gates cannot establish an observed source")
    gate_row_issue = _observation_gate_row_issue(gate_rows)
    if gate_row_issue is not None:
        return ClaimDecision(
            Decision.SCREENING_ONLY,
            f"{gate_row_issue}; {OBSERVATION_CAPABILITY_API_MIGRATION_REASON}",
        )
    indexed = {gate.name: gate for gate in gate_rows}
    missing = tuple(sorted(LOCKED_OBSERVATION_GATES - indexed.keys()))
    if missing:
        return ClaimDecision(
            Decision.SCREENING_ONLY,
            "locked observation gates are incomplete; direction and source claims remain blocked; "
            f"{OBSERVATION_CAPABILITY_API_MIGRATION_REASON}",
            missing_gates=missing,
        )
    capability_issue = _observation_capability_issue(observation_capability, gate_rows)
    if capability_issue is not None:
        return ClaimDecision(
            Decision.SCREENING_ONLY,
            f"{capability_issue}; {OBSERVATION_CAPABILITY_API_MIGRATION_REASON}",
        )
    # Deliberately no current path to ELIGIBLE_FOR_REVIEW. A future change must
    # load and authenticate the referenced artifacts before it may add one.
    return ClaimDecision(
        Decision.SCREENING_ONLY,
        OBSERVATION_CAPABILITY_API_MIGRATION_REASON,
    )
