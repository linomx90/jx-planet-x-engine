"""Evidence labels and a deliberately conservative claim state machine."""

from __future__ import annotations

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
class ClaimDecision:
    decision: Decision
    reason: str
    missing_gates: tuple[str, ...] = ()
    failed_gates: tuple[str, ...] = ()


def assess_claim(gates: Iterable[GateResult], observational: bool = False) -> ClaimDecision:
    indexed = {g.name: g for g in gates}
    failed = tuple(sorted(name for name, gate in indexed.items() if not gate.passed))
    if failed:
        return ClaimDecision(Decision.INVALID, "one or more supplied validity gates failed", failed_gates=failed)
    if not observational:
        return ClaimDecision(Decision.SCREENING_ONLY, "numerical/model gates cannot establish an observed source")
    missing = tuple(sorted(LOCKED_OBSERVATION_GATES - indexed.keys()))
    if missing:
        return ClaimDecision(
            Decision.SCREENING_ONLY,
            "locked observation gates are incomplete; direction and source claims remain blocked",
            missing_gates=missing,
        )
    return ClaimDecision(
        Decision.ELIGIBLE_FOR_REVIEW,
        "all encoded gates passed; external scientific review is still required",
    )

