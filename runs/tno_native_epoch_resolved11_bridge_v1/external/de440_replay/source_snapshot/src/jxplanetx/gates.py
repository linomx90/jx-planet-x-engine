"""Independent numerical gates for the JX engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, localcontext

from .decimal_math import D, norm, precision_context, sin_cos, vec
from .dynamics import Body, State, specific_angular_momentum, specific_energy
from .yoshida6 import COEFFICIENTS, StepStats, integrate


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    metric: str
    value: str
    threshold: str
    evidence_class: str = "MODEL_OUTPUT"

    def as_dict(self) -> dict[str, str | bool]:
        return asdict(self)


def _oscillator_step(q: Decimal, p: Decimal, h: Decimal) -> tuple[Decimal, Decimal]:
    """Yoshida-6 step for H=(p^2+q^2)/2, independent of N-body code."""
    from .yoshida6 import COEFFICIENTS

    for c in COEFFICIENTS:
        p -= c * h * q / D(2)
        q += c * h * p
        p -= c * h * q / D(2)
    return q, p


def oscillator_error(h: Decimal, duration: Decimal = D(1)) -> Decimal:
    steps = int(duration / h)
    q, p = D(1), D(0)
    for _ in range(steps):
        q, p = _oscillator_step(q, p, h)
    # Phase-space norm preservation error is too favorable for a symplectic
    # oscillator map. Compare against high-precision analytic sin/cos instead.
    sin_exact, cos_exact = sin_cos(duration)
    q_exact = cos_exact
    p_exact = -sin_exact
    return ((q - q_exact) ** 2 + (p - p_exact) ** 2).sqrt()


def convergence_gate(decimal_digits: int = 80) -> GateResult:
    with localcontext(precision_context(decimal_digits)):
        e_coarse = oscillator_error(D("0.05"))
        e_fine = oscillator_error(D("0.025"))
        ratio = e_coarse / e_fine
        passed = D("60") <= ratio <= D("68")
        return GateResult(
            "yoshida6_analytic_convergence",
            passed,
            "error(h)/error(h/2)",
            str(ratio),
            "60 <= ratio <= 68 (sixth-order expectation: 64)",
        )


def coefficient_gate(decimal_digits: int = 80) -> GateResult:
    with localcontext(precision_context(decimal_digits)):
        total = sum(COEFFICIENTS, D(0))
        error = abs(total - D(1))
        return GateResult(
            "yoshida6_coefficient_closure",
            error == 0,
            "abs(sum(coefficients)-1)",
            str(error),
            "= 0 at stored decimal precision",
        )


def two_body_gate(decimal_digits: int = 80, steps: int = 1000) -> list[GateResult]:
    with localcontext(precision_context(decimal_digits)):
        sun = Body("Sun", D(1), True)
        tracer = Body("Tracer", D(0), False)
        state = State(
            (sun, tracer),
            [vec((0, 0, 0)), vec((1, 0, 0))],
            [vec((0, 0, 0)), vec((0, 1, 0))],
        )
        e0 = specific_energy(state, sun.gm, 1)
        h0 = norm(specific_angular_momentum(state, 1))
        stats = StepStats()
        integrate(state, D("0.01"), steps, stats)
        e1 = specific_energy(state, sun.gm, 1)
        h1 = norm(specific_angular_momentum(state, 1))
        rel_e = abs((e1 - e0) / e0)
        rel_h = abs((h1 - h0) / h0)
        expected_evaluations = 8 * steps
        return [
            GateResult("two_body_energy", rel_e < D("1e-12"), "relative drift", str(rel_e), "< 1e-12"),
            GateResult("two_body_angular_momentum", rel_h < D("1e-60"), "relative drift", str(rel_h), "< 1e-60"),
            GateResult(
                "optimized_force_count",
                stats.force_evaluations == expected_evaluations,
                "force evaluations",
                str(stats.force_evaluations),
                str(expected_evaluations),
            ),
        ]


def run_core_gates(decimal_digits: int = 80) -> list[GateResult]:
    return [coefficient_gate(decimal_digits), convergence_gate(decimal_digits), *two_body_gate(decimal_digits)]
