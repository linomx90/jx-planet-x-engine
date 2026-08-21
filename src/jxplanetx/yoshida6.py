"""Sixth-order symmetric Yoshida composition with eight force evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .decimal_math import D, add, scale
from .dynamics import State, accelerations

# Yoshida's 7-stage sixth-order composition. Decimal strings prevent hidden
# binary-float rounding. Coefficients are symmetric and sum to one.
W1 = D("0.784513610477557263819497633866349875")
W2 = D("0.235573213359358133684793182978534602")
W3 = D("-1.17767998417887100694641568096431573")
W0 = D("1.315186320683911218884249728238862506")
COEFFICIENTS: tuple[Decimal, ...] = (W1, W2, W3, W0, W3, W2, W1)


@dataclass
class StepStats:
    macro_steps: int = 0
    force_evaluations: int = 0


def step(state: State, dt: Decimal, stats: StepStats | None = None) -> None:
    """Advance one macro-step using merged adjacent KDK kicks."""
    if dt == 0:
        raise ValueError("time step cannot be zero")

    acc = accelerations(state)
    if stats is not None:
        stats.force_evaluations += 1

    first_kick = dt * COEFFICIENTS[0] / D(2)
    state.velocities = [add(v, scale(first_kick, a)) for v, a in zip(state.velocities, acc)]

    for stage, coefficient in enumerate(COEFFICIENTS):
        drift = dt * coefficient
        state.positions = [add(r, scale(drift, v)) for r, v in zip(state.positions, state.velocities)]

        acc = accelerations(state)
        if stats is not None:
            stats.force_evaluations += 1

        if stage == len(COEFFICIENTS) - 1:
            kick = dt * coefficient / D(2)
        else:
            kick = dt * (coefficient + COEFFICIENTS[stage + 1]) / D(2)
        state.velocities = [add(v, scale(kick, a)) for v, a in zip(state.velocities, acc)]

    state.time += dt
    if stats is not None:
        stats.macro_steps += 1


def integrate(state: State, dt: Decimal, steps: int, stats: StepStats | None = None) -> State:
    if steps < 0:
        raise ValueError("steps cannot be negative")
    for _ in range(steps):
        step(state, dt, stats)
    return state
