"""Deterministic Newtonian N-body state, acceleration, and invariants."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .decimal_math import D, Vec3, ZERO3, add, cross, dot, norm, scale, sub


@dataclass(frozen=True)
class Body:
    name: str
    gm: Decimal
    massive: bool = True

    def __post_init__(self) -> None:
        if self.gm < 0:
            raise ValueError("GM cannot be negative")
        if self.massive and self.gm == 0:
            raise ValueError("a massive body must have positive GM")


@dataclass
class State:
    bodies: tuple[Body, ...]
    positions: list[Vec3]
    velocities: list[Vec3]
    time: Decimal = D(0)

    def __post_init__(self) -> None:
        n = len(self.bodies)
        if len(self.positions) != n or len(self.velocities) != n:
            raise ValueError("body, position, and velocity counts must match")
        if len({body.name for body in self.bodies}) != n:
            raise ValueError("body names must be unique")

    def copy(self) -> "State":
        return State(self.bodies, list(self.positions), list(self.velocities), self.time)


def accelerations(state: State) -> list[Vec3]:
    """Return accelerations; massless bodies exert no force on other bodies."""
    result: list[Vec3] = []
    for i, ri in enumerate(state.positions):
        ai = ZERO3
        for j, (source, rj) in enumerate(zip(state.bodies, state.positions)):
            if i == j or not source.massive:
                continue
            delta = sub(rj, ri)
            r2 = dot(delta, delta)
            if r2 == 0:
                raise ZeroDivisionError(f"collision between {state.bodies[i].name} and {source.name}")
            inv_r3 = D(1) / (r2 * r2.sqrt())
            ai = add(ai, scale(source.gm * inv_r3, delta))
        result.append(ai)
    return result


def specific_energy(state: State, reference_gm: Decimal, body_index: int) -> Decimal:
    """Two-body specific energy diagnostic for a test body around a reference."""
    r = norm(state.positions[body_index])
    return dot(state.velocities[body_index], state.velocities[body_index]) / D(2) - reference_gm / r


def specific_angular_momentum(state: State, body_index: int) -> Vec3:
    return cross(state.positions[body_index], state.velocities[body_index])

