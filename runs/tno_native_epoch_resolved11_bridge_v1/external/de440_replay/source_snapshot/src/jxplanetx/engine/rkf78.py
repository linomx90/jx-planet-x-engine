"""Classic 13-stage Fehlberg RK 7(8) step for backend-native arrays.

The tableau is Fehlberg's formula from NASA-TR-R-287.  Coefficients are
spelled as exact integer ratios below and converted once to Python binary64;
the selected array backend then materializes them as native float64 scalars.
The accepted solution is the eighth-order (hatted) formula using stages 11
and 12 (zero based).  The embedded seventh-order formula instead uses stages
0 and 10.  The signed defect is always accepted eighth minus embedded seventh.

This module deliberately owns no controller and exposes no public step
function.  Checkpoint clipping, error normalization, domain validation, and
claim controls belong to :mod:`jxplanetx.engine.trajectory`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .backends import ArrayBackend
from .trajectory_contracts import (
    ADAPTIVE_RKF78_METHOD_ID,
    RKF78_ACCEPTED_ORDER as CONTRACT_ACCEPTED_ORDER,
    RKF78_ACCEPTED_STATE_ACCUMULATION,
    RKF78_DEFECT_ORIENTATION as CONTRACT_DEFECT_ORIENTATION,
    RKF78_EMBEDDED_ORDER as CONTRACT_EMBEDDED_ORDER,
    RKF78_SOURCE_REPORT,
    RKF78_STAGE_COUNT as CONTRACT_STAGE_COUNT,
)


RKF78_METHOD_ID = ADAPTIVE_RKF78_METHOD_ID
RKF78_TABLEAU_ID = "NASA_TR_R_287_FEHLBERG_7_8_13_STAGE"
RKF78_TABLEAU_SOURCE = RKF78_SOURCE_REPORT
RKF78_STAGE_COUNT = CONTRACT_STAGE_COUNT
RKF78_ACCEPTED_ORDER = CONTRACT_ACCEPTED_ORDER
RKF78_EMBEDDED_ORDER = CONTRACT_EMBEDDED_ORDER
RKF78_DEFECT_ORIENTATION = CONTRACT_DEFECT_ORIENTATION


def _q(numerator: int, denominator: int = 1) -> float:
    """Convert one exact rational coefficient to Python binary64."""

    return numerator / denominator


# Stage abscissae, zero based.  Stages 10 and 12 are endpoint evaluations;
# stage 11 returns to the beginning of the step as in Fehlberg's tableau.
_C = (
    _q(0),
    _q(2, 27),
    _q(1, 9),
    _q(1, 6),
    _q(5, 12),
    _q(1, 2),
    _q(5, 6),
    _q(1, 6),
    _q(2, 3),
    _q(1, 3),
    _q(1),
    _q(0),
    _q(1),
)


# Strict lower triangle of the explicit tableau.  Each row has exactly the
# coefficients needed by its stage; omitted coefficients are exact zeros.
_A = (
    (),
    (_q(2, 27),),
    (_q(1, 36), _q(1, 12)),
    (_q(1, 24), _q(0), _q(1, 8)),
    (_q(5, 12), _q(0), _q(-25, 16), _q(25, 16)),
    (_q(1, 20), _q(0), _q(0), _q(1, 4), _q(1, 5)),
    (
        _q(-25, 108),
        _q(0),
        _q(0),
        _q(125, 108),
        _q(-65, 27),
        _q(125, 54),
    ),
    (
        _q(31, 300),
        _q(0),
        _q(0),
        _q(0),
        _q(61, 225),
        _q(-2, 9),
        _q(13, 900),
    ),
    (
        _q(2),
        _q(0),
        _q(0),
        _q(-53, 6),
        _q(704, 45),
        _q(-107, 9),
        _q(67, 90),
        _q(3),
    ),
    (
        _q(-91, 108),
        _q(0),
        _q(0),
        _q(23, 108),
        _q(-976, 135),
        _q(311, 54),
        _q(-19, 60),
        _q(17, 6),
        _q(-1, 12),
    ),
    (
        _q(2383, 4100),
        _q(0),
        _q(0),
        _q(-341, 164),
        _q(4496, 1025),
        _q(-301, 82),
        _q(2133, 4100),
        _q(45, 82),
        _q(45, 164),
        _q(18, 41),
    ),
    (
        _q(3, 205),
        _q(0),
        _q(0),
        _q(0),
        _q(0),
        _q(-6, 41),
        _q(-3, 205),
        _q(-3, 41),
        _q(3, 41),
        _q(6, 41),
        _q(0),
    ),
    (
        _q(-1777, 4100),
        _q(0),
        _q(0),
        _q(-341, 164),
        _q(4496, 1025),
        _q(-289, 82),
        _q(2193, 4100),
        _q(51, 82),
        _q(33, 164),
        _q(12, 41),
        _q(0),
        _q(1),
    ),
)


_COMMON_WEIGHTS = {
    5: _q(34, 105),
    6: _q(9, 35),
    7: _q(9, 35),
    8: _q(9, 280),
    9: _q(9, 280),
}
_B_EIGHTH = tuple(
    _COMMON_WEIGHTS.get(index, _q(41, 840) if index in {11, 12} else _q(0))
    for index in range(RKF78_STAGE_COUNT)
)
_B_SEVENTH = tuple(
    _COMMON_WEIGHTS.get(index, _q(41, 840) if index in {0, 10} else _q(0))
    for index in range(RKF78_STAGE_COUNT)
)
_B_DEFECT = tuple(
    eighth - seventh for eighth, seventh in zip(_B_EIGHTH, _B_SEVENTH)
)


Derivative = Callable[[float, Any, Any], tuple[Any, Any]]


@dataclass(frozen=True, eq=False)
class _RKF78StepResult:
    accepted_positions: Any
    accepted_velocities: Any
    accepted_position_carry: Any
    accepted_velocity_carry: Any
    embedded_positions: Any
    embedded_velocities: Any
    position_defect: Any
    velocity_defect: Any


def _weighted_sum(
    backend: ArrayBackend,
    stages: list[Any],
    coefficients: tuple[float, ...],
    template: Any,
) -> Any:
    """Accumulate one tableau row in its fixed, published stage order."""

    total = backend.xp.zeros_like(template, dtype=backend.xp.float64)
    for stage, coefficient in zip(stages, coefficients):
        if coefficient != 0.0:
            total = total + backend.xp.float64(coefficient) * stage
    return total


def _stage_epoch(epoch: float, endpoint_epoch: float, step: float, c: float) -> float:
    """Preserve exact host endpoint identities for the c=0 and c=1 stages."""

    if c == 0.0:
        return epoch
    if c == 1.0:
        return endpoint_epoch
    return epoch + c * step


def _compensated_add(state: Any, delta: Any, carry: Any) -> tuple[Any, Any]:
    """Add one accepted increment with the fixed componentwise Kahan order."""

    adjusted = delta - carry
    updated = state + adjusted
    next_carry = (updated - state) - adjusted
    return updated, next_carry


def _validate_carry(
    backend: ArrayBackend,
    carry: Any,
    template: Any,
    label: str,
) -> Any:
    """Require one finite backend-native binary64 carry array."""

    checked = backend.require_native_array(carry, label)
    if checked.dtype != backend.float64 or checked.shape != template.shape:
        raise ValueError(f"{label} must match the state shape and float64 dtype")
    if backend.scalar_bool(
        backend.xp.any(~backend.xp.isfinite(checked)),
        f"{label} finiteness check",
    ):
        raise ValueError(f"{label} must contain only finite values")
    return checked


def _rkf78_step(
    *,
    backend: ArrayBackend,
    epoch: float,
    endpoint_epoch: float,
    step: float,
    positions: Any,
    velocities: Any,
    position_carry: Any,
    velocity_carry: Any,
    derivative: Derivative,
) -> _RKF78StepResult:
    """Return one classic Fehlberg 7(8) trial step on native arrays."""

    position_carry = _validate_carry(
        backend, position_carry, positions, "position_carry"
    )
    velocity_carry = _validate_carry(
        backend, velocity_carry, velocities, "velocity_carry"
    )
    position_stages: list[Any] = []
    velocity_stages: list[Any] = []
    native_step = backend.xp.float64(step)

    for stage_index in range(RKF78_STAGE_COUNT):
        if stage_index == 0:
            stage_positions = positions
            stage_velocities = velocities
        else:
            row = _A[stage_index]
            position_increment = _weighted_sum(
                backend, position_stages, row, positions
            )
            velocity_increment = _weighted_sum(
                backend, velocity_stages, row, velocities
            )
            stage_positions = positions + native_step * position_increment
            stage_velocities = velocities + native_step * velocity_increment

        position_derivative, velocity_derivative = derivative(
            _stage_epoch(epoch, endpoint_epoch, step, _C[stage_index]),
            stage_positions,
            stage_velocities,
        )
        position_stages.append(position_derivative)
        velocity_stages.append(velocity_derivative)

    eighth_position_increment = _weighted_sum(
        backend, position_stages, _B_EIGHTH, positions
    )
    eighth_velocity_increment = _weighted_sum(
        backend, velocity_stages, _B_EIGHTH, velocities
    )
    seventh_position_increment = _weighted_sum(
        backend, position_stages, _B_SEVENTH, positions
    )
    seventh_velocity_increment = _weighted_sum(
        backend, velocity_stages, _B_SEVENTH, velocities
    )
    position_defect_increment = _weighted_sum(
        backend, position_stages, _B_DEFECT, positions
    )
    velocity_defect_increment = _weighted_sum(
        backend, velocity_stages, _B_DEFECT, velocities
    )
    position_delta = native_step * eighth_position_increment
    velocity_delta = native_step * eighth_velocity_increment
    accepted_positions, accepted_position_carry = _compensated_add(
        positions, position_delta, position_carry
    )
    accepted_velocities, accepted_velocity_carry = _compensated_add(
        velocities, velocity_delta, velocity_carry
    )

    return _RKF78StepResult(
        accepted_positions=accepted_positions,
        accepted_velocities=accepted_velocities,
        accepted_position_carry=accepted_position_carry,
        accepted_velocity_carry=accepted_velocity_carry,
        embedded_positions=positions + native_step * seventh_position_increment,
        embedded_velocities=velocities + native_step * seventh_velocity_increment,
        position_defect=native_step * position_defect_increment,
        velocity_defect=native_step * velocity_defect_increment,
    )


__all__ = [
    "RKF78_ACCEPTED_ORDER",
    "RKF78_ACCEPTED_STATE_ACCUMULATION",
    "RKF78_DEFECT_ORIENTATION",
    "RKF78_EMBEDDED_ORDER",
    "RKF78_METHOD_ID",
    "RKF78_STAGE_COUNT",
    "RKF78_TABLEAU_ID",
    "RKF78_TABLEAU_SOURCE",
]
