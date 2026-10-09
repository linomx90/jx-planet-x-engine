"""Minimal arbitrary-precision vector operations based on decimal.Decimal."""

from __future__ import annotations

from decimal import Decimal, Context, getcontext
from typing import Iterable, TypeAlias

Vec3: TypeAlias = tuple[Decimal, Decimal, Decimal]


def D(value: str | int | Decimal) -> Decimal:
    """Construct a Decimal without passing through binary floating point."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def precision_context(decimal_digits: int) -> Context:
    if decimal_digits < 34:
        raise ValueError("JX precision must be at least 34 decimal digits")
    return Context(prec=decimal_digits)


def vec(values: Iterable[str | int | Decimal]) -> Vec3:
    x = tuple(D(v) for v in values)
    if len(x) != 3:
        raise ValueError("a JX vector must have exactly three components")
    return x  # type: ignore[return-value]


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(s: Decimal, a: Vec3) -> Vec3:
    return (s * a[0], s * a[1], s * a[2])


def dot(a: Vec3, b: Vec3) -> Decimal:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a: Vec3) -> Decimal:
    return dot(a, a).sqrt()


def sin_cos(x: Decimal) -> tuple[Decimal, Decimal]:
    """Return sin(x), cos(x) without binary floats.

    The direct Taylor recurrence is intentionally small and auditable. It is
    suitable for the analytic validation angles used by the engine (|x| <= 1).
    Wider argument reduction will be added with the ephemeris time module.
    """
    if abs(x) > 1:
        raise ValueError("current Decimal sin/cos kernel requires |x| <= 1")
    epsilon = D(10) ** (-(getcontext().prec - 4))
    x2 = x * x
    sin_term = x
    cos_term = D(1)
    sin_sum = sin_term
    cos_sum = cos_term
    n = 1
    while True:
        sin_term *= -x2 / D((2 * n) * (2 * n + 1))
        cos_term *= -x2 / D((2 * n - 1) * (2 * n))
        sin_sum += sin_term
        cos_sum += cos_term
        if abs(sin_term) < epsilon and abs(cos_term) < epsilon:
            return +sin_sum, +cos_sum
        n += 1
        if n > 1000:
            raise ArithmeticError("Decimal sin/cos series did not converge")


ZERO3: Vec3 = (D(0), D(0), D(0))
