"""JX falsification-first celestial-dynamics and evidence framework."""

from __future__ import annotations

from typing import Any

__version__ = "0.6.0rc10"

_GR15_EXPORTS = (
    "GR15ContractError",
    "GR15Error",
    "GR15IntegrationError",
    "GR15Result",
    "GR15Spec",
    "GR15UnavailableError",
    "GR15Workspace",
    "gr15_runtime_identity",
    "integrate_gr15",
    "prepare_gr15_workspace",
)


def __getattr__(name: str) -> Any:
    """Load the NumPy GR15 surface only when a caller requests it."""

    if name in _GR15_EXPORTS:
        from . import gr15

        value = getattr(gr15, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["__version__", *_GR15_EXPORTS]
