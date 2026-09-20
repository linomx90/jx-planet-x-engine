"""Explicit array-backend selection for the JX force evaluator.

The force runtime deliberately does not coerce arrays.  In particular, asking
for the CuPy backend while supplying NumPy arrays is an error rather than an
implicit host-to-device transfer.  The inverse is rejected for the same
reason.  Callers therefore retain control of allocation, device placement,
and transfers at every public boundary.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType
from typing import Any


class BackendError(ValueError):
    """The requested backend contract is invalid."""


class BackendUnavailableError(RuntimeError):
    """An explicitly requested array backend cannot be used."""


class BackendArrayError(TypeError):
    """An input is not a native array of the explicitly selected backend."""


def _normalized_backend_name(requested: object) -> str:
    """Return a strict backend name without choosing a default.

    ``BackendSpec`` is defined by the public contracts module.  Keeping this
    small adapter attribute-based avoids importing the contracts layer here
    (and, consequently, avoids a circular dependency).
    """

    if type(requested) is str:
        name = requested
    else:
        name = None
        for attribute in ("backend_id", "name", "backend", "kind"):
            candidate = getattr(requested, attribute, None)
            if candidate is not None:
                name = getattr(candidate, "value", candidate)
                break
    if type(name) is not str or name.strip() != name or not name:
        raise BackendError("backend must be an explicit 'numpy' or 'cupy' request")
    if name not in {"numpy", "cupy"}:
        raise BackendError(f"unsupported backend {name!r}; expected 'numpy' or 'cupy'")
    return name


@dataclass(frozen=True)
class ArrayBackend:
    """Resolved native array implementation.

    The module object is intentionally retained internally.  Results expose
    only ``name`` so that metadata never depends on a module representation.
    """

    name: str
    xp: ModuleType
    array_type: type
    device: str
    device_index: int | None = None

    @property
    def float64(self) -> Any:
        return self.xp.float64

    @property
    def bool_(self) -> Any:
        return self.xp.bool_

    def require_native_array(self, value: object, label: str) -> Any:
        """Reject non-native inputs instead of performing a conversion."""

        if not isinstance(value, self.array_type):
            raise BackendArrayError(
                f"{label} must be a native {self.name} array; "
                "JX never performs an implicit host/device transfer"
            )
        if self.name == "cupy" and int(value.device.id) != self.device_index:
            raise BackendArrayError(
                f"{label} is on CUDA device {int(value.device.id)}, but the explicit "
                f"backend request selected device {self.device_index}"
            )
        return value

    def scalar_bool(self, value: object, label: str) -> bool:
        """Synchronize one validation scalar explicitly.

        GPU domain validation necessarily observes a device scalar.  This is
        the only device-to-host operation in the kernel layer and is never
        applied to user arrays or numerical results.
        """

        try:
            item = value.item()  # NumPy and CuPy zero-dimensional scalars.
        except (AttributeError, TypeError, ValueError) as exc:
            raise BackendArrayError(f"{label} did not produce a backend scalar") from exc
        if type(item) is not bool:
            raise BackendArrayError(f"{label} did not produce a boolean scalar")
        return item

    @contextmanager
    def activate(self):
        """Scope allocations to the requested device and restore it afterward."""

        if self.name == "numpy":
            yield
            return
        with self.xp.cuda.Device(self.device_index):
            yield


def resolve_backend(requested: object) -> ArrayBackend:
    """Resolve exactly the requested backend, with no automatic fallback."""

    name = _normalized_backend_name(requested)
    dtype = getattr(requested, "dtype", "float64")
    allow_fallback = getattr(requested, "allow_fallback", False)
    deterministic_reductions = getattr(requested, "deterministic_reductions", True)
    fast_math = getattr(requested, "fast_math", False)
    determinism_scope = getattr(requested, "determinism_scope", "SAME_RUNTIME_DEVICE")
    device = getattr(requested, "device", "cpu" if name == "numpy" else None)
    if type(dtype) is not str or dtype != "float64":
        raise BackendError("the force evaluator supports only float64")
    if type(allow_fallback) is not bool or allow_fallback:
        raise BackendError("backend fallback must be explicitly disabled")
    if type(deterministic_reductions) is not bool or not deterministic_reductions:
        raise BackendError("deterministic_reductions must be enabled")
    if type(fast_math) is not bool or fast_math:
        raise BackendError("fast_math must be disabled")
    if determinism_scope != "SAME_RUNTIME_DEVICE":
        raise BackendError("determinism_scope must be SAME_RUNTIME_DEVICE")
    if name == "numpy":
        if device != "cpu":
            raise BackendError("the NumPy backend requires device='cpu'")
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise BackendUnavailableError("the requested NumPy backend is not installed") from exc
        return ArrayBackend("numpy", np, np.ndarray, "cpu")

    if type(device) is not str or not device:
        raise BackendError("the CuPy backend requires an explicit CUDA device")
    if device in {"cuda", "gpu"}:
        device_index = 0
    else:
        prefix, separator, suffix = device.partition(":")
        if separator != ":" or prefix not in {"cuda", "gpu"} or not suffix.isdigit():
            raise BackendError(
                "the CuPy device must be 'cuda', 'gpu', 'cuda:<index>', or 'gpu:<index>'"
            )
        device_index = int(suffix)

    try:
        import cupy as cp
    except (ImportError, OSError) as exc:  # pragma: no cover - environment dependent
        raise BackendUnavailableError(
            "the requested CuPy backend is unavailable; JX will not fall back to NumPy"
        ) from exc

    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:  # CuPy raises runtime-specific CUDA/HIP errors.
        raise BackendUnavailableError(
            "the requested CuPy backend cannot access a GPU; JX will not fall back to NumPy"
        ) from exc
    if device_count < 1:
        raise BackendUnavailableError(
            "the requested CuPy backend found no GPU; JX will not fall back to NumPy"
        )
    if device_index >= device_count:
        raise BackendUnavailableError(
            f"the requested CUDA device {device_index} does not exist; "
            "JX will not fall back to another device"
        )
    return ArrayBackend("cupy", cp, cp.ndarray, f"cuda:{device_index}", device_index)


__all__ = [
    "ArrayBackend",
    "BackendArrayError",
    "BackendError",
    "BackendUnavailableError",
    "resolve_backend",
]
