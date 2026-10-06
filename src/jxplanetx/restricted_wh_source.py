"""Precision-first massive-source provider for the restricted CUDA map.

This direct-import experimental module advances a small, fully interacting
massive subsystem with JX's compiled ordered-Jacobi Wisdom--Holman map and
materializes an immutable primary-relative :class:`FrozenSourcePathTile`.
Massless tracer state is never accepted, so back-reaction is structurally
impossible.  Every build performs a complete same-runtime bitwise replay.

The provider remains screening-only.  It does not add a public engine export,
disk checkpoint contract, close-encounter source solver, or long-horizon
accuracy claim.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .cuda_restricted_wh import (
    COORDINATE_SYSTEM,
    CUDARestrictedWHResult,
    CUDARestrictedWHState,
    FixedStepCUDARestrictedWHSpec,
    FrozenSourcePathTile,
    advance_cuda_restricted_wh_fused,
    source_path_tile_sha256,
)
from .cuda_restricted_wh_hybrid import (
    CUDARestrictedWHHybridResult,
    DEFAULT_ROUTE_WORKER_COUNT,
    advance_cuda_restricted_wh_hybrid,
)
from .engine import (
    BackendSpec,
    FixedStepWisdomHolmanSpec,
    ForcePlan,
    NewtonianPointMass,
    ParameterMetadata,
    Provenance,
    StateSnapshot,
)
from .fast_wisdom_holman import (
    FastWisdomHolmanLoopError,
    fast_wisdom_holman_runtime_identity,
    integrate_fast_wisdom_holman_trajectory,
)
from .restricted_wh_route import RestrictedWHRouteSpec
from .restricted_wh_lifecycle import (
    RestrictedWHLifecycleResult,
    RestrictedWHLifecycleState,
    advance_cuda_restricted_wh_lifecycle,
)


PROVIDER_ID = "jx.restricted-wh.source-provider.ordered-jacobi.v1"
SCIENTIFIC_CLASSIFICATION = "MODEL_OUTPUT"
PROJECT_SCOPE = "SCREENING_ONLY"
MAXIMUM_SOURCE_STEPS_PER_TILE = 1_000_000
COORDINATE_CANONICALIZATION_ROUNDING_ULP_FACTOR = 1.0
MAXIMUM_COORDINATE_CANONICALIZATION_RELATIVE_ERROR_IN_EPS = 512.0
MAXIMUM_COORDINATE_CANONICALIZATION_RELATIVE_ERROR = (
    float(
        MAXIMUM_COORDINATE_CANONICALIZATION_RELATIVE_ERROR_IN_EPS
        * np.finfo(np.float64).eps
    )
)
MAXIMUM_COORDINATE_CANONICALIZATION_FIXED_POINT_ITERATIONS = 4
IMPLEMENTATION_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

SOURCE_PROVIDER_CLAIMS = MappingProxyType(
    {
        "production_qualified": False,
        "long_horizon_accuracy_qualified": False,
        "source_close_encounter_qualified": False,
        "disk_checkpoint_restart_qualified": False,
        "cross_runtime_bitwise_reproducible": False,
        "science_result": False,
    }
)


class RestrictedWHSourceError(RuntimeError):
    """Base error for source-tile construction."""


class RestrictedWHSourceContractError(RestrictedWHSourceError, ValueError):
    """A source-provider request is ambiguous or outside its fixed envelope."""


class RestrictedWHSourceIntegrationError(RestrictedWHSourceError):
    """The massive source map failed closed without producing a tile."""


def _finite_float(value: object, label: str, *, nonzero: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise RestrictedWHSourceContractError(
            f"{label} must be a finite built-in float"
        )
    if nonzero and value == 0.0:
        raise RestrictedWHSourceContractError(f"{label} cannot be zero")
    return value


def _positive_float(value: object, label: str) -> float:
    result = _finite_float(value, label)
    if not result > 0.0:
        raise RestrictedWHSourceContractError(f"{label} must be positive")
    return result


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise RestrictedWHSourceContractError(
            f"{label} must be a nonempty trimmed string"
        )
    return value


def _sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or value == "0" * 64
    ):
        raise RestrictedWHSourceContractError(
            f"{label} must be a nonzero lowercase SHA-256 digest"
        )
    return value


def _float64_array(
    value: object,
    label: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.float64)
        or value.shape != shape
        or not value.flags.c_contiguous
        or not bool(np.all(np.isfinite(value)))
    ):
        raise RestrictedWHSourceContractError(
            f"{label} must be a finite C-contiguous float64 array with shape {shape}"
        )
    return np.ascontiguousarray(value.copy())


def _canonical_input_sha256(
    source_ids: tuple[str, ...],
    positions: np.ndarray,
    velocities: np.ndarray,
    gm: np.ndarray,
) -> str:
    return _coordinate_state_sha256(
        b"jx.restricted-wh.source-provider.input.v1",
        source_ids,
        positions,
        velocities,
        gm,
    )


def _coordinate_state_sha256(
    domain: bytes,
    source_ids: tuple[str, ...],
    positions: np.ndarray,
    velocities: np.ndarray,
    gm: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(domain + b"\0")
    for source_id in source_ids:
        encoded = source_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    for array in (positions, velocities, gm):
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _coordinate_canonicalization_metrics(
    original: np.ndarray,
    canonical: np.ndarray,
    cumulative_forward_error_bounds: np.ndarray,
    label: str,
) -> Mapping[str, float]:
    """Validate and disclose one binary64 coordinate round trip.

    Primary-relative inputs must be converted to a barycentric massive-system
    state before the ordered-Jacobi map can consume them.  Subtracting the
    primary again is not an identity operation for every binary64 vector.  We
    check each correction against both a conservative forward-error bound for
    the three rounded subtractions and a strict input-relative conditioning
    ceiling.  Input ULP counts remain diagnostics only: near zero, a valid
    barycentre translation can span many ULPs while staying within binary64
    forward error.
    """

    if not (
        original.shape == canonical.shape == cumulative_forward_error_bounds.shape
    ):
        raise RestrictedWHSourceContractError(
            f"{label} canonicalization changed array shape"
        )
    maximum_ulps = 0.0
    maximum_forward_bound_fraction = 0.0
    maximum_relative_error = 0.0
    for original_value, canonical_value, bound_value in zip(
        original.flat,
        canonical.flat,
        cumulative_forward_error_bounds.flat,
        strict=True,
    ):
        before = float(original_value)
        after = float(canonical_value)
        if before == after:
            continue
        correction = abs(after - before)
        forward_bound = float(bound_value)
        if before == 0.0:
            raise RestrictedWHSourceContractError(
                f"{label} canonicalization changed an exact-zero component"
            )
        relative_error = correction / abs(before)
        if not (
            math.isfinite(correction)
            and math.isfinite(forward_bound)
            and correction <= forward_bound
            and math.isfinite(relative_error)
            and relative_error
            <= MAXIMUM_COORDINATE_CANONICALIZATION_RELATIVE_ERROR
        ):
            raise RestrictedWHSourceContractError(
                f"{label} barycentric canonicalization exceeded its "
                "binary64 forward-error or conditioning bound"
            )
        component_ulps = correction / math.ulp(before)
        maximum_ulps = max(maximum_ulps, component_ulps)
        maximum_forward_bound_fraction = max(
            maximum_forward_bound_fraction, correction / forward_bound
        )
        maximum_relative_error = max(maximum_relative_error, relative_error)
    return MappingProxyType(
        {
            "maximum_component_input_ulps": float(maximum_ulps),
            "maximum_forward_error_bound_fraction": float(
                maximum_forward_bound_fraction
            ),
            "maximum_input_relative_error": float(maximum_relative_error),
        }
    )


def _binary64_array_bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(
        left.shape == right.shape
        and left.dtype == np.dtype(np.float64)
        and right.dtype == np.dtype(np.float64)
        and np.array_equal(left.view(np.uint64), right.view(np.uint64))
    )


def _canonicalize_primary_relative_state(
    positions: np.ndarray,
    velocities: np.ndarray,
    gm: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Mapping[str, float],
    Mapping[str, float],
    int,
]:
    """Create a bounded fixed-point barycentric/relative coordinate pair."""

    total_gm = np.sum(gm, dtype=np.float64)
    current_positions = np.ascontiguousarray(positions.copy())
    current_velocities = np.ascontiguousarray(velocities.copy())
    position_forward_bounds = np.zeros_like(positions)
    velocity_forward_bounds = np.zeros_like(velocities)
    seen: set[tuple[bytes, bytes]] = set()
    for iteration_count in range(
        1, MAXIMUM_COORDINATE_CANONICALIZATION_FIXED_POINT_ITERATIONS + 1
    ):
        state_key = (
            current_positions.tobytes(order="C"),
            current_velocities.tobytes(order="C"),
        )
        if state_key in seen:
            raise RestrictedWHSourceContractError(
                "barycentric source canonicalization entered a cycle"
            )
        seen.add(state_key)
        barycenter_position = (
            np.sum(current_positions * gm[:, None], axis=0, dtype=np.float64)
            / total_gm
        )
        barycenter_velocity = (
            np.sum(current_velocities * gm[:, None], axis=0, dtype=np.float64)
            / total_gm
        )
        barycentric_positions = np.ascontiguousarray(
            current_positions - barycenter_position
        )
        barycentric_velocities = np.ascontiguousarray(
            current_velocities - barycenter_velocity
        )
        canonical_positions = np.ascontiguousarray(
            barycentric_positions - barycentric_positions[0]
        )
        canonical_velocities = np.ascontiguousarray(
            barycentric_velocities - barycentric_velocities[0]
        )
        for source_index in range(current_positions.shape[0]):
            for component in range(3):
                position_forward_bounds[source_index, component] += (
                    COORDINATE_CANONICALIZATION_ROUNDING_ULP_FACTOR
                    * (
                        math.ulp(barycentric_positions[source_index, component])
                        + math.ulp(barycentric_positions[0, component])
                        + math.ulp(canonical_positions[source_index, component])
                    )
                )
                velocity_forward_bounds[source_index, component] += (
                    COORDINATE_CANONICALIZATION_ROUNDING_ULP_FACTOR
                    * (
                        math.ulp(barycentric_velocities[source_index, component])
                        + math.ulp(barycentric_velocities[0, component])
                        + math.ulp(canonical_velocities[source_index, component])
                    )
                )
        if _binary64_array_bitwise_equal(
            canonical_positions, current_positions
        ) and _binary64_array_bitwise_equal(
            canonical_velocities, current_velocities
        ):
            break
        current_positions = canonical_positions
        current_velocities = canonical_velocities
    else:
        raise RestrictedWHSourceContractError(
            "barycentric source canonicalization did not reach a fixed point"
        )
    position_metrics = _coordinate_canonicalization_metrics(
        positions,
        canonical_positions,
        position_forward_bounds,
        "initial source positions",
    )
    velocity_metrics = _coordinate_canonicalization_metrics(
        velocities,
        canonical_velocities,
        velocity_forward_bounds,
        "initial source velocities",
    )
    return (
        barycentric_positions,
        barycentric_velocities,
        canonical_positions,
        canonical_velocities,
        position_metrics,
        velocity_metrics,
        iteration_count,
    )


def _derived_epoch(origin: float, global_step: int, step: float) -> float:
    try:
        epoch = origin + global_step * step
    except OverflowError as exc:
        raise RestrictedWHSourceContractError(
            "derived source epoch must remain finite"
        ) from exc
    if not math.isfinite(epoch):
        raise RestrictedWHSourceContractError(
            "derived source epoch must remain finite"
        )
    return epoch


def _freeze_tree(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return MappingProxyType(
            {str(key): _freeze_tree(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_tree(item) for item in value)
    return value


def _canonical_record_sha256(domain: bytes, record: Mapping[str, Any]) -> str:
    try:
        payload = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise RestrictedWHSourceContractError(
            "coordinate-canonicalization receipt is not canonical JSON"
        ) from exc
    return hashlib.sha256(domain + b"\0" + payload).hexdigest()


_COORDINATE_CANONICALIZATION_RECEIPT_SEAL = object()


@dataclass(frozen=True, slots=True, init=False)
class SourceCoordinateCanonicalizationReceipt:
    """Sealed precision custody for the source coordinate conversion."""

    policy: str
    forward_error_bound: str
    forward_error_ulp_factor: float
    maximum_allowed_input_relative_error_in_eps: float
    maximum_allowed_input_relative_error: float
    position_maximum_component_ulps: float
    velocity_maximum_component_ulps: float
    position_maximum_forward_error_bound_fraction: float
    velocity_maximum_forward_error_bound_fraction: float
    position_maximum_input_relative_error: float
    velocity_maximum_input_relative_error: float
    position_changed_component_count: int
    velocity_changed_component_count: int
    fixed_point_iteration_count: int
    position_maximum_absolute_delta_AU: float
    velocity_maximum_absolute_delta_AU_per_year: float
    raw_relative_input_sha256: str
    canonical_relative_input_sha256: str
    canonical_barycentric_input_sha256: str
    source_tile_sha256: str
    canonicalization_idempotent: bool
    tile_node_zero_exact: bool
    content_sha256: str
    _verification_seal: object

    def _canonical_record(self) -> dict[str, Any]:
        return {
            "schema": "jx.restricted-wh.source-coordinate-canonicalization.v2",
            "policy": self.policy,
            "forward_error_bound": self.forward_error_bound,
            "forward_error_ulp_factor_hex": self.forward_error_ulp_factor.hex(),
            "maximum_allowed_input_relative_error_in_eps_hex": (
                self.maximum_allowed_input_relative_error_in_eps.hex()
            ),
            "maximum_allowed_input_relative_error_hex": (
                self.maximum_allowed_input_relative_error.hex()
            ),
            "position_maximum_component_ulps_hex": (
                self.position_maximum_component_ulps.hex()
            ),
            "velocity_maximum_component_ulps_hex": (
                self.velocity_maximum_component_ulps.hex()
            ),
            "position_maximum_forward_error_bound_fraction_hex": (
                self.position_maximum_forward_error_bound_fraction.hex()
            ),
            "velocity_maximum_forward_error_bound_fraction_hex": (
                self.velocity_maximum_forward_error_bound_fraction.hex()
            ),
            "position_maximum_input_relative_error_hex": (
                self.position_maximum_input_relative_error.hex()
            ),
            "velocity_maximum_input_relative_error_hex": (
                self.velocity_maximum_input_relative_error.hex()
            ),
            "position_changed_component_count": self.position_changed_component_count,
            "velocity_changed_component_count": self.velocity_changed_component_count,
            "fixed_point_iteration_count": self.fixed_point_iteration_count,
            "maximum_fixed_point_iteration_count": (
                MAXIMUM_COORDINATE_CANONICALIZATION_FIXED_POINT_ITERATIONS
            ),
            "position_maximum_absolute_delta_AU_hex": (
                self.position_maximum_absolute_delta_AU.hex()
            ),
            "velocity_maximum_absolute_delta_AU_per_year_hex": (
                self.velocity_maximum_absolute_delta_AU_per_year.hex()
            ),
            "raw_relative_input_sha256": self.raw_relative_input_sha256,
            "canonical_relative_input_sha256": (
                self.canonical_relative_input_sha256
            ),
            "canonical_barycentric_input_sha256": (
                self.canonical_barycentric_input_sha256
            ),
            "source_tile_sha256": self.source_tile_sha256,
            "canonicalization_idempotent": self.canonicalization_idempotent,
            "tile_node_zero_exact": self.tile_node_zero_exact,
        }

    def as_record(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": "jx.restricted-wh.source-coordinate-canonicalization.v2",
                "policy": self.policy,
                "forward_error_bound": self.forward_error_bound,
                "forward_error_ulp_factor": self.forward_error_ulp_factor,
                "maximum_allowed_input_relative_error_in_eps": (
                    self.maximum_allowed_input_relative_error_in_eps
                ),
                "maximum_allowed_input_relative_error": (
                    self.maximum_allowed_input_relative_error
                ),
                "position_maximum_component_ulps": (
                    self.position_maximum_component_ulps
                ),
                "velocity_maximum_component_ulps": (
                    self.velocity_maximum_component_ulps
                ),
                "position_maximum_forward_error_bound_fraction": (
                    self.position_maximum_forward_error_bound_fraction
                ),
                "velocity_maximum_forward_error_bound_fraction": (
                    self.velocity_maximum_forward_error_bound_fraction
                ),
                "position_maximum_input_relative_error": (
                    self.position_maximum_input_relative_error
                ),
                "velocity_maximum_input_relative_error": (
                    self.velocity_maximum_input_relative_error
                ),
                "position_changed_component_count": (
                    self.position_changed_component_count
                ),
                "velocity_changed_component_count": (
                    self.velocity_changed_component_count
                ),
                "fixed_point_iteration_count": self.fixed_point_iteration_count,
                "maximum_fixed_point_iteration_count": (
                    MAXIMUM_COORDINATE_CANONICALIZATION_FIXED_POINT_ITERATIONS
                ),
                "position_maximum_absolute_delta_AU": (
                    self.position_maximum_absolute_delta_AU
                ),
                "velocity_maximum_absolute_delta_AU_per_year": (
                    self.velocity_maximum_absolute_delta_AU_per_year
                ),
                "raw_relative_input_sha256": self.raw_relative_input_sha256,
                "canonical_relative_input_sha256": (
                    self.canonical_relative_input_sha256
                ),
                "canonical_barycentric_input_sha256": (
                    self.canonical_barycentric_input_sha256
                ),
                "source_tile_sha256": self.source_tile_sha256,
                "canonicalization_idempotent": self.canonicalization_idempotent,
                "tile_node_zero_exact": self.tile_node_zero_exact,
                "content_sha256": self.content_sha256,
            }
        )

    def __post_init__(self) -> None:
        if self._verification_seal is not _COORDINATE_CANONICALIZATION_RECEIPT_SEAL:
            raise RestrictedWHSourceContractError(
                "coordinate-canonicalization receipts are provider-created only"
            )
        if (
            self.policy
            != "PRIMARY_RELATIVE_TO_GM_BARYCENTRIC_TO_PRIMARY_RELATIVE_V2"
            or self.forward_error_bound
            != "SUM_OF_FULL_ULPS_OF_INTERMEDIATE_PRIMARY_AND_RESULT"
            or self.forward_error_ulp_factor
            != COORDINATE_CANONICALIZATION_ROUNDING_ULP_FACTOR
            or self.maximum_allowed_input_relative_error_in_eps
            != MAXIMUM_COORDINATE_CANONICALIZATION_RELATIVE_ERROR_IN_EPS
            or self.maximum_allowed_input_relative_error
            != MAXIMUM_COORDINATE_CANONICALIZATION_RELATIVE_ERROR
        ):
            raise RestrictedWHSourceContractError(
                "coordinate-canonicalization policy changed"
            )
        for name in (
            "position_maximum_component_ulps",
            "velocity_maximum_component_ulps",
            "position_maximum_forward_error_bound_fraction",
            "velocity_maximum_forward_error_bound_fraction",
            "position_maximum_input_relative_error",
            "velocity_maximum_input_relative_error",
            "position_maximum_absolute_delta_AU",
            "velocity_maximum_absolute_delta_AU_per_year",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise RestrictedWHSourceContractError(
                    f"{name} must be a nonnegative finite float"
                )
        if (
            self.position_maximum_forward_error_bound_fraction > 1.0
            or self.velocity_maximum_forward_error_bound_fraction > 1.0
            or self.position_maximum_input_relative_error
            > MAXIMUM_COORDINATE_CANONICALIZATION_RELATIVE_ERROR
            or self.velocity_maximum_input_relative_error
            > MAXIMUM_COORDINATE_CANONICALIZATION_RELATIVE_ERROR
        ):
            raise RestrictedWHSourceContractError(
                "coordinate-canonicalization numerical bound changed"
            )
        for name in (
            "position_changed_component_count",
            "velocity_changed_component_count",
            "fixed_point_iteration_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise RestrictedWHSourceContractError(
                    f"{name} must be a nonnegative exact integer"
                )
        if not (
            1
            <= self.fixed_point_iteration_count
            <= MAXIMUM_COORDINATE_CANONICALIZATION_FIXED_POINT_ITERATIONS
        ):
            raise RestrictedWHSourceContractError(
                "coordinate-canonicalization iteration count is outside policy"
            )
        for name in (
            "raw_relative_input_sha256",
            "canonical_relative_input_sha256",
            "canonical_barycentric_input_sha256",
            "source_tile_sha256",
            "content_sha256",
        ):
            _sha256(getattr(self, name), name)
        if (
            type(self.canonicalization_idempotent) is not bool
            or not self.canonicalization_idempotent
            or type(self.tile_node_zero_exact) is not bool
            or not self.tile_node_zero_exact
        ):
            raise RestrictedWHSourceContractError(
                "coordinate canonicalization must be idempotent and tile-exact"
            )
        expected = _canonical_record_sha256(
            b"jx.restricted-wh.source-coordinate-canonicalization.v2",
            self._canonical_record(),
        )
        if self.content_sha256 != expected:
            raise RestrictedWHSourceContractError(
                "coordinate-canonicalization receipt digest changed"
            )


def _make_coordinate_canonicalization_receipt(
    *,
    tile: FrozenSourcePathTile,
    raw_relative_input_sha256: str,
    canonical_relative_input_sha256: str,
    canonical_barycentric_input_sha256: str,
    positions: np.ndarray,
    velocities: np.ndarray,
    canonical_positions: np.ndarray,
    canonical_velocities: np.ndarray,
    position_metrics: Mapping[str, float],
    velocity_metrics: Mapping[str, float],
    fixed_point_iteration_count: int,
) -> SourceCoordinateCanonicalizationReceipt:
    values: dict[str, Any] = {
        "policy": "PRIMARY_RELATIVE_TO_GM_BARYCENTRIC_TO_PRIMARY_RELATIVE_V2",
        "forward_error_bound": (
            "SUM_OF_FULL_ULPS_OF_INTERMEDIATE_PRIMARY_AND_RESULT"
        ),
        "forward_error_ulp_factor": (
            COORDINATE_CANONICALIZATION_ROUNDING_ULP_FACTOR
        ),
        "maximum_allowed_input_relative_error_in_eps": (
            MAXIMUM_COORDINATE_CANONICALIZATION_RELATIVE_ERROR_IN_EPS
        ),
        "maximum_allowed_input_relative_error": (
            MAXIMUM_COORDINATE_CANONICALIZATION_RELATIVE_ERROR
        ),
        "position_maximum_component_ulps": position_metrics[
            "maximum_component_input_ulps"
        ],
        "velocity_maximum_component_ulps": velocity_metrics[
            "maximum_component_input_ulps"
        ],
        "position_maximum_forward_error_bound_fraction": position_metrics[
            "maximum_forward_error_bound_fraction"
        ],
        "velocity_maximum_forward_error_bound_fraction": velocity_metrics[
            "maximum_forward_error_bound_fraction"
        ],
        "position_maximum_input_relative_error": position_metrics[
            "maximum_input_relative_error"
        ],
        "velocity_maximum_input_relative_error": velocity_metrics[
            "maximum_input_relative_error"
        ],
        "position_changed_component_count": int(
            np.count_nonzero(
                positions.view(np.uint64) != canonical_positions.view(np.uint64)
            )
        ),
        "velocity_changed_component_count": int(
            np.count_nonzero(
                velocities.view(np.uint64) != canonical_velocities.view(np.uint64)
            )
        ),
        "fixed_point_iteration_count": fixed_point_iteration_count,
        "position_maximum_absolute_delta_AU": float(
            np.max(np.abs(canonical_positions - positions))
        ),
        "velocity_maximum_absolute_delta_AU_per_year": float(
            np.max(np.abs(canonical_velocities - velocities))
        ),
        "raw_relative_input_sha256": raw_relative_input_sha256,
        "canonical_relative_input_sha256": canonical_relative_input_sha256,
        "canonical_barycentric_input_sha256": (
            canonical_barycentric_input_sha256
        ),
        "source_tile_sha256": tile.content_sha256,
        "canonicalization_idempotent": True,
        "tile_node_zero_exact": True,
    }
    receipt = object.__new__(SourceCoordinateCanonicalizationReceipt)
    for name, value in values.items():
        object.__setattr__(receipt, name, value)
    object.__setattr__(
        receipt,
        "_verification_seal",
        _COORDINATE_CANONICALIZATION_RECEIPT_SEAL,
    )
    object.__setattr__(
        receipt,
        "content_sha256",
        _canonical_record_sha256(
            b"jx.restricted-wh.source-coordinate-canonicalization.v2",
            receipt._canonical_record(),
        ),
    )
    receipt.__post_init__()
    return receipt


@dataclass(frozen=True)
class WisdomHolmanSourceTileSpec:
    """Fixed source lattice and fail-closed massive-system guard policy."""

    source_step_years: float
    source_step_count: int
    minimum_encounter_pair_separation_AU: float = 0.01
    minimum_jacobi_periapse_AU: float = 0.1
    maximum_initial_barycenter_position_norm_AU: float = 1.0e-12
    maximum_initial_barycenter_velocity_norm_AU_per_year: float = 1.0e-12

    def __post_init__(self) -> None:
        _finite_float(self.source_step_years, "source_step_years", nonzero=True)
        if type(self.source_step_count) is not int or not (
            1 <= self.source_step_count <= MAXIMUM_SOURCE_STEPS_PER_TILE
        ):
            raise RestrictedWHSourceContractError(
                "source_step_count must be an integer in [1, "
                f"{MAXIMUM_SOURCE_STEPS_PER_TILE}]"
            )
        for name in (
            "minimum_encounter_pair_separation_AU",
            "minimum_jacobi_periapse_AU",
            "maximum_initial_barycenter_position_norm_AU",
            "maximum_initial_barycenter_velocity_norm_AU_per_year",
        ):
            _positive_float(getattr(self, name), name)


@dataclass(frozen=True)
class WisdomHolmanSourceTileBuild:
    """One replay-verified immutable source tile and its numerical custody."""

    tile: FrozenSourcePathTile
    input_content_sha256: str
    source_map_input_content_sha256: str
    source_map_result_content_sha256: str
    source_map_verification_content_sha256: str
    source_map_metrics: tuple[tuple[str, float], ...]
    source_map_primary_seconds: float
    source_map_replay_seconds: float
    coordinate_canonicalization: SourceCoordinateCanonicalizationReceipt
    runtime_identity: Mapping[str, Any]
    claims: Mapping[str, bool] = SOURCE_PROVIDER_CLAIMS
    provider_id: str = PROVIDER_ID
    scientific_classification: str = SCIENTIFIC_CLASSIFICATION
    project_scope: str = PROJECT_SCOPE

    def __post_init__(self) -> None:
        if type(self.tile) is not FrozenSourcePathTile:
            raise RestrictedWHSourceContractError(
                "tile must be an exact FrozenSourcePathTile"
            )
        verify_digest = source_path_tile_sha256(self.tile)
        if verify_digest != self.tile.content_sha256:
            raise RestrictedWHSourceContractError("tile digest changed")
        for name in (
            "input_content_sha256",
            "source_map_input_content_sha256",
            "source_map_result_content_sha256",
            "source_map_verification_content_sha256",
        ):
            _sha256(getattr(self, name), name)
        if (
            type(self.source_map_metrics) is not tuple
            or any(
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or type(item[1]) is not float
                or not math.isfinite(item[1])
                for item in self.source_map_metrics
            )
        ):
            raise RestrictedWHSourceContractError(
                "source_map_metrics must be an immutable finite tuple"
            )
        _positive_float(self.source_map_primary_seconds, "source_map_primary_seconds")
        _positive_float(self.source_map_replay_seconds, "source_map_replay_seconds")
        if type(self.coordinate_canonicalization) is not SourceCoordinateCanonicalizationReceipt:
            raise RestrictedWHSourceContractError(
                "coordinate_canonicalization must be a sealed provider receipt"
            )
        self.coordinate_canonicalization.__post_init__()
        if (
            self.coordinate_canonicalization.raw_relative_input_sha256
            != self.input_content_sha256
            or self.coordinate_canonicalization.source_tile_sha256
            != self.tile.content_sha256
            or self.coordinate_canonicalization.canonical_relative_input_sha256
            != _canonical_input_sha256(
                self.tile.source_ids,
                np.ascontiguousarray(self.tile.source_relative_positions_AU[0]),
                np.ascontiguousarray(
                    self.tile.source_relative_velocities_AU_per_year[0]
                ),
                np.ascontiguousarray(self.tile.source_gm_AU3_per_year2),
            )
        ):
            raise RestrictedWHSourceContractError(
                "coordinate-canonicalization receipt is not linked to build bytes"
            )
        if (
            self.provider_id != PROVIDER_ID
            or self.scientific_classification != SCIENTIFIC_CLASSIFICATION
            or self.project_scope != PROJECT_SCOPE
            or dict(self.claims) != dict(SOURCE_PROVIDER_CLAIMS)
            or any(self.claims.values())
        ):
            raise RestrictedWHSourceContractError(
                "source-provider identity or claim ceiling changed"
            )
        runtime_identity = _freeze_tree(self.runtime_identity)
        runtime_coordinate_receipt = runtime_identity.get(
            "coordinate_canonicalization"
        ) if isinstance(runtime_identity, MappingABC) else None
        if (
            not isinstance(runtime_identity, MappingABC)
            or runtime_identity.get("provider_id") != PROVIDER_ID
            or runtime_identity.get("provider_implementation_source_sha256")
            != IMPLEMENTATION_SOURCE_SHA256
            or runtime_identity.get("exact_replay_required") is not True
            or runtime_identity.get("source_tile_sha256")
            != self.tile.content_sha256
            or runtime_identity.get("coordinate_canonicalization_receipt_sha256")
            != self.coordinate_canonicalization.content_sha256
            or not isinstance(runtime_coordinate_receipt, MappingABC)
            or _canonical_record_sha256(
                b"jx.restricted-wh.source-coordinate-runtime-view.v2",
                dict(runtime_coordinate_receipt),
            )
            != _canonical_record_sha256(
                b"jx.restricted-wh.source-coordinate-runtime-view.v2",
                dict(self.coordinate_canonicalization.as_record()),
            )
        ):
            raise RestrictedWHSourceContractError(
                "runtime identity is not linked to coordinate custody"
            )
        object.__setattr__(self, "runtime_identity", runtime_identity)
        object.__setattr__(
            self, "claims", MappingProxyType(dict(SOURCE_PROVIDER_CLAIMS))
        )


@dataclass(frozen=True)
class WisdomHolmanRestrictedWHSegmentResult:
    """One source-build and tracer-segment result linked by the tile digest."""

    source_build: WisdomHolmanSourceTileBuild
    tracer_result: CUDARestrictedWHResult
    claims: Mapping[str, bool] = SOURCE_PROVIDER_CLAIMS
    provider_id: str = PROVIDER_ID
    scientific_classification: str = SCIENTIFIC_CLASSIFICATION
    project_scope: str = PROJECT_SCOPE

    def __post_init__(self) -> None:
        if type(self.source_build) is not WisdomHolmanSourceTileBuild:
            raise RestrictedWHSourceContractError(
                "source_build must be an exact WisdomHolmanSourceTileBuild"
            )
        if type(self.tracer_result) is not CUDARestrictedWHResult:
            raise RestrictedWHSourceContractError(
                "tracer_result must be an exact CUDARestrictedWHResult"
            )
        if (
            self.tracer_result.source_tile_sha256
            != self.source_build.tile.content_sha256
        ):
            raise RestrictedWHSourceContractError(
                "tracer result is not linked to the retained source build"
            )
        if (
            self.provider_id != PROVIDER_ID
            or self.scientific_classification != SCIENTIFIC_CLASSIFICATION
            or self.project_scope != PROJECT_SCOPE
            or dict(self.claims) != dict(SOURCE_PROVIDER_CLAIMS)
            or any(self.claims.values())
        ):
            raise RestrictedWHSourceContractError(
                "combined source/tracer identity or claim ceiling changed"
            )
        object.__setattr__(
            self, "claims", MappingProxyType(dict(SOURCE_PROVIDER_CLAIMS))
        )


@dataclass(frozen=True)
class WisdomHolmanRestrictedWHHybridSegmentResult:
    """One source build linked to a guarded CUDA/DOP853 tracer segment."""

    source_build: WisdomHolmanSourceTileBuild
    tracer_result: CUDARestrictedWHHybridResult
    claims: Mapping[str, bool] = SOURCE_PROVIDER_CLAIMS
    provider_id: str = PROVIDER_ID
    scientific_classification: str = SCIENTIFIC_CLASSIFICATION
    project_scope: str = PROJECT_SCOPE

    def __post_init__(self) -> None:
        if type(self.source_build) is not WisdomHolmanSourceTileBuild:
            raise RestrictedWHSourceContractError(
                "source_build must be an exact WisdomHolmanSourceTileBuild"
            )
        if type(self.tracer_result) is not CUDARestrictedWHHybridResult:
            raise RestrictedWHSourceContractError(
                "tracer_result must be an exact CUDARestrictedWHHybridResult"
            )
        if (
            self.tracer_result.source_tile_sha256
            != self.source_build.tile.content_sha256
        ):
            raise RestrictedWHSourceContractError(
                "hybrid tracer result is not linked to the retained source build"
            )
        if (
            self.provider_id != PROVIDER_ID
            or self.scientific_classification != SCIENTIFIC_CLASSIFICATION
            or self.project_scope != PROJECT_SCOPE
            or dict(self.claims) != dict(SOURCE_PROVIDER_CLAIMS)
            or any(self.claims.values())
        ):
            raise RestrictedWHSourceContractError(
                "combined hybrid identity or claim ceiling changed"
            )
        object.__setattr__(
            self, "claims", MappingProxyType(dict(SOURCE_PROVIDER_CLAIMS))
        )


@dataclass(frozen=True)
class WisdomHolmanRestrictedWHLifecycleSegmentResult:
    """One source build linked to fixed-lane terminal tracer lifecycle."""

    source_build: WisdomHolmanSourceTileBuild
    tracer_result: RestrictedWHLifecycleResult
    claims: Mapping[str, bool] = SOURCE_PROVIDER_CLAIMS
    provider_id: str = PROVIDER_ID
    scientific_classification: str = SCIENTIFIC_CLASSIFICATION
    project_scope: str = PROJECT_SCOPE

    def __post_init__(self) -> None:
        if type(self.source_build) is not WisdomHolmanSourceTileBuild:
            raise RestrictedWHSourceContractError(
                "source_build must be an exact WisdomHolmanSourceTileBuild"
            )
        if type(self.tracer_result) is not RestrictedWHLifecycleResult:
            raise RestrictedWHSourceContractError(
                "tracer_result must be an exact RestrictedWHLifecycleResult"
            )
        if (
            self.tracer_result.source_tile_sha256
            != self.source_build.tile.content_sha256
        ):
            raise RestrictedWHSourceContractError(
                "lifecycle result is not linked to the retained source build"
            )
        if (
            self.provider_id != PROVIDER_ID
            or self.scientific_classification != SCIENTIFIC_CLASSIFICATION
            or self.project_scope != PROJECT_SCOPE
            or dict(self.claims) != dict(SOURCE_PROVIDER_CLAIMS)
            or any(self.claims.values())
        ):
            raise RestrictedWHSourceContractError(
                "combined lifecycle identity or claim ceiling changed"
            )
        object.__setattr__(
            self, "claims", MappingProxyType(dict(SOURCE_PROVIDER_CLAIMS))
        )


def build_wisdom_holman_source_tile(
    *,
    tile_id: str,
    epoch_origin_years: float,
    global_source_step: int,
    time_scale: str,
    frame: str,
    axes: str,
    source_ids: tuple[str, ...],
    primary_source_id: str,
    source_schedule_sha256: str,
    initial_source_relative_positions_AU: np.ndarray,
    initial_source_relative_velocities_AU_per_year: np.ndarray,
    source_gm_AU3_per_year2: np.ndarray,
    source_encounter_radii_AU: np.ndarray,
    certified_relative_acceleration_bounds_AU_per_year2: np.ndarray,
    provenance: Provenance,
    spec: WisdomHolmanSourceTileSpec,
) -> WisdomHolmanSourceTileBuild:
    """Build one exact-replay source tile or fail without returning partial data."""

    _text(tile_id, "tile_id")
    _finite_float(epoch_origin_years, "epoch_origin_years")
    if type(global_source_step) is not int or global_source_step < 0:
        raise RestrictedWHSourceContractError(
            "global_source_step must be a nonnegative integer"
        )
    for label, value in (("time_scale", time_scale), ("frame", frame), ("axes", axes)):
        _text(value, label)
    if type(source_ids) is not tuple or not source_ids:
        raise RestrictedWHSourceContractError(
            "source_ids must be a nonempty immutable tuple"
        )
    for index, source_id in enumerate(source_ids):
        _text(source_id, f"source_ids[{index}]")
    if len(set(source_ids)) != len(source_ids):
        raise RestrictedWHSourceContractError("source_ids must be unique")
    if primary_source_id != source_ids[0]:
        raise RestrictedWHSourceContractError(
            "primary_source_id must identify source slot zero"
        )
    _sha256(source_schedule_sha256, "source_schedule_sha256")
    if type(provenance) is not Provenance:
        raise RestrictedWHSourceContractError("provenance must be Provenance")
    if type(spec) is not WisdomHolmanSourceTileSpec:
        raise RestrictedWHSourceContractError(
            "spec must be an exact WisdomHolmanSourceTileSpec"
        )
    spec.__post_init__()

    source_count = len(source_ids)
    positions = _float64_array(
        initial_source_relative_positions_AU,
        "initial_source_relative_positions_AU",
        (source_count, 3),
    )
    velocities = _float64_array(
        initial_source_relative_velocities_AU_per_year,
        "initial_source_relative_velocities_AU_per_year",
        (source_count, 3),
    )
    gm = _float64_array(
        source_gm_AU3_per_year2,
        "source_gm_AU3_per_year2",
        (source_count,),
    )
    radii = _float64_array(
        source_encounter_radii_AU,
        "source_encounter_radii_AU",
        (source_count,),
    )
    acceleration_bounds = _float64_array(
        certified_relative_acceleration_bounds_AU_per_year2,
        "certified_relative_acceleration_bounds_AU_per_year2",
        (source_count,),
    )
    if not bool(np.all(positions[0] == 0.0)) or not bool(
        np.all(velocities[0] == 0.0)
    ):
        raise RestrictedWHSourceContractError(
            "initial primary source position and velocity must be exact zero"
        )
    if not bool(np.all(gm > 0.0)):
        raise RestrictedWHSourceContractError("every source GM must be positive")
    if not bool(np.all(radii > 0.0)):
        raise RestrictedWHSourceContractError(
            "every source encounter radius must be positive"
        )
    if not bool(np.all(acceleration_bounds > 0.0)):
        raise RestrictedWHSourceContractError(
            "every acceleration certificate must be positive"
        )

    input_digest = _canonical_input_sha256(source_ids, positions, velocities, gm)
    (
        barycentric_positions,
        barycentric_velocities,
        canonical_positions,
        canonical_velocities,
        position_canonicalization_metrics,
        velocity_canonicalization_metrics,
        canonicalization_iteration_count,
    ) = _canonicalize_primary_relative_state(positions, velocities, gm)
    canonical_input_digest = _canonical_input_sha256(
        source_ids, canonical_positions, canonical_velocities, gm
    )
    canonical_barycentric_input_digest = _coordinate_state_sha256(
        b"jx.restricted-wh.source-provider.canonical-barycentric-input.v1",
        source_ids,
        barycentric_positions,
        barycentric_velocities,
        gm,
    )
    epoch = _derived_epoch(
        epoch_origin_years, global_source_step, spec.source_step_years
    )
    final_epoch = _derived_epoch(
        epoch_origin_years,
        global_source_step + spec.source_step_count,
        spec.source_step_years,
    )
    parameter_metadata = ParameterMetadata(
        "state.gravitational_parameters",
        "AU^3/year^2",
        provenance,
        None,
        None,
        min(epoch, final_epoch) - abs(spec.source_step_years),
        max(epoch, final_epoch) + abs(spec.source_step_years),
    )
    snapshot = StateSnapshot(
        f"{tile_id}.massive-input",
        epoch,
        time_scale,
        "BARYCENTRIC_INERTIAL",
        "BARYCENTER",
        "CARTESIAN_RIGHT_HANDED",
        "AU",
        "year",
        "GM_EQUIVALENT",
        "AU-YEAR",
        source_ids,
        barycentric_positions,
        barycentric_velocities,
        gm.copy(),
        gm.copy(),
        np.zeros(source_count, dtype=np.float64),
        np.ones(source_count, dtype=np.bool_),
        provenance,
    )
    force_plan = ForcePlan(
        f"{tile_id}.massive-plan",
        BackendSpec("numpy", "cpu", 2),
        (
            NewtonianPointMass(
                source_ids,
                source_ids,
                "AU-YEAR",
                (parameter_metadata,),
            ),
        ),
    )
    wh_spec = FixedStepWisdomHolmanSpec(
        checkpoint_step_indices=tuple(range(spec.source_step_count + 1)),
        fixed_step=spec.source_step_years,
        maximum_steps=spec.source_step_count,
        jacobi_body_order=source_ids,
        minimum_encounter_pair_separation=(
            spec.minimum_encounter_pair_separation_AU
        ),
        minimum_jacobi_periapse=spec.minimum_jacobi_periapse_AU,
        maximum_initial_barycenter_position_norm=(
            spec.maximum_initial_barycenter_position_norm_AU
        ),
        maximum_initial_barycenter_velocity_norm=(
            spec.maximum_initial_barycenter_velocity_norm_AU_per_year
        ),
    )
    try:
        source_result = integrate_fast_wisdom_holman_trajectory(
            snapshot,
            force_plan,
            wh_spec,
            exact_replay=True,
        )
    except FastWisdomHolmanLoopError as exc:
        raise RestrictedWHSourceIntegrationError(
            "massive Wisdom--Holman source integration failed closed"
        ) from exc
    if not source_result.exact_replay_verified or source_result.replay_count != 1:
        raise RestrictedWHSourceIntegrationError(
            "massive source integration did not complete its exact replay"
        )

    relative_positions = np.ascontiguousarray(
        source_result.positions - source_result.positions[:, 0:1, :]
    )
    relative_velocities = np.ascontiguousarray(
        source_result.velocities - source_result.velocities[:, 0:1, :]
    )
    if not _binary64_array_bitwise_equal(
        relative_positions[0], canonical_positions
    ) or not _binary64_array_bitwise_equal(
        relative_velocities[0], canonical_velocities
    ):
        raise RestrictedWHSourceIntegrationError(
            "source-map coordinate round trip changed the canonical initial state"
        )

    tile = FrozenSourcePathTile(
        tile_id=tile_id,
        epoch_origin_years=epoch_origin_years,
        global_source_step=global_source_step,
        source_step_years=spec.source_step_years,
        time_scale=time_scale,
        frame=frame,
        axes=axes,
        coordinate_system=COORDINATE_SYSTEM,
        length_unit="AU",
        time_unit="year",
        unit_system_id="AU-YEAR",
        source_ids=source_ids,
        primary_source_id=primary_source_id,
        source_schedule_sha256=source_schedule_sha256,
        source_relative_positions_AU=relative_positions,
        source_relative_velocities_AU_per_year=relative_velocities,
        source_gm_AU3_per_year2=gm,
        source_encounter_radii_AU=radii,
        certified_relative_acceleration_bounds_AU_per_year2=(
            acceleration_bounds
        ),
        provenance=provenance,
    )
    coordinate_canonicalization = _make_coordinate_canonicalization_receipt(
        tile=tile,
        raw_relative_input_sha256=input_digest,
        canonical_relative_input_sha256=canonical_input_digest,
        canonical_barycentric_input_sha256=(
            canonical_barycentric_input_digest
        ),
        positions=positions,
        velocities=velocities,
        canonical_positions=canonical_positions,
        canonical_velocities=canonical_velocities,
        position_metrics=position_canonicalization_metrics,
        velocity_metrics=velocity_canonicalization_metrics,
        fixed_point_iteration_count=canonicalization_iteration_count,
    )
    runtime_identity = {
        "provider_id": PROVIDER_ID,
        "provider_implementation_source_sha256": IMPLEMENTATION_SOURCE_SHA256,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "source_map_runtime_identity": fast_wisdom_holman_runtime_identity(),
        "exact_replay_required": True,
        "coordinate_canonicalization": dict(
            coordinate_canonicalization.as_record()
        ),
        "coordinate_canonicalization_receipt_sha256": (
            coordinate_canonicalization.content_sha256
        ),
        "source_tile_sha256": tile.content_sha256,
    }
    return WisdomHolmanSourceTileBuild(
        tile=tile,
        input_content_sha256=input_digest,
        source_map_input_content_sha256=source_result.input_content_sha256,
        source_map_result_content_sha256=source_result.result_content_sha256,
        source_map_verification_content_sha256=(
            source_result.verification_content_sha256
        ),
        source_map_metrics=source_result.metrics,
        source_map_primary_seconds=float(source_result.primary_seconds),
        source_map_replay_seconds=float(source_result.replay_seconds),
        coordinate_canonicalization=coordinate_canonicalization,
        runtime_identity=runtime_identity,
    )


def advance_cuda_restricted_wh_with_wisdom_holman_source(
    state: CUDARestrictedWHState,
    *,
    tile_id: str,
    initial_source_relative_positions_AU: np.ndarray,
    initial_source_relative_velocities_AU_per_year: np.ndarray,
    source_gm_AU3_per_year2: np.ndarray,
    source_encounter_radii_AU: np.ndarray,
    certified_relative_acceleration_bounds_AU_per_year2: np.ndarray,
    source_provenance: Provenance,
    source_spec: WisdomHolmanSourceTileSpec,
    tracer_spec: FixedStepCUDARestrictedWHSpec,
) -> WisdomHolmanRestrictedWHSegmentResult:
    """Build a replay-verified source tile and consume it in one fused segment."""

    if type(state) is not CUDARestrictedWHState:
        raise RestrictedWHSourceContractError(
            "state must be an exact CUDARestrictedWHState"
        )
    if type(source_spec) is not WisdomHolmanSourceTileSpec:
        raise RestrictedWHSourceContractError(
            "source_spec must be an exact WisdomHolmanSourceTileSpec"
        )
    if type(tracer_spec) is not FixedStepCUDARestrictedWHSpec:
        raise RestrictedWHSourceContractError(
            "tracer_spec must be an exact FixedStepCUDARestrictedWHSpec"
        )
    required_source_steps = (
        tracer_spec.map_step_count * tracer_spec.source_steps_per_map_step
    )
    if source_spec.source_step_count != required_source_steps:
        raise RestrictedWHSourceContractError(
            "source_spec must cover exactly the tracer segment's fine source steps"
        )
    if source_spec.source_step_years != state.source_step_years:
        raise RestrictedWHSourceContractError(
            "source_spec.source_step_years must equal state.source_step_years"
        )
    source_build = build_wisdom_holman_source_tile(
        tile_id=tile_id,
        epoch_origin_years=state.epoch_origin_years,
        global_source_step=state.global_source_step,
        time_scale=state.time_scale,
        frame=state.frame,
        axes=state.axes,
        source_ids=state.source_ids,
        primary_source_id=state.primary_source_id,
        source_schedule_sha256=state.source_schedule_sha256,
        initial_source_relative_positions_AU=(
            initial_source_relative_positions_AU
        ),
        initial_source_relative_velocities_AU_per_year=(
            initial_source_relative_velocities_AU_per_year
        ),
        source_gm_AU3_per_year2=source_gm_AU3_per_year2,
        source_encounter_radii_AU=source_encounter_radii_AU,
        certified_relative_acceleration_bounds_AU_per_year2=(
            certified_relative_acceleration_bounds_AU_per_year2
        ),
        provenance=source_provenance,
        spec=source_spec,
    )
    tracer_result = advance_cuda_restricted_wh_fused(
        state,
        source_build.tile,
        tracer_spec,
    )
    return WisdomHolmanRestrictedWHSegmentResult(
        source_build=source_build,
        tracer_result=tracer_result,
    )


def advance_cuda_restricted_wh_hybrid_with_wisdom_holman_source(
    state: CUDARestrictedWHState,
    *,
    tile_id: str,
    initial_source_relative_positions_AU: np.ndarray,
    initial_source_relative_velocities_AU_per_year: np.ndarray,
    source_gm_AU3_per_year2: np.ndarray,
    source_encounter_radii_AU: np.ndarray,
    certified_relative_acceleration_bounds_AU_per_year2: np.ndarray,
    source_provenance: Provenance,
    source_spec: WisdomHolmanSourceTileSpec,
    tracer_spec: FixedStepCUDARestrictedWHSpec,
    route_spec: RestrictedWHRouteSpec | None = None,
    route_worker_count: int = DEFAULT_ROUTE_WORKER_COUNT,
) -> WisdomHolmanRestrictedWHHybridSegmentResult:
    """Build the replay-verified source tile and run the guarded scheduler."""

    if type(state) is not CUDARestrictedWHState:
        raise RestrictedWHSourceContractError(
            "state must be an exact CUDARestrictedWHState"
        )
    if type(source_spec) is not WisdomHolmanSourceTileSpec:
        raise RestrictedWHSourceContractError(
            "source_spec must be an exact WisdomHolmanSourceTileSpec"
        )
    if type(tracer_spec) is not FixedStepCUDARestrictedWHSpec:
        raise RestrictedWHSourceContractError(
            "tracer_spec must be an exact FixedStepCUDARestrictedWHSpec"
        )
    if route_spec is not None and type(route_spec) is not RestrictedWHRouteSpec:
        raise RestrictedWHSourceContractError(
            "route_spec must be an exact RestrictedWHRouteSpec or None"
        )
    if (
        type(route_worker_count) is not int
        or not 1 <= route_worker_count <= 32
    ):
        raise RestrictedWHSourceContractError(
            "route_worker_count must be an integer in [1, 32]"
        )
    required_source_steps = (
        tracer_spec.map_step_count * tracer_spec.source_steps_per_map_step
    )
    if source_spec.source_step_count != required_source_steps:
        raise RestrictedWHSourceContractError(
            "source_spec must cover exactly the tracer segment's fine source steps"
        )
    if source_spec.source_step_years != state.source_step_years:
        raise RestrictedWHSourceContractError(
            "source_spec.source_step_years must equal state.source_step_years"
        )
    source_build = build_wisdom_holman_source_tile(
        tile_id=tile_id,
        epoch_origin_years=state.epoch_origin_years,
        global_source_step=state.global_source_step,
        time_scale=state.time_scale,
        frame=state.frame,
        axes=state.axes,
        source_ids=state.source_ids,
        primary_source_id=state.primary_source_id,
        source_schedule_sha256=state.source_schedule_sha256,
        initial_source_relative_positions_AU=(
            initial_source_relative_positions_AU
        ),
        initial_source_relative_velocities_AU_per_year=(
            initial_source_relative_velocities_AU_per_year
        ),
        source_gm_AU3_per_year2=source_gm_AU3_per_year2,
        source_encounter_radii_AU=source_encounter_radii_AU,
        certified_relative_acceleration_bounds_AU_per_year2=(
            certified_relative_acceleration_bounds_AU_per_year2
        ),
        provenance=source_provenance,
        spec=source_spec,
    )
    tracer_result = advance_cuda_restricted_wh_hybrid(
        state,
        source_build.tile,
        tracer_spec,
        route_spec=route_spec,
        route_worker_count=route_worker_count,
    )
    return WisdomHolmanRestrictedWHHybridSegmentResult(
        source_build=source_build,
        tracer_result=tracer_result,
    )


def advance_cuda_restricted_wh_lifecycle_with_wisdom_holman_source(
    state: RestrictedWHLifecycleState,
    *,
    tile_id: str,
    initial_source_relative_positions_AU: np.ndarray,
    initial_source_relative_velocities_AU_per_year: np.ndarray,
    source_gm_AU3_per_year2: np.ndarray,
    source_encounter_radii_AU: np.ndarray,
    certified_relative_acceleration_bounds_AU_per_year2: np.ndarray,
    source_provenance: Provenance,
    source_spec: WisdomHolmanSourceTileSpec,
    tracer_spec: FixedStepCUDARestrictedWHSpec,
    route_spec: RestrictedWHRouteSpec | None = None,
    route_worker_count: int = DEFAULT_ROUTE_WORKER_COUNT,
) -> WisdomHolmanRestrictedWHLifecycleSegmentResult:
    """Build a replay-verified source tile and advance fixed tracer lanes."""

    if type(state) is not RestrictedWHLifecycleState:
        raise RestrictedWHSourceContractError(
            "state must be an exact RestrictedWHLifecycleState"
        )
    if type(source_spec) is not WisdomHolmanSourceTileSpec:
        raise RestrictedWHSourceContractError(
            "source_spec must be an exact WisdomHolmanSourceTileSpec"
        )
    if type(tracer_spec) is not FixedStepCUDARestrictedWHSpec:
        raise RestrictedWHSourceContractError(
            "tracer_spec must be an exact FixedStepCUDARestrictedWHSpec"
        )
    if route_spec is not None and type(route_spec) is not RestrictedWHRouteSpec:
        raise RestrictedWHSourceContractError(
            "route_spec must be an exact RestrictedWHRouteSpec or None"
        )
    if (
        type(route_worker_count) is not int
        or not 1 <= route_worker_count <= 32
    ):
        raise RestrictedWHSourceContractError(
            "route_worker_count must be an integer in [1, 32]"
        )
    dynamics = state.dynamics_state
    required_source_steps = (
        tracer_spec.map_step_count * tracer_spec.source_steps_per_map_step
    )
    if source_spec.source_step_count != required_source_steps:
        raise RestrictedWHSourceContractError(
            "source_spec must cover exactly the tracer segment's fine source steps"
        )
    if source_spec.source_step_years != dynamics.source_step_years:
        raise RestrictedWHSourceContractError(
            "source_spec.source_step_years must equal state.source_step_years"
        )
    source_build = build_wisdom_holman_source_tile(
        tile_id=tile_id,
        epoch_origin_years=dynamics.epoch_origin_years,
        global_source_step=dynamics.global_source_step,
        time_scale=dynamics.time_scale,
        frame=dynamics.frame,
        axes=dynamics.axes,
        source_ids=dynamics.source_ids,
        primary_source_id=dynamics.primary_source_id,
        source_schedule_sha256=dynamics.source_schedule_sha256,
        initial_source_relative_positions_AU=(
            initial_source_relative_positions_AU
        ),
        initial_source_relative_velocities_AU_per_year=(
            initial_source_relative_velocities_AU_per_year
        ),
        source_gm_AU3_per_year2=source_gm_AU3_per_year2,
        source_encounter_radii_AU=source_encounter_radii_AU,
        certified_relative_acceleration_bounds_AU_per_year2=(
            certified_relative_acceleration_bounds_AU_per_year2
        ),
        provenance=source_provenance,
        spec=source_spec,
    )
    tracer_result = advance_cuda_restricted_wh_lifecycle(
        state,
        source_build.tile,
        tracer_spec,
        route_spec=route_spec,
        route_worker_count=route_worker_count,
    )
    return WisdomHolmanRestrictedWHLifecycleSegmentResult(
        source_build=source_build,
        tracer_result=tracer_result,
    )


__all__ = [
    "IMPLEMENTATION_SOURCE_SHA256",
    "COORDINATE_CANONICALIZATION_ROUNDING_ULP_FACTOR",
    "MAXIMUM_COORDINATE_CANONICALIZATION_RELATIVE_ERROR",
    "MAXIMUM_COORDINATE_CANONICALIZATION_RELATIVE_ERROR_IN_EPS",
    "MAXIMUM_COORDINATE_CANONICALIZATION_FIXED_POINT_ITERATIONS",
    "MAXIMUM_SOURCE_STEPS_PER_TILE",
    "PROVIDER_ID",
    "PROJECT_SCOPE",
    "RestrictedWHSourceContractError",
    "RestrictedWHSourceError",
    "RestrictedWHSourceIntegrationError",
    "SCIENTIFIC_CLASSIFICATION",
    "SOURCE_PROVIDER_CLAIMS",
    "SourceCoordinateCanonicalizationReceipt",
    "WisdomHolmanRestrictedWHSegmentResult",
    "WisdomHolmanRestrictedWHHybridSegmentResult",
    "WisdomHolmanRestrictedWHLifecycleSegmentResult",
    "WisdomHolmanSourceTileBuild",
    "WisdomHolmanSourceTileSpec",
    "advance_cuda_restricted_wh_with_wisdom_holman_source",
    "advance_cuda_restricted_wh_hybrid_with_wisdom_holman_source",
    "advance_cuda_restricted_wh_lifecycle_with_wisdom_holman_source",
    "build_wisdom_holman_source_tile",
]
