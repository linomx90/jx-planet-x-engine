"""Sealed M2 known-system plus observed-tracer fixed-run inputs.

This module is the narrow bridge between the retained physical M2 source,
the 108 predeclared observed/sensitivity Cartesian states, and the existing
fixed-lane CUDA/DOP853 lifecycle runner.  It performs no integration.  The
CPU-only preparation function freezes every host input and policy; the CUDA
factory then materializes those exact bytes on the requested device and
verifies the resulting lifecycle genesis before releasing it.

The vocabulary adapter changes labels only.  It does not rotate, translate,
propagate, rescale, or otherwise alter any Cartesian component.  Relative
engine time zero is explicitly mapped to JD 2461200.5 TDB.

The 0.1-AU source radii are conservative algorithmic routing surfaces, not
physical radii.  No provenance-bound physical-radius model exists for this
ten-barycenter roster, so terminal source radii are explicitly zero and no
physical contact/removal inference is authorized by this factory.

The default profile owns a one-year bootstrap source segment at 1/4096 year.
Two additional closed profiles materialize the identical tracer genesis at
the retained M2 fine-source cadence (1/8192 year), for the official 1/512-year
candidate and 1/1024-year refinement maps.  Those profiles intentionally do
not reconstruct a source segment: callers must obtain annual tiles from the
independently verified retained-M2 path loader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ..cuda_restricted_wh import (
    COORDINATE_SYSTEM,
    CUDARestrictedWHState,
    FixedStepCUDARestrictedWHSpec,
)
from ..engine import BackendSpec, Provenance
from ..engine.backends import resolve_backend
from ..restricted_wh_checkpoint import restricted_wh_lifecycle_state_sha256
from ..restricted_wh_lifecycle import (
    ACTIVE,
    RestrictedWHLifecycleState,
    initialize_restricted_wh_lifecycle,
)
from ..restricted_wh_route import (
    RestrictedWHRouteSpec,
    restricted_wh_route_policy_identity,
)
from ..restricted_wh_source import WisdomHolmanSourceTileSpec
from .initial_conditions import (
    DE440_SUN_GENESIS_MODEL,
    ObservedInitialConditionInventory,
    build_predeclared_initial_condition_inventory,
)
from .physical_models import (
    KNOWN_SOLAR_SYSTEM_V2,
    KNOWN_SYSTEM_BODY_IDS,
    KNOWN_SYSTEM_BODY_NAMES,
    KnownSolarSystemSourceModel,
)
from .m2_source_path import (
    FINE_SOURCE_SCHEDULE_SHA256,
    FINE_SOURCE_STEP_COUNT,
    FINE_SOURCE_STEP_YEARS,
    FINE_SOURCE_STEPS_PER_YEAR,
)
from .runner import FixedLongHorizonRunConfig, FixedLongHorizonSourceSegment


CONTRACT_SCHEMA = "jx.long-horizon.known-system-observed-fixed-run-inputs.v1"
ADAPTER_SCHEMA = "jx.long-horizon.known-system-vocabulary-epoch-adapter.v1"
ROSTER_SCHEMA = "jx.long-horizon.known-system-observed-tracer-roster.v1"
PROFILE_SCHEMA = "jx.long-horizon.known-system-fixed-run-profile.v1"
MATERIALIZED_SCHEMA = (
    "jx.long-horizon.known-system-observed-fixed-run-materialization.v1"
)
PROFILE_ID = "m2-known-system-observed-fixed-1y-v1"
RUN_ID = "jx-long-horizon-m2-known-system-observed-fixed-1y-v1"
SNAPSHOT_ID = "m2-known-system-observed-genesis-jd2461200p5-v1"
SOURCE_TILE_ID = "m2-known-system-source-1y-1over4096-v1"


class KnownSystemRunProfile(str, Enum):
    """Closed fixed-step profiles authorized by this bridge."""

    BOOTSTRAP_1Y = PROFILE_ID
    M2_FINE_CANDIDATE_10Y_ANNUAL = (
        "m2-retained-fine-observed-candidate-10y-annual-v1"
    )
    M2_FINE_REFINEMENT_10Y_ANNUAL = (
        "m2-retained-fine-observed-refinement-10y-annual-v1"
    )


M2_FINE_CANDIDATE_PROFILE_ID = (
    KnownSystemRunProfile.M2_FINE_CANDIDATE_10Y_ANNUAL.value
)
M2_FINE_REFINEMENT_PROFILE_ID = (
    KnownSystemRunProfile.M2_FINE_REFINEMENT_10Y_ANNUAL.value
)
M2_FINE_CANDIDATE_RUN_ID = (
    "jx-long-horizon-m3-observed-candidate-10y-annual-v1"
)
M2_FINE_REFINEMENT_RUN_ID = (
    "jx-long-horizon-m3-observed-refinement-10y-annual-v1"
)

SOURCE_STEP_YEARS = float.fromhex("0x1.0000000000000p-12")
SOURCE_STEP_COUNT = 4096
TRACER_SOURCE_STEPS_PER_MAP_STEP = 16
TRACER_MAP_STEP_COUNT = 256
TRACER_STEP_YEARS = float.fromhex("0x1.0000000000000p-8")
HORIZON_YEARS = 1
M2_FINE_HORIZON_YEARS = 10
M2_FINE_SEGMENT_YEARS = 1
M2_FINE_SEGMENT_SOURCE_STEP_COUNT = FINE_SOURCE_STEPS_PER_YEAR
M2_FINE_CANDIDATE_SOURCE_STEPS_PER_MAP_STEP = 16
M2_FINE_REFINEMENT_SOURCE_STEPS_PER_MAP_STEP = 8
M2_FINE_CANDIDATE_SEGMENT_MAP_STEP_COUNT = 512
M2_FINE_REFINEMENT_SEGMENT_MAP_STEP_COUNT = 1024
M2_FINE_CANDIDATE_STEP_YEARS = float.fromhex("0x1.0000000000000p-9")
M2_FINE_REFINEMENT_STEP_YEARS = float.fromhex("0x1.0000000000000p-10")
M2_FINE_MAXIMUM_PRIMARY_DYNAMICAL_PHASE = float.fromhex(
    "0x1.0000000000000p-6"
)
M2_FINE_PRIMARY_OUTER_BOUNDARY_AU = 10_000.0
ROUTE_WORKER_COUNT = 6
ALGORITHMIC_ROUTE_RADIUS_AU = float.fromhex("0x1.999999999999ap-4")
SOURCE_PAIR_SEPARATION_FLOOR_AU = ALGORITHMIC_ROUTE_RADIUS_AU
SOURCE_JACOBI_PERIAPSE_FLOOR_AU = float.fromhex("0x1.999999999999ap-5")
MAXIMUM_INITIAL_BARYCENTER_POSITION_NORM_AU = 1.0e-12
MAXIMUM_INITIAL_BARYCENTER_VELOCITY_NORM_AU_PER_YEAR = 1.0e-12
MAXIMUM_PRIMARY_DYNAMICAL_PHASE = float.fromhex("0x1.999999999999ap-5")
THREADS_PER_BLOCK = 128

BOOTSTRAP_SOURCE_PROVENANCE_ID = (
    "jx.long-horizon.m2-known-system-source"
)
BOOTSTRAP_SOURCE_PROVENANCE_CITATION = (
    "Retained NASA/JPL Horizons DE441 state and NAIF DE440 GM known-system "
    "model"
)
BOOTSTRAP_SOURCE_PROVENANCE_VERSION = "v2"

ENGINE_EPOCH_ORIGIN_YEARS = 0.0
ENGINE_AXES = "IAU76_80_ECLIPTIC_J2000"
ENGINE_LENGTH_UNIT = "AU"
ENGINE_TIME_UNIT = "year"
ENGINE_UNIT_SYSTEM_ID = "AU-YEAR"
EPOCH_MAPPING_POLICY = (
    "JD_TDB_EQUALS_2461200P5_PLUS_RELATIVE_JULIAN_YEARS_TIMES_365P25_DAYS_V1"
)
VOCABULARY_ADAPTER_POLICY = (
    "NO_NUMERICAL_CHANGE_EXACT_BINARY64_COMPONENT_COPY_PHYSICAL_LABELS_TO_"
    "RESTRICTED_WH_ENGINE_CONTEXT_V1"
)
TRACER_INDEX_POLICY = (
    "UINT64_TUPLE_OFFSET_IN_EXACT_OBSERVED_INVENTORY_STATE_ORDER_V1"
)
TERMINAL_RADIUS_POLICY = (
    "NO_PROVENANCE_BOUND_PHYSICAL_RADIUS_MODEL_ALL_ZERO_TERMINAL_SURFACES_V1"
)

EXPECTED_DE440_SUN_GENESIS_MODEL_SHA256 = (
    "10f4a04f080a99c84e2152b3c7e88324a1e263e29fec5c72a2f151a78d86e50e"
)
EXPECTED_DE440_SUN_INVENTORY_SHA256 = (
    "e5df3dd28bfe6dfb8bcd380cc7c1f4ea67eec5ee963e751b4feacc1f62c73ee8"
)

ALGORITHMIC_ROUTE_RADII_AU = (ALGORITHMIC_ROUTE_RADIUS_AU,) * len(
    KNOWN_SYSTEM_BODY_IDS
)
PHYSICAL_TERMINAL_RADII_AU = (0.0,) * len(KNOWN_SYSTEM_BODY_IDS)

CLAIMS = MappingProxyType(
    {
        "production_qualified": False,
        "tracer_dynamics_qualified": False,
        "long_duration_accuracy_qualified": False,
        "physical_terminal_radii_modeled": False,
        "physical_collision_inferred": False,
        "science_result": False,
    }
)

_ADAPTER_SEAL = object()
_ROSTER_SEAL = object()
_PROFILE_SEAL = object()
_CONTRACT_SEAL = object()
_SHA256_HEX = frozenset("0123456789abcdef")


class KnownSystemInputContractError(ValueError):
    """The sealed M2 observed fixed-run input contract is inconsistent."""


def _sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
        or value == "0" * 64
    ):
        raise KnownSystemInputContractError(
            f"{label} must be a nonzero lowercase SHA-256 digest"
        )
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise KnownSystemInputContractError(
            f"{label} must be a nonempty trimmed built-in string"
        )
    return value


def _canonical_sha256(domain: bytes, value: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError) as exc:
        raise KnownSystemInputContractError(
            "known-system input identity is not strict canonical JSON"
        ) from exc
    return hashlib.sha256(domain + b"\0" + encoded).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(value.ndim.to_bytes(4, "little"))
    for dimension in value.shape:
        digest.update(int(dimension).to_bytes(8, "little"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _frozen_float64_matrix(
    value: object, label: str, expected_shape: tuple[int, int]
) -> np.ndarray:
    if type(value) is not np.ndarray or value.dtype != np.dtype(np.float64):
        raise KnownSystemInputContractError(
            f"{label} must be an exact binary64 NumPy array"
        )
    if value.shape != expected_shape or not value.flags.c_contiguous:
        raise KnownSystemInputContractError(
            f"{label} must be C-contiguous with shape {expected_shape}"
        )
    if not bool(np.all(np.isfinite(value))):
        raise KnownSystemInputContractError(f"{label} must be finite")
    return np.frombuffer(
        value.tobytes(order="C"), dtype=np.float64
    ).reshape(expected_shape)


def _bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(
        left.shape == right.shape
        and left.dtype == np.dtype(np.float64)
        and right.dtype == np.dtype(np.float64)
        and np.array_equal(left.view(np.uint64), right.view(np.uint64))
    )


def _uniform_acceleration_bound(gm: np.ndarray, radius: float) -> float:
    total_gm = sum(
        (Fraction.from_float(float(value)) for value in gm), Fraction(0, 1)
    )
    exact = (2 * total_gm) / Fraction.from_float(radius) ** 2
    bound = float(exact)
    if Fraction.from_float(bound) < exact:
        bound = math.nextafter(bound, math.inf)
    if not math.isfinite(bound) or bound <= 0.0:
        raise KnownSystemInputContractError(
            "known-system acceleration certificate is not finite and positive"
        )
    return bound


@dataclass(frozen=True, slots=True)
class KnownSystemVocabularyEpochReceipt:
    """Sealed label/epoch adapter that performs no numerical state change."""

    known_system_model_sha256: str
    genesis_model_sha256: str
    epoch_jd_tdb: float
    epoch_scale: str
    engine_epoch_origin_years: float
    physical_frame: str
    physical_coordinate_system: str
    physical_time_unit: str
    engine_frame: str
    engine_axes: str
    engine_coordinate_system: str
    engine_length_unit: str
    engine_time_unit: str
    engine_unit_system_id: str
    physical_primary_id: str
    physical_primary_body_id: int
    engine_primary_source_id: str
    epoch_mapping_policy: str
    vocabulary_adapter_policy: str
    _seal: object = field(repr=False, compare=False)
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self._seal is not _ADAPTER_SEAL:
            raise KnownSystemInputContractError(
                "vocabulary/epoch receipts can only be issued by this module"
            )
        _sha256(self.known_system_model_sha256, "known_system_model_sha256")
        _sha256(self.genesis_model_sha256, "genesis_model_sha256")
        expected = {
            "known_system_model_sha256": KNOWN_SOLAR_SYSTEM_V2.content_sha256,
            "genesis_model_sha256": DE440_SUN_GENESIS_MODEL.content_sha256,
            "epoch_jd_tdb": KNOWN_SOLAR_SYSTEM_V2.epoch_jd_tdb,
            "epoch_scale": KNOWN_SOLAR_SYSTEM_V2.epoch_scale,
            "engine_epoch_origin_years": ENGINE_EPOCH_ORIGIN_YEARS,
            "physical_frame": KNOWN_SOLAR_SYSTEM_V2.output_frame,
            "physical_coordinate_system": (
                KNOWN_SOLAR_SYSTEM_V2.output_coordinate_system
            ),
            "physical_time_unit": KNOWN_SOLAR_SYSTEM_V2.time_unit,
            "engine_frame": KNOWN_SOLAR_SYSTEM_V2.output_frame,
            "engine_axes": ENGINE_AXES,
            "engine_coordinate_system": COORDINATE_SYSTEM,
            "engine_length_unit": ENGINE_LENGTH_UNIT,
            "engine_time_unit": ENGINE_TIME_UNIT,
            "engine_unit_system_id": ENGINE_UNIT_SYSTEM_ID,
            "physical_primary_id": KNOWN_SOLAR_SYSTEM_V2.primary_id,
            "physical_primary_body_id": KNOWN_SYSTEM_BODY_IDS[0],
            "engine_primary_source_id": KNOWN_SYSTEM_BODY_NAMES[0],
            "epoch_mapping_policy": EPOCH_MAPPING_POLICY,
            "vocabulary_adapter_policy": VOCABULARY_ADAPTER_POLICY,
        }
        for name, value in expected.items():
            observed = getattr(self, name)
            if type(observed) is not type(value):
                raise KnownSystemInputContractError(
                    f"adapter field {name} has the wrong exact type"
                )
            if type(value) is float:
                if not math.isfinite(observed) or observed.hex() != value.hex():
                    raise KnownSystemInputContractError(
                        f"adapter field {name} changed"
                    )
            elif observed != value:
                raise KnownSystemInputContractError(
                    f"adapter field {name} changed"
                )
        if self.engine_epoch_origin_years.hex() != "0x0.0p+0":
            raise KnownSystemInputContractError(
                "engine relative epoch zero must be exact positive zero"
            )
        object.__setattr__(
            self,
            "content_sha256",
            _canonical_sha256(
                b"jx.long-horizon.known-system-vocabulary-epoch-adapter.v1",
                self.as_record(),
            ),
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "schema": ADAPTER_SCHEMA,
            "known_system_model_sha256": self.known_system_model_sha256,
            "genesis_model_sha256": self.genesis_model_sha256,
            "epoch_jd_tdb_hex": self.epoch_jd_tdb.hex(),
            "epoch_scale": self.epoch_scale,
            "engine_epoch_origin_years_hex": (
                self.engine_epoch_origin_years.hex()
            ),
            "physical_frame": self.physical_frame,
            "physical_coordinate_system": self.physical_coordinate_system,
            "physical_time_unit": self.physical_time_unit,
            "engine_frame": self.engine_frame,
            "engine_axes": self.engine_axes,
            "engine_coordinate_system": self.engine_coordinate_system,
            "engine_length_unit": self.engine_length_unit,
            "engine_time_unit": self.engine_time_unit,
            "engine_unit_system_id": self.engine_unit_system_id,
            "physical_primary_id": self.physical_primary_id,
            "physical_primary_body_id": self.physical_primary_body_id,
            "engine_primary_source_id": self.engine_primary_source_id,
            "epoch_mapping_policy": self.epoch_mapping_policy,
            "vocabulary_adapter_policy": self.vocabulary_adapter_policy,
            "numerical_position_transform": "NONE_BITWISE_COPY",
            "numerical_velocity_transform": "NONE_BITWISE_COPY",
        }


@dataclass(frozen=True, slots=True)
class KnownSystemTracerRoster:
    """Exact inventory-order mapping from object IDs to uint64 tracer IDs."""

    inventory_content_sha256: str
    object_ids: tuple[str, ...]
    state_content_sha256: tuple[str, ...]
    tracer_indices: tuple[int, ...]
    assignment_policy: str
    _seal: object = field(repr=False, compare=False)
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self._seal is not _ROSTER_SEAL:
            raise KnownSystemInputContractError(
                "tracer rosters can only be issued by this module"
            )
        _sha256(self.inventory_content_sha256, "inventory_content_sha256")
        if (
            type(self.object_ids) is not tuple
            or len(self.object_ids) != 108
            or any(type(item) is not str or not item for item in self.object_ids)
            or len(set(self.object_ids)) != 108
        ):
            raise KnownSystemInputContractError(
                "roster must contain exactly 108 unique object IDs"
            )
        if (
            type(self.state_content_sha256) is not tuple
            or len(self.state_content_sha256) != 108
        ):
            raise KnownSystemInputContractError(
                "roster must bind exactly 108 state digests"
            )
        for index, digest in enumerate(self.state_content_sha256):
            _sha256(digest, f"state_content_sha256[{index}]")
        if (
            type(self.tracer_indices) is not tuple
            or self.tracer_indices != tuple(range(108))
            or any(type(item) is not int for item in self.tracer_indices)
        ):
            raise KnownSystemInputContractError(
                "tracer indices must be exact sequential integers 0 through 107"
            )
        if self.assignment_policy != TRACER_INDEX_POLICY:
            raise KnownSystemInputContractError(
                "tracer-index assignment policy changed"
            )
        object.__setattr__(
            self,
            "content_sha256",
            _canonical_sha256(
                b"jx.long-horizon.known-system-observed-tracer-roster.v1",
                {
                    "schema": ROSTER_SCHEMA,
                    "inventory_content_sha256": self.inventory_content_sha256,
                    "assignment_policy": self.assignment_policy,
                    "rows": [
                        {
                            "tracer_index": tracer_index,
                            "object_id": object_id,
                            "state_content_sha256": state_digest,
                        }
                        for tracer_index, object_id, state_digest in zip(
                            self.tracer_indices,
                            self.object_ids,
                            self.state_content_sha256,
                            strict=True,
                        )
                    ],
                },
            ),
        )

    def object_id_for_tracer_index(self, tracer_index: int) -> str:
        if type(tracer_index) is not int or not 0 <= tracer_index < 108:
            raise KnownSystemInputContractError(
                "tracer_index must be an exact integer in [0, 107]"
            )
        return self.object_ids[tracer_index]


def _source_schedule_sha256(
    model: KnownSolarSystemSourceModel,
    adapter: KnownSystemVocabularyEpochReceipt,
) -> str:
    return _canonical_sha256(
        b"jx.long-horizon.known-system-source-schedule.v1",
        {
            "profile_id": PROFILE_ID,
            "known_system_model_sha256": model.content_sha256,
            "adapter_content_sha256": adapter.content_sha256,
            "epoch_jd_tdb_hex": model.epoch_jd_tdb.hex(),
            "source_step_years_hex": SOURCE_STEP_YEARS.hex(),
            "source_step_count": SOURCE_STEP_COUNT,
            "horizon_years": HORIZON_YEARS,
            "source_integrator": (
                "REPLAY_VERIFIED_ORDERED_JACOBI_WISDOM_HOLMAN"
            ),
        },
    )


def _profile_values(
    profile_id: KnownSystemRunProfile,
    adapter: KnownSystemVocabularyEpochReceipt,
) -> dict[str, Any]:
    if type(profile_id) is not KnownSystemRunProfile:
        raise KnownSystemInputContractError(
            "profile_id must be an exact KnownSystemRunProfile member"
        )
    common = {
        "known_system_model_sha256": KNOWN_SOLAR_SYSTEM_V2.content_sha256,
        "adapter_content_sha256": adapter.content_sha256,
        "route_worker_count": ROUTE_WORKER_COUNT,
        "multirate_enabled": False,
        "owned_episode_enabled": False,
    }
    if profile_id is KnownSystemRunProfile.BOOTSTRAP_1Y:
        return {
            **common,
            "profile_id": profile_id,
            "run_id": RUN_ID,
            "snapshot_id": SNAPSHOT_ID,
            "source_mode": "BOOTSTRAP_GENERATED_SOURCE_SEGMENT",
            "source_schedule_sha256": _source_schedule_sha256(
                KNOWN_SOLAR_SYSTEM_V2, adapter
            ),
            "source_step_years": SOURCE_STEP_YEARS,
            "total_horizon_years": HORIZON_YEARS,
            "segment_horizon_years": HORIZON_YEARS,
            "total_source_step_count": SOURCE_STEP_COUNT,
            "segment_source_step_count": SOURCE_STEP_COUNT,
            "source_steps_per_map_step": (
                TRACER_SOURCE_STEPS_PER_MAP_STEP
            ),
            "segment_map_step_count": TRACER_MAP_STEP_COUNT,
            "tracer_step_years": TRACER_STEP_YEARS,
            "maximum_primary_dynamical_phase": (
                MAXIMUM_PRIMARY_DYNAMICAL_PHASE
            ),
            "primary_outer_boundary_AU": None,
        }
    if profile_id is KnownSystemRunProfile.M2_FINE_CANDIDATE_10Y_ANNUAL:
        return {
            **common,
            "profile_id": profile_id,
            "run_id": M2_FINE_CANDIDATE_RUN_ID,
            "snapshot_id": (
                "m2-fine-observed-candidate-genesis-jd2461200p5-v1"
            ),
            "source_mode": "RETAINED_M2_FINE_SOURCE_PATH",
            "source_schedule_sha256": FINE_SOURCE_SCHEDULE_SHA256,
            "source_step_years": FINE_SOURCE_STEP_YEARS,
            "total_horizon_years": M2_FINE_HORIZON_YEARS,
            "segment_horizon_years": M2_FINE_SEGMENT_YEARS,
            "total_source_step_count": FINE_SOURCE_STEP_COUNT,
            "segment_source_step_count": (
                M2_FINE_SEGMENT_SOURCE_STEP_COUNT
            ),
            "source_steps_per_map_step": (
                M2_FINE_CANDIDATE_SOURCE_STEPS_PER_MAP_STEP
            ),
            "segment_map_step_count": (
                M2_FINE_CANDIDATE_SEGMENT_MAP_STEP_COUNT
            ),
            "tracer_step_years": M2_FINE_CANDIDATE_STEP_YEARS,
            "maximum_primary_dynamical_phase": (
                M2_FINE_MAXIMUM_PRIMARY_DYNAMICAL_PHASE
            ),
            "primary_outer_boundary_AU": (
                M2_FINE_PRIMARY_OUTER_BOUNDARY_AU
            ),
        }
    if profile_id is KnownSystemRunProfile.M2_FINE_REFINEMENT_10Y_ANNUAL:
        return {
            **common,
            "profile_id": profile_id,
            "run_id": M2_FINE_REFINEMENT_RUN_ID,
            "snapshot_id": (
                "m2-fine-observed-refinement-genesis-jd2461200p5-v1"
            ),
            "source_mode": "RETAINED_M2_FINE_SOURCE_PATH",
            "source_schedule_sha256": FINE_SOURCE_SCHEDULE_SHA256,
            "source_step_years": FINE_SOURCE_STEP_YEARS,
            "total_horizon_years": M2_FINE_HORIZON_YEARS,
            "segment_horizon_years": M2_FINE_SEGMENT_YEARS,
            "total_source_step_count": FINE_SOURCE_STEP_COUNT,
            "segment_source_step_count": (
                M2_FINE_SEGMENT_SOURCE_STEP_COUNT
            ),
            "source_steps_per_map_step": (
                M2_FINE_REFINEMENT_SOURCE_STEPS_PER_MAP_STEP
            ),
            "segment_map_step_count": (
                M2_FINE_REFINEMENT_SEGMENT_MAP_STEP_COUNT
            ),
            "tracer_step_years": M2_FINE_REFINEMENT_STEP_YEARS,
            "maximum_primary_dynamical_phase": (
                M2_FINE_MAXIMUM_PRIMARY_DYNAMICAL_PHASE
            ),
            "primary_outer_boundary_AU": (
                M2_FINE_PRIMARY_OUTER_BOUNDARY_AU
            ),
        }
    raise AssertionError("unreachable closed profile member")


@dataclass(frozen=True, slots=True)
class KnownSystemRunProfileReceipt:
    """Sealed fixed-step profile and retained-source schedule receipt."""

    profile_id: KnownSystemRunProfile
    run_id: str
    snapshot_id: str
    source_mode: str
    known_system_model_sha256: str
    adapter_content_sha256: str
    source_schedule_sha256: str
    source_step_years: float
    total_horizon_years: int
    segment_horizon_years: int
    total_source_step_count: int
    segment_source_step_count: int
    source_steps_per_map_step: int
    segment_map_step_count: int
    tracer_step_years: float
    maximum_primary_dynamical_phase: float
    primary_outer_boundary_AU: float | None
    route_worker_count: int
    multirate_enabled: bool
    owned_episode_enabled: bool
    _seal: object = field(repr=False, compare=False)
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self._seal is not _PROFILE_SEAL:
            raise KnownSystemInputContractError(
                "run-profile receipts can only be issued by this module"
            )
        expected = _profile_values(self.profile_id, _adapter_receipt())
        for name, value in expected.items():
            observed = getattr(self, name)
            if type(observed) is not type(value):
                raise KnownSystemInputContractError(
                    f"run-profile field {name} has the wrong exact type"
                )
            if type(value) is float:
                if not math.isfinite(observed) or observed.hex() != value.hex():
                    raise KnownSystemInputContractError(
                        f"run-profile field {name} changed"
                    )
            elif observed != value:
                raise KnownSystemInputContractError(
                    f"run-profile field {name} changed"
                )
        if self.segment_source_step_count != (
            self.segment_map_step_count * self.source_steps_per_map_step
        ):
            raise KnownSystemInputContractError(
                "run-profile source and tracer segment lattices differ"
            )
        if self.total_source_step_count != (
            self.total_horizon_years
            * self.segment_source_step_count
            // self.segment_horizon_years
        ):
            raise KnownSystemInputContractError(
                "run-profile total and segmented source lattices differ"
            )
        if self.tracer_step_years.hex() != (
            self.source_step_years * self.source_steps_per_map_step
        ).hex():
            raise KnownSystemInputContractError(
                "run-profile tracer step is not its exact source-step ratio"
            )
        object.__setattr__(
            self,
            "content_sha256",
            _canonical_sha256(
                b"jx.long-horizon.known-system-fixed-run-profile.v1",
                self.as_record(),
            ),
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "schema": PROFILE_SCHEMA,
            "profile_id": self.profile_id.value,
            "run_id": self.run_id,
            "snapshot_id": self.snapshot_id,
            "source_mode": self.source_mode,
            "known_system_model_sha256": self.known_system_model_sha256,
            "adapter_content_sha256": self.adapter_content_sha256,
            "source_schedule_sha256": self.source_schedule_sha256,
            "source_step_years_hex": self.source_step_years.hex(),
            "total_horizon_years": self.total_horizon_years,
            "segment_horizon_years": self.segment_horizon_years,
            "total_source_step_count": self.total_source_step_count,
            "segment_source_step_count": self.segment_source_step_count,
            "source_steps_per_map_step": self.source_steps_per_map_step,
            "segment_map_step_count": self.segment_map_step_count,
            "tracer_step_years_hex": self.tracer_step_years.hex(),
            "maximum_primary_dynamical_phase_hex": (
                self.maximum_primary_dynamical_phase.hex()
            ),
            "primary_outer_boundary_AU_hex": (
                None
                if self.primary_outer_boundary_AU is None
                else self.primary_outer_boundary_AU.hex()
            ),
            "route_worker_count": self.route_worker_count,
            "multirate_enabled": self.multirate_enabled,
            "owned_episode_enabled": self.owned_episode_enabled,
        }


def _profile_receipt(
    profile_id: KnownSystemRunProfile,
    adapter: KnownSystemVocabularyEpochReceipt,
) -> KnownSystemRunProfileReceipt:
    return KnownSystemRunProfileReceipt(
        **_profile_values(profile_id, adapter),
        _seal=_PROFILE_SEAL,
    )


@dataclass(frozen=True, eq=False)
class KnownSystemObservedFixedRunContract:
    """CPU-prepared, content-addressed fixed-run genesis and policy."""

    known_system_model: KnownSolarSystemSourceModel
    inventory: ObservedInitialConditionInventory
    roster: KnownSystemTracerRoster
    adapter: KnownSystemVocabularyEpochReceipt
    profile: KnownSystemRunProfileReceipt
    tracer_positions_AU: np.ndarray
    tracer_velocities_AU_per_year: np.ndarray
    source_schedule_sha256: str
    source: FixedLongHorizonSourceSegment | None
    config: FixedLongHorizonRunConfig
    _seal: object = field(repr=False, compare=False)
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self._seal is not _CONTRACT_SEAL:
            raise KnownSystemInputContractError(
                "fixed-run contracts can only be issued by this module"
            )
        if (
            type(self.known_system_model) is not KnownSolarSystemSourceModel
            or self.known_system_model.content_sha256
            != KNOWN_SOLAR_SYSTEM_V2.content_sha256
        ):
            raise KnownSystemInputContractError(
                "fixed-run contract does not bind the exact M2 source model"
            )
        if type(self.inventory) is not ObservedInitialConditionInventory:
            raise KnownSystemInputContractError(
                "inventory must be an exact ObservedInitialConditionInventory"
            )
        if (
            self.inventory.content_sha256
            != EXPECTED_DE440_SUN_INVENTORY_SHA256
            or self.inventory.model.content_sha256
            != EXPECTED_DE440_SUN_GENESIS_MODEL_SHA256
            or self.inventory.model != DE440_SUN_GENESIS_MODEL
            or self.inventory.state_count != 108
        ):
            raise KnownSystemInputContractError(
                "inventory is not the exact 108-state DE440-Sun genesis"
            )
        if type(self.roster) is not KnownSystemTracerRoster:
            raise KnownSystemInputContractError(
                "roster must be an exact KnownSystemTracerRoster"
            )
        expected_ids = tuple(state.object_id for state in self.inventory.states)
        expected_state_digests = tuple(
            state.content_sha256 for state in self.inventory.states
        )
        if (
            self.roster.inventory_content_sha256
            != self.inventory.content_sha256
            or self.roster.object_ids != expected_ids
            or self.roster.state_content_sha256 != expected_state_digests
        ):
            raise KnownSystemInputContractError(
                "tracer roster differs from exact inventory order"
            )
        if type(self.adapter) is not KnownSystemVocabularyEpochReceipt:
            raise KnownSystemInputContractError(
                "adapter must be an exact KnownSystemVocabularyEpochReceipt"
            )
        self.adapter.__post_init__()
        if (
            self.adapter.known_system_model_sha256
            != self.known_system_model.content_sha256
            or self.adapter.genesis_model_sha256
            != self.inventory.model.content_sha256
        ):
            raise KnownSystemInputContractError(
                "adapter names a different physical source or genesis model"
            )
        if type(self.profile) is not KnownSystemRunProfileReceipt:
            raise KnownSystemInputContractError(
                "profile must be an exact KnownSystemRunProfileReceipt"
            )
        self.profile.__post_init__()
        if (
            self.profile.known_system_model_sha256
            != self.known_system_model.content_sha256
            or self.profile.adapter_content_sha256
            != self.adapter.content_sha256
        ):
            raise KnownSystemInputContractError(
                "run profile names a different source model or adapter"
            )
        for state in self.inventory.states:
            if (
                state.epoch_jd_tdb.hex()
                != self.adapter.epoch_jd_tdb.hex()
                or state.epoch_scale != self.adapter.epoch_scale
                or state.frame != self.adapter.physical_frame
                or state.coordinate_system
                != self.adapter.physical_coordinate_system
                or state.time_unit != self.adapter.physical_time_unit
                or state.primary_id != self.adapter.physical_primary_id
                or state.primary_gm_AU3_per_year2.hex()
                != self.known_system_model.primary_gm_AU3_per_julian_year2.hex()
            ):
                raise KnownSystemInputContractError(
                    f"state {state.object_id} differs from the sealed adapter"
                )
        positions = _frozen_float64_matrix(
            self.tracer_positions_AU, "tracer_positions_AU", (108, 3)
        )
        velocities = _frozen_float64_matrix(
            self.tracer_velocities_AU_per_year,
            "tracer_velocities_AU_per_year",
            (108, 3),
        )
        expected_positions = np.ascontiguousarray(
            [state.position_AU for state in self.inventory.states],
            dtype=np.float64,
        )
        expected_velocities = np.ascontiguousarray(
            [state.velocity_AU_per_year for state in self.inventory.states],
            dtype=np.float64,
        )
        if not _bitwise_equal(positions, expected_positions) or not _bitwise_equal(
            velocities, expected_velocities
        ):
            raise KnownSystemInputContractError(
                "host tracer arrays differ bitwise from the sealed inventory"
            )
        object.__setattr__(self, "tracer_positions_AU", positions)
        object.__setattr__(self, "tracer_velocities_AU_per_year", velocities)
        if (
            self.source_schedule_sha256
            != self.profile.source_schedule_sha256
        ):
            raise KnownSystemInputContractError(
                "source schedule identity differs from the fixed profile"
            )
        if type(self.config) is not FixedLongHorizonRunConfig:
            raise KnownSystemInputContractError(
                "config must be an exact FixedLongHorizonRunConfig"
            )
        self._validate_source_and_config()
        object.__setattr__(
            self,
            "content_sha256",
            _canonical_sha256(
                b"jx.long-horizon.known-system-observed-fixed-run-inputs.v1",
                self.as_record(),
            ),
        )

    def _validate_source_and_config(self) -> None:
        source = self.source
        config = self.config
        model = self.known_system_model
        profile = self.profile
        if profile.source_mode == "RETAINED_M2_FINE_SOURCE_PATH":
            if source is not None:
                raise KnownSystemInputContractError(
                    "retained-M2-fine profiles must obtain source tiles from "
                    "the verified path loader"
                )
        elif type(source) is not FixedLongHorizonSourceSegment:
            raise KnownSystemInputContractError(
                "bootstrap profile requires its exact source segment"
            )
        if source is not None:
            self._validate_bootstrap_source(source)
        route_spec = config.route_spec
        if route_spec is None:
            raise KnownSystemInputContractError(
                "fixed profile requires an explicit route specification"
            )
        canonical_route_spec = RestrictedWHRouteSpec(
            maximum_primary_dynamical_phase=(
                profile.maximum_primary_dynamical_phase
            ),
            primary_outer_boundary_AU=profile.primary_outer_boundary_AU,
            terminal_source_radii_AU=PHYSICAL_TERMINAL_RADII_AU,
        )
        config_matches = (
            config.run_id == profile.run_id
            and config.source_spec.source_step_years.hex()
            == profile.source_step_years.hex()
            and config.source_spec.source_step_count
            == profile.segment_source_step_count
            and config.source_spec.minimum_encounter_pair_separation_AU.hex()
            == SOURCE_PAIR_SEPARATION_FLOOR_AU.hex()
            and config.source_spec.minimum_jacobi_periapse_AU.hex()
            == SOURCE_JACOBI_PERIAPSE_FLOOR_AU.hex()
            and config.source_spec
            .maximum_initial_barycenter_position_norm_AU.hex()
            == MAXIMUM_INITIAL_BARYCENTER_POSITION_NORM_AU.hex()
            and config.source_spec
            .maximum_initial_barycenter_velocity_norm_AU_per_year.hex()
            == MAXIMUM_INITIAL_BARYCENTER_VELOCITY_NORM_AU_PER_YEAR.hex()
            and config.tracer_spec.map_step_count
            == profile.segment_map_step_count
            and config.tracer_spec.source_steps_per_map_step
            == profile.source_steps_per_map_step
            and config.tracer_spec.maximum_primary_dynamical_phase.hex()
            == profile.maximum_primary_dynamical_phase.hex()
            and config.tracer_spec.threads_per_block == THREADS_PER_BLOCK
            and config.tracer_spec.primary_outer_boundary_AU
            == profile.primary_outer_boundary_AU
            and config.tracer_spec.backend.backend_id == "cupy"
            and config.tracer_spec.backend.tile_size == THREADS_PER_BLOCK
            and config.route_worker_count == profile.route_worker_count
            and config.multirate_enabled is profile.multirate_enabled
            and config.owned_episode_enabled is profile.owned_episode_enabled
            and route_spec.maximum_primary_dynamical_phase.hex()
            == profile.maximum_primary_dynamical_phase.hex()
            and route_spec.primary_outer_boundary_AU
            == profile.primary_outer_boundary_AU
            and restricted_wh_route_policy_identity(route_spec)[
                "content_sha256"
            ]
            == restricted_wh_route_policy_identity(canonical_route_spec)[
                "content_sha256"
            ]
            and route_spec.terminal_source_radii_AU is not None
            and len(route_spec.terminal_source_radii_AU)
            == len(PHYSICAL_TERMINAL_RADII_AU)
            and all(
                type(observed) is float
                and observed.hex() == expected.hex()
                for observed, expected in zip(
                    route_spec.terminal_source_radii_AU,
                    PHYSICAL_TERMINAL_RADII_AU,
                    strict=True,
                )
            )
        )
        if not config_matches:
            raise KnownSystemInputContractError(
                "fixed runner configuration differs from the sealed profile"
            )

    def _validate_bootstrap_source(
        self, source: FixedLongHorizonSourceSegment
    ) -> None:
        model = self.known_system_model
        context_matches = (
            source.tile_id == SOURCE_TILE_ID
            and source.epoch_origin_years.hex()
            == self.adapter.engine_epoch_origin_years.hex()
            and source.global_source_step == 0
            and source.source_step_years.hex()
            == self.profile.source_step_years.hex()
            and source.time_scale == self.adapter.epoch_scale
            and source.frame == self.adapter.engine_frame
            and source.axes == self.adapter.engine_axes
            and source.coordinate_system
            == self.adapter.engine_coordinate_system
            and source.length_unit == self.adapter.engine_length_unit
            and source.time_unit == self.adapter.engine_time_unit
            and source.unit_system_id == self.adapter.engine_unit_system_id
            and source.source_ids == model.body_names
            and source.primary_source_id
            == self.adapter.engine_primary_source_id
            and source.source_schedule_sha256 == self.source_schedule_sha256
            and source.source_provenance.sha256 == model.content_sha256
            and source.source_provenance.source_id
            == BOOTSTRAP_SOURCE_PROVENANCE_ID
            and source.source_provenance.citation
            == BOOTSTRAP_SOURCE_PROVENANCE_CITATION
            and source.source_provenance.version
            == BOOTSTRAP_SOURCE_PROVENANCE_VERSION
        )
        if not context_matches:
            raise KnownSystemInputContractError(
                "source context differs from the sealed adapter/profile"
            )
        expected_source_positions = np.ascontiguousarray(
            model.positions_primary_relative_ecliptic_AU, dtype=np.float64
        )
        expected_source_velocities = np.ascontiguousarray(
            model.velocities_primary_relative_ecliptic_AU_per_julian_year,
            dtype=np.float64,
        )
        expected_gm = np.ascontiguousarray(
            model.gm_AU3_per_julian_year2, dtype=np.float64
        )
        if not _bitwise_equal(
            source.initial_source_relative_positions_AU,
            expected_source_positions,
        ) or not _bitwise_equal(
            source.initial_source_relative_velocities_AU_per_year,
            expected_source_velocities,
        ):
            raise KnownSystemInputContractError(
                "source Cartesian state differs bitwise from M2"
            )
        if not _bitwise_equal(
            source.source_gm_AU3_per_year2.reshape(-1, 1),
            expected_gm.reshape(-1, 1),
        ):
            raise KnownSystemInputContractError(
                "source GM differs bitwise from M2"
            )
        if tuple(float(value) for value in source.source_encounter_radii_AU) != (
            ALGORITHMIC_ROUTE_RADII_AU
        ):
            raise KnownSystemInputContractError(
                "source algorithmic route radii changed"
            )
        expected_bound = _uniform_acceleration_bound(
            expected_gm, ALGORITHMIC_ROUTE_RADIUS_AU
        )
        if any(
            float(value).hex() != expected_bound.hex()
            for value in source.certified_relative_acceleration_bounds_AU_per_year2
        ):
            raise KnownSystemInputContractError(
                "source acceleration-bound certificate changed"
            )

    def as_record(self) -> dict[str, Any]:
        return {
            "schema": CONTRACT_SCHEMA,
            "profile_id": self.profile.profile_id.value,
            "profile_receipt_sha256": self.profile.content_sha256,
            "run_id": self.profile.run_id,
            "known_system_model_sha256": (
                self.known_system_model.content_sha256
            ),
            "known_system_converted_state_sha256": (
                self.known_system_model.converted_state_sha256
            ),
            "inventory_content_sha256": self.inventory.content_sha256,
            "genesis_model_sha256": self.inventory.model.content_sha256,
            "roster_content_sha256": self.roster.content_sha256,
            "adapter_content_sha256": self.adapter.content_sha256,
            "source_schedule_sha256": self.source_schedule_sha256,
            "source_segment_content_sha256": (
                None if self.source is None else self.source.content_sha256
            ),
            "runner_protocol_sha256": self.config.protocol_sha256,
            "tracer_positions": {
                "dtype": self.tracer_positions_AU.dtype.str,
                "shape": list(self.tracer_positions_AU.shape),
                "sha256": _array_sha256(self.tracer_positions_AU),
            },
            "tracer_velocities": {
                "dtype": self.tracer_velocities_AU_per_year.dtype.str,
                "shape": list(self.tracer_velocities_AU_per_year.shape),
                "sha256": _array_sha256(
                    self.tracer_velocities_AU_per_year
                ),
            },
            "profile": {
                "source_mode": self.profile.source_mode,
                "total_horizon_years": self.profile.total_horizon_years,
                "segment_horizon_years": (
                    self.profile.segment_horizon_years
                ),
                "source_step_years_hex": (
                    self.profile.source_step_years.hex()
                ),
                "total_source_step_count": (
                    self.profile.total_source_step_count
                ),
                "segment_source_step_count": (
                    self.profile.segment_source_step_count
                ),
                "tracer_step_years_hex": (
                    self.profile.tracer_step_years.hex()
                ),
                "segment_tracer_map_step_count": (
                    self.profile.segment_map_step_count
                ),
                "source_steps_per_map_step": (
                    self.profile.source_steps_per_map_step
                ),
                "algorithmic_route_radii_AU_hex": [
                    value.hex() for value in ALGORITHMIC_ROUTE_RADII_AU
                ],
                "terminal_source_radii_AU_hex": [
                    value.hex() for value in PHYSICAL_TERMINAL_RADII_AU
                ],
                "terminal_radius_policy": TERMINAL_RADIUS_POLICY,
                "route_worker_count": ROUTE_WORKER_COUNT,
                "multirate_enabled": False,
                "owned_episode_enabled": False,
                "backend": {
                    "backend_id": self.config.tracer_spec.backend.backend_id,
                    "device": self.config.tracer_spec.backend.device,
                    "tile_size": self.config.tracer_spec.backend.tile_size,
                    "dtype": self.config.tracer_spec.backend.dtype,
                    "allow_fallback": (
                        self.config.tracer_spec.backend.allow_fallback
                    ),
                    "deterministic_reductions": (
                        self.config.tracer_spec.backend.deterministic_reductions
                    ),
                    "fast_math": self.config.tracer_spec.backend.fast_math,
                    "determinism_scope": (
                        self.config.tracer_spec.backend.determinism_scope
                    ),
                },
            },
            "claims": dict(CLAIMS),
        }


@dataclass(frozen=True, eq=False)
class KnownSystemObservedFixedRunInputs:
    """Verified CUDA lifecycle materialization of one prepared contract."""

    contract: KnownSystemObservedFixedRunContract
    lifecycle_state: RestrictedWHLifecycleState
    genesis_state_sha256: str
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.contract) is not KnownSystemObservedFixedRunContract:
            raise KnownSystemInputContractError(
                "contract must be an exact KnownSystemObservedFixedRunContract"
            )
        if type(self.lifecycle_state) is not RestrictedWHLifecycleState:
            raise KnownSystemInputContractError(
                "lifecycle_state must be an exact RestrictedWHLifecycleState"
            )
        _sha256(self.genesis_state_sha256, "genesis_state_sha256")
        state = self.lifecycle_state.dynamics_state
        expected_context = (
            state.snapshot_id == self.contract.profile.snapshot_id
            and state.epoch_origin_years.hex()
            == self.contract.adapter.engine_epoch_origin_years.hex()
            and state.global_source_step == 0
            and state.source_step_years.hex()
            == self.contract.profile.source_step_years.hex()
            and state.time_scale == self.contract.adapter.epoch_scale
            and state.frame == self.contract.adapter.engine_frame
            and state.axes == self.contract.adapter.engine_axes
            and state.coordinate_system
            == self.contract.adapter.engine_coordinate_system
            and state.length_unit == self.contract.adapter.engine_length_unit
            and state.time_unit == self.contract.adapter.engine_time_unit
            and state.unit_system_id
            == self.contract.adapter.engine_unit_system_id
            and state.source_ids == self.contract.known_system_model.body_names
            and state.primary_source_id
            == self.contract.adapter.engine_primary_source_id
            and state.source_schedule_sha256
            == self.contract.source_schedule_sha256
            and state.provenance.sha256 == self.contract.content_sha256
        )
        if not expected_context:
            raise KnownSystemInputContractError(
                "materialized CUDA state context differs from its contract"
            )
        backend = self.contract.config.tracer_spec.backend
        resolved = resolve_backend(backend)
        cp = resolved.xp
        with resolved.activate():
            positions = resolved.require_native_array(
                state.tracer_relative_positions_AU,
                "tracer_relative_positions_AU",
            )
            velocities = resolved.require_native_array(
                state.tracer_relative_velocities_AU_per_year,
                "tracer_relative_velocities_AU_per_year",
            )
            indices = resolved.require_native_array(
                state.tracer_indices, "tracer_indices"
            )
            host_positions = np.ascontiguousarray(
                cp.asnumpy(positions), dtype=np.float64
            )
            host_velocities = np.ascontiguousarray(
                cp.asnumpy(velocities), dtype=np.float64
            )
            host_indices = np.ascontiguousarray(
                cp.asnumpy(indices), dtype=np.uint64
            )
            cp.cuda.get_current_stream().synchronize()
        if not _bitwise_equal(
            host_positions, self.contract.tracer_positions_AU
        ) or not _bitwise_equal(
            host_velocities, self.contract.tracer_velocities_AU_per_year
        ):
            raise KnownSystemInputContractError(
                "materialized CUDA state changed a Cartesian component"
            )
        if not np.array_equal(
            host_indices, np.arange(108, dtype=np.uint64)
        ):
            raise KnownSystemInputContractError(
                "materialized CUDA tracer index mapping changed"
            )
        if (
            not np.array_equal(
                self.lifecycle_state.tracer_lifecycle,
                np.full(108, ACTIVE, dtype=np.uint8),
            )
            or self.lifecycle_state.removal_records
        ):
            raise KnownSystemInputContractError(
                "materialized genesis must contain exactly 108 active lanes"
            )
        observed_digest = restricted_wh_lifecycle_state_sha256(
            self.lifecycle_state, backend
        )
        if observed_digest != self.genesis_state_sha256:
            raise KnownSystemInputContractError(
                "materialized lifecycle genesis digest changed"
            )
        object.__setattr__(
            self,
            "content_sha256",
            _canonical_sha256(
                b"jx.long-horizon.known-system-observed-fixed-run-"
                b"materialization.v1",
                {
                    "schema": MATERIALIZED_SCHEMA,
                    "contract_content_sha256": self.contract.content_sha256,
                    "genesis_state_sha256": self.genesis_state_sha256,
                    "backend_device": backend.device,
                    "determinism_scope": backend.determinism_scope,
                },
            ),
        )

    @property
    def config(self) -> FixedLongHorizonRunConfig:
        return self.contract.config

    @property
    def source(self) -> FixedLongHorizonSourceSegment | None:
        return self.contract.source

    @property
    def roster(self) -> KnownSystemTracerRoster:
        return self.contract.roster


def _adapter_receipt() -> KnownSystemVocabularyEpochReceipt:
    return KnownSystemVocabularyEpochReceipt(
        known_system_model_sha256=KNOWN_SOLAR_SYSTEM_V2.content_sha256,
        genesis_model_sha256=DE440_SUN_GENESIS_MODEL.content_sha256,
        epoch_jd_tdb=KNOWN_SOLAR_SYSTEM_V2.epoch_jd_tdb,
        epoch_scale=KNOWN_SOLAR_SYSTEM_V2.epoch_scale,
        engine_epoch_origin_years=ENGINE_EPOCH_ORIGIN_YEARS,
        physical_frame=KNOWN_SOLAR_SYSTEM_V2.output_frame,
        physical_coordinate_system=(
            KNOWN_SOLAR_SYSTEM_V2.output_coordinate_system
        ),
        physical_time_unit=KNOWN_SOLAR_SYSTEM_V2.time_unit,
        engine_frame=KNOWN_SOLAR_SYSTEM_V2.output_frame,
        engine_axes=ENGINE_AXES,
        engine_coordinate_system=COORDINATE_SYSTEM,
        engine_length_unit=ENGINE_LENGTH_UNIT,
        engine_time_unit=ENGINE_TIME_UNIT,
        engine_unit_system_id=ENGINE_UNIT_SYSTEM_ID,
        physical_primary_id=KNOWN_SOLAR_SYSTEM_V2.primary_id,
        physical_primary_body_id=KNOWN_SYSTEM_BODY_IDS[0],
        engine_primary_source_id=KNOWN_SYSTEM_BODY_NAMES[0],
        epoch_mapping_policy=EPOCH_MAPPING_POLICY,
        vocabulary_adapter_policy=VOCABULARY_ADAPTER_POLICY,
        _seal=_ADAPTER_SEAL,
    )


def _tracer_roster(
    inventory: ObservedInitialConditionInventory,
) -> KnownSystemTracerRoster:
    return KnownSystemTracerRoster(
        inventory_content_sha256=inventory.content_sha256,
        object_ids=tuple(state.object_id for state in inventory.states),
        state_content_sha256=tuple(
            state.content_sha256 for state in inventory.states
        ),
        tracer_indices=tuple(range(108)),
        assignment_policy=TRACER_INDEX_POLICY,
        _seal=_ROSTER_SEAL,
    )


def prepare_known_system_observed_fixed_run(
    backend: BackendSpec = BackendSpec("cupy", "cuda:0", THREADS_PER_BLOCK),
    *,
    profile_id: KnownSystemRunProfile = KnownSystemRunProfile.BOOTSTRAP_1Y,
) -> KnownSystemObservedFixedRunContract:
    """Prepare one exact closed-profile M2/108 fixed-run host contract."""

    device_index = (
        backend.device.removeprefix("cuda:")
        if type(backend) is BackendSpec
        else ""
    )
    if (
        type(backend) is not BackendSpec
        or backend.backend_id != "cupy"
        or backend.tile_size != THREADS_PER_BLOCK
        or not backend.device.startswith("cuda:")
        or not device_index.isdigit()
        or (len(device_index) > 1 and device_index.startswith("0"))
    ):
        raise KnownSystemInputContractError(
            "backend must be an exact CuPy BackendSpec with a canonical "
            "cuda:<index> device and tile_size=128"
        )
    backend.__post_init__()
    if type(profile_id) is not KnownSystemRunProfile:
        raise KnownSystemInputContractError(
            "profile_id must be an exact KnownSystemRunProfile member"
        )
    model = KNOWN_SOLAR_SYSTEM_V2
    inventory = build_predeclared_initial_condition_inventory(
        model=DE440_SUN_GENESIS_MODEL
    )
    if DE440_SUN_GENESIS_MODEL.content_sha256 != (
        EXPECTED_DE440_SUN_GENESIS_MODEL_SHA256
    ) or inventory.content_sha256 != EXPECTED_DE440_SUN_INVENTORY_SHA256:
        raise KnownSystemInputContractError(
            "reviewed DE440-Sun genesis identities changed"
        )
    adapter = _adapter_receipt()
    roster = _tracer_roster(inventory)
    profile = _profile_receipt(profile_id, adapter)
    schedule = profile.source_schedule_sha256
    source: FixedLongHorizonSourceSegment | None = None
    if profile_id is KnownSystemRunProfile.BOOTSTRAP_1Y:
        source_positions = np.ascontiguousarray(
            model.positions_primary_relative_ecliptic_AU, dtype=np.float64
        )
        source_velocities = np.ascontiguousarray(
            model.velocities_primary_relative_ecliptic_AU_per_julian_year,
            dtype=np.float64,
        )
        source_gm = np.ascontiguousarray(
            model.gm_AU3_per_julian_year2, dtype=np.float64
        )
        route_radii = np.full(
            model.source_count,
            ALGORITHMIC_ROUTE_RADIUS_AU,
            dtype=np.float64,
        )
        uniform_bound = _uniform_acceleration_bound(
            source_gm, ALGORITHMIC_ROUTE_RADIUS_AU
        )
        acceleration_bounds = np.full(
            model.source_count, uniform_bound, dtype=np.float64
        )
        source = FixedLongHorizonSourceSegment(
            tile_id=SOURCE_TILE_ID,
            epoch_origin_years=adapter.engine_epoch_origin_years,
            global_source_step=0,
            source_step_years=profile.source_step_years,
            time_scale=adapter.epoch_scale,
            frame=adapter.engine_frame,
            axes=adapter.engine_axes,
            coordinate_system=adapter.engine_coordinate_system,
            length_unit=adapter.engine_length_unit,
            time_unit=adapter.engine_time_unit,
            unit_system_id=adapter.engine_unit_system_id,
            source_ids=model.body_names,
            primary_source_id=adapter.engine_primary_source_id,
            source_schedule_sha256=schedule,
            initial_source_relative_positions_AU=source_positions,
            initial_source_relative_velocities_AU_per_year=source_velocities,
            source_gm_AU3_per_year2=source_gm,
            source_encounter_radii_AU=route_radii,
            certified_relative_acceleration_bounds_AU_per_year2=(
                acceleration_bounds
            ),
            source_provenance=Provenance(
                source_id=BOOTSTRAP_SOURCE_PROVENANCE_ID,
                citation=BOOTSTRAP_SOURCE_PROVENANCE_CITATION,
                version=BOOTSTRAP_SOURCE_PROVENANCE_VERSION,
                sha256=model.content_sha256,
            ),
        )
    route_spec = RestrictedWHRouteSpec(
        maximum_primary_dynamical_phase=(
            profile.maximum_primary_dynamical_phase
        ),
        primary_outer_boundary_AU=profile.primary_outer_boundary_AU,
        terminal_source_radii_AU=PHYSICAL_TERMINAL_RADII_AU,
    )
    config = FixedLongHorizonRunConfig(
        run_id=profile.run_id,
        source_spec=WisdomHolmanSourceTileSpec(
            profile.source_step_years,
            profile.segment_source_step_count,
            minimum_encounter_pair_separation_AU=(
                SOURCE_PAIR_SEPARATION_FLOOR_AU
            ),
            minimum_jacobi_periapse_AU=(
                SOURCE_JACOBI_PERIAPSE_FLOOR_AU
            ),
            maximum_initial_barycenter_position_norm_AU=(
                MAXIMUM_INITIAL_BARYCENTER_POSITION_NORM_AU
            ),
            maximum_initial_barycenter_velocity_norm_AU_per_year=(
                MAXIMUM_INITIAL_BARYCENTER_VELOCITY_NORM_AU_PER_YEAR
            ),
        ),
        tracer_spec=FixedStepCUDARestrictedWHSpec(
            profile.segment_map_step_count,
            profile.source_steps_per_map_step,
            backend,
            maximum_primary_dynamical_phase=(
                profile.maximum_primary_dynamical_phase
            ),
            threads_per_block=THREADS_PER_BLOCK,
            primary_outer_boundary_AU=profile.primary_outer_boundary_AU,
        ),
        route_spec=route_spec,
        route_worker_count=profile.route_worker_count,
        multirate_enabled=profile.multirate_enabled,
        owned_episode_enabled=profile.owned_episode_enabled,
    )
    tracer_positions = np.ascontiguousarray(
        [state.position_AU for state in inventory.states], dtype=np.float64
    )
    tracer_velocities = np.ascontiguousarray(
        [state.velocity_AU_per_year for state in inventory.states],
        dtype=np.float64,
    )
    return KnownSystemObservedFixedRunContract(
        known_system_model=model,
        inventory=inventory,
        roster=roster,
        adapter=adapter,
        profile=profile,
        tracer_positions_AU=tracer_positions,
        tracer_velocities_AU_per_year=tracer_velocities,
        source_schedule_sha256=schedule,
        source=source,
        config=config,
        _seal=_CONTRACT_SEAL,
    )


def build_known_system_observed_fixed_run_inputs(
    backend: BackendSpec = BackendSpec("cupy", "cuda:0", THREADS_PER_BLOCK),
    *,
    profile_id: KnownSystemRunProfile = KnownSystemRunProfile.BOOTSTRAP_1Y,
) -> KnownSystemObservedFixedRunInputs:
    """Atomically prepare and materialize the exact CUDA lifecycle genesis."""

    contract = prepare_known_system_observed_fixed_run(
        backend, profile_id=profile_id
    )
    resolved = resolve_backend(backend)
    cp = resolved.xp
    with resolved.activate():
        state = CUDARestrictedWHState(
            snapshot_id=contract.profile.snapshot_id,
            epoch_origin_years=contract.adapter.engine_epoch_origin_years,
            global_source_step=0,
            source_step_years=contract.profile.source_step_years,
            time_scale=contract.adapter.epoch_scale,
            frame=contract.adapter.engine_frame,
            axes=contract.adapter.engine_axes,
            coordinate_system=contract.adapter.engine_coordinate_system,
            length_unit=contract.adapter.engine_length_unit,
            time_unit=contract.adapter.engine_time_unit,
            unit_system_id=contract.adapter.engine_unit_system_id,
            source_ids=contract.known_system_model.body_names,
            primary_source_id=contract.adapter.engine_primary_source_id,
            source_schedule_sha256=contract.source_schedule_sha256,
            tracer_relative_positions_AU=cp.asarray(
                contract.tracer_positions_AU, dtype=cp.float64, order="C"
            ),
            tracer_relative_velocities_AU_per_year=cp.asarray(
                contract.tracer_velocities_AU_per_year,
                dtype=cp.float64,
                order="C",
            ),
            tracer_indices=cp.arange(108, dtype=cp.uint64),
            provenance=Provenance(
                source_id=(
                    "jx.long-horizon.m2-observed-fixed-genesis."
                    + contract.profile.profile_id.value
                ),
                citation=(
                    "Content-addressed M2 known-system plus 108-state "
                    "DE440-Sun observed benchmark genesis"
                ),
                version="v1",
                sha256=contract.content_sha256,
            ),
        )
        lifecycle = initialize_restricted_wh_lifecycle(state)
        genesis_digest = restricted_wh_lifecycle_state_sha256(
            lifecycle, backend
        )
    return KnownSystemObservedFixedRunInputs(
        contract=contract,
        lifecycle_state=lifecycle,
        genesis_state_sha256=genesis_digest,
    )


__all__ = [
    "ADAPTER_SCHEMA",
    "ALGORITHMIC_ROUTE_RADII_AU",
    "ALGORITHMIC_ROUTE_RADIUS_AU",
    "CLAIMS",
    "CONTRACT_SCHEMA",
    "ENGINE_EPOCH_ORIGIN_YEARS",
    "HORIZON_YEARS",
    "KnownSystemInputContractError",
    "KnownSystemObservedFixedRunContract",
    "KnownSystemObservedFixedRunInputs",
    "KnownSystemRunProfile",
    "KnownSystemRunProfileReceipt",
    "KnownSystemTracerRoster",
    "KnownSystemVocabularyEpochReceipt",
    "MATERIALIZED_SCHEMA",
    "M2_FINE_CANDIDATE_PROFILE_ID",
    "M2_FINE_CANDIDATE_RUN_ID",
    "M2_FINE_CANDIDATE_SEGMENT_MAP_STEP_COUNT",
    "M2_FINE_CANDIDATE_SOURCE_STEPS_PER_MAP_STEP",
    "M2_FINE_CANDIDATE_STEP_YEARS",
    "M2_FINE_HORIZON_YEARS",
    "M2_FINE_MAXIMUM_PRIMARY_DYNAMICAL_PHASE",
    "M2_FINE_PRIMARY_OUTER_BOUNDARY_AU",
    "M2_FINE_REFINEMENT_PROFILE_ID",
    "M2_FINE_REFINEMENT_RUN_ID",
    "M2_FINE_REFINEMENT_SEGMENT_MAP_STEP_COUNT",
    "M2_FINE_REFINEMENT_SOURCE_STEPS_PER_MAP_STEP",
    "M2_FINE_REFINEMENT_STEP_YEARS",
    "M2_FINE_SEGMENT_SOURCE_STEP_COUNT",
    "M2_FINE_SEGMENT_YEARS",
    "PHYSICAL_TERMINAL_RADII_AU",
    "PROFILE_ID",
    "PROFILE_SCHEMA",
    "ROSTER_SCHEMA",
    "ROUTE_WORKER_COUNT",
    "RUN_ID",
    "SOURCE_STEP_COUNT",
    "SOURCE_STEP_YEARS",
    "TERMINAL_RADIUS_POLICY",
    "TRACER_MAP_STEP_COUNT",
    "TRACER_SOURCE_STEPS_PER_MAP_STEP",
    "TRACER_STEP_YEARS",
    "build_known_system_observed_fixed_run_inputs",
    "prepare_known_system_observed_fixed_run",
]
