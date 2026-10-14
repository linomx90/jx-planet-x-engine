"""Deterministic, non-pickle artifacts for one unified NumPy RKF78 run.

Run-artifact v1 is deliberately narrower than the unified simulation API.  It
stores the checkpoint trajectory from one successful NumPy/CPU RKF78 run and
the identities needed to bind that trajectory to a caller-owned job manifest.
It does not reconstruct an executable result, grant scientific authority, or
serialize Python objects.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass
import hashlib
import io
import json
import math
import tokenize
from typing import Any
import zipfile

from ._zip_preflight import preflight_exact_zip_roster
from .simulation import (
    SCIENTIFIC_CLAIM_STATE,
    JXSimulationContractError,
    JXSimulationRun,
)
from .trajectory import (
    TrajectoryResult,
    validate_trajectory_result_integrity,
)
from .trajectory_contracts import ADAPTIVE_RKF78_METHOD_ID, RKF78_STAGE_COUNT


ENGINE_RUN_ARTIFACT_SCHEMA = "jxplanetx.engine-run-artifact.v1"
ENGINE_RUN_ARTIFACT_SCOPE = "NUMPY_CPU_RKF78_CHECKPOINT_TRAJECTORY"
ENGINE_RUN_ARTIFACT_HASH_ALGORITHM = "SHA256_ARCHIVE_BYTES"
ENGINE_RUN_ARTIFACT_MEMBER_HASH_ALGORITHM = "SHA256_MEMBER_BYTES"
ENGINE_RUN_ARTIFACT_MAX_BYTES = 256 * 1024 * 1024
ENGINE_RUN_ARTIFACT_MAX_MANIFEST_BYTES = 1024 * 1024
ENGINE_RUN_ARTIFACT_ARRAY_DTYPE = "float64-le"
_ARRAY_NAMES = (
    "checkpoint_epochs",
    "checkpoint_positions",
    "checkpoint_velocities",
)
_MEMBER_NAMES = tuple(f"arrays/{name}.npy" for name in _ARRAY_NAMES) + (
    "manifest.json",
)
_ARTIFACT_LOAD_TOKEN = object()


class EngineRunArtifactError(JXSimulationContractError):
    """A run or artifact violates the closed run-artifact v1 contract."""


@dataclass(frozen=True, slots=True, eq=False)
class JXEngineRunArtifact:
    """Verified checkpoint content and identities loaded from one artifact."""

    artifact_sha256: str
    manifest_sha256: str
    run_id: str
    simulation_id: str
    run_index: int
    integrator_id: str
    force_model_ids: tuple[str, ...]
    snapshot_id: str
    plan_id: str
    body_ids: tuple[str, ...]
    checkpoint_epochs: tuple[float, ...]
    checkpoint_positions: Any
    checkpoint_velocities: Any
    attempted_steps: int
    accepted_steps: int
    rejected_steps: int
    force_evaluations: int
    native_result_digest: str
    native_result_digest_scope: str
    accepted_step_ledger_content_sha256: str
    parent_run_index: int | None
    parent_native_result_digest: str | None
    continuation_state_sha256: str | None
    time_scale: str
    frame: str
    origin: str
    axes: str
    length_unit: str
    time_unit: str
    mass_unit: str
    unit_system_id: str
    backend_id: str = "numpy"
    device: str = "cpu"
    schema: str = ENGINE_RUN_ARTIFACT_SCHEMA
    scope: str = ENGINE_RUN_ARTIFACT_SCOPE
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    evidence_class: str = "MODEL_OUTPUT"
    accuracy_claimed: bool = False
    improvement_claimed: bool = False
    physics_claimed: bool = False
    execution_authenticated: bool = False
    production_authorized: bool = False
    registry_authorized: bool = False
    qualification_authorized: bool = False
    general_superiority_claimed: bool = False
    _load_token: InitVar[object] = None

    def __post_init__(self, _load_token: object) -> None:
        if _load_token is not _ARTIFACT_LOAD_TOKEN:
            raise EngineRunArtifactError(
                "JXEngineRunArtifact must be constructed by "
                "load_engine_run_artifact"
            )
        _digest(self.artifact_sha256, "artifact_sha256")
        _digest(self.manifest_sha256, "manifest_sha256")
        _run_id(self.run_id)
        _text(self.simulation_id, "simulation_id")
        _integer(self.run_index, "run_index")
        if self.integrator_id != ADAPTIVE_RKF78_METHOD_ID:
            raise EngineRunArtifactError("integrator_id is not the fixed RKF78 route")
        _string_tuple(self.force_model_ids, "force_model_ids")
        _text(self.snapshot_id, "snapshot_id")
        _text(self.plan_id, "plan_id")
        body_ids = _string_tuple(self.body_ids, "body_ids")
        if len(set(body_ids)) != len(body_ids):
            raise EngineRunArtifactError("body_ids must be unique")
        if (
            type(self.checkpoint_epochs) is not tuple
            or len(self.checkpoint_epochs) < 2
            or any(
                type(value) is not float or not math.isfinite(value)
                for value in self.checkpoint_epochs
            )
        ):
            raise EngineRunArtifactError(
                "checkpoint_epochs must be a finite float tuple with at least two entries"
            )
        deltas = tuple(
            right - left
            for left, right in zip(
                self.checkpoint_epochs,
                self.checkpoint_epochs[1:],
            )
        )
        if not (all(value > 0.0 for value in deltas) or all(value < 0.0 for value in deltas)):
            raise EngineRunArtifactError("checkpoint_epochs must be strictly monotone")
        _validate_owned_loaded_array(
            self.checkpoint_positions,
            "checkpoint_positions",
            (len(self.checkpoint_epochs), len(body_ids), 3),
        )
        _validate_owned_loaded_array(
            self.checkpoint_velocities,
            "checkpoint_velocities",
            (len(self.checkpoint_epochs), len(body_ids), 3),
        )
        for label in (
            "attempted_steps",
            "accepted_steps",
            "rejected_steps",
            "force_evaluations",
        ):
            _integer(getattr(self, label), label)
        if self.attempted_steps != self.accepted_steps + self.rejected_steps:
            raise EngineRunArtifactError("attempted-step accounting is inconsistent")
        if self.force_evaluations != self.attempted_steps * RKF78_STAGE_COUNT:
            raise EngineRunArtifactError("RKF78 force-evaluation accounting is inconsistent")
        if self.accepted_steps < len(self.checkpoint_epochs) - 1:
            raise EngineRunArtifactError(
                "RKF78 accepted-step count cannot cover every checkpoint interval"
            )
        _digest(self.native_result_digest, "native_result_digest")
        if self.native_result_digest_scope != "FULL_RETAINED_RESULT_CONTENT":
            raise EngineRunArtifactError(
                "native_result_digest_scope is not the fixed NumPy RKF78 scope"
            )
        _digest(
            self.accepted_step_ledger_content_sha256,
            "accepted_step_ledger_content_sha256",
        )
        _validate_lineage(
            run_index=self.run_index,
            parent_run_index=self.parent_run_index,
            parent_native_result_digest=self.parent_native_result_digest,
            continuation_state_sha256=self.continuation_state_sha256,
        )
        for label in (
            "time_scale",
            "frame",
            "origin",
            "axes",
            "length_unit",
            "time_unit",
            "mass_unit",
            "unit_system_id",
        ):
            _text(getattr(self, label), label)
        if (
            self.backend_id != "numpy"
            or self.device != "cpu"
            or self.schema != ENGINE_RUN_ARTIFACT_SCHEMA
            or self.scope != ENGINE_RUN_ARTIFACT_SCOPE
            or self.scientific_claim_state != SCIENTIFIC_CLAIM_STATE
            or self.evidence_class != "MODEL_OUTPUT"
            or type(self.accuracy_claimed) is not bool
            or self.accuracy_claimed
            or type(self.improvement_claimed) is not bool
            or self.improvement_claimed
            or type(self.physics_claimed) is not bool
            or self.physics_claimed
            or type(self.execution_authenticated) is not bool
            or self.execution_authenticated
            or type(self.production_authorized) is not bool
            or self.production_authorized
            or type(self.registry_authorized) is not bool
            or self.registry_authorized
            or type(self.qualification_authorized) is not bool
            or self.qualification_authorized
            or type(self.general_superiority_claimed) is not bool
            or self.general_superiority_claimed
        ):
            raise EngineRunArtifactError(
                "run artifacts retain the fixed NumPy RKF78 screening-only contract"
            )

    @property
    def checkpoint_count(self) -> int:
        return len(self.checkpoint_epochs)

    @property
    def body_count(self) -> int:
        return len(self.body_ids)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(document: object) -> bytes:
    try:
        rendered = json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (RecursionError, TypeError, ValueError) as exc:
        raise EngineRunArtifactError("artifact manifest is not canonical JSON") from exc
    return (rendered + "\n").encode("ascii")


def _strict_json(raw: bytes) -> dict[str, Any]:
    duplicate: str | None = None

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        nonlocal duplicate
        result: dict[str, object] = {}
        for key, value in items:
            if key in result and duplicate is None:
                duplicate = key
            result[key] = value
        return result

    def reject_float(token: str) -> object:
        raise EngineRunArtifactError(
            f"JSON floating token {token!r} is forbidden; trajectory values use NPY"
        )

    def reject_constant(token: str) -> object:
        raise EngineRunArtifactError(f"nonfinite JSON constant {token!r} is forbidden")

    try:
        document = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=pairs,
            parse_float=reject_float,
            parse_constant=reject_constant,
        )
    except EngineRunArtifactError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise EngineRunArtifactError("artifact manifest must be ASCII JSON") from exc
    if duplicate is not None:
        raise EngineRunArtifactError(f"artifact manifest repeats key {duplicate!r}")
    if type(document) is not dict or _canonical(document) != raw:
        raise EngineRunArtifactError("artifact manifest is not canonical")
    return document


def _expect_keys(value: object, keys: tuple[str, ...], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(keys) or len(value) != len(keys):
        raise EngineRunArtifactError(f"{label} must contain exactly {keys!r}")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise EngineRunArtifactError(f"{label} must be nonempty trimmed text")
    return value


def _run_id(value: object) -> str:
    checked = _text(value, "run_id")
    if len(checked) > 256 or any(ord(character) < 0x20 for character in checked):
        raise EngineRunArtifactError(
            "run_id must contain at most 256 non-control characters"
        )
    return checked


def _digest(value: object, label: str) -> str:
    checked = _text(value, label)
    if len(checked) != 64 or any(
        character not in "0123456789abcdef" for character in checked
    ):
        raise EngineRunArtifactError(f"{label} must be lowercase SHA-256 hex")
    return checked


def _optional_digest(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _digest(value, label)


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise EngineRunArtifactError(f"{label} must be an integer >= {minimum}")
    return value


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if (
        type(value) is not tuple
        or not value
        or any(type(item) is not str or not item or item.strip() != item for item in value)
    ):
        raise EngineRunArtifactError(f"{label} must be a nonempty tuple of text")
    return value


def _string_list(value: object, label: str) -> tuple[str, ...]:
    if (
        type(value) is not list
        or not value
        or any(type(item) is not str or not item or item.strip() != item for item in value)
    ):
        raise EngineRunArtifactError(f"{label} must be a nonempty list of text")
    return tuple(value)


def _validate_lineage(
    *,
    run_index: int,
    parent_run_index: object,
    parent_native_result_digest: object,
    continuation_state_sha256: object,
) -> tuple[int | None, str | None, str | None]:
    if parent_run_index is None:
        if parent_native_result_digest is not None or continuation_state_sha256 is not None:
            raise EngineRunArtifactError(
                "an initial run cannot retain continuation identities"
            )
        return None, None, None
    checked_index = _integer(parent_run_index, "parent_run_index")
    if checked_index >= run_index:
        raise EngineRunArtifactError("parent_run_index must identify an earlier run")
    return (
        checked_index,
        _digest(parent_native_result_digest, "parent_native_result_digest"),
        _digest(continuation_state_sha256, "continuation_state_sha256"),
    )


def _validate_owned_loaded_array(value: object, label: str, shape: tuple[int, ...]) -> None:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - engine extra owns NumPy
        raise EngineRunArtifactError("run-artifact v1 requires NumPy") from exc
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype("<f8")
        or value.shape != shape
        or not bool(value.flags.c_contiguous)
        or not bool(value.flags.owndata)
        or bool(value.flags.writeable)
        or not bool(np.all(np.isfinite(value)))
    ):
        raise EngineRunArtifactError(
            f"{label} must be an owned, read-only, finite little-endian float64 "
            f"array with shape {shape}"
        )


def _array_bytes(value: object, label: str) -> bytes:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - engine extra owns NumPy
        raise EngineRunArtifactError("run-artifact v1 requires NumPy") from exc
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype("<f8")
        or not value.flags.c_contiguous
    ):
        raise EngineRunArtifactError(
            f"{label} must be C-contiguous little-endian float64"
        )
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        value,
        version=(2, 0),
        allow_pickle=False,
    )
    return stream.getvalue()


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in _MEMBER_NAMES:
            archive.writestr(_zip_info(name), members[name])
    return stream.getvalue()


def _same_float64_array(left: object, right: object) -> bool:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - engine extra owns NumPy
        raise EngineRunArtifactError("run-artifact v1 requires NumPy") from exc
    return bool(
        type(left) is np.ndarray
        and type(right) is np.ndarray
        and left.dtype == np.dtype(np.float64)
        and right.dtype == np.dtype(np.float64)
        and left.shape == right.shape
        and left.flags.c_contiguous
        and right.flags.c_contiguous
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _validate_source_run(run: object) -> tuple[JXSimulationRun, TrajectoryResult]:
    if type(run) is not JXSimulationRun:
        raise EngineRunArtifactError("run must be an exact JXSimulationRun")
    native = run.native_result
    if type(native) is not TrajectoryResult:
        raise EngineRunArtifactError(
            "run-artifact v1 accepts only an exact RKF78 TrajectoryResult"
        )
    try:
        validate_trajectory_result_integrity(native)
    except Exception as exc:
        raise EngineRunArtifactError("native RKF78 result integrity failed") from exc
    if (
        run.integrator_id != ADAPTIVE_RKF78_METHOD_ID
        or native.method_id != ADAPTIVE_RKF78_METHOD_ID
        or run.backend_id != "numpy"
        or run.device != "cpu"
        or native.backend_id != "numpy"
        or native.device != "cpu"
    ):
        raise EngineRunArtifactError(
            "run-artifact v1 accepts only the NumPy/CPU RKF78 route"
        )
    if (
        run.scientific_claim_state != SCIENTIFIC_CLAIM_STATE
        or type(run.production_authorized) is not bool
        or run.production_authorized
        or type(run.general_superiority_claimed) is not bool
        or run.general_superiority_claimed
        or native.evidence_class != "MODEL_OUTPUT"
        or native.registry_authorized
        or native.qualification_authorized
    ):
        raise EngineRunArtifactError("source run exceeds the screening-only claim ceiling")
    if (
        run.native_result_digest != native.result_content_sha256
        or run.native_result_digest is None
        or run.native_result_digest_scope != "FULL_RETAINED_RESULT_CONTENT"
    ):
        raise EngineRunArtifactError("source run native-result identity is inconsistent")
    _digest(run.native_result_digest, "native_result_digest")
    _text(run.simulation_id, "simulation_id")
    _integer(run.run_index, "run_index")
    _string_tuple(run.force_model_ids, "force_model_ids")
    for label in (
        "attempted_steps",
        "accepted_steps",
        "rejected_steps",
        "force_evaluations",
    ):
        _integer(getattr(run, label), label)
    if (
        type(run.checkpoint_epochs) is not tuple
        or len(run.checkpoint_epochs) < 2
        or any(
            type(value) is not float or not math.isfinite(value)
            for value in run.checkpoint_epochs
        )
    ):
        raise EngineRunArtifactError("source run checkpoint epochs are invalid")
    if run.attempted_steps != run.accepted_steps + run.rejected_steps:
        raise EngineRunArtifactError("source run attempted-step accounting is inconsistent")
    if run.force_evaluations != run.attempted_steps * RKF78_STAGE_COUNT:
        raise EngineRunArtifactError(
            "source run RKF78 force-evaluation accounting is inconsistent"
        )
    if run.accepted_steps < len(run.checkpoint_epochs) - 1:
        raise EngineRunArtifactError(
            "source run accepted-step count cannot cover every checkpoint interval"
        )
    if (
        run.force_model_ids != native.force_model_ids
        or run.checkpoint_epochs != native.checkpoint_epochs
        or run.attempted_steps != native.attempted_steps
        or run.accepted_steps != native.accepted_steps
        or run.rejected_steps != native.rejected_steps
        or run.force_evaluations != native.force_evaluations
    ):
        raise EngineRunArtifactError("source run envelope differs from its native result")
    snapshot = native.initial_snapshot
    _text(native.snapshot_id, "snapshot_id")
    _text(native.plan_id, "plan_id")
    body_ids = _string_tuple(snapshot.body_ids, "body_ids")
    if len(set(body_ids)) != len(body_ids):
        raise EngineRunArtifactError("body_ids must be unique")
    for label in (
        "time_scale",
        "frame",
        "origin",
        "axes",
        "length_unit",
        "time_unit",
        "mass_unit",
        "unit_system_id",
    ):
        _text(getattr(snapshot, label), label)
    if (
        type(run.checkpoint_positions) is not tuple
        or type(run.checkpoint_velocities) is not tuple
        or len(run.checkpoint_positions) != len(native.checkpoints)
        or len(run.checkpoint_velocities) != len(native.checkpoints)
    ):
        raise EngineRunArtifactError("source run checkpoint envelope is invalid")
    for index, checkpoint in enumerate(native.checkpoints):
        if not _same_float64_array(
            run.checkpoint_positions[index],
            checkpoint.positions,
        ) or not _same_float64_array(
            run.checkpoint_velocities[index],
            checkpoint.velocities,
        ):
            raise EngineRunArtifactError(
                "source run checkpoint arrays differ from its native result"
            )
    _validate_lineage(
        run_index=run.run_index,
        parent_run_index=run.parent_run_index,
        parent_native_result_digest=run.parent_native_result_digest,
        continuation_state_sha256=run.continuation_state_sha256,
    )
    return run, native


def dump_engine_run_artifact(
    run: JXSimulationRun,
    *,
    manifest_sha256: str,
    run_id: str,
) -> bytes:
    """Return canonical artifact bytes for one successful NumPy/CPU RKF78 run."""

    checked_manifest_sha256 = _digest(manifest_sha256, "manifest_sha256")
    checked_run_id = _run_id(run_id)
    checked_run, native = _validate_source_run(run)
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - engine extra owns NumPy
        raise EngineRunArtifactError("run-artifact v1 requires NumPy") from exc
    arrays = {
        "checkpoint_epochs": np.ascontiguousarray(
            np.asarray(checked_run.checkpoint_epochs, dtype="<f8")
        ),
        "checkpoint_positions": np.ascontiguousarray(
            np.stack(checked_run.checkpoint_positions, axis=0),
            dtype="<f8",
        ),
        "checkpoint_velocities": np.ascontiguousarray(
            np.stack(checked_run.checkpoint_velocities, axis=0),
            dtype="<f8",
        ),
    }
    for name, value in arrays.items():
        if not bool(np.all(np.isfinite(value))):
            raise EngineRunArtifactError(f"{name} contains a nonfinite value")
    raw_arrays = {
        name: _array_bytes(value, name) for name, value in arrays.items()
    }
    array_manifest = {
        name: {
            "dtype": ENGINE_RUN_ARTIFACT_ARRAY_DTYPE,
            "member": f"arrays/{name}.npy",
            "sha256": _sha256(raw_arrays[name]),
            "shape": list(arrays[name].shape),
            "size_bytes": len(raw_arrays[name]),
        }
        for name in _ARRAY_NAMES
    }
    snapshot = native.initial_snapshot
    manifest = {
        "arrays": array_manifest,
        "binding": {
            "manifest_sha256": checked_manifest_sha256,
            "run_id": checked_run_id,
        },
        "claim": {
            "accuracy_claimed": False,
            "evidence_class": "MODEL_OUTPUT",
            "execution_authenticated": False,
            "general_superiority_claimed": False,
            "improvement_claimed": False,
            "physics_claimed": False,
            "production_authorized": False,
            "qualification_authorized": False,
            "registry_authorized": False,
            "scientific_claim_state": SCIENTIFIC_CLAIM_STATE,
        },
        "coordinates": {
            "axes": snapshot.axes,
            "frame": snapshot.frame,
            "length_unit": snapshot.length_unit,
            "mass_unit": snapshot.mass_unit,
            "origin": snapshot.origin,
            "time_scale": snapshot.time_scale,
            "time_unit": snapshot.time_unit,
            "unit_system_id": snapshot.unit_system_id,
        },
        "hash_algorithm": ENGINE_RUN_ARTIFACT_HASH_ALGORITHM,
        "member_hash_algorithm": ENGINE_RUN_ARTIFACT_MEMBER_HASH_ALGORITHM,
        "run": {
            "accepted_step_ledger_content_sha256": (
                native.accepted_step_ledger_content_sha256
            ),
            "accepted_steps": checked_run.accepted_steps,
            "attempted_steps": checked_run.attempted_steps,
            "backend_id": "numpy",
            "body_count": len(snapshot.body_ids),
            "body_ids": list(snapshot.body_ids),
            "checkpoint_count": len(checked_run.checkpoint_epochs),
            "continuation_state_sha256": checked_run.continuation_state_sha256,
            "device": "cpu",
            "force_evaluations": checked_run.force_evaluations,
            "force_model_ids": list(checked_run.force_model_ids),
            "integrator_id": checked_run.integrator_id,
            "native_result_digest": checked_run.native_result_digest,
            "native_result_digest_scope": checked_run.native_result_digest_scope,
            "parent_native_result_digest": checked_run.parent_native_result_digest,
            "parent_run_index": checked_run.parent_run_index,
            "plan_id": native.plan_id,
            "rejected_steps": checked_run.rejected_steps,
            "run_index": checked_run.run_index,
            "simulation_id": checked_run.simulation_id,
            "snapshot_id": native.snapshot_id,
        },
        "schema": ENGINE_RUN_ARTIFACT_SCHEMA,
        "scope": ENGINE_RUN_ARTIFACT_SCOPE,
    }
    manifest_raw = _canonical(manifest)
    if len(manifest_raw) > ENGINE_RUN_ARTIFACT_MAX_MANIFEST_BYTES:
        raise EngineRunArtifactError("artifact manifest exceeds the v1 size ceiling")
    members = {
        **{f"arrays/{name}.npy": raw_arrays[name] for name in _ARRAY_NAMES},
        "manifest.json": manifest_raw,
    }
    result = _zip_bytes(members)
    if len(result) > ENGINE_RUN_ARTIFACT_MAX_BYTES:
        raise EngineRunArtifactError("artifact exceeds the v1 size ceiling")
    # Revalidation after copying catches ordinary post-run caller mutation.
    _validate_source_run(checked_run)
    return result


def engine_run_artifact_sha256(artifact_bytes: bytes) -> str:
    """Return the exact byte identity of one run artifact."""

    if type(artifact_bytes) is not bytes:
        raise EngineRunArtifactError("artifact_bytes must be exact bytes")
    return _sha256(artifact_bytes)


def _read_array(
    raw: bytes,
    record: object,
    name: str,
    expected_shape: tuple[int, ...],
) -> object:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - engine extra owns NumPy
        raise EngineRunArtifactError("run-artifact v1 requires NumPy") from exc
    document = _expect_keys(
        record,
        ("dtype", "member", "sha256", "shape", "size_bytes"),
        f"arrays.{name}",
    )
    expected_member = f"arrays/{name}.npy"
    shape = document["shape"]
    if (
        document["dtype"] != ENGINE_RUN_ARTIFACT_ARRAY_DTYPE
        or document["member"] != expected_member
        or type(shape) is not list
        or any(type(dimension) is not int for dimension in shape)
        or tuple(shape) != expected_shape
    ):
        raise EngineRunArtifactError(f"arrays.{name} metadata is not canonical")
    size = _integer(document["size_bytes"], f"arrays.{name}.size_bytes", minimum=1)
    if size > ENGINE_RUN_ARTIFACT_MAX_BYTES:
        raise EngineRunArtifactError(f"arrays.{name} exceeds the size ceiling")
    if len(raw) != size or _sha256(raw) != _digest(
        document["sha256"],
        f"arrays.{name}.sha256",
    ):
        raise EngineRunArtifactError(f"arrays.{name} byte identity mismatch")
    header_stream = io.BytesIO(raw)
    try:
        version = np.lib.format.read_magic(header_stream)
        if version != (2, 0):
            raise EngineRunArtifactError(
                f"arrays.{name} must use canonical NPY version 2.0"
            )
        header_shape, fortran_order, header_dtype = (
            np.lib.format.read_array_header_2_0(
                header_stream,
                max_header_size=10_000,
            )
        )
    except EngineRunArtifactError:
        raise
    except (
        EOFError,
        MemoryError,
        OSError,
        OverflowError,
        RecursionError,
        SyntaxError,
        TypeError,
        ValueError,
        tokenize.TokenError,
    ) as exc:
        raise EngineRunArtifactError(
            f"arrays.{name} has an invalid or oversized NPY header"
        ) from exc
    if (
        header_shape != expected_shape
        or fortran_order is not False
        or header_dtype != np.dtype("<f8")
    ):
        raise EngineRunArtifactError(
            f"arrays.{name} has the wrong header dtype, shape, or memory order"
        )
    expected_payload_size = math.prod(expected_shape) * np.dtype("<f8").itemsize
    if (
        expected_payload_size > ENGINE_RUN_ARTIFACT_MAX_BYTES
        or len(raw) - header_stream.tell() != expected_payload_size
    ):
        raise EngineRunArtifactError(
            f"arrays.{name} NPY payload length does not match its declared shape"
        )
    try:
        loaded = np.load(io.BytesIO(raw), allow_pickle=False)
    except (
        MemoryError,
        OSError,
        OverflowError,
        RecursionError,
        SyntaxError,
        ValueError,
        tokenize.TokenError,
    ) as exc:
        raise EngineRunArtifactError(f"arrays.{name} is not a safe NPY array") from exc
    if (
        type(loaded) is not np.ndarray
        or loaded.dtype != np.dtype("<f8")
        or loaded.shape != expected_shape
        or not loaded.flags.c_contiguous
        or not bool(np.all(np.isfinite(loaded)))
    ):
        raise EngineRunArtifactError(
            f"arrays.{name} has the wrong dtype, shape, order, or finiteness"
        )
    if _array_bytes(loaded, name) != raw:
        raise EngineRunArtifactError(f"arrays.{name} NPY encoding is not canonical")
    try:
        owned = np.array(loaded, dtype="<f8", order="C", copy=True)
    except (MemoryError, ValueError) as exc:
        raise EngineRunArtifactError(
            f"arrays.{name} could not be retained within the memory ceiling"
        ) from exc
    owned.setflags(write=False)
    return owned


def load_engine_run_artifact(
    artifact_bytes: bytes,
    *,
    expected_sha256: str,
    expected_manifest_sha256: str,
    expected_run_id: str,
) -> JXEngineRunArtifact:
    """Verify and load a run artifact against caller-owned identities."""

    if type(artifact_bytes) is not bytes or not artifact_bytes:
        raise EngineRunArtifactError("artifact_bytes must be nonempty exact bytes")
    if len(artifact_bytes) > ENGINE_RUN_ARTIFACT_MAX_BYTES:
        raise EngineRunArtifactError("artifact exceeds the v1 size ceiling")
    artifact_sha256 = engine_run_artifact_sha256(artifact_bytes)
    if artifact_sha256 != _digest(expected_sha256, "expected_sha256"):
        raise EngineRunArtifactError("artifact SHA-256 does not match expected_sha256")
    checked_manifest_sha256 = _digest(
        expected_manifest_sha256,
        "expected_manifest_sha256",
    )
    checked_run_id = _run_id(expected_run_id)
    try:
        preflight_exact_zip_roster(artifact_bytes, _MEMBER_NAMES)
        archive_context = zipfile.ZipFile(io.BytesIO(artifact_bytes), "r")
    except ValueError as exc:
        raise EngineRunArtifactError(
            "artifact member roster or central directory is not canonical"
        ) from exc
    except (RuntimeError, zipfile.BadZipFile) as exc:
        raise EngineRunArtifactError("artifact is not a valid ZIP container") from exc
    with archive_context as archive:
        infos = archive.infolist()
        names = tuple(info.filename for info in infos)
        if names != _MEMBER_NAMES or len(set(names)) != len(names):
            raise EngineRunArtifactError(
                "artifact member roster or order is not canonical"
            )
        for info in infos:
            mode = (info.external_attr >> 16) & 0o170000
            if (
                info.is_dir()
                or info.file_size > ENGINE_RUN_ARTIFACT_MAX_BYTES
                or info.compress_size != info.file_size
                or info.compress_type != zipfile.ZIP_STORED
                or mode not in (0, 0o100000)
            ):
                raise EngineRunArtifactError("artifact contains an invalid member")
            if (
                info.filename == "manifest.json"
                and info.file_size > ENGINE_RUN_ARTIFACT_MAX_MANIFEST_BYTES
            ):
                raise EngineRunArtifactError(
                    "artifact manifest exceeds the v1 size ceiling"
                )
        if sum(info.file_size for info in infos) > ENGINE_RUN_ARTIFACT_MAX_BYTES:
            raise EngineRunArtifactError("artifact members exceed the total size ceiling")
        try:
            members = {name: archive.read(name) for name in _MEMBER_NAMES}
        except (KeyError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
            raise EngineRunArtifactError("artifact members could not be read safely") from exc
    if len(members["manifest.json"]) > ENGINE_RUN_ARTIFACT_MAX_MANIFEST_BYTES:
        raise EngineRunArtifactError("artifact manifest exceeds the v1 size ceiling")
    if _zip_bytes(members) != artifact_bytes:
        raise EngineRunArtifactError("artifact ZIP encoding is not canonical")
    manifest = _strict_json(members["manifest.json"])
    top = _expect_keys(
        manifest,
        (
            "arrays",
            "binding",
            "claim",
            "coordinates",
            "hash_algorithm",
            "member_hash_algorithm",
            "run",
            "schema",
            "scope",
        ),
        "manifest",
    )
    if (
        top["schema"] != ENGINE_RUN_ARTIFACT_SCHEMA
        or top["scope"] != ENGINE_RUN_ARTIFACT_SCOPE
        or top["hash_algorithm"] != ENGINE_RUN_ARTIFACT_HASH_ALGORITHM
        or top["member_hash_algorithm"]
        != ENGINE_RUN_ARTIFACT_MEMBER_HASH_ALGORITHM
    ):
        raise EngineRunArtifactError("artifact identity contract changed")
    binding = _expect_keys(
        top["binding"],
        ("manifest_sha256", "run_id"),
        "binding",
    )
    archived_manifest_sha256 = _digest(
        binding["manifest_sha256"],
        "binding.manifest_sha256",
    )
    archived_run_id = _run_id(binding["run_id"])
    if archived_manifest_sha256 != checked_manifest_sha256:
        raise EngineRunArtifactError(
            "artifact manifest binding does not match expected_manifest_sha256"
        )
    if archived_run_id != checked_run_id:
        raise EngineRunArtifactError("artifact run binding does not match expected_run_id")
    claim = _expect_keys(
        top["claim"],
        (
            "accuracy_claimed",
            "evidence_class",
            "execution_authenticated",
            "general_superiority_claimed",
            "improvement_claimed",
            "physics_claimed",
            "production_authorized",
            "qualification_authorized",
            "registry_authorized",
            "scientific_claim_state",
        ),
        "claim",
    )
    expected_claim = {
        "accuracy_claimed": False,
        "evidence_class": "MODEL_OUTPUT",
        "execution_authenticated": False,
        "general_superiority_claimed": False,
        "improvement_claimed": False,
        "physics_claimed": False,
        "production_authorized": False,
        "qualification_authorized": False,
        "registry_authorized": False,
        "scientific_claim_state": SCIENTIFIC_CLAIM_STATE,
    }
    if _canonical(claim) != _canonical(expected_claim):
        raise EngineRunArtifactError("artifact claim ceiling changed")
    coordinates = _expect_keys(
        top["coordinates"],
        (
            "axes",
            "frame",
            "length_unit",
            "mass_unit",
            "origin",
            "time_scale",
            "time_unit",
            "unit_system_id",
        ),
        "coordinates",
    )
    checked_coordinates = {
        label: _text(coordinates[label], f"coordinates.{label}")
        for label in coordinates
    }
    run = _expect_keys(
        top["run"],
        (
            "accepted_step_ledger_content_sha256",
            "accepted_steps",
            "attempted_steps",
            "backend_id",
            "body_count",
            "body_ids",
            "checkpoint_count",
            "continuation_state_sha256",
            "device",
            "force_evaluations",
            "force_model_ids",
            "integrator_id",
            "native_result_digest",
            "native_result_digest_scope",
            "parent_native_result_digest",
            "parent_run_index",
            "plan_id",
            "rejected_steps",
            "run_index",
            "simulation_id",
            "snapshot_id",
        ),
        "run",
    )
    if (
        run["backend_id"] != "numpy"
        or run["device"] != "cpu"
        or run["integrator_id"] != ADAPTIVE_RKF78_METHOD_ID
        or run["native_result_digest_scope"] != "FULL_RETAINED_RESULT_CONTENT"
    ):
        raise EngineRunArtifactError("artifact is not the fixed NumPy/CPU RKF78 route")
    run_index = _integer(run["run_index"], "run.run_index")
    body_count = _integer(run["body_count"], "run.body_count", minimum=1)
    checkpoint_count = _integer(
        run["checkpoint_count"],
        "run.checkpoint_count",
        minimum=2,
    )
    body_ids = _string_list(run["body_ids"], "run.body_ids")
    if len(body_ids) != body_count or len(set(body_ids)) != body_count:
        raise EngineRunArtifactError("run.body_ids does not match body_count or is not unique")
    force_model_ids = _string_list(run["force_model_ids"], "run.force_model_ids")
    attempted_steps = _integer(run["attempted_steps"], "run.attempted_steps")
    accepted_steps = _integer(run["accepted_steps"], "run.accepted_steps")
    rejected_steps = _integer(run["rejected_steps"], "run.rejected_steps")
    if attempted_steps != accepted_steps + rejected_steps:
        raise EngineRunArtifactError("artifact attempted-step accounting is inconsistent")
    force_evaluations = _integer(
        run["force_evaluations"],
        "run.force_evaluations",
    )
    if force_evaluations != attempted_steps * RKF78_STAGE_COUNT:
        raise EngineRunArtifactError(
            "artifact RKF78 force-evaluation accounting is inconsistent"
        )
    if accepted_steps < checkpoint_count - 1:
        raise EngineRunArtifactError(
            "artifact accepted-step count cannot cover every checkpoint interval"
        )
    parent_run_index, parent_digest, continuation_digest = _validate_lineage(
        run_index=run_index,
        parent_run_index=run["parent_run_index"],
        parent_native_result_digest=run["parent_native_result_digest"],
        continuation_state_sha256=run["continuation_state_sha256"],
    )
    array_records = _expect_keys(top["arrays"], _ARRAY_NAMES, "arrays")
    epochs_array = _read_array(
        members["arrays/checkpoint_epochs.npy"],
        array_records["checkpoint_epochs"],
        "checkpoint_epochs",
        (checkpoint_count,),
    )
    positions = _read_array(
        members["arrays/checkpoint_positions.npy"],
        array_records["checkpoint_positions"],
        "checkpoint_positions",
        (checkpoint_count, body_count, 3),
    )
    velocities = _read_array(
        members["arrays/checkpoint_velocities.npy"],
        array_records["checkpoint_velocities"],
        "checkpoint_velocities",
        (checkpoint_count, body_count, 3),
    )
    checkpoint_epochs = tuple(float(value) for value in epochs_array)
    return JXEngineRunArtifact(
        artifact_sha256=artifact_sha256,
        manifest_sha256=archived_manifest_sha256,
        run_id=archived_run_id,
        simulation_id=_text(run["simulation_id"], "run.simulation_id"),
        run_index=run_index,
        integrator_id=run["integrator_id"],
        force_model_ids=force_model_ids,
        snapshot_id=_text(run["snapshot_id"], "run.snapshot_id"),
        plan_id=_text(run["plan_id"], "run.plan_id"),
        body_ids=body_ids,
        checkpoint_epochs=checkpoint_epochs,
        checkpoint_positions=positions,
        checkpoint_velocities=velocities,
        attempted_steps=attempted_steps,
        accepted_steps=accepted_steps,
        rejected_steps=rejected_steps,
        force_evaluations=force_evaluations,
        native_result_digest=_digest(
            run["native_result_digest"],
            "run.native_result_digest",
        ),
        native_result_digest_scope=run["native_result_digest_scope"],
        accepted_step_ledger_content_sha256=_digest(
            run["accepted_step_ledger_content_sha256"],
            "run.accepted_step_ledger_content_sha256",
        ),
        parent_run_index=parent_run_index,
        parent_native_result_digest=parent_digest,
        continuation_state_sha256=continuation_digest,
        time_scale=checked_coordinates["time_scale"],
        frame=checked_coordinates["frame"],
        origin=checked_coordinates["origin"],
        axes=checked_coordinates["axes"],
        length_unit=checked_coordinates["length_unit"],
        time_unit=checked_coordinates["time_unit"],
        mass_unit=checked_coordinates["mass_unit"],
        unit_system_id=checked_coordinates["unit_system_id"],
        _load_token=_ARTIFACT_LOAD_TOKEN,
    )


__all__ = [
    "ENGINE_RUN_ARTIFACT_ARRAY_DTYPE",
    "ENGINE_RUN_ARTIFACT_HASH_ALGORITHM",
    "ENGINE_RUN_ARTIFACT_MAX_BYTES",
    "ENGINE_RUN_ARTIFACT_MAX_MANIFEST_BYTES",
    "ENGINE_RUN_ARTIFACT_MEMBER_HASH_ALGORITHM",
    "ENGINE_RUN_ARTIFACT_SCHEMA",
    "ENGINE_RUN_ARTIFACT_SCOPE",
    "EngineRunArtifactError",
    "JXEngineRunArtifact",
    "dump_engine_run_artifact",
    "engine_run_artifact_sha256",
    "load_engine_run_artifact",
]
