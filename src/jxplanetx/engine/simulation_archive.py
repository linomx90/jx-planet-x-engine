"""Deterministic, non-pickle restart archives for unified simulations.

Archive v1 stores one explicitly selected NumPy continuation snapshot.  A
caller must supply the force plan again when loading; its complete canonical
content must match the archived digest.  This keeps orientation providers and
other executable dependencies out of an unsafe serialization format while
still binding their public identities, parameters, and provenance.
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
from .backends import resolve_backend
from .contracts import ForcePlan, Provenance, StateSnapshot
from .integrator_registry import get_integrator_route
from .simulation import (
    CONTINUATION_DIGEST_ALGORITHM,
    CONTINUATION_DIGEST_DOMAIN,
    SCIENTIFIC_CLAIM_STATE,
    JXSimulation,
    JXSimulationCompatibilityError,
    JXSimulationContinuation,
    JXSimulationContractError,
    _continuation_state_identity_sha256,
    _validate_route_state_force_plan,
)
from .trajectory import _canonical_result_content


SIMULATION_ARCHIVE_SCHEMA = "jxplanetx.simulation-restart-archive.v1"
SIMULATION_ARCHIVE_SCOPE = "NUMPY_CPU_FLOAT64_EXPLICIT_CONTINUATION"
SIMULATION_ARCHIVE_HASH_ALGORITHM = "SHA256_ARCHIVE_BYTES"
FORCE_PLAN_DIGEST_ALGORITHM = "SHA256_CANONICAL_FORCE_PLAN_JSON_V1"
FORCE_PLAN_DIGEST_DOMAIN = "jxplanetx.simulation.force-plan.v1"
SIMULATION_ARCHIVE_MAX_BYTES = 256 * 1024 * 1024
SIMULATION_ARCHIVE_MAX_MANIFEST_BYTES = 1024 * 1024
_ARRAY_NAMES = (
    "gravitational_parameters",
    "masses",
    "massive",
    "positions",
    "radii",
    "velocities",
)
_MEMBER_NAMES = tuple(f"arrays/{name}.npy" for name in _ARRAY_NAMES) + (
    "manifest.json",
)
_ARCHIVE_LOAD_TOKEN = object()


class SimulationArchiveError(JXSimulationContractError):
    """A restart archive is malformed, mismatched, or outside v1 scope."""


@dataclass(frozen=True, slots=True, eq=False)
class JXSimulationArchiveLoad:
    """Content-checked archive bytes and declared lineage rooted at one endpoint."""

    simulation: JXSimulation
    continuation: JXSimulationContinuation
    archive_sha256: str
    force_plan_sha256: str
    schema: str = SIMULATION_ARCHIVE_SCHEMA
    scope: str = SIMULATION_ARCHIVE_SCOPE
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    production_authorized: bool = False
    _load_token: InitVar[object] = None

    def __post_init__(self, _load_token: object) -> None:
        if _load_token is not _ARCHIVE_LOAD_TOKEN:
            raise SimulationArchiveError(
                "JXSimulationArchiveLoad must be constructed by "
                "load_simulation_archive"
            )
        if type(self.simulation) is not JXSimulation:
            raise SimulationArchiveError("simulation must be an exact JXSimulation")
        if type(self.continuation) is not JXSimulationContinuation:
            raise SimulationArchiveError(
                "continuation must be an exact JXSimulationContinuation"
            )
        for value, label in (
            (self.archive_sha256, "archive_sha256"),
            (self.force_plan_sha256, "force_plan_sha256"),
        ):
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise SimulationArchiveError(f"{label} must be lowercase SHA-256 hex")
        if (
            self.schema != SIMULATION_ARCHIVE_SCHEMA
            or self.scope != SIMULATION_ARCHIVE_SCOPE
            or self.scientific_claim_state != SCIENTIFIC_CLAIM_STATE
            or type(self.production_authorized) is not bool
            or self.production_authorized
        ):
            raise SimulationArchiveError(
                "archive load must retain the fixed v1 screening-only contract"
            )


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
        raise SimulationArchiveError("archive manifest is not canonical JSON") from exc
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
        raise SimulationArchiveError(
            f"JSON floating token {token!r} is forbidden; binary64 uses hex strings"
        )

    def reject_constant(token: str) -> object:
        raise SimulationArchiveError(f"nonfinite JSON constant {token!r} is forbidden")

    try:
        document = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=pairs,
            parse_float=reject_float,
            parse_constant=reject_constant,
        )
    except SimulationArchiveError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise SimulationArchiveError("archive manifest must be ASCII JSON") from exc
    if duplicate is not None:
        raise SimulationArchiveError(f"archive manifest repeats key {duplicate!r}")
    if type(document) is not dict or _canonical(document) != raw:
        raise SimulationArchiveError("archive manifest is not canonical")
    return document


def _expect_keys(value: object, keys: tuple[str, ...], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(keys) or len(value) != len(keys):
        raise SimulationArchiveError(f"{label} must contain exactly {keys!r}")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise SimulationArchiveError(f"{label} must be nonempty trimmed text")
    return value


def _digest(value: object, label: str) -> str:
    checked = _text(value, label)
    if len(checked) != 64 or any(character not in "0123456789abcdef" for character in checked):
        raise SimulationArchiveError(f"{label} must be lowercase SHA-256 hex")
    return checked


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SimulationArchiveError(f"{label} must be an integer >= {minimum}")
    return value


def _float_from_hex(value: object, label: str) -> float:
    if type(value) is not str:
        raise SimulationArchiveError(f"{label} must be canonical binary64 hex")
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise SimulationArchiveError(f"{label} must be canonical binary64 hex") from exc
    if not math.isfinite(result) or result.hex() != value:
        raise SimulationArchiveError(f"{label} must be canonical finite binary64 hex")
    return result


def _provenance_document(value: Provenance) -> dict[str, str]:
    return {
        "citation": value.citation,
        "sha256": value.sha256,
        "source_id": value.source_id,
        "version": value.version,
    }


def _provenance_from_document(value: object) -> Provenance:
    document = _expect_keys(
        value,
        ("citation", "sha256", "source_id", "version"),
        "snapshot.provenance",
    )
    return Provenance(
        source_id=_text(document["source_id"], "snapshot.provenance.source_id"),
        citation=_text(document["citation"], "snapshot.provenance.citation"),
        version=_text(document["version"], "snapshot.provenance.version"),
        sha256=_digest(document["sha256"], "snapshot.provenance.sha256"),
    )


def simulation_force_plan_sha256(plan: ForcePlan) -> str:
    """Return the canonical NumPy/CPU force-plan identity used by archives."""

    if type(plan) is not ForcePlan:
        raise SimulationArchiveError("plan must be an exact ForcePlan")
    if plan.backend.backend_id != "numpy" or plan.backend.device != "cpu":
        raise SimulationArchiveError("archive v1 supports only NumPy/CPU force plans")
    backend = resolve_backend(plan.backend)
    with backend.activate():
        canonical = _canonical_result_content(backend, plan)
    payload = _canonical(
        {
            "domain": FORCE_PLAN_DIGEST_DOMAIN,
            "force_plan": canonical,
        }
    )
    return _sha256(FORCE_PLAN_DIGEST_DOMAIN.encode("ascii") + b"\x00" + payload)


def _array_bytes(value: object, label: str) -> bytes:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - engine extra owns NumPy
        raise SimulationArchiveError("archive v1 requires NumPy") from exc
    if type(value) is not np.ndarray or not value.flags.c_contiguous:
        raise SimulationArchiveError(f"{label} must be a C-contiguous NumPy array")
    stream = io.BytesIO()
    np.lib.format.write_array(stream, value, allow_pickle=False)
    return stream.getvalue()


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def dump_simulation_archive(
    simulation: JXSimulation,
    parent_run_index: int,
) -> bytes:
    """Return deterministic bytes for one explicit continuation checkpoint."""

    if type(simulation) is not JXSimulation:
        raise SimulationArchiveError("simulation must be an exact JXSimulation")
    continuation = simulation.prepare_continuation(parent_run_index)
    try:
        route = get_integrator_route(continuation.parent_integrator_id)
    except KeyError as exc:
        raise SimulationArchiveError(
            "the parent integrator is not a registered archive route"
        ) from exc
    backend = simulation.force_plan.backend
    if not route.supports_archive(backend.backend_id, backend.device):
        raise SimulationArchiveError(
            "the parent integrator route does not support restart archives "
            f"on backend {backend.backend_id!r}"
        )
    try:
        _validate_route_state_force_plan(
            route,
            continuation.snapshot,
            simulation.force_plan,
        )
    except JXSimulationCompatibilityError as exc:
        raise SimulationArchiveError(
            "the parent integrator route is incompatible with the force plan"
        ) from exc
    snapshot = continuation.snapshot
    plan_sha256 = simulation_force_plan_sha256(simulation.force_plan)
    arrays = {
        name: _array_bytes(getattr(snapshot, name), name) for name in _ARRAY_NAMES
    }
    array_manifest = {
        name: {
            "member": f"arrays/{name}.npy",
            "sha256": _sha256(raw),
            "size_bytes": len(raw),
        }
        for name, raw in arrays.items()
    }
    manifest = {
        "arrays": array_manifest,
        "claim": {
            "production_authorized": False,
            "scientific_claim_state": SCIENTIFIC_CLAIM_STATE,
        },
        "continuation": {
            "digest_algorithm": CONTINUATION_DIGEST_ALGORITHM,
            "digest_domain": CONTINUATION_DIGEST_DOMAIN,
            "force_model_ids": list(continuation.force_model_ids),
            "parent_integrator_id": continuation.parent_integrator_id,
            "parent_native_result_digest": continuation.parent_native_result_digest,
            "parent_run_index": continuation.parent_run_index,
            "state_sha256": continuation.continuation_state_sha256,
        },
        "force_plan": {
            "digest_algorithm": FORCE_PLAN_DIGEST_ALGORITHM,
            "digest_domain": FORCE_PLAN_DIGEST_DOMAIN,
            "sha256": plan_sha256,
        },
        "hash_algorithm": SIMULATION_ARCHIVE_HASH_ALGORITHM,
        "schema": SIMULATION_ARCHIVE_SCHEMA,
        "scope": SIMULATION_ARCHIVE_SCOPE,
        "simulation_id": simulation.simulation_id,
        "snapshot": {
            "axes": snapshot.axes,
            "body_ids": list(snapshot.body_ids),
            "epoch_hex": float(snapshot.epoch).hex(),
            "frame": snapshot.frame,
            "length_unit": snapshot.length_unit,
            "mass_unit": snapshot.mass_unit,
            "origin": snapshot.origin,
            "provenance": _provenance_document(snapshot.provenance),
            "snapshot_id": snapshot.snapshot_id,
            "time_scale": snapshot.time_scale,
            "time_unit": snapshot.time_unit,
            "unit_system_id": snapshot.unit_system_id,
        },
    }
    members = {
        **{f"arrays/{name}.npy": raw for name, raw in arrays.items()},
        "manifest.json": _canonical(manifest),
    }
    if len(members["manifest.json"]) > SIMULATION_ARCHIVE_MAX_MANIFEST_BYTES:
        raise SimulationArchiveError("archive manifest exceeds the v1 size ceiling")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in _MEMBER_NAMES:
            archive.writestr(_zip_info(name), members[name])
    result = stream.getvalue()
    if len(result) > SIMULATION_ARCHIVE_MAX_BYTES:
        raise SimulationArchiveError("archive exceeds the v1 size ceiling")
    return result


def simulation_archive_sha256(archive_bytes: bytes) -> str:
    """Return the exact restart-archive byte identity."""

    if type(archive_bytes) is not bytes:
        raise SimulationArchiveError("archive_bytes must be exact bytes")
    return _sha256(archive_bytes)


def _read_array(
    archive: zipfile.ZipFile,
    record: object,
    name: str,
    body_count: int,
) -> object:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - engine extra owns NumPy
        raise SimulationArchiveError("archive v1 requires NumPy") from exc
    document = _expect_keys(record, ("member", "sha256", "size_bytes"), f"arrays.{name}")
    expected_member = f"arrays/{name}.npy"
    if document["member"] != expected_member:
        raise SimulationArchiveError(f"arrays.{name}.member is not canonical")
    size = _integer(document["size_bytes"], f"arrays.{name}.size_bytes", minimum=1)
    if size > SIMULATION_ARCHIVE_MAX_BYTES:
        raise SimulationArchiveError(f"arrays.{name} exceeds the size ceiling")
    try:
        raw = archive.read(expected_member)
    except (KeyError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise SimulationArchiveError(f"arrays.{name} could not be read safely") from exc
    if len(raw) != size or _sha256(raw) != _digest(
        document["sha256"], f"arrays.{name}.sha256"
    ):
        raise SimulationArchiveError(f"arrays.{name} byte identity mismatch")
    expected_shape = (body_count, 3) if name in {"positions", "velocities"} else (body_count,)
    expected_dtype = np.dtype(np.bool_) if name == "massive" else np.dtype(np.float64)
    stream = io.BytesIO(raw)
    try:
        version = np.lib.format.read_magic(stream)
        if version != (1, 0):
            raise SimulationArchiveError(
                f"arrays.{name} must use canonical NPY 1.0"
            )
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
            stream,
            max_header_size=4096,
        )
    except SimulationArchiveError:
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
        raise SimulationArchiveError(
            f"arrays.{name} has an invalid NPY header"
        ) from exc
    payload_offset = stream.tell()
    expected_payload_bytes = math.prod(expected_shape) * expected_dtype.itemsize
    if (
        dtype != expected_dtype
        or shape != expected_shape
        or fortran_order is not False
        or len(raw) - payload_offset != expected_payload_bytes
    ):
        raise SimulationArchiveError(
            f"arrays.{name} has the wrong dtype, shape, or memory order"
        )
    try:
        loaded = np.frombuffer(
            raw,
            dtype=expected_dtype,
            count=math.prod(expected_shape),
            offset=payload_offset,
        ).reshape(expected_shape)
        owned = np.ascontiguousarray(loaded.copy())
    except (TypeError, ValueError, MemoryError) as exc:
        raise SimulationArchiveError(
            f"arrays.{name} payload could not be loaded safely"
        ) from exc
    if _array_bytes(owned, name) != raw:
        raise SimulationArchiveError(f"arrays.{name} NPY encoding is not canonical")
    owned.setflags(write=False)
    return owned


def load_simulation_archive(
    archive_bytes: bytes,
    *,
    expected_sha256: str,
    force_plan: ForcePlan,
) -> JXSimulationArchiveLoad:
    """Load restart bytes against a structurally compatible exact live force plan."""

    if type(archive_bytes) is not bytes or not archive_bytes:
        raise SimulationArchiveError("archive_bytes must be nonempty exact bytes")
    if len(archive_bytes) > SIMULATION_ARCHIVE_MAX_BYTES:
        raise SimulationArchiveError("archive exceeds the v1 size ceiling")
    archive_sha256 = simulation_archive_sha256(archive_bytes)
    if archive_sha256 != _digest(expected_sha256, "expected_sha256"):
        raise SimulationArchiveError("archive SHA-256 does not match expected_sha256")
    try:
        preflight_exact_zip_roster(archive_bytes, _MEMBER_NAMES)
        archive_context = zipfile.ZipFile(io.BytesIO(archive_bytes), "r")
    except ValueError as exc:
        raise SimulationArchiveError(
            "archive member roster or central directory is not canonical"
        ) from exc
    except (RuntimeError, zipfile.BadZipFile) as exc:
        raise SimulationArchiveError("archive bytes are not a valid ZIP container") from exc
    with archive_context as archive:
        infos = archive.infolist()
        names = tuple(info.filename for info in infos)
        if names != _MEMBER_NAMES or len(set(names)) != len(names):
            raise SimulationArchiveError("archive member roster or order is not canonical")
        for info in infos:
            if (
                info.is_dir()
                or info.file_size > SIMULATION_ARCHIVE_MAX_BYTES
                or info.compress_type != zipfile.ZIP_STORED
            ):
                raise SimulationArchiveError("archive contains an invalid member")
            if (
                info.filename == "manifest.json"
                and info.file_size > SIMULATION_ARCHIVE_MAX_MANIFEST_BYTES
            ):
                raise SimulationArchiveError(
                    "archive manifest exceeds the v1 size ceiling"
                )
            mode = (info.external_attr >> 16) & 0o170000
            if mode not in (0, 0o100000):
                raise SimulationArchiveError("archive members must be regular files")
        if sum(info.file_size for info in infos) > SIMULATION_ARCHIVE_MAX_BYTES:
            raise SimulationArchiveError("archive members exceed the total size ceiling")
        canonical_stream = io.BytesIO()
        try:
            with zipfile.ZipFile(
                canonical_stream,
                "w",
                compression=zipfile.ZIP_STORED,
            ) as canonical_archive:
                for name in _MEMBER_NAMES:
                    raw_member = archive.read(name)
                    canonical_archive.writestr(_zip_info(name), raw_member)
                    del raw_member
        except (KeyError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
            raise SimulationArchiveError(
                "archive members could not be read safely"
            ) from exc
        if canonical_stream.getvalue() != archive_bytes:
            raise SimulationArchiveError("archive ZIP encoding is not canonical")
        canonical_stream.close()
        try:
            manifest_raw = archive.read("manifest.json")
        except (KeyError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
            raise SimulationArchiveError("archive manifest could not be read safely") from exc
        manifest = _strict_json(manifest_raw)
        top = _expect_keys(
            manifest,
            (
                "arrays",
                "claim",
                "continuation",
                "force_plan",
                "hash_algorithm",
                "schema",
                "scope",
                "simulation_id",
                "snapshot",
            ),
            "manifest",
        )
        if (
            top["schema"] != SIMULATION_ARCHIVE_SCHEMA
            or top["scope"] != SIMULATION_ARCHIVE_SCOPE
            or top["hash_algorithm"] != SIMULATION_ARCHIVE_HASH_ALGORITHM
        ):
            raise SimulationArchiveError("archive identity contract changed")
        claim = _expect_keys(
            top["claim"],
            ("production_authorized", "scientific_claim_state"),
            "claim",
        )
        expected_claim = {
            "production_authorized": False,
            "scientific_claim_state": SCIENTIFIC_CLAIM_STATE,
        }
        if _canonical(claim) != _canonical(expected_claim):
            raise SimulationArchiveError("archive claim ceiling changed")
        force_record = _expect_keys(
            top["force_plan"],
            ("digest_algorithm", "digest_domain", "sha256"),
            "force_plan",
        )
        if (
            force_record["digest_algorithm"] != FORCE_PLAN_DIGEST_ALGORITHM
            or force_record["digest_domain"] != FORCE_PLAN_DIGEST_DOMAIN
        ):
            raise SimulationArchiveError("force-plan digest contract changed")
        archived_plan_sha256 = _digest(force_record["sha256"], "force_plan.sha256")
        if simulation_force_plan_sha256(force_plan) != archived_plan_sha256:
            raise SimulationArchiveError("supplied force plan does not match the archive")

        snapshot_record = _expect_keys(
            top["snapshot"],
            (
                "axes",
                "body_ids",
                "epoch_hex",
                "frame",
                "length_unit",
                "mass_unit",
                "origin",
                "provenance",
                "snapshot_id",
                "time_scale",
                "time_unit",
                "unit_system_id",
            ),
            "snapshot",
        )
        raw_body_ids = snapshot_record["body_ids"]
        if (
            type(raw_body_ids) is not list
            or not raw_body_ids
            or any(type(item) is not str or not item for item in raw_body_ids)
            or len(set(raw_body_ids)) != len(raw_body_ids)
        ):
            raise SimulationArchiveError("snapshot.body_ids is invalid")
        body_ids = tuple(raw_body_ids)
        array_records = _expect_keys(top["arrays"], _ARRAY_NAMES, "arrays")
        arrays = {
            name: _read_array(archive, array_records[name], name, len(body_ids))
            for name in _ARRAY_NAMES
        }
        snapshot = StateSnapshot(
            snapshot_id=_text(snapshot_record["snapshot_id"], "snapshot.snapshot_id"),
            epoch=_float_from_hex(snapshot_record["epoch_hex"], "snapshot.epoch_hex"),
            time_scale=_text(snapshot_record["time_scale"], "snapshot.time_scale"),
            frame=_text(snapshot_record["frame"], "snapshot.frame"),
            origin=_text(snapshot_record["origin"], "snapshot.origin"),
            axes=_text(snapshot_record["axes"], "snapshot.axes"),
            length_unit=_text(snapshot_record["length_unit"], "snapshot.length_unit"),
            time_unit=_text(snapshot_record["time_unit"], "snapshot.time_unit"),
            mass_unit=_text(snapshot_record["mass_unit"], "snapshot.mass_unit"),
            unit_system_id=_text(
                snapshot_record["unit_system_id"], "snapshot.unit_system_id"
            ),
            body_ids=body_ids,
            positions=arrays["positions"],
            velocities=arrays["velocities"],
            gravitational_parameters=arrays["gravitational_parameters"],
            masses=arrays["masses"],
            radii=arrays["radii"],
            massive=arrays["massive"],
            provenance=_provenance_from_document(snapshot_record["provenance"]),
        )

    continuation_record = _expect_keys(
        top["continuation"],
        (
            "digest_algorithm",
            "digest_domain",
            "force_model_ids",
            "parent_integrator_id",
            "parent_native_result_digest",
            "parent_run_index",
            "state_sha256",
        ),
        "continuation",
    )
    if (
        continuation_record["digest_algorithm"] != CONTINUATION_DIGEST_ALGORITHM
        or continuation_record["digest_domain"] != CONTINUATION_DIGEST_DOMAIN
    ):
        raise SimulationArchiveError("continuation digest contract changed")
    raw_force_ids = continuation_record["force_model_ids"]
    if (
        type(raw_force_ids) is not list
        or not raw_force_ids
        or any(type(item) is not str or not item for item in raw_force_ids)
    ):
        raise SimulationArchiveError("continuation.force_model_ids is invalid")
    force_model_ids = tuple(raw_force_ids)
    if force_model_ids != tuple(model.model_id for model in force_plan.models):
        raise SimulationArchiveError("continuation force roster differs from force plan")
    state_sha256 = _digest(
        continuation_record["state_sha256"], "continuation.state_sha256"
    )
    if snapshot.provenance.sha256 != state_sha256:
        raise SimulationArchiveError("snapshot provenance is not the continuation state")
    simulation_id = _text(top["simulation_id"], "simulation_id")
    parent_run_index = _integer(
        continuation_record["parent_run_index"],
        "continuation.parent_run_index",
    )
    parent_integrator_id = _text(
        continuation_record["parent_integrator_id"],
        "continuation.parent_integrator_id",
    )
    try:
        parent_route = get_integrator_route(parent_integrator_id)
    except KeyError as exc:
        raise SimulationArchiveError(
            "the archived parent integrator is not a registered route"
        ) from exc
    backend = force_plan.backend
    if not parent_route.supports_archive(backend.backend_id, backend.device):
        raise SimulationArchiveError(
            "the archived parent integrator route does not support restart archives "
            f"on backend {backend.backend_id!r}"
        )
    try:
        _validate_route_state_force_plan(parent_route, snapshot, force_plan)
    except JXSimulationCompatibilityError as exc:
        raise SimulationArchiveError(
            "the archived parent integrator route is incompatible with the force plan"
        ) from exc
    parent_native_result_digest = _digest(
        continuation_record["parent_native_result_digest"],
        "continuation.parent_native_result_digest",
    )
    recomputed_state_sha256 = _continuation_state_identity_sha256(
        simulation_id=simulation_id,
        parent_run_index=parent_run_index,
        parent_integrator_id=parent_integrator_id,
        parent_native_result_digest=parent_native_result_digest,
        force_model_ids=force_model_ids,
        device="cpu",
        snapshot=snapshot,
    )
    if recomputed_state_sha256 != state_sha256:
        raise SimulationArchiveError(
            "continuation state digest does not match lineage and array content"
        )
    continuation = JXSimulationContinuation(
        simulation_id=simulation_id,
        parent_run_index=parent_run_index,
        parent_integrator_id=parent_integrator_id,
        parent_native_result_digest=parent_native_result_digest,
        continuation_state_sha256=state_sha256,
        snapshot=snapshot,
        force_model_ids=force_model_ids,
    )
    restarted = JXSimulation(simulation_id, snapshot, force_plan)
    return JXSimulationArchiveLoad(
        simulation=restarted,
        continuation=continuation,
        archive_sha256=archive_sha256,
        force_plan_sha256=archived_plan_sha256,
        _load_token=_ARCHIVE_LOAD_TOKEN,
    )


__all__ = [
    "FORCE_PLAN_DIGEST_ALGORITHM",
    "FORCE_PLAN_DIGEST_DOMAIN",
    "JXSimulationArchiveLoad",
    "SIMULATION_ARCHIVE_HASH_ALGORITHM",
    "SIMULATION_ARCHIVE_MAX_BYTES",
    "SIMULATION_ARCHIVE_MAX_MANIFEST_BYTES",
    "SIMULATION_ARCHIVE_SCHEMA",
    "SIMULATION_ARCHIVE_SCOPE",
    "SimulationArchiveError",
    "dump_simulation_archive",
    "load_simulation_archive",
    "simulation_archive_sha256",
    "simulation_force_plan_sha256",
]
