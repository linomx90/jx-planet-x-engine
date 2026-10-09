#!/usr/bin/env python3
"""Materialize deterministic full-covariance element clones for 12 JX TNOs.

The output remains at each provider covariance's native epoch and native
element basis.  It deliberately performs no Cartesian conversion, epoch
propagation, dynamics, classification, inference, or Planet Nine test.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import types
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACQUISITION_TOOL = PROJECT_ROOT / "tools" / "jx_tno_sbdb_covariance_acquisition_v1.py"
INPUT_ROOT = PROJECT_ROOT / "runs" / "tno_sbdb_covariance_acquisition_v1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "tno_covariance_clones_v1"

INPUT_PROTOCOL_FILE_SHA256 = (
    "4f3de80ba98b43c907bb846bd35724a0f1bd0ef9ef152e7642e3af65a2229acc"
)
INPUT_PROTOCOL_CONTENT_SHA256 = (
    "176b56c4784bef58a8eacc6e34d99edb17d21d70c559a245b7c0677cdb1a4a60"
)
INPUT_CUSTODY_FILE_SHA256 = (
    "71e2af48651a8a3d7c6913c551c901705e699f79293f2fcd4b5303d0ac91d288"
)
INPUT_CUSTODY_CONTENT_SHA256 = (
    "d326b424aa417a573c933bd0919a9e3b4a5ebe6b7f258f1f7a6302dabdf5841f"
)
INPUT_RAW_ROSTER_SHA256 = (
    "605b191ec8df5ab52f12f5018f8376b4763e6c4e4b5aeb1998c8dae3018f8f46"
)
ACQUISITION_TOOL_SHA256 = (
    "51e96414e136e2870542383eb505c75b58c46b0e38f70fe0417b81791047eeca"
)

PROTOCOL_SCHEMA = "jx.tno.covariance-clones.protocol.v1"
CLONES_SCHEMA = "jx.tno.covariance-clones.artifact.v1"
PROTOCOL_DOMAIN = b"jx.tno.covariance-clones.protocol.v1\0"
CLONES_DOMAIN = b"jx.tno.covariance-clones.artifact.v1\0"
SEED_DOMAIN = b"jx.tno.covariance-clones.object-seed.v1\0"
UNIFORM_DOMAIN = b"jx.tno.covariance-clones.uniform.v1\0"
PAYLOAD_DOMAIN = b"jx.tno.covariance-clones.binary-payload.v1\0"
CHOLESKY_DOMAIN = b"jx.tno.covariance-clones.cholesky.v1\0"

CLASSIFICATION = "DETERMINISTIC_FULL_COVARIANCE_NATIVE_ELEMENT_CLONES_INPUT_ONLY"
CLONES_PER_OBJECT = 128
EXPECTED_OBJECTS = 12
TOTAL_CLONES = CLONES_PER_OBJECT * EXPECTED_OBJECTS
EXPECTED_LABELS = ("e", "q", "tp", "node", "peri", "i")
EXPECTED_UNITS = (None, "au", "TDB", "deg", "deg", "deg")
UNIFORM_MAPPING_ID = "SHA256_U53_CENTERED_OPEN_WITH_MAX_NEXTAFTER_V1"
NORMAL_MAPPING_ID = "BOX_MULLER_BINARY64_LIBM_V1"
SAMPLER_ID = "EXACT_DECIMAL_CHOLESKY_TO_BINARY64_THEN_FIXED_ORDER_FSUM_V1"

CLAIMS = {
    "sbdb_schema_conformant_covariance_input_verified": True,
    "provider_transport_authenticated": False,
    "full_covariance_clone_elements_materialized": True,
    "native_covariance_basis_and_epoch_preserved": True,
    "native_epoch_cartesian_transform_qualified": False,
    "de440_massive_state_adapter_qualified": False,
    "clone_dynamics_executed": False,
    "dynamical_classifier_qualified": False,
    "long_horizon_execution_authorized": False,
    "observational_likelihood_evaluated": False,
    "population_inference": False,
    "planet_nine_evidence": False,
    "science_result": False,
}

REMAINING_BLOCKERS = (
    "NATIVE_COVARIANCE_EPOCH_ELEMENT_TO_CARTESIAN_TRANSFORM_NOT_YET_QUALIFIED",
    "DE440_MASSIVE_BODY_STATES_NOT_YET_BOUND_AT_EACH_NATIVE_EPOCH",
    "TEN_MYR_HYBRID_CLOSE_ENCOUNTER_DYNAMICS_NOT_YET_QUALIFIED",
    "EXECUTABLE_RESONANCE_SCATTERING_DETACHED_CLASSIFIER_NOT_YET_FROZEN",
    "EXECUTION_RUNTIME_IDENTITY_AND_DEVICE_NOT_YET_BOUND",
)


class CloneMaterializationError(ValueError):
    """The sealed covariance input, sampler, or retained clone artifact failed."""


def _load_acquisition_module():
    # Execute the exact source bytes rather than an adjacent timestamp-valid
    # ``__pycache__`` entry.  SourceFileLoader may otherwise accept bytecode
    # that is not covered by the retained source digest.
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(ACQUISITION_TOOL, flags)
    except OSError as exc:
        raise CloneMaterializationError(
            "cannot open the sealed acquisition verifier source"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise CloneMaterializationError(
                "acquisition verifier source is linked or nonregular"
            )
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            blocks.append(block)
        payload = b"".join(blocks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_after = ACQUISITION_TOOL.lstat()
    except FileNotFoundError as exc:
        raise CloneMaterializationError(
            "acquisition verifier source disappeared while read"
        ) from exc
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    identity_path = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
    )
    if identity_before != identity_after or identity_before != identity_path:
        raise CloneMaterializationError(
            "acquisition verifier source changed while read"
        )
    if len(payload) != before.st_size:
        raise CloneMaterializationError(
            "acquisition verifier source length changed while read"
        )
    if hashlib.sha256(payload).hexdigest() != ACQUISITION_TOOL_SHA256:
        raise CloneMaterializationError("acquisition verifier source digest differs")

    name = "jx_tno_sbdb_covariance_acquisition_v1_for_clones"
    module = types.ModuleType(name)
    module.__file__ = str(ACQUISITION_TOOL)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        code = compile(
            payload,
            str(ACQUISITION_TOOL),
            "exec",
            dont_inherit=True,
            optimize=sys.flags.optimize,
        )
        exec(code, module.__dict__)
    except Exception as exc:
        sys.modules.pop(name, None)
        raise CloneMaterializationError(
            "sealed acquisition verifier source failed to execute"
        ) from exc
    return module


ACQUISITION = _load_acquisition_module()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_sha256(domain: bytes, value: Mapping[str, Any]) -> str:
    record = dict(value)
    record.pop("content_sha256", None)
    return _sha256_bytes(domain + _canonical_json(record))


def _read_regular_bytes(path: Path) -> bytes:
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise CloneMaterializationError(f"required file is absent: {path}") from exc
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise CloneMaterializationError(f"file is linked or nonregular: {path}")
    payload = path.read_bytes()
    after = path.lstat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(payload) != before.st_size
    ):
        raise CloneMaterializationError(f"file changed while read: {path}")
    return payload


def _write_create_only(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _strict_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise CloneMaterializationError("JSON has a duplicate or invalid key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise CloneMaterializationError(f"JSON contains non-finite constant {value}")


def _strict_json(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CloneMaterializationError("input is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise CloneMaterializationError("JSON root must be an object")
    return value


def _exact_input_custody(input_root: Path = INPUT_ROOT) -> dict[str, Any]:
    if _sha256_file(ACQUISITION_TOOL) != ACQUISITION_TOOL_SHA256:
        raise CloneMaterializationError("acquisition verifier source digest differs")
    if _sha256_file(input_root / "PROTOCOL.json") != INPUT_PROTOCOL_FILE_SHA256:
        raise CloneMaterializationError("input protocol file digest differs")
    if _sha256_file(input_root / "CUSTODY.json") != INPUT_CUSTODY_FILE_SHA256:
        raise CloneMaterializationError("input custody file digest differs")
    custody = ACQUISITION.validate_output(input_root)
    if custody.get("content_sha256") != INPUT_CUSTODY_CONTENT_SHA256:
        raise CloneMaterializationError("input custody content digest differs")
    if custody.get("raw_roster_sha256") != INPUT_RAW_ROSTER_SHA256:
        raise CloneMaterializationError("input raw roster digest differs")
    protocol = _strict_json(_read_regular_bytes(input_root / "PROTOCOL.json"))
    if protocol.get("content_sha256") != INPUT_PROTOCOL_CONTENT_SHA256:
        raise CloneMaterializationError("input protocol content digest differs")
    return custody


def _decimal_cholesky(
    matrix: tuple[tuple[Decimal, ...], ...]
) -> tuple[tuple[Decimal, ...], ...]:
    with localcontext() as context:
        context.prec = 100
        lower = [[Decimal(0) for _ in range(6)] for _ in range(6)]
        for row in range(6):
            for column in range(row + 1):
                residual = matrix[row][column]
                for prior in range(column):
                    residual -= lower[row][prior] * lower[column][prior]
                if row == column:
                    if residual <= 0:
                        raise CloneMaterializationError("covariance is not positive definite")
                    lower[row][column] = residual.sqrt()
                else:
                    lower[row][column] = residual / lower[column][column]
        return tuple(tuple(row) for row in lower)


def _provider_mean_matrix(raw_path: Path, expected_raw_sha256: str) -> tuple[
    tuple[Decimal, ...], tuple[tuple[Decimal, ...], ...], str
]:
    if (
        type(expected_raw_sha256) is not str
        or len(expected_raw_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_raw_sha256)
    ):
        raise CloneMaterializationError("expected raw digest is invalid")
    payload = _read_regular_bytes(raw_path)
    if _sha256_bytes(payload) != expected_raw_sha256:
        raise CloneMaterializationError("sampled raw response differs from sealed custody")
    value = _strict_json(payload)
    try:
        covariance = value["orbit"]["covariance"]
        labels = tuple(covariance["labels"])
        units = tuple(row["units"] for row in covariance["elements"])
        mean = tuple(Decimal(str(row["value"])) for row in covariance["elements"])
        matrix = tuple(
            tuple(Decimal(str(component)) for component in row)
            for row in covariance["data"]
        )
        epoch = str(covariance["epoch"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CloneMaterializationError("provider covariance structure changed") from exc
    if labels != EXPECTED_LABELS or units != EXPECTED_UNITS:
        raise CloneMaterializationError("provider covariance basis changed")
    if len(mean) != 6 or len(matrix) != 6 or any(len(row) != 6 for row in matrix):
        raise CloneMaterializationError("provider covariance dimensions changed")
    return mean, matrix, epoch


def _open_uniform(seed: bytes, clone_index: int, slot: int) -> float:
    if len(seed) != 32 or type(clone_index) is not int or type(slot) is not int:
        raise CloneMaterializationError("invalid deterministic sampler coordinate")
    preimage = (
        UNIFORM_DOMAIN
        + seed
        + clone_index.to_bytes(4, "big", signed=False)
        + slot.to_bytes(2, "big", signed=False)
    )
    raw = int.from_bytes(hashlib.sha256(preimage).digest()[:8], "big") >> 11
    value = (raw + 0.5) * (2.0 ** -53)
    if value == 1.0:
        value = math.nextafter(1.0, 0.0)
    if not (0.0 < value < 1.0):
        raise CloneMaterializationError("open-uniform mapping reached an endpoint")
    return value


def _standard_normals(seed: bytes, clone_index: int) -> tuple[float, ...]:
    values: list[float] = []
    for pair_index in range(3):
        u1 = _open_uniform(seed, clone_index, 2 * pair_index)
        u2 = _open_uniform(seed, clone_index, 2 * pair_index + 1)
        radius = math.sqrt(-2.0 * math.log(u1))
        phase = math.tau * u2
        values.extend((radius * math.cos(phase), radius * math.sin(phase)))
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        raise CloneMaterializationError("standard-normal mapping produced invalid output")
    return tuple(values)


def _physical_native_element_vector(values: tuple[float, ...]) -> None:
    if len(values) != 6 or not all(type(value) is float and math.isfinite(value) for value in values):
        raise CloneMaterializationError("clone element vector is not finite binary64")
    eccentricity, perihelion, _tp, _node, _peri, inclination = values
    if not (0.0 < eccentricity < 1.0):
        raise CloneMaterializationError("sampled eccentricity is outside the elliptic domain")
    if perihelion <= 0.0:
        raise CloneMaterializationError("sampled perihelion distance is not positive")
    if not (0.0 <= inclination <= 180.0):
        raise CloneMaterializationError("sampled inclination is outside the physical domain")


def _binary_payload_sha256(objects: list[dict[str, Any]]) -> str:
    payload = bytearray()
    for object_record in objects:
        payload.extend(int(object_record["spk_id"]).to_bytes(8, "little", signed=False))
        for clone in object_record["clones"]:
            payload.extend(int(clone["clone_index"]).to_bytes(4, "little", signed=False))
            for key in ("z_hex", "native_elements_hex"):
                for value in clone[key]:
                    payload.extend(struct.pack("<d", float.fromhex(value)))
    return _sha256_bytes(PAYLOAD_DOMAIN + bytes(payload))


def build_protocol(input_root: Path = INPUT_ROOT) -> dict[str, object]:
    custody = _exact_input_custody(input_root)
    record: dict[str, object] = {
        "schema": PROTOCOL_SCHEMA,
        "classification": CLASSIFICATION,
        "status": "PREDECLARED_NATIVE_ELEMENT_CLONE_MATERIALIZATION_NOT_DYNAMICS",
        "claim_ceiling": "DETERMINISTIC_COVARIANCE_CLONE_ELEMENT_INPUTS_ONLY",
        "input": {
            "root_relative_path": "runs/tno_sbdb_covariance_acquisition_v1",
            "acquisition_tool_file_sha256": ACQUISITION_TOOL_SHA256,
            "protocol_file_sha256": INPUT_PROTOCOL_FILE_SHA256,
            "protocol_content_sha256": INPUT_PROTOCOL_CONTENT_SHA256,
            "custody_file_sha256": INPUT_CUSTODY_FILE_SHA256,
            "custody_content_sha256": INPUT_CUSTODY_CONTENT_SHA256,
            "raw_roster_sha256": INPUT_RAW_ROSTER_SHA256,
            "record_count": custody["record_count"],
        },
        "sampler": {
            "clones_per_object": CLONES_PER_OBJECT,
            "total_objects": EXPECTED_OBJECTS,
            "total_clones": TOTAL_CLONES,
            "basis_labels": list(EXPECTED_LABELS),
            "basis_units": list(EXPECTED_UNITS),
            "sample_at_native_covariance_epoch": True,
            "linear_rule": "x=mu+Lz",
            "cholesky_decimal_precision": 100,
            "uniform_mapping_id": UNIFORM_MAPPING_ID,
            "normal_mapping_id": NORMAL_MAPPING_ID,
            "sampler_id": SAMPLER_ID,
            "object_seed_rule": (
                "SHA256(ASCII('jx.tno.covariance-clones.object-seed.v1') || "
                "NUL || ASCII_HEX64(protocol_content_sha256) || NUL || "
                "ASCII(object_id))"
            ),
            "no_redraw": True,
            "no_clipping": True,
            "no_independent_element_perturbations": True,
            "selection_cuts_reapplied_to_clones": False,
            "angles_remain_unwrapped_in_local_covariance_basis": True,
            "cross_runtime_bitwise_reproducible": False,
            "retained_clone_bytes_are_authoritative": True,
            "libm_operations": ["log", "sqrt", "cos", "sin", "nextafter", "fsum"],
        },
        "implementation": {
            "relative_path": "tools/jx_tno_covariance_clone_materialization_v1.py",
            "file_sha256": _sha256_file(Path(__file__)),
            "python_implementation": sys.implementation.name,
            "python_version": ".".join(str(value) for value in sys.version_info[:3]),
        },
        "claims": dict(CLAIMS),
        "remaining_blockers": list(REMAINING_BLOCKERS),
        "execution_authorized": False,
        "outcomes_generated": False,
        "nonclaim": (
            "The retained vectors are deterministic Gaussian draws in the provider's "
            "native local element covariance. They are not Cartesian initial states, "
            "trajectories, classifications, population inference, or Planet Nine evidence."
        ),
    }
    record["content_sha256"] = _content_sha256(PROTOCOL_DOMAIN, record)
    return record


def validate_protocol(value: Mapping[str, Any], input_root: Path = INPUT_ROOT) -> str:
    expected = build_protocol(input_root)
    if type(value) is not dict or value != expected:
        raise CloneMaterializationError("protocol differs from frozen reconstruction")
    digest = _content_sha256(PROTOCOL_DOMAIN, value)
    if value.get("content_sha256") != digest:
        raise CloneMaterializationError("protocol content digest differs")
    return digest


def build_clones(
    protocol: Mapping[str, Any], input_root: Path = INPUT_ROOT
) -> dict[str, object]:
    validate_protocol(protocol, input_root)
    custody = _exact_input_custody(input_root)
    records_by_spk = {int(record["spk_id"]): record for record in custody["records"]}
    if len(records_by_spk) != EXPECTED_OBJECTS:
        raise CloneMaterializationError("input custody object roster changed")
    objects: list[dict[str, Any]] = []
    for target in ACQUISITION.TARGETS:
        input_record = records_by_spk.get(target.spk_id)
        if input_record is None or input_record.get("object_id") != target.object_id:
            raise CloneMaterializationError("input custody target binding changed")
        mean_decimal, matrix_decimal, epoch = _provider_mean_matrix(
            input_root / "raw" / target.filename,
            str(input_record["raw_sha256"]),
        )
        lower_decimal = _decimal_cholesky(matrix_decimal)
        lower = tuple(tuple(float(value) for value in row) for row in lower_decimal)
        if not all(math.isfinite(value) for row in lower for value in row):
            raise CloneMaterializationError("binary64 Cholesky factor is non-finite")
        seed = hashlib.sha256(
            SEED_DOMAIN
            + str(protocol["content_sha256"]).encode("ascii")
            + b"\0"
            + target.object_id.encode("ascii")
        ).digest()
        clones: list[dict[str, object]] = []
        seen_vectors: set[tuple[str, ...]] = set()
        for clone_index in range(CLONES_PER_OBJECT):
            z = _standard_normals(seed, clone_index)
            sampled = tuple(
                float(mean_decimal[row])
                + math.fsum(lower[row][column] * z[column] for column in range(row + 1))
                for row in range(6)
            )
            _physical_native_element_vector(sampled)
            sampled_hex = tuple(value.hex() for value in sampled)
            if sampled_hex in seen_vectors:
                raise CloneMaterializationError("duplicate clone element vector generated")
            seen_vectors.add(sampled_hex)
            clones.append(
                {
                    "clone_index": clone_index,
                    "z_hex": [value.hex() for value in z],
                    "native_elements_hex": list(sampled_hex),
                }
            )
        lower_hex = [[value.hex() for value in row] for row in lower]
        cholesky_sha256 = _sha256_bytes(
            CHOLESKY_DOMAIN
            + _canonical_json(
                {"labels": list(EXPECTED_LABELS), "lower_binary64_hex": lower_hex}
            )
        )
        objects.append(
            {
                "object_id": target.object_id,
                "spk_id": target.spk_id,
                "cohort": target.cohort,
                "covariance_epoch_jd_tdb": epoch,
                "covariance_matrix_sha256": input_record["covariance_matrix_sha256"],
                "covariance_mean_sha256": input_record["covariance_mean_sha256"],
                "raw_sha256": input_record["raw_sha256"],
                "object_seed_sha256": seed.hex(),
                "nominal_native_elements_hex": [float(value).hex() for value in mean_decimal],
                "cholesky_lower_binary64_hex": lower_hex,
                "cholesky_binary64_sha256": cholesky_sha256,
                "clone_count": CLONES_PER_OBJECT,
                "clones": clones,
            }
        )
    artifact: dict[str, object] = {
        "schema": CLONES_SCHEMA,
        "classification": CLASSIFICATION,
        "status": "TWELVE_OBJECTS_WITH_128_FULL_COVARIANCE_NATIVE_ELEMENT_CLONES_EACH_INPUT_ONLY",
        "claim_ceiling": "DETERMINISTIC_COVARIANCE_CLONE_ELEMENT_INPUTS_ONLY",
        "protocol_content_sha256": protocol["content_sha256"],
        "input_custody_content_sha256": INPUT_CUSTODY_CONTENT_SHA256,
        "object_count": len(objects),
        "clones_per_object": CLONES_PER_OBJECT,
        "total_clone_count": sum(int(row["clone_count"]) for row in objects),
        "objects": objects,
        "binary_payload_sha256": _binary_payload_sha256(objects),
        "sampling_outcome": {
            "redraw_count": 0,
            "clipping_count": 0,
            "selection_rejection_count": 0,
            "nonphysical_sample_count": 0,
            "duplicate_vector_count": 0,
        },
        "claims": dict(CLAIMS),
        "remaining_blockers": list(REMAINING_BLOCKERS),
        "execution_authorized": False,
        "outcomes_generated": False,
        "nonclaim": (
            "These retained element vectors close the independent-sigma defect for the "
            "12-object pilot input only. They do not qualify a Cartesian transform, "
            "dynamics engine, classifier, population result, or Planet Nine result."
        ),
    }
    artifact["content_sha256"] = _content_sha256(CLONES_DOMAIN, artifact)
    return artifact


def _load_canonical(path: Path) -> dict[str, Any]:
    payload = _read_regular_bytes(path)
    value = _strict_json(payload)
    if _canonical_json(value) != payload:
        raise CloneMaterializationError(f"retained file is not canonical JSON: {path}")
    return value


def prepare(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, object]:
    if output_root.is_symlink():
        raise CloneMaterializationError("output root cannot be a symlink")
    output_root.mkdir(parents=True, exist_ok=True)
    unexpected = {entry.name for entry in output_root.iterdir()} - {
        "PROTOCOL.json",
        "CLONES.json",
    }
    if unexpected:
        raise CloneMaterializationError("output root contains unexpected entries")
    protocol = build_protocol()
    payload = _canonical_json(protocol)
    path = output_root / "PROTOCOL.json"
    if path.exists():
        if _read_regular_bytes(path) != payload:
            raise CloneMaterializationError("output root contains a different protocol")
    else:
        _write_create_only(path, payload)
    return protocol


def materialize(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, object]:
    protocol = prepare(output_root)
    path = output_root / "CLONES.json"
    if path.exists() or path.is_symlink():
        raise CloneMaterializationError("clone artifact already exists")
    artifact = build_clones(protocol)
    _write_create_only(path, _canonical_json(artifact))
    validate_output(output_root)
    return artifact


def validate_output(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, object]:
    if output_root.is_symlink() or not output_root.is_dir():
        raise CloneMaterializationError("output root must be a real directory")
    if {entry.name for entry in output_root.iterdir()} != {"PROTOCOL.json", "CLONES.json"}:
        raise CloneMaterializationError("output root roster differs")
    protocol = _load_canonical(output_root / "PROTOCOL.json")
    validate_protocol(protocol)
    expected = build_clones(protocol)
    actual = _load_canonical(output_root / "CLONES.json")
    if actual != expected:
        raise CloneMaterializationError("clone artifact differs from exact reconstruction")
    if actual.get("content_sha256") != _content_sha256(CLONES_DOMAIN, actual):
        raise CloneMaterializationError("clone artifact content digest differs")
    return actual


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "materialize", "validate"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            value = prepare(args.output_root)
        elif args.command == "materialize":
            value = materialize(args.output_root)
        else:
            value = validate_output(args.output_root)
    except CloneMaterializationError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
