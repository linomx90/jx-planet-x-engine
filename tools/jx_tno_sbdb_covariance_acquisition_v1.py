#!/usr/bin/env python3
"""Acquire and seal full JPL/SBDB covariance records for the 12 JX TNOs.

The retained bytes and receipt establish input custody only.  They do not
sample a covariance, transform an orbit, run dynamics, classify a TNO, or
support a population or Planet Nine claim.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "tno_sbdb_covariance_acquisition_v1"

PROTOCOL_SCHEMA = "jx.tno.sbdb-covariance-acquisition.protocol.v1"
CUSTODY_SCHEMA = "jx.tno.sbdb-covariance-acquisition.custody.v1"
PROTOCOL_DOMAIN = b"jx.tno.sbdb-covariance-acquisition.protocol.v1\0"
CUSTODY_DOMAIN = b"jx.tno.sbdb-covariance-acquisition.custody.v1\0"
MATRIX_DOMAIN = b"jx.tno.sbdb-covariance-acquisition.matrix.v1\0"
MEAN_DOMAIN = b"jx.tno.sbdb-covariance-acquisition.mean.v1\0"
ROSTER_DOMAIN = b"jx.tno.sbdb-covariance-acquisition.roster.v1\0"

CLASSIFICATION = "SBDB_SCHEMA_CONFORMANT_FULL_COVARIANCE_BYTES_INPUT_CUSTODY_ONLY"
SBDB_ENDPOINT = "https://ssd-api.jpl.nasa.gov/sbdb.api"
QUERY_SUFFIX = "cov=mat&full-prec=true"
EXPECTED_LABELS = ("e", "q", "tp", "node", "peri", "i")
EXPECTED_ELEMENT_NAMES = ("e", "q", "tp", "om", "w", "i")
EXPECTED_UNITS = (None, "au", "TDB", "deg", "deg", "deg")
EXPECTED_STANDARD_EPOCH = Decimal("2461200.5")
SIGMA_DIAGONAL_RELATIVE_TOLERANCE = Decimal("1e-12")
IMPLEMENTATION_RELATIVE_PATH = "tools/jx_tno_sbdb_covariance_acquisition_v1.py"
SCIENCE_INPUT_RELATIVE_PATH = "src/jxplanetx/long_horizon/science_inputs.py"
OBSERVED_CROSSCHECK_RESULT_RELATIVE_PATH = (
    "runs/tno_observed_numerical_crosscheck_v1/RESULT.json"
)

SCIENCE_INPUT_SOURCE_SHA256 = (
    "810853b60d203679a100c8eca203d62316e902f67ca4be6dbe7fd8042f18c1e2"
)
OBSERVED_CROSSCHECK_RESULT_SHA256 = (
    "8f2237b7ce69a6000bcc5ef7c600c2090a345ce217dc788b865c2a26c6bfbdd2"
)
POPULATION_BINDINGS = (
    {
        "population_id": "jpl-observed-detached-tno-v1",
        "content_sha256": "b3cdcddd91de97138f70f1ddbe3f7ab09d60cc3e2708143a5a3c6cf1ebc7d7b0",
    },
    {
        "population_id": "jpl-observed-scattering-control-v1",
        "content_sha256": "16e6d45e69e135642b8ac4e3710adb47e1e8e50a4183e417cb017722b0abdc53",
    },
)


@dataclass(frozen=True, slots=True)
class Target:
    object_id: str
    spk_id: int
    primary_designation: str
    cohort: str

    @property
    def filename(self) -> str:
        return f"{self.spk_id}.json"

    @property
    def query_url(self) -> str:
        return f"{SBDB_ENDPOINT}?sstr={self.spk_id}&{QUERY_SUFFIX}"

    def as_record(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "spk_id": self.spk_id,
            "primary_designation": self.primary_designation,
            "cohort": self.cohort,
            "retained_filename": self.filename,
            "query_url": self.query_url,
        }


TARGETS = (
    Target("jpl-20090377", 20090377, "90377", "DETACHED_TNO_SCREEN_V1"),
    Target("jpl-20474640", 20474640, "474640", "DETACHED_TNO_SCREEN_V1"),
    Target("jpl-54526835", 54526835, "2023 KQ14", "DETACHED_TNO_SCREEN_V1"),
    Target("jpl-20541132", 20541132, "541132", "DETACHED_TNO_SCREEN_V1"),
    Target("jpl-50666516", 50666516, "2012 VP113", "DETACHED_TNO_SCREEN_V1"),
    Target("jpl-20148209", 20148209, "148209", "DETACHED_TNO_SCREEN_V1"),
    Target("jpl-20574372", 20574372, "574372", "SCATTERING_DISK_CONTROL_V1"),
    Target("jpl-20303775", 20303775, "303775", "SCATTERING_DISK_CONTROL_V1"),
    Target("jpl-20026181", 20026181, "26181", "SCATTERING_DISK_CONTROL_V1"),
    Target("jpl-20042301", 20042301, "42301", "SCATTERING_DISK_CONTROL_V1"),
    Target("jpl-20026375", 20026375, "26375", "SCATTERING_DISK_CONTROL_V1"),
    Target("jpl-20015874", 20015874, "15874", "SCATTERING_DISK_CONTROL_V1"),
)

CLAIMS = {
    "sbdb_schema_conformant_raw_covariance_bytes_retained": True,
    "covariance_schema_validated": True,
    "covariance_positive_definite": True,
    "provider_transport_authenticated": False,
    "authoritative_provider_origin_cryptographically_proven": False,
    "covariance_sampling_authorized": False,
    "native_epoch_cartesian_transform_qualified": False,
    "clone_population_generated": False,
    "dynamical_classifier_qualified": False,
    "long_horizon_execution_authorized": False,
    "observational_likelihood_evaluated": False,
    "population_inference": False,
    "planet_nine_evidence": False,
    "science_result": False,
}

REMAINING_BLOCKERS = (
    "NATIVE_COVARIANCE_EPOCH_TO_CARTESIAN_TRANSFORM_NOT_YET_QUALIFIED",
    "DE440_MASSIVE_BODY_STATES_NOT_YET_BOUND_AT_EACH_COVARIANCE_EPOCH",
    "COVARIANCE_CLONE_SAMPLER_AND_ROUNDTRIP_GATES_NOT_YET_FROZEN",
    "TEN_MYR_HYBRID_DYNAMICS_AND_CLASSIFIER_NOT_YET_QUALIFIED",
    "EXECUTION_RUNTIME_IDENTITY_AND_DEVICE_NOT_YET_BOUND",
)


class CovarianceAcquisitionError(ValueError):
    """The protocol, provider response, or retained bundle is invalid."""


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


def _strict_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise CovarianceAcquisitionError("provider JSON has a duplicate or invalid key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise CovarianceAcquisitionError(f"provider JSON contains non-finite constant {value}")


def _strict_json(payload: bytes) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CovarianceAcquisitionError("provider response is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise CovarianceAcquisitionError("provider JSON root must be an object")
    return value


def _mapping(value: object, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise CovarianceAcquisitionError(f"{label} must be an exact object")
    return value


def _exact_list(value: object, length: int, label: str) -> list[Any]:
    if type(value) is not list or len(value) != length:
        raise CovarianceAcquisitionError(f"{label} must be an exact {length}-item array")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise CovarianceAcquisitionError(f"{label} must be a nonempty trimmed string")
    return value


def _decimal(value: object, label: str) -> Decimal:
    if type(value) not in (str, int, float):
        raise CovarianceAcquisitionError(f"{label} is not a decimal scalar")
    if type(value) is float and not math.isfinite(value):
        raise CovarianceAcquisitionError(f"{label} is non-finite")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise CovarianceAcquisitionError(f"{label} is not decimal") from exc
    if not result.is_finite():
        raise CovarianceAcquisitionError(f"{label} is non-finite")
    return result


def _positive_int(value: object, label: str) -> int:
    if type(value) is bool:
        raise CovarianceAcquisitionError(f"{label} must be a positive integer")
    if type(value) is int:
        result = value
    elif type(value) is str and value.isdigit():
        result = int(value)
    else:
        raise CovarianceAcquisitionError(f"{label} must be a positive integer")
    if result <= 0:
        raise CovarianceAcquisitionError(f"{label} must be positive")
    return result


def _cholesky_positive_definite(matrix: tuple[tuple[Decimal, ...], ...]) -> None:
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
                        raise CovarianceAcquisitionError(
                            "provider covariance is not positive definite"
                        )
                    lower[row][column] = residual.sqrt()
                else:
                    lower[row][column] = residual / lower[column][column]


def _canonical_decimal(value: Decimal) -> str:
    return str(value.normalize()) if value != 0 else "0"


def validate_response(payload: bytes, target: Target) -> dict[str, object]:
    if type(payload) is not bytes or not payload:
        raise CovarianceAcquisitionError("provider response must be nonempty bytes")
    value = _strict_json(payload)
    if "error" in value:
        raise CovarianceAcquisitionError("provider returned an error object")

    signature = _mapping(value.get("signature"), "signature")
    signature_source = _text(signature.get("source"), "signature.source")
    signature_version = _text(signature.get("version"), "signature.version")
    if "NASA/JPL" not in signature_source or "SBDB" not in signature_source:
        raise CovarianceAcquisitionError("provider signature is not NASA/JPL SBDB")
    if signature_version != "1.3":
        raise CovarianceAcquisitionError("provider schema version differs from 1.3")

    object_record = _mapping(value.get("object"), "object")
    if _positive_int(object_record.get("spkid"), "object.spkid") != target.spk_id:
        raise CovarianceAcquisitionError("provider object SPK ID differs from target")
    if _text(object_record.get("des"), "object.des") != target.primary_designation:
        raise CovarianceAcquisitionError("provider designation differs from target")
    orbit_class = _mapping(object_record.get("orbit_class"), "object.orbit_class")
    if orbit_class.get("code") != "TNO":
        raise CovarianceAcquisitionError("provider object is not classed as a TNO")

    orbit = _mapping(value.get("orbit"), "orbit")
    orbit_id = _text(orbit.get("orbit_id"), "orbit.orbit_id")
    if _text(object_record.get("orbit_id"), "object.orbit_id") != orbit_id:
        raise CovarianceAcquisitionError("object and orbit solution IDs differ")
    if orbit.get("equinox") != "J2000" or orbit.get("source") != "JPL":
        raise CovarianceAcquisitionError("provider orbit frame/source differs")
    if orbit.get("pe_used") != "DE441":
        raise CovarianceAcquisitionError("provider planetary ephemeris is not DE441")
    standard_epoch = _decimal(orbit.get("epoch"), "orbit.epoch")
    if standard_epoch != EXPECTED_STANDARD_EPOCH:
        raise CovarianceAcquisitionError("standard solution epoch differs from JX epoch")
    observations = _positive_int(orbit.get("n_obs_used"), "orbit.n_obs_used")
    data_arc_days = _positive_int(orbit.get("data_arc"), "orbit.data_arc")

    covariance = _mapping(orbit.get("covariance"), "orbit.covariance")
    labels = tuple(_exact_list(covariance.get("labels"), 6, "covariance.labels"))
    if labels != EXPECTED_LABELS:
        raise CovarianceAcquisitionError("covariance labels or ordering differ")
    covariance_epoch = _decimal(covariance.get("epoch"), "covariance.epoch")
    if not (Decimal("2400000") < covariance_epoch < Decimal("2500000")):
        raise CovarianceAcquisitionError("covariance epoch is outside the supported JD range")

    elements = _exact_list(covariance.get("elements"), 6, "covariance.elements")
    mean_values: list[str] = []
    sigma_values: list[str] = []
    sigma_decimals: list[Decimal] = []
    for index, (element, expected_label, expected_name, expected_unit) in enumerate(
        zip(elements, EXPECTED_LABELS, EXPECTED_ELEMENT_NAMES, EXPECTED_UNITS)
    ):
        row = _mapping(element, f"covariance.elements[{index}]")
        if row.get("label") != expected_label or row.get("name") != expected_name:
            raise CovarianceAcquisitionError("covariance element semantics differ")
        if row.get("units") != expected_unit:
            raise CovarianceAcquisitionError("covariance element units differ")
        mean = _decimal(row.get("value"), f"covariance.elements[{index}].value")
        sigma = _decimal(row.get("sigma"), f"covariance.elements[{index}].sigma")
        if sigma <= 0:
            raise CovarianceAcquisitionError("covariance element sigma must be positive")
        mean_values.append(_canonical_decimal(mean))
        sigma_values.append(_canonical_decimal(sigma))
        sigma_decimals.append(sigma)

    data = _exact_list(covariance.get("data"), 6, "covariance.data")
    decimal_rows: list[tuple[Decimal, ...]] = []
    canonical_rows: list[list[str]] = []
    for row_index, raw_row in enumerate(data):
        row = _exact_list(raw_row, 6, f"covariance.data[{row_index}]")
        decimals = tuple(
            _decimal(component, f"covariance.data[{row_index}][{column_index}]")
            for column_index, component in enumerate(row)
        )
        decimal_rows.append(decimals)
        canonical_rows.append([_canonical_decimal(component) for component in decimals])
    matrix = tuple(decimal_rows)
    for row in range(6):
        if matrix[row][row] <= 0:
            raise CovarianceAcquisitionError("covariance diagonal must be positive")
        for column in range(6):
            if matrix[row][column] != matrix[column][row]:
                raise CovarianceAcquisitionError("provider covariance is not symmetric")
    _cholesky_positive_definite(matrix)
    relative_sigma_diagonal_errors = tuple(
        abs(sigma_decimals[index] * sigma_decimals[index] - matrix[index][index])
        / matrix[index][index]
        for index in range(6)
    )
    maximum_sigma_diagonal_relative_error = max(relative_sigma_diagonal_errors)
    if maximum_sigma_diagonal_relative_error > SIGMA_DIAGONAL_RELATIVE_TOLERANCE:
        raise CovarianceAcquisitionError(
            "provider covariance sigmas disagree with the matrix diagonal"
        )

    matrix_sha256 = _sha256_bytes(
        MATRIX_DOMAIN + _canonical_json({"labels": list(labels), "data": canonical_rows})
    )
    mean_sha256 = _sha256_bytes(
        MEAN_DOMAIN
        + _canonical_json(
            {
                "labels": list(labels),
                "values": mean_values,
                "sigmas": sigma_values,
                "epoch_jd_tdb": _canonical_decimal(covariance_epoch),
            }
        )
    )
    return {
        "object_id": target.object_id,
        "spk_id": target.spk_id,
        "primary_designation": target.primary_designation,
        "full_name": _text(object_record.get("fullname"), "object.fullname"),
        "cohort": target.cohort,
        "query_url": target.query_url,
        "retained_relative_path": f"raw/{target.filename}",
        "raw_byte_count": len(payload),
        "raw_sha256": _sha256_bytes(payload),
        "provider_signature_source": signature_source,
        "provider_signature_version": signature_version,
        "orbit_solution_id": orbit_id,
        "orbit_solution_date": _text(orbit.get("soln_date"), "orbit.soln_date"),
        "planetary_ephemeris": "DE441",
        "standard_epoch_jd_tdb": _canonical_decimal(standard_epoch),
        "covariance_epoch_jd_tdb": _canonical_decimal(covariance_epoch),
        "covariance_labels": list(labels),
        "covariance_units": list(EXPECTED_UNITS),
        "covariance_matrix_sha256": matrix_sha256,
        "covariance_mean_sha256": mean_sha256,
        "covariance_exactly_symmetric": True,
        "covariance_positive_definite": True,
        "maximum_sigma_diagonal_relative_error": _canonical_decimal(
            maximum_sigma_diagonal_relative_error
        ),
        "sigma_diagonal_relative_tolerance": _canonical_decimal(
            SIGMA_DIAGONAL_RELATIVE_TOLERANCE
        ),
        "observations_used": observations,
        "data_arc_days": data_arc_days,
    }


def _require_bound_project_files() -> dict[str, str]:
    expected = {
        SCIENCE_INPUT_RELATIVE_PATH: SCIENCE_INPUT_SOURCE_SHA256,
        OBSERVED_CROSSCHECK_RESULT_RELATIVE_PATH: OBSERVED_CROSSCHECK_RESULT_SHA256,
    }
    for relative_path, expected_sha256 in expected.items():
        path = PROJECT_ROOT / relative_path
        if _sha256_file(path) != expected_sha256:
            raise CovarianceAcquisitionError(
                f"bound project file differs: {relative_path}"
            )
    return expected


def build_protocol() -> dict[str, object]:
    bound_files = _require_bound_project_files()
    record: dict[str, object] = {
        "schema": PROTOCOL_SCHEMA,
        "classification": CLASSIFICATION,
        "status": "PREDECLARED_ACQUISITION_ROSTER_NOT_YET_SCIENCE_READY",
        "claim_ceiling": "SCHEMA_CONFORMANT_INPUT_CUSTODY_AND_MATRIX_VALIDATION_ONLY",
        "endpoint": SBDB_ENDPOINT,
        "query_contract": {
            "parameters": {"cov": "mat", "full-prec": "true"},
            "documentation_url": "https://ssd-api.jpl.nasa.gov/doc/sbdb.html",
            "expected_covariance_labels": list(EXPECTED_LABELS),
            "expected_covariance_units": list(EXPECTED_UNITS),
            "expected_standard_epoch_jd_tdb": "2461200.5",
            "expected_equinox": "J2000",
            "expected_planetary_ephemeris": "DE441",
            "expected_api_schema_version": "1.3",
            "sigma_diagonal_relative_tolerance": _canonical_decimal(
                SIGMA_DIAGONAL_RELATIVE_TOLERANCE
            ),
        },
        "target_roster": [target.as_record() for target in TARGETS],
        "jx_input_bindings": {
            "science_inputs_relative_path": SCIENCE_INPUT_RELATIVE_PATH,
            "science_inputs_file_sha256": bound_files[SCIENCE_INPUT_RELATIVE_PATH],
            "populations": list(POPULATION_BINDINGS),
            "observed_crosscheck_result_relative_path": OBSERVED_CROSSCHECK_RESULT_RELATIVE_PATH,
            "observed_crosscheck_result_file_sha256": (
                bound_files[OBSERVED_CROSSCHECK_RESULT_RELATIVE_PATH]
            ),
        },
        "implementation": {
            "relative_path": IMPLEMENTATION_RELATIVE_PATH,
            "file_sha256": _sha256_file(Path(__file__)),
        },
        "validation": {
            "strict_json_and_duplicate_key_rejection": True,
            "exact_target_identity": True,
            "exact_six_dimensional_label_and_unit_contract": True,
            "exact_decimal_symmetry": True,
            "high_precision_positive_definite_cholesky": True,
            "sigma_matches_matrix_diagonal_within_declared_tolerance": True,
            "retain_raw_provider_bytes": True,
            "create_only_publication": True,
        },
        "claims": dict(CLAIMS),
        "remaining_blockers": list(REMAINING_BLOCKERS),
        "execution_authorized": False,
        "outcomes_generated": False,
        "nonclaim": (
            "This protocol and its custody receipt retain SBDB-schema-conformant "
            "covariance bytes acquired through an offline staging boundary. Their "
            "provider origin is not cryptographically authenticated. "
            "They do not authorize sampling, coordinate transformation, dynamics, "
            "classification, population inference, or a Planet Nine claim."
        ),
    }
    record["content_sha256"] = _content_sha256(PROTOCOL_DOMAIN, record)
    return record


def validate_protocol(value: Mapping[str, Any]) -> str:
    expected = build_protocol()
    if type(value) is not dict or value != expected:
        raise CovarianceAcquisitionError("protocol differs from the frozen reconstruction")
    digest = _content_sha256(PROTOCOL_DOMAIN, value)
    if value.get("content_sha256") != digest:
        raise CovarianceAcquisitionError("protocol content digest differs")
    return digest


def _read_regular_bytes(path: Path) -> bytes:
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise CovarianceAcquisitionError(f"required file is absent: {path}") from exc
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise CovarianceAcquisitionError(f"file is linked or nonregular: {path}")
    payload = path.read_bytes()
    after = path.lstat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(payload) != before.st_size
    ):
        raise CovarianceAcquisitionError(f"file changed while read: {path}")
    return payload


def _load_json_file(path: Path) -> dict[str, Any]:
    value = _strict_json(_read_regular_bytes(path))
    if _canonical_json(value) != _read_regular_bytes(path):
        raise CovarianceAcquisitionError(f"retained manifest is not canonical JSON: {path}")
    return value


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


def _root_entries(path: Path) -> set[str]:
    return {entry.name for entry in path.iterdir()}


def prepare(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, object]:
    if output_root.is_symlink():
        raise CovarianceAcquisitionError("output root cannot be a symlink")
    output_root.mkdir(parents=True, exist_ok=True)
    allowed = {"PROTOCOL.json", "raw", "CUSTODY.json"}
    unexpected = _root_entries(output_root) - allowed
    if unexpected:
        raise CovarianceAcquisitionError("output root contains unexpected entries")
    protocol = build_protocol()
    path = output_root / "PROTOCOL.json"
    payload = _canonical_json(protocol)
    if path.exists():
        if _read_regular_bytes(path) != payload:
            raise CovarianceAcquisitionError("output root contains a different protocol")
    else:
        _write_create_only(path, payload)
    return protocol


def verify_staging(staging_root: Path) -> tuple[dict[str, object], ...]:
    if staging_root.is_symlink() or not staging_root.is_dir():
        raise CovarianceAcquisitionError("staging root must be a real directory")
    expected = {target.filename for target in TARGETS}
    if _root_entries(staging_root) != expected:
        raise CovarianceAcquisitionError("staging root roster differs from protocol")
    return tuple(
        validate_response(_read_regular_bytes(staging_root / target.filename), target)
        for target in TARGETS
    )


def _copy_raw_create_only(staging_root: Path, output_root: Path) -> None:
    raw = output_root / "raw"
    if raw.exists():
        expected = {target.filename for target in TARGETS}
        if raw.is_symlink() or not raw.is_dir() or _root_entries(raw) != expected:
            raise CovarianceAcquisitionError("existing raw directory is incomplete or unexpected")
        for target in TARGETS:
            if _read_regular_bytes(raw / target.filename) != _read_regular_bytes(
                staging_root / target.filename
            ):
                raise CovarianceAcquisitionError("existing raw response differs from staging")
        return
    pending = output_root / f".raw.pending-{os.getpid()}"
    if pending.exists() or pending.is_symlink():
        raise CovarianceAcquisitionError("pending raw publication path already exists")
    pending.mkdir(mode=0o700)
    try:
        for target in TARGETS:
            source = staging_root / target.filename
            destination = pending / target.filename
            _write_create_only(destination, _read_regular_bytes(source))
        if raw.exists() or raw.is_symlink():
            raise CovarianceAcquisitionError("raw publication target appeared concurrently")
        os.rename(pending, raw)
    except Exception:
        if pending.exists() and pending.is_dir():
            shutil.rmtree(pending)
        raise


def _build_custody(
    protocol: Mapping[str, Any],
    records: tuple[dict[str, object], ...],
    protocol_bytes: bytes,
    published_at_utc: str | None = None,
) -> dict[str, object]:
    if published_at_utc is None:
        published_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if type(published_at_utc) is not str or not published_at_utc.endswith("Z"):
        raise CovarianceAcquisitionError("publication timestamp must be UTC text")
    try:
        parsed_timestamp = datetime.fromisoformat(published_at_utc[:-1] + "+00:00")
    except ValueError as exc:
        raise CovarianceAcquisitionError("publication timestamp is invalid") from exc
    if parsed_timestamp.tzinfo != timezone.utc:
        raise CovarianceAcquisitionError("publication timestamp is not UTC")
    roster_sha256 = _sha256_bytes(
        ROSTER_DOMAIN
        + _canonical_json(
            [
                {
                    "spk_id": record["spk_id"],
                    "raw_sha256": record["raw_sha256"],
                    "covariance_matrix_sha256": record["covariance_matrix_sha256"],
                    "covariance_mean_sha256": record["covariance_mean_sha256"],
                }
                for record in records
            ]
        )
    )
    custody: dict[str, object] = {
        "schema": CUSTODY_SCHEMA,
        "classification": CLASSIFICATION,
        "status": "TWELVE_SBDB_SCHEMA_CONFORMANT_FULL_COVARIANCE_RECORDS_RETAINED_INPUT_ONLY",
        "claim_ceiling": "SCHEMA_CONFORMANT_INPUT_CUSTODY_AND_MATRIX_VALIDATION_ONLY",
        "protocol": {
            "relative_path": "PROTOCOL.json",
            "file_sha256": _sha256_bytes(protocol_bytes),
            "content_sha256": protocol["content_sha256"],
        },
        "record_count": len(records),
        "records": list(records),
        "raw_roster_sha256": roster_sha256,
        "total_retained_raw_bytes": sum(int(record["raw_byte_count"]) for record in records),
        "all_covariances_exactly_symmetric": True,
        "all_covariances_positive_definite": True,
        "all_standard_epochs_match_jx": True,
        "covariance_epochs_are_native_and_object_specific": True,
        "claims": dict(CLAIMS),
        "remaining_blockers": list(REMAINING_BLOCKERS),
        "execution_authorized": False,
        "outcomes_generated": False,
        "nonclaim": (
            "This custody receipt retains 12 SBDB-schema-conformant full covariance "
            "records. Offline staging does not cryptographically prove provider origin. "
            "It is not a clone set, transform, "
            "dynamics run, classification, population result, or Planet Nine result."
        ),
        "published_at_utc": published_at_utc,
    }
    custody["content_sha256"] = _content_sha256(CUSTODY_DOMAIN, custody)
    return custody


def publish(staging_root: Path, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, object]:
    protocol = prepare(output_root)
    protocol_bytes = _read_regular_bytes(output_root / "PROTOCOL.json")
    records = verify_staging(staging_root)
    receipt_path = output_root / "CUSTODY.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        raise CovarianceAcquisitionError("custody receipt already exists")
    _copy_raw_create_only(staging_root, output_root)
    custody = _build_custody(protocol, records, protocol_bytes)
    _write_create_only(receipt_path, _canonical_json(custody))
    validate_output(output_root)
    return custody


def validate_output(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, object]:
    if output_root.is_symlink() or not output_root.is_dir():
        raise CovarianceAcquisitionError("output root must be a real directory")
    if _root_entries(output_root) != {"PROTOCOL.json", "CUSTODY.json", "raw"}:
        raise CovarianceAcquisitionError("retained output root roster differs")
    protocol = _load_json_file(output_root / "PROTOCOL.json")
    validate_protocol(protocol)
    raw = output_root / "raw"
    expected_raw = {target.filename for target in TARGETS}
    if raw.is_symlink() or not raw.is_dir() or _root_entries(raw) != expected_raw:
        raise CovarianceAcquisitionError("retained raw roster differs")
    records = tuple(
        validate_response(_read_regular_bytes(raw / target.filename), target)
        for target in TARGETS
    )
    actual = _load_json_file(output_root / "CUSTODY.json")
    expected = _build_custody(
        protocol,
        records,
        _read_regular_bytes(output_root / "PROTOCOL.json"),
        published_at_utc=actual.get("published_at_utc"),
    )
    if actual != expected:
        raise CovarianceAcquisitionError("custody receipt differs from reconstruction")
    if actual.get("content_sha256") != _content_sha256(CUSTODY_DOMAIN, actual):
        raise CovarianceAcquisitionError("custody content digest differs")
    return actual


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "publish", "validate"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--staging-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            value = prepare(args.output_root)
        elif args.command == "publish":
            if args.staging_root is None:
                raise CovarianceAcquisitionError("publish requires --staging-root")
            value = publish(args.staging_root, args.output_root)
        else:
            value = validate_output(args.output_root)
    except CovarianceAcquisitionError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
