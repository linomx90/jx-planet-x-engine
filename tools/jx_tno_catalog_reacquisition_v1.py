#!/usr/bin/env python3
"""Reacquire and seal the small OSSOS/DES catalog payload set.

This program is deliberately transport-independent: a caller downloads the
nine predeclared public payloads into a staging directory, then this verifier
checks exact byte counts and SHA-256 digests before publishing a create-only
local custody bundle.  It performs no catalog transformation, orbit work,
survey modeling, dynamics, or inference.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import tarfile
from typing import Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
G0_ROOT = PROJECT_ROOT / "audits" / "jx_o2_g0_input_audit_v1"
DOSSIER_PATH = PROJECT_ROOT / "docs" / "TNO_RESEARCH_DOSSIER.md"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "tno_catalog_reacquisition_v1"

PROTOCOL_SCHEMA = "jx.tno.catalog-reacquisition.protocol.v1"
CUSTODY_SCHEMA = "jx.tno.catalog-reacquisition.custody.v1"
PROTOCOL_DOMAIN = b"jx.tno.catalog-reacquisition.protocol.v1\0"
CUSTODY_DOMAIN = b"jx.tno.catalog-reacquisition.custody.v1\0"
CLASSIFICATION = "TNO_CATALOG_REACQUISITION_AND_BYTE_CUSTODY_ONLY"

EXPECTED_G0_FILES = {
    "README.md": "77f3a80d8d9e8b21e309d8fe435e764994cd62eaca5f5edfb02c3826198837a0",
    "ossos_inventory_v1.json": "67223671b3f5a983062ad3f7745eb52c3011cf138b737ad871a41141bb5a908b",
    "des_y6_inventory_v1.json": "ab180a22dab51aa70884d1f7f8db8bb442f2ba85f79bde687d9ca5191e86d18f",
}
EXPECTED_DOSSIER_SHA256 = (
    "6bd71bb7f82ab316068e6a15e5d14117f7e48c97d06474de0d608fa8b179665f"
)


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    artifact_id: str
    staging_name: str
    retained_name: str
    url: str
    byte_count: int
    sha256: str
    survey: str
    role: str
    record_count: int | None = None
    license_status: str = ""

    def as_record(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "staging_name": self.staging_name,
            "retained_name": self.retained_name,
            "url": self.url,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
            "survey": self.survey,
            "role": self.role,
            "record_count": self.record_count,
            "license_status": self.license_status,
        }


ARTIFACTS = (
    ArtifactSpec(
        "OSSOS_CDS_README",
        "ossos_ReadMe",
        "ossos_cds_ReadMe",
        "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/236/18/ReadMe",
        12_708,
        "362db5bb7e4f39675f855285be448154f62eba1ba0997ecabbb4a52bb188e25d",
        "OSSOS",
        "catalog_schema_and_metadata",
        license_status="AUTHORITATIVE_REDISTRIBUTION_LICENSE_NOT_FOUND",
    ),
    ArtifactSpec(
        "OSSOS_CHARACTERIZED_CATALOG",
        "ossos_t3char.dat",
        "ossos_t3char.dat",
        "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/236/18/t3char.dat",
        225_960,
        "35c6af083804788d589e1f7fbcbf3db30fecd6d32d94be3de359815e99da596f",
        "OSSOS",
        "characterized_catalog",
        record_count=840,
        license_status="AUTHORITATIVE_REDISTRIBUTION_LICENSE_NOT_FOUND",
    ),
    ArtifactSpec(
        "OSSOS_PLUS_ENSEMBLE_CATALOG",
        "ossos_t3ens.dat",
        "ossos_t3ens.dat",
        "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/236/18/t3ens.dat",
        308_812,
        "86ddb04586247103207722841b5e78cbea17d009217926ea0690a068b60683b8",
        "OSSOS",
        "ensemble_detection_catalog",
        record_count=1_148,
        license_status="AUTHORITATIVE_REDISTRIBUTION_LICENSE_NOT_FOUND",
    ),
    ArtifactSpec(
        "OSSOS_UNCHARACTERIZED_CATALOG",
        "ossos_t3unc.dat",
        "ossos_t3unc.dat",
        "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/236/18/t3unc.dat",
        29_321,
        "90c50df2ae9172792029611ccc71bb9d7f3bddba151debd0830a089737fce82b",
        "OSSOS",
        "uncharacterized_catalog",
        record_count=109,
        license_status="AUTHORITATIVE_REDISTRIBUTION_LICENSE_NOT_FOUND",
    ),
    ArtifactSpec(
        "DES_Y6_IOP_TABLE3_ARCHIVE",
        "des_revision1_table3.tar.gz",
        "des_revision1_table3.tar.gz",
        "https://content.cld.iop.org/journals/0067-0049/258/2/41/revision1/apjsac3914_table3.tar.gz",
        327_951,
        "1bc41364cb9d0ba3ad21b00e4f4467c5de83da0675a44c5e6ecb31df5457458e",
        "DES_Y6",
        "canonical_revision1_journal_table_archive",
        record_count=814,
        license_status="CC-BY-4.0",
    ),
    ArtifactSpec(
        "DES_Y6_CDS_README",
        "des_cds_ReadMe",
        "des_cds_ReadMe",
        "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/258/41/ReadMe",
        11_035,
        "4dff6940798eafd4ad172ab81b8e531a9952839db5d67de9ca8e1f869f2eb5ca",
        "DES_Y6",
        "catalog_schema_and_metadata",
        license_status="CC-BY-4.0",
    ),
    ArtifactSpec(
        "DES_Y6_CDS_CATALOG",
        "des_cds_catalog.dat",
        "des_cds_catalog.dat",
        "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/258/41/catalog.dat",
        498_168,
        "457554872af8b55b44404f4dfbaeeceafb8120cc43cb09b1350f4e48c9bc1a92",
        "DES_Y6",
        "canonical_revision1_cds_catalog",
        record_count=814,
        license_status="CC-BY-4.0",
    ),
    ArtifactSpec(
        "DES_Y6_ENHANCED_AUTHOR_CATALOG",
        "des_y6_des_tnos_color.fits",
        "des_y6_des_tnos_color.fits",
        "https://raw.githubusercontent.com/bernardinelli/des_tno_catalog/8f254ca30fa0e8fc1576a5a9fad7300124d3b3ee/y6_des_tnos_color.fits",
        527_040,
        "4f2153f8726485bb661cbf9027a840c69e11c6d87c925d386055b3fdff4ea6e8",
        "DES_Y6",
        "enhanced_author_catalog_with_states_and_covariances",
        record_count=814,
        license_status="CC-BY-4.0",
    ),
    ArtifactSpec(
        "DES_Y6_AUTHOR_LICENSE",
        "des_author_LICENSE.txt",
        "des_author_LICENSE.txt",
        "https://raw.githubusercontent.com/bernardinelli/des_tno_catalog/8f254ca30fa0e8fc1576a5a9fad7300124d3b3ee/LICENSE.txt",
        1_354,
        "c407847d0bf5ea5b428b7bb3145b0899ccd7d22e345d045c217567ef520603a8",
        "DES_Y6",
        "software_and_data_license_record",
        license_status="SOFTWARE_MIT_DATA_CC-BY-4.0",
    ),
)

CLAIMS = {
    "catalog_semantics_validated": False,
    "master_catalog_constructed": False,
    "des_epoch_time_scale_locked": False,
    "covariance_sampling_authorized": False,
    "dynamical_classifier_authorized": False,
    "survey_selection_model_ready": False,
    "observational_likelihood_evaluated": False,
    "planet_nine_evidence": False,
    "science_result": False,
    "gpu_execution_authorized": False,
}


class ReacquisitionError(ValueError):
    """The protocol, staged bytes, or retained custody bundle is invalid."""


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
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_sha256(domain: bytes, value: Mapping[str, object]) -> str:
    core = dict(value)
    core.pop("content_sha256", None)
    return _sha256_bytes(domain + _canonical_json(core))


def _sealed(domain: bytes, value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    result["content_sha256"] = _content_sha256(domain, result)
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReacquisitionError(f"invalid SHA-256: {label}")
    return value


def _require_regular_file(path: Path, label: str) -> os.stat_result:
    try:
        observed = path.lstat()
    except OSError as exc:
        raise ReacquisitionError(f"missing staged or retained file: {label}") from exc
    if path.is_symlink() or not stat.S_ISREG(observed.st_mode):
        raise ReacquisitionError(f"file is linked or nonregular: {label}")
    return observed


def _source_bindings() -> dict[str, object]:
    actual_g0: dict[str, str] = {}
    for name, expected in EXPECTED_G0_FILES.items():
        path = G0_ROOT / name
        observed = _sha256_file(path)
        if observed != expected:
            raise ReacquisitionError(f"retained G0 source changed: {name}")
        actual_g0[name] = observed
    dossier = _sha256_file(DOSSIER_PATH)
    if dossier != EXPECTED_DOSSIER_SHA256:
        raise ReacquisitionError("TNO research dossier changed")
    return {
        "g0_files": actual_g0,
        "research_dossier_sha256": dossier,
        "implementation_path": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "implementation_sha256": _sha256_file(Path(__file__).resolve()),
    }


def _validate_artifact_inventory_parity() -> None:
    try:
        ossos = json.loads((G0_ROOT / "ossos_inventory_v1.json").read_bytes())
        des = json.loads((G0_ROOT / "des_y6_inventory_v1.json").read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReacquisitionError("cannot decode retained G0 inventories") from exc
    if type(ossos) is not dict or type(des) is not dict:
        raise ReacquisitionError("retained G0 inventories are not dictionaries")
    ossos_by_id = {
        row["artifact_id"]: row
        for row in ossos.get("complete_catalog_artifacts", [])
        if type(row) is dict and type(row.get("artifact_id")) is str
    }
    try:
        des_rows = {
            "DES_Y6_IOP_TABLE3_ARCHIVE": des["journal_catalog"]["archive"],
            "DES_Y6_CDS_README": des["journal_catalog"]["vizier_mirror"]["readme"],
            "DES_Y6_CDS_CATALOG": des["journal_catalog"]["vizier_mirror"]["catalog"],
            "DES_Y6_ENHANCED_AUTHOR_CATALOG": des["enhanced_author_catalog"]["catalog"],
            "DES_Y6_AUTHOR_LICENSE": des["enhanced_author_catalog"]["license_file"],
        }
    except (KeyError, TypeError) as exc:
        raise ReacquisitionError("retained DES inventory structure differs") from exc
    inventory_rows = {**ossos_by_id, **des_rows}
    expected_ids = {spec.artifact_id for spec in ARTIFACTS}
    if set(inventory_rows).intersection(expected_ids) != expected_ids:
        raise ReacquisitionError("G0 inventories do not contain the artifact roster")
    for spec in ARTIFACTS:
        row = inventory_rows[spec.artifact_id]
        if (
            type(row) is not dict
            or row.get("url") != spec.url
            or row.get("byte_count") != spec.byte_count
            or row.get("sha256") != spec.sha256
        ):
            raise ReacquisitionError(
                f"artifact roster differs from retained G0 inventory: {spec.artifact_id}"
            )
        if spec.record_count is not None and row.get("record_count", spec.record_count) != spec.record_count:
            raise ReacquisitionError(
                f"artifact record count differs from G0 inventory: {spec.artifact_id}"
            )


def build_protocol() -> dict[str, object]:
    _validate_artifact_inventory_parity()
    core: dict[str, object] = {
        "schema": PROTOCOL_SCHEMA,
        "contract_id": "tno-catalog-reacquisition-v1",
        "classification": CLASSIFICATION,
        "claim_ceiling": "INPUT_CUSTODY_ONLY",
        "execution_authorized": False,
        "outcomes_generated": False,
        "evidence_date": "2026-10-08",
        "version": 1,
        "source_bindings": _source_bindings(),
        "artifact_roster": [spec.as_record() for spec in ARTIFACTS],
        "transport_contract": {
            "transport_is_external_to_verifier": True,
            "required_http_behavior": "HTTPS_GET_FOLLOW_REDIRECTS_FAIL_ON_HTTP_ERROR_NO_CONTENT_REENCODING",
            "scientific_authority": "EXACT_RETAINED_BYTES_AND_PINNED_SHA256_NOT_HTTP_HEADERS",
            "staging_filenames_are_exact": True,
        },
        "authorization": {
            "allowed": [
                "download_exact_predeclared_catalog_payloads",
                "verify_byte_count_and_sha256",
                "retain_local_private_copies",
                "publish_create_only_custody_manifest",
                "perform_license_and_semantics_followup_design",
            ],
            "forbidden": [
                "redistribute_ossos_payloads_without_license_resolution",
                "alter_or_normalize_catalog_rows",
                "sample_covariances",
                "integrate_or_classify_orbits",
                "evaluate_survey_selection_or_likelihood",
                "run_gpu_dynamics",
                "make_population_or_planet_nine_claims",
            ],
        },
        "release_policy": {
            "des_canonical_release": "FINAL_REVISION1_814_ROW_JOURNAL_AND_CDS_RELEASE",
            "des_enhanced_catalog_role": "ACQUIRE_FOR_FUTURE_STATE_COVARIANCE_WORK_NOT_YET_SEMANTICALLY_ACTIVATED",
            "ossos_catalog_role": "LOCAL_CUSTODY_ONLY_PENDING_DATA_LICENSE_CLARIFICATION",
            "observed_data_status": "PREVIOUSLY_INSPECTED_NOT_BLIND_HOLDOUT",
        },
        "known_blockers_after_successful_reacquisition": [
            "OSSOS_DATA_REDISTRIBUTION_LICENSE_NOT_RESOLVED",
            "OSSOS_COMPLETE_EIGHT_REGION_SELECTION_FUNCTION_NOT_ACQUIRED",
            "DES_STATE_EPOCH_TIME_SCALE_NOT_LOCKED",
            "DES_SELECTION_AND_ELIGIBILITY_DISCREPANCIES_NOT_RESOLVED",
            "MPC_JPL_CROSSMATCH_QUERY_NOT_YET_PREREGISTERED_OR_ARCHIVED",
            "MASTER_CATALOG_TRANSFORMATION_CONTRACT_NOT_YET_ISSUED",
        ],
        "claims": dict(CLAIMS),
    }
    return _sealed(PROTOCOL_DOMAIN, core)


def validate_protocol(protocol: Mapping[str, object]) -> str:
    if type(protocol) is not dict:
        raise ReacquisitionError("protocol must be an exact dictionary")
    expected = build_protocol()
    if protocol != expected:
        raise ReacquisitionError("protocol differs from the frozen construction")
    declared = _require_sha256(protocol.get("content_sha256"), "protocol")
    if declared != _content_sha256(PROTOCOL_DOMAIN, protocol):
        raise ReacquisitionError("protocol content seal differs")
    return declared


def _read_json(path: Path, label: str) -> dict[str, object]:
    observed = _require_regular_file(path, label)
    if not 1 <= observed.st_size <= 2_000_000:
        raise ReacquisitionError(f"JSON file has invalid size: {label}")
    try:
        raw = path.read_bytes()
        decoded = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReacquisitionError(f"cannot decode JSON: {label}") from exc
    if type(decoded) is not dict or _canonical_json(decoded) != raw:
        raise ReacquisitionError(f"JSON is not canonical: {label}")
    return decoded


def _write_create_only(path: Path, payload: bytes) -> None:
    if type(payload) is not bytes or not payload:
        raise ReacquisitionError("create-only payload must be nonempty bytes")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise ReacquisitionError(f"refusing to overwrite: {path}") from exc
    try:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            count = os.write(descriptor, view[offset:])
            if count <= 0:
                raise ReacquisitionError(f"short write: {path}")
            offset += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prepare(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, object]:
    output = _lexical_absolute(Path(output_root))
    protocol = build_protocol()
    payload = _canonical_json(protocol)
    if output.exists():
        if output.is_symlink() or not output.is_dir():
            raise ReacquisitionError("output root is linked or non-directory")
        actual_names = {entry.name for entry in output.iterdir()}
        if not actual_names.issubset({"PROTOCOL.json", "raw", "CUSTODY.json"}):
            raise ReacquisitionError("output root has unexpected entries")
        existing = output / "PROTOCOL.json"
        if not existing.exists() or existing.read_bytes() != payload:
            raise ReacquisitionError("existing output has a different protocol")
        validate_protocol(_read_json(existing, "protocol"))
        return protocol
    output.mkdir(mode=0o700, parents=False)
    _write_create_only(output / "PROTOCOL.json", payload)
    return protocol


def _verify_des_archive(path: Path) -> dict[str, object]:
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            members = archive.getmembers()
            regular = [member for member in members if member.isfile()]
            if len(regular) != 1 or regular[0].name != "y6_paper_table.fits":
                raise ReacquisitionError("DES journal archive member roster differs")
            member = archive.extractfile(regular[0])
            if member is None:
                raise ReacquisitionError("DES FITS member is absent")
            payload = member.read()
    except (OSError, tarfile.TarError) as exc:
        raise ReacquisitionError("DES journal archive is invalid") from exc
    if len(payload) != 388_800:
        raise ReacquisitionError("DES FITS member byte count differs")
    digest = _sha256_bytes(payload)
    if digest != "15e4493e3f90232b06714d9e7b964c275c749627f25750386fee313d12e579ae":
        raise ReacquisitionError("DES FITS member SHA-256 differs")
    return {
        "member_path": "y6_paper_table.fits",
        "member_byte_count": len(payload),
        "member_sha256": digest,
        "record_count": 814,
        "column_count": 55,
    }


def verify_staging(
    staging_root: Path,
    specs: Iterable[ArtifactSpec] = ARTIFACTS,
) -> list[dict[str, object]]:
    staging = _lexical_absolute(Path(staging_root))
    if staging.is_symlink() or not staging.is_dir():
        raise ReacquisitionError("staging root is linked or non-directory")
    spec_list = tuple(specs)
    expected_names = {spec.staging_name for spec in spec_list}
    if len(expected_names) != len(spec_list):
        raise ReacquisitionError("artifact staging names are not unique")
    actual_names = {entry.name for entry in staging.iterdir()}
    if actual_names != expected_names:
        raise ReacquisitionError("staging roster differs from the protocol")
    records: list[dict[str, object]] = []
    for spec in spec_list:
        path = staging / spec.staging_name
        observed = _require_regular_file(path, spec.artifact_id)
        if observed.st_size != spec.byte_count:
            raise ReacquisitionError(f"byte count differs: {spec.artifact_id}")
        digest = _sha256_file(path)
        if digest != spec.sha256:
            raise ReacquisitionError(f"SHA-256 differs: {spec.artifact_id}")
        record = spec.as_record()
        record.update(
            {
                "verification_status": "EXACT_BYTE_MATCH",
                "observed_byte_count": observed.st_size,
                "observed_sha256": digest,
                "retained_relative_path": f"raw/{spec.retained_name}",
            }
        )
        if spec.artifact_id == "DES_Y6_IOP_TABLE3_ARCHIVE":
            record["archive_member_verification"] = _verify_des_archive(path)
        records.append(record)
    return records


def _publish_raw_file(source: Path, destination: Path, spec: ArtifactSpec) -> None:
    if destination.exists():
        observed = _require_regular_file(destination, spec.artifact_id)
        if observed.st_size != spec.byte_count or _sha256_file(destination) != spec.sha256:
            raise ReacquisitionError(f"retained file collision: {spec.artifact_id}")
        return
    _write_create_only(destination, source.read_bytes())
    observed = _require_regular_file(destination, spec.artifact_id)
    if observed.st_size != spec.byte_count or _sha256_file(destination) != spec.sha256:
        raise ReacquisitionError(f"retained publication changed: {spec.artifact_id}")


def _custody_core(
    records: list[dict[str, object]],
    protocol: Mapping[str, object],
    protocol_file_sha256: str,
    verified_at_utc: str,
) -> dict[str, object]:
    return {
        "schema": CUSTODY_SCHEMA,
        "contract_id": "tno-catalog-reacquisition-v1",
        "classification": CLASSIFICATION,
        "claim_ceiling": "INPUT_CUSTODY_ONLY",
        "execution_authorized": False,
        "outcomes_generated": False,
        "verified_and_published_at_utc": verified_at_utc,
        "protocol": {
            "relative_path": "PROTOCOL.json",
            "file_sha256": protocol_file_sha256,
            "content_sha256": protocol["content_sha256"],
        },
        "artifacts": records,
        "artifact_roster_sha256": _sha256_bytes(_canonical_json(records)),
        "artifact_count": len(records),
        "total_retained_bytes": sum(spec.byte_count for spec in ARTIFACTS),
        "status": "EXACT_CATALOG_PAYLOAD_BYTES_REACQUIRED_LOCAL_CUSTODY_ONLY",
        "readiness": {
            "catalog_payload_custody_established": True,
            "catalog_semantics_validated": False,
            "master_catalog_constructed": False,
            "des_epoch_time_scale_locked": False,
            "covariance_sampling_authorized": False,
            "dynamical_classifier_authorized": False,
            "survey_selection_model_ready": False,
            "science_execution_authorized": False,
        },
        "claims": dict(CLAIMS),
        "remaining_blockers": protocol["known_blockers_after_successful_reacquisition"],
        "nonclaim": (
            "This receipt proves local custody of nine exact public catalog payloads only. "
            "It is not a master catalog, classifier, survey model, likelihood, dynamics run, "
            "or Planet Nine result."
        ),
    }


def publish(
    staging_root: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, object]:
    output = _lexical_absolute(Path(output_root))
    protocol = prepare(output)
    protocol_sha = validate_protocol(protocol)
    records = verify_staging(staging_root, ARTIFACTS)
    manifest_path = output / "CUSTODY.json"
    if manifest_path.exists():
        raise ReacquisitionError("custody manifest already exists; use validate")
    raw = output / "raw"
    if raw.exists():
        if raw.is_symlink() or not raw.is_dir():
            raise ReacquisitionError("retained raw path is linked or non-directory")
        actual_names = {entry.name for entry in raw.iterdir()}
        expected_names = {spec.retained_name for spec in ARTIFACTS}
        if not actual_names.issubset(expected_names):
            raise ReacquisitionError("retained raw directory has unexpected entries")
    else:
        raw.mkdir(mode=0o700)
    for spec in ARTIFACTS:
        _publish_raw_file(
            _lexical_absolute(Path(staging_root)) / spec.staging_name,
            raw / spec.retained_name,
            spec,
        )
    if protocol_sha != protocol["content_sha256"]:
        raise ReacquisitionError("validated protocol seal changed")
    core = _custody_core(
        records,
        protocol,
        _sha256_file(output / "PROTOCOL.json"),
        _utc_now(),
    )
    manifest = _sealed(CUSTODY_DOMAIN, core)
    _write_create_only(manifest_path, _canonical_json(manifest))
    validate_output(output)
    return manifest


def validate_output(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, object]:
    output = _lexical_absolute(Path(output_root))
    if output.is_symlink() or not output.is_dir():
        raise ReacquisitionError("output root is linked or non-directory")
    if {entry.name for entry in output.iterdir()} != {
        "PROTOCOL.json",
        "CUSTODY.json",
        "raw",
    }:
        raise ReacquisitionError("completed output root roster differs")
    protocol = _read_json(output / "PROTOCOL.json", "protocol")
    protocol_sha = validate_protocol(protocol)
    manifest = _read_json(output / "CUSTODY.json", "custody manifest")
    declared = _require_sha256(manifest.get("content_sha256"), "custody manifest")
    if declared != _content_sha256(CUSTODY_DOMAIN, manifest):
        raise ReacquisitionError("custody manifest content seal differs")
    raw = output / "raw"
    if raw.is_symlink() or not raw.is_dir():
        raise ReacquisitionError("retained raw directory differs")
    expected_names = {spec.retained_name for spec in ARTIFACTS}
    if {entry.name for entry in raw.iterdir()} != expected_names:
        raise ReacquisitionError("retained raw roster differs")
    expected_records: list[dict[str, object]] = []
    for spec in ARTIFACTS:
        path = raw / spec.retained_name
        observed = _require_regular_file(path, spec.artifact_id)
        if observed.st_size != spec.byte_count or _sha256_file(path) != spec.sha256:
            raise ReacquisitionError(f"retained bytes differ: {spec.artifact_id}")
        record = spec.as_record()
        record.update(
            {
                "verification_status": "EXACT_BYTE_MATCH",
                "observed_byte_count": observed.st_size,
                "observed_sha256": spec.sha256,
                "retained_relative_path": f"raw/{spec.retained_name}",
            }
        )
        if spec.artifact_id == "DES_Y6_IOP_TABLE3_ARCHIVE":
            record["archive_member_verification"] = _verify_des_archive(path)
        expected_records.append(record)
    verified_at = manifest.get("verified_and_published_at_utc")
    if type(verified_at) is not str or not verified_at.endswith("Z"):
        raise ReacquisitionError("custody UTC timestamp differs")
    try:
        parsed = datetime.fromisoformat(verified_at[:-1] + "+00:00")
    except ValueError as exc:
        raise ReacquisitionError("custody UTC timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ReacquisitionError("custody timestamp is not UTC")
    expected_manifest = _sealed(
        CUSTODY_DOMAIN,
        _custody_core(
            expected_records,
            protocol,
            _sha256_file(output / "PROTOCOL.json"),
            verified_at,
        ),
    )
    if manifest != expected_manifest:
        raise ReacquisitionError("custody manifest differs from full reconstruction")
    return manifest


def _commands() -> str:
    return "\n".join(
        f"curl -fL {json.dumps(spec.url)} -o {json.dumps(spec.staging_name)}"
        for spec in ARTIFACTS
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    commands_parser = subparsers.add_parser("commands")
    commands_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    publish_parser = subparsers.add_parser("publish")
    publish_parser.add_argument("--staging-root", type=Path, required=True)
    publish_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output_root)
        print(result["content_sha256"])
    elif args.command == "commands":
        prepare(args.output_root)
        print(_commands())
    elif args.command == "publish":
        result = publish(args.staging_root, args.output_root)
        print(result["content_sha256"])
    elif args.command == "validate":
        result = validate_output(args.output_root)
        print(result["content_sha256"])
    else:  # pragma: no cover
        raise ReacquisitionError("unknown command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
