"""Fail-closed telescope-selection validation for the JX-O1 milestone.

The module does not implement an astronomical survey simulator.  It prepares
deterministic intrinsic populations for an external, pinned OSSOS
SurveySimulator, normalizes that program's tracked-object output, and evaluates
predeclared calibration and power gates.  A small analytic selector is included
only as a non-final software pilot; it can never satisfy the external-simulator
gate.
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import hmac
import json
import math
import os
import statistics
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .provenance import canonical_json, sha256_data, sha256_file


CONTRACT_SCHEMA = "jx-survey-selection-contract/v1"
POOL_SCHEMA = "jx-survey-selection-pool/v1"
RESULT_SCHEMA = "jx-survey-selection-result/v1"
DETECTION_COLUMNS = (
    "object_id",
    "model_id",
    "seed_block",
    "a_AU",
    "q_AU",
    "i_deg",
    "H_r",
    "r_AU",
    "m_r",
)
MODEL_IDS = ("correct", "wrong")

NONCLAIMS = (
    "JX-O1 does not detect or exclude Planet X",
    "JX-O1 does not estimate a source mass, orbit, distance, or sky direction",
    "the analytic pilot is not an astronomical survey simulation",
    "passing validates only the locked telescope-selection inference workflow",
)


class SurveySelectionVerdict(str, Enum):
    PASSED = "PASSED"
    BLOCKED = "BLOCKED"
    INVALID = "INVALID"


class SurveySelectionError(ValueError):
    """Stable, machine-readable input or integrity failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _fail(code: str, message: str) -> None:
    raise SurveySelectionError(code, message)


def _reject_json_constant(value: str) -> None:
    _fail("nonfinite_json_constant", f"JSON constant {value!r} is forbidden")


def _load_json(path: str | Path, code: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, RecursionError) as exc:
        _fail(code, f"cannot load valid JSON from {source}: {exc}")
    if not isinstance(value, dict):
        _fail(code, f"top-level JSON in {source} must be an object")
    return value


def _atomic_json(path: str | Path, data: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        _fail("output_exists", f"refusing to overwrite immutable output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError, RecursionError) as exc:
        _fail("noncanonical_output", f"cannot serialize finite JSON: {exc}")
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _check_keys(value: Mapping[str, Any], required: set[str], context: str) -> None:
    missing = required - value.keys()
    extra = value.keys() - required
    if missing:
        _fail("missing_field", f"{context} missing fields: {sorted(missing)}")
    if extra:
        _fail("unknown_field", f"{context} has unknown fields: {sorted(extra)}")


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail("invalid_object", f"{context} must be an object")
    return value


def _finite(value: Any, context: str) -> float:
    if isinstance(value, bool):
        _fail("invalid_number", f"{context} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        _fail("invalid_number", f"{context} must be a finite number")
    if not math.isfinite(number):
        _fail("nonfinite_value", f"{context} must be finite")
    return number


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail("invalid_integer", f"{context} must be a positive integer")
    return value


def _nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail("invalid_integer", f"{context} must be a nonnegative integer")
    return value


def _probability(value: Any, context: str, *, allow_zero: bool = True) -> float:
    number = _finite(value, context)
    lower_ok = number >= 0.0 if allow_zero else number > 0.0
    if not lower_ok or number > 1.0:
        _fail("invalid_probability", f"{context} must be in the required unit interval")
    return number


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("invalid_string", f"{context} must be a nonempty string")
    return value


def _sha256(value: Any, context: str) -> str:
    text = _nonempty_string(value, context)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        _fail("invalid_sha256", f"{context} must be a lowercase SHA-256 digest")
    return text


def validate_survey_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a detached canonical copy of a JX-O1 contract."""

    top = {
        "schema",
        "experiment_id",
        "registration_status",
        "milestone",
        "external_simulator",
        "population",
        "statistics",
        "gates",
        "execution",
        "scientific_scope",
        "limitations",
        "nonclaim",
        "allowed_verdicts",
        "locked_files",
    }
    _check_keys(contract, top, "contract")
    if contract["schema"] != CONTRACT_SCHEMA:
        _fail("schema_mismatch", f"expected contract schema {CONTRACT_SCHEMA!r}")
    for name in ("experiment_id", "registration_status", "milestone", "scientific_scope", "nonclaim"):
        _nonempty_string(contract[name], name)
    if contract["registration_status"] != "PRELOCKED_BEFORE_ANY_JX_O1_OUTCOMES":
        _fail("not_prelocked", "JX-O1 contract is not outcome-prelocked")
    if not isinstance(contract["limitations"], list) or not contract["limitations"]:
        _fail("invalid_limitations", "limitations must be a nonempty list")
    for index, item in enumerate(contract["limitations"]):
        _nonempty_string(item, f"limitations[{index}]")

    locked_files = _mapping(contract["locked_files"], "locked_files")
    if not locked_files:
        _fail("missing_locked_files", "at least one JX executable or test file must be locked")
    for name, item_raw in locked_files.items():
        _nonempty_string(name, "locked file name")
        item = _mapping(item_raw, f"locked_files.{name}")
        _check_keys(item, {"path", "sha256"}, f"locked_files.{name}")
        path = _nonempty_string(item["path"], f"locked_files.{name}.path")
        if Path(path).is_absolute():
            _fail("absolute_locked_path", f"locked file path must be relative: {path!r}")
        _sha256(item["sha256"], f"locked_files.{name}.sha256")

    external = _mapping(contract["external_simulator"], "external_simulator")
    _check_keys(
        external,
        {
            "repository_url",
            "commit",
            "license",
            "execution_mode",
            "source_subdirectory",
            "characterization_path",
            "characterization_scope",
            "required_files",
        },
        "external_simulator",
    )
    for name in (
        "repository_url",
        "license",
        "execution_mode",
        "source_subdirectory",
        "characterization_path",
        "characterization_scope",
    ):
        _nonempty_string(external[name], f"external_simulator.{name}")
    commit = _nonempty_string(external["commit"], "external_simulator.commit")
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        _fail("invalid_commit", "external simulator commit must be a full lowercase Git SHA")
    required_files = _mapping(external["required_files"], "external_simulator.required_files")
    if not required_files:
        _fail("missing_external_files", "at least one external file hash must be locked")
    for path, digest in required_files.items():
        _nonempty_string(path, "external file path")
        if Path(path).is_absolute() or ".." in Path(path).parts:
            _fail("unsafe_external_path", f"unsafe external path: {path!r}")
        _sha256(digest, f"external file {path}")

    population = _mapping(contract["population"], "population")
    _check_keys(
        population,
        {
            "epoch_jd",
            "seed_key",
            "seed_blocks",
            "minimum_intrinsic_draws_per_model",
            "minimum_intrinsic_draws_per_seed_block",
            "minimum_tracked_detections_per_model",
            "minimum_tracked_detections_per_seed_block",
            "catalog_size",
            "correct_model",
            "wrong_model",
            "shared",
        },
        "population",
    )
    _finite(population["epoch_jd"], "population.epoch_jd")
    _nonempty_string(population["seed_key"], "population.seed_key")
    seed_blocks = _positive_int(population["seed_blocks"], "population.seed_blocks")
    if seed_blocks < 2:
        _fail("insufficient_seed_blocks", "at least two independent seed blocks are required")
    minimum_intrinsic = _positive_int(
        population["minimum_intrinsic_draws_per_model"], "minimum intrinsic draws"
    )
    minimum_intrinsic_block = _positive_int(
        population["minimum_intrinsic_draws_per_seed_block"],
        "minimum intrinsic draws per seed block",
    )
    minimum_detections = _positive_int(
        population["minimum_tracked_detections_per_model"], "minimum tracked detections"
    )
    minimum_detections_block = _positive_int(
        population["minimum_tracked_detections_per_seed_block"],
        "minimum tracked detections per seed block",
    )
    if minimum_intrinsic != seed_blocks * minimum_intrinsic_block:
        _fail("unbalanced_intrinsic_target", "model and per-block intrinsic targets must agree")
    if minimum_detections != seed_blocks * minimum_detections_block:
        _fail("unbalanced_detection_target", "model and per-block detection targets must agree")
    catalog_size = _positive_int(population["catalog_size"], "population.catalog_size")
    if minimum_detections < catalog_size:
        _fail("insufficient_detection_target", "detection target must cover one catalog")

    correct = _mapping(population["correct_model"], "population.correct_model")
    _check_keys(correct, {"q_distribution", "q_min_AU", "q_max_AU"}, "correct_model")
    if correct["q_distribution"] != "uniform":
        _fail("invalid_correct_model", "correct-model q distribution must be uniform")
    q_min = _finite(correct["q_min_AU"], "correct q minimum")
    q_max = _finite(correct["q_max_AU"], "correct q maximum")
    if q_min <= 0.0 or q_max <= q_min:
        _fail("invalid_q_range", "correct-model q range is invalid")

    wrong = _mapping(population["wrong_model"], "population.wrong_model")
    _check_keys(
        wrong,
        {
            "q_distribution",
            "beta_alpha",
            "beta_beta",
            "q_min_AU",
            "q_max_AU",
            "interpretation",
        },
        "wrong_model",
    )
    if wrong["q_distribution"] != "beta" or _finite(wrong["beta_beta"], "beta_beta") != 1.0:
        _fail("unsupported_wrong_model", "the locked analytic inverse supports only Beta(alpha, 1)")
    if _finite(wrong["beta_alpha"], "beta_alpha") <= 0.0:
        _fail("invalid_beta_alpha", "beta_alpha must be positive")
    if _finite(wrong["q_min_AU"], "wrong q minimum") != q_min or _finite(
        wrong["q_max_AU"], "wrong q maximum"
    ) != q_max:
        _fail("unmatched_q_support", "correct and wrong q models must use identical support")
    _nonempty_string(wrong["interpretation"], "wrong_model.interpretation")

    shared = _mapping(population["shared"], "population.shared")
    _check_keys(
        shared,
        {
            "a_distribution",
            "a_power",
            "a_min_AU",
            "a_max_AU",
            "inclination_distribution",
            "inclination_sigma_deg",
            "inclination_min_deg",
            "inclination_max_deg",
            "angular_distribution",
            "H_distribution",
            "H_min",
            "H_break",
            "H_max",
            "alpha_bright",
            "alpha_faint",
            "contrast",
            "colors_r_reference",
        },
        "population.shared",
    )
    if shared["a_distribution"] != "power_law" or _finite(shared["a_power"], "a_power") != -1.5:
        _fail("unsupported_a_distribution", "JX-O1 locks dN/da proportional to a^-3/2")
    a_min = _finite(shared["a_min_AU"], "a minimum")
    a_max = _finite(shared["a_max_AU"], "a maximum")
    if a_min <= q_max or a_max <= a_min:
        _fail("invalid_a_range", "a support must exceed q support and have positive width")
    if shared["inclination_distribution"] != "truncated_half_normal":
        _fail("unsupported_inclination", "inclination must be a truncated half-normal")
    sigma = _finite(shared["inclination_sigma_deg"], "inclination sigma")
    i_min = _finite(shared["inclination_min_deg"], "inclination minimum")
    i_max = _finite(shared["inclination_max_deg"], "inclination maximum")
    if sigma <= 0.0 or i_min < 0.0 or i_max <= i_min or i_max > 180.0:
        _fail("invalid_inclination", "inclination distribution parameters are invalid")
    if shared["angular_distribution"] != "independent_uniform_0_360_deg":
        _fail("unsupported_angles", "all three angular elements must be uniform")
    if shared["H_distribution"] != "lawler_2018_divot":
        _fail("unsupported_H_distribution", "H distribution must be the locked Lawler divot")
    h_min = _finite(shared["H_min"], "H minimum")
    h_break = _finite(shared["H_break"], "H break")
    h_max = _finite(shared["H_max"], "H maximum")
    if not h_min < h_break < h_max:
        _fail("invalid_H_range", "H_min < H_break < H_max is required")
    for name in ("alpha_bright", "alpha_faint", "contrast"):
        if _finite(shared[name], name) <= 0.0:
            _fail("invalid_H_parameter", f"{name} must be positive")
    colors = shared["colors_r_reference"]
    if not isinstance(colors, list) or len(colors) != 10:
        _fail("invalid_colors", "colors_r_reference must contain ten values")
    for index, value in enumerate(colors):
        _finite(value, f"colors_r_reference[{index}]")

    stats = _mapping(contract["statistics"], "statistics")
    _check_keys(
        stats,
        {
            "variables",
            "primary_pit_variable",
            "mock_catalogs",
            "bootstrap_reference_catalogs",
            "seed_stability_catalogs",
            "alpha",
            "zeta_expected_mean",
            "zeta_expected_sd",
            "resampling_seed",
        },
        "statistics",
    )
    variables = stats["variables"]
    if not isinstance(variables, list) or variables != [
        "a_AU",
        "q_AU",
        "i_deg",
        "H_r",
        "r_AU",
        "m_r",
    ]:
        _fail("invalid_variables", "JX-O1 requires the six predeclared AD variables")
    if stats["primary_pit_variable"] != "q_AU":
        _fail("invalid_primary_variable", "JX-O1 primary PIT variable must be q_AU")
    for name in ("mock_catalogs", "bootstrap_reference_catalogs", "seed_stability_catalogs"):
        _positive_int(stats[name], f"statistics.{name}")
    _probability(stats["alpha"], "statistics.alpha", allow_zero=False)
    _finite(stats["zeta_expected_mean"], "zeta expected mean")
    if _finite(stats["zeta_expected_sd"], "zeta expected sd") <= 0.0:
        _fail("invalid_zeta_sd", "zeta expected sd must be positive")
    _nonempty_string(stats["resampling_seed"], "statistics.resampling_seed")

    gates = _mapping(contract["gates"], "gates")
    _check_keys(
        gates,
        {
            "correct_model_false_rejection_rate_min",
            "correct_model_false_rejection_rate_max",
            "zeta_mean_absolute_tolerance",
            "zeta_sd_absolute_tolerance",
            "wrong_model_rejection_power_min",
            "seed_block_verdict_stability_required",
            "adapter_identity_required",
            "exact_replay_required",
            "max_missing_records",
            "max_nonfinite_records",
        },
        "gates",
    )
    fr_min = _probability(gates["correct_model_false_rejection_rate_min"], "false rejection min")
    fr_max = _probability(gates["correct_model_false_rejection_rate_max"], "false rejection max")
    if fr_max < fr_min:
        _fail("invalid_false_rejection_range", "false rejection maximum is below minimum")
    for name in ("zeta_mean_absolute_tolerance", "zeta_sd_absolute_tolerance"):
        if _finite(gates[name], name) < 0.0:
            _fail("invalid_tolerance", f"{name} must be nonnegative")
    _probability(gates["wrong_model_rejection_power_min"], "wrong-model power minimum")
    for name in (
        "seed_block_verdict_stability_required",
        "adapter_identity_required",
        "exact_replay_required",
    ):
        if not isinstance(gates[name], bool):
            _fail("invalid_boolean", f"gates.{name} must be boolean")
    _nonnegative_int(gates["max_missing_records"], "max_missing_records")
    _nonnegative_int(gates["max_nonfinite_records"], "max_nonfinite_records")

    execution = _mapping(contract["execution"], "execution")
    _check_keys(
        execution,
        {
            "checkpoint_restart_required",
            "immutable_result",
            "pilot_seeds_excluded_from_final",
            "official_backend_name",
            "pilot_backend_name",
        },
        "execution",
    )
    for name in ("checkpoint_restart_required", "immutable_result", "pilot_seeds_excluded_from_final"):
        if not isinstance(execution[name], bool):
            _fail("invalid_boolean", f"execution.{name} must be boolean")
    for name in ("official_backend_name", "pilot_backend_name"):
        _nonempty_string(execution[name], f"execution.{name}")

    allowed = _mapping(contract["allowed_verdicts"], "allowed_verdicts")
    if set(allowed) != {item.value for item in SurveySelectionVerdict}:
        _fail("invalid_allowed_verdicts", "allowed_verdicts must define PASSED, BLOCKED, and INVALID")
    for name, description in allowed.items():
        _nonempty_string(description, f"allowed_verdicts.{name}")
    return json.loads(canonical_json(contract).decode("utf-8"))


def load_survey_contract(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    contract = validate_survey_contract(_load_json(source, "contract_read_error"))
    for name, item in contract["locked_files"].items():
        locked_path = (source.parent / item["path"]).resolve()
        if not locked_path.is_file():
            _fail("locked_file_missing", f"locked file {name!r} is unavailable: {locked_path}")
        observed = sha256_file(locked_path)
        if observed != item["sha256"]:
            _fail(
                "locked_file_hash_mismatch",
                f"locked file {name!r} hash mismatch: expected {item['sha256']}, observed {observed}",
            )
    return contract


def _counter_uniform(seed: str, *parts: object) -> float:
    message = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hmac.new(seed.encode("utf-8"), message, hashlib.sha256).digest()
    integer = int.from_bytes(digest[:8], "big")
    return (integer + 0.5) / float(1 << 64)


def _sample_index(seed: str, size: int, *parts: object) -> int:
    if size <= 0:
        _fail("empty_sampling_pool", "cannot sample from an empty pool")
    limit = (1 << 64) - ((1 << 64) % size)
    attempt = 0
    while True:
        message = "\x1f".join(str(part) for part in (*parts, attempt)).encode("utf-8")
        digest = hmac.new(seed.encode("utf-8"), message, hashlib.sha256).digest()
        integer = int.from_bytes(digest[:8], "big")
        if integer < limit:
            return integer % size
        attempt += 1


def _sample_power_law_minus_three_halves(u: float, minimum: float, maximum: float) -> float:
    lo = minimum ** -0.5
    hi = maximum ** -0.5
    return (lo + u * (hi - lo)) ** -2.0


def _sample_truncated_half_normal(
    seed: str,
    namespace: str,
    block: int,
    index: int,
    sigma: float,
    minimum: float,
    maximum: float,
) -> float:
    for attempt in range(1000):
        u1 = _counter_uniform(seed, namespace, block, index, "inclination-u1", attempt)
        u2 = _counter_uniform(seed, namespace, block, index, "inclination-u2", attempt)
        value = abs(math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)) * sigma
        if minimum <= value <= maximum:
            return value
    _fail("inclination_rejection_failure", "truncated inclination sampler did not converge")


def _sample_divot_H(u: float, shared: Mapping[str, Any]) -> float:
    h_min = float(shared["H_min"])
    h_break = float(shared["H_break"])
    h_max = float(shared["H_max"])
    alpha_b = float(shared["alpha_bright"])
    alpha_f = float(shared["alpha_faint"])
    contrast = float(shared["contrast"])
    log10 = math.log(10.0)
    bright_low = 10.0 ** (alpha_b * (h_min - h_break))
    faint_high = 10.0 ** (alpha_f * (h_max - h_break))
    bright_weight = (1.0 - bright_low) / (alpha_b * log10)
    faint_weight = (faint_high - 1.0) / (contrast * alpha_f * log10)
    split = bright_weight / (bright_weight + faint_weight)
    if u < split:
        fraction = u / split
        value = bright_low + fraction * (1.0 - bright_low)
        return h_break + math.log10(value) / alpha_b
    fraction = (u - split) / (1.0 - split)
    value = 1.0 + fraction * (faint_high - 1.0)
    return h_break + math.log10(value) / alpha_f


def generate_intrinsic_population(
    contract: Mapping[str, Any],
    model_id: str,
    seed_block: int,
    count: int,
    *,
    namespace: str = "final",
    start_index: int = 0,
) -> list[dict[str, Any]]:
    """Generate a paired, byte-reproducible intrinsic population block."""

    locked = validate_survey_contract(contract)
    if model_id not in MODEL_IDS:
        _fail("invalid_model_id", f"model_id must be one of {MODEL_IDS}")
    block_count = int(locked["population"]["seed_blocks"])
    if isinstance(seed_block, bool) or not isinstance(seed_block, int) or not 0 <= seed_block < block_count:
        _fail("invalid_seed_block", f"seed_block must be in [0, {block_count})")
    count = _positive_int(count, "population count")
    start_index = _nonnegative_int(start_index, "start_index")
    namespace = _nonempty_string(namespace, "population namespace")
    population = locked["population"]
    shared = population["shared"]
    q_model = population[f"{model_id}_model"]
    seed = population["seed_key"]
    rows: list[dict[str, Any]] = []
    for local_index in range(count):
        index = start_index + local_index
        q_u = _counter_uniform(seed, namespace, seed_block, index, "q")
        if model_id == "correct":
            q_fraction = q_u
            prefix = "c"
        else:
            q_fraction = q_u ** (1.0 / float(q_model["beta_alpha"]))
            prefix = "w"
        q = float(q_model["q_min_AU"]) + (
            float(q_model["q_max_AU"]) - float(q_model["q_min_AU"])
        ) * q_fraction
        a = _sample_power_law_minus_three_halves(
            _counter_uniform(seed, namespace, seed_block, index, "a"),
            float(shared["a_min_AU"]),
            float(shared["a_max_AU"]),
        )
        inclination = _sample_truncated_half_normal(
            seed,
            namespace,
            seed_block,
            index,
            float(shared["inclination_sigma_deg"]),
            float(shared["inclination_min_deg"]),
            float(shared["inclination_max_deg"]),
        )
        row = {
            "object_id": f"{prefix}{seed_block:02d}{index:08d}",
            "model_id": model_id,
            "seed_block": seed_block,
            "a_AU": a,
            "e": 1.0 - q / a,
            "q_AU": q,
            "i_deg": inclination,
            "node_deg": 360.0 * _counter_uniform(seed, namespace, seed_block, index, "node"),
            "peri_deg": 360.0 * _counter_uniform(seed, namespace, seed_block, index, "peri"),
            "mean_anomaly_deg": 360.0
            * _counter_uniform(seed, namespace, seed_block, index, "mean-anomaly"),
            "H_r": _sample_divot_H(
                _counter_uniform(seed, namespace, seed_block, index, "H-r"), shared
            ),
        }
        if not 0.0 <= row["e"] < 1.0:
            _fail("invalid_generated_eccentricity", "generated orbit is not bound")
        rows.append(row)
    return rows


def write_ossos_model_file(
    path: str | Path,
    contract: Mapping[str, Any],
    model_id: str,
    seed_block: int,
    count: int,
    *,
    namespace: str = "final",
    start_index: int = 0,
) -> dict[str, Any]:
    """Write the lookup-table input expected by OSSOS ReadModelFromFile."""

    target = Path(path)
    if target.exists():
        _fail("output_exists", f"refusing to overwrite model file: {target}")
    locked = validate_survey_contract(contract)
    rows = generate_intrinsic_population(
        locked,
        model_id,
        seed_block,
        count,
        namespace=namespace,
        start_index=start_index,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    colors = " ".join(format(float(value), ".17g") for value in locked["population"]["shared"]["colors_r_reference"])
    lines = [
        f"# Epoch of elements: JD = {float(locked['population']['epoch_jd']):.8f}",
        f"# Colors = {colors}",
        "# a e i node peri M H object_id",
    ]
    for row in rows:
        lines.append(
            " ".join(
                (
                    format(row["a_AU"], ".17g"),
                    format(row["e"], ".17g"),
                    format(row["i_deg"], ".17g"),
                    format(row["node_deg"], ".17g"),
                    format(row["peri_deg"], ".17g"),
                    format(row["mean_anomaly_deg"], ".17g"),
                    format(row["H_r"], ".17g"),
                    row["object_id"],
                )
            )
        )
    encoded = "\n".join(lines) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return {
        "model_id": model_id,
        "seed_block": seed_block,
        "intrinsic_draws": count,
        "start_index": start_index,
        "path": str(target),
        "sha256": sha256_file(target),
    }


def _validate_detection(row: Mapping[str, Any], context: str) -> dict[str, Any]:
    _check_keys(row, set(DETECTION_COLUMNS), context)
    object_id = _nonempty_string(row["object_id"], f"{context}.object_id")
    if any(char.isspace() for char in object_id):
        _fail("invalid_object_id", f"{context}.object_id may not contain whitespace")
    model_id = row["model_id"]
    if model_id not in MODEL_IDS:
        _fail("invalid_model_id", f"{context}.model_id is invalid")
    seed_block = row["seed_block"]
    if isinstance(seed_block, str):
        try:
            seed_block = int(seed_block)
        except ValueError:
            _fail("invalid_seed_block", f"{context}.seed_block is not an integer")
    seed_block = _nonnegative_int(seed_block, f"{context}.seed_block")
    numeric = {name: _finite(row[name], f"{context}.{name}") for name in DETECTION_COLUMNS[3:]}
    if numeric["a_AU"] <= 0.0 or numeric["q_AU"] <= 0.0 or numeric["q_AU"] > numeric["a_AU"]:
        _fail("invalid_orbit", f"{context} has invalid a or q")
    if not 0.0 <= numeric["i_deg"] <= 180.0 or numeric["r_AU"] <= 0.0:
        _fail("invalid_detection_geometry", f"{context} has invalid inclination or distance")
    return {"object_id": object_id, "model_id": model_id, "seed_block": seed_block, **numeric}


def write_detection_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    target = Path(path)
    if target.exists():
        _fail("output_exists", f"refusing to overwrite detection file: {target}")
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        item = _validate_detection(row, f"detection[{index}]")
        if item["object_id"] in seen:
            _fail("duplicate_object_id", f"duplicate detection ID: {item['object_id']}")
        seen.add(item["object_id"])
        validated.append(item)
    validated.sort(key=lambda item: item["object_id"])
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=DETECTION_COLUMNS, lineterminator="\n")
            writer.writeheader()
            for row in validated:
                writer.writerow(
                    {
                        name: format(row[name], ".17g") if name in DETECTION_COLUMNS[3:] else row[name]
                        for name in DETECTION_COLUMNS
                    }
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return {
        "path": str(target),
        "count": len(validated),
        "sha256": sha256_file(target),
        "semantic_sha256": sha256_data(validated),
    }


def load_detection_csv(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != DETECTION_COLUMNS:
                _fail("invalid_detection_header", f"unexpected header in {source}")
            rows = [_validate_detection(row, f"{source}:row[{index}]") for index, row in enumerate(reader)]
    except OSError as exc:
        _fail("detection_read_error", f"cannot read {source}: {exc}")
    seen: set[str] = set()
    for row in rows:
        if row["object_id"] in seen:
            _fail("duplicate_object_id", f"duplicate detection ID in {source}: {row['object_id']}")
        seen.add(row["object_id"])
    if rows != sorted(rows, key=lambda item: item["object_id"]):
        _fail("noncanonical_detection_order", f"detections in {source} are not object-ID sorted")
    return rows


def parse_ossos_tracked_file(
    path: str | Path,
    model_id: str,
    seed_block: int,
) -> list[dict[str, Any]]:
    """Normalize the documented F95 tracked-output columns."""

    if model_id not in MODEL_IDS:
        _fail("invalid_model_id", f"model_id must be one of {MODEL_IDS}")
    seed_block = _nonnegative_int(seed_block, "seed_block")
    source = Path(path)
    rows: list[dict[str, Any]] = []
    try:
        for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 20:
                _fail("invalid_ossos_row", f"{source}:{line_number} has fewer than 20 columns")
            try:
                flag = int(fields[11])
            except ValueError:
                _fail("invalid_ossos_flag", f"{source}:{line_number} has an invalid detection flag")
            if flag <= 0 or flag % 2 != 0:
                _fail("untracked_ossos_row", f"{source}:{line_number} is not a tracked detection")
            row = {
                "object_id": fields[19],
                "model_id": model_id,
                "seed_block": seed_block,
                "a_AU": fields[0],
                "q_AU": fields[3],
                "i_deg": fields[2],
                "H_r": fields[9],
                "r_AU": fields[4],
                "m_r": fields[8],
            }
            rows.append(_validate_detection(row, f"{source}:{line_number}"))
    except OSError as exc:
        _fail("ossos_read_error", f"cannot read OSSOS output {source}: {exc}")
    return rows


def verify_external_simulator(
    contract: Mapping[str, Any], simulator_root: str | Path
) -> dict[str, Any]:
    locked = validate_survey_contract(contract)
    root = Path(simulator_root).resolve()
    external = locked["external_simulator"]
    missing: list[str] = []
    mismatches: list[dict[str, str]] = []
    observed: dict[str, str] = {}
    for relative, expected in external["required_files"].items():
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        digest = sha256_file(path)
        observed[relative] = digest
        if digest != expected:
            mismatches.append({"path": relative, "expected": expected, "observed": digest})
    commit: str | None = None
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if completed.returncode == 0:
            commit = completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {
        "root": str(root),
        "expected_commit": external["commit"],
        "observed_commit": commit,
        "commit_passed": commit == external["commit"],
        "missing_files": missing,
        "hash_mismatches": mismatches,
        "observed_file_hashes": observed,
        "passed": commit == external["commit"] and not missing and not mismatches,
    }


def write_pool_manifest(
    path: str | Path,
    *,
    model_id: str,
    backend: str,
    simulator_commit: str | None,
    detections_path: str | Path,
    intrinsic_draws_by_block: Mapping[int, int],
    raw_tracked_files: Sequence[Mapping[str, Any]],
    checkpoint_replay_passed: bool,
) -> dict[str, Any]:
    target = Path(path)
    rows = load_detection_csv(detections_path)
    if any(row["model_id"] != model_id for row in rows):
        _fail("pool_model_mismatch", "detection rows do not match pool model_id")
    relative_detection = os.path.relpath(Path(detections_path).resolve(), target.parent.resolve())
    manifest = {
        "schema": POOL_SCHEMA,
        "model_id": model_id,
        "backend": _nonempty_string(backend, "backend"),
        "simulator_commit": simulator_commit,
        "detections_path": relative_detection,
        "detections_sha256": sha256_file(detections_path),
        "detection_semantic_sha256": sha256_data(rows),
        "detection_count": len(rows),
        "intrinsic_draws_by_block": {
            str(block): _nonnegative_int(count, f"intrinsic draws block {block}")
            for block, count in sorted(intrinsic_draws_by_block.items())
        },
        "raw_tracked_files": [dict(item) for item in raw_tracked_files],
        "checkpoint_replay_passed": bool(checkpoint_replay_passed),
    }
    _atomic_json(target, manifest)
    return manifest


def register_official_ossos_pool(
    contract: Mapping[str, Any],
    simulator_root: str | Path,
    model_id: str,
    raw_blocks: Sequence[tuple[int, str | Path, int]],
    detections_output: str | Path,
    manifest_output: str | Path,
    *,
    checkpoint_replay_passed: bool,
) -> dict[str, Any]:
    """Strictly register raw official output and prove adapter identity."""

    locked = validate_survey_contract(contract)
    verification = verify_external_simulator(locked, simulator_root)
    if not verification["passed"]:
        _fail("external_simulator_mismatch", "pinned external simulator verification failed")
    manifest_target = Path(manifest_output)
    normalized: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    draws: dict[int, int] = {}
    for block, raw_path, intrinsic_draws in raw_blocks:
        parsed = parse_ossos_tracked_file(raw_path, model_id, block)
        normalized.extend(parsed)
        draws[block] = draws.get(block, 0) + _nonnegative_int(
            intrinsic_draws, f"intrinsic draws block {block}"
        )
        raw_records.append(
            {
                "seed_block": block,
                "path": os.path.relpath(Path(raw_path).resolve(), manifest_target.parent.resolve()),
                "sha256": sha256_file(raw_path),
                "tracked_count": len(parsed),
            }
        )
    write_detection_csv(detections_output, normalized)
    return write_pool_manifest(
        manifest_output,
        model_id=model_id,
        backend=locked["execution"]["official_backend_name"],
        simulator_commit=locked["external_simulator"]["commit"],
        detections_path=detections_output,
        intrinsic_draws_by_block=draws,
        raw_tracked_files=raw_records,
        checkpoint_replay_passed=checkpoint_replay_passed,
    )


def _load_pool(
    path: str | Path,
    contract: Mapping[str, Any],
    expected_model: str,
) -> dict[str, Any]:
    source = Path(path)
    manifest = _load_json(source, "pool_manifest_read_error")
    _check_keys(
        manifest,
        {
            "schema",
            "model_id",
            "backend",
            "simulator_commit",
            "detections_path",
            "detections_sha256",
            "detection_semantic_sha256",
            "detection_count",
            "intrinsic_draws_by_block",
            "raw_tracked_files",
            "checkpoint_replay_passed",
        },
        "pool manifest",
    )
    if manifest["schema"] != POOL_SCHEMA or manifest["model_id"] != expected_model:
        _fail("pool_manifest_mismatch", f"pool manifest does not describe {expected_model!r}")
    _nonempty_string(manifest["backend"], "pool backend")
    if manifest["simulator_commit"] is not None:
        _nonempty_string(manifest["simulator_commit"], "pool simulator commit")
    relative = _nonempty_string(manifest["detections_path"], "pool detections_path")
    if Path(relative).is_absolute():
        _fail("absolute_pool_path", "pool detections_path must be relative")
    detection_path = (source.parent / relative).resolve()
    if sha256_file(detection_path) != _sha256(manifest["detections_sha256"], "detections_sha256"):
        _fail("detection_hash_mismatch", f"detection file hash mismatch for {expected_model}")
    rows = load_detection_csv(detection_path)
    if len(rows) != _nonnegative_int(manifest["detection_count"], "detection_count"):
        _fail("detection_count_mismatch", f"detection count mismatch for {expected_model}")
    if sha256_data(rows) != _sha256(manifest["detection_semantic_sha256"], "semantic hash"):
        _fail("detection_semantic_hash_mismatch", f"semantic hash mismatch for {expected_model}")
    if any(row["model_id"] != expected_model for row in rows):
        _fail("pool_model_mismatch", f"row model mismatch in {expected_model} pool")

    draws_raw = _mapping(manifest["intrinsic_draws_by_block"], "intrinsic_draws_by_block")
    draws: dict[int, int] = {}
    for key, value in draws_raw.items():
        try:
            block = int(key)
        except (TypeError, ValueError):
            _fail("invalid_seed_block", f"invalid seed block key: {key!r}")
        draws[block] = _nonnegative_int(value, f"intrinsic draws block {block}")
    expected_blocks = set(range(int(contract["population"]["seed_blocks"])))
    if set(draws) != expected_blocks:
        _fail("seed_block_set_mismatch", f"pool must contain seed blocks {sorted(expected_blocks)}")
    if any(row["seed_block"] not in expected_blocks for row in rows):
        _fail("unplanned_seed_block", "detection contains an unplanned seed block")

    raw_files = manifest["raw_tracked_files"]
    if not isinstance(raw_files, list):
        _fail("invalid_raw_file_list", "raw_tracked_files must be a list")
    official_backend = contract["execution"]["official_backend_name"]
    adapter_identity_passed = False
    if manifest["backend"] == official_backend:
        if manifest["simulator_commit"] != contract["external_simulator"]["commit"]:
            _fail("simulator_commit_mismatch", "official pool uses the wrong simulator commit")
        parsed: list[dict[str, Any]] = []
        raw_blocks: set[int] = set()
        raw_paths: set[Path] = set()
        for index, item_raw in enumerate(raw_files):
            item = _mapping(item_raw, f"raw_tracked_files[{index}]")
            _check_keys(item, {"seed_block", "path", "sha256", "tracked_count"}, "raw tracked file")
            block = _nonnegative_int(item["seed_block"], "raw seed block")
            raw_blocks.add(block)
            raw_relative = _nonempty_string(item["path"], "raw tracked path")
            if Path(raw_relative).is_absolute():
                _fail("absolute_pool_path", "raw tracked paths must be relative")
            raw_path = (source.parent / raw_relative).resolve()
            if raw_path in raw_paths:
                _fail("duplicate_raw_file", f"raw tracked file appears twice: {raw_path}")
            raw_paths.add(raw_path)
            if sha256_file(raw_path) != _sha256(item["sha256"], "raw tracked hash"):
                _fail("raw_tracked_hash_mismatch", f"raw tracked hash mismatch for block {block}")
            block_rows = parse_ossos_tracked_file(raw_path, expected_model, block)
            if len(block_rows) != _nonnegative_int(item["tracked_count"], "raw tracked count"):
                _fail("raw_tracked_count_mismatch", f"raw tracked count mismatch for block {block}")
            parsed.extend(block_rows)
        parsed.sort(key=lambda row: row["object_id"])
        adapter_identity_passed = raw_blocks == expected_blocks and sha256_data(parsed) == sha256_data(rows)
        if not adapter_identity_passed:
            _fail("adapter_identity_mismatch", "raw OSSOS output and normalized pool differ")
    return {
        "manifest": manifest,
        "manifest_sha256": sha256_file(source),
        "rows": rows,
        "intrinsic_draws_by_block": draws,
        "total_intrinsic_draws": sum(draws.values()),
        "adapter_identity_passed": adapter_identity_passed,
        "checkpoint_replay_passed": manifest["checkpoint_replay_passed"] is True,
    }


def empirical_pit(value: float, sorted_model: Sequence[float]) -> float:
    """Tie-aware empirical probability-integral transform using midranks."""

    if not sorted_model:
        _fail("empty_model_distribution", "empirical PIT requires a nonempty model")
    value = _finite(value, "PIT value")
    left = bisect.bisect_left(sorted_model, value)
    right = bisect.bisect_right(sorted_model, value)
    if left == right:
        position = float(left)
    else:
        position = 0.5 * float(left + right)
    n = float(len(sorted_model))
    return min(1.0 - 0.5 / n, max(0.5 / n, position / n))


def anderson_darling_uniform(pit_values: Sequence[float]) -> float:
    """One-sample Anderson-Darling statistic after an empirical PIT."""

    if not pit_values:
        _fail("empty_catalog", "Anderson-Darling statistic requires observations")
    ordered = sorted(_finite(value, "PIT value") for value in pit_values)
    if ordered[0] <= 0.0 or ordered[-1] >= 1.0:
        _fail("invalid_pit", "PIT values must lie strictly between zero and one")
    n = len(ordered)
    total = 0.0
    for index, value in enumerate(ordered, start=1):
        reverse = ordered[n - index]
        total += (2 * index - 1) * (math.log(value) + math.log1p(-reverse))
    return -float(n) - total / float(n)


def _pit_rows(
    rows: Sequence[Mapping[str, Any]],
    cdfs: Mapping[str, Sequence[float]],
    variables: Sequence[str],
) -> list[tuple[float, ...]]:
    return [tuple(empirical_pit(float(row[name]), cdfs[name]) for name in variables) for row in rows]


def _catalog_statistics(
    pit_rows: Sequence[Sequence[float]],
    indices: Sequence[int],
    primary_index: int,
) -> tuple[float, float]:
    summed_ad = 0.0
    width = len(pit_rows[0])
    for variable_index in range(width):
        summed_ad += anderson_darling_uniform(
            [pit_rows[index][variable_index] for index in indices]
        )
    zeta = sum(math.log10(pit_rows[index][primary_index]) for index in indices)
    return summed_ad, zeta


def _draw_catalog_indices(
    seed: str,
    pool_size: int,
    catalog_size: int,
    label: str,
    catalog_index: int,
) -> list[int]:
    return [
        _sample_index(seed, pool_size, label, catalog_index, draw_index)
        for draw_index in range(catalog_size)
    ]


def _evaluate_statistics(
    correct_rows: Sequence[Mapping[str, Any]],
    wrong_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    label: str,
    mock_catalogs: int | None = None,
    reference_catalogs: int | None = None,
) -> dict[str, Any]:
    stats = contract["statistics"]
    gates = contract["gates"]
    variables = list(stats["variables"])
    catalog_size = int(contract["population"]["catalog_size"])
    if len(correct_rows) < catalog_size or len(wrong_rows) < catalog_size:
        _fail("insufficient_statistical_pool", "each pool must cover at least one mock catalog")
    mock_count = int(mock_catalogs or stats["mock_catalogs"])
    reference_count = int(reference_catalogs or stats["bootstrap_reference_catalogs"])
    seed = stats["resampling_seed"]
    cdfs = {name: sorted(float(row[name]) for row in correct_rows) for name in variables}
    correct_pits = _pit_rows(correct_rows, cdfs, variables)
    wrong_pits = _pit_rows(wrong_rows, cdfs, variables)
    primary_index = variables.index(stats["primary_pit_variable"])

    reference_ad: list[float] = []
    for catalog_index in range(reference_count):
        indices = _draw_catalog_indices(
            seed, len(correct_pits), catalog_size, f"{label}:reference", catalog_index
        )
        reference_ad.append(_catalog_statistics(correct_pits, indices, primary_index)[0])
    reference_ad.sort()

    correct_rejections = 0
    wrong_rejections = 0
    zeta_values: list[float] = []
    for pool_name, pits in (("correct", correct_pits), ("wrong", wrong_pits)):
        for catalog_index in range(mock_count):
            indices = _draw_catalog_indices(
                seed, len(pits), catalog_size, f"{label}:{pool_name}", catalog_index
            )
            summed_ad, zeta = _catalog_statistics(pits, indices, primary_index)
            tail_count = len(reference_ad) - bisect.bisect_left(reference_ad, summed_ad)
            p_value = (1.0 + float(tail_count)) / (1.0 + float(reference_count))
            if pool_name == "correct":
                zeta_values.append(zeta)
                if p_value < float(stats["alpha"]):
                    correct_rejections += 1
            elif p_value < float(stats["alpha"]):
                wrong_rejections += 1

    false_rejection_rate = correct_rejections / float(mock_count)
    rejection_power = wrong_rejections / float(mock_count)
    zeta_mean = statistics.fmean(zeta_values)
    zeta_sd = statistics.stdev(zeta_values)
    gate_results = {
        "correct_model_false_rejection_rate": float(gates["correct_model_false_rejection_rate_min"])
        <= false_rejection_rate
        <= float(gates["correct_model_false_rejection_rate_max"]),
        "zeta_mean": abs(zeta_mean - float(stats["zeta_expected_mean"]))
        <= float(gates["zeta_mean_absolute_tolerance"]),
        "zeta_sd": abs(zeta_sd - float(stats["zeta_expected_sd"]))
        <= float(gates["zeta_sd_absolute_tolerance"]),
        "wrong_model_rejection_power": rejection_power
        >= float(gates["wrong_model_rejection_power_min"]),
    }
    return {
        "mock_catalogs": mock_count,
        "bootstrap_reference_catalogs": reference_count,
        "catalog_size": catalog_size,
        "false_rejection_count": correct_rejections,
        "correct_model_false_rejection_rate": false_rejection_rate,
        "wrong_model_rejection_count": wrong_rejections,
        "wrong_model_rejection_power": rejection_power,
        "zeta_mean": zeta_mean,
        "zeta_sd": zeta_sd,
        "zeta_expected_mean": float(stats["zeta_expected_mean"]),
        "zeta_expected_sd": float(stats["zeta_expected_sd"]),
        "gate_results": gate_results,
        "calibration_passed": all(
            gate_results[name]
            for name in (
                "correct_model_false_rejection_rate",
                "zeta_mean",
                "zeta_sd",
            )
        ),
        "power_passed": gate_results["wrong_model_rejection_power"],
    }


def _gate_classification(result: Mapping[str, Any]) -> str:
    if not result["calibration_passed"]:
        return SurveySelectionVerdict.INVALID.value
    if not result["power_passed"]:
        return SurveySelectionVerdict.BLOCKED.value
    return SurveySelectionVerdict.PASSED.value


def _finalize_impl(
    contract_path: str | Path,
    correct_manifest_path: str | Path,
    wrong_manifest_path: str | Path,
) -> dict[str, Any]:
    contract = load_survey_contract(contract_path)
    correct = _load_pool(correct_manifest_path, contract, "correct")
    wrong = _load_pool(wrong_manifest_path, contract, "wrong")
    invalid_reasons: list[dict[str, str]] = []
    blocked_reasons: list[dict[str, str]] = []

    full = _evaluate_statistics(correct["rows"], wrong["rows"], contract, label="full")
    replay = _evaluate_statistics(correct["rows"], wrong["rows"], contract, label="full")
    exact_replay_passed = sha256_data(full) == sha256_data(replay)
    if not full["calibration_passed"]:
        invalid_reasons.append(
            {"code": "statistical_calibration_failed", "message": "one or more calibration gates failed"}
        )
    if not full["power_passed"]:
        blocked_reasons.append(
            {"code": "insufficient_wrong_model_power", "message": "wrong-model rejection power is below the locked minimum"}
        )
    if contract["gates"]["exact_replay_required"] and not exact_replay_passed:
        invalid_reasons.append(
            {"code": "exact_replay_failed", "message": "deterministic statistical replay changed"}
        )

    expected_blocks = list(range(int(contract["population"]["seed_blocks"])))
    stability: list[dict[str, Any]] = []
    full_classification = _gate_classification(full)
    for excluded in expected_blocks:
        correct_loo = [row for row in correct["rows"] if row["seed_block"] != excluded]
        wrong_loo = [row for row in wrong["rows"] if row["seed_block"] != excluded]
        loo = _evaluate_statistics(
            correct_loo,
            wrong_loo,
            contract,
            label=f"leave-one-block-out:{excluded}",
            mock_catalogs=int(contract["statistics"]["seed_stability_catalogs"]),
            reference_catalogs=int(contract["statistics"]["seed_stability_catalogs"]),
        )
        classification = _gate_classification(loo)
        stability.append(
            {
                "excluded_seed_block": excluded,
                "classification": classification,
                "false_rejection_rate": loo["correct_model_false_rejection_rate"],
                "wrong_model_rejection_power": loo["wrong_model_rejection_power"],
                "zeta_mean": loo["zeta_mean"],
                "zeta_sd": loo["zeta_sd"],
                "stable": classification == full_classification,
            }
        )
    seed_stability_passed = all(item["stable"] for item in stability)
    if contract["gates"]["seed_block_verdict_stability_required"] and not seed_stability_passed:
        invalid_reasons.append(
            {"code": "seed_block_verdict_instability", "message": "a leave-one-block-out verdict changed"}
        )

    official_backend = contract["execution"]["official_backend_name"]
    for name, pool in (("correct", correct), ("wrong", wrong)):
        manifest = pool["manifest"]
        if pool["total_intrinsic_draws"] < int(contract["population"]["minimum_intrinsic_draws_per_model"]):
            blocked_reasons.append(
                {"code": f"{name}_intrinsic_scale_incomplete", "message": "intrinsic draw target is incomplete"}
            )
        if len(pool["rows"]) < int(contract["population"]["minimum_tracked_detections_per_model"]):
            blocked_reasons.append(
                {"code": f"{name}_tracked_scale_incomplete", "message": "tracked-detection target is incomplete"}
            )
        detections_by_block = {
            block: sum(row["seed_block"] == block for row in pool["rows"])
            for block in expected_blocks
        }
        for block in expected_blocks:
            if pool["intrinsic_draws_by_block"][block] < int(
                contract["population"]["minimum_intrinsic_draws_per_seed_block"]
            ):
                blocked_reasons.append(
                    {
                        "code": f"{name}_intrinsic_block_{block}_incomplete",
                        "message": f"seed block {block} has too few intrinsic draws",
                    }
                )
            if detections_by_block[block] < int(
                contract["population"]["minimum_tracked_detections_per_seed_block"]
            ):
                blocked_reasons.append(
                    {
                        "code": f"{name}_tracked_block_{block}_incomplete",
                        "message": f"seed block {block} has too few tracked detections",
                    }
                )
        if manifest["backend"] != official_backend:
            blocked_reasons.append(
                {"code": f"{name}_official_backend_missing", "message": "pool was not produced by the pinned official backend"}
            )
        if contract["gates"]["adapter_identity_required"] and not pool["adapter_identity_passed"]:
            blocked_reasons.append(
                {"code": f"{name}_adapter_identity_unproven", "message": "raw-to-normalized adapter identity is not proven"}
            )
        if contract["execution"]["checkpoint_restart_required"] and not pool["checkpoint_replay_passed"]:
            blocked_reasons.append(
                {"code": f"{name}_checkpoint_replay_unproven", "message": "checkpoint/restart replay is not proven"}
            )

    if invalid_reasons:
        verdict = SurveySelectionVerdict.INVALID.value
    elif blocked_reasons:
        verdict = SurveySelectionVerdict.BLOCKED.value
    else:
        verdict = SurveySelectionVerdict.PASSED.value
    core = {
        "schema": RESULT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "milestone": contract["milestone"],
        "verdict": verdict,
        "claim_decision": "SCREENING_ONLY",
        "contract_sha256": sha256_file(contract_path),
        "input_manifests": {
            "correct": correct["manifest_sha256"],
            "wrong": wrong["manifest_sha256"],
        },
        "pool_summary": {
            "correct": {
                "backend": correct["manifest"]["backend"],
                "intrinsic_draws": correct["total_intrinsic_draws"],
                "tracked_detections": len(correct["rows"]),
                "adapter_identity_passed": correct["adapter_identity_passed"],
                "checkpoint_replay_passed": correct["checkpoint_replay_passed"],
            },
            "wrong": {
                "backend": wrong["manifest"]["backend"],
                "intrinsic_draws": wrong["total_intrinsic_draws"],
                "tracked_detections": len(wrong["rows"]),
                "adapter_identity_passed": wrong["adapter_identity_passed"],
                "checkpoint_replay_passed": wrong["checkpoint_replay_passed"],
            },
        },
        "statistics": full,
        "exact_replay_passed": exact_replay_passed,
        "seed_block_stability": stability,
        "seed_block_stability_passed": seed_stability_passed,
        "invalid_reasons": invalid_reasons,
        "blocked_reasons": blocked_reasons,
        "nonclaims": list(NONCLAIMS),
    }
    core["replay_sha256"] = sha256_data(core)
    return core


def finalize_survey_selection(
    contract_path: str | Path,
    correct_manifest_path: str | Path,
    wrong_manifest_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Evaluate JX-O1 and always emit PASSED, BLOCKED, or INVALID."""

    target = Path(output)
    if target.exists():
        _fail("output_exists", f"refusing to overwrite immutable output: {target}")
    try:
        result = _finalize_impl(contract_path, correct_manifest_path, wrong_manifest_path)
    except SurveySelectionError as exc:
        result = {
            "schema": RESULT_SCHEMA,
            "experiment_id": "UNRESOLVED",
            "milestone": "JX-O1_TELESCOPE_SELECTION_VALIDATION",
            "verdict": SurveySelectionVerdict.INVALID.value,
            "claim_decision": "INVALID",
            "invalid_reasons": [{"code": exc.code, "message": exc.message}],
            "blocked_reasons": [],
            "nonclaims": list(NONCLAIMS),
        }
        result["replay_sha256"] = sha256_data(result)
    _atomic_json(target, result)
    return result


def _kepler_radius(a: float, eccentricity: float, mean_anomaly_deg: float) -> float:
    mean = math.radians(mean_anomaly_deg) % (2.0 * math.pi)
    anomaly = math.pi if eccentricity > 0.8 else mean
    for _ in range(50):
        residual = anomaly - eccentricity * math.sin(anomaly) - mean
        delta = residual / (1.0 - eccentricity * math.cos(anomaly))
        anomaly -= delta
        if abs(delta) < 1e-14:
            break
    return a * (1.0 - eccentricity * math.cos(anomaly))


def _analytic_pilot_detections(
    contract: Mapping[str, Any],
    model_id: str,
    draws_per_block: int,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    seed = contract["population"]["seed_key"]
    rows: list[dict[str, Any]] = []
    draws: dict[int, int] = {}
    for block in range(int(contract["population"]["seed_blocks"])):
        intrinsic = generate_intrinsic_population(
            contract, model_id, block, draws_per_block, namespace="pilot-only"
        )
        draws[block] = draws_per_block
        for index, orbit in enumerate(intrinsic):
            radius = _kepler_radius(orbit["a_AU"], orbit["e"], orbit["mean_anomaly_deg"])
            apparent = orbit["H_r"] + 5.0 * math.log10(radius * max(radius - 1.0, 1e-12))
            scaled = (apparent - 24.2) / 0.3
            if scaled >= 50.0:
                magnitude_efficiency = 0.0
            elif scaled <= -50.0:
                magnitude_efficiency = 1.0
            else:
                magnitude_efficiency = 1.0 / (1.0 + math.exp(scaled))
            latitude_efficiency = math.exp(-0.5 * (orbit["i_deg"] / 30.0) ** 2)
            probability = 0.95 * magnitude_efficiency * latitude_efficiency
            draw = _counter_uniform(seed, "pilot-only", block, index, "analytic-selection", model_id)
            if draw < probability:
                rows.append(
                    {
                        "object_id": orbit["object_id"],
                        "model_id": model_id,
                        "seed_block": block,
                        "a_AU": orbit["a_AU"],
                        "q_AU": orbit["q_AU"],
                        "i_deg": orbit["i_deg"],
                        "H_r": orbit["H_r"],
                        "r_AU": radius,
                        "m_r": apparent,
                    }
                )
    return rows, draws


def run_analytic_survey_pilot(
    contract_path: str | Path,
    run_dir: str | Path,
    output: str | Path,
    *,
    draws_per_block: int = 10_000,
) -> dict[str, Any]:
    """Run a non-final analytic pilot through the complete verdict pipeline."""

    contract = load_survey_contract(contract_path)
    draws_per_block = _positive_int(draws_per_block, "draws_per_block")
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_paths: dict[str, Path] = {}
    for model_id in MODEL_IDS:
        rows, draws = _analytic_pilot_detections(contract, model_id, draws_per_block)
        detections = root / f"{model_id}_detections.csv"
        manifest = root / f"{model_id}_pool.json"
        write_detection_csv(detections, rows)
        write_pool_manifest(
            manifest,
            model_id=model_id,
            backend=contract["execution"]["pilot_backend_name"],
            simulator_commit=None,
            detections_path=detections,
            intrinsic_draws_by_block=draws,
            raw_tracked_files=[],
            checkpoint_replay_passed=False,
        )
        manifest_paths[model_id] = manifest
    return finalize_survey_selection(
        contract_path,
        manifest_paths["correct"],
        manifest_paths["wrong"],
        output,
    )
