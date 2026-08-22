"""Fail-closed ensemble validation for chaotic N-body populations.

The validator compares distributions and paired source/control effects across
predeclared phase/uncertainty draws.  It deliberately does not compare chaotic
particle identities after integration, and a successful numerical verdict can
never establish an astronomical detection.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from enum import Enum
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

from .provenance import (
    canonical_json,
    sha256_data,
    sha256_file,
    write_run_record,
)


SPEC_SCHEMA = "jx-ensemble-contract/v1"
PLAN_SCHEMA = "jx-ensemble-plan/v1"
VALIDITY_SCHEMA = "jx-integrator-validity/v1"
MEMBER_SCHEMA = "jx-ensemble-member-summary/v1"
RESULT_SCHEMA = "jx-ensemble-validation/v1"

NONCLAIMS = (
    "no Planet X detection or nonexistence",
    "no measured mass, distance, orbit, or sky direction",
    "no observational likelihood or survey-detectability claim",
    "no explanatory-sufficiency claim",
    "valid only for the locked priors, bodies, equations, duration, endpoints, and thresholds",
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EnsembleVerdict(str, Enum):
    PASSED = "PASSED"
    BLOCKED = "BLOCKED"
    INVALID = "INVALID"


class EnsembleValidationError(ValueError):
    """Typed input/integrity failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _fail(code: str, message: str) -> None:
    raise EnsembleValidationError(code, message)


def _atomic_json(path: str | Path, data: Mapping[str, Any], *, refuse_overwrite: bool = True) -> None:
    target = Path(path)
    if refuse_overwrite and target.exists():
        _fail("output_exists", f"refusing to overwrite locked artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError, RecursionError) as exc:
        _fail("noncanonical_output", f"cannot serialize finite canonical JSON: {exc}")
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
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


def _reject_json_constant(value: str) -> None:
    _fail("nonfinite_json_constant", f"JSON constant {value!r} is not permitted")


def _load_json(path: str | Path, code: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, RecursionError) as exc:
        _fail(code, f"cannot read valid JSON from {source}: {exc}")
    if not isinstance(value, dict):
        _fail(code, f"top-level JSON value in {source} must be an object")
    return value


def _check_keys(
    value: Mapping[str, Any],
    required: set[str],
    optional: set[str],
    context: str,
) -> None:
    missing = required - value.keys()
    extra = value.keys() - required - optional
    if missing:
        _fail("missing_field", f"{context} missing fields: {sorted(missing)}")
    if extra:
        _fail("unknown_field", f"{context} contains unknown fields: {sorted(extra)}")


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
    if isinstance(value, bool):
        _fail("invalid_integer", f"{context} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        _fail("invalid_integer", f"{context} must be a positive integer")
    if str(number) != str(value) and not isinstance(value, int):
        _fail("invalid_integer", f"{context} must be an exact integer")
    if number <= 0:
        _fail("invalid_integer", f"{context} must be positive")
    return number


def _nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail("invalid_integer", f"{context} must be a nonnegative integer")
    return value


def _safe_identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        _fail(
            "invalid_identifier",
            f"{context} must match {_SAFE_ID.pattern!r}",
        )
    return value


def _sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail("invalid_sha256", f"{context} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_threshold(value: Any, context: str) -> float:
    number = _finite(value, context)
    if number < 0.0:
        _fail("invalid_threshold", f"{context} must be nonnegative")
    return number


def _open_uniform(key: bytes, *parts: object) -> float:
    message = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hmac.new(key, message, hashlib.sha256).digest()
    integer = int.from_bytes(digest[:8], "big") >> 11
    return (integer + 0.5) / float(1 << 53)


def _draw_value(factor: Mapping[str, Any], key: bytes, *parts: object) -> str:
    u = _open_uniform(key, *parts, factor["name"])
    distribution = factor["distribution"]
    if distribution == "uniform":
        low = _finite(factor["minimum"], f"factor {factor['name']} minimum")
        high = _finite(factor["maximum"], f"factor {factor['name']} maximum")
        value = low + (high - low) * u
    elif distribution == "normal":
        mean = _finite(factor["mean"], f"factor {factor['name']} mean")
        sigma = _finite(factor["sigma"], f"factor {factor['name']} sigma")
        value = mean + sigma * NormalDist().inv_cdf(u)
    elif distribution == "phase":
        origin = _finite(factor.get("origin", 0.0), f"factor {factor['name']} origin")
        period = _finite(factor["period"], f"factor {factor['name']} period")
        value = origin + period * u
    else:  # validated before generation
        _fail("invalid_distribution", f"unsupported distribution {distribution!r}")
    if not math.isfinite(value):
        _fail("nonfinite_draw", f"factor {factor['name']} generated a nonfinite draw")
    return format(value, ".17g")


def _cholesky(matrix: Sequence[Sequence[Any]], context: str) -> list[list[float]]:
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        _fail("invalid_covariance", f"{context} covariance must be nonempty and square")
    values = [[_finite(value, f"{context} covariance") for value in row] for row in matrix]
    for i in range(size):
        for j in range(size):
            tolerance = 1e-12 * max(1.0, abs(values[i][j]), abs(values[j][i]))
            if abs(values[i][j] - values[j][i]) > tolerance:
                _fail("invalid_covariance", f"{context} covariance must be symmetric")
    lower = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            residual = values[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if residual <= 0.0 or not math.isfinite(residual):
                    _fail("non_positive_definite_covariance", f"{context} covariance is not positive definite")
                lower[i][j] = math.sqrt(residual)
            else:
                lower[i][j] = residual / lower[j][j]
    return lower


def _validate_factor(factor: Mapping[str, Any], used_names: set[str]) -> dict[str, Any]:
    if not isinstance(factor, dict):
        _fail("invalid_factor", "each factor must be an object")
    required = {"name", "scope", "distribution"}
    distribution = factor.get("distribution")
    if distribution == "uniform":
        required |= {"minimum", "maximum"}
        optional: set[str] = set()
    elif distribution == "normal":
        required |= {"mean", "sigma"}
        optional = set()
    elif distribution == "phase":
        required |= {"period"}
        optional = {"origin"}
    else:
        _fail("invalid_distribution", f"unsupported factor distribution {distribution!r}")
    _check_keys(factor, required, optional, "factor")
    name = factor["name"]
    if not isinstance(name, str) or not name.strip() or name in used_names:
        _fail("invalid_factor_name", "factor names must be unique nonempty strings")
    used_names.add(name)
    if factor["scope"] not in {"replicate", "tracer"}:
        _fail("invalid_factor_scope", f"factor {name} scope must be replicate or tracer")
    if distribution == "uniform":
        low = _finite(factor["minimum"], f"factor {name} minimum")
        high = _finite(factor["maximum"], f"factor {name} maximum")
        if high <= low:
            _fail("invalid_factor_range", f"factor {name} maximum must exceed minimum")
    elif distribution == "normal":
        if _finite(factor["sigma"], f"factor {name} sigma") <= 0.0:
            _fail("invalid_factor_sigma", f"factor {name} sigma must be positive")
        _finite(factor["mean"], f"factor {name} mean")
    else:
        if _finite(factor["period"], f"factor {name} period") <= 0.0:
            _fail("invalid_factor_period", f"factor {name} period must be positive")
        _finite(factor.get("origin", 0.0), f"factor {name} origin")
    return dict(factor)


def _validate_gaussian_block(block: Mapping[str, Any], used_names: set[str]) -> dict[str, Any]:
    if not isinstance(block, dict):
        _fail("invalid_covariance_block", "each Gaussian block must be an object")
    _check_keys(block, {"name", "scope", "variables", "mean", "covariance"}, set(), "Gaussian block")
    name = block["name"]
    if not isinstance(name, str) or not name.strip():
        _fail("invalid_covariance_block", "Gaussian block name must be nonempty")
    if block["scope"] not in {"replicate", "tracer"}:
        _fail("invalid_factor_scope", f"Gaussian block {name} scope must be replicate or tracer")
    variables = block["variables"]
    means = block["mean"]
    if not isinstance(variables, list) or not variables or not isinstance(means, list) or len(means) != len(variables):
        _fail("invalid_covariance_block", f"Gaussian block {name} variables and mean must have equal nonzero length")
    for variable in variables:
        if not isinstance(variable, str) or not variable.strip() or variable in used_names:
            _fail("invalid_factor_name", "Gaussian variable names must be unique nonempty strings")
        used_names.add(variable)
    for index, value in enumerate(means):
        _finite(value, f"Gaussian block {name} mean[{index}]")
    _cholesky(block["covariance"], f"Gaussian block {name}")
    return dict(block)


def _validate_threshold_set(value: Any, context: str) -> dict[str, str]:
    required = {
        "low_q_fraction",
        "injection_fraction",
        "survival_fraction",
        "mean_q_AU",
        "inclination_width_deg",
        "wasserstein_q_AU",
        "wasserstein_i_deg",
    }
    optional = {"wasserstein_min_q_AU"}
    if not isinstance(value, dict):
        _fail("invalid_gate_set", f"{context} must be an object")
    _check_keys(value, required, optional, context)
    normalized: dict[str, str] = {}
    supplied = dict(value)
    supplied.setdefault("wasserstein_min_q_AU", value["wasserstein_q_AU"])
    for name in sorted(required | optional):
        _nonnegative_threshold(supplied[name], f"{context}.{name}")
        normalized[name] = str(supplied[name])
    return normalized


def validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a preregistered ensemble contract."""
    required = {
        "schema", "experiment_id", "purpose", "registration_status", "registration_reference",
        "evidence_class", "dynamics_model_sha256", "initial_state_model_sha256",
        "source_model_sha256", "seed_blocks", "replicates_per_block", "tracers_per_replicate",
        "epochs_year", "duration_years", "frame", "origin", "units", "q_threshold_AU",
        "factors", "gaussian_blocks", "methods", "gates", "inference", "power_plan",
    }
    _check_keys(contract, required, set(), "ensemble contract")
    if contract["schema"] != SPEC_SCHEMA:
        _fail("schema_mismatch", f"contract schema must be {SPEC_SCHEMA}")
    for name in ("experiment_id", "purpose", "frame", "origin", "units", "power_plan"):
        if not isinstance(contract[name], str) or not contract[name].strip():
            _fail("invalid_text", f"contract {name} must be a nonempty string")
    if contract["evidence_class"] != "MODEL_OUTPUT":
        _fail("invalid_evidence_class", "ensemble evidence_class must be MODEL_OUTPUT")
    for name in ("dynamics_model_sha256", "initial_state_model_sha256", "source_model_sha256"):
        _sha256(contract[name], name)
        if contract[name] == "0" * 64:
            _fail("invalid_sha256", f"{name} must not use the all-zero placeholder")
    if contract["registration_status"] not in {"EXPLORATORY", "CONFIRMATORY"}:
        _fail("invalid_registration", "registration_status must be EXPLORATORY or CONFIRMATORY")
    if not isinstance(contract["registration_reference"], str):
        _fail("invalid_registration", "registration_reference must be a string")
    if contract["registration_status"] == "CONFIRMATORY" and not contract["registration_reference"].strip():
        _fail("missing_registration_reference", "CONFIRMATORY contracts require an external or Git reference")

    seeds = contract["seed_blocks"]
    if not isinstance(seeds, list) or not seeds or any(not isinstance(seed, str) or not seed for seed in seeds):
        _fail("invalid_seed_blocks", "seed_blocks must contain nonempty strings")
    if len(set(seeds)) != len(seeds):
        _fail("duplicate_seed", "seed_blocks must be distinct")
    replicates = _positive_int(contract["replicates_per_block"], "replicates_per_block")
    tracers = _positive_int(contract["tracers_per_replicate"], "tracers_per_replicate")
    duration = _positive_int(contract["duration_years"], "duration_years")
    epochs_raw = contract["epochs_year"]
    if not isinstance(epochs_raw, list) or len(epochs_raw) < 2:
        _fail("invalid_epochs", "epochs_year must contain at least initial and final epochs")
    epochs: list[int] = []
    for index, epoch in enumerate(epochs_raw):
        if isinstance(epoch, bool):
            _fail("invalid_epochs", f"epoch {index} must be an integer")
        try:
            parsed = int(epoch)
        except (TypeError, ValueError, OverflowError):
            _fail("invalid_epochs", f"epoch {index} must be an integer")
        if float(epoch) != parsed or parsed < 0:
            _fail("invalid_epochs", "epochs must be nonnegative integers")
        epochs.append(parsed)
    if epochs != sorted(set(epochs)) or epochs[0] != 0 or epochs[-1] != duration:
        _fail("invalid_epochs", "epochs must be unique, sorted, begin at 0, and end at duration_years")
    q_threshold = _finite(contract["q_threshold_AU"], "q_threshold_AU")
    if q_threshold <= 0.0:
        _fail("invalid_q_threshold", "q_threshold_AU must be positive")

    used_names: set[str] = set()
    factors_raw = contract["factors"]
    if not isinstance(factors_raw, list):
        _fail("invalid_factor", "factors must be a list")
    factors = [_validate_factor(factor, used_names) for factor in factors_raw]
    blocks_raw = contract["gaussian_blocks"]
    if not isinstance(blocks_raw, list):
        _fail("invalid_covariance_block", "gaussian_blocks must be a list")
    gaussian_blocks = [_validate_gaussian_block(block, used_names) for block in blocks_raw]

    methods_raw = contract["methods"]
    if not isinstance(methods_raw, list) or not methods_raw:
        _fail("invalid_methods", "methods must be a nonempty list")
    methods: list[dict[str, Any]] = []
    method_ids: set[str] = set()
    method_configurations: set[str] = set()
    for method in methods_raw:
        if not isinstance(method, dict):
            _fail("invalid_method", "each method must be an object")
        _check_keys(method, {"method_id", "implementation", "version", "independence_group", "settings"}, set(), "method")
        for name in ("method_id", "implementation", "version", "independence_group"):
            if not isinstance(method[name], str) or not method[name].strip():
                _fail("invalid_method", f"method {name} must be nonempty")
        _safe_identifier(method["method_id"], "method_id")
        _safe_identifier(method["independence_group"], "independence_group")
        if method["method_id"] in method_ids:
            _fail("duplicate_method", f"duplicate method_id {method['method_id']}")
        method_ids.add(method["method_id"])
        if not isinstance(method["settings"], dict):
            _fail("invalid_method", "method settings must be an object")
        try:
            canonical_json(method["settings"])
        except (TypeError, ValueError, RecursionError) as exc:
            _fail("invalid_method", f"method settings are not JSON serializable: {exc}")
        configuration = sha256_data(
            {
                "implementation": method["implementation"],
                "version": method["version"],
                "settings": method["settings"],
            }
        )
        if configuration in method_configurations:
            _fail("duplicate_method_configuration", "method IDs must not duplicate an identical numerical configuration")
        method_configurations.add(configuration)
        methods.append(dict(method))

    gates = contract["gates"]
    if not isinstance(gates, dict):
        _fail("invalid_gates", "gates must be an object")
    _check_keys(
        gates,
        {
            "minimum_blocks", "minimum_replicates_per_block", "minimum_tracers_per_replicate",
            "minimum_methods", "minimum_independence_groups", "require_within_group_repeat",
            "minimum_bound_samples_per_epoch", "method_equivalence", "repeat_equivalence",
            "max_primary_effect_method_disagreement",
        },
        {"max_primary_effect_repeat_disagreement"},
        "gates",
    )
    normalized_gates = {
        "minimum_blocks": _positive_int(gates["minimum_blocks"], "minimum_blocks"),
        "minimum_replicates_per_block": _positive_int(gates["minimum_replicates_per_block"], "minimum_replicates_per_block"),
        "minimum_tracers_per_replicate": _positive_int(gates["minimum_tracers_per_replicate"], "minimum_tracers_per_replicate"),
        "minimum_methods": _positive_int(gates["minimum_methods"], "minimum_methods"),
        "minimum_independence_groups": _positive_int(gates["minimum_independence_groups"], "minimum_independence_groups"),
        "require_within_group_repeat": gates["require_within_group_repeat"],
        "minimum_bound_samples_per_epoch": _positive_int(gates["minimum_bound_samples_per_epoch"], "minimum_bound_samples_per_epoch"),
        "method_equivalence": _validate_threshold_set(gates["method_equivalence"], "method_equivalence"),
        "repeat_equivalence": _validate_threshold_set(gates["repeat_equivalence"], "repeat_equivalence"),
        "max_primary_effect_method_disagreement": str(gates["max_primary_effect_method_disagreement"]),
        "max_primary_effect_repeat_disagreement": str(
            gates.get(
                "max_primary_effect_repeat_disagreement",
                gates["max_primary_effect_method_disagreement"],
            )
        ),
    }
    if not isinstance(normalized_gates["require_within_group_repeat"], bool):
        _fail("invalid_gates", "require_within_group_repeat must be boolean")
    _nonnegative_threshold(
        gates["max_primary_effect_method_disagreement"],
        "max_primary_effect_method_disagreement",
    )
    _nonnegative_threshold(
        normalized_gates["max_primary_effect_repeat_disagreement"],
        "max_primary_effect_repeat_disagreement",
    )

    inference = contract["inference"]
    if not isinstance(inference, dict):
        _fail("invalid_inference", "inference must be an object")
    _check_keys(
        inference,
        {
            "primary_endpoint", "confidence_level", "bootstrap_repetitions",
            "null_equivalence_margin", "minimum_material_effect",
        },
        set(),
        "inference",
    )
    allowed_endpoints = {"injection_fraction", "survival_fraction", "final_low_q_fraction", "final_mean_q_AU"}
    if inference["primary_endpoint"] not in allowed_endpoints:
        _fail("invalid_endpoint", f"primary_endpoint must be one of {sorted(allowed_endpoints)}")
    confidence = _finite(inference["confidence_level"], "confidence_level")
    if not 0.0 < confidence < 1.0:
        _fail("invalid_confidence", "confidence_level must lie strictly between 0 and 1")
    bootstrap_repetitions = _positive_int(inference["bootstrap_repetitions"], "bootstrap_repetitions")
    null_margin = _nonnegative_threshold(inference["null_equivalence_margin"], "null_equivalence_margin")
    material_margin = _nonnegative_threshold(inference["minimum_material_effect"], "minimum_material_effect")
    if material_margin <= null_margin:
        _fail("invalid_effect_margins", "minimum_material_effect must exceed null_equivalence_margin")
    normalized_inference = {
        "primary_endpoint": inference["primary_endpoint"],
        "confidence_level": str(inference["confidence_level"]),
        "bootstrap_repetitions": bootstrap_repetitions,
        "null_equivalence_margin": str(inference["null_equivalence_margin"]),
        "minimum_material_effect": str(inference["minimum_material_effect"]),
    }

    return {
        **dict(contract),
        "seed_blocks": list(seeds),
        "replicates_per_block": replicates,
        "tracers_per_replicate": tracers,
        "epochs_year": epochs,
        "duration_years": duration,
        "q_threshold_AU": str(contract["q_threshold_AU"]),
        "factors": factors,
        "gaussian_blocks": gaussian_blocks,
        "methods": methods,
        "gates": normalized_gates,
        "inference": normalized_inference,
    }


def _draw_gaussian_block(block: Mapping[str, Any], key: bytes, *parts: object) -> dict[str, str]:
    lower = _cholesky(block["covariance"], f"Gaussian block {block['name']}")
    means = [_finite(value, f"Gaussian block {block['name']} mean") for value in block["mean"]]
    normals = [
        NormalDist().inv_cdf(_open_uniform(key, *parts, block["name"], variable))
        for variable in block["variables"]
    ]
    values: dict[str, str] = {}
    for i, variable in enumerate(block["variables"]):
        value = means[i] + sum(lower[i][j] * normals[j] for j in range(i + 1))
        if not math.isfinite(value):
            _fail("nonfinite_draw", f"Gaussian block {block['name']} generated a nonfinite draw")
        values[variable] = format(value, ".17g")
    return values


def _generate_members(contract: Mapping[str, Any], contract_sha256: str) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    replicate_factors = [factor for factor in contract["factors"] if factor["scope"] == "replicate"]
    tracer_factors = [factor for factor in contract["factors"] if factor["scope"] == "tracer"]
    replicate_blocks = [block for block in contract["gaussian_blocks"] if block["scope"] == "replicate"]
    tracer_blocks = [block for block in contract["gaussian_blocks"] if block["scope"] == "tracer"]
    for block_index, seed in enumerate(contract["seed_blocks"]):
        block_id = f"b{block_index:02d}"
        key = hashlib.sha256(f"{contract_sha256}\x1f{seed}".encode("utf-8")).digest()
        for replicate_index in range(contract["replicates_per_block"]):
            replicate_id = f"r{replicate_index:04d}"
            member_id = f"{block_id}-{replicate_id}"
            replicate_draws = {
                factor["name"]: _draw_value(factor, key, block_id, replicate_id, "replicate")
                for factor in replicate_factors
            }
            for gaussian in replicate_blocks:
                replicate_draws.update(_draw_gaussian_block(gaussian, key, block_id, replicate_id, "replicate"))
            tracers: list[dict[str, Any]] = []
            for tracer_index in range(contract["tracers_per_replicate"]):
                tracer_id = f"t{tracer_index:04d}"
                draws = {
                    factor["name"]: _draw_value(factor, key, block_id, replicate_id, tracer_id)
                    for factor in tracer_factors
                }
                for gaussian in tracer_blocks:
                    draws.update(_draw_gaussian_block(gaussian, key, block_id, replicate_id, tracer_id))
                tracer = {"tracer_id": tracer_id, "draws": dict(sorted(draws.items()))}
                tracer["draw_sha256"] = sha256_data(tracer)
                tracers.append(tracer)
            member = {
                "member_id": member_id,
                "block_id": block_id,
                "replicate_id": replicate_id,
                "draws": dict(sorted(replicate_draws.items())),
                "tracers": tracers,
            }
            member["initial_draw_sha256"] = sha256_data(member)
            members.append(member)
    return members


def prepare_ensemble_plan(contract_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    contract = validate_contract(_load_json(contract_path, "invalid_contract_json"))
    contract_sha256 = sha256_data(contract)
    members = _generate_members(contract, contract_sha256)
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "contract": contract,
        "contract_sha256": contract_sha256,
        "members": members,
        "member_manifest_sha256": sha256_data(members),
    }
    plan["plan_sha256"] = sha256_data(plan)
    _atomic_json(output_path, plan)
    return plan


def load_ensemble_plan(path: str | Path) -> dict[str, Any]:
    plan = _load_json(path, "invalid_plan_json")
    _check_keys(
        plan,
        {"schema", "contract", "contract_sha256", "members", "member_manifest_sha256", "plan_sha256"},
        set(),
        "ensemble plan",
    )
    if plan["schema"] != PLAN_SCHEMA:
        _fail("schema_mismatch", f"plan schema must be {PLAN_SCHEMA}")
    contract = validate_contract(plan["contract"])
    if plan["contract"] != contract:
        _fail("noncanonical_contract", "locked contract is not in canonical normalized form")
    if plan["contract_sha256"] != sha256_data(contract):
        _fail("contract_hash_mismatch", "contract hash does not match locked contract")
    expected_members = _generate_members(contract, plan["contract_sha256"])
    if plan["members"] != expected_members:
        _fail("member_regeneration_mismatch", "stored members do not match deterministic regeneration")
    if plan["member_manifest_sha256"] != sha256_data(plan["members"]):
        _fail("member_manifest_hash_mismatch", "member manifest hash mismatch")
    unsigned = dict(plan)
    supplied_hash = unsigned.pop("plan_sha256")
    if supplied_hash != sha256_data(unsigned):
        _fail("plan_hash_mismatch", "plan hash mismatch")
    return plan


def _parse_bound(value: Any, context: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    _fail("invalid_bound_flag", f"{context} bound must be 0/1 or true/false")


def _parse_epoch(value: Any, context: str) -> int:
    number = _finite(value, context)
    integer = int(number)
    if number != integer or integer < 0:
        _fail("invalid_epoch", f"{context} time_year must be a nonnegative integer")
    return integer


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        _fail("empty_distribution", "quantile requires a nonempty sample")
    if not 0.0 <= probability <= 1.0:
        _fail("invalid_probability", "quantile probability must be in [0,1]")
    ordered = sorted(_finite(value, "quantile sample") for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def wasserstein_1d(
    values_a: Sequence[float],
    values_b: Sequence[float],
    weights_a: Sequence[float] | None = None,
    weights_b: Sequence[float] | None = None,
) -> float:
    """Exact empirical one-dimensional W1 over merged sorted support."""
    if not values_a or not values_b:
        _fail("empty_distribution", "Wasserstein distance requires two nonempty samples")
    a = [_finite(value, "Wasserstein sample A") for value in values_a]
    b = [_finite(value, "Wasserstein sample B") for value in values_b]

    def normalized_weights(values: Sequence[float], weights: Sequence[float] | None, label: str) -> list[float]:
        if weights is None:
            return [1.0 / len(values)] * len(values)
        if len(weights) != len(values):
            _fail("invalid_weights", f"{label} weights length mismatch")
        parsed = [_finite(weight, f"{label} weight") for weight in weights]
        if any(weight <= 0.0 for weight in parsed):
            _fail("invalid_weights", f"{label} weights must be positive")
        total = sum(parsed)
        return [weight / total for weight in parsed]

    wa = normalized_weights(a, weights_a, "A")
    wb = normalized_weights(b, weights_b, "B")
    mass_a: dict[float, float] = defaultdict(float)
    mass_b: dict[float, float] = defaultdict(float)
    for value, weight in zip(a, wa):
        mass_a[value] += weight
    for value, weight in zip(b, wb):
        mass_b[value] += weight
    support = sorted(set(mass_a) | set(mass_b))
    cdf_a = cdf_b = 0.0
    previous = support[0]
    distance = 0.0
    for value in support:
        distance += abs(cdf_a - cdf_b) * (value - previous)
        cdf_a += mass_a.get(value, 0.0)
        cdf_b += mass_b.get(value, 0.0)
        previous = value
    return distance


def _strict_trajectory(path: str | Path, plan: Mapping[str, Any], member: Mapping[str, Any]) -> dict[str, Any]:
    trajectory = Path(path)
    expected_epochs = set(plan["contract"]["epochs_year"])
    expected_tracers = {row["tracer_id"] for row in member["tracers"]}
    required_columns = {"time_year", "name", "q", "i_deg", "bound"}
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    try:
        stream = trajectory.open(newline="", encoding="utf-8")
    except OSError as exc:
        _fail("trajectory_read_failure", f"cannot open {trajectory}: {exc}")
    with stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            _fail("trajectory_schema_mismatch", f"{trajectory} requires columns {sorted(required_columns)}")
        for row_number, raw in enumerate(reader, start=2):
            name = str(raw["name"]).strip()
            if name not in expected_tracers:
                if name.startswith("t"):
                    _fail("unexpected_tracer", f"{trajectory}:{row_number} unexpected tracer {name}")
                continue
            epoch = _parse_epoch(raw["time_year"], f"{trajectory}:{row_number}")
            if epoch not in expected_epochs:
                _fail("unexpected_epoch", f"{trajectory}:{row_number} unexpected epoch {epoch}")
            key = (epoch, name)
            if key in rows:
                _fail("duplicate_trajectory_row", f"{trajectory}:{row_number} duplicate row {key}")
            bound = _parse_bound(raw["bound"], f"{trajectory}:{row_number}")
            if bound:
                q = _finite(raw["q"], f"{trajectory}:{row_number} q")
                inclination = _finite(raw["i_deg"], f"{trajectory}:{row_number} i_deg")
                if q < 0.0:
                    _fail("invalid_perihelion", f"{trajectory}:{row_number} q must be nonnegative")
                if not 0.0 <= inclination <= 180.0:
                    _fail("invalid_inclination", f"{trajectory}:{row_number} inclination must be in [0,180]")
            else:
                q = inclination = None
            rows[key] = {
                "time_year": epoch,
                "tracer_id": name,
                "bound": bound,
                "q_AU": q,
                "i_deg": inclination,
            }
    expected_support = {(epoch, tracer) for epoch in expected_epochs for tracer in expected_tracers}
    missing = expected_support - rows.keys()
    if missing:
        preview = sorted(missing)[:5]
        _fail("missing_trajectory_support", f"{trajectory} missing {len(missing)} locked rows; first {preview}")
    normalized = [rows[key] for key in sorted(rows)]
    return {
        "rows": normalized,
        "raw_sha256": sha256_file(trajectory),
        "semantic_sha256": sha256_data(normalized),
    }


def _trajectory_summary(trajectory: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    q_threshold = float(plan["contract"]["q_threshold_AU"])
    tracers = sorted({row["tracer_id"] for row in trajectory["rows"]})
    epochs = plan["contract"]["epochs_year"]
    indexed = {(row["time_year"], row["tracer_id"]): row for row in trajectory["rows"]}
    injection_count = 0
    minimum_q: list[float] = []
    tracer_outcomes: list[dict[str, Any]] = []
    for tracer in tracers:
        history = [indexed[(epoch, tracer)] for epoch in epochs]
        injected = False
        injection_epoch = None
        injection_from_q = None
        injection_to_q = None
        bound_q = [float(row["q_AU"]) for row in history if row["bound"]]
        tracer_minimum_q = min(bound_q) if bound_q else None
        if bound_q:
            minimum_q.append(tracer_minimum_q)
        for current_epoch, previous, current in zip(epochs[1:], history, history[1:]):
            if (
                not injected
                and previous["bound"]
                and current["bound"]
                and float(previous["q_AU"]) >= q_threshold
                and float(current["q_AU"]) < q_threshold
            ):
                injected = True
                injection_epoch = current_epoch
                injection_from_q = float(previous["q_AU"])
                injection_to_q = float(current["q_AU"])
        injection_count += injected
        tracer_outcomes.append(
            {
                "tracer_id": tracer,
                "initial_bound": history[0]["bound"],
                "initial_q_AU": float(history[0]["q_AU"]) if history[0]["bound"] else None,
                "minimum_q_AU": tracer_minimum_q,
                "injected": injected,
                "injection_epoch_year": injection_epoch,
                "injection_from_q_AU": injection_from_q,
                "injection_to_q_AU": injection_to_q,
                "final_bound": history[-1]["bound"],
                "final_q_AU": float(history[-1]["q_AU"]) if history[-1]["bound"] else None,
                "history": [
                    {
                        "epoch_year": epoch,
                        "bound": row["bound"],
                        "q_AU": float(row["q_AU"]) if row["bound"] else None,
                    }
                    for epoch, row in zip(epochs, history)
                ],
            }
        )
    epoch_summaries: dict[str, Any] = {}
    for epoch in epochs:
        current = [indexed[(epoch, tracer)] for tracer in tracers]
        q_values = [float(row["q_AU"]) for row in current if row["bound"]]
        i_values = [float(row["i_deg"]) for row in current if row["bound"]]
        low_q_count = sum(row["bound"] and float(row["q_AU"]) < q_threshold for row in current)
        bound_count = sum(row["bound"] for row in current)
        epoch_summaries[str(epoch)] = {
            "sample_count": len(current),
            "bound_count": bound_count,
            "low_q_count": low_q_count,
            "bound_fraction": bound_count / len(current),
            "low_q_fraction": low_q_count / len(current),
            "mean_q_AU": math.fsum(q_values) / len(q_values) if q_values else None,
            "inclination_width_deg": quantile(i_values, 0.84) - quantile(i_values, 0.16) if i_values else None,
            "q_values_AU": sorted(q_values),
            "i_values_deg": sorted(i_values),
        }
    final = epoch_summaries[str(epochs[-1])]
    return {
        "tracer_count": len(tracers),
        "injection_count": injection_count,
        "injection_fraction": injection_count / len(tracers),
        "survival_count": final["bound_count"],
        "survival_fraction": final["bound_fraction"],
        "minimum_q_values_AU": sorted(minimum_q),
        "tracer_outcomes": tracer_outcomes,
        "epochs": epoch_summaries,
    }


def _validate_source_manifest(value: Any, context: str) -> None:
    if not isinstance(value, dict):
        _fail("invalid_source_manifest", f"{context} must be an object")
    _check_keys(value, {"schema", "scope", "files", "tree_sha256"}, set(), context)
    if value["schema"] != "jx-source-manifest/v2" or value["scope"] not in {"repository", "installed_package"}:
        _fail("invalid_source_manifest", f"{context} has an unsupported schema or scope")
    files = value["files"]
    if not isinstance(files, dict) or not files:
        _fail("invalid_source_manifest", f"{context} files must be a nonempty object")
    for name, digest in files.items():
        if not isinstance(name, str) or not name:
            _fail("invalid_source_manifest", f"{context} contains an invalid source path")
        _sha256(digest, f"{context} file {name}")
    _sha256(value["tree_sha256"], f"{context} tree_sha256")
    if value["tree_sha256"] != sha256_data(files):
        _fail("source_manifest_hash_mismatch", f"{context} tree hash does not match its files")


def _validate_validity_record(
    validity: Mapping[str, Any],
    plan: Mapping[str, Any],
    member: Mapping[str, Any],
    arm: str,
    method: Mapping[str, Any],
    trajectory_sha256: str,
) -> None:
    if not isinstance(validity, dict):
        _fail("invalid_validity_record", "validity record must be an object")
    _check_keys(
        validity,
        {
            "schema", "plan_sha256", "member_id", "arm", "method_id",
            "method_spec_sha256", "initial_draw_sha256",
            "dynamics_model_sha256", "initial_state_model_sha256",
            "source_model_sha256",
            "relative_initial_state_sha256", "full_initial_state_sha256",
            "trajectory_sha256", "duration_years", "epochs_year", "frame",
            "origin", "units", "runner_source_manifest", "passed", "checks",
        },
        set(),
        "validity record",
    )
    if validity["schema"] != VALIDITY_SCHEMA:
        _fail("schema_mismatch", f"validity schema must be {VALIDITY_SCHEMA}")
    member_id = member["member_id"]
    method_id = method["method_id"]
    if (validity["member_id"], validity["arm"], validity["method_id"]) != (member_id, arm, method_id):
        _fail("validity_identity_mismatch", "validity record identity does not match member run")
    expected = {
        "plan_sha256": plan["plan_sha256"],
        "method_spec_sha256": sha256_data(method),
        "initial_draw_sha256": member["initial_draw_sha256"],
        "dynamics_model_sha256": plan["contract"]["dynamics_model_sha256"],
        "initial_state_model_sha256": plan["contract"]["initial_state_model_sha256"],
        "source_model_sha256": plan["contract"]["source_model_sha256"],
        "trajectory_sha256": trajectory_sha256,
        "duration_years": plan["contract"]["duration_years"],
        "epochs_year": plan["contract"]["epochs_year"],
        "frame": plan["contract"]["frame"],
        "origin": plan["contract"]["origin"],
        "units": plan["contract"]["units"],
    }
    for name, expected_value in expected.items():
        if validity[name] != expected_value:
            _fail("validity_scope_mismatch", f"validity {name} does not match the locked run")
    _sha256(validity["relative_initial_state_sha256"], "relative_initial_state_sha256")
    _sha256(validity["full_initial_state_sha256"], "full_initial_state_sha256")
    if validity["relative_initial_state_sha256"] == "0" * 64 or validity["full_initial_state_sha256"] == "0" * 64:
        _fail("invalid_sha256", "initial-state digests must not use the all-zero placeholder")
    _validate_source_manifest(validity["runner_source_manifest"], "validity runner_source_manifest")
    if not isinstance(validity["passed"], bool) or not isinstance(validity["checks"], dict) or not validity["checks"]:
        _fail("invalid_validity_record", "validity record needs boolean passed and nonempty checks")
    check_results: list[bool] = []
    for name, check in validity["checks"].items():
        if not isinstance(name, str) or not name or not isinstance(check, dict) or not isinstance(check.get("passed"), bool):
            _fail("invalid_validity_record", "each validity check needs a name and boolean passed field")
        check_results.append(check["passed"])
    if validity["passed"] != all(check_results):
        _fail("invalid_validity_record", "validity passed flag disagrees with checks")


def _load_validity(
    path: str | Path,
    plan: Mapping[str, Any],
    member: Mapping[str, Any],
    arm: str,
    method: Mapping[str, Any],
    trajectory_sha256: str,
) -> dict[str, Any]:
    validity = _load_json(path, "invalid_validity_json")
    _validate_validity_record(validity, plan, member, arm, method, trajectory_sha256)
    return validity


def register_ensemble_member(
    plan_path: str | Path,
    member_id: str,
    arm: str,
    method_id: str,
    trajectory_path: str | Path,
    validity_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    plan = load_ensemble_plan(plan_path)
    if arm not in {"control", "source"}:
        _fail("invalid_arm", "arm must be control or source")
    members = {member["member_id"]: member for member in plan["members"]}
    if member_id not in members:
        _fail("unknown_member", f"unknown member_id {member_id}")
    methods = {method["method_id"]: method for method in plan["contract"]["methods"]}
    if method_id not in methods:
        _fail("unknown_method", f"unknown method_id {method_id}")
    if Path(output_path).exists():
        _fail("output_exists", f"refusing to overwrite member record: {output_path}")
    trajectory = _strict_trajectory(trajectory_path, plan, members[member_id])
    validity = _load_validity(
        validity_path,
        plan,
        members[member_id],
        arm,
        methods[method_id],
        trajectory["raw_sha256"],
    )
    summary = _trajectory_summary(trajectory, plan)
    payload = {
        "schema": MEMBER_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "member_id": member_id,
        "block_id": members[member_id]["block_id"],
        "replicate_id": members[member_id]["replicate_id"],
        "arm": arm,
        "method_id": method_id,
        "independence_group": methods[method_id]["independence_group"],
        "method_spec_sha256": sha256_data(methods[method_id]),
        "initial_draw_sha256": members[member_id]["initial_draw_sha256"],
        "relative_initial_state_sha256": validity["relative_initial_state_sha256"],
        "full_initial_state_sha256": validity["full_initial_state_sha256"],
        "trajectory_sha256": trajectory["raw_sha256"],
        "trajectory_semantic_sha256": trajectory["semantic_sha256"],
        "validity_sha256": sha256_file(validity_path),
        "validity_semantic_sha256": sha256_data(validity),
        "validity": validity,
        "validity_passed": validity["passed"],
        "summary": summary,
        "summary_sha256": sha256_data(summary),
        "evidence_class": "MODEL_OUTPUT",
        "dynamics_model_sha256": plan["contract"]["dynamics_model_sha256"],
        "initial_state_model_sha256": plan["contract"]["initial_state_model_sha256"],
        "source_model_sha256": plan["contract"]["source_model_sha256"],
    }
    return write_run_record(output_path, payload)


def _json_number(value: Any, context: str) -> float:
    number = _finite(value, context)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail("invalid_number", f"{context} must be encoded as a JSON number")
    return number


def _same_number(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=1e-15, abs_tol=1e-15)


def _summary_sample(
    value: Any,
    context: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> list[float]:
    if not isinstance(value, list):
        _fail("invalid_summary", f"{context} must be an array")
    parsed = [_json_number(item, f"{context}[{index}]") for index, item in enumerate(value)]
    if any(item < minimum or (maximum is not None and item > maximum) for item in parsed):
        _fail("invalid_summary", f"{context} contains an out-of-range value")
    if parsed != sorted(parsed):
        _fail("invalid_summary", f"{context} must be sorted")
    return parsed


def _validate_member_summary(summary: Any, plan: Mapping[str, Any], context: str) -> None:
    if not isinstance(summary, dict):
        _fail("invalid_summary", f"{context} must be an object")
    _check_keys(
        summary,
        {
            "tracer_count", "injection_count", "injection_fraction",
            "survival_count", "survival_fraction", "minimum_q_values_AU",
            "tracer_outcomes", "epochs",
        },
        set(),
        context,
    )
    tracer_count = _nonnegative_int(summary["tracer_count"], f"{context}.tracer_count")
    if tracer_count != plan["contract"]["tracers_per_replicate"]:
        _fail("summary_support_mismatch", f"{context} tracer_count does not match the locked plan")
    injection_count = _nonnegative_int(summary["injection_count"], f"{context}.injection_count")
    survival_count = _nonnegative_int(summary["survival_count"], f"{context}.survival_count")
    if injection_count > tracer_count or survival_count > tracer_count:
        _fail("invalid_summary", f"{context} event counts exceed tracer_count")
    injection_fraction = _json_number(summary["injection_fraction"], f"{context}.injection_fraction")
    survival_fraction = _json_number(summary["survival_fraction"], f"{context}.survival_fraction")
    if not _same_number(injection_fraction, injection_count / tracer_count):
        _fail("summary_consistency_error", f"{context} injection fraction disagrees with its count")
    if not _same_number(survival_fraction, survival_count / tracer_count):
        _fail("summary_consistency_error", f"{context} survival fraction disagrees with its count")

    minimum_q = _summary_sample(
        summary["minimum_q_values_AU"],
        f"{context}.minimum_q_values_AU",
        minimum=0.0,
    )
    if len(minimum_q) > tracer_count:
        _fail("invalid_summary", f"{context} has more minimum-q values than tracers")
    epochs = summary["epochs"]
    if not isinstance(epochs, dict):
        _fail("invalid_summary", f"{context}.epochs must be an object")
    expected_epoch_keys = {str(epoch) for epoch in plan["contract"]["epochs_year"]}
    if set(epochs) != expected_epoch_keys:
        _fail("summary_support_mismatch", f"{context} epochs do not match the locked plan")

    maximum_bound_count = 0
    final_epoch = str(plan["contract"]["epochs_year"][-1])
    q_threshold = float(plan["contract"]["q_threshold_AU"])
    for epoch in plan["contract"]["epochs_year"]:
        epoch_key = str(epoch)
        row = epochs[epoch_key]
        row_context = f"{context}.epochs.{epoch_key}"
        if not isinstance(row, dict):
            _fail("invalid_summary", f"{row_context} must be an object")
        _check_keys(
            row,
            {
                "sample_count", "bound_count", "low_q_count", "bound_fraction",
                "low_q_fraction", "mean_q_AU", "inclination_width_deg",
                "q_values_AU", "i_values_deg",
            },
            set(),
            row_context,
        )
        sample_count = _nonnegative_int(row["sample_count"], f"{row_context}.sample_count")
        bound_count = _nonnegative_int(row["bound_count"], f"{row_context}.bound_count")
        low_q_count = _nonnegative_int(row["low_q_count"], f"{row_context}.low_q_count")
        if sample_count != tracer_count or low_q_count > bound_count or bound_count > sample_count:
            _fail("summary_consistency_error", f"{row_context} counts are inconsistent")
        maximum_bound_count = max(maximum_bound_count, bound_count)
        bound_fraction = _json_number(row["bound_fraction"], f"{row_context}.bound_fraction")
        low_q_fraction = _json_number(row["low_q_fraction"], f"{row_context}.low_q_fraction")
        if not _same_number(bound_fraction, bound_count / sample_count):
            _fail("summary_consistency_error", f"{row_context} bound fraction disagrees with its count")
        if not _same_number(low_q_fraction, low_q_count / sample_count):
            _fail("summary_consistency_error", f"{row_context} low-q fraction disagrees with its count")
        q_values = _summary_sample(row["q_values_AU"], f"{row_context}.q_values_AU", minimum=0.0)
        i_values = _summary_sample(
            row["i_values_deg"],
            f"{row_context}.i_values_deg",
            minimum=0.0,
            maximum=180.0,
        )
        if len(q_values) != bound_count or len(i_values) != bound_count:
            _fail("summary_consistency_error", f"{row_context} conditional sample lengths disagree with bound_count")
        if sum(value < q_threshold for value in q_values) != low_q_count:
            _fail("summary_consistency_error", f"{row_context} low-q count disagrees with q_values_AU")
        if bound_count == 0:
            if row["mean_q_AU"] is not None or row["inclination_width_deg"] is not None:
                _fail("summary_consistency_error", f"{row_context} empty bound sample must have null moments")
        else:
            mean_q = _json_number(row["mean_q_AU"], f"{row_context}.mean_q_AU")
            width = _json_number(row["inclination_width_deg"], f"{row_context}.inclination_width_deg")
            expected_mean = math.fsum(q_values) / bound_count
            expected_width = quantile(i_values, 0.84) - quantile(i_values, 0.16)
            if not _same_number(mean_q, expected_mean) or not _same_number(width, expected_width):
                _fail("summary_consistency_error", f"{row_context} stored moments do not match the samples")
        if epoch_key == final_epoch and (bound_count != survival_count or not _same_number(bound_fraction, survival_fraction)):
            _fail("summary_consistency_error", f"{row_context} does not match final survival totals")
    if len(minimum_q) < maximum_bound_count:
        _fail("summary_consistency_error", f"{context} minimum-q support is smaller than a bound epoch")

    outcomes = summary["tracer_outcomes"]
    if not isinstance(outcomes, list) or len(outcomes) != tracer_count:
        _fail("summary_support_mismatch", f"{context}.tracer_outcomes must contain every locked tracer")
    expected_tracers = [f"t{index:04d}" for index in range(tracer_count)]
    outcome_minimum_q: list[float] = []
    initial_q_values: list[float] = []
    final_q_values: list[float] = []
    outcome_injections = 0
    outcome_survivors = 0
    seen_tracers: list[str] = []
    outcome_epoch_q: dict[int, list[float]] = {
        epoch: [] for epoch in plan["contract"]["epochs_year"]
    }
    for index, outcome in enumerate(outcomes):
        outcome_context = f"{context}.tracer_outcomes[{index}]"
        if not isinstance(outcome, dict):
            _fail("invalid_summary", f"{outcome_context} must be an object")
        _check_keys(
            outcome,
            {
                "tracer_id", "initial_bound", "initial_q_AU", "minimum_q_AU",
                "injected", "injection_epoch_year", "injection_from_q_AU",
                "injection_to_q_AU", "final_bound", "final_q_AU", "history",
            },
            set(),
            outcome_context,
        )
        tracer_id = outcome["tracer_id"]
        if not isinstance(tracer_id, str):
            _fail("invalid_summary", f"{outcome_context}.tracer_id must be a string")
        seen_tracers.append(tracer_id)
        for name in ("initial_bound", "injected", "final_bound"):
            if not isinstance(outcome[name], bool):
                _fail("invalid_summary", f"{outcome_context}.{name} must be boolean")

        history_raw = outcome["history"]
        locked_epochs = plan["contract"]["epochs_year"]
        if not isinstance(history_raw, list) or len(history_raw) != len(locked_epochs):
            _fail("summary_support_mismatch", f"{outcome_context}.history must cover every locked epoch")
        history: list[tuple[int, bool, float | None]] = []
        for history_index, (expected_epoch, history_row) in enumerate(zip(locked_epochs, history_raw)):
            history_context = f"{outcome_context}.history[{history_index}]"
            if not isinstance(history_row, dict):
                _fail("invalid_summary", f"{history_context} must be an object")
            _check_keys(history_row, {"epoch_year", "bound", "q_AU"}, set(), history_context)
            epoch = _nonnegative_int(history_row["epoch_year"], f"{history_context}.epoch_year")
            if epoch != expected_epoch or not isinstance(history_row["bound"], bool):
                _fail("summary_support_mismatch", f"{history_context} does not match its locked epoch/bound schema")
            if history_row["bound"]:
                q_value = _json_number(history_row["q_AU"], f"{history_context}.q_AU")
                if q_value < 0.0:
                    _fail("invalid_summary", f"{history_context}.q_AU must be nonnegative")
                outcome_epoch_q[epoch].append(q_value)
            else:
                if history_row["q_AU"] is not None:
                    _fail("summary_consistency_error", f"{history_context}.q_AU must be null when unbound")
                q_value = None
            history.append((epoch, history_row["bound"], q_value))

        def bound_q(name: str, bound: bool) -> float | None:
            value = outcome[name]
            if not bound:
                if value is not None:
                    _fail("summary_consistency_error", f"{outcome_context}.{name} must be null when unbound")
                return None
            number = _json_number(value, f"{outcome_context}.{name}")
            if number < 0.0:
                _fail("invalid_summary", f"{outcome_context}.{name} must be nonnegative")
            return number

        initial_q = bound_q("initial_q_AU", outcome["initial_bound"])
        final_q = bound_q("final_q_AU", outcome["final_bound"])
        if outcome["initial_bound"] != history[0][1] or outcome["final_bound"] != history[-1][1]:
            _fail("summary_consistency_error", f"{outcome_context} endpoint bound flags disagree with history")
        for stored, derived, name in (
            (initial_q, history[0][2], "initial_q_AU"),
            (final_q, history[-1][2], "final_q_AU"),
        ):
            if (stored is None) != (derived is None) or (
                stored is not None and derived is not None and not _same_number(stored, derived)
            ):
                _fail("summary_consistency_error", f"{outcome_context}.{name} disagrees with history")
        history_q = [row[2] for row in history if row[1]]
        derived_minimum = min(history_q) if history_q else None
        minimum_value = outcome["minimum_q_AU"]
        if minimum_value is None:
            if derived_minimum is not None:
                _fail("summary_consistency_error", f"{outcome_context} bound history must have a minimum q")
            minimum_number = None
        else:
            minimum_number = _json_number(minimum_value, f"{outcome_context}.minimum_q_AU")
            if minimum_number < 0.0:
                _fail("invalid_summary", f"{outcome_context}.minimum_q_AU must be nonnegative")
            if derived_minimum is None or not _same_number(minimum_number, derived_minimum):
                _fail("summary_consistency_error", f"{outcome_context} minimum q disagrees with history")
            outcome_minimum_q.append(minimum_number)
        if initial_q is not None:
            initial_q_values.append(initial_q)
        if final_q is not None:
            final_q_values.append(final_q)

        derived_injection: tuple[int, float, float] | None = None
        for previous, current in zip(history, history[1:]):
            if (
                previous[1]
                and current[1]
                and previous[2] is not None
                and current[2] is not None
                and previous[2] >= q_threshold
                and current[2] < q_threshold
            ):
                derived_injection = (current[0], previous[2], current[2])
                break
        if outcome["injected"] != (derived_injection is not None):
            _fail("summary_consistency_error", f"{outcome_context} injected flag disagrees with q history")
        if outcome["injected"]:
            injection_epoch = _nonnegative_int(
                outcome["injection_epoch_year"],
                f"{outcome_context}.injection_epoch_year",
            )
            from_q = _json_number(outcome["injection_from_q_AU"], f"{outcome_context}.injection_from_q_AU")
            to_q = _json_number(outcome["injection_to_q_AU"], f"{outcome_context}.injection_to_q_AU")
            if derived_injection is None:
                _fail("summary_consistency_error", f"{outcome_context} lacks a derived injection transition")
            if (
                injection_epoch != derived_injection[0]
                or not _same_number(from_q, derived_injection[1])
                or not _same_number(to_q, derived_injection[2])
            ):
                _fail("summary_consistency_error", f"{outcome_context} injection evidence disagrees with q history")
            outcome_injections += 1
        elif any(
            outcome[name] is not None
            for name in ("injection_epoch_year", "injection_from_q_AU", "injection_to_q_AU")
        ):
            _fail("summary_consistency_error", f"{outcome_context} non-injected tracer has transition evidence")
        outcome_survivors += outcome["final_bound"]

    if seen_tracers != expected_tracers:
        _fail("summary_support_mismatch", f"{context}.tracer_outcomes IDs/order do not match the locked plan")
    for epoch in plan["contract"]["epochs_year"]:
        if sorted(outcome_epoch_q[epoch]) != epochs[str(epoch)]["q_values_AU"]:
            _fail("summary_consistency_error", f"{context} tracer histories disagree with q distribution at epoch {epoch}")
    initial_epoch_q = epochs[str(plan["contract"]["epochs_year"][0])]["q_values_AU"]
    final_epoch_q = epochs[final_epoch]["q_values_AU"]
    if sorted(initial_q_values) != initial_epoch_q or sorted(final_q_values) != final_epoch_q:
        _fail("summary_consistency_error", f"{context} tracer outcomes disagree with endpoint q distributions")
    if sorted(outcome_minimum_q) != minimum_q:
        _fail("summary_consistency_error", f"{context} tracer outcomes disagree with minimum-q distribution")
    if outcome_injections != injection_count or outcome_survivors != survival_count:
        _fail("summary_consistency_error", f"{context} tracer outcomes disagree with event totals")


def _read_member_records(run_root: str | Path, plan: Mapping[str, Any]) -> tuple[dict[tuple[str, str, str], dict[str, Any]], list[dict[str, str]]]:
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    invalid: list[dict[str, str]] = []
    expected_members = {member["member_id"]: member for member in plan["members"]}
    expected_methods = {method["method_id"]: method for method in plan["contract"]["methods"]}
    root = Path(run_root)
    expected_paths = {
        root / member_id / arm / f"{method_id}.json": (member_id, arm, method_id)
        for member_id in expected_members
        for arm in ("control", "source")
        for method_id in expected_methods
    }
    supplied_paths = {
        path
        for path in root.rglob("*.json")
        if len(path.relative_to(root).parts) == 3
        and path.relative_to(root).parts[1] in {"control", "source"}
    }
    for path in sorted(supplied_paths):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
        except EnsembleValidationError as exc:
            invalid.append({"code": exc.code, "message": f"{path}: {exc.message}"})
            continue
        except (OSError, json.JSONDecodeError, RecursionError) as exc:
            invalid.append({"code": "invalid_member_json", "message": f"cannot read valid member JSON from {path}: {exc}"})
            continue
        try:
            if not isinstance(raw, dict):
                _fail("invalid_member_record", f"{path} top-level value must be an object")
            _check_keys(raw, {"schema", "created_utc", "environment", "payload", "payload_sha256"}, set(), f"member record {path}")
            if raw["schema"] != "jx-planet-x-run/v1":
                _fail("schema_mismatch", f"{path} outer schema must be jx-planet-x-run/v1")
            if not isinstance(raw["created_utc"], str) or not raw["created_utc"]:
                _fail("invalid_member_record", f"{path} created_utc must be nonempty")
            environment = raw["environment"]
            if not isinstance(environment, dict):
                _fail("invalid_member_record", f"{path} environment must be an object")
            _check_keys(environment, {"python", "implementation", "platform"}, set(), f"member environment {path}")
            if any(not isinstance(value, str) or not value for value in environment.values()):
                _fail("invalid_member_record", f"{path} environment fields must be nonempty strings")
            payload = raw["payload"]
            if not isinstance(payload, dict):
                _fail("invalid_member_record", f"{path} payload must be an object")
            if payload.get("schema") != MEMBER_SCHEMA:
                _fail("schema_mismatch", f"{path} payload schema must be {MEMBER_SCHEMA}")
            _sha256(raw["payload_sha256"], f"{path} payload_sha256")
            if raw.get("payload_sha256") != sha256_data(payload):
                _fail("member_payload_hash_mismatch", f"{path} payload hash mismatch")
            required = {
                "schema", "plan_sha256", "member_id", "block_id", "replicate_id", "arm", "method_id",
                "independence_group", "method_spec_sha256", "initial_draw_sha256",
                "dynamics_model_sha256", "initial_state_model_sha256",
                "source_model_sha256",
                "relative_initial_state_sha256", "full_initial_state_sha256",
                "trajectory_sha256", "trajectory_semantic_sha256", "validity_sha256",
                "validity_semantic_sha256", "validity", "validity_passed",
                "summary", "summary_sha256", "evidence_class",
            }
            _check_keys(payload, required, set(), f"member record {path}")
            relative = path.relative_to(root)
            path_identity = (relative.parts[0], relative.parts[1], Path(relative.parts[2]).stem)
            if path_identity != (payload["member_id"], payload["arm"], payload["method_id"]):
                _fail("member_path_mismatch", f"{path} path does not match its payload identity")
            if payload["plan_sha256"] != plan["plan_sha256"]:
                _fail("member_plan_mismatch", f"{path} references a different plan")
            member_id = payload["member_id"]
            arm = payload["arm"]
            method_id = payload["method_id"]
            if member_id not in expected_members or method_id not in expected_methods or arm not in {"control", "source"}:
                _fail("unexpected_member_record", f"{path} has unplanned identity")
            expected_member = expected_members[member_id]
            expected_method = expected_methods[method_id]
            if payload["block_id"] != expected_member["block_id"] or payload["replicate_id"] != expected_member["replicate_id"]:
                _fail("member_identity_mismatch", f"{path} block/replicate mismatch")
            if payload["initial_draw_sha256"] != expected_member["initial_draw_sha256"]:
                _fail("draw_hash_mismatch", f"{path} draw hash mismatch")
            if payload["independence_group"] != expected_method["independence_group"]:
                _fail("method_identity_mismatch", f"{path} independence-group mismatch")
            if payload["method_spec_sha256"] != sha256_data(expected_method):
                _fail("method_identity_mismatch", f"{path} method specification hash mismatch")
            for name in ("dynamics_model_sha256", "initial_state_model_sha256", "source_model_sha256"):
                if payload[name] != plan["contract"][name]:
                    _fail("model_identity_mismatch", f"{path} {name} does not match the locked contract")
            if payload["evidence_class"] != "MODEL_OUTPUT":
                _fail("invalid_evidence_class", f"{path} evidence class must be MODEL_OUTPUT")
            if not isinstance(payload["validity_passed"], bool):
                _fail("invalid_validity_record", f"{path} validity_passed must be boolean")
            for name in (
                "plan_sha256", "method_spec_sha256", "initial_draw_sha256",
                "dynamics_model_sha256", "initial_state_model_sha256", "source_model_sha256",
                "relative_initial_state_sha256", "full_initial_state_sha256",
                "trajectory_sha256", "trajectory_semantic_sha256", "validity_sha256",
                "validity_semantic_sha256", "summary_sha256",
            ):
                _sha256(payload[name], f"{path} {name}")
            if payload["validity_semantic_sha256"] != sha256_data(payload["validity"]):
                _fail("validity_hash_mismatch", f"{path} embedded validity hash mismatch")
            _validate_validity_record(
                payload["validity"],
                plan,
                expected_member,
                arm,
                expected_method,
                payload["trajectory_sha256"],
            )
            if payload["validity_passed"] != payload["validity"]["passed"]:
                _fail("invalid_validity_record", f"{path} validity flag disagrees with embedded record")
            if payload["summary_sha256"] != sha256_data(payload["summary"]):
                _fail("summary_hash_mismatch", f"{path} summary hash mismatch")
            _validate_member_summary(payload["summary"], plan, f"member summary {path}")
            key = (member_id, arm, method_id)
            if key in records:
                _fail("duplicate_member_record", f"duplicate member record {key}")
            records[key] = payload
        except EnsembleValidationError as exc:
            invalid.append({"code": exc.code, "message": exc.message})
    unexpected_paths = supplied_paths - expected_paths.keys()
    for path in sorted(unexpected_paths):
        if not any(row["message"].startswith(str(path)) for row in invalid):
            invalid.append({"code": "unexpected_member_record", "message": f"unexpected member-record path {path}"})

    for member_id in expected_members:
        member_records = [record for (record_member, _, _), record in records.items() if record_member == member_id]
        relative_hashes = {record["relative_initial_state_sha256"] for record in member_records}
        if len(relative_hashes) > 1:
            invalid.append({"code": "initial_state_pair_mismatch", "message": f"{member_id} arms/methods do not share a relative initial state"})
        for arm in ("control", "source"):
            full_hashes = {
                record["full_initial_state_sha256"]
                for (record_member, record_arm, _), record in records.items()
                if record_member == member_id and record_arm == arm
            }
            if len(full_hashes) > 1:
                invalid.append({"code": "initial_state_method_mismatch", "message": f"{member_id}/{arm} methods do not share a full initial state"})
    return records, invalid


def _aggregate(records: Sequence[Mapping[str, Any]], epochs: Sequence[int]) -> dict[str, Any]:
    if not records:
        _fail("empty_aggregate", "cannot aggregate an empty record set")
    epoch_output: dict[str, Any] = {}
    for epoch in epochs:
        summaries = [record["summary"]["epochs"][str(epoch)] for record in records]
        sample_count = sum(summary["sample_count"] for summary in summaries)
        bound_count = sum(summary["bound_count"] for summary in summaries)
        low_q_count = sum(summary["low_q_count"] for summary in summaries)
        q_values = [value for summary in summaries for value in summary["q_values_AU"]]
        i_values = [value for summary in summaries for value in summary["i_values_deg"]]
        epoch_output[str(epoch)] = {
            "sample_count": sample_count,
            "bound_count": bound_count,
            "low_q_count": low_q_count,
            "bound_fraction": bound_count / sample_count,
            "low_q_fraction": low_q_count / sample_count,
            "mean_q_AU": math.fsum(q_values) / len(q_values) if q_values else None,
            "inclination_width_deg": quantile(i_values, 0.84) - quantile(i_values, 0.16) if i_values else None,
            "q_values_AU": sorted(q_values),
            "i_values_deg": sorted(i_values),
        }
    total_tracers = sum(record["summary"]["tracer_count"] for record in records)
    injection_count = sum(record["summary"]["injection_count"] for record in records)
    survival_count = sum(record["summary"]["survival_count"] for record in records)
    minimum_q_values = sorted(
        value
        for record in records
        for value in record["summary"]["minimum_q_values_AU"]
    )
    return {
        "member_count": len(records),
        "tracer_count": total_tracers,
        "injection_fraction": injection_count / total_tracers,
        "survival_fraction": survival_count / total_tracers,
        "minimum_q_values_AU": minimum_q_values,
        "epochs": epoch_output,
    }


def _compare_aggregates(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    epochs: Sequence[int],
    thresholds: Mapping[str, str],
) -> dict[str, Any]:
    values = {
        "low_q_fraction": 0.0,
        "injection_fraction": abs(first["injection_fraction"] - second["injection_fraction"]),
        "survival_fraction": abs(first["survival_fraction"] - second["survival_fraction"]),
        "mean_q_AU": 0.0,
        "inclination_width_deg": 0.0,
        "wasserstein_q_AU": 0.0,
        "wasserstein_i_deg": 0.0,
        "wasserstein_min_q_AU": 0.0,
    }
    undefined: list[str] = []
    worst_epoch: dict[str, int | None] = {name: None for name in values}
    first_minimum_q = first["minimum_q_values_AU"]
    second_minimum_q = second["minimum_q_values_AU"]
    if first_minimum_q and second_minimum_q:
        values["wasserstein_min_q_AU"] = wasserstein_1d(first_minimum_q, second_minimum_q)
    else:
        undefined.append("wasserstein_min_q_AU@all")
    for epoch in epochs:
        left = first["epochs"][str(epoch)]
        right = second["epochs"][str(epoch)]
        candidates: dict[str, float] = {
            "low_q_fraction": abs(left["low_q_fraction"] - right["low_q_fraction"]),
        }
        if left["mean_q_AU"] is None or right["mean_q_AU"] is None:
            undefined.extend([f"mean_q_AU@{epoch}", f"wasserstein_q_AU@{epoch}"])
        else:
            candidates["mean_q_AU"] = abs(left["mean_q_AU"] - right["mean_q_AU"])
            candidates["wasserstein_q_AU"] = wasserstein_1d(left["q_values_AU"], right["q_values_AU"])
        if left["inclination_width_deg"] is None or right["inclination_width_deg"] is None:
            undefined.extend([f"inclination_width_deg@{epoch}", f"wasserstein_i_deg@{epoch}"])
        else:
            candidates["inclination_width_deg"] = abs(left["inclination_width_deg"] - right["inclination_width_deg"])
            candidates["wasserstein_i_deg"] = wasserstein_1d(left["i_values_deg"], right["i_values_deg"])
        for name, value in candidates.items():
            if value > values[name]:
                values[name] = value
                worst_epoch[name] = epoch
    metrics = {
        name: {
            "value": value,
            "threshold": float(thresholds.get(name, thresholds["wasserstein_q_AU"])),
            "passed": value <= float(thresholds.get(name, thresholds["wasserstein_q_AU"])) and not any(item.startswith(f"{name}@") for item in undefined),
            "worst_epoch_year": worst_epoch[name],
        }
        for name, value in values.items()
    }
    return {"metrics": metrics, "undefined": sorted(set(undefined)), "passed": all(row["passed"] for row in metrics.values())}


def _distribution_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "minimum": None, "q16": None, "median": None, "q84": None, "maximum": None}
    return {
        "count": len(values),
        "minimum": values[0],
        "q16": quantile(values, 0.16),
        "median": quantile(values, 0.50),
        "q84": quantile(values, 0.84),
        "maximum": values[-1],
    }


def _public_aggregate(method_id: str, arm: str, aggregate: Mapping[str, Any]) -> dict[str, Any]:
    epoch_rows: list[dict[str, Any]] = []
    for epoch, row in sorted(aggregate["epochs"].items(), key=lambda item: int(item[0])):
        epoch_rows.append(
            {
                "epoch_year": int(epoch),
                "sample_count": row["sample_count"],
                "bound_count": row["bound_count"],
                "low_q_count": row["low_q_count"],
                "bound_fraction": row["bound_fraction"],
                "low_q_fraction": row["low_q_fraction"],
                "mean_q_AU": row["mean_q_AU"],
                "inclination_width_deg": row["inclination_width_deg"],
                "q_distribution_AU": _distribution_summary(row["q_values_AU"]),
                "inclination_distribution_deg": _distribution_summary(row["i_values_deg"]),
            }
        )
    return {
        "method_id": method_id,
        "arm": arm,
        "member_count": aggregate["member_count"],
        "tracer_count": aggregate["tracer_count"],
        "injection_fraction": aggregate["injection_fraction"],
        "survival_fraction": aggregate["survival_fraction"],
        "minimum_q_distribution_AU": _distribution_summary(aggregate["minimum_q_values_AU"]),
        "epochs": epoch_rows,
    }


def _source_control_distribution_comparison(
    method_id: str,
    control: Mapping[str, Any],
    source: Mapping[str, Any],
    epochs: Sequence[int],
) -> dict[str, Any]:
    minimum_q_w1 = None
    if control["minimum_q_values_AU"] and source["minimum_q_values_AU"]:
        minimum_q_w1 = wasserstein_1d(control["minimum_q_values_AU"], source["minimum_q_values_AU"])
    epoch_rows: list[dict[str, Any]] = []
    for epoch in epochs:
        left = control["epochs"][str(epoch)]
        right = source["epochs"][str(epoch)]
        q_w1 = i_w1 = None
        if left["q_values_AU"] and right["q_values_AU"]:
            q_w1 = wasserstein_1d(left["q_values_AU"], right["q_values_AU"])
            i_w1 = wasserstein_1d(left["i_values_deg"], right["i_values_deg"])
        epoch_rows.append(
            {
                "epoch_year": epoch,
                "source_minus_control_bound_fraction": right["bound_fraction"] - left["bound_fraction"],
                "source_minus_control_low_q_fraction": right["low_q_fraction"] - left["low_q_fraction"],
                "source_minus_control_mean_q_AU": (
                    None
                    if left["mean_q_AU"] is None or right["mean_q_AU"] is None
                    else right["mean_q_AU"] - left["mean_q_AU"]
                ),
                "source_minus_control_inclination_width_deg": (
                    None
                    if left["inclination_width_deg"] is None or right["inclination_width_deg"] is None
                    else right["inclination_width_deg"] - left["inclination_width_deg"]
                ),
                "wasserstein_q_AU": q_w1,
                "wasserstein_i_deg": i_w1,
            }
        )
    return {
        "method_id": method_id,
        "source_minus_control_injection_fraction": source["injection_fraction"] - control["injection_fraction"],
        "source_minus_control_survival_fraction": source["survival_fraction"] - control["survival_fraction"],
        "wasserstein_minimum_q_AU": minimum_q_w1,
        "epochs": epoch_rows,
        "interpretation": "descriptive model-output distances; Wasserstein values are not p-values",
    }


def _primary_value(summary: Mapping[str, Any], endpoint: str, final_epoch: int) -> float | None:
    if endpoint == "injection_fraction":
        return float(summary["injection_fraction"])
    if endpoint == "survival_fraction":
        return float(summary["survival_fraction"])
    final = summary["epochs"][str(final_epoch)]
    if endpoint == "final_low_q_fraction":
        return float(final["low_q_fraction"])
    if endpoint == "final_mean_q_AU":
        return None if final["mean_q_AU"] is None else float(final["mean_q_AU"])
    _fail("invalid_endpoint", f"unknown endpoint {endpoint}")


def _bootstrap_effect(
    plan: Mapping[str, Any],
    method_id: str,
    paired: Mapping[str, float],
) -> dict[str, Any]:
    by_block: dict[str, list[tuple[str, float]]] = defaultdict(list)
    member_lookup = {member["member_id"]: member for member in plan["members"]}
    for member_id, value in paired.items():
        by_block[member_lookup[member_id]["block_id"]].append((member_id, value))
    repetitions = plan["contract"]["inference"]["bootstrap_repetitions"]
    # All numerical methods use the same deterministic resample indices so
    # method-to-method effect comparisons preserve the locked pairing.
    key = hashlib.sha256(f"{plan['plan_sha256']}\x1fbootstrap".encode("utf-8")).digest()
    estimates: list[float] = []
    for repetition in range(repetitions):
        resampled: list[float] = []
        for block_id in sorted(by_block):
            rows = sorted(by_block[block_id])
            for position in range(len(rows)):
                u = _open_uniform(key, repetition, block_id, position)
                index = min(int(u * len(rows)), len(rows) - 1)
                resampled.append(rows[index][1])
        estimates.append(math.fsum(resampled) / len(resampled))
    confidence = float(plan["contract"]["inference"]["confidence_level"])
    alpha = (1.0 - confidence) / 2.0
    values = list(paired.values())
    point = math.fsum(values) / len(values)
    lower = quantile(estimates, alpha)
    upper = quantile(estimates, 1.0 - alpha)
    null_margin = float(plan["contract"]["inference"]["null_equivalence_margin"])
    material = float(plan["contract"]["inference"]["minimum_material_effect"])
    if lower >= material:
        classification = "MATERIAL_POSITIVE"
    elif upper <= -material:
        classification = "MATERIAL_NEGATIVE"
    elif lower >= -null_margin and upper <= null_margin:
        classification = "PRACTICALLY_EQUIVALENT"
    else:
        classification = "INCONCLUSIVE"
    return {
        "point_estimate": point,
        "confidence_interval": [lower, upper],
        "confidence_level": confidence,
        "bootstrap_repetitions": repetitions,
        "classification": classification,
        "independent_unit": "paired outer replicate, stratified by seed block",
    }


def _invalid_result(code: str, message: str, plan_sha256: str | None = None) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "verdict": EnsembleVerdict.INVALID.value,
        "claim_decision": "INVALID",
        "effect_classification": "UNAVAILABLE",
        "plan_sha256": plan_sha256,
        "invalid_reasons": [{"code": code, "message": message}],
        "blocked_reasons": [],
        "evidence_class": "MODEL_OUTPUT",
        "scientific_scope": "invalid ensemble record; no scientific inference permitted",
        "nonclaims": list(NONCLAIMS),
    }


def _finalize_ensemble_validation(plan_path: str | Path, run_root: str | Path) -> dict[str, Any]:
    """Finalize a locked ensemble without ever promoting it to a detection."""
    try:
        plan = load_ensemble_plan(plan_path)
    except EnsembleValidationError as exc:
        return _invalid_result(exc.code, exc.message)
    records, invalid = _read_member_records(run_root, plan)
    if invalid:
        result = _invalid_result(invalid[0]["code"], invalid[0]["message"], plan["plan_sha256"])
        result["invalid_reasons"] = invalid
        return result

    contract = plan["contract"]
    methods = [method["method_id"] for method in contract["methods"]]
    method_specs = {method["method_id"]: method for method in contract["methods"]}
    expected = {
        (member["member_id"], arm, method_id)
        for member in plan["members"]
        for arm in ("control", "source")
        for method_id in methods
    }
    supplied = set(records)
    extra = supplied - expected
    if extra:
        return _invalid_result("unexpected_member_record", f"unexpected run records: {sorted(extra)[:5]}", plan["plan_sha256"])
    missing = expected - supplied
    blocked: list[dict[str, str]] = []
    if missing:
        blocked.append({"code": "missing_required_runs", "message": f"missing {len(missing)} planned member runs"})
    failed_validity = [key for key, record in records.items() if not record["validity_passed"]]
    if failed_validity:
        return _invalid_result(
            "integrator_validity_failed",
            f"{len(failed_validity)} member runs failed hard validity checks",
            plan["plan_sha256"],
        )

    gates = contract["gates"]
    sample_gates = {
        "blocks": {"value": len(contract["seed_blocks"]), "threshold": gates["minimum_blocks"]},
        "replicates_per_block": {"value": contract["replicates_per_block"], "threshold": gates["minimum_replicates_per_block"]},
        "tracers_per_replicate": {"value": contract["tracers_per_replicate"], "threshold": gates["minimum_tracers_per_replicate"]},
        "methods": {"value": len(methods), "threshold": gates["minimum_methods"]},
        "independence_groups": {
            "value": len({method["independence_group"] for method in contract["methods"]}),
            "threshold": gates["minimum_independence_groups"],
        },
    }
    for name, row in sample_gates.items():
        row["passed"] = row["value"] >= row["threshold"]
        if not row["passed"]:
            blocked.append({"code": f"insufficient_{name}", "message": f"{name} is below its locked minimum"})
    group_counts: dict[str, int] = defaultdict(int)
    for method in contract["methods"]:
        group_counts[method["independence_group"]] += 1
    within_group_repeat = any(count >= 2 for count in group_counts.values())
    if gates["require_within_group_repeat"] and not within_group_repeat:
        blocked.append({"code": "precision_repeat_missing", "message": "no independence group contains a locked precision/repeat pair"})

    if missing:
        return {
            "schema": RESULT_SCHEMA,
            "verdict": EnsembleVerdict.BLOCKED.value,
            "claim_decision": "SCREENING_ONLY",
            "effect_classification": "UNAVAILABLE",
            "plan_sha256": plan["plan_sha256"],
            "invalid_reasons": [],
            "blocked_reasons": blocked,
            "completeness": {"expected": len(expected), "supplied": len(supplied), "missing": len(missing)},
            "sample_size_gates": sample_gates,
            "evidence_class": "MODEL_OUTPUT",
            "scientific_scope": "incomplete ensemble; no effect inference permitted",
            "nonclaims": list(NONCLAIMS),
        }

    epochs = contract["epochs_year"]
    aggregates: dict[tuple[str, str], dict[str, Any]] = {}
    by_block: dict[tuple[str, str, str], dict[str, Any]] = {}
    for method_id in methods:
        for arm in ("control", "source"):
            selected = [record for (member, record_arm, method), record in records.items() if record_arm == arm and method == method_id]
            aggregates[(method_id, arm)] = _aggregate(selected, epochs)
            for block_id in sorted({record["block_id"] for record in selected}):
                block_records = [record for record in selected if record["block_id"] == block_id]
                by_block[(method_id, arm, block_id)] = _aggregate(block_records, epochs)

    population_summaries = [
        _public_aggregate(method_id, arm, aggregates[(method_id, arm)])
        for method_id in methods
        for arm in ("control", "source")
    ]
    source_control_distributions = [
        _source_control_distribution_comparison(
            method_id,
            aggregates[(method_id, "control")],
            aggregates[(method_id, "source")],
            epochs,
        )
        for method_id in methods
    ]

    minimum_bound = gates["minimum_bound_samples_per_epoch"]
    bound_sample_gates: list[dict[str, Any]] = []
    for (method_id, arm), aggregate in sorted(aggregates.items()):
        minimum = min(row["bound_count"] for row in aggregate["epochs"].values())
        passed = minimum >= minimum_bound
        bound_sample_gates.append({"method_id": method_id, "arm": arm, "minimum": minimum, "threshold": minimum_bound, "passed": passed})
        if not passed:
            blocked.append({"code": "insufficient_bound_population", "message": f"{method_id}/{arm} has too few bound samples for conditional distributions"})

    method_comparisons: list[dict[str, Any]] = []
    numerical_invalid: list[dict[str, str]] = []
    for first_index, first_method in enumerate(methods):
        for second_method in methods[first_index + 1:]:
            same_group = (
                method_specs[first_method]["independence_group"]
                == method_specs[second_method]["independence_group"]
            )
            comparison_type = "WITHIN_GROUP_PRECISION" if same_group else "INDEPENDENT_METHOD"
            for arm in ("control", "source"):
                comparison = _compare_aggregates(
                    aggregates[(first_method, arm)],
                    aggregates[(second_method, arm)],
                    epochs,
                    gates["method_equivalence"],
                )
                comparison.update(
                    {
                        "first_method": first_method,
                        "second_method": second_method,
                        "arm": arm,
                        "comparison_type": comparison_type,
                    }
                )
                method_comparisons.append(comparison)
                if not comparison["passed"]:
                    if comparison["undefined"]:
                        blocked.append(
                            {
                                "code": "conditional_method_comparison_unavailable",
                                "message": f"{first_method} and {second_method} lack a required conditional distribution for {arm}",
                            }
                        )
                        continue
                    reason = {
                        "code": "precision_nonconvergence" if same_group else "independent_method_disagreement",
                        "message": f"{first_method} and {second_method} disagree for {arm}",
                    }
                    if same_group:
                        numerical_invalid.append(reason)
                    else:
                        blocked.append(reason)

    if numerical_invalid:
        result = _invalid_result(
            numerical_invalid[0]["code"],
            numerical_invalid[0]["message"],
            plan["plan_sha256"],
        )
        result.update(
            {
                "invalid_reasons": numerical_invalid,
                "blocked_reasons": blocked,
                "contract_sha256": plan["contract_sha256"],
                "registration_status": contract["registration_status"],
                "completeness": {"expected": len(expected), "supplied": len(supplied), "missing": 0},
                "sample_size_gates": sample_gates,
                "bound_sample_gates": bound_sample_gates,
                "population_summaries": population_summaries,
                "source_control_distributions": source_control_distributions,
                "method_comparisons": method_comparisons,
            }
        )
        return result

    repeat_comparisons: list[dict[str, Any]] = []
    block_ids = sorted({member["block_id"] for member in plan["members"]})
    for method_id in methods:
        for arm in ("control", "source"):
            for first_index, first_block in enumerate(block_ids):
                for second_block in block_ids[first_index + 1:]:
                    comparison = _compare_aggregates(
                        by_block[(method_id, arm, first_block)],
                        by_block[(method_id, arm, second_block)],
                        epochs,
                        gates["repeat_equivalence"],
                    )
                    comparison.update({"method_id": method_id, "arm": arm, "first_block": first_block, "second_block": second_block})
                    repeat_comparisons.append(comparison)
                    if not comparison["passed"]:
                        blocked.append({"code": "repeat_block_disagreement", "message": f"{method_id}/{arm} seed blocks disagree"})

    endpoint = contract["inference"]["primary_endpoint"]
    final_epoch = epochs[-1]
    effects: dict[str, Any] = {}
    paired_effects: dict[str, dict[str, float]] = {}
    for method_id in methods:
        paired: dict[str, float] = {}
        for member in plan["members"]:
            member_id = member["member_id"]
            control = _primary_value(records[(member_id, "control", method_id)]["summary"], endpoint, final_epoch)
            source = _primary_value(records[(member_id, "source", method_id)]["summary"], endpoint, final_epoch)
            if control is None or source is None:
                blocked.append({"code": "undefined_primary_endpoint", "message": f"{method_id}/{member_id} primary endpoint is undefined"})
                continue
            paired[member_id] = source - control
        if len(paired) == len(plan["members"]):
            paired_effects[method_id] = paired
            effects[method_id] = _bootstrap_effect(plan, method_id, paired)

    classifications = {row["classification"] for row in effects.values()}
    effect_classification = next(iter(classifications)) if len(classifications) == 1 else "INCONCLUSIVE"
    if not effects or "INCONCLUSIVE" in classifications:
        blocked.append({"code": "effect_inconclusive", "message": "primary source/control effect confidence interval is inconclusive"})
    effect_method_comparisons: list[dict[str, Any]] = []
    effect_invalid: list[dict[str, str]] = []
    effect_threshold = float(gates["max_primary_effect_method_disagreement"])
    for first_index, first_method in enumerate(methods):
        for second_method in methods[first_index + 1:]:
            if first_method not in effects or second_method not in effects:
                continue
            same_group = (
                method_specs[first_method]["independence_group"]
                == method_specs[second_method]["independence_group"]
            )
            difference = abs(effects[first_method]["point_estimate"] - effects[second_method]["point_estimate"])
            classification_match = effects[first_method]["classification"] == effects[second_method]["classification"]
            passed = difference <= effect_threshold and classification_match
            effect_method_comparisons.append(
                {
                    "first_method": first_method,
                    "second_method": second_method,
                    "comparison_type": "WITHIN_GROUP_PRECISION" if same_group else "INDEPENDENT_METHOD",
                    "absolute_point_difference": difference,
                    "threshold": effect_threshold,
                    "classification_match": classification_match,
                    "passed": passed,
                }
            )
            if not passed:
                if not classification_match:
                    code = "precision_effect_classification_conflict" if same_group else "effect_classification_conflict"
                else:
                    code = "precision_effect_nonconvergence" if same_group else "independent_effect_disagreement"
                reason = {
                    "code": code,
                    "message": f"{first_method} and {second_method} disagree on the primary source/control effect",
                }
                if same_group:
                    effect_invalid.append(reason)
                else:
                    blocked.append(reason)

    effect_repeat_comparisons: list[dict[str, Any]] = []
    repeat_effect_threshold = float(gates["max_primary_effect_repeat_disagreement"])
    member_lookup = {member["member_id"]: member for member in plan["members"]}
    for method_id, paired in paired_effects.items():
        block_points: dict[str, float] = {}
        for block_id in block_ids:
            values = [
                value
                for member_id, value in paired.items()
                if member_lookup[member_id]["block_id"] == block_id
            ]
            if values:
                block_points[block_id] = math.fsum(values) / len(values)
        for first_index, first_block in enumerate(block_ids):
            for second_block in block_ids[first_index + 1:]:
                difference = abs(block_points[first_block] - block_points[second_block])
                passed = difference <= repeat_effect_threshold
                effect_repeat_comparisons.append(
                    {
                        "method_id": method_id,
                        "first_block": first_block,
                        "second_block": second_block,
                        "absolute_point_difference": difference,
                        "threshold": repeat_effect_threshold,
                        "passed": passed,
                    }
                )
                if not passed:
                    blocked.append(
                        {
                            "code": "repeat_effect_disagreement",
                            "message": f"{method_id} source/control effect disagrees across {first_block} and {second_block}",
                        }
                    )

    if effect_invalid:
        result = _invalid_result(
            effect_invalid[0]["code"],
            effect_invalid[0]["message"],
            plan["plan_sha256"],
        )
        result.update(
            {
                "invalid_reasons": effect_invalid,
                "blocked_reasons": blocked,
                "contract_sha256": plan["contract_sha256"],
                "registration_status": contract["registration_status"],
                "population_summaries": population_summaries,
                "source_control_distributions": source_control_distributions,
                "method_comparisons": method_comparisons,
                "repeat_comparisons": repeat_comparisons,
                "source_control_effects": effects,
                "primary_effect_method_comparisons": effect_method_comparisons,
                "primary_effect_repeat_comparisons": effect_repeat_comparisons,
            }
        )
        return result
    effect_points = [row["point_estimate"] for row in effects.values()]
    max_effect_disagreement = max(effect_points) - min(effect_points) if effect_points else None
    effect_gate = {
        "value": max_effect_disagreement,
        "threshold": float(gates["max_primary_effect_method_disagreement"]),
        "passed": max_effect_disagreement is not None and max_effect_disagreement <= float(gates["max_primary_effect_method_disagreement"]),
    }
    if not effect_gate["passed"]:
        blocked.append({"code": "primary_effect_method_disagreement", "message": "methods disagree on primary-effect magnitude"})

    conflict_codes = {
        "independent_method_disagreement",
        "independent_effect_disagreement",
        "effect_classification_conflict",
    }
    claim_decision = "CONFLICT" if any(row["code"] in conflict_codes for row in blocked) else "SCREENING_ONLY"
    verdict = EnsembleVerdict.BLOCKED.value if blocked else EnsembleVerdict.PASSED.value
    return {
        "schema": RESULT_SCHEMA,
        "verdict": verdict,
        "claim_decision": claim_decision,
        "effect_classification": effect_classification,
        "plan_sha256": plan["plan_sha256"],
        "contract_sha256": plan["contract_sha256"],
        "registration_status": contract["registration_status"],
        "invalid_reasons": [],
        "blocked_reasons": blocked,
        "completeness": {"expected": len(expected), "supplied": len(supplied), "missing": 0},
        "sample_size_gates": sample_gates,
        "bound_sample_gates": bound_sample_gates,
        "population_summaries": population_summaries,
        "source_control_distributions": source_control_distributions,
        "method_comparisons": method_comparisons,
        "repeat_comparisons": repeat_comparisons,
        "primary_endpoint": endpoint,
        "source_control_effects": effects,
        "primary_effect_method_gate": effect_gate,
        "primary_effect_method_comparisons": effect_method_comparisons,
        "primary_effect_repeat_comparisons": effect_repeat_comparisons,
        "evidence_class": "MODEL_OUTPUT",
        "scientific_scope": "numerical population validation only; PASSED remains SCREENING_ONLY",
        "nonclaims": list(NONCLAIMS),
    }


def finalize_ensemble_validation(plan_path: str | Path, run_root: str | Path) -> dict[str, Any]:
    """Return a machine verdict; controlled validation failures never escape."""
    try:
        return _finalize_ensemble_validation(plan_path, run_root)
    except EnsembleValidationError as exc:
        return _invalid_result(exc.code, exc.message)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, OSError, RecursionError) as exc:
        return _invalid_result(
            "internal_validation_failure",
            f"ensemble finalization failed closed: {type(exc).__name__}: {exc}",
        )


def example_contract() -> dict[str, Any]:
    """Return a documented engineering template, not universal thresholds."""
    method_thresholds = {
        "low_q_fraction": "0.01",
        "injection_fraction": "0.01",
        "survival_fraction": "0.01",
        "mean_q_AU": "0.05",
        "inclination_width_deg": "0.10",
        "wasserstein_q_AU": "0.05",
        "wasserstein_i_deg": "0.10",
        "wasserstein_min_q_AU": "0.05",
    }
    repeat_thresholds = {
        **method_thresholds,
        "low_q_fraction": "0.02",
        "injection_fraction": "0.02",
        "survival_fraction": "0.02",
        "mean_q_AU": "0.10",
        "wasserstein_q_AU": "0.10",
        "wasserstein_min_q_AU": "0.10",
    }
    return {
        "schema": SPEC_SCHEMA,
        "experiment_id": "replace-with-preregistered-id",
        "purpose": "Compare paired source/control chaotic-population effects",
        "registration_status": "EXPLORATORY",
        "registration_reference": "",
        "evidence_class": "MODEL_OUTPUT",
        "dynamics_model_sha256": sha256_data({"placeholder": "replace with governing dynamics artifact"}),
        "initial_state_model_sha256": sha256_data({"placeholder": "replace with initial-state builder artifact"}),
        "source_model_sha256": sha256_data({"placeholder": "replace with source/control model artifact"}),
        "seed_blocks": ["replace-seed-block-a", "replace-seed-block-b"],
        "replicates_per_block": 32,
        "tracers_per_replicate": 64,
        "epochs_year": [0, 1000, 10000, 100000],
        "duration_years": 100000,
        "frame": "declare frame",
        "origin": "declare origin",
        "units": "AU, yr, solar mass",
        "q_threshold_AU": "30",
        "factors": [
            {"name": "source_phase_deg", "scope": "replicate", "distribution": "phase", "origin": "0", "period": "360"},
            {"name": "tracer_phase_deg", "scope": "tracer", "distribution": "phase", "origin": "0", "period": "360"},
        ],
        "gaussian_blocks": [],
        "methods": [
            {
                "method_id": "ias15-primary",
                "implementation": "REBOUND IAS15",
                "version": "4.4.11",
                "independence_group": "rebound-ias15",
                "settings": {"epsilon": "1e-12"},
            },
            {
                "method_id": "ias15-tighter",
                "implementation": "REBOUND IAS15",
                "version": "4.4.11",
                "independence_group": "rebound-ias15",
                "settings": {"epsilon": "1e-14"},
            },
            {
                "method_id": "independent-method",
                "implementation": "replace with independent algorithm/code path",
                "version": "declare version",
                "independence_group": "replace-independent-group",
                "settings": {},
            },
        ],
        "gates": {
            "minimum_blocks": 2,
            "minimum_replicates_per_block": 32,
            "minimum_tracers_per_replicate": 64,
            "minimum_methods": 3,
            "minimum_independence_groups": 2,
            "require_within_group_repeat": True,
            "minimum_bound_samples_per_epoch": 128,
            "method_equivalence": method_thresholds,
            "repeat_equivalence": repeat_thresholds,
            "max_primary_effect_method_disagreement": "0.02",
            "max_primary_effect_repeat_disagreement": "0.01",
        },
        "inference": {
            "primary_endpoint": "injection_fraction",
            "confidence_level": "0.95",
            "bootstrap_repetitions": 9999,
            "null_equivalence_margin": "0.005",
            "minimum_material_effect": "0.02",
        },
        "power_plan": "Replace with a preregistered clustered-power justification; no optional stopping.",
    }


def write_example_contract(output_path: str | Path) -> dict[str, Any]:
    contract = example_contract()
    _atomic_json(output_path, contract)
    return contract
