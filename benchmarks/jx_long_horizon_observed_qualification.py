#!/usr/bin/env python3
"""Prepare the bounded observed-input 1,000-year qualification protocol.

This module deliberately has no execution mode.  ``--prepare-only`` verifies
the retained input artifacts, constructs the exact 108-state inventory, binds
the current numerical source files, and emits a content-addressed protocol.
Importing the public long-horizon package currently loads the runner and
initial-condition modules (and their NumPy dependency), so the protocol says
that explicitly.  Preparation never calls those entry points and never
initializes REBOUND, CuPy, CUDA, or a worker pool.

The protocol is an engineering qualification commitment, not a scientific
result.  In particular, preparing it does not authorize a long run and does
not establish an orbit, detection, exclusion, population, or production-use
claim.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Sequence
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

from jxplanetx.long_horizon.science_inputs import (
    DETACHED_TNO_SCREEN_V1,
    DETACHED_TNO_SENSITIVITY_V1,
    SCATTERING_DISK_CONTROL_V1,
    SCATTERING_DISK_SENSITIVITY_V1,
    SCIENCE_INPUTS_CONTENT_SHA256,
    materialize_perturbation_ensemble,
)


SCHEMA = "jx.long-horizon.observed-qualification.protocol.v1"
PROTOCOL_ID = "jx.long-horizon.observed-qualification.1000y-108.v1"
PROTOCOL_STATUS = "PREPARED_NOT_EXECUTED"
QUALIFICATION_CLASSIFICATION = (
    "BOUNDED_NUMERICAL_ENGINEERING_QUALIFICATION_ONLY"
)
CANONICAL_ARCHITECTURE_ID = "jxplanetx.long-horizon.restricted-wh.v1"
CANONICAL_FAMILY = "RESTRICTED_WISDOM_HOLMAN"

HORIZON_YEARS = 1_000
SPLIT_EPOCH_YEARS = 500
OUTPUT_CADENCE_YEARS = 1
STATE_COUNT = 108
NOMINAL_STATE_COUNT = 12
SENSITIVITY_STATE_COUNT = 96

JX_CANDIDATE_STEP_YEARS = 0.0625
JX_REFINEMENT_STEP_YEARS = 0.03125
REQUIRED_REBOUND_VERSION = "5.1.1"
REBOUND_IAS15_EPSILON = 1.0e-15

EXPECTED_SCIENCE_INPUTS_SHA256 = (
    "ae70d80118860fc20a0da8328bd7f62cf33ab579e2dcc7a11c67d338f5e58a7e"
)
EXPECTED_POPULATION_SHA256 = {
    "jpl-observed-detached-tno-v1": (
        "b3cdcddd91de97138f70f1ddbe3f7ab09d60cc3e2708143a5a3c6cf1ebc7d7b0"
    ),
    "jpl-observed-scattering-control-v1": (
        "16e6d45e69e135642b8ac4e3710adb47e1e8e50a4183e417cb017722b0abdc53"
    ),
}
EXPECTED_ENSEMBLE_SPEC_SHA256 = {
    "detached-tno-sensitivity-v1": (
        "60726a47baaecdfe45353bdd321bc886dded65ad1e675952542bfe45ee7b8b04"
    ),
    "scattering-disk-sensitivity-v1": (
        "f3570dcdce7bff21618b5c2a58befe7301397d639f02a6c438bffd9591ebe67b"
    ),
}
EXPECTED_MATERIALIZED_ENSEMBLE_SHA256 = {
    "detached-tno-sensitivity-v1": (
        "c313549c046fccd764a3b2c9a9c4a3d08957fa57388c9e3992e092b2cdcd7119"
    ),
    "scattering-disk-sensitivity-v1": (
        "d6515919a8b1e7d1f7da9db99e5f888d96d2485e48a435b82ed63b91ff85f204"
    ),
}

# This digest is filled from the canonical 108-row record below.  It is a
# second, qualification-local commitment in addition to the source module's
# population and ensemble commitments.
EXPECTED_OBSERVED_INPUT_SHA256 = (
    "1dae37013e3e141e3f42dc1dc85e9c2d14b6888977fe5fdc3fb4f212fc66093f"
)

SOURCE_STATE_RELATIVE_PATH = (
    "runs/de441_population_100k/states/de441_control_state.csv"
)
PACKAGED_SOURCE_STATE_RELATIVE_PATH = (
    "src/jxplanetx/data/long_horizon_source_model_v1/de441_control_state.csv"
)
PACKAGED_SOURCE_MANIFEST_RELATIVE_PATH = (
    "src/jxplanetx/data/long_horizon_source_model_v1/MANIFEST.json"
)
PACKAGED_SOURCE_MANIFEST_SHA256 = (
    "3695a886bae1b5cbb30daefea6a09dea6305a5b24cd0ccd6aae5d919633de9a1"
)
SOURCE_STATE_SHA256 = (
    "edfbfd50ca29b28b46fe43cae8cb99bd73a9303a5f2e5babf7bf9ad2a820ff14"
)
SOURCE_MANIFEST_RELATIVE_PATH = (
    "runs/de441_population_100k/states/state_manifest.json"
)
SOURCE_MANIFEST_SHA256 = (
    "ec6e2ba8877d206838172dbb573318a2122977e263c054994d4c82a409a43d36"
)
SOURCE_STATE_ARTIFACT_ID = "jx.de441-population.outer-control-state.v1"
SOURCE_EPOCH_JD_TDB = 2_461_200.5
SOURCE_IDS = ("Sun", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto")
SOURCE_PRIMARY_GM_AU3_PER_JULIAN_YEAR2 = float.fromhex(
    "0x1.3bd13a81d2262p+5"
)
SOURCE_FRAME = "J2000_ECLIPTIC_ROTATED_FROM_HORIZONS_ICRF"
SOURCE_ORIGIN = "INNER_SYSTEM_MONOPOLE_CENTER_AT_T0"
SOURCE_CLASSIFICATION = (
    "RECONSTRUCTED_DE441_BACKBONE_PLUS_ASSUMED_INNER_SYSTEM_MONOPOLE"
)

INITIAL_CONDITIONS_MODULE = "jxplanetx.long_horizon.initial_conditions"
INITIAL_CONDITIONS_PATH = "src/jxplanetx/long_horizon/initial_conditions.py"
INITIAL_CONDITIONS_SYMBOLS = (
    "build_predeclared_initial_condition_inventory",
)
RUNNER_MODULE = "jxplanetx.long_horizon.runner"
RUNNER_PATH = "src/jxplanetx/long_horizon/runner.py"
RUNNER_SYMBOLS = (
    "FixedLongHorizonRunConfig",
    "FixedLongHorizonSourceSegment",
    "LongHorizonPreparedSegment",
    "LongHorizonCommittedSegment",
    "advance_fixed_lifecycle_segment",
    "start_fixed_lifecycle_run",
    "resume_fixed_lifecycle_run",
)

EXPECTED_GENESIS_CONVERTER_SHA256 = (
    "b198ea8c3ed39a36bd6aea28f3af51efdf60077bc5f1c2bff26bc886c880bc21"
)
EXPECTED_SOURCE_ALIGNED_GENESIS_MODEL_SHA256 = (
    "13391765e34e9586e6e57472dea03187168e8a724ebf788bfbbe4f4b25ce10fc"
)
EXPECTED_CARTESIAN_INVENTORY_SHA256 = (
    "2531e89fb9623def0110fa0d95de2ccc5bbeb3735379e7acc8304bb7d073eebd"
)
EXPECTED_CARTESIAN_BUNDLE_SHA256 = {
    "jpl-observed-detached-tno-v1": (
        "8a53269551dff8838e4be54b4806325c7756d89d361a4a877a501c53a4ce7544"
    ),
    "jpl-observed-scattering-control-v1": (
        "35c53c45a8b392405bacbce5347eacef752db21d92dec2bfe65fefb8419c2b29"
    ),
}

CORE_IMPLEMENTATION_PATHS = (
    "pyproject.toml",
    "benchmarks/jx_long_horizon_observed_qualification.py",
    "src/jxplanetx/__init__.py",
    "src/jxplanetx/long_horizon/__init__.py",
    "src/jxplanetx/long_horizon/science_inputs.py",
    "src/jxplanetx/long_horizon/contracts.py",
    "src/jxplanetx/long_horizon/restricted_wh.py",
    "src/jxplanetx/long_horizon/initial_conditions.py",
    "src/jxplanetx/long_horizon/runner.py",
    "src/jxplanetx/_all_conic_kepler.py",
    "src/jxplanetx/cuda_tracer_events.py",
    "src/jxplanetx/cuda_restricted_wh.py",
    "src/jxplanetx/cuda_restricted_wh_hybrid.py",
    "src/jxplanetx/fast_wisdom_holman.py",
    "src/jxplanetx/restricted_wh_route.py",
    "src/jxplanetx/restricted_wh_lifecycle.py",
    "src/jxplanetx/restricted_wh_source.py",
    "src/jxplanetx/restricted_wh_checkpoint.py",
    "src/jxplanetx/_wisdom_holman_cpu_v3_module.c",
    "src/jxplanetx/_wisdom_holman_loop.c",
    "src/jxplanetx/_wisdom_holman_loop_v2.c",
    "src/jxplanetx/_wisdom_holman_loop_v3.c",
)
CORE_IMPLEMENTATION_GLOBS = ("src/jxplanetx/engine/*.py",)

NONCLAIMS = {
    "production_qualified": False,
    "science_result": False,
    "scientific_inference_qualified": False,
    "long_horizon_accuracy_qualified": False,
    "four_point_five_gyr_accuracy_qualified": False,
    "observationally_complete_population": False,
    "posterior_or_covariance_sample": False,
    "exact_heliocenter_to_inner_monopole_translation": False,
    "planet_nine_detected": False,
    "planet_nine_excluded": False,
    "physical_collision_inferred": False,
    "physical_ejection_inferred": False,
    "cross_runtime_bitwise_reproducible": False,
    "rebound_is_reference_truth": False,
    "release_authorized": False,
}


class QualificationProtocolError(RuntimeError):
    """A retained input or qualification commitment failed closed."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError) as exc:
        raise QualificationProtocolError(
            "protocol content is not strict canonical JSON"
        ) from exc


def _content_sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + b"\0" + _canonical_json_bytes(value)).hexdigest()


def _regular_file_bytes(path: Path, *, maximum_bytes: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise QualificationProtocolError(f"required regular file is absent: {path}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise QualificationProtocolError(f"cannot inspect required file: {path}") from exc
    if size <= 0 or size > maximum_bytes:
        raise QualificationProtocolError(
            f"required file has an invalid bounded size: {path}"
        )
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise QualificationProtocolError(f"cannot read required file: {path}") from exc
    if len(payload) != size:
        raise QualificationProtocolError(f"required file changed while read: {path}")
    return payload


def _verified_artifact(
    project_root: Path,
    relative_path: str,
    expected_sha256: str,
    *,
    maximum_bytes: int,
) -> dict[str, object]:
    path = project_root / relative_path
    payload = _regular_file_bytes(path, maximum_bytes=maximum_bytes)
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_sha256:
        raise QualificationProtocolError(
            f"artifact digest mismatch for {relative_path}: {observed}"
        )
    return {
        "path": relative_path,
        "sha256": observed,
        "bytes": len(payload),
    }


def _orbit_elements_record(elements: object) -> dict[str, str]:
    names = (
        "semimajor_axis_AU",
        "eccentricity",
        "inclination_rad",
        "longitude_of_ascending_node_rad",
        "argument_of_periapsis_rad",
        "mean_anomaly_rad",
    )
    return {name: getattr(elements, name).hex() for name in names}


def _science_artifact_repository_path(declared_path: str) -> str:
    """Map the package-resource identity to its source-tree custody path."""

    if declared_path.startswith("jxplanetx/"):
        return f"src/{declared_path}"
    return declared_path


def _observed_input_record(project_root: Path) -> dict[str, object]:
    if SCIENCE_INPUTS_CONTENT_SHA256 != EXPECTED_SCIENCE_INPUTS_SHA256:
        raise QualificationProtocolError("science-input registry identity changed")

    rows: list[dict[str, object]] = []
    blocks: list[dict[str, object]] = []
    cases = (
        (DETACHED_TNO_SCREEN_V1, DETACHED_TNO_SENSITIVITY_V1),
        (SCATTERING_DISK_CONTROL_V1, SCATTERING_DISK_SENSITIVITY_V1),
    )
    state_index = 0
    for population, ensemble_spec in cases:
        if population.content_sha256 != EXPECTED_POPULATION_SHA256.get(
            population.population_id
        ):
            raise QualificationProtocolError(
                f"population identity changed: {population.population_id}"
            )
        if ensemble_spec.content_sha256 != EXPECTED_ENSEMBLE_SPEC_SHA256.get(
            ensemble_spec.ensemble_id
        ):
            raise QualificationProtocolError(
                f"ensemble specification changed: {ensemble_spec.ensemble_id}"
            )
        if population.epoch_jd_tdb != SOURCE_EPOCH_JD_TDB:
            raise QualificationProtocolError("population/source epoch mismatch")

        raw = _verified_artifact(
            project_root,
            _science_artifact_repository_path(population.source_artifact_path),
            population.source_artifact_sha256,
            maximum_bytes=2_000_000,
        )
        raw["declared_package_resource"] = population.source_artifact_path
        materialized = materialize_perturbation_ensemble(
            ensemble_spec, population
        )
        if materialized.content_sha256 != EXPECTED_MATERIALIZED_ENSEMBLE_SHA256.get(
            ensemble_spec.ensemble_id
        ):
            raise QualificationProtocolError(
                f"materialized ensemble changed: {ensemble_spec.ensemble_id}"
            )

        block_rows: list[int] = []
        for member in population.members:
            source = member.observed_source
            if source is None:
                raise QualificationProtocolError("nominal observed state lost provenance")
            rows.append(
                {
                    "state_index": state_index,
                    "state_id": member.object_id,
                    "kind": "OBSERVED_NOMINAL",
                    "population_id": population.population_id,
                    "population_sha256": population.content_sha256,
                    "orbit_solution_source_sha256": source.content_sha256,
                    "elements_float64_hex": _orbit_elements_record(member),
                }
            )
            block_rows.append(state_index)
            state_index += 1

        for realization in materialized.members:
            rows.append(
                {
                    "state_index": state_index,
                    "state_id": realization.elements.object_id,
                    "kind": "BOUNDED_INDEPENDENT_SENSITIVITY",
                    "population_id": population.population_id,
                    "population_sha256": population.content_sha256,
                    "ensemble_id": ensemble_spec.ensemble_id,
                    "ensemble_spec_sha256": ensemble_spec.content_sha256,
                    "materialized_ensemble_sha256": materialized.content_sha256,
                    "parent_object_id": realization.parent_object_id,
                    "realization_index": realization.realization_index,
                    "elements_float64_hex": _orbit_elements_record(
                        realization.elements
                    ),
                    "offsets_float64_hex": [
                        {"element_name": name, "offset_hex": value.hex()}
                        for name, value in realization.offsets
                    ],
                }
            )
            block_rows.append(state_index)
            state_index += 1

        blocks.append(
            {
                "population_id": population.population_id,
                "population_sha256": population.content_sha256,
                "ensemble_id": ensemble_spec.ensemble_id,
                "ensemble_spec_sha256": ensemble_spec.content_sha256,
                "materialized_ensemble_sha256": materialized.content_sha256,
                "nominal_count": len(population.members),
                "sensitivity_count": len(materialized.members),
                "state_indices": block_rows,
                "raw_source_artifact": raw,
            }
        )

    state_ids = [row["state_id"] for row in rows]
    nominal_count = sum(row["kind"] == "OBSERVED_NOMINAL" for row in rows)
    sensitivity_count = len(rows) - nominal_count
    if (
        len(rows) != STATE_COUNT
        or nominal_count != NOMINAL_STATE_COUNT
        or sensitivity_count != SENSITIVITY_STATE_COUNT
        or len(set(state_ids)) != STATE_COUNT
        or [row["state_index"] for row in rows] != list(range(STATE_COUNT))
    ):
        raise QualificationProtocolError("the exact 108-state roster changed")

    record: dict[str, object] = {
        "schema": "jx.long-horizon.observed-qualification.inputs.v1",
        "science_inputs_content_sha256": SCIENCE_INPUTS_CONTENT_SHA256,
        "epoch_jd_tdb_float64_hex": SOURCE_EPOCH_JD_TDB.hex(),
        "input_frame": (
            "HELIOCENTRIC_IAU76_80_ECLIPTIC_J2000_OSCULATING"
        ),
        "state_order": (
            "DETACHED_NOMINAL_THEN_DETACHED_SENSITIVITY_THEN_"
            "SCATTERING_NOMINAL_THEN_SCATTERING_SENSITIVITY"
        ),
        "state_count": len(rows),
        "nominal_state_count": nominal_count,
        "sensitivity_state_count": sensitivity_count,
        "population_blocks": blocks,
        "states": rows,
        "interpretation": (
            "observed numerical benchmark plus bounded independent source-sigma "
            "sensitivity envelopes; not covariance or posterior draws"
        ),
    }
    digest = _content_sha256(
        b"jx.long-horizon.observed-qualification.inputs.v1", record
    )
    if digest != EXPECTED_OBSERVED_INPUT_SHA256:
        raise QualificationProtocolError(
            f"108-state qualification identity changed: {digest}"
        )
    return {**record, "content_sha256": digest}


def _module_binding(
    project_root: Path,
    *,
    module: str,
    relative_path: str,
    required_symbols: Sequence[str],
) -> dict[str, object]:
    """Inspect a Python surface by source only; do not import executable code."""

    path = project_root / relative_path
    if not path.exists():
        return {
            "module": module,
            "path": relative_path,
            "present": False,
            "source_sha256": None,
            "required_symbols": list(required_symbols),
            "declared_required_symbols": [],
            "surface_ready": False,
            "inspection": "AST_ONLY_NO_IMPORT",
        }
    payload = _regular_file_bytes(path, maximum_bytes=2_000_000)
    try:
        tree = ast.parse(payload, filename=str(path))
    except (SyntaxError, ValueError) as exc:
        raise QualificationProtocolError(
            f"bound module source cannot be parsed: {relative_path}"
        ) from exc
    declared = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    found = [name for name in required_symbols if name in declared]
    return {
        "module": module,
        "path": relative_path,
        "present": True,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "required_symbols": list(required_symbols),
        "declared_required_symbols": found,
        "surface_ready": len(found) == len(required_symbols),
        "inspection": "AST_ONLY_NO_IMPORT",
    }


def _implementation_bindings(project_root: Path) -> list[dict[str, object]]:
    relative_paths = set(CORE_IMPLEMENTATION_PATHS)
    for pattern in CORE_IMPLEMENTATION_GLOBS:
        matches = tuple(sorted(project_root.glob(pattern)))
        if not matches:
            raise QualificationProtocolError(
                f"implementation source pattern matched no files: {pattern}"
            )
        for path in matches:
            relative_paths.add(path.relative_to(project_root).as_posix())
    bindings: list[dict[str, object]] = []
    for relative_path in sorted(relative_paths):
        payload = _regular_file_bytes(
            project_root / relative_path, maximum_bytes=5_000_000
        )
        bindings.append(
            {
                "path": relative_path,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    return bindings


def _lane_contracts() -> list[dict[str, object]]:
    common = {
        "duration_years": HORIZON_YEARS,
        "initial_epoch_jd_tdb_float64_hex": SOURCE_EPOCH_JD_TDB.hex(),
        "state_count": STATE_COUNT,
        "massive_source_state_sha256": SOURCE_STATE_SHA256,
        "output_target_count": HORIZON_YEARS // OUTPUT_CADENCE_YEARS + 1,
        "output_target_rule": "INTEGER_JULIAN_YEARS_INCLUSIVE_0_THROUGH_1000",
        "event_policy": "RETAIN_ALL_LOCATED_TERMINAL_EVENTS_NO_DROPPING",
    }
    lanes = [
        {
            **common,
            "lane_id": "jx-candidate-fixed-0p0625-year",
            "role": "JX_CANDIDATE",
            "architecture_id": CANONICAL_ARCHITECTURE_ID,
            "family": CANONICAL_FAMILY,
            "source_integrator": "REPLAY_VERIFIED_ORDERED_JACOBI_WISDOM_HOLMAN",
            "tracer_integrator": "BINARY64_PRIMARY_RELATIVE_RESTRICTED_WH",
            "encounter_router": "CONTINUOUS_DOP853_DENSE_BRENTQ",
            "step_years_float64_hex": JX_CANDIDATE_STEP_YEARS.hex(),
            "fixed_step_count": HORIZON_YEARS * 16,
            "backend": "CUPY_CUDA_BINARY64",
            "checkpoint_restart_required": True,
        },
        {
            **common,
            "lane_id": "jx-refinement-fixed-0p03125-year",
            "role": "JX_REFINEMENT_REFERENCE",
            "architecture_id": CANONICAL_ARCHITECTURE_ID,
            "family": CANONICAL_FAMILY,
            "source_integrator": "REPLAY_VERIFIED_ORDERED_JACOBI_WISDOM_HOLMAN",
            "tracer_integrator": "BINARY64_PRIMARY_RELATIVE_RESTRICTED_WH",
            "encounter_router": "CONTINUOUS_DOP853_DENSE_BRENTQ",
            "step_years_float64_hex": JX_REFINEMENT_STEP_YEARS.hex(),
            "fixed_step_count": HORIZON_YEARS * 32,
            "backend": "CUPY_CUDA_BINARY64",
            "checkpoint_restart_required": True,
        },
        {
            **common,
            "lane_id": "rebound-5p1p1-ias15-independent",
            "role": "INDEPENDENT_NUMERICAL_COMPARATOR",
            "distribution": "rebound",
            "required_exact_version": REQUIRED_REBOUND_VERSION,
            "integrator": "ias15",
            "epsilon_float64_hex": REBOUND_IAS15_EPSILON.hex(),
            "initial_dt_years_float64_hex": JX_REFINEMENT_STEP_YEARS.hex(),
            "exact_finish_time_at_every_annual_target": True,
            "massive_sources": "FULL_MUTUAL_NEWTONIAN",
            "tracers": "EXACTLY_MASSLESS_NO_BACKREACTION",
            "comparison_frame": "PRIMARY_RELATIVE_IAU76_80_ECLIPTIC_J2000",
            "adaptive_step_count": True,
            "reference_truth": False,
            "checkpoint_restart_required": False,
        },
    ]
    return lanes


def _measurement_and_gate_contract() -> dict[str, object]:
    return {
        "comparison_rule": (
            "COMPARE_ONLY_MATCHED_ACTIVE_STATE_IDS_AT_IDENTICAL_ANNUAL_TARGETS;"
            "COMPARE_TERMINAL_STATES_SEPARATELY"
        ),
        "metrics": [
            "position_vector_l2_AU",
            "velocity_vector_l2_AU_per_year",
            "absolute_perihelion_delta_AU",
            "absolute_inclination_delta_rad",
            "absolute_minimum_source_distance_delta_AU",
            "absolute_matched_terminal_event_time_delta_years",
        ],
        "jx_candidate_vs_jx_refinement_maxima_float64_hex": {
            "position_vector_l2_AU": (2.0e-3).hex(),
            "velocity_vector_l2_AU_per_year": (2.0e-5).hex(),
            "absolute_perihelion_delta_AU": (2.0e-3).hex(),
            "absolute_inclination_delta_rad": (2.0e-5).hex(),
            "absolute_minimum_source_distance_delta_AU": (2.0e-3).hex(),
            "absolute_matched_terminal_event_time_delta_years": (
                JX_CANDIDATE_STEP_YEARS.hex()
            ),
        },
        "jx_refinement_vs_rebound_ias15_maxima_float64_hex": {
            "position_vector_l2_AU": (2.0e-3).hex(),
            "velocity_vector_l2_AU_per_year": (2.0e-5).hex(),
            "absolute_perihelion_delta_AU": (2.0e-3).hex(),
            "absolute_inclination_delta_rad": (2.0e-5).hex(),
            "absolute_minimum_source_distance_delta_AU": (2.0e-3).hex(),
            "absolute_matched_terminal_event_time_delta_years": (
                JX_CANDIDATE_STEP_YEARS.hex()
            ),
        },
        "event_correspondence": {
            "required_keys": ["state_id", "event_type", "source_id"],
            "unmatched_terminal_event_budget": 0,
            "classification_mismatch_budget": 0,
            "physical_inference_from_guard_event": False,
        },
        "acceptance_semantics": (
            "ALL_LIMITS_AND_RESTART_GATES_MUST_PASS; NO_AGGREGATE_SCORE_CAN_MASK_A_FAILURE"
        ),
        "threshold_provenance": (
            "PREDECLARED_ENGINEERING_ENVELOPE; MUST_NOT_BE_CHANGED_AFTER_ANY_LANE_STARTS"
        ),
    }


def _budgets() -> dict[str, object]:
    samples = HORIZON_YEARS // OUTPUT_CADENCE_YEARS + 1
    return {
        "annual_output": {
            "cadence_years": OUTPUT_CADENCE_YEARS,
            "first_year": 0,
            "last_year": HORIZON_YEARS,
            "exact_epochs_per_lane": samples,
            "maximum_state_rows_per_lane": samples * STATE_COUNT,
            "maximum_state_rows_all_lanes": samples * STATE_COUNT * 3,
            "missing_or_duplicate_epoch_budget": 0,
            "nonfinite_value_budget": 0,
        },
        "events": {
            "maximum_terminal_removal_records_per_lane": STATE_COUNT,
            "maximum_located_terminal_events_per_lane": STATE_COUNT * len(SOURCE_IDS),
            "maximum_unserialized_or_dropped_events": 0,
            "maximum_duplicate_terminal_removals_per_state": 0,
            "event_chain_digest_required": True,
        },
        "artifacts": {
            "maximum_result_bytes_per_lane": 1_073_741_824,
            "maximum_checkpoint_bytes_per_jx_lane": 1_073_741_824,
            "strict_json_no_nan": True,
            "sha256_every_artifact": True,
        },
    }


def _restart_gates() -> dict[str, object]:
    return {
        "split_epoch_years": SPLIT_EPOCH_YEARS,
        "split_output_index": SPLIT_EPOCH_YEARS // OUTPUT_CADENCE_YEARS,
        "jx_candidate_split_global_step": SPLIT_EPOCH_YEARS * 16,
        "jx_refinement_split_global_step": SPLIT_EPOCH_YEARS * 32,
        "required_executions_per_jx_lane": [
            "UNINTERRUPTED_0_TO_1000",
            "FRESH_PROCESS_0_TO_500_CHECKPOINT",
            "FRESH_PROCESS_VERIFIED_RESUME_500_TO_1000",
        ],
        "gates": {
            "checkpoint_contract_status_must_equal": "VERIFIED",
            "checkpoint_protocol_sha256_must_match": True,
            "checkpoint_input_state_sha256_must_match": True,
            "split_state_sha256_matches_uninterrupted_prefix": True,
            "resumed_final_state_sha256_matches_uninterrupted": True,
            "resumed_annual_output_chain_sha256_matches_uninterrupted": True,
            "resumed_event_chain_sha256_matches_uninterrupted": True,
            "checkpoint_index_and_global_step_exact": True,
            "same_runtime_bitwise_replay_required": True,
            "cross_runtime_bitwise_claimed": False,
        },
    }


def _resource_limits() -> dict[str, object]:
    return {
        "physical_cpu_workers": 6,
        "maximum_gpu_devices": 1,
        "required_gpu_device_ordinal": 0,
        "gpu_arithmetic": "BINARY64_ONLY_NO_FAST_MATH",
        "maximum_threads_per_cuda_block": 128,
        "maximum_host_rss_bytes_per_lane": 34_359_738_368,
        "maximum_gpu_allocated_bytes_per_jx_lane": 15_032_385_536,
        "maximum_wall_seconds_per_lane": 43_200,
        "maximum_wall_seconds_total": 86_400,
        "blas_threads_per_process": 1,
        "fail_closed_on_limit": True,
        "concurrent_qualification_lanes": 1,
    }


def build_protocol(project_root: str | os.PathLike[str] | Path) -> dict[str, object]:
    """Build and verify the deterministic preparation receipt.

    The returned document always has ``execution_authorized == False``.  A
    later execution commitment must bind the materialized Cartesian inventory,
    exact runtimes, and this protocol digest before a runner may start.
    """

    root = Path(project_root).resolve()
    if not root.is_dir():
        raise QualificationProtocolError("project_root must be an existing directory")

    observed_inputs = _observed_input_record(root)
    source_artifact = _verified_artifact(
        root,
        SOURCE_STATE_RELATIVE_PATH,
        SOURCE_STATE_SHA256,
        maximum_bytes=1_000_000,
    )
    source_manifest = _verified_artifact(
        root,
        SOURCE_MANIFEST_RELATIVE_PATH,
        SOURCE_MANIFEST_SHA256,
        maximum_bytes=1_000_000,
    )
    packaged_source_artifact = _verified_artifact(
        root,
        PACKAGED_SOURCE_STATE_RELATIVE_PATH,
        SOURCE_STATE_SHA256,
        maximum_bytes=1_000_000,
    )
    packaged_source_manifest = _verified_artifact(
        root,
        PACKAGED_SOURCE_MANIFEST_RELATIVE_PATH,
        PACKAGED_SOURCE_MANIFEST_SHA256,
        maximum_bytes=1_000_000,
    )
    builder = _module_binding(
        root,
        module=INITIAL_CONDITIONS_MODULE,
        relative_path=INITIAL_CONDITIONS_PATH,
        required_symbols=INITIAL_CONDITIONS_SYMBOLS,
    )
    runner = _module_binding(
        root,
        module=RUNNER_MODULE,
        relative_path=RUNNER_PATH,
        required_symbols=RUNNER_SYMBOLS,
    )
    if builder["present"] is not True or builder["surface_ready"] is not True:
        raise QualificationProtocolError(
            "initial-condition builder surface is absent or incomplete"
        )
    if runner["present"] is not True or runner["surface_ready"] is not True:
        raise QualificationProtocolError(
            "long-horizon runner surface is absent or incomplete"
        )

    record: dict[str, object] = {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": PROTOCOL_STATUS,
        "classification": QUALIFICATION_CLASSIFICATION,
        "prepared_only": True,
        "execution_authorized": False,
        "long_run_started": False,
        "architecture": {
            "architecture_id": CANONICAL_ARCHITECTURE_ID,
            "family": CANONICAL_FAMILY,
            "canonical_for_long_horizon": True,
            "public_stability": "EXPERIMENTAL",
            "project_scope": "SCREENING_ONLY",
        },
        "observed_inputs": observed_inputs,
        "source_model": {
            "artifact_id": SOURCE_STATE_ARTIFACT_ID,
            "artifact": source_artifact,
            "packaged_artifact": packaged_source_artifact,
            "construction_manifest": source_manifest,
            "packaged_semantic_manifest": packaged_source_manifest,
            "epoch_jd_tdb_float64_hex": SOURCE_EPOCH_JD_TDB.hex(),
            "source_ids_ordered": list(SOURCE_IDS),
            "primary_source_id": SOURCE_IDS[0],
            "primary_artifact_label": "Sun",
            "primary_physical_model": "COLLAPSED_INNER_SYSTEM_MONOPOLE",
            "primary_collapsed_naif_body_ids": [10, 1, 2, 3, 4],
            "primary_gm_AU3_per_julian_year2_float64_hex": (
                SOURCE_PRIMARY_GM_AU3_PER_JULIAN_YEAR2.hex()
            ),
            "frame": SOURCE_FRAME,
            "origin": SOURCE_ORIGIN,
            "classification": SOURCE_CLASSIFICATION,
            "massive_force_semantics": "FULL_MUTUAL_NEWTONIAN_ALL_ACTIVE_SOURCES",
            "tracer_force_semantics": "DIRECT_NEWTONIAN_FROM_ALL_ACTIVE_SOURCES",
            "tracer_backreaction_semantics": "EXACTLY_MASSLESS_NO_BACKREACTION",
            "source_activity_semantics": "FIXED_ROSTER_FULL_HORIZON_NO_FORCE_SWITCHES",
            "general_relativity": False,
            "galactic_tide": False,
            "stellar_or_rogue_perturber": False,
        },
        "initial_condition_conversion": {
            "builder_binding": builder,
            "required_input_sha256": observed_inputs["content_sha256"],
            "required_source_state_sha256": SOURCE_STATE_SHA256,
            "required_source_model_manifest_sha256": (
                PACKAGED_SOURCE_MANIFEST_SHA256
            ),
            "required_converter_implementation_sha256": (
                EXPECTED_GENESIS_CONVERTER_SHA256
            ),
            "required_genesis_model_id": (
                "de441-control-primary-two-body-genesis-v1"
            ),
            "required_genesis_model_sha256": (
                EXPECTED_SOURCE_ALIGNED_GENESIS_MODEL_SHA256
            ),
            "required_primary_id": "DE441-INNER-SYSTEM-MONOPOLE",
            "required_primary_definition": (
                "FROZEN_DE441_OUTER_CONTROL_ROW_0_EFFECTIVE_INNER_SYSTEM_MONOPOLE"
            ),
            "required_primary_gm_AU3_per_julian_year2_float64_hex": (
                SOURCE_PRIMARY_GM_AU3_PER_JULIAN_YEAR2.hex()
            ),
            "element_origin_to_output_origin_policy": (
                "NO_EPHEMERIS_ORIGIN_TRANSLATION_REINTERPRET_ELEMENTS_IN_"
                "FROZEN_PRIMARY_FORCE_MODEL_AT_SHARED_EPOCH"
            ),
            "same_epoch_no_propagation": True,
            "required_output_frame": (
                "PRIMARY_RELATIVE_IAU76_80_ECLIPTIC_J2000"
            ),
            "required_output_coordinate_system": (
                "PRIMARY_RELATIVE_CARTESIAN_RIGHT_HANDED"
            ),
            "required_length_unit": "AU",
            "required_time_unit": "JULIAN_YEAR_365.25_D",
            "required_velocity_unit": "AU_PER_JULIAN_YEAR_365.25_D",
            "required_gm_unit": "AU3_PER_JULIAN_YEAR_365.25_D2",
            "required_dtype": "IEEE754_BINARY64",
            "required_state_order_preserved": True,
            "required_output_state_count": STATE_COUNT,
            "required_cartesian_inventory_sha256": (
                EXPECTED_CARTESIAN_INVENTORY_SHA256
            ),
            "required_cartesian_bundle_sha256": dict(
                EXPECTED_CARTESIAN_BUNDLE_SHA256
            ),
            "cartesian_inventory_materialized_during_preparation": False,
            "conversion_receipt_sha256": None,
        },
        "lanes": _lane_contracts(),
        "measurements_and_gates": _measurement_and_gate_contract(),
        "output_and_event_budgets": _budgets(),
        "split_restart_qualification": _restart_gates(),
        "resource_limits": _resource_limits(),
        "execution_surface": {
            "runner_binding": runner,
            "required_before_execution": [
                "freeze_this_protocol_sha256",
                "materialize_and_verify_cartesian_inventory",
                "require_builder_surface_ready",
                "require_runner_surface_ready",
                "match_required_cartesian_inventory_sha256",
                "bind_conversion_receipt_sha256",
                "load_exact_rebound_5.1.1_for_independent_lane",
                "record_cuda_device_and_runtime_identity",
                "record_six_physical_cpu_worker_identity",
                "create_fresh_output_directory",
                "independent_preflight_validation",
            ],
            "runner_module_loaded_during_preparation": True,
            "builder_module_loaded_during_preparation": True,
            "numpy_module_loaded_during_preparation": True,
            "runner_entrypoint_called_during_preparation": False,
            "builder_entrypoint_called_during_preparation": False,
            "deterministic_input_materialization_performed": True,
            "dynamics_integration_performed": False,
        },
        "implementation_sources": _implementation_bindings(root),
        "nonclaims": dict(NONCLAIMS),
    }
    protocol_sha256 = _content_sha256(
        b"jx.long-horizon.observed-qualification.protocol.v1", record
    )
    return {**record, "protocol_sha256": protocol_sha256}


def validate_protocol(
    document: object,
    project_root: str | os.PathLike[str] | Path,
) -> dict[str, object]:
    """Reconstruct the preparation protocol and reject every difference."""

    if type(document) is not dict:
        raise QualificationProtocolError("protocol document must be an exact dict")
    expected = build_protocol(project_root)
    if document != expected:
        raise QualificationProtocolError(
            "protocol differs from the exact current preparation commitment"
        )
    if document.get("protocol_sha256") != _content_sha256(
        b"jx.long-horizon.observed-qualification.protocol.v1",
        {key: value for key, value in document.items() if key != "protocol_sha256"},
    ):
        raise QualificationProtocolError("protocol_sha256 is inconsistent")
    if document.get("execution_authorized") is not False:
        raise QualificationProtocolError("prepared protocol cannot authorize execution")
    if any(value is not False for value in document["nonclaims"].values()):
        raise QualificationProtocolError("a prepared protocol widened a claim")
    return expected


def _publish_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise QualificationProtocolError(
            f"refusing to replace existing preparation receipt: {path}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and emit the prepare-only observed-input 1,000-year "
            "qualification protocol; no numerical execution is available."
        )
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        required=True,
        help="required safety acknowledgement; performs no integration",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="create-only JSON path; stdout is used when omitted",
    )
    args = parser.parse_args(argv)
    protocol = build_protocol(args.project_root)
    validate_protocol(protocol, args.project_root)
    payload = json.dumps(
        protocol,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    if args.output is None:
        sys.stdout.buffer.write(payload)
    else:
        _publish_create_only(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
