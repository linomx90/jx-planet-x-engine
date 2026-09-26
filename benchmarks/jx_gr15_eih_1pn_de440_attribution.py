#!/usr/bin/env python3
"""Run the locked rc11 EIH 1PN attribution gate against DE440s.

This is a model-attribution experiment, not an ephemeris reproduction test.
It asks whether adding only mutual point-mass EIH 1PN gravity reduces the
body-resolved DE440 position residual relative to the otherwise identical
Newtonian GR15 control at 10, 25, 50, and 100 Julian years.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

import jxplanetx
from jxplanetx.gr15 import GR15Spec, integrate_gr15
from jxplanetx.gr15_eih_1pn import (
    gr15_eih_1pn_runtime_identity,
    integrate_gr15_eih_1pn,
)
from jxplanetx.solar_system.eih_1pn import EIH1PNParameters


SCHEMA = "jxplanetx.gr15-eih-1pn-de440-attribution-report.v1"
PROTOCOL_SHA256 = "d3215739bae638d775a7c89c7cbc5ef8c4bc0fe4bdba53f684ce391dab60f6e1"
INITIAL_STATE_PATH = Path(
    "runs/jx_gpu_de440_external_epoch_validation_v1/inputs/"
    "SOURCE_ELEVEN_BODY_INPUT.json"
)
INITIAL_STATE_SHA256 = "0718ba6c05444c720d012bc7906e3eee226664b58d7929648917f9f19003d4c5"
DE440S_PATH = Path(
    "runs/jx_gpu_de440_external_epoch_validation_v1/external/de440s/de440s.bsp"
)
DE440S_SHA256 = "c1c7feeab882263fc493a9d5a5b2ddd71b54826cdf65d8d17a76126b260a49f2"
BODY_IDS = (
    "SUN",
    "MERCURY",
    "VENUS",
    "EARTH",
    "MOON",
    "MARS_BARYCENTER",
    "JUPITER_BARYCENTER",
    "SATURN_BARYCENTER",
    "URANUS_BARYCENTER",
    "NEPTUNE_BARYCENTER",
    "PLUTO_BARYCENTER",
)
NAIF_IDS = (10, 199, 299, 399, 301, 4, 5, 6, 7, 8, 9)
CHECKPOINT_YEARS = (0, 10, 25, 50, 100)
JULIAN_YEAR_SECONDS = 365.25 * 86_400.0


class AttributionError(RuntimeError):
    """The locked attribution experiment could not be executed exactly."""


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_artifact(root: Path, relative: Path, expected_sha256: str) -> Path:
    path = (root / relative).resolve(strict=True)
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise AttributionError(f"artifact escapes repository: {relative}") from exc
    if not path.is_file() or _sha256(path) != expected_sha256:
        raise AttributionError(f"artifact identity changed: {relative}")
    return path


def _read_protocol(root: Path) -> dict[str, Any]:
    path = _require_artifact(
        root,
        Path("benchmarks/jx_gr15_eih_1pn_rc11_protocol.json"),
        PROTOCOL_SHA256,
    )
    protocol = json.loads(path.read_text(encoding="ascii"))
    if (
        protocol.get("schema")
        != "jxplanetx.gr15-eih-1pn-rc11-protocol.v1"
        or protocol.get("status") != "LOCKED_BEFORE_NATIVE_GR15_EIH_IMPLEMENTATION"
        or protocol.get("package_candidate") != "0.6.0rc11"
        or protocol.get("claim_controls", {}).get("scientific_claim_state")
        != "SCREENING_ONLY"
        or protocol.get("acceptance_gates", {}).get(
            "de440_100_year_maximum_position_ratio_to_newtonian_control"
        )
        != 0.25
        or protocol.get("acceptance_gates", {}).get(
            "de440_each_checkpoint_aggregate_rms_ratio_to_newtonian_control"
        )
        != 0.25
    ):
        raise AttributionError("the locked rc11 protocol contract changed")
    return protocol


def _load_initial_state(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = json.loads(path.read_text(encoding="ascii"))
    if tuple(value.get("body_ids", ())) != BODY_IDS:
        raise AttributionError("the initial-state body roster changed")
    try:
        gm = np.ascontiguousarray(
            [
                float.fromhex(item) / 1.0e9
                for item in value["gravitational_parameters_m3_s2_hex"]
            ],
            dtype=np.float64,
        )
        positions = np.ascontiguousarray(
            [
                [float.fromhex(item) / 1.0e3 for item in row]
                for row in value["initial"]["positions_metres_hex"]
            ],
            dtype=np.float64,
        )
        velocities = np.ascontiguousarray(
            [
                [float.fromhex(item) / 1.0e3 for item in row]
                for row in value["initial"]["velocities_metres_per_second_hex"]
            ],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AttributionError("the initial state has invalid hexadecimal arrays") from exc
    if (
        gm.shape != (len(BODY_IDS),)
        or positions.shape != (len(BODY_IDS), 3)
        or velocities.shape != (len(BODY_IDS), 3)
        or not np.all(gm > 0.0)
        or not np.all(np.isfinite(gm))
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(velocities))
    ):
        raise AttributionError("the converted initial state is invalid")
    return positions, velocities, gm


def _query_de440(path: Path, epochs: tuple[float, ...]) -> np.ndarray:
    try:
        import spiceypy
    except ModuleNotFoundError as exc:
        raise AttributionError("SpiceyPy 8.2.0 is required") from exc
    if importlib.metadata.version("spiceypy") != "8.2.0":
        raise AttributionError("SpiceyPy must be exact version 8.2.0")
    positions = np.empty((len(epochs), len(NAIF_IDS), 3), dtype=np.float64)
    spiceypy.kclear()
    spiceypy.furnsh(str(path))
    try:
        for checkpoint, epoch in enumerate(epochs):
            for body, target in enumerate(NAIF_IDS):
                state, _ = spiceypy.spkez(target, epoch, "J2000", "NONE", 0)
                positions[checkpoint, body] = state[:3]
    finally:
        spiceypy.kclear()
    if not np.all(np.isfinite(positions)):
        raise AttributionError("DE440 returned nonfinite positions")
    return positions


def _spec(epochs: tuple[float, ...]) -> GR15Spec:
    return GR15Spec(
        epochs[0],
        epochs[-1],
        intermediate_epochs=epochs[1:-1],
        initial_step=21_600.0,
        minimum_step=1.0e-6,
        maximum_step=691_200.0,
        epsilon=1.0e-10,
        maximum_steps=2_000_000,
        maximum_rejections=100_000,
    )


def _heliocentric(positions: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(positions - positions[:, :1, :])


def _residual_rows(
    newtonian_positions: np.ndarray,
    eih_positions: np.ndarray,
    de440_positions: np.ndarray,
) -> list[dict[str, Any]]:
    newtonian_residuals = np.linalg.norm(
        _heliocentric(newtonian_positions) - _heliocentric(de440_positions),
        axis=2,
    )
    eih_residuals = np.linalg.norm(
        _heliocentric(eih_positions) - _heliocentric(de440_positions),
        axis=2,
    )
    rows: list[dict[str, Any]] = []
    for checkpoint, year in enumerate(CHECKPOINT_YEARS[1:], 1):
        newtonian = newtonian_residuals[checkpoint, 1:]
        eih = eih_residuals[checkpoint, 1:]
        newtonian_rms = float(np.sqrt(np.mean(newtonian**2)))
        eih_rms = float(np.sqrt(np.mean(eih**2)))
        newtonian_maximum = float(np.max(newtonian))
        eih_maximum = float(np.max(eih))
        rows.append(
            {
                "julian_year": year,
                "aggregate_rms": {
                    "newtonian_km": newtonian_rms,
                    "eih_1pn_km": eih_rms,
                    "ratio_to_newtonian": eih_rms / newtonian_rms,
                },
                "maximum_body_residual": {
                    "newtonian_km": newtonian_maximum,
                    "eih_1pn_km": eih_maximum,
                    "ratio_to_newtonian": eih_maximum / newtonian_maximum,
                },
                "bodies": [
                    {
                        "body_id": BODY_IDS[body],
                        "newtonian_residual_km": float(newtonian_residuals[checkpoint, body]),
                        "eih_1pn_residual_km": float(eih_residuals[checkpoint, body]),
                        "ratio_to_newtonian": float(
                            eih_residuals[checkpoint, body]
                            / newtonian_residuals[checkpoint, body]
                        ),
                    }
                    for body in range(1, len(BODY_IDS))
                ],
            }
        )
    return rows


def run(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    protocol = _read_protocol(root)
    initial_path = _require_artifact(
        root,
        INITIAL_STATE_PATH,
        INITIAL_STATE_SHA256,
    )
    de440_path = _require_artifact(root, DE440S_PATH, DE440S_SHA256)
    positions, velocities, gm = _load_initial_state(initial_path)
    epochs = tuple(year * JULIAN_YEAR_SECONDS for year in CHECKPOINT_YEARS)
    spec = _spec(epochs)
    newtonian = integrate_gr15(positions, velocities, gm, spec)
    eih = integrate_gr15_eih_1pn(
        positions,
        velocities,
        gm,
        spec,
        EIH1PNParameters(speed_of_light_km_s=299_792.458),
    )
    de440_positions = _query_de440(de440_path, epochs)
    rows = _residual_rows(
        newtonian.checkpoint_positions,
        eih.checkpoint_positions,
        de440_positions,
    )
    limit = float(
        protocol["acceptance_gates"][
            "de440_each_checkpoint_aggregate_rms_ratio_to_newtonian_control"
        ]
    )
    final_limit = float(
        protocol["acceptance_gates"][
            "de440_100_year_maximum_position_ratio_to_newtonian_control"
        ]
    )
    checkpoint_gate = all(
        row["aggregate_rms"]["ratio_to_newtonian"] <= limit for row in rows
    )
    final_gate = (
        rows[-1]["maximum_body_residual"]["ratio_to_newtonian"] <= final_limit
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS_SCREENING_ONLY" if checkpoint_gate and final_gate else "FAIL_LOCKED_GATE",
        "scientific_claim_state": "SCREENING_ONLY",
        "protocol": {
            "path": "benchmarks/jx_gr15_eih_1pn_rc11_protocol.json",
            "sha256": PROTOCOL_SHA256,
        },
        "source_artifacts": {
            "initial_state": {
                "path": INITIAL_STATE_PATH.as_posix(),
                "sha256": INITIAL_STATE_SHA256,
            },
            "de440s": {
                "path": DE440S_PATH.as_posix(),
                "sha256": DE440S_SHA256,
            },
        },
        "model": {
            "body_ids": list(BODY_IDS),
            "naif_ids": list(NAIF_IDS),
            "units": {"length": "km", "time": "s", "gm": "km^3/s^2"},
            "newtonian_control": "JX_GAUSS_RADAU15_V3",
            "candidate": "JX_GR15_EIH1PN_V1",
        },
        "checkpoint_results": rows,
        "acceptance": {
            "each_10_25_50_100_year_aggregate_rms_ratio": {
                "limit": limit,
                "passed": checkpoint_gate,
            },
            "100_year_maximum_body_residual_ratio": {
                "limit": final_limit,
                "value": rows[-1]["maximum_body_residual"]["ratio_to_newtonian"],
                "passed": final_gate,
            },
        },
        "accounting": {
            "newtonian": {
                "accepted_steps": newtonian.accepted_steps,
                "rejected_steps": newtonian.rejected_steps,
                "force_evaluations": newtonian.force_evaluations,
                "replay_digest": newtonian.replay_digest,
            },
            "eih_1pn": {
                "accepted_steps": eih.accepted_steps,
                "rejected_steps": eih.rejected_steps,
                "force_evaluations": eih.force_evaluations,
                "replay_digest": eih.replay_digest,
                "observed_maximum_compactness": eih.observed_maximum_compactness,
                "observed_maximum_speed_fraction_squared": (
                    eih.observed_maximum_speed_fraction_squared
                ),
            },
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "spiceypy": importlib.metadata.version("spiceypy"),
            "jxplanetx": jxplanetx.__version__,
            "candidate_identity": gr15_eih_1pn_runtime_identity(),
        },
        "claim_controls": {
            "de440_equivalence_claimed": False,
            "exact_general_relativity_claimed": False,
            "general_superiority_claimed": False,
            "navigation_or_production_authorized": False,
            "single_machine_observation": True,
        },
    }
    report["semantic_content_sha256"] = hashlib.sha256(
        b"jx.gr15-eih-1pn-de440-attribution-report.v1\0" + _canonical(report)
    ).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = run(root)
    arguments.output.write_bytes(_canonical(report))
    return 0 if report["status"] == "PASS_SCREENING_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
