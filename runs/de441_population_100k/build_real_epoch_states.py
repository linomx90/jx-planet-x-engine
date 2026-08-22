#!/usr/bin/env python3
"""Build matched JX source/control states from the locked DE441 epoch.

The inner planets are collapsed into a single inner-system monopole.  The
source orbit is the preserved Brown--Batygin candidate 9118 hypothesis,
declared at the DE441 epoch; that epoch assignment is an explicit assumption.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
from pathlib import Path


AU_KM = 149_597_870.700
YEAR_SECONDS = 365.25 * 86_400.0
EPOCH_JD_TDB = 2_461_200.5
OBLIQUITY_J2000_RAD = math.radians(23.43929111111111)
REFERENCE_SHA256 = "18308086c9191333448e07a6ab6942a6b0fc7f0100ad2b3abaffbb1a0fd3993e"
GM_SOURCE_SHA256 = "924ddf4fb9ead9fe8a1aa55780bcabde40b09d00065d58226e24b68d8092f140"
CANDIDATE_METADATA_SHA256 = "509917e0093107464d9ee45ed2c8e9f403403b2bb0e94455fa3614825917f8b0"
INNER_IDS = (10, 1, 2, 3, 4)
OUTER_IDS = (5, 6, 7, 8, 9)
OUTER_NAMES = {
    5: "Jupiter",
    6: "Saturn",
    7: "Uranus",
    8: "Neptune",
    9: "Pluto",
}
CANDIDATE = {
    "catalog": "Brown_Batygin_2021_reference_population_v3",
    "index": 9118,
    "name": "P9_BB21_idx9118",
    "mass_Mearth": 5.06,
    "a_AU": 495.19,
    "e": 0.236,
    "i_deg": 20.28,
    "Omega_deg": 96.87,
    "omega_deg": 187.3,
    "M_deg": 126.21,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def rotate_icrf_to_ecliptic(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = vector
    cosine = math.cos(OBLIQUITY_J2000_RAD)
    sine = math.sin(OBLIQUITY_J2000_RAD)
    return x, cosine * y + sine * z, -sine * y + cosine * z


def parse_gm(path: Path) -> dict[int, float]:
    if sha256(path) != GM_SOURCE_SHA256:
        raise ValueError("gm_de440.tpc hash changed")
    text = path.read_text(encoding="utf-8")
    result = {}
    for body_id in (*range(1, 11), 399):
        match = re.search(rf"BODY{body_id}_GM\s*=\s*\(\s*([0-9.+\-EDed]+)\s*\)", text)
        if not match:
            raise ValueError(f"missing BODY{body_id}_GM")
        result[body_id] = float(match.group(1).replace("D", "E").replace("d", "e"))
    return result


def load_epoch(path: Path) -> dict[int, tuple[float, ...]]:
    if sha256(path) != REFERENCE_SHA256:
        raise ValueError("normalized DE441/Horizons reference hash changed")
    result = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if float(row["jd_tdb"]) != EPOCH_JD_TDB:
                continue
            body_id = int(row["body_id"])
            position = rotate_icrf_to_ecliptic(
                (float(row["x_au"]), float(row["y_au"]), float(row["z_au"]))
            )
            velocity_day = rotate_icrf_to_ecliptic(
                (
                    float(row["vx_au_per_day"]),
                    float(row["vy_au_per_day"]),
                    float(row["vz_au_per_day"]),
                )
            )
            result[body_id] = (*position, *(value * 365.25 for value in velocity_day))
    if tuple(sorted(result)) != tuple(range(1, 11)):
        raise ValueError("DE441 epoch does not contain body IDs 1 through 10")
    return result


def weighted_state(states: dict[int, tuple[float, ...]], gm: dict[int, float], ids: tuple[int, ...]) -> tuple[float, ...]:
    total = sum(gm[body_id] for body_id in ids)
    return tuple(sum(gm[body_id] * states[body_id][axis] for body_id in ids) / total for axis in range(6))


def rotate_orbit_plane(vector: tuple[float, float, float], node: float, inclination: float, periapse: float) -> tuple[float, float, float]:
    x, y, z = vector
    cw, sw = math.cos(periapse), math.sin(periapse)
    ci, si = math.cos(inclination), math.sin(inclination)
    cO, sO = math.cos(node), math.sin(node)
    x1, y1 = cw * x - sw * y, sw * x + cw * y
    x2, y2, z2 = x1, ci * y1 - si * z, si * y1 + ci * z
    return cO * x2 - sO * y2, sO * x2 + cO * y2, z2


def candidate_state(central_gm: float, earth_gm: float) -> tuple[float, tuple[float, ...]]:
    gm_conversion = YEAR_SECONDS**2 / AU_KM**3
    source_gm = CANDIDATE["mass_Mearth"] * earth_gm * gm_conversion
    a = CANDIDATE["a_AU"]
    eccentricity = CANDIDATE["e"]
    mean_anomaly = math.radians(CANDIDATE["M_deg"])
    eccentric_anomaly = mean_anomaly
    for _ in range(30):
        correction = (
            eccentric_anomaly - eccentricity * math.sin(eccentric_anomaly) - mean_anomaly
        ) / (1.0 - eccentricity * math.cos(eccentric_anomaly))
        eccentric_anomaly -= correction
        if abs(correction) < 1e-15:
            break
    root = math.sqrt(1.0 - eccentricity * eccentricity)
    denominator = 1.0 - eccentricity * math.cos(eccentric_anomaly)
    position_plane = (
        a * (math.cos(eccentric_anomaly) - eccentricity),
        a * root * math.sin(eccentric_anomaly),
        0.0,
    )
    mean_motion = math.sqrt((central_gm + source_gm) / a**3)
    velocity_plane = (
        -a * mean_motion * math.sin(eccentric_anomaly) / denominator,
        a * mean_motion * root * math.cos(eccentric_anomaly) / denominator,
        0.0,
    )
    angles = tuple(
        math.radians(CANDIDATE[key]) for key in ("Omega_deg", "i_deg", "omega_deg")
    )
    position = rotate_orbit_plane(position_plane, *angles)
    velocity = rotate_orbit_plane(velocity_plane, *angles)
    return source_gm, (*position, *velocity)


def render_state(rows: list[tuple[object, ...]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("index", "name", "mass", "x", "y", "z", "vx", "vy", "vz"))
    writer.writerows(rows)
    return output.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    script = Path(__file__).resolve()
    project = script.parents[2]
    parser.add_argument(
        "--reference",
        type=Path,
        default=project / "runs/de441_horizons_10yr/reference/horizons_de441_vectors.csv",
    )
    parser.add_argument(
        "--gm-source",
        type=Path,
        default=project.parent
        / "imports/de441_family/jx_anomaly_zone_orbit_family_5k500_screen_2026-08-18/kernels/gm_de440.tpc",
    )
    parser.add_argument(
        "--candidate-metadata",
        type=Path,
        default=project.parent / "imports/compact_source_core/source/candidate_metadata.txt",
    )
    parser.add_argument("--output-dir", type=Path, default=script.parent / "states")
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    control_path = output_dir / "de441_control_state.csv"
    source_path = output_dir / "de441_source_9118_state.csv"
    metadata_path = output_dir / "state_manifest.json"
    if any(path.exists() for path in (control_path, source_path, metadata_path)):
        raise SystemExit("refusing to replace an existing state artifact")
    if sha256(arguments.candidate_metadata) != CANDIDATE_METADATA_SHA256:
        raise ValueError("candidate metadata hash changed")

    gm_km = parse_gm(arguments.gm_source)
    states = load_epoch(arguments.reference)
    gm_conversion = YEAR_SECONDS**2 / AU_KM**3
    gm_year = {body_id: value * gm_conversion for body_id, value in gm_km.items()}
    inner_state = weighted_state(states, gm_km, INNER_IDS)
    inner_gm = sum(gm_year[body_id] for body_id in INNER_IDS)
    rows: list[tuple[object, ...]] = [
        (0, "Sun", format(inner_gm, ".17e"), *("0" for _ in range(6)))
    ]
    for index, body_id in enumerate(OUTER_IDS, 1):
        relative = tuple(states[body_id][axis] - inner_state[axis] for axis in range(6))
        rows.append(
            (
                index,
                OUTER_NAMES[body_id],
                format(gm_year[body_id], ".17e"),
                *(format(value, ".17e") for value in relative),
            )
        )
    source_gm, source_vector = candidate_state(inner_gm, gm_km[399])
    source_rows = rows + [
        (
            len(rows),
            CANDIDATE["name"],
            format(source_gm, ".17e"),
            *(format(value, ".17e") for value in source_vector),
        )
    ]
    atomic_write(control_path, render_state(rows))
    atomic_write(source_path, render_state(source_rows))
    source_radius = math.sqrt(sum(value * value for value in source_vector[:3]))
    manifest = {
        "schema": "jx-de441-population-state/v1",
        "classification": "RECONSTRUCTED_DE441_BACKBONE_PLUS_ASSUMED_SOURCE_EPOCH",
        "epoch_jd_tdb": EPOCH_JD_TDB,
        "frame": "J2000 ecliptic, rotated from Horizons ICRF",
        "origin": "inner-system monopole center at t0",
        "units": "AU, Julian year, GM in AU^3/year^2; integration G=1",
        "inner_system_monopole_body_ids": list(INNER_IDS),
        "active_outer_body_ids": list(OUTER_IDS),
        "source_candidate": {**CANDIDATE, "computed_radius_AU": source_radius},
        "source_epoch_assumption": "Candidate 9118 osculating elements are declared at JD 2461200.5; this is a model assumption, not a JPL source state or observational refit.",
        "upstream": {
            "horizons_vectors": {"path": str(arguments.reference.resolve()), "sha256": sha256(arguments.reference)},
            "gm_de440": {"path": str(arguments.gm_source.resolve()), "sha256": sha256(arguments.gm_source)},
            "candidate_metadata": {"path": str(arguments.candidate_metadata.resolve()), "sha256": sha256(arguments.candidate_metadata)},
        },
        "outputs": {
            "control": {"path": control_path.name, "sha256": sha256(control_path)},
            "source": {"path": source_path.name, "sha256": sha256(source_path)},
        },
        "nonclaim": "The planetary backbone is real-epoch DE441/Horizons-derived. The source epoch and inner-system monopole are explicit modeling assumptions.",
    }
    atomic_write(metadata_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest["outputs"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
