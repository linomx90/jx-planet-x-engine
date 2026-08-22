#!/usr/bin/env python3
"""Fetch and normalize the predeclared JPL Horizons DE441 reference states.

This program only acquires reference data.  It does not import or run JX, so
the reference can be hash-locked before any model outcome is computed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path


API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
EPOCHS = tuple(Decimal("2461200.5") + Decimal("365.25") * index for index in range(11))
BODIES = (
    (1, "Mercury Barycenter"),
    (2, "Venus Barycenter"),
    (3, "Earth-Moon Barycenter"),
    (4, "Mars Barycenter"),
    (5, "Jupiter Barycenter"),
    (6, "Saturn Barycenter"),
    (7, "Uranus Barycenter"),
    (8, "Neptune Barycenter"),
    (9, "Pluto Barycenter"),
    (10, "Sun"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
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


def build_url(body_id: int) -> str:
    values = " ".join(format(epoch, "f") for epoch in EPOCHS)
    parameters = {
        "format": "text",
        "COMMAND": f"'{body_id}'",
        "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTORS'",
        "CENTER": "'@0'",
        "REF_PLANE": "'FRAME'",
        "REF_SYSTEM": "'ICRF'",
        "OUT_UNITS": "'AU-D'",
        "VEC_TABLE": "'2'",
        "CSV_FORMAT": "'YES'",
        "VEC_LABELS": "'YES'",
        "TLIST": f"'{values}'",
    }
    return API_URL + "?" + urllib.parse.urlencode(parameters)


def fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "JX-DE441-validation/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read()
    return payload.decode("utf-8")


def parse_vectors(raw: str, body_id: int, body_name: str) -> list[tuple[str, ...]]:
    required_markers = (
        "{source: DE441}",
        "Solar System Barycenter (0)",
        "Output units    : AU-D",
        "Output type     : GEOMETRIC cartesian states",
        "Reference frame : ICRF",
        "$$SOE",
        "$$EOE",
    )
    missing = [marker for marker in required_markers if marker not in raw]
    if missing:
        raise ValueError(f"body {body_id} response missing required markers: {missing}")
    block = raw.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    parsed: list[tuple[str, ...]] = []
    for row in csv.reader(io.StringIO(block)):
        fields = [field.strip() for field in row]
        while fields and fields[-1] == "":
            fields.pop()
        if not fields:
            continue
        if len(fields) != 8:
            raise ValueError(f"body {body_id} vector row has {len(fields)} fields, expected 8")
        jd, calendar, *state = fields
        numeric = [Decimal(jd), *(Decimal(value) for value in state)]
        if not all(value.is_finite() for value in numeric):
            raise ValueError(f"body {body_id} contains a non-finite vector")
        parsed.append(
            (
                str(body_id),
                body_name,
                format(numeric[0], "f"),
                calendar,
                *(str(value) for value in numeric[1:]),
            )
        )
    observed_epochs = tuple(Decimal(row[2]) for row in parsed)
    if observed_epochs != EPOCHS:
        raise ValueError(f"body {body_id} epoch list changed: {observed_epochs}")
    return parsed


def render_csv(rows: list[tuple[str, ...]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "body_id",
            "body_name",
            "jd_tdb",
            "calendar_tdb",
            "x_au",
            "y_au",
            "z_au",
            "vx_au_per_day",
            "vy_au_per_day",
            "vz_au_per_day",
        )
    )
    writer.writerows(rows)
    return output.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "reference")
    parser.add_argument("--force", action="store_true", help="replace an existing unlocked acquisition")
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    manifest_path = output_dir / "horizons_reference_manifest.json"
    if manifest_path.exists() and not arguments.force:
        raise SystemExit(f"refusing to replace existing acquisition: {manifest_path}")

    raw_directory = output_dir / "raw"
    all_rows: list[tuple[str, ...]] = []
    raw_records = []
    for body_id, body_name in BODIES:
        url = build_url(body_id)
        raw = fetch(url)
        rows = parse_vectors(raw, body_id, body_name)
        raw_path = raw_directory / f"body_{body_id:02d}.txt"
        atomic_write_text(raw_path, raw)
        all_rows.extend(rows)
        raw_records.append(
            {
                "body_id": body_id,
                "body_name": body_name,
                "query_url": url,
                "row_count": len(rows),
                "raw_path": str(raw_path.relative_to(output_dir)),
                "raw_sha256": sha256(raw_path),
            }
        )

    normalized_path = output_dir / "horizons_de441_vectors.csv"
    atomic_write_text(normalized_path, render_csv(all_rows))
    manifest = {
        "schema": "jx-horizons-reference/v1",
        "classification": "AUTHORITATIVE_EXTERNAL_REFERENCE",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_url": API_URL,
        "ephemeris_source_required": "DE441",
        "center": "Solar System Barycenter (0)",
        "reference_frame": "ICRF",
        "state_type": "geometric Cartesian position and velocity",
        "time_scale": "TDB",
        "units": "AU and AU/day",
        "epochs_jd_tdb": [format(epoch, "f") for epoch in EPOCHS],
        "body_count": len(BODIES),
        "rows_per_body": len(EPOCHS),
        "row_count": len(all_rows),
        "raw_responses": raw_records,
        "normalized_path": normalized_path.name,
        "normalized_sha256": sha256(normalized_path),
        "nonclaim": "Reference acquisition only; no JX output was computed by this program.",
    }
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "normalized_sha256": manifest["normalized_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
