#!/usr/bin/env python3
"""Build a neutral public DE440s Earth/Jupiter state cache.

This utility contains no private JX source.  It downloads nothing itself; the
workflow supplies exact public NAIF kernels.  It records hashes and samples
Earth center (399) and Jupiter system barycenter (5) relative to the Sun (10)
in J2000 with no aberration correction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import spiceypy as spice

EXPECTED = {
    "de440s.bsp": "c1c7feeab882263fc493a9d5a5b2ddd71b54826cdf65d8d17a76126b260a49f2",
    "naif0012.tls": "678e32bdb5a744117a467cd9601cd6b373f0e9bc9bbde1371d5eee39600a039b",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-day", type=float, default=-2.0)
    parser.add_argument("--end-day", type=float, default=1602.0)
    parser.add_argument("--spacing-day", type=float, default=0.125)
    args = parser.parse_args()

    kernel_dir = args.kernel_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    kernels = [
        kernel_dir / "naif0012.tls",
        kernel_dir / "gm_de440.tpc",
        kernel_dir / "de440s.bsp",
    ]
    for path in kernels:
        if not path.is_file():
            raise FileNotFoundError(path)

    hashes = {path.name: sha256(path) for path in kernels}
    for name, expected in EXPECTED.items():
        if hashes[name] != expected:
            raise RuntimeError(
                f"{name} hash mismatch: expected {expected}, got {hashes[name]}"
            )

    spice.kclear()
    for path in kernels:
        spice.furnsh(str(path))

    count = int(round((args.end_day - args.start_day) / args.spacing_day)) + 1
    days = args.start_day + np.arange(count, dtype=np.float64) * args.spacing_day
    ets = days * 86400.0

    earth = np.empty((count, 6), dtype=np.float64)
    jupiter = np.empty((count, 6), dtype=np.float64)
    for index, et in enumerate(ets):
        earth[index], _ = spice.spkezr("399", float(et), "J2000", "NONE", "10")
        jupiter[index], _ = spice.spkezr("5", float(et), "J2000", "NONE", "10")

    sun_gm = float(spice.bodvrd("SUN", "GM", 1)[1][0])
    jupiter_gm = float(spice.bodvrd("JUPITER BARYCENTER", "GM", 1)[1][0])

    np.savez_compressed(
        output_dir / "de440s_earth399_jupiter5_sun10_j2000_0p125d.npz",
        days=days,
        et_seconds=ets,
        earth_state_km_kmps=earth,
        jupiter_state_km_kmps=jupiter,
        sun_gm_km3_s2=np.array([sun_gm], dtype=np.float64),
        jupiter_barycenter_gm_km3_s2=np.array([jupiter_gm], dtype=np.float64),
    )

    validation_days = np.array(
        [-1.9375, 0.0, 20.0, 220.0, 300.0, 777.125, 1600.0, 1601.9375],
        dtype=np.float64,
    )
    validation = []
    for day in validation_days:
        et = float(day * 86400.0)
        e, _ = spice.spkezr("399", et, "J2000", "NONE", "10")
        j, _ = spice.spkezr("5", et, "J2000", "NONE", "10")
        validation.append(
            {
                "day_from_j2000_tdb": float(day),
                "earth_state_km_kmps": [float(value) for value in e],
                "jupiter_state_km_kmps": [float(value) for value in j],
            }
        )

    metadata = {
        "schema": "jx-public-de440s-earth-jupiter-cache/v1",
        "classification": "PUBLIC_NEUTRAL_EPHEMERIS_CACHE_NO_PRIVATE_JX_SOURCE",
        "frame": "J2000",
        "aberration_correction": "NONE",
        "observer": "SUN (10)",
        "targets": {"earth_center": "399", "jupiter_system_barycenter": "5"},
        "time_origin": "J2000 TDB ET=0",
        "sample_start_day": args.start_day,
        "sample_end_day": args.end_day,
        "sample_spacing_day": args.spacing_day,
        "sample_count": count,
        "kernel_sha256": hashes,
        "spiceypy_version": spice.__version__,
        "numpy_version": np.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "sun_gm_km3_s2": sun_gm,
        "jupiter_barycenter_gm_km3_s2": jupiter_gm,
        "validation_states": validation,
    }
    (output_dir / "CACHE_METADATA.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "KERNEL_SHA256SUMS.txt").write_text(
        "".join(f"{hashes[path.name]}  kernels/{path.name}\n" for path in kernels),
        encoding="utf-8",
    )
    (output_dir / "CACHE_SHA256SUMS.txt").write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in sorted(output_dir.iterdir())
            if path.is_file() and path.name != "CACHE_SHA256SUMS.txt"
        ),
        encoding="utf-8",
    )
    spice.kclear()


if __name__ == "__main__":
    main()
