#!/usr/bin/env python3
"""Generate the outcome-blind 100,000-tracer Latin-hypercube proposal."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tempfile
from pathlib import Path


SCHEMA = "jx-de441-population-draws/v1"
SEED = "jx-de441-bb21-9118-broad-proposal-2026-08-22-v1"
BLOCKS = 100
TRACERS_PER_BLOCK = 1000
FIELDS = ("log_a", "q", "cos_i", "Omega", "omega", "M")


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


def digest(block: int, local: int, field: str, stream: str) -> bytes:
    message = f"jx-lhs/v1\x1f{SEED}\x1f{block}\x1f{local}\x1f{field}\x1f{stream}"
    return hashlib.sha256(message.encode("utf-8")).digest()


def open_uniform(block: int, local: int, field: str) -> float:
    integer = int.from_bytes(digest(block, local, field, "jitter")[:8], "big") >> 11
    return (integer + 0.5) / float(1 << 53)


def ranks(block: int, field: str) -> list[int]:
    order = sorted(range(TRACERS_PER_BLOCK), key=lambda local: digest(block, local, field, "rank"))
    result = [0] * TRACERS_PER_BLOCK
    for rank, local in enumerate(order):
        result[local] = rank
    return result


def main() -> int:
    root = Path(__file__).resolve().parent
    output_path = root / "population_elements_v1.csv"
    manifest_path = root / "population_manifest_v1.json"
    if output_path.exists() or manifest_path.exists():
        raise SystemExit("refusing to replace the locked population proposal")
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "block_index",
            "local_index",
            "logical_id",
            "a0_AU",
            "q0_AU",
            "e0",
            "i0_deg",
            "Omega0_rad",
            "omega0_rad",
            "M0_rad",
        )
    )
    cosine_40 = math.cos(math.radians(40.0))
    for block in range(BLOCKS):
        field_ranks = {field: ranks(block, field) for field in FIELDS}
        for local in range(TRACERS_PER_BLOCK):
            uniforms = {
                field: (field_ranks[field][local] + open_uniform(block, local, field))
                / TRACERS_PER_BLOCK
                for field in FIELDS
            }
            a = 100.0 * 10.0 ** uniforms["log_a"]
            q = 31.0 + 49.0 * uniforms["q"]
            eccentricity = 1.0 - q / a
            cosine_i = 1.0 - uniforms["cos_i"] * (1.0 - cosine_40)
            inclination = math.degrees(math.acos(cosine_i))
            turn = 2.0 * math.pi
            writer.writerow(
                (
                    block,
                    local,
                    f"b{block:03d}-j{local:04d}",
                    format(a, ".17e"),
                    format(q, ".17e"),
                    format(eccentricity, ".17e"),
                    format(inclination, ".17e"),
                    format(turn * uniforms["Omega"], ".17e"),
                    format(turn * uniforms["omega"], ".17e"),
                    format(turn * uniforms["M"], ".17e"),
                )
            )
    atomic_write(output_path, output.getvalue())
    manifest = {
        "schema": SCHEMA,
        "classification": "OUTCOME_BLIND_STRATIFIED_PROPOSAL_NOT_OBSERVATIONAL_PRIOR",
        "seed": SEED,
        "generator": "independent-per-margin SHA-256-ranked Latin hypercube with open jitter/v1",
        "blocks": BLOCKS,
        "tracers_per_block": TRACERS_PER_BLOCK,
        "tracers": BLOCKS * TRACERS_PER_BLOCK,
        "inference_unit": "independent phase/design block; never individual tracer",
        "proposal": {
            "a_AU": "log-uniform [100,1000]",
            "q0_AU": "uniform [31,80]",
            "i_deg": "isotropic in cos(i), truncated to [0,40] degrees",
            "Omega_rad": "uniform [0,2*pi)",
            "omega_rad": "uniform [0,2*pi)",
            "M_rad": "uniform [0,2*pi)",
        },
        "population_csv": output_path.name,
        "population_sha256": sha256(output_path),
        "nonclaim": "This broad proposal is not an observed TNO prior, survey selection function, or posterior population model.",
    }
    atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": BLOCKS * TRACERS_PER_BLOCK, "sha256": manifest["population_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
