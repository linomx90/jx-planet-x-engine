#!/usr/bin/env python3
"""Generate the outcome-blind block selection for independent replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


SCHEMA = "jx-independent-block-selection/v1"
SEED = "jx-independent-dop853-block-selection-2026-08-23-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank(block_index: int) -> str:
    payload = f"jx-block-rank/v1\x1f{SEED}\x1f{block_index:03d}".encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite locked selection: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    population = arguments.population.resolve()
    manifest = arguments.population_manifest.resolve()
    ranked = sorted((rank(index), index) for index in range(100))
    selected = sorted(index for _digest, index in ranked[:10])
    record = {
        "schema": SCHEMA,
        "selection_status": "OUTCOME_BLIND_HASH_RANKED",
        "registered_after_reference_execution": True,
        "outcome_independence_statement": "The selection function consumes only the public seed and integer block IDs; it does not read tracer outcomes, block effects, or reference metrics.",
        "seed": SEED,
        "candidate_block_ids": list(range(100)),
        "rank_function": "SHA256('jx-block-rank/v1' || US || seed || US || zero-padded block ID)",
        "selected_count": 10,
        "selected_blocks": selected,
        "selected_rank_records": [
            {"rank": position, "block_index": index, "sha256": digest}
            for position, (digest, index) in enumerate(ranked[:10], start=1)
        ],
        "population": {"path": str(population), "sha256": sha256_file(population)},
        "population_manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
        "nonclaim": "Hash selection prevents outcome-based block choice but cannot make a post-reference registration equivalent to preregistration before the original REBOUND experiment.",
    }
    atomic_json(arguments.output.resolve(), record)
    print(json.dumps({"selected_blocks": selected, "output": str(arguments.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
