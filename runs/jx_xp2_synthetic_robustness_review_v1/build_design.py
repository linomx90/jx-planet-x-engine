#!/usr/bin/env python3
"""Pre-output deterministic builder for JX-XP2 exact Cartesian input rows.

This utility performs no integration and imports neither REBOUND nor SciPy.  It
emits the complete ``initial_states_v1.json`` artifact to stdout, or verifies an
existing artifact byte-for-byte.  The independent replay verifier contains a
separate implementation of the same frozen mathematical specification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Sequence


EXPERIMENT_ID = "jx-xp2-public-synthetic-robustness-v1"
SUFFIXES = ("LOG_A", "Q", "COS_I", "OMEGA", "OMEGA_ARGUMENT", "MEAN_ANOMALY")
LHS_DOMAIN = b"jx-xp2-lhs-u64/v1\0"
TRACER_DOMAIN = b"jx-xp2-canonical-tracer-design/v1\0"
EXPANDED_DOMAIN = b"jx-xp2-expanded-barycentric-state/v1\0"
INDEX_DOMAIN = b"jx-xp2-configuration-digest-index/v1\0"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode()


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def pack(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("non-finite binary64 design component")
    return struct.pack(">d", float(value)).hex()


def pack3(values: Sequence[float]) -> str:
    if len(values) != 3:
        raise ValueError("expected three components")
    return "".join(pack(value) for value in values)


def pack6(values: Sequence[float]) -> str:
    if len(values) != 6:
        raise ValueError("expected six components")
    return "".join(pack(value) for value in values)


def derive_seed(domain: str, design_hash: str, label: str) -> bytes:
    domain_bytes = domain.encode("ascii"); label_bytes = label.encode("ascii")
    return hashlib.sha256(
        len(domain_bytes).to_bytes(4, "big") + domain_bytes + bytes.fromhex(design_hash)
        + len(label_bytes).to_bytes(4, "big") + label_bytes + (0).to_bytes(8, "big")
    ).digest()[:16]


def lhs(seed: bytes) -> tuple[list[int], list[float]]:
    counter = 0
    def word() -> int:
        nonlocal counter
        result = int.from_bytes(hashlib.sha256(
            LHS_DOMAIN + seed + counter.to_bytes(8, "big")
        ).digest()[:8], "big")
        counter += 1
        return result
    permutation = list(range(16)); modulus = 1 << 64
    for index in range(15, 0, -1):
        divisor = index + 1; limit = modulus - modulus % divisor
        candidate = word()
        while candidate >= limit:
            candidate = word()
        other = candidate % divisor
        permutation[index], permutation[other] = permutation[other], permutation[index]
    values = [(stratum + (word() >> 11) / float(1 << 53)) / 16.0
              for stratum in permutation]
    return permutation, values


def tracer_elements(contract: dict[str, Any], seed_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    policy = contract["seed_policy"]
    if hashlib.sha256(canonical(contract["design_core"])).hexdigest() != policy["design_core_sha256"]:
        raise ValueError("design-core hash mismatch")
    streams = {row["stream_label"]: row["seed_hex_128"] for row in seed_manifest["streams"]}
    rows: list[dict[str, Any]] = []; canonical_rows = []
    test: tuple[list[int], list[float]] | None = None
    for block in range(8):
        dimensions = {}
        for suffix in SUFFIXES:
            label = f"LHS_BLOCK_{block}_{suffix}"
            expected = derive_seed(policy["domain_ascii"], policy["design_core_sha256"], label)
            if streams.get(label) != expected.hex():
                raise ValueError("seed stream mismatch")
            permutation, values = lhs(expected); dimensions[suffix] = values
            if block == 0 and suffix == "LOG_A": test = permutation, values
        for index in range(16):
            a = math.exp(math.log(150.0) + dimensions["LOG_A"][index]
                         * (math.log(800.0) - math.log(150.0)))
            q = 35.0 + 45.0 * dimensions["Q"][index]
            e = 1.0 - q / a
            inc = math.acos(math.cos(math.radians(40.0)) + dimensions["COS_I"][index]
                            * (1.0 - math.cos(math.radians(40.0))))
            row = {
                "logical_id": f"XP2-B{block:02d}-T{index:02d}", "block_index": block,
                "index_within_block": index, "a_AU": a, "q_AU": q, "e": e,
                "i_rad": inc, "Omega_rad": 2.0 * math.pi * dimensions["OMEGA"][index],
                "omega_rad": 2.0 * math.pi * dimensions["OMEGA_ARGUMENT"][index],
                "M_rad": 2.0 * math.pi * dimensions["MEAN_ANOMALY"][index],
            }
            rows.append(row)
            canonical_rows.append({
                "logical_id": row["logical_id"], "block_index": block,
                "index_within_block": index, "a_AU_hex": a.hex(), "q_AU_hex": q.hex(),
                "e_hex": e.hex(), "i_rad_hex": inc.hex(),
                "Omega_rad_hex": row["Omega_rad"].hex(),
                "omega_rad_hex": row["omega_rad"].hex(), "M_rad_hex": row["M_rad"].hex(),
            })
    if test is None or test[0] != policy["block_0_log_a_permutation_test_vector"] \
            or [value.hex() for value in test[1][:4]] != policy["block_0_log_a_first_four_lhs_float_hex"]:
        raise ValueError("LHS test vector mismatch")
    if hashlib.sha256(TRACER_DOMAIN + canonical(canonical_rows)).hexdigest() \
            != policy["canonical_rows_sha256"]:
        raise ValueError("canonical tracer digest mismatch")
    return rows


def eccentric_anomaly(mean_anomaly: float, eccentricity: float) -> float:
    """Exactly 32 binary64 Newton updates; no convergence-based early exit."""
    value = mean_anomaly if eccentricity < 0.8 else math.pi
    for _iteration in range(32):
        value = value - (
            value - eccentricity * math.sin(value) - mean_anomaly
        ) / (1.0 - eccentricity * math.cos(value))
    return value


def cartesian(
    gravitational_constant: float, primary_mass: float, orbiting_mass: float,
    a: float, e: float, inc: float, ascending_node: float,
    argument_perihelion: float, mean_anomaly: float,
) -> list[float]:
    anomaly = eccentric_anomaly(mean_anomaly, e)
    cosine_e = math.cos(anomaly); sine_e = math.sin(anomaly)
    beta = math.sqrt(1.0 - e * e)
    perifocal_x = a * (cosine_e - e)
    perifocal_y = a * beta * sine_e
    mu = gravitational_constant * (primary_mass + orbiting_mass)
    mean_motion = math.sqrt(mu / (a * a * a))
    denominator = 1.0 - e * cosine_e
    perifocal_vx = -a * mean_motion * sine_e / denominator
    perifocal_vy = a * mean_motion * beta * cosine_e / denominator
    cosine_node = math.cos(ascending_node); sine_node = math.sin(ascending_node)
    cosine_omega = math.cos(argument_perihelion); sine_omega = math.sin(argument_perihelion)
    cosine_inc = math.cos(inc); sine_inc = math.sin(inc)
    p = (
        cosine_node * cosine_omega - sine_node * sine_omega * cosine_inc,
        sine_node * cosine_omega + cosine_node * sine_omega * cosine_inc,
        sine_omega * sine_inc,
    )
    q = (
        -cosine_node * sine_omega - sine_node * cosine_omega * cosine_inc,
        -sine_node * sine_omega + cosine_node * cosine_omega * cosine_inc,
        cosine_omega * sine_inc,
    )
    return [perifocal_x * p[axis] + perifocal_y * q[axis] for axis in range(3)] + [
        perifocal_vx * p[axis] + perifocal_vy * q[axis] for axis in range(3)
    ]


def factorized_row(logical_id: str, role: str, mass: float, state: Sequence[float]) -> list[str]:
    return [logical_id, role, pack(mass), pack6(state)]


def generate(contract: dict[str, Any], seed_manifest: dict[str, Any]) -> dict[str, Any]:
    core = contract["design_core"]; G = core["units_and_frame"]["G_AU3_Msun_yr2"]
    active = core["common_active_system"]
    common = [factorized_row("Sun", "A", active["sun_mass_Msun"], [0.0] * 6)]
    for planet in active["giants"]:
        state = cartesian(G, active["sun_mass_Msun"], planet["mass_Msun"],
                          planet["a_AU"], 0.0, 0.0, 0.0, 0.0,
                          math.radians(planet["initial_longitude_deg"]))
        common.append(factorized_row(planet["name"], "A", planet["mass_Msun"], state))
    tracers = tracer_elements(contract, seed_manifest)
    tracer_rows = [factorized_row(
        row["logical_id"], "T", 0.0,
        cartesian(G, active["sun_mass_Msun"], 0.0, row["a_AU"], row["e"],
                  row["i_rad"], row["Omega_rad"], row["omega_rad"], row["M_rad"]),
    ) for row in tracers]
    models = {row["id"]: row for row in core["m1_physical_cases"]}
    probes = {row["id"]: row for row in core["orientation_probes"]}
    configurations = []; digest_index = []
    for arm_id in core["primary_arm_ids"]:
        added = None
        active_rows = list(common)
        if arm_id != "M0":
            model_id, probe_id = arm_id.split("-"); model = models[model_id]; probe = probes[probe_id]
            mass = model["mass_Mearth"] * active["earth_to_sun_mass_ratio"]
            omega_deg = (probe["varpi_deg"] - probe["Omega_deg"]) % 360.0
            state = cartesian(G, active["sun_mass_Msun"], mass, model["a_AU"], model["e"],
                              math.radians(model["i_deg"]), math.radians(probe["Omega_deg"]),
                              math.radians(omega_deg), math.radians(probe["M_deg"]))
            added = factorized_row(f"XP2-{arm_id}", "A", mass, state)
            active_rows.append(added)
        masses = [f64(row[2]) for row in active_rows]
        states = [unpack6(row[3]) for row in active_rows]
        total_mass = math.fsum(masses)
        com_position = [math.fsum(mass * state[axis] for mass, state in zip(masses, states, strict=True))
                        / total_mass for axis in range(3)]
        com_velocity = [math.fsum(mass * state[axis + 3]
                                  for mass, state in zip(masses, states, strict=True))
                        / total_mass for axis in range(3)]
        source = active_rows + tracer_rows; expanded = []
        for logical_id, role, mass_hex, state_hex in source:
            state = unpack6(state_hex)
            shifted = [state[axis] - com_position[axis] for axis in range(3)] + [
                state[axis + 3] - com_velocity[axis] for axis in range(3)
            ]
            expanded.append([logical_id, role, mass_hex, pack6(shifted)])
        state_hash = hashlib.sha256(EXPANDED_DOMAIN + canonical(expanded)).hexdigest()
        configurations.append([
            arm_id, len(active_rows), added, pack3(com_position), pack3(com_velocity), state_hash
        ])
        digest_index.append([arm_id, state_hash])
    return {
        "schema": "jx-xp2-barycentric-initial-states/v1", "experiment_id": EXPERIMENT_ID,
        "artifact_class": "PREOUTPUT_EXACT_BINARY64_INITIAL_STATE_FACTORIZATION",
        "canonicalization": "UTF8_JSON_SORT_KEYS_TRUE_SEPARATORS_COMMA_COLON_ENSURE_ASCII_TRUE_ALLOW_NAN_FALSE",
        "representation": "PACKED_BIG_ENDIAN_IEEE754_BINARY64_HEX_EXACT_EXPANDABLE_ROWS",
        "expansion_rule": "UNPACK_BIG_ENDIAN_BINARY64; SUBTRACT_STORED_ACTIVE_COM_POSITION_FROM_POSITION_AND_ACTIVE_COM_VELOCITY_FROM_VELOCITY_IN_BINARY64; REPACK_BIG_ENDIAN_BINARY64",
        "row_order": "COMMON_ACTIVE_SUN_FOUR_GIANTS_THEN_OPTIONAL_ADDED_BODY_THEN_128_TRACERS",
        "row_tuple_fields": ["logical_id", "role_A_OR_T", "mass_binary64_be_hex", "state_6x_binary64_be_hex"],
        "state_component_order": ["x_AU", "y_AU", "z_AU", "vx_AU_per_year", "vy_AU_per_year", "vz_AU_per_year"],
        "common_active_sun_centered_rows": common, "tracer_sun_centered_rows": tracer_rows,
        "configuration_tuple_fields": ["arm_id", "active_count", "added_body_sun_centered_row_or_null",
            "active_com_position_3x_binary64_be_hex", "active_com_velocity_3x_binary64_be_hex",
            "expanded_barycentric_rows_sha256"],
        "configuration_states": configurations,
        "expanded_digest": "SHA256(ASCII_jx-xp2-expanded-barycentric-state/v1 || BYTE_00 || CANONICAL_EXPANDED_ROW_TUPLES)",
        "configuration_digest_index": "SHA256(ASCII_jx-xp2-configuration-digest-index/v1 || BYTE_00 || CANONICAL_LIST_OF_ARM_ID_AND_EXPANDED_DIGEST_PAIRS_IN_CONFIGURATION_ORDER)",
        "configuration_digest_index_sha256": hashlib.sha256(
            INDEX_DOMAIN + canonical(digest_index)
        ).hexdigest(),
        "independent_element_to_cartesian_recomputation_required": True,
        "dynamics_or_outcomes_generated": False,
        "mandatory_nonclaim": "Exact synthetic numerical inputs only; not observed data or evidence for or against Planet X.",
    }


def f64(value: str) -> float:
    return struct.unpack(">d", bytes.fromhex(value))[0]


def unpack6(value: str) -> list[float]:
    return [f64(value[index:index + 16]) for index in range(0, 96, 16)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    value = generate(json.loads(args.contract.read_text()), json.loads(args.seed_manifest.read_text()))
    payload = canonical(value) + b"\n"
    if args.verify is not None:
        if args.verify.read_bytes() != payload:
            raise SystemExit("initial-state artifact differs from deterministic reconstruction")
        print(json.dumps({"status": "PASS", "sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))
    else:
        sys.stdout.buffer.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
