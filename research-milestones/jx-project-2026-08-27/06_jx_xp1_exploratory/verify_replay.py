#!/usr/bin/env python3
"""Independent, no-dynamics verifier for two registered JX-XP1 executions."""

from __future__ import annotations

import argparse
import decimal
import hashlib
import importlib
import importlib.util
import json
import math
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any


EXPERIMENT = "jx-xp1-public-synthetic-response-v1"
CONTRACT_SHA = "dd4527ef2b7d61bda93395d9dec7107b57c962c88aee3f1f1032af60dd055d63"
SEED_SHA = "92de9fae8c32f322c58216c64355739917dddee2881e823541b7fbad791e1ac7"
DESIGN_SHA = "0865266fa46b3cdf080d783f366f4988a76fb1667bf334bd79b005e9ad68380c"
TRACER_SHA = "b98c8c27f3301f54afff72a0b71847e1508d6ed51dc2ce566c4ca9daec7133ab"
PYTHON_SHA = "021044895e95be79dc2f110367607e684119afbc8ce75f6f0eec94844e0acec7"
REBOUND_BINARY_SHA = "fe7a23bcece1c3f1f869089e9e8d806bedb4727d893d2e551339adbb6665c28a"
REBOUND_SOURCE_SHA = "2c40b16571d57049cbf4bb8329a0c58342f3dc0f0cf49d860ca77fda5a73ae3a"
REBOUND_SOURCE_COUNT = 29
PRIMARY = ("M0", "CI01-A", "CI01-B", "CI05-A", "CI05-B", "CI09-A", "CI09-B")
AUDITS = tuple(f"AUDIT-{arm_id}" for arm_id in PRIMARY)
AUDIT_TO_PRIMARY = {f"AUDIT-{arm_id}": arm_id for arm_id in PRIMARY}
LOCKED_FILES = {
    "README.md", "contract_v1.json", "seed_manifest_v1.json",
    "run_exploratory.py", "verify_replay.py", "test_exploratory.py",
}
LHS_SUFFIXES = ("LOG_A", "Q", "COS_I", "OMEGA", "OMEGA_ARGUMENT", "MEAN_ANOMALY")
LHS_DOMAIN = b"jx-xp1-lhs-u64/v1\0"
TREE_DOMAIN = b"jx-e2-rebound-python-sources/v1\0"
EPS = sys.float_info.epsilon
_SOURCE_CACHE: tempfile.TemporaryDirectory[str] | None = None
ANALYSIS_COMPLETE_STATUS = "COMPLETE_AT_BOTH_RESOLUTIONS"
ANALYSIS_SUPPRESSED_STATUS = "SUPPRESSED_REQUIRED_FINITE_ORBIT_METRICS_INCOMPLETE"
INDEPENDENT_RECOMPUTATION_KEYS = {
    "seeds_and_64_tracers",
    "all_14_initial_states_without_integration",
    "particle_summaries",
    "two_resolution_integer_effects_and_block_effects_or_locked_suppression",
    "wasserstein1_or_locked_suppression",
    "all_seven_timestep_pairs_or_locked_suppression",
    "raw_class_agreement_and_emitted_classification_or_locked_suppression",
}


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def finite_number(text: str) -> float:
    value = float(text)
    exact = decimal.Decimal(text)
    if not math.isfinite(value) or not exact.is_finite() or (value == 0.0 and exact != 0):
        raise ValueError("non-finite JSON number")
    return value


def reject_constant(text: str) -> None:
    raise ValueError(f"nonstandard JSON constant: {text}")


def read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f"artifact is not a unique regular file: {path}")
    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs,
        parse_float=finite_number, parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("JSON root is not an object")
    return value


def write_once(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite replay receipt: {path}")
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending")
    if pending.exists():
        raise FileExistsError("stale replay-receipt pending file exists")
    try:
        with pending.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
    except BaseException:
        try:
            pending.unlink()
        except FileNotFoundError:
            pass
        raise


def trees_overlap(left: Path, right: Path) -> bool:
    left=left.resolve();right=right.resolve()
    return left==right or left.is_relative_to(right) or right.is_relative_to(left)


def protected_tree_roots(contract: dict[str, Any], package_root: Path) -> list[Path]:
    return [package_root.resolve()]+[
        (package_root/binding["path"]).resolve().parent
        for binding in contract["excluded_context_bindings"]
    ]


def source_tree(root: Path) -> tuple[int, str]:
    files = sorted(root.rglob("*.py"), key=lambda item: item.relative_to(root).as_posix())
    digest = hashlib.sha256(TREE_DOMAIN)
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return len(files), digest.hexdigest()


def guarded_rebound() -> Any:
    global _SOURCE_CACHE
    module = sys.modules.get("rebound")
    if module is None:
        specification = importlib.util.find_spec("rebound")
        if specification is None or not specification.submodule_search_locations:
            raise RuntimeError("cannot find REBOUND")
        root = Path(next(iter(specification.submodule_search_locations))).resolve()
        if source_tree(root) != (REBOUND_SOURCE_COUNT, REBOUND_SOURCE_SHA):
            raise RuntimeError("REBOUND source mismatch")
        binaries = sorted(root.parent.glob("librebound*.so"))
        if len(binaries) != 1 or digest_file(binaries[0]) != REBOUND_BINARY_SHA:
            raise RuntimeError("REBOUND native library mismatch")
        _SOURCE_CACHE = tempfile.TemporaryDirectory(prefix="jx-xp1-verify-source-")
        sys.pycache_prefix = _SOURCE_CACHE.name
        sys.dont_write_bytecode = True
        importlib.invalidate_caches()
        module = importlib.import_module("rebound")
        setattr(module, "_jx_xp1_verifier_source_guard", True)
    elif getattr(module, "_jx_xp1_verifier_source_guard", False) is not True:
        raise RuntimeError("REBOUND imported before verifier source guard")
    return module


def actual_runtime() -> dict[str, Any]:
    rebound = guarded_rebound()
    executable = Path(sys.executable).resolve()
    binary = Path(rebound.clibrebound._name).resolve()
    count, source_sha = source_tree(Path(rebound.__file__).resolve().parent)
    return {
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "python_executable_sha256": digest_file(executable),
        "rebound_version": rebound.__version__, "rebound_build": rebound.__build__,
        "rebound_binary_sha256": digest_file(binary),
        "rebound_python_source_file_count": count,
        "rebound_python_source_sha256": source_sha,
    }


def validate_registration(root: Path, registration_path: Path, contract: dict[str, Any]) -> tuple[dict[str, Any], str]:
    if registration_path.is_symlink():
        raise ValueError("registration path must not be a symlink")
    if registration_path.resolve() != root / "registration_v1.json":
        raise ValueError("noncanonical registration path")
    registration = read_json(registration_path)
    if set(registration) != {
        "schema", "experiment_id", "artifact_class", "registration_state",
        "recorded_at_utc", "timestamp_authority", "externally_timestamped",
        "scientific_evidence_artifact", "outcomes_generated", "execution_permissions",
        "locked_files", "mandatory_nonclaim",
    }:
        raise ValueError("registration shape mismatch")
    permissions = {
        "execution_a_authorized": True,
        "execution_b_authorized_only_after_verified_a": True,
        "local_cpu_only": True, "network_access_authorized": False,
        "gpu_execution_authorized": False, "observed_data_access_authorized": False,
        "survey_adapter_execution_authorized": False,
        "jx_o2_execution_or_g0_evidence_authorized": False,
        "planet_x_claim_authorized": False,
    }
    if (
        registration["schema"] != "jx-xp1-local-registration/v1"
        or registration["experiment_id"] != EXPERIMENT
        or registration["artifact_class"] != "LOCAL_CONTENT_HASH_REGISTRATION_ONLY"
        or registration["registration_state"] != "LOCAL_CONTENT_HASH_LOCK_COMPLETE_BEFORE_ANY_XP1_NUMERICAL_OUTPUT"
        or registration["timestamp_authority"] != "LOCAL_CONTENT_HASH_ONLY_NO_EXTERNAL_TIMESTAMP"
        or registration["externally_timestamped"] is not False
        or registration["scientific_evidence_artifact"] is not False
        or registration["outcomes_generated"] is not False
        or registration["execution_permissions"] != permissions
        or registration["mandatory_nonclaim"] != contract["mandatory_nonclaim"]
        or not registration["recorded_at_utc"].endswith("Z")
        or set(registration["locked_files"]) != LOCKED_FILES
    ):
        raise ValueError("registration policy mismatch")
    expected_package_inventory=set(contract["result_policy"]["registered_package_inventory"])
    if expected_package_inventory!=LOCKED_FILES|{"registration_v1.json"}:
        raise ValueError("contract registered-package inventory lock mismatch")
    if {path.name for path in root.iterdir()}!=expected_package_inventory:
        raise ValueError("registered package has an extra or missing filesystem entry")
    forbidden = {
        contract["retired_candidate_nonuse_lock"]["forbidden_metadata_sha256"],
        contract["retired_candidate_nonuse_lock"]["forbidden_historical_source_state_sha256"],
        contract["retired_candidate_nonuse_lock"]["forbidden_catalog_sha256"],
    }
    for relative, expected in registration["locked_files"].items():
        candidate=root/relative
        path = candidate.resolve()
        if candidate.is_symlink() or path!=candidate.absolute() or path.parent != root or not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError(f"locked path invalid: {relative}")
        actual = digest_file(path)
        if actual != expected or actual in forbidden:
            raise ValueError(f"locked file mismatch or forbidden payload: {relative}")
    return registration, digest_file(registration_path)


def seed_digest(domain: str, design_sha: str, label: str) -> bytes:
    domain_bytes = domain.encode("ascii")
    label_bytes = label.encode("ascii")
    payload = (
        len(domain_bytes).to_bytes(4, "big") + domain_bytes + bytes.fromhex(design_sha)
        + len(label_bytes).to_bytes(4, "big") + label_bytes + (0).to_bytes(8, "big")
    )
    return hashlib.sha256(payload).digest()


def independent_lhs(seed: bytes) -> tuple[list[int], list[float]]:
    counter = 0
    permutation = list(range(16))

    def word() -> int:
        nonlocal counter
        value = int.from_bytes(hashlib.sha256(LHS_DOMAIN + seed + counter.to_bytes(8, "big")).digest()[:8], "big")
        counter += 1
        return value

    for position in range(15, 0, -1):
        base = position + 1
        ceiling = (1 << 64) - ((1 << 64) % base)
        candidate = word()
        while candidate >= ceiling:
            candidate = word()
        destination = candidate % base
        permutation[position], permutation[destination] = permutation[destination], permutation[position]
    fractions = [(stratum + (word() >> 11) / (1 << 53)) / 16 for stratum in permutation]
    return permutation, fractions


def independently_realize_tracers(contract: dict[str, Any], seed_manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    expected_labels = [f"LHS_BLOCK_{block}_{suffix}" for block in range(4) for suffix in LHS_SUFFIXES]
    if [row["stream_label"] for row in seed_manifest["streams"]] != expected_labels:
        raise ValueError("seed stream ordering mismatch")
    seeds: dict[str, bytes] = {}
    for row in seed_manifest["streams"]:
        expected = seed_digest(seed_manifest["domain_ascii"], DESIGN_SHA, row["stream_label"])[:16]
        if row != {"stream_label": row["stream_label"], "counter": 0, "seed_hex_128": expected.hex()}:
            raise ValueError("seed row does not recompute")
        seeds[row["stream_label"]] = expected
    canonical_rows = []
    numeric_rows = []
    first_test = None
    for block in range(4):
        fractions = {}
        for suffix in LHS_SUFFIXES:
            value = independent_lhs(seeds[f"LHS_BLOCK_{block}_{suffix}"])
            fractions[suffix] = value[1]
            if block == 0 and suffix == "LOG_A":
                first_test = value
        for index in range(16):
            a = math.exp(math.log(150.0) + fractions["LOG_A"][index] * (math.log(800.0) - math.log(150.0)))
            q = 35.0 + 45.0 * fractions["Q"][index]
            e = 1.0 - q / a
            inc = math.acos(math.cos(math.radians(40.0)) + fractions["COS_I"][index] * (1.0 - math.cos(math.radians(40.0))))
            row = {
                "logical_id": f"XP1-B{block:02d}-T{index:02d}", "block_index": block,
                "index_within_block": index, "a_AU": a, "q_AU": q, "e": e,
                "i_rad": inc, "Omega_rad": 2 * math.pi * fractions["OMEGA"][index],
                "omega_rad": 2 * math.pi * fractions["OMEGA_ARGUMENT"][index],
                "M_rad": 2 * math.pi * fractions["MEAN_ANOMALY"][index],
            }
            numeric_rows.append(row)
            canonical_rows.append({
                "logical_id": row["logical_id"], "block_index": block,
                "index_within_block": index, "a_AU_hex": a.hex(), "q_AU_hex": q.hex(),
                "e_hex": e.hex(), "i_rad_hex": inc.hex(),
                "Omega_rad_hex": row["Omega_rad"].hex(), "omega_rad_hex": row["omega_rad"].hex(),
                "M_rad_hex": row["M_rad"].hex(),
            })
    test = contract["seed_policy"]["lhs_construction"]
    if first_test is None or first_test[0] != test["block_0_log_a_permutation_test_vector"]:
        raise ValueError("independent LHS permutation mismatch")
    if [value.hex() for value in first_test[1][:4]] != test["block_0_log_a_first_four_lhs_float_hex"]:
        raise ValueError("independent LHS jitter mismatch")
    tracer_sha = digest_bytes(b"jx-xp1-canonical-tracer-design/v1\0" + canonical(canonical_rows))
    if tracer_sha != TRACER_SHA or tracer_sha != test["canonical_rows_sha256"]:
        raise ValueError("independent tracer digest mismatch")
    return numeric_rows, tracer_sha


def vector_norm(values: tuple[float, ...]) -> float:
    return math.sqrt(math.fsum(value * value for value in values))


def subtract(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(a - b for a, b in zip(left, right, strict=True))


def independent_initial_bindings(contract: dict[str, Any], tracers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rebound = guarded_rebound()
    core = contract["design_core"]
    models = {row["id"]: row for row in core["m1_physical_cases"]}
    angles = {row["id"]: row for row in core["orientation_probes"]}

    def specs() -> list[tuple[str, str, float]]:
        return [(arm, arm, 0.125) for arm in PRIMARY] + [
            (audit, AUDIT_TO_PRIMARY[audit], 0.0625) for audit in AUDITS
        ]

    results = {}
    reference_digest = None
    for arm_id, physical_id, dt in specs():
        simulation = rebound.Simulation()
        simulation.G = core["units_and_frame"]["G_AU3_Msun_yr2"]
        system = core["common_active_system"]
        simulation.add(m=system["sun_mass_Msun"], hash="Sun")
        common = ["Sun"]
        for body in system["giants"]:
            simulation.add(primary=simulation.particles["Sun"], m=body["mass_Msun"], a=body["a_AU"], e=0, inc=0, Omega=0, omega=0, M=math.radians(body["initial_longitude_deg"]), hash=body["name"])
            common.append(body["name"])
        if physical_id != "M0":
            model_id, angle_id = physical_id.split("-")
            model, angle = models[model_id], angles[angle_id]
            simulation.add(
                primary=simulation.particles["Sun"],
                m=model["mass_Mearth"] * system["earth_to_sun_mass_ratio"],
                a=model["a_AU"], e=model["e"], inc=math.radians(model["i_deg"]),
                Omega=math.radians(angle["Omega_deg"]),
                omega=math.radians((angle["varpi_deg"] - angle["Omega_deg"]) % 360),
                M=math.radians(angle["M_deg"]), hash=f"XP1-{model_id}-{angle_id}",
            )
        simulation.N_active = simulation.N
        for tracer in tracers:
            simulation.add(primary=simulation.particles["Sun"], m=0, a=tracer["a_AU"], e=tracer["e"], inc=tracer["i_rad"], Omega=tracer["Omega_rad"], omega=tracer["omega_rad"], M=tracer["M_rad"], hash=tracer["logical_id"])
            common.append(tracer["logical_id"])
        sun = simulation.particles["Sun"]
        pre = []
        for name in common:
            particle = simulation.particles[name]
            pre.append({"logical_id": name, "components_hex": [
                (particle.x - sun.x).hex(), (particle.y - sun.y).hex(), (particle.z - sun.z).hex(),
                (particle.vx - sun.vx).hex(), (particle.vy - sun.vy).hex(), (particle.vz - sun.vz).hex(),
            ]})
        pre_sha = digest_bytes(b"jx-xp1-pre-com-common-relative-state/v1\0" + canonical(pre))
        if reference_digest is None:
            reference_digest = pre_sha
        elif pre_sha != reference_digest:
            raise ValueError("independent initial reconstruction lost exact pre-COM pairing")
        active = [simulation.particles[index] for index in range(simulation.N_active)]
        total = math.fsum(p.m for p in active)
        rcom = tuple(math.fsum(p.m * getattr(p, field) for p in active) / total for field in ("x", "y", "z"))
        vcom = tuple(math.fsum(p.m * getattr(p, field) for p in active) / total for field in ("vx", "vy", "vz"))
        for particle in simulation.particles:
            particle.x -= rcom[0]; particle.y -= rcom[1]; particle.z -= rcom[2]
            particle.vx -= vcom[0]; particle.vy -= vcom[1]; particle.vz -= vcom[2]
        maximum = 0.0
        sun = simulation.particles["Sun"]
        for expected in pre:
            particle = simulation.particles[expected["logical_id"]]
            actual = (particle.x-sun.x, particle.y-sun.y, particle.z-sun.z, particle.vx-sun.vx, particle.vy-sun.vy, particle.vz-sun.vz)
            for expected_hex, value in zip(expected["components_hex"], actual, strict=True):
                reference = float.fromhex(expected_hex)
                maximum = max(maximum, abs(value-reference)/(EPS*max(1.0, abs(reference))))
        simulation.integrator = "mercurius"; simulation.dt = dt; simulation.testparticle_type = 0
        simulation.ri_mercurius.r_crit_hill = 3.0; simulation.ri_mercurius.safe_mode = 1
        digest = hashlib.sha256(b"jx-xp1-decoded-state/v1\0")
        digest.update(canonical({
            "t_hex": simulation.t.hex(), "G_hex": simulation.G.hex(), "dt_hex": simulation.dt.hex(),
            "N": simulation.N, "N_active": simulation.N_active, "integrator": simulation.integrator,
            "testparticle_type": simulation.testparticle_type,
            "r_crit_hill_hex": simulation.ri_mercurius.r_crit_hill.hex(),
            "safe_mode": int(simulation.ri_mercurius.safe_mode),
        }))
        for index, particle in enumerate(simulation.particles):
            digest.update(struct.pack("!II8d", index, int(particle.hash.value), particle.m, particle.r, particle.x, particle.y, particle.z, particle.vx, particle.vy, particle.vz))
        results[arm_id] = {"pre_sha": pre_sha, "epsilon_units": maximum, "decoded_sha": digest.hexdigest()}
    return results


def w1(left: list[float], right: list[float]) -> float:
    if not left or not right or not all(math.isfinite(x) for x in left + right):
        raise ValueError("invalid W1 input")
    x, y = sorted(left), sorted(right)
    i = j = 0
    rx, ry = 1 / len(x), 1 / len(y)
    total = 0.0
    while i < len(x) and j < len(y):
        weight = min(rx, ry)
        total += weight * abs(x[i] - y[j])
        rx -= weight; ry -= weight
        if rx <= 4 * EPS:
            i += 1; rx = 1 / len(x) if i < len(x) else 0
        if ry <= 4 * EPS:
            j += 1; ry = 1 / len(y) if j < len(y) else 0
    if i != len(x) or j != len(y):
        raise ValueError("W1 mass imbalance")
    return total


def values(rows: list[dict[str, Any]], key: str) -> list[float]:
    result = [row[key] for row in rows]
    if any(item is None or not math.isfinite(item) for item in result):
        raise ValueError(f"incomplete metric: {key}")
    return result


def recompute_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    q30 = sum(
        row["minimum_sampled_q_AU"] is not None
        and row["minimum_sampled_q_AU"] < 30 for row in rows
    )
    q35 = sum(
        row["minimum_sampled_q_AU"] is not None
        and row["minimum_sampled_q_AU"] < 35 for row in rows
    )
    bound = sum(row["final_finite_and_bound"] for row in rows)
    return {
        "particle_count": 64, "q_below_30_hit_count": q30, "q_below_30_fraction": q30/64,
        "q_below_35_hit_count": q35, "q_below_35_fraction": q35/64,
        "final_finite_bound_count": bound, "final_finite_bound_fraction": bound/64,
        "all_particles_have_complete_finite_osculating_history": all(
            row["all_samples_finite_osculating_orbit"] for row in rows
        ),
    }


def recompute_analysis(arms: dict[str, Any]) -> dict[str, Any]:
    control = arms["M0"]
    configurations = []
    for arm_id in PRIMARY[1:]:
        source = arms[arm_id]
        row: dict[str, Any] = {"configuration_id": arm_id}
        for name, field in (("q_below_30","q_below_30_hit_count"),("q_below_35","q_below_35_hit_count"),("final_finite_bound","final_finite_bound_count")):
            numerator = source["summary"][field] - control["summary"][field]
            row[f"{name}_effect_numerator"] = numerator; row[f"{name}_effect_denominator"] = 64; row[f"{name}_effect"] = numerator/64
        row["w1_minimum_sampled_q_AU"] = w1(values(source["particle_metrics"],"minimum_sampled_q_AU"),values(control["particle_metrics"],"minimum_sampled_q_AU"))
        row["w1_final_q_AU"] = w1(values(source["particle_metrics"],"final_q_AU"),values(control["particle_metrics"],"final_q_AU"))
        row["w1_final_i_deg"] = w1(values(source["particle_metrics"],"final_i_deg"),values(control["particle_metrics"],"final_i_deg"))
        configurations.append(row)
    mixture = {}
    for name, field in (("q_below_30","q_below_30_hit_count"),("q_below_35","q_below_35_hit_count"),("final_finite_bound","final_finite_bound_count")):
        numerator = sum(arms[arm]["summary"][field] for arm in PRIMARY[1:]) - 6*control["summary"][field]
        mixture[f"{name}_effect_numerator"] = numerator; mixture[f"{name}_effect_denominator"] = 384; mixture[f"{name}_effect"] = numerator/384
    for output, field in (("w1_minimum_sampled_q_AU","minimum_sampled_q_AU"),("w1_final_q_AU","final_q_AU"),("w1_final_i_deg","final_i_deg")):
        mixture[output] = w1([x for arm in PRIMARY[1:] for x in values(arms[arm]["particle_metrics"],field)],values(control["particle_metrics"],field)*6)
    blocks=[]
    for block in range(4):
        row={"block_index":block}; indices=range(block*16,(block+1)*16)
        for name, threshold in (("q_below_30",30),("q_below_35",35)):
            source=sum(arms[arm]["particle_metrics"][i]["minimum_sampled_q_AU"]<threshold for arm in PRIMARY[1:] for i in indices)
            base=sum(control["particle_metrics"][i]["minimum_sampled_q_AU"]<threshold for i in indices)
            numerator=source-6*base;row[f"{name}_effect_numerator"]=numerator;row[f"{name}_effect_denominator"]=96;row[f"{name}_effect"]=numerator/96
        blocks.append(row)
    return {"configuration_effects":configurations,"mixture_effects":mixture,"block_effects":blocks}


def recompute_timestep(audit: dict[str, Any], primary: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    left,right=audit["particle_metrics"],primary["particle_metrics"]
    differences={
        "q_below_30_fraction_difference":abs(audit["summary"]["q_below_30_fraction"]-primary["summary"]["q_below_30_fraction"]),
        "q_below_35_fraction_difference":abs(audit["summary"]["q_below_35_fraction"]-primary["summary"]["q_below_35_fraction"]),
        "final_finite_bound_fraction_difference":abs(audit["summary"]["final_finite_bound_fraction"]-primary["summary"]["final_finite_bound_fraction"]),
        "w1_minimum_sampled_q_AU":w1(values(left,"minimum_sampled_q_AU"),values(right,"minimum_sampled_q_AU")),
        "w1_final_q_AU":w1(values(left,"final_q_AU"),values(right,"final_q_AU")),
        "w1_final_i_deg":w1(values(left,"final_i_deg"),values(right,"final_i_deg")),
    }
    checks={
        "q_below_30_fraction_within_gate":differences["q_below_30_fraction_difference"]<=gates["max_dt_half_q_below_30_fraction_difference"],
        "q_below_35_fraction_within_gate":differences["q_below_35_fraction_difference"]<=gates["max_dt_half_q_below_35_fraction_difference"],
        "final_finite_bound_fraction_within_gate":differences["final_finite_bound_fraction_difference"]<=gates["max_dt_half_final_finite_bound_fraction_difference"],
        "w1_minimum_sampled_q_within_gate":differences["w1_minimum_sampled_q_AU"]<=gates["max_dt_half_w1_minimum_sampled_q_AU"],
        "w1_final_q_within_gate":differences["w1_final_q_AU"]<=gates["max_dt_half_w1_final_q_AU"],
        "w1_final_i_within_gate":differences["w1_final_i_deg"]<=gates["max_dt_half_w1_final_i_deg"],
    }
    return {"audit_arm_id":audit["arm_id"],"primary_arm_id":primary["arm_id"],"differences":differences,"checks":checks}


def expected_raw_classification(analysis: dict[str, Any]) -> str:
    numerator=int(analysis["mixture_effects"]["q_below_30_effect_numerator"]);blocks=[int(r["q_below_30_effect_numerator"]) for r in analysis["block_effects"]]
    if numerator>=20 and all(x>=1 for x in blocks):return "DIRECTIONALLY_STABLE_INCREASE"
    if numerator<=-20 and all(x<=-1 for x in blocks):return "DIRECTIONALLY_STABLE_DECREASE"
    if abs(numerator)<=7 and max(abs(x) for x in blocks)<=4:return "PRACTICALLY_SMALL"
    return "INCONCLUSIVE"


def validate_output(
    root: Path, contract: dict[str, Any], registration: dict[str, Any],
    registration_sha: str, label: str, initial: dict[str, Any], runtime: dict[str, Any],
) -> dict[str, Any]:
    root=root.resolve()
    if root.is_symlink() or not root.is_dir() or {p.name for p in root.iterdir()}!={"run_manifest.json","result_v1.json"}:
        raise ValueError(f"closed successful output inventory mismatch: {root}")
    manifest=read_json(root/"run_manifest.json");result=read_json(root/"result_v1.json")
    if set(manifest)!={"schema","experiment_id","contract_sha256","seed_manifest_sha256","registration_sha256","runner_sha256","verifier_sha256","execution_label","execution_instance_id","runtime"}:
        raise ValueError("manifest shape mismatch")
    expected_manifest={
        "schema":"jx-xp1-run-manifest/v1","experiment_id":EXPERIMENT,"contract_sha256":CONTRACT_SHA,
        "seed_manifest_sha256":SEED_SHA,"registration_sha256":registration_sha,
        "runner_sha256":registration["locked_files"]["run_exploratory.py"],
        "verifier_sha256":registration["locked_files"]["verify_replay.py"],
        "execution_label":label,"execution_instance_id":digest_bytes(b"jx-xp1-execution-instance/v1\0"+bytes.fromhex(registration_sha)+label.encode())[:32],
        "runtime":runtime,
    }
    if canonical(manifest)!=canonical(expected_manifest):raise ValueError("manifest content mismatch")
    if set(result)!={"schema","experiment_id","state","claim_ceiling","mandatory_nonclaim","semantic_sha256","semantic","provenance"}:
        raise ValueError("result shape mismatch")
    if result["schema"]!="jx-xp1-synthetic-response-result/v1" or result["experiment_id"]!=EXPERIMENT:
        raise ValueError("result identity mismatch")
    semantic=result["semantic"]
    if set(semantic)!={
        "schema","experiment_id","claim_ceiling","mandatory_nonclaim",
        "design_bindings","matrix","arms","analysis","timestep_all_arm_comparisons",
        "gate_summary","exploratory_classification","execution_state",
    }:
        raise ValueError("semantic top-level shape mismatch")
    if result["semantic_sha256"]!=digest_bytes(canonical(semantic)) or result["state"]!=semantic["execution_state"]:
        raise ValueError("semantic hash/state mismatch")
    if result["claim_ceiling"]!=contract["claim_ceiling"] or result["mandatory_nonclaim"]!=contract["mandatory_nonclaim"]:
        raise ValueError("result claim boundary mismatch")
    if (
        semantic["schema"]!="jx-xp1-synthetic-response-semantic/v1"
        or semantic["experiment_id"]!=EXPERIMENT
        or semantic["claim_ceiling"]!=contract["claim_ceiling"]
        or semantic["mandatory_nonclaim"]!=contract["mandatory_nonclaim"]
        or semantic["execution_state"] not in {"NUMERICALLY_UNRESOLVED","EXPLORATORY_COMPLETE"}
    ):raise ValueError("semantic identity/claim/state mismatch")
    expected_bindings={
        "contract_sha256":CONTRACT_SHA,"seed_manifest_sha256":SEED_SHA,
        "registration_sha256":registration_sha,"tracer_rows_sha256":TRACER_SHA,
        "runner_sha256":registration["locked_files"]["run_exploratory.py"],
        "verifier_sha256":registration["locked_files"]["verify_replay.py"],
        "test_sha256":registration["locked_files"]["test_exploratory.py"],
    }
    if semantic["design_bindings"]!=expected_bindings:raise ValueError("semantic design binding mismatch")
    expected_matrix={
        "primary_arm_ids":list(PRIMARY),"audit_arm_ids":list(AUDITS),
        "tracer_count_in_every_arm":64,"analysis_block_count":4,
        "samples_including_t0":5001,
    }
    if canonical(semantic["matrix"])!=canonical(expected_matrix):raise ValueError("semantic matrix mismatch")
    if set(semantic["arms"])!=set(PRIMARY+AUDITS):raise ValueError("arm inventory mismatch")
    expected_ids=[f"XP1-B{b:02d}-T{i:02d}" for b in range(4) for i in range(16)]
    numerical_arm_pass=True
    for arm_id,arm in semantic["arms"].items():
        if set(arm)!={
            "arm_id","arm_class","primary_arm_id","model_id","orientation_id",
            "dt_years","duration_years","sample_count_including_t0",
            "pre_translation_common_relative_state_sha256",
            "post_com_max_binary64_epsilon_units","decoded_initial_state_sha256",
            "decoded_final_state_sha256","sampled_state_stream_sha256",
            "maximum_invariant_metrics","summary","checks","particle_metrics",
        }:raise ValueError(f"arm shape mismatch: {arm_id}")
        expected_class="PRIMARY" if arm_id in PRIMARY else "TIMESTEP_SENTINEL"
        expected_primary=None if arm_id in PRIMARY else AUDIT_TO_PRIMARY[arm_id]
        expected_dt=.125 if arm_id in PRIMARY else .0625
        physical=arm_id if arm_id in PRIMARY else expected_primary
        expected_model=None if physical=="M0" else physical.split("-")[0]
        expected_angle=None if physical=="M0" else physical.split("-")[1]
        if (
            arm["arm_id"]!=arm_id or arm["arm_class"]!=expected_class
            or arm["primary_arm_id"]!=expected_primary or arm["model_id"]!=expected_model
            or arm["orientation_id"]!=expected_angle or arm["dt_years"]!=expected_dt
            or arm["duration_years"]!=250000.0 or arm["sample_count_including_t0"]!=5001
        ):raise ValueError(f"arm identity/dynamics mismatch: {arm_id}")
        if (
            type(arm["dt_years"]) is not float
            or type(arm["duration_years"]) is not float
            or type(arm["sample_count_including_t0"]) is not int
        ):raise ValueError("arm dynamics scalar type mismatch")
        for key in (
            "pre_translation_common_relative_state_sha256","decoded_initial_state_sha256",
            "decoded_final_state_sha256","sampled_state_stream_sha256",
        ):
            if not is_sha256(arm[key]):raise ValueError(f"malformed arm hash: {arm_id}/{key}")
        if (
            type(arm["post_com_max_binary64_epsilon_units"]) is not float
            or not math.isfinite(arm["post_com_max_binary64_epsilon_units"])
            or not 0<=arm["post_com_max_binary64_epsilon_units"]<=64
        ):raise ValueError("invalid post-COM tolerance metric")
        if set(arm["maximum_invariant_metrics"])!={
            "relative_active_energy_drift",
            "relative_active_com_angular_momentum_vector_drift",
            "scale_normalized_active_linear_momentum_residual",
        } or any(
            type(value) is not float or not math.isfinite(value) or value<0
            for value in arm["maximum_invariant_metrics"].values()
        ):raise ValueError("invariant maximum shape/value mismatch")
        if set(arm["summary"])!={
            "particle_count","q_below_30_hit_count","q_below_30_fraction",
            "q_below_35_hit_count","q_below_35_fraction","final_finite_bound_count",
            "final_finite_bound_fraction","all_particles_have_complete_finite_osculating_history",
        }:raise ValueError("arm summary shape mismatch")
        summary=arm["summary"]
        if (
            type(summary["particle_count"]) is not int
            or any(type(summary[key]) is not int for key in (
                "q_below_30_hit_count","q_below_35_hit_count","final_finite_bound_count"
            ))
            or any(type(summary[key]) is not float for key in (
                "q_below_30_fraction","q_below_35_fraction","final_finite_bound_fraction"
            ))
            or type(summary["all_particles_have_complete_finite_osculating_history"]) is not bool
        ):raise ValueError("arm summary scalar type mismatch")
        expected_check_keys={
            "sample_count_exact","final_time_exact","particle_count_unchanged",
            "active_particle_count_unchanged","integrator_and_dt_readback_exact",
            "r_crit_hill_and_safe_mode_readback_exact","pre_translation_common_state_exact",
            "post_com_common_state_within_epsilon_gate","all_samples_cartesian_finite",
            "all_particles_have_complete_finite_osculating_history",
            "active_energy_drift_within_gate","active_com_angular_momentum_drift_within_gate",
            "active_linear_momentum_residual_within_gate",
        }
        if set(arm["checks"])!=expected_check_keys or any(type(value) is not bool for value in arm["checks"].values()):raise ValueError("arm check shape/type mismatch")
        rows=arm["particle_metrics"]
        if [r["logical_id"] for r in rows]!=expected_ids or [r["block_index"] for r in rows]!=[b for b in range(4) for _ in range(16)]:raise ValueError("particle identity/order mismatch")
        for row_index,row in enumerate(rows):
            if set(row)!={
                "logical_id","block_index","index_within_block","minimum_sampled_q_AU",
                "first_sampled_q_below_35_time_year","first_sampled_q_below_30_time_year",
                "final_a_AU","final_e","final_i_deg","final_q_AU","final_distance_AU",
                "final_finite_and_bound","all_samples_finite_osculating_orbit",
            }:raise ValueError("particle metric shape mismatch")
            expected_block=row_index//16;expected_within=row_index%16
            if (
                type(row["block_index"]) is not int
                or type(row["index_within_block"]) is not int
                or row["block_index"]!=expected_block
                or row["index_within_block"]!=expected_within
            ):raise ValueError("particle block/index mismatch")
            if type(row["final_finite_and_bound"]) is not bool or type(row["all_samples_finite_osculating_orbit"]) is not bool:raise ValueError("particle booleans are not JSON booleans")
            minimum=row["minimum_sampled_q_AU"]
            if minimum is not None and (
                type(minimum) is not float or not math.isfinite(minimum) or minimum<0
            ):raise ValueError("invalid minimum sampled q")
            final_keys=("final_a_AU","final_e","final_i_deg","final_q_AU")
            final_values=[row[key] for key in final_keys]
            final_complete=all(value is not None for value in final_values)
            if any(value is None for value in final_values) and final_complete:
                raise AssertionError("unreachable mixed final-orbit state")
            if not final_complete and any(value is not None for value in final_values):
                raise ValueError("partially populated final osculating orbit")
            if final_complete:
                if any(type(value) is not float or not math.isfinite(value) for value in final_values):
                    raise ValueError("non-finite final osculating orbit")
                final_a,final_e,final_i,final_q=final_values
                if final_e<0 or final_q<0 or not 0<=final_i<=180:
                    raise ValueError("final osculating orbit outside physical schema")
                if final_q!=final_a*(1.0-final_e):
                    raise ValueError("final q is inconsistent with final a/e")
                if minimum is None or minimum>final_q:
                    raise ValueError("sampled minimum does not include final q")
            distance=row["final_distance_AU"]
            if type(distance) is not float or not math.isfinite(distance) or distance<0:
                raise ValueError("invalid final Cartesian distance")
            expected_bound=bool(final_complete and final_values[0]>0 and final_values[1]<1)
            if row["final_finite_and_bound"] is not expected_bound:
                raise ValueError("final bound flag is inconsistent with a/e")
            if row["all_samples_finite_osculating_orbit"] and not final_complete:
                raise ValueError("complete orbit history lacks final orbit")
            if row["all_samples_finite_osculating_orbit"] and minimum is None:
                raise ValueError("complete orbit history lacks sampled minimum")
            for threshold,key in ((35,"first_sampled_q_below_35_time_year"),(30,"first_sampled_q_below_30_time_year")):
                first=row[key]
                crossed=minimum is not None and minimum<threshold
                if crossed!=(first is not None):raise ValueError("crossing/minimum inconsistency")
                if first is not None and (
                    type(first) is not float or not math.isfinite(first)
                    or first<0 or first>250000 or first%50!=0
                ):raise ValueError("crossing time off grid")
        if canonical(arm["summary"])!=canonical(recompute_summary(rows)):raise ValueError(f"arm summary mismatch: {arm_id}")
        if arm["pre_translation_common_relative_state_sha256"]!=initial[arm_id]["pre_sha"] or arm["decoded_initial_state_sha256"]!=initial[arm_id]["decoded_sha"] or arm["post_com_max_binary64_epsilon_units"]!=initial[arm_id]["epsilon_units"]:raise ValueError("initial-state binding mismatch")
        maximum=arm["maximum_invariant_metrics"];g=contract["numerical_gates"]
        derived={
            "active_energy_drift_within_gate":maximum["relative_active_energy_drift"]<=g["max_relative_active_energy_drift"],
            "active_com_angular_momentum_drift_within_gate":maximum["relative_active_com_angular_momentum_vector_drift"]<=g["max_relative_active_com_angular_momentum_vector_drift"],
            "active_linear_momentum_residual_within_gate":maximum["scale_normalized_active_linear_momentum_residual"]<=g["max_scale_normalized_active_linear_momentum_residual"],
            "all_particles_have_complete_finite_osculating_history":arm["summary"]["all_particles_have_complete_finite_osculating_history"],
        }
        for key,value in derived.items():
            if arm["checks"][key] is not value:raise ValueError(f"derived arm check mismatch: {arm_id}/{key}")
        for key in ("sample_count_exact","final_time_exact","particle_count_unchanged","active_particle_count_unchanged","integrator_and_dt_readback_exact","r_crit_hill_and_safe_mode_readback_exact","pre_translation_common_state_exact","post_com_common_state_within_epsilon_gate","all_samples_cartesian_finite"):
            if arm["checks"][key] is not True:raise ValueError(f"integrity check false: {arm_id}/{key}")
        numerical_arm_pass &= all(derived.values())
    metrics_complete=all(
        arm["summary"]["all_particles_have_complete_finite_osculating_history"]
        for arm in semantic["arms"].values()
    )
    if metrics_complete:
        primary_analysis=recompute_analysis(semantic["arms"])
        audit_view={primary_id:{**semantic["arms"][f"AUDIT-{primary_id}"],"arm_id":primary_id} for primary_id in PRIMARY}
        audit_analysis=recompute_analysis(audit_view)
        primary_raw=expected_raw_classification(primary_analysis);audit_raw=expected_raw_classification(audit_analysis)
        analysis={"status":ANALYSIS_COMPLETE_STATUS,"primary_dt":primary_analysis,"audit_dt_half":audit_analysis,"primary_raw_classification":primary_raw,"audit_raw_classification":audit_raw}
        timestep=[recompute_timestep(semantic["arms"][a],semantic["arms"][AUDIT_TO_PRIMARY[a]],contract["numerical_gates"]) for a in AUDITS]
        timestep_pass=all(v for row in timestep for v in row["checks"].values())
        class_equal=primary_raw==audit_raw
    else:
        primary_raw=None
        analysis={"status":ANALYSIS_SUPPRESSED_STATUS,"primary_dt":None,"audit_dt_half":None,"primary_raw_classification":None,"audit_raw_classification":None}
        timestep=[]
        timestep_pass=False
        class_equal=False
    if canonical(semantic["analysis"])!=canonical(analysis):raise ValueError("independent two-resolution analysis or suppression recomputation mismatch")
    if canonical(semantic["timestep_all_arm_comparisons"])!=canonical(timestep):raise ValueError("independent timestep recomputation or suppression mismatch")
    numerical=numerical_arm_pass and timestep_pass and class_equal
    gates={"integrity_pass":True,"conservation_and_finite_orbit_history_pass":numerical_arm_pass,"all_seven_timestep_pairs_pass":timestep_pass,"primary_and_audit_raw_classifications_exact":class_equal,"all_numerical_gates_pass":numerical,"timestep_scope":contract["numerical_gates"]["timestep_scope"]}
    if canonical(semantic["gate_summary"])!=canonical(gates):raise ValueError("gate summary mismatch")
    state="EXPLORATORY_COMPLETE" if numerical else "NUMERICALLY_UNRESOLVED"
    emitted=primary_raw if numerical else "SUPPRESSED_NUMERICALLY_UNRESOLVED"
    if semantic["execution_state"]!=state or semantic["exploratory_classification"]!=emitted:raise ValueError("state/classification mismatch")
    provenance=result["provenance"]
    if set(provenance)!={
        "execution_label","execution_instance_id","runtime","arm_records",
        "elapsed_seconds","peak_rss_bytes","output_bytes_before_result",
    }:raise ValueError("provenance shape mismatch")
    if provenance["execution_label"]!=label or provenance["execution_instance_id"]!=manifest["execution_instance_id"] or provenance["runtime"]!=runtime:raise ValueError("provenance binding mismatch")
    if (
        type(provenance["elapsed_seconds"]) is not float
        or type(provenance["peak_rss_bytes"]) is not int
        or type(provenance["output_bytes_before_result"]) is not int
        or not math.isfinite(provenance["elapsed_seconds"])
        or not (0<=provenance["elapsed_seconds"]<3600)
        or not (0<provenance["peak_rss_bytes"]<=2147483648)
        or provenance["output_bytes_before_result"]!=(root/"run_manifest.json").stat().st_size
    ):raise ValueError("provenance resource cap/type mismatch")
    records=provenance["arm_records"]
    if [r["arm_id"] for r in records]!=list(PRIMARY+AUDITS):raise ValueError("arm provenance order mismatch")
    for record in records:
        if (
            set(record)!={"arm_id","elapsed_seconds","peak_rss_bytes"}
            or type(record["elapsed_seconds"]) is not float
            or type(record["peak_rss_bytes"]) is not int
            or not math.isfinite(record["elapsed_seconds"])
            or not (0<=record["elapsed_seconds"]<600)
            or not (0<record["peak_rss_bytes"]<=2147483648)
        ):raise ValueError("arm provenance mismatch")
    if (
        provenance["elapsed_seconds"]<sum(row["elapsed_seconds"] for row in records)
        or provenance["peak_rss_bytes"]<max(row["peak_rss_bytes"] for row in records)
    ):raise ValueError("aggregate provenance is inconsistent with arm records")
    return {"manifest":manifest,"result":result,"semantic":semantic,"semantic_sha256":result["semantic_sha256"]}


def tree_sha(root: Path) -> str:
    digest=hashlib.sha256(b"jx-xp1-output-tree/v1\0")
    for path in sorted(root.rglob("*"),key=lambda p:p.relative_to(root).as_posix()):
        if path.is_symlink() or not path.is_file():raise ValueError("output tree contains nonregular entry")
        relative=path.relative_to(root).as_posix().encode();payload=path.read_bytes()
        digest.update(len(relative).to_bytes(4,"big"));digest.update(relative);digest.update(len(payload).to_bytes(8,"big"));digest.update(payload)
    return digest.hexdigest()


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--contract",type=Path,required=True);parser.add_argument("--seed-manifest",type=Path,required=True);parser.add_argument("--registration",type=Path,required=True);parser.add_argument("--output-a",type=Path,required=True);parser.add_argument("--output-b",type=Path);parser.add_argument("--receipt",type=Path,required=True);parser.add_argument("--verify-a-only",action="store_true");args=parser.parse_args()
    raw_paths=[args.contract,args.seed_manifest,args.registration,args.output_a,args.receipt]+([] if args.output_b is None else [args.output_b])
    if any(path.is_symlink() for path in raw_paths):raise ValueError("verifier inputs, outputs, and receipt must not be symlinks")
    root=Path(__file__).resolve().parent
    output_a=args.output_a.resolve();output_b=None if args.output_b is None else args.output_b.resolve();receipt_path=args.receipt.resolve()
    if args.verify_a_only:
        if output_b is not None:raise ValueError("A-only verification does not accept --output-b")
    elif output_b is None:
        raise ValueError("full replay verification requires --output-b")
    if receipt_path.is_relative_to(root) or receipt_path.is_relative_to(output_a) or (output_b is not None and receipt_path.is_relative_to(output_b)):
        raise ValueError("verification receipt must be outside package and output trees")
    if output_b is not None and (output_a==output_b or output_a.is_relative_to(output_b) or output_b.is_relative_to(output_a)):
        raise ValueError("A and B output roots must be disjoint without ancestry")
    if args.contract.resolve()!=root/"contract_v1.json" or args.seed_manifest.resolve()!=root/"seed_manifest_v1.json" or digest_file(args.contract.resolve())!=CONTRACT_SHA or digest_file(args.seed_manifest.resolve())!=SEED_SHA:raise ValueError("canonical frozen inputs mismatch")
    contract=read_json(args.contract.resolve());seed_manifest=read_json(args.seed_manifest.resolve())
    protected=protected_tree_roots(contract,root)
    for candidate in [output_a,receipt_path]+([] if output_b is None else [output_b]):
        if any(trees_overlap(candidate,protected_root) for protected_root in protected):
            raise ValueError("output or receipt overlaps a package or bound-context tree")
    if digest_bytes(canonical(contract["design_core"]))!=DESIGN_SHA:raise ValueError("design core mismatch")
    for binding in contract["excluded_context_bindings"]:
        candidate=root/binding["path"];path=candidate.resolve()
        if candidate.is_symlink() or not path.is_file() or path.stat().st_nlink!=1 or digest_file(path)!=binding["sha256"]:raise ValueError(f"excluded binding mismatch: {binding['id']}")
    registration,registration_sha=validate_registration(root,args.registration.resolve(),contract)
    runtime=actual_runtime()
    if {k:runtime[k] for k in contract["runtime_lock"]}!=contract["runtime_lock"]:raise ValueError("actual runtime mismatch")
    tracers,tracer_sha=independently_realize_tracers(contract,seed_manifest)
    initial=independent_initial_bindings(contract,tracers)
    a=validate_output(output_a,contract,registration,registration_sha,"A",initial,runtime)
    if set(contract["result_policy"]["verification_recomputation_keys"])!=INDEPENDENT_RECOMPUTATION_KEYS:
        raise ValueError("verification recomputation key lock mismatch")
    recomputation={key:True for key in INDEPENDENT_RECOMPUTATION_KEYS}
    if args.verify_a_only:
        receipt={
            "schema":"jx-xp1-a-verification-receipt/v1","experiment_id":EXPERIMENT,
            "state":"XP1_A_OUTPUT_VERIFIED_FOR_B","contract_sha256":CONTRACT_SHA,
            "seed_manifest_sha256":SEED_SHA,"registration_sha256":registration_sha,
            "verifier_sha256":registration["locked_files"]["verify_replay.py"],
            "execution_a":{"semantic_sha256":a["semantic_sha256"],"result_sha256":digest_file(output_a/"result_v1.json"),"output_tree_sha256":tree_sha(output_a),"execution_instance_id":a["manifest"]["execution_instance_id"]},
            "independent_recomputation":recomputation,
            "mandatory_nonclaim":contract["mandatory_nonclaim"],
        }
        write_once(receipt_path,receipt);print(json.dumps(receipt,indent=2,sort_keys=True));return 0
    assert output_b is not None
    b=validate_output(output_b,contract,registration,registration_sha,"B",initial,runtime)
    if a["manifest"]["execution_instance_id"]==b["manifest"]["execution_instance_id"]:raise ValueError("execution IDs are not distinct")
    exact=canonical(a["semantic"])==canonical(b["semantic"]) and a["semantic_sha256"]==b["semantic_sha256"]
    state="XP1_SEMANTIC_REPLAY_EXACT" if exact else "NONDETERMINISTIC_REPLAY"
    receipt={
        "schema":"jx-xp1-replay-receipt/v1","experiment_id":EXPERIMENT,"state":state,
        "contract_sha256":CONTRACT_SHA,"seed_manifest_sha256":SEED_SHA,"registration_sha256":registration_sha,"tracer_rows_sha256":tracer_sha,"verifier_sha256":registration["locked_files"]["verify_replay.py"],
        "execution_a":{"semantic_sha256":a["semantic_sha256"],"result_sha256":digest_file(output_a/"result_v1.json"),"output_tree_sha256":tree_sha(output_a),"execution_instance_id":a["manifest"]["execution_instance_id"]},
        "execution_b":{"semantic_sha256":b["semantic_sha256"],"result_sha256":digest_file(output_b/"result_v1.json"),"output_tree_sha256":tree_sha(output_b),"execution_instance_id":b["manifest"]["execution_instance_id"]},
        "independent_recomputation":recomputation,
        "mandatory_nonclaim":contract["mandatory_nonclaim"],
    }
    write_once(receipt_path,receipt);print(json.dumps(receipt,indent=2,sort_keys=True));return 0 if exact else 2


if __name__=="__main__":
    raise SystemExit(main())
