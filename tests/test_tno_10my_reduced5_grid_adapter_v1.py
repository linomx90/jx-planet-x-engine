from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from dataclasses import replace
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import types
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "benchmarks" / "jx_tno_10my_reduced5_grid_adapter_v1.py"


def _source_only_load(name: str) -> types.ModuleType:
    payload = SOURCE.read_bytes()
    module = types.ModuleType(name)
    module.__file__ = str(SOURCE)
    module.__package__ = ""
    module.__spec__ = None
    sys.modules[name] = module
    try:
        exec(
            compile(payload, str(SOURCE), "exec", dont_inherit=True, optimize=sys.flags.optimize),
            module.__dict__,
        )
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


GRID = _source_only_load(f"_test_tno_10my_grid_adapter_v1_{sys.flags.optimize}")


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _small_geometry() -> object:
    return GRID.GridGeometry(
        batch_count=1,
        total_steps=64,
        shard_steps=16,
        b22_interval_steps=8,
        short_interval_steps=4,
        short_final_step=32,
    )


def _state_payload(step: int, geometry: object) -> bytes:
    rows = tuple(
        tuple(float(step + body * 10 + component) / 17.0 for component in range(6))
        for body in range(geometry.body_count)
    )
    return GRID.state_record_v1(rows, geometry)


def _fake_derive(batch_order: int, payload: bytes) -> tuple[bytes, bytes]:
    digest = hashlib.sha256(b"test.grid.node\0" + bytes([batch_order]) + payload).digest()
    first = struct.unpack_from("<d", payload, 0)[0]
    elements = struct.pack("<60d", *(first + index / 64.0 for index in range(60)))
    return digest, elements


def _fake_prefix_crossbind(
    batch_order: int,
    checkpoints: tuple[tuple[int, bytes], ...],
) -> dict[str, object]:
    record: dict[str, object] = {
        "batch_order": batch_order,
        "labels": [item[0] for item in checkpoints],
        "state_sha256": [hashlib.sha256(item[1]).hexdigest() for item in checkpoints],
    }
    record["content_sha256"] = hashlib.sha256(
        b"test.grid.prefix\0" + GRID._canonical_json(record)
    ).hexdigest()
    return record


def _write_small_chain(base: Path) -> tuple[list[Path], object]:
    geometry = _small_geometry()
    prefix_receipt = _fake_prefix_crossbind(
        0,
        tuple(
            (index, _state_payload(index * geometry.short_interval_steps, geometry))
            for index in range(3)
        ),
    )
    roots: list[Path] = []
    predecessor = GRID.ZERO_SHA256
    previous_final = _digest("genesis-continuation")
    for shard_index in range(geometry.shard_count):
        root = base / f"shard-{shard_index:04d}"
        checkpoints = tuple(
            (step, _state_payload(step, geometry))
            for step in geometry.expected_steps(shard_index)
        )
        final = _digest(f"continuation-{shard_index}")
        manifest = GRID.write_shard_v1(
            root,
            batch_order=0,
            shard_index=shard_index,
            checkpoints=checkpoints,
            predecessor_manifest_content_sha256=predecessor,
            start_continuation_sha256=previous_final,
            final_continuation_sha256=final,
            source_chunk_receipt_sha256=_digest(f"source-{shard_index}"),
            route_prefix_sha256=_digest(f"route-{shard_index}"),
            runtime_projection_sha256=_digest("runtime"),
            first_1000_year_prefix_crossbinding_sha256=prefix_receipt["content_sha256"],
            geometry=geometry,
            derive=_fake_derive,
            test_only_geometry=True,
        )
        roots.append(root)
        predecessor = manifest["content_sha256"]
        previous_final = final
    return roots, geometry


def _rewrite_final_metadata(output: Path, manifest: dict[str, object]) -> None:
    manifest["content_sha256"] = GRID._domain_hash(GRID.FINAL_MANIFEST_DOMAIN, manifest)
    (output / "MANIFEST.json").write_bytes(GRID._canonical_json(manifest))
    result = json.loads((output / "RESULT.json").read_bytes())
    result["manifest_content_sha256"] = manifest["content_sha256"]
    result["content_sha256"] = GRID._domain_hash(GRID.FINAL_RESULT_DOMAIN, result)
    (output / "RESULT.json").write_bytes(GRID._canonical_json(result))


class Tno10MyrReduced5GridAdapterV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = GRID.build_protocol()

    def test_source_only_import_has_no_dynamics_or_gpu_side_effect(self) -> None:
        before = set(sys.modules)
        module = _source_only_load(f"_test_grid_import_{sys.flags.optimize}_{id(self)}")
        added = set(sys.modules) - before
        self.assertFalse(any(name == "cupy" or name.startswith("cupy.") for name in added))
        self.assertFalse(any(name == "numpy" or name.startswith("numpy.") for name in added))
        self.assertFalse(any(name == "rebound" or name.startswith("rebound.") for name in added))
        self.assertFalse(hasattr(module, "execute"))
        self.assertFalse(hasattr(module, "run"))
        self.assertEqual(module.build_protocol()["implementation"]["cli_surface"], "BUILD_PREPARE_VALIDATE_ONLY")

    def test_production_geometry_and_exact_storage_arithmetic(self) -> None:
        geometry = GRID.PRODUCTION_GEOMETRY
        self.assertEqual(geometry.shard_count, 1_000)
        self.assertEqual(geometry.b22_node_count, 50_001)
        self.assertEqual(geometry.short_node_count, 10_001)
        self.assertEqual(geometry.union_node_count, 57_501)
        self.assertEqual(geometry.state_record_bytes, 672)
        self.assertEqual(geometry.element_record_bytes, 480)
        self.assertEqual(
            geometry.output_sizes(),
            {
                "B22_STATES.f64le": 403_208_064,
                "B22_ELEMENTS.f64le": 288_005_760,
                "B22_NODE_SHA256.raw": 19_200_384,
                "ML_SHORT_STATES.f64le": 80_648_064,
            },
        )
        self.assertEqual(sum(geometry.output_sizes().values()), 791_062_272)
        self.assertEqual(len(geometry.expected_steps(0)), 201)
        self.assertEqual(len(geometry.expected_steps(49)), 200)
        self.assertEqual(len(geometry.expected_steps(50)), 50)
        self.assertEqual(geometry.expected_steps(0)[0], 0)
        self.assertEqual(geometry.expected_steps(999)[-1], 640_000_000)
        self.assertEqual(
            sum(len(geometry.expected_steps(index)) for index in range(geometry.shard_count)),
            57_501,
        )

    def test_exact_offset_formulas_match_frozen_layout(self) -> None:
        geometry = GRID.PRODUCTION_GEOMETRY
        batch = 11
        node = 50_000
        body = 13
        component = 5
        scalar_index = (((batch * geometry.b22_node_count + node) * 14 + body) * 6 + component)
        self.assertEqual(scalar_index * 8 + 8, geometry.output_sizes()["B22_STATES.f64le"])
        entity = 9
        field = 5
        element_index = (((batch * 10 + entity) * 6 + field) * geometry.b22_node_count + node)
        self.assertEqual(element_index * 8 + 8, geometry.output_sizes()["B22_ELEMENTS.f64le"])
        self.assertEqual((batch * geometry.b22_node_count + node) * 32 + 32, geometry.output_sizes()["B22_NODE_SHA256.raw"])

    def test_state_codec_is_row_interleaved_and_type_strict(self) -> None:
        geometry = _small_geometry()
        payload = _state_payload(3, geometry)
        rows = GRID.decode_state_record_v1(payload, geometry)
        self.assertEqual(len(payload), 672)
        self.assertEqual(rows[0][0], 3.0 / 17.0)
        self.assertEqual(rows[1][0], 13.0 / 17.0)
        expected_first_row = struct.pack("<6d", *rows[0])
        self.assertEqual(payload[:48], expected_first_row)
        with self.assertRaises(GRID.GridAdapterError):
            GRID.state_record_v1(tuple(tuple(0 for _ in range(6)) for _ in range(14)), geometry)
        bad = [list(row) for row in rows]
        bad[0][0] = math.inf
        with self.assertRaises(GRID.GridAdapterError):
            GRID.state_record_v1(tuple(tuple(row) for row in bad), geometry)

    def test_small_shards_validate_and_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            roots, geometry = _write_small_chain(Path(directory))
            predecessor = GRID.ZERO_SHA256
            previous_final = None
            for index, root in enumerate(roots):
                manifest = GRID.validate_shard_v1(
                    root,
                    geometry=geometry,
                    derive=_fake_derive,
                    test_only_geometry=True,
                )
                self.assertEqual(manifest["predecessor_manifest_content_sha256"], predecessor)
                if previous_final is not None:
                    self.assertEqual(manifest["start_continuation_sha256"], previous_final)
                predecessor = manifest["content_sha256"]
                previous_final = manifest["final_continuation_sha256"]

    def test_shard_stream_gap_extra_and_bool_identity_fail_closed(self) -> None:
        geometry = _small_geometry()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            expected = geometry.expected_steps(0)
            cases = (
                tuple((step, _state_payload(step, geometry)) for step in expected[:-1]),
                tuple((step, _state_payload(step, geometry)) for step in (*expected, 20)),
                tuple((False if index == 0 else step, _state_payload(step, geometry)) for index, step in enumerate(expected)),
            )
            for case_index, checkpoints in enumerate(cases):
                with self.assertRaises((GRID.GridAdapterError, ValueError)):
                    GRID.write_shard_v1(
                        base / f"bad-{case_index}",
                        batch_order=0,
                        shard_index=0,
                        checkpoints=checkpoints,
                        predecessor_manifest_content_sha256=GRID.ZERO_SHA256,
                        start_continuation_sha256=_digest("start"),
                        final_continuation_sha256=_digest("final"),
                        source_chunk_receipt_sha256=_digest("source"),
                        route_prefix_sha256=_digest("route"),
                        runtime_projection_sha256=_digest("runtime"),
                        first_1000_year_prefix_crossbinding_sha256=_digest("first-1k"),
                        geometry=geometry,
                        derive=_fake_derive,
                        test_only_geometry=True,
                    )
                self.assertFalse((base / f"bad-{case_index}" / "MANIFEST.json").exists())

    def test_fragment_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            roots, geometry = _write_small_chain(Path(directory))
            target = roots[1] / GRID.FRAGMENT_NAMES[0]
            with target.open("r+b") as handle:
                original = handle.read(1)
                handle.seek(0)
                handle.write(bytes([original[0] ^ 1]))
                handle.flush()
                os.fsync(handle.fileno())
            with self.assertRaises(GRID.GridAdapterError):
                GRID.validate_shard_v1(
                    roots[1],
                    geometry=geometry,
                    derive=_fake_derive,
                    test_only_geometry=True,
                )

    def test_rehashed_bool_and_extra_manifest_fields_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            roots, geometry = _write_small_chain(Path(directory))
            for case_index, mutation in enumerate(("bool", "claim_bool_alias", "extra")):
                source = roots[0]
                target = Path(directory) / f"mutated-{case_index}"
                target.mkdir()
                for name in GRID.FRAGMENT_NAMES:
                    (target / name).write_bytes((source / name).read_bytes())
                manifest = json.loads((source / "MANIFEST.json").read_bytes())
                if mutation == "bool":
                    manifest["shard_index"] = False
                elif mutation == "claim_bool_alias":
                    manifest["claims"]["science_result"] = 0
                else:
                    manifest["unexpected"] = "field"
                manifest["content_sha256"] = GRID._domain_hash(GRID.SHARD_DOMAIN, manifest)
                (target / "MANIFEST.json").write_bytes(GRID._canonical_json(manifest))
                with self.assertRaises(GRID.GridAdapterError):
                    GRID.validate_shard_v1(
                        target,
                        geometry=geometry,
                        derive=_fake_derive,
                        test_only_geometry=True,
                    )

    def test_complete_chain_finalizes_transposes_and_replays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            roots, geometry = _write_small_chain(base)
            output = base / "final"
            result = GRID.finalize_shards_v1(
                (tuple(roots),),
                output,
                geometry=geometry,
                derive=_fake_derive,
                prefix_crossbind=_fake_prefix_crossbind,
                test_only_geometry=True,
            )
            self.assertEqual(tuple(sorted(item.name for item in output.iterdir())), tuple(sorted(GRID.FINAL_ROSTER)))
            self.assertEqual(result["status"], "GRID_ARTIFACTS_ASSEMBLED_AND_REPLAYED_BY_MECHANICAL_FINALIZER")
            validated = GRID.validate_finalized_v1(
                output,
                geometry=geometry,
                derive=_fake_derive,
                prefix_crossbind=_fake_prefix_crossbind,
                test_only_geometry=True,
            )
            self.assertEqual(validated["content_sha256"], result["content_sha256"])
            self.assertEqual((output / "B22_STATES.f64le").stat().st_size, geometry.output_sizes()["B22_STATES.f64le"])
            self.assertEqual((output / "B22_ELEMENTS.f64le").stat().st_size, geometry.output_sizes()["B22_ELEMENTS.f64le"])

    def test_final_validator_rejects_rehashed_integer_claim_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            roots, geometry = _write_small_chain(base)
            output = base / "final"
            GRID.finalize_shards_v1(
                (tuple(roots),),
                output,
                geometry=geometry,
                derive=_fake_derive,
                prefix_crossbind=_fake_prefix_crossbind,
                test_only_geometry=True,
            )
            result = json.loads((output / "RESULT.json").read_bytes())
            result["claims"]["science_result"] = 0
            result["content_sha256"] = GRID._domain_hash(GRID.FINAL_RESULT_DOMAIN, result)
            (output / "RESULT.json").write_bytes(GRID._canonical_json(result))
            with self.assertRaises(GRID.GridAdapterError):
                GRID.validate_finalized_v1(
                    output,
                    geometry=geometry,
                    derive=_fake_derive,
                    prefix_crossbind=_fake_prefix_crossbind,
                    test_only_geometry=True,
                )

    def test_finalizer_rejects_chain_reorder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            roots, geometry = _write_small_chain(base)
            roots[1], roots[2] = roots[2], roots[1]
            with self.assertRaises(GRID.GridAdapterError):
                GRID.finalize_shards_v1(
                    (tuple(roots),),
                    base / "bad-final",
                    geometry=geometry,
                    derive=_fake_derive,
                    prefix_crossbind=_fake_prefix_crossbind,
                    test_only_geometry=True,
                )

    def test_final_validator_rejects_rehashed_forged_chain_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            roots, geometry = _write_small_chain(base)
            output = base / "final"
            GRID.finalize_shards_v1(
                (tuple(roots),),
                output,
                geometry=geometry,
                derive=_fake_derive,
                prefix_crossbind=_fake_prefix_crossbind,
                test_only_geometry=True,
            )
            manifest = json.loads((output / "MANIFEST.json").read_bytes())
            manifest["shard_manifests_by_batch"][0][1]["content_sha256"] = "11" * 32
            _rewrite_final_metadata(output, manifest)
            with self.assertRaises(GRID.GridAdapterError):
                GRID.validate_finalized_v1(
                    output,
                    geometry=geometry,
                    derive=_fake_derive,
                    prefix_crossbind=_fake_prefix_crossbind,
                    test_only_geometry=True,
                )

    def test_final_validator_rejects_coordinated_nonfinite_short_only_node(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            roots, geometry = _write_small_chain(base)
            output = base / "final"
            GRID.finalize_shards_v1(
                (tuple(roots),),
                output,
                geometry=geometry,
                derive=_fake_derive,
                prefix_crossbind=_fake_prefix_crossbind,
                test_only_geometry=True,
            )
            short_path = output / "ML_SHORT_STATES.f64le"
            with short_path.open("r+b") as handle:
                handle.seek(geometry.state_record_bytes)  # step 4: short-only
                handle.write(struct.pack("<d", math.nan))
                handle.flush()
                os.fsync(handle.fileno())
            manifest = json.loads((output / "MANIFEST.json").read_bytes())
            short_payload = short_path.read_bytes()
            for row in manifest["files"]:
                if row["name"] == "ML_SHORT_STATES.f64le":
                    row["sha256"] = hashlib.sha256(short_payload).hexdigest()
            retained = manifest["shard_manifests_by_batch"][0]
            first_short_count = retained[0]["short_checkpoint_count"]
            first_short_payload = short_payload[: first_short_count * geometry.state_record_bytes]
            retained[0]["fragments"][3]["sha256"] = hashlib.sha256(first_short_payload).hexdigest()
            predecessor = GRID.ZERO_SHA256
            for shard in retained:
                shard["predecessor_manifest_content_sha256"] = predecessor
                shard["content_sha256"] = GRID._domain_hash(GRID.SHARD_DOMAIN, shard)
                predecessor = shard["content_sha256"]
            range_hashes = {
                name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                for name in GRID.FINAL_BINARY_NAMES
            }
            manifest["batch_stream_roots"][0] = GRID._batch_stream_root_record_v1(
                0, retained, range_hashes
            )
            _rewrite_final_metadata(output, manifest)
            with self.assertRaises(GRID.GridAdapterError):
                GRID.validate_finalized_v1(
                    output,
                    geometry=geometry,
                    derive=_fake_derive,
                    prefix_crossbind=_fake_prefix_crossbind,
                    test_only_geometry=True,
                )

    def test_final_validator_rederives_prefix_against_coordinated_forgery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            roots, geometry = _write_small_chain(base)
            output = base / "final"
            GRID.finalize_shards_v1(
                (tuple(roots),),
                output,
                geometry=geometry,
                derive=_fake_derive,
                prefix_crossbind=_fake_prefix_crossbind,
                test_only_geometry=True,
            )
            manifest = json.loads((output / "MANIFEST.json").read_bytes())
            retained = manifest["shard_manifests_by_batch"][0]
            predecessor = GRID.ZERO_SHA256
            for shard in retained:
                shard["first_1000_year_prefix_crossbinding_sha256"] = "22" * 32
                shard["predecessor_manifest_content_sha256"] = predecessor
                shard["content_sha256"] = GRID._domain_hash(GRID.SHARD_DOMAIN, shard)
                predecessor = shard["content_sha256"]
            range_hashes = {
                name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                for name in GRID.FINAL_BINARY_NAMES
            }
            manifest["batch_stream_roots"][0] = GRID._batch_stream_root_record_v1(
                0, retained, range_hashes
            )
            _rewrite_final_metadata(output, manifest)
            with self.assertRaises(GRID.GridAdapterError):
                GRID.validate_finalized_v1(
                    output,
                    geometry=geometry,
                    derive=_fake_derive,
                    prefix_crossbind=_fake_prefix_crossbind,
                    test_only_geometry=True,
                )

    def test_real_classifier_kernel_replays_one_exact_initial_node(self) -> None:
        with GRID.FrozenClassifierKernelV1() as kernel:
            bridge, custody, _bindings = kernel.stage1._verify_input_documents()
            batches = kernel.stage1._batch_states(bridge, custody, "DE441")
            reduced = kernel.stage1.collapse_reduced_model(batches[0])
            payload = GRID.state_record_from_objects_v1(reduced.massive, reduced.tracers)
            digest, elements = kernel.derive(0, payload)
            direct = kernel.classifier.node_state_sha256_v2(
                reduced.massive,
                reduced.tracers,
                0,
                reduced.epoch_jd_tdb,
            )
            self.assertEqual(digest.hex(), direct)
            self.assertEqual(len(elements), 480)
            self.assertTrue(all(math.isfinite(value) for value in struct.unpack("<60d", elements)))

    def test_primary_relative_endpoint_uses_exact_stage1_barycenter_boundary(self) -> None:
        with GRID.FrozenClassifierKernelV1() as kernel:
            bridge, custody, _bindings = kernel.stage1._verify_input_documents()
            batch = kernel.stage1.collapse_reduced_model(
                kernel.stage1._batch_states(bridge, custody, "DE441")[0]
            )
            primary_position = batch.massive[0].position_AU
            primary_velocity = batch.massive[0].velocity_AU_per_julian_year

            def subtract(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
                return tuple(left[index] - right[index] for index in range(3))

            source_positions = tuple(subtract(item.position_AU, primary_position) for item in batch.massive)
            source_velocities = tuple(subtract(item.velocity_AU_per_julian_year, primary_velocity) for item in batch.massive)
            tracer_positions = tuple(subtract(item.position_AU, primary_position) for item in batch.tracers)
            tracer_velocities = tuple(subtract(item.velocity_AU_per_julian_year, primary_velocity) for item in batch.tracers)
            payload = GRID.checkpoint_from_primary_relative_v1(
                kernel,
                batch_order=0,
                source_relative_positions_AU=source_positions,
                source_relative_velocities_AU_per_julian_year=source_velocities,
                tracer_relative_positions_AU=tracer_positions,
                tracer_relative_velocities_AU_per_julian_year=tracer_velocities,
            )
            rows = GRID.decode_state_record_v1(payload)
            center = tuple(
                math.fsum(float.fromhex(GRID.REDUCED_GM_HEX[index]) * rows[index][axis] for index in range(5))
                / math.fsum(float.fromhex(value) for value in GRID.REDUCED_GM_HEX)
                for axis in range(3)
            )
            self.assertLessEqual(max(abs(value) for value in center), 1.0e-12)
            digest, elements = kernel.derive(0, payload)
            self.assertEqual(len(digest), 32)
            self.assertEqual(len(elements), 480)
            bad_positions = list(source_positions)
            bad_positions[0] = (False, 0.0, 0.0)
            with self.assertRaises(GRID.GridAdapterError):
                GRID.checkpoint_from_primary_relative_v1(
                    kernel,
                    batch_order=0,
                    source_relative_positions_AU=tuple(bad_positions),
                    source_relative_velocities_AU_per_julian_year=source_velocities,
                    tracer_relative_positions_AU=tracer_positions,
                    tracer_relative_velocities_AU_per_julian_year=tracer_velocities,
                )

    def test_adapter_contract_freezes_wh_lifecycle_not_v8_kdk(self) -> None:
        adapter = GRID.adapter_contract_v1()
        self.assertEqual(adapter["massive_source_integrator"], "WISDOM_HOLMAN_SOURCE_TILE_H64")
        self.assertEqual(adapter["quiet_tracer_map"], "CUDA_RESTRICTED_WH_FIXED_STEP")
        self.assertEqual(adapter["encounter_lifecycle"], "RESTRICTED_WH_LIFECYCLE_DOP853_ROUTE")
        self.assertEqual(adapter["time_scale"], "TDB")
        self.assertEqual(adapter["source_step_julian_years_hex"], "0x1.0000000000000p-6")
        self.assertEqual(adapter["source_build_phases"]["phase_0_prefix_gate"]["source_steps"], 64_000)
        self.assertEqual(adapter["source_build_phases"]["phase_1_shard_zero_complement"]["source_steps"], 576_000)
        self.assertEqual(adapter["source_build_phases"]["phase_1_regular_shards"]["source_steps_each"], 640_000)
        self.assertEqual(adapter["source_build_phases"]["source_build_count_total"], 12_012)
        self.assertEqual(adapter["source_build_phases"]["published_shard_count_total"], 12_000)
        self.assertEqual(adapter["segment_schedule"]["segments_total"], 690_000)
        self.assertEqual(adapter["route_policy"]["terminal_source_radii_AU_hex"], ["0x0.0p+0"] * 5)
        self.assertEqual(
            adapter["route_policy"]["certified_relative_acceleration_bounds_AU_per_year2_hex"],
            [GRID.REQUIRED_RELATIVE_ACCELERATION_BOUND_HEX] * 5,
        )
        self.assertEqual(
            adapter["central_source_identity"]["stage1_and_classifier_runtime_body_id"],
            "SUN_PLUS_MERCURY_VENUS_EARTH_MOON_MARS",
        )
        self.assertIs(adapter["execution_surface_present"], False)
        self.assertIs(adapter["scientifically_qualified"], False)

    def test_exact_adapter_specs_construct_without_backend_or_dynamics(self) -> None:
        before_cupy = {name for name in sys.modules if name == "cupy" or name.startswith("cupy.")}
        values = GRID.construct_adapter_specs_v1(
            map_step_count=3_200,
            source_step_count=64_000,
        )
        after_cupy = {name for name in sys.modules if name == "cupy" or name.startswith("cupy.")}
        self.assertEqual(before_cupy, after_cupy)
        self.assertEqual(values["time_scale"], "TDB")
        self.assertEqual(values["source"].source_step_years.hex(), GRID.H64_STEP_HEX)
        self.assertEqual(values["source"].source_step_count, 64_000)
        self.assertEqual(values["source"].minimum_encounter_pair_separation_AU, 1.0)
        self.assertEqual(values["source"].minimum_jacobi_periapse_AU, 0.1)
        self.assertEqual(values["fixed_step"].map_step_count, 3_200)
        self.assertEqual(values["fixed_step"].source_steps_per_map_step, 1)
        self.assertEqual(values["fixed_step"].threads_per_block, 128)
        self.assertEqual(values["fixed_step"].maximum_primary_dynamical_phase, 0.05)
        self.assertEqual(values["event_locator"].maximum_accepted_steps, 100_000)
        self.assertEqual(values["route"].terminal_source_radii_AU, (0.0,) * 5)
        self.assertEqual(values["route_worker_count"], 1)
        self.assertEqual(values["source_builder"].__name__, "build_wisdom_holman_source_tile")
        with self.assertRaises(GRID.GridAdapterError):
            GRID.construct_adapter_specs_v1(map_step_count=1, source_step_count=64_000)
        with self.assertRaises(GRID.GridAdapterError):
            GRID.construct_adapter_specs_v1(map_step_count=3_200, source_step_count=640_001)

    def test_frozen_source_slice_is_exact_and_contiguous(self) -> None:
        import numpy as np
        from jxplanetx.cuda_restricted_wh import COORDINATE_SYSTEM, FrozenSourcePathTile
        from jxplanetx.engine.contracts import Provenance

        positions = np.zeros((64_001, 5, 3), dtype=np.float64)
        positions[:, 1:, 0] = np.asarray((5.0, 10.0, 20.0, 30.0))
        positions[1:, 1:, 1] = 0.25
        velocities = np.zeros_like(positions)
        schedule = _digest("schedule")
        full = FrozenSourcePathTile(
            tile_id="synthetic-full",
            epoch_origin_years=float.fromhex(
                GRID.NATIVE_EPOCH_ORIGINS_JULIAN_YEARS_HEX[0]
            ),
            global_source_step=0,
            source_step_years=float.fromhex(GRID.H64_STEP_HEX),
            time_scale="TDB",
            frame="ICRF_J2000",
            axes="J2000_EQUATORIAL",
            coordinate_system=COORDINATE_SYSTEM,
            length_unit="AU",
            time_unit="year",
            unit_system_id="AU-YEAR",
            source_ids=GRID.REDUCED_BODY_IDS,
            primary_source_id=GRID.REDUCED_BODY_IDS[0],
            source_schedule_sha256=schedule,
            source_relative_positions_AU=positions,
            source_relative_velocities_AU_per_year=velocities,
            source_gm_AU3_per_year2=np.asarray(
                [float.fromhex(value) for value in GRID.REDUCED_GM_HEX], dtype=np.float64
            ),
            source_encounter_radii_AU=np.ones(5, dtype=np.float64),
            certified_relative_acceleration_bounds_AU_per_year2=np.full(
                5, float.fromhex(GRID.REQUIRED_RELATIVE_ACCELERATION_BOUND_HEX), dtype=np.float64
            ),
            provenance=Provenance("test", "test", "v1", schedule),
        )
        sliced = GRID.slice_frozen_source_tile_v1(
            full,
            local_start_step=3_200,
            map_step_count=3_200,
            tile_id="synthetic-slice",
        )
        self.assertEqual(sliced.global_source_step, 3_200)
        self.assertEqual(sliced.node_count, 3_201)
        self.assertTrue(np.array_equal(sliced.source_relative_positions_AU, positions[3_200:6_401]))
        self.assertEqual(sliced.certified_relative_acceleration_bounds_AU_per_year2[0].hex(), GRID.REQUIRED_RELATIVE_ACCELERATION_BOUND_HEX)
        with self.assertRaises(GRID.GridAdapterError):
            GRID.slice_frozen_source_tile_v1(
                full,
                local_start_step=1,
                map_step_count=3_200,
                tile_id="misaligned",
            )
        with self.assertRaises(GRID.GridAdapterError):
            GRID.slice_frozen_source_tile_v1(
                full,
                local_start_step=0,
                map_step_count=12_800,
                tile_id="wrong-segment-length",
            )
        with self.assertRaises(GRID.GridAdapterError):
            GRID.slice_frozen_source_tile_v1(
                replace(full, time_scale="WRONG", content_sha256=None),
                local_start_step=0,
                map_step_count=3_200,
                tile_id="wrong-model",
            )

    def test_first_1000_year_prefix_gate_replays_retained_de441_h64(self) -> None:
        result = GRID._strict_json(GRID.H64_CLOSURE_RESULT_PATH.read_bytes())
        locator = GRID._h64_closure_lane_record(0, result)
        checkpoints = []
        with GRID.H64_CLOSURE_ARRAYS_PATH.open("rb") as handle:
            for year in GRID.H64_PREFIX_YEARS:
                handle.seek(locator["byte_offset"] + year * GRID.H64_CLOSURE_RECORD_BYTES)
                checkpoints.append(
                    (year, GRID._legacy_h64_record_to_row_interleaved_v1(handle.read(672)))
                )
        receipt = GRID.validate_first_1000_year_prefix_v1(0, tuple(checkpoints))
        self.assertIs(receipt["passed"], True)
        self.assertEqual(receipt["years"], list(GRID.H64_PREFIX_YEARS))
        bad = list(checkpoints)
        altered = bytearray(bad[-1][1])
        altered[0] ^= 1
        bad[-1] = (1_000, bytes(altered))
        with self.assertRaises(GRID.GridAdapterError):
            GRID.validate_first_1000_year_prefix_v1(0, tuple(bad))

    def test_production_replay_rejects_substitute_deriver(self) -> None:
        with self.assertRaises(GRID.GridAdapterError):
            GRID._require_production_derive(_fake_derive)
        class CounterfeitKernel(GRID.FrozenClassifierKernelV1):
            pass
        counterfeit = object.__new__(CounterfeitKernel)
        counterfeit._closed = False
        with self.assertRaises(GRID.GridAdapterError):
            GRID._require_production_derive(counterfeit.derive)

    def test_protocol_claim_ceiling_and_source_retention_decision(self) -> None:
        protocol = self.protocol
        self.assertEqual(protocol["schema"], GRID.PROTOCOL_SCHEMA)
        self.assertEqual(protocol["production_geometry"]["union_node_count"], 57_501)
        self.assertEqual(protocol["fixed_output_total_bytes"], 791_062_272)
        self.assertEqual(protocol["full_source_node_retention"]["all_batches_bytes"], 1_843_200_002_880)
        self.assertIs(protocol["full_source_node_retention"]["retained"], False)
        claims = protocol["claims"]
        self.assertIs(claims["writer_and_validator_implemented"], True)
        self.assertIs(claims["production_scale_writer_qualified"], False)
        self.assertIs(claims["reduced5_encounter_adapter_executed"], False)
        self.assertIs(claims["ten_myr_execution_authorized"], False)
        self.assertIs(claims["science_result"], False)
        self.assertIs(protocol["outcomes_generated"], False)
        self.assertIs(protocol["execution_authorized"], False)
        self.assertEqual(
            protocol["implementation"]["runtime_trust_boundary"],
            "NO_HOSTILE_CODE_IN_THE_FRESH_WRITER_OR_VALIDATOR_PROCESS",
        )

    def test_protocol_lineage_and_bindings_are_exact(self) -> None:
        records = {row["role"]: row for row in self.protocol["lineage"]["bound_files"]}
        self.assertEqual(records["pilot_v2_source"]["sha256"], "33ad38cbc89a15a83cf180161544a721b8635affd0d4890a8a39d04389694332")
        self.assertEqual(records["classifier_v2_source"]["sha256"], "f60333e290c72fd769ac9655b5a6bc154f5464915c0988e1972f01b642f30c98")
        self.assertEqual(records["wh_source"]["sha256"], "69d30e621dfb28c60c5a07d63752d4b92d993ba5d74524d7c3b404cd19fe7b52")
        self.assertEqual(records["lifecycle"]["sha256"], "b0211778fcf51dbfef78049e74bafd05d5daee4db7dd18f743f4a5b5a14af346")
        self.assertEqual(self.protocol["lineage"]["pilot_v2_protocol_content_sha256"], GRID.PILOT_V2_PROTOCOL_CONTENT_SHA256)

    def test_prepare_validate_is_create_only_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            GRID.prepare(root)
            self.assertEqual([item.name for item in root.iterdir()], ["PROTOCOL.json"])
            self.assertEqual(GRID.validate_prepared(root), self.protocol)
            with self.assertRaises(GRID.GridAdapterError):
                GRID.prepare(root)
            with (root / "EXTRA").open("wb") as handle:
                handle.write(b"x")
            with self.assertRaises(GRID.GridAdapterError):
                GRID.validate_prepared(root)

    def test_cli_build_and_temp_prepare_validate_under_optimized_python(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            environment = dict(os.environ)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            build = subprocess.run(
                [sys.executable, "-OO", "-B", str(SOURCE), "build"],
                cwd=PROJECT_ROOT,
                env=environment,
                check=True,
                capture_output=True,
            )
            rebuilt = json.loads(build.stdout)
            self.assertEqual(rebuilt["content_sha256"], self.protocol["content_sha256"])
            subprocess.run(
                [sys.executable, "-OO", "-B", str(SOURCE), "prepare", "--root", str(root)],
                cwd=PROJECT_ROOT,
                env=environment,
                check=True,
                capture_output=True,
            )
            validated = subprocess.run(
                [sys.executable, "-OO", "-B", str(SOURCE), "validate", "--root", str(root)],
                cwd=PROJECT_ROOT,
                env=environment,
                check=True,
                capture_output=True,
            )
            self.assertEqual(json.loads(validated.stdout)["content_sha256"], self.protocol["content_sha256"])


if __name__ == "__main__":
    unittest.main()
