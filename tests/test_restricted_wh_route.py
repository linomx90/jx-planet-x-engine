from __future__ import annotations

from dataclasses import replace
import hashlib
import math
import unittest

import numpy as np

from jxplanetx._all_conic_kepler import drift_all_conic
from jxplanetx.cuda_restricted_wh import COORDINATE_SYSTEM, FrozenSourcePathTile
from jxplanetx.cuda_tracer_events import (
    DOP853EventLocatorSpec,
    EVENT_PRIMARY_BOUNDARY_OUTWARD,
    EVENT_SOURCE_RADIUS_CONTACT,
)
from jxplanetx.engine import Provenance
from jxplanetx.restricted_wh_route import (
    ROUTE_CLAIMS,
    RestrictedWHRouteContractError,
    RestrictedWHRouteIntegrationError,
    RestrictedWHRouteSpec,
    restricted_wh_route_policy_identity,
    route_restricted_wh_map_step,
)


SOLAR_GM = 4.0 * math.pi * math.pi
SCHEDULE_SHA256 = hashlib.sha256(b"restricted-wh-route-test-v1").hexdigest()


def _provenance() -> Provenance:
    return Provenance(
        source_id="jx-test.restricted-wh-route",
        citation="Synthetic restricted-WH route fixture",
        version="v1",
        sha256=hashlib.sha256(b"restricted-wh-route-provenance").hexdigest(),
    )


def _tile(
    *,
    step: float = 0.01,
    nodes: int = 2,
    primary_gm: float = SOLAR_GM,
    primary_radius: float = 0.01,
    secondary_positions: np.ndarray | None = None,
) -> FrozenSourcePathTile:
    source_count = 1 if secondary_positions is None else 2
    positions = np.zeros((nodes, source_count, 3), dtype=np.float64)
    velocities = np.zeros_like(positions)
    gm = np.asarray(
        [primary_gm]
        if source_count == 1
        else [primary_gm, 0.01 * primary_gm],
        dtype=np.float64,
    )
    if secondary_positions is not None:
        if secondary_positions.shape != (nodes, 3):
            raise AssertionError("secondary test path has the wrong shape")
        positions[:, 1, :] = secondary_positions
    return FrozenSourcePathTile(
        tile_id="restricted-route-fixture",
        epoch_origin_years=0.0,
        global_source_step=0,
        source_step_years=step,
        time_scale="TDB",
        frame="ICRF",
        axes="J2000_ECLIPTIC",
        coordinate_system=COORDINATE_SYSTEM,
        length_unit="AU",
        time_unit="year",
        unit_system_id="AU-YEAR",
        source_ids=("Sun",) if source_count == 1 else ("Sun", "Secondary"),
        primary_source_id="Sun",
        source_schedule_sha256=SCHEDULE_SHA256,
        source_relative_positions_AU=positions,
        source_relative_velocities_AU_per_year=velocities,
        source_gm_AU3_per_year2=gm,
        source_encounter_radii_AU=np.asarray(
            [primary_radius] if source_count == 1 else [primary_radius, 0.01],
            dtype=np.float64,
        ),
        certified_relative_acceleration_bounds_AU_per_year2=np.full(
            source_count, 100.0, dtype=np.float64
        ),
        provenance=_provenance(),
    )


class RestrictedWHRouteTests(unittest.TestCase):
    def test_primary_only_route_matches_exact_all_conic_drift(self):
        tile = _tile()
        position = np.asarray([10.0, 0.0, 0.0], dtype=np.float64)
        velocity = np.asarray(
            [0.0, math.sqrt(SOLAR_GM / 10.0), 0.0], dtype=np.float64
        )
        result = route_restricted_wh_map_step(
            position,
            velocity,
            tracer_index=7,
            source_tile=tile,
            source_node_index=0,
            source_steps_per_map_step=1,
        )
        exact = drift_all_conic(position, velocity, SOLAR_GM, 0.01)
        self.assertFalse(result.terminated_at_event)
        np.testing.assert_allclose(
            result.endpoint_relative_position_AU,
            exact.position,
            rtol=0.0,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            result.endpoint_relative_velocity_AU_per_year,
            exact.velocity,
            rtol=0.0,
            atol=2.0e-12,
        )
        self.assertEqual(result.source_tile_sha256, tile.content_sha256)
        self.assertEqual(result.completed_map_fraction, 1.0)
        self.assertFalse(any(result.claims.values()))
        self.assertEqual(dict(result.claims), dict(ROUTE_CLAIMS))
        with self.assertRaises(TypeError):
            result.runtime_identity["scipy_version"] = "forged"

    def test_signed_route_matches_exact_drift(self):
        tile = _tile(step=-0.01)
        position = np.asarray([4.0, 0.0, 0.0], dtype=np.float64)
        velocity = np.asarray(
            [0.0, math.sqrt(SOLAR_GM / 4.0), 0.0], dtype=np.float64
        )
        result = route_restricted_wh_map_step(
            position,
            velocity,
            tracer_index=8,
            source_tile=tile,
            source_node_index=0,
            source_steps_per_map_step=1,
        )
        exact = drift_all_conic(position, velocity, SOLAR_GM, -0.01)
        np.testing.assert_allclose(
            result.endpoint_relative_position_AU,
            exact.position,
            rtol=0.0,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            result.endpoint_relative_velocity_AU_per_year,
            exact.velocity,
            rtol=0.0,
            atol=2.0e-12,
        )

    def test_source_contact_is_continuous_and_primary_relative(self):
        tile = _tile(step=2.0, primary_gm=1.0e-12, primary_radius=1.0)
        result = route_restricted_wh_map_step(
            np.asarray([2.0, 0.0, 0.0], dtype=np.float64),
            np.asarray([-1.0, 0.0, 0.0], dtype=np.float64),
            tracer_index=11,
            source_tile=tile,
            source_node_index=0,
            source_steps_per_map_step=1,
        )
        self.assertTrue(result.terminated_at_event)
        self.assertEqual(result.events[0].event_type, EVENT_SOURCE_RADIUS_CONTACT)
        self.assertEqual(result.events[0].source_id, "Sun")
        self.assertAlmostEqual(
            float(np.linalg.norm(result.terminal_relative_position_AU)),
            1.0,
            delta=result.events[0].surface_tolerance_AU,
        )
        self.assertGreater(result.completed_map_fraction, 0.0)
        self.assertLess(result.completed_map_fraction, 1.0)

    def test_terminal_surface_is_distinct_from_encounter_surface(self):
        tile = _tile(step=0.75, primary_gm=1.0e-12, primary_radius=1.5)
        position = np.asarray([2.0, 0.0, 0.0], dtype=np.float64)
        velocity = np.asarray([-1.0, 0.0, 0.0], dtype=np.float64)
        legacy = route_restricted_wh_map_step(
            position,
            velocity,
            tracer_index=16,
            source_tile=tile,
            source_node_index=0,
            source_steps_per_map_step=1,
        )
        separated = route_restricted_wh_map_step(
            position,
            velocity,
            tracer_index=16,
            source_tile=tile,
            source_node_index=0,
            source_steps_per_map_step=1,
            spec=RestrictedWHRouteSpec(
                terminal_source_radii_AU=(0.5,)
            ),
        )
        self.assertTrue(legacy.terminated_at_event)
        self.assertFalse(separated.terminated_at_event)
        self.assertEqual(separated.completed_map_fraction, 1.0)
        self.assertLess(
            float(np.linalg.norm(separated.endpoint_relative_position_AU)),
            1.5,
        )
        self.assertGreater(
            float(np.linalg.norm(separated.endpoint_relative_position_AU)),
            0.5,
        )

        with self.assertRaisesRegex(
            RestrictedWHRouteContractError, "exactly cover"
        ):
            route_restricted_wh_map_step(
                position,
                velocity,
                tracer_index=16,
                source_tile=tile,
                source_node_index=0,
                source_steps_per_map_step=1,
                spec=RestrictedWHRouteSpec(
                    terminal_source_radii_AU=(0.5, 0.5)
                ),
            )
        with self.assertRaisesRegex(
            RestrictedWHRouteContractError, "cannot exceed"
        ):
            route_restricted_wh_map_step(
                position,
                velocity,
                tracer_index=16,
                source_tile=tile,
                source_node_index=0,
                source_steps_per_map_step=1,
                spec=RestrictedWHRouteSpec(
                    terminal_source_radii_AU=(1.75,)
                ),
            )

    def test_outer_boundary_requires_outward_crossing(self):
        tile = _tile(step=2.0, primary_gm=1.0e-12, primary_radius=0.1)
        result = route_restricted_wh_map_step(
            np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            tracer_index=12,
            source_tile=tile,
            source_node_index=0,
            source_steps_per_map_step=1,
            spec=RestrictedWHRouteSpec(primary_outer_boundary_AU=2.0),
        )
        self.assertTrue(result.terminated_at_event)
        self.assertEqual(
            result.events[0].event_type, EVENT_PRIMARY_BOUNDARY_OUTWARD
        )
        self.assertGreater(
            result.events[0].primary_radial_velocity_AU_per_year, 0.0
        )

    def test_every_fine_source_interval_affects_the_route(self):
        baseline_path = np.asarray(
            [[5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        changed_path = baseline_path.copy()
        changed_path[1] = (4.0, 1.0, 0.0)
        position = np.asarray([2.0, 0.5, 0.0], dtype=np.float64)
        velocity = np.asarray([0.0, 0.7, 0.0], dtype=np.float64)
        baseline = route_restricted_wh_map_step(
            position,
            velocity,
            tracer_index=13,
            source_tile=_tile(
                step=0.01,
                nodes=3,
                primary_gm=1.0,
                secondary_positions=baseline_path,
            ),
            source_node_index=0,
            source_steps_per_map_step=2,
        )
        changed = route_restricted_wh_map_step(
            position,
            velocity,
            tracer_index=13,
            source_tile=_tile(
                step=0.01,
                nodes=3,
                primary_gm=1.0,
                secondary_positions=changed_path,
            ),
            source_node_index=0,
            source_steps_per_map_step=2,
        )
        self.assertFalse(
            np.array_equal(
                baseline.endpoint_relative_position_AU,
                changed.endpoint_relative_position_AU,
            )
        )
        self.assertNotEqual(
            baseline.source_segment_sha256, changed.source_segment_sha256
        )

    def test_segment_hash_is_independent_of_outer_tile_chunking(self):
        full_tile = _tile(nodes=3)
        split_tile = replace(
            full_tile,
            tile_id="restricted-route-split-fixture",
            global_source_step=1,
            source_relative_positions_AU=np.ascontiguousarray(
                full_tile.source_relative_positions_AU[1:]
            ),
            source_relative_velocities_AU_per_year=np.ascontiguousarray(
                full_tile.source_relative_velocities_AU_per_year[1:]
            ),
            content_sha256=None,
        )
        position = np.asarray([10.0, 0.0, 0.0], dtype=np.float64)
        velocity = np.asarray(
            [0.0, math.sqrt(SOLAR_GM / 10.0), 0.0], dtype=np.float64
        )
        embedded = route_restricted_wh_map_step(
            position,
            velocity,
            tracer_index=15,
            source_tile=full_tile,
            source_node_index=1,
            source_steps_per_map_step=1,
        )
        standalone = route_restricted_wh_map_step(
            position,
            velocity,
            tracer_index=15,
            source_tile=split_tile,
            source_node_index=0,
            source_steps_per_map_step=1,
        )
        self.assertNotEqual(
            embedded.source_tile_sha256,
            standalone.source_tile_sha256,
        )
        self.assertEqual(
            embedded.source_segment_sha256,
            standalone.source_segment_sha256,
        )

    def test_phase_ceiling_and_contract_fail_closed(self):
        tile = _tile(step=0.1, primary_gm=1.0, primary_radius=0.01)
        with self.assertRaisesRegex(
            RestrictedWHRouteIntegrationError, "phase ceiling"
        ):
            route_restricted_wh_map_step(
                np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
                np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
                tracer_index=14,
                source_tile=tile,
                source_node_index=0,
                source_steps_per_map_step=1,
                spec=RestrictedWHRouteSpec(
                    event_locator=DOP853EventLocatorSpec(
                        maximum_substep_fraction=0.125
                    ),
                    maximum_primary_dynamical_phase=1.0e-4,
                ),
            )
        with self.assertRaises(RestrictedWHRouteContractError):
            route_restricted_wh_map_step(
                np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
                np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
                tracer_index=14,
                source_tile=tile,
                source_node_index=1,
                source_steps_per_map_step=1,
            )
        with self.assertRaises(RestrictedWHRouteContractError):
            replace(RestrictedWHRouteSpec(), maximum_primary_dynamical_phase=0.3)
        with self.assertRaises(RestrictedWHRouteContractError):
            RestrictedWHRouteSpec(terminal_source_radii_AU=(-0.1,))
        with self.assertRaises(RestrictedWHRouteContractError):
            RestrictedWHRouteSpec(terminal_source_radii_AU=[0.1])

    def test_route_policy_identity_is_stable_sensitive_and_immutable(self):
        baseline = restricted_wh_route_policy_identity(
            RestrictedWHRouteSpec()
        )
        repeated = restricted_wh_route_policy_identity(
            RestrictedWHRouteSpec()
        )
        changed = restricted_wh_route_policy_identity(
            RestrictedWHRouteSpec(
                event_locator=DOP853EventLocatorSpec(
                    maximum_substep_fraction=0.0625
                )
            )
        )
        self.assertEqual(
            baseline["content_sha256"], repeated["content_sha256"]
        )
        self.assertNotEqual(
            baseline["content_sha256"], changed["content_sha256"]
        )
        terminal_changed = restricted_wh_route_policy_identity(
            RestrictedWHRouteSpec(terminal_source_radii_AU=(0.1, 0.2))
        )
        self.assertNotEqual(
            baseline["content_sha256"], terminal_changed["content_sha256"]
        )
        self.assertEqual(
            terminal_changed["terminal_source_radii_AU_hex"],
            (0.1.hex(), 0.2.hex()),
        )
        with self.assertRaises(TypeError):
            baseline["content_sha256"] = "forged"
        with self.assertRaises(TypeError):
            baseline["event_locator"]["maximum_accepted_steps"] = 1


if __name__ == "__main__":
    unittest.main()
