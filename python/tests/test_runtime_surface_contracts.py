from __future__ import annotations

from python.tests.runtime_test_helpers import *


class RuntimeSurfaceContractTests(RuntimeContractTestHelpers):
    def test_summary_only_is_byte_deterministic_under_repeated_runs(self) -> None:
        for seed, ticks in ((7, 40), (GOLDEN_SPECIATION_SEED, 80)):
            first = SimulationWorld(WorldConfig(seed=seed, max_ticks=ticks)).run(
                mode=RunMode.SUMMARY_ONLY
            )
            second = SimulationWorld(WorldConfig(seed=seed, max_ticks=ticks)).run(
                mode=RunMode.SUMMARY_ONLY
            )
            first_bytes = json.dumps(first.summary, separators=(",", ":")).encode("utf-8")
            second_bytes = json.dumps(second.summary, separators=(",", ":")).encode("utf-8")
            self.assertEqual(first_bytes, second_bytes, msg=f"seed={seed} ticks={ticks}")

    def test_summary_only_release_span_is_byte_deterministic_under_repeated_runs(self) -> None:
        config = WorldConfig(
            seed=GOLDEN_SPECIATION_SEED,
            max_ticks=800,
            width=16,
            height=12,
            initial_agents=8,
            max_agents=80,
        )

        first = SimulationWorld(config).run(mode=RunMode.SUMMARY_ONLY)
        second = SimulationWorld(config).run(mode=RunMode.SUMMARY_ONLY)

        first_bytes = json.dumps(first.summary, separators=(",", ":")).encode("utf-8")
        second_bytes = json.dumps(second.summary, separators=(",", ":")).encode("utf-8")
        self.assertEqual(first_bytes, second_bytes)

    def test_summary_only_reuses_resolution_action_masks(self) -> None:
        world = SimulationWorld(WorldConfig(seed=17, max_ticks=80))

        world.run(mode=RunMode.SUMMARY_ONLY)

        observation_builds = world.runtime_cost_counters["observation_builds"]
        action_mask_builds = world.runtime_cost_counters["action_mask_builds"]
        self.assertGreater(observation_builds, 0)
        self.assertGreater(action_mask_builds, observation_builds)
        self.assertLessEqual(action_mask_builds, observation_builds * 2)

    def test_full_replay_capture_delegates_to_runtime_frame_boundary(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))

        with patch("evolution_sim.env.runtime.frames.capture_frame") as capture_frame:
            world._capture_frame(births_this_tick=2, deaths_this_tick=1)

        capture_frame.assert_called_once()
        args, kwargs = capture_frame.call_args
        self.assertEqual(args, (world,))
        self.assertEqual(kwargs["births_this_tick"], 2)
        self.assertEqual(kwargs["deaths_this_tick"], 1)
        self.assertIs(kwargs["trophic_role_codes"], TROPHIC_ROLE_CODES)
        self.assertIs(kwargs["meat_mode_codes"], MEAT_MODE_CODES)
        self.assertIsInstance(
            kwargs["frame_context"],
            runtime_frames.FrameCaptureContext,
        )

    def test_frame_capture_uses_explicit_context(self) -> None:
        expected_world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        expected_context = expected_world._frame_capture_context()
        runtime_frames.capture_frame(
            expected_world,
            births_this_tick=2,
            deaths_this_tick=1,
            trophic_role_codes=TROPHIC_ROLE_CODES,
            meat_mode_codes=MEAT_MODE_CODES,
            frame_context=expected_context,
        )

        actual_world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        context = actual_world._frame_capture_context()
        private_reads = (
            "_refresh_population_snapshots",
            "_trait_means",
            "_frame_surface_context",
            "_materialize_frame_surfaces",
            "_build_agent_frame_telemetry",
            "_build_species_metrics",
            "_fresh_kill_patch_summaries",
            "_carcass_patch_summaries",
            "_signal_runtime_context",
        )
        with ExitStack() as stack:
            for name in private_reads:
                stack.enter_context(
                    patch.object(actual_world, name, side_effect=AssertionError(name))
                )
            runtime_frames.capture_frame(
                actual_world,
                births_this_tick=2,
                deaths_this_tick=1,
                trophic_role_codes=TROPHIC_ROLE_CODES,
                meat_mode_codes=MEAT_MODE_CODES,
                frame_context=context,
            )

        self.assertEqual(actual_world.viewer_frames, expected_world.viewer_frames)
        self.assertEqual(
            actual_world.agent_last_species_map,
            expected_world.agent_last_species_map,
        )
        self.assertEqual(
            actual_world.agent_last_ecotype_map,
            expected_world.agent_last_ecotype_map,
        )

    def test_surface_materialization_uses_explicit_frame_context(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        context = world._frame_surface_context()
        expected = world._materialize_frame_surfaces(surface_context=context)

        private_reads = (
            "_habitat_state_grid",
            "_hydrology_snapshot",
            "_refuge_snapshot",
            "_ecology_snapshot",
            "_hazard_snapshot",
            "_biotic_field_snapshot",
            "_signal_field_snapshot",
            "_fresh_kill_snapshot",
            "_carcass_snapshot",
            "_climate_state",
        )
        with ExitStack() as stack:
            for name in private_reads:
                stack.enter_context(
                    patch.object(world, name, side_effect=AssertionError(name))
                )
            actual = runtime_surfaces.materialize_frame_surfaces(
                world,
                surface_context=context,
            )

        self.assertEqual(actual, expected)

    def test_summary_end_surface_state_uses_explicit_snapshot_context(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        context = world._surface_snapshot_context()
        expected = runtime_surface_snapshots.summary_end_surface_state(
            RunMode.SUMMARY_ONLY,
            context=context,
        )

        private_reads = (
            "_habitat_state_grid",
            "_hydrology_snapshot",
            "_refuge_snapshot",
            "_ecology_snapshot",
            "_hazard_snapshot",
            "_biotic_field_snapshot",
            "_signal_field_snapshot",
            "_fresh_kill_snapshot",
            "_carcass_snapshot",
        )
        with ExitStack() as stack:
            for name in private_reads:
                stack.enter_context(
                    patch.object(world, name, side_effect=AssertionError(name))
                )
            actual = runtime_surface_snapshots.summary_end_surface_state(
                RunMode.SUMMARY_ONLY,
                context=context,
            )

        self.assertEqual(actual, expected)

    def test_agent_frame_telemetry_uses_explicit_surface_context(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        alive = world.alive_agents()
        base_context = world._frame_surface_context()
        context = replace(
            base_context,
            energy_ratio=lambda agent: 0.11,
            hydration_ratio=lambda agent: 0.22,
            health_ratio=lambda agent: 0.33,
            agent_energy_drain_modifier=lambda agent, season: 0.44,
            agent_hydration_drain_modifier=lambda agent, season: 0.55,
            is_reproduction_ready=lambda agent: True,
            trophic_role=lambda agent: "carnivore",
            meat_mode=lambda agent: "hunter",
            refuge_score=lambda x, y: 0.66,
            matched_diet_ratio=lambda agent: 0.77,
        )
        surfaces = world._materialize_frame_surfaces(surface_context=context)
        season = str(context.climate_state["season"])
        actual = runtime_surfaces.build_agent_frame_telemetry(
            world,
            alive,
            season=season,
            surfaces=surfaces,
            surface_context=context,
        )

        self.assertTrue(actual)
        for telemetry in actual.values():
            self.assertEqual(telemetry["energy_ratio"], 0.11)
            self.assertEqual(telemetry["hydration_ratio"], 0.22)
            self.assertEqual(telemetry["health_ratio"], 0.33)
            self.assertEqual(telemetry["energy_modifier"], 0.44)
            self.assertEqual(telemetry["hydration_modifier"], 0.55)
            self.assertTrue(telemetry["reproduction_ready"])
            self.assertEqual(telemetry["trophic_role"], "carnivore")
            self.assertEqual(telemetry["meat_mode"], "hunter")
            self.assertEqual(telemetry["refuge_score"], 0.66)
            self.assertEqual(telemetry["matched_diet_ratio"], 0.77)

    def test_full_replay_frame_capture_uses_surface_context_boundary(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=2))

        with (
            patch(
                "evolution_sim.env.runtime.surfaces.materialize_frame_surfaces",
                wraps=runtime_surfaces.materialize_frame_surfaces,
            ) as materialize_surfaces,
            patch(
                "evolution_sim.env.runtime.surfaces.build_agent_frame_telemetry",
                wraps=runtime_surfaces.build_agent_frame_telemetry,
            ) as build_telemetry,
        ):
            result = world.run(mode=RunMode.FULL_REPLAY)

        self.assertIsNotNone(result.viewer)
        self.assertGreater(len(result.viewer["frames"]), 0)
        self.assertGreater(materialize_surfaces.call_count, 0)
        self.assertGreater(build_telemetry.call_count, 0)
        for call in materialize_surfaces.call_args_list:
            self.assertIsInstance(
                call.kwargs["surface_context"],
                runtime_surfaces.FrameSurfaceContext,
            )
        for call in build_telemetry.call_args_list:
            self.assertIsInstance(
                call.kwargs["surface_context"],
                runtime_surfaces.FrameSurfaceContext,
            )

    def test_frame_capture_does_not_write_resource_run_final_totals(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))

        world._capture_frame(births_this_tick=0, deaths_this_tick=0)

        self.assertIn("fresh_kill_stats", world.viewer_frames[-1])
        self.assertIn("carcass_stats", world.viewer_frames[-1])
        self.assertNotIn("fresh_kill_tiles", world.run_fresh_kill_totals)
        self.assertNotIn("total_fresh_kill_energy", world.run_fresh_kill_totals)
        self.assertNotIn("carcass_tiles", world.run_carcass_totals)
        self.assertNotIn("total_carcass_energy", world.run_carcass_totals)
