from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    from evolution_sim.config.schema import (
        CombatConfig,
        DietMatchingConfig,
        ReproductionConfig,
        WorldConfig,
    )
    from evolution_sim.env.runtime.state import RunMode
    from evolution_sim.env.world import SimulationWorld
    from evolution_sim.io.open_ecology_checkpoint import (
        REQUIRED_CHECKPOINT_COMPONENTS,
        VersionedCheckpointState,
        load_open_ecology_checkpoint,
        write_open_ecology_checkpoint,
    )
    from evolution_sim.io.open_ecology_rotating_writer import (
        InterventionEvidence,
        RotatingOpenEcologyEvidenceWriter,
        load_open_ecology_evidence_manifest,
    )
    from evolution_sim.io.open_ecology_runtime_checkpoint import (
        OpenEcologyRuntimeCheckpointError,
        RuntimeCheckpointBinding,
        RuntimeCheckpointComponents,
        capture_open_ecology_runtime_checkpoint,
        restore_open_ecology_runtime_checkpoint,
    )
    from evolution_sim.mind.recurrent_actor_critic import (
        GENOME_CONDITIONING_ACTOR_FILM_V1,
        PublicRecurrentActorCritic,
        RecurrentActorCriticConfig,
    )
    from evolution_sim.mind.recurrent_policy import (
        DeterministicPublicRecurrentPolicy,
    )


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


@unittest.skipIf(torch is None, "optional Mind ML dependency torch is not installed")
class OpenEcologyRuntimeCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        assert torch is not None
        torch.set_num_threads(1)

    def test_h_z_and_recurrent_reset_worlds_continue_exactly(self) -> None:
        arms = (
            ("H", "heritable", False),
            ("Z", "zero_all", False),
            ("R", "heritable", True),
        )
        for arm_name, population_mode, reset_each_decision in arms:
            with self.subTest(arm=arm_name), TemporaryDirectory() as temporary:
                root = Path(temporary)
                source_contract = {
                    "schema_version": "runtime-checkpoint-test-source-v1",
                    "arm": arm_name,
                }
                baseline_world, baseline_policy = self._make_runtime(
                    population_mode=population_mode,
                    reset_each_decision=reset_each_decision,
                )
                evidence_directory = root / "baseline-evidence"
                initial_writer = RotatingOpenEcologyEvidenceWriter(
                    evidence_directory,
                    run_id=f"runtime-checkpoint-{arm_name}",
                    source_contract=source_contract,
                )
                pre_rows = self._run_ticks(
                    baseline_world,
                    start=0,
                    stop=3,
                    writer=initial_writer,
                    event_index_start=0,
                    arm_name=arm_name,
                )
                continuation = initial_writer.checkpoint()
                resumed_evidence_directory = root / "resumed-evidence"
                shutil.copytree(evidence_directory, resumed_evidence_directory)

                config_contract = VersionedCheckpointState(
                    "runtime_checkpoint_test_config_v1",
                    {
                        "arm": arm_name,
                        "world_config": baseline_world.config.to_dict(),
                    },
                )
                seed_contract = VersionedCheckpointState(
                    "runtime_checkpoint_test_seeds_v1",
                    {
                        "environment_seed": baseline_world.config.seed,
                        "genome_stream_seed": 1776,
                        "model_seed": 711,
                        "sampling_seed": 991,
                    },
                )
                binding = RuntimeCheckpointBinding(
                    source_git_sha="a" * 40,
                    source_manifest_sha256="f" * 64,
                    config_contract_sha256=_canonical_digest(
                        {
                            "schema_version": config_contract.schema_version,
                            "payload": dict(config_contract.payload),
                        }
                    ),
                    seed_contract_sha256=_canonical_digest(
                        {
                            "schema_version": seed_contract.schema_version,
                            "payload": dict(seed_contract.payload),
                        }
                    ),
                    run_generation_id="runtime-checkpoint-generation",
                    island_id=f"island-{arm_name}",
                    generation_index=7,
                    completed_tick=2,
                )
                components = capture_open_ecology_runtime_checkpoint(
                    baseline_world,
                    baseline_policy,
                    binding=binding,
                    evidence_writer_continuation_state=continuation,
                )
                if arm_name == "R":
                    self.assertFalse(baseline_policy._state_by_agent)
                    self.assertFalse(baseline_policy._feedback_by_agent)
                    for records in baseline_policy._public_history_by_agent.values():
                        self.assertTrue(
                            all(
                                all(
                                    value == 0.0
                                    for value in record["previous_public_feedback"][
                                        "values"
                                    ]
                                )
                                for record in records
                            )
                        )
                checkpoint_path = root / "runtime-checkpoint.json"
                write_open_ecology_checkpoint(
                    checkpoint_path,
                    source_git_sha=binding.source_git_sha,
                    config_contract=config_contract,
                    seed_contract=seed_contract,
                    run_generation_id=binding.run_generation_id,
                    island_id=binding.island_id,
                    generation_index=binding.generation_index,
                    tick=binding.completed_tick,
                    world_state=components.world_state,
                    environment_rng_state=components.environment_rng_state,
                    recurrent_policy_state=components.recurrent_policy_state,
                    public_feedback_history=components.public_feedback_history,
                    sampling_rng_state=components.sampling_rng_state,
                    genome_population_snapshot=components.genome_population_snapshot,
                    evidence_writer_continuation_state=(
                        components.evidence_writer_continuation_state
                    ),
                )
                loaded_checkpoint = load_open_ecology_checkpoint(
                    checkpoint_path,
                    expected_source_git_sha=binding.source_git_sha,
                    require_restartable=True,
                )
                loaded_envelopes = loaded_checkpoint["components"]
                assert isinstance(loaded_envelopes, dict)
                loaded_states = {}
                for component_name in REQUIRED_CHECKPOINT_COMPONENTS:
                    envelope = loaded_envelopes[component_name]
                    assert isinstance(envelope, dict)
                    loaded_states[component_name] = VersionedCheckpointState(
                        schema_version=str(envelope["schema_version"]),
                        payload=envelope["payload"],  # type: ignore[arg-type]
                    )
                components = RuntimeCheckpointComponents(**loaded_states)
                resumed_world, resumed_policy = self._make_runtime(
                    population_mode=population_mode,
                    reset_each_decision=reset_each_decision,
                )
                restored_continuation = restore_open_ecology_runtime_checkpoint(
                    resumed_world,
                    resumed_policy,
                    binding=binding,
                    components=components,
                )
                self.assertEqual(restored_continuation, continuation)

                baseline_writer = RotatingOpenEcologyEvidenceWriter(
                    evidence_directory,
                    run_id=f"runtime-checkpoint-{arm_name}",
                    source_contract=source_contract,
                    continuation_state=continuation,
                )
                resumed_writer = RotatingOpenEcologyEvidenceWriter(
                    resumed_evidence_directory,
                    run_id=f"runtime-checkpoint-{arm_name}",
                    source_contract=source_contract,
                    continuation_state=restored_continuation,
                )
                baseline_rows = self._run_ticks(
                    baseline_world,
                    start=3,
                    stop=7,
                    writer=baseline_writer,
                    event_index_start=3,
                    arm_name=arm_name,
                )
                resumed_rows = self._run_ticks(
                    resumed_world,
                    start=3,
                    stop=7,
                    writer=resumed_writer,
                    event_index_start=3,
                    arm_name=arm_name,
                )
                baseline_manifest = baseline_writer.finish()
                resumed_manifest = resumed_writer.finish()

                self.assertEqual(baseline_rows, resumed_rows)
                self.assertEqual(
                    [row["requested_action"] for rows in baseline_rows for row in rows],
                    [row["requested_action"] for rows in resumed_rows for row in rows],
                )
                self.assertEqual(
                    [row["resolved_action"] for rows in baseline_rows for row in rows],
                    [row["resolved_action"] for rows in resumed_rows for row in rows],
                )
                self.assertEqual(baseline_world.grid, resumed_world.grid)
                self.assertEqual(baseline_world.agents, resumed_world.agents)
                self.assertEqual(
                    baseline_world.rng.getstate(),
                    resumed_world.rng.getstate(),
                )
                self.assertEqual(
                    baseline_world._build_summary(RunMode.SUMMARY_ONLY),
                    resumed_world._build_summary(RunMode.SUMMARY_ONLY),
                )
                self.assertEqual(
                    baseline_policy._decision_index,
                    resumed_policy._decision_index,
                )
                self.assertEqual(
                    baseline_policy._feedback_by_agent,
                    resumed_policy._feedback_by_agent,
                )
                self.assertEqual(
                    baseline_policy._public_history_by_agent,
                    resumed_policy._public_history_by_agent,
                )
                self.assertEqual(
                    set(baseline_policy._state_by_agent),
                    set(resumed_policy._state_by_agent),
                )
                for agent_id, state in baseline_policy._state_by_agent.items():
                    self.assertTrue(
                        torch.equal(state, resumed_policy._state_by_agent[agent_id])
                    )
                self.assertEqual(
                    baseline_policy._sampling_generator.get_state().tolist(),
                    resumed_policy._sampling_generator.get_state().tolist(),
                )
                self.assertEqual(
                    baseline_policy._genome_population_manager.snapshot_artifact(),
                    resumed_policy._genome_population_manager.snapshot_artifact(),
                )
                self.assertEqual(baseline_manifest, resumed_manifest)
                self.assertEqual(
                    load_open_ecology_evidence_manifest(
                        evidence_directory,
                        verify_shards=True,
                    ),
                    load_open_ecology_evidence_manifest(
                        resumed_evidence_directory,
                        verify_shards=True,
                    ),
                )
                self.assertEqual(
                    _canonical_digest(pre_rows + baseline_rows),
                    _canonical_digest(pre_rows + resumed_rows),
                )

    def test_capture_rejects_unbounded_retained_trajectory_state(self) -> None:
        world, policy = self._make_runtime(
            population_mode="heritable",
            reset_each_decision=False,
        )
        world.retain_trajectory_records = True
        binding = RuntimeCheckpointBinding(
            source_git_sha="a" * 40,
            source_manifest_sha256="f" * 64,
            config_contract_sha256="b" * 64,
            seed_contract_sha256="c" * 64,
            run_generation_id="runtime-checkpoint-generation",
            island_id="island-H",
            generation_index=0,
            completed_tick=0,
        )
        with TemporaryDirectory() as temporary:
            writer = RotatingOpenEcologyEvidenceWriter(
                Path(temporary) / "evidence",
                run_id="runtime-checkpoint-H",
                source_contract={"schema_version": "test-v1"},
            )
            continuation = writer.checkpoint()
            with self.assertRaisesRegex(
                OpenEcologyRuntimeCheckpointError,
                "bounded no-retention",
            ):
                capture_open_ecology_runtime_checkpoint(
                    world,
                    policy,
                    binding=binding,
                    evidence_writer_continuation_state=continuation,
                )

    def test_checkpoint_payload_owns_history_across_capture_and_restore(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            world, policy, binding, components = self._capture_one_tick(root)
            history_state = components.public_feedback_history.payload["state"]
            assert isinstance(history_state, dict)
            history_entries = history_state["history_by_agent"]
            assert isinstance(history_entries, list)
            first_entry = history_entries[0]
            assert isinstance(first_entry, dict)
            captured_records = first_entry["records"]
            assert isinstance(captured_records, list)
            agent_id = int(first_entry["agent_id"])
            self.assertIsNot(
                captured_records,
                policy._public_history_by_agent[agent_id],
            )
            captured_before = deepcopy(captured_records)

            policy._public_history_by_agent[agent_id].append(
                deepcopy(policy._public_history_by_agent[agent_id][-1])
            )
            self.assertEqual(captured_records, captured_before)

            restored_world, restored_policy = self._make_runtime(
                population_mode="heritable",
                reset_each_decision=False,
            )
            restore_open_ecology_runtime_checkpoint(
                restored_world,
                restored_policy,
                binding=binding,
                components=components,
            )
            self.assertIsNot(
                captured_records,
                restored_policy._public_history_by_agent[agent_id],
            )
            restored_policy._public_history_by_agent[agent_id].append(
                deepcopy(restored_policy._public_history_by_agent[agent_id][-1])
            )
            self.assertEqual(captured_records, captured_before)

            second_world, second_policy = self._make_runtime(
                population_mode="heritable",
                reset_each_decision=False,
            )
            restore_open_ecology_runtime_checkpoint(
                second_world,
                second_policy,
                binding=binding,
                components=components,
            )
            self.assertEqual(
                len(second_policy._public_history_by_agent[agent_id]),
                len(captured_before),
            )

    def test_restore_is_transactional_when_late_writer_validation_fails(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, binding, components = self._capture_one_tick(root)
            invalid_components = RuntimeCheckpointComponents(
                world_state=components.world_state,
                environment_rng_state=components.environment_rng_state,
                recurrent_policy_state=components.recurrent_policy_state,
                public_feedback_history=components.public_feedback_history,
                sampling_rng_state=components.sampling_rng_state,
                genome_population_snapshot=components.genome_population_snapshot,
                evidence_writer_continuation_state=VersionedCheckpointState(
                    components.evidence_writer_continuation_state.schema_version,
                    {
                        "binding": binding.to_dict(),
                        "state": {"continuation_state": {}},
                    },
                ),
            )
            fresh_world, fresh_policy = self._make_runtime(
                population_mode="heritable",
                reset_each_decision=False,
            )
            before = {
                "world_tick": fresh_world.tick,
                "world_agents": deepcopy(fresh_world.agents),
                "world_grid": deepcopy(fresh_world.grid),
                "world_rng": fresh_world.rng.getstate(),
                "decision_index": fresh_policy._decision_index,
                "hidden_ids": tuple(fresh_policy._state_by_agent),
                "feedback": deepcopy(fresh_policy._feedback_by_agent),
                "history": deepcopy(fresh_policy._public_history_by_agent),
                "sampling": (fresh_policy._sampling_generator.get_state().tolist()),
                "population": (
                    fresh_policy._genome_population_manager.snapshot_artifact()
                ),
            }
            with self.assertRaisesRegex(
                OpenEcologyRuntimeCheckpointError,
                "evidence writer continuation is invalid",
            ):
                restore_open_ecology_runtime_checkpoint(
                    fresh_world,
                    fresh_policy,
                    binding=binding,
                    components=invalid_components,
                )
            after = {
                "world_tick": fresh_world.tick,
                "world_agents": fresh_world.agents,
                "world_grid": fresh_world.grid,
                "world_rng": fresh_world.rng.getstate(),
                "decision_index": fresh_policy._decision_index,
                "hidden_ids": tuple(fresh_policy._state_by_agent),
                "feedback": fresh_policy._feedback_by_agent,
                "history": fresh_policy._public_history_by_agent,
                "sampling": (fresh_policy._sampling_generator.get_state().tolist()),
                "population": (
                    fresh_policy._genome_population_manager.snapshot_artifact()
                ),
            }
            self.assertEqual(after, before)

    def test_restore_rejects_wrong_fresh_genome_stream_without_mutation(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, binding, components = self._capture_one_tick(root)
            fresh_world, fresh_policy = self._make_runtime(
                population_mode="heritable",
                reset_each_decision=False,
                world_identity="wrong-runtime-world",
                genome_stream_seed=999,
            )
            before_population = (
                fresh_policy._genome_population_manager.snapshot_artifact()
            )
            with self.assertRaisesRegex(
                OpenEcologyRuntimeCheckpointError,
                "genome population genome_stream_seed does not match",
            ):
                restore_open_ecology_runtime_checkpoint(
                    fresh_world,
                    fresh_policy,
                    binding=binding,
                    components=components,
                )
            self.assertEqual(
                fresh_policy._genome_population_manager.snapshot_artifact(),
                before_population,
            )
            self.assertEqual(fresh_policy._decision_index, 0)

    def test_capture_rejects_agent_metadata_outside_genome_population(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            world, policy = self._make_runtime(
                population_mode="heritable",
                reset_each_decision=False,
            )
            writer = RotatingOpenEcologyEvidenceWriter(
                root / "evidence",
                run_id="runtime-checkpoint-H",
                source_contract={"schema_version": "test-v1"},
            )
            self._run_ticks(
                world,
                start=0,
                stop=1,
                writer=writer,
                event_index_start=0,
                arm_name="H",
            )
            world.alive_agents()[0].mind_inheritance_metadata["genome_sha256"] = (
                "0" * 64
            )
            with self.assertRaisesRegex(
                OpenEcologyRuntimeCheckpointError,
                "mind inheritance metadata does not match",
            ):
                capture_open_ecology_runtime_checkpoint(
                    world,
                    policy,
                    binding=self._binding(completed_tick=0),
                    evidence_writer_continuation_state=writer.checkpoint(),
                )

    def test_capture_requires_evidence_for_the_same_finalized_tick(self) -> None:
        world, policy = self._make_runtime(
            population_mode="heritable",
            reset_each_decision=False,
        )
        world.tick = 4_999
        binding = self._binding(completed_tick=4_999)
        with TemporaryDirectory() as temporary:
            writer = RotatingOpenEcologyEvidenceWriter(
                Path(temporary) / "evidence",
                run_id="runtime-checkpoint-H",
                source_contract={"schema_version": "test-v1"},
            )
            with self.assertRaisesRegex(
                OpenEcologyRuntimeCheckpointError,
                "last_tick does not match",
            ):
                capture_open_ecology_runtime_checkpoint(
                    world,
                    policy,
                    binding=binding,
                    evidence_writer_continuation_state=writer.checkpoint(),
                )

    def test_persistent_runner_state_is_exact_json_owned_and_writer_bound(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            world, policy = self._make_runtime(
                population_mode="heritable",
                reset_each_decision=False,
            )
            writer = RotatingOpenEcologyEvidenceWriter(
                root / "evidence",
                run_id="runtime-checkpoint-H",
                source_contract={"schema_version": "test-v1"},
            )
            self._run_ticks(
                world,
                start=0,
                stop=1,
                writer=writer,
                event_index_start=0,
                arm_name="H",
            )
            continuation = writer.checkpoint()
            binding = self._binding(completed_tick=0)
            runner_state = self._runner_state_payload(
                policy=policy,
                continuation=continuation,
            )
            world._open_ecology_persistent_runner_state = runner_state
            components = capture_open_ecology_runtime_checkpoint(
                world,
                policy,
                binding=binding,
                evidence_writer_continuation_state=continuation,
            )
            interval = runner_state["interval"]
            assert isinstance(interval, dict)
            interval["births"] = 999

            fresh_world, fresh_policy = self._make_runtime(
                population_mode="heritable",
                reset_each_decision=False,
            )
            fresh_world._open_ecology_persistent_runner_state = {}
            restore_open_ecology_runtime_checkpoint(
                fresh_world,
                fresh_policy,
                binding=binding,
                components=components,
            )
            restored_state = fresh_world._open_ecology_persistent_runner_state
            self.assertIsNot(restored_state, runner_state)
            restored_interval = restored_state["interval"]
            assert isinstance(restored_interval, dict)
            self.assertEqual(restored_interval["births"], 0)

            invalid_world, invalid_policy = self._make_runtime(
                population_mode="heritable",
                reset_each_decision=False,
            )
            invalid_world.tick = 0
            invalid_state = self._runner_state_payload(
                policy=invalid_policy,
                continuation=continuation,
            )
            invalid_interval = invalid_state["interval"]
            assert isinstance(invalid_interval, dict)
            invalid_interval["arbitrary"] = object()
            invalid_world._open_ecology_persistent_runner_state = invalid_state
            with self.assertRaisesRegex(
                OpenEcologyRuntimeCheckpointError,
                "contains non-JSON value",
            ):
                capture_open_ecology_runtime_checkpoint(
                    invalid_world,
                    invalid_policy,
                    binding=binding,
                    evidence_writer_continuation_state=continuation,
                )

    def test_persistent_runner_checkpoint_allows_evidence_to_lag_world_tick(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            world, policy = self._make_runtime(
                population_mode="heritable",
                reset_each_decision=False,
            )
            writer = RotatingOpenEcologyEvidenceWriter(
                root / "evidence",
                run_id="runtime-checkpoint-H",
                source_contract={"schema_version": "test-v1"},
            )
            self._run_ticks(
                world,
                start=0,
                stop=1,
                writer=writer,
                event_index_start=0,
                arm_name="H",
            )
            continuation = writer.checkpoint()
            world.tick = 1
            world._run_tick()
            binding = self._binding(completed_tick=1)
            world._open_ecology_persistent_runner_state = self._runner_state_payload(
                policy=policy,
                continuation=continuation,
                observed_tick=2,
                completed_world_tick=1,
            )

            components = capture_open_ecology_runtime_checkpoint(
                world,
                policy,
                binding=binding,
                evidence_writer_continuation_state=continuation,
            )

            fresh_world, fresh_policy = self._make_runtime(
                population_mode="heritable",
                reset_each_decision=False,
            )
            fresh_world._open_ecology_persistent_runner_state = {}
            restored_continuation = restore_open_ecology_runtime_checkpoint(
                fresh_world,
                fresh_policy,
                binding=binding,
                components=components,
            )
            self.assertEqual(restored_continuation["last_tick"], 0)
            self.assertEqual(fresh_world.tick, 1)

    @classmethod
    def _capture_one_tick(
        cls,
        root: Path,
    ) -> tuple[
        SimulationWorld,
        DeterministicPublicRecurrentPolicy,
        RuntimeCheckpointBinding,
        RuntimeCheckpointComponents,
    ]:
        world, policy = cls._make_runtime(
            population_mode="heritable",
            reset_each_decision=False,
        )
        writer = RotatingOpenEcologyEvidenceWriter(
            root / "evidence",
            run_id="runtime-checkpoint-H",
            source_contract={"schema_version": "test-v1"},
        )
        cls._run_ticks(
            world,
            start=0,
            stop=1,
            writer=writer,
            event_index_start=0,
            arm_name="H",
        )
        binding = cls._binding(completed_tick=0)
        components = capture_open_ecology_runtime_checkpoint(
            world,
            policy,
            binding=binding,
            evidence_writer_continuation_state=writer.checkpoint(),
        )
        return world, policy, binding, components

    @staticmethod
    def _binding(*, completed_tick: int) -> RuntimeCheckpointBinding:
        return RuntimeCheckpointBinding(
            source_git_sha="a" * 40,
            source_manifest_sha256="f" * 64,
            config_contract_sha256="b" * 64,
            seed_contract_sha256="c" * 64,
            run_generation_id="runtime-checkpoint-generation",
            island_id="island-H",
            generation_index=0,
            completed_tick=completed_tick,
        )

    @staticmethod
    def _runner_state_payload(
        *,
        policy: DeterministicPublicRecurrentPolicy,
        continuation: dict[str, object],
        observed_tick: int = 1,
        completed_world_tick: int = 0,
    ) -> dict[str, object]:
        body: dict[str, object] = {
            "schema_version": (
                "mind_v3_open_ecology_persistent_island_runner_state_v1"
            ),
            "task_sha256": "1" * 64,
            "model_state_sha256": policy._model_state_sha256,
            "observed_tick": observed_tick,
            "completed_world_tick": completed_world_tick,
            "extinct": False,
            "extinction_tick": None,
            "episode_reset_count": 0,
            "world_replacement_count": 0,
            "interval": {
                "births": 0,
                "deaths": 0,
                "requested_actions": {},
                "resolved_actions": {},
            },
            "summaries": [],
            "milestones": [],
            "event_counts": {"intervention": 1},
            "writer_initial_event_count": 0,
            "next_event_index": continuation["next_event_index"],
            "last_evidence_tick": continuation["last_tick"],
            "evidence_continuation_sha256": continuation["state_sha256"],
            "evidence_finished": False,
            "evidence_aborted": False,
        }
        return {**body, "runner_state_sha256": _canonical_digest(body)}

    @staticmethod
    def _make_runtime(
        *,
        population_mode: str,
        reset_each_decision: bool,
        world_identity: str = "runtime-checkpoint-world",
        genome_stream_seed: int = 1776,
    ) -> tuple[SimulationWorld, DeterministicPublicRecurrentPolicy]:
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(
                encoder_size=16,
                hidden_size=16,
                genome_conditioning_mode=GENOME_CONDITIONING_ACTOR_FILM_V1,
            ),
            initialization_seed=711,
        )
        policy = DeterministicPublicRecurrentPolicy(
            model,
            artifact_digest="d" * 64,
            reset_recurrent_state_each_decision=reset_each_decision,
            sampling_seed=991,
            capture_public_history=True,
        )
        policy.start_world(
            world_identity=world_identity,
            genome_stream_seed=genome_stream_seed,
            genome_population_mode=population_mode,
        )
        world = SimulationWorld(
            WorldConfig(
                seed=37,
                max_ticks=20,
                width=5,
                height=5,
                initial_agents=3,
                max_agents=20,
                water_tile_ratio=0.0,
                forest_tile_ratio=0.0,
                wetland_tile_ratio=0.0,
                rocky_tile_ratio=0.0,
                base_energy_drain=0.0,
                base_hydration_drain=0.0,
                reproduction=ReproductionConfig(
                    min_age=1,
                    cooldown_ticks=1_000,
                    min_hydration_fraction=0.0,
                    energy_cost=0.0,
                    child_energy_fraction=0.3,
                ),
                diet_matching=DietMatchingConfig(
                    specialist_threshold=0.0,
                    omnivore_threshold=0.0,
                ),
                combat=CombatConfig(
                    min_attack_health_ratio=1.0,
                    min_attack_energy_ratio=1.0,
                    min_attack_hydration_ratio=1.0,
                    base_attack_damage=0.0,
                    attack_energy_cost=0.0,
                    attack_hydration_cost=0.0,
                ),
            ),
            policy=policy,
        )
        world.record_events = False
        world.record_tick_details = True
        world.record_trajectory = True
        world.retain_trajectory_records = False
        return world, policy

    @staticmethod
    def _run_ticks(
        world: SimulationWorld,
        *,
        start: int,
        stop: int,
        writer: RotatingOpenEcologyEvidenceWriter,
        event_index_start: int,
        arm_name: str,
    ) -> list[list[dict[str, object]]]:
        result: list[list[dict[str, object]]] = []
        for offset, tick in enumerate(range(start, stop)):
            event_index = event_index_start + offset
            world.tick = tick
            world._run_tick()
            rows = deepcopy(world.tick_trajectory_records)
            result.append(rows)
            writer.append(
                InterventionEvidence(
                    tick=tick,
                    event_index=event_index,
                    event_id=f"{arm_name}-tick-{tick}:{event_index}",
                    intervention_id=f"runtime-checkpoint-{arm_name}",
                    intervention_kind="continuation-equivalence",
                    branch_id=f"tick-{tick}",
                    value_sha256=_canonical_digest(rows),
                )
            )
        return result
