from __future__ import annotations

import dataclasses
import unittest

import torch

from evolution_sim.config import (
    CombatConfig,
    DietMatchingConfig,
    ReproductionConfig,
    TrophicConfig,
    WorldConfig,
)
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.state import RunMode
from evolution_sim.env.runtime.ticks import deterministic_agent_turn_order
from evolution_sim.env.world import SimulationWorld
from evolution_sim.mind.policy_inputs import ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE
from evolution_sim.mind.open_ecology_seed_registry import (
    OPEN_ECOLOGY_SEED_REGISTRY,
)
from evolution_sim.mind.recurrent_actor_critic import (
    PublicRecurrentActorCritic,
    RecurrentActorCriticConfig,
)
from evolution_sim.mind.recurrent_experiment import (
    OPEN_ECOLOGY_PHASE_A,
    OPEN_ECOLOGY_PHASE_B,
    RECURRENT_EXPERIMENT_CONTRACT_VERSION,
    RECURRENT_FIXED_BATCH_EXPERIMENT_CONTRACT_VERSION,
    OpenEcologySignalTreatment,
    RecurrentExperimentError,
    RecurrentExperimentRunner,
    _collect_recurrent_rollout_task,
    _validate_rollout_worker_result,
    build_open_ecology_training_schedule,
    build_recurrent_training_schedule,
    collect_recurrent_rollout_batch,
)
from evolution_sim.mind.recurrent_ppo import RecurrentPPOConfig
from evolution_sim.mind.recurrent_rollout import (
    OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
    RECURRENT_FIXED_BATCH_RELEASE_STATUS,
    RECURRENT_FIXED_BATCH_ROLLOUT_DIAGNOSTIC_SCHEMA_VERSION,
    RECURRENT_GENOME_CONDITIONING_ACTOR_FILM_V1,
    RECURRENT_GENOME_FIXED_BATCH_ROLLOUT_DIAGNOSTIC_SCHEMA_VERSION,
    PreviousPublicFeedback,
    RecurrentCoreOutput,
    RecurrentFixedBatchRuntimeContract,
    RecurrentOnPolicyCollector,
    RecurrentRolloutError,
    RecurrentRolloutStep,
    TorchRecurrentPolicyCore,
    _fixed_batch_runtime_binding,
)


class _ExactBatchCore:
    hidden_size = 3

    def __init__(self, *, preferred_action: str | None = None) -> None:
        self.preferred_action = preferred_action
        self.batch_calls: list[tuple[int, int, bool]] = []
        self.scalar_calls = 0

    def initial_hidden(self) -> tuple[float, ...]:
        return (0.0, 0.0, 0.0)

    def _output(
        self,
        observation: tuple[float, ...],
        current_action_mask: tuple[bool, ...],
        previous_feedback: PreviousPublicFeedback,
        hidden: tuple[float, ...],
    ) -> RecurrentCoreOutput:
        signal = (
            sum(observation[:8])
            + sum(current_action_mask)
            + sum(previous_feedback.vector())
        ) * 0.001
        logits = [
            signal + action_index * 0.01 for action_index in range(len(ACTION_NAMES))
        ]
        if self.preferred_action is not None:
            logits[ACTION_NAMES.index(self.preferred_action)] = 100.0
        return RecurrentCoreOutput(
            logits=tuple(logits),
            value=sum(observation[:3]) * 0.1 + sum(hidden) * 0.01,
            next_hidden=tuple(value + 1.0 for value in hidden),
        )

    def forward_step(
        self,
        observation: tuple[float, ...],
        current_action_mask: tuple[bool, ...],
        previous_feedback: PreviousPublicFeedback,
        hidden: tuple[float, ...],
    ) -> RecurrentCoreOutput:
        self.scalar_calls += 1
        return self._output(
            observation,
            current_action_mask,
            previous_feedback,
            hidden,
        )

    def forward_fixed_batch(
        self,
        observations: tuple[tuple[float, ...], ...],
        current_action_masks: tuple[tuple[bool, ...], ...],
        previous_feedback: tuple[PreviousPublicFeedback, ...],
        hidden: tuple[tuple[float, ...], ...],
        *,
        batch_capacity: int,
        genome_values: tuple[tuple[float, ...], ...] | None = None,
    ) -> tuple[RecurrentCoreOutput, ...]:
        self.batch_calls.append(
            (len(observations), batch_capacity, genome_values is not None)
        )
        return tuple(
            self._output(observation, mask, feedback, state)
            for observation, mask, feedback, state in zip(
                observations,
                current_action_masks,
                previous_feedback,
                hidden,
                strict=True,
            )
        )

    def fixed_batch_runtime_metadata(self) -> dict[str, object]:
        return {
            "implementation": "test_exact_batch_core_v1",
            "device_type": "cpu",
            "dtype": "python_float",
        }


class _ConditionedExactBatchCore(_ExactBatchCore):
    genome_conditioning_mode = RECURRENT_GENOME_CONDITIONING_ACTOR_FILM_V1

    def __init__(self) -> None:
        super().__init__()
        self.batch_genome_rows: list[tuple[tuple[float, ...], ...]] = []

    def forward_step(
        self,
        observation: tuple[float, ...],
        current_action_mask: tuple[bool, ...],
        previous_feedback: PreviousPublicFeedback,
        hidden: tuple[float, ...],
        *,
        genome_values: tuple[float, ...] | None = None,
    ) -> RecurrentCoreOutput:
        if genome_values is None:
            raise AssertionError("conditioned scalar forward requires a genome")
        return super().forward_step(
            observation,
            current_action_mask,
            previous_feedback,
            hidden,
        )

    def forward_fixed_batch(
        self,
        observations: tuple[tuple[float, ...], ...],
        current_action_masks: tuple[tuple[bool, ...], ...],
        previous_feedback: tuple[PreviousPublicFeedback, ...],
        hidden: tuple[tuple[float, ...], ...],
        *,
        batch_capacity: int,
        genome_values: tuple[tuple[float, ...], ...] | None = None,
    ) -> tuple[RecurrentCoreOutput, ...]:
        if genome_values is None:
            raise AssertionError("conditioned batched forward requires genomes")
        self.batch_genome_rows.append(genome_values)
        return super().forward_fixed_batch(
            observations,
            current_action_masks,
            previous_feedback,
            hidden,
            batch_capacity=batch_capacity,
            genome_values=genome_values,
        )


class RecurrentFixedBatchTests(unittest.TestCase):
    def test_open_ecology_batching_is_opt_in_and_same_contract_repeats_exactly(
        self,
    ) -> None:
        learner_seed = OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"][0]
        genome_stream_seed = OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_genome_stream"][0]
        signal_treatment = OpenEcologySignalTreatment()
        model_config = RecurrentActorCriticConfig.for_signal_config(
            signal_treatment.as_signal_config(),
            encoder_size=8,
            hidden_size=8,
            genome_conditioning_mode=RECURRENT_GENOME_CONDITIONING_ACTOR_FILM_V1,
        )
        model = PublicRecurrentActorCritic(
            model_config,
            initialization_seed=learner_seed,
        )
        tasks = build_open_ecology_training_schedule(
            training_phase=OPEN_ECOLOGY_PHASE_A,
            update_count=1,
            worlds_per_update=2,
            rollout_ticks=1,
            learner_seed=learner_seed,
            genome_stream_seed=genome_stream_seed,
            genome_population_mode="zero_all",
            environment_seed_offset=4,
        )[0]

        scalar_buffer, scalar_diagnostics = collect_recurrent_rollout_batch(
            model,
            tasks,
        )
        first_buffer, first_diagnostics = collect_recurrent_rollout_batch(
            model,
            tasks,
            fixed_batch_capacity=OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
        )
        second_buffer, second_diagnostics = collect_recurrent_rollout_batch(
            model,
            tasks,
            fixed_batch_capacity=OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
        )
        parallel_buffer, parallel_diagnostics = collect_recurrent_rollout_batch(
            model,
            tasks,
            rollout_workers=2,
            fixed_batch_capacity=OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
        )

        world_id = tasks[0].task_id
        self.assertNotIn(
            "fixed_batch_runtime",
            scalar_buffer.world_seed_provenance[world_id],
        )
        self.assertTrue(
            all(step.fixed_batch_runtime_sha256 is None for step in scalar_buffer.steps)
        )
        runtime = first_buffer.world_seed_provenance[world_id]["fixed_batch_runtime"]
        self.assertEqual(
            runtime["batch_capacity"],
            OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
        )
        self.assertEqual(first_buffer.steps, second_buffer.steps)
        self.assertEqual(first_diagnostics, second_diagnostics)
        self.assertEqual(first_buffer.steps, parallel_buffer.steps)
        self.assertEqual(first_diagnostics, parallel_diagnostics)
        self.assertEqual(
            [
                (step.requested_action, step.resolved_action)
                for step in scalar_buffer.steps
            ],
            [
                (step.requested_action, step.resolved_action)
                for step in first_buffer.steps
            ],
        )
        self.assertEqual(
            scalar_diagnostics.world_summaries[0]["summary"],
            first_diagnostics.world_summaries[0]["summary"],
        )
        ppo_config = RecurrentPPOConfig(
            learner_seed=learner_seed,
            update_epochs=1,
            sequence_minibatch_size=128,
            tbptt_steps=1,
            burn_in_steps=0,
        )
        scalar_result = RecurrentExperimentRunner(
            learner_seed=learner_seed,
            device="cpu",
            model_config=model_config,
            ppo_config=ppo_config,
        ).run((tasks,))
        fixed_result = RecurrentExperimentRunner(
            learner_seed=learner_seed,
            device="cpu",
            model_config=model_config,
            ppo_config=ppo_config,
            fixed_batch_capacity=OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
        ).run((tasks,))
        scalar_execution = scalar_result.rollout_execution["open_ecology"]
        fixed_execution = fixed_result.rollout_execution["open_ecology"]
        self.assertEqual(
            scalar_result.contract_version,
            RECURRENT_EXPERIMENT_CONTRACT_VERSION,
        )
        self.assertEqual(
            scalar_result.contract_version,
            "mind_public_recurrent_ippo_experiment_v5",
        )
        self.assertEqual(
            fixed_result.contract_version,
            RECURRENT_FIXED_BATCH_EXPERIMENT_CONTRACT_VERSION,
        )
        self.assertFalse(scalar_execution["fixed_batch_enabled"])
        self.assertFalse(scalar_execution["fixed_batch_default_enabled"])
        self.assertEqual(
            scalar_execution["fixed_batch_release_status"],
            RECURRENT_FIXED_BATCH_RELEASE_STATUS,
        )
        self.assertIsNone(scalar_execution["fixed_batch_runtime_contract"])
        self.assertIsNone(scalar_execution["fixed_batch_execution_scope"])
        self.assertFalse(
            scalar_execution["fixed_batch_authoritative_launch_gate_satisfied"]
        )
        self.assertTrue(fixed_execution["fixed_batch_enabled"])
        self.assertFalse(fixed_execution["fixed_batch_default_enabled"])
        self.assertEqual(
            fixed_execution["fixed_batch_release_status"],
            RECURRENT_FIXED_BATCH_RELEASE_STATUS,
        )
        self.assertEqual(
            fixed_execution["fixed_batch_runtime_contract"]["batch_capacity"],
            OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
        )
        self.assertEqual(
            fixed_execution["fixed_batch_execution_scope"],
            (
                "experimental_opt_in_intra_world_cpu_worker_only_"
                "cross_world_gpu_batching_excluded_v1"
            ),
        )
        self.assertFalse(
            fixed_execution["fixed_batch_authoritative_launch_gate_satisfied"]
        )

    def test_fixed_batch_matches_scalar_semantics_and_repeats_exactly(self) -> None:
        scalar_core = _ExactBatchCore()
        scalar = self._collect(
            core=scalar_core,
            world_id="scalar-reference",
            fixed=False,
            seed=17,
            initial_agents=4,
            rollout_ticks=2,
        )
        first_core = _ExactBatchCore()
        first = self._collect(
            core=first_core,
            world_id="fixed-first",
            fixed=True,
            seed=17,
            initial_agents=4,
            rollout_ticks=2,
        )
        second_core = _ExactBatchCore()
        second = self._collect(
            core=second_core,
            world_id="fixed-second",
            fixed=True,
            seed=17,
            initial_agents=4,
            rollout_ticks=2,
        )

        self.assertEqual(
            self._semantic_steps(scalar["steps"]),
            self._semantic_steps(first["steps"]),
        )
        self.assertEqual(
            self._semantic_steps(first["steps"]),
            self._semantic_steps(second["steps"]),
        )
        self.assertEqual(first["summary"], second["summary"])
        self.assertEqual(first_core.batch_calls, second_core.batch_calls)
        self.assertTrue(first_core.batch_calls)
        self.assertTrue(
            all(
                capacity == OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY
                and active_rows < capacity
                for active_rows, capacity, _conditioned in first_core.batch_calls
            )
        )
        self.assertEqual(first_core.scalar_calls, 0)

        provenance = first["provenance"]
        runtime = provenance["fixed_batch_runtime"]
        self.assertEqual(
            runtime["batch_capacity"],
            OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
        )
        self.assertFalse(runtime["contract"]["cross_world_batching"])
        self.assertEqual(len(runtime["exact_digest"]), 64)
        self.assertTrue(
            all(
                step.fixed_batch_runtime_sha256 == runtime["exact_digest"]
                and step.fixed_batch_capacity
                == OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY
                for step in first["steps"]
            )
        )
        self.assertTrue(
            all(
                trace["schema_version"]
                == RECURRENT_FIXED_BATCH_ROLLOUT_DIAGNOSTIC_SCHEMA_VERSION
                for trace in first["update_traces"]
            )
        )

    def test_earlier_attack_discards_later_staged_row_without_rng_or_hidden_commit(
        self,
    ) -> None:
        seed = next(
            candidate
            for candidate in range(1, 1_000)
            if deterministic_agent_turn_order((1, 2), seed=candidate, tick=0) == (1, 2)
        )
        scalar_core = _ExactBatchCore(preferred_action="attack_east")
        scalar = self._collect_attack_world(
            core=scalar_core,
            fixed=False,
            seed=seed,
            world_id="attack-scalar",
        )
        batch_core = _ExactBatchCore(preferred_action="attack_east")
        batched = self._collect_attack_world(
            core=batch_core,
            fixed=True,
            seed=seed,
            world_id="attack-batched",
        )

        self.assertEqual(
            self._semantic_steps(scalar["steps"]),
            self._semantic_steps(batched["steps"]),
        )
        self.assertFalse(batched["world"].agents[2].alive)
        self.assertNotIn(2, {step.agent_id for step in batched["steps"]})
        self.assertEqual(
            [step.decision_index for step in batched["steps"]],
            list(range(len(batched["steps"]))),
        )
        attacker_steps = [step for step in batched["steps"] if step.agent_id == 1]
        self.assertEqual(attacker_steps[0].hidden, (0.0, 0.0, 0.0))
        self.assertEqual(attacker_steps[1].hidden, (1.0, 1.0, 1.0))
        self.assertEqual(batch_core.batch_calls[0][0], 2)
        self.assertTrue(
            all(
                active_rows == 1
                for active_rows, _capacity, _ in batch_core.batch_calls[1:]
            )
        )
        passive_records = [
            record
            for record in batched["world"].trajectory_records
            if record["action_source"] == "passive"
        ]
        self.assertEqual(
            [(record["tick"], record["agent_id"]) for record in passive_records],
            [(0, 2)],
        )

    def test_conditioned_births_and_same_tick_child_death_preserve_batch_binding(
        self,
    ) -> None:
        surviving_core = _ConditionedExactBatchCore()
        surviving = self._collect_conditioned_birth_world(
            core=surviving_core,
            child_energy_fraction=0.3,
            world_id="fixed-surviving-child",
        )
        active_rows = [rows for rows, _capacity, _ in surviving_core.batch_calls]
        self.assertEqual(active_rows[0], 1)
        self.assertGreaterEqual(active_rows[1], 2)
        children = [
            agent
            for agent in surviving["world"].agents.values()
            if agent.parent_id == 1
        ]
        self.assertTrue(children)
        surviving_child = children[0]
        child_steps = [
            step
            for step in surviving["steps"]
            if step.agent_id == surviving_child.agent_id
        ]
        self.assertTrue(child_steps)
        self.assertEqual(child_steps[0].hidden, (0.0, 0.0, 0.0))
        self.assertEqual(
            child_steps[0].previous_feedback,
            PreviousPublicFeedback.zero(),
        )
        self.assertEqual(
            child_steps[0].genome_sha256,
            surviving_child.mind_inheritance_metadata["genome_sha256"],
        )
        self.assertTrue(
            all(
                step.fixed_batch_runtime_sha256
                == surviving["provenance"]["fixed_batch_runtime"]["exact_digest"]
                for step in surviving["steps"]
            )
        )
        self.assertTrue(
            all(
                trace["schema_version"]
                == RECURRENT_GENOME_FIXED_BATCH_ROLLOUT_DIAGNOSTIC_SCHEMA_VERSION
                for trace in surviving["update_traces"]
            )
        )

        dying_core = _ConditionedExactBatchCore()
        dying = self._collect_conditioned_birth_world(
            core=dying_core,
            child_energy_fraction=0.0,
            world_id="fixed-same-tick-child-death",
        )
        dead_children = [
            agent for agent in dying["world"].agents.values() if agent.parent_id == 1
        ]
        self.assertEqual(len(dead_children), 1)
        self.assertFalse(dead_children[0].alive)
        self.assertEqual(dead_children[0].birth_tick, dead_children[0].death_tick)
        self.assertNotIn(
            dead_children[0].agent_id,
            {step.agent_id for step in dying["steps"]},
        )
        self.assertTrue(all(rows == 1 for rows, _capacity, _ in dying_core.batch_calls))

    def test_capacity_overflow_fails_before_any_sampling(self) -> None:
        core = _ExactBatchCore()
        collector = RecurrentOnPolicyCollector(
            core,
            fixed_batch_contract=RecurrentFixedBatchRuntimeContract(batch_capacity=1),
        )
        collector.start_world(
            world_id="fixed-overflow",
            environment_seed=31,
            policy_sampling_seed=37,
            rollout_ticks=1,
        )
        world = SimulationWorld(
            self._world_config(
                seed=31,
                max_ticks=2,
                initial_agents=2,
            ),
            policy=collector,
        )
        with self.assertRaisesRegex(
            RecurrentRolloutError,
            "active rows exceed configured capacity",
        ):
            world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
        self.assertEqual(core.batch_calls, [])
        self.assertEqual(core.scalar_calls, 0)

    def test_non_hash_ordered_rows_fail_before_forward_or_sampling(self) -> None:
        core = _ExactBatchCore()
        collector = RecurrentOnPolicyCollector(
            core,
            fixed_batch_contract=RecurrentFixedBatchRuntimeContract.open_ecology(),
        )
        collector.start_world(
            world_id="fixed-order-drift",
            environment_seed=47,
            policy_sampling_seed=53,
            rollout_ticks=1,
        )
        world = SimulationWorld(
            self._world_config(
                seed=47,
                max_ticks=2,
                initial_agents=2,
            ),
            policy=collector,
        )
        expected_order = deterministic_agent_turn_order(
            tuple(world.agents),
            seed=world.config.seed,
            tick=world.tick,
        )
        observations = {
            agent_id: world._observe_agent(world.agents[agent_id])
            for agent_id in expected_order
        }

        with self.assertRaisesRegex(
            RecurrentRolloutError,
            "not in deterministic hash turn order",
        ):
            collector.stage_tick_start_batch(
                tick=world.tick,
                ordered_agent_ids=tuple(reversed(expected_order)),
                observations_by_agent=observations,
            )
        self.assertEqual(core.batch_calls, [])
        self.assertEqual(core.scalar_calls, 0)

    def test_torch_fixed_batch_is_same_contract_exact_and_scalar_close(self) -> None:
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(
                encoder_size=8,
                hidden_size=8,
                recurrent_layers=1,
            ),
            initialization_seed=101,
        )
        core = TorchRecurrentPolicyCore(model)
        observations = (
            tuple(0.01 for _ in range(ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE)),
            tuple(0.02 for _ in range(ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE)),
        )
        masks = tuple(
            tuple(
                action in {"stay", "move_north", "signal_0_profile_0"}
                for action in ACTION_NAMES
            )
            for _ in observations
        )
        feedback = (
            PreviousPublicFeedback.zero(),
            PreviousPublicFeedback.zero(),
        )
        hidden = (core.initial_hidden(), core.initial_hidden())
        first = core.forward_fixed_batch(
            observations,
            masks,
            feedback,
            hidden,
            batch_capacity=OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
        )
        second = core.forward_fixed_batch(
            observations,
            masks,
            feedback,
            hidden,
            batch_capacity=OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
        )
        self.assertEqual(first, second)
        scalar = tuple(
            core.forward_step(observation, mask, previous, state)
            for observation, mask, previous, state in zip(
                observations,
                masks,
                feedback,
                hidden,
                strict=True,
            )
        )
        for batch_row, scalar_row in zip(first, scalar, strict=True):
            torch.testing.assert_close(
                torch.tensor(batch_row.logits),
                torch.tensor(scalar_row.logits),
                rtol=1e-5,
                atol=1e-6,
            )
            torch.testing.assert_close(
                torch.tensor(batch_row.next_hidden),
                torch.tensor(scalar_row.next_hidden),
                rtol=1e-5,
                atol=1e-6,
            )
            self.assertAlmostEqual(batch_row.value, scalar_row.value, places=6)

    def test_production_width_phase_b_density_128_is_exact_across_workers(
        self,
    ) -> None:
        learner_seed = OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"][0]
        genome_stream_seed = OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_genome_stream"][0]
        signal_treatment = OpenEcologySignalTreatment()
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig.for_signal_config(
                signal_treatment.as_signal_config(),
                encoder_size=256,
                hidden_size=256,
                genome_conditioning_mode=(RECURRENT_GENOME_CONDITIONING_ACTOR_FILM_V1),
            ),
            initialization_seed=learner_seed,
        )
        schedule = build_open_ecology_training_schedule(
            training_phase=OPEN_ECOLOGY_PHASE_B,
            update_count=1,
            worlds_per_update=7,
            rollout_ticks=1,
            learner_seed=learner_seed,
            genome_stream_seed=genome_stream_seed,
            genome_population_mode="zero_all",
        )
        tasks = (schedule[0][2], schedule[0][6])
        self.assertEqual(
            tuple(task.open_ecology_treatment.initial_agents for task in tasks),
            (128, 128),
        )

        one_worker_buffer, one_worker_diagnostics = collect_recurrent_rollout_batch(
            model,
            tasks,
            rollout_workers=1,
            fixed_batch_capacity=OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
        )
        two_worker_buffer, two_worker_diagnostics = collect_recurrent_rollout_batch(
            model,
            tasks,
            rollout_workers=2,
            fixed_batch_capacity=OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
        )

        self.assertEqual(one_worker_buffer.steps, two_worker_buffer.steps)
        self.assertEqual(one_worker_diagnostics, two_worker_diagnostics)
        for provenance in one_worker_buffer.world_seed_provenance.values():
            runtime = provenance["fixed_batch_runtime"]
            observed = runtime["observed_runtime"]
            self.assertEqual(observed["torch_num_threads"], 1)
            self.assertEqual(observed["torch_num_interop_threads"], 1)
            self.assertTrue(observed["cpu_architecture"])
            self.assertTrue(observed["aten_parallel_backend"])
            self.assertEqual(len(observed["torch_build_config_sha256"]), 64)

    def test_requested_fixed_batch_rejects_scalar_worker_fallback(self) -> None:
        learner_seed = OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"][0]
        genome_stream_seed = OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_genome_stream"][0]
        signal_treatment = OpenEcologySignalTreatment()
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig.for_signal_config(
                signal_treatment.as_signal_config(),
                encoder_size=8,
                hidden_size=8,
                genome_conditioning_mode=(RECURRENT_GENOME_CONDITIONING_ACTOR_FILM_V1),
            ),
            initialization_seed=learner_seed,
        )
        task = build_open_ecology_training_schedule(
            training_phase=OPEN_ECOLOGY_PHASE_A,
            update_count=1,
            worlds_per_update=1,
            rollout_ticks=1,
            learner_seed=learner_seed,
            genome_stream_seed=genome_stream_seed,
            genome_population_mode="zero_all",
        )[0][0]
        steps, summary = _collect_recurrent_rollout_task(
            model,
            task,
            feed_forward_history_ablation=False,
            fixed_batch_contract=None,
        )

        with self.assertRaisesRegex(
            RecurrentExperimentError,
            "omitted its runtime provenance",
        ):
            _validate_rollout_worker_result(
                task=task,
                summary=summary,
                steps=steps,
                genome_conditioning_mode=model.config.genome_conditioning_mode,
                expected_fixed_batch_contract=(
                    RecurrentFixedBatchRuntimeContract.open_ecology()
                ),
            )

    def test_runtime_identity_changes_with_effective_torch_thread_count(self) -> None:
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(
                encoder_size=8,
                hidden_size=8,
            ),
            initialization_seed=109,
        )
        core = TorchRecurrentPolicyCore(model)
        original_threads = torch.get_num_threads()
        alternate_threads = 2 if original_threads != 2 else 1
        try:
            torch.set_num_threads(1)
            one_thread = _fixed_batch_runtime_binding(
                RecurrentFixedBatchRuntimeContract.open_ecology(),
                core=core,
            )
            torch.set_num_threads(alternate_threads)
            alternate = _fixed_batch_runtime_binding(
                RecurrentFixedBatchRuntimeContract.open_ecology(),
                core=core,
            )
        finally:
            torch.set_num_threads(original_threads)

        self.assertEqual(
            one_thread["observed_runtime"]["torch_num_threads"],
            1,
        )
        self.assertEqual(
            alternate["observed_runtime"]["torch_num_threads"],
            alternate_threads,
        )
        self.assertNotEqual(one_thread["exact_digest"], alternate["exact_digest"])

    def test_fixed_disabled_legacy_run_retains_v5_contract(self) -> None:
        learner_seed = 113
        schedule = build_recurrent_training_schedule(
            update_count=1,
            worlds_per_update=1,
            rollout_ticks=1,
            scenarios=("broad",),
        )
        result = RecurrentExperimentRunner(
            learner_seed=learner_seed,
            device="cpu",
            model_config=RecurrentActorCriticConfig(
                encoder_size=8,
                hidden_size=8,
            ),
            ppo_config=RecurrentPPOConfig(
                learner_seed=learner_seed,
                update_epochs=1,
                sequence_minibatch_size=128,
                tbptt_steps=1,
                burn_in_steps=0,
            ),
        ).run(schedule)

        self.assertEqual(
            result.contract_version,
            "mind_public_recurrent_ippo_experiment_v5",
        )
        self.assertEqual(
            result.contract_version,
            RECURRENT_EXPERIMENT_CONTRACT_VERSION,
        )
        self.assertNotIn("open_ecology", result.rollout_execution)

    def _collect(
        self,
        *,
        core: _ExactBatchCore,
        world_id: str,
        fixed: bool,
        seed: int,
        initial_agents: int,
        rollout_ticks: int,
    ) -> dict[str, object]:
        collector = RecurrentOnPolicyCollector(
            core,
            fixed_batch_contract=(
                RecurrentFixedBatchRuntimeContract.open_ecology() if fixed else None
            ),
        )
        collector.start_world(
            world_id=world_id,
            environment_seed=seed,
            policy_sampling_seed=919,
            rollout_ticks=rollout_ticks,
        )
        world = SimulationWorld(
            self._world_config(
                seed=seed,
                max_ticks=rollout_ticks + 1,
                initial_agents=initial_agents,
            ),
            policy=collector,
        )
        result = world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
        collector.finish_world()
        return {
            "steps": collector.buffer.steps,
            "summary": result.summary,
            "provenance": collector.buffer.world_seed_provenance[world_id],
            "update_traces": tuple(world.policy_update_trace_records),
            "world": world,
        }

    def _collect_attack_world(
        self,
        *,
        core: _ExactBatchCore,
        fixed: bool,
        seed: int,
        world_id: str,
    ) -> dict[str, object]:
        collector = RecurrentOnPolicyCollector(
            core,
            fixed_batch_contract=(
                RecurrentFixedBatchRuntimeContract.open_ecology() if fixed else None
            ),
        )
        collector.start_world(
            world_id=world_id,
            environment_seed=seed,
            policy_sampling_seed=991,
            rollout_ticks=2,
        )
        world = SimulationWorld(
            self._world_config(
                seed=seed,
                max_ticks=3,
                initial_agents=2,
                combat=CombatConfig(
                    min_attack_health_ratio=0.0,
                    min_attack_energy_ratio=0.0,
                    min_attack_hydration_ratio=0.0,
                    min_reproduction_health_ratio=1.0,
                    base_attack_damage=100.0,
                    hunter_mode_attack_damage_multiplier=1.0,
                    hunter_wounded_prey_damage_bonus=0.0,
                    attack_energy_cost=0.0,
                    attack_hydration_cost=0.0,
                ),
            ),
            policy=collector,
        )
        attacker = world.agents[1]
        victim = world.agents[2]
        for row in world.grid:
            for tile in row:
                if tile.occupant_id in {attacker.agent_id, victim.agent_id}:
                    tile.occupant_id = None
        attacker.x, attacker.y = 1, 1
        victim.x, victim.y = 2, 1
        world.grid[1][1].occupant_id = attacker.agent_id
        world.grid[1][2].occupant_id = victim.agent_id
        attacker.energy = attacker.genome.max_energy
        attacker.hydration = attacker.genome.max_hydration
        attacker.health = attacker.max_health
        victim.health = 0.01
        world.reset_derived_caches()
        world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
        collector.finish_world()
        return {
            "steps": collector.buffer.steps,
            "world": world,
        }

    def _collect_conditioned_birth_world(
        self,
        *,
        core: _ConditionedExactBatchCore,
        child_energy_fraction: float,
        world_id: str,
    ) -> dict[str, object]:
        collector = RecurrentOnPolicyCollector(
            core,
            fixed_batch_contract=RecurrentFixedBatchRuntimeContract.open_ecology(),
        )
        collector.start_world(
            world_id=world_id,
            environment_seed=43,
            policy_sampling_seed=997,
            rollout_ticks=2,
            genome_stream_seed=1009,
            genome_population_mode="heritable",
        )
        world = SimulationWorld(
            self._world_config(
                seed=43,
                max_ticks=3,
                initial_agents=1,
                reproduction=ReproductionConfig(
                    min_age=1,
                    cooldown_ticks=1_000,
                    min_hydration_fraction=0.0,
                    energy_cost=0.0,
                    child_energy_fraction=child_energy_fraction,
                ),
            ),
            policy=collector,
        )
        founder = world.agents[1]
        founder.age = 10
        founder.energy = founder.genome.max_energy * 1.25
        founder.hydration = founder.genome.max_hydration
        founder.health = founder.max_health
        founder.last_reproduction_tick = -10_000
        result = world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
        collector.finish_world()
        return {
            "steps": collector.buffer.steps,
            "summary": result.summary,
            "provenance": collector.buffer.world_seed_provenance[world_id],
            "update_traces": tuple(world.policy_update_trace_records),
            "world": world,
        }

    @staticmethod
    def _semantic_steps(
        raw_steps: object,
    ) -> tuple[tuple[tuple[str, object], ...], ...]:
        steps = raw_steps
        if not isinstance(steps, tuple):
            raise AssertionError("test helper expected tuple rollout steps")
        excluded = {
            "world_id",
            "fixed_batch_runtime_sha256",
            "fixed_batch_turn_rank",
            "fixed_batch_active_rows",
            "fixed_batch_capacity",
        }
        return tuple(
            tuple(
                (field.name, getattr(step, field.name))
                for field in dataclasses.fields(RecurrentRolloutStep)
                if field.name not in excluded
            )
            for step in steps
        )

    @staticmethod
    def _world_config(
        *,
        seed: int,
        max_ticks: int,
        initial_agents: int,
        combat: CombatConfig | None = None,
        reproduction: ReproductionConfig | None = None,
    ) -> WorldConfig:
        return WorldConfig(
            seed=seed,
            max_ticks=max_ticks,
            width=7,
            height=7,
            initial_agents=initial_agents,
            max_agents=20,
            water_tile_ratio=0.0,
            forest_tile_ratio=0.0,
            wetland_tile_ratio=0.0,
            rocky_tile_ratio=0.0,
            base_energy_drain=0.0,
            base_hydration_drain=0.0,
            combat=combat or CombatConfig(),
            reproduction=reproduction
            or ReproductionConfig(
                min_age=500,
                cooldown_ticks=500,
                min_hydration_fraction=1.0,
                energy_cost=0.0,
                child_energy_fraction=0.3,
            ),
            diet_matching=DietMatchingConfig(
                specialist_threshold=0.0,
                omnivore_threshold=0.0,
            ),
            trophic=TrophicConfig(attack_channel_threshold=0.0),
        )


if __name__ == "__main__":
    unittest.main()
