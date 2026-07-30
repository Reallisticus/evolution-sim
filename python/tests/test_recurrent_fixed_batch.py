from __future__ import annotations

import copy
import dataclasses
import sys
import unittest
from unittest import mock

import torch

from evolution_sim.config import (
    CombatConfig,
    DietMatchingConfig,
    ReproductionConfig,
    TrophicConfig,
    WorldConfig,
)
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import encode_observation_input
from evolution_sim.env.runtime.state import RunMode
from evolution_sim.env.runtime.ticks import (
    _RUNTIME_OWNED_POLICY_TICK_START_REGISTRY,
    _runtime_owned_policy_tick_start,
    deterministic_agent_turn_order,
)
from evolution_sim.env.world import SimulationWorld
from evolution_sim.mind import recurrent_rollout
from evolution_sim.mind.policy_inputs import (
    ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE,
    ecological_policy_input_values,
    ecological_policy_values_from_observation,
)
from evolution_sim.mind.open_ecology_seed_registry import (
    OPEN_ECOLOGY_SEED_REGISTRY,
)
from evolution_sim.mind.recurrent_actor_critic import (
    CRITIC_GENOME_CONDITIONING_FILM_V1,
    CRITIC_GENOME_CONDITIONING_NONE,
    GENOME_CONDITIONING_DISABLED,
    PublicRecurrentActorCritic,
    RecurrentActorCriticConfig,
    SequenceEvaluation,
)
from evolution_sim.mind.recurrent_genome import RECURRENT_CONTROLLER_GENOME_SIZE
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
    RECURRENT_FIXED_BATCH_EXECUTION_BUCKETS,
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
    _validated_fixed_batch_runtime_binding,
)


REQUIRES_MIND_ML = True


class _ExactBatchCore:
    hidden_size = 3

    def __init__(self, *, preferred_action: str | None = None) -> None:
        self.preferred_action = preferred_action
        self.batch_calls: list[tuple[int, int, int, bool]] = []
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
        execution_batch_rows: int | None = None,
        genome_values: tuple[tuple[float, ...], ...] | None = None,
    ) -> tuple[RecurrentCoreOutput, ...]:
        if execution_batch_rows is None:
            raise AssertionError("collector must bind the physical execution bucket")
        self.batch_calls.append(
            (
                len(observations),
                batch_capacity,
                execution_batch_rows,
                genome_values is not None,
            )
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
        execution_batch_rows: int | None = None,
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
            execution_batch_rows=execution_batch_rows,
            genome_values=genome_values,
        )


class RecurrentFixedBatchTests(unittest.TestCase):
    @staticmethod
    def _real_core_numeric_inputs(
        core: TorchRecurrentPolicyCore,
        *,
        active_rows: int,
        genome_conditioned: bool,
    ) -> tuple[
        tuple[tuple[float, ...], ...],
        tuple[tuple[bool, ...], ...],
        tuple[PreviousPublicFeedback, ...],
        tuple[tuple[float, ...], ...],
        tuple[tuple[float, ...], ...] | None,
        torch.Tensor,
    ]:
        observations = tuple(
            tuple(
                float(((row * 7) + (column * 3)) % 17) / 100.0
                for column in range(core.public_input_size)
            )
            for row in range(active_rows)
        )
        stay_index = ACTION_NAMES.index("stay")
        selected_action_indices: list[int] = []
        action_masks: list[tuple[bool, ...]] = []
        for row in range(active_rows):
            selected_action = (row * 11 + 3) % len(ACTION_NAMES)
            selected_action_indices.append(selected_action)
            action_masks.append(
                tuple(
                    index in {stay_index, selected_action}
                    for index in range(len(ACTION_NAMES))
                )
            )
        feedback = tuple(
            PreviousPublicFeedback.zero() for _ in range(active_rows)
        )
        hidden = tuple(
            tuple(
                float(((row * 5) + index) % 13) / 1000.0
                for index in range(core.hidden_size)
            )
            for row in range(active_rows)
        )
        genomes = (
            tuple(
                tuple(
                    float(((row * 3) + locus) % 9 - 4) / 10.0
                    for locus in range(RECURRENT_CONTROLLER_GENOME_SIZE)
                )
                for row in range(active_rows)
            )
            if genome_conditioned
            else None
        )
        actions = torch.tensor(
            (selected_action_indices,),
            dtype=torch.long,
        )
        return (
            observations,
            tuple(action_masks),
            feedback,
            hidden,
            genomes,
            actions,
        )

    @staticmethod
    def _direct_sequence_evaluation(
        model: PublicRecurrentActorCritic,
        *,
        observations: tuple[tuple[float, ...], ...],
        action_masks: tuple[tuple[bool, ...], ...],
        feedback: tuple[PreviousPublicFeedback, ...],
        hidden: tuple[tuple[float, ...], ...],
        genomes: tuple[tuple[float, ...], ...] | None,
        actions: torch.Tensor,
    ) -> SequenceEvaluation:
        reference = next(model.parameters())
        active_rows = len(observations)
        observation_tensor = torch.tensor(
            (observations,),
            device=reference.device,
            dtype=reference.dtype,
        )
        action_mask_tensor = torch.tensor(
            (action_masks,),
            device=reference.device,
            dtype=torch.bool,
        )
        feedback_tensor = torch.tensor(
            (tuple(row.vector() for row in feedback),),
            device=reference.device,
            dtype=reference.dtype,
        )
        hidden_tensor = torch.tensor(
            hidden,
            device=reference.device,
            dtype=reference.dtype,
        ).reshape(
            active_rows,
            model.config.recurrent_layers,
            model.config.hidden_size,
        ).permute(1, 0, 2)
        genome_tensor = (
            None
            if genomes is None
            else torch.tensor(
                (genomes,),
                device=reference.device,
                dtype=reference.dtype,
            )
        )
        with torch.no_grad():
            return model.evaluate_sequence(
                observation_tensor,
                action_mask_tensor,
                feedback_tensor,
                actions.to(device=reference.device),
                genome_values=genome_tensor,
                initial_state=hidden_tensor,
            )

    def _assert_core_outputs_match_direct_evaluation(
        self,
        outputs: tuple[RecurrentCoreOutput, ...],
        *,
        action_masks: tuple[tuple[bool, ...], ...],
        actions: torch.Tensor,
        direct: SequenceEvaluation,
    ) -> None:
        active_rows = len(outputs)
        raw_logits = torch.tensor(
            tuple(output.logits for output in outputs),
            dtype=torch.float32,
        )
        mask = torch.tensor(action_masks, dtype=torch.bool)
        masked_logits = raw_logits.masked_fill(~mask, -torch.inf)
        distribution = torch.distributions.Categorical(logits=masked_logits)
        values = torch.tensor(
            tuple(output.value for output in outputs),
            dtype=torch.float32,
        )
        hidden = torch.tensor(
            tuple(output.next_hidden for output in outputs),
            dtype=torch.float32,
        ).reshape(
            len(outputs),
            direct.final_state.shape[0],
            direct.final_state.shape[2],
        ).permute(1, 0, 2)
        torch.testing.assert_close(
            raw_logits,
            direct.raw_logits[0, :active_rows].cpu(),
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            masked_logits,
            direct.masked_logits[0, :active_rows].cpu(),
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            distribution.probs,
            torch.distributions.Categorical(
                logits=direct.masked_logits[0, :active_rows].cpu()
            ).probs,
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            distribution.log_prob(actions[0]),
            direct.log_probs[0, :active_rows].cpu(),
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            distribution.entropy(),
            direct.entropy[0, :active_rows].cpu(),
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            values,
            direct.values[0, :active_rows].cpu(),
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            hidden,
            direct.final_state[:, :active_rows].cpu(),
            rtol=1e-5,
            atol=1e-6,
        )

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
                and active_rows <= execution_rows <= capacity
                and execution_rows
                == next(
                    bucket
                    for bucket in RECURRENT_FIXED_BATCH_EXECUTION_BUCKETS
                    if bucket >= active_rows
                )
                for (
                    active_rows,
                    capacity,
                    execution_rows,
                    _conditioned,
                ) in first_core.batch_calls
            )
        )
        self.assertEqual(first_core.scalar_calls, 0)

        provenance = first["provenance"]
        runtime = provenance["fixed_batch_runtime"]
        self.assertEqual(
            runtime["batch_capacity"],
            OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
        )
        self.assertEqual(
            runtime["execution_batch_buckets"],
            list(RECURRENT_FIXED_BATCH_EXECUTION_BUCKETS),
        )
        self.assertEqual(
            runtime["contract"]["execution_batch_buckets"],
            list(RECURRENT_FIXED_BATCH_EXECUTION_BUCKETS),
        )
        self.assertFalse(runtime["contract"]["cross_world_batching"])
        self.assertEqual(len(runtime["exact_digest"]), 64)
        self.assertTrue(
            all(
                step.fixed_batch_runtime_sha256 == runtime["exact_digest"]
                and step.fixed_batch_capacity
                == OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY
                and step.fixed_batch_execution_rows
                == next(
                    bucket
                    for bucket in RECURRENT_FIXED_BATCH_EXECUTION_BUCKETS
                    if bucket >= int(step.fixed_batch_active_rows)
                )
                for step in first["steps"]
            )
        )
        with self.assertRaisesRegex(
            RecurrentRolloutError,
            "fixed-batch row bounds are invalid",
        ):
            dataclasses.replace(
                first["steps"][0],
                fixed_batch_execution_rows=8,
            )
        self.assertTrue(
            all(
                trace["schema_version"]
                == RECURRENT_FIXED_BATCH_ROLLOUT_DIAGNOSTIC_SCHEMA_VERSION
                and trace["fixed_batch_execution_rows"]
                in RECURRENT_FIXED_BATCH_EXECUTION_BUCKETS
                for trace in first["update_traces"]
            )
        )
    def test_capacity_and_execution_rows_are_bound_to_frozen_buckets(self) -> None:
        self.assertEqual(
            RECURRENT_FIXED_BATCH_EXECUTION_BUCKETS,
            (1, 2, 4, 8, 16, 32, 64, 128, 256, 320),
        )
        with self.assertRaisesRegex(
            RecurrentRolloutError,
            "capacity must be a frozen execution bucket",
        ):
            RecurrentFixedBatchRuntimeContract(batch_capacity=3)
        runtime = _fixed_batch_runtime_binding(
            RecurrentFixedBatchRuntimeContract.open_ecology(),
            core=_ExactBatchCore(),
        )
        tampered_runtime = copy.deepcopy(runtime)
        tampered_runtime["execution_batch_buckets"] = [1, 2, 4, 320]
        with self.assertRaisesRegex(
            RecurrentRolloutError,
            "execution buckets drifted",
        ):
            _validated_fixed_batch_runtime_binding(tampered_runtime)

        core = _ExactBatchCore()
        result = self._collect(
            core=core,
            world_id="fixed-bucket-round-up",
            fixed=True,
            seed=23,
            initial_agents=3,
            rollout_ticks=1,
        )

        self.assertTrue(core.batch_calls)
        self.assertEqual(core.batch_calls[0], (3, 320, 4, False))
        self.assertTrue(
            all(step.fixed_batch_execution_rows == 4 for step in result["steps"])
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
                for (
                    active_rows,
                    _capacity,
                    _execution_rows,
                    _conditioned,
                ) in batch_core.batch_calls[1:]
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
        active_rows = [
            rows
            for rows, _capacity, _execution_rows, _ in surviving_core.batch_calls
        ]
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
        self.assertTrue(
            all(
                rows == 1
                for rows, _capacity, _execution_rows, _ in dying_core.batch_calls
            )
        )

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

    def test_runtime_owned_tick_start_skips_only_redundant_generic_guards(
        self,
    ) -> None:
        core = _ExactBatchCore()
        collector = RecurrentOnPolicyCollector(
            core,
            fixed_batch_contract=RecurrentFixedBatchRuntimeContract.open_ecology(),
        )
        collector.start_world(
            world_id="fixed-runtime-owned-fast-path",
            environment_seed=59,
            policy_sampling_seed=61,
            rollout_ticks=1,
        )
        world = SimulationWorld(
            self._world_config(
                seed=59,
                max_ticks=1,
                initial_agents=4,
            ),
            policy=collector,
        )

        with (
            mock.patch(
                "evolution_sim.mind.recurrent_rollout."
                "deterministic_agent_turn_order",
                side_effect=AssertionError("runtime order was recomputed"),
            ),
            mock.patch.object(
                collector,
                "stage_tick_start_batch",
                side_effect=AssertionError("generic staging path was called"),
            ),
        ):
            world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)

        self.assertGreater(len(core.batch_calls), 0)
        self.assertIsNone(collector._staged_tick)
        self.assertEqual(collector._staged_by_agent, {})

    def test_runtime_owned_tick_start_witness_and_identity_fail_closed(
        self,
    ) -> None:
        core = _ExactBatchCore()
        collector = RecurrentOnPolicyCollector(
            core,
            fixed_batch_contract=RecurrentFixedBatchRuntimeContract.open_ecology(),
        )
        collector.start_world(
            world_id="fixed-runtime-owned-witness",
            environment_seed=63,
            policy_sampling_seed=67,
            rollout_ticks=1,
        )
        world = SimulationWorld(
            self._world_config(
                seed=63,
                max_ticks=1,
                initial_agents=2,
            ),
            policy=collector,
        )
        ordered_agent_ids = deterministic_agent_turn_order(
            tuple(world.agents),
            seed=world.config.seed,
            tick=world.tick,
        )
        observations = {
            agent_id: world._observe_agent(world.agents[agent_id])
            for agent_id in ordered_agent_ids
        }
        witness = _runtime_owned_policy_tick_start(
            environment_seed=world.config.seed,
            tick=world.tick,
            ordered_agent_ids=ordered_agent_ids,
            observation_snapshots=observations,
        )
        rng_state_before = collector._rng.getstate()
        invalid_witnesses = (
            dataclasses.replace(witness, environment_seed=999),
            dataclasses.replace(witness, tick=1),
            dataclasses.replace(witness, _authority=object()),
            dataclasses.replace(
                witness,
                ordered_agent_ids=tuple(reversed(ordered_agent_ids)),
            ),
            dataclasses.replace(
                witness,
                observation_snapshots=copy.deepcopy(observations),
            ),
        )
        for invalid in invalid_witnesses:
            with self.assertRaisesRegex(
                RecurrentRolloutError,
                "tick-start witness is invalid",
            ):
                collector._stage_runtime_owned_tick_start_batch(invalid)
            self.assertEqual(core.batch_calls, [])
            self.assertEqual(collector._rng.getstate(), rng_state_before)
            self.assertIsNone(collector._staged_tick)
            self.assertEqual(collector._staged_by_agent, {})

        row_replacement_witness = _runtime_owned_policy_tick_start(
            environment_seed=world.config.seed,
            tick=world.tick,
            ordered_agent_ids=ordered_agent_ids,
            observation_snapshots=observations,
        )
        replaced_agent_id = ordered_agent_ids[-1]
        original_row = observations[replaced_agent_id]
        observations[replaced_agent_id] = copy.deepcopy(original_row)
        with self.assertRaisesRegex(
            RecurrentRolloutError,
            "tick-start witness is invalid",
        ):
            collector._stage_runtime_owned_tick_start_batch(
                row_replacement_witness
            )
        observations[replaced_agent_id] = original_row
        self.assertEqual(core.batch_calls, [])
        self.assertEqual(collector._rng.getstate(), rng_state_before)

        in_place_mutation_witness = _runtime_owned_policy_tick_start(
            environment_seed=world.config.seed,
            tick=world.tick,
            ordered_agent_ids=ordered_agent_ids,
            observation_snapshots=observations,
        )
        first_agent_id = ordered_agent_ids[0]
        self_state = observations[first_agent_id]["self"]
        assert isinstance(self_state, dict)
        original_energy_ratio = self_state["energy_ratio"]
        self_state["energy_ratio"] = 0.123456
        with self.assertRaisesRegex(
            RecurrentRolloutError,
            "tick-start witness is invalid",
        ):
            collector._stage_runtime_owned_tick_start_batch(
                in_place_mutation_witness
            )
        self_state["energy_ratio"] = original_energy_ratio
        self.assertEqual(core.batch_calls, [])
        self.assertEqual(collector._rng.getstate(), rng_state_before)

        collector._stage_runtime_owned_tick_start_batch(witness)
        staged_before = dict(collector._staged_by_agent)
        with self.assertRaisesRegex(
            RecurrentRolloutError,
            "tick-start witness is invalid",
        ):
            collector._stage_runtime_owned_tick_start_batch(witness)
        self.assertEqual(collector._staged_by_agent, staged_before)

        replacement = copy.deepcopy(observations[first_agent_id])
        with self.assertRaisesRegex(
            RecurrentRolloutError,
            "staged observation drifted before sampling",
        ):
            collector.decide(
                replacement,
                dict(replacement["action_mask"]),
            )
        self.assertEqual(collector._rng.getstate(), rng_state_before)
        self.assertEqual(collector._staged_by_agent, staged_before)

        decision = collector.decide(
            observations[first_agent_id],
            dict(observations[first_agent_id]["action_mask"]),
        )
        self.assertEqual(decision.source, "learned_recurrent_on_policy")

        second_agent_id = ordered_agent_ids[1]
        second_self_state = observations[second_agent_id]["self"]
        assert isinstance(second_self_state, dict)
        original_hydration_ratio = second_self_state["hydration_ratio"]
        second_self_state["hydration_ratio"] = 0.654321
        with self.assertRaisesRegex(
            RecurrentRolloutError,
            "staged observation drifted before sampling",
        ):
            collector.decide(
                observations[second_agent_id],
                dict(observations[second_agent_id]["action_mask"]),
            )
        second_self_state["hydration_ratio"] = original_hydration_ratio
        second_decision = collector.decide(
            observations[second_agent_id],
            dict(observations[second_agent_id]["action_mask"]),
        )
        self.assertEqual(
            second_decision.source,
            "learned_recurrent_on_policy",
        )

    def test_runtime_owned_witness_registry_is_retired_on_private_hook_failure(
        self,
    ) -> None:
        for label, side_effect in (
            ("non-consuming", None),
            ("throwing", RuntimeError("hostile private hook")),
        ):
            with self.subTest(label=label):
                expected_error = (
                    RecurrentRolloutError
                    if side_effect is None
                    else RuntimeError
                )
                collector = RecurrentOnPolicyCollector(
                    _ExactBatchCore(),
                    fixed_batch_contract=(
                        RecurrentFixedBatchRuntimeContract.open_ecology()
                    ),
                )
                collector.start_world(
                    world_id=f"fixed-runtime-owned-cleanup-{label}",
                    environment_seed=71,
                    policy_sampling_seed=73,
                    rollout_ticks=1,
                )
                world = SimulationWorld(
                    self._world_config(
                        seed=71,
                        max_ticks=1,
                        initial_agents=2,
                    ),
                    policy=collector,
                )
                with (
                    mock.patch.object(
                        collector,
                        "_stage_runtime_owned_tick_start_batch",
                        side_effect=side_effect,
                    ),
                    self.assertRaises(expected_error),
                ):
                    world.run(
                        mode=RunMode.SUMMARY_ONLY,
                        record_trajectory=True,
                    )
                self.assertEqual(
                    _RUNTIME_OWNED_POLICY_TICK_START_REGISTRY,
                    {},
                )

    def test_fixed_batch_projects_each_staged_observation_once(self) -> None:
        core = _ExactBatchCore()
        with mock.patch(
            "evolution_sim.mind.recurrent_rollout."
            "ecological_policy_values_from_observation",
            wraps=ecological_policy_values_from_observation,
        ) as project:
            self._collect(
                core=core,
                world_id="fixed-single-encoding",
                fixed=True,
                seed=61,
                initial_agents=4,
                rollout_ticks=2,
            )

        staged_row_count = sum(
            active_rows
            for (
                active_rows,
                _capacity,
                _execution_rows,
                _conditioned,
            ) in core.batch_calls
        )
        self.assertGreater(staged_row_count, 0)
        self.assertEqual(project.call_count, staged_row_count)

    def test_scalar_serialized_projection_remains_fixed_batch_reference(
        self,
    ) -> None:
        scalar_core = _ExactBatchCore()
        with (
            mock.patch(
                "evolution_sim.mind.recurrent_rollout.encode_observation_input",
                wraps=encode_observation_input,
            ) as scalar_encode,
            mock.patch(
                "evolution_sim.mind.recurrent_rollout.ecological_policy_input_values",
                wraps=ecological_policy_input_values,
            ) as scalar_project,
            mock.patch(
                "evolution_sim.mind.recurrent_rollout."
                "ecological_policy_values_from_observation",
                wraps=ecological_policy_values_from_observation,
            ) as scalar_direct,
        ):
            self._collect(
                core=scalar_core,
                world_id="scalar-serialized-projection-reference",
                fixed=False,
                seed=73,
                initial_agents=4,
                rollout_ticks=2,
            )

        self.assertGreater(scalar_core.scalar_calls, 0)
        self.assertEqual(scalar_encode.call_count, scalar_core.scalar_calls)
        self.assertEqual(scalar_project.call_count, scalar_core.scalar_calls)
        self.assertEqual(scalar_direct.call_count, 0)

        fixed_core = _ExactBatchCore()
        with (
            mock.patch(
                "evolution_sim.mind.recurrent_rollout.encode_observation_input",
                wraps=encode_observation_input,
            ) as fixed_encode,
            mock.patch(
                "evolution_sim.mind.recurrent_rollout.ecological_policy_input_values",
                wraps=ecological_policy_input_values,
            ) as fixed_serialized,
            mock.patch(
                "evolution_sim.mind.recurrent_rollout."
                "ecological_policy_values_from_observation",
                wraps=ecological_policy_values_from_observation,
            ) as fixed_direct,
        ):
            self._collect(
                core=fixed_core,
                world_id="fixed-direct-projection-candidate",
                fixed=True,
                seed=73,
                initial_agents=4,
                rollout_ticks=2,
            )

        fixed_row_count = sum(
            active_rows
            for (
                active_rows,
                _capacity,
                _execution_rows,
                _conditioned,
            ) in fixed_core.batch_calls
        )
        self.assertGreater(fixed_row_count, 0)
        self.assertEqual(fixed_encode.call_count, 0)
        self.assertEqual(fixed_serialized.call_count, 0)
        self.assertEqual(fixed_direct.call_count, fixed_row_count)

    def test_fixed_batch_validates_staged_state_without_reencoding_or_partial_commit(
        self,
    ) -> None:
        core = _ExactBatchCore()
        collector = RecurrentOnPolicyCollector(
            core,
            fixed_batch_contract=RecurrentFixedBatchRuntimeContract.open_ecology(),
        )
        collector.start_world(
            world_id="fixed-observation-drift",
            environment_seed=67,
            policy_sampling_seed=71,
            rollout_ticks=1,
        )
        world = SimulationWorld(
            self._world_config(
                seed=67,
                max_ticks=2,
                initial_agents=2,
            ),
            policy=collector,
        )
        ordered_agent_ids = deterministic_agent_turn_order(
            tuple(world.agents),
            seed=world.config.seed,
            tick=world.tick,
        )
        observations = {
            agent_id: world._observe_agent(world.agents[agent_id])
            for agent_id in ordered_agent_ids
        }
        collector.stage_tick_start_batch(
            tick=world.tick,
            ordered_agent_ids=ordered_agent_ids,
            observations_by_agent=observations,
        )
        rng_state_before = copy.deepcopy(collector._rng.getstate())
        staged_before = copy.deepcopy(collector._staged_by_agent)
        agent_id = ordered_agent_ids[0]
        self_state = observations[agent_id]["self"]
        self.assertIsInstance(self_state, dict)
        original_energy_ratio = self_state["energy_ratio"]
        self_state["energy_ratio"] = 0.123456789

        with mock.patch(
            "evolution_sim.mind.recurrent_rollout."
            "ecological_policy_values_from_observation",
            wraps=ecological_policy_values_from_observation,
        ) as project:
            with self.assertRaisesRegex(
                RecurrentRolloutError,
                "staged observation drifted before sampling",
            ):
                collector.decide(
                    observations[agent_id],
                    dict(observations[agent_id]["action_mask"]),
                )

            self.assertIn(agent_id, collector._staged_by_agent)
            self_state["energy_ratio"] = original_energy_ratio
            drifted_mask = dict(observations[agent_id]["action_mask"])
            drifted_mask["stay"] = not drifted_mask["stay"]
            with self.assertRaisesRegex(
                RecurrentRolloutError,
                "staged inputs drifted before sampling",
            ):
                collector.decide(observations[agent_id], drifted_mask)

            self.assertEqual(collector._rng.getstate(), rng_state_before)
            self.assertEqual(collector._decision_index, 0)
            self.assertEqual(collector._pending_by_agent, {})
            self.assertEqual(collector._hidden_by_agent, {})
            self.assertEqual(collector._feedback_by_agent, {})
            self.assertEqual(collector._staged_by_agent, staged_before)
            collector._hidden_by_agent[agent_id] = (9.0, 9.0, 9.0)
            with self.assertRaisesRegex(
                RecurrentRolloutError,
                "staged inputs drifted before sampling",
            ):
                collector.decide(
                    observations[agent_id],
                    dict(observations[agent_id]["action_mask"]),
                )

            self.assertEqual(collector._rng.getstate(), rng_state_before)
            self.assertEqual(collector._decision_index, 0)
            self.assertEqual(collector._pending_by_agent, {})
            self.assertEqual(
                collector._hidden_by_agent,
                {agent_id: (9.0, 9.0, 9.0)},
            )
            self.assertEqual(collector._feedback_by_agent, {})
            self.assertEqual(collector._staged_by_agent, staged_before)
            self.assertIn(agent_id, collector._staged_by_agent)
            collector._hidden_by_agent.pop(agent_id)
            decision = collector.decide(
                copy.deepcopy(observations[agent_id]),
                dict(observations[agent_id]["action_mask"]),
            )

        self.assertEqual(project.call_count, 0)
        self.assertEqual(decision.source, "learned_recurrent_on_policy")
        self.assertEqual(
            decision.diagnostics["fixed_batch"],
            {
                "runtime_schema_version": (
                    "mind_v3_recurrent_intra_world_fixed_batch_runtime_v2"
                ),
                "runtime_exact_digest": collector.buffer.world_seed_provenance[
                    "fixed-observation-drift"
                ]["fixed_batch_runtime"]["exact_digest"],
                "batch_capacity": 320,
                "active_rows": 2,
                "execution_batch_rows": 2,
                "padding_rows": 0,
                "unused_capacity_rows": 318,
                "turn_rank": 0,
                "row_order_policy": (
                    "tick_start_seed_tick_agent_hash_permutation_v1"
                ),
                "sequential_sampling": True,
                "sequential_resolution": True,
            },
        )
        self.assertEqual(collector._decision_index, 1)
        self.assertIn(agent_id, collector._pending_by_agent)
        self.assertNotIn(agent_id, collector._staged_by_agent)
        self.assertEqual(collector._hidden_by_agent[agent_id], (1.0, 1.0, 1.0))

    def test_fixed_action_free_bootstrap_encodes_each_observation_once(self) -> None:
        core = _ExactBatchCore()
        collector = RecurrentOnPolicyCollector(
            core,
            fixed_batch_contract=RecurrentFixedBatchRuntimeContract.open_ecology(),
        )
        collector.start_world(
            world_id="fixed-action-free-single-encoding",
            environment_seed=73,
            policy_sampling_seed=79,
            rollout_ticks=1,
        )
        world = SimulationWorld(
            self._world_config(
                seed=73,
                max_ticks=1,
                initial_agents=4,
            ),
            policy=collector,
        )
        world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
        prepared = world.prepare_policy_visible_tick_start_on_clone(tick=1)

        with mock.patch(
            "evolution_sim.mind.recurrent_rollout."
            "ecological_policy_values_from_observation",
            wraps=ecological_policy_values_from_observation,
        ) as project:
            evidence = collector.finalize_action_free_bootstrap(
                tick=1,
                ordered_agent_ids=prepared.ordered_agent_ids,
                observations_by_agent=prepared.observation_snapshots,
            )

        self.assertEqual(project.call_count, len(prepared.ordered_agent_ids))
        self.assertFalse(evidence["action_sampled"])
        collector.finish_world()

    def test_fixed_action_free_bootstrap_is_atomic_on_late_observation_drift(
        self,
    ) -> None:
        core = _ExactBatchCore()
        collector = RecurrentOnPolicyCollector(
            core,
            fixed_batch_contract=RecurrentFixedBatchRuntimeContract.open_ecology(),
        )
        collector.start_world(
            world_id="fixed-action-free-atomicity",
            environment_seed=83,
            policy_sampling_seed=89,
            rollout_ticks=1,
        )
        world = SimulationWorld(
            self._world_config(
                seed=83,
                max_ticks=1,
                initial_agents=4,
            ),
            policy=collector,
        )
        world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
        prepared = world.prepare_policy_visible_tick_start_on_clone(tick=1)
        collector.stage_tick_start_batch(
            tick=1,
            ordered_agent_ids=prepared.ordered_agent_ids,
            observations_by_agent=prepared.observation_snapshots,
        )
        steps_before = collector.buffer.steps
        policy_state_before = collector._action_free_policy_state()
        drifted_agent_id = prepared.ordered_agent_ids[1]
        self_state = prepared.observation_snapshots[drifted_agent_id]["self"]
        self.assertIsInstance(self_state, dict)
        original_energy_ratio = self_state["energy_ratio"]
        self_state["energy_ratio"] = 0.87654321

        with self.assertRaisesRegex(
            RecurrentRolloutError,
            "action-free bootstrap staged observation drifted",
        ):
            collector.finalize_action_free_bootstrap(
                tick=1,
                ordered_agent_ids=prepared.ordered_agent_ids,
                observations_by_agent=prepared.observation_snapshots,
            )

        self.assertEqual(collector.buffer.steps, steps_before)
        self.assertEqual(collector._action_free_policy_state(), policy_state_before)
        self.assertEqual(collector._staged_by_agent, {})
        self.assertIsNone(collector._staged_tick)
        self.assertEqual(collector._bootstrap_value_rows, {})

        self_state["energy_ratio"] = original_energy_ratio
        next_fixed_batch_tick_before = collector._next_fixed_batch_tick
        with (
            mock.patch.object(
                collector,
                "_build_bootstrap_evidence",
                side_effect=RuntimeError("late bootstrap evidence failure"),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "late bootstrap evidence failure",
            ),
        ):
            collector.finalize_action_free_bootstrap(
                tick=1,
                ordered_agent_ids=prepared.ordered_agent_ids,
                observations_by_agent=prepared.observation_snapshots,
            )
        self.assertEqual(collector.buffer.steps, steps_before)
        self.assertEqual(
            collector._next_fixed_batch_tick,
            next_fixed_batch_tick_before,
        )
        self.assertEqual(collector._bootstrap_value_rows, {})
        self.assertEqual(collector._staged_by_agent, {})
        self.assertIsNone(collector._staged_tick)

        evidence = collector.finalize_action_free_bootstrap(
            tick=1,
            ordered_agent_ids=prepared.ordered_agent_ids,
            observations_by_agent=prepared.observation_snapshots,
        )
        self.assertEqual(
            evidence["target_eligible_agent_count"],
            len(prepared.ordered_agent_ids),
        )
        collector.finish_world()

    def test_real_torch_bucket_boundaries_match_scalar_and_direct_numerics(
        self,
    ) -> None:
        cases = (
            (3, 4),
            (64, 64),
            (65, 128),
            (
                OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY - 1,
                OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
            ),
            (
                OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
                OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
            ),
        )
        for genome_conditioned in (False, True):
            model = PublicRecurrentActorCritic(
                RecurrentActorCriticConfig(
                    encoder_size=8,
                    hidden_size=8,
                    recurrent_layers=1,
                    genome_conditioning_mode=(
                        RECURRENT_GENOME_CONDITIONING_ACTOR_FILM_V1
                        if genome_conditioned
                        else GENOME_CONDITIONING_DISABLED
                    ),
                    critic_genome_conditioning=(
                        CRITIC_GENOME_CONDITIONING_FILM_V1
                        if genome_conditioned
                        else CRITIC_GENOME_CONDITIONING_NONE
                    ),
                ),
                initialization_seed=401 + int(genome_conditioned),
            )
            core = TorchRecurrentPolicyCore(model)
            for active_rows, expected_execution_rows in cases:
                with self.subTest(
                    genome_conditioned=genome_conditioned,
                    active_rows=active_rows,
                    execution_rows=expected_execution_rows,
                ):
                    (
                        observations,
                        masks,
                        feedback,
                        hidden,
                        genomes,
                        actions,
                    ) = self._real_core_numeric_inputs(
                        core,
                        active_rows=active_rows,
                        genome_conditioned=genome_conditioned,
                    )
                    recurrent_input_shapes: list[tuple[int, ...]] = []
                    hook = model.recurrent.register_forward_pre_hook(
                        lambda _module, args: recurrent_input_shapes.append(
                            tuple(int(value) for value in args[0].shape)
                        )
                    )
                    try:
                        batched = core.forward_fixed_batch(
                            observations,
                            masks,
                            feedback,
                            hidden,
                            batch_capacity=(
                                OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY
                            ),
                            execution_batch_rows=expected_execution_rows,
                            genome_values=genomes,
                        )
                    finally:
                        hook.remove()
                    self.assertEqual(
                        recurrent_input_shapes,
                        [(1, expected_execution_rows, model.config.encoder_size)],
                    )
                    scalar = tuple(
                        core.forward_step(
                            observation,
                            mask,
                            previous,
                            state,
                            genome_values=genome,
                        )
                        for observation, mask, previous, state, genome in zip(
                            observations,
                            masks,
                            feedback,
                            hidden,
                            (
                                genomes
                                if genomes is not None
                                else (None,) * active_rows
                            ),
                            strict=True,
                        )
                    )
                    direct = self._direct_sequence_evaluation(
                        model,
                        observations=observations,
                        action_masks=masks,
                        feedback=feedback,
                        hidden=hidden,
                        genomes=genomes,
                        actions=actions,
                    )
                    self._assert_core_outputs_match_direct_evaluation(
                        batched,
                        action_masks=masks,
                        actions=actions,
                        direct=direct,
                    )
                    self._assert_core_outputs_match_direct_evaluation(
                        scalar,
                        action_masks=masks,
                        actions=actions,
                        direct=direct,
                    )

    def test_real_torch_padding_is_exact_and_hostile_rows_cannot_cross_talk(
        self,
    ) -> None:
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(
                encoder_size=8,
                hidden_size=8,
                recurrent_layers=1,
                genome_conditioning_mode=(
                    RECURRENT_GENOME_CONDITIONING_ACTOR_FILM_V1
                ),
                critic_genome_conditioning=(
                    CRITIC_GENOME_CONDITIONING_FILM_V1
                ),
            ),
            initialization_seed=409,
        )
        core = TorchRecurrentPolicyCore(model)
        (
            observations,
            masks,
            feedback,
            hidden,
            genomes,
            active_actions,
        ) = self._real_core_numeric_inputs(
            core,
            active_rows=3,
            genome_conditioned=True,
        )
        self.assertIsNotNone(genomes)
        original_forward_sequence = model.forward_sequence
        with (
            mock.patch.object(
                model,
                "forward_sequence",
                wraps=original_forward_sequence,
            ) as observed_forward,
            mock.patch.object(
                torch,
                "from_numpy",
                wraps=torch.from_numpy,
            ) as observed_numpy_bridge,
        ):
            core_outputs = core.forward_fixed_batch(
                observations,
                masks,
                feedback,
                hidden,
                batch_capacity=OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
                execution_batch_rows=4,
                genome_values=genomes,
            )
        observed_forward.assert_called_once()
        self.assertEqual(observed_numpy_bridge.call_count, 5)
        call = observed_forward.call_args
        padded_observations = call.args[0].detach().clone()
        padded_masks = call.args[1].detach().clone()
        padded_feedback = call.args[2].detach().clone()
        padded_genomes = call.kwargs["genome_values"].detach().clone()
        padded_state = call.kwargs["initial_state"].detach().clone()
        self.assertEqual(tuple(padded_observations.shape[:2]), (1, 4))
        expected_observations = torch.zeros_like(padded_observations)
        expected_observations[0, :3] = torch.tensor(
            observations,
            dtype=padded_observations.dtype,
        )
        expected_masks = torch.zeros_like(padded_masks)
        expected_masks[0, :3] = torch.tensor(
            masks,
            dtype=torch.bool,
        )
        expected_masks[0, 3, ACTION_NAMES.index("stay")] = True
        expected_feedback = torch.zeros_like(padded_feedback)
        expected_feedback[0, :3] = torch.tensor(
            tuple(row.vector() for row in feedback),
            dtype=padded_feedback.dtype,
        )
        expected_genomes = torch.zeros_like(padded_genomes)
        expected_genomes[0, :3] = torch.tensor(
            genomes,
            dtype=padded_genomes.dtype,
        )
        expected_state = torch.zeros_like(padded_state)
        expected_state[:, :3] = (
            torch.tensor(hidden, dtype=padded_state.dtype)
            .reshape(3, model.config.recurrent_layers, model.config.hidden_size)
            .permute(1, 0, 2)
        )
        for actual, expected in (
            (padded_observations.cpu(), expected_observations),
            (padded_masks.cpu(), expected_masks),
            (padded_feedback.cpu(), expected_feedback),
            (padded_genomes.cpu(), expected_genomes),
            (padded_state.cpu(), expected_state),
        ):
            self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(
            int(torch.count_nonzero(padded_observations[0, 3]).item()),
            0,
        )
        self.assertEqual(
            int(torch.count_nonzero(padded_feedback[0, 3]).item()),
            0,
        )
        self.assertEqual(
            int(torch.count_nonzero(padded_genomes[0, 3]).item()),
            0,
        )
        self.assertEqual(
            int(torch.count_nonzero(padded_state[:, 3]).item()),
            0,
        )
        expected_padding_mask = torch.zeros(
            len(ACTION_NAMES),
            dtype=torch.bool,
        )
        expected_padding_mask[ACTION_NAMES.index("stay")] = True
        self.assertTrue(
            torch.equal(padded_masks[0, 3].cpu(), expected_padding_mask)
        )

        padded_actions = torch.cat(
            (
                active_actions,
                torch.tensor(
                    ((ACTION_NAMES.index("stay"),),),
                    dtype=torch.long,
                ),
            ),
            dim=1,
        )
        with torch.no_grad():
            zero_padding = model.evaluate_sequence(
                padded_observations,
                padded_masks,
                padded_feedback,
                padded_actions,
                genome_values=padded_genomes,
                initial_state=padded_state,
            )
        hostile_observations = padded_observations.clone()
        hostile_masks = padded_masks.clone()
        hostile_feedback = padded_feedback.clone()
        hostile_genomes = padded_genomes.clone()
        hostile_state = padded_state.clone()
        hostile_observations[0, 3] = 0.9
        hostile_masks[0, 3] = True
        hostile_feedback[0, 3] = torch.tensor(
            PreviousPublicFeedback(
                requested_action_index=ACTION_NAMES.index("stay"),
                resolved_action_index=ACTION_NAMES.index("stay"),
                resolution_action_valid=True,
                moved=False,
                reward_total=0.0,
            ).vector(),
            device=hostile_feedback.device,
            dtype=hostile_feedback.dtype,
        )
        hostile_genomes[0, 3] = -0.9
        hostile_state[:, 3] = 0.8
        with torch.no_grad():
            hostile_padding = model.evaluate_sequence(
                hostile_observations,
                hostile_masks,
                hostile_feedback,
                padded_actions,
                genome_values=hostile_genomes,
                initial_state=hostile_state,
            )

        self._assert_core_outputs_match_direct_evaluation(
            core_outputs,
            action_masks=masks,
            actions=active_actions,
            direct=zero_padding,
        )
        for left, right in (
            (
                zero_padding.raw_logits[:, :3],
                hostile_padding.raw_logits[:, :3],
            ),
            (
                zero_padding.masked_logits[:, :3],
                hostile_padding.masked_logits[:, :3],
            ),
            (
                torch.distributions.Categorical(
                    logits=zero_padding.masked_logits[:, :3]
                ).probs,
                torch.distributions.Categorical(
                    logits=hostile_padding.masked_logits[:, :3]
                ).probs,
            ),
            (
                zero_padding.log_probs[:, :3],
                hostile_padding.log_probs[:, :3],
            ),
            (
                zero_padding.entropy[:, :3],
                hostile_padding.entropy[:, :3],
            ),
            (
                zero_padding.values[:, :3],
                hostile_padding.values[:, :3],
            ),
            (
                zero_padding.final_state[:, :3],
                hostile_padding.final_state[:, :3],
            ),
        ):
            self.assertTrue(torch.equal(left, right))

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
        recurrent_input_shapes: list[tuple[int, ...]] = []
        hook = model.recurrent.register_forward_pre_hook(
            lambda _module, args: recurrent_input_shapes.append(
                tuple(int(value) for value in args[0].shape)
            )
        )
        try:
            first = core.forward_fixed_batch(
                observations,
                masks,
                feedback,
                hidden,
                batch_capacity=OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
                execution_batch_rows=2,
            )
        finally:
            hook.remove()
        self.assertEqual(recurrent_input_shapes, [(1, 2, 8)])
        second = core.forward_fixed_batch(
            observations,
            masks,
            feedback,
            hidden,
            batch_capacity=OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
            execution_batch_rows=2,
        )
        self.assertEqual(first, second)
        with self.assertRaisesRegex(
            RecurrentRolloutError,
            "execution rows drifted",
        ):
            core.forward_fixed_batch(
                observations,
                masks,
                feedback,
                hidden,
                batch_capacity=OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
                execution_batch_rows=4,
            )
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

    def test_torch_collector_forwards_run_in_inference_mode_and_emit_python_floats(
        self,
    ) -> None:
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(
                encoder_size=8,
                hidden_size=8,
                recurrent_layers=1,
            ),
            initialization_seed=103,
        )
        core = TorchRecurrentPolicyCore(model)
        (
            observations,
            masks,
            feedback,
            hidden,
            _genomes,
            _actions,
        ) = self._real_core_numeric_inputs(
            core,
            active_rows=2,
            genome_conditioned=False,
        )
        observed_contexts: list[tuple[bool, bool]] = []
        original_forward_sequence = model.forward_sequence

        def observe_context(*args: object, **kwargs: object) -> SequenceEvaluation:
            observed_contexts.append(
                (
                    torch.is_inference_mode_enabled(),
                    torch.is_grad_enabled(),
                )
            )
            return original_forward_sequence(*args, **kwargs)

        with mock.patch.object(
            model,
            "forward_sequence",
            side_effect=observe_context,
        ):
            scalar = core.forward_step(
                observations[0],
                masks[0],
                feedback[0],
                hidden[0],
            )
            batched = core.forward_fixed_batch(
                observations,
                masks,
                feedback,
                hidden,
                batch_capacity=OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
                execution_batch_rows=2,
            )

        self.assertEqual(observed_contexts, [(True, False), (True, False)])
        for output in (scalar, *batched):
            self.assertIs(type(output.value), float)
            self.assertTrue(all(type(value) is float for value in output.logits))
            self.assertTrue(all(type(value) is float for value in output.next_hidden))
        runtime_metadata = core.fixed_batch_runtime_metadata()
        self.assertEqual(
            runtime_metadata["implementation"],
            (
                "PublicRecurrentActorCritic.forward_sequence_time1_"
                "bounded_bucket_batch_v3"
            ),
        )
        self.assertEqual(
            runtime_metadata["tensor_bridge_actual_branch"],
            "numpy_from_numpy_cpu_float32_bool_native_c_v1",
        )
        self.assertEqual(
            runtime_metadata["inference_context_version"],
            "torch_inference_mode_v1",
        )
        self.assertEqual(
            runtime_metadata["output_materialization_version"],
            "detach_cpu_item_tolist_then_validated_python_float_v1",
        )
        self.assertEqual(
            runtime_metadata["observation_projection_version"],
            "validated_fused_quantized_diagnostic_filter_v1",
        )
        self.assertTrue(runtime_metadata["numpy_version"])
        self.assertIn(runtime_metadata["native_byte_order"], {"little", "big"})

    def test_torch_fixed_batch_preserves_non_float32_model_dtype_fallback(
        self,
    ) -> None:
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(
                encoder_size=8,
                hidden_size=8,
                recurrent_layers=1,
            ),
            initialization_seed=107,
        ).to(dtype=torch.float64)
        core = TorchRecurrentPolicyCore(model)
        (
            observations,
            masks,
            feedback,
            hidden,
            genomes,
            _actions,
        ) = self._real_core_numeric_inputs(
            core,
            active_rows=3,
            genome_conditioned=False,
        )
        self.assertIsNone(genomes)
        original_forward_sequence = model.forward_sequence
        with (
            mock.patch.object(
                model,
                "forward_sequence",
                wraps=original_forward_sequence,
            ) as observed_forward,
            mock.patch.object(
                torch,
                "from_numpy",
                wraps=torch.from_numpy,
            ) as observed_numpy_bridge,
        ):
            outputs = core.forward_fixed_batch(
                observations,
                masks,
                feedback,
                hidden,
                batch_capacity=OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
                execution_batch_rows=4,
            )
        self.assertEqual(observed_numpy_bridge.call_count, 0)
        call = observed_forward.call_args
        self.assertEqual(call.args[0].dtype, torch.float64)
        self.assertEqual(call.args[1].dtype, torch.bool)
        self.assertEqual(call.args[2].dtype, torch.float64)
        self.assertEqual(call.kwargs["initial_state"].dtype, torch.float64)
        self.assertEqual(len(outputs), 3)
        self.assertEqual(
            core.fixed_batch_runtime_metadata()["tensor_bridge_actual_branch"],
            "torch_tensor_device_dtype_layout_fallback_v1",
        )

    def test_torch_fixed_batch_preserves_missing_numpy_fallback(self) -> None:
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(
                encoder_size=8,
                hidden_size=8,
                recurrent_layers=1,
            ),
            initialization_seed=109,
        )
        core = TorchRecurrentPolicyCore(model)
        (
            observations,
            masks,
            feedback,
            hidden,
            genomes,
            _actions,
        ) = self._real_core_numeric_inputs(
            core,
            active_rows=3,
            genome_conditioned=False,
        )
        self.assertIsNone(genomes)
        with (
            mock.patch.dict(sys.modules, {"numpy": None}),
            mock.patch.object(
                torch,
                "from_numpy",
                wraps=torch.from_numpy,
            ) as observed_numpy_bridge,
        ):
            outputs = core.forward_fixed_batch(
                observations,
                masks,
                feedback,
                hidden,
                batch_capacity=OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
                execution_batch_rows=4,
            )
            runtime_metadata = core.fixed_batch_runtime_metadata()
        self.assertEqual(observed_numpy_bridge.call_count, 0)
        self.assertEqual(len(outputs), 3)
        self.assertIsNone(runtime_metadata["numpy_version"])
        self.assertEqual(
            runtime_metadata["tensor_bridge_actual_branch"],
            "torch_tensor_device_dtype_layout_fallback_v1",
        )

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

    def test_runtime_identity_changes_with_observation_projection_version(
        self,
    ) -> None:
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(
                encoder_size=8,
                hidden_size=8,
            ),
            initialization_seed=110,
        )
        core = TorchRecurrentPolicyCore(model)
        baseline = _fixed_batch_runtime_binding(
            RecurrentFixedBatchRuntimeContract.open_ecology(),
            core=core,
        )
        with mock.patch.object(
            recurrent_rollout,
            "ECOLOGICAL_POLICY_OBSERVATION_PROJECTION_VERSION",
            "test_projection_version",
        ):
            alternate = _fixed_batch_runtime_binding(
                RecurrentFixedBatchRuntimeContract.open_ecology(),
                core=core,
            )
        self.assertNotEqual(baseline["exact_digest"], alternate["exact_digest"])
        self.assertEqual(
            alternate["observed_runtime"]["observation_projection_version"],
            "test_projection_version",
        )

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
            "fixed_batch_execution_rows",
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
