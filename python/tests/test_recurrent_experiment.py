from __future__ import annotations

import dataclasses
import math
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.trajectory import REWARD_COMPONENT_BOUNDS
from evolution_sim.mind.policy_inputs import ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE

if torch is not None:
    from evolution_sim.mind.recurrent_actor_critic import (
        RecurrentActorCriticConfig,
    )
    from evolution_sim.mind.recurrent_counterfactual_auxiliary import (
        CounterfactualHorizonScalarization,
        RecurrentCounterfactualAuxiliaryConfig,
        RecurrentCounterfactualAuxiliaryStepConfig,
    )
    from evolution_sim.mind.recurrent_counterfactual_branch import (
        RECURRENT_COUNTERFACTUAL_CONTINUATION_ENVIRONMENT_TAPE_SEED_NAMESPACE,
        RECURRENT_COUNTERFACTUAL_CONTINUATION_POLICY_TAPE_SEED_NAMESPACE,
        derive_recurrent_counterfactual_tape_seed,
    )
    from evolution_sim.mind.recurrent_counterfactual_collection import (
        RecurrentCounterfactualCollectionConfig,
    )
    from evolution_sim.mind.recurrent_experiment import (
        RECURRENT_EXPERIMENT_CONTRACT_VERSION,
        RECURRENT_POLICY_SAMPLING_SEED_NAMESPACE,
        RECURRENT_POLICY_SAMPLING_TASK_IDENTITY_VERSION,
        RECURRENT_SCALE_POLICY_SAMPLING_TASK_IDENTITY_VERSION,
        RECURRENT_TRAINING_SCENARIOS,
        RECURRENT_TRAINING_SEED_REGISTRY_SCALE_DEVELOPMENT,
        RecurrentCounterfactualExperimentConfig,
        RecurrentExperimentError,
        RecurrentExperimentRunner,
        RecurrentRolloutTask,
        _scope_diagnostics,
        build_recurrent_counterfactual_collection_tasks,
        build_recurrent_training_schedule,
        collect_recurrent_rollout_batch,
        derive_recurrent_policy_sampling_seed,
        recurrent_training_run_payload,
    )
    from evolution_sim.mind.recurrent_ppo import RecurrentPPOConfig
    from evolution_sim.mind.recurrent_policy import recurrent_model_state_sha256
    from evolution_sim.mind.recurrent_rollout import (
        PreviousPublicFeedback,
        RecurrentRolloutStep,
    )
    from evolution_sim.mind.recurrent_seed_registry import (
        LEGACY_DIAGNOSTIC_SEEDS,
        RECURRENT_SEED_REGISTRY,
        SCALE_DEVELOPMENT_SEED_REGISTRY,
    )


@unittest.skipIf(torch is None, "optional Mind ML dependency torch is not installed")
class RecurrentExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        assert torch is not None
        torch.set_num_threads(1)

    def test_schedule_uses_only_train_and_curriculum_environment_roles(self) -> None:
        schedule = build_recurrent_training_schedule(
            update_count=3,
            worlds_per_update=7,
            rollout_ticks=8,
        )

        tasks = [task for update in schedule for task in update]
        broad_allowed = set(RECURRENT_SEED_REGISTRY["train"])
        fixture_allowed = set(RECURRENT_SEED_REGISTRY["curriculum"])
        forbidden = set(LEGACY_DIAGNOSTIC_SEEDS)
        self.assertEqual(len(tasks), 21)
        self.assertEqual(len({task.task_id for task in tasks}), len(tasks))
        self.assertEqual(
            len({task.policy_sampling_identity for task in tasks}),
            len(tasks),
        )
        self.assertEqual(
            len({task.policy_sampling_seed for task in tasks}),
            len(tasks),
        )
        for task in tasks:
            allowed = broad_allowed if task.scenario == "broad" else fixture_allowed
            expected_role = "train" if task.scenario == "broad" else "curriculum"
            self.assertIn(task.environment_seed, allowed)
            self.assertEqual(task.seed_role, expected_role)
            self.assertNotIn(task.environment_seed, forbidden)
            self.assertEqual(
                task.policy_sampling_seed,
                derive_recurrent_policy_sampling_seed(
                    task_identity=str(task.policy_sampling_identity)
                ),
            )

    def test_scale_schedule_explicitly_uses_fresh_scale_only_roles(self) -> None:
        learner_seed = SCALE_DEVELOPMENT_SEED_REGISTRY["scale_learner"][0]
        schedule = build_recurrent_training_schedule(
            update_count=2,
            worlds_per_update=10,
            rollout_ticks=120,
            seed_registry_contract=(RECURRENT_TRAINING_SEED_REGISTRY_SCALE_DEVELOPMENT),
            scale_learner_seed=learner_seed,
        )

        tasks = tuple(task for update in schedule for task in update)
        old_seeds = {
            seed for seeds in RECURRENT_SEED_REGISTRY.values() for seed in seeds
        }
        self.assertEqual(len(tasks), 20)
        self.assertTrue(
            all(
                str(task.policy_sampling_identity).startswith(
                    RECURRENT_SCALE_POLICY_SAMPLING_TASK_IDENTITY_VERSION
                )
                for task in tasks
            )
        )
        self.assertTrue(
            all(
                task.task_id.startswith(f"scale-learner-{learner_seed}-")
                for task in tasks
            )
        )
        for task in tasks:
            role = "scale_train" if task.scenario == "broad" else "scale_curriculum"
            self.assertEqual(task.seed_role, role)
            self.assertIn(
                task.environment_seed,
                SCALE_DEVELOPMENT_SEED_REGISTRY[role],
            )
            self.assertNotIn(task.environment_seed, old_seeds)

    def test_scale_learner_namespaces_rollout_branch_and_tape_rng(self) -> None:
        first_learner, second_learner = SCALE_DEVELOPMENT_SEED_REGISTRY[
            "scale_learner"
        ][:2]
        schedule_kwargs = {
            "update_count": 1,
            "worlds_per_update": 10,
            "rollout_ticks": 120,
            "seed_registry_contract": (
                RECURRENT_TRAINING_SEED_REGISTRY_SCALE_DEVELOPMENT
            ),
        }
        first = build_recurrent_training_schedule(
            **schedule_kwargs,
            scale_learner_seed=first_learner,
        )
        repeated = build_recurrent_training_schedule(
            **schedule_kwargs,
            scale_learner_seed=first_learner,
        )
        second = build_recurrent_training_schedule(
            **schedule_kwargs,
            scale_learner_seed=second_learner,
        )

        self.assertEqual(first, repeated)
        first_tasks = first[0]
        second_tasks = second[0]
        self.assertEqual(
            tuple((task.scenario, task.environment_seed) for task in first_tasks),
            tuple((task.scenario, task.environment_seed) for task in second_tasks),
        )
        for first_task, second_task in zip(first_tasks, second_tasks, strict=True):
            self.assertNotEqual(first_task.task_id, second_task.task_id)
            self.assertNotEqual(
                first_task.policy_sampling_identity,
                second_task.policy_sampling_identity,
            )
            self.assertNotEqual(
                first_task.policy_sampling_seed,
                second_task.policy_sampling_seed,
            )

        first_branches = build_recurrent_counterfactual_collection_tasks(
            first_tasks,
            update_index=0,
            bundles_per_update=2,
            branch_tick_candidates=(16, 40, 64, 72),
        )
        second_branches = build_recurrent_counterfactual_collection_tasks(
            second_tasks,
            update_index=0,
            bundles_per_update=2,
            branch_tick_candidates=(16, 40, 64, 72),
        )
        for first_branch, second_branch in zip(
            first_branches,
            second_branches,
            strict=True,
        ):
            self.assertNotEqual(first_branch.task_id, second_branch.task_id)
            self.assertNotEqual(
                first_branch.source_policy_sampling_seed,
                second_branch.source_policy_sampling_seed,
            )
            self.assertNotEqual(
                first_branch.branch_selection_seed,
                second_branch.branch_selection_seed,
            )
            first_tape_identity = f"{first_branch.task_id}:continuation-tapes"
            second_tape_identity = f"{second_branch.task_id}:continuation-tapes"
            for namespace, suffix in (
                (
                    RECURRENT_COUNTERFACTUAL_CONTINUATION_ENVIRONMENT_TAPE_SEED_NAMESPACE,
                    "environment",
                ),
                (
                    RECURRENT_COUNTERFACTUAL_CONTINUATION_POLICY_TAPE_SEED_NAMESPACE,
                    "policy",
                ),
            ):
                self.assertNotEqual(
                    derive_recurrent_counterfactual_tape_seed(
                        namespace=namespace,
                        identity=f"{first_tape_identity}:{suffix}:0",
                    ),
                    derive_recurrent_counterfactual_tape_seed(
                        namespace=namespace,
                        identity=f"{second_tape_identity}:{suffix}:0",
                    ),
                )

    def test_scale_schedule_requires_registered_learner_and_legacy_rejects_it(
        self,
    ) -> None:
        with self.assertRaisesRegex(RecurrentExperimentError, "required"):
            build_recurrent_training_schedule(
                update_count=1,
                worlds_per_update=1,
                rollout_ticks=8,
                seed_registry_contract=(
                    RECURRENT_TRAINING_SEED_REGISTRY_SCALE_DEVELOPMENT
                ),
            )
        with self.assertRaisesRegex(RecurrentExperimentError, "scale_learner role"):
            build_recurrent_training_schedule(
                update_count=1,
                worlds_per_update=1,
                rollout_ticks=8,
                seed_registry_contract=(
                    RECURRENT_TRAINING_SEED_REGISTRY_SCALE_DEVELOPMENT
                ),
                scale_learner_seed=RECURRENT_SEED_REGISTRY["learner_development"][0],
            )
        with self.assertRaisesRegex(RecurrentExperimentError, "only valid"):
            build_recurrent_training_schedule(
                update_count=1,
                worlds_per_update=1,
                rollout_ticks=8,
                scale_learner_seed=SCALE_DEVELOPMENT_SEED_REGISTRY["scale_learner"][0],
            )

    def test_legacy_schedule_identity_shape_is_unchanged(self) -> None:
        task = build_recurrent_training_schedule(
            update_count=1,
            worlds_per_update=1,
            rollout_ticks=8,
            scenarios=("broad",),
        )[0][0]
        environment_seed = RECURRENT_SEED_REGISTRY["train"][0]
        self.assertEqual(
            task.task_id,
            f"update-0000-world-000000-broad-seed-{environment_seed}",
        )
        self.assertEqual(
            task.policy_sampling_identity,
            (
                f"{RECURRENT_POLICY_SAMPLING_TASK_IDENTITY_VERSION}|"
                "update=0000|world=000000|scenario=broad"
            ),
        )

    def test_scale_task_rejects_role_or_registry_mismatch(self) -> None:
        with self.assertRaisesRegex(RecurrentExperimentError, "seed_role"):
            RecurrentRolloutTask(
                "scale-role-mismatch",
                "broad",
                SCALE_DEVELOPMENT_SEED_REGISTRY["scale_train"][0],
                8,
                seed_role="scale_curriculum",
            )
        with self.assertRaisesRegex(RecurrentExperimentError, "canonical"):
            RecurrentRolloutTask(
                "scale-registry-mismatch",
                "broad",
                RECURRENT_SEED_REGISTRY["train"][0],
                8,
                seed_role="scale_train",
            )

    def test_runner_checkpoint_state_restores_model_optimizer_rng_and_progress(
        self,
    ) -> None:
        model_config = RecurrentActorCriticConfig(
            encoder_size=16,
            hidden_size=16,
        )
        ppo_config = RecurrentPPOConfig(
            learner_seed=RECURRENT_SEED_REGISTRY["learner_development"][0],
            update_epochs=1,
            sequence_minibatch_size=2,
            tbptt_steps=4,
            burn_in_steps=0,
            target_kl=0.1,
        )
        schedule = build_recurrent_training_schedule(
            update_count=2,
            worlds_per_update=2,
            rollout_ticks=4,
            scenarios=("broad",),
        )
        original = RecurrentExperimentRunner(
            learner_seed=ppo_config.learner_seed,
            device="cpu",
            model_config=model_config,
            ppo_config=ppo_config,
        )
        original.train_update(schedule[0])
        checkpoint = original.export_training_checkpoint_state()
        model_state = {
            name: value.detach().clone()
            for name, value in original.model.state_dict().items()
        }

        restored = RecurrentExperimentRunner(
            learner_seed=ppo_config.learner_seed,
            device="cpu",
            model_config=model_config,
            ppo_config=ppo_config,
        )
        restored.restore_training_checkpoint_state(
            model_state=model_state,
            optimizer_state=checkpoint["optimizer_state"],  # type: ignore[arg-type]
            rng_state=checkpoint["rng_state"],  # type: ignore[arg-type]
            completed_updates=1,
        )

        self.assertEqual(restored.completed_update_count, 1)
        self.assertEqual(restored.trainer.update_index, 1)
        self.assertEqual(
            recurrent_model_state_sha256(restored.model),
            recurrent_model_state_sha256(original.model),
        )
        with self.assertRaisesRegex(RecurrentExperimentError, "pristine"):
            restored.restore_training_checkpoint_state(
                model_state=model_state,
                optimizer_state=checkpoint["optimizer_state"],  # type: ignore[arg-type]
                rng_state=checkpoint["rng_state"],  # type: ignore[arg-type]
                completed_updates=1,
            )

        restored_update = restored.train_update(schedule[1])
        self.assertEqual(restored_update.update_index, 1)
        self.assertEqual(restored.completed_update_count, 2)

    def test_canonical_fixture_curricula_advance_without_per_scenario_repeats(
        self,
    ) -> None:
        first = build_recurrent_training_schedule(
            update_count=8,
            worlds_per_update=10,
            rollout_ticks=48,
        )
        second = build_recurrent_training_schedule(
            update_count=8,
            worlds_per_update=10,
            rollout_ticks=48,
        )

        self.assertEqual(first, second)
        tasks = [task for update in first for task in update]
        fixture_seeds = tuple(RECURRENT_SEED_REGISTRY["curriculum"][:16])
        for scenario in RECURRENT_TRAINING_SCENARIOS:
            if scenario == "broad":
                continue
            scenario_seeds = tuple(
                task.environment_seed for task in tasks if task.scenario == scenario
            )
            self.assertEqual(scenario_seeds, fixture_seeds)
            self.assertEqual(len(set(scenario_seeds)), 16)

        environment_seeds = {task.environment_seed for task in tasks}
        policy_seeds = {int(task.policy_sampling_seed) for task in tasks}
        self.assertEqual(len(policy_seeds), 80)
        self.assertTrue(environment_seeds.isdisjoint(policy_seeds))
        self.assertTrue(all(seed < 2**63 for seed in policy_seeds))
        self.assertTrue(
            all(
                str(task.policy_sampling_identity).startswith(
                    RECURRENT_POLICY_SAMPLING_TASK_IDENTITY_VERSION
                )
                for task in tasks
            )
        )

    def test_task_policy_sampling_seed_is_independent_of_environment_seed(self) -> None:
        first = RecurrentRolloutTask(
            "same-task",
            "carrion_only",
            RECURRENT_SEED_REGISTRY["curriculum"][0],
            2,
        )
        second = RecurrentRolloutTask(
            "same-task",
            "carrion_only",
            RECURRENT_SEED_REGISTRY["curriculum"][1],
            2,
        )

        self.assertNotEqual(first.environment_seed, second.environment_seed)
        self.assertEqual(
            first.policy_sampling_identity, second.policy_sampling_identity
        )
        self.assertEqual(first.policy_sampling_seed, second.policy_sampling_seed)
        serialized = dataclasses.asdict(first)
        self.assertEqual(
            serialized["environment_seed"],
            RECURRENT_SEED_REGISTRY["curriculum"][0],
        )
        self.assertEqual(serialized["seed_role"], "curriculum")
        self.assertEqual(
            serialized["policy_sampling_seed"],
            first.policy_sampling_seed,
        )

    def test_real_broad_and_fixture_batch_is_closed_and_deterministic(self) -> None:
        model_config = RecurrentActorCriticConfig(encoder_size=16, hidden_size=16)
        tasks = (
            RecurrentRolloutTask(
                "batch-broad",
                "broad",
                RECURRENT_SEED_REGISTRY["train"][0],
                2,
            ),
            RecurrentRolloutTask(
                "batch-carrion",
                "carrion_only",
                RECURRENT_SEED_REGISTRY["curriculum"][0],
                2,
            ),
        )

        def collect() -> tuple[object, object]:
            model = self._model(model_config)
            return collect_recurrent_rollout_batch(model, tasks)

        first_buffer, first = collect()
        second_buffer, second = collect()

        self.assertEqual(first, second)
        self.assertEqual(first.world_count, 2)
        self.assertGreater(first.transition_count, 0)
        self.assertEqual(
            first.terminated_sequence_count + first.truncated_sequence_count,
            first.sequence_count,
        )
        self.assertAlmostEqual(
            sum(first.requested_action_counts.values()),
            first.transition_count,
        )
        self.assertEqual(
            sum(first.chosen_action_counts.values()),
            first.transition_count,
        )
        self.assertEqual(
            set(first.chosen_action_counts),
            set(ACTION_NAMES),
        )
        self.assertEqual(
            set(first.action_mask_availability_counts),
            set(ACTION_NAMES),
        )
        self.assertEqual(
            set(first.reward_component_totals),
            set(REWARD_COMPONENT_BOUNDS),
        )
        self.assertTrue(
            math.isclose(
                sum(first.reward_component_totals.values()),
                first.total_reward,
                abs_tol=1.0e-8,
            )
        )
        for action in ACTION_NAMES:
            chosen = first.chosen_action_counts[action]
            available = first.action_mask_availability_counts[action]
            self.assertLessEqual(chosen, available)
            expected_rate = chosen / available if available else None
            self.assertEqual(first.chosen_given_valid_rates[action], expected_rate)
        self.assertEqual(
            set(first.world_diagnostics),
            {"batch-broad", "batch-carrion"},
        )
        self.assertEqual(
            set(first.scenario_diagnostics),
            {"broad", "carrion_only"},
        )
        self.assertEqual(
            sum(scope.transition_count for scope in first.world_diagnostics.values()),
            first.transition_count,
        )
        self.assertEqual(
            sum(
                scope.transition_count for scope in first.scenario_diagnostics.values()
            ),
            first.transition_count,
        )
        self.assertTrue(
            math.isclose(
                sum(scope.total_reward for scope in first.world_diagnostics.values()),
                first.total_reward,
                abs_tol=1.0e-8,
            )
        )
        self.assertEqual(first_buffer.steps, second_buffer.steps)
        self.assertEqual(
            first.policy_sampling_seed_namespace,
            RECURRENT_POLICY_SAMPLING_SEED_NAMESPACE,
        )
        self.assertEqual(
            set(first.world_seed_provenance),
            {"batch-broad", "batch-carrion"},
        )
        for task in tasks:
            provenance = first.world_seed_provenance[task.task_id]
            self.assertEqual(provenance["environment_seed"], task.environment_seed)
            self.assertEqual(provenance["seed_role"], task.seed_role)
            self.assertEqual(
                provenance["policy_sampling_seed"],
                task.policy_sampling_seed,
            )
            task_steps = tuple(
                step for step in first_buffer.steps if step.world_id == task.task_id
            )
            self.assertTrue(task_steps)
            self.assertTrue(
                all(
                    step.environment_seed == task.environment_seed
                    and step.policy_sampling_seed == task.policy_sampling_seed
                    for step in task_steps
                )
            )
        self.assertEqual(
            set(first.world_summaries[0]["summary"]),
            {
                "run_id",
                "seed",
                "ticks_executed",
                "births",
                "deaths",
                "alive_agents",
                "peak_alive_agents",
                "extinct",
            },
        )

    def test_process_parallel_collection_exactly_matches_sequential_order(self) -> None:
        model = self._model(RecurrentActorCriticConfig(encoder_size=8, hidden_size=8))
        tasks = (
            RecurrentRolloutTask(
                "parallel-broad-0",
                "broad",
                RECURRENT_SEED_REGISTRY["train"][0],
                2,
            ),
            RecurrentRolloutTask(
                "parallel-carrion-1",
                "carrion_only",
                RECURRENT_SEED_REGISTRY["curriculum"][0],
                2,
            ),
            RecurrentRolloutTask(
                "parallel-plant-2",
                "plant_only",
                RECURRENT_SEED_REGISTRY["curriculum"][1],
                2,
            ),
            RecurrentRolloutTask(
                "parallel-prey-3",
                "prey_rich",
                RECURRENT_SEED_REGISTRY["curriculum"][2],
                2,
            ),
        )

        sequential_buffer, sequential_diagnostics = collect_recurrent_rollout_batch(
            model,
            tasks,
            rollout_workers=1,
        )
        parallel_buffer, parallel_diagnostics = collect_recurrent_rollout_batch(
            model,
            tasks,
            rollout_workers=2,
        )

        self.assertEqual(parallel_buffer.steps, sequential_buffer.steps)
        self.assertEqual(parallel_diagnostics, sequential_diagnostics)
        self.assertEqual(
            tuple(parallel_diagnostics.world_diagnostics),
            tuple(task.task_id for task in tasks),
        )
        self.assertEqual(
            tuple(
                summary["task_id"] for summary in parallel_diagnostics.world_summaries
            ),
            tuple(task.task_id for task in tasks),
        )

        sequential_ff, sequential_ff_diagnostics = collect_recurrent_rollout_batch(
            model,
            tasks[:2],
            feed_forward_history_ablation=True,
            rollout_workers=1,
        )
        parallel_ff, parallel_ff_diagnostics = collect_recurrent_rollout_batch(
            model,
            tasks[:2],
            feed_forward_history_ablation=True,
            rollout_workers=2,
        )
        self.assertEqual(parallel_ff.steps, sequential_ff.steps)
        self.assertEqual(parallel_ff_diagnostics, sequential_ff_diagnostics)
        self.assertTrue(
            all(
                step.hidden == tuple(0.0 for _ in range(8))
                for step in parallel_ff.steps
            )
        )

    def test_rollout_worker_count_fails_closed(self) -> None:
        model = self._model(RecurrentActorCriticConfig(encoder_size=8, hidden_size=8))
        tasks = (
            RecurrentRolloutTask(
                "worker-validation",
                "broad",
                RECURRENT_SEED_REGISTRY["train"][0],
                1,
            ),
        )

        for invalid in (True, 0, -1, 1.5, 65):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    RecurrentExperimentError,
                    "rollout_workers",
                ):
                    collect_recurrent_rollout_batch(
                        model,
                        tasks,
                        rollout_workers=invalid,
                    )

    def test_feed_forward_ablation_resets_behavior_hidden_states(self) -> None:
        model = self._model(RecurrentActorCriticConfig(encoder_size=8, hidden_size=8))
        buffer, _summary = collect_recurrent_rollout_batch(
            model,
            (
                RecurrentRolloutTask(
                    "feed-forward",
                    "carrion_only",
                    RECURRENT_SEED_REGISTRY["curriculum"][0],
                    3,
                ),
            ),
            feed_forward_history_ablation=True,
        )

        self.assertTrue(buffer.steps)
        self.assertTrue(
            all(step.hidden == tuple(0.0 for _ in range(8)) for step in buffer.steps)
        )

    def test_scope_diagnostics_include_passive_reward_without_fabricated_action(
        self,
    ) -> None:
        components = {name: 0.0 for name in REWARD_COMPONENT_BOUNDS}
        components["survival_continuation"] = 0.02
        passive_components = {name: 0.0 for name in REWARD_COMPONENT_BOUNDS}
        passive_components["survival_continuation"] = -1.0
        action_mask = tuple(True for _ in ACTION_NAMES)
        step = RecurrentRolloutStep(
            world_id="passive-scope",
            world_seed=103,
            tick=0,
            agent_id=7,
            decision_index=0,
            observation=tuple(0.0 for _ in range(ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE)),
            previous_feedback=PreviousPublicFeedback.zero(),
            action_mask=action_mask,
            hidden=tuple(0.0 for _ in range(8)),
            action_index=ACTION_NAMES.index("stay"),
            requested_action="stay",
            logprob=-1.0,
            entropy=1.0,
            value=0.0,
            reward=0.02,
            reward_components=components,
            resolved_action_index=ACTION_NAMES.index("stay"),
            resolved_action="stay",
            resolution_action_mask=action_mask,
            action_valid=True,
            resolution_action_valid=True,
            moved=False,
            outcome={"died": False},
            terminated=True,
            passive_terminal_reward=-1.0,
            passive_terminal_reward_components=passive_components,
            passive_terminal_tick=1,
            environment_seed=103,
            policy_sampling_seed=derive_recurrent_policy_sampling_seed(
                task_identity="test-passive-scope"
            ),
        )

        diagnostics = _scope_diagnostics((step,), world_count=1)

        self.assertEqual(diagnostics.transition_count, 1)
        self.assertEqual(diagnostics.passive_terminal_count, 1)
        self.assertAlmostEqual(diagnostics.total_reward, -0.98)
        self.assertAlmostEqual(diagnostics.mean_reward, -0.98)
        self.assertAlmostEqual(
            diagnostics.reward_component_totals["survival_continuation"],
            -0.98,
        )
        self.assertEqual(sum(diagnostics.chosen_action_counts.values()), 1)
        self.assertEqual(diagnostics.chosen_action_counts["stay"], 1)
        self.assertEqual(
            sum(diagnostics.action_mask_availability_counts.values()),
            len(ACTION_NAMES),
        )

    def test_real_on_policy_update_changes_parameters_and_reports_finite_work(
        self,
    ) -> None:
        ppo = RecurrentPPOConfig(
            learner_seed=123,
            update_epochs=1,
            sequence_minibatch_size=32,
            tbptt_steps=4,
        )
        runner = RecurrentExperimentRunner(
            learner_seed=123,
            device="cpu",
            model_config=RecurrentActorCriticConfig(
                encoder_size=16,
                hidden_size=16,
            ),
            ppo_config=ppo,
        )
        before = {
            name: value.detach().clone()
            for name, value in runner.model.state_dict().items()
        }
        result = runner.run(
            (
                (
                    RecurrentRolloutTask(
                        "train-carrion-real-world",
                        "carrion_only",
                        RECURRENT_SEED_REGISTRY["curriculum"][0],
                        4,
                    ),
                ),
            )
        )

        changed = any(
            not torch.equal(before[name], value)
            for name, value in runner.model.state_dict().items()
        )
        self.assertTrue(changed)
        self.assertEqual(result.contract_version, RECURRENT_EXPERIMENT_CONTRACT_VERSION)
        self.assertTrue(result.deterministic_algorithms_enabled)
        self.assertEqual(result.total_worlds, 1)
        self.assertGreater(result.total_transitions, 0)
        self.assertEqual(
            result.environment_seed_provenance["environment_seeds_by_role"],
            {
                "train": [],
                "curriculum": [RECURRENT_SEED_REGISTRY["curriculum"][0]],
            },
        )
        self.assertFalse(result.environment_seed_provenance["lockbox_seeds_accessed"])
        self.assertIsNone(result.counterfactual_experiment)
        update = result.updates[0]
        self.assertIsNone(update.counterfactual_collection)
        self.assertIsNone(update.counterfactual_auxiliary)
        self.assertEqual(update.rollout_workers_requested, 1)
        self.assertEqual(update.rollout_workers_resolved, 1)
        self.assertEqual(
            result.rollout_execution,
            {
                "requested_workers": 1,
                "start_method": "spawn",
                "torch_threads_per_worker": 1,
                "ordered_merge": True,
                "sequential_default": True,
            },
        )
        self.assertGreater(update.optimizer.parameter_delta_l2, 0.0)
        self.assertTrue(update.optimizer.old_statistics_frozen)
        self.assertEqual(
            update.optimizer.transition_count,
            update.rollout.transition_count,
        )
        payload = recurrent_training_run_payload(result)
        self.assertEqual(payload["learner_seed"], 123)

    def test_counterfactual_task_selection_is_preoutcome_and_seed_separated(
        self,
    ) -> None:
        tasks = (
            RecurrentRolloutTask(
                "source-broad",
                "broad",
                RECURRENT_SEED_REGISTRY["train"][0],
                8,
            ),
            RecurrentRolloutTask(
                "source-carrion",
                "carrion_only",
                RECURRENT_SEED_REGISTRY["curriculum"][0],
                8,
            ),
            RecurrentRolloutTask(
                "source-plant",
                "plant_only",
                RECURRENT_SEED_REGISTRY["curriculum"][1],
                8,
            ),
        )

        selected = build_recurrent_counterfactual_collection_tasks(
            tasks,
            update_index=0,
            bundles_per_update=2,
            branch_tick_candidates=(1, 3, 5),
        )
        repeated = build_recurrent_counterfactual_collection_tasks(
            tasks,
            update_index=0,
            bundles_per_update=2,
            branch_tick_candidates=(1, 3, 5),
        )

        self.assertEqual(selected, repeated)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0].scenario, "carrion_only")
        self.assertEqual(selected[0].seed_role, "curriculum")
        self.assertEqual(selected[1].seed_role, "train")
        for task in selected:
            self.assertEqual(task.branch_tick_candidates, (1, 3, 5))
            self.assertEqual(
                len(
                    {
                        task.environment_seed,
                        task.source_policy_sampling_seed,
                        task.branch_selection_seed,
                    }
                ),
                3,
            )

    def test_counterfactual_experiment_config_fails_closed_for_ff_ablation(
        self,
    ) -> None:
        counterfactual = self._counterfactual_config()
        with self.assertRaisesRegex(
            RecurrentExperimentError,
            "feed-forward history ablation",
        ):
            RecurrentExperimentRunner(
                learner_seed=123,
                device="cpu",
                ppo_config=RecurrentPPOConfig(
                    learner_seed=123,
                    feed_forward_history_ablation=True,
                ),
                counterfactual_config=counterfactual,
            )

    def test_one_real_post_ppo_counterfactual_transaction_is_reported(self) -> None:
        runner = RecurrentExperimentRunner(
            learner_seed=123,
            device="cpu",
            model_config=RecurrentActorCriticConfig(
                encoder_size=8,
                hidden_size=8,
            ),
            ppo_config=RecurrentPPOConfig(
                learner_seed=123,
                update_epochs=1,
                sequence_minibatch_size=64,
                tbptt_steps=2,
                burn_in_steps=1,
            ),
            counterfactual_config=self._counterfactual_config(),
        )

        result = runner.run(
            (
                (
                    RecurrentRolloutTask(
                        "counterfactual-real",
                        "carrion_only",
                        RECURRENT_SEED_REGISTRY["curriculum"][0],
                        2,
                    ),
                ),
            )
        )

        self.assertIsNotNone(result.counterfactual_experiment)
        update = result.updates[0]
        self.assertIsNotNone(update.counterfactual_collection)
        self.assertIsNotNone(update.counterfactual_auxiliary)
        assert update.counterfactual_collection is not None
        assert update.counterfactual_auxiliary is not None
        self.assertEqual(len(update.counterfactual_collection.bundles), 1)
        self.assertEqual(update.counterfactual_auxiliary.group_count, 1)
        self.assertEqual(update.counterfactual_auxiliary.optimizer_step_count, 1)
        self.assertFalse(update.counterfactual_auxiliary.retry_authorized)
        self.assertFalse(
            update.counterfactual_auxiliary.runtime_action_selection_changed
        )
        self.assertFalse(update.counterfactual_auxiliary.promotion_authorized)

    def test_real_multi_tape_aggregate_transaction_consumes_terminal_target(
        self,
    ) -> None:
        schedule = build_recurrent_training_schedule(
            update_count=1,
            worlds_per_update=1,
            rollout_ticks=3,
            scenarios=("carrion_only",),
            seed_registry_contract=RECURRENT_TRAINING_SEED_REGISTRY_SCALE_DEVELOPMENT,
            scale_learner_seed=SCALE_DEVELOPMENT_SEED_REGISTRY["scale_learner"][0],
        )
        runner = RecurrentExperimentRunner(
            learner_seed=123,
            device="cpu",
            model_config=RecurrentActorCriticConfig(
                encoder_size=8,
                hidden_size=8,
            ),
            ppo_config=RecurrentPPOConfig(
                learner_seed=123,
                update_epochs=1,
                sequence_minibatch_size=64,
                tbptt_steps=2,
                burn_in_steps=1,
            ),
            counterfactual_config=RecurrentCounterfactualExperimentConfig(
                collection=RecurrentCounterfactualCollectionConfig(
                    horizons=(1, 2),
                    gamma=0.99,
                    continuation_tape_count=2,
                    terminal_target_world_tick=3,
                    uncertainty_penalty=0.5,
                ),
                auxiliary=RecurrentCounterfactualAuxiliaryConfig(
                    scalarization=CounterfactualHorizonScalarization(
                        horizon_weights=((1, 0.5), (2, 0.5)),
                    ),
                    terminal_target_weight=0.5,
                    advantage_clip=2.0,
                    behavior_kl_coefficient=1.0,
                ),
                step=RecurrentCounterfactualAuxiliaryStepConfig(
                    mean_behavior_kl_limit=1.0,
                    max_state_behavior_kl_limit=1.0,
                ),
                bundles_per_update=1,
                branch_tick_candidates=(0,),
                workers=1,
            ),
        )

        update = runner.train_update(schedule[0])

        assert update.counterfactual_collection is not None
        assert update.counterfactual_auxiliary is not None
        bundle = update.counterfactual_collection.bundles[0]
        self.assertIsNotNone(bundle.aggregate_rows)
        self.assertIsNotNone(bundle.terminal_target)
        self.assertEqual(update.counterfactual_auxiliary.group_count, 1)
        self.assertEqual(update.counterfactual_auxiliary.row_count, 3)
        self.assertIn(
            update.counterfactual_auxiliary.accepted,
            (True, False),
        )

    def test_post_ppo_collection_failure_restores_whole_update_before_retry(
        self,
    ) -> None:
        model_config = RecurrentActorCriticConfig(encoder_size=8, hidden_size=8)
        ppo_config = RecurrentPPOConfig(
            learner_seed=123,
            update_epochs=1,
            sequence_minibatch_size=64,
            tbptt_steps=2,
            burn_in_steps=1,
        )
        counterfactual_config = self._counterfactual_config()
        tasks = (
            RecurrentRolloutTask(
                "counterfactual-transaction-rollback",
                "carrion_only",
                RECURRENT_SEED_REGISTRY["curriculum"][0],
                2,
            ),
        )
        runner = RecurrentExperimentRunner(
            learner_seed=123,
            device="cpu",
            model_config=model_config,
            ppo_config=ppo_config,
            counterfactual_config=counterfactual_config,
        )
        initial_model_sha256 = recurrent_model_state_sha256(runner.model)
        initial_optimizer = runner.trainer.optimizer.state_dict()

        with patch(
            "evolution_sim.mind.recurrent_experiment."
            "collect_recurrent_counterfactual_bundles",
            side_effect=RuntimeError("injected post-PPO collection failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected post-PPO"):
                runner.train_update(tasks)

        self.assertEqual(
            recurrent_model_state_sha256(runner.model),
            initial_model_sha256,
        )
        self.assertEqual(runner.trainer.optimizer.state_dict(), initial_optimizer)
        self.assertEqual(runner.trainer.update_index, 0)
        self.assertEqual(runner.trainer.counterfactual_auxiliary_update_count, 0)
        self.assertEqual(runner._updates, [])

        retried = runner.train_update(tasks)
        clean_runner = RecurrentExperimentRunner(
            learner_seed=123,
            device="cpu",
            model_config=model_config,
            ppo_config=ppo_config,
            counterfactual_config=counterfactual_config,
        )
        clean = clean_runner.train_update(tasks)

        self.assertEqual(retried, clean)
        self.assertEqual(
            recurrent_model_state_sha256(runner.model),
            recurrent_model_state_sha256(clean_runner.model),
        )

    def test_partial_scheduled_run_failure_is_terminal_and_cannot_duplicate(
        self,
    ) -> None:
        runner = RecurrentExperimentRunner(
            learner_seed=123,
            device="cpu",
            model_config=RecurrentActorCriticConfig(
                encoder_size=8,
                hidden_size=8,
            ),
            ppo_config=RecurrentPPOConfig(
                learner_seed=123,
                update_epochs=1,
                sequence_minibatch_size=64,
                tbptt_steps=1,
                burn_in_steps=0,
            ),
        )
        broad = (
            RecurrentRolloutTask(
                "scheduled-broad",
                "broad",
                RECURRENT_SEED_REGISTRY["train"][0],
                1,
            ),
        )
        carrion = (
            RecurrentRolloutTask(
                "scheduled-carrion",
                "carrion_only",
                RECURRENT_SEED_REGISTRY["curriculum"][0],
                1,
            ),
        )
        original_train_update = runner.train_update
        call_count = 0

        def fail_second_update(
            tasks: tuple[RecurrentRolloutTask, ...],
        ) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("injected second scheduled update failure")
            return original_train_update(tasks)

        with patch.object(
            runner,
            "train_update",
            side_effect=fail_second_update,
        ):
            with self.assertRaisesRegex(RuntimeError, "injected second"):
                runner.run((broad, carrion))

        self.assertEqual(len(runner._updates), 1)
        self.assertEqual(runner._updates[0].tasks, broad)
        for retry_schedule in ((carrion,), (broad, carrion)):
            with self.subTest(retry_schedule=retry_schedule):
                with self.assertRaisesRegex(
                    RecurrentExperimentError,
                    "single-use",
                ):
                    runner.run(retry_schedule)
        with self.assertRaisesRegex(RecurrentExperimentError, "terminal"):
            runner.train_update(carrion)

    def test_scheduled_run_reports_scenarios_from_committed_update_journal(
        self,
    ) -> None:
        runner = RecurrentExperimentRunner(
            learner_seed=123,
            device="cpu",
            model_config=RecurrentActorCriticConfig(
                encoder_size=8,
                hidden_size=8,
            ),
            ppo_config=RecurrentPPOConfig(
                learner_seed=123,
                update_epochs=1,
                sequence_minibatch_size=64,
                tbptt_steps=1,
                burn_in_steps=0,
            ),
        )
        result = runner.run(
            (
                (
                    RecurrentRolloutTask(
                        "journal-broad",
                        "broad",
                        RECURRENT_SEED_REGISTRY["train"][0],
                        1,
                    ),
                ),
                (
                    RecurrentRolloutTask(
                        "journal-carrion",
                        "carrion_only",
                        RECURRENT_SEED_REGISTRY["curriculum"][0],
                        1,
                    ),
                ),
            )
        )

        self.assertEqual(result.training_scenarios, ("broad", "carrion_only"))
        with self.assertRaisesRegex(RecurrentExperimentError, "single-use"):
            runner.run(
                (
                    (
                        RecurrentRolloutTask(
                            "journal-rerun",
                            "broad",
                            RECURRENT_SEED_REGISTRY["train"][1],
                            1,
                        ),
                    ),
                )
            )
        with self.assertRaisesRegex(RecurrentExperimentError, "terminal"):
            runner.train_update(
                (
                    RecurrentRolloutTask(
                        "journal-extra",
                        "broad",
                        RECURRENT_SEED_REGISTRY["train"][2],
                        1,
                    ),
                )
            )

    def test_invalid_schedule_and_learner_seed_mismatch_fail_closed(self) -> None:
        with self.assertRaises(RecurrentExperimentError):
            build_recurrent_training_schedule(
                update_count=1,
                worlds_per_update=1,
                rollout_ticks=2,
                scenarios=(),
            )
        with self.assertRaises(RecurrentExperimentError):
            RecurrentExperimentRunner(
                learner_seed=1,
                device="cpu",
                ppo_config=RecurrentPPOConfig(learner_seed=2),
            )
        with self.assertRaisesRegex(RecurrentExperimentError, "rollout_workers"):
            RecurrentExperimentRunner(
                learner_seed=1,
                device="cpu",
                rollout_workers=0,
            )
        with self.assertRaises(RecurrentExperimentError):
            RecurrentRolloutTask(
                "mismatched-policy-seed",
                "broad",
                RECURRENT_SEED_REGISTRY["train"][0],
                2,
                policy_sampling_identity="mismatched-policy-seed",
                policy_sampling_seed=1,
            )

    def test_training_tasks_reject_holdout_and_cross_role_environment_seeds(
        self,
    ) -> None:
        invalid = (
            ("broad", RECURRENT_SEED_REGISTRY["selection"][0]),
            ("broad", RECURRENT_SEED_REGISTRY["validation"][0]),
            ("broad", RECURRENT_SEED_REGISTRY["lockbox"][0]),
            ("broad", RECURRENT_SEED_REGISTRY["curriculum"][0]),
            ("carrion_only", RECURRENT_SEED_REGISTRY["train"][0]),
            ("plant_only", RECURRENT_SEED_REGISTRY["lockbox"][0]),
        )
        for index, (scenario, seed) in enumerate(invalid):
            with self.subTest(scenario=scenario, seed=seed):
                with self.assertRaisesRegex(
                    RecurrentExperimentError,
                    "canonical .* environment seed",
                ):
                    RecurrentRolloutTask(
                        f"invalid-training-seed-{index}",
                        scenario,
                        seed,
                        1,
                    )
        with self.assertRaisesRegex(RecurrentExperimentError, "seed_role"):
            RecurrentRolloutTask(
                "explicit-cross-role",
                "broad",
                RECURRENT_SEED_REGISTRY["train"][0],
                1,
                seed_role="curriculum",
            )

    def test_schedule_rejects_duplicate_task_and_sampling_identity_preoptimizer(
        self,
    ) -> None:
        runner = RecurrentExperimentRunner(
            learner_seed=123,
            device="cpu",
            model_config=RecurrentActorCriticConfig(
                encoder_size=8,
                hidden_size=8,
            ),
            ppo_config=RecurrentPPOConfig(
                learner_seed=123,
                update_epochs=1,
                sequence_minibatch_size=64,
                tbptt_steps=1,
                burn_in_steps=0,
            ),
        )
        duplicate = RecurrentRolloutTask(
            "duplicate-scheduled-task",
            "broad",
            RECURRENT_SEED_REGISTRY["train"][0],
            1,
        )
        initial_model_sha256 = recurrent_model_state_sha256(runner.model)

        with self.assertRaisesRegex(RecurrentExperimentError, "globally unique"):
            runner.run(((duplicate,), (duplicate,)))

        self.assertEqual(runner.trainer.update_index, 0)
        self.assertEqual(runner._updates, [])
        self.assertEqual(
            recurrent_model_state_sha256(runner.model),
            initial_model_sha256,
        )

    @unittest.skipUnless(
        torch is not None and torch.backends.mps.is_available(),
        "MPS backend is unavailable",
    )
    def test_mps_real_rollout_update_keeps_gradients_finite(self) -> None:
        runner = RecurrentExperimentRunner(
            learner_seed=7,
            device="mps",
            model_config=RecurrentActorCriticConfig(
                encoder_size=8,
                hidden_size=8,
            ),
            ppo_config=RecurrentPPOConfig(
                learner_seed=7,
                update_epochs=1,
                sequence_minibatch_size=32,
                tbptt_steps=2,
                burn_in_steps=1,
            ),
        )

        result = runner.train_update(
            (
                RecurrentRolloutTask(
                    "mps-gradient-regression",
                    "carrion_only",
                    RECURRENT_SEED_REGISTRY["curriculum"][0],
                    2,
                ),
            )
        )

        self.assertGreater(result.optimizer.parameter_delta_l2, 0.0)

    @staticmethod
    def _model(config: object) -> object:
        from evolution_sim.mind.recurrent_actor_critic import (
            PublicRecurrentActorCritic,
        )

        return PublicRecurrentActorCritic(config, initialization_seed=19)

    @staticmethod
    def _counterfactual_config() -> object:
        return RecurrentCounterfactualExperimentConfig(
            collection=RecurrentCounterfactualCollectionConfig(
                horizons=(1, 2),
                gamma=0.99,
            ),
            auxiliary=RecurrentCounterfactualAuxiliaryConfig(
                scalarization=CounterfactualHorizonScalarization(
                    horizon_weights=((1, 0.5), (2, 0.5)),
                ),
                advantage_clip=2.0,
                behavior_kl_coefficient=1.0,
            ),
            step=RecurrentCounterfactualAuxiliaryStepConfig(
                mean_behavior_kl_limit=1.0,
                max_state_behavior_kl_limit=1.0,
            ),
            bundles_per_update=1,
            branch_tick_candidates=(0,),
            workers=1,
        )


if __name__ == "__main__":
    unittest.main()
