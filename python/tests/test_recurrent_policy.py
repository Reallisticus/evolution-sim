from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    from evolution_sim.config.schema import (
        CombatConfig,
        DietMatchingConfig,
        ReproductionConfig,
        SignalConfig,
        WorldConfig,
    )
    from evolution_sim.env.runtime.action_contract import ACTION_NAMES
    import evolution_sim.env.runtime.reproduction as runtime_reproduction
    from evolution_sim.env.runtime.state import (
        RunMode,
        empty_mind_inheritance_metadata,
    )
    from evolution_sim.env.world import SimulationWorld
    from evolution_sim.mind.recurrent_actor_critic import (
        GENOME_CONDITIONING_ACTOR_FILM_V1,
        PublicRecurrentActorCritic,
        RecurrentActorCriticConfig,
    )
    from evolution_sim.mind.recurrent_genome_population import (
        RecurrentGenomePopulationManager,
    )
    from evolution_sim.mind.recurrent_policy import (
        PUBLIC_RECURRENT_DISTRIBUTION_DIAGNOSTIC_SCHEMA_VERSION,
        PUBLIC_RECURRENT_GENOME_DIAGNOSTIC_SCHEMA_VERSION,
        PUBLIC_RECURRENT_SAMPLED_SELECTION,
        RECURRENT_COUNTERFACTUAL_ACTION_SOURCE,
        RECURRENT_GENOME_WORLD_PROVENANCE_SCHEMA_VERSION,
        DeterministicPublicRecurrentPolicy,
        RecurrentPolicyAdapterError,
        frozen_cpu_model_copy,
        validate_public_recurrent_distribution_diagnostics,
        validate_public_recurrent_history_prefix,
    )
    from evolution_sim.mind.recurrent_rollout import RECURRENT_ROLLOUT_ACTION_SOURCE


def _conditioned_model(*, initialization_seed: int) -> PublicRecurrentActorCritic:
    return PublicRecurrentActorCritic(
        RecurrentActorCriticConfig(
            encoder_size=16,
            hidden_size=16,
            genome_conditioning_mode=GENOME_CONDITIONING_ACTOR_FILM_V1,
        ),
        initialization_seed=initialization_seed,
    )


def _small_policy_world_config(
    *,
    seed: int,
    max_ticks: int,
    child_energy_fraction: float = 0.3,
    base_energy_drain: float = 0.0,
) -> WorldConfig:
    return WorldConfig(
        seed=seed,
        max_ticks=max_ticks,
        width=5,
        height=5,
        initial_agents=1,
        max_agents=20,
        water_tile_ratio=0.0,
        forest_tile_ratio=0.0,
        wetland_tile_ratio=0.0,
        rocky_tile_ratio=0.0,
        base_energy_drain=base_energy_drain,
        base_hydration_drain=0.0,
        reproduction=ReproductionConfig(
            min_age=1,
            cooldown_ticks=1_000,
            min_hydration_fraction=0.0,
            energy_cost=0.0,
            child_energy_fraction=child_energy_fraction,
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
    )


@unittest.skipIf(torch is None, "optional Mind ML dependency torch is not installed")
class DeterministicPublicRecurrentPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        assert torch is not None
        torch.set_num_threads(1)

    def test_public_history_prefix_record_count_rejects_bool_alias(self) -> None:
        with self.assertRaisesRegex(
            RecurrentPolicyAdapterError,
            "record count",
        ):
            validate_public_recurrent_history_prefix(
                {
                    "schema_version": ("mind_v3_public_recurrent_history_prefix_v1"),
                    "record_count": False,
                    "records": [],
                }
            )

    def test_public_history_payload_shapes_reject_float_aliases(self) -> None:
        policy = DeterministicPublicRecurrentPolicy(
            PublicRecurrentActorCritic(initialization_seed=17),
            artifact_digest="a" * 64,
            capture_public_history=True,
        )
        world = SimulationWorld(
            WorldConfig(seed=23, max_ticks=1),
            policy=policy,
        )
        world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
        agent = world.alive_agents()[0]
        prefix = policy.diagnostics_checkpoint_state(agent_id=agent.agent_id)[
            "public_history_prefix"
        ]
        self.assertTrue(prefix["records"])
        for field in ("public_observation", "previous_public_feedback"):
            with self.subTest(field=field):
                tampered = deepcopy(prefix)
                expected_size = tampered["records"][0][field]["shape"][0]
                tampered["records"][0][field]["shape"] = [float(expected_size)]
                with self.assertRaisesRegex(
                    RecurrentPolicyAdapterError,
                    "integer size",
                ):
                    validate_public_recurrent_history_prefix(tampered)

    def test_public_history_rejects_fractional_previous_feedback_one_hot(self) -> None:
        policy = DeterministicPublicRecurrentPolicy(
            PublicRecurrentActorCritic(initialization_seed=17),
            artifact_digest="a" * 64,
            capture_public_history=True,
        )
        world = SimulationWorld(
            WorldConfig(seed=23, max_ticks=1),
            policy=policy,
        )
        world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
        agent = world.alive_agents()[0]
        prefix = policy.diagnostics_checkpoint_state(agent_id=agent.agent_id)[
            "public_history_prefix"
        ]
        tampered = deepcopy(prefix)
        tampered["records"][0]["previous_public_feedback"]["values"][0] = 0.5
        with self.assertRaisesRegex(
            RecurrentPolicyAdapterError,
            "semantic contract",
        ):
            validate_public_recurrent_history_prefix(tampered)

    def test_frozen_copy_is_independent_cpu_float32_and_inference_only(self) -> None:
        model = PublicRecurrentActorCritic(initialization_seed=41)
        snapshot = frozen_cpu_model_copy(model)

        self.assertIsNot(snapshot, model)
        self.assertFalse(snapshot.training)
        self.assertTrue(
            all(parameter.device.type == "cpu" for parameter in snapshot.parameters())
        )
        self.assertTrue(
            all(parameter.dtype == torch.float32 for parameter in snapshot.parameters())
        )
        self.assertTrue(
            all(not parameter.requires_grad for parameter in snapshot.parameters())
        )
        with torch.no_grad():
            model.actor.bias.add_(1.0)
        self.assertFalse(torch.equal(model.actor.bias, snapshot.actor.bias))

    def test_real_world_policy_is_legal_learned_and_replay_repeatable(self) -> None:
        def run_once() -> tuple[dict[str, object], tuple[tuple[object, ...], ...]]:
            model = PublicRecurrentActorCritic(initialization_seed=90210)
            policy = DeterministicPublicRecurrentPolicy(
                model,
                artifact_digest="a" * 64,
            )
            world = SimulationWorld(
                WorldConfig(seed=17, max_ticks=4),
                policy=policy,
            )
            result = world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
            rows = tuple(
                (
                    record["tick"],
                    record["agent_id"],
                    record["requested_action"],
                    record["resolved_action"],
                    record["reward"]["total"],
                )
                for record in world.trajectory_records
            )
            self.assertTrue(rows)
            for record in world.trajectory_records:
                requested = str(record["requested_action"])
                if record["action_source"] == "passive":
                    continue
                self.assertEqual(
                    record["action_source"],
                    RECURRENT_ROLLOUT_ACTION_SOURCE,
                )
                self.assertIn(requested, ACTION_NAMES)
                self.assertTrue(record["action_mask"][requested])
            return result.summary, rows

        first_summary, first_rows = run_once()
        second_summary, second_rows = run_once()

        self.assertEqual(first_summary, second_summary)
        self.assertEqual(first_rows, second_rows)

    def test_token_aware_recurrent_policy_executes_real_world_actions(self) -> None:
        signals = SignalConfig(communication_signal_emission_enabled=True)
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig.for_signal_config(
                signals,
                encoder_size=16,
                hidden_size=16,
            ),
            initialization_seed=90210,
        )
        policy = DeterministicPublicRecurrentPolicy(
            model,
            artifact_digest="f" * 64,
        )
        world = SimulationWorld(
            WorldConfig(seed=17, max_ticks=2, signals=signals),
            policy=policy,
        )

        world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)

        learned_records = [
            record
            for record in world.trajectory_records
            if record["action_source"] == RECURRENT_ROLLOUT_ACTION_SOURCE
        ]
        self.assertTrue(learned_records)
        self.assertEqual(
            model.config.public_input_schema_version, "mind_ecological_policy_input_v2"
        )
        self.assertEqual(model.config.public_input_size, 645)
        for record in learned_records:
            requested_action = str(record["requested_action"])
            self.assertEqual(
                record["observation_input"]["schema_version"],
                "mind_observation_v4",
            )
            self.assertEqual(record["observation_input"]["shape"], [646])
            self.assertTrue(record["action_mask"][requested_action])

    def test_parameter_mutation_after_freeze_fails_before_world_action(self) -> None:
        model = PublicRecurrentActorCritic(initialization_seed=7)
        policy = DeterministicPublicRecurrentPolicy(
            model,
            artifact_digest="b" * 64,
            copy_to_cpu=False,
        )
        with torch.no_grad():
            model.actor.bias.add_(0.5)

        with self.assertRaisesRegex(
            RecurrentPolicyAdapterError,
            "parameters changed",
        ):
            SimulationWorld(
                WorldConfig(seed=3, max_ticks=1),
                policy=policy,
            ).run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)

    def test_conditioned_buffer_mutation_after_freeze_fails_before_world(self) -> None:
        model = _conditioned_model(initialization_seed=8)
        policy = DeterministicPublicRecurrentPolicy(
            model,
            artifact_digest="9" * 64,
            copy_to_cpu=False,
        )
        film_scale = dict(model.named_buffers())["_genome_film_scale_coefficients"]
        with torch.no_grad():
            film_scale.add_(0.25)

        with self.assertRaisesRegex(
            RecurrentPolicyAdapterError,
            "buffers changed",
        ):
            policy.start_world(
                world_identity="mutated-conditioned-buffer",
                genome_stream_seed=83,
                genome_population_mode="heritable",
            )

    def test_feed_forward_ablation_uses_zero_state_for_every_world_decision(
        self,
    ) -> None:
        model = PublicRecurrentActorCritic(initialization_seed=31)
        policy = DeterministicPublicRecurrentPolicy(
            model,
            artifact_digest="c" * 64,
            copy_to_cpu=False,
            reset_recurrent_state_each_decision=True,
        )

        with patch.object(model, "act", wraps=model.act) as act:
            SimulationWorld(
                WorldConfig(seed=17, max_ticks=3),
                policy=policy,
            ).run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)

        self.assertGreater(len(act.call_args_list), 1)
        for call in act.call_args_list:
            state = call.kwargs["recurrent_state"]
            self.assertTrue(torch.equal(state, torch.zeros_like(state)))

    def test_seeded_sampled_policy_is_legal_diverse_and_replay_repeatable(self) -> None:
        def run_once() -> tuple[str, ...]:
            model = PublicRecurrentActorCritic(initialization_seed=47)
            policy = DeterministicPublicRecurrentPolicy(
                model,
                artifact_digest="d" * 64,
                sampling_seed=991,
            )
            world = SimulationWorld(
                WorldConfig(seed=23, max_ticks=8),
                policy=policy,
            )
            world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
            actions: list[str] = []
            for record in world.trajectory_records:
                if record["action_source"] == "passive":
                    continue
                requested = str(record["requested_action"])
                self.assertTrue(record["action_mask"][requested])
                self.assertIn(
                    PUBLIC_RECURRENT_SAMPLED_SELECTION,
                    str(record["policy_version"]),
                )
                actions.append(requested)
            return tuple(actions)

        first = run_once()
        second = run_once()
        self.assertEqual(first, second)
        self.assertGreater(len(set(first)), 1)

    def test_single_valid_action_excludes_normalized_entropy_and_top_two_margin(
        self,
    ) -> None:
        world = SimulationWorld(WorldConfig(seed=23, max_ticks=1))
        agent = world.alive_agents()[0]
        observation = world._observe_agent(agent)
        action_mask = {action: action == "stay" for action in ACTION_NAMES}
        policy = DeterministicPublicRecurrentPolicy(
            PublicRecurrentActorCritic(initialization_seed=53),
            artifact_digest="1" * 64,
        )

        decision = policy.decide(observation, action_mask)
        distribution = decision.diagnostics["learned_masked_distribution"]

        validate_public_recurrent_distribution_diagnostics(
            distribution,
            action_mask=action_mask,
        )
        self.assertEqual(decision.requested_action, "stay")
        self.assertEqual(distribution["valid_action_count"], 1)
        self.assertFalse(distribution["normalized_entropy_eligible"])
        self.assertIsNone(distribution["normalized_entropy"])
        self.assertIsNone(distribution["top_two_probability_margin"])
        self.assertEqual(distribution["top_action_probability"], 1.0)

        bool_count = dict(distribution)
        bool_count["valid_action_count"] = True
        with self.assertRaisesRegex(
            RecurrentPolicyAdapterError,
            "positive integer",
        ):
            validate_public_recurrent_distribution_diagnostics(
                bool_count,
                action_mask=action_mask,
            )

    def test_distribution_probabilities_and_entropy_are_bound_to_logits(
        self,
    ) -> None:
        world = SimulationWorld(WorldConfig(seed=23, max_ticks=1))
        agent = world.alive_agents()[0]
        observation = world._observe_agent(agent)
        action_mask = dict(observation["action_mask"])
        self.assertGreater(sum(action_mask.values()), 1)
        policy = DeterministicPublicRecurrentPolicy(
            PublicRecurrentActorCritic(initialization_seed=59),
            artifact_digest="2" * 64,
        )
        distribution = policy.decide(
            observation,
            action_mask,
        ).diagnostics["learned_masked_distribution"]
        validate_public_recurrent_distribution_diagnostics(
            distribution,
            action_mask=action_mask,
        )

        logit_tampered = deepcopy(distribution)
        action = next(name for name in ACTION_NAMES if action_mask[name])
        logit_tampered["masked_logits"][action] += 1.0
        with self.assertRaisesRegex(
            RecurrentPolicyAdapterError,
            "from masked logits",
        ):
            validate_public_recurrent_distribution_diagnostics(
                logit_tampered,
                action_mask=action_mask,
            )

        entropy_tampered = deepcopy(distribution)
        entropy_tampered["entropy"] = 0.0
        entropy_tampered["normalized_entropy"] = 0.0
        with self.assertRaisesRegex(
            RecurrentPolicyAdapterError,
            "entropy from probabilities",
        ):
            validate_public_recurrent_distribution_diagnostics(
                entropy_tampered,
                action_mask=action_mask,
            )

    def test_diagnostics_override_preserves_actor_state_and_sampler_progression(
        self,
    ) -> None:
        observation_world = SimulationWorld(WorldConfig(seed=17, max_ticks=1))
        agent = observation_world.alive_agents()[0]
        observation = observation_world._observe_agent(agent)
        action_mask = dict(observation["action_mask"])

        normal = DeterministicPublicRecurrentPolicy(
            PublicRecurrentActorCritic(initialization_seed=73),
            artifact_digest="e" * 64,
            sampling_seed=991,
            capture_public_history=True,
        )
        intervened = DeterministicPublicRecurrentPolicy(
            PublicRecurrentActorCritic(initialization_seed=73),
            artifact_digest="e" * 64,
            sampling_seed=991,
            capture_public_history=True,
        )
        normal_decision = normal.decide(observation, action_mask)
        forced_action = next(
            action
            for action in ACTION_NAMES
            if action_mask[action] and action != normal_decision.requested_action
        )
        forced_decision = intervened.decide_with_action_override(
            observation,
            action_mask,
            requested_action=forced_action,
            intervention_id="unit-test-counterfactual",
        )

        self.assertEqual(forced_decision.requested_action, forced_action)
        self.assertEqual(
            forced_decision.source,
            RECURRENT_COUNTERFACTUAL_ACTION_SOURCE,
        )
        self.assertNotIn("counterfactual_intervention", normal_decision.diagnostics)
        self.assertEqual(
            forced_decision.diagnostics["natural_requested_action"],
            normal_decision.requested_action,
        )
        normal_distribution = normal_decision.diagnostics["learned_masked_distribution"]
        forced_distribution = forced_decision.diagnostics["learned_masked_distribution"]
        self.assertEqual(normal_distribution, forced_distribution)
        self.assertEqual(
            normal_distribution["schema_version"],
            PUBLIC_RECURRENT_DISTRIBUTION_DIAGNOSTIC_SCHEMA_VERSION,
        )
        validate_public_recurrent_distribution_diagnostics(
            normal_distribution,
            action_mask=action_mask,
        )
        self.assertEqual(
            normal_distribution["selected_action"],
            normal_decision.requested_action,
        )
        self.assertAlmostEqual(
            sum(normal_distribution["probabilities"].values()),
            1.0,
            places=6,
        )
        self.assertEqual(
            normal_distribution["eat_probability"],
            normal_distribution["probabilities"]["eat"],
        )
        self.assertGreaterEqual(normal_distribution["entropy"], 0.0)
        self.assertTrue(normal_distribution["normalized_entropy_eligible"])
        self.assertGreaterEqual(normal_distribution["normalized_entropy"], 0.0)
        self.assertLessEqual(normal_distribution["normalized_entropy"], 1.0)
        for action in ACTION_NAMES:
            if action_mask[action]:
                self.assertIsNotNone(normal_distribution["masked_logits"][action])
            else:
                self.assertIsNone(normal_distribution["masked_logits"][action])
                self.assertEqual(normal_distribution["probabilities"][action], 0.0)
        self.assertTrue(forced_decision.diagnostics["counterfactual_intervention"])
        self.assertFalse(
            forced_decision.diagnostics["runtime_action_selection_integration"]
        )

        def finalize(
            policy: DeterministicPublicRecurrentPolicy,
            decision: object,
        ) -> None:
            policy.observe_transition(
                {
                    "agent_id": agent.agent_id,
                    "policy_id": decision.policy_id,
                    "policy_version": decision.policy_version,
                    "action_source": decision.source,
                    "requested_action": decision.requested_action,
                    "resolved_action": decision.requested_action,
                    "resolution_action_valid": True,
                    "moved": decision.requested_action.startswith("move_"),
                    "reward": {"total": 0.0},
                    "after": {"alive": True},
                }
            )

        finalize(normal, normal_decision)
        finalize(intervened, forced_decision)
        normal_checkpoint = normal.diagnostics_checkpoint_state(agent_id=agent.agent_id)
        forced_checkpoint = intervened.diagnostics_checkpoint_state(
            agent_id=agent.agent_id
        )
        self.assertEqual(
            normal_checkpoint["recurrent_state_sha256"],
            forced_checkpoint["recurrent_state_sha256"],
        )
        self.assertEqual(
            normal_checkpoint["sampling_state_sha256"],
            forced_checkpoint["sampling_state_sha256"],
        )
        forced_feedback = forced_checkpoint["previous_public_feedback"]
        self.assertEqual(forced_feedback[ACTION_NAMES.index(forced_action)], 1.0)
        self.assertNotEqual(
            normal_checkpoint["previous_public_feedback"],
            forced_feedback,
        )

    def test_diagnostics_override_rejects_invalid_action_without_pending_state(
        self,
    ) -> None:
        world = SimulationWorld(WorldConfig(seed=19, max_ticks=1))
        agent = world.alive_agents()[0]
        observation = world._observe_agent(agent)
        action_mask = dict(observation["action_mask"])
        invalid_action = next(
            action for action in ACTION_NAMES if action_mask[action] is False
        )
        policy = DeterministicPublicRecurrentPolicy(
            PublicRecurrentActorCritic(initialization_seed=11),
            artifact_digest="f" * 64,
            sampling_seed=3,
            capture_public_history=True,
        )
        before = policy.diagnostics_checkpoint_state(agent_id=agent.agent_id)
        with self.assertRaisesRegex(
            RecurrentPolicyAdapterError,
            "not valid in the current mask",
        ):
            policy.decide_with_action_override(
                observation,
                action_mask,
                requested_action=invalid_action,
                intervention_id="invalid-action-test",
            )
        after = policy.diagnostics_checkpoint_state(agent_id=agent.agent_id)
        self.assertEqual(before, after)

    def test_disabled_model_preserves_legacy_runtime_surface_exactly(self) -> None:
        model = PublicRecurrentActorCritic(initialization_seed=101)
        policy = DeterministicPublicRecurrentPolicy(
            model,
            artifact_digest="1" * 64,
            copy_to_cpu=False,
        )
        with patch.object(model, "act", wraps=model.act) as act:
            world = SimulationWorld(
                _small_policy_world_config(seed=3, max_ticks=1),
                policy=policy,
            )
            world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)

        self.assertEqual(policy.genome_conditioning_mode, "disabled")
        self.assertIsNone(policy.last_world_genome_provenance)
        self.assertEqual(
            world.alive_agents()[0].mind_inheritance_metadata,
            empty_mind_inheritance_metadata(),
        )
        learned_diagnostics = [
            diagnostics
            for record, diagnostics in zip(
                world.trajectory_records,
                world.policy_decision_diagnostics_records,
                strict=True,
            )
            if record["action_source"] == RECURRENT_ROLLOUT_ACTION_SOURCE
        ]
        self.assertTrue(learned_diagnostics)
        conditioned_fields = {
            "genome_conditioning_mode",
            "genome_population_mode",
            "genome_population_binding_sha256",
            "genome_sha256",
            "genome_stream_seed",
        }
        for diagnostics in learned_diagnostics:
            self.assertEqual(
                diagnostics["schema_version"],
                "mind_v3_public_recurrent_actor_critic_decision_v3",
            )
            self.assertTrue(conditioned_fields.isdisjoint(diagnostics))
        self.assertTrue(act.call_args_list)
        self.assertTrue(
            all("genome_values" not in call.kwargs for call in act.call_args_list)
        )

    def test_conditioned_policy_requires_exact_world_binding_before_construction(
        self,
    ) -> None:
        policy = DeterministicPublicRecurrentPolicy(
            _conditioned_model(initialization_seed=103),
            artifact_digest="2" * 64,
        )
        with self.assertRaisesRegex(
            RecurrentPolicyAdapterError,
            "start_world before SimulationWorld construction",
        ):
            SimulationWorld(
                _small_policy_world_config(seed=5, max_ticks=1),
                policy=policy,
            )

        for stream_seed, population_mode, message in (
            (True, "heritable", "unsigned 64-bit"),
            (7, "disabled", "heritable or zero_all"),
            (7, "unknown", "heritable or zero_all"),
        ):
            with self.subTest(
                stream_seed=stream_seed,
                population_mode=population_mode,
            ):
                with self.assertRaisesRegex(RecurrentPolicyAdapterError, message):
                    policy.start_world(
                        world_identity="strict-conditioned-world",
                        genome_stream_seed=stream_seed,
                        genome_population_mode=population_mode,
                    )

        policy.start_world(
            world_identity="strict-conditioned-world",
            genome_stream_seed=7,
            genome_population_mode="heritable",
        )
        world = SimulationWorld(
            _small_policy_world_config(seed=5, max_ticks=1),
            policy=policy,
        )
        self.assertTrue(
            world.alive_agents()[0].mind_inheritance_metadata["inherited_state"]
        )
        with self.assertRaisesRegex(
            RecurrentPolicyAdapterError,
            "reset the active conditioned world",
        ):
            policy.start_world(
                world_identity="second-world",
                genome_stream_seed=8,
                genome_population_mode="heritable",
            )
        policy.reset_world()

    def test_real_founder_and_child_genomes_reach_actor_and_reset_provenance(
        self,
    ) -> None:
        world_identity = "conditioned-policy-founder-child"
        genome_stream_seed = 1776
        model = _conditioned_model(initialization_seed=107)
        policy = DeterministicPublicRecurrentPolicy(
            model,
            artifact_digest="3" * 64,
            copy_to_cpu=False,
        )
        policy.start_world(
            world_identity=world_identity,
            genome_stream_seed=genome_stream_seed,
            genome_population_mode="heritable",
        )
        world = SimulationWorld(
            _small_policy_world_config(seed=7, max_ticks=1),
            policy=policy,
        )
        founder = world.alive_agents()[0]
        founder.age = 10
        founder.energy = founder.genome.max_energy * 1.25
        founder.hydration = founder.genome.max_hydration
        founder.health = founder.max_health
        founder.last_reproduction_tick = -10_000
        births = runtime_reproduction.run_reproduction_phase(
            world,
            context=world._reproduction_context(),
        )
        child = next(
            agent
            for agent in world.agents.values()
            if agent.parent_id == founder.agent_id
        )
        self.assertEqual(births, 1)
        self.assertEqual(
            founder.mind_inheritance_metadata["inheritance_kind"],
            "founder",
        )
        self.assertEqual(
            child.mind_inheritance_metadata["inheritance_kind"],
            "asexual",
        )

        reference = RecurrentGenomePopulationManager(
            genome_stream_seed=genome_stream_seed,
            world_identity=world_identity,
            mode="heritable",
        )
        reference.founder_metadata(agent_id=founder.agent_id)
        reference.child_metadata(
            child_agent_id=child.agent_id,
            primary_parent_id=founder.agent_id,
            secondary_parent_id=None,
        )
        with patch.object(model, "act", wraps=model.act) as act:
            world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)

        learned_records = [
            (record, diagnostics)
            for record, diagnostics in zip(
                world.trajectory_records,
                world.policy_decision_diagnostics_records,
                strict=True,
            )
            if record["action_source"] == RECURRENT_ROLLOUT_ACTION_SOURCE
        ]
        self.assertEqual(len(learned_records), len(act.call_args_list))
        self.assertEqual(
            {int(record["agent_id"]) for record, _ in learned_records},
            {founder.agent_id, child.agent_id},
        )
        for (record, diagnostics), call in zip(
            learned_records,
            act.call_args_list,
            strict=True,
        ):
            agent_id = int(record["agent_id"])
            binding = reference.genome_binding_for_agent(agent_id)
            observed_genome = call.kwargs["genome_values"]
            expected_genome = torch.tensor(
                binding.genome.values,
                device=observed_genome.device,
                dtype=observed_genome.dtype,
            )
            self.assertTrue(torch.equal(observed_genome, expected_genome))
            self.assertTrue(
                torch.equal(
                    call.kwargs["recurrent_state"],
                    torch.zeros_like(call.kwargs["recurrent_state"]),
                )
            )
            self.assertTrue(
                torch.equal(
                    call.args[2],
                    torch.zeros_like(call.args[2]),
                )
            )
            self.assertEqual(
                diagnostics["schema_version"],
                PUBLIC_RECURRENT_GENOME_DIAGNOSTIC_SCHEMA_VERSION,
            )
            self.assertFalse(diagnostics["previous_feedback_available"])
            self.assertEqual(
                diagnostics["genome_conditioning_mode"],
                GENOME_CONDITIONING_ACTOR_FILM_V1,
            )
            self.assertEqual(diagnostics["genome_population_mode"], "heritable")
            self.assertEqual(
                diagnostics["genome_population_binding_sha256"],
                reference.binding_sha256,
            )
            self.assertEqual(diagnostics["genome_sha256"], binding.genome_sha256)
            self.assertEqual(
                diagnostics["genome_stream_seed"],
                genome_stream_seed,
            )

        policy.reset_world()
        provenance = policy.last_world_genome_provenance
        self.assertIsNotNone(provenance)
        assert provenance is not None
        self.assertEqual(
            provenance["schema_version"],
            RECURRENT_GENOME_WORLD_PROVENANCE_SCHEMA_VERSION,
        )
        self.assertEqual(
            provenance["genome_population_final_state_sha256"],
            reference.state_sha256,
        )
        self.assertNotEqual(
            provenance["genome_population_final_state_sha256"],
            provenance["genome_population_pre_founder_state_sha256"],
        )
        self.assertEqual(
            provenance["genome_population_reset_state_sha256"],
            provenance["genome_population_pre_founder_state_sha256"],
        )
        self.assertEqual(len(str(provenance["provenance_sha256"])), 64)

        policy.start_world(
            world_identity="conditioned-policy-reuse",
            genome_stream_seed=1777,
            genome_population_mode="zero_all",
        )
        reused_world = SimulationWorld(
            _small_policy_world_config(seed=11, max_ticks=1),
            policy=policy,
        )
        reused_world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
        self.assertTrue(reused_world.trajectory_records)
        policy.reset_world()
        self.assertEqual(
            policy.last_world_genome_provenance["genome_population_mode"],
            "zero_all",
        )

    def test_same_tick_born_dead_child_is_removed_from_final_population(self) -> None:
        world_identity = "conditioned-policy-same-tick-child-death"
        genome_stream_seed = 61
        policy = DeterministicPublicRecurrentPolicy(
            _conditioned_model(initialization_seed=109),
            artifact_digest="4" * 64,
        )
        policy.start_world(
            world_identity=world_identity,
            genome_stream_seed=genome_stream_seed,
            genome_population_mode="heritable",
        )
        world = SimulationWorld(
            _small_policy_world_config(
                seed=59,
                max_ticks=1,
                child_energy_fraction=0.0,
            ),
            policy=policy,
        )
        founder = world.alive_agents()[0]
        founder.age = 10
        founder.energy = founder.genome.max_energy * 1.25
        founder.hydration = founder.genome.max_hydration
        founder.health = founder.max_health
        founder.last_reproduction_tick = -10_000

        world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
        children = [
            agent
            for agent in world.agents.values()
            if agent.parent_id == founder.agent_id
        ]
        self.assertEqual(len(children), 1)
        child = children[0]
        self.assertFalse(child.alive)
        self.assertEqual(child.birth_tick, 0)
        self.assertEqual(child.death_tick, 0)
        self.assertTrue(child.mind_inheritance_metadata["inherited_state"])

        reference = RecurrentGenomePopulationManager(
            genome_stream_seed=genome_stream_seed,
            world_identity=world_identity,
            mode="heritable",
        )
        reference.founder_metadata(agent_id=founder.agent_id)
        policy.reset_world()
        provenance = policy.last_world_genome_provenance
        assert provenance is not None
        self.assertEqual(
            provenance["genome_population_final_state_sha256"],
            reference.state_sha256,
        )
        self.assertNotIn(
            child.agent_id,
            {
                int(record["agent_id"])
                for record in world.trajectory_records
                if record["action_source"] == RECURRENT_ROLLOUT_ACTION_SOURCE
            },
        )

    def test_conditioned_reset_rewinds_sampling_stream_for_exact_reuse(self) -> None:
        policy = DeterministicPublicRecurrentPolicy(
            _conditioned_model(initialization_seed=111),
            artifact_digest="7" * 64,
            sampling_seed=991,
        )

        def run_once() -> tuple[tuple[str, ...], dict[str, object]]:
            policy.start_world(
                world_identity="conditioned-policy-sampled-reuse",
                genome_stream_seed=69,
                genome_population_mode="heritable",
            )
            world = SimulationWorld(
                _small_policy_world_config(seed=65, max_ticks=4),
                policy=policy,
            )
            world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
            actions = tuple(
                str(record["requested_action"])
                for record in world.trajectory_records
                if record["action_source"] == RECURRENT_ROLLOUT_ACTION_SOURCE
            )
            policy.reset_world()
            provenance = policy.last_world_genome_provenance
            assert provenance is not None
            return actions, provenance

        first_actions, first_provenance = run_once()
        second_actions, second_provenance = run_once()

        self.assertTrue(first_actions)
        self.assertEqual(first_actions, second_actions)
        self.assertEqual(first_provenance, second_provenance)
        self.assertEqual(
            first_provenance["action_selection"],
            PUBLIC_RECURRENT_SAMPLED_SELECTION,
        )
        self.assertEqual(first_provenance["policy_sampling_seed"], 991)

    def test_conditioned_reset_builds_provenance_before_population_mutation(
        self,
    ) -> None:
        policy = DeterministicPublicRecurrentPolicy(
            _conditioned_model(initialization_seed=112),
            artifact_digest="8" * 64,
        )
        policy.start_world(
            world_identity="conditioned-policy-atomic-reset",
            genome_stream_seed=70,
            genome_population_mode="heritable",
        )
        world = SimulationWorld(
            _small_policy_world_config(seed=66, max_ticks=1),
            policy=policy,
        )
        founder = world.alive_agents()[0]

        with patch(
            "evolution_sim.mind.recurrent_policy.stable_payload_digest",
            side_effect=RuntimeError("injected provenance serialization failure"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "injected provenance serialization failure",
            ):
                policy.reset_world()

        child_metadata = policy.child_metadata(
            child_agent_id=founder.agent_id + 1,
            primary_parent_id=founder.agent_id,
            secondary_parent_id=None,
        )
        self.assertEqual(child_metadata["inheritance_kind"], "asexual")
        policy.reset_world()

    def test_real_terminal_death_discards_conditioned_runtime_ownership(self) -> None:
        policy = DeterministicPublicRecurrentPolicy(
            _conditioned_model(initialization_seed=113),
            artifact_digest="5" * 64,
        )
        policy.start_world(
            world_identity="conditioned-policy-terminal",
            genome_stream_seed=71,
            genome_population_mode="heritable",
        )
        world = SimulationWorld(
            _small_policy_world_config(
                seed=67,
                max_ticks=1,
                base_energy_drain=100.0,
            ),
            policy=policy,
        )
        world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
        self.assertFalse(world.alive_agents())
        self.assertTrue(
            any(
                bool(record["after"]["alive"]) is False
                for record in world.trajectory_records
            )
        )
        policy.reset_world()
        provenance = policy.last_world_genome_provenance
        assert provenance is not None
        self.assertEqual(
            provenance["genome_population_final_state_sha256"],
            provenance["genome_population_pre_founder_state_sha256"],
        )

    def test_passive_terminal_record_discards_conditioned_runtime_ownership(
        self,
    ) -> None:
        policy = DeterministicPublicRecurrentPolicy(
            _conditioned_model(initialization_seed=114),
            artifact_digest="a" * 64,
        )
        policy.start_world(
            world_identity="conditioned-policy-passive-terminal",
            genome_stream_seed=72,
            genome_population_mode="heritable",
        )
        world = SimulationWorld(
            _small_policy_world_config(seed=68, max_ticks=1),
            policy=policy,
        )
        founder = world.alive_agents()[0]

        policy.observe_transition(
            {
                "agent_id": founder.agent_id,
                "action_source": "passive",
                "after": {"alive": False},
            }
        )
        with self.assertRaisesRegex(
            RecurrentPolicyAdapterError,
            "no registered recurrent genome",
        ):
            policy.child_metadata(
                child_agent_id=founder.agent_id + 1,
                primary_parent_id=founder.agent_id,
                secondary_parent_id=None,
            )
        policy.reset_world()
        provenance = policy.last_world_genome_provenance
        assert provenance is not None
        self.assertEqual(
            provenance["genome_population_final_state_sha256"],
            provenance["genome_population_pre_founder_state_sha256"],
        )

    def test_conditioned_transition_tamper_and_checkpoint_fail_closed_atomically(
        self,
    ) -> None:
        policy = DeterministicPublicRecurrentPolicy(
            _conditioned_model(initialization_seed=127),
            artifact_digest="6" * 64,
            capture_public_history=True,
        )
        policy.start_world(
            world_identity="conditioned-policy-tamper",
            genome_stream_seed=79,
            genome_population_mode="heritable",
        )
        world = SimulationWorld(
            _small_policy_world_config(seed=73, max_ticks=1),
            policy=policy,
        )
        founder = world.alive_agents()[0]
        observation = world._observe_agent(founder)
        decision = policy.decide(observation, world._action_mask(founder))
        with self.assertRaisesRegex(
            RecurrentPolicyAdapterError,
            "complete validated recurrent-genome population snapshot",
        ):
            policy.diagnostics_checkpoint_state(agent_id=founder.agent_id)
        with self.assertRaisesRegex(
            RecurrentPolicyAdapterError,
            "dead agents with unfinalized decisions",
        ):
            policy.reconcile_live_agent_ids(live_agent_ids=())

        record = {
            "agent_id": founder.agent_id,
            "policy_id": decision.policy_id,
            "policy_version": decision.policy_version,
            "action_source": decision.source,
            "requested_action": decision.requested_action,
            "resolved_action": decision.requested_action,
            "resolution_action_valid": True,
            "moved": decision.requested_action.startswith("move_"),
            "reward": {"total": 0.0},
            "after": {"alive": True},
            "policy_decision_diagnostics": deepcopy(decision.diagnostics),
        }
        tampered = deepcopy(record)
        tampered["policy_decision_diagnostics"]["genome_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            RecurrentPolicyAdapterError,
            "genome_sha256 does not match",
        ):
            policy.observe_transition(tampered)

        policy.observe_transition(record)
        child_metadata = policy.child_metadata(
            child_agent_id=founder.agent_id + 1,
            primary_parent_id=founder.agent_id,
            secondary_parent_id=None,
        )
        self.assertEqual(child_metadata["inheritance_kind"], "asexual")
        policy.reconcile_live_agent_ids(live_agent_ids=(founder.agent_id,))
        policy.reset_world()


if __name__ == "__main__":
    unittest.main()
