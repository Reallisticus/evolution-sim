from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    from evolution_sim.config.schema import SignalConfig, WorldConfig
    from evolution_sim.env.runtime.action_contract import ACTION_NAMES
    from evolution_sim.env.runtime.state import RunMode
    from evolution_sim.env.world import SimulationWorld
    from evolution_sim.mind.recurrent_actor_critic import (
        PublicRecurrentActorCritic,
        RecurrentActorCriticConfig,
    )
    from evolution_sim.mind.recurrent_policy import (
        PUBLIC_RECURRENT_DISTRIBUTION_DIAGNOSTIC_SCHEMA_VERSION,
        PUBLIC_RECURRENT_SAMPLED_SELECTION,
        RECURRENT_COUNTERFACTUAL_ACTION_SOURCE,
        DeterministicPublicRecurrentPolicy,
        RecurrentPolicyAdapterError,
        frozen_cpu_model_copy,
        validate_public_recurrent_distribution_diagnostics,
        validate_public_recurrent_history_prefix,
    )
    from evolution_sim.mind.recurrent_rollout import RECURRENT_ROLLOUT_ACTION_SOURCE


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


if __name__ == "__main__":
    unittest.main()
