from __future__ import annotations

from copy import deepcopy
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    from evolution_sim.env.runtime.action_contract import ACTION_NAMES
    from evolution_sim.mind.recurrent_actor_critic import (
        ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE,
        PREVIOUS_PUBLIC_FEEDBACK_SIZE,
        PublicRecurrentActorCritic,
        RecurrentActorCriticConfig,
    )
    from evolution_sim.mind.recurrent_counterfactual_branch import (
        RECURRENT_COUNTERFACTUAL_BRANCH_SCHEMA_VERSION,
        RECURRENT_COUNTERFACTUAL_TRAINING_USE,
        RecurrentCounterfactualBranchError,
        _discounted_focal_return,
        build_recurrent_counterfactual_branch_row,
        reconstruct_current_model_hidden_from_branch_row,
        validate_recurrent_counterfactual_branch_row,
        verified_source_recurrent_state_from_branch_row,
    )
    from evolution_sim.mind.recurrent_policy import recurrent_model_state_sha256
    from evolution_sim.mind.recurrent_seed_registry import RECURRENT_SEED_REGISTRY


@unittest.skipIf(torch is None, "optional Mind ML dependency torch is not installed")
class RecurrentCounterfactualBranchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert torch is not None
        torch.set_num_threads(1)
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(
                encoder_size=16,
                hidden_size=16,
                recurrent_layers=1,
            ),
            initialization_seed=47,
        )
        cls.source_model = model
        cls.row = build_recurrent_counterfactual_branch_row(
            model,
            artifact_digest="a" * 64,
            seed_role="curriculum",
            environment_seed=RECURRENT_SEED_REGISTRY["curriculum"][0],
            scenario="carrion_only",
            branch_tick=2,
            horizon_ticks=2,
            policy_sampling_seed=991,
            gamma=0.99,
            verify_replay=True,
        )

    def test_real_learner_state_enumerates_all_valid_actions_with_exact_replay(
        self,
    ) -> None:
        row = self.row
        self.assertEqual(
            row["schema_version"],
            RECURRENT_COUNTERFACTUAL_BRANCH_SCHEMA_VERSION,
        )
        validate_recurrent_counterfactual_branch_row(row)
        context = row["trainable_public_context"]
        mask = context["current_public_action_mask"]
        expected_actions = [action for action in ACTION_NAMES if mask[action]]
        outcomes = row["labels"]["action_outcomes"]
        self.assertEqual(
            [outcome["action"] for outcome in outcomes],
            expected_actions,
        )
        self.assertGreater(len(outcomes), 1)
        source_action = row["labels"]["source_requested_action"]
        for outcome in outcomes:
            self.assertEqual(outcome["intervention_count"], 1)
            self.assertEqual(
                outcome["natural_requested_action"],
                source_action,
            )
            self.assertEqual(
                outcome["first_transition"]["requested_action"],
                outcome["action"],
            )
            self.assertTrue(outcome["first_transition"]["observation_action_valid"])
            self.assertTrue(outcome["replay_verified"])
            self.assertEqual(
                outcome["evidence_digest"],
                outcome["replay_evidence_digest"],
            )
            self.assertEqual(outcome["heuristic_action_source_count"], 0)
            self.assertEqual(outcome["unsupported_requested_action_count"], 0)
            self.assertEqual(outcome["unexpected_action_source_count"], 0)
            self.assertEqual(
                outcome["focal_transition_count"],
                outcome["focal_policy_decision_count"]
                + outcome["focal_passive_transition_count"],
            )

        source_outcome = next(
            outcome for outcome in outcomes if outcome["action"] == source_action
        )
        self.assertTrue(row["labels"]["baseline_source_action_behavior_match"])
        self.assertEqual(
            source_outcome["behavior_digest"],
            row["labels"]["baseline"]["behavior_digest"],
        )
        self.assertEqual(
            source_outcome["paired_vs_baseline"]["focal_discounted_return_delta"],
            0.0,
        )

    def test_passive_terminal_reward_remains_in_focal_discounted_return(self) -> None:
        records = (
            {
                "agent_id": 9,
                "action_source": "public_recurrent_rollout",
                "reward": {"total": 0.5},
            },
            {
                "agent_id": 11,
                "action_source": "public_recurrent_rollout",
                "reward": {"total": 99.0},
            },
            {
                "agent_id": 9,
                "action_source": "passive",
                "reward": {"total": -1.0},
            },
        )

        focal_records, discounted_return, passive_count = _discounted_focal_return(
            records,
            focal_agent_id=9,
            gamma=0.9,
        )

        self.assertEqual(len(focal_records), 2)
        self.assertEqual(passive_count, 1)
        self.assertAlmostEqual(discounted_return, -0.4, places=12)

    def test_public_optimizer_label_and_metadata_surfaces_are_separated(self) -> None:
        row = self.row
        context = row["trainable_public_context"]
        self.assertEqual(
            set(context),
            {
                "public_history_prefix",
                "current_public_observation",
                "current_public_action_mask",
                "previous_public_feedback",
            },
        )
        history = context["public_history_prefix"]
        self.assertEqual(history["record_count"], 2)
        self.assertEqual(len(history["records"]), 2)
        self.assertTrue(history["records"][0]["episode_start"])
        self.assertFalse(history["records"][1]["episode_start"])
        for record in history["records"]:
            self.assertEqual(
                set(record),
                {
                    "public_observation",
                    "public_action_mask",
                    "previous_public_feedback",
                    "episode_start",
                    "recurrent_state_reset_before_decision",
                },
            )
        self.assertEqual(
            len(context["current_public_observation"]["values"]),
            ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE,
        )
        self.assertEqual(
            len(context["previous_public_feedback"]["values"]),
            PREVIOUS_PUBLIC_FEEDBACK_SIZE,
        )
        optimizer = row["optimizer_context"]
        self.assertTrue(optimizer["derived_from_public_history"])
        self.assertTrue(optimizer["source_artifact_match_required"])
        self.assertEqual(optimizer["source_artifact_digest"], "a" * 64)
        self.assertEqual(
            optimizer["source_model_state_sha256"],
            recurrent_model_state_sha256(self.source_model),
        )
        self.assertEqual(
            optimizer["stored_state_usage"],
            "exact_source_artifact_verification_only",
        )
        self.assertEqual(
            optimizer["current_model_state_policy"],
            "reconstruct_from_trainable_public_history_prefix",
        )
        self.assertFalse(optimizer["runtime_environment_input"])
        recurrent_magnitude = sum(
            abs(value)
            for layer in optimizer["source_recurrent_state"]
            for batch in layer
            for value in batch
        )
        self.assertGreater(recurrent_magnitude, 0.0)
        self.assertNotEqual(
            sum(context["previous_public_feedback"]["values"]),
            0.0,
        )
        self.assertFalse(row["metadata"]["private_checkpoint_serialized"])
        self.assertEqual(row["metadata"]["seed_role"], "curriculum")
        self.assertEqual(row["metadata"]["scenario"], "carrion_only")
        self.assertFalse(row["contract"]["metadata_used_as_actor_input"])
        self.assertFalse(row["contract"]["outcome_labels_used_as_actor_input"])
        self.assertFalse(row["contract"]["ordinary_decide_calls_override"])
        self.assertEqual(
            row["contract"]["historical_row_training_use"],
            RECURRENT_COUNTERFACTUAL_TRAINING_USE,
        )
        self.assertTrue(row["contract"]["ppo_ratio_data_use_forbidden"])
        self.assertEqual(
            row["contract"]["target_estimand"],
            "single_exact_rollout_on_one_deepcopied_sequential_rng_tape",
        )
        self.assertFalse(row["contract"]["expected_causal_effect_estimated"])
        self.assertFalse(row["contract"]["continuation_outcome_uncertainty_estimated"])
        self.assertFalse(row["contract"]["event_aligned_common_random_numbers"])
        continuation = row["metadata"]["continuation_provenance"]
        self.assertEqual(
            row["metadata"]["environment_seed"],
            RECURRENT_SEED_REGISTRY["curriculum"][0],
        )
        self.assertEqual(row["metadata"]["policy_sampling_seed"], 991)
        self.assertEqual(continuation["branch_horizon_ticks"], 2)
        self.assertEqual(
            continuation["exact_replay_repeat_count_per_action"],
            2,
        )
        self.assertEqual(continuation["repeat_policy_sampling_seeds"], [991, 991])
        for flag in (
            "training_ran",
            "training_artifact_created",
            "runtime_artifact_created",
            "runtime_action_selection_changed",
            "promotion_authorized",
        ):
            self.assertFalse(row[flag])

    def test_validation_rejects_metadata_leakage_and_digest_tampering(self) -> None:
        tampered = deepcopy(self.row)
        tampered["trainable_public_context"]["environment_seed"] = 7
        with self.assertRaisesRegex(
            RecurrentCounterfactualBranchError,
            "field set drifted",
        ):
            validate_recurrent_counterfactual_branch_row(tampered)

        tampered = deepcopy(self.row)
        tampered["labels"]["action_outcomes"][0]["focal_discounted_return"] = 9.0
        with self.assertRaisesRegex(
            RecurrentCounterfactualBranchError,
            "component digest mismatch",
        ):
            validate_recurrent_counterfactual_branch_row(tampered)

    def test_public_prefix_reconstructs_source_and_updated_model_state_safely(
        self,
    ) -> None:
        source_state = verified_source_recurrent_state_from_branch_row(
            self.source_model,
            self.row,
            artifact_digest="a" * 64,
        )
        reconstructed = reconstruct_current_model_hidden_from_branch_row(
            self.source_model,
            self.row,
        )
        torch.testing.assert_close(
            reconstructed,
            source_state,
            rtol=0.0,
            atol=1.0e-6,
        )

        context = self.row["trainable_public_context"]
        observation = torch.tensor(
            context["current_public_observation"]["values"],
            dtype=torch.float32,
        ).reshape(1, 1, -1)
        action_mask = torch.tensor(
            [context["current_public_action_mask"][action] for action in ACTION_NAMES],
            dtype=torch.bool,
        ).reshape(1, 1, -1)
        feedback = torch.tensor(
            context["previous_public_feedback"]["values"],
            dtype=torch.float32,
        ).reshape(1, 1, -1)
        with torch.no_grad():
            output = self.source_model.forward_sequence(
                observation,
                action_mask,
                feedback,
                initial_state=reconstructed,
            )
            probabilities = torch.softmax(
                output.masked_logits[0, 0],
                dim=-1,
            )
        stored_distribution = self.row["labels"]["source_behavior_distribution"]
        for index, action in enumerate(ACTION_NAMES):
            self.assertAlmostEqual(
                float(probabilities[index].item()),
                stored_distribution["probabilities"][action],
                places=6,
            )
            if context["current_public_action_mask"][action]:
                self.assertAlmostEqual(
                    float(output.masked_logits[0, 0, index].item()),
                    stored_distribution["masked_logits"][action],
                    places=6,
                )

        updated_model = deepcopy(self.source_model)
        with torch.no_grad():
            updated_model.recurrent.weight_ih_l0.add_(0.01)
        updated_state = reconstruct_current_model_hidden_from_branch_row(
            updated_model,
            self.row,
        )
        self.assertEqual(updated_state.shape, reconstructed.shape)
        self.assertTrue(torch.isfinite(updated_state).all())
        self.assertFalse(torch.allclose(updated_state, source_state))
        with self.assertRaisesRegex(
            RecurrentCounterfactualBranchError,
            "exact artifact verification",
        ):
            verified_source_recurrent_state_from_branch_row(
                updated_model,
                self.row,
                artifact_digest="a" * 64,
            )
        with self.assertRaisesRegex(
            RecurrentCounterfactualBranchError,
            "exact artifact verification",
        ):
            verified_source_recurrent_state_from_branch_row(
                self.source_model,
                self.row,
                artifact_digest="b" * 64,
            )

    def test_public_prefix_rejects_actor_input_metadata_leakage(self) -> None:
        tampered = deepcopy(self.row)
        tampered["trainable_public_context"]["public_history_prefix"]["records"][0][
            "environment_seed"
        ] = 7
        with self.assertRaisesRegex(
            RecurrentCounterfactualBranchError,
            "forbidden keys",
        ):
            validate_recurrent_counterfactual_branch_row(tampered)

    def test_query_rejects_non_training_seed_roles_and_cross_role_seeds(self) -> None:
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(encoder_size=8, hidden_size=8),
            initialization_seed=3,
        )
        with self.assertRaisesRegex(
            RecurrentCounterfactualBranchError,
            "only train or curriculum",
        ):
            build_recurrent_counterfactual_branch_row(
                model,
                artifact_digest="b" * 64,
                seed_role="validation",
                environment_seed=RECURRENT_SEED_REGISTRY["validation"][0],
                scenario="broad",
                branch_tick=0,
                horizon_ticks=1,
            )
        with self.assertRaisesRegex(
            RecurrentCounterfactualBranchError,
            "not registered",
        ):
            build_recurrent_counterfactual_branch_row(
                model,
                artifact_digest="b" * 64,
                seed_role="train",
                environment_seed=RECURRENT_SEED_REGISTRY["curriculum"][0],
                scenario="broad",
                branch_tick=0,
                horizon_ticks=1,
            )


if __name__ == "__main__":
    unittest.main()
