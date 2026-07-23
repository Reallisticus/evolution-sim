from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import random
import statistics
import unittest
from unittest.mock import patch

from python.tests.runtime_test_helpers import RuntimeContractTestHelpers

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    from evolution_sim.config import CombatConfig, ReproductionConfig, WorldConfig
    from evolution_sim.env import SimulationWorld
    from evolution_sim.env.runtime.action_contract import ACTION_NAMES
    from evolution_sim.mind import recurrent_counterfactual_branch as branch_module
    from evolution_sim.mind.provenance import stable_payload_digest
    from evolution_sim.mind.recurrent_actor_critic import (
        ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE,
        PREVIOUS_PUBLIC_FEEDBACK_SIZE,
        PublicRecurrentActorCritic,
        RecurrentActorCriticConfig,
    )
    from evolution_sim.mind.recurrent_counterfactual_branch import (
        RECURRENT_COUNTERFACTUAL_BRANCH_SCHEMA_VERSION,
        RECURRENT_COUNTERFACTUAL_CONTINUATION_ENVIRONMENT_TAPE_SEED_NAMESPACE,
        RECURRENT_COUNTERFACTUAL_CONTINUATION_POLICY_TAPE_SEED_NAMESPACE,
        RECURRENT_COUNTERFACTUAL_CONTINUATION_RNG_RETAPE_BOUNDARY,
        RECURRENT_COUNTERFACTUAL_TRAINING_USE,
        RecurrentCounterfactualBranchError,
        _discounted_focal_return,
        build_recurrent_counterfactual_branch_row,
        build_recurrent_counterfactual_nested_horizon_materialization,
        derive_recurrent_counterfactual_tape_seed,
        reconstruct_current_model_hidden_from_branch_row,
        validate_recurrent_counterfactual_aggregate_row,
        validate_recurrent_counterfactual_branch_row,
        verified_source_recurrent_state_from_branch_row,
    )
    from evolution_sim.mind.recurrent_policy import (
        DeterministicPublicRecurrentPolicy,
        recurrent_model_state_sha256,
    )
    from evolution_sim.mind.recurrent_seed_registry import (
        RECURRENT_SEED_REGISTRY,
        SCALE_DEVELOPMENT_SEED_REGISTRY,
    )


@unittest.skipIf(torch is None, "optional Mind ML dependency torch is not installed")
class RecurrentCounterfactualBranchTests(RuntimeContractTestHelpers):
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

    def _decision_boundary_fixture_model(self) -> PublicRecurrentActorCritic:
        assert torch is not None
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(
                encoder_size=8,
                hidden_size=8,
                recurrent_layers=1,
            ),
            initialization_seed=1,
        )
        with torch.no_grad():
            model.actor.weight.zero_()
            model.actor.bias.fill_(-100.0)
            model.actor.bias[ACTION_NAMES.index("stay")] = 0.0
            model.actor.bias[ACTION_NAMES.index("attack_north")] = 0.0
        return model

    def _decision_boundary_fixture_world(
        self,
        *,
        scenario: str,
        environment_seed: int,
        ticks: int,
        policy: DeterministicPublicRecurrentPolicy,
    ) -> SimulationWorld:
        self.assertEqual(scenario, "carrion_only")
        self.assertEqual(environment_seed, 887_847_623)
        world = SimulationWorld(
            WorldConfig(
                seed=environment_seed,
                max_ticks=ticks,
                width=3,
                height=3,
                initial_agents=0,
                max_agents=10,
                water_tile_ratio=0.0,
                forest_tile_ratio=0.0,
                wetland_tile_ratio=0.0,
                rocky_tile_ratio=0.0,
                base_energy_drain=0.0,
                base_hydration_drain=0.0,
                combat=CombatConfig(
                    min_attack_health_ratio=0.0,
                    min_attack_energy_ratio=0.0,
                    min_attack_hydration_ratio=0.0,
                    base_attack_damage=3.0,
                ),
                reproduction=ReproductionConfig(min_age=720),
            ),
            policy=policy,
        )
        hunter = self._hunter_genome()
        prey = replace(
            hunter,
            max_health=0.7,
            defense_rating=0.45,
            attack_power=0.35,
            plant_bias=1.8,
            live_prey_bias=0.2,
            carrion_bias=0.2,
        )
        attacker = self._place_ready_agent(
            world,
            x=1,
            y=1,
            lineage_id=1,
            genome=hunter,
        )
        focal = self._place_ready_agent(
            world,
            x=1,
            y=0,
            lineage_id=2,
            genome=prey,
        )
        self.assertEqual((attacker.agent_id, focal.agent_id), (1, 2))
        focal.health = 0.05
        world.current_species_map = {
            agent.agent_id: agent.lineage_id for agent in world.alive_agents()
        }
        world.agent_last_species_map = world.current_species_map.copy()
        return world

    def _materialize_decision_boundary_fixture(
        self,
        *,
        tape_count: int = 2,
    ) -> dict[str, object]:
        self.assertEqual(RECURRENT_SEED_REGISTRY["curriculum"][0], 887_847_623)
        model = self._decision_boundary_fixture_model()
        with patch.object(
            branch_module,
            "_world_for_scenario",
            side_effect=self._decision_boundary_fixture_world,
        ):
            return build_recurrent_counterfactual_nested_horizon_materialization(
                model,
                artifact_digest="a" * 64,
                seed_role="curriculum",
                environment_seed=887_847_623,
                scenario="carrion_only",
                branch_tick_candidates=(0,),
                horizons=(1, 2),
                source_policy_sampling_seed=1,
                branch_selection_seed=4,
                continuation_tape_count=tape_count,
                continuation_tape_identity="pretape-death:0",
                terminal_target_world_tick=3,
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
        def rehash(
            row: dict[str, object],
            *,
            component: str,
        ) -> None:
            row["component_digests"][component] = stable_payload_digest(row[component])
            row.pop("exact_digest", None)
            row["exact_digest"] = stable_payload_digest(row)

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
            "paired delta|component digest mismatch",
        ):
            validate_recurrent_counterfactual_branch_row(tampered)

        top_level_private = deepcopy(self.row)
        top_level_private["private_world_checkpoint"] = {"forbidden": True}
        top_level_private.pop("exact_digest")
        top_level_private["exact_digest"] = stable_payload_digest(top_level_private)
        with self.assertRaisesRegex(
            RecurrentCounterfactualBranchError,
            "branch row field set drifted",
        ):
            validate_recurrent_counterfactual_branch_row(top_level_private)

        label_private = deepcopy(self.row)
        label_private["labels"]["private_world_checkpoint"] = {"forbidden": True}
        label_private["component_digests"]["labels"] = stable_payload_digest(
            label_private["labels"]
        )
        label_private.pop("exact_digest")
        label_private["exact_digest"] = stable_payload_digest(label_private)
        with self.assertRaisesRegex(
            RecurrentCounterfactualBranchError,
            "branch labels field set drifted",
        ):
            validate_recurrent_counterfactual_branch_row(label_private)

        for public_field in (
            "current_public_observation",
            "previous_public_feedback",
        ):
            with self.subTest(public_field=public_field):
                nested_private = deepcopy(self.row)
                nested_private["trainable_public_context"][public_field]["opaque"] = {
                    "secret_world_snapshot": [1, 2, 3]
                }
                rehash(nested_private, component="trainable_public_context")
                with self.assertRaisesRegex(
                    RecurrentCounterfactualBranchError,
                    "field set drifted",
                ):
                    validate_recurrent_counterfactual_branch_row(nested_private)

        for public_field in (
            "current_public_observation",
            "previous_public_feedback",
        ):
            with self.subTest(shape_type_alias_field=public_field):
                shape_alias = deepcopy(self.row)
                expected_size = shape_alias["trainable_public_context"][
                    public_field
                ]["shape"][0]
                shape_alias["trainable_public_context"][public_field][
                    "shape"
                ] = [float(expected_size)]
                rehash(shape_alias, component="trainable_public_context")
                with self.assertRaisesRegex(
                    RecurrentCounterfactualBranchError,
                    "integer size",
                ):
                    validate_recurrent_counterfactual_branch_row(shape_alias)

        feedback_tampered = deepcopy(self.row)
        feedback_tampered["trainable_public_context"][
            "previous_public_feedback"
        ]["values"][0] = 0.5
        rehash(feedback_tampered, component="trainable_public_context")
        with self.assertRaisesRegex(
            RecurrentCounterfactualBranchError,
            "semantic contract",
        ):
            validate_recurrent_counterfactual_branch_row(feedback_tampered)

        empty_history_nonzero_feedback = deepcopy(self.row)
        prefix = empty_history_nonzero_feedback["trainable_public_context"][
            "public_history_prefix"
        ]
        prefix["record_count"] = 0
        prefix["records"] = []
        feedback_values = empty_history_nonzero_feedback[
            "trainable_public_context"
        ]["previous_public_feedback"]["values"]
        feedback_values[:] = [0.0] * len(feedback_values)
        feedback_values[0] = 1.0
        feedback_values[len(ACTION_NAMES)] = 1.0
        feedback_values[-3] = 1.0
        rehash(
            empty_history_nonzero_feedback,
            component="trainable_public_context",
        )
        with self.assertRaisesRegex(
            RecurrentCounterfactualBranchError,
            "all zero when the public history is empty",
        ):
            validate_recurrent_counterfactual_branch_row(
                empty_history_nonzero_feedback
            )

        for name, mutate, expected_error in (
            (
                "continuation horizon bool",
                lambda continuation: continuation.__setitem__(
                    "branch_horizon_ticks", True
                ),
                "positive integer",
            ),
            (
                "continuation repeat-index bool aliases",
                lambda continuation: continuation.__setitem__(
                    "repeat_indices", [False, True]
                ),
                "non-negative integer",
            ),
            (
                "continuation sampling-seed bool aliases",
                lambda continuation: continuation.__setitem__(
                    "repeat_policy_sampling_seeds", [True, True]
                ),
                "sampling seed",
            ),
        ):
            with self.subTest(name=name):
                continuation_tampered = deepcopy(self.row)
                continuation = continuation_tampered["metadata"][
                    "continuation_provenance"
                ]
                mutate(continuation)
                continuation_tampered["metadata"][
                    "continuation_provenance_digest"
                ] = stable_payload_digest(continuation)
                rehash(continuation_tampered, component="metadata")
                with self.assertRaisesRegex(
                    RecurrentCounterfactualBranchError,
                    expected_error,
                ):
                    validate_recurrent_counterfactual_branch_row(
                        continuation_tampered
                    )

    def test_coherently_rehashed_compact_source_label_tampering_fails_closed(
        self,
    ) -> None:
        def rehash_labels(row: dict[str, object]) -> None:
            row["component_digests"]["labels"] = stable_payload_digest(row["labels"])
            row.pop("exact_digest", None)
            row["exact_digest"] = stable_payload_digest(row)

        def living_focal_in_empty_population(row: dict[str, object]) -> None:
            baseline = row["labels"]["baseline"]
            outcome = next(
                candidate
                for candidate in row["labels"]["action_outcomes"]
                if candidate["action"]
                != row["labels"]["source_requested_action"]
                and candidate["focal_terminal"]["alive"] is True
            )
            outcome["population_alive"] = 0
            outcome["paired_vs_baseline"]["population_alive_delta"] = -int(
                baseline["population_alive"]
            )

        def focal_revives_after_first_transition_death(
            row: dict[str, object],
        ) -> None:
            outcome = next(
                candidate
                for candidate in row["labels"]["action_outcomes"]
                if candidate["action"]
                != row["labels"]["source_requested_action"]
                and candidate["focal_terminal"]["alive"] is True
                and candidate["first_transition"]["died"] is False
            )
            outcome["first_transition"]["died"] = True
            outcome["first_transition"]["after_alive"] = False

        def reproduction_without_horizon_birth(row: dict[str, object]) -> None:
            outcome = row["labels"]["action_outcomes"][0]
            outcome["first_transition"]["reproduced"] = True
            outcome["births_during_horizon"] = 0

        def death_without_horizon_death(row: dict[str, object]) -> None:
            outcome = row["labels"]["action_outcomes"][0]
            outcome["first_transition"]["died"] = True
            outcome["first_transition"]["after_alive"] = False
            outcome["focal_terminal"]["alive"] = False
            for ratio_field in (
                "energy_ratio",
                "hydration_ratio",
                "health_ratio",
            ):
                outcome["focal_terminal"][ratio_field] = None
            outcome["deaths_during_horizon"] = 0

        def transitions_after_first_transition_death(
            row: dict[str, object],
        ) -> None:
            outcome = row["labels"]["action_outcomes"][0]
            outcome["first_transition"]["died"] = True
            outcome["first_transition"]["after_alive"] = False
            outcome["focal_terminal"]["alive"] = False
            for ratio_field in (
                "energy_ratio",
                "hydration_ratio",
                "health_ratio",
            ):
                outcome["focal_terminal"][ratio_field] = None
            outcome["deaths_during_horizon"] = max(
                1,
                outcome["deaths_during_horizon"],
            )
            outcome["focal_transition_count"] = 2
            outcome["focal_policy_decision_count"] = 2
            outcome["focal_passive_transition_count"] = 0

        def terminal_death_without_horizon_death(
            row: dict[str, object],
        ) -> None:
            outcome = row["labels"]["action_outcomes"][0]
            outcome["first_transition"]["died"] = False
            outcome["first_transition"]["after_alive"] = True
            for ratio_field in (
                "after_energy_ratio",
                "after_hydration_ratio",
                "after_health_ratio",
            ):
                outcome["first_transition"][ratio_field] = 1.0
            outcome["focal_terminal"]["alive"] = False
            for ratio_field in (
                "energy_ratio",
                "hydration_ratio",
                "health_ratio",
            ):
                outcome["focal_terminal"][ratio_field] = None
            outcome["deaths_during_horizon"] = 0

        def transition_count_beyond_horizon(row: dict[str, object]) -> None:
            outcome = row["labels"]["action_outcomes"][0]
            excessive_count = row["metadata"]["horizon_ticks"] + 1
            outcome["focal_transition_count"] = excessive_count
            outcome["focal_policy_decision_count"] = excessive_count
            outcome["focal_passive_transition_count"] = 0

        def passive_transition_with_living_terminal(
            row: dict[str, object],
        ) -> None:
            outcome = next(
                candidate
                for candidate in row["labels"]["action_outcomes"]
                if candidate["focal_terminal"]["alive"] is True
            )
            outcome["focal_transition_count"] = 2
            outcome["focal_policy_decision_count"] = 1
            outcome["focal_passive_transition_count"] = 1

        mutations = (
            (
                "action fractional population",
                lambda row: row["labels"]["action_outcomes"][0].__setitem__(
                    "population_alive", 1.5
                ),
                "non-negative integer",
            ),
            (
                "baseline fractional population",
                lambda row: row["labels"]["baseline"].__setitem__(
                    "population_alive", 1.5
                ),
                "non-negative integer",
            ),
            (
                "terminal bool masquerading as int",
                lambda row: row["labels"]["action_outcomes"][0][
                    "focal_terminal"
                ].__setitem__("alive", 1),
                "exact boolean",
            ),
            (
                "forged paired population delta",
                lambda row: row["labels"]["action_outcomes"][0][
                    "paired_vs_baseline"
                ].__setitem__("population_alive_delta", 999),
                "paired delta",
            ),
            (
                "paired bool masquerading as integer delta",
                lambda row: next(
                    outcome
                    for outcome in row["labels"]["action_outcomes"]
                    if outcome["action"]
                    == row["labels"]["source_requested_action"]
                )["paired_vs_baseline"].__setitem__(
                    "population_alive_delta", False
                ),
                "exact integer",
            ),
            (
                "bool intervention count",
                lambda row: row["labels"]["action_outcomes"][0].__setitem__(
                    "intervention_count", True
                ),
                "positive integer",
            ),
            (
                "nonzero unexpected action source",
                lambda row: row["labels"]["action_outcomes"][0].__setitem__(
                    "unexpected_action_source_count", 1
                ),
                "unsupported action source",
            ),
            (
                "bool unsupported action source count",
                lambda row: row["labels"]["action_outcomes"][0].__setitem__(
                    "unsupported_requested_action_count", False
                ),
                "non-negative integer",
            ),
            (
                "bool valid action count",
                lambda row: row["labels"].__setitem__("valid_action_count", True),
                "positive integer",
            ),
            (
                "non-source survival flag drift",
                lambda row: next(
                    outcome
                    for outcome in row["labels"]["action_outcomes"]
                    if outcome["action"]
                    != row["labels"]["source_requested_action"]
                    and outcome["first_transition"]["died"] is False
                )["first_transition"].__setitem__("after_alive", False),
                "death and alive flags",
            ),
            (
                "non-source successful movement flag drift",
                lambda row: next(
                    outcome
                    for outcome in row["labels"]["action_outcomes"]
                    if outcome["action"]
                    != row["labels"]["source_requested_action"]
                    and outcome["first_transition"]["resolution_action_valid"] is True
                    and str(
                        outcome["first_transition"]["resolved_action"]
                    ).startswith("move_")
                )["first_transition"].__setitem__("moved", False),
                "movement flags",
            ),
            (
                "non-source finite reward outside canonical bounds",
                lambda row: next(
                    outcome
                    for outcome in row["labels"]["action_outcomes"]
                    if outcome["action"]
                    != row["labels"]["source_requested_action"]
                )["first_transition"].__setitem__("reward_total", 1.0e300),
                "outside canonical bounds",
            ),
            (
                "alive first-transition ratio outside unit interval",
                lambda row: next(
                    outcome
                    for outcome in row["labels"]["action_outcomes"]
                    if outcome["first_transition"]["after_alive"] is True
                )["first_transition"].__setitem__(
                    "after_energy_ratio", -123.0
                ),
                "must be non-negative",
            ),
            (
                "alive terminal ratio below zero",
                lambda row: next(
                    outcome
                    for outcome in row["labels"]["action_outcomes"]
                    if outcome["focal_terminal"]["alive"] is True
                )["focal_terminal"].__setitem__("health_ratio", -1.5),
                "must be non-negative",
            ),
            (
                "living focal in empty population",
                living_focal_in_empty_population,
                "living focal agent in an empty population",
            ),
            (
                "focal revival after first-transition death",
                focal_revives_after_first_transition_death,
                "cannot revive",
            ),
            (
                "reproduction without horizon birth",
                reproduction_without_horizon_birth,
                "requires at least one horizon birth",
            ),
            (
                "death without horizon death",
                death_without_horizon_death,
                "requires at least one horizon death",
            ),
            (
                "transitions after first-transition death",
                transitions_after_first_transition_death,
                "must terminate focal transitions",
            ),
            (
                "terminal death without horizon death",
                terminal_death_without_horizon_death,
                "terminal-dead focal outcome requires at least one horizon death",
            ),
            (
                "focal transitions beyond horizon",
                transition_count_beyond_horizon,
                "cannot exceed its horizon",
            ),
            (
                "passive transition with living terminal",
                passive_transition_with_living_terminal,
                "passive focal transition requires terminal death",
            ),
            (
                "discounted return outside feasible reward bounds",
                lambda row: row["labels"]["action_outcomes"][0].__setitem__(
                    "focal_discounted_return",
                    1000.0,
                ),
                "outside feasible reward bounds",
            ),
        )
        for name, mutate, expected_error in mutations:
            with self.subTest(name=name):
                tampered = deepcopy(self.row)
                mutate(tampered)
                rehash_labels(tampered)
                with self.assertRaisesRegex(
                    RecurrentCounterfactualBranchError,
                    expected_error,
                ):
                    validate_recurrent_counterfactual_branch_row(tampered)

        optimizer_tampered = deepcopy(self.row)
        optimizer_tampered["optimizer_context"]["source_recurrent_state"][0][0][
            0
        ] += 0.125
        optimizer_tampered["optimizer_context"][
            "source_recurrent_state_sha256"
        ] = "0" * 64
        optimizer_tampered["component_digests"]["optimizer_context"] = (
            stable_payload_digest(optimizer_tampered["optimizer_context"])
        )
        optimizer_tampered.pop("exact_digest")
        optimizer_tampered["exact_digest"] = stable_payload_digest(
            optimizer_tampered
        )
        with self.assertRaisesRegex(
            RecurrentCounterfactualBranchError,
            "does not match its stored values",
        ):
            validate_recurrent_counterfactual_branch_row(optimizer_tampered)

        flattened_state = deepcopy(self.row)
        source_state = flattened_state["optimizer_context"][
            "source_recurrent_state"
        ]
        flattened_state["optimizer_context"]["source_recurrent_state"] = [
            item
            for layer in source_state
            for batch in layer
            for item in batch
        ]
        flattened_state["component_digests"]["optimizer_context"] = (
            stable_payload_digest(flattened_state["optimizer_context"])
        )
        flattened_state.pop("exact_digest")
        flattened_state["exact_digest"] = stable_payload_digest(flattened_state)
        with self.assertRaisesRegex(
            RecurrentCounterfactualBranchError,
            "declared shape",
        ):
            validate_recurrent_counterfactual_branch_row(flattened_state)

        source_control_tampered = deepcopy(self.row)
        source_action = source_control_tampered["labels"]["source_requested_action"]
        source_outcome = next(
            outcome
            for outcome in source_control_tampered["labels"]["action_outcomes"]
            if outcome["action"] == source_action
        )
        source_outcome["population_alive"] += 1
        source_outcome["paired_vs_baseline"]["population_alive_delta"] += 1
        rehash_labels(source_control_tampered)
        with self.assertRaisesRegex(
            RecurrentCounterfactualBranchError,
            "negative control",
        ):
            validate_recurrent_counterfactual_branch_row(source_control_tampered)

    def test_coherently_rehashed_contract_type_aliases_fail_closed(self) -> None:
        def rehash_contract(row: dict[str, object]) -> None:
            contract = row["contract"]
            metadata = row["metadata"]
            optimizer = row["optimizer_context"]
            protocol_digest = stable_payload_digest(contract)
            metadata["branch_protocol_digest"] = protocol_digest
            continuation = metadata["continuation_provenance"]
            continuation_digest = stable_payload_digest(continuation)
            metadata["continuation_provenance_digest"] = continuation_digest
            metadata["branch_identity_digest"] = stable_payload_digest(
                {
                    "protocol_digest": protocol_digest,
                    "artifact_digest": metadata["source_artifact_digest"],
                    "model_state_sha256": metadata["source_model_state_sha256"],
                    "seed_role": metadata["seed_role"],
                    "environment_seed": metadata["environment_seed"],
                    "policy_sampling_seed": metadata["policy_sampling_seed"],
                    "scenario": metadata["scenario"],
                    "branch_tick": metadata["branch_tick"],
                    "horizon_ticks": metadata["horizon_ticks"],
                    "exact_replay_repeat_count_per_action": continuation[
                        "exact_replay_repeat_count_per_action"
                    ],
                    "focal_agent_id": metadata["focal_agent_id"],
                    "source_record_digest": metadata["source_record_digest"],
                    "source_recurrent_state_sha256": optimizer[
                        "source_recurrent_state_sha256"
                    ],
                    "source_sampling_state_sha256": metadata[
                        "source_sampling_state_sha256"
                    ],
                    "source_public_history_prefix_sha256": metadata[
                        "source_public_history_prefix_sha256"
                    ],
                    "continuation_provenance_digest": continuation_digest,
                }
            )
            row["component_digests"]["contract"] = protocol_digest
            row["component_digests"]["metadata"] = stable_payload_digest(metadata)
            row.pop("exact_digest", None)
            row["exact_digest"] = stable_payload_digest(row)

        for name, field, value in (
            ("boolean as integer", "focal_intervention_count_per_action", True),
            ("integer as boolean", "metadata_used_as_actor_input", 0),
        ):
            with self.subTest(name=name):
                tampered = deepcopy(self.row)
                tampered["contract"][field] = value
                rehash_contract(tampered)
                with self.assertRaisesRegex(
                    RecurrentCounterfactualBranchError,
                    "protocol contract drifted",
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
        self.assertTrue(torch.equal(reconstructed, source_state))

        forged_state_row = deepcopy(self.row)
        forged_optimizer = forged_state_row["optimizer_context"]
        forged_state = torch.tensor(
            forged_optimizer["source_recurrent_state"],
            dtype=torch.float32,
        )
        forged_state[0, 0, 0] += 0.25
        forged_optimizer["source_recurrent_state"] = forged_state.tolist()
        forged_optimizer["source_recurrent_state_sha256"] = hashlib.sha256(
            bytes(forged_state.contiguous().view(torch.uint8).reshape(-1).tolist())
        ).hexdigest()
        forged_metadata = forged_state_row["metadata"]
        forged_continuation = forged_metadata["continuation_provenance"]
        forged_metadata["branch_identity_digest"] = stable_payload_digest(
            {
                "protocol_digest": forged_metadata["branch_protocol_digest"],
                "artifact_digest": forged_metadata["source_artifact_digest"],
                "model_state_sha256": forged_metadata["source_model_state_sha256"],
                "seed_role": forged_metadata["seed_role"],
                "environment_seed": forged_metadata["environment_seed"],
                "policy_sampling_seed": forged_metadata["policy_sampling_seed"],
                "scenario": forged_metadata["scenario"],
                "branch_tick": forged_metadata["branch_tick"],
                "horizon_ticks": forged_metadata["horizon_ticks"],
                "exact_replay_repeat_count_per_action": forged_continuation[
                    "exact_replay_repeat_count_per_action"
                ],
                "focal_agent_id": forged_metadata["focal_agent_id"],
                "source_record_digest": forged_metadata["source_record_digest"],
                "source_recurrent_state_sha256": forged_optimizer[
                    "source_recurrent_state_sha256"
                ],
                "source_sampling_state_sha256": forged_metadata[
                    "source_sampling_state_sha256"
                ],
                "source_public_history_prefix_sha256": forged_metadata[
                    "source_public_history_prefix_sha256"
                ],
                "continuation_provenance_digest": forged_metadata[
                    "continuation_provenance_digest"
                ],
            }
        )
        forged_state_row["component_digests"]["optimizer_context"] = (
            stable_payload_digest(forged_optimizer)
        )
        forged_state_row["component_digests"]["metadata"] = stable_payload_digest(
            forged_metadata
        )
        forged_state_row.pop("exact_digest")
        forged_state_row["exact_digest"] = stable_payload_digest(forged_state_row)
        validate_recurrent_counterfactual_branch_row(forged_state_row)
        with self.assertRaisesRegex(
            RecurrentCounterfactualBranchError,
            "does not exactly match its public-history reconstruction",
        ):
            verified_source_recurrent_state_from_branch_row(
                self.source_model,
                forged_state_row,
                artifact_digest="a" * 64,
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

    def test_stratified_branch_selection_draws_tick_before_focal_agent(self) -> None:
        selection_seed = next(
            seed for seed in range(100) if random.Random(seed).randrange(2) == 1
        )
        materialized = build_recurrent_counterfactual_nested_horizon_materialization(
            self.source_model,
            artifact_digest="a" * 64,
            seed_role="curriculum",
            environment_seed=RECURRENT_SEED_REGISTRY["curriculum"][0],
            scenario="carrion_only",
            branch_tick_candidates=(0, 1),
            horizons=(1, 2),
            source_policy_sampling_seed=991,
            branch_selection_seed=selection_seed,
            gamma=0.99,
            continuation_tape_count=1,
            continuation_tape_identity="branch-tests:stratified-selection",
        )

        selection = materialized["selection"]
        self.assertEqual(selection["eligible_branch_ticks"], [0, 1])
        self.assertEqual(selection["eligible_branch_tick_count"], 2)
        self.assertEqual(selection["selected_eligible_tick_index"], 1)
        self.assertEqual(selection["selected_branch_tick"], 1)
        self.assertEqual(
            selection["selection_policy"],
            "stratified_uniform_seeded_tick_then_uniform_current_tick_learner_decision",
        )

    def test_explicit_branch_tick_strata_rotate_without_first_eligible_bias(
        self,
    ) -> None:
        selections = []
        for stratum_index in range(2):
            materialized = (
                build_recurrent_counterfactual_nested_horizon_materialization(
                    self.source_model,
                    artifact_digest="a" * 64,
                    seed_role="curriculum",
                    environment_seed=RECURRENT_SEED_REGISTRY["curriculum"][0],
                    scenario="carrion_only",
                    branch_tick_candidates=(0, 1),
                    horizons=(1, 2),
                    source_policy_sampling_seed=991,
                    branch_selection_seed=123_456,
                    branch_tick_stratum_index=stratum_index,
                    continuation_tape_identity=(
                        f"branch-tests:rotation:{stratum_index}"
                    ),
                )
            )
            selections.append(materialized["selection"])

        self.assertEqual(
            [selection["selected_branch_tick"] for selection in selections],
            [0, 1],
        )
        for stratum_index, selection in enumerate(selections):
            self.assertEqual(
                selection["requested_branch_tick_stratum_index"],
                stratum_index,
            )
            self.assertEqual(
                selection["branch_tick_strata_traversal"],
                [stratum_index, 1 - stratum_index],
            )
            self.assertEqual(
                selection["selection_policy"],
                "deterministic_stratified_tick_rotation_with_eligible_fallback_"
                "then_uniform_current_tick_learner_decision",
            )

    def test_scale_training_seed_roles_are_explicit_and_registry_bound(self) -> None:
        for role, scenario in (
            ("scale_train", "broad"),
            ("scale_curriculum", "carrion_only"),
        ):
            row = build_recurrent_counterfactual_branch_row(
                self.source_model,
                artifact_digest="a" * 64,
                seed_role=role,
                environment_seed=SCALE_DEVELOPMENT_SEED_REGISTRY[role][0],
                scenario=scenario,
                branch_tick=0,
                horizon_ticks=1,
                policy_sampling_seed=991,
            )
            validate_recurrent_counterfactual_branch_row(row)
            self.assertEqual(
                row["contract"]["allowed_seed_roles"],
                ["scale_train", "scale_curriculum"],
            )

    def test_pre_tick_retape_can_kill_focal_but_boundary_retape_preserves_turn(
        self,
    ) -> None:
        identity = "pretape-death:0"
        environment_tape_seed = derive_recurrent_counterfactual_tape_seed(
            namespace=(
                RECURRENT_COUNTERFACTUAL_CONTINUATION_ENVIRONMENT_TAPE_SEED_NAMESPACE
            ),
            identity=f"{identity}:environment:0",
        )
        policy_tape_seed = derive_recurrent_counterfactual_tape_seed(
            namespace=(
                RECURRENT_COUNTERFACTUAL_CONTINUATION_POLICY_TAPE_SEED_NAMESPACE
            ),
            identity=f"{identity}:policy:0",
        )
        model = self._decision_boundary_fixture_model()
        pre_tick_policy = DeterministicPublicRecurrentPolicy(
            model,
            artifact_digest="a" * 64,
            sampling_seed=policy_tape_seed,
            capture_public_history=True,
        )
        pre_tick_world = self._decision_boundary_fixture_world(
            scenario="carrion_only",
            environment_seed=887_847_623,
            ticks=1,
            policy=pre_tick_policy,
        )
        branch_module._configure_manual_world(pre_tick_world)
        pre_tick_world.rng.seed(environment_tape_seed)
        pre_tick_world.tick = 0
        pre_tick_world._run_tick()
        records_by_agent = {
            record["agent_id"]: record
            for record in pre_tick_world.tick_trajectory_records
        }
        self.assertEqual(records_by_agent[1]["requested_action"], "attack_north")
        self.assertEqual(records_by_agent[2]["action_source"], "passive")
        self.assertTrue(records_by_agent[2]["outcome"]["passive"]["killed"])
        self.assertTrue(
            records_by_agent[2]["outcome"]["passive"]["died_before_action"]
        )

        materialized = self._materialize_decision_boundary_fixture(tape_count=2)

        self.assertEqual(materialized["selection"]["selected_focal_agent_id"], 2)
        self.assertEqual(
            materialized["rows"][0]["labels"]["source_requested_action"],
            "stay",
        )
        for row in (
            *materialized["aggregate_rows"],
            materialized["terminal_target"],
        ):
            validate_recurrent_counterfactual_aggregate_row(row)
            self.assertEqual(
                row["tape_contract"]["continuation_rng_retape_boundary"],
                RECURRENT_COUNTERFACTUAL_CONTINUATION_RNG_RETAPE_BOUNDARY,
            )
            self.assertTrue(
                row["tape_contract"]["fixed_source_prefix_through_focal_natural_draw"]
            )
            self.assertTrue(row["tape_contract"]["horizon_includes_branch_tick"])
            self.assertTrue(
                row["tape_contract"][
                    "same_tick_post_focal_consequences_governed_by_tape"
                ]
            )
            for tape in row["tape_provenance"]:
                self.assertTrue(tape["boundary_reached"])
                self.assertTrue(tape["source_natural_action_match"])
                self.assertTrue(tape["fixed_source_prefix_verified"])
                self.assertEqual(
                    tape["baseline"]["natural_requested_action"],
                    row["source_behavior"]["source_requested_action"],
                )
                for outcome in tape["action_outcomes"]:
                    self.assertEqual(
                        outcome["natural_requested_action"],
                        row["source_behavior"]["source_requested_action"],
                    )

    def test_boundary_rng_provenance_is_paired_within_and_distinct_across_tapes(
        self,
    ) -> None:
        materialized = self._materialize_decision_boundary_fixture(tape_count=3)
        row = materialized["aggregate_rows"][-1]
        validate_recurrent_counterfactual_aggregate_row(row)
        tapes = row["tape_provenance"]
        provenance_fields = (
            "pre_boundary_environment_rng_state_sha256",
            "pre_boundary_policy_sampling_state_sha256",
            "post_boundary_environment_rng_state_sha256",
            "post_boundary_policy_sampling_state_sha256",
        )

        self.assertEqual(
            len(
                {
                    tape["pre_boundary_environment_rng_state_sha256"]
                    for tape in tapes
                }
            ),
            1,
        )
        self.assertEqual(
            len(
                {
                    tape["pre_boundary_policy_sampling_state_sha256"]
                    for tape in tapes
                }
            ),
            1,
        )
        self.assertEqual(
            len(
                {
                    tape["post_boundary_environment_rng_state_sha256"]
                    for tape in tapes
                }
            ),
            3,
        )
        self.assertEqual(
            len(
                {
                    tape["post_boundary_policy_sampling_state_sha256"]
                    for tape in tapes
                }
            ),
            3,
        )
        for tape in tapes:
            for outcome in (tape["baseline"], *tape["action_outcomes"]):
                for field in provenance_fields:
                    self.assertEqual(outcome[field], tape[field])

    def test_rehashed_boundary_provenance_tampering_fails_closed(self) -> None:
        materialized = self._materialize_decision_boundary_fixture(tape_count=2)

        def rehash(row: dict[str, object], *, component: str | None = None) -> None:
            if component is not None:
                row["component_digests"][component] = stable_payload_digest(
                    row[component]
                )
            row.pop("exact_digest", None)
            row["exact_digest"] = stable_payload_digest(row)

        coherently_tampered_provenance = deepcopy(
            materialized["aggregate_rows"][0]
        )
        first_tape = coherently_tampered_provenance["tape_provenance"][0]
        first_tape["post_boundary_environment_rng_state_sha256"] = "0" * 64
        for outcome in (
            first_tape["baseline"],
            *first_tape["action_outcomes"],
        ):
            outcome["post_boundary_environment_rng_state_sha256"] = "0" * 64
        coherently_tampered_provenance["component_digests"]["tape_provenance"] = (
            stable_payload_digest(coherently_tampered_provenance["tape_provenance"])
        )
        coherently_tampered_provenance.pop("exact_digest")
        coherently_tampered_provenance["exact_digest"] = stable_payload_digest(
            coherently_tampered_provenance
        )
        with self.assertRaises(RecurrentCounterfactualBranchError):
            validate_recurrent_counterfactual_aggregate_row(
                coherently_tampered_provenance
            )

        tampered_provenance = deepcopy(materialized["aggregate_rows"][0])
        tampered_provenance["tape_provenance"][0][
            "post_boundary_environment_rng_state_sha256"
        ] = "0" * 64
        tampered_provenance["component_digests"]["tape_provenance"] = (
            stable_payload_digest(tampered_provenance["tape_provenance"])
        )
        tampered_provenance.pop("exact_digest")
        tampered_provenance["exact_digest"] = stable_payload_digest(
            tampered_provenance
        )
        with self.assertRaises(RecurrentCounterfactualBranchError):
            validate_recurrent_counterfactual_aggregate_row(tampered_provenance)

        tampered_contract = deepcopy(materialized["aggregate_rows"][0])
        tampered_contract["tape_contract"]["continuation_rng_retape_boundary"] = (
            "before_materialized_branch_tick"
        )
        tampered_contract["component_digests"]["tape_contract"] = (
            stable_payload_digest(tampered_contract["tape_contract"])
        )
        tampered_contract.pop("exact_digest")
        tampered_contract["exact_digest"] = stable_payload_digest(tampered_contract)
        with self.assertRaises(RecurrentCounterfactualBranchError):
            validate_recurrent_counterfactual_aggregate_row(tampered_contract)

        tampered_natural_action = deepcopy(materialized["aggregate_rows"][0])
        source_action = tampered_natural_action["source_behavior"][
            "source_requested_action"
        ]
        replacement_action = next(
            action
            for action, available in tampered_natural_action["source_behavior"][
                "current_public_action_mask"
            ].items()
            if available and action != source_action
        )
        for tape in tampered_natural_action["tape_provenance"]:
            tape["baseline"]["natural_requested_action"] = replacement_action
            for outcome in tape["action_outcomes"]:
                outcome["natural_requested_action"] = replacement_action
        tampered_natural_action["component_digests"]["tape_provenance"] = (
            stable_payload_digest(tampered_natural_action["tape_provenance"])
        )
        tampered_natural_action.pop("exact_digest")
        tampered_natural_action["exact_digest"] = stable_payload_digest(
            tampered_natural_action
        )
        with self.assertRaises(RecurrentCounterfactualBranchError):
            validate_recurrent_counterfactual_aggregate_row(tampered_natural_action)

        injected_private_checkpoint = deepcopy(materialized["aggregate_rows"][0])
        injected_private_checkpoint["tape_provenance"][0][
            "private_world_checkpoint"
        ] = {"forbidden": True}
        rehash(injected_private_checkpoint, component="tape_provenance")
        with self.assertRaises(RecurrentCounterfactualBranchError):
            validate_recurrent_counterfactual_aggregate_row(
                injected_private_checkpoint
            )

        injected_training_flag = deepcopy(materialized["aggregate_rows"][0])
        injected_training_flag["training_artifact_created"] = True
        rehash(injected_training_flag)
        with self.assertRaises(RecurrentCounterfactualBranchError):
            validate_recurrent_counterfactual_aggregate_row(injected_training_flag)

        tampered_baseline_transition = deepcopy(materialized["aggregate_rows"][0])
        baseline = tampered_baseline_transition["tape_provenance"][0]["baseline"]
        baseline["first_transition"]["requested_action"] = replacement_action
        rehash(tampered_baseline_transition, component="tape_provenance")
        with self.assertRaises(RecurrentCounterfactualBranchError):
            validate_recurrent_counterfactual_aggregate_row(
                tampered_baseline_transition
            )

        tampered_source_action_control = deepcopy(materialized["aggregate_rows"][0])
        source_outcome = next(
            outcome
            for outcome in tampered_source_action_control["tape_provenance"][0][
                "action_outcomes"
            ]
            if outcome["action"] == source_action
        )
        source_outcome["behavior_digest"] = "0" * 64
        rehash(tampered_source_action_control, component="tape_provenance")
        with self.assertRaises(RecurrentCounterfactualBranchError):
            validate_recurrent_counterfactual_aggregate_row(
                tampered_source_action_control
            )

        coherent_source_action_control = deepcopy(
            materialized["aggregate_rows"][0]
        )
        for tape in coherent_source_action_control["tape_provenance"]:
            source_outcome = next(
                outcome
                for outcome in tape["action_outcomes"]
                if outcome["action"] == source_action
            )
            source_outcome["population_alive"] += 1
            source_outcome["paired_vs_baseline"]["population_alive_delta"] += 1
        coherent_source_action_control["aggregate"] = (
            branch_module._aggregate_multi_tape_outcomes(
                coherent_source_action_control["tape_provenance"],
                uncertainty_penalty=coherent_source_action_control["aggregate"][
                    "uncertainty_penalty"
                ],
            )
        )
        rehash(coherent_source_action_control, component="tape_provenance")
        coherent_source_action_control["component_digests"]["aggregate"] = (
            stable_payload_digest(coherent_source_action_control["aggregate"])
        )
        coherent_source_action_control.pop("exact_digest")
        coherent_source_action_control["exact_digest"] = stable_payload_digest(
            coherent_source_action_control
        )
        with self.assertRaisesRegex(
            RecurrentCounterfactualBranchError,
            "negative control",
        ):
            validate_recurrent_counterfactual_aggregate_row(
                coherent_source_action_control
            )

        tampered_excluded_seeds = deepcopy(materialized["aggregate_rows"][0])
        tampered_excluded_seeds["tape_contract"]["excluded_source_seeds"] = [
            7,
            8,
            9,
        ]
        rehash(tampered_excluded_seeds, component="tape_contract")
        with self.assertRaises(RecurrentCounterfactualBranchError):
            validate_recurrent_counterfactual_aggregate_row(tampered_excluded_seeds)

        fractional_count = deepcopy(materialized["aggregate_rows"][0])
        fractional_count["tape_provenance"][0]["baseline"][
            "population_alive"
        ] = 1.5
        fractional_count["aggregate"] = (
            branch_module._aggregate_multi_tape_outcomes(
                fractional_count["tape_provenance"],
                uncertainty_penalty=fractional_count["aggregate"][
                    "uncertainty_penalty"
                ],
            )
        )
        fractional_count["component_digests"]["tape_provenance"] = (
            stable_payload_digest(fractional_count["tape_provenance"])
        )
        fractional_count["component_digests"]["aggregate"] = stable_payload_digest(
            fractional_count["aggregate"]
        )
        fractional_count.pop("exact_digest")
        fractional_count["exact_digest"] = stable_payload_digest(fractional_count)
        with self.assertRaisesRegex(
            RecurrentCounterfactualBranchError,
            "non-negative integer",
        ):
            validate_recurrent_counterfactual_aggregate_row(fractional_count)

        malformed_transition = deepcopy(materialized["aggregate_rows"][0])
        malformed_transition["tape_provenance"][0]["baseline"][
            "first_transition"
        ]["moved"] = 1
        rehash(malformed_transition, component="tape_provenance")
        with self.assertRaisesRegex(
            RecurrentCounterfactualBranchError,
            "exact boolean",
        ):
            validate_recurrent_counterfactual_aggregate_row(malformed_transition)

        for name, mutate, expected_error in (
            (
                "aggregate reproduction without birth",
                lambda outcome: (
                    outcome["first_transition"].__setitem__("reproduced", True),
                    outcome.__setitem__("births_during_horizon", 0),
                ),
                "requires at least one horizon birth",
            ),
            (
                "aggregate death without death count",
                lambda outcome: (
                    outcome["first_transition"].__setitem__("died", True),
                    outcome["first_transition"].__setitem__("after_alive", False),
                    outcome.__setitem__("focal_terminal_alive", False),
                    outcome.__setitem__("deaths_during_horizon", 0),
                ),
                "requires at least one horizon death",
            ),
            (
                "aggregate terminal death without death count",
                lambda outcome: (
                    outcome["first_transition"].__setitem__("died", False),
                    outcome["first_transition"].__setitem__("after_alive", True),
                    outcome["first_transition"].__setitem__(
                        "after_energy_ratio", 1.0
                    ),
                    outcome["first_transition"].__setitem__(
                        "after_hydration_ratio", 1.0
                    ),
                    outcome["first_transition"].__setitem__(
                        "after_health_ratio", 1.0
                    ),
                    outcome.__setitem__("focal_terminal_alive", False),
                    outcome.__setitem__("deaths_during_horizon", 0),
                ),
                "terminal-dead focal outcome requires at least one horizon death",
            ),
            (
                "aggregate discounted return outside feasible bounds",
                lambda outcome: outcome.__setitem__(
                    "focal_discounted_return",
                    1000.0,
                ),
                "outside feasible reward bounds",
            ),
        ):
            with self.subTest(name=name):
                causal_tampered = deepcopy(materialized["aggregate_rows"][0])
                mutate(causal_tampered["tape_provenance"][0]["baseline"])
                rehash(causal_tampered, component="tape_provenance")
                with self.assertRaisesRegex(
                    RecurrentCounterfactualBranchError,
                    expected_error,
                ):
                    validate_recurrent_counterfactual_aggregate_row(
                        causal_tampered
                    )

        empty_history_nonzero_feedback = deepcopy(
            materialized["aggregate_rows"][0]
        )
        prefix = empty_history_nonzero_feedback["trainable_public_context"][
            "public_history_prefix"
        ]
        prefix["record_count"] = 0
        prefix["records"] = []
        feedback_values = empty_history_nonzero_feedback[
            "trainable_public_context"
        ]["previous_public_feedback"]["values"]
        feedback_values[:] = [0.0] * len(feedback_values)
        feedback_values[0] = 1.0
        feedback_values[len(ACTION_NAMES)] = 1.0
        feedback_values[-3] = 1.0
        rehash(
            empty_history_nonzero_feedback,
            component="trainable_public_context",
        )
        with self.assertRaisesRegex(
            RecurrentCounterfactualBranchError,
            "all zero when the public history is empty",
        ):
            validate_recurrent_counterfactual_aggregate_row(
                empty_history_nonzero_feedback
            )

        flattened_aggregate_state = deepcopy(materialized["aggregate_rows"][0])
        aggregate_state = flattened_aggregate_state["optimizer_context"][
            "source_recurrent_state"
        ]
        flattened_aggregate_state["optimizer_context"][
            "source_recurrent_state"
        ] = [
            item
            for layer in aggregate_state
            for batch in layer
            for item in batch
        ]
        rehash(flattened_aggregate_state, component="optimizer_context")
        with self.assertRaisesRegex(
            RecurrentCounterfactualBranchError,
            "declared shape",
        ):
            validate_recurrent_counterfactual_aggregate_row(
                flattened_aggregate_state
            )

        for name, mutate in (
            (
                "bool baseline source count",
                lambda row: row["tape_provenance"][0]["baseline"].__setitem__(
                    "unsupported_requested_action_count", False
                ),
            ),
            (
                "bool tape index",
                lambda row: row["tape_provenance"][0].__setitem__(
                    "tape_index", False
                ),
            ),
        ):
            with self.subTest(name=name):
                tampered_integer = deepcopy(materialized["aggregate_rows"][0])
                mutate(tampered_integer)
                rehash(tampered_integer, component="tape_provenance")
                with self.assertRaisesRegex(
                    RecurrentCounterfactualBranchError,
                    "non-negative integer",
                ):
                    validate_recurrent_counterfactual_aggregate_row(
                        tampered_integer
                    )

        stale_schema = deepcopy(materialized["aggregate_rows"][0])
        stale_schema["schema_version"] = (
            "mind_v3_recurrent_counterfactual_multi_tape_aggregate_v1"
        )
        stale_schema.pop("exact_digest")
        stale_schema["exact_digest"] = stable_payload_digest(stale_schema)
        with self.assertRaises(RecurrentCounterfactualBranchError):
            validate_recurrent_counterfactual_aggregate_row(stale_schema)

        beyond_terminal = deepcopy(materialized["aggregate_rows"][-1])
        beyond_terminal["target"]["absolute_terminal_target_world_tick"] = (
            beyond_terminal["target"]["target_world_tick"] - 1
        )
        rehash(beyond_terminal, component="target")
        with self.assertRaisesRegex(
            RecurrentCounterfactualBranchError,
            "cannot extend beyond",
        ):
            validate_recurrent_counterfactual_aggregate_row(beyond_terminal)

    def test_decision_boundary_retape_requires_exact_source_decision_projection(
        self,
    ) -> None:
        original_projection = (
            branch_module._source_decision_prefix_projection_sha256
        )
        projection_calls = 0

        def drift_after_source(diagnostics: object) -> str:
            nonlocal projection_calls
            projection_calls += 1
            if projection_calls == 1:
                return original_projection(diagnostics)
            return "0" * 64

        with patch.object(
            branch_module,
            "_source_decision_prefix_projection_sha256",
            side_effect=drift_after_source,
        ), self.assertRaisesRegex(
            RecurrentCounterfactualBranchError,
            "decision projection drifted",
        ):
            self._materialize_decision_boundary_fixture(tape_count=1)

    def test_multi_tape_aggregate_is_paired_replay_verified_and_terminal_aware(
        self,
    ) -> None:
        identity = "branch-tests:independent-continuation-tapes"
        materialized = build_recurrent_counterfactual_nested_horizon_materialization(
            self.source_model,
            artifact_digest="a" * 64,
            seed_role="curriculum",
            environment_seed=RECURRENT_SEED_REGISTRY["curriculum"][0],
            scenario="carrion_only",
            branch_tick_candidates=(0,),
            horizons=(1, 2),
            source_policy_sampling_seed=991,
            branch_selection_seed=123_456,
            gamma=0.99,
            continuation_tape_count=3,
            continuation_tape_identity=identity,
            terminal_target_world_tick=3,
            uncertainty_penalty=0.5,
        )

        self.assertEqual(len(materialized["aggregate_rows"]), 2)
        self.assertEqual(
            materialized["terminal_target"]["target"]["target_world_tick"],
            3,
        )
        self.assertTrue(
            materialized["terminal_target"]["target"]["is_absolute_terminal_target"]
        )
        for aggregate_row in (
            *materialized["aggregate_rows"],
            materialized["terminal_target"],
        ):
            validate_recurrent_counterfactual_aggregate_row(aggregate_row)
            self.assertEqual(
                aggregate_row["trainable_public_context"],
                materialized["rows"][0]["trainable_public_context"],
            )
            self.assertEqual(
                aggregate_row["source_identity"]["source_model_state_sha256"],
                recurrent_model_state_sha256(self.source_model),
            )
            self.assertEqual(
                aggregate_row["source_identity"]["source_artifact_digest"],
                "a" * 64,
            )
            self.assertEqual(aggregate_row["tape_contract"]["tape_count"], 3)
            tapes = aggregate_row["tape_provenance"]
            self.assertEqual([tape["tape_index"] for tape in tapes], [0, 1, 2])
            self.assertEqual(
                len({tape["environment_sampling_seed"] for tape in tapes}),
                3,
            )
            self.assertEqual(
                len({tape["policy_sampling_seed"] for tape in tapes}),
                3,
            )
            for tape in tapes:
                index = tape["tape_index"]
                self.assertEqual(
                    tape["environment_sampling_seed"],
                    derive_recurrent_counterfactual_tape_seed(
                        namespace=(
                            RECURRENT_COUNTERFACTUAL_CONTINUATION_ENVIRONMENT_TAPE_SEED_NAMESPACE
                        ),
                        identity=f"{identity}:environment:{index}",
                    ),
                )
                self.assertEqual(
                    tape["policy_sampling_seed"],
                    derive_recurrent_counterfactual_tape_seed(
                        namespace=(
                            RECURRENT_COUNTERFACTUAL_CONTINUATION_POLICY_TAPE_SEED_NAMESPACE
                        ),
                        identity=f"{identity}:policy:{index}",
                    ),
                )
                self.assertTrue(tape["baseline"]["replay_verified"])
                self.assertEqual(
                    tape["baseline"]["evidence_digest"],
                    tape["baseline"]["replay_evidence_digest"],
                )
                for outcome in tape["action_outcomes"]:
                    self.assertTrue(outcome["replay_verified"])
                    self.assertEqual(
                        outcome["evidence_digest"],
                        outcome["replay_evidence_digest"],
                    )
                    for field in (
                        "pre_boundary_environment_rng_state_sha256",
                        "pre_boundary_policy_sampling_state_sha256",
                        "post_boundary_environment_rng_state_sha256",
                        "post_boundary_policy_sampling_state_sha256",
                    ):
                        self.assertEqual(outcome[field], tape[field])

            first_action = aggregate_row["aggregate"]["action_outcomes"][0]
            action = first_action["action"]
            raw_deltas = [
                next(
                    outcome
                    for outcome in tape["action_outcomes"]
                    if outcome["action"] == action
                )["paired_vs_baseline"]["focal_discounted_return_delta"]
                for tape in tapes
            ]
            expected_mean = statistics.fmean(raw_deltas)
            expected_variance = statistics.variance(raw_deltas)
            expected_se = (expected_variance / len(raw_deltas)) ** 0.5
            stats = first_action["paired_delta_statistics"][
                "focal_discounted_return_delta"
            ]
            self.assertAlmostEqual(stats["mean"], expected_mean, places=12)
            self.assertAlmostEqual(
                stats["sample_variance"],
                expected_variance,
                places=12,
            )
            self.assertAlmostEqual(stats["standard_error"], expected_se, places=12)
            self.assertAlmostEqual(
                first_action["uncertainty_penalized_score"],
                round(stats["mean"] - 0.5 * stats["standard_error"], 12),
                places=12,
            )

    def test_multi_tape_aggregate_tampering_fails_closed(self) -> None:
        materialized = build_recurrent_counterfactual_nested_horizon_materialization(
            self.source_model,
            artifact_digest="a" * 64,
            seed_role="curriculum",
            environment_seed=RECURRENT_SEED_REGISTRY["curriculum"][0],
            scenario="carrion_only",
            branch_tick_candidates=(0,),
            horizons=(1, 2),
            source_policy_sampling_seed=991,
            branch_selection_seed=123_456,
            continuation_tape_count=2,
            continuation_tape_identity="branch-tests:tamper",
            terminal_target_world_tick=3,
            uncertainty_penalty=0.25,
        )
        tampered = deepcopy(materialized["aggregate_rows"][0])
        tampered["aggregate"]["action_outcomes"][0]["paired_delta_statistics"][
            "focal_discounted_return_delta"
        ]["mean"] += 1.0

        with self.assertRaisesRegex(
            RecurrentCounterfactualBranchError,
            "aggregate statistics|component digest|exact digest",
        ):
            validate_recurrent_counterfactual_aggregate_row(tampered)

    def test_coherently_rehashed_aggregate_numeric_type_aliases_fail_closed(
        self,
    ) -> None:
        materialized = self._materialize_decision_boundary_fixture(tape_count=1)

        def rehash(row: dict[str, object], *, component: str) -> None:
            row["component_digests"][component] = stable_payload_digest(
                row[component]
            )
            row.pop("exact_digest", None)
            row["exact_digest"] = stable_payload_digest(row)

        for name, mutate, component, expected_error in (
            (
                "aggregate tape count bool",
                lambda row: row["aggregate"].__setitem__("tape_count", True),
                "aggregate",
                "positive integer",
            ),
            (
                "aggregate uncertainty bool",
                lambda row: row["aggregate"].__setitem__(
                    "uncertainty_penalty", False
                ),
                "aggregate",
                "must be numeric",
            ),
            (
                "aggregate statistic bool",
                lambda row: row["aggregate"]["baseline_outcome_statistics"][
                    "focal_discounted_return"
                ].__setitem__("sample_variance", False),
                "aggregate",
                "must be numeric",
            ),
            (
                "aggregate action tape count float",
                lambda row: row["aggregate"]["action_outcomes"][0].__setitem__(
                    "tape_count", 1.0
                ),
                "aggregate",
                "positive integer",
            ),
            (
                "tape paired delta bool",
                lambda row: next(
                    outcome
                    for outcome in row["tape_provenance"][0]["action_outcomes"]
                    if outcome["action"]
                    == row["source_behavior"]["source_requested_action"]
                )["paired_vs_baseline"].__setitem__(
                    "population_alive_delta", False
                ),
                "tape_provenance",
                "exact integer",
            ),
            (
                "aggregate target world tick bool",
                lambda row: row["target"].__setitem__("target_world_tick", True),
                "target",
                "positive integer",
            ),
        ):
            with self.subTest(name=name):
                tampered = deepcopy(materialized["aggregate_rows"][0])
                mutate(tampered)
                rehash(tampered, component=component)
                with self.assertRaisesRegex(
                    RecurrentCounterfactualBranchError,
                    expected_error,
                ):
                    validate_recurrent_counterfactual_aggregate_row(tampered)

        empty_population = deepcopy(materialized["aggregate_rows"][0])
        tape = empty_population["tape_provenance"][0]
        baseline = tape["baseline"]
        living_outcome = next(
            outcome
            for outcome in tape["action_outcomes"]
            if outcome["action"]
            != empty_population["source_behavior"]["source_requested_action"]
            and outcome["focal_terminal_alive"] is True
        )
        living_outcome["population_alive"] = 0
        living_outcome["paired_vs_baseline"]["population_alive_delta"] = -int(
            baseline["population_alive"]
        )
        empty_population["aggregate"] = branch_module._aggregate_multi_tape_outcomes(
            empty_population["tape_provenance"],
            uncertainty_penalty=empty_population["aggregate"][
                "uncertainty_penalty"
            ],
        )
        empty_population["component_digests"]["tape_provenance"] = (
            stable_payload_digest(empty_population["tape_provenance"])
        )
        rehash(empty_population, component="aggregate")
        with self.assertRaisesRegex(
            RecurrentCounterfactualBranchError,
            "living focal agent in an empty population",
        ):
            validate_recurrent_counterfactual_aggregate_row(empty_population)

        revived = deepcopy(materialized["aggregate_rows"][0])
        revived_outcome = next(
            outcome
            for outcome in revived["tape_provenance"][0]["action_outcomes"]
            if outcome["action"]
            != revived["source_behavior"]["source_requested_action"]
            and outcome["focal_terminal_alive"] is True
        )
        revived_outcome["first_transition"]["died"] = True
        revived_outcome["first_transition"]["after_alive"] = False
        rehash(revived, component="tape_provenance")
        with self.assertRaisesRegex(
            RecurrentCounterfactualBranchError,
            "cannot revive",
        ):
            validate_recurrent_counterfactual_aggregate_row(revived)

    def test_relative_horizons_cannot_overrun_absolute_terminal_target(self) -> None:
        with self.assertRaisesRegex(
            RecurrentCounterfactualBranchError,
            "must not run beyond the absolute terminal target",
        ):
            build_recurrent_counterfactual_nested_horizon_materialization(
                self.source_model,
                artifact_digest="a" * 64,
                seed_role="curriculum",
                environment_seed=RECURRENT_SEED_REGISTRY["curriculum"][0],
                scenario="carrion_only",
                branch_tick_candidates=(88,),
                horizons=(16, 48),
                source_policy_sampling_seed=991,
                branch_selection_seed=123_456,
                continuation_tape_count=4,
                continuation_tape_identity="branch-tests:terminal-overrun",
                terminal_target_world_tick=120,
            )


if __name__ == "__main__":
    unittest.main()
